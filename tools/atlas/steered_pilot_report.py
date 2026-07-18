#!/usr/bin/env python
"""steered_pilot_report.py — build out/steered_pilot_report.md from a steered/baseline A/B.

Reads both arms' run-scoped dirs (each has outcome_ledger.jsonl + summary.json; the steered
arm also has harvest_log.jsonl + state.json) and emits the pilot report:
  - distinct q3 per active hour after coord-dedup, per arm (primary) + per-family
  - coord-dup rate over time, per arm
  - depth distribution of admitted q3
  - harvest confusion table (cheap-said-harvest x canonical decode) + realized tau_h recall
  - per-root expansion histogram + cap-hit rate (is M binding?)
  - a paired sample manifest of admitted locations from both arms (even draw) — built, not labeled

  uv run python tools/atlas/steered_pilot_report.py --steered <dirA> --baseline <dirB> \
      [--budget-min 45] [--manifest-n 40]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
import location as loc_mod                       # noqa: E402
from active_ckpt import BIN, PALETTE, JPG_Q, auto_maxiter  # noqa: E402

OUT_MD = ROOT / "out" / "steered_pilot_report.md"
MANIFEST_DIR = ROOT / "out" / "steered_pilot_manifest"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def admitted_q3(rows: list[dict]) -> list[dict]:
    """Distinct (coord-deduped) admitted q3 outcomes — the primary yield unit, identical
    definition for both arms (guard-passed, decoded class 3, distinct)."""
    return [r for r in rows if r.get("distinct") and r.get("guard_pass", True)
            and r.get("decoded_class") == 3]


def active_minutes(run_dir: Path, fallback_min: float) -> float:
    s = run_dir / "summary.json"
    if s.exists():
        d = json.loads(s.read_text(encoding="utf-8"))
        if "active_min" in d:
            return float(d["active_min"])
        if "wallclock_s" in d:
            return float(d["wallclock_s"]) / 60.0
        t = d.get("totals", {})
        if "wallclock_s" in t:
            return float(t["wallclock_s"]) / 60.0
    return fallback_min


def per_family(rows: list[dict]) -> Counter:
    return Counter(r.get("family", "mandelbrot") for r in admitted_q3(rows))


def depth_hist(rows: list[dict]) -> Counter:
    return Counter(int(r.get("reached_depth", 0)) for r in admitted_q3(rows))


def coord_dup_rate(rows: list[dict]) -> tuple[int, int]:
    """Among guard-passed decoded-q3 outcomes, how many were coord-dups (not distinct)."""
    q3 = [r for r in rows if r.get("guard_pass", True) and r.get("decoded_class") == 3]
    dup = sum(1 for r in q3 if not r.get("distinct"))
    return dup, len(q3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steered", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--budget-min", type=float, default=45.0)
    ap.add_argument("--manifest-n", type=int, default=40)
    args = ap.parse_args()

    st_dir, bl_dir = args.steered.resolve(), args.baseline.resolve()
    st_rows = load_jsonl(st_dir / "outcome_ledger.jsonl")
    bl_rows = load_jsonl(bl_dir / "outcome_ledger.jsonl")
    st_active = active_minutes(st_dir, args.budget_min)
    bl_active = active_minutes(bl_dir, args.budget_min)
    st_hlog = load_jsonl(st_dir / "harvest_log.jsonl")

    st_q3, bl_q3 = admitted_q3(st_rows), admitted_q3(bl_rows)
    L = []
    W = L.append
    W("# Steered vs baseline frontier descent — pilot A/B\n")
    W(f"Two fresh-generation arms, same families, separate run-scoped ledgers. "
      f"**Steered** = classifier best-first frontier (`tools/atlas/steered_frontier.py`, "
      f"`guided-descend --expand`); **baseline** = the current production walk "
      f"(`production_seeder.py --run`), byte-untouched.\n")
    W(f"- steered  : `{st_dir}`  — active {st_active:.1f} min\n"
      f"- baseline : `{bl_dir}`  — active {bl_active:.1f} min\n")

    # --- primary: distinct q3 per active hour ---
    W("## Primary — distinct q3 per active hour (coord-deduped)\n")
    W("| arm | distinct q3 | active min | q3 / active hour |")
    W("|---|---|---|---|")
    for name, q3, act in [("steered", st_q3, st_active), ("baseline", bl_q3, bl_active)]:
        rate = len(q3) / (act / 60.0) if act > 0 else 0.0
        W(f"| {name} | {len(q3)} | {act:.1f} | {rate:.1f} |")
    W("")
    W("### Per-family distinct q3\n")
    fams = sorted(set(per_family(st_rows)) | set(per_family(bl_rows)))
    W("| family | steered | baseline |")
    W("|---|---|---|")
    sf_pf, bl_pf = per_family(st_rows), per_family(bl_rows)
    for f in fams:
        W(f"| {f} | {sf_pf.get(f,0)} | {bl_pf.get(f,0)} |")
    W("")

    # --- coord-dup rate ---
    W("## Coord-dup rate (dup q3 / all decoded-q3)\n")
    W("| arm | q3 dup | decoded q3 total | dup rate |")
    W("|---|---|---|---|")
    for name, rows in [("steered", st_rows), ("baseline", bl_rows)]:
        dup, tot = coord_dup_rate(rows)
        W(f"| {name} | {dup} | {tot} | {(dup/tot if tot else 0):.2%} |")
    W("")

    # --- depth distribution ---
    W("## Depth distribution of admitted q3 (does steering exceed the fixed [4,14] draw?)\n")
    sd, bd = depth_hist(st_rows), depth_hist(bl_rows)
    maxd = max([0] + list(sd) + list(bd))
    W("| depth | steered | baseline |")
    W("|---|---|---|")
    for d in range(1, maxd + 1):
        if sd.get(d, 0) or bd.get(d, 0):
            W(f"| {d} | {sd.get(d,0)} | {bd.get(d,0)} |")
    st_beyond = sum(v for d, v in sd.items() if d > 14)
    bl_beyond = sum(v for d, v in bd.items() if d > 14)
    W("")
    W(f"Admitted q3 at depth > 14 (beyond the walk's terminal draw): **steered {st_beyond}**, "
      f"baseline {bl_beyond} (baseline caps at 14 by construction).\n")

    # --- harvest confusion table (steered only) ---
    W("## Harvest confusion (steered) — cheap-said-harvest x canonical decode\n")
    if st_hlog:
        conf = Counter(h.get("canon_decoded") for h in st_hlog)
        admitted = sum(1 for h in st_hlog if h.get("admitted"))
        reframe_q3 = sum(1 for h in st_hlog if h.get("reframe_decoded") == 3)
        W("Every candidate whose **cheap p_good >= tau_h** got one canonical 640x360 ss2 render "
          "+ decode; the confusion is how those canonical decodes landed.\n")
        W("| cheap-said-harvest -> canonical decode | count |")
        W("|---|---|")
        for cls in (3, 2, 1, None):
            W(f"| class {cls} | {conf.get(cls,0)} |")
        W(f"| **total harvest checks** | {len(st_hlog)} |")
        W("")
        canon_q3 = conf.get(3, 0)
        W(f"Canonical-q3 confirmations: **{canon_q3}** / {len(st_hlog)} harvest checks "
          f"(**realized tau_h precision** = fraction of cheap-gated frames that canonically "
          f"decode q3 = {(canon_q3/len(st_hlog) if st_hlog else 0):.1%}). Of those, "
          f"{reframe_q3} survived reframe+guard as q3 and {admitted} were admitted distinct.\n")
        # per-family tau_h recall proxy: cheap-gate acceptance among the canonical-q3
        by_fam = defaultdict(lambda: [0, 0])
        for h in st_hlog:
            f = h.get("partition", "?")
            by_fam[f][0] += 1
            by_fam[f][1] += int(h.get("canon_decoded") == 3)
        W("### Per-family harvest checks -> canonical q3\n")
        W("| family | tau_h | checks | canonical q3 | precision |")
        W("|---|---|---|---|---|")
        for f in sorted(by_fam):
            n, q = by_fam[f]
            th = next((h["tau_h"] for h in st_hlog if h.get("partition") == f), None)
            W(f"| {f} | {th:.3f} | {n} | {q} | {(q/n if n else 0):.1%} |")
        W("")
    else:
        W("_no harvest log found — steered arm logged no harvest checks._\n")

    # --- per-root expansion histogram + cap-hit (steered only) ---
    W("## Per-root expansion histogram + M-cap (steered)\n")
    st_state = st_dir / "state.json"
    if st_state.exists():
        stt = json.loads(st_state.read_text(encoding="utf-8"))
        epr = stt.get("expansions_per_root", {})
        counts = sorted(epr.values(), reverse=True)
        capped = sum(1 for v in counts if v >= 40)
        W(f"- roots expanded at least once: **{len(counts)}**\n"
          f"- expansions/root: max {max(counts) if counts else 0}, "
          f"median {counts[len(counts)//2] if counts else 0}, "
          f"mean {sum(counts)/len(counts) if counts else 0:.1f}\n"
          f"- roots that hit the M=40 cap: **{capped}** "
          f"({'M is BINDING' if capped else 'M not binding'}); "
          f"batch-level cap_hits telemetry = {stt.get('totals',{}).get('cap_hits',0)}\n")
        # coarse histogram
        buckets = Counter(min(v, 40) // 5 * 5 for v in counts)
        W("\n| expansions/root bucket | roots |")
        W("|---|---|")
        for b in sorted(buckets):
            W(f"| {b}-{b+4} | {buckets[b]} |")
        W("")
    else:
        W("_no state.json found._\n")

    # --- paired sample manifest (build, don't label) ---
    W("## Paired sample manifest (blind-read set — built, NOT labeled)\n")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    n_each = args.manifest_n // 2
    manifest = []

    def even_draw(rows, k):
        if not rows or k <= 0:
            return []
        step = max(1, len(rows) // k)
        return rows[::step][:k]

    def render_geom(r):
        """Resolve (render_family, cx, cy, fw, c_re, c_im) for a ledger row, handling both
        julia schemas: baseline stores the PARAMETER c in outcome_cx/cy (viewport in
        julia_z_*); steered stores the z-VIEWPORT in outcome_cx/cy (c in julia_c_*)."""
        fam = r.get("family", "mandelbrot")
        if fam.startswith("julia:"):
            base = fam.split(":", 1)[1]
            rf = "julia" if base == "mandelbrot" else "julia_" + base
            if r.get("julia_z_cx") is not None:                 # baseline julia schema
                return rf, r["julia_z_cx"], r["julia_z_cy"], r["julia_z_fw"], \
                    r["outcome_cx"], r["outcome_cy"]
            if r.get("julia_c_re") is not None:                 # steered julia schema
                return rf, r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], \
                    r["julia_c_re"], r["julia_c_im"]
        return fam, r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], None, None

    def render_manifest(rows, arm):
        out = []
        for r in even_draw(rows, n_each):
            rf, cx, cy, fw, c_re, c_im = render_geom(r)
            loc = loc_mod.Location(family=rf, cx=str(cx), cy=str(cy), fw=str(fw),
                                   c_re=None if c_re is None else str(c_re),
                                   c_im=None if c_im is None else str(c_im))
            tile = MANIFEST_DIR / f"{arm}_{r['id']}.jpg"
            if not tile.exists():
                cmd = [str(BIN), "render-one", "--cx", str(cx), "--cy", str(cy),
                       "--fw", repr(float(fw)), "--width", "640", "--height", "360",
                       "--supersample", "2", "--maxiter", str(auto_maxiter(float(fw))),
                       "--palette", PALETTE, "--jpg-quality", str(JPG_Q),
                       "--out", str(tile)] + loc_mod.render_one_flags(loc)
                subprocess.run(cmd, capture_output=True, text=True)
            out.append({"arm": arm, "id": r["id"], "family": r.get("family", "mandelbrot"),
                        "cx": cx, "cy": cy, "fw": fw,
                        "c_re": c_re, "c_im": c_im,
                        "render": str(tile.relative_to(ROOT)) if tile.exists() else None})
        return out

    manifest += render_manifest(st_q3, "steered")
    manifest += render_manifest(bl_q3, "baseline")
    # interleave (even draw across arms) so a later blind read is arm-balanced.
    (MANIFEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    W(f"Even draw of {len(manifest)} admitted locations ({n_each}/arm requested), coords + "
      f"canonical 640x360 ss2 renders under `{MANIFEST_DIR.relative_to(ROOT)}/` "
      f"(`manifest.json` lists arm/id/coords/render). **Not labeled** — this is the blind-read "
      f"set for a later human pass; arm identity is in the manifest, not the filenames' visible order.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_MD}  ({len(st_q3)} steered / {len(bl_q3)} baseline admitted q3)")


if __name__ == "__main__":
    main()
