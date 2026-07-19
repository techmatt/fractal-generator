#!/usr/bin/env python
r"""campaign1_readout.py — price future scheduling off a steered-frontier campaign.

Regenerates `out/campaign1/readout.md` from the campaign's durable artifacts alone
(breadth + dive run dirs: outcome_ledger.jsonl, state.json/dive_state.json, harvest_log.jsonl,
and the run stdout log for the batch->active-time map). Re-runnable at any checkpoint; the
ledger is authoritative for admissions, state.json for accumulated active time.

Five numbers (campaign spec):
  1. Admissions/hr over ACCUMULATED ACTIVE time (does run-2's ~16/hr floor hold at scale?).
  2. Per-family admissions + cost/admission, breadth vs dive separately.
  3. Distinct morph-look count over time (library morph_gray recipe, within-family 0.974).
  4. Library overlap vs ALL prior ledgers: coord-dup fraction (cheap) + morph near-dup
     fraction (0.974 vs the durable library embedding store, if present).
  5. Families with ~zero admissions, flagged (watching multibrot4).

Reused wholesale — nothing reimplemented:
  production_seeder : is_distinct / build_cloud / REJECT_RADIUS / DEDUP_K / t_good_for / julia_partition
  steered_pilot_morph (spm) : admitted_q3 / embed_admissions / connected_components / load_clip / STRICT_CUT
  library_store : load_library_embeddings (prior library morph_clip store)

  uv run python tools/atlas/campaign1_readout.py \
      --breadth data/discovery/campaign1/breadth --dive data/discovery/campaign1/dive
  # cheap-only (skip the GPU morph pass) for a fast intermediate check:
  uv run python tools/atlas/campaign1_readout.py --breadth data/discovery/campaign1/breadth --no-morph
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "atlas", ROOT / "tools" / "corpus",
          ROOT / "tools" / "scoring", ROOT / "tools" / "studies", ROOT / "tools" / "wallpaper"):
    sys.path.insert(0, str(p))

import production_seeder as ps                      # noqa: E402
import steered_pilot_morph as spm                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STRICT = spm.STRICT_CUT                             # 0.974 library near-dup / morph-look cut
C_FAMILIES = ("mandelbrot", "multibrot3", "multibrot4", "multibrot5")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def load_state(run_dir: Path) -> dict:
    for name in ("state.json", "dive_state.json", "summary.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def active_min_of(run_dir: Path) -> float:
    """Accumulated active minutes from the durable checkpoint (survives every resume)."""
    st = load_state(run_dir)
    if "active_s" in st:
        return float(st["active_s"]) / 60.0
    if "active_min" in st:
        return float(st["active_min"])
    return 0.0


def batch_active_map(run_dir: Path) -> dict[int, float]:
    """batch_i -> cumulative active MINUTES, parsed from the run stdout log (durable, in-tree).
    Empty if no stdout log is found (then the time trajectory falls back to the overall rate)."""
    m: dict[int, float] = {}
    for log in list(run_dir.glob("*stdout*.log")) + [run_dir.parent / f"{run_dir.name}_stdout.log"]:
        if not log.exists():
            continue
        for line in open(log, encoding="utf-8", errors="replace"):
            mt = re.search(r"batch (\d+):.*active=([\d.]+)m", line)
            if mt:
                m[int(mt.group(1))] = float(mt.group(2))
    return m


def admit_batch_map(run_dir: Path) -> dict:
    """node_id -> batch for every ADMITTED harvest check (join key to the ledger's admissions)."""
    out = {}
    for h in load_jsonl(run_dir / "harvest_log.jsonl"):
        if h.get("admitted"):
            out[h["node_id"]] = int(h["batch"])
    return out


def admissions(rows: list) -> list:
    """Distinct-q3 admitted rows, in ledger (chronological) order."""
    return spm.admitted_q3(rows)


# --------------------------------------------------------------------------- #
# 4a. Coord library-overlap vs ALL prior ledgers (cheap; no render, no GPU).
# --------------------------------------------------------------------------- #
def prior_clouds(prior_ledgers: list[Path], partitions: list[str]) -> dict:
    """Per-partition distinct-q3 cloud unioned over every prior ledger (built with the exact
    production dedup so 'within coord-dup radius' means the same thing it does in the harvest)."""
    all_rows = []
    for led in prior_ledgers:
        all_rows += load_jsonl(led)
    return {part: ps.build_cloud(all_rows, part) for part in partitions}


def coord_overlap(adm: list, priors: dict) -> tuple[int, int, dict]:
    """# campaign admissions that fall inside a prior admission's coord-dup radius (same partition)."""
    hit, tot = 0, 0
    per_fam = defaultdict(lambda: [0, 0])
    for r in adm:
        part = r.get("family", "mandelbrot")
        cloud = priors.get(part, [])
        distinct, _ = ps.is_distinct(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], cloud)
        tot += 1
        per_fam[part][1] += 1
        if not distinct:
            hit += 1
            per_fam[part][0] += 1
    return hit, tot, per_fam


# --------------------------------------------------------------------------- #
# Morph pass (GPU): embed campaign admissions once, reuse for metrics 3 + 4b.
# --------------------------------------------------------------------------- #
def _norm(E):
    E = np.asarray(E, np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def morph_embed(adm: list):
    """(uids, fams, depths, normalized E[N,768]) via the library morph_gray recipe."""
    model, tf = spm.load_clip()
    tmp = ROOT / "out" / "campaign1" / "morph_fields"
    uids, fams, depths, E = spm.embed_admissions(adm, tmp, model, tf)
    return uids, fams, depths, _norm(E)


def distinct_look_count(uids, fams, E) -> tuple[int, dict]:
    """Within-family single-linkage at 0.974 -> distinct morph looks (headline count)."""
    if len(uids) == 0:
        return 0, {}
    idx_by_fam = defaultdict(list)
    for i, f in enumerate(fams):
        idx_by_fam[f].append(i)
    distinct, per_fam = 0, {}
    for f, idx in idx_by_fam.items():
        sub = E[idx]
        C = sub @ sub.T
        comps = spm.connected_components(len(idx), STRICT, C)
        per_fam[f] = (len(comps), len(idx))     # (distinct, admitted)
        distinct += len(comps)
    return distinct, per_fam


def morph_over_time(adm, uids, fams, E, tstamp) -> list:
    """Cumulative distinct-look count as active time accrues. Processes admissions in
    chronological (ledger) order; an admission is a NEW look iff its max within-family cosine
    to all earlier-admitted looks is < 0.974. Returns [(active_min, cum_admitted, cum_distinct)]."""
    emb_by = {u: E[i] for i, u in enumerate(uids)}
    fam_by = {u: fams[i] for i, u in enumerate(uids)}
    seen: dict[str, list] = defaultdict(list)
    curve, cum_adm, cum_dist = [], 0, 0
    for r in adm:
        u = r["id"]
        if u not in emb_by:
            continue
        cum_adm += 1
        e, f = emb_by[u], fam_by[u]
        prior = seen[f]
        is_new = True
        if prior:
            if float(np.max(np.stack(prior) @ e)) >= STRICT:
                is_new = False
        if is_new:
            cum_dist += 1
            seen[f].append(e)
        curve.append((tstamp.get(u), cum_adm, cum_dist))
    return curve


def morph_library_overlap(uids, fams, E) -> tuple[int, int, dict]:
    """Fraction of campaign admissions that are morph near-dups (cos>=0.974) of a PRIOR library
    admission, using the durable library_store morph_clip embeddings. (-1,-1,{}) if unavailable."""
    try:
        from library_store import load_library_embeddings
    except Exception:
        return -1, -1, {}
    lib = load_library_embeddings()
    if not lib:
        return -1, -1, {}
    L = _norm(np.stack([lib[u] for u in lib]))
    hit, tot = 0, 0
    per_fam = defaultdict(lambda: [0, 0])
    for i, u in enumerate(uids):
        tot += 1
        per_fam[fams[i]][1] += 1
        if float(np.max(L @ E[i])) >= STRICT:
            hit += 1
            per_fam[fams[i]][0] += 1
    return hit, tot, per_fam


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def per_family_counts(adm: list) -> dict:
    c = defaultdict(int)
    for r in adm:
        c[r.get("family", "mandelbrot")] += 1
    return dict(c)


def hourly_admissions(adm, tstamp) -> list:
    """[(hour_index, n_admissions_in_that_active-hour)] — adm/hr directly (1-hr bins)."""
    by_hour = defaultdict(int)
    n_timed = 0
    for r in adm:
        t = tstamp.get(r["id"])
        if t is None:
            continue
        n_timed += 1
        by_hour[int(t // 60.0)] += 1
    return sorted(by_hour.items()), n_timed


def build(args) -> str:
    breadth = Path(args.breadth).resolve()
    dive = Path(args.dive).resolve() if args.dive else None

    b_rows = load_jsonl(breadth / "outcome_ledger.jsonl")
    b_adm = admissions(b_rows)
    b_active = active_min_of(breadth)
    b_bmap = batch_active_map(breadth)
    b_admbatch = admit_batch_map(breadth)
    # chronological active-minute timestamp per breadth admission (node_id -> batch -> active_min)
    b_tstamp = {r["id"]: b_bmap.get(b_admbatch.get(r["node_id"]))
                for r in b_adm}

    d_adm, d_active = [], 0.0
    if dive and (dive / "outcome_ledger.jsonl").exists():
        d_rows = load_jsonl(dive / "outcome_ledger.jsonl")
        d_adm = admissions(d_rows)
        d_active = active_min_of(dive)

    all_adm = b_adm + d_adm
    partitions = list(C_FAMILIES) + [ps.julia_partition(f) for f in C_FAMILIES]

    L = []
    w = L.append
    w("# Campaign 1 — steered frontier + dive: scheduling readout\n")
    w(f"_Regenerated from ledgers + state. Breadth `{breadth.name}` active "
      f"**{b_active:.1f} min** ({b_active/60:.2f} h); "
      + (f"dive `{dive.name}` active **{d_active:.1f} min** ({d_active/60:.2f} h)._\n"
         if dive else "_dive: not run yet._\n"))

    # ---- 1. admissions/hr over accumulated active time ----
    w("## 1. Admissions/hr over accumulated active time\n")
    b_rate = len(b_adm) / (b_active / 60.0) if b_active else 0.0
    w(f"- **Breadth overall: {b_rate:.1f} adm/hr** ({len(b_adm)} admitted / {b_active/60:.2f} active-h).")
    if d_adm:
        d_rate = len(d_adm) / (d_active / 60.0) if d_active else 0.0
        w(f"- **Dive overall: {d_rate:.1f} adm/hr** ({len(d_adm)} admitted / {d_active/60:.2f} active-h).")
    hrs, n_timed = hourly_admissions(b_adm, b_tstamp)
    if hrs:
        w(f"\nBreadth admissions per active-hour bin ({n_timed}/{len(b_adm)} admissions time-stamped "
          f"from stdout×harvest_log):\n")
        w("| active-hr | admissions (=adm/hr) |")
        w("|--:|--:|")
        for h, n in hrs:
            w(f"| {h}–{h+1} | {n} |")
        first = hrs[0][1]
        last = hrs[-1][1]
        verdict = ("HOLDS" if last >= 14 else "DECAYS below the ~16/hr floor")
        w(f"\n_Verdict: first-hr {first} → last-hr {last} adm/hr — floor **{verdict}**._\n")
    else:
        w("\n_No per-batch active-time map (stdout log absent) — only the overall rate above._\n")

    # ---- 2. per-family admissions + cost, breadth vs dive ----
    w("## 2. Per-family admissions & cost (breadth vs dive)\n")
    bpf, dpf = per_family_counts(b_adm), per_family_counts(d_adm)
    b_cost = (b_active / len(b_adm)) if b_adm else float("nan")
    w(f"- Breadth cost/admission: **{b_cost:.2f} active-min** ({b_active:.0f} min / {len(b_adm)}).")
    if d_adm:
        d_cost = (d_active / len(d_adm)) if d_adm else float("nan")
        w(f"- Dive cost/admission: **{d_cost:.2f} active-min** ({d_active:.0f} min / {len(d_adm)}).")
    w("\n| partition | breadth adm | dive adm |")
    w("|---|--:|--:|")
    for part in partitions:
        if bpf.get(part, 0) or dpf.get(part, 0):
            w(f"| {part} | {bpf.get(part,0)} | {dpf.get(part,0)} |")
    # harvest-check cost proxy per family (canonical renders spent per admission)
    checks = defaultdict(lambda: [0, 0])  # part -> [checks, admits]
    for h in load_jsonl(breadth / "harvest_log.jsonl"):
        checks[h["partition"]][0] += 1
        if h.get("admitted"):
            checks[h["partition"]][1] += 1
    if any(v[0] for v in checks.values()):
        w("\nBreadth compute proxy — canonical confirmation renders per admission:\n")
        w("| partition | harvest_checks | admits | checks/admit |")
        w("|---|--:|--:|--:|")
        for part in partitions:
            c, a = checks.get(part, [0, 0])
            if c:
                w(f"| {part} | {c} | {a} | {c/a:.1f} |" if a else f"| {part} | {c} | 0 | ∞ |")

    # ---- 4a. coord library overlap (cheap; compute before the GPU pass) ----
    prior_ledgers = [Path(p) for p in args.prior_ledgers]
    priors = prior_clouds(prior_ledgers, partitions)
    n_prior = sum(len(v) for v in priors.values())
    ch, ct, cpf = coord_overlap(all_adm, priors)

    # ---- 3 + 4b. morph pass ----
    morph_note = ""
    if args.no_morph or not all_adm:
        morph_note = "_Morph pass skipped (--no-morph or no admissions)._"
        dist_total, dist_pf, curve = None, {}, []
        mh = mt = -1
    else:
        print(f"[morph] embedding {len(all_adm)} admissions (library morph_gray recipe) ...", flush=True)
        uids, fams, depths, E = morph_embed(all_adm)
        dist_total, dist_pf = distinct_look_count(uids, fams, E)
        # chronological timestamps: breadth from stdout×harvest; dive admissions get None (still
        # counted in cumulative distinct in ledger order, just not placed on the active-hr axis).
        tstamp = {u: b_tstamp.get(u) for u in uids}
        curve = morph_over_time(all_adm, uids, fams, E, tstamp)
        mh, mt, mpf = morph_library_overlap(uids, fams, E)

    w("## 3. Distinct morph-look count over time\n")
    if dist_total is None:
        w(morph_note + "\n")
    else:
        w(f"- **{dist_total} distinct morph looks** among {len(all_adm)} admissions "
          f"(within-family single-linkage, CLIP≥{STRICT} on the library morph_gray recipe) "
          f"= {dist_total/max(1,len(all_adm)):.0%} distinct.")
        w("\n| partition | distinct looks | admitted |")
        w("|---|--:|--:|")
        for part in partitions:
            if part in dist_pf:
                w(f"| {part} | {dist_pf[part][0]} | {dist_pf[part][1]} |")
        timed = [(t, cd) for (t, ca, cd) in curve if t is not None]
        if timed:
            w("\nCumulative distinct looks vs accumulated active time (breadth-timed admissions):\n")
            w("| active-hr | cum admitted | cum distinct looks |")
            w("|--:|--:|--:|")
            marks = {}
            for t, ca, cd in curve:
                if t is None:
                    continue
                marks[int(t // 60.0)] = (ca, cd)
            for h in sorted(marks):
                ca, cd = marks[h]
                w(f"| ≤{h+1} | {ca} | {cd} |")

    # ---- 4. library overlap ----
    w("## 4. Library overlap vs prior admissions\n")
    w(f"_Prior corpus: {len(prior_ledgers)} ledgers, {n_prior} distinct-q3 places (coord); "
      f"library embedding store for morph._\n")
    if ct:
        w(f"- **Coord-dup overlap: {ch}/{ct} = {ch/ct:.1%}** of campaign admissions fall inside a "
          f"prior admission's coord-dup radius (same partition, DEDUP_K={ps.DEDUP_K}).")
        nz = [(p, h, t) for p, (h, t) in cpf.items() if t]
        if nz:
            w("  - per partition: " + ", ".join(f"{p} {h}/{t}" for p, h, t in sorted(nz)))
    if mh >= 0:
        w(f"- **Morph near-dup overlap: {mh}/{mt} = {mh/mt:.1%}** are CLIP≥{STRICT} near-dups of a "
          f"library admission (library morph_gray recipe).")
    else:
        w(f"- Morph near-dup: not computed ({'--no-morph' if args.no_morph else 'library embedding store absent'}).")
    verdict = "WORTH building a cross-run freshness prior" if (ct and ch / ct > 0.10) \
        else "NOT worth a cross-run freshness prior yet"
    if ct:
        w(f"\n_Verdict: coord overlap {ch/ct:.1%} → **{verdict}**._\n")

    # ---- 5. zero-admission families ----
    w("## 5. Family coverage (zero-admission flags)\n")
    w("| partition | admissions | flag |")
    w("|---|--:|---|")
    tot_pf = defaultdict(int)
    for r in all_adm:
        tot_pf[r.get("family", "mandelbrot")] += 1
    for part in partitions:
        n = tot_pf.get(part, 0)
        flag = "🚩 ZERO" if n == 0 else ("⚠ low" if n <= 2 else "")
        w(f"| {part} | {n} | {flag} |")
    zeros = [p for p in partitions if tot_pf.get(p, 0) == 0]
    if zeros:
        w(f"\n_Zero-admission: {', '.join(zeros)}"
          + ("  — multibrot4 among them (watched)." if "multibrot4" in zeros else "") + "._\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--breadth", required=True, help="breadth run dir")
    ap.add_argument("--dive", default=None, help="dive run dir (optional)")
    ap.add_argument("--out", default=str(ROOT / "out" / "campaign1" / "readout.md"))
    ap.add_argument("--no-morph", action="store_true", help="skip the GPU morph pass (cheap metrics only)")
    ap.add_argument("--prior-ledgers", nargs="*", default=None,
                    help="prior outcome_ledger.jsonl paths (default: all under data/ except campaign1)")
    args = ap.parse_args()
    if args.prior_ledgers is None:
        args.prior_ledgers = [str(p) for p in sorted((ROOT / "data").rglob("outcome_ledger.jsonl"))
                              if "campaign1" not in p.parts]
    md = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nwrote {out}\n")
    print(md)


if __name__ == "__main__":
    main()
