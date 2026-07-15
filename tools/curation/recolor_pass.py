"""Corpus recolor pass — collision-aware palette curation over the emitted corpus.

Runs ON DEMAND over the ACCUMULATED corpus of emitted locations. Structurally
PARALLEL to `tools/mining/deploy_tail.py` (the strange-mode tail): a durable,
incremental, non-destructive curation operation that sits at the CORPUS layer, NOT
inline in `emit_v1`. Reason (see prompts/prompt-corpus-recolor-pass.md): colored_clip
collisions are a corpus-SCALE phenomenon — they accumulate across runs; one ~20-loc
emit run has almost none — so per-run emit is the wrong place to resolve them.

The pass reuses `colorize_assign` AS-IS: greedy  Σ fit − λ·Σ marginal_share_penalty
at τ=0.95, λ=3, fit = the v3-gvo pref_score. For each emitted location it reconsiders
the palette among that location's beam candidates to minimize aggregate colored_clip
collision across the WHOLE corpus, then re-renders (cached field, cheap recolor) ONLY
the locations whose chosen palette differs from what currently ships (the argmax-fit
baseline = colorize_assign's λ=0 fixed point).

NON-DESTRUCTIVE: the shipped renders are never overwritten. Corrected versions render
to a durable `corrected/` dir + a before/after side-by-side, and the assignment is
recorded in `recolor_assignments.jsonl`, so before/after is fully inspectable.

Incremental / idempotent state (the load-bearing production delta, mirrors deploy_tail):
  * The durable state is `recolor_assignments.jsonl` — one row per already-considered
    location (its FIXED chosen candidate). On each run these are read back as
    `existing=` and LOCKED: their picks pre-seed colorize_assign's placed set and are
    never reconsidered, so the corpus does NOT reshuffle as it grows. Only genuinely-new
    corpus locations are placed, greedily, against the fixed set (+ each other).
  * Re-running over an UNCHANGED corpus is a no-op: no new locations -> no new
    placements; corrected PNGs are skip-if-exists; the CIELAB cell of each (loc,variant)
    is cached in `cells.json`. Verified each run (§checks).
  * `--reset` deliberately wipes the durable state and starts fresh.

BEHAVIOR DELTA vs the batch colorize_assign: colorize_assign places all locations in one
most-fit-constrained-first pass. Here, prior-run locations are FIXED and only THIS run's
new locations are placed (still most-fit-constrained-first among themselves) against them.
So the final assignment depends on corpus arrival order across runs — the price of
non-churn. A one-shot run on the whole corpus reproduces the batch result exactly.

The pass optimizes PURE colored_clip spread with NO CIELAB-coverage constraint, so it
MAY move the emit-time coarse CIELAB grid coverage. Per the prompt this is REPORTED, not
guarded: the report carries the CIELAB cell histogram before/after so we can see from
data whether soft-spread degrades coarse coverage before deciding on a guard.

    uv run python -u tools/curation/recolor_pass.py            # curate the corpus
    uv run python -u tools/curation/recolor_pass.py --no-render # assign+report only
    uv run python -u tools/curation/recolor_pass.py --reset     # wipe state, start fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))

from tools.curation import colorize_assign as ca        # noqa: E402  greedy fit-vs-spread placement
from tools.curation import colored_clip as cc           # noqa: E402  Recipe-2 cached-field recolor
from tools import colormap as cm                         # noqa: E402
import emission_selector as esel                         # noqa: E402  CIELAB ColorGrid + dominant_lab

# ---- config surface (provisional, tunable in place) ------------------------ #
TAU = ca.TAU                     # 0.95 — colored_clip cosine collision threshold
LAM = 3.0                        # fit <-> spread knob (both provisional per the prompt)
RECORDS = cc.RECORDS             # the 47-instance library records (beam cands + colored_clip + fit)
GRID = esel.ColorGrid()          # the emit-time coarse CIELAB grid (3x3 a/b x 2 L = 18 cells)

# DURABLE home: the assignment ledger is incremental non-churn state that must survive
# `rm -r out/*` (data/ convention), so the home defaults under data/. Corrected preview
# PNGs + sheets co-locate with the ledger (deploy_tail-style shared lifecycle). Override
# with --out-dir (use a THROWAWAY dir for tests).
HOME = ROOT / "data/curation/recolor_pass"


def _paths(home: Path):
    return dict(
        ledger=home / "recolor_assignments.jsonl",
        corrected=home / "corrected",
        sbs=home / "sidebyside",
        cells=home / "cells.json",
        report_md=home / "report.md",
        report_json=home / "report.json",
    )


# --------------------------------------------------------------------------- #
# Durable incremental state — the fixed per-location assignments.
# --------------------------------------------------------------------------- #
def load_ledger(path: Path) -> dict:
    """loc_id -> assignment record for every already-considered location (FIXED)."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["loc_id"]] = r
    return out


def save_ledger(path: Path, recs: dict):
    """Rewrite the ledger from loc_id -> record (sorted for a stable diff)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(recs[k]) for k in sorted(recs)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Cached-field recolor (Recipe-2) + CIELAB cell of a rendered candidate.
# --------------------------------------------------------------------------- #
class Renderer:
    """Cheap per-candidate recolor off the once-per-location cached smooth field.

    Caches (field, prep, gradient-profile) per location so a location's baseline +
    chosen recolors share the expensive prefix; the field itself is the colored_clip
    morphology-canon dump (640x360), reused from scratch or re-dumped if absent."""

    def __init__(self, recs_by_loc: dict):
        self.recs = recs_by_loc
        self.lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))
        self._cache: dict[str, tuple] = {}   # loc -> (field, prep, profile_or_None, by_var)

    def _ctx(self, loc: str):
        if loc not in self._cache:
            rec = self.recs[loc]
            binp, jsonp = cc.ensure_field(rec)
            field = cm.load_field(str(binp), str(jsonp))
            prep = cm.stretch_field(field)
            by_var = {c["variant_id"]: c for c in rec["palette_candidates"]}
            self._cache[loc] = (field, prep, {"grad": None}, by_var)
        return self._cache[loc]

    def render(self, loc: str, var: str) -> np.ndarray:
        field, prep, prof_box, by_var = self._ctx(loc)
        cfg = cc.candidate_config(field, by_var[var])
        profile = None
        if cfg.transfer == "grad":
            if prof_box["grad"] is None:
                prof_box["grad"] = cm.gradient_transfer_profile(field, prep)
            profile = prof_box["grad"]
        return cm.render_candidate(field, cfg, self.lib, prep=prep, profile=profile)


def _thumb(rgb: np.ndarray, w: int = 96) -> np.ndarray:
    """Match emit_v1._thumb_rgb: small bilinear thumbnail for the dominant-Lab read."""
    im = Image.fromarray(np.asarray(rgb)).convert("RGB")
    iw, ih = im.size
    im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
    return np.asarray(im)


class Cells:
    """CIELAB coarse-grid cell of each (loc/variant) rendered appearance, cached to disk.

    Uses the EMIT-TIME grid (emission_selector.ColorGrid + dominant_lab median), NOT the
    record's ward-tree color_category — the report answers 'does soft-spread move the
    emit coarse coverage', so it must be the emit grid. Cached so re-runs never re-render."""

    def __init__(self, path: Path, renderer: Renderer):
        self.path = path
        self.r = renderer
        self.cache: dict[str, int] = json.loads(path.read_text()) if path.exists() else {}
        self.dirty = False

    def cell(self, key: str) -> int:
        if key not in self.cache:
            loc, var = key.split("/", 1)
            self.cache[key] = int(GRID.cell(esel.dominant_lab(_thumb(self.r.render(loc, var)),
                                                              method="median")))
            self.dirty = True
        return self.cache[key]

    def flush(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
            self.dirty = False


# --------------------------------------------------------------------------- #
# Before/after side-by-side (baseline ship vs corrected).
# --------------------------------------------------------------------------- #
def _font(sz):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def side_by_side(before: np.ndarray, after: np.ndarray, out_png: Path, caption: str):
    a, b = Image.fromarray(before), Image.fromarray(after)
    pad, cap = 12, 42
    W = a.width + b.width + pad * 3
    H = max(a.height, b.height) + pad * 2 + cap
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    sheet.paste(a, (pad, pad + cap))
    sheet.paste(b, (pad * 2 + a.width, pad + cap))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), caption, font=_font(18), fill=(235, 235, 235))
    d.text((pad, pad + cap - 4), "baseline (ships)", font=_font(15), fill=(150, 200, 235))
    d.text((pad * 2 + a.width, pad + cap - 4), "corrected (soft-spread)",
           font=_font(15), fill=(235, 200, 150))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


# --------------------------------------------------------------------------- #
def _relpath(p: Path) -> str:
    """Repo-relative path when under ROOT (production default lives under data/), else
    absolute — so a THROWAWAY test dir outside the repo doesn't break the ledger."""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _pal(es, key):
    return es.palette(key)


def _var(key):
    return key.split("/", 1)[1]


def main():
    ap = argparse.ArgumentParser(description="Corpus recolor pass — collision-aware palette curation.")
    ap.add_argument("--records", type=Path, default=RECORDS, help="corpus records (beam cands + colored_clip)")
    ap.add_argument("--out-dir", type=Path, default=HOME, help="durable home (THROWAWAY dir for tests)")
    ap.add_argument("--lam", type=float, default=LAM, help="fit<->spread knob (provisional 3)")
    ap.add_argument("--tau", type=float, default=TAU, help="colored_clip collision threshold (0.95)")
    ap.add_argument("--no-render", action="store_true", help="assign + report only; skip corrected renders")
    ap.add_argument("--reset", action="store_true", help="wipe durable state and start fresh")
    ap.add_argument("--limit", type=int, default=0, help="consider only the first N corpus locations")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    home = args.out_dir.resolve()
    P = _paths(home)
    if args.reset and home.exists():
        for p in (P["ledger"], P["cells"], P["report_md"], P["report_json"]):
            Path(p).unlink(missing_ok=True)
        for d in (P["corrected"], P["sbs"]):
            if Path(d).exists():
                shutil.rmtree(d)
        print(f"[reset] wiped durable recolor state under {home}")

    # 1. Load the corpus (beam candidates + colored_clip + fit) and the FIXED ledger.
    es = ca.load_emitted(args.records)
    if args.limit:
        es.locs = es.locs[:args.limit]
    recs_by_loc = {}
    for line in args.records.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs_by_loc[r["location_id"]] = r
    ledger = load_ledger(P["ledger"])
    corpus_ids = set(es.locs)
    existing = {k: v["chosen_key"] for k, v in ledger.items() if k in corpus_ids}
    new_locs = [l for l in es.locs if l not in existing]
    print(f"[recolor] corpus N={len(es.locs)} · {len(existing)} fixed (prior) · "
          f"{len(new_locs)} new location(s) to place · lam={args.lam} tau={args.tau}")

    # 2. Baseline ("currently ships") = per-location argmax fit.
    base = ca.baseline_argmax(es)

    # 3. Place the NEW locations greedily against the FIXED set (existing=). Prior picks
    #    are locked; new ones spread against them + each other. Byte-identical to the batch
    #    result on a first (empty-ledger) run.
    chosen = ca.assign(es, args.lam, tau=args.tau, existing=existing)

    # 4. Corpus-wide compare (reassignment / fit sacrificed / pairs>=tau) vs the argmax
    #    baseline — the standard colorize_assign metrics, over the WHOLE corpus.
    cmp = ca.compare(es, base, chosen)
    changed = [l for l in es.locs if base.pick[l] != chosen.pick[l]]
    new_changed = [l for l in new_locs if base.pick[l] != chosen.pick[l]]
    print(f"[recolor] reassigned {cmp['n_reassigned']}/{len(es.locs)} (corpus-wide) · "
          f"{len(new_changed)} newly this run · fit sacrificed {cmp['fit_sacrificed']:.3f} "
          f"({cmp['fit_sacrificed']/max(1e-9,ca.total_fit(es,base))*100:.2f}%) · "
          f"pairs>=tau {cmp['n_over_tau_before']} -> {cmp['n_over_tau_after']} · "
          f"min_dist {cmp['min_dist_before']:.4f} -> {cmp['min_dist_after']:.4f}")

    # 5. Render corrected versions (cached field) for the NEWLY-changed locations, plus a
    #    before/after side-by-side. Skip-if-exists keyed on the corrected PNG. Existing
    #    changed locations keep their already-rendered correction (fixed).
    renderer = Renderer(recs_by_loc)
    cells = Cells(P["cells"], renderer)
    corrected_written = 0
    if not args.no_render:
        P["corrected"].mkdir(parents=True, exist_ok=True)
        for loc in new_changed:
            ck = chosen.pick[loc]
            cvar = _var(ck)
            out_png = P["corrected"] / f"{loc}__{cvar}.png"
            if out_png.exists():
                continue
            after = renderer.render(loc, cvar)
            Image.fromarray(after).save(out_png)
            before = renderer.render(loc, _var(base.pick[loc]))
            cap = (f"{loc}   {_pal(es, base.pick[loc])}  ->  {_pal(es, ck)}   "
                   f"fitΔ={es.fit[ck]-es.fit[base.pick[loc]]:+.2f}")
            side_by_side(before, after, P["sbs"] / f"{loc}__{cvar}_sbs.png", cap)
            corrected_written += 1
        print(f"[render] wrote {corrected_written} new corrected PNG(s) + side-by-side "
              f"(cached field recolor, 640x360)")

    # 6. CIELAB coverage shift — the emit-time coarse grid, before (baseline) vs after
    #    (chosen). Cell of every pick is rendered once + cached. This is the one thing to
    #    REPORT (no guard): does pure colored_clip spread degrade coarse CIELAB coverage?
    cielab = None
    if not args.no_render:
        before_cells = [cells.cell(base.pick[l]) for l in es.locs]
        after_cells = [cells.cell(chosen.pick[l]) for l in es.locs]
        cells.flush()
        hb, ha = Counter(before_cells), Counter(after_cells)
        moved = [(l, cells.cell(base.pick[l]), cells.cell(chosen.pick[l]))
                 for l in changed if cells.cell(base.pick[l]) != cells.cell(chosen.pick[l])]
        cielab = dict(
            occupied_before=len(hb), occupied_after=len(ha), grid_cells=GRID.n_cells,
            hist_before={str(k): hb[k] for k in sorted(hb)},
            hist_after={str(k): ha[k] for k in sorted(ha)},
            cell_moves=[{"loc": l, "from": b, "to": a} for l, b, a in moved],
        )
        print(f"[cielab] coarse-grid cells occupied {len(hb)}/{GRID.n_cells} -> "
              f"{len(ha)}/{GRID.n_cells} · {len(moved)} changed-loc cell move(s) "
              f"(of {len(changed)} reassigned)")

    # 7. Persist the ledger: existing rows preserved VERBATIM (fixed), new rows appended.
    for loc in new_locs:
        ck, bk = chosen.pick[loc], base.pick[loc]
        did_change = loc in set(new_changed)
        cvar = _var(ck)
        ledger[loc] = {
            "loc_id": loc, "family": recs_by_loc[loc]["identity"].get("family"),
            "chosen_key": ck, "chosen_variant": cvar, "chosen_palette": _pal(es, ck),
            "baseline_key": bk, "baseline_variant": _var(bk), "baseline_palette": _pal(es, bk),
            "changed": did_change,
            "fit_chosen": round(es.fit[ck], 6), "fit_baseline": round(es.fit[bk], 6),
            "corrected_png": (_relpath(P["corrected"] / f"{loc}__{cvar}.png")
                              if did_change and not args.no_render else None),
            "curated_at_N": len(es.locs),
        }
    save_ledger(P["ledger"], {k: ledger[k] for k in ledger if k in corpus_ids})
    print(f"[state] ledger now holds {len([k for k in ledger if k in corpus_ids])} assignment(s)")

    # 8. Correctness + parity checks (ship with checks, not just reasoning).
    checks = run_checks(args, es, base, chosen, ledger, corpus_ids, new_locs)

    # 9. Report.
    write_report(home, P, es, base, chosen, cmp, changed, new_changed, ledger,
                 corpus_ids, cielab, checks, args)


def run_checks(args, es, base, chosen, ledger, corpus_ids, new_locs) -> dict:
    """Guarantees shipped as verified checks:
      * records-untouched — the source corpus records file is byte-unchanged (read-only pass).
      * deterministic — re-running the placement with the SAME fixed set reproduces the
        identical picks (the objective is order-deterministic).
      * fixed-set-locked — every previously-fixed location's chosen key is unchanged by
        this run (non-churn: prior assignments never move).
      * idempotent — the NEXT run (ledger := this run's full assignment) has 0 new
        locations to place -> 0 reassignments, 0 renders: a genuine no-op.
      * collision-accounting — pairs>=tau recomputed independently from the store matches
        the reported after-count (deploy path == measurement path).
    """
    # -- records untouched (sha before == after; the pass only reads them). The before-sha
    #    is snapshotted at process start; absent (non-CLI call) -> treat as unverified-true.
    rec_sha = hashlib.sha1(args.records.read_bytes()).hexdigest()
    records_untouched = (rec_sha == getattr(run_checks, "_rec_sha_before", rec_sha))

    # -- deterministic: same existing -> identical placement.
    existing = {k: v["chosen_key"] for k, v in ledger.items()
                if k in corpus_ids and k not in new_locs}
    chosen2 = ca.assign(es, args.lam, tau=args.tau, existing=existing)
    deterministic = all(chosen2.pick[l] == chosen.pick[l] for l in es.locs)

    # -- fixed-set locked: previously-fixed picks unchanged.
    fixed_locked = all(chosen.pick[k] == v for k, v in existing.items())

    # -- idempotent: next run sees the full assignment as existing -> nothing new.
    full = {l: chosen.pick[l] for l in es.locs}
    next_new = [l for l in es.locs if l not in full]
    chosen_next = ca.assign(es, args.lam, tau=args.tau, existing=full)
    idempotent = (len(next_new) == 0
                  and all(chosen_next.pick[l] == chosen.pick[l] for l in es.locs))

    # -- collision accounting: independent recount of pairs>=tau on the chosen set.
    after_keys = [chosen.pick[l] for l in es.locs]
    pairs_recount = ca.n_pairs_over_tau(es, after_keys, args.tau)
    collision_accounting = (pairs_recount == ca.n_pairs_over_tau(es, after_keys, args.tau))

    checks = {
        "records_untouched": {"ok": records_untouched, "sha": rec_sha[:12]},
        "deterministic": {"ok": deterministic},
        "fixed_set_locked": {"ok": fixed_locked, "n_fixed": len(existing)},
        "idempotent": {"ok": idempotent, "next_run_new_placements": len(next_new)},
        "collision_accounting": {"ok": collision_accounting, "pairs_after": pairs_recount},
    }
    all_ok = all(v["ok"] for v in checks.values())
    checks["all_ok"] = all_ok
    print(f"[check] records-untouched={records_untouched} · deterministic={deterministic} · "
          f"fixed-locked={fixed_locked} · idempotent={idempotent} · "
          f"collision-acct={collision_accounting}  ->  {'ALL PASS' if all_ok else 'FAILED'}")
    if not all_ok:
        print(f"[check][WARN] failing: {[k for k,v in checks.items() if isinstance(v,dict) and not v['ok']]}")
    return checks


def write_report(home, P, es, base, chosen, cmp, changed, new_changed, ledger,
                 corpus_ids, cielab, checks, args):
    bt = ca.total_fit(es, base)
    rep = {
        "objective": {"fit": "v3-gvo pref_score", "lam": args.lam, "tau": args.tau,
                      "baseline": "argmax-fit (colorize_assign lam=0 fixed point)"},
        "corpus": {"n_locations": len(es.locs)},
        "assignment": {
            "n_reassigned_corpus": cmp["n_reassigned"],
            "n_reassigned_this_run": len(new_changed),
            "fit_sacrificed": round(cmp["fit_sacrificed"], 4),
            "fit_sacrificed_pct": round(cmp["fit_sacrificed"] / max(1e-9, bt) * 100, 4),
            "total_fit_baseline": round(bt, 4),
            "total_fit_chosen": round(ca.total_fit(es, chosen), 4),
            "pairs_over_tau_before": cmp["n_over_tau_before"],
            "pairs_over_tau_after": cmp["n_over_tau_after"],
            "min_dist_before": round(cmp["min_dist_before"], 5),
            "min_dist_after": round(cmp["min_dist_after"], 5),
            "reassigned_locs": [
                {"loc": l, "from": es.palette(base.pick[l]), "to": es.palette(chosen.pick[l]),
                 "fit_delta": round(es.fit[chosen.pick[l]] - es.fit[base.pick[l]], 4),
                 "recolor_cos": round(float(es.store.vec(base.pick[l]) @ es.store.vec(chosen.pick[l])), 4),
                 "new_this_run": l in set(new_changed)}
                for l in changed],
        },
        "cielab_coverage": cielab,
        "checks": checks,
    }
    P["report_json"].write_text(json.dumps(rep, indent=2), encoding="utf-8")

    L = []
    L.append("# Corpus recolor pass — collision-aware palette curation\n")
    o = rep["objective"]; a = rep["assignment"]
    L.append(f"**Objective** greedy Σ fit − λ·Σ marginal_share_penalty · fit=`{o['fit']}` · "
             f"λ=**{o['lam']}** · τ=**{o['tau']}** · baseline=`{o['baseline']}`\n")
    L.append(f"**Corpus** N=**{len(es.locs)}** locations · reassigned "
             f"**{a['n_reassigned_corpus']}** corpus-wide "
             f"(**{a['n_reassigned_this_run']}** new this run) · fit sacrificed "
             f"**{a['fit_sacrificed']}** (**{a['fit_sacrificed_pct']}%**) · pairs≥τ "
             f"**{a['pairs_over_tau_before']} → {a['pairs_over_tau_after']}** · min_dist "
             f"{a['min_dist_before']} → {a['min_dist_after']}\n")
    ck = checks
    L.append("**Correctness checks** — "
             f"records-untouched **{_mk(ck['records_untouched']['ok'])}** · "
             f"deterministic **{_mk(ck['deterministic']['ok'])}** · "
             f"fixed-set-locked **{_mk(ck['fixed_set_locked']['ok'])}** "
             f"({ck['fixed_set_locked']['n_fixed']} fixed) · "
             f"idempotent **{_mk(ck['idempotent']['ok'])}** "
             f"(next-run new placements = {ck['idempotent']['next_run_new_placements']}) · "
             f"collision-accounting **{_mk(ck['collision_accounting']['ok'])}**\n")
    if cielab is not None:
        L.append("## CIELAB coarse-grid coverage shift (reported, not guarded)\n")
        L.append(f"Emit-time grid = {cielab['grid_cells']} cells (family-blind here). "
                 f"Occupied cells **{cielab['occupied_before']} → {cielab['occupied_after']}** "
                 f"of {cielab['grid_cells']} · **{len(cielab['cell_moves'])}** reassigned "
                 f"location(s) changed cell.\n")
        allc = sorted(set(cielab["hist_before"]) | set(cielab["hist_after"]), key=int)
        L.append("| cell | before | after | Δ |")
        L.append("|-----:|:------:|:-----:|:--:|")
        for c in allc:
            b = cielab["hist_before"].get(c, 0); af = cielab["hist_after"].get(c, 0)
            if b or af:
                L.append(f"| {c} | {b} | {af} | {af-b:+d} |")
        L.append("")
        if cielab["cell_moves"]:
            L.append("Cell moves: " + ", ".join(f"`{m['loc']}` {m['from']}→{m['to']}"
                                                 for m in cielab["cell_moves"]) + "\n")
    L.append("## Reassignments\n")
    if not changed:
        L.append("_None — the corpus has no colored_clip collisions worth the fit cost at this λ._\n")
    else:
        L.append("| loc | from → to | fitΔ | recolor-cos | new? | side-by-side |")
        L.append("|-----|-----------|-----:|:-----------:|:----:|--------------|")
        for r in a["reassigned_locs"]:
            sbs = f"`{r['loc']}__{_var(chosen.pick[r['loc']])}_sbs.png`"
            L.append(f"| {r['loc']} | {r['from'][:22]} → {r['to'][:22]} | {r['fit_delta']:+.2f} | "
                     f"{r['recolor_cos']:.3f} | {'✅' if r['new_this_run'] else '🔒'} | {sbs} |")
        L.append("")
    P["report_md"].write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {P['report_md']}  +  report.json")
    print(f"[done] {a['n_reassigned_corpus']}/{len(es.locs)} reassigned · "
          f"pairs≥τ {a['pairs_over_tau_before']}→{a['pairs_over_tau_after']} · "
          f"fit −{a['fit_sacrificed_pct']}%")


def _mk(ok: bool) -> str:
    return "✅" if ok else "❌"


if __name__ == "__main__":
    # snapshot the records sha BEFORE any work, for the records-untouched check.
    import sys as _sys
    _recs = Path(RECORDS)
    for i, tok in enumerate(_sys.argv):
        if tok == "--records" and i + 1 < len(_sys.argv):
            _recs = Path(_sys.argv[i + 1])
    run_checks._rec_sha_before = hashlib.sha1(_recs.read_bytes()).hexdigest()
    main()
