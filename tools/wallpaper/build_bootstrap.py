"""Wallpaper-quality bootstrap — generate the labelable batch.

Generates ONE labelable training batch for a NEW absolute cross-location
wallpaper-quality classifier (its own forked head, built later — this script ONLY
produces the batch: no training, no label UI). The head must place an absolute
quality boundary on a FINISHED render, so its training data must span the full
quality range WITHIN each source location. The per-location beam render sampler
(`sample_location.run_location`, retain-all path) produces exactly that gradient in
one run — gen-0 spans many weak palettes, refinement climbs to strong renders — so
we run one beam per location and strata-sample across its whole pref-v2 trajectory.

Pipeline:
  1. Source ~60 locations from the gather ledgers (data/discovery/gather/<class>/),
     ~7 per class across all 9 classes, q3 core majority + a q2 marginal minority,
     light spatial dedup. The 4 julia:* classes live as `--julia-hook` sub-descent
     rows inside the c-plane dirs (family="julia:{fam}"), reconstructed from the row's
     parameter c (= outcome_cx/cy) + z-plane render viewport (= julia_z_cx/cy/fw).
  2. Per location: beam sampler with retain_all=True -> every evaluated candidate
     (gen-0 + refinement variants) with its (palette, params, gen, lineage, pref-v2).
  3. Strata-sample ~8 distinct-palette renders across the location's pref-v2 range
     (~3 low / 2 mid / 3 high). pref-v2 is within-location, exactly right here.
  4. Re-render ONLY the ~480 picks at the label-crop spec: 1280x720, ss2 Lanczos-3,
     q90 JPG, center, interior=black. The candidate coloring carries arbitrary
     params (gamma/phase/reverse/n_cycles) across multibrot/phoenix families, which
     `enrich --mode render` cannot express (named palette + mandelbrot point-trap
     only) — so the label crop is produced by the field-dump + colormap.py tail
     (render-one --dump-field at the label geometry, then render_candidate with
     filter=lanczos3), the >=1-LSB-parity equivalent of the same locked spec.
  5. Emit into a NEW namespace data/wallpaper_corpus/batches/<batch_id>/ mirroring
     the location-corpus row/triple format so the label UI + merge_scores operate on
     it unchanged. All label.score null.

    uv run python tools/wallpaper/build_bootstrap.py --estimate     # est + exit
    uv run python tools/wallpaper/build_bootstrap.py --limit 3      # smoke (3 locs)
    uv run python tools/wallpaper/build_bootstrap.py                # full run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))

import sample_location as SL          # noqa: E402  (run_location retain-all, load_v2, recolor)
import query_sampler as qs            # noqa: E402  (load_pool_library, PaletteSampler)
import colormap as cm                 # noqa: E402  (CandidateConfig, load_field, render_candidate)
import location as loc_mod            # noqa: E402  (canonical Location + render_one_flags)
from probe import auto_maxiter        # noqa: E402  (native fw-dependent maxiter policy)
from label_crop import (              # noqa: E402  (shared label-crop spec — Recipe-2 tail)
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q,
    ensure_label_field, render_label_crop,
)

GATHER_DIR = ROOT / "data" / "discovery" / "gather"
OUT_CORPUS = ROOT / "data" / "wallpaper_corpus"

BATCH_ID = "2026-07-05_wallpaper_bootstrap_v1"
GENERATOR_VERSION = "wallpaper_bootstrap_v1"

# The 9 target classes. All 9 ARE present in the gather pool: the 4 julia:* classes live
# as `--julia-hook` sub-descent rows inside the c-plane class dirs (family="julia:{fam}"),
# reconstructable because the row carries the parameter c (= outcome_cx/cy) and the render
# viewport (= julia_z_cx/cy/fw). Each spec: (class, ledger_dir, ledger_family, render_family, kind).
CLASS_SPECS = [
    ("mandelbrot",       "mandelbrot", "mandelbrot",       "mandelbrot",       "cplane"),
    ("multibrot3",       "multibrot3", "multibrot3",       "multibrot3",       "cplane"),
    ("multibrot4",       "multibrot4", "multibrot4",       "multibrot4",       "cplane"),
    ("multibrot5",       "multibrot5", "multibrot5",       "multibrot5",       "cplane"),
    ("phoenix",          "phoenix",    "phoenix",          "phoenix",          "phoenix"),
    ("julia:mandelbrot", "mandelbrot", "julia:mandelbrot", "julia",            "julia"),
    ("julia:multibrot3", "multibrot3", "julia:multibrot3", "julia_multibrot3", "julia"),
    ("julia:multibrot4", "multibrot4", "julia:multibrot4", "julia_multibrot4", "julia"),
    ("julia:multibrot5", "multibrot5", "julia:multibrot5", "julia_multibrot5", "julia"),
]
CLASS_KEYS = [s[0] for s in CLASS_SPECS]

# --- source-selection knobs ------------------------------------------------
# ~7/class across the 9 classes -> ~63 source locations (the prompt's ~60 / 6-8-per-class).
PER_CLASS_TARGET = 7
Q2_PER_CLASS = 2           # marginal (decoded_class==2) minority per class where available
DEDUP_FRAC = 0.5           # two coords within DEDUP_FRAC*fw of each other are the "same spot"

# --- per-location strata sampling ------------------------------------------
PICKS_PER_LOC = 8
STRATA_PLAN = (3, 2, 3)    # (low, mid, high) across the pref-v2 score RANGE

# The label-crop spec (LABEL_W/H/SS, LABEL_FILTER, JPG_Q) + ensure_label_field +
# render_label_crop are the shared canonical wallpaper label geometry (label_crop.py),
# unioned across all wallpaper batches so ss-level never correlates with tier.

# !!! PERF: the keeper label crops are the RUN'S BOTTLENECK, and it's an unintuitive one.
# Beam scoring is coarse+cheap now, the GPU is ~idle, and the Rust field dumps are
# multi-core — so you'd expect this to fly. It doesn't: each label crop is several seconds
# of SINGLE-THREADED ss2 Lanczos-3 numpy in colormap.render_candidate (2560x1440 = 3.69M
# supersampled px through the LUT gather + separable downsample), and there are 8 per
# location => tens of seconds/location on coloring ALONE. On a 12-core box that reads as
# ~8% CPU and looks stalled. We render a location's 8 picks concurrently to claw that back.
# THREADS not processes: the heavy numpy ops release the GIL, and the ~15MB ss2 field is
# shared read-only in-process (a process pool would re-pickle it per task). The ~3.3x wall
# speedup on the 8-crop phase was measured byte-identical to the serial path at the original
# ss4 spec (85s -> 26s/location); ss2 ~halves the absolute per-crop cost but the structure
# is unchanged and it stays short of a full 4x because the ops are memory-bandwidth-bound.
# CAP 4: the project-wide max-workers rule; DO NOT raise.
LABEL_CROP_WORKERS = 4

SEED = 7


# ===========================================================================
# 1. Source-location selection from the gather ledgers.
# ===========================================================================

def _render_coords(spec, row):
    """(cx, cy, fw, c) floats the row will actually be RENDERED at, per family kind.

    c-plane / phoenix: viewport = outcome_*, c = None (phoenix uses engine-default fixed
    c/p). julia: the render target is the z-plane (julia_z_*); the parameter c is the
    row's outcome_* (which for a julia-hook row IS the c-plane cloud coordinate). Returns
    None if a julia row lacks its z-plane viewport."""
    kind = spec[4]
    if kind == "julia":
        if row.get("julia_z_cx") is None:
            return None
        return (float(row["julia_z_cx"]), float(row["julia_z_cy"]), float(row["julia_z_fw"]),
                (float(row["outcome_cx"]), float(row["outcome_cy"])))
    return (float(row["outcome_cx"]), float(row["outcome_cy"]), float(row["outcome_fw"]), None)


def _load_ledger(spec):
    """Rows of `spec`'s class (ledger_dir filtered to ledger_family) with a usable
    decoded_class and render coords; each annotated with `_rc` = (cx,cy,fw,c) floats."""
    rows = []
    p = GATHER_DIR / spec[1] / "outcome_ledger.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("family") != spec[2] or d.get("decoded_class") not in (2, 3):
            continue
        rc = _render_coords(spec, d)
        if rc is None:
            continue
        d["_rc"] = rc
        rows.append(d)
    return rows


def _spatial_dedup(rows):
    """Greedy: drop a row whose render (cx,cy) lands within DEDUP_FRAC*fw of an
    already-kept row (same effective spot). Order-preserving on the input order."""
    kept = []
    for r in rows:
        cx, cy, fw, _ = r["_rc"]
        dup = False
        for k in kept:
            kcx, kcy, kfw, _ = k["_rc"]
            tol = DEDUP_FRAC * min(fw, kfw)
            if abs(cx - kcx) < tol and abs(cy - kcy) < tol:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def select_sources(seed, per_class=PER_CLASS_TARGET):
    """~per_class locations per class (all 9): a q3 core majority + up to Q2_PER_CLASS
    marginal (decoded_class==2). Deterministic (seeded shuffle), spatially deduped on the
    render coords. Returns [(spec, role, ledger_row)] and a per-class realized report."""
    rng = np.random.default_rng(seed)
    chosen, report = [], {}
    for spec in CLASS_SPECS:
        cls = spec[0]
        rows = _load_ledger(spec)
        q3 = _spatial_dedup([r for r in rows if r["decoded_class"] == 3])
        q2 = _spatial_dedup([r for r in rows if r["decoded_class"] == 2])
        rng.shuffle(q3)
        rng.shuffle(q2)
        n_q2 = min(Q2_PER_CLASS, len(q2), max(0, per_class - 1))
        n_q3 = min(len(q3), per_class - n_q2)
        picks = [(spec, "q3_core", r) for r in q3[:n_q3]] + \
                [(spec, "q2_marginal", r) for r in q2[:n_q2]]
        chosen.extend(picks)
        report[cls] = {"q3_core": n_q3, "q2_marginal": n_q2,
                       "q3_avail": len(q3), "q2_avail": len(q2)}
    return chosen, report


def to_location(spec, row):
    """Canonical Location from a gather ledger row. Coords are the f64's exact shortest
    round-tripping decimal (shallow regime — f64 IS ground truth). maxiter is the native
    fw-dependent policy on the render viewport. Julia carries the fixed parameter c and
    renders the z-plane viewport; c-plane/phoenix render the parameter plane (phoenix at
    the engine-default fixed c/p). Multibrot degree lives in the render `family`."""
    cx_f, cy_f, fw_f, c = row["_rc"]
    c_re = repr(c[0]) if c is not None else None
    c_im = repr(c[1]) if c is not None else None
    return loc_mod.Location(
        family=spec[3], cx=repr(cx_f), cy=repr(cy_f), fw=repr(fw_f),
        maxiter=auto_maxiter(fw_f), c_re=c_re, c_im=c_im,
    )


# ===========================================================================
# 3. Per-location strata sampling across the pref-v2 range.
# ===========================================================================

def strata_sample(all_candidates, rng):
    """~PICKS_PER_LOC distinct-palette candidates spread across the pref-v2 score RANGE.

    One representative per palette (its best-scoring evaluated candidate — a weak palette
    is its gen-0 render, a strong palette its refined best), then cut the score RANGE into
    3 equal-width bands and draw (low,mid,high)=STRATA_PLAN, spreading evenly-by-score
    within each band. A short band spills its deficit to the fullest remaining band, so we
    still land PICKS_PER_LOC distinct palettes when scores clump. Distinct palettes by
    construction (one rep each); multiple palettes per location is the point."""
    best = {}
    for c in all_candidates:
        cur = best.get(c["palette"])
        if cur is None or c["score"] > cur["score"]:
            best[c["palette"]] = c
    reps = sorted(best.values(), key=lambda c: c["score"])
    if len(reps) <= PICKS_PER_LOC:
        for c in reps:
            c["stratum"] = "all"
        return reps

    smin, smax = reps[0]["score"], reps[-1]["score"]
    span = max(smax - smin, 1e-9)
    bands = [[], [], []]      # low, mid, high
    for c in reps:
        b = min(2, int((c["score"] - smin) / span * 3))
        bands[b].append(c)

    names = ("low", "mid", "high")
    want = list(STRATA_PLAN)
    picks = []

    def draw_even(members, k):
        """k members spread evenly by score index (members are score-sorted)."""
        if k <= 0 or not members:
            return []
        if k >= len(members):
            return list(members)
        idx = np.linspace(0, len(members) - 1, k).round().astype(int)
        return [members[i] for i in sorted(set(idx.tolist()))]

    # First pass: honour the plan per band.
    taken = [set(), set(), set()]
    for bi in range(3):
        got = draw_even(bands[bi], want[bi])
        for c in got:
            c["stratum"] = names[bi]
            taken[bi].add(c["palette"])
        picks.extend(got)

    # Top-up: fill any deficit (short bands) from the remaining reps, farthest-by-score
    # from what's already picked, to keep the spread.
    deficit = PICKS_PER_LOC - len(picks)
    if deficit > 0:
        remaining = [c for bi in range(3) for c in bands[bi] if c["palette"] not in taken[bi]]
        remaining.sort(key=lambda c: c["score"])
        extra = draw_even(remaining, deficit)
        for c in extra:
            b = min(2, int((c["score"] - smin) / span * 3))
            c["stratum"] = names[b]
        picks.extend(extra)

    # Deterministic order + trim to exactly PICKS_PER_LOC (defensive).
    picks = sorted(picks, key=lambda c: c["score"])[:PICKS_PER_LOC]
    return picks


# ===========================================================================
# 4. Label-spec re-render — ensure_label_field + render_label_crop are the shared
#    Recipe-2 tail (tools/wallpaper/label_crop.py).
# ===========================================================================

# ===========================================================================
# 5. Batch emission.
# ===========================================================================

def render_block(loc, palette):
    """Version-invariant re-render spec (mirrors the location-corpus render block +
    fractal_type/c for non-mandelbrot families). Coloring params live in provenance."""
    blk = {
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
        "fractal_type": loc.family,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "palette": palette,
        "composition": "center",
        "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
        "filter": LABEL_FILTER, "interior_mode": "black",
    }
    for k, v in loc.params.items():
        blk[k] = v
    return blk


def provenance_block(spec, loc, role, source_row, pick):
    cfg = pick["config"]
    params = {
        "palette": cfg.palette, "palette_type": pick["palette_type"],
        "reverse": cfg.reverse, "log_premap": cfg.log_premap, "gamma": cfg.gamma,
        "phase": cfg.phase, "n_cycles": cfg.n_cycles,
        "interior_color": list(cfg.interior_color),
        "eval_filter": cfg.filter,   # the sampler's scoring filter (box); label crop is lanczos3
    }
    return {
        "generator_version": GENERATOR_VERSION,
        "batch_id": BATCH_ID,
        "lineage": "wallpaper_bootstrap",
        "source_class": spec[0],       # the 9-way class key (e.g. "julia:mandelbrot")
        "family": loc.family,          # render family (e.g. "julia") — for render_one_flags
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": cfg.palette,
        "params": params,
        "render_mode": "smooth",
        "beam_gen": pick["gen"],
        "beam_lineage": pick["lineage"],
        "pref_v2_score": pick["score"],
        "score_stratum": pick.get("stratum"),
        "decoded_class": source_row["decoded_class"],
        "source_role": role,
        "source_oid": source_row.get("id"),
    }


def write_batch(rows, source_report, wall_s, args):
    batch_dir = OUT_CORPUS / "batches" / BATCH_ID
    (batch_dir / "crops").mkdir(parents=True, exist_ok=True)
    with (batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "NEW wallpaper-quality namespace; reuses the location-corpus row "
                       "format. Coloring params (gamma/phase/reverse/n_cycles) live in "
                       "provenance.params — the crop is a pure function of render + "
                       "provenance.params (render.palette alone is insufficient here).",
        "source_run": "data/discovery/gather/<class>/outcome_ledger.jsonl (v5 decodes)",
        "sampling_metaparameters": {
            "beam": {"N_GEN0": SL.N_GEN0, "TOP_KEEP": SL.TOP_KEEP,
                     "K_VARIANTS": SL.K_VARIANTS, "R_MAX": SL.R_MAX, "seed": args.seed},
            "picks_per_loc": PICKS_PER_LOC, "strata_plan": list(STRATA_PLAN),
            "per_class_target": PER_CLASS_TARGET, "q2_per_class": Q2_PER_CLASS,
            "dedup_frac": DEDUP_FRAC,
            "scorer": "data/queries/scorer/v2 (within-location pref-v2 utility)",
            "maxiter_policy": "auto_maxiter(fw) — native fw-dependent",
        },
        "present_gates": {"note": "NONE applied — the batch must span the full quality "
                          "range including bad renders (no black/occupancy gate)."},
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(NOT enrich --mode render — see schema_note)",
        },
        "source_report": source_report,
        "classes": CLASS_KEYS,
        "n_rows": len(rows), "wall_seconds": wall_s,
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2))
    return batch_dir


# ===========================================================================
# Driver.
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Wallpaper-quality bootstrap batch generator.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="cap number of source locations (smoke)")
    ap.add_argument("--per-class", type=int, default=PER_CLASS_TARGET)
    ap.add_argument("--estimate", action="store_true", help="print source/runtime estimate and exit")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    sources, report = select_sources(args.seed, args.per_class)
    if args.limit:
        # Round-robin across classes so a --limit smoke still spans families.
        by_cls = {}
        for s in sources:
            by_cls.setdefault(s[0][0], []).append(s)
        inter = []
        while len(inter) < args.limit and any(by_cls.values()):
            for cls in CLASS_KEYS:
                if by_cls.get(cls):
                    inter.append(by_cls[cls].pop(0))
                    if len(inter) >= args.limit:
                        break
        sources = inter

    print(f"[wallpaper] sources selected: {len(sources)} across {len(CLASS_SPECS)} classes")
    for cls in CLASS_KEYS:
        rep = report[cls]
        print(f"    {cls:18} q3_core={rep['q3_core']} q2_marginal={rep['q2_marginal']}  "
              f"(avail q3={rep['q3_avail']} q2={rep['q2_avail']})")

    est_picks = len(sources) * PICKS_PER_LOC
    per_loc_recolors = SL.N_GEN0 + SL.R_MAX * SL.TOP_KEEP * SL.K_VARIANTS
    print(f"[wallpaper] est: {len(sources)} eval-field dumps (ss2) + "
          f"<= {len(sources)*per_loc_recolors} beam recolors + "
          f"{len(sources)} label-field dumps (ss2 2560x1440) + ~{est_picks} label recolors "
          f"-> ~{est_picks} crops")
    if args.estimate:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = SL.load_v2(device)
    print(f"[wallpaper] loaded pref-v2 scorer (epoch {epoch}) on {device.type}")

    rows = []
    strata_tally = {"low": 0, "mid": 0, "high": 0, "all": 0}
    class_tally = {cls: {"q3_core": 0, "q2_marginal": 0} for cls in CLASS_KEYS}
    t_wall = time.time()

    for li, (spec, role, srow) in enumerate(sources):
        cls = spec[0]
        loc = to_location(spec, srow)
        t_loc = time.time()
        # --- beam with full-trajectory retention ---
        # coarse_score=True: the beam's SCORING recolors run on the coarse grid
        # (colormap.render_candidate_coarse, ~13x faster; beam wall ~276min -> ~21min).
        # SELECTION-ONLY — it decides which 8 candidates the strata sampler picks, never
        # how they render. The keepers are re-rendered below at the full ss2 Lanczos-3
        # label spec (render_label_crop) from each pick's config, untouched by the coarse
        # path. Gated on STRATA-BUCKET parity, not winner parity (the batch is human-
        # relabeled, so exact palette/band identity is a scaffold): coarse picks span 87%
        # of each location's full-scorer quality range (validate_coarse_score.py).
        res = SL.run_location(f"{cls}_{li:03d}", loc, lib, sampler, model, device,
                              args.seed, retain_all=True, coarse_score=True)
        rng = np.random.default_rng(args.seed + li)
        picks = strata_sample(res["all_candidates"], rng)

        # --- label-spec re-render (ss2 field once, then each pick) ---
        # The 8 keeper crops are the bottleneck (several seconds single-threaded each — see
        # LABEL_CROP_WORKERS). Dump the ss2 field + percentile-stretch ONCE (both shared
        # read-only across the picks), then color the 8 concurrently. This phase, not the
        # beam, is why a location takes minutes.
        field = ensure_label_field(loc)
        label_prep = cm.stretch_field(field)   # 3.69M-value percentile sort: once per loc
        crops_dir = OUT_CORPUS / "batches" / BATCH_ID / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        def _render_pick(pi_pick):
            pi, pick = pi_pick
            image_id = f"wbv1_{li:03d}_{pi:02d}"
            w, h = render_label_crop(field, pick["config"], lib, crops_dir / f"{image_id}.jpg",
                                     prep=label_prep)
            return pi, image_id, w, h

        with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(picks))) as ex:
            rendered = list(ex.map(_render_pick, list(enumerate(picks))))  # order-preserving

        for pi, image_id, w, h in rendered:
            pick = picks[pi]
            assert (w, h) == (LABEL_W, LABEL_H), (image_id, w, h)
            rows.append({
                "image_id": image_id,
                "render": render_block(loc, pick["config"].palette),
                "provenance": provenance_block(spec, loc, role, srow, pick),
                "label": {"score": None, "labeler": None, "labeled_at": None},
            })
            strata_tally[pick.get("stratum", "all")] += 1
            class_tally[cls][role] += 1

        scores = [p["score"] for p in picks]
        print(f"[wallpaper] {cls}_{li:03d} [{role:11}] {loc.family:11} fw={loc.fw[:10]} "
              f"mi={loc.maxiter}  {len(picks)} picks  "
              f"pref-v2 [{min(scores):.3f},{max(scores):.3f}]  [{time.time()-t_loc:.0f}s]")

    wall = time.time() - t_wall
    batch_dir = write_batch(rows, report, wall, args)

    # --- report ---
    print("\n" + "=" * 74)
    print(f"WALLPAPER BOOTSTRAP BATCH — {BATCH_ID}")
    print("=" * 74)
    print(f"source locations: {len(sources)}   labelable renders: {len(rows)}   "
          f"({wall/60:.1f} min)")
    print("per-class realized (source locations / labelable renders):")
    for cls in CLASS_KEYS:
        ct = class_tally[cls]
        print(f"    {cls:18} locs q3={report[cls]['q3_core']} q2={report[cls]['q2_marginal']} "
              f"-> renders q3={ct['q3_core']} q2={ct['q2_marginal']} (tot {ct['q3_core']+ct['q2_marginal']})")
    print(f"strata spread (renders): low={strata_tally['low']} mid={strata_tally['mid']} "
          f"high={strata_tally['high']} all={strata_tally['all']}")
    print(f"-> {batch_dir}")
    print(f"   images.jsonl ({len(rows)} rows, all label.score=null) + crops/ + batch.json")


if __name__ == "__main__":
    main()
