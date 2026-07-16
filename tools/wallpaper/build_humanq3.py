"""Human-q3 wallpaper training set — the top-K pool-floored batch (primary payload).

Builds the PRIMARY training set for the next wallpaper-quality head: human-verified
good locations only (max-over-crops label == 3), each colored by the beam, top-K by
pref-v2 kept as the emission pool. Budget <=1000 renders. This is the batch that fixes
v1's top-end starvation (the bootstrap only spanned decoded-q3, never human-q3, and
strata-sampled the low end hard).

Difference from the bootstrap (tools/wallpaper/build_bootstrap.py):
  * Sources are HUMAN-q3 LOCATIONS read out of the labeled corpus batches (label.score
    max-over-crops == 3), NOT gather-ledger decoded-class rows.
  * Per location we keep the TOP-K by pref-v2 (one representative per palette, its
    best-scoring candidate; the refined winners land at rank-1 by construction) — the
    emission pool itself, no strata sub-sampling. Serving-consistency pin: K (deployed 12,
    widened from 7 — see the K constant) is also the emission pool depth, so the head
    trains on top-K and serves on top-K.
  * Everything else — the beam (retain_all + coarse_score), the ss2 label field dump,
    the render_candidate label-crop tail, the LOCKED 1280x720 ss2 Lanczos-3 q90 spec
    (shared via label_crop.py) — is byte-for-byte the bootstrap path, so the two batches
    union cleanly at train time.

Composition (see the prompt wallpaper_humanq3_trainset_k7.md):
  1. All human-q3 from gather_v6 (mandatory — carries the new-family coverage).
  2. Augment with human-q3 from the pre-gather_v6 labeled batches (deg-2 mandelbrot +
     julia:mandelbrot), up to the render cap.
  3. v6 is mandatory; fill the remaining budget with pre-v6 q3, sampled to fit if it
     overflows (family-balanced), dedup the whole union by location coords.

    uv run python tools/wallpaper/build_humanq3.py --estimate   # composition + est, exit
    uv run python tools/wallpaper/build_humanq3.py --limit 3     # smoke (3 locs)
    uv run python tools/wallpaper/build_humanq3.py               # full run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import sample_location as SL          # noqa: E402  (run_location retain-all, load_v2)
import query_sampler as qs            # noqa: E402  (load_pool_library, PaletteSampler)
import colormap as cm                 # noqa: E402  (load_field, stretch_field, render_candidate)
import location as loc_mod            # noqa: E402  (canonical Location + render_one_flags)
from active_ckpt import auto_maxiter        # noqa: E402  (native fw-dependent maxiter policy)
from label_crop import (              # noqa: E402  (shared label-crop spec — Recipe-2 tail)
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER, JPG_Q,
    ensure_label_field, render_label_crop,
)

LABEL_CORPUS = ROOT / "data" / "label_corpus"
OUT_CORPUS = ROOT / "data" / "wallpaper_corpus"

BATCH_ID = "2026-07-05_wallpaper_humanq3_v1"
GENERATOR_VERSION = "wallpaper_humanq3_v1"
IMG_PREFIX = "whq3"

# The gather_v6 batch (mandatory source); every other labeled batch is treated as pre-v6.
V6_BATCH = "2026-07-05_gather_v6"

# --- pool / budget knobs ---------------------------------------------------
# K = beam->selector HANDOFF depth == emission pool depth == serving pool depth. This is
# the SOLE knob on per-location survivor count into the selector (TOP_KEEP=18 upstream is
# the beam refinement width, independent). Deployed default widened 7 -> 12 (ship lever A,
# prompts/ship-widen-handoff-12.md): the pref TOP-7 cut concentrated the emission warm skew
# (aperture: warm-fraction 81%@top-7 -> 62%@top-12), and widening recovers neutral-cool
# cells at ~zero head-quality cost. Overridable via --pool-k. Kept in lockstep across the
# three wallpaper builders so train-on-K == serve-on-K (serving consistency).
K = 12                     # emission pool depth == serving pool depth (deployed default)
RENDER_BUDGET = 1000       # hard cap on total label crops

# The label-crop spec (LABEL_W/H/SS, LABEL_FILTER, JPG_Q) + ensure_label_field +
# render_label_crop are the shared canonical wallpaper label geometry (label_crop.py).
LABEL_CROP_WORKERS = 4     # project-wide max-workers cap — DO NOT raise (see build_bootstrap note)

SEED = 7


# ===========================================================================
# 1. Source-location selection from the LABELED corpus (human max-over-crops q3).
# ===========================================================================

def _batch_dirs():
    """Every labeled-corpus batch dir with an images.jsonl."""
    root = LABEL_CORPUS / "batches"
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "images.jsonl").exists())


def _q3_locations(batch_dir):
    """Human-q3 locations in one batch: group rows by canonical location key, take
    max-over-crops of the non-null label scores, keep those whose max == 3.

    Returns {location_key: {"render": <representative render block>, "family": str,
    "n_crops": int}} — one entry per distinct location. The render block is any crop's
    (all crops of a location share the viewport); it reconstructs the Location."""
    best_score = {}          # key -> max non-null score
    rep_render = {}          # key -> representative render block
    fam_of = {}              # key -> family label (provenance.family or fractal_type)
    n_crops = Counter()
    for line in (batch_dir / "images.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        loc = loc_mod.from_render_block(d["render"])
        k = loc.key()
        n_crops[k] += 1
        rep_render.setdefault(k, d["render"])
        fam_of.setdefault(k, (d.get("provenance", {}) or {}).get("family")
                          or d["render"].get("fractal_type") or "mandelbrot")
        sc = (d.get("label", {}) or {}).get("score")
        if sc is not None and (best_score.get(k) is None or sc > best_score[k]):
            best_score[k] = sc
    return {k: {"render": rep_render[k], "family": fam_of[k], "n_crops": n_crops[k]}
            for k, s in best_score.items() if s == 3}


def select_sources(seed):
    """Resolve the composition: v6 q3 (mandatory) + pre-v6 q3 (capped to budget),
    deduped across the union by location coords. Returns (sources, report) where each
    source is {loc, family, source_batch, source_generation, selection_role} and report
    carries the per-family / per-generation breakdown + cap accounting.

    Dedup: a location good in multiple batches counts once. v6 is processed first so a
    coord shared with a pre-v6 batch keeps the v6 stamp (mandatory side wins)."""
    rng = np.random.default_rng(seed)
    batch_dirs = {d.name: d for d in _batch_dirs()}

    # --- v6: mandatory, all of it ---
    v6_q3 = _q3_locations(batch_dirs[V6_BATCH]) if V6_BATCH in batch_dirs else {}
    seen = set(v6_q3.keys())

    # --- pre-v6: every other labeled batch, deduped within pre-v6 and against v6 ---
    prev6_q3 = {}            # key -> {..., source_batch}
    prev6_batch_avail = {}   # batch -> count contributed (pre-dedup, informational)
    for name in sorted(batch_dirs):
        if name == V6_BATCH:
            continue
        found = _q3_locations(batch_dirs[name])
        prev6_batch_avail[name] = len(found)
        for k, meta in found.items():
            if k in seen:
                continue
            seen.add(k)
            meta = dict(meta, source_batch=name)
            prev6_q3[k] = meta

    # --- cap: v6 mandatory; fill the remaining budget with pre-v6, sampled if over ---
    n_v6 = len(v6_q3)
    remaining_locs = max(0, (RENDER_BUDGET - K * n_v6) // K)
    prev6_keys = list(prev6_q3.keys())

    cap_applied = len(prev6_keys) > remaining_locs
    dropped = 0
    if cap_applied:
        # Family-balanced sample. pre-v6 q3 is deg-2 mandelbrot only in practice, so this
        # is a plain seeded draw; the stratification is defensive (keeps julia:* from
        # being dominated if a future pre-v6 batch carries any).
        by_fam = defaultdict(list)
        for k in prev6_keys:
            by_fam[prev6_q3[k]["family"]].append(k)
        for fam in by_fam:
            rng.shuffle(by_fam[fam])
        # Round-robin across families until the budget is filled.
        chosen = []
        fam_cycle = sorted(by_fam)
        while len(chosen) < remaining_locs and any(by_fam[f] for f in fam_cycle):
            for fam in fam_cycle:
                if by_fam[fam]:
                    chosen.append(by_fam[fam].pop())
                    if len(chosen) >= remaining_locs:
                        break
        dropped = len(prev6_keys) - len(chosen)
        prev6_keys = chosen

    # --- assemble ordered source list (v6 first, then the kept pre-v6) ---
    sources = []
    for k, meta in v6_q3.items():
        loc = _reconstruct(meta["render"])
        if loc is None:
            report_flag(k, "v6")
            continue
        sources.append({"loc": loc, "key": k, "family": meta["family"],
                        "source_batch": V6_BATCH, "source_generation": "v6",
                        "selection_role": "human_q3", "n_crops": meta["n_crops"]})
    for k in prev6_keys:
        meta = prev6_q3[k]
        loc = _reconstruct(meta["render"])
        if loc is None:
            report_flag(k, "pre-v6")
            continue
        sources.append({"loc": loc, "key": k, "family": meta["family"],
                        "source_batch": meta["source_batch"], "source_generation": "pre-v6",
                        "selection_role": "human_q3", "n_crops": meta["n_crops"]})

    report = {
        "budget": RENDER_BUDGET, "K": K,
        "v6_q3_locations": n_v6,
        "prev6_q3_raw_predup": sum(prev6_batch_avail.values()),
        "prev6_q3_after_dedup": len(prev6_q3),
        "prev6_q3_kept": len(prev6_keys),
        "prev6_dropped_to_cap": dropped,
        "cap_applied": cap_applied,
        "prev6_batch_available_predup": prev6_batch_avail,
        "total_locations": len(sources),
        "est_renders": len(sources) * K,
        "per_family": dict(Counter(s["family"] for s in sources)),
        "per_generation": dict(Counter(s["source_generation"] for s in sources)),
        "per_family_by_generation": {
            gen: dict(Counter(s["family"] for s in sources if s["source_generation"] == gen))
            for gen in ("v6", "pre-v6")
        },
    }
    return sources, report


_FLAGGED = []


def report_flag(key, gen):
    _FLAGGED.append((gen, key))
    print(f"[humanq3] WARN: could not reconstruct viewport for {gen} location {key!r} — skipped")


def _reconstruct(render):
    """Render block -> canonical Location with the NATIVE fw-dependent maxiter (bootstrap
    parity: both batches feed one head, so the label field must be dumped at the same
    maxiter policy, not the source batch's stored maxiter). Returns None on failure."""
    try:
        loc = loc_mod.from_render_block(render)
        return dataclasses.replace(loc, maxiter=auto_maxiter(float(render["fw"])))
    except Exception:
        return None


# ===========================================================================
# 2. Per-location pool: top-K by pref-v2 (best representative per palette).
#    The rule itself now lives in the neutral `pool_rule` module so the LIVE
#    build_fresh_discovery imports it from there, not from this batch builder.
# ===========================================================================

from pool_rule import top_k_pool  # noqa: E402,F401  (shared pool rule; re-exported for callers)


# ===========================================================================
# 3. Label-spec re-render — ensure_label_field + render_label_crop are the shared
#    Recipe-2 tail (tools/wallpaper/label_crop.py); the bootstrap uses the same path.
# ===========================================================================

# ===========================================================================
# 4. Batch emission.
# ===========================================================================

def render_block(loc, palette):
    """Version-invariant re-render spec (identical shape to the bootstrap / location
    corpus). Coloring params live in provenance.params."""
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


def provenance_block(src, loc, pick):
    cfg = pick["config"]
    params = {
        "palette": cfg.palette, "palette_type": pick["palette_type"],
        "palette_source": pick.get("palette_source"),  # pool `source` bucket (dramatic/curated_*/extracted); absent-reads-None on old records
        "reverse": cfg.reverse, "log_premap": cfg.log_premap, "gamma": cfg.gamma,
        "phase": cfg.phase, "n_cycles": cfg.n_cycles,
        "transfer": cfg.transfer, "transfer_gamma": cfg.transfer_gamma,
        "interior_color": list(cfg.interior_color),
        "eval_filter": cfg.filter,   # sampler scoring filter (box); the crop is lanczos3
    }
    return {
        "generator_version": GENERATOR_VERSION,
        "batch_id": BATCH_ID,
        "lineage": "humanq3_beam",
        "family": loc.family,
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": cfg.palette,
        "params": params,
        "render_mode": "smooth",
        "beam_gen": pick["gen"],
        "beam_lineage": pick["lineage"],
        "pref_v2_score": pick["score"],
        "pref_v2_rank": pick["rank"],           # 1 == highest pref-v2 in the pool
        "pool_size": pick["pool_size"],
        "selection_role": src["selection_role"],
        "source_generation": src["source_generation"],   # v6 / pre-v6
        "source_batch": src["source_batch"],              # human-q3 source stamp
    }


def write_batch(rows, report, wall_s, args):
    batch_dir = OUT_CORPUS / "batches" / BATCH_ID
    (batch_dir / "crops").mkdir(parents=True, exist_ok=True)
    with (batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "Primary wallpaper-quality training payload: human-q3 locations, "
                       "top-7 by pref-v2 emission pool per location. Coloring params "
                       "(gamma/phase/reverse/n_cycles) live in provenance.params — the "
                       "crop is a pure function of render + provenance.params.",
        "source_run": "data/label_corpus/batches/<batch>/images.jsonl (human max-over-crops "
                      "label.score == 3)",
        "sampling_metaparameters": {
            "beam": {"N_GEN0": SL.N_GEN0, "TOP_KEEP": SL.TOP_KEEP,
                     "K_VARIANTS": SL.K_VARIANTS, "R_MAX": SL.R_MAX, "seed": args.seed},
            "pool_K": K, "pool_rule": "top-K by pref-v2 over best-per-palette reps "
                                      "(distinct palettes; refined winners at rank-1)",
            "scorer": "data/queries/scorer/v2 (within-location pref-v2 utility)",
            "maxiter_policy": "auto_maxiter(fw) — native fw-dependent (bootstrap parity)",
            "render_budget": RENDER_BUDGET, "seed": args.seed,
        },
        "present_gates": {"note": "NONE applied — human-q3 locations, in-distribution "
                          "top-7 pool (some picks are human-bad hard negatives)."},
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(byte-parity with the bootstrap label path)",
        },
        "composition_report": report,
        "reconstruct_failures": [{"generation": g, "key": k} for g, k in _FLAGGED],
        "n_rows": len(rows), "wall_seconds": wall_s,
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2))
    return batch_dir


# ===========================================================================
# Driver.
# ===========================================================================

def _print_composition(report):
    print("=" * 74)
    print(f"COMPOSITION — {BATCH_ID}  (budget {report['budget']} renders @ K={report['K']})")
    print("=" * 74)
    print(f"v6 (gather_v6) human-q3 locations : {report['v6_q3_locations']:4}  (mandatory)")
    print(f"pre-v6 human-q3 raw (sum of batches): {report['prev6_q3_raw_predup']:4}")
    print(f"pre-v6 after cross-union dedup      : {report['prev6_q3_after_dedup']:4}"
          f"  ({report['prev6_q3_raw_predup'] - report['prev6_q3_after_dedup']} coord collisions removed)")
    print(f"pre-v6 kept (after budget cap)      : {report['prev6_q3_kept']:4}"
          + (f"  (dropped {report['prev6_dropped_to_cap']} to cap)" if report['cap_applied'] else "  (all fit)"))
    print(f"TOTAL locations                     : {report['total_locations']:4}"
          f"  -> {report['est_renders']} renders (<= {report['budget']})")
    print("\nper source-generation:")
    for gen in ("v6", "pre-v6"):
        fam = report["per_family_by_generation"].get(gen, {})
        tot = report["per_generation"].get(gen, 0)
        print(f"    {gen:7} {tot:4} locs  {fam}")
    print("\nper family (union):")
    for fam, n in sorted(report["per_family"].items(), key=lambda kv: -kv[1]):
        print(f"    {fam:20} {n:4}")
    print("\npre-v6 q3 available per batch (pre-dedup):")
    for b, n in sorted(report["prev6_batch_available_predup"].items()):
        print(f"    {b:50} {n:4}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Human-q3 wallpaper training-set generator (K=12).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pool-k", type=int, default=K,
                    help="beam->selector handoff / emission pool depth (deployed default 12)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of source locations (smoke)")
    ap.add_argument("--estimate", action="store_true", help="print composition + est and exit")
    args = ap.parse_args()
    k = args.pool_k

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    sources, report = select_sources(args.seed)
    _print_composition(report)

    per_loc_recolors = SL.N_GEN0 + SL.R_MAX * SL.TOP_KEEP * SL.K_VARIANTS
    print(f"\n[humanq3] est per-location: 1 eval-field dump (ss2) + <= {per_loc_recolors} beam "
          f"recolors (coarse) + 1 label-field dump (ss{LABEL_SS} "
          f"{LABEL_W*LABEL_SS}x{LABEL_H*LABEL_SS}) + <= {k} label recolors")
    print(f"[humanq3] est total: {len(sources)} locations -> <= {len(sources)*k} crops  (pool-k={k})")

    if args.limit:
        sources = sources[:args.limit]
        print(f"[humanq3] --limit {args.limit}: running {len(sources)} locations")
    if args.estimate:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = SL.load_v2(device)
    print(f"[humanq3] loaded pref-v2 scorer (epoch {epoch}) on {device.type}")

    rows = []
    gen_tally = Counter()
    fam_tally = Counter()
    range_spans = []
    loc_times = []
    t_wall = time.time()

    for li, src in enumerate(sources):
        loc = src["loc"]
        cls = src["family"]
        t_loc = time.time()
        # Beam with full-trajectory retention; coarse scoring (SELECTION-ONLY — keepers
        # are re-rendered below on the full ss2 label path, untouched by the coarse grid).
        res = SL.run_location(f"{cls}_{li:03d}", loc, lib, sampler, model, device,
                              args.seed, retain_all=True, coarse_score=True)
        pool = top_k_pool(res["all_candidates"], k)

        # Label-spec re-render: ss2 field once, percentile-stretch once, then the K picks
        # concurrently (heavy single-threaded numpy per crop; cap 4).
        field = ensure_label_field(loc)
        label_prep = cm.stretch_field(field)
        crops_dir = OUT_CORPUS / "batches" / BATCH_ID / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        def _render_pick(pi_pick):
            pi, pick = pi_pick
            image_id = f"{IMG_PREFIX}_{li:03d}_{pi:02d}"
            w, h = render_label_crop(field, pick["config"], lib, crops_dir / f"{image_id}.jpg",
                                     prep=label_prep)
            return pi, image_id, w, h

        with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(pool))) as ex:
            rendered = list(ex.map(_render_pick, list(enumerate(pool))))

        for pi, image_id, w, h in rendered:
            pick = pool[pi]
            assert (w, h) == (LABEL_W, LABEL_H), (image_id, w, h)
            rows.append({
                "image_id": image_id,
                "render": render_block(loc, pick["config"].palette),
                "provenance": provenance_block(src, loc, pick),
                "label": {"score": None, "labeler": None, "labeled_at": None},
            })
        gen_tally[src["source_generation"]] += 1
        fam_tally[cls] += 1

        scores = [p["score"] for p in pool]
        span = max(scores) - min(scores)
        range_spans.append(span)
        dt_loc = time.time() - t_loc
        loc_times.append(dt_loc)
        print(f"[humanq3] {li:03d}/{len(sources)} [{src['source_generation']:6}] {cls:16} "
              f"fw={loc.fw[:10]} mi={loc.maxiter}  {len(pool)} picks  "
              f"pref-v2 [{min(scores):.3f},{max(scores):.3f}] span={span:.3f}  [{dt_loc:.0f}s]")
        # Running mean, printed once we have >=2 locations so the ss2 speedup shows early.
        if len(loc_times) >= 2:
            lt = np.array(loc_times)
            print(f"[humanq3]   wall/loc: last={dt_loc:.0f}s  mean={lt.mean():.0f}s  "
                  f"(n={len(lt)}, min={lt.min():.0f}s max={lt.max():.0f}s)")

    wall = time.time() - t_wall
    batch_dir = write_batch(rows, report, wall, args)

    print("\n" + "=" * 74)
    print(f"HUMAN-Q3 WALLPAPER BATCH — {BATCH_ID}")
    print("=" * 74)
    print(f"source locations: {len(sources)}   labelable renders: {len(rows)}   ({wall/60:.1f} min)")
    print(f"per generation (locations): {dict(gen_tally)}")
    print(f"per family (locations):     {dict(fam_tally)}")
    if range_spans:
        rs = np.array(range_spans)
        print(f"per-location pref-v2 pool span: min={rs.min():.3f} median={np.median(rs):.3f} "
              f"max={rs.max():.3f}  (confirms a real within-pool gradient)")
    if _FLAGGED:
        print(f"reconstruct failures (skipped): {len(_FLAGGED)}")
    print(f"-> {batch_dir}")
    print(f"   images.jsonl ({len(rows)} rows, all label.score=null) + crops/ + batch.json")


if __name__ == "__main__":
    main()
