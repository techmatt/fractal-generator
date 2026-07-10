"""Dramatic-inclusive wallpaper-quality head batch (~1000 renders).

Builds the next label-ready batch for the WALLPAPER-QUALITY HEAD (the absolute
1/2/3/4 CORN model on finished renders — NOT the location classifier, NOT the pref
scorer), curated by the now-deployed pref scorer (`data.ACTIVE_SCORER_DIR` ==
pref-v3-gvo). Unions with the existing head corpus (bootstrap 504 + humanq3 994).

This is a FUSION of the two existing build front-ends, byte-parity on everything
downstream so the batch unions cleanly at train time:
  * build_humanq3.py  — verified-good (human max-over-crops q3) LOCATION reuse, the
    beam + top-K-by-pref pool (`top_k_pool`, K deployed 12), the shared Recipe-2 label tail.
  * build_fresh_discovery.py — UNSEEN machine-q3 deg-2 location selection out of the
    discovery ledgers (`_to_location`, `_head_corpus_exclusion`, `_spatially_in`).

What THIS batch adds (see prompts/prompt_head_batch_build.md):
  1. Scorer = the DEPLOYED pref scorer via the single-source pointer (SL.load_v2 reads
     data.ACTIVE_SCORER_DIR — currently pref-v3-gvo). The 75%-dramatic gen-0 is the
     deployed beam's `GEN0_SOURCE_WEIGHTS`, unchanged.
  2. Locations = REUSE the verified-good humanq3 set PRESERVING its existing v2-head
     split assignment (every eval-side verified-good location included, to grow the
     eval-good positive pool past the ~64 that blinds good-AP), + FRESH machine-q3 for
     volume / deployment distribution-match. Aim ~60% reuse / ~40% fresh.
  3. ~5% deliberately-BAD dramatic renders: bottom-K dramatic colorings on a subset of
     verified-good (train-side) locations — provably coloring-caused tier-1s, so the head
     cannot learn "dramatic -> good".
  4. Split is STAMPED per render (split_side/split_origin + coord) so the retrain honors
     the preserved humanq3 assignments and the fresh location-disjoint assignment without
     a global re-split.
  5. Provenance stamps palette_source (dramatic/pool) + curation_bucket (topk/bad_inject)
     so the retrain eval can stratify the dramatic tier distribution and isolate the
     injected negatives.

Execution is atomic-per-location with a durable append ledger (resumable) and a
wall-clock cap checked at the per-location boundary.

    uv run python -u tools/wallpaper/build_headbatch_dramatic.py --estimate  # composition + est, exit
    uv run python -u tools/wallpaper/build_headbatch_dramatic.py --limit 3    # smoke (3 units)
    uv run python -u tools/wallpaper/build_headbatch_dramatic.py              # full run (background)
"""
from __future__ import annotations

import argparse
import json
import re
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
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))
sys.path.insert(0, str(HERE))

import sample_location as SL          # noqa: E402  (run_location retain-all, load_v2, ACTIVE scorer)
import query_sampler as qs            # noqa: E402  (load_pool_library, PaletteSampler)
import colormap as cm                 # noqa: E402  (stretch_field)
import location as loc_mod            # noqa: E402  (canonical Location, from_render_block, key)
from probe import auto_maxiter        # noqa: E402  (native fw-dependent maxiter policy)
from label_crop import (              # noqa: E402  (shared label-crop spec — Recipe-2 tail)
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER,
    ensure_label_field, render_label_crop,
)
from build_humanq3 import top_k_pool  # noqa: E402  (humanq3 emission-pool rule — REUSED verbatim)
import build_fresh_discovery as BFD   # noqa: E402  (_to_location, _head_corpus_exclusion, _spatially_in)
import corpus_common as cc            # noqa: E402  (is_v6_decoded — v6-stamp guard)

WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"
LABELS_DIR = ROOT / "labels"

# The head corpus the split preservation + fresh exclusion are defined against.
HEAD_CORPUS_BATCHES = [
    "2026-07-05_wallpaper_bootstrap_v1",
    "2026-07-05_wallpaper_humanq3_v1",
]
HUMANQ3_BATCH = "2026-07-05_wallpaper_humanq3_v1"
HUMANQ3_PREFIX_RE = re.compile(r"(whq3_\d+)_\d+$")
HUMANQ3_LABELS = LABELS_DIR / "wallpaper_humanq3_v1.json"
BOOTSTRAP_BATCH = "2026-07-05_wallpaper_bootstrap_v1"
BOOTSTRAP_PREFIX_RE = re.compile(r"(wbv1_\d+)_\d+$")
BOOTSTRAP_LABELS = LABELS_DIR / "wallpaper_bootstrap_v1.json"

BATCH_ID = "2026-07-09_wallpaper_headbatch_dramatic_v1"
GENERATOR_VERSION = "wallpaper_headbatch_dramatic_v1"
IMG_PREFIX = "whd"

# --- composition knobs -----------------------------------------------------
K = 12                      # beam->selector handoff / emission pool depth == humanq3 serving
                            # depth (deployed default, widened 7->12 ship lever A). --pool-k.
K_BAD = 4                   # bottom-K dramatic per bad-inject location
N_BADINJECT_LOCS = 12       # verified-good (train-side) locations to also carry bad-inject
N_TRAIN_REUSE_FILL = 41     # train-side humanq3 to reuse ON TOP of ALL eval-side (mandatory)
N_FRESH = 54                # fresh machine-q3 locations (deg-2, unseen)
FRESH_EVAL_FRAC = 0.30      # location-disjoint eval fraction for the fresh (new) locations
SPLIT_SEED = 0              # v2-head split seed (split_rows) — PIN to reproduce old eval slice
EVAL_FRAC = 0.30            # v2-head eval_frac — PIN

# --- fresh source (deg-2 machine-q3) ---------------------------------------
# Union the v6 standing ledger with the deg-2 gather ledger; both carry decoded_class +
# guard_pass. The prompt's ~629 estimate predates dedup/exclusion; the honest available
# count is reported at run time (flagged if it differs).
FRESH_LEDGERS = [
    ROOT / "data" / "discovery" / "outcome_ledger.jsonl",
    ROOT / "data" / "discovery" / "gather" / "mandelbrot" / "outcome_ledger.jsonl",
]
DEG2_FAMILIES = BFD.DEG2_FAMILIES   # ("mandelbrot", "julia:mandelbrot")

LABEL_CROP_WORKERS = 4      # project-wide max-workers cap — DO NOT raise
SEED = 7                    # beam seed (gen-0 draws + refinement) — build_humanq3 parity

# --- wall discipline -------------------------------------------------------
DEFAULT_WALL_CAP_MIN = 150  # ~136 units * ~43s/loc ~= 100 min; 2.5h cap w/ headroom
HARD_KILL_MIN = 210         # backstop: abort the loop no matter what past this


# ===========================================================================
# 1a. v2-head split reproduction (which humanq3 locations are eval-side).
#     Mirrors classifier/train_wallpaper_v2.split_rows exactly (seed 0, eval_frac 0.30):
#     bootstrap->train, eval humanq3-only, location-disjoint, max-tier-stratified,
#     forced-train = bootstrap-coord collisions + single-location families.
# ===========================================================================

def _load_batch_rows(batch, prefix_re, labels_path, batch_name):
    labels = json.loads(labels_path.read_text())
    out = []
    p = WALLPAPER_CORPUS / "batches" / batch / "images.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["image_id"]
        m = prefix_re.match(iid)
        rd = r["render"]
        coord = (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"])
        out.append({"iid": iid, "label": int(labels[iid]), "loc": m.group(1),
                    "batch": batch_name, "family": r["provenance"]["family"],
                    "coord": coord, "render": rd})
    return out


def reproduce_v2_split():
    """Return (eval_locs, forced_train, humanq3_table) reproducing the v2 head split.

    humanq3_table: {whq3_loc_key: {"render": rep_render_block, "family": str,
    "loc_max": int, "coord": tuple, "split_side": "eval"|"train"}}."""
    boot = _load_batch_rows(BOOTSTRAP_BATCH, BOOTSTRAP_PREFIX_RE, BOOTSTRAP_LABELS, "bootstrap")
    hq3 = _load_batch_rows(HUMANQ3_BATCH, HUMANQ3_PREFIX_RE, HUMANQ3_LABELS, "humanq3")

    boot_coords = {r["coord"] for r in boot}
    by_loc = defaultdict(list)
    for r in hq3:
        by_loc[r["loc"]].append(r)
    loc_family = {l: rs[0]["family"] for l, rs in by_loc.items()}
    loc_max = {l: max(r["label"] for r in rs) for l, rs in by_loc.items()}
    single_fam = {f for f, n in Counter(loc_family.values()).items() if n == 1}

    forced_train = set()
    for l, rs in by_loc.items():
        if any(r["coord"] in boot_coords for r in rs):
            forced_train.add(l)
        elif loc_family[l] in single_fam:
            forced_train.add(l)

    rng = np.random.RandomState(SPLIT_SEED)
    eval_locs = set()
    eligible = {l for l in by_loc if l not in forced_train}
    for tier in sorted({loc_max[l] for l in eligible}):
        locs = sorted(l for l in eligible if loc_max[l] == tier)
        rng.shuffle(locs)
        n_eval = int(round(EVAL_FRAC * len(locs)))
        eval_locs.update(locs[:n_eval])

    table = {}
    for l, rs in by_loc.items():
        table[l] = {"render": rs[0]["render"], "family": loc_family[l], "loc_max": loc_max[l],
                    "coord": rs[0]["coord"], "split_side": "eval" if l in eval_locs else "train"}
    return eval_locs, forced_train, table


# ===========================================================================
# 1b. Reused-humanq3 source selection (preserve split; all eval-side mandatory).
# ===========================================================================

def _reconstruct(render):
    """Render block -> canonical Location at the NATIVE fw-dependent maxiter (bootstrap /
    humanq3 parity — the label field is dumped at the same maxiter policy, not the source
    batch's stored maxiter). Returns None on failure."""
    import dataclasses
    try:
        loc = loc_mod.from_render_block(render)
        return dataclasses.replace(loc, maxiter=auto_maxiter(float(render["fw"])))
    except Exception:
        return None


def select_reused(seed):
    """All eval-side humanq3 (mandatory) + N_TRAIN_REUSE_FILL seeded train-side. Marks
    N_BADINJECT_LOCS of the reused train-side loc_max>=3 locations for bad-injection."""
    eval_locs, forced_train, table = reproduce_v2_split()
    rng = np.random.default_rng(seed)

    eval_keys = sorted(eval_locs)
    train_keys = sorted(k for k, v in table.items() if v["split_side"] == "train")

    # Fill train-side reuse (seeded draw over all train-side humanq3).
    rng.shuffle(train_keys)
    train_fill = train_keys[:N_TRAIN_REUSE_FILL]

    # Bad-inject locations: from the reused train-side fill, loc_max>=3 (certainly good),
    # deterministically spread. Bad-inject is taught in TRAIN (keep eval renders top-K only).
    bad_pool = sorted(k for k in train_fill if table[k]["loc_max"] >= 3)
    rng.shuffle(bad_pool)
    bad_inject = set(bad_pool[:N_BADINJECT_LOCS])

    sources = []
    for k in eval_keys + train_fill:
        v = table[k]
        loc = _reconstruct(v["render"])
        if loc is None:
            print(f"[headbatch] WARN reused reconstruct failed for {k} — skipped")
            continue
        sources.append({
            "origin": "reused_humanq3", "unit_key": f"reused:{k}",
            "loc": loc, "family": loc.family, "source_family": v["family"],
            "source_loc": k, "loc_max": v["loc_max"],
            "split_side": v["split_side"], "split_origin": "preserved_humanq3",
            "coord": v["coord"], "is_bad_inject": k in bad_inject,
            "oid": None, "p_good": None, "source_ledger": None,
        })
    report = {
        "humanq3_total_locations": len(table),
        "forced_train_side": len(forced_train),
        "eval_side_available": len(eval_locs),
        "eval_side_reused": len(eval_keys),   # ALL of them (mandatory)
        "train_side_available": len(train_keys),
        "train_side_reused": len(train_fill),
        "bad_inject_locations": sorted(bad_inject),
    }
    return sources, report


# ===========================================================================
# 1c. Fresh machine-q3 source selection (deg-2, unseen; location-disjoint split).
# ===========================================================================

def select_fresh(seed, count, reused_sources):
    """Unseen deg-2 machine-q3 locations over FRESH_LEDGERS, disjoint from the head corpus
    (exact key OR same-family spatial proximity) AND from the reused set, family-balanced
    round-robin to `count`, then a seeded location-disjoint eval split (FRESH_EVAL_FRAC)."""
    excl_keys, excl_coords = BFD._head_corpus_exclusion()
    reused_keys = {s["loc"].key() for s in reused_sources}

    per_fam = defaultdict(list)
    seen = set()
    n_raw = n_dup = n_excl_key = n_excl_spatial = n_excl_v5 = 0
    for ledger in FRESH_LEDGERS:
        if not ledger.exists():
            continue
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            # v6-stamp guard: `decoded_class` from a v5-decoded (unstamped) row is
            # NOT a v6 machine-q3 verdict — reject it here, matching the sibling
            # build_fresh_discovery emit path. The gather/mandelbrot partition is
            # 100% v5, so without this the "fresh machine-q3" pool is dominated by
            # v5 verdicts masquerading as v6.
            if not cc.is_v6_decoded(d):
                if (d.get("decoded_class") == 3 and d.get("guard_pass")
                        and d.get("family") in DEG2_FAMILIES):
                    n_excl_v5 += 1
                continue
            if (d.get("decoded_class") != 3 or not d.get("guard_pass")
                    or d.get("family") not in DEG2_FAMILIES):
                continue
            tl = BFD._to_location(d)
            if tl is None:
                continue
            fam, loc = tl
            n_raw += 1
            k = loc.key()
            if k in seen or k in reused_keys:
                n_dup += 1
                continue
            if k in excl_keys:
                n_excl_key += 1
                continue
            if BFD._spatially_in(loc, excl_coords):
                n_excl_spatial += 1
                continue
            seen.add(k)
            per_fam[fam].append({"loc": loc, "key": k, "family": fam,
                                 "oid": d.get("id"), "p_good": d.get("p_good"),
                                 "source_ledger": str(ledger.relative_to(ROOT))})

    rng = np.random.default_rng(seed)
    for fam in per_fam:
        rng.shuffle(per_fam[fam])
    fam_cycle = sorted(per_fam)
    chosen = []
    while len(chosen) < count and any(per_fam[f] for f in fam_cycle):
        for fam in fam_cycle:
            if per_fam[fam]:
                chosen.append(per_fam[fam].pop())
                if len(chosen) >= count:
                    break

    # Location-disjoint eval split over the fresh (each fresh location is new & distinct).
    split_rng = np.random.default_rng(seed + 1)
    idx = list(range(len(chosen)))
    split_rng.shuffle(idx)
    n_eval = int(round(FRESH_EVAL_FRAC * len(chosen)))
    eval_idx = set(idx[:n_eval])

    sources = []
    for i, c in enumerate(chosen):
        loc = c["loc"]
        sources.append({
            "origin": "fresh_machineq3", "unit_key": f"fresh:{c['key']}",
            "loc": loc, "family": loc.family, "source_family": c["family"],
            "source_loc": None, "loc_max": None,
            "split_side": "eval" if i in eval_idx else "train",
            "split_origin": "fresh_assigned",
            "coord": (loc.cx, loc.cy, loc.fw, loc.family), "is_bad_inject": False,
            "oid": c["oid"], "p_good": c["p_good"], "source_ledger": c["source_ledger"],
        })
    report = {
        "ledgers": [str(l.relative_to(ROOT)) for l in FRESH_LEDGERS],
        "filter": "scorer_version==v6 & decoded_class==3 & guard_pass & family∈{mandelbrot,julia:mandelbrot}",
        "raw_matches": n_raw, "within_set_or_reused_dups": n_dup,
        "excluded_v5_decoded_q3": n_excl_v5,
        "excluded_head_corpus_by_key": n_excl_key,
        "excluded_head_corpus_by_proximity": n_excl_spatial,
        "unseen_available": len(chosen) + sum(len(v) for v in per_fam.values()),
        "chosen": len(chosen), "per_family_chosen": dict(Counter(s["source_family"] for s in sources)),
        "fresh_eval_frac": FRESH_EVAL_FRAC, "fresh_eval_locations": n_eval,
    }
    return sources, report


# ===========================================================================
# 2. Bad-injection pool: bottom-K dramatic colorings (distinct palettes).
# ===========================================================================

def bottom_k_dramatic(all_candidates, k, exclude_palettes):
    """The WORST dramatic colorings on a known-good location: best-per-palette among the
    dramatic-source candidates, ascending by pref score, take the lowest `k` whose palette
    is not already in the location's top-K pool. Distinct palettes; each gains bad_rank
    (1 == worst) and pool_size."""
    best = {}
    for c in all_candidates:
        if c.get("palette_source") != "dramatic":
            continue
        cur = best.get(c["palette"])
        if cur is None or c["score"] > cur["score"]:
            best[c["palette"]] = c
    cands = [c for pal, c in best.items() if pal not in exclude_palettes]
    cands.sort(key=lambda c: c["score"])   # ascending — worst first
    picks = cands[:k]
    for rank, c in enumerate(picks, start=1):
        c["bad_rank"] = rank
        c["pool_size"] = len(picks)
    return picks


# ===========================================================================
# 3. Batch emission (schema byte-parity with humanq3 / fresh_discovery).
# ===========================================================================

def render_block(loc, palette):
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


def provenance_block(src, loc, pick, curation_bucket):
    cfg = pick["config"]
    raw_source = pick.get("palette_source")
    params = {
        "palette": cfg.palette, "palette_type": pick["palette_type"],
        "palette_source": raw_source,   # fine-grained pool bucket (dramatic/curated_*/extracted)
        "reverse": cfg.reverse, "log_premap": cfg.log_premap, "gamma": cfg.gamma,
        "phase": cfg.phase, "n_cycles": cfg.n_cycles,
        "transfer": cfg.transfer, "transfer_gamma": cfg.transfer_gamma,
        "interior_color": list(cfg.interior_color),
        "eval_filter": cfg.filter,
    }
    return {
        "generator_version": GENERATOR_VERSION,
        "batch_id": BATCH_ID,
        "lineage": "headbatch_dramatic_beam",
        "family": loc.family,
        "cx": loc.cx, "cy": loc.cy, "fw": loc.fw,
        "c_re": loc.c_re, "c_im": loc.c_im,
        "p_re": loc.params.get("p_re"), "p_im": loc.params.get("p_im"),
        "palette": cfg.palette,
        "params": params,
        "render_mode": "smooth",
        "beam_gen": pick["gen"],
        "beam_lineage": pick["lineage"],
        # coarse palette bucket the prompt asks for + curation bucket
        "palette_source": ("dramatic" if raw_source == "dramatic" else "pool"),
        "curation_bucket": curation_bucket,   # "topk" | "bad_inject"
        # deployed pref scorer (single-source pointer)
        "pref_score": pick["score"],
        "pref_rank": pick.get("rank"),        # 1==best (topk); None for bad_inject
        "bad_rank": pick.get("bad_rank"),     # 1==worst (bad_inject); None for topk
        "pool_size": pick["pool_size"],
        "scorer_version": SL.V2_DIR.name,     # e.g. "v3_gvo"
        # location provenance + split
        "location_origin": src["origin"],     # reused_humanq3 | fresh_machineq3
        "source_family": src["source_family"],
        "source_loc": src["source_loc"],      # the whq3_NNN key for reused (coord-join)
        "loc_max_humanq3": src["loc_max"],
        "split_side": src["split_side"],       # eval | train  (STAMPED — retrain honors)
        "split_origin": src["split_origin"],   # preserved_humanq3 | fresh_assigned
        "source_oid": src["oid"],
        "source_ledger": src["source_ledger"],
        "source_p_good": src["p_good"],
    }


def emit_row(image_id, loc, pick, src, curation_bucket):
    return {
        "image_id": image_id,
        "render": render_block(loc, pick["config"].palette),
        "provenance": provenance_block(src, loc, pick, curation_bucket),
        "label": {"score": None, "labeler": None, "labeled_at": None},
    }


# ===========================================================================
# 4. Durable ledger (resume) + batch assembly.
# ===========================================================================

def batch_dir():
    return WALLPAPER_CORPUS / "batches" / BATCH_ID


def _ledger_path():
    return batch_dir() / "_progress_ledger.jsonl"


def load_ledger():
    """{unit_key: {"unit_index": int, "rows": [...]}} of completed units."""
    p = _ledger_path()
    done = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[rec["unit_key"]] = rec
    return done


def append_ledger(rec):
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def write_batch(all_rows, reused_report, fresh_report, split_summary, wall_s, args,
                completed_units, planned_units, capped):
    bd = batch_dir()
    (bd / "crops").mkdir(parents=True, exist_ok=True)
    with (bd / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "Dramatic-inclusive wallpaper-quality head batch. Verified-good "
                       "(humanq3) location reuse preserving the v2-head split (all eval-side "
                       "included to grow eval-good) + fresh unseen deg-2 machine-q3. Top-K by "
                       "the DEPLOYED pref scorer (data.ACTIVE_SCORER_DIR) + ~5% bottom-K "
                       "dramatic hard-negatives on known-good locations. Coloring params live "
                       "in provenance.params (crop = pure function of render + params). "
                       "provenance.split_side/split_origin are STAMPED for the retrain to "
                       "honor (preserved humanq3 assignments + fresh location-disjoint).",
        "scorer": {"active_scorer_dir": str(SL.V2_DIR.relative_to(ROOT)),
                   "scorer_version": SL.V2_DIR.name,
                   "note": "resolved from data.ACTIVE_SCORER_DIR (pref-v3-gvo at build time)"},
        "sampling_metaparameters": {
            "beam": {"N_GEN0": SL.N_GEN0, "TOP_KEEP": SL.TOP_KEEP,
                     "K_VARIANTS": SL.K_VARIANTS, "R_MAX": SL.R_MAX, "seed": args.seed},
            "gen0_source_weights": SL.GEN0_SOURCE_WEIGHTS,   # 75%-dramatic deployed beam
            "pool_K": K, "k_bad": K_BAD, "n_badinject_locs": N_BADINJECT_LOCS,
            "pool_rule": "top-K by pref (best-per-palette reps); bad_inject = bottom-K "
                         "dramatic (distinct palettes, excl. top-K palettes)",
            "maxiter_policy": "auto_maxiter(fw) — native fw-dependent (bootstrap parity)",
            "split_seed": SPLIT_SEED, "eval_frac": EVAL_FRAC,
            "fresh_eval_frac": FRESH_EVAL_FRAC, "seed": args.seed,
        },
        "split_summary": split_summary,
        "reused_report": reused_report,
        "fresh_report": fresh_report,
        "head_corpus_batches": HEAD_CORPUS_BATCHES,
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(byte-parity with the humanq3 label path)",
        },
        "run_status": {"planned_units": planned_units, "completed_units": completed_units,
                       "capped": capped, "wall_seconds": wall_s},
        "n_rows": len(all_rows),
    }
    (bd / "batch.json").write_text(json.dumps(batch, indent=2))
    return bd


# ===========================================================================
# Driver.
# ===========================================================================

def _print_composition(reused_report, fresh_report, planned, args):
    print("=" * 78)
    print(f"COMPOSITION — {BATCH_ID}  (deployed scorer: {SL.V2_DIR.name})")
    print("=" * 78)
    rr, fr = reused_report, fresh_report
    print("REUSED verified-good (humanq3), split PRESERVED:")
    print(f"  humanq3 total locations        : {rr['humanq3_total_locations']:4}  "
          f"(forced-train {rr['forced_train_side']})")
    print(f"  eval-side available -> reused  : {rr['eval_side_available']:4} -> "
          f"{rr['eval_side_reused']:4}  (ALL — mandatory, grows eval-good)")
    print(f"  train-side available -> reused : {rr['train_side_available']:4} -> "
          f"{rr['train_side_reused']:4}")
    print(f"  bad-inject locations (train)   : {len(rr['bad_inject_locations']):4}  "
          f"({K_BAD} bottom-K dramatic each)")
    print("FRESH unseen deg-2 machine-q3:")
    print(f"  ledgers                        : {fr['ledgers']}")
    print(f"  raw class-3 guard-pass deg-2   : {fr['raw_matches']:4}")
    print(f"  unseen available (post-excl)   : {fr['unseen_available']:4}")
    print(f"  chosen                         : {fr['chosen']:4}  {fr['per_family_chosen']}")
    print(f"  fresh eval-side (disjoint)     : {fr['fresh_eval_locations']:4}  "
          f"(frac {fr['fresh_eval_frac']})")
    if fr["unseen_available"] < 200:
        print(f"  NOTE: unseen_available={fr['unseen_available']} — the prompt's ~629 estimate "
              "predates dedup/head-corpus exclusion; supply still ample for the draw.")
    n_reuse = rr["eval_side_reused"] + rr["train_side_reused"]
    n_units = n_reuse + fr["chosen"]
    est_topk = n_units * K
    est_bad = len(rr["bad_inject_locations"]) * K_BAD
    print("-" * 78)
    print(f"UNITS (distinct locations)       : {n_units}  "
          f"(reuse {n_reuse} = {100*n_reuse/max(n_units,1):.0f}% / fresh {fr['chosen']} "
          f"= {100*fr['chosen']/max(n_units,1):.0f}%)")
    print(f"RENDERS  top-K {est_topk} + bad-inject {est_bad} = {est_topk + est_bad}")
    # eval-side render pool (the measurability read)
    print(f"eval-side TOP-K renders (eval-good candidate pool): "
          f"{(rr['eval_side_reused'] + fresh_report['fresh_eval_locations']) * K}"
          f"  (verified-good eval {rr['eval_side_reused']*K} + fresh eval "
          f"{fresh_report['fresh_eval_locations']*K})")
    print("=" * 78)


def main():
    global K                     # rebind the module constant so the shared report/eval-sizing
                                 # sites (est_topk, eval-side, metadata, top_k_pool) all follow
    ap = argparse.ArgumentParser(description="Dramatic-inclusive wallpaper head batch (~1000).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pool-k", type=int, default=K,
                    help="beam->selector handoff / emission pool depth (deployed default 12)")
    ap.add_argument("--limit", type=int, default=0, help="cap units actually run (smoke)")
    ap.add_argument("--estimate", action="store_true", help="print composition + est and exit")
    ap.add_argument("--wall-cap-min", type=float, default=DEFAULT_WALL_CAP_MIN,
                    help="stop starting new units past this wall-clock budget (min)")
    args = ap.parse_args()
    K = args.pool_k

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    reused, reused_report = select_reused(args.seed)
    fresh, fresh_report = select_fresh(args.seed, N_FRESH, reused)

    # Deterministic unit order: reused (eval then train) first, then fresh. Assign a
    # STABLE unit_index up-front so resume/image_ids don't depend on what's already done.
    planned = reused + fresh
    for i, s in enumerate(planned):
        s["unit_index"] = i

    _print_composition(reused_report, fresh_report, planned, args)

    per_loc_recolors = SL.N_GEN0 + SL.R_MAX * SL.TOP_KEEP * SL.K_VARIANTS
    print(f"\n[headbatch] est per-unit: 1 eval-field dump (ss2) + <= {per_loc_recolors} beam "
          f"recolors (coarse) + 1 label-field dump (ss{LABEL_SS}) + <= {K}(+{K_BAD}) label recolors")

    if args.limit:
        planned = planned[:args.limit]
        print(f"[headbatch] --limit {args.limit}: running {len(planned)} units")
    if args.estimate:
        return

    split_summary = {
        "eval_side_reused_locations": reused_report["eval_side_reused"],
        "train_side_reused_locations": reused_report["train_side_reused"],
        "fresh_eval_locations": fresh_report["fresh_eval_locations"],
        "fresh_train_locations": fresh_report["chosen"] - fresh_report["fresh_eval_locations"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = SL.load_v2(device)
    print(f"[headbatch] loaded DEPLOYED pref scorer '{SL.V2_DIR.name}' (epoch {epoch}) on {device.type}")

    done = load_ledger()
    if done:
        print(f"[headbatch] RESUME: {len(done)} units already in ledger -> skipping")

    all_rows = []
    for rec in done.values():          # carry forward completed units (stable order below)
        all_rows.extend(rec["rows"])
    loc_times = []
    capped = False
    t_wall = time.time()
    hard_kill_s = HARD_KILL_MIN * 60
    cap_s = args.wall_cap_min * 60

    to_run = [s for s in planned if s["unit_key"] not in done]
    print(f"[headbatch] {len(to_run)}/{len(planned)} units to run "
          f"(wall cap {args.wall_cap_min:.0f} min, hard-kill {HARD_KILL_MIN} min)")

    for si, src in enumerate(to_run):
        elapsed = time.time() - t_wall
        # Wall discipline: don't START a unit that can't finish in budget.
        est_unit = (np.mean(loc_times) if loc_times else 60.0)
        if elapsed + est_unit > cap_s or elapsed > hard_kill_s:
            capped = True
            print(f"[headbatch] WALL CAP hit ({elapsed/60:.1f} min, est next {est_unit:.0f}s) "
                  f"— stopping at {si}/{len(to_run)} of this run's queue")
            break

        loc = src["loc"]
        cls = src["source_family"]
        ui = src["unit_index"]
        t_loc = time.time()
        res = SL.run_location(f"{cls}_{ui:03d}", loc, lib, sampler, model, device,
                              args.seed, retain_all=True, coarse_score=True)
        pool = top_k_pool(res["all_candidates"], K)

        picks = [(p, "topk") for p in pool]
        if src["is_bad_inject"]:
            bad = bottom_k_dramatic(res["all_candidates"], K_BAD,
                                    exclude_palettes={p["palette"] for p in pool})
            picks += [(p, "bad_inject") for p in bad]

        field = ensure_label_field(loc)
        label_prep = cm.stretch_field(field)
        crops_dir = batch_dir() / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        def _render(idx_pick):
            pi, (pick, bucket) = idx_pick
            image_id = f"{IMG_PREFIX}_{ui:03d}_{pi:02d}"
            w, h = render_label_crop(field, pick["config"], lib,
                                     crops_dir / f"{image_id}.jpg", prep=label_prep)
            return pi, image_id, w, h, bucket

        with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(picks))) as ex:
            rendered = list(ex.map(_render, list(enumerate(picks))))

        unit_rows = []
        for pi, image_id, w, h, bucket in rendered:
            pick, _ = picks[pi]
            assert (w, h) == (LABEL_W, LABEL_H), (image_id, w, h)
            unit_rows.append(emit_row(image_id, loc, pick, src, bucket))

        append_ledger({"unit_key": src["unit_key"], "unit_index": ui, "rows": unit_rows})
        all_rows.extend(unit_rows)

        dt = time.time() - t_loc
        loc_times.append(dt)
        n_bad = sum(1 for _, b in picks if b == "bad_inject")
        scores = [p["score"] for p in pool]
        print(f"[headbatch] {si+1:03d}/{len(to_run)} [{src['origin'][:6]}|{src['split_side'][:5]}] "
              f"{cls:16} fw={loc.fw[:10]} mi={loc.maxiter}  {len(pool)} topk"
              f"{f'+{n_bad}bad' if n_bad else ''}  pref[{min(scores):.3f},{max(scores):.3f}]  [{dt:.0f}s]")
        if len(loc_times) >= 2:
            lt = np.array(loc_times)
            remaining = len(to_run) - si - 1
            print(f"[headbatch]   wall/unit mean={lt.mean():.0f}s  eta {lt.mean()*remaining/60:.1f} min  "
                  f"(elapsed {(time.time()-t_wall)/60:.1f} min)")

    wall = time.time() - t_wall
    completed = len(load_ledger())
    bd = write_batch(all_rows, reused_report, fresh_report, split_summary, wall, args,
                     completed, len(planned), capped)

    # Report tallies from the assembled rows.
    by_bucket = Counter(r["provenance"]["curation_bucket"] for r in all_rows)
    by_side = Counter(r["provenance"]["split_side"] for r in all_rows)
    by_origin = Counter(r["provenance"]["location_origin"] for r in all_rows)
    eval_topk = sum(1 for r in all_rows if r["provenance"]["split_side"] == "eval"
                    and r["provenance"]["curation_bucket"] == "topk")

    print("\n" + "=" * 78)
    print(f"DRAMATIC-INCLUSIVE HEAD BATCH — {BATCH_ID}")
    print("=" * 78)
    print(f"units completed: {completed}/{len(planned)}"
          f"{'  (CAPPED — rerun to resume)' if capped else ''}   "
          f"labelable renders: {len(all_rows)}   ({wall/60:.1f} min this run)")
    print(f"by curation bucket : {dict(by_bucket)}")
    print(f"by split side      : {dict(by_side)}")
    print(f"by location origin : {dict(by_origin)}")
    print(f"eval-side TOP-K renders (the eval-good candidate pool): {eval_topk}")
    print(f"field dumps ~= distinct locations completed = {completed} "
          f"(1 label-field dump per unit; reused humanq3 fields hit the out/wallpaper_fields cache)")
    print(f"-> {bd}")
    print(f"   images.jsonl ({len(all_rows)} rows, all label.score=null) + crops/ + batch.json")
    print(f"\n[label] open tools/viz/wallpaper_label.html, load {bd.relative_to(ROOT)}/images.jsonl")
    print(f"[merge] uv run python tools/corpus/merge_scores.py  (null->value only; see its --help)")


if __name__ == "__main__":
    main()
