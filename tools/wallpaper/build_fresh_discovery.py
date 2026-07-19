"""Fresh-discovery colorize front-end — pilot.

Point the existing emit_v1 back-end at UNSEEN machine-q3 locations for the first
HONEST emit precision. The v1 emission (tools/wallpaper/emit_v1.py) gates on
locations that were in v2's TRAIN set, so its p_ge3 is optimistic. This front-end
picks locations the wallpaper head never saw (not in bootstrap ∪ humanq3, train or
eval), generates their candidate pools with the same beam the humanq3 pool used, and
emits a batch in the identical images.jsonl+crops format so `emit_v1 --pool <batch>`
runs UNCHANGED — its gate-pass rate on this pool is the honest number.

Only new code here is the front-end (unseen-location selection + pool generation);
everything downstream (v2 gate, MAP-Elites selector, full-res emit) is emit_v1's.

Location set (unseen, machine-q3, ALL families):
  * Source = data/discovery/outcome_ledger.jsonl by default; override with --ledger to
    point at a single fresh-discovery run's ledger (the run-isolation precondition — the
    accumulated default sweeps in every historical q3). Any row decoded by a non-current
    classifier (gather/<class> ledgers are v5-decoded/unstamped; every v6-stamped row once
    the active checkpoint is v7) reads as not-current, so it never leaks in.
  * decoded_class == 3  ∧  guard_pass  ∧  scorer_version == <active version>
    (the live checkpoint's version, resolved from active_ckpt.ACTIVE_VERSION).
  * Families: ALL 9 (mandelbrot, multibrot3/4/5, julia:mandelbrot, julia:multibrot3/4/5,
    phoenix). Each ledger row maps to render coords via the canonical
    gather_select.render_family + outcome_geometry (Julia -> z-plane viewport at fixed
    c=outcome; c-plane -> outcome viewport). Phoenix has no v6 rows in this ledger yet.
  * EXCLUDE every location in the wallpaper head's corpus (bootstrap ∪ humanq3, all
    rows) by location_key AND by same-family spatial proximity (belt-and-suspenders —
    this is what keeps the precision honest). Dedup within the set by key.
  * Pilot count: small (default 20), family-balanced round-robin, seeded.

Pool generation (reuse the beam — byte-identical to build_humanq3):
  * Per location: SL.run_location(retain_all=True, coarse_score=True) → top-K by
    pref-v2 (build_humanq3.top_k_pool, K=7 — the SAME pool rule/format the humanq3
    emission pool used). One ss2 label-field dump per location, reused across the K.
  * Label crops via the shared Recipe-2 tail (label_crop.render_label_crop): the
    LOCKED 1280x720 ss2 Lanczos-3 q90 spec.

Emit (unchanged back-end):
    uv run python -u tools/wallpaper/build_fresh_discovery.py            # build the pool batch
    uv run python -u tools/wallpaper/emit_v1.py --pool \
        data/wallpaper_corpus/batches/2026-07-07_wallpaper_fresh_discovery_v1

    uv run python -u tools/wallpaper/build_fresh_discovery.py --estimate # composition + est, exit
    uv run python -u tools/wallpaper/build_fresh_discovery.py --limit 3  # smoke (3 locs)
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE))

import sample_location as SL          # noqa: E402  (run_location retain-all, load_v2)
import query_sampler as qs            # noqa: E402  (load_pool_library, PaletteSampler)
import colormap as cm                 # noqa: E402  (stretch_field)
import location as loc_mod            # noqa: E402  (canonical Location, from_render_block, location_key)
import corpus_common as cc            # noqa: E402  (is_current_decoded — current-stamp guard)
import gather_select as gs            # noqa: E402  (canonical ledger->render family map + outcome_geometry)
from active_ckpt import auto_maxiter        # noqa: E402  (native fw-dependent maxiter policy)
from label_crop import (              # noqa: E402  (shared label-crop spec — Recipe-2 tail)
    LABEL_W, LABEL_H, LABEL_SS, LABEL_FILTER,
    ensure_label_field, render_label_crop,
)
from pool_rule import top_k_pool  # noqa: E402  (shared pool rule; lifted out of build_humanq3)

DISCOVERY_LEDGER = ROOT / "data" / "discovery" / "outcome_ledger.jsonl"
WALLPAPER_CORPUS = ROOT / "data" / "wallpaper_corpus"
# The wallpaper head's corpus (bootstrap ∪ humanq3) — every location the head saw.
HEAD_CORPUS_BATCHES = [
    "2026-07-05_wallpaper_bootstrap_v1",
    "2026-07-05_wallpaper_humanq3_v1",
]

BATCH_ID = "2026-07-07_wallpaper_fresh_discovery_v1"
GENERATOR_VERSION = "wallpaper_fresh_discovery_v1"
IMG_PREFIX = "wfd"

# Output batch dir. Default = the committed wallpaper-corpus batch dir; overridden by
# --batch-dir so an orchestrator can write each cycle's pool to its own disposable dir
# (no image_id / crop collisions across cycles). None -> the default path is derived from
# BATCH_ID at use time (see _batch_dir).
BATCH_DIR = None
# Skip the first N ledger lines in source selection (the per-cycle watermark): an
# orchestrator sets this so a cycle pools ONLY the fresh q3s its discovery phase just
# appended, never rows from earlier cycles of the same run. 0 -> whole ledger.
LEDGER_START_LINE = 0

# --- selection knobs -------------------------------------------------------
# ALL families flow (no family filter): mandelbrot, multibrot3/4/5, julia:mandelbrot,
# julia:multibrot3/4/5, phoenix. The ledger->render mapping (family name + viewport +
# fixed c) is the canonical gather_select.render_family/outcome_geometry — see _render_coords.
PILOT_COUNT = 20            # default pilot size (config knob; scale later)
DEDUP_FRAC = 0.5           # two same-family coords within DEDUP_FRAC*fw = the "same spot"

# --- pool / label knobs (PINNED to the humanq3 emission pool) ---------------
K = 12                     # beam->selector handoff / emission pool depth (deployed default,
                           # widened 7->12, ship lever A); == build_humanq3.K. Override --pool-k.
LABEL_CROP_WORKERS = 4     # project-wide max-workers cap — DO NOT raise
SEED = 7


# ===========================================================================
# 1. Unseen machine-q3 source selection.
# ===========================================================================

def _batch_dir() -> Path:
    """The output batch dir: --batch-dir override, else the default committed corpus path."""
    return BATCH_DIR if BATCH_DIR is not None else (WALLPAPER_CORPUS / "batches" / BATCH_ID)


def _render_coords(row):
    """(cx, cy, fw, c, render_family) the ledger row will be RENDERED at — for ALL
    families. Reuses the canonical ledger->render mapping (gather_select.render_family +
    outcome_geometry, the same one gather_v6/v6-manifest render through): every Julia
    family (julia:mandelbrot, julia:multibrot3/4/5) renders the z-plane viewport
    (julia_z_*) at the fixed parameter c = the parent c-plane spot (outcome_*), render
    family julia / julia_multibrot{d}; c-plane families (mandelbrot, multibrot3/4/5,
    phoenix) render the outcome viewport directly with no c. Returns None if a Julia row
    lacks its z-plane viewport."""
    if gs.is_julia(row["family"]) and row.get("julia_z_cx") is None:
        return None
    cx, cy, fw, c_re, c_im = gs.outcome_geometry(row)
    c = None if c_re is None else (float(c_re), float(c_im))
    return (float(cx), float(cy), float(fw), c, gs.render_family(row["family"]))


def _to_location(row):
    """Ledger row -> canonical Location (bootstrap parity: repr() of the exact f64,
    native fw-dependent maxiter). Returns (family, Location) or None."""
    rc = _render_coords(row)
    if rc is None:
        return None
    cx, cy, fw, c, rfam = rc
    c_re = repr(c[0]) if c is not None else None
    c_im = repr(c[1]) if c is not None else None
    loc = loc_mod.Location(
        family=rfam, cx=repr(cx), cy=repr(cy), fw=repr(fw),
        maxiter=auto_maxiter(fw), c_re=c_re, c_im=c_im,
    )
    return row["family"], loc


def _head_corpus_exclusion():
    """(key_set, per_family_coords) over the wallpaper head's corpus. key_set = every
    distinct location_key; coords = {render_family: [(cx,cy,fw,c_re,c_im), ...]} for the
    proximity guard. c is carried because a julia location's identity is viewport AND the
    fixed parameter c — a near-viewport match with a DIFFERENT c is a different fractal."""
    keys = set()
    coords = defaultdict(list)
    for b in HEAD_CORPUS_BATCHES:
        p = WALLPAPER_CORPUS / "batches" / b / "images.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            loc = loc_mod.from_render_block(d["render"])
            keys.add(loc.key())
            try:
                coords[loc.family].append((float(loc.cx), float(loc.cy), float(loc.fw),
                                           loc.c_re, loc.c_im))
            except (TypeError, ValueError):
                pass
    return keys, coords


def _spatially_in(loc, corpus_coords):
    """True if `loc` is a near-duplicate of a corpus location: same render-family, SAME
    parameter c (identity for julia — ignoring c falsely collides different fractals that
    share a z-plane viewport), and viewport within DEDUP_FRAC*fw."""
    try:
        cx, cy, fw = float(loc.cx), float(loc.cy), float(loc.fw)
    except (TypeError, ValueError):
        return False
    for kcx, kcy, kfw, kc_re, kc_im in corpus_coords.get(loc.family, []):
        if kc_re != loc.c_re or kc_im != loc.c_im:
            continue
        tol = DEDUP_FRAC * min(fw, kfw)
        if abs(cx - kcx) < tol and abs(cy - kcy) < tol:
            return True
    return False


def select_sources(seed, count, head_exclude=True):
    """Unseen machine-q3 locations (all families), family-balanced round-robin to `count`.

    Filter: scorer_version==<current> ∧ decoded_class==3 ∧ guard_pass (ALL families).
    Exclude: any location in the head corpus (exact key OR same-family spatial proximity) —
      UNLESS `head_exclude=False` (the Phase-1 library loop, which pools every fresh q3 and
      dedups against the library STORE downstream, not against the wallpaper head's corpus).
    Dedup: within-set by key (a true coordinate-duplicate; counted as within_set_dups_dropped).
    Returns (sources, report) where each source is {loc, key, family, oid, p_good}."""
    excl_keys, excl_coords = _head_corpus_exclusion() if head_exclude else (set(), defaultdict(list))

    per_fam = defaultdict(list)   # ledger_family -> [source dict]
    seen = set()
    n_raw = 0
    n_unrenderable = 0
    n_excl_key = n_excl_spatial = n_dup = 0
    ledger_lines = DISCOVERY_LEDGER.read_text(encoding="utf-8").splitlines()[LEDGER_START_LINE:]
    for line in ledger_lines:
        if not line.strip():
            continue
        d = json.loads(line)
        if (not cc.is_current_decoded(d) or d.get("decoded_class") != 3
                or not d.get("guard_pass")):
            continue
        tl = _to_location(d)
        if tl is None:
            # current/class-3/guard-pass but not renderable (e.g. a julia row missing its z-plane
            # viewport). Counted so the Phase-1 reconciliation can see it as a real (non-dup)
            # drop rather than a silent leak.
            n_unrenderable += 1
            continue
        fam, loc = tl
        n_raw += 1
        k = loc.key()
        if k in seen:
            n_dup += 1
            continue
        if k in excl_keys:
            n_excl_key += 1
            continue
        if _spatially_in(loc, excl_coords):
            n_excl_spatial += 1
            continue
        seen.add(k)
        per_fam[fam].append({"loc": loc, "key": k, "family": fam,
                             "oid": d.get("id"), "p_good": d.get("p_good")})

    # Deterministic family-balanced round-robin to `count`.
    rng = np.random.default_rng(seed)
    for fam in per_fam:
        rng.shuffle(per_fam[fam])
    fam_cycle = sorted(per_fam)          # ('julia:mandelbrot', 'mandelbrot')
    chosen = []
    while len(chosen) < count and any(per_fam[f] for f in fam_cycle):
        for fam in fam_cycle:
            if per_fam[fam]:
                chosen.append(per_fam[fam].pop())
                if len(chosen) >= count:
                    break

    report = {
        "source_ledger": str(DISCOVERY_LEDGER.relative_to(ROOT)),
        "ledger_start_line": LEDGER_START_LINE,
        "filter": f"scorer_version=={cc.active_scorer_version()} & decoded_class==3 & guard_pass (all families)",
        "head_exclude": head_exclude,
        "raw_matches": n_raw,
        "unrenderable_dropped": n_unrenderable,
        "within_set_dups_dropped": n_dup,
        "excluded_head_corpus_by_key": n_excl_key,
        "excluded_head_corpus_by_proximity": n_excl_spatial,
        # per_fam still holds the un-chosen leftovers (round-robin pops), so total
        # unseen-available = chosen + leftovers.
        "unseen_available": len(chosen) + sum(len(v) for v in per_fam.values()),
        "pilot_count": len(chosen),
        "per_family_chosen": dict(Counter(s["family"] for s in chosen)),
        "head_corpus_batches": HEAD_CORPUS_BATCHES,
        "head_corpus_distinct_keys": len(excl_keys),
    }
    return chosen, report


# ===========================================================================
# 2. Batch emission (schema byte-parity with build_humanq3 / bootstrap).
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
        "lineage": "fresh_discovery_beam",
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
        "pref_v2_rank": pick["rank"],
        "pool_size": pick["pool_size"],
        "selection_role": "machine_q3",             # UNSEEN by the wallpaper head
        "source_generation": "v6",
        "source_ledger": str(DISCOVERY_LEDGER.relative_to(ROOT)),
        "source_oid": src["oid"],
        "seeder_decoded_class": 3,
        "seeder_p_good": src["p_good"],
    }


def write_batch(rows, report, wall_s, args):
    batch_dir = _batch_dir()
    (batch_dir / "crops").mkdir(parents=True, exist_ok=True)
    with (batch_dir / "images.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    batch = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "schema_note": "Fresh-discovery HONEST-emit pool: UNSEEN machine-q3 locations "
                       "(not in the wallpaper head's bootstrap ∪ humanq3 corpus), top-K "
                       "by pref-v2 — the SAME pool rule/format as the humanq3 emission "
                       "pool, so emit_v1 --pool runs unchanged. Coloring params live in "
                       "provenance.params (crop = pure function of render + params).",
        "source_run": f"{report['source_ledger']} (v6-scored production seeder; "
                      "decoded_class==3 & guard_pass; all families)",
        "sampling_metaparameters": {
            "beam": {"N_GEN0": SL.N_GEN0, "TOP_KEEP": SL.TOP_KEEP,
                     "K_VARIANTS": SL.K_VARIANTS, "R_MAX": SL.R_MAX, "seed": args.seed},
            "pool_K": args.pool_k, "pool_rule": "top-K by pref-v2 over best-per-palette reps "
                                      "(build_humanq3.top_k_pool — verbatim)",
            "scorer": "data/queries/scorer/v2 (within-location pref-v2 utility)",
            "maxiter_policy": "auto_maxiter(fw) — native fw-dependent (bootstrap parity)",
            "pilot_count": args.count, "dedup_frac": DEDUP_FRAC, "seed": args.seed,
        },
        "unseen_contract": {
            "excluded": "bootstrap ∪ humanq3 (all rows, train+eval)",
            "method": "location_key exact-match OR same-family spatial proximity "
                      f"(< {DEDUP_FRAC}*fw)",
            "note": "the head vouches for these locations blind — this is the honest "
                    "emit precision vs humanq3's optimistic ~0.60.",
        },
        "present_gates": {"note": "NONE at build time — emit_v1's v2 gate is the filter."},
        "render_defaults": {
            "width": LABEL_W, "height": LABEL_H, "ss": LABEL_SS,
            "filter": LABEL_FILTER, "interior_mode": "black", "composition": "center",
            "render_path": "render-one --dump-field + colormap.render_candidate "
                           "(byte-parity with the humanq3 label path)",
        },
        "selection_report": report,
        "n_rows": len(rows), "wall_seconds": wall_s,
    }
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2))
    # Standalone machine-readable selection report — the Phase-1 orchestrator reads this to
    # reconcile fresh-q3-found vs pooled (within_set_dups / unrenderable / head-exclusion counts).
    (batch_dir / "selection_report.json").write_text(json.dumps(report, indent=2))
    return batch_dir


# ===========================================================================
# Driver.
# ===========================================================================

def _print_composition(report, count):
    print("=" * 74)
    print(f"UNSEEN MACHINE-Q3 SELECTION — {BATCH_ID}")
    print("=" * 74)
    print(f"source: {report['source_ledger']}")
    print(f"filter: {report['filter']}")
    print(f"raw current/class-3/guard-pass matches (all fam): {report['raw_matches']:4}")
    print(f"  within-set key dups dropped           : {report['within_set_dups_dropped']:4}")
    print(f"  excluded (in head corpus, exact key)  : {report['excluded_head_corpus_by_key']:4}")
    print(f"  excluded (head corpus, spatial prox)  : {report['excluded_head_corpus_by_proximity']:4}")
    print(f"  head-corpus distinct location keys    : {report['head_corpus_distinct_keys']:4}")
    print(f"PILOT chosen (target {count})               : {report['pilot_count']:4}  "
          f"{report['per_family_chosen']}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Fresh-discovery colorize front-end (unseen machine-q3).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ledger", type=Path, default=None,
                    help="run-scoped outcome ledger to source locations from (default: the "
                         "accumulated data/discovery/outcome_ledger.jsonl). Point this at a "
                         "single fresh-discovery run's ledger to guarantee the emit pool "
                         "contains ONLY that run — the accumulated default sweeps in every "
                         "historical v6 q3 (jm3/4/5 revival, julia_recall, prior runs).")
    ap.add_argument("--count", type=int, default=PILOT_COUNT, help="pilot location count")
    ap.add_argument("--pool-all", action="store_true",
                    help="Phase-1 library loop: pool EVERY eligible fresh q3 (ignore --count). The "
                         "emit pipeline is volume-bound so a count cap is correct there; the "
                         "prospect/library loop is NOT, and a cap here would silently discard the "
                         "very q3s the loop exists to keep. Selection never decides which "
                         "locations survive — only a true coordinate-duplicate does (downstream).")
    ap.add_argument("--no-head-exclude", action="store_true",
                    help="Phase-1 library loop: do NOT exclude the wallpaper head's corpus. The "
                         "library dedups against its OWN store (library_annotate, coordinate-based) "
                         "— the head corpus is irrelevant to it. Emit keeps head exclusion (default).")
    ap.add_argument("--pool-k", type=int, default=K,
                    help="beam->selector handoff / emission pool depth (deployed default 12)")
    ap.add_argument("--limit", type=int, default=0, help="cap locations actually run (smoke)")
    ap.add_argument("--estimate", action="store_true", help="print composition + est and exit")
    ap.add_argument("--batch-dir", type=Path, default=None,
                    help="write the pool batch (images.jsonl + crops/ + batch.json) to this dir "
                         "instead of the committed corpus path. Use a disposable per-cycle dir so "
                         "an orchestrator's cycles never collide on image_ids/crops. BATCH_ID is "
                         "taken from the dir name.")
    ap.add_argument("--ledger-start-line", type=int, default=0,
                    help="skip the first N ledger lines in source selection (the per-cycle "
                         "watermark): pool ONLY the fresh q3s appended after this offset, so a "
                         "cycle never re-emits an earlier cycle's locations.")
    args = ap.parse_args()
    k = args.pool_k

    # Run-isolation hook: a run-scoped ledger overrides the accumulated default so the
    # emit pool is exactly one fresh-discovery run (the fresh-generation precondition).
    if args.ledger is not None:
        global DISCOVERY_LEDGER
        DISCOVERY_LEDGER = args.ledger.resolve()
    if args.batch_dir is not None:
        global BATCH_DIR, BATCH_ID
        BATCH_DIR = args.batch_dir.resolve()
        BATCH_ID = BATCH_DIR.name          # keep provenance batch_id == output dir name
    if args.ledger_start_line:
        global LEDGER_START_LINE
        LEDGER_START_LINE = int(args.ledger_start_line)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    # Phase-1 library loop pools EVERY eligible fresh q3 (--pool-all) and defers dedup to the
    # library store (--no-head-exclude); emit keeps the count cap + head exclusion.
    sel_count = 10**9 if args.pool_all else args.count
    sources, report = select_sources(args.seed, sel_count, head_exclude=not args.no_head_exclude)
    _print_composition(report, "ALL" if args.pool_all else sel_count)

    per_loc_recolors = SL.N_GEN0 + SL.R_MAX * SL.TOP_KEEP * SL.K_VARIANTS
    print(f"\n[fresh] est per-location: 1 eval-field dump (ss2) + <= {per_loc_recolors} beam "
          f"recolors (coarse) + 1 label-field dump (ss{LABEL_SS} "
          f"{LABEL_W*LABEL_SS}x{LABEL_H*LABEL_SS}) + <= {k} label recolors")
    print(f"[fresh] est total: {len(sources)} locations -> <= {len(sources)*k} crops  (pool-k={k})")

    if args.limit:
        sources = sources[:args.limit]
        print(f"[fresh] --limit {args.limit}: running {len(sources)} locations")
    if args.estimate:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = SL.load_v2(device)
    print(f"[fresh] loaded pref-v2 scorer (epoch {epoch}) on {device.type}")

    rows = []
    fam_tally = Counter()
    loc_times = []
    t_wall = time.time()

    for li, src in enumerate(sources):
        loc = src["loc"]
        cls = src["family"]
        t_loc = time.time()
        # Beam with full-trajectory retention; coarse scoring (SELECTION-ONLY — keepers
        # re-rendered below on the full ss2 label path). Byte-identical to build_humanq3.
        res = SL.run_location(f"{cls}_{li:03d}", loc, lib, sampler, model, device,
                              args.seed, retain_all=True, coarse_score=True)
        pool = top_k_pool(res["all_candidates"], k)

        field = ensure_label_field(loc)
        label_prep = cm.stretch_field(field)
        crops_dir = _batch_dir() / "crops"
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
        fam_tally[cls] += 1

        scores = [p["score"] for p in pool]
        dt_loc = time.time() - t_loc
        loc_times.append(dt_loc)
        print(f"[fresh] {li:03d}/{len(sources)} {cls:16} fw={loc.fw[:10]} mi={loc.maxiter}  "
              f"{len(pool)} picks  pref-v2 [{min(scores):.3f},{max(scores):.3f}]  [{dt_loc:.0f}s]")
        if len(loc_times) >= 2:
            lt = np.array(loc_times)
            print(f"[fresh]   wall/loc: last={dt_loc:.0f}s mean={lt.mean():.0f}s "
                  f"(n={len(lt)}) -> eta {lt.mean()*(len(sources)-li-1)/60:.1f} min")

    wall = time.time() - t_wall
    batch_dir = write_batch(rows, report, wall, args)

    print("\n" + "=" * 74)
    print(f"FRESH-DISCOVERY POOL BATCH — {BATCH_ID}")
    print("=" * 74)
    print(f"unseen locations: {len(sources)}   pool renders: {len(rows)}   ({wall/60:.1f} min)")
    print(f"per family (locations): {dict(fam_tally)}")
    print(f"-> {batch_dir}")
    print(f"   images.jsonl ({len(rows)} rows) + crops/ + batch.json")
    print(f"\n[next] uv run python -u tools/wallpaper/emit_v1.py --pool {batch_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
