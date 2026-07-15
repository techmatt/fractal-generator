#!/usr/bin/env python
r"""Atlas production discovery seeder — the standing discovery flow, per family.

Native-only discovery: the guided-descend engine's own root draw proposes depth-1
seeds; coverage is controlled by REJECTION SAMPLING over a point cloud of distinct
q3 outcomes (there is no atlas, no coverage grid, no per-cell cap). Per batch:

Family grammar (`--family` / `--julia --c` / `--phoenix`; see `resolve_family`): the
c-plane families (mandelbrot + multibrot3/4/5) run the full loop — the engine carries
the per-degree bands + degree_bbox flat-box, so only `--family` threads through. The q3
cloud is PARTITIONED by family (ledger `family` key): a d3 outcome and a Mandelbrot seed
at the same (cx, cy) are different parameter planes, so the radius query + startup
rebuild filter to the active partition. Julia is plumbed (fixed anchor, borrowed bands),
Phoenix is deliberately rejected (no parameter plane to prospect).

  draw native depth-1 seed  (engine root draw, ~96% descendability pre-gated)
    -> REJECT if >= Q3_DENSITY_CAP distinct q3 outcomes lie within REJECT_RADIUS
       of the seed's (cx, cy); else accept  (test is FREE next to one descent)
    -> depth-2 descendability probe  (reuse prescreen.prescreen, verbatim engine step-1)
    -> full guided-descend walks      (--seed-list --per-walk-rng, production config)
    -> k3 best-frame reward + outcome center + 1280-D v5 penultimate feature
    -> CORN-decode the k3-winning frame; a guard-passing class-3 outcome that is not a
       1.5*max(fw) near-dup of an existing cloud member joins the q3 cloud
    -> append every scored outcome to the durable ledger (distinct + dup + guarded)

Why rejection sampling: the descent (~seconds-minutes) dwarfs a point-cloud radius
query (microseconds), so burning many rejected seed draws to find one admissible seed
costs nothing next to one descent. The test runs BEFORE the descent, so a saturated
region's seeds are rejected before we pay to descend there. We test the *seed*
position against the *outcome* cloud; descent drift in (cx, cy) is bounded and
REJECT_RADIUS is set generously to absorb it. MAX_SEED_REDRAWS consecutive rejections
declares global saturation and stops the run cleanly.

"q3" = the v5 CORN hard-class decode == 3 (score_lib.corn_decode on the k3-winning
reframed frame), NOT a cutoff on the summed k3 = E[ord] scalar (two frames with equal
E[ord] can decode differently). k3 stays the reward/ranking value; class-3 is the
admission gate. There is no q3 cutoff knob.

This is v1 production wiring: NO harvest->refit loop, NO atlas refit. The Mandelbrot
(cx,cy,fw) distance is the ONLY harvest gate — the 1280-D feature is logged, never gates.

Reuse (located, not reinvented):
  * guided-descend engine (Rust) w/ --seed-list --per-walk-rng      (prescreen.BIN)
  * depth-2 descendability pre-screen                  prescreen.prescreen (verbatim)
  * reframe path                                       tools/reframe/reframe.py
  * v5 scorer bridge (explicit model_path)             probe.make_scorer / score_lib.Scorer
  * canonical v5 CORN hard-class decode                score_lib.corn_decode
  * k3 reward primitives                               step0_reanalysis
  * v5 1280-D penultimate hook                         prescreen.embed_paths / _render
  * degenerate-outcome guard                           tools/atlas/guard.py
  * canonical location layer                           tools/corpus/location.py

  uv run python tools/atlas/production_seeder.py --smoke     # ~20 seeds, rejection forced to fire
  uv run python tools/atlas/production_seeder.py --run       # 30-min time-boxed production run
  uv run python tools/atlas/production_seeder.py --time-only # project per-batch wallclock
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# reuse roots (sibling tool packages)
sys.path.insert(0, str(HERE))                                   # prescreen.py, guard.py
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))        # step0_reanalysis primitives
sys.path.insert(0, str(ROOT / "tools" / "reframe"))            # reframe_location
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))     # probe.make_scorer
sys.path.insert(0, str(ROOT / "tools" / "corpus"))            # location.py
sys.path.insert(0, str(ROOT / "tools" / "mining"))            # score_lib.Scorer + corn_decode

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import prescreen  # noqa: E402  (BIN, prescreen, write_seed_list, SCREEN_*, _render, embed_paths, RENDER_*)
import step0_reanalysis as sr  # noqa: E402  (KRAW, raw_screen_walk, _mand_location, load_frames_by_walk)
from step0_reanalysis import (  # noqa: E402
    KRAW, raw_screen_walk, _mand_location, load_frames_by_walk,
)
import reframe  # noqa: E402  (reframe_location + the DUMP_GUARD_FIELD hook)
from reframe import reframe_location  # noqa: E402
import guard  # noqa: E402  (degenerate-outcome guard: make_guarded_scorer + the field gate)
from score_lib import corn_decode  # noqa: E402  (canonical v5 CORN hard-class decode)
from probe import make_scorer as make_raw_scorer, ACTIVE_CKPT  # noqa: E402  (UNGUARDED raw — gather mode; ACTIVE_CKPT = single-source live checkpoint)

# =========================================================================== #
# Config (top-of-file constants — the experiment knobs)
# =========================================================================== #
# --- coverage control: rejection sampling over the q3 outcome cloud ---
REJECT_RADIUS = 0.20     # Euclidean distance in (cx, cy) parameter space; the primary knob.
                         # Set generously large on purpose (force spread; absorb descent drift).
Q3_DENSITY_CAP = 5       # reject a seed if >= this many distinct q3 outcomes lie within REJECT_RADIUS
MAX_SEED_REDRAWS = 200   # consecutive rejections before declaring global saturation and stopping.
DEDUP_K = 1.5            # cloud hygiene: near-dup iff plane dist < DEDUP_K * max(A.fw, B.fw)
                         # (one point per distinct q3 place; NOT a cell-saturation cap).

# --- production walk config (matches the atlas theta-walk config so the reward stays comparable) ---
NODE_WIDTH = 384             # foci-finder low-pass; 384 is outcome-value-preserving (efficiency study)
SIGMA_BAND = "8,10,12,14,16" # x0.5 (dilation=sigma, isolation=2sigma)
DEPTH_MIN = 4                # engine default; production walks descend deep
DEPTH_MAX = 14               # value ~= depth-17 at lower cost; the reward config was fit at 14
OCC_FLOOR = 0.321
BLACK_CAP = 0.30
PER_WALK_RNG = True          # ON for all walk + probe runs

# --- probe / scorer ---
PROBE_DEPTH = 2              # --depth-min 2 --depth-max 2, keep reached>=2
SCORER_PATH = ACTIVE_CKPT   # single source of truth (probe.ACTIVE_CKPT — currently v6); explicit, NEVER a default scorer
# Provenance stamp for new outcome rows: the classifier version dir ("v6") parsed off the
# active checkpoint path, so it tracks ACTIVE_CKPT automatically. Stamped by both ledger
# writers (append_outcome + GatherLedger.append) so post-deploy rows are distinguishable
# from v5-era rows in the ledger and the q3 cloud.
SCORER_VERSION = Path(SCORER_PATH).parent.name   # "v6"

# --- run ---
WALLCLOCK_BUDGET_MIN = 30
BATCH_SEEDS = 24             # accepted proposals per batch (amortizes engine startup)
NATIVE_POOL_WALKS = 400      # native depth-1 walks generated per refill (seed source)
WORKERS = 4              # multiprocessing worker cap (project rule: max 4)

# --- Julia sub-descent hook (--julia-hook; strictly additive, default off) ---
JULIA_WALKS_PER_DESCENT = 3  # native --julia walks per qualifying parent outcome (tunable)

# --- durable store (committed via .gitignore negation, like the atlas) ---
DISCOVERY_DIR = ROOT / "data" / "discovery"
OUTCOME_LEDGER = DISCOVERY_DIR / "outcome_ledger.jsonl"
OUTCOME_FEATS = DISCOVERY_DIR / "outcome_feats.npz"
PROBE_REJECTS = DISCOVERY_DIR / "probe_rejects.jsonl"
RUNS_DIR = DISCOVERY_DIR / "runs"
# disposable render scratch (never data/): native run, probe pools, walk pools, reward tiles
SCRATCH_ROOT = ROOT / "out" / "atlas" / "production_seeder"
# contact-sheet review PNGs are disposable render VIEWS -> out/ (a sibling of the per-run
# scratch dirs, so _purge_run_scratch never touches it), NEVER data/. data/ is the
# never-delete tier reserved for the unregenerable (ledgers + feats); a render view there
# erodes that distinction. The sheet is a pure function of the durable ledger + tiles.
SHEETS_DIR = SCRATCH_ROOT / "sheets"

# --- Gather mode (--gather): the guard-OFF oversampling harvest for the v6 label pass.
# A SEPARATE durable subtree per class, so guard-off rows NEVER pollute the production
# q3 cloud: production `build_cloud` requires guard_pass, and gather rows are raw-scored
# (guard-pass always true), so mixing them into the production ledger would silently
# admit degenerate decoded-class-3 outcomes into live discovery. Each class run appends
# its own ledger + writes its own walks.jsonl; nothing is cleaned up between classes. ---
GATHER_DIR = DISCOVERY_DIR / "gather"
GATHER_SCRATCH_ROOT = ROOT / "out" / "atlas" / "gather"

# Loosened, degree-aware Julia DESCENT bands (from the Julia-band assessment) — applied
# to JULIA descents ONLY (the --julia-hook sub-descents + any standalone --julia gather),
# NEVER the parameter plane (c-plane keeps its calibrated per-degree defaults). Keyed by
# the c-plane family the Julia descent hangs off (degree = that family's z^d degree).
# Values sit below the assessment's per-degree real-set medians / lower tails, so genuine
# filled and dendritic Julia sets pass while true far-exterior instant-escape is still
# excluded. (esc_median_min, spread_min).
#
# NOTE: these were validated by the overnight gather yield and PROMOTED into the engine as
# the permanent Julia-specific defaults — `WalkFamily::julia_band_defaults` in
# src/guided_descend.rs. Every `--julia` descent now resolves to exactly these WITHOUT any
# CLI override, so the explicit `--esc-median-min`/`--spread-min` pass below is redundant;
# it is kept as a self-documenting explicit override (gather logs its bands in the summary).
# **Keep this table bit-identical to `julia_band_defaults`** — if you retune one, retune both.
JULIA_GATHER_BANDS = {
    "mandelbrot": (3.0, 14.0),   # d2: esc 3.0 (unchanged), flat 20 -> 14
    "multibrot3": (2.0, 10.0),   # d3: esc 3.0 -> 2.0,        flat 15 -> 10
    "multibrot4": (2.0, 13.0),   # d4: esc 3.0 -> 2.0,        flat 16 -> 13
    "multibrot5": (1.8, 13.0),   # d5: esc 3.0 -> 1.8,        flat 17 -> 13
}

NCOL_DUP = 6   # cap the near-dup / non-q3 strip on the contact sheet


# =========================================================================== #
# Family grammar (mirror src/guided_descend.rs mutual-exclusion / requirement rules).
# The engine carries the per-degree bands + degree_bbox flat-box internally
# (WalkFamily::{root8k_band_defaults, flat_spread_min_default, flat_box_default}), so
# the seeder only passes `--family multibrot{d}` — band/box selection is downstream and
# correct-by-construction. `partition` is the ledger `family` discriminator that keys the
# q3 outcome cloud: a d3 outcome and a Mandelbrot seed at the SAME (cx, cy) live in
# different parameter planes, so both the radius-rejection query and the startup rebuild
# filter to the active family's partition (else rejection is meaningless).
# =========================================================================== #
from collections import namedtuple

# Known-good Julia z-plane anchor (mirrors cross_family_shakeout.JULIA_C). `c` is fixed
# for a standalone --julia run; the partition is degree-only (`julia:{fam}`, see
# julia_partition) and `c` is the cloud coordinate, not part of the key.
JULIA_C = ("-0.07810228973371881", "-0.6514609012382414")

_MULTIBROT = ("multibrot3", "multibrot4", "multibrot5")


def julia_partition(fam: str) -> str:
    """Degree-only Julia cloud partition key: `julia:{fam}` (fam is the c-plane family
    the descent hangs off — mandelbrot / multibrot{d}). The parameter `c` is the cloud
    COORDINATE (outcome_cx/cy), NOT part of the key, so the partition matches the
    multibrot partitions structurally and every Julia found-point in the plane repels
    future `c` regardless of its z-plane spot. Defined once; reused by resolve_family
    (standalone --julia runs) and the julia-hook found-point commit."""
    return f"julia:{fam}"


# =========================================================================== #
# Per-partition q3 operating point (t_good), keyed on the ledger `family` partition.
# A partition earns a non-baseline threshold only from its OWN labeled sweep; anything
# unswept falls through to the conservative baseline (0.50). The table is the single
# source of truth — adding a family after its sweep is a one-line entry, not a branch.
# Every value is stamped per outcome row (`t_good`) so the ledger self-describes across
# the mixed-threshold eras.
#
# Provenance of each override:
#   mandelbrot / julia:mandelbrot -> 0.24
#     v6 threshold sweep (tools/v6/threshold_sweep.py, labeled eval split): the deg-2
#     slice — where the classifier is actually POWERED — has a knee at p_good=0.24,
#     ~2.5x the baseline q3 recall at equal precision. deg-2 = the c-plane quadratic
#     Mandelbrot and its dynamical Julia twin.
#   julia:multibrot3 -> 0.30
#     jm3 revival sweep (scratchpad/jm3_tgood_sweep.py vs labels/jm3_band_v1.json,
#     2026-07-11): the band is 64% q3 yet the baseline 0.50 rejected all of it; p_good
#     carries directional resolution and the sweep supports anywhere in 0.28-0.36. 0.30
#     is a deliberately round mid-low pick, not a fitted decimal.
#   julia:multibrot4 / julia:multibrot5 -> 0.30
#     jm4/jm5 revival sweep (scratchpad/jm45_tgood_sweep.py vs labels/jm45_band_v1.json,
#     2026-07-12): both bands are 86% q3 (19/22 locations) yet the baseline 0.50 rejected
#     all of it — even richer than jm3. Both take jm3's lean-low 0.30 (sweep admits with
#     0.86-0.88 q3 precision and no q1 leak at 0.30); the Stage-2 quality gate nets any
#     location-level FPs. Independent calls that happen to land on the same value.
#   phoenix -> 0.18  (PROVISIONAL)
#     phoenix "take-the-best" study (prompts/phoenix_study_prompt.md, 2026-07-12; N=36,
#     NOT converged). The fixed Ushiki z-plane is a variety-poor garnish: bulk p_good
#     continuum 0.042-0.148, a good tail 0.171-0.220, then a lone gap to two standouts at
#     0.321/0.365. 0.18 sits just above the bulk p75 (0.138), capturing the good tail +
#     standouts at ~0.14 raw yield (~7 descents/keeper). PROVISIONAL: the 6h run is itself
#     the big phoenix sampler (every emission records its recipe), so re-threshold post-hoc
#     if 0.18 proves off. See [[phoenix-tgood-yield-study]].
#
# High-degree families NOT listed (c-plane multibrot3/4/5) stay unpowered on the eval and
# are HELD at the baseline until their own sweeps land.
# =========================================================================== #
T_GOOD_BASELINE = 0.50    # conservative default for every unswept / high-degree partition
T_GOOD_OVERRIDES = {
    "mandelbrot": 0.24,        # v6 deg-2 sweep knee
    "julia:mandelbrot": 0.24,  # deg-2 dynamical twin
    "julia:multibrot3": 0.30,  # jm3 revival sweep (2026-07-11)
    "julia:multibrot4": 0.30,  # jm4 revival sweep (2026-07-12)
    "julia:multibrot5": 0.30,  # jm5 revival sweep (2026-07-12)
    "phoenix": 0.18,           # phoenix take-the-best study (2026-07-12, PROVISIONAL)
}


def t_good_for(partition: str) -> float:
    """q3 p_good threshold for a cloud partition (family+degree). Swept partitions get
    their own value from T_GOOD_OVERRIDES; everything else -> 0.50 (baseline). See the
    block above for per-partition provenance."""
    return T_GOOD_OVERRIDES.get(partition, T_GOOD_BASELINE)


# flags        : extra guided-descend CLI flags (family/julia grammar; [] for mandelbrot).
# partition    : ledger `family` discriminator keying the q3 cloud (family+degree; Julia
#                is degree-only `julia:{fam}`, c is a coordinate). "mandelbrot" for the
#                native family (byte-identical to old rows).
# render_family: canonical location.family for the outcome/reward render path.
# c            : fixed (re, im) decimal-string pair under --julia, else None.
# id_tag       : short outcome-id prefix (keeps ids legible + collision-free across families).
# julia        : bool, the dynamical z-plane mode flag.
FamilyResolved = namedtuple(
    "FamilyResolved", "flags partition render_family c id_tag julia")


def resolve_family(args) -> FamilyResolved:
    """Validate the family grammar and resolve it to guided-descend flags + the cloud
    partition key + the render Location family + the fixed c + a short id tag. Mirrors the
    engine's rules (src/guided_descend.rs): `--julia` ⊥ `--phoenix`; `--phoenix` is
    degree-2 (⊥ multibrot); `--julia` requires `--c`; `--c` only with `--julia`. Raises
    SystemExit with a clear message on any invalid combo — rejected before any descent."""
    fam = args.family
    julia = bool(args.julia)
    phoenix = bool(args.phoenix)
    c = tuple(str(v) for v in args.c) if args.c is not None else None

    # --- mutual exclusion (engine: --julia and --phoenix are exclusive dynamical modes) ---
    if julia and phoenix:
        raise SystemExit("--julia and --phoenix are mutually exclusive dynamical modes")

    # --- Phoenix: recognized, then deliberately rejected (step 6). Phoenix is a single
    # fixed location with NO parameter plane, so spatial radius-rejection discovery has
    # nothing to prospect. Rejection is the correct resolution — NOT a one-point descent. ---
    if phoenix:
        if fam != "mandelbrot":
            raise SystemExit(
                "--phoenix is the degree-2 two-state plane; incompatible with "
                "--family multibrot*")
        raise SystemExit(
            "--phoenix has no (cx, cy) parameter plane to prospect: it is a single fixed "
            "location, so the seeder's spatial radius-rejection discovery does not apply. "
            "Descend Phoenix directly with `guided-descend --phoenix` (the shakeout-style "
            "single-location z-plane descent); the production seeder covers only families "
            "with a parameter plane to sample.")

    # --- --julia requires --c; --c only meaningful under --julia (Phoenix handled above) ---
    if julia:
        if c is None:
            raise SystemExit(
                "--julia requires --c <re> <im> (the fixed dynamical parameter); the "
                f"shakeout's known-good anchor is --c {JULIA_C[0]} {JULIA_C[1]}")
    elif c is not None:
        raise SystemExit(
            "--c given without --julia: it is the fixed dynamical parameter and is only "
            "valid under --julia")

    # --- resolve flags / partition / render family / id tag ---
    if julia:
        # Julia (quadratic) or Julia-multibrot: z-plane descent at the fixed c. Rides
        # BORROWED/UNTUNED Mandelbrot-family bands (plumbed, not yet validated — no band
        # tuning in scope). Also: the engine's `--seed-list` path is c-plane-only, so the
        # depth-2 probe + full walks (both seed-list) fail loudly under --julia — Julia is
        # wired correctly in construction but not yet runnable end-to-end through this
        # seed-list pipeline (would need a z-plane seed-list mode in the engine).
        deg = "" if fam == "mandelbrot" else fam[len("multibrot"):]
        flags = (["--family", fam] if fam != "mandelbrot" else []) + \
            ["--julia", "--c", c[0], c[1]]
        render_family = "julia" if fam == "mandelbrot" else f"julia_{fam}"
        # Degree-only partition (`c` is the cloud coordinate, not the key) — structurally
        # identical to the multibrot partitions; shared with the julia-hook found-points.
        partition = julia_partition(fam)
        id_tag = "jm" if fam == "mandelbrot" else f"jmb{deg}"
    elif fam in _MULTIBROT:
        deg = fam[len("multibrot"):]
        flags = ["--family", fam]
        render_family = fam
        partition = fam
        id_tag = f"mb{deg}"
    else:  # mandelbrot c-plane — byte-identical to the historical pipeline
        flags = []
        render_family = "mandelbrot"
        partition = "mandelbrot"
        id_tag = "m"

    return FamilyResolved(flags=flags, partition=partition, render_family=render_family,
                          c=c, id_tag=id_tag, julia=julia)


def make_loc_of(render_family: str, c):
    """Per-family reframe.Location factory for the raw-screen + reframe reward path
    (mirrors cross_family_shakeout.make_loc_of). `make_loc_of("mandelbrot", None)` is
    byte-identical to step0_reanalysis._mand_location, so the Mandelbrot reward path is
    unchanged; a multibrot/julia factory routes those frames through the same render path
    (render_one_flags reads the family off the Location)."""
    from reframe import Location
    c_re, c_im = (c if c is not None else (None, None))

    def loc_of(cx, cy, fw):
        return Location(family=render_family, c_re=c_re, c_im=c_im,
                        cx=str(cx), cy=str(cy), fw=str(fw), family_params={})
    return loc_of


# =========================================================================== #
# Distinctness predicates (pure — unit-tested)
# =========================================================================== #
def near_dup(a_cx, a_cy, a_fw, b_cx, b_cy, b_fw, k=DEDUP_K) -> bool:
    """fw-relative dedup: A near-dup of B iff plane distance < k * max(A.fw, B.fw).
    max(fw) => two outcomes at ~same center but different zoom are the SAME place."""
    d = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
    return d < k * max(float(a_fw), float(b_fw))


def is_distinct(cx, cy, fw, cloud, k=DEDUP_K):
    """Distinctness vs the q3 outcome cloud. Returns (distinct: bool, dup_of: id|None).
    `cloud` = iterable of dicts with outcome_cx/outcome_cy/outcome_fw/id."""
    for h in cloud:
        if near_dup(cx, cy, fw, h["outcome_cx"], h["outcome_cy"], h["outcome_fw"], k):
            return False, h["id"]
    return True, None


def count_within(cloud, cx, cy, radius=REJECT_RADIUS) -> int:
    """# distinct q3 cloud members within `radius` of (cx, cy) in (cx, cy) space.
    Linear scan — the cloud is small (<1e3) and the descent dwarfs this query."""
    if not cloud:
        return 0
    return sum(1 for m in cloud
               if np.hypot(m["outcome_cx"] - cx, m["outcome_cy"] - cy) < radius)


# =========================================================================== #
# Durable ledger (atomic temp+rename; cross-run cumulative; resumable)
# =========================================================================== #
def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Ledgers:
    """Loads existing state (cross-run cumulative) and appends/rewrites atomically.

    The q3 outcome cloud (the coverage state) is reconstructed by `build_cloud` from
    these rows; there is no separate cloud file. `rows` holds every scored outcome ever
    logged (distinct + dup + guarded)."""

    def __init__(self):
        self.rows: list[dict] = []                 # every scored-walk row (cross-run)
        self.feats: dict[str, np.ndarray] = {}     # id -> 1280-D
        self.load()

    @property
    def n_outcomes_logged(self) -> int:
        return len(self.rows)

    @property
    def harvested(self) -> list[dict]:
        """Guard-passed pool (cumulative). Kept for the confirmatory report's pool-size
        readout; the coverage state proper is the q3 cloud (`build_cloud`)."""
        return [r for r in self.rows if r.get("guard_pass", True)]

    def load(self):
        if OUTCOME_LEDGER.exists():
            for line in open(OUTCOME_LEDGER, encoding="utf-8"):
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if OUTCOME_FEATS.exists():
            z = np.load(OUTCOME_FEATS, allow_pickle=False)
            self.feats = {k: z[k] for k in z.files}

    # --- outcome append (jsonl) + feature store (npz) ---
    def append_outcome(self, row: dict, feat: np.ndarray | None):
        row.setdefault("scorer_version", SCORER_VERSION)   # which classifier produced this row
        OUTCOME_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTCOME_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        if feat is not None:
            self.feats[row["id"]] = np.asarray(feat, np.float32)

    def save_feats(self):
        if not self.feats:
            return
        DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
        # numpy auto-appends .npz unless the name already ends in it, so keep the
        # temp name .npz-suffixed (else savez writes elsewhere and the rename fails).
        tmp = OUTCOME_FEATS.parent / (OUTCOME_FEATS.stem + "_tmp.npz")
        np.savez_compressed(tmp, **self.feats)
        os.replace(tmp, OUTCOME_FEATS)


def append_probe_rejects(rows: list[dict]):
    if not rows:
        return
    PROBE_REJECTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROBE_REJECTS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# =========================================================================== #
# The q3 outcome cloud — reconstruct from the durable ledger (cross-run coverage
# state). Keep guard_pass && decoded_class == 3 rows, deduped by 1.5*max(fw).
# =========================================================================== #
def build_cloud(rows: list[dict], family: str) -> list[dict]:
    """One position per distinct q3 place *within the active `family` partition*:
    row family == `family` && guard_pass && decoded_class == 3, deduped by 1.5*max(fw).
    Order-stable: the earliest distinct row wins a dedup cluster, matching the live-harvest
    add order.

    The `family` filter is the correctness fix: cross-family outcomes at the same (cx, cy)
    are different parameter planes and must never interact in the radius query / dedup.
    Rows missing `family` default to "mandelbrot" (all pre-grammar rows are Mandelbrot),
    so the existing Mandelbrot cloud survives with no reset. Rows predating the
    decoded_class field (no historical backfill — see the module note) lack decoded_class
    and are simply excluded."""
    cloud: list[dict] = []
    for r in rows:
        if r.get("family", "mandelbrot") != family:
            continue
        if r.get("guard_pass", True) and r.get("decoded_class") == 3:
            distinct, _ = is_distinct(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"],
                                      cloud, DEDUP_K)
            if distinct:
                cloud.append(r)
    return cloud


def cloud_diagnostic(rows: list[dict], cloud: list[dict], family: str) -> dict:
    """Startup summary scoped to the active `family` partition: partition rows, guard_pass
    count, class-1/2/3 split among guard-clean *decoded* partition rows (pre-decoded_class
    rows are not counted), and the distinct q3 cloud size after dedup."""
    fam_rows = [r for r in rows if r.get("family", "mandelbrot") == family]
    guard_clean = [r for r in fam_rows
                   if r.get("guard_pass", True) and r.get("decoded_class") is not None]
    split = {c: sum(1 for r in guard_clean if r["decoded_class"] == c) for c in (1, 2, 3)}
    n_undecoded = sum(1 for r in fam_rows
                      if r.get("guard_pass", True) and r.get("decoded_class") is None)
    return {"family": family, "total_rows": len(rows), "partition_rows": len(fam_rows),
            "guard_pass": sum(1 for r in fam_rows if r.get("guard_pass", True)),
            "guard_clean_decoded": len(guard_clean), "undecoded_guard_pass": n_undecoded,
            "class_split": split, "cloud_size": len(cloud)}


# =========================================================================== #
# Native seed source + rejection sampler. The engine's own root draw (already
# descendability-pre-gated by the native 8k/flat root gate) proposes depth-1 seeds;
# each is accepted only if the q3 cloud is sparse within REJECT_RADIUS. The pool
# refills on demand (fresh engine sub-seed) so the ONLY stop conditions are global
# saturation (MAX_SEED_REDRAWS consecutive rejections) and the wallclock budget.
# =========================================================================== #
def generate_native_seeds(n_walks: int, seed: int, workdir: Path,
                          family_flags: list | None = None) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(prescreen.BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", "1", "--depth-max", "1",
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(workdir),
    ] + (list(family_flags) if family_flags else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"native seed generation failed:\n{r.stderr[-2000:]}")
    seeds = []
    for line in open(workdir / "walks.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        w = json.loads(line)
        if w.get("reached_depth", 0) >= 1 and w.get("root_cx") is not None:
            seeds.append({"cx": float(w["root_cx"]), "cy": float(w["root_cy"]),
                          "fw": float(w["root_fw"]), "root_src": w.get("root_src", "")})
    if not seeds:
        raise SystemExit("native seed generation produced no depth-1 roots")
    return seeds


class NativeSeeder:
    """Draws native depth-1 seeds and applies the q3-density rejection test. Refills the
    native pool from a fresh engine sub-seed when depleted. Tracks draw/reject counters
    and the saturation flag (consecutive rejections > MAX_SEED_REDRAWS)."""

    def __init__(self, base_seed: int, scratch: Path, rng: np.random.Generator,
                 family_flags: list | None = None):
        self.base_seed = base_seed
        self.scratch = scratch
        self.rng = rng
        self.family_flags = family_flags or []   # native root draw runs on the active family
        self.gen = 0
        self.q: list[dict] = []
        self.draws = 0          # total native seeds examined (accepted + rejected)
        self.rejects = 0        # total rejected by the density test
        self.consec = 0         # consecutive rejections (resets on an accept)
        self.saturated = False

    def _refill(self):
        wd = self.scratch / f"native_{self.gen}"
        seeds = generate_native_seeds(NATIVE_POOL_WALKS, self.base_seed + 1000 * self.gen, wd,
                                      self.family_flags)
        self.rng.shuffle(seeds)
        self.q.extend(seeds)
        self.gen += 1

    def _next_raw(self) -> dict:
        if not self.q:
            self._refill()
        return self.q.pop()

    def draw_batch(self, cloud: list[dict], n_batch: int) -> list[dict]:
        """Fill a batch with accepted seeds. A seed is rejected (and redrawn) if
        >= Q3_DENSITY_CAP distinct q3 cloud members lie within REJECT_RADIUS. Returns
        the accepted proposals (possibly < n_batch if saturation trips mid-batch)."""
        props = []
        while len(props) < n_batch:
            s = self._next_raw()
            self.draws += 1
            if count_within(cloud, s["cx"], s["cy"], REJECT_RADIUS) >= Q3_DENSITY_CAP:
                self.rejects += 1
                self.consec += 1
                if self.consec > MAX_SEED_REDRAWS:
                    self.saturated = True
                    break
                continue
            self.consec = 0
            props.append({"mix_source": "native", "seed_cx": s["cx"], "seed_cy": s["cy"],
                          "fw": s["fw"], "root_src": s.get("root_src", "")})
        return props


# =========================================================================== #
# Depth-2 descendability probe (reuse prescreen.prescreen VERBATIM; read the walks it
# writes for per-seed reached + cause). reached>=2 -> survivor; else probe-reject.
# =========================================================================== #
def depth2_probe(props: list[dict], workdir: Path, seed: int, family_flags: list | None = None):
    """Returns (survivors, rejects, causes) where survivor rows carry the proposal +
    reached, reject rows carry seed_cx/cy/reached/cause/child_occ. `family_flags` thread
    the active family's grammar into the depth-2 probe (c-plane families only; a --julia
    probe surfaces the engine's own "--seed-list is c-plane-only" error — see step 5)."""
    cloud = np.array([[p["seed_cx"], p["seed_cy"]] for p in props], float)
    fw = np.array([p["fw"] for p in props], float)
    scr = prescreen.prescreen(cloud, fw, workdir, NODE_WIDTH, OCC_FLOOR, BLACK_CAP, seed,
                              extra_flags=family_flags)
    reached = scr["reached"]
    # per-seed cause + chosen-child occupancy from the probe's own walks.jsonl (row
    # order == proposal order). child_occ is the engine's OWN depth-2 admission-point
    # occupancy (energy::occupancy, emitted per walk) — the value the 0.321 floor
    # gates against, reused verbatim. null iff the walk died before reaching depth 2.
    causes, child_occ = {}, {}
    wpath = workdir / "probe_pool" / "walks.jsonl"
    for line in open(wpath, encoding="utf-8"):
        line = line.strip()
        if line:
            w = json.loads(line)
            causes[int(w["walk"])] = w.get("cause", "")
            child_occ[int(w["walk"])] = w.get("child_occ")

    survivors, rejects = [], []
    for i, p in enumerate(props):
        p2 = dict(p); p2["probe_reached"] = int(reached[i]); p2["probe_cause"] = causes.get(i, "")
        p2["probe_child_occ"] = child_occ.get(i)
        if scr["pass"][i]:
            survivors.append(p2)
        else:
            rejects.append({
                "seed_cx": p["seed_cx"], "seed_cy": p["seed_cy"],
                "mix_source": p["mix_source"], "reached": int(reached[i]),
                "cause": causes.get(i, ""),
                "child_occ": child_occ.get(i),
            })
    return survivors, rejects, scr["causes"]


# =========================================================================== #
# Full walks + k3 best-frame reward (reuse step0_reanalysis primitives) + outcome
# center + 1280-D penultimate feature.
# =========================================================================== #
def run_full_walks(survivors: list[dict], workdir: Path, seed: int,
                   family_flags: list | None = None):
    """One --seed-list --per-walk-rng production walk run over the survivors, on the active
    family (c-plane families only — see the depth-2 probe / step 5 note)."""
    workdir.mkdir(parents=True, exist_ok=True)
    seed_in = workdir / "survivor_seeds.jsonl"
    prescreen.write_seed_list(seed_in, [s["seed_cx"] for s in survivors],
                            [s["seed_cy"] for s in survivors], [s["fw"] for s in survivors])
    pool = workdir / "pool"
    cmd = [
        str(prescreen.BIN), "guided-descend",
        "--seed-list", str(seed_in), "--per-walk-rng", "--seed", str(seed),
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ] + (list(family_flags) if family_flags else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"full walk run failed:\n{r.stderr[-2000:]}")
    return pool


def run_julia_descent(c, mode: str, seed: int, workdir: Path, n_walks: int,
                      family: str = "mandelbrot",
                      esc_median_min: float | None = None,
                      spread_min: float | None = None) -> Path:
    """Native Julia z-plane descent at the fixed parameter `c = (c_re, c_im)`, run at
    PRODUCTION depth (not depth-1). A variant of `generate_native_seeds` that shells
    `guided-descend --julia --c <c_re> <c_im>` (+ `--julia-center` when `mode ==
    "center"`, + `--family multibrot{d}` for a multibrot parent so the dynamics match
    the render family). NATIVE only — never `--seed-list` (the engine rejects a z-plane
    seed list under `--julia`); the julia root step is the engine's own base-scale
    z-plane draw, so native descent needs no injected seeds. Writes `pool.jsonl` under
    `workdir/pool` and returns that pool dir (for `load_frames_by_walk`).

    `esc_median_min` / `spread_min` (default None → engine per-family defaults) inject the
    loosened, degree-aware Julia bands for gather mode via `--esc-median-min` /
    `--spread-min`. Julia-only: passed ONLY on the julia sub-descent, never the c-plane."""
    workdir.mkdir(parents=True, exist_ok=True)
    pool = workdir / "pool"
    fam_flags = [] if family == "mandelbrot" else ["--family", family]
    cmd = [
        str(prescreen.BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--julia", "--c", str(c[0]), str(c[1]),
        "--out-dir", str(pool),
    ] + fam_flags + (["--julia-center"] if mode == "center" else [])
    if esc_median_min is not None:
        cmd += ["--esc-median-min", str(esc_median_min)]
    if spread_min is not None:
        cmd += ["--spread-min", str(spread_min)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"julia descent failed (c={c}, mode={mode}):\n{r.stderr[-2000:]}")
    return pool


def run_phoenix_descent(seed: int, workdir: Path, n_walks: int) -> Path:
    """Native Phoenix z-plane descent at the fixed Ushiki location (engine defaults
    c=0.5667,0 / p=-0.5,0), run at PRODUCTION depth. Phoenix has NO parameter plane to
    prospect (single fixed location), so gather harvests it by repeatedly descending the
    one z-plane and logging every outcome — no seed list, no density rejection. Writes
    `pool.jsonl` under `workdir/pool` and returns that pool dir."""
    workdir.mkdir(parents=True, exist_ok=True)
    pool = workdir / "pool"
    cmd = [
        str(prescreen.BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--phoenix",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"phoenix descent failed:\n{r.stderr[-2000:]}")
    return pool


def _chosen_probs(res) -> tuple[float, float]:
    """CORN (p_notbad, p_good) of a reframe winner's chosen frame — pulled from the
    reframe trace (already computed when k3 was scored; no extra render/forward)."""
    ch = res.trace["chosen"]
    for rc in res.trace["recenter"]:
        if rc["dx"] == ch["dx"] and rc["dy"] == ch["dy"]:
            return float(rc["p_notbad"]), float(rc["p_good"])
    # unreachable (chosen is always one of the recenter candidates); degrade to class-1.
    return 0.0, 0.0


def harvest_walk_reward(scorer, wid, frames, workers, scratch, loc_of=_mand_location):
    """k3 best-frame reward for one walk, composing the step0_reanalysis primitives
    (raw_screen_walk / reframe_location / KRAW),
    plus the raw-top3 list, the k3-winner's reframed outcome geometry, and the winner's
    CORN (p_notbad, p_good) for the hard-class decode.

    `loc_of(cx, cy, fw) -> reframe.Location` is the active family's location factory
    (default `_mand_location`, byte-identical for Mandelbrot); it routes both the
    raw-screen render AND the top-3 reframe through the correct fractal so a multibrot /
    Julia walk's reward is scored on its own frames, not Mandelbrot crops."""
    sr.SCRATCH = scratch
    triples = raw_screen_walk(scorer, wid, frames, workers, loc_of=loc_of,
                              return_triples=True)
    raws = [t[0] for t in triples]
    # Parent gate for the Julia sub-descent hook: decode the RAW frames (un-reframed —
    # the frames raw_screen_walk already scored; no new render/forward) and surface the
    # counts. Harmless to compute for Julia sub-walks too; only read on parents.
    # DELIBERATELY at the baseline t_good=0.5 (default): this is a hook-FIRING heuristic on
    # the parent's raw frames, not a q3 ADMISSION decode, so it stays byte-identical and
    # is not routed through the per-degree t_good (which gates only the committed outcome).
    n_frames_q2plus = sum(1 for (_, nb, g) in triples if corn_decode(nb, g) >= 2)
    n_frames_q3 = sum(1 for (_, nb, g) in triples if corn_decode(nb, g) == 3)
    # Run-scoped guard observability (pure read of the raw scores — changes nothing):
    # a raw frame that scored GUARD_SENTINEL was pushed out of top-3 contention as
    # degenerate. `frames_gated` counts them; the salvage-breakdown classifier uses it.
    frames_gated = int(sum(1 for r in raws if r <= guard.GUARD_SENTINEL + 1e-6))
    order = sorted(range(len(frames)), key=lambda i: raws[i], reverse=True)
    topk = order[:KRAW]
    raw_top3 = [float(raws[i]) for i in topk]

    best = None   # (reframed_score, res, idx)
    reward_k1 = None
    for rank, i in enumerate(topk):
        fr = frames[i]
        loc = loc_of(fr["cx"], fr["cy"], fr["fw"])
        wd = scratch / f"walk_{wid:04d}" / f"reframe_top{rank}"
        res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=workers)
        # monotone-non-decreasing by construction (the x1.0 rung is in the search space).
        if res.score < res.trace["original_score"] - 1e-4:
            raise SystemExit(f"MONOTONICITY VIOLATED walk {wid} idx {fr['idx']}: "
                             f"{res.score:.4f} < {res.trace['original_score']:.4f}")
        if rank == 0:
            reward_k1 = float(res.score)
        if best is None or res.score > best[0]:
            best = (float(res.score), res, int(fr["idx"]))

    reward_k3, res, k3_idx = best
    p_notbad, p_good = _chosen_probs(res)
    reached = max(int(f["depth"]) for f in frames)
    return {
        "reward_k3": reward_k3, "reward_k1": reward_k1, "raw_top3": raw_top3,
        "reached_depth": reached, "k3_argmax_idx": k3_idx,
        "outcome_cx": float(res.cx), "outcome_cy": float(res.cy), "outcome_fw": float(res.fw),
        "p_notbad": p_notbad, "p_good": p_good,
        "frames_gated": frames_gated, "n_frames": len(frames),
        "n_frames_q2plus": n_frames_q2plus, "n_frames_q3": n_frames_q3,
    }


def outcome_feature(scorer, cx, cy, fw, tile: Path, *, family="mandelbrot", c=None) -> np.ndarray:
    """Render the k3 winner's reframed crop once at deploy search fidelity (640x360 ss2,
    twilight_shifted) on the active `family`/`c` and forward it through the v5 penultimate
    hook -> 1280-D. Default (mandelbrot, no c) is byte-identical to the historical call."""
    ok, err = prescreen._render(cx, cy, fw, tile, family=family, c=c)
    if not ok:
        raise SystemExit(f"outcome tile render failed [{tile.name}]: {err}")
    return prescreen.embed_paths(scorer, [tile])[0]


# =========================================================================== #
# Gather mode (--gather): guard-OFF harvest for the v6 label pass. Reuses the
# native seeder / probe / walk / reward machinery, but (1) scores with the RAW v5
# (degenerate frames survive — v6 needs the OOD cases), (2) records the guard
# would-pass verdict per outcome as a prior (not a gate), (3) writes to a separate
# durable per-class subtree with the full walks.jsonl and NO image/feature dumps.
# =========================================================================== #
def outcome_guard_verdict(cx, cy, fw, out_bin: Path, *, family="mandelbrot", c=None,
                          family_params=None):
    """Compute the degenerate-outcome guard's would-pass verdict on ONE outcome crop,
    model-free from the dumped f64 smooth field (same path production uses inside the
    guarded scorer). Returns (verdict, GuardStats) where verdict ∈
    {"pass","interior","flat","both"} or "render_error" (field render failed twice).
    This is a PRIOR logged per outcome — never a gate in gather mode — so a transient
    render failure logs "render_error" with null stats and the OUTCOME IS STILL KEPT
    (it must never abort the harvest). `c=(re,im)` for dynamical families, else None."""
    c_re, c_im = (c if c is not None else (None, None))
    err = ""
    for _ in range(2):   # one retry absorbs transient Windows spawn / render hiccups
        ok, err = guard.render_field(cx, cy, fw, out_bin, family=family, c_re=c_re,
                                     c_im=c_im, family_params=family_params)
        if ok:
            stats = guard.field_measures(guard.load_field(out_bin).values)
            reason = guard.guard_fail(stats.interior_frac, stats.field_std)
            return ("pass" if reason is None else reason), stats
    print(f"  WARN guard field render failed (verdict=render_error, outcome kept) "
          f"[{Path(out_bin).name}]: {err.strip()[:120]}")
    return "render_error", guard.GuardStats(interior_frac=None, field_std=None,
                                            n_px=0, n_escaped=0)


class GatherLedger:
    """Append-only guard-OFF ledger for one gather class, under GATHER_DIR/<partition>/.
    Cross-run cumulative (reloads existing rows so a resumed class keeps its q3 cloud +
    Julia repulsion). No feature store (gather logs coords only; selection re-renders
    from coords), no image tiles."""

    def __init__(self, class_dir: Path):
        self.class_dir = class_dir
        self.path = class_dir / "outcome_ledger.jsonl"
        self.rows: list[dict] = []
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def append(self, row: dict):
        row.setdefault("scorer_version", SCORER_VERSION)   # which classifier produced this row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)


def load_walk_meta(pool: Path) -> dict:
    """Per-walk metadata rows the engine writes to `pool/walks.jsonl` (root_cx/cy/fw,
    cause, child_occ, reached_depth), keyed by walk id. Empty dict if absent."""
    meta = {}
    p = pool / "walks.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                w = json.loads(line)
                meta[int(w["walk"])] = w
    return meta


def _frame_lite(fr: dict) -> dict:
    """One trajectory frame's coords + occupancy (the light per-frame record)."""
    return {"idx": int(fr["idx"]), "depth": int(fr["depth"]),
            "cx": fr["cx"], "cy": fr["cy"], "fw": fr["fw"], "occ": fr.get("occ")}


def persist_walk(fh, *, family, batch, walk, outcome_id, frames, meta,
                 parent_oid=None, descend_mode="cplane"):
    """Append one walk's full trajectory (every frame's coords + occupancy) + the engine
    per-walk metadata to the gather run's walks.jsonl. No images — coords only."""
    m = meta or {}
    rec = {
        "family": family, "batch": int(batch), "walk": int(walk),
        "outcome_id": outcome_id, "parent_oid": parent_oid, "descend_mode": descend_mode,
        "reached_depth": max((int(f["depth"]) for f in frames), default=0),
        "root_cx": m.get("root_cx"), "root_cy": m.get("root_cy"), "root_fw": m.get("root_fw"),
        "cause": m.get("cause"), "child_occ": m.get("child_occ"),
        "frames": [_frame_lite(f) for f in frames],
    }
    fh.write(json.dumps(rec) + "\n")


# =========================================================================== #
# Contact sheet (harvested distinct q3 outcomes + a small dup/non-q3 strip)
# =========================================================================== #
def build_contact_sheet(distinct_tiles, dup_tiles, out_png: Path, title: str):
    from PIL import Image, ImageDraw
    TW, TH, PAD, LBL, GUT = 224, 126, 6, 16, 40
    NCOL = 6
    items = [(p, lab, (60, 220, 90)) for p, lab in distinct_tiles]
    if dup_tiles:
        items.append((None, "--- dup / non-q3 / guarded ---", (245, 215, 40)))
        items += [(p, lab, (245, 215, 40)) for p, lab in dup_tiles]
    nrow = (len(items) + NCOL - 1) // NCOL
    cell_w, cell_h = TW + 2 * PAD, TH + LBL + 2 * PAD
    W, H = NCOL * cell_w, GUT + nrow * cell_h
    sheet = Image.new("RGB", (W, max(H, GUT + cell_h)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 12), title, fill=(235, 235, 235))
    for k, (tp, lab, col) in enumerate(items):
        r, c = divmod(k, NCOL)
        x, y = c * cell_w + PAD, GUT + r * cell_h + PAD
        if tp is not None and Path(tp).exists():
            im = Image.open(tp).convert("RGB").resize((TW, TH))
            sheet.paste(im, (x, y))
            for t in range(2):
                draw.rectangle([x - 1 - t, y - 1 - t, x + TW + t, y + TH + t], outline=col)
        draw.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        draw.text((x + 3, y + TH + 2), lab, fill=(210, 210, 218))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


# =========================================================================== #
# Orchestration
# =========================================================================== #
# Transient render scratch under SCRATCH_ROOT/run_ts (per-walk reward frames,
# reframe rungs, native walk pools, field bins) is the overnight loop's dominant
# disk sink — GBs per run, hundreds of runs/night, ~130GB accumulated. It is pure
# scoring intermediate: the durable outputs (outcome_ledger + feats npz, and the
# run's summary/telemetry under RUNS_DIR in data/; the contact_sheet is a disposable
# render view under SHEETS_DIR in out/) never live here,
# and the pool builder + emitter read only the ledger, so nothing downstream needs
# it once the run process has scored its walks. Purge it on clean exit.
_SCRATCH_KEEP = ("outcome_tiles",)   # tiny per-outcome jpgs; only --finalize (manual,
                                     # for KILLED runs) rereads them — cheap to keep.


def _purge_run_scratch(scratch: Path, keep: tuple[str, ...] = _SCRATCH_KEEP):
    """Remove the heavy transient render dirs under a run's scratch (reward_*/walk_*/
    batch_*/round_*/jreward_* …), keeping only `keep`. Best-effort: a failure to unlink
    (Windows file lock, race) is logged, never fatal — the sweep is a disk courtesy, not
    a correctness step."""
    if not scratch.exists():
        return
    freed = 0
    for child in scratch.iterdir():
        if child.name in keep:
            continue
        try:
            if child.is_dir():
                freed += sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                shutil.rmtree(child, ignore_errors=True)
            else:
                freed += child.stat().st_size
                child.unlink()
        except OSError as e:
            print(f"  scratch purge: could not remove {child.name}: {e}")
    print(f"  scratch purged: freed ~{freed/2**30:.2f} GiB of transient render dirs "
          f"under {scratch} (kept {list(keep)})")


def _sheet_path(run_ts: str) -> Path:
    """Disposable contact-sheet path under SHEETS_DIR (out/), keyed by run_ts. Rebuilt on
    demand by `_finalize` from the durable ledger + tiles, so it never belongs in data/."""
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    return SHEETS_DIR / f"{run_ts}.png"


def _run(args, fam: FamilyResolved):
    smoke = args.smoke
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_ts
    scratch = SCRATCH_ROOT / run_ts
    scratch.mkdir(parents=True, exist_ok=True)
    tiles_dir = scratch / "outcome_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # smoke: small native pool + lowered rejection knobs so the density test fires fast.
    global Q3_DENSITY_CAP, MAX_SEED_REDRAWS, NATIVE_POOL_WALKS
    if smoke:
        Q3_DENSITY_CAP = 2       # a couple of nearby q3 outcomes already reject
        MAX_SEED_REDRAWS = 30    # saturation reachable within a smoke
        NATIVE_POOL_WALKS = 60
    batch_seeds = args.batch or (12 if smoke else BATCH_SEEDS)
    budget_min = args.budget if args.budget is not None else (0 if smoke else WALLCLOCK_BUDGET_MIN)

    print(f"=== atlas production seeder ({'SMOKE' if smoke else 'RUN'}) ts={run_ts} ===")
    print(f"family: {fam.partition}  (descend flags: {' '.join(fam.flags) or '(none: native mandelbrot)'}"
          f"; render family {fam.render_family})")
    if fam.julia:
        print("  NOTE: --julia is PLUMBED, not validated — rides borrowed/untuned Mandelbrot "
              "bands, and the seed-list probe/walk steps are c-plane-only (engine will reject).")
    print(f"coverage: q3-density REJECTION  radius={REJECT_RADIUS} cap={Q3_DENSITY_CAP} "
          f"max_redraws={MAX_SEED_REDRAWS} dedup_k={DEDUP_K}  batch={batch_seeds} budget={budget_min}min")
    print(f"walk cfg: node={NODE_WIDTH} sigma={SIGMA_BAND} depth[{DEPTH_MIN},{DEPTH_MAX}] "
          f"occ={OCC_FLOOR} black={BLACK_CAP} per_walk_rng={PER_WALK_RNG}")

    # Guarded scorer (SCORER_PATH = probe.ACTIVE_CKPT): raw-frame scoring, reframe candidate
    # scoring, and the k3 reward all inherit the model-free field guard (degenerate crops
    # -> GUARD_SENTINEL).
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX, (
        f"guard field suffix drift: reframe {reframe.GUARD_FIELD_SUFFIX!r} != "
        f"guard {guard.FIELD_SIDECAR_SUFFIX!r}")
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    print(f"scorer: GUARDED CORN ({SCORER_PATH}, {SCORER_VERSION})  geometry={scorer.cfg.get('geometry')}  "
          f"guard: interior_frac>={guard.INTERIOR_CAP} | field_std<{guard.FIELD_STD_FLOOR} "
          f"@ {guard.GUARD_STAT_RES}")

    ledgers = Ledgers()
    # No historical backfill (by design): rows predating the decoded_class field don't
    # enter the q3 cloud; the cross-run coverage cloud rebuilds from rows the new pipeline
    # logs going forward.
    cloud = build_cloud(ledgers.rows, fam.partition)
    diag = cloud_diagnostic(ledgers.rows, cloud, fam.partition)
    print(f"ledgers: {diag['total_rows']} rows ({diag['partition_rows']} in partition "
          f"'{fam.partition}'), {diag['guard_pass']} guard_pass "
          f"({diag['guard_clean_decoded']} decoded; class1/2/3="
          f"{diag['class_split'][1]}/{diag['class_split'][2]}/{diag['class_split'][3]}; "
          f"{diag['undecoded_guard_pass']} pre-decode rows excluded)"
          f"  | q3 cloud {diag['cloud_size']} distinct places")

    rng = np.random.default_rng(args.seed)
    native = NativeSeeder(args.seed, scratch, rng, fam.flags)
    # active family's reward-render factory (raw-screen + reframe route through this).
    loc_of = make_loc_of(fam.render_family, fam.c)

    # --- Julia sub-descent hook state (only when --julia-hook; else fully inert) ---
    # The hook hangs a native --julia descent off each qualifying c-plane outcome `c`,
    # scores it, and commits q3 results in the degree-only `julia:{fam}` partition. That
    # partition's found-points are a SEPARATE intra-run cloud (repel future `c`); cross-run
    # repulsion is automatic (rebuilt from the ledger next run). The render/descent family
    # is this run's c-plane family flipped to its dynamical Julia twin.
    julia_hook = bool(args.julia_hook)
    if julia_hook:
        julia_render_family = "julia" if fam.partition == "mandelbrot" \
            else f"julia_{fam.partition}"
        julia_part = julia_partition(fam.partition)
        julia_cloud = build_cloud(ledgers.rows, julia_part)
        jdiag = cloud_diagnostic(ledgers.rows, julia_cloud, julia_part)
        print(f"julia-hook: ON  partition '{julia_part}' render_family={julia_render_family} "
              f"walks/descent={JULIA_WALKS_PER_DESCENT}  | julia q3 cloud "
              f"{jdiag['cloud_size']} distinct c ({jdiag['partition_rows']} partition rows)")
    julia_added_this_run = 0

    totals = {"proposed": 0, "probe_rejected": 0, "walked": 0,
              "harvested_distinct": 0, "q3_dup": 0, "not_q3": 0, "guarded": 0,
              "julia_descents": 0, "julia_walks": 0, "julia_q3": 0, "julia_distinct": 0}
    distinct_tiles, dup_tiles = [], []
    batch_timings = []
    # Run-scoped guard telemetry (per walk). Observes scoring; never alters it or the
    # durable ledger schema. Written to the run dir as guard_telemetry.jsonl.
    telemetry = []
    t0 = time.time()
    seq = 0
    batch_i = 0
    q3_added_this_run = 0

    while True:
        tb = time.time()
        batch_i += 1
        # 1. draw a batch of accepted seeds (density-rejection pre-descent).
        props = native.draw_batch(cloud, batch_seeds)
        if not props:
            if native.saturated:
                print(f"  GLOBAL SATURATION: {MAX_SEED_REDRAWS} consecutive rejections "
                      f"(q3 cloud dense everywhere the sampler proposes); stopping cleanly.")
            else:
                print("  native seed source produced no proposals; stopping.")
            break
        totals["proposed"] += len(props)

        # 2. depth-2 descendability probe (survivors reached>=2).
        pw = scratch / f"batch_{batch_i:03d}" / "probe"
        survivors, rejects, pcauses = depth2_probe(props, pw, args.seed, fam.flags)
        append_probe_rejects(rejects)
        totals["probe_rejected"] += len(rejects)
        if not survivors:
            print(f"  batch {batch_i}: 0/{len(props)} descendable (causes {pcauses}); next batch.")
            batch_timings.append(time.time() - tb)
            if native.saturated:
                break
            if budget_min and (time.time() - t0) / 60 >= budget_min:
                break
            if smoke:
                break
            continue

        # 3. full production walks over survivors.
        ww = scratch / f"batch_{batch_i:03d}" / "walks"
        pool = run_full_walks(survivors, ww, args.seed, fam.flags)
        by_walk = load_frames_by_walk(pool)
        totals["walked"] += len(by_walk)

        # 4. per-walk k3 reward + outcome center + decode + 1280-D feature; 5. cloud add.
        b_distinct = b_q3dup = b_notq3 = b_guarded = 0
        b_jdesc = b_jdist = 0   # julia-hook: descents fired / distinct julia q3 this batch
        for wid in sorted(by_walk):
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS,
                                      scratch / f"reward_b{batch_i:03d}", loc_of)
            sv = survivors[wid] if wid < len(survivors) else survivors[-1]
            # Guard verdict: k3 collapses to the sentinel iff EVERY framing of the top-3
            # failed the guard. A guarded outcome carries decoded_class=None and can never
            # be a q3 cloud member.
            guard_pass = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
            t_good = t_good_for(fam.partition)   # per-degree q3 operating point
            decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good) if guard_pass else None
            is_q3 = guard_pass and decoded == 3
            if is_q3:
                distinct, dup_of = is_distinct(rew["outcome_cx"], rew["outcome_cy"],
                                               rew["outcome_fw"], cloud, DEDUP_K)
            else:
                distinct, dup_of = False, None

            oid = f"{fam.id_tag}_{run_ts}_{seq:06d}"; seq += 1
            tile = tiles_dir / f"{oid}.jpg"
            feat = outcome_feature(scorer, rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"],
                                   tile, family=fam.render_family, c=fam.c)
            row = {
                # `family` is the cloud partition discriminator (family+degree, +anchor for
                # Julia). Recoverable, so — unlike decoded_class — no fresh-start is needed;
                # existing rows lack it and default to "mandelbrot" in build_cloud.
                "id": oid, "ts": run_ts, "family": fam.partition,
                "mix_source": sv["mix_source"],
                "seed_cx": sv["seed_cx"], "seed_cy": sv["seed_cy"],
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"], "raw_top3": rew["raw_top3"],
                "probe_child_occ": sv.get("probe_child_occ"), "probe_reached": sv.get("probe_reached"),
                "probe_cause": sv.get("probe_cause"), "reached_depth": rew["reached_depth"],
                # decoded_class: the CORN hard class of the k3-winning frame (None if guarded).
                # cross-run cloud reconstruction is then a direct filter (guard_pass &&
                # decoded_class == 3) with no re-decode.
                "decoded_class": decoded,
                # CORN cumulative probs of the k3 winner + the per-degree t_good used to
                # decode them. Stamped so the ledger self-describes (mixed-era decodes) and
                # the harvest-monitor can build the p_good/p_notbad histograms.
                "p_notbad": rew["p_notbad"], "p_good": rew["p_good"], "t_good": t_good,
                "distinct": distinct, "dup_of": dup_of,
                "guard_pass": guard_pass, "guard_fail": None if guard_pass else "sentinel",
            }
            ledgers.append_outcome(row, feat)
            telemetry.append({
                "walk_uid": f"b{batch_i:03d}_w{wid:04d}", "batch": batch_i, "walk": int(wid),
                "outcome_id": oid, "mix_source": sv["mix_source"],
                "frames_gated": int(rew["frames_gated"]), "n_frames": int(rew["n_frames"]),
                "k3": float(rew["reward_k3"]), "k3_is_sentinel": (not guard_pass),
                "decoded_class": decoded, "guard_pass": bool(guard_pass),
                "harvested_distinct": bool(distinct),
            })
            if distinct:
                cloud.append(row); q3_added_this_run += 1; b_distinct += 1
                distinct_tiles.append((tile, f"{oid[-6:]} k3={rew['reward_k3']:.2f} q3"))
            elif is_q3:
                b_q3dup += 1
                if len(dup_tiles) < NCOL_DUP:
                    dup_tiles.append((tile, f"k3={rew['reward_k3']:.2f} q3->dup"))
            elif not guard_pass:
                b_guarded += 1
                if len(dup_tiles) < NCOL_DUP:
                    dup_tiles.append((tile, f"k3={rew['reward_k3']:.2f} GUARDED"))
            else:
                b_notq3 += 1
                if len(dup_tiles) < NCOL_DUP:
                    dup_tiles.append((tile, f"k3={rew['reward_k3']:.2f} cls{decoded}"))

            # --- Julia sub-descent hook (strictly additive; runs only with --julia-hook,
            # AFTER the parent has committed above; the parent path is byte-unchanged). ---
            if julia_hook:
                # Parent gate (un-dropped raw-frame decodes): fire only if the parent walk
                # shows real structure — >=2 raw frames decode q2+ OR >=1 raw frame decodes
                # q3 (the q3 clause rescues a single-spike walk). No reframing / new renders.
                qualifies = rew["n_frames_q2plus"] >= 2 or rew["n_frames_q3"] >= 1
                jc = (rew["outcome_cx"], rew["outcome_cy"])   # the parameter c (cloud coord)
                jc_fw = rew["outcome_fw"]                      # parent plane fw = dedup scale
                # Density pre-check: skip a c already saturated with Julia found-points.
                dense = count_within(julia_cloud, jc[0], jc[1], REJECT_RADIUS) >= Q3_DENSITY_CAP
                if qualifies and not dense:
                    totals["julia_descents"] += 1; b_jdesc += 1
                    jmode = "center" if rng.random() < 0.5 else "normal"   # 50/50
                    # deterministic per-parent seed (stable across runs; oid embeds seq+ts).
                    jseed = int(hashlib.md5(f"{oid}:{run_ts}".encode()).hexdigest()[:8], 16)
                    jwork = scratch / f"batch_{batch_i:03d}" / "julia" / f"w{wid:04d}"
                    jpool = run_julia_descent(jc, jmode, jseed, jwork,
                                              JULIA_WALKS_PER_DESCENT, family=fam.partition)
                    # score each Julia walk on ITS OWN frames (fixed c) via the julia factory.
                    jloc_of = make_loc_of(julia_render_family, (str(jc[0]), str(jc[1])))
                    for jwid, jframes in sorted(load_frames_by_walk(jpool).items()):
                        totals["julia_walks"] += 1
                        jrew = harvest_walk_reward(
                            scorer, jwid, jframes, WORKERS,
                            scratch / f"jreward_b{batch_i:03d}_w{wid:04d}", jloc_of)
                        # guard verdict / decode / q3 — EXACTLY as the parent loop does.
                        jguard_pass = jrew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
                        jt_good = t_good_for(julia_part)   # per-degree q3 operating point
                        jdecoded = corn_decode(jrew["p_notbad"], jrew["p_good"], jt_good) if jguard_pass else None
                        jis_q3 = jguard_pass and jdecoded == 3
                        # distinctness is on `c` (the parameter), NOT the z-plane spot:
                        # multiple q3 walks at the same c collapse to one found-point.
                        if jis_q3:
                            jdistinct, jdup_of = is_distinct(jc[0], jc[1], jc_fw,
                                                             julia_cloud, DEDUP_K)
                        else:
                            jdistinct, jdup_of = False, None

                        joid = f"j{fam.id_tag}_{run_ts}_{seq:06d}"; seq += 1
                        jtile = tiles_dir / f"{joid}.jpg"
                        # 1280-D feature on the Julia z-plane outcome crop (fixed c).
                        jfeat = outcome_feature(
                            scorer, jrew["outcome_cx"], jrew["outcome_cy"], jrew["outcome_fw"],
                            jtile, family=julia_render_family, c=(str(jc[0]), str(jc[1])))
                        jrow = {
                            # cloud role: outcome_* IS the parameter c (build_cloud repels on c).
                            "id": joid, "ts": run_ts, "family": julia_part,
                            "mix_source": "julia_hook", "parent_oid": oid,
                            "descend_mode": jmode,
                            "outcome_cx": jc[0], "outcome_cy": jc[1], "outcome_fw": jc_fw,
                            # render target: the Julia walk's own z-plane wallpaper location.
                            "julia_z_cx": jrew["outcome_cx"], "julia_z_cy": jrew["outcome_cy"],
                            "julia_z_fw": jrew["outcome_fw"],
                            "k3": jrew["reward_k3"], "raw_top3": jrew["raw_top3"],
                            "reached_depth": jrew["reached_depth"],
                            "decoded_class": jdecoded,
                            "p_notbad": jrew["p_notbad"], "p_good": jrew["p_good"],
                            "t_good": jt_good,
                            "distinct": jdistinct, "dup_of": jdup_of,
                            "guard_pass": jguard_pass,
                            "guard_fail": None if jguard_pass else "sentinel",
                        }
                        # commit EVERY Julia walk (feeds the v6 pool + record), mirroring
                        # the parent loop committing all outcomes.
                        ledgers.append_outcome(jrow, jfeat)
                        if jis_q3:
                            totals["julia_q3"] += 1
                        if jis_q3 and jdistinct:
                            julia_cloud.append(jrow)
                            julia_added_this_run += 1
                            totals["julia_distinct"] += 1; b_jdist += 1
                            distinct_tiles.append(
                                (jtile, f"{joid[-6:]} k3={jrew['reward_k3']:.2f} Jq3 {jmode}"))
        ledgers.save_feats()
        totals["harvested_distinct"] += b_distinct
        totals["q3_dup"] += b_q3dup
        totals["not_q3"] += b_notq3
        totals["guarded"] += b_guarded

        dt = time.time() - tb
        batch_timings.append(dt)
        el_min = (time.time() - t0) / 60
        rej_frac = native.rejects / max(1, native.draws)
        jinfo = (f"| julia desc={b_jdesc} jq3+{b_jdist} jcloud={len(julia_cloud)} "
                 if julia_hook else "")
        print(f"  batch {batch_i}: props={len(props)} surv={len(survivors)} walked={len(by_walk)} "
              f"| q3+{b_distinct} q3dup+{b_q3dup} notq3+{b_notq3} guarded+{b_guarded} "
              f"{jinfo}| draws={native.draws} rej={native.rejects} ({rej_frac:.0%}) cloud={len(cloud)} "
              f"| {dt:.0f}s (elapsed {el_min:.1f}m)")

        if native.saturated:
            print(f"  GLOBAL SATURATION reached during batch {batch_i}; stopping cleanly.")
            break
        if budget_min and el_min >= budget_min:
            print(f"  wallclock budget {budget_min}min reached; stopping cleanly.")
            break
        if smoke and batch_i >= 3:
            break

    # ---- persist + report ----
    ledgers.save_feats()

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "guard_telemetry.jsonl", "w", encoding="utf-8") as f:
        for t in telemetry:
            f.write(json.dumps(t) + "\n")
    dropped = sum(1 for t in telemetry if t["k3_is_sentinel"])
    clean = sum(1 for t in telemetry if not t["k3_is_sentinel"] and t["frames_gated"] == 0)
    salvaged = sum(1 for t in telemetry if not t["k3_is_sentinel"] and t["frames_gated"] > 0)
    guard_telemetry = {
        "n_walks": len(telemetry), "clean_harvest": clean,
        "salvaged_harvest": salvaged, "dropped": dropped,
        "frames_gated_total": sum(t["frames_gated"] for t in telemetry),
    }
    print(f"  guard telemetry: clean={clean} salvaged={salvaged} dropped={dropped} "
          f"(over {len(telemetry)} walks; {guard_telemetry['frames_gated_total']} frames gated)")

    rej_frac = round(native.rejects / max(1, native.draws), 4)
    summary = {
        "ts": run_ts, "smoke": smoke, "wallclock_s": round(time.time() - t0, 1),
        "batches": batch_i, "batch_timings_s": [round(x, 1) for x in batch_timings],
        "config": {"family": fam.partition, "family_flags": fam.flags,
                   "reject_radius": REJECT_RADIUS, "q3_density_cap": Q3_DENSITY_CAP,
                   "max_seed_redraws": MAX_SEED_REDRAWS, "dedup_k": DEDUP_K,
                   "node_width": NODE_WIDTH, "sigma_band": SIGMA_BAND,
                   "depth": [DEPTH_MIN, DEPTH_MAX], "occ_floor": OCC_FLOOR, "black_cap": BLACK_CAP,
                   "batch_seeds": batch_seeds, "budget_min": budget_min, "scorer": SCORER_PATH,
                   "julia_hook": julia_hook,
                   "julia_walks_per_descent": JULIA_WALKS_PER_DESCENT if julia_hook else None},
        "totals": totals,
        "seeds": {"draws": native.draws, "rejected": native.rejects,
                  "rejected_fraction": rej_frac, "saturation": native.saturated},
        "q3_added_this_run": q3_added_this_run,
        "guard_telemetry": guard_telemetry,
        "cumulative": {"q3_cloud_size": len(cloud),
                       "scored_rows": ledgers.n_outcomes_logged},
    }
    if julia_hook:
        summary["julia"] = {
            "partition": julia_part, "render_family": julia_render_family,
            "descents": totals["julia_descents"], "walks": totals["julia_walks"],
            "q3": totals["julia_q3"], "distinct_added": julia_added_this_run,
            "cloud_size": len(julia_cloud),
        }
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    sheet = build_contact_sheet(
        distinct_tiles, dup_tiles, _sheet_path(run_ts),
        f"production seeder {run_ts} — {len(distinct_tiles)} new q3 (green) + "
        f"dup/non-q3/guarded (yellow)")

    print("\n=== RUN SUMMARY ===")
    print(f"  proposed={totals['proposed']} probe_rejected={totals['probe_rejected']} "
          f"walked={totals['walked']} | q3_new={totals['harvested_distinct']} "
          f"q3_dup={totals['q3_dup']} not_q3={totals['not_q3']} guarded={totals['guarded']}")
    print(f"  seeds: {native.draws} drawn, {native.rejects} rejected ({rej_frac:.1%}), "
          f"saturation={native.saturated}")
    print(f"  q3 cloud: {len(cloud)} distinct places (+{q3_added_this_run} this run)")
    if julia_hook:
        print(f"  julia-hook: {totals['julia_descents']} descents / {totals['julia_walks']} walks "
              f"-> {totals['julia_q3']} q3 ({julia_added_this_run} distinct new) "
              f"| julia cloud {len(julia_cloud)} distinct c  [partition '{julia_part}']")
    print(f"  wallclock {summary['wallclock_s']}s over {batch_i} batches")
    print(f"  ledgers -> {OUTCOME_LEDGER.name}, {OUTCOME_FEATS.name}, {PROBE_REJECTS.name}")
    print(f"  summary -> {run_dir / 'summary.json'}\n  sheet   -> {sheet}")
    if not args.keep_scratch:
        _purge_run_scratch(scratch)
    return summary


def _gather(args, fam: FamilyResolved):
    """Guard-OFF gathering harvest for one c-plane class (+ optional --julia-hook). Raw
    v5 scoring; guard would-pass verdict logged per outcome as a prior; density rejection
    keyed on decoded_class == 3 alone (guard is off); durable per-class ledger + full
    walks.jsonl; NO image/feature dumps. Byte-independent of production discovery (its
    own GATHER_DIR/<partition> subtree)."""
    smoke = args.smoke
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    class_dir = GATHER_DIR / fam.partition
    run_dir = class_dir / "runs" / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = GATHER_SCRATCH_ROOT / fam.partition / run_ts
    scratch.mkdir(parents=True, exist_ok=True)

    global Q3_DENSITY_CAP, MAX_SEED_REDRAWS, NATIVE_POOL_WALKS
    if smoke:
        Q3_DENSITY_CAP = 2
        MAX_SEED_REDRAWS = 30
        NATIVE_POOL_WALKS = 60
    batch_seeds = args.batch or (8 if smoke else BATCH_SEEDS)
    budget_min = args.budget if args.budget is not None else (0 if smoke else WALLCLOCK_BUDGET_MIN)

    print(f"=== GATHER (guard-OFF) {'SMOKE ' if smoke else ''}ts={run_ts} class={fam.partition} ===")
    print(f"family: {fam.partition}  (descend flags: {' '.join(fam.flags) or '(none: native mandelbrot)'}"
          f"; render family {fam.render_family})")
    print(f"coverage: q3-density REJECTION (decoded_class==3 only; guard OFF)  radius={REJECT_RADIUS} "
          f"cap={Q3_DENSITY_CAP} max_redraws={MAX_SEED_REDRAWS} dedup_k={DEDUP_K}  batch={batch_seeds} "
          f"budget={budget_min}min")

    # RAW (unguarded) v5 — degenerate frames survive scoring so v6 gets the OOD cases.
    # The reframe guard-field hook stays OFF (raw scorer ignores sidecars; skip the cost).
    reframe.DUMP_GUARD_FIELD = False
    scorer = make_raw_scorer(SCORER_PATH)
    print(f"scorer: RAW CORN ({SCORER_PATH}, {SCORER_VERSION})  geometry={scorer.cfg.get('geometry')}  (guard OFF)")
    print(f"guard-verdict prior (logged, NOT gated): interior_frac>={guard.INTERIOR_CAP} | "
          f"field_std<{guard.FIELD_STD_FLOOR} @ {guard.GUARD_STAT_RES}")

    ledger = GatherLedger(class_dir)
    cloud = build_cloud(ledger.rows, fam.partition)
    print(f"gather ledger: {len(ledger.rows)} rows in '{fam.partition}'  | q3 cloud "
          f"{len(cloud)} distinct places (decoded_class==3)")

    rng = np.random.default_rng(args.seed)
    native = NativeSeeder(args.seed, scratch, rng, fam.flags)
    loc_of = make_loc_of(fam.render_family, fam.c)

    # --- Julia sub-descent hook (parent gate stays ON; loosened degree-aware bands) ---
    julia_hook = bool(args.julia_hook)
    if julia_hook:
        julia_render_family = "julia" if fam.partition == "mandelbrot" else f"julia_{fam.partition}"
        julia_part = julia_partition(fam.partition)
        julia_cloud = build_cloud(ledger.rows, julia_part)
        j_esc, j_spread = JULIA_GATHER_BANDS[fam.partition]
        print(f"julia-hook: ON  partition '{julia_part}' render_family={julia_render_family} "
              f"walks/descent={JULIA_WALKS_PER_DESCENT}  bands: esc_median_min={j_esc} "
              f"spread_min={j_spread}  | julia q3 cloud {len(julia_cloud)} distinct c")
    julia_added_this_run = 0

    totals = {"outcomes": 0, "decoded": {1: 0, 2: 0, 3: 0}, "q3_distinct": 0, "walk_errors": 0,
              "guard": {"pass": 0, "interior": 0, "flat": 0, "both": 0, "render_error": 0},
              "julia_descents": 0, "julia_outcomes": 0, "julia_q3": 0, "julia_distinct": 0,
              "julia_guard": {"pass": 0, "interior": 0, "flat": 0, "both": 0, "render_error": 0}}
    gf_dir = scratch / "guard_fields"
    t0 = time.time()
    seq = 0
    batch_i = 0
    walks_fh = open(run_dir / "walks.jsonl", "w", encoding="utf-8")

    while True:
        batch_i += 1
        # Engine subprocess spawns (native draw / depth-2 probe / full walks) are the
        # first thing to fail under system resource exhaustion (they can't spawn
        # render-one/guided-descend). A failure here must END THE CLASS CLEANLY — write
        # the summary and exit 0 — NOT hard-crash: a dirty crash mid-GPU-work wedges CUDA
        # and cascades DLL-init failures into every subsequent class. The next class then
        # starts as a fresh process with all memory reclaimed.
        try:
            props = native.draw_batch(cloud, batch_seeds)
            if not props:
                print("  GLOBAL SATURATION; stopping." if native.saturated
                      else "  native seed source produced no proposals; stopping.")
                break

            pw = scratch / f"batch_{batch_i:03d}" / "probe"
            survivors, rejects, pcauses = depth2_probe(props, pw, args.seed, fam.flags)
            if not survivors:
                print(f"  batch {batch_i}: 0/{len(props)} descendable (causes {pcauses}); next batch.")
                if native.saturated:
                    break
                if budget_min and (time.time() - t0) / 60 >= budget_min:
                    break
                if smoke:
                    break
                continue

            ww = scratch / f"batch_{batch_i:03d}" / "walks"
            pool = run_full_walks(survivors, ww, args.seed, fam.flags)
            by_walk = load_frames_by_walk(pool)
            wmeta = load_walk_meta(pool)
        except (SystemExit, Exception) as e:
            totals["engine_errors"] = totals.get("engine_errors", 0) + 1
            print(f"  ENGINE FAILURE batch {batch_i} (likely resource exhaustion); ending class "
                  f"cleanly to avoid a dirty-crash CUDA wedge: {type(e).__name__}: {str(e)[:160]}")
            break

        b_out = b_q3 = 0
        for wid in sorted(by_walk):
          # One walk's reward/render can hit a transient render failure; it must NOT
          # abort a multi-hour class (a hard crash mid-GPU-work also wedges the next
          # class's CUDA init). Log, count, and move on — the ledger is per-outcome so
          # nothing already committed is lost.
          try:
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS,
                                      scratch / f"reward_b{batch_i:03d}", loc_of)
            t_good = t_good_for(fam.partition)   # per-degree q3 operating point
            decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good)
            oid = f"{fam.id_tag}_{run_ts}_{seq:06d}"; seq += 1
            verdict, gstats = outcome_guard_verdict(
                rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"],
                gf_dir / f"{oid}.field.bin", family=fam.render_family, c=fam.c)
            is_q3 = decoded == 3
            distinct, dup_of = (is_distinct(rew["outcome_cx"], rew["outcome_cy"],
                                            rew["outcome_fw"], cloud, DEDUP_K)
                                if is_q3 else (False, None))
            sv = survivors[wid] if wid < len(survivors) else survivors[-1]
            row = {
                "id": oid, "ts": run_ts, "family": fam.partition, "descend_mode": "cplane",
                "parent_oid": None, "mix_source": sv["mix_source"],
                "seed_cx": sv["seed_cx"], "seed_cy": sv["seed_cy"],
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"],
                "raw_top3": rew["raw_top3"], "reached_depth": rew["reached_depth"],
                "decoded_class": decoded,
                "p_notbad": rew["p_notbad"], "p_good": rew["p_good"], "t_good": t_good,
                # guard would-pass verdict — a recorded PRIOR, never a gate in gather.
                "guard_verdict": verdict, "guard_pass": verdict == "pass",
                "interior_frac": gstats.interior_frac, "field_std": gstats.field_std,
                "distinct": distinct, "dup_of": dup_of,
            }
            ledger.append(row)
            persist_walk(walks_fh, family=fam.partition, batch=batch_i, walk=wid,
                         outcome_id=oid, frames=frames, meta=wmeta.get(wid),
                         descend_mode="cplane")
            totals["outcomes"] += 1; b_out += 1
            totals["decoded"][decoded] += 1
            totals["guard"][verdict] += 1
            if is_q3 and distinct:
                cloud.append(row); totals["q3_distinct"] += 1; b_q3 += 1

            # --- Julia sub-descent hook (parent gate ON; loosened degree-aware bands) ---
            if julia_hook:
                qualifies = rew["n_frames_q2plus"] >= 2 or rew["n_frames_q3"] >= 1
                jc = (rew["outcome_cx"], rew["outcome_cy"])
                jc_fw = rew["outcome_fw"]
                dense = count_within(julia_cloud, jc[0], jc[1], REJECT_RADIUS) >= Q3_DENSITY_CAP
                if qualifies and not dense:
                    totals["julia_descents"] += 1
                    jmode = "center" if rng.random() < 0.5 else "normal"
                    jseed = int(hashlib.md5(f"{oid}:{run_ts}".encode()).hexdigest()[:8], 16)
                    jwork = scratch / f"batch_{batch_i:03d}" / "julia" / f"w{wid:04d}"
                    jpool = run_julia_descent(jc, jmode, jseed, jwork, JULIA_WALKS_PER_DESCENT,
                                              family=fam.partition,
                                              esc_median_min=j_esc, spread_min=j_spread)
                    jloc_of = make_loc_of(julia_render_family, (str(jc[0]), str(jc[1])))
                    jwmeta = load_walk_meta(jpool)
                    for jwid, jframes in sorted(load_frames_by_walk(jpool).items()):
                        jrew = harvest_walk_reward(
                            scorer, jwid, jframes, WORKERS,
                            scratch / f"jreward_b{batch_i:03d}_w{wid:04d}", jloc_of)
                        jt_good = t_good_for(julia_part)   # per-degree q3 operating point
                        jdecoded = corn_decode(jrew["p_notbad"], jrew["p_good"], jt_good)
                        joid = f"j{fam.id_tag}_{run_ts}_{seq:06d}"; seq += 1
                        jverdict, jgstats = outcome_guard_verdict(
                            jrew["outcome_cx"], jrew["outcome_cy"], jrew["outcome_fw"],
                            gf_dir / f"{joid}.field.bin", family=julia_render_family,
                            c=(str(jc[0]), str(jc[1])))
                        jis_q3 = jdecoded == 3
                        jdistinct, jdup_of = (is_distinct(jc[0], jc[1], jc_fw, julia_cloud, DEDUP_K)
                                              if jis_q3 else (False, None))
                        jrow = {
                            "id": joid, "ts": run_ts, "family": julia_part,
                            "mix_source": "julia_hook", "parent_oid": oid,
                            "descend_mode": jmode,
                            "outcome_cx": jc[0], "outcome_cy": jc[1], "outcome_fw": jc_fw,
                            "julia_z_cx": jrew["outcome_cx"], "julia_z_cy": jrew["outcome_cy"],
                            "julia_z_fw": jrew["outcome_fw"],
                            "k3": jrew["reward_k3"], "raw_top3": jrew["raw_top3"],
                            "reached_depth": jrew["reached_depth"], "decoded_class": jdecoded,
                            "p_notbad": jrew["p_notbad"], "p_good": jrew["p_good"],
                            "t_good": jt_good,
                            "guard_verdict": jverdict, "guard_pass": jverdict == "pass",
                            "interior_frac": jgstats.interior_frac, "field_std": jgstats.field_std,
                            "distinct": jdistinct, "dup_of": jdup_of,
                        }
                        ledger.append(jrow)
                        persist_walk(walks_fh, family=julia_part, batch=batch_i, walk=jwid,
                                     outcome_id=joid, frames=jframes, meta=jwmeta.get(jwid),
                                     parent_oid=oid, descend_mode=jmode)
                        totals["julia_outcomes"] += 1
                        totals["julia_guard"][jverdict] += 1
                        if jis_q3:
                            totals["julia_q3"] += 1
                        if jis_q3 and jdistinct:
                            julia_cloud.append(jrow); julia_added_this_run += 1
                            totals["julia_distinct"] += 1
          except (SystemExit, Exception) as e:   # render SystemExit OR torch CUDA/OOM RuntimeError
            totals["walk_errors"] += 1
            print(f"  WARN walk b{batch_i:03d}_w{wid:04d} skipped ({totals['walk_errors']} total): "
                  f"{type(e).__name__}: {str(e)[:140]}")
            continue

        el_min = (time.time() - t0) / 60
        jinfo = (f"| julia desc={totals['julia_descents']} jout={totals['julia_outcomes']} "
                 f"jcloud={len(julia_cloud)} " if julia_hook else "")
        print(f"  batch {batch_i}: props={len(props)} surv={len(survivors)} walked={len(by_walk)} "
              f"| out+{b_out} q3+{b_q3} {jinfo}| cloud={len(cloud)} rej={native.rejects}/{native.draws} "
              f"| elapsed {el_min:.1f}m")

        if native.saturated:
            print(f"  GLOBAL SATURATION during batch {batch_i}; stopping.")
            break
        if budget_min and el_min >= budget_min:
            print(f"  wallclock budget {budget_min}min reached; stopping.")
            break
        if smoke and batch_i >= 2:
            break

    walks_fh.close()
    summary = {
        "ts": run_ts, "mode": "gather", "smoke": smoke, "family": fam.partition,
        "wallclock_s": round(time.time() - t0, 1), "batches": batch_i,
        "config": {"family": fam.partition, "family_flags": fam.flags,
                   "reject_radius": REJECT_RADIUS, "q3_density_cap": Q3_DENSITY_CAP,
                   "dedup_k": DEDUP_K, "depth": [DEPTH_MIN, DEPTH_MAX], "batch_seeds": batch_seeds,
                   "budget_min": budget_min, "scorer": SCORER_PATH, "guard": "OFF (verdict logged as prior)",
                   "julia_hook": julia_hook,
                   "julia_bands": JULIA_GATHER_BANDS[fam.partition] if julia_hook else None},
        "totals": totals,
        "seeds": {"draws": native.draws, "rejected": native.rejects, "saturation": native.saturated},
        "cumulative": {"ledger_rows": len(ledger.rows), "q3_cloud_size": len(cloud)},
    }
    if julia_hook:
        summary["julia"] = {"partition": julia_part, "cloud_size": len(julia_cloud),
                            "distinct_added": julia_added_this_run}
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))

    g = totals["guard"]
    print("\n=== GATHER SUMMARY ===")
    print(f"  class {fam.partition}: {totals['outcomes']} c-plane outcomes  "
          f"decoded 1/2/3={totals['decoded'][1]}/{totals['decoded'][2]}/{totals['decoded'][3]}  "
          f"q3_distinct={totals['q3_distinct']}")
    print(f"  guard verdict (prior): pass={g['pass']} interior={g['interior']} flat={g['flat']} "
          f"both={g['both']} render_error={g['render_error']}  | walk_errors={totals['walk_errors']}")
    if julia_hook:
        jg = totals["julia_guard"]
        print(f"  julia-hook: {totals['julia_descents']} descents -> {totals['julia_outcomes']} outcomes "
              f"(q3={totals['julia_q3']}, distinct+{totals['julia_distinct']})  "
              f"guard pass={jg['pass']} int={jg['interior']} flat={jg['flat']} both={jg['both']} "
              f"render_error={jg['render_error']}")
    print(f"  wallclock {summary['wallclock_s']}s over {batch_i} batches")
    print(f"  ledger -> {ledger.path}")
    print(f"  walks  -> {run_dir / 'walks.jsonl'}\n  summary-> {run_dir / 'summary.json'}")
    return summary


def _gather_phoenix(args):
    """Guard-OFF gathering harvest for Phoenix — the single fixed Ushiki location (no
    parameter plane, so no density rejection, no --julia-hook). Repeatedly descends the
    z-plane time-boxed and logs every outcome with its guard would-pass verdict prior."""
    smoke = args.smoke
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    class_dir = GATHER_DIR / "phoenix"
    run_dir = class_dir / "runs" / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = GATHER_SCRATCH_ROOT / "phoenix" / run_ts
    scratch.mkdir(parents=True, exist_ok=True)
    budget_min = args.budget if args.budget is not None else (0 if smoke else WALLCLOCK_BUDGET_MIN)
    walks_per = args.batch or (3 if smoke else 12)   # native --phoenix walks per descent round

    print(f"=== GATHER (guard-OFF) {'SMOKE ' if smoke else ''}ts={run_ts} class=phoenix ===")
    print(f"phoenix native (fixed Ushiki location; no parameter plane -> no rejection/hook)  "
          f"walks/round={walks_per} budget={budget_min}min")

    reframe.DUMP_GUARD_FIELD = False
    scorer = make_raw_scorer(SCORER_PATH)
    print(f"scorer: RAW CORN ({SCORER_PATH}, {SCORER_VERSION})  (guard OFF; verdict logged as prior)")

    ledger = GatherLedger(class_dir)
    loc_of = make_loc_of("phoenix", None)
    gf_dir = scratch / "guard_fields"

    totals = {"outcomes": 0, "decoded": {1: 0, 2: 0, 3: 0}, "walk_errors": 0,
              "guard": {"pass": 0, "interior": 0, "flat": 0, "both": 0, "render_error": 0}}
    t0 = time.time()
    seq = 0
    rnd = 0
    walks_fh = open(run_dir / "walks.jsonl", "w", encoding="utf-8")

    while True:
        rnd += 1
        try:
            pool = run_phoenix_descent(args.seed + rnd, scratch / f"round_{rnd:03d}", walks_per)
            by_walk = load_frames_by_walk(pool)
            wmeta = load_walk_meta(pool)
        except (SystemExit, Exception) as e:
            totals["engine_errors"] = totals.get("engine_errors", 0) + 1
            print(f"  ENGINE FAILURE round {rnd} (likely resource exhaustion); ending cleanly: "
                  f"{type(e).__name__}: {str(e)[:160]}")
            break
        b_out = 0
        for wid in sorted(by_walk):
          # A transient render failure must not abort a multi-hour class (see _gather).
          try:
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS,
                                      scratch / f"reward_r{rnd:03d}", loc_of)
            t_good = t_good_for("phoenix")   # phoenix -> baseline (not in the swept eval)
            decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good)
            oid = f"ph_{run_ts}_{seq:06d}"; seq += 1
            verdict, gstats = outcome_guard_verdict(
                rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"],
                gf_dir / f"{oid}.field.bin", family="phoenix", c=None)
            row = {
                "id": oid, "ts": run_ts, "family": "phoenix", "descend_mode": "phoenix",
                "parent_oid": None,
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"],
                "raw_top3": rew["raw_top3"], "reached_depth": rew["reached_depth"],
                "decoded_class": decoded,
                "p_notbad": rew["p_notbad"], "p_good": rew["p_good"], "t_good": t_good,
                "guard_verdict": verdict, "guard_pass": verdict == "pass",
                "interior_frac": gstats.interior_frac, "field_std": gstats.field_std,
            }
            ledger.append(row)
            persist_walk(walks_fh, family="phoenix", batch=rnd, walk=wid, outcome_id=oid,
                         frames=frames, meta=wmeta.get(wid), descend_mode="phoenix")
            totals["outcomes"] += 1; b_out += 1
            totals["decoded"][decoded] += 1
            totals["guard"][verdict] += 1
          except (SystemExit, Exception) as e:   # render SystemExit OR torch CUDA/OOM RuntimeError
            totals["walk_errors"] += 1
            print(f"  WARN phoenix round {rnd} walk {wid} skipped "
                  f"({totals['walk_errors']} total): {type(e).__name__}: {str(e)[:140]}")
            continue
        el_min = (time.time() - t0) / 60
        print(f"  round {rnd}: walked={len(by_walk)} out+{b_out} "
              f"decoded 1/2/3={totals['decoded'][1]}/{totals['decoded'][2]}/{totals['decoded'][3]} "
              f"| elapsed {el_min:.1f}m")
        if budget_min and el_min >= budget_min:
            print(f"  wallclock budget {budget_min}min reached; stopping.")
            break
        if smoke and rnd >= 1:
            break

    walks_fh.close()
    g = totals["guard"]
    summary = {
        "ts": run_ts, "mode": "gather", "smoke": smoke, "family": "phoenix",
        "wallclock_s": round(time.time() - t0, 1), "rounds": rnd,
        "config": {"family": "phoenix", "walks_per_round": walks_per, "budget_min": budget_min,
                   "depth": [DEPTH_MIN, DEPTH_MAX], "scorer": SCORER_PATH,
                   "guard": "OFF (verdict logged as prior)"},
        "totals": totals, "cumulative": {"ledger_rows": len(ledger.rows)},
    }
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    print("\n=== GATHER SUMMARY (phoenix) ===")
    print(f"  {totals['outcomes']} outcomes  decoded 1/2/3="
          f"{totals['decoded'][1]}/{totals['decoded'][2]}/{totals['decoded'][3]}")
    print(f"  guard verdict (prior): pass={g['pass']} interior={g['interior']} flat={g['flat']} "
          f"both={g['both']} render_error={g['render_error']}  | walk_errors={totals['walk_errors']}")
    print(f"  wallclock {summary['wallclock_s']}s over {rnd} rounds")
    print(f"  ledger -> {ledger.path}")
    print(f"  walks  -> {run_dir / 'walks.jsonl'}\n  summary-> {run_dir / 'summary.json'}")
    return summary


def _run_phoenix(args):
    """Fresh-discovery Phoenix z-plane descent -> the RUN-SCOPED discovery ledger.

    The phoenix analogue of `_run`. Phoenix has no parameter plane, so it cannot ride the
    c-plane spatial radius-rejection loop (resolve_family rejects `--phoenix` under `--run`);
    instead it repeatedly descends the single fixed Ushiki z-plane and scores each outcome
    with the SAME guarded v6 scorer + per-degree t_good the c-plane `_run` uses. Every
    scored outcome is appended to the run-scoped OUTCOME_LEDGER (redirected by
    --discovery-dir) with the c-plane row schema (id / family=phoenix / decoded_class /
    p_good / t_good / guard_pass, + scorer_version="v6" stamped by append_outcome), so the
    orchestrator's per-cycle watermark (new_fresh_q3) and fresh-isolation assertion admit
    phoenix rows exactly like the other families and build_fresh_discovery renders them as
    the 9th family (gs.render_family("phoenix") -> outcome viewport, no c).

    Writes FRESH to the run-scoped ledger — NEVER the banked GATHER_DIR/phoenix subtree
    (reusing banked q3s would violate the fresh-generation precondition). Phoenix is a
    low-yield, variety-poor garnish (~7 descents / keeper at t_good=0.18), so the
    orchestrator gives this phase an elevated per-cycle descent budget; dominance is capped
    downstream by build_fresh_discovery's family-balanced round-robin.

    Bounded by --budget (minutes) and/or --phoenix-walks (total-walk cap; 0 = budget-only).
    The walk cap is the deterministic lever the mini-run uses to guarantee >=1 keeper."""
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = SCRATCH_ROOT / run_ts
    scratch.mkdir(parents=True, exist_ok=True)
    budget_min = args.budget if args.budget is not None else WALLCLOCK_BUDGET_MIN
    walks_per = args.batch or 12               # native --phoenix walks per descent round
    walks_cap = args.phoenix_walks or 0        # 0 -> budget-only (prod); >0 caps total (mini)

    print(f"=== atlas production seeder (RUN-PHOENIX) ts={run_ts} ===")
    print(f"phoenix native (fixed Ushiki z-plane; no parameter plane -> no rejection/hook)  "
          f"walks/round={walks_per} budget={budget_min}min walks_cap={walks_cap or 'none'}")

    # Guarded scorer — byte-identical scoring path to the c-plane _run (degenerate crops
    # -> GUARD_SENTINEL; guard_pass = k3 > sentinel), so phoenix q3s are guard-clean.
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX, (
        f"guard field suffix drift: reframe {reframe.GUARD_FIELD_SUFFIX!r} != "
        f"guard {guard.FIELD_SIDECAR_SUFFIX!r}")
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    t_good = t_good_for("phoenix")
    print(f"scorer: GUARDED CORN ({SCORER_PATH}, {SCORER_VERSION})  t_good={t_good}")
    print(f"run ledger (run-scoped): {OUTCOME_LEDGER}")

    ledgers = Ledgers()                        # OUTCOME_LEDGER is the fresh run-scoped file
    loc_of = make_loc_of("phoenix", None)

    totals = {"walked": 0, "q3": 0, "not_q3": 0, "guarded": 0, "walk_errors": 0}
    t0 = time.time()
    seq = 0
    rnd = 0
    while True:
        rnd += 1
        try:
            pool = run_phoenix_descent(args.seed + rnd, scratch / f"round_{rnd:03d}", walks_per)
            by_walk = load_frames_by_walk(pool)
        except (SystemExit, Exception) as e:
            totals["engine_errors"] = totals.get("engine_errors", 0) + 1
            print(f"  ENGINE FAILURE round {rnd} (likely resource exhaustion); ending cleanly: "
                  f"{type(e).__name__}: {str(e)[:160]}")
            break
        b_q3 = 0
        for wid in sorted(by_walk):
          # A transient render / CUDA failure must not abort the phase (mirrors _run).
          try:
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS,
                                      scratch / f"reward_r{rnd:03d}", loc_of)
            # Guard verdict: k3 collapses to the sentinel iff EVERY framing of the top-3
            # failed the guard. A guarded outcome carries decoded_class=None (never q3).
            guard_pass = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
            decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good) if guard_pass else None
            is_q3 = guard_pass and decoded == 3
            oid = f"ph_{run_ts}_{seq:06d}"; seq += 1
            row = {
                "id": oid, "ts": run_ts, "family": "phoenix", "descend_mode": "phoenix",
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"],
                "raw_top3": rew["raw_top3"], "reached_depth": rew["reached_depth"],
                "decoded_class": decoded,
                "p_notbad": rew["p_notbad"], "p_good": rew["p_good"], "t_good": t_good,
                # No spatial cloud for a single fixed plane: `distinct` mirrors is_q3 for
                # schema parity; near-dup pile-up is collapsed downstream by
                # build_fresh_discovery's key-dedup + family-balanced round-robin.
                "distinct": is_q3, "dup_of": None,
                "guard_pass": guard_pass, "guard_fail": None if guard_pass else "sentinel",
            }
            ledgers.append_outcome(row, None)   # stamps scorer_version="v6"; no feature store
            totals["walked"] += 1
            if is_q3:
                totals["q3"] += 1; b_q3 += 1
            elif not guard_pass:
                totals["guarded"] += 1
            else:
                totals["not_q3"] += 1
          except (SystemExit, Exception) as e:
            totals["walk_errors"] += 1
            print(f"  WARN phoenix round {rnd} walk {wid} skipped "
                  f"({totals['walk_errors']} total): {type(e).__name__}: {str(e)[:140]}")
            continue
        el_min = (time.time() - t0) / 60
        print(f"  round {rnd}: walked={len(by_walk)} q3+{b_q3}  "
              f"(cum walked={totals['walked']} q3={totals['q3']}) | elapsed {el_min:.1f}m")
        if walks_cap and totals["walked"] >= walks_cap:
            print(f"  walks cap {walks_cap} reached ({totals['walked']} walked); stopping.")
            break
        if budget_min and el_min >= budget_min:
            print(f"  wallclock budget {budget_min}min reached; stopping.")
            break

    yld = totals["q3"] / totals["walked"] if totals["walked"] else 0.0
    per_keeper = (1.0 / yld) if yld else float("inf")
    summary = {
        "ts": run_ts, "mode": "run-phoenix", "family": "phoenix",
        "wallclock_s": round(time.time() - t0, 1), "rounds": rnd, "t_good": t_good,
        "walks_per_round": walks_per, "walks_cap": walks_cap,
        "totals": totals, "yield_q3_per_walk": round(yld, 4),
        "run_ledger": str(OUTCOME_LEDGER),
    }
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    print("\n=== RUN-PHOENIX SUMMARY ===")
    print(f"  walked={totals['walked']} q3={totals['q3']} not_q3={totals['not_q3']} "
          f"guarded={totals['guarded']} walk_errors={totals['walk_errors']}")
    print(f"  yield {yld:.3f} q3/walk (~{per_keeper:.1f} descents/keeper) @ t_good={t_good}")
    print(f"  ledger -> {OUTCOME_LEDGER}\n  summary -> {run_dir / 'summary.json'}")
    if not args.keep_scratch:
        _purge_run_scratch(scratch)
    return summary


def _finalize(run_ts: str):
    """Rebuild a run's summary.json + contact_sheet.png from the DURABLE ledger + the
    on-disk outcome tiles. The ledger is written per-outcome, so a kill in the cosmetic
    final stage loses no data — this reconstructs the missing cosmetic artifacts. The q3
    cloud is reconstructed from outcome_ledger.jsonl (there are no cells to rebuild)."""
    run_dir = RUNS_DIR / run_ts
    tiles_dir = SCRATCH_ROOT / run_ts / "outcome_tiles"
    all_rows = [json.loads(l) for l in open(OUTCOME_LEDGER, encoding="utf-8") if l.strip()]
    rows = [r for r in all_rows if r.get("ts") == run_ts]
    if not rows:
        raise SystemExit(f"no outcome rows with ts={run_ts} in {OUTCOME_LEDGER}")
    # A run is single-family; infer its partition from the rows so the reconstructed cloud
    # is that family's (rows missing `family` are pre-grammar Mandelbrot).
    partition = rows[0].get("family", "mandelbrot")
    cloud = build_cloud(all_rows, partition)
    distinct = [r for r in rows if r.get("distinct")]
    dup = [r for r in rows if not r.get("distinct")]
    totals = {"walked": len(rows), "harvested_distinct": len(distinct),
              "q3_dup": sum(1 for r in rows if not r.get("distinct")
                            and r.get("guard_pass", True) and r.get("decoded_class") == 3),
              "not_q3": sum(1 for r in rows if r.get("guard_pass", True)
                            and r.get("decoded_class") not in (None, 3)),
              "guarded": sum(1 for r in rows if not r.get("guard_pass", True))}
    summary = {"ts": run_ts, "finalized_from_ledger": True, "family": partition, "totals": totals,
               "cumulative": {"q3_cloud_size": len(cloud), "scored_rows": len(all_rows)},
               "config": {"family": partition, "reject_radius": REJECT_RADIUS,
                          "q3_density_cap": Q3_DENSITY_CAP,
                          "dedup_k": DEDUP_K, "scorer": SCORER_PATH}}
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    dtiles = [(tiles_dir / f"{r['id']}.jpg", f"{r['id'][-6:]} k3={r['k3']:.2f} q3")
              for r in sorted(distinct, key=lambda r: -r["k3"])]
    utiles = [(tiles_dir / f"{r['id']}.jpg", f"k3={r['k3']:.2f}->dup") for r in dup[:NCOL_DUP]]
    sheet = build_contact_sheet(dtiles, utiles, _sheet_path(run_ts),
                                f"production seeder {run_ts} (finalized) — {len(distinct)} new q3 "
                                f"(green) + dup/non-q3/guarded (yellow)")
    print(f"finalized run {run_ts}: {len(distinct)} new q3 / {len(dup)} other  "
          f"| q3 cloud {len(cloud)} distinct places")
    print(f"  summary -> {run_dir / 'summary.json'}\n  sheet   -> {sheet}")
    return summary


def _time_only(args, fam: FamilyResolved):
    """Project per-batch wallclock from one native-gen + one small batch."""
    args.smoke = True
    args.batch = args.batch or 12
    print("(--time-only: running one smoke batch to project per-batch cost)")
    t = time.time()
    _run(args, fam)
    print(f"\n  one smoke batch (native-gen + probe + walks + reward) took {time.time()-t:.0f}s")
    print(f"  a 30-min run fits ~{int(30*60/max(1,(time.time()-t))):d} batches of this size "
          f"(minus one-time native-gen).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="~20-seed run, lowered knobs so rejection fires")
    ap.add_argument("--run", action="store_true", help="30-min time-boxed production run")
    ap.add_argument("--gather", action="store_true",
                    help="guard-OFF oversampling harvest for the v6 label pass: raw v5 scoring "
                         "(degenerate frames survive), guard would-pass verdict logged per outcome "
                         "as a PRIOR (not a gate), density rejection keyed on decoded_class==3 alone, "
                         "durable per-class GATHER_DIR/<class> ledger + full walks.jsonl, NO image/"
                         "feature dumps. Honors --family / --julia-hook / --phoenix / --budget / --smoke.")
    ap.add_argument("--run-phoenix", action="store_true",
                    help="fresh-discovery Phoenix z-plane descent -> the run-scoped ledger "
                         "(pair with --discovery-dir). The phoenix analogue of --run: guarded "
                         "v6 scoring at t_good=0.18, run-scoped OUTCOME_LEDGER, honors --budget "
                         "and --phoenix-walks. NEVER touches the banked gather/phoenix subtree.")
    ap.add_argument("--phoenix-walks", type=int, default=0,
                    help="cap total phoenix walks (outcomes) for --run-phoenix (0 = budget-only). "
                         "The deterministic knob the mini-run uses to guarantee >=1 keeper.")
    ap.add_argument("--time-only", action="store_true", help="project per-batch wallclock")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="keep the transient render scratch (reward_*/walk_*/batch_*/round_* under "
                         "out/atlas/production_seeder/<ts>/). Default: purge it on clean exit — it is "
                         "scoring intermediate (~GBs/run) that nothing downstream reads; the durable "
                         "ledger/feats/summary live elsewhere. Pass this to debug a run's frames or to "
                         "preserve outcome_tiles for a later --finalize of a run that finished cleanly.")
    ap.add_argument("--finalize", metavar="RUN_TS", default=None,
                    help="rebuild summary + contact sheet for a run from the durable ledger")
    ap.add_argument("--seed", type=int, default=0, help="rng + engine seed")
    ap.add_argument("--batch", type=int, default=0, help="seeds per batch (0 = default)")
    ap.add_argument("--budget", type=float, default=None, help="wallclock budget minutes override")
    ap.add_argument("--discovery-dir", type=Path, default=None,
                    help="redirect the durable discovery store (outcome_ledger.jsonl, "
                         "outcome_feats.npz, probe_rejects.jsonl, runs/) to this dir instead of "
                         "data/discovery. Point at a FRESH, EMPTY dir for a run-scoped ledger so a "
                         "downstream pool build (build_fresh_discovery --ledger) reads ONLY this "
                         "run's fresh q3s — the fresh-generation precondition. The q3 rejection "
                         "cloud is rebuilt from THIS dir, so cross-run repulsion is intentionally "
                         "reset to the run's own accumulating cloud.")
    # --- family grammar (mirrors render_one / guided_descend; see resolve_family) ---
    ap.add_argument("--family", default="mandelbrot",
                    choices=["mandelbrot", "multibrot3", "multibrot4", "multibrot5"],
                    help="parameter-plane escape family (default mandelbrot). Multibrot 3/4/5 "
                         "ride the engine's per-degree bands + degree_bbox flat-box.")
    ap.add_argument("--julia", action="store_true",
                    help="dynamical z-plane at a fixed c (requires --c; pairs with mandelbrot or "
                         "multibrot{d}; PLUMBED, not yet validated). Rejected with --phoenix.")
    ap.add_argument("--c", nargs=2, metavar=("RE", "IM"), default=None,
                    help=f"fixed dynamical parameter (Julia). Shakeout anchor: {JULIA_C[0]} {JULIA_C[1]}")
    ap.add_argument("--phoenix", action="store_true",
                    help="recognized but rejected: Phoenix is a single fixed location with no "
                         "parameter plane to prospect (descend it directly via guided-descend).")
    ap.add_argument("--julia-hook", action="store_true",
                    help="for each qualifying parameter-plane outcome c, run a native --julia "
                         "descent at that c, score it, and commit q3 results as found-points in "
                         "a degree-only julia:{fam} partition (strictly additive; default off — "
                         "c-plane runs are byte-unchanged when off).")
    args = ap.parse_args()
    if args.discovery_dir is not None:
        # Run-scoped store redirect: rebind the durable-discovery globals BEFORE any
        # dispatch so Ledgers/append_outcome/_finalize/RUNS_DIR all target the fresh dir.
        global DISCOVERY_DIR, OUTCOME_LEDGER, OUTCOME_FEATS, PROBE_REJECTS, RUNS_DIR
        DISCOVERY_DIR = args.discovery_dir.resolve()
        OUTCOME_LEDGER = DISCOVERY_DIR / "outcome_ledger.jsonl"
        OUTCOME_FEATS = DISCOVERY_DIR / "outcome_feats.npz"
        PROBE_REJECTS = DISCOVERY_DIR / "probe_rejects.jsonl"
        RUNS_DIR = DISCOVERY_DIR / "runs"
        DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[seeder] discovery store -> {DISCOVERY_DIR}  (run-scoped ledger; "
              f"cloud rebuilt from this dir only)")
    if args.finalize:
        _finalize(args.finalize)
    elif args.run_phoenix:
        # Fresh-discovery phoenix z-descent -> run-scoped ledger (the wired production path;
        # resolve_family still rejects --phoenix under --run/--gather's c-plane loop).
        _run_phoenix(args)
    elif args.gather:
        # Phoenix has no parameter plane (resolve_family rejects it): route it to the
        # dedicated single-location gather path. Every other family resolves normally.
        if args.phoenix:
            _gather_phoenix(args)
        else:
            _gather(args, resolve_family(args))
    elif args.time_only:
        _time_only(args, resolve_family(args))
    elif args.smoke or args.run:
        _run(args, resolve_family(args))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
