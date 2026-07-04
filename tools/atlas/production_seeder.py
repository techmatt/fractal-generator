#!/usr/bin/env python
r"""Atlas production discovery seeder (Mandelbrot) — the standing discovery flow.

Wires the validated round-2 atlas proposer into one time-boxed production loop
(prompts/cc-prompt-atlas-production-seeder.md). Per batch:

  propose (native / exploit / explore mix)
    -> seed-launch cell cap filter (+ backfill to explore)
    -> depth-2 descendability probe  (reuse propose.prescreen, verbatim engine step-1)
    -> full guided-descend walks      (--seed-list --per-walk-rng, production config)
    -> k3 best-frame reward + outcome center + 1280-D v5 penultimate feature
    -> location-space distinct-outcome cap (fw-relative dedup) + harvest
    -> update both durable ledgers atomically

Two independent throttles on two different spaces (neither substitutes the other):
  * SEED-LAUNCH cell cap  (pre-run, compute economy + forced exploration): reject a
    proposal whose coverage cell already has >= SEED_LAUNCH_CAP launches.
  * LOCATION distinct-outcome cap (post-run, diversity of place): an outcome is a
    near-dup iff  dist(A,B) < DEDUP_K * max(A.fw, B.fw)  against ALL harvested
    outcomes; distinct -> harvest + bump the seed cell's distinct tally. A cell
    saturates at distinct >= OUTCOME_DISTINCT_CAP OR launches >= SEED_LAUNCH_CAP;
    a saturated cell rejects further EXPLOIT proposals -> backfill to explore.

This is v1 production wiring: NO harvest->refit loop, NO atlas refit. Everything is
assembled from existing parts; only the two ledgers + cap logic are new. The
Mandelbrot (cx,cy,fw) distance is the ONLY harvest gate — the 1280-D feature is
logged, never gates.

Reuse (located, not reinvented — see the prompt's "what already exists"):
  * guided-descend engine (Rust) w/ --seed-list --per-walk-rng      (propose.BIN)
  * atlas object                                       tools/atlas/atlas.py
  * proposer acquisition (exploit=conf*theta_norm, explore=1-conf)  tools/atlas/propose.py
  * coverage-region bin (14x12 over atlas.mask_bounds) tools/atlas/round1_analyze.py
  * depth-2 descendability pre-screen                  propose.prescreen (verbatim)
  * reframe path                                       tools/reframe/reframe.py
  * v5 scorer bridge (explicit model_path)             probe.make_scorer / score_lib.Scorer
  * k3 reward primitives                               step0_reanalysis (via round1_harvest)
  * v5 1280-D penultimate hook                         round1_embed.embed_paths / _render
  * canonical location layer                           tools/corpus/location.py

  uv run python tools/atlas/production_seeder.py --smoke     # ~20 seeds, caps forced to fire
  uv run python tools/atlas/production_seeder.py --run       # 30-min time-boxed production run
  uv run python tools/atlas/production_seeder.py --time-only # project per-batch wallclock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# reuse roots (same layering round1_harvest uses)
sys.path.insert(0, str(HERE))                                   # atlas.py, propose.py, round1_*.py
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))        # step0_reanalysis primitives
sys.path.insert(0, str(ROOT / "tools" / "reframe"))            # reframe_location
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))     # probe.make_scorer
sys.path.insert(0, str(ROOT / "tools" / "corpus"))            # location.py
sys.path.insert(0, str(ROOT / "tools" / "mining"))            # score_lib.Scorer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import propose  # noqa: E402  (Atlas, domain_cloud, fps, load_fw_pool, prescreen, write_seed_list, BIN, SCREEN_*)
from atlas import Atlas, ARTIFACT_PATH  # noqa: E402
from round1_analyze import COVER_NCOLS, COVER_NROWS  # noqa: E402  (the coverage-region bin)
import step0_reanalysis as sr  # noqa: E402  (KRAW, raw_screen_walk, _mand_location, load_frames_by_walk, _seed_rows)
from step0_reanalysis import (  # noqa: E402
    KRAW, raw_screen_walk, _mand_location, load_frames_by_walk, _seed_rows,
)
import reframe  # noqa: E402  (reframe_location + the DUMP_GUARD_FIELD hook)
from reframe import reframe_location  # noqa: E402
import round1_embed as r1e  # noqa: E402  (_render, embed_paths, RENDER_*)
import location as loc_mod  # noqa: E402
import guard  # noqa: E402  (degenerate-outcome guard: make_guarded_scorer + the field gate)
import subprocess  # noqa: E402

# =========================================================================== #
# Config (top-of-file constants — pinned per the prompt)
# =========================================================================== #
# --- mixing ---
NATIVE_FRAC = 0.05           # native productivity floor
EXPLOIT_FRAC = 0.80          # of the non-native share
EXPLORE_FRAC = 0.20          # of the non-native share

# --- throttles ---
SEED_LAUNCH_CAP = 20         # launches per coverage cell (pre-run: compute + forced explore)
OUTCOME_DISTINCT_CAP = 10    # distinct harvested outcomes per cell -> saturates
DEDUP_K = 1.5                # near-dup iff dist < DEDUP_K * max(A.fw, B.fw)

# --- production walk config (matches the atlas theta-walk config so theta_hat stays valid) ---
NODE_WIDTH = 384             # foci-finder low-pass; 384 is outcome-value-preserving (efficiency study)
SIGMA_BAND = "8,10,12,14,16" # x0.5 (dilation=sigma, isolation=2sigma)
DEPTH_MIN = 4                # engine default; production walks descend deep
DEPTH_MAX = 14               # value ~= depth-17 at lower cost; theta_hat was fit at 14
OCC_FLOOR = 0.321
BLACK_CAP = 0.30
PER_WALK_RNG = True          # ON for all walk + probe runs

# --- probe / scorer ---
PROBE_DEPTH = 2              # --depth-min 2 --depth-max 2, keep reached>=2
SCORER_PATH = "data/classifier/v5/model_best.pt"   # explicit; NEVER a default scorer

# --- proposal cloud / selection quantiles (reuse round-2 arm semantics) ---
N_CLOUD = 40000              # one-time in-domain candidate cloud
EXPLOIT_ACQ_QUANTILE = 0.80  # exploit = keep acq (conf*theta_norm) above this quantile
EXPLORE_CONF_QUANTILE = 0.50 # explore = keep conf below this quantile (uncovered frontier)

# --- run ---
WALLCLOCK_BUDGET_MIN = 30
BATCH_SEEDS = 24             # proposals per batch (amortizes engine startup)
NATIVE_POOL_WALKS = 400      # native depth-1 walks generated once (seed source + fw pool)
WORKERS = 6

# Coverage-cell bounds when NO atlas is present (native-only mode). Frozen to the
# atlas_v1 mask_bounds so cell IDs stay identical to the cross-run cell_ledger built
# while the atlas existed (the ledger is cumulative; the bin must not shift under it).
DEFAULT_COVER_BOUNDS = (-1.818157959004375, 0.527935791035625,
                        -1.141661376973125, 1.116453857441875)

# --- durable store (committed via .gitignore negation, like the atlas) ---
DISCOVERY_DIR = ROOT / "data" / "discovery"
OUTCOME_LEDGER = DISCOVERY_DIR / "outcome_ledger.jsonl"
OUTCOME_FEATS = DISCOVERY_DIR / "outcome_feats.npz"
CELL_LEDGER = DISCOVERY_DIR / "cell_ledger.json"
PROBE_REJECTS = DISCOVERY_DIR / "probe_rejects.jsonl"
RUNS_DIR = DISCOVERY_DIR / "runs"
# disposable render scratch (never data/): native run, probe pools, walk pools, reward tiles
SCRATCH_ROOT = ROOT / "out" / "atlas" / "production_seeder"


# =========================================================================== #
# Coverage-region cell (reuse the 14x12 bin over atlas.mask_bounds VERBATIM —
# same grid + index math as round1_analyze.coverage_bins; this is the seed-cap
# cell and the saturation-tally cell so cap accounting speaks the coverage units).
# =========================================================================== #
def seed_cell(cx: float, cy: float, bounds) -> int:
    """Integer coverage-cell id for a depth-1 seed. Identical index math to
    round1_analyze.coverage_bins (COVER_NCOLS x COVER_NROWS over mask_bounds)."""
    x0, x1, y0, y1 = bounds
    ix = int(np.clip(int((cx - x0) / (x1 - x0) * COVER_NCOLS), 0, COVER_NCOLS - 1))
    iy = int(np.clip(int((cy - y0) / (y1 - y0) * COVER_NROWS), 0, COVER_NROWS - 1))
    return iy * COVER_NCOLS + ix


# =========================================================================== #
# Cap predicates (pure — unit-tested)
# =========================================================================== #
def near_dup(a_cx, a_cy, a_fw, b_cx, b_cy, b_fw, k=DEDUP_K) -> bool:
    """fw-relative dedup: A near-dup of B iff plane distance < k * max(A.fw, B.fw).
    max(fw) => two outcomes at ~same center but different zoom are the SAME place."""
    d = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
    return d < k * max(float(a_fw), float(b_fw))


def is_distinct(cx, cy, fw, harvested, k=DEDUP_K):
    """Distinctness vs ALL harvested outcomes (global, not per-cell). Returns
    (distinct: bool, dup_of: id|None). `harvested` = iterable of dicts with
    outcome_cx/outcome_cy/outcome_fw/id."""
    for h in harvested:
        if near_dup(cx, cy, fw, h["outcome_cx"], h["outcome_cy"], h["outcome_fw"], k):
            return False, h["id"]
    return True, None


def cell_saturated(state, seed_cap=None, distinct_cap=None) -> bool:
    """A coverage cell is saturated when distinct >= distinct_cap OR launches >= seed_cap.
    A saturated cell rejects further EXPLOIT proposals (-> backfill to explore).

    Caps default to the LIVE module globals (resolved at call time, not def time) so the
    smoke's lowered caps take effect after `_run` reassigns them."""
    seed_cap = SEED_LAUNCH_CAP if seed_cap is None else seed_cap
    distinct_cap = OUTCOME_DISTINCT_CAP if distinct_cap is None else distinct_cap
    return state.get("distinct", 0) >= distinct_cap or state.get("launches", 0) >= seed_cap


def cell_launch_capped(state, seed_cap=None) -> bool:
    """Pre-run seed-launch cap: reject ANY proposal (native/exploit/explore) whose cell
    already has >= seed_cap launches (compute economy + forced exploration). `seed_cap`
    defaults to the live module global (call-time), as in `cell_saturated`."""
    seed_cap = SEED_LAUNCH_CAP if seed_cap is None else seed_cap
    return state.get("launches", 0) >= seed_cap


# =========================================================================== #
# Durable ledgers (atomic temp+rename; cross-run cumulative; resumable)
# =========================================================================== #
def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Ledgers:
    """Loads existing state (cross-run cumulative) and appends/rewrites atomically."""

    def __init__(self):
        self.cells: dict[str, dict] = {}          # cell_id(str) -> {launches, distinct, saturated}
        self.harvested: list[dict] = []           # distinct outcomes (for global dedup)
        self.feats: dict[str, np.ndarray] = {}     # id -> 1280-D
        self.n_outcomes_logged = 0                 # total scored-walk rows (distinct + dup)
        self.load()

    def load(self):
        if CELL_LEDGER.exists():
            self.cells = json.loads(CELL_LEDGER.read_text(encoding="utf-8"))
        if OUTCOME_LEDGER.exists():
            for line in open(OUTCOME_LEDGER, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.n_outcomes_logged += 1
                # Harvested/dedup pool = distinct AND guard_pass rows only. Pre-guard
                # rows carry no guard_pass key -> treated as pass (True), so historical
                # state reloads unchanged until the Part-3 re-gate marks the failures.
                if r.get("distinct") and r.get("guard_pass", True):
                    self.harvested.append(r)
        if OUTCOME_FEATS.exists():
            z = np.load(OUTCOME_FEATS, allow_pickle=False)
            self.feats = {k: z[k] for k in z.files}

    # --- cell accounting ---
    def cell_state(self, cid: int) -> dict:
        return self.cells.get(str(cid), {"launches": 0, "distinct": 0, "saturated": False})

    def bump_launch(self, cid: int):
        s = self.cells.setdefault(str(cid), {"launches": 0, "distinct": 0, "saturated": False})
        s["launches"] += 1
        s["saturated"] = cell_saturated(s)

    def bump_distinct(self, cid: int):
        s = self.cells.setdefault(str(cid), {"launches": 0, "distinct": 0, "saturated": False})
        s["distinct"] += 1
        s["saturated"] = cell_saturated(s)

    def save_cells(self):
        _atomic_write_text(CELL_LEDGER, json.dumps(self.cells, indent=2))

    # --- outcome append (jsonl) + feature store (npz) ---
    def append_outcome(self, row: dict, feat: np.ndarray | None):
        OUTCOME_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTCOME_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.n_outcomes_logged += 1
        if row.get("distinct") and row.get("guard_pass", True):
            self.harvested.append(row)
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
# Native seed source (the engine's own root draw — already descendability-pre-gated
# by the native 8k/flat root gate). Run guided-descend natively at depth 1 and read
# the depth-1 root frames from walks.jsonl.  Doubles as the empirical fw pool the
# atlas proposer draws from (propose.load_fw_pool contract).
# =========================================================================== #
def generate_native_seeds(n_walks: int, seed: int, workdir: Path) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(propose.BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", "1", "--depth-max", "1",
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(workdir),
    ]
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


# =========================================================================== #
# Proposer — one big scored cloud (exploit/explore pools) + native pool, drawn
# per batch with the seed-launch cap + backfill (prefer explore).
# =========================================================================== #
class Proposer:
    """Holds the run-fixed candidate pools. `draw_batch` returns proposals honoring
    the mix, the seed-launch cap, and the exploit->explore backfill rule."""

    def __init__(self, atlas, fw_pool: np.ndarray, native_seeds: list[dict],
                 rng: np.random.Generator, bounds=None):
        self.atlas = atlas
        self.bounds = bounds if bounds is not None else (
            atlas.mask_bounds if atlas is not None else DEFAULT_COVER_BOUNDS)
        self.rng = rng
        self.native_seeds = native_seeds
        self.native_q = list(range(len(native_seeds)))
        rng.shuffle(self.native_q)

        # No atlas -> native-only: no value/uncertainty cloud, no exploit/explore pools.
        # The guarded scorer, the seed-launch cap, and the location distinct-cap all
        # stay active (they live in the harvest loop + _next_native, not the atlas).
        if atlas is None:
            self.cloud = np.empty((0, 2))
            self.fw = np.empty(0)
            self.theta = self.conf = self.theta_norm = self.acq = np.empty(0)
            self.exploit_q = []
            self.explore_q = []
            self.meta = dict(atlas=False, exploit_pool=0, explore_pool=0,
                             acq_thr=None, conf_thr=None, rmin=None, rmax=None)
            return

        cloud = propose.domain_cloud(atlas, N_CLOUD, rng)
        theta, conf, _ = atlas.query(cloud[:, 0], cloud[:, 1])
        rmin, rmax = float(atlas.reward.min()), float(atlas.reward.max())
        theta_norm = np.clip((theta - rmin) / (rmax - rmin + 1e-9), 0.0, 1.0)
        acq = conf * theta_norm                       # exploit acquisition
        fw = fw_pool[rng.integers(len(fw_pool), size=len(cloud))]

        acq_thr = float(np.quantile(acq, EXPLOIT_ACQ_QUANTILE))
        conf_thr = float(np.quantile(conf, EXPLORE_CONF_QUANTILE))
        exploit_idx = np.where(acq >= acq_thr)[0]
        explore_idx = np.where(conf <= conf_thr)[0]
        rng.shuffle(exploit_idx)                       # random order within the value band
        rng.shuffle(explore_idx)

        self.cloud, self.fw, self.theta, self.conf = cloud, fw, theta, conf
        self.theta_norm, self.acq = theta_norm, acq
        self.exploit_q = list(exploit_idx)             # consumable queues
        self.explore_q = list(explore_idx)
        self.meta = dict(atlas=True, acq_thr=acq_thr, conf_thr=conf_thr,
                         exploit_pool=len(exploit_idx), explore_pool=len(explore_idx),
                         rmin=rmin, rmax=rmax)

    def _mk(self, i, source):
        return {
            "mix_source": source,
            "seed_cx": float(self.cloud[i, 0]), "seed_cy": float(self.cloud[i, 1]),
            "fw": float(self.fw[i]),
            "theta": float(self.theta[i]), "conf": float(self.conf[i]),
            "acq": float(self.acq[i]),
        }

    def _mk_native(self, j):
        s = self.native_seeds[j]
        return {"mix_source": "native", "seed_cx": s["cx"], "seed_cy": s["cy"],
                "fw": s["fw"], "theta": None, "conf": None, "acq": None,
                "root_src": s.get("root_src", "")}

    def _next_explore(self, cells: Ledgers):
        """Pop the next explore candidate whose cell is not launch-capped."""
        while self.explore_q:
            i = self.explore_q.pop()
            cid = seed_cell(self.cloud[i, 0], self.cloud[i, 1], self.bounds)
            if not cell_launch_capped(cells.cell_state(cid)):
                return self._mk(i, "explore"), cid
        return None, None

    def _next_native(self, cells: Ledgers):
        while self.native_q:
            j = self.native_q.pop()
            p = self._mk_native(j)
            cid = seed_cell(p["seed_cx"], p["seed_cy"], self.bounds)
            if not cell_launch_capped(cells.cell_state(cid)):
                return p, cid
        return None, None

    def draw_batch(self, cells: Ledgers, n_batch: int):
        """Return (proposals, mix_report). Each proposal carries seed_cell. Applies the
        seed-launch cap pre-probe; exploit rejected on a launch-capped OR saturated cell
        -> backfilled to explore (fallback native), and every backfill is logged.

        No atlas -> native-only: the whole batch is drawn from the native root pool,
        honoring the seed-launch cap (no exploit/explore, no value/uncertainty steer)."""
        if self.atlas is None:
            props = []
            realized = {"native": 0, "exploit": 0, "explore": 0}
            for _ in range(n_batch):
                p, cid = self._next_native(cells)
                if p is None:
                    break                               # native pool exhausted (or all capped)
                p["seed_cell"] = cid
                props.append(p); realized["native"] += 1; cells.bump_launch(cid)
            mix = {"target": {"native": n_batch, "exploit": 0, "explore": 0},
                   "realized": realized, "backfills": 0}
            return props, mix

        n_native = round(n_batch * NATIVE_FRAC)
        n_rest = n_batch - n_native
        n_exploit = round(n_rest * EXPLOIT_FRAC)
        n_explore = n_rest - n_exploit

        props = []
        realized = {"native": 0, "exploit": 0, "explore": 0}
        backfills = 0

        def take_explore():
            nonlocal backfills
            p, cid = self._next_explore(cells)
            if p is None:                              # explore exhausted -> native fallback
                p, cid = self._next_native(cells)
            if p is not None:
                p["seed_cell"] = cid
                props.append(p)
                realized[p["mix_source"]] += 1
                cells.bump_launch(cid)
            return p is not None

        # native
        for _ in range(n_native):
            p, cid = self._next_native(cells)
            if p is None:
                break
            p["seed_cell"] = cid
            props.append(p); realized["native"] += 1; cells.bump_launch(cid)

        # exploit (with seed-cap + saturation -> explore backfill)
        for _ in range(n_exploit):
            placed = False
            while self.exploit_q:
                i = self.exploit_q.pop()
                cid = seed_cell(self.cloud[i, 0], self.cloud[i, 1], self.bounds)
                st = cells.cell_state(cid)
                if cell_launch_capped(st) or cell_saturated(st):
                    backfills += 1
                    placed = take_explore()          # forced exploration on a capped exploit cell
                    break
                p = self._mk(i, "exploit"); p["seed_cell"] = cid
                props.append(p); realized["exploit"] += 1; cells.bump_launch(cid)
                placed = True
                break
            if not placed and not self.exploit_q:
                backfills += 1
                take_explore()

        # explore
        for _ in range(n_explore):
            take_explore()

        mix = {"target": {"native": n_native, "exploit": n_exploit, "explore": n_explore},
               "realized": realized, "backfills": backfills}
        return props, mix


# =========================================================================== #
# Depth-2 descendability probe (reuse propose.prescreen VERBATIM; read the walks it
# writes for per-seed reached + cause). reached>=2 -> survivor; else probe-reject.
# =========================================================================== #
def depth2_probe(props: list[dict], workdir: Path, seed: int):
    """Returns (survivors, rejects) where survivor rows carry the proposal + reached,
    reject rows carry seed_cx/cy/seed_cell/reached/cause/child_occ(null)."""
    cloud = np.array([[p["seed_cx"], p["seed_cy"]] for p in props], float)
    fw = np.array([p["fw"] for p in props], float)
    scr = propose.prescreen(cloud, fw, workdir, NODE_WIDTH, OCC_FLOOR, BLACK_CAP, seed)
    reached = scr["reached"]
    # per-seed cause from the probe's own walks.jsonl (row order == proposal order).
    causes = {}
    wpath = workdir / "probe_pool" / "walks.jsonl"
    for line in open(wpath, encoding="utf-8"):
        line = line.strip()
        if line:
            w = json.loads(line)
            causes[int(w["walk"])] = w.get("cause", "")

    survivors, rejects = [], []
    for i, p in enumerate(props):
        p2 = dict(p); p2["probe_reached"] = int(reached[i]); p2["probe_cause"] = causes.get(i, "")
        if scr["pass"][i]:
            survivors.append(p2)
        else:
            rejects.append({
                "seed_cx": p["seed_cx"], "seed_cy": p["seed_cy"], "seed_cell": p["seed_cell"],
                "mix_source": p["mix_source"], "reached": int(reached[i]),
                "cause": causes.get(i, ""),
                # child occupancy is engine-internal (best-of-N gate) and not emitted
                # per-walk; recorded null so the occupancy-floor piggyback stays honest.
                "child_occ": None,
            })
    return survivors, rejects, scr["causes"]


# =========================================================================== #
# Full walks + k3 best-frame reward (reuse step0_reanalysis primitives, exactly as
# round1_harvest.harvest_walk does) + outcome center + 1280-D penultimate feature.
# =========================================================================== #
def run_full_walks(survivors: list[dict], workdir: Path, seed: int):
    """One --seed-list --per-walk-rng production walk run over the survivors."""
    workdir.mkdir(parents=True, exist_ok=True)
    seed_in = workdir / "survivor_seeds.jsonl"
    propose.write_seed_list(seed_in, [s["seed_cx"] for s in survivors],
                            [s["seed_cy"] for s in survivors], [s["fw"] for s in survivors])
    pool = workdir / "pool"
    cmd = [
        str(propose.BIN), "guided-descend",
        "--seed-list", str(seed_in), "--per-walk-rng", "--seed", str(seed),
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"full walk run failed:\n{r.stderr[-2000:]}")
    return pool


def harvest_walk_reward(scorer, wid, frames, workers, scratch):
    """k3 best-frame reward for one walk, composing the SAME step0_reanalysis primitives
    round1_harvest uses (raw_screen_walk / reframe_location / _mand_location / KRAW),
    plus the raw-top3 list and the k3-winner's reframed outcome geometry.

    Returns (reward_k3, reward_k1, raw_top3, reached_depth, outcome_cx/cy/fw, argmax_idx)."""
    sr.SCRATCH = scratch
    raws = raw_screen_walk(scorer, wid, frames, workers)
    order = sorted(range(len(frames)), key=lambda i: raws[i], reverse=True)
    topk = order[:KRAW]
    raw_top3 = [float(raws[i]) for i in topk]

    best = None   # (reframed_score, res, idx)
    reward_k1 = None
    for rank, i in enumerate(topk):
        fr = frames[i]
        loc = _mand_location(fr["cx"], fr["cy"], fr["fw"])
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
    reached = max(int(f["depth"]) for f in frames)
    return {
        "reward_k3": reward_k3, "reward_k1": reward_k1, "raw_top3": raw_top3,
        "reached_depth": reached, "k3_argmax_idx": k3_idx,
        "outcome_cx": float(res.cx), "outcome_cy": float(res.cy), "outcome_fw": float(res.fw),
    }


def outcome_feature(scorer, cx, cy, fw, tile: Path) -> np.ndarray:
    """Render the k3 winner's reframed crop once at deploy search fidelity (640x360 ss2,
    twilight_shifted) and forward it through the v5 penultimate hook -> 1280-D."""
    ok, err = r1e._render(cx, cy, fw, tile)
    if not ok:
        raise SystemExit(f"outcome tile render failed [{tile.name}]: {err}")
    return r1e.embed_paths(scorer, [tile])[0]


# =========================================================================== #
# Contact sheet (harvested distinct outcomes + a small near-dup strip)
# =========================================================================== #
def build_contact_sheet(distinct_tiles, dup_tiles, out_png: Path, title: str):
    from PIL import Image, ImageDraw
    TW, TH, PAD, LBL, GUT = 224, 126, 6, 16, 40
    NCOL = 6
    items = [(p, lab, (60, 220, 90)) for p, lab in distinct_tiles]
    if dup_tiles:
        items.append((None, "--- near-dups ---", (245, 215, 40)))
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
def _run(args):
    smoke = args.smoke
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_ts
    scratch = SCRATCH_ROOT / run_ts
    scratch.mkdir(parents=True, exist_ok=True)
    tiles_dir = scratch / "outcome_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # smoke: small cloud + small native pool + LOWERED caps so both caps fire.
    global N_CLOUD, SEED_LAUNCH_CAP, OUTCOME_DISTINCT_CAP, NATIVE_POOL_WALKS
    if smoke:
        N_CLOUD = 4000
        SEED_LAUNCH_CAP = 3
        OUTCOME_DISTINCT_CAP = 2
        NATIVE_POOL_WALKS = 60
    batch_seeds = args.batch or (20 if smoke else BATCH_SEEDS)
    budget_min = args.budget if args.budget is not None else (0 if smoke else WALLCLOCK_BUDGET_MIN)

    print(f"=== atlas production seeder ({'SMOKE' if smoke else 'RUN'}) ts={run_ts} ===")
    print(f"caps: SEED_LAUNCH_CAP={SEED_LAUNCH_CAP} OUTCOME_DISTINCT_CAP={OUTCOME_DISTINCT_CAP} "
          f"DEDUP_K={DEDUP_K}  batch={batch_seeds} budget={budget_min}min")
    print(f"walk cfg: node={NODE_WIDTH} sigma={SIGMA_BAND} depth[{DEPTH_MIN},{DEPTH_MAX}] "
          f"occ={OCC_FLOOR} black={BLACK_CAP} per_walk_rng={PER_WALK_RNG}")

    # Atlas is OPTIONAL. Present -> the round-2 exploit/explore proposer steers the
    # batch; absent -> native-only mode (seed purely from guided-descend's root draw),
    # with the guarded scorer, seed-launch cap, and location distinct-cap all still active.
    atlas = Atlas.load() if ARTIFACT_PATH.exists() else None
    if atlas is None:
        print(f"atlas: NONE ({ARTIFACT_PATH.name} absent) -> NATIVE-ONLY mode "
              f"(guided-descend root draw; exploit/explore disabled)")
    else:
        print(f"atlas: {ARTIFACT_PATH.name}")
    # Guarded v5 scorer: raw-frame scoring, reframe candidate scoring, and the k3
    # reward all inherit the model-free field guard (degenerate crops -> GUARD_SENTINEL).
    # The reframe/raw render paths dump a co-located field per tile (the reframe hook)
    # that the guard reads; enable it and pin the suffix contract.
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX, (
        f"guard field suffix drift: reframe {reframe.GUARD_FIELD_SUFFIX!r} != "
        f"guard {guard.FIELD_SIDECAR_SUFFIX!r}")
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    print(f"scorer: GUARDED v5 CORN ({SCORER_PATH})  geometry={scorer.cfg.get('geometry')}  "
          f"guard: interior_frac>={guard.INTERIOR_CAP} | field_std<{guard.FIELD_STD_FLOOR} "
          f"@ {guard.GUARD_STAT_RES}")
    ledgers = Ledgers()
    print(f"ledgers: {len(ledgers.harvested)} harvested outcomes, "
          f"{len(ledgers.cells)} cells, {ledgers.n_outcomes_logged} scored rows (cross-run)")

    rng = np.random.default_rng(args.seed)
    print(f"generating {NATIVE_POOL_WALKS} native depth-1 seeds (root draw = fw pool)...")
    native = generate_native_seeds(NATIVE_POOL_WALKS, args.seed, scratch / "native")
    fw_pool = np.array([s["fw"] for s in native], float)
    print(f"  native seeds: {len(native)}  fw range[{fw_pool.min():.4f},{fw_pool.max():.4f}] "
          f"median {np.median(fw_pool):.4f}")

    proposer = Proposer(atlas, fw_pool, native, rng)
    if atlas is not None:
        print(f"  cloud: {N_CLOUD} in-domain  exploit_pool={proposer.meta['exploit_pool']} "
              f"(acq>={proposer.meta['acq_thr']:.3f})  explore_pool={proposer.meta['explore_pool']} "
              f"(conf<={proposer.meta['conf_thr']:.3f})")
    else:
        print(f"  native-only: {len(native)} native seeds drive the batch "
              f"(no proposal cloud; seed-launch + distinct caps active)")

    totals = {"proposed": 0, "seed_capped": 0, "probe_rejected": 0, "walked": 0,
              "harvested_distinct": 0, "near_dup": 0, "guarded": 0, "backfills": 0,
              "realized": {"native": 0, "exploit": 0, "explore": 0}}
    distinct_tiles, dup_tiles = [], []
    batch_timings = []
    t0 = time.time()
    seq = 0
    batch_i = 0

    while True:
        tb = time.time()
        batch_i += 1
        # 1. propose (mix + seed-cap + backfill). bump_launch happens inside draw_batch.
        props, mix = proposer.draw_batch(ledgers, batch_seeds)
        if not props:
            print("  proposer exhausted (no non-capped candidates left); stopping.")
            break
        totals["proposed"] += len(props)
        totals["backfills"] += mix["backfills"]
        totals["seed_capped"] += mix["backfills"]   # exploit draws rejected by the pre-probe seed-launch cap
        for k in totals["realized"]:
            totals["realized"][k] += mix["realized"][k]
        ledgers.save_cells()   # launches are durable even if the batch dies mid-way

        # 2. depth-2 descendability probe (survivors reached>=2).
        pw = scratch / f"batch_{batch_i:03d}" / "probe"
        survivors, rejects, pcauses = depth2_probe(props, pw, args.seed)
        append_probe_rejects(rejects)
        totals["probe_rejected"] += len(rejects)
        if not survivors:
            print(f"  batch {batch_i}: 0/{len(props)} descendable (causes {pcauses}); next batch.")
            batch_timings.append(time.time() - tb)
            if budget_min and (time.time() - t0) / 60 >= budget_min:
                break
            if smoke:
                break
            continue

        # 3. full production walks over survivors.
        ww = scratch / f"batch_{batch_i:03d}" / "walks"
        pool = run_full_walks(survivors, ww, args.seed)
        by_walk = load_frames_by_walk(pool)
        totals["walked"] += len(by_walk)

        # 4. per-walk k3 reward + outcome center + 1280-D feature; 5. dedup + harvest.
        b_distinct = b_dup = b_guarded = 0
        for wid in sorted(by_walk):
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS, scratch / f"reward_b{batch_i:03d}")
            # the survivor that produced this walk (row order: walk w <- survivor w)
            sv = survivors[wid] if wid < len(survivors) else survivors[-1]
            # Guard verdict for the harvested outcome: the k3 winner is the best
            # reframed crop, and a failing crop scores GUARD_SENTINEL, so k3 collapses
            # to the sentinel iff EVERY framing of the top-3 failed the guard. A guarded
            # outcome is not counted toward the cell distinct-tally nor the dedup pool
            # (harvested set = guard_pass rows only; matches the re-gated ledger).
            guard_pass = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
            distinct, dup_of = is_distinct(rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"],
                                           ledgers.harvested, DEDUP_K)
            harvest_distinct = bool(distinct) and guard_pass
            oid = f"m_{run_ts}_{seq:06d}"; seq += 1
            tile = tiles_dir / f"{oid}.jpg"
            feat = outcome_feature(scorer, rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"], tile)
            row = {
                "id": oid, "ts": run_ts, "family": "mandelbrot",
                "mix_source": sv["mix_source"],
                "seed_cx": sv["seed_cx"], "seed_cy": sv["seed_cy"], "seed_cell": sv["seed_cell"],
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"], "raw_top3": rew["raw_top3"],
                "probe_child_occ": None, "probe_reached": sv.get("probe_reached"),
                "probe_cause": sv.get("probe_cause"), "reached_depth": rew["reached_depth"],
                "distinct": harvest_distinct, "dup_of": dup_of,
                # guard_pass gates the harvested set; guard_fail 'sentinel' = the k3
                # collapse (every top-3 framing degenerate). Precise interior/flat/both
                # attribution is a re-gate concern (Part 3), not a live-harvest one.
                "guard_pass": guard_pass, "guard_fail": None if guard_pass else "sentinel",
            }
            ledgers.append_outcome(row, feat)
            if harvest_distinct:
                ledgers.bump_distinct(sv["seed_cell"])
                b_distinct += 1
                distinct_tiles.append((tile, f"{oid[-6:]} k3={rew['reward_k3']:.2f} {sv['mix_source'][:3]}"))
            elif not guard_pass:
                b_guarded += 1
                if len(dup_tiles) < NCOL_DUP:
                    dup_tiles.append((tile, f"k3={rew['reward_k3']:.2f} GUARDED"))
            else:
                b_dup += 1
                if len(dup_tiles) < NCOL_DUP:
                    dup_tiles.append((tile, f"k3={rew['reward_k3']:.2f}->dup"))
        ledgers.save_cells(); ledgers.save_feats()
        totals["harvested_distinct"] += b_distinct
        totals["near_dup"] += b_dup
        totals["guarded"] += b_guarded

        dt = time.time() - tb
        batch_timings.append(dt)
        el_min = (time.time() - t0) / 60
        n_sat = sum(1 for s in ledgers.cells.values() if s.get("saturated"))
        print(f"  batch {batch_i}: props={len(props)} surv={len(survivors)} walked={len(by_walk)} "
              f"| distinct+{b_distinct} dup+{b_dup} guarded+{b_guarded} | backfills={mix['backfills']} "
              f"realized={mix['realized']} | cells_sat={n_sat} | {dt:.0f}s (elapsed {el_min:.1f}m)")

        if budget_min and el_min >= budget_min:
            print(f"  wallclock budget {budget_min}min reached; stopping cleanly.")
            break
        if smoke and batch_i >= 3:
            # smoke: 3 batches is enough for the lowered caps (3 launches / 2 distinct)
            # to fire — the run then stops so the contact sheet can be eyeballed.
            break

    # ---- persist + report ----
    ledgers.save_cells(); ledgers.save_feats()
    n_sat = sum(1 for s in ledgers.cells.values() if s.get("saturated"))
    n_launch_capped = sum(1 for s in ledgers.cells.values() if cell_launch_capped(s))
    summary = {
        "ts": run_ts, "smoke": smoke, "wallclock_s": round(time.time() - t0, 1),
        "batches": batch_i, "batch_timings_s": [round(x, 1) for x in batch_timings],
        "config": {"seed_launch_cap": SEED_LAUNCH_CAP, "outcome_distinct_cap": OUTCOME_DISTINCT_CAP,
                   "dedup_k": DEDUP_K, "node_width": NODE_WIDTH, "sigma_band": SIGMA_BAND,
                   "depth": [DEPTH_MIN, DEPTH_MAX], "occ_floor": OCC_FLOOR, "black_cap": BLACK_CAP,
                   "batch_seeds": batch_seeds, "budget_min": budget_min, "scorer": SCORER_PATH},
        "totals": totals,
        "cells_saturated": n_sat, "cells_launch_capped": n_launch_capped,
        "cumulative": {"harvested_outcomes": len(ledgers.harvested),
                       "cells_tracked": len(ledgers.cells),
                       "scored_rows": ledgers.n_outcomes_logged},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    sheet = build_contact_sheet(
        distinct_tiles, dup_tiles, run_dir / "contact_sheet.png",
        f"production seeder {run_ts} — {len(distinct_tiles)} harvested distinct "
        f"(green) + near-dups (yellow)")

    print("\n=== RUN SUMMARY ===")
    print(f"  proposed={totals['proposed']} probe_rejected={totals['probe_rejected']} "
          f"walked={totals['walked']} harvested_distinct={totals['harvested_distinct']} "
          f"near_dup={totals['near_dup']} guard_failed={totals['guarded']} "
          f"backfills={totals['backfills']}")
    print(f"  realized mix: {totals['realized']}")
    print(f"  cells: {n_launch_capped} launch-capped, {n_sat} saturated "
          f"(cumulative {len(ledgers.cells)} tracked)")
    print(f"  wallclock {summary['wallclock_s']}s over {batch_i} batches")
    print(f"  ledgers -> {OUTCOME_LEDGER.name}, {CELL_LEDGER.name}, {OUTCOME_FEATS.name}, "
          f"{PROBE_REJECTS.name}")
    print(f"  summary -> {run_dir / 'summary.json'}\n  sheet   -> {sheet}")
    return summary


NCOL_DUP = 6   # cap the near-dup strip on the contact sheet


def _finalize(run_ts: str):
    """Rebuild a run's summary.json + contact_sheet.png from the DURABLE ledger + the
    on-disk outcome tiles. This is the resume path for the durability contract: the
    ledgers are written per-outcome, so a kill in the cosmetic final stage loses no data
    — this reconstructs the missing cosmetic artifacts for run `run_ts`."""
    run_dir = RUNS_DIR / run_ts
    tiles_dir = SCRATCH_ROOT / run_ts / "outcome_tiles"
    rows = [json.loads(l) for l in open(OUTCOME_LEDGER, encoding="utf-8")
            if l.strip() and json.loads(l).get("ts") == run_ts]
    if not rows:
        raise SystemExit(f"no outcome rows with ts={run_ts} in {OUTCOME_LEDGER}")
    cells = json.loads(CELL_LEDGER.read_text(encoding="utf-8")) if CELL_LEDGER.exists() else {}
    distinct = [r for r in rows if r.get("distinct")]
    dup = [r for r in rows if not r.get("distinct")]
    realized = {"native": 0, "exploit": 0, "explore": 0}
    for r in rows:
        realized[r["mix_source"]] = realized.get(r["mix_source"], 0) + 1
    totals = {"walked": len(rows), "harvested_distinct": len(distinct), "near_dup": len(dup),
              "realized": realized}
    n_sat = sum(1 for s in cells.values() if s.get("saturated"))
    n_capped = sum(1 for s in cells.values() if cell_launch_capped(s))
    summary = {"ts": run_ts, "finalized_from_ledger": True, "totals": totals,
               "cells_saturated": n_sat, "cells_launch_capped": n_capped,
               "cumulative": {"cells_tracked": len(cells)},
               "config": {"seed_launch_cap": SEED_LAUNCH_CAP, "outcome_distinct_cap": OUTCOME_DISTINCT_CAP,
                          "dedup_k": DEDUP_K, "scorer": SCORER_PATH}}
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    dtiles = [(tiles_dir / f"{r['id']}.jpg", f"{r['id'][-6:]} k3={r['k3']:.2f} {r['mix_source'][:3]}")
              for r in sorted(distinct, key=lambda r: -r["k3"])]
    utiles = [(tiles_dir / f"{r['id']}.jpg", f"k3={r['k3']:.2f}->dup") for r in dup[:NCOL_DUP]]
    sheet = build_contact_sheet(dtiles, utiles, run_dir / "contact_sheet.png",
                                f"production seeder {run_ts} (finalized) — {len(distinct)} harvested "
                                f"distinct (green) + near-dups (yellow)")
    print(f"finalized run {run_ts}: {len(distinct)} distinct / {len(dup)} dup  "
          f"realized={realized}  cells {n_capped} launch-capped, {n_sat} saturated")
    print(f"  summary -> {run_dir / 'summary.json'}\n  sheet   -> {sheet}")
    return summary


def _time_only(args):
    """Project per-batch wallclock from one native-gen + one small batch."""
    args.smoke = True
    args.batch = args.batch or 12
    print("(--time-only: running one smoke batch to project per-batch cost)")
    t = time.time()
    _run(args)
    print(f"\n  one smoke batch (native-gen + probe + walks + reward) took {time.time()-t:.0f}s")
    print(f"  a 30-min run fits ~{int(30*60/max(1,(time.time()-t))):d} batches of this size "
          f"(minus one-time native-gen).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="~20-seed run, lowered caps so both fire")
    ap.add_argument("--run", action="store_true", help="30-min time-boxed production run")
    ap.add_argument("--time-only", action="store_true", help="project per-batch wallclock")
    ap.add_argument("--finalize", metavar="RUN_TS", default=None,
                    help="rebuild summary + contact sheet for a run from the durable ledger")
    ap.add_argument("--seed", type=int, default=0, help="rng + engine seed")
    ap.add_argument("--batch", type=int, default=0, help="seeds per batch (0 = default)")
    ap.add_argument("--budget", type=float, default=None, help="wallclock budget minutes override")
    args = ap.parse_args()
    if args.finalize:
        _finalize(args.finalize)
    elif args.time_only:
        _time_only(args)
    elif args.smoke or args.run:
        _run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
