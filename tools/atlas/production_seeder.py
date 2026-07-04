#!/usr/bin/env python
r"""Atlas production discovery seeder (Mandelbrot) — the standing discovery flow.

Native-only discovery: the guided-descend engine's own root draw proposes depth-1
seeds; coverage is controlled by REJECTION SAMPLING over a point cloud of distinct
q3 outcomes (there is no atlas, no coverage grid, no per-cell cap). Per batch:

  draw native depth-1 seed  (engine root draw, ~96% descendability pre-gated)
    -> REJECT if >= Q3_DENSITY_CAP distinct q3 outcomes lie within REJECT_RADIUS
       of the seed's (cx, cy); else accept  (test is FREE next to one descent)
    -> depth-2 descendability probe  (reuse propose.prescreen, verbatim engine step-1)
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
  * guided-descend engine (Rust) w/ --seed-list --per-walk-rng      (propose.BIN)
  * depth-2 descendability pre-screen                  propose.prescreen (verbatim)
  * reframe path                                       tools/reframe/reframe.py
  * v5 scorer bridge (explicit model_path)             probe.make_scorer / score_lib.Scorer
  * canonical v5 CORN hard-class decode                score_lib.corn_decode
  * k3 reward primitives                               step0_reanalysis (via round1_harvest)
  * v5 1280-D penultimate hook                         round1_embed.embed_paths / _render
  * degenerate-outcome guard                           tools/atlas/guard.py
  * canonical location layer                           tools/corpus/location.py

  uv run python tools/atlas/production_seeder.py --smoke     # ~20 seeds, rejection forced to fire
  uv run python tools/atlas/production_seeder.py --run       # 30-min time-boxed production run
  uv run python tools/atlas/production_seeder.py --time-only # project per-batch wallclock
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# reuse roots (same layering round1_harvest uses)
sys.path.insert(0, str(HERE))                                   # propose.py, round1_*.py
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

import propose  # noqa: E402  (BIN, prescreen, write_seed_list, SCREEN_*)
import step0_reanalysis as sr  # noqa: E402  (KRAW, raw_screen_walk, _mand_location, load_frames_by_walk)
from step0_reanalysis import (  # noqa: E402
    KRAW, raw_screen_walk, _mand_location, load_frames_by_walk,
)
import reframe  # noqa: E402  (reframe_location + the DUMP_GUARD_FIELD hook)
from reframe import reframe_location  # noqa: E402
import round1_embed as r1e  # noqa: E402  (_render, embed_paths, RENDER_*)
import guard  # noqa: E402  (degenerate-outcome guard: make_guarded_scorer + the field gate)
from score_lib import corn_decode  # noqa: E402  (canonical v5 CORN hard-class decode)

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
SCORER_PATH = "data/classifier/v5/model_best.pt"   # explicit; NEVER a default scorer

# --- run ---
WALLCLOCK_BUDGET_MIN = 30
BATCH_SEEDS = 24             # accepted proposals per batch (amortizes engine startup)
NATIVE_POOL_WALKS = 400      # native depth-1 walks generated per refill (seed source)
WORKERS = 4              # multiprocessing worker cap (project rule: max 4)

# --- durable store (committed via .gitignore negation, like the atlas) ---
DISCOVERY_DIR = ROOT / "data" / "discovery"
OUTCOME_LEDGER = DISCOVERY_DIR / "outcome_ledger.jsonl"
OUTCOME_FEATS = DISCOVERY_DIR / "outcome_feats.npz"
PROBE_REJECTS = DISCOVERY_DIR / "probe_rejects.jsonl"
RUNS_DIR = DISCOVERY_DIR / "runs"
# disposable render scratch (never data/): native run, probe pools, walk pools, reward tiles
SCRATCH_ROOT = ROOT / "out" / "atlas" / "production_seeder"

NCOL_DUP = 6   # cap the near-dup / non-q3 strip on the contact sheet


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
def build_cloud(rows: list[dict]) -> list[dict]:
    """One position per distinct q3 place: guard_pass && decoded_class == 3, deduped by
    1.5*max(fw). Order-stable: the earliest distinct row wins a dedup cluster, matching
    the live-harvest add order.

    Rows predating the decoded_class field (no historical backfill — see the module note)
    lack decoded_class and are simply excluded; the cross-run cloud rebuilds from rows the
    new pipeline logs going forward."""
    cloud: list[dict] = []
    for r in rows:
        if r.get("guard_pass", True) and r.get("decoded_class") == 3:
            distinct, _ = is_distinct(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"],
                                      cloud, DEDUP_K)
            if distinct:
                cloud.append(r)
    return cloud


def cloud_diagnostic(rows: list[dict], cloud: list[dict]) -> dict:
    """Startup summary: total rows, guard_pass count, class-2 vs class-3 split among
    guard-clean *decoded* rows (rows the new pipeline logged; pre-decoded_class rows are
    not counted), and the distinct q3 cloud size after dedup."""
    guard_clean = [r for r in rows
                   if r.get("guard_pass", True) and r.get("decoded_class") is not None]
    split = {c: sum(1 for r in guard_clean if r["decoded_class"] == c) for c in (1, 2, 3)}
    n_undecoded = sum(1 for r in rows
                      if r.get("guard_pass", True) and r.get("decoded_class") is None)
    return {"total_rows": len(rows), "guard_pass": sum(1 for r in rows if r.get("guard_pass", True)),
            "guard_clean_decoded": len(guard_clean), "undecoded_guard_pass": n_undecoded,
            "class_split": split, "cloud_size": len(cloud)}


# =========================================================================== #
# Native seed source + rejection sampler. The engine's own root draw (already
# descendability-pre-gated by the native 8k/flat root gate) proposes depth-1 seeds;
# each is accepted only if the q3 cloud is sparse within REJECT_RADIUS. The pool
# refills on demand (fresh engine sub-seed) so the ONLY stop conditions are global
# saturation (MAX_SEED_REDRAWS consecutive rejections) and the wallclock budget.
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


class NativeSeeder:
    """Draws native depth-1 seeds and applies the q3-density rejection test. Refills the
    native pool from a fresh engine sub-seed when depleted. Tracks draw/reject counters
    and the saturation flag (consecutive rejections > MAX_SEED_REDRAWS)."""

    def __init__(self, base_seed: int, scratch: Path, rng: np.random.Generator):
        self.base_seed = base_seed
        self.scratch = scratch
        self.rng = rng
        self.gen = 0
        self.q: list[dict] = []
        self.draws = 0          # total native seeds examined (accepted + rejected)
        self.rejects = 0        # total rejected by the density test
        self.consec = 0         # consecutive rejections (resets on an accept)
        self.saturated = False

    def _refill(self):
        wd = self.scratch / f"native_{self.gen}"
        seeds = generate_native_seeds(NATIVE_POOL_WALKS, self.base_seed + 1000 * self.gen, wd)
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
# Depth-2 descendability probe (reuse propose.prescreen VERBATIM; read the walks it
# writes for per-seed reached + cause). reached>=2 -> survivor; else probe-reject.
# =========================================================================== #
def depth2_probe(props: list[dict], workdir: Path, seed: int):
    """Returns (survivors, rejects, causes) where survivor rows carry the proposal +
    reached, reject rows carry seed_cx/cy/reached/cause/child_occ."""
    cloud = np.array([[p["seed_cx"], p["seed_cy"]] for p in props], float)
    fw = np.array([p["fw"] for p in props], float)
    scr = propose.prescreen(cloud, fw, workdir, NODE_WIDTH, OCC_FLOOR, BLACK_CAP, seed)
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


def _chosen_probs(res) -> tuple[float, float]:
    """CORN (p_notbad, p_good) of a reframe winner's chosen frame — pulled from the
    reframe trace (already computed when k3 was scored; no extra render/forward)."""
    ch = res.trace["chosen"]
    for rc in res.trace["recenter"]:
        if rc["dx"] == ch["dx"] and rc["dy"] == ch["dy"]:
            return float(rc["p_notbad"]), float(rc["p_good"])
    # unreachable (chosen is always one of the recenter candidates); degrade to class-1.
    return 0.0, 0.0


def harvest_walk_reward(scorer, wid, frames, workers, scratch):
    """k3 best-frame reward for one walk, composing the SAME step0_reanalysis primitives
    round1_harvest uses (raw_screen_walk / reframe_location / _mand_location / KRAW),
    plus the raw-top3 list, the k3-winner's reframed outcome geometry, and the winner's
    CORN (p_notbad, p_good) for the hard-class decode."""
    sr.SCRATCH = scratch
    raws = raw_screen_walk(scorer, wid, frames, workers)
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
    p_notbad, p_good = _chosen_probs(res)
    reached = max(int(f["depth"]) for f in frames)
    return {
        "reward_k3": reward_k3, "reward_k1": reward_k1, "raw_top3": raw_top3,
        "reached_depth": reached, "k3_argmax_idx": k3_idx,
        "outcome_cx": float(res.cx), "outcome_cy": float(res.cy), "outcome_fw": float(res.fw),
        "p_notbad": p_notbad, "p_good": p_good,
        "frames_gated": frames_gated, "n_frames": len(frames),
    }


def outcome_feature(scorer, cx, cy, fw, tile: Path) -> np.ndarray:
    """Render the k3 winner's reframed crop once at deploy search fidelity (640x360 ss2,
    twilight_shifted) and forward it through the v5 penultimate hook -> 1280-D."""
    ok, err = r1e._render(cx, cy, fw, tile)
    if not ok:
        raise SystemExit(f"outcome tile render failed [{tile.name}]: {err}")
    return r1e.embed_paths(scorer, [tile])[0]


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
def _run(args):
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
    print(f"coverage: q3-density REJECTION  radius={REJECT_RADIUS} cap={Q3_DENSITY_CAP} "
          f"max_redraws={MAX_SEED_REDRAWS} dedup_k={DEDUP_K}  batch={batch_seeds} budget={budget_min}min")
    print(f"walk cfg: node={NODE_WIDTH} sigma={SIGMA_BAND} depth[{DEPTH_MIN},{DEPTH_MAX}] "
          f"occ={OCC_FLOOR} black={BLACK_CAP} per_walk_rng={PER_WALK_RNG}")

    # Guarded v5 scorer: raw-frame scoring, reframe candidate scoring, and the k3
    # reward all inherit the model-free field guard (degenerate crops -> GUARD_SENTINEL).
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX, (
        f"guard field suffix drift: reframe {reframe.GUARD_FIELD_SUFFIX!r} != "
        f"guard {guard.FIELD_SIDECAR_SUFFIX!r}")
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    print(f"scorer: GUARDED v5 CORN ({SCORER_PATH})  geometry={scorer.cfg.get('geometry')}  "
          f"guard: interior_frac>={guard.INTERIOR_CAP} | field_std<{guard.FIELD_STD_FLOOR} "
          f"@ {guard.GUARD_STAT_RES}")

    ledgers = Ledgers()
    # No historical backfill (by design): rows predating the decoded_class field don't
    # enter the q3 cloud; the cross-run coverage cloud rebuilds from rows the new pipeline
    # logs going forward.
    cloud = build_cloud(ledgers.rows)
    diag = cloud_diagnostic(ledgers.rows, cloud)
    print(f"ledgers: {diag['total_rows']} rows, {diag['guard_pass']} guard_pass "
          f"({diag['guard_clean_decoded']} decoded; class1/2/3="
          f"{diag['class_split'][1]}/{diag['class_split'][2]}/{diag['class_split'][3]}; "
          f"{diag['undecoded_guard_pass']} pre-decode rows excluded)"
          f"  | q3 cloud {diag['cloud_size']} distinct places")

    rng = np.random.default_rng(args.seed)
    native = NativeSeeder(args.seed, scratch, rng)

    totals = {"proposed": 0, "probe_rejected": 0, "walked": 0,
              "harvested_distinct": 0, "q3_dup": 0, "not_q3": 0, "guarded": 0}
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
        survivors, rejects, pcauses = depth2_probe(props, pw, args.seed)
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
        pool = run_full_walks(survivors, ww, args.seed)
        by_walk = load_frames_by_walk(pool)
        totals["walked"] += len(by_walk)

        # 4. per-walk k3 reward + outcome center + decode + 1280-D feature; 5. cloud add.
        b_distinct = b_q3dup = b_notq3 = b_guarded = 0
        for wid in sorted(by_walk):
            frames = by_walk[wid]
            rew = harvest_walk_reward(scorer, wid, frames, WORKERS, scratch / f"reward_b{batch_i:03d}")
            sv = survivors[wid] if wid < len(survivors) else survivors[-1]
            # Guard verdict: k3 collapses to the sentinel iff EVERY framing of the top-3
            # failed the guard. A guarded outcome carries decoded_class=None and can never
            # be a q3 cloud member.
            guard_pass = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
            decoded = corn_decode(rew["p_notbad"], rew["p_good"]) if guard_pass else None
            is_q3 = guard_pass and decoded == 3
            if is_q3:
                distinct, dup_of = is_distinct(rew["outcome_cx"], rew["outcome_cy"],
                                               rew["outcome_fw"], cloud, DEDUP_K)
            else:
                distinct, dup_of = False, None

            oid = f"m_{run_ts}_{seq:06d}"; seq += 1
            tile = tiles_dir / f"{oid}.jpg"
            feat = outcome_feature(scorer, rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"], tile)
            row = {
                "id": oid, "ts": run_ts, "family": "mandelbrot",
                "mix_source": sv["mix_source"],
                "seed_cx": sv["seed_cx"], "seed_cy": sv["seed_cy"],
                "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"], "raw_top3": rew["raw_top3"],
                "probe_child_occ": sv.get("probe_child_occ"), "probe_reached": sv.get("probe_reached"),
                "probe_cause": sv.get("probe_cause"), "reached_depth": rew["reached_depth"],
                # decoded_class: the CORN hard class of the k3-winning frame (None if guarded).
                # This is the ONLY schema addition — cross-run cloud reconstruction is then a
                # direct filter (guard_pass && decoded_class == 3) with no re-decode.
                "decoded_class": decoded,
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
        ledgers.save_feats()
        totals["harvested_distinct"] += b_distinct
        totals["q3_dup"] += b_q3dup
        totals["not_q3"] += b_notq3
        totals["guarded"] += b_guarded

        dt = time.time() - tb
        batch_timings.append(dt)
        el_min = (time.time() - t0) / 60
        rej_frac = native.rejects / max(1, native.draws)
        print(f"  batch {batch_i}: props={len(props)} surv={len(survivors)} walked={len(by_walk)} "
              f"| q3+{b_distinct} q3dup+{b_q3dup} notq3+{b_notq3} guarded+{b_guarded} "
              f"| draws={native.draws} rej={native.rejects} ({rej_frac:.0%}) cloud={len(cloud)} "
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
        "config": {"reject_radius": REJECT_RADIUS, "q3_density_cap": Q3_DENSITY_CAP,
                   "max_seed_redraws": MAX_SEED_REDRAWS, "dedup_k": DEDUP_K,
                   "node_width": NODE_WIDTH, "sigma_band": SIGMA_BAND,
                   "depth": [DEPTH_MIN, DEPTH_MAX], "occ_floor": OCC_FLOOR, "black_cap": BLACK_CAP,
                   "batch_seeds": batch_seeds, "budget_min": budget_min, "scorer": SCORER_PATH},
        "totals": totals,
        "seeds": {"draws": native.draws, "rejected": native.rejects,
                  "rejected_fraction": rej_frac, "saturation": native.saturated},
        "q3_added_this_run": q3_added_this_run,
        "guard_telemetry": guard_telemetry,
        "cumulative": {"q3_cloud_size": len(cloud),
                       "scored_rows": ledgers.n_outcomes_logged},
    }
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    sheet = build_contact_sheet(
        distinct_tiles, dup_tiles, run_dir / "contact_sheet.png",
        f"production seeder {run_ts} — {len(distinct_tiles)} new q3 (green) + "
        f"dup/non-q3/guarded (yellow)")

    print("\n=== RUN SUMMARY ===")
    print(f"  proposed={totals['proposed']} probe_rejected={totals['probe_rejected']} "
          f"walked={totals['walked']} | q3_new={totals['harvested_distinct']} "
          f"q3_dup={totals['q3_dup']} not_q3={totals['not_q3']} guarded={totals['guarded']}")
    print(f"  seeds: {native.draws} drawn, {native.rejects} rejected ({rej_frac:.1%}), "
          f"saturation={native.saturated}")
    print(f"  q3 cloud: {len(cloud)} distinct places (+{q3_added_this_run} this run)")
    print(f"  wallclock {summary['wallclock_s']}s over {batch_i} batches")
    print(f"  ledgers -> {OUTCOME_LEDGER.name}, {OUTCOME_FEATS.name}, {PROBE_REJECTS.name}")
    print(f"  summary -> {run_dir / 'summary.json'}\n  sheet   -> {sheet}")
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
    cloud = build_cloud(all_rows)
    distinct = [r for r in rows if r.get("distinct")]
    dup = [r for r in rows if not r.get("distinct")]
    totals = {"walked": len(rows), "harvested_distinct": len(distinct),
              "q3_dup": sum(1 for r in rows if not r.get("distinct")
                            and r.get("guard_pass", True) and r.get("decoded_class") == 3),
              "not_q3": sum(1 for r in rows if r.get("guard_pass", True)
                            and r.get("decoded_class") not in (None, 3)),
              "guarded": sum(1 for r in rows if not r.get("guard_pass", True))}
    summary = {"ts": run_ts, "finalized_from_ledger": True, "totals": totals,
               "cumulative": {"q3_cloud_size": len(cloud), "scored_rows": len(all_rows)},
               "config": {"reject_radius": REJECT_RADIUS, "q3_density_cap": Q3_DENSITY_CAP,
                          "dedup_k": DEDUP_K, "scorer": SCORER_PATH}}
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(run_dir / "summary.json", json.dumps(summary, indent=2))
    dtiles = [(tiles_dir / f"{r['id']}.jpg", f"{r['id'][-6:]} k3={r['k3']:.2f} q3")
              for r in sorted(distinct, key=lambda r: -r["k3"])]
    utiles = [(tiles_dir / f"{r['id']}.jpg", f"k3={r['k3']:.2f}->dup") for r in dup[:NCOL_DUP]]
    sheet = build_contact_sheet(dtiles, utiles, run_dir / "contact_sheet.png",
                                f"production seeder {run_ts} (finalized) — {len(distinct)} new q3 "
                                f"(green) + dup/non-q3/guarded (yellow)")
    print(f"finalized run {run_ts}: {len(distinct)} new q3 / {len(dup)} other  "
          f"| q3 cloud {len(cloud)} distinct places")
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
    ap.add_argument("--smoke", action="store_true", help="~20-seed run, lowered knobs so rejection fires")
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
