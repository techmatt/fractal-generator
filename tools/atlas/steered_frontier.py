#!/usr/bin/env python
"""steered_frontier.py — classifier-steered frontier descent (a new mode beside the walk).

The current production descent (`production_seeder.py` -> `guided-descend` walk -> reward)
picks a **uniform-random survivor** per rung and only scores FINISHED frames; the aesthetic
classifier never touches the trajectory (see `out/descent_algorithm_current.md`). Here the
classifier STEERS: a best-first frontier where each pop expands one rung
(`guided-descend --expand`, all gate survivors), scores every survivor's cheap 384-wide
twilight_shifted field with the active checkpoint, and re-prioritises by
`E[ord] + Gumbel - dup-penalty`. The fidelity study (`out/descent_score_fidelity.md`)
proved v7 on that cheap presentation ranks frames at Spearman 0.95 vs canonical, so the
steering signal is nearly free.

Everything downstream of "which node to expand" is REUSED verbatim from the production
seeder — the gates (black-cap 0.30 -> band -> occ-floor 0.321, node 384 / sigma-band), the
root pipeline (native depth-1 seeds + q3-density rejection + depth-2 probe), the julia hook,
the harvest (reframe + CORN decode at the per-partition t_good), the near-dup cloud, the
guard, and the ledger schema. Only the trajectory POLICY is new; the current walk path is
byte-untouched.

v1.1 priority (both coefficients default-on; set BOTH to 0 to reproduce the pilot exactly):
  priority = cheap_eord + Gumbel(T) - dup_penalty - novelty_penalty + beta*depth
`novelty_penalty` (`--lambda-m`, default 0.5) damps morph-space near-repeats: every scored
candidate's cheap twilight image is CLIP-embedded (library recipe) alongside the v7 forward
and compared (cos_max) against a run-scoped morph memory of all admitted + already-expanded
looks; the penalty ramps 0->lambda_m across cos [lo, hi], where the knee is re-anchored
EMPIRICALLY on this cheap substrate (morph_anchor_calibrate.py -> data/atlas/morph_anchors.json;
the library morph_gray anchors 0.851/0.974 are grayscale-scale and do not transfer). Siblings look alike,
so a hot lineage self-suppresses and perceptual re-buys sink before expansion. `beta*depth`
(`--beta`, default 0.02) is a small depth tie-breaker. Per-term contributions are logged to
`prio_terms.jsonl` per pushed candidate.

Crash safety is load-bearing (long processes here get killed at random): the frontier +
budget + RNG + per-root cap counters checkpoint to state.json every batch; `--resume`
continues; a STOP sentinel halts at a batch boundary; the admitted-outcome cloud is rebuilt
from the run ledger (the durable source of truth) so a kill/resume can never lose or
duplicate an admission.

  # one arm (steered), fresh run-scoped dir, 45 min:
  uv run python tools/atlas/steered_frontier.py --run-dir data/discovery/steered_runs/A \
      --families mandelbrot,multibrot3,multibrot4,multibrot5 --julia-hook --budget 45
  uv run python tools/atlas/steered_frontier.py --run-dir <dir> --resume        # after a kill
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

# production_seeder wires its own sub-imports (prescreen / reframe / guard / score_lib /
# active_ckpt) and owns the constants, root pipeline, near-dup machinery, guard, and the
# per-partition t_good table. Reuse it wholesale.
import production_seeder as ps          # noqa: E402
import prescreen                        # noqa: E402
import reframe                          # noqa: E402
import guard                            # noqa: E402
import location as loc_mod              # noqa: E402
from score_lib import corn_decode       # noqa: E402
from active_ckpt import ACTIVE_CKPT, auto_maxiter  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BIN = ps.prescreen.BIN

# --- steering knobs ---
B_DEFAULT = 32            # nodes popped + expanded per batch
T_GUMBEL = 0.08          # priority exploration temperature (Gumbel scale)
M_CAP = 40               # hard cap on expansions per root_id
DUP_P0 = 1.0             # dup-penalty magnitude at zero distance to the q3 cloud (E[ord] units)
DUP_SCALE = ps.REJECT_RADIUS   # Gaussian decay scale of the dup penalty (plane coords)
NEUTRAL_PRIOR = 1.0      # root prior priority (mid E[ord] in [0,2])

# --- morph-novelty + depth knobs (v1.1; both zero => byte-identical pilot behaviour) ---
LAMBDA_M_DEFAULT = 0.5   # morph-novelty penalty magnitude (E[ord] units); CLI --lambda-m
BETA_DEFAULT = 0.02      # depth bonus per rung (E[ord] units); CLI --beta
CLIP_MODEL = "vit_base_patch16_clip_224.openai"  # matches the library morph_clip recipe
# The penalty knee is on the CHEAP-JPG substrate (not grayscale morph_gray), so the library
# morph_gray anchors do NOT transfer. Re-anchored empirically by morph_anchor_calibrate.py ->
# data/atlas/morph_anchors.json; these are only the last-resort fallback if that file is absent.
ANCHORS_PATH = ROOT / "data" / "atlas" / "morph_anchors.json"
MORPH_LO_FALLBACK = 0.85
MORPH_HI_FALLBACK = 0.974


def load_morph_anchors(cli_lo=None, cli_hi=None):
    """Resolve (lo, hi, source) for the novelty knee: CLI override > calibrated anchors file >
    fallback. Either CLI value alone overrides just that knee."""
    lo, hi, src = MORPH_LO_FALLBACK, MORPH_HI_FALLBACK, "fallback"
    if ANCHORS_PATH.exists():
        a = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
        lo, hi, src = float(a["lo"]), float(a["hi"]), "morph_anchors.json"
    if cli_lo is not None:
        lo, src = float(cli_lo), src + "+cli_lo"
    if cli_hi is not None:
        hi, src = float(cli_hi), src + "+cli_hi"
    if hi <= lo:
        hi = lo + 0.05
    return lo, hi, src
FRONTIER_CAP = 6000      # prune the frontier to the top-N by priority (memory bound)
JULIA_ROOT_FW = 3.0      # fixed z-plane base-scale root view (matches --julia-root-fw)
EXPAND_TIMEOUT_S = 900   # hard-kill backstop on a hung --expand call
ROOT_LOW_WATER = None    # replenish roots when frontier < this (set to B at runtime)

# Steered production walk config (mirror of production_seeder; keeps the gates identical).
EXPAND_FLAGS = [
    "--node-width", str(ps.NODE_WIDTH), "--sigma-band", ps.SIGMA_BAND,
    "--descent-occ-floor", str(ps.OCC_FLOOR), "--descent-black-cap", str(ps.BLACK_CAP),
]

FIDELITY_RECORDS = ROOT / "out" / "descent_score_fidelity_records.json"
C_PLANE = ("mandelbrot", "multibrot3", "multibrot4", "multibrot5")


# --------------------------------------------------------------------------- #
# Family <-> partition helpers (mirror production_seeder.resolve_family grammar).
# --------------------------------------------------------------------------- #
def render_family_of(partition: str) -> str:
    if partition == "mandelbrot" or partition in ("multibrot3", "multibrot4", "multibrot5"):
        return partition
    if partition == "julia:mandelbrot":
        return "julia"
    if partition.startswith("julia:multibrot"):
        return "julia_" + partition.split(":", 1)[1]
    raise ValueError(f"unknown partition {partition!r}")


def descend_flags(partition: str, c) -> list:
    """guided-descend --expand kernel flags for a homogeneous group (mirrors the walk grammar)."""
    if partition == "mandelbrot":
        return []
    if partition in ("multibrot3", "multibrot4", "multibrot5"):
        return ["--family", partition]
    if partition == "julia:mandelbrot":
        return ["--julia", "--c", str(c[0]), str(c[1])]
    if partition.startswith("julia:multibrot"):
        base = partition.split(":", 1)[1]
        return ["--family", base, "--julia", "--c", str(c[0]), str(c[1])]
    raise ValueError(f"unknown partition {partition!r}")


def loc_of(partition: str, c, cx, cy, fw):
    return ps.make_loc_of(render_family_of(partition), c)(cx, cy, fw)


# --------------------------------------------------------------------------- #
# tau_h — per-partition cheap p_good harvest cut from the fidelity study's paired scores.
# The cheap p_good cut that RETAINS ~90% of frames whose canonical p_good clears the
# family's t_good (= the 10th percentile of cheap p_good among those frames).
# --------------------------------------------------------------------------- #
def derive_tau_h(partitions: list[str], keep=0.90) -> dict:
    if not FIDELITY_RECORDS.exists():
        raise SystemExit(f"missing {FIDELITY_RECORDS} — run tools/studies/descent_score_fidelity.py")
    rec = json.loads(FIDELITY_RECORDS.read_text(encoding="utf-8"))
    can, cheap = rec["scores"]["canonical"], rec["scores"]["cheap"]
    fam_of = {s["id"]: s["family"] for s in rec["samples"]}
    q = 1.0 - keep

    def cut(ids):
        vals = [cheap[i][2] for i in ids if i in cheap and i in can]  # cheap p_good
        return float(np.quantile(vals, q)) if len(vals) >= 5 else None

    # pooled fallback over every frame clearing its own family's t_good.
    pooled_pass = [i for i in can
                   if can[i][2] >= ps.t_good_for(fam_of.get(i, "mandelbrot"))]
    pooled = cut(pooled_pass)
    if pooled is None:
        pooled = 0.5

    tau = {}
    for part in partitions:
        tg = ps.t_good_for(part)
        ids = [i for i in can if fam_of.get(i) == part and can[i][2] >= tg]
        tau[part] = cut(ids)
        if tau[part] is None:
            tau[part] = pooled
    return tau


# --------------------------------------------------------------------------- #
# Priority.
# --------------------------------------------------------------------------- #
def gumbel(rng: np.random.Generator, T: float) -> float:
    u = float(rng.random())
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return -T * math.log(-math.log(u))


def dup_penalty(cx, cy, cloud) -> float:
    """Large near an admitted q3, decaying (Gaussian, scale DUP_SCALE) with plane distance."""
    if not cloud:
        return 0.0
    d = min(math.hypot(cx - m["outcome_cx"], cy - m["outcome_cy"]) for m in cloud)
    return DUP_P0 * math.exp(-(d / DUP_SCALE) ** 2)


def priority_terms(eord, g, dup_pen, cos_max, lambda_m, beta, depth, lo, hi):
    """Pure priority decomposition. Returns (priority, {terms}). At lambda_m==0 AND beta==0 this
    is byte-identical to the pilot's `eord + gumbel - dup_pen` (novelty/depth terms vanish)."""
    nov_pen = novelty_penalty(cos_max, lambda_m, lo, hi)
    depth_bonus = beta * depth
    prio = eord + g - dup_pen - nov_pen + depth_bonus
    return prio, dict(eord=eord, gumbel=g, dup_pen=dup_pen, cos_max=cos_max,
                      nov_pen=nov_pen, depth_bonus=depth_bonus, priority=prio)


def novelty_penalty(cos_max: float, lambda_m: float, lo: float, hi: float) -> float:
    """Morph-space near-repeat penalty: zero at substrate-typical similarity (cos<=lo), ramping
    linearly to full lambda_m at the near-repeat knee (cos>=hi). Anchors are empirical on the
    cheap-JPG substrate (morph_anchor_calibrate.py). A near-perceptual-dup of an admitted/
    expanded look sinks by ~lambda_m E[ord] units BEFORE it is popped. lambda_m=0 -> zero."""
    if lambda_m <= 0.0:
        return 0.0
    frac = (cos_max - lo) / (hi - lo)
    return lambda_m * min(max(frac, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# Run-scoped morph memory — CLIP embeddings (library recipe) of admitted locations +
# already-expanded nodes, keyed to the cheap twilight presentation each candidate already
# carries (no extra render; the CLIP forward batches alongside the v7 forward). cos_max vs
# this set is the novelty signal. Embeddings are L2-normalized; the max-cosine reduction
# runs on the CLIP device. Persisted as a plain (M,768) matrix so a resume rebuilds it.
# --------------------------------------------------------------------------- #
class MorphMemory:
    def __init__(self, device: str, path: Path):
        self.device = device
        self.path = path
        self._np = np.zeros((0, 768), np.float32)  # normalized memory rows
        self._buf: list = []                        # pending rows not yet folded into _np/tensor
        self.mem = None                             # torch (M,768) on device (lazy)
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self._np = z["mem"].astype(np.float32)
        self._sync()

    def _sync(self):
        self.mem = torch.from_numpy(self._np).to(self.device) if len(self._np) else None

    def _flush(self):
        if not self._buf:
            return
        rows = [self._np] + self._buf if len(self._np) else self._buf
        self._np = np.concatenate([r.reshape(-1, 768) for r in rows], axis=0).astype(np.float32)
        self._buf = []
        self._sync()

    def add(self, emb: np.ndarray):
        """Fold one normalized embedding into memory (buffered; applied on next reduce/save)."""
        if emb is not None:
            self._buf.append(np.asarray(emb, np.float32).reshape(1, 768))

    def cos_max(self, embs: np.ndarray) -> np.ndarray:
        """Max cosine of each row of `embs` (normalized, N x 768) vs memory; 0 if empty."""
        self._flush()
        n = len(embs)
        if self.mem is None or n == 0:
            return np.zeros(n, np.float32)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(embs, np.float32)).to(self.device)
            c = (x @ self.mem.T).max(dim=1).values
        return c.float().cpu().numpy()

    def save(self):
        self._flush()
        if not len(self._np):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.stem + "_tmp.npz")
        np.savez_compressed(tmp, mem=self._np)
        os.replace(tmp, self.path)

    def __len__(self):
        return len(self._np) + len(self._buf)


# --------------------------------------------------------------------------- #
# Run-scoped ledger (append-only jsonl + atomic npz feature store). Schema parity with
# production's outcome_ledger.jsonl; the q3 cloud is rebuilt from these rows.
# --------------------------------------------------------------------------- #
class RunLedger:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.path = run_dir / "outcome_ledger.jsonl"
        self.feats_path = run_dir / "outcome_feats.npz"
        self.rows: list[dict] = []
        self.feats: dict = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if self.feats_path.exists():
            z = np.load(self.feats_path, allow_pickle=False)
            self.feats = {k: z[k] for k in z.files}

    def append(self, row: dict, feat):
        row.setdefault("scorer_version", ps.SCORER_VERSION)
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.rows.append(row)
        if feat is not None:
            self.feats[row["id"]] = np.asarray(feat, np.float32)

    def save_feats(self):
        if not self.feats:
            return
        tmp = self.feats_path.parent / (self.feats_path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **self.feats)
        os.replace(tmp, self.feats_path)

    def clouds(self, partitions: list[str]) -> dict:
        return {p: ps.build_cloud(self.rows, p) for p in partitions}


# --------------------------------------------------------------------------- #
# The driver.
# --------------------------------------------------------------------------- #
class SteeredFrontier:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        self.scratch = self.run_dir / "scratch"
        self.state_path = self.run_dir / "state.json"
        self.stop_path = self.run_dir / "STOP"
        self.harvest_log = self.run_dir / "harvest_log.jsonl"
        self.families = [f.strip() for f in args.families.split(",") if f.strip()]
        for f in self.families:
            if f not in C_PLANE:
                raise SystemExit(f"--families must be c-plane ({C_PLANE}); got {f!r}")
        self.julia_hook = bool(args.julia_hook)
        self.B = args.batch or B_DEFAULT
        self.budget_s = args.budget * 60.0
        self.seed = args.seed
        # v1.1 steering coefficients (both 0 -> byte-identical pilot behaviour).
        self.lambda_m = float(args.lambda_m)
        self.beta = float(args.beta)
        self.morph_lo, self.morph_hi, self.anchor_src = load_morph_anchors(
            args.morph_lo, args.morph_hi)
        self.prio_log = self.run_dir / "prio_terms.jsonl"

        # partitions this run tracks a cloud for (c-plane + julia twins if hooked).
        self.partitions = list(self.families)
        if self.julia_hook:
            self.partitions += [ps.julia_partition(f) for f in self.families]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)

        # Guarded scorer: cheap images (no field sidecar) pass through unguarded == raw
        # scoring; reframe tiles (DUMP_GUARD_FIELD) get the model-free field guard.
        assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
        reframe.DUMP_GUARD_FIELD = True
        self.scorer = guard.make_guarded_scorer(ps.SCORER_PATH)

        self.tau_h = derive_tau_h(self.partitions)

        # mutable run state
        self.frontier: list[dict] = []
        self.expansions_per_root: dict[str, int] = {}
        self.node_ctr = 0
        self.seq = 0
        self.batch_i = 0
        self.active_s = 0.0            # accumulated active wall time (survives resume)
        self.est_batch_s = 0.0
        self.totals = dict(expanded=0, candidates=0, harvest_checks=0,
                           canonical_q3=0, admitted=0, q3_dup=0, guarded=0,
                           julia_roots=0, cap_hits=0, dead_nodes=0, novelty_hits=0)
        self.rng = np.random.default_rng(self.seed)
        # per-family native seeders (root source) — re-created fresh on resume.
        self.seeders = {f: ps.NativeSeeder(self.seed, self.scratch / f"native_{f}",
                                           np.random.default_rng(self.seed + i + 1),
                                           self._flags(f))
                        for i, f in enumerate(self.families)}

        self.ledger = RunLedger(self.run_dir)
        self.clouds = self.ledger.clouds(self.partitions)   # rebuilt from the durable ledger
        self.hooked_c = defaultdict(list)                   # jpart -> [(c_re,c_im)] already hooked
        self.rebuild_hooked_c()

        # --- morph-novelty machinery (only when lambda_m > 0; off == pilot). ---
        self.clip_model = self.clip_tf = None
        self.node_embs: dict = {}                           # node_id -> normalized emb (frontier)
        clip_dev = "cpu"
        if self.lambda_m > 0.0:
            from tools.curation.colored_clip import load_clip   # noqa: E402  (heavy; lazy)
            self.clip_model, self.clip_tf = load_clip()
            clip_dev = str(next(self.clip_model.parameters()).device)
            self.node_embs = self.load_node_embs()
        self.morph = MorphMemory(clip_dev, self.run_dir / "morph_mem.npz")

    def rebuild_hooked_c(self):
        """Reconstruct the set of already-hooked julia parameters from the ledger (the
        durable record) so root-density rejection survives a resume."""
        self.hooked_c = defaultdict(list)
        for r in self.ledger.rows:
            fam = r.get("family", "")
            if fam.startswith("julia:") and r.get("julia_c_re") is not None:
                self.hooked_c[fam].append((float(r["julia_c_re"]), float(r["julia_c_im"])))

    # --- c-plane family_flags for the native seeder / probe ---
    @staticmethod
    def _flags(family: str) -> list:
        return [] if family == "mandelbrot" else ["--family", family]

    # ---------------------------------------------------------------- morph
    @property
    def node_embs_path(self) -> Path:
        return self.run_dir / "node_embs.npz"

    def load_node_embs(self) -> dict:
        """Reload frontier-node embeddings (node_id -> normalized emb) so a resume can fold a
        popped node into morph memory. Keyed by str(node_id)."""
        p = self.node_embs_path
        if not p.exists():
            return {}
        z = np.load(p, allow_pickle=False)
        return {int(k): z[k].astype(np.float32) for k in z.files}

    def save_node_embs(self):
        """Persist embeddings only for node_ids still on the frontier (drop popped/pruned)."""
        if self.lambda_m <= 0.0:
            return
        live = {n["node_id"] for n in self.frontier}
        keep = {str(k): v for k, v in self.node_embs.items() if k in live}
        p = self.node_embs_path
        tmp = p.parent / (p.stem + "_tmp.npz")
        np.savez_compressed(tmp, **keep)
        os.replace(tmp, p)

    @torch.no_grad()
    def clip_embed(self, imgs: list, bs: int = 64) -> np.ndarray:
        """L2-normalized CLIP embeddings (library recipe) of PIL RGB images (N x 768)."""
        outs = []
        for i in range(0, len(imgs), bs):
            xb = torch.stack([self.clip_tf(im) for im in imgs[i:i + bs]])
            xb = xb.to(next(self.clip_model.parameters()).device)
            outs.append(self.clip_model(xb).float().cpu().numpy())
        E = np.concatenate(outs, axis=0).astype(np.float32)
        E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E

    def fold_expanded_into_memory(self, batch):
        """A node that is about to be expanded joins morph memory (its cheap emb). Roots carry
        no emb and contribute nothing (they are whole-view seeds, not the near-repeats we damp)."""
        if self.lambda_m <= 0.0:
            return
        for n in batch:
            e = self.node_embs.pop(n["node_id"], None)
            if e is not None:
                self.morph.add(e)

    def score_morph(self, cands):
        """Embed each candidate's cheap twilight image, stash the normalized emb on the cand,
        and set cand['cos_max'] = max cosine vs the current morph memory (admitted + expanded).
        No-op (cos_max=0) when the novelty term is disabled."""
        for c in cands:
            c["cos_max"] = 0.0
            c["emb"] = None
        if self.lambda_m <= 0.0 or not cands:
            return
        imgs = []
        for c in cands:
            with Image.open(c["img"]) as im:
                im.load()
                imgs.append(im.convert("RGB"))
        E = self.clip_embed(imgs)
        cm = self.morph.cos_max(E)
        for c, e, v in zip(cands, E, cm):
            c["emb"] = e
            c["cos_max"] = float(v)

    def new_node_id(self) -> int:
        self.node_ctr += 1
        return self.node_ctr

    # ---------------------------------------------------------------- roots
    def draw_roots(self):
        """Draw a batch of native depth-1 seeds per family (q3-density rejection +
        depth-2 descendability probe) and enter the survivors as depth-1 frontier nodes
        with a neutral prior priority — exactly the current path's root pipeline."""
        added = 0
        for fam in self.families:
            cloud = self.clouds[fam]
            props = self.seeders[fam].draw_batch(cloud, self.B)
            if not props:
                continue
            pw = self.scratch / f"roots_b{self.batch_i:04d}_{fam}"
            survivors, rejects, _ = ps.depth2_probe(props, pw, self.seed, self._flags(fam))
            for sv in survivors:
                nid = self.new_node_id()
                self.frontier.append(dict(
                    node_id=nid, root_id=nid, partition=fam, c=None,
                    cx=float(sv["seed_cx"]), cy=float(sv["seed_cy"]), fw=float(sv["fw"]),
                    depth=1, priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
                    cheap_eord=None, cheap_pgood=None, branch="root",
                    mix_source=sv.get("mix_source", "native"),
                ))
                added += 1
        return added

    def add_julia_root(self, partition: str, c, parent_oid: str):
        """Julia hook: a fixed z-plane base-scale root at the parent's outcome `c` — the
        current path's julia hook, fired per qualifying (admitted-q3) c-plane parent.

        Adaptation vs production: the steered frontier explores the z-plane, so a julia
        partition's OUTCOME cloud is keyed on the z-viewport (correct image-distinctness +
        steering penalty). Root spawning is instead gated by the PARAMETER c against a
        separate `hooked_c` set (so the same c is not re-hooked) — production keys its julia
        cloud on c directly; here the two roles are split."""
        jpart = ps.julia_partition(partition)
        cr, ci = float(c[0]), float(c[1])
        hooked = self.hooked_c[jpart]
        if sum(1 for (hr, hi) in hooked if math.hypot(hr - cr, hi - ci) < ps.REJECT_RADIUS) \
                >= ps.Q3_DENSITY_CAP:
            return False
        hooked.append((cr, ci))
        nid = self.new_node_id()
        self.frontier.append(dict(
            node_id=nid, root_id=nid, partition=jpart, c=[str(c[0]), str(c[1])],
            cx=0.0, cy=0.0, fw=JULIA_ROOT_FW, depth=1,
            priority=NEUTRAL_PRIOR + gumbel(self.rng, T_GUMBEL),
            cheap_eord=None, cheap_pgood=None, branch="julia_root",
            mix_source=f"julia_hook<{parent_oid}", parent_oid=parent_oid,
        ))
        self.totals["julia_roots"] += 1
        return True

    # ---------------------------------------------------------------- expand
    def pop_batch(self) -> list[dict]:
        """Top-B expandable nodes by priority. A node whose root has hit the M cap can NEVER be
        expanded, so it is EVICTED from the frontier (not merely skipped): a capped root spawns
        ~M*b children before capping, so if capped nodes are retained they accumulate ~b faster
        than they drain and eventually saturate FRONTIER_CAP, starving pop_batch and forcing
        perpetual root replenishment (observed live at batch ~110: 100% of a 6000-node frontier
        was capped-root dead weight, throughput ~0). Eviction is a no-op below the cap regime the
        pilot ran in (few caps, frontier << cap), so it does not change short-run behaviour."""
        self.frontier.sort(key=lambda n: -n["priority"])
        batch, rest = [], []
        for n in self.frontier:
            if self.expansions_per_root.get(str(n["root_id"]), 0) >= M_CAP:
                self.node_embs.pop(n["node_id"], None)   # evict: capped root -> dead weight
                continue
            if len(batch) < self.B:
                batch.append(n)
            else:
                rest.append(n)
        self.frontier = rest
        # cap_hits = distinct roots that have reached the M cap (derived, not per-batch).
        self.totals["cap_hits"] = sum(1 for v in self.expansions_per_root.values() if v >= M_CAP)
        for n in batch:
            self.expansions_per_root[str(n["root_id"])] = \
                self.expansions_per_root.get(str(n["root_id"]), 0) + 1
        return batch

    def expand_group(self, key, nodes) -> list[dict]:
        partition, c = key
        gdir = self.scratch / f"expand_b{self.batch_i:04d}" / f"{partition.replace(':','_')}"
        gdir.mkdir(parents=True, exist_ok=True)
        nodes_in = gdir / "nodes.jsonl"
        with open(nodes_in, "w", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps(dict(node_id=n["node_id"], root_id=n["root_id"],
                                        cx=n["cx"], cy=n["cy"], fw=n["fw"], depth=n["depth"])) + "\n")
        cmd = [str(BIN), "guided-descend", "--expand", str(nodes_in),
               "--seed", str(self.seed), "--out-dir", str(gdir)] + EXPAND_FLAGS + \
              descend_flags(partition, c)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=EXPAND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"  WARN expand group {partition} timed out ({EXPAND_TIMEOUT_S}s) — skipped", flush=True)
            return []
        if r.returncode != 0:
            print(f"  WARN expand group {partition} failed: {r.stderr[-400:]}", flush=True)
            return []
        by_node = {n["node_id"]: n for n in nodes}
        cands = []
        ep = gdir / "expand.jsonl"
        if not ep.exists():
            return []
        for line in open(ep, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parent = by_node[row["node_id"]]
            if row["kind"] == "dead":
                self.totals["dead_nodes"] += 1
                continue
            cands.append(dict(
                node_id=self.new_node_id(), root_id=parent["root_id"],
                partition=partition, c=c,
                cx=float(row["cx"]), cy=float(row["cy"]), fw=float(row["fw"]),
                depth=int(row["depth"]), branch=row["branch"],
                img=str((gdir / row["img"]).resolve()),
                int_frac=row["int_frac"], occ=row["occ"],
            ))
        return cands

    def expand_batch(self, batch) -> list[dict]:
        # group by (partition, tuple(c)) so each --expand call is homogeneous in kernel.
        groups: dict = {}
        for n in batch:
            key = (n["partition"], tuple(n["c"]) if n["c"] else None)
            groups.setdefault(key, []).append(n)
        cands = []
        for key, nodes in groups.items():
            cands += self.expand_group(key, nodes)
        return cands

    # ---------------------------------------------------------------- score
    def score_cheap(self, cands):
        if not cands:
            return
        triples = self.scorer.score_paths([c["img"] for c in cands])
        for c, (eord, nb, pg) in zip(cands, triples):
            c["cheap_eord"] = float(eord)
            c["cheap_nb"] = float(nb)
            c["cheap_pgood"] = float(pg)

    # ---------------------------------------------------------------- harvest
    def harvest(self, cands):
        """cheap p_good >= tau_h -> single canonical render + decode -> if q3, reframe +
        near-dup + admission. Logs every harvest check's (cheap, canonical, decode) triple."""
        checks = [c for c in cands if c["cheap_pgood"] >= self.tau_h[c["partition"]]]
        if not checks:
            return
        self.totals["harvest_checks"] += len(checks)
        # 1. batch the single canonical confirmation renders (640x360 ss2, the reward fidelity).
        cdir = self.scratch / f"harvest_b{self.batch_i:04d}"
        cdir.mkdir(parents=True, exist_ok=True)
        import concurrent.futures as cf
        tiles = []
        for i, c in enumerate(checks):
            tiles.append(cdir / f"confirm_{i:04d}.jpg")
        with cf.ThreadPoolExecutor(max_workers=ps.WORKERS) as ex:
            futs = {ex.submit(prescreen._render, c["cx"], c["cy"], c["fw"], tiles[i],
                              family=render_family_of(c["partition"]), c=c["c"]): i
                    for i, c in enumerate(checks)}
            for fut in cf.as_completed(futs):
                fut.result()
        triples = self.scorer.score_paths([str(t) for t in tiles])
        for c, (eord, nb, pg) in zip(checks, triples):
            c["canon_nb"], c["canon_pg"], c["canon_eord"] = float(nb), float(pg), float(eord)
            c["canon_decoded"] = corn_decode(nb, pg, ps.t_good_for(c["partition"]))

        # 2. reframe + admit the canonical-q3 confirmations. Cheap pre-reframe dedup:
        # reframe only nudges the center by <=0.25*fw and fw by <=1.41x, so a candidate
        # already inside an admitted q3's dedup radius cannot escape it — skip the 12-render
        # reframe and log it as a dup (this is where most compute is saved in a hot region).
        for c in checks:
            admitted = False
            reframe_decoded = None
            if c["canon_decoded"] == 3:
                self.totals["canonical_q3"] += 1
                pre_distinct, _ = ps.is_distinct(c["cx"], c["cy"], c["fw"],
                                                 self.clouds.get(c["partition"], []), ps.DEDUP_K)
                if not pre_distinct:
                    self.totals["q3_dup"] += 1
                else:
                    admitted, reframe_decoded = self.admit(c, cdir)
            self._log_harvest(c, admitted, reframe_decoded)

    def admit(self, c, cdir):
        """Existing reframe + near-dup + admission path (guarded scorer, per-partition t_good)."""
        loc = loc_of(c["partition"], c["c"], c["cx"], c["cy"], c["fw"])
        wd = cdir / f"reframe_n{c['node_id']}"
        res = reframe.reframe_location(loc, scorer=self.scorer, seed=0, workdir=wd, workers=ps.WORKERS)
        guard_pass = res.score > guard.GUARD_SENTINEL + 1e-6
        nb, pg = ps._chosen_probs(res)
        t_good = ps.t_good_for(c["partition"])
        decoded = corn_decode(nb, pg, t_good) if guard_pass else None
        is_q3 = guard_pass and decoded == 3
        ocx, ocy, ofw = float(res.cx), float(res.cy), float(res.fw)
        distinct, dup_of = (False, None)
        if is_q3:
            distinct, dup_of = ps.is_distinct(ocx, ocy, ofw, self.clouds[c["partition"]], ps.DEDUP_K)

        run_ts = self.run_dir.name
        id_tag = {"mandelbrot": "m"}.get(c["partition"], c["partition"].replace(":", "_"))
        oid = f"st_{id_tag}_{run_ts}_{self.seq:06d}"
        self.seq += 1
        feat = None
        if is_q3 and distinct:
            tile = cdir / f"{oid}.jpg"
            feat = ps.outcome_feature(self.scorer, ocx, ocy, ofw, tile,
                                      family=render_family_of(c["partition"]), c=c["c"])
        row = dict(
            id=oid, ts=run_ts, family=c["partition"], mix_source="steered",
            node_id=c["node_id"], root_id=c["root_id"],
            seed_cx=c["cx"], seed_cy=c["cy"],
            outcome_cx=ocx, outcome_cy=ocy, outcome_fw=ofw,
            k3=float(res.score), raw_top3=[float(c["cheap_eord"])],
            reached_depth=int(c["depth"]),
            decoded_class=decoded, p_notbad=nb, p_good=pg, t_good=t_good,
            distinct=distinct, dup_of=dup_of,
            guard_pass=guard_pass, guard_fail=None if guard_pass else "sentinel",
            cheap_pgood=c["cheap_pgood"], canon_pgood=c["canon_pg"], branch=c["branch"],
        )
        if c["c"] is not None:                       # julia twin outcome carries the parameter c
            row["julia_c_re"], row["julia_c_im"] = c["c"][0], c["c"][1]
        self.ledger.append(row, feat)
        if is_q3 and distinct:
            self.clouds[c["partition"]].append(row)
            self.totals["admitted"] += 1
            # fold the admitted location's look into morph memory (its cheap emb; reframe only
            # nudges the frame <=0.25*fw, so the candidate's cheap look stands in for it).
            if self.lambda_m > 0.0 and c.get("emb") is not None:
                self.morph.add(c["emb"])
            # julia hook: fire per qualifying (admitted-q3) c-plane parent.
            if self.julia_hook and c["partition"] in self.families:
                self.add_julia_root(c["partition"], (ocx, ocy), oid)
            return True, decoded
        elif is_q3:
            self.totals["q3_dup"] += 1
        elif not guard_pass:
            self.totals["guarded"] += 1
        return False, decoded

    def _log_harvest(self, c, admitted, reframe_decoded):
        with open(self.harvest_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(
                batch=self.batch_i, partition=c["partition"], depth=c["depth"],
                node_id=c["node_id"], root_id=c["root_id"],
                cheap_pgood=c["cheap_pgood"], cheap_eord=c["cheap_eord"],
                canon_nb=c.get("canon_nb"), canon_pgood=c.get("canon_pg"),
                canon_decoded=c.get("canon_decoded"), reframe_decoded=reframe_decoded,
                admitted=bool(admitted), tau_h=self.tau_h[c["partition"]],
            )) + "\n")

    # ---------------------------------------------------------------- push
    def push_children(self, cands):
        prio_rows = []
        for c in cands:
            dup_pen = dup_penalty(c["cx"], c["cy"], self.clouds.get(c["partition"], []))
            cos_max = float(c.get("cos_max", 0.0))
            g = gumbel(self.rng, T_GUMBEL)               # RNG draw order unchanged from pilot
            prio, terms = priority_terms(
                c["cheap_eord"], g, dup_pen, cos_max,
                self.lambda_m, self.beta, c["depth"], self.morph_lo, self.morph_hi)
            self.frontier.append(dict(
                node_id=c["node_id"], root_id=c["root_id"], partition=c["partition"], c=c["c"],
                cx=c["cx"], cy=c["cy"], fw=c["fw"], depth=c["depth"], priority=prio,
                cheap_eord=c["cheap_eord"], cheap_pgood=c["cheap_pgood"], branch=c["branch"],
            ))
            if c.get("emb") is not None:
                self.node_embs[c["node_id"]] = c["emb"]
            if terms["nov_pen"] > 0.0:
                self.totals["novelty_hits"] += 1
            prio_rows.append(dict(
                batch=self.batch_i, node_id=c["node_id"], root_id=c["root_id"],
                partition=c["partition"], depth=c["depth"],
                **{k: round(v, 5) for k, v in terms.items()},
            ))
        if prio_rows:
            with open(self.prio_log, "a", encoding="utf-8") as f:
                for r in prio_rows:
                    f.write(json.dumps(r) + "\n")
        # prune to the memory bound (keep the best); drop pruned nodes' cached embeddings.
        if len(self.frontier) > FRONTIER_CAP:
            self.frontier.sort(key=lambda n: -n["priority"])
            dropped = self.frontier[FRONTIER_CAP:]
            self.frontier = self.frontier[:FRONTIER_CAP]
            for n in dropped:
                self.node_embs.pop(n["node_id"], None)

    # ---------------------------------------------------------------- state
    def save_state(self):
        state = dict(
            run_ts=self.run_dir.name, families=self.families, julia_hook=self.julia_hook,
            seed=self.seed, B=self.B, budget_s=self.budget_s, tau_h=self.tau_h,
            lambda_m=self.lambda_m, beta=self.beta,
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            node_ctr=self.node_ctr, seq=self.seq, batch_i=self.batch_i,
            active_s=self.active_s, est_batch_s=self.est_batch_s,
            expansions_per_root=self.expansions_per_root, totals=self.totals,
            frontier=self.frontier, rng=self.rng.bit_generator.state,
        )
        # morph memory + frontier-node embeddings first (state.json references them), then the
        # checkpoint. Both are heuristic (priority only) — a stale copy never loses an admission.
        self.morph.save()
        self.save_node_embs()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, self.state_path)
        self.ledger.save_feats()

    def load_state(self):
        st = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.node_ctr = st["node_ctr"]; self.seq = st["seq"]; self.batch_i = st["batch_i"]
        self.active_s = st["active_s"]; self.est_batch_s = st["est_batch_s"]
        self.expansions_per_root = st["expansions_per_root"]; self.totals = st["totals"]
        self.frontier = st["frontier"]; self.tau_h = st["tau_h"]
        self.totals.setdefault("novelty_hits", 0)
        self.rng.bit_generator.state = st["rng"]
        # cloud is rebuilt from the DURABLE ledger (source of truth), not the checkpoint,
        # so a kill between ledger-append and checkpoint cannot lose/duplicate an admission.
        self.clouds = self.ledger.clouds(self.partitions)
        self.rebuild_hooked_c()
        # morph memory (+ frontier-node embeddings) reload from their npz sidecars.
        if self.lambda_m > 0.0:
            self.node_embs = self.load_node_embs()
        print(f"[resume] batch {self.batch_i}, frontier {len(self.frontier)}, "
              f"active {self.active_s/60:.1f}m, admitted {self.totals['admitted']} "
              f"(cloud rebuilt from ledger: "
              f"{sum(len(v) for v in self.clouds.values())} places)", flush=True)

    # ---------------------------------------------------------------- run
    def run(self):
        global ROOT_LOW_WATER
        ROOT_LOW_WATER = self.B
        if self.args.resume and self.state_path.exists():
            self.load_state()
        else:
            print(f"[fresh] run {self.run_dir.name}: families={self.families} "
                  f"julia_hook={self.julia_hook} budget={self.budget_s/60:.0f}m B={self.B} "
                  f"lambda_m={self.lambda_m} beta={self.beta}", flush=True)
            if self.lambda_m > 0.0:
                print(f"[morph-anchors] lo={self.morph_lo:.4f} hi={self.morph_hi:.4f} "
                      f"({self.anchor_src})", flush=True)
            print(f"[tau_h] {self.tau_h}", flush=True)
            self.draw_roots()
            self.save_state()

        while True:
            if self.stop_path.exists():
                print("[STOP] sentinel present — halting at batch boundary.", flush=True)
                break
            if self.budget_s and self.active_s + self.est_batch_s > self.budget_s:
                print(f"[budget] active {self.active_s/60:.1f}m + est batch "
                      f"{self.est_batch_s:.0f}s would exceed {self.budget_s/60:.0f}m — stopping.", flush=True)
                break
            if len(self.frontier) < ROOT_LOW_WATER:
                self.draw_roots()
            if not self.frontier:
                print("[frontier] empty and no fresh roots — stopping.", flush=True)
                break

            tb = time.time()
            self.batch_i += 1
            batch = self.pop_batch()
            if not batch:
                # everything capped; try fresh roots, else stop.
                if self.draw_roots() == 0:
                    print("[frontier] all roots capped (M) and no fresh seeds — stopping.", flush=True)
                    break
                self.batch_i -= 1
                continue
            self.fold_expanded_into_memory(batch)   # parents join morph memory before scoring
            self.totals["expanded"] += len(batch)
            cands = self.expand_batch(batch)
            self.totals["candidates"] += len(cands)
            self.score_cheap(cands)
            self.score_morph(cands)                  # embed + cos_max vs memory (parents incl.)
            self.harvest(cands)                      # admissions fold into memory
            self.push_children(cands)                # novelty penalty applied from cos_max

            dt = time.time() - tb
            self.active_s += dt
            self.est_batch_s = dt if self.est_batch_s == 0 else 0.5 * self.est_batch_s + 0.5 * dt
            self.save_state()
            if self.batch_i % 1 == 0:
                print(f"  batch {self.batch_i}: exp={len(batch)} cand={len(cands)} "
                      f"admitted(cum)={self.totals['admitted']} julia_roots={self.totals['julia_roots']} "
                      f"frontier={len(self.frontier)} | {dt:.0f}s active={self.active_s/60:.1f}m", flush=True)

        self.finish()

    def finish(self):
        self.save_state()
        summary = dict(
            run_ts=self.run_dir.name, mode="steered", families=self.families,
            julia_hook=self.julia_hook, budget_min=self.budget_s / 60.0,
            lambda_m=self.lambda_m, beta=self.beta, morph_mem=len(self.morph),
            morph_lo=self.morph_lo, morph_hi=self.morph_hi, anchor_src=self.anchor_src,
            active_min=round(self.active_s / 60.0, 2), batches=self.batch_i,
            tau_h=self.tau_h, totals=self.totals,
            cloud_sizes={p: len(v) for p, v in self.clouds.items()},
        )
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n=== STEERED FRONTIER SUMMARY ===")
        print(f"  active {self.active_s/60:.1f}m over {self.batch_i} batches")
        print(f"  expanded={self.totals['expanded']} candidates={self.totals['candidates']} "
              f"harvest_checks={self.totals['harvest_checks']} canonical_q3={self.totals['canonical_q3']}")
        print(f"  ADMITTED distinct q3={self.totals['admitted']}  q3_dup={self.totals['q3_dup']} "
              f"guarded={self.totals['guarded']} julia_roots={self.totals['julia_roots']} "
              f"cap_hits={self.totals['cap_hits']}")
        print(f"  lambda_m={self.lambda_m} beta={self.beta} novelty_hits={self.totals['novelty_hits']} "
              f"morph_mem={len(self.morph)}")
        print(f"  cloud: {summary['cloud_sizes']}")
        print(f"  ledger -> {self.ledger.path}\n  summary -> {self.run_dir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="fresh run-scoped dir (ledger + state.json)")
    ap.add_argument("--families", default="mandelbrot,multibrot3,multibrot4,multibrot5")
    ap.add_argument("--julia-hook", action="store_true")
    ap.add_argument("--budget", type=float, default=45.0, help="active-time budget (minutes)")
    ap.add_argument("--batch", type=int, default=0, help="nodes per batch (0 = default 32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-m", type=float, default=LAMBDA_M_DEFAULT,
                    help="morph-novelty penalty magnitude (0 disables; == pilot)")
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT,
                    help="depth bonus per rung (0 disables; == pilot)")
    ap.add_argument("--morph-lo", type=float, default=None,
                    help="override the zero-penalty cos knee (default: calibrated anchors file)")
    ap.add_argument("--morph-hi", type=float, default=None,
                    help="override the full-penalty cos knee (default: calibrated anchors file)")
    ap.add_argument("--resume", action="store_true", help="continue from state.json")
    args = ap.parse_args()
    SteeredFrontier(args).run()


if __name__ == "__main__":
    main()
