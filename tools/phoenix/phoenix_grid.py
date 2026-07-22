#!/usr/bin/env python
r"""Phoenix Phase B — the seed grid (variance-decomposition run + intake-ready admissions
ledger). Governing docs: prompts/phoenix_phase_b.md, docs/design/phoenix_seed_sampler_spec.md
§5.1 (step 0), docs/findings/phoenix_z_m1_symmetry.md.

One backgrounded, resume-safe GPU run: ~N seeds (stratified draw over the proposal axes) x
K descents each (distinct descent RNG per repeat), scored through the EXACT production reward
path (`production_seeder.harvest_walk_reward` -> canonical p_good / CORN decode), keeping every
q3 (guard-clean, decoded_class==3 at the configured phoenix t_good). Per descent it records the
between/within-decomposition variables (distinct-looks-per-descent, max-p_good) plus admissions,
depth, active seconds; every admitted q3 outcome is stamped with the full (c,p,z_-1) identity and
its 1280-D feature so the ledger is library-intake-ready.

Scope: plain execution of a FIXED stratified plan. No surrogate, no fertility memory, no measure
changes, no scheduler (spec §5.2 is DEFERRED — its go/no-go is what this run's decomposition, and
the human labels, decide). The variance-decomposition itself is `phoenix_decomp.py`; the label
batch is `phoenix_label_batch.py`; both read this run's durable outputs.

Durable outputs under `data/discovery/phoenix_grid/<run>/` (survive `rm -r out/*`):
  seeds.jsonl            one row per drawn seed (identity + branch/theta/offset + stratum + features)
  descent_records.jsonl  one row per (seed, repeat) descent — the decomposition unit + resume key
  all_outcomes.jsonl     one row per scored walk (q3 AND sub-threshold AND reject) — label-batch source
  outcome_ledger.jsonl   admitted q3 outcomes only, intake-ready (identity + p_good + guard + distinct)
  outcome_feats.npz      id -> 1280-D v7 penultimate feature for each admitted q3 (library intake)
  distinct_looks.npz     the run-global morph-embed distinct-look tally (cos 0.974), resumable
  summary.json           config + totals + realized min-per-look price + stratification plan
Disposable render scratch: out/phoenix/grid/<run>/ (purged on clean exit unless --keep-scratch).

  uv run python tools/phoenix/phoenix_grid.py --smoke            # 3 seeds x 2, tiny — validates the pipeline
  uv run python tools/phoenix/phoenix_grid.py --run --budget 240 # the real 4-hour-cap grid (backgrounded)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))                                  # phoenix_sampler
sys.path.insert(0, str(ROOT / "tools"))                        # (namespace) tools.*
sys.path.insert(0, str(ROOT / "tools" / "atlas"))             # production_seeder, prescreen, guard, deficit_scheduler
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))       # step0_reanalysis
sys.path.insert(0, str(ROOT / "tools" / "reframe"))           # reframe
sys.path.insert(0, str(ROOT / "tools" / "scoring"))           # active_ckpt
sys.path.insert(0, str(ROOT / "tools" / "mining"))            # score_lib
sys.path.insert(0, str(ROOT / "tools" / "corpus"))            # location
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))         # library_annotate
sys.path.insert(0, str(ROOT / "tools" / "curation"))          # colored_clip

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import phoenix_sampler as psamp             # noqa: E402
import production_seeder as ps              # noqa: E402  (reward path, guard, identity, config)
import prescreen                            # noqa: E402  (BIN, embed_paths, render constants)
import guard                                # noqa: E402
import reframe                              # noqa: E402
from score_lib import corn_decode           # noqa: E402
from step0_reanalysis import load_frames_by_walk  # noqa: E402
from deficit_scheduler import DistinctLookTally    # noqa: E402  (cos-0.974 tally)
import location as loc_mod                   # noqa: E402

# =========================================================================== #
# Config
# =========================================================================== #
# Phoenix descent walk config = production (matches production_seeder so p_good is comparable).
NODE_WIDTH = ps.NODE_WIDTH
SIGMA_BAND = ps.SIGMA_BAND
DEPTH_MIN, DEPTH_MAX = ps.DEPTH_MIN, ps.DEPTH_MAX
OCC_FLOOR, BLACK_CAP = ps.OCC_FLOOR, ps.BLACK_CAP
WORKERS = ps.WORKERS
SCORER_PATH, SCORER_VERSION = ps.SCORER_PATH, ps.SCORER_VERSION

# Phoenix q3 operating point. Source it from the production table so the grid always runs at
# whatever the seeder is currently calibrated to (label-derived, now 0.45 — see production_seeder
# T_GOOD_OVERRIDES / docs/findings/phoenix_grid_labels.md §2). The original grid ran at a hardcoded
# 0.18 that was NEVER a production value — a stale copy of the retired v6-era provisional that
# nobody ordered; it admitted ~everything (in-batch precision 0.19). Raw p_good is stored per
# outcome, so any t_good re-decodes for free (tools/phoenix/redecode_grid.py). --t-good overrides.
T_GOOD_DEFAULT = ps.t_good_for("phoenix")

NEAR_DUP_THRESHOLD = 0.974   # morph-embed distinct-look cosine (matches the library / scheduler)
DEDUP_K = ps.DEDUP_K         # cloud viewport dedup (identity-aware via near_dup)

DISCOVERY_ROOT = ROOT / "data" / "discovery" / "phoenix_grid"
SCRATCH_ROOT = ROOT / "out" / "phoenix" / "grid"


# =========================================================================== #
# Stratified seed draw (deliberate coverage of the proposal axes)
# =========================================================================== #
# Strata cross three axes the decomposition + labels must cover deliberately (spec §5.1, §3):
#   |p| band       — 3 bands over the |p|<1 disk (near-M perturbation -> larger |p|)
#   branch         — cardioid / period2 / root (the skeleton locus sampled)
#   z_-1 class     — zero / real-nonzero / non-real (the symmetry lever; non-real breaks reflection)
# arg p is covered incidentally by the uniform-disk p draw (logged per seed, reported as coverage).
P_BAND_EDGES = (0.33, 0.66)   # -> bands 0:[0,.33) 1:[.33,.66) 2:[.66,.95]


def p_band(absp: float) -> int:
    return 0 if absp < P_BAND_EDGES[0] else (1 if absp < P_BAND_EDGES[1] else 2)


def z_class(z: complex) -> str:
    if abs(z) < 1e-12:
        return "zero"
    return "real" if abs(z.imag) < 1e-9 else "nonreal"


def stratum_of(s: psamp.Seed) -> tuple:
    return (p_band(abs(s.p)), s.branch, z_class(s.z_m1))


def stratum_key(st: tuple) -> str:
    return f"p{st[0]}|{st[1]}|z_{st[2]}"


def stratified_seeds(seed: int, n: int, pool_mult: int = 24) -> tuple[list, dict]:
    """Draw `n` seeds with deliberate stratum coverage: over-propose a pool, bucket by
    stratum, then round-robin across non-empty strata for maximum spread and EXACT `n`.
    Deterministic in `seed`. Returns (seeds, plan) where plan records realized per-stratum
    counts + the axis marginals (the stratification plan for the readout)."""
    rng = np.random.default_rng(seed)
    pool = [psamp.propose_seed(rng) for _ in range(pool_mult * n)]
    buckets: dict[tuple, list] = defaultdict(list)
    for s in pool:
        buckets[stratum_of(s)].append(s)
    keys = sorted(buckets)
    selected: list[psamp.Seed] = []
    i = 0
    while len(selected) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            selected.append(buckets[k].pop(0))
        i += 1
    plan = {
        "n_requested": n, "n_drawn": len(selected), "pool_size": len(pool),
        "n_strata_observed": len(keys),
        "stratum_counts": dict(Counter(stratum_key(stratum_of(s)) for s in selected)),
        "branch_counts": dict(Counter(s.branch for s in selected)),
        "p_band_counts": dict(Counter(f"p{p_band(abs(s.p))}" for s in selected)),
        "z_class_counts": dict(Counter(z_class(s.z_m1) for s in selected)),
        "classic_frac": float(np.mean([s.classic for s in selected])) if selected else 0.0,
        "frac_p_complex": float(np.mean([abs(s.p.imag) > 1e-9 for s in selected])) if selected else 0.0,
    }
    return selected, plan


# =========================================================================== #
# Phoenix location factories (both carry the full (c,p,z_-1) identity)
# =========================================================================== #
def _fp(s: psamp.Seed) -> dict:
    return {"p_re": repr(s.p.real), "p_im": repr(s.p.imag),
            "zm1_re": repr(s.z_m1.real), "zm1_im": repr(s.z_m1.imag)}


def make_phoenix_loc_of(s: psamp.Seed):
    """Reward-path location factory (reframe.Location) at this seed's (c,p,z_-1). Threaded
    into harvest_walk_reward so the raw-screen + reframe renders carry the seed's parameters
    (make_loc_of drops family_params — good only for the fixed-Ushiki plane)."""
    fp = _fp(s)
    c_re, c_im = repr(s.c.real), repr(s.c.imag)

    def loc_of(cx, cy, fw):
        return reframe.Location(family="phoenix", c_re=c_re, c_im=c_im,
                                cx=str(cx), cy=str(cy), fw=str(fw), family_params=fp)
    return loc_of


def phoenix_canonical_loc(cx, cy, fw, s: psamp.Seed) -> loc_mod.Location:
    """Canonical Location (has .maxiter/.key) for the morph field dump + label render."""
    return loc_mod.Location(
        family="phoenix", cx=str(cx), cy=str(cy), fw=str(fw),
        maxiter=int(prescreen.auto_maxiter(float(fw))),
        c_re=repr(s.c.real), c_im=repr(s.c.imag), family_params=_fp(s))


def phoenix_outcome_feature(scorer, cx, cy, fw, s: psamp.Seed, tile: Path) -> np.ndarray:
    """Render the admitted outcome crop at v7 search fidelity (640x360 ss2, twilight_shifted)
    carrying the seed's (c,p,z_-1), forward it through the v7 penultimate hook -> 1280-D. Mirrors
    production_seeder.outcome_feature but threads phoenix family_params (which prescreen._render
    hardcodes empty)."""
    tile.parent.mkdir(parents=True, exist_ok=True)
    loc = reframe.Location(family="phoenix", c_re=repr(s.c.real), c_im=repr(s.c.imag),
                           cx=str(cx), cy=str(cy), fw=str(fw), family_params=_fp(s))
    cmd = [
        str(prescreen.BIN), "render-one", "--cx", str(cx), "--cy", str(cy),
        "--fw", repr(float(fw)), "--width", str(prescreen.RENDER_W),
        "--height", str(prescreen.RENDER_H), "--supersample", str(prescreen.RENDER_SS),
        "--maxiter", str(prescreen.auto_maxiter(float(fw))), "--palette", prescreen.PALETTE,
        "--jpg-quality", str(prescreen.JPG_Q), "--out", str(tile),
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not (r.returncode == 0 and tile.exists()):
        raise RuntimeError(f"outcome tile render failed [{tile.name}]: {r.stderr[-300:]}")
    return prescreen.embed_paths(scorer, [tile])[0]


# =========================================================================== #
# Morph-embed distinct look (cos 0.974) — the library/scheduler recipe
# =========================================================================== #
class MorphEmbedder:
    """Lazy CLIP + the library morph recipe (640x360 ss2 smooth field -> robust-z tanh gray
    -> CLIP, L2-normalized) at a phoenix location's (c,p,z_-1). One embed per admitted outcome."""

    def __init__(self, scratch: Path):
        self.scratch = scratch
        self._mt = None

    def _clip(self):
        if self._mt is None:
            from colored_clip import load_clip
            self._mt = load_clip()
        return self._mt

    def embed(self, cx, cy, fw, s: psamp.Seed, fcache: Path) -> np.ndarray:
        import library_annotate as la
        from colored_clip import embed_clip
        loc = phoenix_canonical_loc(cx, cy, fw, s)
        field = la.ensure_field(loc, retain=False, tmp_dir=fcache, cache_root=fcache)
        gray = la.morph_gray_image(field)
        model, tf = self._clip()
        e = embed_clip(model, tf, [gray])[0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


# =========================================================================== #
# One scored descent (one guided-descend run at the seed's (c,p,z_-1))
# =========================================================================== #
def run_phoenix_seed_descent(s: psamp.Seed, rng_seed: int, n_walks: int, workdir: Path) -> Path:
    """Native phoenix z-plane descent from the seed's standard start view (base-scale, center 0)
    at production depth, per-walk RNG. Mirrors production_seeder.run_phoenix_descent but stamps
    the seed's (c,p,z_-1) via --c/--p/--phoenix-z1."""
    workdir.mkdir(parents=True, exist_ok=True)
    pool = workdir / "pool"
    cmd = [
        str(prescreen.BIN), "guided-descend", "--phoenix",
        "--c", repr(s.c.real), repr(s.c.imag),
        "--p", repr(s.p.real), repr(s.p.imag),
        "--phoenix-z1", repr(s.z_m1.real), repr(s.z_m1.imag),
        "--n-walks", str(n_walks), "--seed", str(rng_seed), "--per-walk-rng",
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"phoenix descent failed (c={s.c}, p={s.p}, z={s.z_m1}):\n{r.stderr[-1200:]}")
    return pool


def score_descent(s: psamp.Seed, seed_idx: int, repeat: int, rng_seed: int, *, scorer,
                  embedder: MorphEmbedder, n_walks: int, scratch: Path, t_good: float):
    """Run + score one descent. Returns (record, outcome_rows, admitted_rows) where:
      record      — the per-descent decomposition unit (distinct_looks_within, max_p_good, ...)
      outcome_rows— every scored walk (q3/sub-threshold/reject) for the label-batch source
      admitted_rows — (row, feat, emb) for each q3 admission (intake ledger + global tally)
    active_seconds spans descent + scoring."""
    t0 = time.time()
    dwork = scratch / f"s{seed_idx:04d}_r{repeat}"
    pool = run_phoenix_seed_descent(s, rng_seed, n_walks, dwork)
    by_walk = load_frames_by_walk(pool)
    loc_of = make_phoenix_loc_of(s)
    ident = ps.phoenix_ident_fields((s.c.real, s.c.imag), (s.p.real, s.p.imag),
                                    (s.z_m1.real, s.z_m1.imag))

    outcome_rows, admitted = [], []
    within_embs: list[np.ndarray] = []          # within-descent distinct-look dedup (order-independent count)
    max_p_good, max_depth, n_adm, walk_errors = 0.0, 0, 0, 0
    for wi, wid in enumerate(sorted(by_walk)):
        try:
            frames = by_walk[wid]
            rew = ps.harvest_walk_reward(scorer, wid, frames, WORKERS, dwork / "reward", loc_of)
        except (SystemExit, Exception) as e:
            walk_errors += 1
            print(f"    WARN s{seed_idx} r{repeat} walk {wid} skipped: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            continue
        guard_pass = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
        decoded = corn_decode(rew["p_notbad"], rew["p_good"], t_good) if guard_pass else None
        is_q3 = guard_pass and decoded == 3
        p_good = float(rew["p_good"]) if guard_pass else 0.0
        max_p_good = max(max_p_good, p_good)
        max_depth = max(max_depth, int(rew["reached_depth"]))
        oid = f"phg_{seed_idx:04d}_{repeat}_{wid:03d}"
        base = {
            "id": oid, "seed_idx": seed_idx, "repeat": repeat, "walk": int(wid),
            "family": "phoenix", "branch": s.branch, "theta": s.theta, "offset": s.offset,
            "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
            "outcome_fw": rew["outcome_fw"], "k3": rew["reward_k3"], "raw_top3": rew["raw_top3"],
            "reached_depth": int(rew["reached_depth"]), "decoded_class": decoded,
            "p_notbad": float(rew["p_notbad"]) if guard_pass else 0.0, "p_good": p_good,
            "t_good": t_good, "guard_pass": guard_pass,
            "guard_fail": None if guard_pass else "sentinel", **ident,
        }
        outcome_rows.append(base)
        if is_q3:
            n_adm += 1
            emb = embedder.embed(rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"], s,
                                 dwork / "morph")
            # within-descent distinct-look count (order-independent decomposition variable)
            if not within_embs or float(np.max(np.stack(within_embs) @ emb)) < NEAR_DUP_THRESHOLD:
                within_embs.append(emb)
            tile = dwork / "feats" / f"{oid}.jpg"
            feat = phoenix_outcome_feature(scorer, rew["outcome_cx"], rew["outcome_cy"],
                                           rew["outcome_fw"], s, tile)
            admitted.append((dict(base, mix_source="phoenix_grid", canon_pgood=p_good,
                                  scorer_version=SCORER_VERSION), feat, emb))

    active_s = time.time() - t0
    record = {
        "seed_idx": seed_idx, "repeat": repeat, "rng_seed": rng_seed,
        "stratum": stratum_key(stratum_of(s)), "branch": s.branch,
        "abs_p": abs(s.p), "arg_p": math.atan2(s.p.imag, s.p.real), "abs_z_m1": abs(s.z_m1),
        "n_walks": len(by_walk), "walk_errors": walk_errors, "n_admissions": n_adm,
        "distinct_looks_within": len(within_embs), "max_p_good": max_p_good,
        "max_reached_depth": max_depth, "active_seconds": round(active_s, 2),
    }
    return record, outcome_rows, admitted


# =========================================================================== #
# Orchestration (resume-safe, budget-gated)
# =========================================================================== #
def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((int(r["seed_idx"]), int(r["repeat"])))
    return done


def run(args):
    run_name = args.run_name or ("smoke" if args.smoke else "grid")
    run_dir = DISCOVERY_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = SCRATCH_ROOT / run_name
    scratch.mkdir(parents=True, exist_ok=True)

    n_seeds = args.n_seeds if args.n_seeds else (3 if args.smoke else 120)
    k = args.k if args.k else (2 if args.smoke else 4)
    n_walks = args.walks if args.walks else (3 if args.smoke else 8)
    budget_min = args.budget if args.budget is not None else (0 if args.smoke else 240.0)
    t_good = args.t_good

    seeds, plan = stratified_seeds(args.seed, n_seeds)
    print(f"=== phoenix grid ({'SMOKE' if args.smoke else 'RUN'}) run={run_name} ===")
    print(f"grid: {len(seeds)} seeds x {k} descents ({n_walks} walks/descent) "
          f"| depth[{DEPTH_MIN},{DEPTH_MAX}] t_good={t_good} budget={budget_min}min")
    print(f"strata: {plan['n_strata_observed']} observed | branch={plan['branch_counts']} "
          f"| p_band={plan['p_band_counts']} | z={plan['z_class_counts']}")

    # durable paths
    p_seeds = run_dir / "seeds.jsonl"
    p_records = run_dir / "descent_records.jsonl"
    p_all = run_dir / "all_outcomes.jsonl"
    p_ledger = run_dir / "outcome_ledger.jsonl"
    p_feats = run_dir / "outcome_feats.npz"
    p_looks = run_dir / "distinct_looks.npz"

    # seeds.jsonl is rewritten each start (deterministic from --seed); records/ledger append.
    with open(p_seeds, "w", encoding="utf-8") as f:
        for i, s in enumerate(seeds):
            f.write(json.dumps(dict(psamp.seed_to_record(s), seed_idx=i,
                                    stratum=stratum_key(stratum_of(s)))) + "\n")

    done = _load_done(p_records)
    if done:
        print(f"resume: {len(done)} descents already recorded — skipping them")

    # guarded scorer — byte-identical scoring path to production _run_phoenix
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    print(f"scorer: GUARDED CORN ({SCORER_VERSION}); morph-embed distinct-look cos={NEAR_DUP_THRESHOLD}")

    embedder = MorphEmbedder(scratch)
    looks = DistinctLookTally(p_looks, threshold=NEAR_DUP_THRESHOLD)   # run-global, resumable
    feats: dict[str, np.ndarray] = {}
    if p_feats.exists():
        # MUST context-close: a bare `np.load` keeps the .npz OPEN for the process lifetime
        # (NpzFile is lazy), which on Windows locks the file so the first _save_feats os.replace
        # onto it fails with WinError 5. Copy each array out, then close.
        with np.load(p_feats, allow_pickle=False) as z:
            feats = {kk: z[kk].copy() for kk in z.files}

    # Resume resilience: re-embed any admitted ledger row whose 1280-D feature was lost to an
    # interrupted _save_feats (a killed/failed session), so outcome_feats stays consistent with
    # the ledger. A no-op on a clean store.
    if p_ledger.exists() and feats is not None:
        led0 = [json.loads(l) for l in open(p_ledger, encoding="utf-8") if l.strip()]
        missing = [r for r in led0 if r["id"] not in feats]
        if missing:
            print(f"backfill: {len(missing)} admitted rows missing features — re-embedding", flush=True)
            for r in missing:
                sd = psamp.Seed(c=complex(r["phoenix_c_re"], r["phoenix_c_im"]),
                                p=complex(r["phoenix_p_re"], r["phoenix_p_im"]),
                                z_m1=complex(r["phoenix_zm1_re"], r["phoenix_zm1_im"]),
                                branch=r.get("branch", "cardioid"), theta=r.get("theta", 0.0),
                                offset=r.get("offset", 0.0))
                try:
                    feats[r["id"]] = phoenix_outcome_feature(
                        scorer, r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], sd,
                        scratch / "backfill" / f"{r['id']}.jpg")
                except Exception as e:
                    print(f"  backfill {r['id']} failed (skipped): {type(e).__name__}: {str(e)[:100]}",
                          flush=True)
            _save_feats(p_feats, feats)
            shutil.rmtree(scratch / "backfill", ignore_errors=True)

    # feats file handles (append)
    f_records = open(p_records, "a", encoding="utf-8")
    f_all = open(p_all, "a", encoding="utf-8")
    f_ledger = open(p_ledger, "a", encoding="utf-8")

    totals = {"descents": 0, "walks": 0, "admissions": 0, "distinct_global": looks.count("phoenix"),
              "walk_errors": 0}
    durations: list[float] = []
    t0 = time.time()
    stopped = "complete"

    for seed_idx, s in enumerate(seeds):
        for repeat in range(k):
            if (seed_idx, repeat) in done:
                continue
            # budget gate: don't start a descent that can't finish inside the cap
            el_min = (time.time() - t0) / 60.0
            if budget_min:
                est = (np.median(durations) / 60.0) if durations else 0.0
                if el_min + est > budget_min:
                    print(f"  budget gate: elapsed {el_min:.1f}m + est {est:.2f}m > {budget_min}m; "
                          f"stopping cleanly", flush=True)
                    stopped = "budget"
                    break
            rng_seed = args.seed + seed_idx * 1000 + repeat
            dwork = scratch / f"s{seed_idx:04d}_r{repeat}"
            try:
                rec, orows, adm = score_descent(
                    s, seed_idx, repeat, rng_seed, scorer=scorer, embedder=embedder,
                    n_walks=n_walks, scratch=scratch, t_good=t_good)
            except (SystemExit, Exception) as e:
                totals["walk_errors"] += 1
                print(f"  ENGINE FAILURE s{seed_idx} r{repeat}: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                if not args.keep_scratch:
                    shutil.rmtree(dwork, ignore_errors=True)   # never leak a failed descent's scratch
                if isinstance(e, KeyboardInterrupt):
                    stopped = "interrupt"; break
                continue

            # persist walk outcomes (label source) + admissions (intake ledger + global tally)
            for o in orows:
                f_all.write(json.dumps(o) + "\n")
            n_new_distinct = 0
            for row, feat, emb in adm:
                distinct = looks.add("phoenix", emb)             # run-global distinct look
                n_new_distinct += int(distinct)
                row = dict(row, distinct=bool(distinct), dup_of=None)
                f_ledger.write(json.dumps(row) + "\n")
                feats[row["id"]] = np.asarray(feat, np.float32)
            rec["distinct_looks_global_new"] = n_new_distinct
            f_records.write(json.dumps(rec) + "\n")
            for fh in (f_records, f_all, f_ledger):
                fh.flush()
            looks.save()
            _save_feats(p_feats, feats)

            # Purge this descent's render scratch immediately — a grid is hundreds of descents,
            # and each descent's reframe/guard-field intermediates are ~0.4 GB. Purging per
            # descent (not per run like production_seeder) bounds live scratch to one descent.
            if not args.keep_scratch:
                shutil.rmtree(dwork, ignore_errors=True)

            durations.append(rec["active_seconds"])
            totals["descents"] += 1
            totals["walks"] += rec["n_walks"]
            totals["admissions"] += rec["n_admissions"]
            totals["distinct_global"] = looks.count("phoenix")
            totals["walk_errors"] += rec["walk_errors"]
            el_min = (time.time() - t0) / 60.0
            print(f"  s{seed_idx:03d}/{len(seeds)} r{repeat} [{rec['stratum']}]: walks={rec['n_walks']} "
                  f"adm={rec['n_admissions']} distinct_in={rec['distinct_looks_within']} "
                  f"(+{n_new_distinct} global) maxpg={rec['max_p_good']:.3f} d={rec['max_reached_depth']} "
                  f"| {rec['active_seconds']:.0f}s (elapsed {el_min:.1f}m)", flush=True)
        else:
            continue
        break   # inner loop broke (budget/interrupt) -> stop outer too

    for fh in (f_records, f_all, f_ledger):
        fh.close()
    looks.save()
    _save_feats(p_feats, feats)

    session_wall = time.time() - t0
    # CUMULATIVE totals from the durable store (correct across resume sessions — session wall
    # would undercount a resumed run). active-minutes = Σ per-descent active_seconds; the
    # distinct-look count + price come from the ledger's authoritative `distinct` flags.
    all_recs = [json.loads(l) for l in open(p_records, encoding="utf-8") if l.strip()]
    all_led = [json.loads(l) for l in open(p_ledger, encoding="utf-8") if l.strip()]
    cum_active_min = sum(r.get("active_seconds", 0.0) for r in all_recs) / 60.0
    n_distinct = sum(1 for r in all_led if r.get("distinct"))
    cum_totals = {"descents": len(all_recs),
                  "walks": sum(int(r.get("n_walks", 0)) for r in all_recs),
                  "admissions": len(all_led),
                  "seeds_with_descents": len({r["seed_idx"] for r in all_recs}),
                  "session_descents": totals["descents"], "walk_errors": totals["walk_errors"]}
    price = round(cum_active_min / n_distinct, 3) if n_distinct else None
    summary = {
        "run": run_name, "smoke": args.smoke, "stopped": stopped,
        "session_wallclock_s": round(session_wall, 1), "active_minutes": round(cum_active_min, 2),
        "config": {"n_seeds": len(seeds), "k": k, "walks_per_descent": n_walks,
                   "depth": [DEPTH_MIN, DEPTH_MAX], "t_good": t_good, "budget_min": budget_min,
                   "scorer": SCORER_PATH, "scorer_version": SCORER_VERSION,
                   "near_dup_threshold": NEAR_DUP_THRESHOLD, "rng_seed": args.seed},
        "totals": cum_totals,
        "distinct_looks_phoenix": n_distinct,
        "realized_min_per_look_phoenix": price,   # cumulative active-min / distinct look (measure/scheduler prior)
        "stratification_plan": plan,
        "outputs": {"seeds": str(p_seeds), "descent_records": str(p_records),
                    "all_outcomes": str(p_all), "outcome_ledger": str(p_ledger),
                    "outcome_feats": str(p_feats), "distinct_looks": str(p_looks)},
    }
    _atomic_write(run_dir / "summary.json", json.dumps(summary, indent=2))
    print("\n=== GRID SUMMARY (cumulative) ===")
    print(f"  {cum_totals['descents']} descents / {cum_totals['walks']} walks -> "
          f"{cum_totals['admissions']} admissions, {n_distinct} distinct looks "
          f"(cos {NEAR_DUP_THRESHOLD}); {cum_totals['seeds_with_descents']} seeds")
    print(f"  realized min/look (phoenix): {price}  | this session {session_wall/60:.1f}m "
          f"(+{cum_totals['session_descents']} descents), stopped={stopped}")
    print(f"  durable -> {run_dir}")
    if not args.keep_scratch and stopped == "complete":
        _purge(scratch)
    return summary


def _save_feats(path: Path, feats: dict):
    if not feats:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + "_tmp.npz")
    np.savez_compressed(tmp, **feats)
    for attempt in range(6):                       # absorb transient Windows locks (AV/indexer)
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.5)


def _purge(scratch: Path):
    if not scratch.exists():
        return
    try:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"  scratch purged: {scratch}")
    except OSError as e:
        print(f"  scratch purge failed (non-fatal): {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="full grid (N=120 x K=4, 4-hour budget)")
    ap.add_argument("--smoke", action="store_true", help="tiny grid (3x2) — validate the pipeline")
    ap.add_argument("--run-name", default=None, help="durable subdir name (default: grid / smoke)")
    ap.add_argument("--n-seeds", type=int, default=0, help="override seed count")
    ap.add_argument("--k", type=int, default=0, help="descents per seed (>=3; default 4)")
    ap.add_argument("--walks", type=int, default=0, help="walks per descent (default 8)")
    ap.add_argument("--budget", type=float, default=None, help="active-time cap minutes (default 240)")
    ap.add_argument("--t-good", type=float, default=T_GOOD_DEFAULT, help="phoenix q3 operating point")
    ap.add_argument("--seed", type=int, default=0, help="stratified-draw + descent RNG base")
    ap.add_argument("--keep-scratch", action="store_true")
    args = ap.parse_args(argv)
    if not (args.run or args.smoke):
        ap.print_help(); return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
