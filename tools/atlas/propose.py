#!/usr/bin/env python
r"""Atlas round-1 proposer — the "bad proposer" the 3-arm acceptance test evaluates
(prompts/atlas-round1-proposer-acceptance-prompt.md).

Emits guided-descend `--seed-list` files (one `{"cx","cy","fw",...}` object per line)
for the two *injected* arms of the acceptance test:

  arm 2  uniform-over-domain  — de-clustering control: FPS-spread over the atlas
                                boundary-band DOMAIN, no value targeting.
  arm 3  atlas acquisition    — value + uncertainty targeting.

The acquisition is a principled tightening of the round-0 dry-run rule:

    a(seed) = conf · theta_norm  +  lambda · (1 - conf)
              \_______________/     \_______________/
               exploit (only where    explore (uncertainty-
               block-CV trusts theta)  driven where theta does
                                       not extrapolate)

conf-gates theta_hat so it drives selection only where support is dense (high conf);
low-conf regions are steered by uncertainty alone (theta-agnostic explore). `lambda`
is FIXED (not swept in the acceptance) — the exploit/explore attribution tells us
which way to move it later. Each seed is tagged exploit vs explore by which term
dominated.

Selection (both arms): draw a dense in-domain candidate cloud, (arm 3) threshold to
the high-acquisition set, then farthest-point-spread to N over the c-plane — seed-space
diversity within the value-selected set. Starting fw is drawn from the SAME empirical
root-fw distribution as the current seeder (arm 1's own native seeds), so the arms
differ ONLY in cx/cy targeting, not scale.

Reuses `tools/atlas/atlas.py` (`Atlas.load().query()`), nothing else. Pure numpy.

  uv run python tools/atlas/propose.py --mode uniform    --arm1-seeds <a1.jsonl> --n 250 --out <a2.jsonl>
  uv run python tools/atlas/propose.py --mode acquisition --arm1-seeds <a1.jsonl> --n 250 --out <a3.jsonl>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from atlas import Atlas  # noqa: E402

DEFAULT_LAMBDA = 0.5
# arm-3 acquisition threshold: keep candidates in the top (1 - this) of acq before FPS.
DEFAULT_ACQ_QUANTILE = 0.60

# --------------------------------------------------------------------------- #
# round-2 constants (prompts/atlas-round2-prescreen-discovery-prompt.md)
# --------------------------------------------------------------------------- #
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
# Efficient round-2 descent config — the pre-screen probe MUST match it so a seed
# that reaches depth 2 in the probe reaches depth 2 in the full descent (identical
# step-1 machinery under the same node/screen params).
SCREEN_NODE_WIDTH = 384
SCREEN_SIGMA_BAND = "8,10,12,14,16"
SCREEN_OCC_FLOOR = 0.321
SCREEN_BLACK_CAP = 0.30
# exploit arm: keep candidates above this acq quantile (over descendable survivors)
# before FPS — the "pure high-conf theta_hat targeting" set.
EXPLOIT_ACQ_QUANTILE = 0.80
# explore arm: keep descendable survivors BELOW this conf quantile (the genuinely
# unsampled / low-conf boundary) before FPS — theta-agnostic discovery.
EXPLORE_CONF_QUANTILE = 0.50
# in-domain candidate cloud size (uniform draws in mask_bounds, kept if in_domain).
N_CLOUD = 40000


# --------------------------------------------------------------------------- #
# root-fw distribution = the current seeder's own (empirical, from arm-1 seeds)
# --------------------------------------------------------------------------- #
def load_fw_pool(arm1_seeds: Path) -> np.ndarray:
    """The empirical depth-1 fw distribution of the current seeder (arm 1's native
    seeds). Sampling arm-2/3 fw from THIS makes scale identical in distribution across
    arms, isolating cx/cy targeting as the only difference."""
    fws = []
    for line in open(arm1_seeds, encoding="utf-8"):
        line = line.strip()
        if line:
            fws.append(float(json.loads(line)["fw"]))
    if not fws:
        raise SystemExit(f"no fw values in {arm1_seeds}")
    return np.asarray(fws, float)


# --------------------------------------------------------------------------- #
# in-domain candidate cloud + FPS
# --------------------------------------------------------------------------- #
def domain_cloud(atlas: Atlas, n_cloud: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform draws over mask_bounds, kept where in_domain. Returns (M,2) cx,cy."""
    x0, x1, y0, y1 = atlas.mask_bounds
    keep = []
    # oversample in blocks until we have enough (domain is ~25% of the bbox).
    while sum(len(k) for k in keep) < n_cloud:
        cx = rng.uniform(x0, x1, n_cloud)
        cy = rng.uniform(y0, y1, n_cloud)
        m = atlas.in_domain(cx, cy)
        keep.append(np.stack([cx[m], cy[m]], axis=1))
    return np.concatenate(keep, axis=0)[:n_cloud]


def fps(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sampling of n indices over `points` (M,2). Deterministic start
    (seeded), greedy max-min. O(n·M)."""
    m = len(points)
    n = min(n, m)
    chosen = [int(rng.integers(m))]
    d2 = ((points - points[chosen[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(np.argmax(d2))
        chosen.append(i)
        d2 = np.minimum(d2, ((points - points[i]) ** 2).sum(1))
    return np.asarray(chosen)


# --------------------------------------------------------------------------- #
# proposer
# --------------------------------------------------------------------------- #
def propose(mode: str, atlas: Atlas, fw_pool: np.ndarray, n: int, lam: float,
            acq_quantile: float, seed: int):
    rng = np.random.default_rng(seed)
    cloud = domain_cloud(atlas, N_CLOUD, rng)
    theta, conf, _ = atlas.query(cloud[:, 0], cloud[:, 1])

    # theta normalized on the SAME scale the round-0 dry-run used (training reward
    # min/max), so conf·theta_norm is comparable to lambda·(1-conf).
    rmin, rmax = float(atlas.reward.min()), float(atlas.reward.max())
    theta_norm = np.clip((theta - rmin) / (rmax - rmin + 1e-9), 0.0, 1.0)
    exploit = conf * theta_norm
    explore = lam * (1.0 - conf)
    acq = exploit + explore

    if mode == "uniform":
        # de-clustering control: FPS over the whole domain, value-agnostic.
        sel = fps(cloud, n, rng)
    elif mode == "acquisition":
        thr = float(np.quantile(acq, acq_quantile))
        hi = np.where(acq >= thr)[0]
        if len(hi) < n:
            hi = np.argsort(acq)[::-1][: max(n, len(hi))]
        sub = fps(cloud[hi], n, rng)
        sel = hi[sub]
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    # starting fw ~ empirical current-seeder distribution (resample with replacement).
    fw = fw_pool[rng.integers(len(fw_pool), size=len(sel))]

    rows = []
    for j, i in enumerate(sel):
        tag = "exploit" if exploit[i] >= explore[i] else "explore"
        rows.append({
            "cx": float(cloud[i, 0]), "cy": float(cloud[i, 1]), "fw": float(fw[j]),
            "arm": mode, "tag": tag,
            "theta": float(theta[i]), "theta_norm": float(theta_norm[i]),
            "conf": float(conf[i]), "exploit_term": float(exploit[i]),
            "explore_term": float(explore[i]), "acq": float(acq[i]),
        })
    return rows, dict(rmin=rmin, rmax=rmax, cloud=len(cloud),
                      acq_thr=(float(np.quantile(acq, acq_quantile)) if mode == "acquisition" else None))


# =========================================================================== #
# round-2: descendability pre-screen (Build 1) + discovery-decomposed arms (Build 2)
# =========================================================================== #
def write_seed_list(path: Path, cx, cy, fw):
    """Write a guided-descend / screen-seeds seed-list JSONL (one cx/cy/fw per row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b, c in zip(cx, cy, fw):
            f.write(json.dumps({"cx": float(a), "cy": float(b), "fw": float(c)}) + "\n")


def prescreen(cloud: np.ndarray, fw: np.ndarray, workdir: Path,
              node_width: int, occ_floor: float, black_cap: float, seed: int) -> dict:
    """Descendability pre-screen = guided-descend's OWN step-1, run for real.

    A seed's descendability is NOT a property of its (wide) root frame — the occupancy
    floor over-fires there, which is exactly why guided-descend SKIPS it at the d1->d2
    step. Descendability is whether the depth-2 best-of-N step finds a surviving child.
    So the faithful "would this seed survive step-1" screen is to inject the whole
    candidate cloud as `--seed-list` and run a **depth-2 descent probe** (identical
    node/sigma/screen config to the full run, `--per-walk-rng`): walk w reaches depth 2
    iff its seed is descendable. This reuses the descent machinery verbatim (zero parity
    risk) and directly targets the productivity metric (reached_depth >= 2).

    Returns per-candidate `pass` (reached >= 2) in cloud-row order + the death-cause
    tally (the undescendable-frontier diagnostic)."""
    workdir.mkdir(parents=True, exist_ok=True)
    seed_in = workdir / "cloud_seeds.jsonl"
    write_seed_list(seed_in, cloud[:, 0], cloud[:, 1], fw)
    pool = workdir / "probe_pool"
    cmd = [
        str(BIN), "guided-descend",
        "--seed-list", str(seed_in), "--per-walk-rng", "--seed", str(seed),
        "--depth-min", "2", "--depth-max", "2",
        "--node-width", str(node_width), "--sigma-band", SCREEN_SIGMA_BAND,
        "--descent-occ-floor", str(occ_floor), "--descent-black-cap", str(black_cap),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"guided-descend probe failed:\n{r.stderr[-2000:]}")
    walks = {}
    for line in open(pool / "walks.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            w = json.loads(line)
            walks[int(w["walk"])] = w
    if len(walks) != len(cloud):
        raise SystemExit(f"probe walks {len(walks)} != cloud {len(cloud)} (row-order join broken)")
    reached = np.array([walks[i]["reached_depth"] for i in range(len(cloud))])
    passed = reached >= 2
    causes = {}
    for i in range(len(cloud)):
        c = walks[i]["cause"]
        causes[c] = causes.get(c, 0) + 1
    return {"pass": passed, "reached": reached, "causes": causes,
            "probe_stderr_tail": r.stderr.strip().splitlines()[-1:]}


def _seed_rows(cloud, fw, theta, conf, theta_norm, sel, arm, term):
    """Build emitted seed rows for a selected subset `sel` of the (screened) cloud.
    `term` labels which term drove selection (exploit/explore) — the round-2 arms are
    single-term, so `tag` == arm, but we keep the value columns for attribution."""
    rows = []
    for i in sel:
        rows.append({
            "cx": float(cloud[i, 0]), "cy": float(cloud[i, 1]), "fw": float(fw[i]),
            "arm": arm, "tag": arm, "term": term,
            "theta": float(theta[i]), "theta_norm": float(theta_norm[i]),
            "conf": float(conf[i]),
        })
    return rows


def propose_round2(atlas: Atlas, fw_pool: np.ndarray, n: int, cloud_size: int,
                   node_width: int, occ_floor: float, black_cap: float,
                   exploit_acq_q: float, explore_conf_q: float,
                   workdir: Path, seed: int):
    """Draw ONE in-domain candidate cloud, assign empirical fw, pre-screen
    descendability once, then carve the two discovery-decomposed arms from the shared
    descendable survivor pool:

      exploit — pure high-conf theta_hat targeting: acq = conf * theta_norm, threshold
                to the top (1 - exploit_acq_q), FPS to n. The re-mining reference.
      explore — pure low-conf / uncertainty targeting (the DISCOVERY arm): restrict to
                survivors below the explore_conf_q conf quantile (the genuinely
                unsampled boundary), FPS to n. theta-agnostic by construction.

    Both arms draw from the SAME screened cloud so they differ ONLY in selection."""
    rng = np.random.default_rng(seed)
    cloud = domain_cloud(atlas, cloud_size, rng)
    # Assign fw BEFORE screening — the band/occupancy of a seed frame depend on its fw,
    # so a candidate must be screened at the scale it will descend at.
    fw = fw_pool[rng.integers(len(fw_pool), size=len(cloud))]

    theta, conf, _ = atlas.query(cloud[:, 0], cloud[:, 1])
    rmin, rmax = float(atlas.reward.min()), float(atlas.reward.max())
    theta_norm = np.clip((theta - rmin) / (rmax - rmin + 1e-9), 0.0, 1.0)

    scr = prescreen(cloud, fw, workdir, node_width, occ_floor, black_cap, seed)
    surv = scr["pass"]
    n_surv = int(surv.sum())
    print(f"  pre-screen (depth-2 probe): {n_surv}/{len(cloud)} descendable "
          f"({100*n_surv/len(cloud):.1f}%)  causes={scr['causes']}")
    if n_surv < n:
        print(f"  WARNING: only {n_surv} survivors < requested n={n}; grow --cloud")

    surv_idx = np.where(surv)[0]

    # --- exploit: high-conf value targeting over survivors ---
    acq = conf * theta_norm
    acq_s = acq[surv_idx]
    thr = float(np.quantile(acq_s, exploit_acq_q)) if len(acq_s) else 0.0
    hi = surv_idx[acq_s >= thr]
    if len(hi) < n:  # fall back to top-n by acq among survivors
        hi = surv_idx[np.argsort(acq_s)[::-1][:max(n, len(hi))]]
    sub = fps(cloud[hi], n, rng)
    exploit_sel = hi[sub]
    exploit_rows = _seed_rows(cloud, fw, theta, conf, theta_norm, exploit_sel, "exploit", "exploit")

    # --- explore: low-conf / uncertainty targeting over survivors ---
    conf_s = conf[surv_idx]
    cthr = float(np.quantile(conf_s, explore_conf_q)) if len(conf_s) else 1.0
    lo = surv_idx[conf_s <= cthr]
    if len(lo) < n:  # relax the conf cut if the low-conf survivor pool is thin
        lo = surv_idx[np.argsort(conf_s)[:max(n, len(lo))]]
    sub2 = fps(cloud[lo], n, rng)
    explore_sel = lo[sub2]
    explore_rows = _seed_rows(cloud, fw, theta, conf, theta_norm, explore_sel, "explore", "explore")

    # Descendability of the low-conf (explore-eligible) vs high-conf boundary — the
    # "is the uncovered frontier descendable?" diagnostic. Split the cloud at the
    # global conf median and report pass rate each side.
    conf_med = float(np.median(conf))
    lo_mask = conf <= conf_med
    frontier = {
        "conf_median": conf_med,
        "lowconf_pass_rate": float(surv[lo_mask].mean()) if lo_mask.any() else 0.0,
        "highconf_pass_rate": float(surv[~lo_mask].mean()) if (~lo_mask).any() else 0.0,
        "lowconf_n": int(lo_mask.sum()), "highconf_n": int((~lo_mask).sum()),
    }
    meta = {
        "cloud": len(cloud), "survivors": n_surv,
        "pass_rate": n_surv / len(cloud),
        "probe_causes": scr["causes"],
        "frontier_descendability": frontier,
        "exploit_acq_thr": thr, "explore_conf_thr": cthr,
        "exploit_hi_pool": int(len(hi)), "explore_lo_pool": int(len(lo)),
        "rmin": rmin, "rmax": rmax,
    }
    return exploit_rows, explore_rows, meta


def run_round2(args):
    atlas = Atlas.load()
    fw_pool = load_fw_pool(args.arm1_seeds)
    workdir = Path(args.workdir)
    exploit_rows, explore_rows, meta = propose_round2(
        atlas, fw_pool, args.n, args.cloud, args.node_width, args.occ_floor,
        args.black_cap, args.exploit_acq_quantile, args.explore_conf_quantile,
        workdir, args.seed,
    )
    for rows, out in ((exploit_rows, args.out_exploit), (explore_rows, args.out_explore)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _summ(rows, name):
        cx = np.array([r["cx"] for r in rows]); cy = np.array([r["cy"] for r in rows])
        th = np.array([r["theta"] for r in rows]); cf = np.array([r["conf"] for r in rows])
        print(f"  {name}: {len(rows)} seeds  cx_std {cx.std():.3f} cy_std {cy.std():.3f}  "
              f"theta mean {th.mean():.3f}  conf mean {cf.mean():.3f}")

    fr = meta["frontier_descendability"]
    print("=== propose round-2 (dual arm) ===")
    print(f"  fw pool: {len(fw_pool)} arm-1 fws  range[{fw_pool.min():.4f},{fw_pool.max():.4f}] "
          f"median {np.median(fw_pool):.4f}")
    print(f"  pre-screen pass rate {100*meta['pass_rate']:.1f}%  "
          f"({meta['survivors']}/{meta['cloud']} descendable, causes {meta['probe_causes']})")
    print(f"  frontier descendability: low-conf {100*fr['lowconf_pass_rate']:.1f}% "
          f"(n={fr['lowconf_n']})  vs  high-conf {100*fr['highconf_pass_rate']:.1f}% "
          f"(n={fr['highconf_n']})")
    _summ(exploit_rows, "exploit -> " + str(args.out_exploit))
    _summ(explore_rows, "explore -> " + str(args.out_explore))
    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_out).write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {args.meta_out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["uniform", "acquisition", "round2"])
    ap.add_argument("--arm1-seeds", required=True, type=Path,
                    help="arm-1 native seed jsonl (empirical fw distribution source)")
    ap.add_argument("--n", type=int, default=250, help="seeds to emit")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="explore weight (FIXED)")
    ap.add_argument("--acq-quantile", type=float, default=DEFAULT_ACQ_QUANTILE,
                    help="(acquisition) keep candidates above this acq quantile before FPS")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    ap.add_argument("--out", type=Path, help="(uniform/acquisition) single output jsonl")
    # --- round2 (dual-arm, pre-screened) ---
    ap.add_argument("--out-exploit", type=Path, help="(round2) exploit arm seed jsonl")
    ap.add_argument("--out-explore", type=Path, help="(round2) explore arm seed jsonl")
    ap.add_argument("--meta-out", type=Path, help="(round2) pre-screen meta json")
    ap.add_argument("--workdir", type=Path, default=ROOT / "out" / "atlas" / "round2" / "prescreen",
                    help="(round2) scratch dir for the screen seed-list + table")
    ap.add_argument("--cloud", type=int, default=8000, help="(round2) in-domain candidate cloud size")
    ap.add_argument("--node-width", type=int, default=SCREEN_NODE_WIDTH,
                    help="(round2) pre-screen node render width")
    ap.add_argument("--black-cap", type=float, default=SCREEN_BLACK_CAP)
    ap.add_argument("--occ-floor", type=float, default=SCREEN_OCC_FLOOR)
    ap.add_argument("--exploit-acq-quantile", type=float, default=EXPLOIT_ACQ_QUANTILE)
    ap.add_argument("--explore-conf-quantile", type=float, default=EXPLORE_CONF_QUANTILE)
    args = ap.parse_args()

    if args.mode == "round2":
        for req in ("out_exploit", "out_explore", "meta_out"):
            if getattr(args, req) is None:
                ap.error(f"--mode round2 requires --{req.replace('_', '-')}")
        run_round2(args)
        return
    if args.out is None:
        ap.error("--out is required for --mode uniform/acquisition")

    atlas = Atlas.load()
    fw_pool = load_fw_pool(args.arm1_seeds)
    rows, meta = propose(args.mode, atlas, fw_pool, args.n, args.lam,
                         args.acq_quantile, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    tags = [r["tag"] for r in rows]
    n_exploit = tags.count("exploit")
    cx = np.array([r["cx"] for r in rows]); cy = np.array([r["cy"] for r in rows])
    th = np.array([r["theta"] for r in rows]); cf = np.array([r["conf"] for r in rows])
    print(f"=== propose {args.mode}: {len(rows)} seeds -> {args.out} ===")
    print(f"  fw pool: {len(fw_pool)} arm-1 fws  range[{fw_pool.min():.4f},{fw_pool.max():.4f}] "
          f"median {np.median(fw_pool):.4f}")
    print(f"  cx std {cx.std():.3f}  cy std {cy.std():.3f}  (spread over domain)")
    print(f"  theta: mean {th.mean():.3f} range[{th.min():.3f},{th.max():.3f}]  "
          f"conf: mean {cf.mean():.3f}")
    if args.mode == "acquisition":
        print(f"  acq threshold (q{args.acq_quantile}): {meta['acq_thr']:.3f}  "
              f"tags: exploit {n_exploit}, explore {len(rows)-n_exploit}  (lambda={args.lam})")


if __name__ == "__main__":
    main()
