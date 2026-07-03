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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["uniform", "acquisition"])
    ap.add_argument("--arm1-seeds", required=True, type=Path,
                    help="arm-1 native seed jsonl (empirical fw distribution source)")
    ap.add_argument("--n", type=int, default=250, help="seeds to emit")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="explore weight (FIXED)")
    ap.add_argument("--acq-quantile", type=float, default=DEFAULT_ACQ_QUANTILE,
                    help="(acquisition) keep candidates above this acq quantile before FPS")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

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
