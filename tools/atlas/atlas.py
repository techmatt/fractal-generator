#!/usr/bin/env python
"""Queryable Mandelbrot seed atlas -- the durable object round 0 fits and the
round-1+ proposer consumes (prompts/atlas-round0-prompt.md).

theta_hat(seed) is an empirical value map: expected best-over-walk reward (v5 k3)
for a guided-descend seeded at (cx, cy), fit by kNN regression over the 600 step-0
walk seeds. conf(seed) is the local data-density (how much training support backs
that theta_hat). Both are restricted to the boundary-band domain (a raster mask);
off-band queries return in_domain=False.

Pure numpy -- no sklearn/torch at query time. Reload with `Atlas.load()` and call
`atlas.query(cx, cy)` (scalar or vectorized). The .npz + this module ARE the atlas
artifact; build_atlas.py writes the .npz, this file reads it.

  from atlas import Atlas
  a = Atlas.load()                      # data/atlas/atlas_v1.npz
  theta, conf, in_dom = a.query(-0.74, 0.13)
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACT_PATH = ROOT / "data" / "atlas" / "atlas_v1.npz"


def _pairwise_knn(query_xy: np.ndarray, train_xy: np.ndarray, k: int):
    """Return (idx[Q,k], dist[Q,k]) of the k nearest train points per query row.
    Brute force (600 train pts -> trivial); no sklearn dependency at reload."""
    # (Q,N) squared distances
    q2 = (query_xy ** 2).sum(1)[:, None]
    t2 = (train_xy ** 2).sum(1)[None, :]
    d2 = q2 + t2 - 2.0 * query_xy @ train_xy.T
    np.maximum(d2, 0.0, out=d2)
    k = min(k, train_xy.shape[0])
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    dk = np.take_along_axis(d2, idx, axis=1)
    order = np.argsort(dk, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dk = np.take_along_axis(dk, order, axis=1)
    return idx, np.sqrt(dk)


@dataclass
class Atlas:
    # training seeds + target
    seed_xy: np.ndarray          # (N,2) cx,cy
    reward: np.ndarray           # (N,) reward_k3
    # estimator config
    k: int                       # kNN neighbor count (chosen by CV)
    weighted: bool               # distance-weighted mean
    r_ref: float                 # median LOO k-th-neighbor distance (conf normalizer)
    # boundary-band domain mask (raster)
    mask: np.ndarray             # (NY,NX) bool, cy-ascending rows
    mask_bounds: tuple           # (x0,x1,y0,y1)
    # bookkeeping
    meta: dict

    # ----- domain ----------------------------------------------------------- #
    def in_domain(self, cx, cy):
        cx = np.asarray(cx, float); cy = np.asarray(cy, float)
        x0, x1, y0, y1 = self.mask_bounds
        ny, nx = self.mask.shape
        ix = np.clip(((cx - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
        iy = np.clip(((cy - y0) / (y1 - y0) * ny).astype(int), 0, ny - 1)
        return self.mask[iy, ix]

    # ----- value + confidence ---------------------------------------------- #
    def query(self, cx, cy):
        """Return (theta_hat, conf, in_domain). Accepts scalars or arrays; theta
        outside the boundary band is still computed (nearest-seed extrapolation)
        but in_domain flags it so the caller can veto."""
        scalar = np.isscalar(cx)
        q = np.stack([np.atleast_1d(np.asarray(cx, float)),
                      np.atleast_1d(np.asarray(cy, float))], axis=1)
        idx, dist = _pairwise_knn(q, self.seed_xy, self.k)
        vals = self.reward[idx]                       # (Q,k)
        if self.weighted:
            w = 1.0 / (dist + 1e-9)
            theta = (w * vals).sum(1) / w.sum(1)
        else:
            theta = vals.mean(1)
        r_k = dist[:, -1]                             # distance to k-th neighbor
        conf = np.clip(self.r_ref / (r_k + 1e-12), 0.0, 1.0)
        dom = self.in_domain(q[:, 0], q[:, 1])
        if scalar:
            return float(theta[0]), float(conf[0]), bool(dom[0])
        return theta, conf, dom

    # ----- io --------------------------------------------------------------- #
    def save(self, path: Path = ARTIFACT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seed_xy=self.seed_xy, reward=self.reward,
            k=self.k, weighted=self.weighted, r_ref=self.r_ref,
            mask=self.mask, mask_bounds=np.array(self.mask_bounds, float),
            meta_json=np.array(_dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "Atlas":
        z = np.load(path, allow_pickle=False)
        return cls(
            seed_xy=z["seed_xy"], reward=z["reward"],
            k=int(z["k"]), weighted=bool(z["weighted"]), r_ref=float(z["r_ref"]),
            mask=z["mask"], mask_bounds=tuple(z["mask_bounds"].tolist()),
            meta=_loads(str(z["meta_json"])),
        )


def _dumps(d: dict) -> str:
    import json
    return json.dumps(d)


def _loads(s: str) -> dict:
    import json
    return json.loads(s)
