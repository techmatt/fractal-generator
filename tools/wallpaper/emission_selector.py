"""Emission diversity selector — Stage 2d core.

Pure selection logic: pick a diverse *emission set* from a pool of candidate
renders. No rendering, no head training, no full-res emit. Fitness is just an
input column (the wallpaper-head continuous readout), so swapping v1 -> v2 is a
data change, not a code change.

Behavior space = MAP-Elites cells over ``family x color_cell``:
  - ``color_cell`` bins a candidate's dominant CIELAB color on the a/b plane x a
    coarse L axis (default 3x3 a/b x 2 L = 18 cells), a/b clamped so cells aren't
    wasted on the empty chroma extremes.
  - dominant color = median pixel in Lab (swappable to k-means top cluster).

Constraints:
  - <=1 render per location (hard).
  - palette cap (hard): <= max(2, ceil(0.05 * N)) renders per palette, where N =
    number of reachable (family x color) cells. Palette diversity beats cell fill.
  - gate: a pluggable predicate applied before selection (quality floor), kept
    decoupled from the selection logic.

Selection = greedy joint assignment (pure per-cell argmax fails: a cell's best
candidate may reuse a spent location or an over-quota palette). Gate-filter, sort
survivors by fitness desc, walk top-down; a candidate is accepted only when it
fills an *empty* cell AND its location is unused AND its palette is under quota —
and a location is spent only on acceptance, so a candidate whose cell is already
filled steps aside and keeps its location free for an empty cell later. When a
cell's best is blocked by a constraint, the next-best for that cell gets its turn
further down the walk. Output = one elite per occupied cell.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# color: sRGB8 -> CIELAB (D65)                                                 #
# --------------------------------------------------------------------------- #

_SRGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(...,3) sRGB (uint8 [0,255] or float [0,1]) -> (...,3) CIELAB (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.size and rgb.max() > 1.0:
        rgb = rgb / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _SRGB2XYZ.T) / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def dominant_lab(rgb_img: np.ndarray, method: str = "median", k: int = 4,
                 seed: int = 0) -> np.ndarray:
    """Dominant CIELAB color of an (H,W,3) sRGB image.

    ``median`` (default): component-wise median over all pixels in Lab — robust,
    no fit. ``kmeans``: largest-mass cluster centroid over Lab pixels.
    """
    lab = srgb_to_lab(np.asarray(rgb_img).reshape(-1, 3))
    if method == "median":
        return np.median(lab, axis=0)
    if method == "kmeans":
        return _kmeans_top_cluster(lab, k=k, seed=seed)
    raise ValueError(f"unknown dominant-color method: {method!r}")


def _kmeans_top_cluster(pts: np.ndarray, k: int, seed: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n <= k:
        return np.median(pts, axis=0)
    cen = pts[rng.choice(n, size=k, replace=False)].astype(np.float64)
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                cen[j] = pts[m].mean(0)
    counts = np.bincount(assign, minlength=k)
    return cen[counts.argmax()]


# --------------------------------------------------------------------------- #
# behavior grid                                                               #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ColorGrid:
    """Bins a dominant Lab color into one of ``n_cells`` color cells."""
    a_bins: int = 3
    b_bins: int = 3
    l_bins: int = 2
    ab_clamp: tuple[float, float] = (-60.0, 60.0)
    l_range: tuple[float, float] = (0.0, 100.0)

    @property
    def n_cells(self) -> int:
        return self.a_bins * self.b_bins * self.l_bins

    @staticmethod
    def _bin(x: float, lo: float, hi: float, nbins: int) -> int:
        if hi <= lo or nbins <= 1:
            return 0
        t = (x - lo) / (hi - lo)
        return int(min(nbins - 1, max(0, math.floor(t * nbins))))

    def cell(self, lab: Sequence[float]) -> int:
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        ai = self._bin(a, self.ab_clamp[0], self.ab_clamp[1], self.a_bins)
        bi = self._bin(b, self.ab_clamp[0], self.ab_clamp[1], self.b_bins)
        li = self._bin(L, self.l_range[0], self.l_range[1], self.l_bins)
        return (li * self.b_bins + bi) * self.a_bins + ai


# --------------------------------------------------------------------------- #
# candidates + selection                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    location_id: str
    palette_id: str
    family: str
    fitness: float            # head continuous readout; elite score within a cell
    color_cell: int
    image_id: str = ""
    meta: dict = field(default_factory=dict)   # carried through for reporting only

    @property
    def behavior_cell(self) -> tuple[str, int]:
        return (self.family, self.color_cell)


@dataclass
class SelectionResult:
    picks: list[Candidate]
    palette_cap: int
    n_reachable_cells: int
    report: dict


def select(
    candidates: Iterable[Candidate],
    gate: Optional[Callable[[Candidate], bool]] = None,
    *,
    grid: Optional[ColorGrid] = None,
    palette_cap_frac: float = 0.05,
    palette_cap_floor: int = 2,
) -> SelectionResult:
    """Greedy joint MAP-Elites selection. See module docstring for the contract."""
    cands = list(candidates)
    survivors = [c for c in cands if (gate is None or gate(c))]

    reachable = {c.behavior_cell for c in survivors}
    n_reachable = len(reachable)
    cap = max(palette_cap_floor, math.ceil(palette_cap_frac * n_reachable))

    # fitness desc; deterministic tiebreak so runs are reproducible.
    order = sorted(survivors, key=lambda c: (-c.fitness, c.image_id, c.location_id, c.palette_id))

    used_loc: set[str] = set()
    pal_ct: Counter[str] = Counter()
    filled: dict[tuple[str, int], Candidate] = {}
    for c in order:
        cell = c.behavior_cell
        if cell in filled:                       # cell already has a better elite
            continue
        if c.location_id in used_loc:            # <=1 render per location
            continue
        if pal_ct[c.palette_id] >= cap:          # palette cap
            continue
        filled[cell] = c                         # location spent only on acceptance
        used_loc.add(c.location_id)
        pal_ct[c.palette_id] += 1

    picks = list(filled.values())
    report = _build_report(cands, survivors, picks, cap, n_reachable, grid)
    return SelectionResult(picks=picks, palette_cap=cap, n_reachable_cells=n_reachable, report=report)


def _fitness_dist(vals: Sequence[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {"n": len(a), "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "mean": float(a.mean()), "p75": float(q[3]), "max": float(q[4])}


def _build_report(cands, survivors, picks, cap, n_reachable, grid) -> dict:
    families = sorted({c.family for c in cands})
    grid_total = (grid.n_cells * len(families)) if grid is not None else None
    per_family = Counter(c.family for c in picks)
    palette_hist = Counter(c.palette_id for c in picks)
    return {
        "n_candidates": len(cands),
        "n_survivors": len(survivors),
        "n_picks": len(picks),
        "cells_filled": len(picks),
        "cells_reachable": n_reachable,
        "grid_cells_total": grid_total,
        "coverage_of_reachable": (len(picks) / n_reachable) if n_reachable else 0.0,
        "families": families,
        "per_family_spread": dict(sorted(per_family.items())),
        "palette_cap": cap,
        "n_distinct_palettes_picked": len(palette_hist),
        "palette_hist": dict(palette_hist.most_common()),
        "fitness_dist_picks": _fitness_dist([c.fitness for c in picks]),
        "fitness_dist_survivors": _fitness_dist([c.fitness for c in survivors]),
    }
