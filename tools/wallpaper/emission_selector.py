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
  - <=1 render per DISTINCT FRACTAL (hard). "Distinct" mirrors the SEEDER's own
    per-family dedup identity (`atlas.production_seeder.near_dup`), NOT exact
    location-key equality: sibling descent walks converge on the same fractal at
    slightly different final (cx,cy,fw) (and, for julia, the identical seed `c`),
    each recolored into a different palette -> a different Lab cell -> the old
    exact-key `<=1/location` guard never collided and the selector kept them all as
    separate niches. `same_fractal` (below) is the real identity. See its docstring
    for the per-family rule.
  - palette cap (hard): <= max(2, ceil(0.05 * N)) renders per palette, where N =
    number of reachable (family x color) cells. Palette diversity beats cell fill.
  - gate: a pluggable predicate applied before selection (quality floor), kept
    decoupled from the selection logic.

Selection = greedy joint assignment (pure per-cell argmax fails: a cell's best
candidate may reuse a spent fractal or an over-quota palette). Gate-filter, sort
survivors by fitness desc, walk top-down; a candidate is accepted only when it
fills an *empty* cell AND its fractal is not already emitted AND its palette is
under quota — and a fractal is spent only on acceptance, so a candidate whose cell
is already filled steps aside and keeps its fractal free for an empty cell later.
When a cell's best is blocked by a constraint, the next-best for that cell gets its
turn further down the walk. Output = one elite per occupied cell. Because the
fractal-identity guard walks in fitness order, the single kept render of a
same-fractal cluster is its highest-fitness recolor; the other recolors are
rejected and the cells they alone would have filled empty out — intended, so
color-cell coverage becomes honest instead of counting the same geometry N times.
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
# fractal identity — mirrors the seeder's per-family dedup                     #
# --------------------------------------------------------------------------- #
# The seeder's coverage state is a q3 cloud deduped by `production_seeder.near_dup`
# (plane distance < DEDUP_K * max(fw), a "one point per distinct place" rule), but
# that dedup runs on the c-PLANE and never sees the julia z-plane viewport or the
# phoenix z-plane. So the emission-side identity re-derives it per family:
#
#   c-plane (mandelbrot / multibrot{3,4,5}): the seeder rule VERBATIM — same place
#     iff plane dist(cx,cy) < K*max(fw). Same-center/different-zoom merges, exactly
#     as the seeder intends — and safe here because the pool's c-plane candidates are
#     ALREADY this-rule-deduped upstream (they are q3-cloud members), so a big-fw
#     frame can never swallow a genuinely-distinct deep one at emission: it was never
#     admitted to the cloud in the first place.
#
#   julia* (julia, julia_multibrot{3,4,5}) AND phoenix: z-plane viewports that the
#     seeder NEVER deduped, so their fw spans ~3 decades within one plane and a flat
#     K*max(fw) lets one shallow (big-fw) base view swallow genuinely-distinct deep
#     zooms at different sub-locations (observed on the pilot run: a fw=0.86 julia
#     base view merging fit~2.9 deep zooms 5800x finer). Use a SCALE-AWARE viewport
#     rule instead (`_same_viewport`): same place iff the centers are close at the
#     FINER frame (dist < K*min(fw)) AND the two zooms are comparable
#     (max(fw) <= ZOOM_RATIO*min(fw)). Recolor siblings (~same center, ~same fw, the
#     redundancy we DO want to kill) still merge; a decade-deeper zoom stays distinct.
#     Julia additionally requires a seed-`c` match FIRST: `c` IS the fractal, so two
#     DIFFERENT-c views are distinct even when both sit at the base-scale (0,0) frame
#     (all base-scale julias share viewport (0,0,~3)). Match c, not viewport alone.
#
#   NOTE ON THE SPEC. The fix prompt prescribed a flat 1.5*max(fw) for julia and
#   carved out only phoenix as scale-aware. But julia's z-plane viewport has the
#   identical never-upstream-deduped / decade-spanning structure as phoenix, and on
#   the pilot run the flat rule DROPPED several high-fitness genuinely-distinct julia
#   viewports (see the validation report) — which the same prompt forbids ("do NOT
#   collapse genuinely-distinct locations"). So julia gets the same scale-aware
#   viewport rule as phoenix; the c-plane rule is unchanged and matches the seeder.
DEDUP_K = 1.5                # mirrors production_seeder.DEDUP_K
ZOOM_RATIO = 4.0            # z-plane (julia/phoenix): frames farther apart than this in zoom are distinct places
_C_TOL = 1e-9               # julia seed-c match tolerance (siblings share the exact seed; distinct c differ by >>tol)


def _is_julia(family: str) -> bool:
    return family.startswith("julia")


def _plane_dist(a: "Candidate", b: "Candidate") -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _c_match(a: "Candidate", b: "Candidate") -> bool:
    if a.c_re is None or a.c_im is None or b.c_re is None or b.c_im is None:
        return a.c_re == b.c_re and a.c_im == b.c_im       # both None -> match; one None -> distinct
    return abs(a.c_re - b.c_re) <= _C_TOL and abs(a.c_im - b.c_im) <= _C_TOL


def _same_viewport(a: "Candidate", b: "Candidate", k: float) -> bool:
    """Scale-aware viewport identity for never-upstream-deduped z-planes (julia/phoenix):
    centers close at the FINER frame AND zooms comparable — so a base view can't swallow
    a genuinely-distinct deep zoom of a different sub-location."""
    lo, hi = min(a.fw, b.fw), max(a.fw, b.fw)
    if lo <= 0.0:
        return _plane_dist(a, b) < k * hi
    return _plane_dist(a, b) < k * lo and hi <= ZOOM_RATIO * lo


def same_fractal(a: "Candidate", b: "Candidate", k: float = DEDUP_K) -> bool:
    """Do `a` and `b` render the SAME fractal (up to recolor / sibling-descent jitter)?

    Per-family identity (see the block comment above). Falls back to exact
    location-id equality when either candidate carries no viewport geometry, so
    geometry-free callers keep the historical <=1/location behavior exactly.
    """
    if a.family != b.family:
        return False
    if not (a.has_geometry and b.has_geometry):
        return a.location_id == b.location_id
    if _is_julia(a.family):
        return _c_match(a, b) and _same_viewport(a, b, k)
    if a.family == "phoenix":
        return _same_viewport(a, b, k)
    # c-plane: the seeder rule verbatim (pool candidates already max(fw)-deduped upstream).
    return _plane_dist(a, b) < k * max(a.fw, b.fw)


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
    # --- fractal-identity geometry (for `same_fractal`) --------------------- #
    # The viewport + (julia) seed constant. Optional so pre-existing callers that
    # only pass `location_id` keep working: with no finite geometry, `same_fractal`
    # falls back to exact location-id equality (the historical <=1/location rule),
    # so a no-geometry candidate set selects byte-identically to before.
    cx: float = float("nan")
    cy: float = float("nan")
    fw: float = float("nan")
    c_re: Optional[float] = None    # julia seed c (the fractal identity for julia*)
    c_im: Optional[float] = None

    @property
    def behavior_cell(self) -> tuple[str, int]:
        return (self.family, self.color_cell)

    @property
    def has_geometry(self) -> bool:
        return math.isfinite(self.cx) and math.isfinite(self.cy) and math.isfinite(self.fw)


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
    palette_family_of: Optional[Callable[[Candidate], str]] = None,
    palette_family_cap: Optional[int] = 3,
) -> SelectionResult:
    """Greedy joint MAP-Elites selection. See module docstring for the contract.

    ``palette_family_of`` + ``palette_family_cap`` add the tunable palette-family
    diversity control (Stage-2d portfolio dial): a hard cap of at most
    ``palette_family_cap`` emitted renders per *palette* family (as classified by
    ``palette_family_of``), mirroring the per-palette reuse cap one level up. The
    behavior axis (MAP-Elites cell = fractal-family x color-cell) is unchanged;
    this only prunes the greedy walk so the *same few palette families can't win
    every location narrowly*. The quality floor (``gate``) is applied first, so
    the family cap trades away fitness preference, never quality.

    ``palette_family_cap`` defaults to **3**: a gentle family-diversity guardrail.
    At current production volume (0.90 gate, ~7 picks) it is a *no-op* — it does
    not bind until some family would exceed 3 emitted renders, and today's max
    family count is already <=3. It auto-engages as emission volume scales, then
    trading a small pref-rank cost + minor portfolio shrinkage for family spread.
    Retunable via the kwarg (cap=2 is the knee if more spread is wanted). Note the
    deliberate default change off (``None``) -> 3: the "byte-identical to the
    pre-existing / control-OFF behavior" guarantee now holds only when the cap is
    *explicitly* set to ``None``. The control is OFF iff ``palette_family_of is
    None`` or ``palette_family_cap is None``.
    """
    cands = list(candidates)
    survivors = [c for c in cands if (gate is None or gate(c))]

    reachable = {c.behavior_cell for c in survivors}
    n_reachable = len(reachable)
    cap = max(palette_cap_floor, math.ceil(palette_cap_frac * n_reachable))
    fam_control = palette_family_of is not None and palette_family_cap is not None

    # fitness desc; deterministic tiebreak so runs are reproducible.
    order = sorted(survivors, key=lambda c: (-c.fitness, c.image_id, c.location_id, c.palette_id))

    emitted: list[Candidate] = []                # accepted fractal representatives (identity guard)
    n_dup_rejected = 0                            # candidates blocked as same-fractal near-dups
    pal_ct: Counter[str] = Counter()
    fam_ct: Counter[str] = Counter()
    filled: dict[tuple[str, int], Candidate] = {}
    for c in order:
        cell = c.behavior_cell
        if cell in filled:                       # cell already has a better elite
            continue
        if any(same_fractal(c, e) for e in emitted):   # <=1 render per DISTINCT FRACTAL
            n_dup_rejected += 1                   # (seeder-identity dedup, not exact-key)
            continue
        if pal_ct[c.palette_id] >= cap:          # palette cap
            continue
        if fam_control and fam_ct[palette_family_of(c)] >= palette_family_cap:
            continue                             # palette-family cap (diversity dial)
        filled[cell] = c                         # fractal spent only on acceptance
        emitted.append(c)
        pal_ct[c.palette_id] += 1
        if fam_control:
            fam_ct[palette_family_of(c)] += 1

    picks = list(filled.values())
    report = _build_report(cands, survivors, picks, cap, n_reachable, grid)
    report["n_dup_rejected"] = n_dup_rejected
    report["palette_family_cap"] = palette_family_cap
    if fam_control:
        report["palette_family_spread"] = dict(
            Counter(palette_family_of(c) for c in picks).most_common())
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
