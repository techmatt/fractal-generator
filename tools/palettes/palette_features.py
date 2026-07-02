"""Palette trajectory features + derived type, for diversity sampling and downstream
parameter dispatch.

Each q3 palette (`data/palettes/score3_colormaps.json`, 33 sRGB stops on t in
[0, 32/33]) is turned into:

  * a **(32, 3) Oklab trajectory** -- the diversity feature, reverse-canonicalized so
    a palette and its reverse map to the same trajectory, and
  * a **derived type** in {cyclic, non_cyclic}, computed from the palette's *literal*
    terminal-stop seam (cyclic iff the OKLab gap between the first and last authored
    stops is < EPS_CYC; the JSON's declared `cycle` field is reference-only). Two
    endpoint conventions for two jobs: the type test reads the literal terminals, while
    the 32-anchor trajectory feature stays **cell-centered** (t = (i+0.5)/n) so
    sequential palettes don't false-wrap in the distance/FPS metric.
    The old three-way split (cyclic/diverging/sequential) collapsed to binary once
    center-pivot -- the only knob the sequential/diverging distinction ever dispatched
    -- was dropped; the diverging *signals* are still computed (see `_compute_signals`)
    for a future center-pivot re-introduction, just not used for dispatch.

The public API downstream samplers build on: `palette_feature`, `derive_type`,
`distance_matrix`, `farthest_point_order`, plus `load_palettes` /
`compute_all_features` for bulk work.

Known limitation (do NOT fix): `palette_distance` is a per-anchor L2, hence
*shift-variant* for cyclic palettes -- a phase-shifted cyclic reads as "moved". That
is acceptable for diversity sampling; a DFT-magnitude variant would make it
shift-invariant if ever needed.
"""

import json
import os
import sys

import numpy as np

# sibling import regardless of how this file is loaded (mirrors tools/ convention)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color  # noqa: E402

# repo root = three levels up from tools/palettes/palette_features.py
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PALETTES = os.path.join(ROOT, "data", "palettes", "score3_colormaps.json")

N_ANCHORS = 32

# --- provisional type-derivation thresholds (tunable by eye from the printed
# --- distributions; see build_features.py's report). Exposed as module constants so
# --- downstream / experiments can override before calling derive_type.
EPS_CYC = 0.05           # literal terminal-stop seam gap (Oklab) below which -> cyclic.
#                          The ONLY threshold derive_type dispatches on in v1.
# The remaining thresholds are diverging-only and RETAINED-FOR-OPTIONALITY: derive_type
# no longer consumes them (binary {cyclic, non_cyclic}), but they still tune the
# diverging signals surfaced in the report / stored per entry, so a future center-pivot
# re-introduction can re-derive diverging membership with no recomputation.
END_CHROMA_MIN = 0.045   # diverging(a): ends must be chromatic at least this much
MID_CHROMA_RATIO = 0.45  # diverging(a): mid chroma < this * end chroma
END_L_MATCH_A = 0.30     # diverging(a): ends must be lightness-comparable (a diverging
#                          palette turns about a center; without this, a monotonic
#                          sequential that dips through gray -- e.g. cividis, blue->gray
#                          ->yellow -- misfires signal (a)).
END_L_MATCH_EPS = 0.20   # diverging(b): |L[0]-L[-1]| below this = "matched ends"
#                          (0.20 catches seismic, blue->white->red with mildly
#                          unequal end lightness; no sequential has a real interior L
#                          extremum, so raising it flips nothing false)
INTERIOR_PROM_MIN = 0.15  # diverging(b): interior L extremum must stick out this far


# ---------------------------------------------------------------- LUT sampling ----

def load_palettes(path=DEFAULT_PALETTES):
    """Load the raw palette list (list of dicts with name/source/stops/cycle/...)."""
    with open(path) as f:
        return json.load(f)


def _lut_sample(stops, ts):
    """Sample a stop list at positions `ts` in [0,1] by clamped linear interpolation
    in sRGB (0-1). `stops` = [[t, [r,g,b] 0-255], ...]. Returns (len(ts), 3) sRGB 0-1.

    Clamped (not wrapped): beyond the last stop we hold the end color. Wrapping would
    seam sequential palettes at the top; clamping keeps the trajectory type-agnostic
    (cyclic palettes still read as cyclic because their authored end stops are ~1/33
    apart on the loop, hence perceptually adjacent)."""
    st = np.array([s[0] for s in stops], dtype=np.float64)
    sc = np.array([s[1] for s in stops], dtype=np.float64) / 255.0  # (M,3) sRGB 0-1
    ts = np.asarray(ts, dtype=np.float64)
    out = np.empty((ts.shape[0], 3), dtype=np.float64)
    for ch in range(3):
        out[:, ch] = np.interp(ts, st, sc[:, ch])  # np.interp clamps outside [st0, stN]
    return out


def _anchor_positions(n=N_ANCHORS):
    """t = (i + 0.5)/n, i in 0..n-1 -- cell-centered sampling of [0,1]."""
    return (np.arange(n) + 0.5) / n


def sample_trajectory_oklab(stops, n=N_ANCHORS):
    """(n, 3) Oklab trajectory, as-authored orientation (no canonicalization)."""
    srgb = _lut_sample(stops, _anchor_positions(n))
    return color.srgb_to_oklab(srgb)


# ---------------------------------------------------------------- features --------

def _chroma(traj):
    """Per-anchor Oklab chroma sqrt(a^2+b^2)."""
    return np.hypot(traj[:, 1], traj[:, 2])


def _literal_seam_gap(stops):
    """OKLab distance between the palette's LITERAL terminal stops (stop[0] <-> stop[-1]),
    whatever the t-encoding. This is the seam the *type test* reads -- the actual authored
    endpoints, not the cell-centered trajectory anchors (which sit 1/(2n) inside each end
    and spuriously separate when color moves fast across the seam, the false-negative that
    misclassified genuine loops as non_cyclic). Orientation-independent, so the trajectory's
    reverse-canonicalization does not affect it."""
    c0 = color.srgb_to_oklab(np.asarray(stops[0][1], dtype=np.float64) / 255.0)
    cN = color.srgb_to_oklab(np.asarray(stops[-1][1], dtype=np.float64) / 255.0)
    return float(np.linalg.norm(c0 - cN))


def _compute_signals(traj):
    """Geometry signals surfaced in the report, all computed from the cell-centered
    trajectory.

    NONE of these feed derive_type -- the type test dispatches on `seam_gap` (the literal
    terminal-stop seam, injected in `palette_feature`). `endpoint_dist` here is the
    cell-centered anchor-endpoint distance, retained as a diagnostic (it is what the old
    type test used, and its gap vs `seam_gap` is the false-negative surface). The diverging
    signals -- `end_L_match`, `interior_L_prominence`, `mid_vs_end_chroma` (plus
    `end_chroma`/`mid_chroma`) -- are retained-for-optionality: computed and stored but NOT
    used for dispatch, so a future center-pivot re-introduction can re-derive diverging
    membership with no recomputation."""
    L = traj[:, 0]
    ch = _chroma(traj)
    n = traj.shape[0]

    endpoint_dist = float(np.linalg.norm(traj[0] - traj[-1]))
    end_L_match = float(abs(L[0] - L[-1]))

    # interior lightness extremum prominence: how far the interior L max/min sticks
    # out beyond the more-extreme endpoint (positive = a real interior bump/dip).
    interior = L[1:-1]
    L_hi_end, L_lo_end = max(L[0], L[-1]), min(L[0], L[-1])
    prom_up = float(interior.max() - L_hi_end)
    prom_down = float(L_lo_end - interior.min())
    interior_L_prominence = max(prom_up, prom_down)

    # chroma dip: mid window vs end windows. Windows scale with n.
    w = max(1, n // 8)                       # end window (~4 for n=32)
    m0, m1 = n // 2 - w, n // 2 + w          # central window (~8 wide for n=32)
    end_chroma = float((ch[:w].mean() + ch[-w:].mean()) / 2.0)
    mid_chroma = float(ch[m0:m1].mean())
    mid_vs_end_chroma = float(mid_chroma / (end_chroma + 1e-9))

    return {
        "endpoint_dist": endpoint_dist,
        "end_L_match": end_L_match,
        "interior_L_prominence": interior_L_prominence,
        "end_chroma": end_chroma,
        "mid_chroma": mid_chroma,
        "mid_vs_end_chroma": mid_vs_end_chroma,
    }


def palette_feature(stops, n=N_ANCHORS):
    """Full feature for one palette's stops.

    Returns {trajectory: (n,3) list, canonical_reversed: bool, signals: {...}}.
    The trajectory is reverse-canonicalized to darker-end-first (mean-L of the first
    half vs second half) so feature(P) == feature(reverse(P))."""
    traj = sample_trajectory_oklab(stops, n)

    half = n // 2
    first_L = traj[:half, 0].mean()
    second_L = traj[half:, 0].mean()
    canonical_reversed = bool(first_L > second_L)  # flip so darker end leads
    if canonical_reversed:
        traj = traj[::-1].copy()

    signals = _compute_signals(traj)
    # literal terminal-stop seam gap -- the type test's endpoint convention (measured on
    # the source stops, NOT the cell-centered anchors). Orientation-independent.
    signals["seam_gap"] = _literal_seam_gap(stops)

    return {
        "trajectory": traj,
        "canonical_reversed": canonical_reversed,
        "signals": signals,
    }


def derive_type(feature):
    """Derived type in {cyclic, non_cyclic}: cyclic iff the LITERAL terminal-stop seam gap
    (`seam_gap`, OKLab distance between the first and last authored stops) < EPS_CYC, else
    non_cyclic.

    The seam is measured at the palette's real endpoints, not the cell-centered trajectory
    anchors -- the anchors sit 1/(2n) inside each end and spuriously separate when color
    moves fast across the seam, which false-classified genuine loops as non_cyclic.

    Binary by design -- the old diverging/sequential split only ever dispatched
    center-pivot, which is dropped, so it no longer earns its keep. The diverging
    signals remain in `feature['signals']` for optional future re-derivation."""
    if feature["signals"]["seam_gap"] < EPS_CYC:
        return "cyclic"
    return "non_cyclic"


def compute_all_features(palettes, n=N_ANCHORS):
    """name -> feature dict (with numpy trajectory kept in-memory)."""
    return {p["name"]: palette_feature(p["stops"], n) for p in palettes}


# ---------------------------------------------------------------- distance / FPS --

def palette_distance(feat_a, feat_b):
    """Mean over anchors of per-anchor Euclidean Oklab distance. Symmetric, 0 on id.
    Accepts feature dicts or raw (n,3) trajectories."""
    ta = feat_a["trajectory"] if isinstance(feat_a, dict) else feat_a
    tb = feat_b["trajectory"] if isinstance(feat_b, dict) else feat_b
    ta, tb = np.asarray(ta), np.asarray(tb)
    return float(np.linalg.norm(ta - tb, axis=1).mean())


def distance_matrix(features_by_name, names):
    """(M, M) symmetric pairwise palette_distance over `names`."""
    trajs = np.stack([np.asarray(features_by_name[nm]["trajectory"]) for nm in names])
    m = len(names)
    D = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        # ||trajs[i]-trajs[j]||2 per anchor, mean over anchors, for all j at once
        d = np.linalg.norm(trajs - trajs[i], axis=2).mean(axis=1)
        D[i] = d
    return D


def farthest_point_order(names, features_by_name=None, k=None, weights=None, dmat=None):
    """Greedy farthest-point sampling order over the distance matrix.

    Seeds with the two most-distant palettes, then iteratively appends the palette
    maximizing its min-distance to the already-chosen set. Returns the first `k`
    names (all, if k is None) in selection order.

    Distance source: normally the OKLab trajectory `distance_matrix` over
    `features_by_name`. Pass `dmat` (an (M, M) symmetric matrix aligned to `names`) to
    spread over a *different* distance instead — e.g. the render-space mean-CIEDE2000
    matrix the param-pool selection feeds — reusing this exact greedy primitive rather
    than reimplementing FPS. When `dmat` is given `features_by_name` may be None.

    `weights` is reserved for a later global-score prior and is currently a no-op."""
    if weights is not None:
        raise NotImplementedError("weights prior not wired yet; pass None")
    names = list(names)
    m = len(names)
    if m == 0:
        return []
    if k is None:
        k = m
    k = min(k, m)

    if dmat is not None:
        D = np.asarray(dmat, dtype=np.float64)
        if D.shape != (m, m):
            raise ValueError(f"dmat shape {D.shape} != ({m}, {m}) for the given names")
    else:
        D = distance_matrix(features_by_name, names)
    i0, i1 = np.unravel_index(int(np.argmax(D)), D.shape)  # two most-distant
    chosen = [int(i0), int(i1)]
    min_d = np.minimum(D[i0], D[i1])
    while len(chosen) < k:
        min_d[chosen] = -np.inf
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
    return [names[i] for i in chosen[:k]]
