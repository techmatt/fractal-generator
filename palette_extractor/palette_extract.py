"""Extract a single, well-defined, closed color palette from a fractal-like image.

See PALETTE_EXTRACTION.md for the design rationale. Pipeline in one breath:
  pixels -> OKLab -> density voxels -> dominant ridge -> kNN graph
  -> MST + tree-diameter (the principal curve) -> trim sparse tips
  -> detect closure (native loop) or mirror-close (fallback) -> uniform-arc resample.

Dependencies: numpy, scipy, pillow  (matplotlib only for --diagnostics).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Color conversion (Ottosson OKLab). ΔE in OKLab is plain Euclidean distance.
# --------------------------------------------------------------------------- #

def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (...,3) in [0,255] -> OKLab (...,3)."""
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], axis=-1)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (...,3) -> sRGB (...,3) in [0,255] uint8."""
    lab = np.asarray(lab, float)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m_ = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s_ = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    lin = np.stack([r, g, b], axis=-1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Failure gate (Phase 2 — eye-set cutoffs over the wallpaper harvest)
# --------------------------------------------------------------------------- #
# Cutoffs were set by Matt against the worst-N strips of the 746-image harvest
# (see prompts/wallpaper_harvest_prompt.md, tools/viz/harvest_gate.html):
#   low_range  = palette OKLab extent (gyration / color range) below EXTENT_FLOOR
#   degenerate = palette curve arc-length below ARCLEN_FLOOR
# A palette is rejected if EITHER fires (the union catches every near-constant
# map). Coverage is intentionally NOT a quality signal: on real photos image-
# coverage is legitimately low (a 1-D rope through a 2-D color sheet), so a
# coverage floor would reject good extractions. `low_coverage` is kept in the
# vocabulary but not wired.
EXTENT_FLOOR = 0.05
ARCLEN_FLOOR = 0.30
QUALITY_REASONS = ("low_coverage", "low_range", "degenerate")  # vocabulary


def gate_quality(extent: float, arclen: float) -> list[str]:
    """Return the quality_flags list for a palette's (extent, arclen).
    Empty list == passes the gate. Single source of truth for the harvest
    classifier, the wired PaletteResult, and the Phase-4 survivor split."""
    flags = []
    if extent < EXTENT_FLOOR:
        flags.append("low_range")
    if arclen < ARCLEN_FLOOR:
        flags.append("degenerate")
    return flags


class PaletteRejected(ValueError):
    """Raised by extract_palette(..., reject=True) when the gate flags a palette."""
    def __init__(self, flags: list[str], extent: float, arclen: float):
        self.flags, self.extent, self.arclen = flags, extent, arclen
        super().__init__(f"palette rejected {flags} (extent={extent:.4f} arclen={arclen:.4f})")


def _loop_arclen(stops_lab: np.ndarray) -> float:
    """Closed-loop OKLab arc length of a stop sequence (palette curve length)."""
    closed = np.vstack([stops_lab, stops_lab[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class PaletteResult:
    stops_rgb: np.ndarray            # (n_stops, 3) uint8, ordered around the closed loop
    stops_lab: np.ndarray            # (n_stops, 3) float OKLab
    closure: str                     # "native" or "mirrored"
    coverage: float                  # fraction of pixels within eps of the palette
    max_step: float                  # largest consecutive ΔE around the (closed) palette
    mean_step: float
    endpoint_gap: float              # OKLab distance between raw spine ends (pre-closure)
    n_ridge: int                     # ridge voxels feeding the graph
    raw_spine_max_step: float = 0.0  # largest jump in the raw spine (real discontinuity check)
    spine_lab: np.ndarray = field(repr=False, default=None)  # raw ordered spine (debug)
    # branch-drop diagnostics (DIAGNOSTIC ONLY; see extract_spine)
    branch_drop_frac: float = 0.0    # chosen-component nodes off the diameter path
    dropped_extent: float = 0.0      # OKLab gyration of those dropped voxels
    n_chosen: int = 0                # chosen-component node count
    n_path: int = 0                  # diameter-path node count (pre-trim)
    # failure gate (Phase 2; additive — to_colormap unchanged, so the emitted
    # colormap JSON is byte-identical whether or not a palette is flagged)
    extent: float = 0.0              # palette OKLab gyration (color range)
    arclen: float = 0.0              # palette curve arc length
    quality_flags: list = field(default_factory=list)  # [] == passes the gate

    def to_colormap(self, name: str) -> dict:
        n = len(self.stops_rgb)
        return {
            "name": name,
            "source": "extracted",
            "closed": True,
            "closure": self.closure,
            "coverage": round(float(self.coverage), 4),
            "max_step_oklab": round(float(self.max_step), 4),
            "stops": [[i / n, self.stops_rgb[i].tolist()] for i in range(n)],
        }


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def load_pixels(path: Path, max_samples: int, seed: int = 0) -> np.ndarray:
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    if len(arr) > max_samples:
        arr = arr[np.random.default_rng(seed).choice(len(arr), max_samples, replace=False)]
    return arr.astype(np.float64)


def density_voxels(lab: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize OKLab; return per-voxel centroid and pixel mass."""
    mins, maxs = lab.min(0), lab.max(0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    vox = np.clip(((lab - mins) / span * res).astype(int), 0, res - 1)
    key = (vox[:, 0] * res + vox[:, 1]) * res + vox[:, 2]
    order = np.argsort(key, kind="stable")
    key_s, lab_s = key[order], lab[order]
    uniq, start, counts = np.unique(key_s, return_index=True, return_counts=True)
    centroids = np.add.reduceat(lab_s, start, axis=0) / counts[:, None]
    return centroids, counts.astype(float)


def select_ridge(cent: np.ndarray, mass: np.ndarray,
                 mass_fraction: float,
                 support_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Admit a voxel if its mass clears the cumulative-rank cutoff *OR* an absolute
    `support_floor` (pixel-mass units; 0 disables the floor).

    The cumulative cutoff (densest voxels summing to `mass_fraction` of total mass)
    discards AA haze, but in the *mass-imbalanced* regime a single dominant blob
    fills the budget within its top few voxels, lifting the rank threshold above
    every thin-subject voxel — they never enter the graph. `support_floor` is a
    rank-independent floor (set just above per-voxel noise) that re-admits those
    thin voxels. Since a floor below the rank threshold subsumes it, the union is
    effectively `mass >= min(thr, support_floor)` whenever the floor is active."""
    sc = np.sort(mass)[::-1]
    cum = np.cumsum(sc) / mass.sum()
    thr = sc[min(np.searchsorted(cum, mass_fraction), len(sc) - 1)]
    keep = mass >= thr
    if support_floor > 0:
        keep = keep | (mass >= support_floor)
    return cent[keep], mass[keep]


def _knn_graph(P: np.ndarray, k: int) -> csr_matrix:
    k = min(k + 1, len(P))
    d, nbr = cKDTree(P).query(P, k=k)
    n = len(P)
    rows = np.repeat(np.arange(n), k - 1)
    cols = nbr[:, 1:].ravel()
    w = d[:, 1:].ravel()
    G = csr_matrix((w, (rows, cols)), shape=(n, n))
    return G.maximum(G.T)


def _mst_diameter(Pb: np.ndarray, Gb: csr_matrix) -> np.ndarray:
    """MST + two-sweep tree-diameter -> ordered spine points."""
    return Pb[_mst_diameter_idx(Gb)]


def _mst_diameter_idx(Gb: csr_matrix) -> np.ndarray:
    """MST + two-sweep tree-diameter -> ordered node *indices* of the simple path.
    Nodes off this path (side branches) are exactly what the traversal discards."""
    T = minimum_spanning_tree(Gb); T = T.maximum(T.T)
    far, _ = dijkstra(T, directed=False, indices=0, return_predecessors=True)
    u = int(np.argmax(far))
    dist_u, pred = dijkstra(T, directed=False, indices=u, return_predecessors=True)
    v = int(np.argmax(dist_u))
    path = [v]
    while path[-1] != u and path[-1] >= 0:
        path.append(int(pred[path[-1]]))
    return np.array(path[::-1])


def _extent(Pc: np.ndarray) -> float:
    """Radius of gyration in OKLab = sqrt(total variance). 'Spanned color space':
    rewards reach, not path length (arc) and not filled volume."""
    return float(np.sqrt(((Pc - Pc.mean(0)) ** 2).sum(1).mean()))


def extract_spine(P: np.ndarray, mass: np.ndarray, k: int, trim_delta: float,
                  min_component_nodes: int | None = None, verbose: bool = True,
                  diag: dict | None = None) -> np.ndarray:
    """Order the dominant ridge into a single open principal curve.
    Component selection = max spanned color-space extent (gyration), gated by a
    node-count support floor — NOT largest mass (which collapses onto flat dark
    regions) and NOT arc length (which a noisy compact blob inflates).

    If `diag` is a dict it is filled with branch-drop diagnostics for the chosen
    component (DIAGNOSTIC ONLY — the traversal still returns just the diameter
    path; nothing here changes what is dropped):
      n_admitted   - total ridge voxels fed to the graph
      n_chosen     - nodes in the selected (max-extent) component
      n_path       - nodes on the tree-diameter path (the spine, pre-trim)
      branch_drop_frac - (n_chosen - n_path) / n_chosen  [traversal loss]
      dropped_extent   - OKLab gyration of the chosen-component off-path voxels
      offpath_frac_admitted - (n_admitted - n_path) / n_admitted  [+ non-chosen comps]
    """
    if min_component_nodes is None:
        min_component_nodes = k + 1
    G = _knn_graph(P, k)
    _, cc = connected_components(G, directed=False)

    best_spine, best_ext, rows = None, -1.0, []
    best_Pc, best_path = None, None
    for c in np.unique(cc):
        sel = cc == c
        if int(sel.sum()) < min_component_nodes:
            continue
        Pc = P[sel]
        path_c = _mst_diameter_idx(G[sel][:, sel])
        spine_c = Pc[path_c]
        ext = _extent(Pc)
        rows.append((int(c), int(sel.sum()), float(mass[sel].sum()), ext, float(Pc[:, 0].mean())))
        if ext > best_ext:
            best_ext, best_spine = ext, spine_c
            best_Pc, best_path = Pc, path_c

    if best_spine is None:                       # degenerate fallback: all specks
        sel = cc == np.argmax(np.bincount(cc, weights=mass))
        best_Pc = P[sel]
        best_path = _mst_diameter_idx(G[sel][:, sel])
        best_spine = best_Pc[best_path]

    if diag is not None:
        n_chosen = len(best_Pc)
        n_path = len(best_path)
        off = np.ones(n_chosen, bool); off[best_path] = False
        dropped = best_Pc[off]
        diag.update(
            n_admitted=int(len(P)),
            n_chosen=int(n_chosen),
            n_path=int(n_path),
            branch_drop_frac=float((n_chosen - n_path) / n_chosen) if n_chosen else 0.0,
            dropped_extent=_extent(dropped) if len(dropped) >= 2 else 0.0,
            offpath_frac_admitted=float((len(P) - n_path) / len(P)) if len(P) else 0.0,
        )

    if verbose and rows:
        rows.sort(key=lambda r: -r[3])
        print("  component selection (chosen = max spanned extent / gyration):")
        for ci, nn, m, e, ml in rows:
            print(f"    comp{ci:3d} nodes={nn:5d} mass={m:12.0f} extent={e:.4f} "
                  f"meanL={ml:.3f}" + ("  <- chosen" if e == best_ext else ""))

    spine = best_spine
    # trim leading/trailing nodes joined by an over-budget (sparse) jump
    step = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    lo, hi = 0, len(spine) - 1
    while lo < hi - 2 and step[lo] > trim_delta:
        lo += 1
    while hi > lo + 2 and step[hi - 1] > trim_delta:
        hi -= 1
    return spine[lo:hi + 1]


def close_palette(spine: np.ndarray, tau_close: float) -> tuple[np.ndarray, str, float]:
    """Return (loop_points, closure_type, endpoint_gap).
    Native loop if the spine's ends already meet; otherwise mirror out-and-back."""
    gap = float(np.linalg.norm(spine[0] - spine[-1]))
    if gap <= tau_close:
        return spine, "native", gap                      # ends meet -> already a loop
    mirror = np.vstack([spine, spine[-2:0:-1]])           # c0..cn..c1  (-> c0 closes)
    return mirror, "mirrored", gap


# --------------------------------------------------------------------------- #
# Best-open vs best-cycle: soft seam-penalised path over the SAME MST.
#   score(P) = arclength(P) - lam * ‖lab(end_a) - lab(end_b)‖   (OKLab seam)
# lam=0 recovers the tree-diameter (= best-open) exactly; lam>0 trades a little
# length for a closeable seam. Single-pass: no Euler tour, no branch coverage.
# --------------------------------------------------------------------------- #

def _mst_sym(Gb: csr_matrix) -> csr_matrix:
    """Symmetric MST of a (sub)graph."""
    T = minimum_spanning_tree(Gb)
    return T.maximum(T.T)


def _tree_leaves(T: csr_matrix) -> np.ndarray:
    """Degree-1 node indices of a tree (candidate path endpoints)."""
    deg = np.asarray((T > 0).sum(axis=1)).ravel()
    return np.where(deg == 1)[0]


def _reconstruct(pred_row: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Walk predecessors from dst back to src; return src..dst index path."""
    path = [int(dst)]
    while path[-1] != src and path[-1] >= 0:
        path.append(int(pred_row[path[-1]]))
    return np.array(path[::-1])


def best_path_soft(T: csr_matrix, P: np.ndarray, lam: float,
                   leaf_cap: int = 80,
                   min_arclen: float = 0.0) -> tuple[np.ndarray, float, float]:
    """argmax over tree-leaf pairs of arclength - lam * endpoint OKLab gap.

    Tree geodesic distance == polyline arclength (edge weights are OKLab steps),
    so dijkstra from each leaf gives both the candidate lengths and the path.
    Returns (path_idx, arclength, endpoint_gap). lam=0 -> tree diameter.

    `min_arclen` floors the admissible arclength: without it the soft penalty has
    a *trivial* optimum at large lam — a pair of adjacent leaves with gap≈0 AND
    arclength≈0 (a 2-node loop) scores higher than the real long-but-open loop, so
    a genuinely-cyclic palette closes onto garbage. Flooring arclength (set to a
    fraction of the diameter by the caller) keeps closure restricted to long paths;
    lam=0 with min_arclen=0 is the exact diameter.

    O(L^2) over leaves; if a real ridge has many leaves we cap to the `leaf_cap`
    most peripheral ones (farthest from the component centroid) — the extreme
    endpoints are the only plausible path tips anyway."""
    leaves = _tree_leaves(T)
    if len(leaves) < 2:                                   # degenerate: fall to diameter
        far, _ = dijkstra(T, directed=False, indices=0, return_predecessors=True)
        u = int(np.argmax(far))
        du, predu = dijkstra(T, directed=False, indices=u, return_predecessors=True)
        v = int(np.argmax(du))
        return _reconstruct(predu, u, v), float(du[v]), \
            float(np.linalg.norm(P[u] - P[v]))
    if len(leaves) > leaf_cap:
        d = np.linalg.norm(P[leaves] - P.mean(0), axis=1)
        leaves = leaves[np.argsort(d)[::-1][:leaf_cap]]
    dist, preds = dijkstra(T, directed=False, indices=leaves,
                           return_predecessors=True)        # (L, N)
    arclen = dist[:, leaves]                               # (L, L) geodesic lengths
    gaps = np.linalg.norm(P[leaves][:, None, :] - P[leaves][None, :, :], axis=2)
    score = arclen - lam * gaps
    np.fill_diagonal(score, -np.inf)
    score[~np.isfinite(arclen)] = -np.inf                 # disconnected leaf pairs
    if min_arclen > 0:
        score[arclen < min_arclen] = -np.inf
    ii, jj = np.unravel_index(int(np.argmax(score)), score.shape)
    path = _reconstruct(preds[ii], int(leaves[ii]), int(leaves[jj]))
    return path, float(arclen[ii, jj]), float(gaps[ii, jj])


def count_revisit_branches(T: csr_matrix, P: np.ndarray, path_idx: np.ndarray,
                           floor: float) -> tuple[int, float]:
    """LOG-ONLY (Build 3): count off-path branches whose OKLab extent exceeds
    `floor` — a genuine color excursion the single path discards (a candidate
    real revisit, NOT preserved). Returns (n_branches, max_branch_extent)."""
    n = T.shape[0]
    on = np.zeros(n, bool); on[path_idx] = True
    off = np.where(~on)[0]
    if len(off) == 0:
        return 0, 0.0
    sub = T[off][:, off]
    n_cc, lbl = connected_components(sub, directed=False)
    n_rev, max_ext = 0, 0.0
    for c in range(n_cc):
        members = off[lbl == c]
        if len(members) < 2:
            continue
        ext = _extent(P[members])
        if ext > floor:
            n_rev += 1
            max_ext = max(max_ext, ext)
    return n_rev, float(max_ext)


@dataclass
class CycleResult:
    """Best-open vs best-cycle extraction over one image (see extract_palette_cycles)."""
    stops_open_rgb: np.ndarray
    stops_cycle_rgb: np.ndarray
    stops_open_lab: np.ndarray
    stops_cycle_lab: np.ndarray
    seam_open: float                 # endpoint OKLab gap of the diameter (best-open) path
    seam_cycle: float                # endpoint OKLab gap of the soft (best-cycle) path
    arclen_open: float
    arclen_cycle: float
    seam_open_pretrim: float         # diameter gap BEFORE tip-trim (Step-0 trim probe)
    cycle_label: str                 # "native" | "sequential" (seam_cycle > threshold)
    lam: float
    arc_retain: float = 0.5          # arclength floor (fraction of diameter) for the cycle
    n_ridge: int = 0
    n_chosen: int = 0                # chosen-component node count
    revisit_branches: int = 0        # Build 3 log-only
    revisit_max_extent: float = 0.0


def _choose_max_extent_component(P: np.ndarray, mass: np.ndarray, k: int,
                                 min_component_nodes: int | None):
    """Component selection identical to extract_spine (max OKLab gyration), but
    returns the chosen component's points + its MST for the cycle path."""
    if min_component_nodes is None:
        min_component_nodes = k + 1
    G = _knn_graph(P, k)
    _, cc = connected_components(G, directed=False)
    best_ext, best_sel = -1.0, None
    for c in np.unique(cc):
        sel = cc == c
        if int(sel.sum()) < min_component_nodes:
            continue
        ext = _extent(P[sel])
        if ext > best_ext:
            best_ext, best_sel = ext, sel
    if best_sel is None:
        best_sel = cc == np.argmax(np.bincount(cc, weights=mass))
    Pc = P[best_sel]
    T = _mst_sym(G[best_sel][:, best_sel])
    return Pc, T


def _trim_spine(spine: np.ndarray, trim_delta: float) -> np.ndarray:
    """Drop leading/trailing nodes joined by an over-budget (sparse) jump."""
    step = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    lo, hi = 0, len(spine) - 1
    while lo < hi - 2 and step[lo] > trim_delta:
        lo += 1
    while hi > lo + 2 and step[hi - 1] > trim_delta:
        hi -= 1
    return spine[lo:hi + 1]


def extract_palette_cycles(
    path: Path,
    n_stops: int = 256,
    max_samples: int = 600_000,
    voxel_res: int = 48,
    mass_fraction: float = 0.90,
    knn_k: int = 8,
    trim_delta: float = 0.06,
    lam: float = 0.5,
    arc_retain: float = 0.5,
    seam_seq_threshold: float = 0.10,
    revisit_floor: float = 0.06,
    smooth_frac: float = 0.012,
    support_floor: float = 0.0,
    seed: int = 0,
) -> CycleResult:
    """Emit BOTH best-open (diameter) and best-cycle (soft-closed) for one image.

    Best-open mirrors the shipped extractor (diameter + tip-trim); seam_open is
    its endpoint gap (post-trim, as the shipped close_palette measures it), and
    seam_open_pretrim is the same gap before trimming (Step-0 probe). Best-cycle
    joins its chosen endpoints natively and reports seam_cycle; mirror is used
    ONLY as a labelled "sequential" fallback when seam_cycle exceeds the
    (reported, not tuned) `seam_seq_threshold`. The default extract_palette path
    is untouched."""
    rgb = load_pixels(path, max_samples, seed)
    lab = srgb_to_oklab(rgb)
    cent, mass = density_voxels(lab, voxel_res)
    P, M = select_ridge(cent, mass, mass_fraction, support_floor)
    Pc, T = _choose_max_extent_component(P, M, knn_k, None)

    open_path, arc_open, _ = best_path_soft(T, Pc, 0.0)
    cyc_path, arc_cycle, seam_cycle = best_path_soft(
        T, Pc, lam, min_arclen=arc_retain * arc_open)

    open_spine_raw = Pc[open_path]
    seam_open_pre = float(np.linalg.norm(open_spine_raw[0] - open_spine_raw[-1]))
    open_spine = _trim_spine(open_spine_raw, trim_delta)
    seam_open = float(np.linalg.norm(open_spine[0] - open_spine[-1]))

    cyc_spine = Pc[cyc_path]
    n_rev, rev_ext = count_revisit_branches(T, Pc, cyc_path, revisit_floor)

    # best-open: honest closure is mirror out-and-back (a diameter rarely self-meets).
    open_loop = np.vstack([open_spine, open_spine[-2:0:-1]])
    # best-cycle: close natively unless the seam is too wide -> labelled sequential.
    if seam_cycle <= seam_seq_threshold:
        cyc_label = "native"
        cyc_loop = cyc_spine
    else:
        cyc_label = "sequential"
        cyc_loop = np.vstack([cyc_spine, cyc_spine[-2:0:-1]])

    open_loop = smooth_loop(open_loop, smooth_frac)
    cyc_loop = smooth_loop(cyc_loop, smooth_frac)
    open_lab = resample_closed(open_loop, n_stops)
    cyc_lab = resample_closed(cyc_loop, n_stops)

    return CycleResult(
        stops_open_rgb=oklab_to_srgb(open_lab), stops_cycle_rgb=oklab_to_srgb(cyc_lab),
        stops_open_lab=open_lab, stops_cycle_lab=cyc_lab,
        seam_open=seam_open, seam_cycle=seam_cycle,
        arclen_open=arc_open, arclen_cycle=arc_cycle,
        seam_open_pretrim=seam_open_pre, cycle_label=cyc_label, lam=lam,
        arc_retain=arc_retain, n_ridge=len(P), n_chosen=len(Pc),
        revisit_branches=n_rev, revisit_max_extent=rev_ext,
    )


def resample_closed(loop: np.ndarray, n_stops: int) -> np.ndarray:
    """Uniform arc-length resample around a closed loop (no duplicated endpoint)."""
    closed = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, arc[-1], n_stops, endpoint=False)
    return np.stack([np.interp(targets, arc, closed[:, c]) for c in range(3)], axis=1)


def smooth_loop(loop: np.ndarray, sigma_frac: float, dense: int = 2048) -> np.ndarray:
    """Periodic (wrap) Gaussian smoothing of a closed loop in OKLab.
    Resamples to `dense` uniform-arc points, convolves each channel circularly.
    Kills the zig-zag that MST-diameter produces on thick / sheet-like ridges."""
    pts = resample_closed(loop, dense)
    if sigma_frac <= 0:
        return pts
    sigma = max(sigma_frac * dense, 0.8)
    half = int(np.ceil(3 * sigma))
    x = np.arange(-half, half + 1)
    ker = np.exp(-(x ** 2) / (2 * sigma ** 2)); ker /= ker.sum()
    out = np.empty_like(pts)
    for c in range(3):
        out[:, c] = np.convolve(np.r_[pts[-half:, c], pts[:, c], pts[:half, c]], ker, "same")[half:-half]
    return out


def polyline_coverage(lab_all: np.ndarray, loop: np.ndarray, eps: float,
                      dense: int = 2048) -> float:
    """Fraction of pixels within `eps` (OKLab) of the palette *curve*
    (not just its discrete stops). Dense-resample + KD-tree is an accurate proxy."""
    curve = resample_closed(loop, dense)
    mind, _ = cKDTree(curve).query(lab_all, k=1)
    return float((mind <= eps).mean())


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #

def extract_palette(
    path: Path,
    n_stops: int = 256,
    max_samples: int = 600_000,
    voxel_res: int = 48,
    mass_fraction: float = 0.90,
    knn_k: int = 8,
    trim_delta: float = 0.06,
    tau_close: float = 0.10,
    smooth_frac: float = 0.012,
    coverage_eps: float = 0.05,
    support_floor: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
    reject: bool = False,
) -> PaletteResult:
    """Extract one closed palette. `reject=True` (opt-in) raises PaletteRejected
    when the failure gate (gate_quality) flags the result; the default path is
    unchanged and always returns the PaletteResult (now carrying extent/arclen/
    quality_flags as additive fields)."""
    rgb = load_pixels(path, max_samples, seed)
    lab = srgb_to_oklab(rgb)
    cent, mass = density_voxels(lab, voxel_res)
    P, M = select_ridge(cent, mass, mass_fraction, support_floor)
    diag: dict = {}
    spine = extract_spine(P, M, knn_k, trim_delta, verbose=verbose, diag=diag)

    raw_step = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    loop, closure, gap = close_palette(spine, tau_close)
    loop = smooth_loop(loop, smooth_frac)
    stops_lab = resample_closed(loop, n_stops)
    stops_rgb = oklab_to_srgb(stops_lab)

    closed = np.vstack([stops_lab, stops_lab[:1]])
    step = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    coverage = polyline_coverage(lab, loop, coverage_eps)

    extent = _extent(stops_lab)
    arclen = _loop_arclen(stops_lab)
    flags = gate_quality(extent, arclen)
    if reject and flags:
        raise PaletteRejected(flags, extent, arclen)

    return PaletteResult(
        stops_rgb=stops_rgb, stops_lab=stops_lab, closure=closure,
        coverage=coverage, max_step=float(step.max()), mean_step=float(step.mean()),
        endpoint_gap=gap, n_ridge=len(P), spine_lab=spine,
        raw_spine_max_step=float(raw_step.max()) if len(raw_step) else 0.0,
        branch_drop_frac=diag.get("branch_drop_frac", 0.0),
        dropped_extent=diag.get("dropped_extent", 0.0),
        n_chosen=diag.get("n_chosen", 0),
        n_path=diag.get("n_path", 0),
        extent=extent, arclen=arclen, quality_flags=flags,
    )


def _diagnostics(path: Path, res: PaletteResult, out_png: Path,
                 coverage_eps: float, max_samples: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rgb = load_pixels(path, max_samples, seed)
    lab = srgb_to_oklab(rgb)
    cent, mass = density_voxels(lab, 48)
    P, M = select_ridge(cent, mass, 0.90)
    col = np.clip(oklab_to_srgb(P) / 255.0, 0, 1)
    s = res.stops_lab
    srgb = res.stops_rgb / 255.0

    small = np.asarray(Image.open(path).convert("RGB").resize((720, 450)))
    H, W, _ = small.shape
    dmin, _ = cKDTree(s).query(srgb_to_oklab(small.reshape(-1, 3).astype(float)))
    ov = (small.reshape(-1, 3) / 255.0).copy()
    ov[dmin > coverage_eps] = [1, 0, 1]
    ov = ov.reshape(H, W, 3)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    ax[0, 0].scatter(P[:, 1], P[:, 2], c=col, s=6 + 30 * M / M.max(), edgecolors="none")
    ax[0, 0].plot(s[:, 1], s[:, 2], "k-", lw=1.2, alpha=.8)
    ax[0, 0].scatter(s[:, 1], s[:, 2], c=srgb, s=70, edgecolors="k", linewidths=.5, zorder=5)
    ax[0, 0].set_title("ridge (a,b) + extracted loop"); ax[0, 0].set_aspect("equal")
    ax[0, 1].imshow(np.tile(srgb[None], (50, 1, 1)), aspect="auto")
    ax[0, 1].set_title(f"palette [{res.closure}]  cov {res.coverage*100:.0f}%  max-step {res.max_step:.3f}")
    ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([])
    closed = np.vstack([s, s[:1]])
    ax[0, 2].plot(np.linalg.norm(np.diff(closed, axis=0), axis=1), lw=1)
    ax[0, 2].axhline(0.055, color="r", ls="--", lw=1, label="δ~0.055"); ax[0, 2].legend()
    ax[0, 2].set_title("consecutive ΔE around loop")
    ax[1, 0].imshow(small); ax[1, 0].set_title("original"); ax[1, 0].axis("off")
    ax[1, 1].imshow(ov); ax[1, 1].set_title(f"magenta = uncovered (ε={coverage_eps})"); ax[1, 1].axis("off")
    ch = np.sqrt(P[:, 1]**2 + P[:, 2]**2); chs = np.sqrt(s[:, 1]**2 + s[:, 2]**2)
    ax[1, 2].scatter(ch, P[:, 0], c=col, s=6 + 30 * M / M.max(), edgecolors="none")
    ax[1, 2].plot(chs, s[:, 0], "k-", lw=1.2, alpha=.8); ax[1, 2].set_title("ridge (chroma, L) + loop")
    plt.tight_layout(); plt.savefig(out_png, dpi=92); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a single closed palette from a fractal image.")
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None, help="output colormap .json")
    ap.add_argument("-n", "--stops", type=int, default=256)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--mass-fraction", type=float, default=0.90)
    ap.add_argument("--voxel-res", type=int, default=48)
    ap.add_argument("--trim-delta", type=float, default=0.06)
    ap.add_argument("--tau-close", type=float, default=0.10)
    ap.add_argument("--smooth-frac", type=float, default=0.012)
    ap.add_argument("--support-floor", type=float, default=0.0,
                    help="absolute per-voxel mass floor (pixel counts); 0 disables")
    ap.add_argument("--eps", type=float, default=0.05, help="coverage threshold (OKLab)")
    ap.add_argument("--diagnostics", type=Path, default=None, help="write a diagnostic .png")
    args = ap.parse_args()

    res = extract_palette(
        args.image, n_stops=args.stops, voxel_res=args.voxel_res,
        mass_fraction=args.mass_fraction, trim_delta=args.trim_delta, tau_close=args.tau_close,
        smooth_frac=args.smooth_frac, coverage_eps=args.eps, support_floor=args.support_floor,
    )
    name = args.name or args.image.stem
    print(f"{name}: closure={res.closure} coverage={res.coverage*100:.1f}% "
          f"max_step={res.max_step:.4f} mean_step={res.mean_step:.4f} "
          f"endpoint_gap={res.endpoint_gap:.4f} ridge={res.n_ridge} "
          f"branch_drop={res.branch_drop_frac*100:.1f}% dropped_extent={res.dropped_extent:.4f}")
    if args.out:
        args.out.write_text(json.dumps(res.to_colormap(name)))
        print(f"wrote {args.out}")
    if args.diagnostics:
        _diagnostics(args.image, res, args.diagnostics, args.eps, 600_000, 0)
        print(f"wrote {args.diagnostics}")


if __name__ == "__main__":
    main()
