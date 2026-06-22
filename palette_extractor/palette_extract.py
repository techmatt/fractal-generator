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
                 mass_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the densest voxels accounting for `mass_fraction` of total pixel mass.
    This is what discards antialiasing haze and thin tendrils."""
    sc = np.sort(mass)[::-1]
    cum = np.cumsum(sc) / mass.sum()
    thr = sc[min(np.searchsorted(cum, mass_fraction), len(sc) - 1)]
    keep = mass >= thr
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


def extract_spine(P: np.ndarray, mass: np.ndarray, k: int,
                  trim_delta: float) -> np.ndarray:
    """Order the dominant ridge into a single open principal curve via
    MST + two-sweep tree diameter, then trim over-budget sparse tips."""
    G = _knn_graph(P, k)
    _, cc = connected_components(G, directed=False)
    big = np.argmax(np.bincount(cc, weights=mass))
    sel = cc == big
    Pb, Gb = P[sel], G[sel][:, sel]

    T = minimum_spanning_tree(Gb)
    T = T.maximum(T.T)
    far, _ = dijkstra(T, directed=False, indices=0, return_predecessors=True)
    u = int(np.argmax(far))
    dist_u, pred = dijkstra(T, directed=False, indices=u, return_predecessors=True)
    v = int(np.argmax(dist_u))

    path = [v]
    while path[-1] != u and path[-1] >= 0:
        path.append(int(pred[path[-1]]))
    spine = Pb[np.array(path[::-1])]

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
    seed: int = 0,
) -> PaletteResult:
    rgb = load_pixels(path, max_samples, seed)
    lab = srgb_to_oklab(rgb)
    cent, mass = density_voxels(lab, voxel_res)
    P, M = select_ridge(cent, mass, mass_fraction)
    spine = extract_spine(P, M, knn_k, trim_delta)

    raw_step = np.linalg.norm(np.diff(spine, axis=0), axis=1)
    loop, closure, gap = close_palette(spine, tau_close)
    loop = smooth_loop(loop, smooth_frac)
    stops_lab = resample_closed(loop, n_stops)
    stops_rgb = oklab_to_srgb(stops_lab)

    closed = np.vstack([stops_lab, stops_lab[:1]])
    step = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    coverage = polyline_coverage(lab, loop, coverage_eps)

    return PaletteResult(
        stops_rgb=stops_rgb, stops_lab=stops_lab, closure=closure,
        coverage=coverage, max_step=float(step.max()), mean_step=float(step.mean()),
        endpoint_gap=gap, n_ridge=len(P), spine_lab=spine,
        raw_spine_max_step=float(raw_step.max()) if len(raw_step) else 0.0,
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
    ap.add_argument("--eps", type=float, default=0.05, help="coverage threshold (OKLab)")
    ap.add_argument("--diagnostics", type=Path, default=None, help="write a diagnostic .png")
    args = ap.parse_args()

    res = extract_palette(
        args.image, n_stops=args.stops, voxel_res=args.voxel_res,
        mass_fraction=args.mass_fraction, trim_delta=args.trim_delta, tau_close=args.tau_close,
        smooth_frac=args.smooth_frac, coverage_eps=args.eps,
    )
    name = args.name or args.image.stem
    print(f"{name}: closure={res.closure} coverage={res.coverage*100:.1f}% "
          f"max_step={res.max_step:.4f} mean_step={res.mean_step:.4f} "
          f"endpoint_gap={res.endpoint_gap:.4f} ridge={res.n_ridge}")
    if args.out:
        args.out.write_text(json.dumps(res.to_colormap(name)))
        print(f"wrote {args.out}")
    if args.diagnostics:
        _diagnostics(args.image, res, args.diagnostics, args.eps, 600_000, 0)
        print(f"wrote {args.diagnostics}")


if __name__ == "__main__":
    main()
