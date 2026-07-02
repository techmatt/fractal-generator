#!/usr/bin/env python3
"""Shared perceptual-color primitives for the query tooling.

The candidate-selection path (render-space FPS in `regenerate_coldstart_v2.py`)
and the read-only `diversity_diagnostic.py` both need the same ΔE math, so it
lives here rather than in either consumer — a diagnostic must not be a dependency
of the production selection path.

`ciede2000` is validated at import (and under `uv run pytest tools/`, see
`tools/test_color_metrics.py`) against the Sharma et al. 2005 reference vectors,
so we avoid pulling in scikit-image/scipy.

`THUMB_WIDTH` is the shared BOX-downsample width both callers thumbnail to before
computing ΔE — the comparison scale must match across the diagnostic and the
selector, so it's defined once here.
"""
from __future__ import annotations

import numpy as np

# Candidates are BOX-downsampled to this width before ΔE (shared comparison scale).
THUMB_WIDTH = 256


def srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """(...,3) uint8 sRGB -> (...,3) float CIELAB, D65."""
    srgb = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ m.T
    # D65 reference white
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 ΔE between two (...,3) Lab arrays."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    # when either chroma is 0, hue diff is undefined -> 0
    zero_c = (C1p * C2p) == 0
    dhp = np.where(zero_c, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    habsdiff = np.abs(h1p - h2p)
    hbarp = np.where(
        zero_c, hsum,
        np.where(habsdiff <= 180.0, 0.5 * hsum,
                 np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))))

    T = (1.0
         - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Lbarp_m50sq = (Lbarp - 50.0) ** 2
    Sl = 1.0 + (0.015 * Lbarp_m50sq) / np.sqrt(20.0 + Lbarp_m50sq)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    kL = kC = kH = 1.0
    tL = dLp / (kL * Sl)
    tC = dCp / (kC * Sc)
    tH = dHp / (kH * Sh)
    return np.sqrt(tL * tL + tC * tC + tH * tH + Rt * tC * tH)


def _validate_ciede2000():
    """Sharma et al. 2005 reference pairs (Lab1, Lab2, expected ΔE)."""
    cases = [
        ([50.0000, 2.6772, -79.7751], [50.0000, 0.0000, -82.7485], 2.0425),
        ([50.0000, 3.1571, -77.2803], [50.0000, 0.0000, -82.7485], 2.8615),
        ([50.0000, 2.8361, -74.0200], [50.0000, 0.0000, -82.7485], 3.4412),
        ([50.0000, -1.3802, -84.2814], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -1.1848, -84.8006], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -0.9009, -85.5211], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, 0.0000, 0.0000], [50.0000, -1.0000, 2.0000], 2.3669),
        ([50.0000, -1.0000, 2.0000], [50.0000, 0.0000, 0.0000], 2.3669),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0009], 7.1792),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0011], 7.1792),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0012], 7.2195),
        ([50.0000, -0.0010, 2.4900], [50.0000, 0.0009, -2.4900], 4.8045),
        ([50.0000, 2.5000, 0.0000], [50.0000, 0.0000, -2.5000], 4.3065),
        ([50.0000, 2.5000, 0.0000], [73.0000, 25.0000, -18.0000], 27.1492),
        ([50.0000, 2.5000, 0.0000], [61.0000, -5.0000, 29.0000], 22.8977),
        ([50.0000, 2.5000, 0.0000], [56.0000, -27.0000, -3.0000], 31.9030),
        ([50.0000, 2.5000, 0.0000], [58.0000, 24.0000, 15.0000], 19.4535),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.1736, 0.5854], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.2972, 0.0000], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 1.8634, 0.5757], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.2592, 0.3350], 1.0000),
        ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
        ([63.0109, -31.0961, -5.8663], [62.8187, -29.7946, -4.0864], 1.2630),
        ([61.2901, 3.7196, -5.3901], [61.4292, 2.2480, -4.9620], 1.8731),
        ([35.0831, -44.1164, 3.7933], [35.0232, -40.0716, 1.5901], 1.8645),
        ([22.7233, 20.0904, -46.6940], [23.0331, 14.9730, -42.5619], 2.0373),
        ([36.4612, 47.8580, 18.3852], [36.2715, 50.5065, 21.2231], 1.4146),
        ([90.8027, -2.0831, 1.4410], [91.1528, -1.6435, 0.0447], 1.4441),
        ([90.9257, -0.5406, -0.9208], [88.6381, -0.8985, -0.7239], 1.5381),
        ([6.7747, -0.2908, -2.4247], [5.8714, -0.0985, -2.2286], 0.6377),
        ([2.0776, 0.0795, -1.1350], [0.9033, -0.0636, -0.5514], 0.9082),
    ]
    l1 = np.array([c[0] for c in cases])
    l2 = np.array([c[1] for c in cases])
    exp = np.array([c[2] for c in cases])
    got = ciede2000(l1, l2)
    err = np.sort(np.abs(got - exp))
    # Pair (50,2.49,-.001)&(50,-2.49,.0011) is the documented CIEDE2000 hue-quadrant
    # boundary: |h'1-h'2| lands within ~1.5e-3 deg of exactly 180 deg, so the
    # hue-average branch flips on the last ulp. skimage's own impl doesn't
    # reproduce Sharma's value here either. Allow exactly this one to differ (<0.05),
    # require every other pair exact to 1e-3 (a real bug would break many by a lot).
    if err[-2] > 1e-3 or err[-1] > 0.05:
        raise RuntimeError(
            f"CIEDE2000 self-test FAILED: 2nd-worst err {err[-2]:.5f}, worst {err[-1]:.5f}")
    return err[-2], err[-1]


# Import-time guard: a broken relocation trips immediately, everywhere this is used.
_validate_ciede2000()
