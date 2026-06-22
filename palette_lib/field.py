"""Mandelbrot smooth-iteration value-fields to colorize.

The original prototype's saved field (`nu_deep.npy`) doesn't survive, so we
generate one. Palettes are the point: any decently-structured field exposes a
palette's character, so a couple of classic moderate-zoom Mandelbrot crops
(seahorse valley, a spiral) are plenty. Smooth iter:
    nu = n + 1 - log2(log|z|)
exterior-only (the engine's Smooth channel); interior returned as a mask so the
colorizer can paint it black (InteriorMode::Black).
"""

from __future__ import annotations

import numpy as np

# (name, center_re, center_im, half_width) — f64 numpy is exact at these depths.
LOCATIONS = [
    ("seahorse", -0.745, 0.113, 0.0145),
    ("spiral", -0.7453, 0.1127, 0.0055),
]


def smooth_field(cre, cim, half_w, w, h, maxiter=600, bailout=1e6):
    """Return (nu, interior) for a crop centered at (cre,cim).

    nu: (h,w) smooth-iter, normalized to [0,1) over the escaped range.
    interior: (h,w) bool, True where the orbit never escaped.
    """
    half_h = half_w * h / w
    xs = np.linspace(cre - half_w, cre + half_w, w)
    ys = np.linspace(cim - half_h, cim + half_h, h)
    C = xs[None, :] + 1j * ys[:, None]
    Z = np.zeros_like(C)
    nu = np.zeros(C.shape, dtype=np.float64)
    escaped = np.zeros(C.shape, dtype=bool)
    active = np.ones(C.shape, dtype=bool)
    for n in range(maxiter):
        Z[active] = Z[active] * Z[active] + C[active]
        mag2 = (Z.real * Z.real + Z.imag * Z.imag)
        now = active & (mag2 > bailout * bailout)
        if now.any():
            az = np.sqrt(mag2[now])
            nu[now] = n + 1.0 - np.log2(np.log(az))
            escaped[now] = True
            active[now] = False
        if not active.any():
            break
    interior = ~escaped
    if escaped.any():
        lo = nu[escaped].min()
        hi = nu[escaped].max()
        if hi > lo:
            nu = (nu - lo) / (hi - lo)
    return nu, interior
