"""sRGB <-> Oklab color conversion (Ottosson formulation), vectorized over numpy.

All sRGB values are 0-1 floats. Oklab is Cartesian (L, a, b) -- we deliberately do
NOT convert to L/C/h: hue wraparound near the neutral axis is exactly the artifact
the trajectory feature is built to avoid.

Reference: https://bottosson.github.io/posts/oklab/  (Ottosson, "A perceptual color
space for image processing"). The sRGB EOTF is the true piecewise curve (0.04045 /
12.92 / 1.055 threshold), not a 2.2 gamma approximation.
"""

import numpy as np

# linear-sRGB -> LMS (Ottosson M1)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
# LMS' (cube-rooted) -> Oklab (Ottosson M2)
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_linear(c):
    """sRGB EOTF (gamma-decode). c in [0,1] -> linear [0,1]."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Inverse sRGB EOTF (gamma-encode). linear [0,1] -> sRGB [0,1]."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def srgb_to_oklab(srgb):
    """(..., 3) sRGB in [0,1] -> (..., 3) Oklab (L, a, b) Cartesian."""
    srgb = np.asarray(srgb, dtype=np.float64)
    lin = srgb_to_linear(srgb)
    lms = lin @ _M1.T
    lms_ = np.cbrt(lms)  # cube root (Ottosson uses cbrt, handles the non-negative LMS here)
    return lms_ @ _M2.T


def oklab_to_srgb(lab):
    """(..., 3) Oklab -> (..., 3) sRGB in [0,1] (clipped). For validation swatches only."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lms = lms_ ** 3
    lin = lms @ _M1_INV.T
    return np.clip(linear_to_srgb(lin), 0.0, 1.0)
