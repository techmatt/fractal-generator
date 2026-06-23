"""Validated coloring path, ported from the Rust engine.

Faithful port of:
  - src/palette_io.rs : parse_ugr / parse_map
  - src/palette.rs    : sRGB transfer, Ottosson OKLab matrices, cyclic OKLab
                        interpolation, 4096-entry linear-RGB LUT bake, reverse
  - src/coloring.rs   : t = value*density + offset (mod 1) -> lookup_linear

The Rust is the reference (the original Python `coloring.py` no longer exists on
disk). Constants below are copied verbatim from palette.rs. Vectorized with numpy;
the bake uses np.interp over a cyclically-extended stop list, which reproduces
interp_oklab_cyclic exactly (passes through stops, no 1->0 seam).
"""

from __future__ import annotations

import numpy as np

LUT_SIZE = 4096

# Density multiplier applied when a palette is pre-mirrored (selective seam fix for
# SEQUENTIAL maps). Pre-mirror folds the gradient into an out-and-back, doubling the
# spatial band frequency; this keeps a mirrored map's band count ~matched to the
# un-mirrored original. Matched to `palette::MIRROR_DENSITY_SCALE` (Rust).
MIRROR_DENSITY_SCALE = 0.5

# ---------------------------------------------------------------------------
# sRGB transfer function (palette.rs)
# ---------------------------------------------------------------------------


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# OKLab (Ottosson) — linear sRGB <-> OKLab (palette.rs, exact constants)
# ---------------------------------------------------------------------------

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M2_INV = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
_M1_INV = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


def linear_srgb_to_oklab(rgb):
    """rgb: (...,3) linear sRGB -> (...,3) OKLab."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = rgb @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_linear_srgb(lab):
    """lab: (...,3) OKLab -> (...,3) linear sRGB."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lms = lms_ ** 3
    return lms @ _M1_INV.T


def srgb8_to_oklab(rgb8):
    """rgb8: (...,3) uint8/0-255 -> (...,3) OKLab."""
    return linear_srgb_to_oklab(srgb_to_linear(np.asarray(rgb8, dtype=np.float64) / 255.0))


# ---------------------------------------------------------------------------
# Parsers (palette_io.rs) — output common stop-list form: list[(pos, (r,g,b))]
# ---------------------------------------------------------------------------


def parse_map(text):
    """Fractint .map: lines of `R G B`. pos = i / N. Returns [(pos,(r,g,b))]."""
    colors = []
    for raw in text.splitlines():
        line = raw.split(";")[0].split("#")[0].strip()
        if not line:
            continue
        nums = line.split()
        if len(nums) < 3:
            continue
        try:
            rgb = tuple(int(np.clip(int(float(nums[k])), 0, 255)) for k in range(3))
        except ValueError:
            continue
        colors.append(rgb)
    if len(colors) < 2:
        return []
    n = len(colors)
    return [(i / n, c) for i, c in enumerate(colors)]


def parse_ugr(text):
    """UltraFractal .ugr -> list of (name, [(pos,(r,g,b))]).

    Tokenized port of parse_ugr/parse_ugr_block: `{`/`}` are their own tokens;
    the identifier before `{` is the block name; inside a block `index=N` (0-400
    -> pos N/400) followed by `color=INT` (COLORREF 0x00BBGGRR, R low byte) makes
    a stop. The opacity section's stray `index=` (no following `color=`) is
    dropped. Multi-line index/color runs are handled by the flat token stream.
    """
    spaced = text.replace("{", " { ").replace("}", " } ")
    toks = spaced.split()
    grads = []
    last_ident = None
    i = 0
    n = len(toks)
    while i < n:
        tok = toks[i]
        if tok == "{":
            name = last_ident if last_ident is not None else f"gradient{len(grads)}"
            last_ident = None
            i += 1
            stops, i = _parse_ugr_block(toks, i)
            if stops:
                grads.append((name, stops))
        elif tok == "}":
            i += 1
        else:
            last_ident = tok
            i += 1
    return grads


def _parse_ugr_block(toks, i):
    stops = []
    pending = None
    n = len(toks)
    while i < n:
        tok = toks[i]
        i += 1
        if tok == "}":
            break
        if tok == "{":
            continue
        if tok.startswith("index="):
            try:
                idx = float(tok[len("index="):])
                pending = (idx / 400.0) % 1.0
            except ValueError:
                pending = None
        elif tok.startswith("color="):
            val = tok[len("color="):]
            try:
                if val.lower().startswith("0x"):
                    colorref = int(val, 16)
                else:
                    colorref = int(val)
            except ValueError:
                continue
            if pending is not None:
                c = colorref & 0xFFFFFF
                r = c & 0xFF
                g = (c >> 8) & 0xFF
                b = (c >> 16) & 0xFF
                stops.append((pending, (r, g, b)))
                pending = None
    return stops, i


# ---------------------------------------------------------------------------
# Bake: stops -> cyclic OKLab interpolation -> LUT_SIZE linear-RGB entries
# (palette.rs from_oklab_stops / interp_oklab_cyclic, vectorized via np.interp)
# ---------------------------------------------------------------------------


def mirror_stops(stops):
    """Pre-mirror a stop list into a symmetric out-and-back (triangle).

    For a SEQUENTIAL (`mirror_needed`) palette the raw cyclic bake compresses the
    endpoint (last->first color) transition into the tiny wrap segment, producing a
    visible seam band. Reflecting the stops removes the seam: the forward gradient
    occupies positions [0, 0.5], its reflection occupies (0.5, 1), and the cyclic
    wrap (last->first) mirrors the opening segment, so endpoints meet on the same
    color. Density d then yields d out-and-back passes (a triangle wave).

    Matches `palette.rs::mirror_stops` byte-for-byte: same normalize+stable-sort,
    same u=(p-p0)/span remap, same 0.5*u / 1-0.5*u positions. Cyclic palettes are
    NOT passed here — the caller gates strictly on `mirror_needed`.
    """
    s = sorted(((p % 1.0, c) for p, c in stops), key=lambda x: x[0])
    n = len(s)
    p0 = s[0][0]
    span = s[-1][0] - p0
    if not (span > 0.0):
        return [(p, c) for p, c in s]  # degenerate: all stops coincide
    u = [(s[i][0] - p0) / span for i in range(n)]
    out = [(0.5 * u[i], s[i][1]) for i in range(n)]            # forward -> [0, 0.5]
    out += [(1.0 - 0.5 * u[i], s[i][1]) for i in range(n - 2, 0, -1)]  # reflection -> (0.5, 1)
    return out


def bake_lut(stops, lut_size=LUT_SIZE, reverse=False, mirror=False):
    """stops: list[(pos, (r,g,b))]. Returns (lut_size, 3) linear-RGB LUT.

    Positions normalized into [0,1) and stable-sorted; <2 distinct stops is an
    error (matches the Rust assert). Cyclic: the last stop wraps to the first.
    `mirror=True` (selective; pass only for `mirror_needed` palettes) first
    reflects the stops into a seamless out-and-back via `mirror_stops`.
    """
    if len(stops) < 2:
        raise ValueError("a palette needs at least two control points")
    if mirror:
        stops = mirror_stops(stops)
    pos = np.array([p % 1.0 for p, _ in stops], dtype=np.float64)
    lab = srgb8_to_oklab(np.array([c for _, c in stops], dtype=np.float64))
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    lab = lab[order]

    # Cyclically extend so np.interp covers the full [0,1) including the wrap
    # segment (last->first) and the region below the first stop.
    ext_pos = np.concatenate(([pos[-1] - 1.0], pos, [pos[0] + 1.0]))
    ext_lab = np.concatenate((lab[-1:], lab, lab[:1]), axis=0)

    t = np.arange(lut_size, dtype=np.float64) / lut_size
    lab_t = np.empty((lut_size, 3), dtype=np.float64)
    for ch in range(3):
        lab_t[:, ch] = np.interp(t, ext_pos, ext_lab[:, ch])
    lut = oklab_to_linear_srgb(lab_t)

    if reverse:
        src = lut.copy()
        idx = (lut_size - np.arange(lut_size)) % lut_size
        lut = src[idx]
    return lut


def lookup_linear(lut, t):
    """Vectorized cyclic LUT lookup with lerp. t any shape -> (...,3) linear RGB."""
    lut_size = lut.shape[0]
    t = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    x = t * lut_size
    i0 = np.floor(x).astype(np.int64)
    f = (x - i0)[..., None]
    i0 = i0 % lut_size
    i1 = (i0 + 1) % lut_size
    return lut[i0] * (1.0 - f) + lut[i1] * f


def colorize(field, lut, density=1.0, offset=0.0, interior_mask=None, mirror=False):
    """Map a value-field through a baked LUT (coloring.rs Smooth channel).

    field: (H,W) float value (e.g. smooth-iter). Returns linear-RGB (H,W,3).
    interior_mask: optional bool (H,W); True pixels -> black (InteriorMode::Black).
    mirror: pass True when `lut` was baked with `mirror=True` (pre-mirrored
        sequential palette) so the density is scaled by MIRROR_DENSITY_SCALE,
        matching `coloring::shade` (which reads `palette.density_scale()`).
    """
    if mirror:
        density = density * MIRROR_DENSITY_SCALE
    t = field * density + offset
    rgb_lin = lookup_linear(lut, t)
    if interior_mask is not None:
        rgb_lin = rgb_lin.copy()
        rgb_lin[interior_mask] = 0.0
    return rgb_lin
