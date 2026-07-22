"""Python coloring tail for the field⊗colormap split.

The Rust engine dumps the **raw smooth scalar field** once per location
(`render-one --dump-field`); this module owns the *entire* coloring tail and never
recomputes field math. Given a super-res field (NaN interior) plus a
`CandidateConfig`, `render_candidate` produces an sRGB image at the eval size.

This is the shared candidate-generation code path for both label-time (1024px) and
the inference sweep (thousands of recolors per cached field). The correctness
contract is empirical: at canonical params it reproduces a known-good Rust smooth
render bit-close (see `colormap_acceptance.py` / `test_colormap.py`).

Pipeline order — **pinned to the Rust `render_modes.rs` smooth path**, NOT the order
listed in the build prompt (Step 0 reads the code and pins it):

    raw field  ->  percentile-stretch (0.5 / 99.5, over non-NaN)  ->  x in [0,1]
               ->  transform curve (log_premap) + gamma            ->  gray in [0,1]
               ->  n_cycles / phase (cyclic-only)                   ->  t in [0,1)
               ->  OKLab LUT sample (4096, cyclic, reverse/mirror baked)
               ->  interior fill (NaN -> interior_color)
               ->  linear-light downsample (box / lanczos3 / mitchell) -> sRGB8

i.e. the percentile-stretch is applied to the RAW field FIRST, then the transform
curve operates on the [0,1]-stretched value (Rust `apply_transform`). Every numeric
step mirrors the Rust source so the outputs match.

Palette type is binary {cyclic, non_cyclic}. The former diverging **center-pivot**
was the ONLY coloring knob with no Rust analog (it was Python-only); it has been
dropped -- diverging balance is now handled by reverse + gamma -- so every remaining
knob in this tail is either Rust-validated or trivially exact.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Color conversion — Rust-exact constants (src/palette.rs).
#
# The forward sRGB<->OKLab matrices match `tools/palettes/color.py` (identical to
# Rust). The OKLab->linear-sRGB *inverse* is Ottosson's published constants, which
# Rust hardcodes and which are NOT bit-identical to `np.linalg.inv(M)` — so we
# hardcode them here to keep the baked LUT bit-for-bit with the Rust bake.
# ---------------------------------------------------------------------------

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])
# OKLab -> LMS' (Ottosson inverse, matching Rust oklab_to_linear_srgb).
_M2_INV = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])
# LMS -> linear sRGB (Ottosson inverse, matching Rust oklab_to_linear_srgb).
_M1_INV = np.array([
    [ 4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
])


def srgb_to_linear(c):
    """sRGB EOTF (gamma-decode). c in [0,1] -> linear."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    """Inverse sRGB EOTF (gamma-encode), clamped to [0,1] input (Rust-exact)."""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def srgb_to_oklab(srgb):
    """(...,3) sRGB in [0,1] -> (...,3) OKLab (L,a,b)."""
    srgb = np.asarray(srgb, dtype=np.float64)
    lms = srgb_to_linear(srgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear_srgb(lab):
    """(...,3) OKLab -> (...,3) linear sRGB (Rust-exact inverse constants)."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lms = lms_ ** 3
    return lms @ _M1_INV.T


# ---------------------------------------------------------------------------
# Palette LUT bake — mirrors src/palette.rs (OKLab cyclic interp, 4096 entries).
# ---------------------------------------------------------------------------

LUT_SIZE = 4096


def mirror_stops(stops):
    """Pre-mirror a stop list into a symmetric out-and-back (Rust `mirror_stops`).

    `stops` is a list of (pos, [r,g,b]) with 8-bit rgb. Returns the reflected list
    (2n-2 stops). Matches the Rust construction: normalize into [0,1), stable sort,
    `u=(p-p0)/span`, forward -> 0.5u in [0,0.5], reflected -> 1-0.5u in (0.5,1),
    dropping the two endpoints.
    """
    s = sorted(((p % 1.0, rgb) for p, rgb in stops), key=lambda x: x[0])
    n = len(s)
    p0 = s[0][0]
    span = s[n - 1][0] - p0
    if not (span > 0.0):
        return s
    def u(i):
        return (s[i][0] - p0) / span
    out = []
    for i in range(n):
        out.append((0.5 * u(i), s[i][1]))          # forward -> [0, 0.5]
    for i in range(n - 2, 0, -1):
        out.append((1.0 - 0.5 * u(i), s[i][1]))    # reflection -> (0.5, 1)
    return out


def _interp_oklab_cyclic(pos, lab, t):
    """OKLab color at cyclic t in [0,1) — Rust `interp_oklab_cyclic`. `pos`/`lab`
    are the sorted stop positions / OKLab colors (1-D / (n,3))."""
    n = len(pos)
    for i in range(n):
        a_pos = pos[i]
        if i + 1 < n:
            pb, cb = pos[i + 1], lab[i + 1]
        else:
            pb, cb = pos[0] + 1.0, lab[0]
        if a_pos <= t < pb:
            f = (t - a_pos) / (pb - a_pos)
            return lab[i] + (cb - lab[i]) * f
    # Wrap segment: last stop -> first stop, below the first stop's position.
    last_p, last_c = pos[n - 1], lab[n - 1]
    first_p, first_c = pos[0], lab[0]
    span = (first_p + 1.0) - last_p
    f = (t + 1.0 - last_p) / span
    return last_c + (first_c - last_c) * f


# Module-level LUT memo. The baked LUT is a PURE function of (stops, reverse,
# mirror): every other coloring knob (gamma / phase / n_cycles / log_premap /
# transform) is applied to the LUT's *output* downstream in `render_candidate`,
# never to the bake — so one bake per distinct (stops, reverse, mirror) suffices for
# the whole process. Shared across every PaletteLibrary instance and every
# `render_candidate` loop (beam, bootstrap, query gen), which is why it lives at
# module scope rather than on the library. Keyed on the stops' CONTENT (not a palette
# name) so it stays correct if two colormap files ever carry the same name with
# different stops. Bounded by the pool's distinct (palette, reverse) count (~hundreds);
# LUTs are tiny; no eviction. A redundant concurrent bake under threads is harmless
# (identical result), so no lock. The cached array is never mutated downstream
# (`lookup_linear` reads it; interior fill / downsample touch other buffers), so
# returning the shared object is safe — same contract the per-instance cache relied on.
_LUT_MEMO = {}


def _stops_key(stops):
    """Hashable, exact content signature of `stops` (list of (pos, rgb)) for the LUT
    memo. Values kept verbatim (no rounding) so the key is a faithful identity of the
    bake input; list rgb -> tuple so it is hashable."""
    return tuple((float(p), tuple(float(v) for v in rgb)) for p, rgb in stops)


def build_lut(stops, reverse=False, mirror=False):
    """Baked (LUT_SIZE, 3) linear-RGB LUT for `stops`, memoized on (stops-content,
    reverse, mirror). Pure memoization: byte-identical to an uncached bake. See
    `_LUT_MEMO` above for why (stops, reverse, mirror) is the complete key."""
    key = (_stops_key(stops), bool(reverse), bool(mirror))
    cached = _LUT_MEMO.get(key)
    if cached is not None:
        return cached
    lut = _bake_lut(stops, reverse=reverse, mirror=mirror)
    _LUT_MEMO[key] = lut
    return lut


def _bake_lut(stops, reverse=False, mirror=False):
    """Bake sRGB8 stops -> (LUT_SIZE, 3) linear-RGB LUT (Rust `from_srgb8_stops_mirrored`).

    `stops`: list of (pos, [r,g,b]) 8-bit. `mirror` pre-reflects (sequential seam
    fix); `reverse` flips direction about t=0 keeping the seam continuous.
    """
    if mirror:
        stops = mirror_stops(stops)
    # Normalize positions into [0,1), stable-sort (Python sort is stable).
    norm = [(p % 1.0, np.asarray(rgb, dtype=np.float64)) for p, rgb in stops]
    norm.sort(key=lambda x: x[0])
    pos = np.array([p for p, _ in norm], dtype=np.float64)
    labs = np.array([srgb_to_oklab(rgb / 255.0) for _, rgb in norm], dtype=np.float64)

    # Vectorized cyclic OKLab interpolation over the whole LUT grid — byte-identical to
    # the scalar `_interp_oklab_cyclic` per-entry loop it replaces (this was the pref-pick's
    # 96%-of-cost hot spot: a Python 4096-iteration loop, cold per distinct palette). Append
    # the wrap stop (pos[0]+1, lab[0]) so the last cyclic segment [pos[-1], pos[0]+1) is a
    # plain interval; t below pos[0] shifts by +1 into it (the scalar wrap branch). Each
    # t=i/LUT_SIZE then lerps in OKLab within its segment [epos[k], epos[k+1]).
    n = len(pos)
    t = np.arange(LUT_SIZE, dtype=np.float64) / LUT_SIZE
    epos = np.concatenate([pos, [pos[0] + 1.0]])          # (n+1,)
    elab = np.concatenate([labs, labs[:1]], axis=0)       # (n+1,3)
    tt = np.where(t < pos[0], t + 1.0, t)
    k = np.searchsorted(epos, tt, side="right") - 1       # segment index
    k = np.clip(k, 0, n - 1)
    a, b = epos[k], epos[k + 1]
    f = ((tt - a) / (b - a))[:, None]
    lab = elab[k] + (elab[k + 1] - elab[k]) * f
    lut = oklab_to_linear_srgb(lab)

    if reverse:
        # new[i] = old[(N - i) mod N] (seam fixed point at i=0).
        idx = (LUT_SIZE - np.arange(LUT_SIZE)) % LUT_SIZE
        lut = lut[idx]
    return lut


def lookup_linear(lut, t):
    """Cyclic LUT sample at t (array), linear RGB — Rust `lookup_linear` (index+lerp)."""
    t = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    x = t * LUT_SIZE
    i0 = np.floor(x).astype(np.int64)
    f = (x - i0)[..., None]
    i0 = i0 % LUT_SIZE
    i1 = (i0 + 1) % LUT_SIZE
    return lut[i0] * (1.0 - f) + lut[i1] * f


# ---------------------------------------------------------------------------
# Normalization / transform — mirrors src/render_modes.rs colorize stage.
# ---------------------------------------------------------------------------

PCT_LO = 0.5
PCT_HI = 99.5

# Gradient-weighted transfer (structure-aware field->palette-index remap). A field-only
# alternative to the plain percentile-stretch: it concentrates palette movement where the
# field has high spatial gradient (edges), so palette transitions align with geometric
# transitions instead of arbitrary isovalues. Reduces to the pct-stretch behavior EXACTLY
# at transfer_gamma=0. See `gradient_transfer_profile` / `_apply_transfer`.
N_TRANSFER_BINS = 200
TRANSFER_EPS = 0.02


def percentile_nearest(sorted_vals, p):
    """p-th percentile (p in [0,100]) via Rust nearest-rank: idx=round((p/100)*(n-1)).

    `sorted_vals` must be ascending. Rust uses round-half-away-from-zero; for a
    non-negative index that is floor(x+0.5)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = int(math.floor((p / 100.0) * (n - 1) + 0.5))
    idx = min(idx, n - 1)
    return float(sorted_vals[idx])


def percentile_stretch(field):
    """Raw field (NaN interior) -> (lo, span) over non-NaN finite values (Rust PCT)."""
    valid = field[np.isfinite(field)]
    if valid.size == 0:
        return 0.0, 1.0
    sv = np.sort(valid)
    lo = percentile_nearest(sv, PCT_LO)
    hi = percentile_nearest(sv, PCT_HI)
    span = (hi - lo) if hi > lo else 1.0
    return lo, span


def apply_transform(x, log_premap, gamma):
    """[0,1] value -> transformed [0,1]. `log_premap` in {'none','log'} then gamma.

    Matches Rust `apply_transform`: the curve, then `.clamp(0,1).powf(gamma)`.
    'none' is the Rust `linear` transform; 'log' is `ln(1+x)/ln2` ([0,1]->[0,1])."""
    x = np.asarray(x, dtype=np.float64)
    if log_premap == "none":
        y = x
    elif log_premap == "log":
        y = np.log1p(np.maximum(x, 0.0)) / math.log(2.0)
    else:
        raise ValueError(f"unknown log_premap '{log_premap}' (want none|log)")
    y = np.clip(y, 0.0, 1.0)
    return y ** gamma if gamma != 1.0 else y


# ---------------------------------------------------------------------------
# Downsample — mirrors src/render.rs downsample_linear_filtered.
# ---------------------------------------------------------------------------

def _filter_radius(name):
    return {"box": 0.5, "mitchell": 2.0, "lanczos3": 3.0}[name]


def _filter_eval(name, t):
    """1-D kernel at |t| destination-pixel units (Rust `DownsampleFilter::eval`)."""
    x = abs(t)
    if name == "box":
        return 1.0 if x < 0.5 else 0.0
    if name == "mitchell":
        B = C = 1.0 / 3.0
        if x < 1.0:
            return ((12 - 9 * B - 6 * C) * x**3 + (-18 + 12 * B + 6 * C) * x**2 + (6 - 2 * B)) / 6.0
        if x < 2.0:
            return ((-B - 6 * C) * x**3 + (6 * B + 30 * C) * x**2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6.0
        return 0.0
    if name == "lanczos3":
        if x < 1e-12:
            return 1.0
        if x < 3.0:
            pix = math.pi * x
            pix3 = pix / 3.0
            return (math.sin(pix) / pix) * (math.sin(pix3) / pix3)
        return 0.0
    raise ValueError(f"unknown filter '{name}'")


def _build_banded_taps(dst_len, src_len, ss, name):
    """Per-output kernel taps for a src_len->dst_len minification (Rust `build_taps`),
    in **banded** form: `(starts, weights)` where `weights` is (dst_len, K) padded and
    `starts[d]` is the first source index. Column `k` reads `src[starts[d]+k]` with
    weight `weights[d,k]` (0 past the edge). Rows sum to 1. K = max support width."""
    ssf = float(ss)
    r = _filter_radius(name) * ssf
    starts = np.zeros(dst_len, dtype=np.int64)
    rows = []
    K = 0
    for d in range(dst_len):
        center = (d + 0.5) * ssf
        lo = math.floor(center - r)
        hi = math.ceil(center + r)
        first = None
        ws = []
        s = 0.0
        for sx in range(lo, hi + 1):
            if sx < 0 or sx >= src_len:
                continue
            if first is None:
                first = sx
            wt = _filter_eval(name, (sx + 0.5 - center) / ssf)
            ws.append(wt)
            s += wt
        if s != 0.0:
            ws = [w / s for w in ws]
        starts[d] = first if first is not None else 0
        rows.append(ws)
        K = max(K, len(ws))
    weights = np.zeros((dst_len, K), dtype=np.float64)
    for d, ws in enumerate(rows):
        weights[d, : len(ws)] = ws
    return starts, weights


def _banded_pass(src, starts, weights):
    """Apply a 1-D banded filter along axis 1 of `src` (N, src_len, 3) -> (N, dst_len, 3)."""
    N, src_len, C = src.shape
    dst_len, K = weights.shape
    out = np.zeros((N, dst_len, C), dtype=np.float64)
    for k in range(K):
        # start+k can run past the edge on short (padded) tap rows; the weight there
        # is 0, so clip the index into range and let the zero weight nullify it.
        cols = np.clip(starts + k, 0, src_len - 1)
        contrib = src[:, cols, :]               # (N, dst_len, 3)
        out += contrib * weights[:, k][None, :, None]
    return out


def _encode_srgb8(linear):
    """linear (...,3) -> uint8 sRGB, Rust `(linear_to_srgb(v)*255+0.5) as u8` (truncate)."""
    v = linear_to_srgb(linear) * 255.0 + 0.5
    return np.floor(v).clip(0, 255).astype(np.uint8)


def downsample(linear, ss, name):
    """(H_sub, W_sub, 3) linear RGB -> (H_out, W_out, 3) uint8 sRGB. Linear-light.

    Box is the flat ss×ss average (Rust byte-identical path); mitchell/lanczos3 run
    two separable banded passes with an f32 intermediate (matching Rust's f32 store)."""
    Hs, Ws, _ = linear.shape
    out_h, out_w = Hs // ss, Ws // ss
    if name == "box":
        r = linear[: out_h * ss, : out_w * ss].reshape(out_h, ss, out_w, ss, 3).mean(axis=(1, 3))
        return _encode_srgb8(r)

    hstart, hw = _build_banded_taps(out_w, Ws, ss, name)
    vstart, vw = _build_banded_taps(out_h, Hs, ss, name)
    # Horizontal pass -> (Hs, out_w, 3), stored as f32 (Rust intermediate is f32).
    inter = _banded_pass(linear, hstart, hw).astype(np.float32).astype(np.float64)
    # Vertical pass: filter along rows -> transpose to put height on axis 1.
    out = _banded_pass(np.transpose(inter, (1, 0, 2)), vstart, vw)  # (out_w, out_h, 3)
    return _encode_srgb8(np.transpose(out, (1, 0, 2)))


# ---------------------------------------------------------------------------
# Palette library — cache LUTs + types from the two data files.
# ---------------------------------------------------------------------------

DEFAULT_COLORMAPS = "data/palettes/score3_colormaps.json"
DEFAULT_FEATURES = "data/palettes/palette_features.json"


class PaletteLibrary:
    """Loads score3_colormaps.json (stops + mirror_needed) and palette_features.json
    (type). Bakes/caches a LUT per (name, reverse, mirror).

    TWO cyclic-ness fields, DIFFERENT jobs — do not use one for the other's decision:

        field          file                    binary values          governs
        -----------    --------------------    -------------------    ---------------------
        `type`         palette_features.json   {cyclic, non_cyclic}   coloring knobs:
                                                                       phase / n_cycles apply
                                                                       ONLY to `type==cyclic`
                                                                       (`validate_config` raises
                                                                       otherwise). = `palette_type()`.
        `cycle`        *_colormaps.json        {cyclic, sequential}   the mirror seam-fix:
        (-> `mirror_needed`)                                          sequential maps bake
                                                                       pre-mirrored to de-seam;
                                                                       cyclic maps do not. = `lut()`.

    They agree for most palettes but are NOT interchangeable: a few maps are
    `type==non_cyclic` yet `cycle==cyclic` (get no cyclic knobs, no mirror). The one
    render-relevant invariant — enforced at pool-build time (build_pool.py) — is that
    NO palette is `type==cyclic` while `cycle==sequential`: a genuinely-cyclic palette
    handed n_cycles/phase must never also be pre-mirrored (that would halve+reflect its
    intended cycle). Deciding a knob from `cycle`, or the mirror from `type`, is the bug
    this table exists to prevent."""

    def __init__(self, colormaps_path=DEFAULT_COLORMAPS, features_path=DEFAULT_FEATURES):
        cms = json.loads(Path(colormaps_path).read_text())
        self.colormaps = {c["name"]: c for c in cms}
        feats = json.loads(Path(features_path).read_text())
        self.types = {name: v["type"] for name, v in feats.items()}
        self._lut_cache = {}

    def palette_type(self, name):
        """'cyclic' | 'non_cyclic'. Falls back to the colormap's `cycle` field
        (mapped into the binary space) when a palette is absent from the features
        file: declared cyclic -> cyclic, everything else -> non_cyclic."""
        if name in self.types:
            return self.types[name]
        cm = self.colormaps.get(name)
        if cm is None:
            raise KeyError(f"palette '{name}' not in colormaps or features")
        return "cyclic" if cm.get("cycle") == "cyclic" else "non_cyclic"

    def lut(self, name, reverse=False):
        """Baked linear-RGB LUT for `name`, mirror per the colormap's `mirror_needed`
        (matching the Rust render load through `from_srgb8_stops_mirrored`)."""
        cm = self.colormaps.get(name)
        if cm is None:
            raise KeyError(f"palette '{name}' not in {DEFAULT_COLORMAPS}")
        mirror = bool(cm.get("mirror_needed", False))
        key = (name, reverse, mirror)
        if key not in self._lut_cache:
            stops = [(p, rgb) for p, rgb in cm["stops"]]
            self._lut_cache[key] = build_lut(stops, reverse=reverse, mirror=mirror)
        return self._lut_cache[key]


# ---------------------------------------------------------------------------
# CandidateConfig — the durable recipe.
# ---------------------------------------------------------------------------

@dataclass
class LocationRef:
    """Version-invariant location reference (render keys). Coords are decimal strings."""
    kind: str          # 'mandelbrot' | 'julia'
    cx: str
    cy: str
    fw: str
    maxiter: int
    c_re: Optional[str] = None
    c_im: Optional[str] = None


@dataclass
class CandidateConfig:
    """Every coloring param + location + eval size. JSON-serializable; this IS the
    per-label recipe, replayable at any resolution.

    Coloring params:
      palette       colormap name (looked up in the PaletteLibrary)
      reverse       flip the LUT
      log_premap    'none' | 'log'      pre-map before gamma
      gamma         power u = t**gamma
      phase         cyclic-only: t -> (t+phase) mod 1
      n_cycles      cyclic-only, positive int: t -> (t*n_cycles) mod 1
      interior_color linear-RGB fill for NaN pixels (default black, = Rust)
      filter        'box' | 'mitchell' | 'lanczos3'
      transfer      'pct' | 'grad'   value->index map. 'pct' (default) is the plain
                    percentile-stretch (BIT-IDENTICAL to the pre-transfer path);
                    'grad' is the gradient-weighted transfer at `transfer_gamma`.
      transfer_gamma  grad only: gradient-weight exponent. 0 == pct (the family's edge).
    """
    palette: str
    location: LocationRef
    eval_width: int
    eval_height: int
    reverse: bool = False
    log_premap: str = "none"
    gamma: float = 1.0
    phase: float = 0.0
    n_cycles: int = 1
    interior_color: tuple = (0.0, 0.0, 0.0)
    filter: str = "box"
    transfer: str = "pct"
    transfer_gamma: float = 0.0

    def to_json(self):
        d = asdict(self)
        d["interior_color"] = list(self.interior_color)
        return json.dumps(d, sort_keys=True)

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        loc = LocationRef(**d.pop("location"))
        ic = d.get("interior_color", [0.0, 0.0, 0.0])
        d["interior_color"] = tuple(ic)
        return CandidateConfig(location=loc, **d)


# ---------------------------------------------------------------------------
# Field loading.
# ---------------------------------------------------------------------------

@dataclass
class FieldData:
    """A dumped super-res smooth field (NaN interior) + its sidecar metadata."""
    values: np.ndarray   # (height, width) float32/float64, NaN interior
    supersample: int
    location: LocationRef
    bailout_b: Optional[float] = None

    @property
    def out_size(self):
        """(out_w, out_h) after downsample by ss."""
        h, w = self.values.shape
        return w // self.supersample, h // self.supersample


def load_field(bin_path, json_path=None):
    """Load a `--dump-field` binary + sidecar into a FieldData."""
    bin_path = Path(bin_path)
    if json_path is None:
        json_path = bin_path.with_suffix(".json") if bin_path.suffix == ".bin" else Path(str(bin_path) + ".json")
    meta = json.loads(Path(json_path).read_text())
    w, h = int(meta["width"]), int(meta["height"])
    assert meta["dtype"] == "f32" and meta["layout"] == "row_major", meta
    raw = np.frombuffer(bin_path.read_bytes(), dtype="<f4")
    assert raw.size == w * h, f"field size {raw.size} != {w}*{h}"
    values = raw.reshape(h, w).astype(np.float64)
    loc = meta["location"]
    return FieldData(
        values=values,
        supersample=int(meta["supersample"]),
        location=LocationRef(
            kind=loc["kind"], cx=loc["cx"], cy=loc["cy"], fw=loc["fw"],
            maxiter=int(loc["maxiter"]), c_re=loc.get("c_re"), c_im=loc.get("c_im"),
        ),
        bailout_b=meta.get("bailout_b"),
    )


# ---------------------------------------------------------------------------
# Type dispatch (Part 4).
# ---------------------------------------------------------------------------

def validate_config(config, library):
    """Reject a config whose params don't apply to its palette type. Type is binary
    {cyclic, non_cyclic}: phase/n_cycles apply only to cyclic (raises ValueError
    otherwise). 'Applies' = a non-default value is set."""
    ptype = library.palette_type(config.palette)
    if (config.phase != 0.0 or config.n_cycles != 1) and ptype != "cyclic":
        raise ValueError(
            f"phase/n_cycles apply only to cyclic palettes; '{config.palette}' is {ptype}"
        )
    if not isinstance(config.n_cycles, int) or config.n_cycles < 1:
        raise ValueError(f"n_cycles must be a positive integer, got {config.n_cycles}")
    if config.log_premap not in ("none", "log"):
        raise ValueError(f"log_premap must be none|log, got {config.log_premap}")
    if config.filter not in ("box", "mitchell", "lanczos3"):
        raise ValueError(f"filter must be box|mitchell|lanczos3, got {config.filter}")
    if config.transfer not in ("pct", "grad"):
        raise ValueError(f"transfer must be pct|grad, got {config.transfer}")
    if config.transfer_gamma < 0.0:
        raise ValueError(f"transfer_gamma must be >= 0, got {config.transfer_gamma}")


# ---------------------------------------------------------------------------
# The one shared entry point.
# ---------------------------------------------------------------------------

@dataclass
class StretchedField:
    """The config-independent prefix of the coloring tail: the percentile-stretched
    field `x` in [0,1] (invalid -> 0) plus the interior `valid` mask. Depends only on
    the raw field, so it is computed ONCE per dumped field and reused across every
    recolor — the cache seam the inference sweep (thousands of recolors per field)
    lives on. `render_candidate` builds it lazily when not supplied."""
    x: np.ndarray
    valid: np.ndarray


def stretch_field(field):
    """(FieldData) -> StretchedField. Percentile-stretch on the RAW field (Rust PCT)."""
    raw = field.values
    valid = np.isfinite(raw)
    lo, span = percentile_stretch(raw)
    x = np.zeros_like(raw)
    x[valid] = np.clip((raw[valid] - lo) / span, 0.0, 1.0)
    return StretchedField(x=x, valid=valid)


# ---------------------------------------------------------------------------
# Gradient-weighted transfer — a structure-aware value->palette-index remap.
#
# The plain pct-stretch spends palette arc by field VALUE, so transitions land on
# arbitrary isovalues. This transfer spends arc by spatial GRADIENT, so transitions
# align with geometric edges. It is a PURE function of the raw field (no palette / gamma
# dependence) up to the `w(v)` profile, which is therefore computed ONCE per field and
# reused across every recolor AND every transfer_gamma — the same once-per-field cache
# seam as `StretchedField`. The per-gamma curve `g` is derived cheaply from the cached
# `w`. transfer_gamma=0 => weight≡1 => g linear => bit-identical to the pct-stretch.
# ---------------------------------------------------------------------------

@dataclass
class GradientTransferProfile:
    """Field-only gradient-weight profile `w(v)` for the gradient transfer.

    `w[b]` = mean spatial |∇field| over EXTERIOR pixels whose percentile-stretched value
    falls in value-bin `b` (N_TRANSFER_BINS bins over the pct-stretch's [lo, lo+span]
    anchors), normalized to max 1. Independent of palette / gamma / n_cycles /
    transfer_gamma, so compute it ONCE per dumped field and reuse."""
    w: np.ndarray   # (N_TRANSFER_BINS,) float64 in [0,1]


def gradient_transfer_profile(field, prep):
    """(FieldData, StretchedField) -> GradientTransferProfile. Field-only; cache it.

    Steps 1-3 of the transfer: |∇field| by forward finite differences over exterior
    pixels (interior NaN excluded; nan-safe — a partial toward an interior / off-grid
    neighbor drops to 0), binned by the stretched value into N_TRANSFER_BINS bins, mean
    |∇field| per bin, normalized to max 1. Reuses the same [lo,span] anchors as
    `prep` (via `prep.x`, already the clipped stretched value in [0,1])."""
    raw = field.values
    gx = np.zeros_like(raw)
    gy = np.zeros_like(raw)
    fx = np.zeros(raw.shape, dtype=bool)     # x-partial available (both endpoints finite)
    fy = np.zeros(raw.shape, dtype=bool)
    dx = raw[:, 1:] - raw[:, :-1]
    dy = raw[1:, :] - raw[:-1, :]
    mx = np.isfinite(dx)
    my = np.isfinite(dy)
    gx[:, :-1] = np.where(mx, dx, 0.0)
    gy[:-1, :] = np.where(my, dy, 0.0)
    fx[:, :-1] = mx
    fy[:-1, :] = my
    gm = np.sqrt(gx * gx + gy * gy)
    inc = prep.valid & (fx | fy)             # exterior pixel with >=1 finite partial
    b = np.clip((prep.x[inc] * N_TRANSFER_BINS).astype(np.int64), 0, N_TRANSFER_BINS - 1)
    counts = np.bincount(b, minlength=N_TRANSFER_BINS).astype(np.float64)
    sums = np.bincount(b, weights=gm[inc], minlength=N_TRANSFER_BINS)
    w = np.zeros(N_TRANSFER_BINS, dtype=np.float64)
    nz = counts > 0.0
    w[nz] = sums[nz] / counts[nz]
    mxw = w.max()
    if mxw > 0.0:
        w = w / mxw
    return GradientTransferProfile(w=w)


def _apply_transfer(x, profile, gamma):
    """Remap stretched value `x`∈[0,1] through the gradient-weighted CDF at `gamma`.

    `weight = (w+eps)**gamma`; `g = cumsum(weight)` normalized to [0,1] at the N+1 bin
    edges; `base = interp(g at x)`. gamma=0 => weight≡1 => g linear => returns `x`
    bit-identically (the pct-stretch edge of the family)."""
    w = profile.w
    n = w.shape[0]
    weight = (w + TRANSFER_EPS) ** gamma
    csum = np.concatenate(([0.0], np.cumsum(weight)))     # (n+1,) at bin edges
    total = csum[-1]
    g = csum / total if total > 0.0 else np.linspace(0.0, 1.0, n + 1)
    edges = np.arange(n + 1, dtype=np.float64) / n        # value-axis edges in [0,1]
    return np.interp(x, edges, g)


def render_candidate(field, config, library, prep=None, profile=None):
    """(FieldData, CandidateConfig, PaletteLibrary) -> (H_out, W_out, 3) uint8 sRGB.

    The full coloring tail, in the Rust-pinned order. `field.values` is the raw
    super-res smooth field with NaN interior; never recomputes field math. `prep`
    (a `StretchedField` from `stretch_field`) skips the config-independent
    percentile-stretch prefix — pass it to recolor a cached field cheaply; when None
    it is computed here, so the single-call contract is unchanged. `profile` (a
    `GradientTransferProfile`) is the field-only gradient-weight profile, needed ONLY
    for `transfer='grad'`; pass the cached one to avoid recomputing it per candidate
    (it is built lazily here when None)."""
    validate_config(config, library)
    if prep is None:
        prep = stretch_field(field)
    x, valid = prep.x, prep.valid

    # 1. percentile-stretch on the RAW field -> x in [0,1] (done in `prep`).

    # 1b. optional gradient-weighted transfer: remap the stretched value so palette arc
    #     concentrates on high-gradient (edge) isovalues. 'pct' (default) SKIPS this ->
    #     bit-identical to the pre-transfer path; 'grad' at transfer_gamma=0 is the same
    #     identity edge. Field-only; interior pixels keep x=0 (overwritten below anyway).
    if config.transfer == "grad":
        if profile is None:
            profile = gradient_transfer_profile(field, prep)
        base = _apply_transfer(x, profile, config.transfer_gamma)
    else:
        base = x

    # 2. transform curve + gamma.
    gray = apply_transform(base, config.log_premap, config.gamma)

    # 3. LUT stage: n_cycles, phase (Rust: gray*cycles+offset). Cyclic-only knobs;
    #    non_cyclic palettes leave gray untouched (n_cycles=1, phase=0).
    t = gray
    t = np.mod(t * config.n_cycles, 1.0)
    t = np.mod(t + config.phase, 1.0)
    lut = library.lut(config.palette, reverse=config.reverse)
    linear = lookup_linear(lut, t)          # (H_sub, W_sub, 3) linear RGB

    # 4. interior fill: NaN pixels -> interior_color (linear).
    linear[~valid] = np.asarray(config.interior_color, dtype=np.float64)

    # 5. linear-light downsample -> sRGB8.
    return downsample(linear, field.supersample, config.filter)


# ===========================================================================
# COARSE SCORING-RECOLOR PATH — SCORING-ONLY. DO NOT USE FOR KEEPERS.
#
# This is a *distinct* coloring path used ONLY to feed the beam's throwaway pref
# scoring images (sample_location.run_location, coarse_score=True). It colors a
# small field pre-downsampled to ~the scorer's input geometry, skipping the ss2
# LUT gather (36% of recolor) AND the ss2->eval AA downsample (29%) — the two hot
# stages — at the cost of the ss2 anti-aliasing (acceptable: the image is a
# throwaway scorer input, never a keeper).
#
# CORRECTNESS FENCE: `render_candidate` / `render_corpus_crop` / the label-crop and
# wallpaper emitters are the ONLY keeper paths and MUST stay on the full-res tail.
# `coarse_field` / `render_candidate_coarse` are reachable ONLY from the scoring
# loop. Do NOT call them from any render that becomes a stored crop. The two paths
# differ (average-field-then-color vs color-then-average-field), so crossing this
# boundary silently degrades real output.
#
# Faithfulness to the full path: the coarse image is emitted at the SAME 16:9 eval
# aspect the scorer always consumes, so the scorer's own bicubic squash-to-224 is
# byte-identical between the two paths — the ONLY thing that changed is the coloring
# resolution and the (field-space, not color-space) area average.
# ===========================================================================

# Coarse scoring grid (16:9, a small margin above the scorer's 224 input so its
# bicubic squash still has real content to filter). Colored per candidate; the
# scorer resizes it to 224x224. Validated for ranking-parity against the full path
# (see tools/queries/validate_coarse_score.py).
SCORE_COARSE_W = 512
SCORE_COARSE_H = 288

# Module-level area-resample matrix memo. The (dst_len, src_len) box/area matrix is a
# pure function of the two lengths (both constant across every location — src is always
# EVAL*SS, dst is the fixed coarse grid), so one build per (dst,src) serves the process.
_AREA_MAT_MEMO = {}


def _area_matrix(dst_len, src_len):
    """Dense (dst_len, src_len) box/area resample matrix for a 1-D src->dst resize (any
    ratio). Dest pixel d integrates the source over [d*s,(d+1)*s), s=src/dst, weighted by
    overlap length; rows are normalized to sum 1. Memoized on (dst_len, src_len)."""
    key = (dst_len, src_len)
    cached = _AREA_MAT_MEMO.get(key)
    if cached is not None:
        return cached
    s = src_len / dst_len
    M = np.zeros((dst_len, src_len), dtype=np.float64)
    for d in range(dst_len):
        a, b = d * s, (d + 1) * s
        i0 = int(math.floor(a))
        i1 = int(math.ceil(b))
        for sx in range(i0, min(i1, src_len)):
            lo = max(a, float(sx))
            hi = min(b, float(sx + 1))
            if hi > lo:
                M[d, sx] = hi - lo
        rs = M[d].sum()
        if rs > 0.0:
            M[d] /= rs
    _AREA_MAT_MEMO[key] = M
    return M


def _area_downsample(a, out_h, out_w):
    """Separable box/area resize of a 2-D array (H,W) -> (out_h,out_w) via the cached
    row/col area matrices: out = Mv @ a @ Mh.T (each an area-weighted mean)."""
    H, W = a.shape
    Mv = _area_matrix(out_h, H)
    Mh = _area_matrix(out_w, W)
    return Mv @ a @ Mh.T


@dataclass
class CoarseField:
    """SCORING-ONLY location-invariant, colormap-independent coarse prefix (analogue of
    `StretchedField`, cached once per location). Built by area-downsampling the ss2
    stretched field to the coarse scoring grid:

      xmean  (h,w)  exterior area-mean of the stretched value in [0,1] (0 where fully
                    interior); the input to the per-candidate transform+LUT.
      vfrac  (h,w)  exterior area-fraction in [0,1] — the interior-boundary blend weight
                    (a coarse pixel straddling the interior gets its color scaled toward
                    interior_color by 1-vfrac, mirroring the linear-light boundary blend).
    """
    xmean: np.ndarray
    vfrac: np.ndarray


def coarse_field(prep, out_w=SCORE_COARSE_W, out_h=SCORE_COARSE_H):
    """(StretchedField) -> CoarseField at the coarse scoring grid. SCORING-ONLY.

    Area-downsamples the ss2 stretched field. `prep.x` is 0 at interior pixels, so its
    area-mean is the interior-as-0 mean; dividing by the exterior fraction `vfrac`
    recovers the exterior-only mean `xmean`. Location-invariant / colormap-independent —
    cache it like `prep` and reuse across all of a location's candidates."""
    v = prep.valid.astype(np.float64)
    vfrac = _area_downsample(v, out_h, out_w)          # exterior area-fraction
    xsum = _area_downsample(prep.x, out_h, out_w)      # interior-as-0 area-mean
    xmean = np.zeros_like(xsum)
    m = vfrac > 1e-9
    xmean[m] = np.clip(xsum[m] / vfrac[m], 0.0, 1.0)
    return CoarseField(xmean=xmean, vfrac=vfrac)


def render_candidate_coarse(coarse, config, library, profile=None):
    """(CoarseField, CandidateConfig, PaletteLibrary) -> (h,w,3) uint8 sRGB. SCORING-ONLY.

    The coloring tail (transform+gamma -> n_cycles/phase -> OKLab LUT -> interior blend)
    run on the small pre-downsampled field, at its native resolution — NO supersample,
    NO AA downsample. Numerically identical to `render_candidate`'s per-pixel color math;
    it differs from the keeper path ONLY in that the area-average happened on the scalar
    field (before color) instead of on the colors (after). MUST NOT feed a stored crop —
    see the fence above.

    For `transfer='grad'` a `profile` (GradientTransferProfile, built once per location
    from the full-res field/prep) is REQUIRED — there is no field here to derive it. The
    transfer is applied to the area-mean stretched value; like the coarse path itself
    that is an average-then-map approximation of the keeper path, acceptable for scoring."""
    validate_config(config, library)
    xin = coarse.xmean
    if config.transfer == "grad":
        if profile is None:
            raise ValueError("render_candidate_coarse needs a GradientTransferProfile for transfer='grad'")
        xin = _apply_transfer(coarse.xmean, profile, config.transfer_gamma)
    gray = apply_transform(xin, config.log_premap, config.gamma)
    t = np.mod(gray * config.n_cycles, 1.0)
    t = np.mod(t + config.phase, 1.0)
    lut = library.lut(config.palette, reverse=config.reverse)
    linear = lookup_linear(lut, t)                     # (h,w,3) linear RGB
    ic = np.asarray(config.interior_color, dtype=np.float64)
    vf = coarse.vfrac[..., None]                        # interior-boundary blend
    linear = linear * vf + ic[None, None, :] * (1.0 - vf)
    return _encode_srgb8(linear)


def render_candidates_coarse(coarse, configs, library, profiles=None):
    """Batched SCORING-ONLY analogue of `render_candidate_coarse`: color one CoarseField
    under K CandidateConfigs in a single numpy op. Returns a (K,h,w,3) uint8 array whose
    k-th slice is byte-identical to `render_candidate_coarse(coarse, configs[k], library,
    profiles[k])`. This is the vectorized pref-pick recolor — the ≤32 candidate colormaps of
    a location's cached coarse field are gathered through their stacked LUTs at once instead
    of a Python per-candidate loop. Same fence as `render_candidate_coarse`: SCORING-ONLY,
    never a stored crop. `profiles` (list aligned with `configs`, or None) supplies the
    per-config GradientTransferProfile required by any `transfer='grad'` config."""
    K = len(configs)
    if profiles is None:
        profiles = [None] * K
    h, w = coarse.xmean.shape
    luts = np.empty((K, LUT_SIZE, 3), dtype=np.float64)
    ts = np.empty((K, h, w), dtype=np.float64)
    ics = np.empty((K, 3), dtype=np.float64)
    for k, cfg in enumerate(configs):
        validate_config(cfg, library)
        xin = coarse.xmean
        if cfg.transfer == "grad":
            if profiles[k] is None:
                raise ValueError("render_candidates_coarse needs a GradientTransferProfile for transfer='grad'")
            xin = _apply_transfer(coarse.xmean, profiles[k], cfg.transfer_gamma)
        gray = apply_transform(xin, cfg.log_premap, cfg.gamma)
        t = np.mod(gray * cfg.n_cycles, 1.0)
        ts[k] = np.mod(t + cfg.phase, 1.0)
        luts[k] = library.lut(cfg.palette, reverse=cfg.reverse)
        ics[k] = np.asarray(cfg.interior_color, dtype=np.float64)

    # Batched cyclic LUT gather — `lookup_linear` over K stacked LUTs (same index+lerp
    # math). When every candidate shares one t-plane (the canonical pref-pick case: configs
    # differ only in palette), the index math is done ONCE on (h,w) and all K LUTs are
    # gathered by a single basic slice `luts[:, i0]` — ~2x over per-candidate advanced
    # indexing on a (K,h,w) index. Byte-identical to the general path either way.
    if np.array_equal(ts, np.broadcast_to(ts[0], ts.shape)):
        tm = np.mod(ts[0], 1.0)
        x = tm * LUT_SIZE
        i0 = np.floor(x).astype(np.int64)
        f = (x - i0)[..., None]
        i0 = i0 % LUT_SIZE
        i1 = (i0 + 1) % LUT_SIZE
        linear = luts[:, i0] * (1.0 - f) + luts[:, i1] * f     # (K,h,w,3)
    else:
        tm = np.mod(ts, 1.0)
        x = tm * LUT_SIZE
        i0 = np.floor(x).astype(np.int64)
        f = (x - i0)[..., None]
        i0 = i0 % LUT_SIZE
        i1 = (i0 + 1) % LUT_SIZE
        kidx = np.arange(K)[:, None, None]
        linear = luts[kidx, i0] * (1.0 - f) + luts[kidx, i1] * f    # (K,h,w,3)

    # Interior-boundary blend (per-config interior_color, shared vfrac).
    vf = coarse.vfrac[None, ..., None]                          # (1,h,w,1)
    linear = linear * vf + ics[:, None, None, :] * (1.0 - vf)
    return _encode_srgb8(linear)


if __name__ == "__main__":
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser(description="Color a dumped smooth field via a CandidateConfig.")
    ap.add_argument("field", help="path to the .bin field (sidecar .json alongside)")
    ap.add_argument("--palette", default="twilight")
    ap.add_argument("--out", default="out/colormap_render.png")
    ap.add_argument("--filter", default="box", choices=["box", "mitchell", "lanczos3"])
    ap.add_argument("--log-premap", default="none", choices=["none", "log"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--n-cycles", type=int, default=1)
    ap.add_argument("--transfer", default="pct", choices=["pct", "grad"])
    ap.add_argument("--transfer-gamma", type=float, default=0.0)
    ap.add_argument("--colormaps", default=DEFAULT_COLORMAPS)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    args = ap.parse_args()

    fld = load_field(args.field)
    lib = PaletteLibrary(args.colormaps, args.features)
    ow, oh = fld.out_size
    cfg = CandidateConfig(
        palette=args.palette, location=fld.location, eval_width=ow, eval_height=oh,
        reverse=args.reverse, log_premap=args.log_premap, gamma=args.gamma,
        phase=args.phase, n_cycles=args.n_cycles, filter=args.filter,
        transfer=args.transfer, transfer_gamma=args.transfer_gamma,
    )
    img = render_candidate(fld, cfg, lib)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]})")
