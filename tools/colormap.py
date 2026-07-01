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


def build_lut(stops, reverse=False, mirror=False):
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

    lut = np.empty((LUT_SIZE, 3), dtype=np.float64)
    for i in range(LUT_SIZE):
        t = i / LUT_SIZE
        lab = _interp_oklab_cyclic(pos, labs, t)
        lut[i] = oklab_to_linear_srgb(lab)

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
      n_cycles      cyclic-only, {1,2}: t -> (t*n_cycles) mod 1
      interior_color linear-RGB fill for NaN pixels (default black, = Rust)
      filter        'box' | 'mitchell' | 'lanczos3'
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
    if config.n_cycles not in (1, 2):
        raise ValueError(f"n_cycles must be 1 or 2, got {config.n_cycles}")
    if config.log_premap not in ("none", "log"):
        raise ValueError(f"log_premap must be none|log, got {config.log_premap}")
    if config.filter not in ("box", "mitchell", "lanczos3"):
        raise ValueError(f"filter must be box|mitchell|lanczos3, got {config.filter}")


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


def render_candidate(field, config, library, prep=None):
    """(FieldData, CandidateConfig, PaletteLibrary) -> (H_out, W_out, 3) uint8 sRGB.

    The full coloring tail, in the Rust-pinned order. `field.values` is the raw
    super-res smooth field with NaN interior; never recomputes field math. `prep`
    (a `StretchedField` from `stretch_field`) skips the config-independent
    percentile-stretch prefix — pass it to recolor a cached field cheaply; when None
    it is computed here, so the single-call contract is unchanged."""
    validate_config(config, library)
    if prep is None:
        prep = stretch_field(field)
    x, valid = prep.x, prep.valid

    # 1. percentile-stretch on the RAW field -> x in [0,1] (done in `prep`).

    # 2. transform curve + gamma.
    gray = apply_transform(x, config.log_premap, config.gamma)

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
    )
    img = render_candidate(fld, cfg, lib)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]})")
