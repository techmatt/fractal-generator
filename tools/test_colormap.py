"""Tests for the Python coloring tail (tools/colormap.py).

Run:  uv run python -m pytest tools/test_colormap.py -v

The reference-match test (the load-bearing one) shells out to the release binary; it
is skipped if the binary isn't built. The rest are pure-Python unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colormap as cm  # noqa: E402
import colormap_acceptance as acc  # noqa: E402


def _synthetic_field(h=48, w=64, ss=2, interior_frac=0.2, seed=0):
    """A deterministic super-res smooth field with a NaN interior blob."""
    rng = np.random.default_rng(seed)
    hs, ws = h * ss, w * ss
    vals = rng.uniform(3.0, 40.0, size=(hs, ws)).astype(np.float64)
    n_int = int(interior_frac * hs * ws)
    idx = rng.choice(hs * ws, size=n_int, replace=False)
    flat = vals.ravel()
    flat[idx] = np.nan
    loc = cm.LocationRef(kind="mandelbrot", cx="-0.75", cy="0.1", fw="0.01", maxiter=500)
    return cm.FieldData(values=flat.reshape(hs, ws), supersample=ss, location=loc)


@pytest.fixture(scope="module")
def library():
    return cm.PaletteLibrary()


# --------------------------------------------------------------------------- #
# Determinism — same (field, config) -> identical image.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("filt", ["box", "mitchell", "lanczos3"])
def test_determinism(library, filt):
    field = _synthetic_field()
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(palette="twilight", location=field.location,
                             eval_width=ow, eval_height=oh, filter=filt)
    a = cm.render_candidate(field, cfg, library)
    b = cm.render_candidate(field, cfg, library)
    assert np.array_equal(a, b)
    assert a.shape == (oh, ow, 3) and a.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Type dispatch — inapplicable params rejected on the wrong palette type.
# --------------------------------------------------------------------------- #

def test_type_dispatch_cyclic_only(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    base = dict(location=field.location, eval_width=ow, eval_height=oh)
    # phase/n_cycles on a non-cyclic palette (magma = sequential) -> reject.
    for kw in ({"phase": 0.3}, {"n_cycles": 2}):
        cfg = cm.CandidateConfig(palette="magma", **base, **kw)
        with pytest.raises(ValueError, match="cyclic"):
            cm.render_candidate(field, cfg, library)
    # Same params on a cyclic palette (twilight) -> allowed.
    cfg = cm.CandidateConfig(palette="twilight", **base, phase=0.3, n_cycles=2)
    cm.render_candidate(field, cfg, library)


def test_type_dispatch_domain_checks(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    base = dict(palette="twilight", location=field.location, eval_width=ow, eval_height=oh)
    # n_cycles=3 is a *valid* value on cyclic twilight (cf. test_type_dispatch_cyclic_only,
    # which allows n_cycles=2); the domain rule is "positive integer", so probe with 0.
    with pytest.raises(ValueError, match="n_cycles"):
        cm.render_candidate(field, cm.CandidateConfig(**base, n_cycles=0), library)
    with pytest.raises(ValueError, match="log_premap"):
        cm.render_candidate(field, cm.CandidateConfig(**base, log_premap="bogus"), library)
    with pytest.raises(ValueError, match="filter"):
        cm.render_candidate(field, cm.CandidateConfig(**base, filter="bogus"), library)


# --------------------------------------------------------------------------- #
# Recipe round-trip — CandidateConfig -> JSON -> back -> identical render.
# --------------------------------------------------------------------------- #

def test_recipe_roundtrip_json():
    loc = cm.LocationRef(kind="julia", cx="0.0", cy="0.0", fw="0.75", maxiter=800,
                         c_re="0.27", c_im="0.48")
    cfg = cm.CandidateConfig(palette="twilight", location=loc, eval_width=100, eval_height=60,
                             reverse=True, log_premap="log", gamma=1.5, phase=0.25,
                             n_cycles=2, interior_color=(0.1, 0.2, 0.3),
                             filter="lanczos3")
    back = cm.CandidateConfig.from_json(cfg.to_json())
    assert back == cfg


def test_recipe_roundtrip_render(library):
    field = _synthetic_field()
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(palette="twilight", location=field.location, eval_width=ow,
                             eval_height=oh, reverse=True, log_premap="log", gamma=1.3,
                             phase=0.2, n_cycles=2, filter="box")
    back = cm.CandidateConfig.from_json(cfg.to_json())
    a = cm.render_candidate(field, cfg, library)
    b = cm.render_candidate(field, back, library)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Interior fill — NaN pixels take the configured interior color.
# --------------------------------------------------------------------------- #

def test_interior_fill(library):
    # An all-interior field must render to a flat color = the interior fill.
    loc = cm.LocationRef(kind="mandelbrot", cx="0", cy="0", fw="0.01", maxiter=500)
    field = cm.FieldData(values=np.full((8, 8), np.nan), supersample=1, location=loc)
    cfg = cm.CandidateConfig(palette="twilight", location=loc, eval_width=8, eval_height=8,
                             interior_color=(1.0, 0.0, 0.0), filter="box")
    img = cm.render_candidate(field, cfg, library)
    # linear (1,0,0) -> sRGB (255,0,0).
    assert np.array_equal(img, np.tile([255, 0, 0], (8, 8, 1)))


# --------------------------------------------------------------------------- #
# LUT parity — a baked twilight LUT reproduces its control stops (OKLab interp).
# --------------------------------------------------------------------------- #

def test_lut_reproduces_stops(library):
    import json
    cms = {c["name"]: c for c in json.loads(Path("data/palettes/score3_colormaps.json").read_text())}
    stops = cms["twilight"]["stops"]
    lut = library.lut("twilight")
    for pos, rgb in stops[:8]:
        got = cm.lookup_linear(lut, np.array([pos % 1.0]))[0]
        want = cm.srgb_to_linear(np.asarray(rgb) / 255.0)
        assert np.max(np.abs(got - want)) < 5e-3, f"stop {pos}: {got} vs {want}"


# --------------------------------------------------------------------------- #
# LUT memo — the module-level cache is PURE: byte-identical to an uncached bake. #
# --------------------------------------------------------------------------- #

def test_lut_memo_byte_identical(library):
    """`build_lut` (memoized) must equal `_bake_lut` (uncached) exactly, and a render
    with the memo warm must equal one that re-bakes fresh — for varied palette + reverse
    + mirror. Any nonzero delta means the memo key is wrong (silent color corruption)."""
    import json
    cms = {c["name"]: c for c in json.loads(Path("data/palettes/score3_colormaps.json").read_text())}
    field = _synthetic_field()
    ow, oh = field.out_size
    # (a) LUT-level: memoized == fresh bake, both reverse states, both mirror states.
    for name in ("twilight", "magma", "viridis"):
        stops = [(p, rgb) for p, rgb in cms[name]["stops"]]
        for reverse in (False, True):
            for mirror in (False, True):
                cm._LUT_MEMO.clear()
                fresh = cm._bake_lut(stops, reverse=reverse, mirror=mirror)
                memo1 = cm.build_lut(stops, reverse=reverse, mirror=mirror)   # miss -> bake
                memo2 = cm.build_lut(stops, reverse=reverse, mirror=mirror)   # hit
                assert np.array_equal(fresh, memo1)
                assert memo1 is memo2                                         # cached object reused
    # (b) render-level: memo-warm == re-bake-every-render, across coloring knobs.
    configs = [
        dict(palette="twilight", reverse=False, gamma=1.0, phase=0.0, n_cycles=1),
        dict(palette="twilight", reverse=False, gamma=1.7, phase=0.35, n_cycles=2),
        dict(palette="magma", reverse=True, gamma=0.8, log_premap="log"),
        dict(palette="viridis", reverse=False, gamma=1.2),
    ]
    for kw in configs:
        cfg = cm.CandidateConfig(location=field.location, eval_width=ow, eval_height=oh, **kw)
        cm._LUT_MEMO.clear(); library._lut_cache.clear()
        warm = cm.render_candidate(field, cfg, library)                      # populates caches
        again = cm.render_candidate(field, cfg, library)                     # memo hit
        cm._LUT_MEMO.clear(); library._lut_cache.clear()
        fresh = cm.render_candidate(field, cfg, library)                     # cold re-bake
        assert np.array_equal(warm, again), kw
        assert np.array_equal(warm, fresh), kw


# --------------------------------------------------------------------------- #
# Reference-match — the headline gate (shells out to the release binary).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not acc.BIN.exists(), reason="release binary not built")
@pytest.mark.parametrize("test_id,filt", [
    ("test_01", "box"),        # mandelbrot
    ("test_01", "lanczos3"),   # mandelbrot, production filter
    ("test_03", "box"),        # julia
])
def test_reference_match(test_id, filt):
    m = acc.run_gate(test_id, filt=filt, width=480, height=270, ss=2)
    assert m["max_diff"] <= acc.TOL_MAX, m
    assert m["frac_gt1"] <= acc.TOL_FRAC_GT1, m
