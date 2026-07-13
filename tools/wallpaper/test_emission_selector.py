#!/usr/bin/env python
"""Unit tests for the emission selector — the fractal-identity dedup (mirrors the
seeder's per-family near-dup) and the <=1/distinct-fractal selection guard.

  uv run pytest tools/wallpaper/test_emission_selector.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("emission_selector", HERE / "emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)


def C(family, cx=float("nan"), cy=float("nan"), fw=float("nan"), c_re=None, c_im=None,
      loc="", fit=1.0, cell=0, pal="p", iid=""):
    return es.Candidate(location_id=loc, palette_id=pal, family=family, fitness=fit,
                        color_cell=cell, image_id=iid or loc, cx=cx, cy=cy, fw=fw,
                        c_re=c_re, c_im=c_im)


# --------------------------------------------------------------------------- #
# same_fractal — per-family identity
# --------------------------------------------------------------------------- #
def test_cplane_same_center_diff_zoom_merges():
    # mirror production_seeder.near_dup: same center, decade-apart zoom -> same place.
    assert es.same_fractal(C("mandelbrot", 0.0, 0.0, 1e-3), C("mandelbrot", 1e-6, 0.0, 2.0))


def test_cplane_distant_centers_distinct():
    assert not es.same_fractal(C("mandelbrot", 5.0, 5.0, 1e-3), C("mandelbrot", 0.0, 0.0, 1e-3))


def test_cplane_within_radius_merges():
    # dist 0.1 < 1.5 * max(fw=0.1) = 0.15 -> merge; dist 0.2 > 0.15 -> distinct.
    assert es.same_fractal(C("mandelbrot", 0.1, 0.0, 0.1), C("mandelbrot", 0.0, 0.0, 0.1))
    assert not es.same_fractal(C("mandelbrot", 0.2, 0.0, 0.1), C("mandelbrot", 0.0, 0.0, 0.1))


def test_julia_requires_c_match():
    # identical base-scale (0,0) viewport, DIFFERENT seed c -> distinct fractals.
    a = C("julia", 0.0, 0.0, 3.0, c_re=-0.77, c_im=-0.13)
    b = C("julia", 0.0, 0.0, 3.0, c_re=-0.62, c_im=-0.40)
    assert not es.same_fractal(a, b)


def test_julia_same_c_base_recolors_merge():
    # same seed c, base (0,0) view, comparable zoom (recolor siblings) -> merge.
    a = C("julia", 0.0, 0.0, 0.14, c_re=-0.779, c_im=-0.134)
    b = C("julia", 0.0, 0.0, 0.18, c_re=-0.779, c_im=-0.134)
    assert es.same_fractal(a, b)


def test_julia_same_c_deep_zoom_distinct():
    # same seed c, same (0,0) center, but a 45x-deeper zoom is a genuinely-distinct view.
    a = C("julia", 0.0, 0.0, 1.61, c_re=-0.744, c_im=0.126)
    b = C("julia", 0.0, 0.0, 0.0356, c_re=-0.744, c_im=0.126)
    assert not es.same_fractal(a, b)


def test_julia_same_c_far_viewport_distinct():
    # same seed c, viewports far apart in the z-plane -> distinct sub-locations.
    a = C("julia_multibrot3", 0.716, 0.629, 0.02, c_re=0.525, c_im=-0.144)
    b = C("julia_multibrot3", 0.569, -0.067, 0.011, c_re=0.525, c_im=-0.144)
    assert not es.same_fractal(a, b)


def test_phoenix_recolor_siblings_merge():
    a = C("phoenix", -0.444237, 0.838584, 0.078)
    b = C("phoenix", -0.443955, 0.844047, 0.080)
    assert es.same_fractal(a, b)


def test_phoenix_decade_zoom_not_over_collapsed():
    # nearby centers but a ~1000x zoom gap -> NOT the same fractal (the phoenix carve-out).
    a = C("phoenix", -0.375556, 0.551152, 0.382)
    b = C("phoenix", -0.411817, 0.540521, 3.7e-4)
    assert not es.same_fractal(a, b)


def test_no_geometry_falls_back_to_exact_key():
    assert es.same_fractal(C("mandelbrot", loc="K"), C("mandelbrot", loc="K"))
    assert not es.same_fractal(C("mandelbrot", loc="K1"), C("mandelbrot", loc="K2"))


def test_different_family_never_same():
    a = C("julia", 0.0, 0.0, 3.0, c_re=0.1, c_im=0.1)
    b = C("julia_multibrot3", 0.0, 0.0, 3.0, c_re=0.1, c_im=0.1)
    assert not es.same_fractal(a, b)


# --------------------------------------------------------------------------- #
# select() — the <=1/distinct-fractal guard
# --------------------------------------------------------------------------- #
_NOCAP = dict(palette_cap_frac=1e9, palette_family_cap=None)


def test_select_drops_recolor_dups_keeps_best():
    # three recolors of ONE julia fractal, each in a different color cell.
    cands = [
        C("julia", 0.0, 0.0, 0.15, c_re=-0.779, c_im=-0.134, cell=2, fit=3.0, iid="hi"),
        C("julia", 0.0, 0.0, 0.14, c_re=-0.779, c_im=-0.134, cell=4, fit=2.5, iid="mid"),
        C("julia", 0.0, 0.0, 0.18, c_re=-0.779, c_im=-0.134, cell=8, fit=2.0, iid="lo"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert [c.image_id for c in res.picks] == ["hi"]          # only the best-fitness recolor
    assert res.report["n_dup_rejected"] == 2


def test_select_keeps_distinct_fractals():
    # two genuinely-distinct julia (different seed c) -> both kept.
    cands = [
        C("julia", 0.0, 0.0, 3.0, c_re=-0.779, c_im=-0.134, cell=2, fit=3.0, iid="a"),
        C("julia", 0.0, 0.0, 3.0, c_re=-0.62, c_im=-0.40, cell=4, fit=2.0, iid="b"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert sorted(c.image_id for c in res.picks) == ["a", "b"]
    assert res.report["n_dup_rejected"] == 0


def test_select_no_regression_without_geometry():
    # geometry-free candidates must select exactly as the historical exact-key guard did.
    cands = [
        C("mandelbrot", loc="L1", cell=1, fit=3.0, iid="a"),
        C("mandelbrot", loc="L1", cell=2, fit=2.0, iid="b"),   # same exact loc -> dropped
        C("mandelbrot", loc="L2", cell=3, fit=1.0, iid="c"),
    ]
    res = es.select(cands, grid=es.ColorGrid(), **_NOCAP)
    assert sorted(c.image_id for c in res.picks) == ["a", "c"]
    assert res.report["n_dup_rejected"] == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
