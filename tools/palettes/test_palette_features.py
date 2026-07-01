"""Tests for the palette feature module.

Run either way:
  uv run pytest tools/palettes/test_palette_features.py
  uv run python tools/palettes/test_palette_features.py     # prints PASS/FAIL summary

Type-derivation checks are kept SOFT (report, not hard-assert) since the eps
thresholds are meant to be tuned by eye from build_features.py's printed distributions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color  # noqa: E402
import palette_features as pf  # noqa: E402


# ------------------------------------------------------------------ color ---------

def test_color_reference_points():
    L, a, b = color.srgb_to_oklab([1.0, 1.0, 1.0])
    assert abs(L - 1.0) < 1e-4 and abs(a) < 1e-4 and abs(b) < 1e-4, "white"
    assert np.allclose(color.srgb_to_oklab([0.0, 0.0, 0.0]), 0.0, atol=1e-6), "black"
    for g in (0.25, 0.5, 0.75):
        _, ga, gb = color.srgb_to_oklab([g, g, g])
        assert abs(ga) < 1e-6 and abs(gb) < 1e-6, "gray is neutral (a,b=0)"


def test_color_roundtrip():
    rng = np.random.default_rng(0)
    srgb = rng.random((512, 3))
    back = color.oklab_to_srgb(color.srgb_to_oklab(srgb))
    assert np.abs(back - srgb).max() < 1e-6, "sRGB->Oklab->sRGB roundtrip"


# ------------------------------------------------------------- reverse-invariance -

def _reversed_stops(stops):
    """Explicit LUT reverse: color'(t) = color(1-t). Reflect each stop's t about 0.5,
    re-sort ascending -- keeps the pipeline's clamped interp exactly symmetric."""
    rs = [[1.0 - t, rgb] for t, rgb in stops]
    rs.sort(key=lambda s: s[0])
    return rs


def test_reverse_invariance():
    pals = pf.load_palettes()
    worst = 0.0
    for p in pals:
        fa = pf.palette_feature(p["stops"])
        fb = pf.palette_feature(_reversed_stops(p["stops"]))
        d = float(np.abs(np.asarray(fa["trajectory"]) - np.asarray(fb["trajectory"])).max())
        worst = max(worst, d)
        # canonicalization must land both on the SAME orientation
        assert fa["canonical_reversed"] != fb["canonical_reversed"] or d < 1e-9
    assert worst < 1e-9, "canonical feature is reverse-invariant (worst=%.2e)" % worst


# --------------------------------------------------------------- distance metric --

def test_distance_metric_properties():
    pals = pf.load_palettes()
    feats = pf.compute_all_features(pals)
    names = [p["name"] for p in pals]
    D = pf.distance_matrix(feats, names)
    assert np.allclose(np.diag(D), 0.0, atol=1e-12), "zero diagonal"
    assert np.allclose(D, D.T, atol=1e-12), "symmetric"
    assert (D >= -1e-12).all(), "non-negative"
    # identity: distance to self is 0, to a distinct palette is > 0
    assert pf.palette_distance(feats[names[0]], feats[names[0]]) == 0.0
    assert pf.palette_distance(feats["twilight"], feats["viridis"]) > 0.0


def test_fps_order_wellformed():
    pals = pf.load_palettes()
    feats = pf.compute_all_features(pals)
    names = [p["name"] for p in pals]
    order = pf.farthest_point_order(names, feats, k=10)
    assert len(order) == 10 and len(set(order)) == 10, "distinct, right length"
    assert set(order).issubset(set(names))


# ------------------------------------------------------------- soft type checks ---

def _report_type_spotchecks():
    pals = pf.load_palettes()
    feats = pf.compute_all_features(pals)
    types = {nm: pf.derive_type(f) for nm, f in feats.items()}
    # Type is now binary {cyclic, non_cyclic}: cyclic iff endpoints meet. The old
    # diverging cases (cmr.fusion/coolwarm/seismic) and the sequentials all fold into
    # non_cyclic.
    expect = {
        "twilight": "cyclic",
        "cmr.fusion": "non_cyclic",  # was diverging (blue<->white<->red); ends don't meet
        "coolwarm": "non_cyclic",
        "seismic": "non_cyclic",
        "viridis": "non_cyclic",
        "cividis": "non_cyclic",
        "magma": "non_cyclic",
    }
    print("\n[soft] type spot-checks (report only):")
    for nm, exp in expect.items():
        got = types.get(nm)
        print("   %-12s expect=%-11s got=%-11s %s"
              % (nm, exp, got, "ok" if got == exp else "<-- DIVERGES from expectation"))


# --------------------------------------------------------------------- runner -----

def main():
    hard = [test_color_reference_points, test_color_roundtrip, test_reverse_invariance,
            test_distance_metric_properties, test_fps_order_wellformed]
    failed = 0
    for t in hard:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    _report_type_spotchecks()
    print("\n%d/%d hard tests passed" % (len(hard) - failed, len(hard)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
