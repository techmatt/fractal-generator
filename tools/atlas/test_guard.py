#!/usr/bin/env python
"""Unit tests for the degenerate-outcome guard predicate + field measures.

The control for the *gate logic* (the manifest reproduction is the separate hard
gate in control_guard_manifest.py). Pure, GPU-free, no render.

  uv run pytest tools/atlas/test_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import guard  # noqa: E402


# --------------------------------------------------------------------------- #
# guard_fail — the pinned gate predicate.
# --------------------------------------------------------------------------- #
def test_interior_gate_fails_at_cap():
    # interior_frac >= 0.25 -> fail; field_std comfortably above the floor.
    assert guard.guard_fail(0.25, 100.0) == "interior"
    assert guard.guard_fail(0.30, 50.0) == "interior"


def test_flat_gate_fails_below_floor():
    # field_std < 6 -> fail; interior_frac comfortably below the cap.
    assert guard.guard_fail(0.0, 5.999) == "flat"
    assert guard.guard_fail(0.01, 0.0) == "flat"


def test_both_gates_fail():
    assert guard.guard_fail(0.30, 3.0) == "both"


def test_clean_crop_passes():
    # below interior cap AND at/above field-std floor -> pass (None).
    assert guard.guard_fail(0.24999, 6.0) is None
    assert guard.guard_fail(0.10, 400.0) is None


def test_gate_boundaries_are_the_pinned_thresholds():
    # interior gate is inclusive at the cap; flat gate is strict below the floor.
    assert guard.guard_fail(guard.INTERIOR_CAP, 100.0) == "interior"      # >= cap fails
    assert guard.guard_fail(guard.INTERIOR_CAP - 1e-9, 100.0) is None     # just under passes
    assert guard.guard_fail(0.0, guard.FIELD_STD_FLOOR) is None           # == floor passes
    assert guard.guard_fail(0.0, guard.FIELD_STD_FLOOR - 1e-9) == "flat"  # under floor fails


# --------------------------------------------------------------------------- #
# field_measures — reproduces diag_outcome_guards.measures (interior/std half).
# --------------------------------------------------------------------------- #
def test_field_measures_interior_frac_from_nan_mask():
    # 4x4 field, 4 NaN (interior) -> interior_frac 0.25.
    v = np.arange(16, dtype=np.float64).reshape(4, 4)
    v[0, 0] = v[0, 1] = v[1, 0] = v[1, 1] = np.nan
    st = guard.field_measures(v)
    assert st.n_px == 16 and st.n_escaped == 12
    assert abs(st.interior_frac - 0.25) < 1e-12
    assert abs(st.field_std - float(v[np.isfinite(v)].std())) < 1e-12


def test_field_measures_all_interior():
    v = np.full((3, 3), np.nan)
    st = guard.field_measures(v)
    assert st.interior_frac == 1.0 and st.field_std == 0.0 and st.n_escaped == 0


def test_field_measures_no_interior():
    v = np.ones((5, 5), dtype=np.float64) * 7.0
    st = guard.field_measures(v)
    assert st.interior_frac == 0.0 and st.field_std == 0.0


# --------------------------------------------------------------------------- #
# guarded scorer forces the sentinel on a failing sidecar (no v5 forward needed).
# --------------------------------------------------------------------------- #
class _FakeScorer:
    """Stand-in for the v5 Scorer: returns a fixed non-sentinel triple per path."""
    def score_paths(self, paths, batch_size=64):
        return [(1.5, 0.9, 0.6) for _ in paths]


def test_guarded_scorer_sentinels_failing_tiles(tmp_path, monkeypatch):
    gs = guard.GuardedScorer(_FakeScorer())

    # tile A: no field sidecar -> pass-through (real score).
    a = tmp_path / "a.jpg"; a.write_bytes(b"x")
    # tile B: field sidecar that FAILS (all-interior -> interior_frac 1.0).
    b = tmp_path / "b.jpg"; b.write_bytes(b"x")
    fake_fail = np.full((8, 8), np.nan)
    # tile C: field sidecar that PASSES (structured, std >> floor, no interior).
    c = tmp_path / "c.jpg"; c.write_bytes(b"x")
    fake_pass = (np.arange(64, dtype=np.float64).reshape(8, 8)) * 5.0

    # monkeypatch load_field to return our synthetic fields for the sidecars.
    import types
    fields = {
        str(guard.field_sidecar_for(b)): fake_fail,
        str(guard.field_sidecar_for(c)): fake_pass,
    }
    for p in (b, c):
        guard.field_sidecar_for(p).write_bytes(b"")  # presence marker
    monkeypatch.setattr(guard, "load_field",
                        lambda p: types.SimpleNamespace(values=fields[str(p)]))

    out = gs.score_paths([a, b, c])
    assert out[0] == (1.5, 0.9, 0.6)                     # A: pass-through
    assert out[1] == (guard.GUARD_SENTINEL, 0.0, 0.0)    # B: interior fail -> sentinel
    assert out[2] == (1.5, 0.9, 0.6)                     # C: passes -> real score
