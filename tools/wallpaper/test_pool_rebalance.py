"""Tests for the Phase-1 pool-cap + discovery-budget rebalance (pool-cap-and-discovery-rebalance).

Three deliverable properties, all GPU-free (pure dedup / accounting / knob logic — no torch, no
render, no seeder subprocess):

  * reconciliation assert fires on a synthetic drop (the harvest-leak halt);
  * identity-aware coord dedup honors julia `c` and phoenix z-plane scale;
  * the per-family c-plane budget knob changes descent budget WITHOUT zeroing parent supply.

Run either way:
  uv run pytest tools/wallpaper/test_pool_rebalance.py
  uv run python tools/wallpaper/test_pool_rebalance.py     # prints PASS/FAIL summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "tools" / "corpus"))

import library_dedup as dedup            # noqa: E402
import prospect_orchestrator as po       # noqa: E402
from tools.corpus import location as loc_mod  # noqa: E402


# =========================================================================== #
# Fixtures.
# =========================================================================== #
def _record(family, cx, cy, fw, c=None, p=None):
    """A minimal store record (only the identity block the dedup index reads)."""
    return {"identity": {"family": family, "cx": str(cx), "cy": str(cy), "fw": str(fw),
                         "c": c, "p": p}}


def _store(tmp_path, records):
    p = tmp_path / "records.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _loc(family, cx, cy, fw, c_re=None, c_im=None, **fp):
    return loc_mod.Location(family=family, cx=str(cx), cy=str(cy), fw=str(fw), maxiter=1500,
                            c_re=c_re, c_im=c_im, family_params=fp)


# =========================================================================== #
# 1. Identity-aware coordinate dedup.
# =========================================================================== #
def test_cplane_proximity_scale_aware(tmp_path):
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("mandelbrot", "0.10", "0.20", "0.010")]))
    # within 0.5*min(fw) of the same spot -> dup; far -> not.
    assert idx.is_dup("mandelbrot", "0.1004", "0.2003", "0.010", None, None)
    assert not idx.is_dup("mandelbrot", "0.20", "0.20", "0.010", None, None)
    # a MUCH tighter incoming fw shrinks the tolerance (min(fw)): the same 0.004 offset that
    # merged at fw=0.01 no longer merges when the incoming frame is 100x tighter.
    assert not idx.is_dup("mandelbrot", "0.1004", "0.2003", "0.0001", None, None)


def test_julia_requires_matching_c(tmp_path):
    # Two julia locations sharing a z-plane viewport but DIFFERENT c are different fractals.
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("julia", "0.0", "0.0", "0.01", c={"re": "0.233", "im": "0.538"})]))
    same_c = dedup.coord_of_location(_loc("julia", "0.0", "0.0", "0.01",
                                          c_re="0.233", c_im="0.538"))
    diff_c = dedup.coord_of_location(_loc("julia", "0.0", "0.0", "0.01",
                                          c_re="-0.4", c_im="0.6"))
    assert idx.is_dup(*same_c)          # same viewport AND same c -> dup
    assert not idx.is_dup(*diff_c)      # same viewport, different c -> NOT a dup


def test_julia_multibrot_degree_and_c(tmp_path):
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("julia_multibrot3", "1.0", "1.0", "0.02", c={"re": "-0.387", "im": "-0.629"})]))
    # different render-family (degree) never collides even at identical coords+c
    assert not idx.is_dup("julia_multibrot4", "1.0", "1.0", "0.02", "-0.387", "-0.629")
    assert idx.is_dup("julia_multibrot3", "1.0001", "1.0001", "0.02", "-0.387", "-0.629")


def test_phoenix_scale_aware_zplane(tmp_path):
    # Phoenix carries the fixed Ushiki c, so its c always matches: the test reduces to a
    # SCALE-AWARE z-plane viewport proximity. A deep spot under a shallow neighbour must NOT
    # over-merge (the min(fw), not 1.5*max(fw), rule).
    ph_c = {"re": "0.5667", "im": "0.0"}
    idx = dedup.StoreIndex.from_records(_store(tmp_path, [
        _record("phoenix", "0.30", "0.40", "1.0", c=ph_c),      # shallow
        _record("phoenix", "0.300000", "0.400000", "1e-4", c=ph_c)]))  # deep, ~same center
    # a NEW deep spot 0.02 away from the shallow one: 0.5*min(fw)=0.5*1e-4 tolerance -> NOT a dup
    assert not idx.is_dup("phoenix", "0.32", "0.42", "1e-4", "0.5667", "0.0")
    # a new spot truly on top of the deep record (within 0.5*1e-4) IS a dup
    assert idx.is_dup("phoenix", "0.3000003", "0.4000002", "1e-4", "0.5667", "0.0")
    # and the shallow record still merges a nearby shallow spot (its own scale)
    assert idx.is_dup("phoenix", "0.31", "0.41", "1.0", "0.5667", "0.0")


def test_within_batch_accumulation():
    # add_location makes a second same-spot q3 in the SAME cycle collapse onto the first.
    idx = dedup.StoreIndex()
    a = _loc("mandelbrot", "0.5", "0.5", "0.01")
    assert not idx.is_location_dup(a)   # empty index
    idx.add_location(a)
    b = _loc("mandelbrot", "0.5001", "0.5001", "0.01")
    assert idx.is_location_dup(b)       # now a within-batch dup of `a`


# =========================================================================== #
# 2. Per-cycle reconciliation.
# =========================================================================== #
def test_reconcile_clean_passes():
    # 10 fresh q3 -> 8 records + 2 true coord-dups (1 within-set at build, 1 store/batch). Clean.
    sel = {"within_set_dups_dropped": 1, "unrenderable_dropped": 0,
           "excluded_head_corpus_by_key": 0, "excluded_head_corpus_by_proximity": 0}
    ann = {"dropped_coord_dup": 1, "dropped_field_fail": 0, "records_written": 8}
    bd = po.reconcile_cycle(10, 8, sel, ann)
    assert bd["dropped_other"] == 0
    assert bd["dropped_coord_dup"] == 2
    assert bd["q3_found"] == bd["records_written"] + bd["dropped_coord_dup"]


def test_reconcile_head_exclusion_leak_halts():
    # A head-corpus exclusion (a selection drop) leaves a q3 unaccounted -> dropped_other != 0.
    sel = {"within_set_dups_dropped": 0, "unrenderable_dropped": 0,
           "excluded_head_corpus_by_key": 1, "excluded_head_corpus_by_proximity": 0}
    ann = {"dropped_coord_dup": 0, "dropped_field_fail": 0, "records_written": 9}
    try:
        po.reconcile_cycle(10, 9, sel, ann)
        assert False, "expected the reconciliation assert to fire on a silent selection drop"
    except SystemExit as e:
        assert "LEAK" in str(e) and "excl_head=1" in str(e)


def test_reconcile_field_fail_leak_halts():
    # A field render failure is a genuine leak (not a dup) -> halt.
    sel = {"within_set_dups_dropped": 0, "unrenderable_dropped": 0,
           "excluded_head_corpus_by_key": 0, "excluded_head_corpus_by_proximity": 0}
    ann = {"dropped_coord_dup": 2, "dropped_field_fail": 1, "records_written": 7}
    try:
        po.reconcile_cycle(10, 7, sel, ann)     # 7 + 2 + (1 leak) = 10
        assert False, "expected halt on a field-fail leak"
    except SystemExit as e:
        assert "field_fail=1" in str(e)


def test_reconcile_missing_report_surfaces_as_leak():
    # An absent annotate report ({}) means the store delta must fully account for q3_found on its
    # own; any shortfall surfaces as a leak rather than passing silently.
    try:
        po.reconcile_cycle(5, 3, {}, {})        # 3 records, nothing else explained -> 2 leaked
        assert False, "expected halt when reports are missing and records < q3_found"
    except SystemExit:
        pass
    # but if every q3 became a record, missing reports still reconcile cleanly.
    assert po.reconcile_cycle(5, 5, {}, {})["dropped_other"] == 0


# =========================================================================== #
# 3. Per-family c-plane budget knob (Part 2) — changes budget, never zeroes supply.
# =========================================================================== #
def _args(per_family_min=7.0, mb_cplane_min=None, mb5_every=1):
    return argparse.Namespace(per_family_min=per_family_min, mb_cplane_min=mb_cplane_min,
                              mb5_every=mb5_every)


def test_family_budget_default_no_cut():
    a = _args(per_family_min=7.0, mb_cplane_min=None)
    # default: multibrot inherits per_family_min (conservative — NO cut until measured)
    for fam in ("mandelbrot", "multibrot3", "multibrot4", "multibrot5"):
        assert po._family_cplane_min(fam, a) == 7.0


def test_family_budget_cut_multibrot_only_and_nonzero():
    a = _args(per_family_min=7.0, mb_cplane_min=3.0)
    assert po._family_cplane_min("mandelbrot", a) == 7.0      # mandelbrot untouched
    for fam in ("multibrot3", "multibrot4", "multibrot5"):
        b = po._family_cplane_min(fam, a)
        assert b == 3.0                                       # cut applied...
        assert b > 0                                          # ...but parent supply NOT zeroed
    # a tighter cut still stays strictly positive (a knob, never a delete)
    assert po._family_cplane_min("multibrot3", _args(mb_cplane_min=0.5)) == 0.5


def test_mb5_every_gating_predicate():
    # the orchestrator runs mb5 iff (cycle-1) % N == 0; N=1 -> every cycle, N=3 -> 1,4,7,...
    def runs(cycle, n):
        return (cycle - 1) % max(1, n) == 0
    assert [c for c in range(1, 8) if runs(c, 1)] == [1, 2, 3, 4, 5, 6, 7]
    assert [c for c in range(1, 8) if runs(c, 3)] == [1, 4, 7]
    assert [c for c in range(1, 8) if runs(c, 2)] == [1, 3, 5, 7]


# =========================================================================== #
# Standalone runner.
# =========================================================================== #
def _run_standalone():
    import tempfile
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    npass = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS {name}")
            npass += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{npass}/{len(tests)} passed")
    return npass == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_standalone() else 1)
