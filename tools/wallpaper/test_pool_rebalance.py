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


def _sel(within=0, unrenderable=0, excl_key=0, excl_prox=0):
    return {"within_set_dups_dropped": within, "unrenderable_dropped": unrenderable,
            "excluded_head_corpus_by_key": excl_key, "excluded_head_corpus_by_proximity": excl_prox}


def _ann(coord_dup=0, field_fail=0, records=0):
    return {"dropped_coord_dup": coord_dup, "dropped_field_fail": field_fail,
            "records_written": records}


# --- SELECTION-SHAPED loss (a cap/ranking sneaking back) STILL HALTS loudly ---------------- #
def test_reconcile_head_exclusion_leak_halts():
    # A head-corpus exclusion (a selection drop) in a --no-head-exclude phase -> HALT.
    try:
        po.reconcile_cycle(10, 9, _sel(excl_key=1), _ann(records=9))
        assert False, "expected the reconciliation assert to fire on a silent selection drop"
    except SystemExit as e:
        assert "LEAK" in str(e) and "excl_head=1" in str(e) and "selection-shaped" in str(e)


def test_reconcile_unexplained_loss_halts():
    # A q3 that vanished with NO reason recorded (not record/coord-dup/field-fail/deferral) -> HALT.
    try:
        po.reconcile_cycle(10, 8, _sel(), _ann(records=8))   # 10 - 8 = 2 unexplained
        assert False, "expected halt on unexplained loss"
    except SystemExit as e:
        assert "unexplained" in str(e)


def test_reconcile_missing_report_surfaces_as_unexplained():
    # An absent annotate report ({}) means the store delta must fully account for q3_found on its
    # own; any shortfall surfaces as unexplained rather than passing silently.
    try:
        po.reconcile_cycle(5, 3, {}, {})        # 3 records, nothing else explained -> 2 unexplained
        assert False, "expected halt when reports are missing and records < q3_found"
    except SystemExit:
        pass
    # but if every q3 became a record, missing reports still reconcile cleanly.
    assert po.reconcile_cycle(5, 5, {}, {})["dropped_other"] == 0


# --- FIELD-FAIL is OPERATIONAL: tolerated by RATE, never per-event ------------------------- #
def test_reconcile_single_field_fail_is_noise_not_a_halt():
    # ONE render failure is location-specific noise: accounted as field_fail, no halt (was a halt
    # under the old per-event rule). 7 records + 2 coord-dup + 1 field_fail = 10, all accounted.
    bd = po.reconcile_cycle(10, 7, _sel(), _ann(coord_dup=2, field_fail=1))
    assert bd["field_fail"] == 1 and bd["unexplained"] == 0 and bd["dropped_other"] == 0


def test_reconcile_field_fail_below_rate_continues():
    # 4/10 = 40% render failures (< 50% floor) -> tolerated, no halt.
    bd = po.reconcile_cycle(10, 6, _sel(), _ann(field_fail=4))
    assert bd["field_fail"] == 4 and bd["field_fail_rate"] == 0.4 and bd["unexplained"] == 0


def test_reconcile_field_fail_at_rate_halts():
    # 5/10 = 50% (>= floor, >= 2 failures) -> systemic render defect -> HALT (fires AT the rate).
    try:
        po.reconcile_cycle(10, 5, _sel(), _ann(field_fail=5))
        assert False, "expected halt at the field-fail rate floor"
    except SystemExit as e:
        assert "field-fail rate" in str(e) and "50%" in str(e)


def test_reconcile_field_fail_rate_only_above_min_count():
    # A single failure that is 50% of a TINY cycle (1/2) is still just noise: the min-count floor
    # (2) keeps it from false-halting. Rate fires only when BOTH rate and count thresholds are met.
    bd = po.reconcile_cycle(2, 1, _sel(), _ann(field_fail=1))   # 50% but only 1 failure
    assert bd["field_fail"] == 1 and bd["dropped_other"] == 0


# --- unrenderable (pool-build render failure) folds into the field-fail bucket ------------- #
def test_reconcile_unrenderable_is_field_fail():
    bd = po.reconcile_cycle(10, 8, _sel(unrenderable=1), _ann(coord_dup=1, records=8))
    assert bd["field_fail"] == 1 and bd["unexplained"] == 0


# --- DEFERRED (a retried-then-failed cycle's q3) is accounted, never a halt ---------------- #
def test_reconcile_deferred_accounted():
    # The whole cycle's q3 were deferred to a re-run; passed in as `deferred`, reconciles clean.
    bd = po.reconcile_cycle(10, 0, _sel(), _ann(), deferred=10)
    assert bd["deferred"] == 10 and bd["unexplained"] == 0 and bd["dropped_other"] == 0


# =========================================================================== #
# 2b. Annotate crash -> defer, retry once, record-and-continue (no halt).
# =========================================================================== #
def test_annotate_retry_succeeds_on_second_attempt():
    # First attempt fails, retry succeeds -> None (proceed to reconcile), attempt called twice.
    calls = []
    def attempt(i):
        calls.append(i)
        return (i == 1, "" if i == 1 else "crash")
    fc = po.annotate_with_retry(attempt, cycle=3, q3_count=12, watermark=100,
                                salvaged=lambda: 12)
    assert fc is None and calls == [0, 1]      # retried exactly once, then proceeded


def test_annotate_double_failure_defers_and_continues():
    # Both attempts fail -> a failed-cycle dict (recorded), NOT a raise: q3 deferred, run continues.
    calls = []
    def attempt(i):
        calls.append(i)
        return (False, "annotate_report.json missing (crash)")
    fc = po.annotate_with_retry(attempt, cycle=3, q3_count=12, watermark=100,
                                salvaged=lambda: 4)
    assert calls == [0, 1]                      # tried once + retried once, then gave up (no 3rd)
    assert fc is not None                       # recorded, not raised
    assert fc["cycle"] == 3 and fc["q3_deferred"] == 12 and fc["records_salvaged"] == 4
    assert fc["ledger_watermark"] == 100 and "failed twice" in fc["reason"]


def test_annotate_first_attempt_success_no_retry():
    calls = []
    def attempt(i):
        calls.append(i)
        return (True, "")
    fc = po.annotate_with_retry(attempt, cycle=3, q3_count=12, watermark=100, salvaged=lambda: 12)
    assert fc is None and calls == [0]         # no retry when the first attempt succeeds


# =========================================================================== #
# 3. Per-family c-plane budget knob (Part 2) — changes budget, never zeroes supply.
# =========================================================================== #
def _args(per_family_min=7.0, mb_cplane_min=None):
    return argparse.Namespace(per_family_min=per_family_min, mb_cplane_min=mb_cplane_min)


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
