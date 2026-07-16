"""Tests for the query sampler's location pool — the Part-4 hardening guards.

Guards against the "0 Julia" regression: the pool must ingest BOTH families from the
labels/ store (merged images.jsonl score OR labels/*.json sidecar joined by image_id),
and its per-family Julia count must match the v5 pipeline's — the authoritative label
set (tools/v5/build_manifest.py). A drift in EITHER direction fails here.

Run either way:
  uv run pytest tools/queries/test_location_pool.py
  uv run python tools/queries/test_location_pool.py     # prints PASS/FAIL summary
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import query_sampler as qs  # noqa: E402


def _pool():
    return qs.LocationPool.from_corpus(verbose=False)


def test_julia_matches_v5():
    """The v5-era Julia location count (julia_ladder_j0 batch only) == the v5 pipeline's (join
    recipe + manifest). SCOPED to julia_ladder_j0 on BOTH sides — the frozen reference always was,
    and the live side now is (pool.v5_julia_count), NOT the all-batch family total. A future
    harvest that labels a new julia location must NOT trip this guard; that is exactly what scoping
    the live side buys (see test_new_julia_source_does_not_trip_guard)."""
    pool = _pool()
    got = pool.v5_julia_count()
    assert got == qs.v5_julia_q23_count(), (got, qs.v5_julia_q23_count())
    man = qs.v5_manifest_julia_q23()
    if man is not None:
        assert got == man, (got, man)
    # assert_matches_v5 bundles both checks; must not raise.
    assert pool.assert_matches_v5() == got


def test_new_julia_source_does_not_trip_guard():
    """A NEW julia source (another batch labeling julia locations — every future harvest) grows
    the julia family total but must leave the v5-era count and the guard untouched. This is the
    regression the scope fix exists to prevent: the OLD guard compared the all-batch family total
    against the julia_ladder_j0-frozen reference, so it fired on routine library growth."""
    pool = _pool()
    base_v5 = pool.v5_julia_count()
    base_family = pool.family_counts().get("julia", 0)
    # Inject a synthetic julia location from a DIFFERENT batch (not julia_ladder_j0).
    j = qs.loc_mod.Location(family="julia", cx="0.0", cy="0.0", fw="0.75", maxiter=800,
                            c_re="0.27", c_im="0.48")
    pool.locations.append(qs.PooledLocation(ref=j, scores=[3], batch_ids={"some_new_batch"}))
    assert pool.family_counts().get("julia", 0) == base_family + 1   # family total grew
    assert pool.v5_julia_count() == base_v5                          # v5-era count unchanged
    assert pool.assert_matches_v5() == base_v5                       # guard still green


def test_both_families_present():
    """Neither family may be silently dropped — both must be non-empty."""
    fc = _pool().family_counts()
    assert fc.get("mandelbrot", 0) > 0, fc
    assert fc.get("julia", 0) > 0, fc


def test_registered_sidecar_batches_contribute():
    """Every registered sidecar batch must join >0 q2+q3 rows (the join is live)."""
    census = _pool().census
    for bid in qs.SIDECAR_LABELS:
        assert bid in census, f"registered batch {bid!r} not found on disk"
        assert sum(census[bid].values()) > 0, f"{bid} joined 0 q2+q3 rows"


def test_julia_rows_carry_c():
    """Julia locations must carry c_re/c_im (needed for the field dump), Mandelbrot not."""
    pool = _pool()
    julia = [pl for pl in pool.locations if pl.kind == "julia"]
    assert julia, "no Julia locations"
    for pl in julia:
        assert pl.ref.c_re is not None and pl.ref.c_im is not None, pl.ref
    mand = [pl for pl in pool.locations if pl.kind == "mandelbrot"]
    assert all(pl.ref.c_re is None and pl.ref.c_im is None for pl in mand)


def main():
    tests = [
        test_julia_matches_v5,
        test_new_julia_source_does_not_trip_guard,
        test_both_families_present,
        test_registered_sidecar_batches_contribute,
        test_julia_rows_carry_c,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    print("\n%d/%d tests passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
