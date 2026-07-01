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
    """The sampler's Julia location count == the v5 pipeline's (join recipe + manifest)."""
    pool = _pool()
    got = pool.family_counts().get("julia", 0)
    assert got == qs.v5_julia_q23_count(), (got, qs.v5_julia_q23_count())
    man = qs.v5_manifest_julia_q23()
    if man is not None:
        assert got == man, (got, man)
    # assert_matches_v5 bundles both checks; must not raise.
    assert pool.assert_matches_v5() == got


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
