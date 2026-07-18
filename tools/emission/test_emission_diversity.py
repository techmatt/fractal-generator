"""Tests for diversity-aware emission v1 — pure logic + the two acceptance proofs
(current-decode rejection of an old-ledger v6 row; append-only pool resume).

All tests are torch-free / render-free: the descriptor module's clustering + Location +
admitted-loader, the deficit machinery, the selector, and the pool are exercised directly.
Run: uv run pytest tools/emission/test_emission_diversity.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import cells as C          # noqa: E402
from tools.emission import selection as SEL     # noqa: E402
from tools.emission import descriptor as D     # noqa: E402
from tools.emission.pool import Pool           # noqa: E402
import corpus_common as cc                     # noqa: E402


# --------------------------------------------------------------------------- #
# cells.py — target measure, feasible cells, deficit, attempt cap, colorizer choice.
# --------------------------------------------------------------------------- #
def test_target_measure_overrides():
    tm = C.TargetMeasure.from_config({
        "attempt_cap": 3,
        "weight_overrides": [
            {"match": {"palette_flavor": ["k16:5"]}, "weight": 2.0},
            {"match": {"fractal_type": ["mandelbrot"], "render_style": ["tia"]}, "weight": 3.0},
        ],
    })
    # cell = (type, cluster, flavor, style)
    assert tm.weight(("mandelbrot", "m#0", "k16:5", "smooth")) == 2.0
    assert tm.weight(("mandelbrot", "m#0", "k16:1", "tia")) == 3.0
    assert tm.weight(("mandelbrot", "m#0", "k16:5", "tia")) == 6.0   # both overrides
    assert tm.weight(("multibrot3", "x#0", "k16:1", "smooth")) == 1.0


def test_feasible_cells_and_deficit_sign():
    tm = C.TargetMeasure.from_config({})
    observed = [("mandelbrot", "m#0"), ("multibrot3", "x#0")]
    flavors = ["k16:1", "k16:2"]
    styles = ["smooth", "tia"]
    cells = C.build_feasible_cells(observed, flavors, styles)
    assert len(cells) == 2 * 2 * 2
    m = C.DeficitModel(cells, tm)
    # empty pool: every cell deficit == its target fraction (all equal, uniform)
    d0 = m.deficit(cells[0])
    assert d0 == pytest.approx(1.0 / len(cells))
    # fill one cell → its deficit drops below an unfilled cell's
    m.record_fill(cells[0])
    assert m.deficit(cells[0]) < m.deficit(cells[1])


def test_attempt_cap_evicts_cell():
    tm = C.TargetMeasure.from_config({"attempt_cap": 3})
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1"], ["smooth", "tia"])
    m = C.DeficitModel(cells, tm)
    target = ("mandelbrot", "m#0", "k16:1", "smooth")
    assert m.record_attempt(target) is False   # 1
    assert m.record_attempt(target) is False   # 2
    assert m.record_attempt(target) is True    # 3 → capped (zero fills)
    assert target in m.capped and target not in m.support
    # a filled cell is never capped no matter how many attempts
    other = ("mandelbrot", "m#0", "k16:1", "tia")
    m.record_fill(other)
    for _ in range(10):
        assert m.record_attempt(other) is False


def test_range_normalized_softmax_prefers_max():
    p = C.range_normalized_softmax([0.1, 0.0, 0.0], temp=0.2)
    assert p[0] > p[1] and p[0] > p[2]
    assert p[1] == pytest.approx(p[2])
    assert sum(p) == pytest.approx(1.0)
    # all equal → uniform
    q = C.range_normalized_softmax([0.5, 0.5, 0.5], temp=0.2)
    assert all(x == pytest.approx(1 / 3) for x in q)


def test_choose_option_avoids_filled():
    tm = C.TargetMeasure.from_config({"softmax_temp": 0.05})
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1", "k16:2"], ["smooth"])
    m = C.DeficitModel(cells, tm)
    # fill (k16:1, smooth) heavily so the deficit strongly favors (k16:2, smooth)
    for _ in range(5):
        m.record_fill(("mandelbrot", "m#0", "k16:1", "smooth"))
    rng = np.random.default_rng(0)
    picks = [C.choose_option(m, "mandelbrot", "m#0", ["k16:1", "k16:2"], ["smooth"], rng)[0]
             for _ in range(200)]
    from collections import Counter
    ct = Counter(picks)
    assert ct["k16:2"] > ct["k16:1"]      # deficit steers away from the filled flavor


# --------------------------------------------------------------------------- #
# select.py — kernel, niche percentile, greedy coverage.
# --------------------------------------------------------------------------- #
def _entry(id, type, cluster, flavor, style, score, emb):
    return {"id": id, "type": type, "cluster": cluster, "flavor": flavor,
            "style": style, "score": score, "emb": emb}


def test_kernel_zero_across_cells_cosine_within():
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.9, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#0", "k16:1", "smooth", 0.8, [1.0, 0.0])   # same cell, cos 1
    c = _entry("c", "mandelbrot", "m#0", "k16:2", "smooth", 0.8, [1.0, 0.0])   # diff flavor
    assert SEL.kernel(a, b) == pytest.approx(1.0)
    assert SEL.kernel(a, c) == 0.0


def test_greedy_prefers_distinct_cells():
    # two near-duplicate entries in ONE cell + one entry in another cell; N=2 → one per cell.
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.95, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#0", "k16:1", "smooth", 0.90, [1.0, 0.0])
    c = _entry("c", "mandelbrot", "m#0", "k16:2", "smooth", 0.80, [0.0, 1.0])
    selected, log = SEL.greedy_select([a, b, c], 2)
    cells = {(e["type"], e["cluster"], e["flavor"], e["style"]) for e in selected}
    assert len(cells) == 2                     # spread across cells, not two from the crowded one
    assert {e["id"] for e in selected} == {"a", "c"}


def test_niche_percentile_singleton_is_one():
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.5, [1.0])
    pct = SEL.niche_percentiles([a])
    assert pct["a"] == 1.0


# --------------------------------------------------------------------------- #
# descriptor.py — clustering + Location mapping.
# --------------------------------------------------------------------------- #
def test_cluster_incremental_join_and_new():
    items = [("a", np.array([1.0, 0.0, 0.0], np.float32)),
             ("b", np.array([1.0, 0.0, 0.0], np.float32)),   # cos 1 → joins a
             ("c", np.array([0.0, 1.0, 0.0], np.float32))]    # cos 0 → new
    assign = D.cluster_incremental(items, threshold=0.974)
    assert assign["a"] == assign["b"]
    assert assign["c"] != assign["a"]


def test_assign_morph_clusters_within_type():
    rows = [{"id": "a", "family": "mandelbrot"}, {"id": "b", "family": "mandelbrot"},
            {"id": "c", "family": "multibrot3"}]
    embs = {"a": np.array([1.0, 0.0], np.float32),
            "b": np.array([1.0, 0.0], np.float32),
            "c": np.array([1.0, 0.0], np.float32)}
    tags = D.assign_morph_clusters(rows, embs)
    assert tags["a"] == tags["b"] == "mandelbrot#0"
    assert tags["c"] == "multibrot3#0"          # different type → own namespace


def test_location_of_partition_mapping():
    m = D.location_of({"family": "mandelbrot", "outcome_cx": -0.5, "outcome_cy": 0.1,
                       "outcome_fw": 0.03})
    assert m.family == "mandelbrot" and m.c_re is None
    j = D.location_of({"family": "julia:multibrot3", "outcome_cx": 0.0, "outcome_cy": 0.0,
                       "outcome_fw": 3.0, "julia_c_re": 0.28, "julia_c_im": 0.008})
    assert j.family == "julia_multibrot3" and j.c_re == "0.28"


# --------------------------------------------------------------------------- #
# ACCEPTANCE — current-decode rejects an old-ledger v6 row.
# --------------------------------------------------------------------------- #
def _row(id, ver, dc=3, guard=True, distinct=True):
    return {"id": id, "family": "mandelbrot", "outcome_cx": -0.5, "outcome_cy": 0.1,
            "outcome_fw": 0.03, "decoded_class": dc, "guard_pass": guard,
            "distinct": distinct, "scorer_version": ver}


def test_v6_row_rejected(tmp_path):
    assert cc.active_scorer_version() == "v7"    # environment sanity: current is v7
    led = tmp_path / "outcome_ledger.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in [
        _row("cur", "v7"), _row("old", "v6"), _row("older", "v5"),
    ]) + "\n", encoding="utf-8")

    # soft form: stale rows silently skipped, only the current row admitted.
    admitted = D.load_admitted(led)
    assert [r["id"] for r in admitted] == ["cur"]

    # strict form: a v6 row RAISES rather than being consumed as a current verdict.
    only_v6 = tmp_path / "v6_only.jsonl"
    only_v6.write_text(json.dumps(_row("old", "v6")) + "\n", encoding="utf-8")
    with pytest.raises(cc.StaleDecodeError):
        D.load_admitted(only_v6, require_current=True)


# --------------------------------------------------------------------------- #
# ACCEPTANCE — append-only pool resume (no lost / duplicated entries).
# --------------------------------------------------------------------------- #
def _prec(id, loc, passed, cell):
    return {"id": id, "location_id": loc, "cell": list(cell), "passed": passed,
            "p_ge3": 0.8 if passed else 0.4}


def test_pool_resume_no_loss_no_dup(tmp_path):
    p = Pool(tmp_path)
    assert p.next_id() == "em_000000"
    cell = ("mandelbrot", "m#0", "k16:1", "smooth")
    p.append(_prec(p.next_id(), "loc0", True, cell))
    p.append(_prec(p.next_id(), "loc0", False, cell))
    p.append(_prec(p.next_id(), "loc1", True, cell))
    assert p.next_id() == "em_000003"

    # simulate kill + resume: a brand-new Pool over the same dir replays the durable log.
    q = Pool(tmp_path)
    assert q.n_attempts() == 3
    assert q.next_id() == "em_000003"                 # sequence continues, no collision
    assert [r["id"] for r in q.gated()] == ["em_000000", "em_000002"]
    assert q.attempts_per_location() == {"loc0": 2, "loc1": 1}
    # ids are unique (no duplication of a logged row)
    ids = [r["id"] for r in q.rows]
    assert len(ids) == len(set(ids))

    # a resumed append does not rewrite or duplicate prior rows.
    q.append(_prec(q.next_id(), "loc2", True, cell))
    r = Pool(tmp_path)
    assert [x["id"] for x in r.rows] == ["em_000000", "em_000001", "em_000002", "em_000003"]


def test_deficit_rebuild_from_pool_log(tmp_path):
    """The build_axes resume path: replaying the pool log reproduces fill+attempt counts."""
    cells = C.build_feasible_cells([("mandelbrot", "m#0")], ["k16:1"], ["smooth", "tia"])
    tm = C.TargetMeasure.from_config({"attempt_cap": 99})
    p = Pool(tmp_path)
    recs = [_prec(p.next_id(), "loc0", True, cells[0]),
            _prec(p.next_id(), "loc0", False, cells[0]),
            _prec(p.next_id(), "loc0", True, cells[1])]
    for rc in recs:
        p.append(rc)
    q = Pool(tmp_path)
    m = C.DeficitModel(cells, tm)
    for rr in q.rows:
        cell = tuple(rr["cell"])
        m.record_attempt(cell)
        if rr["passed"]:
            m.record_fill(cell)
    assert m.attempt_counts[cells[0]] == 2 and m.fill_counts[cells[0]] == 1
    assert m.attempt_counts[cells[1]] == 1 and m.fill_counts[cells[1]] == 1
