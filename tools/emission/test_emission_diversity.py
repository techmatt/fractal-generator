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


def test_source_tag_override_resolves_to_clusters():
    # source_tag names a location set durably; resolve rewrites it to the morph clusters those
    # locations currently occupy, and up-weights exactly those cells.
    tm = C.TargetMeasure.from_config({
        "weight_overrides": [
            {"match": {"fractal_type": ["phoenix"]}, "weight": 0.21},
            {"match": {"source_tag": ["classic_phoenix"]}, "weight": 1.9},
        ],
    })
    # 2 classic locations (in one cluster) + 1 varied location (another cluster).
    loc_src = {"a": "classic_phoenix", "b": "classic_phoenix", "c": "phoenix_grid"}
    loc_cl = {"a": "phoenix#7", "b": "phoenix#7", "c": "phoenix#3"}
    diag = tm.resolve_source_tags(loc_src, loc_cl)
    assert len(diag) == 1
    assert diag[0]["n_locations"] == 2 and diag[0]["resolved_clusters"] == ["phoenix#7"]
    assert diag[0]["impure_clusters"] == []                      # cluster #7 is all-classic
    # no source_tag key survives (rewritten to morph_cluster); idempotent second call.
    assert all("source_tag" not in ov.get("match", {}) for ov in tm.weight_overrides)
    assert tm.resolve_source_tags(loc_src, loc_cl) == []
    # classic cell carries BOTH the type budget knob and the split knob; varied only the budget.
    assert tm.weight(("phoenix", "phoenix#7", "k16:1", "smooth")) == pytest.approx(0.21 * 1.9)
    assert tm.weight(("phoenix", "phoenix#3", "k16:1", "smooth")) == pytest.approx(0.21)


def test_source_tag_override_survives_cluster_id_permutation():
    # The whole point of keying on source_tag: a re-cluster that RENAMES cluster ids must not
    # change which locations the override up-weights. Resolve against a permuted loc->cluster
    # map and confirm the SAME locations (by source tag) still get the multiplier.
    cfg = {"weight_overrides": [{"match": {"source_tag": ["classic_phoenix"]}, "weight": 1.9}]}
    loc_src = {"a": "classic_phoenix", "b": "classic_phoenix", "c": "phoenix_grid"}
    base_cl = {"a": "phoenix#196", "b": "phoenix#197", "c": "phoenix#3"}
    perm_cl = {"a": "phoenix#42", "b": "phoenix#99", "c": "phoenix#201"}   # ids permuted
    w_base = C.TargetMeasure.from_config(cfg)
    w_base.resolve_source_tags(loc_src, base_cl)
    w_perm = C.TargetMeasure.from_config(cfg)
    w_perm.resolve_source_tags(loc_src, perm_cl)
    # tagged locations keep the multiplier under BOTH clusterings; the untagged one never gets it.
    assert w_base.weight(("phoenix", "phoenix#196", "k", "smooth")) == pytest.approx(1.9)
    assert w_base.weight(("phoenix", "phoenix#197", "k", "smooth")) == pytest.approx(1.9)
    assert w_base.weight(("phoenix", "phoenix#3", "k", "smooth")) == pytest.approx(1.0)
    assert w_perm.weight(("phoenix", "phoenix#42", "k", "smooth")) == pytest.approx(1.9)
    assert w_perm.weight(("phoenix", "phoenix#99", "k", "smooth")) == pytest.approx(1.9)
    assert w_perm.weight(("phoenix", "phoenix#201", "k", "smooth")) == pytest.approx(1.0)
    # config carries NO cluster id — the override text is identical across both intakes.
    assert cfg["weight_overrides"][0]["match"] == {"source_tag": ["classic_phoenix"]}


def test_source_tag_unresolved_is_noop_not_crash():
    # A consumer that never resolves (the discovery-side projection) must see a source_tag
    # override as a never-matching no-op, NOT an axis-index crash.
    tm = C.TargetMeasure.from_config(
        {"weight_overrides": [{"match": {"source_tag": ["classic_phoenix"]}, "weight": 1.9}]})
    assert tm.weight(("phoenix", "phoenix#196", "k16:1", "smooth")) == pytest.approx(1.0)


def test_source_tag_impure_cluster_flagged():
    # A resolved cluster that also holds an untagged location is reported impure (the override
    # would up-weight that member too) — the caller's equivalence gate is what forbids it.
    tm = C.TargetMeasure.from_config(
        {"weight_overrides": [{"match": {"source_tag": ["classic_phoenix"]}, "weight": 1.9}]})
    loc_src = {"a": "classic_phoenix", "b": "phoenix_grid"}
    loc_cl = {"a": "phoenix#7", "b": "phoenix#7"}                # mixed cluster
    diag = tm.resolve_source_tags(loc_src, loc_cl)
    assert diag[0]["impure_clusters"] == ["phoenix#7"] and diag[0]["n_locations"] == 1


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


def test_kernel_continuous_cos_across_cells():
    # continuous morph cos, NO categorical gate: a near-identical look is discounted even
    # across cells (this is the coverage-engages fix — the old kernel returned 0 for c).
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "smooth", 0.9, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#0", "k16:1", "smooth", 0.8, [1.0, 0.0])   # same cell, cos 1
    c = _entry("c", "mandelbrot", "m#0", "k16:2", "smooth", 0.8, [1.0, 0.0])   # diff flavor, cos 1
    d = _entry("d", "mandelbrot", "m#0", "k16:2", "smooth", 0.8, [0.0, 1.0])   # diff flavor, cos 0
    assert SEL.kernel(a, b) == pytest.approx(1.0)
    assert SEL.kernel(a, c) == pytest.approx(1.0)   # was 0.0 under the categorical gate
    assert SEL.kernel(a, d) == pytest.approx(0.0)


def test_kernel_style_weight_floors_same_mode():
    # morph-distinct (orthogonal) tiles of the SAME render style are floored at style_weight;
    # a different style stays at the (here 0) cosine — how the strange pass spreads modes.
    a = _entry("a", "mandelbrot", "m#0", "k16:1", "tia", 0.6, [1.0, 0.0])
    b = _entry("b", "mandelbrot", "m#1", "k16:2", "tia", 0.6, [0.0, 1.0])       # same style, cos 0
    c = _entry("c", "mandelbrot", "m#2", "k16:3", "stripe", 0.6, [0.0, 1.0])    # diff style, cos 0
    assert SEL.kernel(a, b) == pytest.approx(0.0)                # no floor → 0
    assert SEL.kernel(a, b, style_weight=0.5) == pytest.approx(0.5)
    assert SEL.kernel(a, c, style_weight=0.5) == pytest.approx(0.0)


def test_greedy_style_weight_spreads_modes():
    # 3 tia + 1 stripe, all morph-distinct, N=2; the style floor makes the 2nd pick switch
    # modes to stripe rather than take a 2nd (lower-score) tia.
    e = [_entry("t0", "mandelbrot", "m#0", "k16:1", "tia", 0.90, [1.0, 0.0, 0.0, 0.0]),
         _entry("t1", "mandelbrot", "m#1", "k16:1", "tia", 0.80, [0.0, 1.0, 0.0, 0.0]),
         _entry("t2", "mandelbrot", "m#2", "k16:1", "tia", 0.70, [0.0, 0.0, 1.0, 0.0]),
         _entry("s0", "mandelbrot", "m#3", "k16:1", "stripe", 0.60, [0.0, 0.0, 0.0, 1.0])]
    sel, _log = SEL.greedy_select(e, 2, style_weight=0.5)
    styles = {x["style"] for x in sel}
    assert styles == {"tia", "stripe"}         # spread, not two tia
    assert sel[0]["id"] == "t0"                # best tia first


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


# --------------------------------------------------------------------------- #
# ranker (pref_loc_v0) — percentiles + cache-only scoring parity (render-free).
# --------------------------------------------------------------------------- #
from tools.ranker.score_locations import rank_percentiles, LocationRanker, DEFAULT_FEATURES  # noqa: E402


def test_rank_percentiles_ties_share_higher_rank():
    pct = rank_percentiles({"a": 1.0, "b": 2.0, "c": 2.0})
    assert pct["a"] == pytest.approx(1 / 3)      # smallest → bottom third
    assert pct["b"] == pct["c"] == 1.0           # ties both count each other as <= → top
    assert rank_percentiles({}) == {}
    assert rank_percentiles({"solo": 5.0})["solo"] == 1.0


@pytest.mark.skipif(not (ROOT / "data/ranker/pref_loc_v0/model.npz").exists()
                    or not DEFAULT_FEATURES.exists(),
                    reason="pref_loc_v0 artifacts absent")
def test_location_ranker_cache_hit_matches_direct_scoring():
    from tools.ranker.scorer import RankerScorer
    z = np.load(DEFAULT_FEATURES, allow_pickle=True)
    s = RankerScorer.load()
    direct = {str(z["ids"][k]): float(v)
              for k, v in enumerate(s.score_matrix({b: z[b] for b in s.sets}))}
    lr = LocationRanker()
    rows = [{"id": str(i)} for i in z["ids"]]
    mine = lr.score_rows(rows, ROOT / "out" / "_test_ranker_tiles")   # all cache hits
    assert lr._stack is None                     # torch feature stack never loaded
    assert max(abs(mine[i] - direct[i]) for i in direct) < 1e-9


# --------------------------------------------------------------------------- #
# driver — per-head release floors + short-fill + multi-ledger intake dedup.
# --------------------------------------------------------------------------- #
from tools.emission import build_emission_diversity_v1 as B     # noqa: E402


def _args(tmp_path, **over):
    import argparse
    a = argparse.Namespace(
        ledger=["x.jsonl"], out=str(tmp_path / "out"), report=None, release_n=5,
        target_gated=0, floor=B.DEFAULT_FLOOR, mining_floor=B.DEFAULT_MINING_FLOOR,
        release_floor=B.DEFAULT_RELEASE_FLOOR, mining_release_floor=B.DEFAULT_MINING_RELEASE_FLOOR,
        intake_floor=None, target_measure=str(B.DEFAULT_TARGET_MEASURE),
        strange_frac=B.DEFAULT_STRANGE_FRAC,
        max_attempts=240, time_budget_min=45.0, seed=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _gate_rec(id, loc, style, p_ge3, cell):
    return {"id": id, "location_id": loc, "type": cell[0], "morph_cluster": cell[1],
            "palette_flavor": cell[2], "render_style": style, "cell": list(cell),
            "p_ge3": p_ge3, "passed": True, "head": B.head_for_style(style)}


def test_release_floors_exclude_subfloor_and_short_fill(tmp_path):
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    # 4 pool-admitted rows in distinct cells: one eligible + one sub-floor per head.
    recs = [
        _gate_rec("em_0", "l0", "smooth", 0.95, ("mandelbrot", "m#0", "k16:1", "smooth")),  # ≥0.90 ✓
        _gate_rec("em_1", "l1", "smooth", 0.80, ("mandelbrot", "m#1", "k16:2", "smooth")),  # <0.90 inv
        _gate_rec("em_2", "l2", "tia",    0.60, ("mandelbrot", "m#2", "k16:3", "tia")),     # ≥0.50 ✓
        _gate_rec("em_3", "l3", "tia",    0.30, ("mandelbrot", "m#3", "k16:4", "tia")),     # <0.50 inv
    ]
    for r in recs:
        eng.pool.append(r)
    elig = {r["id"] for r in eng.release_eligible()}
    assert elig == {"em_0", "em_2"}                       # sub-floor rows banked as inventory
    selected, _log = eng.select_release()
    sel_ids = {e["_rec"]["id"] for e in selected}
    assert sel_ids == {"em_0", "em_2"}                    # never dips below the floor to fill N=5
    sf = eng.release_short_fill
    assert (sf["requested"], sf["eligible"], sf["selected"], sf["short_by"]) == (5, 2, 2, 3)
    # head-split: one smooth (wallpaper) + one strange (mining), never compared in one step
    assert eng.release_split["smooth_selected"] == 1 and eng.release_split["strange_selected"] == 1


def test_release_floor_per_head_boundary(tmp_path):
    # a mining tile at exactly 0.50 is eligible; a smooth at 0.50 is NOT (its floor is 0.90).
    eng = B.EmissionDiversity(_args(tmp_path))
    eng.embs = {}
    eng.pool.append(_gate_rec("em_0", "l0", "tia", 0.50, ("mandelbrot", "m#0", "k16:1", "tia")))
    eng.pool.append(_gate_rec("em_1", "l1", "smooth", 0.50, ("mandelbrot", "m#1", "k16:2", "smooth")))
    assert {r["id"] for r in eng.release_eligible()} == {"em_0"}


def test_multi_ledger_intake_dedup_and_source_tag(tmp_path):
    l1 = tmp_path / "a.jsonl"
    l2 = tmp_path / "b.jsonl"
    l1.write_text(json.dumps(_row("shared", "v7")) + "\n"
                  + json.dumps(_row("only_a", "v7")) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(_row("shared", "v7")) + "\n"      # dup id across ledgers
                  + json.dumps(_row("only_b", "v7")) + "\n", encoding="utf-8")
    eng = B.EmissionDiversity(_args(tmp_path, ledger=[str(l1), str(l2)]))
    rows = eng._load_all_admitted()
    ids = [r["id"] for r in rows]
    assert ids == ["shared", "only_a", "only_b"]            # dedup, first-ledger wins
    src = {r["id"]: r["_source_ledger"] for r in rows}
    assert src["shared"].endswith("a.jsonl") and src["only_b"].endswith("b.jsonl")


def test_intake_raises_on_run_scoped_id_collision(tmp_path):
    """Same id, DIFFERENT location across ledgers = run-scoped-id collision: RAISE, don't
    silently drop a distinct wallpaper (union-by-id would). Same id + same location dedups."""
    def _at(id, cx):
        r = _row(id, "v7")
        r["outcome_cx"] = cx
        return r
    l1 = tmp_path / "a.jsonl"
    l2 = tmp_path / "b.jsonl"
    l1.write_text(json.dumps(_at("st_x", -0.5)) + "\n", encoding="utf-8")
    l2.write_text(json.dumps(_at("st_x", 0.9)) + "\n", encoding="utf-8")   # SAME id, other coord
    eng = B.EmissionDiversity(_args(tmp_path, ledger=[str(l1), str(l2)]))
    with pytest.raises(SystemExit, match="COLLISION"):
        eng._load_all_admitted()
    # same id + identical location is NOT a collision (legitimate cross-ledger overlap)
    l2.write_text(json.dumps(_at("st_x", -0.5)) + "\n", encoding="utf-8")
    eng2 = B.EmissionDiversity(_args(tmp_path, ledger=[str(l1), str(l2)]))
    assert [r["id"] for r in eng2._load_all_admitted()] == ["st_x"]


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
