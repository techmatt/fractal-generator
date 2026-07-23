"""Tests for the family-level deficit scheduler (tools/atlas/deficit_scheduler.py).

Torch-free / render-free: the order-book projection, the distinct-look tally (against a
hand-built embedding set), the price update + attempt-cap fire/redistribute, and the
STRUCTURAL guarantee that the cross-partition pop decision is a pure function of deficits
and prices (a cross-partition p_good comparison is impossible by construction).

Run: uv run pytest tools/atlas/test_deficit_scheduler.py -q
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "atlas"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import deficit_scheduler as D                    # noqa: E402
from tools.emission import cells as C            # noqa: E402


PARTS = ["mandelbrot", "multibrot5", "julia:mandelbrot", "julia:multibrot5"]


def _unit(vec):
    v = np.asarray(vec, np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _emb(seed, dim=D.EMB_DIM):
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim))


# --------------------------------------------------------------------------- #
# 1. Order book: deficit computed from a measure file.
# --------------------------------------------------------------------------- #
def test_projection_uniform_is_flat():
    tm = C.TargetMeasure.from_config({"mode": "uniform"})
    obs = [(p, f"{p}#0") for p in PARTS]
    mg = D.project_type_marginals(tm, obs, PARTS)
    assert abs(sum(mg.values()) - 1.0) < 1e-9
    for v in mg.values():
        assert abs(v - 0.25) < 1e-9


def test_projection_type_override_skews():
    tm = C.TargetMeasure.from_config(
        {"weight_overrides": [{"match": {"fractal_type": ["multibrot5"]}, "weight": 4.0}]})
    obs = [(p, f"{p}#0") for p in PARTS]
    mg = D.project_type_marginals(tm, obs, PARTS)
    assert abs(sum(mg.values()) - 1.0) < 1e-9
    assert mg["multibrot5"] > mg["mandelbrot"]
    # 4 / (4+1+1+1) for the boosted type; 1/7 each for the rest.
    assert abs(mg["multibrot5"] - 4.0 / 7.0) < 1e-9
    assert abs(mg["mandelbrot"] - 1.0 / 7.0) < 1e-9


def test_projection_ignores_unresolved_source_tag_override():
    # The discovery-side projection does NOT resolve source-tag overrides (it has no intake
    # loc->tag map), so a source_tag override must be a NO-OP here — never crash, never skew
    # per-type marginals. This is the gate that the classic-phoenix source-tag rewrite leaves
    # the deficit scheduler's per-type marginals unaffected.
    obs = [(p, f"{p}#0") for p in PARTS]
    base = D.project_type_marginals(C.TargetMeasure.from_config({"mode": "uniform"}), obs, PARTS)
    tm = C.TargetMeasure.from_config({"weight_overrides": [
        {"match": {"source_tag": ["classic_phoenix"]}, "weight": 1.9}]})
    got = D.project_type_marginals(tm, obs, PARTS)      # must not raise
    for p in PARTS:
        assert abs(got[p] - base[p]) < 1e-12


def test_projection_flavor_style_cancel():
    # an override on a FREE axis (palette_flavor) multiplies the same cell subset for every
    # type, so it must NOT change the per-type marginal (cancels under normalization).
    tm = C.TargetMeasure.from_config(
        {"weight_overrides": [{"match": {"palette_flavor": ["k16:5"]}, "weight": 9.0}]})
    obs = [(p, f"{p}#0") for p in PARTS]
    mg_dummy = D.project_type_marginals(tm, obs, PARTS)
    mg_real = D.project_type_marginals(tm, obs, PARTS,
                                       flavors=["k16:5", "k16:6"], styles=["smooth", "tia"])
    for p in PARTS:
        assert abs(mg_dummy[p] - 0.25) < 1e-9
        assert abs(mg_real[p] - 0.25) < 1e-9


def test_projection_independent_of_cluster_count(tmp_path):
    # Campaign-2 preflight fix: a type's marginal is set by its MULTIPLIER, INDEPENDENT of how
    # many observed morph clusters it has (occupancy belongs on the deficit's pool side, not the
    # target side). Uniform base with 3 mandelbrot clusters vs 1 multibrot5 cluster => still 50/50.
    tm = C.TargetMeasure.from_config({"mode": "uniform"})
    obs = [("mandelbrot", "mandelbrot#0"), ("mandelbrot", "mandelbrot#1"),
           ("mandelbrot", "mandelbrot#2"), ("multibrot5", "multibrot5#0")]
    parts = ["mandelbrot", "multibrot5"]
    mg = D.project_type_marginals(tm, obs, parts)
    assert abs(mg["mandelbrot"] - 0.5) < 1e-9
    assert abs(mg["multibrot5"] - 0.5) < 1e-9


def test_projection_multiplier_beats_cluster_count(tmp_path):
    # The exact campaign-2 inversion the fix targets: a high-multiplier type with FEW clusters
    # must out-weight a low-multiplier type with MANY clusters. julia:mandelbrot (2.5x, 4 obs
    # clusters) must exceed mandelbrot (1.2x, 102 obs clusters) — the reverse of the buggy
    # count-weighted projection, which crushed julia:mandelbrot to <2%.
    tm = C.TargetMeasure.from_config({"weight_overrides": [
        {"match": {"fractal_type": ["julia:mandelbrot"]}, "weight": 2.5},
        {"match": {"fractal_type": ["mandelbrot"]}, "weight": 1.2}]})
    obs = ([("mandelbrot", f"mandelbrot#{i}") for i in range(102)]
           + [("julia:mandelbrot", f"julia:mandelbrot#{i}") for i in range(4)])
    parts = ["mandelbrot", "julia:mandelbrot"]
    mg = D.project_type_marginals(tm, obs, parts)
    assert mg["julia:mandelbrot"] > mg["mandelbrot"]
    # share ∝ multiplier: 2.5/(2.5+1.2) vs 1.2/(2.5+1.2), regardless of the 102-vs-4 counts.
    assert abs(mg["julia:mandelbrot"] - 2.5 / 3.7) < 1e-9
    assert abs(mg["mandelbrot"] - 1.2 / 3.7) < 1e-9


def test_deficit_from_measure_file(tmp_path):
    # write a skewed measure file, build a scheduler, check deficits with an empty tally
    # equal the projected target (look_frac all zero).
    mfile = tmp_path / "measure.json"
    mfile.write_text(json.dumps(
        {"weight_overrides": [{"match": {"fractal_type": ["multibrot5"]}, "weight": 3.0}]}))
    sch = D.DeficitScheduler(PARTS, tmp_path, target_path=mfile, prices_path=tmp_path / "none.json")
    defs = sch.deficits()
    assert abs(defs["multibrot5"] - sch.target_frac["multibrot5"]) < 1e-12
    assert defs["multibrot5"] > defs["mandelbrot"]


# --------------------------------------------------------------------------- #
# 2. Distinct-look tally against a hand-built embedding set.
# --------------------------------------------------------------------------- #
def test_distinct_look_tally(tmp_path):
    t = D.DistinctLookTally(tmp_path / "looks.npz")
    a = _emb(1)
    # first look is always distinct.
    assert t.add("mandelbrot", a) is True
    # a near-identical look (cos >= 0.974) is NOT a new distinct look.
    near = _unit(a + 0.001 * _emb(999))
    assert float(near @ a) >= 0.974
    assert t.add("mandelbrot", near) is False
    # a clearly different look IS distinct.
    b = _emb(2)
    assert float(b @ a) < 0.974
    assert t.add("mandelbrot", b) is True
    assert t.count("mandelbrot") == 2
    # partitions are independent: the same vector is a fresh distinct look elsewhere.
    assert t.add("multibrot5", a) is True
    assert t.count("multibrot5") == 1
    assert t.total() == 3


def test_tally_persist_roundtrip(tmp_path):
    p = tmp_path / "looks.npz"
    t = D.DistinctLookTally(p)
    t.add("mandelbrot", _emb(1))
    t.add("mandelbrot", _emb(2))
    t.add("julia:mandelbrot", _emb(3))
    t.save()
    t2 = D.DistinctLookTally(p)
    assert t2.counts() == {"mandelbrot": 2, "julia:mandelbrot": 1}
    # the reloaded set still dedups against its persisted members.
    assert t2.add("mandelbrot", _emb(1)) is False


# --------------------------------------------------------------------------- #
# 3. Price update (online EMA of minutes-per-distinct-look).
# --------------------------------------------------------------------------- #
def test_price_ema_update():
    pm = D.PriceModel(PARTS, {"seed_price_min": 3.0, "price_ema": 0.5, "cap_minutes": 100})
    assert pm.price["mandelbrot"] == 3.0
    pm.charge("mandelbrot", 5.0)      # 5 active minutes, no look yet
    pm.record_look("mandelbrot")      # a look after 5 min -> price EMA toward 5
    assert abs(pm.price["mandelbrot"] - (0.5 * 3.0 + 0.5 * 5.0)) < 1e-9
    assert pm.min_since_look["mandelbrot"] == 0.0


# --------------------------------------------------------------------------- #
# 4. Attempt cap fires + redistributes; re-opens on a look.
# --------------------------------------------------------------------------- #
def test_attempt_cap_fire_and_redistribute():
    pm = D.PriceModel(PARTS, {"cap_minutes": 10.0})
    assert pm.charge("mandelbrot", 6.0) is False
    assert "mandelbrot" not in pm.capped
    assert pm.charge("mandelbrot", 6.0) is True    # 12 >= 10 with zero looks -> capped
    assert "mandelbrot" in pm.capped
    # a distinct look re-opens the partition (productive again).
    pm.record_look("mandelbrot")
    assert "mandelbrot" not in pm.capped
    assert pm.min_since_look["mandelbrot"] == 0.0


def test_cap_redistributes_serving():
    # a capped partition is excluded from the pop candidates -> demand redistributes.
    rng = np.random.default_rng(0)
    deficits = {"mandelbrot": 0.9, "multibrot5": 0.1}
    prices = {"mandelbrot": 1.0, "multibrot5": 1.0}
    servable = {"mandelbrot", "multibrot5"}
    # uncapped: highest price-weighted deficit (mandelbrot) wins with no exploration.
    assert D.choose_partition(deficits, prices, set(), servable, rng, explore_floor=0.0) == "mandelbrot"
    # capped: mandelbrot excluded, so multibrot5 is served instead.
    assert D.choose_partition(deficits, prices, {"mandelbrot"}, servable, rng,
                              explore_floor=0.0) == "multibrot5"


# --------------------------------------------------------------------------- #
# 5. Cross-partition p_good comparison is structurally impossible.
# --------------------------------------------------------------------------- #
def test_choose_partition_signature_has_no_pgood():
    sig = inspect.signature(D.choose_partition)
    params = set(sig.parameters)
    assert params == {"deficits", "prices", "capped", "servable", "rng", "explore_floor"}
    # nothing p_good / node / score-shaped in the pop decision's inputs.
    for bad in ("p_good", "pgood", "eord", "score", "node", "nodes", "frontier", "priority"):
        assert bad not in params


def test_choose_partition_ignores_everything_but_deficit_and_price():
    # price-weighted deficit only: with equal deficits, the cheaper partition wins; with
    # equal prices, the larger deficit wins. No other signal can enter (there is none to pass).
    rng = np.random.default_rng(0)
    serv = {"a", "b"}
    assert D.choose_partition({"a": 0.5, "b": 0.5}, {"a": 2.0, "b": 1.0}, set(), serv, rng, 0.0) == "b"
    assert D.choose_partition({"a": 0.8, "b": 0.2}, {"a": 1.0, "b": 1.0}, set(), serv, rng, 0.0) == "a"


def test_choose_partition_none_when_all_capped():
    rng = np.random.default_rng(0)
    assert D.choose_partition({"a": 1.0}, {"a": 1.0}, {"a"}, {"a"}, rng, 0.0) is None
    assert D.choose_partition({"a": 1.0}, {"a": 1.0}, set(), set(), rng, 0.0) is None


# --------------------------------------------------------------------------- #
# 6. Julia routing: twin deficit with an empty queue buys c-plane work.
# --------------------------------------------------------------------------- #
def test_julia_routing_folds_into_cplane(tmp_path):
    sch = D.DeficitScheduler(["multibrot5", "julia:multibrot5"], tmp_path,
                             target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    # force a large julia deficit, zero c-plane deficit.
    sch.target_frac = {"multibrot5": 0.0, "julia:multibrot5": 1.0}
    # julia queue empty, c-plane has nodes -> julia demand routes onto the c-plane parent.
    queue_lens = {"multibrot5": 5, "julia:multibrot5": 0}
    eff = sch.effective_deficits(queue_lens)
    assert eff["multibrot5"] > sch.deficits()["multibrot5"]     # boosted by the twin
    part = sch.pick_partition(queue_lens, np.random.default_rng(0))
    assert part == "multibrot5"           # serving the parent to buy julia looks


def test_julia_routing_no_double_count_when_twin_servable(tmp_path):
    sch = D.DeficitScheduler(["multibrot5", "julia:multibrot5"], tmp_path,
                             target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    sch.target_frac = {"multibrot5": 0.0, "julia:multibrot5": 1.0}
    # julia queue NON-empty -> it competes on its own, no fold onto the parent.
    queue_lens = {"multibrot5": 5, "julia:multibrot5": 3}
    eff = sch.effective_deficits(queue_lens)
    assert abs(eff["multibrot5"] - sch.deficits()["multibrot5"]) < 1e-12


# --------------------------------------------------------------------------- #
# 7. Root allocation follows deficit and sums to the batch.
# --------------------------------------------------------------------------- #
def test_root_allocation_sums_and_favors_deficit(tmp_path):
    sch = D.DeficitScheduler(["mandelbrot", "multibrot5"], tmp_path,
                             target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    sch.target_frac = {"mandelbrot": 0.9, "multibrot5": 0.1}
    rng = np.random.default_rng(0)
    tot = np.zeros(2)
    for _ in range(200):
        a = sch.root_allocation(["mandelbrot", "multibrot5"], 32, rng)
        assert sum(a.values()) == 32          # every draw sums to the batch
        tot += [a["mandelbrot"], a["multibrot5"]]
    assert tot[0] > tot[1]                     # the high-deficit family draws more roots


# --------------------------------------------------------------------------- #
# 8. Scheduler state round-trip (resume safety).
# --------------------------------------------------------------------------- #
def test_seed_from_library_and_resume_safety(tmp_path):
    # Seeding pre-loads the tally with library looks; deficits then measure library-wide
    # scarcity. It seeds ONLY when empty (resume-safe / idempotent) and persists immediately.
    parts = ["mandelbrot", "julia:mandelbrot"]
    embs = {"mandelbrot": np.stack([_emb(10), _emb(11), _emb(12)]),      # 3 distinct looks
            "julia:mandelbrot": np.stack([_emb(20)]),                    # 1 distinct look
            "multibrot5": np.stack([_emb(30)])}                          # untracked -> ignored
    sch = D.DeficitScheduler(parts, tmp_path,
                             target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    seeded = sch.seed_from_library(embs)
    assert seeded == {"mandelbrot": 3, "julia:mandelbrot": 1}
    assert sch.tally.counts() == {"mandelbrot": 3, "julia:mandelbrot": 1}
    assert (tmp_path / "distinct_looks.npz").exists()      # persisted before any batch
    # library-wide scarcity: an admission duplicating a seeded look is NOT a new distinct look.
    assert sch.on_admission("mandelbrot", _emb(10)) is False
    # re-seeding is a no-op (tally non-empty) — never double-counts on resume.
    assert sch.seed_from_library(embs) == {}
    # a genuinely fresh scheduler over the SAME dir reloads the seed from npz and still no-ops.
    sch2 = D.DeficitScheduler(parts, tmp_path,
                              target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    assert sch2.tally.total() == 4
    assert sch2.seed_from_library(embs) == {}


def test_scheduler_state_roundtrip(tmp_path):
    sch = D.DeficitScheduler(PARTS, tmp_path,
                             target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    sch.on_admission("mandelbrot", _emb(1))
    sch.on_admission("mandelbrot", _emb(2))
    sch.charge("multibrot5", 5.0)
    sch.prices.record_look("multibrot5")
    st = sch.state_dict()
    sch.save()

    sch2 = D.DeficitScheduler(PARTS, tmp_path,
                              target_path=tmp_path / "none.json", prices_path=tmp_path / "none.json")
    sch2.load_state(st, reopen_caps=True)
    assert sch2.tally.counts() == {"mandelbrot": 2}          # reloaded from npz
    assert abs(sch2.prices.price["multibrot5"] - sch.prices.price["multibrot5"]) < 1e-9
