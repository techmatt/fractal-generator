#!/usr/bin/env python
"""Cap-logic unit tests for the atlas production seeder (the control on the cap logic;
the smoke eyeball is the visual gate). Pure predicates + a small ledger round-trip +
one atlas-backed backfill integration.

  uv run pytest tools/atlas/test_production_seeder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402


# --------------------------------------------------------------------------- #
# fw-relative dedup predicate
# --------------------------------------------------------------------------- #
def test_near_dup_within_and_outside():
    # B at origin, fw=1.0 -> dedup radius = 1.5*max(fw).
    assert ps.near_dup(1.4, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is True    # 1.4 < 1.5
    assert ps.near_dup(1.6, 0.0, 1.0, 0.0, 0.0, 1.0, k=1.5) is False   # 1.6 > 1.5


def test_near_dup_same_center_different_zoom_merges():
    # (nearly) same center, very different fw -> same PLACE (max(fw) dominates).
    assert ps.near_dup(1e-6, 0.0, 1e-3, 0.0, 0.0, 2.0, k=1.5) is True


def test_near_dup_distant_pair_distinct():
    # genuinely distant centers at small fw -> distinct.
    assert ps.near_dup(5.0, 5.0, 1e-3, 0.0, 0.0, 1e-3, k=1.5) is False


def test_is_distinct_against_harvested():
    harvested = [
        {"id": "a", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1.0},
        {"id": "b", "outcome_cx": 10.0, "outcome_cy": 0.0, "outcome_fw": 1e-3},
    ]
    d, dup = ps.is_distinct(0.5, 0.0, 1.0, harvested, k=1.5)   # within a's radius
    assert d is False and dup == "a"
    d, dup = ps.is_distinct(3.0, 3.0, 1e-3, harvested, k=1.5)  # far from both
    assert d is True and dup is None


# --------------------------------------------------------------------------- #
# cell-saturation predicate (distinct >= 10 OR launches >= 20, not before)
# --------------------------------------------------------------------------- #
def test_cell_saturation_thresholds():
    assert ps.cell_saturated({"launches": 19, "distinct": 9}) is False
    assert ps.cell_saturated({"launches": 20, "distinct": 0}) is True      # launch cap
    assert ps.cell_saturated({"launches": 0, "distinct": 10}) is True      # distinct cap
    assert ps.cell_saturated({"launches": 5, "distinct": 5}) is False


def test_cell_launch_capped_thresholds():
    assert ps.cell_launch_capped({"launches": 19}) is False
    assert ps.cell_launch_capped({"launches": 20}) is True
    # distinct alone never launch-caps (that only affects EXPLOIT via saturation).
    assert ps.cell_launch_capped({"launches": 0, "distinct": 999}) is False


def test_cell_saturation_custom_caps():
    assert ps.cell_saturated({"launches": 3, "distinct": 0}, seed_cap=3, distinct_cap=2) is True
    assert ps.cell_saturated({"launches": 0, "distinct": 2}, seed_cap=3, distinct_cap=2) is True
    assert ps.cell_saturated({"launches": 2, "distinct": 1}, seed_cap=3, distinct_cap=2) is False


# --------------------------------------------------------------------------- #
# ledger round-trip (write -> reload -> state preserved; cross-run cumulative)
# --------------------------------------------------------------------------- #
def _isolate_ledgers(tmp_path, monkeypatch):
    d = tmp_path / "discovery"
    monkeypatch.setattr(ps, "DISCOVERY_DIR", d)
    monkeypatch.setattr(ps, "OUTCOME_LEDGER", d / "outcome_ledger.jsonl")
    monkeypatch.setattr(ps, "OUTCOME_FEATS", d / "outcome_feats.npz")
    monkeypatch.setattr(ps, "CELL_LEDGER", d / "cell_ledger.json")
    monkeypatch.setattr(ps, "PROBE_REJECTS", d / "probe_rejects.jsonl")
    return d


def test_ledger_round_trip(tmp_path, monkeypatch):
    _isolate_ledgers(tmp_path, monkeypatch)
    led = ps.Ledgers()
    led.bump_launch(7)
    led.bump_launch(7)
    led.bump_distinct(7)
    row_d = {"id": "m_x_000001", "distinct": True, "outcome_cx": 0.1, "outcome_cy": 0.2,
             "outcome_fw": 0.01, "seed_cell": 7, "k3": 1.3}
    row_u = {"id": "m_x_000002", "distinct": False, "dup_of": "m_x_000001",
             "outcome_cx": 0.1, "outcome_cy": 0.2, "outcome_fw": 0.01, "k3": 1.1}
    led.append_outcome(row_d, np.arange(1280, dtype=np.float32))
    led.append_outcome(row_u, np.ones(1280, dtype=np.float32))
    led.save_cells(); led.save_feats()

    led2 = ps.Ledgers()   # fresh reload
    assert led2.cell_state(7)["launches"] == 2
    assert led2.cell_state(7)["distinct"] == 1
    assert led2.n_outcomes_logged == 2                     # both rows counted
    assert len(led2.harvested) == 1                        # only the distinct one
    assert led2.harvested[0]["id"] == "m_x_000001"
    assert "m_x_000001" in led2.feats and led2.feats["m_x_000001"].shape == (1280,)
    assert float(led2.feats["m_x_000001"][5]) == 5.0       # feature preserved

    # cross-run cumulative: a second run appends and reloads with combined state.
    led2.bump_launch(7)
    led2.save_cells()
    assert ps.Ledgers().cell_state(7)["launches"] == 3


# --------------------------------------------------------------------------- #
# backfill: a saturated EXPLOIT cell yields an EXPLORE draw
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (ROOT / "data" / "atlas" / "atlas_v1.npz").exists(),
                    reason="atlas_v1.npz required for the backfill integration")
def test_backfill_saturated_exploit_to_explore(tmp_path, monkeypatch):
    _isolate_ledgers(tmp_path, monkeypatch)
    from atlas import Atlas
    atlas = Atlas.load()
    rng = np.random.default_rng(0)
    # small cloud keeps it fast; no native seeds so the fallback is pure explore.
    monkeypatch.setattr(ps, "N_CLOUD", 3000)
    fw_pool = np.array([0.01, 0.02, 0.05], float)
    prop = ps.Proposer(atlas, fw_pool, native_seeds=[], rng=rng)

    led = ps.Ledgers()
    # Saturate (distinct-cap) EVERY cell the exploit queue would land in — leave
    # launches at 0 so explore cells (which ignore saturation) stay drawable.
    for i in prop.exploit_q:
        cid = ps.seed_cell(prop.cloud[i, 0], prop.cloud[i, 1], atlas.mask_bounds)
        led.cells[str(cid)] = {"launches": 0, "distinct": ps.OUTCOME_DISTINCT_CAP, "saturated": True}

    props, mix = prop.draw_batch(led, n_batch=10)
    # n_native = round(0.05*10) = 0 ; n_exploit = 8 ; n_explore = 2.
    assert mix["realized"]["exploit"] == 0          # every exploit cell was saturated
    assert mix["backfills"] >= 8                     # each rejected exploit backfilled
    assert mix["realized"]["explore"] >= 1          # forced exploration happened
    # no placed proposal sits in a saturated exploit cell
    for p in props:
        st = led.cell_state(p["seed_cell"])
        if p["mix_source"] == "exploit":
            assert not ps.cell_saturated(st)
