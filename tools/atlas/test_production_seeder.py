#!/usr/bin/env python
"""Unit tests for the atlas production seeder (control on the pure predicates + the
q3-density rejection rule + a ledger round-trip). The smoke eyeball is the visual gate.

  uv run pytest tools/atlas/test_production_seeder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402


# --------------------------------------------------------------------------- #
# fw-relative dedup predicate (cloud hygiene)
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


def test_is_distinct_against_cloud():
    cloud = [
        {"id": "a", "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 1.0},
        {"id": "b", "outcome_cx": 10.0, "outcome_cy": 0.0, "outcome_fw": 1e-3},
    ]
    d, dup = ps.is_distinct(0.5, 0.0, 1.0, cloud, k=1.5)   # within a's radius
    assert d is False and dup == "a"
    d, dup = ps.is_distinct(3.0, 3.0, 1e-3, cloud, k=1.5)  # far from both
    assert d is True and dup is None


# --------------------------------------------------------------------------- #
# q3-density rejection rule (the coverage-control mechanism)
# --------------------------------------------------------------------------- #
def _cloud(pts):
    """Build a cloud of point members at (cx, cy) with tiny fw (points, not zoom)."""
    return [{"id": f"m{i}", "outcome_cx": x, "outcome_cy": y, "outcome_fw": 1e-9}
            for i, (x, y) in enumerate(pts)]


def test_count_within_radius():
    cloud = _cloud([(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.5, 0.5)])
    # radius 0.20 around the origin catches the first three, not the far corner.
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.20) == 3
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.08) == 2   # only (0,0) + (0.05,0)
    assert ps.count_within([], 0.0, 0.0, radius=0.20) == 0


def test_rejection_rule_dense_vs_open(monkeypatch):
    monkeypatch.setattr(ps, "REJECT_RADIUS", 0.20)
    monkeypatch.setattr(ps, "Q3_DENSITY_CAP", 5)
    # 5 distinct members clustered at the origin -> a seed there hits the cap -> reject.
    dense = _cloud([(0.0, 0.0), (0.03, 0.0), (0.0, 0.03), (-0.03, 0.0), (0.0, -0.03)])
    assert ps.count_within(dense, 0.0, 0.0, ps.REJECT_RADIUS) >= ps.Q3_DENSITY_CAP
    # a seed in open space (far from every member) is under the cap -> accept.
    assert ps.count_within(dense, 5.0, 5.0, ps.REJECT_RADIUS) < ps.Q3_DENSITY_CAP


def test_near_dup_does_not_double_count_a_region():
    """A near-dup outcome does not enter the cloud, so it can't push a region over the
    density cap by being counted twice. build_cloud dedups by 1.5*max(fw)."""
    rows = [
        {"id": "a", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # near-dup of a (within 1.5*max(fw)=0.015): must NOT create a second member.
        {"id": "a2", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.005, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # genuinely distinct q3 place.
        {"id": "b", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.10, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # class-2 and guard-failed rows never enter the q3 cloud.
        {"id": "c2", "guard_pass": True, "decoded_class": 2,
         "outcome_cx": 0.11, "outcome_cy": 0.0, "outcome_fw": 0.01},
        {"id": "gf", "guard_pass": False, "decoded_class": None,
         "outcome_cx": 0.12, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    cloud = ps.build_cloud(rows, "mandelbrot")     # keyless rows default to mandelbrot
    ids = {m["id"] for m in cloud}
    assert ids == {"a", "b"}                       # a2 deduped; c2/gf excluded
    # the region around a holds exactly ONE counted member, not two.
    assert ps.count_within(cloud, 0.0, 0.0, radius=0.02) == 1


# --------------------------------------------------------------------------- #
# ledger round-trip (write -> reload -> rows + feats preserved; cross-run cumulative)
# --------------------------------------------------------------------------- #
def _isolate_ledgers(tmp_path, monkeypatch):
    d = tmp_path / "discovery"
    monkeypatch.setattr(ps, "DISCOVERY_DIR", d)
    monkeypatch.setattr(ps, "OUTCOME_LEDGER", d / "outcome_ledger.jsonl")
    monkeypatch.setattr(ps, "OUTCOME_FEATS", d / "outcome_feats.npz")
    monkeypatch.setattr(ps, "PROBE_REJECTS", d / "probe_rejects.jsonl")
    return d


def test_ledger_round_trip(tmp_path, monkeypatch):
    _isolate_ledgers(tmp_path, monkeypatch)
    led = ps.Ledgers()
    row_q3 = {"id": "m_x_000001", "distinct": True, "guard_pass": True, "decoded_class": 3,
              "outcome_cx": 0.1, "outcome_cy": 0.2, "outcome_fw": 0.01, "k3": 1.9}
    row_dup = {"id": "m_x_000002", "distinct": False, "dup_of": "m_x_000001",
               "guard_pass": True, "decoded_class": 3,
               "outcome_cx": 0.1, "outcome_cy": 0.2, "outcome_fw": 0.01, "k3": 1.8}
    led.append_outcome(row_q3, np.arange(1280, dtype=np.float32))
    led.append_outcome(row_dup, np.ones(1280, dtype=np.float32))
    led.save_feats()

    led2 = ps.Ledgers()   # fresh reload
    assert led2.n_outcomes_logged == 2
    assert len(led2.harvested) == 2                        # both guard_pass
    cloud = ps.build_cloud(led2.rows, "mandelbrot")        # keyless rows default to mandelbrot
    assert [m["id"] for m in cloud] == ["m_x_000001"]      # dup collapses to one place
    assert "m_x_000001" in led2.feats and led2.feats["m_x_000001"].shape == (1280,)
    assert float(led2.feats["m_x_000001"][5]) == 5.0       # feature preserved

    # cross-run cumulative: a second run appends and reloads with combined state.
    led2.append_outcome({"id": "m_x_000003", "distinct": True, "guard_pass": True,
                         "decoded_class": 3, "outcome_cx": 9.0, "outcome_cy": 9.0,
                         "outcome_fw": 0.01, "k3": 2.0}, None)
    assert ps.Ledgers().n_outcomes_logged == 3


def test_build_cloud_excludes_pre_decode_rows():
    """No historical backfill: rows predating the decoded_class field (no key) never enter
    the q3 cloud — only rows the new pipeline logged with decoded_class == 3 do."""
    rows = [
        # historical row: guard_pass but no decoded_class key -> excluded.
        {"id": "old", "guard_pass": True,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        # new-pipeline q3 row -> included.
        {"id": "new", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 5.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    assert [m["id"] for m in ps.build_cloud(rows, "mandelbrot")] == ["new"]


def test_build_cloud_partitions_by_family():
    """The `family` arg is the correctness fix: cross-family outcomes at the SAME (cx, cy)
    are different parameter planes and must never interact. build_cloud returns only the
    active partition; keyless rows count as mandelbrot."""
    rows = [
        # same coords, three different planes -> each partition sees exactly its own row.
        {"id": "m", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},              # keyless
        {"id": "j", "family": "julia", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
        {"id": "mb", "family": "multibrot_d3", "guard_pass": True, "decoded_class": 3,
         "outcome_cx": 0.0, "outcome_cy": 0.0, "outcome_fw": 0.01},
    ]
    assert [m["id"] for m in ps.build_cloud(rows, "mandelbrot")] == ["m"]
    assert [m["id"] for m in ps.build_cloud(rows, "julia")] == ["j"]
    assert [m["id"] for m in ps.build_cloud(rows, "multibrot_d3")] == ["mb"]
    assert ps.build_cloud(rows, "phoenix") == []   # no member in that partition
