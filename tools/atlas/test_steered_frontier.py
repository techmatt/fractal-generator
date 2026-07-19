#!/usr/bin/env python
"""Unit tests for the steered-frontier v1.1 priority terms + the keeper cut.

Pure / fast — no render, no GPU, no binary. The control for the two new priority terms
(morph-novelty + depth) and the acceptance guarantee that BOTH coefficients at zero
reproduce the pilot priority exactly, plus the F0.5 keeper-cut metric math.

  uv run pytest tools/atlas/test_steered_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import steered_frontier as sf   # noqa: E402
import keeper_cut as kc         # noqa: E402


# =========================================================================== #
# novelty penalty — the re-anchored ramp.
# =========================================================================== #
def test_novelty_zero_when_disabled():
    # lambda_m == 0 -> identically zero for ANY cosine (incl. a perfect dup).
    for cos in (-1.0, 0.0, 0.85, 0.974, 1.0):
        assert sf.novelty_penalty(cos, 0.0, 0.80, 0.97) == 0.0


def test_novelty_ramp_anchors():
    lo, hi, lam = 0.80, 0.90, 0.5
    assert sf.novelty_penalty(lo - 0.01, lam, lo, hi) == 0.0          # below lo: zero
    assert sf.novelty_penalty(lo, lam, lo, hi) == 0.0                 # at lo: zero
    assert abs(sf.novelty_penalty((lo + hi) / 2, lam, lo, hi) - lam / 2) < 1e-9  # midpoint: half
    assert abs(sf.novelty_penalty(hi, lam, lo, hi) - lam) < 1e-9      # at hi: full
    assert abs(sf.novelty_penalty(1.0, lam, lo, hi) - lam) < 1e-9     # above hi: clamped full


# =========================================================================== #
# priority decomposition — the acceptance guarantee.
# =========================================================================== #
def test_priority_reduces_to_pilot_at_zero_coeffs():
    # With lambda_m == 0 AND beta == 0, priority == pilot's eord + gumbel - dup_pen, and the
    # novelty/depth terms vanish REGARDLESS of cos_max or depth (so the frontier order, hence
    # the whole run, is byte-identical to the pilot).
    for eord, g, dup, cos, depth in [
        (1.3, 0.02, 0.0, 0.99, 7), (0.4, -0.05, 0.8, 0.5, 2), (1.9, 0.1, 0.3, 0.9741, 13),
    ]:
        prio, terms = sf.priority_terms(eord, g, dup, cos, 0.0, 0.0, depth, 0.85, 0.974)
        assert prio == eord + g - dup
        assert terms["nov_pen"] == 0.0
        assert terms["depth_bonus"] == 0.0
        assert terms["priority"] == prio


def test_priority_full_terms_contribute():
    # Both terms live: a near-dup at high depth is penalised by novelty and lifted by depth.
    lo, hi, lam, beta = 0.80, 0.90, 0.5, 0.02
    prio, terms = sf.priority_terms(1.0, 0.0, 0.0, hi, lam, beta, 10, lo, hi)
    assert abs(terms["nov_pen"] - lam) < 1e-9            # full novelty penalty
    assert abs(terms["depth_bonus"] - beta * 10) < 1e-9  # depth bonus
    assert abs(prio - (1.0 - lam + beta * 10)) < 1e-9


# =========================================================================== #
# anchor resolution — CLI override > file > fallback; degenerate ramp guarded.
# =========================================================================== #
def test_anchor_cli_override_and_guard():
    lo, hi, src = sf.load_morph_anchors(cli_lo=0.6, cli_hi=0.95)
    assert (lo, hi) == (0.6, 0.95) and "cli_lo" in src and "cli_hi" in src
    # hi <= lo is repaired to a positive-width ramp.
    lo2, hi2, _ = sf.load_morph_anchors(cli_lo=0.9, cli_hi=0.8)
    assert hi2 > lo2


# =========================================================================== #
# keeper cut — F0.5 metric math + calibration gate.
# =========================================================================== #
def test_fbeta_precision_weighted():
    # F0.5 weights precision over recall: at equal-ish P/R it sits between them, closer to P.
    p_heavy = kc.prf_beta(tp=8, fp=1, fn=4)   # P=0.889 R=0.667
    prec, rec, f = p_heavy
    assert prec > rec
    # F0.5 must lie strictly between recall and precision (precision-leaning).
    assert rec < f < prec


def test_fbeta_beta_half_formula():
    # explicit check against the closed form (1+0.25)*P*R / (0.25*P + R).
    prec, rec, f = kc.prf_beta(tp=6, fp=2, fn=3)
    p, r = 6 / 8, 6 / 9
    expect = 1.25 * p * r / (0.25 * p + r)
    assert abs(f - expect) < 1e-9


def test_keeper_derivation_calibration_gate():
    # The four julia families + mandelbrot clear the >=15-positive floor; native multibrot
    # (0 positives) falls back to baseline, flagged uncalibrated.
    cuts = kc.derive()
    for part in ("julia:mandelbrot", "julia:multibrot3", "julia:multibrot4",
                 "julia:multibrot5", "mandelbrot"):
        assert cuts[part]["calibrated"] is True
        assert 0.02 <= cuts[part]["t"] <= 0.98
    for part in ("multibrot3", "multibrot4", "multibrot5"):
        assert cuts[part]["calibrated"] is False
        assert cuts[part]["t"] == kc.T_GOOD_BASELINE


def test_pop_batch_evicts_capped_root_nodes():
    # A node whose root is at M_CAP must be EVICTED (removed from the frontier + its cached
    # embedding dropped), not merely skipped-and-retained — else capped nodes clog the frontier.
    import types
    obj = types.SimpleNamespace(
        B=2,
        expansions_per_root={"1": sf.M_CAP, "2": 0},   # root 1 capped, root 2 open
        node_embs={101: None, 102: None, 201: None},
        totals={"cap_hits": 0},
        frontier=[
            {"node_id": 101, "root_id": 1, "priority": 5.0},   # capped -> evicted
            {"node_id": 102, "root_id": 1, "priority": 4.0},   # capped -> evicted
            {"node_id": 201, "root_id": 2, "priority": 3.0},   # open  -> popped
        ],
    )
    batch = sf.SteeredFrontier.pop_batch(obj)
    assert [n["node_id"] for n in batch] == [201]
    assert obj.frontier == []                                  # capped nodes gone, not retained
    assert 101 not in obj.node_embs and 102 not in obj.node_embs
    assert obj.expansions_per_root["2"] == 1                   # popped root incremented by 1


def test_is_keeper_uses_corn_decode():
    cuts = {"mandelbrot": {"t": 0.5}}
    # p_notbad>=0.5 AND p_good>=0.5 -> keeper; either failing -> not.
    assert kc.is_keeper("mandelbrot", 0.9, 0.9, cuts) is True
    assert kc.is_keeper("mandelbrot", 0.9, 0.4, cuts) is False
    assert kc.is_keeper("mandelbrot", 0.4, 0.9, cuts) is False


# =========================================================================== #
# MorphMemory — the v1.2 novelty-memory fix (legacy vs recency semantics).
# =========================================================================== #
def _unit(i, d=768):
    import numpy as np
    v = np.zeros(d, np.float32); v[i] = 1.0
    return v


def test_morph_memory_legacy_is_all_permanent(tmp_path):
    # recency_k == 0: admitted AND expanded looks are permanent; end_batch is a no-op.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=0)
    m.add_admitted(_unit(0))
    m.add_expanded(_unit(1))
    m.end_batch()                                   # no-op in legacy
    assert m.n_perm == 2 and m.n_recency == 0 and len(m) == 2
    # a perfect dup of either look reads cos_max ~ 1.
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(5)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] < 1e-6


def test_morph_memory_recency_window_evicts(tmp_path):
    # recency_k == 2: admitted permanent; expanded looks live in a 2-batch rolling window.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=2)
    m.add_admitted(_unit(0))                        # permanent
    m.add_expanded(_unit(1)); m.end_batch()         # block A = {e1}
    m.add_expanded(_unit(2)); m.end_batch()         # block B = {e2}
    # both windows + the admitted look are live here.
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] > 0.999 and cm[3] < 1e-6
    m.add_expanded(_unit(3)); m.end_batch()         # block C = {e3}; e1's block evicted (>K)
    cm = m.cos_max(np.stack([_unit(0), _unit(1), _unit(2), _unit(3)]))
    assert cm[0] > 0.999                            # admitted look survives (permanent)
    assert cm[1] < 1e-6                             # e1 evicted from the window
    assert cm[2] > 0.999 and cm[3] > 0.999          # e2, e3 still in window
    assert m.n_perm == 1 and m.n_recency == 2


def test_morph_memory_current_batch_excluded_until_end(tmp_path):
    # A look expanded THIS batch is NOT visible to cos_max until end_batch finalizes it into a
    # block — comparing a candidate to its own just-expanded parent would trivially saturate.
    import numpy as np
    m = sf.MorphMemory("cpu", tmp_path / "m.npz", recency_k=4)
    m.add_expanded(_unit(7))                         # pending, not yet in the window
    assert m.cos_max(np.stack([_unit(7)]))[0] < 1e-6
    m.end_batch()                                    # finalized -> visible to the NEXT batch
    assert m.cos_max(np.stack([_unit(7)]))[0] > 0.999


def test_morph_memory_roundtrip_persists_window(tmp_path):
    # save() + reload preserves permanent + window blocks (so a resume evicts on the same K).
    import numpy as np
    p = tmp_path / "m.npz"
    m = sf.MorphMemory("cpu", p, recency_k=2)
    m.add_admitted(_unit(0))
    m.add_expanded(_unit(1)); m.end_batch()
    m.add_expanded(_unit(2)); m.end_batch()
    m.save()
    m2 = sf.MorphMemory("cpu", p, recency_k=2)
    assert m2.n_perm == 1 and m2.n_recency == 2
    cm = m2.cos_max(np.stack([_unit(0), _unit(1), _unit(2)]))
    assert cm[0] > 0.999 and cm[1] > 0.999 and cm[2] > 0.999
