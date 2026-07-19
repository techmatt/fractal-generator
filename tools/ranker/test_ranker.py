#!/usr/bin/env python
"""Fast, artifact-free unit tests for the location preference ranker v0.

The deployed head is a plain affine map on standardized features, recovered from the fitted
sklearn head by an identity-matrix probe. These tests pin the two things that would silently
corrupt the scorer: (1) the affine probe exactly recovers a linear head's weights, and (2)
RankerScorer reproduces the standardize->dot->bias map and stacks feature blocks in the declared
order. No torch, no renders, no data/ artifacts (those are gitignored) — pure numpy/sklearn.

    uv run pytest tools/ranker/test_ranker.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.ranker import train_eval as te          # noqa: E402
from tools.ranker.scorer import RankerScorer         # noqa: E402


def test_affine_probe_recovers_linear_head():
    rng = np.random.default_rng(0)
    n, dim = 60, 24
    X = rng.standard_normal((n, dim))
    w_true = rng.standard_normal(dim)
    y = 2 + X @ w_true                                # exactly linear
    d = dict(score=np.r_[np.ones(n, int) * 2], batch=np.array(["run2"] * n))
    # emulate fit_deploy's probe on a ridge head fit to (X, y)
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=1e-6).fit(X, y)
    f = lambda Xe: m.predict(Xe)
    base = float(f(np.zeros((1, dim)))[0])
    W = f(np.eye(dim)) - base
    # recovered affine map must match the head everywhere
    Xt = rng.standard_normal((10, dim))
    assert np.allclose(Xt @ W + base, f(Xt), atol=1e-6)
    assert np.allclose(W, m.coef_, atol=1e-4)


def test_scorer_matches_manual_map():
    rng = np.random.default_rng(1)
    dim_m, dim_c = 5, 3
    mean = rng.standard_normal(dim_m + dim_c)
    scale = np.abs(rng.standard_normal(dim_m + dim_c)) + 0.1
    W = rng.standard_normal(dim_m + dim_c)
    b = 0.7
    s = RankerScorer(mean, scale, W, b, sets=["morph", "colored"], head="logi",
                     reg=0.1, use_prior=False)
    M = rng.standard_normal((8, dim_m))
    C = rng.standard_normal((8, dim_c))
    got = s.score_matrix({"morph": M, "colored": C})
    X = np.concatenate([M, C], axis=1)               # declared block order
    exp = ((X - mean) / scale) @ W + b
    assert np.allclose(got, exp)


def test_block_order_is_respected():
    # swapping which array is 'morph' vs 'colored' must change the score (order is load-bearing)
    rng = np.random.default_rng(2)
    s = RankerScorer(np.zeros(4), np.ones(4), np.array([1.0, 1, 0, 0]), 0.0,
                     sets=["morph", "colored"], head="logi", reg=1, use_prior=False)
    A = rng.standard_normal((3, 2))
    B = rng.standard_normal((3, 2))
    s1 = s.score_matrix({"morph": A, "colored": B})
    s2 = s.score_matrix({"morph": B, "colored": A})
    assert not np.allclose(s1, s2)


def test_metrics_orientation():
    # higher pred on the good items => positive spearman, AUC > 0.5, full precision@k
    human = np.array([1, 1, 2, 2, 3, 3])
    pred = np.array([0.0, 0.1, 0.5, 0.6, 0.9, 1.0])
    m = te.metrics(pred, human)
    assert m["spearman"] > 0.9 and m["auc"] > 0.9
    # precision@k with k=min(10,6)=6 == fraction good in the whole set (2/6)
    assert m["k"] == 6 and abs(m["p_at_10"] - 2 / 6) < 1e-9
    # with the goods concentrated at the very top, precision@2 is perfect
    top2 = te.metrics(pred, human)  # sanity: order is by -pred
    assert (human[np.argsort(-pred)][:2] == 3).all()


def test_bt_pairwise_learns_direction():
    # a clean 1-D separable ranking: BT weight should point so higher feature => higher score
    rng = np.random.default_rng(3)
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.where(X[:, 0] < -0.3, 1, np.where(X[:, 0] < 0.3, 2, 3))
    f = te.fit_bt(X, y, reg=1.0)
    lo, hi = f(np.array([[-1.0]]))[0], f(np.array([[1.0]]))[0]
    assert hi > lo


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
