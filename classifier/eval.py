"""Ranking metrics + grouped splits for the v1 classifier.

This is a *ranking* problem (75/25 class skew makes accuracy meaningless), so the
metrics are AP and precision@k over a monotone score. Splits group by
seed/location — a crop-level split would leak structure (same location recurs
across 2 palettes x 3 comps) into val.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


def _ap(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")  # AP undefined with no/all positives
    if not np.isfinite(score).all():
        # a diverged model emits NaN scores; rank them worst rather than crash
        score = np.nan_to_num(score, nan=np.nanmin(score) - 1.0 if np.isfinite(score).any() else 0.0)
    return float(average_precision_score(y_true, score))


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)  # NaN -> worst
    k = min(k, len(score))
    if k == 0:
        return float("nan")
    top = np.argsort(-np.asarray(score))[:k]
    return float(y_true[top].sum() / k)


def crop_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    not_bad = (labels >= 2).astype(int)
    good = (labels == 3).astype(int)
    return {
        "ap_not_bad": _ap(not_bad, scores),
        "ap_good": _ap(good, scores),
        "p_at_10_not_bad": precision_at_k(not_bad, scores, 10),
        "p_at_25_not_bad": precision_at_k(not_bad, scores, 25),
        "p_at_10_good": precision_at_k(good, scores, 10),
        "p_at_25_good": precision_at_k(good, scores, 25),
        "n": int(len(labels)),
        "n_not_bad": int(not_bad.sum()),
        "n_good": int(good.sum()),
    }


def location_metrics(scores: np.ndarray, labels: np.ndarray, seeds: np.ndarray) -> dict:
    """Aggregate crops -> per-location (seed) via best-over-palettes: location
    score = max crop score; location label = max crop label ('how good can this
    location look')."""
    seeds = np.asarray(seeds)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)  # NaN -> worst
    labels = np.asarray(labels)
    loc_score, loc_label = [], []
    for s in np.unique(seeds):
        m = seeds == s
        loc_score.append(scores[m].max())
        loc_label.append(labels[m].max())
    loc_score = np.array(loc_score)
    loc_label = np.array(loc_label)
    not_bad = (loc_label >= 2).astype(int)
    good = (loc_label == 3).astype(int)
    return {
        "loc_ap_not_bad": _ap(not_bad, loc_score),
        "loc_ap_good": _ap(good, loc_score),
        "loc_p_at_10_not_bad": precision_at_k(not_bad, loc_score, 10),
        "loc_p_at_25_not_bad": precision_at_k(not_bad, loc_score, 25),
        "n_locations": int(len(loc_label)),
        "n_loc_not_bad": int(not_bad.sum()),
        "n_loc_good": int(good.sum()),
    }


# --------------------------------------------------------------------------- #
# Grouped splits (group = seed). Stratify on each seed's MAX label.
# --------------------------------------------------------------------------- #
def _seed_strat(rows) -> tuple[np.ndarray, np.ndarray]:
    seeds = np.array([r.seed for r in rows])
    labels = np.array([r.label for r in rows])
    seed_max = {}
    for s, l in zip(seeds, labels):
        seed_max[s] = max(seed_max.get(s, 0), l)
    strat = np.array([seed_max[s] for s in seeds])  # per-row stratify target
    return seeds, strat


def make_folds(rows, n_splits: int = 5, seed: int = 0):
    seeds, strat = _seed_strat(rows)
    X = np.zeros(len(rows))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, strat, groups=seeds))


def make_holdout(rows, frac: float = 0.15, seed: int = 0):
    """Single grouped train/holdout split (group = seed). Stratification isn't
    available in GroupShuffleSplit, but the deliverable holdout only needs an
    unseen-seed val set."""
    seeds, _ = _seed_strat(rows)
    X = np.zeros(len(rows))
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    tr, va = next(gss.split(X, groups=seeds))
    return tr, va


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """mean +- std across folds for every scalar key (NaNs ignored)."""
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {"mean": float(finite.mean()), "std": float(finite.std())}
    return out
