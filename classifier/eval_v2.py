"""Batch-aware ranking metrics + grouped splits for the v2 cross-batch model.

Reuses v1's metric primitives (`_ap`, `precision_at_k`, `crop_metrics`) verbatim
so the numbers are on the SAME protocol as v1 — only the grouping/aggregation
keys become batch-qualified, and a per-batch breakdown is added.

Splits operate on integer position arrays (0..N-1) plus parallel `groups` /
`strat` arrays, rather than reading attributes off rows, so a "location" (for
metrics) and a "correlation unit" (for folds) can differ — in rev4 the location
is the frame (seed) while the fold group is the whole walk.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .eval import _ap, crop_metrics, precision_at_k  # reused verbatim


# --------------------------------------------------------------------------- #
# Location aggregation by an arbitrary (batch-qualified) key.
# --------------------------------------------------------------------------- #
def location_metrics_keyed(scores, labels, loc_keys, prefix: str = "loc_") -> dict:
    """best-over-palettes within a location: loc score = max crop score,
    loc label = max crop label. `loc_keys` is any array of hashables."""
    scores = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    labels = np.asarray(labels)
    loc_keys = np.asarray(loc_keys, dtype=object)
    uniq = list(dict.fromkeys(loc_keys.tolist()))  # stable unique
    loc_score, loc_label = [], []
    for k in uniq:
        m = loc_keys == k
        loc_score.append(scores[m].max())
        loc_label.append(labels[m].max())
    loc_score = np.array(loc_score, float)
    loc_label = np.array(loc_label)
    not_bad = (loc_label >= 2).astype(int)
    good = (loc_label == 3).astype(int)
    return {
        f"{prefix}ap_not_bad": _ap(not_bad, loc_score),
        f"{prefix}ap_good": _ap(good, loc_score),
        f"{prefix}p_at_10_not_bad": precision_at_k(not_bad, loc_score, 10),
        f"{prefix}p_at_25_not_bad": precision_at_k(not_bad, loc_score, 25),
        f"{prefix}n_locations": int(len(loc_label)),
        f"{prefix}n_loc_not_bad": int(not_bad.sum()),
        f"{prefix}n_loc_good": int(good.sum()),
    }


def full_metrics(scores, labels, loc_keys) -> dict:
    """crop-level + location-level on one set."""
    m = crop_metrics(np.asarray(scores), np.asarray(labels))
    m.update(location_metrics_keyed(scores, labels, loc_keys))
    return m


def per_batch_metrics(scores, labels, batch_ids, loc_keys) -> dict:
    """Split a scored set by batch_id, report crop+loc metrics per batch."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels)
    batch_ids = np.asarray(batch_ids, dtype=object)
    loc_keys = np.asarray(loc_keys, dtype=object)
    out = {}
    for b in dict.fromkeys(batch_ids.tolist()):
        m = batch_ids == b
        out[b] = full_metrics(scores[m], labels[m], loc_keys[m])
    return out


# --------------------------------------------------------------------------- #
# Grouped, batch×label-stratified holdout + CV-on-remainder.
# --------------------------------------------------------------------------- #
def _strat_codes(strat) -> np.ndarray:
    """Map arbitrary stratify labels to dense int codes for sklearn."""
    uniq = {v: i for i, v in enumerate(dict.fromkeys(np.asarray(strat, dtype=object).tolist()))}
    return np.array([uniq[v] for v in strat], dtype=int)


def make_grouped_holdout(groups, strat, frac: float = 0.15, seed: int = 0):
    """Single grouped, stratified holdout: take fold 0 of a
    StratifiedGroupKFold(round(1/frac)) as the holdout, the rest as train.
    Groups never span the split; strata (batch×label) are balanced across it.
    Returns (train_idx, holdout_idx) into 0..N-1."""
    n_splits = max(2, round(1.0 / frac))
    y = _strat_codes(strat)
    g = np.asarray(groups, dtype=object)
    X = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, va = next(iter(skf.split(X, y, groups=g)))
    return tr, va


def make_grouped_folds(groups, strat, subset_idx=None, n_splits: int = 5, seed: int = 0):
    """StratifiedGroupKFold over `subset_idx` (default all). Yields
    (train_global_idx, val_global_idx) mapped back to original positions."""
    groups = np.asarray(groups, dtype=object)
    strat = np.asarray(strat, dtype=object)
    idx = np.arange(len(groups)) if subset_idx is None else np.asarray(subset_idx)
    y = _strat_codes(strat[idx])
    g = groups[idx]
    X = np.zeros(len(idx))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups=g):
        yield idx[tr], idx[va]


def group_strat_arrays(rows):
    """(groups, strat) for a list of CorpusRow. strat = batch × per-group-max-label
    so the holdout/folds keep both batches and all label tiers represented."""
    groups = [r.group_unit for r in rows]
    gmax = {}
    for r in rows:
        gmax[r.group_unit] = max(gmax.get(r.group_unit, 0), r.label)
    strat = [f"{r.batch_id}|{gmax[r.group_unit]}" for r in rows]
    return np.array(groups, dtype=object), np.array(strat, dtype=object)


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([fm[k] for fm in fold_metrics], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = ({"mean": float(finite.mean()), "std": float(finite.std())}
                  if len(finite) else {"mean": float("nan"), "std": float("nan")})
    return out
