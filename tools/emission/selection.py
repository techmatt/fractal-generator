"""selection.py — greedy max-marginal-gain release selection (pure; no torch/GPU).

(Named `selection`, not `select`, so it never shadows the stdlib `select` module when
this directory lands on sys.path[0] as a run script's home.)

Select a release set of N wallpapers from the gated pool. Each greedy step picks the
pool entry with the largest marginal gain, where

    marginal_gain(c | selected) = niche_relative_quality(c) × coverage_gain(c | selected)

* niche_relative_quality(c) — the within-niche percentile of c's head score, so the
  best of a sparse niche can beat the tenth-best of a crowded one. The niche is the
  full descriptor cell (type, cluster, flavor, style); a singleton niche scores 1.0.

* coverage_gain(c | selected) = 1 − max_{s∈selected} K(c, s), the standard
  facility-location marginal: an item adds coverage in proportion to how UNlike the
  nearest already-selected item it is. K is the similarity kernel below.

* K(a, b) = [∏ over categorical axes 1{a_axis == b_axis}] × max(0, cos(emb_a, emb_b)).
  Product of exact-match categorical kernels × cosine on the morph-CLIP embedding, so
  two entries are "similar" only when they share the SAME descriptor cell AND look
  alike; entries in different cells are orthogonal (K = 0) and never suppress each
  other. This makes the first pick from each occupied cell maximally valuable and only
  down-weights a second, visually-near-duplicate pick from the same cell.

Ties in marginal gain (ubiquitous when most niches are singletons, all at quality
percentile 1.0 and coverage 1.0) are broken by absolute head score, so a genuinely
strong singleton beats a weak one. Every pick logs the niche it filled and its nearest
already-selected neighbour (the "displacement" it caused).

CAVEAT — cross-head score incommensurability. `score` is compared directly only in the
tie-break, and the niche-percentile normalisation only bites when a niche holds >1 entry.
With one colorize per location (the norm at library scale) EVERY niche is a singleton, so
niche_pct ≡ 1.0 and coverage ≡ 1.0 for the first pick of each cell — greedy_select then
degenerates to top-N-by-absolute-`score`. When the pool mixes entries scored by DIFFERENT
heads (e.g. the wallpaper head's `p_ge3` for smooth vs the mining head's for strange
styles), those absolute scores are on incommensurable scales, and the smaller-scaled head
can be shut out entirely (this is exactly how 82 release-eligible strange tiles lost every
slot to smooth). If you need guaranteed cross-head representation, pre-partition by head and
select a quota from each, or pass a within-head-normalised `score` — this selector treats
`score` as a single comparable quality axis and does NOT reconcile heads itself.

Pure Python + math; embeddings arrive as plain float lists so this is unit-testable.
"""
from __future__ import annotations

import math
from collections import defaultdict


CATEGORICAL_AXES = ("type", "cluster", "flavor", "style")


def _cos(a, b) -> float:
    if a is None or b is None:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def kernel(a: dict, b: dict) -> float:
    """Similarity kernel between two pool entries (see module docstring)."""
    for ax in CATEGORICAL_AXES:
        if a[ax] != b[ax]:
            return 0.0
    return max(0.0, _cos(a.get("emb"), b.get("emb")))


def niche_percentiles(entries: list) -> dict:
    """id → within-niche percentile of `score` (fraction of same-niche entries whose
    score is ≤ this one). A singleton niche → 1.0. Ties share the higher rank."""
    by_niche = defaultdict(list)
    for e in entries:
        by_niche[tuple(e[ax] for ax in CATEGORICAL_AXES)].append(e)
    pct = {}
    for niche, es in by_niche.items():
        n = len(es)
        for e in es:
            le = sum(1 for o in es if o["score"] <= e["score"])
            pct[e["id"]] = le / n
    return pct


def greedy_select(entries: list, n: int) -> tuple:
    """Greedy max-marginal-gain selection of ≤ n entries.

    `entries`: list of dicts with keys id, type, cluster, flavor, style, score
    (head p_ge3 or ordinal score — any within-niche-comparable quality), emb (list|None).
    Returns (selected, log) where selected is the ordered list of chosen entries and log
    is a per-pick list of {id, niche, niche_pct, coverage_gain, marginal_gain, score,
    nearest_selected, nearest_sim}."""
    pct = niche_percentiles(entries)
    remaining = list(entries)
    selected: list = []
    log: list = []
    while remaining and len(selected) < n:
        best = None
        best_key = None
        for c in remaining:
            if selected:
                sims = [(kernel(c, s), s) for s in selected]
                nn_sim, nn = max(sims, key=lambda t: t[0])
            else:
                nn_sim, nn = 0.0, None
            cov = 1.0 - nn_sim
            q = pct[c["id"]]
            gain = q * cov
            # primary: marginal gain; tie-break: absolute score, then id for determinism.
            key = (gain, c["score"], c["id"])
            if best_key is None or key > best_key:
                best_key, best = key, (c, q, cov, gain, nn, nn_sim)
        c, q, cov, gain, nn, nn_sim = best
        selected.append(c)
        remaining.remove(c)
        log.append({
            "id": c["id"],
            "niche": [c[ax] for ax in CATEGORICAL_AXES],
            "niche_pct": round(q, 4),
            "coverage_gain": round(cov, 4),
            "marginal_gain": round(gain, 4),
            "score": round(float(c["score"]), 4),
            "nearest_selected": (nn["id"] if nn else None),
            "nearest_sim": round(nn_sim, 4),
        })
    return selected, log
