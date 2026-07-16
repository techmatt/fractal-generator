"""Shared emission-pool rule — top-K-by-pref, one representative per palette.

Lifted out of `build_humanq3.py` (a one-shot batch builder) so the LIVE
`build_fresh_discovery` imports the pool rule from a neutral module instead of
from a study builder. `build_humanq3` and `build_headbatch_dramatic` import it
from here too. All callers pass `k` explicitly; the deployed default is K=12
(see `build_humanq3.K`)."""

DEFAULT_K = 12  # deployed emission-pool depth == serving pool depth (mirrors build_humanq3.K)


def top_k_pool(all_candidates, k=DEFAULT_K):
    """The emission pool: one representative per palette (its best-scoring evaluated
    candidate — a weak palette is its gen-0 render, a strong palette its refined best),
    then the top-k by pref-v2. Distinct palettes by construction, so the k form a real
    within-location gradient (rank-1 highest pref-v2 .. rank-k the pool floor); the R=2
    refined winners land at rank-1 by construction. Each pick gains `rank` (1..k) and
    `pool_size`. Fewer than k distinct palettes -> take what's there."""
    best = {}
    for c in all_candidates:
        cur = best.get(c["palette"])
        if cur is None or c["score"] > cur["score"]:
            best[c["palette"]] = c
    reps = sorted(best.values(), key=lambda c: -c["score"])[:k]
    for rank, c in enumerate(reps, start=1):
        c["rank"] = rank
        c["pool_size"] = len(reps)
    return reps
