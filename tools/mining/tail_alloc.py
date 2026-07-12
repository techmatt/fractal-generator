"""Diversity allocation for the strange-mode deploy tail (pure, dep-free).

Batch-level keeper assignment that SPREADS strange renders across the promoted
modes instead of letting the abundant mode (tia) monopolize the budget. Lives in
its own module (no torch / no render deps) so the allocation logic is unit-tested
cheaply — see `test_tail_alloc.py`.

The spec (prompts/deploy_tail_diversity_allocation_prompt.md):
  * n = number of promoted modes (the tail roster). N = number of emitted locations.
  * Strange budget  B = round(BUDGET_FRAC * N)   (BUDGET_FRAC = 0.25).
  * Per-mode floor  = B // (n + 2)   (each mode guaranteed ~1/(n+2) of the budget).
    n*floor < B, so the leftover ~2/(n+2)*B is SURPLUS -> given to whichever modes
    can still supply, by quality (so high-yield modes land above their floor).
  * Graceful degradation: a mode with fewer distinct-location passers than its floor
    takes all it has; the shortfall rolls into the surplus pool. If total passers < B,
    keep them all — under-fill is fine, NEVER pad to hit the budget.
  * <= 1 strange alternate per location. A location may pass in several modes, so this
    is a batch-level assignment (each location fills at most one mode's slot), NOT
    per-location argmax. Greedy: fair round-robin floor fill first, then surplus by
    global quality, never double-assigning a location.
  * Only gate-passers are eligible; caller filters to passers before calling.

Incremental / idempotent state (production curation pass, `existing=`):
  * The curation pass runs on demand over the ACCUMULATED emission corpus and may run
    again after the corpus grows. Already-emitted strange alternates are FIXED: pass
    them in via `existing` (each exposing 'loc_id' + 'mode'). They are counted toward
    the budget B and toward their modes' floors, and their locations are locked out of
    re-assignment. `allocate_strange` then allocates ONLY the remaining shortfall
    (B - #existing) over the `passers` (which the caller has already restricted to
    not-yet-curated locations). The returned list is the NEW picks only; `meta.achieved`
    is the CORPUS-wide count (existing + new) per mode. `existing=()` (the default)
    reproduces the from-scratch single-batch behaviour byte-for-byte.
  * This makes a re-run over an unchanged corpus a no-op: B is a ceiling and a correct
    prior run already filled it (or exhausted supply), so the shortfall is 0.
"""
from __future__ import annotations

BUDGET_FRAC = 0.25   # strange budget as a fraction of emitted locations (raised 0.20 -> 0.25)


def budget(n_emit: int, budget_frac: float = BUDGET_FRAC) -> int:
    """B = round(budget_frac * N)."""
    return int(round(budget_frac * n_emit))


def mode_floor(B: int, n_modes: int) -> int:
    """Per-mode guaranteed floor = B // (n + 2)."""
    return B // (n_modes + 2)


def allocate_strange(passers, n_emit, mode_order, budget_frac: float = BUDGET_FRAC,
                     existing=()):
    """Assign strange keepers across modes for diversity.

    Args:
      passers:    iterable of gate-passer candidates. Each is a dict (or any obj)
                  exposing 'loc_id', 'mode', 'p_ge3' via item access. The caller
                  restricts this to NOT-YET-CURATED locations (no existing alternate).
      n_emit:     N, the number of emitted locations (sets budget B over the whole corpus).
      mode_order: the promoted-mode roster (list of mode names); n = len. Also the
                  deterministic tie-break order when two modes contend for a location.
      budget_frac: fraction of N to spend on strange alternates (default 0.25).
      existing:   already-emitted (FIXED) alternates, each exposing 'loc_id' + 'mode'.
                  Counted toward B and toward per-mode floors; their locations are
                  locked out of re-assignment. Only the shortfall (B - #existing) is
                  allocated over `passers`. `existing=()` == from-scratch single batch.

    Returns (selected, meta):
      selected: the NEW picks only (subset of `passers`), each a distinct loc_id not
                in `existing`, each assigned to exactly one mode. #existing + len <= B.
      meta:     dict with B, floor, n_modes, budget_frac, corpus-wide per-mode `achieved`
                (existing + new), n_fixed (distinct existing locs), and n_new (== len).
    """
    n = len(mode_order)
    B = budget(n_emit, budget_frac)
    floor = mode_floor(B, n)

    order_index = {m: i for i, m in enumerate(mode_order)}

    # Seed the state from the FIXED existing alternates: their locations are used up,
    # and they already count toward each mode's achieved total (hence toward its floor).
    used_locs: set = set()
    achieved = {m: 0 for m in mode_order}
    for e in existing:
        used_locs.add(e["loc_id"])
        if e["mode"] in achieved:
            achieved[e["mode"]] += 1
    n_fixed = len(used_locs)
    remaining_budget = max(0, B - n_fixed)   # how many NEW picks we may still add

    by_mode = {m: [] for m in mode_order}
    for c in passers:
        if c["mode"] in by_mode and c["loc_id"] not in used_locs:
            by_mode[c["mode"]].append(c)
    # highest p_ge3 first within a mode; loc_id tie-break for determinism.
    for m in by_mode:
        by_mode[m].sort(key=lambda c: (-c["p_ge3"], c["loc_id"]))

    selected: list = []

    def next_unused(m):
        for c in by_mode[m]:
            if c["loc_id"] not in used_locs:
                return c
        return None

    def take(c):
        used_locs.add(c["loc_id"])
        selected.append(c)
        achieved[c["mode"]] += 1

    # -- Phase A: floor fill, fair round-robin across modes (one pick/mode per round).
    #    Round-robin (not mode-by-mode-to-completion) so a contended location isn't
    #    hoarded by whichever mode is processed first — each mode advances in lockstep.
    #    A mode whose floor is already met by `existing` is skipped (achieved >= floor).
    if floor > 0:
        while len(selected) < remaining_budget:
            progressed = False
            for m in mode_order:
                if achieved[m] >= floor:
                    continue
                pick = next_unused(m)
                if pick is not None:
                    take(pick)
                    progressed = True
                    if len(selected) >= remaining_budget:
                        break
            if not progressed:
                break

    # -- Phase B: surplus by global quality. Abundant modes have more passers left in
    #    the pool, so they naturally absorb the surplus above their floor. Sort by
    #    p_ge3 desc, then roster order, then loc_id — fully deterministic.
    if len(selected) < remaining_budget:
        remaining = [c for m in mode_order for c in by_mode[m]
                     if c["loc_id"] not in used_locs]
        remaining.sort(key=lambda c: (-c["p_ge3"], order_index[c["mode"]], c["loc_id"]))
        for c in remaining:
            if c["loc_id"] in used_locs:
                continue
            take(c)
            if len(selected) >= remaining_budget:
                break

    meta = {"B": B, "floor": floor, "n_modes": n, "budget_frac": budget_frac,
            "achieved": achieved, "n_fixed": n_fixed, "n_new": len(selected)}
    return selected, meta
