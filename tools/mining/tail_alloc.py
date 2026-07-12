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
"""
from __future__ import annotations

BUDGET_FRAC = 0.25   # strange budget as a fraction of emitted locations (raised 0.20 -> 0.25)


def budget(n_emit: int, budget_frac: float = BUDGET_FRAC) -> int:
    """B = round(budget_frac * N)."""
    return int(round(budget_frac * n_emit))


def mode_floor(B: int, n_modes: int) -> int:
    """Per-mode guaranteed floor = B // (n + 2)."""
    return B // (n_modes + 2)


def allocate_strange(passers, n_emit, mode_order, budget_frac: float = BUDGET_FRAC):
    """Assign strange keepers across modes for diversity.

    Args:
      passers:    iterable of gate-passer candidates. Each is a dict (or any obj)
                  exposing 'loc_id', 'mode', 'p_ge3' via item access.
      n_emit:     N, the number of emitted locations (sets budget B).
      mode_order: the promoted-mode roster (list of mode names); n = len. Also the
                  deterministic tie-break order when two modes contend for a location.
      budget_frac: fraction of N to spend on strange alternates (default 0.25).

    Returns (selected, meta):
      selected: list of chosen passer objects (subset of `passers`), each a distinct
                loc_id, each assigned to exactly one mode. len(selected) <= B.
      meta:     dict with B, floor, n_modes, budget_frac, and per-mode achieved counts.
    """
    n = len(mode_order)
    B = budget(n_emit, budget_frac)
    floor = mode_floor(B, n)

    order_index = {m: i for i, m in enumerate(mode_order)}
    by_mode = {m: [] for m in mode_order}
    for c in passers:
        if c["mode"] in by_mode:
            by_mode[c["mode"]].append(c)
    # highest p_ge3 first within a mode; loc_id tie-break for determinism.
    for m in by_mode:
        by_mode[m].sort(key=lambda c: (-c["p_ge3"], c["loc_id"]))

    used_locs: set = set()
    selected: list = []
    achieved = {m: 0 for m in mode_order}

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
    if floor > 0:
        while len(selected) < B:
            progressed = False
            for m in mode_order:
                if achieved[m] >= floor:
                    continue
                pick = next_unused(m)
                if pick is not None:
                    take(pick)
                    progressed = True
                    if len(selected) >= B:
                        break
            if not progressed:
                break

    # -- Phase B: surplus by global quality. Abundant modes have more passers left in
    #    the pool, so they naturally absorb the surplus above their floor. Sort by
    #    p_ge3 desc, then roster order, then loc_id — fully deterministic.
    if len(selected) < B:
        remaining = [c for m in mode_order for c in by_mode[m]
                     if c["loc_id"] not in used_locs]
        remaining.sort(key=lambda c: (-c["p_ge3"], order_index[c["mode"]], c["loc_id"]))
        for c in remaining:
            if c["loc_id"] in used_locs:
                continue
            take(c)
            if len(selected) >= B:
                break

    meta = {"B": B, "floor": floor, "n_modes": n, "budget_frac": budget_frac,
            "achieved": achieved}
    return selected, meta
