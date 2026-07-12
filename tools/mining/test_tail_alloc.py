"""Synthetic-pool tests for the strange-mode diversity allocation (tail_alloc).

Cheap logic check — no torch, no renders. Constructs per-mode `p_ge3` supply so
each property is exercised in isolation. Run:  uv run python tools/mining/test_tail_alloc.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tail_alloc import allocate_strange, budget, mode_floor  # noqa: E402

ROSTER = ["tia", "stripe", "c13", "c17", "dtm", "dts"]   # n = 6 -> floor = B // 8


def cand(loc_id, mode, p):
    return {"loc_id": loc_id, "mode": mode, "p_ge3": p}


def counts(selected):
    return Counter(c["mode"] for c in selected)


def _distinct_locs(selected):
    return len({c["loc_id"] for c in selected})


def test_floors_met_when_supply_allows():
    """Every mode over-supplied on DISTINCT locations -> each mode hits its floor,
    surplus goes somewhere, budget respected, <=1/loc."""
    N = 80                       # B = 20, n=6 -> floor = 20//8 = 2
    B = budget(N)
    floor = mode_floor(B, len(ROSTER))
    assert (B, floor) == (20, 2), (B, floor)

    passers = []
    lid = 0
    for m in ROSTER:
        for k in range(6):       # 6 distinct-location passers per mode
            passers.append(cand(f"L{lid}", m, 0.5 + 0.01 * k))
            lid += 1
    selected, meta = allocate_strange(passers, N, ROSTER)

    c = counts(selected)
    assert len(selected) == B, len(selected)                 # budget filled
    assert _distinct_locs(selected) == len(selected)         # <=1 per loc
    for m in ROSTER:
        assert c[m] >= floor, (m, c[m], floor)               # floors met
    assert sum(c.values()) == B


def test_surplus_lands_on_abundant_modes():
    """All modes meet floor from a thin supply; only tia has extra high-quality
    passers left -> the entire surplus lands on tia."""
    N = 80                       # B=20, floor=2
    B = budget(N)
    floor = mode_floor(B, len(ROSTER))
    passers = []
    lid = 0
    # every non-tia mode: exactly `floor` passers (just enough, nothing to spare).
    for m in ROSTER[1:]:
        for _ in range(floor):
            passers.append(cand(f"L{lid}", m, 0.60)); lid += 1
    # tia: floor + a big surplus of the highest-quality passers.
    tia_supply = floor + 30
    for k in range(tia_supply):
        passers.append(cand(f"L{lid}", "tia", 0.90 + 1e-4 * k)); lid += 1

    selected, meta = allocate_strange(passers, N, ROSTER)
    c = counts(selected)
    assert len(selected) == B
    for m in ROSTER[1:]:
        assert c[m] == floor, (m, c[m])                      # thin modes stay at floor
    surplus = B - floor * len(ROSTER)
    assert c["tia"] == floor + surplus, (c["tia"], floor, surplus)   # all surplus -> tia


def test_starved_mode_degrades_gracefully():
    """One mode supplies fewer than its floor -> takes all it has; the shortfall
    redistributes to modes that can supply (here tia). Budget still filled, <=1/loc."""
    N = 80                       # B=20, floor=2
    B = budget(N)
    floor = mode_floor(B, len(ROSTER))
    passers = []
    lid = 0
    # 'dts' is starved: only 1 passer (< floor of 2).
    passers.append(cand(f"L{lid}", "dts", 0.55)); lid += 1
    # other non-tia modes: exactly floor each.
    for m in ["stripe", "c13", "c17", "dtm"]:
        for _ in range(floor):
            passers.append(cand(f"L{lid}", m, 0.60)); lid += 1
    # tia: abundant, highest quality -> absorbs its floor + the redistributed shortfall.
    for k in range(floor + 40):
        passers.append(cand(f"L{lid}", "tia", 0.90 + 1e-4 * k)); lid += 1

    selected, meta = allocate_strange(passers, N, ROSTER)
    c = counts(selected)
    assert c["dts"] == 1, c["dts"]                           # took all it had
    for m in ["stripe", "c13", "c17", "dtm"]:
        assert c[m] == floor, (m, c[m])
    assert len(selected) == B                                # shortfall absorbed -> still full
    assert _distinct_locs(selected) == len(selected)
    # the 1-unit shortfall from dts landed on the abundant mode.
    assert c["tia"] == B - (1 + floor * 4), c["tia"]


def test_underfill_never_pads():
    """Total gate-passers < B -> keep them all, never pad. Budget is a CEILING."""
    N = 80                       # B=20
    B = budget(N)
    passers = [cand(f"L{i}", ROSTER[i % len(ROSTER)], 0.6) for i in range(5)]
    selected, meta = allocate_strange(passers, N, ROSTER)
    assert len(selected) == 5 < B                            # under-fill, no padding
    assert _distinct_locs(selected) == 5


def test_at_most_one_per_location():
    """A location that passes in MANY modes still fills at most one mode's slot."""
    N = 80
    B = budget(N)
    passers = []
    # 10 locations, each a passer in ALL six modes.
    for i in range(10):
        for m in ROSTER:
            passers.append(cand(f"L{i}", m, 0.5 + 0.01 * ROSTER.index(m)))
    # plus filler so the budget can be reached from distinct locs if it wanted to.
    for j in range(40):
        passers.append(cand(f"F{j}", "tia", 0.55))
    selected, meta = allocate_strange(passers, N, ROSTER)
    assert _distinct_locs(selected) == len(selected)         # no loc used twice
    assert len(selected) <= B


def test_budget_is_25pct():
    """B tracks 25% of N across a range."""
    for N, exp in [(0, 0), (4, 1), (10, 2), (40, 10), (100, 25), (1000, 250)]:
        assert budget(N) == exp, (N, budget(N), exp)


def test_small_n_floor_is_subunit_trivial():
    """At small N the floor collapses to 0 -> pure top-quality fill (acknowledged
    near-trivial regime). Verify it degrades to global argmax without crashing."""
    N = 10                       # B = round(2.5) = 2, floor = 2//8 = 0
    B = budget(N)
    assert (B, mode_floor(B, len(ROSTER))) == (2, 0)
    passers = [cand("La", "tia", 0.95), cand("La", "stripe", 0.80),
               cand("Lb", "tia", 0.86), cand("Lc", "c13", 0.60)]
    selected, meta = allocate_strange(passers, N, ROSTER)
    assert len(selected) == 2
    # global top-2 by p_ge3 on distinct locs: tia@La(0.95), tia@Lb(0.86).
    assert {c["loc_id"] for c in selected} == {"La", "Lb"}
    assert counts(selected)["tia"] == 2


def test_existing_counts_toward_budget_and_floor():
    """Incremental: a fixed existing alternate consumes one budget slot AND satisfies
    its mode's floor — so the shortfall is B-1 and that mode is NOT floor-filled again."""
    N = 80                       # B=20, floor=2
    B = budget(N)
    floor = mode_floor(B, len(ROSTER))
    assert (B, floor) == (20, 2)
    # tia already has 2 fixed alternates (meets its floor of 2) from a prior run.
    existing = [{"loc_id": "E0", "mode": "tia"}, {"loc_id": "E1", "mode": "tia"}]
    # fresh supply on distinct (new) locations, over-supplied on every mode.
    passers, lid = [], 100
    for m in ROSTER:
        for k in range(6):
            passers.append(cand(f"L{lid}", m, 0.5 + 0.01 * k)); lid += 1
    selected, meta = allocate_strange(passers, N, ROSTER, existing=existing)

    assert meta["n_fixed"] == 2
    assert len(selected) == B - 2                         # shortfall only
    # no NEW pick reuses a fixed location.
    assert not ({c["loc_id"] for c in selected} & {"E0", "E1"})
    c = counts(selected)
    # tia's floor was already met by `existing`, so it gets NO new floor pick; other
    # modes still reach their floor from the fresh supply.
    for m in ROSTER[1:]:
        assert c[m] >= floor, (m, c[m])
    # corpus-wide achieved (existing + new) respects the budget.
    assert sum(meta["achieved"].values()) == B
    assert meta["achieved"]["tia"] >= 2                   # the 2 fixed still counted


def test_existing_never_reassigns_a_curated_location():
    """A location that already has an alternate is locked out even if it would now be
    the top passer in a different mode — never churn."""
    N = 40                       # B=10
    B = budget(N)
    existing = [{"loc_id": "Lx", "mode": "tia"}]
    # Lx passes strongly in stripe too, but it's already curated -> ineligible.
    passers = [cand("Lx", "stripe", 0.99), cand("Ly", "c13", 0.60),
               cand("Lz", "c17", 0.58)]
    selected, meta = allocate_strange(passers, N, ROSTER, existing=existing)
    assert "Lx" not in {c["loc_id"] for c in selected}
    assert _distinct_locs(selected) == len(selected)
    assert meta["achieved"]["tia"] == 1                   # only the fixed one


def test_rerun_unchanged_corpus_is_noop():
    """Idempotency: feed back the full prior selection as `existing` with the SAME
    remaining supply — the shortfall is 0, so nothing new is allocated."""
    N = 10                       # B=2, floor=0
    B = budget(N)
    passers = [cand("La", "tia", 0.95), cand("Lb", "tia", 0.86),
               cand("Lc", "c13", 0.60)]
    first, _ = allocate_strange(passers, N, ROSTER)
    assert len(first) == B
    # second run: prior picks are now FIXED existing; only un-picked locs remain eligible.
    existing = [{"loc_id": c["loc_id"], "mode": c["mode"]} for c in first]
    kept_ids = {c["loc_id"] for c in first}
    remaining_passers = [c for c in passers if c["loc_id"] not in kept_ids]
    second, meta = allocate_strange(remaining_passers, N, ROSTER, existing=existing)
    assert second == []                                   # no-op
    assert meta["n_fixed"] == B


def test_existing_default_is_backward_compatible():
    """existing=() must reproduce the from-scratch result exactly."""
    N = 80
    passers, lid = [], 0
    for m in ROSTER:
        for k in range(4):
            passers.append(cand(f"L{lid}", m, 0.5 + 0.01 * k)); lid += 1
    a, ma = allocate_strange(passers, N, ROSTER)
    b, mb = allocate_strange(passers, N, ROSTER, existing=())
    assert [c["loc_id"] for c in a] == [c["loc_id"] for c in b]
    assert ma["achieved"] == mb["achieved"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n[test_tail_alloc] {len(tests)} passed")


if __name__ == "__main__":
    main()
