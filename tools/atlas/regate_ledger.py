#!/usr/bin/env python
"""Part 3 — re-gate the harvested ledger against the drop manifest.

MARKS (does not delete) the 20 degenerate union outcomes in outcome_ledger.jsonl and
recomputes the cell-ledger distinct tallies from the surviving (guard_pass) rows:

  * Each outcome row gets `guard_pass` (bool) and `guard_fail`
    ('interior'|'flat'|'both'|null). The 20 union outcomes -> guard_pass=false with the
    per-gate attribution from the manifest; all others -> guard_pass=true, guard_fail
    null. The rows STAY (the quality-head training set excludes them explicitly).
  * cell_ledger.json distinct tallies are recomputed from `distinct AND guard_pass`
    rows only (LAUNCHES ARE UNCHANGED — a launch spent compute regardless of outcome
    quality; the distinct tally is what should reflect good yield). `saturated` is
    recomputed from the new (launches, distinct).
  * Cells whose distinct drops below OUTCOME_DISTINCT_CAP and thus RE-OPEN for future
    exploit are reported (correct: they didn't yield enough good distinct outcomes).

Atomic (temp+rename) on both files; idempotent (re-running reproduces the same state).
Downstream consumers read `guard_pass == true` as the harvested set.

  uv run python tools/atlas/regate_ledger.py            # apply
  uv run python tools/atlas/regate_ledger.py --dry-run  # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from production_seeder import (  # noqa: E402  reuse the exact caps + atomic writer + predicate
    OUTCOME_LEDGER, CELL_LEDGER, OUTCOME_DISTINCT_CAP, SEED_LAUNCH_CAP,
    cell_saturated, cell_launch_capped, _atomic_write_text,
)

MANIFEST = ROOT / "out" / "atlas" / "diag_outcome_guards" / "drop_manifest.json"


def gate_reason(oid, interior, flat, both) -> str | None:
    if oid in both:
        return "both"
    if oid in interior:
        return "interior"
    if oid in flat:
        return "flat"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    interior, flat, both, union = (set(man["interior_gate"]), set(man["flat_gate"]),
                                   set(man["both"]), set(man["union"]))
    print(f"=== Part 3: re-gate ledger against drop_manifest "
          f"(union {len(union)}: interior_only {len(interior-flat)}, "
          f"flat_only {len(flat-interior)}, both {len(both)}) ===")

    rows = [json.loads(l) for l in open(OUTCOME_LEDGER, encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in union if i not in by_id]
    if missing:
        raise SystemExit(f"union IDs absent from ledger: {missing}")

    # --- 1. mark rows (guard_pass / guard_fail) --- #
    n_fail = 0
    reason_ct = Counter()
    for r in rows:
        reason = gate_reason(r["id"], interior, flat, both)
        r["guard_pass"] = reason is None
        r["guard_fail"] = reason
        if reason is not None:
            n_fail += 1
            reason_ct[reason] += 1
    print(f"\nmarked {len(rows)} rows: {n_fail} guard_fail "
          f"(interior {reason_ct['interior']}, flat {reason_ct['flat']}, both {reason_ct['both']}), "
          f"{len(rows)-n_fail} guard_pass")
    # sanity: every union member marked fail, and only union members
    marked_fail = {r["id"] for r in rows if not r["guard_pass"]}
    assert marked_fail == union, f"marked-fail set != union (diff {marked_fail ^ union})"

    # --- 2. recompute cell distinct tallies from distinct AND guard_pass rows --- #
    cells = json.loads(CELL_LEDGER.read_text(encoding="utf-8"))
    new_distinct = Counter()
    for r in rows:
        if r.get("distinct") and r["guard_pass"]:
            new_distinct[str(r["seed_cell"])] += 1

    reopened, changed = [], []
    for cid, st in cells.items():
        old_d = st.get("distinct", 0)
        new_d = new_distinct.get(cid, 0)
        old_sat = cell_saturated(st)
        st["distinct"] = new_d
        st["saturated"] = cell_saturated(st)
        if new_d != old_d:
            changed.append((cid, old_d, new_d))
        # re-opens for EXPLOIT: was saturated, now not, and not still launch-capped.
        if old_sat and not st["saturated"] and not cell_launch_capped(st):
            reopened.append((cid, old_d, new_d, st.get("launches", 0)))

    print(f"\ncell tallies: {len(changed)} cells changed distinct count "
          f"(dropped total {sum(o-n for _,o,n in changed)} distinct across cells)")
    for cid, o, n in sorted(changed, key=lambda t: t[1]-t[2], reverse=True):
        print(f"  cell {cid:>4}: distinct {o} -> {n}  (launches {cells[cid].get('launches',0)}, "
              f"saturated -> {cells[cid]['saturated']})")

    print(f"\nRE-OPENED for exploit ({len(reopened)} cells: were distinct-saturated, "
          f"now distinct<{OUTCOME_DISTINCT_CAP} and launches<{SEED_LAUNCH_CAP}):")
    for cid, o, n, lz in reopened:
        print(f"  cell {cid:>4}: distinct {o} -> {n}, launches {lz}  -> exploit-eligible again")
    if not reopened:
        print("  (none — no cell was distinct-saturated, or all remain launch-capped)")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    # --- 3. atomic writes (both files git-tracked, so recoverable) --- #
    _atomic_write_text(OUTCOME_LEDGER,
                       "".join(json.dumps(r) + "\n" for r in rows))
    _atomic_write_text(CELL_LEDGER, json.dumps(cells, indent=2))
    print(f"\nwrote {OUTCOME_LEDGER.relative_to(ROOT)} ({len(rows)} rows) + "
          f"{CELL_LEDGER.relative_to(ROOT)} (atomic). Downstream reads guard_pass==true.")


if __name__ == "__main__":
    main()
