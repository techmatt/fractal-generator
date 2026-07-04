#!/usr/bin/env python
"""Part-2 control (HARD GATE): reproduce the drop manifest through the guard.

Re-score the 81 DISTINCT harvested outcomes at their stored (outcome_cx, outcome_cy,
outcome_fw) through the guard's OWN field path (guard.render_field at GUARD_STAT_RES
-> guard.field_measures -> guard.guard_fail). The set that fails the guard (would
return GUARD_SENTINEL) MUST equal drop_manifest.json's union of 20, with matching
per-gate attribution (interior_gate 13, flat_gate 9, both 2). This proves the
in-scorer field path measures identically to diag_outcome_guards.py's --dump-field
diagnostic.

Any mismatch -> STOP and report the diverging IDs with the in-scorer interior_frac /
field_std vs the diagnostic's table.csv values, so the measurement divergence is
reconciled BEFORE anything downstream is wired.

  uv run python tools/atlas/control_guard_manifest.py
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import guard  # noqa: E402

LEDGER = ROOT / "data" / "discovery" / "outcome_ledger.jsonl"
MANIFEST = ROOT / "out" / "atlas" / "diag_outcome_guards" / "drop_manifest.json"
DIAG_TABLE = ROOT / "out" / "atlas" / "diag_outcome_guards" / "table.csv"
FIELD_DIR = ROOT / "out" / "atlas" / "control_guard" / "fields"   # ephemeral, guard's own render
WORKERS = 6


def load_distinct_outcomes():
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("distinct")]


def load_diag_stats():
    """id -> (interior_frac, field_std) from the diagnostic table (for mismatch diag)."""
    out = {}
    if not DIAG_TABLE.exists():
        return out
    with open(DIAG_TABLE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = (float(r["interior_frac"]), float(r["field_std"]))
    return out


def measure_one(row):
    oid = row["id"]
    st = guard.measure_location(
        row["outcome_cx"], row["outcome_cy"], row["outcome_fw"],
        FIELD_DIR / f"{oid}.bin", family=row.get("family", "mandelbrot"))
    reason = guard.guard_fail(st.interior_frac, st.field_std)
    return oid, st, reason


def main():
    distinct = load_distinct_outcomes()
    n = len(distinct)
    print(f"=== Part-2 control: guard vs drop_manifest over {n} distinct outcomes ===")
    print(f"gates: interior_frac >= {guard.INTERIOR_CAP}  |  field_std < {guard.FIELD_STD_FLOOR}  "
          f"@ {guard.GUARD_STAT_RES}")
    assert n == 81, f"expected 81 distinct outcomes, got {n}"

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m_interior = set(manifest["interior_gate"])
    m_flat = set(manifest["flat_gate"])
    m_both = set(manifest["both"])
    m_union = set(manifest["union"])
    print(f"manifest: interior_gate={len(m_interior)} flat_gate={len(m_flat)} "
          f"both={len(m_both)} union={len(m_union)}")

    # --- render + measure all 81 through the guard's own field path (parallel) --- #
    print(f"rendering {n} fields via guard.render_field @ {guard.GUARD_STAT_RES} "
          f"(workers={WORKERS}) ...")
    stats = {}
    reasons = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for oid, st, reason in ex.map(measure_one, distinct):
            stats[oid] = st
            reasons[oid] = reason

    # --- reconstruct the gate sets from the guard --- #
    g_interior = {o for o, r in reasons.items() if r in ("interior", "both")}
    g_flat = {o for o, r in reasons.items() if r in ("flat", "both")}
    g_both = {o for o, r in reasons.items() if r == "both"}
    g_union = {o for o, r in reasons.items() if r is not None}

    print(f"\nguard:    interior={len(g_interior)} flat={len(g_flat)} "
          f"both={len(g_both)} union={len(g_union)}")

    # --- exact-set comparison --- #
    checks = [
        ("interior_gate", g_interior, m_interior),
        ("flat_gate", g_flat, m_flat),
        ("both", g_both, m_both),
        ("union", g_union, m_union),
    ]
    all_ok = True
    diag = load_diag_stats()
    for name, gset, mset in checks:
        if gset == mset:
            print(f"  [OK]   {name}: {len(gset)} IDs match exactly")
            continue
        all_ok = False
        only_guard = sorted(gset - mset)
        only_manifest = sorted(mset - gset)
        print(f"  [FAIL] {name}: guard {len(gset)} vs manifest {len(mset)}")
        for oid in only_guard:
            print(f"         + guard-only  {oid[-6:]}")
        for oid in only_manifest:
            print(f"         - manifest-only {oid[-6:]}")

    if not all_ok:
        print("\n=== DIVERGENCE (in-scorer field path vs diagnostic table.csv) ===")
        diverging = sorted((g_union ^ m_union))
        for oid in diverging:
            gi, gf = stats[oid].interior_frac, stats[oid].field_std
            di, df = diag.get(oid, (float("nan"), float("nan")))
            print(f"  {oid[-6:]}  guard: if={gi:.6f} fs={gf:8.3f}  |  "
                  f"diag: if={di:.6f} fs={df:8.3f}  |  "
                  f"dIF={gi-di:+.2e} dFS={gf-df:+.2e}")
        print("\nHARD GATE FAILED: the guard field path measures differently than the")
        print("diagnostic. Reconcile the render fidelity / maxiter policy before wiring.")
        sys.exit(1)

    print(f"\nHARD GATE PASSED: guard reproduces the drop manifest exactly "
          f"(union {len(g_union)}/{n}; interior {len(g_interior)}, flat {len(g_flat)}, "
          f"both {len(g_both)}).")
    print("  -> Part 1 guard is measurement-faithful; downstream wiring is unblocked.")


if __name__ == "__main__":
    main()
