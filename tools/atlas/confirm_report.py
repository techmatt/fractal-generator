#!/usr/bin/env python
r"""Confirmatory-harvest post-run report (prompts/cc-prompt-confirmatory-harvest-run.md).

Run-scoped, read-only over the durable ledger + the run's guard_telemetry.jsonl. Adds
the two confirmatory checks the seeder run doesn't itself perform, and folds the salvage
breakdown into one report written to the run dir:

  1. SELF-CONSISTENCY (the direct contrast with the pre-guard 20/81 degenerate): re-run
     the guard predicate (guard.field_measures + guard.guard_fail at GUARD_STAT_RES) on
     every guard-passed outcome this run harvested. The failure count MUST be zero — they
     passed the guard to be harvested; a non-zero is a wiring bug.
  2. OCCUPANCY ACCRUAL: probe_child_occ distribution over this run's descendable survivors
     (non-null == reached depth 2), + the sub-floor fraction (< OCC_FLOOR). Floor-straddling
     survivors are the population that makes the floor question answerable.

Never mutates scoring, the guard, the floor, the caps, or the durable ledger. Reuses the
guard's OWN field path (guard.measure_location) so the re-gate measures identically to the
live scorer.

  uv run python tools/atlas/confirm_report.py                 # latest run
  uv run python tools/atlas/confirm_report.py --run-ts <ts>   # a specific run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import guard  # noqa: E402
import production_seeder as ps  # noqa: E402

OCC_FLOOR = ps.OCC_FLOOR   # 0.321 — pinned; observed here, never changed.


def _rows_for_run(run_ts: str) -> list[dict]:
    return [json.loads(l) for l in open(ps.OUTCOME_LEDGER, encoding="utf-8")
            if l.strip() and json.loads(l).get("ts") == run_ts]


def _latest_run_ts() -> str:
    runs = sorted(p.name for p in ps.RUNS_DIR.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"no runs under {ps.RUNS_DIR}")
    return runs[-1]


def self_consistency(rows: list[dict], scratch: Path) -> dict:
    """Re-gate every guard-passed outcome this run at GUARD_STAT_RES. failures MUST be 0."""
    scratch.mkdir(parents=True, exist_ok=True)
    passed = [r for r in rows if r.get("guard_pass")]
    failures = []
    for i, r in enumerate(passed):
        out_bin = scratch / f"regate_{i:05d}.bin"
        st = guard.measure_location(r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], out_bin,
                                    family=r.get("family", "mandelbrot"))
        reason = guard.guard_fail(st.interior_frac, st.field_std)
        if reason is not None:
            failures.append({"id": r["id"], "reason": reason,
                             "interior_frac": st.interior_frac, "field_std": st.field_std})
    return {"n_guard_passed": len(passed), "n_regate_failures": len(failures),
            "failures": failures}


def occupancy_accrual(rows: list[dict]) -> dict:
    """probe_child_occ distribution over descendable survivors this run + sub-floor frac."""
    occ = [r.get("probe_child_occ") for r in rows]
    non_null = [float(o) for o in occ if o is not None]
    n_null = sum(1 for o in occ if o is None)
    if not non_null:
        return {"n_rows": len(rows), "n_non_null": 0, "n_null": n_null,
                "note": "no non-null probe_child_occ — occupancy emit may be inactive"}
    a = np.array(non_null)
    sub = int((a < OCC_FLOOR).sum())
    return {
        "n_rows": len(rows), "n_non_null": len(non_null), "n_null": n_null,
        "floor": OCC_FLOOR,
        "min": float(a.min()), "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
        "max": float(a.max()), "mean": float(a.mean()),
        "sub_floor_count": sub, "sub_floor_fraction": round(sub / len(non_null), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-ts", default=None, help="run timestamp (default: latest)")
    args = ap.parse_args()

    run_ts = args.run_ts or _latest_run_ts()
    run_dir = ps.RUNS_DIR / run_ts
    rows = _rows_for_run(run_ts)
    if not rows:
        raise SystemExit(f"no outcome rows with ts={run_ts} in {ps.OUTCOME_LEDGER}")

    print(f"=== confirmatory report: run {run_ts} ({len(rows)} scored rows) ===")

    # --- salvage breakdown from the run's guard telemetry (if present) ---
    tele_path = run_dir / "guard_telemetry.jsonl"
    salvage = None
    if tele_path.exists():
        tele = [json.loads(l) for l in open(tele_path, encoding="utf-8") if l.strip()]
        dropped = sum(1 for t in tele if t["k3_is_sentinel"])
        clean = sum(1 for t in tele if not t["k3_is_sentinel"] and t["frames_gated"] == 0)
        salvaged = sum(1 for t in tele if not t["k3_is_sentinel"] and t["frames_gated"] > 0)
        salvage = {"n_walks": len(tele), "clean_harvest": clean,
                   "salvaged_harvest": salvaged, "dropped": dropped,
                   "frames_gated_total": sum(t["frames_gated"] for t in tele)}
        print(f"  salvage breakdown: clean={clean} salvaged={salvaged} dropped={dropped} "
              f"(over {len(tele)} walks; {salvage['frames_gated_total']} frames gated)")
    else:
        print(f"  [!] no guard_telemetry.jsonl in {run_dir} — salvage breakdown unavailable")

    # --- self-consistency re-gate ---
    sc = self_consistency(rows, ps.SCRATCH_ROOT / run_ts / "regate")
    verdict = "PASS (0 degenerate)" if sc["n_regate_failures"] == 0 else \
              f"FAIL — {sc['n_regate_failures']} degenerate harvested (WIRING BUG)"
    print(f"  self-consistency: {sc['n_guard_passed']} guard-passed re-gated -> {verdict}")
    for f in sc["failures"]:
        print(f"     FAIL {f['id']}: {f['reason']} "
              f"(interior={f['interior_frac']:.3f} std={f['field_std']:.2f})")

    # --- occupancy accrual ---
    occ = occupancy_accrual(rows)
    if occ["n_non_null"]:
        print(f"  occupancy: {occ['n_non_null']} non-null / {occ['n_null']} null | "
              f"min={occ['min']:.3f} med={occ['median']:.3f} max={occ['max']:.3f} | "
              f"sub-floor(<{OCC_FLOOR}) {occ['sub_floor_count']}/{occ['n_non_null']} "
              f"= {occ['sub_floor_fraction']:.1%}")
    else:
        print(f"  occupancy: {occ.get('note')}")

    # --- pool size ---
    led = ps.Ledgers()
    pool_size = len(led.harvested)
    print(f"  guard-passed pool (cumulative, was 61): {pool_size}")

    report = {"run_ts": run_ts, "n_scored_rows": len(rows),
              "salvage_breakdown": salvage, "self_consistency": sc,
              "occupancy_accrual": occ, "guard_passed_pool": pool_size}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "confirm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  report -> {run_dir / 'confirm_report.json'}")

    if sc["n_regate_failures"] != 0:
        raise SystemExit("SELF-CONSISTENCY FAILED — a harvested outcome fails the guard re-gate.")
    return report


if __name__ == "__main__":
    main()
