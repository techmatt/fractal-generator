#!/usr/bin/env python
"""Monitored, bounded v6 discovery harvest at the per-degree q3 threshold (Part B).

Runs standing production discovery (production_seeder.py --run, guard ON) SEQUENTIALLY
across the four parameter-plane families, each with --julia-hook, so every c-plane run
also yields its julia:{fam} children — 8 partitions from 4 runs. The per-degree t_good
lookup routes deg-2 (mandelbrot / julia:mandelbrot) to 0.24 and every higher degree to
0.50 automatically inside the seeder.

SEQUENTIAL is mandatory, not a choice: all runs share the one durable ledger + feats npz
(atomic rewrite per run), and the resource note warns concurrent CPU-heavy descents
contend badly (a 113-min outlier). One family at a time keeps the machine sane.

Bounded: small batches + a modest per-family wallclock budget. The budget is a
between-batch floor (a single deep batch can overshoot it — see the 37-min single-batch
outlier), so total wallclock is roughly (families x [budget .. budget + one batch]).

Writes a manifest (out/atlas/v6_monitored_harvest/manifest.json) mapping each family to
its run_ts + summary path, consumed by harvest_report.py. Self-monitoring: each family's
full stdout is teed to a per-family log; a heartbeat + running tally prints here.

  uv run python tools/v6/monitored_harvest.py            # the bounded pass
  uv run python tools/v6/monitored_harvest.py --smoke    # tiny 1-family shakeout
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDER = ROOT / "tools" / "atlas" / "production_seeder.py"
RUNS_DIR = ROOT / "data" / "discovery" / "runs"
OUT_DIR = ROOT / "out" / "atlas" / "v6_monitored_harvest"

# The four parameter-plane families. Phoenix is seeder-exempt (no plane) and left out.
FAMILIES = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"]

# Bounded budget: small batch (finer budget granularity than the default 24) + a modest
# per-family cap. ~20 min/family floor; with the julia hook + a possible overshoot batch,
# expect ~2-3.5 h total across the four — a read-then-scale calibration pass, not overnight.
BUDGET_MIN = 20
BATCH_SEEDS = 12


def newest_run_ts(before: set[str]) -> str | None:
    """The run dir that appeared since `before` (the run just finished writes one)."""
    now = {p.name for p in RUNS_DIR.iterdir()} if RUNS_DIR.exists() else set()
    fresh = sorted(now - before)
    return fresh[-1] if fresh else None


def run_family(fam: str, budget: int, batch: int, seed: int, smoke: bool) -> dict:
    before = {p.name for p in RUNS_DIR.iterdir()} if RUNS_DIR.exists() else set()
    log = OUT_DIR / f"{fam}.log"
    # -u + PYTHONUNBUFFERED: the child's stdout is a PIPE (block-buffered by default), which
    # would starve the line-by-line tee below and defeat live monitoring. Force it unbuffered.
    cmd = [sys.executable, "-u", str(SEEDER), "--run", "--julia-hook",
           "--family", fam, "--seed", str(seed),
           "--budget", str(budget), "--batch", str(batch)]
    if smoke:
        cmd = [sys.executable, "-u", str(SEEDER), "--smoke", "--julia-hook",
               "--family", fam, "--seed", str(seed)]
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] START {fam}\n  {' '.join(cmd)}\n"
          f"  log -> {log}\n{'='*70}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace", env=child_env)
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            s = line.rstrip()
            # surface the batch heartbeats + the run summary block to this process's stdout.
            if any(k in s for k in ("batch ", "RUN SUMMARY", "q3 cloud", "julia-hook",
                                    "GLOBAL SATURATION", "budget", "guard telemetry",
                                    "summary ->")):
                print(f"  [{fam}] {s}", flush=True)
        rc = proc.wait()
    dt = time.time() - t0
    run_ts = newest_run_ts(before)
    summary_path = str(RUNS_DIR / run_ts / "summary.json") if run_ts else None
    entry = {"family": fam, "rc": rc, "wallclock_s": round(dt, 1),
             "run_ts": run_ts, "summary": summary_path, "log": str(log)}
    ok = rc == 0 and run_ts is not None
    print(f"[{time.strftime('%H:%M:%S')}] DONE {fam}  rc={rc} {dt/60:.1f}min "
          f"run_ts={run_ts} {'OK' if ok else 'CHECK LOG'}", flush=True)
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny 1-family (mandelbrot) shakeout")
    ap.add_argument("--budget", type=int, default=BUDGET_MIN)
    ap.add_argument("--batch", type=int, default=BATCH_SEEDS)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--families", nargs="+", default=None,
                    help="override the family list (default: the 4 parameter planes)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing manifest and start a new pass (default: resume)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fams = args.families or (["mandelbrot"] if args.smoke else FAMILIES)

    # Resume: a long single process can be killed (wall-clock caps). Each completed family
    # is durably recorded in the manifest, so on relaunch we carry forward its entry and
    # skip re-running it — the pass converges across as many relaunches as it takes.
    manifest_path = OUT_DIR / "manifest.json"
    runs: list[dict] = []
    done: set[str] = set()
    if manifest_path.exists() and not args.fresh:
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        for e in prev.get("runs", []):
            if e.get("rc") == 0 and e.get("run_ts"):
                runs.append(e)
                done.add(e["family"])
        started = prev.get("started", time.strftime("%Y%m%d_%H%M%S"))
        if done:
            print(f"RESUME: carrying forward completed families {sorted(done)}")
    else:
        started = time.strftime("%Y%m%d_%H%M%S")

    print(f"=== v6 monitored bounded harvest ===  start {started}")
    print(f"families: {fams}  budget={args.budget}min/family batch={args.batch} "
          f"seed={args.seed}  (guard ON, --julia-hook, per-degree t_good)")

    t0 = time.time()
    for fam in fams:
        if fam in done:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP {fam} (already complete in manifest)")
            continue
        entry = run_family(fam, args.budget, args.batch, args.seed, args.smoke)
        runs.append(entry)
        # persist manifest incrementally so a crash mid-pass still leaves a usable report input.
        manifest = {"started": started, "budget_min": args.budget, "batch": args.batch,
                    "seed": args.seed, "families": fams, "runs": runs}
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_min = (time.time() - t0) / 60
    print(f"\n=== harvest complete === {total_min:.1f}min total over {len(runs)} families")
    for e in runs:
        print(f"  {e['family']:<14} rc={e['rc']} {e['wallclock_s']/60:5.1f}min "
              f"run_ts={e['run_ts']}")
    print(f"\nmanifest -> {OUT_DIR / 'manifest.json'}")
    print(f"report   -> uv run python tools/v6/harvest_report.py")
    return 0 if all(e["rc"] == 0 and e["run_ts"] for e in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
