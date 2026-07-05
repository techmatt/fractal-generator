#!/usr/bin/env python
r"""Overnight v6 gathering driver — 5 sequential, time-boxed guard-OFF harvest processes.

Drives `production_seeder.py --gather` across all 9 fractal classes in five sequential
processes (fresh model/GPU state each), each capped at `--minutes-per-class`. Guard is
OFF throughout (raw v5 scoring; the guard would-pass verdict is logged per outcome as a
prior, not a gate). The Julia classes ride their parameter twin's window via the hook:

  1. mandelbrot  + --julia-hook  -> mandelbrot outcomes AND julia (d2, from qualifying parents)
  2. multibrot3  + --julia-hook  -> multibrot3 + julia_multibrot3
  3. multibrot4  + --julia-hook  -> multibrot4 + julia_multibrot4
  4. multibrot5  + --julia-hook  -> multibrot5 + julia_multibrot5
  5. phoenix     (native, no hook, fixed Ushiki location)

Harvest + persistence only. Selection (80/20 best/random, dedup, disagreement slice) and
the labeling UI are a separate follow-up. Floored-and-weighted per-class allocation is a
selection-time concern; here every class just oversamples within its time box.

  uv run python tools/atlas/gather_overnight.py --smoke                 # ~1-2 min/class end-to-end check
  uv run python tools/atlas/gather_overnight.py --minutes-per-class 50  # the real overnight run (background it)
  uv run python tools/atlas/gather_overnight.py --only phoenix --smoke  # one class
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDER = HERE / "production_seeder.py"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# (class key, seeder --gather flags). Order = the driver's sequential order.
CLASSES = [
    ("mandelbrot", ["--family", "mandelbrot", "--julia-hook"]),
    ("multibrot3", ["--family", "multibrot3", "--julia-hook"]),
    ("multibrot4", ["--family", "multibrot4", "--julia-hook"]),
    ("multibrot5", ["--family", "multibrot5", "--julia-hook"]),
    ("phoenix",    ["--phoenix"]),
]

GATHER_DIR = ROOT / "data" / "discovery" / "gather"


def ledger_recap(class_key: str) -> dict:
    """CUMULATIVE tally straight from the class's durable gather ledger (spans every
    chunk + prior runs), partitioned by family. The per-chunk summary.json only covers
    one process; the ledger is the source of truth for what was harvested."""
    led = GATHER_DIR / class_key / "outcome_ledger.jsonl"
    by_fam = {}
    if led.exists():
        for line in open(led, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fam = r.get("family", "?")
            d = by_fam.setdefault(fam, {"n": 0, "dec": {1: 0, 2: 0, 3: 0},
                                        "guard": {}})
            d["n"] += 1
            dc = r.get("decoded_class")
            if dc in (1, 2, 3):
                d["dec"][dc] += 1
            v = r.get("guard_verdict", "?")
            d["guard"][v] = d["guard"].get(v, 0) + 1
    return by_fam


def report_class(class_key: str):
    by_fam = ledger_recap(class_key)
    if not by_fam:
        print(f"  [{class_key}] no ledger rows yet")
        return
    for fam in sorted(by_fam):
        d = by_fam[fam]
        dec, g = d["dec"], d["guard"]
        gstr = " ".join(f"{k}={v}" for k, v in sorted(g.items()))
        print(f"  [{fam}] {d['n']} outcomes  decoded 1/2/3="
              f"{dec[1]}/{dec[2]}/{dec[3]}  guard[{gstr}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes-per-class", type=float, default=30.0,
                    help="TOTAL wallclock budget per class (overnight: 45-60 is fine), spread "
                         "across --chunk-minutes chunks")
    ap.add_argument("--chunk-minutes", type=float, default=12.0,
                    help="per-PROCESS budget: each class runs ceil(total/chunk) short seeder "
                         "processes so peak memory stays well under the exhaustion wall (~29min "
                         "observed on this 8GB/32GB box) and each exits CLEANLY, fully reclaiming "
                         "memory + CUDA before the next chunk. Cumulative via the durable ledger.")
    ap.add_argument("--smoke", action="store_true",
                    help="run each class as a short --smoke pass (end-to-end check, no budget)")
    ap.add_argument("--seed", type=int, default=0, help="base seed (offset per class + chunk)")
    ap.add_argument("--only", default=None,
                    help="comma-separated class keys to run (default all 5)")
    ap.add_argument("--cooldown-seconds", type=float, default=90.0,
                    help="pause between chunks so the GPU/CUDA + kernel resource recovers before "
                         "the next process inits torch (15s was too short on this box)")
    ap.add_argument("--wedge-recovery-seconds", type=float, default=180.0,
                    help="on a 0xC0000142 fast-fail (CUDA/DLL wedge), sleep this long then retry "
                         "the chunk once")
    ap.add_argument("--passes", type=int, default=1,
                    help="repeat the whole class sweep N times (each pass is cumulative via the "
                         "ledger) — lets an overnight run keep accreting after one sweep")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    classes = [c for c in CLASSES if only is None or c[0] in only]
    if not classes:
        raise SystemExit(f"--only matched no classes; valid: {[c[0] for c in CLASSES]}")

    import math
    chunk_min = max(1.0, args.chunk_minutes)
    n_chunks = 1 if args.smoke else max(1, math.ceil(args.minutes_per_class / chunk_min))
    passes = max(1, args.passes)
    mode = "SMOKE" if args.smoke else f"{args.minutes_per_class:.0f}min/class in {n_chunks}x{chunk_min:.0f}min chunks x {passes} pass(es)"
    eta_min = 0 if args.smoke else passes * len(classes) * args.minutes_per_class * 1.25  # +25% (restarts + cooldowns)
    print(f"=== overnight v6 gather ({mode}) — {len(classes)} classes, guard OFF ===")
    print(f"classes: {', '.join(c[0] for c in classes)}")
    if not args.smoke:
        print(f"RUNTIME ETA: ~{eta_min:.0f} min (~{eta_min/60:.1f} h)  = {passes} pass x {len(classes)} "
              f"classes x {args.minutes_per_class:.0f}min + restart/cooldown overhead")
    print(f"gather dir: {GATHER_DIR}\n")

    # 0xC0000142 (STATUS_DLL_INIT_FAILED) — a fresh process can't init its torch/CUDA
    # DLLs because ~10-12min of heavy render-one spawning exhausted a slow-recovering
    # Windows kernel resource (desktop heap / GPU driver state). It DOES recover on its
    # own given idle time, so a wedged chunk is retried once after a long recovery sleep.
    WEDGE = 3221225794

    def run_chunk(cmd, label):
        """Run one seeder chunk; on a 0xC0000142 fast-fail, sleep out the wedge and retry
        once. Returns the final returncode."""
        for attempt in (1, 2):
            t = time.time()
            r = subprocess.run(cmd, cwd=str(ROOT))
            dt = time.time() - t
            tag = f"EXITED {r.returncode}" if r.returncode != 0 else "done"
            print(f"  -- {label} {tag} in {dt:.0f}s --")
            if r.returncode == WEDGE and dt < 30 and attempt == 1:
                print(f"     (CUDA/DLL wedge; recovering {args.wedge_recovery_seconds:.0f}s then retry)")
                time.sleep(args.wedge_recovery_seconds)
                continue
            return r.returncode
        return r.returncode

    t_all = time.time()
    for p in range(passes):
        if passes > 1:
            print(f"\n########## PASS {p + 1}/{passes} ##########", flush=True)
        for i, (key, flags) in enumerate(classes):
            print(f"\n===== [{i + 1}/{len(classes)}] class={key} "
                  f"({n_chunks} chunk{'s' if n_chunks > 1 else ''}) =====")
            remaining = args.minutes_per_class
            for ci in range(n_chunks):
                # Seed varies per pass/class/chunk so each explores a different native draw.
                seed = args.seed + 100000 * p + 1000 * i + ci
                cmd = [sys.executable, str(SEEDER), "--gather"] + flags + ["--seed", str(seed)]
                if args.smoke:
                    cmd += ["--smoke"]
                else:
                    chunk_budget = min(chunk_min, remaining)
                    if chunk_budget <= 0:
                        break
                    remaining -= chunk_budget
                    cmd += ["--budget", str(chunk_budget)]
                print(f"  -- chunk {ci + 1}/{n_chunks} seed={seed} "
                      f"{'(smoke)' if args.smoke else f'budget={cmd[-1]}min'} --", flush=True)
                run_chunk(cmd, f"chunk {ci + 1}")
                # Cooldown so the GPU/CUDA + OS kernel resources settle before the next
                # process inits its own torch/CUDA context (15s was too short — the resource
                # recovers on the order of a minute). Skip only at the very end of the run.
                last = (p + 1 == passes) and (i + 1 == len(classes)) and (ci + 1 == n_chunks)
                if not last:
                    time.sleep(args.cooldown_seconds)
            print(f"  cumulative [{key}]:", flush=True)
            report_class(key)

    total = time.time() - t_all
    print(f"\n=== ALL PASSES DONE in {total:.0f}s ({total/60:.1f} min) ===")
    print("cumulative per-class recap (from durable ledgers):")
    for key, _ in classes:
        report_class(key)
    print(f"\ngather ledgers under {GATHER_DIR}/<class>/outcome_ledger.jsonl")
    print("next (separate follow-up): selection 80/20 + dedup + disagreement slice + labeling UI.")


if __name__ == "__main__":
    main()
