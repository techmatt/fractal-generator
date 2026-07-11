#!/usr/bin/env python
r"""Co-evolution round — guard-OFF discovery gather on the shipped random-survivor
default, isolated to a dedicated round dir for a clean v6-gap diagnostic snapshot.

Reuses `production_seeder._gather` VERBATIM (native seeder / depth-2 probe / full
walks / k3 reward / v6 CORN decode / density rejection / degenerate-guard-as-prior),
but redirects the durable GATHER_DIR + scratch into `out/coevo_round/<ts>/` so this
round's ledger is a self-contained snapshot that never touches the shared
`data/discovery/gather/` store. Mandelbrot c-plane deg-2 only (NO --julia-hook);
current binary = random-survivor selection default (no --selection flag anywhere).

Resilience mirrors gather_overnight.py: chunked short seeder PROCESSES (each exits
cleanly, reclaiming memory + CUDA before the next), cooldown between chunks, a
0xC0000142 CUDA/DLL-wedge retry, and a hard wall-cap backstop. Cumulative via the
round's own ledger; adaptive stop once the guard-pass target is reached.

  # driver (background this):
  uv run python -u tools/coevo/coevo_round.py --wall-min 150 --target-guardpass 350
  # single worker chunk (spawned by the driver; not usually run by hand):
  uv run python -u tools/coevo/coevo_round.py --worker --round-dir <dir> --budget 6 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDER_DIR = ROOT / "tools" / "atlas"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

WEDGE = 3221225794   # 0xC0000142 STATUS_DLL_INIT_FAILED (CUDA/DLL wedge fast-fail)


# =========================================================================== #
# Worker: one guard-OFF gather chunk into the round dir (own ledger subtree).
# =========================================================================== #
def _worker(args):
    sys.path.insert(0, str(SEEDER_DIR))
    import production_seeder as ps   # heavy (torch): imported only inside the worker

    round_dir = Path(args.round_dir).resolve()
    # Redirect the durable gather subtree + scratch into the round dir. _gather reads
    # these module globals at call time, so reassigning here fully isolates the round:
    #   class_dir = GATHER_DIR/mandelbrot  ->  <round_dir>/gather/mandelbrot
    ps.GATHER_DIR = round_dir / "gather"
    ps.GATHER_SCRATCH_ROOT = ROOT / "out" / "coevo_round" / "_scratch" / round_dir.name

    # Full args namespace for _gather + resolve_family (mandelbrot c-plane, guard-OFF,
    # NO julia-hook: deg-2 only). Matches production_seeder.main()'s --gather path.
    ns = SimpleNamespace(
        smoke=False, run=False, gather=True, time_only=False, finalize=None,
        seed=args.seed, batch=0, budget=args.budget,
        family="mandelbrot", julia=False, c=None, phoenix=False, julia_hook=False,
    )
    fam = ps.resolve_family(ns)
    ps._gather(ns, fam)
    return 0


# =========================================================================== #
# Round ledger recap (source of truth across chunks).
# =========================================================================== #
def round_ledger_path(round_dir: Path) -> Path:
    return round_dir / "gather" / "mandelbrot" / "outcome_ledger.jsonl"


def recap(round_dir: Path) -> dict:
    """Cumulative tally from the round's mandelbrot gather ledger."""
    led = round_ledger_path(round_dir)
    n = 0
    dec = {1: 0, 2: 0, 3: 0}
    guard = {}
    v6 = 0
    if led.exists():
        for line in open(led, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            d = r.get("decoded_class")
            if d in (1, 2, 3):
                dec[d] += 1
            v = r.get("guard_verdict", "?")
            guard[v] = guard.get(v, 0) + 1
            if r.get("scorer_version") == "v6":
                v6 += 1
    return {"n": n, "dec": dec, "guard": guard, "guard_pass": guard.get("pass", 0),
            "v6": v6}


def print_recap(round_dir: Path):
    r = recap(round_dir)
    g = r["guard"]
    gstr = " ".join(f"{k}={v}" for k, v in sorted(g.items()))
    print(f"  cumulative: {r['n']} outcomes  decoded 1/2/3="
          f"{r['dec'][1]}/{r['dec'][2]}/{r['dec'][3]}  guard[{gstr}]  "
          f"v6-stamped={r['v6']}/{r['n']}  | GUARD-PASS={r['guard_pass']}", flush=True)


# =========================================================================== #
# Driver: chunked worker processes, cooldown, wall-cap + guard-pass target.
# =========================================================================== #
def run_chunk(cmd, label, wedge_recovery_s):
    """Run one worker chunk; on a 0xC0000142 fast-fail, sleep the wedge out and retry
    once. Returns the final returncode."""
    for attempt in (1, 2):
        t = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT))
        dt = time.time() - t
        tag = f"EXITED {r.returncode}" if r.returncode != 0 else "done"
        print(f"  -- {label} {tag} in {dt:.0f}s --", flush=True)
        if r.returncode == WEDGE and dt < 30 and attempt == 1:
            print(f"     (CUDA/DLL wedge; recovering {wedge_recovery_s:.0f}s then retry)", flush=True)
            time.sleep(wedge_recovery_s)
            continue
        return r.returncode
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="internal: run one gather chunk")
    ap.add_argument("--round-dir", default=None, help="worker: the round dir to write into")
    ap.add_argument("--budget", type=float, default=6.0, help="worker: per-chunk wallclock min")
    ap.add_argument("--seed", type=int, default=0, help="worker: rng+engine seed / driver base seed")
    # driver knobs
    ap.add_argument("--round-ts", default=None, help="reuse/resume an existing round dir name")
    ap.add_argument("--wall-min", type=float, default=150.0,
                    help="hard wall-cap backstop (min) — stop launching chunks past this")
    ap.add_argument("--target-guardpass", type=int, default=350,
                    help="adaptive stop once this many guard-pass outcomes accrue")
    ap.add_argument("--chunk-min", type=float, default=6.0,
                    help="per-PROCESS budget (short so peak memory stays under the wall + "
                         "each exits cleanly, reclaiming CUDA before the next)")
    ap.add_argument("--cooldown-s", type=float, default=90.0,
                    help="pause between chunks so GPU/CUDA + kernel resources settle")
    ap.add_argument("--wedge-recovery-s", type=float, default=180.0)
    args = ap.parse_args()

    if args.worker:
        if not args.round_dir:
            raise SystemExit("--worker requires --round-dir")
        sys.exit(_worker(args))

    round_ts = args.round_ts or time.strftime("%Y%m%d_%H%M%S")
    round_dir = ROOT / "out" / "coevo_round" / round_ts
    round_dir.mkdir(parents=True, exist_ok=True)

    n_chunks_cap = max(1, math.ceil(args.wall_min / max(1.0, args.chunk_min)))
    print(f"=== coevo round (guard-OFF gather, mandelbrot c-plane, v6) ts={round_ts} ===")
    print(f"round dir : {round_dir}")
    print(f"selection : random-survivor (binary default; NO --selection flag)")
    print(f"stop      : wall-cap {args.wall_min:.0f}min OR guard-pass>={args.target_guardpass}")
    print(f"chunks    : up to {n_chunks_cap} x {args.chunk_min:.0f}min (cooldown {args.cooldown_s:.0f}s)\n", flush=True)

    manifest = {
        "round_ts": round_ts, "mode": "coevo_gather_diagnostic",
        "family": "mandelbrot", "julia_hook": False, "selection": "random-survivor (default)",
        "scorer": "v6 (probe.ACTIVE_CKPT)", "guard": "OFF (verdict logged as prior)",
        "wall_min": args.wall_min, "target_guardpass": args.target_guardpass,
        "chunk_min": args.chunk_min, "cooldown_s": args.cooldown_s, "base_seed": args.seed,
        "ledger": str(round_ledger_path(round_dir)),
    }
    (round_dir / "round_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t0 = time.time()
    ci = 0
    while True:
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min >= args.wall_min:
            print(f"\nWALL-CAP {args.wall_min:.0f}min reached; stopping.", flush=True)
            break
        gp = recap(round_dir)["guard_pass"]
        if gp >= args.target_guardpass:
            print(f"\nTARGET reached: {gp} guard-pass >= {args.target_guardpass}; stopping.", flush=True)
            break
        ci += 1
        seed = args.seed + ci                  # fresh native draw per chunk
        budget = min(args.chunk_min, args.wall_min - elapsed_min)
        cmd = [sys.executable, "-u", str(HERE / "coevo_round.py"), "--worker",
               "--round-dir", str(round_dir), "--budget", str(budget), "--seed", str(seed)]
        print(f"-- chunk {ci} seed={seed} budget={budget:.1f}min "
              f"(elapsed {elapsed_min:.0f}min, guard-pass {gp}/{args.target_guardpass}) --", flush=True)
        run_chunk(cmd, f"chunk {ci}", args.wedge_recovery_s)
        print_recap(round_dir)
        # cooldown unless we're about to stop
        if (time.time() - t0) / 60 < args.wall_min and recap(round_dir)["guard_pass"] < args.target_guardpass:
            time.sleep(args.cooldown_s)

    total = time.time() - t0
    print(f"\n=== ROUND DONE in {total/60:.1f}min over {ci} chunks ===", flush=True)
    print_recap(round_dir)
    print(f"ledger -> {round_ledger_path(round_dir)}", flush=True)


if __name__ == "__main__":
    main()
