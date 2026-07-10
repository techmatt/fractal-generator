#!/usr/bin/env python
"""Descent ablation + percentile-strategy overnight campaign driver.

Single unattended pass. Freezes ONE root seed-list (native 8k/flat draw), then runs
the priority-ordered arm matrix on the **Mandelbrot c-plane**, every arm sharing that
seed-list + the same --seed + --per-walk-rng (matched roots + matched per-walk
sub-seeds; the diversity-distribution control). Self-terminates at a 6h wall cap with
a ~60min reserve for the finalize stage (contact sheets + probes + report). All
rendering at node/thumb res — never full-res.

Durability: every arm writes its own durable pool.jsonl/walks.jsonl (written by the
Rust binary BEFORE its cosmetic preview stage) under its own dir; the campaign appends
a per-arm row to `campaign.jsonl` as each arm completes. Whatever completed is usable;
finalize runs over whatever arm dirs contain a pool.jsonl. `--finalize-only` re-runs the
report over an existing campaign dir without launching arms.

Arms (priority order; core A0-A4 guaranteed, probes A5-A7 fill remaining budget):
  A0 legacy  0.70,0.10,0.20  least-interior   (current)
  A1 legacy  0.10,0.20,0.70  least-interior   (knob A: weights)
  A2 legacy  0.70,0.10,0.20  random-survivor  (knob B: selection)
  A3 legacy  0.10,0.20,0.70  random-survivor  (A x B)
  A4 percentile 0.60,0.80     random-survivor  (strategy P)
  A5 percentile 0.60,0.80     least-interior   (does the selector mask the finder?)
  A6 percentile 0.40,0.60     random-survivor
  A7 percentile 0.75,0.95     random-survivor
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

EXE = "C:/Code/fractal-generator/target/release/fractal-generator.exe"
COLORMAPS = "data/palettes/clean_colormaps.json"
PREVIEW_PALETTE = "twilight_shifted"  # fixed neutral palette across ALL arms
PREVIEW_WIDTH = 320                   # node/thumb res for the tile previews (geometry eyeball)

ARMS = [
    dict(id="A0", core=True,  finder="legacy",     weights="0.70,0.10,0.20", selection="least-interior",  pct=None,
         desc="current"),
    dict(id="A1", core=True,  finder="legacy",     weights="0.10,0.20,0.70", selection="least-interior",  pct=None,
         desc="knob A (weights)"),
    dict(id="A2", core=True,  finder="legacy",     weights="0.70,0.10,0.20", selection="random-survivor", pct=None,
         desc="knob B (selection)"),
    dict(id="A3", core=True,  finder="legacy",     weights="0.10,0.20,0.70", selection="random-survivor", pct=None,
         desc="A x B"),
    dict(id="A4", core=True,  finder="percentile", weights=None,             selection="random-survivor", pct="0.60,0.80",
         desc="strategy P"),
    dict(id="A5", core=False, finder="percentile", weights=None,             selection="least-interior",  pct="0.60,0.80",
         desc="does selector mask finder?"),
    dict(id="A6", core=False, finder="percentile", weights=None,             selection="random-survivor", pct="0.40,0.60",
         desc="lower band"),
    dict(id="A7", core=False, finder="percentile", weights=None,             selection="random-survivor", pct="0.75,0.95",
         desc="higher band"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def base_flags(out_dir):
    return [
        "guided-descend",
        "--per-walk-rng",
        "--colormaps", COLORMAPS,
        "--preview-palette", PREVIEW_PALETTE,
        "--preview-width", str(PREVIEW_WIDTH),
        "--out-dir", str(out_dir),
    ]


def arm_flags(arm):
    f = ["--finder", arm["finder"], "--selection", arm["selection"]]
    if arm["weights"] is not None:
        f += ["--branch-weights", arm["weights"]]
    if arm["pct"] is not None:
        f += ["--pct-band", arm["pct"]]
    return f


def run_exe(flags, log_path, timeout):
    """Run the binary; capture stdout+stderr to log_path. Return (rc, elapsed, timed_out)."""
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        try:
            p = subprocess.run([EXE] + flags, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=timeout, cwd="C:/Code/fractal-generator")
            return p.returncode, time.time() - t0, False
        except subprocess.TimeoutExpired:
            return None, time.time() - t0, True


def harvest_seeds(campaign, seed, n_harvest, timeout):
    """Native (8k/flat) run → depth-1 (cx,cy,fw) roots → seeds.jsonl. Returns list of rows."""
    hdir = campaign / "_harvest"
    hdir.mkdir(parents=True, exist_ok=True)
    flags = base_flags(hdir) + ["--n-walks", str(n_harvest), "--seed", str(seed)]
    log(f"harvest: native root draw, {n_harvest} walks -> {hdir}")
    rc, el, to = run_exe(flags, campaign / "_harvest.log", timeout)
    log(f"harvest done rc={rc} timed_out={to} ({el:.0f}s)")
    pool = hdir / "pool.jsonl"
    if not pool.exists():
        raise SystemExit("harvest produced no pool.jsonl — cannot build seed-list")
    roots = []
    for line in open(pool, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["depth"] == 1 and all(r[k] is not None for k in ("cx", "cy", "fw")):
            roots.append({"cx": r["cx"], "cy": r["cy"], "fw": r["fw"]})
    return roots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out/descent_ablation")
    ap.add_argument("--cap-seconds", type=float, default=6 * 3600)
    ap.add_argument("--reserve-seconds", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--timestamp", default=None, help="reuse an existing campaign dir name")
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--pilot-walks", type=int, default=10)
    ap.add_argument("--r-override", type=int, default=None,
                    help="force R (walks/arm), bypassing budget sizing (smoke tests)")
    args = ap.parse_args()

    t_start = time.time()
    deadline = t_start + args.cap_seconds
    work_deadline = deadline - args.reserve_seconds  # stop launching arms after this

    ts = args.timestamp or time.strftime("%Y-%m-%d_%H%M%S")
    campaign = Path(args.out_root) / ts
    campaign.mkdir(parents=True, exist_ok=True)
    arms_dir = campaign / "arms"
    arms_dir.mkdir(exist_ok=True)
    ledger = campaign / "campaign.jsonl"

    if args.finalize_only:
        import finalize
        finalize.finalize(campaign)
        return

    log(f"campaign dir: {campaign}")
    log(f"cap {args.cap_seconds/3600:.1f}h, reserve {args.reserve_seconds/60:.0f}min, seed {args.seed}")

    # --- Phase 0a: timing pilot (native, small) → per-walk cost estimate ---------
    pilot = campaign / "_pilot"
    pilot.mkdir(exist_ok=True)
    pflags = base_flags(pilot) + ["--n-walks", str(args.pilot_walks), "--seed", str(args.seed + 1)]
    log(f"pilot: {args.pilot_walks} native walks for timing")
    rc, pilot_el, to = run_exe(pflags, campaign / "_pilot.log", timeout=args.cap_seconds * 0.1)
    if rc != 0 or to:
        log(f"WARNING pilot rc={rc} timed_out={to}; falling back to t_walk=15s")
        t_walk = 15.0
    else:
        t_walk = max(0.5, pilot_el / max(1, args.pilot_walks))
    log(f"pilot elapsed {pilot_el:.0f}s -> t_walk ~= {t_walk:.1f}s/walk")

    # --- Phase 0b: size R and harvest the frozen seed-list -----------------------
    # Budget model: overhead(pilot+harvest ~1.2R) + 5 core arms (5R) fit in ~0.8*usable.
    usable = args.cap_seconds - args.reserve_seconds
    if args.r_override is not None:
        R = args.r_override
        log(f"R override = {R} walks/arm (budget sizing bypassed)")
    else:
        R = int((0.8 * usable / t_walk - args.pilot_walks) / 6.2)
        R = max(40, min(250, R))
        log(f"sized R = {R} walks/arm (usable {usable/3600:.1f}h, t_walk {t_walk:.1f}s)")

    n_harvest = math.ceil(R * 1.2)
    roots = harvest_seeds(campaign, args.seed, n_harvest, timeout=args.cap_seconds * 0.35)
    if len(roots) < R:
        log(f"harvest yielded {len(roots)} roots < R={R}; using all {len(roots)}")
        R = len(roots)
    roots = roots[:R]
    seeds_path = campaign / "seeds.jsonl"
    with open(seeds_path, "w", encoding="utf-8") as f:
        for r in roots:
            f.write(json.dumps(r) + "\n")
    log(f"froze {len(roots)} roots -> {seeds_path}")

    manifest = dict(
        timestamp=ts, seed=args.seed, R=R, t_walk_pilot=t_walk,
        cap_seconds=args.cap_seconds, reserve_seconds=args.reserve_seconds,
        preview_palette=PREVIEW_PALETTE, preview_width=PREVIEW_WIDTH,
        seeds=str(seeds_path), arms=[a["id"] for a in ARMS], node_width=768,
    )
    (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- Phase 1: run arms in priority order, budget-gated -----------------------
    arm_cost = R * t_walk
    hard_kill = max(600.0, arm_cost * 5.0)
    for arm in ARMS:
        now = time.time()
        remaining = work_deadline - now
        est = arm_cost * 1.25
        if not arm["core"] and remaining < est:
            log(f"skip {arm['id']} (probe): remaining {remaining/60:.0f}min < est {est/60:.0f}min")
            continue
        if arm["core"] and remaining < arm_cost * 0.5:
            # Even a core arm can't be guaranteed if we're nearly out of budget.
            log(f"skip {arm['id']} (CORE) — insufficient budget: remaining {remaining/60:.0f}min")
            continue
        adir = arms_dir / arm["id"]
        adir.mkdir(exist_ok=True)
        flags = base_flags(adir) + ["--seed-list", str(seeds_path), "--seed", str(args.seed)] + arm_flags(arm)
        log(f"RUN {arm['id']} ({arm['desc']}): finder={arm['finder']} sel={arm['selection']} "
            f"w={arm['weights']} pct={arm['pct']} | remaining {remaining/60:.0f}min")
        rc, el, to = run_exe(flags, adir / "run.log", timeout=hard_kill)
        n_cands = 0
        pool = adir / "pool.jsonl"
        if pool.exists():
            n_cands = sum(1 for _ in open(pool, encoding="utf-8"))
        row = dict(arm=arm["id"], rc=rc, timed_out=to, elapsed=round(el, 1),
                   candidates=n_cands, ok=(rc == 0 and pool.exists()),
                   t_end=round(time.time() - t_start, 1), **{k: arm[k] for k in
                   ("finder", "weights", "selection", "pct", "core", "desc")})
        with open(ledger, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(row) + "\n")
        log(f"  {arm['id']} done rc={rc} timed_out={to} cands={n_cands} ({el:.0f}s)")

    # --- Phase 2: finalize (gated — always attempted within reserve) -------------
    log(f"arm phase complete at {(time.time()-t_start)/60:.0f}min; finalizing")
    sys.path.insert(0, str(Path(__file__).parent))
    import finalize
    finalize.finalize(campaign)
    log(f"CAMPAIGN COMPLETE in {(time.time()-t_start)/60:.0f}min -> {campaign}")


if __name__ == "__main__":
    main()
