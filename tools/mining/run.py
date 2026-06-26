"""Top-level v3-guided biased mining run: descent -> finalize -> labeling batch.

One process, progress-logged to stdout (background it; well over 30s). Loads v3
once and reuses it across both phases.

  uv run python tools/mining/run.py --run run1 2>&1 | tee data/mining/run1/run.log

Defaults are the AGGRESSIVE mining params (bias is the point); none touch the
frozen guided-descend / enrich production defaults.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from score_lib import Scorer  # noqa: E402
from descend import mine_locations  # noqa: E402
from harvest import finalize, GENERATOR_VERSION  # noqa: E402
import corpus_common as cc  # noqa: E402


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--date", default="2026-06-25")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-descent", action="store_true",
                    help="reuse an existing <run>/descent/pool.jsonl")
    # descent (aggressive)
    ap.add_argument("--candidates-n", type=int, default=16)
    ap.add_argument("--beam-k", type=int, default=3)
    ap.add_argument("--landmark-perturbs", type=int, default=3)
    ap.add_argument("--landmark-depth", type=int, default=3)
    ap.add_argument("--secondary-label2", action="store_true")
    ap.add_argument("--root-count", type=int, default=150)
    ap.add_argument("--root-depth", type=int, default=4)
    ap.add_argument("--no-root", action="store_true")
    ap.add_argument("--descent-width", type=int, default=640)
    ap.add_argument("--descent-height", type=int, default=360)
    # finalize
    ap.add_argument("--cap-locations", type=int, default=800)
    ap.add_argument("--budget", type=int, default=450)
    ap.add_argument("--spread-width", type=int, default=640)
    ap.add_argument("--spread-height", type=int, default=360)
    ap.add_argument("--dedup-thresh", type=int, default=6)
    ap.add_argument("--maxiter", type=int, default=8000)
    a = ap.parse_args()

    run_dir = str(ROOT / "data" / "mining" / a.run)
    descent_dir = os.path.join(run_dir, "descent")
    descent_pool = os.path.join(descent_dir, "pool.jsonl")
    batch_id = f"{a.date}_{GENERATOR_VERSION}"
    batch_dir = cc.batch_dir(batch_id)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    log(f"=== v3-guided biased mining: run={a.run} batch_id={batch_id} ===")
    scorer = Scorer()
    log(f"v3 loaded on {scorer.device}")

    if not a.skip_descent:
        log("--- PHASE 1: v3-beam descent ---")
        mine_locations(
            scorer, out_dir=descent_dir, seed=a.seed,
            do_root=not a.no_root, secondary_label2=a.secondary_label2,
            beam_k=a.beam_k, candidates_n=a.candidates_n,
            landmark_perturbs=a.landmark_perturbs, landmark_depth=a.landmark_depth,
            root_count=a.root_count, root_depth=a.root_depth,
            width=a.descent_width, height=a.descent_height, maxiter=a.maxiter, log=log)
        log(f"descent done at {time.time()-t0:.0f}s")
    else:
        log(f"--- PHASE 1 skipped, reusing {descent_pool} ---")

    log("--- PHASE 2: spread / gate / dedup / render / batch ---")
    finalize(
        scorer, descent_pool, batch_id=batch_id, out_batch_dir=batch_dir,
        cap_locations=a.cap_locations, budget=a.budget,
        spread_width=a.spread_width, spread_height=a.spread_height,
        maxiter=a.maxiter, dedup_thresh=a.dedup_thresh, seed=a.seed, log=log)

    log(f"=== DONE in {time.time()-t0:.0f}s  BATCH_ID={batch_id} ===")
    log(f"label with: set corpus_label.html BATCH_ID='{batch_id}', open, export scores.json,")
    log(f"  then: uv run python tools/corpus/merge_scores.py --batch {batch_id} --apply")


if __name__ == "__main__":
    main()
