#!/usr/bin/env python
"""Atlas round-1 acceptance — k3 best-over-walk harvest for one arm's pool.

Per walk: raw-score (v5) EVERY frame in the pool, take the top-3 by raw score,
REFRAME those 3 (v5), and take reward_k3 = max reframed. This is the SAME reward the
atlas was fit on (step0_reanalysis) — reused VERBATIM: `raw_screen_walk`,
`reframe_location`, `_mand_location`, `make_scorer` (v5, NOT v3), `KRAW`,
`load_frames_by_walk`, `_seed_rows`. The only addition over step0_reanalysis is that
this ALSO records the k3-winner's reframed frame geometry (reframed_cx/cy/fw) so the
diversity pass can embed the exact best frame per outcome.

  uv run python tools/atlas/round1_harvest.py --time  --pool <arm_dir>
  uv run python tools/atlas/round1_harvest.py --build --pool <arm_dir> --out <table.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import step0_reanalysis as sr  # noqa: E402
from step0_reanalysis import (  # noqa: E402
    KRAW, raw_screen_walk, _mand_location, load_frames_by_walk, _seed_rows,
)
from reframe import reframe_location  # noqa: E402
from probe import make_scorer  # noqa: E402

MODEL = "data/classifier/v5/model_best.pt"

FIELDS = ["walk_id", "seed_cx", "seed_cy", "seed_fw", "n_frames",
          "reached_depth", "reward_k1", "reward_k3",
          "k3_argmax_idx", "k3_argmax_depth", "k3_reframed_cx", "k3_reframed_cy",
          "k3_reframed_fw", "raw_max", "raw_mean"]


def harvest_walk(scorer, wid, frames, seed_row, workers, scratch):
    """k3 best-over-walk for one walk, keeping the k3-winner's reframed geometry.
    Mirrors step0_reanalysis.process_walk but returns reframed_cx/cy/fw of the k3 best."""
    sr.SCRATCH = scratch  # raw_screen_walk / reframe workdirs land here
    raws = raw_screen_walk(scorer, wid, frames, workers)
    order = sorted(range(len(frames)), key=lambda i: raws[i], reverse=True)
    topk = order[:KRAW]

    best = None  # (reframed_score, res, depth, idx)
    reward_k1 = None
    for rank, i in enumerate(topk):
        fr = frames[i]
        loc = _mand_location(fr["cx"], fr["cy"], fr["fw"])
        wd = scratch / f"walk_{wid:04d}" / f"reframe_top{rank}"
        res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=workers)
        raw_orig = res.trace["original_score"]
        if res.score < raw_orig - 1e-4:
            raise SystemExit(f"MONOTONICITY VIOLATED walk {wid} idx {fr['idx']}: "
                             f"{res.score:.4f} < {raw_orig:.4f}")
        if rank == 0:
            reward_k1 = float(res.score)
        if best is None or res.score > best[0]:
            best = (float(res.score), res, int(fr["depth"]), int(fr["idx"]))

    reward_k3, res, k3_depth, k3_idx = best
    reached = max(int(f["depth"]) for f in frames)
    return {
        "walk_id": wid,
        "seed_cx": float(seed_row["cx"]), "seed_cy": float(seed_row["cy"]),
        "seed_fw": float(seed_row["fw"]),
        "n_frames": len(frames), "reached_depth": reached,
        "reward_k1": reward_k1, "reward_k3": reward_k3,
        "k3_argmax_idx": k3_idx, "k3_argmax_depth": k3_depth,
        "k3_reframed_cx": float(res.cx), "k3_reframed_cy": float(res.cy),
        "k3_reframed_fw": float(res.fw),
        "raw_max": float(max(raws)), "raw_mean": float(np.mean(raws)),
    }


def run(pool_dir: Path, out: Path, workers: int, time_only: bool, resume: bool):
    by_walk = load_frames_by_walk(pool_dir)
    seeds = _seed_rows(by_walk)
    scratch = ROOT / "out" / "atlas" / "round1" / "_scratch" / pool_dir.name
    scorer = make_scorer(MODEL)
    print(f"loaded {len(by_walk)} walks / {sum(len(v) for v in by_walk.values())} frames "
          f"from {pool_dir}")
    print(f"scorer: v5 CORN ({MODEL})  geometry={scorer.cfg.get('geometry')}")

    wids = sorted(by_walk)
    if time_only:
        per = []
        for wid in wids[:5]:
            t = time.time()
            row = harvest_walk(scorer, wid, by_walk[wid], seeds[wid], workers, scratch)
            el = time.time() - t
            per.append(el)
            print(f"  walk {wid:>4} nframes={len(by_walk[wid]):>2}: {el:.2f}s  "
                  f"k1={row['reward_k1']:.3f} k3={row['reward_k3']:.3f} reached d{row['reached_depth']}")
        avg = sum(per) / len(per)
        print(f"\n  avg {avg:.2f}s/walk -> PROJECTED {len(wids)} walks: ~{avg*len(wids):.0f}s "
              f"(~{avg*len(wids)/60:.1f} min) @ workers={workers}")
        return

    done = set()
    if resume and out.exists():
        for l in open(out, encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["walk_id"])
        print(f"[resume] {len(done)} walks already done")
    todo = [w for w in wids if w not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if done else "w"
    t0 = time.time()
    with open(out, mode, encoding="utf-8") as jf:
        for i, wid in enumerate(todo):
            row = harvest_walk(scorer, wid, by_walk[wid], seeds[wid], workers, scratch)
            jf.write(json.dumps(row) + "\n"); jf.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>4}/{len(todo)}] walk {wid:>4} k3={row['reward_k3']:.3f}  "
                      f"{el:.0f}s elapsed, ETA {eta:.0f}s")
    (out.parent / f"{out.stem}.COMPLETE").write_text(
        f"done {len(wids)} walks in {time.time()-t0:.0f}s\n")
    print(f"\nDONE {len(wids)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path, help="arm pool dir (has pool.jsonl)")
    ap.add_argument("--out", type=Path, help="output table jsonl (build mode)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--time", action="store_true", help="time 5 walks, project total")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.time and args.out is None:
        raise SystemExit("--out required in build mode")
    run(args.pool, args.out, args.workers, args.time, args.resume)


if __name__ == "__main__":
    main()
