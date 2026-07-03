#!/usr/bin/env python
"""Atlas round-2 — outcome-appearance embeddings (the diversity substrate).

Identical to round1_embed but pointed at `data/atlas/round2/`. For every harvested
walk, render its k3-winner's reframed frame at the classifier's search fidelity
(640x360 ss2, twilight_shifted) and embed it with the v5 backbone's PENULTIMATE
features (the 1280-D vector before the CORN head). Save per-arm
{emb[N,1280], walk_id, reward_k3, tag, seed_cx/cy}. Renders are cached on disk.

Reuses round1_embed's `_render`, `embed_paths`, `load_tags` verbatim (same render +
forward-hook path) — only the run dir + tiles dir change.

  uv run python tools/atlas/round2_embed.py --arm arm1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json  # noqa: E402
import round1_embed as r1e  # noqa: E402
from score_lib import Scorer  # noqa: E402

MODEL = r1e.MODEL
D = ROOT / "data" / "atlas" / "round2"
TILES_ROOT = ROOT / "out" / "atlas" / "round2" / "embed_tiles"


def run(arm: str, workers: int):
    table = D / f"{arm}_table.jsonl"
    if not table.exists():
        raise SystemExit(f"no {table}; harvest {arm} first")
    rows = [json.loads(l) for l in open(table, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    tags = r1e.load_tags(D / f"{arm}_seeds.jsonl")

    tiles = TILES_ROOT / arm
    tiles.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in rows:
        p = tiles / f"walk_{r['walk_id']:04d}.jpg"
        if not p.exists():
            todo.append((r["k3_reframed_cx"], r["k3_reframed_cy"], r["k3_reframed_fw"], p))
    print(f"[{arm}] {len(rows)} walks; rendering {len(todo)} embed tiles "
          f"@ {r1e.RENDER_W}x{r1e.RENDER_H} ss{r1e.RENDER_SS}")
    t0 = time.time()
    if todo:
        fails = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(r1e._render, cx, cy, fw, p): p for cx, cy, fw, p in todo}
            for fut in cf.as_completed(futs):
                ok, err = fut.result()
                if not ok:
                    fails.append((futs[fut], err))
        if fails:
            raise SystemExit(f"render failed ({len(fails)}): {fails[0][0].name}: {fails[0][1]}")
    print(f"  rendered in {time.time()-t0:.0f}s; embedding (v5 penultimate) ...")

    scorer = Scorer(model_path=MODEL)
    paths = [tiles / f"walk_{r['walk_id']:04d}.jpg" for r in rows]
    emb = r1e.embed_paths(scorer, paths)
    print(f"  embeddings {emb.shape}")

    np.savez_compressed(
        D / f"{arm}_embed.npz",
        emb=emb.astype(np.float32),
        walk_id=np.array([r["walk_id"] for r in rows]),
        reward_k3=np.array([r["reward_k3"] for r in rows], float),
        reward_k1=np.array([r["reward_k1"] for r in rows], float),
        reached_depth=np.array([r["reached_depth"] for r in rows]),
        seed_cx=np.array([r["seed_cx"] for r in rows], float),
        seed_cy=np.array([r["seed_cy"] for r in rows], float),
        seed_fw=np.array([r["seed_fw"] for r in rows], float),
        tag=np.array([tags.get(r["walk_id"], "") for r in rows]),
    )
    print(f"  -> {D / f'{arm}_embed.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(args.arm, args.workers)


if __name__ == "__main__":
    main()
