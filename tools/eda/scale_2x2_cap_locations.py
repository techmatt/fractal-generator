# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
"""Phase 2/3 prep — uniform-random subsample each cell's pool to a capped
present-shaped locations.jsonl. The present pass renders 2-3 full AA frames/seed,
so presenting all ~2274 candidates is ~5.5h; capping to N/cell (uniform, seeded)
is the logged cap. ~98% present-accept means N=200/cell comfortably clears the
150/cell label target + a tight v3 not-bad sample. The cap is uniform so the
per-cell good-rate read stays unbiased (uniform subsample of a uniform pool).

  uv run python tools/eda/scale_2x2_cap_locations.py [--per-cell 200] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import random

POOL_BASE = "data/guided_descend/scale_2x2"
PRESENT_BASE = "out/present/scale_2x2"
CELLS = ["a", "b", "c", "d"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    total_in = total_out = 0
    for cell in CELLS:
        pool_path = os.path.join(POOL_BASE, f"cell_{cell}", "pool.jsonl")
        rows = [json.loads(l) for l in open(pool_path, encoding="utf-8") if l.strip()]
        rng = random.Random(a.seed)
        rng.shuffle(rows)
        pick = rows[: a.per_cell]
        out_dir = os.path.join(PRESENT_BASE, f"cell_{cell}")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "locations.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for c in pick:
                f.write(json.dumps({
                    "keeper_index": c["idx"], "draw_index": c["idx"],
                    "interior_frac": 0.0,
                    "center_re": c["cx"], "center_im": c["cy"], "frame_width": c["fw"],
                }))
                f.write("\n")
        total_in += len(rows)
        total_out += len(pick)
        print(f"cell {cell}: pool={len(rows)} -> capped={len(pick)} -> {out}")
    print(f"TOTAL: {total_in} candidates -> {total_out} presented (cap {a.per_cell}/cell, seed {a.seed})")
    print(f"DROPPED {total_in - total_out} candidates (logged cap; uniform subsample, unbiased per cell)")


if __name__ == "__main__":
    main()
