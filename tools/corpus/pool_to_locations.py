"""Part 3 (bridge) — convert a guided-descend pool.jsonl into a present-shaped
`locations.jsonl`.

present's seed parser wants `keeper_index, draw_index, interior_frac, center_re,
center_im, frame_width`. The pool carries `idx, cx, cy, fw` (+ provenance). We map
idx→keeper_index and idx→draw_index so the present manifest's `seed_index`/
`draw_index` both round-trip back to the pool `idx`, letting build_rev4_batch.py
re-attach provenance by a single key. interior_frac is set to 0 (the pool has no
per-frame measure; present only logs it — the real measures come from present's
own render and land in the store via the manifest).

Run:  uv run python tools/corpus/pool_to_locations.py \
          --pool data/guided_descend/run4/pool.jsonl \
          --out  out/present/run4_bridge/locations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/guided_descend/run4/pool.jsonl")
    ap.add_argument("--out", default="out/present/run4_bridge/locations.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n = 0
    with open(a.pool, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            seed = {
                "keeper_index": c["idx"],
                "draw_index": c["idx"],
                "interior_frac": 0.0,
                "center_re": c["cx"],
                "center_im": c["cy"],
                "frame_width": c["fw"],
            }
            fout.write(json.dumps(seed))
            fout.write("\n")
            n += 1
    print(f"wrote {n} seeds -> {a.out}")


if __name__ == "__main__":
    main()
