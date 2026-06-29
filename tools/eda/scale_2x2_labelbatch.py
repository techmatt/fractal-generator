# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
"""Phase 3 — prepare the UNBIASED labeling batch for the scale-controlled 2x2.

Balanced, uniform-random sample of N per cell (default 150) from the full 2x2
corpus batch. NO v3-filtering (the good-rate read must come from a uniform sample
or it is biased). Each crop keeps its cell-tagged provenance. Writes a standalone
batch dir openable in tools/viz/corpus_label.html.

  uv run python tools/eda/scale_2x2_labelbatch.py [--per-cell 150] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus"))
import corpus_common as cc  # noqa: E402

SRC_BATCH_ID = "2026-06-25_scale_controlled_2x2"
DST_BATCH_ID = "2026-06-25_scale_2x2_labelset"
CELL_ORDER = ["A", "B", "C", "D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src_dir = cc.batch_dir(SRC_BATCH_ID)
    src_crops = os.path.join(src_dir, "crops")
    rows = cc.read_jsonl(os.path.join(src_dir, "images.jsonl"))

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["provenance"]["cell"]].append(r)

    rng = random.Random(a.seed)
    sampled = []
    taken = Counter()
    for cell in CELL_ORDER:
        pool = by_cell.get(cell, [])
        rng.shuffle(pool)
        pick = pool[: a.per_cell]
        sampled.extend(pick)
        taken[cell] = len(pick)
        if len(pool) < a.per_cell:
            print(f"  NOTE cell {cell}: only {len(pool)} available (< {a.per_cell})")

    dst_dir = cc.batch_dir(DST_BATCH_ID)
    dst_crops = os.path.join(dst_dir, "crops")
    os.makedirs(dst_crops, exist_ok=True)

    n_copied = 0
    for r in sampled:
        uid = r["image_id"]
        src_jpg = os.path.join(src_crops, uid + ".jpg")
        dst_jpg = os.path.join(dst_crops, uid + ".jpg")
        if os.path.exists(src_jpg) and not os.path.exists(dst_jpg):
            shutil.copy2(src_jpg, dst_jpg)
            n_copied += 1

    cc.write_jsonl(sampled, os.path.join(dst_dir, "images.jsonl"))
    json.dump({}, open(os.path.join(dst_dir, "scores.json"), "w", encoding="utf-8"))
    json.dump({
        "batch_id": DST_BATCH_ID,
        "schema_version": 1,
        "created": "2026-06-25",
        "labeler": None,
        "derived_from": SRC_BATCH_ID,
        "note": "unbiased uniform-random sample per cell for hand-labeling the 2x2 good-rate. NO v3 filtering.",
        "sampling": {"per_cell": a.per_cell, "seed": a.seed},
        "counts": {"units": len(sampled), "by_cell": dict(taken)},
    }, open(os.path.join(dst_dir, "batch.json"), "w", encoding="utf-8"), indent=2)

    print(f"label batch -> {dst_dir}")
    print(f"  units={len(sampled)}  by_cell={dict(taken)}  crops_copied={n_copied}")
    print("\n=== OPEN INSTRUCTIONS ===")
    print(f"1. Edit tools/viz/corpus_label.html line ~84:  const BATCH_ID='{DST_BATCH_ID}';")
    print("2. From the repo root, serve over http (fetch is blocked under file://):")
    print("     uv run python -m http.server 8000")
    print("3. Open:  http://localhost:8000/tools/viz/corpus_label.html")
    print("4. Label all units (bad/okay/good = 1/2/3); the page exports scores to localStorage.")
    print("   Export scores.json, then merge with tools/corpus/merge_scores.py.")


if __name__ == "__main__":
    main()
