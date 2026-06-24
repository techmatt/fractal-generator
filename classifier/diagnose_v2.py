"""v2 diagnosis — cross-batch corpus census before any training.

Reports per-batch label distributions, provenance fields available for grouping,
walk_id presence in rev4, crop-file existence, and the grouping key plan.
Read-only. Run:  uv run python -m classifier.diagnose_v2
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "corpus"))
import corpus_common as cc  # noqa: E402
import corpus_reader as cr  # noqa: E402


def main():
    print("=== corpus_reader.iter_labeled census ===")
    crops = list(cr.iter_labeled())
    print(f"total labeled crops: {len(crops)}")

    by_batch = defaultdict(list)
    for c in crops:
        by_batch[c.batch_id].append(c)

    missing_total = 0
    for batch_id, items in sorted(by_batch.items()):
        labs = Counter(c.score for c in items)
        miss = [c for c in items if not os.path.exists(c.crop_path)]
        missing_total += len(miss)
        print(f"\n--- {batch_id} ---")
        print(f"  labeled crops: {len(items)}")
        print(f"  label dist (1/2/3): {labs.get(1,0)} / {labs.get(2,0)} / {labs.get(3,0)}")
        print(f"  missing crop files: {len(miss)}")
        if miss:
            for m in miss[:5]:
                print(f"    MISSING {m.crop_path}")

    print(f"\nTOTAL missing crop files: {missing_total}")

    # union label distribution
    union = Counter(c.score for c in crops)
    print(f"\nUNION label dist (1/2/3): {union.get(1,0)} / {union.get(2,0)} / {union.get(3,0)}")

    # provenance fields — read raw jsonl (corpus_reader does NOT expose provenance, by design)
    print("\n=== provenance fields per batch (raw jsonl; for eval grouping ONLY) ===")
    for images_path in cr._batch_images(cc.CORPUS_DIR):
        batch_id = os.path.basename(os.path.dirname(images_path))
        rows = cc.read_jsonl(images_path)
        labeled = [r for r in rows if r.get("label", {}).get("score") is not None]
        print(f"\n--- {batch_id} ({len(labeled)} labeled / {len(rows)} units) ---")
        # which provenance keys are non-null on labeled rows?
        nonnull = {}
        for k in cc.PROVENANCE_KEYS:
            n = sum(1 for r in labeled if r.get("provenance", {}).get(k) is not None)
            nonnull[k] = n
        print("  provenance non-null counts (labeled rows):")
        for k in cc.PROVENANCE_KEYS:
            flag = "" if nonnull[k] else "   <ALL NULL>"
            print(f"    {k:20s}: {nonnull[k]}{flag}")
        # grouping-relevant
        seeds = set(r["provenance"].get("seed_index") for r in labeled)
        walks = set(r["provenance"].get("walk_id") for r in labeled)
        print(f"  distinct seed_index (labeled): {len(seeds)}")
        print(f"  distinct walk_id (labeled): {len(walks)}  values sample: "
              f"{sorted(w for w in walks if w is not None)[:12]}")
        # for rev4: crops per walk, depth spread, label-by-depth
        if any(r["provenance"].get("walk_id") is not None for r in labeled):
            per_walk = Counter(r["provenance"]["walk_id"] for r in labeled)
            print(f"  crops/walk: min={min(per_walk.values())} max={max(per_walk.values())} "
                  f"mean={sum(per_walk.values())/len(per_walk):.1f} n_walks={len(per_walk)}")
            depths = Counter(r["provenance"].get("depth") for r in labeled)
            print(f"  depth dist: {dict(sorted(depths.items(), key=lambda x:(x[0] is None, x[0])))}")
            # does a single walk span multiple labels? (within-walk correlation evidence)
            walk_labels = defaultdict(set)
            for r in labeled:
                walk_labels[r["provenance"]["walk_id"]].add(r["label"]["score"])
            multi = sum(1 for v in walk_labels.values() if len(v) > 1)
            print(f"  walks spanning >1 distinct label: {multi}/{len(walk_labels)}")

    # cross-batch seed namespace collision check
    print("\n=== seed namespace collision (why grouping must be batch-qualified) ===")
    seedsets = {}
    for images_path in cr._batch_images(cc.CORPUS_DIR):
        batch_id = os.path.basename(os.path.dirname(images_path))
        rows = cc.read_jsonl(images_path)
        seedsets[batch_id] = set(r["provenance"].get("seed_index") for r in rows
                                 if r.get("label", {}).get("score") is not None)
    bids = list(seedsets)
    if len(bids) == 2:
        inter = seedsets[bids[0]] & seedsets[bids[1]]
        print(f"  seed_index overlap between batches: {len(inter)} values "
              f"(e.g. {sorted(x for x in inter if x is not None)[:8]})  "
              f"-> seed alone is NOT a safe group key across batches")


if __name__ == "__main__":
    main()
