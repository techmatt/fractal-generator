"""Part 4 — merge a harness `scores.json` export into a batch's images.jsonl.

The ONE allowed mutation in the store is a label's score going `null -> value`.
This merger enforces that: it fills `label.score` for rows whose score is
currently null, and **warns and refuses** (never silently clobbers) when a
scores.json entry would change an already-non-null label to a different value.
Re-applying the same score is a no-op.

Run:
  uv run python tools/corpus/merge_scores.py \
      --batch 2026-06-24_guided_descend_rev4 \
      [--scores <path>]  [--labeler matt] [--labeled-at 2026-06-25] [--apply]

Without --apply it's a dry run (reports what would change, writes nothing).
"""
from __future__ import annotations

import argparse
import json
import os

import corpus_common as cc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="batch_id under data/label_corpus/batches/")
    ap.add_argument("--scores", default=None, help="scores.json (default: <batch>/scores.json)")
    ap.add_argument("--labeler", default="matt")
    ap.add_argument("--labeled-at", default=None)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    a = ap.parse_args()

    bdir = cc.batch_dir(a.batch)
    images_path = os.path.join(bdir, "images.jsonl")
    scores_path = a.scores or os.path.join(bdir, "scores.json")

    rows = cc.read_jsonl(images_path)
    scores = json.load(open(scores_path, encoding="utf-8"))
    scores = {k: (int(v) if v is not None else None) for k, v in scores.items()}

    filled, reaffirmed, conflicts, unknown = 0, 0, [], []
    by_id = {r["image_id"]: r for r in rows}

    for image_id, new_score in scores.items():
        if new_score is None:
            continue
        row = by_id.get(image_id)
        if row is None:
            unknown.append(image_id)
            continue
        cur = row["label"]["score"]
        if cur is None:
            row["label"]["score"] = new_score
            row["label"]["labeler"] = a.labeler
            row["label"]["labeled_at"] = a.labeled_at
            filled += 1
        elif cur == new_score:
            reaffirmed += 1
        else:
            conflicts.append((image_id, cur, new_score))

    print(f"batch {a.batch}: {len(rows)} rows, {len(scores)} scores in export")
    print(f"  null -> value (fill): {filled}")
    print(f"  already == score (no-op): {reaffirmed}")
    if unknown:
        print(f"  WARNING: {len(unknown)} scores reference unknown image_id (skipped), e.g. {unknown[:3]}")
    if conflicts:
        print(f"  REFUSED: {len(conflicts)} would CHANGE a non-null label - NOT applied:")
        for image_id, cur, new in conflicts[:20]:
            print(f"    {image_id}: existing {cur} != export {new}")
        if len(conflicts) > 20:
            print(f"    ... and {len(conflicts) - 20} more")

    if not a.apply:
        print("  DRY RUN - pass --apply to write (conflicts are never written either way)")
        return

    cc.write_jsonl(rows, images_path)
    labeled = sum(1 for r in rows if r["label"]["score"] is not None)
    print(f"  WROTE {images_path}: {labeled}/{len(rows)} now labeled "
          f"({len(conflicts)} conflicts left untouched)")


if __name__ == "__main__":
    main()
