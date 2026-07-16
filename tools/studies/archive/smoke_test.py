"""Part 4 — smoke test: prove the cross-batch union before any training.

Counts labelable (crop, score) pairs across every batch via corpus_reader
(version-blind), and asserts the invariants we expect right now:

  - loose0_v3 (flat_generate) batch is FULLY labeled,
  - the rev4 batch exists with 0 labels until Matt scores it,
  - the union yields > the single-batch count (the two batches actually compose),
  - every yielded crop_path resolves to a file on disk,
  - the label-score distribution is sane (scores in {1,2,3}).

Run:  uv run python tools/corpus/smoke_test.py
"""
from __future__ import annotations

import os
from collections import Counter

import corpus_common as cc
import corpus_reader as cr

FLAT_BATCH = "2026-06-23_flat_generate_loose0_v3"
REV4_BATCH = "2026-06-24_guided_descend_rev4"


def main() -> None:
    census = cr.count_pairs()
    print("=== per-batch census ===")
    for batch_id, c in census.items():
        print(f"  {batch_id}: {c['labeled']}/{c['units']} labeled")

    # version-blind union over whatever is labeled right now.
    crops = list(cr.iter_labeled())
    dist = Counter(c.score for c in crops)
    by_batch = Counter(c.batch_id for c in crops)
    total_units = sum(c["units"] for c in census.values())
    max_units = max((c["units"] for c in census.values()), default=0)
    print("=== version-blind union ===")
    print(f"  total UNITS across batches:  {total_units}  (labelable surface)")
    print(f"  total LABELED pairs now:     {len(crops)}")
    print(f"  score distribution:          {dict(sorted(dist.items()))}")
    print(f"  labeled by batch:            {dict(by_batch)}")

    missing = [c.crop_path for c in crops if not os.path.exists(c.crop_path)]

    failures = []

    # 1. both expected batches are discovered by the version-blind glob.
    if FLAT_BATCH not in census:
        failures.append(f"flat batch {FLAT_BATCH} not found")
    elif census[FLAT_BATCH]["labeled"] != census[FLAT_BATCH]["units"]:
        failures.append(f"flat batch not fully labeled: {census[FLAT_BATCH]}")

    if REV4_BATCH not in census:
        failures.append(f"rev4 batch {REV4_BATCH} not found")
    elif census[REV4_BATCH]["labeled"] != 0:
        # expected to be 0 PRE-labeling; once Matt scores it this is just info.
        print(f"  NOTE: rev4 batch has {census[REV4_BATCH]['labeled']} labels "
              f"(was 0 until Matt scored it).")

    # 2. the batches actually COMPOSE: the cross-batch glob sees more labelable
    #    units than any single batch holds (this is the union mechanism, and it
    #    holds before any rev4 label exists).
    if len(census) < 2:
        failures.append(f"expected >=2 batches in the union, found {len(census)}")
    if total_units <= max_units:
        failures.append("total units do not exceed the biggest batch — batches not composing")

    # 3. the labeled pairs the trainer would see are well-formed.
    if not set(dist).issubset({1, 2, 3}):
        failures.append(f"scores outside {{1,2,3}}: {set(dist)}")
    if missing:
        failures.append(f"{len(missing)} labeled crops missing on disk, e.g. {missing[:3]}")

    print("=== smoke test ===")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        raise SystemExit(1)
    print(f"  PASS: union spans {len(census)} batches / {total_units} labelable units; "
          f"{len(crops)} labeled pairs present now (all crop files on disk). "
          f"Adding rev4 labels grows the union with zero code change.")


if __name__ == "__main__":
    main()
