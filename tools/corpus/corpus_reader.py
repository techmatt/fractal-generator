"""Part 4 — version-blind cross-batch reader for the label corpus.

The classifier trainer's view of the store: glob every batch's images.jsonl and
yield `(crop_path, score)` for non-null labels, **blind to generator_version**.
Provenance is NEVER read here — that is exactly what makes v4's metaparameters
not matching v1's cost nothing on the training side.

Label resolution goes through the shared `label_store` primitive (merged
`label.score` ELSE the registered `labels/*.json` sidecar joined by image_id), so
the sidecar-only batches (Julia/mining/scale) are seen here just as the query
sampler sees them — the two consumers cannot drift.

API:
  iter_labeled(corpus_dir=None) -> yields LabeledCrop(crop_path, score, image_id, batch_id, render)
  count_pairs(corpus_dir=None)  -> {batch_id: {"units": n, "labeled": m}}

CLI (a quick census):
  uv run python tools/corpus/corpus_reader.py
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import corpus_common as cc
import label_store as ls


@dataclass
class LabeledCrop:
    crop_path: str   # absolute path to crops/<image_id>.jpg
    score: int       # 1 | 2 | 3
    image_id: str
    batch_id: str
    render: dict     # the version-invariant render block (also available to the trainer)


def _batch_images(corpus_dir: str):
    pattern = os.path.join(corpus_dir, "batches", "*", "images.jsonl")
    return sorted(glob.glob(pattern))


def iter_labeled(corpus_dir: str | None = None):
    """Yield one LabeledCrop per non-null label across ALL batches, version-blind.

    Labels resolve through `label_store` (merged score ELSE registered sidecar join),
    so sidecar-only batches are recovered. Guards at the end of a full pass that every
    registered sidecar batch actually joined rows (a lazy consumer that stops early
    simply doesn't trigger the guard)."""
    corpus_dir = corpus_dir or cc.CORPUS_DIR
    joined = {}
    for images_path in _batch_images(corpus_dir):
        batch_dir = os.path.dirname(images_path)
        batch_id = os.path.basename(batch_dir)
        crops_dir = os.path.join(batch_dir, "crops")
        sidecar = ls.sidecar_for(batch_id)
        joined.setdefault(batch_id, 0)
        with open(images_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                score = ls.resolve_score(row, sidecar)
                if score is None:
                    continue
                joined[batch_id] += 1
                image_id = row["image_id"]
                yield LabeledCrop(
                    crop_path=os.path.join(crops_dir, image_id + ".jpg"),
                    score=int(score),
                    image_id=image_id,
                    batch_id=batch_id,
                    render=row.get("render", {}),
                )
    ls.assert_sidecars_joined(joined)


def count_pairs(corpus_dir: str | None = None) -> dict:
    """Per-batch {units, labeled} census (units = all rows, labeled = non-null score)."""
    corpus_dir = corpus_dir or cc.CORPUS_DIR
    out = {}
    joined = {}
    for images_path in _batch_images(corpus_dir):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        units = labeled = 0
        with open(images_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                units += 1
                if ls.resolve_score(json.loads(line), sidecar) is not None:
                    labeled += 1
        out[batch_id] = {"units": units, "labeled": labeled}
        joined[batch_id] = labeled
    ls.assert_sidecars_joined(joined)
    return out


if __name__ == "__main__":
    census = count_pairs()
    total_labeled = sum(b["labeled"] for b in census.values())
    total_units = sum(b["units"] for b in census.values())
    print("=== label_corpus census (version-blind) ===")
    for batch_id, c in census.items():
        print(f"  {batch_id}: {c['labeled']}/{c['units']} labeled")
    print(f"  TOTAL: {total_labeled}/{total_units} labeled pairs across {len(census)} batches")
