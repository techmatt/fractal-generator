"""v2 cross-batch row loader — the eval-split helper's view of the label corpus.

This is the ONLY place provenance is read. The split between what the *model*
sees and what the *grouping* sees is the corpus contract (v2 prompt):

  * MODEL INPUT  = the render-crop JPG + its 1/2/3 label.  Nothing else.
  * GROUPING ONLY (this module, never fed to the net) = batch_id, seed_index,
    walk_id, black_fraction.  Provenance is allowed to differ across generator
    versions, so letting it into the model would break cross-version training.

`CorpusRow` is drop-in for `data.CropDataset` (it reads only `.jpg` and
`.label`); the extra fields are consumed exclusively by `eval_v2`/`train_v2`
when forming folds, the holdout, and the per-batch / location aggregations.

The correlation unit (the CV group key) is batch-appropriate because the seed
namespace COLLIDES across batches (seed 0 means different things in loose0 vs
rev4 — 187 shared seed ids), so the group key is always batch-qualified:
  * flat loose0  -> seed_index   (IID draws, v1's grouping)
  * descent rev4 -> walk_id      (frames d1..d10 of one walk are correlated;
                                  splitting them train/val would leak)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "corpus"))
import corpus_common as cc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLACK_THRESH = 0.30  # held fixed from v1 (mirrors present.rs): accept iff bf < 0.30


@dataclass
class CorpusRow:
    # --- model-facing (the only fields data.CropDataset touches) ---
    jpg: Path
    label: int                 # raw 1/2/3
    # --- grouping / eval only (NEVER a model input) ---
    image_id: str
    batch_id: str
    seed_index: int
    walk_id: int | None
    black_fraction: float
    palette: str
    composition: str
    # derived keys
    group_unit: str = field(default="")   # batch-qualified CV/holdout correlation unit
    loc_unit: str = field(default="")     # batch-qualified location (best-over-palettes) key

    def __post_init__(self):
        # location = a single rendered frame (one (cx,cy,fw)); its crops are the
        # palette/composition variants.  Always batch-qualified.
        self.loc_unit = f"{self.batch_id}|s{self.seed_index}"
        # correlation unit: walk if the batch carries one, else the seed.
        if self.walk_id is not None:
            self.group_unit = f"{self.batch_id}|w{self.walk_id}"
        else:
            self.group_unit = f"{self.batch_id}|s{self.seed_index}"

    # `.seed` alias kept so any v1 helper expecting a location attr still works.
    @property
    def seed(self) -> str:
        return self.loc_unit


def _batch_images(corpus_dir: str):
    import glob
    return sorted(glob.glob(os.path.join(corpus_dir, "batches", "*", "images.jsonl")))


def load_corpus_rows(corpus_dir: str | None = None,
                     apply_black_filter: bool = True) -> list[CorpusRow]:
    """Union of all batches' labeled rows, black-gated like v1.

    Reads provenance for grouping ONLY (see module docstring). Verifies every
    crop file is on disk.
    """
    corpus_dir = corpus_dir or cc.CORPUS_DIR
    rows: list[CorpusRow] = []
    for images_path in _batch_images(corpus_dir):
        batch_dir = os.path.dirname(images_path)
        batch_id = os.path.basename(batch_dir)
        crops_dir = os.path.join(batch_dir, "crops")
        for r in cc.read_jsonl(images_path):
            score = r.get("label", {}).get("score")
            if score is None:
                continue
            prov = r.get("provenance", {})
            bf = prov.get("black_fraction")
            bf = float(bf) if bf is not None else 0.0
            if apply_black_filter and not (bf < BLACK_THRESH):  # mirror present.rs / v1
                continue
            image_id = r["image_id"]
            jpg = Path(crops_dir) / f"{image_id}.jpg"
            if not jpg.exists():
                raise FileNotFoundError(f"corpus crop missing: {jpg}")
            wid = prov.get("walk_id")
            rows.append(CorpusRow(
                jpg=jpg, label=int(score), image_id=image_id, batch_id=batch_id,
                seed_index=int(prov.get("seed_index")),
                walk_id=(int(wid) if wid is not None else None),
                black_fraction=bf,
                palette=r["render"].get("palette", ""),
                composition=r["render"].get("composition", ""),
            ))
    return rows


def census(rows: list[CorpusRow]) -> dict:
    from collections import Counter, defaultdict
    out = {}
    by_batch = defaultdict(list)
    for r in rows:
        by_batch[r.batch_id].append(r)
    for bid, items in by_batch.items():
        c = Counter(r.label for r in items)
        groups = set(r.group_unit for r in items)
        locs = set(r.loc_unit for r in items)
        out[bid] = {"n": len(items), "hist": {k: c.get(k, 0) for k in (1, 2, 3)},
                    "n_groups": len(groups), "n_locations": len(locs)}
    cu = Counter(r.label for r in rows)
    out["UNION"] = {"n": len(rows), "hist": {k: cu.get(k, 0) for k in (1, 2, 3)}}
    return out
