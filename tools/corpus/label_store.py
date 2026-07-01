"""Shared label resolution — the single source of truth for "what score a row has".

A batch's hand labels live in ONE of two places:
  (a) merged into its images.jsonl `label.score` (via tools/corpus/merge_scores.py)
      — the loose0 / rev4 / rev4occfix batches; or
  (b) ONLY in a `labels/*.json` sidecar keyed by image_id, because the merge into
      images.jsonl was never run — the Julia (`julia_ladder_j0`), `mining`, and
      `scale` batches.
A loader that reads `label.score` alone silently drops the (b) batches: for Julia
that wiped out the entire family (0 Julia locations), and it dropped the mining/scale
Mandelbrot labels too.

Every consumer that turns a corpus row into a label MUST route through this module —
`corpus_reader.iter_labeled` (the version-blind trainer view) and
`query_sampler.LocationPool.from_corpus` (the q2+q3 location universe) both do — so the
resolution logic + the `SIDECAR_LABELS` registry live in exactly ONE place and the two
can never drift. NEW unmerged batches MUST be registered in `SIDECAR_LABELS` (or have
their labels merged into images.jsonl); `assert_sidecars_joined` makes a broken join
loud at load.

REFERENCE for the complete label set: the v5 unified classifier's training-data
assembly, tools/v5/build_manifest.py. It recovers the J0 Julia labels from
labels/location_labels_julia_ladder_j0.json JOINED to the batch's images.jsonl by
image_id — exactly the join mirrored here. See data/label_corpus/CORPUS_SCHEMA.md.
"""
from __future__ import annotations

import json
import os

# repo root = two levels up from tools/corpus/
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LABELS_DIR = os.path.join(ROOT, "labels")

# The (b)-case batches: batch_id -> its labels/*.json sidecar file. The registry is
# the single source of truth for "which batches carry labels only in a sidecar".
SIDECAR_LABELS = {
    "julia_ladder_j0": "location_labels_julia_ladder_j0.json",
    "2026-06-25_mining_v3guided_v1": "mining_v3guided_v1.json",
    "2026-06-25_scale_2x2_labelset": "scale_2x2_labelset.json",
    "2026-06-25_scale_controlled_2x2": "scale_2x2_labelset.json",
}


def load_sidecar(filename):
    """Load a labels/*.json sidecar as {image_id: int score}, dropping nulls.

    Tolerates both a bare {image_id: score} map and a {"labels": {...}} wrapper."""
    d = json.loads((open(os.path.join(LABELS_DIR, filename), encoding="utf-8")).read())
    body = d["labels"] if isinstance(d.get("labels"), dict) else d
    return {k: int(v) for k, v in body.items() if v is not None}


def sidecar_for(batch_id):
    """The {image_id: score} sidecar map for a batch, or None if it isn't registered."""
    fn = SIDECAR_LABELS.get(batch_id)
    return load_sidecar(fn) if fn is not None else None


def resolve_score(row, sidecar):
    """A row's label: merged `label.score` ELSE the sidecar join by image_id.

    `sidecar` is the map from `sidecar_for(batch_id)` (or None for a merged batch).
    Returns None if the row is unlabeled in both places. This is the ONE resolution
    rule; both consumers call it so they cannot disagree on a row."""
    sc = (row.get("label") or {}).get("score")
    if sc is None and sidecar is not None:
        sc = sidecar.get(row["image_id"])
    return sc


def assert_sidecars_joined(joined):
    """Raise if a REGISTERED sidecar batch present in `joined` contributed 0 rows.

    `joined`: {batch_id: rows_resolved_via_this_batch} accumulated over a full pass. A
    registered sidecar that resolves nothing is a corpus/registry error (image_id keys
    diverged / wrong file) — not an empty batch. Batches absent from `joined` (not on
    disk) are skipped."""
    for bid, fn in SIDECAR_LABELS.items():
        if bid in joined and joined[bid] == 0:
            raise RuntimeError(
                f"batch {bid!r} has a registered label sidecar ({fn}) but joined 0 "
                f"rows — image_id keys likely diverged. Fix SIDECAR_LABELS in "
                f"tools/corpus/label_store.py.")
