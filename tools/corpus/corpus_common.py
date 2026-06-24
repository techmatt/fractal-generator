"""Shared helpers for the permanent label corpus (data/label_corpus/).

See data/label_corpus/CORPUS_SCHEMA.md for the contract. This module owns the
*shape* of an images.jsonl row so every batch writer agrees on the field set:
`render` is version-invariant (identical keys across all batches), `provenance`
is version-tagged (free to be null/absent), `label` is the verdict.
"""
from __future__ import annotations

import json
import os

# repo root = two levels up from tools/corpus/
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.path.join(ROOT, "data", "label_corpus")
BATCHES_DIR = os.path.join(CORPUS_DIR, "batches")

# The version-invariant render field set — the ONLY thing the classifier sees
# (alongside the crop). Every batch's every row carries exactly these keys.
RENDER_KEYS = (
    "cx", "cy", "fw", "maxiter", "palette", "composition",
    "width", "height", "ss", "filter", "interior_mode",
)

# Provenance keys we currently model. A given generator version fills what it
# has and leaves the rest null — provenance is allowed to differ across batches.
PROVENANCE_KEYS = (
    "generator_version", "batch_id", "root_src", "branch", "depth",
    "target_depth", "walk_id", "placement", "focus_score",
    "draw_index", "seed_index", "black_fraction", "interior_frac",
    "occupancy", "void_guard",
)


def hp_str(x) -> str:
    """Render a coordinate as a high-precision decimal string.

    The store keeps cx/cy/fw as strings (an f64 center is meaningless at deep
    zoom). For shallow f64-sourced batches the f64's shortest round-tripping
    decimal IS its full precision, so repr() is faithful; a future deep batch
    would pass an already-arbitrary-precision string straight through.
    """
    if isinstance(x, str):
        return x
    return repr(float(x))


def image_id_from_output(output_path: str) -> str:
    """The crop's basename without extension — the batch-unique, fs-safe stem.

    present writes `{seed_index}_{composition}_{palette}.{ext}`, unique per
    (seed_index, composition, palette) within a run; the store reuses it as
    `image_id`, and the crop lives at `crops/<image_id>.jpg`.
    """
    base = os.path.basename(str(output_path).replace("\\", "/"))
    stem, _ext = os.path.splitext(base)
    return stem


def render_block(*, cx, cy, fw, maxiter, palette, composition,
                 width, height, ss, filter, interior_mode) -> dict:
    """Build a version-invariant render block (coordinates → hi-prec strings)."""
    return {
        "cx": hp_str(cx),
        "cy": hp_str(cy),
        "fw": hp_str(fw),
        "maxiter": int(maxiter),
        "palette": str(palette),
        "composition": str(composition),
        "width": int(width),
        "height": int(height),
        "ss": int(ss),
        "filter": str(filter),
        "interior_mode": str(interior_mode),
    }


def provenance_block(generator_version: str, batch_id: str, **fields) -> dict:
    """Build a provenance block: every modeled key present, unspecified → null.

    Pass only the fields this generator version actually produced; the rest are
    explicitly null (never fabricated).
    """
    prov = {k: None for k in PROVENANCE_KEYS}
    prov["generator_version"] = generator_version
    prov["batch_id"] = batch_id
    for k, v in fields.items():
        if k not in PROVENANCE_KEYS:
            raise KeyError(f"unknown provenance key {k!r}; add it to PROVENANCE_KEYS first")
        prov[k] = v
    return prov


def label_block(score=None, labeler=None, labeled_at=None) -> dict:
    return {"score": score, "labeler": labeler, "labeled_at": labeled_at}


def make_row(image_id: str, render: dict, provenance: dict, label: dict) -> dict:
    missing = [k for k in RENDER_KEYS if k not in render]
    if missing:
        raise KeyError(f"render block missing keys {missing} for {image_id}")
    return {
        "image_id": image_id,
        "render": render,
        "provenance": provenance,
        "label": label,
    }


def write_jsonl(rows, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batch_dir(batch_id: str) -> str:
    return os.path.join(BATCHES_DIR, batch_id)
