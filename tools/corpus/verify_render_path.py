#!/usr/bin/env python
"""Guard B — the location-corpus render-path reproducibility check.

The enforceable form of the "crops are rebuildable" contract: sample K crops from a
batch, **rebuild each from its stored render block via the canonical
`render-one --palette` path** (`corpus_common.render_corpus_crop`), and assert the
rebuild matches the stored crop within JPEG-quantization noise.

Why this is the durable backstop: it fires regardless of HOW a bad batch was
produced. The canonical native colorer and the off-recipe `dump-field` +
`colormap.render_candidate` tail produce **different images** — measured mean |Δ|
16.2 (max 209), ~75% of pixels. A legit rebuild through the same deterministic
recipe is ~0–3 mean |Δ| (JPEG re-quantization only). The default threshold 5.0
separates the two cleanly: a batch coloured off-recipe rebuilds at ~16 and trips it.

Also checks the self-identifying recipe stamp (`batch.json["render_recipe"]`): if
present it MUST be the canonical path, and its `palette_source`/`jpg_quality` drive
the rebuild. If absent, `--palette-source` (default score3_colormaps.json) is used
and a warning is emitted.

Wired two ways:
  * `check_batch(batch_dir, k=...)` — called at the end of every location-batch
    emission (gather_select / recolor) to auto-verify each new/modified batch.
  * standalone:  uv run python tools/corpus/verify_render_path.py <batch_dir> [--k K]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # tools/corpus/
from corpus_common import (CANONICAL_CROP_RECIPE, DEFAULT_CROP_JPGQ,       # noqa: E402
                           render_corpus_crop, read_jsonl)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PALETTE_SOURCE = ROOT / "data" / "palettes" / "score3_colormaps.json"

# legit rebuild ~0–3, off-recipe ~16.2 → 5.0 cleanly separates (prompt Guard B).
DEFAULT_THRESHOLD = 5.0
DEFAULT_K = 6
DEFAULT_SEED = 12345

BELOW_NORMAL = getattr(__import__("subprocess"), "BELOW_NORMAL_PRIORITY_CLASS", 0)


def _mean_abs_delta(a_path, b_path) -> float:
    """Mean |Δ| per channel over two same-size RGB JPGs (0..255 scale)."""
    with Image.open(a_path) as ia, Image.open(b_path) as ib:
        a = np.asarray(ia.convert("RGB"), dtype=np.float64)
        b = np.asarray(ib.convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape} ({a_path} vs {b_path})")
    return float(np.abs(a - b).mean())


def _resolve_recipe(batch_dir: Path, palette_source_override):
    """Read `batch.json["render_recipe"]`. Returns (palette_source, jpg_quality).
    A present stamp MUST be canonical (else raise). Absent → override/default + warn."""
    bj_path = batch_dir / "batch.json"
    stamp = None
    if bj_path.exists():
        stamp = json.loads(bj_path.read_text(encoding="utf-8")).get("render_recipe")
    if stamp is not None:
        if stamp.get("path") != CANONICAL_CROP_RECIPE:
            raise AssertionError(
                f"batch.json render_recipe.path is {stamp.get('path')!r}, not the canonical "
                f"{CANONICAL_CROP_RECIPE!r} -- this batch was NOT built through render_corpus_crop")
        ps = palette_source_override or (ROOT / stamp["palette_source"])
        jq = int(stamp.get("jpg_quality", DEFAULT_CROP_JPGQ))
        return Path(ps), jq
    sys.stderr.write("[verify] WARNING: batch.json has no render_recipe stamp -- "
                     f"using palette_source={palette_source_override or DEFAULT_PALETTE_SOURCE}\n")
    return Path(palette_source_override or DEFAULT_PALETTE_SOURCE), DEFAULT_CROP_JPGQ


def check_batch(batch_dir, k: int = DEFAULT_K, seed: int = DEFAULT_SEED,
                threshold: float = DEFAULT_THRESHOLD, palette_source=None,
                verbose: bool = True) -> dict:
    """Rebuild K sampled crops from their render blocks and compare to the stored crops.

    Returns {"ok", "k", "threshold", "worst", "mean", "deltas":[(image_id, delta)]}.
    Raises AssertionError if any sampled crop's mean |Δ| >= threshold (off-recipe /
    non-reproducible), or if the recipe stamp is present but non-canonical."""
    batch_dir = Path(batch_dir)
    rows = read_jsonl(str(batch_dir / "images.jsonl"))
    crops = batch_dir / "crops"
    have = [r for r in rows if (crops / f"{r['image_id']}.jpg").exists()]
    if not have:
        raise AssertionError(f"no crops found under {crops}")

    palette_source, jpg_quality = _resolve_recipe(batch_dir, palette_source)
    rng = random.Random(seed)
    sample = rng.sample(have, min(k, len(have)))

    deltas = []
    with tempfile.TemporaryDirectory() as td:
        for r in sample:
            iid = r["image_id"]
            stored = crops / f"{iid}.jpg"
            rebuilt = Path(td) / f"{iid}.jpg"
            render_corpus_crop(r["render"], rebuilt, palette_source=palette_source,
                               jpg_quality=jpg_quality, cwd=str(ROOT),
                               creationflags=BELOW_NORMAL)
            d = _mean_abs_delta(stored, rebuilt)
            deltas.append((iid, d))
            if verbose:
                flag = "OK " if d < threshold else "FAIL"
                print(f"  [{flag}] {iid:40s} mean|d|={d:6.3f}")

    worst = max(d for _, d in deltas)
    mean = sum(d for _, d in deltas) / len(deltas)
    ok = worst < threshold
    res = {"ok": ok, "k": len(deltas), "threshold": threshold,
           "worst": worst, "mean": mean, "deltas": deltas,
           "palette_source": str(palette_source)}
    if verbose:
        print(f"[verify] {batch_dir.name}: k={len(deltas)} worst mean|d|={worst:.3f} "
              f"(mean {mean:.3f}) threshold {threshold}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(
            f"reproducibility check FAILED for {batch_dir.name}: worst mean|d|={worst:.3f} "
            f">= {threshold} — crops are NOT rebuildable from their render block via the "
            f"canonical {CANONICAL_CROP_RECIPE} path (off-recipe or nondeterministic render).")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_dir", help="a data/label_corpus/batches/<batch_id>/ directory")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="crops to sample (render-one ~2s each)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--palette-source", default=None,
                    help="override the --colormaps library (else the stamp / score3 default)")
    a = ap.parse_args()
    try:
        check_batch(a.batch_dir, k=a.k, seed=a.seed, threshold=a.threshold,
                    palette_source=a.palette_source)
    except AssertionError as e:
        sys.stderr.write(f"\nFAIL: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
