#!/usr/bin/env python
"""Recolor the 2026-07-05_gather_v6 label batch from the 76 curated q3 palettes.

The v6 gather batch was rendered from the 777-palette pool, which admitted too-dark,
unlabelable crops. This recolors every crop from the **76 curated q3 palettes**
(`data/palettes/score3_colormaps.json`, bright by construction) at the row's own
coords/viewport — so ONLY the palette changes; framing, location, and the coloring
RECIPE are preserved. `image_id` is unchanged; `label` (all null) is untouched; only
`render.palette` is rewritten to the 76-palette actually rendered.

Render path = `render-one --palette --colormaps score3_colormaps.json` — the EXACT
path `gather_select` used to build the original batch (the Rust wallpaper default
colorer, one native pass, all 9 families). This is deliberate, not `render-one
--dump-field` + `colormap.render_candidate`: the dump-field tail reproduces only a
plain smooth pct-stretch and diverges from the wallpaper colorer by ~16 mean levels
(75% of pixels), so it would (a) render a DIFFERENT look than both the original batch
and every sibling gather batch, and (b) break the corpus contract that a crop is a
pure function of its render block THROUGH render-one. Named palette + default coloring
params means `render.palette` alone is the full recipe, so the direct path keeps the
crop render-one-reproducible. (`enrich --mode render` is unusable here — mandelbrot-
only / named-palette — which is why gather_select uses `render-one` in the first place.)

Julia rows reconstruct viewport + fixed c from the render block's `fractal_type` +
c_re/c_im via `location.from_render_block` / `render_one_flags` (same mechanism the
batch build used).

  uv run python tools/corpus/recolor_gather_v6.py --limit 2   # smoke
  uv run python tools/corpus/recolor_gather_v6.py             # full 640
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))

from corpus_common import (read_jsonl, write_jsonl, render_label_crop,     # noqa: E402
                           render_recipe_stamp)
from verify_render_path import check_batch                                # noqa: E402

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
BATCH_DIR = ROOT / "data" / "label_corpus" / "batches" / "2026-07-05_gather_v6"
IMAGES = BATCH_DIR / "images.jsonl"
CROPS = BATCH_DIR / "crops"
SCORE3 = ROOT / "data" / "palettes" / "score3_colormaps.json"

# --- locked label-crop spec (matches the batch's render_defaults exactly) ---
W, H, SS, MAXITER, JPGQ, FILTER = 1280, 720, 4, 8000, 90, "lanczos3"

SEED = 20260705
WORKERS = 3          # same discipline as the batch build (<=3 concurrent renders)
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def assign_palettes(rows, names):
    """Seeded-random 76-palette per row, deterministic in image_id order (independent
    of render order / parallelism)."""
    import random
    rng = random.Random(SEED)
    return {r["image_id"]: rng.choice(names) for r in sorted(rows, key=lambda r: r["image_id"])}


def recolor_one(row, palette):
    """Recolor ONE crop through the canonical `render-one --palette` path — ONLY the
    palette changes (block copied, `palette` overridden), so framing/location/recipe
    are preserved and the crop stays render-block-reproducible (Guard B)."""
    iid = row["image_id"]
    out = CROPS / f"{iid}.jpg"
    block = {**row["render"], "palette": palette}     # same block, new palette only
    try:
        render_label_crop(block, out, palette_source=SCORE3, bin_path=EXE,
                          jpg_quality=JPGQ, cwd=str(ROOT), creationflags=BELOW_NORMAL)
    except RuntimeError as e:
        return (iid, False, str(e)[-300:])
    with Image.open(out) as im:
        ok = im.size == (W, H)
    return (iid, ok, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke); 0 = all 640")
    a = ap.parse_args()

    names = [c["name"] for c in json.loads(SCORE3.read_text(encoding="utf-8"))]
    print(f"loaded {len(names)} score3 palettes", flush=True)

    rows = read_jsonl(str(IMAGES))
    pal = assign_palettes(rows, names)
    todo = rows[:a.limit] if a.limit else rows
    print(f"recoloring {len(todo)}/{len(rows)} crops with {WORKERS} workers "
          f"({W}x{H} ss{SS} {FILTER} q{JPGQ}) via render-one --palette ...", flush=True)

    t0 = time.time()
    results = {}
    done = ok = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(recolor_one, r, pal[r["image_id"]]): r["image_id"] for r in todo}
        for fut in cf.as_completed(futs):
            iid, good, err = fut.result()
            results[iid] = good
            done += 1
            ok += int(good)
            if err:
                sys.stderr.write(f"[recolor {iid}] FAILED: {err}\n")
            if done % 25 == 0 or done == len(todo):
                dt = time.time() - t0
                print(f"  {done}/{len(todo)}  ({ok} ok)  {dt:.0f}s  "
                      f"[{dt/max(done,1):.1f}s/crop]", flush=True)

    # --- rewrite render.palette for every successfully recolored row ---
    n_updated = 0
    for r in rows:
        iid = r["image_id"]
        if results.get(iid):
            r["render"]["palette"] = pal[iid]
            n_updated += 1
    if not a.limit:
        write_jsonl(rows, str(IMAGES))
        print(f"rewrote {IMAGES}  ({n_updated} render.palette updated)", flush=True)
        # stamp/refresh the canonical render-recipe provenance on batch.json, then
        # Guard B: assert the recoloured crops are rebuildable from their render blocks.
        bj_path = BATCH_DIR / "batch.json"
        bj = json.loads(bj_path.read_text(encoding="utf-8"))
        bj["render_recipe"] = render_recipe_stamp(SCORE3, jpg_quality=JPGQ)
        bj_path.write_text(json.dumps(bj, indent=2), encoding="utf-8")
        print("\n===== Guard B: render-path reproducibility (K-sample) =====")
        check_batch(BATCH_DIR, k=6)
    else:
        print(f"[smoke: --limit {a.limit}] images.jsonl NOT rewritten "
              f"(would update {n_updated} palettes)", flush=True)

    n_distinct = len({pal[iid] for iid, g in results.items() if g})
    print(f"\ndone: {ok}/{len(todo)} recolored  |  {len(todo)-ok} failed  |  "
          f"{n_distinct} distinct palettes  |  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
