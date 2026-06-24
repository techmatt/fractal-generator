"""Part 2 — import the loose0_v3 labeled corpus as label_corpus batch 0.

Wraps the existing flat-generate labeled crops as
`batches/2026-06-23_flat_generate_loose0_v3/`:

  - `render` populated from the old present manifest (version-invariant),
  - `label.score` from `labels/location_labels.json` (key draw_index|comp|palette),
  - `provenance.generator_version = "flat_generate"` carrying ONLY the flat-era
    fields that actually exist (draw_index, seed_index, the per-frame measures,
    void_guard); rev4-only fields (root_src/branch/depth/…) are null — not invented.

Crops are COPIED into the batch's crops/<image_id>.jpg so the batch is
self-contained (crops are gitignored; rebuildable from `render`).

Run:  uv run python tools/corpus/import_loose0_v3.py
"""
from __future__ import annotations

import json
import os
import shutil

import corpus_common as cc

BATCH_ID = "2026-06-23_flat_generate_loose0_v3"
GENERATOR_VERSION = "flat_generate"

SRC_DIR = os.path.join(cc.ROOT, "data", "label_crops", "loose0_v3")
MANIFEST = os.path.join(SRC_DIR, "manifest.json")
LABELS = os.path.join(cc.ROOT, "labels", "location_labels.json")
GEN_MANIFEST = os.path.join(cc.ROOT, "data", "generated", "loose0", "manifest.json")

# render constants for this batch. Crop dims 1280×720 + maxiter 2000 are from the
# present manifest. SS=2 is present's actual default (the manifest's hardcoded
# "grid ss4" render string mislabels it — generate's keeper_render was ss2 and
# loose0_v3 was rendered with the default --ss). Lanczos-3 downsample is the locked
# wallpaper-quality reconstruction (FULL_FILTER in present.rs).
WIDTH, HEIGHT, SS = 1280, 720, 2
FILTER, INTERIOR_MODE, MAXITER = "lanczos3", "black", 2000


def label_key(c: dict) -> str:
    return f"{c['draw_index']}|{c['composition']}|{c['palette']}"


def main() -> None:
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    labels = json.load(open(LABELS, encoding="utf-8"))
    crops = manifest["crops"]

    out_dir = cc.batch_dir(BATCH_ID)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    seen = set()
    n_labeled = 0
    n_copied = 0
    score_hist = {1: 0, 2: 0, 3: 0, None: 0}

    for c in crops:
        image_id = cc.image_id_from_output(c["output"])
        if image_id in seen:
            continue  # the with-replacement palette dup rows collapse to one unit
        seen.add(image_id)

        render = cc.render_block(
            cx=c["cx"], cy=c["cy"], fw=c["fw"],
            maxiter=MAXITER, palette=c["palette"], composition=c["composition"],
            width=WIDTH, height=HEIGHT, ss=SS, filter=FILTER, interior_mode=INTERIOR_MODE,
        )
        # flat-era provenance: only fields that genuinely exist for flat generate.
        prov = cc.provenance_block(
            GENERATOR_VERSION, BATCH_ID,
            draw_index=c["draw_index"],
            seed_index=c["seed_index"],
            interior_frac=c.get("interior_frac"),
            black_fraction=c.get("black_fraction"),
            occupancy=c.get("occupancy"),
            void_guard=c.get("void_guard"),
        )
        score = labels.get(label_key(c))
        if score is not None:
            score = int(score)
            n_labeled += 1
        score_hist[score] = score_hist.get(score, 0) + 1
        label = cc.label_block(
            score=score,
            labeler="matt" if score is not None else None,
            labeled_at=None,
        )
        rows.append(cc.make_row(image_id, render, prov, label))

        # copy the crop in (self-contained batch).
        src_jpg = os.path.join(SRC_DIR, image_id + ".jpg")
        dst_jpg = os.path.join(crops_dir, image_id + ".jpg")
        if os.path.exists(src_jpg) and not os.path.exists(dst_jpg):
            shutil.copy2(src_jpg, dst_jpg)
            n_copied += 1

    cc.write_jsonl(rows, os.path.join(out_dir, "images.jsonl"))

    # scores.json: the harness-export shape (image_id -> score), for non-null labels.
    scores = {r["image_id"]: r["label"]["score"] for r in rows if r["label"]["score"] is not None}
    json.dump(scores, open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"))

    # batch.json
    gen = {}
    if os.path.exists(GEN_MANIFEST):
        gen = json.load(open(GEN_MANIFEST, encoding="utf-8"))
    batch = {
        "batch_id": BATCH_ID,
        "schema_version": 1,
        "created": "2026-06-23",
        "labeler": "matt",
        "generator_version": GENERATOR_VERSION,
        "source_run": manifest.get("source_jsonl", "data/generated/loose0/locations.jsonl"),
        "imported_from": "data/label_crops/loose0_v3",
        "imported_labels_from": "labels/location_labels.json (key draw_index|composition|palette)",
        "sampling_metaparameters": {
            # flat-generate concepts; rev4-only knobs (root_mix, depth range, …) absent.
            "lineage": gen.get("lineage"),
            "seed": gen.get("seed"),
            "box": gen.get("box"),
            "frame_width_range": gen.get("frame_width_range"),
            "screen_maxiter": gen.get("maxiter"),
            "accept_band": gen.get("accept_band"),
            "keepers_target": gen.get("keepers_target"),
            "keepers_produced": gen.get("keepers_produced"),
            "root_mix": None, "target_mix": None, "zoom_band": None,
            "depth_range": None, "black_cap": None, "candidates": None,
            "fps_radius": None, "root_zoom_8k": None, "orbit_band": None,
        },
        "present_gates": {
            "black_thresh": 0.30,
            "occupancy_floor": manifest.get("occupancy_gate", {}).get("floor", 0.23),
            "edge_floor": manifest.get("occupancy_gate", {}).get("edge_floor", 0.01),
            "tile_grid": manifest.get("occupancy_gate", {}).get("tile_grid", [32, 18]),
        },
        "render_defaults": {
            "maxiter": MAXITER, "width": WIDTH, "height": HEIGHT, "ss": SS,
            "filter": FILTER, "interior_mode": INTERIOR_MODE,
            "zoom_factor": manifest.get("zoom_factor", 0.4),
            "note": "ss=2 (present default; manifest render string mislabels it ss4). Lanczos-3 downsample.",
        },
        "counts": {
            "units": len(rows),
            "labeled": n_labeled,
            "score_hist": {str(k): v for k, v in score_hist.items()},
            "crops_copied": n_copied,
        },
    }
    json.dump(batch, open(os.path.join(out_dir, "batch.json"), "w", encoding="utf-8"), indent=2)

    print(f"batch {BATCH_ID}")
    print(f"  units (distinct image_id): {len(rows)}")
    print(f"  labeled: {n_labeled}  (score hist {score_hist})")
    print(f"  crops copied: {n_copied}  (into {os.path.relpath(crops_dir, cc.ROOT)})")
    print(f"  wrote images.jsonl, scores.json, batch.json under {os.path.relpath(out_dir, cc.ROOT)}")


if __name__ == "__main__":
    main()
