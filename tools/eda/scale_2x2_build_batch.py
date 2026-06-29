# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
"""Phase 1 (store) — build the scale-controlled 2x2 corpus batch.

Joins each cell's present manifest -> that cell's pool (seed_index == pool idx) and
writes ONE batch with cell-tagged provenance. The render block is version-invariant
(the final reframed+zoomed frame present rendered); provenance carries the rev4-era
descent fields PLUS the experiment factors (cell, center_proposer, start_fw,
rev4_fix). Labels start null. Crops are copied in (gitignored, rebuildable).

  uv run python tools/eda/scale_2x2_build_batch.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus"))
import corpus_common as cc  # noqa: E402

BATCH_ID = "2026-06-25_scale_controlled_2x2"
GENERATOR_VERSION = "guided_descend_scale2x2"
PRESENT_MAXITER, WIDTH, HEIGHT, SS = 8000, 1280, 720, 2
FILTER, INTERIOR_MODE = "lanczos3", "black"

PRESENT_BASE = os.path.join(cc.ROOT, "out", "present", "scale_2x2")
POOL_BASE = os.path.join(cc.ROOT, "data", "guided_descend", "scale_2x2")

# cell -> (center_proposer, start_fw)
CELLS = {
    "A": ("8k_content_focus", 0.10),
    "B": ("8k_content_focus", 0.014093),
    "C": ("flat_acceptband", 0.10),
    "D": ("flat_acceptband", 0.014093),
}


def load_pool(cell: str) -> dict:
    path = os.path.join(POOL_BASE, f"cell_{cell.lower()}", "pool.jsonl")
    pool = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                pool[c["idx"]] = c
    return pool


def main() -> None:
    out_dir = cc.batch_dir(BATCH_ID)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    n_copied, missing_prov = 0, 0
    by_cell, by_palette, by_comp, by_depth = Counter(), Counter(), Counter(), Counter()
    per_cell_units = Counter()

    for cell, (proposer, start_fw) in CELLS.items():
        manifest_path = os.path.join(PRESENT_BASE, f"cell_{cell.lower()}", "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"WARN: missing manifest for cell {cell}: {manifest_path}; skipping")
            continue
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        present_dir = os.path.dirname(manifest_path)
        pool = load_pool(cell)
        seen = set()

        for c in manifest["crops"]:
            image_id = cc.image_id_from_output(c["output"])
            if image_id in seen:
                continue
            seen.add(image_id)
            # namespace the image_id by cell so the 4 cells never collide
            uid = f"{cell}_{image_id}"

            render = cc.render_block(
                cx=c["cx"], cy=c["cy"], fw=c["fw"],
                maxiter=PRESENT_MAXITER, palette=c["palette"], composition=c["composition"],
                width=WIDTH, height=HEIGHT, ss=SS, filter=FILTER, interior_mode=INTERIOR_MODE,
            )
            src = pool.get(c["seed_index"])
            if src is None:
                missing_prov += 1
            src = src or {}
            prov = cc.provenance_block(
                GENERATOR_VERSION, BATCH_ID,
                cell=cell, center_proposer=proposer, start_fw=start_fw, rev4_fix=True,
                root_src=src.get("root_src"), branch=src.get("branch"),
                depth=src.get("depth"), target_depth=src.get("target_depth"),
                walk_id=src.get("walk"), placement=src.get("placement"),
                focus_score=src.get("focus_score"),
                draw_index=c["draw_index"], seed_index=c["seed_index"],
                black_fraction=c.get("black_fraction"), interior_frac=c.get("interior_frac"),
                occupancy=c.get("occupancy"), void_guard=c.get("void_guard"),
            )
            rows.append(cc.make_row(uid, render, prov, cc.label_block()))

            by_cell[cell] += 1
            per_cell_units[cell] += 1
            by_palette[c["palette"]] += 1
            by_comp[c["composition"]] += 1
            by_depth[src.get("depth")] += 1

            src_jpg = os.path.join(present_dir, image_id + ".jpg")
            dst_jpg = os.path.join(crops_dir, uid + ".jpg")
            if os.path.exists(src_jpg) and not os.path.exists(dst_jpg):
                shutil.copy2(src_jpg, dst_jpg)
                n_copied += 1

    cc.write_jsonl(rows, os.path.join(out_dir, "images.jsonl"))
    json.dump({}, open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"))

    run_config = json.load(open(os.path.join(POOL_BASE, "run_config.json"), encoding="utf-8"))
    batch = {
        "batch_id": BATCH_ID,
        "schema_version": 1,
        "created": "2026-06-25",
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "experiment": "scale_controlled_2x2 (decouple center-proposer from start-fw)",
        "source_runs": {cell: f"data/guided_descend/scale_2x2/cell_{cell.lower()}/pool.jsonl" for cell in CELLS},
        "present_recipe": "single best-composition + 1 score-3 palette per candidate; 1280x720 ss2 m8000 lanczos3; content focus; zoom 0.4; occ-floor 0.321",
        "run_config": run_config,
        "render_defaults": {
            "maxiter": PRESENT_MAXITER, "width": WIDTH, "height": HEIGHT, "ss": SS,
            "filter": FILTER, "interior_mode": INTERIOR_MODE, "zoom_factor": 0.4,
            "palette_roster": "data/palettes/score3_colormaps.json (76 score-3 palettes)",
            "palettes_per_crop": 1, "all_compositions": False,
        },
        "present_gates": {"black_thresh": 0.30, "occupancy_floor": 0.321},
        "counts": {
            "units": len(rows), "labeled": 0,
            "by_cell": dict(by_cell), "by_composition": dict(by_comp),
            "by_palette": dict(by_palette),
            "depth_range_seen": [min(by_depth), max(by_depth)] if by_depth else None,
        },
    }
    json.dump(batch, open(os.path.join(out_dir, "batch.json"), "w", encoding="utf-8"), indent=2)

    print(f"batch -> {out_dir}")
    print(f"  units={len(rows)} copied={n_copied} missing_prov={missing_prov}")
    print(f"  by_cell={dict(by_cell)}")


if __name__ == "__main__":
    main()
