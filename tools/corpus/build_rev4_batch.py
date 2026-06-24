"""Part 3 (store) — build label_corpus batch 1 from the present output of run4.

Reads the present manifest produced by routing run4/pool.jsonl through present
(via pool_to_locations.py), re-attaches guided-descend provenance by joining the
manifest's seed_index back to the pool `idx`, and writes the store batch:

  batches/2026-06-24_guided_descend_rev4/{images.jsonl, batch.json, crops/}

The render block is version-invariant (the FINAL reframed+zoomed frame present
rendered); provenance carries the rev4-era fields (root_src, branch, depth,
target_depth, walk_id, placement, focus_score) plus present's per-frame measures.
Crops are copied in (gitignored, rebuildable from `render`).

Run (after present finishes):
  uv run python tools/corpus/build_rev4_batch.py \
      --manifest out/present/run4_present/manifest.json \
      --pool data/guided_descend/run4/pool.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter

import corpus_common as cc

BATCH_ID = "2026-06-24_guided_descend_rev4"
GENERATOR_VERSION = "guided_descend_rev4"

WIDTH, HEIGHT, SS = 1280, 720, 2
FILTER, INTERIOR_MODE = "lanczos3", "black"
PRESENT_MAXITER = 8000  # present default; the gates (black 0.30) are calibrated here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/present/run4_present/manifest.json")
    ap.add_argument("--pool", default="data/guided_descend/run4/pool.jsonl")
    ap.add_argument("--present-dir", default=None,
                    help="dir holding the present crop jpgs (default: manifest's dir)")
    ap.add_argument("--palettes-per-crop", type=int, default=1,
                    help="cap palette variants kept per (seed,composition) so the batch "
                         "lands near the ~1000 labeling-unit target (present rendered 2; "
                         "default 1 keeps the first palette per distinct crop)")
    a = ap.parse_args()

    manifest_path = os.path.join(cc.ROOT, a.manifest) if not os.path.isabs(a.manifest) else a.manifest
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    present_dir = a.present_dir or os.path.dirname(manifest_path)

    # provenance lookup: pool idx -> candidate (seed_index round-trips to idx).
    pool = {}
    with open(os.path.join(cc.ROOT, a.pool) if not os.path.isabs(a.pool) else a.pool, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                pool[c["idx"]] = c

    out_dir = cc.batch_dir(BATCH_ID)
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    seen = set()
    per_crop = Counter()  # (seed_index, composition) -> palette variants kept
    missing_prov = 0
    n_copied = 0
    by_palette, by_comp, by_root = Counter(), Counter(), Counter()
    by_depth, by_branch = Counter(), Counter()

    for c in manifest["crops"]:
        image_id = cc.image_id_from_output(c["output"])
        if image_id in seen:
            continue  # palette-with-replacement dup -> one unit
        # cap palette variants per distinct (seed, composition) crop.
        crop_key = (c["seed_index"], c["composition"])
        if per_crop[crop_key] >= a.palettes_per_crop:
            continue
        per_crop[crop_key] += 1
        seen.add(image_id)

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
            root_src=src.get("root_src"),
            branch=src.get("branch"),
            depth=src.get("depth"),
            target_depth=src.get("target_depth"),
            walk_id=src.get("walk"),
            placement=src.get("placement"),
            focus_score=src.get("focus_score"),
            draw_index=c["draw_index"],
            seed_index=c["seed_index"],
            black_fraction=c.get("black_fraction"),
            interior_frac=c.get("interior_frac"),
            occupancy=c.get("occupancy"),
            void_guard=c.get("void_guard"),
        )
        label = cc.label_block()  # unlabeled until Matt scores it
        rows.append(cc.make_row(image_id, render, prov, label))

        by_palette[c["palette"]] += 1
        by_comp[c["composition"]] += 1
        by_root[src.get("root_src")] += 1
        by_branch[src.get("branch")] += 1
        by_depth[src.get("depth")] += 1

        src_jpg = os.path.join(present_dir, image_id + ".jpg")
        dst_jpg = os.path.join(crops_dir, image_id + ".jpg")
        if os.path.exists(src_jpg) and not os.path.exists(dst_jpg):
            shutil.copy2(src_jpg, dst_jpg)
            n_copied += 1

    cc.write_jsonl(rows, os.path.join(out_dir, "images.jsonl"))
    # scores.json starts empty (nothing labeled yet); harness will populate it.
    json.dump({}, open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"))

    # observed-from-pool root mixture (the 50/50 root_mix target check).
    pool_roots = Counter(v.get("root_src") for v in pool.values())

    batch = {
        "batch_id": BATCH_ID,
        "schema_version": 1,
        "created": "2026-06-24",
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "source_run": os.path.relpath(os.path.join(cc.ROOT, a.pool), cc.ROOT).replace("\\", "/")
        if not os.path.isabs(a.pool) else a.pool,
        "present_input": "out/present/run4_bridge/locations.jsonl (pool_to_locations.py bridge)",
        "note": ("guided-descend run4 invocation was not persisted; sampling_metaparameters "
                 "below are the rev4 documented defaults (src/cli.rs GuidedDescendArgs) plus "
                 "observed-from-pool stats. The d1->d2 occ-floor disable (Part 0) postdates run4."),
        "sampling_metaparameters": {
            "root_mix": 0.5,
            "root_zoom_8k": 0.10,
            "root8k_criterion": {"black_max": 0.80, "mean_lo": 8.0, "mean_hi": 120.0, "var_floor": 6.0},
            "flat_root": {"box": "-2.0,0.7,-1.2,1.2", "fw_lo": 0.003, "fw_hi": 0.05, "screen_width": 320},
            "target_mix": {"w_foci": 0.70, "w_density": 0.10, "w_random": 0.20},
            "placement": "0.25,0.40,0.35",
            "zoom_band": [0.35, 0.50],
            "depth_range": [4, 10],
            "n_walks": 80,
            "sigma_band": "16,20,24,28,32",
            "foci_diversity_radius": 0.12,
            "random_boundary": True,
            "node_width": 768,
            "descent_candidates": 4,
            "descent_black_cap": 0.30,
            "descent_occ_floor": 0.321,
            "field_maxiter": 1000,
            "bailout": 1e6,
            "seed": 0,
            "observed_pool": {
                "candidates": len(pool),
                "root_src_counts": dict(pool_roots),
                "depth_range_seen": [min(by_depth), max(by_depth)] if by_depth else None,
            },
        },
        "present_gates": {
            "black_thresh": 0.30,
            "occupancy_floor": manifest.get("occupancy_gate", {}).get("floor", 0.321),
            "edge_floor": manifest.get("occupancy_gate", {}).get("edge_floor", 0.01),
            "tile_grid": manifest.get("occupancy_gate", {}).get("tile_grid", [32, 18]),
        },
        "render_defaults": {
            "maxiter": PRESENT_MAXITER, "width": WIDTH, "height": HEIGHT, "ss": SS,
            "filter": FILTER, "interior_mode": INTERIOR_MODE,
            "zoom_factor": manifest.get("zoom_factor", 0.4),
            "palette_roster": "data/palettes/score3_colormaps.json (76 score-3 palettes)",
            "palettes_rendered_per_crop": 2,
            "palettes_kept_per_crop": a.palettes_per_crop,
        },
        "counts": {
            "units": len(rows),
            "labeled": 0,
            "by_palette": dict(by_palette),
            "by_composition": dict(by_comp),
            "by_root_src": {str(k): v for k, v in by_root.items()},
            "by_branch": {str(k): v for k, v in by_branch.items()},
            "by_depth": {str(k): v for k, v in sorted(by_depth.items(), key=lambda kv: (kv[0] is None, kv[0]))},
            "crops_copied": n_copied,
            "missing_provenance": missing_prov,
        },
    }
    json.dump(batch, open(os.path.join(out_dir, "batch.json"), "w", encoding="utf-8"), indent=2)

    print(f"batch {BATCH_ID}")
    print(f"  units: {len(rows)}  (crops copied {n_copied}, missing provenance {missing_prov})")
    print(f"  per-composition: {dict(by_comp)}")
    print(f"  per-root_src:    {dict(by_root)}")
    print(f"  per-branch:      {dict(by_branch)}")
    print(f"  per-depth:       {dict(sorted(by_depth.items(), key=lambda kv:(kv[0] is None,kv[0])))}")
    print(f"  distinct palettes used: {len(by_palette)}")
    print(f"  wrote under {os.path.relpath(out_dir, cc.ROOT)}")


if __name__ == "__main__":
    main()
