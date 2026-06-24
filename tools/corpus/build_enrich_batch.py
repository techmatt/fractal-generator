"""Stage D (store) — write the v2-filtered enrichment batch into the label corpus.

Reads the Stage C `selection_full.jsonl` (enriched + random_eval rows, each with
its v2 filter_score / argmax_palette / k_scores / selection_role) and the source
`pool.jsonl` (re-attaches guided-descend provenance by `idx`), and writes:

  batches/<BATCH_ID>/{images.jsonl, batch.json, scores.json (empty)}

The crops themselves are rendered separately by `enrich --mode render` straight
into this batch's `crops/` dir (ss4 + Lanczos-3, q90 JPG) — this script only
writes the metadata. The `render` block is version-invariant (center composition
at the stored cx/cy/fw under the argmax palette); provenance carries the rev4-era
fields PLUS the v2 selection bias (so it's always recoverable).

batch.json states PLAINLY: this batch is v2-selection-biased; only the
`random_eval` slice is an unbiased rev4 sample.

Run (after enrich_select.py, before/after the render):
  uv run python tools/corpus/build_enrich_batch.py \
      --selection data/enrich/run5/selection_full.jsonl \
      --pool data/guided_descend/run5/pool.jsonl \
      --meta data/enrich/run5/score_meta.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import corpus_common as cc

BATCH_ID = "2026-06-24_guided_descend_rev4occfix_v2filtered"
GENERATOR_VERSION = "guided_descend_rev4occfix"
V2_MODEL_ID = "data/classifier/v2/model_best.pt"

WIDTH, HEIGHT, SS = 1280, 720, 4
FILTER, INTERIOR_MODE = "lanczos3", "black"
COMPOSITION = "center"
MAXITER = 8000


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/enrich/run5/selection_full.jsonl")
    ap.add_argument("--pool", default="data/guided_descend/run5/pool.jsonl")
    ap.add_argument("--meta", default="data/enrich/run5/score_meta.jsonl")
    a = ap.parse_args()

    sel = load_jsonl(os.path.join(cc.ROOT, a.selection) if not os.path.isabs(a.selection) else a.selection)
    pool_rows = load_jsonl(os.path.join(cc.ROOT, a.pool) if not os.path.isabs(a.pool) else a.pool)
    pool = {r["idx"]: r for r in pool_rows}

    # score_meta header: the K palette roster + filter config (for batch.json).
    meta_path = os.path.join(cc.ROOT, a.meta) if not os.path.isabs(a.meta) else a.meta
    filter_cfg = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("kind") == "header":
                filter_cfg = o
                break

    out_dir = cc.batch_dir(BATCH_ID)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    rows = []
    by_palette, by_role, by_root, by_depth, by_branch = (Counter() for _ in range(5))
    missing_prov = 0
    for r in sel:
        idx = r["idx"]
        image_id = r["image_id"]
        render = cc.render_block(
            cx=r["cx"], cy=r["cy"], fw=r["fw"],
            maxiter=MAXITER, palette=r["argmax_palette"], composition=COMPOSITION,
            width=WIDTH, height=HEIGHT, ss=SS, filter=FILTER, interior_mode=INTERIOR_MODE,
        )
        src = pool.get(idx)
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
            draw_index=idx,
            seed_index=idx,
            black_fraction=r.get("black_fraction"),   # from the scoring render
            interior_frac=None,
            occupancy=r.get("occupancy"),
            void_guard=None,
            # --- v2 enrichment bias (recoverable) ---
            selection_role=r["selection_role"],
            filter_score=r.get("filter_score"),
            argmax_palette=r.get("argmax_palette"),
            k_scores=r.get("k_scores"),
            v2_est_class=r.get("est_class"),
            v2_model_id=V2_MODEL_ID,
        )
        rows.append(cc.make_row(image_id, render, prov, cc.label_block()))
        by_palette[r["argmax_palette"]] += 1
        by_role[r["selection_role"]] += 1
        by_root[src.get("root_src")] += 1
        by_branch[src.get("branch")] += 1
        by_depth[src.get("depth")] += 1

    cc.write_jsonl(rows, os.path.join(out_dir, "images.jsonl"))
    json.dump({}, open(os.path.join(out_dir, "scores.json"), "w", encoding="utf-8"))

    tau = min((r.get("filter_score") for r in sel if r["selection_role"] == "enriched"),
              default=None)
    pool_roots = Counter(v.get("root_src") for v in pool.values())

    batch = {
        "batch_id": BATCH_ID,
        "schema_version": 1,
        "created": "2026-06-24",
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "source_run": os.path.relpath(os.path.join(cc.ROOT, a.pool), cc.ROOT).replace("\\", "/")
        if not os.path.isabs(a.pool) else a.pool,
        "bias_statement": (
            "THIS BATCH IS v2-SELECTION-BIASED. The `enriched` rows (selection_role) were "
            "chosen as the top locations by v2 P(not-bad) (best over K palettes), so their "
            "labeled positive rate will look BETTER than reality. Only the `random_eval` rows "
            "are an unbiased, uniform sample of the post-gate pool and form the honest rev4 "
            "eval set for v3. Provenance.filter_score + selection_role keep the bias fully "
            "recoverable. Training (corpus_reader) unions crops blind to this; evaluation of "
            "rev4 for v3 MUST restrict to selection_role==random_eval."
        ),
        "note": (
            "guided_descend rev4 policy + the d1->d2 occupancy-floor disable (Part-0 fix, "
            "default-on: --descent-occ-at-d1d2 NOT passed). Locations are labeled at center "
            "composition under the v2-argmax score-3 palette (1 palette per location); no "
            "present zoom/composition search. black_fraction/occupancy are from the cheap "
            "scoring render (ss1), not the final ss4 crop."
        ),
        "sampling_metaparameters": {
            "root_mix": 0.5, "root_zoom_8k": 0.10,
            "root8k_criterion": {"black_max": 0.80, "mean_lo": 8.0, "mean_hi": 120.0, "var_floor": 6.0},
            "flat_root": {"box": "-2.0,0.7,-1.2,1.2", "fw_lo": 0.003, "fw_hi": 0.05, "screen_width": 320},
            "target_mix": {"w_foci": 0.70, "w_density": 0.10, "w_random": 0.20},
            "placement": "0.25,0.40,0.35", "zoom_band": [0.35, 0.50], "depth_range": [4, 10],
            "n_walks": 660, "sigma_band": "16,20,24,28,32", "foci_diversity_radius": 0.12,
            "random_boundary": True, "node_width": 768, "descent_candidates": 4,
            "descent_black_cap": 0.30, "descent_occ_floor": 0.321, "descent_occ_at_d1d2": False,
            "field_maxiter": 1000, "bailout": 1e6, "seed": 5,
            "observed_pool": {
                "candidates": len(pool),
                "root_src_counts": dict(pool_roots),
                "depth_range_seen": [min(by_depth), max(by_depth)] if by_depth else None,
            },
        },
        "v2_filter": {
            "model_id": V2_MODEL_ID,
            "transform": "1280x720 -> 384x224 bicubic stretch + normalize (inference.py deploy mirror)",
            "filter_score": "max over K palettes of P(not-bad) = sigmoid(logit_0), CORN ordinal head",
            "k_palettes": filter_cfg.get("k"),
            "palette_sampling": "per-location seeded draw from the score-3 roster",
            "roster": filter_cfg.get("roster"),
            "roster_size": filter_cfg.get("roster_size"),
            "palette_seed": filter_cfg.get("seed"),
            "score_ss": filter_cfg.get("score_ss"),
            "implied_tau_enriched": tau,
        },
        "present_gates": {"black_thresh": 0.30, "occupancy_floor": 0.321,
                          "edge_floor": 0.010, "tile_grid": [32, 18]},
        "render_defaults": {
            "maxiter": MAXITER, "width": WIDTH, "height": HEIGHT, "ss": SS,
            "filter": FILTER, "interior_mode": INTERIOR_MODE, "composition": COMPOSITION,
            "palette_roster": "data/palettes/score3_colormaps.json",
            "palettes_per_location": 1,
        },
        "counts": {
            "units": len(rows), "labeled": 0,
            "by_selection_role": dict(by_role),
            "by_palette": dict(by_palette),
            "by_root_src": {str(k): v for k, v in by_root.items()},
            "by_branch": {str(k): v for k, v in by_branch.items()},
            "by_depth": {str(k): v for k, v in sorted(by_depth.items(), key=lambda kv: (kv[0] is None, kv[0]))},
            "distinct_palettes": len(by_palette),
            "missing_provenance": missing_prov,
        },
    }
    json.dump(batch, open(os.path.join(out_dir, "batch.json"), "w", encoding="utf-8"), indent=2)

    print(f"batch {BATCH_ID}")
    print(f"  units: {len(rows)}  (missing provenance {missing_prov})")
    print(f"  by role:    {dict(by_role)}")
    print(f"  by root:    {dict(by_root)}")
    print(f"  by depth:   {dict(sorted(by_depth.items(), key=lambda kv:(kv[0] is None,kv[0])))}")
    print(f"  distinct palettes: {len(by_palette)}   tau(enriched)={tau}")
    print(f"  wrote {os.path.relpath(out_dir, cc.ROOT)} (render crops with enrich --mode render)")


if __name__ == "__main__":
    main()
