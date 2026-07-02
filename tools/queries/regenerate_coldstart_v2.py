"""Regenerate the cold-start batch as coldstart_v2 — render-space-diverse param candidates.

The v1 diagnostic showed candidate collapse is confined to PARAM queries (palette/joint
stay ~6 effective-distinct; param drops to ~4.2/6 at ΔE<10), and that a sampler dedup
alone would fix only 7 of 25 near-dup pairs — the other 18 are distinct recipes that
collapse in RENDER space. This driver fixes param-candidate *selection* in render space
while leaving palette/joint generation byte-for-byte identical to v1.

Clean A/B: v1's exact locations and query-type assignments are REUSED (read from
`data/queries/coldstart_v1/records/`), so param-candidate-selection is the only variable.

  palette / joint : copied verbatim from v1 (record + 6 images + contact sheet).
  param           : same fixed palette + same rev/γ ranges as v1, but draw a POOL of
                    POOL_MULTIPLIER*6 (=24) with a min-γ-spacing guard (kills the literal
                    sampler-dups), recolor each on the cached ss2 field at THUMB_WIDTH,
                    then farthest-point-select 6 in render space (mean CIEDE2000) via the
                    shared `farthest_point_order` primitive fed that render-space matrix.

Ranges are UNCHANGED and the coloring path is untouched — this only changes which of the
same-range draws survive. The selected 6 are the full-res pool recolors themselves (no
re-render), saved at the same ss/eval as v1.

    uv run python tools/queries/regenerate_coldstart_v2.py            # full run
    uv run python tools/queries/regenerate_coldstart_v2.py --estimate # print est + exit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_sampler as qs                     # noqa: E402
import assemble_queries as aq                  # noqa: E402  (ensure_field, candidate_record, contact_sheet)
import color_metrics as cmet                   # noqa: E402  (validated CIEDE2000 + srgb_to_lab + THUMB_WIDTH)
sys.path.insert(0, str(qs.ROOT / "tools" / "palettes"))
import palette_features as pf                  # noqa: E402  (farthest_point_order — reused w/ render-space dmat)

ROOT = qs.ROOT
V1_DIR = ROOT / "data" / "queries" / "coldstart_v1"
V2_DIR = ROOT / "data" / "queries" / "coldstart_v2"

CANDIDATES_PER_QUERY = qs.CANDIDATES_PER_QUERY
POOL_SIZE = qs.POOL_MULTIPLIER * CANDIDATES_PER_QUERY
THUMB_WIDTH = cmet.THUMB_WIDTH


def per_query_rng(qid):
    """Deterministic, per-query independent RNG stream (reproducible from qid)."""
    seed = int(hashlib.sha1(f"{qid}|coldstart_v2".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def thumb_lab(img_u8):
    """In-memory mirror of diversity_diagnostic.load_thumb_lab: BOX-resize to THUMB_WIDTH
    then sRGB->CIELAB. (PNG save is lossless, so array == reopened-PNG.)"""
    im = Image.fromarray(img_u8).convert("RGB")
    w, h = im.size
    th = max(1, round(THUMB_WIDTH * h / w))
    im = im.resize((THUMB_WIDTH, th), Image.BOX)
    return cmet.srgb_to_lab(np.asarray(im))


def render_space_dmat(labs):
    """(n,n) symmetric mean-CIEDE2000 render-space distance over the pool thumbnails."""
    n = len(labs)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(cmet.ciede2000(labs[i], labs[j]).mean())
    return D


def copy_query_verbatim(qid, v2_dir):
    """Copy a palette/joint query byte-for-byte from v1: record, 6 images, contact sheet."""
    rec = json.loads((V1_DIR / "records" / f"{qid}.json").read_text())
    for cand in rec["candidates"]:
        rel = cand["image"]                     # 'images/q001_XXXX_k.png'
        shutil.copy2(V1_DIR / rel, v2_dir / rel)
    (v2_dir / "records" / f"{qid}.json").write_text(json.dumps(rec, indent=1))
    shutil.copy2(V1_DIR / f"{qid}.png", v2_dir / f"{qid}.png")   # contact sheet
    return rec["query_type"]


def regen_param_query(qid, v1_rec, sampler, lib, field_cache, prep_cache):
    """Regenerate one param query with render-space FPS selection. Returns (record,
    stats) where stats = {pool_drawn, recolor_s}. Location + type reused verbatim."""
    loc = v1_rec["location"]
    ref = qs.cm.LocationRef(
        kind=loc["family"], cx=loc["cx"], cy=loc["cy"], fw=loc["fw"],
        maxiter=int(loc["maxiter"]), c_re=loc.get("c_re"), c_im=loc.get("c_im"),
    )
    fixed_palette = v1_rec["candidates"][0]["palette"]   # param query => single fixed palette

    stem = aq._field_key(ref)
    if stem not in field_cache:
        fld, _ = aq.ensure_field(ref)           # cache hit (out/fields) => 0s
        field_cache[stem] = fld
        prep_cache[stem] = qs.cm.stretch_field(fld)
    fld = field_cache[stem]
    prep = prep_cache[stem]

    rng = per_query_rng(qid)
    pool = qs.draw_param_pool(ref, rng, sampler, palette=fixed_palette, pool_size=POOL_SIZE)

    # Recolor each pool candidate full-res (reusing the config-independent prep), keep the
    # full-res image (the selected 6 are saved from these — no re-render) + a thumb for ΔE.
    t0 = time.time()
    pool_imgs = [qs.cm.render_candidate(fld, cfg, lib, prep=prep) for cfg in pool]
    recolor_s = time.time() - t0
    labs = [thumb_lab(im) for im in pool_imgs]

    D = render_space_dmat(labs)
    sel = pf.farthest_point_order(list(range(len(pool))), k=CANDIDATES_PER_QUERY, dmat=D)

    sel_cfgs = [pool[i] for i in sel]
    sel_imgs = [pool_imgs[i] for i in sel]
    image_rels = []
    for ci, im in enumerate(sel_imgs):
        rel = f"images/{qid}_{ci}.png"
        Image.fromarray(im).save(V2_DIR / rel)
        image_rels.append(rel)

    cand_records = [aq.candidate_record(cfg, sampler, rel)
                    for cfg, rel in zip(sel_cfgs, image_rels)]
    record = {
        "query_id": qid,
        "query_type": v1_rec["query_type"],
        "location": loc,                        # byte-identical to v1
        "candidates": cand_records,
    }
    (V2_DIR / "records" / f"{qid}.json").write_text(json.dumps(record, indent=1))
    aq.contact_sheet(sel_imgs, sel_cfgs, qid, v1_rec["query_type"], V2_DIR / f"{qid}.png")
    return record, {"pool_drawn": len(pool), "recolor_s": recolor_s}


def main():
    ap = argparse.ArgumentParser(description="Regenerate coldstart_v2 (render-space param selection).")
    ap.add_argument("--estimate", action="store_true", help="print runtime estimate and exit")
    ap.add_argument("--no-resume", action="store_true", help="regenerate even completed param queries")
    args = ap.parse_args()

    (V2_DIR / "images").mkdir(parents=True, exist_ok=True)
    (V2_DIR / "records").mkdir(parents=True, exist_ok=True)

    v1_recs = {}
    for rp in sorted((V1_DIR / "records").glob("q*.json")):
        r = json.loads(rp.read_text())
        v1_recs[r["query_id"]] = r
    param_qids = [q for q, r in v1_recs.items() if r["query_type"] == "param"]
    other_qids = [q for q, r in v1_recs.items() if r["query_type"] != "param"]
    type_counts = {}
    for r in v1_recs.values():
        type_counts[r["query_type"]] = type_counts.get(r["query_type"], 0) + 1

    # --- runtime estimate (recolor-dominated; fields all cached => ~0 dump) ---------
    recolor_ms = 771.0          # v1 measured mean per recolor at ss2/1024x576
    est_recolors = len(param_qids) * POOL_SIZE
    est_s = est_recolors * recolor_ms / 1000.0 * 1.15   # +15% for thumb/ΔE/FPS/IO
    print(f"[v2] constants: POOL_MULTIPLIER={qs.POOL_MULTIPLIER} POOL_SIZE={POOL_SIZE} "
          f"GAMMA_MIN_SPACING={qs.GAMMA_MIN_SPACING} THUMB_WIDTH={THUMB_WIDTH}")
    print(f"[v2] v1 type counts: {type_counts}  (param={len(param_qids)} regen, "
          f"{len(other_qids)} palette/joint copied verbatim)")
    print(f"[v2] est: {est_recolors} param pool recolors @~{recolor_ms:.0f}ms "
          f"=> ~{est_s/60:.1f} min (fields all cached, 0 dump)")
    if args.estimate:
        return

    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    field_cache, prep_cache = {}, {}

    t_wall = time.time()

    # 1. palette/joint copied verbatim.
    for qid in other_qids:
        copy_query_verbatim(qid, V2_DIR)
    print(f"[v2] copied {len(other_qids)} palette/joint queries verbatim")

    # 2. param regenerated with render-space FPS selection.
    pool_short = []
    recolor_times = []
    done = 0
    for qid in param_qids:
        rec_path = V2_DIR / "records" / f"{qid}.json"
        imgs_ok = all((V2_DIR / f"images/{qid}_{k}.png").exists() for k in range(CANDIDATES_PER_QUERY))
        if not args.no_resume and rec_path.exists() and imgs_ok:
            done += 1
            continue
        _, st = regen_param_query(qid, v1_recs[qid], sampler, lib, field_cache, prep_cache)
        recolor_times.append(st["recolor_s"])
        if st["pool_drawn"] < POOL_SIZE:
            pool_short.append((qid, st["pool_drawn"]))
        done += 1
        if done % 10 == 0 or done == len(param_qids):
            el = time.time() - t_wall
            rate = el / max(1, len(recolor_times))
            eta = rate * (len(param_qids) - done)
            print(f"[v2] param {done}/{len(param_qids)}  elapsed {el/60:.1f}min  "
                  f"eta {eta/60:.1f}min", flush=True)

    wall = time.time() - t_wall

    # 3. batch_meta + gen.log
    meta = {
        "batch_id": "coldstart_v2",
        "purpose": ("Render-space-diverse cold-start batch: supersedes coldstart_v1. Param "
                    "candidates farthest-point-selected in render space (mean CIEDE2000) from "
                    "a guarded pool; palette/joint copied byte-for-byte from v1."),
        "derived_from": "coldstart_v1 (locations + query-type assignments reused verbatim)",
        "invocation": "uv run python tools/queries/regenerate_coldstart_v2.py",
        "n": len(v1_recs),
        "candidate_ss": qs.CANDIDATE_SS,
        "eval": [qs.EVAL_WIDTH, qs.EVAL_HEIGHT],
        "query_type_counts": type_counts,
        "param_selection": {
            "POOL_MULTIPLIER": qs.POOL_MULTIPLIER, "POOL_SIZE": POOL_SIZE,
            "GAMMA_MIN_SPACING": qs.GAMMA_MIN_SPACING, "THUMB_WIDTH": THUMB_WIDTH,
            "distance": "mean CIEDE2000 over BOX-thumbnails (diversity_diagnostic)",
            "selector": "palette_features.farthest_point_order(dmat=render_space)",
        },
        "results": {
            "queries": len(v1_recs),
            "candidates": len(v1_recs) * CANDIDATES_PER_QUERY,
            "param_regenerated": len(param_qids),
            "verbatim_copied": len(other_qids),
            "pool_short_queries": pool_short,
            "recolor_s_mean": float(np.mean(recolor_times)) if recolor_times else 0.0,
            "wall_seconds": wall,
        },
    }
    V2_DIR.joinpath("batch_meta.json").write_text(json.dumps(meta, indent=2))
    log = [
        f"[v2] param selection: POOL_SIZE={POOL_SIZE} GAMMA_MIN_SPACING={qs.GAMMA_MIN_SPACING} "
        f"THUMB_WIDTH={THUMB_WIDTH}",
        f"[v2] type counts: {type_counts}",
        f"[v2] param regenerated: {len(param_qids)}; palette/joint copied: {len(other_qids)}",
        f"[v2] pool-short queries (guard starved): {pool_short}",
        f"[v2] recolor mean: {meta['results']['recolor_s_mean']*1000:.0f}ms/query-pool"
        if recolor_times else "[v2] recolor mean: n/a (all resumed)",
        f"[wall] {wall:.1f}s total",
        "EXIT=0",
    ]
    V2_DIR.joinpath("gen.log").write_text("\n".join(log) + "\n")

    print()
    print(f"[done] coldstart_v2 -> {V2_DIR}")
    print(f"[v2] type counts: {type_counts}")
    print(f"[v2] pool-short queries: {pool_short if pool_short else 'none'}")
    print(f"[wall] {wall:.1f}s total")


if __name__ == "__main__":
    main()
