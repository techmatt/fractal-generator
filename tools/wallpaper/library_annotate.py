#!/usr/bin/env python
r"""Phase-1 annotate tail — pool q3 locations -> durable library records + morph_clip + thumbnails.

The tail that replaces `emit_v1` in the prospecting loop (`prospect_orchestrator.py`). Given ONE
cycle's fresh pool (`build_fresh_discovery` output) + the run-scoped ledger, it emits ONE library
record per LOCATION (not per palette variant), computes the palette-blind grayscale morphology CLIP
embedding + a thumbnail, retains the smooth field for cheap Phase-2 colorize, and hands everything
to `library_store` for crash-safe persistence. Nothing renders at wallpaper res; nothing is emitted.

DENSE-CHEAP annotation set (build spec §Annotation):
  * identity            — per-family key; phoenix stamps fixed Ushiki c/p + coord_kind=z_viewport;
                          julia/julia_multibrot identity is viewport AND c.
  * location_potential  — v6 k3 / p_good / decoded_class JOINED from the ledger row (never recomputed).
  * descriptors.morph_clip — canonical grayscale CLIP (imported model+transform from colored_clip).
  * palette_candidates / mode_candidacy / colored_clip / predicted_p_ge3 / actual_p_ge3 — all persist
    as their reserved null/empty values (demand-driven at Phase 2, NOT filled here). No beam, no v3 gate.

GPU-exclusive child phase: loads CLIP once, embeds every location in the cycle, EXITS (freeing the
GPU) — same decoupled-phase discipline as the emit phase it replaces.

    uv run python -u tools/wallpaper/library_annotate.py \
        --pool <batch_dir> --ledger <ledger.jsonl> --ledger-start-line <n> \
        --run-id <run> --cycle <n> [--field-cache-gb 20] [--no-retain-fields]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tools import colormap as cm                       # noqa: E402
from tools.corpus import location as loc_mod           # noqa: E402
from tools.wallpaper import library_store as store     # noqa: E402

EXE = ROOT / "target" / "release" / "fractal-generator.exe"

# Morphology canon geometry — MUST match the grayscale/colored CLIP descriptor input res
# (tools/curation/colored_clip.py W,H,SS). Smooth base, box downsample, palette OFF.
W, H, SS = 640, 360, 2
THUMB_W, THUMB_H = 384, 216
GRAY_PAL = "__morph_gray__"                             # synthetic palette-off ramp (OKLab black->white)

# Fixed Ushiki phoenix constants (src/v4_cache.rs) — stamped into phoenix identity/render.
PHOENIX_C = {"re": "0.5667", "im": "0.0"}
PHOENIX_P = {"re": "-0.5", "im": "0.0"}
_JULIA_FAMILIES = {"julia", "julia_multibrot3", "julia_multibrot4", "julia_multibrot5"}


# --------------------------------------------------------------------------- #
# Grayscale morphology render (palette-OFF, via the committed coloring tail).
# --------------------------------------------------------------------------- #
def gray_library() -> cm.PaletteLibrary:
    """A normal PaletteLibrary + a synthetic non-cyclic black->white ramp, so the morphology
    grayscale render routes through the EXACT committed render_candidate tail (percentile
    stretch -> box downsample -> sRGB) that every colored render uses — palette OFF, nothing
    reimplemented."""
    lib = cm.PaletteLibrary()
    lib.colormaps[GRAY_PAL] = {"name": GRAY_PAL,
                               "stops": [[0.0, [0, 0, 0]], [1.0, [255, 255, 255]]],
                               "mirror_needed": False}
    lib.types[GRAY_PAL] = "non_cyclic"
    return lib


def morph_gray_image(field, lib) -> Image.Image:
    """(FieldData) -> 640x360 grayscale PIL RGB via the committed tail (smooth base, box, gamma 1)."""
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(
        palette=GRAY_PAL, location=field.location, eval_width=ow, eval_height=oh,
        reverse=False, log_premap="none", gamma=1.0, phase=0.0, n_cycles=1,
        interior_color=(0.0, 0.0, 0.0), filter="box", transfer="pct", transfer_gamma=0.0)
    rgb = cm.render_candidate(field, cfg, lib)          # (oh, ow, 3) uint8 sRGB
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
# Field retention — canonical smooth-field cache key, dumped once, reused.
# --------------------------------------------------------------------------- #
def ensure_field(loc, retain: bool, tmp_dir: Path, cache_root: Path):
    """Return a loaded FieldData for `loc`'s smooth field at the morph geometry. Retained mode
    caches under `cache_root` keyed by the canonical `field_stem` (deterministic from coords);
    non-retained dumps to a throwaway tmp and unlinks after load."""
    stem = store.field_stem(loc, "smooth", W, H, SS)
    cache_dir = cache_root if retain else tmp_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    binp, jsonp = cache_dir / f"{stem}.bin", cache_dir / f"{stem}.json"
    if not (binp.exists() and jsonp.exists()):
        cmd = [str(EXE), "render-one", "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
               "--width", str(W), "--height", str(H), "--supersample", str(SS),
               "--maxiter", str(loc.maxiter), "--dump-field", str(binp)]
        cmd += loc_mod.render_one_flags(loc)
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"field dump failed for {loc.key()}: {r.stderr[-400:]}")
    field = cm.load_field(str(binp), str(jsonp))
    if not retain:
        binp.unlink(missing_ok=True)
        jsonp.unlink(missing_ok=True)
    return field


# --------------------------------------------------------------------------- #
# Record construction (identity + ledger join; dense-cheap only).
# --------------------------------------------------------------------------- #
def render_location(render: dict) -> loc_mod.Location:
    """Pool render block -> canonical Location for the field dump. Phoenix's fixed Ushiki c/p are
    stamped when the block leaves them null (a z-viewport of the one fixed system)."""
    fam = render.get("fractal_type") or render.get("family") or "mandelbrot"
    if fam == "phoenix":
        return loc_mod.Location(
            family="phoenix", cx=render["cx"], cy=render["cy"], fw=render["fw"],
            maxiter=int(render["maxiter"]),
            c_re=render.get("c_re") or PHOENIX_C["re"], c_im=render.get("c_im") or PHOENIX_C["im"],
            family_params={"p_re": render.get("p_re") or PHOENIX_P["re"],
                           "p_im": render.get("p_im") or PHOENIX_P["im"]})
    return loc_mod.from_render_block(render)


def build_identity(render: dict, family: str, source_oid: str) -> dict:
    """Per-family identity block. Phoenix -> Ushiki c/p + z_viewport; julia* -> viewport AND c."""
    is_phoenix = family == "phoenix"
    is_julia = family in _JULIA_FAMILIES
    c = None
    if render.get("c_re") is not None:
        c = {"re": render["c_re"], "im": render["c_im"]}
    elif is_phoenix:
        c = dict(PHOENIX_C)
    return {
        "family": family,
        "fractal_type": render.get("fractal_type"),
        "cx": render["cx"], "cy": render["cy"], "fw": render["fw"],
        "maxiter": int(render["maxiter"]),
        "c": c,
        "p": dict(PHOENIX_P) if is_phoenix else None,
        "coord_kind": "z_viewport" if is_phoenix else ("julia_c_fixed" if is_julia else "c_plane"),
        "source_oid": source_oid,
    }


def build_record(source_oid: str, render: dict, prov: dict, led: dict,
                 run_id: str, cycle: int, source_ledger: str) -> dict:
    """One library record from a location's pool row + its joined ledger row. Dense-cheap blocks
    only; the demand-driven blocks persist as their reserved null/empty values."""
    fam = prov.get("family") or render.get("fractal_type") or "mandelbrot"
    lr = led.get(source_oid, {})
    return {
        "record_version": "0.1",
        "location_id": source_oid,               # ledger id == stable primary key (dedup anchor)
        "curated_from": None,                    # Phase-1 records are pre-curation
        "run_id": run_id, "cycle": cycle,        # provenance to tell fresh runs apart on inspection
        "source_ledger": source_ledger,
        "identity": build_identity(render, fam, source_oid),
        "location_potential": {
            "scorer_version": lr.get("scorer_version"),
            "k3": lr.get("k3"), "raw_top3": lr.get("raw_top3"),
            "decoded_class": lr.get("decoded_class"),
            "p_good": lr.get("p_good"), "p_notbad": lr.get("p_notbad"),
            "t_good": lr.get("t_good"), "reached_depth": lr.get("reached_depth"),
            "guard_pass": lr.get("guard_pass"),
            "seeder_decoded_class": prov.get("seeder_decoded_class"),
            "seeder_p_good": prov.get("seeder_p_good"),
            "source_ledger": prov.get("source_ledger"),
        },
        "descriptors": {
            "npz": "data/library_embeddings/",   # logical uid-addressable store (base + shards)
            "uid": source_oid,
            "morph_clip_dim": store.base_morph_dim(),
            "morph_v6": None,                    # SKIPPED: grayscale v6 prelogits not free for fresh locs
            "colored_clip": None,                # RESERVED (Phase-2 demand-driven)
            "thumbnail": f"thumbs/{source_oid}.jpg",
        },
        # --- reserved null/empty (demand-driven at Phase 2) ---
        "palette_candidates": [],
        "mode_candidacy": None,
        "wallpaper_quality": {
            "predicted_p_ge3": None,
            "actual_p_ge3": None,
        },
    }


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def load_ledger(ledger: Path) -> dict:
    led = {}
    if ledger.exists():
        with open(ledger, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    led[r["id"]] = r
    return led


def unique_locations(pool_images: Path) -> list[dict]:
    """One pool row per unique source_oid (first variant wins — geometry is identical across the
    location's beam variants). Skips rows with no source_oid (defensive)."""
    seen, out = set(), []
    with open(pool_images, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            oid = (row.get("provenance") or {}).get("source_oid")
            if not oid or oid in seen:
                continue
            seen.add(oid)
            out.append(row)
    return out


def annotate(args) -> dict:
    pool_images = Path(args.pool) / "images.jsonl"
    if not pool_images.exists():
        raise SystemExit(f"pool images.jsonl not found: {pool_images}")
    ledger = Path(args.ledger)
    led = load_ledger(ledger)
    rows = unique_locations(pool_images)

    # skip locations already in the store (resume idempotence) BEFORE any GPU/render work
    have = store.existing_location_ids(Path(args.records) if args.records else store.RECORDS_PATH)
    pending = [r for r in rows
               if (r["provenance"]["source_oid"] not in have)]
    print(f"[annotate] cycle {args.cycle}: {len(rows)} unique locations in pool, "
          f"{len(pending)} new (after dedup vs store)", flush=True)

    records_path = Path(args.records) if args.records else store.RECORDS_PATH
    thumbs_dir = Path(args.thumbs) if args.thumbs else store.THUMBS_DIR
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    tmp_field_dir = ROOT / "out" / "prospect" / "_field_tmp"
    field_cache_root = Path(args.field_cache_dir) if args.field_cache_dir else store.FIELD_CACHE_DIR

    shards_dir = Path(args.emb_shards) if args.emb_shards else store.EMB_SHARDS
    if not pending:
        return store.store_summary(records_path, thumbs_dir, shards_dir=shards_dir)

    # --- CLIP model (imported byte-identical from colored_clip) ---
    from tools.curation.colored_clip import load_clip, embed_clip
    model, tf = load_clip()
    lib = gray_library()

    records, uids, embs = [], [], []
    for i, row in enumerate(pending):
        prov, render = row["provenance"], row["render"]
        oid = prov["source_oid"]
        loc = render_location(render)
        try:
            field = ensure_field(loc, retain=args.retain_fields, tmp_dir=tmp_field_dir,
                                 cache_root=field_cache_root)
        except RuntimeError as e:
            print(f"  [{i+1}/{len(pending)}] {oid}: field FAILED ({e}); skipping", flush=True)
            continue
        img = morph_gray_image(field, lib)
        emb = embed_clip(model, tf, [img])[0].astype(np.float32)
        img.resize((THUMB_W, THUMB_H), Image.LANCZOS).convert("L").save(
            thumbs_dir / f"{oid}.jpg", "JPEG", quality=85)
        records.append(build_record(oid, render, prov, led, args.run_id, args.cycle,
                                    str(ledger)))
        uids.append(oid)
        embs.append(emb)
        if (i + 1) % 10 == 0 or i + 1 == len(pending):
            print(f"  [{i+1}/{len(pending)}] embedded+thumbed", flush=True)

    # --- persist: embedding shard FIRST (atomic), then records (dedup append) ---
    if uids:
        clip = np.stack(embs)
        emb_base = Path(args.emb_base) if args.emb_base else store.EMB_BASE
        shard = store.write_embedding_shard(args.run_id, args.cycle, uids, clip,
                                            shards_dir=shards_dir, emb_base=emb_base)
        written = store.append_records(records, records_path)
        print(f"[annotate] cycle {args.cycle}: +{len(written)} records, "
              f"embedding shard -> {shard.name} ({len(uids)} vecs)", flush=True)

    # --- LRU-evict retained fields under the cap ---
    if args.retain_fields:
        store.evict_field_cache_lru(args.field_cache_gb, cache_dir=field_cache_root,
                                    log=lambda m: print(m, flush=True))

    return store.store_summary(records_path, thumbs_dir, shards_dir=shards_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="build_fresh_discovery batch dir (has images.jsonl)")
    ap.add_argument("--ledger", required=True, help="run-scoped outcome_ledger.jsonl")
    ap.add_argument("--ledger-start-line", type=int, default=0,
                    help="(accepted for parity; dedup is by source_oid, watermark-agnostic)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--retain-fields", dest="retain_fields", action="store_true", default=True)
    ap.add_argument("--no-retain-fields", dest="retain_fields", action="store_false")
    ap.add_argument("--field-cache-gb", type=float, default=20.0)
    ap.add_argument("--field-cache-dir", default=None,
                    help="retained-field cache root (default data/library/field_cache)")
    # override sinks (tests point these at scratch)
    ap.add_argument("--records", default=None)
    ap.add_argument("--thumbs", default=None)
    ap.add_argument("--emb-shards", default=None)
    ap.add_argument("--emb-base", default=None)
    args = ap.parse_args()
    summ = annotate(args)
    print(f"[annotate] store now: {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
