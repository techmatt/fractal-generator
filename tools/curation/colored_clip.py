"""colored_clip producer — per-candidate color-appearance CLIP descriptor.

Fills the one remaining schema gap in the location-library record
(`descriptors.colored_clip`, `docs/findings/library_record_schema.md` §B.3): a CLIP
embedding of each palette candidate's *delivered colored appearance*, the soft substrate
a within-cell spread selection would run on. This is the palette-ON twin of the
palette-BLIND grayscale morphology descriptor (the wiped `visual_dup/embed.py`; its
canonical robust-z transfer was recovered into `library_annotate.morph_gray_image`, see
`docs/findings/morph_parity.md`):

  * SAME CLIP model (`vit_base_patch16_clip_224.openai`), SAME timm eval transform,
    SAME 640x360 source resolution as the morphology canon renders — the only change
    is palette-on vs palette-off. So grayscale ("same shape") and colored ("same
    feel") land in the SAME embedding space.
  * PER-CANDIDATE, not per-location: color depends on palette+recipe, so there are
    K colored_clips per location (one per `palette_candidates[]` entry), vs the single
    morphology CLIP per location. Each embedding is keyed to `location_id/variant_id`.
  * FIELD-CACHE recolor path (Recipe-2): the smooth scalar field is dumped ONCE per
    location (reused from the `out/curation/morph_fields` cache, re-dumped if absent), then
    recolored per candidate via `tools.colormap.render_candidate` (cheap — no field
    math). CLIP only sees ~224px, so we color at the morphology 640x360 (smooth base,
    box filter, normal_map off), never wallpaper res.

Storage — LOAD-BEARING, out of scratch (fixes the dangling-reference risk flagged in
`docs/findings/library_gap_report.md` §C5 and `docs/findings/visual_dup.md`). Writes a
single `data/library_embeddings/embeddings.npz`:

    morph_uids  (62,)      str    corpus uid == curated_from
    morph_clip  (62,768)   f32    PROMOTED grayscale CLIP (verbatim copy) — PRIMARY
    morph_v6    (62,1280)  f32    PROMOTED in-house prelogits
    colored_keys (564,)    str    "location_id/variant_id"
    colored_clip (564,768) f32    palette-ON CLIP per candidate

and rewrites `library_records.jsonl` in place: `descriptors.npz` repointed at the
data/ store, `descriptors.colored_clip` filled (was null) with a store reference, and
each `palette_candidates[]` gains a `colored_clip_row` index. `data/` survives
`rm -r out/*` and scratch sweeps, so by-reference descriptors no longer dangle.

Producer + backfill + store-promotion ONLY — NOT the soft-spread selection that
consumes it. Re-runnable: reads candidate recipes straight from the records.

    uv run python -m tools.curation.colored_clip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import colormap as cm                      # noqa: E402
from tools.corpus import location as loc_mod          # noqa: E402
import timm                                            # noqa: E402

RECORDS = ROOT / "data/library/library_records.jsonl"
# Historical grayscale-morphology source. Its producer (visual_dup/embed.py) was wiped;
# its rows are already PROMOTED into STORE (morph_uids/morph_clip/morph_v6). build()/report()
# below only run to (re)promote — dead unless the gray producer is rebuilt (see morph_parity.md).
GRAY_NPZ = ROOT / "data/library_embeddings/gray_embeddings.npz"
FIELDS = ROOT / "out/curation/morph_fields"                # regenerable smooth-field cache
STORE = ROOT / "data/library_embeddings/embeddings.npz"    # LOAD-BEARING sink
EXE = ROOT / "target/release/fractal-generator.exe"
POOL_COLORMAPS = ROOT / "data/palettes/pool_colormaps.json"   # covers all 56 palettes
FEATURES = ROOT / "data/palettes/palette_features.json"
W, H, SS = 640, 360, 2   # morphology-canon geometry — MUST match embed.py's input res
CLIP_MODEL = "vit_base_patch16_clip_224.openai"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def uid_stem(uid: str) -> str:
    return uid.replace("/", "__")


# --------------------------------------------------------------------------- #
# CLIP — byte-for-byte the morphology descriptor's model + transform (embed.py).
# --------------------------------------------------------------------------- #
def load_clip():
    model = timm.create_model(CLIP_MODEL, pretrained=True, num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    return model, tf


@torch.no_grad()
def embed_clip(model, tf, imgs):
    outs = []
    for im in imgs:
        x = tf(im.convert("RGB")).unsqueeze(0).to(DEV)
        outs.append(model(x).squeeze(0).float().cpu().numpy())
    return np.stack(outs)


# --------------------------------------------------------------------------- #
# Field-cache recolor (Recipe-2). Field is palette-independent -> dump once, reuse.
# --------------------------------------------------------------------------- #
def location_from_record(rec) -> loc_mod.Location:
    """Record identity -> canonical Location (for the re-dump fallback). Phoenix's
    fixed Ushiki p is carried as a family_param; c is the additive constant."""
    idn = rec["identity"]
    c = idn.get("c") or {}
    p = idn.get("p") or {}
    fam = idn["family"]
    fparams = {}
    if fam == "phoenix" and p:
        fparams = {"p_re": p.get("re"), "p_im": p.get("im")}
    return loc_mod.Location(
        family=fam, cx=idn["cx"], cy=idn["cy"], fw=idn["fw"],
        maxiter=int(idn["maxiter"]),
        c_re=(c.get("re") if c else None), c_im=(c.get("im") if c else None),
        family_params=fparams,
    )


def ensure_field(rec):
    """Return (bin, json) for the location's cached smooth field, dumping it if absent."""
    stem = uid_stem(rec["curated_from"])
    binp, jsonp = FIELDS / f"{stem}.bin", FIELDS / f"{stem}.json"
    if binp.exists() and jsonp.exists():
        return binp, jsonp
    import subprocess
    FIELDS.mkdir(parents=True, exist_ok=True)
    loc = location_from_record(rec)
    cmd = [str(EXE), "render-one",
           "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
           "--width", str(W), "--height", str(H), "--supersample", str(SS),
           "--maxiter", str(loc.maxiter), "--dump-field", str(binp)]
    cmd += loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{rec['curated_from']}: field dump failed: {r.stderr[-400:]}")
    return binp, jsonp


def candidate_config(field, cand):
    """palette_candidates[] entry -> colormap.CandidateConfig (the durable recipe)."""
    col = cand["coloring"]
    ow, oh = field.out_size
    return cm.CandidateConfig(
        palette=cand["palette_ref"]["name"],
        location=field.location, eval_width=ow, eval_height=oh,
        reverse=bool(col.get("reverse", False)),
        log_premap=col.get("log_premap", "none"),
        gamma=float(col.get("gamma", 1.0)),
        phase=float(col.get("phase", 0.0)),
        n_cycles=int(col.get("n_cycles", 1)),
        interior_color=tuple(col.get("interior_color", (0.0, 0.0, 0.0))),
        filter="box",   # matches the morphology-canon box downsample (smooth base)
        transfer=col.get("transfer", "pct"),
        transfer_gamma=float(col.get("transfer_gamma", 0.0)),
    )


def render_candidates(rec, lib):
    """Recolor all K candidates of one location off its single cached field.

    Returns (keys, images): keys are "location_id/variant_id"; images are PIL RGB
    at 640x360. Caches the field's config-independent stretch + (lazily) the gradient
    profile so K recolors share the once-per-location prefix."""
    binp, jsonp = ensure_field(rec)
    field = cm.load_field(str(binp), str(jsonp))
    prep = cm.stretch_field(field)
    profile = None   # built lazily iff a candidate uses transfer='grad'
    keys, imgs = [], []
    for cand in rec["palette_candidates"]:
        cfg = candidate_config(field, cand)
        if cfg.transfer == "grad" and profile is None:
            profile = cm.gradient_transfer_profile(field, prep)
        rgb = cm.render_candidate(field, cfg, lib, prep=prep, profile=profile)
        keys.append(f"{rec['location_id']}/{cand['variant_id']}")
        imgs.append(Image.fromarray(rgb))
    return keys, imgs


# --------------------------------------------------------------------------- #
# Store + record backfill.
# --------------------------------------------------------------------------- #
def build():
    recs = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]

    # --- recolor every candidate off its cached field ---
    lib = cm.PaletteLibrary(str(POOL_COLORMAPS), str(FEATURES))
    all_keys, all_imgs = [], []
    for i, rec in enumerate(recs):
        keys, imgs = render_candidates(rec, lib)
        all_keys += keys
        all_imgs += imgs
        print(f"[{i+1}/{len(recs)}] {rec['location_id']}  +{len(keys)} candidates "
              f"(total {len(all_keys)})")

    # --- CLIP-embed the colored renders (palette-ON) ---
    print(f"CLIP {CLIP_MODEL} on {DEV}: embedding {len(all_imgs)} colored renders ...")
    model, tf = load_clip()
    colored = embed_clip(model, tf, all_imgs).astype(np.float32)
    print(f"  colored_clip dim {colored.shape}")

    # --- promote grayscale morphology embeddings (verbatim copy, no recompute) ---
    gz = np.load(GRAY_NPZ)
    morph_uids = np.asarray(gz["uids"])
    morph_clip = gz["clip"].astype(np.float32)
    morph_v6 = gz["v6"].astype(np.float32)

    STORE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        STORE,
        morph_uids=morph_uids, morph_clip=morph_clip, morph_v6=morph_v6,
        colored_keys=np.asarray(all_keys), colored_clip=colored,
    )
    size_mb = STORE.stat().st_size / 1e6
    print(f"wrote {STORE}  ({size_mb:.2f} MB)")

    # --- backfill records: repoint npz, fill colored_clip, index each candidate ---
    morph_row = {u: i for i, u in enumerate(morph_uids.tolist())}
    colored_row = {k: i for i, k in enumerate(all_keys)}
    store_ref = str(STORE.relative_to(ROOT)).replace("\\", "/")
    for rec in recs:
        d = rec["descriptors"]
        d["npz"] = store_ref                                  # promoted store
        uid = rec["curated_from"]
        d["clip_vitb16_row"] = morph_row.get(uid)             # rows re-resolved in store
        d["v6_prelogits_row"] = morph_row.get(uid)
        d["colored_clip"] = {                                 # was null — now a reference
            "npz": store_ref,
            "keys_array": "colored_keys",
            "embed_array": "colored_clip",
            "dim": int(colored.shape[1]),
            "model": CLIP_MODEL,
            "keyed_by": "location_id/variant_id",
            "per_candidate_row": "palette_candidates[].colored_clip_row",
            "note": "palette-ON CLIP; same model/transform/res as grayscale morph_clip",
        }
        for cand in rec["palette_candidates"]:
            key = f"{rec['location_id']}/{cand['variant_id']}"
            cand["colored_clip_row"] = colored_row.get(key)

    with RECORDS.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    print(f"backfilled {len(recs)} records -> {RECORDS}")

    report(recs, morph_uids, morph_clip, colored, all_keys)


def report(recs, morph_uids, morph_clip, colored, keys):
    ncand = sum(len(r["palette_candidates"]) for r in recs)
    embedded = sum(1 for r in recs for c in r["palette_candidates"]
                   if c.get("colored_clip_row") is not None)
    W_ = 74
    print("\n" + "=" * W_)
    print("colored_clip PRODUCER — report")
    print("=" * W_)
    print(f"coverage: {embedded}/{ncand} candidates embedded "
          f"({embedded/ncand*100:.1f}%) across {len(recs)} records")
    print(f"store: {STORE.relative_to(ROOT)}  {STORE.stat().st_size/1e6:.2f} MB")
    print(f"  morph_clip {morph_clip.shape} (PROMOTED grayscale, verbatim) | "
          f"colored_clip {colored.shape}")

    # promotion integrity: read the store BACK from disk, assert its grayscale rows
    # are byte-identical to the GRAY_NPZ source, and every record clip/v6 row
    # reference resolves to the correct uid position in the promoted store.
    src = np.load(GRAY_NPZ)
    st = np.load(STORE)
    same = (np.array_equal(src["clip"], st["morph_clip"])
            and np.array_equal(src["v6"], st["morph_v6"])
            and np.array_equal(src["uids"], st["morph_uids"]))
    upos = {u: i for i, u in enumerate(morph_uids.tolist())}
    ref_ok = all(r["descriptors"]["clip_vitb16_row"] == upos[r["curated_from"]]
                 and r["descriptors"]["v6_prelogits_row"] == upos[r["curated_from"]]
                 for r in recs)
    print(f"promotion: grayscale morph_clip/morph_v6/uids byte-identical to source: "
          f"{same}; record clip/v6-row references resolve: {ref_ok}")

    # spot-check: a location's K colored_clips DIFFER (distinct palettes) while its
    # single morphology embedding is UNCHANGED by this pass.
    r0 = recs[0]
    idx = [keys.index(f"{r0['location_id']}/{c['variant_id']}")
           for c in r0["palette_candidates"]]
    sub = colored[idx]
    unit = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    cos = unit @ unit.T
    off = cos[~np.eye(len(idx), dtype=bool)]
    mrow = upos[r0["curated_from"]]
    morph_unchanged = np.array_equal(st["morph_clip"][mrow], src["clip"][mrow])
    print(f"spot-check [{r0['location_id']}] K={len(idx)} colored candidates:")
    print(f"  distinct palettes -> distinct embeddings: pairwise cos "
          f"min={off.min():.3f} mean={off.mean():.3f} max={off.max():.3f} "
          f"(all <1.0 => distinct)")
    print(f"  its single morphology embedding unchanged by this pass: {morph_unchanged}")
    print("=" * W_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=RECORDS)
    args = ap.parse_args()
    if args.records != RECORDS:
        RECORDS = args.records  # noqa: F811 (allow override)
    build()
