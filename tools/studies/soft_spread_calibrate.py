"""soft-spread calibration harness — rank colored_clip pairs + a by-eye contact sheet.

The eyeball that calibrates the share-limit threshold tau consumed by
`colored_clip_spread.share_penalty`. Reads the 564 palette-ON CLIP vectors already
in the store (no re-embed) and:

  1. Full pairwise cosine over the 564; every unordered pair ranked most-similar-first.
  2. Each pair tagged on two axes: SAME- vs CROSS-location (recolor of one geometry
     vs two different geometries) and WITHIN- vs CROSS-cell (color_category, k16 cut).
  3. Ranked pair list -> `pairs.json` (keys, cosine, both tags) + a small `summary.json`.
  4. Contact sheet PNG: the display pairs (top overall UNION top cross-location, so the
     operative cross-location band is well represented), most-similar -> least, two
     Recipe-2 thumbnails per row labeled `cosine . same|cross-location . same|cross-cell`.
     One field dump per location (reused across its variants), recolor per variant.

The point (see prompt notes): same-location recolors legitimately sit ~0.85-0.97 and
stay distinct, so the too-close signal lives in CROSS-location pairs — surface both,
tagged, and read the "feels identical -> clearly different" gradient by eye to set the
band. This writes NOTHING load-bearing; tau stays uncalibrated until a human reads the
sheet.

    uv run python -m tools.curation.soft_spread_calibrate
    uv run python -m tools.curation.soft_spread_calibrate --no-sheet   # pairs.json only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.curation import colored_clip_spread as ccs   # noqa: E402
from tools.curation import colored_clip as cc           # noqa: E402
from tools import colormap as cm                         # noqa: E402

OUT = ROOT / "scratchpad/soft_spread"
CELL_LEVEL = ccs.DEFAULT_CELL_LEVEL

# display-pair selection: guarantee cross-location representation
N_OVERALL = 28      # top pairs regardless of tag
N_CROSS = 28        # top cross-location pairs
MAX_ROWS = 52

# thumbnail geometry (downscaled from the 640x360 recolor)
THUMB_W, THUMB_H = 256, 144
PAD = 8
LABEL_H = 22


# --------------------------------------------------------------------------- #
# 1-3. Pairwise cosine, tagging, ranked JSON.
# --------------------------------------------------------------------------- #
def rank_pairs(store: ccs.ColoredStore) -> list[dict]:
    sim = ccs.cosine_sim_matrix(store.unit)
    n = len(store.keys)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-sim[iu])   # most-similar first
    ai, bi = iu[0][order], iu[1][order]
    pairs = []
    for a, b in zip(ai.tolist(), bi.tolist()):
        ka, kb = store.keys[a], store.keys[b]
        same_loc = store.location_of(ka) == store.location_of(kb)
        same_cell = store.cell_of(ka, CELL_LEVEL) == store.cell_of(kb, CELL_LEVEL)
        pairs.append(dict(
            a=ka, b=kb, cosine=float(sim[a, b]),
            same_location=same_loc, same_cell=same_cell,
            cell_a=store.cell_of(ka, CELL_LEVEL), cell_b=store.cell_of(kb, CELL_LEVEL),
            palette_a=store.meta[ka]["palette"], palette_b=store.meta[kb]["palette"],
        ))
    return pairs


def summarize(pairs: list[dict]) -> dict:
    cos = np.array([p["cosine"] for p in pairs])
    same = np.array([p["same_location"] for p in pairs])

    def band(mask):
        c = cos[mask]
        if not len(c):
            return {}
        return dict(n=int(len(c)), max=float(c.max()), p95=float(np.percentile(c, 95)),
                    p75=float(np.percentile(c, 75)), median=float(np.median(c)),
                    min=float(c.min()))
    return dict(
        n_vectors=None, n_pairs=len(pairs), cell_level=CELL_LEVEL,
        same_location=band(same), cross_location=band(~same),
        note="cross_location is where the too-close signal lives; read the sheet to set tau.",
    )


def select_display(pairs: list[dict]) -> list[dict]:
    """Top overall UNION top cross-location, deduped, sorted most-similar first."""
    cross = [p for p in pairs if not p["same_location"]]
    chosen, seen = [], set()
    for p in pairs[:N_OVERALL] + cross[:N_CROSS]:
        k = (p["a"], p["b"])
        if k not in seen:
            seen.add(k)
            chosen.append(p)
    chosen.sort(key=lambda p: -p["cosine"])
    return chosen[:MAX_ROWS]


# --------------------------------------------------------------------------- #
# 4. Recolor thumbnails (Recipe-2, one field dump per location) + contact sheet.
# --------------------------------------------------------------------------- #
def load_records() -> dict[str, dict]:
    recs = {}
    for line in cc.RECORDS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["location_id"]] = r
    return recs


def recolor_needed(display: list[dict], recs: dict[str, dict],
                   lib: cm.PaletteLibrary) -> dict[str, Image.Image]:
    """key -> 640x360 RGB recolor for every variant appearing in the display pairs.

    One field dump per location (reused across its needed variants), recolor per
    variant via the cached-field Recipe-2 path (tools.colormap)."""
    # group needed variant_ids by location
    by_loc: dict[str, set[str]] = {}
    for p in display:
        for key in (p["a"], p["b"]):
            loc, var = key.split("/", 1)
            by_loc.setdefault(loc, set()).add(var)

    imgs: dict[str, Image.Image] = {}
    for i, (loc, variants) in enumerate(by_loc.items(), 1):
        rec = recs[loc]
        binp, jsonp = cc.ensure_field(rec)
        field = cm.load_field(str(binp), str(jsonp))
        prep = cm.stretch_field(field)
        profile = None
        cand_by_var = {c["variant_id"]: c for c in rec["palette_candidates"]}
        for var in sorted(variants):
            cand = cand_by_var[var]
            cfg = cc.candidate_config(field, cand)
            if cfg.transfer == "grad" and profile is None:
                profile = cm.gradient_transfer_profile(field, prep)
            rgb = cm.render_candidate(field, cfg, lib, prep=prep, profile=profile)
            imgs[f"{loc}/{var}"] = Image.fromarray(rgb)
        print(f"[{i}/{len(by_loc)}] {loc}  +{len(variants)} variants "
              f"(total {len(imgs)})", flush=True)
    return imgs


def _font(size: int):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheet(display: list[dict], imgs: dict[str, Image.Image], out_png: Path):
    font = _font(13)
    row_w = 2 * THUMB_W + 3 * PAD
    row_h = LABEL_H + THUMB_H + PAD
    sheet = Image.new("RGB", (row_w, PAD + row_h * len(display)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(display):
        y = PAD + i * row_h
        loc_tag = "same-loc " if p["same_location"] else "CROSS-loc"
        cell_tag = "same-cell" if p["same_cell"] else "cross-cell"
        label = (f"{p['cosine']:.4f}  .  {loc_tag}  .  {cell_tag}    "
                 f"{p['a']} [{p['palette_a']}]  |  {p['b']} [{p['palette_b']}]")
        color = (255, 210, 120) if not p["same_location"] else (170, 190, 210)
        draw.text((PAD, y + 4), label, fill=color, font=font)
        for j, key in enumerate((p["a"], p["b"])):
            th = imgs[key].resize((THUMB_W, THUMB_H), Image.LANCZOS)
            sheet.paste(th, (PAD + j * (THUMB_W + PAD), y + LABEL_H))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"wrote {out_png}  ({sheet.size[0]}x{sheet.size[1]})", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-sheet", action="store_true",
                    help="write pairs.json/summary.json only, skip the slow recolor+sheet")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    store = ccs.load_store()
    pairs = rank_pairs(store)
    summary = summarize(pairs)
    summary["n_vectors"] = len(store.keys)
    display = select_display(pairs)

    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "display_pairs.json").write_text(json.dumps(display, indent=1))
    print(f"ranked {len(pairs)} pairs over {len(store.keys)} vectors")
    print(f"  same-loc band:  {summary['same_location']}")
    print(f"  cross-loc band: {summary['cross_location']}")
    print(f"  display: {len(display)} rows "
          f"({sum(1 for p in display if not p['same_location'])} cross-location)")
    print(f"wrote {OUT/'pairs.json'}, summary.json, display_pairs.json")

    if args.no_sheet:
        return
    recs = load_records()
    lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))
    print(f"recoloring thumbnails for {len(display)} display pairs ...", flush=True)
    imgs = recolor_needed(display, recs, lib)
    build_sheet(display, imgs, OUT / "contact_sheet.png")


if __name__ == "__main__":
    main()
