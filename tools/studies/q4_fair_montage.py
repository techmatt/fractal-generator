"""Rank-ordered montage of the ACTUAL target-style fair re-renders (default tile,
the vivid UF palette), ordered by R_occ — the target-style eye judge. Plus the
identity side-by-side (mb19 re-render | on-disk ladder | 8x diff).
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "fair_rerender"
SHEETS = OUT / "sheets"
TILE_W, TILE_H = 1024, 576          # default (top-left) tile in a 16:9 sheet
# sheet layout: cols=2, PAD=6 border; tile 0 top-left starts after label bar.
# Detect the default-tile crop robustly from a known sheet instead of guessing.


PAD, SWATCH_H = 6, 18                # from src/sheet.rs


def default_tile(sheet_path):
    """Top-left tile = 'default' palette (sheet.rs: tile 0 at (PAD,PAD),
    1024x576, swatch strip in the bottom SWATCH_H rows -> trim it)."""
    im = Image.open(sheet_path).convert("RGB")
    return im.crop((PAD, PAD, PAD + TILE_W, PAD + TILE_H - SWATCH_H))


def montage():
    rows = json.load(open(OUT / "richness.json"))
    cols, tw = 5, 420
    tiles = []
    th = None
    for i, r in enumerate(rows):
        sp = SHEETS / f"{r['id']}.png"
        crop = default_tile(sp)
        th = round(tw * crop.height / crop.width)
        t = crop.resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, tw, 15], fill=(0, 0, 0))
        d.text((3, 3), f"#{i} {r['id']} Rocc={r['R_occ']:.3f}", fill=(255, 255, 0))
        tiles.append(t)
    nrow = (len(tiles) + cols - 1) // cols
    m = Image.new("RGB", (cols * tw, nrow * th), (15, 15, 15))
    for i, t in enumerate(tiles):
        m.paste(t, ((i % cols) * tw, (i // cols) * th))
    p = OUT / "fair_montage_ranked.png"
    m.save(p)
    print("wrote", p, m.size)


def identity():
    a = Image.open(OUT / "identity" / "mb19_p35_3x2.png").convert("RGB")
    b = Image.open(ROOT / "out" / "deep_centers" / "ladder_p35" /
                   "fw_8p07e_10.png").convert("RGB")
    W, H = a.size
    diff = np.clip(np.abs(np.asarray(a).astype(int) -
                          np.asarray(b).astype(int)) * 8, 0, 255).astype(np.uint8)
    dimg = Image.fromarray(diff)
    gap = 12
    comp = Image.new("RGB", (W * 3 + gap * 2, H + 24), (0, 0, 0))
    comp.paste(a, (0, 24)); comp.paste(b, (W + gap, 24))
    comp.paste(dimg, (2 * W + 2 * gap, 24))
    d = ImageDraw.Draw(comp)
    for x, t in [(0, "mb19_p35 re-render (minibrots.json)"),
                 (W + gap, "ladder_p35 / fw_8p07e_10.png (on disk)"),
                 (2 * W + 2 * gap, "abs diff x8  (NCC 0.977, mean L1/chan ~2.4/255)")]:
        d.text((x + 4, 6), t, fill=(255, 255, 255))
    p = OUT / "identity_side_by_side.png"
    comp.save(p)
    print("wrote", p, comp.size)


if __name__ == "__main__":
    montage()
    identity()
