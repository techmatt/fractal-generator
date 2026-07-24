"""Ranking contact sheet from the dumped smooth fields, held-constant coloring
(same cyclic map for all 30) so the eye compares richness without palette
confound. Ranked by R_occ from richness.json. Not the target-style deliverable
(that's the rendered 3-palette sheets); this is the fair consistent-color judge.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ROOT / "out" / "q4_stage1" / "fields"
OUT = ROOT / "out" / "fair_rerender"
TILE_W = 384
COLS = 5
DENSITY = 0.9          # palette cycles per unit log-smooth (reveals filigree)
CMAP = plt.get_cmap("turbo")


def color_tile(stem, w):
    meta = json.load(open(FIELDS / f"{stem}.json"))
    fw, fh = meta["width"], meta["height"]
    a = np.fromfile(FIELDS / f"{stem}.bin", dtype=np.float32).reshape(fh, fw)
    fin = np.isfinite(a)
    L = np.log(np.where(fin, a, 1.0))
    lo, hi = np.percentile(L[fin], [1, 99])
    t = np.clip((L - lo) / max(hi - lo, 1e-9), 0, 1)
    t = (t * (hi - lo) * DENSITY) % 1.0          # cyclic banding
    rgb = (CMAP(t)[..., :3] * 255).astype(np.uint8)
    rgb[~fin] = 0                                 # interior lake black
    img = Image.fromarray(rgb)
    h = round(w * fh / fw)
    return img.resize((w, h), Image.LANCZOS)


def main():
    rows = json.load(open(OUT / "richness.json"))
    tiles = []
    for i, r in enumerate(rows):
        t = color_tile(r["id"], TILE_W)
        d = ImageDraw.Draw(t)
        label = f"#{i} {r['id']} Rocc={r['R_occ']:.3f}"
        d.rectangle([0, 0, TILE_W, 16], fill=(0, 0, 0))
        d.text((3, 3), label, fill=(255, 255, 255))
        tiles.append(t)
    th = tiles[0].height
    nrow = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE_W, nrow * th), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % COLS) * TILE_W, (i // COLS) * th))
    p = OUT / "richness_grid.png"
    sheet.save(p)
    print("wrote", p, sheet.size)


if __name__ == "__main__":
    main()
