"""v4 augmentation montage — small label-preservation eyeball sample (NO TRAINING).

For ONE label-3 and ONE label-2 location, render the v4 augmentation family:
neutral twilight_shifted + K=5 family-stratified palettes (warm/cool/cyclic/
diverging/mono), each at 3 framings (scale 0.7 / shift@1.0 / scale 1.3, all
inside the label's recolor+shift<=0.5fw+scale[0.5,1.5] band). Stitches a labeled
montage per location so we can confirm the augmentation is label-preserving.

Uses the existing `render-one` subcommand (read-only). Writes only under data/v4/.

  uv run python tools/v4/montage.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "data" / "v4"
SCRATCH = OUT / "_montage_tiles"
SCRATCH.mkdir(parents=True, exist_ok=True)

W, H, SS = 640, 360, 2  # fast preview res (montage is for eyeballing, not perf)

# neutral always-included + K=5 family-stratified (all score-3 in palette_scores)
PALETTES = [
    ("twilight_shifted", "neutral"),
    ("RdPu", "warm"),
    ("cmr.jungle", "cool"),
    ("twilight", "cyclic"),
    ("PuOr", "diverging"),
    ("cividis", "mono"),
]
# framing: (label, fw_scale, shift_x_frac, shift_y_frac) — |shift| <= 0.5*fw
FRAMINGS = [
    ("scale0.7", 0.7, 0.0, 0.0),
    ("shift@1.0", 1.0, 0.4, 0.3),   # |shift| = 0.5*fw (band edge)
    ("scale1.3", 1.3, 0.0, 0.0),
]

# (name, cx, cy, fw, maxiter, label) — representative loose0 (unbiased) picks
LOCATIONS = [
    ("L3", "-0.7496970501636706", "0.041299430402989716", 0.00140391778141019, 2000, 3),
    ("L2", "0.007920043772864385", "-0.6519205772162964",  0.00174851822266184, 2000, 2),
]


def render(cx, cy, fw, maxiter, palette, out):
    cmd = [str(EXE), "render-one", "--cx", cx, "--cy", cy, "--fw", repr(fw),
           "--palette", palette, "--maxiter", str(maxiter),
           "--width", str(W), "--height", str(H), "--supersample", str(SS), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def build_montage(name, cx, cy, fw, maxiter, label):
    rows, cols = len(PALETTES), len(FRAMINGS)
    pad, top, lblw = 6, 34, 120
    cell_w, cell_h = W // 2, H // 2  # downscale tiles for the grid
    canvas = Image.new("RGB", (lblw + cols * (cell_w + pad) + pad,
                               top + rows * (cell_h + pad) + pad), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), f"{name}  label={label}  (cx={cx[:10]} cy={cy[:10]} fw={fw:.3e})",
           fill=(255, 255, 0))
    for ci, (fname, scale, sx, sy) in enumerate(FRAMINGS):
        x = lblw + ci * (cell_w + pad) + pad
        d.text((x, top - 14), fname, fill=(200, 200, 200))
    for ri, (pal, family) in enumerate(PALETTES):
        y = top + ri * (cell_h + pad) + pad
        d.text((6, y + cell_h // 2 - 6), f"{family}\n{pal[:14]}", fill=(200, 200, 200))
        for ci, (fname, scale, sx, sy) in enumerate(FRAMINGS):
            ncx = repr(float(cx) + sx * fw)
            ncy = repr(float(cy) + sy * fw)
            nfw = fw * scale
            tile = SCRATCH / f"{name}_{ri}_{ci}.png"
            render(ncx, ncy, nfw, maxiter, pal, tile)
            im = Image.open(tile).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(im, (lblw + ci * (cell_w + pad) + pad, y))
    out = OUT / f"montage_{name}.png"
    canvas.save(out)
    return out


def main():
    for loc in LOCATIONS:
        name = loc[0]
        out = build_montage(*loc)
        print(f"montage {name} -> {out}")


if __name__ == "__main__":
    main()
