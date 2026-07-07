"""Cliff-jarring diagnostic contact sheet.

Isolates *why* the authored cream-glow -> dark-shadow hard cliff reads as jarring.
Renders 4 high-contrast-cliff palettes x 4 variants through the production
`render-one --palette` path, then assembles a labeled row=palette / col=variant
contact sheet with a per-cell densified-LUT strip.

Variants:
  A  baseline (hard cliff)          on whq3_000 (deep, smooth location)
  B  --segments smooth              on whq3_000   (cliff removed entirely)
  C  --soft-cliff 0.03              on whq3_000   (crisp width-0.03 ramp)
  D  baseline (hard cliff)          on a busier/shallower q3 location

The densified colormap libraries (dense_baseline/smooth/softcliff.json) are built
by densify_authored.py; this script only renders + composits. Reuses lut_strip/
font from preview_render.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preview_render import lut_strip, font, sanitize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUTDIR = ROOT / "out/palette_preview/cliff-diag"
WORKERS = 4  # project cap
W, H, SS = 1024, 576, 2

PALETTES = ["Oxblood Reliquary", "Navy Sext", "Teal Monstrance", "Amber Vestment"]

# whq3_000 — deep, smooth q3 location (first-pass reference, matches preview_render).
LOC_SMOOTH = dict(cx="-0.76694625104943", cy="0.10338595858407715",
                  fw="5.9463444109435557e-05", maxiter=2843)
# Busier/shallower q3 mandelbrot: dense multi-hub filigree spiral, human label==3
# (source batch 2026-06-24_guided_descend_rev4occfix_v2filtered). fw ~118x larger
# than whq3_000 -> hard cliffs land on fine filigree instead of a smooth field.
LOC_BUSY = dict(cx="-0.7443218217097564", cy="-0.1647426033628334",
                fw="7.0058e-03", maxiter=8000)

# variant -> (densified colormaps file, location, column title)
VARIANTS = [
    ("A", OUTDIR / "dense_baseline.json",  LOC_SMOOTH, "A  hard cliff (baseline) — smooth loc"),
    ("B", OUTDIR / "dense_smooth.json",    LOC_SMOOTH, "B  all-smooth (no cliff) — smooth loc"),
    ("C", OUTDIR / "dense_softcliff.json", LOC_SMOOTH, "C  soft-cliff 0.03 — smooth loc"),
    ("D", OUTDIR / "dense_baseline.json",  LOC_BUSY,   "D  hard cliff (baseline) — BUSY loc"),
]


def render(colormaps: Path, palette: str, loc: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EXE), "render-one",
        "--cx", loc["cx"], "--cy", loc["cy"], "--fw", loc["fw"],
        "--maxiter", str(loc["maxiter"]),
        "--width", str(W), "--height", str(H), "--supersample", str(SS),
        "--colormaps", str(colormaps), "--palette", palette,
        "--out", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"render {palette} [{out.name}] failed:\n{r.stderr}")


def contact_sheet(render_dir: Path, luts: dict, out: Path, cell_w: int = 460) -> None:
    """rows = palettes, cols = variants A/B/C/D. Each cell: render + per-cell
    densified-LUT strip + (top row) column title + (left col) palette name."""
    cell_h = round(cell_w * H / W)
    strip_h = 16
    pad = 10
    head_h = 30   # column-title band on the top row
    label_w = 150  # palette-name gutter on the left
    tile_w = cell_w + pad
    tile_h = cell_h + strip_h + pad
    ncol, nrow = len(VARIANTS), len(PALETTES)
    sheet = Image.new("RGB", (label_w + ncol * tile_w + pad,
                              head_h + nrow * tile_h + pad), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    fhead, fname = font(15), font(17)

    for c, (tag, _cm, _loc, title) in enumerate(VARIANTS):
        x0 = label_w + c * tile_w
        draw.text((x0 + 2, 8), title, fill=(235, 235, 235), font=fhead)

    for r, pal in enumerate(PALETTES):
        y0 = head_h + r * tile_h
        draw.text((8, y0 + cell_h // 2 - 8), pal, fill=(235, 235, 235), font=fname)
        for c, (tag, _cm, _loc, _t) in enumerate(VARIANTS):
            x0 = label_w + c * tile_w
            img = Image.open(render_dir / f"{sanitize(pal)}__{tag}.png").resize(
                (cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x0, y0))
            sheet.paste(lut_strip(luts[tag][pal], cell_w, strip_h), (x0, y0 + cell_h))
    sheet.save(out)


def main() -> None:
    render_dir = OUTDIR / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    # per-variant LUTs, keyed by palette name, for the strips
    luts = {}
    for tag, cm, _loc, _t in VARIANTS:
        lib = json.loads(cm.read_text())
        luts[tag] = {p["name"]: p["stops"] for p in lib}

    jobs = [(pal, tag, cm, loc) for tag, cm, loc, _t in VARIANTS for pal in PALETTES]
    print(f"rendering {len(jobs)} frames @ {W}x{H} ss{SS} ...")

    def job(j):
        pal, tag, cm, loc = j
        render(cm, pal, loc, render_dir / f"{sanitize(pal)}__{tag}.png")
        return f"{pal} [{tag}]"

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for done in ex.map(job, jobs):
            print(f"  ok {done}")

    out = OUTDIR / "contact_sheet_cliff_diag.png"
    contact_sheet(render_dir, luts, out)
    print(f"contact sheet -> {out}")
    print(f"renders       -> {render_dir}")


if __name__ == "__main__":
    main()
