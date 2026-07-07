"""Palette preview harness: render every palette in a densified library at ONE
fixed q3 location (smooth mode) + assemble a labeled contact sheet.

Standalone eyeball tool — does NOT touch the label corpus or any training set.
Renders through the production Rust `render-one --palette` colorer (location
profile / smooth), so the preview matches what production will emit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
WORKERS = 4  # project cap

# --- fixed q3 location: whq3_000 (human label==3), mandelbrot ----------------
LOC = dict(
    name="whq3_000",
    cx="-0.76694625104943",
    cy="0.10338595858407715",
    fw="5.9463444109435557e-05",
    maxiter=2843,
)
W, H, SS = 1024, 576, 2


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def render_one(colormaps: Path, palette: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(EXE), "render-one",
        "--cx", LOC["cx"], "--cy", LOC["cy"], "--fw", LOC["fw"],
        "--maxiter", str(LOC["maxiter"]),
        "--width", str(W), "--height", str(H), "--supersample", str(SS),
        "--colormaps", str(colormaps), "--palette", palette,
        "--out", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"render {palette} failed:\n{r.stderr}")


def lut_strip(stops: list, width: int, height: int) -> Image.Image:
    """Horizontal strip of the densified LUT sampled across [0,1]."""
    strip = Image.new("RGB", (width, height))
    px = strip.load()
    n = len(stops)
    for x in range(width):
        t = x / max(1, width - 1)
        i = min(n - 1, int(t * (n - 1)))
        _, rgb = stops[i]
        for y in range(height):
            px[x, y] = tuple(rgb)
    return strip


def font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(entries: list[dict], render_dir: Path, out: Path,
                  cols: int = 4, cell_w: int = 512) -> None:
    """entries: [{name, sanitized, stops}]. Each cell = render + name label +
    thin densified-LUT strip beneath."""
    cell_h = round(cell_w * H / W)
    strip_h = 18
    label_h = 26
    pad = 8
    tile_w = cell_w + 2 * pad
    tile_h = cell_h + strip_h + label_h + 2 * pad
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    f = font(16)

    for k, e in enumerate(entries):
        r, c = divmod(k, cols)
        x0, y0 = c * tile_w + pad, r * tile_h + pad
        img = Image.open(render_dir / f"{e['sanitized']}.png").resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(img, (x0, y0))
        # LUT strip beneath the render
        sheet.paste(lut_strip(e["stops"], cell_w, strip_h), (x0, y0 + cell_h))
        # name label beneath the strip
        ty = y0 + cell_h + strip_h + 4
        draw.text((x0 + 2, ty), e["name"], fill=(235, 235, 235), font=f)
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--colormaps", type=Path,
                    default=ROOT / "out/palette_preview/dramatic-test/densified.json")
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "out/palette_preview/dramatic-test")
    args = ap.parse_args()

    lib = json.loads(args.colormaps.read_text())
    render_dir = args.outdir / "renders"
    entries = [dict(name=c["name"], sanitized=sanitize(c["name"]), stops=c["stops"]) for c in lib]

    print(f"location {LOC['name']}: cx={LOC['cx']} cy={LOC['cy']} fw={LOC['fw']} maxiter={LOC['maxiter']}")
    print(f"rendering {len(entries)} palette(s) @ {W}x{H} ss{SS} smooth ...")

    def job(e):
        render_one(args.colormaps, e["name"], render_dir / f"{e['sanitized']}.png")
        return e["name"]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for name in ex.map(job, entries):
            print(f"  ok {name}")

    sheet = args.outdir / "contact_sheet.png"
    contact_sheet(entries, render_dir, sheet)
    print(f"contact sheet -> {sheet}")
    print(f"renders       -> {render_dir}")


if __name__ == "__main__":
    main()
