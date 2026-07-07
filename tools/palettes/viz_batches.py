"""Incremental palette-inspection sheets for every `dramatic_palettes` batch.

Render-free eyeball tool: for each results file (a bare array of authored
palettes) it densifies every palette and lays out a labeled sheet — one row per
palette — showing the densified LUT, the same LUT cycled ~3x (the banded read a
fractal actually samples), authored-stop tick marks colored by role, and the
name/skeleton/axes label. No `render-one`, no fractal pass — pure palette
structure, fast.

Incremental: rebuilds only results files with no `viz/<stem>.png` (or whose JSON
is newer than the PNG). Prints built vs skipped.

Usage:
    uv run python tools/palettes/viz_batches.py
    uv run python tools/palettes/viz_batches.py --force   # rebuild all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from densify_authored import densify_palette  # noqa: E402  reuse the OKLab densifier
from preview_render import font, lut_strip, sanitize  # noqa: E402  shared helpers

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "dramatic_palettes" / "results"
VIZ_DIR = ROOT / "dramatic_palettes" / "viz"

# Layout (px).
PAD = 12
LABEL_W = 380
STRIP_W = 1040
LUT_H = 42
TICK_H = 10
GAP = 5
CYC_H = 42
ROW_H = LUT_H + TICK_H + GAP + CYC_H
ROW_PAD = 14
HEADER_H = 40
CYCLES = 3
BG = (22, 22, 25)

# Authored-stop role -> tick color (legible on the LUT band).
ROLE_COLORS = {
    "ground": (120, 150, 255),
    "field": (90, 200, 230),
    "glow": (255, 240, 170),
    "anchor": (255, 150, 70),
    "accent": (255, 90, 90),
    "mid": (200, 200, 200),
}
ROLE_DEFAULT = (150, 150, 150)


def cycled_strip(stops: list, width: int, height: int, cycles: int) -> Image.Image:
    """Densified LUT sampled across [0, cycles) with wraparound — the banded read."""
    strip = Image.new("RGB", (width, height))
    px = strip.load()
    n = len(stops)
    for x in range(width):
        t = (x / max(1, width - 1)) * cycles
        frac = t - int(t)
        i = min(n - 1, int(frac * (n - 1)))
        _, rgb = stops[i]
        col = tuple(rgb)
        for y in range(height):
            px[x, y] = col
    return strip


def draw_ticks(draw: ImageDraw.ImageDraw, x0: int, y0: int, width: int,
               height: int, authored: list[dict]) -> None:
    """Colored vertical ticks at each authored stop pos, in a band of `height`."""
    draw.rectangle([x0, y0, x0 + width - 1, y0 + height - 1], fill=BG)
    for s in authored:
        p = float(s["pos"])
        x = x0 + round(p * (width - 1))
        col = ROLE_COLORS.get(s.get("role"), ROLE_DEFAULT)
        draw.line([(x, y0), (x, y0 + height - 1)], fill=col, width=2)


def sheet_for_batch(palettes: list[dict], out: Path) -> None:
    n = len(palettes)
    W = PAD + LABEL_W + STRIP_W + PAD
    H = HEADER_H + n * (ROW_H + ROW_PAD) + PAD
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    f_title = font(20)
    f_name = font(17)
    f_meta = font(13)
    f_leg = font(12)

    # Header: title + role legend.
    draw.text((PAD, 10), out.stem, fill=(240, 240, 240), font=f_title)
    lx = PAD + LABEL_W + 8
    for role, col in ROLE_COLORS.items():
        draw.rectangle([lx, 14, lx + 12, 26], fill=col)
        draw.text((lx + 16, 13), role, fill=(210, 210, 210), font=f_leg)
        lx += 16 + 8 + int(draw.textlength(role, font=f_leg)) + 16

    strip_x = PAD + LABEL_W
    for k, p in enumerate(palettes):
        stops = p.get("stops", [])
        dense = densify_palette(stops)  # authored widths / W=0.08 default
        y0 = HEADER_H + k * (ROW_H + ROW_PAD)

        # LUT strip + authored-stop ticks beneath it.
        img.paste(lut_strip(dense, STRIP_W, LUT_H), (strip_x, y0))
        draw_ticks(draw, strip_x, y0 + LUT_H, STRIP_W, TICK_H, stops)
        # cycled read beneath.
        img.paste(cycled_strip(dense, STRIP_W, CYC_H, CYCLES),
                  (strip_x, y0 + LUT_H + TICK_H + GAP))

        # Row label: name + skeleton + axes.
        axes = p.get("axes", {}) or {}
        draw.text((PAD, y0), p.get("name", "?"), fill=(240, 240, 240), font=f_name)
        meta1 = f"skeleton: {p.get('skeleton', '?')}"
        meta2 = (f"value_key {axes.get('value_key', '?')}  ·  "
                 f"complexity {axes.get('complexity', '?')}")
        temp = axes.get("temperature", "")
        draw.text((PAD, y0 + 24), meta1, fill=(190, 190, 190), font=f_meta)
        draw.text((PAD, y0 + 40), meta2, fill=(190, 190, 190), font=f_meta)
        if temp:
            # wrap the temperature blurb to the label column
            _wrap(draw, temp, PAD, y0 + 58, LABEL_W - 8, f_meta, (150, 150, 155))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _wrap(draw, text, x, y, max_w, f, fill):
    words = text.split()
    line, ly = "", y
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=f) > max_w and line:
            draw.text((x, ly), line, fill=fill, font=f)
            line, ly = w, ly + 15
        else:
            line = trial
    if line:
        draw.text((x, ly), line, fill=fill, font=f)


def is_stale(src: Path, dst: Path) -> bool:
    return (not dst.exists()) or src.stat().st_mtime > dst.stat().st_mtime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--viz", type=Path, default=VIZ_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild every batch")
    args = ap.parse_args()

    srcs = sorted(args.results.glob("*.json"))
    if not srcs:
        print(f"no results files under {args.results}")
        return

    built, skipped = 0, 0
    for src in srcs:
        dst = args.viz / f"{src.stem}.png"
        if not args.force and not is_stale(src, dst):
            print(f"  skip  {src.name} (up to date)")
            skipped += 1
            continue
        palettes = json.loads(src.read_text())
        sheet_for_batch(palettes, dst)
        print(f"  BUILT {src.name} -> {dst.relative_to(ROOT)}  ({len(palettes)} palettes)")
        built += 1

    print(f"done: {built} built, {skipped} skipped")


if __name__ == "__main__":
    main()
