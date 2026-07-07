"""Soft-cliff adoption: width sweep + full 20-palette re-render at the default.

Builds on the cliff-jarring diagnostic. `densify_authored` now realizes `hard`
cliffs as a smoothstep ramp by default; this harness (1) sweeps the ramp width to
lock the default and (2) re-renders the whole authored set at the default width on
both the smooth and busy diagnostic locations. Visual-first, no metrics. Renders
through the production `render-one --palette` path.

Outputs under out/palette_preview/softcliff/:
  * dense_w{030,060,080,120}.json   — full set densified at each swept width
  * sweep.png                        — Oxblood/Amber x 2 locs x 4 widths
  * fullset_smooth.png / fullset_busy.png  — all 20 palettes at W=0.08, per loc
  * renders/                         — every individual frame
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import densify_authored as DA          # noqa: E402
from preview_render import lut_strip, font, sanitize  # noqa: E402
from cliff_diag import render, W, H, SS, LOC_SMOOTH, LOC_BUSY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AUTHORED = ROOT / "dramatic_palettes/dramatic-test.json"
OUTDIR = ROOT / "out/palette_preview/softcliff"
RENDER_DIR = OUTDIR / "renders"
WORKERS = 4  # project cap

WIDTHS = [0.03, 0.06, 0.08, 0.12]
DEFAULT_W = 0.08
SWEEP_PALETTES = ["Oxblood Reliquary", "Amber Vestment"]
LOCS = {"smooth": LOC_SMOOTH, "busy": LOC_BUSY}


def wtag(w: float) -> str:
    return f"{int(round(w * 1000)):03d}"  # 0.08 -> "080"


def build_densified(authored: list[dict]) -> dict:
    """Densify the full set at each swept width; return {wtag: {name: stops}}."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    luts = {}
    for w in WIDTHS:
        lib = DA.densify_library(authored, soft_cliff=w)
        (OUTDIR / f"dense_w{wtag(w)}.json").write_text(json.dumps(lib, indent=1))
        luts[wtag(w)] = {p["name"]: p["stops"] for p in lib}
    print(f"densified full set ({len(authored)}) at widths {WIDTHS}")
    return luts


def _run(jobs, fn):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for msg in ex.map(fn, jobs):
            print(f"  ok {msg}")


# ---- Render 1: width sweep --------------------------------------------------
def sweep(luts: dict) -> None:
    jobs = [(pal, ln, w) for pal in SWEEP_PALETTES for ln in LOCS for w in WIDTHS]

    def job(j):
        pal, ln, w = j
        cm = OUTDIR / f"dense_w{wtag(w)}.json"
        out = RENDER_DIR / f"sweep__{sanitize(pal)}__{ln}__w{wtag(w)}.png"
        render(cm, pal, LOCS[ln], out)
        return f"{pal} {ln} W={w}"

    print(f"sweep: rendering {len(jobs)} frames ...")
    _run(jobs, job)

    # rows = palette x location, cols = widths
    rows = [(pal, ln) for pal in SWEEP_PALETTES for ln in LOCS]
    cell_w, cell_h = 460, round(460 * H / W)
    strip_h, pad, head_h, label_w = 16, 10, 30, 210
    tile_w, tile_h = cell_w + pad, cell_h + strip_h + pad
    sheet = Image.new("RGB", (label_w + len(WIDTHS) * tile_w + pad,
                              head_h + len(rows) * tile_h + pad), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    fhead, fname = font(16), font(16)
    for c, w in enumerate(WIDTHS):
        x0 = label_w + c * tile_w
        draw.text((x0 + 2, 8), f"W = {w:g}" + ("  (default)" if w == DEFAULT_W else ""),
                  fill=(235, 235, 235), font=fhead)
    for r, (pal, ln) in enumerate(rows):
        y0 = head_h + r * tile_h
        draw.text((8, y0 + cell_h // 2 - 16), pal, fill=(235, 235, 235), font=fname)
        draw.text((8, y0 + cell_h // 2 + 4), f"[{ln} loc]", fill=(170, 170, 175), font=fname)
        for c, w in enumerate(WIDTHS):
            x0 = label_w + c * tile_w
            img = Image.open(RENDER_DIR / f"sweep__{sanitize(pal)}__{ln}__w{wtag(w)}.png").resize(
                (cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x0, y0))
            sheet.paste(lut_strip(luts[wtag(w)][pal], cell_w, strip_h), (x0, y0 + cell_h))
    out = OUTDIR / "sweep.png"
    sheet.save(out)
    print(f"sweep sheet -> {out}")


# ---- Render 2: full set at default width, per location ----------------------
def fullset(authored: list[dict], luts: dict) -> None:
    names = [p["name"] for p in authored]
    stops = luts[wtag(DEFAULT_W)]
    jobs = [(pal, ln) for ln in LOCS for pal in names]

    def job(j):
        pal, ln = j
        cm = OUTDIR / f"dense_w{wtag(DEFAULT_W)}.json"
        out = RENDER_DIR / f"full__{sanitize(pal)}__{ln}.png"
        render(cm, pal, LOCS[ln], out)
        return f"{pal} {ln}"

    print(f"fullset: rendering {len(jobs)} frames (W={DEFAULT_W}) ...")
    _run(jobs, job)

    cols = 4
    cell_w, cell_h = 460, round(460 * H / W)
    strip_h, label_h, pad = 16, 24, 8
    tile_w, tile_h = cell_w + 2 * pad, cell_h + strip_h + label_h + 2 * pad
    rows = (len(names) + cols - 1) // cols
    for ln in LOCS:
        loc = LOCS[ln]
        head = 34
        sheet = Image.new("RGB", (cols * tile_w, head + rows * tile_h), (24, 24, 26))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 9), f"Full set @ W={DEFAULT_W} (soft-cliff default) — {ln} location "
                           f"(fw {loc['fw']}, maxiter {loc['maxiter']})",
                  fill=(235, 235, 235), font=font(17))
        f = font(16)
        for k, pal in enumerate(names):
            r, c = divmod(k, cols)
            x0, y0 = c * tile_w + pad, head + r * tile_h + pad
            img = Image.open(RENDER_DIR / f"full__{sanitize(pal)}__{ln}.png").resize(
                (cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x0, y0))
            sheet.paste(lut_strip(stops[pal], cell_w, strip_h), (x0, y0 + cell_h))
            draw.text((x0 + 2, y0 + cell_h + strip_h + 4), pal, fill=(235, 235, 235), font=f)
        out = OUTDIR / f"fullset_{ln}.png"
        sheet.save(out)
        print(f"fullset sheet ({ln}) -> {out}")


def main() -> None:
    authored = json.loads(AUTHORED.read_text())
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    luts = build_densified(authored)
    sweep(luts)
    fullset(authored, luts)
    print(f"renders -> {RENDER_DIR}")


if __name__ == "__main__":
    main()
