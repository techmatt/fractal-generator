"""Per-batch fractal-render viz: each dramatic palette × n_cycles ∈ {2,3,4}.

Heavier eyeball tool than the strip sheet (`viz_batches.py`, which shows LUT +
cycled read only). This one shows the palette actually **coloring a fractal** at
each cycle count — but it does NOT re-render the fractal per cell. The smooth
scalar field at a fixed location is invariant to both palette and n_cycles
(n_cycles is a colorization-stage remap of the escape value), so we:

  1. dump the smooth field ONCE for whq3_000 at preview resolution (`render-one
     --dump-field`, beautiful/smooth source — the colormap-split source), persist
     it to disk, and reuse that single dump across every batch and every rebuild;
  2. `stretch_field` the cached field ONCE (the config-independent percentile
     prefix — `colormap.StretchedField`), reused across all 20×3 recolors;
  3. recolor each cell through the exact `colormap.py` field⊗colormap tail the
     production wallpaper emitter ships through (`apply_transform` -> n_cycles
     remap -> OKLab LUT gather -> interior fill -> linear-light downsample), so
     the viz matches production coloring. NOT `render-one --palette` per cell.

So a 20-palette × 3-cycle sheet is ~60 array recolors over one cached field, not
60 fractal renders. n_cycles here is applied as the palette-cycle multiplier at
recolor time (`t = (gray * n_cycles) mod 1`); we lift `render_candidate`'s
{1,2} n_cycles guard by inlining its numeric tail (steps 2–5) — same primitives,
same math, just a wider cycle range for inspection.

Incremental: rebuilds only results files whose `viz_render/<stem>.png` is missing
or older than the JSON. Prints built vs skipped.

Usage:
    uv run python tools/palettes/viz_render.py
    uv run python tools/palettes/viz_render.py --force        # rebuild all
    uv run python tools/palettes/viz_render.py --filter lanczos3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))            # colormap.py

from densify_authored import densify_palette        # noqa: E402  OKLCH densifier (honors per-stop width, W=0.08)
from preview_render import font, lut_strip, sanitize, LOC  # noqa: E402  shared helpers + whq3_000 location
import colormap as cm                                # noqa: E402  the field⊗colormap seam

RESULTS_DIR = ROOT / "dramatic_palettes" / "results"
VIZ_DIR = ROOT / "dramatic_palettes" / "viz_render"
FIELDS_DIR = ROOT / "out" / "palette_viz_render" / "fields"   # persisted field cache (disposable)
EXE = ROOT / "target" / "release" / "fractal-generator.exe"

# --- field spec (the cache key: location × resolution × render-mode) ---------
PREVIEW_W, PREVIEW_H, PREVIEW_SS = 768, 432, 2
RENDER_MODE = "smooth"                              # dump-field default source = beautiful smooth
CYCLES = (2, 3, 4)                                  # skip 1 (the strip sheet covers the single pass)

# --- layout (px) -------------------------------------------------------------
CELL_W = 384
CELL_H = round(CELL_W * PREVIEW_H / PREVIEW_W)      # 216
LABEL_W = 260
PAD = 10
HEADER_H = 34
ROW_PAD = 12
LUT_STRIP_H = 16
BG = (22, 22, 25)


def field_stem() -> str:
    return f"{LOC['name']}_{PREVIEW_W}x{PREVIEW_H}ss{PREVIEW_SS}_{RENDER_MODE}"


def ensure_field() -> cm.FieldData:
    """Dump (or reuse) the smooth field for whq3_000 at preview resolution.

    Keyed on (location, resolution, render-mode) and persisted to disk, so building
    ALL batches costs one field dump total — not one per batch. The field is a pure
    function of loc+geometry+maxiter, so a hit is byte-identical to a fresh dump."""
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    stem = field_stem()
    bin_path = FIELDS_DIR / f"{stem}.bin"
    json_path = FIELDS_DIR / f"{stem}.json"
    if not (bin_path.exists() and json_path.exists()):
        print(f"  field cache MISS -> dumping {stem} (render-one --dump-field, smooth) ...")
        import time
        cmd = [
            str(EXE), "render-one",
            "--cx", LOC["cx"], "--cy", LOC["cy"], "--fw", LOC["fw"],
            "--maxiter", str(LOC["maxiter"]),
            "--width", str(PREVIEW_W), "--height", str(PREVIEW_H),
            "--supersample", str(PREVIEW_SS),
            "--dump-field", str(bin_path),
        ]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError(f"dump-field failed for {stem}:\n{r.stderr[-800:]}")
        print(f"  field dumped in {dt:.1f}s -> {bin_path.relative_to(ROOT)}")
    else:
        print(f"  field cache HIT  {stem}")
    return cm.load_field(str(bin_path), str(json_path))


def recolor(field: cm.FieldData, prep: cm.StretchedField, lut: np.ndarray,
            n_cycles: int, filt: str) -> np.ndarray:
    """Recolor the cached field with a baked LUT + cycle count -> (H_out,W_out,3) sRGB8.

    Inlines `render_candidate` steps 2–5 (transform+gamma at defaults -> n_cycles
    remap -> OKLab LUT gather -> interior fill -> linear-light downsample) so we can
    pass n_cycles > 2. Numerically identical to the production emitter's tail for the
    range they share; the ONLY change is the wider cycle multiplier."""
    gray = cm.apply_transform(prep.x, "none", 1.0)          # defaults: no log premap, gamma 1
    t = np.mod(gray * n_cycles, 1.0)                        # n_cycles as the cycle multiplier
    linear = cm.lookup_linear(lut, t)                       # (H_sub,W_sub,3) linear RGB
    linear[~prep.valid] = 0.0                               # interior -> black (Rust default)
    return cm.downsample(linear, field.supersample, filt)   # linear-light AA -> sRGB8


def sheet_for_batch(palettes: list[dict], field: cm.FieldData,
                    prep: cm.StretchedField, out: Path, filt: str) -> None:
    n = len(palettes)
    W = PAD + LABEL_W + len(CYCLES) * (CELL_W + PAD) + PAD
    H = HEADER_H + n * (CELL_H + ROW_PAD) + PAD
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    f_title = font(19)
    f_name = font(16)
    f_meta = font(13)
    f_hdr = font(15)

    # Header: batch title + per-column cycle labels.
    draw.text((PAD, 9), f"{out.stem}   ·   whq3_000   ·   {filt}", fill=(240, 240, 240), font=f_title)
    for ci, cyc in enumerate(CYCLES):
        cx = PAD + LABEL_W + ci * (CELL_W + PAD) + CELL_W // 2
        label = f"n_cycles = {cyc}"
        w = draw.textlength(label, font=f_hdr)
        draw.text((cx - w / 2, 11), label, fill=(210, 210, 210), font=f_hdr)

    for k, p in enumerate(palettes):
        stops = p.get("stops", [])
        dense = densify_palette(stops)                       # honors per-stop width, W=0.08 default
        lut = cm.build_lut(dense, reverse=False, mirror=False)  # baked ONCE, reused across cycles
        y0 = HEADER_H + k * (CELL_H + ROW_PAD)

        # Row cells: one recolor per cycle count.
        for ci, cyc in enumerate(CYCLES):
            rgb = recolor(field, prep, lut, cyc, filt)       # (432,768,3) sRGB8
            cell = Image.fromarray(rgb).resize((CELL_W, CELL_H), Image.LANCZOS)
            x = PAD + LABEL_W + ci * (CELL_W + PAD)
            img.paste(cell, (x, y0))

        # Label column: name + skeleton + densified-LUT strip.
        draw.text((PAD, y0 + 2), p.get("name", "?"), fill=(240, 240, 240), font=f_name)
        draw.text((PAD, y0 + 24), f"skeleton: {p.get('skeleton', '?')}",
                  fill=(185, 185, 190), font=f_meta)
        # terse v3.1 dropped `cycle` (always cyclic) -> show the real metadata instead.
        vk = p.get("value_key", p.get("axes", {}).get("value_key", "?"))
        cx = p.get("complexity", p.get("axes", {}).get("complexity", "?"))
        draw.text((PAD, y0 + 42), f"value: {vk}   ·   cx: {cx}", fill=(150, 150, 155), font=f_meta)
        img.paste(lut_strip(dense, LABEL_W - PAD, LUT_STRIP_H), (PAD, y0 + CELL_H - LUT_STRIP_H))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def is_stale(src: Path, dst: Path) -> bool:
    return (not dst.exists()) or src.stat().st_mtime > dst.stat().st_mtime


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--viz", type=Path, default=VIZ_DIR)
    ap.add_argument("--filter", default="box", choices=["box", "mitchell", "lanczos3"],
                    help="downsample AA filter (default box; lanczos3 = production emit)")
    ap.add_argument("--force", action="store_true", help="rebuild every batch")
    args = ap.parse_args()

    srcs = sorted(args.results.glob("*.json"))
    if not srcs:
        print(f"no results files under {args.results}")
        return

    # Which batches need building? (Determine BEFORE the field dump so an all-skip
    # run never pays for the field.)
    todo = [s for s in srcs if args.force or is_stale(s, args.viz / f"{s.stem}.png")]
    if not todo:
        print(f"all {len(srcs)} batch(es) up to date; nothing to build")
        return

    field = ensure_field()                        # one dump total, shared across batches
    prep = cm.stretch_field(field)                # one percentile sort, shared across all recolors

    built = skipped = len(srcs) - len(todo)
    built = 0
    for src in srcs:
        dst = args.viz / f"{src.stem}.png"
        if src not in todo:
            print(f"  skip  {src.name} (up to date)")
            continue
        palettes = json.loads(src.read_text())
        sheet_for_batch(palettes, field, prep, dst, args.filter)
        print(f"  BUILT {src.name} -> {dst.relative_to(ROOT)}  "
              f"({len(palettes)} palettes x {len(CYCLES)} cycles)")
        built += 1

    print(f"done: {built} built, {skipped} skipped")


if __name__ == "__main__":
    main()
