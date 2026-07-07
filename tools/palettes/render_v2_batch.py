"""Render the authored **v2 palette batch** through the production render path
for eyeballing. Visual-first, no metrics — a re-run of the soft-cliff harness on
new (v2-schema) input.

v2 schema notes: stops carry `segment: cliff` (alias for `hard`, realized as the
default soft-cliff ramp) with an optional per-stop `width`; a top-level `skeleton`
field tags each palette's value-shape. `skeleton` is metadata for *ordering* the
contact-sheet cells (grouped peak-early / peak-late / double-peak / cliff-in-mids
/ no-cliff / inverted-arc), not for rendering.

Reuses `densify_authored` (default soft-cliff W=0.08) and the `render` /
`lut_strip` / `font` / `sanitize` helpers from the existing preview harness.
Renders all 20 palettes on BOTH diagnostic locations (smooth whq3_000, busy) at
1024x576 ss2 smooth via `render-one --palette`, into two labeled contact sheets.

Outputs under out/palette_preview/v2-batch/:
  * densified.json                 — v2 batch densified at W=0.08
  * renders/                       — every individual frame
  * v2_batch_smooth.png            — 20 palettes on the smooth location
  * v2_batch_busy.png              — 20 palettes on the busy location
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import densify_authored as DA                              # noqa: E402
from preview_render import lut_strip, font, sanitize       # noqa: E402
from cliff_diag import render, W, H, SS, LOC_SMOOTH, LOC_BUSY  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "palette_v2.json"
OUTDIR = ROOT / "out/palette_preview/v2-batch"
RENDER_DIR = OUTDIR / "renders"
WORKERS = 4  # project cap
DEFAULT_W = 0.08

LOCS = {"smooth": LOC_SMOOTH, "busy": LOC_BUSY}
# contact-sheet cell ordering: skeleton groups, then batch order within each.
SKELETON_ORDER = ["peak-early", "peak-late", "double-peak",
                  "cliff-in-mids", "no-cliff", "inverted-arc"]


def order_by_skeleton(palettes: list[dict]) -> list[dict]:
    rank = {s: i for i, s in enumerate(SKELETON_ORDER)}
    idx = sorted(range(len(palettes)),
                 key=lambda i: (rank.get(palettes[i].get("skeleton"), len(rank)), i))
    return [palettes[i] for i in idx]


def build_densified(palettes: list[dict]) -> dict:
    """Densify the batch at W=0.08 -> {name: stops}, write densified.json."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lib = DA.densify_library(palettes, soft_cliff=DEFAULT_W)
    (OUTDIR / "densified.json").write_text(json.dumps(lib, indent=1))
    print(f"densified {len(lib)} palette(s) @ W={DEFAULT_W} -> {OUTDIR/'densified.json'}")
    return {p["name"]: p["stops"] for p in lib}


def render_all(names: list[str]) -> None:
    cm = OUTDIR / "densified.json"
    jobs = [(pal, ln) for ln in LOCS for pal in names]

    def job(j):
        pal, ln = j
        render(cm, pal, LOCS[ln], RENDER_DIR / f"{sanitize(pal)}__{ln}.png")
        return f"{pal} {ln}"

    print(f"rendering {len(jobs)} frames @ {W}x{H} ss{SS} smooth ...")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for msg in ex.map(job, jobs):
            print(f"  ok {msg}")


def contact_sheet(ordered: list[dict], luts: dict, ln: str, out: Path,
                  cols: int = 4, cell_w: int = 460) -> None:
    """rows of `cols` cells, ordered by skeleton group. Each cell: render +
    densified-LUT strip + skeleton-tagged name. A group-break band separates
    skeleton groups (drawn as a full-width header row when the group changes)."""
    loc = LOCS[ln]
    cell_h = round(cell_w * H / W)
    strip_h, label_h, pad = 16, 26, 8
    tile_w = cell_w + 2 * pad
    tile_h = cell_h + strip_h + label_h + 2 * pad
    head_h = 34
    band_h = 26  # per-group header band

    # lay out rows, inserting a group band whenever skeleton changes
    groups = []
    for p in ordered:
        sk = p.get("skeleton", "?")
        if not groups or groups[-1][0] != sk:
            groups.append((sk, []))
        groups[-1][1].append(p)

    # compute total height: for each group a band + ceil(n/cols) tile rows
    total_rows = 0
    for _sk, ps in groups:
        total_rows += (len(ps) + cols - 1) // cols
    sheet_h = head_h + sum(band_h for _ in groups) + total_rows * tile_h + pad
    sheet = Image.new("RGB", (cols * tile_w, sheet_h), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 9),
              f"v2 batch — {ln} location (fw {loc['fw']}, maxiter {loc['maxiter']}) "
              f"@ W={DEFAULT_W} soft-cliff, ordered by skeleton",
              fill=(235, 235, 235), font=font(17))
    fname, fband = font(16), font(16)

    y = head_h
    for sk, ps in groups:
        draw.rectangle([0, y, cols * tile_w, y + band_h], fill=(40, 40, 46))
        draw.text((10, y + 4), f"▍ {sk}  ({len(ps)})", fill=(210, 210, 220), font=fband)
        y += band_h
        for k, p in enumerate(ps):
            r, c = divmod(k, cols)
            x0 = c * tile_w + pad
            y0 = y + r * tile_h + pad
            img = Image.open(RENDER_DIR / f"{sanitize(p['name'])}__{ln}.png").resize(
                (cell_w, cell_h), Image.LANCZOS)
            sheet.paste(img, (x0, y0))
            sheet.paste(lut_strip(luts[p["name"]], cell_w, strip_h), (x0, y0 + cell_h))
            draw.text((x0 + 2, y0 + cell_h + strip_h + 4), p["name"],
                      fill=(235, 235, 235), font=fname)
        y += ((len(ps) + cols - 1) // cols) * tile_h
    sheet.save(out)
    print(f"contact sheet ({ln}) -> {out}")


def main() -> None:
    palettes = json.loads(BATCH.read_text())
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    ordered = order_by_skeleton(palettes)
    print(f"v2 batch: {len(palettes)} palettes, skeletons "
          f"{[p.get('skeleton') for p in ordered]}")

    luts = build_densified(palettes)
    render_all([p["name"] for p in palettes])
    for ln in LOCS:
        contact_sheet(ordered, luts, ln, OUTDIR / f"v2_batch_{ln}.png")
    print(f"renders -> {RENDER_DIR}")


if __name__ == "__main__":
    main()
