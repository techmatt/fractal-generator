"""End-to-end: harvest -> import -> sample -> contact-sheet PNG (the deliverable).

Run:  python -m palette_lib.build_sheet
Outputs (under out/, the disposable tree):
  out/palette_contact_sheet.png            (location 1 x N sampled palettes)
  out/palette_contact_sheet_<loc>.png      (a second location, same palettes)
Plus the clean, committable colormap-derived library:
  data/palettes/clean_colormaps.json

Visual-first: the sheet is the artifact Matt selects from. No quality scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import coloring, field
from .classify import classify_palette, criterion_text
from .download import harvest_gnofract4d
from .importer import build_library
from .sampler import Sampler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
DATA = ROOT / "data" / "palettes"

# Contact-sheet geometry.
N_PALETTES = 30
COLS = 6
TILE_W, TILE_H = 240, 160
SS = 2                 # supersample, downsampled in linear light (engine parity)
CAPTION_H = 13
PAD = 4
DENSITY = 2.5          # a few gradient cycles across the escape range
OFFSET = 0.0


def render_tile(nu, interior, lut):
    """Colorize a (supersampled) field through a baked LUT; average in linear
    light, sRGB-encode. Returns uint8 (TILE_H, TILE_W, 3)."""
    lin = coloring.colorize(nu, lut, density=DENSITY, offset=OFFSET, interior_mask=interior)
    # Downsample SSxSS by averaging in linear light (matches shade_and_downsample).
    h, w, _ = lin.shape
    lin = lin.reshape(h // SS, SS, w // SS, SS, 3).mean(axis=(1, 3))
    srgb = coloring.linear_to_srgb(lin)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def build_sheet(nu, interior, palettes, title):
    """Assemble one location's grid: each tile = field through one palette,
    captioned with name + source."""
    rows = (len(palettes) + COLS - 1) // COLS
    cell_w = TILE_W + PAD
    cell_h = TILE_H + CAPTION_H + PAD
    W = COLS * cell_w + PAD
    H = rows * cell_h + PAD + 16
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    draw.text((PAD, 3), title, fill=(220, 220, 220), font=font)

    for i, pal in enumerate(palettes):
        lut = coloring.bake_lut(pal["stops"])
        tile = render_tile(nu, interior, lut)
        r, c = divmod(i, COLS)
        x = PAD + c * cell_w
        y = 16 + PAD + r * cell_h
        sheet.paste(Image.fromarray(tile), (x, y))
        label = f"{pal['source']}:{pal['name']}"
        if len(label) > 34:
            label = label[:33] + "…"
        draw.text((x + 2, y + TILE_H + 1), label, fill=(200, 200, 200), font=font)
    return sheet


def persist_clean_library(clean):
    """Classify every colormap-derived palette and persist the durable split.

    Folds the eye-validated quarantine (`classify.py`) into the harvest pipeline
    so a rebuild *reproduces* the hand-verified library instead of clobbering it:
      - survivors  -> data/palettes/clean_colormaps.json  (each carries the
        `cycle` + `mirror_needed` label inline; the Rust loaders ignore extra
        per-entry keys, so this is byte-shape-safe).
      - quarantined -> data/palettes/quarantined_colormaps.json  (moved, not
        deleted; carries its metrics for audit).
      - all -> data/palettes/quarantine_log.json  (criterion + per-palette
        metrics for the full set — the reproducible audit trail).

    Order follows the importer's survivor order, so the survivor file matches the
    prior hand pass and a re-run is byte-for-byte identical (idempotent).
    """
    metrics = [(p, classify_palette(p["stops"])) for p in clean]
    survivors = [(p, c) for p, c in metrics if not c["quarantine"]]
    quarantined = [(p, c) for p, c in metrics if c["quarantine"]]

    (DATA / "clean_colormaps.json").write_text(
        json.dumps(
            [{"name": p["name"], "source": p["source"], "stops": p["stops"],
              "cycle": c["cycle"], "mirror_needed": c["mirror_needed"]}
             for p, c in survivors],
            indent=1),
        encoding="utf-8",
    )
    (DATA / "quarantined_colormaps.json").write_text(
        json.dumps(
            [{"name": p["name"], "source": p["source"], "stops": p["stops"],
              "seam": c["seam"], "internal_max_step": c["internal_max_step"],
              "n_jump": c["n_jump"], "max_stop_step": c["max_stop_step"],
              "mean_stop_step": c["mean_stop_step"]}
             for p, c in quarantined],
            indent=1),
        encoding="utf-8",
    )
    n_cyc = sum(1 for _, c in survivors if c["cycle"] == "cyclic")
    log = {
        "criterion": criterion_text(),
        "counts": {
            "total": len(metrics),
            "survivors": len(survivors),
            "quarantined": len(quarantined),
            "sequential_survivors": len(survivors) - n_cyc,
            "cyclic_survivors": n_cyc,
        },
        "palettes": [
            {"name": p["name"], "source": p["source"],
             "seam": c["seam"], "internal_max_step": c["internal_max_step"],
             "n_jump": c["n_jump"], "max_stop_step": c["max_stop_step"],
             "mean_stop_step": c["mean_stop_step"],
             "quarantined": c["quarantine"], "cycle": c["cycle"]}
            for p, c in metrics
        ],
    }
    (DATA / "quarantine_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[clean] {len(survivors)} survivors "
          f"({n_cyc} cyclic / {len(survivors) - n_cyc} sequential), "
          f"{len(quarantined)} quarantined -> data/palettes/")
    return [p for p, _ in survivors], [p for p, _ in quarantined]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # 1. Harvest third-party collections (cached, gitignored) + 2. import/dedup.
    harvest = harvest_gnofract4d()
    library, report = build_library(harvest)

    # Committable clean artifact: colormap-derived palettes only (no harvested
    # colors), classified + split into survivors / quarantined (durable, see
    # persist_clean_library).
    clean = [p for p in library if p["source"] in ("matplotlib", "colorcet", "cmasher")]
    persist_clean_library(clean)

    # 3. Sampler: uniform-over-library distribution (the exposed knob).
    sampler = Sampler(library)
    palettes = sampler.draw(N_PALETTES, seed=0)
    print(f"[sample] drew {len(palettes)} palettes "
          f"({sum(p['source'] in ('ugr','map') for p in palettes)} harvested, "
          f"{sum(p['source'] not in ('ugr','map') for p in palettes)} colormap-derived)")

    # 4. Render the same sampled set through a couple of locations.
    outputs = []
    for li, (name, cre, cim, hw) in enumerate(field.LOCATIONS):
        nu, interior = field.smooth_field(cre, cim, hw, TILE_W * SS, TILE_H * SS)
        title = f"{name}  ({cre},{cim})  hw={hw}   {len(palettes)} palettes, uniform sample (seed 0)"
        sheet = build_sheet(nu, interior, palettes, title)
        path = OUT / ("palette_contact_sheet.png" if li == 0 else f"palette_contact_sheet_{name}.png")
        sheet.save(path)
        outputs.append(path)
        print(f"[sheet] {path.relative_to(ROOT)}  ({sheet.width}x{sheet.height})")

    return report, outputs


if __name__ == "__main__":
    main()
