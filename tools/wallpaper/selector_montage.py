"""Contact-sheet montage of the emission-selector's bootstrap picks.

Visual-first eyeball of Stage-2d selection: re-runs `emission_selector.select`
on the 2026-07-05_wallpaper_bootstrap_v1 batch exactly as the selector test does
(gate = human tier >= 2, fitness = v1 wallpaper-head continuous readout), then
composes a PNG contact sheet of the picks sorted by v1 fitness desc.

Each tile is annotated with human tier / v1 fitness / family / palette_id, and
bordered by gate regime:
  GREEN  = human tier 3  -> passes the good-only gate (what automated v1 emits)
  ORANGE = human tier 2  -> fill-pick a tier>=2 gate let in

No rendering, no batch-builder changes. v1 inference for any crop missing from
`eval_scores.jsonl` runs on CPU (device forced), so this never touches the GPU.

    uv run python tools/wallpaper/selector_montage.py [out.png]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "data/wallpaper_corpus/batches/2026-07-05_wallpaper_bootstrap_v1"
LABELS = REPO / "labels/wallpaper_bootstrap_v1.json"
CACHED = REPO / "data/wallpaper_head/v1/eval_scores.jsonl"
CKPT = REPO / "data/wallpaper_head/v1/model_best.pt"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "out/wallpaper/selector_bootstrap_montage.png"

sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "emission_selector", REPO / "tools/wallpaper/emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)


# --------------------------------------------------------------------------- #
# rebuild candidates + run the selector (mirror of the Stage-2d test)         #
# --------------------------------------------------------------------------- #
def thumb_rgb(iid: str, w: int = 96) -> np.ndarray:
    with Image.open(BATCH / "crops" / f"{iid}.jpg") as im:
        im = im.convert("RGB")
        iw, ih = im.size
        im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
        return np.asarray(im)


def load_fitness(rows) -> dict[str, float]:
    fit = {}
    if CACHED.exists():
        for l in CACHED.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                fit[r["image_id"]] = float(r["score"])
    missing = [r["image_id"] for r in rows if r["image_id"] not in fit]
    print(f"[fitness] {len(fit)} cached, {len(missing)} to infer via v1 head (CPU)")
    if missing:
        from classifier.inference import load_scorer
        scorer = load_scorer(str(CKPT), device="cpu")
        paths = [str(BATCH / "crops" / f"{iid}.jpg") for iid in missing]
        for iid, s in zip(missing, scorer.score_paths(paths)):
            fit[iid] = float(s)
    return fit


def main():
    rows = [json.loads(l) for l in (BATCH / "images.jsonl").read_text().splitlines() if l.strip()]
    labels = json.loads(LABELS.read_text())
    fit = load_fitness(rows)

    grid = es.ColorGrid()
    cands = []
    for r in rows:
        iid = r["image_id"]
        lab = es.dominant_lab(thumb_rgb(iid), method="median")
        cands.append(es.Candidate(
            location_id=iid.rsplit("_", 1)[0],
            palette_id=r["render"]["palette"],
            family=r["provenance"]["family"],
            fitness=float(fit[iid]),
            color_cell=grid.cell(lab),
            image_id=iid,
            meta={"label": labels.get(iid)},
        ))

    res = es.select(cands, gate=lambda c: (c.meta.get("label") or 0) >= 2, grid=grid)
    picks = sorted(res.picks, key=lambda c: -c.fitness)
    tier_hist = Counter(c.meta["label"] for c in picks)
    n3 = tier_hist.get(3, 0)
    n2 = tier_hist.get(2, 0)
    print(f"[selector] {len(picks)} picks | tier-3 (good-only emit) {n3} | tier-2 (fill) {n2}")
    print(f"[selector] palette cap {res.palette_cap}, cells {res.report['cells_filled']}"
          f"/{res.report['cells_reachable']}, distinct palettes {res.report['n_distinct_palettes_picked']}")

    render_sheet(picks, n3, n2, res)


# --------------------------------------------------------------------------- #
# contact sheet                                                               #
# --------------------------------------------------------------------------- #
def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


TIER_COLOR = {3: (46, 160, 67), 2: (219, 149, 30)}   # green / orange


def _short_palette(p: str, n: int = 30) -> str:
    p = p.replace("commons_", "").replace("Mandelbrot_", "M").replace("Julia_", "J")
    return p if len(p) <= n else p[:n - 1] + "…"


def render_sheet(picks, n3, n2, res):
    COLS = 9
    TW, TH = 300, 169          # thumb (16:9)
    BORDER = 5                 # colored regime border
    CAP = 62                   # caption strip under each thumb
    PAD = 10
    cell_w = TW + 2 * BORDER
    cell_h = TH + 2 * BORDER + CAP
    rows_n = (len(picks) + COLS - 1) // COLS

    HEADER = 96
    W = COLS * cell_w + (COLS + 1) * PAD
    H = HEADER + rows_n * cell_h + (rows_n + 1) * PAD

    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f_title = _font(30)
    f_sub = _font(18)
    f_cap = _font(15)
    f_capb = _font(16)

    d.text((PAD, 14), "Emission-selector bootstrap picks", font=f_title, fill=(235, 235, 235))
    sub = (f"{len(picks)} picks (gate = human tier ≥ 2), sorted by v1 fitness desc  ·  "
           f"cells {res.report['cells_filled']}/{res.report['cells_reachable']}  ·  "
           f"{res.report['n_distinct_palettes_picked']} palettes, cap {res.palette_cap}")
    d.text((PAD, 52), sub, font=f_sub, fill=(170, 170, 175))
    lx = W - 470
    d.rectangle([lx, 20, lx + 22, 42], fill=TIER_COLOR[3])
    d.text((lx + 30, 22), f"tier 3 → good-only emits  ({n3})", font=f_sub, fill=(210, 210, 210))
    d.rectangle([lx, 52, lx + 22, 74], fill=TIER_COLOR[2])
    d.text((lx + 30, 54), f"tier 2 → fill-pick  ({n2})", font=f_sub, fill=(210, 210, 210))

    for i, c in enumerate(picks):
        r, col = divmod(i, COLS)
        x = PAD + col * (cell_w + PAD)
        y = HEADER + PAD + r * (cell_h + PAD)
        tier = c.meta["label"]
        bc = TIER_COLOR.get(tier, (120, 120, 120))

        d.rectangle([x, y, x + cell_w - 1, y + TH + 2 * BORDER - 1], fill=bc)
        with Image.open(BATCH / "crops" / f"{c.image_id}.jpg") as im:
            im = im.convert("RGB").resize((TW, TH), Image.LANCZOS)
        sheet.paste(im, (x + BORDER, y + BORDER))

        cy = y + TH + 2 * BORDER
        d.rectangle([x, cy, x + cell_w - 1, cy + CAP - 1], fill=(30, 30, 34))
        chip_w = 40
        d.rectangle([x + 4, cy + 5, x + 4 + chip_w, cy + 24], fill=bc)
        d.text((x + 11, cy + 6), f"T{tier}", font=f_capb, fill=(15, 15, 15))
        d.text((x + 4 + chip_w + 8, cy + 6), f"fit {c.fitness:.3f}", font=f_capb, fill=(230, 230, 230))
        d.text((x + 6, cy + 27), c.family, font=f_cap, fill=(150, 200, 235))
        d.text((x + 6, cy + 44), _short_palette(c.palette_id), font=f_cap, fill=(150, 150, 155))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(f"[montage] wrote {OUT}  ({W}x{H}, {len(picks)} tiles, {rows_n} rows)")


if __name__ == "__main__":
    main()
