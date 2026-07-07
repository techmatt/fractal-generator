"""Emission dry-run — real v2 gate + diversity selector on held-out humanq3 eval.

First honest end-to-end look at what automated v1 would *emit*: run the Stage-2d
emission selector over the **held-out humanq3 eval locations** (41 loc / 287
renders the wallpaper head never trained on), gated by the **real v2 marginal
`p_ge3`** and ranked by the continuous readout `score`. No rendering, no retrain —
reads existing crops + `data/wallpaper_head/v2/eval_scores.jsonl`.

For a sweep of `p_ge3` thresholds it reports, per threshold:
  - raw candidates passing the gate (before selection)
  - emitted count, cells filled, palette spread
  - **precision = share of emitted renders that are actually human tier >= 3**
and montages the emitted set, each tile annotated with human tier / p_ge3 / score.

    uv run python tools/wallpaper/emission_dryrun_v2gate.py

Precision is indicative, not exact — the emitted n is small.
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
BATCH = REPO / "data/wallpaper_corpus/batches/2026-07-05_wallpaper_humanq3_v1"
EVAL = REPO / "data/wallpaper_head/v2/eval_scores.jsonl"
CELL_CACHE = REPO / "out/wallpaper/emission_dryrun_colorcells.json"
OUT_DIR = REPO / "out/wallpaper/emission_dryrun_v2gate"

THRESHOLDS = [0.3, 0.5, 0.7]   # conservative->permissive; see p_ge3 dist in report

sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "emission_selector", REPO / "tools/wallpaper/emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)


# --------------------------------------------------------------------------- #
# build candidates (color cells cached; join palette from images.jsonl)       #
# --------------------------------------------------------------------------- #
def thumb_rgb(iid: str, w: int = 96) -> np.ndarray:
    with Image.open(BATCH / "crops" / f"{iid}.jpg") as im:
        im = im.convert("RGB")
        iw, ih = im.size
        im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
        return np.asarray(im)


def load_color_cells(iids: list[str], grid: es.ColorGrid) -> dict[str, int]:
    cache = json.loads(CELL_CACHE.read_text()) if CELL_CACHE.exists() else {}
    missing = [i for i in iids if i not in cache]
    if missing:
        print(f"[color] computing dominant Lab for {len(missing)} crops "
              f"({len(cache)} cached)")
        for iid in missing:
            lab = es.dominant_lab(thumb_rgb(iid), method="median")
            cache[iid] = grid.cell(lab)
        CELL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CELL_CACHE.write_text(json.dumps(cache))
    return {i: cache[i] for i in iids}


def build_candidates():
    ev = [json.loads(l) for l in EVAL.read_text().splitlines() if l.strip()]
    imgs = {json.loads(l)["image_id"]: json.loads(l)
            for l in (BATCH / "images.jsonl").read_text().splitlines() if l.strip()}
    grid = es.ColorGrid()
    cells = load_color_cells([r["image_id"] for r in ev], grid)
    cands = []
    for r in ev:
        iid = r["image_id"]
        cands.append(es.Candidate(
            location_id=r["loc"],
            palette_id=imgs[iid]["render"]["palette"],
            family=r["family"],
            fitness=float(r["score"]),
            color_cell=cells[iid],
            image_id=iid,
            meta={"label": int(r["label"]), "p_ge3": float(r["p_ge3"])},
        ))
    return cands, grid


# --------------------------------------------------------------------------- #
# sweep                                                                        #
# --------------------------------------------------------------------------- #
def run_threshold(cands, grid, thr: float):
    n_pass = sum(1 for c in cands if c.meta["p_ge3"] >= thr)
    res = es.select(cands, gate=lambda c: c.meta["p_ge3"] >= thr, grid=grid)
    picks = sorted(res.picks, key=lambda c: -c.fitness)
    labs = [c.meta["label"] for c in picks]
    tier_hist = Counter(labs)
    n_ge3 = sum(1 for l in labs if l >= 3)
    prec = (n_ge3 / len(picks)) if picks else float("nan")
    return {
        "thr": thr, "raw_pass": n_pass, "emitted": len(picks),
        "cells_filled": res.report["cells_filled"],
        "cells_reachable": res.report["cells_reachable"],
        "n_palettes": res.report["n_distinct_palettes_picked"],
        "palette_cap": res.palette_cap,
        "per_family": res.report["per_family_spread"],
        "precision_ge3": prec, "n_ge3": n_ge3,
        "tier_hist": {int(k): int(v) for k, v in sorted(tier_hist.items())},
        "picks": picks, "res": res,
    }


def main():
    cands, grid = build_candidates()
    p = np.array([c.meta["p_ge3"] for c in cands])
    lab = np.array([c.meta["label"] for c in cands])
    print(f"\n[pool] {len(cands)} renders / {len({c.location_id for c in cands})} "
          f"locations · base rate tier>=3 = {(lab>=3).mean():.3f} ({int((lab>=3).sum())})")
    print(f"[p_ge3] quantiles .5/.75/.9/.95/max = "
          f"{np.round(np.quantile(p,[.5,.75,.9,.95,1]),4).tolist()}")

    print(f"\n{'thr':>5} {'rawPass':>8} {'emit':>5} {'cells':>9} {'pals':>5} "
          f"{'prec>=3':>8} {'tiers(emitted)':>22}")
    print("-" * 70)
    results = []
    for thr in THRESHOLDS:
        r = run_threshold(cands, grid, thr)
        results.append(r)
        cells = f"{r['cells_filled']}/{r['cells_reachable']}"
        tiers = " ".join(f"T{k}:{v}" for k, v in r["tier_hist"].items())
        print(f"{thr:>5.2f} {r['raw_pass']:>8} {r['emitted']:>5} {cells:>9} "
              f"{r['n_palettes']:>5} {r['precision_ge3']:>8.3f}   {tiers:<22}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in results:
        render_montage(r)
    # persist the numeric report
    rep = [{k: v for k, v in r.items() if k not in ("picks", "res")} for r in results]
    (OUT_DIR / "sweep_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\n[report] wrote {OUT_DIR/'sweep_report.json'}")


# --------------------------------------------------------------------------- #
# montage                                                                      #
# --------------------------------------------------------------------------- #
def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


TIER_COLOR = {4: (56, 200, 90), 3: (46, 160, 67), 2: (219, 149, 30), 1: (200, 60, 55)}


def _short_palette(p: str, n: int = 30) -> str:
    p = p.replace("commons_", "").replace("Mandelbrot_", "M").replace("Julia_", "J")
    return p if len(p) <= n else p[:n - 1] + "…"


def render_montage(r):
    picks = r["picks"]
    thr = r["thr"]
    if not picks:
        print(f"[montage] thr {thr}: no picks, skipping")
        return
    COLS = min(8, len(picks))
    TW, TH = 300, 169
    BORDER, CAP, PAD = 5, 62, 10
    cell_w = TW + 2 * BORDER
    cell_h = TH + 2 * BORDER + CAP
    rows_n = (len(picks) + COLS - 1) // COLS
    HEADER = 96
    W = COLS * cell_w + (COLS + 1) * PAD
    H = HEADER + rows_n * cell_h + (rows_n + 1) * PAD

    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f_title, f_sub, f_cap, f_capb = _font(30), _font(18), _font(15), _font(16)

    d.text((PAD, 14), f"Emission dry-run · v2 gate p_ge3 ≥ {thr:g}",
           font=f_title, fill=(235, 235, 235))
    prec = r["precision_ge3"]
    sub = (f"{r['emitted']} emitted (of {r['raw_pass']} passing gate)  ·  "
           f"precision tier≥3 = {prec:.2f} ({r['n_ge3']}/{r['emitted']})  ·  "
           f"cells {r['cells_filled']}/{r['cells_reachable']}  ·  "
           f"{r['n_palettes']} palettes")
    d.text((PAD, 52), sub, font=f_sub, fill=(170, 170, 175))
    lx = W - 360
    for i, (t, txt) in enumerate([(4, "tier 4"), (3, "tier 3"), (2, "tier 2 (FP)")]):
        yy = 18 + i * 24
        d.rectangle([lx, yy, lx + 20, yy + 20], fill=TIER_COLOR[t])
        d.text((lx + 28, yy + 1), txt, font=f_cap, fill=(210, 210, 210))

    for i, c in enumerate(picks):
        rr, col = divmod(i, COLS)
        x = PAD + col * (cell_w + PAD)
        y = HEADER + PAD + rr * (cell_h + PAD)
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
        d.text((x + 4 + chip_w + 8, cy + 6),
               f"p{c.meta['p_ge3']:.2f}  s{c.fitness:.2f}", font=f_capb, fill=(230, 230, 230))
        d.text((x + 6, cy + 27), c.family, font=f_cap, fill=(150, 200, 235))
        d.text((x + 6, cy + 44), _short_palette(c.palette_id), font=f_cap, fill=(150, 150, 155))

    out = OUT_DIR / f"montage_p_ge3_{thr:g}.png"
    sheet.save(out)
    print(f"[montage] thr {thr:g}: wrote {out}  ({W}x{H}, {len(picks)} tiles)")


if __name__ == "__main__":
    main()
