"""Sanity contact sheet for the v2-filtered enrichment batch (pre-labeling).

Reads a label-corpus batch's images.jsonl + crops, and composes ONE PNG that
stratifies the enriched set by v2 score (top / upper-mid / near-cut) plus a row
of the random_eval reserve, each tile captioned with its v2 P(not-bad), est
class, selection role, and palette. The point: let Matt eyeball that the
enriched batch reads higher-quality BEFORE committing label budget. No labels,
no quality claim — just a stratified visual.

Run:
  uv run python tools/viz/enrich_sanity_sheet.py \
      --batch data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered \
      --out data/enrich/run5/sanity_sheet.png
"""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw

TW, TH = 320, 180          # thumbnail
CAP_H = 34                 # caption strip
PAD = 6
COLS = 8


def load_rows(batch):
    rows = []
    with open(os.path.join(batch, "images.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fscore(r):
    v = r.get("provenance", {}).get("filter_score")
    return v if isinstance(v, (int, float)) else None


def take_even(rows, n):
    """Evenly sample n rows across a sorted list (keeps the spread)."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered")
    ap.add_argument("--out", default="data/enrich/run5/sanity_sheet.png")
    ap.add_argument("--per-band", type=int, default=COLS)
    a = ap.parse_args()

    rows = load_rows(a.batch)
    crops = os.path.join(a.batch, "crops")
    enr = sorted([r for r in rows if r["provenance"].get("selection_role") == "enriched"],
                 key=lambda r: -(fscore(r) or 0))
    res = sorted([r for r in rows if r["provenance"].get("selection_role") == "random_eval"],
                 key=lambda r: -(fscore(r) or 0))

    n = a.per_band
    bands = []
    if enr:
        third = max(1, len(enr) // 3)
        bands.append(("ENRICHED · top", take_even(enr[:third], n)))
        bands.append(("ENRICHED · mid", take_even(enr[third:2 * third], n)))
        bands.append(("ENRICHED · near-cut", take_even(enr[2 * third:], n)))
    if res:
        bands.append(("RANDOM_EVAL (unbiased reserve)", take_even(res, n)))

    rows_n = len(bands)
    cell_w = TW + PAD
    cell_h = TH + CAP_H + PAD
    header_h = 22
    W = PAD + COLS * cell_w
    H = PAD + rows_n * (cell_h + header_h)
    sheet = Image.new("RGB", (W, H), (14, 15, 19))
    draw = ImageDraw.Draw(sheet)

    y = PAD
    for title, band in bands:
        draw.text((PAD, y + 4), title, fill=(180, 200, 180))
        y += header_h
        x = PAD
        for r in band:
            jpg = os.path.join(crops, r["image_id"] + ".jpg")
            if os.path.exists(jpg):
                with Image.open(jpg) as im:
                    sheet.paste(im.convert("RGB").resize((TW, TH)), (x, y))
            else:
                draw.rectangle([x, y, x + TW, y + TH], outline=(80, 40, 40))
                draw.text((x + 6, y + 6), "missing crop", fill=(200, 100, 100))
            pv = r["provenance"]
            fs = fscore(r)
            ec = pv.get("v2_est_class")
            cap_y = y + TH + 2
            col = (94, 192, 122) if (fs or 0) >= 0.5 else (224, 178, 74)
            draw.text((x + 2, cap_y),
                      f"P={fs:.3f}" if fs is not None else "P=?", fill=col)
            draw.text((x + 90, cap_y), f"est {ec}" if ec else "", fill=(170, 170, 170))
            pal = (r["render"]["palette"] or "")[:30]
            draw.text((x + 2, cap_y + 15), pal, fill=(130, 140, 150))
            x += cell_w
        y += cell_h

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out}  ({W}x{H}, {rows_n} bands x {COLS})")
    print(f"  enriched {len(enr)}  random_eval {len(res)}")


if __name__ == "__main__":
    main()
