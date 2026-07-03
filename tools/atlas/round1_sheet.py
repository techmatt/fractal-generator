#!/usr/bin/env python
"""Atlas round-1 acceptance — good-outcome contact sheets (the eyeball check behind the
diversity metric). Per arm: a grid of the GOOD outcomes' best reframed frames (the
embed tiles), sorted by k3, captioned with k3 + exploit/explore tag. This is what the
outcome-appearance-diversity number is measuring — a yield win that is one location
mined over and over is visible here as visual repetition.

Run AFTER round1_embed.py (needs the embed tiles under out/atlas/round1/embed_tiles/).

  uv run python tools/atlas/round1_sheet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "data" / "atlas" / "round1"
TILES = ROOT / "out" / "atlas" / "round1" / "embed_tiles"
OUT = ROOT / "out" / "atlas" / "round1"
ARMS = [("arm1", "current seeder"), ("arm2", "uniform-over-domain"), ("arm3", "atlas acquisition")]
GOOD = 1.0
TW, TH, PAD, LBL, COLS = 240, 135, 4, 16, 8


def build_arm_sheet(arm: str, name: str):
    z = np.load(D / f"{arm}_embed.npz", allow_pickle=False)
    k3 = z["reward_k3"].astype(float)
    wid = z["walk_id"]
    tag = z["tag"].astype(str)
    good = np.where(k3 >= GOOD)[0]
    good = good[np.argsort(k3[good])[::-1]]   # best first
    n = len(good)
    rows = (n + COLS - 1) // COLS if n else 1
    cell_w, cell_h = TW + 2 * PAD, TH + LBL + 2 * PAD
    W, Htop = COLS * cell_w, 30
    H = Htop + rows * cell_h
    sheet = Image.new("RGB", (W, H), (14, 15, 19))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"{arm} — {name}: {n} good outcomes (k3>={GOOD}), best-first"
                   f"  [caption: k3 · tag]", fill=(235, 235, 235))
    for k, i in enumerate(good):
        r, c = divmod(k, COLS)
        x, y = c * cell_w + PAD, Htop + r * cell_h + PAD
        tp = TILES / arm / f"walk_{int(wid[i]):04d}.jpg"
        if tp.exists():
            sheet.paste(Image.open(tp).convert("RGB").resize((TW, TH)), (x, y))
        d.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        tg = tag[i] if tag[i] else "-"
        col = (245, 215, 40) if tg == "explore" else (200, 200, 210)
        d.text((x + 3, y + TH + 2), f"{k3[i]:.2f} · {tg}", fill=col)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{arm}_good_sheet.png"
    sheet.save(out)
    return out, n


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for a, name in ARMS:
        if not (D / f"{a}_embed.npz").exists():
            print(f"skip {a}: no embed npz")
            continue
        out, n = build_arm_sheet(a, name)
        print(f"{a}: {n} good -> {out}")


if __name__ == "__main__":
    main()
