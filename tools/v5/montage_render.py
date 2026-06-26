#!/usr/bin/env python
"""Part 5.5 visual-first montage: render high/low v5-scored eval locations via
`render-one --julia` (and `render-one` for Mandelbrot) at wallpaper-ish quality,
then tile into a labeled sheet.

Joins data/classifier/v5/eval_scores_v5.jsonl (location_id, v5_score, label,
fractal_type) back to data/v5/manifest.jsonl (location_id = line index -> cx/cy/fw,
and c_re/c_im for Julia) so the render params are exact.

  uv run python tools/v5/montage_render.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
MANIFEST = ROOT / "data" / "v5" / "manifest.jsonl"
SCORES = ROOT / "data" / "classifier" / "v5" / "eval_scores_v5.jsonl"
OUT_DIR = ROOT / "data" / "classifier" / "v5" / "montages"
RENDER_DIR = OUT_DIR / "renders"
PALETTE = "twilight_shifted"   # deploy-canonical / labeling palette
W, H, SS = 640, 360, 2


def render(loc, tag):
    out = RENDER_DIR / f"{tag}.jpg"
    if out.exists():
        return out
    cmd = [str(BIN), "render-one", "--cx", loc["cx"], "--cy", loc["cy"], "--fw", loc["fw"],
           "--width", str(W), "--height", str(H), "--supersample", str(SS),
           "--palette", PALETTE, "--out", str(out)]
    if loc.get("fractal_type") == "julia":
        cmd += ["--julia", "--c", loc["c_re"], loc["c_im"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {tag}: {r.stderr[-200:]}")
        return None
    return out


def main():
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    scores = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    for s in scores:
        s["loc"] = manifest[s["location_id"]]

    jul = [s for s in scores if s["fractal_type"] == "julia"]
    man = [s for s in scores if s["fractal_type"] == "mandelbrot"]
    jul.sort(key=lambda s: s["v5_score"])
    man.sort(key=lambda s: s["v5_score"])

    # 4 rows: Julia high, Julia low, Mandelbrot high, Mandelbrot low (6 each)
    rows = [
        ("JULIA  HIGH", jul[-6:][::-1]),
        ("JULIA  LOW ", jul[:6]),
        ("MAND   HIGH", man[-6:][::-1]),
        ("MAND   LOW ", man[:6]),
    ]
    cols = 6
    tiles = []
    for label, group in rows:
        for k, s in enumerate(group):
            tag = f"{label.split()[0].lower()}_{label.split()[1].lower()}_{k}"
            p = render(s["loc"], tag)
            tiles.append((label, s, p))

    pad = 22
    sheet = Image.new("RGB", (cols * W, len(rows) * (H + pad)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for i, (label, s, p) in enumerate(tiles):
        cx, cy = i % cols, i // cols
        if p and p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (cx * W, cy * (H + pad)))
        txt = f"{label}  v5={s['v5_score']:.2f} lab={s['label']}"
        draw.text((cx * W + 4, cy * (H + pad) + H + 4), txt, fill=(235, 235, 235))
    out = OUT_DIR / "v5_render_montage.jpg"
    sheet.save(out, quality=92)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
