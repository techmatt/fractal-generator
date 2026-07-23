#!/usr/bin/env python
"""q4_sweep_validation — does a LABEL-FREE q4 composite window-sweep recover
Matt's hand-drawn (magenta-boxed) picks in its top-K?

Measurement only (NO net / training / production; no config or data/ changes).
See prompts/q4_sweep_validation.md and docs/findings/q4_sweep_validation.md.

Ground truth = Matt's magenta rectangles burned on subframe 0 of three deep-center
contact sheets (out/deep_centers/{preview_p58.png, ladder_mis/fw_1e_8.png,
ladder_p35/fw_8p07e_10.png}) — his q4 picks, the target to RECOVER, never fit against.

The composite feature family + weights are transferred VERBATIM from the julia q4 work
(tools/studies/q4_neighborhood_sweep.py: two_scale / compute_metrics / score_A). Weights
are hand-set q4 priors, NOT fit to Matt's boxes (that is the test set). Uncalibrated
recall is the headline.

Stages (idempotent):
  boxes     color-key magenta rectangles -> normalized frame windows  (out/.../boxes.json)
  fields    render-one --dump-field each frame at pool/ladder geometry (out/.../fields/)
  sweep     16:9 window sweep x 3 scales -> score_A -> NMS -> top-K + recall (recall.json)
  overlays  Matt boxes vs sweep top-K per frame (overlay_*.png, overlay_all.png)
  diagnose  per-box metrics vs top-K distribution -> named failing feature

Run:  uv run python -m tools.studies.q4_sweep_validation all
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.studies.q4_neighborhood_sweep import compute_metrics, score_A  # verbatim transfer

OUT = ROOT / "out" / "q4_sweep_val"
FIELDS = OUT / "fields"
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
PAD = 6  # sheet.rs grid gutter

# The three magenta-boxed frames: PNG + (cx, cy, fw, maxiter) from pool.jsonl /
# deep_center_sourcer ladders. Field aspect follows the sheet TILE aspect (below).
FRAMES = {
    "p58": dict(png="out/deep_centers/preview_p58.png",
                cx="-0.74903659502693622988762003916537468",
                cy="0.12465575243193049482586137195444164",
                fw="1.587612e-10", maxiter=14699, W=768, H=432),   # 16:9 tile
    "mis": dict(png="out/deep_centers/ladder_mis/fw_1e_8.png",
                cx="0.32187663879025893205691900369603022",
                cy="0.033260752306371290736322529793821926",
                fw="1e-8", maxiter=13500, W=768, H=432),           # 16:9 tile
    "p35": dict(png="out/deep_centers/ladder_p35/fw_8p07e_10.png",
                cx="-0.74977483272365342795786040375088960",
                cy="0.10761724352653678278696798751738616",
                fw="8.069624e-10", maxiter=13640, W=768, H=512),   # 3:2 tile
}

# Sweep scales (window width as fraction of frame width), 16:9 windows in square
# field pixels. Span Matt's measured box widths (0.057..0.233). The spec's >=1/6
# (0.167) floor is deliberately relaxed at the low end: 5 of 7 of Matt's OWN boxes
# are <0.10w, so honoring the floor makes IoU recovery structurally impossible.
SCALES = [0.10, 0.16, 0.25]
STRIDE_FRAC = 0.34
TOPKS = [5, 10, 20]
KEYS = ["interior_frac", "mid_detail_frac", "flat_frac", "busy_frac",
        "distributed_interior", "high_struct_frac"]


# --------------------------------------------------------------------------- #
def magenta_mask(a):
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return (r > 180) & (b > 120) & (g < 120) & (r - g > 80) & (b - g > 40)


def stage_boxes():
    out = {}
    for name, fr in FRAMES.items():
        a = np.asarray(Image.open(ROOT / fr["png"]).convert("RGB"))
        H, W = a.shape[:2]
        tw, th = (W - 3 * PAD) // 2, (H - 3 * PAD) // 2      # tile 0 dims
        x0, y0 = PAD, PAD                                    # tile 0 origin
        lab, n = label(magenta_mask(a))
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if xs.size < 40:
                continue
            bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if not (x0 <= cx < x0 + tw and y0 <= cy < y0 + th):
                continue
            boxes.append(dict(u=round((bx0 - x0) / tw, 4), v=round((by0 - y0) / th, 4),
                              w=round((bx1 - bx0) / tw, 4), h=round((by1 - by0) / th, 4)))
        boxes.sort(key=lambda b: (b["v"], b["u"]))
        out[name] = dict(img_wh=[W, H], tile_wh=[tw, th], boxes=boxes)
        print(f"{name}: tile {tw}x{th}  {len(boxes)} boxes  "
              f"w-range [{min(b['w'] for b in boxes):.3f},{max(b['w'] for b in boxes):.3f}]")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "boxes.json", "w"), indent=2)


def stage_fields():
    FIELDS.mkdir(parents=True, exist_ok=True)
    for name, fr in FRAMES.items():
        b = FIELDS / f"{name}.bin"
        if b.exists():
            print(f"{name}: cached")
            continue
        # DEFAULT beautiful source: uses the real render path so DEEP frames (p58/p35)
        # auto-select the perturbation backend. f64 source would be garbage past ~1e-13.
        cmd = [str(EXE), "render-one", "--cx", fr["cx"], "--cy", fr["cy"], "--fw", fr["fw"],
               "--family", "mandelbrot", "--maxiter", str(fr["maxiter"]),
               "--width", str(fr["W"]), "--height", str(fr["H"]), "--supersample", "1",
               "--dump-field", str(b)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        print(f"{name}: dumped {fr['W']}x{fr['H']}")


def load_field(name):
    meta = json.load(open(FIELDS / f"{name}.json"))
    w, h = int(meta["width"]), int(meta["height"])
    a = np.frombuffer((FIELDS / f"{name}.bin").read_bytes(), dtype="<f4").reshape(h, w)
    return a.astype(np.float64), w, h


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def sweep_frame(name):
    field, W, H = load_field(name)
    cands = []
    for s in SCALES:
        Wp = max(8, int(round(s * W)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= H or Wp >= W:
            continue
        st = max(4, int(round(STRIDE_FRAC * Wp)))
        for y in range(0, H - Hp + 1, st):
            for x in range(0, W - Wp + 1, st):
                m = compute_metrics(field[y:y + Hp, x:x + Wp])
                u, v, uw, vh = x / W, y / H, Wp / W, Hp / H
                cands.append(dict(score=float(score_A(m)), box=(u, v, uw, vh),
                                  cx=u + uw / 2, cy=v + vh / 2, scale=s, m=m))
    cands.sort(key=lambda c: c["score"], reverse=True)
    kept = []                                     # greedy NMS, IoU>0.30
    for c in cands:
        if all(iou(c["box"], k["box"]) <= 0.30 for k in kept):
            kept.append(c)
    return kept


def cin(c, gt):
    return gt["u"] <= c["cx"] <= gt["u"] + gt["w"] and gt["v"] <= c["cy"] <= gt["v"] + gt["h"]


def stage_sweep():
    boxes = json.load(open(OUT / "boxes.json"))
    rows, report = [], {}
    for name in FRAMES:
        kept = sweep_frame(name)
        report[name] = dict(n_cands=len(kept),
                            topk=[dict(score=round(c["score"], 4),
                                       box=[round(x, 4) for x in c["box"]], scale=c["scale"])
                                  for c in kept[:max(TOPKS)]])
        for j, gt in enumerate(boxes[name]["boxes"]):
            gtb = (gt["u"], gt["v"], gt["w"], gt["h"])
            row = dict(frame=name, box=j, gt=gt)
            for K in TOPKS:
                topk = kept[:K]
                ch = [i for i, c in enumerate(topk) if cin(c, gt)]
                ih = [iou(c["box"], gtb) for c in topk if iou(c["box"], gtb) >= 0.30]
                row[f"cin@{K}"] = ch[0] if ch else None
                row[f"iou@{K}"] = round(max(ih), 3) if ih else None
            bi = max((iou(c["box"], gtb), i) for i, c in enumerate(kept))
            row["best_iou"], row["best_iou_rank"] = round(bi[0], 3), bi[1]
            row["best_cin_rank"] = next((i for i, c in enumerate(kept) if cin(c, gt)), None)
            rows.append(row)
    report["per_box"] = rows
    n = len(rows)
    report["pooled_recall"] = {
        f"K={K}": dict(
            center_in_box=f"{sum(1 for r in rows if r[f'cin@{K}'] is not None)}/{n}",
            iou30=f"{sum(1 for r in rows if r[f'iou@{K}'] is not None)}/{n}")
        for K in TOPKS}
    json.dump(report, open(OUT / "recall.json", "w"), indent=2)

    print(f"\n=== recall (n={n} boxes) scales={SCALES} NMS IoU>0.30 ===")
    print(f"{'fr':4s}{'bx':3s}{'gt_w':6s}| " + "".join(f"cin@{K:<3d}iou@{K:<3d}" for K in TOPKS) +
          "| best_iou(rank) cin_rank")
    for r in rows:
        cells = "".join(f"{('r'+str(r[f'cin@{K}'])) if r[f'cin@{K}'] is not None else '-':>6s}"
                        f"{('%.2f'%r[f'iou@{K}']) if r[f'iou@{K}'] is not None else '-':>6s}" for K in TOPKS)
        print(f"{r['frame']:4s}{r['box']:<3d}{r['gt']['w']:<6.3f}| {cells}| "
              f"{r['best_iou']:.2f} (r{r['best_iou_rank']})  "
              f"{'r'+str(r['best_cin_rank']) if r['best_cin_rank'] is not None else 'MISS'}")
    print("\nPOOLED:")
    for K in TOPKS:
        p = report["pooled_recall"][f"K={K}"]
        print(f"  @{K}: center-in-box {p['center_in_box']}   IoU>=0.30 {p['iou30']}")


def _tilepx(box, tw, th):
    u, v, w, h = box
    return [PAD + u * tw, PAD + v * th, PAD + (u + w) * tw, PAD + (v + h) * th]


def stage_overlays():
    boxes = json.load(open(OUT / "boxes.json"))
    panels = []
    for name, fr in FRAMES.items():
        img = Image.open(ROOT / fr["png"]).convert("RGB")
        W, H = img.size
        tw, th = (W - 3 * PAD) // 2, (H - 3 * PAD) // 2
        crop = img.crop((0, 0, 2 * PAD + tw, 2 * PAD + th))
        d = ImageDraw.Draw(crop)
        kept = sweep_frame(name)
        for i, c in enumerate(kept[:10]):
            x0, y0, x1, y1 = _tilepx(c["box"], tw, th)
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 255), width=2)
            d.text((x0 + 2, y0 + 1), str(i), fill=(0, 255, 255))
        for gt in boxes[name]["boxes"]:
            gtb = (gt["u"], gt["v"], gt["w"], gt["h"])
            best = max(kept, key=lambda c: iou(c["box"], gtb))
            d.rectangle(_tilepx(best["box"], tw, th), outline=(255, 240, 0), width=2)
        crop.save(OUT / f"overlay_{name}.png")
        panels.append((name, crop))
    w = 900
    ims = [(n, im.resize((w, int(im.height * w / im.width)))) for n, im in panels]
    Hc = sum(im.height for _, im in ims) + 26 * len(ims) + 8
    canvas = Image.new("RGB", (w, Hc), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    y = 4
    for n, im in ims:
        d.text((6, y), f"{n}   magenta=Matt  cyan=sweep top-10  yellow=best-overlap-per-box",
               fill=(235, 235, 235))
        y += 22
        canvas.paste(im, (0, y))
        y += im.height + 4
    canvas.save(OUT / "overlay_all.png")
    print("wrote overlay_*.png + overlay_all.png")


def stage_diagnose():
    boxes = json.load(open(OUT / "boxes.json"))
    print(f"\n{'':16s}" + "".join(f"{k[:9]:>10s}" for k in KEYS) + "   score_A")
    for name in FRAMES:
        field, W, H = load_field(name)
        top = sweep_frame(name)[:10]
        tmet = {k: np.array([c["m"][k] for c in top]) for k in KEYS}
        tlo = {k: np.percentile(tmet[k], 10) for k in KEYS}
        thi = {k: np.percentile(tmet[k], 90) for k in KEYS}
        tsc = np.array([c["score"] for c in top])
        print(f"\n--- {name} ---")
        print(f"{'TOP10 median':16s}" + "".join(f"{np.median(tmet[k]):10.3f}" for k in KEYS) +
              f"   {np.median(tsc):.3f}")
        for j, gt in enumerate(boxes[name]["boxes"]):
            x0, y0 = int(gt["u"] * W), int(gt["v"] * H)
            x1, y1 = int((gt["u"] + gt["w"]) * W), int((gt["v"] + gt["h"]) * H)
            m = compute_metrics(field[max(0, y0):y1, max(0, x0):x1])
            flags = [f"{k}-LO" if m[k] < tlo[k] else f"{k}-HI" for k in KEYS
                     if m[k] < tlo[k] or m[k] > thi[k]]
            print(f"{'box'+str(j)+' w'+format(gt['w'],'.2f'):16s}" +
                  "".join(f"{m[k]:10.3f}" for k in KEYS) +
                  f"   {score_A(m):.3f}   OUT: {', '.join(flags) or '(in range)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["boxes", "fields", "sweep", "overlays", "diagnose", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("boxes", "all"):
        stage_boxes()
    if args.stage in ("fields", "all"):
        stage_fields()
    if args.stage in ("sweep", "all"):
        stage_sweep()
    if args.stage in ("overlays", "all"):
        stage_overlays()
    if args.stage in ("diagnose", "all"):
        stage_diagnose()
