"""Phase A pilot: lock the reduced cache resolution + timing projection.

Selects ~30 label/source-diverse locations, renders three batches via
`v4-render-batch` — aliased ss1 @ 512x288, antialiased ss4 @ 512x288, and
(reference) production ss4 @ 1280x720 — and reports per-render throughput + the
full 152,124-render projection. Then builds a parity montage (5 locations x
{aliased-512, ss4-512, prod-720}), every cell downsampled to the consumed 384x224
via the deploy resize_core (bicubic stretch), so we can confirm:
  - aliased is genuinely crunchier than ss4 (intended),
  - ss4-reduced ~= production-720p after downsample (deploy parity),
  - all label-preserving.

  uv run python tools/v4/pilot.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "v4" / "manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "out" / "v4" / "pilot"
N_PER_LABEL = 10           # ~30 total across labels {1,2,3}
MONTAGE_N = 5
MONTAGE_PAL = "twilight_shifted"   # neutral, for a clean parity read
TARGET = (384, 224)        # consumed resolution (deploy resize_core stretch)

SCALES = [0.7, 1.0, 1.3]
SHIFT_FRAC = 0.4


def fmt(x): return repr(float(x))


def pick_locations():
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    by_label = {1: [], 2: [], 3: []}
    for i, r in enumerate(rows):
        by_label[r["label"]].append((i, r))
    picked = []
    for lab, lst in by_label.items():
        # spread across sources: round-robin by source, deterministic
        bysrc = {}
        for i, r in lst:
            bysrc.setdefault(r["source"], []).append((i, r))
        srcs = sorted(bysrc)
        out, k = [], 0
        while len(out) < min(N_PER_LABEL, len(lst)):
            s = srcs[k % len(srcs)]
            if bysrc[s]:
                out.append(bysrc[s].pop(0))
            k += 1
        picked.extend(out)
    return picked


def angle_schedule(palettes):
    n = len(palettes) * len(SCALES)
    a = {}
    for pi, p in enumerate(palettes):
        for si, sc in enumerate(SCALES):
            a[(p, sc)] = 2 * math.pi * (pi * len(SCALES) + si) / n
    return a


def emit_plan(path, picked, palettes, ang, which):
    """which: 'ss1' (full 36/loc), 'ss4' (6/loc center scale1)."""
    tag = which
    n = 0
    with open(path, "w") as f:
        for loc_id, r in picked:
            cx0, cy0, fw0 = float(r["cx"]), float(r["cy"]), float(r["fw"])
            d = OUT / tag / str(loc_id)
            if which == "ss1":
                for p in palettes:
                    for sc in SCALES:
                        for shift in ("center", "shifted"):
                            fw = sc * fw0
                            if shift == "shifted":
                                m = SHIFT_FRAC * fw
                                cx, cy = cx0 + m*math.cos(ang[(p, sc)]), cy0 + m*math.sin(ang[(p, sc)])
                            else:
                                cx, cy = cx0, cy0
                            out = d / f"{p}__s{sc:.1f}__sh{shift}__ss1.jpg"
                            f.write(json.dumps({"cx": fmt(cx), "cy": fmt(cy), "fw": fmt(fw),
                                                "palette": p, "ss": 1, "filter": "box",
                                                "out": str(out).replace("\\", "/")}) + "\n")
                            n += 1
            else:
                for p in palettes:
                    out = d / f"{p}__s1.0__shcenter__ss4.jpg"
                    f.write(json.dumps({"cx": fmt(cx0), "cy": fmt(cy0), "fw": fmt(fw0),
                                        "palette": p, "ss": 4, "filter": "lanczos3",
                                        "out": str(out).replace("\\", "/")}) + "\n")
                    n += 1
    return n


def run_batch(plan, w, h):
    t = time.time()
    r = subprocess.run([str(EXE), "v4-render-batch", "--plan", str(plan),
                        "--width", str(w), "--height", str(h),
                        "--log-every", "100000"],
                       capture_output=True, text=True)
    dt = time.time() - t
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise SystemExit(f"batch failed: {plan}")
    return dt


def montage(picked, palettes):
    locs = picked[:MONTAGE_N]
    cols = ["aliased-512", "ss4-512", "prod-720"]
    cw, ch = TARGET
    pad = 8
    W = len(cols) * cw + (len(cols) + 1) * pad
    H = len(locs) * ch + (len(locs) + 1) * pad
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    for ri, (loc_id, r) in enumerate(locs):
        srcs = [
            OUT / "ss1" / str(loc_id) / f"{MONTAGE_PAL}__s1.0__shcenter__ss1.jpg",
            OUT / "ss4" / str(loc_id) / f"{MONTAGE_PAL}__s1.0__shcenter__ss4.jpg",
            OUT / "ss4_720" / str(loc_id) / f"{MONTAGE_PAL}__s1.0__shcenter__ss4.jpg",
        ]
        for ci, sp in enumerate(srcs):
            im = Image.open(sp).convert("RGB").resize(TARGET, Image.BICUBIC)
            x = pad + ci * (cw + pad)
            y = pad + ri * (ch + pad)
            canvas.paste(im, (x, y))
    mp = OUT / "parity_montage.png"
    canvas.save(mp)
    return mp


def main():
    palettes = [r["name"] for r in json.loads(ROSTER.read_text())]
    ang = angle_schedule(palettes)
    picked = pick_locations()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"pilot locations: {len(picked)}  "
          f"labels={ {l: sum(1 for _,r in picked if r['label']==l) for l in (1,2,3)} }")

    p_ss1 = OUT / "plan_ss1.jsonl"
    p_ss4 = OUT / "plan_ss4.jsonl"
    n1 = emit_plan(p_ss1, picked, palettes, ang, "ss1")
    n4 = emit_plan(p_ss4, picked, palettes, ang, "ss4")
    # reuse ss4 plan rows but render at 720 to a different dir
    p_720 = OUT / "plan_ss4_720.jsonl"
    txt = p_ss4.read_text().replace("/pilot/ss4/", "/pilot/ss4_720/")
    p_720.write_text(txt)

    print(f"\nrendering {n1} ss1@512x288 ...")
    t_ss1 = run_batch(p_ss1, 512, 288)
    print(f"  {t_ss1:.1f}s  -> {1000*t_ss1/n1:.1f} ms/render  ({n1/t_ss1:.1f}/s)")

    print(f"rendering {n4} ss4@512x288 ...")
    t_ss4 = run_batch(p_ss4, 512, 288)
    print(f"  {t_ss4:.1f}s  -> {1000*t_ss4/n4:.1f} ms/render  ({n4/t_ss4:.1f}/s)")

    print(f"rendering {n4} ss4@1280x720 (production ref) ...")
    t_720 = run_batch(p_720, 1280, 720)
    print(f"  {t_720:.1f}s  -> {1000*t_720/n4:.1f} ms/render  ({n4/t_720:.1f}/s)")

    # full projection: 3622*36 ss1 + 3622*6 ss4 @512
    N_SS1, N_SS4 = 3622 * 36, 3622 * 6
    proj = N_SS1 / (n1 / t_ss1) + N_SS4 / (n4 / t_ss4)
    print(f"\n=== projection (full cache @512x288) ===")
    print(f"  ss1: {N_SS1} renders @ {n1/t_ss1:.1f}/s = {N_SS1/(n1/t_ss1)/3600:.2f} h")
    print(f"  ss4: {N_SS4} renders @ {n4/t_ss4:.1f}/s = {N_SS4/(n4/t_ss4)/3600:.2f} h")
    print(f"  TOTAL: {proj/3600:.2f} h  ({N_SS1+N_SS4} renders)")

    mp = montage(picked, palettes)
    print(f"\nparity montage -> {mp}")


if __name__ == "__main__":
    main()
