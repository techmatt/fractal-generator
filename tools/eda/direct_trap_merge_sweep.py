"""Direct-orbit-traps merge-mode × threshold sweep (prompts/direct_trap_merge_sweep_prompt.md).

Rows = merge mode {normal, multiply, screen, overlay}; cols = direct_threshold
{0.1, 0.2, 0.4}. Everything else held at the faithful direct_trap default (cross
shape, distance color key, distance feather on, twilight, opacity 0.85, bottom_up
merge order, maxiter 1500). Renders 12 montage-res cells (1280x720 ss2) via
render-one @spec.json, prints a lum_std/mean/black-fraction table, and assembles a
labeled 4x3 contact sheet.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "out" / "uf_modes"
SPECS = OUT / "specs"
OUT.mkdir(parents=True, exist_ok=True)
SPECS.mkdir(parents=True, exist_ok=True)

# Shared render config (all cells identical except merge_mode / threshold).
C_RE, C_IM = "-0.07810228973371881", "-0.6514609012382414"
CX, CY = "0.4104135054546244", "0.20967482476903096"
FW = 0.5622541254857749
W, H, SS, MAXITER = 1280, 720, 2, 1500
PALETTE = "twilight"
OPACITY = 0.85

MODES = ["normal", "multiply", "screen", "overlay"]
THRESHOLDS = [0.1, 0.2, 0.4]


def spec_for(mode, thr):
    return {
        "field": "direct_trap",
        "merge_mode": mode,
        "merge_order": "bottom_up",
        "direct_threshold": thr,
        "direct_opacity": OPACITY,
    }


def cell_path(mode, thr):
    return OUT / f"cell_{mode}_t{thr:g}.png"


def render(mode, thr):
    spec = spec_for(mode, thr)
    spec_path = SPECS / f"{mode}_t{thr:g}.json"
    spec_path.write_text(json.dumps(spec))
    out = cell_path(mode, thr)
    cmd = [
        str(EXE), "render-one",
        "--julia", "--c", C_RE, C_IM,
        "--cx", CX, "--cy", CY, "--fw", str(FW),
        "--width", str(W), "--height", str(H),
        "--supersample", str(SS), "--maxiter", str(MAXITER),
        "--palette", PALETTE,
        "--coloring", f"@{spec_path}",
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    return out


def stats(p):
    im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
    # Rec.709 luma in sRGB space (good enough for a relative eyeball table).
    lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    black = float(np.mean(np.all(np.asarray(Image.open(p).convert("RGB")) < 8, axis=-1)))
    return float(lum.std()), float(lum.mean()), black


def label(img, text, fill=(255, 255, 255)):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=fill)


def main():
    print(f"{'mode':<9}{'thr':>5}{'lum_std':>10}{'lum_mean':>10}{'black':>9}")
    table = {}
    thumbs = {}
    THUMB_W = 540
    for mode in MODES:
        for thr in THRESHOLDS:
            p = render(mode, thr)
            ls, lm, bk = stats(p)
            table[(mode, thr)] = (ls, lm, bk)
            print(f"{mode:<9}{thr:>5g}{ls:>10.4f}{lm:>10.4f}{bk:>9.4f}")
            im = Image.open(p).convert("RGB")
            r = THUMB_W / im.width
            t = im.resize((THUMB_W, int(im.height * r)), Image.LANCZOS)
            label(t, f"{mode}  t={thr:g}  std={ls:.3f} blk={bk:.2f}")
            thumbs[(mode, thr)] = t

    # Assemble labeled 4x3 grid: rows = mode, cols = threshold.
    tw, th = next(iter(thumbs.values())).size
    pad, top, left = 6, 22, 70
    grid_w = left + len(THRESHOLDS) * (tw + pad) - pad
    grid_h = top + len(MODES) * (th + pad) - pad
    mont = Image.new("RGB", (grid_w, grid_h), (16, 16, 16))
    d = ImageDraw.Draw(mont)
    for j, thr in enumerate(THRESHOLDS):
        x = left + j * (tw + pad)
        d.text((x + 4, 6), f"threshold = {thr:g}", fill=(255, 255, 255))
    for i, mode in enumerate(MODES):
        y = top + i * (th + pad)
        d.text((4, y + th // 2 - 6), mode, fill=(255, 255, 255))
        for j, thr in enumerate(THRESHOLDS):
            x = left + j * (tw + pad)
            mont.paste(thumbs[(mode, thr)], (x, y))
    dst = OUT / "direct_trap_merge_sweep_montage.png"
    mont.save(dst)
    print(f"\nwrote {dst}  ({mont.width}x{mont.height})")


if __name__ == "__main__":
    main()
