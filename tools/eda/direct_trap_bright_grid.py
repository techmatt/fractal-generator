"""Direct-trap ring/lines bright-region grid: threshold x opacity
(prompts/direct-trap-ring-lines-bright-grid.md).

Per shape (ring, lines), a 3x3 sweep: cols = direct_threshold at the shape's
measured p75/p85/p95 closest-approach (FRACTAL_DT_STATS at the anchor), rows =
direct_opacity {0.15, 0.30, 0.45}. Carrier held constant: screen / bottom_up /
black start / twilight, ring trap_radius=1.0, maxiter 1500. Preview res, no 4K.
Writes one labeled contact sheet per shape + a lum_std/mean/black table.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "out" / "direct_shapes" / "bright"
SPECS = OUT / "specs"
OUT.mkdir(parents=True, exist_ok=True)
SPECS.mkdir(parents=True, exist_ok=True)

# Shared anchor (Julia) + render config.
C_RE, C_IM = "-0.07810228973371881", "-0.6514609012382414"
CX, CY = "0.4104135054546244", "0.20967482476903096"
FW = 0.5622541254857749
W, H, SS, MAXITER = 1280, 720, 2, 1500
PALETTE = "twilight"

# Per-shape p75/p85/p95 closest-approach (measured via FRACTAL_DT_STATS at anchor).
THRESHOLDS = {
    "ring": [("p75", 0.046323), ("p85", 0.059698), ("p95", 0.077429)],
    "lines": [("p75", 0.058047), ("p85", 0.080707), ("p95", 0.127504)],
}
OPACITIES = [0.15, 0.30, 0.45]


def spec_for(shape, thr, op):
    s = {
        "field": "direct_trap",
        "transform": "linear",
        "merge_mode": "screen",
        "merge_order": "bottom_up",
        "start_color": "black",
        "shape": shape,
        "direct_threshold": thr,
        "direct_opacity": op,
    }
    if shape == "ring":
        s["trap_radius"] = 1.0
    return s


def render(shape, pct, thr, op):
    spec = spec_for(shape, thr, op)
    spec_path = SPECS / f"{shape}_{pct}_op{op:g}.json"
    spec_path.write_text(json.dumps(spec))
    out = OUT / f"cell_{shape}_{pct}_op{op:g}.png"
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
    rgb = np.asarray(Image.open(p).convert("RGB"))
    im = rgb.astype(np.float64) / 255.0
    lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
    black = float(np.mean(np.all(rgb < 8, axis=-1)))
    return float(lum.std()), float(lum.mean()), black


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 7 * len(text) + 8, 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))


def build_sheet(shape):
    print(f"\n=== shape={shape} ===")
    print(f"{'thr':>5}{'op':>6}{'lum_std':>10}{'lum_mean':>10}{'black':>9}")
    thumbs = {}
    THUMB_W = 540
    cols = THRESHOLDS[shape]
    for op in OPACITIES:
        for pct, thr in cols:
            p = render(shape, pct, thr, op)
            ls, lm, bk = stats(p)
            print(f"{pct:>5}{op:>6g}{ls:>10.4f}{lm:>10.4f}{bk:>9.4f}")
            im = Image.open(p).convert("RGB")
            r = THUMB_W / im.width
            t = im.resize((THUMB_W, int(im.height * r)), Image.LANCZOS)
            label(t, f"{thr:.4f} / {op:g}  std={ls:.3f} blk={bk:.2f}")
            thumbs[(pct, op)] = t

    tw, th = next(iter(thumbs.values())).size
    pad, top, left = 6, 24, 76
    grid_w = left + len(cols) * (tw + pad) - pad
    grid_h = top + len(OPACITIES) * (th + pad) - pad
    mont = Image.new("RGB", (grid_w, grid_h), (16, 16, 16))
    d = ImageDraw.Draw(mont)
    for j, (pct, thr) in enumerate(cols):
        x = left + j * (tw + pad)
        d.text((x + 4, 7), f"{pct}  thr={thr:.4f}", fill=(255, 255, 255))
    for i, op in enumerate(OPACITIES):
        y = top + i * (th + pad)
        d.text((4, y + th // 2 - 6), f"op\n{op:g}", fill=(255, 255, 255))
        for j, (pct, thr) in enumerate(cols):
            x = left + j * (tw + pad)
            mont.paste(thumbs[(pct, op)], (x, y))
    dst = OUT / f"bright_grid_{shape}.png"
    mont.save(dst)
    print(f"wrote {dst}  ({mont.width}x{mont.height})")
    return dst


def main():
    for shape in ("ring", "lines"):
        build_sheet(shape)


if __name__ == "__main__":
    main()
