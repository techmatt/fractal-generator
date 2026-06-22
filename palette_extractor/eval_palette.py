"""Eval harness for the single-palette extractor (see palette_extract.py).

For each input image: extract a palette, then emit
  - the palette as a long strip + the extracted .json
  - coverage % and a coverage visualization (which pixels the palette captures)
  - a standard Mandelbrot spiral rendered with the palette, at 0.5x / 1.0x / 1.5x
    cycle rate (to read the palette's frequency / rate-of-change)
  - whether closure was native or mirrored
all assembled into a self-navigating index.html.

The Mandelbrot render is a small self-contained numpy escape-time render — deliberately
independent of the Rust engine, so this tool stays pure-Python and lives beside the
extractor. It is an eval aid, not the production renderer.

Usage:
  python eval_palette.py <dir-or-images...> --out <outdir> [-n 2] [--seed 0]
  # sample N random images from a directory:
  python eval_palette.py "C:/Users/techm/Desktop/Wallpapers" --out eval_out -n 2
  # or pass explicit image paths:
  python eval_palette.py a.png b.jpg --out eval_out
"""
from __future__ import annotations
from pathlib import Path
import argparse
import base64
import html
import json
import random

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

from palette_extract import (
    extract_palette, srgb_to_oklab, oklab_to_srgb, resample_closed, PaletteResult,
)

# Canonical spiral palette-reading location (from the project handoff).
SPIRAL_CENTER = (-0.7453, 0.1127)
SPIRAL_HALF_W = 0.0050           # half-width in real units; tuned to frame the spiral
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Palette LUT (cyclic OKLab interpolation -> sRGB), matching the engine's model
# --------------------------------------------------------------------------- #

def build_lut(stops_lab: np.ndarray, n: int = 1024) -> np.ndarray:
    """Dense cyclic LUT (n,3) uint8 from uniform-arc OKLab stops."""
    lab = resample_closed(stops_lab, n)          # cyclic, uniform-arc in OKLab
    return oklab_to_srgb(lab)


# --------------------------------------------------------------------------- #
# Self-contained Mandelbrot spiral (smooth iteration + min-modulus trap)
# --------------------------------------------------------------------------- #

def render_mandelbrot(width: int, height: int, center=SPIRAL_CENTER,
                      half_w: float = SPIRAL_HALF_W, max_iter: int = 450):
    """Return (phase_field [h,w] in [0,1), where phase_field is a single smooth
    coordinate covering exterior (smooth iteration) and interior (min-modulus trap),
    so there is no dead-black interior)."""
    cx, cy = center
    half_h = half_w * height / width
    xs = np.linspace(cx - half_w, cx + half_w, width)
    ys = np.linspace(cy - half_h, cy + half_h, height)
    C = xs[None, :] + 1j * ys[:, None]
    Z = np.zeros_like(C)
    bailout = 256.0
    sm = np.zeros(C.shape, float)            # smooth iteration (exterior)
    minmod = np.full(C.shape, np.inf)        # min |z| over orbit (trap, interior)
    active = np.ones(C.shape, bool)

    for i in range(max_iter):
        Z = np.where(active, Z * Z + C, Z)
        az = np.abs(Z)
        minmod = np.where(active, np.minimum(minmod, az), minmod)
        escnow = active & (az > bailout)
        if escnow.any():
            azc = np.maximum(az[escnow], bailout * 1.0000001)
            sm[escnow] = i + 1.0 - np.log(np.log(azc)) / np.log(2.0)
        active &= ~escnow
        if not active.any():
            break

    escaped = ~active
    field = np.zeros(C.shape, float)

    # exterior: normalize smooth iteration over a robust percentile range -> [0,1)
    if escaped.any():
        e = sm[escaped]
        lo, hi = np.percentile(e, 2), np.percentile(e, 98)
        if hi <= lo:
            hi = lo + 1.0
        field[escaped] = np.clip((sm[escaped] - lo) / (hi - lo), 0, 1)

    # interior: normalize the min-modulus trap -> [0,1), so the inside is colored too
    if active.any():
        t = minmod[active]
        tlo, thi = np.percentile(t, 2), np.percentile(t, 98)
        if thi <= tlo:
            thi = tlo + 1.0
        field[active] = np.clip((minmod[active] - tlo) / (thi - tlo), 0, 1)

    return field


def colorize(field01: np.ndarray, lut: np.ndarray, rate: float,
             n_cycles: float = 3.0, offset: float = 0.0) -> np.ndarray:
    """Map the smooth field through the cyclic LUT at a given cycle rate.
    n_cycles sets how many palette traversals span the field at rate=1."""
    phase = (field01 * n_cycles * rate + offset) % 1.0
    idx = np.minimum((phase * len(lut)).astype(int), len(lut) - 1)
    return lut[idx].astype(np.uint8)


# --------------------------------------------------------------------------- #
# Visual artifacts
# --------------------------------------------------------------------------- #

def palette_strip(lut: np.ndarray, width: int = 1024, height: int = 64) -> Image.Image:
    idx = np.minimum((np.linspace(0, 1, width, endpoint=False) * len(lut)).astype(int),
                     len(lut) - 1)
    row = lut[idx]                                   # (width,3)
    return Image.fromarray(np.tile(row[None], (height, 1, 1)).astype(np.uint8))


def coverage_visual(img_path: Path, stops_lab: np.ndarray, eps: float,
                    max_dim: int = 900):
    """Return (PIL.Image, coverage_fraction_on_this_view).
    Covered pixels keep their true color; uncovered pixels are blanked to magenta,
    so the magenta regions are exactly what one palette cannot represent."""
    im = Image.open(img_path).convert("RGB")
    scale = min(1.0, max_dim / max(im.size))
    if scale < 1.0:
        im = im.resize((max(1, int(im.size[0] * scale)),
                        max(1, int(im.size[1] * scale))))
    arr = np.asarray(im)
    h, w, _ = arr.shape
    lab = srgb_to_oklab(arr.reshape(-1, 3).astype(float))
    curve = resample_closed(stops_lab, 2048)
    mind, _ = cKDTree(curve).query(lab, k=1)
    covered = (mind <= eps).reshape(h, w)
    out = arr.copy()
    out[~covered] = (255, 0, 255)
    return Image.fromarray(out), float(covered.mean())


# --------------------------------------------------------------------------- #
# Per-image eval + HTML
# --------------------------------------------------------------------------- #

def eval_one(img_path: Path, out_root: Path, *, eps: float, n_stops: int,
             mass_fraction: float, smooth_frac: float, render_wh=(760, 475),
             max_iter: int = 450) -> dict:
    name = img_path.stem
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)

    res: PaletteResult = extract_palette(
        img_path, n_stops=n_stops, mass_fraction=mass_fraction,
        smooth_frac=smooth_frac, coverage_eps=eps,
    )
    lut = build_lut(res.stops_lab, n=1024)

    # palette strip + json
    palette_strip(lut).save(d / "palette.png")
    (d / "palette.json").write_text(json.dumps(res.to_colormap(name)))

    # original (resized copy for the page) + coverage visual
    orig = Image.open(img_path).convert("RGB")
    osc = min(1.0, 900 / max(orig.size))
    if osc < 1.0:
        orig = orig.resize((int(orig.size[0] * osc), int(orig.size[1] * osc)))
    orig.save(d / "original.png")
    cov_img, cov_view = coverage_visual(img_path, res.stops_lab, eps)
    cov_img.save(d / "coverage.png")

    # mandelbrot at 3 cycle rates
    w, h = render_wh
    field = render_mandelbrot(w, h, max_iter=max_iter)
    rates = {"0.5x": 0.5, "1.0x": 1.0, "1.5x": 1.5}
    for label, r in rates.items():
        Image.fromarray(colorize(field, lut, r)).save(d / f"mandel_{label}.png")

    return {
        "name": name,
        "src": str(img_path),
        "closure": res.closure,
        "coverage": res.coverage,
        "coverage_view": cov_view,
        "max_step": res.max_step,
        "mean_step": res.mean_step,
        "endpoint_gap": res.endpoint_gap,
        "raw_spine_max_step": res.raw_spine_max_step,
        "n_ridge": res.n_ridge,
        "n_stops": n_stops,
    }


def write_index(results: list[dict], out_root: Path, eps: float) -> None:
    cards = []
    for r in results:
        n = r["name"]
        mirror_badge = (
            f'<span class="badge mirror">mirrored</span>' if r["closure"] == "mirrored"
            else f'<span class="badge native">native loop</span>'
        )
        cov_pct = f'{r["coverage"]*100:.1f}%'
        cards.append(f"""
    <section class="card">
      <div class="hd">
        <h2>{html.escape(n)}</h2>
        <div class="meta">
          {mirror_badge}
          <span class="badge">coverage <b>{cov_pct}</b></span>
          <span class="badge">max-step {r["max_step"]:.4f}</span>
          <span class="badge">mean-step {r["mean_step"]:.4f}</span>
          <span class="badge">endpoint-gap {r["endpoint_gap"]:.4f}</span>
          <span class="badge">ridge {r["n_ridge"]}</span>
        </div>
      </div>

      <div class="strip">
        <img src="{n}/palette.png" alt="palette strip">
      </div>

      <div class="row two">
        <figure><img src="{n}/original.png"><figcaption>original</figcaption></figure>
        <figure><img src="{n}/coverage.png">
          <figcaption>coverage — <span class="mag">magenta</span> = uncovered (ε={eps}); {r["coverage_view"]*100:.1f}% on this view</figcaption></figure>
      </div>

      <div class="row three">
        <figure><img src="{n}/mandel_0.5x.png"><figcaption>spiral · 0.5× rate</figcaption></figure>
        <figure><img src="{n}/mandel_1.0x.png"><figcaption>spiral · 1.0× rate</figcaption></figure>
        <figure><img src="{n}/mandel_1.5x.png"><figcaption>spiral · 1.5× rate</figcaption></figure>
      </div>
      <div class="src">{html.escape(r["src"])}</div>
    </section>""")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>palette extractor — eval</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0e0f13; color:#e7e7ea;
         font:14px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace; }}
  header {{ padding:22px 28px; border-bottom:1px solid #23252e; }}
  header h1 {{ margin:0; font-size:18px; font-weight:600; letter-spacing:.02em; }}
  header p {{ margin:6px 0 0; color:#8a8d99; }}
  main {{ padding:24px 28px; display:flex; flex-direction:column; gap:30px; }}
  .card {{ background:#15171d; border:1px solid #23252e; border-radius:12px;
           padding:18px 18px 22px; }}
  .hd {{ display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; margin-bottom:14px; }}
  .hd h2 {{ margin:0; font-size:16px; }}
  .meta {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .badge {{ background:#1d2029; border:1px solid #2c2f3a; color:#b8bbc7;
            padding:3px 9px; border-radius:999px; font-size:12px; }}
  .badge b {{ color:#fff; }}
  .badge.mirror {{ background:#3a2a12; border-color:#6b4a18; color:#f4c98a; }}
  .badge.native {{ background:#123524; border-color:#1f6b46; color:#8af4c2; }}
  .strip img {{ width:100%; height:48px; display:block; border-radius:6px;
                image-rendering:pixelated; }}
  .row {{ display:grid; gap:14px; margin-top:14px; }}
  .row.two {{ grid-template-columns:1fr 1fr; }}
  .row.three {{ grid-template-columns:1fr 1fr 1fr; }}
  figure {{ margin:0; }}
  figure img {{ width:100%; display:block; border-radius:8px; border:1px solid #23252e; }}
  figcaption {{ color:#8a8d99; font-size:12px; margin-top:6px; }}
  .mag {{ color:#ff5cff; }}
  .src {{ color:#5c5f6b; font-size:11px; margin-top:14px; word-break:break-all; }}
  @media (max-width:760px) {{ .row.two,.row.three {{ grid-template-columns:1fr; }} }}
</style></head>
<body>
<header>
  <h1>palette extractor — eval</h1>
  <p>{len(results)} image(s) · spiral render at {SPIRAL_CENTER} · coverage ε={eps}
     · palette = cyclic OKLab LUT from extracted stops</p>
</header>
<main>{''.join(cards)}
</main>
</body></html>"""
    (out_root / "index.html").write_text(doc, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def gather_images(inputs: list[Path], n: int, seed: int) -> list[Path]:
    if len(inputs) == 1 and inputs[0].is_dir():
        pool = sorted(p for p in inputs[0].rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not pool:
            raise SystemExit(f"no images found under {inputs[0]}")
        rng = random.Random(seed)
        return rng.sample(pool, min(n, len(pool)))
    return [p for p in inputs if p.suffix.lower() in IMG_EXTS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval harness for the palette extractor.")
    ap.add_argument("inputs", type=Path, nargs="+",
                    help="a directory (sampled) or explicit image paths")
    ap.add_argument("--out", type=Path, default=Path("eval_out"))
    ap.add_argument("-n", "--num", type=int, default=2, help="images to sample from a dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.05, help="coverage tolerance (OKLab)")
    ap.add_argument("--stops", type=int, default=256)
    ap.add_argument("--mass-fraction", type=float, default=0.90)
    ap.add_argument("--smooth-frac", type=float, default=0.012)
    ap.add_argument("--max-iter", type=int, default=450)
    args = ap.parse_args()

    imgs = gather_images(args.inputs, args.num, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {len(imgs)} image(s) -> {args.out}/index.html")

    results = []
    for p in imgs:
        print(f"  {p.name} ...", end="", flush=True)
        try:
            r = eval_one(p, args.out, eps=args.eps, n_stops=args.stops,
                         mass_fraction=args.mass_fraction, smooth_frac=args.smooth_frac,
                         max_iter=args.max_iter)
            print(f" closure={r['closure']} coverage={r['coverage']*100:.1f}%")
            results.append(r)
        except Exception as e:
            print(f" FAILED: {e}")
    if results:
        write_index(results, args.out, args.eps)
        print(f"wrote {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
