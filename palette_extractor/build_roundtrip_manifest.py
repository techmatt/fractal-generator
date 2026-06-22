"""
build_roundtrip_manifest.py

For N randomly chosen palettes from clean_colormaps.json:
  1. Render the canonical spiral location via the Rust engine
  2. Run palette_extract.py on the rendered image
  3. Sample the LUT curve uniformly (ground-truth palette)
  4. Sample rendered pixel colors (what the extractor actually saw)
  5. Emit all three point clouds + metadata to data/palette_roundtrip/<name>/

Produces data/palette_roundtrip/manifest.json consumed by
tools/viz/palette_roundtrip.html.

Usage:
    python build_roundtrip_manifest.py [--n 10] [--seed 42] [--width 960] [--height 640]

Requirements:
    - Rust engine built: cargo build --release
    - palette_extractor/palette_extract.py importable
    - numpy, Pillow
"""

from __future__ import annotations
import argparse
import json
import math
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).parent
RUST_BIN = REPO / "target" / "release" / "fractal-generator.exe"
PALETTES_JSON = REPO / "data" / "palettes" / "clean_colormaps.json"
EXTRACTOR = REPO / "palette_extractor" / "palette_extract.py"
OUT_ROOT = REPO / "data" / "palette_roundtrip"

# Canonical spiral location (validated as the best contact-sheet location)
SPIRAL_CENTER_X = -0.7453
SPIRAL_CENTER_Y =  0.1127
FRAME_WIDTH     =  0.004   # shallow zoom, structure-rich

LUT_SAMPLES = 256          # uniform samples along the colormap curve
PIXEL_SAMPLES = 4096       # random pixels sampled from the render


def oklab_to_rgb_u8(L: float, a: float, b: float) -> tuple[int,int,int]:
    """Ottosson OKLab → sRGB u8."""
    l_ = L + 0.3963377774*a + 0.2158037573*b
    m_ = L - 0.1055613458*a - 0.0638541728*b
    s_ = L - 0.0894841775*a - 1.2914855480*b
    l3 = l_**3; m3 = m_**3; s3 = s_**3
    r =  4.0767416621*l3 - 3.3077115913*m3 + 0.2309699292*s3
    g = -1.2684380046*l3 + 2.6097574011*m3 - 0.3413193965*s3
    bv= -0.0041960863*l3 - 0.7034186147*m3 + 1.7076147010*s3
    def gc(v):
        v = max(0.0, min(1.0, v))
        return 12.92*v if v <= 0.0031308 else 1.055*v**(1/2.4)-0.055
    return (round(gc(r)*255), round(gc(g)*255), round(gc(bv)*255))


def rgb_u8_to_oklab(r: int, g: int, b: int) -> tuple[float,float,float]:
    """sRGB u8 → OKLab."""
    def ungamma(v):
        v = v / 255.0
        return v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4
    rl, gl, bl = ungamma(r), ungamma(g), ungamma(b)
    l_ = (0.4122214708*rl + 0.5363325363*gl + 0.0514459929*bl)**(1/3)
    m_ = (0.2119034982*rl + 0.6806995451*gl + 0.1073969566*bl)**(1/3)
    s_ = (0.0883024619*rl + 0.2817188376*gl + 0.6299787005*bl)**(1/3)
    L  =  0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a  =  1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    bv = -0.0259040371*l_ + 0.4120456635*m_ - 0.8827513165*s_
    return (L, a, bv)


def load_palettes() -> dict:
    with open(PALETTES_JSON) as f:
        return json.load(f)


def sample_lut(stops: list[dict], n: int = LUT_SAMPLES) -> list[dict]:
    """
    Sample n evenly-spaced points along the palette curve in OKLab.
    stops: list of {t, L, a, b} (already in OKLab from clean_colormaps.json).
    Returns list of {t, L, a, b}.
    """
    # sort by t
    stops = sorted(stops, key=lambda s: s["t"])
    if len(stops) < 2:
        return stops

    def lerp_stop(t: float) -> tuple[float, float, float]:
        # find bracketing stops
        for i in range(len(stops)-1):
            s0, s1 = stops[i], stops[i+1]
            if s0["t"] <= t <= s1["t"]:
                span = s1["t"] - s0["t"]
                if span < 1e-9:
                    return s0["L"], s0["a"], s0["b"]
                frac = (t - s0["t"]) / span
                return (
                    s0["L"] + frac*(s1["L"]-s0["L"]),
                    s0["a"] + frac*(s1["a"]-s0["a"]),
                    s0["b"] + frac*(s1["b"]-s0["b"]),
                )
        # clamp
        s = stops[-1]
        return s["L"], s["a"], s["b"]

    result = []
    for i in range(n):
        t = i / (n - 1)
        L, a, b = lerp_stop(t)
        result.append({"t": round(t, 4), "L": round(L, 4), "a": round(a, 4), "b": round(b, 4)})
    return result


def sample_pixels(img_path: Path, n: int = PIXEL_SAMPLES) -> list[dict]:
    """Random pixel sample from the rendered image, converted to OKLab."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).reshape(-1, 3)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    sampled = arr[idx]
    result = []
    for r, g, b in sampled.tolist():
        L, a, bv = rgb_u8_to_oklab(r, g, b)
        result.append({"L": round(L,4), "a": round(a,4), "b": round(bv,4)})
    return result


def render_palette(palette_name: str, stops: list[dict], out_dir: Path,
                   width: int, height: int) -> Path:
    """Call the Rust engine to render the spiral with this palette."""
    img_path = out_dir / "render.png"
    # Write a temporary single-palette JSON for this run
    tmp_palette = out_dir / "palette_tmp.json"
    payload = {"colormaps": [{
        "name": palette_name,
        "colorspace": "oklab",
        "stops": stops,
    }]}
    with open(tmp_palette, "w") as f:
        json.dump(payload, f)

    cmd = [
        str(RUST_BIN), "render",
        "--cx", str(SPIRAL_CENTER_X),
        "--cy", str(SPIRAL_CENTER_Y),
        "--fw", str(FRAME_WIDTH),
        "--width",  str(width),
        "--height", str(height),
        "--palette", palette_name,
        "--palette-file", str(tmp_palette),
        "--output", str(img_path),
        "--ss", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Rust render failed:\n{result.stderr}")
    return img_path


def run_extractor(img_path: Path) -> dict:
    """Import and run palette_extract.py, return its result dict."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("palette_extract", EXTRACTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_palette(str(img_path))


def process_one(palette_name: str, palette_data: dict, out_dir: Path,
                width: int, height: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    source_stops = palette_data.get("stops", [])

    # 1. Render
    img_path = render_palette(palette_name, source_stops, out_dir, width, height)

    # 2. Extract
    extracted = run_extractor(img_path)

    # 3. Sample LUT
    lut_points = sample_lut(source_stops, LUT_SAMPLES)

    # 4. Sample rendered pixels
    pixel_points = sample_pixels(img_path, PIXEL_SAMPLES)

    # 5. Save per-run JSON
    run_data = {
        "name": palette_name,
        "render": str(img_path).replace("\\", "/"),
        "coverage": round(extracted.get("coverage", 0.0), 4),
        "native": extracted.get("native", True),
        "lut_points":       lut_points,
        "pixel_points":     pixel_points,
        "extracted_stops":  extracted.get("stops", []),
    }
    run_json = out_dir / "data.json"
    with open(run_json, "w") as f:
        json.dump(run_data, f)

    return {
        "name": palette_name,
        "dir":  str(out_dir).replace("\\", "/"),
        "data_json": str(run_json).replace("\\", "/"),
        "render": str(img_path).replace("\\", "/"),
        "coverage": run_data["coverage"],
        "native": run_data["native"],
        "error": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int, default=10)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--width",  type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()

    palettes = load_palettes()
    names = list(palettes.keys())
    rng = random.Random(args.seed)
    chosen = rng.sample(names, min(args.n, len(names)))
    print(f"Selected {len(chosen)} palettes: {', '.join(chosen)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []

    for i, name in enumerate(chosen, 1):
        out_dir = OUT_ROOT / name
        print(f"[{i}/{len(chosen)}] {name}...", end=" ", flush=True)
        try:
            entry = process_one(name, palettes[name], out_dir, args.width, args.height)
            print(f"coverage={entry['coverage']:.3f}")
        except Exception:
            tb = traceback.format_exc(limit=4)
            print(f"ERROR\n{tb}")
            entry = {
                "name": name, "dir": str(out_dir).replace("\\","/"),
                "data_json": None, "render": None,
                "coverage": None, "native": None, "error": tb,
            }
        entries.append(entry)

    manifest = {
        "spiral_center": [SPIRAL_CENTER_X, SPIRAL_CENTER_Y],
        "frame_width": FRAME_WIDTH,
        "lut_samples": LUT_SAMPLES,
        "pixel_samples": PIXEL_SAMPLES,
        "entries": entries,
    }
    manifest_path = OUT_ROOT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
