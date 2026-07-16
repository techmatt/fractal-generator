"""Palette round-trip test: render canonical spiral, extract palette, compare point clouds.

Preliminary findings (see report section at bottom of this file):
  - Rust CLI has NO 'render' subcommand. Default behavior (no subcommand) IS the render.
  - Correct flags: --center-re, --center-im, --frame-width, --width, --height,
                   --supersample, --palette, --output
  - NO --palette-file override. CLI accepts only: 'default', 'cubehelix', 'viridis' as
    named built-ins, or a .ugr / .map file path. clean_colormaps.json stops are not loadable.
  - clean_colormaps.json format: list of {name, source, stops: [[t, [r,g,b]], ...]}
    (sRGB u8 lists, NOT {t,L,a,b} OKLab dicts as build_roundtrip_manifest.py assumed).
  - seed=7 picks (tab20, YlOrBr, cet_cyclic_mygbm_30_95_c78) cannot be rendered.
    Fallback: cubehelix + viridis (both in JSON + CLI), default (CLI-only, 5 hardcoded stops).
"""
import sys
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from palette_extract import extract_palette, srgb_to_oklab, resample_closed  # noqa: E402

RUST_BIN = ROOT / "target" / "release" / "fractal-generator.exe"
PALETTES_JSON = ROOT / "data" / "palettes" / "clean_colormaps.json"
OUT_ROOT = ROOT / "data" / "palette_roundtrip"

SPIRAL_RE = "-0.7453"
SPIRAL_IM = "0.1127"
FRAME_WIDTH = "0.004"
RENDER_W = 960
RENDER_H = 640
SS = 2

# default palette stops from Rust source (palette.rs ultra_fractal()) — not in clean_colormaps.json
DEFAULT_STOPS = [
    [0.0,    [0,   7,  100]],
    [0.16,   [32,  107, 203]],
    [0.42,   [237, 255, 255]],
    [0.6425, [255, 170,   0]],
    [0.8575, [0,     2,   0]],
]


def load_json_palettes() -> dict:
    """Returns {name: stops} where stops = [[t, [r,g,b]], ...]."""
    with open(PALETTES_JSON) as f:
        entries = json.load(f)
    return {e["name"]: e["stops"] for e in entries}


def render_spiral(palette_name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(RUST_BIN),
        "--center-re", SPIRAL_RE,
        "--center-im", SPIRAL_IM,
        "--frame-width", FRAME_WIDTH,
        "--width", str(RENDER_W),
        "--height", str(RENDER_H),
        "--supersample", str(SS),
        "--palette", palette_name,
        "--output", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Render failed:\nSTDERR: {r.stderr}\nSTDOUT: {r.stdout}")


def lut_points_from_srgb_stops(stops_trrgb: list, n: int = 256) -> list:
    """Interpolate n uniform t points along palette curve in OKLab.
    stops_trrgb: [[t, [r,g,b]], ...] (sRGB u8).
    Returns list of {t, L, a, b} dicts.
    """
    stops_sorted = sorted(stops_trrgb, key=lambda x: x[0])
    ts = np.array([s[0] for s in stops_sorted], dtype=float)
    rgbs = np.array([s[1] for s in stops_sorted], dtype=float)
    labs = srgb_to_oklab(rgbs)  # (k, 3) OKLab

    # Cyclic wrap: append first stop at t+1.0
    ts_c = np.concatenate([ts, [ts[0] + 1.0]])
    labs_c = np.vstack([labs, labs[:1]])

    targets = np.linspace(0.0, 1.0, n, endpoint=False)
    result = []
    for t in targets:
        i = int(np.searchsorted(ts_c, t, side="right")) - 1
        i = max(0, min(i, len(ts_c) - 2))
        t0, t1 = float(ts_c[i]), float(ts_c[i + 1])
        span = t1 - t0
        frac = (t - t0) / span if span > 1e-12 else 0.0
        lab = (1.0 - frac) * labs_c[i] + frac * labs_c[i + 1]
        result.append({"t": round(float(t), 6), "L": round(float(lab[0]), 4),
                       "a": round(float(lab[1]), 4), "b": round(float(lab[2]), 4)})
    return result


def pixel_points(img_path: Path, n: int = 4096, seed: int = 0) -> list:
    """Sample n random pixels from the render, convert to OKLab."""
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=float).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    labs = srgb_to_oklab(arr[idx])
    return [{"L": round(float(p[0]), 4), "a": round(float(p[1]), 4), "b": round(float(p[2]), 4)}
            for p in labs]


def extracted_points(stops_lab: np.ndarray, n: int = 256) -> list:
    """Resample the extracted palette's OKLab stops to n uniform arc points."""
    pts = resample_closed(stops_lab, n)  # (n, 3) OKLab
    return [{"t": round(i / n, 6), "L": round(float(p[0]), 4),
             "a": round(float(p[1]), 4), "b": round(float(p[2]), 4)}
            for i, p in enumerate(pts)]


def process_one(name: str, stops_trrgb: list, out_dir: Path) -> dict:
    render_path = out_dir / "render.png"
    extracted_json_path = out_dir / "extracted.json"
    lut_points_path = out_dir / "lut_points.json"
    pixel_points_path = out_dir / "pixel_points.json"
    extracted_points_path = out_dir / "extracted_points.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Render
    print(f"  rendering with --palette {name} ...", end=" ", flush=True)
    render_spiral(name, render_path)
    print("done")

    # Step 3: Extract
    print(f"  extracting palette ...", end=" ", flush=True)
    res = extract_palette(render_path)
    print(f"closure={res.closure} coverage={res.coverage*100:.1f}%")
    extracted_json_path.write_text(json.dumps(res.to_colormap(name), indent=2))

    # Step 4a: LUT points (source palette, OKLab)
    lut_pts = lut_points_from_srgb_stops(stops_trrgb, n=256)
    lut_points_path.write_text(json.dumps(lut_pts, indent=2))

    # Step 4b: Pixel samples from render
    pix_pts = pixel_points(render_path)
    pixel_points_path.write_text(json.dumps(pix_pts, indent=2))

    # Step 4c: Extracted palette resampled to 256 OKLab points
    ext_pts = extracted_points(res.stops_lab, n=256)
    extracted_points_path.write_text(json.dumps(ext_pts, indent=2))

    return {
        "name": name,
        "render": str(render_path).replace("\\", "/"),
        "extracted_json": str(extracted_json_path).replace("\\", "/"),
        "lut_points": str(lut_points_path).replace("\\", "/"),
        "pixel_points": str(pixel_points_path).replace("\\", "/"),
        "extracted_points": str(extracted_points_path).replace("\\", "/"),
        "coverage": round(float(res.coverage), 4),
        "closure": res.closure,
        "error": None,
    }


def main() -> None:
    # Step 1: Report stop format
    with open(PALETTES_JSON) as f:
        raw = json.load(f)
    print("=== Step 1: clean_colormaps.json format ===")
    ex = raw[0]
    print(f"  Top-level type: list of {len(raw)} entries")
    print(f"  Entry keys: {list(ex.keys())}")
    print(f"  First entry name: {ex['name']!r}, source: {ex['source']!r}, n_stops: {len(ex['stops'])}")
    print(f"  Stop format (first 2): {ex['stops'][:2]}")
    print(f"  => [[t_float, [r_u8, g_u8, b_u8]], ...] — sRGB8, NOT OKLab dicts\n")

    json_palettes = {e["name"]: e["stops"] for e in raw}

    # Palette selection: cubehelix + viridis from JSON, default hardcoded
    palettes_to_run = [
        ("cubehelix", json_palettes["cubehelix"]),
        ("viridis",   json_palettes["viridis"]),
        ("default",   DEFAULT_STOPS),
    ]
    print(f"Using palettes: {[p[0] for p in palettes_to_run]}")
    print(f"(seed=7 picks tab20/YlOrBr/cet_cyclic_mygbm_30_95_c78 cannot be rendered — "
          f"CLI has no --palette-file JSON override)\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, stops in palettes_to_run:
        print(f"[{name}]")
        try:
            entry = process_one(name, stops, OUT_ROOT / name)
        except Exception as exc:
            traceback.print_exc()
            entry = {
                "name": name,
                "render": str(OUT_ROOT / name / "render.png").replace("\\", "/"),
                "extracted_json": None, "lut_points": None,
                "pixel_points": None, "extracted_points": None,
                "coverage": None, "closure": None,
                "error": str(exc),
            }
        entries.append(entry)
        print()

    manifest = {
        "spiral_center": [float(SPIRAL_RE), float(SPIRAL_IM)],
        "frame_width": float(FRAME_WIDTH),
        "entries": entries,
    }
    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {manifest_path}")

    # Spot-check: print a few L/a/b values from each palette
    print("\n=== Spot-check: first 3 LUT points per palette ===")
    for entry in entries:
        if entry["lut_points"]:
            pts = json.loads(Path(entry["lut_points"]).read_text())
            print(f"  {entry['name']}: {pts[:3]}")
    print("\n=== Spot-check: first 3 extracted points per palette ===")
    for entry in entries:
        if entry["extracted_points"]:
            pts = json.loads(Path(entry["extracted_points"]).read_text())
            print(f"  {entry['name']}: {pts[:3]}")


if __name__ == "__main__":
    main()
