"""Acceptance gate for the field⊗colormap split — the load-bearing empirical test.

Renders a reference location the **normal Rust way** (beautiful smooth, canonical
params) → reference PNG; then `--dump-field`s the same location and colors it in
Python with the equivalent `CandidateConfig`; then asserts the Python output matches
the Rust reference within a tight pixel tolerance. This simultaneously validates the
field dump, pins the transform-vs-stretch order, and confirms LUT space / interior /
downsample. Saves a side-by-side (ref | python | 8×diff) for a visual check.

Run:  uv run python tools/colormap_acceptance.py [--test test_01] [--filter box]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colormap as cm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "target" / "release" / ("fractal-generator.exe" if sys.platform == "win32" else "fractal-generator")
COLORMAPS = "data/palettes/score3_colormaps.json"
TEST_RENDERS = REPO / "data" / "test_renders.json"

# Max acceptable per-channel diff (LSB) and fraction of pixels allowed to exceed 1.
TOL_MAX = 2
TOL_FRAC_GT1 = 1e-4


def _location_args(loc):
    """CLI location flags for a test_renders.json entry. `system` selects the family:
    mandelbrot (bare), julia (--julia/--c), multibrot3/4/5 (--family), or phoenix
    (--family, optional --c/--p)."""
    a = ["--cx", loc["cx"], "--cy", loc["cy"], "--fw", loc["fw"], "--maxiter", str(loc["maxiter"])]
    system = loc["system"]
    if system == "julia":
        a += ["--julia", "--c", loc["c_re"], loc["c_im"]]
    elif system.startswith("multibrot"):
        a += ["--family", system]
    elif system == "phoenix":
        a += ["--family", "phoenix"]
        if loc.get("c_re") is not None:
            a += ["--c", loc["c_re"], loc["c_im"]]
        if loc.get("p_re") is not None:
            a += ["--p", loc["p_re"], loc["p_im"]]
    return a


def run_gate(test_id="test_01", palette="twilight", filt="box",
             width=640, height=360, ss=2, out_dir=None, tol_max=TOL_MAX):
    """Render Rust ref + dump field, color in Python, compare. Returns a metrics dict."""
    if not BIN.exists():
        raise FileNotFoundError(f"release binary not found at {BIN} — run `cargo build --release`")
    locs = {l["id"]: l for l in json.loads(TEST_RENDERS.read_text())["locations"]}
    loc = locs[test_id]
    out_dir = Path(out_dir) if out_dir else (REPO / "out" / "colormap_acceptance")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_png = out_dir / f"{test_id}_{filt}_ref.png"
    field_bin = out_dir / f"{test_id}_{filt}.bin"
    common = _location_args(loc) + [
        "--width", str(width), "--height", str(height), "--supersample", str(ss),
        "--filter", filt, "--palette", palette, "--colormaps", COLORMAPS,
        "--coloring", '{"field":"smooth"}',
    ]
    subprocess.run([str(BIN), "render-one", *common, "--out", str(ref_png)],
                   check=True, capture_output=True, cwd=REPO)
    subprocess.run([str(BIN), "render-one", *common, "--dump-field", str(field_bin)],
                   check=True, capture_output=True, cwd=REPO)

    field = cm.load_field(field_bin)
    lib = cm.PaletteLibrary()
    ow, oh = field.out_size
    cfg = cm.CandidateConfig(palette=palette, location=field.location,
                             eval_width=ow, eval_height=oh, filter=filt)
    py = cm.render_candidate(field, cfg, lib)
    py_png = out_dir / f"{test_id}_{filt}_py.png"
    Image.fromarray(py).save(py_png)

    ref = np.asarray(Image.open(ref_png).convert("RGB"))
    diff = np.abs(py.astype(int) - ref.astype(int))
    metrics = {
        "test_id": test_id, "system": loc["system"], "filter": filt,
        "size": [int(ow), int(oh)],
        "max_diff": int(diff.max()),
        "mean_diff": float(diff.mean()),
        "frac_gt0": float((diff.max(axis=2) > 0).mean()),
        "frac_gt1": float((diff.max(axis=2) > 1).mean()),
    }

    # Side-by-side montage: ref | python | 8×diff (clipped).
    diff_vis = np.clip(diff * 8, 0, 255).astype(np.uint8)
    montage = np.concatenate([ref, py, diff_vis], axis=1)
    montage_png = out_dir / f"{test_id}_{filt}_montage.png"
    Image.fromarray(montage).save(montage_png)
    metrics["montage"] = str(montage_png)

    passed = metrics["max_diff"] <= tol_max and metrics["frac_gt1"] <= TOL_FRAC_GT1
    metrics["passed"] = passed
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test_01")
    ap.add_argument("--palette", default="twilight")
    ap.add_argument("--filter", default="box", choices=["box", "mitchell", "lanczos3"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--ss", type=int, default=2)
    args = ap.parse_args()
    m = run_gate(args.test, args.palette, args.filter, args.width, args.height, args.ss)
    print(json.dumps(m, indent=2))
    print("PASS" if m["passed"] else "FAIL")
    sys.exit(0 if m["passed"] else 1)


if __name__ == "__main__":
    main()
