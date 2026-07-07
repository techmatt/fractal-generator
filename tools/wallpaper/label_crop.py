"""Shared label-crop rendering spec for the wallpaper batches (Recipe-2 tail).

Single source of truth for the LOCKED wallpaper label-crop geometry + the two
render primitives that were previously copy-pasted across build_bootstrap.py,
build_humanq3.py, and rerender_bootstrap_ss2.py:

  * the constants LABEL_W/H/SS, LABEL_FILTER, JPG_Q,
  * ensure_label_field  — dump (or reuse) the ss2 label-geometry smooth field,
  * render_label_crop    — color one candidate config at the label spec -> q90 JPG.

This is the Recipe-2 path (render-one --dump-field + colormap.render_candidate),
NOT the Recipe-1 corpus renderer (render_corpus_crop) — the name render_label_crop
is deliberately kept distinct from render_corpus_crop.

The field-cache directory is a PARAMETER of ensure_label_field (default
out/wallpaper_fields, shared by the bootstrap + humanq3 build scripts);
rerender_bootstrap_ss2 passes its own out/wallpaper_fields_ss2 so its ss2 cache
stays separate from the old ss4-era stems. The cache location never affects crop
bytes — the field is a pure function of loc + geometry + maxiter.
"""
from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))            # colormap.py
sys.path.insert(0, str(ROOT / "tools" / "corpus"))  # location.py

import colormap as cm                 # noqa: E402  (load_field, render_candidate)
import location as loc_mod            # noqa: E402  (render_one_flags)

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
DEFAULT_FIELDS_DIR = ROOT / "out" / "wallpaper_fields"   # bootstrap + humanq3 share this

# --- label-crop spec (LOCKED — the canonical wallpaper label geometry) -----
# ss2: the ss2/ss4 difference is washed out by the q90 JPEG + 384x224 training
# stretch and never flips a human tier label; unified across all wallpaper batches
# so ss-level does not correlate with tier (a batch-effect confound on the 3/4 axis).
LABEL_W, LABEL_H, LABEL_SS = 1280, 720, 2
LABEL_FILTER = "lanczos3"
JPG_Q = 90


def ensure_label_field(loc, fields_dir=DEFAULT_FIELDS_DIR):
    """Dump (or reuse) the label-geometry smooth field for `loc`. Returns FieldData.

    This is the expensive Rust pass — one per location, shared by that location's
    picks (coloring-independent). `fields_dir` is the cache directory; distinct
    directories never change the rendered field (pure function of loc + geometry +
    maxiter), so callers may keep separate caches (e.g. the ss2 rerender)."""
    fields_dir = Path(fields_dir)
    fields_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{loc.key()}|{LABEL_W}x{LABEL_H}ss{LABEL_SS}|{loc.maxiter}".encode()).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{LABEL_W}x{LABEL_H}ss{LABEL_SS}"
    bin_path = fields_dir / f"{stem}.bin"
    json_path = fields_dir / f"{stem}.json"
    if not (bin_path.exists() and json_path.exists()):
        cmd = [str(EXE), "render-one",
               "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
               "--width", str(LABEL_W), "--height", str(LABEL_H),
               "--supersample", str(LABEL_SS),
               "--maxiter", str(loc.maxiter),
               "--dump-field", str(bin_path)]
        cmd += loc_mod.render_one_flags(loc)
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"label dump-field failed for {stem}:\n{r.stderr[-500:]}")
    return cm.load_field(str(bin_path), str(json_path))


def render_label_crop(field, cfg, lib, out_path, prep=None):
    """Color one candidate config at the label spec (ss2 Lanczos-3) -> q90 JPG.

    Returns (w, h). SLOW — coloring the supersampled field through the LUT gather +
    Lanczos-3 separable downsample is a big single-threaded numpy job. ALWAYS pass a
    shared `prep` (cm.stretch_field(field)) so the percentile sort runs ONCE per
    location instead of once per crop. Thread-safe: reads `field`/`prep`, allocates
    its own buffers; the LUT cache tolerates concurrent bakes (colormap._LUT_MEMO)."""
    label_cfg = dataclasses.replace(cfg, filter=LABEL_FILTER)
    img = cm.render_candidate(field, label_cfg, lib, prep=prep)   # (720,1280,3) uint8 sRGB
    Image.fromarray(img).convert("RGB").save(out_path, "JPEG", quality=JPG_Q)
    return img.shape[1], img.shape[0]
