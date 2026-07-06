"""Re-render the wallpaper-bootstrap crops at ss2 (union homogeneity — pixels only).

The wallpaper head trains on the UNION of the humanq3 batch (ss2) + the bootstrap
batch (built ss4). Re-render the 504 bootstrap crops at ss2 so the union is
homogeneous — otherwise ss-level correlates with tier (bootstrap is almost all
low-tier), landing a batch-effect confound right on the tier-3/4 axis.

PIXELS ONLY. The label is on the (location, palette, params) triple and a human tier
judgment is ss-invariant, so image_ids and labels stay valid. This script does NOT
re-run candidate selection: it reads the EXISTING `images.jsonl` and re-renders the
same 504 triples at ss2, overwriting only the crop JPEGs. images.jsonl / batch.json /
the label sidecar are left untouched.

Render path is byte-parity with build_bootstrap / build_humanq3's label crop, only ss
changes: render-one --dump-field at the label geometry (ss2) -> colormap.render_candidate
with filter=lanczos3, 1280x720, q90 JPG, center, interior=black. Field dumped once per
location (field invariance) and its picks recolored.

    uv run python tools/wallpaper/rerender_bootstrap_ss2.py --dry-run   # plan + exit
    uv run python tools/wallpaper/rerender_bootstrap_ss2.py --limit 2   # smoke (2 locs)
    uv run python tools/wallpaper/rerender_bootstrap_ss2.py             # full 504
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools"))

import query_sampler as qs            # noqa: E402  (load_pool_library)
import colormap as cm                 # noqa: E402  (CandidateConfig, load_field, render_candidate)
import location as loc_mod            # noqa: E402  (from_render_block + render_one_flags)

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
BATCH_ID = "2026-07-05_wallpaper_bootstrap_v1"
BATCH_DIR = ROOT / "data" / "wallpaper_corpus" / "batches" / BATCH_ID
OUT_FIELDS = ROOT / "out" / "wallpaper_fields_ss2"   # ss2 label-spec field cache (disposable)

# --- label-crop spec (must match build_bootstrap after the ss2 edit) -------
LABEL_W, LABEL_H, LABEL_SS = 1280, 720, 2
LABEL_FILTER = "lanczos3"
JPG_Q = 90
LABEL_CROP_WORKERS = 4    # project-wide max-workers cap — DO NOT raise


def read_rows():
    rows = []
    for line in (BATCH_DIR / "images.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ensure_field(loc):
    """Dump (or reuse) the ss2 label-geometry smooth field for `loc` -> FieldData.
    maxiter is the manifest's stored value (defines the pixels) — not recomputed."""
    OUT_FIELDS.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{loc.key()}|{LABEL_W}x{LABEL_H}ss{LABEL_SS}|{loc.maxiter}".encode()).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{LABEL_W}x{LABEL_H}ss{LABEL_SS}"
    bin_path = OUT_FIELDS / f"{stem}.bin"
    json_path = OUT_FIELDS / f"{stem}.json"
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


def config_from_row(row):
    """Rebuild the label-crop CandidateConfig from a manifest row's provenance.params.
    filter is forced to lanczos3 (the label-crop spec); eval_filter='box' was only the
    sampler's scoring filter and never colored the crop."""
    p = row["provenance"]["params"]
    loc = loc_mod.from_render_block(row["render"])
    return cm.CandidateConfig(
        palette=p["palette"],
        location=loc_mod.to_location_ref(loc),
        eval_width=LABEL_W, eval_height=LABEL_H,
        reverse=p["reverse"],
        log_premap=p["log_premap"],
        gamma=p["gamma"],
        phase=p["phase"],
        n_cycles=p["n_cycles"],
        interior_color=tuple(p["interior_color"]),
        filter=LABEL_FILTER,
    )


def main():
    ap = argparse.ArgumentParser(description="Re-render bootstrap crops at ss2 (pixels only).")
    ap.add_argument("--limit", type=int, default=0, help="cap number of locations (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="print plan and exit")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    rows = read_rows()
    crops_dir = BATCH_DIR / "crops"

    # Group rows by canonical location (field invariance — one dump per location).
    by_loc = OrderedDict()
    for row in rows:
        loc = loc_mod.from_render_block(row["render"])
        by_loc.setdefault(loc.key(), (loc, []))[1].append(row)

    n_locs = len(by_loc)
    n_crops = len(rows)
    print(f"[rerender] batch {BATCH_ID}")
    print(f"[rerender] {n_crops} crops across {n_locs} locations -> ss{LABEL_SS} "
          f"({LABEL_W}x{LABEL_H} {LABEL_FILTER} q{JPG_Q})")
    print(f"[rerender] every image_id already has a crop: "
          f"{all((crops_dir / (r['image_id'] + '.jpg')).exists() for r in rows)}")
    if args.dry_run:
        for i, (loc, grp) in enumerate(by_loc.values()):
            print(f"    loc[{i:02d}] {loc.family:16} fw={loc.fw[:10]} mi={loc.maxiter} "
                  f"{len(grp)} crops")
        return

    items = list(by_loc.values())
    if args.limit:
        items = items[:args.limit]
        print(f"[rerender] --limit {args.limit}: {len(items)} locations")

    lib = qs.load_pool_library()
    t_wall = time.time()
    done = 0

    for li, (loc, grp) in enumerate(items):
        t_loc = time.time()
        field = ensure_field(loc)
        prep = cm.stretch_field(field)

        def _render(row):
            cfg = config_from_row(row)
            out_path = crops_dir / f"{row['image_id']}.jpg"
            img = cm.render_candidate(field, cfg, lib, prep=prep)   # (720,1280,3) uint8 sRGB
            assert img.shape == (LABEL_H, LABEL_W, 3), (row["image_id"], img.shape)
            Image.fromarray(img).convert("RGB").save(out_path, "JPEG", quality=JPG_Q)
            return row["image_id"]

        with ThreadPoolExecutor(max_workers=min(LABEL_CROP_WORKERS, len(grp))) as ex:
            list(ex.map(_render, grp))
        done += len(grp)
        print(f"[rerender] loc {li:02d}/{len(items)} {loc.family:16} fw={loc.fw[:10]} "
              f"{len(grp)} crops  [{time.time()-t_loc:.0f}s]  ({done}/{n_crops})")

    wall = time.time() - t_wall
    print(f"\n[rerender] DONE — {done} crops re-rendered at ss{LABEL_SS} in {wall/60:.1f} min")
    print(f"[rerender] overwritten in-place: {crops_dir}")
    print(f"[rerender] images.jsonl / batch.json / labels untouched")


if __name__ == "__main__":
    main()
