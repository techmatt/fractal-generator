"""Phase 1 harvest: extract a palette from every wallpaper in the external corpus.

Per image runs BOTH shipped paths (reuse, never reimplement):
  - extract_palette         -> coverage (polyline/image-coverage), branch_drop_frac,
                               dropped_extent, closure, max_step  (open/diameter diagnostics)
  - extract_palette_cycles  -> seam_cycle, cycle_label, arclen_open/cycle, revisit_*
                               and the canonical *harvested* palette (best-cycle stops)
  - classify_palette        -> seam, n_jump, quarantine, cyclic/sequential, internal_max_step
                               (the same audit the library build uses)

Logged per image (manifest, outside out/):  cov, extent (palette gyration), arclen
(palette curve length, the complexity-proxy partner of extent), branch_drop_frac,
dropped_extent, seam_cycle, cycle_label, n_stops, plus the classify fields.

The harvested palette stored per image = the best-CYCLE result (Phase-1 mandates
extract_palette_cycles); mirror_needed comes from classify (sequential => mirror on
the Phase-4 render).

Persists to data/wallpaper_harvest/  (data/ = persistent store, survives rm -r out/*).

Usage:  python palette_extractor/harvest_wallpapers.py [--workers N] [--limit N]
"""
from __future__ import annotations
import argparse, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

WALLPAPERS = Path(r"C:/Users/techm/Desktop/Wallpapers")
OUT = ROOT / "data" / "wallpaper_harvest"
PAL_DIR = OUT / "palettes"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _one(path_str: str) -> dict:
    from palette_extract import (
        extract_palette, extract_palette_cycles, _extent, resample_closed,
    )
    from palette_lib.classify import classify_palette
    path = Path(path_str)
    stem = path.stem
    try:
        t0 = time.monotonic()
        op = extract_palette(path, verbose=False)
        cy = extract_palette_cycles(path)

        stops_rgb = cy.stops_cycle_rgb            # canonical harvested palette
        stops_lab = cy.stops_cycle_lab
        n = len(stops_rgb)
        stop_list = [[i / n, stops_rgb[i].tolist()] for i in range(n)]

        cls = classify_palette(stop_list)
        extent = _extent(stops_lab)               # palette self-extent (gyration)

        cmap = {
            "name": stem, "source": "extracted-harvest", "closed": True,
            "closure": cls["cycle"], "mirror_needed": cls["mirror_needed"],
            "cycle_label": cy.cycle_label, "stops": stop_list,
        }
        (PAL_DIR / f"{stem}.json").write_text(json.dumps(cmap))

        # resampled curve for dedup (uniform-arc, N=32)
        curve = resample_closed(stops_lab, 32)

        return {
            "name": stem, "image": str(path).replace("\\", "/"), "error": None,
            # --- gate-bearing measures ---
            "coverage": round(float(op.coverage), 4),         # image-coverage (polyline)
            "extent": round(float(extent), 4),                # palette gyration / color range
            "arclen": round(float(cy.arclen_cycle), 4),       # palette curve length (complexity partner)
            # --- traversal diagnostics ---
            "branch_drop_frac": round(float(op.branch_drop_frac), 4),
            "dropped_extent": round(float(op.dropped_extent), 4),
            "n_ridge": int(op.n_ridge), "n_chosen": int(op.n_chosen), "n_path": int(op.n_path),
            # --- closure ---
            "seam_cycle": round(float(cy.seam_cycle), 4),
            "cycle_label": cy.cycle_label,
            "arclen_open": round(float(cy.arclen_open), 4),
            "revisit_branches": int(cy.revisit_branches),
            "revisit_max_extent": round(float(cy.revisit_max_extent), 4),
            # --- classify (library-consistent audit) ---
            "seam": cls["seam"], "n_jump": cls["n_jump"],
            "internal_max_step": cls["internal_max_step"],
            "max_stop_step": cls["max_stop_step"],
            "classify_cycle": cls["cycle"], "mirror_needed": cls["mirror_needed"],
            "quarantine": cls["quarantine"],
            "n_stops": n,
            # dedup curve
            "_curve": curve.tolist(),
            "secs": round(time.monotonic() - t0, 2),
        }
    except Exception:
        return {"name": stem, "image": str(path).replace("\\", "/"),
                "error": traceback.format_exc(limit=4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    PAL_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in WALLPAPERS.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"harvesting {len(imgs)} images, {args.workers} workers -> {OUT}")

    entries: list[dict] = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, str(p)): p for p in imgs}
        for i, fut in enumerate(as_completed(futs), 1):
            e = fut.result()
            entries.append(e)
            if e.get("error"):
                print(f"  [{i:3d}/{len(imgs)}] {e['name']:40s} ERROR")
            else:
                print(f"  [{i:3d}/{len(imgs)}] {e['name']:40s} "
                      f"cov={e['coverage']:.2f} ext={e['extent']:.2f} arc={e['arclen']:.2f} "
                      f"bd={e['branch_drop_frac']:.2f} {e['cycle_label']}")

    entries.sort(key=lambda e: e["name"].lower())
    errors = [e for e in entries if e.get("error")]

    # separate the heavy dedup curves into their own file
    curves = {e["name"]: e.pop("_curve") for e in entries if "_curve" in e}
    np.savez_compressed(OUT / "dedup_curves.npz",
                        **{k: np.array(v, np.float32) for k, v in curves.items()})

    manifest = {
        "wallpaper_dir": str(WALLPAPERS).replace("\\", "/"),
        "total": len(entries), "errors": len(errors),
        "params": {"voxel_res": 48, "mass_fraction": 0.90, "support_floor": 0.0,
                   "n_stops": 256, "extractor": "extract_palette_cycles (canonical) "
                   "+ extract_palette (coverage/branch-drop diag)"},
        "entries": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {OUT/'manifest.json'}  ({len(entries)} entries, {len(errors)} errors) "
          f"in {time.monotonic()-t0:.0f}s")
    for e in errors:
        print(f"  ERROR {e['name']}: {e['error'].splitlines()[-1]}")


if __name__ == "__main__":
    main()
