"""
build_wallpaper_manifest.py

Walk the wallpaper corpus, run palette_extract.py on each image, and emit
data/palette_viz/manifest.json for the viewer (tools/viz/palette_gallery.html).

Usage:
    python build_wallpaper_manifest.py [--wallpapers DIR] [--out DIR] [--workers N]

Defaults:
    --wallpapers  C:/Users/techm/Desktop/Wallpapers
    --out         data/palette_viz
    --workers     4
"""

from __future__ import annotations
import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXTRACTOR = SCRIPT_DIR / "palette_extractor" / "palette_extract.py"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _run_one(args: tuple[Path, Path]) -> dict:
    """Worker: run extractor on one image, return manifest entry."""
    img_path, out_dir = args

    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("palette_extract", EXTRACTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stem = img_path.stem
    out_json = out_dir / f"{stem}.json"

    try:
        result = mod.extract_palette(str(img_path))
        # result expected shape (from palette_extract.py):
        #   { "stops": [...], "coverage": float, "native": bool, ... }
        with open(out_json, "w") as f:
            json.dump(result, f)
        return {
            "name": stem,
            "image": str(img_path).replace("\\", "/"),
            "palette_json": str(out_json).replace("\\", "/"),
            "coverage": round(result.get("coverage", 0.0), 4),
            "native": result.get("native", True),
            "error": None,
        }
    except Exception:
        return {
            "name": stem,
            "image": str(img_path).replace("\\", "/"),
            "palette_json": None,
            "coverage": None,
            "native": None,
            "error": traceback.format_exc(limit=3),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallpapers", default=r"C:/Users/techm/Desktop/Wallpapers")
    ap.add_argument("--out", default="data/palette_viz")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    wallpaper_dir = Path(args.wallpapers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in wallpaper_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"No images found in {wallpaper_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images. Running extractor with {args.workers} workers...")

    entries: list[dict] = []
    work = [(img, out_dir) for img in images]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futures), 1):
            entry = fut.result()
            entries.append(entry)
            status = f"cov={entry['coverage']:.3f}" if entry["coverage"] is not None else "ERROR"
            print(f"  [{i:3d}/{len(images)}] {entry['name']}  {status}")

    # sort by name for stable viewer order
    entries.sort(key=lambda e: e["name"].lower())

    errors = [e for e in entries if e["error"]]
    manifest = {
        "wallpaper_dir": str(wallpaper_dir).replace("\\", "/"),
        "total": len(entries),
        "errors": len(errors),
        "entries": entries,
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {manifest_path}  ({len(entries)} entries, {len(errors)} errors)")
    if errors:
        print("Failed images:")
        for e in errors:
            print(f"  {e['name']}: {e['error'].splitlines()[-1]}")


if __name__ == "__main__":
    main()
