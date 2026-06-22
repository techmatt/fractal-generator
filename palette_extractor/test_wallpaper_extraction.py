"""Test palette extraction on 10 random wallpapers (seed=42).

Samples 10 images from C:/Users/techm/Desktop/Wallpapers/, extracts a palette from each,
saves per-image JSON to data/palette_viz/test/<stem>.json, and writes a summary manifest.
"""
import sys
import json
import random
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from palette_extract import extract_palette  # noqa: E402

WALLPAPER_DIR = Path("C:/Users/techm/Desktop/Wallpapers")
OUT_DIR = ROOT / "data" / "palette_viz" / "test"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_images = sorted(p for p in WALLPAPER_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not all_images:
        raise SystemExit(f"No images found in {WALLPAPER_DIR}")

    rng = random.Random(42)
    picks = rng.sample(all_images, min(10, len(all_images)))
    print(f"Sampled {len(picks)} images (seed=42) from {WALLPAPER_DIR}\n")

    results = []
    for path in picks:
        stem = path.stem
        try:
            res = extract_palette(path)
            print(f"{stem}  closure={res.closure}  coverage={res.coverage*100:.1f}%  "
                  f"max_step={res.max_step:.4f}  ridge={res.n_ridge}")
            cm = res.to_colormap(stem)
            (OUT_DIR / f"{stem}.json").write_text(json.dumps(cm, indent=2))
            entry = {
                "name": stem,
                "closure": res.closure,
                "coverage": round(res.coverage * 100, 1),
                "max_step": round(float(res.max_step), 4),
                "ridge": res.n_ridge,
                "error": None,
            }
        except Exception as exc:
            traceback.print_exc()
            print(f"ERROR {stem}: {exc}")
            entry = {"name": stem, "closure": "?", "coverage": 0.0,
                     "max_step": 0.0, "ridge": 0, "error": str(exc)}
        results.append(entry)

    # Summary table sorted by coverage descending
    print("\n--- Summary (sorted by coverage desc) ---")
    print(f"{'Name':<52} {'Coverage%':>10} {'Closure':>10} {'max_step':>10} {'Ridge':>8}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: -x["coverage"]):
        flag = "  *** LOW COVERAGE" if r["coverage"] < 40 else ""
        print(f"{r['name']:<52} {r['coverage']:>10.1f} {r['closure']:>10} "
              f"{r['max_step']:>10.4f} {r['ridge']:>8}{flag}")

    manifest = {
        "wallpaper_dir": str(WALLPAPER_DIR),
        "seed": 42,
        "n": len(picks),
        "images": sorted(results, key=lambda x: -x["coverage"]),
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
