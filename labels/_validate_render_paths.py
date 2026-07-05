"""Scratch: validate the remaining wallpaper-bootstrap render paths (phoenix,
multibrot3/4/5, julia:multibrot deg>=3) end-to-end through the driver's OWN render
function. Sources one decoded_class==3 OUTCOME per family from the gather ledgers,
dumps the ss4 label-geometry field, renders one crop, reports field stats.
No batch writes — everything under a scratch dir."""
from __future__ import annotations
import dataclasses, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))

import build_bootstrap as BB
import query_sampler as qs
import colormap as cm

SCRATCH = ROOT / "out" / "wallpaper_validate"
SCRATCH.mkdir(parents=True, exist_ok=True)

# The families whose render path is NOT yet validated. Each -> its CLASS_SPECS entry.
TARGETS = ["phoenix", "multibrot3", "multibrot4", "multibrot5", "julia:multibrot3"]
SPEC_BY_CLASS = {s[0]: s for s in BB.CLASS_SPECS}


def pick_row(spec, seed=7):
    """First decoded_class==3 row after the driver's seeded shuffle (matches select_sources)."""
    rows = BB._load_ledger(spec)
    q3 = BB._spatial_dedup([r for r in rows if r["decoded_class"] == 3])
    rng = np.random.default_rng(seed)
    rng.shuffle(q3)
    return q3[0] if q3 else None


def field_stats(field):
    v = field.values
    finite = np.isfinite(v)
    black_frac = float((~finite).mean())        # interior (NaN) fraction at ss4
    fstd = float(np.std(v[finite])) if finite.any() else 0.0
    fspan = float(np.ptp(v[finite])) if finite.any() else 0.0
    return black_frac, fstd, fspan


def rendered_black_frac(img):
    """Fraction of near-black output pixels (post-downsample sRGB) — the real gate proxy."""
    return float((img.max(axis=2) < 8).mean())


def main():
    lib = qs.load_pool_library()
    # deterministic sane palette present in the clean 661 pool
    palette = sorted(lib.colormaps.keys())[0]
    print(f"[validate] pool={len(lib.colormaps)}  test palette={palette!r}\n")

    for cls in TARGETS:
        spec = SPEC_BY_CLASS[cls]
        row = pick_row(spec)
        if row is None:
            print(f"[{cls:17}] NO class-3 row available"); continue
        loc = BB.to_location(spec, row)
        try:
            field = BB.ensure_label_field(loc)
        except Exception as e:
            print(f"[{cls:17}] FIELD-DUMP FAILED: {e}"); continue
        cfg = cm.CandidateConfig(
            palette=palette, location=cm.LocationRef(kind=loc.family, cx=loc.cx, cy=loc.cy,
                fw=loc.fw, maxiter=loc.maxiter, c_re=loc.c_re, c_im=loc.c_im),
            eval_width=BB.LABEL_W, eval_height=BB.LABEL_H, filter=BB.LABEL_FILTER)
        out_path = SCRATCH / f"{cls.replace(':','_')}.jpg"
        w, h = BB.render_label_crop(field, cfg, lib, out_path)
        img = np.asarray(Image.open(out_path))
        bf, fstd, fspan = field_stats(field)
        rbf = rendered_black_frac(img)
        ok = (fstd > 1.0) and (bf < 0.9) and (rbf < 0.9)
        print(f"[{cls:17}] {'OK ' if ok else 'CHECK'} {loc.family:16} "
              f"fw={loc.fw[:9]} mi={loc.maxiter}  field_std={fstd:6.2f} span={fspan:7.2f} "
              f"interior_frac={bf:.3f} rendered_black={rbf:.3f}  {w}x{h} -> {out_path.name}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    main()
