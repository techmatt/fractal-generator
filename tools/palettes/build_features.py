"""Build the durable palette feature file + a visual-first validation report.

NOTE: the durable `data/palettes/palette_features.json` is now owned by
`build_pool.py`, which regenerates it over the full pool. This 76-entry builder is
kept for its visual-first report; running it writes the durable file over the
*subset*, silently regressing it. Use `build_pool.py` to regenerate the artifact.

  uv run python tools/palettes/build_features.py

Writes:
  * data/palettes/palette_features.json  -- durable per-palette feature (type,
    canonical_reversed, (32,3) Oklab trajectory, signals). Downstream loads this.
  * out/palette_types.png                -- swatch grid grouped by derived type.

Prints:
  * per-palette signal distributions (so the tunable eps thresholds can be set by eye)
  * a table sorted to surface derived-vs-declared mismatches at the top
  * the derived type for cmr.fusion, highlighted (expected: non_cyclic -- it was
    diverging before the type collapsed to binary {cyclic, non_cyclic}).
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color  # noqa: E402
import palette_features as pf  # noqa: E402

OUT_JSON = os.path.join(pf.ROOT, "data", "palettes", "palette_features.json")
OUT_PNG = os.path.join(pf.ROOT, "out", "palette_types.png")

TYPE_ORDER = ["cyclic", "non_cyclic"]


def _round(x, n=5):
    return round(float(x), n)


def write_features_json(palettes, feats, path=OUT_JSON):
    out = {}
    for p in palettes:
        nm = p["name"]
        f = feats[nm]
        out[nm] = {
            "type": pf.derive_type(f),
            "declared_cycle": p.get("cycle"),
            "canonical_reversed": f["canonical_reversed"],
            "trajectory": [[_round(v) for v in anchor] for anchor in f["trajectory"]],
            "signals": {k: _round(v) for k, v in f["signals"].items()},
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def print_distributions(feats):
    keys = ["endpoint_dist", "end_L_match", "interior_L_prominence",
            "end_chroma", "mid_chroma", "mid_vs_end_chroma"]
    print("\n=== signal distributions (n=%d) ===" % len(feats))
    print("%-22s %7s %7s %7s %7s %7s" % ("signal", "min", "p25", "med", "p75", "max"))
    for k in keys:
        v = np.array([f["signals"][k] for f in feats.values()])
        print("%-22s %7.3f %7.3f %7.3f %7.3f %7.3f"
              % (k, v.min(), np.percentile(v, 25), np.median(v),
                 np.percentile(v, 75), v.max()))
    print("thresholds: EPS_CYC=%.3f END_CHROMA_MIN=%.3f MID_CHROMA_RATIO=%.2f "
          "END_L_MATCH_A=%.2f END_L_MATCH_EPS=%.2f INTERIOR_PROM_MIN=%.2f"
          % (pf.EPS_CYC, pf.END_CHROMA_MIN, pf.MID_CHROMA_RATIO,
             pf.END_L_MATCH_A, pf.END_L_MATCH_EPS, pf.INTERIOR_PROM_MIN))


def _declared_binary(declared):
    """Map the declared `cycle` field to the binary derived space: cyclic stays
    cyclic, everything else (sequential, and any legacy diverging) -> non_cyclic."""
    return "cyclic" if declared == "cyclic" else "non_cyclic"


def print_table(palettes, feats):
    rows = []
    for p in palettes:
        nm = p["name"]
        f = feats[nm]
        s = f["signals"]
        dt = pf.derive_type(f)
        declared = p.get("cycle")
        # both spaces are now binary {cyclic, non_cyclic}; flag disagreement directly.
        mismatch = dt != _declared_binary(declared)
        rows.append((nm, dt, declared, mismatch, s))

    # sort: mismatches first, then the rest, name-stable within each group
    rows.sort(key=lambda r: (0 if r[3] else 1, r[0]))
    print("\n=== derived vs declared (mismatches first) ===")
    print("%-40s %-11s %-11s %8s %8s %8s %7s  %s"
          % ("name", "derived", "declared", "endpt", "Lmatch", "Lprom", "midR", "flag"))
    for nm, dt, decl, mism, s in rows:
        flag = "MISMATCH" if mism else ""
        print("%-40s %-11s %-11s %8.3f %8.3f %8.3f %7.2f  %s"
              % (nm, dt, decl, s["endpoint_dist"], s["end_L_match"],
                 s["interior_L_prominence"], s["mid_vs_end_chroma"], flag))


def render_swatch_grid(palettes, feats, path=OUT_PNG):
    """Swatch grid grouped by derived type. Each palette = a horizontal sRGB strip
    (sampled from its stops), labeled name + type. Sections headed by type."""
    STRIP_W, STRIP_H = 520, 34
    LABEL_W = 8  # unused text offset handled below
    PAD = 6
    HEADER_H = 26
    LABEL_H = 14
    row_h = STRIP_H + LABEL_H + PAD

    by_type = {t: [] for t in TYPE_ORDER}
    for p in palettes:
        by_type[pf.derive_type(feats[p["name"]])].append(p)
    for t in by_type:
        by_type[t].sort(key=lambda p: p["name"])

    total_rows = sum(len(v) for v in by_type.values())
    n_sections = sum(1 for v in by_type.values() if v)
    H = n_sections * HEADER_H + total_rows * row_h + PAD
    W = STRIP_W + 2 * PAD
    img = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(img)

    ts = (np.arange(STRIP_W) + 0.5) / STRIP_W
    y = PAD
    for t in TYPE_ORDER:
        pals_t = by_type[t]
        if not pals_t:
            continue
        draw.text((PAD, y + 6), "%s  (%d)" % (t.upper(), len(pals_t)), fill=(235, 235, 235))
        y += HEADER_H
        for p in pals_t:
            nm = p["name"]
            srgb = pf._lut_sample(p["stops"], ts)  # (STRIP_W, 3) sRGB 0-1
            strip = np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
            strip = np.broadcast_to(strip[None, :, :], (STRIP_H, STRIP_W, 3))
            img.paste(Image.fromarray(strip), (PAD, y))
            rev = " (rev)" if feats[nm]["canonical_reversed"] else ""
            hi = nm == "cmr.fusion"
            draw.text((PAD, y + STRIP_H + 1),
                      "%s  [decl:%s]%s%s" % (nm, p.get("cycle"), rev,
                                             "  <-- expect non_cyclic" if hi else ""),
                      fill=(255, 220, 90) if hi else (200, 200, 205))
            y += row_h

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def main():
    palettes = pf.load_palettes()
    feats = pf.compute_all_features(palettes)

    print_distributions(feats)
    print_table(palettes, feats)

    ft = pf.derive_type(feats["cmr.fusion"])
    mark = "OK" if ft == "non_cyclic" else "!! expected non_cyclic"
    print("\n>>> cmr.fusion derived type: %s  [%s]" % (ft.upper(), mark))

    write_features_json(palettes, feats)
    png = render_swatch_grid(palettes, feats)
    print("\nwrote %s" % os.path.relpath(OUT_JSON, pf.ROOT))
    print("wrote %s" % os.path.relpath(png, pf.ROOT))


if __name__ == "__main__":
    main()
