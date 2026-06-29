"""Non-hyperbolic interior test analysis: SHA table + maxiter-distinctness
finding, contact sheet, interior 1:1 crops, interior-contrast metrics.
Render-only validation (see prompts/interior-nonhyperbolic-prompt.md). No engine
changes. Companion to interior_trap_validation.py (the hyperbolic control)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "out/interior_nonhyperbolic/cells"
OUT = ROOT / "out/interior_nonhyperbolic"

FIELDS = ["trap_cross", "trap_circle"]
ANCHORS = ["siegel_golden", "parabolic_2", "parabolic_3"]
MAXITERS = [500, 2000, 8000]

# Interior windows (x0, y0, w, h) — a region INSIDE the filled interior, chosen
# per anchor by eyeballing each anchor's m500 cell. Filled in after inspection.
INTERIOR_WINDOW = {
    "siegel_golden": (700, 130, 200, 200),
    "parabolic_2": (540, 260, 200, 200),
    "parabolic_3": (540, 260, 200, 200),
}

# Hyperbolic baseline (from the prior interior_trap_validation run) for direct
# comparison. lum_std within the interior window, flat across maxiter.
HYPERBOLIC_BASELINE = {
    ("trap_cross", "rabbit"): 0.0034,
    ("trap_circle", "rabbit"): 0.0354,
    ("trap_cross", "basilica"): 0.0000,
    ("trap_circle", "basilica"): 0.0000,
}


def cell_path(field, anchor, m):
    return CELLS / f"{field}_{anchor}_m{m}.png"


def lum(arr):
    a = arr.astype(np.float64) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def load_font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def classify(vals):
    """Classify [v500, v2000, v8000] as distinct/identical and trend."""
    v0, v1, v2 = vals
    mx = max(vals)
    spread = mx - min(vals)
    if mx < 1e-6 or spread < 0.05 * mx:
        return f"FLAT ({v0:.4f}->{v1:.4f}->{v2:.4f}, spread {spread:.4f} < 5% of peak)"
    pk = int(np.argmax(vals))
    label = ["500", "2000", "8000"][pk]
    if pk == 1:
        return f"RISE-then-FALL — peak m{label} ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    if v0 <= v1 <= v2:
        return f"RISES monotonically ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    if v0 >= v1 >= v2:
        return f"FALLS monotonically ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    return f"peak m{label} ({v0:.4f}->{v1:.4f}->{v2:.4f})"


def main():
    # ---- 1. SHA-256 table + collision/distinctness guards ----
    shas = {}
    for field in FIELDS:
        for anchor in ANCHORS:
            for m in MAXITERS:
                p = cell_path(field, anchor, m)
                if not p.exists():
                    print(f"MISSING: {p}", file=sys.stderr)
                    sys.exit(1)
                shas[(field, anchor, m)] = hashlib.sha256(p.read_bytes()).hexdigest()

    sha_lines = ["# SHA-256 — interior NON-hyperbolic test (18 cells)\n"]
    sha_lines.append(f"{'field':12} {'anchor':14} {'maxiter':>7}  sha256")
    for field in FIELDS:
        for anchor in ANCHORS:
            for m in MAXITERS:
                sha_lines.append(
                    f"{field:12} {anchor:14} {m:7d}  {shas[(field, anchor, m)]}")

    # field-key guard: trap_cross != trap_circle at matched anchor/maxiter
    flags = []
    for anchor in ANCHORS:
        for m in MAXITERS:
            if shas[("trap_cross", anchor, m)] == shas[("trap_circle", anchor, m)]:
                flags.append(f"FIELD-KEY NO-OP: trap_cross==trap_circle at {anchor}/m{m}")

    # maxiter-distinctness FINDING (per field+anchor): distinct or identical?
    maxiter_finding = ["# Maxiter-distinctness finding (per field+anchor)\n"]
    for field in FIELDS:
        for anchor in ANCHORS:
            triple = [shas[(field, anchor, m)] for m in MAXITERS]
            n_distinct = len(set(triple))
            if n_distinct == 3:
                verb = "ALL 3 DISTINCT (maxiter drives pixels)"
            elif n_distinct == 1:
                verb = "ALL IDENTICAL (converged / maxiter-invariant)"
            else:
                # find which pair(s) collide
                pairs = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        if triple[i] == triple[j]:
                            pairs.append(f"m{MAXITERS[i]}==m{MAXITERS[j]}")
                verb = f"{n_distinct} distinct ({', '.join(pairs)})"
            maxiter_finding.append(f"{field:12} {anchor:14}: {verb}")

    inv = {}
    for k, s in shas.items():
        inv.setdefault(s, []).append(k)
    sha_lines.append("")
    sha_lines.append(f"distinct hashes: {len(inv)}/18")
    sha_lines.append("")
    sha_lines.extend(maxiter_finding)
    sha_lines.append("")
    if flags:
        sha_lines.append("FIELD-KEY FLAGS:")
        sha_lines.extend("  " + f for f in flags)
    else:
        sha_lines.append("FIELD-KEY: ok — trap_cross != trap_circle at every anchor/maxiter.")
    (OUT / "sha_table.txt").write_text("\n".join(sha_lines) + "\n")
    print("\n".join(sha_lines))

    # ---- 2. Labeled contact sheet (6 rows field×anchor × 3 cols maxiter) ----
    thumb_w, thumb_h = 480, 270
    pad, label_h, col_hdr, row_hdr_w = 8, 26, 24, 150
    rows = [(f, a) for f in FIELDS for a in ANCHORS]
    ncol, nrow = len(MAXITERS), len(rows)
    sheet_w = row_hdr_w + ncol * (thumb_w + pad) + pad
    sheet_h = col_hdr + nrow * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    font_sm = load_font(13)

    for ci, m in enumerate(MAXITERS):
        x = row_hdr_w + ci * (thumb_w + pad) + thumb_w // 2
        draw.text((x, 4), f"maxiter {m}", fill=(230, 230, 235), font=font, anchor="ma")

    for ri, (field, anchor) in enumerate(rows):
        y0 = col_hdr + ri * (thumb_h + label_h + pad)
        draw.text((6, y0 + thumb_h // 2), f"{field}\n{anchor}", fill=(230, 230, 235),
                  font=font, anchor="lm")
        for ci, m in enumerate(MAXITERS):
            img = Image.open(cell_path(field, anchor, m)).convert("RGB")
            th = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            x0 = row_hdr_w + ci * (thumb_w + pad)
            sheet.paste(th, (x0, y0))
            wx, wy, ww, wh = INTERIOR_WINDOW[anchor]
            sx, sy = thumb_w / img.width, thumb_h / img.height
            draw.rectangle(
                [x0 + wx * sx, y0 + wy * sy, x0 + (wx + ww) * sx, y0 + (wy + wh) * sy],
                outline=(255, 80, 80), width=2)
            draw.text((x0 + 4, y0 + thumb_h + 4), f"{field} {anchor} m{m}",
                      fill=(200, 200, 205), font=font_sm)
    sheet.save(OUT / "contact_sheet.png")
    print(f"\nwrote {OUT/'contact_sheet.png'}  ({sheet_w}x{sheet_h})")

    # ---- 3. Interior 1:1 crops + montage + 4. contrast metric ----
    metric_lines = ["# Interior contrast vs maxiter (luminance std + Laplacian detail "
                    "within the red window)\n"]
    metric_lines.append(f"interior windows (x,y,w,h): {INTERIOR_WINDOW}\n")
    metric_lines.append("Hyperbolic baseline (prior run, lum_std, FLAT across maxiter): "
                        f"{HYPERBOLIC_BASELINE}\n")
    for anchor in ANCHORS:
        wx, wy, ww, wh = INTERIOR_WINDOW[anchor]
        mont = Image.new("RGB", (ncol * (ww + pad) + pad,
                                 len(FIELDS) * (wh + label_h + pad) + pad), (20, 20, 24))
        md = ImageDraw.Draw(mont)
        for fi, field in enumerate(FIELDS):
            metric_lines.append(f"## {field} / {anchor}")
            stats = []
            for ci, m in enumerate(MAXITERS):
                img = Image.open(cell_path(field, anchor, m)).convert("RGB")
                crop = img.crop((wx, wy, wx + ww, wy + wh))
                crop.save(OUT / f"crop_{field}_{anchor}_m{m}.png")
                L = lum(np.asarray(crop))
                std = float(L.std())
                lap = np.abs(4 * L[1:-1, 1:-1] - L[:-2, 1:-1] - L[2:, 1:-1]
                             - L[1:-1, :-2] - L[1:-1, 2:])
                detail = float(lap.mean())
                arr = np.asarray(crop).reshape(-1, 3)
                uniq = len(np.unique(arr, axis=0))
                stats.append((m, std, detail, float(L.mean()), uniq))
                x0 = pad + ci * (ww + pad)
                y0 = pad + fi * (wh + label_h + pad)
                mont.paste(crop, (x0, y0))
                md.text((x0 + 3, y0 + wh + 3), f"{field} m{m} std={std:.4f} det={detail:.4f}",
                        fill=(210, 210, 215), font=font_sm)
            for m, std, detail, mean, uniq in stats:
                metric_lines.append(
                    f"  m{m:<5d} lum_mean={mean:.4f} lum_std={std:.4f} "
                    f"detail={detail:.4f} uniqRGB={uniq}")
            s = [x[1] for x in stats]
            d = [x[2] for x in stats]
            metric_lines.append(f"  -> std verdict:    {classify(s)}")
            metric_lines.append(f"  -> detail verdict: {classify(d)}\n")
        mont.save(OUT / f"interior_crops_{anchor}.png")
        print(f"wrote {OUT/f'interior_crops_{anchor}.png'}")

    (OUT / "maxiter_contrast_note.txt").write_text("\n".join(metric_lines) + "\n")
    print("\n" + "\n".join(metric_lines))


if __name__ == "__main__":
    main()
