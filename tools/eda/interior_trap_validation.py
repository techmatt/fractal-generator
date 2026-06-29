"""Interior-trap validation analysis: SHA table, contact sheet, interior crops,
maxiter-contrast note. Render-only validation (see
prompts/interior-trap-validation-prompt.md). No engine changes."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "out/interior_trap_validation/cells"
OUT = ROOT / "out/interior_trap_validation"

FIELDS = ["trap_cross", "trap_circle"]
ANCHORS = ["rabbit", "basilica"]
MAXITERS = [500, 2000, 8000]

# Interior windows (x0, y0, w, h) — a region that renders as filled interior.
# Both anchors are full-set Julias centered at 0 fw=3, so the central bulb is interior.
INTERIOR_WINDOW = {
    "rabbit": (512, 232, 256, 256),
    "basilica": (512, 232, 256, 256),
}


def cell_path(field, anchor, m):
    return CELLS / f"{field}_{anchor}_m{m}.png"


def lum(arr):
    # Rec.709 luma on sRGB-encoded uint8 → float [0,1]
    a = arr.astype(np.float64) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def load_font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    # ---- 1. SHA-256 table + collision guards ----
    shas = {}
    for field in FIELDS:
        for anchor in ANCHORS:
            for m in MAXITERS:
                p = cell_path(field, anchor, m)
                if not p.exists():
                    print(f"MISSING: {p}", file=sys.stderr)
                    sys.exit(1)
                shas[(field, anchor, m)] = hashlib.sha256(p.read_bytes()).hexdigest()

    sha_lines = ["# SHA-256 — interior trap validation (12 cells)\n"]
    sha_lines.append(f"{'field':12} {'anchor':9} {'maxiter':>7}  sha256")
    for field in FIELDS:
        for anchor in ANCHORS:
            for m in MAXITERS:
                s = shas[(field, anchor, m)]
                sha_lines.append(f"{field:12} {anchor:9} {m:7d}  {s}")

    # Guard 1a: maxiter collision within a field+anchor (sweep no-op).
    flags = []
    for field in FIELDS:
        for anchor in ANCHORS:
            seen = {}
            for m in MAXITERS:
                s = shas[(field, anchor, m)]
                if s in seen:
                    flags.append(
                        f"COLLISION [maxiter no-op]: {field}/{anchor} m{seen[s]} == m{m}"
                    )
                seen[s] = m
    # Guard 1b: field collision at matched anchor+maxiter (field-key no-op).
    for anchor in ANCHORS:
        for m in MAXITERS:
            a = shas[("trap_cross", anchor, m)]
            b = shas[("trap_circle", anchor, m)]
            if a == b:
                flags.append(
                    f"COLLISION [field-key no-op]: trap_cross==trap_circle at {anchor}/m{m}"
                )
    # Any other unexpected dup across all 12.
    inv = {}
    for k, s in shas.items():
        inv.setdefault(s, []).append(k)
    n_distinct = len(inv)

    sha_lines.append("")
    sha_lines.append(f"distinct hashes: {n_distinct}/12")
    if flags:
        sha_lines.append("FLAGS:")
        sha_lines.extend("  " + f for f in flags)
    else:
        sha_lines.append("FLAGS: none — all 12 distinct, no maxiter/field collisions.")
    (OUT / "sha_table.txt").write_text("\n".join(sha_lines) + "\n")
    print("\n".join(sha_lines))

    # ---- 2. Labeled contact sheet (4 rows field×anchor × 3 cols maxiter) ----
    thumb_w, thumb_h = 480, 270
    pad = 8
    label_h = 26
    col_hdr = 24
    row_hdr_w = 130
    rows = [(f, a) for f in FIELDS for a in ANCHORS]
    ncol = len(MAXITERS)
    nrow = len(rows)
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
            # draw the interior-window rect (scaled to thumb)
            wx, wy, ww, wh = INTERIOR_WINDOW[anchor]
            sx, sy = thumb_w / img.width, thumb_h / img.height
            draw.rectangle(
                [x0 + wx * sx, y0 + wy * sy, x0 + (wx + ww) * sx, y0 + (wy + wh) * sy],
                outline=(255, 80, 80), width=2,
            )
            draw.text((x0 + 4, y0 + thumb_h + 4),
                      f"{field} {anchor} m{m}", fill=(200, 200, 205), font=font_sm)
    sheet.save(OUT / "contact_sheet.png")
    print(f"\nwrote {OUT/'contact_sheet.png'}  ({sheet_w}x{sheet_h})")

    # ---- 3. Interior 1:1 crops + per-anchor montage + 4. contrast metric ----
    metric_lines = ["# Interior contrast vs maxiter (luminance std within the red window)\n"]
    metric_lines.append(f"interior windows (x,y,w,h): {INTERIOR_WINDOW}\n")
    for anchor in ANCHORS:
        wx, wy, ww, wh = INTERIOR_WINDOW[anchor]
        # montage: rows=field, cols=maxiter, of the 1:1 interior crops
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
                # high-freq detail: mean abs of a 3x3 Laplacian
                lap = (np.abs(4 * L[1:-1, 1:-1] - L[:-2, 1:-1] - L[2:, 1:-1]
                              - L[1:-1, :-2] - L[1:-1, 2:]))
                detail = float(lap.mean())
                stats.append((m, std, detail, float(L.mean())))
                x0 = pad + ci * (ww + pad)
                y0 = pad + fi * (wh + label_h + pad)
                mont.paste(crop, (x0, y0))
                md.text((x0 + 3, y0 + wh + 3),
                        f"{field} m{m}  std={std:.4f} det={detail:.4f}",
                        fill=(210, 210, 215), font=font_sm)
            for m, std, detail, mean in stats:
                metric_lines.append(
                    f"  m{m:<5d} lum_mean={mean:.4f}  lum_std={std:.4f}  detail={detail:.4f}")
            s = [x[1] for x in stats]  # std at 500,2000,8000
            d = [x[2] for x in stats]
            metric_lines.append(f"  -> std verdict:    {classify(s)}")
            metric_lines.append(f"  -> detail verdict: {classify(d)}\n")
        mont.save(OUT / f"interior_crops_{anchor}.png")
        print(f"wrote {OUT/f'interior_crops_{anchor}.png'}")

    (OUT / "maxiter_contrast_note.txt").write_text("\n".join(metric_lines) + "\n")
    print("\n" + "\n".join(metric_lines))


def classify(vals):
    """Classify a 3-point sequence [v500, v2000, v8000] as rise-then-fall / rise /
    fall / flat. 'Flat' if the spread is a tiny fraction of the max."""
    v0, v1, v2 = vals
    mx = max(vals)
    spread = mx - min(vals)
    if mx < 1e-6 or spread < 0.05 * mx:
        return f"flat ({v0:.4f}->{v1:.4f}->{v2:.4f}, spread {spread:.4f} < 5% of peak)"
    peak = "500/2000/8000"[["500", "2000", "8000"].index(
        ["500", "2000", "8000"][int(np.argmax(vals))])]
    pk = int(np.argmax(vals))
    label = ["500", "2000", "8000"][pk]
    if pk == 1:
        return f"RISE-then-FALL — peak at m{label} ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    if v0 <= v1 <= v2:
        return f"rises monotonically ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    if v0 >= v1 >= v2:
        return f"falls monotonically ({v0:.4f}->{v1:.4f}->{v2:.4f})"
    return f"peak at m{label} ({v0:.4f}->{v1:.4f}->{v2:.4f})"


if __name__ == "__main__":
    main()
