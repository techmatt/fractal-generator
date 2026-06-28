"""Before/after montage for the deband-normalization fix (stripe/tia/curvature).
Left col = OLD naive mean, right col = NEW bailout-normalized deband. A zoomed
crop strip under each pair surfaces residual banding / terrace cells."""
from PIL import Image, ImageDraw
from pathlib import Path

OUT = Path("out/deband_fix")
PAIRS = [
    ("stripe", "stripe_d6_linear_OLDnaive.png", "stripe_d6_linear.png"),
    ("tia", "tia_OLDnaive.png", "tia.png"),
    ("curvature", "curvature_OLDnaive.png", "curvature.png"),
]
THUMB_W = 760
# A 512x288 crop window (in full-res px) for the banding zoom, scaled 1.5x.
CX, CY, CW, CH = 1024, 576, 512, 288
ZOOM = 1.5


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 * len(text) + 8, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))


def thumb(p):
    im = Image.open(p).convert("RGB")
    r = THUMB_W / im.width
    return im.resize((THUMB_W, int(im.height * r)), Image.LANCZOS)


def crop(p):
    im = Image.open(p).convert("RGB").crop((CX, CY, CX + CW, CY + CH))
    return im.resize((int(CW * ZOOM), int(CH * ZOOM)), Image.NEAREST)


rows = []
for name, old, new in PAIRS:
    to, tn = thumb(OUT / old), thumb(OUT / new)
    co, cn = crop(OUT / old), crop(OUT / new)
    label(to, f"{name} OLD naive-mean")
    label(tn, f"{name} NEW deband")
    label(co, "crop OLD")
    label(cn, "crop NEW")
    top_h = max(to.height, tn.height)
    bot_h = max(co.height, cn.height)
    w = max(to.width + tn.width, co.width + cn.width)
    row = Image.new("RGB", (w, top_h + bot_h + 6), (32, 32, 32))
    row.paste(to, (0, 0))
    row.paste(tn, (to.width, 0))
    row.paste(co, (0, top_h + 6))
    row.paste(cn, (co.width, top_h + 6))
    rows.append(row)

W = max(r.width for r in rows)
H = sum(r.height for r in rows) + 8 * (len(rows) - 1)
mont = Image.new("RGB", (W, H), (16, 16, 16))
y = 0
for r in rows:
    mont.paste(r, (0, y))
    y += r.height + 8
dst = OUT / "deband_fix_montage.png"
mont.save(dst)
print(f"wrote {dst}  ({mont.width}x{mont.height})")
