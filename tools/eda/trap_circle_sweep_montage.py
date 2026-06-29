"""Assemble the trap_circle render-mode sweep montage (5 trap_radius x 5 transform)."""
from PIL import Image, ImageDraw, ImageFont

OUT = "out/render_modes/trap_circle_sweep"
RADII = ["0.25", "0.5", "1.0", "1.5", "2.0"]
TRANSFORMS = ["linear", "sqrt", "log", "scurve", "histeq"]

# Anchor (random label-3 Julia, seed 20260627, src v5 manifest ds_g0056_r000398)
C_RE, C_IM = "-0.07810228973371881", "-0.6514609012382414"
CX, CY, FW = "0.4104135054546244", "0.20967482476903096", "0.5622541254857749"

TW, TH = 384, 216          # tile thumb size
PAD = 6
LABEL_H = 22
HEADER_H = 200

def font(sz):
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()

f_lbl = font(16)
f_hdr = font(18)
f_hdr_b = font(22)

cell_w = TW + 2 * PAD
cell_h = TH + LABEL_H + 2 * PAD
grid_w = cell_w * len(TRANSFORMS)
grid_h = cell_h * len(RADII)
W = grid_w
H = HEADER_H + grid_h

canvas = Image.new("RGB", (W, H), (18, 18, 20))
draw = ImageDraw.Draw(canvas)

# ---- header band ----
draw.text((PAD, 8), "trap_circle render-mode sweep  —  random label-3 Julia anchor",
          fill=(235, 235, 235), font=f_hdr_b)
lines = [
    f"c     = ({C_RE}, {C_IM})",
    f"cx/cy = ({CX}, {CY})",
    f"fw    = {FW}",
    "src   = v5 manifest  ds_g0056_r000398  (julia_ladder_j0, descent rung 3)  seed=20260627",
    "palette = RdBu (mirrored)   bailout_b = 65536   maxiter = 8000   ss2 lanczos3",
    "rows: trap_radius 0.25->2.0 (top->bottom)   cols: linear/sqrt/log/scurve/histeq (L->R)",
]
for i, ln in enumerate(lines):
    draw.text((PAD, 40 + i * 22), ln, fill=(200, 200, 205), font=f_hdr)

# reference smooth thumb (right side of header)
try:
    ref = Image.open(f"{OUT}/_ref_smooth.png").convert("RGB")
    rw = 320
    rh = int(rw * ref.height / ref.width)
    ref = ref.resize((rw, rh), Image.LANCZOS)
    rx = W - rw - PAD
    canvas.paste(ref, (rx, 6))
    draw.text((rx, 6 + rh + 1), "ref: location profile (smooth)",
              fill=(180, 180, 185), font=f_lbl)
except FileNotFoundError:
    pass

# ---- grid ----
for ri, r in enumerate(RADII):
    for ci, t in enumerate(TRANSFORMS):
        x0 = ci * cell_w
        y0 = HEADER_H + ri * cell_h
        try:
            tile = Image.open(f"{OUT}/r{r}_{t}.png").convert("RGB").resize((TW, TH), Image.LANCZOS)
            canvas.paste(tile, (x0 + PAD, y0 + PAD))
        except FileNotFoundError:
            draw.rectangle([x0 + PAD, y0 + PAD, x0 + PAD + TW, y0 + PAD + TH], outline=(80, 40, 40))
        draw.text((x0 + PAD, y0 + PAD + TH + 3), f"r={r}  {t}",
                  fill=(225, 225, 230), font=f_lbl)

canvas.save(f"{OUT}/MONTAGE.png")
print(f"wrote {OUT}/MONTAGE.png  ({W}x{H})")
