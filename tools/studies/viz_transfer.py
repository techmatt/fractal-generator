"""Validation sheet for the gradient-weighted transfer (colormap.render_candidate).

Cheap, cached-locations-only check that the new structure-aware transfer behaves:

  1. PARITY (load-bearing): transfer='pct' and transfer='grad' @ transfer_gamma=0 both
     reproduce the pre-transfer render BIT-IDENTICALLY. Asserted per cached field.
  2. A comparison sheet for ONE cyclic palette:
       row A  n_cycles=3 fixed, columns = pct(linear) | grad gamma in TRANSFER_GAMMAS
       row B  grad gamma=0.5 fixed, columns = n_cycles 1..5
     Eyeball: the grad gamma0.25/0.5 tiles read as GENTLE structure-alignment against the
     pct(linear) reference; gamma up = more arc pulled onto edges.

Reuses the cached ss2 smooth fields under out/palette_viz_render/fields/ (the field⊗colormap
seam) — no re-dump, no batch sweep. The `w(v)` profile is built ONCE per field and reused
across every gamma (the whole point of the once-per-field cache seam).

    uv run python tools/palettes/viz_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import colormap as cm                                  # noqa: E402

FIELDS_DIR = ROOT / "out" / "palette_viz_render" / "fields"
OUT_DIR = ROOT / "out" / "palette_viz_render"

# The transfer sweep (mirrors sample_location's TRANSFER_GAMMAS / N_CYCLES).
TRANSFER_GAMMAS = [0.25, 0.5, 1.0, 2.0]
N_CYCLES = [1, 2, 3, 4, 5]
FIXED_NCYC = 3      # row A: n_cycles held here across the gamma sweep
FIXED_GAMMA = 0.5   # row B: transfer_gamma held here across the n_cycles sweep


def pick_cyclic_palette(lib):
    """First palette that is BOTH cyclic-typed (so n_cycles applies) and present in the
    colormaps file (so it has a LUT). Deterministic (sorted)."""
    for name in sorted(lib.colormaps):
        try:
            if lib.palette_type(name) == "cyclic":
                return name
        except KeyError:
            continue
    raise SystemExit("no cyclic palette found in the library")


def _cfg(loc, palette, ow, oh, *, n_cycles, transfer, transfer_gamma):
    return cm.CandidateConfig(
        palette=palette, location=loc, eval_width=ow, eval_height=oh,
        n_cycles=n_cycles, filter="box", transfer=transfer, transfer_gamma=transfer_gamma,
    )


def parity_check(field, lib, palette, prep, profile):
    """transfer='pct' vs transfer='grad'@gamma=0 must be byte-identical to each other AND
    to a render with no transfer field touched. Returns (ok, max_abs_diff)."""
    ow, oh = field.out_size
    common = dict(n_cycles=FIXED_NCYC)
    img_pct = cm.render_candidate(field, _cfg(field.location, palette, ow, oh, transfer="pct",
                                              transfer_gamma=0.0, **common), lib, prep=prep)
    img_g0 = cm.render_candidate(field, _cfg(field.location, palette, ow, oh, transfer="grad",
                                             transfer_gamma=0.0, **common), lib,
                                 prep=prep, profile=profile)
    diff = int(np.abs(img_pct.astype(np.int32) - img_g0.astype(np.int32)).max())
    return diff == 0, diff


def _thumb(img, w):
    im = Image.fromarray(img).convert("RGB")
    h = round(w * im.height / im.width)
    return im.resize((w, h), Image.BILINEAR), h


def build_sheet(field, lib, palette, prep, profile, out_path, tw=320, pad=6, bar=26, head=26):
    """Two labeled rows (A: gamma sweep @ n_cycles=3, B: n_cycles sweep @ gamma=0.5)."""
    ow, oh = field.out_size

    # Row A: pct(linear) reference, then grad over TRANSFER_GAMMAS.
    rowA = [("pct (linear, gamma0)", _cfg(field.location, palette, ow, oh, n_cycles=FIXED_NCYC,
                                          transfer="pct", transfer_gamma=0.0))]
    for g in TRANSFER_GAMMAS:
        rowA.append((f"grad gamma{g:g}", _cfg(field.location, palette, ow, oh, n_cycles=FIXED_NCYC,
                                              transfer="grad", transfer_gamma=g)))
    # Row B: grad @ FIXED_GAMMA across n_cycles.
    rowB = [(f"grad gamma{FIXED_GAMMA:g} n{n}", _cfg(field.location, palette, ow, oh, n_cycles=n,
                                                     transfer="grad", transfer_gamma=FIXED_GAMMA))
            for n in N_CYCLES]

    def render_row(row):
        return [(lbl, cm.render_candidate(field, c, lib, prep=prep, profile=profile)) for lbl, c in row]

    imgsA = render_row(rowA)
    imgsB = render_row(rowB)
    ncol = max(len(imgsA), len(imgsB))
    _, th = _thumb(imgsA[0][1], tw)

    W = ncol * tw + (ncol + 1) * pad
    H = head + 2 * (head + th + bar + pad) + pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 5), f"gradient transfer — {field.location.kind}  palette='{palette}'  "
                        f"[pct == grad@gamma0 bit-identical]", fill=(235, 235, 235))
    y = head
    for title, imgs in (("A) transfer_gamma sweep @ n_cycles=3", imgsA),
                        (f"B) n_cycles sweep @ transfer_gamma={FIXED_GAMMA:g}", imgsB)):
        draw.text((pad, y + 2), title, fill=(210, 210, 160))
        yy = y + head
        for c, (lbl, img) in enumerate(imgs):
            x = pad + c * (tw + pad)
            tim, _ = _thumb(img, tw)
            sheet.paste(tim, (x, yy))
            draw.text((x + 2, yy + th + 6), lbl, fill=(160, 200, 230))
        y = yy + th + bar + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    bins = sorted(FIELDS_DIR.glob("*_smooth.bin"))
    if not bins:
        raise SystemExit(f"no cached fields under {FIELDS_DIR} — run viz_render.py first")
    # Prefer a mandelbrot + one other for varied structure; cap at 2 (cheap).
    picks = [b for b in bins if b.name.startswith("mandelbrot")][:1] + \
            [b for b in bins if not b.name.startswith("mandelbrot")][:1]
    picks = picks[:2] or bins[:1]

    lib = cm.PaletteLibrary()
    palette = pick_cyclic_palette(lib)
    print(f"[transfer] cyclic palette = '{palette}'")

    all_ok = True
    for b in picks:
        field = cm.load_field(str(b))
        prep = cm.stretch_field(field)
        profile = cm.gradient_transfer_profile(field, prep)   # ONCE per field
        nz = int((profile.w > 0).sum())
        ok, diff = parity_check(field, lib, palette, prep, profile)
        all_ok &= ok
        stem = b.stem
        out_path = OUT_DIR / f"transfer_sheet_{stem}.png"
        build_sheet(field, lib, palette, prep, profile, out_path)
        print(f"[transfer] {stem}: parity {'OK' if ok else 'FAIL'} (max|d|={diff})  "
              f"profile w: {nz}/{cm.N_TRANSFER_BINS} nonzero bins  -> {out_path.relative_to(ROOT)}")

    print(f"\n[transfer] PARITY {'PASS — pct == grad@gamma0 bit-identical' if all_ok else 'FAIL'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
