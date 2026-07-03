#!/usr/bin/env python
"""Part A of the atlas re-analysis (prompts/atlas-reanalysis-prompt.md): of the 103
EMPTY bins in the step-0 14x12 seed grid, how many are genuine proposer holes
(boundary-straddling -- the atlas would need to fill them) vs correctly-empty
(interior / far-exterior, nothing to seed there)?

No GPU, no re-descend. Reuses the step-0 grid + bin edges VERBATIM (imported from
step0.py: bin_seeds / choose_grid / load_walks) and evaluates the Mandelbrot
smooth-escape field on a fine subgrid inside every bin.

A bin is BOUNDARY-STRADDLING iff, on a >=32x32 subgrid, it contains BOTH escaping
and bounded points, OR the smooth-escape-field variance exceeds a floor (the
variance clause catches thin dendrites/antennae the coarse membership test misses).

  uv run python tools/atlas_probe/step0_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Reuse step-0's grid machinery VERBATIM.
from step0 import bin_seeds, choose_grid, load_walks, DESCEND_DIR  # noqa: E402

OUT_DIR = ROOT / "data" / "atlas_probe" / "step0_reanalysis"
COVERAGE_PNG = OUT_DIR / "coverage_grid.png"

SUB = 48           # subgrid resolution per bin (>=32 as required; catches filaments)
MAXITER = 1500     # deep enough that dendrite interiors read as bounded
BAILOUT2 = 1.0e6   # |z|^2 escape radius; large -> clean smooth-iter


# --------------------------------------------------------------------------- #
# Vectorized Mandelbrot smooth-escape field over an array of c values.
# --------------------------------------------------------------------------- #
def mandel_smooth(cre: np.ndarray, cim: np.ndarray, maxiter=MAXITER):
    """Return (escaped_mask, smooth_iter). smooth_iter for bounded points is set to
    `maxiter` (so an all-bounded bin has zero variance)."""
    c = cre + 1j * cim
    z = np.zeros_like(c)
    escaped = np.zeros(c.shape, dtype=bool)
    smooth = np.full(c.shape, float(maxiter))
    # Main-cardioid / period-2 bulb shortcut: these interior regions never escape,
    # skip iterating them (huge speedup, exact).
    q = (cre - 0.25) ** 2 + cim ** 2
    in_card = q * (q + (cre - 0.25)) <= 0.25 * cim ** 2
    in_bulb = (cre + 1.0) ** 2 + cim ** 2 <= 0.0625
    known_bounded = in_card | in_bulb
    active = ~known_bounded
    zc = z.copy()
    for n in range(maxiter):
        zc[active] = zc[active] * zc[active] + c[active]
        mag2 = (zc.real ** 2 + zc.imag ** 2)
        newly = active & (mag2 > BAILOUT2)
        if newly.any():
            m = np.sqrt(mag2[newly])
            # smooth iteration count (normalized escape)
            smooth[newly] = n + 1 - np.log(np.log(m)) / np.log(2.0)
            escaped[newly] = True
            active &= ~newly
        if not active.any():
            break
    return escaped, smooth


def bin_edges(bounds, ncols, nrows):
    x0, x1, y0, y1 = bounds
    xe = np.linspace(x0, x1, ncols + 1)
    ye = np.linspace(y0, y1, nrows + 1)
    return xe, ye


def classify_bins(bounds, ncols, nrows, counts, var_floor):
    """Per bin: sample SUBxSUB c-points inside it, run the field, decide
    boundary-straddling. Returns dict of per-bin arrays + the seeded mask."""
    xe, ye = bin_edges(bounds, ncols, nrows)
    n = ncols * nrows
    both = np.zeros(n, dtype=bool)          # both escaping AND bounded present
    hi_var = np.zeros(n, dtype=bool)        # variance clause
    frac_bounded = np.zeros(n)
    field_std = np.zeros(n)
    for iy in range(nrows):
        for ix in range(ncols):
            b = iy * ncols + ix
            # sample interior of the bin (avoid the shared edges: half-cell inset)
            xs = np.linspace(xe[ix], xe[ix + 1], SUB)
            ys = np.linspace(ye[iy], ye[iy + 1], SUB)
            CX, CY = np.meshgrid(xs, ys)
            esc, sm = mandel_smooth(CX.ravel(), CY.ravel())
            fb = float((~esc).mean())
            frac_bounded[b] = fb
            # variance over the smooth field (escaped points only; bounded==maxiter
            # would dominate and is already handled by the both-present clause).
            if esc.any():
                std = float(sm[esc].std())
            else:
                std = 0.0
            field_std[b] = std
            both[b] = esc.any() and (~esc).any()
            hi_var[b] = std > var_floor
    boundary = both | hi_var
    return {
        "both": both, "hi_var": hi_var, "boundary": boundary,
        "frac_bounded": frac_bounded, "field_std": field_std,
    }


def _cmap_cat(kind: str):
    return {
        "seeded":            (70, 180, 250),   # blue  -- has proposer seeds
        "empty_boundary":    (240, 90, 70),    # red   -- real hole (should have seeds)
        "empty_nonboundary": (55, 60, 70),     # grey  -- correctly empty
    }[kind]


def draw_coverage(bounds, ncols, nrows, counts, cls):
    from PIL import Image, ImageDraw
    seeded = counts >= 1
    boundary = cls["boundary"]
    CELL, GUT_L, GUT_T = 74, 70, 78
    W = GUT_L + ncols * CELL + 20
    H = GUT_T + nrows * CELL + 60
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 10), "COVERAGE: empty bins -- real proposer hole (boundary) vs correctly-empty",
           fill=(235, 235, 235))
    d.text((10, 28), "blue=seeded   red=empty+boundary (HOLE)   grey=empty+non-boundary (ok)",
           fill=(185, 185, 195))
    for r in range(nrows):
        for c in range(ncols):
            b = r * ncols + c
            x = GUT_L + c * CELL
            y = GUT_T + (nrows - 1 - r) * CELL   # cy up
            if seeded[b]:
                kind = "seeded"
            elif boundary[b]:
                kind = "empty_boundary"
            else:
                kind = "empty_nonboundary"
            d.rectangle([x, y, x + CELL - 2, y + CELL - 2], fill=_cmap_cat(kind))
            if seeded[b]:
                d.text((x + 4, y + 4), f"n{int(counts[b])}", fill=(15, 20, 30))
            else:
                d.text((x + 4, y + 4), f"{cls['field_std'][b]:.1f}",
                       fill=(230, 230, 235))
                d.text((x + 4, y + CELL - 16),
                       f"b{cls['frac_bounded'][b]:.2f}", fill=(210, 210, 215))
    d.text((GUT_L, GUT_T + nrows * CELL + 8),
           f"cx [{bounds[0]:.2f}, {bounds[1]:.2f}]   cy [{bounds[2]:.2f}, {bounds[3]:.2f}]   "
           f"grid {ncols}x{nrows}   cell label: field_std / frac_bounded",
           fill=(170, 170, 180))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(COVERAGE_PNG)


def main():
    walks = load_walks(DESCEND_DIR)
    cx = np.array([w["seed_cx"] for w in walks])
    cy = np.array([w["seed_cy"] for w in walks])
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    bounds = (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)
    ncols, nrows, _ = choose_grid(cx, cy, bounds)
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    counts = np.bincount(labels, minlength=ncols * nrows)
    seeded = counts >= 1
    print(f"grid {ncols}x{nrows}  seeded {int(seeded.sum())}  empty {int((~seeded).sum())}")

    # ---- pick the variance floor from the field itself ----
    # First pass at floor=inf disables the variance clause so we can see the
    # smooth-field-std distribution split by seeded/empty, then set a floor in the
    # gap between far-exterior (low std) and boundary-adjacent (high std).
    cls0 = classify_bins(bounds, ncols, nrows, counts, var_floor=np.inf)
    std = cls0["field_std"]
    fb = cls0["frac_bounded"]
    # Diagnostic: seeded bins are known-boundary; use them to calibrate.
    print("\n[field_std by group]  (seeded bins are known boundary -> calibrate floor)")
    for name, m in [("seeded", seeded), ("empty", ~seeded)]:
        s = std[m]
        print(f"  {name:<7} n={m.sum():>3}  std p10/50/90 = "
              f"{np.quantile(s,0.1):.2f} / {np.quantile(s,0.5):.2f} / {np.quantile(s,0.9):.2f}"
              f"   both-present bins: {int(cls0['both'][m].sum())}")
    # Far-exterior all-escaped bins have small but nonzero std; boundary-adjacent
    # all-escaped bins spike. Floor = a low percentile of seeded-bin std (seeded
    # are boundary, so nearly all should clear it) but well above the far-exterior
    # floor. Use 20th pct of seeded std, clamped to a sane minimum.
    seeded_esc_only = std[seeded & ~cls0["both"]]
    floor = max(3.0, float(np.quantile(std[seeded], 0.15)))
    print(f"\n[var floor] = {floor:.2f}  "
          f"(15th pct of seeded field_std; both-present clause is primary)")

    cls = classify_bins(bounds, ncols, nrows, counts, var_floor=floor)
    boundary = cls["boundary"]

    # ---- cross-tab ----
    seeded_bound = int((seeded & boundary).sum())
    seeded_nb = int((seeded & ~boundary).sum())
    empty_bound = int((~seeded & boundary).sum())
    empty_nb = int((~seeded & ~boundary).sum())
    print("\n=== CROSS-TAB (14x12 = 168 bins) ===")
    print(f"  seeded + boundary      : {seeded_bound:>3}   (sanity: should ~= all seeded)")
    print(f"  seeded + NON-boundary  : {seeded_nb:>3}   (seed in a flat bin -- expect ~0)")
    print(f"  EMPTY  + boundary      : {empty_bound:>3}   <-- REAL PROPOSER HOLES")
    print(f"  EMPTY  + NON-boundary  : {empty_nb:>3}   (interior/far-exterior, correctly empty)")
    print(f"  totals: seeded {int(seeded.sum())}  empty {int((~seeded).sum())}")

    draw_coverage(bounds, ncols, nrows, counts, cls)
    print(f"\n  PNG -> {COVERAGE_PNG}")

    # ---- verdict ----
    empty_total = int((~seeded).sum())
    hole_frac = empty_bound / max(1, empty_total)
    print("\n=== VERDICT A (coverage) ===")
    print(f"  {empty_bound}/{empty_total} empty bins are boundary-straddling "
          f"({hole_frac*100:.0f}% of empties).")
    if empty_bound >= 20 and hole_frac >= 0.35:
        print("  -> COVERAGE IS PRIORITY-1: many empty bins are genuine seeding holes on the")
        print("     Mandelbrot boundary the proposer never reaches; the atlas must fill them.")
    elif empty_bound <= 8 or hole_frac <= 0.15:
        print("  -> COVERAGE IS A NON-ISSUE: empty bins are overwhelmingly interior/far-exterior")
        print("     (correctly empty). The proposer covers the boundary it can reach.")
    else:
        print("  -> COVERAGE IS SECONDARY: a modest minority of empty bins are real holes;")
        print("     worth a targeted fill but not the priority-1 lever.")


if __name__ == "__main__":
    main()
