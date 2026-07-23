#!/usr/bin/env python
"""q4 exemplar-neighborhood sweep — is the artist-quality 'corner' a reachable
region or an isolated spike?

Measurement pass (NO descent / config / data/ changes). See
prompts/q4_neighborhood_sweep.md. Field-dumps a zoom x pan grid around the
center-view Julia exemplar, computes colormap-invariant quiet-region + detail
metrics per framing, judges fragility, then renders judge-quality colored
contact sheets for the two target bins + the two failure modes.

Field source is the f64 escape-time backend (colormap-invariant); colored sheet
renders use render-one with the exemplar palette. Everything writes under
out/q4_sweep/ (disposable); the field bins are purged per-unit.

Stages (idempotent, resume from checkpoint):
  measure   field-dump the full grid, compute metrics -> metrics.jsonl
  analyze   fragility verdict + bin selection -> bins.json + fragility.png
  sheets    colored judge-quality renders for the selected framings -> *.png
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "q4_sweep"
FIELDS = OUT / "fields"
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
PALETTE = "twilight_shifted"

# Exemplar (center-view Julia, origin-centered).
EX_C = (0.26103, -0.48932)
EX_FW = 0.75

# --- measurement geometry --------------------------------------------------
MEAS_W, MEAS_H = 768, 432            # ss1 field-dump res (colormap-invariant)
WIN = 5                              # local-std window (pixels @ meas res)

# --- two-scale detail bands on the normalized [0,1] escape field -----------
# Speckle vs ornate is a SCALE distinction, not a magnitude one, so detail is
# decomposed into `fine` (pixel-scale high-freq residual) and `struct` (mid-scale
# structure). Calibrated on exemplar / deep-zoom / void reference framings
# (scratchpad/q4/calib2.py; struct_e[50,90,99] ≈ 0.086/0.161/0.215 on the exemplar).
STRUCT_FLAT = 0.030                  # struct_e below -> flat/boring (no structure at any scale)
STRUCT_MID_HI = 0.180                # struct_e in [FLAT, MID_HI) -> healthy ornate mid-detail
FINE_SPECKLE = 0.30                  # fine above this AND struct_e < SPECKLE_STRUCT -> speckle
SPECKLE_STRUCT = 0.05                # pixel-scale energy without mid-scale structure = speckle
DEEP_NORM = 0.80                     # normalized escape >= this -> slow-escape ("almost-negative")

# auto_maxiter policy (mirror tools/emission/descriptor.py — pure fn of fw).
_FW_HOME, _MB, _MK, _MMIN, _MMAX = 3.0, 500, 0.30, 200, 8000


def auto_maxiter(fw: float) -> int:
    ratio = _FW_HOME / fw if fw > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    return int(max(_MMIN, min(_MMAX, _MB * (1.0 + _MK * lz))))


# --------------------------------------------------------------------------- #
# Grid                                                                        #
# --------------------------------------------------------------------------- #
def build_grid():
    """zoom x pan grid. Returns list of dicts (idx, cx, cy, fw, iz, ipx, ipy)."""
    n_zoom = 11
    octaves = np.linspace(-2.5, 2.5, n_zoom)          # +/-2.5 octaves around fw
    fws = EX_FW * (2.0 ** octaves)
    pan = np.linspace(-0.5, 0.5, 5)                    # fraction of fw, 5x5
    grid = []
    idx = 0
    for iz, fw in enumerate(fws):
        for ipy, fy in enumerate(pan):
            for ipx, fx in enumerate(pan):
                grid.append(dict(idx=idx, iz=iz, ipx=ipx, ipy=ipy,
                                 cx=float(fx * fw), cy=float(fy * fw), fw=float(fw)))
                idx += 1
    return grid


# --------------------------------------------------------------------------- #
# Field dump + metrics                                                         #
# --------------------------------------------------------------------------- #
def dump_field(cx, cy, fw, out_bin, maxiter):
    cmd = [str(EXE), "render-one", "--julia", "--c", str(EX_C[0]), str(EX_C[1]),
           "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(maxiter),
           "--width", str(MEAS_W), "--height", str(MEAS_H), "--supersample", "1",
           "--dump-field", str(out_bin), "--dump-field-source", "f64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field failed: {r.stderr[-400:]}")


def load_values(bin_path):
    meta = json.loads(Path(str(bin_path).replace(".bin", ".json")).read_text())
    w, h = int(meta["width"]), int(meta["height"])
    raw = np.frombuffer(Path(bin_path).read_bytes(), dtype="<f4")
    return raw.reshape(h, w).astype(np.float64)


def two_scale(work):
    """Decompose a filled [0,1] field into (fine, struct_e).

    fine    = |field - 3x3 lowpass|            pixel-scale high-freq energy
    struct_e = 5x5 local std of (3x3 lp - 11x11 lp)  mid-scale structure (ornate signal)
    """
    lp3 = uniform_filter(work, 3, mode="nearest")
    fine = np.abs(work - lp3)
    struct = lp3 - uniform_filter(work, 11, mode="nearest")
    m = uniform_filter(struct, 5, mode="nearest")
    m2 = uniform_filter(struct * struct, 5, mode="nearest")
    struct_e = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return fine, struct_e


def compute_metrics(values, return_maps=False):
    """Colormap-invariant metrics from an f64 smooth field (NaN interior)."""
    finite = np.isfinite(values)
    interior_frac = float(1.0 - finite.mean())

    vals = values[finite]
    if vals.size < 16:
        z = dict(interior_frac=interior_frac, deep_frac=0.0, detail_in_deep=0.0,
                 flat_frac=float(1.0 - vals.size / values.size), mid_detail_frac=0.0,
                 high_struct_frac=0.0, busy_frac=0.0, occupancy=0.0,
                 distributed_interior=0.0, mean_struct=0.0)
        return (z, None) if return_maps else z

    lo, hi = np.percentile(vals, [0.5, 99.5])
    span = max(hi - lo, 1e-9)
    norm = np.clip((values - lo) / span, 0.0, 1.0)   # higher = slower escape
    work = np.where(finite, norm, 1.0)               # interior -> deepest, flat
    fine, struct_e = two_scale(work)

    deep_mask = finite & (norm >= DEEP_NORM)
    deep_frac = float(deep_mask.mean())
    detail_in_deep = float(struct_e[deep_mask].mean()) if deep_mask.any() else 0.0

    flat = struct_e < STRUCT_FLAT
    mid = (struct_e >= STRUCT_FLAT) & (struct_e < STRUCT_MID_HI)
    high = struct_e >= STRUCT_MID_HI
    speckle = (fine > FINE_SPECKLE) & (struct_e < SPECKLE_STRUCT)
    flat_frac = float(flat.mean())
    mid_detail_frac = float(mid.mean())
    high_struct_frac = float(high.mean())
    busy_frac = float(speckle.mean())
    occupancy = float((~flat).mean())

    # distributed-interior: fraction of 8x8 tiles containing any interior pixel
    H, W = values.shape
    th, tw = H // 8, W // 8
    tiles = sum(1 for ty in range(8) for tx in range(8)
                if not finite[ty*th:(ty+1)*th, tx*tw:(tx+1)*tw].all())
    distributed_interior = tiles / 64.0

    m = dict(interior_frac=interior_frac, deep_frac=deep_frac,
             detail_in_deep=detail_in_deep, flat_frac=flat_frac,
             mid_detail_frac=mid_detail_frac, high_struct_frac=high_struct_frac,
             busy_frac=busy_frac, occupancy=occupancy,
             distributed_interior=distributed_interior,
             mean_struct=float(struct_e.mean()))
    return (m, (fine, struct_e)) if return_maps else m


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_calibrate():
    """Inspect the exemplar local-std distribution to set FLAT_FLOOR / BUSY_CEIL."""
    FIELDS.mkdir(parents=True, exist_ok=True)
    b = FIELDS / "calib.bin"
    dump_field(0.0, 0.0, EX_FW, b, auto_maxiter(EX_FW))
    vals = load_values(b)
    m, (fine, struct_e) = compute_metrics(vals, return_maps=True)
    print("exemplar struct_e pct [50,75,90,95,99]:", np.round(np.percentile(struct_e, [50,75,90,95,99]), 4))
    print("exemplar fine    pct [50,75,90,95,99]:", np.round(np.percentile(fine, [50,75,90,95,99]), 4))
    print("exemplar metrics:", json.dumps({k: round(v, 4) for k, v in m.items()}))


def stage_measure():
    FIELDS.mkdir(parents=True, exist_ok=True)
    grid = build_grid()
    ckpt = OUT / "metrics.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["idx"])
    print(f"grid: {len(grid)} framings, {len(done)} already done")
    t0 = time.time()
    with ckpt.open("a") as f:
        for i, g in enumerate(grid):
            if g["idx"] in done:
                continue
            mi = auto_maxiter(g["fw"])
            b = FIELDS / f"f_{g['idx']:04d}.bin"
            dump_field(g["cx"], g["cy"], g["fw"], b, mi)
            m = compute_metrics(load_values(b))
            rec = {**g, "maxiter": mi, **m}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            # purge scratch per-unit
            b.unlink(missing_ok=True)
            Path(str(b).replace(".bin", ".json")).unlink(missing_ok=True)
            n = len(done) + (i - sum(1 for gg in grid[:i] if gg["idx"] in done)) + 1
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(grid)}  {el:.1f}s")
    print(f"measure done in {time.time()-t0:.1f}s -> {ckpt}")


def load_metrics():
    recs = [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["idx"])
    return recs


def exemplar_record(recs):
    """The framing at (cx=cy=0, fw≈EX_FW) — the center pan of the center zoom."""
    best = min(recs, key=lambda r: (abs(math.log2(r["fw"]/EX_FW)), abs(r["cx"])+abs(r["cy"])))
    return best


# --- bin scoring -----------------------------------------------------------
def score_A(r):
    """Exemplar-like: composed black lakes + distributed mid-detail."""
    it = r["interior_frac"]
    band_pen = max(0.0, it - 0.35) + max(0.0, 0.10 - it)   # want interior in [0.10,0.35]
    return (r["mid_detail_frac"] + 0.3 * r["distributed_interior"]
            - 0.4 * r["flat_frac"] - 2.0 * band_pen - 5.0 * r["busy_frac"])


def score_B(r):
    """Variant B target: less interior, most slow-escape ('deep') texture.
    NOTE: literal high-deep_frac B is unreachable here (deep_frac caps ~0.085);
    this ranks the best-available slow-escape-textured, low-interior framings."""
    return (4.0 * r["deep_frac"] + 3.0 * r["detail_in_deep"] + 0.3 * r["mid_detail_frac"]
            - 0.6 * r["interior_frac"] - 0.4 * r["flat_frac"])


def score_flat(r):
    return r["flat_frac"]


def score_busy(r):
    return (r["busy_frac"], r["high_struct_frac"])


def metric_dist(a, b):
    """Fragility distance in (interior, deep, mid_detail) space (normalized spreads)."""
    scale = dict(interior_frac=0.15, deep_frac=0.02, mid_detail_frac=0.20)
    return math.sqrt(sum(((a[k] - b[k]) / s) ** 2 for k, s in scale.items()))


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_metrics()
    ex = exemplar_record(recs)
    for r in recs:
        r["dist"] = metric_dist(r, ex)

    # persistence: threshold on fragility distance; report fw / pan reach
    THRESH = 2.0  # within ~2 normalized units of the exemplar = "same look"
    near = [r for r in recs if r["dist"] <= THRESH]
    fws = sorted({round(r["fw"], 3) for r in near})
    # pan reach per zoom: max chebyshev pan-ring that stays near
    pan_reach = {}
    for iz in range(11):
        rings = [max(abs(r["ipx"] - 2), abs(r["ipy"] - 2)) for r in recs if r["iz"] == iz and r["dist"] <= THRESH]
        pan_reach[iz] = max(rings) if rings else -1

    bins = {
        "exemplar_idx": ex["idx"],
        "A": [r["idx"] for r in sorted(recs, key=score_A, reverse=True)[:9]],
        "B": [r["idx"] for r in sorted(recs, key=score_B, reverse=True)[:9]],
        "flat": [r["idx"] for r in sorted(recs, key=score_flat, reverse=True)[:9]],
        "busy": [r["idx"] for r in sorted(recs, key=score_busy, reverse=True)[:9]],
        "near_thresh": THRESH,
        "near_count": len(near),
        "near_fw_range": [min(fws), max(fws)] if fws else None,
        "pan_reach_by_zoom": pan_reach,
        "deep_frac_max": max(r["deep_frac"] for r in recs),
        "busy_frac_max": max(r["busy_frac"] for r in recs),
        "flat_frac_max": max(r["flat_frac"] for r in recs),
    }
    (OUT / "bins.json").write_text(json.dumps(bins, indent=2))

    # fragility scatter: interior x mid_detail, colored by deep_frac, sized by 1/dist
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    xs = [r["interior_frac"] for r in recs]
    ys = [r["mid_detail_frac"] for r in recs]
    cs = [r["deep_frac"] for r in recs]
    sc = axes[0].scatter(xs, ys, c=cs, cmap="viridis", s=28, alpha=0.8, edgecolors="none")
    axes[0].scatter([ex["interior_frac"]], [ex["mid_detail_frac"]], marker="*", s=420,
                    facecolor="red", edgecolor="k", zorder=5, label="exemplar")
    axes[0].set_xlabel("interior_frac (black lakes)")
    axes[0].set_ylabel("mid_detail_frac (ornate detail)")
    axes[0].set_title("Fragility: exemplar neighborhood\n(color = deep_frac)")
    axes[0].legend(loc="upper right")
    fig.colorbar(sc, ax=axes[0], label="deep_frac")

    # mid_detail vs fw, one point per framing, exemplar starred
    fwv = [r["fw"] for r in recs]
    md = [r["mid_detail_frac"] for r in recs]
    fl = [r["flat_frac"] for r in recs]
    axes[1].scatter(fwv, md, s=22, alpha=0.55, label="mid_detail_frac", color="tab:blue")
    axes[1].scatter(fwv, fl, s=22, alpha=0.35, label="flat_frac", color="tab:orange")
    axes[1].scatter([ex["fw"]], [ex["mid_detail_frac"]], marker="*", s=420,
                    facecolor="red", edgecolor="k", zorder=5)
    axes[1].axvspan(min(fws), max(fws), color="green", alpha=0.08) if fws else None
    axes[1].set_xscale("log")
    axes[1].set_xlabel("frame width (log)")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Persistence vs zoom\n(green band = exemplar-like fw reach)")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fragility.png", dpi=110)
    print("wrote bins.json + fragility.png")
    print(f"near-exemplar framings: {len(near)}/{len(recs)}  fw reach {bins['near_fw_range']}")
    print(f"deep_frac max {bins['deep_frac_max']:.3f}  busy_frac max {bins['busy_frac_max']:.4f}")
    print("pan reach by zoom (max ring within thresh):", pan_reach)


# --------------------------------------------------------------------------- #
# Sheets                                                                       #
# --------------------------------------------------------------------------- #
SHEET_W, SHEET_H, SHEET_SS = 1024, 576, 2


def render_color(r, out_png):
    cmd = [str(EXE), "render-one", "--julia", "--c", str(EX_C[0]), str(EX_C[1]),
           "--cx", repr(r["cx"]), "--cy", repr(r["cy"]), "--fw", repr(r["fw"]),
           "--family", "mandelbrot", "--maxiter", str(auto_maxiter(r["fw"])),
           "--width", str(SHEET_W), "--height", str(SHEET_H), "--supersample", str(SHEET_SS),
           "--palette", PALETTE, "--out", str(out_png)]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        raise RuntimeError(rr.stderr[-400:])


def montage(records, scores, out_png, title, cols=3):
    from PIL import Image, ImageDraw
    rend = OUT / "renders"
    rend.mkdir(exist_ok=True)
    thumbs = []
    for r, s in zip(records, scores):
        p = rend / f"r_{r['idx']:04d}.png"
        if not p.exists():
            render_color(r, p)
        thumbs.append((r, s, p))
    from PIL import Image
    tw, th = 512, 288
    rows = (len(thumbs) + cols - 1) // cols
    pad, top = 6, 34
    W = cols * tw + (cols + 1) * pad
    H = top + rows * (th + 22) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), title, fill=(235, 235, 235))
    for i, (r, s, p) in enumerate(thumbs):
        im = Image.open(p).resize((tw, th))
        cx, cy = i % cols, i // cols
        x = pad + cx * (tw + pad)
        y = top + cy * (th + 22)
        canvas.paste(im, (x, y))
        cap = (f"idx{r['idx']} fw{r['fw']:.3f} c({r['cx']:+.2f},{r['cy']:+.2f}) "
               f"int{r['interior_frac']:.2f} mid{r['mid_detail_frac']:.2f} "
               f"deep{r['deep_frac']:.3f} flat{r['flat_frac']:.2f} s{s:.2f}")
        d.text((x, y + th + 4), cap, fill=(200, 200, 200))
    canvas.save(out_png)
    print("wrote", out_png)


def stage_sheets():
    recs = load_metrics()
    by_idx = {r["idx"]: r for r in recs}
    bins = json.loads((OUT / "bins.json").read_text())

    # large exemplar reference (sheet quality)
    ex = by_idx[bins["exemplar_idx"]]
    render_color(ex, OUT / "exemplar_large.png")
    print("wrote exemplar_large.png")

    specs = [
        ("A", "BIN A — exemplar-like: composed black lakes + distributed mid-detail (best-first)", score_A),
        ("B", "BIN B — variant target: less interior, most slow-escape texture (best-first; literal high-deep B unreachable)", score_B),
        ("flat", "FAILURE — flat/boring (highest flat_frac): zoom-out / pan-into-void", score_flat),
    ]
    for key, title, fn in specs:
        rs = [by_idx[i] for i in bins[key]]
        montage(rs, [fn(r) if key != "busy" else 0 for r in rs], OUT / f"sheet_{key}.png", title)

    # busy failure sheet — honestly labeled (speckle unreachable in pan/zoom family)
    rs = [by_idx[i] for i in bins["busy"]]
    montage(rs, [r["busy_frac"] for r in rs], OUT / "sheet_busy.png",
            f"FAILURE — 'busiest' reachable (busy_frac max {bins['busy_frac_max']:.4f} — TRUE SPECKLE UNREACHABLE here)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "measure", "analyze", "sheets", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("calibrate",):
        stage_calibrate()
    if args.stage in ("measure", "all"):
        stage_measure()
    if args.stage in ("analyze", "all"):
        stage_analyze()
    if args.stage in ("sheets", "all"):
        stage_sheets()
