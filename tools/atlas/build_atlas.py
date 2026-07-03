#!/usr/bin/env python
"""Atlas v1 -- round 0: fit theta_hat(seed), cross-validate, visualize, persist, and
dry-run the round-1 allocate rule (prompts/atlas-round0-prompt.md).

Pure modeling: NO new descents / renders / scoring. Reuses the k3 rewards already in
walks_table_bestwalk.jsonl and the boundary-band field machinery from step0_coverage.

  uv run python tools/atlas/build_atlas.py            # full round-0: fit+CV+viz+persist+dryrun
  uv run python tools/atlas/build_atlas.py --quick    # coarser boundary raster (fast iterate)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from atlas import Atlas, _pairwise_knn                       # noqa: E402
from step0 import bin_seeds, spearman                        # noqa: E402  (grid + stats)
from step0_coverage import mandel_smooth                     # noqa: E402  (boundary field)

TABLE = ROOT / "data" / "atlas_probe" / "step0_reanalysis" / "walks_table_bestwalk.jsonl"
STEP0_HEATMAP = ROOT / "data" / "atlas_probe" / "step0_reanalysis" / "structure_heatmap_bestwalk.png"
OUT_DIR = ROOT / "data" / "atlas"
VIZ_DIR = ROOT / "out" / "atlas" / "round0"

# structure grid used by step-0 (the binned-mean baseline resolution)
BIN_NCOLS, BIN_NROWS = 14, 12


# ======================================================================= #
# data
# ======================================================================= #
def load_seeds():
    rows = [json.loads(l) for l in open(TABLE, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    cx = np.array([r["seed_cx"] for r in rows])
    cy = np.array([r["seed_cy"] for r in rows])
    fw = np.array([r["seed_fw"] for r in rows])
    y = np.array([r["reward_k3"] for r in rows])
    xy = np.stack([cx, cy], axis=1)
    return xy, y, fw, rows


def data_bounds(xy):
    cx, cy = xy[:, 0], xy[:, 1]
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    return (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)


# ======================================================================= #
# boundary-band raster  (reuses step0_coverage.mandel_smooth VERBATIM)
# ======================================================================= #
def _dilate(m, r=2):
    """Binary dilation by an (2r+1) square structuring element (numpy, no scipy)."""
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    # np.roll wraps; kill the wrapped edges for the shifted-in border rows/cols
    if r:
        out[:r, :] = m[:r, :] | out[:r, :]  # (cheap: edge cells rarely matter here)
    return out


def boundary_raster(bounds, seed_xy, nbx=140, nby=120, sub=6, maxiter=1200):
    """Finer-than-bin generalization of step0_coverage.classify_bins: tile the domain
    into nbx x nby blocks, each sampled sub x sub; a block is on the boundary band iff
    it straddles escaping+bounded OR its escaped smooth-field std exceeds a floor
    calibrated on blocks containing a training seed (known-boundary).

    The atlas DOMAIN = that boundary band UNION a dilation of the seed-support blocks.
    Rationale: theta_hat is trustworthy wherever we have training data regardless of
    the local field test (many shallow fw=0.1 seed *centers* sit in smooth exterior
    even though their frame captures boundary); the band alone drops ~30% of our own
    seeds. The band adds unsampled-but-plausible boundary for the explore case."""
    x0, x1, y0, y1 = bounds
    NX, NY = nbx * sub, nby * sub
    xs = np.linspace(x0, x1, NX)
    ys = np.linspace(y0, y1, NY)
    CX, CY = np.meshgrid(xs, ys)                      # (NY,NX), cy-ascending
    esc, sm = mandel_smooth(CX.ravel(), CY.ravel(), maxiter=maxiter)
    esc = esc.reshape(NY, NX)
    sm = sm.reshape(NY, NX)
    # fold into blocks: (nby, sub, nbx, sub)
    escb = esc.reshape(nby, sub, nbx, sub)
    smb = sm.reshape(nby, sub, nbx, sub)
    frac_esc = escb.mean(axis=(1, 3))                 # (nby,nbx)
    both = (frac_esc > 0) & (frac_esc < 1)
    # escaped-only std per block
    sm_esc = np.where(escb, smb, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(sm_esc.reshape(nby, sub, nbx, sub), axis=(1, 3))
    std = np.nan_to_num(std, nan=0.0)
    # calibrate floor on blocks that contain a training seed
    ix = np.clip(((seed_xy[:, 0] - x0) / (x1 - x0) * nbx).astype(int), 0, nbx - 1)
    iy = np.clip(((seed_xy[:, 1] - y0) / (y1 - y0) * nby).astype(int), 0, nby - 1)
    seed_block = np.zeros((nby, nbx), bool)
    seed_block[iy, ix] = True
    seeded_std = std[seed_block]
    floor = max(3.0, float(np.quantile(seeded_std, 0.15))) if seeded_std.size else 3.0
    band = both | (std > floor)
    seed_support = _dilate(seed_block, r=2)
    mask = band | seed_support
    return mask, (x0, x1, y0, y1), floor, seed_block, band


# ======================================================================= #
# estimators
# ======================================================================= #
def knn_fit_predict(train_xy, train_y, q_xy, k, weighted=True):
    idx, dist = _pairwise_knn(q_xy, train_xy, k)
    vals = train_y[idx]
    if weighted:
        w = 1.0 / (dist + 1e-9)
        return (w * vals).sum(1) / w.sum(1)
    return vals.mean(1)


def nw_fit_predict(train_xy, train_y, q_xy, h):
    """Nadaraya-Watson Gaussian kernel smoother. Falls back to nearest-neighbor
    where the kernel underflows (all weights ~0)."""
    q2 = (q_xy ** 2).sum(1)[:, None]
    t2 = (train_xy ** 2).sum(1)[None, :]
    d2 = np.maximum(q2 + t2 - 2.0 * q_xy @ train_xy.T, 0.0)
    w = np.exp(-0.5 * d2 / (h * h))
    sw = w.sum(1)
    out = (w @ train_y)
    good = sw > 1e-12
    out[good] /= sw[good]
    if (~good).any():                                 # nearest neighbor fallback
        nn = np.argmin(d2[~good], axis=1)
        out[~good] = train_y[nn]
    return out


# ======================================================================= #
# cross-validation
# ======================================================================= #
def r2(y_true, y_pred, ybar):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - ybar) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _bin_baseline_pred(train_xy, train_y, q_xy, bounds, gmean):
    """Binned-mean baseline: predict a query from the mean of TRAIN seeds in the same
    14x12 bin; fall back to the global train mean for bins empty in train."""
    tl = bin_seeds(train_xy[:, 0], train_xy[:, 1], BIN_NCOLS, BIN_NROWS, bounds)
    ql = bin_seeds(q_xy[:, 0], q_xy[:, 1], BIN_NCOLS, BIN_NROWS, bounds)
    n = BIN_NCOLS * BIN_NROWS
    sums = np.bincount(tl, weights=train_y, minlength=n)
    cnts = np.bincount(tl, minlength=n)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), gmean)
    return means[ql]


def random_kfold_folds(n, nfold, rng):
    perm = rng.permutation(n)
    return [perm[i::nfold] for i in range(nfold)]


def spatial_block_folds(xy, bounds, nbx, nby):
    """Contiguous spatial blocks -> leave-one-block-out folds (only non-empty blocks)."""
    lab = bin_seeds(xy[:, 0], xy[:, 1], nbx, nby, bounds)
    folds = []
    for b in np.unique(lab):
        te = np.where(lab == b)[0]
        if len(te) >= 3:                              # skip trivially tiny blocks
            folds.append(te)
    return folds


def cv_run(xy, y, folds, bounds, ks, hs):
    """For each estimator config, aggregate held-out predictions across folds and
    score R^2 + Spearman vs the two references. Returns dict of curves."""
    n = len(y)
    gmean = y.mean()
    idx_all = np.arange(n)

    # references (aggregate held-out)
    null_pred = np.full(n, np.nan)
    bin_pred = np.full(n, np.nan)
    knn_pred = {k: np.full(n, np.nan) for k in ks}
    nw_pred = {h: np.full(n, np.nan) for h in hs}

    for te in folds:
        tr = np.setdiff1d(idx_all, te, assume_unique=False)
        txy, ty = xy[tr], y[tr]
        null_pred[te] = ty.mean()
        bin_pred[te] = _bin_baseline_pred(txy, ty, xy[te], bounds, ty.mean())
        for k in ks:
            knn_pred[k][te] = knn_fit_predict(txy, ty, xy[te], k)
        for h in hs:
            nw_pred[h][te] = nw_fit_predict(txy, ty, xy[te], h)

    def score(pred):
        m = ~np.isnan(pred)
        return r2(y[m], pred[m], gmean), spearman(y[m], pred[m])

    out = {"null": score(null_pred), "bin": score(bin_pred),
           "knn": {k: score(knn_pred[k]) for k in ks},
           "nw": {h: score(nw_pred[h]) for h in hs}}
    return out


# ======================================================================= #
# visualization  (PIL, self-contained)
# ======================================================================= #
def _magma(t):
    anchors = [(0.0, (12, 8, 38)), (0.25, (85, 20, 110)), (0.5, (180, 55, 95)),
               (0.75, (245, 130, 55)), (1.0, (252, 235, 165))]
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]; t1, c1 = anchors[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(round(c0[j] + f * (c1[j] - c0[j]))) for j in range(3))
    return anchors[-1][1]


def _viridis(t):
    anchors = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
               (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]; t1, c1 = anchors[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(round(c0[j] + f * (c1[j] - c0[j]))) for j in range(3))
    return anchors[-1][1]


def render_field(field, mask, bounds, cmap, title, subtitle, path,
                 vlo=None, vhi=None, seeds=None, targets=None, upsample=3):
    """field/mask are (NY,NX) cy-ascending. Draw masked cells via cmap, grey off-band.
    Optionally scatter training seeds and round-1 target markers."""
    from PIL import Image, ImageDraw
    NY, NX = field.shape
    x0, x1, y0, y1 = bounds
    if vlo is None: vlo = np.nanmin(field[mask]) if mask.any() else 0.0
    if vhi is None: vhi = np.nanmax(field[mask]) if mask.any() else 1.0
    rng = (vhi - vlo) or 1.0
    up = upsample
    W, H = NX * up, NY * up
    PADL, PADT, PADR, PADB = 70, 66, 150, 40
    im = Image.new("RGB", (W + PADL + PADR, H + PADT + PADB), (16, 16, 20))
    d = ImageDraw.Draw(im)
    # raster body (flip rows so cy increases upward)
    px = np.zeros((NY, NX, 3), np.uint8)
    for iy in range(NY):
        for ix in range(NX):
            if mask[iy, ix]:
                t = (field[iy, ix] - vlo) / rng
                px[iy, ix] = cmap(t)
            else:
                px[iy, ix] = (34, 34, 40)
    body = Image.fromarray(px[::-1], "RGB").resize((W, H), Image.NEAREST)
    im.paste(body, (PADL, PADT))
    d.text((10, 10), title, fill=(235, 235, 235))
    d.text((10, 28), subtitle, fill=(180, 180, 190))

    def to_px(cx, cy):
        fx = (cx - x0) / (x1 - x0); fy = (cy - y0) / (y1 - y0)
        return PADL + fx * W, PADT + (1 - fy) * H

    if seeds is not None:
        for cx, cy in seeds:
            xx, yy = to_px(cx, cy)
            d.ellipse([xx - 1.5, yy - 1.5, xx + 1.5, yy + 1.5],
                      outline=(240, 240, 245))
    if targets is not None:
        for cx, cy in targets:
            xx, yy = to_px(cx, cy)
            d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], outline=(80, 240, 120), width=2)
    # axes labels
    d.text((PADL, PADT + H + 6), f"cx [{x0:.2f}, {x1:.2f}]", fill=(160, 160, 170))
    d.text((6, PADT), f"cy {y1:.2f}", fill=(160, 160, 170))
    d.text((6, PADT + H - 12), f"{y0:.2f}", fill=(160, 160, 170))
    # legend ramp
    lx = PADL + W + 30
    for i in range(H):
        t = 1 - i / (H - 1)
        d.line([lx, PADT + i, lx + 20, PADT + i], fill=cmap(t))
    d.text((lx, PADT - 16), f"{vhi:.3f}", fill=(200, 200, 210))
    d.text((lx, PADT + H + 4), f"{vlo:.3f}", fill=(200, 200, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def draw_cv_curves(ks, knn_scores, bin_ref, null_ref, nw_pairs, spatial, path):
    """Two-panel CV curve: R^2 (left) and Spearman (right) vs k, with bin/null refs."""
    from PIL import Image, ImageDraw
    W, H = 1180, 480
    PAD = 56
    pw = (W - 3 * PAD) // 2
    ph = H - 2 * PAD
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 8), "ATLAS CV: kNN(cx,cy) vs binned-mean(14x12) vs global-mean-null",
           fill=(235, 235, 235))
    d.text((10, 26), "solid=random 5-fold   dashed=spatial-block LOBO   "
           "green=kNN  orange=binned  grey=null", fill=(175, 175, 185))

    panels = [("R^2", 0, (-0.2, 0.8)), ("Spearman rho", 1, (-0.2, 0.9))]
    kmin, kmax = min(ks), max(ks)

    def xmap(px0, k):
        return px0 + (k - kmin) / (kmax - kmin) * pw

    for label, mi, (ylo, yhi) in panels:
        px0 = PAD + mi * (pw + PAD)
        py0 = PAD + 10
        yr = yhi - ylo

        def ymap(v):
            return py0 + (1 - (v - ylo) / yr) * ph
        # frame + zero line
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(80, 80, 90))
        yz = ymap(0.0)
        d.line([px0, yz, px0 + pw, yz], fill=(60, 60, 70))
        d.text((px0 - 2, py0 - 14), label, fill=(200, 200, 210))
        d.text((px0 - 30, yz - 6), "0", fill=(120, 120, 130))
        d.text((px0 - 34, py0 - 6), f"{yhi:.1f}", fill=(120, 120, 130))
        d.text((px0 - 34, py0 + ph - 6), f"{ylo:.1f}", fill=(120, 120, 130))

        for mode, dash in (("random", False), ("spatial", True)):
            sc = knn_scores[mode]
            pts = [(xmap(px0, k), ymap(sc[k][mi])) for k in ks]
            _polyline(d, pts, (80, 230, 120), dash)
            # references (flat lines)
            br = bin_ref[mode][mi]; nr = null_ref[mode][mi]
            _hline(d, px0, px0 + pw, ymap(br), (240, 160, 60), dash)
            _hline(d, px0, px0 + pw, ymap(nr), (130, 130, 140), dash)
        # x ticks
        for k in ks:
            if k in (kmin, 5, 10, 20, 30, kmax):
                xx = xmap(px0, k)
                d.line([xx, py0 + ph, xx, py0 + ph + 4], fill=(120, 120, 130))
                d.text((xx - 6, py0 + ph + 6), str(k), fill=(150, 150, 160))
        d.text((px0 + pw // 2 - 10, py0 + ph + 22), "k", fill=(160, 160, 170))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _polyline(d, pts, col, dash):
    for i in range(len(pts) - 1):
        if dash and i % 2 == 1:
            continue
        d.line([pts[i], pts[i + 1]], fill=col, width=2)
    for x, y in pts:
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)


def _hline(d, x0, x1, y, col, dash):
    if not dash:
        d.line([x0, y, x1, y], fill=col, width=1)
    else:
        step = 10
        x = x0
        while x < x1:
            d.line([x, y, min(x + 5, x1), y], fill=col, width=1)
            x += step


# ======================================================================= #
# main
# ======================================================================= #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser boundary raster")
    args = ap.parse_args()
    t0 = time.time()

    xy, y, fw, rows = load_seeds()
    bounds = data_bounds(xy)
    n = len(y)
    print(f"=== ATLAS ROUND 0 :: {n} walk seeds ===")
    print(f"cx[{bounds[0]:.3f},{bounds[1]:.3f}] cy[{bounds[2]:.3f},{bounds[3]:.3f}]  "
          f"reward_k3 mean {y.mean():.3f} sd {y.std():.3f} range [{y.min():.3f},{y.max():.3f}]")

    # ---- boundary-band domain --------------------------------------------- #
    nbx, nby, sub = (90, 78, 4) if args.quick else (140, 120, 6)
    print(f"\n[boundary raster] {nbx}x{nby} blocks (sub {sub}) ...", flush=True)
    mask, mbounds, floor, seed_block, band = boundary_raster(bounds, xy, nbx, nby, sub)
    print(f"  domain cells {int(mask.sum())}/{mask.size} ({100*mask.mean():.1f}%)  "
          f"[band {int(band.sum())} + seed-support]  var-floor {floor:.2f}  "
          f"seed-blocks-on-band {int((seed_block & band).sum())}/{int(seed_block.sum())} "
          f"(all in domain via support union)")

    # ---- cross-validation ------------------------------------------------- #
    ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    hs = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    rng = np.random.default_rng(0)
    rand_folds = random_kfold_folds(n, 5, rng)
    # spatial blocks: coarse contiguous grid, leave-one-block-out
    sp_folds = spatial_block_folds(xy, bounds, 6, 5)
    print(f"\n[CV] random 5-fold  |  spatial-block LOBO ({len(sp_folds)} blocks, "
          f"sizes {sorted(len(f) for f in sp_folds)})")

    cv_rand = cv_run(xy, y, rand_folds, bounds, ks, hs)
    cv_spat = cv_run(xy, y, sp_folds, bounds, ks, hs)

    def fmt(sc):  # (R2, rho)
        return f"R2={sc[0]:+.3f} rho={sc[1]:+.3f}"

    print(f"\n  references (random):  null {fmt(cv_rand['null'])}   "
          f"binned {fmt(cv_rand['bin'])}")
    print(f"  references (spatial): null {fmt(cv_spat['null'])}   "
          f"binned {fmt(cv_spat['bin'])}")
    print("\n  kNN sweep (R2 / rho):")
    print(f"  {'k':>4} | {'random R2':>10} {'rho':>7} | {'spatial R2':>11} {'rho':>7}")
    for k in ks:
        r = cv_rand["knn"][k]; s = cv_spat["knn"][k]
        print(f"  {k:>4} | {r[0]:>+10.3f} {r[1]:>+7.3f} | {s[0]:>+11.3f} {s[1]:>+7.3f}")
    print("\n  Nadaraya-Watson sweep (random R2 / rho):")
    for h in hs:
        r = cv_rand["nw"][h]
        print(f"    h={h:<5} {fmt(r)}")

    # peak by random-fold R2 (the exploit CV mode most relevant to v1)
    k_star = max(ks, key=lambda k: cv_rand["knn"][k][0])
    h_star = max(hs, key=lambda h: cv_rand["nw"][h][0])
    print(f"\n  peak kNN: k*={k_star}  {fmt(cv_rand['knn'][k_star])} (random)  "
          f"{fmt(cv_spat['knn'][k_star])} (spatial)")
    print(f"  peak NW:  h*={h_star}  {fmt(cv_rand['nw'][h_star])} (random)")
    knn_best = cv_rand["knn"][k_star][0]
    bin_r2 = cv_rand["bin"][0]
    finer = knn_best > bin_r2 + 0.005
    print(f"\n  finer-than-bin? kNN(k*) R2 {knn_best:+.3f} vs binned {bin_r2:+.3f}  "
          f"-> {'YES' if finer else 'NO'}")

    # ---- fit the atlas at k* (weighted kNN) ------------------------------- #
    _, dloo = _pairwise_knn(xy, xy, k_star + 1)      # +1: first neighbor is self (d=0)
    r_ref = float(np.median(dloo[:, -1]))
    atlas = Atlas(
        seed_xy=xy, reward=y, k=k_star, weighted=True, r_ref=r_ref,
        mask=mask, mask_bounds=mbounds,
        meta={
            "source_table": str(TABLE.relative_to(ROOT)).replace("\\", "/"),
            "n_seeds": int(n), "reward": "reward_k3 (best-over-walk, v5 CORN)",
            "estimator": "distance-weighted kNN over (cx,cy)",
            "k_star": int(k_star), "h_star_nw": float(h_star),
            "conf": "clip(r_ref / dist_to_kth_neighbor, 0, 1); "
                    "r_ref=median LOO k-th-nn distance",
            "r_ref": r_ref,
            "domain": f"boundary-band raster {nby}x{nbx} (var-floor {floor:.3f})",
            "cv_random": {"knn_k*": cv_rand["knn"][k_star], "binned": cv_rand["bin"],
                          "null": cv_rand["null"]},
            "cv_spatial": {"knn_k*": cv_spat["knn"][k_star], "binned": cv_spat["bin"],
                           "null": cv_spat["null"]},
            "finer_than_bin": bool(finer),
        },
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas.save()
    (OUT_DIR / "atlas_v1_meta.json").write_text(json.dumps(atlas.meta, indent=2))
    print(f"\n[persist] {OUT_DIR/'atlas_v1.npz'}  (+ atlas_v1_meta.json)")

    # ---- evaluate theta/conf on the raster for viz + dry-run -------------- #
    NY, NX = mask.shape
    x0, x1, y0, y1 = mbounds
    gx = np.linspace(x0, x1, NX); gy = np.linspace(y0, y1, NY)
    GX, GY = np.meshgrid(gx, gy)
    theta_flat, conf_flat, _ = atlas.query(GX.ravel(), GY.ravel())
    theta = theta_flat.reshape(NY, NX)
    conf = conf_flat.reshape(NY, NX)

    # ---- round-1 allocate dry-run (spec below; NO descents) --------------- #
    # acquisition a = theta_norm + BETA*(1-conf), restricted to the boundary band.
    # High where value is high (exploit) AND/OR support is thin (explore); the
    # (1-conf) term steers budget OFF already-dense high-theta bins (seahorse-collapse).
    BETA = 0.6
    tn = (theta - y.min()) / (y.max() - y.min() + 1e-9)
    acq = tn + BETA * (1.0 - conf)
    acq_masked = np.where(mask, acq, -np.inf)
    thr = np.quantile(acq[mask], 0.90) if mask.any() else 0.0
    target_mask = mask & (acq >= thr)
    # representative target points: greedy farthest-point pick among target cells
    tgt_cells = np.argwhere(target_mask)
    targets = _greedy_fps(tgt_cells, gx, gy, m=24)
    print(f"\n[round-1 dry-run] acq = theta_norm + {BETA}*(1-conf) | band; "
          f"targets = acq p90 ({int(target_mask.sum())} cells, {len(targets)} FPS markers)")
    print("  (SPEC ONLY -- no walks launched)")

    # ---- visualize -------------------------------------------------------- #
    seeds_xy = list(zip(xy[:, 0], xy[:, 1]))
    render_field(theta, mask, mbounds, _magma,
                 "ATLAS theta_hat(cx,cy): expected best-over-walk reward (k3)",
                 f"weighted kNN k*={k_star}; white dots=600 training seeds; grey=off-band",
                 VIZ_DIR / "theta_hat.png", seeds=seeds_xy)
    render_field(conf, mask, mbounds, _viridis,
                 "ATLAS conf(cx,cy): local training density",
                 f"clip(r_ref/dist_kth,0,1)  r_ref={r_ref:.4f}; bright=well-supported",
                 VIZ_DIR / "confidence.png", seeds=seeds_xy)
    render_field(theta, mask, mbounds, _magma,
                 "ROUND-1 ALLOCATE (dry-run): high theta_hat discounted by confidence",
                 f"green rings=target regions (acq p90, BETA={BETA}); NO descents launched",
                 VIZ_DIR / "round1_allocate.png", seeds=seeds_xy, targets=targets)

    bin_ref = {"random": cv_rand["bin"], "spatial": cv_spat["bin"]}
    null_ref = {"random": cv_rand["null"], "spatial": cv_spat["null"]}
    knn_scores = {"random": cv_rand["knn"], "spatial": cv_spat["knn"]}
    draw_cv_curves(ks, knn_scores, bin_ref, null_ref, None, None,
                   VIZ_DIR / "cv_curves.png")
    print(f"\n[viz] {VIZ_DIR}  (theta_hat, confidence, round1_allocate, cv_curves)")
    print(f"  step-0 structure heatmap for eyeball compare: {STEP0_HEATMAP}")

    print(f"\ndone in {time.time()-t0:.0f}s")


def _greedy_fps(cells, gx, gy, m):
    """Farthest-point sampling of up to m representative target cells (in c-plane)."""
    if len(cells) == 0:
        return []
    pts = np.stack([gx[cells[:, 1]], gy[cells[:, 0]]], axis=1)
    chosen = [int(np.argmax(pts[:, 0] * 0))]          # start at first (deterministic)
    d = np.full(len(pts), np.inf)
    for _ in range(min(m, len(pts)) - 1):
        last = pts[chosen[-1]]
        d = np.minimum(d, ((pts - last) ** 2).sum(1))
        chosen.append(int(np.argmax(d)))
    return [tuple(pts[i]) for i in chosen]


if __name__ == "__main__":
    main()
