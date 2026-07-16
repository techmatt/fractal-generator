#!/usr/bin/env python
"""Efficiency study — Lever 2 (descent resolution), paired re-runs.
(prompts/descent-efficiency-study-prompt.md)

The foci-finder is low-pass (Gaussian-smooth the node mu-field at sigma in {16..32}px,
take maxima), so full 768px node fidelity is wasted on it. Lower the node/probe
resolution and rescale the pixel-unit constants so the frame-RELATIVE smoothing scale
is preserved. In guided_descend the dilation radius (=sigma), the isolation window
(=2*sigma), the per-scale local-max radius (=0.4*sigma) and the cross-scale merge
radius (=0.75*mean_sigma) are ALL derived from sigma, so rescaling --sigma-band alone
rescales every pixel-unit constant; the dimensionless fractions (occ floor 0.321,
black-cap 0.30) are left untouched.

Runs (launched by out/_run_eff_descents.sh): 150 walks at node-width 768 / 512 / 384,
SAME global seed + --per-walk-rng, so each walk's depth-1 seed is bit-identical across
resolutions (validated) and the walks then diverge at depth>=2 (the thing under test).
Depth cap D from Lever 1 (--depth-max 14) applied to all three so the levers compound.

  --score  : score each run's pool with the k3 best-over-walk reward at the scorer's
             own fixed reframe geometry (640x360 ss2) -- resolution-independent, apples
             to apples. Reuses step0_reanalysis.process_walk VERBATIM. -> res_<R>_table.jsonl
  --analyze: per-seed paired Delta (512-768, 384-768); Spearman rho(theta_res, theta_768)
             over seed bins; filament-residual VETO (residual darkening over high
             Mandelbrot-field-std bins, from step0_coverage); pick the coarsest res
             passing rho>=0.85 + flat filament residual; combined config + speedup.
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
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import step0  # noqa: E402
import step0_reanalysis as ra  # noqa: E402
import step0_coverage as cov  # noqa: E402
from step0 import bin_seeds, choose_grid, choose_grid_stability, spearman  # noqa: E402
from active_ckpt import make_scorer  # noqa: E402

MODEL = ra.MODEL
RESOLUTIONS = [768, 512, 384]
RUN_DIR = {r: ROOT / "data" / "guided_descend" / f"eff_res_{r}" for r in RESOLUTIONS}
OUT_DIR = ROOT / "data" / "atlas_probe" / "step0_efficiency"
TABLE = {r: OUT_DIR / f"res_{r}_table.jsonl" for r in RESOLUTIONS}
SCRATCH_ROOT = ROOT / "out" / "atlas_probe" / "eff_res"

RHO_TARGET = 0.85       # theta preservation bar per resolution (prompt)
# effective sigma sets actually passed to each descent (linear in res, x res/768).
SIGMA_SET = {
    768: [16, 20, 24, 28, 32],
    512: [10.6667, 13.3333, 16, 18.6667, 21.3333],
    384: [8, 10, 12, 14, 16],
}


# --------------------------------------------------------------------------- #
# --score : k3 best-over-walk per run (reuse process_walk).
# --------------------------------------------------------------------------- #
def score_run(res: int, workers: int):
    pool_dir = RUN_DIR[res]
    if not (pool_dir / "pool.jsonl").exists():
        raise SystemExit(f"no pool for res {res} at {pool_dir}")
    # Point the reused machinery at THIS run's scratch (raw tiles + reframe tiles).
    ra.SCRATCH = SCRATCH_ROOT / f"res_{res}" / "_scratch"
    by_walk = ra.load_frames_by_walk(pool_dir)
    seeds = ra._seed_rows(by_walk)
    scorer = make_scorer(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    if TABLE[res].exists():
        for l in open(TABLE[res], encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["walk_id"])
    wids = [w for w in sorted(by_walk) if w not in done]
    print(f"[res {res}] scoring {len(wids)}/{len(by_walk)} walks (k3) -> {TABLE[res].name}"
          + (f"  ({len(done)} cached)" if done else ""))
    t0 = time.time()
    with open(TABLE[res], "a" if done else "w", encoding="utf-8") as jf:
        for i, wid in enumerate(wids):
            row = ra.process_walk(scorer, wid, by_walk[wid], seeds[wid], None, workers)
            # keep the fields we need (drop the verbose per-frame ones)
            out = {k: row[k] for k in ("walk_id", "seed_cx", "seed_cy", "seed_fw",
                                       "n_frames", "reward_k1", "reward_k3",
                                       "k3_argmax_depth", "raw_max")}
            jf.write(json.dumps(out) + "\n"); jf.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(wids):
                el = time.time() - t0
                eta = (len(wids) - i - 1) / ((i + 1) / el) if el > 0 else 0
                print(f"  [{i+1}/{len(wids)}] walk {wid} k3={row['reward_k3']:.3f}  "
                      f"{el:.0f}s, ETA {eta:.0f}s")
    print(f"[res {res}] done in {time.time()-t0:.0f}s")


def run_score(args):
    todo = [args.res] if args.res else RESOLUTIONS
    for r in todo:
        score_run(r, args.workers)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_table(res):
    rows = [json.loads(l) for l in open(TABLE[res], encoding="utf-8") if l.strip()]
    return {r["walk_id"]: r for r in rows}


def theta_by_bin(walk_val: dict, labels, counts, keep, wid_order):
    """Per-bin mean of walk_val over kept bins. walk_val keyed by walk_id."""
    n = len(counts)
    vals = np.array([walk_val[w] for w in wid_order])
    sums = np.bincount(labels, weights=vals, minlength=n)
    means = np.full(n, np.nan)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    bins = np.array([b for b in range(n) if keep[b]])
    return bins, means[bins]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# --analyze
# --------------------------------------------------------------------------- #
def run_analyze(args):
    tab = {r: load_table(r) for r in RESOLUTIONS if TABLE[r].exists()}
    if 768 not in tab:
        raise SystemExit("need the 768 baseline table; run --score first")
    have = sorted(tab, reverse=True)
    print(f"=== LEVER 2: DESCENT RESOLUTION :: tables for {have} ===")
    for r in have:
        print(f"  res {r}: {len(tab[r])} walks  sigma set {SIGMA_SET[r]}  "
              f"(eff sigma/768 = {SIGMA_SET[r][0]/16:.3f})")

    # common walk ids across all present resolutions (identical depth-1 seeds)
    common = set(tab[768])
    for r in have:
        common &= set(tab[r])
    common = sorted(common)
    print(f"  common walks (paired, identical depth-1 seed): {len(common)}")

    base = tab[768]
    cx = np.array([base[w]["seed_cx"] for w in common])
    cy = np.array([base[w]["seed_cy"] for w in common])

    # ---- verify pairing: depth-1 seeds bit-identical across res ----
    mism = 0
    for w in common:
        for r in have:
            if (round(tab[r][w]["seed_cx"], 10) != round(base[w]["seed_cx"], 10) or
                    round(tab[r][w]["seed_cy"], 10) != round(base[w]["seed_cy"], 10)):
                mism += 1
    print(f"  seed-identity check: {mism} mismatched (expect 0 -- per-walk-rng pairing)")

    # ---- per-seed paired Delta (the tight headline) ----
    print(f"\n[paired Delta]  k3_res - k3_768   (per walk, same depth-1 seed)")
    print(f"{'res':>5} {'mean':>8} {'median':>8} {'sd':>7} {'p10':>7} {'p90':>7} "
          f"{'|d|>0.1':>8}")
    k768 = np.array([base[w]["reward_k3"] for w in common])
    for r in have:
        if r == 768:
            continue
        kr = np.array([tab[r][w]["reward_k3"] for w in common])
        d = kr - k768
        print(f"{r:>5} {d.mean():>+8.4f} {np.median(d):>+8.4f} {d.std():>7.4f} "
              f"{np.percentile(d,10):>+7.3f} {np.percentile(d,90):>+7.3f} "
              f"{float((np.abs(d)>0.1).mean()):>8.2f}")

    # ---- grid + kept bins (structure grid over seeds) ----
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    bounds = (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)
    ncols, nrows, _ = choose_grid(cx, cy, bounds)
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    counts = np.bincount(labels, minlength=ncols * nrows)
    keep = counts >= 10
    print(f"\n[grid] {ncols}x{nrows}  bins>=10 walks: {int(keep.sum())}")

    # ---- per-bin Mandelbrot field-std (filament axis, from step0_coverage) ----
    cls = cov.classify_bins(bounds, ncols, nrows, counts, var_floor=np.inf)
    field_std_all = cls["field_std"]           # per bin id
    fs_keep = field_std_all[keep]

    # theta_768 baseline
    bins, th768 = theta_by_bin({w: base[w]["reward_k3"] for w in common},
                               labels, counts, keep, common)

    print(f"\n[theta preservation vs 768]  (per-bin mean k3, {int(keep.sum())} bins)")
    print(f"{'res':>5} {'rho':>7} {'meanResid':>10} {'residVsFieldStd':>16} "
          f"{'cost/node':>10}  verdict")
    results = {}
    for r in have:
        if r == 768:
            continue
        _, thr = theta_by_bin({w: tab[r][w]["reward_k3"] for w in common},
                              labels, counts, keep, common)
        rho = spearman(thr, th768)
        resid = thr - th768
        # filament veto: residual DARKENING (resid<0) correlated with high field-std.
        rvf = pearson(resid, fs_keep)
        cost = (r / 768.0) ** 2           # per-node render cost ~ quadratic in linear res
        results[r] = dict(rho=rho, mean_resid=float(resid.mean()),
                          resid_vs_fieldstd=rvf, cost=cost, resid=resid)
        # veto fires if darkening trends with filament bins: negative corr AND net drop
        veto = (rvf < -0.35 and resid.mean() < -0.01)
        tag = "VETO(filament)" if veto else ("rho<0.85" if rho < RHO_TARGET else "PASS")
        print(f"{r:>5} {rho:>7.3f} {resid.mean():>+10.4f} {rvf:>+16.3f} "
              f"{cost:>10.3f}  {tag}")

    # ---- filament veto detail per resolution ----
    order = np.argsort(fs_keep)
    for r in have:
        if r == 768:
            continue
        resid = results[r]["resid"]
        print(f"\n[filament detail res {r}] residual sorted by bin field-std "
              f"(filament bins last):")
        print(f"  hi-field-std 5 bins field_std {fs_keep[order][-5:].round(1)}")
        print(f"    residuals                 {resid[order][-5:].round(4)}")
        print(f"  lo-field-std 5 bins field_std {fs_keep[order][:5].round(1)}")
        print(f"    residuals                 {resid[order][:5].round(4)}")

    # ---- pick coarsest passing resolution ----
    passing = []
    for r in have:
        if r == 768:
            continue
        R = results[r]
        veto = (R["resid_vs_fieldstd"] < -0.35 and R["mean_resid"] < -0.01)
        if R["rho"] >= RHO_TARGET and not veto:
            passing.append(r)
    chosen_res = min(passing) if passing else 768   # min linear res = coarsest = cheapest
    print(f"\n[pick] coarsest res passing rho>={RHO_TARGET} + no filament veto: "
          f"{chosen_res}" + ("" if passing else "  (none passed -> stay at 768)"))

    _draw_paired(common, tab, base, have, chosen_res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "chosen_res": chosen_res,
        "per_res": {str(r): {k: results[r][k] for k in
                             ("rho", "mean_resid", "resid_vs_fieldstd", "cost")}
                    for r in results},
        "sigma_set_chosen": SIGMA_SET[chosen_res],
    }
    (OUT_DIR / "chosen_res.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[chosen res] = {chosen_res}  -> {OUT_DIR / 'chosen_res.json'}")
    _combined_report(chosen_res, results)


def _combined_report(chosen_res, results):
    chosen_D = 17
    try:
        chosen_D = json.loads((OUT_DIR / "chosen_D.json").read_text())["D"]
    except Exception:
        pass
    res_cost = (chosen_res / 768.0) ** 2
    print("\n" + "=" * 68)
    print("COMBINED RECOMMENDED CONFIG (Lever 1 depth x Lever 2 resolution)")
    print("=" * 68)
    print(f"  depth cap D      = {chosen_D}  (of max 17)")
    print(f"  node resolution  = {chosen_res}  (of 768)")
    print(f"  --sigma-band     = {','.join(str(s) for s in SIGMA_SET[chosen_res])}")
    print(f"    (dilation radius = sigma, isolation window = 2*sigma, local-max radius")
    print(f"     = 0.4*sigma, merge radius = 0.75*mean_sigma -- all sigma-derived, so")
    print(f"     the rescaled --sigma-band carries every pixel-unit constant.)")
    print(f"  dimensionless (UNCHANGED): occ-floor 0.321, black-cap 0.30")
    # speedup: node-render cost ~ res^2; depth cap trims walk length.
    # Lever-1 pool cost saving was reported separately; here give the node-render factor.
    print(f"\n  per-node render cost factor vs 768: {res_cost:.3f} "
          f"(~{1/res_cost:.2f}x faster per node render)")
    print(f"  depth cap saving is small (Lever 1: ~4% on the pool) -- resolution is the")
    print(f"  load-bearing lever. NET descent speedup ~ {1/res_cost:.2f}x.")


def _draw_paired(common, tab, base, have, chosen_res):
    from PIL import Image, ImageDraw
    S, PAD = 420, 60
    cols = [r for r in have if r != 768]
    W = PAD * (len(cols) + 1) + S * len(cols)
    H = S + 2 * PAD
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 10), "LEVER 2: per-seed paired k3 (res vs 768) -- identity line = no change",
           fill=(235, 235, 235))
    k768 = np.array([base[w]["reward_k3"] for w in common])
    lo, hi = float(k768.min()), float(k768.max())
    for ci, r in enumerate(cols):
        kr = np.array([tab[r][w]["reward_k3"] for w in common])
        lo = min(lo, kr.min()); hi = max(hi, kr.max())
    rng = (hi - lo) or 1.0
    for ci, r in enumerate(cols):
        ox = PAD + ci * (S + PAD)
        oy = PAD
        d.rectangle([ox, oy, ox + S, oy + S], outline=(80, 80, 90))
        d.line([ox, oy + S, ox + S, oy], fill=(70, 70, 80))
        kr = np.array([tab[r][w]["reward_k3"] for w in common])
        col = (250, 220, 90) if r == chosen_res else (120, 210, 250)
        for a, b in zip(k768, kr):
            x = ox + int((a - lo) / rng * S)
            y = oy + int((1 - (b - lo) / rng) * S)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
        d.text((ox, oy + S + 8), f"res {r}  vs 768"
               + ("  <- chosen" if r == chosen_res else ""), fill=col)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(OUT_DIR / "paired_scatter.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--res", type=int, default=0, help="(score) one resolution only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.score:
        run_score(args)
    elif args.analyze:
        run_analyze(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
