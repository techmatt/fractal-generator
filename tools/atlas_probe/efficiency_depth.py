#!/usr/bin/env python
"""Efficiency study — Lever 1 (depth cap), post-hoc on the existing step-0 pool.
(prompts/descent-efficiency-study-prompt.md)

The atlas will run MANY guided-descend walks to estimate regional value theta(seed).
Best frames are usually mid-walk, not terminal (Part B). If they cluster shallow, a
depth cap D trades wasted deep steps for more walks/budget WITHOUT losing theta.

This runs ENTIRELY on the 600-walk step-0 pool: every frame's raw center tile was
already rendered+scored during the best-over-walk re-analysis and cached under
out/atlas_probe/step0_reanalysis/_scratch/walk_XXXX/raw/raw_<idx>.jpg. We:

  --score   : re-score every cached raw tile -> per-frame table
              (walk, idx, depth, raw, seed_cx, seed_cy) in data/.../frames_raw.jsonl.
              (GPU inference only, NO rendering.)
  --analyze : best-frame-depth histogram+percentiles; raw theta_D sweep vs
              theta_uncapped (Spearman over >=10-walk bins); residual-vs-mean-best-
              depth VETO; pick shallowest D with rho>=0.95 AND flat residual; cost
              saving. (GPU-free, reads frames_raw.jsonl.)
  --confirm : reframed-k3 confirm at the chosen D (reframe the top-3-raw-<=D frames
              per walk where the capped top-3 differs from the uncapped top-3 already
              scored in the re-analysis table). (GPU.)

Reward semantics match the atlas reward (best-over-walk k3): raw-score every frame ->
top-3 raw -> reframe -> max reframed. The raw sweep uses raw-argmax as the faithful
k3~=k1 proxy (per the re-analysis), and --confirm upgrades the CHOSEN D to the proper
reframed k3.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
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
from step0 import bin_seeds, choose_grid, spearman, DESCEND_DIR  # noqa: E402
from step0_reanalysis import (  # noqa: E402
    SCRATCH as REANALYSIS_SCRATCH, load_frames_by_walk, _seed_rows, _mand_location,
    _raw_tile_name, TABLE_JSONL as REANALYSIS_TABLE,
)
from reframe import reframe_location  # noqa: E402
from active_ckpt import make_scorer  # noqa: E402

MODEL = "data/classifier/v5/model_best.pt"
OUT_DIR = ROOT / "data" / "atlas_probe" / "step0_efficiency"
FRAMES_JSONL = OUT_DIR / "frames_raw.jsonl"
CONFIRM_JSONL = OUT_DIR / "confirm_depthcap.jsonl"
HIST_PNG = OUT_DIR / "bestframe_depth_hist.png"
SWEEP_PNG = OUT_DIR / "depthcap_sweep.png"

RHO_TARGET = 0.95   # theta preservation bar for choosing D
KRAW = 3            # top-k raw frames reframed for the k3 reward (matches re-analysis)


# --------------------------------------------------------------------------- #
# --score : re-score every cached raw tile -> per-frame table.
# --------------------------------------------------------------------------- #
def run_score(args):
    by_walk = load_frames_by_walk(DESCEND_DIR)
    seeds = _seed_rows(by_walk)
    scorer = make_scorer(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gather every cached raw tile in pool order; assert the cache is complete.
    tile_paths, meta = [], []
    missing = 0
    for wid in sorted(by_walk):
        raw_dir = REANALYSIS_SCRATCH / f"walk_{wid:04d}" / "raw"
        s = seeds[wid]
        for fr in by_walk[wid]:
            p = raw_dir / _raw_tile_name(fr["idx"])
            if not p.exists():
                missing += 1
                continue
            tile_paths.append(p)
            meta.append((wid, fr["idx"], int(fr["depth"]),
                         float(s["cx"]), float(s["cy"])))
    total = sum(len(v) for v in by_walk.values())
    print(f"{len(tile_paths)}/{total} cached raw tiles found ({missing} missing) "
          f"over {len(by_walk)} walks")
    if missing:
        raise SystemExit(f"{missing} raw tiles missing -- re-run step0_reanalysis --build first")

    t0 = time.time()
    triples = scorer.score_paths(tile_paths, batch_size=128)
    print(f"scored {len(triples)} tiles in {time.time()-t0:.0f}s")

    with open(FRAMES_JSONL, "w", encoding="utf-8") as f:
        for (wid, idx, depth, scx, scy), (raw, nb, g) in zip(meta, triples):
            f.write(json.dumps({
                "walk": wid, "idx": idx, "depth": depth,
                "raw": float(raw), "seed_cx": scx, "seed_cy": scy,
            }) + "\n")
    print(f"wrote {FRAMES_JSONL}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_frames() -> dict[int, list[dict]]:
    if not FRAMES_JSONL.exists():
        raise SystemExit(f"no {FRAMES_JSONL}; run --score first")
    by_walk: dict[int, list[dict]] = {}
    for l in open(FRAMES_JSONL, encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            by_walk.setdefault(r["walk"], []).append(r)
    for wid in by_walk:
        by_walk[wid].sort(key=lambda r: r["depth"])
    return by_walk


def _grid(cx, cy):
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    bounds = (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)
    ncols, nrows, _ = choose_grid(cx, cy, bounds)
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    counts = np.bincount(labels, minlength=ncols * nrows)
    return bounds, ncols, nrows, labels, counts


def _theta_by_bin(per_walk_val, labels, counts, keep):
    """Mean per-walk value within each kept bin -> (bin_ids, theta)."""
    n = len(counts)
    sums = np.bincount(labels, weights=per_walk_val, minlength=n)
    means = np.full(n, np.nan)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    bins = [b for b in range(n) if keep[b]]
    return np.array(bins), means[np.array(bins)]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
# --analyze
# --------------------------------------------------------------------------- #
def run_analyze(args):
    by_walk = load_frames()
    wids = sorted(by_walk)

    # per-walk arrays
    seed_cx = np.array([by_walk[w][0]["seed_cx"] for w in wids])
    seed_cy = np.array([by_walk[w][0]["seed_cy"] for w in wids])
    term_depth = np.array([max(fr["depth"] for fr in by_walk[w]) for w in wids])
    raw_all = [np.array([fr["raw"] for fr in by_walk[w]]) for w in wids]
    depth_all = [np.array([fr["depth"] for fr in by_walk[w]]) for w in wids]
    # depth of the max-raw frame (the best-frame depth; k3~=k1 so raw-argmax is faithful)
    best_depth = np.array([depth_all[i][int(np.argmax(raw_all[i]))]
                           for i in range(len(wids))])
    raw_max = np.array([raw_all[i].max() for i in range(len(wids))])

    print(f"\n=== LEVER 1: DEPTH CAP :: {len(wids)} walks ===")
    print(f"terminal depth: min {term_depth.min()} max {term_depth.max()} "
          f"mean {term_depth.mean():.1f}")

    # ---- best-frame-depth distribution ----
    dmax = int(term_depth.max())
    hist = np.bincount(best_depth, minlength=dmax + 1)
    print("\n[best-frame depth] (depth of the max-raw frame per walk)")
    pcts = {p: float(np.percentile(best_depth, p)) for p in (50, 75, 90, 95)}
    print("  P50 {P50:.0f}  P75 {P75:.0f}  P90 {P90:.0f}  P95 {P95:.0f}".format(
        P50=pcts[50], P75=pcts[75], P90=pcts[90], P95=pcts[95]))
    print("  depth :  " + " ".join(f"{d:>3}" for d in range(1, dmax + 1)))
    print("  count :  " + " ".join(f"{hist[d]:>3}" for d in range(1, dmax + 1)))
    frac_le = {d: float((best_depth <= d).mean()) for d in range(1, dmax + 1)}
    print("  cumfrac: " + " ".join(f"{frac_le[d]:.2f}"[1:] for d in range(1, dmax + 1)))

    # ---- grid + kept bins ----
    bounds, ncols, nrows, labels, counts = _grid(seed_cx, seed_cy)
    keep = counts >= 10
    print(f"\n[grid] {ncols}x{nrows}  bins>=10 walks: {int(keep.sum())}")

    # theta_uncapped (== mean raw_max per bin)
    bins, theta_unc = _theta_by_bin(raw_max, labels, counts, keep)

    # per-bin mean best-frame depth (the veto axis)
    _, bin_mean_bestdepth = _theta_by_bin(best_depth.astype(float), labels, counts, keep)

    # ---- cap sweep ----
    print(f"\n[cap sweep] theta_D(bin) = mean_walk( max raw over frames depth<=D )")
    print(f"{'D':>3} {'rho':>7} {'meanResid':>10} {'maxDrop':>8} "
          f"{'residVsDepth':>13} {'cost':>6}  verdict")
    sweep = []
    for D in range(1, dmax + 1):
        capped_val = np.array([
            raw_all[i][depth_all[i] <= D].max() if (depth_all[i] <= D).any()
            else raw_all[i].min()
            for i in range(len(wids))
        ])
        _, theta_D = _theta_by_bin(capped_val, labels, counts, keep)
        rho = spearman(theta_D, theta_unc)
        resid = theta_D - theta_unc                     # <= 0 (capping can only drop)
        mean_resid = float(resid.mean())
        max_drop = float(resid.min())
        # veto axis: does capping darken DEEP-favoring bins? (residual more negative
        # where bin_mean_bestdepth is larger => negative correlation => bias)
        rvd = pearson(resid, bin_mean_bestdepth)
        cost = float(np.minimum(term_depth, D).mean() / term_depth.mean())
        sweep.append(dict(D=D, rho=rho, mean_resid=mean_resid, max_drop=max_drop,
                          resid_vs_depth=rvd, cost=cost))
        flagged = "rho>=.95" if rho >= RHO_TARGET else ""
        print(f"{D:>3} {rho:>7.3f} {mean_resid:>10.4f} {max_drop:>8.4f} "
              f"{rvd:>13.3f} {cost:>6.3f}  {flagged}")

    # ---- pick D: shallowest with rho>=target AND flat residual ----
    # "flat residual" := |resid_vs_depth correlation| small (no deep-region bias) AND
    # per-bin mean residual shallow. Require |rvd| < 0.35 (weak/no depth coupling).
    RVD_MAX = 0.35
    ok = [s for s in sweep if s["rho"] >= RHO_TARGET and abs(s["resid_vs_depth"]) < RVD_MAX]
    chosen = min(ok, key=lambda s: s["D"]) if ok else None
    print(f"\n[pick] shallowest D with rho>={RHO_TARGET} AND |resid-vs-depth|<{RVD_MAX}:")
    if chosen is None:
        # fall back to shallowest rho>=target regardless of veto, but FLAG the veto
        cand = [s for s in sweep if s["rho"] >= RHO_TARGET]
        chosen = min(cand, key=lambda s: s["D"]) if cand else sweep[-1]
        print(f"  !! no D passes BOTH gates; shallowest rho>=target = D={chosen['D']} "
              f"(resid-vs-depth={chosen['resid_vs_depth']:+.3f} -- CHECK VETO)")
    else:
        print(f"  D={chosen['D']}  rho={chosen['rho']:.3f}  "
              f"cost={chosen['cost']:.3f} (saves {(1-chosen['cost'])*100:.0f}%)  "
              f"resid-vs-depth={chosen['resid_vs_depth']:+.3f} (flat)")

    # ---- veto detail at chosen D ----
    D = chosen["D"]
    capped_val = np.array([
        raw_all[i][depth_all[i] <= D].max() if (depth_all[i] <= D).any()
        else raw_all[i].min() for i in range(len(wids))
    ])
    _, theta_D = _theta_by_bin(capped_val, labels, counts, keep)
    resid = theta_D - theta_unc
    order = np.argsort(bin_mean_bestdepth)
    print(f"\n[veto] residual (theta_D - theta_unc) across bins, sorted by "
          f"mean-best-depth (deep bins last):")
    print(f"  deepest-5 bins mean-best-depth {bin_mean_bestdepth[order][-5:].round(1)}")
    print(f"    their residuals            {resid[order][-5:].round(4)}")
    print(f"  shallowest-5 bins mean-best-depth {bin_mean_bestdepth[order][:5].round(1)}")
    print(f"    their residuals               {resid[order][:5].round(4)}")
    print(f"  Pearson(residual, bin-mean-best-depth) = {pearson(resid, bin_mean_bestdepth):+.3f}")
    print(f"  -> {'VETO: capping darkens deep-favoring bins' if pearson(resid, bin_mean_bestdepth) < -RVD_MAX else 'no heterogeneous depth bias'}")

    _draw_hist(best_depth, dmax, pcts)
    _draw_sweep(sweep, chosen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chosen_D.json").write_text(json.dumps(chosen, indent=2))
    print(f"\n[chosen D] = {chosen['D']}  -> {OUT_DIR / 'chosen_D.json'}")
    print(f"  cost saving = {chosen['cost']:.3f}  (mean(min(depth,D))/mean(depth); "
          f"saves {(1-chosen['cost'])*100:.0f}% of node renders on THIS pool)")
    print(f"  run --confirm to upgrade D={chosen['D']} to the reframed-k3 reward.")


def _draw_hist(best_depth, dmax, pcts):
    from PIL import Image, ImageDraw
    W, H, PAD = 720, 340, 46
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 8), "LEVER 1: best-frame (max-raw) depth distribution", fill=(235, 235, 235))
    hist = np.bincount(best_depth, minlength=dmax + 1)[1:dmax + 1]
    hmax = max(1, hist.max())
    bw = (W - 2 * PAD) / dmax
    for i, hc in enumerate(hist):
        x = PAD + i * bw
        bh = (H - 2 * PAD) * hc / hmax
        d.rectangle([x, H - PAD - bh, x + bw - 2, H - PAD], fill=(120, 200, 250))
        d.text((x + 2, H - PAD + 3), f"{i+1}", fill=(150, 150, 160))
        if hc:
            d.text((x + 2, H - PAD - bh - 12), f"{hc}", fill=(200, 200, 210))
    for p, col in [(50, (250, 220, 90)), (90, (250, 130, 80))]:
        xv = PAD + (pcts[p] - 0.5) * bw
        d.line([xv, PAD, xv, H - PAD], fill=col)
        d.text((xv + 2, PAD), f"P{p}", fill=col)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(HIST_PNG)


def _draw_sweep(sweep, chosen):
    from PIL import Image, ImageDraw
    W, H, PADL, PADB, PADT = 720, 360, 56, 44, 40
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 8), "LEVER 1: theta preservation vs depth cap D (rho + cost)",
           fill=(235, 235, 235))
    Ds = [s["D"] for s in sweep]
    xw = (W - PADL - 20) / max(1, len(Ds) - 1)

    def X(i): return PADL + i * xw

    def Y(v, lo, hi): return PADT + (1 - (v - lo) / (hi - lo)) * (H - PADT - PADB)
    # rho curve (0.5..1.0)
    d.line([PADL, Y(0.95, 0.5, 1.0), W - 20, Y(0.95, 0.5, 1.0)], fill=(80, 120, 80))
    d.text((W - 60, Y(0.95, 0.5, 1.0) - 12), "rho=.95", fill=(120, 200, 120))
    for i in range(len(sweep) - 1):
        d.line([X(i), Y(sweep[i]["rho"], 0.5, 1.0),
                X(i + 1), Y(sweep[i + 1]["rho"], 0.5, 1.0)], fill=(120, 210, 250), width=2)
        d.line([X(i), Y(sweep[i]["cost"], 0.0, 1.0),
                X(i + 1), Y(sweep[i + 1]["cost"], 0.0, 1.0)], fill=(250, 170, 90), width=2)
    for i, s in enumerate(sweep):
        d.text((X(i) - 4, H - PADB + 6), f"{s['D']}", fill=(150, 150, 160))
    cx = X(Ds.index(chosen["D"]))
    d.line([cx, PADT, cx, H - PADB], fill=(250, 220, 90))
    d.text((cx + 3, PADT), f"D*={chosen['D']}", fill=(250, 220, 90))
    d.text((PADL, H - 16), "blue=rho[.5,1]  orange=cost[0,1]  yellow=chosen",
           fill=(170, 170, 180))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(SWEEP_PNG)


# --------------------------------------------------------------------------- #
# --confirm : reframed-k3 at chosen D.
# --------------------------------------------------------------------------- #
def _reanalysis_rows():
    out = {}
    if REANALYSIS_TABLE.exists():
        for l in open(REANALYSIS_TABLE, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                out[r["walk_id"]] = r
    return out


def run_confirm(args):
    chosen_path = OUT_DIR / "chosen_D.json"
    if not chosen_path.exists():
        raise SystemExit("run --analyze first (need chosen_D.json)")
    D = args.cap if args.cap else json.loads(chosen_path.read_text())["D"]
    conf_path = OUT_DIR / f"confirm_depthcap_D{D}.jsonl"
    print(f"[confirm] reframed-k3 preservation at D={D} -> {conf_path.name}")

    frames = load_frames()
    pool_by_walk = load_frames_by_walk(DESCEND_DIR)   # for cx/cy/fw per idx
    rean = _reanalysis_rows()                          # uncapped reward_k3 baseline
    scorer = make_scorer(MODEL)
    wids = sorted(frames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if args.resume and conf_path.exists():
        for l in open(conf_path, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                done[r["walk"]] = r
        print(f"[resume] {len(done)} walks already confirmed")

    def top3_set(frs, mask):
        idx = np.where(mask)[0]
        raws = np.array([frs[i]["raw"] for i in idx])
        return set(idx[np.argsort(raws)[::-1]][:KRAW].tolist())

    t0 = time.time()
    reused = reframed_n = 0
    with open(conf_path, "a" if done else "w", encoding="utf-8") as jf:
        for n, wid in enumerate(wids):
            if wid in done:
                continue
            frs = frames[wid]
            raws = np.array([fr["raw"] for fr in frs])
            depths = np.array([fr["depth"] for fr in frs])
            le = depths <= D
            if not le.any():
                le = depths == depths.min()
            capped = top3_set(frs, le)
            base = rean.get(wid, {})
            # SHORTCUT: if the capped top-3-raw == the uncapped top-3-raw (all <=D),
            # the reframed-k3 is unchanged -> reuse the re-analysis cached reward_k3.
            if (base.get("reward_k3") is not None
                    and capped == top3_set(frs, np.ones(len(frs), bool))):
                reward_k3_D = float(base["reward_k3"])
                reused += 1
                top_depths = [int(depths[i]) for i in sorted(capped)]
            else:
                idx_le = np.where(le)[0]
                order = idx_le[np.argsort(raws[idx_le])[::-1]][:KRAW]
                pool_rows = {r["idx"]: r for r in pool_by_walk[wid]}
                reframed = []
                for fi in order:
                    gidx = frs[fi]["idx"]
                    pr = pool_rows[gidx]
                    loc = _mand_location(pr["cx"], pr["cy"], pr["fw"])
                    wd = (ROOT / "out" / "atlas_probe" / "step0_efficiency" / "_confirm"
                          / f"walk_{wid:04d}" / f"idx_{gidx}")
                    res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd,
                                           workers=args.workers)
                    reframed.append(float(res.score))
                reward_k3_D = max(reframed)
                reframed_n += 1
                top_depths = [int(depths[fi]) for fi in order]
            row = {
                "walk": wid, "D": D,
                "reward_k3_capped": reward_k3_D,
                "reward_k3_uncapped": base.get("reward_k3"),
                "seed_cx": frs[0]["seed_cx"], "seed_cy": frs[0]["seed_cy"],
                "capped_top_depths": top_depths,
            }
            jf.write(json.dumps(row) + "\n"); jf.flush()
            if (n + 1) % 50 == 0 or n + 1 == len(wids):
                el = time.time() - t0
                print(f"  [{n+1}/{len(wids)}] reused {reused} / reframed {reframed_n} "
                      f"({el:.0f}s)")

    # ---- preservation stats ----
    rows = [json.loads(l) for l in open(conf_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["reward_k3_uncapped"] is not None]
    cx = np.array([r["seed_cx"] for r in rows])
    cy = np.array([r["seed_cy"] for r in rows])
    cap = np.array([r["reward_k3_capped"] for r in rows])
    unc = np.array([r["reward_k3_uncapped"] for r in rows])
    bounds, ncols, nrows, labels, counts = _grid(cx, cy)
    keep = counts >= 10
    _, th_cap = _theta_by_bin(cap, labels, counts, keep)
    _, th_unc = _theta_by_bin(unc, labels, counts, keep)
    rho = spearman(th_cap, th_unc)
    print(f"\n=== REFRAMED-k3 CONFIRM at D={D} :: {len(rows)} walks ===")
    print(f"  per-walk k3: capped mean {cap.mean():.3f}  uncapped mean {unc.mean():.3f}  "
          f"delta {np.mean(cap-unc):+.4f}")
    print(f"  per-bin theta Spearman rho(capped, uncapped) = {rho:.3f}  "
          f"over {int(keep.sum())} bins")
    print(f"  -> {'REFRAMED theta PRESERVED at D' if rho >= RHO_TARGET else 'reframed theta NOT preserved -- back D off'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--cap", type=int, default=0, help="(confirm) override chosen D")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.score:
        run_score(args)
    elif args.analyze:
        run_analyze(args)
    elif args.confirm:
        run_confirm(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
