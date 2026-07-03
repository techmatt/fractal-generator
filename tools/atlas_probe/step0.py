#!/usr/bin/env python
"""Step-0 atlas probe: does seed->outcome reward have exploitable spatial structure,
and is it stable under resampling?  (prompts/step0-atlas-probe-prompt.md)

This is a MEASUREMENT, not machinery. It builds no atlas. It runs the real
guided-descend engine (Mandelbrot, natural seeding), reframes each walk's TERMINAL
frame with the v5 location-quality classifier to get a per-walk reward, and answers
two go/no-go questions:

  1. STRUCTURE  -- does expected reward vary coherently with seed (cx,cy)
                   (one-way ANOVA eta^2 / ICC vs a label-shuffle null)?
  2. STABILITY  -- do per-bin mean rewards hold across resampled halves
                   (split-half Spearman rho)?

Reward = reframed terminal E[ord] (v5 CORN, continuous [0,2]). Reframing is
monotone-non-decreasing, so reward >= raw_terminal_score for every walk (asserted).

Reused verbatim (NOT reimplemented): reframe.reframe_location, probe.make_scorer
(v5), Location. Pool/walks parsed directly (harvest.py just json.loads each line).

Usage (uv):
  uv run python tools/atlas_probe/step0.py --time      # reframe 5 terminals, project total
  uv run python tools/atlas_probe/step0.py --build     # full reframe pass -> walks_table (BACKGROUND)
  uv run python tools/atlas_probe/step0.py --analyze    # structure + stability + PNGs + verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from reframe import reframe_location, Location  # noqa: E402
from probe import make_scorer  # noqa: E402

DESCEND_DIR = ROOT / "data" / "guided_descend" / "atlas_probe_step0"
OUT_DIR = ROOT / "data" / "atlas_probe" / "step0"          # DURABLE table
SCRATCH = ROOT / "out" / "atlas_probe" / "step0" / "_scratch"  # disposable reframe tiles
MODEL = "data/classifier/v5/model_best.pt"
TABLE_JSONL = OUT_DIR / "walks_table.jsonl"
TABLE_CSV = OUT_DIR / "walks_table.csv"
HEATMAP_PNG = OUT_DIR / "structure_heatmap.png"
SCATTER_PNG = OUT_DIR / "stability_scatter.png"
COMPLETE = OUT_DIR / "COMPLETE"

FIELDS = ["walk_id", "seed_cx", "seed_cy", "seed_fw", "term_cx", "term_cy", "term_fw",
          "reframed_cx", "reframed_cy", "reframed_fw", "reward", "raw_terminal_score",
          "terminal_depth"]


# --------------------------------------------------------------------------- #
# Load the guided-descend run: per-walk seed frame + terminal frame.
# --------------------------------------------------------------------------- #
def load_walks(descend_dir: Path) -> list[dict]:
    """Return one dict per walk: seed frame (depth==1 pool row) + terminal frame
    (max-depth pool row). Pool/walks are plain jsonl (one obj per line)."""
    pool_path = descend_dir / "pool.jsonl"
    if not pool_path.exists():
        raise SystemExit(f"missing {pool_path}; run guided-descend first")
    by_walk: dict[int, list[dict]] = {}
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_walk.setdefault(r["walk"], []).append(r)

    walks = []
    for wid, rows in sorted(by_walk.items()):
        rows.sort(key=lambda r: r["depth"])
        seed = next((r for r in rows if r["depth"] == 1), rows[0])
        term = max(rows, key=lambda r: r["depth"])
        walks.append({
            "walk_id": wid,
            "seed_cx": float(seed["cx"]), "seed_cy": float(seed["cy"]),
            "seed_fw": float(seed["fw"]),
            "term_cx": term["cx"], "term_cy": term["cy"], "term_fw": term["fw"],
            "terminal_depth": int(term["depth"]),
        })
    return walks


def _mand_location(cx, cy, fw) -> Location:
    # Parameter-plane Mandelbrot: c is the pixel, so no fixed Julia c.
    return Location(family="mandelbrot", c_re=None, c_im=None,
                    cx=str(cx), cy=str(cy), fw=str(fw), family_params={})


def reframe_walk(scorer, w: dict) -> dict:
    """Reframe a walk's terminal frame; return the assembled table row."""
    loc = _mand_location(w["term_cx"], w["term_cy"], w["term_fw"])
    wd = SCRATCH / f"walk_{w['walk_id']:04d}"
    res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=4)
    raw = res.trace["original_score"]
    reward = res.score
    # Monotone-non-decreasing sanity: a violation means the wrong scorer or a bug.
    if reward < raw - 1e-6:
        raise SystemExit(
            f"MONOTONICITY VIOLATED walk {w['walk_id']}: reward {reward:.4f} < raw {raw:.4f} "
            f"-- wrong scorer or reframe bug")
    return {
        "walk_id": w["walk_id"],
        "seed_cx": w["seed_cx"], "seed_cy": w["seed_cy"], "seed_fw": w["seed_fw"],
        "term_cx": float(w["term_cx"]), "term_cy": float(w["term_cy"]),
        "term_fw": float(w["term_fw"]),
        "reframed_cx": float(res.cx), "reframed_cy": float(res.cy),
        "reframed_fw": float(res.fw),
        "reward": reward, "raw_terminal_score": raw,
        "terminal_depth": w["terminal_depth"],
    }


# --------------------------------------------------------------------------- #
# --time : reframe 5 terminals, project the full-pass cost.
# --------------------------------------------------------------------------- #
def run_time(args):
    walks = load_walks(DESCEND_DIR)
    print(f"loaded {len(walks)} walks from {DESCEND_DIR}")
    scorer = make_scorer(MODEL)
    print(f"scorer: v5 CORN ({MODEL})  geometry={scorer.cfg.get('geometry')} "
          f"target={scorer.cfg.get('target')}")
    n = min(5, len(walks))
    per = []
    for w in walks[:n]:
        t = time.time()
        row = reframe_walk(scorer, w)
        el = time.time() - t
        per.append(el)
        print(f"  walk {w['walk_id']:>4} d{w['terminal_depth']} fw={float(w['term_fw']):.3e}: "
              f"{el:.2f}s  raw={row['raw_terminal_score']:.3f} -> reward={row['reward']:.3f}")
    avg = sum(per) / len(per)
    total = avg * len(walks)
    print(f"\n  avg {avg:.2f}s/walk  ->  PROJECTED {len(walks)} walks: "
          f"~{total:.0f}s (~{total/60:.1f} min)")
    print(f"  -> {'BACKGROUND recommended' if total > 30 else 'foreground OK'}")


# --------------------------------------------------------------------------- #
# --build : full reframe pass -> durable table (incremental jsonl + csv).
# --------------------------------------------------------------------------- #
def run_build(args):
    walks = load_walks(DESCEND_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and TABLE_JSONL.exists():
        with open(TABLE_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["walk_id"])
        print(f"[resume] {len(done)} walks already in {TABLE_JSONL}")
    todo = [w for w in walks if w["walk_id"] not in done]
    print(f"reframing {len(todo)} / {len(walks)} walks (v5) -> {TABLE_JSONL}")
    scorer = make_scorer(MODEL)

    mode = "a" if done else "w"
    t0 = time.time()
    with open(TABLE_JSONL, mode, encoding="utf-8") as jf:
        for i, w in enumerate(todo):
            row = reframe_walk(scorer, w)
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>4}/{len(todo)}] walk {w['walk_id']:>4} "
                      f"reward={row['reward']:.3f}  {el:.0f}s elapsed, ETA {eta:.0f}s")

    # Rebuild the CSV from the full jsonl (durable, sorted by walk_id).
    rows = [json.loads(l) for l in open(TABLE_JSONL, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    with open(TABLE_CSV, "w", newline="", encoding="utf-8") as cf:
        wtr = csv.DictWriter(cf, fieldnames=FIELDS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r[k] for k in FIELDS})
    COMPLETE.write_text(f"done {len(rows)} walks in {time.time()-t0:.0f}s\n")
    print(f"\nDONE {len(rows)} rows -> {TABLE_JSONL} + {TABLE_CSV}")


# --------------------------------------------------------------------------- #
# Stats helpers (no scipy dependency).
# --------------------------------------------------------------------------- #
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson on ranks (average ranks for ties)."""
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average tied ranks
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        start = csum - counts
        avg = (start + csum - 1) / 2.0
        return avg[inv]
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """One-way ANOVA eta^2 = SS_between / SS_total (fraction of variance explained
    by bin membership). ICC-flavoured: pure between-bin share of total variance."""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for lb in np.unique(labels):
        v = values[labels == lb]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total)


# --------------------------------------------------------------------------- #
# Binning: uniform spatial grid over (seed_cx, seed_cy); auto-pick resolution.
# --------------------------------------------------------------------------- #
def bin_seeds(cx: np.ndarray, cy: np.ndarray, ncols: int, nrows: int, bounds):
    x0, x1, y0, y1 = bounds
    ix = np.clip(((cx - x0) / (x1 - x0) * ncols).astype(int), 0, ncols - 1)
    iy = np.clip(((cy - y0) / (y1 - y0) * nrows).astype(int), 0, nrows - 1)
    return iy * ncols + ix  # flat bin id


CANDIDATE_GRIDS = [(14, 12), (12, 10), (10, 9), (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (4, 4)]


def choose_grid(cx, cy, bounds, min_per_bin=10):
    """Pick the finest grid whose #(bins with >=min_per_bin walks) is a few dozen."""
    best = None
    for ncols, nrows in CANDIDATE_GRIDS:
        labels = bin_seeds(cx, cy, ncols, nrows, bounds)
        counts = np.bincount(labels, minlength=ncols * nrows)
        n_pop = int((counts >= min_per_bin).sum())
        if 20 <= n_pop <= 60:
            return ncols, nrows, n_pop
        if best is None or abs(n_pop - 36) < abs(best[2] - 36):
            best = (ncols, nrows, n_pop)
    return best


def choose_grid_stability(cx, cy, bounds, min_total=20, want_bins=12):
    """Coarser grid for split-half: bins need to survive halving with >=10 walks in
    EACH half, i.e. >=~20 total. Pick the finest grid with >=want_bins such bins."""
    best = None
    for ncols, nrows in CANDIDATE_GRIDS:
        labels = bin_seeds(cx, cy, ncols, nrows, bounds)
        counts = np.bincount(labels, minlength=ncols * nrows)
        n_ok = int((counts >= min_total).sum())
        if n_ok >= want_bins:
            return ncols, nrows, n_ok
        if best is None or n_ok > best[2]:
            best = (ncols, nrows, n_ok)
    return best


# --------------------------------------------------------------------------- #
# PIL rendering (self-contained; no matplotlib).
# --------------------------------------------------------------------------- #
def _cmap(t: float) -> tuple[int, int, int]:
    """Perceptual-ish magma ramp for t in [0,1]."""
    anchors = [(0.0, (12, 8, 38)), (0.25, (85, 20, 110)), (0.5, (180, 55, 95)),
               (0.75, (245, 130, 55)), (1.0, (252, 235, 165))]
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    for i in range(len(anchors) - 1):
        t0, c0 = anchors[i]; t1, c1 = anchors[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(round(c0[j] + f * (c1[j] - c0[j]))) for j in range(3))
    return anchors[-1][1]


def draw_heatmap(cx, cy, reward, ncols, nrows, bounds, min_per_bin=10):
    from PIL import Image, ImageDraw
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    n = ncols * nrows
    sums = np.bincount(labels, weights=reward, minlength=n)
    counts = np.bincount(labels, minlength=n)
    means = np.full(n, np.nan)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    populated = counts >= min_per_bin
    vmin = np.nanmin(means[populated]) if populated.any() else np.nanmin(means[nz])
    vmax = np.nanmax(means[populated]) if populated.any() else np.nanmax(means[nz])
    rng = (vmax - vmin) or 1.0

    CELL, GUT_L, GUT_T, LEG = 78, 70, 70, 40
    W = GUT_L + ncols * CELL + 130
    H = GUT_T + nrows * CELL + LEG + 30
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 10), "STRUCTURE: mean reward (reframed E[ord]) by seed (cx,cy) bin",
           fill=(235, 235, 235))
    d.text((10, 28), f"grid {ncols}x{nrows}  scale [{vmin:.3f}..{vmax:.3f}]  "
           f"dim cell = <{min_per_bin} walks (n shown)", fill=(180, 180, 190))

    for r in range(nrows):
        for c in range(ncols):
            b = r * ncols + c
            x = GUT_L + c * CELL
            # invert row so cy increases upward
            y = GUT_T + (nrows - 1 - r) * CELL
            if counts[b] == 0:
                d.rectangle([x, y, x + CELL - 2, y + CELL - 2], fill=(30, 30, 36))
                continue
            t = (means[b] - vmin) / rng
            col = _cmap(t)
            if not populated[b]:
                col = tuple(int(ch * 0.35 + 8) for ch in col)  # dim sparse bins
            d.rectangle([x, y, x + CELL - 2, y + CELL - 2], fill=col)
            txt = (255, 255, 255) if t < 0.6 else (20, 20, 20)
            d.text((x + 4, y + 4), f"{means[b]:.2f}", fill=txt)
            d.text((x + 4, y + CELL - 16), f"n{int(counts[b])}", fill=txt)

    # axes
    d.text((GUT_L, GUT_T + nrows * CELL + 6),
           f"cx {bounds[0]:.2f}", fill=(170, 170, 180))
    d.text((GUT_L + ncols * CELL - 60, GUT_T + nrows * CELL + 6),
           f"{bounds[1]:.2f}", fill=(170, 170, 180))
    d.text((6, GUT_T), f"cy {bounds[3]:.2f}", fill=(170, 170, 180))
    d.text((6, GUT_T + nrows * CELL - 12), f"{bounds[2]:.2f}", fill=(170, 170, 180))

    # legend ramp
    lx = GUT_L + ncols * CELL + 24
    for i in range(120):
        t = i / 119
        col = _cmap(t)
        yy = GUT_T + nrows * CELL - int(t * (nrows * CELL))
        d.line([lx, yy, lx + 22, yy], fill=col)
    d.text((lx, GUT_T - 16), f"{vmax:.3f}", fill=(200, 200, 210))
    d.text((lx, GUT_T + nrows * CELL + 4), f"{vmin:.3f}", fill=(200, 200, 210))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(HEATMAP_PNG)
    return int((counts > 0).sum()), int(populated.sum())


def draw_scatter(xa, xb, rho, ncols, nrows):
    from PIL import Image, ImageDraw
    S, PAD = 460, 60
    W = H = S + 2 * PAD
    im = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(im)
    d.text((10, 10), "STABILITY: per-bin mean reward, half-A vs half-B (one split)",
           fill=(235, 235, 235))
    d.text((10, 28), f"grid {ncols}x{nrows}  bins>=10 in both: {len(xa)}  "
           f"Spearman rho={rho:.3f}", fill=(180, 180, 190))
    lo = min(xa.min(), xb.min()); hi = max(xa.max(), xb.max())
    rng = (hi - lo) or 1.0

    def px(v, flip=False):
        f = (v - lo) / rng
        return PAD + int((1 - f) * S) if flip else PAD + int(f * S)
    # frame + identity line
    d.rectangle([PAD, PAD, PAD + S, PAD + S], outline=(80, 80, 90))
    d.line([PAD, PAD + S, PAD + S, PAD], fill=(70, 70, 80))
    for a, b in zip(xa, xb):
        x = px(a); y = px(b, flip=True)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(120, 210, 250))
    d.text((PAD, PAD + S + 8), f"half-A  [{lo:.3f}..{hi:.3f}]", fill=(170, 170, 180))
    d.text((6, PAD), "half-B", fill=(170, 170, 180))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(SCATTER_PNG)


# --------------------------------------------------------------------------- #
# --analyze : structure + stability + PNGs + verdict.
# --------------------------------------------------------------------------- #
def run_analyze(args):
    if not TABLE_JSONL.exists():
        raise SystemExit(f"no {TABLE_JSONL}; run --build first")
    rows = [json.loads(l) for l in open(TABLE_JSONL, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    cx = np.array([r["seed_cx"] for r in rows])
    cy = np.array([r["seed_cy"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    raw = np.array([r["raw_terminal_score"] for r in rows])
    depth = np.array([r["terminal_depth"] for r in rows])
    n = len(rows)
    rng = np.random.default_rng(0)

    print(f"\n=== ATLAS PROBE STEP-0 :: {n} walks ===")
    assert (reward >= raw - 1e-6).all(), "monotonicity violated in table"
    print(f"reward (reframed E[ord]): mean {reward.mean():.3f}  sd {reward.std():.3f}  "
          f"range [{reward.min():.3f}, {reward.max():.3f}]")
    print(f"raw terminal E[ord]:      mean {raw.mean():.3f}  "
          f"(reframe lift mean {np.mean(reward-raw):+.3f})")
    print(f"terminal depth: min {depth.min()} max {depth.max()} mean {depth.mean():.1f}")

    # small margin around the data
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    bounds = (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)
    ncols, nrows, n_pop = choose_grid(cx, cy, bounds)
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    counts = np.bincount(labels, minlength=ncols * nrows)
    print(f"\n[grid] {ncols}x{nrows} over cx[{bounds[0]:.3f},{bounds[1]:.3f}] "
          f"cy[{bounds[2]:.3f},{bounds[3]:.3f}]  "
          f"populated bins (>=1): {(counts>0).sum()}  (>=10): {(counts>=10).sum()}")

    # ---- STRUCTURE: eta^2 restricted to >=10-walk bins vs shuffle null ----
    keep_bins = set(np.where(counts >= 10)[0].tolist())
    mask = np.array([lb in keep_bins for lb in labels])
    v_use, l_use = reward[mask], labels[mask]
    eta = eta_squared(v_use, l_use)
    NSH = 1000
    null = np.empty(NSH)
    for i in range(NSH):
        null[i] = eta_squared(v_use, rng.permutation(l_use))
    pct = float((null < eta).mean() * 100)
    z = float((eta - null.mean()) / (null.std() + 1e-12))
    p_val = float((null >= eta).mean())
    print(f"\n[STRUCTURE] eta^2 = {eta:.4f}  over {len(keep_bins)} bins, {mask.sum()} walks")
    print(f"  shuffle null (n={NSH}): mean {null.mean():.4f} sd {null.std():.4f} "
          f"p95 {np.quantile(null,0.95):.4f}")
    print(f"  observed percentile {pct:.1f}%  z={z:+.2f}  p(null>=obs)={p_val:.4f}")

    # ---- STABILITY: split-half Spearman over per-bin means ----
    # Coarser grid than STRUCTURE: split-half needs >=10 walks in EACH half per bin
    # (>=~20 total), which the fine structure grid (~15/bin) can't sustain.
    sc, sr, s_ok = choose_grid_stability(cx, cy, bounds)
    slabels = bin_seeds(cx, cy, sc, sr, bounds)
    scounts = np.bincount(slabels, minlength=sc * sr)
    MIN_BOTH = 10
    print(f"\n[stability grid] {sc}x{sr}  bins with >=20 walks: {int((scounts>=20).sum())} "
          f"(>=10-in-both feasible)")
    NSPLIT = 50
    splits = []   # (xa, xb, rho, ncommon)
    for s in range(NSPLIT):
        perm = rng.permutation(n)
        ha, hb = perm[: n // 2], perm[n // 2:]
        la, lb = slabels[ha], slabels[hb]
        ma = {b: reward[ha][la == b].mean() for b in np.unique(la)
              if (la == b).sum() >= MIN_BOTH}
        mb = {b: reward[hb][lb == b].mean() for b in np.unique(lb)
              if (lb == b).sum() >= MIN_BOTH}
        common = sorted(set(ma) & set(mb))
        if len(common) < 3:
            continue
        xa = np.array([ma[b] for b in common])
        xb = np.array([mb[b] for b in common])
        splits.append((xa, xb, spearman(xa, xb), len(common)))
    rhos = np.array([s[2] for s in splits])
    ncommon = [s[3] for s in splits]
    stab_med = float(np.median(rhos)) if len(rhos) else float("nan")
    # Representative scatter = the split whose rho is closest to the median.
    rep = min(splits, key=lambda t: abs(t[2] - stab_med)) if splits else None
    if len(rhos):
        print(f"[STABILITY] split-half Spearman over {len(rhos)}/{NSPLIT} usable splits "
              f"(bins>=10 in both; median {int(np.median(ncommon))} common bins/split)")
        print(f"  median rho {stab_med:.3f}  "
              f"IQR [{np.quantile(rhos,0.25):.3f}, {np.quantile(rhos,0.75):.3f}]  "
              f"range [{rhos.min():.3f}, {rhos.max():.3f}]")
    else:
        print(f"[STABILITY] NO usable splits even at {sc}x{sr} -- too few walks/bin")

    # ---- PNGs ----
    tot_pop, pop10 = draw_heatmap(cx, cy, reward, ncols, nrows, bounds)
    if rep is not None:
        draw_scatter(rep[0], rep[1], rep[2], sc, sr)

    # ---- coverage: sizeable empty regions ----
    empty = int((counts == 0).sum())
    print(f"\n[COVERAGE] {tot_pop}/{ncols*nrows} bins have >=1 seed, {pop10} have >=10; "
          f"{empty} empty bins (proposer gaps)")

    # ---- verdict ----
    struct_ok = (pct >= 95.0 and eta > null.mean() + 3 * null.std())
    stab_ok = (not math.isnan(stab_med)) and stab_med >= 0.5
    print("\n=== VERDICT ===")
    print(f"  STRUCTURE {'HOLDS' if struct_ok else 'WEAK/ABSENT'}: eta^2={eta:.4f} "
          f"at {pct:.1f}%ile (z={z:+.2f}) of shuffle null")
    print(f"  STABILITY {'HOLDS' if stab_ok else 'WEAK/ABSENT'}: median split-half "
          f"rho={stab_med:.3f}")
    if struct_ok and stab_ok:
        v = "ATLAS PREMISE HOLDS -- structured AND stable; a rankable seed atlas exists."
    elif not struct_ok and not stab_ok:
        v = "ATLAS PREMISE FAILS -- reward is spatial noise; no atlas to exploit."
    else:
        v = ("AMBIGUOUS -- one axis holds, one does not. Prime suspect: terminal-depth "
             "reward noise (single deepest frame is a noisy outcome estimate); the "
             "follow-up knob is best-over-walk reward instead of terminal-only.")
    print(f"  {v}")
    print(f"\n  table:   {TABLE_JSONL}")
    print(f"  heatmap: {HEATMAP_PNG}")
    print(f"  scatter: {SCATTER_PNG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", action="store_true", help="reframe 5 terminals, project total")
    ap.add_argument("--build", action="store_true", help="full reframe pass -> table")
    ap.add_argument("--analyze", action="store_true", help="structure+stability+PNGs+verdict")
    ap.add_argument("--resume", action="store_true", help="(build) skip walks already in table")
    args = ap.parse_args()
    if args.time:
        run_time(args)
    elif args.build:
        run_build(args)
    elif args.analyze:
        run_analyze(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
