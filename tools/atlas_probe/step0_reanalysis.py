#!/usr/bin/env python
"""Part B of the atlas re-analysis (prompts/atlas-reanalysis-prompt.md): replace the
step-0 TERMINAL-only reward with a BEST-OVER-WALK reward, computed off the frames
already in pool.jsonl (NO re-descend).

guided-descend's contract is *every frame visited*, scored downstream -- a walk's real
value is its best frame, wherever in the walk it lands. Terminal-only estimated the
wrong quantity and carried terminal-depth noise. Fix it:

  per walk: raw-score (v5, as-framed @ the reframe center fidelity) EVERY frame,
            take the top-3 by raw score, REFRAME those 3 (v5), and define
              reward_k1 = reframed score of the top-1-raw frame,
              reward_k3 = max reframed over the top-3.

The SEED is unchanged (still the depth-1 frame), so binning is identical to step-0 --
only the reward column changes. Reuses VERBATIM: reframe.reframe_location / _candidate
/ _render / RENDER_* (the raw center tile is the reframe fw-ladder x1.0 rung, so
raw==reframe.original_score and reframed>=raw holds), probe.make_scorer (v5),
step0.load_walks + the grid/stats machinery.

  uv run python tools/atlas_probe/step0_reanalysis.py --time      # 5 walks, project total
  uv run python tools/atlas_probe/step0_reanalysis.py --build     # full pass -> table (BACKGROUND)
  uv run python tools/atlas_probe/step0_reanalysis.py --analyze   # structure+stability, 3-way report
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import sys
import time
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

import step0  # noqa: E402  (reuse load_walks, grid + stats machinery, DESCEND_DIR, MODEL)
from step0 import (  # noqa: E402
    DESCEND_DIR, MODEL, bin_seeds, choose_grid, choose_grid_stability,
    eta_squared, spearman, load_walks,
)
from reframe import (  # noqa: E402
    reframe_location, Location, _candidate, _render, _tile_name,
    RENDER_W, RENDER_H, RENDER_SS,
)
from probe import make_scorer  # noqa: E402

OUT_DIR = ROOT / "data" / "atlas_probe" / "step0_reanalysis"       # DURABLE table
SCRATCH = ROOT / "out" / "atlas_probe" / "step0_reanalysis" / "_scratch"
STEP0_TABLE = ROOT / "data" / "atlas_probe" / "step0" / "walks_table.jsonl"

TABLE_JSONL = OUT_DIR / "walks_table_bestwalk.jsonl"
TABLE_CSV = OUT_DIR / "walks_table_bestwalk.csv"
HEATMAP_PNG = OUT_DIR / "structure_heatmap_bestwalk.png"
SCATTER_PNG = OUT_DIR / "stability_scatter_bestwalk.png"
COMPLETE = OUT_DIR / "COMPLETE"

KRAW = 3   # reframe the top-3 raw frames

FIELDS = ["walk_id", "seed_cx", "seed_cy", "seed_fw", "n_frames",
          "reward_k1", "reward_k3", "reward_terminal",
          "top1_idx", "top1_depth", "top1_raw", "top1_reframed",
          "k3_argmax_idx", "k3_argmax_depth", "raw_max", "raw_mean"]


# --------------------------------------------------------------------------- #
# Load pool -> all frames grouped by walk (NOT just seed+terminal).
# --------------------------------------------------------------------------- #
def load_frames_by_walk(descend_dir: Path) -> dict[int, list[dict]]:
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
    for wid in by_walk:
        by_walk[wid].sort(key=lambda r: r["depth"])
    return by_walk


def _mand_location(cx, cy, fw) -> Location:
    return Location(family="mandelbrot", c_re=None, c_im=None,
                    cx=str(cx), cy=str(cy), fw=str(fw), family_params={})


def _raw_tile_name(idx: int) -> str:
    return f"raw_{idx:05d}.jpg"


def raw_screen_walk(scorer, wid: int, frames: list[dict], workers: int,
                    loc_of=_mand_location) -> list[float]:
    """Render the reframe center tile (fw x1.0 rung, 640x360 ss2) for every frame and
    score it -- this is exactly reframe_location's `original_score` for that frame, so
    the later reframe of a top-k frame satisfies reframed >= this raw.

    `loc_of(cx, cy, fw) -> reframe.Location` is the per-family location factory
    (default `_mand_location`, byte-identical for every existing Mandelbrot caller).
    Pass a Julia/multibrot factory to route those families' frames through the SAME
    raw-screen render path -- `_render` reads the family off the Location via
    `render_one_flags`, so nothing else changes."""
    tiles = SCRATCH / f"walk_{wid:04d}" / "raw"
    tiles.mkdir(parents=True, exist_ok=True)
    to_render = []
    for fr in frames:
        out = tiles / _raw_tile_name(fr["idx"])
        if not out.exists():
            loc = loc_of(fr["cx"], fr["cy"], fr["fw"])
            c = _candidate(loc, 1.0, 0.0, 0.0)   # the x1.0 center rung
            to_render.append((loc, c, out))
    if to_render:
        fails = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_render, loc, c, out, RENDER_W, RENDER_H, RENDER_SS): out
                    for loc, c, out in to_render}
            for fut in cf.as_completed(futs):
                ok, err = fut.result()
                if not ok:
                    fails.append((futs[fut], err))
        if fails:
            out, err = fails[0]
            raise SystemExit(f"raw render failed ({len(fails)}) [{out.name}]: {err}")
    paths = [tiles / _raw_tile_name(fr["idx"]) for fr in frames]
    triples = scorer.score_paths(paths)
    return [float(t[0]) for t in triples]


def process_walk(scorer, wid: int, frames: list[dict], seed_row: dict,
                 term_reward: float | None, workers: int) -> dict:
    raws = raw_screen_walk(scorer, wid, frames, workers)
    order = sorted(range(len(frames)), key=lambda i: raws[i], reverse=True)
    topk = order[:KRAW]

    reframed = []   # (frame_index_in_list, raw, reframed_score, depth, idx)
    for rank, i in enumerate(topk):
        fr = frames[i]
        loc = _mand_location(fr["cx"], fr["cy"], fr["fw"])
        wd = SCRATCH / f"walk_{wid:04d}" / f"reframe_top{rank}"
        res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=workers)
        raw_orig = res.trace["original_score"]   # == raws[i] up to GPU nondeterminism
        # Monotone-non-decreasing by construction (x1.0 rung is in the search space).
        if res.score < raw_orig - 1e-4:
            raise SystemExit(
                f"MONOTONICITY VIOLATED walk {wid} frame idx {fr['idx']}: "
                f"reframed {res.score:.4f} < raw {raw_orig:.4f}")
        reframed.append((i, raw_orig, float(res.score), int(fr["depth"]), int(fr["idx"])))

    # reward_k1 = reframed score of the TOP-1 raw frame (rank 0)
    reward_k1 = reframed[0][2]
    top1 = reframed[0]
    # reward_k3 = max reframed over the top-3
    k3 = max(reframed, key=lambda t: t[2])
    reward_k3 = k3[2]

    return {
        "walk_id": wid,
        "seed_cx": float(seed_row["cx"]), "seed_cy": float(seed_row["cy"]),
        "seed_fw": float(seed_row["fw"]),
        "n_frames": len(frames),
        "reward_k1": reward_k1, "reward_k3": reward_k3,
        "reward_terminal": term_reward,
        "top1_idx": top1[4], "top1_depth": top1[3],
        "top1_raw": top1[1], "top1_reframed": top1[2],
        "k3_argmax_idx": k3[4], "k3_argmax_depth": k3[3],
        "raw_max": float(max(raws)), "raw_mean": float(np.mean(raws)),
    }


# --------------------------------------------------------------------------- #
# helpers: seed rows + terminal reward join
# --------------------------------------------------------------------------- #
def _seed_rows(by_walk):
    seeds = {}
    for wid, rows in by_walk.items():
        seeds[wid] = next((r for r in rows if r["depth"] == 1), rows[0])
    return seeds


def _terminal_rewards():
    if not STEP0_TABLE.exists():
        return {}
    out = {}
    for l in open(STEP0_TABLE, encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l)
            out[r["walk_id"]] = float(r["reward"])
    return out


# --------------------------------------------------------------------------- #
# --time
# --------------------------------------------------------------------------- #
def run_time(args):
    by_walk = load_frames_by_walk(DESCEND_DIR)
    seeds = _seed_rows(by_walk)
    term = _terminal_rewards()
    scorer = make_scorer(MODEL)
    print(f"loaded {len(by_walk)} walks / {sum(len(v) for v in by_walk.values())} frames")
    print(f"scorer: v5 CORN ({MODEL})  geometry={scorer.cfg.get('geometry')}")
    wids = sorted(by_walk)[:5]
    n_frames = 0
    per = []
    for wid in wids:
        frames = by_walk[wid]
        t = time.time()
        row = process_walk(scorer, wid, frames, seeds[wid], term.get(wid), args.workers)
        el = time.time() - t
        per.append(el); n_frames += len(frames)
        print(f"  walk {wid:>4} nframes={len(frames):>2}: {el:.2f}s  "
              f"raw_max={row['raw_max']:.3f}  k1={row['reward_k1']:.3f}  "
              f"k3={row['reward_k3']:.3f}  term={row['reward_terminal']}")
    avg = sum(per) / len(per)
    total_walks = len(by_walk)
    total = avg * total_walks
    fpw = n_frames / len(wids)
    print(f"\n  avg {avg:.2f}s/walk ({fpw:.1f} frames/walk raw + {KRAW} reframes)")
    print(f"  -> PROJECTED {total_walks} walks: ~{total:.0f}s (~{total/60:.1f} min) "
          f"at workers={args.workers}")
    print(f"  -> {'BACKGROUND recommended' if total > 30 else 'foreground OK'}")


# --------------------------------------------------------------------------- #
# --build (incremental jsonl + csv; resumable)
# --------------------------------------------------------------------------- #
def run_build(args):
    by_walk = load_frames_by_walk(DESCEND_DIR)
    seeds = _seed_rows(by_walk)
    term = _terminal_rewards()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and TABLE_JSONL.exists():
        for l in open(TABLE_JSONL, encoding="utf-8"):
            l = l.strip()
            if l:
                done.add(json.loads(l)["walk_id"])
        print(f"[resume] {len(done)} walks already done")
    wids = [w for w in sorted(by_walk) if w not in done]
    print(f"processing {len(wids)} / {len(by_walk)} walks (best-over-walk, k={KRAW}) "
          f"-> {TABLE_JSONL}")
    scorer = make_scorer(MODEL)

    mode = "a" if done else "w"
    t0 = time.time()
    with open(TABLE_JSONL, mode, encoding="utf-8") as jf:
        for i, wid in enumerate(wids):
            row = process_walk(scorer, wid, by_walk[wid], seeds[wid],
                               term.get(wid), args.workers)
            jf.write(json.dumps(row) + "\n"); jf.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(wids):
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(wids) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>4}/{len(wids)}] walk {wid:>4} "
                      f"k1={row['reward_k1']:.3f} k3={row['reward_k3']:.3f}  "
                      f"{el:.0f}s elapsed, ETA {eta:.0f}s")

    rows = [json.loads(l) for l in open(TABLE_JSONL, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    with open(TABLE_CSV, "w", newline="", encoding="utf-8") as cf_:
        wtr = csv.DictWriter(cf_, fieldnames=FIELDS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k) for k in FIELDS})
    COMPLETE.write_text(f"done {len(rows)} walks in {time.time()-t0:.0f}s\n")
    print(f"\nDONE {len(rows)} rows -> {TABLE_JSONL} + {TABLE_CSV}")


# --------------------------------------------------------------------------- #
# --analyze : structure + stability on k1 & k3, three-way vs terminal.
# --------------------------------------------------------------------------- #
def _structure(reward, labels, counts, rng, nsh=1000):
    keep = set(np.where(counts >= 10)[0].tolist())
    mask = np.array([lb in keep for lb in labels])
    v, l = reward[mask], labels[mask]
    eta = eta_squared(v, l)
    null = np.array([eta_squared(v, rng.permutation(l)) for _ in range(nsh)])
    z = float((eta - null.mean()) / (null.std() + 1e-12))
    pct = float((null < eta).mean() * 100)
    return eta, z, pct, int(len(keep)), int(mask.sum())


def _stability(reward, slabels, n, rng, nsplit=50, min_both=10):
    rhos = []
    splits = []
    for _ in range(nsplit):
        perm = rng.permutation(n)
        ha, hb = perm[: n // 2], perm[n // 2:]
        la, lb = slabels[ha], slabels[hb]
        ma = {b: reward[ha][la == b].mean() for b in np.unique(la)
              if (la == b).sum() >= min_both}
        mb = {b: reward[hb][lb == b].mean() for b in np.unique(lb)
              if (lb == b).sum() >= min_both}
        common = sorted(set(ma) & set(mb))
        if len(common) < 3:
            continue
        xa = np.array([ma[b] for b in common]); xb = np.array([mb[b] for b in common])
        r = spearman(xa, xb)
        rhos.append(r); splits.append((xa, xb, r, len(common)))
    rhos = np.array(rhos)
    med = float(np.median(rhos)) if len(rhos) else float("nan")
    rep = min(splits, key=lambda t: abs(t[2] - med)) if splits else None
    return rhos, med, rep, splits


def run_analyze(args):
    if not TABLE_JSONL.exists():
        raise SystemExit(f"no {TABLE_JSONL}; run --build first")
    rows = [json.loads(l) for l in open(TABLE_JSONL, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    cx = np.array([r["seed_cx"] for r in rows])
    cy = np.array([r["seed_cy"] for r in rows])
    k1 = np.array([r["reward_k1"] for r in rows])
    k3 = np.array([r["reward_k3"] for r in rows])
    term = np.array([r["reward_terminal"] if r["reward_terminal"] is not None else np.nan
                     for r in rows])
    n = len(rows)
    rng = np.random.default_rng(0)

    print(f"\n=== ATLAS RE-ANALYSIS (best-over-walk) :: {n} walks ===")
    print(f"reward_k1: mean {k1.mean():.3f} sd {k1.std():.3f} range [{k1.min():.3f},{k1.max():.3f}]")
    print(f"reward_k3: mean {k3.mean():.3f} sd {k3.std():.3f} range [{k3.min():.3f},{k3.max():.3f}]")
    if not np.isnan(term).all():
        print(f"reward_terminal (baseline): mean {np.nanmean(term):.3f} sd {np.nanstd(term):.3f}")
        print(f"  lift k1-term: mean {np.nanmean(k1-term):+.3f}   "
              f"k3-term: mean {np.nanmean(k3-term):+.3f}   "
              f"k3-k1: mean {np.nanmean(k3-k1):+.3f}")

    # SAME grid + bounds as step-0.
    mx = 0.02 * (cx.max() - cx.min() + 1e-9)
    my = 0.02 * (cy.max() - cy.min() + 1e-9)
    bounds = (cx.min() - mx, cx.max() + mx, cy.min() - my, cy.max() + my)
    ncols, nrows, _ = choose_grid(cx, cy, bounds)
    labels = bin_seeds(cx, cy, ncols, nrows, bounds)
    counts = np.bincount(labels, minlength=ncols * nrows)
    sc, sr, _ = choose_grid_stability(cx, cy, bounds)
    slabels = bin_seeds(cx, cy, sc, sr, bounds)
    print(f"\n[grid] structure {ncols}x{nrows} (>=10-walk bins {(counts>=10).sum()})  "
          f"stability {sc}x{sr}")

    # Baseline terminal numbers from the prompt (step-0's own run).
    TERM = {"eta": 0.30, "z": 14.8, "rho": 0.53}

    results = {}
    for name, rew in [("terminal", term), ("k1", k1), ("k3", k3)]:
        if name == "terminal" and np.isnan(rew).all():
            continue
        r = rew.copy()
        # (terminal recomputed here on the SAME machinery for an apples-to-apples row)
        rng2 = np.random.default_rng(0)
        eta, z, pct, nbins, nw = _structure(r, labels, counts, rng2)
        rng3 = np.random.default_rng(0)
        rhos, med, rep, splits = _stability(r, slabels, n, rng3)
        iqr = (float(np.quantile(rhos, 0.25)), float(np.quantile(rhos, 0.75))) if len(rhos) else (float("nan"),)*2
        results[name] = dict(eta=eta, z=z, pct=pct, nbins=nbins, nw=nw,
                             med=med, iqr=iqr, nsplit=len(rhos), rep=rep)

    # ---- three-way table ----
    print("\n=== THREE-WAY COMPARISON (same grid + machinery) ===")
    print(f"{'reward':<12} {'eta^2':>7} {'z':>7} {'pct':>6} {'med_rho':>8} "
          f"{'IQR':>18} {'splits':>7}")
    print(f"{'terminal*':<12} {TERM['eta']:>7.3f} {TERM['z']:>7.1f} {'--':>6} "
          f"{TERM['rho']:>8.3f} {'(step-0 run)':>18} {'--':>7}   <- prompt baseline")
    for name in ("terminal", "k1", "k3"):
        if name not in results:
            continue
        R = results[name]
        tag = " (recomputed)" if name == "terminal" else ""
        print(f"{name:<12} {R['eta']:>7.3f} {R['z']:>7.1f} {R['pct']:>5.1f}% "
              f"{R['med']:>8.3f} [{R['iqr'][0]:>6.3f},{R['iqr'][1]:>6.3f}] "
              f"{R['nsplit']:>7}{tag}")

    # ---- PNGs for the best variant (higher median rho) ----
    best = max(("k1", "k3"), key=lambda k: results[k]["med"])
    rew_best = k1 if best == "k1" else k3
    step0.OUT_DIR = OUT_DIR
    step0.HEATMAP_PNG = HEATMAP_PNG
    step0.SCATTER_PNG = SCATTER_PNG
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step0.draw_heatmap(cx, cy, rew_best, ncols, nrows, bounds)
    rep = results[best]["rep"]
    if rep is not None:
        step0.draw_scatter(rep[0], rep[1], rep[2], sc, sr)
    print(f"\n  best variant = {best}  -> heatmap {HEATMAP_PNG.name}, scatter {SCATTER_PNG.name}")

    # ---- verdict ----
    print("\n=== VERDICT B (reward definition) ===")
    for name in ("k1", "k3"):
        R = results[name]
        print(f"  {name}: eta^2={R['eta']:.3f} (z={R['z']:+.1f}, {R['pct']:.1f}%ile)  "
              f"median split-half rho={R['med']:.3f}")
    dk1 = results["k1"]["med"] - TERM["rho"]
    dk3 = results["k3"]["med"] - TERM["rho"]
    bestmed = results[best]["med"]
    print(f"\n  terminal baseline rho=0.53. best-over-walk: k1 {dk1:+.3f}, k3 {dk3:+.3f}")
    if bestmed >= 0.65:
        print(f"  -> BEST-OVER-WALK IS THE ATLAS REWARD (rho {bestmed:.3f} >= 0.65): fine-grained")
        print(f"     region ranking is viable; terminal depth WAS the stability limiter.")
    elif bestmed >= TERM["rho"] + 0.05:
        print(f"  -> PARTIAL LIFT (rho {bestmed:.3f}): best-over-walk helps but doesn't clear the")
        print(f"     0.65 bar; region ranking is borderline, v1 stays coarse budget-steering.")
    else:
        print(f"  -> MARGINAL (rho {bestmed:.3f} ~= 0.53): terminal depth was NOT the main stability")
        print(f"     limiter; v1 stays coarse-budget-steering-only, not fine region ranking.")
    dk = results["k3"]["med"] - results["k1"]["med"]
    print(f"  k3 vs k1: median rho {dk:+.3f}  -> "
          f"{'looking past top-1 raw is worth the extra reframes' if dk >= 0.02 else 'top-1 raw suffices (k3 not worth 2x reframes)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--resume", action="store_true", help="(build) skip walks already in table")
    ap.add_argument("--workers", type=int, default=6, help="parallel render-one workers")
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
