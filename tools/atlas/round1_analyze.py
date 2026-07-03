#!/usr/bin/env python
"""Atlas round-1 acceptance — the 3-arm analysis + verdict
(prompts/atlas-round1-proposer-acceptance-prompt.md).

Two axes + attribution, decomposed so the win (if any) is attributable:

  YIELD       per arm: k3 reward mean/median/high-tail fractions + CDF.
  SPREAD      per arm: (a) seed-space coverage of GOOD outcomes (distinct boundary-band
              bins), (b) OUTCOME-APPEARANCE DIVERSITY (decisive) — distinct clusters of
              the good outcomes' v5 penultimate embeddings at a fixed appearance
              distance, read AT MATCHED YIELD (subsample to equal good-count) so a
              yield win driven by one repeated location shows up as low diversity.
  ATTRIBUTION atlas arm split by exploit vs explore: do EXPLORE seeds produce
              high-value outcomes in appearance regions arm-1 never generated (genuine
              new territory) vs exploit re-finding known-good spots?

Decompositions: (2)v(1) = de-clustering benefit; (3)v(2) = value-targeting benefit.

  uv run python tools/atlas/round1_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from atlas import Atlas  # noqa: E402

D = ROOT / "data" / "atlas" / "round1"
ARMS = [("arm1", "current seeder"), ("arm2", "uniform-over-domain"), ("arm3", "atlas acquisition")]
GOOD = 1.0          # good-outcome k3 cut (E[ord] midpoint)
STRONG = 1.4        # strong-tail cut
COVER_NCOLS, COVER_NROWS = 14, 12   # seed-space coverage grid (matches step-0 structure grid)
# Fixed appearance-distance threshold on the v5 penultimate cosine geometry. The v5
# head is a QUALITY classifier, so all twilight_shifted frames land in a tight cone
# (pooled NN p99 ~0.011, pairwise median ~0.05): two frames within TAU cosine ARE the
# same appearance. Single-linkage chains that dense cone into one blob, so diversity is
# measured by GREEDY-LEADER dedup (near-duplicate survivors), which the sweep shows is
# stable in arm ORDERING across TAU in [0.005, 0.05].
TAU = 0.01
TAU_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05]


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_arm(arm: str) -> dict:
    # The embed npz carries everything (emb + rewards + seed geometry + tag), all in
    # one walk-aligned row order — no table re-join needed.
    z = np.load(D / f"{arm}_embed.npz", allow_pickle=False)
    emb = z["emb"].astype(np.float64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)   # L2 -> cosine geometry
    return {
        "k3": z["reward_k3"].astype(float), "k1": z["reward_k1"].astype(float),
        "reached": z["reached_depth"], "scx": z["seed_cx"].astype(float),
        "scy": z["seed_cy"].astype(float), "emb": emb,
        "tag": z["tag"].astype(str), "walk_id": z["walk_id"],
    }


# --------------------------------------------------------------------------- #
# diversity primitives (cosine distance = 1 - dot on L2-normed embeddings)
# --------------------------------------------------------------------------- #
def cos_dmat(emb: np.ndarray) -> np.ndarray:
    d = 1.0 - emb @ emb.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0)


def greedy_leaders(dmat: np.ndarray, order: np.ndarray, tau: float) -> int:
    """Near-duplicate-survivor count: walk `order` (reward-descending) and keep a point
    only if it is > tau (cosine) from every already-kept survivor. A max-ish tau-packing
    that, unlike single-linkage, does NOT chain a dense cloud into one cluster — so it
    counts genuinely distinct appearances. `order` indexes into `dmat`."""
    kept: list[int] = []
    for i in order:
        if all(dmat[i, j] > tau for j in kept):
            kept.append(int(i))
    return len(kept)


def matched_diversity(dmat_good: np.ndarray, k3_good: np.ndarray, tau: float,
                      m: int, reps: int = 200, seed: int = 0) -> dict:
    """Distinct-survivor count over the good outcomes at `tau`, plus a matched-count
    read: (a) deterministic top-`m`-by-reward subset, (b) mean over `reps` random
    `m`-subsets — both isolate diversity from raw good-count (the decisive at-matched-
    yield read). Order is reward-descending so the best representative of each
    appearance is the survivor."""
    n = len(k3_good)
    order = np.argsort(k3_good)[::-1]
    full = greedy_leaders(dmat_good, order, tau)
    out = {"n_good": n, "distinct_full": full,
           "distinct_per_good": (full / n) if n else 0.0}
    if n >= m and m > 0:
        top = np.argsort(k3_good)[::-1][:m]
        top_order = top[np.argsort(k3_good[top])[::-1]]
        out["distinct_topM"] = greedy_leaders(dmat_good, top_order, tau)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(reps):
            s = rng.choice(n, m, replace=False)
            s = s[np.argsort(k3_good[s])[::-1]]
            vals.append(greedy_leaders(dmat_good, s, tau))
        out["distinct_randM_mean"] = float(np.mean(vals))
        out["distinct_randM_sd"] = float(np.std(vals))
    else:
        out["distinct_topM"] = full
        out["distinct_randM_mean"] = float(full)
        out["distinct_randM_sd"] = 0.0
    return out


# --------------------------------------------------------------------------- #
# seed-space coverage
# --------------------------------------------------------------------------- #
def coverage_bins(scx, scy, mask_good, bounds):
    x0, x1, y0, y1 = bounds
    ix = np.clip(((scx - x0) / (x1 - x0) * COVER_NCOLS).astype(int), 0, COVER_NCOLS - 1)
    iy = np.clip(((scy - y0) / (y1 - y0) * COVER_NROWS).astype(int), 0, COVER_NROWS - 1)
    bid = iy * COVER_NCOLS + ix
    return len(set(bid[mask_good].tolist()))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def frac(a, t):
    return float((a >= t).mean())


def main():
    atlas = Atlas.load()
    bounds = atlas.mask_bounds
    data = {a: load_arm(a) for a, _ in ARMS}

    print("=" * 78)
    print("ATLAS ROUND-1 :: 3-ARM ACCEPTANCE  (k3 best-over-walk reward, v5 CORN)")
    print("=" * 78)
    print(f"good-outcome cut k3>={GOOD}  strong-tail k3>={STRONG}")
    print(f"appearance diversity = greedy-leader near-dup survivors on v5 penultimate "
          f"cosine geometry, TAU={TAU} (headline)")

    # ---------------- YIELD ----------------
    print("\n" + "-" * 78)
    print("YIELD  (per arm, over all injected walks — matched N)")
    print("-" * 78)
    print(f"{'arm':<22} {'N':>4} {'mean':>6} {'med':>6} {'p90':>6} "
          f"{'>=1.0':>6} {'>=1.4':>6} {'reachd':>7} {'prod%':>6}")
    yields = {}
    for a, name in ARMS:
        k3 = data[a]["k3"]
        reached = data[a]["reached"]
        prod = float((reached >= 2).mean())   # walk descended past depth-1
        yields[a] = dict(n=len(k3), mean=float(k3.mean()), med=float(np.median(k3)),
                         p90=float(np.percentile(k3, 90)), f10=frac(k3, GOOD),
                         f14=frac(k3, STRONG), reached=float(reached.mean()), prod=prod)
        y = yields[a]
        print(f"{name:<22} {y['n']:>4} {y['mean']:>6.3f} {y['med']:>6.3f} {y['p90']:>6.3f} "
              f"{y['f10']:>6.2f} {y['f14']:>6.2f} {y['reached']:>7.1f} {y['prod']:>6.2f}")

    def delta(b, a, key):
        return yields[b][key] - yields[a][key]

    print("\nDECOMPOSITION (yield):")
    print(f"  (2)v(1) de-clustering : d_mean {delta('arm2','arm1','mean'):+.3f}  "
          f"d_med {delta('arm2','arm1','med'):+.3f}  d(>=1.0) {delta('arm2','arm1','f10'):+.3f}  "
          f"d(>=1.4) {delta('arm2','arm1','f14'):+.3f}")
    print(f"  (3)v(2) value-target  : d_mean {delta('arm3','arm2','mean'):+.3f}  "
          f"d_med {delta('arm3','arm2','med'):+.3f}  d(>=1.0) {delta('arm3','arm2','f10'):+.3f}  "
          f"d(>=1.4) {delta('arm3','arm2','f14'):+.3f}")
    print(f"  (3)v(1) total         : d_mean {delta('arm3','arm1','mean'):+.3f}  "
          f"d_med {delta('arm3','arm1','med'):+.3f}  d(>=1.0) {delta('arm3','arm1','f10'):+.3f}  "
          f"d(>=1.4) {delta('arm3','arm1','f14'):+.3f}")

    # CDF (a few reward levels)
    print("\nCDF  P(k3 >= level):")
    levels = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6]
    print(f"  {'arm':<22} " + " ".join(f"{lv:>6.1f}" for lv in levels))
    for a, name in ARMS:
        k3 = data[a]["k3"]
        print(f"  {name:<22} " + " ".join(f"{frac(k3, lv):>6.2f}" for lv in levels))

    # ---------------- SPREAD ----------------
    print("\n" + "-" * 78)
    print(f"SPREAD  (good outcomes: k3>={GOOD})")
    print("-" * 78)
    # matched good-count = min across arms (so diversity is read at equal budget)
    good_counts = {a: int((data[a]["k3"] >= GOOD).sum()) for a, _ in ARMS}
    M = max(1, min(good_counts.values()))
    print(f"good counts: " + "  ".join(f"{a}={good_counts[a]}" for a, _ in ARMS) +
          f"   matched-count M={M}")
    # tau-sweep: show diversity-ordering across arms is stable in the threshold.
    print("\ndistinct-survivor sweep over TAU (full good set):")
    print(f"  {'TAU':>6} " + " ".join(f"{a:>6}" for a, _ in ARMS))
    for t in TAU_SWEEP:
        row = []
        for a, _ in ARMS:
            g = data[a]["k3"] >= GOOD
            row.append(greedy_leaders(cos_dmat(data[a]["emb"][g]),
                                      np.argsort(data[a]["k3"][g])[::-1], t))
        print(f"  {t:>6.3f} " + " ".join(f"{v:>6}" for v in row) +
              ("   <- headline" if t == TAU else ""))
    print(f"\n{'arm':<22} {'good':>5} {'cover_bins':>10} {'distinct':>9} "
          f"{'/good':>6} {'topM':>5} {'randM':>12}")
    spread = {}
    for a, name in ARMS:
        d = data[a]
        good = d["k3"] >= GOOD
        cov = coverage_bins(d["scx"], d["scy"], good, bounds)
        emb_g = d["emb"][good]
        k3_g = d["k3"][good]
        dm = cos_dmat(emb_g)
        div = matched_diversity(dm, k3_g, TAU, M)
        spread[a] = dict(cover=cov, **div)
        print(f"{name:<22} {int(good.sum()):>5} {cov:>10} {div['distinct_full']:>9} "
              f"{div['distinct_per_good']:>6.2f} {div['distinct_topM']:>5} "
              f"{div['distinct_randM_mean']:>6.1f}+-{div['distinct_randM_sd']:<4.1f}")

    print("\nDECOMPOSITION (diversity at matched count M, greedy-leader survivors @TAU):")
    print(f"  (2)v(1) de-clustering : d_randM {spread['arm2']['distinct_randM_mean']-spread['arm1']['distinct_randM_mean']:+.1f}  "
          f"d_cover {spread['arm2']['cover']-spread['arm1']['cover']:+d}")
    print(f"  (3)v(2) value-target  : d_randM {spread['arm3']['distinct_randM_mean']-spread['arm2']['distinct_randM_mean']:+.1f}  "
          f"d_cover {spread['arm3']['cover']-spread['arm2']['cover']:+d}")

    # ---------------- ATTRIBUTION (arm 3) ----------------
    print("\n" + "-" * 78)
    print("ATTRIBUTION  (atlas arm 3: exploit vs explore)")
    print("-" * 78)
    d3 = data["arm3"]
    tag = d3["tag"]
    attrib = {}
    for t in ("exploit", "explore"):
        sel = tag == t
        k3 = d3["k3"][sel]
        if len(k3) == 0:
            print(f"  {t}: (none)")
            continue
        attrib[t] = dict(n=int(sel.sum()), mean=float(k3.mean()), med=float(np.median(k3)),
                         f10=frac(k3, GOOD), f14=frac(k3, STRONG))
        print(f"  {t:<8} n={sel.sum():>3}  mean k3={k3.mean():.3f}  med={np.median(k3):.3f}  "
              f">=1.0 {frac(k3, GOOD):.2f}  >=1.4 {frac(k3, STRONG):.2f}")

    # NOVELTY vs arm-1: for each arm-3 GOOD outcome, min cosine dist to ALL arm-1
    # outcomes. Novel = appearance region arm-1 never generated (min dist > TAU).
    a1_emb = data["arm1"]["emb"]
    print(f"\n  novelty vs arm-1 (min-cos-dist > TAU={TAU:.4f} = appearance region arm-1 never made):")
    for t in ("exploit", "explore"):
        sel = (tag == t) & (d3["k3"] >= GOOD)
        if sel.sum() == 0:
            print(f"    {t}: 0 good outcomes")
            continue
        emb_g = d3["emb"][sel]
        mind = (1.0 - emb_g @ a1_emb.T).min(axis=1)
        novel = mind > TAU
        print(f"    {t:<8} good={int(sel.sum()):>3}  novel(new territory)={int(novel.sum()):>3} "
              f"({novel.mean()*100:>4.0f}%)  median min-dist to arm-1 {np.median(mind):.4f}")

    # ---------------- VERDICT ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    dc_mean = delta("arm2", "arm1", "mean")
    vt_mean = delta("arm3", "arm2", "mean")
    tot_mean = delta("arm3", "arm1", "mean")
    dc_div = spread["arm2"]["distinct_randM_mean"] - spread["arm1"]["distinct_randM_mean"]
    vt_div = spread["arm3"]["distinct_randM_mean"] - spread["arm2"]["distinct_randM_mean"]

    def tag_dir(x, eps=0.02):
        return "HELPS" if x > eps else ("HURTS" if x < -eps else "~neutral")

    print(f"  de-clustering  (2)v(1): yield {tag_dir(dc_mean)} ({dc_mean:+.3f}), "
          f"matched-diversity {tag_dir(dc_div,0.5)} ({dc_div:+.1f})")
    print(f"  value-target   (3)v(2): yield {tag_dir(vt_mean)} ({vt_mean:+.3f}), "
          f"matched-diversity {tag_dir(vt_div,0.5)} ({vt_div:+.1f})")
    print(f"  atlas total    (3)v(1): yield {tot_mean:+.3f}")
    if "explore" in attrib and "exploit" in attrib:
        print(f"  attribution: exploit re-mines known-good (mean {attrib['exploit']['mean']:.3f}); "
              f"explore mean {attrib['explore']['mean']:.3f} — see novelty split above for genuine discovery")

    # persist report
    report = {
        "config": {"good": GOOD, "strong": STRONG, "TAU": TAU, "tau_sweep": TAU_SWEEP,
                   "matched_count_M": M, "cover_grid": [COVER_NCOLS, COVER_NROWS]},
        "yield": yields, "spread": {a: spread[a] for a, _ in ARMS},
        "attribution": attrib,
        "decomposition": {
            "declustering_yield_dmean": dc_mean, "valuetarget_yield_dmean": vt_mean,
            "total_yield_dmean": tot_mean,
            "declustering_div_drandM": dc_div, "valuetarget_div_drandM": vt_div,
        },
    }
    (D / "round1_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  report -> {D / 'round1_report.json'}")


if __name__ == "__main__":
    main()
