#!/usr/bin/env python
"""v7 evaluation — paired DeLong + bootstrap CIs on the frozen eval scores.

Reads data/classifier/v7/eval_scores_v7.jsonl (v6_* and v7_* columns per eval location,
frozen by train_v7 — nothing is re-scored here) and applies the pre-registered v7
acceptance battery:

  * PRIMARY — census-144 (source=prospect_census): AUC(q3 vs rest) for v6 and v7, each
    with a 5000-boot 95% CI, and the **paired DeLong** p (both scored on the same 144).
    Pre-registered bucket rule: 0.55-0.65 == indistinguishable from v6's 0.571 -> "label
    more, not v7 failed"; credible win ~= 0.68.
  * NON-REGRESSION — the frozen v6 eval's 942 mandelbrot + 178 J0 julia: paired DeLong
    (same locations, v6 vs v7), require v7 non-inferior.

Score used = the monotone rank score `*_score` (Sigma sigma(logit_k)), the same instrument
the census readout reported AUC 0.571 on. `--use-pgood` switches to P(good)=sigma(logit_good).

  uv run python tools/v7/eval_delong.py

DeLong: fast algorithm (Sun & Xu 2014); the paired covariance uses the correlation
between v6 and v7 scores, which is the whole point (maximizes power on the same samples).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "classifier" / "v7" / "eval_scores_v7.jsonl"
OUT = ROOT / "data" / "classifier" / "v7" / "eval_delong.json"
CENSUS_SOURCE = "prospect_census"
V6_CENSUS_BASELINE = 0.571          # census readout, for the bucket rule


# ------------------------- fast DeLong (Sun & Xu 2014) ------------------------- #
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first. Returns (aucs[k], cov[k,k])."""
    k, n_tot = preds_sorted.shape
    n = n_tot - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, n_tot])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_paired(y, s1, s2):
    """AUCs of (s1, s2) vs binary y + two-sided paired DeLong p for AUC1==AUC2."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")            # positives (1) first, stable
    m = int(y.sum())
    preds = np.vstack((np.asarray(s1, float), np.asarray(s2, float)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T).item())
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), p


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n_boot=5000, seed=0):
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        out.append(auc(yb, s[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def slice_block(rows, name, score_key):
    y = np.array([1 if r["label"] == 3 else 0 for r in rows])
    s6 = np.array([r[f"v6_{score_key}"] for r in rows])
    s7 = np.array([r[f"v7_{score_key}"] for r in rows])
    a6, a7, z, p = delong_paired(y, s6, s7)
    ci6 = boot_ci(y, s6); ci7 = boot_ci(y, s7)
    return {
        "name": name, "n": len(rows), "n_q3": int(y.sum()),
        "auc_v6": round(a6, 4), "auc_v6_ci95": [round(ci6[0], 4), round(ci6[1], 4)],
        "auc_v7": round(a7, 4), "auc_v7_ci95": [round(ci7[0], 4), round(ci7[1], 4)],
        "delta": round(a7 - a6, 4), "delong_z": round(z, 3), "delong_p": round(p, 4),
    }


def bucket(auc_v7):
    if auc_v7 is None:
        return "undetermined"
    if auc_v7 >= 0.68:
        return "CREDIBLE WIN (>=0.68)"
    if 0.55 <= auc_v7 <= 0.65:
        return "INDISTINGUISHABLE from v6 0.571 -> LABEL MORE (not v7 failed)"
    if auc_v7 > 0.65:
        return "PROMISING (0.65-0.68) — check paired p"
    return "BELOW v6 band (<0.55)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-pgood", action="store_true",
                    help="rank on P(good)=sigma(logit_good) instead of the Sigma-sigma score")
    args = ap.parse_args()
    score_key = "p_good" if args.use_pgood else "score"

    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    census = [r for r in rows if r["source"] == CENSUS_SOURCE]
    mandel = [r for r in rows if r["fractal_type"] == "mandelbrot"]
    j0 = [r for r in rows if r["fractal_type"] == "julia"]

    print("=" * 78)
    print(f"v7 EVAL — paired DeLong  (score = v*_{score_key})")
    print("=" * 78)

    result = {"score_key": score_key}

    print("\n--- PRIMARY: census-144 (julia:multibrot, q3 vs rest) ---")
    cb = slice_block(census, "census", score_key)
    result["census"] = cb
    result["census"]["bucket"] = bucket(cb["auc_v7"])
    print(f"  n={cb['n']} (q3={cb['n_q3']})")
    print(f"  v6 AUC {cb['auc_v6']} CI{cb['auc_v6_ci95']}   (readout baseline {V6_CENSUS_BASELINE})")
    print(f"  v7 AUC {cb['auc_v7']} CI{cb['auc_v7_ci95']}")
    print(f"  paired DeLong: delta={cb['delta']:+.4f}  z={cb['delong_z']}  p={cb['delong_p']}")
    print(f"  >>> BUCKET: {result['census']['bucket']}")

    print("\n--- NON-REGRESSION (paired DeLong, same locations) ---")
    result["non_regression"] = {}
    for rows_s, nm in [(mandel, "mandelbrot"), (j0, "j0_julia")]:
        b = slice_block(rows_s, nm, score_key)
        # non-inferiority read: v7 worse only if delta<0 AND significant
        b["verdict"] = ("REGRESSION" if (b["delta"] < 0 and b["delong_p"] < 0.05)
                        else "non-inferior")
        result["non_regression"][nm] = b
        print(f"  {nm:10s} n={b['n']:4d} q3={b['n_q3']:3d}  "
              f"v6 {b['auc_v6']} / v7 {b['auc_v7']}  delta={b['delta']:+.4f} "
              f"p={b['delong_p']}  -> {b['verdict']}")

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
