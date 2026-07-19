#!/usr/bin/env python
"""Grid + leave-one-batch-out evaluation for the location preference ranker v0.

Pre-registered (see prompts/build_pref_loc_ranker_v0.md):
  * Leave-one-batch-out: train run2 -> eval dive, and train dive -> eval run2. The corpus prior
    (v7-only, prior.npz) is available in BOTH folds; it never appears in eval.
  * Metrics per eval fold: Spearman vs human score, AUC good-vs-rest, precision@10.
  * Baselines that must be beaten: canonical p_good (`canon_pgood`) and random.
  * Interpretation rule: a middling result means "label more," not "the approach failed."

Feature blocks (frozen): morph-CLIP (768), v7 penultimate (1280), colored-CLIP (768). Heads:
Bradley-Terry pairwise logistic on within-batch score-ordered pairs, ridge on the raw score, and
logistic good-vs-rest. Regularization strength is chosen per (fold, config) by inner 5-fold CV on
the TRAIN batch only (leak-free); the family of strengths is heavy by construction.

Winner = max mean-LOBO Spearman (tie-break mean AUC), refit on all 81 labeled -> deployed artifact
`data/ranker/pref_loc_v0/model.npz` + `metrics.json`. Headline Spearman carries a within-fold
permutation p and a bootstrap CI (n small: n=81).

    uv run python -m tools.ranker.train_eval
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "ranker" / "pref_loc_v0"
BLOCKS = ("morph", "v7", "colored")
FEATURE_SETS = [
    ("morph",), ("v7",), ("colored",),
    ("morph", "colored"), ("morph", "v7"), ("v7", "colored"),
    ("morph", "v7", "colored"),
]
HEADS = ["bt", "ridge", "logi"]
RIDGE_ALPHAS = [30.0, 100.0, 300.0, 1000.0, 3000.0]     # heavy by construction
LOGI_CS = [0.003, 0.01, 0.03, 0.1]                       # small C == strong L2
PRIOR_WEIGHT = 0.3                                       # per-row weight on appended corpus rows
SEED = 0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load():
    z = np.load(OUT_DIR / "features.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    pri = OUT_DIR / "prior.npz"
    prior = {k: v for k, v in np.load(pri, allow_pickle=True).items()} if pri.exists() else None
    return d, prior


def stack_blocks(d, sets, idx):
    return np.concatenate([d[b][idx] for b in sets], axis=1).astype(np.float64)


# --------------------------------------------------------------------------- #
# heads: each returns a per-item real-valued rank score on Xe (higher == better)
# --------------------------------------------------------------------------- #
def fit_bt(Xtr, ytr, reg, sw=None, prior=None):
    # pairwise differences within the training batch (score order strict). prior pairs appended.
    def pairs(X, y, w):
        di, lab, ww = [], [], []
        n = len(y)
        for i in range(n):
            for j in range(n):
                if y[i] > y[j]:
                    di.append(X[i] - X[j]); lab.append(1); ww.append(w[i] * w[j] if w is not None else 1.0)
        return di, lab, ww
    w = np.ones(len(ytr)) if sw is None else sw
    di, lab, ww = pairs(Xtr, ytr, w)
    if prior is not None:
        Xp, yp = prior
        pdi, plab, pww = pairs(Xp, yp, np.full(len(yp), PRIOR_WEIGHT))
        di += pdi; lab += plab; ww += pww
    di = np.asarray(di); lab = np.asarray(lab); ww = np.asarray(ww)
    # symmetrize so the intercept-free logistic is balanced
    di = np.concatenate([di, -di]); lab = np.concatenate([lab, 1 - lab]); ww = np.concatenate([ww, ww])
    clf = LogisticRegression(C=reg, fit_intercept=False, max_iter=2000)
    clf.fit(di, lab, sample_weight=ww)
    return lambda Xe: clf.decision_function(Xe)


def fit_ridge(Xtr, ytr, reg, sw=None, prior=None):
    X, y, w = Xtr, ytr.astype(float), (np.ones(len(ytr)) if sw is None else sw.copy())
    if prior is not None:
        Xp, yp = prior
        X = np.concatenate([X, Xp]); y = np.concatenate([y, yp.astype(float)])
        w = np.concatenate([w, np.full(len(yp), PRIOR_WEIGHT)])
    m = Ridge(alpha=reg).fit(X, y, sample_weight=w)
    return lambda Xe: m.predict(Xe)


def fit_logi(Xtr, ytr, reg, sw=None, prior=None):
    X, y, w = Xtr, (ytr == 3).astype(int), (np.ones(len(ytr)) if sw is None else sw.copy())
    if prior is not None:
        Xp, yp = prior
        X = np.concatenate([X, Xp]); y = np.concatenate([y, (yp == 3).astype(int)])
        w = np.concatenate([w, np.full(len(yp), PRIOR_WEIGHT)])
    if len(np.unique(y)) < 2:
        return lambda Xe: np.zeros(len(Xe))
    clf = LogisticRegression(C=reg, max_iter=2000, class_weight="balanced")
    clf.fit(X, y, sample_weight=w)
    return lambda Xe: clf.decision_function(Xe)


FITTERS = {"bt": fit_bt, "ridge": fit_ridge, "logi": fit_logi}
REGGRID = {"bt": LOGI_CS, "ridge": RIDGE_ALPHAS, "logi": LOGI_CS}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(pred, human):
    sp = spearmanr(pred, human).correlation
    good = (human == 3).astype(int)
    auc = roc_auc_score(good, pred) if len(np.unique(good)) == 2 else np.nan
    order = np.argsort(-pred)
    k = min(10, len(pred))
    p_at = good[order[:k]].mean()
    return dict(spearman=float(sp), auc=float(auc), p_at_10=float(p_at), k=k,
                base_good=float(good.mean()))


def inner_cv_reg(fitter, Xtr, ytr, regs, prior, rng, folds=5):
    """Pick reg maximizing pooled inner-CV Spearman on the train batch (leak-free)."""
    n = len(ytr)
    idx = rng.permutation(n)
    parts = np.array_split(idx, min(folds, n))
    best, best_reg = -np.inf, regs[0]
    for reg in regs:
        preds = np.full(n, np.nan)
        for p in parts:
            tr = np.setdiff1d(np.arange(n), p)
            if len(np.unique(ytr[tr])) < 2:
                continue
            f = fitter(Xtr[tr], ytr[tr], reg, prior=prior)
            preds[p] = f(Xtr[p])
        ok = ~np.isnan(preds)
        if ok.sum() < 3 or len(np.unique(ytr[ok])) < 2:
            continue
        sc = spearmanr(preds[ok], ytr[ok]).correlation
        if np.isfinite(sc) and sc > best:
            best, best_reg = sc, reg
    return best_reg


# --------------------------------------------------------------------------- #
# eval one config over the two LOBO folds
# --------------------------------------------------------------------------- #
def eval_config(d, prior, sets, head, use_prior, rng):
    lab = d["score"] > 0
    fold_res = {}
    for train_b, eval_b in [("run2", "dive"), ("dive", "run2")]:
        tr = lab & (d["batch"] == train_b)
        ev = lab & (d["batch"] == eval_b)
        Xtr_raw, Xev_raw = stack_blocks(d, sets, tr), stack_blocks(d, sets, ev)
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xev = sc.transform(Xtr_raw), sc.transform(Xev_raw)
        pr = None
        if use_prior:
            Xp = sc.transform(prior["v7"].astype(np.float64)) if sets == ("v7",) else None
            pr = (Xp, prior["score"]) if Xp is not None else None
        reg = inner_cv_reg(FITTERS[head], Xtr, d["score"][tr], REGGRID[head], pr,
                           np.random.default_rng(SEED))
        f = FITTERS[head](Xtr, d["score"][tr], reg, prior=pr)
        pred = f(Xev)
        m = metrics(pred, d["score"][ev])
        m["reg"] = reg
        m["pred"] = pred.tolist()
        m["human"] = d["score"][ev].tolist()
        fold_res[eval_b] = m
    mean_sp = np.nanmean([fold_res["dive"]["spearman"], fold_res["run2"]["spearman"]])
    mean_auc = np.nanmean([fold_res["dive"]["auc"], fold_res["run2"]["auc"]])
    mean_p10 = np.nanmean([fold_res["dive"]["p_at_10"], fold_res["run2"]["p_at_10"]])
    return dict(folds=fold_res, mean_spearman=float(mean_sp), mean_auc=float(mean_auc),
                mean_p_at_10=float(mean_p10))


def perm_p(pred, human, rng, n=5000):
    obs = spearmanr(pred, human).correlation
    human = np.asarray(human)
    cnt = 0
    for _ in range(n):
        if spearmanr(pred, rng.permutation(human)).correlation >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


def boot_ci(pred, human, rng, n=5000):
    pred, human = np.asarray(pred), np.asarray(human)
    vals = []
    m = len(pred)
    for _ in range(n):
        b = rng.integers(0, m, m)
        if len(np.unique(human[b])) < 2:
            continue
        vals.append(spearmanr(pred[b], human[b]).correlation)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def baseline_rows(d):
    """canon_pgood and random baselines under the same LOBO folds."""
    lab = d["score"] > 0
    out = {}
    for name in ("canon_pgood", "random"):
        fold = {}
        rng = np.random.default_rng(SEED)
        for eval_b in ("dive", "run2"):
            ev = lab & (d["batch"] == eval_b)
            human = d["score"][ev]
            pred = d["canon_pgood"][ev] if name == "canon_pgood" else rng.random(int(ev.sum()))
            fold[eval_b] = metrics(pred, human)
        out[name] = dict(folds=fold,
                         mean_spearman=float(np.nanmean([fold["dive"]["spearman"], fold["run2"]["spearman"]])),
                         mean_auc=float(np.nanmean([fold["dive"]["auc"], fold["run2"]["auc"]])),
                         mean_p_at_10=float(np.nanmean([fold["dive"]["p_at_10"], fold["run2"]["p_at_10"]])))
    return out


# --------------------------------------------------------------------------- #
# deploy: refit winner on all 81 labeled
# --------------------------------------------------------------------------- #
def fit_deploy(d, prior, sets, head, use_prior):
    lab = d["score"] > 0
    Xraw = stack_blocks(d, sets, lab)
    sc = StandardScaler().fit(Xraw)
    X = sc.transform(Xraw)
    pr = None
    if use_prior and sets == ("v7",):
        pr = (sc.transform(prior["v7"].astype(np.float64)), prior["score"])
    reg = inner_cv_reg(FITTERS[head], X, d["score"][lab], REGGRID[head], pr, np.random.default_rng(SEED))
    # recover an explicit linear (w, b) so the scorer is dependency-light
    f = FITTERS[head](X, d["score"][lab], reg, prior=pr)
    # probe the affine map: score is linear in standardized X for all three heads
    dim = X.shape[1]
    base = float(f(np.zeros((1, dim)))[0])
    W = f(np.eye(dim)) - base                    # vectorized: linear head accepts the identity
    return dict(sets=list(sets), head=head, use_prior=bool(use_prior and sets == ("v7",)),
                reg=float(reg), mean=sc.mean_, scale=sc.scale_, W=W, b=float(base))


def main():
    d, prior = load()
    rng = np.random.default_rng(SEED)
    print(f"loaded {len(d['ids'])} admissions, {int((d['score']>0).sum())} labeled; "
          f"prior rows = {0 if prior is None else len(prior['score'])}\n")

    results = {}
    for sets in FEATURE_SETS:
        for head in HEADS:
            variants = [False]
            # +prior appends corpus rows; only the row-based heads use it (BT would form
            # ~O(prior^2) pairs that swamp the 81 target rows and the fit).
            if prior is not None and sets == ("v7",) and head in ("ridge", "logi"):
                variants.append(True)
            for up in variants:
                name = "+".join(sets) + f":{head}" + ("+prior" if up else "")
                results[name] = eval_config(d, prior, sets, head, up, rng)
                r = results[name]
                print(f"{name:34s} meanSp={r['mean_spearman']:+.3f}  meanAUC={r['mean_auc']:.3f}  "
                      f"meanP@10={r['mean_p_at_10']:.3f}  "
                      f"(dive Sp={r['folds']['dive']['spearman']:+.3f} / run2 Sp={r['folds']['run2']['spearman']:+.3f})", flush=True)

    base = baseline_rows(d)
    for k, v in base.items():
        print(f"{'BASE '+k:34s} meanSp={v['mean_spearman']:+.3f}  meanAUC={v['mean_auc']:.3f}  "
              f"meanP@10={v['mean_p_at_10']:.3f}")

    # winner = max mean-LOBO Spearman, tie-break mean AUC
    winner = max(results, key=lambda k: (results[k]["mean_spearman"], results[k]["mean_auc"]))
    print(f"\nWINNER: {winner}")

    # headline uncertainty: pool the two folds by percentile-normalizing predictions within fold,
    # then permutation p + bootstrap CI on pooled (pred_pct, human).
    w = results[winner]
    pp, hh = [], []
    for fb in ("dive", "run2"):
        pr_ = np.asarray(w["folds"][fb]["pred"])
        pct = pr_.argsort().argsort() / (len(pr_) - 1)      # within-fold rank -> [0,1]
        pp += pct.tolist(); hh += w["folds"][fb]["human"]
    pp, hh = np.asarray(pp), np.asarray(hh)
    p_perm = perm_p(pp, hh, np.random.default_rng(SEED))
    ci = boot_ci(pp, hh, np.random.default_rng(SEED))
    pooled_sp = spearmanr(pp, hh).correlation
    print(f"pooled Spearman (pct-normalized) = {pooled_sp:+.3f}  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]  "
          f"perm p = {p_perm:.4f}")

    # per-family breakdown for the winner (pooled over both eval folds), multibrot4 called out
    fam_break = family_breakdown(d, w)

    # deploy artifact
    sets = tuple(winner.split(":")[0].split("+"))
    head = winner.split(":")[1].replace("+prior", "")
    use_prior = winner.endswith("+prior")
    dep = fit_deploy(d, prior, sets, head, use_prior)
    np.savez(OUT_DIR / "model.npz", mean=dep["mean"], scale=dep["scale"], W=dep["W"],
             b=np.float64(dep["b"]),
             sets=np.array(dep["sets"]), head=np.array(dep["head"]),
             use_prior=np.array(dep["use_prior"]), reg=np.float64(dep["reg"]))

    metrics_out = dict(
        winner=winner, deploy=dict(sets=dep["sets"], head=dep["head"], use_prior=dep["use_prior"],
                                   reg=dep["reg"]),
        pooled_spearman=float(pooled_sp), pooled_ci=list(ci), perm_p=float(p_perm),
        results={k: {kk: vv for kk, vv in v.items() if kk != "folds"} |
                 {"folds": {fb: {kk: vv for kk, vv in fm.items() if kk not in ("pred", "human")}
                            for fb, fm in v["folds"].items()}}
                 for k, v in results.items()},
        baselines=base, family_breakdown=fam_break, n_labeled=int((d["score"] > 0).sum()),
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print(f"\nwrote {OUT_DIR/'model.npz'} + metrics.json")


def family_breakdown(d, w):
    """Winner predictions vs human, grouped by family, pooled over both eval folds."""
    lab = d["score"] > 0
    fam = {}
    for fb in ("dive", "run2"):
        ev = lab & (d["batch"] == fb)
        fams = d["family"][ev]
        pr_ = np.asarray(w["folds"][fb]["pred"])
        pct = pr_.argsort().argsort() / (len(pr_) - 1)
        hum = np.asarray(w["folds"][fb]["human"])
        for f in np.unique(fams):
            fam.setdefault(f, dict(pred=[], human=[]))
            fam[f]["pred"] += pct[fams == f].tolist()
            fam[f]["human"] += hum[fams == f].tolist()
    out = {}
    for f, v in fam.items():
        p, h = np.asarray(v["pred"]), np.asarray(v["human"])
        sp = spearmanr(p, h).correlation if len(np.unique(h)) > 1 and len(h) > 2 else None
        out[f] = dict(n=len(h), n_good=int((h == 3).sum()),
                      mean_pct_good=float(p[h == 3].mean()) if (h == 3).any() else None,
                      mean_pct_bad=float(p[h == 1].mean()) if (h == 1).any() else None,
                      spearman=None if sp is None or not np.isfinite(sp) else float(sp))
    return out


if __name__ == "__main__":
    main()
