#!/usr/bin/env python
"""pref_loc_v1 — 3-batch leave-one-batch-out refit over run2 + dive + campaign1.

Extends the pref_loc_v0 standing protocol (`train_eval.py`) to a third blind read,
the campaign-1 stratified quality read (298 labels). Same heads (bt / ridge / logi),
same inner-CV reg selection, same baselines (canon_pgood, random), same headline
uncertainty (pct-normalized pooled Spearman + permutation p + bootstrap CI).

TWO deliberate scope changes from v0, both forced and both aligned with v0's findings:
  * MORPH-FREE feature sets only — {v7, colored, v7+colored}. campaign-1 carries no
    morph-CLIP block (the deployed head is v7+colored and v0 found morph adds ~nothing),
    so morph configs are simply not evaluable here and are dropped.
  * NO corpus prior — v0 REJECTED it (degraded the dive fold); not reintroduced.

LOBO is now 3 folds (each read held out once). Winner = max mean-LOBO Spearman
(tie-break mean AUC). CERTIFY (ship provisional) iff the winner (a) beats the
canon_pgood baseline on mean-LOBO Spearman AND (b) its pooled 95% CI excludes 0.
Deploy artifact -> data/ranker/pref_loc_v1/{model.npz, metrics.json, features.npz}.

    uv run python -m tools.ranker.train_eval_v1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.ranker.train_eval import (  # noqa: E402  reuse the exact v0 recipe
    FITTERS, REGGRID, stack_blocks, inner_cv_reg, metrics, perm_p, boot_ci,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

V0_DIR = ROOT / "data" / "ranker" / "pref_loc_v0"
C1_FEATS = ROOT / "data" / "ranker" / "campaign1" / "features.npz"
C1_KEY = ROOT / "out" / "campaign1_blind" / "manifest_key.json"
C1_LABELS = ROOT / "labels" / "campaign1_blind_blind_scores.json"
OUT_DIR = ROOT / "data" / "ranker" / "pref_loc_v1"

BATCHES = ["run2", "dive", "campaign1"]
FEATURE_SETS = [("v7",), ("colored",), ("v7", "colored")]
HEADS = ["bt", "ridge", "logi"]
SEED = 0


def load_v1() -> dict:
    """Assemble the 3-batch labeled matrix: v7 (1280) + colored (768) + score + family +
    batch + canon_pgood. Only labeled rows are kept (score in {1,2,3})."""
    z = np.load(V0_DIR / "features.npz", allow_pickle=True)
    lab = z["score"] > 0
    d = {
        "ids": z["ids"][lab].astype(object),
        "batch": z["batch"][lab].astype(object),
        "family": z["family"][lab].astype(object),
        "canon_pgood": z["canon_pgood"][lab].astype(np.float64),
        "score": z["score"][lab].astype(np.int64),
        "v7": z["v7"][lab].astype(np.float64),
        "colored": z["colored"][lab].astype(np.float64),
    }

    c1 = np.load(C1_FEATS, allow_pickle=True)
    c1i = {str(i): k for k, i in enumerate(c1["ids"])}
    labels = json.load(open(C1_LABELS))
    key = json.load(open(C1_KEY))
    ldg = {}
    for leg in ("breadth", "dive"):
        for line in open(ROOT / f"data/discovery/campaign1/{leg}/outcome_ledger.jsonl", encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                ldg[r["id"]] = r
    ids, batch, fam, cpg, sco, v7, col = [], [], [], [], [], [], []
    for e in key["entries"]:
        i = e["id"]
        ids.append(i); batch.append("campaign1"); fam.append(e["family"])
        cpg.append(float(ldg[i].get("canon_pgood", np.nan)))
        sco.append(int(labels[e["tile"]]))
        v7.append(c1["v7"][c1i[i]]); col.append(c1["colored"][c1i[i]])
    d["ids"] = np.concatenate([d["ids"], np.array(ids, object)])
    d["batch"] = np.concatenate([d["batch"], np.array(batch, object)])
    d["family"] = np.concatenate([d["family"], np.array(fam, object)])
    d["canon_pgood"] = np.concatenate([d["canon_pgood"], np.array(cpg, np.float64)])
    d["score"] = np.concatenate([d["score"], np.array(sco, np.int64)])
    d["v7"] = np.concatenate([d["v7"], np.array(v7, np.float64)])
    d["colored"] = np.concatenate([d["colored"], np.array(col, np.float64)])
    return d


def eval_config(d, sets, head) -> dict:
    """N-batch LOBO for one (sets, head). Returns per-fold metrics + means."""
    fold_res = {}
    for eval_b in BATCHES:
        tr = d["batch"] != eval_b
        ev = d["batch"] == eval_b
        Xtr_raw, Xev_raw = stack_blocks(d, sets, tr), stack_blocks(d, sets, ev)
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xev = sc.transform(Xtr_raw), sc.transform(Xev_raw)
        reg = inner_cv_reg(FITTERS[head], Xtr, d["score"][tr], REGGRID[head], None,
                           np.random.default_rng(SEED))
        f = FITTERS[head](Xtr, d["score"][tr], reg, prior=None)
        pred = f(Xev)
        m = metrics(pred, d["score"][ev])
        m["reg"] = reg
        m["pred"] = pred.tolist()
        m["human"] = d["score"][ev].tolist()
        fold_res[eval_b] = m
    mean_sp = float(np.nanmean([fold_res[b]["spearman"] for b in BATCHES]))
    mean_auc = float(np.nanmean([fold_res[b]["auc"] for b in BATCHES]))
    mean_p10 = float(np.nanmean([fold_res[b]["p_at_10"] for b in BATCHES]))
    return dict(folds=fold_res, mean_spearman=mean_sp, mean_auc=mean_auc, mean_p_at_10=mean_p10)


def baselines(d) -> dict:
    out = {}
    for name in ("canon_pgood", "random"):
        rng = np.random.default_rng(SEED)
        fold = {}
        for eval_b in BATCHES:
            ev = d["batch"] == eval_b
            human = d["score"][ev]
            pred = d["canon_pgood"][ev] if name == "canon_pgood" else rng.random(int(ev.sum()))
            fold[eval_b] = metrics(pred, human)
        out[name] = dict(folds=fold,
                         mean_spearman=float(np.nanmean([fold[b]["spearman"] for b in BATCHES])),
                         mean_auc=float(np.nanmean([fold[b]["auc"] for b in BATCHES])))
    return out


def pooled_uncertainty(w) -> tuple:
    pp, hh = [], []
    for b in BATCHES:
        pr = np.asarray(w["folds"][b]["pred"])
        pct = pr.argsort().argsort() / (len(pr) - 1)
        pp += pct.tolist(); hh += w["folds"][b]["human"]
    pp, hh = np.asarray(pp), np.asarray(hh)
    return (float(spearmanr(pp, hh).correlation),
            boot_ci(pp, hh, np.random.default_rng(SEED)),
            perm_p(pp, hh, np.random.default_rng(SEED)))


def family_breakdown(d, w) -> dict:
    fam = {}
    for b in BATCHES:
        ev = d["batch"] == b
        fams = d["family"][ev]
        pr = np.asarray(w["folds"][b]["pred"])
        pct = pr.argsort().argsort() / (len(pr) - 1)
        hum = np.asarray(w["folds"][b]["human"])
        for f in np.unique(fams):
            fam.setdefault(f, dict(pred=[], human=[]))
            fam[f]["pred"] += pct[fams == f].tolist()
            fam[f]["human"] += hum[fams == f].tolist()
    out = {}
    for f, v in fam.items():
        p, h = np.asarray(v["pred"]), np.asarray(v["human"])
        sp = spearmanr(p, h).correlation if len(np.unique(h)) > 1 and len(h) > 2 else None
        out[f] = dict(n=len(h), n_good=int((h == 3).sum()),
                      spearman=None if sp is None or not np.isfinite(sp) else float(sp))
    return out


def fit_deploy(d, sets, head) -> dict:
    Xraw = stack_blocks(d, sets, np.ones(len(d["score"]), bool))
    sc = StandardScaler().fit(Xraw)
    X = sc.transform(Xraw)
    reg = inner_cv_reg(FITTERS[head], X, d["score"], REGGRID[head], None, np.random.default_rng(SEED))
    f = FITTERS[head](X, d["score"], reg, prior=None)
    dim = X.shape[1]
    base = float(f(np.zeros((1, dim)))[0])
    W = f(np.eye(dim)) - base
    return dict(sets=list(sets), head=head, use_prior=False, reg=float(reg),
                mean=sc.mean_, scale=sc.scale_, W=W, b=float(base))


def main() -> None:
    d = load_v1()
    n = len(d["score"])
    print(f"loaded {n} labeled across {BATCHES} "
          f"({', '.join(f'{b}={int((d['batch']==b).sum())}' for b in BATCHES)})\n")

    results = {}
    for sets in FEATURE_SETS:
        for head in HEADS:
            name = "+".join(sets) + f":{head}"
            r = results[name] = eval_config(d, sets, head)
            print(f"{name:20s} meanSp={r['mean_spearman']:+.3f}  meanAUC={r['mean_auc']:.3f}  "
                  f"meanP@10={r['mean_p_at_10']:.3f}  ("
                  + " / ".join(f"{b} Sp={r['folds'][b]['spearman']:+.3f}" for b in BATCHES) + ")", flush=True)

    base = baselines(d)
    for k, v in base.items():
        print(f"{'BASE '+k:20s} meanSp={v['mean_spearman']:+.3f}  meanAUC={v['mean_auc']:.3f}")

    winner = max(results, key=lambda k: (results[k]["mean_spearman"], results[k]["mean_auc"]))
    w = results[winner]
    pooled_sp, ci, p_perm = pooled_uncertainty(w)
    beats_pgood = w["mean_spearman"] > base["canon_pgood"]["mean_spearman"]
    ci_excl0 = ci[0] > 0
    clears = beats_pgood and ci_excl0
    print(f"\nWINNER: {winner}  meanSp={w['mean_spearman']:+.3f}")
    print(f"pooled Spearman (pct-norm) = {pooled_sp:+.3f}  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]  perm p = {p_perm:.4f}")
    print(f"beats canon_pgood LOBO ({base['canon_pgood']['mean_spearman']:+.3f})? {beats_pgood}   "
          f"CI excludes 0? {ci_excl0}   => CERTIFY: {clears}")

    fam_break = family_breakdown(d, w)
    print("\nper-family LOBO Spearman (winner, pooled over held-out folds):")
    for f in sorted(fam_break, key=lambda f: -fam_break[f]["n"]):
        fb = fam_break[f]
        print(f"  {f:20s} n={fb['n']:3d} good={fb['n_good']:2d} "
              f"Sp={'' if fb['spearman'] is None else f'{fb['spearman']:+.3f}'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out = dict(
        winner=winner, n_labeled=n, batches={b: int((d["batch"] == b).sum()) for b in BATCHES},
        certified=bool(clears), certify_reasons=dict(beats_canon_pgood=bool(beats_pgood),
                                                     ci_excludes_0=bool(ci_excl0)),
        pooled_spearman=pooled_sp, pooled_ci=list(ci), perm_p=p_perm,
        results={k: {kk: vv for kk, vv in v.items() if kk != "folds"} |
                 {"folds": {b: {kk: vv for kk, vv in fm.items() if kk not in ("pred", "human")}
                            for b, fm in v["folds"].items()}}
                 for k, v in results.items()},
        baselines=base, family_breakdown=fam_break,
    )
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    if clears:
        dep = fit_deploy(d, tuple(winner.split(":")[0].split("+")), winner.split(":")[1])
        np.savez(OUT_DIR / "model.npz", mean=dep["mean"], scale=dep["scale"], W=dep["W"],
                 b=np.float64(dep["b"]), sets=np.array(dep["sets"]), head=np.array(dep["head"]),
                 use_prior=np.array(False), reg=np.float64(dep["reg"]))
        # persist the exact 3-batch labeled matrix the deploy was fit on
        np.savez_compressed(OUT_DIR / "features.npz",
                            ids=d["ids"], batch=d["batch"], family=d["family"],
                            canon_pgood=d["canon_pgood"], score=d["score"],
                            v7=d["v7"].astype(np.float32), colored=d["colored"].astype(np.float32))
        print(f"\nCERTIFIED — wrote {OUT_DIR/'model.npz'} + features.npz + metrics.json (PROVISIONAL)")
    else:
        print(f"\nNOT certified — wrote {OUT_DIR/'metrics.json'} only; pref_loc_v0 stays deployed")


if __name__ == "__main__":
    main()
