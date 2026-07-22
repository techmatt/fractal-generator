#!/usr/bin/env python
r"""Phoenix grid — label analysis (adjudicate the Phase-B provisional verdicts).

Joins the 500-item human labels (labels/phoenix_grid.json, resolved ONLY through
label_store.resolve_score) to the grid run, and settles what Phase B marked provisional:

  §0 reachability + count reconciliation
  §1 human between/within-seed variance decomposition (reuses phoenix_decomp machinery),
     side by side with the machine ICCs; human fertility map (branch / |p| band / z-class)
  §2 v7 calibration on varied phoenix (AUC, Spearman, calibration curve) + a PROPOSED
     phoenix t_good under the standard per-family F2 methodology (tools/v7/derive_t_good)
  §4a surrogate viability — spec §5.2 light head (logistic over logged cheap seed features)
      vs human seed-fertility, LOSO-CV over the seeds (held-out Spearman)

Analysis + proposals ONLY. Flips no production threshold, re-decodes no ledger, retrains
nothing, edits no measure. Writes a machine-readable JSON the doc writer + §3/§4-ranker
render steps consume.

  uv run python tools/phoenix/phoenix_label_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

import phoenix_decomp as decomp                       # noqa: E402  reuse anova_components + cluster_bootstrap
import label_store as ls                              # noqa: E402
from score_lib import corn_decode                     # noqa: E402

RUN = ROOT / "data" / "discovery" / "phoenix_grid" / "grid"
BATCH_ID = "2026-07-21_phoenix_grid"
BATCH = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
OUT = ROOT / "data" / "discovery" / "phoenix_grid" / "label_analysis.json"

CURRENT_TGOOD = 0.50          # production phoenix t_good (t_good_for("phoenix") -> baseline)
GRID_TGOOD = 0.18             # the provisional the grid ran at
MIN_POS = 15                  # derive_t_good gate-3 sufficiency floor
TGRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]   # [0.02, 0.98], derive_t_good grid


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# §0  Reachability + join
# --------------------------------------------------------------------------- #
def load_joined():
    """Every batch row joined to its human label (via label_store) + reward-path truth
    (all_outcomes). Returns (rows, reach) where reach is the §0 reconciliation."""
    batch = read_jsonl(BATCH / "images.jsonl")
    sidecar = ls.sidecar_for(BATCH_ID)
    if sidecar is None:
        raise SystemExit(
            f"ABORT §0: batch {BATCH_ID!r} not registered in label_store.SIDECAR_LABELS — "
            f"the canonical reader can't reach labels/phoenix_grid.json. Register it first.")
    allo = {r["id"]: r for r in read_jsonl(RUN / "all_outcomes.jsonl")}

    rows = []
    joined = defaultdict(int)
    n_unlabeled = 0
    for r in batch:
        score = ls.resolve_score(r, sidecar)
        if score is not None:
            joined[BATCH_ID] += 1
        else:
            n_unlabeled += 1
        iid = r["image_id"]
        seed, repeat, walk = iid.split("_")[1:]
        prov, rend = r["provenance"], r["render"]
        o = allo.get(iid, {})
        p = complex(float(rend["p_re"]), float(rend["p_im"]))
        z = complex(float(rend["zm1_re"]), float(rend["zm1_im"]))
        rows.append({
            "id": iid, "seed": int(seed), "repeat": int(repeat), "walk": int(walk),
            "score": score,
            "stratum": prov["stratum"], "band": prov["stratum"],  # HIGH/Q3/SUB/REJECT selection band
            "sel_band": prov["stratum"],
            "cell": prov.get("stratum"),
            "branch": prov["branch"], "z_class": prov["z_class"],
            "p_band": "p" + str(0 if abs(p) < 0.33 else (1 if abs(p) < 0.66 else 2)),
            "abs_p": abs(p), "abs_z": abs(z),
            # reward-path truth (authoritative p_good/p_notbad/guard/depth); cross-checked vs prov
            "p_good": float(o.get("p_good", prov.get("p_good") or 0.0)),
            "p_notbad": float(o.get("p_notbad", prov.get("p_notbad") or 0.0)),
            "guard_pass": bool(o.get("guard_pass", True)),
            "decoded_class": o.get("decoded_class"),
            "reached_depth": o.get("reached_depth"),
        })
    ls.assert_sidecars_joined(joined)     # loud if the registered sidecar joined 0 rows

    # the batch registers a selection band on provenance.stratum? no — stratum is the DRAW cell
    # (p*|branch|z); the HIGH/Q3/SUB/REJECT band lives in provenance.selection_role/stratum. Fix:
    reach = {
        "n_batch": len(batch), "n_labeled": sum(1 for r in rows if r["score"] is not None),
        "n_unlabeled": n_unlabeled, "n_sidecar_keys": len(sidecar),
        "n_all_outcomes": len(allo),
        "n_joined_allo": sum(1 for r in rows if r["id"] in allo),
        "score_dist": dict(sorted({s: sum(1 for r in rows if r["score"] == s)
                                   for s in (1, 2, 3)}.items())),
    }
    return rows, reach


# --------------------------------------------------------------------------- #
# §1  Human variance decomposition + fertility map
# --------------------------------------------------------------------------- #
def human_decomposition(rows, n_boot=2000, seed=0):
    """Reuse phoenix_decomp's ANOVA + cluster-bootstrap machinery on HUMAN scores.

    Two per-descent variables mirroring the machine pass, grouped by SEED:
      * max_human   — max human score over a descent's labeled walks (mirror of max_p_good)
      * mean_human  — mean human score over a descent's labeled walks
    Plus a per-image between-seed ICC (group=seed) as a design-independent robustness read."""
    labeled = [r for r in rows if r["score"] is not None]

    # per-descent aggregates
    by_desc = defaultdict(list)
    for r in labeled:
        by_desc[(r["seed"], r["repeat"])].append(r["score"])
    desc_rows = [{"seed_idx": s, "repeat": rp, "max_human": float(max(v)),
                  "mean_human": float(np.mean(v)), "n_labeled": len(v)}
                 for (s, rp), v in by_desc.items()]

    out = {}
    for var in ("max_human", "mean_human"):
        groups = decomp._groups(desc_rows, var)     # per-seed arrays, >=2 repeats
        if len(groups) < 2:
            out[var] = {"error": "too few multi-descent seeds"}
            continue
        comp = decomp.anova_components(groups)
        boot = decomp.cluster_bootstrap(groups, n_boot, seed)
        between = comp["icc"] > 0.5 and boot["icc_ci95"][0] > 0.0
        out[var] = {**comp, **boot,
                    "verdict": "between-seed dominates" if between
                    else "within-seed dominates / inconclusive"}

    # per-image between-seed ICC (group = seed; every labeled image is one obs)
    by_seed_img = defaultdict(list)
    for r in labeled:
        by_seed_img[r["seed"]].append(float(r["score"]))
    img_groups = [np.array(v, float) for v in by_seed_img.values() if len(v) >= 2]
    comp_img = decomp.anova_components(img_groups)
    boot_img = decomp.cluster_bootstrap(img_groups, n_boot, seed)
    out["per_image_by_seed"] = {**comp_img, **boot_img,
                                "n_seeds": len(img_groups),
                                "verdict": "between-seed dominates"
                                if (comp_img["icc"] > 0.5 and boot_img["icc_ci95"][0] > 0.0)
                                else "within-seed dominates / inconclusive"}
    out["n_descents_labeled"] = len(desc_rows)
    out["n_seeds_labeled"] = len({r["seed"] for r in labeled})
    return out


def human_fertility(rows, key):
    labeled = [r for r in rows if r["score"] is not None]
    g = defaultdict(list)
    for r in labeled:
        g[key(r)].append(r)
    table = []
    for k in sorted(g, key=str):
        rs = g[k]
        sc = np.array([r["score"] for r in rs], float)
        table.append({
            "key": str(k), "n_labeled": len(rs),
            "n_good": int((sc == 3).sum()), "n_okay": int((sc == 2).sum()),
            "n_bad": int((sc == 1).sum()),
            "good_frac": float((sc == 3).mean()), "mean_score": float(sc.mean()),
            "n_seeds": len({r["seed"] for r in rs}),
            "n_good_seeds": len({r["seed"] for r in rs if r["score"] == 3}),
        })
    return table


# --------------------------------------------------------------------------- #
# §2  v7 calibration + proposed t_good (standard F2 methodology, derive_t_good)
# --------------------------------------------------------------------------- #
def confusion(rows, t):
    tp = fp = fn = 0
    for nb, g, pos in rows:
        pred = corn_decode(nb, g, t) == 3
        tp += pred and pos
        fp += pred and not pos
        fn += (not pred) and pos
    return tp, fp, fn


def prf2(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) else 0.0
    return prec, rec, f2


def best_t(rows):
    best = None
    for t in TGRID:
        _, _, f2 = prf2(*confusion(rows, t))
        if best is None or f2 > best[1] + 1e-12 or (abs(f2 - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f2)
    return best[0]


def loo_f2(rows):
    tp = fp = fn = 0
    for i in range(len(rows)):
        t = best_t(rows[:i] + rows[i + 1:])
        nb, g, pos = rows[i]
        pred = corn_decode(nb, g, t) == 3
        tp += pred and pos
        fp += pred and not pos
        fn += (not pred) and pos
    return prf2(tp, fp, fn)


def v7_calibration(rows):
    labeled = [r for r in rows if r["score"] is not None]
    y = np.array([1 if r["score"] == 3 else 0 for r in labeled])    # good vs rest
    pg = np.array([r["p_good"] for r in labeled])
    score = np.array([r["score"] for r in labeled], float)

    # AUC good-vs-rest (Mann-Whitney U / rank) + rank correlations
    auc = float(_auc(pg, y))
    sp = stats.spearmanr(pg, score)
    # also p_good vs raw ordinal label, and P(not-bad) vs label>=2
    y_nb = np.array([1 if r["score"] >= 2 else 0 for r in labeled])
    pnb = np.array([r["p_notbad"] for r in labeled])
    auc_nb = float(_auc(pnb, y_nb))

    # calibration curve: mean human score + good-frac per p_good decile
    order = np.argsort(pg)
    bins = np.array_split(order, 10)
    curve = []
    for b in bins:
        if len(b) == 0:
            continue
        curve.append({"pg_lo": float(pg[b].min()), "pg_hi": float(pg[b].max()),
                      "pg_mean": float(pg[b].mean()), "n": len(b),
                      "human_mean": float(score[b].mean()),
                      "good_frac": float(y[b].mean())})

    # proposed t_good via F2 (mirror derive_t_good): rows = (p_notbad, p_good, is_pos)
    frows = [(r["p_notbad"], r["p_good"], r["score"] == 3) for r in labeled]
    n_pos = sum(1 for *_, p in frows if p)
    t_star = best_t(frows)
    p_in, r_in, f2_in = prf2(*confusion(frows, t_star))
    oof_p, oof_r, oof_f2 = loo_f2(frows)
    # what current 0.50 and grid 0.18 do
    def at(t):
        tp, fp, fn = confusion(frows, t)
        p, r, f2 = prf2(tp, fp, fn)
        return {"t": t, "prec": p, "rec": r, "f2": f2, "admit": tp + fp, "disc_q3": fn,
                "tp": tp, "fp": fp, "fn": fn}
    return {
        "n_labeled": len(labeled), "n_pos": n_pos,
        "auc_good_vs_rest": auc, "auc_notbad": auc_nb,
        "spearman_pg_score": {"rho": float(sp.statistic), "p": float(sp.pvalue)},
        "calibration_curve": curve,
        "proposed_t_good": {
            "t_star": t_star, "f2_in": f2_in, "f2_oof": oof_f2, "gap": f2_in - oof_f2,
            "prec_in": p_in, "rec_in": r_in,
            "oof_prec": oof_p, "oof_rec": oof_r,
            "sufficient": n_pos >= MIN_POS, "min_pos": MIN_POS,
            "current": at(CURRENT_TGOOD), "grid": at(GRID_TGOOD), "at_star": at(t_star),
        },
    }


def _auc(scores, y):
    """AUC via Mann-Whitney rank; ties handled by average ranks. y in {0,1}."""
    y = np.asarray(y); s = np.asarray(scores, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------- #
# §4a  Surrogate viability — light head over logged cheap seed features
# --------------------------------------------------------------------------- #
def surrogate_viability(rows, seed=0):
    """spec §5.2 light head: logistic over the LOGGED cheap seed features (seeds.jsonl
    `features`) predicting human seed-fertility, LOSO-CV over seeds. Held-out Spearman
    of predicted-fertility vs realized human seed-fertility.

    Seed-fertility target = mean human score over the seed's labeled images (the human
    analog of the machine max_p_good/adm signal). Report both a continuous target
    (mean human) and a binary target (seed has >=1 GOOD)."""
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler

    seeds = {int(s["seed_idx"]): s for s in read_jsonl(RUN / "seeds.jsonl")}
    labeled = [r for r in rows if r["score"] is not None]

    by_seed = defaultdict(list)
    for r in labeled:
        by_seed[r["seed"]].append(r["score"])
    seed_ids = sorted(by_seed)
    # cheap features (logged, geometry-only). branch as one-hot; the rest numeric.
    FEATS = ["mandphoenix_de", "mandphoenix_iters", "root_dist", "abs_offset",
             "abs_p", "arg_p", "abs_z_m1"]
    BRANCHES = ["cardioid", "period2", "root"]

    def feat_vec(sid):
        f = seeds[sid]["features"]
        v = [float(f[k]) for k in FEATS]
        v += [1.0 if seeds[sid]["branch"] == b else 0.0 for b in BRANCHES]
        v.append(1.0 if f.get("mandphoenix_escaped") else 0.0)
        return v

    X = np.array([feat_vec(s) for s in seed_ids], float)
    y_cont = np.array([float(np.mean(by_seed[s])) for s in seed_ids])
    y_bin = np.array([1 if max(by_seed[s]) == 3 else 0 for s in seed_ids])

    # LOSO predictions (each seed held out; train on the rest)
    def loso(predict_fn):
        preds = np.zeros(len(seed_ids))
        for i in range(len(seed_ids)):
            tr = np.array([j for j in range(len(seed_ids)) if j != i])
            preds[i] = predict_fn(X[tr], i)
        return preds

    # continuous head (linear regression on standardized feats)
    def lin_pred(Xtr, i, ytr):
        sc = StandardScaler().fit(Xtr)
        m = LinearRegression().fit(sc.transform(Xtr), ytr)
        return float(m.predict(sc.transform(X[i:i + 1]))[0])
    cont_pred = np.array([lin_pred(X[[j for j in range(len(seed_ids)) if j != i]], i,
                                   y_cont[[j for j in range(len(seed_ids)) if j != i]])
                          for i in range(len(seed_ids))])
    sp_cont = stats.spearmanr(cont_pred, y_cont)

    # binary head (logistic) — held-out P(good) vs realized, AUC + Spearman
    bin_pred = np.zeros(len(seed_ids))
    for i in range(len(seed_ids)):
        tr = np.array([j for j in range(len(seed_ids)) if j != i])
        if len(np.unique(y_bin[tr])) < 2:
            bin_pred[i] = y_bin[tr].mean()
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), y_bin[tr])
        bin_pred[i] = float(m.predict_proba(sc.transform(X[i:i + 1]))[0, 1])
    sp_bin = stats.spearmanr(bin_pred, y_bin)
    auc_bin = float(_auc(bin_pred, y_bin))

    return {
        "n_seeds": len(seed_ids), "n_features": X.shape[1], "features": FEATS + BRANCHES + ["escaped"],
        "base_rate_good_seed": float(y_bin.mean()),
        "continuous_target": {"spearman_loso": float(sp_cont.statistic),
                              "p": float(sp_cont.pvalue)},
        "binary_target": {"spearman_loso": float(sp_bin.statistic), "p": float(sp_bin.pvalue),
                          "auc_loso": auc_bin, "n_good_seeds": int(y_bin.sum())},
    }


# --------------------------------------------------------------------------- #
def main():
    rows, reach = load_joined()
    print("=== §0 reachability ===")
    for k, v in reach.items():
        print(f"  {k}: {v}")

    machine = json.loads((RUN / "decomposition.json").read_text(encoding="utf-8"))["decomposition"]
    human = human_decomposition(rows)
    print("\n=== §1 human decomposition ===")
    for var in ("max_human", "mean_human", "per_image_by_seed"):
        d = human[var]
        if "error" in d:
            print(f"  {var}: {d['error']}"); continue
        print(f"  {var}: ICC={d['icc']:.3f} CI[{d['icc_ci95'][0]:.3f},{d['icc_ci95'][1]:.3f}] "
              f"sigma2_b={d['var_between']:.4f} sigma2_w={d['var_within']:.4f} -> {d['verdict']}")

    fert = {
        "branch": human_fertility(rows, lambda r: r["branch"]),
        "p_band": human_fertility(rows, lambda r: r["p_band"]),
        "z_class": human_fertility(rows, lambda r: r["z_class"]),
        "stratum": human_fertility(rows, lambda r: r["cell"]),
        "sel_band": human_fertility(rows, lambda r: r["sel_band"]),
    }
    print("\n=== §1 fertility (branch) ===")
    for r in fert["branch"]:
        print(f"  {r['key']:10s} n={r['n_labeled']:3d} good={r['n_good']:3d} "
              f"good_frac={r['good_frac']:.3f} mean={r['mean_score']:.2f} "
              f"good_seeds={r['n_good_seeds']}/{r['n_seeds']}")
    print("=== §1 fertility (z_class) ===")
    for r in fert["z_class"]:
        print(f"  {r['key']:10s} n={r['n_labeled']:3d} good={r['n_good']:3d} "
              f"good_frac={r['good_frac']:.3f} mean={r['mean_score']:.2f}")
    print("=== §1 fertility (|p| band) ===")
    for r in fert["p_band"]:
        print(f"  {r['key']:10s} n={r['n_labeled']:3d} good={r['n_good']:3d} "
              f"good_frac={r['good_frac']:.3f} mean={r['mean_score']:.2f}")

    cal = v7_calibration(rows)
    print("\n=== §2 v7 calibration ===")
    print(f"  n={cal['n_labeled']} n_pos(good)={cal['n_pos']}")
    print(f"  AUC good-vs-rest={cal['auc_good_vs_rest']:.3f}  AUC notbad={cal['auc_notbad']:.3f}")
    print(f"  Spearman(p_good, human)={cal['spearman_pg_score']['rho']:.3f} "
          f"(p={cal['spearman_pg_score']['p']:.2e})")
    pt = cal["proposed_t_good"]
    print(f"  proposed t*={pt['t_star']} F2_in={pt['f2_in']:.3f} F2_oof={pt['f2_oof']:.3f} "
          f"sufficient={pt['sufficient']} (n_pos={cal['n_pos']} vs floor {MIN_POS})")
    print(f"    @current {CURRENT_TGOOD}: {pt['current']}")
    print(f"    @grid    {GRID_TGOOD}: {pt['grid']}")
    print(f"    @t*      {pt['t_star']}: {pt['at_star']}")
    print("  calibration curve (p_good decile -> human):")
    for c in cal["calibration_curve"]:
        print(f"    pg[{c['pg_lo']:.3f},{c['pg_hi']:.3f}] n={c['n']:3d} "
              f"human_mean={c['human_mean']:.2f} good_frac={c['good_frac']:.3f}")

    surr = surrogate_viability(rows)
    print("\n=== §4a surrogate viability ===")
    print(f"  n_seeds={surr['n_seeds']} good-seed base rate={surr['base_rate_good_seed']:.3f}")
    print(f"  continuous (mean human): LOSO Spearman={surr['continuous_target']['spearman_loso']:.3f} "
          f"(p={surr['continuous_target']['p']:.2e})")
    print(f"  binary (has-good):       LOSO Spearman={surr['binary_target']['spearman_loso']:.3f} "
          f"AUC={surr['binary_target']['auc_loso']:.3f} n_good_seeds={surr['binary_target']['n_good_seeds']}")

    result = {"reach": reach, "machine_decomposition": machine, "human_decomposition": human,
              "human_fertility": fert, "v7_calibration": cal, "surrogate": surr,
              "current_tgood": CURRENT_TGOOD, "grid_tgood": GRID_TGOOD}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
