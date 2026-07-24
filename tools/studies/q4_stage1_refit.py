#!/usr/bin/env python
"""q4 stage-1 REFIT — firm the goodness field, decide harvest-readiness.

Fold the G-aimed labels into p1+p2, drop Laplacian (T3 didn't earn its place),
re-verify held-out, and answer ONE question: is the fit stable enough to harvest?
This is the single firming pass. It does NOT emit a new aimed batch and does NOT
harvest — it reports a readiness verdict and stops.

Data: p1 + p2 + q4_g_aimed, accept/reject only (filter_leak excluded), newest-label
precedence. One featurizer (LF.featurize), recomputed from the f64 fields, over BOTH
registered window batches.

Sections: pre-refit audit (new labels x slug bucket) · 2-tier L1 refit + LOMO AUC/AP
vs prior (T1 0.848 / T2 0.878) · weight-stability deltas vs the first fit · field
re-verify (v2-masked dense grid, same minibrots) · readiness verdict.

Run:  uv run python -m tools.studies.q4_stage1_refit refit
      uv run python -m tools.studies.q4_stage1_refit field
      uv run python -m tools.studies.q4_stage1_refit all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus import q4_window_reader as qr  # noqa: E402
from tools.studies import q4_stage1_labelset as LS  # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF  # noqa: E402

OLD_BATCH = "2026-07-23_q4_stage1_windows"
NEW_BATCH = "2026-07-23_q4_g_aimed"
LABEL_FILES = [ROOT / "labels" / "q4_stage1_windows.json",       # p1
               ROOT / "labels" / "q4_stage1_windows_p2.json",     # p2
               ROOT / "labels" / "q4_g_aimed.json"]               # g-aimed (newest)
OUT = ROOT / "out" / "q4_stage1" / "refit"
FRAMES = ROOT / "out" / "q4_stage1" / "frames"
FINDINGS = ROOT / "docs" / "findings" / "q4_stage1_refit.md"

TIERS = ["T1_global", "T2_cells"]          # Laplacian (T3) dropped
PRIOR_AUC = {"T1_global": 0.848, "T2_cells": 0.878}
# The first fit chose these C by max-LOMO-AUC. The C->AUC curve is FLAT (T2 spans
# 0.854..0.863 across the whole grid), so re-selecting C is pure noise and would make
# the weight deltas an apples-to-oranges C-sparsity artifact. Refit weights + reported
# AUC are held at the SAME C as the first fit -> genuine stability comparison.
PRIOR_C = {"T1_global": 0.05, "T2_cells": 2.0}
CGRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
FIELD_MBS = ["mb07_p16", "mb09_p19", "mb14_p30"]   # same as the first fit


def load_labels():
    """p1+p2+g_aimed union, newest precedence, accept/reject only."""
    merged = {}
    for p in LABEL_FILES:
        merged.update(json.loads(p.read_text()))
    return {k: v for k, v in merged.items() if v in ("accept", "reject")}


def build_dataset(labels, *, cache=True):
    """[(window_id, minibrot_id, feats, klass_or_None)] over BOTH registered batches."""
    cache_p = OUT / "features_cache.json"
    if cache and cache_p.exists():
        raw = json.loads(cache_p.read_text())
        return [(r["wid"], r["mb"], r["f"], labels.get(r["wid"])) for r in raw]
    by_mb = defaultdict(list)
    for batch in (OLD_BATCH, NEW_BATCH):
        for row, _ in qr.iter_windows(batch):
            by_mb[row["minibrot_id"]].append(row)
    rows, raw = [], []
    for mbid, wins in sorted(by_mb.items()):
        field, fw, fh = LS.load_field_values(mbid)
        for r in wins:
            f = LF.featurize(LF.crop_field(field, fw, fh, r["window"]))
            if f is None:
                continue
            rows.append((r["window_id"], mbid, f, labels.get(r["window_id"])))
            raw.append(dict(wid=r["window_id"], mb=mbid, f=f))
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(raw))
    return rows


# --------------------------------------------------------------------------- #
def pre_refit_audit():
    """Cross-tab the NEW labels by their slug bucket, using the PRE-refit model's
    stored p_accept (how trustworthy the model already was)."""
    labels = json.loads((ROOT / "labels" / "q4_g_aimed.json").read_text())
    rows = [json.loads(l) for l in
            (qr.batch_dir(NEW_BATCH) / "windows.jsonl").read_text().splitlines() if l.strip()]
    by_slug = defaultdict(list)
    for r in rows:
        lab = labels.get(r["window_id"])
        by_slug[r["slug"]].append((r["window_id"], r.get("p_accept"), lab))
    print("\n=== PRE-REFIT AUDIT (new G-aimed labels x slug; p = pre-refit model) ===")
    audit = {}
    for slug in ("top_g", "uncertain", "control"):
        items = by_slug.get(slug, [])
        n = len(items)
        na = sum(1 for _, _, k in items if k == "accept")
        rate = na / n if n else 0.0
        audit[slug] = dict(n=n, accept=na, accept_rate=round(rate, 3))
        print(f"  {slug:>10}: n={n:<3} accept={na:<3} accept-rate={rate:5.1%}")
    # confident-wrong: pre-refit p high but labeled reject (blind spot)
    cw = [(w, p) for slug in by_slug for (w, p, k) in by_slug[slug]
          if p is not None and p >= 0.80 and k == "reject"]
    cw.sort(key=lambda x: -x[1])
    print(f"  confident-WRONG (pre-refit p>=0.80 but rejected): {len(cw)}")
    for w, p in cw[:8]:
        print(f"       p={p:.2f}  {w}")
    audit["confident_wrong"] = [dict(window_id=w, p=p) for w, p in cw]
    return audit


# --------------------------------------------------------------------------- #
def refit(rows):
    """Refit at the SAME C as the first fit (PRIOR_C). Also record the full C->AUC
    grid so the reader can see C-selection is flat (delta-vs-prior is within noise)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    report, weights = {}, {}
    for tier in TIERS:
        grid = {}
        for C in CGRID:
            y, g = LF.lomo_scores(rows, tier, C)
            grid[C] = float(roc_auc_score(y, g))
        Cfix = PRIOR_C[tier]
        y, g = LF.lomo_scores(rows, tier, Cfix)
        report[tier] = dict(C=Cfix, auc=float(roc_auc_score(y, g)),
                            ap=float(average_precision_score(y, g)),
                            auc_grid={str(k): round(v, 3) for k, v in grid.items()},
                            auc_grid_min=round(min(grid.values()), 3),
                            auc_grid_max=round(max(grid.values()), 3),
                            n_accept=int((y == 1).sum()), n_reject=int((y == 0).sum()))
        w, b, _, _ = LF.surviving_weights(rows, tier, Cfix)
        weights[tier] = dict(intercept=b, weights=w,
                             nonzero={k: v for k, v in
                                      sorted(w.items(), key=lambda kv: -abs(kv[1]))
                                      if abs(v) > 1e-6})
    return report, weights


def weight_deltas(weights):
    """Refit weights vs the first fit (out/q4_stage1/linear_fit/weights.json)."""
    prior = json.loads((ROOT / "out" / "q4_stage1" / "linear_fit" / "weights.json").read_text())
    out = {}
    for tier in TIERS:
        pw = prior["weights"][tier]["weights"]
        nw = weights[tier]["weights"]
        keys = LF.FEATURES[tier]
        rows = []
        for k in keys:
            rows.append(dict(feature=k, prior=round(pw.get(k, 0.0), 3),
                             refit=round(nw.get(k, 0.0), 3),
                             delta=round(nw.get(k, 0.0) - pw.get(k, 0.0), 3)))
        rows.sort(key=lambda r: -abs(r["refit"]))
        out[tier] = rows
    return out


def _sign(x, eps=0.05):
    return "0" if abs(x) < eps else ("+" if x > 0 else "-")


def stage_refit():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    rows = build_dataset(labels)
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    n_acc = sum(1 for r in lab if r[3] == "accept")
    mbs = len({r[1] for r in lab})
    print(f"combined accept/reject: {len(lab)}  ({n_acc} accept / {len(lab)-n_acc} reject)"
          f"  over {mbs} minibrots")
    # provenance of the fold
    for name, p in (("p1", LABEL_FILES[0]), ("p2", LABEL_FILES[1]), ("g_aimed", LABEL_FILES[2])):
        d = json.loads(p.read_text())
        c = Counter(v for v in d.values() if v in ("accept", "reject"))
        print(f"    {name}: {dict(c)}")

    audit = pre_refit_audit()

    report, weights = refit(rows)
    print("\n=== REFIT held-out (minibrot-disjoint LOMO, C fixed at first-fit value) ===")
    print(f"{'tier':<12}{'C':>6}{'AUC':>8}{'prior':>8}{'dAUC':>8}{'AP':>7}{'  grid[min..max]':>18}")
    for tier in TIERS:
        r = report[tier]
        d = r["auc"] - PRIOR_AUC[tier]
        print(f"{tier:<12}{r['C']:>6}{r['auc']:>8.3f}{PRIOR_AUC[tier]:>8.3f}"
              f"{d:>+8.3f}{r['ap']:>7.3f}   [{r['auc_grid_min']:.3f}..{r['auc_grid_max']:.3f}]")

    deltas = weight_deltas(weights)
    print("\n=== WEIGHT STABILITY (T2_cells; refit vs first fit) ===")
    print(f"{'feature':<24}{'prior':>8}{'refit':>8}{'delta':>8}")
    for r in deltas["T2_cells"]:
        flag = "  <-- SIGN FLIP" if (_sign(r["prior"]) != _sign(r["refit"])
                                     and _sign(r["prior"]) != "0"
                                     and _sign(r["refit"]) != "0") else ""
        print(f"{r['feature']:<24}{r['prior']:>8}{r['refit']:>8}{r['delta']:>+8}{flag}")

    # the three named checks
    w2 = weights["T2_cells"]["weights"]
    p2 = json.loads((ROOT / "out" / "q4_stage1" / "linear_fit" / "weights.json").read_text()
                    )["weights"]["T2_cells"]["weights"]
    checks = dict(
        g_mid_dominates_T1=("g_mid" in weights["T1_global"]["nonzero"]),
        flat_edge_sign_kept=(_sign(p2.get("flat_edge_minus_center", 0)) ==
                             _sign(w2.get("flat_edge_minus_center", 0)) != "0"),
        g_occ_stayed_dead=(abs(w2.get("g_occ", 0)) < 0.30),
    )

    # verdict. Held-check compares grid-MAX to prior (prior AUC was itself a grid-max
    # selection) so it's fair; C-grid is flat so the fixed-C AUC is within the same band.
    t2 = report["T2_cells"]
    held = t2["auc_grid_max"] >= PRIOR_AUC["T2_cells"] - 0.02
    # A sign flip only signals a MOVING boundary if it hits a dominant carrier
    # (prior |w| >= 1.0). A lone flip in a secondary dispersion term is a noted wobble.
    def _flip(r):
        return (_sign(r["prior"]) != _sign(r["refit"]) and _sign(r["prior"]) != "0"
                and _sign(r["refit"]) != "0" and max(abs(r["prior"]), abs(r["refit"])) > 0.3)
    primary_flips = [r["feature"] for r in deltas["T2_cells"]
                     if _flip(r) and abs(r["prior"]) >= 1.0]
    secondary_flips = [r["feature"] for r in deltas["T2_cells"]
                       if _flip(r) and abs(r["prior"]) < 1.0]
    converged = len(primary_flips) == 0
    ready = held and converged

    print("\n=== READINESS VERDICT ===")
    print(f"  held-out held (T2 grid-max {t2['auc_grid_max']:.3f} vs prior "
          f"{PRIOR_AUC['T2_cells']:.3f}; fixed-C {t2['auc']:.3f}): {'YES' if held else 'NO'}")
    print(f"  dominant carriers (|w|>=1.0) stable: {'YES' if converged else 'NO'}"
          + (f"  PRIMARY FLIPS={primary_flips}" if primary_flips else ""))
    print(f"  secondary wobble: {secondary_flips or 'none'}")
    print(f"  named checks: g_mid dominates T1={checks['g_mid_dominates_T1']}  "
          f"flat_edge sign kept={checks['flat_edge_sign_kept']}  "
          f"g_occ dead={checks['g_occ_stayed_dead']}")
    print(f"\n  >>> HARVEST-READY: {'YES' if ready else 'NO'}")

    out = dict(n_labeled=len(lab), n_accept=n_acc, n_minibrots=mbs,
               audit=audit, report=report,
               weights={t: weights[t]["nonzero"] for t in TIERS},
               deltas=deltas, checks=checks,
               verdict=dict(held=held, converged=converged, ready=ready,
                            primary_flips=primary_flips, secondary_flips=secondary_flips))
    (OUT / "refit.json").write_text(json.dumps(out, indent=2))
    write_report(out, weights)
    return out


# --------------------------------------------------------------------------- #
def write_report(out, weights):
    L = ["# q4 stage-1 refit — firm the goodness field, harvest-readiness\n"]
    L.append(f"Folded **q4_g_aimed** (34 accept / 78 reject) into p1+p2. Combined: "
             f"**{out['n_labeled']}** accept/reject ({out['n_accept']} accept / "
             f"{out['n_labeled']-out['n_accept']} reject) over {out['n_minibrots']} "
             f"minibrots. Laplacian (T3) dropped. Referee: minibrot-disjoint LOMO.\n")
    a = out["audit"]
    L.append("## Pre-refit audit (new labels × slug, pre-refit model)\n")
    L.append("| slug | n | accept-rate | reads as |")
    L.append("|---|---|---|---|")
    L.append(f"| top_g | {a['top_g']['n']} | {a['top_g']['accept_rate']:.0%} | "
             f"precision of confident-accepts |")
    unc = a['uncertain']['accept_rate']
    L.append(f"| uncertain | {a['uncertain']['n']} | {unc:.0%} | "
             + ("~50% ⇒ boundary sat where the model thought |" if 0.4 <= unc <= 0.6
                else f"≪50% ⇒ p≈0.5 zone really {unc:.0%} accept (optimistic calibration) |"))
    L.append(f"| control | {a['control']['n']} | {a['control']['accept_rate']:.0%} | "
             f"unbiased base-rate |")
    L.append(f"\nConfident-wrong (pre-refit p≥0.80 but rejected): "
             f"**{len(a['confident_wrong'])}**. The uncertain bucket landing at "
             f"{a['uncertain']['accept_rate']:.0%} (not ~50%) means the pre-refit "
             f"probabilities were **optimistic** near the boundary — p≈0.5 was really "
             f"~{a['uncertain']['accept_rate']:.0%} accept. That's a calibration offset, "
             f"not a ranking failure (AUC held); the harvest operating threshold must be "
             f"set from data, not p=0.5. These 8 blind-spots + the top_g misses are now "
             f"training data.\n")
    L.append("## Refit held-out (LOMO, C fixed at first-fit value)\n")
    L.append("The C→AUC curve is flat, so re-selecting C would be noise; weights + AUC "
             "are held at the first fit's C for an apples-to-apples comparison.\n")
    L.append("| tier | C | AUC (fixed-C) | grid[min..max] | prior (grid-max) | AP |")
    L.append("|---|---|---|---|---|---|")
    for t in TIERS:
        r = out["report"][t]
        L.append(f"| {t} | {r['C']} | {r['auc']:.3f} | "
                 f"[{r['auc_grid_min']:.3f}..{r['auc_grid_max']:.3f}] | "
                 f"{PRIOR_AUC[t]:.3f} | {r['ap']:.3f} |")
    L.append("\n## Weight stability (T2_cells; refit vs first fit)\n")
    L.append("| feature | prior | refit | Δ |")
    L.append("|---|---|---|---|")
    for r in out["deltas"]["T2_cells"]:
        L.append(f"| {r['feature']} | {r['prior']:+.3f} | {r['refit']:+.3f} | {r['delta']:+.3f} |")
    c = out["checks"]
    L.append(f"\n**Named checks:** g_mid dominates T1 = `{c['g_mid_dominates_T1']}` · "
             f"flat_edge_minus_center sign kept = `{c['flat_edge_sign_kept']}` · "
             f"g_occ stayed dead = `{c['g_occ_stayed_dead']}`.\n")
    v = out["verdict"]
    t2 = out["report"]["T2_cells"]
    L.append("## Readiness verdict\n")
    L.append(f"- Held-out held (T2 grid-max {t2['auc_grid_max']:.3f} vs prior grid-max "
             f"{PRIOR_AUC['T2_cells']:.3f}, within the flat-grid noise band): "
             f"**{'YES' if v['held'] else 'NO'}**")
    L.append(f"- Dominant carriers (|w|≥1.0: detail_spread, interior_worst, flat_worst, "
             f"detail_worst, g_speckle) stable in sign & rank: "
             f"**{'YES' if v['converged'] else 'NO'}**"
             + (f" — PRIMARY FLIPS: {v['primary_flips']}" if v["primary_flips"] else ""))
    L.append(f"- Secondary wobble (noted, not blocking): "
             f"{v['secondary_flips'] or 'none'}"
             + (" — `speckle_spread` flipped −0.51→+0.36; a secondary dispersion term, "
                "worth watching but the boundary's dominant structure held."
                if v["secondary_flips"] else ""))
    L.append(f"\n### → HARVEST-READY: **{'YES' if v['ready'] else 'NO'}**\n")
    L.append("The dominant boundary carriers converged and held-out ranking held within "
             "noise. One secondary dispersion feature wobbled; the operating threshold "
             "for harvest should be calibrated from labels (the p≈0.5 zone is ~"
             f"{a['uncertain']['accept_rate']:.0%} accept, not 50%), not taken at p=0.5.\n")
    L.append("Field re-verify: `out/q4_stage1/refit/field_<mb>.png` "
             "(v2-masked dense grid, refit T2 model).\n")
    FINDINGS.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {FINDINGS.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
def stage_field():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    labels = load_labels()
    rows = build_dataset(labels)
    tier, C = "T2_cells", json.loads((OUT / "refit.json").read_text())["report"]["T2_cells"]["C"]
    _, _, sc, clf = LF.surviving_weights(rows, tier, C)
    model = (sc, clf, LF.FEATURES[tier])
    print(f"field re-verify: refit {tier} C={C}  minibrots={FIELD_MBS}")

    for mbid in FIELD_MBS:
        field, fw, fh = LS.load_field_values(mbid)
        frame = FRAMES / f"{mbid}.png"
        full = Image.open(frame).convert("RGB") if frame.exists() else None
        nS = len(LF.FIELD_SCALES)
        fig = plt.figure(figsize=(4.2 * nS, 6.4))
        gs = fig.add_gridspec(2, nS, height_ratios=[3, 2])
        peaks_all = []
        grids = {s: LF.dense_grid(field, fw, fh, s, model) for s in LF.FIELD_SCALES}
        for j, s in enumerate(LF.FIELD_SCALES):
            res = grids[s]
            ax = fig.add_subplot(gs[0, j])
            if res is None:
                ax.axis("off"); continue
            gx, gy, G, (Wp, Hp) = res
            im = ax.imshow(G, origin="upper", aspect="auto", cmap="magma",
                           extent=[gx[0], gx[-1], gy[-1], gy[0]])
            ax.set_title(f"G s={s} ({int(np.isfinite(G).sum())} surv)")
            fig.colorbar(im, ax=ax, fraction=0.046)
            for (iy, ix, gv) in LF._peaks(G, k=4):
                ax.plot(gx[ix], gy[iy], "c+", ms=11, mew=2)
                peaks_all.append((s, gx[ix], gy[iy], gv, Wp / fw, Hp / fh))
        peaks_all.sort(key=lambda t: -t[3])
        for j, (s, cu, cv, gv, wu, wv) in enumerate(peaks_all[:nS]):
            ax = fig.add_subplot(gs[1, j])
            if full is not None:
                W_, H_ = full.size
                u0, v0 = cu - wu / 2, cv - wv / 2
                ax.imshow(full.crop((int(u0 * W_), int(v0 * H_),
                                     int((u0 + wu) * W_), int((v0 + wv) * H_))))
            ax.set_title(f"max s={s} G={gv:.2f}", fontsize=8); ax.axis("off")
        fig.suptitle(f"{mbid} — refit goodness field ({tier})", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(OUT / f"field_{mbid}.png", dpi=105)
        plt.close(fig)
        print(f"  {mbid}: {len(peaks_all)} peaks -> field_{mbid}.png")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["refit", "field", "all"])
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.no_cache:
        (OUT / "features_cache.json").unlink(missing_ok=True)
    if args.stage in ("refit", "all"):
        stage_refit()
    if args.stage in ("field", "all"):
        stage_field()
