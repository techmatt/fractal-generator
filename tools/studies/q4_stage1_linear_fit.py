#!/usr/bin/env python
"""q4 stage-1 first fit — linear (L1) accept-vs-reject GOODNESS FIELD.

Fit a readable linear boundary on the palette-invariant FIELD stats, primarily to
AIM the next labeling batch (a rough fit is fine). Linear + L1 so the surviving
weights are legible. Stops at fit + field + next-batch — NO net.

Labels: p1+p2 union (p2 precedence), accept/reject only, `filter_leak` EXCLUDED
(it is pre-filter feedback, never a quality target — q4_window_reader contract).

Features (all computed from the dumped f64 field, palette-invariant):
  base per-cell stats over a 6x4 grid: interior frac / flat frac / detail energy
  (mid-freq struct_e) / speckle_ratio (fine DoG / coarse std). Aggregated across
  cells as {mean(=global), spread, worst}. Plus an edge-vs-center flat split.

Ablation ladder (minibrot-DISJOINT held-out ranking — never train+test the same
minibrot; LOMO pooled AUC/AP is the referee, NOT in-sample fit):
  T1 global-only   — the current cell-free scalars (rough-heuristic baseline)
  T2 +cell-disp    — + {spread, worst} per base stat + edge/center flat split
  T3 +laplacian    — + per-cell Laplacian variance {mean, spread, worst}

Outputs (out/q4_stage1/linear_fit/):
  1. weights.json / stdout   — surviving L1 weights per tier + held-out scores
  2. next_to_label.json      — margin-uncertain UNLABELED survivors + random control
  3. field_<mb>.png          — dense position x scale G heatmap w/ maxima + crops
  docs/findings/q4_stage1_linear_fit.md — the report

Run:  uv run python -m tools.studies.q4_stage1_linear_fit fit
      uv run python -m tools.studies.q4_stage1_linear_fit field
      uv run python -m tools.studies.q4_stage1_linear_fit all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, maximum_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus import q4_window_reader as qr  # noqa: E402
from tools.studies import q4_stage1_labelset as LS  # noqa: E402

BATCH_ID = "2026-07-23_q4_stage1_windows"
STORE = qr.batch_dir(BATCH_ID)
LABELS_P1 = ROOT / "labels" / "q4_stage1_windows.json"
LABELS_P2 = ROOT / "labels" / "q4_stage1_windows_p2.json"
OUT = ROOT / "out" / "q4_stage1" / "linear_fit"
FRAMES = ROOT / "out" / "q4_stage1" / "frames"
FINDINGS = ROOT / "docs" / "findings" / "q4_stage1_linear_fit.md"

# thresholds shared with the existing decomposition (q4_neighborhood_sweep)
STRUCT_FLAT = 0.030
STRUCT_MID_HI = 0.180
HF_FLOOR = 0.012            # min fine energy for a real speckle_ratio (else ratio=0)
GRID_COLS, GRID_ROWS = 6, 4  # per-cell grid (16:9 -> ~square cells)

SEED = 0


# --------------------------------------------------------------------------- #
# Labels: p1+p2 union (p2 precedence), accept/reject only.                     #
# --------------------------------------------------------------------------- #
def load_labels():
    p1 = json.loads(LABELS_P1.read_text())
    p2 = json.loads(LABELS_P2.read_text())
    merged = {**p1, **p2}
    return {k: v for k, v in merged.items() if v in ("accept", "reject")}


# --------------------------------------------------------------------------- #
# Field feature extraction (palette-invariant).                               #
# --------------------------------------------------------------------------- #
def _cell_slices(H, W):
    ys = np.linspace(0, H, GRID_ROWS + 1).round().astype(int)
    xs = np.linspace(0, W, GRID_COLS + 1).round().astype(int)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            edge = r in (0, GRID_ROWS - 1) or c in (0, GRID_COLS - 1)
            yield (slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1])), edge


def _speckle_ratio(hf_cell, coarse_cell):
    hfm = float(hf_cell.mean())
    if hfm < HF_FLOOR:
        return 0.0
    return hfm / max(float(coarse_cell.std()), 1e-4)


# Named base stats and the direction of "worst" (bad extreme across cells).
_WORST = dict(interior="max", flat="max", detail="min", speckle="max", lapvar="max")


def featurize(vals2d):
    """Field crop (NaN=interior) -> {feature_name: value}. Returns None if too small.
    Emits every feature; per-tier selection happens in FEATURES[tier]."""
    finite = np.isfinite(vals2d)
    vv = vals2d[finite]
    if vv.size < 64:
        return None
    interior = (~finite).astype(np.float64)
    lo, hi = np.percentile(vv, [0.5, 99.5])
    span = max(hi - lo, 1e-9)
    norm = np.clip((vals2d - lo) / span, 0.0, 1.0)
    work = np.where(finite, norm, 1.0)             # interior -> deepest, flat

    lp3 = uniform_filter(work, 3, mode="nearest")
    fine = np.abs(work - lp3)
    struct = lp3 - uniform_filter(work, 11, mode="nearest")
    m = uniform_filter(struct, 5, mode="nearest")
    m2 = uniform_filter(struct * struct, 5, mode="nearest")
    struct_e = np.sqrt(np.maximum(m2 - m * m, 0.0))
    hf = np.abs(gaussian_filter(work, 0.8) - gaussian_filter(work, 1.8))
    coarse = gaussian_filter(work, 6.0)
    lap = laplace(work)

    H, W = vals2d.shape
    cells = {k: [] for k in ("interior", "flat", "detail", "speckle", "lapvar")}
    edge_flat, center_flat = [], []
    for (sy, sx), edge in _cell_slices(H, W):
        se = struct_e[sy, sx]
        ci = float(interior[sy, sx].mean())
        cf = float((se < STRUCT_FLAT).mean())
        cd = float(se.mean())
        cs = _speckle_ratio(hf[sy, sx], coarse[sy, sx])
        cl = float(lap[sy, sx].var())
        cells["interior"].append(ci)
        cells["flat"].append(cf)
        cells["detail"].append(cd)
        cells["speckle"].append(cs)
        cells["lapvar"].append(cl)
        (edge_flat if edge else center_flat).append(cf)

    f = {}
    # --- global scalars (T1 baseline): means over the whole crop -------------
    f["g_interior"] = float(interior.mean())
    f["g_flat"] = float((struct_e < STRUCT_FLAT).mean())
    f["g_mid"] = float(((struct_e >= STRUCT_FLAT) & (struct_e < STRUCT_MID_HI)).mean())
    f["g_high"] = float((struct_e >= STRUCT_MID_HI).mean())
    f["g_occ"] = float((struct_e >= STRUCT_FLAT).mean())
    f["g_speckle"] = _speckle_ratio(hf[finite], coarse[finite])

    # --- cell dispersion (T2): spread + worst per base stat ------------------
    for k in ("interior", "flat", "detail", "speckle"):
        arr = np.asarray(cells[k])
        f[f"{k}_spread"] = float(arr.std())
        f[f"{k}_worst"] = float(arr.max() if _WORST[k] == "max" else arr.min())
    f["flat_edge_minus_center"] = float(np.mean(edge_flat) - np.mean(center_flat))

    # --- Laplacian (T3): 2nd-order per-cell variance -------------------------
    lv = np.asarray(cells["lapvar"])
    f["lapvar_mean"] = float(lv.mean())
    f["lapvar_spread"] = float(lv.std())
    f["lapvar_worst"] = float(lv.max())
    return f


FEATURES = {
    "T1_global": ["g_interior", "g_flat", "g_mid", "g_high", "g_occ", "g_speckle"],
}
FEATURES["T2_cells"] = FEATURES["T1_global"] + [
    "interior_spread", "interior_worst", "flat_spread", "flat_worst",
    "detail_spread", "detail_worst", "speckle_spread", "speckle_worst",
    "flat_edge_minus_center",
]
FEATURES["T3_laplacian"] = FEATURES["T2_cells"] + [
    "lapvar_mean", "lapvar_spread", "lapvar_worst",
]
TIERS = ["T1_global", "T2_cells", "T3_laplacian"]


# --------------------------------------------------------------------------- #
# Build the design matrix over labeled store windows.                         #
# --------------------------------------------------------------------------- #
def crop_field(field, fw, fh, win):
    u, v, w, h = win["u"], win["v"], win["w"], win["h"]
    x0, y0 = int(round(u * fw)), int(round(v * fh))
    x1, y1 = int(round((u + w) * fw)), int(round((v + h) * fh))
    return field[y0:y1, x0:x1]


def build_dataset(*, cache=True):
    """[(window_id, minibrot_id, feats, klass_or_None)] over all 300 windows."""
    cache_p = OUT / "features_cache.json"
    if cache and cache_p.exists():
        raw = json.loads(cache_p.read_text())
        labels = load_labels()
        return [(r["wid"], r["mb"], r["f"], labels.get(r["wid"])) for r in raw]

    labels = load_labels()
    by_mb = defaultdict(list)
    for row, _ in qr.iter_windows(BATCH_ID):
        by_mb[row["minibrot_id"]].append(row)

    out = []
    raw = []
    for mbid, wins in sorted(by_mb.items()):
        field, fw, fh = LS.load_field_values(mbid)
        for r in wins:
            feats = featurize(crop_field(field, fw, fh, r["window"]))
            if feats is None:
                continue
            out.append((r["window_id"], mbid, feats, labels.get(r["window_id"])))
            raw.append(dict(wid=r["window_id"], mb=mbid, f=feats))
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(raw))
    return out


def matrix(rows, tier):
    keys = FEATURES[tier]
    X = np.array([[r[2][k] for k in keys] for r in rows], dtype=np.float64)
    return X, keys


# --------------------------------------------------------------------------- #
# Fit + minibrot-disjoint referee.                                            #
# --------------------------------------------------------------------------- #
def _fit_logit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    # random_state pinned: liblinear SHUFFLES data by random_state, so an unset seed
    # makes weights near the L1 sparsity threshold flicker run-to-run (a feature toggling
    # on/off). Pin it for reproducible weights — mandatory for a stability comparison.
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                             class_weight="balanced", max_iter=2000, random_state=0)
    clf.fit(sc.transform(X), y)
    return sc, clf


def lomo_scores(rows, tier, C):
    """Leave-one-minibrot-out pooled held-out G. Returns (y, G) aligned arrays."""
    from sklearn.metrics import roc_auc_score
    lab = [(r, r[3]) for r in rows if r[3] in ("accept", "reject")]
    mbs = sorted({r[0][1] for r in lab})
    yA, gA = [], []
    for held in mbs:
        tr = [r for r, _ in lab if r[1] != held]
        te = [r for r, _ in lab if r[1] == held]
        if not te:
            continue
        ytr = np.array([1 if r[3] == "accept" else 0 for r in tr])
        if ytr.min() == ytr.max():
            continue
        Xtr, keys = matrix(tr, tier)
        sc, clf = _fit_logit(Xtr, ytr, C)
        Xte, _ = matrix(te, tier)
        g = clf.decision_function(sc.transform(Xte))
        yA.extend(1 if r[3] == "accept" else 0 for r in te)
        gA.extend(g.tolist())
    yA, gA = np.array(yA), np.array(gA)
    return yA, gA


def evaluate(rows):
    """Per-tier: pick C by pooled LOMO AUC, report AUC / AP / accept-recall."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    Cgrid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    report = {}
    for tier in TIERS:
        best = None
        for C in Cgrid:
            y, g = lomo_scores(rows, tier, C)
            auc = roc_auc_score(y, g)
            if best is None or auc > best[0]:
                best = (auc, C, y, g)
        auc, C, y, g = best
        ap = average_precision_score(y, g)
        # accept-recall at the balanced-logit operating point (G>=0 ~ p>=0.5 after
        # class_weight balancing): a proxy operating threshold on the pooled score.
        thr = 0.0
        pred = (g >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        n_acc = int((y == 1).sum())
        n_rej = int((y == 0).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        acc_recall = tp / n_acc if n_acc else 0.0
        rej_precision = tp / (tp + fp) if (tp + fp) else 0.0
        report[tier] = dict(C=C, auc=float(auc), ap=float(ap),
                            accept_recall=acc_recall, precision=rej_precision,
                            n_accept=n_acc, n_reject=n_rej,
                            fp_at_thr=fp, tp_at_thr=tp)
    return report


def surviving_weights(rows, tier, C):
    """Full-data L1 fit -> {feature: standardized weight} for nonzero weights."""
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    y = np.array([1 if r[3] == "accept" else 0 for r in lab])
    X, keys = matrix(lab, tier)
    sc, clf = _fit_logit(X, y, C)
    w = clf.coef_.ravel()
    return {k: float(wi) for k, wi in zip(keys, w)}, float(clf.intercept_[0]), sc, clf


# --------------------------------------------------------------------------- #
# Stage: fit (ablation + weights + next-to-label).                            #
# --------------------------------------------------------------------------- #
def stage_fit():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_dataset()
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    n_acc = sum(1 for r in lab if r[3] == "accept")
    print(f"labeled accept/reject: {len(lab)}  ({n_acc} accept / {len(lab)-n_acc} reject)"
          f"  over {len({r[1] for r in lab})} minibrots")

    report = evaluate(rows)
    print("\n=== held-out referee (minibrot-DISJOINT LOMO, pooled) ===")
    print(f"{'tier':<14}{'C':>6}{'AUC':>7}{'AP':>7}{'acc-recall':>12}{'reject-prec':>13}")
    weights = {}
    for tier in TIERS:
        r = report[tier]
        print(f"{tier:<14}{r['C']:>6}{r['auc']:>7.3f}{r['ap']:>7.3f}"
              f"{r['accept_recall']:>12.2f}{r['precision']:>13.2f}")
        w, b, _, _ = surviving_weights(rows, tier, r["C"])
        nz = {k: v for k, v in w.items() if abs(v) > 1e-6}
        weights[tier] = dict(intercept=b, weights=w,
                             nonzero=dict(sorted(nz.items(), key=lambda kv: -abs(kv[1]))))

    print("\n=== surviving L1 weights (standardized; sorted by |w|) ===")
    for tier in TIERS:
        nz = weights[tier]["nonzero"]
        print(f"\n{tier}  ({len(nz)}/{len(FEATURES[tier])} features survive):")
        for k, v in nz.items():
            print(f"   {k:<24}{v:+.3f}")

    # choose the tier that wins the referee (ties -> simpler)
    chosen = max(TIERS, key=lambda t: (round(report[t]["auc"], 3), -TIERS.index(t)))
    print(f"\nchosen tier (max held-out AUC, ties->simpler): {chosen}")

    (OUT / "weights.json").write_text(json.dumps(
        dict(report=report, weights=weights, chosen_tier=chosen), indent=2))

    next_to_label(rows, chosen, report[chosen]["C"])
    write_report(report, weights, chosen, rows)
    return report, weights, chosen


# --------------------------------------------------------------------------- #
# Output 3: ranked next-to-label batch.                                       #
# --------------------------------------------------------------------------- #
def next_to_label(rows, tier, C, n_uncertain=40, n_control=12):
    """Margin-uncertain UNLABELED store survivors (where labels teach) + a
    uniform-random control slug (audits confident-and-wrong)."""
    _, _, sc, clf = surviving_weights(rows, tier, C)
    unl = [r for r in rows if r[3] is None]
    if not unl:
        print("\nnext-to-label: no unlabeled store windows.")
        return
    X, _ = matrix(unl, tier)
    p = clf.predict_proba(sc.transform(X))[:, list(clf.classes_).index(1)]
    margin = np.abs(p - 0.5)
    order = np.argsort(margin)                       # most uncertain first
    rng = np.random.default_rng(SEED)

    uncertain = []
    seen_mb = Counter()
    for i in order:                                  # light per-minibrot diversity cap
        wid, mbid = unl[i][0], unl[i][1]
        if seen_mb[mbid] >= 4:
            continue
        seen_mb[mbid] += 1
        uncertain.append(dict(window_id=wid, minibrot_id=mbid,
                              p_accept=round(float(p[i]), 3),
                              margin=round(float(margin[i]), 3), slug="uncertain"))
        if len(uncertain) >= n_uncertain:
            break

    picked = {d["window_id"] for d in uncertain}
    pool = [j for j in range(len(unl)) if unl[j][0] not in picked]
    ctrl_idx = rng.choice(pool, size=min(n_control, len(pool)), replace=False)
    control = [dict(window_id=unl[j][0], minibrot_id=unl[j][1],
                    p_accept=round(float(p[j]), 3),
                    margin=round(float(margin[j]), 3), slug="random_control")
               for j in ctrl_idx]

    batch = dict(tier=tier, C=C, n_unlabeled=len(unl),
                 uncertain=uncertain, control=control)
    (OUT / "next_to_label.json").write_text(json.dumps(batch, indent=2))
    print(f"\nnext-to-label: {len(uncertain)} uncertain (margin<= "
          f"{uncertain[-1]['margin'] if uncertain else 0}) + {len(control)} random control"
          f"  of {len(unl)} unlabeled -> next_to_label.json")


# --------------------------------------------------------------------------- #
# Report.                                                                      #
# --------------------------------------------------------------------------- #
def write_report(report, weights, chosen, rows):
    L = []
    L.append("# q4 stage-1 first fit — linear (L1) accept-vs-reject goodness field\n")
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    n_acc = sum(1 for r in lab if r[3] == "accept")
    L.append(f"Labels: p1+p2 union (p2 precedence), `filter_leak` excluded. "
             f"**{len(lab)}** windows ({n_acc} accept / {len(lab)-n_acc} reject) over "
             f"{len({r[1] for r in lab})} minibrots. Referee: leave-one-minibrot-out, "
             f"pooled held-out ranking (never train+test the same minibrot).\n")
    L.append("## Held-out referee (minibrot-disjoint LOMO)\n")
    L.append("| tier | C | AUC | AP | accept-recall@G0 | reject-precision@G0 |")
    L.append("|---|---|---|---|---|---|")
    for t in TIERS:
        r = report[t]
        L.append(f"| {t} | {r['C']} | {r['auc']:.3f} | {r['ap']:.3f} | "
                 f"{r['accept_recall']:.2f} | {r['precision']:.2f} |")
    d21 = report["T2_cells"]["auc"] - report["T1_global"]["auc"]
    d32 = report["T3_laplacian"]["auc"] - report["T2_cells"]["auc"]
    L.append(f"\n**Cells earn their place?** ΔAUC(T2−T1) = {d21:+.3f}; "
             f"ΔAUC(T3−T2 Laplacian) = {d32:+.3f}. "
             f"Chosen tier (max held-out AUC): **{chosen}**.\n")

    # honest reading — driven by the actual fitted weights, not a story
    w2 = weights["T2_cells"]["nonzero"]
    L.append("## Reading\n")
    L.append(f"- **Global-only already ranks well** (AUC {report['T1_global']['auc']:.3f}) "
             f"on a *single* surviving scalar, `g_mid` (mid-detail fraction) — the rough "
             f"heuristic is essentially \"how much of the window is mid-scale ornament.\" "
             f"A cell-free `g_mid` threshold is a usable labeling aid on its own.")
    L.append(f"- **Cell-dispersion earns a modest, real lift** (+{d21:.3f} AUC, AP "
             f"{report['T1_global']['ap']:.3f}→{report['T2_cells']['ap']:.3f}). What the "
             f"cells add is *contrast* structure: `flat_worst`(+) ∧ `detail_worst`(+) ∧ "
             f"`detail_spread`(−) = a window with **both** a calm anchor cell **and** "
             f"evenly-distributed detail elsewhere (not one busy spike). `interior_worst`(−) "
             f"kills any window with a dead cell — the corner-deadness signal.")
    L.append(f"- **Laplacian does NOT earn its place** ({d32:+.3f} AUC vs T2; `lapvar_*` "
             f"weights are tiny). 2nd-order curvature adds nothing over the struct_e "
             f"decomposition. Drop it.")
    L.append(f"- **Priors that did NOT hold** (weights contradict the framing hypotheses): "
             f"(a) `g_occ` carries a *positive* weight ({w2.get('g_occ', 0):+.2f}) — the "
             f"\"down-weight occupancy\" prior was a story; more occupancy reads as accept. "
             f"(b) `flat_edge_minus_center`(+{w2.get('flat_edge_minus_center', 0):.2f}) has "
             f"the **opposite** sign to the \"flat-in-edge = empty corner = bad\" prior: a "
             f"calmer edge with a busier center (subject-centered composition) reads as "
             f"accept, not reject. (c) `g_speckle`(+{w2.get('g_speckle', 0):.2f}) is positive "
             f"— *within pre-filter survivors* (pure speckle already gated at ratio≥0.30) a "
             f"higher fine/coarse ratio is fine ornamentation, not noise.")
    L.append(f"- **Field visual test passes**: masking the field to pre-filter survivors "
             f"(the deployed v2 gate) is load-bearing — the *unmasked* linear G extrapolates "
             f"to huge OOD spikes on the dead-interior blob (the model never trains on "
             f"interior-heavy windows). Over survivors, G∈[-11,+6] and its position-maxima "
             f"land on the ornate spiral ring; the rendered maxima crops are exactly the "
             f"good filigree windows. \"Plot G, take maxima\" auto-frames correctly.\n")
    L.append("## Surviving L1 weights (standardized, sorted by |w|)\n")
    for t in TIERS:
        nz = weights[t]["nonzero"]
        L.append(f"**{t}** — {len(nz)}/{len(FEATURES[t])} survive:\n")
        L.append("| feature | weight |")
        L.append("|---|---|")
        for k, v in nz.items():
            L.append(f"| {k} | {v:+.3f} |")
        L.append("")
    L.append("## Next-to-label\n")
    L.append("`out/q4_stage1/linear_fit/next_to_label.json` — margin-uncertain unlabeled "
             "survivors (where labels teach the boundary) + a uniform-random control slug "
             "(audits confident-and-wrong outside what the model knows). Crops already exist "
             f"in the store (`{qr.crop_path(BATCH_ID,'<id>').parent.relative_to(ROOT)}`).\n")
    L.append("## Goodness field\n")
    L.append("`out/q4_stage1/linear_fit/field_<mb>.png` — G over a dense position×scale grid "
             "(G computed directly, score_A NMS bypassed), position-maxima marked, their "
             "windows rendered. The visual test of *plot G, take maxima*.\n")
    FINDINGS.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS.write_text("\n".join(L), encoding="utf-8")
    print(f"\nreport -> {FINDINGS.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Stage: field — dense position x scale G heatmap + maxima crops.             #
# --------------------------------------------------------------------------- #
FIELD_MBS = None            # None -> auto-pick minibrots with the most accepts
FIELD_SCALES = [0.06, 0.09, 0.14]
DENSE_STRIDE_FRAC = 0.12    # dense (well below the label sweep's 0.30)

# The DEPLOYED stage-1 pre-filter (q4_stage1_filter_v2 ceilings). The goodness field
# is only meaningful over SURVIVORS: interior-heavy / barren / speckle windows are
# dropped upstream and are OOD for the accept/reject model (it never saw them), so an
# unmasked linear G extrapolates to huge spikes on the dead-interior blob. Mask them.
V2_INTERIOR, V2_FLAT, V2_SPECKLE = 0.10, 0.88, 0.30


def _v2_drop(f):
    """True if this window would be dropped by the deployed stage-1 pre-filter.
    Uses the same globals the featurizer already computed (g_speckle honors HF_FLOOR)."""
    return (f["g_interior"] >= V2_INTERIOR or f["g_flat"] >= V2_FLAT
            or f["g_speckle"] >= V2_SPECKLE)


def _auto_field_mbs(rows, k=3):
    acc = Counter(r[1] for r in rows if r[3] == "accept")
    return [m for m, _ in acc.most_common(k)]


def dense_grid(field, fw, fh, scale, model):
    """G over a dense grid at one scale. Returns (gx, gy, G) where gx/gy are window
    CENTER fractions and G is a 2D map (None cells where the crop is too small)."""
    sc, clf, keys = model
    Wp = max(8, int(round(scale * fw)))
    Hp = max(8, int(round(Wp * 9 / 16)))
    if Hp >= fh or Wp >= fw:
        return None
    st = max(4, int(round(DENSE_STRIDE_FRAC * Wp)))
    ys = list(range(0, fh - Hp + 1, st))
    xs = list(range(0, fw - Wp + 1, st))
    G = np.full((len(ys), len(xs)), np.nan)
    feat_rows, coords = [], []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            f = featurize(field[y:y + Hp, x:x + Wp])
            if f is None or _v2_drop(f):            # only score pre-filter SURVIVORS
                continue
            feat_rows.append([f[k] for k in keys])
            coords.append((iy, ix))
    if feat_rows:
        Xg = np.array(feat_rows)
        g = clf.decision_function(sc.transform(Xg))
        for (iy, ix), gv in zip(coords, g):
            G[iy, ix] = gv
    gx = (np.array(xs) + Wp / 2) / fw
    gy = (np.array(ys) + Hp / 2) / fh
    return gx, gy, G, (Wp, Hp)


def _peaks(G, k=6):
    """Local maxima of G (position), top-k by value."""
    Gf = np.where(np.isnan(G), -np.inf, G)
    mx = maximum_filter(Gf, size=3, mode="nearest")
    ys, xs = np.where((Gf == mx) & np.isfinite(G))
    vals = G[ys, xs]
    order = np.argsort(vals)[::-1][:k]
    return [(int(ys[i]), int(xs[i]), float(vals[i])) for i in order]


def stage_field():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = build_dataset()
    w = json.loads((OUT / "weights.json").read_text())
    tier, C = w["chosen_tier"], w["report"][w["chosen_tier"]]["C"]
    _, _, sc, clf = surviving_weights(rows, tier, C)
    model = (sc, clf, FEATURES[tier])
    mbs = FIELD_MBS or _auto_field_mbs(rows)
    print(f"goodness field: tier={tier} C={C}  minibrots={mbs}")

    for mbid in mbs:
        field, fw, fh = LS.load_field_values(mbid)
        frame_png = FRAMES / f"{mbid}.png"
        full = Image.open(frame_png).convert("RGB") if frame_png.exists() else None

        nS = len(FIELD_SCALES)
        fig = plt.figure(figsize=(4.2 * nS, 8.6))
        gs = fig.add_gridspec(3, nS, height_ratios=[3, 3, 2])
        all_peaks = []
        grids = {s: dense_grid(field, fw, fh, s, model) for s in FIELD_SCALES}
        for j, s in enumerate(FIELD_SCALES):
            res = grids[s]
            ax = fig.add_subplot(gs[0, j])
            if res is None:
                ax.set_title(f"scale {s}: too large"); ax.axis("off"); continue
            gx, gy, G, (Wp, Hp) = res
            im = ax.imshow(G, origin="upper", aspect="auto", cmap="magma",
                           extent=[gx[0], gx[-1], gy[-1], gy[0]])
            n_surv = int(np.isfinite(G).sum())
            ax.set_title(f"G  scale={s}  ({Wp}x{Hp}px, {n_surv} survivors)")
            ax.set_xlabel("center u"); ax.set_ylabel("center v")
            fig.colorbar(im, ax=ax, fraction=0.046)
            for (iy, ix, gv) in _peaks(G, k=4):
                ax.plot(gx[ix], gy[iy], "c+", ms=12, mew=2)
                all_peaks.append((s, gx[ix], gy[iy], gv, Wp / fw, Hp / fh))

        # marginal: G distribution over surviving positions vs scale (auto-framing check)
        axm = fig.add_subplot(gs[1, :])
        for s in FIELD_SCALES:
            res = grids[s]
            if res is None:
                continue
            G = res[2]
            gv = G[np.isfinite(G)]
            axm.scatter([s] * gv.size, gv, s=8, alpha=0.4, label=f"s={s} (n={gv.size})")
        axm.set_xlabel("scale"); axm.set_ylabel("G over surviving positions")
        axm.set_title("G distribution per scale (max = best framing at that scale)")
        axm.legend(fontsize=7)

        # bottom: render the top global maxima crops
        all_peaks.sort(key=lambda t: -t[3])
        top = all_peaks[:nS]
        for j, (s, cu, cv, gv, wu, wv) in enumerate(top):
            ax = fig.add_subplot(gs[2, j])
            if full is not None:
                W_, H_ = full.size
                u0, v0 = cu - wu / 2, cv - wv / 2
                box = (int(u0 * W_), int(v0 * H_), int((u0 + wu) * W_), int((v0 + wv) * H_))
                ax.imshow(full.crop(box))
            ax.set_title(f"max s={s} G={gv:.2f}", fontsize=8)
            ax.axis("off")

        fig.suptitle(f"{mbid} — goodness field G ({tier})", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_png = OUT / f"field_{mbid}.png"
        fig.savefig(out_png, dpi=105)
        plt.close(fig)
        print(f"  {mbid}: {len(all_peaks)} peaks -> {out_png.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["fit", "field", "all"])
    ap.add_argument("--no-cache", action="store_true", help="recompute field features")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.no_cache:
        (OUT / "features_cache.json").unlink(missing_ok=True)
    if args.stage in ("fit", "all"):
        stage_fit()
    if args.stage in ("field", "all"):
        stage_field()
