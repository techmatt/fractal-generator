"""Calibrate the mining gate threshold T2 on labeled data.

The mining harness keeps a (location x palette) render for labeling iff its v3
score clears T2. We pick T2 where `v3_score >= T2` best matches `label >= 2`
(not-bad), maximizing F1, on the union of every labeled crop:

  - the 3 v3-training batches (labels baked into images.jsonl)
  - the 600 fresh scale_2x2 labels (labels/scale_2x2_labelset.json) -- these were
    NOT in v3 training, so they are the honest holdout. We pick T2 to max F1 on
    the union but ALSO report the held-out scale_2x2 F1 at that T2 as the honest
    number.

We evaluate two candidate score variables and pick the one with the higher
holdout F1:
  - p_notbad = sigma(logit_0) = P(label>=2)  (the direct not-bad probability)
  - score    = sigma(l0)+sigma(l1) in [0,2]  (the monotone ordinal score)

Writes data/mining/t2_calibration.json: {score_kind, t2, ...metrics}.

  uv run python tools/mining/calibrate_t2.py

MANUAL-ONLY: the descend.py/run.py mining orchestrator was removed. Run by hand to
recalibrate the T2 gate that harvest.py applies. Scores with v3 (DEFAULT_V3) by
design — T2 is defined against the same scorer harvest.py gates with.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from score_lib import Scorer, DEFAULT_V3  # noqa: E402
import corpus_common as cc  # noqa: E402

V3_TRAIN_BATCHES = [
    "2026-06-23_flat_generate_loose0_v3",
    "2026-06-24_guided_descend_rev4",
    "2026-06-24_guided_descend_rev4occfix_v2filtered",
]
HOLDOUT_BATCH = "2026-06-25_scale_2x2_labelset"
HOLDOUT_LABELS = ROOT / "labels" / "scale_2x2_labelset.json"
OUT = ROOT / "data" / "mining" / "t2_calibration.json"


def gather():
    """Yield (crop_path, label, is_holdout) over every labeled crop on disk."""
    rows = []
    for b in V3_TRAIN_BATCHES:
        bdir = cc.batch_dir(b)
        for r in cc.read_jsonl(os.path.join(bdir, "images.jsonl")):
            score = r["label"]["score"]
            if score is None:
                continue
            crop = os.path.join(bdir, "crops", r["image_id"] + ".jpg")
            if os.path.exists(crop):
                rows.append((crop, int(score), False))
    # holdout: scale_2x2 standalone label map
    hdir = cc.batch_dir(HOLDOUT_BATCH)
    labels = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
    for image_id, score in labels.items():
        if score is None:
            continue
        crop = os.path.join(hdir, "crops", image_id + ".jpg")
        if os.path.exists(crop):
            rows.append((crop, int(score), True))
    return rows


def best_threshold(scores: np.ndarray, pos: np.ndarray):
    """Sweep candidate cuts; return (t2, f1, prec, rec) maximizing F1 of
    (scores >= t2) vs pos (bool)."""
    cands = np.unique(scores)
    cands = np.concatenate([[-1e9], cands, [cands.max() + 1e-6]])
    best = (0.0, -1.0, 0.0, 0.0)
    P = pos.sum()
    for t in cands:
        pred = scores >= t
        tp = np.logical_and(pred, pos).sum()
        fp = np.logical_and(pred, ~pos).sum()
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / P
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best[1]:
            best = (float(t), float(f1), float(prec), float(rec))
    return best


def f1_at(scores, pos, t):
    pred = scores >= t
    tp = np.logical_and(pred, pos).sum()
    fp = np.logical_and(pred, ~pos).sum()
    P = pos.sum()
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp)
    rec = tp / P
    return 2 * prec * rec / (prec + rec), prec, rec


def main():
    rows = gather()
    print(f"gathered {len(rows)} labeled crops "
          f"({sum(not h for _, _, h in rows)} train-batch + {sum(h for _, _, h in rows)} holdout)")
    scorer = Scorer(DEFAULT_V3)
    paths = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows])
    holdout = np.array([r[2] for r in rows])
    pos = labels >= 2

    print("scoring with v3...")
    triples = scorer.score_paths(paths)
    score = np.array([t[0] for t in triples])
    p_notbad = np.array([t[1] for t in triples])

    print(f"\nlabel mix: bad={int((labels==1).sum())} okay={int((labels==2).sum())} "
          f"good={int((labels==3).sum())}  not-bad prevalence={pos.mean():.3f}")

    results = {}
    for kind, sc in (("p_notbad", p_notbad), ("score", score)):
        # pick T2 on the UNION (max F1)
        t2, f1u, pu, ru = best_threshold(sc, pos)
        # honest holdout F1 at that same T2
        f1h, ph, rh = f1_at(sc[holdout], pos[holdout], t2)
        results[kind] = dict(t2=t2, union=dict(f1=f1u, prec=pu, rec=ru),
                             holdout=dict(f1=f1h, prec=ph, rec=rh),
                             n_pass=int((sc >= t2).sum()))
        print(f"\n[{kind}] T2={t2:.6g}")
        print(f"  union   F1={f1u:.3f} P={pu:.3f} R={ru:.3f}  ({int((sc>=t2).sum())}/{len(sc)} pass)")
        print(f"  holdout F1={f1h:.3f} P={ph:.3f} R={rh:.3f}  "
              f"({int((sc[holdout]>=t2).sum())}/{int(holdout.sum())} pass)")

    # enrichment curve: precision (= realized not-bad rate in the pool) vs recall
    # as the gate tightens. For prospecting this is the lever Matt cares about.
    print("\nenrichment curve [p_notbad] (union):  base not-bad rate = "
          f"{pos.mean():.3f}")
    print("  T2      pass   precision(=pool not-bad rate)  recall   enrichment")
    for t in (1e-4, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        f1, pr, rc = f1_at(p_notbad, pos, t)
        npass = int((p_notbad >= t).sum())
        print(f"  {t:<7.4f} {npass:<6d} {pr:<29.3f} {rc:<8.3f} {pr/pos.mean():.2f}x")

    # choose the score kind with the better HOLDOUT F1 (honest generalization)
    chosen = max(results, key=lambda k: results[k]["holdout"]["f1"])
    out = dict(
        score_kind=chosen,
        t2=results[chosen]["t2"],
        method="max-F1 on label>=2, T2 picked on union, holdout=scale_2x2 (v3-blind)",
        n_labeled=len(rows),
        not_bad_prevalence=float(pos.mean()),
        results=results,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== CHOSEN: score_kind={chosen}  T2={out['t2']:.4f} "
          f"(holdout F1={results[chosen]['holdout']['f1']:.3f}) ===")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
