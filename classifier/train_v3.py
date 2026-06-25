"""Train the v3 aesthetic classifier — three-batch union with the v2-filtered
enrichment batch evaluated HONESTLY (loose0 + run4 + rev4occfix_v2filtered).

Recipe (architecture, augmentation, optimizer, loss, sampler, save policy) is
HELD FIXED from v1/v2 (imported from model.py / data.py / eval_v2.py) so any
metric delta is attributable to the corpus, not the recipe. Only the data
composition + the eval-integrity split logic are new.

The hazard this script defends against is MIXED SELECTION BIAS: the 966
`enriched` crops are v2-selected, so any rev4 metric they touch is inflated.
The integrity rules (see prompts/v3-finetune-threebatch.md):

  * Training set = full union, provenance-BLIND to the model (enriched included
    — that's the point of enriching). selection_role/walk_id are read ONLY here,
    never as model input.
  * rev4 eval = UNBIASED samples only: all of run4 + batch-2 `random_eval` (100).
    The 966 `enriched` are TRAIN-LOCKED — never in any val/holdout fold.
  * Walk-purity on batch 2: any walk containing a `random_eval` location is
    EVAL-ONLY; its `enriched` siblings DROP from training (else within-walk
    correlation leaks the eval). Counted + reported.
  * Batch-qualified grouping (batch_id, correlation_unit): walk for the descent
    batches (run4, batch2), seed for flat loose0. StratifiedGroupKFold.

Headline ablation — did the 966 enriched labels earn their cost?
  * v3-full       : trained on all three batches (incl. enriched).
  * v3-no-enriched: trained on loose0 + run4 + batch2-random_eval only.
  Both scored on the SAME unbiased rev4 eval (holdout run4 + random_eval).

  uv run python -m classifier.train_v3

Outputs -> data/classifier/v3/ (does NOT touch v1 or v2).
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import eval_v2 as ev
from .corpus_data import load_corpus_rows, census
from .model import BACKBONE, build_model, data_config
from .train_v2 import (LOOSE0, detect_device, fit, hist, metrics_union,
                       score_rows, set_seed)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v3"
log = logging.getLogger("train_v3")

RUN4 = "2026-06-24_guided_descend_rev4"            # unfiltered descent -> wholly unbiased rev4
BATCH2 = "2026-06-24_guided_descend_rev4occfix_v2filtered"  # v2-filtered: enriched + random_eval


# --------------------------------------------------------------------------- #
# Eval-integrity partition: who may be evaluated, who is train-locked, who drops.
# --------------------------------------------------------------------------- #
def partition_rows(rows):
    """Split the union into eval-eligible (A), train-locked enriched (B), and
    walk-purity drops. Returns (A, B, info-dict).

      A  = eval-eligible (UNBIASED): loose0, run4, batch2 `random_eval`. These
           are the ONLY rows allowed into val/holdout folds + any rev4 number.
      B  = TRAIN-LOCKED enriched: batch2 `enriched` NOT sharing a walk with any
           `random_eval` row. Always on the training side, never evaluated.
      dropped = batch2 `enriched` that DO share a walk with a `random_eval` row.
           Eval-only walk + biased row => neither trained nor evaluated.
    """
    re_walks = {r.group_unit for r in rows if r.selection_role == "random_eval"}
    A, B, dropped = [], [], []
    for r in rows:
        if r.selection_role == "enriched":
            (dropped if r.group_unit in re_walks else B).append(r)
        else:  # None (loose0/run4) or "random_eval"
            A.append(r)
    info = {
        "n_eval_eligible": len(A), "n_train_locked_enriched": len(B),
        "n_dropped_enriched_walk_purity": len(dropped),
        "n_random_eval_walks": len(re_walks),
    }
    return A, B, info


def eval_batch_of(r) -> str:
    """Coarse eval-bucket for the per-batch breakdown. Only these three ever
    reach an eval set (enriched is train-locked)."""
    if r.batch_id == LOOSE0:
        return "loose0"
    if r.batch_id == RUN4:
        return "run4"
    return "batch2_random_eval"  # batch2 rows in eval are random_eval by construction


def is_rev4_eval(r) -> bool:
    """Unbiased rev4 sample: run4 (any) or batch2 random_eval."""
    return r.batch_id == RUN4 or (r.batch_id == BATCH2 and r.selection_role == "random_eval")


def fold_train_rows(A_idx, A, B):
    """Training rows for a fold/holdout model = eval-eligible-train slice + ALL
    train-locked enriched (B). B is provenance-blind to the model."""
    return [A[i] for i in A_idx] + list(B)


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v3 three-batch classifier.")
    ap.add_argument("--target", choices=["ordinal", "binary"], default="ordinal")
    ap.add_argument("--geometry", choices=["stretch", "pad", "letterbox384"], default="stretch")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-jpeg-aug", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    set_seed(args.seed)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    # --- load union (all 3 batches, black-gated; loader verifies crops on disk) ---
    rows = load_corpus_rows()
    cen = census(rows)
    log.info(f"rows after black filter (<0.30): {len(rows)}")
    for k, v in cen.items():
        log.info(f"  {k}: {v}")

    # --- eval-integrity partition ---
    A, B, pinfo = partition_rows(rows)
    log.info("=== eval-integrity partition ===")
    log.info(f"  A eval-eligible (loose0+run4+random_eval): {len(A)} {hist(A)}")
    for b in (LOOSE0, RUN4, BATCH2):
        ab = [r for r in A if r.batch_id == b]
        log.info(f"    A/{b}: n={len(ab)} {hist(ab)}")
    log.info(f"  B train-locked enriched: {len(B)} {hist(B)}")
    log.info(f"  DROPPED enriched (walk-purity, in {pinfo['n_random_eval_walks']} "
             f"random_eval walks): {pinfo['n_dropped_enriched_walk_purity']}")

    cfg = {
        "target": args.target, "geometry": args.geometry, "epochs": args.epochs,
        "batch_size": args.batch_size, "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "drop_rate": args.drop_rate, "drop_path_rate": args.drop_path_rate,
        "patience": args.patience, "seed": args.seed, "num_workers": args.num_workers,
        "amp": "off", "grad_clip": 1.0, "no_jpeg_aug": args.no_jpeg_aug,
        "sampler": "WeightedRandomSampler(sqrt-inv-freq)",
        "loss": ("CORN ordinal (K-1=2)" if args.target == "ordinal" else "BCE 1-vs-{2,3}"),
        "folds": args.folds, "holdout_frac": args.holdout_frac,
        "corpus_batches": [LOOSE0, RUN4, BATCH2],
        "group_keys": {"loose0": "seed_index", "run4": "walk_id", "batch2": "walk_id"},
        "provenance_blind": True,
        "eval_integrity": {
            "rev4_eval": "run4 (any) + batch2 random_eval ONLY",
            "enriched": "train-locked (never in val/holdout/eval)",
            "walk_purity": "random_eval walks eval-only; enriched siblings dropped",
            **pinfo,
        },
    }
    probe = build_model(target=args.target, pretrained=True)
    data_cfg = data_config(probe); del probe
    log.info(f"data_config: {data_cfg}")
    cfg.update({"mean": data_cfg["mean"], "std": data_cfg["std"],
                "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
                "backbone": BACKBONE, "src_dims": [1280, 720], "target_dims": [384, 224],
                "black_thresh": 0.30})

    # --- splits computed over A ONLY (enriched can never land in val/holdout) ---
    groups_A, strat_A = ev.group_strat_arrays(A)
    metrics = {"partition": pinfo, "cv": None, "holdout": None,
               "per_batch_holdout": None, "rev4_unbiased": None, "enrichment_ablation": None}

    # fixed grouped, batch×label-stratified holdout over A; CV on the A-remainder.
    trA, hoA = ev.make_grouped_holdout(groups_A, strat_A, frac=args.holdout_frac, seed=args.seed)
    rows_ho = [A[i] for i in hoA]
    log.info(f"=== fixed holdout (A only): {len(hoA)} crops {hist(rows_ho)} / "
             f"A-remainder {len(trA)} ===")
    ho_eb = Counter(eval_batch_of(r) for r in rows_ho)
    log.info(f"  holdout per eval-batch: {dict(ho_eb)}")
    for eb in ("loose0", "run4", "batch2_random_eval"):
        hr = [r for r in rows_ho if eval_batch_of(r) == eb]
        log.info(f"    holdout/{eb}: n={len(hr)} {hist(hr)}")

    # --- StratifiedGroupKFold CV on the A-remainder (train += all B) ---
    if args.folds and args.folds >= 2:
        log.info(f"=== CV ({args.folds} folds, batch-qualified groups; train += {len(B)} enriched) ===")
        fold_crop, fold_loc, fold_counts = [], [], []
        for fi, (tr_idx, va_idx) in enumerate(
                ev.make_grouped_folds(groups_A, strat_A, subset_idx=trA,
                                      n_splits=args.folds, seed=args.seed)):
            rows_tr = fold_train_rows(tr_idx, A, B)
            rows_va = [A[i] for i in va_idx]
            vh = hist(rows_va)
            # zero-leakage check: no val group appears in train (A or B side).
            tr_groups = set(A[i].group_unit for i in tr_idx) | set(r.group_unit for r in B)
            va_groups = set(A[i].group_unit for i in va_idx)
            leak = tr_groups & va_groups
            assert not leak, f"fold {fi} group leakage: {list(leak)[:5]}"
            log.info(f"--- fold {fi}: train {len(rows_tr)} {hist(rows_tr)} "
                     f"val {len(rows_va)} {vh} val_groups {len(va_groups)} (leak=0) ---")
            m, _, be, _, _ = fit(rows_tr, rows_va, cfg, data_cfg, device)
            log.info(f"--- fold {fi} best ep {be}: AP_notbad {m['ap_not_bad']:.4f} "
                     f"AP_good {m['ap_good']:.4f} loc_AP_notbad {m['loc_ap_not_bad']:.4f} ---")
            fold_crop.append({k: m[k] for k in
                              ["ap_not_bad", "ap_good", "p_at_10_not_bad", "p_at_25_not_bad",
                               "p_at_10_good", "p_at_25_good"]})
            fold_loc.append({k: m[k] for k in
                             ["loc_ap_not_bad", "loc_ap_good", "loc_p_at_10_not_bad",
                              "loc_p_at_25_not_bad"]})
            fold_counts.append({"val_hist": vh, "n_good": vh[3], "n_train": len(rows_tr)})
        metrics["cv"] = {"per_fold_crop": fold_crop, "per_fold_loc": fold_loc,
                         "per_fold_counts": fold_counts,
                         "crop": ev.aggregate_folds(fold_crop),
                         "loc": ev.aggregate_folds(fold_loc)}
        log.info(f"CV crop AP_notbad: {metrics['cv']['crop']['ap_not_bad']}")
        log.info(f"CV loc  AP_notbad: {metrics['cv']['loc']['loc_ap_not_bad']}")

    # --- v3-FULL deliverable model: train = A-remainder + all enriched; select on holdout ---
    log.info("=== v3-FULL holdout model (train=A-remainder + enriched, select=holdout) ===")
    rows_tr_full = fold_train_rows(trA, A, B)
    m_u, scores_u, be_u, hist_u, state_u = fit(rows_tr_full, rows_ho, cfg, data_cfg, device,
                                               save_dir=out_dir)
    cfg["best_epoch"] = be_u
    metrics["holdout"] = {**m_u, "best_epoch": be_u, "history": hist_u}
    log.info(f"v3-FULL holdout best ep {be_u}: AP_notbad {m_u['ap_not_bad']:.4f} "
             f"loc_AP_notbad {m_u['loc_ap_not_bad']:.4f}")

    # --- per-batch holdout breakdown (loose0 / run4 / batch2_random_eval) ---
    pb = ev.per_batch_metrics(scores_u, [r.label for r in rows_ho],
                              [eval_batch_of(r) for r in rows_ho],
                              [r.loc_unit for r in rows_ho])
    metrics["per_batch_holdout"] = pb
    for b, mm in pb.items():
        log.info(f"  [holdout/{b}] crop AP_notbad {mm['ap_not_bad']:.4f}  "
                 f"loc AP_notbad {mm['loc_ap_not_bad']:.4f}  "
                 f"(n={mm['n']}, not_bad={mm['n_not_bad']}, good={mm['n_good']})")

    # --- UNBIASED rev4 headline: holdout (run4 + random_eval) combined ---
    rev4_mask = np.array([is_rev4_eval(r) for r in rows_ho])
    rev4_rows = [r for r, k in zip(rows_ho, rev4_mask) if k]
    rev4_scores_full = scores_u[rev4_mask]
    m_rev4 = metrics_union(rev4_scores_full, rev4_rows)
    metrics["rev4_unbiased"] = {**m_rev4, "n": len(rev4_rows),
                                "composition": dict(Counter(eval_batch_of(r) for r in rev4_rows))}
    log.info(f"=== UNBIASED rev4 (holdout run4+random_eval, n={len(rev4_rows)}): "
             f"loc AP_notbad {m_rev4['loc_ap_not_bad']:.4f}  "
             f"crop AP_notbad {m_rev4['ap_not_bad']:.4f}  (vs v2 ~0.47) ===")

    # --- ENRICHMENT ABLATION: v3-full vs v3-no-enriched on the SAME rev4 eval ---
    log.info("=== enrichment ablation: train v3-no-enriched (A-remainder only) ===")
    rows_tr_ne = [A[i] for i in trA]  # NO enriched
    log.info(f"  v3-no-enriched train {len(rows_tr_ne)} {hist(rows_tr_ne)} (select on holdout)")
    _, scores_ne_ho, be_ne, _, _ = fit(rows_tr_ne, rows_ho, cfg, data_cfg, device)
    rev4_scores_ne = scores_ne_ho[rev4_mask]
    m_rev4_ne = metrics_union(rev4_scores_ne, rev4_rows)
    d_loc = m_rev4["loc_ap_not_bad"] - m_rev4_ne["loc_ap_not_bad"]
    metrics["enrichment_ablation"] = {
        "rev4_eval_n": len(rev4_rows),
        "v3_full": {k: m_rev4[k] for k in ("ap_not_bad", "loc_ap_not_bad",
                                           "loc_p_at_10_not_bad", "ap_good")},
        "v3_no_enriched": {k: m_rev4_ne[k] for k in ("ap_not_bad", "loc_ap_not_bad",
                                                     "loc_p_at_10_not_bad", "ap_good")},
        "delta_loc_ap_not_bad": d_loc, "v3_full_best_epoch": be_u,
        "v3_no_enriched_best_epoch": be_ne,
    }
    log.info(f"  v3-full        rev4 loc AP_notbad {m_rev4['loc_ap_not_bad']:.4f}")
    log.info(f"  v3-no-enriched rev4 loc AP_notbad {m_rev4_ne['loc_ap_not_bad']:.4f}")
    log.info(f"  ENRICHMENT Δ loc AP_notbad (full - no_enriched) = {d_loc:+.4f}  "
             f"-> enrichment {'COMPOUNDS' if d_loc > 0 else 'does NOT improve'} the rev4 ranker")

    # --- persist ---
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")
    log.info(f"artifacts -> {out_dir}")
    log.info("DONE")


if __name__ == "__main__":
    main()
