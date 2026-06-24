"""Train the v2 aesthetic classifier — first cross-batch run (loose0_v3 + rev4).

Architecture, augmentation, optimizer, loss, sampler, save policy are HELD FIXED
from v1 (imported from model.py / data.py) so any metric delta is attributable to
the corpus, not the recipe. What's new is cross-batch handling:

  * union data, provenance-BLIND to the model (only crop+label reach the net);
  * batch-qualified fold groups (loose0->seed, rev4->walk) so within-walk
    correlation can't leak train->val;
  * grouped, batch×label-stratified holdout carved first; CV on the remainder;
  * per-batch holdout breakdown + leave-one-batch-out (loose0-only vs union,
    both scored on the SAME rev4 holdout) — the batch-shortcut diagnostic.

  uv run python -m classifier.train_v2

Outputs -> data/classifier/v2/ (does NOT touch v1).
"""
from __future__ import annotations

import gc
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import eval_v2 as ev
from .corpus_data import load_corpus_rows, census
from .data import CropDataset, Transform, make_weighted_sampler
from .model import (BACKBONE, build_model, compute_loss, data_config,
                    score_from_logits)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v2"
log = logging.getLogger("train_v2")

LOOSE0 = "2026-06-23_flat_generate_loose0_v3"
REV4 = "2026-06-24_guided_descend_rev4"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def detect_device(req):
    if req and req != "auto":
        return req
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(rows, transform, batch_size, device, train, sampler=None,
                num_workers=4, seed=0):
    # cache=False: the per-worker decoded-array cache (1280x720 uint8) is held
    # alive by persistent_workers and its committed memory accumulates across the
    # 7 sequential models — on Windows that hits ERROR_COMMITMENT_LIMIT (1455).
    # Re-decoding per epoch is cheap vs. that crash; recipe is unaffected.
    ds = CropDataset(rows, transform, seed=seed, cache=False)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None and train), num_workers=num_workers,
                      pin_memory=(device == "cuda"),
                      persistent_workers=(num_workers > 0), drop_last=False)


@torch.no_grad()
def predict(model, loader, n, device, target):
    """Deterministic fp32 scoring; scores aligned to the loader dataset order."""
    model.eval()
    scores = np.zeros(n, dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        s = score_from_logits(logits.float(), target)
        scores[idx.numpy()] = s.cpu().numpy()
    return scores


def metrics_union(scores, rows):
    return ev.full_metrics(scores, [r.label for r in rows], [r.loc_unit for r in rows])


def build_deploy_loader(rows, data_cfg, cfg, device):
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"],
                          data_cfg["mean"], data_cfg["std"], train=False)
    return make_loader(rows, deploy_tf, cfg["batch_size"], device, train=False,
                       num_workers=min(4, cfg["num_workers"]))


# --------------------------------------------------------------------------- #
def fit(rows_tr, rows_va, cfg, data_cfg, device, save_dir: Path | None = None):
    """Frozen v1 training loop. Selection = max held-out crop AP not-bad.
    Returns (best_metrics, best_scores_on_va, best_epoch, history, best_state)."""
    target = cfg["target"]
    train_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                         data_cfg["std"], train=True,
                         jpeg_q=(85, 95) if not cfg["no_jpeg_aug"] else None)

    model = build_model(target=target, drop_rate=cfg["drop_rate"],
                        drop_path_rate=cfg["drop_path_rate"], pretrained=True).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    sampler = make_weighted_sampler(rows_tr, target)
    train_loader = make_loader(rows_tr, train_tf, cfg["batch_size"], device, train=True,
                               sampler=sampler, num_workers=cfg["num_workers"], seed=cfg["seed"])
    val_loader = build_deploy_loader(rows_va, data_cfg, cfg, device)

    best_metric, best_state, best_epoch = -1.0, None, -1
    best_m, best_scores = None, None
    since_improve = 0
    history = []

    for epoch in range(cfg["epochs"]):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits.float(), y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(rows_tr)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting model"); break

        scores = predict(model, val_loader, len(rows_va), device, target)
        m = metrics_union(scores, rows_va)
        sel = m["ap_not_bad"]
        sel = -1.0 if (sel is None or not np.isfinite(sel)) else sel
        history.append({"epoch": epoch, "train_loss": train_loss, "ap_not_bad": sel,
                        "ap_good": m["ap_good"], "loc_ap_not_bad": m["loc_ap_not_bad"]})
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  AP_notbad {sel:.4f}  "
                 f"AP_good {m['ap_good']:.4f}  loc_AP_notbad {m['loc_ap_not_bad']:.4f}  "
                 f"({time.time()-t0:.1f}s)")

        if sel > best_metric:
            best_metric, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_m, best_scores = m, scores
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg["patience"]:
                log.info(f"  early stop at {epoch} (best {best_epoch}, AP_notbad {best_metric:.4f})")
                break

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cfg = dict(cfg)
        ckpt_cfg.update({"backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
                         "interpolation": data_cfg["interpolation"],
                         "input_size": data_cfg["input_size"], "best_epoch": best_epoch})
        torch.save({"state_dict": best_state, "config": ckpt_cfg}, save_dir / "model_best.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": ckpt_cfg}, save_dir / "model_last.pt")

    # Tear the persistent dataloader workers down NOW so their committed memory is
    # reclaimed before the next model's loaders spawn (Windows commitment limit).
    del train_loader, val_loader, model, opt, sampler
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_m, best_scores, best_epoch, history, best_state


def score_rows(state, rows, cfg, data_cfg, device):
    """Load `state` into a fresh model and deterministically score `rows`."""
    model = build_model(target=cfg["target"], drop_rate=cfg["drop_rate"],
                        drop_path_rate=cfg["drop_path_rate"], pretrained=False).to(device)
    model.load_state_dict(state)
    loader = build_deploy_loader(rows, data_cfg, cfg, device)
    scores = predict(model, loader, len(rows), device, cfg["target"])
    del loader, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return scores


def hist(rows):
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in (1, 2, 3)}


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v2 cross-batch classifier.")
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

    rows = load_corpus_rows()
    cen = census(rows)
    log.info(f"rows after black filter (<0.30): {len(rows)}")
    for k, v in cen.items():
        log.info(f"  {k}: {v}")

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
        "corpus_batches": [LOOSE0, REV4],
        "group_keys": {"loose0": "seed_index", "rev4": "walk_id"},
        "provenance_blind": True,
    }
    probe = build_model(target=args.target, pretrained=True)
    data_cfg = data_config(probe); del probe
    log.info(f"data_config: {data_cfg}")
    cfg.update({"mean": data_cfg["mean"], "std": data_cfg["std"],
                "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
                "backbone": BACKBONE, "src_dims": [1280, 720], "target_dims": [384, 224],
                "black_thresh": 0.30})

    groups, strat = ev.group_strat_arrays(rows)
    metrics = {"cv": None, "holdout": None, "per_batch_holdout": None, "lobo": None}

    # --- carve fixed holdout FIRST (grouped, batch×label-stratified) ---
    tr_full_idx, ho_idx = ev.make_grouped_holdout(groups, strat, frac=args.holdout_frac,
                                                  seed=args.seed)
    rows_ho = [rows[i] for i in ho_idx]
    log.info(f"=== fixed holdout: {len(ho_idx)} crops {hist(rows_ho)} / remainder {len(tr_full_idx)} ===")
    log.info(f"  holdout per-batch: " +
             ", ".join(f"{b}={hist([r for r in rows_ho if r.batch_id==b])}"
                       for b in (LOOSE0, REV4)))

    # --- StratifiedGroupKFold CV on the REMAINDER ---
    if args.folds and args.folds >= 2:
        log.info(f"=== CV ({args.folds} folds, batch-qualified groups) on remainder ===")
        fold_crop, fold_loc, fold_counts = [], [], []
        for fi, (tr_idx, va_idx) in enumerate(
                ev.make_grouped_folds(groups, strat, subset_idx=tr_full_idx,
                                      n_splits=args.folds, seed=args.seed)):
            rows_tr = [rows[i] for i in tr_idx]; rows_va = [rows[i] for i in va_idx]
            vh = hist(rows_va)
            log.info(f"--- fold {fi}: train {len(rows_tr)} {hist(rows_tr)} "
                     f"val {len(rows_va)} {vh} "
                     f"val_groups {len(set(groups[va_idx]))} ---")
            m, _, be, _, _ = fit(rows_tr, rows_va, cfg, data_cfg, device)
            log.info(f"--- fold {fi} best ep {be}: AP_notbad {m['ap_not_bad']:.4f} "
                     f"AP_good {m['ap_good']:.4f} loc_AP_notbad {m['loc_ap_not_bad']:.4f} ---")
            fold_crop.append({k: m[k] for k in
                              ["ap_not_bad", "ap_good", "p_at_10_not_bad", "p_at_25_not_bad",
                               "p_at_10_good", "p_at_25_good"]})
            fold_loc.append({k: m[k] for k in
                             ["loc_ap_not_bad", "loc_ap_good", "loc_p_at_10_not_bad",
                              "loc_p_at_25_not_bad"]})
            fold_counts.append({"val_hist": vh, "n_good": vh[3]})
        metrics["cv"] = {"per_fold_crop": fold_crop, "per_fold_loc": fold_loc,
                         "per_fold_counts": fold_counts,
                         "crop": ev.aggregate_folds(fold_crop),
                         "loc": ev.aggregate_folds(fold_loc)}
        log.info(f"CV crop AP_notbad: {metrics['cv']['crop']['ap_not_bad']}")
        log.info(f"CV loc  AP_notbad: {metrics['cv']['loc']['loc_ap_not_bad']}")

    # --- UNION deliverable model: train on remainder, select on full holdout ---
    log.info("=== UNION holdout model (train=remainder, select=full holdout) ===")
    rows_tr = [rows[i] for i in tr_full_idx]
    m_u, scores_u, be_u, hist_u, state_u = fit(rows_tr, rows_ho, cfg, data_cfg, device,
                                               save_dir=out_dir)
    cfg["best_epoch"] = be_u
    metrics["holdout"] = {**m_u, "best_epoch": be_u, "history": hist_u}
    log.info(f"UNION holdout best ep {be_u}: AP_notbad {m_u['ap_not_bad']:.4f} "
             f"loc_AP_notbad {m_u['loc_ap_not_bad']:.4f}")

    # --- per-batch holdout breakdown (union model) ---
    pb = ev.per_batch_metrics(scores_u, [r.label for r in rows_ho],
                              [r.batch_id for r in rows_ho], [r.loc_unit for r in rows_ho])
    metrics["per_batch_holdout"] = pb
    for b, mm in pb.items():
        log.info(f"  [holdout/{b}] crop AP_notbad {mm['ap_not_bad']:.4f}  "
                 f"loc AP_notbad {mm['loc_ap_not_bad']:.4f}  "
                 f"(n={mm['n']}, not_bad={mm['n_not_bad']}, good={mm['n_good']})")

    # --- LOBO: loose0-only vs union, both scored on the SAME rev4 holdout ---
    log.info("=== LOBO: rank rev4 — loose0-only vs union ===")
    rev4_ho = [r for r in rows_ho if r.batch_id == REV4]
    loose0_rem = [rows[i] for i in tr_full_idx if rows[i].batch_id == LOOSE0]
    loose0_ho = [r for r in rows_ho if r.batch_id == LOOSE0]
    log.info(f"  loose0-only train {len(loose0_rem)} {hist(loose0_rem)} "
             f"select-on loose0-holdout {len(loose0_ho)} {hist(loose0_ho)}")
    _, _, be_l, _, state_l = fit(loose0_rem, loose0_ho, cfg, data_cfg, device)

    def rev4_report(state, tag):
        s = score_rows(state, rev4_ho, cfg, data_cfg, device)
        m = metrics_union(s, rev4_ho)
        log.info(f"  [{tag} -> rev4 holdout] crop AP_notbad {m['ap_not_bad']:.4f}  "
                 f"loc AP_notbad {m['loc_ap_not_bad']:.4f}  P@10 {m['loc_p_at_10_not_bad']}")
        return m

    lobo = {
        "rev4_holdout_n": len(rev4_ho),
        "loose0_only": rev4_report(state_l, "loose0-only"),
        "union": rev4_report(state_u, "union"),
    }
    metrics["lobo"] = lobo
    d_loc = lobo["union"]["loc_ap_not_bad"] - lobo["loose0_only"]["loc_ap_not_bad"]
    log.info(f"  LOBO Δ loc AP_notbad (union - loose0only) on rev4 = {d_loc:+.4f}  "
             f"-> descent labels {'HELP' if d_loc > 0 else 'do NOT help'} rank rev4")

    # --- persist ---
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")
    log.info(f"artifacts -> {out_dir}")
    log.info("DONE")


if __name__ == "__main__":
    main()
