"""Train the v1 aesthetic classifier (MobileNetV4, ordinal/binary rank score).

Runs StratifiedGroupKFold CV for trustworthy ranking metrics (mean+-std), then
trains a deliverable model on a fixed grouped 85/15 holdout and emits the visual
results sheet. Save policy: keep the checkpoint that MAXIMIZES held-out AP
not-bad, not min val-loss.

  uv run python -m classifier.train --target ordinal --geometry stretch

Outputs -> data/classifier/v1/ : model_best.pt, model_last.pt, config.json,
metrics.json, results_sheet.html, train.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import eval as ev
from .data import (CropDataset, Transform, class_histogram, load_rows,
                   make_weighted_sampler)
from .model import (BACKBONE, build_model, compute_loss, data_config,
                    score_from_logits)
from .sheet import write_results_sheet

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v1"

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(rows, transform, batch_size, device, train, sampler=None,
                num_workers=4, seed=0):
    ds = CropDataset(rows, transform, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None and train),
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0), drop_last=False,
    )


@torch.no_grad()
def predict(model, loader, n, device):
    """Deterministic fp32 scoring over a prebuilt deploy loader. Scores aligned to
    the loader's dataset order."""
    model.eval()
    scores = np.zeros(n, dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)  # eval in fp32: deterministic + overflow-immune
        s = score_from_logits(logits.float(), predict.target)
        scores[idx.numpy()] = s.cpu().numpy()
    return scores


def evaluate(model, rows, loader, device):
    scores = predict(model, loader, len(rows), device)
    labels = np.array([r.label for r in rows])
    seeds = np.array([r.seed for r in rows])
    m = ev.crop_metrics(scores, labels)
    m.update(ev.location_metrics(scores, labels, seeds))
    return m, scores


# --------------------------------------------------------------------------- #
def train_one(rows_tr, rows_va, cfg, data_cfg, device, save_dir: Path | None = None):
    """Train one model; return (best_metrics, best_scores_on_va, best_epoch,
    history). If save_dir, write model_best.pt / model_last.pt there."""
    target = cfg["target"]
    predict.target = target  # used inside predict()

    train_tf = Transform(cfg["geometry"], data_cfg["interpolation"],
                         data_cfg["mean"], data_cfg["std"], train=True,
                         jpeg_q=(85, 95) if not cfg["no_jpeg_aug"] else None)
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"],
                          data_cfg["mean"], data_cfg["std"], train=False)

    model = build_model(target=target, drop_rate=cfg["drop_rate"],
                        drop_path_rate=cfg["drop_path_rate"], pretrained=True).to(device)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}],
        weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    # AMP off by default: fp16 forward-overflow can corrupt BN running stats, and
    # BN's EMA makes a NaN buffer permanent (kills the model). fp32 is plenty fast
    # for ~1k imgs. --amp fp16 (scaler) / bf16 (no scaler) opt-in.
    amp = cfg.get("amp", "off")
    use_autocast = (device == "cuda") and amp in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    use_scaler = use_autocast and amp == "fp16"
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    sampler = make_weighted_sampler(rows_tr, target)
    train_loader = make_loader(rows_tr, train_tf, cfg["batch_size"], device,
                               train=True, sampler=sampler,
                               num_workers=cfg["num_workers"], seed=cfg["seed"])
    # Built ONCE and reused across epochs (persistent workers + decode cache) — a
    # fresh per-epoch loader would re-spawn workers and re-decode every epoch.
    val_workers = min(4, cfg["num_workers"])
    val_loader = make_loader(rows_va, deploy_tf, cfg["batch_size"], device,
                             train=False, num_workers=val_workers)

    best_metric, best_state, best_epoch = -1.0, None, -1
    best_m, best_scores = None, None
    patience, since_improve = cfg["patience"], 0
    history = []

    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
                logits = model(x)
                loss = compute_loss(logits.float(), y, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(rows_tr)

        # Guard: a corrupted (NaN) model can't recover (BN EMA) — abort the run early.
        if any(not torch.isfinite(p).all() for p in model.parameters()) or \
           any(not torch.isfinite(b.float()).all() for b in model.buffers() if b.is_floating_point()):
            log.error(f"  NaN/Inf in model params/buffers at epoch {epoch} — aborting this model")
            break

        m, scores = evaluate(model, rows_va, val_loader, device)
        sel = m["ap_not_bad"]
        sel = -1.0 if (sel is None or not np.isfinite(sel)) else sel
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": sel, "ap_good": m["ap_good"],
                        "loc_ap_not_bad": m["loc_ap_not_bad"]})
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  "
                 f"AP_notbad {sel:.4f}  AP_good {m['ap_good']:.4f}  "
                 f"loc_AP_notbad {m['loc_ap_not_bad']:.4f}  ({time.time()-t0:.1f}s)")

        if sel > best_metric:
            best_metric, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_m, best_scores = m, scores
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= patience:
                log.info(f"  early stop at epoch {epoch} (best {best_epoch}, AP_notbad {best_metric:.4f})")
                break

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_cfg = dict(cfg)
        ckpt_cfg.update({"backbone": BACKBONE, "mean": data_cfg["mean"],
                         "std": data_cfg["std"], "interpolation": data_cfg["interpolation"],
                         "input_size": data_cfg["input_size"], "best_epoch": best_epoch})
        torch.save({"state_dict": best_state, "config": ckpt_cfg}, save_dir / "model_best.pt")
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": ckpt_cfg}, save_dir / "model_last.pt")

    return best_m, best_scores, best_epoch, history


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train v1 aesthetic classifier.")
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
    ap.add_argument("--folds", type=int, default=5, help="0 to skip CV")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off",
                    help="off (fp32, default, safest) avoids fp16 BN-stat corruption")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-jpeg-aug", action="store_true")
    ap.add_argument("--mixup", action="store_true", help="OFF by default (blends palettes)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "train.log"), logging.StreamHandler(sys.stdout)])

    device = detect_device(args.device)
    set_seed(args.seed)
    if args.mixup:
        log.warning("--mixup requested but intentionally NOT implemented (palette blending). Ignoring.")

    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    rows = load_rows()
    log.info(f"rows after black filter (bf<0.30): {len(rows)}  hist={class_histogram(rows)}")
    n_seeds = len(set(r.seed for r in rows))
    log.info(f"distinct seeds (groups): {n_seeds}")

    cfg = {
        "target": args.target, "geometry": args.geometry, "epochs": args.epochs,
        "batch_size": args.batch_size, "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "drop_rate": args.drop_rate, "drop_path_rate": args.drop_path_rate,
        "patience": args.patience, "seed": args.seed, "num_workers": args.num_workers,
        "amp": args.amp, "grad_clip": args.grad_clip,
        "no_jpeg_aug": args.no_jpeg_aug, "sampler": "WeightedRandomSampler(sqrt-inv-freq)",
        "loss": ("CORN ordinal (K-1=2)" if args.target == "ordinal" else "BCE 1-vs-{2,3}"),
        "folds": args.folds, "holdout_frac": args.holdout_frac,
    }

    # resolve data config from the actual checkpoint (mean/std/interp)
    probe = build_model(target=args.target, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config: {data_cfg}")
    cfg.update({"mean": data_cfg["mean"], "std": data_cfg["std"],
                "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
                "backbone": BACKBONE, "src_dims": [1280, 720], "target_dims": [384, 224],
                "black_thresh": 0.30, "black_gate": "accept iff black_fraction < 0.30 (mirrors present.rs)"})

    metrics = {"cv": None, "holdout": None}

    # --- CV ---
    if args.folds and args.folds >= 2:
        log.info(f"=== StratifiedGroupKFold CV ({args.folds} folds, group=seed) ===")
        fold_crop, fold_loc = [], []
        for fi, (tr_idx, va_idx) in enumerate(ev.make_folds(rows, n_splits=args.folds, seed=args.seed)):
            rows_tr = [rows[i] for i in tr_idx]
            rows_va = [rows[i] for i in va_idx]
            log.info(f"--- fold {fi}: train {len(rows_tr)} ({class_histogram(rows_tr)}) "
                     f"val {len(rows_va)} ({class_histogram(rows_va)}) "
                     f"val_seeds {len(set(r.seed for r in rows_va))} ---")
            m, _, best_ep, _ = train_one(rows_tr, rows_va, cfg, data_cfg, device, save_dir=None)
            log.info(f"--- fold {fi} best epoch {best_ep}: AP_notbad {m['ap_not_bad']:.4f} "
                     f"AP_good {m['ap_good']:.4f} loc_AP_notbad {m['loc_ap_not_bad']:.4f} ---")
            crop_keys = ["ap_not_bad", "ap_good", "p_at_10_not_bad", "p_at_25_not_bad",
                         "p_at_10_good", "p_at_25_good"]
            loc_keys = ["loc_ap_not_bad", "loc_ap_good", "loc_p_at_10_not_bad", "loc_p_at_25_not_bad"]
            fold_crop.append({k: m[k] for k in crop_keys})
            fold_loc.append({k: m[k] for k in loc_keys})
        metrics["cv"] = {"per_fold_crop": fold_crop, "per_fold_loc": fold_loc,
                         "crop": ev.aggregate_folds(fold_crop),
                         "loc": ev.aggregate_folds(fold_loc)}
        log.info(f"CV crop AP_notbad: {metrics['cv']['crop']['ap_not_bad']}")
        log.info(f"CV loc  AP_notbad: {metrics['cv']['loc']['loc_ap_not_bad']}")

    # --- final deliverable: grouped 85/15 holdout ---
    log.info("=== final holdout model (grouped 85/15) ===")
    tr_idx, va_idx = ev.make_holdout(rows, frac=args.holdout_frac, seed=args.seed)
    rows_tr = [rows[i] for i in tr_idx]
    rows_va = [rows[i] for i in va_idx]
    log.info(f"holdout: train {len(rows_tr)} ({class_histogram(rows_tr)}) "
             f"val {len(rows_va)} ({class_histogram(rows_va)}) "
             f"val_seeds {len(set(r.seed for r in rows_va))}")
    m, scores, best_ep, history = train_one(rows_tr, rows_va, cfg, data_cfg, device, save_dir=out_dir)
    cfg["best_epoch"] = best_ep
    metrics["holdout"] = {**m, "best_epoch": best_ep, "history": history}
    log.info(f"HOLDOUT best epoch {best_ep}: AP_notbad {m['ap_not_bad']:.4f} "
             f"AP_good {m['ap_good']:.4f} loc_AP_notbad {m['loc_ap_not_bad']:.4f}")

    # --- persist config + metrics ---
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- results sheet on the holdout val ---
    sheet_path = out_dir / "results_sheet.html"
    write_results_sheet(sheet_path, rows_va, scores, m, metrics, cfg, root=ROOT)
    log.info(f"results sheet -> {sheet_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
