"""Resolution-sensitivity diagnostic for the wallpaper-quality head.

Hypothesis: v2's mushy good-AP (~0.50) vs strong not-bad AP (~0.90) is because the
good<->okay distinction lives in fine structure (filament crispness, banding,
micro-contrast in gradients) that the 384x224 stretch destroys, while
bad<->not-bad (dead zones, flat mass, blown color) survives at any resolution.

Test: retrain the v2 head at a RESOLUTION LADDER, everything else identical to the
v2 union run (same union data, same held-out humanq3 eval split via
``train_wallpaper_v2.split_rows``, same CORN K=4, same geometric-only aug, same
schedule/optimizer). The ONLY variable is the input spatial resolution fed to the
backbone (MobileNetV4 conv_medium global-pools, so variable input just works).

Rungs (v2 aspect 12:7 = 1.714, capped under the 1280x720 native crop):
    384x224  (v2 baseline)  /  640x373  (~1.67x lin)  /  1024x597 (~2.67x lin)

Decisive read:
  * good-AP rises with resolution while not-bad stays flat -> fine-structure
    bottleneck confirmed; the fix is architectural (higher input res, permanent).
  * good-AP flat across resolution -> resolution cleared; look to
    backbone/recipe/objective next.

Report-only. Heads under ``data/wallpaper_head/resdiag/<WxH>/``. No changes to v2
or the batch builders.

    uv run python -m classifier.train_wallpaper_resdiag

This forks the v2 training loop verbatim except for the parameterized resize; the
only new surface is ``ResStretchTransform`` (a resolution-parameterized mirror of
``data.Transform`` for geometry="stretch") and the per-rung driver.
"""
from __future__ import annotations

import gc
import io
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CropDataset, Transform, _to_tensor
from .eval import _ap
from .model import BACKBONE, corn_loss, data_config, score_from_logits
from .train_v2 import detect_device, set_seed
from .train_wallpaper_v2 import (
    K,
    POWERED_FAMILIES,
    build_montage,
    build_wallpaper_model,
    eval_block,
    label_hist,
    load_rows,
    predict_all,
    split_rows,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "wallpaper_head" / "resdiag"

# Resolution ladder — v2 aspect (12:7 = 1.7143), capped under the 1280x720 crop.
# (width, height). Baseline is byte-identical geometry to v2 (384x224 stretch).
RUNGS = [(384, 224), (640, 373), (1024, 597)]

log = logging.getLogger("train_wallpaper_resdiag")


# --------------------------------------------------------------------------- #
# Resolution-parameterized transform. Exact mirror of data.Transform for
# geometry="stretch" (the only geometry the wallpaper head uses), with the target
# (tw, th) carried on the instance so it survives pickling to spawn workers
# (Windows) — data.resize_core's module-global 384x224 would NOT.
#
# Augmentation pipeline is byte-for-byte the v2 pipeline: border_crop -> stretch
# resize -> h/v flip -> [jpeg_q disabled in v2] -> to_tensor -> [bright/contrast
# disabled in v2] -> normalize. Deploy path: stretch resize -> to_tensor ->
# normalize. Nothing here differs from data.Transform except the resize target.
# --------------------------------------------------------------------------- #
class ResStretchTransform(Transform):
    def __init__(self, tw: int, th: int, **kw):
        super().__init__(geometry="stretch", **kw)
        self.tw = tw
        self.th = th

    def _resize(self, img: Image.Image) -> Image.Image:
        from .data import _pil_resample
        return img.resize((self.tw, self.th), _pil_resample(self.interp))

    def __call__(self, img: Image.Image, rng: random.Random | None = None) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.train:
            r = rng or random
            if self.border_crop > 0:
                w, h = img.size
                l = int(round(r.uniform(0, self.border_crop) * w))
                t = int(round(r.uniform(0, self.border_crop) * h))
                rr = int(round(r.uniform(0, self.border_crop) * w))
                b = int(round(r.uniform(0, self.border_crop) * h))
                if l + rr < w - 8 and t + b < h - 8:
                    img = img.crop((l, t, w - rr, h - b))
            img = self._resize(img)
            if r.random() < self.hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if r.random() < self.vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if self.jpeg_q is not None:
                q = r.randint(self.jpeg_q[0], self.jpeg_q[1])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            t = _to_tensor(img)
            if self.brightness > 0:
                t = t * (1.0 + r.uniform(-self.brightness, self.brightness))
            if self.contrast > 0:
                mean_g = t.mean()
                t = (t - mean_g) * (1.0 + r.uniform(-self.contrast, self.contrast)) + mean_g
            t = t.clamp(0, 1)
        else:
            img = self._resize(img)
            t = _to_tensor(img)
        return (t - self.mean) / self.std


def make_loader(rows, transform, batch_size, device, train, num_workers, seed=0):
    # cache=False: 3 models run sequentially in one process; the per-worker
    # decoded-1280x720 cache accumulates committed memory across rungs and trips
    # Windows ERROR_COMMITMENT_LIMIT (mem-1455). Re-decode per epoch is cheap.
    ds = CropDataset(rows, transform, seed=seed, cache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=(device == "cuda"),
                      persistent_workers=(num_workers > 0), drop_last=False)


def _auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def tier2_vs_tier3(labels, marg, ssum):
    """good<->okay separation, the primary probe. Among labels in {2,3} only,
    rank tier-3 by marginal P(>=3). Also the raw score gap 2->3."""
    lb = np.asarray(labels)
    sub = np.isin(lb, [2, 3])
    y = (lb[sub] == 3).astype(int)
    auc = _auc(y, marg[sub, 1]) if sub.sum() >= 2 else None
    ap = _ap(y, marg[sub, 1]) if sub.sum() >= 2 else None
    m2 = float(ssum[lb == 2].mean()) if (lb == 2).any() else None
    m3 = float(ssum[lb == 3].mean()) if (lb == 3).any() else None
    gap = (m3 - m2) if (m2 is not None and m3 is not None) else None
    return {"n_okay": int((lb == 2).sum()), "n_good_tier3": int((lb == 3).sum()),
            "auc_tier3_vs_tier2": auc, "ap_tier3_vs_tier2": ap,
            "mean_score_tier2": m2, "mean_score_tier3": m3, "score_gap_2to3": gap}


# --------------------------------------------------------------------------- #
def train_rung(tw, th, tr, ev, eval_labels, eval_ft, strata, forced,
               n_eval_good, n_eval_exc, thin_good, thin_exc,
               data_cfg, device, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)  # re-seed per rung -> only resolution differs

    train_tf = ResStretchTransform(tw, th, interp=data_cfg["interpolation"],
                                   mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                                   border_crop=args.border_crop, jpeg_q=None,
                                   brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = ResStretchTransform(tw, th, interp=data_cfg["interpolation"],
                                    mean=data_cfg["mean"], std=data_cfg["std"], train=False)

    model = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True).to(device)
    if args.grad_checkpoint:
        # Numerically transparent (deterministic forward recompute) — keeps batch
        # 32 (hence BatchNorm stats) intact at high res on the 8.6GB card, where a
        # plain fp32 forward would spill to WDDM shared RAM. Enabled uniformly so
        # every rung shares one code path; only resolution differs.
        model.set_grad_checkpointing(True)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = make_loader(tr, train_tf, args.batch_size, device, train=True,
                               num_workers=args.num_workers, seed=args.seed)
    eval_loader = make_loader(ev, deploy_tf, args.batch_size, device, train=False,
                              num_workers=min(4, args.num_workers))

    log.info(f"=== RUNG {tw}x{th}: TRAIN {len(tr)} renders/epoch, batch {args.batch_size}, "
             f"{args.epochs} epochs ===")
    best_ap, best_state, best_epoch = -1.0, None, -1
    best_cond, best_marg, best_sum = None, None, None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            ranks = (y - 1).long()
            loss = corn_loss(logits, ranks, num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting rung"); break

        cond, marg, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), marg[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), marg[:, 1])
        ap_ex = _ap((eval_labels >= 4).astype(int), marg[:, 2]) if (eval_labels >= 4).any() else float("nan")
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss, "ap_not_bad": sel,
                        "ap_good": None if not np.isfinite(ap_gd) else float(ap_gd),
                        "ap_exceptional": None if not np.isfinite(ap_ex) else float(ap_ex)})
        log.info(f"  [{tw}x{th}] epoch {epoch:2d}  loss {train_loss:.4f}  AP_nb {sel:.4f}  "
                 f"AP_good {ap_gd:.4f}  AP_exc {ap_ex:.4f}  ({time.time()-t0:.1f}s)")

        # Selection identical to v2: best eval not-bad AP; patience==epochs.
        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"=== [{tw}x{th}] best epoch {best_epoch}: not-bad AP {best_ap:.4f} "
             f"(wall {time.time()-t_start:.0f}s) ===")

    overall = eval_block(eval_labels, best_cond, best_marg, best_sum, np.ones(len(ev), bool))
    t2v3 = tier2_vs_tier3(eval_labels, best_marg, best_sum)
    powered_mask = np.isin(eval_ft, list(POWERED_FAMILIES))
    powered = eval_block(eval_labels, best_cond, best_marg, best_sum, powered_mask)
    powered_t2v3 = tier2_vs_tier3(eval_labels[powered_mask], best_marg[powered_mask],
                                  best_sum[powered_mask]) if powered_mask.sum() else None

    cfg = {"model": "wallpaper_head_resdiag", "target_dims": [tw, th],
           "backbone": BACKBONE, "num_classes": K, "seed": args.seed,
           "epochs": args.epochs, "batch_size": args.batch_size,
           "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
           "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
           "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
           "eval_frac": args.eval_frac, "interpolation": data_cfg["interpolation"],
           "mean": data_cfg["mean"], "std": data_cfg["std"], "src_dims": [1280, 720],
           "geometry": "stretch", "best_epoch": best_epoch,
           "note": "resolution-sensitivity rung; all-else-identical fork of train_wallpaper_v2"}
    torch.save({"state_dict": best_state, "config": cfg}, out_dir / "model_best.pt")

    metrics = {"target_dims": [tw, th], "eval_n": len(ev), "best_epoch": best_epoch,
               "val_best_not_bad_ap": float(best_ap),
               "eval_tier_hist": label_hist(ev),
               "eval_good_count": n_eval_good, "eval_good_low_power": bool(thin_good),
               "eval_exceptional_count": n_eval_exc, "eval_exc_low_power": bool(thin_exc),
               "overall": overall, "tier2_vs_tier3": t2v3,
               "powered_mass": {"families": sorted(POWERED_FAMILIES),
                                "block": powered, "tier2_vs_tier3": powered_t2v3},
               "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    try:
        build_montage(ev, best_sum, out_dir / "eval_montage.png")
    except Exception as e:
        log.warning(f"  montage failed: {e}")

    def f(x): return "  n/a" if x is None else f"{x:.3f}"
    log.info(f"  [{tw}x{th}] AP not-bad {f(overall['ap_not_bad'])}  good {f(overall['ap_good'])}  "
             f"exc {f(overall['ap_exceptional'])}")
    log.info(f"  [{tw}x{th}] tier3-vs-tier2 AUC {f(t2v3['auc_tier3_vs_tier2'])}  "
             f"AP {f(t2v3['ap_tier3_vs_tier2'])}  score-gap 2->3 {f(t2v3['score_gap_2to3'])}  "
             f"(mean s2 {f(t2v3['mean_score_tier2'])} s3 {f(t2v3['mean_score_tier3'])})")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Resolution-sensitivity diagnostic for the wallpaper head.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0,
                    help="TRAINING stochasticity seed (init/order/aug). Vary for confirmation runs.")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="eval-split seed — held FIXED (default 0) so confirmation seeds "
                         "score the SAME held-out eval set; only training stochasticity varies.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
    ap.add_argument("--grad-checkpoint", action="store_true", default=True,
                    help="gradient checkpointing (default on) — fits batch32 at high res")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rungs", default="", help="override, e.g. '384x224,640x373'")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_root / "resdiag.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    # Cross-rung comparability: deterministic cuDNN so the only difference is res.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
             f"cudnn.deterministic=True")

    rungs = RUNGS
    if args.rungs.strip():
        rungs = [tuple(int(v) for v in r.lower().split("x")) for r in args.rungs.split(",")]
    log.info(f"resolution ladder: {rungs}")

    # --- data + split: IDENTICAL to v2 (same seed, same eval_frac) so every rung
    # trains/evals on the exact same rows; only the resize target changes. ---
    rows = load_rows()
    log.info(f"loaded {len(rows)} renders  union_tier_hist={label_hist(rows)}")
    tr, ev, eval_locs, strata, forced = split_rows(rows, args.eval_frac, args.split_seed)
    eh = label_hist(ev)
    n_eval_good, n_eval_exc = eh[3], eh[4]
    thin_good, thin_exc = n_eval_good < 10, n_eval_exc < 10
    log.info(f"  train {len(tr)} / {len({r.loc for r in tr})} loc  {label_hist(tr)}")
    log.info(f"  eval  {len(ev)} / {len(eval_locs)} loc (humanq3)  {label_hist(ev)}")
    if thin_good:
        log.info(f"  ** CAVEAT: eval-good renders = {n_eval_good} (<10) — good-AP LOW-POWER.")
    eval_labels = np.asarray([r.label for r in ev])
    eval_ft = np.asarray([r.family for r in ev])

    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config (mean/std/interp reused across rungs): {data_cfg}")

    all_metrics = {}
    for (tw, th) in rungs:
        m = train_rung(tw, th, tr, ev, eval_labels, eval_ft, strata, forced,
                       n_eval_good, n_eval_exc, thin_good, thin_exc,
                       data_cfg, device, args, out_root / f"{tw}x{th}")
        all_metrics[f"{tw}x{th}"] = m

    # --- ladder summary + decisive read ---
    def g(m, *keys):
        d = m
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d

    summary = {"rungs": [f"{w}x{h}" for (w, h) in rungs],
               "eval_good_count": n_eval_good, "eval_good_low_power": bool(thin_good),
               "eval_exceptional_count": n_eval_exc, "seed": args.seed,
               "per_rung": {}}
    log.info("================= RESOLUTION LADDER SUMMARY =================")
    log.info(f"{'rung':>10} | {'AP_not_bad':>10} | {'AP_good':>8} | {'t3v2_AUC':>9} | "
             f"{'gap_2to3':>8}")
    for key in summary["rungs"]:
        m = all_metrics[key]
        ap_nb = g(m, "overall", "ap_not_bad")
        ap_gd = g(m, "overall", "ap_good")
        auc = g(m, "tier2_vs_tier3", "auc_tier3_vs_tier2")
        gap = g(m, "tier2_vs_tier3", "score_gap_2to3")
        summary["per_rung"][key] = {"ap_not_bad": ap_nb, "ap_good": ap_gd,
                                    "auc_tier3_vs_tier2": auc, "score_gap_2to3": gap,
                                    "ap_powered_good": g(m, "powered_mass", "block", "ap_good")}

        def ff(x): return "     n/a" if x is None else f"{x:8.4f}"
        log.info(f"{key:>10} | {ff(ap_nb)[-10:]:>10} | {ff(ap_gd):>8} | {ff(auc):>9} | {ff(gap):>8}")

    # decisive read
    rk = summary["rungs"]
    nb0, nbN = summary["per_rung"][rk[0]]["ap_not_bad"], summary["per_rung"][rk[-1]]["ap_not_bad"]
    gd0, gdN = summary["per_rung"][rk[0]]["ap_good"], summary["per_rung"][rk[-1]]["ap_good"]
    d_nb = (nbN - nb0) if (nb0 is not None and nbN is not None) else None
    d_gd = (gdN - gd0) if (gd0 is not None and gdN is not None) else None
    verdict = "INCONCLUSIVE"
    if d_gd is not None and d_nb is not None:
        if d_gd > 0.05 and abs(d_nb) < 0.03:
            verdict = "FINE-STRUCTURE BOTTLENECK CONFIRMED (good-AP rises, not-bad flat)"
        elif abs(d_gd) < 0.03:
            verdict = "RESOLUTION CLEARED (good-AP flat) -> look to backbone/recipe/objective"
        else:
            verdict = "MIXED (good-AP and not-bad both move) — read the table"
    summary["deltas"] = {"d_ap_not_bad": d_nb, "d_ap_good": d_gd,
                         "from": rk[0], "to": rk[-1]}
    summary["verdict"] = verdict
    if thin_good:
        summary["power_caveat"] = (f"eval-good renders={n_eval_good} (<10): good-AP is LOW-POWER; "
                                   "treat cross-rung good-AP deltas as directional, confirm with the montage.")
    log.info(f"  delta 384->top: AP_not_bad {d_nb}  AP_good {d_gd}")
    log.info(f"  VERDICT: {verdict}")
    if thin_good:
        log.info(f"  ** {summary['power_caveat']}")
    (out_root / "ladder_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"  wrote {out_root / 'ladder_summary.json'}")
    log.info("DONE")


if __name__ == "__main__":
    main()
