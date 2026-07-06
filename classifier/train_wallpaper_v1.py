"""Train the wallpaper-quality head (v1) — first cross-location, absolute
"is this finished render a good wallpaper" score.

This is a **new forked model** in its own namespace (``data/wallpaper_head/v1/``),
distinct from the location classifier (v5/v6). It reuses the MobileNetV4 + CORN
backbone/loss (``model.py``) and the deterministic deploy resize (``data.py``),
and forks everything else. The differences from the location classifier — each a
silent-wrong-model footgun — are pinned here:

  1. **Label unit = the render.** One label per ``image_id``. NO max-over-crops
     aggregation (the location classifier does that; this does not).
  2. **Taxonomy 4-tier -> CORN K=4 (K-1=3 logits), pinned explicitly.** Max
     observed label is 3 (0 exceptionals), so auto-inference would build a 3-tier
     head and silently break the taxonomy. K=4 regardless.
  3. **0 exceptional labels** -> task2 = P(rank>=3 | rank>=2) = P(>=4|>=3) is
     positive-free. It trains on the 26 goods, all-negative, learns constant-no
     (the CORN conditional loss handles an all-negative subset — it just drives
     that logit negative). Reported as INERT.
  4. **No color augmentation of any kind** — coloring IS the label. Only minor
     geometric aug (border crop + flips). Consequence: no render/palette cache
     step; the existing 1280x720 JPGs are augmented on the fly.
  5. **Input:** stretch to 384x224, matching the checkpoint's mean/std.

Split: location-disjoint (a location's 8 palette renders never straddle
train/eval) AND stratified by location max-tier (goods are concentrated, so this
spreads the good-bearing locations across the split).

    uv run python -m classifier.train_wallpaper_v1

Outputs -> data/wallpaper_head/v1/.
"""
from __future__ import annotations

import gc
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CropDataset, Transform
from .eval import _ap, precision_at_k
from .model import BACKBONE, corn_loss, data_config, score_from_logits
from .train_v2 import detect_device, set_seed

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "data" / "wallpaper_corpus" / "batches" / "2026-07-05_wallpaper_bootstrap_v1"
LABELS = ROOT / "labels" / "wallpaper_bootstrap_v1.json"
OUT_DIR = ROOT / "data" / "wallpaper_head" / "v1"

K = 4  # taxonomy tiers: 1 bad / 2 okay / 3 good / 4 exceptional. PINNED (do not infer).
log = logging.getLogger("train_wallpaper_v1")

_LOC_RE = re.compile(r"(wbv1_\d+)_\d+$")  # image_id = wbv1_<loc>_<pick>


# --------------------------------------------------------------------------- #
# Rows — one per render (image_id). No crop aggregation.
# --------------------------------------------------------------------------- #
@dataclass
class WRow:
    image_id: str
    label: int          # raw tier 1..4 (here only 1..3 observed)
    jpg: Path           # .jpg / .label are the CropDataset contract
    loc: str            # location group key (image_id prefix) — the split unit
    fractal_type: str


def load_rows(batch: Path = BATCH, labels_path: Path = LABELS) -> list[WRow]:
    labels = json.loads(Path(labels_path).read_text())
    rows: list[WRow] = []
    seen: set[str] = set()
    for line in (batch / "images.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = r["image_id"]
        if iid not in labels:
            raise ValueError(f"row {iid} has no label — batch must be fully labeled")
        m = _LOC_RE.match(iid)
        if m is None:
            raise ValueError(f"image_id does not match wbv1_<loc>_<pick>: {iid}")
        jpg = batch / "crops" / f"{iid}.jpg"
        if not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        rows.append(WRow(iid, int(labels[iid]), jpg, m.group(1), r["render"]["fractal_type"]))
        seen.add(iid)
    extra = set(labels) - seen
    if extra:
        raise ValueError(f"{len(extra)} labels have no row: {sorted(extra)[:5]}...")
    return rows


def label_hist(rows) -> dict[int, int]:
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in range(1, K + 1)}


# --------------------------------------------------------------------------- #
# Location-disjoint, max-tier-stratified split.
# --------------------------------------------------------------------------- #
def split_rows(rows: list[WRow], eval_frac: float, seed: int):
    by_loc: dict[str, list[WRow]] = defaultdict(list)
    for r in rows:
        by_loc[r.loc].append(r)
    loc_max = {loc: max(r.label for r in rs) for loc, rs in by_loc.items()}

    rng = np.random.RandomState(seed)
    eval_locs: set[str] = set()
    strata_report = {}
    for tier in sorted(set(loc_max.values())):
        locs = sorted(loc for loc, m in loc_max.items() if m == tier)  # sorted -> deterministic
        rng.shuffle(locs)
        n_eval = int(round(eval_frac * len(locs)))
        eval_locs.update(locs[:n_eval])
        strata_report[tier] = {"n_loc": len(locs), "n_eval_loc": n_eval}

    train = [r for r in rows if r.loc not in eval_locs]
    ev = [r for r in rows if r.loc in eval_locs]
    return train, ev, eval_locs, strata_report


# --------------------------------------------------------------------------- #
# Model / loaders.
# --------------------------------------------------------------------------- #
def build_wallpaper_model(drop_rate, drop_path_rate, pretrained):
    # K-1 = 3 CORN logits, pinned. n_out is NOT inferred from observed labels.
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=K - 1,
                             drop_rate=drop_rate, drop_path_rate=drop_path_rate)


def make_loader(rows, transform, batch_size, device, train, num_workers, seed=0):
    # cache=True: single model, 504 small JPGs — the per-worker decoded cache is
    # cheap and the Windows commitment-limit concern (many sequential models in
    # v2..v6) does not apply to a single train here.
    ds = CropDataset(rows, transform, seed=seed, cache=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=(device == "cuda"),
                      persistent_workers=(num_workers > 0), drop_last=False)


@torch.no_grad()
def predict_all(model, loader, n, device):
    """Deterministic fp32 scoring, aligned to dataset order. Returns per-task
    probs (p[:,0]=P(>=2 not-bad), p[:,1]=P(>=3 good), p[:,2]=P(>=4, inert)) and
    the monotone sum score."""
    model.eval()
    probs = np.zeros((n, K - 1), dtype=np.float64)
    ssum = np.zeros(n, dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).float()
        probs[idx.numpy()] = torch.sigmoid(logits).cpu().numpy()
        ssum[idx.numpy()] = score_from_logits(logits, "ordinal").cpu().numpy()
    return probs, ssum


# --------------------------------------------------------------------------- #
# Eval battery.
# --------------------------------------------------------------------------- #
def _auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def _spearman(labels, score):
    if len(set(labels.tolist())) < 2:
        return None
    rho = spearmanr(score, labels).correlation
    return float(rho) if np.isfinite(rho) else None


def _pred_class(probs):
    """CORN rank = #{sigma(logit_k) > 0.5}; class = rank + 1 (in 1..K)."""
    rank = (probs > 0.5).sum(axis=1)
    return rank + 1


def _confusion(labels, pred):
    m = np.zeros((K, K), dtype=int)  # rows = true 1..K, cols = pred 1..K
    for t, p in zip(labels, pred):
        m[int(t) - 1, int(p) - 1] += 1
    return m.tolist()


def _nan(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def eval_block(labels, probs, ssum, mask):
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    p = probs[mask]
    sc = ssum[mask]
    nb, gd = (lb >= 2).astype(int), (lb >= 3).astype(int)
    p_nb, p_gd = p[:, 0], p[:, 1]
    return {
        "n": int(mask.sum()), "n_not_bad": int(nb.sum()), "n_good": int(gd.sum()),
        "ap_not_bad": _nan(_ap(nb, p_nb)), "ap_good": _nan(_ap(gd, p_gd)),
        "auc_not_bad_vs_bad": _auc(nb, p_nb), "auc_good_vs_rest": _auc(gd, p_gd),
        "p_at_10_not_bad": _nan(precision_at_k(nb, p_nb, 10)),
        "p_at_10_good": _nan(precision_at_k(gd, p_gd, 10)),
        "spearman_score_vs_tier": _spearman(lb, sc),
        "mean_score_by_tier": {int(t): _nan(sc[lb == t].mean()) if (lb == t).any() else None
                               for t in range(1, K + 1)},
        "confusion_true_x_pred": _confusion(lb, _pred_class(p)),
        "label_hist": {int(k): int(v) for k, v in sorted(Counter(lb.tolist()).items())},
    }


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train wallpaper-quality head v1 (CORN K=4).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=40)  # == epochs: full schedule (v5 shakeout)
    ap.add_argument("--eval-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
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

    # --- data ---
    rows = load_rows()
    n_loc = len({r.loc for r in rows})
    log.info(f"loaded {len(rows)} renders over {n_loc} locations  "
             f"tier_hist={label_hist(rows)}  (tier4 exceptional = 0)")

    tr, ev, eval_locs, strata = split_rows(rows, args.eval_frac, args.seed)
    log.info(f"=== split (location-disjoint, max-tier-stratified, eval_frac={args.eval_frac}) ===")
    log.info(f"  train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}")
    log.info(f"  eval  {len(ev)} renders / {len(eval_locs)} loc  {label_hist(ev)}")
    for tier, s in strata.items():
        log.info(f"  loc max-tier {tier}: {s['n_loc']} loc -> {s['n_eval_loc']} eval")
    n_eval_good = label_hist(ev)[3]
    thin = n_eval_good < 10
    if thin:
        log.info(f"  ** CAVEAT: eval-good renders = {n_eval_good} (<10) — good-AP is "
                 f"LOW-POWER; treat the good/exceptional readouts as indicative only.")

    # --- config / transforms ---
    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    log.info(f"data_config: {data_cfg}")

    # Geometric-only aug: border crop + h/v flips. NO color (brightness/contrast=0),
    # NO jpeg-quality jitter (jpeg_q=None) — coloring is the label.
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)

    cfg = {
        "model": "wallpaper_head_v1", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=3, K pinned=4)", "geometry": "stretch",
        "label_unit": "render (image_id) — NO max-over-crops",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "class_weighting": "none (plain, v1)",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "patience": args.patience,
        "border_crop": args.border_crop, "eval_frac": args.eval_frac,
        "seed": args.seed, "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": "max eval not-bad AP (rank by sigma(logit0)); patience==epochs",
        "split": "location-disjoint AND max-tier-stratified",
        "batch_id": "2026-07-05_wallpaper_bootstrap_v1",
        "init": "imagenet_backbone_fresh (NOT warm-started from location classifier)",
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "task2_note": "P(>=4|>=3) positive-free (0 exceptionals) — INERT, forward-compat only",
    }

    # --- model / optimizer ---
    model = probe.to(device)
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
    eval_labels = np.asarray([r.label for r in ev])
    eval_ft = np.asarray([r.fractal_type for r in ev])

    # --- train ---
    log.info(f"=== TRAIN: {len(tr)} renders/epoch, batch {args.batch_size}, "
             f"<= {args.epochs} epochs (patience {args.patience}) ===")
    best_ap, best_state, best_epoch, best_probs, best_sum = -1.0, None, -1, None, None
    since = 0
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            ranks = (y - 1).long()                       # {1,2,3} -> {0,1,2}
            loss = corn_loss(logits, ranks, num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting"); break

        probs, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), probs[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), probs[:, 1])
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": sel, "ap_good": _nan(ap_gd)})
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  AP_notbad {sel:.4f}  "
                 f"AP_good {ap_gd:.4f}  ({time.time()-t0:.1f}s)")

        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs, best_sum = probs, ssum
            since = 0
        else:
            since += 1
            if since >= args.patience:
                log.info(f"  early stop at {epoch} (best {best_epoch}, AP_notbad {best_ap:.4f})")
                break
    log.info(f"=== best epoch {best_epoch}: eval not-bad AP {best_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    # --- save checkpoints ---
    cfg["best_epoch"] = best_epoch
    torch.save({"state_dict": best_state, "config": cfg}, out_dir / "model_best.pt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": cfg}, out_dir / "model_last.pt")

    # ---------------------------------------------------------------------- #
    # EVAL BATTERY on the frozen eval split (best-epoch scores).
    # ---------------------------------------------------------------------- #
    metrics = {"eval_n": len(ev), "best_epoch": best_epoch,
               "val_best_not_bad_ap": _nan(best_ap),
               "eval_tier_hist": label_hist(ev),
               "eval_good_count": n_eval_good, "eval_good_low_power": bool(thin),
               "split_strata": strata,
               "task2_inert": {"positives": 0,
                               "note": "P(>=4|>=3) has zero positives; logit2 -> constant-no. "
                                       "Not a signal; kept for forward-compat when 4s arrive."}}

    overall = eval_block(eval_labels, best_probs, best_sum, np.ones(len(ev), bool))
    metrics["overall"] = overall
    log.info("=== EVAL BATTERY (frozen eval split, best epoch) ===")

    def f(x): return "  n/a" if x is None else f"{x:.3f}"
    log.info(f"  [overall] n={overall['n']} (not_bad={overall['n_not_bad']}, good={overall['n_good']})")
    log.info(f"     AP   not-bad {f(overall['ap_not_bad'])}   good {f(overall['ap_good'])}")
    log.info(f"     AUC  nb-vs-bad {f(overall['auc_not_bad_vs_bad'])}   "
             f"good-vs-rest {f(overall['auc_good_vs_rest'])}")
    log.info(f"     Spearman(score vs tier) {f(overall['spearman_score_vs_tier'])}   "
             f"mean_score_by_tier {overall['mean_score_by_tier']}")
    log.info(f"     confusion(true x pred, 1..4) {overall['confusion_true_x_pred']}")

    # --- per-family good% + block where n permits ---
    metrics["families"] = {}
    for fam in sorted(set(eval_ft.tolist())):
        mask = eval_ft == fam
        blk = eval_block(eval_labels, best_probs, best_sum, mask)
        metrics["families"][fam] = blk
        lb = eval_labels[mask]
        good_pct = 100.0 * (lb >= 3).mean()
        log.info(f"  [{fam:18s}] n={blk['n']:3d}  good%={good_pct:5.1f}  "
                 f"AP_nb {f(blk['ap_not_bad'])}  AP_good {f(blk['ap_good'])}")

    # --- freeze per-render eval scores ---
    scores_path = out_dir / "eval_scores.jsonl"
    with open(scores_path, "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "fractal_type": r.fractal_type,
                "label": r.label,
                "p_not_bad": float(best_probs[i, 0]), "p_good": float(best_probs[i, 1]),
                "p_exceptional": float(best_probs[i, 2]), "score": float(best_sum[i]),
            }) + "\n")
    log.info(f"  froze {len(ev)} eval scores -> {scores_path}")

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics["history"] = history
    metrics["checkpoints"] = {"best": str(out_dir / "model_best.pt"),
                              "last": str(out_dir / "model_last.pt")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    log.info("================= WALLPAPER HEAD v1 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  eval not-bad AP {best_ap:.4f}  "
             f"good AP {f(overall['ap_good'])}  (eval-good n={n_eval_good}"
             f"{' — LOW POWER' if thin else ''})")
    log.info(f"  checkpoints -> {metrics['checkpoints']}")
    log.info(f"  score one render: uv run python -c \""
             f"from classifier.inference import load_scorer; "
             f"print(load_scorer('{out_dir / 'model_best.pt'}').score_paths(['CROP.jpg']))\"")
    log.info("DONE")


if __name__ == "__main__":
    main()
