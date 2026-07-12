"""Train the render-mode quality (mining) head — v1.

The "strange-mode" wallpaper-quality head. Judges a render's mode/palette/color
CHOICE at a fixed location — so the label lives in exactly the thing color/hue
augmentation would destroy. Same load-bearing constraint as head-v3: **geometric
aug only** (border crop + h/v flip), NO brightness/contrast/hue/jpeg jitter.

Forks the head-v2/v3 harness (CORN + eval battery), retuned for K=3 and a
PRE-SPLIT dataset:

  * **Data** = ``data/render_mode_corpus/dataset_v1/{train,eval}.jsonl`` (774 /
    625), already location-disjoint, labels 1/2/3 (bad/okay/good). We do NOT
    re-derive the split — the dataset builder owns it.
  * **Model** = ``mobilenetv4_conv_small`` (a small backbone; this is a mining
    head, run at scale), CORN ordinal K=3 (K-1=2 rank-consistent logits),
    384x224 stretch, the checkpoint's OWN mean/std.
  * **Gate** = marginal ``p_ge`` = cumprod(sigma), NEVER the CORN conditional
    (cond[:,1] is P(>=3|>=2), not P(>=3)).
  * **Multi-seed** (default 5): the split is fixed, seeds differ only in
    init/shuffle/aug. Report per-seed + mean +/- SD (eval q3 is modest, ~87).

Reads (eval, held-out locations):
  - overall q3 good-vs-rest AUC on p_ge3; not-bad AP; good AP.
  - ordinal calibration: Spearman(p_ge3, label) + 3-tier confusion.
  - per-mode q3 AP for the RICH modes (tia / stripe / direct_trap_multiply,
    ~12-14 eval-q3 each); curv_linear / direct_trap_ring / direct_trap_screen
    (2-3 eval-q3) reported as DIRECTIONAL only.

    uv run python -m classifier.train_mining_head --seeds "0 1 2 3 4"

Outputs -> data/render_mode_head/v1/  (per-seed under v1/seed_<s>/).
"""
from __future__ import annotations

import gc
import json
import logging
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
from .eval import _ap
from .model import corn_loss, data_config, score_from_logits
from .train_v2 import detect_device, set_seed

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "render_mode_corpus" / "dataset_v1"
OUT_DIR = ROOT / "data" / "render_mode_head" / "v1"

BACKBONE = "mobilenetv4_conv_small.e2400_r224_in1k"
K = 3  # tiers: 1 bad / 2 okay / 3 good. PINNED.

# Modes with enough eval-q3 mass to make a per-mode good-AP CLAIM.
RICH_MODES = ["tia", "stripe", "direct_trap_multiply"]
# Modes reported per-mode but DIRECTIONAL ONLY (2-3 eval-q3 — a trend, not a claim).
DIRECTIONAL_MODES = ["curv_linear", "direct_trap_ring", "direct_trap_screen"]

log = logging.getLogger("train_mining_head")


# --------------------------------------------------------------------------- #
# Rows — one per render. `.jpg` + `.label` are the CropDataset contract.
# --------------------------------------------------------------------------- #
@dataclass
class MRow:
    image_id: str
    label: int          # raw tier 1..3
    jpg: Path
    loc: str            # location_key — the (pre-decided) split unit
    mode: str
    family: str
    fractal_type: str


def load_split(name: str) -> list[MRow]:
    rows: list[MRow] = []
    for line in (DATASET / f"{name}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        jpg = ROOT / r["crop"]
        if not jpg.exists():
            raise FileNotFoundError(f"crop missing: {jpg}")
        lab = int(r["label"])
        if lab not in (1, 2, 3):
            raise ValueError(f"{r['image_id']}: label {lab} out of 1..3")
        rows.append(MRow(r["image_id"], lab, jpg, r["location_key"],
                         r["mode"], r["family"], r["render"]["fractal_type"]))
    return rows


def label_hist(rows) -> dict[int, int]:
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in range(1, K + 1)}


# --------------------------------------------------------------------------- #
# Model / loaders.
# --------------------------------------------------------------------------- #
def build_mining_model(drop_rate, drop_path_rate, pretrained):
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=K - 1,
                             drop_rate=drop_rate, drop_path_rate=drop_path_rate)


def make_loader(rows, transform, batch_size, device, train, num_workers, seed=0):
    # cache=False: matches train_v2's Windows-commitment-limit fix (per-worker
    # decoded-array caches accumulate across sequential seed models).
    ds = CropDataset(rows, transform, seed=seed, cache=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=(device == "cuda"),
                      persistent_workers=(num_workers > 0), drop_last=False)


@torch.no_grad()
def predict_all(model, loader, n, device):
    """Deterministic fp32 scoring, aligned to dataset order.

    cond[:,0] = sigma(logit0) = P(>=2) (marginal — subset is everyone).
    cond[:,1] = sigma(logit1) = P(>=3 | >=2) (CONDITIONAL).
    marg = cumprod(cond, axis=1) -> P(>=2), P(>=3) (the true marginal gate).
    Returns (cond, marg, ssum) with ssum = Sum sigma(logit_k) in [0, K-1]."""
    model.eval()
    cond = np.zeros((n, K - 1), dtype=np.float64)
    ssum = np.zeros(n, dtype=np.float64)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).float()
        cond[idx.numpy()] = torch.sigmoid(logits).cpu().numpy()
        ssum[idx.numpy()] = score_from_logits(logits, "ordinal").cpu().numpy()
    marg = np.cumprod(cond, axis=1)
    return cond, marg, ssum


# --------------------------------------------------------------------------- #
# Metric helpers.
# --------------------------------------------------------------------------- #
def _nan(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def _spearman(a, b):
    if len(set(np.asarray(a).tolist())) < 2:
        return None
    rho = spearmanr(a, b).correlation
    return float(rho) if np.isfinite(rho) else None


def _pred_class(cond):
    return (cond > 0.5).sum(axis=1) + 1  # {0,1,2} ranks -> {1,2,3}


def _confusion(labels, pred):
    m = np.zeros((K, K), dtype=int)  # rows = true 1..K, cols = pred 1..K
    for t, p in zip(labels, pred):
        m[int(t) - 1, int(p) - 1] += 1
    return m.tolist()


def eval_block(labels, cond, marg, ssum, mask):
    """Overall/slice reads. Headline AP/AUC use MARGINAL probs (marg)."""
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    c, m, sc = cond[mask], marg[mask], ssum[mask]
    nb, gd = (lb >= 2).astype(int), (lb >= 3).astype(int)
    p_nb, p_gd = m[:, 0], m[:, 1]
    return {
        "n": int(mask.sum()), "n_not_bad": int(nb.sum()), "n_good": int(gd.sum()),
        "ap_not_bad": _nan(_ap(nb, p_nb)),
        "ap_good": _nan(_ap(gd, p_gd)),
        "auc_good_vs_rest": _auc(gd, p_gd),
        "auc_not_bad_vs_bad": _auc(nb, p_nb),
        "spearman_pge3_vs_tier": _spearman(p_gd, lb),
        "spearman_score_vs_tier": _spearman(sc, lb),
        "mean_score_by_tier": {int(t): (_nan(sc[lb == t].mean()) if (lb == t).any() else None)
                               for t in range(1, K + 1)},
        "confusion_true_x_pred": _confusion(lb, _pred_class(c)),
        "label_hist": {int(k): int(v) for k, v in sorted(Counter(lb.tolist()).items())},
    }


def per_mode_good_ap(labels, marg, modes, mode_name):
    """q3 (good-vs-rest) AP within one mode, on the marginal p_ge3."""
    mask = modes == mode_name
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    gd = (lb >= 3).astype(int)
    return {"n": int(mask.sum()), "n_good": int(gd.sum()),
            "ap_good": _nan(_ap(gd, marg[mask, 1])),
            "auc_good_vs_rest": _auc(gd, marg[mask, 1])}


# --------------------------------------------------------------------------- #
# One training run (single seed).
# --------------------------------------------------------------------------- #
def train_one_seed(seed, tr, ev, args, device, train_tf, deploy_tf, cfg, seed_dir):
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=True).to(device)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": head_params, "lr": args.head_lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_loader = make_loader(tr, train_tf, args.batch_size, device, train=True,
                               num_workers=args.num_workers, seed=seed)
    eval_loader = make_loader(ev, deploy_tf, args.batch_size, device, train=False,
                              num_workers=min(4, args.num_workers))
    eval_labels = np.asarray([r.label for r in ev])

    best_ap, best_epoch = -1.0, -1
    best_state = best_cond = best_marg = best_sum = None
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            loss = corn_loss(logits, (y - 1).long(), num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"[seed {seed}] NaN/Inf at epoch {epoch} — aborting seed"); break

        cond, marg, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), marg[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), marg[:, 1])
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": sel, "ap_good": _nan(ap_gd)})
        log.info(f"[seed {seed}] epoch {epoch:2d}  loss {train_loss:.4f}  "
                 f"AP_nb {sel:.4f}  AP_good {ap_gd:.4f}  ({time.time()-t0:.1f}s)")
        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"[seed {seed}] best epoch {best_epoch}: not-bad AP {best_ap:.4f} "
             f"(wall {time.time()-t_start:.0f}s)")

    seed_cfg = dict(cfg, seed=seed, best_epoch=best_epoch)
    torch.save({"state_dict": best_state, "config": seed_cfg}, seed_dir / "model_best.pt")
    with open(seed_dir / "eval_scores.jsonl", "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "mode": r.mode, "family": r.family,
                "fractal_type": r.fractal_type, "label": r.label,
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "score": float(best_sum[i]),
            }) + "\n")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, "best_epoch": best_epoch, "val_best_not_bad_ap": _nan(best_ap),
            "history": history, "checkpoint": str(seed_dir / "model_best.pt")}, \
        best_cond, best_marg, best_sum


# --------------------------------------------------------------------------- #
# Cross-seed aggregation.
# --------------------------------------------------------------------------- #
def agg(blocks, key):
    vals = [b[key] for b in blocks if b is not None and b.get(key) is not None]
    if not vals:
        return {"mean": None, "sd": None, "n_seeds": 0, "values": []}
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)),
            "n_seeds": len(vals), "values": [float(v) for v in a]}


def fmt(d):
    return "  n/a" if d["mean"] is None else f"{d['mean']:.3f}+/-{d['sd']:.3f}"


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train render-mode (mining) head v1 (CORN K=3, multi-seed).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--seeds", default="0 1 2 3 4", help="space-separated train seeds (>=3)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--border-crop", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    if len(seeds) < 3:
        raise SystemExit("need >=3 seeds for a measured band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
             f"backbone={BACKBONE}  seeds={seeds}")

    # --- data (pre-split; NOT re-derived) ---
    tr = load_split("train")
    ev = load_split("eval")
    tr_locs = {r.loc for r in tr}
    ev_locs = {r.loc for r in ev}
    span = tr_locs & ev_locs
    if span:
        raise AssertionError(f"{len(span)} locations span train+eval (e.g. {list(span)[:3]})")
    log.info(f"train {len(tr)} renders / {len(tr_locs)} loc  {label_hist(tr)}")
    log.info(f"eval  {len(ev)} renders / {len(ev_locs)} loc  {label_hist(ev)}  "
             f"(location-disjoint OK)")
    eh = label_hist(ev)
    log.info(f"eval q3 (good) = {eh[3]}  [modest — reads reported as mean +/- SD]")
    log.info(f"eval by-mode: {dict(sorted(Counter(r.mode for r in ev).items()))}")

    # --- config / transforms (checkpoint's own mean/std; geometric aug only) ---
    probe = build_mining_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    del probe
    log.info(f"data_config: {data_cfg}")
    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)
    cfg = {
        "model": "render_mode_head_v1", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=2, K pinned=3)", "geometry": "stretch",
        "label_unit": "render (image_id)",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "aug_rationale": "palette/mode/color IS the label (same as head-v3)",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": "max eval not-bad AP (marginal P>=2); full schedule (no early stop)",
        "split": "PRE-SPLIT dataset_v1 (location-disjoint); NOT re-derived",
        "dataset": str(DATASET), "init": "imagenet_backbone_fresh",
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "gate": "marginal p_ge = cumprod(sigma); NEVER the CORN conditional",
        "seeds": seeds,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    eval_labels = np.asarray([r.label for r in ev])
    eval_modes = np.asarray([r.mode for r in ev])

    # --- multi-seed train ---
    per_seed = []
    overall_blocks = []
    mode_blocks = defaultdict(list)   # mode -> [per_mode_good_ap per seed]
    best_for_stage = None             # (not_bad_ap, seed, sum, dir)
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, args, device, train_tf, deploy_tf, cfg, out_dir / f"seed_{seed}")
        per_seed.append(info)
        overall_blocks.append(eval_block(eval_labels, cond, marg, ssum, np.ones(len(ev), bool)))
        for mode in RICH_MODES + DIRECTIONAL_MODES:
            mode_blocks[mode].append(per_mode_good_ap(eval_labels, marg, eval_modes, mode))
        nb = info["val_best_not_bad_ap"] or -1.0
        if best_for_stage is None or nb > best_for_stage[0]:
            best_for_stage = (nb, seed, ssum, out_dir / f"seed_{seed}")
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # --- cross-seed aggregation ---
    log.info("================= CROSS-SEED AGGREGATION =================")
    ov_keys = ["ap_not_bad", "ap_good", "auc_good_vs_rest", "auc_not_bad_vs_bad",
               "spearman_pge3_vs_tier", "spearman_score_vs_tier"]
    b0 = overall_blocks[0]
    overall_agg = {"n": b0["n"], "n_not_bad": b0["n_not_bad"], "n_good": b0["n_good"],
                   **{k: agg(overall_blocks, k) for k in ov_keys}}
    log.info(f"  [OVERALL] n={overall_agg['n']}  not_bad={overall_agg['n_not_bad']}  "
             f"good={overall_agg['n_good']}")
    log.info(f"     good-vs-rest AUC(p_ge3) {fmt(overall_agg['auc_good_vs_rest'])}   "
             f"not-bad AP {fmt(overall_agg['ap_not_bad'])}   good AP {fmt(overall_agg['ap_good'])}")
    log.info(f"     Spearman(p_ge3, tier) {fmt(overall_agg['spearman_pge3_vs_tier'])}   "
             f"Spearman(score, tier) {fmt(overall_agg['spearman_score_vs_tier'])}")

    # mean 3-tier confusion (rounded) for a readable calibration eyeball
    conf_stack = np.array([b["confusion_true_x_pred"] for b in overall_blocks], dtype=float)
    mean_conf = conf_stack.mean(axis=0).round(1).tolist()
    log.info(f"     mean confusion(true x pred, 1..3) {mean_conf}")
    mean_by_tier = {t: float(np.mean([b["mean_score_by_tier"][t] for b in overall_blocks
                                      if b["mean_score_by_tier"][t] is not None]))
                    for t in range(1, K + 1)}
    log.info(f"     mean score by tier {mean_by_tier}")

    # --- per-mode q3 AP ---
    log.info("=== PER-MODE q3 AP (marginal p_ge3) ===")
    mode_agg = {}
    for mode in RICH_MODES + DIRECTIONAL_MODES:
        blks = mode_blocks[mode]
        b0m = next((b for b in blks if b is not None), None)
        tag = "RICH " if mode in RICH_MODES else "DIR* "
        mode_agg[mode] = {
            "tier": ("rich" if mode in RICH_MODES else "directional"),
            "n": (b0m["n"] if b0m else 0), "n_good": (b0m["n_good"] if b0m else 0),
            "ap_good": agg(blks, "ap_good"), "auc_good_vs_rest": agg(blks, "auc_good_vs_rest"),
        }
        m = mode_agg[mode]
        log.info(f"  {tag}[{mode:20s}] n={m['n']:3d} good={m['n_good']:2d}  "
                 f"AP_good {fmt(m['ap_good'])}  AUC {fmt(m['auc_good_vs_rest'])}"
                 f"{'   (directional only)' if mode in DIRECTIONAL_MODES else ''}")

    # --- stage the best-not-bad-AP seed as v1/model_best.pt ---
    import shutil
    stage_nb, stage_seed, s_sum, s_dir = best_for_stage
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")

    # --- verdict one-liner ---
    auc = overall_agg["auc_good_vs_rest"]
    ap_g = overall_agg["ap_good"]
    good_base = overall_agg["n_good"] / max(overall_agg["n"], 1)
    separates = (auc["mean"] is not None and auc["mean"] > 0.60
                 and ap_g["mean"] is not None and ap_g["mean"] > 1.3 * good_base)
    verdict = (
        f"{'SEPARATES' if separates else 'WEAK'}: overall good-vs-rest AUC "
        f"{fmt(auc)} and good AP {fmt(ap_g)} vs base {good_base:.3f}. "
        f"Rich-mode good AP — " +
        "; ".join(f"{m}={fmt(mode_agg[m]['ap_good'])}" for m in RICH_MODES) + ".")

    metrics = {
        "seeds": seeds, "backbone": BACKBONE, "num_classes": K,
        "eval_n": len(ev), "eval_tier_hist": eh, "eval_good_base_rate": float(good_base),
        "overall": overall_agg,
        "mean_confusion_true_x_pred": mean_conf,
        "mean_score_by_tier": mean_by_tier,
        "per_mode_good_ap": mode_agg,
        "per_seed": per_seed,
        "staged": {"seed": stage_seed, "not_bad_ap": float(stage_nb),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": "best per-seed eval not-bad AP"},
        "separates": bool(separates), "verdict": verdict,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("================= RENDER-MODE (MINING) HEAD v1 SUMMARY =================")
    log.info(f"  seeds={seeds}  eval n={len(ev)}  eval-good={overall_agg['n_good']} "
             f"(base {good_base:.3f})")
    log.info(f"  OVERALL  good-vs-rest AUC {fmt(auc)}  not-bad AP {fmt(overall_agg['ap_not_bad'])}  "
             f"good AP {fmt(ap_g)}  Spearman(p_ge3) {fmt(overall_agg['spearman_pge3_vs_tier'])}")
    for m in RICH_MODES:
        log.info(f"  RICH {m:20s} good AP {fmt(mode_agg[m]['ap_good'])} (n_good={mode_agg[m]['n_good']})")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'}  (seed {stage_seed}, not-bad AP {stage_nb:.3f})")
    log.info(f"  VERDICT: {verdict}")
    log.info("DONE")


if __name__ == "__main__":
    main()
