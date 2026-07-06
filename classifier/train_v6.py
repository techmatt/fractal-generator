"""Train v6 — the multi-family location-quality classifier (ordinal 1->3).

v6 = v5's architecture and training recipe VERBATIM, only the data changes: the
cache now carries the 2026-07-05_gather_v6 batch (9 families: mandelbrot,
multibrot3/4/5, julia, julia_multibrot3/4/5, phoenix). Like train_v5, this module
**imports and reuses v4's `train()` loop and scoring helpers unchanged** — the only
deltas are:
  * data source  -> data/v6/cache_manifest.jsonl (v5 rows reused + gather appended)
  * output dir   -> data/classifier/v6/ (never touches v1..v5)
  * eval compare -> v6 vs **v5** on the frozen eval split, sliced by fractal_type
  * eval battery -> per-family good/not-bad AP, continuous-readout Spearman,
                    AUC(good vs rest), AUC(not-bad vs bad), and the 3-class confusion.

`fractal_type`/`c` are provenance only — never fed to the model (the loader reads
JPGs; fractal_type is used here solely to slice metrics). Pseudo-labels/guard
verdicts never enter training: human labels are the only ground truth.

Default patience == epochs (full schedule): the v5 shakeout found patience-8 early
stop collapsed seed-0 (see mem: v5-unified-julia-classifier); the full schedule is
the fix and is the fixed v6 contract.

  uv run python -m classifier.train_v6

Outputs -> data/classifier/v6/.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import Transform
from .data_v4 import hist, load_locations, make_weighted_sampler
from .eval import _ap
from .model import BACKBONE, build_model, data_config
from .train_v2 import detect_device, set_seed
# Reuse v4's recipe VERBATIM: train loop, model builder, scoring + AP helpers.
from .train_v4 import (BETA_BIASED, ap_block, build_v4_model, derive, montage,
                       score_renders, train)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v6"
V5_CKPT = ROOT / "data" / "classifier" / "v5" / "model_best.pt"
V6_CACHE = ROOT / "data" / "v6" / "cache_manifest.jsonl"
log = logging.getLogger("train_v6")

# Coarse family group for the v5-comparability note: v5 only had deg-2 mandelbrot +
# deg-2 julia as trained capability; every other family is v5-zero-shot.
V5_TRAINED = {"mandelbrot", "julia"}


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


def _pred_class(p_nb, p_gd):
    """CORN rank -> class 1/2/3: rank = #{sigma(logit_k) > 0.5}, class = rank+1."""
    rank = (p_nb > 0.5).astype(int) + (p_gd > 0.5).astype(int)
    return rank + 1


def _confusion(labels, pred):
    m = np.zeros((3, 3), dtype=int)   # rows = true 1/2/3, cols = pred 1/2/3
    for t, p in zip(labels, pred):
        m[int(t) - 1, int(p) - 1] += 1
    return m.tolist()


def family_block(labels, p_nb, p_gd, score, mask):
    """Full per-family metric block over a boolean mask."""
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    nb, gd = (lb >= 2).astype(int), (lb == 3).astype(int)
    pnb, pgd, sc = p_nb[mask], p_gd[mask], score[mask]
    return {
        "n": int(mask.sum()), "n_not_bad": int(nb.sum()), "n_good": int(gd.sum()),
        "ap_not_bad": _ap(nb, pnb), "ap_good": _ap(gd, pgd),
        "auc_not_bad_vs_bad": _auc(nb, pnb), "auc_good_vs_rest": _auc(gd, pgd),
        "spearman": _spearman(lb, sc),
        "confusion_true_x_pred": _confusion(lb, _pred_class(pnb, pgd)),
        "label_hist": {int(k): int(v) for k, v in sorted(Counter(lb.tolist()).items())},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v6 multi-family classifier.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=40)   # == epochs: full schedule
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-jpeg-aug", action="store_true")
    ap.add_argument("--beta", type=float, default=BETA_BIASED)
    ap.add_argument("--class-balance", choices=["sqrt", "inv"], default="sqrt")
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

    # --- load UNIFIED v6 cache, split ---
    locs = load_locations(cache_path=V6_CACHE)
    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    assert all(not l.biased for l in eval_locs), "eval split must be unbiased-only"
    ftypes = Counter(l.fractal_type for l in locs)
    log.info(f"locations: {len(locs)} (train {len(train_locs)} {hist(train_locs)}, "
             f"eval {len(eval_locs)} {hist(eval_locs)})  fractal_type={dict(ftypes)}")
    for ft in sorted(ftypes):
        tr = [l for l in train_locs if l.fractal_type == ft]
        ev = [l for l in eval_locs if l.fractal_type == ft]
        log.info(f"  {ft:18s}: train {len(tr):4d} {hist(tr)}  eval {len(ev):3d} {hist(ev)}")

    # --- sampler (identical recipe to v4/v5) ---
    sampler, mass_table = make_weighted_sampler(train_locs, beta=args.beta,
                                                class_balance=args.class_balance)
    log.info(f"=== sampled mass (beta={args.beta}, class_balance={args.class_balance}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v,4) for k,v in mass_table['w_class'].items()} }")
    for k in mass_table["sampled_mass_fraction"]:
        log.info(f"  {k:22s} n={mass_table['n_locations'][k]:4d}  "
                 f"sampled_mass={mass_table['sampled_mass_fraction'][k]:.4f}")

    # --- config: clone v5/v4 verbatim (only cache path changes) ---
    probe = build_model(target="ordinal", pretrained=True)
    data_cfg = data_config(probe); del probe
    cfg = {
        "target": "ordinal", "geometry": "stretch", "epochs": args.epochs,
        "batch_size": args.batch_size, "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "drop_rate": args.drop_rate, "drop_path_rate": args.drop_path_rate,
        "patience": args.patience, "seed": args.seed, "num_workers": args.num_workers,
        "amp": "off", "grad_clip": 1.0, "no_jpeg_aug": args.no_jpeg_aug,
        "init": "imagenet_backbone_fresh (NOT warm-started)",
        "loss": "CORN ordinal (K-1=2)",
        "sampler": "per-location WeightedRandomSampler(w_class[sqrt] x w_group[1/group] x w_source[beta])",
        "beta_biased": args.beta, "class_balance": args.class_balance,
        "selection": "max eval not-bad AP (rank by sigma(logit0)); patience==epochs (full schedule)",
        "eval_split_is_val": True,
        "cache_manifest": "data/v6/cache_manifest.jsonl",
        "train_unit": "base location; __getitem__ draws 1 of 42 cached renders uniformly (epoch-varying)",
        "recipe_vs_v5": "IDENTICAL recipe (reuses train_v4.train); only data = +gather_v6 (9 families)",
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "backbone": BACKBONE, "src_dims": [512, 288], "target_dims": [384, 224],
        "black_thresh": 0.30,
    }
    log.info(f"data_config: {data_cfg}")

    eval_canon = [l.canonical() for l in eval_locs]
    eval_labels = np.asarray([l.label for l in eval_locs])
    eval_ft = np.asarray([l.fractal_type for l in eval_locs])

    # --- train (verbatim v4 loop) ---
    log.info(f"=== TRAIN: {len(train_locs)} loc/epoch, batch {args.batch_size}, "
             f"<= {args.epochs} epochs (patience {args.patience}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    # ======================================================================= #
    # EVAL BATTERY — v6 vs v5 on the frozen eval set, sliced by fractal_type
    # ======================================================================= #
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)
    v6 = build_v4_model(device, cfg["drop_rate"], cfg["drop_path_rate"], pretrained=False)
    v6.load_state_dict(best_state)
    v5_ck = torch.load(V5_CKPT, map_location="cpu", weights_only=False)
    v5cfg = v5_ck["config"]
    v5 = build_model(target=v5cfg["target"], drop_rate=v5cfg.get("drop_rate", 0.2),
                     drop_path_rate=v5cfg.get("drop_path_rate", 0.1), pretrained=False).to(device)
    v5.load_state_dict(v5_ck["state_dict"])
    v5_tf = Transform(v5cfg["geometry"], v5cfg["interpolation"],
                      tuple(v5cfg["mean"]), tuple(v5cfg["std"]), train=False)

    def score(model, renders, tf):
        return derive(score_renders(model, renders, tf, device, batch_size=64,
                                    num_workers=args.num_workers))

    v6_nb, v6_gd, v6_sum = score(v6, eval_canon, deploy_tf)
    v5_nb, v5_gd, v5_sum = score(v5, eval_canon, v5_tf)

    metrics = {"eval_split_n": len(eval_locs), "best_epoch": best_epoch,
               "val_best_not_bad_ap": best_val_ap,
               "eval_fractal_type": dict(Counter(eval_ft.tolist()))}

    # --- (1) overall + per-family full block, v6 (vs v5 where comparable) ---
    log.info("=== v6 eval battery (deploy-canonical views), per fractal_type ===")
    metrics["families"] = {}
    fam_order = ["__overall__"] + sorted(set(eval_ft.tolist()))
    for fam in fam_order:
        mask = np.ones(len(eval_ft), bool) if fam == "__overall__" else (eval_ft == fam)
        v6b = family_block(eval_labels, v6_nb, v6_gd, v6_sum, mask)
        if v6b is None:
            continue
        v5b = family_block(eval_labels, v5_nb, v5_gd, v5_sum, mask)
        zshot = "" if (fam == "__overall__" or fam in V5_TRAINED) else "  (v5 ZERO-SHOT)"
        metrics["families"][fam] = {"v6": v6b, "v5": v5b, "v5_zero_shot": bool(zshot)}

        def f(x): return "  n/a" if x is None else f"{x:.3f}"
        log.info(f"  [{fam:16s}] n={v6b['n']:4d} (nb={v6b['n_not_bad']}, gd={v6b['n_good']}){zshot}")
        log.info(f"       AP     not-bad v6 {f(v6b['ap_not_bad'])} / v5 {f(v5b['ap_not_bad'])}   "
                 f"good v6 {f(v6b['ap_good'])} / v5 {f(v5b['ap_good'])}")
        log.info(f"       AUC    nb-vs-bad v6 {f(v6b['auc_not_bad_vs_bad'])} / v5 {f(v5b['auc_not_bad_vs_bad'])}   "
                 f"good-vs-rest v6 {f(v6b['auc_good_vs_rest'])} / v5 {f(v5b['auc_good_vs_rest'])}")
        log.info(f"       Spearman v6 {f(v6b['spearman'])} / v5 {f(v5b['spearman'])}   "
                 f"confusion(true x pred) v6 {v6b['confusion_true_x_pred']}")

    # --- (2) montages: overall + new-family top/bottom ---
    mdir = out_dir / "montages"; mdir.mkdir(exist_ok=True)
    order = np.argsort(-v6_sum)
    montage([eval_canon[i] for i in order[:16]], None, None, "top16", mdir / "top16.jpg")
    montage([eval_canon[i] for i in order[::-1][:16]], None, None, "bottom16",
            mdir / "bottom16.jpg")
    newfam = np.where(~np.isin(eval_ft, list(V5_TRAINED)))[0]
    if len(newfam):
        nord = newfam[np.argsort(-v6_sum[newfam])]
        montage([eval_canon[i] for i in nord[:16]], None, None, "newfam_top16",
                mdir / "newfam_top16.jpg")
        montage([eval_canon[i] for i in nord[::-1][:16]], None, None, "newfam_bottom16",
                mdir / "newfam_bottom16.jpg")
    metrics["montages"] = {p.stem: str(p) for p in mdir.glob("*.jpg")}

    # --- freeze eval scores (with fractal_type + v5/v6 columns) ---
    scores_path = out_dir / "eval_scores_v6.jsonl"
    with open(scores_path, "w") as f:
        for li, l in enumerate(eval_locs):
            f.write(json.dumps({
                "location_id": l.location_id, "label": l.label, "source": l.source,
                "group_id": l.group_id, "fractal_type": l.fractal_type,
                "v6_p_not_bad": float(v6_nb[li]), "v6_p_good": float(v6_gd[li]),
                "v6_score": float(v6_sum[li]),
                "v5_p_not_bad": float(v5_nb[li]), "v5_p_good": float(v5_gd[li]),
                "v5_score": float(v5_sum[li]),
            }) + "\n")
    log.info(f"  froze eval battery -> {scores_path}")

    cfg["best_epoch"] = best_epoch
    metrics["mass_table"] = mass_table
    metrics["history"] = history
    metrics["checkpoints"] = {"best": str(out_dir / "model_best.pt"),
                              "last": str(out_dir / "model_last.pt")}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    # --- summary ---
    log.info("================= V6 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    ov = metrics["families"]["__overall__"]
    log.info(f"  overall not-bad AP  v6 {ov['v6']['ap_not_bad']:.4f}  vs v5 {ov['v5']['ap_not_bad']:.4f}"
             f"   good AP  v6 {ov['v6']['ap_good']:.4f}  vs v5 {ov['v5']['ap_good']:.4f}")
    for fam in sorted(metrics["families"]):
        if fam == "__overall__":
            continue
        b = metrics["families"][fam]
        tag = " [v5 zero-shot]" if b["v5_zero_shot"] else ""
        log.info(f"  {fam:16s} not-bad AP v6 {b['v6']['ap_not_bad'] if b['v6']['ap_not_bad'] is None else round(b['v6']['ap_not_bad'],3)}"
                 f"  good AP v6 {b['v6']['ap_good'] if b['v6']['ap_good'] is None else round(b['v6']['ap_good'],3)}{tag}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE")


if __name__ == "__main__":
    main()
