"""Train v5 — the unified Mandelbrot+Julia location-quality classifier (ordinal 1->3).

v5 = v4's architecture and training recipe VERBATIM, only the data changes: the
cache now carries Julia rows (J0 hand labels folded in). To make "verbatim" literal
rather than asserted, this module **imports and reuses v4's `train()` loop and
scoring helpers unchanged** — the only deltas are:
  * data source  -> data/v5/cache_manifest.jsonl (Mandelbrot reused + Julia appended)
  * output dir   -> data/classifier/v5/ (never touches v1..v4)
  * eval compare -> v5 vs **v4** (not v3), split by fractal_type:
       - overall loc AP
       - Mandelbrot-only AP, v4 vs v5  (regression check: did Julia hurt it?)
       - Julia-only AP, v4 vs v5       (new capability + lift over v4's zero-shot)
       - per-class score separation per fractal_type
       - high/low montage incl. Julia

`fractal_type`/`c` are provenance only — never fed to the model (the loader reads
JPGs; fractal_type is used here solely to slice metrics).

  uv run python -m classifier.train_v5

Outputs -> data/classifier/v5/.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import Transform
from .data_v4 import (CANON_SCALE, NEUTRAL_PALETTE, LocationDataset, hist,
                      load_locations, make_weighted_sampler)
from .eval import _ap
from .model import BACKBONE, build_model, data_config
from .train_v2 import detect_device, set_seed
# Reuse v4's recipe VERBATIM: the train loop, model builder, scoring + AP helpers.
from .train_v4 import (BETA_BIASED, ap_block, build_v4_model, derive, montage,
                       score_renders, train)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v5"
V4_CKPT = ROOT / "data" / "classifier" / "v4" / "model_best.pt"
V5_CACHE = ROOT / "data" / "v5" / "cache_manifest.jsonl"
log = logging.getLogger("train_v5")


def ap_split(labels, p_nb, p_gd, ftypes, tag):
    """AP block over a fractal_type subset."""
    m = np.asarray([f == tag for f in ftypes])
    if m.sum() == 0:
        return None
    return ap_block(np.asarray(labels)[m], np.asarray(p_nb)[m], np.asarray(p_gd)[m])


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v5 unified (Mandelbrot+Julia) classifier.")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--drop-path-rate", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
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

    # --- load UNIFIED cache, split ---
    locs = load_locations(cache_path=V5_CACHE)
    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    assert all(not l.biased for l in eval_locs), "eval split must be unbiased-only"
    ftypes = Counter(l.fractal_type for l in locs)
    log.info(f"locations: {len(locs)} (train {len(train_locs)} {hist(train_locs)}, "
             f"eval {len(eval_locs)} {hist(eval_locs)})  fractal_type={dict(ftypes)}")
    for ft in ("mandelbrot", "julia"):
        tr = [l for l in train_locs if l.fractal_type == ft]
        ev = [l for l in eval_locs if l.fractal_type == ft]
        log.info(f"  {ft:10s}: train {len(tr)} {hist(tr)}  eval {len(ev)} {hist(ev)}")

    # --- sampler (per-location w_class x w_group x w_source); identical to v4 ---
    sampler, mass_table = make_weighted_sampler(train_locs, beta=args.beta,
                                                class_balance=args.class_balance)
    log.info(f"=== effective sampled mass (beta={args.beta}, class_balance={args.class_balance}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v,4) for k,v in mass_table['w_class'].items()} }")
    for k in mass_table["sampled_mass_fraction"]:
        log.info(f"  {k:22s} n={mass_table['n_locations'][k]:4d}  "
                 f"sampled_mass={mass_table['sampled_mass_fraction'][k]:.4f}")

    # --- config: clone v4 verbatim (only cache_manifest path changes) ---
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
        "selection": "max eval not-bad AP (rank by sigma(logit0)); early stop patience",
        "eval_split_is_val": True,
        "cache_manifest": "data/v5/cache_manifest.jsonl",
        "train_unit": "base location; __getitem__ draws 1 of 42 cached renders uniformly (epoch-varying)",
        "recipe_vs_v4": "IDENTICAL recipe (reuses train_v4.train); only data = unified Mandelbrot+Julia cache",
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "backbone": BACKBONE, "src_dims": [512, 288], "target_dims": [384, 224],
        "black_thresh": 0.30,
    }
    log.info(f"data_config: {data_cfg}")

    eval_canon = [l.canonical() for l in eval_locs]
    eval_labels = np.asarray([l.label for l in eval_locs])
    eval_ft = [l.fractal_type for l in eval_locs]

    # --- train (verbatim v4 loop) ---
    log.info(f"=== TRAIN: {len(train_locs)} loc/epoch, batch {args.batch_size}, "
             f"<= {args.epochs} epochs (patience {args.patience}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    # ======================================================================= #
    # EVAL BATTERY — v5 vs v4 on the frozen eval set, sliced by fractal_type
    # ======================================================================= #
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)
    v5 = build_v4_model(device, cfg["drop_rate"], cfg["drop_path_rate"], pretrained=False)
    v5.load_state_dict(best_state)
    v4_ck = torch.load(V4_CKPT, map_location="cpu", weights_only=False)
    v4cfg = v4_ck["config"]
    v4 = build_model(target=v4cfg["target"], drop_rate=v4cfg.get("drop_rate", 0.2),
                     drop_path_rate=v4cfg.get("drop_path_rate", 0.1), pretrained=False).to(device)
    v4.load_state_dict(v4_ck["state_dict"])
    v4_tf = Transform(v4cfg["geometry"], v4cfg["interpolation"],
                      tuple(v4cfg["mean"]), tuple(v4cfg["std"]), train=False)

    def score(model, renders, tf):
        return derive(score_renders(model, renders, tf, device, batch_size=64,
                                    num_workers=args.num_workers))

    v5_nb, v5_gd, v5_sum = score(v5, eval_canon, deploy_tf)
    v4_nb, v4_gd, v4_sum = score(v4, eval_canon, v4_tf)

    metrics = {"eval_split_n": len(eval_locs), "best_epoch": best_epoch,
               "val_best_not_bad_ap": best_val_ap,
               "eval_fractal_type": dict(Counter(eval_ft))}

    # --- (1/2/3) AP overall + per fractal_type, v5 vs v4 ---
    log.info("=== AP on deploy-canonical eval views (v5 vs v4) ===")
    metrics["ap"] = {}
    for tag, sl in [("overall", None), ("mandelbrot", "mandelbrot"), ("julia", "julia")]:
        if sl is None:
            v5b = ap_block(eval_labels, v5_nb, v5_gd)
            v4b = ap_block(eval_labels, v4_nb, v4_gd)
        else:
            v5b = ap_split(eval_labels, v5_nb, v5_gd, eval_ft, sl)
            v4b = ap_split(eval_labels, v4_nb, v4_gd, eval_ft, sl)
        metrics["ap"][tag] = {"v5": v5b, "v4": v4b}
        log.info(f"  [{tag:10s}] v5: not-bad {v5b['ap_not_bad']:.4f} good {v5b['ap_good']:.4f}  "
                 f"|| v4: not-bad {v4b['ap_not_bad']:.4f} good {v4b['ap_good']:.4f}  "
                 f"(n={v5b['n']}, nb={v5b['n_not_bad']}, gd={v5b['n_good']})")

    # --- (4) per-class score separation, per fractal_type (monotone 1<2<3?) ---
    log.info("=== per-class score separation (v5 mean score), per fractal_type ===")
    metrics["separation"] = {}
    for ft in ("mandelbrot", "julia"):
        m = np.asarray([f == ft for f in eval_ft])
        block = {}
        for c in (1, 2, 3):
            mc = m & (eval_labels == c)
            block[c] = {"n": int(mc.sum()),
                        "v5_mean_score": float(v5_sum[mc].mean()) if mc.sum() else None,
                        "v5_median_score": float(np.median(v5_sum[mc])) if mc.sum() else None}
        metrics["separation"][ft] = block
        med = [block[c]["v5_median_score"] for c in (1, 2, 3)]
        mono = all(med[i] is not None and med[i + 1] is not None and med[i] < med[i + 1]
                   for i in range(2))
        log.info(f"  {ft:10s} median score 1/2/3 = "
                 f"{med[0] if med[0] is None else round(med[0],3)} / "
                 f"{med[1] if med[1] is None else round(med[1],3)} / "
                 f"{med[2] if med[2] is None else round(med[2],3)}  "
                 f"-> {'MONOTONE' if mono else 'NOT monotone'}")

    # --- (5) montages: overall top/bottom + Julia top/bottom ---
    mdir = out_dir / "montages"; mdir.mkdir(exist_ok=True)
    order = np.argsort(-v5_sum)
    montage([eval_canon[i] for i in order[:16]], None, None, "top16", mdir / "top16.jpg")
    montage([eval_canon[i] for i in order[::-1][:16]], None, None, "bottom16",
            mdir / "bottom16.jpg")
    jmask = np.where(np.asarray([f == "julia" for f in eval_ft]))[0]
    jorder = jmask[np.argsort(-v5_sum[jmask])]
    montage([eval_canon[i] for i in jorder[:16]], None, None, "julia_top16",
            mdir / "julia_top16.jpg")
    montage([eval_canon[i] for i in jorder[::-1][:16]], None, None, "julia_bottom16",
            mdir / "julia_bottom16.jpg")
    metrics["montages"] = {k: str(mdir / f"{k}.jpg")
                           for k in ("top16", "bottom16", "julia_top16", "julia_bottom16")}

    # --- freeze eval scores (with fractal_type + v4/v5 columns) ---
    scores_path = out_dir / "eval_scores_v5.jsonl"
    with open(scores_path, "w") as f:
        for li, l in enumerate(eval_locs):
            f.write(json.dumps({
                "location_id": l.location_id, "label": l.label, "source": l.source,
                "group_id": l.group_id, "fractal_type": l.fractal_type,
                "v5_p_not_bad": float(v5_nb[li]), "v5_p_good": float(v5_gd[li]),
                "v5_score": float(v5_sum[li]),
                "v4_p_not_bad": float(v4_nb[li]), "v4_p_good": float(v4_gd[li]),
                "v4_score": float(v4_sum[li]),
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
    log.info("================= V5 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    for tag in ("overall", "mandelbrot", "julia"):
        a = metrics["ap"][tag]
        log.info(f"  {tag:10s} not-bad AP  v5 {a['v5']['ap_not_bad']:.4f}  vs v4 {a['v4']['ap_not_bad']:.4f}"
                 f"   good AP  v5 {a['v5']['ap_good']:.4f}  vs v4 {a['v4']['ap_good']:.4f}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info(f"  montages    {metrics['montages']}")
    log.info("DONE")


if __name__ == "__main__":
    main()
