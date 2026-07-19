"""Train v7 — the julia:multibrot retrain of the multi-family location classifier.

v7 = v6's architecture and training recipe VERBATIM, only the data changes: the cache
now carries the 536 post-freeze locations appended onto the byte-frozen v6 prefix (the
125 jm-band + 22 prospect-native + 219 blindspot + 26 loose0_v3 to TRAIN, the 144
julia:mb census to EVAL). Like train_v6/v5, this imports and reuses v4's `train()` loop
and scoring helpers unchanged — the only deltas are:
  * data source  -> data/v7/cache_manifest.jsonl (v6 rows reused + 536 appended)
  * output dir   -> data/classifier/v7/ (never touches v1..v6)
  * eval compare -> v7 vs **v6** on the (frozen v6 + census) eval split, sliced by
                    fractal_type, PLUS the census-144 slice (Option A — the primary
                    julia:mb instrument; see docs/findings/v7_build_gate_stop.md).
  * eval freeze  -> eval_scores_v7.jsonl carries v6_* AND v7_* per-location columns so
                    the paired-DeLong eval (tools/v7/eval_delong.py) re-scores nothing.

`fractal_type`/`c` are provenance only — never fed to the model. Human labels are the
only ground truth. Default patience == epochs (full schedule), the fixed v5/v6 contract.

**ACTIVE_CKPT is NOT switched here and t_good is NOT set** — v6 stays deployed until v7
is measured (build_metadata.deploy_note).

  uv run python -m classifier.train_v7

Outputs -> data/classifier/v7/.
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

import gc

from torch.utils.data import DataLoader

from .data import Transform
from .data_v4 import LocationDataset, hist, load_locations, make_weighted_sampler
from .eval import _ap
from .model import BACKBONE, build_model, compute_loss, data_config
from .train_v2 import detect_device, set_seed
from .train_v4 import BETA_BIASED, build_v4_model, derive, montage, score_renders

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v7"
V6_CKPT = ROOT / "data" / "classifier" / "v6" / "model_best.pt"
V7_CACHE = ROOT / "data" / "v7" / "cache_manifest.jsonl"
log = logging.getLogger("train_v7")

# The census batch source tag (the julia:mb eval instrument; Option A reporting slice).
CENSUS_SOURCE = "prospect_census"


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
    rank = (p_nb > 0.5).astype(int) + (p_gd > 0.5).astype(int)
    return rank + 1


def _confusion(labels, pred):
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, pred):
        m[int(t) - 1, int(p) - 1] += 1
    return m.tolist()


def family_block(labels, p_nb, p_gd, score, mask):
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


def train_resumable(train_locs, eval_renders, eval_labels, cfg, data_cfg, device,
                    sampler, out_dir: Path):
    """train_v4.train() replicated VERBATIM (same recipe, LR schedule, selection =
    max eval not-bad AP) + PER-EPOCH resume checkpointing.

    Motivation: this environment reaps the training process at random times (observed:
    epoch 6, ~0, ~0, 32) and the shared train() only writes checkpoints at the very end,
    so a kill discards the whole run. Here, after every epoch we snapshot
    {model, optimizer, scheduler, epoch, best-tracking, history, RNG} to out_dir/resume.pt;
    a relaunch restores it and continues from the next epoch. Optimizer+scheduler+model
    are restored exactly, so the LR curve and trajectory continue as a valid 40-epoch run
    (the only post-resume difference is the stochastic augmentation/sampling draw, already
    epoch-varying within the recipe). Comparability to v6 is preserved: same data, 40
    epochs, same val-AP selection. A clean uninterrupted run never reads resume.pt."""
    train_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                         data_cfg["std"], train=True,
                         jpeg_q=(85, 95) if not cfg["no_jpeg_aug"] else None)
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)
    train_ds = LocationDataset(train_locs, train_tf, seed=cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                              num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
                              persistent_workers=False, drop_last=False)

    model = build_v4_model(device, cfg["drop_rate"], cfg["drop_path_rate"], pretrained=True)
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg["backbone_lr"]},
         {"params": head_params, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    eval_labels = np.asarray(eval_labels)
    best_metric, best_state, best_epoch, history = -1.0, None, -1, []
    since_improve = 0
    start_epoch = 0

    resume_path = out_dir / "resume.pt"
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best_metric, best_epoch = ck["best_metric"], ck["best_epoch"]
        best_state = ck["best_state"]; since_improve = ck["since_improve"]
        history = ck["history"]; start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        log.info(f"  RESUMED from {resume_path}: continuing at epoch {start_epoch} "
                 f"(best so far {best_metric:.4f} @ epoch {best_epoch})")

    for epoch in range(start_epoch, cfg["epochs"]):
        train_ds.set_epoch(epoch)
        model.train(); t0 = time.time(); running = 0.0; nseen = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = compute_loss(logits.float(), y, "ordinal")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0); nseen += x.size(0)
        sched.step()
        train_loss = running / max(nseen, 1)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting"); break

        logits = score_renders(model, eval_renders, deploy_tf, device,
                               batch_size=cfg["batch_size"], num_workers=0)
        p_nb, p_gd, _ = derive(logits)
        ap_nb = _ap((eval_labels >= 2).astype(int), p_nb)
        ap_gd = _ap((eval_labels == 3).astype(int), p_gd)
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_ap_not_bad": sel, "val_ap_good": ap_gd})
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  "
                 f"val_AP_notbad {sel:.4f}  val_AP_good {ap_gd:.4f}  ({time.time()-t0:.1f}s)")

        if sel > best_metric:
            best_metric, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg["patience"]:
                log.info(f"  early stop at {epoch} (best {best_epoch}, "
                         f"val_AP_notbad {best_metric:.4f})")
                break

        # --- per-epoch resume snapshot (atomic: write tmp, replace) ---
        tmp = out_dir / "resume.pt.tmp"
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "best_metric": best_metric,
            "best_epoch": best_epoch, "best_state": best_state,
            "since_improve": since_improve, "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "numpy_rng": np.random.get_state(),
        }, tmp)
        tmp.replace(resume_path)

    last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt_cfg = dict(cfg)
    ckpt_cfg.update({"backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
                     "interpolation": data_cfg["interpolation"],
                     "input_size": data_cfg["input_size"], "best_epoch": best_epoch})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": ckpt_cfg}, out_dir / "model_best.pt")
    torch.save({"state_dict": last_state, "config": ckpt_cfg}, out_dir / "model_last.pt")
    if resume_path.exists():
        resume_path.unlink()   # completed cleanly — drop the resume snapshot

    del train_loader, opt, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_epoch, best_metric, history, ckpt_cfg


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v7 julia:multibrot retrain.")
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

    # --- load UNIFIED v7 cache, split ---
    locs = load_locations(cache_path=V7_CACHE)
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
    n_census = sum(1 for l in eval_locs if l.source == CENSUS_SOURCE)
    log.info(f"  census slice (source={CENSUS_SOURCE!r}): {n_census} eval locations")

    # --- sampler (identical recipe to v4/v5/v6) ---
    sampler, mass_table = make_weighted_sampler(train_locs, beta=args.beta,
                                                class_balance=args.class_balance)
    log.info(f"=== sampled mass (beta={args.beta}, class_balance={args.class_balance}) ===")
    log.info(f"  class_count={mass_table['class_count']}  "
             f"w_class={ {k: round(v,4) for k,v in mass_table['w_class'].items()} }")

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
        "cache_manifest": "data/v7/cache_manifest.jsonl",
        "train_unit": "base location; __getitem__ draws 1 of 42 cached renders uniformly (epoch-varying)",
        "recipe_vs_v6": "IDENTICAL recipe (reuses train_v4.train); only data = +536 post-freeze appended",
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "backbone": BACKBONE, "src_dims": [512, 288], "target_dims": [384, 224],
        "black_thresh": 0.30,
        "ss2_aug_gap": "ACCEPTED (Amendment 1): no 640x360 ss2 slot; deploy is ss2, aug is "
                       "ss1+ss4 only. Known second-order covariate shift.",
    }
    log.info(f"data_config: {data_cfg}")

    eval_canon = [l.canonical() for l in eval_locs]
    eval_labels = np.asarray([l.label for l in eval_locs])
    eval_ft = np.asarray([l.fractal_type for l in eval_locs])
    eval_src = np.asarray([l.source for l in eval_locs])

    log.info(f"=== TRAIN: {len(train_locs)} loc/epoch, batch {args.batch_size}, "
             f"<= {args.epochs} epochs (patience {args.patience}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train_resumable(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    # ======================================================================= #
    # EVAL BATTERY — v7 vs v6 on the eval set, sliced by fractal_type + census
    # ======================================================================= #
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)
    v7 = build_v4_model(device, cfg["drop_rate"], cfg["drop_path_rate"], pretrained=False)
    v7.load_state_dict(best_state)
    v6_ck = torch.load(V6_CKPT, map_location="cpu", weights_only=False)
    v6cfg = v6_ck["config"]
    v6 = build_model(target=v6cfg["target"], drop_rate=v6cfg.get("drop_rate", 0.2),
                     drop_path_rate=v6cfg.get("drop_path_rate", 0.1), pretrained=False).to(device)
    v6.load_state_dict(v6_ck["state_dict"])
    v6_tf = Transform(v6cfg["geometry"], v6cfg["interpolation"],
                      tuple(v6cfg["mean"]), tuple(v6cfg["std"]), train=False)

    def score(model, renders, tf):
        return derive(score_renders(model, renders, tf, device, batch_size=64,
                                    num_workers=args.num_workers))

    v7_nb, v7_gd, v7_sum = score(v7, eval_canon, deploy_tf)
    v6_nb, v6_gd, v6_sum = score(v6, eval_canon, v6_tf)

    metrics = {"eval_split_n": len(eval_locs), "best_epoch": best_epoch,
               "val_best_not_bad_ap": best_val_ap,
               "eval_fractal_type": dict(Counter(eval_ft.tolist())),
               "census_n": int(n_census)}

    log.info("=== v7 eval battery (deploy-canonical views), per fractal_type ===")
    metrics["families"] = {}
    fam_order = ["__overall__"] + sorted(set(eval_ft.tolist()))
    for fam in fam_order:
        mask = np.ones(len(eval_ft), bool) if fam == "__overall__" else (eval_ft == fam)
        v7b = family_block(eval_labels, v7_nb, v7_gd, v7_sum, mask)
        if v7b is None:
            continue
        v6b = family_block(eval_labels, v6_nb, v6_gd, v6_sum, mask)
        metrics["families"][fam] = {"v7": v7b, "v6": v6b}

        def f(x): return "  n/a" if x is None else f"{x:.3f}"
        log.info(f"  [{fam:16s}] n={v7b['n']:4d} (nb={v7b['n_not_bad']}, gd={v7b['n_good']})")
        log.info(f"       AP     not-bad v7 {f(v7b['ap_not_bad'])} / v6 {f(v6b['ap_not_bad'])}   "
                 f"good v7 {f(v7b['ap_good'])} / v6 {f(v6b['ap_good'])}")
        log.info(f"       AUC    good-vs-rest v7 {f(v7b['auc_good_vs_rest'])} / v6 {f(v6b['auc_good_vs_rest'])}")

    # --- census slice (Option A: the julia:mb instrument) ---
    cmask = (eval_src == CENSUS_SOURCE)
    metrics["census"] = {
        "n": int(cmask.sum()),
        "v7": family_block(eval_labels, v7_nb, v7_gd, v7_sum, cmask),
        "v6": family_block(eval_labels, v6_nb, v6_gd, v6_sum, cmask),
    }
    cb7, cb6 = metrics["census"]["v7"], metrics["census"]["v6"]
    log.info(f"  [CENSUS-144   ] n={cb7['n']} (gd={cb7['n_good']})  "
             f"AUC good-vs-rest v7 {cb7['auc_good_vs_rest']} / v6 {cb6['auc_good_vs_rest']}")
    log.info("  (paired DeLong + CIs: tools/v7/eval_delong.py on eval_scores_v7.jsonl)")

    # --- montages ---
    mdir = out_dir / "montages"; mdir.mkdir(exist_ok=True)
    order = np.argsort(-v7_sum)
    montage([eval_canon[i] for i in order[:16]], None, None, "top16", mdir / "top16.jpg")
    montage([eval_canon[i] for i in order[::-1][:16]], None, None, "bottom16",
            mdir / "bottom16.jpg")
    cidx = np.where(cmask)[0]
    if len(cidx):
        cord = cidx[np.argsort(-v7_sum[cidx])]
        montage([eval_canon[i] for i in cord[:16]], None, None, "census_top16",
                mdir / "census_top16.jpg")
        montage([eval_canon[i] for i in cord[::-1][:16]], None, None, "census_bottom16",
                mdir / "census_bottom16.jpg")
    metrics["montages"] = {p.stem: str(p) for p in mdir.glob("*.jpg")}

    # --- freeze eval scores (v6 + v7 columns, + source for the census slice) ---
    scores_path = out_dir / "eval_scores_v7.jsonl"
    with open(scores_path, "w") as f:
        for li, l in enumerate(eval_locs):
            f.write(json.dumps({
                "location_id": l.location_id, "label": l.label, "source": l.source,
                "group_id": l.group_id, "fractal_type": l.fractal_type,
                "v7_p_not_bad": float(v7_nb[li]), "v7_p_good": float(v7_gd[li]),
                "v7_score": float(v7_sum[li]),
                "v6_p_not_bad": float(v6_nb[li]), "v6_p_good": float(v6_gd[li]),
                "v6_score": float(v6_sum[li]),
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

    log.info("================= V7 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    ov = metrics["families"]["__overall__"]
    log.info(f"  overall not-bad AP  v7 {ov['v7']['ap_not_bad']:.4f}  vs v6 {ov['v6']['ap_not_bad']:.4f}")
    log.info(f"  CENSUS good-vs-rest AUC  v7 {cb7['auc_good_vs_rest']}  vs v6 {cb6['auc_good_vs_rest']}")
    log.info(f"  checkpoints {metrics['checkpoints']}")
    log.info("DONE — ACTIVE_CKPT NOT switched; t_good NOT set. Run tools/v7/eval_delong.py.")


if __name__ == "__main__":
    main()
