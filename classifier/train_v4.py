"""Train v4 — the unified location-quality classifier (ordinal 1->3).

v4 is the **unified, Mandelbrot+Julia-ready** model; it just happens to have only
Mandelbrot cache rows today. Nothing here is Mandelbrot-specific — every axis is
read from the cache manifest (incl. `fractal_type`), so the Julia semi-final
retrain points this at an augmented cache and reuses it verbatim.

Locked recipe (see prompts/v4_train_eval.md):
  * INIT fresh from the ImageNet MobileNetV4 backbone (NOT warm-started from v3).
  * CORN ordinal head, 3-class (K-1=2 logits): sigma(logit0)=P(not-bad),
    sigma(logit1)=P(good); deploy score = sigma(logit0)+sigma(logit1) in [0,2].
  * Optimizer / schedule / regularizers INHERITED from v3 (AdamW two-group LRs,
    cosine, drop_rate/drop_path, grad-clip, sqrt class balance). Deltas documented
    in `config.json["recipe_deltas_vs_v3"]`.
  * SELECTION: best checkpoint + early stop on **not-bad AP** (positives label>=2,
    rank by sigma(logit0)) on the **unbiased eval split**. good AP reported, never
    selected on (n=29).
  * Sampler: per-LOCATION weight = w_class x w_group x w_source(beta=BETA_BIASED).

  uv run python -m classifier.train_v4

Outputs -> data/classifier/v4/ (does NOT touch v1/v2/v3).
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
from PIL import Image
from torch.utils.data import DataLoader

from .data import Transform
from .data_v4 import (CANON_SCALE, NEUTRAL_PALETTE, LocationDataset, hist,
                      load_locations, make_weighted_sampler)
from .eval import _ap
from .model import (BACKBONE, build_model, compute_loss, data_config,
                    score_from_logits)
from .train_v2 import detect_device, set_seed

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "classifier" / "v4"
V3_CKPT = ROOT / "data" / "classifier" / "v3" / "model_best.pt"
log = logging.getLogger("train_v4")

# --- top-of-file lever: down-weight of biased (v3-mined) positives (prompt §4). ---
# If good-class AP later collapses, RAISE this toward 1.0 (restores biased-good mass).
BETA_BIASED = 0.4


# --------------------------------------------------------------------------- #
# Scoring: list of Render -> logits (N, 2) through the deterministic deploy xform.
# --------------------------------------------------------------------------- #
class _RenderSet(torch.utils.data.Dataset):
    def __init__(self, renders, transform):
        self.renders = renders; self.transform = transform
    def __len__(self): return len(self.renders)
    def __getitem__(self, i):
        with Image.open(self.renders[i].path) as im:
            im.load(); img = im.convert("RGB")
        return self.transform(img), i


@torch.no_grad()
def score_renders(model, renders, deploy_tf, device, batch_size=64, num_workers=4):
    """Returns logits (N, 2) aligned to `renders` order."""
    model.eval()
    ds = _RenderSet(renders, deploy_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    out = np.zeros((len(renders), 2), dtype=np.float64)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        out[idx.numpy()] = model(x).float().cpu().numpy()
    del loader
    return out


def derive(logits: np.ndarray):
    """logits (N,2) -> (p_notbad, p_good, score_sum)."""
    p = 1.0 / (1.0 + np.exp(-logits))
    return p[:, 0], p[:, 1], p.sum(axis=1)


def ap_block(labels, p_notbad, p_good):
    labels = np.asarray(labels)
    nb = (labels >= 2).astype(int); gd = (labels == 3).astype(int)
    return {
        "ap_not_bad": _ap(nb, p_notbad),   # rank by P(not-bad)=sigma(logit0)
        "ap_good": _ap(gd, p_good),        # rank by P(good)=sigma(logit1)
        "n": int(len(labels)), "n_not_bad": int(nb.sum()), "n_good": int(gd.sum()),
    }


# --------------------------------------------------------------------------- #
def build_v4_model(device, drop_rate, drop_path_rate, pretrained=True):
    return build_model(target="ordinal", drop_rate=drop_rate,
                       drop_path_rate=drop_path_rate, pretrained=pretrained).to(device)


def train(train_locs, eval_renders, eval_labels, cfg, data_cfg, device, sampler,
          out_dir: Path):
    """Frozen-from-v3 loop; selection = max eval not-bad AP (sigma(logit0))."""
    train_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                         data_cfg["std"], train=True,
                         jpeg_q=(85, 95) if not cfg["no_jpeg_aug"] else None)
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)

    train_ds = LocationDataset(train_locs, train_tf, seed=cfg["seed"])
    # cache=False discipline from v2: persistent_workers + committed decode memory
    # tripped Windows ERROR_COMMITMENT_LIMIT; re-decode per epoch is cheap.
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

    for epoch in range(cfg["epochs"]):
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

        # num_workers=0: per-epoch eval re-spawning Windows workers every epoch costs
        # more (spawn tax) than single-thread decoding ~937 small cache JPGs.
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

    last_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt_cfg = dict(cfg)
    ckpt_cfg.update({"backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
                     "interpolation": data_cfg["interpolation"],
                     "input_size": data_cfg["input_size"], "best_epoch": best_epoch})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": ckpt_cfg}, out_dir / "model_best.pt")
    torch.save({"state_dict": last_state, "config": ckpt_cfg}, out_dir / "model_last.pt")

    del train_loader, opt, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_epoch, best_metric, history, ckpt_cfg


# --------------------------------------------------------------------------- #
# Montages
# --------------------------------------------------------------------------- #
def montage(renders, scores, labels, title, path, ncol=4, nrow=4, thumb=(256, 144)):
    from .data import Transform  # noqa
    n = min(ncol * nrow, len(renders))
    if n == 0:
        return
    W, H = thumb
    canvas = Image.new("RGB", (W * ncol, H * nrow), (16, 16, 16))
    for j in range(n):
        with Image.open(renders[j].path) as im:
            im.load(); th = im.convert("RGB").resize((W, H), Image.BICUBIC)
        canvas.paste(th, ((j % ncol) * W, (j // ncol) * H))
    canvas.save(path, quality=92)
    log.info(f"  montage {title}: {path}")


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train v4 unified location-quality classifier.")
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

    # --- load cache, split ---
    locs = load_locations()
    train_locs = [l for l in locs if l.split == "train"]
    eval_locs = [l for l in locs if l.split == "eval"]
    assert all(not l.biased for l in eval_locs), "eval split must be unbiased-only"
    ftypes = Counter(l.fractal_type for l in locs)
    log.info(f"locations: {len(locs)} (train {len(train_locs)} {hist(train_locs)}, "
             f"eval {len(eval_locs)} {hist(eval_locs)})  fractal_type={dict(ftypes)}")

    # --- sampler + DEMANDED effective-mass artifact (beta verification) ---
    sampler, mass_table = make_weighted_sampler(train_locs, beta=args.beta,
                                                class_balance=args.class_balance)
    log.info(f"=== effective sampled mass per (class x source), beta={args.beta} "
             f"class_balance={args.class_balance} ===")
    log.info(f"  class_count={mass_table['class_count']}  w_class={ {k: round(v,4) for k,v in mass_table['w_class'].items()} }")
    for k in mass_table["sampled_mass_fraction"]:
        log.info(f"  {k:22s} n={mass_table['n_locations'][k]:4d}  "
                 f"sampled_mass={mass_table['sampled_mass_fraction'][k]:.4f}  "
                 f"mean/loc={mass_table['mean_mass_per_location'][k]:.2e}")
    smf = mass_table["sampled_mass_fraction"]; mpl = mass_table["mean_mass_per_location"]
    gub, gbi = "label3|unbiased", "label3|biased"
    log.info(f"  GOOD aggregate mass: unbiased {smf.get(gub,0):.4f} vs biased {smf.get(gbi,0):.4f} "
             f"-> {'UNBIASED' if smf.get(gub,0)>smf.get(gbi,0) else 'biased'} dominates aggregate")
    log.info(f"  GOOD per-location mass: unbiased {mpl.get(gub,0):.2e} vs biased {mpl.get(gbi,0):.2e} "
             f"-> source down-weight {'SURVIVED (unbiased>biased per loc)' if mpl.get(gub,0)>mpl.get(gbi,0) else 'FAILED'}")

    # --- config (inherit v3; document deltas) ---
    probe = build_model(target="ordinal", pretrained=True)
    data_cfg = data_config(probe); del probe
    cfg = {
        "target": "ordinal", "geometry": "stretch", "epochs": args.epochs,
        "batch_size": args.batch_size, "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "drop_rate": args.drop_rate, "drop_path_rate": args.drop_path_rate,
        "patience": args.patience, "seed": args.seed, "num_workers": args.num_workers,
        "amp": "off", "grad_clip": 1.0, "no_jpeg_aug": args.no_jpeg_aug,
        "init": "imagenet_backbone_fresh (NOT warm-started from v3)",
        "loss": "CORN ordinal (K-1=2)",
        "sampler": "per-location WeightedRandomSampler(w_class[sqrt] x w_group[1/group] x w_source[beta])",
        "beta_biased": args.beta, "class_balance": args.class_balance,
        "selection": "max eval not-bad AP (rank by sigma(logit0)); early stop patience",
        "eval_split_is_val": True,
        "cache_manifest": "data/v4/cache_manifest.jsonl",
        "train_unit": "base location; __getitem__ draws 1 of 42 cached renders uniformly (epoch-varying)",
        "recipe_deltas_vs_v3": [
            "training unit = base location (uniform-over-42 cached-render draw) vs v3 per-crop rows",
            "sampler gains w_group=1/group_size + w_source=beta (v3 had class balance only)",
            "selection/early-stop val = the unbiased eval split itself (no separate CV/holdout; "
            "the augmentation cache replaces v3's grouped-CV machinery)",
            "augmentation read from cache JPGs (no hue/sat — palette is a cache axis), same v3 Transform knobs",
        ],
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "backbone": BACKBONE, "src_dims": [512, 288], "target_dims": [384, 224],
        "black_thresh": 0.30,
    }
    log.info(f"data_config: {data_cfg}")

    # --- eval renders (deploy-canonical view per eval location) ---
    eval_canon = [l.canonical() for l in eval_locs]
    eval_labels = [l.label for l in eval_locs]

    # --- train ---
    n_ep_est = args.epochs
    log.info(f"=== TRAIN: ~{len(train_locs)} loc/epoch, batch {args.batch_size}, "
             f"<= {n_ep_est} epochs (early-stop patience {args.patience}) ===")
    t_start = time.time()
    best_state, best_epoch, best_val_ap, history, ckpt_cfg = train(
        train_locs, eval_canon, eval_labels, cfg, data_cfg, device, sampler, out_dir)
    log.info(f"=== best epoch {best_epoch}: val not-bad AP {best_val_ap:.4f} "
             f"(train wall {time.time()-t_start:.0f}s) ===")

    # ======================================================================= #
    # EVAL BATTERY (unbiased split only; v3 + v4 on byte-identical cache inputs)
    # ======================================================================= #
    deploy_tf = Transform(cfg["geometry"], data_cfg["interpolation"], data_cfg["mean"],
                          data_cfg["std"], train=False)

    # load best v4 + read-only v3 into fresh models
    v4 = build_v4_model(device, cfg["drop_rate"], cfg["drop_path_rate"], pretrained=False)
    v4.load_state_dict(best_state)
    v3_ck = torch.load(V3_CKPT, map_location="cpu", weights_only=False)
    v3cfg = v3_ck["config"]
    v3 = build_model(target=v3cfg["target"], drop_rate=v3cfg.get("drop_rate", 0.2),
                     drop_path_rate=v3cfg.get("drop_path_rate", 0.1), pretrained=False).to(device)
    v3.load_state_dict(v3_ck["state_dict"])
    v3_tf = Transform(v3cfg["geometry"], v3cfg["interpolation"],
                      tuple(v3cfg["mean"]), tuple(v3cfg["std"]), train=False)

    def score(model, renders, tf):
        return derive(score_renders(model, renders, tf, device,
                                    batch_size=64, num_workers=args.num_workers))

    eval_labels = np.asarray(eval_labels)
    metrics = {"eval_split_n": len(eval_locs), "best_epoch": best_epoch,
               "val_best_not_bad_ap": best_val_ap}

    # --- (1) AP: v3 vs v4 on the canonical views ---
    v4_nb, v4_gd, v4_sum = score(v4, eval_canon, deploy_tf)
    v3_nb, v3_gd, v3_sum = score(v3, eval_canon, v3_tf)
    metrics["ap"] = {"v4": ap_block(eval_labels, v4_nb, v4_gd),
                     "v3": ap_block(eval_labels, v3_nb, v3_gd)}
    log.info("=== (1) AP on deploy-canonical eval views ===")
    for tag, m in metrics["ap"].items():
        log.info(f"  {tag}: not-bad AP {m['ap_not_bad']:.4f}  good AP {m['ap_good']:.4f}  "
                 f"(n={m['n']}, not_bad={m['n_not_bad']}, good={m['n_good']})")

    # --- (2) palette-invariance: per-location score std across 6 ss4 palettes ---
    log.info("=== (2) palette-invariance (per-loc score std across 6 ss4 palettes) ===")
    pal_renders, pal_owner = [], []
    for li, l in enumerate(eval_locs):
        for r in l.palette_renders():
            pal_renders.append(r); pal_owner.append(li)
    pal_owner = np.asarray(pal_owner)
    _, _, v4_pal = score(v4, pal_renders, deploy_tf)
    _, _, v3_pal = score(v3, pal_renders, v3_tf)

    def per_loc_std(scores):
        stds = [scores[pal_owner == li].std() for li in range(len(eval_locs))]
        return np.asarray(stds)
    v4_std, v3_std = per_loc_std(v4_pal), per_loc_std(v3_pal)

    # warmth proxy: regress score on a per-palette warm/cool code (warm=+1, cool=-1, else 0)
    WARMTH = {"cmr.amber": 1.0, "cmr.jungle": 0.0, "twilight_shifted": 0.0,
              "cet_cyclic_mybm_20_100_c48_s25": 0.0, "coolwarm": -1.0,
              "cet_linear_grey_10_95_c0": 0.0}
    warm_vec = np.asarray([WARMTH[r.palette] for r in pal_renders])

    def warmth_trend(scores):
        m = warm_vec != 0.0
        if m.sum() < 2 or np.std(warm_vec[m]) == 0:
            return float("nan")
        return float(np.corrcoef(warm_vec[m], scores[m])[0, 1])
    metrics["palette_invariance"] = {
        "v4_mean_per_loc_std": float(v4_std.mean()), "v3_mean_per_loc_std": float(v3_std.mean()),
        "v4_median_per_loc_std": float(np.median(v4_std)),
        "v3_median_per_loc_std": float(np.median(v3_std)),
        "v4_warmth_corr": warmth_trend(v4_pal), "v3_warmth_corr": warmth_trend(v3_pal),
        "score_range": 2.0,
    }
    log.info(f"  v4 mean per-loc std {v4_std.mean():.4f} (median {np.median(v4_std):.4f})")
    log.info(f"  v3 mean per-loc std {v3_std.mean():.4f} (median {np.median(v3_std):.4f})  "
             f"-> v4 {'MORE' if v4_std.mean()<v3_std.mean() else 'LESS'} palette-invariant")
    log.info(f"  warmth corr (score vs amber/coolwarm): v4 {metrics['palette_invariance']['v4_warmth_corr']:.3f}  "
             f"v3 {metrics['palette_invariance']['v3_warmth_corr']:.3f}")

    # --- (3) AA-invariance: aliased vs ss4 twin (neutral palette, canonical, center) ---
    log.info("=== (3) AA-invariance (aliased vs ss4 twin, neutral palette) ===")
    aa_alias = [l.aa_twin() for l in eval_locs]
    _, _, v4_alias = score(v4, aa_alias, deploy_tf)
    _, _, v3_alias = score(v3, aa_alias, v3_tf)
    v4_aa_d = np.abs(v4_sum - v4_alias); v3_aa_d = np.abs(v3_sum - v3_alias)
    metrics["aa_invariance"] = {
        "v4_mean_abs_diff": float(v4_aa_d.mean()), "v4_p95_abs_diff": float(np.percentile(v4_aa_d, 95)),
        "v3_mean_abs_diff": float(v3_aa_d.mean()), "v3_p95_abs_diff": float(np.percentile(v3_aa_d, 95)),
    }
    log.info(f"  v4 |ss4-aliased| mean {v4_aa_d.mean():.4f} p95 {np.percentile(v4_aa_d,95):.4f}")
    log.info(f"  v3 |ss4-aliased| mean {v3_aa_d.mean():.4f} p95 {np.percentile(v3_aa_d,95):.4f}")

    # --- (4) diagnostics: per-class score, confusion (P(not-bad)@0.5), calibration ---
    log.info("=== (4) diagnostics ===")
    per_class = {int(c): {"n": int((eval_labels == c).sum()),
                          "v4_mean_score": float(v4_sum[eval_labels == c].mean()),
                          "v4_mean_p_notbad": float(v4_nb[eval_labels == c].mean()),
                          "v4_mean_p_good": float(v4_gd[eval_labels == c].mean())}
                 for c in (1, 2, 3)}
    pred_nb = (v4_nb >= 0.5).astype(int); true_nb = (eval_labels >= 2).astype(int)
    confusion_notbad = {"tp": int(((pred_nb == 1) & (true_nb == 1)).sum()),
                        "fp": int(((pred_nb == 1) & (true_nb == 0)).sum()),
                        "tn": int(((pred_nb == 0) & (true_nb == 0)).sum()),
                        "fn": int(((pred_nb == 0) & (true_nb == 1)).sum())}
    bins = np.linspace(0, 1, 11)
    binid = np.clip(np.digitize(v4_nb, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        m = binid == b
        if m.sum():
            calib.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int(m.sum()),
                          "mean_p_notbad": float(v4_nb[m].mean()),
                          "emp_not_bad_rate": float(true_nb[m].mean())})
    metrics["diagnostics"] = {"per_class": per_class, "confusion_notbad_at_0.5": confusion_notbad,
                              "calibration_notbad": calib}
    for c in (1, 2, 3):
        pc = per_class[c]
        log.info(f"  class {c} (n={pc['n']}): v4 mean score {pc['v4_mean_score']:.3f}  "
                 f"P(not-bad) {pc['v4_mean_p_notbad']:.3f}  P(good) {pc['v4_mean_p_good']:.3f}")
    log.info(f"  confusion not-bad@0.5: {confusion_notbad}")

    # --- montages: top / bottom / high-score-but-label-1 ---
    mdir = out_dir / "montages"; mdir.mkdir(exist_ok=True)
    order = np.argsort(-v4_sum)
    montage([eval_canon[i] for i in order[:16]], v4_sum[order[:16]],
            eval_labels[order[:16]], "top16", mdir / "top16.jpg")
    montage([eval_canon[i] for i in order[::-1][:16]], None, None, "bottom16",
            mdir / "bottom16.jpg")
    lab1 = np.where(eval_labels == 1)[0]
    lab1_sorted = lab1[np.argsort(-v4_sum[lab1])]
    montage([eval_canon[i] for i in lab1_sorted[:16]], None, None,
            "high_score_label1_disagreements", mdir / "high_score_label1.jpg")
    metrics["montages"] = {"top16": str(mdir / "top16.jpg"),
                           "bottom16": str(mdir / "bottom16.jpg"),
                           "high_score_label1": str(mdir / "high_score_label1.jpg")}

    # --- (5) freeze the eval battery (byte-identical inputs for the semi-final) ---
    pal_by_loc = {li: {} for li in range(len(eval_locs))}
    for k, (r, li) in enumerate(zip(pal_renders, pal_owner)):
        pal_by_loc[li][r.palette] = {"v4_score": float(v4_pal[k]), "v3_score": float(v3_pal[k])}
    scores_path = out_dir / "eval_scores_v4.jsonl"
    with open(scores_path, "w") as f:
        for li, l in enumerate(eval_locs):
            f.write(json.dumps({
                "location_id": l.location_id, "label": l.label, "source": l.source,
                "biased": l.biased, "group_id": l.group_id, "fractal_type": l.fractal_type,
                "v4_p_not_bad": float(v4_nb[li]), "v4_p_good": float(v4_gd[li]),
                "v4_score": float(v4_sum[li]),
                "v3_p_not_bad": float(v3_nb[li]), "v3_p_good": float(v3_gd[li]),
                "v3_score": float(v3_sum[li]),
                "v4_aa_aliased_score": float(v4_alias[li]),
                "v3_aa_aliased_score": float(v3_alias[li]),
                "v4_palette_std": float(v4_std[li]), "v3_palette_std": float(v3_std[li]),
                "palette_scores": pal_by_loc[li],
            }) + "\n")
    log.info(f"  froze eval battery -> {scores_path}")

    # --- persist ---
    cfg["best_epoch"] = best_epoch
    metrics["mass_table"] = mass_table
    metrics["history"] = history
    metrics["checkpoints"] = {"best": str(out_dir / "model_best.pt"),
                              "last": str(out_dir / "model_last.pt")}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import shutil
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")

    # --- final summary block ---
    log.info("================= V4 SUMMARY =================")
    log.info(f"effective sampled mass (beta={args.beta}): "
             f"good unbiased/biased mean-per-loc "
             f"{mpl.get(gub,0):.2e}/{mpl.get(gbi,0):.2e}")
    log.info(f"best epoch {best_epoch}  val not-bad AP {best_val_ap:.4f}")
    log.info(f"  not-bad AP   v4 {metrics['ap']['v4']['ap_not_bad']:.4f}  vs v3 {metrics['ap']['v3']['ap_not_bad']:.4f}")
    log.info(f"  good AP      v4 {metrics['ap']['v4']['ap_good']:.4f}  vs v3 {metrics['ap']['v3']['ap_good']:.4f}")
    log.info(f"  palette std  v4 {metrics['palette_invariance']['v4_mean_per_loc_std']:.4f}  "
             f"vs v3 {metrics['palette_invariance']['v3_mean_per_loc_std']:.4f}")
    log.info(f"  AA |diff|    v4 {metrics['aa_invariance']['v4_mean_abs_diff']:.4f}  "
             f"vs v3 {metrics['aa_invariance']['v3_mean_abs_diff']:.4f}")
    log.info(f"  checkpoints  {metrics['checkpoints']}")
    log.info(f"  montages     {metrics['montages']}")
    log.info(f"  eval scores  {scores_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
