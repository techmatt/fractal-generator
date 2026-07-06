"""Train the wallpaper-quality head (v2) — full-union retrain.

Forks ``train_wallpaper_v1`` (all its footguns stay pinned; see that module's
docstring). What v2 changes:

  1. **Data = the UNION of both ss2 batches**, all labeled:
       - bootstrap  (``wbv1_*``): 504 renders / 63 loc  (347/131/26/0)
       - humanq3    (``whq3_*``): 994 renders / 142 loc (224/403/239/128)
     Union tier hist = 571 bad / 534 okay / 265 good / 128 exceptional.
     Each row is tagged with its ``batch`` and its ``family`` (from provenance).

  2. **Split — eval is humanq3-only.** The bootstrap is a location-bad negative
     tail (its "good" renders are still bad *locations*); measuring good-AP /
     tier-3 recall / tier-4 AP on it would score the wrong distribution. So:
       - **Full bootstrap -> train** (all 63 loc / 504 renders).
       - Eval is drawn ONLY from humanq3 held-out locations, eval_frac 0.30,
         stratified by location max-tier (same stratified split as v1).
       - **Forced train-side** (never eligible for eval), honoring
         location-disjointness and family coverage:
           * any humanq3 location whose (cx,cy,fw,type) collides with a
             bootstrap location (keeps a location on ONE side of the split);
           * single-location rare families (would strip the only exemplar of a
             family from train, or land it unpowered in eval).

  3. **Task-2 (P(>=4|>=3)) is now ACTIVE** — 128 exceptionals (all humanq3),
     no longer v1's positive-free constant. Confirmed non-degenerate; tier-4 AP
     and tier-4-vs-3 separation are measurable for the first time.

Everything else is v1: MobileNetV4 + CORN K=4 (pinned via ``num_classes``),
label-unit = render (no max-over-crops), geometric-only aug, no render/palette
cache, 384x224 stretch, imagenet-backbone fresh init, location-disjoint split.

    uv run python -m classifier.train_wallpaper_v2

Outputs -> data/wallpaper_head/v2/.
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
BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"
# (image_id prefix regex, batch dir, label json) per source batch.
SOURCES = [
    ("bootstrap", re.compile(r"(wbv1_\d+)_\d+$"),
     BATCHES / "2026-07-05_wallpaper_bootstrap_v1", ROOT / "labels" / "wallpaper_bootstrap_v1.json"),
    ("humanq3", re.compile(r"(whq3_\d+)_\d+$"),
     BATCHES / "2026-07-05_wallpaper_humanq3_v1", ROOT / "labels" / "wallpaper_humanq3_v1.json"),
]
OUT_DIR = ROOT / "data" / "wallpaper_head" / "v2"

K = 4  # taxonomy tiers: 1 bad / 2 okay / 3 good / 4 exceptional. PINNED (do not infer).
# "Powered" families for the headline report — deg-2 mandelbrot + the julia mass.
# Rare families (multibrot*, phoenix, julia_multibrot*) are coverage-only.
POWERED_FAMILIES = {"mandelbrot", "julia"}
log = logging.getLogger("train_wallpaper_v2")


# --------------------------------------------------------------------------- #
# Rows — one per render (image_id). No crop aggregation.
# --------------------------------------------------------------------------- #
@dataclass
class WRow:
    image_id: str
    label: int          # raw tier 1..4
    jpg: Path           # .jpg / .label are the CropDataset contract
    loc: str            # location group key (image_id prefix) — the split unit
    fractal_type: str
    batch: str          # "bootstrap" | "humanq3"
    family: str         # provenance.family — split + report axis
    coord: tuple        # (cx, cy, fw, fractal_type) — cross-batch collision key


def load_rows() -> list[WRow]:
    rows: list[WRow] = []
    for batch_name, loc_re, batch, labels_path in SOURCES:
        labels = json.loads(labels_path.read_text())
        seen: set[str] = set()
        for line in (batch / "images.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r["image_id"]
            if iid not in labels:
                raise ValueError(f"[{batch_name}] row {iid} has no label — batch must be fully labeled")
            m = loc_re.match(iid)
            if m is None:
                raise ValueError(f"[{batch_name}] image_id does not match expected prefix: {iid}")
            jpg = batch / "crops" / f"{iid}.jpg"
            if not jpg.exists():
                raise FileNotFoundError(f"crop missing: {jpg}")
            rd = r["render"]
            coord = (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"])
            rows.append(WRow(iid, int(labels[iid]), jpg, m.group(1), rd["fractal_type"],
                             batch_name, r["provenance"]["family"], coord))
            seen.add(iid)
        extra = set(labels) - seen
        if extra:
            raise ValueError(f"[{batch_name}] {len(extra)} labels have no row: {sorted(extra)[:5]}...")
    return rows


def label_hist(rows) -> dict[int, int]:
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in range(1, K + 1)}


# --------------------------------------------------------------------------- #
# Split. Bootstrap -> all train. Eval drawn ONLY from eligible humanq3 locations
# (location-disjoint, max-tier-stratified). Forced-train humanq3 locations:
# bootstrap-coord collisions + single-location rare families.
# --------------------------------------------------------------------------- #
def split_rows(rows: list[WRow], eval_frac: float, seed: int):
    boot = [r for r in rows if r.batch == "bootstrap"]
    hq3 = [r for r in rows if r.batch == "humanq3"]

    boot_coords = {r.coord for r in boot}

    by_loc: dict[str, list[WRow]] = defaultdict(list)
    for r in hq3:
        by_loc[r.loc].append(r)
    loc_family = {loc: rs[0].family for loc, rs in by_loc.items()}
    loc_max = {loc: max(r.label for r in rs) for loc, rs in by_loc.items()}

    fam_loc_count = Counter(loc_family.values())
    single_fam = {f for f, n in fam_loc_count.items() if n == 1}

    forced_train: set[str] = set()
    collide_locs, single_fam_locs = [], []
    for loc, rs in by_loc.items():
        if any(r.coord in boot_coords for r in rs):
            forced_train.add(loc); collide_locs.append(loc)
        elif loc_family[loc] in single_fam:
            forced_train.add(loc); single_fam_locs.append(loc)

    # Stratified eval split over the ELIGIBLE humanq3 locations only.
    rng = np.random.RandomState(seed)
    eval_locs: set[str] = set()
    strata_report = {}
    eligible = {loc for loc in by_loc if loc not in forced_train}
    for tier in sorted({loc_max[loc] for loc in eligible}):
        locs = sorted(loc for loc in eligible if loc_max[loc] == tier)  # sorted -> deterministic
        rng.shuffle(locs)
        n_eval = int(round(eval_frac * len(locs)))
        eval_locs.update(locs[:n_eval])
        strata_report[str(tier)] = {"n_loc_eligible": len(locs), "n_eval_loc": n_eval}

    train = [r for r in rows if r.loc not in eval_locs]   # all bootstrap + non-eval humanq3
    ev = [r for r in rows if r.loc in eval_locs]
    forced = {"collide_bootstrap": sorted(collide_locs),
              "single_location_family": sorted(single_fam_locs),
              "single_fam_names": sorted(single_fam)}
    return train, ev, eval_locs, strata_report, forced


# --------------------------------------------------------------------------- #
# Model / loaders.
# --------------------------------------------------------------------------- #
def build_wallpaper_model(drop_rate, drop_path_rate, pretrained):
    return timm.create_model(BACKBONE, pretrained=pretrained, num_classes=K - 1,
                             drop_rate=drop_rate, drop_path_rate=drop_path_rate)


def make_loader(rows, transform, batch_size, device, train, num_workers, seed=0):
    ds = CropDataset(rows, transform, seed=seed, cache=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=(device == "cuda"),
                      persistent_workers=(num_workers > 0), drop_last=False)


@torch.no_grad()
def predict_all(model, loader, n, device):
    """Deterministic fp32 scoring, aligned to dataset order.

    CORN CONDITIONAL vs MARGINAL (the load-bearing subtlety):
    ``corn_loss`` trains each logit_k as a CONDITIONAL P(rank>k | rank>k-1) on
    the rank>=k subset. So the raw sigmoids are:
        cond[:,0] = P(>=2)              (marginal — subset is everyone)
        cond[:,1] = P(>=3 | >=2)        (CONDITIONAL, not marginal P(>=3))
        cond[:,2] = P(>=4 | >=3)        (CONDITIONAL)
    The true MARGINAL gate probs are the cumulative product:
        marg[:,k] = prod_{j<=k} cond[:,j]  ->  P(>=2), P(>=3), P(>=4).
    v1 (and its comment) treated cond[:,k] as marginal — fine as a monotone AP
    RANK for not-bad (k=0 is marginal), but wrong for the >=3 / >=4 gate. We
    return both; the eval headlines the marginal.
    Returns (cond, marg, ssum) where ssum = Σ σ(logit_k) in [0, K-1]."""
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
    rank = (probs > 0.5).sum(axis=1)
    return rank + 1


def _confusion(labels, pred):
    m = np.zeros((K, K), dtype=int)  # rows = true 1..K, cols = pred 1..K
    for t, p in zip(labels, pred):
        m[int(t) - 1, int(p) - 1] += 1
    return m.tolist()


def _nan(x):
    return None if (x is None or not np.isfinite(x)) else float(x)


def _recall_at_gate(y_true, p, thresh=0.5):
    """Recall of the positive class at the P>thresh operating point."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return None
    pred = p > thresh
    return float((pred & (y_true == 1)).sum() / y_true.sum())


def _precision_at_gate(y_true, p, thresh=0.5):
    y_true = np.asarray(y_true)
    fires = p > thresh
    if fires.sum() == 0:
        return None
    return float((fires & (y_true == 1)).sum() / fires.sum())


def eval_block(labels, cond, marg, ssum, mask):
    """Headline AP/recall use the MARGINAL probs (marg); the *_cond twins use the
    raw conditional sigmoids for v1-parity. cond[:,0]==marg[:,0] (not-bad)."""
    if mask.sum() == 0:
        return None
    lb = np.asarray(labels)[mask]
    c, m = cond[mask], marg[mask]
    sc = ssum[mask]
    nb, gd, exc = (lb >= 2).astype(int), (lb >= 3).astype(int), (lb >= 4).astype(int)
    p_nb = m[:, 0]                       # marginal P(>=2) (== cond[:,0])
    mg, me = m[:, 1], m[:, 2]            # marginal P(>=3), P(>=4)
    cg, ce = c[:, 1], c[:, 2]            # conditional P(>=3|>=2), P(>=4|>=3)
    # tier-4 vs tier-3 separation: among {3,4} only, rank 4s by P(>=4|>=3).
    top = lb >= 3
    if top.sum() >= 2 and (lb[top] == 4).any() and (lb[top] == 3).any():
        ap_4v3 = _nan(_ap((lb[top] == 4).astype(int), ce[top]))
        auc_4v3 = _auc((lb[top] == 4).astype(int), ce[top])
    else:
        ap_4v3 = auc_4v3 = None
    return {
        "n": int(mask.sum()), "n_not_bad": int(nb.sum()), "n_good": int(gd.sum()),
        "n_exceptional": int(exc.sum()),
        # headline = marginal
        "ap_not_bad": _nan(_ap(nb, p_nb)),
        "ap_good": _nan(_ap(gd, mg)), "ap_good_cond": _nan(_ap(gd, cg)),
        "ap_exceptional": (_nan(_ap(exc, me)) if exc.sum() else None),
        "ap_exceptional_cond": (_nan(_ap(exc, ce)) if exc.sum() else None),
        "ap_4_vs_3": ap_4v3, "auc_4_vs_3": auc_4v3,
        "auc_not_bad_vs_bad": _auc(nb, p_nb), "auc_good_vs_rest": _auc(gd, mg),
        # tier-3 gate = marginal P(>=3) > 0.5 (the real downstream gate)
        "recall_good_at_gate": _recall_at_gate(gd, mg),
        "precision_good_at_gate": _precision_at_gate(gd, mg),
        "gate_fires_frac": float((mg > 0.5).mean()),
        "recall_good_at_gate_cond": _recall_at_gate(gd, cg),   # v1-style (inflated: conditional)
        "recall_not_bad_at_gate": _recall_at_gate(nb, p_nb),
        "p_at_10_not_bad": _nan(precision_at_k(nb, p_nb, 10)),
        "p_at_10_good": _nan(precision_at_k(gd, mg, 10)),
        "spearman_score_vs_tier": _spearman(lb, sc),
        "mean_score_by_tier": {int(t): _nan(sc[lb == t].mean()) if (lb == t).any() else None
                               for t in range(1, K + 1)},
        "confusion_true_x_pred": _confusion(lb, _pred_class(c)),
        "label_hist": {int(k): int(v) for k, v in sorted(Counter(lb.tolist()).items())},
    }


# --------------------------------------------------------------------------- #
# Montage — per-tier rows, renders sorted by continuous readout (monotonicity
# eyeball). Small: up to 8 columns per tier row.
# --------------------------------------------------------------------------- #
def build_montage(ev, ssum, out_path, cols=8, thumb=(256, 144)):
    from PIL import Image, ImageDraw
    by_tier: dict[int, list[tuple[float, WRow]]] = defaultdict(list)
    for r, s in zip(ev, ssum):
        by_tier[r.label].append((float(s), r))
    tiers = sorted(by_tier)
    pad, header = 4, 18
    tw, th = thumb
    W = pad + cols * (tw + pad)
    H = pad + len(tiers) * (th + header + pad)
    canvas = Image.new("RGB", (W, H), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    tier_name = {1: "bad", 2: "okay", 3: "good", 4: "excep"}
    for row, tier in enumerate(tiers):
        items = sorted(by_tier[tier], key=lambda t: -t[0])  # high score first
        # evenly sample `cols` across the sorted list so we see the spread
        if len(items) > cols:
            idx = np.linspace(0, len(items) - 1, cols).round().astype(int)
            items = [items[i] for i in idx]
        y0 = pad + row * (th + header + pad)
        draw.text((pad, y0), f"tier {tier} ({tier_name[tier]})  n={len(by_tier[tier])}",
                  fill=(230, 230, 230))
        for col, (s, r) in enumerate(items):
            x0 = pad + col * (tw + pad)
            y1 = y0 + header
            try:
                with Image.open(r.jpg) as im:
                    canvas.paste(im.convert("RGB").resize((tw, th)), (x0, y1))
            except Exception:
                continue
            draw.text((x0 + 2, y1 + 2), f"{s:.2f}", fill=(255, 255, 0))
    canvas.save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train wallpaper-quality head v2 (full union, CORN K=4).")
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
    by_batch = Counter(r.batch for r in rows)
    log.info(f"loaded {len(rows)} renders  by_batch={dict(by_batch)}  "
             f"union_tier_hist={label_hist(rows)}")
    for b in ("bootstrap", "humanq3"):
        sub = [r for r in rows if r.batch == b]
        log.info(f"  {b:9s}: {len(sub)} renders / {len({r.loc for r in sub})} loc  {label_hist(sub)}")

    tr, ev, eval_locs, strata, forced = split_rows(rows, args.eval_frac, args.seed)
    log.info(f"=== split (bootstrap->train; eval humanq3-only, location-disjoint, "
             f"max-tier-stratified, eval_frac={args.eval_frac}) ===")
    log.info(f"  train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}")
    log.info(f"  eval  {len(ev)} renders / {len(eval_locs)} loc (humanq3)  {label_hist(ev)}")
    log.info(f"  forced train-side: {len(forced['collide_bootstrap'])} bootstrap-coord collisions "
             f"{forced['collide_bootstrap']}, "
             f"{len(forced['single_location_family'])} single-loc families "
             f"{forced['single_location_family']} (families {forced['single_fam_names']})")
    for tier, s in strata.items():
        log.info(f"  loc max-tier {tier}: {s['n_loc_eligible']} eligible loc -> {s['n_eval_loc']} eval")
    eh = label_hist(ev)
    n_eval_good, n_eval_exc = eh[3], eh[4]
    thin_good = n_eval_good < 10
    thin_exc = n_eval_exc < 10
    if thin_good:
        log.info(f"  ** CAVEAT: eval-good renders = {n_eval_good} (<10) — good-AP LOW-POWER.")
    if thin_exc:
        log.info(f"  ** CAVEAT: eval-exceptional renders = {n_eval_exc} (<10) — tier-4 AP LOW-POWER.")

    # --- config / transforms ---
    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
    data_cfg = data_config(probe)
    log.info(f"data_config: {data_cfg}")

    train_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                         mean=data_cfg["mean"], std=data_cfg["std"], train=True,
                         border_crop=args.border_crop, jpeg_q=None,
                         brightness=0.0, contrast=0.0, hflip=0.5, vflip=0.5)
    deploy_tf = Transform(geometry="stretch", interp=data_cfg["interpolation"],
                          mean=data_cfg["mean"], std=data_cfg["std"], train=False)

    cfg = {
        "model": "wallpaper_head_v2", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=3, K pinned=4)", "geometry": "stretch",
        "label_unit": "render (image_id) — NO max-over-crops",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "class_weighting": "none",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "patience": args.patience,
        "border_crop": args.border_crop, "eval_frac": args.eval_frac,
        "seed": args.seed, "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": "max eval not-bad AP (rank by sigma(logit0)); patience==epochs",
        "split": "bootstrap->train, eval humanq3-only; location-disjoint AND max-tier-stratified",
        "batch_ids": ["2026-07-05_wallpaper_bootstrap_v1", "2026-07-05_wallpaper_humanq3_v1"],
        "init": "imagenet_backbone_fresh (NOT warm-started)",
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "task2_note": "P(>=4|>=3) ACTIVE (128 exceptionals, all humanq3) — tier-4 measurable",
        "forced_train_side": forced,
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
    eval_ft = np.asarray([r.family for r in ev])

    # --- train ---
    log.info(f"=== TRAIN: {len(tr)} renders/epoch, batch {args.batch_size}, "
             f"<= {args.epochs} epochs (patience {args.patience}) ===")
    best_ap, best_state, best_epoch = -1.0, None, -1
    best_cond, best_marg, best_sum = None, None, None
    since = 0
    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x).float()
            ranks = (y - 1).long()                       # {1,2,3,4} -> {0,1,2,3}
            loss = corn_loss(logits, ranks, num_classes=K)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(tr)

        if any(not torch.isfinite(p).all() for p in model.parameters()):
            log.error(f"  NaN/Inf at epoch {epoch} — aborting"); break

        cond, marg, ssum = predict_all(model, eval_loader, len(ev), device)
        ap_nb = _ap((eval_labels >= 2).astype(int), marg[:, 0])
        ap_gd = _ap((eval_labels >= 3).astype(int), marg[:, 1])   # marginal
        ap_ex = _ap((eval_labels >= 4).astype(int), marg[:, 2]) if (eval_labels >= 4).any() else float("nan")
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "ap_not_bad": sel, "ap_good": _nan(ap_gd), "ap_exceptional": _nan(ap_ex)})
        log.info(f"  epoch {epoch:2d}  loss {train_loss:.4f}  AP_notbad {sel:.4f}  "
                 f"AP_good {ap_gd:.4f}  AP_exc {ap_ex:.4f}  ({time.time()-t0:.1f}s)")

        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
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
               "eval_good_count": n_eval_good, "eval_good_low_power": bool(thin_good),
               "eval_exceptional_count": n_eval_exc, "eval_exc_low_power": bool(thin_exc),
               "split_strata": strata, "forced_train_side": forced,
               "task2_active": {"positives": int((eval_labels >= 4).sum()),
                                "note": "P(>=4|>=3) is now populated; tier-4 AP measurable."}}

    def f(x): return "  n/a" if x is None else f"{x:.3f}"

    log.info("=== NOTE: AP_good / AP_exc / gate are MARGINAL P(>=3), P(>=4) "
             "(cumprod of CORN conditionals); *_cond twins kept for v1-parity. ===")
    overall = eval_block(eval_labels, best_cond, best_marg, best_sum, np.ones(len(ev), bool))
    metrics["overall"] = overall
    log.info("=== EVAL BATTERY (frozen eval split, best epoch, humanq3-only) ===")
    log.info(f"  [overall] n={overall['n']} (not_bad={overall['n_not_bad']}, "
             f"good={overall['n_good']}, exc={overall['n_exceptional']})")
    log.info(f"     AP   not-bad {f(overall['ap_not_bad'])}   good {f(overall['ap_good'])} "
             f"(cond {f(overall['ap_good_cond'])})   exc {f(overall['ap_exceptional'])}   "
             f"4-vs-3 {f(overall['ap_4_vs_3'])}")
    log.info(f"     tier-3 gate P(>=3)>0.5:  recall {f(overall['recall_good_at_gate'])}  "
             f"precision {f(overall['precision_good_at_gate'])}  "
             f"fires {overall['gate_fires_frac']*100:.0f}%   "
             f"(cond-gate recall {f(overall['recall_good_at_gate_cond'])})")
    log.info(f"     AUC  nb-vs-bad {f(overall['auc_not_bad_vs_bad'])}   "
             f"good-vs-rest {f(overall['auc_good_vs_rest'])}   4-vs-3 {f(overall['auc_4_vs_3'])}")
    log.info(f"     Spearman(score vs tier) {f(overall['spearman_score_vs_tier'])}   "
             f"mean_score_by_tier {overall['mean_score_by_tier']}")
    log.info(f"     confusion(true x pred, 1..4) {overall['confusion_true_x_pred']}")

    # --- powered-mass block (deg-2 mandelbrot + julia) ---
    powered_mask = np.isin(eval_ft, list(POWERED_FAMILIES))
    powered = eval_block(eval_labels, best_cond, best_marg, best_sum, powered_mask)
    metrics["powered_mass"] = {"families": sorted(POWERED_FAMILIES), "block": powered}
    if powered:
        log.info(f"  [POWERED deg-2+julia] n={powered['n']} (good={powered['n_good']}, "
                 f"exc={powered['n_exceptional']})")
        log.info(f"     AP  not-bad {f(powered['ap_not_bad'])}  good {f(powered['ap_good'])}  "
                 f"exc {f(powered['ap_exceptional'])}  4-vs-3 {f(powered['ap_4_vs_3'])}   "
                 f"gate recall {f(powered['recall_good_at_gate'])} prec {f(powered['precision_good_at_gate'])}")

    # --- re-tilt check: did good/tier-4 discrimination survive the bad-heavy union? ---
    # Two independent tests: (a) good-vs-rest still real; (b) top-end 4-vs-3 separable.
    good_base = overall["n_good"] / max(overall["n"], 1)
    good_ok = overall["ap_good"] is not None and overall["ap_good"] > 1.4 * good_base
    top_ok = overall["auc_4_vs_3"] is not None and overall["auc_4_vs_3"] > 0.58
    good_collapsed = not good_ok
    top_collapsed = not top_ok
    metrics["retilt_check"] = {
        "ap_good_marginal": overall["ap_good"], "good_base_rate": float(good_base),
        "ap_exceptional_marginal": overall["ap_exceptional"],
        "auc_4_vs_3": overall["auc_4_vs_3"], "exc_base_rate": float(n_eval_exc / max(len(ev), 1)),
        "mean_score_by_tier": overall["mean_score_by_tier"],
        "good_discrimination_survived": bool(good_ok),
        "top_end_separable": bool(top_ok),
        "note": (
            ("GOOD survived (good-AP > 1.4x base). " if good_ok else
             "GOOD weak vs base rate -> bad-negatives may be pulling the head to a bad-vs-okay "
             "boundary; subsample bootstrap next pass. ") +
            ("TOP-END (4-vs-3) separable." if top_ok else
             "TOP-END collapsed: 4-vs-3 ~ chance and mean-score(tier4) <= mean-score(tier3) — the "
             "exceptional tier, though populated, yields no usable top-end signal (a labeling/"
             "task-subtlety limit, not obviously fixed by rebalancing bad negatives).")),
    }
    log.info(f"  [RE-TILT] good-AP {f(overall['ap_good'])} (base {good_base:.3f}) -> "
             f"{'GOOD SURVIVED' if good_ok else 'GOOD WEAK — subsample bootstrap'} | "
             f"4-vs-3 AUC {f(overall['auc_4_vs_3'])} -> "
             f"{'TOP-END SEPARABLE' if top_ok else 'TOP-END COLLAPSED (tier-4 not distinguishable from tier-3)'}")

    # --- per-family block ---
    metrics["families"] = {}
    for fam in sorted(set(eval_ft.tolist())):
        mask = eval_ft == fam
        blk = eval_block(eval_labels, best_cond, best_marg, best_sum, mask)
        metrics["families"][fam] = blk
        lb = eval_labels[mask]
        good_pct = 100.0 * (lb >= 3).mean()
        powered_tag = "*" if fam in POWERED_FAMILIES else " "
        log.info(f"  {powered_tag}[{fam:18s}] n={blk['n']:3d}  good%={good_pct:5.1f}  "
                 f"AP_nb {f(blk['ap_not_bad'])}  AP_good {f(blk['ap_good'])}  "
                 f"AP_exc {f(blk['ap_exceptional'])}")

    # --- freeze per-render eval scores ---
    scores_path = out_dir / "eval_scores.jsonl"
    with open(scores_path, "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "batch": r.batch,
                "family": r.family, "fractal_type": r.fractal_type, "label": r.label,
                # marginal gate probs (downstream gate uses these)
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_ge4": float(best_marg[i, 2]),
                # raw CORN conditionals (v1-parity)
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "p_exc_cond": float(best_cond[i, 2]),
                "score": float(best_sum[i]),
            }) + "\n")
    log.info(f"  froze {len(ev)} eval scores -> {scores_path}")

    # --- montage ---
    try:
        mpath = build_montage(ev, best_sum, out_dir / "eval_montage.png")
        log.info(f"  wrote montage -> {mpath}")
    except Exception as e:
        log.warning(f"  montage failed: {e}")

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

    log.info("================= WALLPAPER HEAD v2 SUMMARY =================")
    log.info(f"best epoch {best_epoch}  eval not-bad AP {best_ap:.4f}  "
             f"good AP {f(overall['ap_good'])}  tier-4 AP {f(overall['ap_exceptional'])}  "
             f"tier-3 recall@gate {f(overall['recall_good_at_gate'])}  "
             f"Spearman {f(overall['spearman_score_vs_tier'])}")
    log.info(f"  (eval humanq3-only, n={len(ev)}; good n={n_eval_good}"
             f"{' LOW-POWER' if thin_good else ''}, exc n={n_eval_exc}"
             f"{' LOW-POWER' if thin_exc else ''})")
    log.info(f"  checkpoints -> {metrics['checkpoints']}")
    log.info("DONE")


if __name__ == "__main__":
    main()
