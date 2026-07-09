"""Train the wallpaper-quality head (v3) — dramatic batch folded in, multi-seed.

Forks ``train_wallpaper_v2`` (every footgun there stays pinned; read that
docstring). What v3 changes — **data only + a multi-seed harness**; the model
config is byte-for-byte v2 (MobileNetV4, CORN K=4, geometric-only aug, 384x224
stretch, same loss/optim/schedule):

  1. **Data = the UNION of THREE batches**, all fully labeled:
       - bootstrap  (``wbv1_*``): 504 renders / 63 loc
       - humanq3    (``whq3_*``): 994 renders / 142 loc
       - dramatic   (``whd_*`` ): 1000 renders / 136 loc  (tier 168/392/277/163)
     = **2498 renders**. The dramatic batch is dramatic-inclusive: 512
     dramatic-palette + 488 pool-palette renders on reused-humanq3 and fresh
     machine-q3 locations. Its tier-3/4 mass grows the eval-good pool ~3x.

  2. **Split is NOT re-derived globally (load-bearing).**
       - bootstrap  -> train (as v2).
       - humanq3    -> the *identical* v2 split (``split_rows`` with a FIXED
         ``split_seed=0``). This keeps the old eval slice BYTE-IDENTICAL, so the
         regression comparison against deployed v2 is exact.
       - dramatic   -> honors its STAMPED ``provenance.split_side`` (preserved
         humanq3 rows sit on their original v2 side; fresh rows are
         location-disjoint and pre-assigned).
     A c-inclusive coordinate key (``full_coord`` incl. c_re/c_im so distinct
     Julia c at a shared base viewport are NOT conflated) is asserted
     single-sided across the whole union — no location spans train and eval.

  3. **Multi-seed (the measurability fix).** The split is fixed; we train
     ``--seeds`` (default 5) that differ ONLY in init/shuffle/aug seed, and
     report good-AP **mean +/- empirical SD** across seeds. This replaces v2's
     asserted +/-0.05 with a *measured* band. Deltas inside the band are noise.

  4. **Stratified eval on the preserved slice** (marginal ``p_ge`` gate, never
     the CORN conditional): overall + **by palette_source (dramatic vs pool)**
     [did the fold-in kill the OOD gap?] + **humanq3-only old slice** [pool
     regression vs deployed v2, side-by-side] + tier-4 ``p_ge4`` AP [readable
     for the first time?] + Spearman.

  HOLD: this stages v3 but does NOT flip ACTIVE. emit_v1 still points at v2.

    uv run python -m classifier.train_wallpaper_v3 --seeds "0 1 2 3 4"

Outputs -> data/wallpaper_head/v3/  (per-seed under v3/seed_<s>/).
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

from .data import Transform
from .model import BACKBONE, corn_loss, data_config
from .train_v2 import detect_device, set_seed
# Reuse v2's split + eval battery verbatim (byte-identity + no metric drift).
from .train_wallpaper_v2 import (
    K, POWERED_FAMILIES, build_montage, build_wallpaper_model, eval_block,
    label_hist, make_loader, predict_all, split_rows as split_v2, _ap, _nan,
)

ROOT = Path(__file__).resolve().parent.parent
BATCHES = ROOT / "data" / "wallpaper_corpus" / "batches"
# (batch name, image_id prefix regex, batch dir, label json). Order matters:
# bootstrap + humanq3 must be the first two (fed to v2's split verbatim).
SOURCES = [
    ("bootstrap", re.compile(r"(wbv1_\d+)_\d+$"),
     BATCHES / "2026-07-05_wallpaper_bootstrap_v1", ROOT / "labels" / "wallpaper_bootstrap_v1.json"),
    ("humanq3", re.compile(r"(whq3_\d+)_\d+$"),
     BATCHES / "2026-07-05_wallpaper_humanq3_v1", ROOT / "labels" / "wallpaper_humanq3_v1.json"),
    ("dramatic", re.compile(r"(whd_\d+)_\d+$"),
     BATCHES / "2026-07-09_wallpaper_headbatch_dramatic_v1",
     ROOT / "labels" / "wallpaper_headbatch_dramatic_v1.json"),
]
OUT_DIR = ROOT / "data" / "wallpaper_head" / "v3"
V2_EVAL_SCORES = ROOT / "data" / "wallpaper_head" / "v2" / "eval_scores.jsonl"
SPLIT_SEED = 0  # FIXED — reproduces v2's humanq3 split byte-identically.

log = logging.getLogger("train_wallpaper_v3")


# --------------------------------------------------------------------------- #
# Rows — one per render (image_id). No crop aggregation (v2 contract).
# --------------------------------------------------------------------------- #
@dataclass
class WRow:
    image_id: str
    label: int
    jpg: Path
    loc: str
    fractal_type: str
    batch: str            # "bootstrap" | "humanq3" | "dramatic"
    family: str
    coord: tuple          # (cx,cy,fw,type) — v2-compatible key for split_v2()
    full_coord: tuple     # (cx,cy,fw,type,c_re,c_im) — c-inclusive disjointness key
    palette_source: str   # "dramatic" | "pool"  (pool = boot/hq3 pool-palette renders)
    split_side: str | None    # dramatic: stamped "train"/"eval"; else None
    split_origin: str | None  # dramatic provenance.split_origin; else None


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
            full_coord = (rd["cx"], rd["cy"], rd["fw"], rd["fractal_type"],
                          rd.get("c_re"), rd.get("c_im"))
            prov = r["provenance"]
            if batch_name == "dramatic":
                psrc = prov["palette_source"]           # "dramatic" | "pool"
                side = prov["split_side"]               # stamped "train"/"eval"
                origin = prov["split_origin"]
            else:
                psrc, side, origin = "pool", None, None
            rows.append(WRow(iid, int(labels[iid]), jpg, m.group(1), rd["fractal_type"],
                             batch_name, prov["family"], coord, full_coord, psrc, side, origin))
            seen.add(iid)
        extra = set(labels) - seen
        if extra:
            raise ValueError(f"[{batch_name}] {len(extra)} labels have no row: {sorted(extra)[:5]}...")
    return rows


# --------------------------------------------------------------------------- #
# Union split. humanq3 side comes from v2's split (fixed seed); dramatic honors
# its stamped side; bootstrap -> train. Disjointness asserted on full_coord.
# --------------------------------------------------------------------------- #
def split_union(rows: list[WRow]):
    boot_hq3 = [r for r in rows if r.batch in ("bootstrap", "humanq3")]
    # v2's split over boot+hq3 with the FIXED split seed -> byte-identical eval.
    _, ev_v2, hq3_eval_locs, strata, forced = split_v2(boot_hq3, eval_frac=0.30, seed=SPLIT_SEED)
    assert all(r.batch == "humanq3" for r in ev_v2), "v2 eval must be humanq3-only"

    def side_of(r: WRow) -> str:
        if r.batch == "bootstrap":
            return "train"
        if r.batch == "humanq3":
            return "eval" if r.loc in hq3_eval_locs else "train"
        # dramatic honors its stamped side
        if r.split_side not in ("train", "eval"):
            raise ValueError(f"dramatic row {r.image_id} has bad split_side={r.split_side!r}")
        return r.split_side

    train = [r for r in rows if side_of(r) == "train"]
    ev = [r for r in rows if side_of(r) == "eval"]

    # Location-disjointness across the WHOLE union on the c-inclusive key.
    coord_sides: dict[tuple, set] = defaultdict(set)
    for r in rows:
        coord_sides[r.full_coord].add(side_of(r))
    spanning = {c for c, s in coord_sides.items() if len(s) > 1}
    if spanning:
        raise AssertionError(f"{len(spanning)} locations span both sides (e.g. {list(spanning)[:3]})")

    # The humanq3 eval slice must be byte-identical to v2's eval set.
    old_slice_ids = {r.image_id for r in ev if r.batch == "humanq3"}
    v2_ids = {r.image_id for r in ev_v2}
    if old_slice_ids != v2_ids:
        raise AssertionError("humanq3 eval slice diverged from v2 — old slice not byte-identical")
    return train, ev, hq3_eval_locs, strata, forced, sorted(old_slice_ids)


# --------------------------------------------------------------------------- #
# One training run (single seed). Returns per-render best-epoch scores + a
# per-seed metrics dict. Saves the checkpoint under out_dir/seed_<seed>/.
# --------------------------------------------------------------------------- #
def train_one_seed(seed, tr, ev, args, device, data_cfg, train_tf, deploy_tf, cfg, seed_dir):
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True).to(device)
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
        ap_ex = _ap((eval_labels >= 4).astype(int), marg[:, 2]) if (eval_labels >= 4).any() else float("nan")
        sel = -1.0 if (ap_nb is None or not np.isfinite(ap_nb)) else ap_nb
        history.append({"epoch": epoch, "train_loss": train_loss, "ap_not_bad": sel,
                        "ap_good": _nan(ap_gd), "ap_exceptional": _nan(ap_ex)})
        log.info(f"[seed {seed}] epoch {epoch:2d}  loss {train_loss:.4f}  AP_nb {sel:.4f}  "
                 f"AP_good {ap_gd:.4f}  AP_exc {ap_ex:.4f}  ({time.time()-t0:.1f}s)")
        if sel > best_ap:
            best_ap, best_epoch = sel, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cond, best_marg, best_sum = cond, marg, ssum
    log.info(f"[seed {seed}] best epoch {best_epoch}: not-bad AP {best_ap:.4f} "
             f"(wall {time.time()-t_start:.0f}s)")

    seed_cfg = dict(cfg, seed=seed, best_epoch=best_epoch, split_seed=SPLIT_SEED)
    torch.save({"state_dict": best_state, "config": seed_cfg}, seed_dir / "model_best.pt")

    # freeze this seed's eval scores (marginal + conditional twins)
    with open(seed_dir / "eval_scores.jsonl", "w") as fh:
        for i, r in enumerate(ev):
            fh.write(json.dumps({
                "image_id": r.image_id, "loc": r.loc, "batch": r.batch, "family": r.family,
                "fractal_type": r.fractal_type, "palette_source": r.palette_source,
                "label": r.label,
                "p_ge2": float(best_marg[i, 0]), "p_ge3": float(best_marg[i, 1]),
                "p_ge4": float(best_marg[i, 2]),
                "p_not_bad": float(best_cond[i, 0]), "p_good_cond": float(best_cond[i, 1]),
                "p_exc_cond": float(best_cond[i, 2]), "score": float(best_sum[i]),
            }) + "\n")

    del train_loader, eval_loader, model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, "best_epoch": best_epoch, "val_best_not_bad_ap": _nan(best_ap),
            "history": history, "checkpoint": str(seed_dir / "model_best.pt")}, \
        best_cond, best_marg, best_sum


# --------------------------------------------------------------------------- #
# Cross-seed aggregation of a single scalar metric produced by eval_block.
# --------------------------------------------------------------------------- #
def agg(blocks, key):
    vals = [b[key] for b in blocks if b is not None and b.get(key) is not None]
    if not vals:
        return {"mean": None, "sd": None, "n_seeds": 0, "values": []}
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=0)),
            "n_seeds": len(vals), "values": [float(v) for v in a]}


def fmt(d):
    if d["mean"] is None:
        return "  n/a"
    return f"{d['mean']:.3f}+/-{d['sd']:.3f}"


# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train wallpaper head v3 (dramatic union, multi-seed).")
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
        raise SystemExit("need >=3 seeds for a measured good-AP band")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "train.log"),
                                  logging.StreamHandler(sys.stdout)])
    device = detect_device(args.device)
    log.info(f"device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}  seeds={seeds}")

    # --- data + split (once; split is seed-independent) ---
    rows = load_rows()
    log.info(f"loaded {len(rows)} renders  by_batch={dict(Counter(r.batch for r in rows))}  "
             f"union_tier_hist={label_hist(rows)}")
    for b in ("bootstrap", "humanq3", "dramatic"):
        sub = [r for r in rows if r.batch == b]
        log.info(f"  {b:9s}: {len(sub)} renders / {len({r.loc for r in sub})} loc  {label_hist(sub)}")

    tr, ev, hq3_eval_locs, strata, forced, old_slice_ids = split_union(rows)
    log.info(f"=== union split (split_seed={SPLIT_SEED}, FIXED) ===")
    log.info(f"  train {len(tr)} renders / {len({r.loc for r in tr})} loc  {label_hist(tr)}  "
             f"by_batch={dict(Counter(r.batch for r in tr))}")
    log.info(f"  eval  {len(ev)} renders / {len({r.loc for r in ev})} loc  {label_hist(ev)}  "
             f"by_batch={dict(Counter(r.batch for r in ev))}")
    eh = label_hist(ev)
    n_eval_good, n_eval_exc = eh[3], eh[4]
    log.info(f"  realized eval-good renders (tier>=3) = {n_eval_good + n_eval_exc}  "
             f"(tier3={n_eval_good}, tier4={n_eval_exc})  [v2 had 96: 64+32]")
    log.info(f"  old slice (humanq3 eval, byte-identical to v2) = {len(old_slice_ids)} renders")
    ev_ps = Counter(r.palette_source for r in ev)
    log.info(f"  eval palette_source: {dict(ev_ps)}")

    # --- config / transforms (identical to v2) ---
    probe = build_wallpaper_model(args.drop_rate, args.drop_path_rate, pretrained=True)
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
        "model": "wallpaper_head_v3", "target": "ordinal", "num_classes": K,
        "loss": "CORN ordinal (K-1=3, K pinned=4)", "geometry": "stretch",
        "label_unit": "render (image_id) — NO max-over-crops",
        "augmentation": "geometric only (border_crop + h/v flip); NO color, NO jpeg jitter",
        "class_weighting": "none", "epochs": args.epochs, "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
        "weight_decay": args.weight_decay, "drop_rate": args.drop_rate,
        "drop_path_rate": args.drop_path_rate, "border_crop": args.border_crop,
        "num_workers": args.num_workers, "grad_clip": 1.0, "amp": "off",
        "selection": "max eval not-bad AP (marginal P>=2); full schedule (no early stop)",
        "split": ("bootstrap->train; humanq3->v2 split (split_seed=0, byte-identical); "
                  "dramatic->stamped split_side; disjointness asserted on c-inclusive key"),
        "batch_ids": [s[2].name for s in SOURCES],
        "init": "imagenet_backbone_fresh (NOT warm-started)",
        "backbone": BACKBONE, "mean": data_cfg["mean"], "std": data_cfg["std"],
        "interpolation": data_cfg["interpolation"], "input_size": data_cfg["input_size"],
        "src_dims": [1280, 720], "target_dims": [384, 224], "black_thresh": 0.30,
        "split_seed": SPLIT_SEED, "seeds": seeds, "forced_train_side": forced,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    eval_labels = np.asarray([r.label for r in ev])
    eval_ft = np.asarray([r.family for r in ev])
    eval_ps = np.asarray([r.palette_source for r in ev])
    eval_batch = np.asarray([r.batch for r in ev])

    # masks for the reported slices
    mask_all = np.ones(len(ev), bool)
    mask_pool = eval_ps == "pool"
    mask_dram = eval_ps == "dramatic"
    mask_old = eval_batch == "humanq3"          # byte-identical old slice
    slices = {"overall": mask_all, "pool": mask_pool, "dramatic": mask_dram,
              "old_humanq3": mask_old}

    # --- multi-seed train ---
    per_seed = []                 # per-seed run info
    seed_blocks = defaultdict(list)   # slice -> [eval_block per seed]
    best_for_stage = None             # (not_bad_ap, seed, cond, marg, sum, dir)
    for seed in seeds:
        log.info(f"================= SEED {seed} =================")
        info, cond, marg, ssum = train_one_seed(
            seed, tr, ev, args, device, data_cfg, train_tf, deploy_tf, cfg,
            out_dir / f"seed_{seed}")
        per_seed.append(info)
        for name, mask in slices.items():
            seed_blocks[name].append(eval_block(eval_labels, cond, marg, ssum, mask))
        nb = info["val_best_not_bad_ap"] or -1.0
        if best_for_stage is None or nb > best_for_stage[0]:
            best_for_stage = (nb, seed, cond, marg, ssum, out_dir / f"seed_{seed}")
        # persist partial aggregate after each seed
        (out_dir / "per_seed.json").write_text(json.dumps(per_seed, indent=2))

    # --- cross-seed aggregation ---
    log.info("================= CROSS-SEED AGGREGATION =================")
    agg_metrics = {}
    headline_keys = ["ap_not_bad", "ap_good", "ap_exceptional", "ap_4_vs_3",
                     "auc_4_vs_3", "spearman_score_vs_tier"]
    for name, blocks in seed_blocks.items():
        b0 = next((b for b in blocks if b is not None), None)
        agg_metrics[name] = {
            "n": (b0["n"] if b0 else 0),
            "n_good": (b0["n_good"] if b0 else 0),
            "n_exceptional": (b0["n_exceptional"] if b0 else 0),
            **{k: agg(blocks, k) for k in headline_keys},
        }
        m = agg_metrics[name]
        log.info(f"  [{name:12s}] n={m['n']:3d} good={m['n_good']:3d} exc={m['n_exceptional']:3d}  "
                 f"AP_nb {fmt(m['ap_not_bad'])}  AP_good {fmt(m['ap_good'])}  "
                 f"AP_exc {fmt(m['ap_exceptional'])}  4v3AUC {fmt(m['auc_4_vs_3'])}  "
                 f"Spear {fmt(m['spearman_score_vs_tier'])}")

    # --- regression vs deployed v2 on the byte-identical old slice ---
    v2_scores = {}
    for line in V2_EVAL_SCORES.read_text().splitlines():
        if line.strip():
            d = json.loads(line); v2_scores[d["image_id"]] = d
    if set(old_slice_ids) != set(v2_scores):
        raise AssertionError("old-slice ids != v2 eval_scores ids — regression compare invalid")
    old_rows = [r for r in ev if r.batch == "humanq3"]
    lb_old = np.asarray([r.label for r in old_rows])
    v2_nb = _ap((lb_old >= 2).astype(int), np.asarray([v2_scores[r.image_id]["p_ge2"] for r in old_rows]))
    v2_gd = _ap((lb_old >= 3).astype(int), np.asarray([v2_scores[r.image_id]["p_ge3"] for r in old_rows]))
    v3_old = agg_metrics["old_humanq3"]
    regression = {
        "old_slice_n": len(old_rows),
        "v2_ap_not_bad": _nan(v2_nb), "v2_ap_good": _nan(v2_gd),
        "v3_ap_not_bad": v3_old["ap_not_bad"], "v3_ap_good": v3_old["ap_good"],
        "delta_not_bad_mean": (None if v3_old["ap_not_bad"]["mean"] is None or v2_nb is None
                               else float(v3_old["ap_not_bad"]["mean"] - v2_nb)),
        "delta_good_mean": (None if v3_old["ap_good"]["mean"] is None or v2_gd is None
                            else float(v3_old["ap_good"]["mean"] - v2_gd)),
        "note": ("v3 mean+/-SD vs deployed-v2 point estimate on the SAME renders. "
                 "A degradation beyond the seed band on not-bad or good AP is a regression."),
    }
    log.info("=== REGRESSION (old humanq3 eval, byte-identical; new v3 vs deployed v2) ===")
    log.info(f"  not-bad AP:  v2 {v2_nb:.3f}  ->  v3 {fmt(v3_old['ap_not_bad'])}  "
             f"(delta {regression['delta_not_bad_mean']:+.3f})")
    log.info(f"  good    AP:  v2 {v2_gd:.3f}  ->  v3 {fmt(v3_old['ap_good'])}  "
             f"(delta {regression['delta_good_mean']:+.3f})")

    # --- dramatic-vs-pool verdict ---
    dv = agg_metrics["dramatic"]; pv = agg_metrics["pool"]
    dram_gap_nb = (None if dv["ap_not_bad"]["mean"] is None or pv["ap_not_bad"]["mean"] is None
                   else float(dv["ap_not_bad"]["mean"] - pv["ap_not_bad"]["mean"]))
    dram_gap_gd = (None if dv["ap_good"]["mean"] is None or pv["ap_good"]["mean"] is None
                   else float(dv["ap_good"]["mean"] - pv["ap_good"]["mean"]))
    log.info(f"=== DRAMATIC vs POOL:  not-bad gap {dram_gap_nb:+.3f}   good gap {dram_gap_gd:+.3f}  "
             f"(dramatic - pool; ~0 => fold-in closed the OOD gap) ===")

    # --- stage the checkpoint (best not-bad-AP seed) — HOLD, do NOT flip ---
    stage_nb, stage_seed, s_cond, s_marg, s_sum, s_dir = best_for_stage
    import shutil
    shutil.copy(s_dir / "model_best.pt", out_dir / "model_best.pt")
    shutil.copy(s_dir / "eval_scores.jsonl", out_dir / "eval_scores.jsonl")
    shutil.copy(ROOT / "classifier" / "inference.py", out_dir / "inference.py")
    try:
        build_montage(ev, s_sum, out_dir / "eval_montage.png")
    except Exception as e:
        log.warning(f"montage failed: {e}")

    metrics = {
        "seeds": seeds, "split_seed": SPLIT_SEED,
        "eval_n": len(ev), "eval_tier_hist": eh,
        "realized_eval_good_tier3plus": int(n_eval_good + n_eval_exc),
        "realized_eval_tier3": int(n_eval_good), "realized_eval_tier4": int(n_eval_exc),
        "v2_eval_good_tier3plus": 96, "old_slice_n": len(old_slice_ids),
        "split_strata": strata, "forced_train_side": forced,
        "eval_palette_source_hist": dict(ev_ps),
        "aggregate": agg_metrics, "per_seed": per_seed,
        "regression_vs_v2": regression,
        "dramatic_vs_pool": {"not_bad_gap_mean": dram_gap_nb, "good_gap_mean": dram_gap_gd},
        "staged": {"seed": stage_seed, "not_bad_ap": float(stage_nb),
                   "checkpoint": str(out_dir / "model_best.pt"),
                   "rule": "best per-seed eval not-bad AP",
                   "ACTIVE_STATUS": "STAGED — NOT flipped; emit_v1 still points at v2",
                   "one_line_flip": ("edit tools/wallpaper/emit_v1.py line 62: "
                                     'HEAD_CKPT = REPO / "data/wallpaper_head/v3/model_best.pt"'),
                   "rollback": 'revert emit_v1.py HEAD_CKPT to "data/wallpaper_head/v2/model_best.pt"'},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info("================= WALLPAPER HEAD v3 SUMMARY =================")
    ov = agg_metrics["overall"]
    log.info(f"  seeds={seeds}  eval n={len(ev)}  realized eval-good(tier>=3)="
             f"{n_eval_good + n_eval_exc} (was 96)")
    log.info(f"  OVERALL   not-bad {fmt(ov['ap_not_bad'])}  good {fmt(ov['ap_good'])}  "
             f"tier4 {fmt(ov['ap_exceptional'])}  Spearman {fmt(ov['spearman_score_vs_tier'])}")
    log.info(f"  DRAMATIC  not-bad {fmt(dv['ap_not_bad'])}  good {fmt(dv['ap_good'])}   "
             f"POOL not-bad {fmt(pv['ap_not_bad'])}  good {fmt(pv['ap_good'])}")
    log.info(f"  REGRESSION(old): not-bad v2 {v2_nb:.3f}->v3 {fmt(v3_old['ap_not_bad'])}  "
             f"good v2 {v2_gd:.3f}->v3 {fmt(v3_old['ap_good'])}")
    log.info(f"  STAGED -> {out_dir / 'model_best.pt'}  (seed {stage_seed}, HELD; ACTIVE still v2)")
    log.info(f"  FLIP: {metrics['staged']['one_line_flip']}")
    log.info("DONE")


if __name__ == "__main__":
    main()
