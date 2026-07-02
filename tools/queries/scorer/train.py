"""Train the bootstrap palette-preference scorer.

Single-tower f(image)->scalar utility model (MobileNetV4-conv-small, timm,
ImageNet-pretrained, head replaced with a single scalar). Trained with a
margin-ranking loss over CROSS-TIER ordered pairs only, batched by query and
normalized per query. Location-disjoint 80/20 split (gate: zero overlap).

Scope: bootstrap scorer only. Trains, reports held-out cross-tier pair-direction
accuracy vs the ~94.3% human ceiling, saves artifacts. No active-learning loop,
no sweep wiring.

Artifacts (persistent, survive `rm -r out/*`):
    data/queries/scorer/v1/
        model_best.pt   (state_dict + config; best by val pair accuracy)
        model_last.pt
        metrics.json    (headline + per-pair-type + per-epoch log)
        config.json     (all constants, seed, model string, aug)
        split_manifest.json
        train.log

Run:
    uv run python tools/queries/scorer/train.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402  (sibling module, script-style import)

# ================= NAMED CONSTANTS =================
SEED = 0
MODEL = "mobilenetv4_conv_small.e2400_r224_in1k"
INPUT_SIZE = D.INPUT_SIZE            # 224 (model native)
GEOMETRY = "squash"                  # squash-resize 1024x576 -> 224x224 (no letterbox)
AUG = "geometric-only: hflip p=0.5, vflip p=0.5 (NO color/photometric aug)"

MARGIN = 1.0                         # margin-ranking hinge margin
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
BATCH_QUERIES = 16                   # queries per batch (each = 6 candidate forwards)
MAX_EPOCHS = 40
PATIENCE = 8                         # early-stop on val pair accuracy
VAL_FRAC = 0.20

OUT_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v1")

_LOG_FH = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    # Windows console is cp1252 — keep stdout/log ASCII-safe.
    safe = line.encode("ascii", "replace").decode("ascii")
    print(safe, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


# ================= MODEL =================
def build_model():
    import timm

    m = timm.create_model(MODEL, pretrained=True, num_classes=1)
    return m


# ================= LOSS + PAIR ACCURACY =================
def query_pairs_from_ranks(ranks: torch.Tensor):
    """ranks: [6] long tier ranks. Return list of (hi_idx, lo_idx) cross-tier pairs."""
    pairs = []
    n = ranks.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ranks[i] > ranks[j]:
                pairs.append((i, j))
    return pairs


def batch_margin_loss(scores: torch.Tensor, ranks: torch.Tensor):
    """scores: [B,6], ranks: [B,6]. Per-query mean hinge over cross-tier pairs,
    then mean across queries that HAVE pairs. Returns (loss, n_correct, n_pairs)."""
    B = scores.shape[0]
    per_query_losses = []
    n_correct = 0
    n_pairs = 0
    for b in range(B):
        pairs = query_pairs_from_ranks(ranks[b])
        if not pairs:
            continue
        losses = []
        for hi, lo in pairs:
            diff = scores[b, hi] - scores[b, lo]
            losses.append(torch.clamp(MARGIN - diff, min=0.0))
            n_pairs += 1
            if diff.item() > 0:
                n_correct += 1
        per_query_losses.append(torch.stack(losses).mean())
    if not per_query_losses:
        return None, 0, 0
    loss = torch.stack(per_query_losses).mean()
    return loss, n_correct, n_pairs


@torch.no_grad()
def eval_pair_accuracy(model, loader, device):
    """Returns overall accuracy + per-pair-type {name: (correct, total)}."""
    model.eval()
    tier_name = {v: k for k, v in D.TIER_RANK.items()}
    per_type = {"good_vs_bad": [0, 0], "good_vs_okay": [0, 0], "okay_vs_bad": [0, 0]}
    tot_correct = tot = 0
    for imgs, ranks, _ in loader:
        B, C = imgs.shape[0], imgs.shape[1]
        flat = imgs.view(B * C, *imgs.shape[2:]).to(device)
        scores = model(flat).view(B, C)
        for b in range(B):
            for hi, lo in query_pairs_from_ranks(ranks[b]):
                correct = int((scores[b, hi] - scores[b, lo]).item() > 0)
                name = D.pair_type_name(tier_name[ranks[b, hi].item()], tier_name[ranks[b, lo].item()])
                per_type[name][1] += 1
                per_type[name][0] += correct
                tot += 1
                tot_correct += correct
    acc = tot_correct / tot if tot else 0.0
    return acc, per_type


# ================= MAIN =================
def main():
    global _LOG_FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(OUT_DIR, "train.log"), "w")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data assembly ----
    queries = D.load_queries()
    train_q, val_q, manifest = D.split_by_location(queries, VAL_FRAC, SEED)

    # assembly + pair stats (report)
    from collections import Counter

    tier_ct = Counter(t for q in queries for t in q.tiers)
    pair_ct = Counter()
    within_dropped = 0
    tier_name = {v: k for k, v in D.TIER_RANK.items()}
    for q in queries:
        seen_pairs = set()
        n = len(q.tiers)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = D.TIER_RANK[q.tiers[i]], D.TIER_RANK[q.tiers[j]]
                if ri == rj:
                    within_dropped += 1
                else:
                    hi, lo = (q.tiers[i], q.tiers[j]) if ri > rj else (q.tiers[j], q.tiers[i])
                    pair_ct[D.pair_type_name(hi, lo)] += 1

    log("=== DATA ASSEMBLY ===")
    log(f"queries used (pass-1 only): {len(queries)}  (excluded 20 pass-2 consistency repeats)")
    log(f"candidate tier counts: {dict(tier_ct)}")
    log(f"cross-tier ordered pairs by type: {dict(pair_ct)}  total={sum(pair_ct.values())}")
    log(f"within-tier (tie) pairs dropped: {within_dropped}")

    log("=== SPLIT (location-disjoint 80/20) ===")
    log(f"seed={SEED} val_frac={VAL_FRAC}")
    log(f"locations: total={manifest['n_locations_total']} "
        f"train={manifest['n_locations_train']} val={manifest['n_locations_val']}")
    log(f"queries: train={manifest['n_queries_train']} val={manifest['n_queries_val']}")
    log(f"ZERO-OVERLAP PROOF: train-cap-val locations = {manifest['location_overlap_count']} (must be 0)")
    log(f"type breakdown train: {manifest['type_breakdown_train']}")
    log(f"type breakdown val:   {manifest['type_breakdown_val']}")

    json.dump(manifest, open(os.path.join(OUT_DIR, "split_manifest.json"), "w"), indent=2)

    # ---- loaders ----
    train_ds = D.QueryDataset(train_q, train=True)
    val_ds = D.QueryDataset(val_q, train=False)
    nw = 4 if device.type == "cuda" else 0
    train_ld = DataLoader(train_ds, batch_size=BATCH_QUERIES, shuffle=True,
                          collate_fn=D.collate_queries, num_workers=nw, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=BATCH_QUERIES, shuffle=False,
                        collate_fn=D.collate_queries, num_workers=nw)

    # ---- model + optim ----
    model = build_model().to(device)
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "classifier" in name else backbone_params).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # ---- runtime estimate ----
    log("=== RUNTIME ESTIMATE ===")
    model.train()
    t0 = time.time()
    imgs, ranks, _ = next(iter(train_ld))
    B, Cn = imgs.shape[0], imgs.shape[1]
    flat = imgs.view(B * Cn, *imgs.shape[2:]).to(device)
    scores = model(flat).view(B, Cn)
    loss, _, _ = batch_margin_loss(scores, ranks)
    opt.zero_grad(); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_s = time.time() - t0
    steps_per_epoch = (len(train_ds) + BATCH_QUERIES - 1) // BATCH_QUERIES
    est_epoch = step_s * steps_per_epoch
    log(f"first train step (incl warmup/JIT): {step_s:.2f}s; steps/epoch={steps_per_epoch}; "
        f"rough est <={est_epoch:.0f}s/epoch, <={est_epoch*MAX_EPOCHS/60:.0f} min for {MAX_EPOCHS} epochs "
        f"(early-stop patience={PATIENCE} typically ends sooner)")

    # ---- config dump ----
    config = {
        "task": "bootstrap palette-preference scorer (single-tower utility, margin-ranking)",
        "model": MODEL,
        "input_size": INPUT_SIZE,
        "geometry": GEOMETRY,
        "aug": AUG,
        "loss": "margin-ranking over cross-tier pairs; per-query mean then batch mean",
        "margin": MARGIN,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_queries": BATCH_QUERIES,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "mean": list(D.IMAGENET_MEAN),
        "std": list(D.IMAGENET_STD),
        "interpolation": "bicubic",
        "device": device.type,
        "human_consistency_ceiling": 0.943,
        "pass1_only": True,
        "excluded_pass2_repeats": 20,
    }
    json.dump(config, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)

    # ---- train loop ----
    log("=== TRAIN ===")
    best_val = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    epoch_log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = ep_pairs = 0
        nb = 0
        for imgs, ranks, _ in train_ld:
            B, Cn = imgs.shape[0], imgs.shape[1]
            flat = imgs.view(B * Cn, *imgs.shape[2:]).to(device)
            scores = model(flat).view(B, Cn)
            loss, nc, npr = batch_margin_loss(scores, ranks)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
            ep_correct += nc; ep_pairs += npr
        train_acc = ep_correct / ep_pairs if ep_pairs else 0.0

        val_acc, val_per_type = eval_pair_accuracy(model, val_ld, device)
        rec = {
            "epoch": epoch,
            "train_loss": ep_loss / max(nb, 1),
            "train_pair_acc": train_acc,
            "val_pair_acc": val_acc,
            "val_per_type": {k: (v[0] / v[1] if v[1] else None) for k, v in val_per_type.items()},
        }
        epoch_log.append(rec)

        def _f(x):
            return f"{x:.3f}" if x is not None else "n/a"

        vp = rec["val_per_type"]
        log(f"epoch {epoch:02d}  loss={rec['train_loss']:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"[gvb={_f(vp['good_vs_bad'])} gvo={_f(vp['good_vs_okay'])} ovb={_f(vp['okay_vs_bad'])}]")

        # checkpoint last always
        torch.save({"state_dict": model.state_dict(), "config": config},
                   os.path.join(OUT_DIR, "model_last.pt"))
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch},
                       os.path.join(OUT_DIR, "model_best.pt"))
            log(f"  new best val_pair_acc={best_val:.4f} @ epoch {epoch} -> model_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log(f"early stop: no val improvement for {PATIENCE} epochs")
                break

    # ---- final metrics ----
    # reload best, report per-type on val
    ck = torch.load(os.path.join(OUT_DIR, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    val_acc, val_per_type = eval_pair_accuracy(model, val_ld, device)
    train_eval_ld = DataLoader(D.QueryDataset(train_q, train=False), batch_size=BATCH_QUERIES,
                               shuffle=False, collate_fn=D.collate_queries, num_workers=nw)
    train_acc_final, train_per_type = eval_pair_accuracy(model, train_eval_ld, device)

    metrics = {
        "headline_val_pair_acc": val_acc,
        "human_consistency_ceiling": 0.943,
        "best_epoch": best_epoch,
        "val_per_pair_type": {k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]} for k, v in val_per_type.items()},
        "train_pair_acc_final": train_acc_final,
        "train_per_pair_type": {k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]} for k, v in train_per_type.items()},
        "data_assembly": {
            "queries_used": len(queries),
            "excluded_pass2_repeats": 20,
            "tier_counts": dict(tier_ct),
            "cross_tier_pairs": dict(pair_ct),
            "within_tier_dropped": within_dropped,
        },
        "split": {
            "n_locations_train": manifest["n_locations_train"],
            "n_locations_val": manifest["n_locations_val"],
            "location_overlap_count": manifest["location_overlap_count"],
            "type_breakdown_train": manifest["type_breakdown_train"],
            "type_breakdown_val": manifest["type_breakdown_val"],
        },
        "epoch_log": epoch_log,
    }
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)

    log("=== RESULT ===")
    log(f"HEADLINE val cross-tier pair-direction acc = {val_acc:.4f}  (human ceiling ~0.943)  best_epoch={best_epoch}")
    log(f"val per-pair-type: " + ", ".join(
        f"{k}={(v[0]/v[1] if v[1] else float('nan')):.4f} (n={v[1]})" for k, v in val_per_type.items()))
    log(f"train pair acc (overfit check) = {train_acc_final:.4f}")
    log(f"artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
