"""Train the palette-preference scorer v2 on combined coldstart_v2 + warmstart_v1.

Single-tower f(image)->scalar utility model (MobileNetV4-conv-small, timm,
ImageNet-pretrained, head replaced with a single scalar). Trained with a
margin-ranking loss over CROSS-TIER ordered pairs only, batched by query and
normalized per query. Location-disjoint 80/20 split (gate: zero overlap).

v2 vs v1 — two deliberate changes: (a) DATA: v1 trained on coldstart alone (raw
6-of-draw pools -> wide good/bad spread); v2 unions coldstart_v2 + warmstart_v1,
the latter adding 200 v1-concentrated queries whose within-query distinctions are
good-vs-good (the fine top-end resolution v1 never saw). (b) TRAINING PAIRS: v2
EXCLUDES good_vs_okay from the margin-ranking loss (good_vs_bad + okay_vs_bad
only, TRAIN_INCLUDE_PAIR_TYPES). The all-pairs recipe collapses at epoch 1 (that
run is archived under data/queries/scorer/v2_combined_allpairs_FAILED/); dropping
the ambiguous good-vs-okay direction from TRAINING is what unblocks it. good_vs_okay
is still SCORED at eval as a held-out direction. Everything else — model, aug,
loss form, LRs, margin, batch size, patience — is frozen from v1.

Matched split (interpretability seam): coldstart locations reuse v1's assignment
VERBATIM (v2's coldstart-val == v1's val locations -> clean regression check);
warmstart gets a fresh location-disjoint 80/20 (seed 0). See data.split_combined.

Reporting is decomposed per batch: coldstart-val regresses against the promoted
v2 reference (V2_REFERENCE); warmstart-val is the new fine-resolution regime (no
baseline).

Artifacts (persistent, survive `rm -r out/*`):
    data/queries/scorer/v2/
        model_best.pt   (state_dict + config; best by val pair accuracy)
        model_last.pt
        metrics.json    (headline + per-pair-type + per-batch + surfacing + per-epoch)
        config.json     (all constants, seed, model string, aug)
        split_manifest.json  (combined, reproducible)
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

OUT_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v2")
V1_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v1")
V1_MANIFEST = os.path.join(V1_DIR, "split_manifest.json")

# Which cross-tier pair types TRAINING optimizes. good_vs_okay is EXCLUDED by
# default: the all-pairs recipe COLLAPSES at epoch 1 (the combined-all-pairs run,
# archived under data/queries/scorer/v2_combined_allpairs_FAILED/); good_vs_bad +
# okay_vs_bad only is the validated, promoted recipe (best_epoch 26). This is an
# EXPLICIT, LOGGED choice (dumped as config["train_include_pair_types"]), never an
# implicit None=all. To reproduce the old collapsing all-pairs behavior you must
# ask for it by name -- set this to all three types. EVALUATION is unaffected:
# eval_pair_accuracy always scores all three types (include_types stays None there),
# so good_vs_okay is reported as a HELD-OUT direction in the val tables.
TRAIN_INCLUDE_PAIR_TYPES = ("good_vs_bad", "okay_vs_bad")

# v1 reported val baselines (coldstart-only), kept for the historical record.
V1_BASELINE = {
    "pair_overall": 0.656,
    "pair_good_vs_bad": 0.773,
    "pair_okay_vs_bad": 0.678,
    "pair_good_vs_okay": 0.578,
    "surf_top1_good": 0.425,
    "surf_top1_bad": 0.075,
    "surf_top3_contains_good": 0.850,
}

# Promoted v2 (gvo-excluded) reference — coldstart-val numbers from the promoted
# exp_no_gvo run (best_epoch 26). good_vs_okay here is HELD-OUT (trained on neither
# batch). Future runs regress the matched coldstart-val against THESE, not v1's.
V2_REFERENCE = {
    "pair_overall": 0.677,
    "pair_good_vs_bad": 0.825,
    "pair_okay_vs_bad": 0.726,
    "pair_good_vs_okay": 0.561,   # HELD-OUT (excluded from training)
    "surf_top1_good": 0.500,
    "surf_top1_bad": 0.075,
    "surf_top3_contains_good": 0.850,
    "best_epoch": 26,
}

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
TIER_NAME = {v: k for k, v in D.TIER_RANK.items()}


def query_pairs_from_ranks(ranks: torch.Tensor, include_types=None):
    """ranks: [6] long tier ranks. Return list of (hi_idx, lo_idx) cross-tier pairs.

    include_types: optional set of pair-type names (subset of
    {good_vs_bad, okay_vs_bad, good_vs_okay}). When given, keep only pairs whose
    type is in the set -- the mechanism for dropping a pair type from TRAINING.
    Default None -> every cross-tier pair (unchanged v1/v2 behavior; EVALUATION
    always calls with None so the val report scores all three types regardless)."""
    pairs = []
    n = ranks.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ranks[i] > ranks[j]:
                if include_types is not None:
                    name = D.pair_type_name(TIER_NAME[ranks[i].item()], TIER_NAME[ranks[j].item()])
                    if name not in include_types:
                        continue
                pairs.append((i, j))
    return pairs


def batch_margin_loss(scores: torch.Tensor, ranks: torch.Tensor, include_types=None):
    """scores: [B,6], ranks: [B,6]. Per-query mean hinge over cross-tier pairs,
    then mean across queries that HAVE pairs. Returns (loss, n_correct, n_pairs).
    include_types filters which cross-tier pair types contribute (see
    query_pairs_from_ranks); None -> all types."""
    B = scores.shape[0]
    per_query_losses = []
    n_correct = 0
    n_pairs = 0
    for b in range(B):
        pairs = query_pairs_from_ranks(ranks[b], include_types)
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


# ================= CENSUS =================
def census(qlist):
    """Tier counts, cross-tier pair counts by type, within-tier ties dropped."""
    from collections import Counter

    tier_ct = Counter(t for q in qlist for t in q.tiers)
    pair_ct = Counter()
    within = 0
    for q in qlist:
        n = len(q.tiers)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = D.TIER_RANK[q.tiers[i]], D.TIER_RANK[q.tiers[j]]
                if ri == rj:
                    within += 1
                else:
                    hi, lo = (q.tiers[i], q.tiers[j]) if ri > rj else (q.tiers[j], q.tiers[i])
                    pair_ct[D.pair_type_name(hi, lo)] += 1
    return dict(tier_ct), dict(pair_ct), within


def eval_subset(model, qlist, device, nw):
    """Pair-direction accuracy (overall + per-type) over a query subset, deploy transform."""
    if not qlist:
        return 0.0, {k: [0, 0] for k in ("good_vs_bad", "good_vs_okay", "okay_vs_bad")}
    ds = D.QueryDataset(qlist, train=False)
    ld = DataLoader(ds, batch_size=BATCH_QUERIES, shuffle=False,
                    collate_fn=D.collate_queries, num_workers=nw)
    return eval_pair_accuracy(model, ld, device)


def _acc(pt):
    return {k: (v[0] / v[1] if v[1] else None) for k, v in pt.items()}


# ================= REPORT =================
def _report_tables(val_acc, val_pt, cold_acc, cold_pt, warm_acc, warm_pt,
                   surf_cold, surf_warm, train_acc, best_epoch):
    B = V2_REFERENCE  # promoted gvo-excluded reference; future runs regress against this

    def a(pt, k):
        c, n = pt[k]
        return (c / n if n else None), n

    def d(x, base):
        if x is None:
            return "n/a"
        return f"{x:.3f} (ref {base:.3f}, d{x-base:+.3f})"

    def f3(x):
        return f"{x:.3f}" if x is not None else "n/a"

    log("")
    log("=== (1) PAIR-DIRECTION ACCURACY ===")
    log("--- coldstart-val [MATCHED to v2 reference -> regression check; good_vs_okay HELD-OUT] ---")
    log(f"  overall      {d(cold_acc, B['pair_overall'])}")
    log(f"  good_vs_bad  {d(a(cold_pt,'good_vs_bad')[0], B['pair_good_vs_bad'])}  (n={a(cold_pt,'good_vs_bad')[1]})")
    log(f"  okay_vs_bad  {d(a(cold_pt,'okay_vs_bad')[0], B['pair_okay_vs_bad'])}  (n={a(cold_pt,'okay_vs_bad')[1]})")
    log(f"  good_vs_okay {d(a(cold_pt,'good_vs_okay')[0], B['pair_good_vs_okay'])}  (n={a(cold_pt,'good_vs_okay')[1]})")
    log("--- warmstart-val [new fine-resolution regime, no v1 baseline] ---")
    log(f"  overall      {f3(warm_acc)}")
    log(f"  good_vs_bad  {f3(a(warm_pt,'good_vs_bad')[0])}  (n={a(warm_pt,'good_vs_bad')[1]})")
    log(f"  okay_vs_bad  {f3(a(warm_pt,'okay_vs_bad')[0])}  (n={a(warm_pt,'okay_vs_bad')[1]})")
    log(f"  good_vs_okay {f3(a(warm_pt,'good_vs_okay')[0])}  (n={a(warm_pt,'good_vs_okay')[1]})")
    log(f"--- combined-val overall {f3(val_acc)}   | train overall {f3(train_acc)} (overfit check) | best_epoch={best_epoch}")

    log("")
    log("=== (2) SURFACING / SELECTION (within-query, per batch) ===")

    def surf_block(name, m, matched):
        if m is None:
            log(f"--- {name}: no val queries ---")
            return
        p, s = m["primary"], m["secondary"]
        mr = s["mean_norm_rank_by_tier"]
        log(f"--- {name} ({m['n_val_queries']} queries){'  [v2-ref-comparable]' if matched else '  [v1-concentrated, NOT coldstart-comparable]'} ---")
        if matched:
            log(f"  top1-good {d(p['top1_good_rate'], B['surf_top1_good'])}")
            log(f"  top1-bad  {d(p['top1_bad_rate'], B['surf_top1_bad'])}")
            log(f"  top3-contains-good {d(s['top3_contains_good_rate'], B['surf_top3_contains_good'])}")
        else:
            log(f"  top1-good {f3(p['top1_good_rate'])}   top1-bad {f3(p['top1_bad_rate'])}")
            log(f"  top3-contains-good {f3(s['top3_contains_good_rate'])}")
        log(f"  top2-contains-good {f3(s['top2_contains_good_rate'])}   "
            f"good-recall@3 {f3(s['good_recall_in_top3'])} ({s['total_good_candidates']} goods)")
        log(f"  mean norm rank good/okay/bad (0=top) {f3(mr['good'])}/{f3(mr['okay'])}/{f3(mr['bad'])}")

    surf_block("coldstart-val", surf_cold, matched=True)
    surf_block("warmstart-val", surf_warm, matched=False)


# ================= MAIN =================
def main():
    global _LOG_FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(OUT_DIR, "train.log"), "w")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data assembly (combined) ----
    queries = D.load_combined_queries()
    train_q, val_q, manifest = D.split_combined(queries, VAL_FRAC, SEED, V1_MANIFEST)

    cold_q = [q for q in queries if q.batch == D.COLDSTART.name]
    warm_q = [q for q in queries if q.batch == D.WARMSTART.name]

    log("=== DATA ASSEMBLY (combined coldstart_v2 + warmstart_v1) ===")
    log(f"excluded queries (named, zero pairs): {D.EXCLUDED_QUERIES}")
    for name, ql in (("coldstart_v2", cold_q), ("warmstart_v1", warm_q)):
        tc, pc, wd = census(ql)
        log(f"[{name}] queries(pass-1)={len(ql)}  tiers={tc}")
        log(f"[{name}] cross-tier pairs={pc} total={sum(pc.values())}  within-tier dropped={wd}")
    tc_all, pc_all, wd_all = census(queries)
    log(f"[combined] queries={len(queries)}  tiers={tc_all}  "
        f"cross-tier total={sum(pc_all.values())}  within-tier dropped={wd_all}")

    log("=== SPLIT (matched combined, location-disjoint 80/20) ===")
    log(f"seed={SEED} val_frac={VAL_FRAC}  (coldstart split reused verbatim from v1)")
    log(f"locations: train={manifest['n_locations_train']} val={manifest['n_locations_val']}")
    log(f"queries:   train={manifest['n_queries_train']} val={manifest['n_queries_val']}")
    log(f"  coldstart train/val queries: {manifest['coldstart_train']['n_queries']}"
        f"/{manifest['coldstart_val']['n_queries']}  "
        f"locations: {manifest['coldstart_train']['n_locations']}"
        f"/{manifest['coldstart_val']['n_locations']}")
    log(f"  warmstart train/val queries: {manifest['warmstart_train']['n_queries']}"
        f"/{manifest['warmstart_val']['n_queries']}  "
        f"locations: {manifest['warmstart_train']['n_locations']}"
        f"/{manifest['warmstart_val']['n_locations']}")
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
    loss, _, _ = batch_margin_loss(scores, ranks, TRAIN_INCLUDE_PAIR_TYPES)
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
        "task": "palette-preference scorer v2 (single-tower utility, margin-ranking)",
        "version": "v2",
        "data": "combined coldstart_v2 + warmstart_v1 (union, provenance-blind)",
        "excluded_queries": list(D.EXCLUDED_QUERIES),
        "split_design": "matched: coldstart reuses v1 split verbatim; warmstart fresh 80/20 seed 0",
        "frozen_from_v1": "model, input/geometry, aug, loss, margin, LRs, wd, batch, patience",
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "eval_pair_types": "ALL (held-out types still scored in the val report)",
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
        "excluded_pass2_repeats_per_batch": 20,
        "v1_baseline": V1_BASELINE,
        "v2_reference": V2_REFERENCE,
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
            loss, nc, npr = batch_margin_loss(scores, ranks, TRAIN_INCLUDE_PAIR_TYPES)
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

    # ---- final metrics (reload best) ----
    ck = torch.load(os.path.join(OUT_DIR, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])

    # (1) pair-direction accuracy: overall + per-batch (coldstart-val is v1-matched)
    val_acc, val_per_type = eval_pair_accuracy(model, val_ld, device)
    cold_val_q = [q for q in val_q if q.batch == D.COLDSTART.name]
    warm_val_q = [q for q in val_q if q.batch == D.WARMSTART.name]
    cold_acc, cold_per_type = eval_subset(model, cold_val_q, device, nw)
    warm_acc, warm_per_type = eval_subset(model, warm_val_q, device, nw)
    train_eval_ld = DataLoader(D.QueryDataset(train_q, train=False), batch_size=BATCH_QUERIES,
                               shuffle=False, collate_fn=D.collate_queries, num_workers=nw)
    train_acc_final, train_per_type = eval_pair_accuracy(model, train_eval_ld, device)

    # (2) surfacing/selection metrics, per batch (reuse surfacing_eval within-query logic)
    import surfacing_eval as SE  # lazy: avoids circular import at module load
    surf_cold = SE.compute_metrics(SE.score_queries(model, cold_val_q, device)) if cold_val_q else None
    surf_warm = SE.compute_metrics(SE.score_queries(model, warm_val_q, device)) if warm_val_q else None

    def _pt(pt):
        return {k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]} for k, v in pt.items()}

    # (3) census per batch (train/val split-aware)
    def batch_census(name):
        tr = [q for q in train_q if q.batch == name]
        va = [q for q in val_q if q.batch == name]
        tc_t, pc_t, wd_t = census(tr)
        tc_v, pc_v, wd_v = census(va)
        return {
            "train": {"n_queries": len(tr), "n_locations": len({q.location_key for q in tr}),
                      "tier_counts": tc_t, "cross_tier_pairs": pc_t, "within_tier_dropped": wd_t},
            "val": {"n_queries": len(va), "n_locations": len({q.location_key for q in va}),
                    "tier_counts": tc_v, "cross_tier_pairs": pc_v, "within_tier_dropped": wd_v},
        }

    metrics = {
        "headline_val_pair_acc": val_acc,
        "human_consistency_ceiling": 0.943,
        "best_epoch": best_epoch,
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "v1_baseline": V1_BASELINE,
        "v2_reference": V2_REFERENCE,
        "pair_direction": {
            "combined_val": {"overall": val_acc, "per_type": _pt(val_per_type)},
            "coldstart_val": {"overall": cold_acc, "per_type": _pt(cold_per_type),
                              "note": "v1-matched (same val locations) -> regression check"},
            "warmstart_val": {"overall": warm_acc, "per_type": _pt(warm_per_type),
                              "note": "new fine-resolution regime, no v1 baseline"},
            "train_overall": train_acc_final,
            "train_per_type": _pt(train_per_type),
        },
        "surfacing": {
            "coldstart_val": surf_cold,
            "warmstart_val": surf_warm,
            "note": "within-query; warmstart is v1-concentrated so top1-good is mechanically high, "
                    "NOT comparable to coldstart -- read per batch, never pooled",
        },
        "census": {
            "coldstart_v2": batch_census(D.COLDSTART.name),
            "warmstart_v1": batch_census(D.WARMSTART.name),
            "excluded_queries": list(D.EXCLUDED_QUERIES),
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

    _report_tables(val_acc, val_per_type, cold_acc, cold_per_type, warm_acc, warm_per_type,
                   surf_cold, surf_warm, train_acc_final, best_epoch)
    log(f"artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
