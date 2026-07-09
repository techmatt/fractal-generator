"""Train pref-v3: fold prefv2_dramatic_v1 into the pref-v2 corpus and retrain.

Deliverable: a scorer that ranks DRAMATIC palettes (within-dramatic and
dramatic-vs-pool) WITHOUT regressing pool ranking. The union is coldstart_v2 +
warmstart_v1 (399 queries) + prefv2_dramatic_v1 (600) = 999 queries.

Training config is IDENTICAL to the deployed pref-v2 (train.py): same arch
(mobilenetv4_conv_small margin-ranking single-tower), same loss, same pair
construction (good_vs_bad + okay_vs_bad; good_vs_okay EXCLUDED from train, scored
at eval). Only the data changes; no hyperparameter re-tuning. Every mechanical
piece (build_model, batch_margin_loss, eval_pair_accuracy, query_pairs_from_ranks,
census, the constants) is imported from train.py so there is one source of truth.

Split (data.split_union_v3): coldstart reuses v1's assignment VERBATIM and
warmstart reuses its seed-0 stratified split -> the OLD-corpus val slice is
byte-identical to the deployed v2's, so the regression guard is a clean matched
comparison. dramatic gets a fresh location-disjoint stratified 80/20 (its 350
locations are all fresh/disjoint from the corpus's 388).

Eval is STRATIFIED (the point):
  - overall held-out pairwise accuracy
  - by query_type: within_dramatic / cross_source / param_variation
  - by palette_source pair-composition: dramatic-vs-dramatic / pool-vs-pool /
    dramatic-vs-pool (cross)
  - by dramatic skeleton + architecture axis (localize a thin axis if one exists)
  - held-out good_vs_okay (train-excluded hard direction)
Every number is read against the human consistency ceiling (95.6% this batch).

Regression guard: the retrained scorer AND the deployed v2 are BOTH scored on the
identical old-corpus-only val slice (coldstart + warmstart), side by side.

Deploy discipline: writes to data/queries/scorer/v3/ and does NOT flip the ACTIVE
pointer (data.ACTIVE_SCORER_DIR stays v2). One-line flip reported at the end.

Run:
    uv run python tools/queries/scorer/train_v3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import train as TV2  # noqa: E402  (mechanical helpers + FROZEN config, one source of truth)

# ---- config: FROZEN from the deployed pref-v2 (train.py). Only data changes. ----
SEED = TV2.SEED
MODEL = TV2.MODEL
MARGIN = TV2.MARGIN
BACKBONE_LR = TV2.BACKBONE_LR
HEAD_LR = TV2.HEAD_LR
WEIGHT_DECAY = TV2.WEIGHT_DECAY
BATCH_QUERIES = TV2.BATCH_QUERIES
MAX_EPOCHS = TV2.MAX_EPOCHS
PATIENCE = TV2.PATIENCE
VAL_FRAC = TV2.VAL_FRAC
TRAIN_INCLUDE_PAIR_TYPES = TV2.TRAIN_INCLUDE_PAIR_TYPES

build_model = TV2.build_model
batch_margin_loss = TV2.batch_margin_loss
eval_pair_accuracy = TV2.eval_pair_accuracy
query_pairs_from_ranks = TV2.query_pairs_from_ranks
census = TV2.census
TIER_NAME = TV2.TIER_NAME

OUT_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v3")
V2_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v2")
V1_MANIFEST = os.path.join(D.REPO, "data", "queries", "scorer", "v1", "split_manifest.json")

# Human consistency ceilings (pairwise ranking, both-passes-strict). Read EVERY
# accuracy against these, not 100%. 95.6% is prefv2_dramatic_v1's own pass-1/pass-2
# ceiling (537/562 = 0.9555); 94.3% is the old-corpus ceiling.
CEILING_DRAMATIC = 0.956
CEILING_OLD = 0.943

_LOG_FH = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


# ================= STRATIFIED PAIR EVAL =================
@torch.no_grad()
def collect_pair_records(model, qlist, device, nw):
    """Score every val query and emit one dict per cross-tier ordered pair, tagged
    with every stratification axis. All eval slices are computed from this list
    (no repeated forward passes)."""
    if not qlist:
        return []
    ds = D.QueryDataset(qlist, train=False)
    ld = DataLoader(ds, batch_size=BATCH_QUERIES, shuffle=False,
                    collate_fn=D.collate_queries, num_workers=nw)
    recs = []
    model.eval()
    for imgs, ranks, idxs in ld:
        B, C = imgs.shape[0], imgs.shape[1]
        flat = imgs.view(B * C, *imgs.shape[2:]).to(device)
        scores = model(flat).view(B, C).cpu()
        for b in range(B):
            q = qlist[idxs[b].item()]
            for hi, lo in query_pairs_from_ranks(ranks[b]):
                correct = int((scores[b, hi] - scores[b, lo]).item() > 0)
                ptype = D.pair_type_name(TIER_NAME[ranks[b, hi].item()],
                                         TIER_NAME[ranks[b, lo].item()])
                # pair source composition
                shi, slo = q.src_class[hi], q.src_class[lo]
                if shi == "dramatic" and slo == "dramatic":
                    src_pair = "dramatic_vs_dramatic"
                elif shi == "pool" and slo == "pool":
                    src_pair = "pool_vs_pool"
                elif {shi, slo} == {"dramatic", "pool"}:
                    src_pair = "dramatic_vs_pool"
                else:
                    src_pair = "other"
                recs.append({
                    "batch": q.batch,
                    "query_type": q.query_type,
                    "pair_type": ptype,
                    "src_pair": src_pair,
                    "hi_src": shi, "lo_src": slo,
                    "hi_skel": q.skeletons[hi], "lo_skel": q.skeletons[lo],
                    "hi_arch": q.architectures[hi], "lo_arch": q.architectures[lo],
                    "correct": correct,
                })
    return recs


def _acc(recs):
    n = len(recs)
    return (sum(r["correct"] for r in recs) / n if n else None), n


def _slice(recs, keyfn):
    """Group recs by keyfn(rec) -> {key: (acc, n)} sorted by key."""
    g = defaultdict(list)
    for r in recs:
        k = keyfn(r)
        if k is not None:
            g[k].append(r)
    return {k: _acc(g[k]) for k in sorted(g)}


# ================= REPORT =================
def _fmt(acc_n, ceiling=None):
    acc, n = acc_n
    if acc is None:
        return f"n/a (n=0)"
    s = f"{acc:.3f} (n={n})"
    if ceiling is not None:
        s += f"  [ceil {ceiling:.3f}, gap {acc-ceiling:+.3f}]"
    return s


# ================= MAIN =================
def main():
    global _LOG_FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(OUT_DIR, "train.log"), "w")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data assembly (3-batch union) ----
    queries = D.load_combined_queries()
    train_q, val_q, manifest = D.split_union_v3(queries, VAL_FRAC, SEED, V1_MANIFEST)

    log("=== DATA ASSEMBLY (union coldstart_v2 + warmstart_v1 + prefv2_dramatic_v1) ===")
    log(f"excluded queries (named, zero pairs): {D.EXCLUDED_QUERIES}")
    for spec in D.BATCHES:
        ql = [q for q in queries if q.batch == spec.name]
        tc, pc, wd = census(ql)
        log(f"[{spec.name}] queries(pass-1)={len(ql)}  tiers={tc}")
        log(f"[{spec.name}] cross-tier pairs={pc} total={sum(pc.values())}  within-tier dropped={wd}")
    tc_all, pc_all, wd_all = census(queries)
    log(f"[union] queries={len(queries)}  tiers={tc_all}  "
        f"cross-tier total={sum(pc_all.values())}  within-tier dropped={wd_all}")

    log("=== SPLIT (union, location-disjoint 80/20; cold verbatim-v1, warm+dramatic fresh) ===")
    log(f"seed={SEED} val_frac={VAL_FRAC}")
    log(f"locations: train={manifest['n_locations_train']} val={manifest['n_locations_val']} "
        f"total={manifest['n_locations_total']}")
    log(f"queries:   train={manifest['n_queries_train']} val={manifest['n_queries_val']}")
    for b in ("coldstart", "warmstart", "dramatic"):
        tr, va = manifest[f"{b}_train"], manifest[f"{b}_val"]
        log(f"  {b:9s} train/val queries: {tr['n_queries']}/{va['n_queries']}  "
            f"locations: {tr['n_locations']}/{va['n_locations']}  val types: {va['type_breakdown']}")
    log(f"ZERO-OVERLAP PROOF: train-cap-val locations = {manifest['location_overlap_count']} (must be 0)")
    log(f"DRAMATIC-DISJOINT PROOF: dramatic-cap-old locations = {manifest['dramatic_old_overlap_count']} (must be 0)")
    log(f"type breakdown val: {manifest['type_breakdown_val']}")

    json.dump(manifest, open(os.path.join(OUT_DIR, "split_manifest.json"), "w"), indent=2)

    # ---- loaders ----
    nw = 4 if device.type == "cuda" else 0
    train_ld = DataLoader(D.QueryDataset(train_q, train=True), batch_size=BATCH_QUERIES,
                          shuffle=True, collate_fn=D.collate_queries, num_workers=nw, drop_last=False)
    val_ld = DataLoader(D.QueryDataset(val_q, train=False), batch_size=BATCH_QUERIES,
                        shuffle=False, collate_fn=D.collate_queries, num_workers=nw)

    # ---- model + optim (identical to v2) ----
    model = build_model().to(device)
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "classifier" in name else backbone_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": BACKBONE_LR},
         {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )

    config = {
        "task": "palette-preference scorer v3 (single-tower utility, margin-ranking)",
        "version": "v3",
        "data": "union coldstart_v2 + warmstart_v1 + prefv2_dramatic_v1 (provenance-blind)",
        "excluded_queries": list(D.EXCLUDED_QUERIES),
        "split_design": "cold verbatim-v1; warm+dramatic fresh stratified 80/20 seed 0",
        "frozen_from_v2": "model, input/geometry, aug, loss, margin, LRs, wd, batch, patience, pair-inclusion",
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "eval_pair_types": "ALL (good_vs_okay held-out, still scored)",
        "model": MODEL, "input_size": D.INPUT_SIZE, "geometry": "squash",
        "aug": TV2.AUG, "margin": MARGIN, "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY, "batch_queries": BATCH_QUERIES, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "val_frac": VAL_FRAC, "seed": SEED,
        "mean": list(D.IMAGENET_MEAN), "std": list(D.IMAGENET_STD), "interpolation": "bicubic",
        "device": device.type, "pass1_only": True,
        "human_consistency_ceiling_dramatic": CEILING_DRAMATIC,
        "human_consistency_ceiling_old": CEILING_OLD,
    }
    json.dump(config, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)

    # ---- train loop (identical mechanics to v2) ----
    log("=== TRAIN ===")
    best_val, best_epoch, no_improve = -1.0, -1, 0
    epoch_log = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        ep_loss = ep_correct = ep_pairs = nb = 0
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
        val_acc, val_pt = eval_pair_accuracy(model, val_ld, device)

        def _f(x):
            return f"{x:.3f}" if x is not None else "n/a"
        vp = {k: (v[0] / v[1] if v[1] else None) for k, v in val_pt.items()}
        epoch_log.append({"epoch": epoch, "train_loss": ep_loss / max(nb, 1),
                          "train_pair_acc": train_acc, "val_pair_acc": val_acc, "val_per_type": vp})
        log(f"epoch {epoch:02d}  loss={ep_loss/max(nb,1):.4f}  train_acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  [gvb={_f(vp['good_vs_bad'])} gvo={_f(vp['good_vs_okay'])} "
            f"ovb={_f(vp['okay_vs_bad'])}]")

        torch.save({"state_dict": model.state_dict(), "config": config},
                   os.path.join(OUT_DIR, "model_last.pt"))
        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch},
                       os.path.join(OUT_DIR, "model_best.pt"))
            log(f"  new best val_pair_acc={best_val:.4f} @ epoch {epoch} -> model_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log(f"early stop: no val improvement for {PATIENCE} epochs")
                break

    # ---- reload best, stratified eval ----
    ck = torch.load(os.path.join(OUT_DIR, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])

    val_recs = collect_pair_records(model, val_q, device, nw)
    dram_val_q = [q for q in val_q if q.batch == D.DRAMATIC.name]
    old_val_q = [q for q in val_q if q.batch != D.DRAMATIC.name]
    dram_recs = [r for r in val_recs if r["batch"] == D.DRAMATIC.name]
    old_recs = [r for r in val_recs if r["batch"] != D.DRAMATIC.name]

    # regression guard: deployed v2 on the IDENTICAL old-slice
    v2_model = build_model().to(device)
    v2_ck = torch.load(os.path.join(V2_DIR, "model_best.pt"), map_location=device, weights_only=False)
    v2_model.load_state_dict(v2_ck["state_dict"])
    v2_old_recs = collect_pair_records(v2_model, old_val_q, device, nw)

    # ---- assemble stratified numbers ----
    strat = {
        "overall_union": _acc(val_recs),
        "held_out_good_vs_okay_union": _acc([r for r in val_recs if r["pair_type"] == "good_vs_okay"]),
        "held_out_good_vs_okay_dramatic": _acc([r for r in dram_recs if r["pair_type"] == "good_vs_okay"]),
        "by_query_type_dramatic": _slice(dram_recs, lambda r: r["query_type"]),
        "by_src_pair": _slice(val_recs, lambda r: r["src_pair"] if r["src_pair"] != "other" else None),
        # dramatic axis: accuracy of within-dramatic pairs attributed to the WINNER's tag
        "by_skeleton_hi": _slice([r for r in dram_recs if r["src_pair"] == "dramatic_vs_dramatic"],
                                 lambda r: r["hi_skel"]),
        "by_architecture_hi": _slice([r for r in dram_recs if r["src_pair"] == "dramatic_vs_dramatic"],
                                     lambda r: r["hi_arch"]),
        "dramatic_overall": _acc(dram_recs),
        "old_overall_v3": _acc(old_recs),
    }
    # per-old-query-type regression, both models on the matched old slice
    def by_old_qtype(recs):
        return _slice(recs, lambda r: r["query_type"])
    reg = {
        "v3": {"overall": _acc(old_recs), "by_query_type": by_old_qtype(old_recs)},
        "v2_deployed": {"overall": _acc(v2_old_recs), "by_query_type": by_old_qtype(v2_old_recs)},
    }

    # ---- persist metrics ----
    def jsonable(slice_dict):
        return {k: {"acc": a, "n": n} for k, (a, n) in slice_dict.items()}
    metrics = {
        "headline_val_pair_acc": best_val,
        "best_epoch": best_epoch,
        "ceiling_dramatic": CEILING_DRAMATIC,
        "ceiling_old": CEILING_OLD,
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "stratified": {
            "overall_union": {"acc": strat["overall_union"][0], "n": strat["overall_union"][1]},
            "dramatic_overall": {"acc": strat["dramatic_overall"][0], "n": strat["dramatic_overall"][1]},
            "old_overall_v3": {"acc": strat["old_overall_v3"][0], "n": strat["old_overall_v3"][1]},
            "held_out_good_vs_okay_union": {"acc": strat["held_out_good_vs_okay_union"][0],
                                            "n": strat["held_out_good_vs_okay_union"][1]},
            "held_out_good_vs_okay_dramatic": {"acc": strat["held_out_good_vs_okay_dramatic"][0],
                                               "n": strat["held_out_good_vs_okay_dramatic"][1]},
            "by_query_type_dramatic": jsonable(strat["by_query_type_dramatic"]),
            "by_src_pair": jsonable(strat["by_src_pair"]),
            "by_skeleton_hi": jsonable(strat["by_skeleton_hi"]),
            "by_architecture_hi": jsonable(strat["by_architecture_hi"]),
        },
        "regression_guard_old_slice": {
            "v3": {"overall": {"acc": reg["v3"]["overall"][0], "n": reg["v3"]["overall"][1]},
                   "by_query_type": jsonable(reg["v3"]["by_query_type"])},
            "v2_deployed": {"overall": {"acc": reg["v2_deployed"]["overall"][0], "n": reg["v2_deployed"]["overall"][1]},
                            "by_query_type": jsonable(reg["v2_deployed"]["by_query_type"])},
            "note": "same matched old-corpus val slice (cold verbatim-v1 + warm seed-0); v3 vs deployed v2",
        },
        "split": {k: manifest[k] for k in
                  ("n_locations_train", "n_locations_val", "n_queries_train", "n_queries_val",
                   "location_overlap_count", "dramatic_old_overlap_count",
                   "type_breakdown_val")},
        "dramatic_val_pair_counts_by_type": {
            qt: _acc([r for r in dram_recs if r["query_type"] == qt])[1]
            for qt in ("within_dramatic", "cross_source", "param_variation")
        },
        "epoch_log": epoch_log,
    }
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)

    # ---- report tables ----
    log("")
    log(f"=== (1) HEADLINE  best_epoch={best_epoch}  union val overall {_fmt(strat['overall_union'])} ===")
    log("")
    log("=== (2) DRAMATIC-BATCH STRATIFIED (ceiling 0.956) ===")
    log(f"  dramatic-batch overall           {_fmt(strat['dramatic_overall'], CEILING_DRAMATIC)}")
    log("  -- by query_type --")
    for qt in ("within_dramatic", "cross_source", "param_variation"):
        v = strat["by_query_type_dramatic"].get(qt, (None, 0))
        tag = "  <-- LOAD-BEARING" if qt == "within_dramatic" else ""
        log(f"    {qt:16s} {_fmt(v, CEILING_DRAMATIC)}{tag}")
    log("  -- by palette-source pair composition (union val) --")
    for sp in ("dramatic_vs_dramatic", "pool_vs_pool", "dramatic_vs_pool"):
        log(f"    {sp:20s} {_fmt(strat['by_src_pair'].get(sp, (None, 0)), CEILING_DRAMATIC)}")
    log("  -- by dramatic skeleton (winner tag; dramatic-vs-dramatic pairs) --")
    for k, v in strat["by_skeleton_hi"].items():
        log(f"    {str(k):16s} {_fmt(v)}")
    log("  -- by dramatic architecture (winner tag; dramatic-vs-dramatic pairs) --")
    for k, v in strat["by_architecture_hi"].items():
        log(f"    {str(k):22s} {_fmt(v)}")
    log(f"  held-out good_vs_okay (dramatic) {_fmt(strat['held_out_good_vs_okay_dramatic'], CEILING_DRAMATIC)}")
    log(f"  held-out good_vs_okay (union)    {_fmt(strat['held_out_good_vs_okay_union'])}")

    log("")
    log("=== (3) REGRESSION GUARD -- old-corpus-only val slice, v3 vs deployed v2 (ceiling 0.943) ===")
    log(f"  {'slice':20s} {'v3':>26s}   {'v2 (deployed)':>26s}")
    def row(name, v3s, v2s):
        log(f"  {name:20s} {_fmt(v3s):>26s}   {_fmt(v2s):>26s}")
    row("overall", reg["v3"]["overall"], reg["v2_deployed"]["overall"])
    all_qt = sorted(set(reg["v3"]["by_query_type"]) | set(reg["v2_deployed"]["by_query_type"]))
    for qt in all_qt:
        row(qt, reg["v3"]["by_query_type"].get(qt, (None, 0)),
            reg["v2_deployed"]["by_query_type"].get(qt, (None, 0)))

    log("")
    log(f"artifacts staged in {OUT_DIR}")
    log("ACTIVE pointer NOT flipped (data.ACTIVE_SCORER_DIR still -> v2).")
    log("To promote after review, flip the single-source pointer in tools/queries/scorer/data.py:")
    log('  ACTIVE_SCORER_DIR = os.path.join(REPO, "data", "queries", "scorer", "v3")')


if __name__ == "__main__":
    main()
