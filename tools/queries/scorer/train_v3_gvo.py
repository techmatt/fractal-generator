"""Train pref-v3-gvo: the good_vs_okay training-inclusion A/B against pref-v3.

IDENTICAL to train_v3 in every respect -- same arch, LRs, margin, batch, patience,
seed, and the SAME location-disjoint split (partitioned VERBATIM from v3's
split_manifest.json, so val is byte-identical). The ONLY moving part: training
pairs = good_vs_bad + okay_vs_bad + good_vs_okay (v3 trains the first two and holds
good_vs_okay out). Same MARGIN (1.0) on every pair type for this primary run.

Stage-only: writes to data/queries/scorer/v3_gvo/ and does NOT touch the ACTIVE
pointer (v3 stays deployed). Step-C side-by-side eval is in compare_v3_gvo.py.

Run:
    uv run python tools/queries/scorer/train_v3_gvo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import train as TV2  # noqa: E402  (FROZEN config + mechanics, one source of truth)

# ---- config: FROZEN from v3 (== train.py). Only the training pair-set changes. ----
SEED = TV2.SEED
MARGIN = TV2.MARGIN
BACKBONE_LR = TV2.BACKBONE_LR
HEAD_LR = TV2.HEAD_LR
WEIGHT_DECAY = TV2.WEIGHT_DECAY
BATCH_QUERIES = TV2.BATCH_QUERIES
MAX_EPOCHS = TV2.MAX_EPOCHS
PATIENCE = TV2.PATIENCE
MODEL = TV2.MODEL

build_model = TV2.build_model
batch_margin_loss = TV2.batch_margin_loss
eval_pair_accuracy = TV2.eval_pair_accuracy
census = TV2.census

# THE moving part: add good_vs_okay to the trained pairs (v3 excludes it).
TRAIN_INCLUDE_PAIR_TYPES = ("good_vs_bad", "okay_vs_bad", "good_vs_okay")

OUT_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v3_gvo")
V3_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v3")
V3_MANIFEST = os.path.join(V3_DIR, "split_manifest.json")

_LOG_FH = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def partition_by_manifest(queries, manifest_path):
    """Assign whole queries to train/val by v3's train/val LOCATION sets, verbatim.
    Asserts every location is accounted for, zero overlap, and query counts match."""
    m = json.load(open(manifest_path))
    train_locs, val_locs = set(m["train_locations"]), set(m["val_locations"])
    train_q, val_q, unknown = [], [], set()
    for q in queries:
        if q.location_key in train_locs:
            train_q.append(q)
        elif q.location_key in val_locs:
            val_q.append(q)
        else:
            unknown.add(q.location_key)
    assert not unknown, f"{len(unknown)} loc(s) not in v3 manifest"
    assert not (train_locs & val_locs), "LOCATION LEAK in v3 manifest"
    assert len(val_q) == m["n_queries_val"], f"val {len(val_q)} != manifest {m['n_queries_val']}"
    assert len(train_q) == m["n_queries_train"], f"train {len(train_q)} != manifest {m['n_queries_train']}"
    return train_q, val_q, m


def main():
    global _LOG_FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(OUT_DIR, "train.log"), "w")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    queries = D.load_combined_queries()
    train_q, val_q, manifest = partition_by_manifest(queries, V3_MANIFEST)

    log("=== DATA (union; split partitioned VERBATIM from v3/split_manifest.json) ===")
    log(f"queries: train={len(train_q)} val={len(val_q)} (v3 manifest match asserted)")
    tc_tr, pc_tr, wd_tr = census(train_q)
    log(f"train tiers={tc_tr}")
    log(f"train cross-tier pairs={pc_tr}")
    log(f"TRAIN_INCLUDE_PAIR_TYPES={list(TRAIN_INCLUDE_PAIR_TYPES)}  "
        f"(v3 excludes good_vs_okay; this run ADDS it -- the only moving part)")
    trained = sum(pc_tr.get(t, 0) for t in TRAIN_INCLUDE_PAIR_TYPES)
    gvo_added = pc_tr.get("good_vs_okay", 0)
    log(f"trained pairs total={trained}  (of which good_vs_okay ADDED = {gvo_added})")

    # dump split manifest (identical to v3's, re-dumped for the run dir)
    json.dump(manifest, open(os.path.join(OUT_DIR, "split_manifest.json"), "w"), indent=2)

    nw = 4 if device.type == "cuda" else 0
    train_ld = DataLoader(D.QueryDataset(train_q, train=True), batch_size=BATCH_QUERIES,
                          shuffle=True, collate_fn=D.collate_queries, num_workers=nw, drop_last=False)
    val_ld = DataLoader(D.QueryDataset(val_q, train=False), batch_size=BATCH_QUERIES,
                        shuffle=False, collate_fn=D.collate_queries, num_workers=nw)

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
        "task": "palette-preference scorer v3-gvo (good_vs_okay INCLUDED in training)",
        "version": "v3_gvo",
        "ab_against": "v3",
        "moving_part": "training pairs += good_vs_okay (v3 holds it out); everything else frozen",
        "data": "union coldstart_v2 + warmstart_v1 + prefv2_dramatic_v1 (provenance-blind)",
        "split_design": "VERBATIM from v3/split_manifest.json (byte-identical val)",
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "eval_pair_types": "ALL",
        "model": MODEL, "margin": MARGIN, "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY, "batch_queries": BATCH_QUERIES, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "seed": SEED, "device": device.type,
    }
    json.dump(config, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)

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

    metrics = {
        "version": "v3_gvo",
        "headline_val_pair_acc": best_val,
        "best_epoch": best_epoch,
        "train_include_pair_types": list(TRAIN_INCLUDE_PAIR_TYPES),
        "epoch_log": epoch_log,
    }
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)
    log(f"best_epoch={best_epoch} best_val={best_val:.4f}")
    log(f"artifacts staged in {OUT_DIR}  (ACTIVE pointer NOT flipped -- v3 stays deployed)")
    log("run compare_v3_gvo.py for the Step-C side-by-side.")


if __name__ == "__main__":
    main()
