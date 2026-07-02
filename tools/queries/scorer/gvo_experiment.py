"""control -> gated experiment: reproduce v1, then drop good-vs-okay pairs from training.

Two full retrains in one process, sequential. Everything is FROZEN from v1 (backbone,
single-scalar head, squash 224 + ImageNet norm, geometric-only aug, margin-ranking
MARGIN=1.0, per-query-normalized batched-by-query, AdamW 1e-4/1e-3, wd 0.05,
16 queries/batch, patience 8) -- reused directly from train.py's named constants and
building blocks. The ONLY moving parts are named per run.

RUN 1 -- CONTROL: coldstart_v2 only, ALL pair types, v1 split reused verbatim. Gates
Run 2: if the refactored path doesn't reproduce v1 (within ~0.015 AND best_epoch clearly
>1), STOP -- it's a loading/pairing bug and the experiment is meaningless until fixed.

RUN 2 -- EXPERIMENT: combined coldstart_v2 + warmstart_v1, v2 split reused verbatim,
good_vs_okay pairs EXCLUDED from training (train on good_vs_bad + okay_vs_bad only).
EVALUATION always scores all three pair types -- dropping gvo from training must not
drop it from the val report (we want to see where held-out gvo ordering lands).

Writes ONLY to probe dirs; does not touch data/queries/scorer/v2/.

Run:
    uv run python tools/queries/scorer/gvo_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import train as TR  # noqa: E402
import surfacing_eval as SE  # noqa: E402

# ---- probe output dirs (never v2/) --------------------------------------
CTRL_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "control_coldstart_repro")
EXP_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "exp_no_gvo")
V1_MANIFEST = os.path.join(D.REPO, "data", "queries", "scorer", "v1", "split_manifest.json")
V2_MANIFEST = os.path.join(D.REPO, "data", "queries", "scorer", "v2", "split_manifest.json")

# ---- baselines for the report -------------------------------------------
V1 = {"overall": 0.656, "good_vs_bad": 0.773, "okay_vs_bad": 0.678, "good_vs_okay": 0.578,
      "surf_top1_good": 0.425, "surf_top1_bad": 0.075, "surf_top3": 0.850}
V2 = {"cold": {"overall": 0.588, "good_vs_bad": 0.711, "okay_vs_bad": 0.596, "good_vs_okay": 0.519},
      "warm": {"overall": 0.605, "good_vs_bad": 0.642, "okay_vs_bad": 0.721, "good_vs_okay": 0.470},
      "surf_top1_good": 0.325, "surf_top1_bad": 0.125, "surf_top3": 0.825}

GATE_TOL = 0.015

# Run-2 training pair types: drop good_vs_okay.
EXP_INCLUDE_TYPES = {"good_vs_bad", "okay_vs_bad"}

PAIR_ORDER = ["overall", "good_vs_bad", "okay_vs_bad", "good_vs_okay"]

_LOG_FH = None


def log(msg: str = ""):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}" if msg else ""
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def f3(x):
    return f"{x:.3f}" if x is not None else "n/a"


def acc_of(pt, k):
    """pt: {name:[correct,total]} from eval_pair_accuracy. -> (acc_or_None, n)."""
    c, n = pt[k]
    return (c / n if n else None), n


# ---- split reuse (verbatim from a prior manifest) -----------------------
def partition_by_manifest(queries, manifest_path, tag):
    """Assign whole queries to train/val by a prior manifest's train/val LOCATION sets
    (verbatim reuse -- no split recompute). Asserts every location is accounted for and
    zero train/val overlap."""
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
    assert not unknown, f"[{tag}] {len(unknown)} loc(s) not in manifest {manifest_path}"
    overlap = train_locs & val_locs
    assert not overlap, f"[{tag}] LOCATION LEAK: {overlap}"
    # cross-check query counts if the manifest recorded them
    if "n_queries_val" in m:
        assert len(val_q) == m["n_queries_val"], \
            f"[{tag}] val count {len(val_q)} != manifest {m['n_queries_val']}"
    if "n_queries_train" in m:
        assert len(train_q) == m["n_queries_train"], \
            f"[{tag}] train count {len(train_q)} != manifest {m['n_queries_train']}"
    return train_q, val_q, m


# ---- training (parametrized; all v1 hyperparams frozen from train.py) ----
def run_training(name, train_q, val_q, include_types, out_dir, device):
    """Full retrain from scratch through the refactored path. include_types filters
    TRAINING pairs only (None -> all three types); validation always scores all three.
    Returns (best_epoch, epoch_log, best_model, config)."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(TR.SEED)  # re-seed per run so each is a clean from-scratch init

    train_ds = D.QueryDataset(train_q, train=True)
    val_ds = D.QueryDataset(val_q, train=False)
    nw = 4 if device.type == "cuda" else 0
    train_ld = DataLoader(train_ds, batch_size=TR.BATCH_QUERIES, shuffle=True,
                          collate_fn=D.collate_queries, num_workers=nw, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=TR.BATCH_QUERIES, shuffle=False,
                        collate_fn=D.collate_queries, num_workers=nw)

    model = TR.build_model().to(device)
    head_params, backbone_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if "classifier" in pname else backbone_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": TR.BACKBONE_LR},
         {"params": head_params, "lr": TR.HEAD_LR}],
        weight_decay=TR.WEIGHT_DECAY,
    )

    config = {
        "run": name, "frozen_from_v1": TR.AUG, "model": TR.MODEL, "geometry": TR.GEOMETRY,
        "margin": TR.MARGIN, "backbone_lr": TR.BACKBONE_LR, "head_lr": TR.HEAD_LR,
        "weight_decay": TR.WEIGHT_DECAY, "batch_queries": TR.BATCH_QUERIES,
        "max_epochs": TR.MAX_EPOCHS, "patience": TR.PATIENCE, "seed": TR.SEED,
        "train_include_pair_types": sorted(include_types) if include_types else "ALL",
        "eval_pair_types": "ALL (always -- held-out types still scored)",
        "n_queries_train": len(train_q), "n_queries_val": len(val_q),
    }
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    # runtime estimate (one timed step incl warmup/JIT)
    model.train()
    t0 = time.time()
    imgs, ranks, _ = next(iter(train_ld))
    B, Cn = imgs.shape[0], imgs.shape[1]
    scores = model(imgs.view(B * Cn, *imgs.shape[2:]).to(device)).view(B, Cn)
    loss, _, _ = TR.batch_margin_loss(scores, ranks, include_types)
    opt.zero_grad()
    if loss is not None:
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_s = time.time() - t0
    spe = (len(train_ds) + TR.BATCH_QUERIES - 1) // TR.BATCH_QUERIES
    log(f"[{name}] first step {step_s:.2f}s; steps/epoch={spe}; "
        f"rough <={step_s * spe:.0f}s/epoch, <={step_s * spe * TR.MAX_EPOCHS / 60:.0f} min "
        f"for {TR.MAX_EPOCHS} epochs (patience {TR.PATIENCE} usually ends sooner)")

    best_val, best_epoch, no_improve = -1.0, -1, 0
    epoch_log = []
    for epoch in range(1, TR.MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        ep_correct = ep_pairs = nb = 0
        for imgs, ranks, _ in train_ld:
            B, Cn = imgs.shape[0], imgs.shape[1]
            scores = model(imgs.view(B * Cn, *imgs.shape[2:]).to(device)).view(B, Cn)
            loss, nc, npr = TR.batch_margin_loss(scores, ranks, include_types)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
            ep_correct += nc; ep_pairs += npr
        train_acc = ep_correct / ep_pairs if ep_pairs else 0.0  # over TRAINED pair types

        val_acc, val_pt = TR.eval_pair_accuracy(model, val_ld, device)  # ALL types
        vp = {k: (v[0] / v[1] if v[1] else None) for k, v in val_pt.items()}
        epoch_log.append({"epoch": epoch, "train_loss": ep_loss / max(nb, 1),
                          "train_pair_acc": train_acc, "val_pair_acc": val_acc,
                          "val_per_type": vp})
        log(f"[{name}] epoch {epoch:02d}  loss={ep_loss / max(nb, 1):.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
            f"[gvb={f3(vp['good_vs_bad'])} gvo={f3(vp['good_vs_okay'])} ovb={f3(vp['okay_vs_bad'])}]")

        torch.save({"state_dict": model.state_dict(), "config": config},
                   os.path.join(out_dir, "model_last.pt"))
        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch},
                       os.path.join(out_dir, "model_best.pt"))
            log(f"[{name}]   new best val_pair_acc={best_val:.4f} @ epoch {epoch}")
        else:
            no_improve += 1
            if no_improve >= TR.PATIENCE:
                log(f"[{name}] early stop: no val improvement for {TR.PATIENCE} epochs")
                break

    ck = torch.load(os.path.join(out_dir, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    return best_epoch, epoch_log, model, config


# ---- census helpers ------------------------------------------------------
def batch_census(qlist, name):
    b = [q for q in qlist if q.batch == name]
    tc, pc, wd = TR.census(b)
    return {"n_queries": len(b), "n_locations": len({q.location_key for q in b}),
            "tier_counts": tc, "cross_tier_pairs": pc, "within_tier_dropped": wd}


# =========================== MAIN ===========================
def main():
    global _LOG_FH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CTRL_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(CTRL_DIR, "experiment.log"), "w")

    log("=================================================================")
    log("CONTROL + GATED EXPERIMENT (drop good-vs-okay pairs from training)")
    log(f"device={device.type}  seed={TR.SEED}  model={TR.MODEL}")
    log(f"frozen: margin={TR.MARGIN} lr={TR.BACKBONE_LR}/{TR.HEAD_LR} wd={TR.WEIGHT_DECAY} "
        f"batch={TR.BATCH_QUERIES} patience={TR.PATIENCE} aug=geometric-only")
    log("=================================================================")

    # ================= RUN 1 -- CONTROL =================
    log("")
    log("########## RUN 1 -- CONTROL: reproduce v1 (coldstart-only, new code path) ##########")
    cold = D.load_queries()  # coldstart_v2 pass-1, no exclusions (matches v1)
    ctrl_train, ctrl_val, _ = partition_by_manifest(cold, V1_MANIFEST, "control")
    tc, pc, wd = TR.census(ctrl_train)
    tcv, pcv, wdv = TR.census(ctrl_val)
    log(f"[control] coldstart pass-1 queries: total={len(cold)}  "
        f"train={len(ctrl_train)} val={len(ctrl_val)} (v1 split verbatim)")
    log(f"[control] train tiers={tc} pairs={pc} within-dropped={wd}")
    log(f"[control] val   tiers={tcv} pairs={pcv} within-dropped={wdv}")
    log("[control] ALL three pair types included in training (this IS v1's recipe)")

    best_epoch1, epoch_log1, model1, cfg1 = run_training(
        "control", ctrl_train, ctrl_val, None, CTRL_DIR, device)

    # gate eval: coldstart-val pair-direction, all types
    nw = 4 if device.type == "cuda" else 0
    cval_acc, cval_pt = TR.eval_subset(model1, ctrl_val, device, nw)
    got = {"overall": cval_acc,
           "good_vs_bad": acc_of(cval_pt, "good_vs_bad")[0],
           "okay_vs_bad": acc_of(cval_pt, "okay_vs_bad")[0],
           "good_vs_okay": acc_of(cval_pt, "good_vs_okay")[0]}

    log("")
    log("=== CONTROL GATE CHECK (coldstart-val vs v1, tol +-%.3f) ===" % GATE_TOL)
    cell_ok = {}
    for k in PAIR_ORDER:
        g, base = got[k], V1[k]
        dv = (g - base) if g is not None else None
        ok = (g is not None) and abs(dv) <= GATE_TOL
        cell_ok[k] = ok
        log(f"  {k:<13} {f3(g)}  (v1 {base:.3f}, d{dv:+.3f})  {'OK' if ok else 'MISS'}")
    cells_pass = all(cell_ok.values())
    epoch_pass = best_epoch1 > 1
    log(f"  best_epoch={best_epoch1}  (v1 was 17; require >1: {'OK' if epoch_pass else 'MISS'})")
    gate_pass = cells_pass and epoch_pass

    # persist control metrics
    ctrl_metrics = {
        "run": "control_coldstart_repro", "gate_pass": gate_pass,
        "best_epoch": best_epoch1, "v1_baseline": V1, "gate_tol": GATE_TOL,
        "coldstart_val": {"overall": cval_acc,
                          "per_type": {k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]}
                                       for k, v in cval_pt.items()}},
        "cells_within_tol": cell_ok, "epoch_gt_1": epoch_pass,
        "epoch_log": epoch_log1,
    }
    json.dump(ctrl_metrics, open(os.path.join(CTRL_DIR, "metrics.json"), "w"), indent=2)

    log("")
    if not gate_pass:
        log("################################################################")
        log("##  CONTROL GATE: FAIL  -- NOT running Run 2.                  ##")
        log("##  Divergence above (cells and/or best_epoch). This is a code ##")
        log("##  loading/pairing bug to diagnose first; the experiment is   ##")
        log("##  meaningless until the control reproduces v1.               ##")
        log("################################################################")
        log(f"artifacts: {CTRL_DIR}")
        return
    log("################################################################")
    log("##  CONTROL GATE: PASS  -- refactored path reproduces v1.      ##")
    log("##  Proceeding to Run 2 (drop good-vs-okay from training).     ##")
    log("################################################################")

    # ================= RUN 2 -- EXPERIMENT =================
    log("")
    log("########## RUN 2 -- EXPERIMENT: combined, exclude good_vs_okay from TRAINING ##########")
    comb = D.load_combined_queries()  # coldstart_v2 + warmstart_v1, EXCLUDED_QUERIES dropped
    exp_train, exp_val, _ = partition_by_manifest(comb, V2_MANIFEST, "exp_no_gvo")

    # census: per batch, cross-tier pairs (gvo present in data but DROPPED from training)
    log(f"[exp] combined pass-1 queries: total={len(comb)} (excluded {D.EXCLUDED_QUERIES})  "
        f"train={len(exp_train)} val={len(exp_val)} (v2 split verbatim)")
    census_out = {}
    for label, name in (("coldstart_v2", D.COLDSTART.name), ("warmstart_v1", D.WARMSTART.name)):
        bc_tr = batch_census(exp_train, name)
        bc_va = batch_census(exp_val, name)
        census_out[label] = {"train": bc_tr, "val": bc_va}
        remain = {k: v for k, v in bc_tr["cross_tier_pairs"].items() if k in EXP_INCLUDE_TYPES}
        dropped = bc_tr["cross_tier_pairs"].get("good_vs_okay", 0)
        log(f"[exp][{label}] train pairs TRAINED (gvb+ovb)={remain} "
            f"| good_vs_okay DROPPED from training (n={dropped}) | within-dropped={bc_tr['within_tier_dropped']}")
        log(f"[exp][{label}] val queries={bc_va['n_queries']} locations={bc_va['n_locations']} "
            f"(all 3 pair types scored: {bc_va['cross_tier_pairs']})")
    tl = {q.location_key for q in exp_train}
    vl = {q.location_key for q in exp_val}
    log(f"[exp] ZERO-OVERLAP PROOF: train-cap-val locations = {len(tl & vl)} (must be 0)")

    best_epoch2, epoch_log2, model2, cfg2 = run_training(
        "exp_no_gvo", exp_train, exp_val, EXP_INCLUDE_TYPES, EXP_DIR, device)

    # ---- final eval (all three pair types, always) ----
    exp_val_ld = DataLoader(D.QueryDataset(exp_val, train=False), batch_size=TR.BATCH_QUERIES,
                            shuffle=False, collate_fn=D.collate_queries, num_workers=nw)
    comb_acc, comb_pt = TR.eval_pair_accuracy(model2, exp_val_ld, device)
    cold_val_q = [q for q in exp_val if q.batch == D.COLDSTART.name]
    warm_val_q = [q for q in exp_val if q.batch == D.WARMSTART.name]
    cold_acc, cold_pt = TR.eval_subset(model2, cold_val_q, device, nw)
    warm_acc, warm_pt = TR.eval_subset(model2, warm_val_q, device, nw)
    train_eval_ld = DataLoader(D.QueryDataset(exp_train, train=False), batch_size=TR.BATCH_QUERIES,
                               shuffle=False, collate_fn=D.collate_queries, num_workers=nw)
    train_acc2, train_pt2 = TR.eval_pair_accuracy(model2, train_eval_ld, device)

    surf_cold = SE.compute_metrics(SE.score_queries(model2, cold_val_q, device)) if cold_val_q else None
    surf_warm = SE.compute_metrics(SE.score_queries(model2, warm_val_q, device)) if warm_val_q else None

    # ---- persist run-2 metrics ----
    def pt_json(pt):
        return {k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]} for k, v in pt.items()}

    exp_metrics = {
        "run": "exp_no_gvo",
        "moving_part": "combined coldstart_v2+warmstart_v1; good_vs_okay EXCLUDED from training",
        "eval_note": "all three pair types scored; good_vs_okay is HELD-OUT (trained on neither batch)",
        "best_epoch": best_epoch2,
        "control_best_epoch": best_epoch1,
        "v1_baseline": V1, "v2_baseline": V2,
        "pair_direction": {
            "combined_val": {"overall": comb_acc, "per_type": pt_json(comb_pt)},
            "coldstart_val": {"overall": cold_acc, "per_type": pt_json(cold_pt)},
            "warmstart_val": {"overall": warm_acc, "per_type": pt_json(warm_pt)},
            "train_overall": train_acc2, "train_per_type": pt_json(train_pt2),
        },
        "surfacing": {"coldstart_val": surf_cold, "warmstart_val": surf_warm},
        "census": census_out,
        "excluded_queries": list(D.EXCLUDED_QUERIES),
        "epoch_log": epoch_log2,
    }
    json.dump(exp_metrics, open(os.path.join(EXP_DIR, "metrics.json"), "w"), indent=2)

    # ============== RUN 2 REPORT ==============
    log("")
    log("=================== RUN 2 REPORT ===================")
    log("")
    log("=== (A) EPOCH-1 COLLAPSE QUESTION ===")
    log(f"  exp best_epoch = {best_epoch2}   (v2 collapsed at best_epoch=1; control reached {best_epoch1})")
    log(f"  {'PAST epoch 1 -- dropping gvo changed the collapse' if best_epoch2 > 1 else 'STILL best-epoch-1 -- gvo drop did NOT prevent collapse'}")
    log("  train/val overall-acc curve by epoch (train_acc = over TRAINED pairs gvb+ovb):")
    log("    epoch | train_loss | train_acc | val_acc | val[gvb/gvo/ovb]")
    for r in epoch_log2:
        vp = r["val_per_type"]
        log(f"    {r['epoch']:>5} | {r['train_loss']:.4f}     | {r['train_pair_acc']:.4f}    "
            f"| {r['val_pair_acc']:.4f}  | {f3(vp['good_vs_bad'])}/{f3(vp['good_vs_okay'])}/{f3(vp['okay_vs_bad'])}")

    log("")
    log("=== (B) PAIR-DIRECTION ACCURACY (all three types; gvo = HELD-OUT) ===")

    def pair_block(title, acc, pt, base_v1, base_v2):
        log(f"--- {title} ---")
        rows = [("overall", acc, None)] + [(k, acc_of(pt, k)[0], acc_of(pt, k)[1]) for k in
                                           ("good_vs_bad", "okay_vs_bad", "good_vs_okay")]
        for k, g, n in rows:
            b1 = base_v1.get(k) if base_v1 else None
            b2 = base_v2.get(k) if base_v2 else None
            seg = f"  {f3(g)}"
            if b1 is not None:
                seg += f"  vs v1 {b1:.3f} (d{(g - b1):+.3f})" if g is not None else f"  vs v1 {b1:.3f}"
            if b2 is not None:
                seg += f"  vs v2 {b2:.3f} (d{(g - b2):+.3f})" if g is not None else f"  vs v2 {b2:.3f}"
            if n is not None:
                seg += f"  (n={n})"
            tag = "  [HELD-OUT: trained on neither batch]" if k == "good_vs_okay" else ""
            log(f"  {k:<13}{seg}{tag}")

    pair_block("coldstart-val", cold_acc, cold_pt, V1, V2["cold"])
    pair_block("warmstart-val", warm_acc, warm_pt, None, V2["warm"])
    log(f"--- combined-val overall {f3(comb_acc)} | train overall {f3(train_acc2)} (overfit check) ---")

    log("")
    log("=== (C) SURFACING / SELECTION (within-query, per batch, never pooled) ===")

    def surf_block(title, m, base):
        if m is None:
            log(f"--- {title}: no val queries ---")
            return
        p, s = m["primary"], m["secondary"]
        mr = s["mean_norm_rank_by_tier"]
        log(f"--- {title} ({m['n_val_queries']} queries) ---")
        if base:
            log(f"  top1-good {f3(p['top1_good_rate'])}  (v1 {base['v1_g']:.3f} / v2 {base['v2_g']:.3f})")
            log(f"  top1-bad  {f3(p['top1_bad_rate'])}  (v1 {base['v1_b']:.3f} / v2 {base['v2_b']:.3f})")
            log(f"  top3-contains-good {f3(s['top3_contains_good_rate'])}  (v1 {base['v1_t3']:.3f} / v2 {base['v2_t3']:.3f})")
        else:
            log(f"  top1-good {f3(p['top1_good_rate'])}   top1-bad {f3(p['top1_bad_rate'])}   "
                f"top3-contains-good {f3(s['top3_contains_good_rate'])}")
        log(f"  top2-contains-good {f3(s['top2_contains_good_rate'])}   "
            f"good-recall@3 {f3(s['good_recall_in_top3'])} ({s['total_good_candidates']} goods)")
        log(f"  mean norm rank good/okay/bad (0=top) "
            f"{f3(mr['good'])}/{f3(mr['okay'])}/{f3(mr['bad'])}")

    surf_block("coldstart-val", surf_cold,
               {"v1_g": V1["surf_top1_good"], "v2_g": V2["surf_top1_good"],
                "v1_b": V1["surf_top1_bad"], "v2_b": V2["surf_top1_bad"],
                "v1_t3": V1["surf_top3"], "v2_t3": V2["surf_top3"]})
    surf_block("warmstart-val [v1-concentrated, NOT coldstart-comparable]", surf_warm, None)

    log("")
    log("=== (D) CENSUS (training-pair counts after dropping gvo) ===")
    for label in ("coldstart_v2", "warmstart_v1"):
        tr = census_out[label]["train"]
        va = census_out[label]["val"]
        log(f"--- {label} ---")
        log(f"  train: queries={tr['n_queries']} locations={tr['n_locations']} tiers={tr['tier_counts']}")
        log(f"    cross-tier pairs (all)={tr['cross_tier_pairs']}  within-dropped={tr['within_tier_dropped']}")
        remain = {k: tr['cross_tier_pairs'].get(k, 0) for k in ("good_vs_bad", "okay_vs_bad")}
        log(f"    TRAINED pairs (gvb+ovb only)={remain}  | good_vs_okay DROPPED={tr['cross_tier_pairs'].get('good_vs_okay', 0)}")
        log(f"  val:   queries={va['n_queries']} locations={va['n_locations']} "
            f"cross-tier pairs (all scored)={va['cross_tier_pairs']}")
    log(f"  ZERO-OVERLAP: train-cap-val locations = {len(tl & vl)}")

    log("")
    log(f"artifacts: control -> {CTRL_DIR}")
    log(f"           experiment -> {EXP_DIR}")


if __name__ == "__main__":
    main()
