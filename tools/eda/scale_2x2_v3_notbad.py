# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
"""Phase 2 — v3 not-bad read over the scale-controlled 2x2 batch.

v3 is a RANKING model (eval = AP / precision@k, no fixed operating threshold).
Its sigma(logit_0) is poorly calibrated as an absolute probability (even rev4
label-3 'good' crops average ~0.16, so a 0.5 threshold marks ~everything bad), so
the decision-capable per-cell signal is the **monotone score** mono = Sigma
sigma(logit_k) in [0,2] — v3's trustworthy not-bad ranking axis (loc AP ~0.52).

Reports per cell: mean monotone score (+95% bootstrap CI, the primary B-vs-D read)
and a predicted not-bad-RATE at a threshold tau calibrated on the rev4 labeled set
(tau = the rev4 monotone-score quantile matching rev4's true not-bad base rate).
Because the 2x2 crops are out-of-sample vs rev4 (v3 trained on rev4), absolute
rates run DEFLATED — but the deflation is uniform across all 4 cells, so the
RELATIVE cell ordering is valid. NOT-BAD axis only; good-rate waits on hand-labels.

  uv run python tools/eda/scale_2x2_v3_notbad.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)  # so `classifier` (repo-root package) imports
sys.path.insert(0, os.path.join(_ROOT, "tools", "corpus"))
import corpus_common as cc  # noqa: E402
from classifier.inference import load_scorer  # noqa: E402
from PIL import Image  # noqa: E402

BATCH_ID = "2026-06-25_scale_controlled_2x2"
CKPT = "data/classifier/v3/model_best.pt"
OUT_DIR = os.path.join(cc.ROOT, "data", "eda", "scale_2x2")
CELL_ORDER = ["A", "B", "C", "D"]
CELL_LABEL = {
    "A": "A: 8k / wide",
    "B": "B: 8k / narrow",
    "C": "C: flat / wide",
    "D": "D: flat / narrow",
}


@torch.no_grad()
def notbad_and_score(scorer, paths, batch_size=32):
    """Returns (p_notbad list, monotone score list) for ordinal v3."""
    p_nb, mono = [], []
    buf = []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(scorer.device)
        dev = scorer.device.split(":")[0]
        with torch.autocast(device_type=dev, enabled=(scorer.device != "cpu")):
            logits = scorer.model(x).float()
        p_nb.extend(torch.sigmoid(logits[:, 0]).cpu().tolist())          # P(label>=2)
        mono.extend(torch.sigmoid(logits).sum(dim=1).cpu().tolist())     # Sigma sigma
        buf.clear()

    for p in paths:
        with Image.open(p) as im:
            im.load()
            buf.append(scorer.transform(im.convert("RGB")))
        if len(buf) == batch_size:
            flush()
    flush()
    return p_nb, mono


REV4_BATCH = "2026-06-24_guided_descend_rev4"


def calibrate_tau(scorer):
    """tau = the rev4 monotone-score quantile matching rev4's true not-bad base
    rate. Scoring rev4 (v3's train distribution) is optimistic in absolute terms,
    but tau is only used to define a CONSISTENT operating point applied identically
    to every 2x2 cell (the cross-cell comparison is what matters)."""
    rows = cc.read_jsonl(os.path.join(cc.batch_dir(REV4_BATCH), "images.jsonl"))
    crops = os.path.join(cc.batch_dir(REV4_BATCH), "crops")
    labels, ids = [], []
    for r in rows:
        s = r["label"]["score"]
        if s in (1, 2, 3):
            labels.append(s)
            ids.append(r["image_id"])
    paths = [os.path.join(crops, i + ".jpg") for i in ids]
    keep = [k for k, p in enumerate(paths) if os.path.exists(p)]
    paths = [paths[k] for k in keep]
    labels = [labels[k] for k in keep]
    _, mono = notbad_and_score(scorer, paths)
    base_rate = sum(1 for s in labels if s >= 2) / len(labels)
    tau = float(_quantile(sorted(mono), 1.0 - base_rate))
    print(f"rev4 calibration: n={len(labels)} not-bad base-rate={base_rate:.3f} "
          f"-> tau(mono)={tau:.4f}")
    return tau, base_rate


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _boot_ci(vals, seed=0, n_boot=2000):
    import random
    rng = random.Random(seed)
    means = []
    nv = len(vals)
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(nv)] for _ in range(nv)) / nv
        means.append(s)
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = cc.read_jsonl(os.path.join(cc.batch_dir(BATCH_ID), "images.jsonl"))
    crops_dir = os.path.join(cc.batch_dir(BATCH_ID), "crops")

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["provenance"]["cell"]].append(r["image_id"])

    assert scorer_target_ok(), "v3 must be ordinal"
    scorer = load_scorer(os.path.join(cc.ROOT, CKPT))
    print(f"v3 loaded: target={scorer.target} device={scorer.device}")
    tau, base_rate = calibrate_tau(scorer)

    results = {}
    for cell in CELL_ORDER:
        ids = by_cell.get(cell, [])
        paths = [os.path.join(crops_dir, f"{i}.jpg") for i in ids]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        p_nb, mono = notbad_and_score(scorer, paths)
        n = len(mono)
        lo, hi = _boot_ci(mono)
        rate = sum(1 for m in mono if m >= tau) / n
        results[cell] = {
            "n": n,
            "mean_monotone": sum(mono) / n,
            "mono_ci95": [lo, hi],
            "median_monotone": _quantile(sorted(mono), 0.5),
            "notbad_rate_at_tau": rate,
            "mean_p_notbad": sum(p_nb) / n,
        }
        print(f"cell {cell} ({CELL_LABEL[cell]:16s}): n={n:4d}  "
              f"mean_mono={results[cell]['mean_monotone']:.3f} "
              f"[{lo:.3f},{hi:.3f}]  median={results[cell]['median_monotone']:.3f}  "
              f"not-bad@tau={rate:.3f}")

    json.dump({"tau": tau, "rev4_base_rate": base_rate, "cells": results},
              open(os.path.join(OUT_DIR, "v3_notbad.json"), "w"), indent=2)

    # --- two-panel figure: mean monotone (+CI) | not-bad-rate@tau ---
    cells = [c for c in CELL_ORDER if c in results]
    labels = [CELL_LABEL[c] for c in cells]
    colors = ["#3b6ea5" if "8k" in CELL_LABEL[c] else "#a55a3b" for c in cells]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    means = [results[c]["mean_monotone"] for c in cells]
    err = [[means[i] - results[c]["mono_ci95"][0] for i, c in enumerate(cells)],
           [results[c]["mono_ci95"][1] - means[i] for i, c in enumerate(cells)]]
    ax1.bar(labels, means, color=colors, yerr=err, capsize=5)
    for i, c in enumerate(cells):
        ax1.text(i, results[c]["mono_ci95"][1] + 0.01, f"{means[i]:.3f}\n(n={results[c]['n']})",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("mean v3 monotone score  (Sigma sigma, in [0,2])")
    ax1.set_title("Primary not-bad signal: mean monotone (95% CI)")
    ax1.grid(axis="y", alpha=0.3)

    rates = [results[c]["notbad_rate_at_tau"] for c in cells]
    ax2.bar(labels, rates, color=colors)
    for i, c in enumerate(cells):
        ax2.text(i, rates[i] + 0.005, f"{rates[i]:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel(f"not-bad-rate @ rev4-calibrated tau={tau:.3f}")
    ax2.set_title("Predicted not-bad-rate (deflated abs; relative valid)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Scale-controlled 2x2 — v3 not-bad read (PROVISIONAL; good-rate awaits hand-labels)")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "v3_notbad.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nfigure -> {fig_path}\njson    -> {os.path.join(OUT_DIR, 'v3_notbad.json')}")

    # --- decision-logic cribs (relative, not-bad axis only) ---
    def cmp(x, y, name):
        rx, ry = results[x], results[y]
        d = rx["mean_monotone"] - ry["mean_monotone"]
        sep = "DISTINCT" if (rx["mono_ci95"][0] > ry["mono_ci95"][1] or ry["mono_ci95"][0] > rx["mono_ci95"][1]) else "overlapping CIs"
        print(f"  {name}: mono {rx['mean_monotone']:.3f} vs {ry['mean_monotone']:.3f} "
              f"(delta {d:+.3f}, {sep})")
    print("\n=== provisional not-bad-axis comparisons (mean monotone; good-rate waits on labels) ===")
    if all(c in results for c in "BD"):
        cmp("B", "D", "B vs D (field vs flat, both narrow) [the clean spatial-selection test]")
    if all(c in results for c in "AB"):
        cmp("A", "B", "A vs B (8k wide vs narrow)")
    if all(c in results for c in "CD"):
        cmp("C", "D", "C vs D (flat wide vs narrow)")


def scorer_target_ok() -> bool:
    cfg = torch.load(os.path.join(cc.ROOT, CKPT), map_location="cpu", weights_only=False)["config"]
    return cfg.get("target") == "ordinal"


if __name__ == "__main__":
    main()
