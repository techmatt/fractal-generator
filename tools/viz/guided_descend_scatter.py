#!/usr/bin/env python
"""guided-descend rev4 attribution scatter.

c-plane scatter of a run's candidate centers, faceted by (a) root sampler
(8k vs flat) and (b) branch type (foci/density/random/root*), with a flat-`generate`
baseline run's keepers overlaid on the same axes. The picture that shows whether the
dendrite-ridge clustering broke up and which source contributes what.

Diagnosis-only: no scoring, no quality claims. Run via `uv run python`.

Usage:
  uv run python tools/viz/guided_descend_scatter.py \
      --pool data/guided_descend/run4/pool.jsonl \
      --baseline data/generated/loose0/locations.jsonl \
      --out data/guided_descend/run4/attribution_scatter.png
"""
import argparse
import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Full-set bounding box (matches root_field.rs).
RE_LO, RE_HI, IM_LO, IM_HI = -2.5, 1.0, -1.75, 1.75

ROOT_COLORS = {"8k": "#4aa3ff", "flat": "#ff8c42", "": "#888888"}
BRANCH_COLORS = {
    "foci": "#5ec07a",
    "density": "#e0b24a",
    "random": "#d05ad0",
    "root8k": "#4aa3ff",
    "rootflat": "#ff8c42",
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mandelbrot_mask(res=700, maxiter=200):
    """Coarse interior mask over the bounding box for visual orientation."""
    xs = np.linspace(RE_LO, RE_HI, res)
    ys = np.linspace(IM_HI, IM_LO, res)  # row 0 = top = max im
    cre, cim = np.meshgrid(xs, ys)
    c = cre + 1j * cim
    z = np.zeros_like(c)
    inside = np.ones(c.shape, dtype=bool)
    for _ in range(maxiter):
        z = z * z + c
        escaped = np.abs(z) > 2.0
        inside &= ~escaped
        z[escaped] = 2.0  # clamp to avoid overflow warnings
    return inside


def draw_set(ax, mask):
    ax.imshow(
        mask, extent=[RE_LO, RE_HI, IM_LO, IM_HI], origin="upper",
        cmap=matplotlib.colors.ListedColormap([(0, 0, 0, 0), (1, 1, 1, 0.06)]),
        interpolation="nearest", aspect="equal", zorder=0,
    )


def style(ax, title):
    ax.set_xlim(RE_LO, RE_HI)
    ax.set_ylim(IM_LO, IM_HI)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, color="#eee")
    ax.set_xlabel("Re(c)", fontsize=9)
    ax.set_ylabel("Im(c)", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_facecolor("#0e0f13")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leaf-only", action="store_true",
                    help="restrict to terminal (leaf) frames of each walk")
    args = ap.parse_args()

    pool = read_jsonl(args.pool)
    base = read_jsonl(args.baseline)
    if not pool:
        sys.exit("empty pool")

    if args.leaf_only:
        leaf = {}
        for r in pool:
            w = r["walk"]
            if w not in leaf or r["depth"] > leaf[w]["depth"]:
                leaf[w] = r
        pool = list(leaf.values())

    cx = np.array([r["cx"] for r in pool])
    cy = np.array([r["cy"] for r in pool])
    root_src = [r.get("root_src", "") for r in pool]
    branch = [r.get("branch", "") for r in pool]
    depth = np.array([r["depth"] for r in pool])

    bx = np.array([r["center_re"] for r in base])
    by = np.array([r["center_im"] for r in base])

    mask = mandelbrot_mask()

    plt.rcParams["figure.facecolor"] = "#0e0f13"
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 0: flat-generate baseline keepers only (reference distribution).
    ax = axes[0]
    draw_set(ax, mask)
    ax.scatter(bx, by, s=10, c="#cccccc", alpha=0.55, edgecolors="none", zorder=2)
    style(ax, f"flat-generate baseline keepers (n={len(base)})")

    # Panel 1: candidates colored by ROOT SAMPLER, baseline as faint backdrop.
    ax = axes[1]
    draw_set(ax, mask)
    ax.scatter(bx, by, s=8, c="#555555", alpha=0.35, edgecolors="none", zorder=1)
    for src in ("8k", "flat", ""):
        m = np.array([s == src for s in root_src])
        if m.any():
            label = {"8k": "8k root", "flat": "flat root", "": "(none)"}[src]
            ax.scatter(cx[m], cy[m], s=14, c=ROOT_COLORS[src], alpha=0.75,
                       edgecolors="none", zorder=3, label=f"{label} ({int(m.sum())})")
    ax.legend(fontsize=8, loc="upper left", facecolor="#12141a", labelcolor="#ddd", framealpha=0.8)
    style(ax, "run4 candidates by root sampler")

    # Panel 2: candidates colored by BRANCH TYPE.
    ax = axes[2]
    draw_set(ax, mask)
    ax.scatter(bx, by, s=8, c="#555555", alpha=0.35, edgecolors="none", zorder=1)
    order = ["foci", "density", "random", "root8k", "rootflat"]
    for b in order:
        m = np.array([x == b for x in branch])
        if m.any():
            ax.scatter(cx[m], cy[m], s=14, c=BRANCH_COLORS.get(b, "#fff"), alpha=0.75,
                       edgecolors="none", zorder=3, label=f"{b} ({int(m.sum())})")
    ax.legend(fontsize=8, loc="upper left", facecolor="#12141a", labelcolor="#ddd", framealpha=0.8)
    style(ax, "run4 candidates by branch type")

    scope = "leaf frames" if args.leaf_only else "all visited frames"
    fig.suptitle(
        f"guided-descend rev4 attribution — {len(pool)} candidates ({scope}), depth {depth.min()}–{depth.max()}"
        "   ·   diagnosis only, no quality claims",
        fontsize=12, color="#e0b24a", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=110, facecolor="#0e0f13")
    print(f"wrote {args.out}  ({len(pool)} candidates, {len(base)} baseline keepers)")


if __name__ == "__main__":
    main()
