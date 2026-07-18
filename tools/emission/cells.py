"""cells.py — joint-count cells, target measure, and deficit (pure; no torch/GPU).

A wallpaper's full descriptor is a point in the product space

    cell = (fractal_type, morph_cluster, palette_flavor, render_style)

The first two are FIXED by a location's intake; the last two are chosen at colorize
time. This module maintains, for the *gated pool*, the joint count over these cells,
a hand-editable target measure, and the resulting per-cell deficit that drives the
conditional-deficit colorizer.

Joint counts (not per-axis marginals) are the whole point: a marginal view ("plenty
of warm palettes, plenty of spirals") cannot see the hole "warm spirals plentiful,
cold spirals absent" that the joint count exposes directly.

The `TargetMeasure` is uniform over feasible cells by default, with optional
per-region weight overrides keyed on any subset of the axes (e.g. "cells whose
palette_flavor is in {k16:5, k16:6}: weight 2.0"). A cell that repeatedly fails to
fill (attempt cap reached with zero fills) leaves the support and is logged.

Everything here is pure Python + stdlib so the deficit logic is unit-testable without
loading a model or rendering a frame.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

# A cell is a 4-tuple of strings. Axis order is fixed and load-bearing (the target
# measure overrides and the report both address axes by this name order).
AXES = ("fractal_type", "morph_cluster", "palette_flavor", "render_style")
Cell = tuple  # (type, cluster, flavor, style)


# --------------------------------------------------------------------------- #
# Target measure — hand-editable config.
# --------------------------------------------------------------------------- #
DEFAULT_ATTEMPT_CAP = 6


@dataclass
class TargetMeasure:
    """Hand-editable target measure over feasible cells.

    config schema (JSON):
      {
        "mode": "uniform",                     # only uniform base supported in v1
        "attempt_cap": 6,                       # per-cell colorize attempts before eviction
        "softmax_temp": 0.35,                   # colorizer choice temperature (range-normalized)
        "weight_overrides": [                   # optional per-region multipliers
          {"match": {"palette_flavor": ["k16:5", "k16:6"]}, "weight": 2.0},
          {"match": {"fractal_type": ["mandelbrot"]},        "weight": 1.5}
        ]
      }

    A cell's base target weight = 1.0 × ∏ (override.weight for every override whose
    `match` the cell satisfies). An override matches a cell iff, for every axis it
    names, the cell's value on that axis is in the override's listed values.
    """
    mode: str = "uniform"
    attempt_cap: int = DEFAULT_ATTEMPT_CAP
    softmax_temp: float = 0.35
    weight_overrides: list = field(default_factory=list)

    @staticmethod
    def from_config(cfg: dict | None) -> "TargetMeasure":
        cfg = cfg or {}
        return TargetMeasure(
            mode=cfg.get("mode", "uniform"),
            attempt_cap=int(cfg.get("attempt_cap", DEFAULT_ATTEMPT_CAP)),
            softmax_temp=float(cfg.get("softmax_temp", 0.35)),
            weight_overrides=list(cfg.get("weight_overrides", [])),
        )

    def _axis_index(self, axis: str) -> int:
        return AXES.index(axis)

    def weight(self, cell: Cell) -> float:
        w = 1.0
        for ov in self.weight_overrides:
            match = ov.get("match", {})
            ok = True
            for axis, vals in match.items():
                if cell[self._axis_index(axis)] not in vals:
                    ok = False
                    break
            if ok:
                w *= float(ov.get("weight", 1.0))
        return w


# --------------------------------------------------------------------------- #
# Feasible-cell enumeration.
# --------------------------------------------------------------------------- #
def build_feasible_cells(observed_type_cluster: Iterable[tuple],
                         flavors: Iterable[str],
                         styles: Iterable[str]) -> list:
    """Feasible cells = (observed (type, cluster) pairs) × all flavors × all styles.

    A morph cluster that no location realizes cannot be produced, so only OBSERVED
    (type, cluster) pairs anchor the grid; palette flavor and render style are free
    choices at colorize time, so every one of them is feasible for every observed
    (type, cluster)."""
    flavors = list(flavors)
    styles = list(styles)
    cells = []
    for (t, cl) in observed_type_cluster:
        for f in flavors:
            for s in styles:
                cells.append((t, cl, f, s))
    return cells


# --------------------------------------------------------------------------- #
# Deficit model — maintained over the GATED pool.
# --------------------------------------------------------------------------- #
class DeficitModel:
    """Joint-count deficit over feasible cells for the gated pool.

    deficit(cell) = target_frac(cell) − pool_frac(cell)

    where target_frac is the target measure normalized to sum 1 over the live support,
    and pool_frac is the gated-pool joint count normalized to sum 1 (0 when empty).
    Both fill counts and attempt counts are rebuilt from the durable pool log on
    resume; nothing here is trusted from a checkpoint."""

    def __init__(self, feasible_cells: list, target: TargetMeasure):
        self.target = target
        self.support: set = set(feasible_cells)
        self.weights: dict = {c: target.weight(c) for c in feasible_cells}
        self.fill_counts: Counter = Counter()
        self.attempt_counts: Counter = Counter()
        self.capped: set = set()

    # ---- rebuild-from-log entry points (resume safety) -------------------- #
    def record_fill(self, cell: Cell):
        """A gated (floor-passing) wallpaper landed in `cell`."""
        self.fill_counts[cell] += 1

    def record_attempt(self, cell: Cell) -> bool:
        """A colorize attempt targeted `cell` (whether or not it passed the floor).
        Returns True iff this attempt tips the cell over the attempt cap with zero
        fills, evicting it from the support."""
        self.attempt_counts[cell] += 1
        if (cell in self.support and self.fill_counts[cell] == 0
                and self.attempt_counts[cell] >= self.target.attempt_cap):
            self.support.discard(cell)
            self.capped.add(cell)
            return True
        return False

    # ---- deficit -------------------------------------------------------- #
    def _target_frac(self) -> dict:
        tot = sum(self.weights[c] for c in self.support)
        if tot <= 0:
            return {c: 0.0 for c in self.support}
        return {c: self.weights[c] / tot for c in self.support}

    def _pool_total(self) -> int:
        return sum(self.fill_counts.values())

    def deficit(self, cell: Cell, target_frac: dict | None = None,
                pool_total: int | None = None) -> float:
        if cell not in self.support:
            return float("-inf")
        tf = target_frac if target_frac is not None else self._target_frac()
        pt = pool_total if pool_total is not None else self._pool_total()
        pool_frac = (self.fill_counts[cell] / pt) if pt > 0 else 0.0
        return tf.get(cell, 0.0) - pool_frac

    def feasible_options(self, ftype: str, cluster: str,
                         flavors: Iterable[str], styles: Iterable[str]) -> list:
        """The (flavor, style) options still in support for a fixed (type, cluster)."""
        opts = []
        for f in flavors:
            for s in styles:
                if (ftype, cluster, f, s) in self.support:
                    opts.append((f, s))
        return opts

    # ---- diagnostics ---------------------------------------------------- #
    def occupancy(self) -> dict:
        """Report snapshot: how many feasible cells are populated / capped / empty."""
        feasible = len(self.support) + len(self.capped)
        populated = sum(1 for c, n in self.fill_counts.items() if n > 0)
        return {
            "feasible_cells": feasible,
            "in_support": len(self.support),
            "capped": len(self.capped),
            "populated_cells": populated,
            "empty_in_support": len(self.support) - sum(
                1 for c in self.support if self.fill_counts[c] > 0),
            "pool_total": self._pool_total(),
        }


# --------------------------------------------------------------------------- #
# Conditional-deficit colorizer choice (softmax over per-option deficit).
# --------------------------------------------------------------------------- #
def range_normalized_softmax(deficits: list, temp: float) -> list:
    """Softmax over deficits after normalizing them to [0,1] by their own range, so the
    distribution is scale-free: the best option maps to 1, the worst to 0, ties to a
    flat (uniform) distribution. `temp` then controls how sharply the best option is
    preferred (small → argmax-like, large → uniform)."""
    if not deficits:
        return []
    lo, hi = min(deficits), max(deficits)
    span = hi - lo
    if span <= 1e-12:                       # all equal → uniform
        return [1.0 / len(deficits)] * len(deficits)
    norm = [(d - lo) / span for d in deficits]
    t = max(temp, 1e-6)
    exps = [math.exp(v / t) for v in norm]
    z = sum(exps)
    return [e / z for e in exps]


def choose_option(model: DeficitModel, ftype: str, cluster: str,
                  flavors: Iterable[str], styles: Iterable[str],
                  rng) -> tuple | None:
    """Pick a (flavor, style) for a location whose (type, cluster) is fixed, by
    softmax over the per-option joint deficit (softmax tie-break, not strict argmax).
    Returns (flavor, style, deficit, n_options, probs) or None if nothing feasible."""
    opts = model.feasible_options(ftype, cluster, flavors, styles)
    if not opts:
        return None
    tf = model._target_frac()
    pt = model._pool_total()
    defs = [model.deficit((ftype, cluster, f, s), tf, pt) for (f, s) in opts]
    probs = range_normalized_softmax(defs, model.target.softmax_temp)
    idx = int(rng.choice(len(opts), p=probs))
    f, s = opts[idx]
    return f, s, defs[idx], len(opts), probs
