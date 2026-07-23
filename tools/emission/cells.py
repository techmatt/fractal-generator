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

An override may also key on a durable `source_tag` (the ledger-carried intake source,
e.g. `classic_phoenix`) INSTEAD of an unstable `morph_cluster` id. `source_tag` is not a
cell axis, so a consumer that has the intake's (location -> source_tag, location -> cluster)
maps calls `resolve_source_tags` once to rewrite each such override into the concrete
`morph_cluster` set those tagged locations currently occupy — re-derived every intake, so
config never references cluster ids that a re-cluster silently invalidates. A consumer that
does NOT resolve (the discovery-side per-type projection) sees the `source_tag` key as a
non-cell axis that never matches, so the override is a no-op there — exactly as a
cluster-id override on clusters that projection never observes was.

Everything here is pure Python + stdlib so the deficit logic is unit-testable without
loading a model or rendering a frame.
"""
from __future__ import annotations

import copy
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
          {"match": {"fractal_type": ["mandelbrot"]},        "weight": 1.5},
          {"match": {"source_tag": ["classic_phoenix"]},     "weight": 1.9}
        ]
      }

    A cell's base target weight = 1.0 × ∏ (override.weight for every override whose
    `match` the cell satisfies). An override matches a cell iff, for every axis it
    names, the cell's value on that axis is in the override's listed values.

    `source_tag` is a durable, non-cell key (see `resolve_source_tags`): a consumer with the
    intake maps resolves it to concrete morph clusters before scoring; an unresolved consumer
    treats it as a never-matching axis (the override is a no-op there).
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
            # deep-copy: resolve_source_tags mutates override match dicts in place, which must
            # never reach back into the caller's cfg or a sibling TargetMeasure built from it.
            weight_overrides=copy.deepcopy(list(cfg.get("weight_overrides", []))),
        )

    def _axis_index(self, axis: str) -> int | None:
        """Cell-axis position, or None for a non-cell axis (e.g. an UNRESOLVED `source_tag`)."""
        return AXES.index(axis) if axis in AXES else None

    def _matches(self, match: dict, cell: Cell) -> bool:
        """True iff, for every axis `match` names, the cell's value is in that axis's set. A
        non-cell axis (an unresolved `source_tag`) can never match a bare cell → False (no-op)."""
        for axis, vals in match.items():
            idx = self._axis_index(axis)
            if idx is None or cell[idx] not in vals:
                return False
        return True

    def weight(self, cell: Cell) -> float:
        w = 1.0
        for ov in self.weight_overrides:
            # an unsolved target_share override (no numeric `weight`) is a no-op here — it needs
            # solve_target_shares (which needs the feasible set) to become a concrete multiplier;
            # until then `.get("weight", 1.0)` folds it to 1.0. On the discovery-side projection
            # (which never solves) a source-tag target_share stays a no-op, exactly as intended.
            if self._matches(ov.get("match", {}), cell):
                w *= float(ov.get("weight", 1.0))
        return w

    def resolve_source_tags(self, loc_source_tags: dict, loc_clusters: dict) -> list:
        """Rewrite every `source_tag` override into a concrete `morph_cluster` override, using
        THIS intake's live maps `location_id -> source_tag` and `location_id -> morph_cluster`.

        A source-tag override names a location set DURABLY (by the tag its ledger row carries),
        then re-derives which morph clusters those locations occupy at intake time — so the
        config never references cluster ids, which are unstable across re-clustering. Mutates
        `weight_overrides` in place: a match's `source_tag` entry becomes a `morph_cluster`
        entry (intersected with an existing `morph_cluster` entry if the override names both).
        Idempotent — a second call finds no `source_tag` keys. Returns per-resolved-override
        diagnostics (`source_tags`, `n_locations`, `resolved_clusters`, `impure_clusters`)."""
        members: dict = {}
        for i, c in loc_clusters.items():
            members.setdefault(c, []).append(i)
        diags = []
        for ov in self.weight_overrides:
            match = ov.get("match", {})
            if "source_tag" not in match:
                continue
            tags = set(match.pop("source_tag"))
            tagged = {i for i, t in loc_source_tags.items() if t in tags}
            clusters = {loc_clusters[i] for i in tagged if i in loc_clusters}
            if "morph_cluster" in match:
                clusters &= set(match["morph_cluster"])
            resolved_locs = {i for i in tagged if loc_clusters.get(i) in clusters}
            # purity: a resolved cluster that ALSO holds an untagged location means the override
            # up-weights that non-source-tag member too (the caller's equivalence gate forbids it).
            impure = sorted(c for c in clusters
                            if any(i not in tagged for i in members.get(c, [])))
            match["morph_cluster"] = sorted(clusters)
            diags.append({
                "source_tags": sorted(tags),
                "n_locations": len(resolved_locs),
                "resolved_clusters": sorted(clusters),
                "impure_clusters": impure,
            })
        return diags

    def _base_weight(self, cell: Cell, skip_ov: dict) -> float:
        """Cell weight from every override EXCEPT `skip_ov` that already carries a numeric
        `weight` — i.e. all fixed-weight overrides plus any target-share override solved earlier
        in the same pass. Unsolved target-share overrides (no `weight` key) are excluded, so a
        share is always sized against the fixed-weight measure it stacks on."""
        w = 1.0
        for ov in self.weight_overrides:
            if ov is skip_ov or "weight" not in ov:
                continue
            if self._matches(ov.get("match", {}), cell):
                w *= float(ov["weight"])
        return w

    def solve_target_shares(self, feasible_cells: Iterable[Cell]) -> list:
        """Convert each `target_share` override into a solved `weight` multiplier so its matched
        cell set holds EXACTLY that share of the total measure over `feasible_cells`.

        Solved AFTER all fixed-weight overrides: a cell's BASE weight (used to size the share) is
        the product of every override that already carries a numeric `weight` — the fixed-weight
        overrides plus any target-share override solved earlier in this pass (`_base_weight`). The
        share is therefore ABSOLUTE and DECOUPLED from any type-budget knob that also multiplies
        the matched set: moving that knob scales the base symmetrically, and the re-solved
        multiplier restores the exact share. It is likewise DENOMINATOR-INVARIANT — enlarging the
        intake grows the non-matched mass, and the multiplier grows to compensate.

        With base masses A = Σ_{c∈M} base(c) over the matched cells M and B = Σ_{c∉M} base(c) over
        the rest, the multiplier λ making λA/(λA+B) = s is λ = sB / (A(1−s)). Mutates each solved
        override in place (`target_share` → `weight`) so `weight()` and every downstream consumer
        see a plain weight override; idempotent (a second call finds no `target_share` keys).

        Must run AFTER `resolve_source_tags`. A source-tag target_share whose tag never resolved
        (the discovery-side projection, which does not resolve) matches no feasible cell (A=0) and
        is LEFT UNTOUCHED — a no-op, exactly as the pre-solve state. Returns per-override
        diagnostics (`target_share`, `matched_cells`, `solved_multiplier`, `realized_share`)."""
        feasible = list(feasible_cells)
        diags = []
        for ov in self.weight_overrides:
            if "target_share" not in ov:
                continue
            s = float(ov["target_share"])
            if not (0.0 < s < 1.0):
                raise ValueError(f"target_share must be in (0,1), got {s}")
            match = ov.get("match", {})
            A = B = 0.0
            nM = 0
            for c in feasible:
                base = self._base_weight(c, skip_ov=ov)
                if self._matches(match, c):
                    A += base
                    nM += 1
                else:
                    B += base
            if A <= 0.0:
                # matched set empty over feasible cells (e.g. an unresolved source_tag): leave the
                # override a no-op — its `target_share` stays and `weight()` folds it to 1.0.
                diags.append({"target_share": s, "matched_cells": 0, "solved_multiplier": None,
                              "realized_share": 0.0, "unsolved_reason": "empty matched set"})
                continue
            if B <= 0.0:
                raise ValueError(
                    f"target_share {s}: matched set is the ENTIRE measure (B=0); no multiplier "
                    f"can make it less than 100%")
            lam = s * B / (A * (1.0 - s))
            del ov["target_share"]
            ov["weight"] = lam
            diags.append({"target_share": s, "matched_cells": nM, "solved_multiplier": lam,
                          "realized_share": (lam * A) / (lam * A + B)})
        return diags


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
