#!/usr/bin/env python
"""Degenerate-outcome guard — the model-free, field-based gate inside the reward.

Two calibrated gates (pinned thresholds; the canonical 81-set + verdicts they produce
are the tripwire fixture `data/atlas/guard_tripwire.json`) become a HARD guard that
raw-frame scoring, reframe candidate scoring, and outcome scoring all inherit:

    interior gate:  interior_frac >= INTERIOR_CAP   -> fail (black gate)
    flat gate:      field_std     <  FIELD_STD_FLOOR -> fail (flat gate)

Both measures are MODEL-FREE and come from the crop's *field* (never RGB luminance):
`render-one --dump-field --dump-field-source f64` emits the smooth scalar field with
NaN at interior / non-escaped subpixels, exactly the seam the calibration diagnostic
measured. The field is sourced from the fast escape-time F64Backend smooth channel
(NOT the slow beautiful kernel the diagnostic dumped) — its value differs by the
constant `ln(ln B)/ln d` bailout-normalization offset, to which interior_frac (an
escape-mask fraction) and field_std (a std) are both invariant. So the in-scorer
field path's VERDICTS match `diag_outcome_guards.py`, and the re-render tripwire
(`test_guard_tripwire.py`) regresses this exact path against the pinned 81-set
verdicts every run (field values are NOT byte-identical, by design).

`field_measures` is the byte-for-byte reproduction of `diag_outcome_guards.measures`
(the interior_frac / field_std half); `guard_fail` applies the pinned thresholds;
`make_guarded_scorer` wraps the v5 `Scorer` so a failing crop returns GUARD_SENTINEL
instead of the v5 forward. The scored image stays the deploy-canonical view (the
guard only *rejects*; a passing crop scores exactly as it does today).

Reuses (does not reinvent): `probe.{auto_maxiter, BIN, PALETTE, JPG_Q, make_scorer}`,
`colormap.load_field` (the dumped-field reader, NaN interior), `location.{Location,
render_one_flags}`. Render fidelity is pinned to GUARD_STAT_RES (640x360 ss2), the
reframe/deploy search fidelity the diagnostic used.

  uv run pytest tools/atlas/test_guard.py
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "scoring", ROOT / "tools" / "mining"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import location as loc_mod                               # noqa: E402
from active_ckpt import auto_maxiter, BIN, make_scorer, ACTIVE_CKPT  # noqa: E402
from colormap import load_field                           # noqa: E402

# =========================================================================== #
# Config (pinned — the drop-manifest thresholds).
# =========================================================================== #
INTERIOR_CAP = 0.25       # interior_frac >= this  -> fail (black gate)
FIELD_STD_FLOOR = 6.0     # field_std < this       -> fail (flat gate)
GUARD_SENTINEL = -1.0     # returned score for a failing crop (< any real E[ord] in [0,2])

# Compute guard stats at the reframe/deploy search fidelity (matches the diagnostic's
# --dump-field measurement, so the Part-2 control reproduces the manifest exactly).
GUARD_W, GUARD_H, GUARD_SS = 640, 360, 2
GUARD_STAT_RES = f"{GUARD_W}x{GUARD_H} ss{GUARD_SS} 16:9"

# The location-quality classifier the guard wraps. Resolved from the single source of
# truth (probe.ACTIVE_CKPT — currently v6); explicit, NEVER a bare default scorer.
SCORER_PATH = ACTIVE_CKPT


# =========================================================================== #
# Field measures — reproduce diag_outcome_guards.measures (the interior_frac /
# field_std half) EXACTLY. Model-free; from the dumped field only.
# =========================================================================== #
@dataclass(frozen=True)
class GuardStats:
    interior_frac: float
    field_std: float
    n_px: int
    n_escaped: int


def field_measures(values: np.ndarray) -> GuardStats:
    """(H,W) smooth field, NaN at interior/non-escaped -> GuardStats.

    interior_frac == fraction NaN == non-escaped fraction (from the escape mask, NOT
    RGB). field_std == std of the smooth field over the finite (escaped) pixels, 0.0
    if none. Byte-identical to diag_outcome_guards.measures."""
    v = np.asarray(values)
    finite = np.isfinite(v)
    interior_frac = float(1.0 - finite.mean())
    vals = v[finite]
    field_std = float(vals.std()) if vals.size else 0.0
    return GuardStats(interior_frac=interior_frac, field_std=field_std,
                      n_px=int(v.size), n_escaped=int(finite.sum()))


def guard_fail(interior_frac: float, field_std: float) -> str | None:
    """Pinned gate. Returns the fail reason ('interior'|'flat'|'both') or None (pass).

    A crop FAILS iff interior_frac >= INTERIOR_CAP OR field_std < FIELD_STD_FLOOR."""
    hit_interior = interior_frac >= INTERIOR_CAP
    hit_flat = field_std < FIELD_STD_FLOOR
    if hit_interior and hit_flat:
        return "both"
    if hit_interior:
        return "interior"
    if hit_flat:
        return "flat"
    return None


# =========================================================================== #
# Field render — one dumped smooth field at GUARD_STAT_RES. Mirrors
# diag_outcome_guards.dump_field exactly (same fidelity + maxiter policy), so the
# guard's own field path reproduces the diagnostic.
# =========================================================================== #
def render_field(cx, cy, fw, out_bin: Path, *, family: str = "mandelbrot",
                 c_re=None, c_im=None, family_params=None, maxiter: int | None = None):
    """render-one --dump-field at GUARD_STAT_RES. Returns (ok, err). Writes <out_bin>
    (+ .json sidecar). maxiter defaults to auto_maxiter(fw) (the fw-dependent policy
    the diagnostic used)."""
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    mi = auto_maxiter(float(fw)) if maxiter is None else int(maxiter)
    loc = loc_mod.Location(family=family, cx=str(cx), cy=str(cy), fw=str(fw),
                           c_re=c_re, c_im=c_im, family_params=family_params or {})
    cmd = [
        str(BIN), "render-one", "--cx", str(cx), "--cy", str(cy), "--fw", repr(float(fw)),
        "--width", str(GUARD_W), "--height", str(GUARD_H), "--supersample", str(GUARD_SS),
        "--maxiter", str(mi), "--dump-field", str(out_bin),
        # Guard reads only interior_frac (escape mask) + field_std (a std) — both
        # invariant to the bailout-normalization offset between the beautiful and
        # escape-time smooth kernels. Source the field from the fast F64Backend
        # smooth channel (deletes the redundant slow beautiful-kernel render); the
        # re-render tripwire (test_guard_tripwire.py) proves verdict parity.
        "--dump-field-source", "f64",
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out_bin.exists()
    return ok, ("" if ok else r.stderr[-300:])


def measure_location(cx, cy, fw, out_bin: Path, **kw) -> GuardStats:
    """Render the field for one location + measure it. Raises on render failure."""
    ok, err = render_field(cx, cy, fw, out_bin, **kw)
    if not ok:
        raise SystemExit(f"guard field render failed [{Path(out_bin).name}]: {err}")
    fd = load_field(out_bin)
    return field_measures(fd.values)


# =========================================================================== #
# Guarded scorer — wraps the v5 Scorer. A failing crop returns GUARD_SENTINEL;
# a passing crop scores exactly as the bare v5 does.
#
# The guard needs the crop's FIELD, which score_paths(paths) does not carry. The
# reframe/raw render paths therefore write a co-located field sidecar per tile (the
# small reframe hook), keyed by the tile stem; score_paths reads it when present.
# When no field sidecar is present (any un-hooked caller), the crop scores unguarded
# (pass-through) — the guard is opt-in per render path, never silently mis-gating.
# =========================================================================== #
FIELD_SIDECAR_SUFFIX = ".field.bin"   # co-located with <tile>.jpg as <tile>.jpg.field.bin


def field_sidecar_for(tile_path) -> Path:
    """The co-located guard field for a rendered tile (<tile>.field.bin)."""
    return Path(str(tile_path) + FIELD_SIDECAR_SUFFIX)


class GuardedScorer:
    """v5 Scorer + the model-free field guard. Same interface (score_paths /
    score_pils / cfg / model / transform / device) so it is a drop-in for the bare
    Scorer everywhere make_scorer was used.

    score_paths guards each tile that has a co-located field sidecar: a failing tile
    is forced to GUARD_SENTINEL (the v5 forward is skipped for it); passing tiles keep
    the exact v5 triple. Tiles with no sidecar pass through unguarded."""

    def __init__(self, scorer):
        self._scorer = scorer

    # --- delegate the deploy surface so reframe / embed_paths keep working --- #
    @property
    def model(self):
        return self._scorer.model

    @property
    def transform(self):
        return self._scorer.transform

    @property
    def device(self):
        return self._scorer.device

    @property
    def cfg(self):
        return self._scorer.cfg

    def score_pils(self, imgs):
        return self._scorer.score_pils(imgs)

    def _guard_of(self, path) -> str | None:
        """None if the tile passes (or has no field sidecar); else the fail reason."""
        fp = field_sidecar_for(path)
        if not fp.exists():
            return None
        stats = field_measures(load_field(fp).values)
        return guard_fail(stats.interior_frac, stats.field_std)

    def score_paths(self, paths, batch_size: int = 64):
        """Score JPGs; a tile whose co-located field fails the guard returns the
        sentinel triple (GUARD_SENTINEL, 0.0, 0.0) without a v5 forward."""
        paths = list(paths)
        fails = {i: self._guard_of(p) for i, p in enumerate(paths)}
        keep = [i for i in range(len(paths)) if fails[i] is None]
        scored = {}
        if keep:
            triples = self._scorer.score_paths([paths[i] for i in keep], batch_size=batch_size)
            for i, t in zip(keep, triples):
                scored[i] = tuple(float(x) for x in t)
        out = []
        for i in range(len(paths)):
            if fails[i] is None:
                out.append(scored[i])
            else:
                out.append((GUARD_SENTINEL, 0.0, 0.0))
        return out


def make_guarded_scorer(model_path: str = SCORER_PATH, device: str | None = None) -> GuardedScorer:
    """v5 CORN scorer wrapped in the field guard (drop-in for probe.make_scorer)."""
    base = make_scorer(model_path) if device is None else make_scorer(model_path)
    return GuardedScorer(base)
