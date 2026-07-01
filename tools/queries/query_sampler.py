"""The sampler D and query composition — the shared candidate-generation front end.

This is the ONE module the labeling batch driver and the future inference sweep both
import. It never re-implements the coloring path: every candidate is a
`colormap.CandidateConfig` rendered by `colormap.render_candidate` (the field⊗colormap
tail). By construction, label-space ≡ sweep-space — both draw candidates from the same
`sample_candidate` / `PaletteSampler` here.

Three pieces:

  1. `LocationPool` — the q2+q3 location universe (both Mandelbrot and Julia families),
     read from the label corpus batch `images.jsonl` render blocks. Uniform draw.
  2. `PaletteSampler` — draws palettes from `pool_colormaps.json` (777) with the
     stratification the raw pool demands: the cyclic stratum is ~95% extracted (582 vs
     33 curated), so a uniform draw would confound *source* with *type*. Within a type
     we draw diversity-weighted (farthest-point flavored over the trajectory features),
     de-duplicating the internally-redundant extracted set and over-weighting the
     sparse curated cyclics well above their pool frequency.
  3. `sample_candidate` / `compose_query` — the sampler D over coloring params, and the
     query-type axis (palette / param / joint) that assembles 6 candidates on one
     held-constant location.

Everything continuous is a named module constant so ranges are tunable in one place.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "palettes"))

import colormap as cm  # noqa: E402  (CandidateConfig, LocationRef, PaletteLibrary, render_candidate, load_field)
import palette_features as pf  # noqa: E402  (distance_matrix over trajectory features)

# ---------------------------------------------------------------------------
# Settled inputs (paths).
# ---------------------------------------------------------------------------

POOL_COLORMAPS = ROOT / "data" / "palettes" / "pool_colormaps.json"
PALETTE_FEATURES = ROOT / "data" / "palettes" / "palette_features.json"
BATCHES_DIR = ROOT / "data" / "label_corpus" / "batches"

# ---------------------------------------------------------------------------
# Candidate-generation resolution — PINNED. Label-time and the future sweep share
# these; do not default them elsewhere.
# ---------------------------------------------------------------------------

CANDIDATE_SS = 2          # supersample for candidate generation (both label + sweep)
EVAL_WIDTH = 1024         # eval image width (16:9, matches the corpus 1280x720 aspect)
EVAL_HEIGHT = 576
CANDIDATE_FILTER = "box"  # ss2 box == flat 2x2 average; keeps recolor well under a second

# ---------------------------------------------------------------------------
# The sampler D — coloring param ranges (all centered on the smooth-mode sensibles).
# ---------------------------------------------------------------------------

GAMMA_LO = 1.0 / 3.0      # gamma is log-uniform over [1/3, 3], centered at 1
GAMMA_HI = 3.0
GAMMA_CANON_LO = 0.85     # "near-canonical" gamma band for palette queries (gamma ~= 1)
GAMMA_CANON_HI = 1.18
LOG_PREMAP_P = 0.15       # P(log_premap == 'log'); 0 in canonical draws
REVERSE_P = 0.5           # Bernoulli(reverse); non-cyclic only (redundant with phase on cyclic)
N_CYCLES_2_P = 0.25       # P(n_cycles == 2) on cyclic palettes; else 1

# Palette-stratum draw.
P_CYCLIC_TYPE = 0.5       # P(draw a cyclic palette) when type is unconstrained — balances
                          #   representation against the imbalanced 615/162 pool counts.
CURATED_WEIGHT = 10.0     # multiplier on curated palettes' sampling weight. Lifts the 33
                          #   curated cyclics from 33/615 (~5%) of the pool to ~40% of
                          #   realized cyclic draws (see report_stratum_mix()).
DEDUP_EPS = 0.06          # trajectory distance below which two palettes are near-dups;
                          #   a k-member near-dup cluster shares ~one unit of weight.
FPS_ALPHA = 1.0           # exponent on min-distance-to-chosen when drawing DISTINCT
                          #   palettes (palette query): farthest-point spreading.
FPS_FLOOR = 1e-3          # floor added to the min-distance so an exact-dup can't zero out.

# ---------------------------------------------------------------------------
# Query composition — the query-type axis.
# ---------------------------------------------------------------------------

QUERY_TYPES = ("palette", "param", "joint")
QUERY_SPLIT = (0.50, 0.35, 0.15)   # starting guess; retune against ranking-accuracy plateaus
CANDIDATES_PER_QUERY = 6


# ===========================================================================
# 1. Location pool.
# ===========================================================================

def _loc_key(kind, cx, cy, fw, c_re, c_im):
    return (kind, cx, cy, fw, c_re, c_im)


@dataclass
class PooledLocation:
    """A q2+q3 location plus a light provenance tail (kept for reporting only; never
    enters a candidate recipe — the recipe carries `cm.LocationRef`)."""
    ref: "cm.LocationRef"
    scores: list                # the q2/q3 scores that qualified this location
    batch_ids: set

    @property
    def kind(self):
        return self.ref.kind


class LocationPool:
    """All q2+q3 locations from the label corpus (both families), uniform draw.

    A *location* is a unique (kind, cx, cy, fw, c_re, c_im) across the corpus. A row
    qualifies if its merged `label.score` is 2 or 3. cx/cy already bake the composition
    offset (the render block stores the framed center), so no offset is re-applied.
    """

    def __init__(self, locations):
        self.locations = list(locations)

    @classmethod
    def from_corpus(cls, batches_dir=BATCHES_DIR, scores=(2, 3)):
        keep = set(scores)
        by_key = {}
        for jl in sorted(Path(batches_dir).glob("*/images.jsonl")):
            for line in jl.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sc = (row.get("label") or {}).get("score")
                if sc not in keep:
                    continue
                r = row["render"]
                kind = "julia" if r.get("fractal_type") == "julia" else "mandelbrot"
                c_re = r.get("c_re")
                c_im = r.get("c_im")
                key = _loc_key(kind, r["cx"], r["cy"], r["fw"], c_re, c_im)
                bid = (row.get("provenance") or {}).get("batch_id") or jl.parent.name
                if key in by_key:
                    pl = by_key[key]
                    pl.scores.append(sc)
                    pl.batch_ids.add(bid)
                else:
                    ref = cm.LocationRef(
                        kind=kind, cx=r["cx"], cy=r["cy"], fw=r["fw"],
                        maxiter=int(r["maxiter"]), c_re=c_re, c_im=c_im,
                    )
                    by_key[key] = PooledLocation(ref=ref, scores=[sc], batch_ids={bid})
        return cls(by_key.values())

    def by_family(self):
        out = {}
        for pl in self.locations:
            out.setdefault(pl.kind, []).append(pl)
        return out

    def report(self):
        fam = self.by_family()
        parts = [f"{k}={len(v)}" for k, v in sorted(fam.items())]
        return f"{len(self.locations)} q2+q3 locations ({', '.join(parts)})"

    def sample(self, rng):
        """Uniform draw of one PooledLocation."""
        i = int(rng.integers(len(self.locations)))
        return self.locations[i]


# ===========================================================================
# 2. Palette sampler — stratified, diversity + source weighted.
# ===========================================================================

class PaletteSampler:
    """Draws palettes from the 777-entry pool with per-stratum diversity+source weights.

    Weight of a palette within its type stratum:
        w = source_mult(source) / (1 + n_near_dups)
    where `source_mult` over-weights curated palettes (CURATED_WEIGHT) and `n_near_dups`
    is the count of same-stratum palettes within DEDUP_EPS trajectory distance (so a
    redundant extracted cluster of size k shares ~one unit of weight). Type is taken
    from the features file — the SAME `library.palette_type` the coloring path validates
    against, so a cyclic-only knob is never handed to a non-cyclic palette.
    """

    def __init__(self, library, features_path=PALETTE_FEATURES,
                 curated_weight=CURATED_WEIGHT, dedup_eps=DEDUP_EPS):
        self.library = library
        self.feats = json.loads(Path(features_path).read_text())
        self.curated_weight = curated_weight
        self.dedup_eps = dedup_eps

        # Source is on the colormap record; type is authoritative from the library.
        self._source = {name: c.get("source", "extracted") for name, c in library.colormaps.items()}

        self.strata = {}          # type -> list[name]
        self._weight = {}         # name -> base sampling weight
        self._dmat = {}           # type -> (names, DxD distance matrix)  (for FPS spreading)
        self._name_idx = {}       # type -> {name: row index in _dmat}
        for name in self.feats:
            if name not in library.colormaps:
                continue
            self.strata.setdefault(library.palette_type(name), []).append(name)

        for stratum, names in self.strata.items():
            D = pf.distance_matrix(self.feats, names)
            self._dmat[stratum] = (names, D)
            self._name_idx[stratum] = {n: i for i, n in enumerate(names)}
            n_near = (D < self.dedup_eps).sum(axis=1) - 1  # exclude self
            for i, name in enumerate(names):
                src_mult = self.curated_weight if self._source[name].startswith("curated") else 1.0
                self._weight[name] = src_mult / (1.0 + max(0, int(n_near[i])))

    # -- single draws ------------------------------------------------------

    def _draw_type(self, rng, constraint=None):
        if constraint is not None:
            return constraint
        return "cyclic" if rng.random() < P_CYCLIC_TYPE else "non_cyclic"

    def sample_palette(self, rng, type_constraint=None):
        """Draw one palette name: type first (or the constraint), then weighted within it."""
        stratum = self._draw_type(rng, type_constraint)
        names = self.strata[stratum]
        w = np.array([self._weight[n] for n in names], dtype=np.float64)
        w /= w.sum()
        return names[int(rng.choice(len(names), p=w))], stratum

    def sample_distinct(self, rng, k, type_constraint=None):
        """Draw `k` DISTINCT palettes, diversity-spread (farthest-point flavored).

        Pick 1 is weighted by the base weight; each subsequent pick multiplies the base
        weight by (min-distance-to-already-chosen + floor)**FPS_ALPHA, so picks push
        apart in trajectory space while still honoring source/dedup weighting. Types are
        mixed freely (a fresh type is drawn per pick unless constrained)."""
        chosen = []            # (name, stratum)
        chosen_by_stratum = {}  # stratum -> list of row indices already chosen
        for _ in range(k):
            stratum = self._draw_type(rng, type_constraint)
            names = self.strata[stratum]
            idx = self._name_idx[stratum]
            _, D = self._dmat[stratum]
            taken = chosen_by_stratum.get(stratum, [])
            w = np.array([self._weight[n] for n in names], dtype=np.float64)
            already = {n for n, s in chosen if s == stratum}
            for i, n in enumerate(names):
                if n in already:
                    w[i] = 0.0
            if taken:
                min_d = D[:, taken].min(axis=1)
                w = w * np.power(min_d + FPS_FLOOR, FPS_ALPHA)
            if w.sum() <= 0.0:  # stratum exhausted (tiny stratum, many picks) — retry a type
                continue
            w /= w.sum()
            j = int(rng.choice(len(names), p=w))
            nm = names[j]
            chosen.append((nm, stratum))
            chosen_by_stratum.setdefault(stratum, []).append(idx[nm])
        return chosen

    def source_of(self, name):
        return self._source.get(name, "extracted")

    def report_stratum_mix(self, rng, n=5000, type_constraint="cyclic"):
        """Diagnostic: realized curated fraction of draws in a stratum (default cyclic)."""
        cur = 0
        for _ in range(n):
            nm, _ = self.sample_palette(rng, type_constraint=type_constraint)
            if self._source[nm].startswith("curated"):
                cur += 1
        return cur / n


# ===========================================================================
# 3. The sampler D over params + query composition.
# ===========================================================================

def _log_uniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _draw_core_params(rng, canonical):
    """gamma + log_premap, shared by both types. `canonical` narrows to gamma~=1, no log."""
    if canonical:
        gamma = _log_uniform(rng, GAMMA_CANON_LO, GAMMA_CANON_HI)
        log_premap = "none"
    else:
        gamma = _log_uniform(rng, GAMMA_LO, GAMMA_HI)
        log_premap = "log" if rng.random() < LOG_PREMAP_P else "none"
    return gamma, log_premap


def sample_candidate(location_ref, rng, sampler, palette=None,
                     palette_type_constraint=None, canonical=False):
    """Draw one `cm.CandidateConfig` from D on a held-constant location.

    - `palette` fixed  -> param query: that palette, full param draw.
    - else             -> draw a palette (type first, then diversity+source weighted).
    - `canonical`      -> near-canonical params (gamma~=1, no log_premap): palette query.

    Per-type param rules:
      cyclic      : reverse=False (redundant with phase), phase~U[0,1), n_cycles in {1,2}.
      non_cyclic  : reverse~Bernoulli, no phase/n_cycles (core only).
    """
    if palette is not None:
        ptype = sampler.library.palette_type(palette)
        name = palette
    else:
        name, ptype = sampler.sample_palette(rng, type_constraint=palette_type_constraint)

    gamma, log_premap = _draw_core_params(rng, canonical)

    if ptype == "cyclic":
        reverse = False
        phase = float(rng.random())
        n_cycles = 2 if rng.random() < N_CYCLES_2_P else 1
    else:
        reverse = bool(rng.random() < REVERSE_P)
        phase = 0.0
        n_cycles = 1

    return cm.CandidateConfig(
        palette=name, location=location_ref,
        eval_width=EVAL_WIDTH, eval_height=EVAL_HEIGHT,
        reverse=reverse, log_premap=log_premap, gamma=gamma,
        phase=phase, n_cycles=n_cycles, filter=CANDIDATE_FILTER,
    )


def draw_query_type(rng):
    r = rng.random()
    acc = 0.0
    for t, p in zip(QUERY_TYPES, QUERY_SPLIT):
        acc += p
        if r < acc:
            return t
    return QUERY_TYPES[-1]


def compose_query(location, rng, sampler, query_type=None):
    """6 candidates on ONE held-constant location. Returns (query_type, [CandidateConfig]).

    palette (~50%): 6 distinct palettes, near-canonical params, types mixed freely.
    param   (~35%): 1 diversity-drawn fixed palette, 6 full-random param variants.
    joint   (~15%): 6 full-random draws from D.
    """
    loc_ref = location.ref if isinstance(location, PooledLocation) else location
    if query_type is None:
        query_type = draw_query_type(rng)

    cands = []
    if query_type == "palette":
        for name, _stratum in sampler.sample_distinct(rng, CANDIDATES_PER_QUERY):
            cands.append(sample_candidate(loc_ref, rng, sampler, palette=name, canonical=True))
    elif query_type == "param":
        fixed, _ = sampler.sample_palette(rng)   # diversity-weighted fixed palette
        for _ in range(CANDIDATES_PER_QUERY):
            cands.append(sample_candidate(loc_ref, rng, sampler, palette=fixed, canonical=False))
    elif query_type == "joint":
        for _ in range(CANDIDATES_PER_QUERY):
            cands.append(sample_candidate(loc_ref, rng, sampler, canonical=False))
    else:
        raise ValueError(f"unknown query_type {query_type!r}")
    return query_type, cands


# ===========================================================================
# Palette library wired to the POOL (adds mirror_needed the pool file omits).
# ===========================================================================

def load_pool_library(colormaps_path=POOL_COLORMAPS, features_path=PALETTE_FEATURES):
    """`cm.PaletteLibrary` over pool_colormaps.json, injecting `mirror_needed`.

    pool_colormaps.json (unlike score3/clean) carries no `mirror_needed`; the render
    load derives it from the declared cycle exactly as score3 did — sequential maps
    need the selective pre-mirror to de-seam, cyclic maps do not. Rule verified against
    score3_colormaps.json: mirror_needed <=> cycle == 'sequential'."""
    lib = cm.PaletteLibrary(colormaps_path=str(colormaps_path), features_path=str(features_path))
    for name, c in lib.colormaps.items():
        c["mirror_needed"] = (c.get("cycle") == "sequential")
    return lib


if __name__ == "__main__":
    # Smoke: report the pool + realized curated cyclic mix.
    rng = np.random.default_rng(0)
    pool = LocationPool.from_corpus()
    print(pool.report())
    lib = load_pool_library()
    sampler = PaletteSampler(lib)
    for st, names in sorted(sampler.strata.items()):
        print(f"stratum {st}: {len(names)} palettes")
    print(f"realized curated fraction of cyclic draws: "
          f"{sampler.report_stratum_mix(rng):.3f} (pool freq 33/615 = 0.054)")
