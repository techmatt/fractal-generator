"""The sampler D and query composition — the shared candidate-generation front end.

This is the ONE module the labeling batch driver and the future inference sweep both
import. It never re-implements the coloring path: every candidate is a
`colormap.CandidateConfig` rendered by `colormap.render_candidate` (the field⊗colormap
tail). By construction, label-space ≡ sweep-space — both draw candidates from the same
`sample_candidate` / `PaletteSampler` here.

Three pieces:

  1. `LocationPool` — the q2+q3 location universe (both Mandelbrot and Julia families),
     read from the label corpus batches. Labels are resolved from the labels/ store:
     a batch's merged `images.jsonl` score, else its `labels/*.json` sidecar joined by
     image_id (Julia/mining/scale live only in sidecars). Mirrors the v5 pipeline's join
     (tools/v5/build_manifest.py) so no family is silently dropped. Uniform draw.
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
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import _bootstrap  # noqa: E402,F401  (adds tools/{palettes,corpus,queries} to sys.path)

import colormap as cm  # noqa: E402  (CandidateConfig, LocationRef, PaletteLibrary, render_candidate, load_field)
import palette_features as pf  # noqa: E402  (distance_matrix over trajectory features)
import label_store as ls  # noqa: E402  (SIDECAR_LABELS + resolve_score — the shared resolver)
import corpus_reader as cr  # noqa: E402  (iter_labeled — the ONE version-blind batch reader)
import location as loc_mod  # noqa: E402  (canonical Location + key + render-one flags)

# ---------------------------------------------------------------------------
# Settled inputs (paths).
# ---------------------------------------------------------------------------

POOL_COLORMAPS = ROOT / "data" / "palettes" / "pool_colormaps.json"
PALETTE_FEATURES = ROOT / "data" / "palettes" / "palette_features.json"
BATCHES_DIR = ROOT / "data" / "label_corpus" / "batches"

# ---------------------------------------------------------------------------
# Authoritative label sources.
#
# Label resolution — the SIDECAR_LABELS registry, the merged-score-else-sidecar-join
# rule, and the load-time join guard — lives in ONE place, tools/corpus/label_store.py,
# and is shared with the version-blind trainer reader (corpus_reader.iter_labeled) so
# the two consumers can never drift. SIDECAR_LABELS is re-exported here for callers /
# tests that reference `query_sampler.SIDECAR_LABELS`. NEW unmerged batches MUST be
# registered in label_store (or have their labels merged into images.jsonl).
#
# REFERENCE for the complete label set: the v5 unified Mandelbrot+Julia classifier's
# training-data assembly, tools/v5/build_manifest.py. It recovers the J0 Julia labels
# from labels/location_labels_julia_ladder_j0.json JOINED to the batch's images.jsonl
# by image_id — exactly the join label_store mirrors. See also CORPUS_SCHEMA.md.
#
# Row schema differs by FAMILY (both families share the one images.jsonl render block):
#   Mandelbrot: cx, cy, fw                 (c_re/c_im absent or null)
#   Julia:      cx, cy, fw + c_re, c_im    (cx/cy/fw is the z-plane VIEWPORT; c_re/c_im
#                                           is the Julia parameter c — the fractal's
#                                           identity, and part of the location dedup key)
SIDECAR_LABELS = ls.SIDECAR_LABELS   # single source of truth = label_store

# The v5 pipeline's authoritative Julia label source — the cross-check reference
# (see v5_julia_q23_count() and tools/queries/test_location_pool.py).
V5_JULIA_BATCH = "julia_ladder_j0"
V5_MANIFEST = ROOT / "data" / "v5" / "manifest.jsonl"


def v5_julia_q23_count(scores=(2, 3)):
    """Q2+q3 Julia count computed by the SAME join tools/v5/build_manifest.py uses:
    labels/location_labels_julia_ladder_j0.json (image_id -> score) INTERSECT the
    julia_ladder_j0 batch images.jsonl (render params) by image_id. This is the
    authoritative reference the sampler's Julia location count must match; empirically
    no q2/q3 row is dropped by build_manifest's v4-seed split filter, so this equals
    the Julia q2/q3 count in data/v5/manifest.jsonl."""
    labels = ls.load_sidecar(SIDECAR_LABELS[V5_JULIA_BATCH])
    jl = BATCHES_DIR / V5_JULIA_BATCH / "images.jsonl"
    batch_ids = {json.loads(l)["image_id"]
                 for l in jl.read_text(encoding="utf-8").splitlines() if l.strip()}
    keep = set(scores)
    return sum(1 for iid, sc in labels.items() if iid in batch_ids and sc in keep)


def v5_manifest_julia_q23(scores=(2, 3)):
    """Julia q2+q3 count straight from data/v5/manifest.jsonl (the v5 pipeline's own
    output), or None if the manifest is absent. The literal 'v5 pipeline's count'."""
    if not V5_MANIFEST.exists():
        return None
    keep = set(scores)
    return sum(1 for l in V5_MANIFEST.read_text(encoding="utf-8").splitlines()
               if l.strip()
               for r in [json.loads(l)]
               if r.get("fractal_type") == "julia" and r.get("label") in keep)

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

# Param-query render-space selection (coldstart_v2). Param candidates collapse in
# render space even when their recipes differ, so instead of drawing 6 directly we draw
# a larger POOL and farthest-point-select 6 in render space (mean CIEDE2000 over the
# recolored thumbnails — see the v2 driver). Ranges are UNCHANGED; this only changes
# which of the same-range draws survive.
POOL_MULTIPLIER = 4        # pool size = POOL_MULTIPLIER * CANDIDATES_PER_QUERY (=> 24)
GAMMA_MIN_SPACING = 0.15   # a new pool draw is rejected if a same-discrete-tuple member
                           #   (palette fixed; same reverse/log_premap/n_cycles/phase) is
                           #   within this |Δγ|. > GAMMA_DUP_EPS (0.10) in the diagnostic,
                           #   so no surviving same-discrete pair can be a sampler_dup.


# ===========================================================================
# 1. Location pool.
# ===========================================================================

@dataclass
class PooledLocation:
    """A q2+q3 location plus a light provenance tail (scores/batch_ids kept for
    reporting only). `ref` is the canonical `location.Location` (family + geometry +
    family_params) — it exposes `.kind`/`.cx`/`.cy`/`.fw`/`.maxiter`/`.c_re`/`.c_im`
    like `cm.LocationRef`, so downstream is unchanged, and additionally carries
    `family_params` (Phoenix's `p`) for the field cache + render args. It is converted
    to a `cm.LocationRef` (dropping family_params) only when it enters a coloring
    recipe (`location.to_location_ref`)."""
    ref: "loc_mod.Location"
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

    def __init__(self, locations, census=None):
        self.locations = list(locations)
        # census[batch_id][family] = q2+q3 ROWS contributed (pre-dedup); for reporting.
        self.census = census or {}

    @classmethod
    def from_corpus(cls, batches_dir=BATCHES_DIR, scores=(2, 3), verbose=True):
        """Build the q2+q3 pool from every label-corpus batch, BOTH families.

        Batch reading + label resolution are DELEGATED to `corpus_reader.iter_labeled`
        — the one version-blind reader — so a schema change to images.jsonl is absorbed
        in a single place. `iter_labeled` yields one `LabeledCrop(score, image_id,
        batch_id, render)` per non-null label (merged `label.score` ELSE the registered
        `labels/*.json` sidecar joined by image_id — how Julia/mining/scale are
        recovered) and runs the shared sidecar-join guard over its full pass. Here we
        filter that stream to `scores` and dedup to unique locations.

        A *location* is a unique (kind, cx, cy, fw, c_re, c_im); cx/cy already bake the
        composition offset. `batch_ids` is the label-corpus batch(es) the location was
        labeled in — provenance is NOT surfaced by the version-blind reader (by design),
        so this is the batch folder, not any upstream `provenance.batch_id`. Prints a
        loud per-batch, per-family census (a dropped family/batch shows as a `0`) and
        RAISES if a registered sidecar batch joins 0 q2+q3 rows."""
        keep = set(scores)
        by_key = {}
        census = {}
        for lc in cr.iter_labeled(str(Path(batches_dir).parent)):
            fams = census.setdefault(lc.batch_id, {})   # every labeled batch shows in the census
            if lc.score not in keep:
                continue
            r = lc.render
            # Family-general: read `fractal_type` (mandelbrot when absent) and the
            # per-family params slot. The canonical Location is the pool's `ref` — it
            # exposes `.kind`/`.cx`/... like LocationRef but also carries family_params,
            # so Phoenix's `p` threads through the field cache + render args downstream.
            ref = loc_mod.from_render_block(r)
            kind = ref.family
            fams[kind] = fams.get(kind, 0) + 1
            key = ref.key()
            if key in by_key:
                pl = by_key[key]
                pl.scores.append(lc.score)
                pl.batch_ids.add(lc.batch_id)
            else:
                by_key[key] = PooledLocation(ref=ref, scores=[lc.score], batch_ids={lc.batch_id})
        pool = cls(by_key.values(), census=census)
        if verbose:
            pool.print_census()
        # Join-integrity guard on the q2+q3 census specifically (iter_labeled runs its
        # own guard over ALL labels): every registered sidecar batch is known to hold
        # q2/q3 labels, so a 0 here means the image_id join broke (renamed keys / wrong
        # file).
        ls.assert_sidecars_joined({bid: sum(f.values()) for bid, f in census.items()})
        return pool

    def by_family(self):
        out = {}
        for pl in self.locations:
            out.setdefault(pl.kind, []).append(pl)
        return out

    def family_counts(self):
        """{family: unique-location count} over the deduped pool."""
        return {k: len(v) for k, v in self.by_family().items()}

    def report(self):
        fam = self.by_family()
        parts = [f"{k}={len(v)}" for k, v in sorted(fam.items())]
        return f"{len(self.locations)} q2+q3 locations ({', '.join(parts)})"

    def print_census(self, stream=sys.stderr):
        """Loud per-batch x per-family q2+q3 census at load (Part-4 hardening)."""
        print("[LocationPool] q2+q3 census by batch x family "
              "(labels/ store; sidecar-joined where noted):", file=stream)
        fam_tot = {}
        for bid in sorted(self.census):
            fams = self.census[bid]
            parts = ", ".join(f"{k}={fams[k]}" for k in sorted(fams)) or "(none labeled)"
            src = f"   <- {SIDECAR_LABELS[bid]}" if bid in SIDECAR_LABELS else ""
            print(f"    {bid}: {parts}{src}", file=stream)
            for k, v in fams.items():
                fam_tot[k] = fam_tot.get(k, 0) + v
        # All families present (additive): mandelbrot/julia first, then any new family.
        order = [k for k in ("mandelbrot", "julia") if k in fam_tot]
        order += sorted(k for k in fam_tot if k not in ("mandelbrot", "julia"))
        rows = ", ".join(f"{k}={fam_tot[k]}" for k in order) or "mandelbrot=0, julia=0"
        print(f"    -- q2+q3 rows (pre-dedup): {rows}", file=stream)
        print(f"    -- {self.report()}", file=stream)

    def assert_matches_v5(self):
        """Cross-check the sampler's Julia location count against the v5 pipeline's
        (Part-4 validation guard). Julia is keyed by image_id in both paths, so the two
        must agree; a mismatch means the sampler drifted from the authoritative label
        set (in EITHER direction). Raises AssertionError on mismatch.

        SCOPED to the v5 (mandelbrot+julia) subset by construction: it asserts on the
        `julia` family count ONLY, so future new-family locations (multibrot/phoenix),
        which don't exist in the v5 manifest, add to the pool without tripping this
        check."""
        got = self.family_counts().get("julia", 0)
        want_join = v5_julia_q23_count()
        assert got == want_join, (
            f"Julia location count {got} != v5 join-recipe count {want_join} "
            f"(labels/{SIDECAR_LABELS[V5_JULIA_BATCH]} x {V5_JULIA_BATCH})")
        want_man = v5_manifest_julia_q23()
        if want_man is not None:
            assert got == want_man, (
                f"Julia location count {got} != v5 manifest count {want_man} "
                f"({V5_MANIFEST})")
        return got

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
        palette=name, location=loc_mod.to_location_ref(location_ref),
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


def _discrete_key(cfg):
    """The sampler_dup discrete tuple (everything but gamma) for the γ-spacing guard —
    mirrors diversity_diagnostic.classify_pair's `same_discrete`. Palette is held fixed
    across a param pool, so it's not part of the key."""
    return (cfg.reverse, cfg.log_premap, cfg.n_cycles, cfg.phase)


def draw_param_pool(loc_ref, rng, sampler, palette, pool_size=None,
                    gamma_min_spacing=GAMMA_MIN_SPACING, max_attempts_mult=50):
    """Draw a param-query candidate POOL on one fixed palette + held-constant location.

    Same D draw as v1's param branch (`sample_candidate(..., palette=palette,
    canonical=False)` — identical rev/γ/phase/n_cycles ranges), but sized to `pool_size`
    (default POOL_MULTIPLIER * CANDIDATES_PER_QUERY) with a min-γ-spacing rejection guard:
    a draw is dropped if a pool member with the same discrete tuple (reverse/log_premap/
    n_cycles/phase) sits within `gamma_min_spacing` in γ. This removes the literal
    sampler-dups cheaply at draw time; the render-space farthest-point select (in the
    driver) then thins the perceptual collapse the guard can't see. Returns a list of
    CandidateConfig (may be shorter than `pool_size` only if the guard starves, which is
    reported by the caller)."""
    if pool_size is None:
        pool_size = POOL_MULTIPLIER * CANDIDATES_PER_QUERY
    pool = []
    attempts = 0
    cap = pool_size * max_attempts_mult
    while len(pool) < pool_size and attempts < cap:
        attempts += 1
        cfg = sample_candidate(loc_ref, rng, sampler, palette=palette, canonical=False)
        key = _discrete_key(cfg)
        if any(_discrete_key(p) == key and abs(p.gamma - cfg.gamma) < gamma_min_spacing
               for p in pool):
            continue
        pool.append(cfg)
    return pool


# ===========================================================================
# Palette library wired to the POOL.
# ===========================================================================

def load_pool_library(colormaps_path=POOL_COLORMAPS, features_path=PALETTE_FEATURES):
    """`cm.PaletteLibrary` over pool_colormaps.json — a thin path-binding wrapper.

    pool_colormaps.json now carries `mirror_needed` natively (emitted by
    tools/palettes/build_pool.py, one schema with score3/clean), so there is nothing
    to inject: `PaletteLibrary.lut` reads it directly. Sequential maps de-seam via the
    selective pre-mirror; cyclic maps do not."""
    return cm.PaletteLibrary(colormaps_path=str(colormaps_path), features_path=str(features_path))


if __name__ == "__main__":
    # Smoke: report the pool + realized curated cyclic mix.
    rng = np.random.default_rng(0)
    pool = LocationPool.from_corpus()
    print(pool.report())
    print(f"v5 Julia cross-check: sampler={pool.family_counts().get('julia', 0)} "
          f"join-recipe={v5_julia_q23_count()} manifest={v5_manifest_julia_q23()}")
    pool.assert_matches_v5()
    print("assert_matches_v5: OK")
    lib = load_pool_library()
    sampler = PaletteSampler(lib)
    for st, names in sorted(sampler.strata.items()):
        print(f"stratum {st}: {len(names)} palettes")
    print(f"realized curated fraction of cyclic draws: "
          f"{sampler.report_stratum_mix(rng):.3f} (pool freq 33/615 = 0.054)")
