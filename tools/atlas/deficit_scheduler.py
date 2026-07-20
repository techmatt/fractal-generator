#!/usr/bin/env python
"""deficit_scheduler.py — family-level (cross-partition) deficit scheduler for the
steered frontier (v1).

WHY. Campaign 1 proved the frontier's single global priority queue lets raw p_good
allocate ACROSS families — a comparison the classifier is not calibrated for (a family's
mean p_good is *negatively* correlated with its human good-rate). p_good's certified role
ends at the family boundary: it steers and floors WITHIN a partition only. Cross-family
allocation therefore has to become an explicit scheduler, and this is it.

HARD SCOPE INVARIANT. **No p_good value is ever compared across partitions anywhere in
this module.** The within-partition priority (E[ord]+Gumbel-dup-novelty+beta*depth) is
computed by the caller and only ever sorts nodes of ONE partition. The cross-partition
choice here uses ONLY per-partition DEFICITS and PRICES. `choose_partition` is a pure
function of (deficits, prices, capped, servable) precisely so this is testable and
structurally guaranteed (test_deficit_scheduler.py).

MECHANICS (spec: prompts/deficit_scheduler.md).
 1. Per-partition sub-queues. The existing priority formula is unchanged within a
    partition; the caller keeps one frontier list keyed by `partition` and pops the
    top-B of whichever partition this scheduler names.
 2. Each batch, serve the partition with the largest PRICE-WEIGHTED deficit
    (deficit / price = deficit per unit expected cost), with a small exploration floor
    so no partition with remaining demand starves on a stale price.
 3. Deficit is denominated in DISTINCT LOOKS, not admissions. `DistinctLookTally` keeps a
    per-partition CLIP-embedding set (the library morph recipe, embedded by the caller);
    an admission counts iff its max cosine vs that partition's admitted-look set is < 0.974.
 4. Prices = active-minutes per distinct look, per partition. Seeded from a config file,
    updated online (EMA). A partition that burns `cap_minutes` of active time with zero new
    distinct looks is capped (demand redistributed); caps re-open on resume/config.
 5. Target = `data/emission/target_measure.json` projected to per-type (== per-partition)
    marginals over feasible cells. The order book; no separate discovery-side target file.
 6. Julia twins are BOUGHT, not popped: julia:X demand routes into (a) the root family mix
    and (b) willingness to spend c-plane X expansions on its behalf (see JULIA ROUTING).
 7. Root draws are deficit-aware under the same (twin-inclusive) rule.

fractal_type == the emission cell's first axis == the ledger `family` == our `partition`
(mandelbrot / multibrot{3,4,5} / julia:{...}); the projection reuses `emission.cells`
verbatim so the deficit definition is one shared, unit-tested code path with the colorizer.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.emission import cells as C  # noqa: E402  (pure; no torch)

# ------------------------------------------------------------------------- #
# Defaults / paths.
# ------------------------------------------------------------------------- #
DEFAULT_TARGET_PATH = ROOT / "data" / "emission" / "target_measure.json"
DEFAULT_PRICES_PATH = ROOT / "data" / "atlas" / "scheduler_prices.json"
INTAKE_ARTIFACT = ROOT / "out" / "emission" / "campaign1" / "intake.json"
INTAKE_EMB_DIR = ROOT / "out" / "emission" / "campaign1" / "embs"

NEAR_DUP_THRESHOLD = 0.974   # distinct-look cosine knee (== emission/descriptor)
EMB_DIM = 768                # CLIP vit_base_patch16_clip_224.openai

# price-model / scheduling defaults (overridable in the prices config)
SEED_PRICE_MIN = 3.0         # neutral per-partition seed: active-minutes / distinct look
PRICE_EMA = 0.30            # online price smoothing (weight on the newest per-look sample)
CAP_MINUTES = 20.0          # attempt cap: active minutes with zero new looks before capping
EXPLORE_FLOOR = 0.10        # prob. of a uniform draw among partitions with remaining demand
JULIA_ROUTE_GAIN = 1.0      # weight of a twin's deficit folded into its c-plane parent


# ------------------------------------------------------------------------- #
# Partition topology helpers (mirror production_seeder grammar; kept local so this module
# imports no torch-side code).
# ------------------------------------------------------------------------- #
def is_julia(partition: str) -> bool:
    return partition.startswith("julia:")


def cplane_of(partition: str) -> str | None:
    """The c-plane parent a julia partition descends off (julia:X -> X); None for c-plane."""
    return partition.split(":", 1)[1] if is_julia(partition) else None


def julia_partition(cplane: str) -> str:
    return f"julia:{cplane}"


# ------------------------------------------------------------------------- #
# 5. Order book: target_measure.json projected to per-partition marginals.
# ------------------------------------------------------------------------- #
def load_target(path: Path | str | None) -> C.TargetMeasure:
    path = Path(path) if path else DEFAULT_TARGET_PATH
    cfg = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
    return C.TargetMeasure.from_config(cfg)


def load_observed_type_cluster(partitions: list[str]) -> list[tuple]:
    """(type, cluster) pairs anchoring the feasible grid. Prefer the emission intake artifact
    (real within-family morph clusters); else fall back to one cluster per tracked partition
    (uniform-over-present base that overrides then skew). Restricted to `partitions` so the
    order book covers exactly the run's tracked families."""
    pset = set(partitions)
    if INTAKE_ARTIFACT.exists():
        tags = json.loads(INTAKE_ARTIFACT.read_text(encoding="utf-8")).get("cluster_tags", {})
        obs = set()
        for tag in tags.values():                 # tag == "<type>#<k>"
            t = tag.rsplit("#", 1)[0]
            if t in pset:
                obs.add((t, tag))
        # any tracked partition the artifact never observed still gets a singleton anchor,
        # so it has non-zero target and can be served.
        for p in partitions:
            if not any(t == p for (t, _c) in obs):
                obs.add((p, f"{p}#0"))
        return sorted(obs)
    return [(p, f"{p}#0") for p in partitions]


def project_type_marginals(target: C.TargetMeasure, observed_type_cluster: list[tuple],
                           partitions: list[str],
                           flavors: list[str] | None = None,
                           styles: list[str] | None = None) -> dict:
    """Project the joint target measure down to per-TYPE marginals over feasible cells,
    normalized over `partitions`.

    marginal(type) ∝ MEAN cell weight over that type's feasible cells (then normalized).

    Per-TYPE normalization by design (campaign-2 preflight fix): a type's target share is set
    by its own multiplier, INDEPENDENT of how many morph clusters it currently occupies. The
    naive `Σ weight(cell)` sum scales a type's marginal with its cluster COUNT, which is
    backwards for a discovery ORDER BOOK — current cluster count is *occupancy* and belongs on
    the deficit's pool side, not the target side (e.g. mandelbrot's 102 observed clusters would
    otherwise swamp julia:mandelbrot's 4 regardless of the 2.5× vs 1.2× multipliers, inverting
    the intended julia-heavy order). Dividing the sum by the type's cell count removes the count
    entirely: for a pure type-level override the mean equals the multiplier exactly; a
    cluster-level override is honoured as an average over that type's clusters.

    palette_flavor / render_style are FREE choices available to every (type, cluster), so an
    override on them multiplies the same cell FRACTION for every type and cancels under both the
    mean and the final normalization; we therefore project over a single sentinel flavor/style
    unless real lists are supplied."""
    flavors = list(flavors) if flavors else ["_"]
    styles = list(styles) if styles else ["_"]
    feasible = C.build_feasible_cells(observed_type_cluster, flavors, styles)
    wsum = Counter()
    wcnt = Counter()
    for cell in feasible:
        wsum[cell[0]] += target.weight(cell)     # cell[0] == fractal_type == partition
        wcnt[cell[0]] += 1
    weights = {p: (wsum[p] / wcnt[p] if wcnt.get(p) else 0.0) for p in partitions}
    tot = sum(weights.values())
    if tot <= 0:                                  # degenerate: fall back to uniform
        return {p: 1.0 / len(partitions) for p in partitions}
    return {p: w / tot for p, w in weights.items()}


# ------------------------------------------------------------------------- #
# Library-look seed: the campaign-1 intake's per-cluster medoid embeddings, grouped by
# partition (== family). Deficits must measure LIBRARY-WIDE scarcity, not run-local scarcity,
# so a fresh run's distinct-look tally is pre-loaded with the looks the library already holds
# (and their embeddings become dedup memory: a new admission near a known library look is not
# counted as new). Same CLIP recipe (morph_gray -> vit_base_patch16_clip_224.openai, 768-d)
# emission clusters at cos 0.974, so the seed is metric-consistent with the tally.
# ------------------------------------------------------------------------- #
def load_library_seed_embeddings(intake_path: Path | None = None,
                                 emb_dir: Path | None = None) -> dict[str, np.ndarray]:
    """partition -> (N, 768) float32 medoid embeddings from the campaign-1 intake. One medoid
    (cluster founder) per distinct look. Returns {} if the intake artifact is absent (seeding
    then no-ops, and the run starts at run-local scarcity — logged by the caller)."""
    ip = Path(intake_path) if intake_path else INTAKE_ARTIFACT
    ed = Path(emb_dir) if emb_dir else INTAKE_EMB_DIR
    if not ip.exists():
        return {}
    intake = json.loads(ip.read_text(encoding="utf-8"))
    medoid_id = intake.get("medoid_id", {})          # cluster_tag "<family>#<k>" -> location id
    by_part: dict[str, list] = defaultdict(list)
    for tag, loc_id in medoid_id.items():
        part = tag.rsplit("#", 1)[0]                  # partition == family
        p = ed / f"{loc_id}.npy"
        if not p.exists():
            continue
        e = np.load(p).astype(np.float32).reshape(-1)
        if e.shape[0] != EMB_DIM:
            continue
        by_part[part].append(e / (np.linalg.norm(e) + 1e-9))
    return {p: np.stack(v).astype(np.float32) for p, v in by_part.items() if v}


# ------------------------------------------------------------------------- #
# 3. Distinct-look tally — per-partition CLIP-embedding set + max-cosine gate.
# ------------------------------------------------------------------------- #
class DistinctLookTally:
    """Per-partition set of admitted-look embeddings (L2-normalized, N x 768). An admission's
    embedding counts as a NEW distinct look iff its max cosine vs that partition's existing
    set is < NEAR_DUP_THRESHOLD; on a distinct look the embedding joins the set. Pure numpy
    (embeddings produced by the caller) so it is unit-testable with hand-built vectors, and
    serialized to an npz (per-partition matrices) for lossless resume."""

    def __init__(self, path: Path, threshold: float = NEAR_DUP_THRESHOLD):
        self.path = Path(path)
        self.threshold = float(threshold)
        self.sets: dict[str, np.ndarray] = {}    # partition -> (N, 768) float32
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            for k in z.files:
                self.sets[k] = z[k].astype(np.float32)

    def count(self, partition: str) -> int:
        m = self.sets.get(partition)
        return 0 if m is None else int(m.shape[0])

    def counts(self) -> dict:
        return {p: int(m.shape[0]) for p, m in self.sets.items()}

    def total(self) -> int:
        return sum(int(m.shape[0]) for m in self.sets.values())

    def add(self, partition: str, emb) -> bool:
        """Test-and-add: True (and appends) iff `emb` is a new distinct look for `partition`."""
        e = np.asarray(emb, np.float32).reshape(1, EMB_DIM)
        e = e / (np.linalg.norm(e) + 1e-9)
        m = self.sets.get(partition)
        if m is not None and m.shape[0]:
            cos_max = float((m @ e[0]).max())
            if cos_max >= self.threshold:
                return False
        self.sets[partition] = e if m is None else np.concatenate([m, e], axis=0)
        return True

    def save(self):
        if not self.sets:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / (self.path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **{p: m for p, m in self.sets.items()})
        import os
        os.replace(tmp, self.path)


# ------------------------------------------------------------------------- #
# 4. Price model — active-minutes per distinct look, per partition, w/ attempt caps.
# ------------------------------------------------------------------------- #
class PriceModel:
    """Per-partition price = active-minutes / distinct look. Seeded from config, updated
    online by an EMA of the minutes spent between successive distinct looks. Attempt cap:
    a partition that accrues `cap_minutes` of active time with zero new distinct looks is
    capped (excluded from serving; its demand redistributes). Caps re-open on resume /
    config reload — nothing here is trusted from a checkpoint that config can override."""

    def __init__(self, partitions: list[str], config: dict | None = None):
        config = config or {}
        seeds = config.get("prices", {})
        self.seed_default = float(config.get("seed_price_min", SEED_PRICE_MIN))
        self.ema = float(config.get("price_ema", PRICE_EMA))
        self.cap_minutes = float(config.get("cap_minutes", CAP_MINUTES))
        self.price = {p: float(seeds.get(p, self.seed_default)) for p in partitions}
        self.min_since_look = {p: 0.0 for p in partitions}   # minutes since last distinct look
        self.min_spent = {p: 0.0 for p in partitions}        # cumulative active minutes
        self.capped: set[str] = set()

    def ensure(self, partition: str):
        if partition not in self.price:
            self.price[partition] = self.seed_default
            self.min_since_look[partition] = 0.0
            self.min_spent[partition] = 0.0

    def charge(self, partition: str, minutes: float):
        """Account `minutes` of active time to `partition`. Trips the attempt cap if the
        partition has now burned cap_minutes with zero new looks. Returns True iff it just
        capped this call."""
        self.ensure(partition)
        self.min_spent[partition] += minutes
        self.min_since_look[partition] += minutes
        if (partition not in self.capped
                and self.min_since_look[partition] >= self.cap_minutes):
            self.capped.add(partition)
            return True
        return False

    def record_look(self, partition: str):
        """A new distinct look landed in `partition`: update its price EMA from the minutes
        spent since the last look, reset the dry-time counter, and un-cap it (it is
        productive again)."""
        self.ensure(partition)
        sample = self.min_since_look[partition]
        if sample > 0.0:
            self.price[partition] = (1 - self.ema) * self.price[partition] + self.ema * sample
        self.min_since_look[partition] = 0.0
        self.capped.discard(partition)

    def reopen_caps(self):
        """Re-open every capped partition (resume / config reload)."""
        for p in list(self.capped):
            self.min_since_look[p] = 0.0
        self.capped.clear()

    def state_dict(self) -> dict:
        return dict(price=self.price, min_since_look=self.min_since_look,
                    min_spent=self.min_spent, capped=sorted(self.capped),
                    ema=self.ema, cap_minutes=self.cap_minutes,
                    seed_default=self.seed_default)

    def load_state(self, d: dict):
        self.price.update({k: float(v) for k, v in d.get("price", {}).items()})
        self.min_since_look.update({k: float(v) for k, v in d.get("min_since_look", {}).items()})
        self.min_spent.update({k: float(v) for k, v in d.get("min_spent", {}).items()})
        self.capped = set(d.get("capped", []))


# ------------------------------------------------------------------------- #
# 2. The cross-partition choice — PURE (deficits/prices only; NO p_good).
# ------------------------------------------------------------------------- #
def choose_partition(deficits: dict, prices: dict, capped: set, servable: set,
                     rng, explore_floor: float = EXPLORE_FLOOR) -> str | None:
    """Name the partition to serve next. Uses ONLY per-partition deficits and prices — never
    any per-node score, never a p_good — so cross-partition p_good comparison is structurally
    impossible (this is the certified boundary). Exploration floor: with prob `explore_floor`
    draw uniformly among servable partitions that still have positive demand, so none starves
    on a stale price.

    Returns the chosen partition, or None if nothing is servable."""
    cand = [p for p in servable if p not in capped]
    if not cand:
        return None
    demand = [p for p in cand if deficits.get(p, 0.0) > 0.0]
    if demand and float(rng.random()) < explore_floor:
        return demand[int(rng.integers(len(demand)))]
    # price-weighted deficit = deficit per unit expected cost.
    def pwd(p):
        price = max(float(prices.get(p, SEED_PRICE_MIN)), 1e-6)
        return deficits.get(p, 0.0) / price
    return max(cand, key=lambda p: (pwd(p), p))   # p tie-break => deterministic


# ------------------------------------------------------------------------- #
# The scheduler object — ties the order book, tally, and prices together and owns the
# cross-partition routing (incl. julia twins). Within-partition ordering is the caller's.
# ------------------------------------------------------------------------- #
class DeficitScheduler:
    def __init__(self, partitions: list[str], run_dir: Path,
                 target_path: Path | str | None = None,
                 prices_path: Path | str | None = None,
                 explore_floor: float = EXPLORE_FLOOR,
                 julia_route_gain: float = JULIA_ROUTE_GAIN):
        self.partitions = list(partitions)
        self.run_dir = Path(run_dir)
        self.explore_floor = float(explore_floor)
        self.julia_route_gain = float(julia_route_gain)

        # order book (per-partition target fraction).
        self.target = load_target(target_path)
        self.observed = load_observed_type_cluster(self.partitions)
        self.target_frac = project_type_marginals(self.target, self.observed, self.partitions)

        # price config (+ scheduling knobs the config may override).
        pcfg = {}
        pp = Path(prices_path) if prices_path else DEFAULT_PRICES_PATH
        if pp.exists():
            pcfg = json.loads(pp.read_text(encoding="utf-8"))
        self.explore_floor = float(pcfg.get("explore_floor", self.explore_floor))
        self.julia_route_gain = float(pcfg.get("julia_route_gain", self.julia_route_gain))
        self.prices = PriceModel(self.partitions, pcfg)

        self.tally = DistinctLookTally(self.run_dir / "distinct_looks.npz")
        # allocation trace (per-batch partition choice + deficits) for the readout.
        self.trace_path = self.run_dir / "scheduler_trace.jsonl"

    # ---- deficit -------------------------------------------------------- #
    def look_frac(self) -> dict:
        tot = self.tally.total()
        if tot <= 0:
            return {p: 0.0 for p in self.partitions}
        return {p: self.tally.count(p) / tot for p in self.partitions}

    def deficits(self) -> dict:
        lf = self.look_frac()
        return {p: self.target_frac.get(p, 0.0) - lf.get(p, 0.0) for p in self.partitions}

    def effective_deficits(self, queue_lens: dict) -> dict:
        """6. Julia routing. A julia:X partition whose OWN queue is empty cannot be popped, so
        its (positive) deficit is folded into its c-plane parent X's effective deficit
        (weighted by julia_route_gain): serving c-plane X fires the julia hook, seeding julia:X
        roots that later become directly poppable. A julia twin that HAS a queue competes on
        its own and is not double-counted. c-plane / already-servable julia deficits pass
        through unchanged. Deficit arithmetic only — no p_good anywhere."""
        base = self.deficits()
        eff = dict(base)
        for jp in self.partitions:
            if not is_julia(jp):
                continue
            if queue_lens.get(jp, 0) > 0:            # directly servable -> competes on its own
                continue
            cp = cplane_of(jp)
            if cp in eff and base[jp] > 0.0:
                eff[cp] = eff[cp] + self.julia_route_gain * base[jp]
        return eff

    # ---- the pop decision ---------------------------------------------- #
    def pick_partition(self, queue_lens: dict, rng) -> str | None:
        """Choose which partition's sub-queue to pop this batch. `queue_lens` maps partition
        -> number of frontier nodes currently in that partition. Servable = non-empty queue.
        Returns the partition name (a c-plane family may be chosen to buy julia twin looks),
        or None if every queue is empty."""
        servable = {p for p, n in queue_lens.items() if n > 0}
        eff = self.effective_deficits(queue_lens)
        return choose_partition(eff, self.prices.price, self.prices.capped, servable, rng,
                                self.explore_floor)

    # ---- 7. deficit-aware root allocation ------------------------------- #
    def root_allocation(self, families: list[str], n_draws: int, rng) -> dict:
        """Split `n_draws` root draws across c-plane `families` proportional to a softmax of
        their price-weighted, twin-inclusive effective deficit (item 7). A family carrying
        julia twin deficit thus draws more roots on its twin's behalf. Returns {family: count}
        summing to `n_draws` (empty queues => all families equally eligible)."""
        # roots are c-plane; fold in each family's julia twin demand (empty-queue routing).
        base = self.deficits()
        scores = {}
        for f in families:
            s = base.get(f, 0.0)
            jp = julia_partition(f)
            if jp in base and base[jp] > 0.0:
                s += self.julia_route_gain * base[jp]
            price = max(self.prices.price.get(f, SEED_PRICE_MIN), 1e-6)
            scores[f] = s / price
        vals = np.array([scores[f] for f in families], dtype=np.float64)
        # range-normalized softmax (scale-free), temperature 0.5; ties -> uniform.
        span = float(vals.max() - vals.min())
        if span <= 1e-12:
            probs = np.full(len(families), 1.0 / len(families))
        else:
            norm = (vals - vals.min()) / span
            ex = np.exp(norm / 0.5)
            probs = ex / ex.sum()
        draws = rng.multinomial(int(n_draws), probs)
        return {f: int(n) for f, n in zip(families, draws)}

    # ---- admission hook ------------------------------------------------- #
    def on_admission(self, partition: str, emb) -> bool:
        """Register an admitted look. Embeds already done by the caller (library morph recipe).
        Returns True iff it was a NEW distinct look (tally + price EMA updated)."""
        distinct = self.tally.add(partition, emb)
        if distinct:
            self.prices.record_look(partition)
        return distinct

    def charge(self, partition: str, minutes: float) -> bool:
        """Account a batch's active time to the served partition (attempt-cap accounting)."""
        return self.prices.charge(partition, minutes)

    def seed_from_library(self, embeddings: dict[str, np.ndarray] | None = None) -> dict:
        """One-time baseline seed of the distinct-look tally from the library's existing looks
        (campaign-1 intake medoids), so deficits measure LIBRARY-WIDE scarcity rather than
        run-local scarcity, and the seeded embeddings become dedup memory (a new admission that
        duplicates a known library look does not count as a new look).

        Resume-safe + idempotent: seeds ONLY when the tally is empty. A resume reloads the
        persisted npz (total > 0) and this is a no-op — the seed is never double-counted, and
        after seeding the tally is persisted immediately so the very first kill can't lose it.
        Restricted to this run's tracked partitions. Returns {partition: seeded_count}."""
        if self.tally.total() > 0:                    # already populated (resume) — never re-seed
            return {}
        if embeddings is None:
            embeddings = load_library_seed_embeddings()
        seeded: dict[str, int] = {}
        for part in self.partitions:                  # only families this run tracks
            mat = embeddings.get(part)
            if mat is None:
                continue
            for e in mat:
                if self.tally.add(part, e):           # dedup at the same 0.974 knee as admissions
                    seeded[part] = seeded.get(part, 0) + 1
        if seeded:
            self.tally.save()                         # persist the baseline before any batch runs
        return seeded

    def log_choice(self, batch: int, chosen: str | None, queue_lens: dict):
        eff = self.effective_deficits(queue_lens)
        rec = dict(batch=batch, chosen=chosen,
                   deficits={p: round(self.deficits()[p], 5) for p in self.partitions},
                   eff_deficits={p: round(eff[p], 5) for p in self.partitions},
                   prices={p: round(self.prices.price[p], 4) for p in self.partitions},
                   looks=self.tally.counts(), capped=sorted(self.prices.capped),
                   queue_lens={p: int(queue_lens.get(p, 0)) for p in self.partitions})
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- state (embedded in the driver's state.json; embeddings in npz) - #
    def state_dict(self) -> dict:
        return dict(partitions=self.partitions, target_frac=self.target_frac,
                    explore_floor=self.explore_floor, julia_route_gain=self.julia_route_gain,
                    prices=self.prices.state_dict())

    def load_state(self, d: dict, reopen_caps: bool = False):
        self.target_frac = {p: float(v) for p, v in d.get("target_frac", {}).items()} \
            or self.target_frac
        self.explore_floor = float(d.get("explore_floor", self.explore_floor))
        self.julia_route_gain = float(d.get("julia_route_gain", self.julia_route_gain))
        self.prices.load_state(d.get("prices", {}))
        if reopen_caps:
            self.prices.reopen_caps()
        # the distinct-look tally reloads from its own npz in __init__.

    def save(self):
        self.tally.save()

    def summary(self) -> dict:
        return dict(target_frac={p: round(v, 4) for p, v in self.target_frac.items()},
                    look_frac={p: round(v, 4) for p, v in self.look_frac().items()},
                    looks=self.tally.counts(), total_looks=self.tally.total(),
                    prices={p: round(v, 3) for p, v in self.prices.price.items()},
                    min_spent={p: round(v, 2) for p, v in self.prices.min_spent.items()},
                    capped=sorted(self.prices.capped),
                    n_observed_cells=len(self.observed))
