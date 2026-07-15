"""colored_clip soft-spread — greedy maximin selection + soft share-limit penalty.

The soft-selection substrate a within-cell palette allocator runs on. Reads the
palette-ON CLIP descriptors already computed by `tools.curation.colored_clip`
(`data/library_embeddings/embeddings.npz`, `colored_clip` = 564 per-candidate
vectors keyed `location_id/variant_id`) and provides two things:

  * `greedy_maximin` — farthest-point (maximin) ordering of any key subset: start
    from a seed, then repeatedly add the item whose MINIMUM cosine-distance to the
    already-selected set is largest. Returns the ordered selection AND the spread
    curve (that maximin min-distance at each step) — the curve is monotone
    non-increasing and is the diversity budget an allocator reads.
  * `share_penalty` / `marginal_share_penalty` — a SOFT share-limit scoring hook
    parameterized by a cosine threshold tau. Pairs with cosine >= tau incur a soft
    ramp penalty (0 at tau, 1 at cos=1); below tau, zero. tau defaults to None
    (report-only, penalty == 0): it is UNCALIBRATED — the companion
    `soft_spread_calibrate.py` sets up the eyeball that fixes it. Do not hardcode a
    band here.

Scope-agnostic: every function operates on an arbitrary list of keys. `within_cell`
is a convenience that partitions candidates by `color_category` (default the
committed k16 cut) and runs the maximin selection per cell.

    uv run python -m tools.curation.colored_clip_spread     # light unit sanity
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data/library_embeddings/embeddings.npz"
RECORDS = ROOT / "scratchpad/library_records/library_records.jsonl"
DEFAULT_CELL_LEVEL = "k16"   # the committed default ward cut (palette_categories.json)


# --------------------------------------------------------------------------- #
# Store + metadata join.
# --------------------------------------------------------------------------- #
@dataclass
class ColoredStore:
    """The 564 palette-ON CLIP vectors + the record-side metadata join.

    keys    : ordered "location_id/variant_id" keys (== colored_keys in the npz).
    unit    : (N, D) L2-normalized colored_clip rows, aligned to `keys`.
    meta     : key -> {location_id, variant_id, color_category, palette, emitted}.
    """

    keys: list[str]
    unit: np.ndarray
    meta: dict[str, dict]
    _row: dict[str, int]

    def vec(self, key: str) -> np.ndarray:
        return self.unit[self._row[key]]

    def rows(self, keys: list[str]) -> np.ndarray:
        return self.unit[[self._row[k] for k in keys]]

    def location_of(self, key: str) -> str:
        return self.meta[key]["location_id"]

    def cell_of(self, key: str, level: str = DEFAULT_CELL_LEVEL) -> str:
        return cell_label(self.meta[key]["color_category"], level)


def cell_label(color_category: dict | None, level: str = DEFAULT_CELL_LEVEL) -> str:
    """color_category dict -> a single hashable cell tag at the requested cut.

    A candidate flagged `special` (neutral/spectral/outlier) is its own cell — those
    fixed cells sit outside the numbered ward leaves and must not merge with them.
    """
    if not color_category:
        return "unknown"
    special = color_category.get("special")
    if special and special != "chromatic":
        return f"special:{special}"
    val = color_category.get(level)
    return f"{level}:{val}" if val is not None else "unknown"


def load_store(store_path: Path = STORE, records_path: Path = RECORDS) -> ColoredStore:
    """Load colored_clip + join each key to its record-side color metadata."""
    z = np.load(store_path, allow_pickle=True)
    keys = [str(k) for k in z["colored_keys"]]
    emb = z["colored_clip"].astype(np.float64)
    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    meta: dict[str, dict] = {}
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        loc = rec["location_id"]
        for cand in rec["palette_candidates"]:
            key = f"{loc}/{cand['variant_id']}"
            meta[key] = dict(
                location_id=loc,
                variant_id=cand["variant_id"],
                color_category=cand.get("color_category"),
                palette=cand["palette_ref"]["name"],
                emitted=bool(cand.get("emitted", False)),
            )
    # every store key must resolve a record (the producer wrote them in lockstep)
    missing = [k for k in keys if k not in meta]
    if missing:
        raise KeyError(f"{len(missing)} store keys have no record metadata, e.g. {missing[:3]}")
    return ColoredStore(keys=keys, unit=unit, meta=meta,
                        _row={k: i for i, k in enumerate(keys)})


# --------------------------------------------------------------------------- #
# Cosine distance.
# --------------------------------------------------------------------------- #
def cosine_sim_matrix(unit: np.ndarray) -> np.ndarray:
    """Full pairwise cosine SIMILARITY over unit-normalized rows (clamped [-1, 1])."""
    return np.clip(unit @ unit.T, -1.0, 1.0)


def cosine_dist_matrix(unit: np.ndarray) -> np.ndarray:
    """1 - cosine similarity; diagonal forced to 0."""
    d = 1.0 - cosine_sim_matrix(unit)
    np.fill_diagonal(d, 0.0)
    return d


# --------------------------------------------------------------------------- #
# Greedy maximin (farthest-point) selection.
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    """Ordered maximin pick over a key subset.

    order : keys in selection order (order[0] is the seed).
    curve : per-step maximin min-distance. curve[0] is inf (seed has no prior set);
            curve[i] (i>=1) is the min cosine-distance of order[i] to {order[:i]} at
            the moment it was added. Monotone non-increasing over i>=1.
    """

    order: list[str]
    curve: list[float]

    def truncated(self, min_dist: float) -> list[str]:
        """Prefix whose incremental spread stays >= min_dist (the seed always kept)."""
        out = [self.order[0]]
        for k, d in zip(self.order[1:], self.curve[1:]):
            if d < min_dist:
                break
            out.append(k)
        return out


def greedy_maximin(keys: list[str], vecs: np.ndarray, start: str | int | None = None) -> Selection:
    """Farthest-point ordering by cosine distance.

    keys : the candidate subset. vecs : (len(keys), D) unit rows aligned to `keys`
    (i.e. store.rows(keys)). start : seed key, index into `keys`, or None (=> index 0).

    O(N^2) via an incrementally-updated nearest-selected distance vector.
    """
    n = len(keys)
    if n == 0:
        return Selection(order=[], curve=[])
    if n == 1:
        return Selection(order=list(keys), curve=[float("inf")])

    D = cosine_dist_matrix(vecs)
    idx = {k: i for i, k in enumerate(keys)}
    if start is None:
        s0 = 0
    elif isinstance(start, int):
        s0 = start
    else:
        s0 = idx[start]

    order = [s0]
    curve = [float("inf")]
    # nearest-selected distance for every point; +inf for already-selected
    mind = D[s0].copy()
    mind[s0] = -np.inf
    for _ in range(n - 1):
        nxt = int(np.argmax(mind))
        order.append(nxt)
        curve.append(float(mind[nxt]))
        mind = np.minimum(mind, D[nxt])
        mind[nxt] = -np.inf
    return Selection(order=[keys[i] for i in order], curve=curve)


# --------------------------------------------------------------------------- #
# Soft share-limit penalty — the allocator scoring hook (tau UNCALIBRATED).
# --------------------------------------------------------------------------- #
def _soft_ramp(cos: np.ndarray | float, tau: float, power: float) -> np.ndarray | float:
    """0 below tau, then ((cos-tau)/(1-tau))**power ramping to 1 at cos=1."""
    span = max(1.0 - tau, 1e-9)
    frac = np.clip((np.asarray(cos, dtype=float) - tau) / span, 0.0, 1.0)
    return frac ** power


def share_penalty(keys: list[str], store: ColoredStore,
                  tau: float | None = None, power: float = 1.0) -> float:
    """Total soft share-limit penalty over a selected set.

    Sum over all pairs (cos >= tau) of the soft ramp. tau=None => 0 (report-only).
    This is the hook a later allocator adds (weighted) to a diversity objective; it
    is intentionally uncalibrated — soft_spread_calibrate.py fixes tau by eye.
    """
    if tau is None or len(keys) < 2:
        return 0.0
    unit = store.rows(keys)
    sim = cosine_sim_matrix(unit)
    iu = np.triu_indices(len(keys), k=1)
    return float(_soft_ramp(sim[iu], tau, power).sum())


def marginal_share_penalty(candidate: str, selected: list[str], store: ColoredStore,
                           tau: float | None = None, power: float = 1.0) -> float:
    """Soft penalty of ADDING `candidate` to `selected` (sum of ramps vs each member)."""
    if tau is None or not selected:
        return 0.0
    cv = store.vec(candidate)
    sims = store.rows(selected) @ cv
    return float(np.sum(_soft_ramp(sims, tau, power)))


# --------------------------------------------------------------------------- #
# within_cell convenience.
# --------------------------------------------------------------------------- #
def group_by_cell(keys: list[str], store: ColoredStore,
                  level: str = DEFAULT_CELL_LEVEL) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = {}
    for k in keys:
        cells.setdefault(store.cell_of(k, level), []).append(k)
    return cells


def within_cell(keys: list[str], store: ColoredStore,
                level: str = DEFAULT_CELL_LEVEL) -> dict[str, Selection]:
    """Partition `keys` by color_category cell, run maximin selection per cell."""
    out: dict[str, Selection] = {}
    for cell, members in group_by_cell(keys, store, level).items():
        out[cell] = greedy_maximin(members, store.rows(members))
    return out


# --------------------------------------------------------------------------- #
# Light unit sanity.
# --------------------------------------------------------------------------- #
def _sanity():
    W = 66
    print("=" * W)
    print("colored_clip_spread — unit sanity")
    print("=" * W)

    # --- synthetic: 4 axis clusters; maximin must hit all 4 before doubling up ---
    rng = np.random.RandomState(0)
    centers = np.eye(4)
    pts, labels = [], []
    for ci, c in enumerate(centers):
        for _ in range(5):
            v = c + 0.02 * rng.randn(4)
            pts.append(v / np.linalg.norm(v))
            labels.append(ci)
    pts = np.asarray(pts)
    skeys = [f"c{labels[i]}_{i}" for i in range(len(pts))]
    sel = greedy_maximin(skeys, pts, start=0)
    first4 = {labels[skeys.index(k)] for k in sel.order[:4]}
    print(f"synthetic: 4 clusters x 5 pts; first 4 picks span clusters {sorted(first4)} "
          f"-> {'OK' if first4 == {0,1,2,3} else 'FAIL'}")
    # spread curve monotone non-increasing after the seed
    tail = sel.curve[1:]
    mono = all(tail[i] >= tail[i + 1] - 1e-9 for i in range(len(tail) - 1))
    print(f"           spread curve monotone non-increasing: {'OK' if mono else 'FAIL'}")
    print(f"           curve[1:5]={[round(x,3) for x in sel.curve[1:5]]} "
          f"... curve[-1]={sel.curve[-1]:.3f}")

    # soft penalty: identical points at cos=1 penalized, orthogonal not; tau=None => 0
    dup = np.stack([centers[0], centers[0], centers[1]])
    dk = ["a", "b", "c"]
    st_syn = ColoredStore(keys=dk, unit=dup, meta={}, _row={k: i for i, k in enumerate(dk)})
    p_none = share_penalty(dk, st_syn, tau=None)
    p_hi = share_penalty(dk, st_syn, tau=0.9)     # a,b identical (cos 1) -> penalized
    p_lo = share_penalty(["a", "c"], st_syn, tau=0.9)  # orthogonal -> 0
    print(f"soft penalty: tau=None -> {p_none} (report-only), "
          f"dup-pair tau=0.9 -> {p_hi:.3f} (>0), orthogonal -> {p_lo:.3f} "
          f"-> {'OK' if p_none==0 and p_hi>0 and p_lo==0 else 'FAIL'}")

    # --- one real cell from the store ---
    try:
        store = load_store()
    except Exception as e:  # noqa: BLE001
        print(f"\nreal cell: store unavailable ({e}) — skipping")
        print("=" * W)
        return
    cells = group_by_cell(store.keys, store)
    biggest = max(cells, key=lambda c: len(cells[c]))
    members = cells[biggest]
    rsel = greedy_maximin(members, store.rows(members))
    print(f"\nreal cell [{biggest}]: {len(members)} candidates across "
          f"{len({store.location_of(k) for k in members})} locations")
    print(f"           maximin curve[1]={rsel.curve[1]:.3f} (most-diverse add) "
          f"curve[-1]={rsel.curve[-1]:.3f} (tightest add)")
    within = within_cell(store.keys, store)
    print(f"           within_cell partitioned {len(store.keys)} keys into "
          f"{len(within)} cells")
    print("=" * W)


if __name__ == "__main__":
    _sanity()
