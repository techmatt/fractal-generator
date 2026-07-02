"""Data assembly + location-disjoint split for the bootstrap palette-preference scorer.

Read-only on the label store and query records. Assembles per-query tiered
candidates into a Dataset whose unit is a *query* (6 candidates forwarded
together), and provides a location-grouped 80/20 split so no location leaks
across the train/val boundary.

Sources (read-only) — v2 unions TWO batches, each a (batch dir, label store) pair:
- coldstart_v2  (data/queries/coldstart_v2/ + data/queries/labels/coldstart_v2.json)
  raw 6-of-draw pools: wide good/bad spread.
- warmstart_v1  (data/queries/warmstart_v1/ + data/queries/labels/warmstart_v1.json)
  v1-concentrated queries: fine good-vs-good top-end resolution.
The two batches are location-disjoint by construction. Each label store maps
per-presentation tiers (good/okay/bad); each record carries location +
query_type + candidate image paths (relative to its batch dir).

Only the *pass-1* (base/authoritative) tiering is used. The 20 consistency-repeat
second passes (pass==2) in each batch are excluded from training/eval — same
location+candidates would double-weight and leak.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

# ---- paths ---------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass(frozen=True)
class BatchSpec:
    name: str
    batch_dir: str      # absolute; records/ + images/ live under here
    labels_path: str    # absolute; the label store JSON

    @property
    def records_dir(self) -> str:
        return os.path.join(self.batch_dir, "records")


def _batch(name: str) -> BatchSpec:
    return BatchSpec(
        name=name,
        batch_dir=os.path.join(REPO, "data", "queries", name),
        labels_path=os.path.join(REPO, "data", "queries", "labels", f"{name}.json"),
    )


# The two batches v2 unions. coldstart first (its split is reused verbatim from v1).
COLDSTART = _batch("coldstart_v2")
WARMSTART = _batch("warmstart_v1")
BATCHES = [COLDSTART, WARMSTART]

# Explicit, auditable exclusion list (NOT a silent filter). q002_0040 is the single
# eff@10==1 collapse query in warmstart; drop it at query-load so it contributes zero
# pairs. Its label file is left untouched.
EXCLUDED_QUERIES = ["q002_0040"]

# Back-compat single-batch defaults (surfacing_eval.py + any v1 caller of
# load_queries()). Point at coldstart, matching the pre-v2 behavior.
LABELS_PATH = COLDSTART.labels_path
BATCH_DIR = COLDSTART.batch_dir
RECORDS_DIR = COLDSTART.records_dir

# ---- constants (data side) ----------------------------------------------
TIER_RANK = {"bad": 0, "okay": 1, "good": 2}
# Cross-tier ordered pair types (lo_tier, hi_tier) -> name
PAIR_TYPES = {
    ("bad", "good"): "good_vs_bad",
    ("okay", "good"): "good_vs_okay",
    ("bad", "okay"): "okay_vs_bad",
}
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Query:
    query_id: str
    query_type: str          # palette | param | joint
    location_key: str        # family + c_re + c_im + cx + cy + fw
    image_paths: list[str]   # absolute, in candidate order 0..5
    tiers: list[str]         # tier per candidate, aligned with image_paths
    batch: str = COLDSTART.name  # source batch name (for per-batch decomposition)


def _location_key(loc: dict) -> str:
    # Full location identity: family + Julia c + viewport. String-exact on the
    # decimal fields (they are stored as decimal strings; never float them).
    return "|".join(
        [
            str(loc.get("family")),
            str(loc.get("c_re")),
            str(loc.get("c_im")),
            str(loc.get("cx")),
            str(loc.get("cy")),
            str(loc.get("fw")),
        ]
    )


def load_batch_queries(spec: BatchSpec, exclude=()) -> list[Query]:
    """Assemble one batch's pass-1 queries with tiers + location. Read-only.
    `exclude` is a collection of query_ids to drop at load (contribute zero pairs)."""
    exclude = set(exclude)
    store = json.load(open(spec.labels_path))
    labels = store["labels"]

    out: list[Query] = []
    for pres_id, entry in labels.items():
        if entry["pass"] != 1:
            continue  # exclude consistency-repeat second passes
        qid = entry["query_id"]
        if qid in exclude:
            continue  # named exclusion (see EXCLUDED_QUERIES)
        rec = json.load(open(os.path.join(spec.records_dir, f"{qid}.json")))
        loc = rec["location"]
        cands = rec["candidates"]
        tiers_map = entry["tiers"]  # candidate_id -> tier

        image_paths, tiers = [], []
        for ci, cand in enumerate(cands):
            cand_id = f"{qid}_{ci}"
            image_paths.append(os.path.join(spec.batch_dir, cand["image"]))
            tiers.append(tiers_map[cand_id])
        out.append(
            Query(
                query_id=qid,
                query_type=rec["query_type"],
                location_key=_location_key(loc),
                image_paths=image_paths,
                tiers=tiers,
                batch=spec.name,
            )
        )
    return out


def load_queries() -> list[Query]:
    """Back-compat: coldstart-only pass-1 queries (the pre-v2 single-batch loader)."""
    return load_batch_queries(COLDSTART)


def load_combined_queries() -> list[Query]:
    """v2 union: every batch in BATCHES, pass-1 only, with EXCLUDED_QUERIES dropped."""
    out: list[Query] = []
    for spec in BATCHES:
        out.extend(load_batch_queries(spec, exclude=EXCLUDED_QUERIES))
    return out


def cross_tier_pairs(tiers: list[str]) -> list[tuple[int, int]]:
    """Return ordered (hi_idx, lo_idx) candidate index pairs where hi tier > lo tier.
    Within-tier pairs (ties) are dropped."""
    pairs = []
    n = len(tiers)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if TIER_RANK[tiers[i]] > TIER_RANK[tiers[j]]:
                pairs.append((i, j))
    return pairs


def pair_type_name(hi_tier: str, lo_tier: str) -> str:
    return PAIR_TYPES[(lo_tier, hi_tier)]


# ---- location-disjoint split --------------------------------------------
def split_by_location(queries: list[Query], val_frac: float, seed: int):
    """Assign whole locations to train/val. Stratify by a per-location primary
    query-type so type proportions hold roughly across the boundary. Returns
    (train_queries, val_queries, manifest_dict)."""
    import random

    rng = random.Random(seed)

    loc2queries: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        loc2queries[q.location_key].append(q)

    # Per-location primary type: fixed priority so multi-type locations are
    # deterministic (joint > param > palette). Used only for stratification.
    prio = {"joint": 0, "param": 1, "palette": 2}

    def primary_type(qs):
        return sorted({q.query_type for q in qs}, key=lambda t: prio[t])[0]

    strata: dict[str, list[str]] = defaultdict(list)
    for lk, qs in loc2queries.items():
        strata[primary_type(qs)].append(lk)

    train_locs, val_locs = set(), set()
    for t, locs in strata.items():
        locs = sorted(locs)  # deterministic before shuffle
        rng.shuffle(locs)
        n_val = round(len(locs) * val_frac)
        val_locs.update(locs[:n_val])
        train_locs.update(locs[n_val:])

    train_q = [q for q in queries if q.location_key in train_locs]
    val_q = [q for q in queries if q.location_key in val_locs]

    # zero-overlap proof
    overlap = train_locs & val_locs
    assert not overlap, f"LOCATION LEAK: {overlap}"

    def type_breakdown(qs):
        return dict(Counter(q.query_type for q in qs))

    manifest = {
        "seed": seed,
        "val_frac": val_frac,
        "n_locations_total": len(loc2queries),
        "n_locations_train": len(train_locs),
        "n_locations_val": len(val_locs),
        "n_queries_train": len(train_q),
        "n_queries_val": len(val_q),
        "location_overlap_count": len(overlap),
        "type_breakdown_train": type_breakdown(train_q),
        "type_breakdown_val": type_breakdown(val_q),
        "train_locations": sorted(train_locs),
        "val_locations": sorted(val_locs),
        "location_to_queries": {lk: sorted(q.query_id for q in qs) for lk, qs in loc2queries.items()},
    }
    return train_q, val_q, manifest


# ---- combined (matched) split -------------------------------------------
def split_combined(queries: list[Query], val_frac: float, seed: int, v1_manifest_path: str):
    """Matched combined split (the interpretability seam):

    - Coldstart locations: reuse v1's assignment VERBATIM from v1/split_manifest.json,
      so v2's coldstart-val is a matched comparison to v1 (same val locations).
    - Warmstart locations: fresh location-disjoint stratified 80/20 (split_by_location,
      same logic v1 used), seed 0.
    - Combined: train = coldstart-train u warmstart-train; val = coldstart-val u
      warmstart-val. Asserts zero train/val location overlap across the FULL combined set.

    Returns (train_q, val_q, manifest).
    """
    v1 = json.load(open(v1_manifest_path))
    cold_train_locs = set(v1["train_locations"])
    cold_val_locs = set(v1["val_locations"])

    cold_q = [q for q in queries if q.batch == COLDSTART.name]
    warm_q = [q for q in queries if q.batch == WARMSTART.name]

    # coldstart: assign each query by its location's v1 membership (verbatim reuse)
    cold_train, cold_val, unknown = [], [], set()
    for q in cold_q:
        if q.location_key in cold_train_locs:
            cold_train.append(q)
        elif q.location_key in cold_val_locs:
            cold_val.append(q)
        else:
            unknown.add(q.location_key)
    assert not unknown, f"COLDSTART LOC NOT IN v1 MANIFEST: {unknown}"

    # warmstart: fresh stratified location-disjoint 80/20, seed 0
    warm_train, warm_val, warm_manifest = split_by_location(warm_q, val_frac, seed)

    train_q = cold_train + warm_train
    val_q = cold_val + warm_val

    # zero-overlap proof across the FULL combined set (catches any cross-batch leak too)
    train_locs = {q.location_key for q in train_q}
    val_locs = {q.location_key for q in val_q}
    overlap = train_locs & val_locs
    assert not overlap, f"LOCATION LEAK (combined): {overlap}"

    def type_breakdown(qs):
        return dict(Counter(q.query_type for q in qs))

    def per_batch(qs, name):
        b = [q for q in qs if q.batch == name]
        return {
            "n_queries": len(b),
            "n_locations": len({q.location_key for q in b}),
            "type_breakdown": type_breakdown(b),
            "locations": sorted({q.location_key for q in b}),
        }

    manifest = {
        "design": "matched combined: coldstart reuses v1 split verbatim; warmstart fresh 80/20",
        "seed": seed,
        "val_frac": val_frac,
        "v1_manifest_reused": os.path.relpath(v1_manifest_path, REPO).replace("\\", "/"),
        "excluded_queries": list(EXCLUDED_QUERIES),
        "n_locations_train": len(train_locs),
        "n_locations_val": len(val_locs),
        "n_queries_train": len(train_q),
        "n_queries_val": len(val_q),
        "location_overlap_count": len(overlap),
        "type_breakdown_train": type_breakdown(train_q),
        "type_breakdown_val": type_breakdown(val_q),
        "coldstart_train": per_batch(train_q, COLDSTART.name),
        "coldstart_val": per_batch(val_q, COLDSTART.name),
        "warmstart_train": per_batch(train_q, WARMSTART.name),
        "warmstart_val": per_batch(val_q, WARMSTART.name),
        "train_locations": sorted(train_locs),
        "val_locations": sorted(val_locs),
    }
    return train_q, val_q, manifest


# ---- transforms ----------------------------------------------------------
def build_transform(train: bool):
    """Squash-resize to INPUT_SIZE, ImageNet normalize. Geometric-only aug on
    train (h/v flip). NO photometric/color aug — color IS the label signal."""
    ops = [T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    if train:
        ops += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]
    ops += [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


class QueryDataset(Dataset):
    """One item = one query: 6 candidate images + their tier ranks."""

    def __init__(self, queries: list[Query], train: bool):
        self.queries = queries
        self.tf = build_transform(train)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        imgs = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in q.image_paths])
        ranks = torch.tensor([TIER_RANK[t] for t in q.tiers], dtype=torch.long)
        return imgs, ranks, idx


def collate_queries(batch):
    imgs = torch.stack([b[0] for b in batch])   # [B, 6, 3, H, W]
    ranks = torch.stack([b[1] for b in batch])  # [B, 6]
    idxs = torch.tensor([b[2] for b in batch])
    return imgs, ranks, idxs
