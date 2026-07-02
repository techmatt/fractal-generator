"""Data assembly + location-disjoint split for the bootstrap palette-preference scorer.

Read-only on the label store and query records. Assembles per-query tiered
candidates into a Dataset whose unit is a *query* (6 candidates forwarded
together), and provides a location-grouped 80/20 split so no location leaks
across the train/val boundary.

Sources (read-only):
- data/queries/labels/coldstart_v2.json  -> per-presentation tiers (good/okay/bad)
- data/queries/coldstart_v2/records/*.json -> per-query location + query_type +
  candidate image paths (relative to the batch dir)

Only the *pass-1* (base/authoritative) tiering is used. The 20 consistency-repeat
second passes (pass==2) are excluded from training/eval — same location+candidates
would double-weight and leak.
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
LABELS_PATH = os.path.join(REPO, "data", "queries", "labels", "coldstart_v2.json")
BATCH_DIR = os.path.join(REPO, "data", "queries", "coldstart_v2")
RECORDS_DIR = os.path.join(BATCH_DIR, "records")

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


def load_queries() -> list[Query]:
    """Assemble pass-1 queries with tiers + location. Read-only."""
    store = json.load(open(LABELS_PATH))
    labels = store["labels"]

    out: list[Query] = []
    for pres_id, entry in labels.items():
        if entry["pass"] != 1:
            continue  # exclude consistency-repeat second passes
        qid = entry["query_id"]
        rec = json.load(open(os.path.join(RECORDS_DIR, f"{qid}.json")))
        loc = rec["location"]
        cands = rec["candidates"]
        tiers_map = entry["tiers"]  # candidate_id -> tier

        image_paths, tiers = [], []
        for ci, cand in enumerate(cands):
            cand_id = f"{qid}_{ci}"
            image_paths.append(os.path.join(BATCH_DIR, cand["image"]))
            tiers.append(tiers_map[cand_id])
        out.append(
            Query(
                query_id=qid,
                query_type=rec["query_type"],
                location_key=_location_key(loc),
                image_paths=image_paths,
                tiers=tiers,
            )
        )
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
