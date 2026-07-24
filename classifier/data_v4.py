"""Location-level dataset + sampling weights for the v4 location-quality model.

v4 trains on the **precomputed augmentation cache** (`data/v4/cache_manifest.jsonl`):
42 cached renders per base location across palette x scale x shift x AA axes. The
training unit is the **base location** (not the render): `__getitem__` draws **one
of the 42 cached renders uniformly** (epoch-varying, reproducible) and applies the
**v3 on-the-fly transform** (`data.Transform`, train=True: stretch->384x224, <=5%
border crop, h/v flips, +-3% brightness/contrast, JPEG-q 85-95, NO hue/sat — palette
is already a cache axis).

Everything is keyed off the manifest (incl. `fractal_type`) with **no Mandelbrot
assumption** anywhere — the Julia semi-final retrain points the same loader at a
cache that has Julia rows and reuses this verbatim.

The deploy-canonical eval view is the single render with
`palette==twilight_shifted, aa_level==antialiased(ss4), scale==1.0, shift_id==center`;
`palette_renders` returns the 6 ss4/center/scale-1.0 renders (one per palette) for the
palette-invariance test; `aa_twin` returns the aliased twilight/1.0/center render for
the AA-invariance spot-check.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
# Regenerable aug_cache JPGs were relocated out of the working tree; the manifest
# still stores repo-relative paths, so route them through the shared resolver.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "corpus"))
from artifacts import resolve as resolve_artifact  # noqa: E402

DEFAULT_CACHE = ROOT / "data" / "v4" / "cache_manifest.jsonl"

# Deploy-canonical axis selection (neutral palette / ss4 / canonical scale / center).
NEUTRAL_PALETTE = "twilight_shifted"
CANON_SCALE = 1.0
CANON_SHIFT = "center"


@dataclass
class Render:
    path: Path
    palette: str
    palette_family: str
    scale: float
    shift_id: str
    aa_level: str  # "aliased" | "antialiased"


@dataclass
class Loc:
    location_id: int
    label: int          # raw 1/2/3
    split: str          # "train" | "eval"
    group_id: int       # neighborhood id (hot-spot equalization unit)
    source: str
    biased: bool        # True == v3-mined (down-weighted in the sampler)
    fractal_type: str
    renders: list[Render] = field(default_factory=list)

    # --- eval view selectors (read straight off the cache axes) ---
    def _pick(self, palette=None, aa=None, scale=None, shift=None):
        for r in self.renders:
            if palette is not None and r.palette != palette: continue
            if aa is not None and r.aa_level != aa: continue
            if scale is not None and r.scale != scale: continue
            if shift is not None and r.shift_id != shift: continue
            yield r

    def canonical(self) -> Render:
        """Deploy-canonical: neutral palette, ss4, canonical scale, center."""
        got = list(self._pick(NEUTRAL_PALETTE, "antialiased", CANON_SCALE, CANON_SHIFT))
        assert len(got) == 1, f"loc {self.location_id}: {len(got)} canonical renders"
        return got[0]

    def palette_renders(self) -> list[Render]:
        """The 6 ss4 renders (one per palette), canonical scale + center — the
        palette-invariance battery."""
        got = list(self._pick(aa="antialiased", scale=CANON_SCALE, shift=CANON_SHIFT))
        assert len(got) == 6, f"loc {self.location_id}: {len(got)} ss4 palette renders"
        return sorted(got, key=lambda r: r.palette)

    def aa_twin(self) -> Render:
        """Aliased counterpart of the canonical view (neutral/scale1.0/center)."""
        got = list(self._pick(NEUTRAL_PALETTE, "aliased", CANON_SCALE, CANON_SHIFT))
        assert len(got) == 1, f"loc {self.location_id}: {len(got)} aliased twins"
        return got[0]


def load_locations(cache_path: Path = DEFAULT_CACHE,
                   verify_paths: bool = True) -> list[Loc]:
    """Group the cache manifest into base locations (42 renders each)."""
    by_id: dict[int, Loc] = {}
    for line in Path(cache_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lid = int(r["location_id"])
        loc = by_id.get(lid)
        if loc is None:
            loc = by_id[lid] = Loc(
                location_id=lid, label=int(r["label"]), split=r["split"],
                group_id=int(r["group_id"]), source=r["source"],
                biased=bool(r["biased"]), fractal_type=r["fractal_type"],
            )
        else:  # sanity: per-location metadata must agree across its 42 rows
            assert loc.label == int(r["label"]) and loc.split == r["split"] \
                and loc.group_id == int(r["group_id"]) and loc.biased == bool(r["biased"]), \
                f"loc {lid}: inconsistent per-location metadata in cache"
        loc.renders.append(Render(
            path=resolve_artifact(r["path"]), palette=r["palette"],
            palette_family=r["palette_family"], scale=float(r["scale"]),
            shift_id=r["shift_id"], aa_level=r["aa_level"],
        ))
    locs = [by_id[k] for k in sorted(by_id)]
    if verify_paths:
        missing = [str(rr.path) for lc in locs for rr in lc.renders if not rr.path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cache JPGs missing, e.g. {missing[0]}")
    return locs


# --------------------------------------------------------------------------- #
# Training dataset: one item per location, uniform-over-42 render draw.
# --------------------------------------------------------------------------- #
class LocationDataset(Dataset):
    """Yields (tensor, raw_label_1_2_3, loc_index). Each __getitem__ draws ONE of
    the location's 42 cached renders uniformly and applies `transform`. Render
    choice + augmentation are reproducible but **epoch-varying** (call
    `set_epoch` each epoch) so the 42-render cache is actually exercised."""

    def __init__(self, locs: list[Loc], transform, seed: int = 0):
        self.locs = locs
        self.transform = transform  # data.Transform(train=True)
        self.base_seed = seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, i):
        loc = self.locs[i]
        # reproducible per (seed, epoch, index): varies the drawn render + aug each epoch.
        rng = random.Random((self.base_seed * 2_654_435_761
                             + self.epoch * 1_000_003 + i) & 0xFFFFFFFFFFFF)
        rnd = loc.renders[rng.randrange(len(loc.renders))]
        with Image.open(rnd.path) as im:
            im.load()
            img = im.convert("RGB")
        t = self.transform(img, rng)
        return t, loc.label, i


# --------------------------------------------------------------------------- #
# Sampling weights — multiplicative, per training location.
#   weight = w_class(label) x w_group(loc) x w_source(biased)
#     w_group  = 1 / train_group_size                 (neighborhood equalization)
#     w_source = beta if biased else 1.0              (v3-mined down-weight)
#     w_class  = 1 / sqrt(N_locations_in_class)        (softened class balance)
#
# ORDERING: w_class is a PURE per-class scalar applied last, so it scales each
# class uniformly and can lift the rare good class WITHOUT touching the
# unbiased/biased ratio inside a class — the source down-weight cannot be
# laundered back out. Verified by the effective-mass table.
# --------------------------------------------------------------------------- #
def compute_sampler_weights(train_locs: list[Loc], beta: float = 0.4,
                            class_balance: str = "sqrt"):
    """Returns (weights float64 tensor aligned to train_locs, mass_table dict)."""
    from collections import Counter
    group_size = Counter(l.group_id for l in train_locs)
    class_count = Counter(l.label for l in train_locs)

    def w_group(l): return 1.0 / group_size[l.group_id]
    def w_source(l): return beta if l.biased else 1.0
    if class_balance == "sqrt":
        w_class = {c: 1.0 / np.sqrt(n) for c, n in class_count.items()}
    elif class_balance == "inv":
        w_class = {c: 1.0 / n for c, n in class_count.items()}
    else:
        raise ValueError(class_balance)

    base = np.array([w_group(l) * w_source(l) for l in train_locs])          # group x source
    full = np.array([w_class[l.label] * w_group(l) * w_source(l) for l in train_locs])
    weights = torch.tensor(full, dtype=torch.double)

    # --- effective sampled-mass artifact (probability-normalized so it sums to 1) ---
    prob = full / full.sum()
    cells, percell, n_cell = {}, {}, {}
    for l, p, b in zip(train_locs, prob, base):
        key = f"label{l.label}|{'biased' if l.biased else 'unbiased'}"
        cells[key] = cells.get(key, 0.0) + float(p)          # realized sampled mass (fraction)
        percell[key] = percell.get(key, 0.0) + float(p)
        n_cell[key] = n_cell.get(key, 0) + 1
    mean_per_loc = {k: cells[k] / n_cell[k] for k in cells}   # mean per-location sampled mass
    mass_table = {
        "beta": beta, "class_balance": class_balance,
        "class_count": {int(k): int(v) for k, v in sorted(class_count.items())},
        "w_class": {int(k): float(v) for k, v in sorted(w_class.items())},
        "sampled_mass_fraction": {k: cells[k] for k in sorted(cells)},
        "n_locations": {k: n_cell[k] for k in sorted(n_cell)},
        "mean_mass_per_location": {k: mean_per_loc[k] for k in sorted(mean_per_loc)},
    }
    return weights, mass_table


def make_weighted_sampler(train_locs: list[Loc], beta: float = 0.4,
                          class_balance: str = "sqrt"):
    weights, mass_table = compute_sampler_weights(train_locs, beta, class_balance)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_locs), replacement=True)
    return sampler, mass_table


def hist(locs: list[Loc]) -> dict[int, int]:
    from collections import Counter
    c = Counter(l.label for l in locs)
    return {k: c.get(k, 0) for k in (1, 2, 3)}
