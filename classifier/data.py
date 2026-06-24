"""Dataset, geometry, and augmentation for the v1 aesthetic classifier.

The key invariant (see the CC prompt): the **train resize and the deploy resize
are the same deterministic core** — only train wraps it in light geometric +
palette-preserving color augmentation. `build_transform(..., train=False)` is the
exact mirror of `present.rs`'s 1280x720 JPG path, so bias-loop scores match the
trained distribution.

Parity with `src/present.rs`:
  - black gate: accept a crop iff `black_fraction < 0.30` (strict `<`, BLACK_THRESH).
  - source JPGs are 1280x720 q90.
"""
from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "labels" / "location_labels.json"
DEFAULT_MANIFEST = ROOT / "data" / "label_crops" / "loose0_v3" / "manifest.json"

BLACK_THRESH = 0.30  # present.rs: const BLACK_THRESH: f32 = 0.30; accept iff bf < THRESH
SRC_W, SRC_H = 1280, 720  # present.rs JPG output dims

# Target tensor geometry (rectangular 16:9-ish; not letterbox-to-square).
TARGET_W, TARGET_H = 384, 224
# Aspect-preserving inner size used by `pad` / `letterbox384` (1280x720 * 0.3).
ASPECT_W, ASPECT_H = 384, 216


# --------------------------------------------------------------------------- #
# Rows: join labels -> manifest, apply the present.rs black gate.
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    key: str
    label: int          # raw 1/2/3
    jpg: Path
    seed: int           # group key for the split (seed_index)
    draw_index: int
    composition: str
    palette: str
    black_fraction: float


def load_rows(
    labels_path: Path = DEFAULT_LABELS,
    manifest_path: Path = DEFAULT_MANIFEST,
    apply_black_filter: bool = True,
) -> list[Row]:
    labels = json.loads(Path(labels_path).read_text())
    man = json.loads(Path(manifest_path).read_text())
    by_key: dict[str, dict] = {}
    for c in man["crops"]:
        by_key[f"{c['draw_index']}|{c['composition']}|{c['palette']}"] = c
    rows: list[Row] = []
    for key, lab in labels.items():
        c = by_key.get(key)
        if c is None:
            raise ValueError(f"label key not in manifest: {key}")
        jpg = ROOT / c["output"]
        if not jpg.exists():
            raise FileNotFoundError(f"manifest JPG missing: {jpg}")
        bf = float(c["black_fraction"])
        if apply_black_filter and not (bf < BLACK_THRESH):  # mirror present.rs
            continue
        rows.append(Row(
            key=key, label=int(lab), jpg=jpg, seed=int(c["seed_index"]),
            draw_index=int(c["draw_index"]), composition=c["composition"],
            palette=c["palette"], black_fraction=bf,
        ))
    return rows


# --------------------------------------------------------------------------- #
# Geometry — the deterministic resize core (identical in train & deploy).
# --------------------------------------------------------------------------- #
def _pil_resample(interp: str):
    return {
        "nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
    }.get(interp, Image.BICUBIC)


def resize_core(img: Image.Image, geometry: str, interp: str) -> Image.Image:
    """1280x720 PIL -> target geometry. Pure, deterministic, no randomness.

    stretch      : direct resize to 384x224 (uniform ~4% vertical stretch).
    pad          : resize to 384x216 (aspect-exact) then pad 4px top/bottom -> 384x224.
    letterbox384 : resize to 384x216 then pad to 384x384 (documented safe fallback).
    """
    rs = _pil_resample(interp)
    if geometry == "stretch":
        return img.resize((TARGET_W, TARGET_H), rs)
    if geometry == "pad":
        inner = img.resize((ASPECT_W, ASPECT_H), rs)
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        canvas.paste(inner, (0, (TARGET_H - ASPECT_H) // 2))  # 4px top/bottom
        return canvas
    if geometry == "letterbox384":
        inner = img.resize((ASPECT_W, ASPECT_H), rs)
        canvas = Image.new("RGB", (384, 384), (0, 0, 0))
        canvas.paste(inner, (0, (384 - ASPECT_H) // 2))
        return canvas
    raise ValueError(f"unknown geometry: {geometry}")


def target_size(geometry: str) -> tuple[int, int]:
    return (384, 384) if geometry == "letterbox384" else (TARGET_W, TARGET_H)


# --------------------------------------------------------------------------- #
# Transform: PIL(1280x720) -> normalized CHW tensor.
# --------------------------------------------------------------------------- #
class Transform:
    """Callable PIL->tensor. `train=True` adds the lean geometric + palette-
    preserving color augmentations; `train=False` is the bare deterministic
    deploy mirror (resize_core + normalize, center, no jitter/flips)."""

    def __init__(self, geometry: str, interp: str,
                 mean: tuple[float, ...], std: tuple[float, ...],
                 train: bool,
                 border_crop: float = 0.05,
                 jpeg_q: tuple[int, int] = (85, 95),
                 brightness: float = 0.03, contrast: float = 0.03,
                 hflip: float = 0.5, vflip: float = 0.5):
        self.geometry = geometry
        self.interp = interp
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.train = train
        self.border_crop = border_crop
        self.jpeg_q = jpeg_q
        self.brightness = brightness
        self.contrast = contrast
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self, img: Image.Image, rng: random.Random | None = None) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.train:
            r = rng or random
            # --- geometric: border crop 0..border_crop off each edge, then resize back ---
            if self.border_crop > 0:
                w, h = img.size
                l = int(round(r.uniform(0, self.border_crop) * w))
                t = int(round(r.uniform(0, self.border_crop) * h))
                rr = int(round(r.uniform(0, self.border_crop) * w))
                b = int(round(r.uniform(0, self.border_crop) * h))
                if l + rr < w - 8 and t + b < h - 8:
                    img = img.crop((l, t, w - rr, h - b))
            img = resize_core(img, self.geometry, self.interp)
            # --- flips (conjugate-symmetric set: h, v both valid) ---
            if r.random() < self.hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if r.random() < self.vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # --- palette-preserving color: JPEG q jitter (mirrors q90 deploy) ---
            if self.jpeg_q is not None:
                q = r.randint(self.jpeg_q[0], self.jpeg_q[1])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            t = _to_tensor(img)
            # --- brightness/contrast +-3% (NO hue/sat — palette is the label) ---
            if self.brightness > 0:
                t = t * (1.0 + r.uniform(-self.brightness, self.brightness))
            if self.contrast > 0:
                mean_g = t.mean()
                t = (t - mean_g) * (1.0 + r.uniform(-self.contrast, self.contrast)) + mean_g
            t = t.clamp(0, 1)
        else:
            img = resize_core(img, self.geometry, self.interp)
            t = _to_tensor(img)
        return (t - self.mean) / self.std


def _to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.array(img, dtype=np.uint8)  # np.array copies -> writable tensor
    return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CropDataset(Dataset):
    """Yields (tensor, raw_label_1_2_3, row_index). Decode is per-item so the
    border-crop / JPEG augs see the full 1280x720 source."""

    def __init__(self, rows: list[Row], transform: Transform, seed: int = 0,
                 cache: bool = True):
        self.rows = rows
        self.transform = transform
        self.base_seed = seed
        self.cache = cache
        self._cache: dict[int, np.ndarray] = {}  # per-worker; persists w/ persistent_workers

    def __len__(self):
        return len(self.rows)

    def _decode(self, i) -> np.ndarray:
        arr = self._cache.get(i)
        if arr is None:
            with Image.open(self.rows[i].jpg) as im:
                im.load()
                arr = np.array(im.convert("RGB"), dtype=np.uint8)
            if self.cache:
                self._cache[i] = arr  # decode the 1280x720 JPG once, reuse every epoch
        return arr

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.fromarray(self._decode(i))
        # deterministic-per-item rng so workers diverge but stay reproducible.
        rng = random.Random((self.base_seed * 1_000_003 + i) & 0xFFFFFFFF) if self.transform.train else None
        t = self.transform(img, rng)
        return t, row.label, i


# --------------------------------------------------------------------------- #
# Sampler — sqrt-inverse-frequency class weights (softened).
# --------------------------------------------------------------------------- #
def make_weighted_sampler(rows: list[Row], target: str = "ordinal") -> WeightedRandomSampler:
    """Per-sample weight = 1/sqrt(freq[class]). For binary the classes are the
    two BCE groups {1} vs {2,3}; for ordinal we balance the raw 3 classes."""
    if target == "binary":
        cls = [0 if r.label == 1 else 1 for r in rows]
    else:
        cls = [r.label for r in rows]
    counts: dict[int, int] = {}
    for c in cls:
        counts[c] = counts.get(c, 0) + 1
    w_per_class = {c: 1.0 / np.sqrt(n) for c, n in counts.items()}
    weights = torch.tensor([w_per_class[c] for c in cls], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


def class_histogram(rows: list[Row]) -> dict[int, int]:
    h: dict[int, int] = {}
    for r in rows:
        h[r.label] = h.get(r.label, 0) + 1
    return dict(sorted(h.items()))
