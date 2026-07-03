"""Dependency-free perceptual hashing for crop-level near-dup detection.

Near-duplicate locations collapsing onto the same hot spot are the expected
failure of aggressive location mining. We hash each crop's GRAYSCALE structure
(64-bit DCT pHash), so the hash is palette-robust: the same fractal structure
under a different palette still collides. We dedup mined crops against each other
AND against the already-labeled corpus, so Matt never re-judges a known location.

MANUAL-ONLY: imported by harvest.py (phash, DedupIndex); no standalone entry point.
Kept as harvest.py's pHash dependency now that the mining orchestrator is gone.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def phash(img: Image.Image, hash_size: int = 8, highfreq: int = 4) -> int:
    """64-bit DCT perceptual hash on grayscale (palette-robust structure hash)."""
    n = hash_size * highfreq
    im = img.convert("L").resize((n, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64)
    # 2D DCT-II via matrix; keep the low-freq hash_size x hash_size block (sans DC)
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    dct = basis @ a @ basis.T
    block = dct[:hash_size, :hash_size]
    med = np.median(block[1:].flatten() if hash_size > 1 else block.flatten())
    bits = (block > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class DedupIndex:
    """Greedy near-dup index. add() returns True if the hash is novel (kept),
    False if it collides with something already in the index (within `thresh`
    Hamming bits)."""

    def __init__(self, thresh: int = 6):
        self.thresh = thresh
        self.hashes: list[int] = []

    def seed(self, hashes):
        """Pre-load reference hashes (e.g. the labeled corpus) that should be
        treated as already-seen but are not themselves 'kept' results."""
        self.hashes.extend(hashes)

    def is_dup(self, h: int) -> bool:
        return any(hamming(h, x) <= self.thresh for x in self.hashes)

    def add(self, h: int) -> bool:
        if self.is_dup(h):
            return False
        self.hashes.append(h)
        return True
