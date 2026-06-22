"""Distribution sampler over the deduped library.

The deliverable is a *distribution* of palettes on output, not a per-candidate
optimizer. Default is uniform-over-library; the distribution is the exposed knob
(`weights`) — reweight by source, by hand-set scores, or any future signal to
shift what shows up, without touching the render path.
"""

from __future__ import annotations

import numpy as np


class Sampler:
    def __init__(self, library, weights=None):
        self.library = list(library)
        n = len(self.library)
        if weights is None:
            weights = np.ones(n)
        weights = np.asarray(weights, dtype=np.float64)
        assert weights.shape == (n,), "weights must match library length"
        assert (weights >= 0).all() and weights.sum() > 0
        self.p = weights / weights.sum()

    @classmethod
    def by_source(cls, library, source_weights):
        """Weight palettes by a {source: weight} dict (the distribution knob)."""
        w = np.array([source_weights.get(p["source"], 1.0) for p in library], dtype=np.float64)
        return cls(library, w)

    def draw(self, n, seed=0, replace=False):
        """Draw n palettes. Sampling is the only place randomness enters; seed
        it explicitly (the engine forbids Math.random-style nondeterminism)."""
        rng = np.random.default_rng(seed)
        n = min(n, len(self.library)) if not replace else n
        idx = rng.choice(len(self.library), size=n, replace=replace, p=self.p)
        return [self.library[i] for i in idx]
