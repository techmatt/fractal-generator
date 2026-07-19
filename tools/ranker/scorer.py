#!/usr/bin/env python
"""Location preference ranker v0 — scorer entry point.

Loads the deployed linear head (`data/ranker/pref_loc_v0/model.npz`) and maps a joined frozen
feature record -> a scalar rank score (higher == more likely human-good). The head is a plain
affine map on standardized features, so scoring is dependency-light (no torch): standardize with
the stored (mean, scale), dot with W, add b.

    >>> from tools.ranker.scorer import RankerScorer
    >>> s = RankerScorer.load()
    >>> s.score_matrix({"morph": M, "v7": V, "colored": C})   # each (N, dim)

============================ HARD SCOPE — READ BEFORE WIRING ============================
This head ranks the NOT-BAD; it must NEVER steer discovery. Do NOT wire RankerScorer into
frontier priority (steered_frontier.py), dive-start selection (--dive selection), production
seeding, or any generation-side decision. A model that both selects and ranks degrades on its own
selections — that is exactly how canonical p_good became a badness filter rather than a goodness
ranker (docs/findings/steered_run2_keeper_calibration.md). Legitimate consumers ONLY: keeper
ranking, emission feed ordering, and dive-result sorting — all of which rank an already-produced
set without feeding back into what gets produced.
========================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "ranker" / "pref_loc_v0" / "model.npz"


class RankerScorer:
    def __init__(self, mean, scale, W, b, sets, head, reg, use_prior):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.W = np.asarray(W, np.float64)
        self.b = float(b)
        self.sets = [str(s) for s in sets]      # feature blocks, in concat order
        self.head = str(head)
        self.reg = float(reg)
        self.use_prior = bool(use_prior)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL):
        z = np.load(path, allow_pickle=True)
        return cls(z["mean"], z["scale"], z["W"], z["b"], z["sets"], z["head"],
                   z["reg"], bool(z["use_prior"]))

    def _stack(self, blocks: dict) -> np.ndarray:
        return np.concatenate([np.atleast_2d(blocks[b]).astype(np.float64) for b in self.sets],
                              axis=1)

    def score_matrix(self, blocks: dict) -> np.ndarray:
        """blocks: {'morph': (N,768), 'v7': (N,1280), 'colored': (N,768)} (only the deployed
        blocks are required). Returns (N,) rank scores."""
        X = self._stack(blocks)
        Xs = (X - self.mean) / self.scale
        return Xs @ self.W + self.b


def _cli():
    """Score data/ranker/pref_loc_v0/features.npz and print id, score, human (if labeled)."""
    feat = ROOT / "data/ranker/pref_loc_v0/features.npz"
    z = np.load(feat, allow_pickle=True)
    s = RankerScorer.load()
    blocks = {b: z[b] for b in s.sets}
    sc = s.score_matrix(blocks)
    order = np.argsort(-sc)
    print(f"# ranker v0  head={s.head} sets={s.sets} prior={s.use_prior}")
    for i in order:
        hs = int(z["score"][i])
        print(f"{sc[i]:+.4f}  {z['ids'][i]:44s} {z['family'][i]:16s} "
              f"human={hs if hs else '-'}")


if __name__ == "__main__":
    _cli()
