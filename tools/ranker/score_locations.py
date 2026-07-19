#!/usr/bin/env python
"""score_locations.py — score ledger-row LOCATIONS with the pref_loc_v0 ranker.

The shared *consumer-side* seam for the ranker's three legitimate consumers (keeper
ranking, emission feed ordering, dive-result sorting — see the HARD SCOPE box in
`scorer.py`). Given a list of admitted ledger rows it returns `id -> rank score` by:

  1. reusing the deployed frozen feature blocks (the model's `sets`, currently
     v7 + colored) from any provided cache — the ranker's own
     `data/ranker/pref_loc_v0/features.npz` covers every steered_run2 + dive
     admission, so scoring those is a pure cache hit with no render; ELSE
  2. rendering the twilight_shifted canonical tile (`spm.render_colored` — the exact
     recipe `features.npz` was built on) and computing v7 (prescreen penultimate) +
     colored (CLIP), caching the result so a resume never recomputes.

RANKS AN ALREADY-PRODUCED SET. This must never be called on the generation side
(frontier priority / dive-start / production seeding); doing so is the failure mode
`scorer.py` forbids. Both consumers here order candidates that were already admitted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "atlas", ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.ranker.scorer import RankerScorer  # noqa: E402

DEFAULT_FEATURES = ROOT / "data" / "ranker" / "pref_loc_v0" / "features.npz"
# Blocks this helper knows how to recompute on a cache miss. If a future deploy adds
# 'morph' to `sets`, teach `_compute` the morph_gray recipe before wiring — asserted below.
COMPUTABLE = ("v7", "colored")


def rank_percentiles(scores: dict) -> dict:
    """id -> fraction of the set whose score is <= this one (a singleton -> 1.0).
    Matches `selection.niche_percentiles`: ties share the higher rank."""
    items = list(scores.items())
    n = len(items)
    if n == 0:
        return {}
    vals = [s for _, s in items]
    return {i: sum(1 for v in vals if v <= s) / n for i, s in items}


class LocationRanker:
    """Loads the deployed pref_loc_v0 head + any frozen feature caches; scores rows.

    The heavy feature stack (the v7 Scorer + CLIP) is imported lazily and only if a
    cache miss forces a render, so a fully-cached consumer (keeper report over
    steered_run2, emission intake over run2+dive) never pays the torch import."""

    def __init__(self, model_path=None, feature_caches=(DEFAULT_FEATURES,)):
        self.scorer = (RankerScorer.load() if model_path is None
                       else RankerScorer.load(model_path))
        self.sets = list(self.scorer.sets)
        unknown = [b for b in self.sets if b not in COMPUTABLE]
        if unknown:
            raise NotImplementedError(
                f"LocationRanker can recompute only {COMPUTABLE}; deployed sets={self.sets} "
                f"needs {unknown}. Extend `_compute` with that block's recipe before wiring.")
        self._cache: dict = {}          # id -> {block_name: 1-D float64 vec}
        for c in feature_caches:
            self._load_cache(c)
        self._stack = None              # lazy (v7 Scorer, clip_model, clip_tf)

    # ---- caches ---------------------------------------------------------- #
    def _load_cache(self, path):
        path = Path(path)
        if not path.exists():
            return
        z = np.load(path, allow_pickle=True)
        if "ids" not in getattr(z, "files", []):
            return
        ids = [str(i) for i in z["ids"]]
        have = [b for b in self.sets if b in z.files]
        for k, rid in enumerate(ids):
            slot = self._cache.setdefault(rid, {})
            for b in have:
                slot.setdefault(b, z[b][k].astype(np.float64))

    def _has_all(self, rid) -> bool:
        feat = self._cache.get(rid)
        return feat is not None and all(b in feat for b in self.sets)

    # ---- lazy feature stack --------------------------------------------- #
    def _ensure_stack(self):
        if self._stack is not None:
            return self._stack
        import production_seeder as ps                       # noqa: E402
        from score_lib import Scorer                          # noqa: E402
        from tools.curation.colored_clip import load_clip     # noqa: E402
        scorer = Scorer(str(ps.SCORER_PATH))
        clip_model, clip_tf = load_clip()
        self._stack = (scorer, clip_model, clip_tf)
        return self._stack

    def _compute(self, row, tile_dir) -> dict:
        """Cache miss: render the canonical tile once, compute the deployed blocks."""
        import tools.studies.steered_pilot_morph as spm       # noqa: E402
        import prescreen                                       # noqa: E402
        from tools.curation.colored_clip import embed_clip     # noqa: E402
        from PIL import Image
        v7_scorer, clip_model, clip_tf = self._ensure_stack()
        tile = Path(tile_dir) / f"{row['id']}.jpg"
        if not tile.exists():
            tile.parent.mkdir(parents=True, exist_ok=True)
            spm.render_colored(spm.loc_of_row(row), tile)
        feat = {}
        if "v7" in self.sets:
            feat["v7"] = prescreen.embed_paths(v7_scorer, [tile])[0].astype(np.float64)
        if "colored" in self.sets:
            feat["colored"] = embed_clip(clip_model, clip_tf,
                                         [Image.open(tile)])[0].astype(np.float64)
        return feat

    # ---- scoring --------------------------------------------------------- #
    def score_rows(self, rows, tile_dir, persist_npz=None) -> dict:
        """rows: admitted ledger rows (each needs an 'id'). Returns id -> float score
        (higher == more human-good). Cache misses render + embed; newly-computed rows
        are folded into `persist_npz` (resume-safe) when given. Duplicate ids collapse
        to one score."""
        seen, order = set(), []
        for row in rows:
            rid = row["id"]
            if rid in seen:
                continue
            seen.add(rid)
            order.append(rid)
            if not self._has_all(rid):
                self._cache.setdefault(rid, {}).update(self._compute(row, tile_dir))
        if persist_npz is not None:
            self._persist(persist_npz)
        mats = {b: np.stack([self._cache[i][b] for i in order]) for b in self.sets}
        sc = self.scorer.score_matrix(mats)
        return {i: float(s) for i, s in zip(order, sc)}

    def _persist(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = [i for i in self._cache if self._has_all(i)]
        save = {"ids": np.array(ids, dtype=object)}
        for b in self.sets:
            save[b] = np.stack([self._cache[i][b] for i in ids]).astype(np.float32)
        tmp = path.parent / (path.stem + "_tmp.npz")
        np.savez_compressed(tmp, **save)
        os.replace(tmp, path)
