"""The pinned render-mode (mining) quality gate — ``mining_v1``.

This is the SINGLE SOURCE OF TRUTH for the strange-mode quality gate, the mining
analogue of ``scoring.active_ckpt.ACTIVE_CKPT`` (the location-quality pin). It does
one job: **gate** — raster in, ``p_ge3`` + pass/fail out. The candidate-generation
loop (per-location mode x param rendering + quota policy) is the separate deploy
tail and does NOT live here; it *calls* this.

Pin (§1)
    ``ACTIVE_MINING_CKPT`` -> the staged seed-0 checkpoint of the render-mode head
    v1 (``data/render_mode_head/v1/model_best.pt`` = best per-seed eval not-bad AP,
    seed 0). Version tag ``mining_v1``. First version -> no rollback target yet
    (``MINING_V1_ROLLBACK`` is reserved, currently None). Flip ``ACTIVE_MINING_CKPT``
    and the whole strange-mode gate moves; nothing else hardcodes a mining version.

Deploy config MATCHES TRAINING EXACTLY (``classifier.train_mining_head`` /
``classifier.data.Transform(train=False)``): 384x224 bicubic stretch, the
checkpoint's OWN mean/std, and the marginal ``p_ge = cumprod(sigma(logits))`` gate
probability -- NEVER the CORN conditional ``sigma(logit_1)`` (that is
``P(>=3 | >=2)``, not ``P(>=3)``; see [[corn-conditional-vs-marginal]]). This is
the load-bearing difference from ``tools/mining/score_lib.Scorer``, which returns
the conditional ``p_good`` -- do not swap one for the other here.

Threshold (§2)
    ``MINING_GATE_THRESHOLD = 0.50`` on marginal ``p_ge3``. The canonical CORN
    marginal boundary (P(label>=3) > 1/2), chosen conservative / high-precision:
    the strange-mode quota is a ceiling, so under-emitting is the safe direction.
    Seed-0 (deployed) operating point on the held-out eval set: precision 0.548,
    recall 0.195, pass-rate 0.050 (~3.9x the 0.139 good base rate). Full PR curve,
    the operating point, and the 5-seed stability cross-check are frozen in
    ``mining_gate_lock.json`` next to the checkpoint (written by
    ``lock_mining_gate.py``).

Scoring entry point (§3)
    ``MiningScorer`` is provenance-blind (raster in, score out) and head-agnostic
    (backbone / num_classes / mean / std / geometry all read from the checkpoint's
    own config), so pointing ``ACTIVE_MINING_CKPT`` at a future ``mining_v2`` with a
    different backbone needs no code change here.

Provenance (§5)
    ``gate_stamp(p_ge3)`` returns the ``{gate, threshold, p_ge3, passed}`` block to
    embed in a gate-passed strange-mode emission, so every emission carries which
    gate passed it and at what operating point.

    uv run python tools/mining/mining_gate.py <img.jpg> [<img.jpg> ...]   # score+gate
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timm  # noqa: E402

from classifier.data import Transform  # noqa: E402

# --------------------------------------------------------------------------- #
# The pin (§1) + threshold (§2). Flip ACTIVE_MINING_CKPT to move the gate.
# --------------------------------------------------------------------------- #
MINING_GATE_VERSION = "mining_v1"
ACTIVE_MINING_CKPT = "data/render_mode_head/v1/model_best.pt"   # staged seed-0 (LIVE)
MINING_V1_ROLLBACK = None       # first version -> no prior gate to fall back to
MINING_GATE_THRESHOLD = 0.50    # marginal p_ge3 boundary; conservative / high-precision
LOCK_PATH = "data/render_mode_head/v1/mining_gate_lock.json"   # frozen curve + parity


@dataclass(frozen=True)
class MiningScore:
    """One raster's gate read. ``p_ge`` are MARGINAL cumulative-rank probs."""
    p_ge2: float        # marginal P(label >= 2) = sigma(l0)
    p_ge3: float        # marginal P(label >= 3) = sigma(l0) * sigma(l1)   <-- the gate signal
    score: float        # E[ord] = sigma(l0) + sigma(l1) in [0, 2] (monotone rank score)
    passed: bool        # p_ge3 >= MINING_GATE_THRESHOLD


class MiningScorer:
    """Pinned mining-head gate scorer. Provenance-blind, head-agnostic.

    Mirrors the training preprocessing EXACTLY (deploy ``Transform(train=False)``)
    and returns the marginal gate probability. ``model_path`` defaults to the pin;
    pass one explicitly only to score a non-canonical checkpoint (e.g. a per-seed
    model during the lock's stability cross-check)."""

    def __init__(self, model_path: str = ACTIVE_MINING_CKPT,
                 threshold: float = MINING_GATE_THRESHOLD, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        path = model_path if os.path.isabs(model_path) else str(ROOT / model_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if cfg.get("target") != "ordinal":
            raise SystemExit(f"mining gate expects an ordinal head, got target={cfg.get('target')!r}")
        self.cfg = cfg
        self.k = int(cfg["num_classes"])                 # tiers (3); logits = k-1
        # Head-agnostic build: backbone + head width come from the checkpoint's config.
        model = timm.create_model(cfg["backbone"], pretrained=False, num_classes=self.k - 1,
                                  drop_rate=cfg.get("drop_rate", 0.2),
                                  drop_path_rate=cfg.get("drop_path_rate", 0.1))
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.eval().to(self.device)
        # Deploy transform = the train harness's eval transform, byte-for-byte.
        self.transform = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                                   mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)

    @torch.no_grad()
    def _cond_marg(self, imgs: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Conditional and marginal cumulative-rank probs, each (N, k-1).

        cond[:,r] = sigma(logit_r)              (conditional: P(>=r+2 | >=r+1))
        marg      = cumprod(cond, axis=1)       (marginal: [:,0]=P(>=2), [:,1]=P(>=3))
        E[ord] = cond.sum(axis=1) = score_from_logits (the monotone rank score).

        Deterministic fp32 (no autocast) so deploy scores match the train-harness
        eval scores bit-for-bit -- the parity lock (§4) depends on this."""
        x = torch.stack([self.transform(im) for im in imgs]).to(self.device)
        logits = self.model(x).float().cpu()
        cond = torch.sigmoid(logits).numpy().astype(np.float64)
        return cond, np.cumprod(cond, axis=1)

    def score_pils(self, imgs: list[Image.Image]) -> list[MiningScore]:
        cond, marg = self._cond_marg(imgs)
        out = []
        for c, m in zip(cond, marg):
            p2, p3 = float(m[0]), float(m[1])
            out.append(MiningScore(p_ge2=p2, p_ge3=p3, score=float(c.sum()),
                                   passed=(p3 >= self.threshold)))
        return out

    def score_paths(self, paths, batch_size: int = 64) -> list[MiningScore]:
        out: list[MiningScore] = []
        buf: list[Image.Image] = []

        def flush():
            if buf:
                out.extend(self.score_pils(buf))
                buf.clear()

        for p in paths:
            with Image.open(p) as im:
                im.load()
                buf.append(im.convert("RGB"))
            if len(buf) >= batch_size:
                flush()
        flush()
        return out

    def gate(self, p_ge3: float) -> bool:
        return float(p_ge3) >= self.threshold

    def stamp(self, p_ge3: float) -> dict:
        return gate_stamp(p_ge3, self.threshold)


# --------------------------------------------------------------------------- #
# Provenance (§5). Head-side stamp for a gate-passed strange-mode emission.
# --------------------------------------------------------------------------- #
def gate_stamp(p_ge3: float, threshold: float = MINING_GATE_THRESHOLD) -> dict:
    """The provenance block to embed in a gate-passed strange-mode raster."""
    return {
        "gate": MINING_GATE_VERSION,
        "checkpoint": ACTIVE_MINING_CKPT,
        "threshold": float(threshold),
        "p_ge3": float(p_ge3),
        "passed": float(p_ge3) >= float(threshold),
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(f"usage: python {Path(__file__).name} <img.jpg> [...]  "
                         f"(gate={MINING_GATE_VERSION} thr={MINING_GATE_THRESHOLD})")
    sc = MiningScorer()
    print(f"gate={MINING_GATE_VERSION}  ckpt={ACTIVE_MINING_CKPT}  thr={sc.threshold}")
    for p, r in zip(paths, sc.score_paths(paths)):
        print(f"  {'PASS' if r.passed else 'fail'}  p_ge3={r.p_ge3:.4f}  "
              f"p_ge2={r.p_ge2:.4f}  E[ord]={r.score:.4f}  {p}")


if __name__ == "__main__":
    main()
