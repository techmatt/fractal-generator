"""Shared v3 scoring helpers for the biased mining harness.

Two scoring paths, both through the EXACT v3 deploy transform
(`classifier.data.Transform(train=False)` = 1280x720 -> 384x224 bicubic-stretch +
normalize), so scores stay on the trained distribution:

1. `Scorer.score_pils` / `score_paths` — score in-memory PIL images or JPGs on
   disk. Returns the CORN triple per frame: ordinal score in [0,2]
   (= sigma(l0)+sigma(l1) = `score_from_logits`), P(not-bad) = sigma(l0)
   (= P(label>=2)), P(good) = sigma(l1) (= P(label>=3)).

2. `run_enrich_score` — the in-memory Rust->Python bridge. Launches the frozen
   `enrich --mode score` subcommand (iterate-once at label geometry, present
   gates, recolor under the roster, stream raw RGB), scores every streamed frame,
   and returns {idx: {ki: (score, p_notbad, p_good)}} joined with the per-location
   gate verdict + palette list from the Rust --meta-out sidecar. No crops to disk.

The mining harness drives the same `enrich` machinery with *aggressive params*
(custom rosters, geometries) without touching any production default.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classifier.data import Transform  # noqa: E402
from classifier.model import build_model  # noqa: E402

HDR = struct.Struct("<IIII")  # idx, ki, w, h  (little-endian u32 x4)
# The model the biased mining harness was calibrated against. NOT a default:
# `Scorer(model_path=...)` is required so no path can *silently* score with v3.
# The v5-intended callers (reframe/atlas/step0) always pass v5 explicitly via
# `make_scorer`; the two mining tools (harvest.py, calibrate_t2.py) pass this.
DEFAULT_V3 = "data/classifier/v3/model_best.pt"
BIN = "target/release/fractal-generator.exe"


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def corn_decode(p_notbad: float, p_good: float, t_good: float = 0.5) -> int:
    """Canonical v5 CORN hard-class decode -> {1, 2, 3} (bad / okay / good).

    The two ordinal sigmoids are the cumulative rank probabilities
    ``p_notbad = sigma(l0) = P(class >= 2)`` and ``p_good = sigma(l1) = P(class >= 3)``.
    Rank-consistent hard class = ``1 + #{cumulative probs >= threshold}``. This is NOT
    recoverable from the summed ``E[ord] = p_notbad + p_good`` scalar (two frames with
    equal E[ord] can decode to different classes), so callers pass the two
    probabilities and MUST NOT threshold the score. Single source of truth for the
    decode; reuse it, don't reimplement the >= threshold counting inline.

    ``t_good`` is the q3 (rank-3) operating point on ``p_good``. It defaults to 0.5,
    which is BYTE-IDENTICAL to the historical decode — every existing caller stays put.
    Discovery sites opt in to a lower per-degree threshold (the v6 sweep knee) by
    passing ``t_good`` explicitly. The rank-2 gate on ``p_notbad`` stays fixed at 0.5:
    a class-3 outcome must still be not-bad, so lowering ``t_good`` below 0.5 can only
    turn a would-be class-2 into class-3, never resurrect a class-1 (the AND rule holds
    because ``p_notbad >= p_good`` is not guaranteed — see the monotonicity check in
    tools/v6/threshold_sweep.py — but a class-1 has ``p_notbad < 0.5`` and is capped at
    ``1 + 0 + 1 = 2`` regardless, i.e. it can reach class-2 but not class-3)."""
    return 1 + int(p_notbad >= 0.5) + int(p_good >= t_good)


class Scorer:
    """v3 model + deploy transform, exposing the full CORN triple per frame."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = pick_device(device)
        path = model_path if os.path.isabs(model_path) else str(ROOT / model_path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        if cfg["target"] != "ordinal":
            raise SystemExit(f"expected ordinal head, got target={cfg['target']!r}")
        model = build_model(
            target=cfg["target"], drop_rate=cfg.get("drop_rate", 0.2),
            drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.eval().to(self.device)
        self.transform = Transform(
            geometry=cfg["geometry"], interp=cfg["interpolation"],
            mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False,
        )
        self.cfg = cfg

    @torch.no_grad()
    def score_pils(self, imgs: list[Image.Image]):
        """Returns (score[0,2], p_notbad, p_good) numpy arrays, one row per image."""
        x = torch.stack([self.transform(im) for im in imgs]).to(self.device)
        if self.device != "cpu":
            with torch.autocast(device_type=self.device.split(":")[0]):
                logits = self.model(x)
        else:
            logits = self.model(x)
        logits = logits.float().cpu()
        p_notbad = torch.sigmoid(logits[:, 0]).numpy()  # P(rank>=1) = P(label>=2)
        p_good = torch.sigmoid(logits[:, 1]).numpy()     # P(rank>=2) = P(label>=3)
        score = p_notbad + p_good                        # score_from_logits, [0,2]
        return score, p_notbad, p_good

    def score_paths(self, paths, batch_size: int = 64):
        """Score JPGs on disk. Returns list of (score, p_notbad, p_good)."""
        out = []
        buf: list[Image.Image] = []

        def flush():
            if not buf:
                return
            s, nb, g = self.score_pils(buf)
            out.extend(zip(s.tolist(), nb.tolist(), g.tolist()))
            buf.clear()

        for p in paths:
            with Image.open(p) as im:
                im.load()
                buf.append(im.convert("RGB"))
            if len(buf) >= batch_size:
                flush()
        flush()
        return out


def read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise EOFError(f"stream ended mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


def run_enrich_score(
    scorer: Scorer,
    pool_path: str,
    colormaps: str,
    *,
    k: int,
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
    score_ss: int = 1,
    maxiter: int = 8000,
    black_cap: float = 0.30,
    occ_floor: float = 0.321,
    meta_out: str,
    batch_size: int = 96,
    bin_path: str = BIN,
    progress_every: int = 4096,
    frame_cb=None,
    log=print,
):
    """Stream every (location x roster-palette) frame through v3.

    Returns (scores, locs):
      scores: {idx: {ki: (score, p_notbad, p_good)}}
      locs:   list of per-location meta dicts (idx, cx, cy, fw, gated, gate_reason,
              black_fraction, occupancy, palettes[]) read from the Rust sidecar.
    """
    meta_abs = meta_out if os.path.isabs(meta_out) else str(ROOT / meta_out)
    os.makedirs(os.path.dirname(meta_abs), exist_ok=True)
    cmd = [
        str(ROOT / bin_path) if not os.path.isabs(bin_path) else bin_path,
        "enrich", "--mode", "score",
        "--pool", pool_path, "--colormaps", colormaps,
        "--k", str(k), "--seed", str(seed),
        "--width", str(width), "--height", str(height), "--score-ss", str(score_ss),
        "--maxiter", str(maxiter), "--black-cap", str(black_cap), "--occ-floor", str(occ_floor),
        "--meta-out", meta_out,
    ]
    log("launching: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=None, bufsize=0)

    scores: dict[int, dict[int, tuple]] = {}
    pend_imgs: list[Image.Image] = []
    pend_meta: list[tuple[int, int]] = []
    n_frames = 0

    def flush():
        nonlocal n_frames
        if not pend_imgs:
            return
        s, nb, g = scorer.score_pils(pend_imgs)
        for (idx, ki), a, b, c in zip(pend_meta, s.tolist(), nb.tolist(), g.tolist()):
            scores.setdefault(idx, {})[ki] = (a, b, c)
        n_frames += len(pend_imgs)
        pend_imgs.clear()
        pend_meta.clear()

    stream = proc.stdout
    while True:
        hdr = read_exact(stream, HDR.size)
        if hdr is None:
            break
        idx, ki, w, h = HDR.unpack(hdr)
        payload = read_exact(stream, w * h * 3)
        if payload is None:
            raise EOFError("EOF before image payload")
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3)
        pil = Image.fromarray(arr, "RGB")
        if frame_cb is not None:
            frame_cb(idx, ki, pil)
        pend_imgs.append(pil)
        pend_meta.append((idx, ki))
        if len(pend_imgs) >= batch_size:
            flush()
            if progress_every and n_frames % progress_every == 0:
                log(f"  scored {n_frames} frames ({len(scores)} locations)...")
    flush()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"enrich --mode score exited {rc}")

    locs: list[dict] = []
    with open(meta_abs, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or json.loads(line).get("kind") == "header":
                continue
            locs.append(json.loads(line))
    return scores, locs
