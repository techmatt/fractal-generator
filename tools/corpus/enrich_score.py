"""Stage B — score a guided-descend pool with v2, in-memory, no crops to disk.

Launches the Rust `enrich --mode score` subcommand, which iterates each pool
location ONCE at the label geometry, applies the present gates (black<0.30 +
occ>=0.321), recolors each survivor under K seeded score-3 palettes, and streams
the recolored 1280x720 RGB frames to stdout as length-prefixed records. This
script reads that stream and scores every frame with the **v2** classifier
through inference.py's EXACT deterministic transform (the 1280x720 -> 384x224
bicubic-stretch + normalize deploy mirror), so scores stay on the trained
distribution. The frames never touch disk.

For each location: filter_score = MAX P(not-bad) over the K palettes (a location
good under *some* palette isn't dropped for a bad recolor); argmax_palette = the
winner. P(not-bad) for the CORN ordinal head is sigma(logit_0) = P(rank>=1) =
P(label>=2) -- exact. The K (palette, P) pairs and the gate verdict (from the
Rust `--meta-out` sidecar) are recorded per location in scored.jsonl.

Sibling bridge: `tools/mining/score_lib.py:run_enrich_score` drives the same
`enrich --mode score` machinery for the mining harness, but is a *different
contract* (a v3-pinned library returning the CORN triple, not a v2 CLI writing
scored.jsonl) — deliberately not unified. The 16-byte stream header both parse
(`HDR = struct.Struct("<IIII")`) is owned by the Rust side (`src/enrich.rs`), not
by either script.

Run:
  uv run python tools/corpus/enrich_score.py \
      --pool data/guided_descend/run5/pool.jsonl \
      --bin  target/release/fractal-generator.exe \
      --model data/classifier/v2/model_best.pt \
      --out  data/enrich/run5/scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# repo root = three levels up from tools/corpus/enrich_score.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from classifier.data import Transform  # noqa: E402
from classifier.model import build_model  # noqa: E402

HDR = struct.Struct("<IIII")  # idx, ki, w, h  (little-endian u32 x4)
MODEL_ID = "data/classifier/v2/model_best.pt"


def load_v2(model_path: str, device: str):
    """Replicate inference.load_scorer's model+transform build, but expose raw
    logits so we can take sigma(logit_0) = P(not-bad) exactly."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["target"] != "ordinal":
        raise SystemExit(f"expected ordinal head, got target={cfg['target']!r}")
    model = build_model(
        target=cfg["target"], drop_rate=cfg.get("drop_rate", 0.2),
        drop_path_rate=cfg.get("drop_path_rate", 0.1), pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    transform = Transform(
        geometry=cfg["geometry"], interp=cfg["interpolation"],
        mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False,
    )
    return model, transform, cfg


def read_exact(stream, n: int) -> bytes | None:
    """Read exactly n bytes from a binary stream; None on clean EOF at a record
    boundary, raises if EOF mid-record."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise EOFError(f"stream ended mid-record ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


@torch.no_grad()
def score_batch(model, transform, device, imgs: list[Image.Image]):
    x = torch.stack([transform(im) for im in imgs]).to(device)
    if device != "cpu":
        with torch.autocast(device_type=device.split(":")[0]):
            logits = model(x)
    else:
        logits = model(x)
    logits = logits.float().cpu()
    p_notbad = torch.sigmoid(logits[:, 0]).tolist()  # P(rank>=1) = P(not-bad)
    p_good = torch.sigmoid(logits[:, 1]).tolist()     # CORN second threshold
    return p_notbad, p_good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/guided_descend/run5/pool.jsonl")
    ap.add_argument("--bin", default="target/release/fractal-generator.exe")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--colormaps", default="data/palettes/score3_colormaps.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--score-ss", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=8000)
    ap.add_argument("--black-cap", type=float, default=0.30)
    ap.add_argument("--occ-floor", type=float, default=0.321)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--meta-out", default="data/enrich/run5/score_meta.jsonl")
    ap.add_argument("--out", default="data/enrich/run5/scored.jsonl")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, cfg = load_v2(str(ROOT / a.model) if not os.path.isabs(a.model) else a.model, device)
    print(f"v2 loaded ({cfg['backbone']}, {cfg['target']} head) on {device}", flush=True)

    os.makedirs(os.path.dirname(str(ROOT / a.meta_out)), exist_ok=True)
    cmd = [
        str(ROOT / a.bin) if not os.path.isabs(a.bin) else a.bin, "enrich", "--mode", "score",
        "--pool", a.pool, "--colormaps", a.colormaps, "--k", str(a.k), "--seed", str(a.seed),
        "--width", str(a.width), "--height", str(a.height), "--score-ss", str(a.score_ss),
        "--maxiter", str(a.maxiter), "--black-cap", str(a.black_cap), "--occ-floor", str(a.occ_floor),
        "--meta-out", a.meta_out,
    ]
    print("launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=None, bufsize=0)

    # idx -> {ki: (p_notbad, p_good)}
    scores: dict[int, dict[int, tuple[float, float]]] = {}
    pend_meta: list[tuple[int, int]] = []  # (idx, ki) parallel to pend_imgs
    pend_imgs: list[Image.Image] = []
    n_frames = 0

    def flush_batch():
        nonlocal n_frames
        if not pend_imgs:
            return
        pnb, pg = score_batch(model, transform, device, pend_imgs)
        for (idx, ki), a_nb, a_g in zip(pend_meta, pnb, pg):
            scores.setdefault(idx, {})[ki] = (a_nb, a_g)
        n_frames += len(pend_imgs)
        pend_meta.clear()
        pend_imgs.clear()

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
        pend_imgs.append(Image.fromarray(arr, "RGB"))
        pend_meta.append((idx, ki))
        if len(pend_imgs) >= a.batch_size:
            flush_batch()
            if n_frames % 2048 == 0:
                print(f"  scored {n_frames} frames ({len(scores)} locations)…", flush=True)
    flush_batch()

    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"enrich --mode score exited {rc}")
    print(f"scored {n_frames} frames over {len(scores)} kept locations", flush=True)

    # --- join the gate sidecar -> scored.jsonl --------------------------------
    # Each kept meta line carries its OWN per-location palette list (the K draw is
    # per-location, seeded by idx), so (idx, ki) -> palette is read per location.
    meta_path = str(ROOT / a.meta_out) if not os.path.isabs(a.meta_out) else a.meta_out
    locs: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("kind") == "header":
                continue
            locs.append(o)

    out_path = str(ROOT / a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_gated = n_kept = n_scored = 0
    fs_dist: list[float] = []
    with open(out_path, "w", encoding="utf-8") as fout:
        for o in locs:
            idx = o["idx"]
            row = {
                "idx": idx, "cx": o["cx"], "cy": o["cy"], "fw": o["fw"],
                "gated": bool(o["gated"]), "gate_reason": o["gate_reason"],
                "black_fraction": o.get("black_fraction"), "occupancy": o.get("occupancy"),
            }
            if o["gated"]:
                n_gated += 1
                row.update(filter_score=None, est_class=None, argmax_palette=None, k_scores=None)
            else:
                n_kept += 1
                sc = scores.get(idx, {})
                palettes = o.get("palettes", [])
                k_scores = []
                for ki, pname in enumerate(palettes):
                    if ki in sc:
                        nb, g = sc[ki]
                        k_scores.append({"palette": pname, "p_notbad": nb, "p_good": g})
                if not k_scores:
                    # kept by gate but no frames scored (shouldn't happen) -> treat as 0
                    row.update(filter_score=0.0, est_class=1, argmax_palette=None, k_scores=[])
                else:
                    best = max(k_scores, key=lambda d: d["p_notbad"])
                    fs = best["p_notbad"]
                    est = 1 + int(fs >= 0.5) + int(best["p_good"] >= 0.5)
                    row.update(filter_score=fs, est_class=est,
                               argmax_palette=best["palette"], k_scores=k_scores)
                    fs_dist.append(fs)
                    n_scored += 1
            fout.write(json.dumps(row) + "\n")

    fs_dist.sort()
    def pct(p):
        if not fs_dist:
            return float("nan")
        return fs_dist[min(len(fs_dist) - 1, int(p * len(fs_dist)))]
    print(f"\n=== Stage B summary ===")
    print(f"locations:        {len(locs)}")
    print(f"  gated:          {n_gated}  ({n_gated/max(1,len(locs))*100:.1f}%)")
    print(f"  kept (scored):  {n_kept}")
    if fs_dist:
        print(f"filter_score (P not-bad, best-over-K) distribution over kept:")
        print(f"  min {fs_dist[0]:.3f}  p10 {pct(.10):.3f}  p25 {pct(.25):.3f}  "
              f"p50 {pct(.50):.3f}  p75 {pct(.75):.3f}  p90 {pct(.90):.3f}  max {fs_dist[-1]:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
