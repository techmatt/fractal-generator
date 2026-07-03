#!/usr/bin/env python
"""Atlas round-1 acceptance — outcome-appearance embeddings (the diversity substrate).

For every harvested walk, render its k3-winner's reframed frame at the classifier's
search fidelity (640x360 ss2, twilight_shifted — the SAME view reframe scored) and
embed it with the v5 backbone's PENULTIMATE features (a forward pass through
`forward_features` + `forward_head(pre_logits=True)` = the 1280-D vector before the
CORN head — the render embedding the diversity carve uses). Save per-arm
{emb[N,1280], walk_id, reward_k3, tag, seed_cx/cy}. The analysis pass measures
outcome-appearance diversity (distinct clusters at a fixed appearance distance) over
these, at matched yield.

Reuses `reframe._render` geometry helpers, `probe.auto_maxiter/BIN/PALETTE/JPG_Q`,
`score_lib.Scorer` (v5), and `location.render_one_flags` (mandelbrot). Renders are
cached on disk so re-runs are cheap.

  uv run python tools/atlas/round1_embed.py --arm arm1   # embeds data/atlas/round1/arm1_table.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from probe import BIN, PALETTE, JPG_Q, auto_maxiter  # noqa: E402
import location as loc_mod  # noqa: E402
from score_lib import Scorer  # noqa: E402

MODEL = "data/classifier/v5/model_best.pt"
D = ROOT / "data" / "atlas" / "round1"
RENDER_W, RENDER_H, RENDER_SS = 640, 360, 2  # reframe search fidelity


def _render(cx, cy, fw, out: Path) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    loc = loc_mod.Location(family="mandelbrot", cx=str(cx), cy=str(cy), fw=str(fw),
                           c_re=None, c_im=None, family_params={})
    cmd = [
        str(BIN), "render-one", "--cx", str(cx), "--cy", str(cy), "--fw", repr(float(fw)),
        "--width", str(RENDER_W), "--height", str(RENDER_H), "--supersample", str(RENDER_SS),
        "--maxiter", str(auto_maxiter(float(fw))), "--palette", PALETTE,
        "--jpg-quality", str(JPG_Q), "--out", str(out),
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists()
    return ok, ("" if ok else r.stderr[-300:])


@torch.no_grad()
def embed_paths(scorer: Scorer, paths: list[Path], batch: int = 64) -> np.ndarray:
    """v5 penultimate features (N,1280) via forward_features + pre_logits head."""
    m = scorer.model
    out = []
    for i in range(0, len(paths), batch):
        pils = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        x = torch.stack([scorer.transform(im) for im in pils]).to(scorer.device)
        if scorer.device != "cpu":
            with torch.autocast(device_type=scorer.device.split(":")[0]):
                feats = m.forward_head(m.forward_features(x), pre_logits=True)
        else:
            feats = m.forward_head(m.forward_features(x), pre_logits=True)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_tags(seeds_path: Path) -> dict:
    """walk_index -> tag from the proposer seed file (row order = walk order)."""
    tags = {}
    if not seeds_path.exists():
        return tags
    for w, line in enumerate(open(seeds_path, encoding="utf-8")):
        line = line.strip()
        if line:
            tags[w] = json.loads(line).get("tag", "")
    return tags


def run(arm: str, workers: int):
    table = D / f"{arm}_table.jsonl"
    if not table.exists():
        raise SystemExit(f"no {table}; harvest {arm} first")
    rows = [json.loads(l) for l in open(table, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["walk_id"])
    tags = load_tags(D / f"{arm}_seeds.jsonl")

    tiles = ROOT / "out" / "atlas" / "round1" / "embed_tiles" / arm
    tiles.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in rows:
        p = tiles / f"walk_{r['walk_id']:04d}.jpg"
        if not p.exists():
            todo.append((r["k3_reframed_cx"], r["k3_reframed_cy"], r["k3_reframed_fw"], p))
    print(f"[{arm}] {len(rows)} walks; rendering {len(todo)} embed tiles @ {RENDER_W}x{RENDER_H} ss{RENDER_SS}")
    t0 = time.time()
    if todo:
        fails = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_render, cx, cy, fw, p): p for cx, cy, fw, p in todo}
            for fut in cf.as_completed(futs):
                ok, err = fut.result()
                if not ok:
                    fails.append((futs[fut], err))
        if fails:
            raise SystemExit(f"render failed ({len(fails)}): {fails[0][0].name}: {fails[0][1]}")
    print(f"  rendered in {time.time()-t0:.0f}s; embedding (v5 penultimate) ...")

    scorer = Scorer(model_path=MODEL)
    paths = [tiles / f"walk_{r['walk_id']:04d}.jpg" for r in rows]
    emb = embed_paths(scorer, paths)
    print(f"  embeddings {emb.shape}")

    np.savez_compressed(
        D / f"{arm}_embed.npz",
        emb=emb.astype(np.float32),
        walk_id=np.array([r["walk_id"] for r in rows]),
        reward_k3=np.array([r["reward_k3"] for r in rows], float),
        reward_k1=np.array([r["reward_k1"] for r in rows], float),
        reached_depth=np.array([r["reached_depth"] for r in rows]),
        seed_cx=np.array([r["seed_cx"] for r in rows], float),
        seed_cy=np.array([r["seed_cy"] for r in rows], float),
        seed_fw=np.array([r["seed_fw"] for r in rows], float),
        tag=np.array([tags.get(r["walk_id"], "") for r in rows]),
    )
    print(f"  -> {D / f'{arm}_embed.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(args.arm, args.workers)


if __name__ == "__main__":
    main()
