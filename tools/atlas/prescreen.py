#!/usr/bin/env python
"""Live-harvest utilities salvaged from the (deleted) atlas round-1/round-2 cluster.

This module holds the two Atlas-free util sets the standing harvest still needs —
nothing here touches the dormant `Atlas` value-map, which is why they outlived the
cluster that once hosted them:

  * depth-2 descendability pre-screen (`prescreen`, `write_seed_list`, `BIN`,
    `SCREEN_*`) — guided-descend's own step-1 run for real, used by
    `production_seeder.py` (and `BIN` alone by `cross_family_shakeout.py`).
  * outcome-appearance render+embed (`_render`, `embed_paths`, `RENDER_*`) — render
    a k3-winner's reframed frame at v5 search fidelity and pull the 1280-D
    penultimate feature, used by `production_seeder.outcome_feature`.

Both sets were formerly `propose.prescreen`/`propose.write_seed_list` and
`round1_embed._render`/`round1_embed.embed_paths`; relocated verbatim (behavior
byte-identical) when the atlas value-map cluster was removed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))   # probe
sys.path.insert(0, str(ROOT / "tools" / "corpus"))          # location
sys.path.insert(0, str(ROOT / "tools" / "mining"))          # score_lib (Scorer type)

from probe import PALETTE, JPG_Q, auto_maxiter  # noqa: E402
import location as loc_mod  # noqa: E402

# The Rust engine (guided-descend / render-one). Same path as probe.BIN / the old
# propose.BIN.
BIN = ROOT / "target" / "release" / "fractal-generator.exe"

# =========================================================================== #
# Depth-2 descendability pre-screen (was propose.prescreen / propose.write_seed_list)
# =========================================================================== #
# Efficient descent config — the pre-screen probe MUST match it so a seed that
# reaches depth 2 in the probe reaches depth 2 in the full descent (identical
# step-1 machinery under the same node/screen params).
SCREEN_NODE_WIDTH = 384
SCREEN_SIGMA_BAND = "8,10,12,14,16"
SCREEN_OCC_FLOOR = 0.321
SCREEN_BLACK_CAP = 0.30


def write_seed_list(path: Path, cx, cy, fw):
    """Write a guided-descend / screen-seeds seed-list JSONL (one cx/cy/fw per row)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b, c in zip(cx, cy, fw):
            f.write(json.dumps({"cx": float(a), "cy": float(b), "fw": float(c)}) + "\n")


def prescreen(cloud: np.ndarray, fw: np.ndarray, workdir: Path,
              node_width: int, occ_floor: float, black_cap: float, seed: int) -> dict:
    """Descendability pre-screen = guided-descend's OWN step-1, run for real.

    A seed's descendability is NOT a property of its (wide) root frame — the occupancy
    floor over-fires there, which is exactly why guided-descend SKIPS it at the d1->d2
    step. Descendability is whether the depth-2 best-of-N step finds a surviving child.
    So the faithful "would this seed survive step-1" screen is to inject the whole
    candidate cloud as `--seed-list` and run a **depth-2 descent probe** (identical
    node/sigma/screen config to the full run, `--per-walk-rng`): walk w reaches depth 2
    iff its seed is descendable. This reuses the descent machinery verbatim (zero parity
    risk) and directly targets the productivity metric (reached_depth >= 2).

    Returns per-candidate `pass` (reached >= 2) in cloud-row order + the death-cause
    tally (the undescendable-frontier diagnostic)."""
    workdir.mkdir(parents=True, exist_ok=True)
    seed_in = workdir / "cloud_seeds.jsonl"
    write_seed_list(seed_in, cloud[:, 0], cloud[:, 1], fw)
    pool = workdir / "probe_pool"
    cmd = [
        str(BIN), "guided-descend",
        "--seed-list", str(seed_in), "--per-walk-rng", "--seed", str(seed),
        "--depth-min", "2", "--depth-max", "2",
        "--node-width", str(node_width), "--sigma-band", SCREEN_SIGMA_BAND,
        "--descent-occ-floor", str(occ_floor), "--descent-black-cap", str(black_cap),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"guided-descend probe failed:\n{r.stderr[-2000:]}")
    walks = {}
    for line in open(pool / "walks.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            w = json.loads(line)
            walks[int(w["walk"])] = w
    if len(walks) != len(cloud):
        raise SystemExit(f"probe walks {len(walks)} != cloud {len(cloud)} (row-order join broken)")
    reached = np.array([walks[i]["reached_depth"] for i in range(len(cloud))])
    passed = reached >= 2
    causes = {}
    for i in range(len(cloud)):
        c = walks[i]["cause"]
        causes[c] = causes.get(c, 0) + 1
    return {"pass": passed, "reached": reached, "causes": causes,
            "probe_stderr_tail": r.stderr.strip().splitlines()[-1:]}


# =========================================================================== #
# Outcome-appearance render + embed (was round1_embed._render / embed_paths)
# =========================================================================== #
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


def embed_paths(scorer, paths: list[Path], batch: int = 64) -> np.ndarray:
    """v5 penultimate features (N,1280) via forward_features + pre_logits head."""
    import torch
    from PIL import Image

    m = scorer.model
    out = []
    with torch.no_grad():
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
