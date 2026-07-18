#!/usr/bin/env python
"""Re-anchor the live morph-novelty penalty knee for the CHEAP-JPG substrate.

The steered frontier's novelty penalty runs on the SAME cheap twilight JPG the v7 forward
scores (free, batched) — NOT the library grayscale morph_gray render. So the library
morph_gray anchors (0.851 typical / 0.974 strict near-dup) do not transfer: they are
grayscale-scale numbers. This pass re-anchors both knees EMPIRICALLY on the cheap-JPG CLIP
substrate, using the pilot's own admissions:

  lower anchor := median pairwise cosine over the pilot's admissions on the cheap substrate
                  (the substrate analog of the library-typical median 0.851 — typical cosine
                  between DISTINCT admitted looks).
  upper anchor := median cheap-substrate cosine over the KNOWN near-repeat pairs — the pairs
                  the pilot morph report flags as the same look (grayscale morph_gray cos >
                  the perceptual cut LOOSE_CUT=0.95). i.e. "what a real near-repeat scores on
                  this substrate."

Both are measured on the exact CLIP recipe the run uses (vit_base_patch16_clip_224.openai over
the color JPG). Near-repeats are (re)identified from the grayscale morph_gray recipe (the
library canon) so the anchor is tied to the same near-repeat semantics as the 0.851/0.938/0.974
yardsticks. Writes `data/atlas/morph_anchors.json`; `steered_frontier.py` loads it (CLI
--morph-lo/--morph-hi override). Report-only calibration — nothing here is on the live path.

  uv run python tools/atlas/morph_anchor_calibrate.py \
      --steered data/discovery/steered_pilot/steered
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import location as loc_mod                                       # noqa: E402
from active_ckpt import BIN, PALETTE, JPG_Q, auto_maxiter        # noqa: E402
from tools.curation.colored_clip import load_clip, embed_clip    # noqa: E402
import tools.studies.steered_pilot_morph as spm                  # noqa: E402

OUT = ROOT / "data" / "atlas" / "morph_anchors.json"
# Cheap substrate geometry (mirror of guided-descend --expand: node_w=384, node_h=9/16, ss1).
CW, CH, CSS = 384, round(384 * 9 / 16), 1


def render_cheap_jpg(loc: loc_mod.Location, tile: Path):
    """Render the cheap twilight substrate (384x216 ss1 twilight_shifted, auto_maxiter) — the
    same presentation the frontier scores/embeds live."""
    tile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "render-one", "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
           "--width", str(CW), "--height", str(CH), "--supersample", str(CSS),
           "--maxiter", str(auto_maxiter(float(loc.fw))), "--palette", PALETTE,
           "--jpg-quality", str(JPG_Q), "--out", str(tile)] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cheap render failed [{tile.name}]: {r.stderr[-300:]}")


def cheap_embeddings(rows, model, tf, tmp: Path) -> np.ndarray:
    """Cheap-substrate CLIP embeddings (L2-normalized) of the admitted-q3 rows, in row order."""
    imgs = []
    for r in rows:
        loc = spm.loc_of_row(r)
        tile = tmp / f"{r['id']}.jpg"
        if not tile.exists():
            render_cheap_jpg(loc, tile)
        imgs.append(Image.open(tile).convert("RGB"))
    E = embed_clip(model, tf, imgs).astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E


def pair_cos(C, pairs):
    return [float(C[a, b]) for a, b in pairs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steered", type=Path, default=ROOT / "data/discovery/steered_pilot/steered")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = spm.admitted_q3(spm.load_jsonl(args.steered / "outcome_ledger.jsonl"))
    if len(rows) < 4:
        raise SystemExit(f"need >=4 pilot admissions to calibrate; got {len(rows)}")
    print(f"pilot admissions: {len(rows)}", flush=True)

    print("loading CLIP ...", flush=True)
    model, tf = load_clip()

    # --- grayscale morph_gray (library canon) -> identify near-repeat pairs ---------------
    tmp_fields = ROOT / "out" / "morph_anchor_fields"
    print("morph_gray embeddings (near-repeat identification) ...", flush=True)
    su, sf, sd, Eg = spm.embed_admissions(rows, tmp_fields, model, tf)
    Cg = spm.cos_matrix(Eg)
    n = len(su)
    near_pairs = [(a, b) for a, b in combinations(range(n), 2) if Cg[a, b] >= spm.LOOSE_CUT]
    strict_pairs = [(a, b) for a, b in combinations(range(n), 2) if Cg[a, b] >= spm.STRICT_CUT]
    print(f"  near-repeat pairs (morph_gray cos>{spm.LOOSE_CUT}): {len(near_pairs)}; "
          f"strict (cos>{spm.STRICT_CUT}): {len(strict_pairs)}", flush=True)

    # --- cheap-JPG substrate embeddings (the live recipe) --------------------------------
    tmp_cheap = ROOT / "out" / "morph_anchor_cheap"
    print("cheap-substrate embeddings (the live penalty recipe) ...", flush=True)
    Ec = cheap_embeddings(rows, model, tf, tmp_cheap)
    Cc = Ec @ Ec.T

    all_pairs = list(combinations(range(n), 2))
    cheap_all = pair_cos(Cc, all_pairs)
    lower = float(np.median(cheap_all))                       # substrate library-typical median

    if near_pairs:
        near_cheap = pair_cos(Cc, near_pairs)
        upper = float(np.median(near_cheap))
        upper_def = f"median cheap cos of {len(near_pairs)} morph_gray-near-repeat pairs"
    else:
        upper = float(np.quantile(cheap_all, 0.95))
        near_cheap = []
        upper_def = "no morph_gray near-repeats; 95th pctile of cheap pairwise cos (fallback)"

    if upper <= lower + 0.01:                                 # guard a degenerate ramp
        upper = lower + 0.05

    # cross-check: typical cosine over a broad sample of the pilot's on-disk --expand JPGs.
    sample_med = None
    cheap_disk = sorted((args.steered / "scratch").glob("expand_b*/**/cheap/*.jpg"))
    if len(cheap_disk) >= 30:
        rng = np.random.default_rng(0)
        pick = [cheap_disk[i] for i in rng.choice(len(cheap_disk), size=min(200, len(cheap_disk)),
                                                  replace=False)]
        Ed = embed_clip(model, tf, [Image.open(p).convert("RGB") for p in pick]).astype(np.float32)
        Ed /= (np.linalg.norm(Ed, axis=1, keepdims=True) + 1e-9)
        Cd = Ed @ Ed.T
        sample_med = float(np.median([Cd[a, b] for a, b in combinations(range(len(pick)), 2)]))

    out = dict(
        substrate="cheap_twilight_jpg (vit_base_patch16_clip_224.openai over color JPG)",
        lo=round(lower, 4), hi=round(upper, 4),
        lower_def="median cheap-substrate pairwise cos over pilot admissions (distinct looks)",
        upper_def=upper_def,
        n_admissions=n, n_near_repeat_pairs=len(near_pairs), n_strict_pairs=len(strict_pairs),
        near_repeat_cheap_cos=[round(x, 4) for x in near_cheap],
        admission_pairwise_cheap_median=round(lower, 4),
        expand_jpg_sample_median=(round(sample_med, 4) if sample_med is not None else None),
        morphgray_ref=dict(loose_cut=spm.LOOSE_CUT, strict_cut=spm.STRICT_CUT,
                           lib_median=spm.LIB_MEDIAN, lib_phoenix=spm.LIB_PHOENIX),
        source=str(args.steered.relative_to(ROOT)).replace("\\", "/"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== MORPH ANCHORS (cheap-JPG substrate) ===")
    print(f"  lo (zero-penalty knee) = {lower:.4f}  [{out['lower_def']}]")
    print(f"  hi (full-penalty knee) = {upper:.4f}  [{upper_def}]")
    if near_cheap:
        print(f"  near-repeat cheap cos: {[round(x,3) for x in near_cheap]}")
    if sample_med is not None:
        print(f"  cross-check (--expand JPG sample median cos): {sample_med:.4f}")
    print(f"  morph_gray refs: lib_median={spm.LIB_MEDIAN} loose={spm.LOOSE_CUT} strict={spm.STRICT_CUT}")
    print(f"  wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
