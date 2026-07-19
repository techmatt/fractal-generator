#!/usr/bin/env python
"""Assemble the frozen-feature matrix for the location preference ranker v0.

Target population = the 96 admissions of the two blind-read batches the ranker will serve:
`data/discovery/steered_run2` (75) + `data/discovery/steered_v1_2_dive` (21). For each admission
three frozen feature blocks are joined by admission id:

  * morph_clip  (768)  grayscale-morphology CLIP  -- from <run>/morph_admissions.npz /
                       dive_admissions.npz (the palette-BLIND "same shape" descriptor).
  * v7          (1280) v7 penultimate on the twilight_shifted canonical search render --
                       from <run>/outcome_feats.npz (== production_seeder.outcome_feature); the
                       one dive admission missing from the store is recomputed here.
  * colored_clip(768)  CLIP on the twilight_shifted canonical color tile the human scored --
                       computed here from the blind tiles (15 unlabeled run2 admissions are
                       rendered first with the same spm.render_colored recipe).

Human scores (1/2/3 == bad/okay/good) come from the two committed blind reads joined through the
hidden manifest keys; unlabeled admissions carry score = 0. Baseline `canon_pgood` and family are
read from each run's outcome ledger.

Prior corpus (`--with-prior`): the older location-label batches, v7 penultimate ONLY. Their crops
carry VARIED delivered palettes, so colored/morph CLIP do NOT share the target's uniform
twilight_shifted appearance space and are deliberately omitted; v7 was trained across those
palettes and transfers. Bounded, balanced subsample -> prior.npz, used strictly for
pretraining/regularization of the v7 head (never in eval).

    uv run python -m tools.ranker.build_features            # target only
    uv run python -m tools.ranker.build_features --with-prior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import tools.studies.steered_pilot_morph as spm            # noqa: E402  loc_of_row / render_colored
import prescreen                                            # noqa: E402  embed_paths (v7 penultimate)
import production_seeder as ps                              # noqa: E402  outcome_feature (missing v7)
from score_lib import Scorer                                # noqa: E402
from tools.curation.colored_clip import load_clip, embed_clip  # noqa: E402

OUT_DIR = ROOT / "data" / "ranker" / "pref_loc_v0"
TILE_DIR = OUT_DIR / "tiles"

BATCHES = [
    dict(name="run2", run_dir=ROOT / "data/discovery/steered_run2",
         morph=ROOT / "data/discovery/steered_run2/morph_admissions.npz",
         key=ROOT / "out/steered_run2_manifest/manifest_key.json",
         scores=ROOT / "labels/steered_run2_blind_scores.json",
         tiles=ROOT / "out/steered_run2_manifest/tiles"),
    dict(name="dive", run_dir=ROOT / "data/discovery/steered_v1_2_dive",
         morph=ROOT / "data/discovery/steered_v1_2_dive/dive_admissions.npz",
         key=ROOT / "out/dive_manifest/manifest_key.json",
         scores=ROOT / "labels/steered_v1_2_dive_blind_scores.json",
         tiles=ROOT / "out/dive_manifest/tiles"),
]

# Prior corpus: (labels json -> batch crop dir). v7-only; bounded balanced subsample.
PRIOR = [
    ("labels/location_labels.json", "data/label_corpus/batches/2026-06-23_flat_generate_loose0_v3"),
    ("labels/location_labels_rev4.json", "data/label_corpus/batches/2026-06-24_guided_descend_rev4"),
    ("labels/location_labels_rev4occfix_v2filtered.json",
     "data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered"),
    ("labels/location_labels_gather_v6.json", "data/label_corpus/batches/2026-07-05_gather_v6"),
    ("labels/location_labels_julia_ladder_j0.json", "data/label_corpus/batches/julia_ladder_j0"),
]
PRIOR_CAP_PER_CLASS = 120   # keeps the v7 pass small and stops the prior swamping 81 target rows


def load_ledger(run_dir: Path) -> dict:
    rows = {}
    for line in open(run_dir / "outcome_ledger.jsonl", encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def build_target(scorer, clip_model, clip_tf) -> dict:
    ids, batch, family, depth, canon_pg, score = [], [], [], [], [], []
    morph = []
    v7_store: dict = {}          # id -> 1280 (from run stores; recompute misses below)
    id2rundir: dict = {}
    tiles: list[Path] = []

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    for b in BATCHES:
        m = np.load(b["morph"], allow_pickle=True)
        muids = list(m["uids"])
        memb = {u: m["emb"][i] for i, u in enumerate(muids)}
        ledger = load_ledger(b["run_dir"])
        of = np.load(b["run_dir"] / "outcome_feats.npz")
        of_have = set(of.files)
        key = {e["tile"]: e for e in json.load(open(b["key"]))["entries"]}
        id2tile = {e["id"]: t for t, e in key.items()}     # blind tile per labeled id
        blind = json.load(open(b["scores"]))
        id2score = {key[t]["id"]: int(s) for t, s in blind.items()}

        for uid in muids:                                   # every admission of this batch
            row = ledger[uid]
            # tile: reuse the blind tile if labeled, else render the same way.
            if uid in id2tile and (b["tiles"] / id2tile[uid]).exists():
                tile = b["tiles"] / id2tile[uid]
            else:
                tile = TILE_DIR / f"{uid}.jpg"
                if not tile.exists():
                    spm.render_colored(spm.loc_of_row(row), tile)
            tiles.append(tile)

            ids.append(uid)
            batch.append(b["name"])
            id2rundir[uid] = b["run_dir"]
            family.append(row.get("family", "mandelbrot"))
            depth.append(int(row.get("reached_depth", 0)))
            canon_pg.append(float(row.get("canon_pgood", np.nan)))
            score.append(id2score.get(uid, 0))
            morph.append(memb[uid].astype(np.float32))
            if uid in of_have:
                v7_store[uid] = of[uid].astype(np.float32)

    # colored-CLIP over all tiles (one forward per tile via the library recipe).
    from PIL import Image
    print(f"colored-CLIP: embedding {len(tiles)} tiles ...", flush=True)
    colored = [embed_clip(clip_model, clip_tf, [Image.open(t)])[0].astype(np.float32) for t in tiles]

    # recompute any missing v7 penultimate from the canonical render (deploy path).
    miss = [i for i in ids if i not in v7_store]
    if miss:
        print(f"v7: recomputing {len(miss)} missing penultimate feature(s): {miss}", flush=True)
        for uid in miss:
            row = load_ledger(id2rundir[uid])[uid]
            tile = TILE_DIR / f"{uid}.jpg"
            if not tile.exists():
                spm.render_colored(spm.loc_of_row(row), tile)
            v7_store[uid] = prescreen.embed_paths(scorer, [tile])[0].astype(np.float32)

    v7 = np.stack([v7_store[i] for i in ids]).astype(np.float32)
    return dict(
        ids=np.array(ids), batch=np.array(batch), family=np.array(family),
        depth=np.array(depth, np.int32), canon_pgood=np.array(canon_pg, np.float32),
        score=np.array(score, np.int32),
        morph=np.stack(morph).astype(np.float32),
        v7=v7,
        colored=np.stack(colored).astype(np.float32),
        tiles=np.array([str(t) for t in tiles]),
    )


def build_prior(scorer, rng) -> dict:
    feats, scores, srcs = [], [], []
    buckets = {1: [], 2: [], 3: []}
    for labf, batchdir in PRIOR:
        labf = ROOT / labf
        batchdir = ROOT / batchdir
        crops = batchdir / "crops"
        if not labf.exists() or not crops.exists():
            print(f"prior: skip {labf.name} (missing)", flush=True)
            continue
        labels = json.load(open(labf))
        for iid, s in labels.items():
            s = int(s)
            if s not in buckets:
                continue
            p = crops / f"{iid}.jpg"
            if p.exists():
                buckets[s].append((p, labf.name))
    picks = []
    for s, lst in buckets.items():
        rng.shuffle(lst)
        picks += [(p, src, s) for p, src in lst[:PRIOR_CAP_PER_CLASS]]
    rng.shuffle(picks)
    if not picks:
        return dict(v7=np.zeros((0, 1280), np.float32), score=np.zeros((0,), np.int32),
                    src=np.array([]))
    print(f"prior: v7-embedding {len(picks)} corpus crops "
          f"({[ (s, sum(1 for _,_,ss in picks if ss==s)) for s in (1,2,3)]}) ...", flush=True)
    paths = [p for p, _, _ in picks]
    v7 = prescreen.embed_paths(scorer, paths).astype(np.float32)
    return dict(v7=v7, score=np.array([s for _, _, s in picks], np.int32),
                src=np.array([src for _, src, _ in picks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-prior", action="store_true")
    ap.add_argument("--scorer", default=str(ps.SCORER_PATH))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"scorer (v7 penultimate): {args.scorer}", flush=True)
    scorer = Scorer(args.scorer)
    print("loading CLIP (vit_base_patch16_clip_224.openai) ...", flush=True)
    clip_model, clip_tf = load_clip()

    tgt = build_target(scorer, clip_model, clip_tf)
    np.savez_compressed(OUT_DIR / "features.npz", **tgt)
    n = len(tgt["ids"])
    lab = int((tgt["score"] > 0).sum())
    print(f"features.npz: {n} admissions ({lab} labeled), "
          f"morph{tgt['morph'].shape[1]} v7{tgt['v7'].shape[1]} colored{tgt['colored'].shape[1]}")

    if args.with_prior:
        prior = build_prior(scorer, np.random.default_rng(args.seed))
        np.savez_compressed(OUT_DIR / "prior.npz", **prior)
        print(f"prior.npz: {len(prior['score'])} corpus rows (v7 only)")


if __name__ == "__main__":
    main()
