"""v3-beam descent — the biased location engine.

Round-based beam search (width k=3) driven by the v3 classifier as per-step
selection pressure, scored at a FIXED neutral palette (twilight_shifted) so
palette preference never enters LOCATION selection. Two seeding engines:

  landmark_mine -- seed from confirmed goods (label>=3, optionally >=2). Perturb
                   each center within perturb_frac*fw, then v3-beam descend.
                   Exploits proven-good self-similar neighborhoods.
  root_mine     -- propose roots inside the bounding region the goods occupy, at
                   a shallow fw band, v3-rank, descend. Broadens beyond landmarks.

Each descent step: for every beam node, generate N child candidates
(content-biased random offset + zoom), render+gate each ONCE at the label
geometry via the frozen `enrich --mode score` (neutral roster, k=1), score with
v3, advance the per-walk top-k. Terminal beam nodes (and the last survivors of a
walk that dies) are harvested with their full selection-bias provenance.

This drives the production `enrich` machinery with aggressive params only; no
production default is touched.

Output: <out>/pool.jsonl (one harvested location per row, with provenance).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from score_lib import Scorer, run_enrich_score  # noqa: E402
import corpus_common as cc  # noqa: E402

LABEL_BATCHES = [
    "2026-06-23_flat_generate_loose0_v3",
    "2026-06-24_guided_descend_rev4",
    "2026-06-24_guided_descend_rev4occfix_v2filtered",
]
HOLDOUT_BATCH = "2026-06-25_scale_2x2_labelset"
HOLDOUT_LABELS = ROOT / "labels" / "scale_2x2_labelset.json"


def gather_seeds(min_label: int = 3):
    """Distinct (cx, cy, fw, image_id, label) seeds with label >= min_label.
    Deduped by (cx, cy, fw) rounded — many goods are zoom-chain neighbors."""
    seeds = {}
    sources = []
    for b in LABEL_BATCHES:
        for r in cc.read_jsonl(os.path.join(cc.batch_dir(b), "images.jsonl")):
            sc = r["label"]["score"]
            if sc is not None and sc >= min_label:
                sources.append((r["render"], r["image_id"], sc))
    # holdout standalone label map -> join to its batch render blocks
    hl = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
    hrows = {r["image_id"]: r for r in
             cc.read_jsonl(os.path.join(cc.batch_dir(HOLDOUT_BATCH), "images.jsonl"))}
    for image_id, sc in hl.items():
        if sc is not None and sc >= min_label and image_id in hrows:
            sources.append((hrows[image_id]["render"], image_id, sc))
    for render, image_id, sc in sources:
        cx, cy, fw = float(render["cx"]), float(render["cy"]), float(render["fw"])
        key = (round(cx, 10), round(cy, 10), f"{fw:.4g}")
        if key not in seeds:
            seeds[key] = dict(cx=cx, cy=cy, fw=fw, image_id=image_id, label=sc)
    return list(seeds.values())


def goods_region(seeds, pad_frac: float = 0.15):
    """Bounding box (re_lo,re_hi,im_lo,im_hi) over the goods' centers, padded."""
    xs = [s["cx"] for s in seeds]
    ys = [s["cy"] for s in seeds]
    rx, ry = max(xs) - min(xs), max(ys) - min(ys)
    px, py = rx * pad_frac + 1e-3, ry * pad_frac + 1e-3
    return (min(xs) - px, max(xs) + px, min(ys) - py, max(ys) + py)


# ---------------------------------------------------------------------------

class Walk:
    __slots__ = ("wid", "src", "seed_id", "perturb_frac", "target_depth",
                 "nodes", "alive")

    def __init__(self, wid, src, seed_id, perturb_frac, target_depth, node0):
        self.wid = wid
        self.src = src
        self.seed_id = seed_id
        self.perturb_frac = perturb_frac
        self.target_depth = target_depth
        self.nodes = [node0]   # list of node dicts (current beam frontier)
        self.alive = True


def node(cx, cy, fw, depth, path, loc_score=None):
    return dict(cx=cx, cy=cy, fw=fw, depth=depth, beam_path=list(path),
                loc_score=loc_score)


def gen_children(rng, parent, n, zoom_lo, zoom_hi, offset_frac):
    """N content-biased child candidates: random offset within the parent frame
    (where the parent was itself v3-selected for content) + a zoom step."""
    out = []
    for _ in range(n):
        child_fw = parent["fw"] * rng.uniform(zoom_lo, zoom_hi)
        r = parent["fw"] * offset_frac * np.sqrt(rng.uniform(0.0, 1.0))
        th = rng.uniform(0.0, 2 * np.pi)
        cx = parent["cx"] + r * np.cos(th)
        cy = parent["cy"] + r * np.sin(th)
        out.append(node(cx, cy, child_fw, parent["depth"] + 1,
                        parent["beam_path"] + [len(out)]))
    return out


def mine_locations(
    scorer: Scorer,
    *,
    out_dir: str,
    neutral_roster: str = "data/mining/neutral_roster.json",
    seed: int = 0,
    # engines
    do_landmark: bool = True,
    do_root: bool = True,
    min_seed_label: int = 3,
    secondary_label2: bool = False,
    # beam
    beam_k: int = 3,
    candidates_n: int = 20,
    zoom_lo: float = 0.35,
    zoom_hi: float = 0.50,
    offset_frac: float = 0.35,
    # landmark engine
    landmark_perturbs: int = 4,
    perturb_lo: float = 0.1,
    perturb_hi: float = 0.5,
    landmark_depth: int = 3,
    # root engine
    root_count: int = 200,
    root_fw_lo: float = 0.003,
    root_fw_hi: float = 0.05,
    root_keep_frac: float = 0.5,
    root_depth: int = 5,
    # render/gate
    width: int = 640,
    height: int = 360,
    maxiter: int = 8000,
    black_cap: float = 0.30,
    occ_floor: float = 0.321,
    log=print,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    seeds = gather_seeds(min_seed_label)
    if secondary_label2:
        seeds += gather_seeds(2)
    log(f"[seeds] {len(seeds)} distinct goods (label>={min_seed_label}"
        f"{'+2' if secondary_label2 else ''})")

    walks: list[Walk] = []
    wid = 0

    if do_landmark:
        for s in seeds:
            for _ in range(landmark_perturbs):
                pf = rng.uniform(perturb_lo, perturb_hi)
                r = s["fw"] * pf * np.sqrt(rng.uniform(0.0, 1.0))
                th = rng.uniform(0.0, 2 * np.pi)
                n0 = node(s["cx"] + r * np.cos(th), s["cy"] + r * np.sin(th),
                          s["fw"], 0, [])
                walks.append(Walk(wid, "landmark_mine", s["image_id"], pf,
                                  landmark_depth, n0))
                wid += 1
        log(f"[landmark] {sum(w.src=='landmark_mine' for w in walks)} walks "
            f"({landmark_perturbs} perturbs x {len(seeds)} seeds)")

    root_walk_ids = []
    if do_root:
        # "Restrict the root box to the boundary arcs the goods occupy": anchor each
        # shallow root window NEAR a randomly chosen good (offset ~ one root frame
        # width), so roots land on the boundary neighborhoods the goods occupy
        # instead of the mostly-void global bbox. Broadens beyond exact landmarks
        # (wider, shallower frames) while staying location-biased + content-rich.
        for _ in range(root_count):
            s = seeds[rng.integers(len(seeds))]
            fw = float(np.exp(rng.uniform(np.log(root_fw_lo), np.log(root_fw_hi))))
            r = fw * rng.uniform(0.0, 1.0)
            th = rng.uniform(0.0, 2 * np.pi)
            cx = s["cx"] + r * np.cos(th)
            cy = s["cy"] + r * np.sin(th)
            n0 = node(cx, cy, fw, 0, [])
            w = Walk(wid, "root_mine", s["image_id"], None, root_depth, n0)
            walks.append(w)
            root_walk_ids.append(wid)
            wid += 1
        log(f"[root] {root_count} root proposals (good-anchored, fw[{root_fw_lo},{root_fw_hi}])")

    # Root pre-rank: score the root proposals at neutral, keep the top tail, so
    # we only descend roots that already sit in v3's upper range.
    harvested: list[dict] = []
    if do_root and root_walk_ids:
        rootmap = {w.wid: w for w in walks if w.wid in root_walk_ids}
        pool = os.path.join(out_dir, "_round_roots.pool.jsonl")
        with open(pool, "w") as f:
            for w in rootmap.values():
                n0 = w.nodes[0]
                f.write(json.dumps(dict(idx=w.wid, cx=n0["cx"], cy=n0["cy"], fw=n0["fw"])) + "\n")
        scores, locs = run_enrich_score(
            scorer, pool, neutral_roster, k=1, seed=seed, width=width, height=height,
            maxiter=maxiter, black_cap=black_cap, occ_floor=occ_floor,
            meta_out=os.path.join(out_dir, "_round_roots.meta.jsonl"), log=log)
        ranked = []
        for l in locs:
            idx = l["idx"]
            if l["gated"] or idx not in scores:
                rootmap[idx].alive = False
                continue
            s = scores[idx][0][0]  # loc_score [0,2]
            rootmap[idx].nodes[0]["loc_score"] = s
            ranked.append((s, idx))
        ranked.sort(reverse=True)
        keep = set(i for _, i in ranked[:max(1, int(len(ranked) * root_keep_frac))])
        for idx, w in rootmap.items():
            if idx not in keep:
                w.alive = False
        log(f"[root] pre-rank kept {len(keep)}/{len(rootmap)} roots above v3 tail")

    # ---- round loop: advance every alive walk one depth -------------------
    round_no = 0
    while any(w.alive and w.nodes and w.nodes[0]["depth"] < w.target_depth for w in walks):
        active = [w for w in walks if w.alive and w.nodes
                  and w.nodes[0]["depth"] < w.target_depth]
        # build the candidate pool across all active walks
        pool = os.path.join(out_dir, f"_round_{round_no}.pool.jsonl")
        idx2cand: dict[int, tuple] = {}   # idx -> (walk, child_node)
        gidx = 0
        with open(pool, "w") as f:
            for w in active:
                for parent in w.nodes:
                    for child in gen_children(rng, parent, candidates_n,
                                              zoom_lo, zoom_hi, offset_frac):
                        idx2cand[gidx] = (w, child)
                        f.write(json.dumps(dict(idx=gidx, cx=child["cx"],
                                                cy=child["cy"], fw=child["fw"])) + "\n")
                        gidx += 1
        log(f"[round {round_no}] {len(active)} walks, {gidx} candidates")
        scores, locs = run_enrich_score(
            scorer, pool, neutral_roster, k=1, seed=seed, width=width, height=height,
            maxiter=maxiter, black_cap=black_cap, occ_floor=occ_floor,
            meta_out=os.path.join(out_dir, f"_round_{round_no}.meta.jsonl"),
            log=lambda *_: None)
        # collect surviving scored children per walk
        per_walk: dict[int, list] = {}
        for l in locs:
            idx = l["idx"]
            w, child = idx2cand[idx]
            if l["gated"] or idx not in scores:
                continue
            child["loc_score"] = scores[idx][0][0]
            child["black_fraction"] = l.get("black_fraction")
            child["occupancy"] = l.get("occupancy")
            per_walk.setdefault(w.wid, []).append(child)
        # advance each walk's top-k; walks with no survivors die and harvest parents
        for w in active:
            kids = per_walk.get(w.wid, [])
            if not kids:
                for p in w.nodes:
                    harvested.append(_harvest(w, p, terminal=False))
                w.alive = False
                continue
            kids.sort(key=lambda c: c["loc_score"], reverse=True)
            w.nodes = kids[:beam_k]
            if w.nodes[0]["depth"] >= w.target_depth:
                for p in w.nodes:
                    harvested.append(_harvest(w, p, terminal=True))
                w.alive = False
        round_no += 1

    # any walks that ended exactly at target via the loop are already harvested;
    # harvest any still-alive frontier (defensive)
    for w in walks:
        if w.alive and w.nodes:
            for p in w.nodes:
                harvested.append(_harvest(w, p, terminal=True))
            w.alive = False

    # write the harvested pool
    pool_out = os.path.join(out_dir, "pool.jsonl")
    with open(pool_out, "w") as f:
        for i, h in enumerate(harvested):
            h["idx"] = i
            f.write(json.dumps(h) + "\n")
    log(f"[harvest] {len(harvested)} locations -> {pool_out}")
    return harvested


def _harvest(w: Walk, p: dict, terminal: bool) -> dict:
    return dict(
        cx=p["cx"], cy=p["cy"], fw=p["fw"], depth=p["depth"],
        source=w.src, seed_landmark_id=w.seed_id, perturbation_frac=w.perturb_frac,
        target_depth=w.target_depth, walk_id=w.wid, beam_path=p["beam_path"],
        loc_score=p["loc_score"], terminal=terminal,
        black_fraction=p.get("black_fraction"), occupancy=p.get("occupancy"),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mining/run1/descent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-landmark", action="store_true")
    ap.add_argument("--no-root", action="store_true")
    ap.add_argument("--secondary-label2", action="store_true")
    ap.add_argument("--beam-k", type=int, default=3)
    ap.add_argument("--candidates-n", type=int, default=20)
    ap.add_argument("--landmark-perturbs", type=int, default=4)
    ap.add_argument("--landmark-depth", type=int, default=3)
    ap.add_argument("--root-count", type=int, default=200)
    ap.add_argument("--root-depth", type=int, default=5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--maxiter", type=int, default=8000)
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else str(ROOT / a.out)
    t0 = time.time()
    scorer = Scorer()
    mine_locations(
        scorer, out_dir=out, seed=a.seed,
        do_landmark=not a.no_landmark, do_root=not a.no_root,
        secondary_label2=a.secondary_label2, beam_k=a.beam_k,
        candidates_n=a.candidates_n, landmark_perturbs=a.landmark_perturbs,
        landmark_depth=a.landmark_depth, root_count=a.root_count,
        root_depth=a.root_depth, width=a.width, height=a.height, maxiter=a.maxiter)
    print(f"descent done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
