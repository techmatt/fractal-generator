"""Finalize a v3-beam descent into a biased, deduped, palette-stratified,
T2-gated label-corpus batch.

Stages (yield funnel reported at each):
  1. cap     -- take the top `cap_locations` harvested locations by neutral
                loc_score (bounds the spread-scoring cost).
  2. spread  -- score every (location x spread-roster palette) with v3 through the
                frozen `enrich --mode score`, capturing each frame's grayscale
                pHash for dedup. Each location is scored against all roster
                families, so the gate -- not the generator -- decides palette mix.
  3. gate    -- keep (location, palette) units whose v3 gate score >= T2
                (calibrate_t2.py). The labeling pool is exactly the >=T2 set.
  4. stratify+dedup -- group passers by palette family, allocate the label budget
                across families, take the top-v3 per family, dropping near-dups
                (vs each other AND the already-labeled corpus) as we go.
  5. render  -- ss4 Lanczos3 1280x720 crops for the survivors (`enrich --mode
                render`); re-score the actual crop for the provenance trail.
  6. batch   -- write data/label_corpus/batches/<batch_id>/{images.jsonl,
                batch.json} with the full selection-bias provenance, biased=True.

  uv run python tools/mining/harvest.py --descent data/mining/run1/descent/pool.jsonl
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from score_lib import Scorer, run_enrich_score, BIN  # noqa: E402
from dedup import phash, DedupIndex  # noqa: E402
import corpus_common as cc  # noqa: E402

SPREAD_ROSTER = ROOT / "data" / "mining" / "spread_roster.json"
FAMILIES = ROOT / "data" / "mining" / "palette_families.json"
T2_CONFIG = ROOT / "data" / "mining" / "t2_calibration.json"
CORPUS_HASH_CACHE = ROOT / "data" / "mining" / "corpus_hashes.json"
LABEL_BATCHES = [
    "2026-06-23_flat_generate_loose0_v3", "2026-06-24_guided_descend_rev4",
    "2026-06-24_guided_descend_rev4occfix_v2filtered",
    "2026-06-25_scale_2x2_labelset",
]
GENERATOR_VERSION = "mining_v3guided_v1"
NEUTRAL_PALETTE = "twilight_shifted"


def corpus_crop_hashes(log=print):
    """pHash every already-labeled crop (cached). Mined crops are deduped against
    these so Matt never re-judges a known location."""
    from PIL import Image
    if CORPUS_HASH_CACHE.exists():
        return [int(h) for h in json.loads(CORPUS_HASH_CACHE.read_text())]
    hashes = []
    for b in LABEL_BATCHES:
        cdir = os.path.join(cc.batch_dir(b), "crops")
        if not os.path.isdir(cdir):
            continue
        files = [f for f in os.listdir(cdir) if f.endswith(".jpg")]
        for i, f in enumerate(files):
            with Image.open(os.path.join(cdir, f)) as im:
                im.load()
                hashes.append(phash(im))
        log(f"  hashed {len(files)} corpus crops in {b}")
    CORPUS_HASH_CACHE.write_text(json.dumps([str(h) for h in hashes]))
    log(f"[dedup] {len(hashes)} corpus crop hashes cached")
    return hashes


def finalize(
    scorer: Scorer,
    descent_pool: str,
    *,
    batch_id: str,
    out_batch_dir: str,
    cap_locations: int = 800,
    budget: int = 450,
    width: int = 1280,
    height: int = 720,
    spread_width: int = 640,
    spread_height: int = 360,
    maxiter: int = 8000,
    render_ss: int = 4,
    jpg_quality: int = 90,
    dedup_thresh: int = 6,
    seed: int = 0,
    log=print,
):
    work = os.path.join(out_batch_dir, "_work")
    os.makedirs(work, exist_ok=True)
    roster = json.loads(SPREAD_ROSTER.read_text())
    fam_map = json.loads(FAMILIES.read_text())
    t2cfg = json.loads(T2_CONFIG.read_text())
    score_kind, t2 = t2cfg["score_kind"], t2cfg["t2"]
    log(f"[gate] score_kind={score_kind} T2={t2:.6g}  roster={len(roster)} palettes  budget={budget}")

    # 1. cap ----------------------------------------------------------------
    locs = [json.loads(l) for l in open(descent_pool)]
    locs.sort(key=lambda r: r.get("loc_score") or -1, reverse=True)
    locs = locs[:cap_locations]
    locmap = {i: r for i, r in enumerate(locs)}
    pool = os.path.join(work, "spread_pool.jsonl")
    with open(pool, "w") as f:
        for i, r in locmap.items():
            f.write(json.dumps(dict(idx=i, cx=r["cx"], cy=r["cy"], fw=r["fw"])) + "\n")
    log(f"[cap] {len(locs)} locations (top by loc_score) -> spread scoring")

    # 2. spread-score, capturing a pHash per streamed frame -----------------
    hashes: dict[tuple, int] = {}

    def cap_frame(idx, ki, pil):
        hashes[(idx, ki)] = phash(pil)

    scores, meta = run_enrich_score(
        scorer, pool, str(SPREAD_ROSTER), k=len(roster), seed=seed,
        width=spread_width, height=spread_height, maxiter=maxiter,
        meta_out=os.path.join(work, "spread_meta.jsonl"), frame_cb=cap_frame, log=log)
    metamap = {m["idx"]: m for m in meta}
    n_gated_loc = sum(1 for m in meta if m["gated"])
    log(f"[spread] {len(meta)} locations scored, {n_gated_loc} gate-failed (black/occ)")

    # 3. gate: build (loc, palette) units that clear T2 ---------------------
    units = []
    for idx, m in metamap.items():
        if m["gated"]:
            continue
        palettes = m.get("palettes", [])
        sc = scores.get(idx, {})
        for ki, pname in enumerate(palettes):
            if ki not in sc:
                continue
            s2, pnb, pg = sc[ki]
            gate_val = pnb if score_kind == "p_notbad" else s2
            if gate_val < t2:
                continue
            units.append(dict(idx=idx, ki=ki, palette=pname,
                              family=fam_map.get(pname, "unknown"),
                              score=s2, p_notbad=pnb, p_good=pg,
                              gate_score=gate_val, hash=hashes.get((idx, ki))))
    n_pass = len(units)
    by_fam_pass = Counter(u["family"] for u in units)
    log(f"[gate] {n_pass} (loc x palette) units clear T2  families={dict(by_fam_pass)}")

    # 4. stratify across families + dedup, allocate the budget --------------
    fams = [f for f in ("warm", "cool", "cyclic", "diverging", "mono")
            if by_fam_pass.get(f)]
    dedup = DedupIndex(dedup_thresh)
    dedup.seed(corpus_crop_hashes(log))
    per_fam_units = defaultdict(list)
    for u in units:
        per_fam_units[u["family"]].append(u)
    for f in per_fam_units:
        per_fam_units[f].sort(key=lambda u: u["p_notbad"], reverse=True)

    # equal allocation across present families, round-robin best-first, skipping
    # near-dups (a location may already be represented under another palette)
    selected = []
    seen_loc_pal = set()
    quota = {f: budget // max(1, len(fams)) for f in fams}
    cursor = {f: 0 for f in fams}
    n_dup = 0
    progressing = True
    while len(selected) < budget and progressing:
        progressing = False
        for f in fams:
            if len([s for s in selected if s["family"] == f]) >= quota[f]:
                continue
            lst = per_fam_units[f]
            while cursor[f] < len(lst):
                u = lst[cursor[f]]
                cursor[f] += 1
                progressing = True
                if (u["idx"], u["ki"]) in seen_loc_pal:
                    continue
                h = u["hash"]
                if h is not None and not dedup.add(h):
                    n_dup += 1
                    continue
                seen_loc_pal.add((u["idx"], u["ki"]))
                selected.append(u)
                break
    # fill remaining budget from any family if quotas left gaps
    if len(selected) < budget:
        leftovers = []
        for f in fams:
            leftovers.extend(per_fam_units[f][cursor[f]:])
        leftovers.sort(key=lambda u: u["p_notbad"], reverse=True)
        for u in leftovers:
            if len(selected) >= budget:
                break
            if (u["idx"], u["ki"]) in seen_loc_pal:
                continue
            h = u["hash"]
            if h is not None and not dedup.add(h):
                n_dup += 1
                continue
            seen_loc_pal.add((u["idx"], u["ki"]))
            selected.append(u)
    log(f"[stratify+dedup] selected {len(selected)} units, dropped {n_dup} near-dups  "
        f"families={dict(Counter(s['family'] for s in selected))}")

    # 5. render ss4 crops + re-score the actual crop ------------------------
    sel_path = os.path.join(work, "selection.jsonl")
    crops_dir = os.path.join(out_batch_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    rows_meta = []
    with open(sel_path, "w") as f:
        for u in selected:
            loc = locmap[u["idx"]]
            image_id = _image_id(u["idx"], loc, u["palette"])
            u["image_id"] = image_id
            f.write(json.dumps(dict(image_id=image_id, cx=loc["cx"], cy=loc["cy"],
                                    fw=loc["fw"], palette=u["palette"])) + "\n")
    cmd = [str(ROOT / BIN), "enrich", "--mode", "render", "--selection", sel_path,
           "--colormaps", str(SPREAD_ROSTER), "--crops-dir", crops_dir,
           "--width", str(width), "--height", str(height), "--render-ss", str(render_ss),
           "--maxiter", str(maxiter), "--jpg-quality", str(jpg_quality)]
    log("[render] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    # re-score the actual ss4 crops (provenance filter_score on the labeling crop)
    crop_paths = [os.path.join(crops_dir, u["image_id"] + ".jpg") for u in selected]
    present = [p for p in crop_paths if os.path.exists(p)]
    log(f"[render] {len(present)}/{len(selected)} crops written")
    crop_score = {}
    if present:
        triples = scorer.score_paths(present)
        for p, t in zip(present, triples):
            crop_score[os.path.basename(p)[:-4]] = t

    # 6. batch images.jsonl + batch.json ------------------------------------
    rows = []
    for u in selected:
        loc = locmap[u["idx"]]
        image_id = u["image_id"]
        cs = crop_score.get(image_id)
        render = cc.render_block(
            cx=loc["cx"], cy=loc["cy"], fw=loc["fw"], maxiter=maxiter,
            palette=u["palette"], composition="center", width=width, height=height,
            ss=render_ss, filter="lanczos3", interior_mode="black")
        prov = cc.provenance_block(
            GENERATOR_VERSION, batch_id,
            source=loc["source"], seed_landmark_id=loc.get("seed_landmark_id"),
            perturbation_frac=loc.get("perturbation_frac"),
            beam_path=loc.get("beam_path"), depth=loc.get("depth"),
            target_depth=loc.get("target_depth"), walk_id=loc.get("walk_id"),
            loc_score=loc.get("loc_score"), location_score_palette=NEUTRAL_PALETTE,
            palette_family=u["family"], gate_kind=t2cfg["score_kind"], gate_t2=t2,
            gate_score=u["gate_score"], biased=True, v3_model_id="data/classifier/v3/model_best.pt",
            black_fraction=loc.get("black_fraction"), occupancy=loc.get("occupancy"),
            filter_score=(cs[1] if cs else None),  # P(not-bad) on the actual crop
            argmax_palette=u["palette"])
        rows.append(cc.make_row(image_id, render, prov, cc.label_block()))

    images_path = os.path.join(out_batch_dir, "images.jsonl")
    cc.write_jsonl(rows, images_path)
    # empty scores.json so corpus_label.html has a target to write back to
    sj = os.path.join(out_batch_dir, "scores.json")
    if not os.path.exists(sj):
        Path(sj).write_text("{}")

    funnel = dict(
        descent_locations=len(open(descent_pool).readlines()),
        capped=len(locs), spread_scored=len(meta), gate_failed_loc=n_gated_loc,
        units_pass_t2=n_pass, dropped_near_dups=n_dup, selected=len(selected),
        crops_written=len(present), budget=budget,
        gate=dict(score_kind=score_kind, t2=t2),
        family_pass=dict(by_fam_pass), family_selected=dict(Counter(s["family"] for s in selected)),
        by_source=dict(Counter(locmap[u["idx"]]["source"] for u in selected)),
    )
    batch_json = dict(
        batch_id=batch_id, schema_version=1, generator_version=GENERATOR_VERSION,
        created=None, labeler=None, biased=True,
        note="v3-guided BIASED prospecting harvest. Positive-enriched; usable for "
             "growing positives, NOT for unbiased eval. Location selected at neutral "
             f"{NEUTRAL_PALETTE}; palette spread + T2 gate for labeling pool.",
        source_run=descent_pool, sampling=dict(cap_locations=cap_locations, budget=budget,
            spread_geometry=[spread_width, spread_height], render_geometry=[width, height],
            render_ss=render_ss, dedup_thresh=dedup_thresh, seed=seed),
        gate=dict(score_kind=score_kind, t2=t2, method=t2cfg.get("method")),
        funnel=funnel)
    Path(os.path.join(out_batch_dir, "batch.json")).write_text(json.dumps(batch_json, indent=2))

    _report_funnel(funnel, log)
    return funnel


def _image_id(idx, loc, palette):
    safe = palette.replace("/", "_")
    return f"{loc['source'][:4]}_{idx}_{safe}"


def _report_funnel(f, log):
    log("\n===================== YIELD FUNNEL =====================")
    log(f"  descent locations      {f['descent_locations']}")
    log(f"  capped (top loc_score) {f['capped']}")
    log(f"  spread-scored          {f['spread_scored']}  (gate-failed loc: {f['gate_failed_loc']})")
    log(f"  units clearing T2      {f['units_pass_t2']}")
    log(f"  dropped near-dups      {f['dropped_near_dups']}")
    log(f"  SELECTED for labeling  {f['selected']}  (crops written: {f['crops_written']})")
    log(f"  by source              {f['by_source']}")
    log(f"  palette family (pass)  {f['family_pass']}")
    log(f"  palette family (final) {f['family_selected']}")
    log("=======================================================")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--descent", default="data/mining/run1/descent/pool.jsonl")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--cap-locations", type=int, default=800)
    ap.add_argument("--budget", type=int, default=450)
    ap.add_argument("--spread-width", type=int, default=640)
    ap.add_argument("--spread-height", type=int, default=360)
    ap.add_argument("--dedup-thresh", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default="2026-06-25")
    a = ap.parse_args()
    batch_id = a.batch_id or f"{a.date}_{GENERATOR_VERSION}"
    out = cc.batch_dir(batch_id)
    descent = a.descent if os.path.isabs(a.descent) else str(ROOT / a.descent)
    t0 = time.time()
    scorer = Scorer()
    finalize(scorer, descent, batch_id=batch_id, out_batch_dir=out,
             cap_locations=a.cap_locations, budget=a.budget,
             spread_width=a.spread_width, spread_height=a.spread_height,
             dedup_thresh=a.dedup_thresh, seed=a.seed)
    print(f"\nBATCH_ID: {batch_id}\nbatch dir: {out}\nfinalize done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
