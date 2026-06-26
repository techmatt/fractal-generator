#!/usr/bin/env python
"""Julia batch ladder generator -> J0 unlabeled corpus batch.

Phased driver (see prompts/julia_ladder_generator.md):

  enumerate : build deduped seed pool -> jittered candidates -> center-zoom +
              descent rungs (descent shells out to `guided-descend --julia`).
              Writes _work/rungs.jsonl (params only, no preview render).
  render    : render a 1280x720 preview JPG per rung via `render-one --julia`.
  score     : v4 CORN score every preview; writes _work/scores.jsonl.
  assemble  : dedup (c-aware union-find) -> stratified sample 1000 -> batch
              images.jsonl (3-block schema, label null) + montage + report.

No auto-labels. No fold/retrain. v4 scores are ranking/provenance only.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
MANIFEST = ROOT / "data" / "v4" / "manifest.jsonl"
BATCH_ID = "julia_ladder_j0"
BATCH_DIR = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
WORK = BATCH_DIR / "_work"
PREVIEW_DIR = WORK / "previews"
GD_DIR = WORK / "gd"
CROPS_DIR = BATCH_DIR / "crops"

# --- generation params (LOCKED) ---
ROOT_FW = 3.0                  # guided-descend --julia-root-fw base scale (z-plane)
PALETTE = "twilight_shifted"   # single consistent preview palette
CENTER_FACTORS = [4, 8]        # center-zoom rungs: fw = ROOT_FW / factor  (>=4x, then 2x)
JITTER_SIGMA_FRAC = 0.10       # c-jitter sigma as fraction of source Mandelbrot fw
JITTER_PER_SEED = 1            # extra jittered center-zoom candidates per seed
DESCENT_WALKS = 1
DESCENT_DMIN = 3
DESCENT_DMAX = 5
PREVIEW_W, PREVIEW_H, PREVIEW_SS = 1280, 720, 2
SAMPLE_N = 1000
PER_SEED_CAP = 4               # stratified-sample cap per seed neighborhood
SEED = 20260625

# §5 union-find predicate constants (faithful to tools/v4/assemble.py)
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5

# depth-aware maxiter (mirrors tools/explorer/app.py auto_maxiter; fw_home=ROOT_FW)
MAXITER_BASE, MAXITER_K, MAXITER_MIN, MAXITER_MAX = 500, 0.30, 200, 8000


def auto_maxiter(fw: float) -> int:
    lz = math.log2(ROOT_FW / fw) if fw > 0 else 0.0
    val = MAXITER_BASE * (1.0 + MAXITER_K * lz)
    return int(max(MAXITER_MIN, min(MAXITER_MAX, val)))


# ---------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster(cx, cy, fw):
    """§5 neighborhood union-find. Returns dense group ids (list, len n)."""
    n = len(cx)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = fw[i] / fw[j]
            if ratio < SCALE_LO or ratio > SCALE_HI:
                continue
            tol = SHIFT_FRAC * min(fw[i], fw[j])
            dx, dy = cx[i] - cx[j], cy[i] - cy[j]
            if dx * dx + dy * dy <= tol * tol:
                uf.union(i, j)
    roots = {}
    out = []
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return out


# --------------------------------------------------------------- enumerate
def load_seeds():
    rows = [json.loads(l) for l in open(MANIFEST)]
    seeds = [r for r in rows if r.get("label") in (2, 3)]
    cx = [float(s["cx"]) for s in seeds]
    cy = [float(s["cy"]) for s in seeds]
    fw = [float(s["fw"]) for s in seeds]
    gid = cluster(cx, cy, fw)
    # one representative per neighborhood (first seen)
    reps = {}
    for s, g in zip(seeds, gid):
        if g not in reps:
            reps[g] = s
    return reps  # {group_id: seed_row}


def run_descent(group, seed_row, rng_seed):
    """Shell out guided-descend --julia for one seed c. Returns list of rung dicts."""
    out_dir = GD_DIR / f"g{group:04d}"
    pool = out_dir / "pool.jsonl"
    cx, cy = seed_row["cx"], seed_row["cy"]
    if not pool.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(BIN), "guided-descend", "--julia", "--c", str(cx), str(cy),
            "--julia-root-fw", str(ROOT_FW),
            "--n-walks", str(DESCENT_WALKS),
            "--depth-min", str(DESCENT_DMIN), "--depth-max", str(DESCENT_DMAX),
            "--node-width", "384", "--preview-width", "96",
            "--seed", str(rng_seed),
            "--out-dir", str(out_dir),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not pool.exists():
            sys.stderr.write(f"[descent g{group}] FAILED: {r.stderr[-300:]}\n")
            return []
    rungs = []
    for line in open(pool):
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        if p["depth"] <= 1:          # drop the deterministic (0,0,ROOT_FW) root rung
            continue
        rungs.append({
            "mode": "descent",
            "rung_index": p["depth"],
            "cx": repr(p["cx"]), "cy": repr(p["cy"]), "fw": repr(p["fw"]),
            "branch": p.get("branch"), "placement": p.get("placement"),
            "focus_score": p.get("focus_score"),
            "walk": p.get("walk"),
        })
    return rungs


def enumerate_rungs(args):
    rng = random.Random(SEED)
    reps = load_seeds()
    groups = sorted(reps)
    print(f"label-2/3 seeds deduped -> {len(groups)} neighborhoods")

    # --- candidate c list: un-jittered + jittered (center-zoom); un-jittered (descent)
    rungs = []
    uid = 0
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # descent: parallel guided-descend over un-jittered seeds
    print(f"running guided-descend (julia) on {len(groups)} seeds ...")
    descent_map = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_descent, g, reps[g], (SEED + g) & 0xFFFFFFFF): g
                for g in groups}
        done = 0
        for fut in cf.as_completed(futs):
            g = futs[fut]
            descent_map[g] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(groups):
                print(f"  descent {done}/{len(groups)}")

    for g in groups:
        s = reps[g]
        src_fw = float(s["fw"])
        base_cx, base_cy = float(s["cx"]), float(s["cy"])
        sig = JITTER_SIGMA_FRAC * src_fw
        # center-zoom candidates: un-jittered + JITTER_PER_SEED jittered
        cand_cs = [(s["cx"], s["cy"], False, 0.0)]
        for _ in range(JITTER_PER_SEED):
            jx = base_cx + rng.gauss(0, sig)
            jy = base_cy + rng.gauss(0, sig)
            cand_cs.append((repr(jx), repr(jy), True, sig))
        for ci, (c_re, c_im, jit, jmag) in enumerate(cand_cs):
            for fac in CENTER_FACTORS:
                fw = ROOT_FW / fac
                rungs.append(_mk(uid, g, s, c_re, c_im, jit, jmag, ci,
                                 "center_zoom", fac, "0.0", "0.0", repr(fw),
                                 None, None, None, None))
                uid += 1
        # descent rungs (un-jittered c only)
        for d in descent_map.get(g, []):
            rungs.append(_mk(uid, g, s, s["cx"], s["cy"], False, 0.0, 0,
                             d["mode"], d["rung_index"], d["cx"], d["cy"], d["fw"],
                             d["branch"], d["placement"], d["focus_score"], d["walk"]))
            uid += 1

    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "rungs.jsonl", "w") as f:
        for r in rungs:
            f.write(json.dumps(r) + "\n")

    n_center = sum(1 for r in rungs if r["mode"] == "center_zoom")
    n_descent = sum(1 for r in rungs if r["mode"] == "descent")
    print(f"\nRUNG POOL: {len(rungs)} total  "
          f"(center_zoom={n_center}, descent={n_descent})")
    print(f"seeds={len(groups)}  center_candidates/seed={1+JITTER_PER_SEED}  "
          f"center_factors={CENTER_FACTORS}")
    print(f"wrote {WORK/'rungs.jsonl'}")


def _mk(uid, g, s, c_re, c_im, jit, jmag, cand_id, mode, rung_index,
        cx, cy, fw, branch, placement, focus_score, walk):
    return {
        "rung_uid": uid,
        "seed_group": g,
        "seed_label": s["label"],
        "seed_source": s.get("source"),
        "src_cx": s["cx"], "src_cy": s["cy"], "src_fw": s["fw"],
        "c_re": c_re, "c_im": c_im,
        "jitter": jit, "jitter_mag": jmag, "candidate_id": cand_id,
        "mode": mode, "rung_index": rung_index,
        "cx": cx, "cy": cy, "fw": fw,
        "maxiter": auto_maxiter(float(fw)),
        "palette": PALETTE,
        "branch": branch, "placement": placement, "focus_score": focus_score,
        "walk": walk,
    }


# ----------------------------------------------------------------- render
def _render_one(r):
    out = PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg"
    if out.exists():
        return True
    cmd = [
        str(BIN), "render-one", "--julia", "--c", r["c_re"], r["c_im"],
        "--cx", r["cx"], "--cy", r["cy"], "--fw", r["fw"],
        "--width", str(PREVIEW_W), "--height", str(PREVIEW_H),
        "--supersample", str(PREVIEW_SS), "--maxiter", str(r["maxiter"]),
        "--palette", PALETTE, "--out", str(out),
    ]
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    if r2.returncode != 0 or not out.exists():
        sys.stderr.write(f"[render {r['rung_uid']}] FAILED: {r2.stderr[-200:]}\n")
        return False
    return True


def render(args):
    rungs = [json.loads(l) for l in open(WORK / "rungs.jsonl")]
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rungs if not (PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg").exists()]
    print(f"rendering {len(todo)}/{len(rungs)} previews "
          f"({PREVIEW_W}x{PREVIEW_H} ss{PREVIEW_SS}) ...")
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_render_one, todo):
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"  rendered {done}/{len(todo)}")
    print("render done")


# ------------------------------------------------------------------ score
def score(args):
    sys.path.insert(0, str(ROOT))
    import torch  # noqa
    from classifier.data import Transform
    from classifier.model import build_model
    import numpy as np
    from PIL import Image

    ckpt = torch.load(ROOT / "data/classifier/v4/model_best.pt",
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(target="ordinal", drop_rate=cfg.get("drop_rate", 0.2),
                        drop_path_rate=cfg.get("drop_path_rate", 0.1),
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    tf = Transform(cfg["geometry"], cfg["interpolation"], tuple(cfg["mean"]),
                   tuple(cfg["std"]), train=False)

    rungs = [json.loads(l) for l in open(WORK / "rungs.jsonl")]
    paths = [(r["rung_uid"], PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg") for r in rungs]
    paths = [(u, p) for u, p in paths if p.exists()]
    print(f"scoring {len(paths)} previews with v4 ...")

    out = {}
    B = 64
    with torch.no_grad():
        for i in range(0, len(paths), B):
            chunk = paths[i:i + B]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for _, p in chunk]).to(dev)
            logits = model(x).float().cpu().numpy()
            pp = 1.0 / (1.0 + np.exp(-logits))
            for (u, _), lg in zip(chunk, pp):
                out[u] = {"v4_p_not_bad": float(lg[0]), "v4_p_good": float(lg[1]),
                          "v4_score": float(lg[0] + lg[1])}
            if (i // B) % 10 == 0:
                print(f"  scored {min(i+B, len(paths))}/{len(paths)}")
    with open(WORK / "scores.jsonl", "w") as f:
        for u, s in out.items():
            f.write(json.dumps({"rung_uid": u, **s}) + "\n")
    sc = sorted(s["v4_score"] for s in out.values())
    print(f"v4_score: min={sc[0]:.3f} med={sc[len(sc)//2]:.3f} max={sc[-1]:.3f}")
    print(f"wrote {WORK/'scores.jsonl'}")


# --------------------------------------------------------------- assemble
C_TOL_FRAC = 0.05   # c-cluster tol = C_TOL_FRAC * src_fw (half the jitter sigma)


def dedup(rungs):
    """c-aware §5 dedup. Returns list of survivor rungs (one per
    (seed_group, c-cluster, (cx,cy,fw)-cluster))."""
    survivors = []
    # bucket by seed_group (c only collides within a seed neighborhood)
    from collections import defaultdict
    by_seed = defaultdict(list)
    for r in rungs:
        by_seed[r["seed_group"]].append(r)
    for g, group in by_seed.items():
        # (1) c-cluster within seed: union-find on candidate c, tol = 0.05*src_fw
        src_fw = float(group[0]["src_fw"])
        ctol = C_TOL_FRAC * src_fw
        cre = [float(r["c_re"]) for r in group]
        cim = [float(r["c_im"]) for r in group]
        uf = UF(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dx, dy = cre[i] - cre[j], cim[i] - cim[j]
                if dx * dx + dy * dy <= ctol * ctol:
                    uf.union(i, j)
        cclusters = defaultdict(list)
        for i, r in enumerate(group):
            cclusters[uf.find(i)].append(r)
        # (2) §5 (cx,cy,fw) union-find within each c-cluster
        for cc in cclusters.values():
            cx = [float(r["cx"]) for r in cc]
            cy = [float(r["cy"]) for r in cc]
            fw = [float(r["fw"]) for r in cc]
            gid = cluster(cx, cy, fw)
            seen = {}
            for r, k in zip(cc, gid):
                if k not in seen:
                    seen[k] = r          # first survivor wins
            survivors.extend(seen.values())
    return survivors


def stratified_sample(rungs, n, cap, rng):
    """Round-robin across seed neighborhoods, alternating mode, per-seed cap."""
    from collections import defaultdict
    by_seed = defaultdict(lambda: {"center_zoom": [], "descent": []})
    for r in rungs:
        by_seed[r["seed_group"]][r["mode"]].append(r)
    seeds = list(by_seed)
    rng.shuffle(seeds)
    for g in seeds:
        rng.shuffle(by_seed[g]["center_zoom"])
        rng.shuffle(by_seed[g]["descent"])
    picked = []
    taken = defaultdict(int)
    rnd = 0
    while len(picked) < n:
        progressed = False
        modes = ["center_zoom", "descent"] if rnd % 2 == 0 else ["descent", "center_zoom"]
        for g in seeds:
            if taken[g] >= cap:
                continue
            for m in modes:
                if by_seed[g][m]:
                    picked.append(by_seed[g][m].pop())
                    taken[g] += 1
                    progressed = True
                    break
            if len(picked) >= n:
                break
        rnd += 1
        if not progressed:
            break   # pool exhausted under cap
    return picked


def _image_id(r):
    tag = "cz" if r["mode"] == "center_zoom" else "ds"
    return f"{tag}_g{r['seed_group']:04d}_r{r['rung_uid']:06d}"


def assemble(args):
    import datetime
    rungs = [json.loads(l) for l in open(WORK / "rungs.jsonl")]
    scores = {s["rung_uid"]: s for s in (json.loads(l) for l in open(WORK / "scores.jsonl"))}
    rng = random.Random(SEED + 7)

    # keep only rungs that have a rendered+scored preview
    rungs = [r for r in rungs if r["rung_uid"] in scores and
             (PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg").exists()]
    for r in rungs:
        r["v4"] = scores[r["rung_uid"]]

    n_raw = len(rungs)
    n_seeds = len({r["seed_group"] for r in rungs})
    n_cand = len({(r["seed_group"], r["candidate_id"], r["mode"]) for r in rungs})

    deduped = dedup(rungs)
    sample = stratified_sample(deduped, SAMPLE_N, PER_SEED_CAP, rng)

    # --- write batch ---
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sample:
        iid = _image_id(r)
        shutil.copyfile(PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg", CROPS_DIR / f"{iid}.jpg")
        rows.append({
            "image_id": iid,
            "render": {
                "cx": r["cx"], "cy": r["cy"], "fw": r["fw"],
                "maxiter": r["maxiter"], "palette": PALETTE,
                "composition": "center",
                "width": PREVIEW_W, "height": PREVIEW_H, "ss": PREVIEW_SS,
                "filter": "lanczos3", "interior_mode": "black",
                # --- Julia schema extension (fractal_type + c travel on every record) ---
                "fractal_type": "julia", "c_re": r["c_re"], "c_im": r["c_im"],
            },
            "provenance": {
                "generator_version": "julia_ladder_j0",
                "batch_id": BATCH_ID,
                "mode": r["mode"],
                "rung_index": r["rung_index"],
                "seed_group": r["seed_group"],
                "seed_label": r["seed_label"],
                "seed_source": r["seed_source"],
                "src_cx": r["src_cx"], "src_cy": r["src_cy"], "src_fw": r["src_fw"],
                "jitter": r["jitter"], "jitter_mag": r["jitter_mag"],
                "branch": r["branch"], "placement": r["placement"],
                "focus_score": r["focus_score"], "walk": r["walk"],
                "v4_p_not_bad": r["v4"]["v4_p_not_bad"],
                "v4_p_good": r["v4"]["v4_p_good"],
                "v4_score": r["v4"]["v4_score"],
                "root_src": "julia",
            },
            "label": {"score": None, "labeler": None, "labeled_at": None},
        })
    with open(BATCH_DIR / "images.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_cz = sum(1 for r in sample if r["mode"] == "center_zoom")
    n_ds = sum(1 for r in sample if r["mode"] == "descent")
    batch_json = {
        "created": "2026-06-25",
        "labeler": None,
        "generator_version": "julia_ladder_j0",
        "source_run": "data/v4/manifest.jsonl (label-2/3 Mandelbrot centers as Julia c)",
        "fractal_type": "julia",
        "schema_extension": "render block adds fractal_type/c_re/c_im for Julia rows",
        "sampling_metaparameters": {
            "root_fw": ROOT_FW, "center_factors": CENTER_FACTORS,
            "jitter_sigma_frac": JITTER_SIGMA_FRAC, "jitter_per_seed": JITTER_PER_SEED,
            "descent_walks": DESCENT_WALKS, "descent_depth_min": DESCENT_DMIN,
            "descent_depth_max": DESCENT_DMAX, "palette": PALETTE,
            "sample_n": SAMPLE_N, "per_seed_cap": PER_SEED_CAP, "seed": SEED,
            "dedup": {"shift_frac": SHIFT_FRAC, "scale_band": [SCALE_LO, SCALE_HI],
                      "c_tol_frac": C_TOL_FRAC},
        },
        "present_gates": None,   # NO quality gate — junk rungs are wanted as negatives
        "render_defaults": {
            "width": PREVIEW_W, "height": PREVIEW_H, "ss": PREVIEW_SS,
            "filter": "lanczos3", "interior_mode": "black", "palette": PALETTE,
        },
    }
    with open(BATCH_DIR / "batch.json", "w") as f:
        json.dump(batch_json, f, indent=2)
    with open(BATCH_DIR / "scores.json", "w") as f:
        json.dump({}, f)   # empty harness export (no labels yet)

    # --- report ---
    sc = sorted(r["v4"]["v4_score"] for r in sample)
    print("\n===== J0 PIPELINE REPORT =====")
    print(f"raw seeds (label 2/3):        893")
    print(f"deduped seed neighborhoods:   618")
    print(f"used seed neighborhoods:      {n_seeds}")
    print(f"candidates (seed x cand x mode): {n_cand}")
    print(f"raw rungs (rendered+scored):  {n_raw}")
    print(f"deduped rung pool:            {len(deduped)}")
    print(f"sampled:                      {len(sample)}")
    print(f"  mode split: center_zoom={n_cz}  descent={n_ds}")
    print(f"  per-seed cap: {PER_SEED_CAP}")
    # v4 histogram (10 bins over [0,2])
    import numpy as np
    h, edges = np.histogram([r["v4"]["v4_score"] for r in sample], bins=10, range=(0, 2))
    print("  v4_score histogram (sampled 1000):")
    for c, lo, hi in zip(h, edges[:-1], edges[1:]):
        print(f"    [{lo:.1f},{hi:.1f}): {'#'*int(c/ max(1,max(h))*40)} {c}")
    print(f"  v4_score min/med/max: {sc[0]:.3f} / {sc[len(sc)//2]:.3f} / {sc[-1]:.3f}")
    print(f"\nwrote {BATCH_DIR/'images.jsonl'} ({len(rows)} rows)")
    print(f"wrote crops to {CROPS_DIR}")

    # --- montage ---
    build_montage(sample, rng)


def build_montage(sample, rng):
    from PIL import Image, ImageDraw
    pool = list(sample)
    by_score = sorted(pool, key=lambda r: r["v4"]["v4_score"])
    lo = by_score[:8]
    hi = by_score[-8:]
    rest = [r for r in pool if r not in lo and r not in hi]
    rng.shuffle(rest)
    rand = rest[:32]
    tiles = [("LOW", r) for r in lo] + [("RAND", r) for r in rand] + [("HIGH", r) for r in hi]
    cols, tw, th = 8, 320, 180
    rows_n = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * tw, rows_n * (th + 22)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (tag, r) in enumerate(tiles):
        cx, cy = i % cols, i // cols
        p = PREVIEW_DIR / f"r{r['rung_uid']:06d}.jpg"
        im = Image.open(p).convert("RGB").resize((tw, th))
        sheet.paste(im, (cx * tw, cy * (th + 22)))
        m = "CZ" if r["mode"] == "center_zoom" else "DS"
        lab = f"{tag} {m} v4={r['v4']['v4_score']:.2f} d{r['rung_index']}"
        draw.text((cx * tw + 3, cy * (th + 22) + th + 4), lab, fill=(230, 230, 230))
    out = WORK / "montage.png"
    sheet.save(out)
    print(f"wrote montage {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("enumerate"); pe.add_argument("--workers", type=int, default=6)
    pr = sub.add_parser("render"); pr.add_argument("--workers", type=int, default=8)
    ps = sub.add_parser("score")
    pa = sub.add_parser("assemble")
    args = ap.parse_args()
    {"enumerate": enumerate_rungs, "render": render, "score": score,
     "assemble": assemble}[args.cmd](args)
