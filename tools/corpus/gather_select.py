#!/usr/bin/env python
"""v6 gather-pool selector -> standard label batch.

Turns the guard-OFF gather pool (`data/discovery/gather/<class>/`) into a
`data/label_corpus/batches/<batch_id>/` batch. **Selection is the only new stage** —
the store schema, labeling UI, `merge_scores`, taxonomy, and the assemble->train tail
are all reused as-is.

Three selection roles, disjoint by guard verdict (see prompts/build-v6-selection.md):

  best         (~75%) : guard-PASS & decoded_class>=2, ranked by k3 desc (q3->q2 fallback).
  random_eval  (~15%) : uniform-random mid-walk trajectory frames (any verdict) — the
                        unbiased coverage slice, carries mid-walk degenerates.
  disagreement (~10%) : guard-FAIL & high k3 (v5 says good, guard flags degenerate) —
                        the highest-value OOD label data. Under-fills where unavailable.

Dedup (bounds renders, never re-presents a labeled location):
  1. spatial pre-dedup on coords (greedy sec-5 neighborhood, rank-preserving) BEFORE
     rendering — collapses near-copies (e.g. the phoenix q3 pile-up) cheaply.
  2. pHash dedup on rendered survivors (DedupIndex, Hamming<=6) seeded with the prior
     corpus's LABELED-location hashes — so no already-scored location reappears.

Render: each kept pick -> `render-one` at the canonical crop spec (1280x720, ss4,
Lanczos-3, q90 JPG, center, interior black, maxiter 8000), the multi-family renderer
(the Mandelbrot-only `enrich --mode render` cannot render julia/multibrot/phoenix).
Palette per pick = seeded-random from the 76 curated q3 palettes (score3_colormaps.json;
bright by construction — palette is legibility-only, location quality is palette-invariant).

Phases (run `all`, or a phase for iteration):
  select : ledgers+walks -> per-(class,role) spatially-deduped candidate lists -> picks.jsonl
  render : render every candidate to _work/crops_stage/  (<=4 workers)
  emit   : pHash dedup (prior-labeled seed) -> images.jsonl + crops/ + batch.json + scores.json

  uv run python tools/corpus/gather_select.py all
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # tools/corpus/
from corpus_common import (make_row, provenance_block, label_block,       # noqa: E402
                           render_block, hp_str, write_jsonl, read_jsonl,
                           render_corpus_crop, render_recipe_stamp)
from verify_render_path import check_batch                                # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mining"))
from dedup import phash, DedupIndex                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "target" / "release" / "fractal-generator.exe"
GATHER_DIR = ROOT / "data" / "discovery" / "gather"
# Location quality is palette-invariant, so the label crop's palette is legibility-only.
# We draw from the 76 curated q3 palettes (bright by construction — no dark, unlabelable
# crop can result), NOT the 777 pool (which admitted too-dark crops in the first v6 batch).
SCORE3_COLORMAPS = ROOT / "data" / "palettes" / "score3_colormaps.json"
BATCHES_DIR = ROOT / "data" / "label_corpus" / "batches"

BATCH_ID = "2026-07-05_gather_v6"
GENERATOR_VERSION = "gather_v6"
BATCH_DIR = BATCHES_DIR / BATCH_ID
WORK = BATCH_DIR / "_work"
STAGE_CROPS = WORK / "crops_stage"       # every rendered candidate (pre pHash-dedup)
CROPS_DIR = BATCH_DIR / "crops"          # kept crops (post-dedup)
PICKS = WORK / "picks.jsonl"

# --- canonical crop spec (match the enrich label-crop exactly) ---
W, H, SS, MAXITER, JPGQ = 1280, 720, 4, 8000, 90
FILTER, COMPOSITION, INTERIOR = "lanczos3", "center", "black"

SEED = 20260705
WORKERS = 3          # capped at 3 (was 4): 4+ concurrent renders starved the desktop

# Windows: launch renders BELOW_NORMAL so the render fleet yields the CPU to the UI.
# subprocess.BELOW_NORMAL_PRIORITY_CLASS exists only on win32; 0 = no-op elsewhere.
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

CLASSES = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
           "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4",
           "julia:multibrot5", "phoenix"]

# per-class targets (prompt table). realized differs where dedup/availability bind.
TARGET = {
    "mandelbrot":        {"best": 110, "random_eval": 14, "disagreement": 42},
    "multibrot3":        {"best":  55, "random_eval": 13, "disagreement":  6},
    "multibrot4":        {"best":  50, "random_eval": 13, "disagreement":  5},
    "multibrot5":        {"best":  55, "random_eval": 13, "disagreement":  6},
    "julia:mandelbrot":  {"best": 105, "random_eval": 14, "disagreement":  1},
    "julia:multibrot3":  {"best":  75, "random_eval": 13, "disagreement":  1},
    "julia:multibrot4":  {"best":  60, "random_eval": 13, "disagreement":  0},
    "julia:multibrot5":  {"best":  40, "random_eval": 13, "disagreement":  0},
    "phoenix":           {"best":  50, "random_eval": 14, "disagreement": 19},
}

# sec-5 neighborhood predicate constants (faithful to tools/julia_ladder/build_j0.py)
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5


# --------------------------------------------------------------------------- family map
def render_family(fam: str) -> str:
    """Ledger class string -> canonical location.py family for render_one_flags."""
    if fam == "julia:mandelbrot":
        return "julia"
    if fam.startswith("julia:multibrot"):
        return "julia_" + fam.split(":", 1)[1]      # julia:multibrot3 -> julia_multibrot3
    return fam                                        # mandelbrot, multibrot{3,4,5}, phoenix


def is_julia(fam: str) -> bool:
    return fam.startswith("julia:")


# --------------------------------------------------------------------------- load pool
def load_ledger() -> dict:
    """All gather outcome rows, keyed by outcome id (1:1 with a walk)."""
    led = {}
    for f in sorted(GATHER_DIR.glob("*/outcome_ledger.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                led[r["id"]] = r
    return led


def load_walks() -> dict:
    """outcome_id -> walk row (with `frames`). 1:1 with the ledger."""
    walks = {}
    for f in sorted(GATHER_DIR.glob("*/runs/*/walks.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                w = json.loads(line)
                walks[w["outcome_id"]] = w
    return walks


# ------------------------------------------------------------ render-geometry per outcome
def outcome_geometry(r: dict):
    """(cx, cy, fw, c_re, c_im) render geometry for an outcome row. Julia renders the
    z-plane wallpaper (julia_z_*) at the fixed parameter c = the parent c-plane spot
    (outcome_*); native families render the outcome viewport directly (no c)."""
    if is_julia(r["family"]):
        return (r["julia_z_cx"], r["julia_z_cy"], r["julia_z_fw"],
                r["outcome_cx"], r["outcome_cy"])
    return (r["outcome_cx"], r["outcome_cy"], r["outcome_fw"], None, None)


# ------------------------------------------------------------------- spatial pre-dedup
def _neighbors(a, b) -> bool:
    """sec-5 neighborhood: overlapping viewports at a comparable scale (a,b are
    (cx,cy,fw) tuples of floats)."""
    fwa, fwb = a[2], b[2]
    if fwb == 0:
        return False
    ratio = fwa / fwb
    if ratio < SCALE_LO or ratio > SCALE_HI:
        return False
    tol = SHIFT_FRAC * min(fwa, fwb)
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy <= tol * tol


def spatial_dedup(cands: list) -> list:
    """Greedy, RANK-PRESERVING sec-5 dedup: iterate cands in the given order, keep a
    candidate iff it does not sec-5-neighbor an already-kept one in the same c-bucket.
    Keeping the first (= highest-ranked) representative of each spatial cluster.

    Each cand is a dict with float cx/cy/fw and a `cbucket` (parent_oid for Julia,
    so two z-plane spots only collide at the SAME fixed c; a constant for natives)."""
    kept_by_bucket = defaultdict(list)
    out = []
    for c in cands:
        v = (c["cx"], c["cy"], c["fw"])
        bucket = kept_by_bucket[c["cbucket"]]
        if not any(_neighbors(v, k) for k in bucket):
            bucket.append(v)
            out.append(c)
    return out


def cbucket(r: dict) -> str:
    return r.get("parent_oid") or "_" if is_julia(r["family"]) else "_"


# --------------------------------------------------------------------------- candidate build
def make_cand(role, r, cx, cy, fw, c_re, c_im, image_id, extra=None):
    d = {
        "image_id": image_id, "role": role, "family": r["family"],
        "cx": float(cx), "cy": float(cy), "fw": float(fw),
        "c_re": None if c_re is None else float(c_re),
        "c_im": None if c_im is None else float(c_im),
        "cbucket": cbucket(r),
        "k3": r.get("k3"), "decoded_class": r.get("decoded_class"),
        "guard_verdict": r.get("guard_verdict"), "descend_mode": r.get("descend_mode"),
        "parent_oid": r.get("parent_oid"), "oid": r["id"],
    }
    if extra:
        d.update(extra)
    return d


def build_candidates(led, walks, rng):
    """Per (class, role): ranked, spatially-deduped candidate lists, capped to a render
    budget just above target (margin for pHash drops). Returns flat list of cands."""
    by_class = defaultdict(list)
    for r in led.values():
        by_class[r["family"]].append(r)

    def render_cap(target):
        if target == 0:
            return 0
        return target + max(12, math.ceil(0.30 * target))

    all_cands = []
    stats = defaultdict(dict)
    for cls in CLASSES:
        rows = by_class.get(cls, [])

        # --- best: guard-pass & decoded>=2, ranked k3 desc ---
        best_pool = [r for r in rows
                     if r.get("guard_verdict") == "pass"
                     and r.get("decoded_class") is not None and r["decoded_class"] >= 2]
        best_pool.sort(key=lambda r: (-r["k3"], r["id"]))
        best_ded = spatial_dedup([_cand_from_outcome("best", r) for r in best_pool])
        cap = render_cap(TARGET[cls]["best"])
        best_sel = best_ded[:cap]
        stats[cls]["best"] = (len(best_pool), len(best_ded), len(best_sel))

        # --- disagreement: guard-fail, ranked k3 desc (top = "high k3") ---
        dis_pool = [r for r in rows if r.get("guard_verdict") != "pass"]
        dis_pool.sort(key=lambda r: (-r["k3"], r["id"]))
        dis_ded = spatial_dedup([_cand_from_outcome("disagreement", r) for r in dis_pool])
        cap = render_cap(TARGET[cls]["disagreement"])
        dis_sel = dis_ded[:cap]
        stats[cls]["disagreement"] = (len(dis_pool), len(dis_ded), len(dis_sel))

        # --- random_eval: uniform mid-walk (depth>=2) frames, any verdict ---
        frames = []
        for r in rows:
            w = walks.get(r["id"])
            if not w:
                continue
            for fr in w["frames"]:
                if int(fr["depth"]) < 2:            # drop the deterministic root frame
                    continue
                frames.append((r, fr))
        rng.shuffle(frames)
        re_cands = []
        for i, (r, fr) in enumerate(frames):
            if is_julia(r["family"]):
                cre, cim = r["outcome_cx"], r["outcome_cy"]      # fixed c = parent spot
            else:
                cre, cim = None, None
            iid = f"r_{r['id']}_f{int(fr['idx'])}"
            re_cands.append(make_cand("random_eval", r, fr["cx"], fr["cy"], fr["fw"],
                                      cre, cim, iid, extra={"cbucket": cbucket(r)}))
        re_ded = spatial_dedup(re_cands)
        cap = render_cap(TARGET[cls]["random_eval"])
        re_sel = re_ded[:cap]
        stats[cls]["random_eval"] = (len(frames), len(re_ded), len(re_sel))

        all_cands.extend(best_sel + dis_sel + re_sel)

    # --- palette assignment (seeded, stable order) ---
    names = [c["name"] for c in json.load(open(SCORE3_COLORMAPS, encoding="utf-8"))]
    for c in sorted(all_cands, key=lambda c: c["image_id"]):
        c["palette"] = rng.choice(names)
    return all_cands, stats


def _cand_from_outcome(role, r):
    cx, cy, fw, cre, cim = outcome_geometry(r)
    prefix = "b" if role == "best" else "d"
    iid = f"{prefix}_{r['id']}"
    return make_cand(role, r, cx, cy, fw, cre, cim, iid)


# --------------------------------------------------------------------------- render
def cand_render_block(c) -> dict:
    """The version-invariant render block for a candidate — the SINGLE source of the
    crop's pixels. Built once here and reused verbatim at emit, so the stored block
    is exactly what was rendered (the crop stays a pure function of its render block,
    rebuildable via `render-one --palette`; see corpus_common.render_corpus_crop)."""
    render = render_block(cx=hp_str(c["cx"]), cy=hp_str(c["cy"]), fw=hp_str(c["fw"]),
                          maxiter=MAXITER, palette=c["palette"], composition=COMPOSITION,
                          width=W, height=H, ss=SS, filter=FILTER, interior_mode=INTERIOR)
    # multi-family render-block extension (fractal_type + fixed c travel on the record,
    # as tools/julia_ladder/build_j0.py established for Julia rows).
    render["fractal_type"] = render_family(c["family"])
    if c["c_re"] is not None:
        render["c_re"] = hp_str(c["c_re"])
        render["c_im"] = hp_str(c["c_im"])
    return render


def _render(c):
    out = STAGE_CROPS / f"{c['image_id']}.jpg"
    if out.exists():
        return (c["image_id"], True)
    try:
        render_corpus_crop(cand_render_block(c), out, palette_source=SCORE3_COLORMAPS,
                           bin_path=BIN, jpg_quality=JPGQ, cwd=str(ROOT),
                           creationflags=BELOW_NORMAL)
    except RuntimeError as e:
        sys.stderr.write(f"[render {c['image_id']}] FAILED: {e}\n")
        return (c["image_id"], False)
    return (c["image_id"], True)


def render_all(cands):
    STAGE_CROPS.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cands if not (STAGE_CROPS / f"{c['image_id']}.jpg").exists()]
    print(f"rendering {len(todo)}/{len(cands)} candidates "
          f"({W}x{H} ss{SS} {FILTER} q{JPGQ} maxiter{MAXITER}) with {WORKERS} workers ...")
    done = ok = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _iid, good in ex.map(_render, todo):
            done += 1
            ok += int(good)
            if done % 50 == 0 or done == len(todo):
                print(f"  rendered {done}/{len(todo)}  ({ok} ok)")
    print("render done")


# --------------------------------------------------------------------------- pHash seed
def seed_prior_hashes(index: DedupIndex):
    """Seed with pHashes of every already-LABELED prior-corpus crop (score != null), so
    no human-scored location is re-presented. pHash is on grayscale structure, so the
    prior crop's palette differing from ours does not defeat the match."""
    from PIL import Image
    labeled = []
    for imgs in sorted(BATCHES_DIR.glob("*/images.jsonl")):
        bdir = imgs.parent
        if bdir.name == BATCH_ID:
            continue
        for r in read_jsonl(imgs):
            if (r.get("label") or {}).get("score") is not None:
                p = bdir / "crops" / f"{r['image_id']}.jpg"
                if p.exists():
                    labeled.append(p)
    print(f"seeding pHash index with {len(labeled)} prior LABELED crops ...")

    def _h(p):
        try:
            return phash(Image.open(p).convert("RGB"))
        except Exception as e:               # noqa: BLE001
            sys.stderr.write(f"[seed hash {p.name}] {e}\n")
            return None
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        hashes = [h for h in ex.map(_h, labeled) if h is not None]
    index.seed(hashes)
    print(f"  seeded {len(hashes)} hashes")


# --------------------------------------------------------------------------- emit
def emit(cands):
    from PIL import Image
    index = DedupIndex(thresh=6)
    seed_prior_hashes(index)

    # order: best, then disagreement, then random_eval; within a cell by rank
    # (best/disagreement already k3-desc; random_eval by its sampled shuffle order).
    role_order = {"best": 0, "disagreement": 1, "random_eval": 2}
    # preserve build order within (class, role) = the ranked/sampled order
    for i, c in enumerate(cands):
        c["_ord"] = i
    ordered = sorted(cands, key=lambda c: (role_order[c["role"]], c["_ord"]))

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    kept, dropped_dup, dropped_target, missing = [], 0, 0, 0
    filled = defaultdict(int)   # (class, role) -> count kept
    for c in ordered:
        stage = STAGE_CROPS / f"{c['image_id']}.jpg"
        if not stage.exists():
            missing += 1
            continue
        key = (c["family"], c["role"])
        if filled[key] >= TARGET[c["family"]][c["role"]]:
            dropped_target += 1
            continue
        try:
            h = phash(Image.open(stage).convert("RGB"))
        except Exception as e:                       # noqa: BLE001
            sys.stderr.write(f"[hash {c['image_id']}] {e}\n")
            missing += 1
            continue
        if not index.add(h):                          # collides with prior/earlier -> drop
            dropped_dup += 1
            continue
        filled[key] += 1
        kept.append(c)

    # --- write kept crops + images.jsonl ---
    import shutil
    rows = []
    for c in kept:
        shutil.copyfile(STAGE_CROPS / f"{c['image_id']}.jpg", CROPS_DIR / f"{c['image_id']}.jpg")
        render = cand_render_block(c)            # SAME block that produced the crop
        prov = provenance_block(
            GENERATOR_VERSION, BATCH_ID,
            family=c["family"], selection_role=c["role"],
            filter_score=c["k3"], k3=c["k3"], decoded_class=c["decoded_class"],
            guard_verdict=c["guard_verdict"], descend_mode=c["descend_mode"],
            parent_oid=c["parent_oid"], lineage="gather",
        )
        rows.append(make_row(c["image_id"], render, prov, label_block()))
    write_jsonl(rows, str(BATCH_DIR / "images.jsonl"))

    write_batch_json()
    json.dump({}, open(BATCH_DIR / "scores.json", "w"))    # empty harness export (unlabeled)

    report(kept, rows, dropped_dup, dropped_target, missing, len(index.hashes))

    # Guard B — auto-verify this freshly-emitted batch is rebuildable from its render
    # blocks via the canonical render-one --palette path (fails loudly if off-recipe).
    print("\n===== Guard B: render-path reproducibility (K-sample) =====")
    check_batch(BATCH_DIR, k=6)
    return kept


def write_batch_json():
    bj = {
        "created": "2026-07-05",
        "labeler": None,
        "generator_version": GENERATOR_VERSION,
        "source_run": "data/discovery/gather/<class>/ (guard-OFF v6 gather harvest)",
        "schema_extension": "render block adds fractal_type (+c_re/c_im for Julia) for "
                            "the multi-family crops; provenance adds a v6 gather block",
        "sampling_metaparameters": {
            "roles": "best (guard-pass & decoded>=2, k3-desc) / random_eval (uniform "
                     "mid-walk depth>=2 frames) / disagreement (guard-fail, high k3)",
            "targets": TARGET,
            "palette_pool": "data/palettes/score3_colormaps.json (76 curated q3, "
                            "bright by construction; seeded-random per pick)",
            "seed": SEED,
            "spatial_dedup": {"shift_frac": SHIFT_FRAC, "scale_band": [SCALE_LO, SCALE_HI],
                              "julia_cbucket": "parent_oid (fixed-c partition)"},
            "phash_dedup": {"thresh": 6, "seed": "prior-corpus LABELED crops"},
        },
        "present_gates": None,   # guard-OFF gather: degenerates are WANTED (OOD negatives)
        "render_defaults": {
            "width": W, "height": H, "ss": SS, "filter": FILTER,
            "interior_mode": INTERIOR, "maxiter": MAXITER, "composition": COMPOSITION,
            "jpg_quality": JPGQ,
        },
        # self-identifying render provenance — the canonical path these crops were
        # produced through. Guard B asserts this is CANONICAL_CROP_RECIPE.
        "render_recipe": render_recipe_stamp(SCORE3_COLORMAPS, jpg_quality=JPGQ),
    }
    json.dump(bj, open(BATCH_DIR / "batch.json", "w"), indent=2)


# --------------------------------------------------------------------------- report
def report(kept, rows, dropped_dup, dropped_target, missing, index_size):
    got = defaultdict(int)
    for c in kept:
        got[(c["family"], c["role"])] += 1
    print("\n===== v6 gather batch — realized per-class x per-role =====")
    hdr = f"{'class':20s} " + "".join(f"{r:>16s}" for r in ("best", "random_eval", "disagreement"))
    print(hdr)
    tot = defaultdict(int)
    ttgt = defaultdict(int)
    for cls in CLASSES:
        cells = []
        for role in ("best", "random_eval", "disagreement"):
            g, t = got[(cls, role)], TARGET[cls][role]
            tot[role] += g
            ttgt[role] += t
            flag = "" if g >= t else f" (-{t-g})"
            cells.append(f"{g:>4d}/{t:<4d}{flag:>7s}")
        print(f"{cls:20s} " + "".join(f"{c:>16s}" for c in cells))
    print(f"{'TOTAL':20s} " + "".join(
        f"{str(tot[r])+'/'+str(ttgt[r]):>16s}" for r in ("best", "random_eval", "disagreement")))
    print(f"\nkept {len(kept)}  |  pHash-dropped {dropped_dup}  |  "
          f"over-target-dropped {dropped_target}  |  missing-render {missing}")
    print(f"pHash index final size: {index_size}")
    n_pal = len({c['palette'] for c in kept})
    print(f"distinct palettes in batch: {n_pal}")
    print(f"wrote {BATCH_DIR/'images.jsonl'} ({len(rows)} rows)")


def report_select(stats):
    print("\n===== select: pool -> spatial-dedup -> render-cap (per class x role) =====")
    for cls in CLASSES:
        parts = []
        for role in ("best", "random_eval", "disagreement"):
            pool, ded, sel = stats[cls][role]
            parts.append(f"{role[:4]}: {pool}->{ded}->{sel}")
        print(f"  {cls:20s} " + "  ".join(parts))


# --------------------------------------------------------------------------- driver
def do_select():
    rng = random.Random(SEED)
    led, walks = load_ledger(), load_walks()
    print(f"loaded {len(led)} outcomes, {len(walks)} walks")
    cands, stats = build_candidates(led, walks, rng)
    WORK.mkdir(parents=True, exist_ok=True)
    write_jsonl(cands, str(PICKS))
    report_select(stats)
    print(f"\nwrote {PICKS} ({len(cands)} render candidates)")
    return cands


def load_picks():
    return read_jsonl(PICKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["select", "render", "emit", "all"], default="all",
                    nargs="?")
    a = ap.parse_args()
    if a.phase in ("select", "all"):
        cands = do_select()
    else:
        cands = load_picks()
    if a.phase in ("render", "all"):
        render_all(cands)
    if a.phase in ("emit", "all"):
        emit(cands)


if __name__ == "__main__":
    main()
