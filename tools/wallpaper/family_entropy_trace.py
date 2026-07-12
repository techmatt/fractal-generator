"""Palette family-entropy trace through the emission pipeline (analysis-only).

Stages:
  1. Library      — full pool_colormaps.json (987)
  2. gen-0 draw   — deterministic gen0_palettes(N_GEN0=60): source-stratified FP
  3. pref top-K   — dramatic headbatch topk survivors (pref-v3-gvo, top_k_pool)
  4. selector     — emission_selector over stage-3 survivors (family x color_cell)

Family key (hybrid, per task): dramatic roster mood-family where it exists
(from dramatic_palettes/results/<family>_*.json filename), else a coarse
hue/chroma bucket from palette_features. A uniform hue-bucket key is also
computed as a robustness cross-check.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

REPO = "C:/Code/fractal-generator"
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "palettes"))
sys.path.insert(0, os.path.join(REPO, "tools", "wallpaper"))
import palette_features as pf                      # noqa: E402
import emission_selector as es                     # noqa: E402

POOL = json.load(open(f"{REPO}/data/palettes/pool_colormaps.json"))
FEATS = json.load(open(f"{REPO}/data/palettes/palette_features.json"))
NAME2SRC = {p["name"]: p.get("source", "extracted") for p in POOL}
NAMES_ALL = [p["name"] for p in POOL]

# ---------------------------------------------------------------- family key --
# (a) dramatic roster family from results/ filenames (span_* excluded -> hue bucket)
ROSTER = {}
for f in glob.glob(f"{REPO}/dramatic_palettes/results/*.json"):
    fam = os.path.basename(f).split("_c")[0]
    if fam == "span":
        continue
    for p in json.load(open(f)):
        ROSTER[p["name"]] = fam

# (b) hue/chroma bucket from OKLab trajectory (chroma-weighted circular-mean hue)
HUE_SECTORS = [  # (name, lo_deg, hi_deg)
    ("red-rust",     0,   45),
    ("amber-gold",   45,  100),
    ("green",        100, 160),
    ("teal-cyan",    160, 220),
    ("blue",         220, 275),
    ("violet-purple",275, 320),
    ("magenta-pink", 320, 360),
]

def hue_bucket(name):
    traj = np.asarray(FEATS[name]["trajectory"])   # (n,3) L,a,b OKLab
    a, b = traj[:, 1], traj[:, 2]
    chroma = np.hypot(a, b)
    if chroma.mean() < 0.03:
        return "neutral"
    # chroma-weighted circular mean hue
    ang = np.arctan2(b, a)
    x = (chroma * np.cos(ang)).sum()
    y = (chroma * np.sin(ang)).sum()
    hue = math.degrees(math.atan2(y, x)) % 360.0
    for nm, lo, hi in HUE_SECTORS:
        if lo <= hue < hi:
            return nm
    return "red-rust"

def family_hybrid(name):
    return ROSTER.get(name) or hue_bucket(name)

def family_hue(name):
    return hue_bucket(name)

FAM = {n: family_hybrid(n) for n in NAMES_ALL}
FAMH = {n: family_hue(n) for n in NAMES_ALL}

# ---------------------------------------------------------------- entropy ----
def stats(names, keymap):
    fams = [keymap[n] for n in names if n in keymap]
    c = Counter(fams)
    tot = sum(c.values())
    ps = np.array([v / tot for v in c.values()])
    H = float(-(ps * np.log2(ps)).sum())
    Hmax = math.log2(len(c)) if len(c) > 1 else 0.0
    top3 = c.most_common(3)
    top3_share = sum(v for _, v in top3) / tot
    return {
        "n_items": tot,
        "n_families": len(c),
        "entropy_bits": round(H, 3),
        "entropy_norm": round(H / Hmax, 3) if Hmax else 0.0,
        "top1": (top3[0][0], round(top3[0][1] / tot, 3)) if top3 else None,
        "top3_share": round(top3_share, 3),
        "top3": [(k, round(v / tot, 3)) for k, v in top3],
        "counter": c,
    }

def show(label, s):
    print(f"\n### {label}")
    print(f"  items={s['n_items']}  families(>=1)={s['n_families']}  "
          f"H={s['entropy_bits']} bits  H_norm={s['entropy_norm']}")
    print(f"  top1={s['top1']}  top3_share={s['top3_share']}")
    print(f"  top3: {s['top3']}")
    print("  full: " + ", ".join(f"{k}:{v}" for k, v in s["counter"].most_common()))


# ================================================================ STAGE 1 ====
lib = stats(NAMES_ALL, FAM)
libh = stats(NAMES_ALL, FAMH)

# ================================================================ STAGE 2 ====
# Replicate sample_location.gen0_palettes(sampler, 60): per-source-bucket FP.
GEN0_W = {"dramatic": 0.75, "curated_q3": 0.0833, "curated_q2": 0.05, "extracted": 0.1167}
N_GEN0 = 60

def hamilton(weights, N):
    order = list(weights)
    ws = sum(weights.values())
    raw = {b: weights[b] / ws * N for b in order}
    base = {b: int(math.floor(raw[b])) for b in order}
    left = N - sum(base.values())
    fo = sorted(order, key=lambda b: (-(raw[b] - math.floor(raw[b])), order.index(b)))
    for b in fo[:left]:
        base[b] += 1
    return base

names = [n for n in FEATS if n in NAME2SRC]
buckets = {}
for n in names:
    buckets.setdefault(NAME2SRC[n], []).append(n)
avail = {b: len(buckets.get(b, [])) for b in GEN0_W}
quotas = hamilton(GEN0_W, N_GEN0)   # no underfill at N=60
gen0 = []
for b in GEN0_W:
    nb, kb = buckets.get(b, []), quotas[b]
    if kb <= 0 or not nb:
        continue
    D = pf.distance_matrix(FEATS, nb)
    gen0.extend(pf.farthest_point_order(nb, k=kb, dmat=D))
print(f"[gen0] quotas={quotas}  drew {len(gen0)} palettes")
gen0_src = Counter(NAME2SRC[n] for n in gen0)
print(f"[gen0] source mix: {dict(gen0_src)}")
g0 = stats(gen0, FAM)
g0h = stats(gen0, FAMH)

# ================================================================ STAGE 3 ====
BATCH = f"{REPO}/data/wallpaper_corpus/batches/2026-07-09_wallpaper_headbatch_dramatic_v1"
rows = [json.loads(l) for l in open(f"{BATCH}/images.jsonl") if l.strip()]
topk = [r for r in rows if r["provenance"].get("curation_bucket") == "topk"]
topk_names = [r["provenance"]["palette"] for r in topk]
missing = [n for n in set(topk_names) if n not in FAM]
if missing:
    print(f"[warn] {len(missing)} survivor palettes not in family map: {missing[:5]}")
s3 = stats(topk_names, FAM)
s3h = stats(topk_names, FAMH)
print(f"\n[stage3] {len(topk)} topk survivor instances, "
      f"{len(set(topk_names))} distinct palettes, over "
      f"{len(set(r['provenance'].get('source_loc') for r in topk))} locations")
# how many of the 60 gen0 palettes actually survive anywhere
surv_in_gen0 = set(topk_names) & set(gen0)
print(f"[stage3] distinct survivor palettes in gen0 set: "
      f"{len(surv_in_gen0)}/{len(set(topk_names))} distinct "
      f"(gen0 has {len(gen0)})")

# ================================================================ STAGE 4 ====
# Reconstruct emission_selector over the topk survivors.
GRID = es.ColorGrid()
crops = f"{BATCH}/crops"
cell_cache = f"{REPO}/scratchpad/_stage4_cells.json"
cache = json.load(open(cell_cache)) if os.path.exists(cell_cache) else {}

def thumb_rgb(jpg, w=96):
    with Image.open(jpg) as im:
        im = im.convert("RGB")
        iw, ih = im.size
        im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
        return np.asarray(im)

cands = []
n_new = 0
for r in topk:
    iid = r["image_id"]
    if iid not in cache:
        cache[iid] = GRID.cell(es.dominant_lab(thumb_rgb(f"{crops}/{iid}.jpg"), method="median"))
        n_new += 1
    prov = r["provenance"]
    fam_fractal = prov.get("family") or r["render"].get("fractal_type") or "mandelbrot"
    loc_id = prov.get("source_loc") or f"{prov['cx']},{prov['cy']},{prov['fw']}"
    cands.append(es.Candidate(
        location_id=loc_id,
        palette_id=prov["palette"],
        family=fam_fractal,
        fitness=float(r["head_v2"]["score"]),
        color_cell=cache[iid],
        image_id=iid,
        meta={"p_ge3": float(r["head_v2"]["p_ge3"])},
    ))
if n_new:
    json.dump(cache, open(cell_cache, "w"))
print(f"[stage4] {len(cands)} candidates ({n_new} new color cells), "
      f"gate p_ge3>0.90 (head_v2 persisted)")

res = es.select(cands, gate=lambda c: c.meta["p_ge3"] > 0.90,
                grid=GRID, palette_cap_frac=0.05)
picks = res.picks
pick_names = [c.palette_id for c in picks]
print(f"[stage4] selector: {len(cands)} -> {res.report['n_survivors']} gate-pass "
      f"-> {len(picks)} picks  (cap={res.palette_cap}, "
      f"cells {res.report['cells_filled']}/{res.report['cells_reachable']}, "
      f"{res.report['n_distinct_palettes_picked']} distinct palettes)")
s4 = stats(pick_names, FAM)
s4h = stats(pick_names, FAMH)

# ================================================================ REPORT ======
print("\n" + "=" * 70)
print("HYBRID FAMILY KEY  (dramatic roster where exists, else hue bucket)")
print("=" * 70)
show("STAGE 1 — Library (987)", lib)
show("STAGE 2 — gen-0 draw (60)", g0)
show("STAGE 3 — pref top-K survivors", s3)
show("STAGE 4 — selector output", s4)

print("\n" + "=" * 70)
print("HUE-BUCKET KEY (uniform, cross-check)")
print("=" * 70)
show("STAGE 1 — Library", libh)
show("STAGE 2 — gen-0", g0h)
show("STAGE 3 — top-K", s3h)
show("STAGE 4 — selector", s4h)

print("\n" + "=" * 70)
print("SUMMARY TABLE (hybrid key)")
print("=" * 70)
print(f"{'stage':<22}{'items':>7}{'#fam':>6}{'H(bits)':>9}{'H_norm':>8}{'top1':>22}{'top3%':>7}")
for lbl, s in [("1 Library", lib), ("2 gen-0", g0), ("3 pref top-K", s3), ("4 selector", s4)]:
    t1 = f"{s['top1'][0]}={s['top1'][1]}"
    print(f"{lbl:<22}{s['n_items']:>7}{s['n_families']:>6}{s['entropy_bits']:>9}"
          f"{s['entropy_norm']:>8}{t1:>22}{s['top3_share']:>7}")
print("\n(hue-bucket key)")
print(f"{'stage':<22}{'items':>7}{'#fam':>6}{'H(bits)':>9}{'H_norm':>8}{'top1':>22}{'top3%':>7}")
for lbl, s in [("1 Library", libh), ("2 gen-0", g0h), ("3 pref top-K", s3h), ("4 selector", s4h)]:
    t1 = f"{s['top1'][0]}={s['top1'][1]}"
    print(f"{lbl:<22}{s['n_items']:>7}{s['n_families']:>6}{s['entropy_bits']:>9}"
          f"{s['entropy_norm']:>8}{t1:>22}{s['top3_share']:>7}")
