"""Selector-side family-diversity policy — tradeoff curve (analysis-only).

Adds NOTHING to generation; sweeps the emission_selector palette-family cap over
the SAME dramatic funnel the family-entropy trace used, and measures the
family-spread <-> pref-cost tradeoff so the dial can be set by eye.

Fitness  = pref-v3-gvo score (from provenance) -> "off" == pref-greedy selection.
Gate     = wallpaper head_v2 p_ge3 quality floor (unchanged from the trace).
Dial     = emission_selector.select(palette_family_of=..., palette_family_cap=M),
           M in {off, 5, 4, 3, 2, 1}.
Family   = hybrid key (dramatic roster mood-family else hue/chroma bucket),
           identical to tools/wallpaper/family_entropy_trace.py so before/after is
           directly comparable to its stage-4.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

REPO = "C:/Code/fractal-generator"
sys.path.insert(0, os.path.join(REPO, "tools", "wallpaper"))
import emission_selector as es  # noqa: E402

BATCH = f"{REPO}/data/wallpaper_corpus/batches/2026-07-09_wallpaper_headbatch_dramatic_v1"
FEATS = json.load(open(f"{REPO}/data/palettes/palette_features.json"))
CELLS = json.load(open(f"{REPO}/scratchpad/_stage4_cells.json"))  # image_id -> color_cell

# ---------------------------------------------------------- family key (hybrid)
ROSTER = {}
for f in glob.glob(f"{REPO}/dramatic_palettes/results/*.json"):
    fam = os.path.basename(f).split("_c")[0]
    if fam == "span":
        continue
    for p in json.load(open(f)):
        ROSTER[p["name"]] = fam

HUE_SECTORS = [("red-rust", 0, 45), ("amber-gold", 45, 100), ("green", 100, 160),
               ("teal-cyan", 160, 220), ("blue", 220, 275),
               ("violet-purple", 275, 320), ("magenta-pink", 320, 360)]

def hue_bucket(name):
    traj = np.asarray(FEATS[name]["trajectory"])
    a, b = traj[:, 1], traj[:, 2]
    chroma = np.hypot(a, b)
    if chroma.mean() < 0.03:
        return "neutral"
    ang = np.arctan2(b, a)
    x = (chroma * np.cos(ang)).sum(); y = (chroma * np.sin(ang)).sum()
    hue = math.degrees(math.atan2(y, x)) % 360.0
    for nm, lo, hi in HUE_SECTORS:
        if lo <= hue < hi:
            return nm
    return "red-rust"

def pal_family(name):
    return ROSTER.get(name) or hue_bucket(name)

# ---------------------------------------------------------- candidate pool -----
rows = [json.loads(l) for l in open(f"{BATCH}/images.jsonl") if l.strip()]
topk = [r for r in rows if r["provenance"].get("curation_bucket") == "topk"]

def locid(p):
    return p.get("source_loc") or f"{p['cx']},{p['cy']},{p['fw']}"

def build_cands(rows):
    out = []
    for r in rows:
        p = r["provenance"]; iid = r["image_id"]
        out.append(es.Candidate(
            location_id=locid(p),
            palette_id=p["palette"],
            family=p.get("family") or r["render"].get("fractal_type") or "mandelbrot",
            fitness=float(p["pref_score"]),        # pref-v3-gvo score is the ranking driver
            color_cell=CELLS[iid],
            image_id=iid,
            meta={"pref_rank": int(p["pref_rank"]),
                  "pref_score": float(p["pref_score"]),
                  "p_ge3": float(r["head_v2"]["p_ge3"]),
                  "pal_family": pal_family(p["palette"])},
        ))
    return out

CANDS = build_cands(topk)
PALFAM = {c.image_id: c.meta["pal_family"] for c in CANDS}

# ---------------------------------------------------------- metrics ------------
def spread(picks):
    fams = [p.meta["pal_family"] for p in picks]
    c = Counter(fams); tot = sum(c.values())
    if tot == 0:
        return dict(n=0, nfam=0, H=0.0, Hn=0.0, top1=None, top1_share=0.0, top3_share=0.0, hist={})
    ps = np.array([v / tot for v in c.values()])
    H = float(-(ps * np.log2(ps)).sum())
    Hmax = math.log2(len(c)) if len(c) > 1 else 0.0
    top = c.most_common(3)
    return dict(n=tot, nfam=len(c), H=round(H, 3),
                Hn=round(H / Hmax, 3) if Hmax else (1.0 if len(c) == 1 else 0.0),
                top1=top[0][0], top1_share=round(top[0][1] / tot, 3),
                top3_share=round(sum(v for _, v in top) / tot, 3),
                hist=dict(c.most_common()))

def pref_cost(picks, off_mean_score):
    if not picks:
        return dict(mean_rank=None, mean_score=None, dscore=None)
    ranks = [p.meta["pref_rank"] for p in picks]
    scores = [p.meta["pref_score"] for p in picks]
    ms = float(np.mean(scores))
    return dict(mean_rank=round(float(np.mean(ranks)), 3),
                mean_score=round(ms, 3),
                dscore=round(ms - off_mean_score, 3) if off_mean_score is not None else 0.0)

CAPS = [None, 5, 4, 3, 2, 1]   # off -> strong

def run_floor(floor):
    cands = CANDS
    gate = lambda c: c.meta["p_ge3"] > floor
    survivors = [c for c in cands if gate(c)]
    print(f"\n{'='*78}\nQUALITY FLOOR  head_v2 p_ge3 > {floor}"
          f"   ({len(survivors)} gate-survivors, "
          f"{len(set(c.location_id for c in survivors))} locs, "
          f"{len(set(c.palette_id for c in survivors))} palettes, "
          f"{len(set(c.behavior_cell for c in survivors))} reachable cells)\n{'='*78}")
    # baseline off-selection for delta reference
    off = es.select(cands, gate=gate, grid=es.ColorGrid(),
                    palette_family_of=lambda c: c.meta["pal_family"], palette_family_cap=None)
    off_ms = float(np.mean([p.meta["pref_score"] for p in off.picks])) if off.picks else None
    print(f"{'cap':>5} | {'picks':>5} {'#fam':>4} {'H':>6} {'Hn':>5} "
          f"{'top1':>16} {'top1%':>6} {'top3%':>6} | {'mRank':>6} {'mScore':>7} {'dScore':>7}")
    print("-" * 100)
    table = []
    for cap in CAPS:
        res = es.select(cands, gate=gate, grid=es.ColorGrid(),
                        palette_family_of=lambda c: c.meta["pal_family"],
                        palette_family_cap=cap)
        sp = spread(res.picks); pc = pref_cost(res.picks, off_ms)
        capname = "off" if cap is None else str(cap)
        print(f"{capname:>5} | {sp['n']:>5} {sp['nfam']:>4} {sp['H']:>6} {sp['Hn']:>5} "
              f"{str(sp['top1']):>16} {sp['top1_share']:>6} {sp['top3_share']:>6} | "
              f"{str(pc['mean_rank']):>6} {str(pc['mean_score']):>7} {str(pc['dscore']):>7}")
        table.append((capname, sp, pc, res))
    return table

def dump_picks(res, label):
    print(f"\n--- emitted picks @ {label}  ({len(res.picks)} wallpapers) ---")
    picks = sorted(res.picks, key=lambda c: (c.meta["pal_family"], -c.fitness))
    for c in picks:
        print(f"  {c.meta['pal_family']:<16} | {c.palette_id:<28} | "
              f"{c.family:<16} | loc {c.location_id:<10} | "
              f"pref_rank {c.meta['pref_rank']} | pref {c.meta['pref_score']:.2f} | "
              f"cell {c.color_cell:>2}")

# ---------------------------------------------------------- run ----------------
print("Selector family-diversity sweep — dramatic funnel "
      "(2026-07-09_wallpaper_headbatch_dramatic_v1)")
print(f"pool: {len(topk)} topk candidates | fitness=pref_score | "
      f"dial=palette-family cap | family=hybrid key")

t90 = run_floor(0.90)   # faithful: comparable to trace stage-4 (production emit gate)
t50 = run_floor(0.50)   # richer portfolio: dial resolution

# low / mid / high picks at the richer floor
by = {c[0]: c for c in t50}
dump_picks(by["off"][3], "0.50 floor, cap=off (LOW diversity / pref-greedy)")
dump_picks(by["2"][3],   "0.50 floor, cap=2  (MID diversity)")
dump_picks(by["1"][3],   "0.50 floor, cap=1  (HIGH diversity / one-per-family)")

# also the faithful-gate picks at off vs strongest, for the stage-4-comparable view
by90 = {c[0]: c for c in t90}
dump_picks(by90["off"][3], "0.90 floor, cap=off (stage-4-comparable baseline)")
dump_picks(by90["1"][3],   "0.90 floor, cap=1  (max spread)")
