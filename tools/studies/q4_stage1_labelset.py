#!/usr/bin/env python
"""q4 stage-1 labeling harness — produce the stratified window set stage-1 needs.

Find ~30 minibrots (varied periods), render each at 4×size (the minibrot framed by
its ring), sweep 16:9 windows at 2-3 scales, drop OBVIOUS rejects, **stratify
survivors by the current composite score** into bands, sample ~300 across bands,
and capture them to a REGISTERED, single-reader store SEPARATE from the v7 location
corpus. Present in a fast accept/reject flow (tools/viz/q4_window_label.html).

This build STOPS at "windows ready to label + capture wired". Fitting the basic
functions + the per-scale heatmap is the NEXT prompt, after labels exist.

Reuse, don't rebuild:
  * finder            tools/sourcing/deep_center_finder.py   (nucleus Newton + size est)
  * sweep + metrics   tools/studies/q4_neighborhood_sweep.py (compute_metrics, score_A)
  * field dump        render-one --dump-field (auto perturbation for deep frames)
  * coloring tail     tools/colormap.py (field⊗colormap — ONE render per minibrot)

CAVEAT (recorded in docs/findings/q4_stage1_labelset.md): windows are crops of ONE
medium render, so small-window frequency stats are SCALE-BIASED. A true-scale
per-window re-render is a stage-2 refinement, not now.

Stages (idempotent, resume from checkpoint):
  minibrots  seed grammar -> Newton nuclei across periods -> dedup -> ~30 -> minibrots.json
  fields     render-one --dump-field per minibrot (DETACHED-friendly, resumable)
  sweep      window sweep x3 scales -> metrics -> score_A -> NMS -> prefilter -> windows_all.jsonl
  stratify   stratify survivors by score_A into bands -> sample ~300 -> selected.jsonl
  capture    colorize each field once, crop selected windows -> the label store + windows.jsonl

Run:  uv run python -m tools.studies.q4_stage1_labelset all
      uv run python -m tools.studies.q4_stage1_labelset fields   # (detach this one)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mpmath as mp
from tools.sourcing import deep_center_finder as dcf
from tools.studies.q4_neighborhood_sweep import compute_metrics, score_A  # verbatim transfer

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "out" / "q4_stage1"
FIELDS = OUT / "fields"

# The label store — SEPARATE from the v7 location corpus (distribution-bound).
BATCH_ID = "2026-07-23_q4_stage1_windows"
STORE = ROOT / "data" / "q4_window_corpus" / "batches" / BATCH_ID
CROPS = STORE / "crops"

# --- render geometry -------------------------------------------------------
# ~2.1k wide so the smallest sampled window (SCALES[0]=0.06 frame width) is >=128 px
# (0.06 * 2176 = 130). 16:9 (wallpaper aspect); field + crops share this geometry.
W, H = 2176, 1224
ASPECT = "16:9"
# Vivid blue->white->orange->near-black. The fair re-render proved the old
# twilight_shifted purple-on-black ramp crushed the mid-tone filigree to invisible
# noise ("30 useless" was a palette artifact — docs/findings/fair_rerender_richness.md);
# labeling under it teaches garbage. This is the Rust built-in `default` (Ultra Fractal)
# palette — the SAME vivid, iteration-CYCLED banding that made the fair montage pop.
#
# Capture renders via the Rust `render` path (NOT the field⊗colormap recolor): the UF
# default's classic banded look is the palette CYCLED with smooth-iteration, whereas
# colormap.py percentile-stretches the whole field into ONE palette pass (flat,
# washed-out gradient — a different coloring entirely). So the fields are re-rendered
# full-frame in `default` and the windows cropped from those, exactly reproducing the
# montage tiles Matt approved (prompt's sanctioned "else re-render the 30 vivid" branch).
PALETTE = "default"

# Full-frame capture render: 2x the field grid (same 16:9 frame / fw / center — window
# rects are frame-normalized, so any resolution maps) at ss2, for crisp label crops.
CAP_W, CAP_H, CAP_SS = 2 * W, 2 * H, 2
FRAMES = OUT / "frames"          # disposable full-frame captures (out/ tree)

# --- sweep -----------------------------------------------------------------
# 3 scales spanning Matt's hand-drawn box widths (0.057..0.099 frame-normalized),
# with headroom at the top. Window is a 16:9 fraction of the (16:9) frame.
SCALES = [0.06, 0.09, 0.14]
STRIDE_FRAC = 0.30           # window stride as fraction of window width
NMS_IOU = 0.35               # per-minibrot spatial de-dup before stratifying
TARGET_N = 300               # ~300 windows to label
N_BANDS = 6                  # composite-score stratification bands
SEED = 0

# feature vector stored per window (compute_metrics keys — fitting-ready)
FEATURE_KEYS = ["interior_frac", "deep_frac", "detail_in_deep", "flat_frac",
                "mid_detail_frac", "high_struct_frac", "busy_frac", "occupancy",
                "distributed_interior", "mean_struct"]

# --- dumb pre-filter: OBVIOUS rejects only (loose; keep borderline) --------
# A window is an obvious reject if ANY of these fire. Deliberately generous so the
# stratifier still sees the mid-band; the high-recall label gate does the rest.
def is_obvious_reject(m):
    if m["interior_frac"] > 0.85:                       # dead: near-all black interior
        return "interior_heavy"
    if m["flat_frac"] > 0.92 and m["occupancy"] < 0.06:  # barren/sparse: no structure
        return "barren"
    if m["busy_frac"] > 0.15:                            # speckle-noisy
        return "speckle"
    return None


# --------------------------------------------------------------------------- #
# Stage 1 — minibrots                                                          #
# --------------------------------------------------------------------------- #
# Anchor seeds near ∂M, each Newton-refined across a band of periods. Newton from a
# fixed valley seed to period p lands the nearest period-p nucleus, so a spread of
# (anchor, period) pairs yields varied minibrots (like Matt's boxed p35/p58). We
# over-provide, keep converged + minimal-period + renderable-size, dedup by center,
# then select ~30 spanning the period range.
ANCHORS = [
    (-0.7453, 0.1127, "seahorse"),
    (-0.7460, 0.1080, "seahorse2"),
    (-0.7500, 0.1075, "seahorse3"),
    (0.2925, 0.0149, "elephant"),
    (0.3220, 0.0330, "elephant_spiral"),
    (0.2600, 0.0020, "elephant2"),
    (-1.2500, 0.0200, "west_antenna"),
    (-0.1568, 1.0322, "north_bulb"),
    (-0.5600, 0.6400, "nw_valley"),
    (0.3800, 0.1400, "ne_valley"),
]
PERIODS = list(range(4, 66))            # try this whole band per anchor
# --dump-field lives ONLY on the f64 render-one (perturbation paths can't dump a
# field), so the field dump is f64-bound: at W=2176 the spacing must stay >1e-13,
# i.e. fw=4*size > ~2.2e-10 -> size > ~5.5e-11. Floor at 1e-10 for margin (fw>=4e-10,
# spacing ~1.8e-13). Minibrots are self-similar, so a moderate-depth period-p nucleus
# is a valid COMPOSITIONAL proxy for a deep one (see the second caveat in the findings
# doc); deep-specific precision behavior is exactly the stage-2 true-scale re-render.
SIZE_LO, SIZE_HI = 1e-10, 3e-2          # f64-dumpable depth band
DEDUP_DPS = 22                          # round centers to this many digits to dedup


def _minimal_period(c, period, tol):
    """True iff no proper divisor q|period also closes z_q(c)=0 (period is minimal)."""
    for q in range(1, period):
        if period % q == 0 and abs(dcf._orbit(c, q)[0]) < tol:
            return False
    return True


def stage_minibrots():
    mp.mp.dps = 60
    tol = mp.mpf(10) ** (-(mp.mp.dps - 6))
    found = {}                          # dedup key -> record
    t0 = time.time()
    for ar, ai, aname in ANCHORS:
        seed = mp.mpc(ar, ai)
        for p in PERIODS:
            r = dcf.newton_nucleus(seed, p)
            if not r.converged:
                continue
            if not _minimal_period(r.c, p, tol):
                continue                # p is a multiple of a smaller true period
            size = dcf.nucleus_size_estimate(r.c, p)
            sabs = float(abs(size)) if size != 0 else 0.0
            if not (SIZE_LO <= sabs <= SIZE_HI):
                continue
            key = (mp.nstr(r.c.real, DEDUP_DPS), mp.nstr(r.c.imag, DEDUP_DPS))
            if key in found:
                continue
            r.newton_residual_log10 = r.residual
            dc = dcf.make_deep_center(r)   # fw_suggest = 4*size, render_maxiter set
            found[key] = dict(
                anchor=aname, period=p,
                cx=dc.cx, cy=dc.cy,
                fw=dc.fw_suggest, maxiter=dc.render_maxiter,
                size=sabs, newton_res_log10=round(r.residual, 1),
            )
    recs = list(found.values())
    # Select ~30 spanning the period range: sort by period, take an even stride.
    recs.sort(key=lambda d: (d["period"], d["cx"]))
    n_target = 30
    if len(recs) > n_target:
        idx = np.linspace(0, len(recs) - 1, n_target).round().astype(int)
        recs = [recs[i] for i in sorted(set(idx.tolist()))]
    for i, d in enumerate(recs):
        d["id"] = f"mb{i:02d}_p{d['period']:02d}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "minibrots.json").write_text(json.dumps(recs, indent=2))
    periods = sorted(d["period"] for d in recs)
    print(f"minibrots: {len(recs)} kept in {time.time()-t0:.1f}s  "
          f"periods {periods[0]}..{periods[-1]}  "
          f"fw range [{min(float(d['fw']) for d in recs):.2e}, "
          f"{max(float(d['fw']) for d in recs):.2e}]")
    print("  periods:", periods)
    return recs


def load_minibrots():
    return json.loads((OUT / "minibrots.json").read_text())


# --------------------------------------------------------------------------- #
# Stage 2 — fields (resumable; detach this stage)                             #
# --------------------------------------------------------------------------- #
def dump_field(mb, out_bin):
    cmd = [str(EXE), "render-one", "--cx", mb["cx"], "--cy", mb["cy"], "--fw", mb["fw"],
           "--family", "mandelbrot", "--maxiter", str(mb["maxiter"]),
           "--width", str(W), "--height", str(H), "--supersample", "1",
           "--dump-field", str(out_bin)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field {mb['id']} failed: {r.stderr[-400:]}")


def stage_fields():
    FIELDS.mkdir(parents=True, exist_ok=True)
    mbs = load_minibrots()
    t0 = time.time()
    done = 0
    for i, mb in enumerate(mbs):
        b = FIELDS / f"{mb['id']}.bin"
        if b.exists() and b.with_suffix(".json").exists():
            done += 1
            continue
        ts = time.time()
        dump_field(mb, b)
        print(f"  [{i+1}/{len(mbs)}] {mb['id']} fw={mb['fw']} maxiter={mb['maxiter']} "
              f"-> {time.time()-ts:.1f}s", flush=True)
    print(f"fields done: {len(mbs)} total ({done} cached) in {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------- #
# Stage 3 — sweep + prefilter                                                 #
# --------------------------------------------------------------------------- #
def load_field_values(mb_id):
    meta = json.loads((FIELDS / f"{mb_id}.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    a = np.frombuffer((FIELDS / f"{mb_id}.bin").read_bytes(), dtype="<f4")
    return a.reshape(h, w).astype(np.float64), w, h


def _iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def sweep_minibrot(mb_id):
    """All windows over one field: metrics + score, greedy NMS, prefilter flag."""
    field, fw, fh = load_field_values(mb_id)
    cands = []
    for s in SCALES:
        Wp = max(8, int(round(s * fw)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= fh or Wp >= fw:
            continue
        st = max(4, int(round(STRIDE_FRAC * Wp)))
        for y in range(0, fh - Hp + 1, st):
            for x in range(0, fw - Wp + 1, st):
                m = compute_metrics(field[y:y + Hp, x:x + Wp])
                u, v, uw, vh = x / fw, y / fh, Wp / fw, Hp / fh
                cands.append(dict(scale=s, box=(u, v, uw, vh),
                                  cx=u + uw / 2, cy=v + vh / 2,
                                  score=float(score_A(m)), m=m))
    cands.sort(key=lambda c: c["score"], reverse=True)
    kept = []                                     # greedy NMS across scales
    for c in cands:
        if all(_iou(c["box"], k["box"]) <= NMS_IOU for k in kept):
            kept.append(c)
    return kept


def stage_sweep():
    mbs = load_minibrots()
    rows = []
    n_reject = {}
    for mb in mbs:
        if not (FIELDS / f"{mb['id']}.bin").exists():
            print(f"  WARN no field for {mb['id']} — run `fields` first", file=sys.stderr)
            continue
        kept = sweep_minibrot(mb["id"])
        for c in kept:
            rej = is_obvious_reject(c["m"])
            if rej:
                n_reject[rej] = n_reject.get(rej, 0) + 1
            u, v, uw, vh = c["box"]
            rows.append(dict(
                minibrot_id=mb["id"], period=mb["period"],
                cx=mb["cx"], cy=mb["cy"], fw=mb["fw"], maxiter=mb["maxiter"],
                scale=c["scale"],
                window=dict(u=round(u, 5), v=round(v, 5), w=round(uw, 5), h=round(vh, 5)),
                score_composite=round(c["score"], 5),
                prefilter_reject=rej,
                features={k: round(float(c["m"][k]), 5) for k in FEATURE_KEYS},
            ))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "windows_all.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_survive = sum(1 for r in rows if r["prefilter_reject"] is None)
    print(f"sweep: {len(rows)} windows over {len(mbs)} minibrots; "
          f"prefilter rejects {sum(n_reject.values())} {n_reject}; "
          f"survivors {n_survive}")
    return rows


def load_windows_all():
    return [json.loads(l) for l in (OUT / "windows_all.jsonl").read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Stage 4 — stratify survivors by composite score into bands -> ~300          #
# --------------------------------------------------------------------------- #
def stage_stratify():
    rows = load_windows_all()
    surv = [r for r in rows if r["prefilter_reject"] is None]
    if not surv:
        raise RuntimeError("no survivors — run sweep first")
    scores = np.array([r["score_composite"] for r in surv])
    # Equal-count (quantile) bands over the composite score. Stratify-by-score is the
    # point: Matt's good picks ranked mid-pack, so uniform/top sampling misses them.
    edges = np.quantile(scores, np.linspace(0, 1, N_BANDS + 1))
    edges[-1] += 1e-9
    band_of = np.clip(np.digitize(scores, edges[1:-1]), 0, N_BANDS - 1)
    for r, b in zip(surv, band_of):
        r["band"] = int(b)

    rng = np.random.default_rng(SEED)
    quota = TARGET_N // N_BANDS
    selected = []
    # round-robin across minibrots WITHIN each band for spatial/period diversity
    leftover_pool = []
    for b in range(N_BANDS):
        band_rows = [r for r in surv if r["band"] == b]
        by_mb = {}
        for r in band_rows:
            by_mb.setdefault(r["minibrot_id"], []).append(r)
        for mbid in by_mb:
            rng.shuffle(by_mb[mbid])
        order = list(by_mb.keys())
        rng.shuffle(order)
        picked = []
        while len(picked) < quota and any(by_mb[m] for m in order):
            for m in order:
                if by_mb[m]:
                    picked.append(by_mb[m].pop())
                    if len(picked) >= quota:
                        break
        selected.extend(picked)
        leftover_pool.extend([r for m in order for r in by_mb[m]])
    # top up to TARGET_N from leftovers (any band) to hit ~300 exactly
    rng.shuffle(leftover_pool)
    while len(selected) < TARGET_N and leftover_pool:
        selected.append(leftover_pool.pop())

    # stable window ids + deterministic order
    for r in selected:
        wk = f"{r['minibrot_id']}|{r['scale']}|{r['window']['u']}|{r['window']['v']}"
        r["window_id"] = f"{r['minibrot_id']}_s{int(r['scale']*1000):03d}_" \
                         f"{hashlib.sha1(wk.encode()).hexdigest()[:8]}"
    selected.sort(key=lambda r: (r["minibrot_id"], r["scale"], r["window"]["u"], r["window"]["v"]))
    with (OUT / "selected.jsonl").open("w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    counts = {b: sum(1 for r in selected if r["band"] == b) for b in range(N_BANDS)}
    per_mb = {}
    for r in selected:
        per_mb[r["minibrot_id"]] = per_mb.get(r["minibrot_id"], 0) + 1
    print(f"stratify: selected {len(selected)} windows "
          f"from {len(surv)} survivors across {N_BANDS} score bands")
    print(f"  band edges (score_A): {np.round(edges, 3).tolist()}")
    print(f"  per-band counts: {counts}")
    print(f"  minibrots covered: {len(per_mb)}  (per-mb min/median/max "
          f"{min(per_mb.values())}/{int(np.median(list(per_mb.values())))}/{max(per_mb.values())})")
    return selected


def load_selected():
    return [json.loads(l) for l in (OUT / "selected.jsonl").read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Stage 5 — capture: colorize each field once, crop windows -> label store    #
# --------------------------------------------------------------------------- #
def render_full_frame(mb, out_png):
    """Render one minibrot full-frame in the vivid UF `default` palette via the Rust
    engine — the exact banded coloring of the approved fair montage (bare `render`,
    --palette default). Window crops come from this, NOT from a colormap.py recolor."""
    cmd = [str(EXE), "--center-re", mb["cx"], "--center-im", mb["cy"],
           "--frame-width", mb["fw"], "--maxiter", str(mb["maxiter"]),
           "--width", str(CAP_W), "--height", str(CAP_H),
           "--supersample", str(CAP_SS), "--palette", PALETTE,
           "--output", str(out_png)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"render {mb['id']} failed: {r.stderr[-400:]}")


def stage_capture():
    from PIL import Image

    selected = load_selected()
    CROPS.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    by_mb = {}
    for r in selected:
        by_mb.setdefault(r["minibrot_id"], []).append(r)

    mbs = {m["id"]: m for m in load_minibrots()}
    t0 = time.time()
    n_crop = 0
    for k, (mbid, wins) in enumerate(by_mb.items()):
        frame_png = FRAMES / f"{mbid}.png"
        ts = time.time()
        render_full_frame(mbs[mbid], frame_png)        # ONE vivid Rust render per minibrot
        full = Image.open(frame_png).convert("RGB")
        fw_px, fh_px = full.size
        for r in wins:
            u, v, ww, hh = (r["window"][x] for x in ("u", "v", "w", "h"))
            x0, y0 = int(round(u * fw_px)), int(round(v * fh_px))
            x1, y1 = int(round((u + ww) * fw_px)), int(round((v + hh) * fh_px))
            crop = full.crop((x0, y0, x1, y1))
            crop.save(CROPS / f"{r['window_id']}.jpg", quality=90)
            n_crop += 1
        print(f"  [{k+1}/{len(by_mb)}] {mbid}: {len(wins)} crops "
              f"({time.time()-ts:.1f}s)", flush=True)

    # windows.jsonl — the label-store rows (SEPARATE schema, window-bound)
    with (STORE / "windows.jsonl").open("w") as f:
        for r in selected:
            row = dict(
                window_id=r["window_id"],
                minibrot_id=r["minibrot_id"],
                period=r["period"],
                render=dict(cx=r["cx"], cy=r["cy"], fw=r["fw"], maxiter=r["maxiter"],
                            family="mandelbrot", width=W, height=H, aspect=ASPECT,
                            palette=PALETTE),
                window=r["window"],
                scale=r["scale"],
                band=r["band"],
                score_composite=r["score_composite"],
                features=r["features"],
                # three-way: null | "accept" | "reject" | "filter_leak".
                # filter_leak = prefilter feedback ("dead/noisy/barren — step-3 should
                # have dropped this"), NOT a quality judgment: it is EXCLUDED from the
                # accept-vs-reject fit and reported only as a leak-rate diagnostic.
                # null->value is the only allowed mutation.
                label=dict(klass=None),
            )
            f.write(json.dumps(row) + "\n")

    meta = dict(
        batch_id=BATCH_ID, created="2026-07-23", generator=__file__.replace(str(ROOT) + "\\", ""),
        purpose="q4 stage-1 three-way labels: accept ('worth stage-2 time', high-recall) / "
                "reject ('clean but not q4-worthy') / filter_leak (prefilter feedback — "
                "EXCLUDED from the accept-vs-reject fit, reported as a leak-rate diagnostic)",
        label_classes=["accept", "reject", "filter_leak"],
        n_windows=len(selected), n_minibrots=len(by_mb),
        render=dict(width=W, height=H, aspect=ASPECT, palette=PALETTE),
        sweep=dict(scales=SCALES, stride_frac=STRIDE_FRAC, nms_iou=NMS_IOU),
        prefilter="obvious rejects only: interior_frac>0.85 | (flat>0.92 & occ<0.06) | busy>0.15",
        stratify=dict(by="score_A composite", n_bands=N_BANDS, target_n=TARGET_N, seed=SEED),
        feature_keys=FEATURE_KEYS,
        caveat="windows are crops of ONE medium render -> small-window freq stats are "
               "SCALE-BIASED; true-scale per-window re-render is a stage-2 refinement.",
        separate_store="NOT the v7 location corpus (data/label_corpus). Distribution-bound; "
                       "canonical reader = tools/corpus/q4_window_reader.py.",
    )
    (STORE / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"capture: {n_crop} crops + windows.jsonl -> {STORE} in {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["minibrots", "fields", "sweep", "stratify",
                                      "capture", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("minibrots", "all"):
        stage_minibrots()
    if args.stage in ("fields", "all"):
        stage_fields()
    if args.stage in ("sweep", "all"):
        stage_sweep()
    if args.stage in ("stratify", "all"):
        stage_stratify()
    if args.stage in ("capture", "all"):
        stage_capture()
