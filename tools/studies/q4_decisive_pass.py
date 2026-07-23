#!/usr/bin/env python
"""q4 DECISIVE pass — is the exemplar conjunction a structural class or a low-base-rate bonus?

Measurement pass (NO descent / config / data/ changes). See prompts/q4_decisive_pass.md.

Settles the one open q4 question by SAMPLE SIZE. The prior passes proved the exemplar's
conjunction (interior lakes + busy filament detail + composed rest) is essentially c-UNIQUE
within a handful of hand-picked ladders, and that scalar ∂M screens (dist_dM, M_richness)
bias but do not pinpoint it. But "0/10 non-exemplar corners" was UNDERPOWERED: the organic
base rate is ~1/1000, so 10 draws cannot distinguish a real 1% class from a true one-off.

This pass deep-sweeps ~300 VIABLE near-∂M c's and COUNTS band hits, with a pre-registered
read on the achieved n:
  h >= 3  (>~1%)  -> structural class CONFIRMED (targetable); dump the characterization set.
  h <= 1  (<~1%)  -> low-base-rate BONUS, bank the viability prior.
  h == 2 / n<<300 -> inconclusive-lean-bank; report rate CI, leave the call to Matt.

The large boundary-screened pool + cheap viability screen built here ARE the campaign-3
near-∂M c-sampler — reusable regardless of outcome.

Reuses q4_dM_property.{dump_julia_field,dump_mandel_field,purge,mandel_inside,mandel_de,
signed_dist_dM}, q4_c_perturbation.{compute_metrics,band_dist,TGT_*,S_*}, and
q4_neighborhood_sweep.{auto_maxiter,load_values}. Band geometry (768x432 ss1, the calibrated
band mid≈0.73/int≈0.24/flat≈0.23, CORNER_THRESH 1.5) is held IDENTICAL to the prior passes so
band_dist is directly comparable. All output under out/q4_decisive/ (disposable); fields
purged per-unit.

Stages (idempotent, resume from checkpoint):
  pool     vectorized boundary rejection sampler -> large near-∂M c pool -> pool.json
  screen   one mid-fw julia render per candidate; classify blob/dust/viable; ∂M props for
           viable -> screen.jsonl (checkpoint) + viable.json
  measure  deep-sweep viable (20 framings, band-calibrated res); throughput-projected +
           per-c TIME-GATED + hard-kill backstop -> metrics.jsonl (checkpoint)
  analyze  count hits @ CORNER_THRESH + sensitivity curve + Wilson 95% CI + pre-registered
           branch + characterization set (hits vs near-miss) + ridge/cliff -> analysis.json
  morph    motif variety among hits (needs torch/timm/GPU) -> morph.json
  sheets   exemplar ref + hit contact sheet (annotated) + runner-up ranked strip -> *.png
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 console chokes on ∂/->
except Exception:
    pass

from tools.studies.q4_neighborhood_sweep import auto_maxiter, load_values          # noqa: E402
from tools.studies.q4_c_perturbation import compute_metrics, band_dist              # noqa: E402
from tools.studies.q4_dM_property import (                                          # noqa: E402
    EXE, PALETTE, EX_C, mandel_inside, mandel_de, signed_dist_dM,
    dump_mandel_field, purge,
    MEAS_W, MEAS_H, FW_M, M_MAXITER,
)

OUT = ROOT / "out" / "q4_decisive"
FIELDS = OUT / "fields"

# --- boundary sampler ------------------------------------------------------
BOX_RE = (-2.05, 0.55)              # bounding box that contains ∂M (needle to cardioid tip)
BOX_IM = (-1.20, 1.20)
SHELL_EPS = 0.02                    # near-∂M shell half-width (matches the viable-shell scale)
RING_DIRS = 8                       # ring samples for the vectorized near-boundary test
SAMP_MAXITER = 1200                 # membership maxiter for the cheap near-boundary test
MIN_SEP = 0.006                     # greedy c-plane dedup separation (distinct julia parents)
POOL_TARGET = 750                   # candidate c's to screen (yields ~300 viable at ~40%)
SAMP_BATCH = 60000                  # vectorized draw batch
SAMP_MAX_BATCHES = 40

# --- viability screen (one mid-fw julia render / candidate) ----------------
SCREEN_W, SCREEN_H = 512, 288       # coarse classify res (NOT band-calibrated — screen only)
SCREEN_FW = 0.60                    # representative center framing
SCR_BLOB_INT = 0.85                 # interior_frac above -> solid blob (deep inside the set)
SCR_DUST_MID = 0.04                 # mid_detail below AND ...
SCR_DUST_OCC = 0.06                 # ... occupancy below -> dust (outside / sparse)

# --- julia deep sweep (band-calibrated geometry — MUST match prior passes) --
# fw hand-set: one deep probe (0.03) + coverage of the productive band incl. the exemplar's
# own best fw (0.858). The prior pass showed corners live at fw>=0.16; deep rungs never won,
# but the 0.03 probe is retained per spec so depth-confound stays ruled out.
SWEEP_FWS = [0.03, 0.15, 0.40, 0.858, 1.40]
PAN_FRAC = 0.15
PANS = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]     # 5 fw x 4 pan = 20 framings/c
RENDER_TIMEOUT = 30                 # per-render hard-kill backstop (hung-c protection, s)

# --- corner / hit threshold (IDENTICAL to prior passes -> directly comparable) -
CORNER_THRESH = 1.5
SENS_THRESHOLDS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# --- morph geometry (library morph-canon) ----------------------------------
MORPH_W, MORPH_H, MORPH_SS = 640, 360, 2
NEAR_DUP = 0.974
MORPH_MEDIAN_YARD = 0.851


# --------------------------------------------------------------------------- #
# field dump (local — band-calibrated geometry + per-render hard-kill timeout)  #
# --------------------------------------------------------------------------- #
def dump_julia_field(c_re, c_im, cx, cy, fw, out_bin, maxiter, w=MEAS_W, h=MEAS_H, ss=1,
                     timeout=None):
    cmd = [str(EXE), "render-one", "--julia", "--c", repr(c_re), repr(c_im),
           "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(maxiter),
           "--width", str(w), "--height", str(h), "--supersample", str(ss),
           "--dump-field", str(out_bin), "--dump-field-source", "f64"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"julia dump-field failed: {r.stderr[-400:]}")


# --------------------------------------------------------------------------- #
# pool — vectorized boundary rejection sampler                                 #
# --------------------------------------------------------------------------- #
def _near_boundary_mask(re, im, eps, maxiter=SAMP_MAXITER, n_dirs=RING_DIRS):
    """Vectorized near-∂M test: True where membership is NOT constant over {c} ∪ ring(eps).

    A point within ~eps of ∂M has at least one of its `n_dirs` ring neighbours on the other
    side of the set. Weights the kept set by boundary arc length -> samples the WHOLE ∂M
    (bulbs, seahorse/elephant valleys, dendrites) in proportion to how much boundary is there.
    """
    m0 = mandel_inside(re, im, maxiter)
    near = np.zeros(re.shape, dtype=bool)
    ang = np.linspace(0.0, 2.0 * math.pi, n_dirs, endpoint=False)
    for a in ang:
        mem = mandel_inside(re + eps * math.cos(a), im + eps * math.sin(a), maxiter)
        near |= (mem != m0)
    return near, m0


def _greedy_dedup(pts, min_sep):
    """Greedy min-separation thinning via a cell hash. Keeps a spatially spread subset."""
    cell = min_sep
    occupied = {}
    kept = []
    for x, y in pts:
        cx, cy = int(math.floor(x / cell)), int(math.floor(y / cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (px, py) in occupied.get((cx + dx, cy + dy), ()):  # neighbour cells
                    if (px - x) ** 2 + (py - y) ** 2 < min_sep * min_sep:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            occupied.setdefault((cx, cy), []).append((x, y))
            kept.append((x, y))
    return kept


def build_pool(seed=0):
    FIELDS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    collected = []
    t0 = time.time()
    for it in range(SAMP_MAX_BATCHES):
        re = rng.uniform(*BOX_RE, SAMP_BATCH)
        im = rng.uniform(*BOX_IM, SAMP_BATCH)
        near, _ = _near_boundary_mask(re, im, SHELL_EPS)
        for x, y in zip(re[near], im[near]):
            collected.append((float(x), float(y)))
        # dedup incrementally so we can stop as soon as we clear the target
        ded = _greedy_dedup(collected, MIN_SEP)
        print(f"  batch {it}: +{int(near.sum())} near-∂M  cumulative-distinct {len(ded)} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if len(ded) >= POOL_TARGET:
            collected = ded
            break
    else:
        collected = _greedy_dedup(collected, MIN_SEP)

    # shuffle then cap so the kept set is an unbiased spatial spread, not draw-ordered
    idx = rng.permutation(len(collected))
    kept = [collected[i] for i in idx[:POOL_TARGET]]

    pool = [dict(cid="exemplar", c_re=EX_C[0], c_im=EX_C[1], anchor=True)]
    for i, (x, y) in enumerate(kept):
        pool.append(dict(cid=f"b{i:04d}", c_re=x, c_im=y, anchor=False))
    (OUT / "pool.json").write_text(json.dumps(pool, indent=2))
    print(f"pool: {len(pool)} c's (1 exemplar + {len(kept)} boundary) -> pool.json "
          f"[{time.time()-t0:.1f}s]")


# --------------------------------------------------------------------------- #
# screen — one mid-fw julia render / candidate; classify + ∂M props for viable #
# --------------------------------------------------------------------------- #
def classify(m):
    """blob / dust / viable from a single mid-fw center render's metrics."""
    if m["interior_frac"] > SCR_BLOB_INT:
        return "blob"
    if m["mid_detail_frac"] < SCR_DUST_MID and m["occupancy"] < SCR_DUST_OCC:
        return "dust"
    return "viable"


def _dM_props(cre, cim):
    """Cheap per-c ∂M properties for a VIABLE candidate (dist_dM + local M_richness)."""
    dist, m0 = signed_dist_dM(cre, cim)
    de = mandel_de(cre, cim, M_MAXITER)
    b = FIELDS / f"M_{abs(hash((cre, cim))) & 0xffffff:06x}.bin"
    dump_mandel_field(cre, cim, FW_M, b, M_MAXITER)
    mm = compute_metrics(load_values(b))
    purge(b)
    return dict(dist_dM=dist, c_inside=m0, exterior_de=de,
                M_occupancy=mm["occupancy"], M_mid_detail=mm["mid_detail_frac"],
                M_interior_frac=mm["interior_frac"])


def stage_screen():
    FIELDS.mkdir(parents=True, exist_ok=True)
    pool = json.loads((OUT / "pool.json").read_text())
    ckpt = OUT / "screen.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["cid"]] = d
    print(f"screen: {len(pool)} candidates, {len(done)} done")
    t0 = time.time()
    n_new = 0
    with ckpt.open("a") as f:
        for c in pool:
            if c["cid"] in done:
                continue
            b = FIELDS / f"S_{c['cid']}.bin"
            try:
                dump_julia_field(c["c_re"], c["c_im"], 0.0, 0.0, SCREEN_FW, b,
                                 auto_maxiter(SCREEN_FW), w=SCREEN_W, h=SCREEN_H)
                m = compute_metrics(load_values(b))
            finally:
                purge(b)
            cls = classify(m)
            rec = dict(cid=c["cid"], c_re=c["c_re"], c_im=c["c_im"],
                       anchor=c.get("anchor", False), klass=cls,
                       scr_interior=m["interior_frac"], scr_mid=m["mid_detail_frac"],
                       scr_occ=m["occupancy"])
            if cls == "viable":
                rec.update(_dM_props(c["c_re"], c["c_im"]))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_new += 1
            if n_new % 100 == 0:
                print(f"  {n_new} screened ({time.time()-t0:.1f}s)", flush=True)

    rows = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    counts = {}
    for r in rows:
        counts[r["klass"]] = counts.get(r["klass"], 0) + 1
    viable = [r for r in rows if r["klass"] == "viable"]
    (OUT / "viable.json").write_text(json.dumps(viable, indent=2))
    ex = next((r for r in rows if r["cid"] == "exemplar"), None)
    print(f"screen done: {counts}  ({time.time()-t0:.1f}s)")
    print(f"  exemplar screened as: {ex['klass'] if ex else 'MISSING'}")
    print(f"  viable -> {len(viable)} c's -> viable.json")


# --------------------------------------------------------------------------- #
# measure — deep-sweep viable (throughput-projected + per-c time-gated)         #
# --------------------------------------------------------------------------- #
def build_framings():
    fr, fid = [], 0
    for iz, fw in enumerate(SWEEP_FWS):
        for ip, (sx, sy) in enumerate(PANS):
            fr.append(dict(fid=fid, iz=iz, ip=ip, fw=float(fw),
                           dcx=float(sx * PAN_FRAC * fw), dcy=float(sy * PAN_FRAC * fw)))
            fid += 1
    return fr


def stage_measure(cap_seconds=1800):
    FIELDS.mkdir(parents=True, exist_ok=True)
    viable = json.loads((OUT / "viable.json").read_text())
    frs = build_framings()
    ckpt = OUT / "metrics.jsonl"
    done = set()
    done_cids = set()
    if ckpt.exists():
        percid = {}
        for line in ckpt.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["cid"], d["fid"]))
                percid[d["cid"]] = percid.get(d["cid"], 0) + 1
        done_cids = {cid for cid, n in percid.items() if n >= len(frs)}
    todo = [c for c in viable if c["cid"] not in done_cids]
    print(f"measure: {len(viable)} viable x {len(frs)} framings; {len(done_cids)} c's "
          f"complete, {len(todo)} to go. cap={cap_seconds}s", flush=True)

    t0 = time.time()
    per_c_times = []
    est_per_c = len(frs) * 0.12          # rough prior until we have real timings
    n_done_this_run = 0
    stopped = None
    with ckpt.open("a") as f:
        for c in todo:
            elapsed = time.time() - t0
            if elapsed + est_per_c > cap_seconds:
                stopped = f"time-gate: elapsed {elapsed:.0f}s + est_per_c {est_per_c:.1f}s " \
                          f"> cap {cap_seconds}s"
                print(f"  STOP before {c['cid']}: {stopped}", flush=True)
                break
            c_t0 = time.time()
            for fr in frs:
                key = (c["cid"], fr["fid"])
                if key in done:
                    continue
                mi = auto_maxiter(fr["fw"])
                b = FIELDS / f"J_{c['cid']}_{fr['fid']:02d}.bin"
                try:
                    dump_julia_field(c["c_re"], c["c_im"], fr["dcx"], fr["dcy"], fr["fw"],
                                     b, mi, timeout=RENDER_TIMEOUT)
                    m = compute_metrics(load_values(b))
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    print(f"    !! {c['cid']} fid{fr['fid']} render fail ({type(e).__name__}) "
                          f"— skipping framing", flush=True)
                    purge(b)
                    continue
                purge(b)
                rec = {"cid": c["cid"], "c_re": c["c_re"], "c_im": c["c_im"],
                       "fid": fr["fid"], "iz": fr["iz"], "ip": fr["ip"],
                       "fw": fr["fw"], "cx": fr["dcx"], "cy": fr["dcy"], "maxiter": mi, **m}
                f.write(json.dumps(rec) + "\n")
                f.flush()
            per_c_times.append(time.time() - c_t0)
            n_done_this_run += 1
            # refine throughput estimate + project achieved n after the first ~10 c's
            if n_done_this_run == 10:
                est_per_c = float(np.median(per_c_times))
                remaining_budget = cap_seconds - (time.time() - t0)
                projectable = int(remaining_budget / max(est_per_c, 1e-6))
                projected_n = len(done_cids) + n_done_this_run + max(0, projectable)
                projected_n = min(projected_n, len(viable))
                print(f"  [timing] first 10 c's: median {est_per_c:.1f}s/c -> "
                      f"projected achieved n ≈ {projected_n} of {len(viable)} viable "
                      f"in the {cap_seconds}s window", flush=True)
            elif n_done_this_run % 25 == 0:
                print(f"  {n_done_this_run} c's swept this run "
                      f"({time.time()-t0:.1f}s, {np.median(per_c_times):.1f}s/c)", flush=True)

    n_complete = len(done_cids) + n_done_this_run
    (OUT / "measure_status.json").write_text(json.dumps(dict(
        n_viable=len(viable), n_complete=n_complete, n_this_run=n_done_this_run,
        cap_seconds=cap_seconds, elapsed=time.time() - t0,
        est_per_c=est_per_c, stopped=stopped), indent=2))
    print(f"measure done: {n_done_this_run} new c's, {n_complete}/{len(viable)} complete "
          f"({time.time()-t0:.1f}s). {'STOPPED: '+stopped if stopped else 'all viable swept'}")


def load_metrics():
    return [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines() if l.strip()]


def by_cid(recs):
    d = {}
    for r in recs:
        d.setdefault(r["cid"], []).append(r)
    return d


# --------------------------------------------------------------------------- #
# analyze — count hits + sensitivity + Wilson CI + branch + characterization    #
# --------------------------------------------------------------------------- #
def _wilson(k, n, z=1.959963985):
    """Wilson score 95% CI for a binomial rate k/n (closed form, no scipy)."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viable = {v["cid"]: v for v in json.loads((OUT / "viable.json").read_text())}
    recs = load_metrics()
    groups = by_cid(recs)
    frs = build_framings()

    # per-c: best (band-nearest) framing among a COMPLETE sweep only
    per_c = {}
    for cid, rows in groups.items():
        if len(rows) < len(frs):
            continue                      # partial c (time-gate cutoff) — exclude from n
        best = min(rows, key=band_dist)
        mbd = band_dist(best)
        p = dict(cid=cid, min_band_dist=mbd,
                 best_fid=best["fid"], best_fw=best["fw"], best_ip=best["ip"],
                 best_interior=best["interior_frac"], best_mid=best["mid_detail_frac"],
                 best_flat=best["flat_frac"], best_bnb=best["busy_near_black"],
                 best_cr=best["coherent_rest"], best_cx=best["cx"], best_cy=best["cy"],
                 c_re=best["c_re"], c_im=best["c_im"],
                 anchor=(cid == "exemplar"))
        v = viable.get(cid, {})
        p.update(dist_dM=v.get("dist_dM"), M_occupancy=v.get("M_occupancy"),
                 M_mid_detail=v.get("M_mid_detail"), c_inside=v.get("c_inside"))
        per_c[cid] = p

    # n = viable c's actually deep-swept, EXCLUDING the exemplar (it is the positive control,
    # not a random draw). h = non-exemplar c's whose best framing hits the band.
    swept = [p for p in per_c.values() if p["cid"] != "exemplar"]
    n = len(swept)
    ex = per_c.get("exemplar")
    hits = [p for p in swept if p["min_band_dist"] <= CORNER_THRESH]
    h = len(hits)

    p_hat, lo, hi = _wilson(h, n)
    rule_of_three = 3.0 / n if n else float("nan")     # h==0 upper-rate rule of thumb

    # sensitivity curve — hit count vs threshold (verdict robustness)
    sens = [dict(thresh=t, hits=int(sum(1 for p in swept if p["min_band_dist"] <= t)))
            for t in SENS_THRESHOLDS]

    # pre-registered branch
    if h >= 3:
        branch = "CONFIRMED_structural_class"
    elif h <= 1:
        branch = "BONUS_low_base_rate_bank"
    else:
        branch = "INCONCLUSIVE_lean_bank"
    n_short = n < 250
    if n_short and branch == "INCONCLUSIVE_lean_bank":
        branch += "_and_n_short"

    # ---- characterization: what do hits share that near-miss viable c's don't? ----
    swept_sorted = sorted(swept, key=lambda p: p["min_band_dist"])
    near_miss = [p for p in swept if CORNER_THRESH < p["min_band_dist"] <= CORNER_THRESH + 1.0]

    def block(group):
        return dict(
            n=len(group),
            min_band_dist=_mean([p["min_band_dist"] for p in group]),
            abs_dist_dM=_mean([abs(p["dist_dM"]) for p in group if p["dist_dM"] is not None]),
            M_mid_detail=_mean([p["M_mid_detail"] for p in group]),
            M_occupancy=_mean([p["M_occupancy"] for p in group]),
            best_fw=_mean([p["best_fw"] for p in group]),
            best_interior=_mean([p["best_interior"] for p in group]),
            best_mid=_mean([p["best_mid"] for p in group]),
            best_flat=_mean([p["best_flat"] for p in group]),
            best_bnb=_mean([p["best_bnb"] for p in group]),
            best_cr=_mean([p["best_cr"] for p in group]),
        )

    charac = dict(exemplar=block([ex]) if ex else None,
                  hits=block(hits), near_miss=block(near_miss),
                  all_viable=block(swept))

    # ---- runner-up structure: graded ridge or a cliff to the exemplar? ----
    head = [dict(cid=p["cid"], min_band_dist=p["min_band_dist"], best_fw=p["best_fw"],
                 dist_dM=p["dist_dM"], interior=p["best_interior"], mid=p["best_mid"],
                 flat=p["best_flat"]) for p in swept_sorted[:20]]
    bds = np.array([p["min_band_dist"] for p in swept_sorted])
    # gap between exemplar's band_dist and the best non-exemplar (cliff = large gap)
    ex_bd = ex["min_band_dist"] if ex else None
    best_nonex = swept_sorted[0]["min_band_dist"] if swept_sorted else None
    ridge = dict(
        exemplar_band_dist=ex_bd,
        best_nonexemplar_band_dist=best_nonex,
        exemplar_to_best_nonex_gap=(best_nonex - ex_bd) if (ex_bd is not None and best_nonex is not None) else None,
        band_dist_quantiles={q: float(np.quantile(bds, q / 100)) for q in [5, 25, 50, 75, 95]} if n else {},
        n_within_2x_exemplar=int(sum(1 for b in bds if ex_bd is not None and b <= 2 * ex_bd)),
    )

    analysis = dict(
        n_viable_total=len(viable), n_swept=n, h=h,
        rate=p_hat, ci95=[lo, hi], rule_of_three_upper=rule_of_three,
        corner_thresh=CORNER_THRESH, branch=branch,
        exemplar_band_dist=ex_bd,
        sensitivity=sens,
        hit_cids=[p["cid"] for p in sorted(hits, key=lambda x: x["min_band_dist"])],
        characterization=charac,
        runner_up=ridge,
        per_c={p["cid"]: p for p in per_c.values()},
    )
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # ---- plots: sensitivity curve + band_dist ECDF + dist_dM vs band_dist ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    axs[0].plot([s["thresh"] for s in sens], [s["hits"] for s in sens], "o-")
    axs[0].axvline(CORNER_THRESH, color="green", ls="--", label=f"corner thr {CORNER_THRESH}")
    axs[0].set_xlabel("band_dist threshold"); axs[0].set_ylabel("hit count (non-exemplar)")
    axs[0].set_title(f"Sensitivity: hits vs threshold (n={n})"); axs[0].legend(fontsize=8)

    xs = np.sort(bds)
    axs[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs) if len(xs) else [], drawstyle="steps-post")
    axs[1].axvline(CORNER_THRESH, color="green", ls="--", label="corner thr")
    if ex_bd is not None:
        axs[1].axvline(ex_bd, color="red", ls=":", label=f"exemplar {ex_bd:.2f}")
    axs[1].set_xlabel("min band_dist"); axs[1].set_ylabel("ECDF")
    axs[1].set_title("Runner-up structure (ridge vs cliff)"); axs[1].legend(fontsize=8)

    dd = [abs(p["dist_dM"]) for p in swept if p["dist_dM"] is not None]
    bb = [p["min_band_dist"] for p in swept if p["dist_dM"] is not None]
    hc = ["tab:green" if b <= CORNER_THRESH else "tab:blue" for b in bb]
    axs[2].scatter(dd, bb, c=hc, s=22, edgecolors="k", linewidths=0.3)
    if ex and ex["dist_dM"] is not None:
        axs[2].scatter([abs(ex["dist_dM"])], [ex["min_band_dist"]], c="red", s=90,
                       marker="*", edgecolors="k", label="exemplar", zorder=5)
    axs[2].axhline(CORNER_THRESH, color="green", ls="--", lw=1)
    axs[2].set_xlabel("abs(dist_dM)"); axs[2].set_ylabel("min band_dist")
    axs[2].set_title("|dist_dM| vs J-quality (viable)"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "decisive.png", dpi=115); plt.close(fig)

    # ---- print verdict ----
    print(f"\n{'='*74}\nq4 DECISIVE — VERDICT\n{'='*74}")
    print(f"viable total {len(viable)}  |  deep-swept (complete, non-exemplar) n = {n}")
    print(f"exemplar (positive control) band_dist = {ex_bd:.3f}  "
          f"{'HIT' if ex_bd is not None and ex_bd<=CORNER_THRESH else 'MISS'}"
          if ex_bd is not None else "exemplar: not swept")
    print(f"HITS (band_dist <= {CORNER_THRESH}): h = {h}")
    print(f"rate = {p_hat*100:.2f}%   Wilson 95% CI [{lo*100:.2f}%, {hi*100:.2f}%]"
          f"   (rule-of-3 upper if h=0: {rule_of_three*100:.2f}%)")
    print(f"PRE-REGISTERED BRANCH -> {branch}")
    print("\nsensitivity (hits vs threshold):")
    for s in sens:
        print(f"   thr {s['thresh']:<5} -> {s['hits']} hits")
    if h:
        print(f"\nhit cids: {analysis['hit_cids']}")
    print("\ncharacterization (mean over group):")
    for gname in ("exemplar", "hits", "near_miss", "all_viable"):
        g = charac[gname]
        if not g:
            continue
        print(f"  {gname:11s} n={g['n']:<4} |dist_dM|={g['abs_dist_dM']:.4f} "
              f"M_mid={g['M_mid_detail']:.3f} best_fw={g['best_fw']:.3f} "
              f"int={g['best_interior']:.3f} mid={g['best_mid']:.3f} flat={g['best_flat']:.3f}")
    print("\nrunner-up structure:")
    print(f"  exemplar {ridge['exemplar_band_dist']}  best-nonex {ridge['best_nonexemplar_band_dist']}"
          f"  gap {ridge['exemplar_to_best_nonex_gap']}")
    print(f"  band_dist quantiles {ridge['band_dist_quantiles']}")
    print(f"  n within 2x exemplar band_dist: {ridge['n_within_2x_exemplar']}")
    print(f"\ntop of the viable ridge:")
    for p in head[:10]:
        print(f"  {p['cid']:8s} bd {p['min_band_dist']:.2f} fw {p['best_fw']:.3f} "
              f"dM {p['dist_dM'] if p['dist_dM'] is None else round(p['dist_dM'],4)} "
              f"int{p['interior']:.2f} mid{p['mid']:.2f}")
    print("\nwrote analysis.json + decisive.png")


# --------------------------------------------------------------------------- #
# morph — motif variety among hits (if >= 2)                                   #
# --------------------------------------------------------------------------- #
def stage_morph():
    from tools.wallpaper.library_annotate import morph_gray_image
    from tools.curation.colored_clip import load_clip, embed_clip

    analysis = json.loads((OUT / "analysis.json").read_text())
    per_c = analysis["per_c"]
    # motif set = exemplar + hit c's; if <2 hits, fall back to the top viable ridge so the
    # variety readout is non-trivial.
    hit_cids = analysis["hit_cids"]
    if len(hit_cids) >= 2:
        cids = ["exemplar"] + [c for c in hit_cids if c != "exemplar"]
    else:
        ridge = sorted([c for c in per_c if c != "exemplar"],
                       key=lambda c: per_c[c]["min_band_dist"])[:11]
        cids = ["exemplar"] + ridge
        print(f"  <2 hits: morphing exemplar + top-{len(ridge)} viable ridge instead")
    mdir = OUT / "morph_fields"
    mdir.mkdir(parents=True, exist_ok=True)

    class _Field:
        def __init__(self, values, ss):
            self.values = values
            self.supersample = ss

    imgs, kept = [], []
    for cid in cids:
        p = per_c[cid]
        b = mdir / f"m_{cid}.bin"
        if not b.exists():
            dump_julia_field(p["c_re"], p["c_im"], p["best_cx"], p["best_cy"], p["best_fw"],
                             b, auto_maxiter(p["best_fw"]), MORPH_W, MORPH_H, MORPH_SS)
        imgs.append(morph_gray_image(_Field(load_values(b), MORPH_SS)))
        kept.append(cid)

    print(f"embedding {len(imgs)} deep bests via morph_gray + CLIP ...")
    model, tf = load_clip()
    emb = embed_clip(model, tf, imgs)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = emb @ emb.T
    n = len(kept)
    off = sim[np.triu_indices(n, k=1)] if n > 1 else np.array([np.nan])
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= NEAR_DUP:
                parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(kept[i])
    dup_pairs = [(kept[i], kept[j], float(sim[i, j]))
                 for i in range(n) for j in range(i + 1, n) if sim[i, j] >= NEAR_DUP]
    morph = dict(n=n, cids=kept, is_hit_set=len(hit_cids) >= 2,
                 median_offdiag=float(np.median(off)), mean_offdiag=float(np.mean(off)),
                 max_offdiag=float(np.max(off)) if n > 1 else float("nan"),
                 near_dup_threshold=NEAR_DUP, median_yardstick=MORPH_MEDIAN_YARD,
                 distinct_look_count=len(clusters),
                 near_dup_pairs=dup_pairs, clusters=[sorted(v) for v in clusters.values()])
    np.savez(OUT / "morph_sim.npz", sim=sim, cids=np.array(kept))
    (OUT / "morph.json").write_text(json.dumps(morph, indent=2))
    print(f"\nMOTIF VARIETY: {n} deep bests -> {len(clusters)} distinct looks (@ {NEAR_DUP})")
    print(f"  median off-diag cos {morph['median_offdiag']:.3f} "
          f"(yardstick {MORPH_MEDIAN_YARD}); max {morph['max_offdiag']:.3f}")
    for a, b, s in sorted(dup_pairs, key=lambda x: -x[2]):
        print(f"    near-dup {s:.4f}  {a} <-> {b}")


# --------------------------------------------------------------------------- #
# sheets — exemplar ref + hit contact sheet + runner-up ranked strip            #
# --------------------------------------------------------------------------- #
SHEET_W, SHEET_H, SHEET_SS = 1024, 576, 2


def render_color(c_re, c_im, cx, cy, fw, out_png):
    cmd = [str(EXE), "render-one", "--julia", "--c", repr(c_re), repr(c_im),
           "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(auto_maxiter(fw)),
           "--width", str(SHEET_W), "--height", str(SHEET_H), "--supersample", str(SHEET_SS),
           "--palette", PALETTE, "--out", str(out_png)]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        raise RuntimeError(rr.stderr[-400:])


def stage_sheets():
    from PIL import Image, ImageDraw
    analysis = json.loads((OUT / "analysis.json").read_text())
    per_c = analysis["per_c"]
    ex = per_c["exemplar"]
    rend = OUT / "renders"
    rend.mkdir(exist_ok=True)
    render_color(ex["c_re"], ex["c_im"], ex["best_cx"], ex["best_cy"], ex["best_fw"],
                 OUT / "exemplar_large.png")

    # order: exemplar, then hits (if any), then the top of the viable ridge
    hit_cids = analysis["hit_cids"]
    ridge = [c for c in sorted(per_c, key=lambda c: per_c[c]["min_band_dist"])
             if c != "exemplar"]
    order = ["exemplar"] + [c for c in ridge if c in hit_cids]
    order += [c for c in ridge if c not in hit_cids][:max(0, 23 - len(order))]

    def cap(cid):
        p = per_c[cid]
        tag = "*HIT" if (cid != "exemplar" and p["min_band_dist"] <= CORNER_THRESH) else \
              ("EX" if cid == "exemplar" else "")
        dM = "" if p["dist_dM"] is None else f"dM{p['dist_dM']:+.3f}"
        return f"{cid[:9]} bd{p['min_band_dist']:.2f} fw{p['best_fw']:.3f} {dM} {tag}"

    tw, th = 380, 214
    cols = 4
    rows = (len(order) + cols - 1) // cols
    pad, top = 6, 34
    W = cols * tw + (cols + 1) * pad
    H = top + rows * (th + 22) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), "q4 DECISIVE — per-c DEEP best (band-nearest). red=exemplar "
                     "green=*HIT (band_dist<=1.5). Rest = top of the viable ridge.",
           fill=(235, 235, 235))
    for i, cid in enumerate(order):
        p = per_c[cid]
        pp = rend / f"r_{cid}.png"
        if not pp.exists():
            render_color(p["c_re"], p["c_im"], p["best_cx"], p["best_cy"], p["best_fw"], pp)
        im = Image.open(pp).resize((tw, th))
        cx, cy = i % cols, i // cols
        x = pad + cx * (tw + pad); y = top + cy * (th + 22)
        canvas.paste(im, (x, y))
        is_hit = cid != "exemplar" and p["min_band_dist"] <= CORNER_THRESH
        color = (255, 140, 140) if cid == "exemplar" else \
            ((150, 255, 150) if is_hit else (205, 205, 205))
        d.text((x, y + th + 4), cap(cid), fill=color)
    canvas.save(OUT / "sheet_hits_and_ridge.png")
    print("wrote sheet_hits_and_ridge.png + exemplar_large.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "screen", "measure", "analyze", "morph",
                                      "sheets", "all"])
    ap.add_argument("--cap-seconds", type=int, default=1800,
                    help="measure wall-clock cap (per-c time-gated)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("pool", "all"):
        build_pool(seed=args.seed)
    if args.stage in ("screen", "all"):
        stage_screen()
    if args.stage in ("measure", "all"):
        stage_measure(cap_seconds=args.cap_seconds)
    if args.stage in ("analyze", "all"):
        stage_analyze()
    if args.stage in ("morph", "all"):
        stage_morph()
    if args.stage in ("sheets", "all"):
        stage_sheets()
