#!/usr/bin/env python
"""q4 c-perturbation study — which exemplar-calibrated criteria GENERALIZE across Julia c?

Measurement pass (NO descent / config / data/ changes). See
prompts/q4_c_perturbation.md. The neighborhood sweep (q4_neighborhood_sweep.py)
proved the artist-quality "corner" is a coherent, reachable region at the *exemplar's*
c — but at one c it is a single motif family. This pass sweeps ACROSS Julia c:
rings around the exemplar c + a few deliberately-farther c's near dM, a center-descent
framing sweep per c, and asks:

  1. GENERALIZATION  — do the exemplar-calibrated criteria pick good framings at OTHER
                       c's, or are they overfit? (which axes generalize vs c-specific)
  2. CORNER per c    — does a target-band corner exist at each c; where (fw/pan)?
  3. MOTIF VARIETY   — morph_clip cos across per-c bests (median~0.851, near-dup 0.974):
                       distinct looks or one motif re-framed?
  4. VARIANT B       — any c with large slow-escape basins (deep_frac > ~0.085)?

Reuses the neighborhood sweep's calibrated two-scale detail bands + auto_maxiter, and
the library morph_gray/CLIP recipe. Two NEW axes:
  * busy_near_black  — fine-scale detail in a dilated ring around interior boundaries
                       (distracting busyness that also breaks strange render modes). PENALTY.
  * coherent_rest    — size of the LARGEST connected low-variance region (composed smooth
                       sweep), as frame fraction. Distinct from flat_frac (total).

Field source is the f64 escape-time backend (colormap-invariant); colored sheet renders
use render-one with the exemplar palette. Everything writes under out/q4_cperturb/
(disposable); field bins are purged per-unit.

Stages (idempotent, resume from checkpoint):
  measure   field-dump the c x framing grid, compute metrics -> metrics.jsonl
  analyze   generalization / corner / variant-B verdict -> analysis.json + plots
  morph     morph_gray + CLIP over per-c bests -> morph.json  (needs torch/timm/GPU)
  sheets    colored judge-quality renders grouped by c + flag sheets -> *.png
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
from scipy.ndimage import binary_dilation, label as cc_label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse the neighborhood sweep's load-bearing, calibrated pieces (bands, two_scale,
# auto_maxiter, load_values). Do NOT reimplement the detail decomposition.
from tools.studies.q4_neighborhood_sweep import (  # noqa: E402
    auto_maxiter, load_values, two_scale,
    STRUCT_FLAT, STRUCT_MID_HI, FINE_SPECKLE, SPECKLE_STRUCT, DEEP_NORM,
)

OUT = ROOT / "out" / "q4_cperturb"
FIELDS = OUT / "fields"
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
PALETTE = "twilight_shifted"

# Exemplar (center-view Julia, origin-centered).
EX_C = (0.26103, -0.48932)
EX_FW = 0.75

# --- measurement geometry --------------------------------------------------
MEAS_W, MEAS_H = 768, 432            # ss1 field-dump res (colormap-invariant)
RING_PX = 4                          # busy_near_black dilation ring width @ meas res

# --- calibration target (mild preference, Matt's sheet_A picks) ------------
TGT_MID, TGT_INT, TGT_FLAT = 0.73, 0.24, 0.23
S_MID, S_INT, S_FLAT = 0.15, 0.10, 0.15   # band-distance scales (~sheet_A spreads)
W_BNB, W_CR = 2.0, 2.0                     # composite weights on the two new axes

# --- morph geometry (MUST match library morph-canon: colored_clip W,H,SS) --
MORPH_W, MORPH_H, MORPH_SS = 640, 360, 2
NEAR_DUP = 0.974                     # morph_clip near-dup yardstick
MORPH_MEDIAN_YARD = 0.851           # inter-location morph_clip median yardstick


# --------------------------------------------------------------------------- #
# c sampling                                                                   #
# --------------------------------------------------------------------------- #
RING_RADII = [0.03, 0.08, 0.16]
RING_ANGLES = 6
# Deliberately-farther c's near dM, spread across regions (real generalization test).
FAR_CS = [
    ("far_west_neck", -0.8, 0.156),
    ("far_upper_card", 0.285, 0.535),
    ("far_upper_bulb", -0.4, 0.6),
    ("far_rabbit", -0.70176, -0.3842),
]


def build_cs():
    """List of c dicts: cid, kind, c_re, c_im, radius, angle_deg."""
    cs = [dict(cid="exemplar", kind="exemplar", c_re=EX_C[0], c_im=EX_C[1],
               radius=0.0, angle_deg=0.0)]
    for ri, r in enumerate(RING_RADII):
        for ai in range(RING_ANGLES):
            th = 2 * math.pi * ai / RING_ANGLES
            cs.append(dict(cid=f"ring{ri}_a{ai}", kind="ring",
                           c_re=EX_C[0] + r * math.cos(th),
                           c_im=EX_C[1] + r * math.sin(th),
                           radius=r, angle_deg=math.degrees(th)))
    for name, cre, cim in FAR_CS:
        cs.append(dict(cid=name, kind="far", c_re=cre, c_im=cim,
                       radius=None, angle_deg=None))
    return cs


# --------------------------------------------------------------------------- #
# framing sweep (center-descent + small pan, per c)                            #
# --------------------------------------------------------------------------- #
N_FW = 7                             # log-spaced fw in [0.13, 1.5]
PAN_FRAC = 0.15                      # small pan radius (fraction of fw)


def build_framings():
    """Per-c center-descent sweep: fw log-spaced [0.13,1.5] x {center + 4 small pans}.
    Center pan preserves the z->-z symmetry bonus; the 4 small pans probe local
    robustness without leaving the composed region. 7 x 5 = 35 framings/c."""
    fws = np.geomspace(0.13, 1.5, N_FW)
    # small pans: center + 4 diagonals at PAN_FRAC*fw
    pans = [(0.0, 0.0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    fr = []
    fid = 0
    for iz, fw in enumerate(fws):
        for ip, (sx, sy) in enumerate(pans):
            fr.append(dict(fid=fid, iz=iz, ip=ip, fw=float(fw),
                           dcx=float(sx * PAN_FRAC * fw), dcy=float(sy * PAN_FRAC * fw)))
            fid += 1
    return fr


# --------------------------------------------------------------------------- #
# field dump + metrics                                                         #
# --------------------------------------------------------------------------- #
def dump_field(c_re, c_im, cx, cy, fw, out_bin, maxiter, w=MEAS_W, h=MEAS_H, ss=1):
    cmd = [str(EXE), "render-one", "--julia", "--c", repr(c_re), repr(c_im),
           "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(maxiter),
           "--width", str(w), "--height", str(h), "--supersample", str(ss),
           "--dump-field", str(out_bin), "--dump-field-source", "f64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump-field failed: {r.stderr[-400:]}")


def compute_metrics(values):
    """Colormap-invariant metrics from an f64 smooth field (NaN interior).

    Reuses the neighborhood sweep's two-scale (fine/struct_e) detail bands; adds the
    two NEW axes (busy_near_black, coherent_rest) and symmetry."""
    finite = np.isfinite(values)
    interior = ~finite
    interior_frac = float(interior.mean())

    vals = values[finite]
    if vals.size < 16:
        return dict(interior_frac=interior_frac, deep_frac=0.0, detail_in_deep=0.0,
                    flat_frac=float(1.0 - vals.size / values.size), mid_detail_frac=0.0,
                    high_struct_frac=0.0, busy_frac=0.0, occupancy=0.0,
                    distributed_interior=0.0, mean_struct=0.0,
                    busy_near_black=0.0, coherent_rest=0.0, symmetry=1.0)

    lo, hi = np.percentile(vals, [0.5, 99.5])
    span = max(hi - lo, 1e-9)
    norm = np.clip((values - lo) / span, 0.0, 1.0)   # higher = slower escape
    work = np.where(finite, norm, 1.0)               # interior -> deepest, flat
    fine, struct_e = two_scale(work)

    deep_mask = finite & (norm >= DEEP_NORM)
    deep_frac = float(deep_mask.mean())
    detail_in_deep = float(struct_e[deep_mask].mean()) if deep_mask.any() else 0.0

    flat = struct_e < STRUCT_FLAT
    mid = (struct_e >= STRUCT_FLAT) & (struct_e < STRUCT_MID_HI)
    high = struct_e >= STRUCT_MID_HI
    speckle = (fine > FINE_SPECKLE) & (struct_e < SPECKLE_STRUCT)
    flat_frac = float(flat.mean())
    mid_detail_frac = float(mid.mean())
    high_struct_frac = float(high.mean())
    busy_frac = float(speckle.mean())
    occupancy = float((~flat).mean())

    # NEW busy_near_black — fine-scale detail in a dilated ring around interior lakes.
    if interior.any():
        ring = binary_dilation(interior, iterations=RING_PX) & ~interior
        busy_near_black = float((fine[ring] > FINE_SPECKLE).mean()) if ring.any() else 0.0
    else:
        busy_near_black = 0.0

    # NEW coherent_rest — largest connected low-variance region as frame fraction.
    lbl, n = cc_label(flat)
    if n > 0:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        coherent_rest = float(counts.max() / values.size)
    else:
        coherent_rest = 0.0

    # distributed-interior: fraction of 8x8 tiles containing any interior pixel
    tiles = 0
    H, W = values.shape
    th, tw = H // 8, W // 8
    for ty in range(8):
        for tx in range(8):
            if not finite[ty*th:(ty+1)*th, tx*tw:(tx+1)*tw].all():
                tiles += 1
    distributed_interior = tiles / 64.0

    # symmetry — 180deg rotation correlation (center descents ~1.0; pan breaks it)
    rot = np.rot90(work, 2)
    a = work.ravel() - work.mean()
    b = rot.ravel() - rot.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    symmetry = float((a * b).sum() / denom) if denom > 1e-12 else 1.0

    return dict(interior_frac=interior_frac, deep_frac=deep_frac,
                detail_in_deep=detail_in_deep, flat_frac=flat_frac,
                mid_detail_frac=mid_detail_frac, high_struct_frac=high_struct_frac,
                busy_frac=busy_frac, occupancy=occupancy,
                distributed_interior=distributed_interior,
                mean_struct=float(struct_e.mean()),
                busy_near_black=busy_near_black, coherent_rest=coherent_rest,
                symmetry=symmetry)


# --------------------------------------------------------------------------- #
# composite q4 bias score                                                      #
# --------------------------------------------------------------------------- #
def band_dist(r):
    """Distance to the exemplar target band on the 3 stable band axes."""
    return math.sqrt(((r["mid_detail_frac"] - TGT_MID) / S_MID) ** 2
                     + ((r["interior_frac"] - TGT_INT) / S_INT) ** 2
                     + ((r["flat_frac"] - TGT_FLAT) / S_FLAT) ** 2)


def q4_score(r):
    """Candidate composite: -band_dist - busy_near_black + coherent_rest."""
    return -band_dist(r) - W_BNB * r["busy_near_black"] + W_CR * r["coherent_rest"]


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_measure(timing_budget_s=120.0):
    FIELDS.mkdir(parents=True, exist_ok=True)
    cs = build_cs()
    frs = build_framings()
    total = len(cs) * len(frs)
    ckpt = OUT / "metrics.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["cid"], d["fid"]))
    print(f"grid: {len(cs)} c x {len(frs)} framings = {total}, {len(done)} already done")
    t0 = time.time()
    n_new = 0
    with ckpt.open("a") as f:
        for c in cs:
            for fr in frs:
                key = (c["cid"], fr["fid"])
                if key in done:
                    continue
                cx = fr["dcx"]
                cy = fr["dcy"]
                mi = auto_maxiter(fr["fw"])
                b = FIELDS / f"f_{c['cid']}_{fr['fid']:03d}.bin"
                dump_field(c["c_re"], c["c_im"], cx, cy, fr["fw"], b, mi)
                m = compute_metrics(load_values(b))
                rec = {"cid": c["cid"], "kind": c["kind"],
                       "c_re": c["c_re"], "c_im": c["c_im"],
                       "radius": c["radius"], "angle_deg": c["angle_deg"],
                       "fid": fr["fid"], "iz": fr["iz"], "ip": fr["ip"],
                       "fw": fr["fw"], "cx": cx, "cy": cy, "maxiter": mi, **m}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                for p in (b, Path(str(b).replace(".bin", ".json"))):
                    try:
                        p.unlink(missing_ok=True)
                    except PermissionError:
                        pass   # Windows transient exe/file lock — purge is best-effort

                n_new += 1
                if n_new == 20:
                    el = time.time() - t0
                    proj = el / 20 * (total - len(done))
                    print(f"  [timing] first 20 dumps: {el:.1f}s -> "
                          f"projected total {proj:.0f}s for {total - len(done)} remaining")
                if n_new % 50 == 0:
                    print(f"  {n_new} new  ({time.time()-t0:.1f}s)")
    print(f"measure done: {n_new} new in {time.time()-t0:.1f}s -> {ckpt}")


def load_metrics():
    recs = [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return recs


def by_cid(recs):
    d = {}
    for r in recs:
        d.setdefault(r["cid"], []).append(r)
    return d


def is_degenerate(rows):
    """A c is degenerate if it never composes structure + lakes across its sweep."""
    mids = [r["mid_detail_frac"] for r in rows]
    ints = [r["interior_frac"] for r in rows]
    occ = [r["occupancy"] for r in rows]
    med_int = float(np.median(ints))
    if max(mids) < 0.05:
        return True, "no mid-detail (max<0.05) — flat/dust everywhere"
    if med_int > 0.9:
        return True, "median interior>0.9 — solid interior (inside the set)"
    if med_int < 0.005 and max(occ) < 0.05:
        return True, "near-zero interior & occupancy — dust (disconnected set)"
    return False, ""


BAND_AXES = ["mid_detail_frac", "interior_frac", "flat_frac"]
TARGETS = {"mid_detail_frac": TGT_MID, "interior_frac": TGT_INT, "flat_frac": TGT_FLAT}


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_metrics()
    groups = by_cid(recs)
    cs = build_cs()
    order = [c["cid"] for c in cs]

    per_c = {}
    degenerate = {}
    for cid in order:
        rows = groups.get(cid, [])
        if not rows:
            continue
        deg, why = is_degenerate(rows)
        if deg:
            degenerate[cid] = why
            continue
        best = max(rows, key=q4_score)
        bd = min(rows, key=band_dist)
        per_c[cid] = dict(
            cid=cid, kind=rows[0]["kind"], c_re=rows[0]["c_re"], c_im=rows[0]["c_im"],
            best_fid=best["fid"], best=best,
            min_band_dist=band_dist(bd), min_band_fid=bd["fid"],
            min_band_fw=bd["fw"], min_band_pan=bd["ip"],
            deep_frac_max=max(r["deep_frac"] for r in rows),
            n_framings=len(rows),
        )

    # ---- GENERALIZATION: at each c's composite-best, the axis values (drift) ----
    live = [per_c[cid] for cid in order if cid in per_c]
    gen = {}
    for ax in BAND_AXES + ["busy_near_black", "coherent_rest"]:
        vals = [p["best"][ax] for p in live]
        gen[ax] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                       min=float(np.min(vals)), max=float(np.max(vals)),
                       target=TARGETS.get(ax))
    # per-axis "generalizes" heuristic: composite-best clusters near target with low spread
    for ax in BAND_AXES:
        t = TARGETS[ax]
        g = gen[ax]
        # normalized drift: std relative to the target band scale
        scale = {"mid_detail_frac": S_MID, "interior_frac": S_INT, "flat_frac": S_FLAT}[ax]
        g["drift_norm"] = g["std"] / scale
        g["bias_norm"] = (g["mean"] - t) / scale
        g["generalizes"] = bool(g["drift_norm"] < 1.5 and abs(g["bias_norm"]) < 1.5)

    # ---- CORNER per c: does a target-band framing exist; where (fw/pan)? ----
    CORNER_THRESH = 1.5    # within 1.5 normalized band-units = target-band corner exists
    corner = {}
    for p in live:
        exists = p["min_band_dist"] <= CORNER_THRESH
        corner[p["cid"]] = dict(exists=bool(exists), min_band_dist=p["min_band_dist"],
                                fw=p["min_band_fw"], pan_ip=p["min_band_pan"])

    # ---- VARIANT B: deep_frac range across all c's ----
    all_deep = [r["deep_frac"] for r in recs]
    variant_b = dict(
        deep_frac_max=float(np.max(all_deep)),
        deep_frac_p99=float(np.percentile(all_deep, 99)),
        cs_over_0085=[p["cid"] for p in live if p["deep_frac_max"] > 0.085],
        per_c_deep_max={p["cid"]: p["deep_frac_max"] for p in live},
    )

    analysis = dict(
        n_cs=len(order), n_live=len(live), n_degenerate=len(degenerate),
        degenerate=degenerate,
        target=dict(mid_detail=TGT_MID, interior=TGT_INT, flat=TGT_FLAT),
        composite_weights=dict(W_BNB=W_BNB, W_CR=W_CR),
        generalization=gen,
        corner=corner,
        corner_thresh=CORNER_THRESH,
        n_corner_exists=sum(1 for v in corner.values() if v["exists"]),
        variant_b=variant_b,
        per_c_best={p["cid"]: dict(fid=p["best_fid"], fw=p["best"]["fw"],
                                   ip=p["best"]["ip"], q4=q4_score(p["best"]),
                                   band_dist=band_dist(p["best"]),
                                   **{a: p["best"][a] for a in BAND_AXES +
                                      ["busy_near_black", "coherent_rest", "symmetry"]})
                    for p in live},
    )
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # ---- plot: per-axis drift of the composite-best across c's ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(BAND_AXES):
        vals = [p["best"][ax] for p in live]
        labels = [p["cid"] for p in live]
        colors = ["red" if p["kind"] == "exemplar" else
                  ("tab:orange" if p["kind"] == "far" else "tab:blue") for p in live]
        axes[i].scatter(range(len(vals)), vals, c=colors, s=40)
        axes[i].axhline(TARGETS[ax], color="green", ls="--", label="target")
        axes[i].set_title(f"{ax}\n(composite-best per c; drift_norm={gen[ax]['drift_norm']:.2f})")
        axes[i].set_xticks(range(len(vals)))
        axes[i].set_xticklabels(labels, rotation=90, fontsize=6)
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(OUT / "generalization_drift.png", dpi=110)
    plt.close(fig)

    # ---- print summary ----
    print(f"\n{'='*70}\nq4 c-perturbation ANALYSIS\n{'='*70}")
    print(f"c's: {len(order)} total, {len(live)} live, {len(degenerate)} degenerate")
    for cid, why in degenerate.items():
        print(f"  DEGENERATE {cid}: {why}")
    print(f"\nGENERALIZATION (composite-best axis drift across c's):")
    for ax in BAND_AXES:
        g = gen[ax]
        print(f"  {ax:20s} target {g['target']:.2f}  mean {g['mean']:.3f} "
              f"std {g['std']:.3f}  drift_norm {g['drift_norm']:.2f}  "
              f"bias_norm {g['bias_norm']:+.2f}  -> "
              f"{'GENERALIZES' if g['generalizes'] else 'c-SPECIFIC'}")
    for ax in ["busy_near_black", "coherent_rest"]:
        g = gen[ax]
        print(f"  {ax:20s} mean {g['mean']:.3f} std {g['std']:.3f} "
              f"[{g['min']:.3f},{g['max']:.3f}]")
    print(f"\nCORNER exists (band_dist<={CORNER_THRESH}) at "
          f"{analysis['n_corner_exists']}/{len(live)} live c's")
    for cid, v in corner.items():
        flag = "YES" if v["exists"] else "no "
        print(f"  {flag} {cid:16s} min_band_dist {v['min_band_dist']:.2f} "
              f"@ fw {v['fw']:.3f} pan {v['pan_ip']}")
    print(f"\nVARIANT B: deep_frac max {variant_b['deep_frac_max']:.3f} "
          f"(p99 {variant_b['deep_frac_p99']:.3f}); "
          f"c's over 0.085: {variant_b['cs_over_0085'] or 'NONE'}")
    print(f"\nwrote analysis.json + generalization_drift.png")


# --------------------------------------------------------------------------- #
# Morph — motif variety via morph_gray + CLIP over per-c bests                 #
# --------------------------------------------------------------------------- #
def stage_morph():
    from tools.wallpaper.library_annotate import morph_gray_image
    from tools.curation.colored_clip import load_clip, embed_clip

    analysis = json.loads((OUT / "analysis.json").read_text())
    recs = load_metrics()
    idx = {(r["cid"], r["fid"]): r for r in recs}
    live_cids = list(analysis["per_c_best"].keys())

    mdir = OUT / "morph_fields"
    mdir.mkdir(parents=True, exist_ok=True)

    class _Field:
        def __init__(self, values, ss):
            self.values = values
            self.supersample = ss

    imgs, cids = [], []
    for cid in live_cids:
        pb = analysis["per_c_best"][cid]
        r = idx[(cid, pb["fid"])]
        b = mdir / f"m_{cid}.bin"
        if not b.exists():
            dump_field(r["c_re"], r["c_im"], r["cx"], r["cy"], r["fw"], b,
                       auto_maxiter(r["fw"]), MORPH_W, MORPH_H, MORPH_SS)
        vals = load_values(b)  # super-res, NaN interior
        img = morph_gray_image(_Field(vals, MORPH_SS))
        imgs.append(img)
        cids.append(cid)

    print(f"embedding {len(imgs)} per-c bests via morph_gray + CLIP ...")
    model, tf = load_clip()
    emb = embed_clip(model, tf, imgs)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = emb @ emb.T

    n = len(cids)
    off = sim[np.triu_indices(n, k=1)]
    # single-linkage clustering at NEAR_DUP -> distinct-look count
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
        clusters.setdefault(find(i), []).append(cids[i])
    distinct = len(clusters)
    dup_pairs = [(cids[i], cids[j], float(sim[i, j]))
                 for i in range(n) for j in range(i + 1, n) if sim[i, j] >= NEAR_DUP]

    morph = dict(
        n=n, cids=cids,
        median_offdiag=float(np.median(off)),
        mean_offdiag=float(np.mean(off)),
        max_offdiag=float(np.max(off)),
        near_dup_threshold=NEAR_DUP,
        median_yardstick=MORPH_MEDIAN_YARD,
        distinct_look_count=distinct,
        near_dup_pairs=dup_pairs,
        clusters=[sorted(v) for v in clusters.values()],
    )
    np.savez(OUT / "morph_sim.npz", sim=sim, cids=np.array(cids))
    (OUT / "morph.json").write_text(json.dumps(morph, indent=2))
    print(f"\nMOTIF VARIETY: {n} per-c bests -> {distinct} distinct looks "
          f"(single-linkage @ {NEAR_DUP})")
    print(f"  median off-diag cos {morph['median_offdiag']:.3f} "
          f"(yardstick {MORPH_MEDIAN_YARD}); max {morph['max_offdiag']:.3f}")
    if dup_pairs:
        print(f"  near-dup pairs (>= {NEAR_DUP}):")
        for a, b, s in sorted(dup_pairs, key=lambda x: -x[2]):
            print(f"    {s:.4f}  {a} <-> {b}")
    else:
        print("  no near-dup pairs — all per-c bests are distinct looks")


# --------------------------------------------------------------------------- #
# Sheets — colored judge-quality renders                                       #
# --------------------------------------------------------------------------- #
SHEET_W, SHEET_H, SHEET_SS = 1024, 576, 2


def render_color(r, out_png):
    cmd = [str(EXE), "render-one", "--julia", "--c", repr(r["c_re"]), repr(r["c_im"]),
           "--cx", repr(r["cx"]), "--cy", repr(r["cy"]), "--fw", repr(r["fw"]),
           "--family", "mandelbrot", "--maxiter", str(auto_maxiter(r["fw"])),
           "--width", str(SHEET_W), "--height", str(SHEET_H), "--supersample", str(SHEET_SS),
           "--palette", PALETTE, "--out", str(out_png)]
    rr = subprocess.run(cmd, capture_output=True, text=True)
    if rr.returncode != 0:
        raise RuntimeError(rr.stderr[-400:])


def _ensure_render(r):
    from PIL import Image  # noqa
    rend = OUT / "renders"
    rend.mkdir(exist_ok=True)
    p = rend / f"r_{r['cid']}_{r['fid']:03d}.png"
    if not p.exists():
        render_color(r, p)
    return p


def _cap(r, s=None):
    cap = (f"{r['cid']} fw{r['fw']:.2f} int{r['interior_frac']:.2f} "
           f"mid{r['mid_detail_frac']:.2f} flat{r['flat_frac']:.2f} "
           f"bnb{r['busy_near_black']:.2f} cr{r['coherent_rest']:.2f}")
    if s is not None:
        cap += f" q{s:+.2f}"
    return cap


def montage_grouped(rows_by_group, out_png, title, cols=4):
    """One labeled ROW per group (c), that c's top-K side by side."""
    from PIL import Image, ImageDraw
    tw, th = 340, 191
    row_h = th + 18
    grp_hdr = 16
    pad, top = 6, 30
    n_groups = len(rows_by_group)
    W = cols * tw + (cols + 1) * pad
    H = top + n_groups * (grp_hdr + row_h) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), title, fill=(235, 235, 235))
    y = top
    for gname, rows, scores in rows_by_group:
        d.text((pad, y), gname, fill=(180, 220, 255))
        y += grp_hdr
        for i, (r, s) in enumerate(zip(rows[:cols], scores[:cols])):
            p = _ensure_render(r)
            im = Image.open(p).resize((tw, th))
            x = pad + i * (tw + pad)
            canvas.paste(im, (x, y))
            d.text((x, y + th + 2), _cap(r, s), fill=(200, 200, 200), font=None)
        y += row_h
    canvas.save(out_png)
    print("wrote", out_png)


def montage_flat(rows, scores, out_png, title, cols=3):
    from PIL import Image, ImageDraw
    tw, th = 460, 259
    pad, top = 6, 30
    rowsn = (len(rows) + cols - 1) // cols
    W = cols * tw + (cols + 1) * pad
    H = top + rowsn * (th + 18) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), title, fill=(235, 235, 235))
    for i, (r, s) in enumerate(zip(rows, scores)):
        p = _ensure_render(r)
        im = Image.open(p).resize((tw, th))
        cx, cy = i % cols, i // cols
        x = pad + cx * (tw + pad)
        yy = top + cy * (th + 18)
        canvas.paste(im, (x, yy))
        d.text((x, yy + th + 2), _cap(r, s), fill=(200, 200, 200))
    canvas.save(out_png)
    print("wrote", out_png)


def stage_sheets(topk=4):
    recs = load_metrics()
    groups = by_cid(recs)
    analysis = json.loads((OUT / "analysis.json").read_text())
    cs = build_cs()
    order = [c["cid"] for c in cs]
    live = [cid for cid in order if cid in analysis["per_c_best"]]

    # exemplar reference
    ex_rows = groups["exemplar"]
    ex_best = max(ex_rows, key=q4_score)
    render_color(ex_best, OUT / "exemplar_large.png")
    print("wrote exemplar_large.png")

    # grouped-by-c sheets (primary): rings block + far block
    def group_rows(cid):
        rows = sorted(groups[cid], key=q4_score, reverse=True)[:topk]
        return (f"{cid}  c=({rows[0]['c_re']:+.3f},{rows[0]['c_im']:+.3f})",
                rows, [q4_score(r) for r in rows])

    ring_cids = [cid for cid in live if groups[cid][0]["kind"] in ("exemplar", "ring")]
    far_cids = [cid for cid in live if groups[cid][0]["kind"] == "far"]
    montage_grouped([group_rows(cid) for cid in ring_cids],
                    OUT / "sheet_by_c_rings.png",
                    "q4 c-perturbation — per-c top-K (exemplar + rings), best-first by composite q4",
                    cols=topk)
    if far_cids:
        montage_grouped([group_rows(cid) for cid in far_cids],
                        OUT / "sheet_by_c_far.png",
                        "q4 c-perturbation — per-c top-K (FARTHER c's near dM), best-first by composite q4",
                        cols=topk)

    # best-per-c motif-variety sheet (the CLIP-embedded set)
    best_rows = [max(groups[cid], key=q4_score) for cid in live]
    montage_flat(best_rows, [q4_score(r) for r in best_rows],
                 OUT / "sheet_best_per_c.png",
                 "q4 c-perturbation — THE composite-best per c (motif-variety set)", cols=4)

    # busy_near_black flag sheet (worst offenders across all framings)
    bnb_rows = sorted(recs, key=lambda r: r["busy_near_black"], reverse=True)[:9]
    montage_flat(bnb_rows, [r["busy_near_black"] for r in bnb_rows],
                 OUT / "sheet_busy_near_black.png",
                 "FLAG — highest busy_near_black (distracting near-lake speckle)", cols=3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "analyze", "morph", "sheets", "all"])
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("measure", "all"):
        stage_measure()
    if args.stage in ("analyze", "all"):
        stage_analyze()
    if args.stage in ("morph", "all"):
        stage_morph()
    if args.stage in ("sheets", "all"):
        stage_sheets(args.topk)
