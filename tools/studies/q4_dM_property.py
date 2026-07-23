#!/usr/bin/env python
"""q4 ∂M-property study — which LOCAL MANDELBROT property at c predicts good Julia c's?

Measurement pass (NO descent / config / data/ changes). See prompts/q4_dM_property.md.
Prototypes the campaign-3 c-selection screen. The c-perturbation pass
(q4_c_perturbation.py) proved the exemplar's conjunction (interior lakes + busy
mid-detail + composed rest) is essentially c-UNIQUE within fw∈[0.13,1.5], and that the
generalizing lever is c-SELECTION, not framing. This pass tests which computable
*∂M-local* property at c predicts the conjunction — so julia parent c's can be SCREENED.

Two load-bearing fixes over the last pass:
  * DEEP ZOOM   — julia sweep fw down to 0.03 (the look intensifies below the old 0.13
                  floor; the prior "c-unique" verdict was depth-confounded).
  * c ALONG ∂M  — c pool is boundary-screened: anchors projected onto ∂M across varied
                  local structure (smooth arcs / seahorse / elephant / dendrite / cusps),
                  each a signed-normal LADDER so dist_dM has real spread.

Hypotheses:
  H1  exemplar-like J_c needs c in a thin shell JUST INSIDE ∂M (small |dist_dM|): deep-in
      -> solid blob, outside -> dust.
  H2  among near-∂M c's, local-M-richness (filamentary detail of ∂M around c) predicts
      J_c busy-detail quality.
  H3  (the correspondence) local-M-richness at c predicts overall J_c quality -> julia
      parents can be screened with MANDELBROT scoring.

Per-c ∂M properties (cheap, once per c):
  dist_dM      signed distance to ∂M (ring-probe: - inside, + outside), + exterior DE x-check.
  M_richness   occupancy / mid_detail on the local MANDELBROT escape field around c.
Per-c Julia measurement: center-descent deep sweep fw∈[0.03,1.5] (deep-weighted) + small
pan, f64 field-dump each; the calibrated axes (interior_frac, mid_detail_frac, coherent_rest,
busy_near_black, deep_frac). Per-c J-quality = best framing's distance to the exemplar band.

Reuses q4_c_perturbation.compute_metrics/band_dist (the two-new-axis metrics + the
calibrated target band) and q4_neighborhood_sweep.{auto_maxiter,load_values,two_scale}, plus
the library morph_gray + colored_clip recipe. Field source is f64 (colormap-invariant);
colored sheets use render-one with the exemplar palette. All output under out/q4_dM/
(disposable); field bins purged per-unit.

Stages (idempotent, resume from checkpoint):
  pool      build boundary-screened c ladders; per-c dist_dM + M_richness -> pool.json
  measure   julia deep-sweep field-dumps + calibrated metrics -> metrics.jsonl
  analyze   PREDICTION (dist_dM/M_richness vs J-quality) + screen rule + conjunction@depth
  morph     morph_gray + CLIP over per-c deep bests -> morph.json (needs torch/timm/GPU)
  sheets    colored judge renders, per-c deep best, annotated dist_dM/M_richness -> *.png
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

from tools.studies.q4_neighborhood_sweep import (  # noqa: E402
    auto_maxiter, load_values,
)
from tools.studies.q4_c_perturbation import (  # noqa: E402
    compute_metrics, band_dist, TGT_MID, TGT_INT, TGT_FLAT, S_MID, S_INT, S_FLAT,
)

OUT = ROOT / "out" / "q4_dM"
FIELDS = OUT / "fields"
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
PALETTE = "twilight_shifted"

# Exemplar (center-view Julia, origin-centered) — the conjunction anchor.
EX_C = (0.26103, -0.48932)

# --- measurement geometry --------------------------------------------------
MEAS_W, MEAS_H = 768, 432            # ss1 julia field-dump res (colormap-invariant)
M_W, M_H = 512, 512                  # square mandelbrot window for M_richness
FW_M = 0.10                          # mandelbrot window width around c (local ∂M structure)
M_MAXITER = 2000                     # richer maxiter for local ∂M filament resolution

# --- julia deep sweep ------------------------------------------------------
N_FW = 8                             # log fw in [0.03, 1.5] (deep-weighted by geomspace)
FW_LO, FW_HI = 0.03, 1.5
PAN_FRAC = 0.15

# --- boundary probe --------------------------------------------------------
PROBE_MAXITER = 2000                 # membership decision near ∂M
MEMBER_MAXITER = 4000                # per-c inside/outside check
R_MAX = 0.30                         # ring-probe search cap (clamp for deep in/out)
N_ANGLE = 24

# --- morph geometry (MUST match library morph-canon: colored_clip W,H,SS) ---
MORPH_W, MORPH_H, MORPH_SS = 640, 360, 2
NEAR_DUP = 0.974
MORPH_MEDIAN_YARD = 0.851

# --- "good c" / corner threshold (same as prior pass -> directly comparable) -
CORNER_THRESH = 1.5                  # min band_dist <= this => exemplar-conjunction reached


# --------------------------------------------------------------------------- #
# Mandelbrot scalar/vector helpers (pure numpy — cheap, no exe)                #
# --------------------------------------------------------------------------- #
def mandel_inside(cre, cim, maxiter):
    """Vectorized membership: True where z→z²+c does NOT escape |z|>2 within maxiter."""
    c = np.asarray(cre, dtype=np.complex128) + 1j * np.asarray(cim, dtype=np.complex128)
    z = np.zeros_like(c)
    escaped = np.zeros(c.shape, dtype=bool)
    for _ in range(maxiter):
        z = np.where(escaped, 0.0, z * z + c)          # freeze escaped -> no overflow spam
        escaped |= z.real * z.real + z.imag * z.imag > 4.0
        if escaped.all():
            break
    return ~escaped


def mandel_de(cre, cim, maxiter, bail=1e6):
    """Exterior distance estimate |z|·ln|z|/|z'| (nan if it doesn't escape)."""
    c = complex(cre, cim)
    z = 0j
    dz = 0j
    for _ in range(maxiter):
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        if abs(z) > bail:
            az = abs(z)
            return float(az * math.log(az) / (abs(dz) + 1e-300))
    return float("nan")


def signed_dist_dM(cre, cim, maxiter=PROBE_MAXITER, r_max=R_MAX, n_ang=N_ANGLE):
    """Signed distance to ∂M via ring-probe. NEGATIVE inside M, POSITIVE outside.

    m0 = membership of c. Scan geometric radii; at the smallest radius where ANY ring
    sample flips membership, bisect c->sample to locate the crossing. |dist| = crossing
    radius; sign from m0. Clamp to ±r_max when no flip is found (deep interior / far dust).
    """
    m0 = bool(mandel_inside(np.array([cre]), np.array([cim]), MEMBER_MAXITER)[0])
    ang = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    radii = np.geomspace(5e-4, r_max, 28)
    for r in radii:
        pre = cre + r * ca
        pim = cim + r * sa
        mem = mandel_inside(pre, pim, maxiter)
        flip = np.where(mem != m0)[0]
        if flip.size:
            # bisect along c -> nearest flipped sample for a crossing radius
            k = flip[0]
            a = np.array([cre, cim])
            b = np.array([pre[k], pim[k]])
            for _ in range(38):
                mid = 0.5 * (a + b)
                mm = bool(mandel_inside(np.array([mid[0]]), np.array([mid[1]]), maxiter)[0])
                if mm == m0:
                    a = mid
                else:
                    b = mid
            dist = float(np.hypot(*(0.5 * (a + b) - np.array([cre, cim]))))
            return (-dist if m0 else dist), m0
    return ((-r_max) if m0 else r_max), m0


def project_to_boundary(cre, cim, maxiter=PROBE_MAXITER, r_max=R_MAX, n_ang=N_ANGLE):
    """Return (b_re, b_im, nout_re, nout_im, m0) — nearest ∂M point to the anchor and the
    OUTWARD unit normal (toward escaping side). None if no boundary within r_max."""
    m0 = bool(mandel_inside(np.array([cre]), np.array([cim]), MEMBER_MAXITER)[0])
    ang = np.linspace(0.0, 2.0 * math.pi, n_ang, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    for r in np.geomspace(5e-4, r_max, 28):
        pre = cre + r * ca
        pim = cim + r * sa
        mem = mandel_inside(pre, pim, maxiter)
        flip = np.where(mem != m0)[0]
        if flip.size:
            k = flip[0]
            a = np.array([cre, cim]); b = np.array([pre[k], pim[k]])
            for _ in range(40):
                mid = 0.5 * (a + b)
                mm = bool(mandel_inside(np.array([mid[0]]), np.array([mid[1]]), maxiter)[0])
                if mm == m0:
                    a = mid
                else:
                    b = mid
            bnd = 0.5 * (a + b)
            # outward = toward the escaping side. If m0 inside, the flipped sample is
            # outside -> outward ≈ (flip - anchor). If m0 outside, invert.
            d = np.array([pre[k], pim[k]]) - np.array([cre, cim])
            d = d / (np.hypot(*d) + 1e-30)
            nout = d if m0 else -d
            return float(bnd[0]), float(bnd[1]), float(nout[0]), float(nout[1]), m0
    return None


# --------------------------------------------------------------------------- #
# c pool — anchors projected to ∂M, signed-normal ladders                      #
# --------------------------------------------------------------------------- #
# Anchor seeds spanning varied ∂M structure. Projection lands the exact boundary point;
# the label is descriptive. Structure classes: filamentary valleys (seahorse/elephant),
# smooth arcs (cardioid/period-2 disk), near cusps/roots (period-3 bulb), dendrite tips.
ANCHORS = [
    ("exemplar_reg", 0.261, -0.489),   # the exemplar's own ∂M location (lower shoulder)
    ("seahorse",    -0.745, 0.113),    # seahorse valley — filamentary
    ("elephant",     0.283, 0.012),    # elephant valley — filamentary
    ("p2disk_top",  -1.000, 0.290),    # period-2 disk — smooth circular arc
    ("p3bulb_root", -0.122, 0.649),    # period-3 (rabbit) bulb, near cusp
    ("cardioid_rt",  0.378, 0.150),    # right shoulder of the main cardioid — smoothish
    ("dendrite_up", -0.100, 0.940),    # thin dendrite / filament tip
    ("card_lower",   0.360, -0.300),   # lower cardioid shoulder (near exemplar region, distinct)
]
# signed-normal offsets (c-plane units): + outside (dust), - inside. Spans the dist_dM axis.
LADDER_OFFSETS = [0.020, -0.006, -0.030, -0.080]


def build_pool():
    """Project each anchor to ∂M, build its normal ladder, measure per-c ∂M properties +
    local M_richness. Writes pool.json. Cheap (numpy probes + 1 mandelbrot dump per c)."""
    FIELDS.mkdir(parents=True, exist_ok=True)
    pool = []
    # explicit exemplar c (its literal position, whatever its measured dist_dM)
    seeds = [("exemplar", EX_C[0], EX_C[1], None)]
    for name, are, aim in ANCHORS:
        proj = project_to_boundary(are, aim)
        if proj is None:
            print(f"  SKIP anchor {name} ({are},{aim}) — no ∂M within r_max")
            continue
        bre, bim, nre, nim, m0 = proj
        print(f"  anchor {name:12s} -> dM ({bre:+.5f},{bim:+.5f}) nout=({nre:+.3f},{nim:+.3f})")
        for s in LADDER_OFFSETS:
            cre = bre + s * nre
            cim = bim + s * nim
            tag = f"{name}_s{s:+.3f}".replace("+", "p").replace("-", "m").replace(".", "")
            seeds.append((tag, cre, cim, dict(region=name, offset=s,
                                              b_re=bre, b_im=bim)))
    for cid, cre, cim, prov in seeds:
        dist, m0 = signed_dist_dM(cre, cim)
        de = mandel_de(cre, cim, M_MAXITER)  # finite iff outside
        # local M_richness: mandelbrot escape field in a FW_M window centered on c
        b = FIELDS / f"M_{cid}.bin"
        dump_mandel_field(cre, cim, FW_M, b, M_MAXITER)
        mvals = load_values(b)
        mm = compute_metrics(mvals)
        purge(b)
        rec = dict(cid=cid, c_re=cre, c_im=cim,
                   dist_dM=dist, c_inside=m0, exterior_de=de,
                   M_occupancy=mm["occupancy"], M_mid_detail=mm["mid_detail_frac"],
                   M_interior_frac=mm["interior_frac"], M_high_struct=mm["high_struct_frac"])
        if prov:
            rec["prov"] = prov
        pool.append(rec)
        print(f"    {cid:22s} dist_dM {dist:+.4f} {'IN ' if m0 else 'OUT'} "
              f"M_occ {mm['occupancy']:.3f} M_mid {mm['mid_detail_frac']:.3f} "
              f"M_int {mm['interior_frac']:.3f}")
    (OUT / "pool.json").write_text(json.dumps(pool, indent=2))
    print(f"pool: {len(pool)} c's -> pool.json")


# --------------------------------------------------------------------------- #
# field dumps                                                                  #
# --------------------------------------------------------------------------- #
def purge(b: Path):
    for p in (b, Path(str(b).replace(".bin", ".json"))):
        try:
            p.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows transient exe/file lock — purge is best-effort


def dump_mandel_field(cx, cy, fw, out_bin, maxiter, w=M_W, h=M_H):
    """Plain MANDELBROT parameter-plane field-dump (no --julia)."""
    cmd = [str(EXE), "render-one", "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(maxiter),
           "--width", str(w), "--height", str(h), "--supersample", "1",
           "--dump-field", str(out_bin), "--dump-field-source", "f64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"mandel dump-field failed: {r.stderr[-400:]}")


def dump_julia_field(c_re, c_im, cx, cy, fw, out_bin, maxiter, w=MEAS_W, h=MEAS_H, ss=1):
    cmd = [str(EXE), "render-one", "--julia", "--c", repr(c_re), repr(c_im),
           "--cx", repr(cx), "--cy", repr(cy), "--fw", repr(fw),
           "--family", "mandelbrot", "--maxiter", str(maxiter),
           "--width", str(w), "--height", str(h), "--supersample", str(ss),
           "--dump-field", str(out_bin), "--dump-field-source", "f64"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"julia dump-field failed: {r.stderr[-400:]}")


def build_framings():
    """Per-c center-descent DEEP sweep: fw log-spaced [0.03,1.5] x {center + 4 small pans}.
    geomspace weights the deep end (equal log density -> most rungs below 0.13)."""
    fws = np.geomspace(FW_LO, FW_HI, N_FW)
    pans = [(0.0, 0.0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    fr, fid = [], 0
    for iz, fw in enumerate(fws):
        for ip, (sx, sy) in enumerate(pans):
            fr.append(dict(fid=fid, iz=iz, ip=ip, fw=float(fw),
                           dcx=float(sx * PAN_FRAC * fw), dcy=float(sy * PAN_FRAC * fw)))
            fid += 1
    return fr


# --------------------------------------------------------------------------- #
# measure — julia deep sweep                                                   #
# --------------------------------------------------------------------------- #
def stage_measure():
    FIELDS.mkdir(parents=True, exist_ok=True)
    pool = json.loads((OUT / "pool.json").read_text())
    frs = build_framings()
    total = len(pool) * len(frs)
    ckpt = OUT / "metrics.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["cid"], d["fid"]))
    print(f"grid: {len(pool)} c x {len(frs)} framings = {total}, {len(done)} done")
    t0 = time.time()
    n_new = 0
    with ckpt.open("a") as f:
        for c in pool:
            for fr in frs:
                key = (c["cid"], fr["fid"])
                if key in done:
                    continue
                mi = auto_maxiter(fr["fw"])
                b = FIELDS / f"J_{c['cid']}_{fr['fid']:03d}.bin"
                dump_julia_field(c["c_re"], c["c_im"], fr["dcx"], fr["dcy"], fr["fw"], b, mi)
                m = compute_metrics(load_values(b))
                rec = {"cid": c["cid"], "c_re": c["c_re"], "c_im": c["c_im"],
                       "fid": fr["fid"], "iz": fr["iz"], "ip": fr["ip"],
                       "fw": fr["fw"], "cx": fr["dcx"], "cy": fr["dcy"], "maxiter": mi, **m}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                purge(b)
                n_new += 1
                if n_new == 20:
                    el = time.time() - t0
                    proj = el / 20 * (total - len(done))
                    print(f"  [timing] first 20: {el:.1f}s -> projected {proj:.0f}s "
                          f"for {total - len(done)} remaining", flush=True)
                if n_new % 50 == 0:
                    print(f"  {n_new} new ({time.time()-t0:.1f}s)", flush=True)
    print(f"measure done: {n_new} new in {time.time()-t0:.1f}s -> {ckpt}")


def load_metrics():
    return [json.loads(l) for l in (OUT / "metrics.jsonl").read_text().splitlines() if l.strip()]


def by_cid(recs):
    d = {}
    for r in recs:
        d.setdefault(r["cid"], []).append(r)
    return d


def is_degenerate(rows):
    mids = [r["mid_detail_frac"] for r in rows]
    ints = [r["interior_frac"] for r in rows]
    occ = [r["occupancy"] for r in rows]
    med_int = float(np.median(ints))
    if max(mids) < 0.05:
        return True, "no mid-detail (max<0.05) — flat/dust"
    if med_int > 0.9:
        return True, "median interior>0.9 — solid blob (inside the set)"
    if med_int < 0.005 and max(occ) < 0.05:
        return True, "near-zero interior & occupancy — dust"
    return False, ""


# --------------------------------------------------------------------------- #
# analyze — PREDICTION / correspondence / conjunction@depth                    #
# --------------------------------------------------------------------------- #
def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return _pearson(rx, ry)


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pool = {p["cid"]: p for p in json.loads((OUT / "pool.json").read_text())}
    recs = load_metrics()
    groups = by_cid(recs)

    per_c = {}
    for cid, rows in groups.items():
        deg, why = is_degenerate(rows)
        bd_best = min(rows, key=band_dist)
        min_bd = band_dist(bd_best)
        j_quality = -min_bd                       # higher = closer to the exemplar band
        best = bd_best
        per_c[cid] = dict(
            cid=cid, degenerate=deg, why=why,
            min_band_dist=min_bd, j_quality=j_quality,
            best_fid=best["fid"], best_fw=best["fw"], best_ip=best["ip"],
            best_interior=best["interior_frac"], best_mid=best["mid_detail_frac"],
            best_flat=best["flat_frac"], best_bnb=best["busy_near_black"],
            best_cr=best["coherent_rest"],
            deep_frac_max=max(r["deep_frac"] for r in rows),
            corner=bool(min_bd <= CORNER_THRESH),
            # ∂M properties (from pool)
            dist_dM=pool[cid]["dist_dM"], c_inside=pool[cid]["c_inside"],
            M_occupancy=pool[cid]["M_occupancy"], M_mid_detail=pool[cid]["M_mid_detail"],
            M_interior_frac=pool[cid]["M_interior_frac"],
            region=pool[cid].get("prov", {}).get("region", "exemplar"),
        )

    cids = list(per_c.keys())
    P = [per_c[c] for c in cids]

    # ---- PREDICTION correlations (all c's; degenerate INCLUDED — they show the U-shape) ---
    Jq = [p["j_quality"] for p in P]
    dd = [p["dist_dM"] for p in P]
    abs_dd = [abs(p["dist_dM"]) for p in P]
    m_occ = [p["M_occupancy"] for p in P]
    m_mid = [p["M_mid_detail"] for p in P]

    corr = {
        "dist_dM_vs_Jq":       dict(pearson=_pearson(dd, Jq), spearman=_spearman(dd, Jq)),
        "abs_dist_dM_vs_Jq":   dict(pearson=_pearson(abs_dd, Jq), spearman=_spearman(abs_dd, Jq)),
        "M_occupancy_vs_Jq":   dict(pearson=_pearson(m_occ, Jq), spearman=_spearman(m_occ, Jq)),
        "M_mid_detail_vs_Jq":  dict(pearson=_pearson(m_mid, Jq), spearman=_spearman(m_mid, Jq)),
    }

    # Global vs within-viable: does M_richness FINELY rank toward the exemplar, or does
    # it merely separate viable near-∂M c's from degenerate blob/dust? Restrict to live.
    live = [p for p in P if not p["degenerate"]]
    lJq = [p["j_quality"] for p in live]
    corr_live = {
        "abs_dist_dM_vs_Jq":  dict(pearson=_pearson([abs(p["dist_dM"]) for p in live], lJq),
                                   spearman=_spearman([abs(p["dist_dM"]) for p in live], lJq)),
        "M_occupancy_vs_Jq":  dict(pearson=_pearson([p["M_occupancy"] for p in live], lJq),
                                   spearman=_spearman([p["M_occupancy"] for p in live], lJq)),
        "M_mid_detail_vs_Jq": dict(pearson=_pearson([p["M_mid_detail"] for p in live], lJq),
                                   spearman=_spearman([p["M_mid_detail"] for p in live], lJq)),
        "n_live": len(live),
    }
    # exemplar's rank on the richness axis (1 = highest) — does M_richness single it out?
    ex_rank = dict(
        M_mid_all=int(1 + sum(1 for p in P if p["M_mid_detail"] > per_c["exemplar"]["M_mid_detail"])),
        M_mid_live=int(1 + sum(1 for p in live if p["M_mid_detail"] > per_c["exemplar"]["M_mid_detail"])),
        M_occ_all=int(1 + sum(1 for p in P if p["M_occupancy"] > per_c["exemplar"]["M_occupancy"])),
        n_all=len(P), n_live=len(live),
    )

    # ---- SCREEN RULE: just-inside ∂M AND M_richness > X. Grid-search the best rule. ---
    good = np.array([p["corner"] for p in P])
    base_rate = float(good.mean())
    # straddle shell, NOT inside-only: the exemplar sits at dist_dM +0.0005 (marginally
    # OUTSIDE) — the sweet spot is a thin two-sided shell ON ∂M, sign not essential.
    absd = np.array(abs_dd)
    # candidate richness axis = the more predictive of occupancy / mid_detail
    rich_axis = "M_mid_detail" if abs(corr["M_mid_detail_vs_Jq"]["spearman"] or 0) >= \
        abs(corr["M_occupancy_vs_Jq"]["spearman"] or 0) else "M_occupancy"
    rich = np.array([p[rich_axis] for p in P])

    best_rule = None
    for shell in [0.008, 0.02, 0.035, 0.05, 0.08, 0.12]:
        for xr in np.percentile(rich, [30, 40, 50, 60, 70, 80]):
            sel = (absd <= shell) & (rich >= xr)
            if sel.sum() == 0:
                continue
            prec = float(good[sel].mean())
            rec = float(good[sel].sum() / max(1, good.sum()))
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            cand = dict(shell=float(shell), rich_axis=rich_axis, rich_thresh=float(xr),
                        n_selected=int(sel.sum()), precision=prec, recall=rec, f1=f1,
                        selected=[cids[i] for i in np.where(sel)[0]])
            if best_rule is None or cand["f1"] > best_rule["f1"] or \
               (cand["f1"] == best_rule["f1"] and cand["precision"] > best_rule["precision"]):
                best_rule = cand

    # single-axis lift refs
    def lift_tail(vals, hi=True, frac=0.4):
        v = np.array(vals)
        k = max(1, int(len(v) * frac))
        idx = np.argsort(v)[::-1][:k] if hi else np.argsort(v)[:k]
        return float(good[idx].mean()), base_rate

    straddle = absd <= 0.035
    lifts = {
        "straddle(|dist|<=0.035)": float(good[straddle].mean()) if straddle.any() else 0.0,
        f"{rich_axis}_high_tail": lift_tail(rich, hi=True)[0],
        "M_occupancy_high_tail": lift_tail(m_occ, hi=True)[0],
    }

    # ---- CONJUNCTION @ DEPTH: any NON-exemplar c reach band? (the prior pass's open Q) ---
    non_ex_corner = [p for p in P if p["cid"] != "exemplar" and not p["degenerate"]
                     and p["corner"]]
    conjunction_at_depth = dict(
        exemplar_min_band_dist=per_c["exemplar"]["min_band_dist"] if "exemplar" in per_c else None,
        n_nonexemplar_live=sum(1 for p in P if p["cid"] != "exemplar" and not p["degenerate"]),
        n_nonexemplar_corner=len(non_ex_corner),
        cs=[dict(cid=p["cid"], min_band_dist=p["min_band_dist"], best_fw=p["best_fw"],
                 dist_dM=p["dist_dM"], region=p["region"],
                 interior=p["best_interior"], mid=p["best_mid"], flat=p["best_flat"],
                 bnb=p["best_bnb"], cr=p["best_cr"]) for p in
            sorted(non_ex_corner, key=lambda x: x["min_band_dist"])],
        deepest_reach_nonexemplar=min(
            ((p["min_band_dist"], p["cid"]) for p in P if p["cid"] != "exemplar"
             and not p["degenerate"]), default=None),
    )

    analysis = dict(
        n_cs=len(P), base_rate_corner=base_rate,
        n_degenerate=sum(1 for p in P if p["degenerate"]),
        n_corner=int(good.sum()),
        correlations=corr, correlations_live=corr_live, exemplar_rank=ex_rank,
        rich_axis=rich_axis,
        screen_rule=best_rule, single_axis_lifts=lifts,
        conjunction_at_depth=conjunction_at_depth,
        per_c={p["cid"]: p for p in P},
    )
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # ---- PLOTS: dist_dM vs Jq, M_richness vs Jq ----
    def _scatter(ax, xs, ys, title, xlabel, logx=False):
        col = ["red" if p["cid"] == "exemplar" else
               ("gray" if p["degenerate"] else "tab:blue") for p in P]
        ax.scatter(xs, ys, c=col, s=45, edgecolors="k", linewidths=0.4)
        for p, x, y in zip(P, xs, ys):
            if p["cid"] == "exemplar" or (not p["degenerate"] and p["corner"]):
                ax.annotate(p["cid"][:10], (x, y), fontsize=6, alpha=0.8)
        ax.axhline(-CORNER_THRESH, color="green", ls="--", lw=1,
                   label=f"corner thr (band_dist {CORNER_THRESH})")
        ax.set_xlabel(xlabel); ax.set_ylabel("J-quality (= −min band_dist)")
        ax.set_title(title); ax.legend(fontsize=7)
        if logx:
            ax.set_xscale("log")

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
    _scatter(axs[0], dd, Jq,
             f"H1: dist_dM vs J-quality\n(ρ={corr['dist_dM_vs_Jq']['spearman']:.2f})",
             "signed dist_dM (− inside, + outside)")
    axs[0].axvline(0, color="k", lw=0.6, alpha=0.5)
    _scatter(axs[1], m_occ, Jq,
             f"H2/H3: M_occupancy vs J-quality\n(ρ={corr['M_occupancy_vs_Jq']['spearman']:.2f})",
             "local Mandelbrot occupancy at c")
    _scatter(axs[2], m_mid, Jq,
             f"H2/H3: M_mid_detail vs J-quality\n(ρ={corr['M_mid_detail_vs_Jq']['spearman']:.2f})",
             "local Mandelbrot mid_detail at c")
    fig.tight_layout()
    fig.savefig(OUT / "prediction.png", dpi=115)
    plt.close(fig)

    # ---- print summary ----
    print(f"\n{'='*72}\nq4 ∂M-property ANALYSIS\n{'='*72}")
    print(f"c's: {len(P)}  ({analysis['n_degenerate']} degenerate)  "
          f"corner(base) rate {base_rate:.2f}  n_corner {int(good.sum())}")
    print("\nPREDICTION correlations (ALL c's; Jq = −min band_dist):")
    for k, v in corr.items():
        print(f"  {k:24s} pearson {v['pearson']:+.3f}  spearman {v['spearman']:+.3f}")
    print(f"\nWITHIN-VIABLE correlations (live c's only, n={corr_live['n_live']}) — "
          f"does M_richness FINELY rank, or just separate viable from degenerate?")
    for k, v in corr_live.items():
        if k == "n_live":
            continue
        print(f"  {k:24s} pearson {v['pearson']:+.3f}  spearman {v['spearman']:+.3f}")
    print(f"exemplar M_richness rank: M_mid {ex_rank['M_mid_all']}/{ex_rank['n_all']} all, "
          f"{ex_rank['M_mid_live']}/{ex_rank['n_live']} live; "
          f"M_occ {ex_rank['M_occ_all']}/{ex_rank['n_all']} all "
          f"(1=highest -> does richness SINGLE OUT the exemplar?)")
    print(f"\nSCREEN RULE (just-inside ∂M AND {rich_axis} >= X):")
    if best_rule:
        print(f"  shell |dist_dM|<={best_rule['shell']}  {rich_axis}>={best_rule['rich_thresh']:.3f}"
              f"  -> n={best_rule['n_selected']}  precision {best_rule['precision']:.2f}"
              f"  recall {best_rule['recall']:.2f}  f1 {best_rule['f1']:.2f}")
        print(f"  selected: {best_rule['selected']}")
    else:
        print("  no rule selected any c")
    print("  single-axis refs:", {k: round(v, 2) for k, v in lifts.items()})
    cad = conjunction_at_depth
    print(f"\nCONJUNCTION @ DEPTH (fw->0.03): exemplar min band_dist "
          f"{cad['exemplar_min_band_dist']:.2f}")
    print(f"  non-exemplar live c's: {cad['n_nonexemplar_live']}  "
          f"reaching corner (<= {CORNER_THRESH}): {cad['n_nonexemplar_corner']}")
    for c in cad["cs"]:
        print(f"    {c['cid']:22s} band_dist {c['min_band_dist']:.2f} @ fw {c['best_fw']:.3f}"
              f"  dist_dM {c['dist_dM']:+.4f}  int{c['interior']:.2f} mid{c['mid']:.2f}")
    if cad["deepest_reach_nonexemplar"]:
        bd, cc = cad["deepest_reach_nonexemplar"]
        print(f"  closest non-exemplar approach: {cc} @ band_dist {bd:.2f}")
    print("\nwrote analysis.json + prediction.png")


# --------------------------------------------------------------------------- #
# morph — motif variety across per-c deep bests (good c's)                     #
# --------------------------------------------------------------------------- #
def _good_cids(analysis):
    return [cid for cid, p in analysis["per_c"].items()
            if not p["degenerate"] and p["corner"]]


def stage_morph():
    from tools.wallpaper.library_annotate import morph_gray_image
    from tools.curation.colored_clip import load_clip, embed_clip

    analysis = json.loads((OUT / "analysis.json").read_text())
    per_c = analysis["per_c"]
    # motif set = all VIABLE (non-degenerate) per-c deep bests. Only the exemplar is a
    # corner, so "distinct looks among good c's" degenerates to n=1; the useful variety
    # readout is across the viable near-∂M c's (matches the prior pass's live-bests set).
    cids = ["exemplar"] + sorted(
        [c for c in per_c if c != "exemplar" and not per_c[c]["degenerate"]],
        key=lambda c: -per_c[c]["j_quality"])
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
        # re-dump the per-c best framing at morph res
        cre, cim = _c_of(cid, analysis)
        if not b.exists():
            dump_julia_field(cre, cim, *_best_pan(cid, analysis), p["best_fw"], b,
                             auto_maxiter(p["best_fw"]), MORPH_W, MORPH_H, MORPH_SS)
        vals = load_values(b)
        imgs.append(morph_gray_image(_Field(vals, MORPH_SS)))
        kept.append(cid)

    print(f"embedding {len(imgs)} per-c deep bests via morph_gray + CLIP ...")
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
    morph = dict(n=n, cids=kept,
                 median_offdiag=float(np.median(off)), mean_offdiag=float(np.mean(off)),
                 max_offdiag=float(np.max(off)) if n > 1 else float("nan"),
                 near_dup_threshold=NEAR_DUP, median_yardstick=MORPH_MEDIAN_YARD,
                 distinct_look_count=len(clusters), near_dup_pairs=dup_pairs,
                 clusters=[sorted(v) for v in clusters.values()])
    np.savez(OUT / "morph_sim.npz", sim=sim, cids=np.array(kept))
    (OUT / "morph.json").write_text(json.dumps(morph, indent=2))
    print(f"\nMOTIF VARIETY: {n} deep bests -> {len(clusters)} distinct looks "
          f"(single-linkage @ {NEAR_DUP})")
    print(f"  median off-diag cos {morph['median_offdiag']:.3f} "
          f"(yardstick {MORPH_MEDIAN_YARD}); max {morph['max_offdiag']:.3f}")
    for a, b, s in sorted(dup_pairs, key=lambda x: -x[2]):
        print(f"    near-dup {s:.4f}  {a} <-> {b}")


def _c_of(cid, analysis):
    recs = load_metrics()
    for r in recs:
        if r["cid"] == cid:
            return r["c_re"], r["c_im"]
    raise KeyError(cid)


def _best_pan(cid, analysis):
    recs = load_metrics()
    fid = analysis["per_c"][cid]["best_fid"]
    for r in recs:
        if r["cid"] == cid and r["fid"] == fid:
            return r["cx"], r["cy"]
    return 0.0, 0.0


# --------------------------------------------------------------------------- #
# sheets — colored judge renders, per-c deep best, annotated dist_dM / M_rich  #
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
    recs = load_metrics()
    idx = {(r["cid"], r["fid"]): r for r in recs}
    rend = OUT / "renders"
    rend.mkdir(exist_ok=True)

    # exemplar reference
    ex = per_c["exemplar"]
    exr = idx[("exemplar", ex["best_fid"])]
    render_color(exr["c_re"], exr["c_im"], exr["cx"], exr["cy"], exr["fw"],
                 OUT / "exemplar_large.png")

    # order: exemplar first, then non-degenerate by J-quality desc
    order = ["exemplar"] + sorted(
        [c for c in per_c if c != "exemplar" and not per_c[c]["degenerate"]],
        key=lambda c: -per_c[c]["j_quality"])
    # include a few degenerate extremes at the end for the U-shape story
    degs = sorted([c for c in per_c if per_c[c]["degenerate"]],
                  key=lambda c: per_c[c]["dist_dM"])[:4]
    order += degs

    def cap(cid):
        p = per_c[cid]
        return (f"{cid[:16]} bd{p['min_band_dist']:.2f} fw{p['best_fw']:.3f} "
                f"dM{p['dist_dM']:+.3f} Mocc{p['M_occupancy']:.2f} Mmid{p['M_mid_detail']:.2f} "
                f"{'DEG' if p['degenerate'] else ('*CORNER' if p['corner'] else '')}")

    tw, th = 380, 214
    cols = 4
    rows = (len(order) + cols - 1) // cols
    pad, top = 6, 34
    W = cols * tw + (cols + 1) * pad
    H = top + rows * (th + 22) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), "q4 ∂M-property — per-c DEEP best (band-nearest framing), "
                     "annotated dist_dM / M_richness. red=exemplar *=corner DEG=degenerate",
           fill=(235, 235, 235))
    for i, cid in enumerate(order):
        p = per_c[cid]
        r = idx[(cid, p["best_fid"])]
        pp = rend / f"r_{cid}.png"
        if not pp.exists():
            render_color(r["c_re"], r["c_im"], r["cx"], r["cy"], r["fw"], pp)
        im = Image.open(pp).resize((tw, th))
        cx, cy = i % cols, i // cols
        x = pad + cx * (tw + pad); y = top + cy * (th + 22)
        canvas.paste(im, (x, y))
        color = (255, 140, 140) if cid == "exemplar" else \
            ((150, 150, 150) if p["degenerate"] else
             ((150, 255, 150) if p["corner"] else (205, 205, 205)))
        d.text((x, y + th + 4), cap(cid), fill=color)
    canvas.save(OUT / "sheet_best_per_c.png")
    print("wrote sheet_best_per_c.png + exemplar_large.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pool", "measure", "analyze", "morph", "sheets", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("pool", "all"):
        build_pool()
    if args.stage in ("measure", "all"):
        stage_measure()
    if args.stage in ("analyze", "all"):
        stage_analyze()
    if args.stage in ("morph", "all"):
        stage_morph()
    if args.stage in ("sheets", "all"):
        stage_sheets()
