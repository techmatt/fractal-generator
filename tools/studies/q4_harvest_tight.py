#!/usr/bin/env python
"""q4 stage-1 harvest (tight) — first real q4 candidates.

Run the harvest-ready G (refit T2 model) over EVERY minibrot, take masked
position-maxima, gate at a LABEL-derived high-precision cutoff, render survivors
vivid. This is the delivery — a tight, high-precision first look at auto-framed q4
locations. No new labels, no aimed batch — Matt's eyeball is the test.

Pipeline:
  1. Field over all 33 corpus minibrots — dense position×scale G, masked to v2
     pre-filter survivors (G extrapolates OOD on dead interior).
  2. Position-maxima — local maxima of the smoothed G per minibrot (position+scale),
     cross-scale deduped by IoU. Maxima of the NEW G, not NMS on score_A.
  3. Gate from labels, tight — G cutoff at the high-precision operating point measured
     on the 340 labeled windows (labeled precision ≥0.85). NOT p=0.5 (p is optimistically
     shifted). Cutoff on the deployment model's own G scale -> directly transferable.
  4. Render survivors at wallpaper quality in ONE vivid target palette (UF `default`);
     contact sheets ordered by G. Palette diversity is emission's job, later.

Run:  uv run python -m tools.studies.q4_harvest_tight build       # field+maxima+gate
      uv run python -m tools.studies.q4_harvest_tight render      # renders + sheets (resumable)
      uv run python -m tools.studies.q4_harvest_tight all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mpmath as mp  # noqa: E402
from tools.corpus import q4_window_reader as qr  # noqa: E402
from tools.studies import q4_stage1_labelset as LS  # noqa: E402
from tools.studies import q4_stage1_linear_fit as LF  # noqa: E402
from tools.studies import q4_stage1_refit as R  # noqa: E402
from tools.studies.q4_neighborhood_sweep import auto_maxiter  # noqa: E402

EXE = ROOT / "target" / "release" / "fractal-generator.exe"
OUT = ROOT / "out" / "q4_stage1" / "harvest_tight"
RENDERS = OUT / "renders"
BATCHES = ("2026-07-23_q4_stage1_windows", "2026-07-23_q4_g_aimed")

TIER, C = "T2_cells", 2.0
PEAK_SIZE = 5                     # local-maxima neighborhood (grid cells)
# The ornate ring is a CONTINUUM of high-G windows, so raw local maxima number in the
# hundreds per minibrot. Collapse to distinct, well-separated framings: elliptical
# center-separation NMS (a peak within SEP window-widths of a higher one is the SAME
# composition) + a per-minibrot cap. This is what makes the harvest "tight".
SEP = 1.0                        # min center separation, in window-widths (elliptical)
PER_MB_CAP = 4                   # distinct framings kept per minibrot (diversity)
TARGET_PREC = 0.85               # tight operating point
LOOSE_PREC = 0.78                # fallback if the tight gate is too sparse (<15)
MIN_N = 12                        # min labeled windows above cutoff for a stable estimate
WORKERS = 4

# wallpaper-quality render (one consistent vivid palette — composition test, not color)
RW, RH, RSS, PALETTE = 1920, 1080, 2, "default"
SHEET_COLS, SHEET_PER = 4, 24


# --------------------------------------------------------------------------- #
def mb_info():
    """{minibrot_id: render dict (cx/cy/fw/maxiter decimal-ish strings)}."""
    info = {}
    for b in BATCHES:
        for row, _ in qr.iter_windows(b):
            info.setdefault(row["minibrot_id"], row["render"])
    return info


def _all_peaks(G, size=PEAK_SIZE):
    Gf = np.where(np.isnan(G), -np.inf, G)
    mx = maximum_filter(Gf, size=size, mode="nearest")
    ys, xs = np.where((Gf == mx) & np.isfinite(G))
    return [(int(y), int(x), float(G[y, x])) for y, x in zip(ys, xs)]


def harvest_minibrot(args):
    """One minibrot -> cross-scale-deduped position-maxima (top-level = picklable)."""
    mb_id, sc, clf, keys = args
    field, fw, fh = LS.load_field_values(mb_id)
    model = (sc, clf, keys)
    peaks = []
    for s in LF.FIELD_SCALES:
        res = LF.dense_grid(field, fw, fh, s, model)   # already v2-masked
        if res is None:
            continue
        gx, gy, G, (Wp, Hp) = res
        for (iy, ix, gv) in _all_peaks(G):
            peaks.append(dict(minibrot_id=mb_id, scale=s,
                              cu=float(gx[ix]), cv=float(gy[iy]),
                              wu=Wp / fw, wv=Hp / fh, G=gv))
    # elliptical center-separation NMS across ALL scales (merges same-region peaks and
    # the same spot seen at adjacent scales), then cap to the top PER_MB_CAP by G.
    peaks.sort(key=lambda c: -c["G"])
    kept = []
    for c in peaks:
        clash = False
        for k in kept:
            du = (c["cu"] - k["cu"]) / (0.5 * (c["wu"] + k["wu"]))
            dv = (c["cv"] - k["cv"]) / (0.5 * (c["wv"] + k["wv"]))
            if du * du + dv * dv < SEP * SEP:
                clash = True
                break
        if not clash:
            c["box"] = [c["cu"] - c["wu"] / 2, c["cv"] - c["wv"] / 2, c["wu"], c["wv"]]
            kept.append(c)
        if len(kept) >= PER_MB_CAP:
            break
    return kept


# --------------------------------------------------------------------------- #
def precision_curve(g, y):
    """Sorted by G desc -> arrays (cutoff, precision, recall, n_above)."""
    order = np.argsort(g)[::-1]
    gs, ys = g[order], y[order]
    tp = np.cumsum(ys)
    n = np.arange(1, len(ys) + 1)
    prec = tp / n
    rec = tp / max(int(y.sum()), 1)
    return gs, prec, rec, n


def pick_cutoff(g, y, target, min_n):
    """Lowest cutoff with precision>=target and n_above>=min_n (max recall at target)."""
    gs, prec, rec, n = precision_curve(g, y)
    ok = np.where((prec >= target) & (n >= min_n))[0]
    if len(ok) == 0:
        i = int(np.argmax(prec[n >= min_n])) if (n >= min_n).any() else int(np.argmax(prec))
        return dict(cutoff=float(gs[i]), precision=float(prec[i]), recall=float(rec[i]),
                    n_above=int(n[i]), reached=False)
    i = int(ok[-1])                                   # deepest into the ranking still >=target
    return dict(cutoff=float(gs[i]), precision=float(prec[i]), recall=float(rec[i]),
                n_above=int(n[i]), reached=True)


# --------------------------------------------------------------------------- #
def stage_build():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = R.load_labels()
    rows = R.build_dataset(labels)
    lab = [r for r in rows if r[3] in ("accept", "reject")]
    mbs = sorted({r[1] for r in lab})
    print(f"harvest over {len(mbs)} minibrots; {len(lab)} labeled windows")

    # deployment model (full-data) — candidate G AND label cutoff share this scale
    _, _, sc, clf = LF.surviving_weights(rows, TIER, C)
    keys = LF.FEATURES[TIER]
    acc_idx = list(clf.classes_).index(1)

    # label-derived cutoff on the deployment model's in-sample G (same scale as harvest)
    Xl = np.array([[r[2][k] for k in keys] for r in lab])
    yl = np.array([1 if r[3] == "accept" else 0 for r in lab])
    gl = clf.decision_function(sc.transform(Xl))
    tight = pick_cutoff(gl, yl, TARGET_PREC, MIN_N)
    loose = pick_cutoff(gl, yl, LOOSE_PREC, MIN_N)
    # honest cross-check: held-out (LOMO) precision at the SAME recall level
    y_ho, g_ho = LF.lomo_scores(rows, TIER, C)
    gs, prec, rec, n = precision_curve(g_ho, y_ho)
    j = int(np.argmin(np.abs(rec - tight["recall"])))
    tight["heldout_precision_at_recall"] = float(prec[j])
    print(f"\nlabel-derived cutoff (in-sample, deployment G scale):")
    print(f"  TIGHT  G>={tight['cutoff']:.3f}  precision={tight['precision']:.2f} "
          f"recall={tight['recall']:.2f}  ({tight['n_above']} labeled above; "
          f"held-out prec@same-recall={tight['heldout_precision_at_recall']:.2f})"
          f"{'' if tight['reached'] else '  [target UNREACHED - best shown]'}")
    print(f"  LOOSE  G>={loose['cutoff']:.3f}  precision={loose['precision']:.2f} "
          f"recall={loose['recall']:.2f}  ({loose['n_above']} labeled above)")

    # field + maxima over all minibrots (parallel)
    print(f"\nfield+maxima over {len(mbs)} minibrots ({WORKERS} workers)...")
    tasks = [(m, sc, clf, keys) for m in mbs]
    cands = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(harvest_minibrot, tasks)):
            cands.extend(res)
            print(f"  {mbs[i]}: {len(res)} maxima (total {len(cands)})", flush=True)

    # gate + absolute geometry for the survivors
    info = mb_info()
    mp.mp.dps = 40
    tight_c = [c for c in cands if c["G"] >= tight["cutoff"]]
    loose_c = [c for c in cands if c["G"] >= loose["cutoff"]]
    survivors = tight_c if len(tight_c) >= 15 else loose_c
    gate = "tight" if survivors is tight_c else "loose"
    for c in survivors:
        rd = info[c["minibrot_id"]]
        cx, cy, fw = mp.mpf(rd["cx"]), mp.mpf(rd["cy"]), mp.mpf(rd["fw"])
        fh = fw * mp.mpf(9) / mp.mpf(16)
        cxw = cx + (mp.mpf(c["cu"]) - mp.mpf("0.5")) * fw
        cyw = cy + (mp.mpf("0.5") - mp.mpf(c["cv"])) * fh
        fww = mp.mpf(c["scale"]) * fw          # window frame width (scale = w-fraction)
        c["cx_win"] = mp.nstr(cxw, 30)
        c["cy_win"] = mp.nstr(cyw, 30)
        c["fw_win"] = mp.nstr(fww, 18)
        c["maxiter"] = int(max(int(rd["maxiter"]), auto_maxiter(float(fww))))

    survivors.sort(key=lambda c: -c["G"])
    per_mb = defaultdict(int)
    for c in survivors:
        per_mb[c["minibrot_id"]] += 1

    print(f"\ncandidates: {len(tight_c)} tight / {len(loose_c)} loose  "
          f"-> using {gate} gate ({len(survivors)} rendered)")
    print(f"  over {len(per_mb)} minibrots; per-mb "
          f"{dict(sorted(per_mb.items(), key=lambda x:-x[1]))}")

    manifest = dict(
        n_labeled=len(lab), n_minibrots_covered=len(per_mb),
        gate_used=gate, tight=tight, loose=loose,
        n_tight=len(tight_c), n_loose=len(loose_c), n_maxima_total=len(cands),
        render=dict(width=RW, height=RH, ss=RSS, palette=PALETTE),
        per_minibrot=dict(sorted(per_mb.items())),
        candidates=survivors)
    (OUT / "candidates.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n-> {(OUT/'candidates.json').relative_to(ROOT)}")
    return manifest


# --------------------------------------------------------------------------- #
def _cand_id(c):
    return f"{c['minibrot_id']}_s{int(c['scale']*1000):03d}_g{c['G']:.2f}".replace(".", "p")


def render_one(c):
    png = RENDERS / f"{_cand_id(c)}.png"
    if png.exists():
        return png
    cmd = [str(EXE), "--center-re", c["cx_win"], "--center-im", c["cy_win"],
           "--frame-width", c["fw_win"], "--maxiter", str(c["maxiter"]),
           "--width", str(RW), "--height", str(RH), "--supersample", str(RSS),
           "--palette", PALETTE, "--output", str(png)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not png.exists():
        raise RuntimeError(f"render {_cand_id(c)} failed: {r.stderr[-300:]}")
    return png


def stage_render():
    from PIL import Image, ImageDraw
    man = json.loads((OUT / "candidates.json").read_text())
    cands = man["candidates"]
    RENDERS.mkdir(parents=True, exist_ok=True)
    print(f"rendering {len(cands)} candidates @ {RW}x{RH} ss{RSS} ({PALETTE})...")
    for i, c in enumerate(cands):
        render_one(c)
        if (i + 1) % 5 == 0 or i + 1 == len(cands):
            print(f"  {i+1}/{len(cands)}", flush=True)

    # contact sheets, ordered by G
    tw, th = 480, 270
    for pg in range((len(cands) + SHEET_PER - 1) // SHEET_PER):
        chunk = cands[pg * SHEET_PER:(pg + 1) * SHEET_PER]
        rows_ = (len(chunk) + SHEET_COLS - 1) // SHEET_COLS
        pad, top = 6, 30
        Wc = SHEET_COLS * tw + (SHEET_COLS + 1) * pad
        Hc = top + rows_ * (th + 20) + pad
        canvas = Image.new("RGB", (Wc, Hc), (16, 16, 20))
        d = ImageDraw.Draw(canvas)
        d.text((pad, 8), f"q4 tight harvest — {man['gate_used']} gate "
               f"(G>={man[man['gate_used']]['cutoff']:.2f}, "
               f"labeled prec {man[man['gate_used']]['precision']:.2f}) — "
               f"page {pg+1}, ordered by G", fill=(235, 235, 235))
        for k, c in enumerate(chunk):
            im = Image.open(RENDERS / f"{_cand_id(c)}.png").resize((tw, th))
            cx_, cy_ = k % SHEET_COLS, k // SHEET_COLS
            x = pad + cx_ * (tw + pad)
            y = top + cy_ * (th + 20)
            canvas.paste(im, (x, y))
            d.text((x + 2, y + th + 3),
                   f"{c['minibrot_id']} s{c['scale']} G={c['G']:.2f}", fill=(200, 200, 200))
        out = OUT / f"sheet_{pg+1:02d}.png"
        canvas.save(out)
        print(f"  wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all", choices=["build", "render", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("build", "all"):
        stage_build()
    if args.stage in ("render", "all"):
        stage_render()
