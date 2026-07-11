"""Beam-equivalence check (flag-and-signal-read.md step 3, niche 2).

Dump the exp_smoothing field and the smooth field at a few DIVERSE pilot locations
(filigree/exterior anchor + high-interior anchor + across families), then compute
Spearman rank-corr over the escaped (finite-in-both) pixels. If rank-corr ~= 1 the
beam's gamma/transfer/n_cycles freedom makes exp and smooth beam-equivalent -> the
per-mode-head niche is null.

    uv run python -u tools/render_mode_pilot/exp_vs_smooth_rankcorr.py
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))
EXE = str(REPO / "target/release/fractal-generator.exe")
BATCH = REPO / "data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1"
OUT = REPO / "out/render_mode_pilot/exp_vs_smooth"
OUT.mkdir(parents=True, exist_ok=True)
W, H, SS = 640, 360, 2

import colormap as cm
import location as loc_mod


def locflags(loc):
    return loc_mod.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
            "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def dump(loc, field, tag):
    binp = OUT / f"{tag}_{field}.bin"
    env = dict(os.environ, RAYON_NUM_THREADS="4")
    r = subprocess.run([EXE, "render-one"] + locflags(loc) +
        ["--width", str(W), "--height", str(H), "--supersample", str(SS),
         "--coloring", json.dumps({"field": field}), "--dump-field", str(binp)],
        cwd=str(REPO), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-600:])
    fld = cm.load_field(str(binp))
    return fld.values.astype(np.float64)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb)))


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    rows = [json.loads(l) for l in (BATCH / "images.jsonl").read_text().splitlines() if l.strip()]

    # pick diverse anchors: one per family, prefer distinct location_keys; then we
    # report interior fraction so filigree/exterior vs interior-heavy is visible.
    byfam, seen = {}, set()
    for r in rows:
        f = r["provenance"]["family"]; lk = r["provenance"]["location_key"]
        if lk in seen: continue
        byfam.setdefault(f, r); seen.add(lk)
    anchors = list(byfam.values())
    # add the smoothest exp_smoothing raster's location (the flagged filigree case),
    # plus a couple extra mandelbrot/julia to widen interior-fraction coverage.
    exp_locs = [r for r in rows if r["render"]["render_mode"] == "exp_smoothing"]
    anchors.append(exp_locs[0])
    # dedup by location_key
    uniq = {}
    for r in anchors:
        uniq.setdefault(r["provenance"]["location_key"], r)
    anchors = list(uniq.values())

    print(f"{'tag':26s} {'family':16s} {'interior%':>9} {'esc_px':>8} "
          f"{'spearman':>9} {'pearson':>8}")
    results = []
    for i, r in enumerate(anchors):
        loc = loc_mod.from_render_block(r["render"])
        fam = r["provenance"]["family"]
        tag = f"a{i:02d}_{fam}"
        try:
            sm = dump(loc, "smooth", tag)
            ex = dump(loc, "exp_smoothing", tag)
        except Exception as e:
            print(f"{tag:26s} ERR {str(e)[:80]}"); continue
        m = np.isfinite(sm) & np.isfinite(ex)
        interior = 1.0 - m.mean()
        a, b = sm[m], ex[m]
        if a.size < 100 or a.std() == 0 or b.std() == 0:
            print(f"{tag:26s} {fam:16s} {interior:8.1%} {a.size:8d}   degenerate")
            continue
        sp = spearman(a, b)
        pe = float(np.corrcoef(a, b)[0, 1])
        results.append((tag, sp, pe))
        print(f"{tag:26s} {fam:16s} {interior:8.1%} {a.size:8d} {sp:9.6f} {pe:8.4f}")

    if results:
        sps = [x[1] for x in results]
        print(f"\nspearman: min={min(sps):.6f}  median={sorted(sps)[len(sps)//2]:.6f}  "
              f"max={max(sps):.6f}  (n={len(sps)} anchors)")
        worst = min(results, key=lambda x: x[1])
        print(f"worst anchor: {worst[0]}  spearman={worst[1]:.6f}")


if __name__ == "__main__":
    main()
