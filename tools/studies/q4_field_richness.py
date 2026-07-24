"""Palette-invariant field-richness ranking for the 30 q4 stage-1 minibrots.

Measures multi-scale, non-dead decoration density directly on the dumped SMOOTH
fields (out/q4_stage1/fields/*.bin) at the 4x-size "money-shot" framing they were
dumped at. Nothing is fit; the stat is a fixed transform reported for the eye.

Stat (per center):
  - F = smooth-iter field (f32, NaN = interior lake). L = log(F), NaN->median(L).
    log because fractal escape structure is log-periodic; compresses the near-
    boundary spike so decoration at every depth contributes equally.
  - range = p99(L) - p1(L) over escaped pixels (robust dynamic range; the
    normalizer that makes the stat invariant to smooth-iter amplitude / palette).
  - DoG band-pass at 4 octaves: band_k = G(L, s_k) - G(L, 2*s_k), s in {1,2,4,8}.
    |band_k|/range is dimensionless local relief at scale k.
  - occupancy_k  = fraction of frame with |band_k|/range > TAU   (decoration AREA)
  - energy_k     = mean over frame of |band_k|/range              (threshold-free)
  R_occ  = mean_k occupancy_k   (primary: "non-dead decoration density")
  R_energy = mean_k energy_k    (cross-check; robustness vs TAU choice)

Interior lake (NaN, filled to median) reads as flat -> contributes ~0 structure,
so mostly-black frames rank low, which is the intent.
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ROOT / "out" / "q4_stage1" / "fields"
MINIS = ROOT / "out" / "q4_stage1" / "minibrots.json"
OUT = ROOT / "out" / "fair_rerender"
TAU = 0.01                      # fixed, NOT fit
SIGMAS = [1.0, 2.0, 4.0, 8.0]   # octave band-pass scales


def load_field(stem):
    meta = json.load(open(FIELDS / f"{stem}.json"))
    w, h = meta["width"], meta["height"]
    a = np.fromfile(FIELDS / f"{stem}.bin", dtype=np.float32).reshape(h, w)
    return a, meta


def richness(a):
    fin = np.isfinite(a)
    interior_frac = float((~fin).mean())
    L = np.log(np.where(fin, a, 1.0)).astype(np.float64)
    med = np.median(L[fin])
    L[~fin] = med
    lo, hi = np.percentile(L[fin], [1, 99])
    rng = max(hi - lo, 1e-9)
    occ, ene = [], []
    for s in SIGMAS:
        band = gaussian_filter(L, s) - gaussian_filter(L, 2 * s)
        rel = np.abs(band) / rng
        occ.append(float((rel > TAU).mean()))
        ene.append(float(rel.mean()))
    return {
        "R_occ": float(np.mean(occ)),
        "R_energy": float(np.mean(ene)),
        "occ_per_scale": occ,
        "energy_per_scale": ene,
        "interior_frac": interior_frac,
    }


def main():
    minis = {m["id"]: m for m in json.load(open(MINIS))}
    rows = []
    for jf in sorted(FIELDS.glob("*.json")):
        stem = jf.stem
        a, meta = load_field(stem)
        r = richness(a)
        m = minis.get(stem, {})
        r.update(id=stem, period=m.get("period"), fw=m.get("fw"),
                 anchor=m.get("anchor"))
        rows.append(r)
    rows.sort(key=lambda r: r["R_occ"], reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "richness.json", "w"), indent=2)

    # energy-rank for cross-check
    by_e = {r["id"]: i for i, r in
            enumerate(sorted(rows, key=lambda r: r["R_energy"], reverse=True))}
    print(f"{'rank':>4} {'id':<10} {'per':>3} {'R_occ':>7} {'R_ene':>7} "
          f"{'int%':>5} {'Erank':>5}  occ_per_scale(1/2/4/8)")
    for i, r in enumerate(rows):
        occ = "/".join(f"{x:.2f}" for x in r["occ_per_scale"])
        print(f"{i:>4} {r['id']:<10} {r['period']:>3} {r['R_occ']:>7.4f} "
              f"{r['R_energy']:>7.4f} {r['interior_frac']*100:>4.0f}% "
              f"{by_e[r['id']]:>5}  {occ}")
    # rank correlation occ vs energy
    ids = [r["id"] for r in rows]
    ro = np.arange(len(ids))
    re = np.array([by_e[i] for i in ids])
    rho = np.corrcoef(ro, re)[0, 1]
    print(f"\nSpearman-ish (rank corr) R_occ vs R_energy: {rho:.3f}")


if __name__ == "__main__":
    main()
