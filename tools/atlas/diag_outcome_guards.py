#!/usr/bin/env python
"""Diagnostic: calibrate degenerate-outcome guards against the harvested ledger.

**Measurement only — builds nothing, gates nothing, refits nothing.** Reads the
181 harvested outcomes in `data/discovery/outcome_ledger.jsonl`, re-renders each
outcome's smooth field at the reframe/deploy search fidelity (640x360 ss2,
`--dump-field`), and computes three *model-free* discriminators for the two
degenerate failure modes:

  interior_frac  — fraction of NON-ESCAPED (max-iter) subpixels, read from the
                   ESCAPE MASK (dumped field is NaN at interior/non-escaped px),
                   NEVER from RGB luminance. The black-gate discriminator.
  field_std      — global std of the smooth field over the frame (step-0
                   exterior-vs-boundary discriminator). Primary flat candidate.
  hf_energy      — std of the 4-neighbour Laplacian of the smooth field, over
                   interior-masked taps. Ramp-robust: a linear purple ramp has a
                   ~zero Laplacian while a filament thicket does not, so this
                   separates "strong smooth gradient" from "real structure" where
                   field_std alone conflates them. (grad_mag = mean gradient
                   magnitude reported alongside as a cross-check.)

Occupancy is deliberately NOT computed — see the OCCUPANCY finding printed in the
report: it is an engine energy primitive read from the Rust sidecar
(`enrich`/present), and applying it to a dumped field would require reimplementing
that OKLab edge-energy primitive in Python (parity risk). That skip itself decides
the flat gate away from occupancy.

Reuses (does not reinvent): `location.render_one_flags` + `--dump-field`,
`colormap.load_field`, `round1_embed._render` (outcome tiles, the SAME
640x360 ss2 twilight_shifted view that was eyeballed), `probe.{BIN,PALETTE,
JPG_Q,auto_maxiter}`, and `production_seeder.build_contact_sheet` (tiling).

Ground truth = the eyeballed `contact_sheet.png` (Matt's eyes on the k3-descending
sheet); the ledger carries no stored verdict. So the primary calibration artifact
is the annotated sheet (same k3-descending order, each tile overlaid with its three
measures) — the threshold is read at the gap against the eye. NO code path changes.

  uv run python tools/atlas/diag_outcome_guards.py
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools", ROOT / "tools" / "corpus",
           ROOT / "tools" / "reframe_probe", ROOT / "tools" / "atlas_probe",
           ROOT / "tools" / "reframe", ROOT / "tools" / "mining"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np  # noqa: E402

import location as loc_mod                        # noqa: E402  render_one_flags / Location
from probe import BIN, PALETTE, JPG_Q, auto_maxiter  # noqa: E402
import round1_embed as r1e                         # noqa: E402  _render (eyeballed tile path)
from colormap import load_field                    # noqa: E402  dumped-field reader (NaN interior)
from production_seeder import build_contact_sheet, NCOL_DUP  # noqa: E402  reuse tiling

LEDGER = ROOT / "data" / "discovery" / "outcome_ledger.jsonl"
OUT = ROOT / "out" / "atlas" / "diag_outcome_guards"
FIELD_DIR = OUT / "fields"     # ephemeral raw fields
TILE_DIR = OUT / "tiles"       # ephemeral rendered outcome tiles
RENDER_W, RENDER_H, RENDER_SS = r1e.RENDER_W, r1e.RENDER_H, r1e.RENDER_SS
WORKERS = 6

# Interior gate is expected near this per the prompt (the confirmed black-gate).
INTERIOR_GATE_EXPECTED = 0.25


# --------------------------------------------------------------------------- #
# Render: one dumped field + one eyeballed tile per outcome.
# --------------------------------------------------------------------------- #
def dump_field(cx, cy, fw, maxiter, out_bin: Path) -> tuple[bool, str]:
    """render-one --dump-field at 640x360 ss2 (smooth field, NaN interior). Exits
    before coloring — no PNG. Same maxiter policy as the tile so interior_frac
    matches what the tile shows."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    loc = loc_mod.Location(family="mandelbrot", cx=str(cx), cy=str(cy), fw=str(fw),
                           c_re=None, c_im=None, family_params={})
    cmd = [
        str(BIN), "render-one", "--cx", str(cx), "--cy", str(cy), "--fw", repr(float(fw)),
        "--width", str(RENDER_W), "--height", str(RENDER_H), "--supersample", str(RENDER_SS),
        "--maxiter", str(maxiter), "--dump-field", str(out_bin),
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out_bin.exists()
    return ok, ("" if ok else r.stderr[-300:])


def render_one_outcome(row) -> tuple[str, str]:
    """Render the field + eyeballed tile for one ledger row. Returns (id, err|'')."""
    oid = row["id"]
    cx, cy, fw = row["outcome_cx"], row["outcome_cy"], row["outcome_fw"]
    mi = auto_maxiter(float(fw))
    fbin = FIELD_DIR / f"{oid}.bin"
    if not fbin.exists():
        ok, err = dump_field(cx, cy, fw, mi, fbin)
        if not ok:
            return oid, f"field: {err}"
    tile = TILE_DIR / f"{oid}.jpg"
    if not tile.exists():
        ok, err = r1e._render(cx, cy, fw, tile)   # 640x360 ss2 twilight_shifted (eyeballed view)
        if not ok:
            return oid, f"tile: {err}"
    return oid, ""


# --------------------------------------------------------------------------- #
# Measures — all from the dumped field, all model-free.
# --------------------------------------------------------------------------- #
def measures(oid: str) -> dict:
    fd = load_field(FIELD_DIR / f"{oid}.bin")
    v = fd.values                       # (H, W) float64, NaN at interior/non-escaped
    finite = np.isfinite(v)
    n = v.size
    interior_frac = float(1.0 - finite.mean())   # == fraction NaN == non-escaped fraction

    vals = v[finite]
    field_std = float(vals.std()) if vals.size else 0.0

    # 4-neighbour Laplacian; any tap touching a NaN -> NaN -> excluded (masks interior).
    up, down = v[:-2, 1:-1], v[2:, 1:-1]
    left, right = v[1:-1, :-2], v[1:-1, 2:]
    ctr = v[1:-1, 1:-1]
    lap = 4.0 * ctr - up - down - left - right
    lv = lap[np.isfinite(lap)]
    hf_energy = float(lv.std()) if lv.size else 0.0

    # Mean gradient magnitude (forward diff), NaN-touching diffs excluded — cross-check.
    gx = v[1:-1, 2:] - v[1:-1, 1:-1]
    gy = v[2:, 1:-1] - v[1:-1, 1:-1]
    gm = np.sqrt(gx * gx + gy * gy)
    gmv = gm[np.isfinite(gm)]
    grad_mag = float(gmv.mean()) if gmv.size else 0.0

    return dict(interior_frac=interior_frac, field_std=field_std,
                hf_energy=hf_energy, grad_mag=grad_mag, n_px=int(n),
                n_escaped=int(finite.sum()))


# --------------------------------------------------------------------------- #
# Reporting helpers.
# --------------------------------------------------------------------------- #
def ascii_hist(xs, nbins=24, width=48, label=""):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return f"{label}: (empty)"
    lo, hi = float(xs.min()), float(xs.max())
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(xs, bins=edges)
    mx = max(1, counts.max())
    lines = [f"{label}  [n={xs.size}] min={lo:.4g} med={np.median(xs):.4g} max={hi:.4g}"]
    for i in range(nbins):
        bar = "#" * int(round(width * counts[i] / mx))
        lines.append(f"  {edges[i]:9.4g}..{edges[i+1]:<9.4g} {counts[i]:4d} {bar}")
    return "\n".join(lines)


def largest_gap(sorted_vals):
    """Return (gap_size, lo_val, hi_val) of the largest consecutive gap in a sorted 1-D array."""
    a = np.asarray(sorted_vals, float)
    a = np.sort(a[np.isfinite(a)])
    if a.size < 2:
        return (0.0, float("nan"), float("nan"))
    d = np.diff(a)
    i = int(np.argmax(d))
    return (float(d[i]), float(a[i]), float(a[i + 1]))


def make_plots(rows_d, out_dir: Path):
    """Distributions (interior_frac/field_std/hf_energy over distinct) + scatter vs k3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k3 = np.array([r["k3"] for r in rows_d], float)
    IF = np.array([r["interior_frac"] for r in rows_d], float)
    FS = np.array([r["field_std"] for r in rows_d], float)
    HF = np.array([r["hf_energy"] for r in rows_d], float)

    # Distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, x, name, gate in ((ax[0], IF, "interior_frac", INTERIOR_GATE_EXPECTED),
                             (ax[1], FS, "field_std", None),
                             (ax[2], HF, "hf_energy", None)):
        a.hist(x, bins=30, color="#4477aa", edgecolor="#223")
        a.set_title(f"{name}  (n={len(x)} distinct)")
        a.set_xlabel(name)
        g, lo, hi = largest_gap(x)
        mid = 0.5 * (lo + hi)
        if np.isfinite(mid):
            a.axvline(mid, color="#cc6677", ls="--", lw=1.2,
                      label=f"largest gap @ {mid:.3g}")
        if gate is not None:
            a.axvline(gate, color="#228833", ls=":", lw=1.5, label=f"expected gate {gate}")
        a.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "distributions.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    # Scatter vs k3 — the failure this guard fixes: high-k3 outcomes exist at high
    # interior_frac / low structure (v5's k3 does NOT track these).
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, x, name, gate in ((ax[0], IF, "interior_frac", INTERIOR_GATE_EXPECTED),
                             (ax[1], FS, "field_std", None),
                             (ax[2], HF, "hf_energy", None)):
        a.scatter(x, k3, s=18, c="#4477aa", alpha=0.7, edgecolors="none")
        a.set_xlabel(name)
        a.set_ylabel("k3")
        r = np.corrcoef(x, k3)[0, 1] if len(x) > 2 else float("nan")
        a.set_title(f"k3 vs {name}   pearson={r:+.2f}")
        if gate is not None:
            a.axvline(gate, color="#228833", ls=":", lw=1.5)
    fig.tight_layout()
    p2 = out_dir / "scatter_vs_k3.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    return p1, p2


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    print(f"=== diag_outcome_guards: {len(rows)} outcomes from {LEDGER.name} ===")
    fams = {r["family"] for r in rows}
    assert fams == {"mandelbrot"}, f"expected all mandelbrot, got {fams}"

    # 1. Render fields + eyeballed tiles (parallel).
    print(f"rendering fields + tiles @ {RENDER_W}x{RENDER_H} ss{RENDER_SS} "
          f"(workers={WORKERS}) ...")
    fails = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for oid, err in ex.map(render_one_outcome, rows):
            if err:
                fails.append((oid, err))
    if fails:
        print(f"  WARNING: {len(fails)} render failure(s); first: {fails[0]}")
        bad = {oid for oid, _ in fails}
        rows = [r for r in rows if r["id"] not in bad]

    # 2. Measures per outcome.
    print("computing measures (interior_frac / field_std / hf_energy / grad_mag) ...")
    recs = []
    for r in rows:
        m = measures(r["id"])
        recs.append(dict(id=r["id"], distinct=bool(r["distinct"]),
                         mix_source=r["mix_source"], k3=float(r["k3"]),
                         dup_of=r.get("dup_of"), **m))

    # 3. Per-outcome CSV (all 181).
    csv_path = OUT / "table.csv"
    cols = ["id", "distinct", "mix_source", "k3", "interior_frac", "field_std",
            "hf_energy", "grad_mag", "n_escaped", "n_px"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in sorted(recs, key=lambda x: -x["k3"]):
            w.writerow({k: rec[k] for k in cols})
    print(f"  table -> {csv_path}")

    distinct = [r for r in recs if r["distinct"]]
    dups = [r for r in recs if not r["distinct"]]
    print(f"  {len(distinct)} distinct / {len(dups)} near-dup")

    # 4. Distributions + scatter over the 81 distinct.
    p1, p2 = make_plots(distinct, OUT)
    print(f"  distributions -> {p1}\n  scatter       -> {p2}")

    # 5. Annotated contact sheet — SAME k3-descending order as the eyeballed sheet.
    by_id = {r["id"]: r for r in recs}

    def _lab(r):
        return (f"{r['id'][-6:]} k{r['k3']:.2f} "
                f"i{r['interior_frac']*100:.0f}% "
                f"fs{r['field_std']:.0f} hf{r['hf_energy']:.0f}")

    dtiles = [(TILE_DIR / f"{r['id']}.jpg", _lab(r))
              for r in sorted(distinct, key=lambda x: -x["k3"])]
    utiles = [(TILE_DIR / f"{r['id']}.jpg", _lab(r))
              for r in sorted(dups, key=lambda x: -x["k3"])[:NCOL_DUP]]
    sheet = OUT / "contact_sheet_annotated.png"
    build_contact_sheet(
        dtiles, utiles, sheet,
        f"diag outcome guards — {len(dtiles)} distinct  (label: id  k=k3  "
        f"i=interior%  fs=field_std  hf=hf_energy[lap-std])")
    print(f"  annotated sheet -> {sheet}")

    # 6. Separation report (printed).
    _report(distinct, dups)


def _report(distinct, dups):
    d = distinct
    IF = np.array([r["interior_frac"] for r in d])
    FS = np.array([r["field_std"] for r in d])
    HF = np.array([r["hf_energy"] for r in d])
    GM = np.array([r["grad_mag"] for r in d])
    K3 = np.array([r["k3"] for r in d])

    P = print
    P("\n" + "=" * 78)
    P("SEPARATION REPORT  (calibration over the 81 distinct; ground truth = the")
    P("eyeballed contact_sheet_annotated.png — the ledger stores no verdict, so the")
    P("threshold is read at the gap against the eye. Values below LOCATE the gap.)")
    P("=" * 78)

    P("\n--- OCCUPANCY finding (decides the flat gate away from occupancy) ---")
    P("occupancy is NOT outcome-measurable from a dumped field without an engine emit.")
    P("It is an engine energy primitive (energy::occupancy, OKLab forward-diff edge")
    P("energy) computed INSIDE the Rust render path and read back from the enrich/")
    P("present sidecar (score_lib.py: 'occupancy ... read from the Rust sidecar').")
    P("render-one --dump-field exits BEFORE coloring, so it emits no occupancy, and")
    P("computing it here would mean reimplementing the OKLab edge primitive in Python")
    P("(parity risk — explicitly disallowed). => occupancy is skipped; field_std / ")
    P("hf_energy are the flat-gate candidates.")

    P("\n--- interior_frac (black gate) ---")
    P(ascii_hist(IF, label="interior_frac (distinct)"))
    g, lo, hi = largest_gap(IF)
    P(f"largest gap: {g:.3g} between {lo:.3g} and {hi:.3g}  -> midpoint {0.5*(lo+hi):.3g}")
    for thr in (0.20, 0.25, 0.30):
        n = int((IF >= thr).sum())
        P(f"  interior_frac >= {thr:.2f}: {n:2d}/{len(d)} distinct  "
          f"(their mean k3={K3[IF>=thr].mean() if n else float('nan'):.3f})")
    P(f"PROPOSED interior gate: {INTERIOR_GATE_EXPECTED:.2f} (expected/confirmed); "
      f"cross-check the >= tail on the sheet.")

    P("\n--- field_std (flat candidate #1) ---")
    P(ascii_hist(FS, label="field_std (distinct)"))
    g, lo, hi = largest_gap(np.sort(FS))
    P(f"largest gap: {g:.3g} between {lo:.3g} and {hi:.3g}  -> midpoint {0.5*(lo+hi):.3g}")
    P("step-0 reference: exterior ~<=2.1, boundary ~70-390. Low tail below:")
    for r in sorted(d, key=lambda x: x["field_std"])[:8]:
        P(f"  {r['id'][-6:]}  field_std={r['field_std']:8.2f}  hf={r['hf_energy']:8.2f}"
          f"  if={r['interior_frac']:.3f}  k3={r['k3']:.3f}  {r['mix_source']}")

    P("\n--- hf_energy = Laplacian std (flat candidate #2, ramp-robust) ---")
    P(ascii_hist(HF, label="hf_energy (distinct)"))
    g, lo, hi = largest_gap(np.sort(HF))
    P(f"largest gap: {g:.3g} between {lo:.3g} and {hi:.3g}  -> midpoint {0.5*(lo+hi):.3g}")
    P("low tail (flat/gradient candidates):")
    for r in sorted(d, key=lambda x: x["hf_energy"])[:8]:
        P(f"  {r['id'][-6:]}  hf={r['hf_energy']:8.2f}  field_std={r['field_std']:8.2f}"
          f"  grad_mag={r['grad_mag']:7.2f}  if={r['interior_frac']:.3f}  k3={r['k3']:.3f}")

    P("\n--- k3 does NOT track the degenerate measures (the failure the guard fixes) ---")
    for name, x in (("interior_frac", IF), ("field_std", FS), ("hf_energy", HF),
                    ("grad_mag", GM)):
        r = np.corrcoef(x, K3)[0, 1] if len(x) > 2 else float("nan")
        P(f"  pearson(k3, {name:13s}) = {r:+.3f}")
    hi_k3 = K3 >= np.median(K3)
    P(f"  high-k3 (>=median {np.median(K3):.3f}) outcomes at interior_frac>=0.25: "
      f"{int((hi_k3 & (IF>=0.25)).sum())}  |  at field_std<10: "
      f"{int((hi_k3 & (FS<10)).sum())}  -> these score good yet are degenerate.")

    # Degenerate set by the proposed guards (union), for mix_source / k3 skew.
    deg = [r for r in d if r["interior_frac"] >= INTERIOR_GATE_EXPECTED or r["field_std"] < 10.0]
    P("\n--- degenerate set (interior_frac>=0.25 OR field_std<10) ---")
    P(f"  |degenerate| = {len(deg)}/{len(d)} distinct")
    if deg:
        from collections import Counter
        src = Counter(r["mix_source"] for r in deg)
        allsrc = Counter(r["mix_source"] for r in d)
        P(f"  by mix_source (degenerate / all-distinct): "
          + ", ".join(f"{k} {src.get(k,0)}/{allsrc[k]}" for k in sorted(allsrc)))
        dk = np.array([r["k3"] for r in deg])
        P(f"  degenerate k3: mean={dk.mean():.3f} median={np.median(dk):.3f} "
          f"[{dk.min():.3f}, {dk.max():.3f}]   (all-distinct k3 mean={K3.mean():.3f})")
        P("  -> feeds the later theta_hat-shift question: how much re-baselining moves")
        P("     once these are gated out of the reward pool.")

    P("\n--- RECOMMENDATION (measurement-only; confirm against the sheet) ---")
    P("  * interior gate: interior_frac >= 0.25 (confirmed black-gate).")
    P("  * flat gate: prefer field_std (clean step-0 separation, no coloring); use")
    P("    hf_energy (Laplacian std) as the ramp-robust tie-breaker where field_std")
    P("    conflates a smooth gradient with structure. Pin the threshold at the")
    P("    field_std gap read off distributions.png / the annotated sheet.")
    P("  * occupancy: dropped (not field-measurable without an engine emit).")
    P("=" * 78)


if __name__ == "__main__":
    main()
