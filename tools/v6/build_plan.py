#!/usr/bin/env python
"""v6 render plan + unified cache manifest (augment) — gather_v6 fold.

Reuses the EXACT v4/v5 augmentation recipe (tools/v4/build_plan.py): the identical
42-slot multiset per location (palette(6) x scale(3) x shift(2) x AA(2): 36 aliased
ss1 + 6 ss4), the same fixed shift-angle schedule, the same scale/shift fractions.
Only the data changes — the ~639 NEW gather_v6 locations (9 families) are appended.

RECIPE-PARITY GATE (run before emitting anything): regenerate the FROZEN v5 cache
rows (Mandelbrot loc 0..3621 + Julia 3622..4621) from the unified manifest and assert
they are byte-identical to the existing data/v5/cache_manifest.jsonl. If the recipe
drifted at all (the same `emit_location` builds the gather rows), this aborts.

Outputs:
  data/v6/plan_gather.jsonl     : gather-only render plan ({...,fractal_type,c_re,c_im})
                                  -> the ONLY thing v4-render-batch renders.
  data/v6/cache_manifest.jsonl  : unified (v5 rows reused VERBATIM, pointing at the
                                  existing data/v4/aug_cache + data/v5/aug_cache_julia
                                  JPGs, + new gather rows -> data/v6/aug_cache_gather).

  uv run python tools/v6/build_plan.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "v6" / "manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
V5_CACHE_MANIFEST = ROOT / "data" / "v5" / "cache_manifest.jsonl"
V4_CACHE_DIR = ROOT / "data" / "v4" / "aug_cache"            # frozen Mandelbrot JPGs
V5_JULIA_CACHE_DIR = ROOT / "data" / "v5" / "aug_cache_julia"  # frozen J0 Julia JPGs
GATHER_CACHE_DIR = ROOT / "data" / "v6" / "aug_cache_gather"   # new gather JPGs
PLAN_OUT = ROOT / "data" / "v6" / "plan_gather.jsonl"
CACHE_MANIFEST_OUT = ROOT / "data" / "v6" / "cache_manifest.jsonl"

N_V5 = 4622                 # frozen v5 rows: 0..3621 Mandelbrot, 3622..4621 Julia
N_MANDEL = 3622

# --- recipe constants (MUST match tools/v4|v5/build_plan.py verbatim) ---
SCALES = [0.7, 1.0, 1.3]
SHIFT_FRAC = 0.4
# Families that carry a fixed parameter c in the plan/cache (dynamical planes).
_C_FAMILIES = {"julia", "julia_multibrot3", "julia_multibrot4", "julia_multibrot5"}


def scale_tok(s: float) -> str:
    return f"{s:.1f}"


def fmt_f64(x: float) -> str:
    return repr(float(x))


def emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, cm_rows,
                  cache_dir: Path, emit_plan: bool):
    """Generate the 42 cache rows for one location. `r["fractal_type"]` selects the
    family; `c_re`/`c_im` are carried into the plan/cache row for the dynamical
    families. Frozen rows (`emit_plan=False`) write cm_rows only (JPGs already
    exist); gather rows (`emit_plan=True`) also write plan_rows (to render)."""
    cx0, cy0 = float(r["cx"]), float(r["cy"])
    fw0 = float(r["fw"])
    loc_dir = cache_dir / str(loc_id)
    base = dict(label=r["label"], split=r["split"], group_id=r["group_id"],
                source=r["source"], biased=r["biased"])
    ft = r.get("fractal_type", "mandelbrot")
    c_re, c_im = (r.get("c_re"), r.get("c_im")) if ft in _C_FAMILIES else (None, None)

    def emit(pal: str, sc: float, shift_id: str, ss: int):
        fw_slot = sc * fw0
        if shift_id == "shifted":
            ang = angle_of[(pal, sc)]
            mag = SHIFT_FRAC * fw_slot
            dx, dy = mag * math.cos(ang), mag * math.sin(ang)
            cx, cy = cx0 + dx, cy0 + dy
        else:
            dx = dy = 0.0
            cx, cy = cx0, cy0
        aa = "aliased" if ss == 1 else "antialiased"
        filt = "box" if ss == 1 else "lanczos3"
        fname = f"{pal}__s{scale_tok(sc)}__sh{shift_id}__ss{ss}.jpg"
        out = (loc_dir / fname).relative_to(ROOT).as_posix()
        if emit_plan:
            row = {
                "cx": fmt_f64(cx), "cy": fmt_f64(cy), "fw": fmt_f64(fw_slot),
                "palette": pal, "ss": ss, "filter": filt, "out": out,
                "fractal_type": ft,
            }
            if c_re is not None:
                row["c_re"] = c_re
                row["c_im"] = c_im
            plan_rows.append(row)
        cm_rows.append({
            "location_id": loc_id, **base,
            "palette": pal, "palette_family": fam_of[pal],
            "scale": sc, "shift_id": shift_id,
            "shift_dx": dx, "shift_dy": dy,
            "aa_level": aa, "fractal_type": ft,
            "path": out,
        })

    for pal in palettes:
        for sc in SCALES:
            for shift_id in ("center", "shifted"):
                emit(pal, sc, shift_id, ss=1)
    for pal in palettes:
        emit(pal, 1.0, "center", ss=4)


def _load_recipe_inputs():
    """(palettes, fam_of, angle_of, rows) — the shared front matter of main() and
    the parity gate, recomputed from the committed roster + on-disk unified manifest."""
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    palettes = [r["name"] for r in roster]
    fam_of = {r["name"]: r["palette_family"] for r in roster}
    assert len(palettes) == 6, f"expected 6 palettes, got {len(palettes)}"

    n_combo = len(palettes) * len(SCALES)
    angle_of = {}
    for pi, pal in enumerate(palettes):
        for si, sc in enumerate(SCALES):
            angle_of[(pal, sc)] = 2.0 * math.pi * (pi * len(SCALES) + si) / n_combo

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    return palettes, fam_of, angle_of, rows


def verify_recipe_parity() -> int:
    """RECIPE-PARITY GATE: regenerate the FROZEN v5 cache rows (Mandelbrot 0..3621 +
    J0 Julia 3622..4621) from the committed recipe and assert they are byte-identical to
    data/v5/cache_manifest.jsonl. Returns the matched row count; raises AssertionError on
    any drift. Pure check — writes nothing — so `tools/v6/test_recipe_parity.py` can run
    it in CI instead of it only firing when someone manually runs build_plan.py.

    V4_CACHE_DIR / V5_JULIA_CACHE_DIR are load-bearing, not superseded repro: this gate
    asserts the frozen Mandelbrot + J0 Julia JPGs are byte-identical to the recipe, and
    the unified cache manifest reuses them VERBATIM for the v6 train. Deleting those cache
    dirs breaks every location-classifier build (.audit-keep sentinel'd).
    """
    palettes, fam_of, angle_of, rows = _load_recipe_inputs()
    frozen = [(i, r) for i, r in enumerate(rows) if i < N_V5]
    assert len(frozen) == N_V5, f"frozen prefix drift: {len(frozen)} != {N_V5}"

    frozen_cm = []
    for loc_id, r in frozen:
        cache_dir = V4_CACHE_DIR if loc_id < N_MANDEL else V5_JULIA_CACHE_DIR
        emit_location(loc_id, r, palettes, fam_of, angle_of, [], frozen_cm,
                      cache_dir, emit_plan=False)
    v5_cm = [json.loads(l) for l in V5_CACHE_MANIFEST.read_text().splitlines() if l.strip()]
    assert len(frozen_cm) == len(v5_cm), \
        f"recipe drift: regenerated {len(frozen_cm)} frozen rows vs v5 {len(v5_cm)}"
    for a, b in zip(frozen_cm, v5_cm):
        for k in ("location_id", "palette", "palette_family", "scale", "shift_id",
                  "aa_level", "fractal_type", "path", "label", "split", "group_id"):
            assert a[k] == b[k], f"recipe drift at loc {a['location_id']} key {k}: {a[k]} != {b[k]}"
        assert abs(a["shift_dx"] - b["shift_dx"]) < 1e-12 \
            and abs(a["shift_dy"] - b["shift_dy"]) < 1e-12, \
            f"recipe drift: shift offset at loc {a['location_id']}"
    return len(frozen_cm)


def main() -> None:
    palettes, fam_of, angle_of, rows = _load_recipe_inputs()
    frozen = [(i, r) for i, r in enumerate(rows) if i < N_V5]
    gather = [(i, r) for i, r in enumerate(rows) if i >= N_V5]
    print(f"unified v6 manifest: {len(rows)} locations "
          f"(frozen v5 {len(frozen)}, gather_v6 {len(gather)})")
    assert len(frozen) == N_V5, f"frozen prefix drift: {len(frozen)} != {N_V5}"

    # ---- RECIPE-PARITY GATE (extracted → tools/v6/test_recipe_parity.py) ----
    n_frozen = verify_recipe_parity()
    print(f"RECIPE-PARITY GATE PASS: {n_frozen} frozen v5 cache rows byte-match v5")

    # ---- emit gather plan + gather cache rows ----
    plan_rows, gather_cm = [], []
    for loc_id, r in gather:
        emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, gather_cm,
                      GATHER_CACHE_DIR, emit_plan=True)

    PLAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_OUT.open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    # unified cache manifest: v5 rows VERBATIM + new gather rows
    with CACHE_MANIFEST_OUT.open("w", encoding="utf-8") as f:
        for line in V5_CACHE_MANIFEST.read_text().splitlines():
            if line.strip():
                f.write(line + "\n")
        for row in gather_cm:
            f.write(json.dumps(row) + "\n")

    print(f"\ngather renders to build : {len(plan_rows)}  ({len(gather)} loc x 42)")
    assert len(plan_rows) == len(gather) * 42
    print(f"unified cache rows      : {n_frozen + len(gather_cm)}  "
          f"(frozen v5 {n_frozen} reused + gather {len(gather_cm)} new)")
    print(f"wrote {PLAN_OUT}")
    print(f"wrote {CACHE_MANIFEST_OUT}")
    # per-family plan breakdown + ss split for ETA
    from collections import Counter
    fam = Counter(p["fractal_type"] for p in plan_rows)
    print(f"plan per-family rows: {dict(sorted(fam.items()))}")
    n_ss1 = sum(1 for p in plan_rows if p["ss"] == 1)
    n_ss4 = sum(1 for p in plan_rows if p["ss"] == 4)
    print(f"plan ss split: ss1(box)={n_ss1}  ss4(lanczos3)={n_ss4}")


if __name__ == "__main__":
    main()
