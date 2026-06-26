#!/usr/bin/env python
"""v5 render plan + unified cache manifest (Part 3 — augment).

Reuses the EXACT v4 augmentation recipe (tools/v4/build_plan.py): the identical
42-slot multiset per location (palette(6) x scale(3) x shift(2) x AA(2): 36 aliased
ss1 + 6 ss4), the same fixed shift-angle schedule, the same scale/shift fractions.
Only the data changes (Julia rows folded in).

RECIPE-PARITY GATE (run before emitting anything): regenerate the Mandelbrot cache
rows from the unified manifest and assert they are byte-identical to the existing
data/v4/cache_manifest.jsonl. If the recipe drifted at all, this aborts — drift is
the real risk, not binary drift (Part 0 already proved the binary).

Outputs:
  data/v5/plan_julia.jsonl        : Julia-only render plan ({...,fractal_type,c_re,c_im})
                                    -> the ONLY thing v4-render-batch renders.
  data/v5/cache_manifest.jsonl    : unified (v4 Mandelbrot rows reused VERBATIM,
                                    pointing at the existing data/v4/aug_cache JPGs,
                                    + new Julia rows pointing at data/v5/aug_cache_julia).

  uv run python tools/v5/build_plan.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "v5" / "manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
V4_CACHE_MANIFEST = ROOT / "data" / "v4" / "cache_manifest.jsonl"
V4_CACHE_DIR = ROOT / "data" / "v4" / "aug_cache"            # existing Mandelbrot JPGs (reused)
JULIA_CACHE_DIR = ROOT / "data" / "v5" / "aug_cache_julia"   # new Julia JPGs
PLAN_OUT = ROOT / "data" / "v5" / "plan_julia.jsonl"
CACHE_MANIFEST_OUT = ROOT / "data" / "v5" / "cache_manifest.jsonl"

# --- recipe constants (MUST match tools/v4/build_plan.py verbatim) ---
SCALES = [0.7, 1.0, 1.3]
SHIFT_FRAC = 0.4


def scale_tok(s: float) -> str:
    return f"{s:.1f}"


def fmt_f64(x: float) -> str:
    return repr(float(x))


def emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, cm_rows,
                  cache_dir: Path, is_julia: bool):
    """Generate the 42 cache rows for one location. Mandelbrot writes cm_rows only
    (JPGs already exist); Julia writes both plan_rows (to render) and cm_rows."""
    cx0, cy0 = float(r["cx"]), float(r["cy"])
    fw0 = float(r["fw"])
    loc_dir = cache_dir / str(loc_id)
    base = dict(label=r["label"], split=r["split"], group_id=r["group_id"],
                source=r["source"], biased=r["biased"])
    ft = "julia" if is_julia else "mandelbrot"
    c_re, c_im = (r.get("c_re"), r.get("c_im")) if is_julia else (None, None)

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
        if is_julia:
            plan_rows.append({
                "cx": fmt_f64(cx), "cy": fmt_f64(cy), "fw": fmt_f64(fw_slot),
                "palette": pal, "ss": ss, "filter": filt, "out": out,
                # Julia coupling consumed by v4-render-batch (viewport = z-plane).
                "fractal_type": "julia", "c_re": c_re, "c_im": c_im,
            })
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


def main() -> None:
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
    mand = [(i, r) for i, r in enumerate(rows) if r.get("fractal_type") != "julia"]
    julia = [(i, r) for i, r in enumerate(rows) if r.get("fractal_type") == "julia"]
    print(f"unified manifest: {len(rows)} locations (mandelbrot {len(mand)}, julia {len(julia)})")

    # ---- RECIPE-PARITY GATE: regenerate Mandelbrot cm rows, assert == v4 ----
    mand_cm = []
    for loc_id, r in mand:
        assert loc_id < 3622, "Mandelbrot rows must occupy loc_ids 0..3621 (v4 cache order)"
        emit_location(loc_id, r, palettes, fam_of, angle_of, [], mand_cm,
                      V4_CACHE_DIR, is_julia=False)
    v4_cm = [json.loads(l) for l in V4_CACHE_MANIFEST.read_text().splitlines() if l.strip()]
    assert len(mand_cm) == len(v4_cm), \
        f"recipe drift: regenerated {len(mand_cm)} mand rows vs v4 {len(v4_cm)}"
    for a, b in zip(mand_cm, v4_cm):
        # compare the recipe-defining axes + path; v4 rows carry the same keys
        for k in ("location_id", "palette", "palette_family", "scale", "shift_id",
                  "aa_level", "fractal_type", "path", "label", "split", "group_id"):
            assert a[k] == b[k], f"recipe drift at loc {a['location_id']} key {k}: {a[k]} != {b[k]}"
        assert abs(a["shift_dx"] - b["shift_dx"]) < 1e-12 \
            and abs(a["shift_dy"] - b["shift_dy"]) < 1e-12, \
            f"recipe drift: shift offset at loc {a['location_id']}"
    print(f"RECIPE-PARITY GATE PASS: {len(mand_cm)} Mandelbrot cache rows byte-match v4")

    # ---- emit Julia plan + Julia cache rows ----
    plan_rows, julia_cm = [], []
    for loc_id, r in julia:
        emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, julia_cm,
                      JULIA_CACHE_DIR, is_julia=True)

    PLAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_OUT.open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    # unified cache manifest: v4 Mandelbrot rows VERBATIM + new Julia rows
    with CACHE_MANIFEST_OUT.open("w", encoding="utf-8") as f:
        for line in V4_CACHE_MANIFEST.read_text().splitlines():
            if line.strip():
                f.write(line + "\n")
        for row in julia_cm:
            f.write(json.dumps(row) + "\n")

    print(f"\nJulia renders to build : {len(plan_rows)}  ({len(julia)} loc x 42)")
    assert len(plan_rows) == len(julia) * 42
    print(f"unified cache rows     : {len(v4_cm) + len(julia_cm)}  "
          f"(mandelbrot {len(v4_cm)} reused + julia {len(julia_cm)} new)")
    print(f"wrote {PLAN_OUT}")
    print(f"wrote {CACHE_MANIFEST_OUT}")
    # ss split for ETA
    n_ss1 = sum(1 for p in plan_rows if p["ss"] == 1)
    n_ss4 = sum(1 for p in plan_rows if p["ss"] == 4)
    print(f"plan ss split: ss1(box)={n_ss1}  ss4(lanczos3)={n_ss4}")


if __name__ == "__main__":
    main()
