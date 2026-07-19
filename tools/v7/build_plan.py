#!/usr/bin/env python
"""v7 render plan + unified cache manifest (augment) — post-freeze fold.

Same as tools/v6/build_plan.py, one generation on: it reuses the EXACT v4/v5/v6
augmentation recipe (the identical 42-slot multiset per location — palette(6) x
scale(3) x shift(2) x AA(2): 36 aliased ss1 + 6 ss4 lanczos3) via v6's
`emit_location`. Only the data changes — the 536 NEW post-freeze locations (loc_ids
5261..5796) are appended.

**AMENDMENT 1 — no ss2 slot.** The 640x360 ss2 deploy-geometry aug slot is NOT added
(see build_metadata.amendment_1_ss2_gap): confined to the appended block by the frozen
byte gate, it would correlate with family+label and let the model shortcut. The
recipe stays the frozen 42-slot multiset; the ss2 AA gap is a knowingly-accepted,
second-order covariate shift.

TWO byte gates (both ABORTS):
  (a) RECIPE-PARITY: regenerate the frozen v5 cache rows (loc 0..4621) from the committed
      recipe and assert byte-identity to data/v5/cache_manifest.jsonl — inherited from
      v6's gate (tools/v6/build_plan.verify_recipe_parity).
  (b) FROZEN-CACHE byte gate: assert the v7 cache_manifest's frozen prefix (loc 0..5260,
      = v5 verbatim + gather verbatim) is byte-identical to data/v6/cache_manifest.jsonl.
      Regenerated per-row via emit_location against the three frozen cache dirs, so any
      recipe drift on the gather rows aborts too.

Outputs:
  data/v7/plan.jsonl            : post-freeze-only render plan (536 loc x 42 = 22512 rows)
                                  -> the ONLY thing v4-render-batch renders.
  data/v7/cache_manifest.jsonl  : unified (v6 cache rows reused VERBATIM -> the existing
                                  data/v4|v5|v6 JPGs; + new rows -> data/v7/aug_cache).

  uv run python tools/v7/build_plan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "v6"))
import build_plan as v6bp  # noqa: E402  (emit_location + recipe constants, reused verbatim)

MANIFEST = ROOT / "data" / "v7" / "manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
V6_CACHE_MANIFEST = ROOT / "data" / "v6" / "cache_manifest.jsonl"
V4_CACHE_DIR = ROOT / "data" / "v4" / "aug_cache"             # frozen Mandelbrot JPGs
V5_JULIA_CACHE_DIR = ROOT / "data" / "v5" / "aug_cache_julia"  # frozen J0 Julia JPGs
GATHER_CACHE_DIR = ROOT / "data" / "v6" / "aug_cache_gather"   # frozen gather_v6 JPGs
V7_CACHE_DIR = ROOT / "data" / "v7" / "aug_cache"             # new post-freeze JPGs
PLAN_OUT = ROOT / "data" / "v7" / "plan.jsonl"
CACHE_MANIFEST_OUT = ROOT / "data" / "v7" / "cache_manifest.jsonl"

N_V6 = 5261                 # frozen prefix loc count (0..5260)
N_MANDEL = 3622             # loc < 3622 -> v4 cache
N_V5 = 4622                 # 3622..4621 -> v5 julia cache; 4622..5260 -> gather cache
SLOTS = 42                  # frozen 42-slot recipe (Amendment 1: no ss2 slot added)


def _frozen_cache_dir(loc_id: int) -> Path:
    if loc_id < N_MANDEL:
        return V4_CACHE_DIR
    if loc_id < N_V5:
        return V5_JULIA_CACHE_DIR
    return GATHER_CACHE_DIR


def main() -> None:
    import math
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    palettes = [r["name"] for r in roster]
    fam_of = {r["name"]: r["palette_family"] for r in roster}
    assert len(palettes) == 6, f"expected 6 palettes, got {len(palettes)}"
    n_combo = len(palettes) * len(v6bp.SCALES)
    angle_of = {}
    for pi, pal in enumerate(palettes):
        for si, sc in enumerate(v6bp.SCALES):
            angle_of[(pal, sc)] = 2.0 * math.pi * (pi * len(v6bp.SCALES) + si) / n_combo

    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    frozen = [(i, r) for i, r in enumerate(rows) if i < N_V6]
    appended = [(i, r) for i, r in enumerate(rows) if i >= N_V6]
    print(f"unified v7 manifest: {len(rows)} locations "
          f"(frozen v6 {len(frozen)}, appended {len(appended)})")
    assert len(frozen) == N_V6, f"frozen prefix drift: {len(frozen)} != {N_V6}"
    assert len(appended) == 536, f"appended drift: {len(appended)} != 536"

    # ---- GATE (a): inherited v5 recipe-parity (loc 0..4621) ----
    n_v5_parity = v6bp.verify_recipe_parity()
    print(f"GATE (a) v5 recipe-parity PASS: {n_v5_parity} rows byte-match v5 cache")

    # ---- GATE (b): frozen-cache byte gate — regenerate loc 0..5260, match v6 cache ----
    frozen_cm = []
    for loc_id, r in frozen:
        v6bp.emit_location(loc_id, r, palettes, fam_of, angle_of, [], frozen_cm,
                           _frozen_cache_dir(loc_id), emit_plan=False)
    v6_cm_lines = [l for l in V6_CACHE_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(frozen_cm) == len(v6_cm_lines) == N_V6 * SLOTS, \
        f"GATE (b) FAIL: regenerated {len(frozen_cm)} vs v6 {len(v6_cm_lines)} (expect {N_V6*SLOTS})"
    drift = []
    for i, (a, b) in enumerate(zip((json.dumps(x) for x in frozen_cm), v6_cm_lines)):
        if a != b:
            drift.append(i)
    assert not drift, f"GATE (b) FAIL: {len(drift)} frozen cache rows drift, e.g. row {drift[0]}"
    print(f"GATE (b) frozen-cache byte gate PASS: {len(frozen_cm)} rows byte-identical to v6")

    # ---- emit appended plan + cache rows (loc 5261..5796 -> data/v7/aug_cache) ----
    plan_rows, appended_cm = [], []
    for loc_id, r in appended:
        v6bp.emit_location(loc_id, r, palettes, fam_of, angle_of, plan_rows, appended_cm,
                           V7_CACHE_DIR, emit_plan=True)
    assert len(plan_rows) == len(appended) * SLOTS, \
        f"plan row count {len(plan_rows)} != {len(appended)}*{SLOTS}"
    assert len(appended_cm) == len(appended) * SLOTS

    PLAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_OUT.open("w", encoding="utf-8") as f:
        for row in plan_rows:
            f.write(json.dumps(row) + "\n")
    # unified cache manifest: v6 rows VERBATIM (byte-copy) + appended rows
    with CACHE_MANIFEST_OUT.open("w", encoding="utf-8") as f:
        for line in v6_cm_lines:
            f.write(line + "\n")
        for row in appended_cm:
            f.write(json.dumps(row) + "\n")

    from collections import Counter
    print(f"\nappended renders to build : {len(plan_rows)}  ({len(appended)} loc x {SLOTS})")
    print(f"unified cache rows        : {len(v6_cm_lines) + len(appended_cm)}  "
          f"(frozen v6 {len(v6_cm_lines)} reused + appended {len(appended_cm)} new)")
    print(f"wrote {PLAN_OUT}")
    print(f"wrote {CACHE_MANIFEST_OUT}")
    fam = Counter(p["fractal_type"] for p in plan_rows)
    print(f"plan per-family rows: {dict(sorted(fam.items()))}")
    n_ss1 = sum(1 for p in plan_rows if p["ss"] == 1)
    n_ss4 = sum(1 for p in plan_rows if p["ss"] == 4)
    print(f"plan ss split: ss1(box)={n_ss1}  ss4(lanczos3)={n_ss4}  (no ss2 — Amendment 1)")
    c_rows = sum(1 for p in plan_rows if "c_re" in p)
    print(f"plan rows carrying c (julia:mb): {c_rows}")


if __name__ == "__main__":
    main()
