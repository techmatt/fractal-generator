#!/usr/bin/env python
"""Cheap pre-harvest verification of the per-degree q3 threshold wiring (Part A).

No render, no scoring — just the decode + the per-degree lookup, checked against the
labeled eval scores. Three asserts, mirroring the prompt:

  (a) a deg-2 frame with p_good in [0.24, 0.50) now decodes 3 (was 2) once routed through
      the deg-2 t_good = 0.24;
  (b) the SAME (p_notbad, p_good) routed through a high-degree partition (t_good = 0.50)
      still decodes 2 — the split is load-bearing;
  (c) the default-0.5 corn_decode path is byte-identical to the historical inline decode
      for every frame in the eval (frozen callers unaffected).

Plus a routing check on t_good_for over every partition the seeder emits.

  uv run python tools/v6/verify_t_good.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

from score_lib import corn_decode                      # noqa: E402
from production_seeder import (                         # noqa: E402
    t_good_for, T_GOOD_OVERRIDES, T_GOOD_BASELINE, julia_partition,
)

T_GOOD_DEG2 = T_GOOD_OVERRIDES["mandelbrot"]  # 0.24; the deg-2 knee this test exercises

EVAL = ROOT / "data" / "classifier" / "v6" / "eval_scores_v6.jsonl"


def main() -> int:
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    print(f"=== verify per-degree t_good wiring ===  ({len(rows)} eval frames)")
    print(f"T_GOOD_DEG2={T_GOOD_DEG2}  T_GOOD_BASELINE={T_GOOD_BASELINE}\n")

    ok = True

    # --- routing: every partition the seeder can emit resolves correctly. --- #
    print("-- t_good_for routing --")
    deg2 = ["mandelbrot", julia_partition("mandelbrot")]           # julia:mandelbrot
    # jm3 carries its own revival threshold (0.30); every other high-degree partition
    # is still held at the baseline.
    jm3 = julia_partition("multibrot3")                            # julia:multibrot3 -> 0.30
    hi = ["multibrot3", "multibrot4", "multibrot5",
          julia_partition("multibrot4"),
          julia_partition("multibrot5"), "phoenix"]
    for p in deg2:
        got = t_good_for(p)
        flag = "OK" if got == T_GOOD_DEG2 else "FAIL"
        ok &= got == T_GOOD_DEG2
        print(f"  {p:<22} -> {got}  ({flag}, expect {T_GOOD_DEG2})")
    got = t_good_for(jm3)
    expect_jm3 = T_GOOD_OVERRIDES["julia:multibrot3"]
    flag = "OK" if got == expect_jm3 else "FAIL"
    ok &= got == expect_jm3
    print(f"  {jm3:<22} -> {got}  ({flag}, expect {expect_jm3})")
    for p in hi:
        got = t_good_for(p)
        flag = "OK" if got == T_GOOD_BASELINE else "FAIL"
        ok &= got == T_GOOD_BASELINE
        print(f"  {p:<22} -> {got}  ({flag}, expect {T_GOOD_BASELINE})")

    # --- (c) default path byte-identical to the historical inline decode. --- #
    def legacy(nb, g):   # the exact pre-change formula
        return 1 + int(nb >= 0.5) + int(g >= 0.5)

    mism = sum(1 for r in rows
               if corn_decode(r["v6_p_not_bad"], r["v6_p_good"]) !=
               legacy(r["v6_p_not_bad"], r["v6_p_good"]))
    print(f"\n-- (c) default-0.5 == legacy inline decode --")
    print(f"  mismatches over {len(rows)} frames: {mism}  ({'OK' if mism == 0 else 'FAIL'})")
    ok &= mism == 0

    # --- (a)/(b) real in-band deg-2 frame: split on identical probs. --- #
    band = [r for r in rows
            if r["fractal_type"] in ("mandelbrot", "julia")
            and r["v6_p_not_bad"] >= 0.5
            and T_GOOD_DEG2 <= r["v6_p_good"] < T_GOOD_BASELINE]
    print(f"\n-- (a)/(b) deg-2 frames in p_good band [{T_GOOD_DEG2}, {T_GOOD_BASELINE}) --")
    if not band:
        print("  FAIL: no real in-band deg-2 frame found in the eval")
        return 1
    print(f"  {len(band)} in-band frames; demonstrating the split on the first:")
    r = band[0]
    nb, g = r["v6_p_not_bad"], r["v6_p_good"]
    d_default = corn_decode(nb, g)                                   # legacy 0.5
    d_deg2 = corn_decode(nb, g, t_good_for("mandelbrot"))            # 0.24
    d_hi = corn_decode(nb, g, t_good_for("multibrot3"))             # 0.50
    print(f"  frame: type={r['fractal_type']} p_notbad={nb:.3f} p_good={g:.3f} label={r['label']}")
    print(f"    (a) default 0.5   -> decode {d_default}   (expect 2)")
    print(f"    (a) deg-2  0.24   -> decode {d_deg2}   (expect 3)   [was {d_default}]")
    print(f"    (b) high-deg 0.50 -> decode {d_hi}   (expect 2, same probs)")
    a_ok = d_default == 2 and d_deg2 == 3
    b_ok = d_hi == 2
    ok &= a_ok and b_ok
    print(f"  (a) {'OK' if a_ok else 'FAIL'}   (b) {'OK' if b_ok else 'FAIL'}")

    # every in-band deg-2 frame must flip 2->3 under 0.24 and hold at 2 under 0.50.
    flips = sum(1 for r in band
                if corn_decode(r["v6_p_not_bad"], r["v6_p_good"], T_GOOD_DEG2) == 3
                and corn_decode(r["v6_p_not_bad"], r["v6_p_good"], T_GOOD_BASELINE) == 2)
    print(f"  all in-band frames flip 2->3 at 0.24 AND hold at 2 under 0.50: "
          f"{flips}/{len(band)}  ({'OK' if flips == len(band) else 'FAIL'})")
    ok &= flips == len(band)

    print(f"\nverdict: {'PASS' if ok else 'FAIL — check above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
