# Palette pool — add a luminance floor (rebuild + by-eye review)

## Goal
The pool builder gates extracted palettes on a complexity/quality composite (≥ 0.422) but has **no brightness gate**, so dark palettes entered the 777 and produce unrenderable/unlabelable crops. Add a **luminance floor** to the pool build so dark palettes are never drawn again — for location rendering, the preference sampler, and coloring alike. This pass implements the gate, rebuilds the pool at a default floor, and produces the material to set the floor **by eye**. Do not wire the batch re-render here (separate next pass).

## 1. Locate current gates
Find `tools/palettes/build_pool.py` and its existing gates: the extracted composite ≥ 0.422 threshold + the ~20 gate-rejects, and the curated q2+q3 inclusion. The luminance floor is **additive** — keep every existing gate intact; the new gate stacks on top.

## 2. Per-palette brightness
Compute a brightness stat over each palette's stops:
- **mean luma** (primary),
- plus **min** and a **low percentile (P10)** luma, so a palette with a bright mean but a long dark stretch is visible and not falsely passed.

Use the **same luma definition** the darkness detector used (stay consistent with `out/dark_detect_v6/`). Report the brightness distribution **split by source**: curated-q3 / curated-q2 / extracted (this shows where the dark ones concentrate — expected: extracted).

## 3. Survivor table at candidate floors
For a few mean-luma floors — e.g. **60 / 70 / 84** (anchor near the curated-q3 floor ~84) — report:
- palettes surviving, total and per source,
- how many of the palettes **actually used in `gather_v6`** each floor would have excluded (tie it back to the observed problem).
Compact table.

## 4. Boundary swatch montage (the by-eye cut)
Render each palette as a horizontal **color-bar** (stops left→right; one cycle for cyclic). Montage the palettes in the **boundary zone** (say mean-luma 40–90), sorted by luma, each annotated with mean/min luma and source. This is the sheet to place the floor on. **Palette bars only — no fractal renders.** (If the call is ambiguous we can add fractal swatches on a standard bright location in a follow-up; don't do it now.)

## 5. Apply + rebuild
Add the luminance floor as a **named constant**, default near the curated-q3 floor, and rebuild `pool_colormaps.json`. Report the new pool count and the **full drop list** (name, source, mean luma). The old pool is git-tracked, so it's recoverable — no separate backup step.

The floor is **provisional at the default** until confirmed from the montage; re-running at a different floor is a one-constant change with no re-renders, so leave it trivially re-runnable.

## Output
Survivor-by-floor table, the boundary swatch montage, the drop list, and the rebuilt pool.
