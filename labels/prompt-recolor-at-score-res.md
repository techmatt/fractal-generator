# Recolor at score resolution — beam scoring speedup (scoring-only, ranking-parity gated)

## Goal
Beam recolor is ~96% of beam wall (~710 ms/candidate) and runs the full colormap tail at ss2 (2048×1152) purely to feed the ~224²/384×224 scorer — coloring ~47× the pixels the net consumes (`lookup_linear` 36% + downsample 29%). Color the **scoring** candidates at a small fixed grid matched to the scorer's input instead of ss2. This is the real beam lever and directly speeds the pending bootstrap run.

## Correctness boundary (load-bearing — read twice)
Coarse coloring is for **beam SCORING recolors ONLY** — the throwaway images that feed pref-v2. Keeper renders — the strata picks re-rendered at label spec, and any wallpaper emission — **MUST stay on the full-res canonical path**. Make the coarse recolor a **distinct path reachable only from the scoring loop**; it must not be reachable from `render_corpus_crop`, the label-crop re-render, or wallpaper emission. Crossing this boundary silently degrades real output.

## 1. Coarse scoring-recolor path
- Read the scorer's exact input transform (e.g. the 384×224 stretch / 224² squash — use whatever pref-v2 actually consumes).
- **Downsample the ss2 scalar field to that input geometry once per location** (area/mean downsample of the scalar field — not nearest), and **cache it** like the ss2 field: it's location-invariant and colormap-independent, so all ~348 candidates reuse it.
- Per candidate: run the colormap tail (LUT gather + transform + interior fill) on the cached coarse field at the scorer's input res, then score directly (no ss2→1024 downsample, no giant LUT gather). A small margin above input res is fine if it helps parity — your call.
- This drops the ss2 anti-aliasing for the scoring image. That's acceptable — it's a throwaway scoring input, never a keeper.

## 2. Ranking-parity validation (the gate — pixel-parity is N/A here)
On ~4–5 locations spanning families, run the beam scoring **both ways** — full-res coloring vs coarse coloring — with the **existing** pref-v2, and compare the **decisions**, not the pixels:
- gen-0 top-K survivors (which candidates advance to the beam),
- final winner picks (the beam's output),
- within-location rank correlation of scores.

**Gate:** survivors/winners substantially agree. Report top-K overlap (Jaccard) + rank correlation per location. Outcomes:
- **Agree → ship it** as the beam's scoring default. The existing net is already robust to coarse coloring — no retrain needed.
- **Winners shift materially → do NOT ship as default.** Report it; the fix is retraining pref-v2 on coarse-colored inputs, which is a separate follow-up (not this prompt).

## 3. Measure
Per-candidate scoring-recolor time, full-res vs coarse, and projected beam wall + bootstrap-run time (expect ~710 ms → tens of ms).

## Report
The coarse path and where it's fenced (explicitly confirm keeper/emit renders are untouched), the ranking-parity result (overlap + correlation per location), the measured speedup, and the ship/no-ship call from the gate. If shipped, note the bootstrap run can now proceed fast.
