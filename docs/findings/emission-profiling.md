# Emission / re-colorize profiling — where the pipeline spends compute

Measured directly (light `perf_counter` instrumentation reusing the
`build_emission_diversity_v1` classes verbatim) on stratified samples of the committed
`out/recolor_release` intake snapshot (1387-loc library, deficit pick). Pool render geometry
960×540 ss2 — the production default. Supersedes the two-point regression in
`recolorize-deficit-pick.md` (which didn't close: 5.7+4.1=9.8 vs measured steady 7.6).

## Table 1 — per-stage colorize breakdown (s/loc)

Stratified n=27 (3 per type), full `colorize()` step sequence timed per stage:

| stage | s/loc | % | what it is |
|---|---|---|---|
| **pref-pick + score** | **5.51 ± 4.10** | **48%** | ≤32 coarse recolors of the *cached* 640×360 field + 1 batched v3-gvo forward |
| **colorize render** | **5.89 ± 5.43** | **51%** | 1 full fractal **re-iteration** at 960×540 ss2 + color tail |
| head-score | 0.06 ± 0.07 | 0.5% | 1 forward (wallpaper *or* mining head), 384×224 |
| realized-stats | 0.10 ± 0.01 | 0.8% | numpy hue/chroma histogram of the JPG |
| pool append / select | ~0 | — | JSONL append; greedy select is once-per-run, not per-loc |
| **total** | **11.55** | 100% | (sample skews high vs the run's steady 7.6 — equal-per-type over-weights the expensive families + first-call CUDA warmup) |

**The two co-dominant halves are pref-pick (~5.5 s) and colorize render (~5.9 s); everything
else is noise.** head-score is 0.06 s — a single head forward is ~free; the ML cost is entirely
in the pref-pick's per-candidate pass, not the gate. This confirms the doc's ~5.7 s pref-pick
figure by direct measurement.

Note: the "pref-pick+score" pass is **not** colored-CLIP (an earlier assumption). It is the
pref-v3-gvo ranking head (`conditioned_colorize.Scorer` → `query_batch_gen.score_frames`)
scoring ≤32 candidate recolors of the **cached** 640×360 smooth field — **no fractal iteration**.
Candidate count is capped at `PaletteRanker.MAX_PALETTES = 32`; the "152 options/loc" in the log
is the deficit model's *cell* count, not the palette count.

### Why pref-pick costs what it does

Decomposed (n=9, 32 candidates each):

| sub-step | s/loc | % |
|---|---:|---:|
| field load + stretch + coarse | 0.05 | 0.5% |
| **32× coarse recolor (numpy)** | **9.29** | **96.4%** |
| transform + batched v3-gvo forward (GPU) | 0.30 | 3.1% |

**The concrete reason: the pref-pick is unbatched CPU numpy recoloring, not model inference.**
~290 ms per candidate × ≤32, done one-at-a-time. The GPU forward is already batched and costs
0.30 s (3%); the field is cached (no iteration). Each `render_candidate_coarse` runs the full
color tail on a 512×288 field, and the palette LUT is built by a **Python 4096-entry loop**
(`colormap.py:206`, memoized per-palette but cold on first touch of each distinct palette — this
sample hit all-distinct flavors, so it reflects cold-LUT cost). So the pass is **already batched
where it matters (GPU) and cacheable** (scores are deterministic in `(loc, flavor, palette-set)`),
but it wastes ~5 s/loc doing serial numpy work that should vectorize to ~1 s.

## Table 2 — render cost by (fractal-type × render-mode), s/loc @ 960×540 ss2

Single location per type, all 8 promoted render styles, fixed palette. Sorted by mean cost:

| type | n in lib | mean render | rel. | smooth | cheapest strange |
|---|---:|---:|---:|---:|---:|
| julia:multibrot3 | 106 | 1.40 | 1.0× | 2.07 | 1.03 |
| julia:multibrot4 | 88 | 1.62 | 1.2× | 2.15 | 1.28 |
| julia:multibrot5 | 79 | 2.28 | 1.6× | 2.80 | 1.95 |
| multibrot5 | 246 | 3.96 | 2.8× | 4.57 | 3.57 |
| julia:mandelbrot | 81 | 6.20 | 4.4× | 6.69 | 5.71 |
| multibrot3 | 212 | 6.42 | 4.6× | 6.95 | 6.03 |
| mandelbrot | 227 | 7.88 | 5.6× | 8.23 | 7.39 |
| phoenix | 237 | 9.41 | 6.7× | 9.86 | 8.92 |
| multibrot4 | 111 | 13.20 | 9.4× | 14.18 | 12.39 |

Library-weighted mean render: **6.9 s/loc** (smooth) / 6.3 (all-styles).

**Two facts for the compute-aware measure:**
1. **Render cost is set by fractal TYPE, not render mode** — a 9.4× spread across types
   (julia:multibrot3 ~1.4 s → multibrot4 ~13 s) vs only ±10–28% within a type across modes.
   Deep/near-boundary iteration (multibrot4, phoenix, mandelbrot) dominates; shallow low-degree
   Julias are ~an order of magnitude cheaper. **Difficulty ≈ type; mode is a second-order tweak.**
2. **Smooth is the *most* expensive mode, not the cheapest** (+10% to +50% over the cheapest
   strange). Smooth renders via `render-one --dump-field` then the **Python** colormap tail;
   the Rust-native strange modes color in-engine. The Python tail is a fixed ~0.5–1 s that is a
   large fraction on cheap-iteration types (julia:multibrot3 +50%) and swamped on expensive ones
   (mandelbrot +10%). So a "smooth is cheap" prior is wrong — and it's exactly the tail the
   field-cache lever removes.

## Table 3 — ranked optimization levers

Whole-run reference: the `recolor_release` cover-all ran ~1387 loc at steady **7.6 s/loc**
(~2.9 h); the two co-dominant halves (~48% pick, ~51% render) are what the levers attack.

| # | lever | est. saving | code cost | risk |
|---|---|---|---|---|
| 1 | **field-cache → recolor** (render half). Dump each loc's 960×540 smooth field once to a persistent cache; future re-colorizes recolor from it (Python tail ~0.5–1 s) instead of re-iterating (~6.9 s). | **~6 s/loc on every 2nd+ re-colorize** (≈85% of the render stage). Break-even after **1** re-colorize (dump == 1 render). | med — persistent field cache keyed by (loc, geometry); wire `render_smooth` to read it (recolor path already exists). Strange *scalar* modes (tia/stripe/curvature) cacheable too; composites/direct_trap need re-iteration (partial). | low — recolor is byte-identical to dump+recolor; separability already test-locked. Storage ~8.3 MB/loc × 1387 ≈ **11.5 GB** (smooth f32). |
| 2 | **Vectorize the pref-pick recolor.** Batch the 32 candidate color-tails (one stacked transform/sRGB; loop only the LUT gather) and vectorize `build_lut`'s Python 4096-loop. | pick **~5.5 → ~1.5 s/loc**, on **every** pass (first included). | low–med — `colormap.py` refactor (`build_lut` + a batched coarse-recolor). | low — same numbers, no storage. |
| 3 | **Persist `(loc,flavor)→v3-gvo scores`** across re-colorizes (deterministic). 2nd+ pass skips the whole pick compute, leaving only the cheap deficit argmax. | **~5.5 s/loc on 2nd+ re-colorize** (kills the pick half). Complements #1. | low — the in-memory `PaletteRanker.cache` dict just needs to be made durable, keyed by scorer-version + palette-set hash. | low — must invalidate on scorer / palette-library change. |
| 4 | Per-style pool res (640 ss2 for strange via the mining head, 960 for smooth). | ~30% off the strange render stage. | low | med — violates the single-default pin (`pool-render-res-960.md`); deferred. |
| 5 | head-score / select / stats. | — | — | already <1.5% combined; **no action**. |

## Biggest lever

**field-cache → recolor (#1), paired with the persistent pref-score cache (#3).** The workflow is
*repeated* re-colorize over a **fixed** library (deficit/palette tuning), where both the fractal
field and the per-`(loc,flavor)` v3-gvo scores are identical run-to-run. Caching them turns the two
co-dominant halves into cache reads, taking a repeat re-colorize from **~7.6 → ~1 s/loc (~8×)** —
break-even after a single pass. If only one *code* fix ships, vectorizing the pref-pick recolor (#2)
is the highest-ROI standalone change: ~4 s/loc of pure unbatched-numpy waste, removed on every run,
at zero storage and zero correctness risk.

