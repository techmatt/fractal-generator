# q4 ∂M-property — what local Mandelbrot property predicts good Julia c's?

Measurement pass (no descent / config / `data/` changes; prototypes the campaign-3
c-selection screen). Follow-up to [`q4_c_perturbation.md`](q4_c_perturbation.md), which
proved the exemplar conjunction (interior lakes + busy mid-detail + composed rest) is
essentially **c-unique** within fw∈[0.13,1.5] and that the generalizing lever is
**c-selection**, not framing. This pass tests which computable **∂M-local property** at `c`
predicts the conjunction — so julia parent c's can be *screened*, not stumbled on. Two fixes
over the last pass were load-bearing: **deep zoom** (julia fw swept to 0.03) and **c sampled
along ∂M** (boundary-screened ladders, not rings that drift off the boundary).

**Exemplar** — center-view Julia, `c=(0.26103, −0.48932)`, origin-centered,
`twilight_shifted`. See `out/q4_dM/exemplar_large.png`.

## Method

- **c pool (33, boundary-screened).** 8 anchors spanning varied ∂M structure
  (seahorse/elephant filamentary valleys, period-2-disk & cardioid smooth arcs, period-3
  bulb cusp, dendrite tip, two cardioid shoulders) each **projected onto ∂M** (ring-probe +
  bisection) and expanded into a **signed-normal ladder** (offsets {+0.02, −0.006, −0.03,
  −0.08} in c-plane units) so `dist_dM` carries real spread; + the exemplar's literal `c`.
- **Per-c ∂M properties (cheap, once per c).** `dist_dM` = signed distance to ∂M via
  ring-probe (− inside / + outside; exterior DE cross-check). `M_richness` = occupancy /
  mid_detail on the **local Mandelbrot** escape field (fw=0.10 window centered on `c`,
  512², f64).
- **Per-c Julia deep sweep.** center-descent fw log-spaced **[0.03, 1.5]** (8 rungs,
  geomspace → deep-weighted) × {center + 4 small ±0.15·fw pans} = 40 framings/c. **1320
  field-dumps**, f64 768×432 ss1, colormap-invariant, purged per-unit. ~226 s.
- **Axes reused** from `q4_c_perturbation` (calibrated bands + `busy_near_black` /
  `coherent_rest`). **J-quality** = −min over framings of `band_dist` to the exemplar band
  (`mid≈0.73, interior≈0.24, flat≈0.23`); **corner** = min band_dist ≤ 1.5 (same threshold
  as the prior pass → directly comparable). Tool:
  `tools/studies/q4_dM_property.py` (`pool`/`measure`/`analyze`/`morph`/`sheets`).

## HEADLINE — proximity to ∂M predicts J-quality; local-M-richness does NOT (it only screens out degenerates)

**22 of 33 c's are degenerate** (deep-inside → solid blob, outside → dust) — the ladder
directly renders H1's thin-shell structure: viability is a narrow band straddling ∂M. Among
the **11 viable** c's the prediction picture is sharp and splits cleanly:

| predictor | ρ (all 33 c's) | ρ (viable 11 only) | reading |
|---|---|---|---|
| `abs(dist_dM)` → Jq | **−0.70** | **−0.86** | proximity to ∂M is the real signal |
| `M_occupancy` → Jq | +0.74 | **−0.17** | ← global corr is degenerate-separation, not ranking |
| `M_mid_detail` → Jq | +0.75 | **−0.19** | same |

The strong *global* M_richness correlation (ρ≈0.75) is almost entirely the
**degenerate-vs-viable separation**: deep-inside blobs have both zero local-M-richness and
zero J-quality. **Restricted to viable c's, M_richness is null (even slightly negative,
ρ≈−0.19).** Decisively, **the exemplar is mid-pack in local-M-richness — rank 11/33 overall,
8/11 among viable** — yet it is the *only* corner. Local-M-richness does **not** single it
out. What *does* rank within-viable is **`abs(dist_dM)` (ρ=−0.86): the closer `c` sits to the
∂M knife-edge, the better the best-available julia framing.** (`prediction.png`.)

## The screen rule — a viability filter, not a conjunction locator (precision 0.11)

Best grid-searched rule: **straddle ∂M (`|dist_dM| ≤ 0.008`) AND `M_mid_detail ≥ 0.078`** →
selects 9 c's, **recall 1.00, precision 0.11**. It catches the exemplar but drags in 8 false
positives, because being near-∂M-and-richish is *necessary but nowhere near sufficient*. The
runner-up `cardioid_rt_sm0030` is instructive: `dist_dM +0.0008` (as close to ∂M as the
exemplar) with near-identical `M_mid 0.079` (vs 0.082) — yet **band_dist 2.52 vs the
exemplar's 0.41, 6× worse**. Matching both cheap ∂M axes does not reproduce the conjunction.
The exemplar's defining property lives in the `c`-specific *shape* of its ∂M neighborhood, not
in scalar proximity or richness.

**Correspondence (H3): NO — julia parents cannot be screened by Mandelbrot richness.** M-local
richness separates viable near-∂M `c` from degenerate blob/dust (a useful *coarse viability
prior*) but does not predict good julia parents finely and does not identify the exemplar.

## Does the conjunction appear at non-exemplar c's once we zoom deep? — NO (the open question, now closed)

This is exactly what the prior pass's fw≥0.13 floor could not answer. With the sweep pushed to
**fw 0.03**: **0 of 10 non-exemplar viable c's reach the corner** (band_dist ≤ 1.5); the
closest approach is `cardioid_rt_sm0030` at **2.52** vs the exemplar's **0.41**. Moreover
**every c's band-nearest framing sits at fw ≥ 0.16** — the deep rungs (0.03–0.13) never
produced the best framing at any `c`, exemplar included (its best is fw 0.858). **Deep zoom
does not rescue generalization; the conjunction is a genuine `c`-knife-edge, not a
depth-confound.** `sheet_best_per_c.png` shows it visually: the exemplar (tagged `*CORNER`) is
the sole tile with distributed multi-scale complexity **and** composed black lakes; every
viable non-exemplar `c` gives busy dendrite-dust-on-void **or** clean spirals **or**
one-big-lake — busy **or** lakes, never both — and the degenerate row is solid blobs.

## Motif variety — real, and from c (positive)

The 11 viable per-c deep bests are **11 distinct looks**: morph_clip (grayscale robust-z /
CLIP, library recipe) median off-diagonal cos **0.823** (just under the inter-location
yardstick 0.851), max **0.949** (under the 0.974 near-dup line), **zero** near-dup pairs
(`morph.json`, `morph_sim.npz`). Different `c` genuinely gives different looks — but, as in the
prior pass, **variety across `c` ≠ the exemplar look at other `c`**: only one of the eleven is
the conjunction.

## Verdict — for the campaign-3 c-selection screen

A cheap **Mandelbrot-only** screen genuinely helps, but only as a two-stage *prior*, not a
conjunction locator:
1. **Viability filter** — reject `c` that is deep-inside (solid-blob) or outside (dust); keep
   the thin straddle shell. `M_richness > 0` and small `|dist_dM|` both encode this (they
   collapse together off the boundary), and it removes ~2/3 of candidates for free.
2. **Rank survivors by `abs(dist_dM)`** — minimize distance to ∂M (within-viable ρ=−0.86 with
   J-quality). Get as close to the knife-edge as the precision allows.

But the honest ceiling: **neither axis reproduces the exemplar conjunction.** Being closest to
∂M with matching local richness still lands 6× off the band (`cardioid_rt_sm0030`). The
conjunction (distributed busy-ness **with** composed interior lakes) is a `c`-specific
property of the ∂M neighborhood's *geometry* that these scalar screens do not capture — it is
rarer than "near-∂M-and-rich," and deep zoom does not manufacture it. The screen usefully
**biases the sourcing distribution toward viable, varied, near-edge julia parents** (a valid
campaign-3 prior), but exemplar-grade conjunctions remain a low-base-rate find that a scalar
∂M screen shrinks the haystack for rather than pinpoints.

### Artifacts (`out/q4_dM/`)
`sheet_best_per_c.png` (primary — 11 viable bests + degenerate row, annotated
`dist_dM`/`M_richness`), `prediction.png` (H1 `dist_dM` + H2/H3 `M_richness` vs J-quality),
`exemplar_large.png`, `analysis.json` (correlations all + within-viable, screen rule,
conjunction@depth, per-c), `pool.json` (33 c's + ∂M properties), `morph.json` / `morph_sim.npz`,
`metrics.jsonl` (1320 framings). Regenerate:
`uv run python -m tools.studies.q4_dM_property {pool,measure,analyze,morph,sheets}`.
