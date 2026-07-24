# Emission v1 — diversity-aware emission (deficit colorize + selection)

Source ledger: `data\discovery\steered_run2\outcome_ledger.jsonl` · floor **0.75** (production gate 0.90) · release N=**12** · target gated ≥ **36**.

## Intake — morph clusters among admitted locations

- **54** admitted locations (current-decode ∧ decoded_class==3 ∧ guard_pass ∧ distinct)
- **54** morph clusters (within-type, cos>0.974) across **8** fractal types:
  - `julia:mandelbrot`: 1 locations → 1 clusters
  - `julia:multibrot3`: 5 locations → 5 clusters
  - `julia:multibrot4`: 1 locations → 1 clusters
  - `julia:multibrot5`: 1 locations → 1 clusters
  - `mandelbrot`: 12 locations → 12 clusters
  - `multibrot3`: 10 locations → 10 clusters
  - `multibrot4`: 8 locations → 8 clusters
  - `multibrot5`: 16 locations → 16 clusters

## Niche occupancy + deficit (before → after)

- feasible cells: **8208** ((type,cluster) × 19 flavors × 8 styles)
- BEFORE (empty pool): 0 populated, deficit = uniform target over all 8208 feasible cells
- AFTER: **36** distinct cells populated by the 36-wallpaper gated pool; **0** cells hit the attempt cap and left support
- **36** distinct cells did the 36-surplus populate (out of 8208 feasible).
- axis coverage in the gated pool: **7** types · **28** morph clusters · **18**/19 palette flavors · **8**/8 render styles
  - render styles present: stripe×10, composite_c13_smooth_stripe×8, smooth_angle_min×5, composite_c17_smooth_curvature×4, tia×3, smooth×2, smooth_mean_angle×2, composite_c7_smooth_trap_circle×2

## Realized vs nominal surplus

Render styles route to two heads: **smooth → wallpaper head** (floor 0.75, production gate 0.90); **strange → mining head** (floor 0.05, production gate 0.50). Quality is only compared within a niche, which pins the style/head, so the heads never mix.

- attempts: **80** · gated: **36** → post-floor pass rate **45.0%** · render errors: 0
- wallpaper-head (smooth): **2** gated, **2** also clear the 0.90 production gate → the floor's extra admits (0.75–0.90) = 0
- mining-head (strange): **34** gated, **2** also clear the 0.50 production gate → the floor's extra admits (0.05–0.50) = 32
- reading: the permissive floors are doing a large amount of work (32/36 gated wallpapers sit below their production gate).

## Colorizer choice — deficit-driven palette/style spread

Chosen palette-flavor distribution over 80 colorize attempts vs the uniform-random expectation (4.2/flavor):

| palette flavor | chosen | uniform-random |
|---|---:|---:|
| k16:1 | 6 | 4.2 |
| k16:5 | 6 | 4.2 |
| k16:6 | 5 | 4.2 |
| special:outlier | 5 | 4.2 |
| k16:7 | 5 | 4.2 |
| special:neutral | 5 | 4.2 |
| k16:2 | 5 | 4.2 |
| special:spectral | 5 | 4.2 |
| k16:15 | 5 | 4.2 |
| k16:14 | 4 | 4.2 |
| k16:9 | 4 | 4.2 |
| k16:4 | 4 | 4.2 |
| k16:11 | 4 | 4.2 |
| k16:16 | 4 | 4.2 |
| k16:12 | 3 | 4.2 |
| k16:3 | 3 | 4.2 |
| k16:13 | 3 | 4.2 |
| k16:10 | 3 | 4.2 |
| k16:8 | 1 | 4.2 |

Render-style distribution (uniform-random 10.0/style):

| render style | chosen |
|---|---:|
| stripe | 15 |
| composite_c13_smooth_stripe | 14 |
| composite_c17_smooth_curvature | 12 |
| composite_c7_smooth_trap_circle | 11 |
| smooth | 9 |
| smooth_angle_min | 9 |
| tia | 5 |
| smooth_mean_angle | 5 |

## Release selection — 12 picks (greedy max-marginal-gain)

Marginal gain = niche-relative quality (within-niche p_ge3 percentile) × coverage gain (1 − max similarity to already-selected under the per-axis kernel). `nearest` = the closest already-selected wallpaper (displacement).

| # | id | type/cluster | flavor/style | p_ge3 | niche% | cov.gain | nearest (sim) |
|--:|---|---|---|--:|--:|--:|---|
| 1 | em_000000 | julia:mandelbrot/julia:mandelbrot#0 | k16:6/smooth | 0.949 | 1.00 | 1.00 | — |
| 2 | em_000053 | multibrot5/multibrot5#15 | k16:10/smooth | 0.945 | 1.00 | 1.00 | em_000000 (0.00) |
| 3 | em_000036 | multibrot4/multibrot4#6 | k16:3/tia | 0.651 | 1.00 | 1.00 | em_000000 (0.00) |
| 4 | em_000039 | multibrot5/multibrot5#1 | k16:15/composite_c13_smooth_stripe | 0.615 | 1.00 | 1.00 | em_000000 (0.00) |
| 5 | em_000035 | multibrot4/multibrot4#5 | k16:14/composite_c17_smooth_curvature | 0.426 | 1.00 | 1.00 | em_000000 (0.00) |
| 6 | em_000075 | multibrot3/multibrot3#1 | k16:2/tia | 0.408 | 1.00 | 1.00 | em_000000 (0.00) |
| 7 | em_000042 | multibrot5/multibrot5#4 | k16:5/stripe | 0.365 | 1.00 | 1.00 | em_000000 (0.00) |
| 8 | em_000003 | julia:multibrot3/julia:multibrot3#2 | k16:1/stripe | 0.357 | 1.00 | 1.00 | em_000000 (0.00) |
| 9 | em_000046 | multibrot5/multibrot5#8 | k16:13/stripe | 0.346 | 1.00 | 1.00 | em_000000 (0.00) |
| 10 | em_000057 | julia:multibrot3/julia:multibrot3#2 | k16:9/tia | 0.315 | 1.00 | 1.00 | em_000000 (0.00) |
| 11 | em_000009 | mandelbrot/mandelbrot#1 | special:outlier/composite_c13_smooth_stripe | 0.292 | 1.00 | 1.00 | em_000000 (0.00) |
| 12 | em_000054 | julia:mandelbrot/julia:mandelbrot#0 | k16:15/stripe | 0.259 | 1.00 | 1.00 | em_000000 (0.00) |

Release full-res PNGs: **12** under `out\emission_v1\release/`.

## Contact sheets

- `out\emission_v1\release_sheet.png` — the 12-wallpaper release
- `out\emission_v1\pool_sheet.png` — the gated pool grouped by niche
