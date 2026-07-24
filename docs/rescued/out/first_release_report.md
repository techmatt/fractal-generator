# Emission — diversity-aware emission (deficit colorize + ranker-ordered intake + per-head release floors)

Source ledger(s): `out\first_release\ledgers\c1__breadth.jsonl`, `out\first_release\ledgers\c1__dive.jsonl`, `data\discovery\campaign2\breadth\outcome_ledger.jsonl`, `data\discovery\campaign2\dive\outcome_ledger.jsonl`, `data\discovery\phoenix_grid\grid\outcome_ledger_v7_t45.jsonl`, `data\discovery\classic_phoenix\outcome_ledger.jsonl`.

Location ranker (pref_loc_v0, **logi:v7+colored**) ORDERS the colorize queue (order, not filter — diversity supply untouched). Pool floors (permissive): wallpaper **0.75** / mining **0.05**. **Release floors** (per head, distinct): wallpaper **0.9** / mining **0.5**. Release N=**50** · target **150** release-eligible (post-floor surplus).

## Intake — morph clusters among admitted locations

- **1387** admitted locations (current-decode ∧ decoded_class==3 ∧ guard_pass ∧ distinct)
- **1268** morph clusters (within-type, cos>0.974) across **9** fractal types:
  - `julia:mandelbrot`: 81 locations → 68 clusters
  - `julia:multibrot3`: 106 locations → 71 clusters
  - `julia:multibrot4`: 88 locations → 71 clusters
  - `julia:multibrot5`: 79 locations → 68 clusters
  - `mandelbrot`: 227 locations → 205 clusters
  - `multibrot3`: 212 locations → 205 clusters
  - `multibrot4`: 111 locations → 105 clusters
  - `multibrot5`: 246 locations → 238 clusters
  - `phoenix`: 237 locations → 237 clusters

## Niche occupancy + deficit (before → after)

- feasible cells: **192736** ((type,cluster) × 19 flavors × 8 styles)
- BEFORE (empty pool): 0 populated, deficit = uniform target over all 192736 feasible cells
- AFTER: **660** distinct cells populated by the 660-wallpaper gated pool; **0** cells hit the attempt cap and left support
- **660** distinct cells did the 660-surplus populate (out of 192736 feasible).
- axis coverage in the gated pool: **9** types · **632** morph clusters · **19**/19 palette flavors · **8**/8 render styles
  - render styles present: tia×124, stripe×122, composite_c13_smooth_stripe×112, smooth_angle_min×80, smooth×78, composite_c17_smooth_curvature×60, smooth_mean_angle×54, composite_c7_smooth_trap_circle×30

## Pool inventory + per-head release floors

Render styles route to two heads: **smooth → wallpaper head** (pool floor 0.75, **release floor 0.9**); **strange → mining head** (pool floor 0.05, **release floor 0.5**). Quality is only compared within a niche, which pins the style/head, so the heads never mix. Pool admission is permissive (weak wallpapers persist as inventory); SELECTION only draws above the release floor.

- attempts: **1387** · pool-admitted (gated): **660** → pool pass rate **47.6%** · render errors: 0

| head | pool-admitted | release-eligible | inventory (below release floor) |
|---|--:|--:|--:|
| wallpaper (smooth, rel≥0.9) | 78 | 65 | 13 |
| mining (strange, rel≥0.5) | 582 | 82 | 500 |
| **total** | **660** | **147** | **513** |

**513/660** pool wallpapers are banked as inventory below their head's release floor — exactly the weak tiles the v1 permissive-only bar would have let compete for a release slot. The colorize targeted **150** release-eligible (post-floor) and reached **147**.

## Ranker reach — did ranked intake concentrate budget on good locations?

Admitted locations ordered by pref_loc_v0 score (desc); 'reach' = the deepest rank the colorize actually touched. If ranked intake works, colorize fills its surplus from the TOP of the ordering and never has to reach deep.

- 1387 admitted locations; **1387** got a colorize attempt, reaching rank **1387** (top 100% of the ordering).
- **147** locations contributed a release-eligible wallpaper, the deepest at rank **1381** (top 100%).
- reading: the surplus was filled within the top **100%** of ranker-ordered locations (reached deep — pool is quality-thin, not a ranking failure).

## Colorizer choice — deficit-driven palette/style spread

Chosen palette-flavor distribution over 1387 colorize attempts vs the uniform-random expectation (73.0/flavor):

| palette flavor | chosen | uniform-random |
|---|---:|---:|
| k16:9 | 86 | 73.0 |
| k16:4 | 86 | 73.0 |
| special:neutral | 82 | 73.0 |
| k16:1 | 78 | 73.0 |
| k16:7 | 77 | 73.0 |
| k16:16 | 77 | 73.0 |
| k16:15 | 77 | 73.0 |
| k16:11 | 75 | 73.0 |
| k16:5 | 74 | 73.0 |
| k16:12 | 73 | 73.0 |
| k16:8 | 72 | 73.0 |
| k16:3 | 71 | 73.0 |
| special:outlier | 69 | 73.0 |
| k16:6 | 68 | 73.0 |
| k16:2 | 68 | 73.0 |
| k16:14 | 65 | 73.0 |
| k16:10 | 65 | 73.0 |
| special:spectral | 63 | 73.0 |
| k16:13 | 61 | 73.0 |

Render-style distribution (uniform-random 173.4/style):

| render style | chosen |
|---|---:|
| smooth_angle_min | 191 |
| stripe | 186 |
| composite_c7_smooth_trap_circle | 183 |
| smooth_mean_angle | 170 |
| smooth | 167 |
| tia | 167 |
| composite_c13_smooth_stripe | 166 |
| composite_c17_smooth_curvature | 157 |

## Release selection — 50 picks (greedy max-marginal-gain)

**Render-mode split (heads never compared in one step).** Smooth slots are filled from the wallpaper head, strange from the mining head, by two DISJOINT within-head greedy passes. Target strange frac **0.5** → slots smooth **25** / strange **25**. Eligible: smooth **65** / strange **82**. Realized: smooth **25** / strange **25** (strange frac **0.50**). Strange modes: tia×10, composite_c13_smooth_stripe×5, composite_c17_smooth_curvature×4, stripe×4, smooth_mean_angle×1, smooth_angle_min×1.

Each head's pass draws ONLY from its release-eligible subset and tie-breaks on its OWN `p_ge3`. Marginal gain = niche-relative quality (within-niche p_ge3 percentile) × coverage gain (1 − max morph-CLIP cos to already-selected; the strange pass adds a same-mode floor of 0.5). `rk%` = the location's pref_loc_v0 percentile among admitted; `nearest` = the closest already-selected wallpaper (displacement).

| # | id | type/cluster | flavor/style | p_ge3 | niche% | rk% | cov.gain | nearest (sim) |
|--:|---|---|---|--:|--:|--:|--:|---|
| 1 | em_000427 | multibrot5/multibrot5#3 | k16:14/smooth | 1.000 | 1.00 | 0.69 | 1.00 | — |
| 2 | em_001172 | julia:multibrot5/julia:multibrot5#67 | k16:11/smooth | 0.966 | 1.00 | 0.16 | 0.22 | em_000427 (0.78) |
| 3 | em_001219 | mandelbrot/mandelbrot#36 | k16:3/smooth | 0.968 | 1.00 | 0.12 | 0.19 | em_001172 (0.81) |
| 4 | em_001370 | phoenix/phoenix#165 | k16:10/smooth | 0.971 | 1.00 | 0.01 | 0.18 | em_000427 (0.81) |
| 5 | em_001205 | phoenix/phoenix#188 | special:neutral/smooth | 0.988 | 1.00 | 0.13 | 0.17 | em_000427 (0.83) |
| 6 | em_000693 | julia:multibrot5/julia:multibrot5#10 | special:spectral/smooth | 0.922 | 1.00 | 0.50 | 0.17 | em_000427 (0.83) |
| 7 | em_000915 | mandelbrot/mandelbrot#202 | k16:4/smooth | 0.986 | 1.00 | 0.34 | 0.16 | em_000427 (0.84) |
| 8 | em_000606 | mandelbrot/mandelbrot#1 | k16:9/smooth | 0.999 | 1.00 | 0.56 | 0.16 | em_000427 (0.84) |
| 9 | em_001380 | phoenix/phoenix#46 | k16:16/smooth | 0.940 | 1.00 | 0.01 | 0.14 | em_001205 (0.86) |
| 10 | em_001099 | multibrot5/multibrot5#197 | k16:13/smooth | 0.979 | 1.00 | 0.21 | 0.14 | em_001219 (0.86) |
| 11 | em_000593 | julia:mandelbrot/julia:mandelbrot#24 | k16:13/smooth | 0.964 | 1.00 | 0.57 | 0.14 | em_000427 (0.86) |
| 12 | em_000282 | mandelbrot/mandelbrot#101 | k16:15/smooth | 0.982 | 1.00 | 0.80 | 0.13 | em_001172 (0.87) |
| 13 | em_001240 | phoenix/phoenix#20 | k16:12/smooth | 0.959 | 1.00 | 0.11 | 0.13 | em_001380 (0.87) |
| 14 | em_001273 | phoenix/phoenix#170 | k16:9/smooth | 0.980 | 1.00 | 0.08 | 0.13 | em_001205 (0.87) |
| 15 | em_001028 | multibrot3/multibrot3#175 | k16:14/smooth | 0.946 | 1.00 | 0.26 | 0.12 | em_000427 (0.88) |
| 16 | em_000465 | multibrot3/multibrot3#94 | k16:4/smooth | 0.997 | 1.00 | 0.66 | 0.12 | em_001099 (0.88) |
| 17 | em_000449 | multibrot5/multibrot5#102 | k16:11/smooth | 0.994 | 1.00 | 0.68 | 0.12 | em_001219 (0.88) |
| 18 | em_001297 | phoenix/phoenix#137 | k16:11/smooth | 1.000 | 1.00 | 0.06 | 0.12 | em_000427 (0.88) |
| 19 | em_001278 | phoenix/phoenix#36 | k16:15/smooth | 0.996 | 1.00 | 0.08 | 0.11 | em_001099 (0.89) |
| 20 | em_000799 | phoenix/phoenix#232 | k16:11/smooth | 0.977 | 1.00 | 0.42 | 0.11 | em_000427 (0.89) |
| 21 | em_000021 | multibrot5/multibrot5#180 | k16:6/smooth | 0.927 | 1.00 | 0.98 | 0.11 | em_000427 (0.89) |
| 22 | em_000049 | julia:multibrot3/julia:multibrot3#43 | k16:7/smooth | 0.960 | 1.00 | 0.96 | 0.11 | em_001240 (0.89) |
| 23 | em_001306 | phoenix/phoenix#94 | k16:5/smooth | 0.984 | 1.00 | 0.06 | 0.10 | em_001240 (0.90) |
| 24 | em_000250 | multibrot4/multibrot4#104 | k16:15/smooth | 0.991 | 1.00 | 0.82 | 0.10 | em_000606 (0.90) |
| 25 | em_001030 | julia:multibrot3/julia:multibrot3#7 | k16:14/smooth | 0.984 | 1.00 | 0.26 | 0.10 | em_000282 (0.90) |
| 26 | em_000579 | mandelbrot/mandelbrot#168 | k16:6/composite_c13_smooth_stripe | 0.938 | 1.00 | 0.58 | 1.00 | — |
| 27 | em_001281 | phoenix/phoenix#63 | k16:7/composite_c13_smooth_stripe | 0.502 | 1.00 | 0.08 | 0.21 | em_000579 (0.79) |
| 28 | em_001029 | multibrot5/multibrot5#6 | k16:1/composite_c17_smooth_curvature | 0.529 | 1.00 | 0.26 | 0.21 | em_000579 (0.79) |
| 29 | em_001060 | phoenix/phoenix#234 | k16:2/smooth_mean_angle | 0.836 | 1.00 | 0.24 | 0.21 | em_000579 (0.79) |
| 30 | em_000374 | multibrot3/multibrot3#145 | k16:7/tia | 0.772 | 1.00 | 0.73 | 0.20 | em_001060 (0.80) |
| 31 | em_001097 | multibrot5/multibrot5#201 | k16:8/tia | 0.509 | 1.00 | 0.21 | 0.20 | em_000579 (0.80) |
| 32 | em_001104 | phoenix/phoenix#164 | special:neutral/stripe | 0.596 | 1.00 | 0.20 | 0.18 | em_000579 (0.82) |
| 33 | em_001215 | phoenix/phoenix#219 | k16:9/smooth_angle_min | 0.593 | 1.00 | 0.12 | 0.17 | em_000579 (0.83) |
| 34 | em_001341 | phoenix/phoenix#93 | k16:1/composite_c17_smooth_curvature | 0.588 | 1.00 | 0.03 | 0.16 | em_001104 (0.84) |
| 35 | em_000249 | multibrot4/multibrot4#38 | k16:6/composite_c13_smooth_stripe | 0.531 | 1.00 | 0.82 | 0.16 | em_000374 (0.84) |
| 36 | em_000645 | julia:multibrot3/julia:multibrot3#44 | k16:8/tia | 0.750 | 1.00 | 0.53 | 0.15 | em_000579 (0.85) |
| 37 | em_000220 | julia:multibrot4/julia:multibrot4#22 | k16:5/tia | 0.645 | 1.00 | 0.84 | 0.15 | em_000579 (0.85) |
| 38 | em_001293 | phoenix/phoenix#152 | k16:8/composite_c17_smooth_curvature | 0.559 | 1.00 | 0.07 | 0.15 | em_001029 (0.85) |
| 39 | em_001007 | multibrot5/multibrot5#109 | k16:14/tia | 0.688 | 1.00 | 0.27 | 0.14 | em_001029 (0.86) |
| 40 | em_000244 | julia:multibrot3/julia:multibrot3#14 | k16:10/stripe | 0.713 | 1.00 | 0.82 | 0.14 | em_001007 (0.86) |
| 41 | em_000760 | multibrot4/multibrot4#10 | k16:8/composite_c13_smooth_stripe | 0.662 | 1.00 | 0.45 | 0.13 | em_000220 (0.87) |
| 42 | em_000553 | multibrot3/multibrot3#89 | k16:9/composite_c13_smooth_stripe | 0.810 | 1.00 | 0.60 | 0.12 | em_000249 (0.88) |
| 43 | em_000294 | mandelbrot/mandelbrot#131 | k16:7/stripe | 0.594 | 1.00 | 0.79 | 0.12 | em_000645 (0.88) |
| 44 | em_000200 | multibrot5/multibrot5#234 | k16:8/tia | 0.765 | 1.00 | 0.86 | 0.12 | em_000249 (0.88) |
| 45 | em_001155 | multibrot5/multibrot5#133 | special:outlier/tia | 0.738 | 1.00 | 0.17 | 0.11 | em_000645 (0.89) |
| 46 | em_001307 | phoenix/phoenix#45 | k16:10/composite_c17_smooth_curvature | 0.718 | 1.00 | 0.06 | 0.11 | em_000579 (0.89) |
| 47 | em_000510 | multibrot5/multibrot5#77 | k16:5/tia | 0.726 | 1.00 | 0.63 | 0.10 | em_001007 (0.90) |
| 48 | em_000281 | julia:mandelbrot/julia:mandelbrot#38 | k16:9/tia | 0.530 | 1.00 | 0.80 | 0.10 | em_000553 (0.90) |
| 49 | em_000987 | phoenix/phoenix#225 | k16:12/stripe | 0.920 | 1.00 | 0.29 | 0.10 | em_001060 (0.90) |
| 50 | em_000626 | julia:multibrot4/julia:multibrot4#62 | k16:5/tia | 0.597 | 1.00 | 0.55 | 0.10 | em_000553 (0.90) |

### vs the v1 release (no release floors) — side-by-side

v1 selected from ALL gated pool rows (permissive floor only). Reconstructed here by the same greedy select over the durable v1 pool, annotated with which picks would now fall BELOW their head's release floor (→ inventory, not a release).

| v1 pick | type/style | p_ge3 | ≥ release floor? |
|---|---|--:|---|
| em_000000 | julia:mandelbrot/smooth | 0.949 | ✓ (0.95 ≥ 0.9) |
| em_000053 | multibrot5/smooth | 0.945 | ✓ (0.94 ≥ 0.9) |
| em_000036 | multibrot4/tia | 0.651 | ✓ (0.65 ≥ 0.5) |
| em_000039 | multibrot5/composite_c13_smooth_stripe | 0.615 | ✓ (0.61 ≥ 0.5) |
| em_000035 | multibrot4/composite_c17_smooth_curvature | 0.426 | ✗ 0.43 BELOW 0.5 → inventory |
| em_000075 | multibrot3/tia | 0.408 | ✗ 0.41 BELOW 0.5 → inventory |
| em_000042 | multibrot5/stripe | 0.365 | ✗ 0.37 BELOW 0.5 → inventory |
| em_000003 | julia:multibrot3/stripe | 0.357 | ✗ 0.36 BELOW 0.5 → inventory |
| em_000046 | multibrot5/stripe | 0.346 | ✗ 0.35 BELOW 0.5 → inventory |
| em_000057 | julia:multibrot3/tia | 0.315 | ✗ 0.32 BELOW 0.5 → inventory |
| em_000009 | mandelbrot/composite_c13_smooth_stripe | 0.292 | ✗ 0.29 BELOW 0.5 → inventory |
| em_000054 | julia:mandelbrot/stripe | 0.259 | ✗ 0.26 BELOW 0.5 → inventory |
| em_000047 | multibrot5/composite_c13_smooth_stripe | 0.230 | ✗ 0.23 BELOW 0.5 → inventory |
| em_000056 | julia:multibrot3/smooth_angle_min | 0.226 | ✗ 0.23 BELOW 0.5 → inventory |
| em_000074 | multibrot3/smooth_angle_min | 0.224 | ✗ 0.22 BELOW 0.5 → inventory |
| em_000058 | julia:multibrot3/stripe | 0.213 | ✗ 0.21 BELOW 0.5 → inventory |
| em_000031 | multibrot4/smooth_mean_angle | 0.155 | ✗ 0.15 BELOW 0.5 → inventory |
| em_000062 | mandelbrot/composite_c7_smooth_trap_circle | 0.153 | ✗ 0.15 BELOW 0.5 → inventory |
| em_000017 | mandelbrot/stripe | 0.137 | ✗ 0.14 BELOW 0.5 → inventory |
| em_000006 | julia:multibrot4/smooth_angle_min | 0.125 | ✗ 0.13 BELOW 0.5 → inventory |
| em_000032 | multibrot4/smooth_angle_min | 0.117 | ✗ 0.12 BELOW 0.5 → inventory |
| em_000034 | multibrot4/composite_c17_smooth_curvature | 0.116 | ✗ 0.12 BELOW 0.5 → inventory |
| em_000025 | multibrot3/stripe | 0.110 | ✗ 0.11 BELOW 0.5 → inventory |
| em_000065 | mandelbrot/composite_c13_smooth_stripe | 0.108 | ✗ 0.11 BELOW 0.5 → inventory |
| em_000037 | multibrot4/composite_c17_smooth_curvature | 0.106 | ✗ 0.11 BELOW 0.5 → inventory |
| em_000004 | julia:multibrot3/smooth_mean_angle | 0.088 | ✗ 0.09 BELOW 0.5 → inventory |
| em_000079 | multibrot3/composite_c13_smooth_stripe | 0.087 | ✗ 0.09 BELOW 0.5 → inventory |
| em_000027 | multibrot3/composite_c7_smooth_trap_circle | 0.086 | ✗ 0.09 BELOW 0.5 → inventory |
| em_000063 | mandelbrot/composite_c17_smooth_curvature | 0.076 | ✗ 0.08 BELOW 0.5 → inventory |
| em_000012 | mandelbrot/stripe | 0.074 | ✗ 0.07 BELOW 0.5 → inventory |
| em_000066 | mandelbrot/composite_c13_smooth_stripe | 0.067 | ✗ 0.07 BELOW 0.5 → inventory |
| em_000020 | multibrot3/smooth_angle_min | 0.065 | ✗ 0.06 BELOW 0.5 → inventory |
| em_000014 | mandelbrot/composite_c13_smooth_stripe | 0.063 | ✗ 0.06 BELOW 0.5 → inventory |
| em_000005 | julia:multibrot3/stripe | 0.060 | ✗ 0.06 BELOW 0.5 → inventory |
| em_000049 | multibrot5/composite_c13_smooth_stripe | 0.058 | ✗ 0.06 BELOW 0.5 → inventory |
| em_000021 | multibrot3/stripe | 0.055 | ✗ 0.06 BELOW 0.5 → inventory |

**32/36** v1 picks now drop to inventory under the release floors — the sub-floor tiles the v1 permissive-only bar shipped.

## Contact sheets

- `out\first_release\release_sheet.png` — the 50-wallpaper release
- `out\first_release\pool_sheet.png` — the gated pool grouped by niche
