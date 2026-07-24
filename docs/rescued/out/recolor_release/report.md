# Emission — diversity-aware emission (deficit colorize + ranker-ordered intake + per-head release floors)

Source ledger(s): `out\recolor_release\ledgers\c1__breadth.jsonl`, `out\recolor_release\ledgers\c1__dive.jsonl`, `data\discovery\campaign2\breadth\outcome_ledger.jsonl`, `data\discovery\campaign2\dive\outcome_ledger.jsonl`, `data\discovery\phoenix_grid\grid\outcome_ledger_v7_t45.jsonl`, `data\discovery\classic_phoenix\outcome_ledger.jsonl`.

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
- AFTER: **725** distinct cells populated by the 725-wallpaper gated pool; **0** cells hit the attempt cap and left support
- **725** distinct cells did the 725-surplus populate (out of 192736 feasible).
- axis coverage in the gated pool: **9** types · **698** morph clusters · **19**/19 palette flavors · **8**/8 render styles
  - render styles present: tia×139, stripe×134, composite_c13_smooth_stripe×129, smooth_angle_min×86, composite_c17_smooth_curvature×67, smooth×65, composite_c7_smooth_trap_circle×55, smooth_mean_angle×50

## Pool inventory + per-head release floors

Render styles route to two heads: **smooth → wallpaper head** (pool floor 0.75, **release floor 0.9**); **strange → mining head** (pool floor 0.05, **release floor 0.5**). Quality is only compared within a niche, which pins the style/head, so the heads never mix. Pool admission is permissive (weak wallpapers persist as inventory); SELECTION only draws above the release floor.

- attempts: **1387** · pool-admitted (gated): **725** → pool pass rate **52.3%** · render errors: 0

| head | pool-admitted | release-eligible | inventory (below release floor) |
|---|--:|--:|--:|
| wallpaper (smooth, rel≥0.9) | 65 | 54 | 11 |
| mining (strange, rel≥0.5) | 660 | 58 | 602 |
| **total** | **725** | **112** | **613** |

**613/725** pool wallpapers are banked as inventory below their head's release floor — exactly the weak tiles the v1 permissive-only bar would have let compete for a release slot. The colorize targeted **150** release-eligible (post-floor) and reached **112**.

## Ranker reach — did ranked intake concentrate budget on good locations?

Admitted locations ordered by pref_loc_v0 score (desc); 'reach' = the deepest rank the colorize actually touched. If ranked intake works, colorize fills its surplus from the TOP of the ordering and never has to reach deep.

- 1387 admitted locations; **1387** got a colorize attempt, reaching rank **1387** (top 100% of the ordering).
- **112** locations contributed a release-eligible wallpaper, the deepest at rank **1385** (top 100%).
- reading: the surplus was filled within the top **100%** of ranker-ordered locations (reached deep — pool is quality-thin, not a ranking failure).

## Colorizer choice — deficit-driven palette/style spread

Chosen palette-flavor distribution over 1387 colorize attempts vs the uniform-random expectation (73.0/flavor):

| palette flavor | chosen | uniform-random |
|---|---:|---:|
| k16:9 | 87 | 73.0 |
| k16:4 | 84 | 73.0 |
| special:neutral | 81 | 73.0 |
| k16:16 | 78 | 73.0 |
| k16:15 | 77 | 73.0 |
| k16:1 | 76 | 73.0 |
| k16:7 | 76 | 73.0 |
| k16:11 | 76 | 73.0 |
| k16:5 | 73 | 73.0 |
| k16:8 | 73 | 73.0 |
| k16:12 | 72 | 73.0 |
| k16:3 | 72 | 73.0 |
| special:outlier | 69 | 73.0 |
| k16:2 | 69 | 73.0 |
| k16:6 | 68 | 73.0 |
| special:spectral | 66 | 73.0 |
| k16:10 | 65 | 73.0 |
| k16:14 | 63 | 73.0 |
| k16:13 | 62 | 73.0 |

Render-style distribution (uniform-random 173.4/style):

| render style | chosen |
|---|---:|
| smooth_angle_min | 187 |
| composite_c7_smooth_trap_circle | 186 |
| stripe | 180 |
| tia | 173 |
| smooth_mean_angle | 173 |
| smooth | 166 |
| composite_c13_smooth_stripe | 165 |
| composite_c17_smooth_curvature | 157 |

## Release selection — 50 picks (greedy max-marginal-gain)

**Render-mode split (heads never compared in one step).** Smooth slots are filled from the wallpaper head, strange from the mining head, by two DISJOINT within-head greedy passes. Target strange frac **0.5** → slots smooth **25** / strange **25**. Eligible: smooth **54** / strange **58**. Realized: smooth **25** / strange **25** (strange frac **0.50**). Strange modes: tia×12, stripe×5, composite_c13_smooth_stripe×5, composite_c17_smooth_curvature×2, smooth_angle_min×1.

Each head's pass draws ONLY from its release-eligible subset and tie-breaks on its OWN `p_ge3`. Marginal gain = niche-relative quality (within-niche p_ge3 percentile) × coverage gain (1 − max morph-CLIP cos to already-selected; the strange pass adds a same-mode floor of 0.5). `rk%` = the location's pref_loc_v0 percentile among admitted; `nearest` = the closest already-selected wallpaper (displacement).

| # | id | type/cluster | flavor/style | p_ge3 | niche% | rk% | cov.gain | nearest (sim) |
|--:|---|---|---|--:|--:|--:|--:|---|
| 1 | em_000428 | julia:multibrot4/julia:multibrot4#0 | k16:11/smooth | 1.000 | 1.00 | 0.69 | 1.00 | — |
| 2 | em_001370 | phoenix/phoenix#165 | k16:9/smooth | 0.931 | 1.00 | 0.01 | 0.26 | em_000428 (0.74) |
| 3 | em_000795 | multibrot3/multibrot3#181 | k16:16/smooth | 0.937 | 1.00 | 0.43 | 0.24 | em_001370 (0.76) |
| 4 | em_001258 | phoenix/phoenix#120 | special:neutral/smooth | 0.991 | 1.00 | 0.09 | 0.20 | em_000428 (0.80) |
| 5 | em_001079 | phoenix/phoenix#26 | special:neutral/smooth | 0.996 | 1.00 | 0.22 | 0.17 | em_001258 (0.83) |
| 6 | em_000686 | julia:mandelbrot/julia:mandelbrot#48 | k16:8/smooth | 0.923 | 1.00 | 0.51 | 0.15 | em_000428 (0.85) |
| 7 | em_001184 | multibrot5/multibrot5#111 | special:neutral/smooth | 0.957 | 1.00 | 0.15 | 0.15 | em_001370 (0.85) |
| 8 | em_000983 | multibrot3/multibrot3#78 | k16:11/smooth | 0.929 | 1.00 | 0.29 | 0.14 | em_001370 (0.86) |
| 9 | em_001325 | phoenix/phoenix#91 | k16:16/smooth | 0.993 | 1.00 | 0.04 | 0.14 | em_001258 (0.86) |
| 10 | em_000849 | multibrot5/multibrot5#53 | special:neutral/smooth | 0.998 | 1.00 | 0.39 | 0.13 | em_001258 (0.87) |
| 11 | em_001055 | phoenix/phoenix#229 | special:neutral/smooth | 0.989 | 1.00 | 0.24 | 0.13 | em_000686 (0.87) |
| 12 | em_001056 | mandelbrot/mandelbrot#119 | k16:14/smooth | 0.911 | 1.00 | 0.24 | 0.12 | em_000686 (0.88) |
| 13 | em_000597 | multibrot3/multibrot3#18 | k16:15/smooth | 0.989 | 1.00 | 0.57 | 0.12 | em_001258 (0.88) |
| 14 | em_000406 | mandelbrot/mandelbrot#183 | k16:14/smooth | 0.995 | 1.00 | 0.71 | 0.12 | em_000983 (0.88) |
| 15 | em_000778 | multibrot3/multibrot3#24 | k16:11/smooth | 0.985 | 1.00 | 0.44 | 0.12 | em_000795 (0.88) |
| 16 | em_000030 | mandelbrot/mandelbrot#129 | k16:7/smooth | 0.994 | 1.00 | 0.98 | 0.12 | em_000849 (0.89) |
| 17 | em_000599 | multibrot4/multibrot4#3 | k16:14/smooth | 0.903 | 1.00 | 0.57 | 0.11 | em_000795 (0.89) |
| 18 | em_000541 | multibrot3/multibrot3#28 | k16:15/smooth | 0.996 | 1.00 | 0.61 | 0.11 | em_001184 (0.89) |
| 19 | em_001007 | multibrot5/multibrot5#109 | k16:14/smooth | 0.991 | 1.00 | 0.27 | 0.11 | em_000597 (0.89) |
| 20 | em_000142 | julia:multibrot5/julia:multibrot5#57 | special:neutral/smooth | 0.916 | 1.00 | 0.90 | 0.10 | em_000428 (0.90) |
| 21 | em_001369 | phoenix/phoenix#50 | k16:8/smooth | 0.937 | 1.00 | 0.01 | 0.10 | em_001325 (0.90) |
| 22 | em_001160 | phoenix/phoenix#69 | k16:6/smooth | 0.946 | 1.00 | 0.16 | 0.09 | em_000795 (0.91) |
| 23 | em_001310 | phoenix/phoenix#32 | k16:16/smooth | 0.959 | 1.00 | 0.06 | 0.09 | em_001258 (0.91) |
| 24 | em_000041 | multibrot3/multibrot3#182 | k16:15/smooth | 0.966 | 1.00 | 0.97 | 0.09 | em_001056 (0.91) |
| 25 | em_000063 | multibrot5/multibrot5#181 | k16:5/smooth | 0.992 | 1.00 | 0.95 | 0.09 | em_000849 (0.91) |
| 26 | em_000396 | multibrot4/multibrot4#100 | k16:8/tia | 0.837 | 1.00 | 0.71 | 1.00 | — |
| 27 | em_000179 | mandelbrot/mandelbrot#53 | k16:8/tia | 0.564 | 1.00 | 0.87 | 0.28 | em_000396 (0.72) |
| 28 | em_001275 | phoenix/phoenix#37 | k16:5/stripe | 0.757 | 1.00 | 0.08 | 0.16 | em_000396 (0.84) |
| 29 | em_000489 | mandelbrot/mandelbrot#124 | k16:5/tia | 0.623 | 1.00 | 0.65 | 0.16 | em_000396 (0.84) |
| 30 | em_001099 | multibrot5/multibrot5#197 | k16:3/tia | 0.599 | 1.00 | 0.21 | 0.15 | em_000396 (0.85) |
| 31 | em_000674 | phoenix/phoenix#199 | k16:10/stripe | 0.522 | 1.00 | 0.51 | 0.14 | em_000396 (0.86) |
| 32 | em_000219 | multibrot3/multibrot3#56 | k16:8/smooth_angle_min | 0.682 | 1.00 | 0.84 | 0.13 | em_001099 (0.87) |
| 33 | em_000162 | multibrot3/multibrot3#42 | k16:3/stripe | 0.586 | 1.00 | 0.88 | 0.12 | em_000489 (0.88) |
| 34 | em_001017 | julia:multibrot3/julia:multibrot3#56 | k16:10/composite_c13_smooth_stripe | 0.723 | 1.00 | 0.27 | 0.12 | em_000489 (0.88) |
| 35 | em_001320 | phoenix/phoenix#90 | k16:1/composite_c17_smooth_curvature | 0.791 | 1.00 | 0.05 | 0.11 | em_000396 (0.89) |
| 36 | em_000089 | multibrot5/multibrot5#199 | k16:7/composite_c13_smooth_stripe | 0.717 | 1.00 | 0.94 | 0.11 | em_000396 (0.89) |
| 37 | em_000213 | multibrot5/multibrot5#214 | k16:7/composite_c13_smooth_stripe | 0.620 | 1.00 | 0.85 | 0.11 | em_000396 (0.89) |
| 38 | em_001323 | phoenix/phoenix#34 | k16:11/composite_c13_smooth_stripe | 0.566 | 1.00 | 0.05 | 0.10 | em_001275 (0.90) |
| 39 | em_000586 | multibrot3/multibrot3#71 | k16:5/tia | 0.807 | 1.00 | 0.58 | 0.10 | em_000489 (0.90) |
| 40 | em_000253 | mandelbrot/mandelbrot#170 | k16:8/composite_c13_smooth_stripe | 0.721 | 1.00 | 0.82 | 0.10 | em_000396 (0.90) |
| 41 | em_001344 | phoenix/phoenix#128 | k16:1/stripe | 0.550 | 1.00 | 0.03 | 0.09 | em_000489 (0.91) |
| 42 | em_000760 | multibrot4/multibrot4#10 | k16:15/tia | 0.540 | 1.00 | 0.45 | 0.09 | em_000162 (0.91) |
| 43 | em_000626 | julia:multibrot4/julia:multibrot4#62 | k16:7/tia | 0.601 | 1.00 | 0.55 | 0.09 | em_000162 (0.91) |
| 44 | em_000267 | multibrot5/multibrot5#81 | k16:11/tia | 0.600 | 1.00 | 0.81 | 0.09 | em_000396 (0.91) |
| 45 | em_000435 | phoenix/phoenix#208 | k16:3/composite_c17_smooth_curvature | 0.537 | 1.00 | 0.69 | 0.08 | em_000674 (0.92) |
| 46 | em_001174 | julia:multibrot3/julia:multibrot3#29 | k16:9/stripe | 0.502 | 1.00 | 0.15 | 0.08 | em_000396 (0.92) |
| 47 | em_000504 | multibrot3/multibrot3#9 | k16:1/tia | 0.552 | 1.00 | 0.64 | 0.08 | em_000586 (0.92) |
| 48 | em_001113 | phoenix/phoenix#151 | k16:8/tia | 0.799 | 1.00 | 0.20 | 0.08 | em_001275 (0.92) |
| 49 | em_000001 | julia:multibrot5/julia:multibrot5#21 | k16:14/tia | 0.570 | 1.00 | 1.00 | 0.08 | em_000396 (0.92) |
| 50 | em_000352 | multibrot5/multibrot5#83 | k16:5/tia | 0.633 | 1.00 | 0.75 | 0.08 | em_000219 (0.92) |

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

- `out\recolor_release\release_sheet.png` — the 50-wallpaper release
- `out\recolor_release\pool_sheet.png` — the gated pool grouped by niche
