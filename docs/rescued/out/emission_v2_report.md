# Emission — diversity-aware emission (deficit colorize + ranker-ordered intake + per-head release floors)

Source ledger(s): `data\discovery\steered_run2\outcome_ledger.jsonl`, `data\discovery\steered_v1_2_dive\outcome_ledger.jsonl`.

Location ranker (pref_loc_v0, **logi:v7+colored**) ORDERS the colorize queue (order, not filter — diversity supply untouched). Pool floors (permissive): wallpaper **0.75** / mining **0.25**. **Release floors** (per head, distinct): wallpaper **0.9** / mining **0.5**. Release N=**12** · target **36** release-eligible (post-floor surplus).

## Intake — morph clusters among admitted locations

- **96** admitted locations (current-decode ∧ decoded_class==3 ∧ guard_pass ∧ distinct)
- **90** morph clusters (within-type, cos>0.974) across **8** fractal types:
  - `julia:mandelbrot`: 1 locations → 1 clusters
  - `julia:multibrot3`: 8 locations → 8 clusters
  - `julia:multibrot4`: 1 locations → 1 clusters
  - `julia:multibrot5`: 1 locations → 1 clusters
  - `mandelbrot`: 21 locations → 19 clusters
  - `multibrot3`: 18 locations → 15 clusters
  - `multibrot4`: 18 locations → 18 clusters
  - `multibrot5`: 28 locations → 27 clusters

## Niche occupancy + deficit (before → after)

- feasible cells: **13680** ((type,cluster) × 19 flavors × 8 styles)
- BEFORE (empty pool): 0 populated, deficit = uniform target over all 13680 feasible cells
- AFTER: **27** distinct cells populated by the 27-wallpaper gated pool; **0** cells hit the attempt cap and left support
- **27** distinct cells did the 27-surplus populate (out of 13680 feasible).
- axis coverage in the gated pool: **5** types · **20** morph clusters · **14**/19 palette flavors · **7**/8 render styles
  - render styles present: tia×7, stripe×6, smooth×5, composite_c13_smooth_stripe×5, smooth_angle_min×2, composite_c17_smooth_curvature×1, composite_c7_smooth_trap_circle×1

## Pool inventory + per-head release floors

Render styles route to two heads: **smooth → wallpaper head** (pool floor 0.75, **release floor 0.9**); **strange → mining head** (pool floor 0.25, **release floor 0.5**). Quality is only compared within a niche, which pins the style/head, so the heads never mix. Pool admission is permissive (weak wallpapers persist as inventory); SELECTION only draws above the release floor.

- attempts: **203** · pool-admitted (gated): **27** → pool pass rate **13.3%** · render errors: 0

| head | pool-admitted | release-eligible | inventory (below release floor) |
|---|--:|--:|--:|
| wallpaper (smooth, rel≥0.9) | 5 | 4 | 1 |
| mining (strange, rel≥0.5) | 22 | 6 | 16 |
| **total** | **27** | **10** | **17** |

**17/27** pool wallpapers are banked as inventory below their head's release floor — exactly the weak tiles the v1 permissive-only bar would have let compete for a release slot. The colorize targeted **36** release-eligible (post-floor) and reached **10**.

## Ranker reach — did ranked intake concentrate budget on good locations?

Admitted locations ordered by pref_loc_v0 score (desc); 'reach' = the deepest rank the colorize actually touched. If ranked intake works, colorize fills its surplus from the TOP of the ordering and never has to reach deep.

- 96 admitted locations; **96** got a colorize attempt, reaching rank **96** (top 100% of the ordering).
- **9** locations contributed a release-eligible wallpaper, the deepest at rank **85** (top 89%).
- reading: the surplus was filled within the top **89%** of ranker-ordered locations — ranked intake concentrated budget on the good end.

## Colorizer choice — deficit-driven palette/style spread

Chosen palette-flavor distribution over 203 colorize attempts vs the uniform-random expectation (10.7/flavor):

| palette flavor | chosen | uniform-random |
|---|---:|---:|
| special:outlier | 15 | 10.7 |
| special:spectral | 15 | 10.7 |
| k16:1 | 14 | 10.7 |
| special:neutral | 14 | 10.7 |
| k16:9 | 13 | 10.7 |
| k16:7 | 13 | 10.7 |
| k16:14 | 12 | 10.7 |
| k16:5 | 12 | 10.7 |
| k16:4 | 11 | 10.7 |
| k16:16 | 11 | 10.7 |
| k16:11 | 10 | 10.7 |
| k16:3 | 10 | 10.7 |
| k16:6 | 9 | 10.7 |
| k16:2 | 8 | 10.7 |
| k16:15 | 8 | 10.7 |
| k16:13 | 8 | 10.7 |
| k16:10 | 8 | 10.7 |
| k16:12 | 6 | 10.7 |
| k16:8 | 6 | 10.7 |

Render-style distribution (uniform-random 25.4/style):

| render style | chosen |
|---|---:|
| composite_c7_smooth_trap_circle | 34 |
| stripe | 30 |
| composite_c17_smooth_curvature | 27 |
| smooth_mean_angle | 26 |
| composite_c13_smooth_stripe | 25 |
| smooth_angle_min | 23 |
| tia | 21 |
| smooth | 17 |

## Release selection — 10 picks (greedy max-marginal-gain) — **SHORT-FILL 10/12**: only 10 pool rows clear the release floors; shipping fewer rather than dipping below the floor

Selection draws ONLY from the release-eligible subset (per-head floor). Marginal gain = niche-relative quality (within-niche p_ge3 percentile) × coverage gain (1 − max similarity to already-selected under the per-axis kernel). `rk%` = the location's pref_loc_v0 percentile among admitted; `nearest` = the closest already-selected wallpaper (displacement).

| # | id | type/cluster | flavor/style | p_ge3 | niche% | rk% | cov.gain | nearest (sim) |
|--:|---|---|---|--:|--:|--:|--:|---|
| 1 | em_000008 | multibrot4/multibrot4#5 | k16:6/smooth | 1.000 | 1.00 | 0.92 | 1.00 | — |
| 2 | em_000112 | multibrot5/multibrot5#24 | k16:14/smooth | 0.997 | 1.00 | 0.83 | 1.00 | em_000008 (0.00) |
| 3 | em_000000 | julia:multibrot3/julia:multibrot3#6 | k16:6/smooth | 0.945 | 1.00 | 1.00 | 1.00 | em_000008 (0.00) |
| 4 | em_000038 | multibrot3/multibrot3#12 | k16:7/smooth | 0.943 | 1.00 | 0.60 | 1.00 | em_000008 (0.00) |
| 5 | em_000044 | multibrot4/multibrot4#11 | k16:3/tia | 0.804 | 1.00 | 0.54 | 1.00 | em_000008 (0.00) |
| 6 | em_000084 | multibrot4/multibrot4#17 | k16:5/composite_c13_smooth_stripe | 0.666 | 1.00 | 0.12 | 1.00 | em_000008 (0.00) |
| 7 | em_000005 | multibrot4/multibrot4#2 | special:outlier/stripe | 0.660 | 1.00 | 0.95 | 1.00 | em_000008 (0.00) |
| 8 | em_000104 | multibrot4/multibrot4#5 | k16:11/composite_c13_smooth_stripe | 0.622 | 1.00 | 0.92 | 1.00 | em_000008 (0.00) |
| 9 | em_000129 | multibrot3/multibrot3#9 | k16:2/tia | 0.610 | 1.00 | 0.66 | 1.00 | em_000008 (0.00) |
| 10 | em_000176 | multibrot4/multibrot4#14 | special:neutral/composite_c13_smooth_stripe | 0.547 | 1.00 | 0.17 | 1.00 | em_000008 (0.00) |

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

**8/12** v1 picks now drop to inventory under the release floors — the sub-floor tiles the v1 permissive-only bar shipped.

## Contact sheets

- `out\emission_v2\release_sheet.png` — the 10-wallpaper release
- `out\emission_v2\pool_sheet.png` — the gated pool grouped by niche
