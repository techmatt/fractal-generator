# First release — render-mode-split re-selection readout

Re-selection over the existing gated pool (no re-colorize, no pool re-render, no measure edit). Released **50** wallpapers. Reads durable artifacts + the rendered `release/` dir.

## 1. Realized render-mode split (heads never compared in one step)

Smooth slots filled from the **wallpaper head** (rel ≥ 0.9), strange from the **mining head** (rel ≥ 0.5), by two DISJOINT within-head greedy passes — the two heads' scores never enter the same comparison.

- target strange frac **0.5** → slots smooth **25** / strange **25**
- eligible (above head floor): smooth **65** / strange **82**
- **realized: smooth 25 / strange 25** (strange frac **0.50**, target 0.5)

### per-mode counts among the released

| render mode | head | count in release |
|---|---|--:|
| smooth | wallpaper | 25 |
| tia | mining | 10 |
| composite_c13_smooth_stripe | mining | 5 |
| stripe | mining | 4 |
| composite_c17_smooth_curvature | mining | 4 |
| smooth_mean_angle | mining | 1 |
| smooth_angle_min | mining | 1 |

## 2. Morph-diversity check — pairwise morph-CLIP cos among the released

Did the continuous-cos coverage term actually spread the spirals? Distribution over all 1225 released pairs (0 = orthogonal look, 1 = identical). See `out/first_release/release_morph_diversity.png`.

| quantile | pairwise cos |
|---|--:|
| p50 | 0.818 |
| p75 | 0.848 |
| p90 | 0.872 |
| p95 | 0.887 |
| p99 | 0.919 |
| p100 | 0.950 |

- mean **0.814**, max **0.950**
- nearest released pair: **em_000760** ↔ **em_001030** at cos **0.950**
- pairs above 0.9 (near-duplicate look): **30** / 1225

**Reading:** the coverage term is non-inert — it holds the released set's pairwise look-similarity with a p95 of 0.89; the old categorical-gated kernel could not see cross-cell duplicates at all.

## 3. Strange-candidates sheet — realizable strange supply

- strange pool tiles ≥ 0.5 mining release floor: **82**
- of which released (★): **25**
- by mode: tia×28, stripe×22, composite_c13_smooth_stripe×14, composite_c17_smooth_curvature×9, smooth_angle_min×4, smooth_mean_angle×3, composite_c7_smooth_trap_circle×2

`out/first_release/strange_candidates_sheet.png` — all 82 ranked by mining p_ge3 at deploy fidelity.
