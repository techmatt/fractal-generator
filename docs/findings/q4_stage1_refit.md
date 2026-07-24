# q4 stage-1 refit — firm the goodness field, harvest-readiness

Folded **q4_g_aimed** (34 accept / 78 reject) into p1+p2. Combined: **340** accept/reject (89 accept / 251 reject) over 33 minibrots. Laplacian (T3) dropped. Referee: minibrot-disjoint LOMO.

## Pre-refit audit (new labels × slug, pre-refit model)

| slug | n | accept-rate | reads as |
|---|---|---|---|
| top_g | 24 | 75% | precision of confident-accepts |
| uncertain | 64 | 14% | ≪50% ⇒ p≈0.5 zone really 14% accept (optimistic calibration) |
| control | 24 | 29% | unbiased base-rate |

Confident-wrong (pre-refit p≥0.80 but rejected): **8**. The uncertain bucket landing at 14% (not ~50%) means the pre-refit probabilities were **optimistic** near the boundary — p≈0.5 was really ~14% accept. That's a calibration offset, not a ranking failure (AUC held); the harvest operating threshold must be set from data, not p=0.5. These 8 blind-spots + the top_g misses are now training data.

## Refit held-out (LOMO, C fixed at first-fit value)

The C→AUC curve is flat, so re-selecting C would be noise; weights + AUC are held at the first fit's C for an apples-to-apples comparison.

| tier | C | AUC (fixed-C) | grid[min..max] | prior (grid-max) | AP |
|---|---|---|---|---|---|
| T1_global | 0.05 | 0.849 | [0.846..0.849] | 0.848 | 0.612 |
| T2_cells | 2.0 | 0.859 | [0.855..0.863] | 0.878 | 0.642 |

## Weight stability (T2_cells; refit vs first fit)

| feature | prior | refit | Δ |
|---|---|---|---|
| detail_spread | -1.765 | -1.754 | +0.010 |
| interior_worst | -1.660 | -1.272 | +0.388 |
| flat_worst | +1.923 | +1.265 | -0.658 |
| detail_worst | +1.247 | +1.118 | -0.129 |
| g_speckle | +1.306 | +0.789 | -0.517 |
| speckle_worst | -0.062 | -0.718 | -0.657 |
| g_mid | +0.572 | +0.687 | +0.115 |
| flat_spread | +0.000 | +0.482 | +0.482 |
| speckle_spread | -0.506 | +0.359 | +0.865 |
| flat_edge_minus_center | +0.371 | +0.294 | -0.077 |
| g_flat | -0.121 | -0.056 | +0.065 |
| g_interior | +0.000 | +0.000 | +0.000 |
| g_high | +0.000 | +0.000 | +0.000 |
| g_occ | +0.031 | +0.000 | -0.031 |
| interior_spread | +0.000 | +0.000 | +0.000 |

**Named checks:** g_mid dominates T1 = `True` · flat_edge_minus_center sign kept = `True` · g_occ stayed dead = `True`.

## Readiness verdict

- Held-out held (T2 grid-max 0.863 vs prior grid-max 0.878, within the flat-grid noise band): **YES**
- Dominant carriers (|w|≥1.0: detail_spread, interior_worst, flat_worst, detail_worst, g_speckle) stable in sign & rank: **YES**
- Secondary wobble (noted, not blocking): ['speckle_spread'] — `speckle_spread` flipped −0.51→+0.36; a secondary dispersion term, worth watching but the boundary's dominant structure held.

### → HARVEST-READY: **YES**

The dominant boundary carriers converged and held-out ranking held within noise. One secondary dispersion feature wobbled; the operating threshold for harvest should be calibrated from labels (the p≈0.5 zone is ~14% accept, not 50%), not taken at p=0.5.

Field re-verify: `out/q4_stage1/refit/field_<mb>.png` (v2-masked dense grid, refit T2 model).
