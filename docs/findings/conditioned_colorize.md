# Category-conditioned colorizer — realizability matrix (v3-gvo as-is)

Pool = **production** `pool_colormaps.json` (987 palettes = the set `palette_categories.json` k16 is cut over). Scorer = **pref-v3-gvo**, canonical recipe (pct/γ1/no reverse·phase·cycles). Fit is within-location only; every number below is a within-location Δ.

**47 locations × 19 cells** (16 k16 chromatic leaves + spectral/outlier/neutral specials).

## Cost of conditioning

Over all **846** (location, non-argmax cell) pairs, fitΔ = best-in-cell − global-argmax:

- median **-3.6737** · mean **-4.2485** · p10 **-8.2616** · p90 **-0.9999**
- within 1 pt of the global argmax: **10%** · within 2 pt: **29%**

## Cells hardest to realize (most-negative median fitΔ)

| cell | #pal | locs | median Δ | worst Δ | best Δ |
|------|-----:|-----:|---------:|--------:|-------:|
| k16:2 | 27 | 47 | -7.2342 | -14.5339 | -1.0252 |
| special:neutral | 15 | 47 | -7.0047 | -12.9154 | -0.9742 |
| k16:3 | 40 | 47 | -5.5482 | -10.7207 | -0.5149 |
| k16:1 | 68 | 47 | -5.4665 | -10.9767 | -0.091 |
| k16:12 | 70 | 47 | -5.2902 | -11.9832 | -1.2486 |
| special:spectral | 57 | 47 | -5.242 | -10.9295 | 0.0 |

Easiest (least cost to condition into):

| cell | #pal | locs | median Δ |
|------|-----:|-----:|---------:|
| k16:8 | 55 | 47 | -0.3923 |
| k16:4 | 38 | 47 | -1.2907 |
| k16:13 | 22 | 47 | -1.6975 |
| k16:6 | 69 | 47 | -2.1318 |
| k16:15 | 54 | 47 | -2.5591 |
| k16:5 | 93 | 47 | -2.9247 |

## Does v3-gvo discriminate WITHIN a cell?

Within-cell fit spread (max−min raw fit across the palettes in one cell), per (loc, cell), median over all cells with ≥2 palettes:

- **median within-cell spread = 10.2076** (mean 11.1231) · median within-cell std 2.2623
- Compare to the cost-of-conditioning scale above (p10 -8.2616). If the within-cell spread is on the ORDER of the between-cell cost, v3-gvo IS resolving palettes inside a cell and within-cell argmax is meaningful; if it is ~0, within-cell choice is arbitrary and a conditioned scorer is worth building.

## cet_linear_bmw over-dense cluster cross-check

`cet_linear_bmw_5_95_c86` sits in cell **k16:8**. **19** location(s) have their GLOBAL argmax in that cell. Cheapest alternative cell per such location:

| loc | global palette | best alt cell | alt palette | alt fitΔ |
|-----|----------------|---------------|-------------|---------:|
| cycle_001_wfd_000_02 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -2.8214 |
| cycle_001_wfd_002_02 | cet_linear_bmw_5_95_c8 | k16:4 | Orchid Triad | -0.2227 |
| cycle_001_wfd_004_00 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -0.0786 |
| cycle_002_wfd_002_01 | cet_linear_bmw_5_95_c8 | k16:6 | cmr.flamingo | -2.3928 |
| cycle_002_wfd_004_02 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -1.4131 |
| cycle_003_wfd_000_02 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -1.4877 |
| cycle_003_wfd_001_00 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -2.3385 |
| cycle_003_wfd_002_04 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -4.1068 |
| cycle_003_wfd_003_09 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -0.5523 |
| cycle_003_wfd_007_00 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -1.8726 |
| cycle_003_wfd_010_09 | cet_linear_bmw_5_95_c8 | k16:13 | gist_heat | -0.3834 |
| cycle_003_wfd_023_02 | cet_linear_bmw_5_95_c8 | k16:6 | cmr.flamingo | -3.3951 |
| cycle_003_wfd_024_04 | bwr | k16:7 | cet_diverging_bwg_20_9 | -0.675 |
| cycle_004_wfd_001_00 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -2.1653 |
| cycle_004_wfd_003_01 | cet_linear_bmw_5_95_c8 | k16:4 | Orchid Triad | -1.5075 |
| cycle_005_wfd_001_09 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -3.2803 |
| cycle_005_wfd_014_00 | cet_linear_bmw_5_95_c8 | k16:4 | cmr.gothic | -0.901 |
| cycle_005_wfd_015_00 | cet_linear_bmw_5_95_c8 | k16:6 | cmr.flamingo | -0.8219 |
| cycle_005_wfd_017_05 | cet_linear_bmw_5_95_c8 | k16:6 | Ember Against Steel | -2.1318 |

## Contact sheet

`contact_sheet.png` — 4 locations × 8 cells (one best-in-cell thumbnail per cell, ARGMAX cell tagged).
