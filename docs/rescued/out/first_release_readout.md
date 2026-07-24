# First release — supplementary readout

Status: **COMPLETE** — pool has **1387** colorize attempts so far (660 gated). Reads only the durable pool log + snapshot; complements the driver's `out/first_release_report.md`.

## 0. Per-stage reconciliation

- attempts (found) **1387** == passed (written) **660** + floor-dropped **727** + errored **0** → **OK**
- pool pass rate 47.6%

## 1. Realized shares vs the target measure (order book)

Target = the measure's **cell-normalized** target fraction over the full feasible support (192736 cells) — what the DeficitModel actually drives (cluster-count-weighted, so a type with more morph clusters draws proportionally more unless its type-weight offsets it). Realized = the gated pool / the reconstructed release.

### per fractal_type

| type | target | gated | release |
|---|--:|--:|--:|
| julia:mandelbrot | 12.3% | 3.8% | 6.0% |
| julia:multibrot3 | 12.9% | 7.0% | 10.0% |
| julia:multibrot4 | 12.9% | 7.0% | 2.0% |
| julia:multibrot5 | 12.3% | 6.1% | 2.0% |
| mandelbrot | 17.8% | 13.9% | 12.0% |
| multibrot3 | 11.9% | 17.3% | 16.0% |
| multibrot4 | 1.9% | 7.7% | 10.0% |
| multibrot5 | 13.8% | 15.8% | 18.0% |
| phoenix | 4.2% | 21.5% | 24.0% |

### render_style marginal (gated)

| style | target | gated | release |
|---|--:|--:|--:|
| smooth | 12.5% | 11.8% | 100.0% |
| tia | 12.5% | 18.8% | 0.0% |
| stripe | 12.5% | 18.5% | 0.0% |
| smooth_mean_angle | 12.5% | 8.2% | 0.0% |
| smooth_angle_min | 12.5% | 12.1% | 0.0% |
| composite_c7_smooth_trap_circle | 12.5% | 4.5% | 0.0% |
| composite_c13_smooth_stripe | 12.5% | 17.0% | 0.0% |
| composite_c17_smooth_curvature | 12.5% | 9.1% | 0.0% |

## 2. Cell reachability at library scale

- feasible (type,cluster) pairs: **1268**; produced ≥1 gated wallpaper: **632** (49.8%)
- distinct joint cells (type,cluster,flavor,style) filled in the gated pool: **660**
- (with one colorize per location, each location fills exactly one joint cell, so joint-cell coverage tracks the per-location deficit pick, not exhaustive cell sweep)

## 3. Per-niche percentile health at scale

Niche = full descriptor cell. Gated pool occupies **660** niches; **660** are singletons (**100.0%**).

| niche size | # niches |
|--:|--:|
| 1 | 660 |

**Reading:** at ~one colorize per location the within-cell percentile is still largely degenerate (singletons → percentile 1.0; selection tie-breaks on absolute p_ge3).

## 4. Strange inventory (mining head)

- strange (non-smooth) gated wallpapers: **582**
- above the 0.5 mining release floor: **82** (toward mining-head calibration)

## 5. Realized hue / chroma (pooled over the gated pool)

Accumulated over **660** gated wallpapers (chroma-weighted hue histogram + chroma histogram). See `out/first_release/hue_chroma.png`.

| hue bin | share |   | chroma bin | share |
|---|--:|---|---|--:|
| red | 16.4% |   | 0.00–0.12 | 16.6% |
| orange | 13.0% |   | 0.12–0.25 | 22.4% |
| yellow | 1.8% |   | 0.25–0.38 | 21.7% |
| chartreuse | 2.8% |   | 0.38–0.50 | 17.4% |
| green | 2.7% |   | 0.50–0.62 | 11.4% |
| spring | 4.7% |   | 0.62–0.75 | 6.1% |
| cyan | 15.1% |   | 0.75–0.88 | 3.6% |
| azure | 13.2% |   | 0.88–1.00 | 0.9% |
| blue | 8.3% |   |  |  |
| violet | 9.6% |   |  |  |
| magenta | 5.7% |   |  |  |
| rose | 6.7% |   |  |  |

## 6. Reject autopsy — fate-stratified sheet

`out/first_release/reject_autopsy_sheet.png` — a visual sample across every fate band:

- release-eligible (≥ head release floor): 147
- pool inventory (passed pool floor, below release floor): 513
- floor-rejected (below pool floor): 727
- render error: 0
