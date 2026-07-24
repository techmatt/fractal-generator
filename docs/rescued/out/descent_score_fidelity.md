# Descent score fidelity — can v7 steer from cheap renders?

Sample: **465** outcome-ledger frames from prospect_run1, stratified across 9 families x 3 depth buckets. Scorer: **data/classifier/v7/model_best.pt** (v7). Reference arm = canonical 640x360 ss2 twilight_shifted.

## Coverage (family x depth-bucket)

| family | shallow | mid | deep | total |
|---|---|---|---|---|
| julia:mandelbrot | 18 | 18 | 18 | 54 |
| julia:multibrot3 | 13 | 18 | 18 | 49 |
| julia:multibrot4 | 17 | 18 | 18 | 53 |
| julia:multibrot5 | 12 | 16 | 11 | 39 |
| mandelbrot | 18 | 18 | 18 | 54 |
| multibrot3 | 18 | 18 | 18 | 54 |
| multibrot4 | 18 | 18 | 18 | 54 |
| multibrot5 | 18 | 18 | 18 | 54 |
| phoenix | 18 | 18 | 18 | 54 |

## Sanity anchor — pipeline reproduces the stored v6 reward-pass score

Re-scored 465 canonical renders with **v6** (data/classifier/v6/model_best.pt) vs the stored v6 reward-pass scores: mean|Δp_notbad|=0.0017 (max 0.0478), mean|Δp_good|=0.0003 (max 0.0194), Spearman p_notbad=0.9998, p_good=0.9997. Small deltas + ~1.0 rank correlation confirm the geometry/plane resolution and render path reproduce the reward pass (residual = GPU nondeterminism + the reward pass's reframe-winner vs our fixed-outcome-geometry render).

## Spearman vs canonical — E[ord]

| arm | pooled | pooled (upper half) |
|---|---|---|
| cheap-node | 0.949 (n=465) | 0.902 |
| parent-crop | 0.952 (n=460) | 0.909 |

### Per-family Spearman — E[ord]

| family | cheap-node | parent-crop |
|---|---|---|
| julia:mandelbrot | 0.918 (n=54) | 0.961 (n=54) |
| julia:multibrot3 | 0.957 (n=49) | 0.918 (n=49) |
| julia:multibrot4 | 0.984 (n=53) | 0.969 (n=53) |
| julia:multibrot5 | 0.948 (n=39) | 0.936 (n=39) |
| mandelbrot | 0.842 (n=54) | 0.938 (n=53) |
| multibrot3 | 0.955 (n=54) | 0.881 (n=54) |
| multibrot4 | 0.921 (n=54) | 0.915 (n=52) |
| multibrot5 | 0.905 (n=54) | 0.808 (n=52) |
| phoenix | 0.835 (n=54) | 0.743 (n=54) |

## Spearman vs canonical — p_good

| arm | pooled | pooled (upper half) |
|---|---|---|
| cheap-node | 0.947 (n=465) | 0.901 |
| parent-crop | 0.953 (n=460) | 0.910 |

## Simulated rung choice — random 4-frame groups within (family, depth-bucket)

Top-1 agreement = cheap-arm argmax equals canonical argmax. Regret = mean canonical E[ord] lost by picking the cheap arm's argmax.

| arm | top-1 agreement | mean regret | groups |
|---|---|---|---|
| cheap-node | 0.840 | +0.0175 | 8100 |
| parent-crop | 0.835 | +0.0176 | 8100 |

## Verdict

- cheap-node: **usable for steering** — Spearman(E[ord])=0.949, rung top-1 agreement=0.840, regret=+0.0175.
- parent-crop: **usable for steering** — Spearman(E[ord])=0.952, rung top-1 agreement=0.835, regret=+0.0176.

### Caveats

- Cheap-node arm renders 384x216 ss1 and *colorizes* the smooth field with twilight_shifted; the walk itself never colorizes (it gates on the raw f64 mu field). Coloring map is identical to canonical; only resolution+AA differ, which is exactly the presentation variable under test.
- Parent-crop uses a **concentric** parent (child centered, parent_fw = child_fw / 0.418 = geometric-mean of the [0.35,0.5] zoom band). This is EXACT for julia `center`-descend rows (straight z-plane zoom) and an approximation for the recentering c-plane / `normal` rows (real child sits off-center).