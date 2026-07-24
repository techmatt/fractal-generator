# Steered run 2 — morph-novelty + depth + keeper tier

Run `steered_run2`: lambda_m=0.5, beta=0.02, 75 distinct-q3 admissions over 179.83 active min / 341 batches (pilot: 16 in 7.8 min). Morphology below is the LIBRARY grayscale morph_gray recipe (offline, admissions-only) — comparable to the pilot and the 0.851/0.938/0.974 yardsticks. The LIVE novelty penalty ran on the cheap-JPG substrate with re-anchored knees **lo=0.8775 hi=0.939** (`median cheap-substrate pairwise cos over pilot admissions (distinct looks)` / `median cheap cos of 4 morph_gray-near-repeat pairs`; the grayscale 0.85/0.974 anchors do not transfer to this substrate — cross-check: --expand sample median cos 0.8579).

## q3(discovery) and q3(keeper) per family

Admission is the per-partition discovery `t_good` (unchanged). **Keeper** is the stricter F0.5 cut on the persisted canonical p_good (`tools/atlas/keeper_cut.py`, report-only — PROVISIONAL pending the blind human read). A partition below the >=15-positive floor is uncalibrated (keeper cut = baseline 0.50, flagged *).

| family | keeper cut | q3(discovery) | q3(keeper) | pilot q3(disc) |
|---|---:|---:|---:|---:|
| mandelbrot | 0.51 | 21 | 0 | 2 |
| multibrot3 | 0.5* | 14 | 14 | 2 |
| multibrot4 | 0.5* | 12 | 12 | 1 |
| multibrot5 | 0.5* | 19 | 19 | 4 |
| julia:mandelbrot | 0.55 | 1 | 0 | 3 |
| julia:multibrot3 | 0.27 | 6 | 6 | 1 |
| julia:multibrot4 | 0.53 | 1 | 1 | 1 |
| julia:multibrot5 | 0.15 | 1 | 1 | 2 |
| **total** | | **75** | **53** | **16** |

## Keeper tier — ranker ordering within the eligibility floor

The per-family F0.5 cut above is the **eligibility floor** ("not clearly bad"). Within the eligible keeper set, ordering is by the **location preference ranker** (`pref_loc_v0`, head logi, sets v7+colored) — a goodness ranker, not the p_good badness filter. Both numbers are shown: the raw ranker score and its percentile within this keeper set. The ranker ranks; it never steers admission (scorer.py HARD SCOPE).

| rank | id | family | p_good | keeper cut | ranker score | ranker pct |
|--:|---|---|--:|--:|--:|--:|
| 1 | st_multibrot5_steered_run2_000056 | multibrot5 | 0.649 | 0.50 | +3.923 | 1.00 |
| 2 | st_multibrot3_steered_run2_000002 | multibrot3 | 0.806 | 0.50 | +3.752 | 0.98 |
| 3 | st_julia_multibrot3_steered_run2_000009 | julia:multibrot3 | 0.790 | 0.27 | +3.172 | 0.96 |
| 4 | st_multibrot4_steered_run2_000022 | multibrot4 | 0.682 | 0.50 | +3.085 | 0.94 |
| 5 | st_julia_multibrot3_steered_run2_000003 | julia:multibrot3 | 0.625 | 0.27 | +3.048 | 0.92 |
| 6 | st_multibrot4_steered_run2_000034 | multibrot4 | 0.850 | 0.50 | +3.022 | 0.91 |
| 7 | st_multibrot5_steered_run2_000046 | multibrot5 | 0.596 | 0.50 | +2.445 | 0.89 |
| 8 | st_multibrot5_steered_run2_000031 | multibrot5 | 0.592 | 0.50 | +2.347 | 0.87 |
| 9 | st_multibrot3_steered_run2_000037 | multibrot3 | 0.735 | 0.50 | +2.211 | 0.85 |
| 10 | st_multibrot5_steered_run2_000052 | multibrot5 | 0.634 | 0.50 | +2.108 | 0.83 |
| 11 | st_multibrot5_steered_run2_000055 | multibrot5 | 0.709 | 0.50 | +1.990 | 0.81 |
| 12 | st_multibrot3_steered_run2_000040 | multibrot3 | 0.663 | 0.50 | +1.654 | 0.79 |
| 13 | st_multibrot5_steered_run2_000049 | multibrot5 | 0.709 | 0.50 | -0.586 | 0.77 |
| 14 | st_julia_multibrot3_steered_run2_000015 | julia:multibrot3 | 0.723 | 0.27 | -1.005 | 0.75 |
| 15 | st_multibrot3_steered_run2_000011 | multibrot3 | 0.747 | 0.50 | -1.226 | 0.74 |
| 16 | st_multibrot5_steered_run2_000035 | multibrot5 | 0.636 | 0.50 | -1.497 | 0.72 |
| 17 | st_julia_multibrot3_steered_run2_000051 | julia:multibrot3 | 0.688 | 0.27 | -1.514 | 0.70 |
| 18 | st_multibrot5_steered_run2_000054 | multibrot5 | 0.785 | 0.50 | -1.572 | 0.68 |
| 19 | st_multibrot5_steered_run2_000045 | multibrot5 | 0.660 | 0.50 | -1.626 | 0.66 |
| 20 | st_multibrot5_steered_run2_000036 | multibrot5 | 0.560 | 0.50 | -1.725 | 0.64 |
| 21 | st_multibrot3_steered_run2_000061 | multibrot3 | 0.666 | 0.50 | -1.769 | 0.62 |
| 22 | st_multibrot5_steered_run2_000005 | multibrot5 | 0.630 | 0.50 | -1.778 | 0.60 |
| 23 | st_multibrot5_steered_run2_000057 | multibrot5 | 0.563 | 0.50 | -1.845 | 0.58 |
| 24 | st_julia_multibrot3_steered_run2_000050 | julia:multibrot3 | 0.827 | 0.27 | -1.862 | 0.57 |
| 25 | st_multibrot3_steered_run2_000048 | multibrot3 | 0.804 | 0.50 | -1.889 | 0.55 |
| 26 | st_multibrot4_steered_run2_000068 | multibrot4 | 0.721 | 0.50 | -1.967 | 0.53 |
| 27 | st_multibrot3_steered_run2_000070 | multibrot3 | 0.648 | 0.50 | -2.056 | 0.51 |
| 28 | st_multibrot3_steered_run2_000071 | multibrot3 | 0.583 | 0.50 | -2.102 | 0.49 |
| 29 | st_multibrot3_steered_run2_000029 | multibrot3 | 0.588 | 0.50 | -2.399 | 0.47 |
| 30 | st_multibrot3_steered_run2_000025 | multibrot3 | 0.676 | 0.50 | -2.501 | 0.45 |
| 31 | st_multibrot4_steered_run2_000069 | multibrot4 | 0.821 | 0.50 | -2.561 | 0.43 |
| 32 | st_multibrot4_steered_run2_000042 | multibrot4 | 0.889 | 0.50 | -2.678 | 0.42 |
| 33 | st_multibrot3_steered_run2_000026 | multibrot3 | 0.775 | 0.50 | -2.699 | 0.40 |
| 34 | st_multibrot4_steered_run2_000041 | multibrot4 | 0.838 | 0.50 | -2.787 | 0.38 |
| 35 | st_multibrot5_steered_run2_000028 | multibrot5 | 0.530 | 0.50 | -2.971 | 0.36 |
| 36 | st_multibrot5_steered_run2_000001 | multibrot5 | 0.674 | 0.50 | -2.980 | 0.34 |
| 37 | st_multibrot5_steered_run2_000047 | multibrot5 | 0.644 | 0.50 | -3.009 | 0.32 |
| 38 | st_multibrot3_steered_run2_000043 | multibrot3 | 0.519 | 0.50 | -3.075 | 0.30 |
| 39 | st_multibrot4_steered_run2_000016 | multibrot4 | 0.563 | 0.50 | -3.146 | 0.28 |
| 40 | st_multibrot4_steered_run2_000023 | multibrot4 | 0.556 | 0.50 | -3.221 | 0.26 |
| 41 | st_julia_multibrot5_steered_run2_000004 | julia:multibrot5 | 0.551 | 0.15 | -3.343 | 0.25 |
| 42 | st_julia_multibrot4_steered_run2_000013 | julia:multibrot4 | 0.571 | 0.53 | -3.400 | 0.23 |
| 43 | st_multibrot4_steered_run2_000024 | multibrot4 | 0.776 | 0.50 | -3.830 | 0.21 |
| 44 | st_multibrot5_steered_run2_000053 | multibrot5 | 0.718 | 0.50 | -3.838 | 0.19 |
| 45 | st_multibrot4_steered_run2_000012 | multibrot4 | 0.904 | 0.50 | -4.233 | 0.17 |
| 46 | st_julia_multibrot3_steered_run2_000073 | julia:multibrot3 | 0.614 | 0.27 | -4.251 | 0.15 |
| 47 | st_multibrot4_steered_run2_000067 | multibrot4 | 0.858 | 0.50 | -4.393 | 0.13 |
| 48 | st_multibrot5_steered_run2_000017 | multibrot5 | 0.767 | 0.50 | -4.460 | 0.11 |
| 49 | st_multibrot4_steered_run2_000066 | multibrot4 | 0.895 | 0.50 | -4.539 | 0.09 |
| 50 | st_multibrot5_steered_run2_000033 | multibrot5 | 0.696 | 0.50 | -4.686 | 0.08 |
| 51 | st_multibrot3_steered_run2_000064 | multibrot3 | 0.563 | 0.50 | -4.794 | 0.06 |
| 52 | st_multibrot5_steered_run2_000006 | multibrot5 | 0.514 | 0.50 | -5.871 | 0.04 |
| 53 | st_multibrot3_steered_run2_000021 | multibrot3 | 0.571 | 0.50 | -6.281 | 0.02 |

Note (n small, see ranker validation): the lift is largely CROSS-family (steered mandelbrot ranks to the bottom, julia to the top); WITHIN a good-rich family the order is still noisy. Legitimate for a pooled keeper set; not yet fine within-family taste.

## Admissions / active-hour over time (decay or floor?)

| active-time third (min) | admissions | admissions / active-hour |
|---|---:|---:|
| 0–60 | 42 | 42.0 |
| 60–120 | 16 | 16.0 |
| 120–180 | 17 | 17.0 |

Yield trajectory: **DECAYS from an initial burst, then FLOORS** (42 -> 16 -> 17 adm/active-hr across the thirds). The steered walk does not run dry — after the rich near-root regions are mined it holds a steady floor from fresh roots + deeper lineages.

## Morph-cluster count trajectory (distinct looks over time)

morph_gray single-linkage clusters over the admissions in admission order; the count must keep GROWING if steering keeps finding new looks (vs re-buying). Cuts: strict cos>0.974, perceptual cos>0.95.

| after k admissions | strict clusters | perceptual clusters |
|---:|---:|---:|
| 18 | 18 | 15 |
| 37 | 37 | 33 |
| 56 | 55 | 45 |
| 75 | 71 | 51 |

**75 admissions -> 71 distinct morphs (strict), 51 (perceptual).** Pilot: 16 -> 16 / 13. Median pairwise morph_gray cos 0.850 (pilot 0.882; library 0.851).

## Depth distribution of admitted q3 vs pilot

| depth | steered run2 | pilot |
|---:|---:|---:|
| 2 | 6 | 5 |
| 3 | 9 | 8 |
| 4 | 7 | 2 |
| 5 | 9 | 1 |
| 6 | 9 | 0 |
| 7 | 9 | 0 |
| 8 | 8 | 0 |
| 9 | 5 | 0 |
| 10 | 5 | 0 |
| 11 | 5 | 0 |
| 12 | 1 | 0 |
| 13 | 1 | 0 |
| 15 | 1 | 0 |

Median admitted depth **6.0** (pilot ~3), max **15** (pilot 5); **44/75** admissions are depth>5 (pilot 1/16). The pilot admitted 13/16 at depth<=3; run2's distribution is broad through depth 6–11. This is the depth bonus (beta=0.02) AND the capped-node eviction together — evicting hot shallow roots frees the frontier to single-track fresh lineages deep, which the clogged pilot frontier could not.

## Per-term priority contribution (which term is actually steering?)

| term | mean | mean |abs| | share of |abs| |
|---|---:|---:|---:|
| eord | +0.6663 | 0.6663 | 40.7% |
| gumbel | +0.0461 | 0.0811 | 4.9% |
| dup_pen | +0.2667 | 0.2667 | 16.3% |
| nov_pen | +0.4868 | 0.4868 | 29.7% |
| depth_bonus | +0.1383 | 0.1383 | 8.4% |

**Novelty-penalty hit rate: 38256/38419 = 99.6%** of pushed candidates; mean penalty among hits 0.489 (max 0.500). cos_max distribution: median 0.961, p90 0.975.

**Saturation caveat.** 89.7% of candidates hit ~FULL penalty (cos_max >= hi=0.939), and the morph memory grew to **10420** looks. With that many memory rows the cheap-substrate cos_max is almost always past the knee, so the penalty acted as a near-CONSTANT down-shift for most of the run rather than a discriminating gradient — the anchors were calibrated on the pilot's 16-look (sparse) memory and do not account for memory DENSITY. Diversity below is still high, but it is carried more by the coord dup-penalty + density rejection than by a live morph gradient. v1.2 lever: cap/subsample the memory or raise hi as |memory| grows.

## Coord-dup and morph near-repeat rates vs pilot

- **Coord-dup rate** (q3_dup / all decoded-q3): 4930/5005 = **98.5%** (pilot 11.1%).
- **Morph near-repeat pairs** (morph_gray cos>0.95): **33** of 2775 admission pairs (pilot 4); i.e. admissions collapse 75->51 at the perceptual cut (24 merges, pilot 3).
- Cross-partition perceptual clusters (partitions sharing a look): **5**.

## M-cap hit rate under the new policy

- roots expanded: **420**; expansions/root max 70, median 25, mean 25.6
- roots at/over M=40: **205** (48.8% of roots). The pilot's 8 is NOT comparable — this run is ~23x longer, so many more roots reach the cap; the load-bearing change is that capped nodes are now EVICTED from the frontier (pop_batch), not retained. In the pilot design capped-but-retained nodes saturated the 6000-node FRONTIER_CAP by batch ~110 (100% dead weight) and collapsed throughput; eviction keeps the frontier all-expandable, which is what let the run reach depth 6–15 and floor its yield instead of stalling.
