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
