# Location preference ranker v0 — report

Ranks locations by human quality on steered/dive output — the missing third leg beside canonical p_good (a *badness* filter, not a goodness ranker; see `docs/findings/steered_run2_keeper_calibration.md`). **Scope: ranks the not-bad; never steers.** Consumers = keeper ranking, emission feed, dive-result sorting ONLY — never frontier priority, dive-start selection, or any discovery-side decision.

- Labeled admissions: **81** (run2 60 + dive 21), 96 decoded total.
- Frozen features: morph-CLIP (768), v7 penultimate (1280), colored-CLIP (768).
- Heads: Bradley-Terry pairwise (bt), ridge on score, logistic good-vs-rest (logi); reg by inner 5-fold CV on train (leak-free).
- Prior corpus (v7-only, 360 balanced rows) available in both folds via `+prior`; never in eval. Colored/morph CLIP priors omitted — corpus crops carry varied delivered palettes, so they do not share the target's uniform twilight_shifted appearance space.

## Winner: `v7+colored:logi`  (reg=0.1)

Pooled Spearman (within-fold pct-normalized) = **+0.349**, 95% bootstrap CI [+0.136, +0.534], permutation p = **0.0014**  (n=81).

**SHIPS AS KEEPER-RANKER (provisional).** Beats both baselines on mean-LOBO Spearman with a CI excluding zero; consume in keeper ranking / emission feed / dive sorting only. n=81 is small — the standing next-read loop must keep feeding it.

## Leave-one-batch-out grid

Mean = average over the two eval folds. Fold columns: **dive** = train run2/eval dive, **run2** = train dive/eval run2.

| config | meanSp | meanAUC | meanP@10 | dive Sp | dive AUC | run2 Sp | run2 AUC |
|---|---|---|---|---|---|---|---|
| `v7+colored:logi` | +0.430 | 0.750 | 0.400 | +0.524 | 0.788 | +0.335 | 0.712 |
| `v7+colored:bt` | +0.379 | 0.742 | 0.400 | +0.402 | 0.740 | +0.357 | 0.744 |
| `morph+v7:logi` | +0.369 | 0.693 | 0.350 | +0.282 | 0.644 | +0.455 | 0.742 |
| `morph+v7+colored:logi` | +0.366 | 0.715 | 0.400 | +0.336 | 0.673 | +0.396 | 0.756 |
| `v7:logi` | +0.351 | 0.706 | 0.400 | +0.298 | 0.692 | +0.403 | 0.720 |
| `v7+colored:ridge` | +0.336 | 0.700 | 0.350 | +0.327 | 0.692 | +0.345 | 0.708 |
| `morph+v7+colored:bt` | +0.326 | 0.701 | 0.450 | +0.253 | 0.644 | +0.400 | 0.758 |
| `morph+v7:ridge` | +0.325 | 0.676 | 0.350 | +0.238 | 0.635 | +0.413 | 0.718 |
| `morph+v7:bt` | +0.312 | 0.675 | 0.350 | +0.203 | 0.625 | +0.421 | 0.724 |
| `morph+v7+colored:ridge` | +0.300 | 0.683 | 0.400 | +0.213 | 0.625 | +0.388 | 0.742 |
| `colored:logi` | +0.254 | 0.674 | 0.350 | +0.454 | 0.731 | +0.053 | 0.618 |
| `morph+colored:logi` | +0.240 | 0.651 | 0.350 | +0.370 | 0.683 | +0.110 | 0.620 |
| `colored:bt` | +0.204 | 0.634 | 0.350 | +0.212 | 0.606 | +0.196 | 0.662 |
| `morph+colored:ridge` | +0.193 | 0.624 | 0.450 | +0.153 | 0.587 | +0.232 | 0.662 |
| `v7:ridge` | +0.191 | 0.632 | 0.400 | +0.002 | 0.558 | +0.380 | 0.706 |
| `colored:ridge` | +0.191 | 0.600 | 0.400 | +0.142 | 0.548 | +0.239 | 0.652 |
| `v7:bt` | +0.177 | 0.628 | 0.400 | -0.037 | 0.538 | +0.392 | 0.718 |
| `morph+colored:bt` | +0.172 | 0.631 | 0.400 | +0.153 | 0.587 | +0.191 | 0.676 |
| `morph:ridge` | +0.123 | 0.567 | 0.300 | +0.149 | 0.596 | +0.098 | 0.538 |
| `morph:logi` | +0.102 | 0.538 | 0.300 | +0.177 | 0.577 | +0.027 | 0.500 |
| `morph:bt` | +0.082 | 0.560 | 0.300 | +0.130 | 0.596 | +0.033 | 0.524 |
| `v7:ridge+prior` | -0.005 | 0.509 | 0.250 | -0.381 | 0.298 | +0.370 | 0.720 |
| `v7:logi+prior` | -0.081 | 0.467 | 0.250 | -0.440 | 0.260 | +0.279 | 0.674 |
| **BASE canon_pgood** | +0.120 | 0.565 | 0.300 | -0.039 | 0.500 | +0.278 | 0.630 |
| **BASE random** | +0.023 | 0.510 | 0.300 | +0.204 | 0.644 | -0.158 | 0.376 |

## Per-family breakdown (winner, pooled over eval folds)

`mean_pct_good`/`mean_pct_bad` = mean within-fold rank-percentile of that family's human-good / human-bad tiles (1.0 = ranker put them top). A useful ranker pushes good high, bad low.

| family | n | n_good | mean_pct_good | mean_pct_bad | Spearman |
|---|---|---|---|---|---|
| julia:mandelbrot | 1 | 0 | — | 0.22 | — |
| julia:multibrot3 | 7 | 4 | 0.73 | 0.92 | -0.48 |
| julia:multibrot5 | 1 | 0 | — | — | — |
| mandelbrot | 16 | 0 | — | 0.15 | +0.30 |
| multibrot3 | 15 | 4 | 0.46 | 0.67 | -0.38 |
| multibrot4 | 14 | 2 | 0.82 | 0.50 | +0.24 |
| multibrot5 | 27 | 8 | 0.74 | 0.58 | +0.34 |

**multibrot4 callout.** n=14, n_good=2 (the dive read had 0/6 multibrot4-good — its 2 goods here are both from run2). The ranker is **not blind** to it: its good tiles land at mean percentile 0.82, its bad at 0.50. In the dive fold specifically there are no multibrot4 goods to elevate, so there it is judged only on not over-ranking the family (neutral).


**Where the signal comes from (honest).** Much of the mean-LOBO lift is *cross-family*: steered mandelbrot is almost all bad (n_good=0) and the ranker pushes its bad tiles to percentile 0.15, and julia families read strong — so a large slice of the Spearman is the head learning family-level quality priors, which is legitimate for a *pooled* keeper set but is not fine within-family taste. Within the good-rich families it is uneven: multibrot5 orders sensibly (Sp +0.34, good 0.74 > bad 0.58), but julia:multibrot3 **inverts** (Sp -0.48, ranks its bad 0.92 *above* its good 0.73) on n=7. This is the n=81 story: real cross-family separation, noisy within-family ordering. Label more to firm up within-family taste.

**Corpus prior verdict: rejected.** `v7:logi+prior` scored meanSp -0.081 (dive fold -0.440) — the older label corpus *degrades* the ranker, collapsing the dive fold. Different sampling distribution + delivered-palette appearance; the v7 penultimate does not carry it across cleanly. Do not fold the prior in without a distribution-matched re-derivation.

## Deliverables

- Ranked contact sheet (all 96, sorted by ranker score, human label marked): `out\ranker\pref_loc_v0_ranked_sheet.png`
- Next-read manifest: `out\ranker_next_read/` — 15 tiles (2 top-ranked/candidate + 7 max-uncertainty + 6 confident-low control), shuffled, hidden key. Unread pool = 15 unlabeled admissions (< 20 requested — this is the entire steered_run2 non-blind remainder, and it sits mostly below the good/not-good boundary, so only 2 read as candidate keepers; the standing loop refills the top band as new runs land).
- Model artifact: `data\ranker\pref_loc_v0\model.npz`; scorer `tools/ranker/scorer.py` (`RankerScorer.load()`).