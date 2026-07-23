# q4 axis discovery — geometric axes past q3 for artist-quality Julia locations

Measurement pass (no descent/config/`data/` changes). Goal: find label-free,
field-computable geometric axes on which the one known artist-quality exemplar sits at
an **extreme** relative to the labeled corpus — axes pointing *past* q3 toward the
unlabeled "q4" tier.

**Exemplar** `cz_g0302_r002110` (julia, `julia_ladder_j0`, center view fw=0.75,
c=(0.26103, −0.48932)). Its distinctive look: many high-complexity clusters
**distributed across the frame**, punctuated by dark interior "lakes" — composed
negative space. See `out/q4_axis/exemplar_large.png`.

> **Label caveat.** The prompt calls the exemplar "human-labeled good", but its
> resolved label (`label_store.resolve_score`, max-over-crops) is **2 (okay)**, not 3.
> The study proceeds on the exemplar's *geometry* regardless — but its being an okay,
> not a good, is itself consistent with the headline result (its extremity is orthogonal
> to the good/bad gradient).

## Method

- **Population** — all human-labeled **julia** locations via `label_store.resolve_score`
  over every batch: **1087 distinct locations** (good=252, okay=294, bad=541), from
  `julia_ladder_j0` (1000) + `gather_v6` (87). No subsample needed (835 non-goods <
  5×252). Two julia ledger schemas present: `cz_` center views (cx=cy=0, fw=0.75/0.375)
  and `ds_` z-plane descents (off-origin viewport); both carry `c` in `c_re/c_im`.
- **Substrate** — `render-one --dump-field --dump-field-source f64` at each location's
  **labeled framing** (1280×720, ss1), escape-time field, NaN interior. Colormap-invariant.
- **GUARD passed** — coord round-trip on exemplar + 5 random goods spanning both schemas
  and both batches (incl. gather_v6 fw=3.4e-4): re-rendered fields match the labeled
  crops structurally (`out/q4_axis/guard_sheet.png`). No silent wrong-coord failure.
- Axes computed on the percentile-stretched field n01 (0.5/99.5, interior→0), mirroring
  the colormap stretch. `axes.py`: occupancy (frac active, local-std>0.05), mean_detail,
  8×8 tile complexity → spread_frac / tile_gini / tile_entropy, radial power-spectrum
  slope, 180° symmetry, + context (interior_frac, mean/median escape).

## HEADLINE — exemplar percentile per axis

| axis | ex value | %ile (goods) | %ile (all) | AUC good/bad | AUC good/rest |
|---|---|---|---|---|---|
| occupancy | 0.701 | **100.0** | **99.3** | 0.505 | 0.501 |
| mean_detail | 0.183 | **100.0** | **99.8** | 0.502 | 0.499 |
| spread_frac | 0.969 | **100.0** | **99.4** | 0.554 | 0.539 |
| tile_entropy | 0.985 | 95.6 | 94.5 | 0.549 | 0.537 |
| tile_gini | 0.162 | **1.6** | **4.0** | 0.447 | 0.462 |
| slope | −1.57 | **100.0** | **99.6** | 0.435 | 0.445 |
| symmetry | 1.000 | 78.6 | 75.3 | 0.470 | 0.483 |
| interior_frac | 0.246 | **99.6** | **99.9** | 0.545 | 0.516 |
| mean_esc | 257.9 | 98.8 | 99.2 | 0.634 | 0.617 |

**The exemplar pins the extreme of essentially every axis** — maximally busy
(occupancy/detail/spread ≈100th %ile), maximally distributed (tile_gini ≈4th %ile =
least concentrated), broadest spectrum (shallowest slope), and — most strikingly —
**interior_frac 0.246 at the 99.9th %ile (rank 2 of 1087)**, an almost-unique amount of
composed negative space.

**But every AUC ≈ 0.5.** None of these axes separate good from bad (best is mean_esc at
0.63, weak). Good/okay/bad distributions are near-identical (`out/q4_axis/strips.png`).
**The exemplar's geometry is orthogonal to the existing label gradient** — these axes
point *past* q3, not along good/bad. Exactly the q4 hypothesis.

## The axes collapse to TWO independent directions

Pearson over the corpus:

- `occupancy ~ mean_detail 0.92`, `~ spread_frac 0.96`, `~ tile_gini −0.89` — these are
  **one collinear "busy-ness / distributedness" axis**.
- `occupancy ~ interior_frac **0.15**`, `interior_frac ~ tile_gini −0.05` —
  **interior_frac is independent**: a second, orthogonal "composed negative-space" axis.

Single-axis good-rate lift (top decile vs base rate 0.232):

| axis (tail) | good rate | lift |
|---|---|---|
| occupancy (high) | 0.110 | **0.47×** (anti-quality) |
| mean_detail (high) | 0.174 | 0.75× |
| tile_gini (low) | 0.211 | 0.91× |
| **interior_frac (high)** | **0.330** | **1.42×** (only positive) |

## Visual autopsy — why no single axis works

- **occupancy / spread_frac / mean_detail tail** (`sheet_corpus_HIGH_occupancy.png`):
  **degenerate speckle** — diffuse fine dust filling the frame. Mostly bad. Busy-ness
  alone rewards noise (hence the 0.47× lift).
- **low tile_gini tail** (`sheet_corpus_LOW_tile_gini.png`): **structureless pinwheel
  dust** — evenly spread but no composed negative space, no varied clusters.
- **high interior_frac tail** (`sheet_corpus_HIGH_interior_frac.png`): **this is the
  exemplar's look** — dendrite/spiral clusters interspersed with dark interior lakes,
  visibly enriched in goods (top entries include several s3). The exemplar is rank 2 here.

## Verdict — the q4 direction

**A single simple axis does not exist.** Busy-ness (occupancy/spread/detail/gini — one
axis) is *anti*-quality on its own (degenerates to speckle). The genuinely useful,
exemplar-resembling axis is **interior_frac** (composed negative space) — the only one
with positive good-lift (1.42×) and a non-degenerate, good-enriched tail — but it is weak
alone because interior lakes occur in goods and bads alike.

The exemplar's true signature is the **joint corner**: high busy-ness × high
interior_frac. Because the two axes are near-independent, that conjunction is almost
empty — **only 1 of 1087 locations (the exemplar itself) dominates on both**
(`scatter_q4plane.png`). This is the concrete q4-hunting direction: **steer toward high
occupancy/spread AND high interior_frac simultaneously** — distributed complexity that
also composes negative space — a sparsely populated, good-enriched region no current
generator targets.

### Notes
- **symmetry is degenerate**: Julia sets are z→−z symmetric, so every `cz_` center view
  is exactly 180°-symmetric (symmetry=1.0). The axis separates center-views from
  descents, not aesthetics — drop it.
- The 2-term logistic tops out at AUC 0.641 (occupancy × spread_frac) — i.e. no axis pair
  predicts the labels, reconfirming these are past-q3 directions, not quality classifiers.

### Artifacts (`out/q4_axis/`)
`headline.txt`, `combos.txt`, `strips.png`, `scatter_q4plane.png`,
`scatter_occ_gini.png`, `exemplar_large.png`, `guard_sheet.png`,
`sheet_corpus_HIGH_{occupancy,interior_frac,spread_frac,mean_detail}.png`,
`sheet_corpus_LOW_tile_gini.png`, `sheet_goods_high_*.png`, `axes.jsonl` (per-location
axis values + 8×8 tile vectors). Regenerate: `build_pop.py` → `run_pass.py` → `analyze.py`.
