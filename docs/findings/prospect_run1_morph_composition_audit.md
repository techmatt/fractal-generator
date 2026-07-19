# Prospect run 1 — morph_clip composition audit

Read-only audit of the run-1 location library. Run still live (PID 15352); nothing written to
the store or ledger.

**Watermark: 106 library records** (`data/library/records.jsonl` at 106 lines) = cycles 1–8
committed (93) + cycle-9 partial (13). Composition/clustering use all 106; the per-cycle yield
curve uses cycles 1–8 only (cycle 9 is mid-flight and excluded). morph_clip = 768-d grayscale
CLIP, **100% vector coverage** (106/106 join the base+shard embedding store).

Population by source: **phoenix 44 · julia twin 46 · c-plane base 16** (mandelbrot 14, multibrot4 2).
No mb3/mb5 base, no mb5 twin — consistent with the rebalance readout.

---

## 0. Metric validity (checked first, per the method notes)

**The metric is NOT lying, but absolute cosine is compressed (CLIP-cone) — only relative structure is usable.**

- The **entire** 106-record library spans cosine **[0.557, 0.987]** (farthest pair phoenix↔mandelbrot
  at cos 0.557). Nothing is ever "cosine-far"; CLIP embeddings sit in a narrow cone. Do **not** read
  an absolute cosine as a similarity — only the *ordering* is meaningful.
- **Controls separate cleanly** (relative): known-distinct cross-family pairs land at **cos 0.75–0.84**
  (mandelbrot↔multibrot4 0.750, phoenix↔mandelbrot 0.772, m4↔julia:mb3 0.809), while genuine near-dups
  land at **cos 0.987** (nearest pair: two julia:mb3 from cyc5). A ~0.15–0.23 cosine gap between
  "different family" and "near-duplicate" is large and consistent → the metric ranks correctly.
- **Eyeball confirms** (contact sheet `out/wallpaper/morph_audit/cluster0_size9.png`): the tightest
  cluster (9 members, cos>0.974) is visibly the *same* phoenix spiral at different zoom/framing. Near
  = near.
- **The distance distribution is a smooth right-skewed continuum, no bimodal valley** (bulk at
  median cos 0.851; near-dup tail 7 pairs >cos0.98, 23 pairs >cos0.974, ramping). **So there is no
  natural cut** — every cluster count below is threshold-sensitive and reported with sensitivity.
  Primary cut = **cos>0.974 (d<0.026)**, the threshold the prior curation pass used
  ([[morphology-dedup-pass]]); a looser perceptual cut cos>0.95 is shown alongside.

---

## 1. Composition

### Groups at the strict near-dup cut (cos>0.974)

**86 distinct morphology groups from 106 records** — 75 singletons, 31 records in 11 multi-member
groups, largest group **9 (all phoenix)**.

Distinct-morphology rate by source:

| source | records | groups | distinct rate |
|---|---:|---:|---:|
| c-plane base | 16 | 16 | **1.00** |
| julia twin | 46 | 41 | **0.89** |
| phoenix | 44 | 30 | **0.68** |

Sources barely mix: **1** cross-source cluster in the whole library (1 mandelbrot base + 2
julia:mandelbrot twins — a Julia inheriting its parent's structure). Base and twin morphology are
essentially disjoint from each other and from phoenix.

### Threshold sensitivity (single-linkage connected components)

| cut | cos> | clusters | largest | singletons |
|---|---:|---:|---:|---:|
| d<0.020 | 0.980 | 100 | 5 | 97 |
| **d<0.026** | **0.974** | **86** | **9** | **75** |
| d<0.030 | 0.970 | 72 | 16 | 64 |
| d<0.050 | 0.950 | 41 | 39 | 33 |
| d<0.075 | 0.925 | 15 | 84 | 10 |

Cluster count is highly cut-dependent (86 → 15 across a narrow cosine band 0.974→0.925) — a direct
consequence of the no-valley continuum. Treat any single number as "at this cut," not absolute.

### Phoenix — is it collapsing into one blob?

**Yes, effectively — the numbers understate it and the eyeball settles it.** Phoenix is the tightest
source: intra-phoenix median distance cos **0.938** vs library-wide median cos 0.851, and it collapses
monotonically with the cut:

| cut cos> | 0.980 | 0.974 | 0.970 | 0.950 | 0.925 | 0.900 |
|---|---:|---:|---:|---:|---:|---:|
| phoenix groups | 40 | 30 | 17 | 5 | 2 | **1** |

The contact sheet `out/wallpaper/morph_audit/phoenix_all.png` (all 44) is decisive: **every tile is
the same morphological theme** — a hooked logarithmic spiral with a serrated/toothed edge on a smooth
gradient — re-framed at different zoom, rotation, and spiral position. The "30 distinct groups" at
cos>0.974 are splitting on *framing*, not morphology; a human would call these ~**one look**. Phoenix
is a variety-poor, single-plane garnish (confirms [[phoenix-tgood-yield-study]],
[[visual-dup-morphology-measured]]). It is ~43% of records but contributes ≈one morphological theme.

---

## 2. Distinct-morphology yield curve (cycles 1–8)

Online single-link novelty: a record opens a new morphology group iff it is not a near-dup (at the
cut) of any earlier record, processed in cycle order.

**Strict cut cos>0.974** — 78 groups from 93 survivors (16% morph-dup):

| cyc | records | new morph | active h | newMorph/hr | cum |
|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 16 | 1.12 | 14.3 | 16 |
| 2 | 16 | 15 | 1.05 | 14.3 | 31 |
| 3 | 21 | 18 | 1.23 | 14.6 | 49 |
| 4 | 4 | 1 | 0.96 | 1.0 | 50 |
| 5 | 13 | 11 | 1.03 | 10.6 | 61 |
| 6 | 6 | 3 | 0.91 | 3.3 | 64 |
| 7 | 13 | 13 | 1.06 | 12.2 | 77 |
| 8 | 3 | 1 | 0.90 | 1.1 | 78 |

**Loose cut cos>0.95** — 49 groups from 93 survivors (47% morph-dup):

| cyc | new morph | newMorph/hr | cum |
|---:|---:|---:|---:|
| 1 | 12 | 10.7 | 12 |
| 2 | 12 | 11.5 | 24 |
| 3 | 8 | 6.5 | 32 |
| 5 | 9 | 8.7 | 42 |
| 7 | 5 | 4.7 | 49 |
| 8 | 0 | 0.0 | 49 |

### Verdict: decaying **faster** than coords — and the gap widens at perceptual thresholds

- **At the strict cut, morphology ≈ coords.** Only 16% of coord-survivors are morph-near-dups, so
  newMorph/hr (~14 early) tracks records/hr (15.2 early) and decays at ~the same rate.
- **At the loose (perceptual) cut, morphology decays faster.** 47% of survivors are morph-similar to
  something already stored; cumulative distinct morphology **flattens** (cyc7→8 added 5 then **0** new
  at cos>0.95, while records/hr was still 11.3 and coords kept arriving). So the perceptually-relevant
  novelty rate is falling off ahead of the coordinate rate.
- Per-cycle numbers are **noisy** (cyc4=4, cyc8=3 records) — the trend is directional, not a clean
  monotone over 8 cycles. But the direction is consistent at the loose cut.

**The number that matters for cell coverage is the loose-cut one, and it is decaying faster than
records/hr (15.2→11.3).** Coordinate-distinctness overstates how much *visual* variety the library is
banking, and the overstatement grows with cycles.

---

## 3. Are coord-dups and morph-dups the same population?

**No — they are complementary, not the same.** All 106 records are coord-dedup *survivors* (present in
the store). Of the cycle-1–8 survivors (93), the fraction that are morph-near-dups of an **earlier
stored** record:

- **16.1%** at cos>0.974 (strict, near-exact frame dup)
- **47.3%** at cos>0.95 (perceptual "same look")

So coordinate dedup passes through 16–47% morphological repeats that a morph-dedup would catch. The two
layers catch different things: **coord-dedup catches exact coordinate re-visits** (the 27 rejected in
the reconciliation readout), **morph-dedup would additionally catch visually-similar locations at
different coordinates** — the "coord-distinct-but-visually-identical" population that is real here
(confirms [[visual-dup-morphology-measured]]). A morph gate would be additive to, not redundant with,
the coord gate.

---

## Honest limits

1. **Small library (106 records).** All verdicts are directional; the per-cycle yield curve is noisy
   (two cycles have ≤4 records).
2. **No natural threshold valley** — the distance distribution is a smooth continuum, so every cluster
   count is a function of the cut. Sensitivity is reported throughout; don't quote a single number as
   absolute.
3. **CLIP-cone compression** — absolute cosine is uninformative (whole library in [0.557, 0.987]); only
   relative ordering is used. Controls confirm the ordering is correct.
4. **morph_clip at 0.974 splits on framing**, so "distinct group" counts *over-state* perceptual
   variety — most visible for phoenix, where 30 metric-groups are ≈1 human-theme. The contact sheets are
   the tiebreaker and are the reason the phoenix verdict is "collapsing" despite 30 strict-cut groups.

## Bottom line

1. **Library composition (106 recs, cos>0.974): 86 groups**, base fully distinct (1.00), twins mostly
   distinct (0.89), **phoenix the tight one (0.68 — and ≈1 theme to the eye)**.
2. **Phoenix (~43% of records) is collapsing** — one hooked-spiral morphology re-framed; it inflates
   record count far more than morphological coverage.
3. **Distinct-morphology yield decays faster than records/hr at any perceptual cut**, and cumulative
   distinct morphology is already flattening at cos>0.95 by cycle 8. records/hr (15→11) is an optimistic
   proxy for cell coverage.
4. **Coord-dedup ≠ morph-dedup**: 16–47% of coord-survivors are morph-repeats a morph gate would catch.
   A visual-dedup gate would be additive. (No change made — reporting only.)

Contact sheets (regenerable, disposable `out/`): `out/wallpaper/morph_audit/{cluster0_size9,cluster1_size3,cluster2_size3,phoenix_all}.png`.
