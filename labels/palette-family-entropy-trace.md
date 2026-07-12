# Palette family collapse — entropy trace through the pipeline

Localize the "everything is electric-purple / magenta-lava / ice" collapse. Analysis-only, no renders. warm% missed this because the attractors straddle temperature; measure **family concentration**, not temperature.

**Family key:** assign each palette a family from its descriptors — the dramatic mood-family label where it exists, else a coarse bucket from `palette_features` (hue/chroma cluster). Report the key you used; keep it stable across stages.

**Trace family-entropy (and top-family share) through each stage**, same spirit as the warm% trace:
1. **Library** — the full 987 pool.
2. **gen-0 draw** — the source-stratified farthest-point picks that actually get sampled.
3. **pref top-K handoff** — the `top_k_pool` survivors pref-v3-gvo ranks up (the stage the warm% work already fingered as the cut where skew enters).
4. **selector output** — final emitted picks.

For each stage report: family-entropy, the top-3 families' share, and the count of families with ≥1 survivor. The shape of the drop localizes it — a cliff at stage 3 is the pref-v3 claim confirmed; a cliff earlier moves it generation-side.

**Self-reinforcement check:** trace the provenance of pref-v3-gvo's training queries (coldstart_v2 + warmstart_v1 + prefv2_dramatic_v1). Were the candidate pools those queries were drawn from *themselves* pref-ranked by an earlier pref version? If yes, quantify how often truly-diverse (non-attractor-family) palettes even appeared as query options — a bias that never sees diverse candidates can't be fixed by widening K.

Report the per-stage entropy/top-family tables, where the cliff is, and the query-provenance finding. No fix proposed yet — this is the localizer.

---

# FINDINGS (2026-07-11)

**The cliff is at stage 3 — pref-v3-gvo's ranking. The pref-v3 claim is confirmed, hard.**
gen-0 is the *most* diverse stage in the pipeline; the pref head collapses it to two
hue families in a single step. Generation is exonerated.

## Family key used

Hybrid, per spec, computed once per palette and held stable across stages:
- **dramatic roster mood-family** where it exists — recovered by joining each pool
  palette name to its `dramatic_palettes/results/<family>_c*.json` source file
  (13 roster families: fire-ice, jewel-earth, atmospheric-deep, antique-faded,
  high-key-luminous, pastel-iridescent, autumn-ember, oceanic, orchid-twilight,
  verdigris-copper, ember-in-ash, tonal-restrained, sapphire-rose). 260 of the 326
  dramatic-source palettes carry one; the 66 `span_*` palettes (deliberately
  cross-family) fall through to the bucket.
- **else a hue/chroma bucket** from `palette_features` — chroma-weighted circular-mean
  OKLab hue → 7 sectors {red-rust, amber-gold, green, teal-cyan, blue, violet-purple,
  magenta-pink} + neutral (mean-chroma < 0.03). Covers all extracted/curated + span.

Because the library is mostly extracted (hue-named) and the emission is mostly dramatic
(roster-named), the two namespaces don't overlap-count cleanly, so a **uniform
hue-bucket key** was also run as a robustness cross-check. Both keys put the cliff in the
same place, so the finding is invariant to the key.

## Per-stage tables

The funnel traced is one run: Library(987) → gen-0 draw(60, `sample_location.gen0_palettes`,
`GEN0_SOURCE_WEIGHTS` 75% dramatic) → pref top-K survivors (the deployed dramatic beam,
`2026-07-09_wallpaper_headbatch_dramatic_v1`, scorer_version=`v3_gvo`, curation_bucket=topk,
952 survivor-instances / 49 distinct palettes / 83 locations) → emission_selector.

**Hybrid key**

| stage | items | #fam ≥1 | H (bits) | H_norm | top-1 | top-3 share |
|---|---|---|---|---|---|---|
| 1 Library      | 987 | 21 | 3.85 | 0.876 | amber-gold 0.20 | 0.42 |
| 2 gen-0 draw   |  60 | 19 | 4.08 | **0.961** | amber-gold 0.15 | **0.30** |
| 3 pref top-K   | 952 | 17 | 3.13 | **0.766** | amber-gold 0.28 | **0.62** |
| 4 selector     |   6 |  3 | 1.46 | 0.921 | violet-purple 0.50 | 1.00 |

**Hue-bucket key (cross-check)**

| stage | items | #fam ≥1 | H (bits) | H_norm | top-1 | top-3 share |
|---|---|---|---|---|---|---|
| 1 Library    | 987 | 8 | 2.76 | 0.919 | amber-gold 0.28 | 0.57 |
| 2 gen-0      |  60 | 8 | 2.71 | 0.903 | amber-gold 0.28 | 0.63 |
| 3 pref top-K | 952 | 7 | 2.09 | **0.744** | amber-gold 0.39 | **0.85** |
| 4 selector   |   6 | 2 | 1.00 | 1.00 | magenta-pink 0.50 | 1.00 |

Stage-3 top-3 (hybrid): **amber-gold 0.28, violet-purple 0.25, fire-ice 0.09**.
In hue terms: **amber-gold 0.39 + violet-purple 0.36 = 75% of all survivors**. This is the
reported "electric-purple / magenta-lava / ice" collapse in family coordinates
(violet-purple = electric-purple; amber-gold + fire-ice = molten/lava; fire-ice's blue +
sapphire-rose = ice).

## Where the cliff is

- **1 → 2 (Library → gen-0): no drop; entropy RISES.** Per-source-bucket farthest-point
  spreads the 45 dramatic + 15 pool draws across ~19 families (H_norm 0.876 → 0.961,
  top-3 0.42 → 0.30). The 75%-dramatic gen-0 weighting is NOT the culprit — it draws
  diversely within dramatic.
- **2 → 3 (gen-0 → pref top-K): the cliff.** H_norm 0.961 → 0.766 (hybrid), 0.903 → 0.744
  (hue); top-3 share 0.30 → 0.62 (hybrid), 0.63 → 0.85 (hue). pref-v3-gvo concentrates the
  per-location top-K onto amber-gold + violet-purple.
- **The sharpest evidence — pref-v3-gvo's #1 pick per location:** across all 83 locations,
  the pref-rank-1 palette is **amber-gold in 48% and violet-purple in 45% — 93% of
  locations, 96% counting all six attractor families.** This is a *systematic ranking
  bias*, not a sampling artifact: in nearly every location, independent of geometry, the
  head declares a warm-gold or an electric-purple palette the single best.
- **3 → 4 (selector): no recovery.** The selector cannot repair family collapse — its
  MAP-Elites behavior axis is **fractal-family × dominant-color-cell**, not palette
  mood-family, and its only palette lever is a reuse cap (2). It subsamples the
  already-collapsed pool; both stage-4 top families are attractors (violet-purple,
  magenta-pink). (N=6 picks — entropy not meaningful at this count; the qualitative read
  is what stands. Stage-4 gate used the batch's persisted head_v2 scores, since the
  deployed v3 head is not persisted per-row; the selector, not the gate, is the funnel
  here. emit_v1's *shipped* default pool is the earlier humanq3 batch — stage 4 here
  applies the selector to the dramatic funnel for a single consistent trace.)

## Self-reinforcement / query-provenance finding

pref-v3-gvo trains on a 999-query union of three batches (`scorer/data.py`): coldstart_v2
(~199 q), warmstart_v1 (~200 q), prefv2_dramatic_v1 (600 q). Were the candidate pools
those queries drew from themselves pref-ranked by an earlier pref version?

- **warmstart_v1 — YES, a closed loop.** `query_batch_gen.py` gates every candidate
  through the **pref-v1** scorer before the labeler sees it: palette queries score 48
  palettes with v1 and keep the **top-18 by v1** (then FP to 6); param queries let **v1
  pick the single palette (top-1 of 48)**; joint keeps a v1 top-k. Truly-diverse
  non-attractor palettes that v1 disliked are filtered out *before labeling*. This is
  genuine self-reinforcement — but it is only ~1/6 of the training union and only through
  the weakest (v1) head.
- **coldstart_v2 — no.** Palette/joint queries are the cold bootstrap (no pref prior
  existed); param selection is render-space CIEDE2000 FP. Pref-free options.
- **prefv2_dramatic_v1 — no, and deliberately diverse.** `pref-v2 is NOT consulted in
  selection`; candidates are farthest-point over dramatic+pool. Measured on its surviving
  ledger (600 q × 6 = 3600 options): **H_norm 0.953, 21 families, 62% of options are
  non-attractor-family.** The labeler *did* see diverse candidates, in bulk.

**Conclusion on the "can't be fixed by widening K" hypothesis:** the training set as a
whole is NOT starved of diverse candidates — its 60%-majority batch is deliberately
family-balanced and the labeler saw 62% non-attractor options. So the stage-3 collapse is
**not primarily a candidate-starvation artifact.** pref-v3-gvo *saw* diverse families and
still learned to rank amber-gold / violet-purple #1 in 93% of locations. That points to a
**learned preference** (either the human good/okay tiers themselves favor these families,
or the head generalizes to them) as the dominant driver, with warmstart's v1-gating a
real but minor secondary loop. Widening K downstream will not touch it; the bias lives in
the ranking head's learned scores, not in what it was allowed to see.

_Repro: `tools/wallpaper/family_entropy_trace.py` (stages 1–4) + the per-location #1 and
options-coverage one-offs in the session._
