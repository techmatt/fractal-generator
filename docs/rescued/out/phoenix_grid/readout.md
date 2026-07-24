# Phoenix Phase B — seed-grid readout

> **ADJUDICATED (2026-07-21).** The 500-item batch is labeled and the provisional verdicts
> below are settled in **`docs/findings/phoenix_grid_labels.md`**. Headlines: between-seed
> dominance CONFIRMED (human ICC 0.72–0.82, magnitude below the machine 0.90–0.965); v7 *ranks*
> varied phoenix well (AUC 0.86) so the "zero training coverage" caveat holds only for the
> absolute operating point — proposed t_good **0.45 ≈ production 0.50**, and the as-run **0.18
> was far too low** (precision 0.19), so the 354-distinct-look intake is really **~190** at a
> sane threshold; root-branch death CONFIRMED, z-symmetry is NOT a quality lever; surrogate
> go/no-go = **GO**; diversity is real in framing but a **narrow (mostly log-spiral) motif**.
> The per-section notes below are the machine (v7) read; see the findings doc for the human
> adjudication. Governing: `docs/design/phoenix_seed_sampler_spec.md` §5.1, `prompts/phoenix_phase_b.md`.

## Run

- Grid: **91 seeds x up to 4 descents** (5 walks/descent), depth [4, 14], t_good=**0.18** (provisional), scorer **v7**. Stopped: `budget` — **233 cumulative active min** (4-hour active-time cap), run resumed once after a scratch-disk fix.
- Descents scored: **361** / 1805 walks → **656 admissions** (keep-every-q3), **354 distinct looks** (morph embed, cos 0.974).
- **Realized min-per-look price (phoenix): 0.66** active-min / distinct look — the prior the measure/scheduler will want.

## Variance decomposition (spec §5.1, step 0) — ADJUDICATED (human ICC 0.72–0.82 CONFIRMS between-seed dominance; see findings doc)

Method: one-way **random-effects ANOVA** (seed = grouping factor, unbalanced-safe); ICC = σ²_between/(σ²_between+σ²_within) is the between-seed variance share; CIs are **nonparametric cluster bootstrap** (resample whole seed-clusters). The spec's prior is that **between-seed dominates** (a phoenix has a thin z-repertoire, so variety lives across seeds and a fertile seed can't be amortized by re-descending).

**distinct_looks_within** (a=90 seeds, N=360 descents): ICC = **0.899** (between-seed variance share), 95% CI [0.849, 0.934]; σ²_between=3.0743, σ²_within=0.3444; bootstrap draws with σ²_between<0: 0%. → _between-seed dominates_.

**max_p_good** (a=90 seeds, N=360 descents): ICC = **0.965** (between-seed variance share), 95% CI [0.919, 0.990]; σ²_between=0.1030, σ²_within=0.0038; bootstrap draws with σ²_between<0: 0%. → _between-seed dominates_.

## Fertility map — yield by parameter-space region

Which skeleton regions produce keepers and which are dead (0 admissions). `stratum` = `p<|p|-band>|<branch>|z_<class>` (the draw cell). **Human adjudication (findings doc):** root-death and the mid-|p| peak CONFIRMED; the non-real-z₋₁ advantage seen here is v7 admission *volume*, NOT human goodness (good-frac nonreal 0.146 ≈ zero 0.168) — z-symmetry is not a quality lever.

### by stratum
| stratum | descents | dead % | mean adm | mean distinct | mean max p_good |
|---|---|---|---|---|---|
| p0|cardioid|z_nonreal | 21 | 19% | 3.71 | 3.05 | 0.425 |
| p0|cardioid|z_zero | 20 | 70% | 1.10 | 0.85 | 0.249 |
| p0|period2|z_nonreal | 20 | 0% | 4.90 | 3.60 | 0.588 |
| p0|period2|z_zero | 20 | 40% | 2.30 | 1.90 | 0.319 |
| p0|root|z_nonreal | 20 | 100% | 0.00 | 0.00 | 0.000 |
| p0|root|z_zero | 20 | 100% | 0.00 | 0.00 | 0.016 |
| p1|cardioid|z_nonreal | 20 | 0% | 3.90 | 3.05 | 0.638 |
| p1|cardioid|z_zero | 20 | 10% | 3.75 | 3.25 | 0.512 |
| p1|period2|z_nonreal | 20 | 20% | 3.55 | 3.20 | 0.623 |
| p1|period2|z_zero | 20 | 45% | 2.65 | 1.55 | 0.372 |
| p1|root|z_nonreal | 20 | 100% | 0.00 | 0.00 | 0.000 |
| p1|root|z_zero | 20 | 100% | 0.00 | 0.00 | 0.011 |
| p2|cardioid|z_nonreal | 20 | 40% | 2.90 | 2.10 | 0.442 |
| p2|cardioid|z_zero | 20 | 80% | 0.75 | 0.50 | 0.088 |
| p2|period2|z_nonreal | 20 | 70% | 1.10 | 0.45 | 0.090 |
| p2|period2|z_zero | 20 | 80% | 1.00 | 1.00 | 0.212 |
| p2|root|z_nonreal | 20 | 80% | 1.00 | 0.20 | 0.078 |
| p2|root|z_zero | 20 | 100% | 0.00 | 0.00 | 0.015 |

### by branch
| branch | descents | dead % | mean adm | mean distinct | mean max p_good |
|---|---|---|---|---|---|
| cardioid | 121 | 36% | 2.69 | 2.14 | 0.392 |
| period2 | 120 | 42% | 2.58 | 1.95 | 0.367 |
| root | 120 | 97% | 0.17 | 0.03 | 0.020 |

### by |p| band
| |p| | descents | dead % | mean adm | mean distinct | mean max p_good |
|---|---|---|---|---|---|
| |p| band 0 | 121 | 55% | 2.02 | 1.58 | 0.267 |
| |p| band 1 | 120 | 46% | 2.31 | 1.84 | 0.359 |
| |p| band 2 | 120 | 75% | 1.12 | 0.71 | 0.154 |

## Admissions ledger — intake-ready check

Predicate (library intake): `is_current_decoded` (scorer_version==`v7`) ∧ decoded_class==3 ∧ guard_pass ∧ distinct ∧ full (c,p,z₋₁) identity stamped.

- Ledger rows: **656**; identity-stamped: **656/656**; **intake-ready: 354/656**.
- Non-ready breakdown: `{'not_distinct': 302}`.
- **Adjudication caveat:** the 656 admissions / 354 distinct looks are decoded at the as-run t_good **0.18** (precision 0.19 vs human). Re-decoded at the proposed **0.45** the ledger is **337 q3 / 196 distinct** (production 0.50: 321 / 184). The scheduler prior should use **~190** distinct looks, not 354. See findings doc §2.
- Ledger: `data\discovery\phoenix_grid\grid\outcome_ledger.jsonl`  |  features: `data\discovery\phoenix_grid\grid\outcome_feats.npz` (1280-D v7)  |  distinct-look tally: `data\discovery\phoenix_grid\grid\distinct_looks.npz`. **Confirmed intake-ready.**

## Visual sheets (standing habit)

- Admissions (top 24 by p_good): `out\phoenix_grid\admissions_sheet.png`
- Guarded rejects: `out\phoenix_grid\rejects_sheet.png`

## Label batch

- Batch `2026-07-21_phoenix_grid`: **500 items**, 91 seeds, realized bands `{'HIGH': 243, 'Q3': 153, 'SUB': 67, 'REJECT': 37}`.
- Location: `C:\Code\fractal-generator\data\label_corpus\batches\2026-07-21_phoenix_grid` (images.jsonl + batch.json + crops/ + scores.json).
- Render identity: fractal_type **+ (c,p,z₋₁)** stamped in every render block; identity round-trip asserted + **Guard B byte-reproducibility PASS** (the baserate_v1 three-axis check).
- Label sheets: `out/phoenix_grid/label_sheets/`.

## Next — DONE

The batch is labeled and joined. The t_good re-derivation, decomposition adjudication, and
spec §5.2 surrogate go/no-go are settled in **`docs/findings/phoenix_grid_labels.md`**
(surrogate = **GO**; build the §5.2 surrogate-ranked, memory-backed proposer — draw
cardioid+period2 at mid-|p|, skip root).
