# Phoenix grid — label adjudication (Phase B provisional verdicts settled)

**Date:** 2026-07-21 · **Batch:** `2026-07-21_phoenix_grid` (500 items) ·
**Run:** `data/discovery/phoenix_grid/grid` ·
**Tools:** `tools/phoenix/phoenix_label_analysis.py` (§0/§1/§2/§4a),
`tools/phoenix/phoenix_label_diversity.py` (§3/§4b) ·
**Artifacts:** `data/discovery/phoenix_grid/label_analysis.json`,
`.../diversity_ranker.json`, `out/phoenix_grid/good_look_medoids.png`.

Analysis + proposals only. No production threshold flipped, no ledger re-decoded on disk,
nothing retrained, no measure edited. The one code change is corpus bookkeeping: the batch
is now registered in `label_store.SIDECAR_LABELS` (§5).

---

## §0 Reachability — PASS

All **500/500** items resolve through the canonical reader (`label_store.resolve_score`
after registering the batch). `phoenix_grid.json` sidecar = 500 keys, 500 joined, **0
unlabeled / 0 skipped**; all 500 join `all_outcomes.jsonl` (reward-path p_good/p_notbad
truth). Human score distribution: **305 bad (1) / 118 okay (2) / 77 good (3)** — 15.4% good.

## §1 Decomposition — fertility is structural (CONFIRMED); machine magnitude inflated

Human between/within-seed decomposition, same ANOVA + cluster-bootstrap machinery as the
machine pass (`phoenix_decomp`), seed = grouping factor:

| variable (per-descent, group=seed) | ICC | 95% CI | σ²_between | σ²_within | verdict |
|---|---|---|---|---|---|
| **max_human** (mirror of max_p_good) | **0.724** | [0.591, 0.825] | 0.517 | 0.197 | between dominates |
| **mean_human** | **0.818** | [0.710, 0.893] | 0.426 | 0.095 | between dominates |
| per-image (every labeled img, group=seed) | 0.659 | [0.539, 0.749] | 0.391 | 0.203 | between dominates |
| — machine distinct_looks_within | 0.899 | [0.849, 0.934] | — | — | between dominates |
| — machine max_p_good | 0.965 | [0.919, 0.990] | — | — | between dominates |

**Verdict: CONFIRMED — between-seed variance dominates.** Every human ICC clears the
"ICC > 0.5 ∧ CI-floor > 0.5" bar; 0% of bootstrap draws went negative. Fertility is a
**structural, cacheable seed property**, not descent stochasticity — a fertile seed cannot
be amortized by re-descending, and the surrogate/memory design (spec §5.2) is on the right
footing. Note the human ICCs (0.66–0.82) sit **well below** the machine ICCs (0.90–0.965):
v7 over-separates seeds relative to human quality, so the machine number is a ceiling, not
the truth — but the direction and the go/no-go both hold.

### Human fertility map (confirm/overturn each machine finding)

- **Root-branch death — CONFIRMED, absolute.** root: **0/30 good, mean 1.00, 0 good seeds**
  (machine: 97% dead, mean_adm 0.17). The root skeleton locus is dead to humans, full stop.
  Good images come only from **cardioid (33/258, 12.8%)** and **period2 (44/212, 20.8%)** —
  period2 the mildly richer branch.
- **|p| band — CONFIRMED ordering.** good-frac **p1 0.185 > p0 0.135 ≈ p2 0.126** (machine
  max_p_good: band1 0.359 > band0 0.267 > band2 0.154). Mid-|p| is the sweet band; the near-M
  (small |p|) and large-|p| edges are both weaker. p0/p2 are nearly tied for humans (machine
  separated them more).
- **z₋₁ real vs non-real — OVERTURNED as a quality lever.** good-frac **nonreal 0.146 ≈ zero
  0.168** (flat, slight edge to z=0). The machine fertility map showed non-real z₋₁ producing
  **far more admissions** (e.g. p0|cardioid 3.71 vs 1.10 adm) — but that gradient is v7
  admission *volume*, not human *goodness*: the extra non-real admissions are not better to
  the human. The z-symmetry axis is a real morphology lever but **not** a fertility lever.
  (Caveat: the grid drew only z_zero and z_nonreal — no real-nonzero z — so "real vs non-real"
  here reads as "z=0 vs non-real".)

## §2 v7 on varied phoenix — it ranks well; the absolute t_good needs raising off 0.18

All four selection bands (HIGH/Q3/SUB/REJECT) present, so this is not selection-biased.

- **v7 ranks varied phoenix strongly.** AUC good-vs-rest **0.862**, AUC not-bad **0.910**,
  **Spearman(p_good, human) 0.676** (p≈5e-68, n=500). The calibration curve is monotone across
  p_good deciles (bottom-5 deciles good-frac ≈ 0; top three 0.34 / 0.54 / 0.40), with only a
  mild top-decile softening (the 0.80–0.90 decile edges out the 0.90–0.99 decile). **This
  overturns the readout's blanket "zero training coverage → suspect" framing for _ranking_:**
  v7 orders phoenix quality despite training only on fixed-Ushiki phoenix. The caveat holds
  only for the **absolute operating point**, below.
- **Proposed phoenix t_good (standard per-family F2 methodology, `derive_t_good`):
  t\* = 0.45** (F2_in 0.724, F2_oof 0.689; n_pos=77 ≥ 15 sufficiency floor). **Delta from the
  current production 0.50 is −0.05 → immaterial** (F2 0.724 vs 0.707, within noise; P 0.39
  R 0.92 at t\*). Recommendation: **leave production phoenix at baseline 0.50** — the derived
  optimum is indistinguishable from it.
- **The material fact is the grid's provisional 0.18, not the production 0.50.** At 0.18 the
  precision collapses to **0.19** (it admits essentially everything: 396/500 in-batch). Under
  a sane threshold the grid's headline intake roughly halves — re-decoding the **656-admission
  ledger** (report-only, not written):

  | t_good | ledger q3 | distinct looks |
  |---|---|---|
  | 0.18 (as-run) | 656 | 354 |
  | 0.45 (proposed) | 337 | **196** |
  | 0.50 (production) | 321 | 184 |

  So "656 admissions / 354 distinct looks" is a 0.18-inflated count; the honest intake at the
  production operating point is **~184–196 distinct looks**. The min-per-look price (0.66
  active-min/look) should be re-quoted against ~190, not 354, before it feeds the scheduler.

## §3 Theme diversity — real framing variety, **narrow motif vocabulary**

77 good images → **57 distinct good looks** at the library near-dup threshold (cos 0.974),
across **18 seeds**, from **cardioid + period2 only** (root contributes zero, per §1). But the
headline count is threshold-sensitive and the eyeball read matters:

| cut | distinct good looks |
|---|---|
| cos 0.974 (library near-dup) | **57** |
| cos 0.90 | 9 |
| cos 0.85 | 2 |
| cos 0.80 | 1 |

The medoid sheet (`out/phoenix_grid/good_look_medoids.png`) settles the one-theme question
**partially, not cleanly**: there is genuine variety in framing, zoom depth, and spiral
density — but the dominant motif is a single one, the **logarithmic double-scroll whorl** (the
seahorse-tail spiral), with only a handful of exceptions (star-burst / dendrite medoids
L14/L15/L23/L36/L55). That is exactly why the count collapses to 1–2 clusters below cos 0.85.
**Read:** the historical "one theme in many costumes" failure is *softened, not dispelled* —
the good phoenix corpus is a narrow morphology vocabulary (mostly spirals) delivered across
diverse framings. For wallpaper this is workable (framing + palette diversify the delivered
look), but phoenix should **not** be leaned on for morphological breadth. **multi-seed looks =
1** (near-every good look is seed-specific), reinforcing §1: variety lives across seeds, and
each fertile seed contributes its own whorl.

## §4 Feed-forward checks

- **§4a Surrogate viability — GO.** Spec §5.2 light head (logistic / linear over the logged
  cheap seed geometry — mandphoenix DE/iters, root_dist, |offset|, |p|, arg p, |z₋₁|, branch)
  vs human seed-fertility, LOSO over 91 seeds: continuous target (mean human) **Spearman
  0.620** (p≈6e-11); binary target (seed has ≥1 good) **Spearman 0.455, AUC 0.830** (good-seed
  base rate 0.198). Fertility is both **structural** (§1) and **predictable from pure geometry
  out-of-sample** — the two conditions the spec set for building the surrogate-ranked,
  memory-backed proposer rather than proposing fresh every time. Recommendation: **build the
  §5.2 surrogate.**
- **§4b Ranker transfer — orders phoenix, but adds nothing over p_good.** `pref_loc_v1`
  (v7+colored, held-out by construction — zero phoenix in its run2+dive+campaign1 training)
  on the 396 admitted labeled images: **Spearman(rank, human) 0.540** (p≈2e-31, AUC 0.763).
  It transfers. **But raw v7 p_good on the same 396 scores Spearman 0.631 — higher.** So the
  ranker is safe to keep in the intake-ordering slot (positive, real), yet for phoenix it is
  **not additive**; p_good alone already orders phoenix admissions well. No action — a v-next
  ranker refit could fold phoenix labels, but the intake ordering is not broken today.

## §5 Corpus bookkeeping

- Batch registered in `tools/corpus/label_store.py::SIDECAR_LABELS`
  (`2026-07-21_phoenix_grid → phoenix_grid.json`), so every canonical consumer reaches the 500
  labels. Sampling regime: **stratified from grid output → biased → train-side only** for any
  future CORN manifest; its varied-phoenix v7-calibration eval role (§2) lives in its own
  stratification, not the training union. Labels are **never-delete**.

---

## Bottom line

1. Fertility is structural and cacheable (ICC 0.72–0.82) — **build the §5.2 surrogate**; §4a
   confirms cheap geometry predicts it out-of-sample.
2. v7 **ranks** varied phoenix well (AUC 0.86); the only calibration fix is the operating
   point — **0.18 was far too low; ~0.45–0.50 is right** (leave production at 0.50). The grid's
   354-distinct-look headline is really ~190 at a sane threshold.
3. Skeleton map: **root is dead**, mid-|p| best, z-symmetry is not a quality lever. Draw
   cardioid+period2, mid-|p|, skip root.
4. Diversity is real in framing but **narrow in motif** (mostly log-spirals) — phoenix is a
   depth vein, not a breadth vein.
