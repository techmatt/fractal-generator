# Task: julia:multibrot census readout

`prospect_run1_baserate_R_v1.json` is written. Merge it via the canonical path, then combine with stage 1. **The census is now complete and labeled end-to-end: stage1 (61) ∪ stage2 (86) = 147 rows → 144 distinct locations** after the 3 within-census collisions you flagged. Dedup to 144, location label = max over crops, and say which count each number uses.

Report to `docs/findings/`. **No threshold changes. Propose, don't apply.**

---

## 1. The headline test — full-range AUC, no range restriction

This is the point of the whole exercise. Stage 1's AUC ≈ 0.49 was **range-restricted by construction** — M/G/H only, R removed, and `p_good` was the stratifying variable. I said attenuation was expected and not to over-read it. **The census has no such defect: the entire `p_good` range [0, 1] is now labeled for these three families.**

So compute, per family and pooled, over all 144:

- **AUC(q3 vs rest)** and **AUC(≥2 vs 1)** against `p_good`, plus Spearman.
- The ROC, and where `t_good = 0.30` actually sits on it.

**This is the fork the retrain decision hangs on. State which side it lands on plainly:**

- **AUC is good and only the boundary is misplaced** → v6 ranks these families fine, `t_good` is the fix, and the "presentation" and "data starvation" hypotheses both lose their main support.
- **AUC is ≈0.5 across the full range** → no threshold placement can help, because the score carries no signal for these families. That's the retrain case, and it's consistent with v6 having trained on **1 / 8 / 4 positive examples** for jm3/jm4/jm5 respectively.

Put an interval on the AUC — n=144 pooled, ~50 per family. Per-family AUCs will be weak; say so rather than ranking them.

## 2. The first honest rate for these families

Exact, not a floor: **`P(q3 | surfaced by the prospect pipeline)`** per family, with CP95. This is the first unbiased-given-descent number anything in this project has had outside mandelbrot's `loose0`.

Compare against stage 1's floor bounds (jm3 ≥0.24 · jm4 ≥0.20 · jm5 ≥0.21) and against the pipeline's own recorded twin yield. **Still never `P(q3 | family)`** — the population is descent-surfaced.

## 3. Was R rich or barren?

R was v6's confident-reject band. Give its q3 rate per family with CP95, and compare to M/G/H.

- **Rich** → the julia analogue of the mandelbrot blindspot finding, and much bigger: blindspot found mild 3/100 and v6-bad 0/100 for mandelbrot.
- **Barren** → the M-band cluster is the real edge and the gate merely sits on the wrong side of it.

Either way, say which, and say what it implies about whether the mandelbrot blindspot result generalizes.

## 4. Downstream consequence — flag it, don't recompute it

If the true q3 rate among surfaced julia locations is well above what v6's decode admitted, then run 1 **under-recorded twins**, and the price list's `twins/hook = 0.357` and mb3's ~1000 sec/twin are pricing the gate rather than the family. Note the implication and its size; don't rebuild the price list.

## 5. Eval-set qualification

State whether the census now qualifies as an eval set for a retrain: census-complete, deduped to 144, disjoint from the 125 band locations at identity / seed `c` / `parent_oid` (you verified 0 at all three). Note what it can't do — 144 locations across three families is a small eval, and it says nothing about native multibrot or mandelbrot.

## 6. Native R — recommend

517 crops still deferred. Recommend for or against based on what julia's R just showed, priced in eye-hours. Native's M/G/H was 6/22 q3 with q3s below its uncalibrated 0.50 gate, so the "c-plane multibrot is barren" verdict may be a threshold artifact — say whether native R would settle that.
