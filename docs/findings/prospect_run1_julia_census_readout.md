# Prospect run 1 — julia:multibrot census readout (stage 1 ∪ stage 2)

`prospect_run1_baserate_R_v1.json` merged via `merge_scores.py` into batch
`2026-07-17_prospect_run1_baserate_R_v1` (**86/86 filled, 0 conflicts**, matt,
2026-07-17). The census is now complete and labeled end-to-end.

**The census unit is stage 1 (julia only) ∪ stage 2.** Stage 1's batch holds 83
crops; its 61 julia:multibrot3/4/5 crops are the M/G/H half. Stage 2 adds the 86
R-band julia crops. **61 MGH + 86 R = 147 crops → 144 distinct locations** after 3
within-census coordinate collisions (all jm4, all identical-geometry duplicate
crops of the same location, listed below). **Location label = max over its crops.**

Which count each number uses is stated inline. Rates are over **locations (n=144)**
unless a table says "crops". **No thresholds changed — everything below is a
proposal.**

The 3 collisions (crop_id, band, score, p_good) — each pair is one location:
- `jmb4_..._105014_000010` / `000012` — M, both 3, p_good 0.258
- `jmb4_..._000819_000007` / `000009` — R, both 3, p_good 0.111
- `jmb4_..._071756_000012` / `000014` — R, both 3, p_good 0.128

Both crops in every collision carry the same score and p_good, so dedup is
label-neutral and AUC-neutral.

Location counts after dedup: **jm3 54 · jm4 51 · jm5 39 = 144.**

---

## 1. The headline test — full-range AUC, no range restriction

The whole point of the exercise. Stage 1's julia AUC 0.443 was range-restricted
(M/G/H only, R removed, p_good the stratifying variable) and read *below* chance;
I called that an attenuation artifact of range restriction. **The census removes
the defect: the entire p_good range [0, 1] is now labeled for these three
families.** Locations, n=144, p_good of the score-defining crop; AUC via
Mann-Whitney (0.5-credit ties), 5000-bootstrap 95% CI.

| subset | n | q3 | AUC(q3 vs rest) | AUC(≥2 vs 1) | Spearman(p_good,score) |
|---|---:|---:|---|---|---|
| **pooled julia** | 144 | 67 | **0.571 [0.48, 0.66]** | 0.574 [0.45, 0.70] | +0.132 (p=0.12) |
| julia:mb3 | 54 | 21 | 0.539 [0.38, 0.69] | 0.657 [0.46, 0.84] | +0.166 (p=0.23) |
| julia:mb4 | 51 | 24 | 0.601 [0.44, 0.75] | 0.496 [0.33, 0.66] | +0.146 (p=0.31) |
| julia:mb5 | 39 | 22 | 0.600 [0.42, 0.78] | 0.689 [0.50, 0.86] | +0.188 (p=0.25) |

**Per-family AUCs are weak and every CI straddles 0.5 — do not rank them.** They
are reported only to show the pooled number is not hiding one strong family.

**Where `t_good = 0.30` sits on the pooled ROC (q3 vs rest, P=67, N=77):**

| t_good | 0.10 | 0.15 | 0.18 | 0.20 | 0.25 | **0.30** | 0.40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPR (q3 recall) | 0.72 | 0.46 | 0.45 | 0.39 | 0.34 | **0.22** | 0.19 |
| FPR | 0.60 | 0.38 | 0.36 | 0.30 | 0.29 | **0.26** | 0.22 |
| kept | 94 | 60 | 58 | 49 | 45 | **35** | 30 |

### Which side of the fork — plainly

**Neither clean side; it lands on the retrain side, not the boundary side.**

- The full-range AUC is a **weak positive** (0.571 pooled) whose CI **includes
  chance** [0.48, 0.66], and Spearman is non-significant (p=0.12). This is *not*
  "AUC is good and only the boundary is misplaced" — the score does not rank q3
  from rest well enough for any boundary to exploit. The ROC is nearly diagonal:
  every operating point trades TPR for FPR almost one-for-one (t=0.10 → 0.72/0.60,
  t=0.30 → 0.22/0.26). There is no threshold that recovers the q3 mass cleanly.
- It is also *not* the inverted 0.443 stage 1 saw. Adding R **flipped the sign
  back to positive** — confirming the stage-1 inversion was the range-restriction
  artifact I flagged, not a real anti-correlation. But the recovered signal is
  negligible in magnitude.

So: **`t_good = 0.30` is badly misplaced** (it discards 78% of the q3s — TPR 0.22)
**and moving it cannot fix the gate**, because the underlying score carries almost
no ordering signal for these families. This is the retrain case, and it is exactly
what you expect from v6 having trained on **1 / 8 / 4** positive examples for
jm3/jm4/jm5. The "presentation" hypothesis loses its strongest piece of
support — the inversion it leaned on was range restriction — while
**data-starvation is the surviving explanation.**

**Interval, honored:** n=144 pooled, ~50/family. The pooled AUC CI is [0.48, 0.66];
per-family CIs are all wide and chance-straddling. Do not read a per-family AUC as
a number; read the pooled one as "weak-to-null."

## 2. The first honest rate for these families

**`P(q3 | surfaced by the prospect pipeline)`**, exact over locations, CP95. First
unbiased-given-descent rate this project has had for these families (the census is
the *complete* frozen-ledger surfaced population, not a sample):

| family | q3 / n (locations) | rate | CP95 | stage-1 floor | recorded twins (cyc 1–8) |
|---|---:|---:|---|---:|---:|
| julia:mb3 | 21 / 54 | 0.389 | [0.26, 0.53] | ≥0.24 | 5 |
| julia:mb4 | 24 / 51 | 0.471 | [0.33, 0.62] | ≥0.20 | 4 |
| julia:mb5 | 22 / 39 | 0.564 | [0.40, 0.72] | ≥0.21 | 0 |
| **pooled** | **67 / 144** | **0.465** | **[0.38, 0.55]** | — | 9 |

Every rate clears its stage-1 floor comfortably — jm3/jm4 land ~2× their floor —
so **R was not junk** (§3). Still **not `P(q3 | family)`**: the population is
descent-surfaced, not a random draw over the family. This is
`P(q3 | surfaced ∧ julia:mbN)`.

Against the pipeline's own recorded twin yield: run 1 stored **9** julia q3-twins
(mb3 5 · mb4 4 · mb5 0) over cycles 1–8, decoded at `t_good=0.30`. The census finds
**67** human-q3 among the same surfaced locations. The gate admitted a small
fraction of the real keepers — see §4.

## 3. Was R rich or barren? — **Rich.**

R (p_good < 0.15) was v6's confident-reject band. Per family, over locations
(band = the defining crop's stratum):

| family | R q3/n | R rate [CP95] | M/G/H q3/n | M/G/H rate [CP95] |
|---|---:|---|---:|---|
| julia:mb3 | 8/23 | 0.348 [0.16, 0.57] | 13/31 | 0.419 [0.25, 0.61] |
| julia:mb4 | 14/32 | 0.438 [0.26, 0.62] | 10/19 | 0.526 [0.29, 0.76] |
| julia:mb5 | 14/29 | 0.483 [0.29, 0.67] | 8/10 | 0.800 [0.44, 0.97] |
| **pooled** | **36/84** | **0.429 [0.32, 0.54]** | **31/60** | **0.517 [0.38, 0.65]** |

**R is rich — 43% human-q3 inside v6's confident-reject band.** The direction
(R 0.43 < M/G/H 0.52) is finally *correct* (monotone in p_good — the thing stage 1
couldn't produce), but the gap is small and the CIs overlap heavily: R is *nearly
as good as* M/G/H, not a lean tail below it. **The M-band cluster is therefore not
the real edge** — the gate sits deep inside a broadly rich region and rejects it.

**This is the julia analogue of the mandelbrot blindspot finding, and far bigger.**
Blindspot found mild-band 3/100 and v6-bad 0/100 for mandelbrot; here the v6-reject
band is **36/84 = 43% q3**. So the blindspot result not only generalizes to julia —
it is an order of magnitude worse for these families. Where mandelbrot's confident
rejects were genuinely mostly-bad, julia:multibrot's confident rejects are nearly a
coin-flip for q3. That is consistent with §1: for these families the score isn't
ordering quality, so its "reject" bin is uninformative rather than safe.

## 4. Downstream consequence — flagged, not recomputed

The census q3 rate among surfaced julia locations (0.39–0.56) is far above what
v6's decode admitted. On the pooled ROC, only **15 of 67 q3 locations** clear
`p_good ≥ 0.30` (TPR 0.22); the actual stored twin count is lower still (**9**).
So **run 1 under-recorded julia twins by roughly 4–7×** — the pipeline saw ~1/5 to
~1/7 of the keepers that were actually there.

Consequence for the price list (flag only): `twins/hook = 0.357` and mb3's
**~1000 sec/twin** are pricing **the gate, not the family**. The true keeper rate
per hook descent is the census rate, several times higher, so the real
seconds-per-*human*-keeper is several times *lower* than the list states. The
ordering of families may also shift (mb5 recorded 0 twins but is the *richest* at
0.564 human-q3). **Do not rebuild the price list off this** — but every sec/twin in
it should be read as an upper bound inflated by the decode gate, size ≈4–7×.

## 5. Eval-set qualification

**Qualifies.** The census is:
- **census-complete** — 147 crops = the complete pipeline-surfaced julia:multibrot
  population in the frozen ledger (batch.json's complete-condition; stage 1 and
  stage 2 compose against the same 1616-row ledger).
- **deduped to 144** distinct locations (3 within-census collisions removed).
- **disjoint from the 125 jm-band train locations** at all three keys —
  re-verified here independently: **0** overlap at exact identity, **0** at julia
  seed `c`, **0** at `parent_oid` (band has exactly 125 distinct locations; census
  has 144; intersection empty at every key). Different run epochs (band July 5–7,
  census July 16–17) make the identity/parent namespaces trivially disjoint, and
  seed-`c` disjointness is the substantive check — it holds.

So a v7 retrain can hold the 144 out as a genuine unbiased-given-descent eval for
these three families. **What it cannot do:** 144 locations across three families is
a *small* eval — per-family AUC CIs will remain wide (§1). And it is silent on
**native multibrot** and **mandelbrot** entirely; it certifies nothing outside
julia:multibrot3/4/5.

## 6. Native R — recommend

**Recommend labeling native R (or at least a stratified sample), because julia's R
just made the "native multibrot is barren" verdict look like a threshold
artifact.** 517 native crops remain deferred.

The logic: native's M/G/H was **6/22 q3**, with those q3s sitting *below* its
uncalibrated 0.50 gate (stage-1 §2). That is the same shape julia showed in M/G/H —
q3s below the gate — and julia's R band then turned out **43% q3** inside the
confident-reject region. If native multibrot shares julia's score pathology (and
§1 says the score doesn't order these degree-3/4/5 families), then native R is
likely rich too, and "c-plane multibrot is barren" is a decode-gate artifact rather
than a property of the family. **Only native R can settle that** — it is precisely
the band the barren verdict rests on.

**Price (eye-hours, same 200–400 crops/eye-hour assumption as stage 1):**
- Full native R census (517): **~1.3–2.6 eye-hours.**
- Stratified native R sample (~100/family, 300): **~0.75–1.5 eye-hours**, lands
  q3-rate CIs at ±0.08–0.10 — enough to tell "≈0" from "≈0.15", which is the whole
  question.

**Recommendation: the stratified 300-crop native R sample.** It is the
minimum that answers "is native barren, or just gated wrong?" at useful precision;
a full 517-census is only worth it if the sample comes back rich and native R
becomes a v7 eval candidate in its own right. If native R is also rich, "c-plane
multibrot barren" is dead and the retrain (§1) becomes the priority across native
*and* julia multibrot.

---

## Bottom line

1. **Full-range AUC is weak-positive-to-null** (pooled 0.571 [0.48, 0.66], Spearman
   p=0.12). This is the **retrain case**, not the boundary case: `t_good=0.30` is
   badly placed (discards 78% of q3s) but no threshold can fix a score that doesn't
   rank. The stage-1 inversion was confirmed a range-restriction artifact.
   Consistent with v6's 1/8/4 positive training examples.
2. **First honest rate `P(q3 | surfaced)`:** jm3 0.389 · jm4 0.471 · jm5 0.564 ·
   pooled 0.465 — all clear their stage-1 floors, jm3/jm4 by ~2×.
3. **R is rich, not barren — 43% q3 in v6's confident-reject band.** The julia
   analogue of the mandelbrot blindspot, ~10× larger. The M-band cluster is not the
   edge; the gate sits inside a broadly rich region.
4. **Run 1 under-recorded twins ~4–7×**; the price list prices the gate, not the
   family (flag only, not rebuilt).
5. **The census qualifies as a v7 eval** — complete, deduped to 144, disjoint from
   the 125 band locations at identity/seed-c/parent_oid (0/0/0, re-verified). Small;
   silent on native multibrot and mandelbrot.
6. **Label native R (stratified 300-crop sample)** — julia's R makes "native
   barren" look like a threshold artifact; native R is the only thing that settles it.
