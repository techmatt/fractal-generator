# Prospect run 1 — stage-1 label readout (M/G/H)

Batch `2026-07-17_prospect_run1_baserate_v1`, 83 crops labeled (matt, 2026-07-17),
merged via `merge_scores.py` (83/83 filled, 0 conflicts). This is stage 1: a **census
of the M/G/H p_good strata** (0.15–1.01) across the six degree-3/4/5 multibrot + julia
families. **R (0.0–0.15) is deferred: 603 of 686 rows, 88% of the population, is
unlabeled.** No thresholds were changed — everything below is a proposal.

Every number here is conditioned on **(surfaced by the prospect pipeline) ∧ (p_good ≥
0.15)**. See §5 before quoting any rate.

---

## 0. What the eye saw (score distribution, M/G/H census)

| family | t_good | n | q3 | q2 | q1 | q3 rate (M/G/H) [CP95] | M/G/H = % of family |
|---|---:|---:|---:|---:|---:|---|---:|
| multibrot3 | 0.50 | 9 | 1 | 6 | 2 | 0.11 [0.00, 0.48] | 9/239 = **3.8%** |
| multibrot4 | 0.50 | 9 | 4 | 3 | 2 | 0.44 [0.14, 0.79] | 9/166 = **5.4%** |
| multibrot5 | 0.50 | 4 | 1 | 3 | 0 | 0.25 [0.01, 0.81] | 4/134 = **3.0%** |
| julia:mb3 | 0.30 | 31 | 13 | 12 | 6 | 0.42 [0.25, 0.61] | 31/54 = 57% |
| julia:mb4 | 0.30 | 20 | 11 | 7 | 2 | 0.55 [0.32, 0.77] | 20/54 = 37% |
| julia:mb5 | 0.30 | 10 | 8 | 2 | 0 | 0.80 [0.44, 0.97] | 10/39 = 26% |

CP95 = Clopper–Pearson exact 95% interval. The "% of family" column is the leash: for
**native** multibrot we have labeled 3–5% of each family and the rate is an M/G/H-only
statistic, not a base rate (§5).

---

## 1. The live question — is jm3/4/5's `t_good = 0.30` right?

**Verdict: too tight, for all three julia families. q3s live well below 0.30 — the
opposite of what the gate assumes.** The revival lowered the gate to 0.30 to admit more;
the data says even 0.30 sits *above* where the julia q3s are.

Per family (M/G/H strata, CP95):

**julia:mb3** (n=31, M band = 17 rows)
- H (p_good 0.72–0.88): **0/5** q3 — the model's most-confident picks, all rejected.
- G (0.40–0.66): 3/9 = 0.33 [0.07, 0.70]
- M (0.16–0.33): **10/17 = 0.59 [0.33, 0.82]**
- q3 p_good spans **0.168 → 0.657**. **10 of 13 q3s (77%) sit below t_good=0.30.**

**julia:mb4** (n=20, M band = 7)
- H: 1/2 (can't tell, n=2)
- G (0.40–0.70): 7/11 = 0.64 [0.31, 0.89]
- M (0.19–0.37): 3/7 = 0.43 [0.10, 0.82]
- q3 p_good 0.258 → 0.740. 3 of 11 q3s below 0.30.

**julia:mb5** (n=10 — whole family stage-1 is 10 crops)
- G: 2/3; M: 6/7. Both intervals are wide; the family total 8/10 = 0.80 [0.44, 0.97].
- q3 p_good 0.194 → 0.538. 4 of 8 q3s below 0.30.
- **Refuse a per-stratum verdict here: n=3 and n=7 carry nothing.** The only defensible
  read is "q3-dense across the labeled range," direction only.

**Pooled julia** (justified: all three share the *same* 0.30 override; pooling assumes
the p_good→quality miscalibration is common across degree — reasonable given the M-band
signal repeats in all three): 32 q3 / 61 = 0.52. **17 of 32 julia q3s (53%) sit below
the 0.30 gate.** The M band (0.15–0.40) is the *richest* stratum, not the leanest:
pooled M q3 rate 19/31 = 0.61 vs G 12/23 = 0.52 vs H 1/7. Quality is going the *wrong
way* against p_good inside M/G/H.

**Proposal (not applied):** the gate is cutting real q3s. But note §3 — the p_good
ordering itself is broken in this range, so "lower t_good to 0.16" is not obviously the
fix; it would admit the whole M band, q1s and all. The honest conclusion is that
**0.30 is mis-placed relative to the julia q3 mass, and the low end (R) must be labeled
before any new threshold is chosen** (§4). Do not re-set t_good off this batch alone.

---

## 2. Native multibrot — the good band is **not** empty (refuted)

mb3/mb4/mb5 stage-1 = 9/9/4 crops covering each family's entire M+G+H. The assumption
was that the native good band is empty. **The eye disagrees: 6 of 22 native crops are
human-q3.** Called out specifically:

| image_id | family | stratum | p_good | note |
|---|---|---|---:|---|
| `mb4_20260717_011014_000002` | mb4 | H | 0.856 | high-confidence *and* real q3 |
| `mb4_20260717_105014_000009` | mb4 | G | 0.626 | q3 |
| `mb4_20260717_000819_000006` | mb4 | M | 0.327 | q3 **below** native t_good 0.50 |
| `mb4_20260717_071756_000011` | mb4 | M | 0.257 | q3 **below** 0.50 |
| `mb3_20260717_070547_000018` | mb3 | G | 0.498 | q3, right at the 0.50 line |
| `mb5_20260717_115946_000002` | mb5 | M | 0.364 | q3 **below** 0.50 |

mb4 is the standout: **4/9 q3**, one at the top of H and two in the M band below its own
gate. mb3 and mb5 contribute one q3 each. So native multibrot's good band is real but
thin, and — like julia — its q3s reach down into M, below the 0.50 native gate. The
"native band is empty" prior is refuted; the "native is barren" version survives only in
the weak sense that the *rate* is low (§5) and n is tiny.

---

## 3. Does `p_good` rank correctly here? — **No ordering signal. Say it loudly.**

v6's standing claim is that ranking is sound and only the decode boundary is off. **In
this labeled range that claim is not supported, and for julia it is mildly inverted.**

| subset | n | Spearman(p_good, score) | AUC(q3 vs rest) | AUC(≥2 vs 1) |
|---|---:|---:|---:|---:|
| all M/G/H | 83 | +0.00 | 0.494 | 0.518 |
| julia only | 61 | −0.06 | **0.443** | 0.560 |
| native only | 22 | +0.11 | 0.661 | 0.403 |

AUC 0.494 overall is pure chance. Julia AUC 0.443 is **below** chance — within M/G/H,
higher p_good is if anything *anti*-correlated with human-q3. The cleanest single
picture is julia:mb3's H band: the five highest-p_good julia crops in the whole family
(0.72–0.88) scored 2,1,2,1,2 — **zero q3s** — while the bottom-of-M crops (p_good ≈
0.17–0.29) are mostly q3.

**Caveat, honored:** this is a range-restricted sample (M/G/H, R removed, and p_good was
the stratifying variable), so attenuation toward 0 is *expected* — a weak correlation
here would not be alarming. But this is not weak-positive; it is flat-to-inverted. Range
restriction attenuates a real signal toward zero; it does not flip it negative. So the
result points past the threshold at the **scored presentation itself** (single
`twilight_shifted` palette, 640×360 ss2 — the v6 deploy geometry). Within M/G/H, v6's
p_good does not tell q3 from q2/q1 for these six families. That reframes the batch:
**the problem is not (only) where the decode line sits, it's that the score isn't
ordering quality in this regime.** Native's 0.661 is a hint of residual order but n=22
and it rests heavily on the two extreme mb4 crops.

---

## 4. R — worth it, and specifically *how*

M/G/H is **not** barren for julia, and the q3s **cluster at the bottom of M** with no
visible falloff toward the M/R boundary (jm3 q3s continue to p_good 0.168, the floor of
M). Per the plan, "if q3s cluster at the bottom of M, R matters much more" — that is
exactly the situation. Combined with §3 (p_good can't rank inside M/G/H), we have **no
basis to assume R is empty**; the model's low-p_good bin is not a quality signal we can
trust to be low.

**Recommendation: label R, but asymmetrically — census julia, sample native.**

The R population is dominated by native multibrot:

| band | mb3 | mb4 | mb5 | jm3 | jm4 | jm5 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| R (deferred) | 230 | 157 | 130 | 23 | 34 | 29 | **603** |

- **Julia R = 86 crops total** (23+34+29). Census the whole thing. This closes the julia
  low-end question outright — where the q3 mass already is — for under 90 crops.
- **Native R = 517 crops.** Sample ~100/family (300 total) stratified within R; a full
  census is not worth it given native's low M/G/H rate. n≈100/family lands q3-rate CIs
  around ±0.08–0.10, enough to tell "≈0" from "≈0.15."
- **Do not** do n=15/family: 0/15 tops out near an 18% CP upper bound and answers nothing.

**Price (eye-hours, stated assumption).** Stage 1 was 83 crops of rapid 1/2/3 triage.
At a plausible **200–400 crops/eye-hour** for this judgment:
- Julia R census (86): **~15–25 min**.
- Native R sample (300): **~45–90 min**.
- Both: **~1–2 eye-hours.** The julia census alone is the high-value half and is ~20
  minutes — do it regardless.

If only one thing gets done: the **julia R census**, because that is where the confirmed
q3 mass is trending and it's cheap and complete.

---

## 5. What these rates are, and are not

**Every rate in this document is `P(q3 | M/G/H stratum, surfaced by the prospect
pipeline, p_good ≥ 0.15)`. None of them is `P(q3 | family)`.** The population is 686
rows per the six families; 603 (88%) sit in the unlabeled R band. R's contribution to
any family rate is **unknown, not zero.**

Concretely, the family-rate *bound* if R were entirely q3-free (a floor, not an
estimate — R could be higher):

| family | q3 in M/G/H | family N | floor P(q3\|family) | M/G/H coverage |
|---|---:|---:|---:|---:|
| multibrot3 | 1 | 239 | ≥ 0.004 | 3.8% |
| multibrot4 | 4 | 166 | ≥ 0.024 | 5.4% |
| multibrot5 | 1 | 134 | ≥ 0.007 | 3.0% |
| julia:mb3 | 13 | 54 | ≥ 0.24 | 57% |
| julia:mb4 | 11 | 54 | ≥ 0.20 | 37% |
| julia:mb5 | 8 | 39 | ≥ 0.21 | 26% |

For **native** multibrot we have seen 3–5% of the population; the M/G/H q3 rate (0.11 /
0.44 / 0.25) says almost nothing about the family until R is labeled. Quoting "mb4 base
rate 44%" would be lifting a number out of a 5%-coverage stratum — the exact hazard this
corpus keeps repeating. For **julia** the floor bounds are informative (each family is
≥0.20 q3 *even if all of R is junk*), because M/G/H already covers a quarter to a half of
the population.

The single portable sentence: **within the p_good≥0.15 slice the pipeline surfaced,
julia q3s are common (≥20% of each family, floor) and concentrated in the low-p_good M
band; native q3s are rarer (single-digit floor) but the band is not empty; and v6's
p_good does not order quality inside this slice. R must be labeled before any threshold
moves.**
