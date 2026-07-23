# Native multibrot label analysis — `2026-07-22_native_multibrot_band_v1`

300 labels (100 each mb3/mb4/mb5), first human measurement of native multibrot.
Report-only: no threshold flips, no re-decodes. Proposed per-family t_good below;
**Matt approves before any adoption.**

## Gates (done, committed `a7ca949`)

- **Gate 0 — sidecar reconciliation.** Two files existed: `native_multibrot_band_v1.json`
  (batch-builder placeholder, verified **0 labels**) and
  `2026-07-22_native_multibrot_band_v1.json` (labeling export, **300 labels**, values
  {1,2,3}, no nulls). Canonical = the labeled `2026-07-22_*` file; placeholder verified
  empty and removed. Registry line points at the labeled filename (not the placeholder
  the manifest had pre-drafted).
- **Gate 1 — registration + reachability.** `SIDECAR_LABELS["2026-07-22_native_multibrot_band_v1"]
  = "2026-07-22_native_multibrot_band_v1.json"`. All 300 resolve non-null via
  `label_store.resolve_score`; clean 300/300 coord+image_id join; `assert_sidecars_joined`
  green corpus-wide (7892 labeled crops total). Sidecars apply per-batch (keyed on batch
  dir), so image_id collisions across batches cannot leak.

## Label distribution (bad / okay / good = score 1/2/3)

| family | n | bad | okay | good | good-rate |
|---|---|---|---|---|---|
| mb3 | 100 | 61 | 25 | 14 | 14% |
| mb4 | 100 | 55 | 33 | 12 | 12% |
| mb5 | 100 | 68 | 20 | 12 | 12% |

**Native multibrot is a weak vein.** Even the top v7 band (p_good 0.65–1.00) tops out at
20–23% good in every family; bad dominates all bands. This is not a base rate — see regime
note below.

## 1. Human bad/okay/good rate by v7 p_good band

good-rate (%) per band [0–.20, .20–.35, .35–.50, .50–.65, .65–1.0]:

| family | .00–.20 | .20–.35 | .35–.50 | .50–.65 | .65–1.0 | monotone↑ |
|---|---|---|---|---|---|---|
| mb3 | 5 | 5 | **25** | 15 | 20 | **no** |
| mb4 | 0 | 9.5 | 10.5 | 16.7 | 22.7 | yes |
| mb5 | 0 | 10 | 15 | 15 | 20 | yes |

mb3 is **non-monotone** — its .35–.50 band (25% good) beats both higher bands. mb4/mb5 rise
monotonically but shallowly (good-rate never exceeds ~23%).

## 2. v7 calibration (positive = label==3 "good")

| family | n | good | AUC(p_good) | Spearman(p_good,label) |
|---|---|---|---|---|
| mb3 | 100 | 14 | 0.632 | +0.449 |
| mb4 | 100 | 12 | 0.733 | +0.457 |
| mb5 | 100 | 12 | 0.706 | +0.260 |

v7 has **weak but real** ranking power on native multibrot. mb4 best-calibrated (AUC 0.73,
monotone); mb3 weakest (AUC 0.63, non-monotone); mb5 mid AUC but low Spearman (good-rate
barely climbs).

## 3. mb4 provenance split — safe to pool

mb4's below-threshold bands are gather_v6→v7 rescored fills (label-batch-only); the high band
is campaign_v7. **Not a distinct-population artifact:**

| source | n | good-rate | mean p_good | within-source AUC |
|---|---|---|---|---|
| campaign_v7 | 31 | 19.4% | 0.710 | 0.640 |
| gather_v6_rescored_v7 | 69 | 8.7% | 0.306 | **0.759** |

The mean-p_good gap is the **band-stratification design** (campaign supplied the high band,
gather the low bands), not a calibration shift. In the overlapping bands the within-band
good-rates cross rather than diverge systematically (.50–.65: campaign 0/5 vs gather 3/13;
.65–1.0: campaign 5/21 vs gather 0/1), and v7 ranks the rescored fills at least as well
(AUC 0.76 ≥ 0.64). Pooling the two for calibration is justified.

## 4. Proposed per-family t_good

Convention (from `tools/v7/derive_t_good.py`): positive = label==3, `corn_decode==3`
(p_notbad≥0.5 ∧ p_good≥t), t = argmax F_β over [0.02,0.98], tie→higher t. The v7 discovery
table shipped F2 (recall) then moved mandelbrot to F0.5 (precision) once F2 was shown to
admit human-rejected junk. **F2 is degenerate here too** — it lands at t≈0.31–0.41, admitting
48–70% of the pool at 17–21% precision (the same failure). So all three lean precision.

Supply drives the exact β per the stated rule (precision where abundant, recall where scarce).
Real *emittable* ledger pool ≠ label-batch pool: mb4's 729 is mostly label-only gather fill —
its emittable campaign ledger is only **136 rows**.

| family | emittable ledger | supply | β | proposed t | P | R | why |
|---|---|---|---|---|---|---|---|
| **mb3** | 767 | abundant | F0.5 | **0.83** | 0.50 | 0.14 | abundant → precision |
| **mb4** | 136 | thin | F1 | **0.79** | 0.33 | 0.42 | thin emittable vein → don't over-restrict; F0.5→0.88 is the precision-max alt |
| **mb5** | 925 | abundant | F0.5 | **0.80** | 0.67 | 0.17 | abundant → precision |

Numbers land at ~0.80 across the board — reassuring, but the F0.5 optima rest on only 3–4
above-cut label samples (P from tp=2, fp=1–2), so treat ±0.05 as noise. mb3's t is the most
fragile (non-monotone, AUC 0.63): raising its bar mainly trims volume rather than cleanly
concentrating good — its ceiling is a **generation-quality** problem, not a threshold one.

### Admission deltas (report-only, over the real campaign ledger pool; mb4 gather fills excluded)

| family | pool | admit@0.50 | admit@prop | Δ |
|---|---|---|---|---|
| mb3 | 767 | 233 | 21 (@0.83) | −212 (−91%) |
| mb4 | 136 | 129 | 92 (@0.79) | −37 (−29%);  @0.88 → 30 (−99) |
| mb5 | 925 | 289 | 31 (@0.80) | −258 (−89%) |

The abundant families (mb3/mb5) see ~90% admission cuts — a large precision gain paid for by
heavy recall loss on an already-thin good vein. mb4's ledger is 95% above 0.5 today (campaigns
are admissions-heavy, p_good inflated vs human: mean 0.71 but only 19% good); F1@0.79 trims a
moderate 29%.

## Corpus regime (pin for any future retrain)

This batch is **stratified / selection-biased** — 20/band × 5 bands deliberately oversamples
the p_good tails (3 of 5 bands are sub-0.5, incl. canon-rejects and gather fills). The observed
12–14% good-rate is **not** a population base rate. Use these labels **train-side only** for any
future CORN retrain; **never** as an unbiased base-rate source. (Registered as such in
`SIDECAR_LABELS` with a train-side comment, mirroring the phoenix-grid precedent.)
