# Campaign 2 — deficit scheduler + freshness prior: readout

_Run `breadth`, active **353.2 min** (5.89 h), 167 distinct-q3 admissions. Scheduler + freshness prior both ON._

## Regime comparison — julia-hook-spacing 0.20 → 0.10 (resume boundary)

Regimes (spacing → first batch): **0.2**@b3, **0.1**@b1260.


### 1. Hook skip fraction (over-thinning check)

| spacing | decisions | fired | skipped | skip frac |
|--:|--:|--:|--:|--:|
| 0.2 | 94 | 36 | 58 | 0.62 |
| 0.1 | 1 | 1 | 0 | 0.00 |

_Smoke reference skip frac 0.25. A large drop from the 0.20 to the 0.10 regime is the flag change taking effect (more distinct-c parents now clear spacing)._


### 2. Conversion: hook fires → julia roots → julia admissions

| spacing | hooks fired (=julia roots) | julia admissions | adm / root |
|--:|--:|--:|--:|
| 0.2 | 36 | 71 | 1.97 |
| 0.1 | 1 | 1 | 1.00 |

_Roots-per-regime is the direct lever: fewer spacing-skips → more julia roots seeded → more julia descents to admit. (Prospective only — spacing-skipped parents from the 0.20 regime are NOT re-hooked; their seed c stays recoverable in julia_hooks.jsonl.)_


### 3. Julia distinct-look share — per-regime incremental

Julia target share (order book): **77%**. Incremental = new julia looks / new total looks produced *within* each regime (library seed + prior regime excluded):

| spacing | batch span | Δ total looks | Δ julia looks | incremental julia share |
|--:|--:|--:|--:|--:|
| 0.2 | b1–b1259 | 160 | 68 | 42% |
| 0.1 | b1260–b1265 | 2 | 1 | 50% |


### 4. Per-partition allocation share (batches served)

| partition | target | 0.2 | 0.1 |
|---|--:|--:|--:|
| mandelbrot | 9% | 37% | 0% |
| multibrot3 | 6% | 5% | 0% |
| multibrot4 | 2% | 37% | 0% |
| multibrot5 | 6% | 10% | 33% |
| julia:mandelbrot | 19% | 4% | 0% |
| julia:multibrot3 | 19% | 2% | 0% |
| julia:multibrot4 | 19% | 3% | 0% |
| julia:multibrot5 | 19% | 3% | 67% |


### 5. Learned-price drift (active-min per distinct look)

Per-partition price at each regime's first vs last traced batch (online EMA):

| partition | 0.2 start→end | 0.1 start→end |
|---|--:|--:|
| mandelbrot | 3.00→6.07 | 6.07→6.07 |
| multibrot3 | 3.00→1.79 | 1.79→1.79 |
| multibrot4 | 3.00→6.13 | 6.13→6.13 |
| multibrot5 | 3.00→1.84 | 1.84→3.05 |
| julia:mandelbrot | 4.00→0.32 | 0.32→0.32 |
| julia:multibrot3 | 4.00→0.67 | 0.67→0.67 |
| julia:multibrot4 | 4.00→0.51 | 0.51→0.51 |
| julia:multibrot5 | 4.00→0.52 | 0.52→0.54 |

_Julia prices are the campaign-3 seed; watch whether the cheaper 0.10 regime pushes the julia price further down (more looks per active-min as hook throughput rises)._

## A. Scheduler allocation — distinct-look shares vs target over time

Batches traced: **1272**; served (non-null pop): **1265**. Served-partition histogram (the scheduler's realized cross-partition allocation):

| partition | batches served | share | target |
|---|--:|--:|--:|
| mandelbrot | 471 | 37.2% | 9.2% |
| multibrot3 | 65 | 5.1% | 6.1% |
| multibrot4 | 464 | 36.7% | 1.9% |
| multibrot5 | 123 | 9.7% | 6.1% |
| julia:mandelbrot | 45 | 3.6% | 19.2% |
| julia:multibrot3 | 20 | 1.6% | 19.2% |
| julia:multibrot4 | 41 | 3.2% | 19.2% |
| julia:multibrot5 | 36 | 2.8% | 19.2% |

Run-only distinct-look shares (library seed subtracted) at trace checkpoints — the acceptance the scheduler produced, converging toward target:

| batch | run looks | mandelbrot | multibrot3 | multibrot4 | multibrot5 | julia:mandelbrot | julia:multibrot3 | julia:multibrot4 | julia:multibrot5 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 319 | 43 | 42% | 5% | 9% | 7% | 19% | 2% | 9% | 7% |
| 635 | 101 | 23% | 5% | 12% | 18% | 9% | 9% | 14% | 11% |
| 949 | 133 | 26% | 5% | 11% | 17% | 11% | 10% | 11% | 10% |
| 1265 | 162 | 25% | 7% | 11% | 14% | 13% | 8% | 13% | 9% |
| target | — | 9% | 6% | 2% | 6% | 19% | 19% | 19% | 19% |

Σ|deficit| trajectory (0 == every partition at target): start **1.246** → latest **1.112** over 1272 batches.

## B. Final learned prices (the re-priced table for campaign 3)

Price = active-minutes per DISTINCT LOOK (online EMA). **Campaign-1 julia prices are void; this table replaces them.**

| partition | seed price | learned price | active-min spent | looks (incl. seed) |
|---|--:|--:|--:|--:|
| mandelbrot | 3.00 | **6.07** | 132.4 | 143 |
| multibrot3 | 3.00 | **1.79** | 18.9 | 170 |
| multibrot4 | 3.00 | **6.12** | 129.8 | 60 |
| multibrot5 | 3.00 | **1.84** | 36.3 | 167 |
| julia:mandelbrot | 4.00 | **0.32** | 7.3 | 25 |
| julia:multibrot3 | 4.00 | **0.67** | 3.1 | 50 |
| julia:multibrot4 | 4.00 | **0.51** | 6.6 | 40 |
| julia:multibrot5 | 4.00 | **0.52** | 4.8 | 28 |

_Capped partitions: none._

## C. Freshness-prior effect

- **Throughput: 28.4 adm/hr** vs campaign-1 context **21.8 adm/hr** (167 admitted / 5.89 active-h). Δ **+6.6 adm/hr**.
- **Pre-canonical dup skips (renders saved): 6547** (82.2% of 7967 harvest checks).
- q3_dup (canonical-render-then-coord-dup): 145.
- **Coord overlap vs 801 prior places (ctx 507): 0/167 = 0.0%** of campaign-2 admissions fall inside a prior admission's coord-dup radius. With the prior ON this should be LOW (the prior actively steers off prior coords).
## D. Julia hook spacing at scale

- Hook decisions: **95** (37 fired, 58 skipped-by-spacing). **Skip fraction 0.61** vs smoke 0.25.
  - Verdict: **over-thinning (skip fraction well above the smoke's 0.25 — spacing may be too wide)**.

| julia partition | fired | skipped |
|---|--:|--:|
| julia:mandelbrot | 12 | 30 |
| julia:multibrot3 | 5 | 6 |
| julia:multibrot4 | 11 | 8 |
| julia:multibrot5 | 9 | 14 |
## E. Julia admission depth distribution

- **72 julia admissions**, depth median **3**, range [2, 5]. The shallow (depth 1–2) center-descent population — suppressed in prior runs — should now appear.

| reached_depth | julia admissions |
|--:|--:|
| 2 | 32 |
| 3 | 17 |
| 4 | 13 |
| 5 | 10 |

_Shallow (depth ≤ 2): 32/72 = 44%._

## F. Novelty-memory saturation (sat_frac) trajectory

- **Overall sat_frac 0.725** over 1263 scored batches (campaign-1 context 0.735; permanent novelty memory now larger). Report, don't tune.
  - trajectory (batch-quartile means): 0.70 → 0.73 → 0.74 → 0.74 → 0.85
  - final memory: perm 167 + recency 231 = 398 looks.

## G. Visual samples

Admission / reject contact sheets: `uv run python tools/atlas/campaign1_contact_sheet.py --run-dir data\discovery\campaign2\breadth` (reused; run-dir-agnostic).

---

# Base scheduling numbers (campaign1_readout, reused verbatim)

# Campaign 1 — steered frontier + dive: scheduling readout

_Regenerated from ledgers + state. Breadth `breadth` active **353.2 min** (5.89 h); _dive: not run yet._

## 1. Admissions/hr over accumulated active time

- **Breadth overall: 28.5 adm/hr** (168 admitted / 5.89 active-h).

_No per-batch active-time map (stdout log absent) — only the overall rate above._

## 2. Per-family admissions & cost (breadth vs dive)

- Breadth cost/admission: **2.10 active-min** (353 min / 168).

| partition | breadth adm | dive adm |
|---|--:|--:|
| mandelbrot | 42 | 0 |
| multibrot3 | 11 | 0 |
| multibrot4 | 19 | 0 |
| multibrot5 | 24 | 0 |
| julia:mandelbrot | 22 | 0 |
| julia:multibrot3 | 15 | 0 |
| julia:multibrot4 | 21 | 0 |
| julia:multibrot5 | 14 | 0 |

Breadth compute-fate per family — where each canonical confirmation render goes (checks/admit = renders spent per admission):

| partition | checks | admit | checks/admit | canon_not_q3 | q3_dup | reframe_fail |
|---|--:|--:|--:|--:|--:|--:|
| mandelbrot | 2268 | 42 | 54.0 | 2215 (98%) | 9 (0%) | 2 (0%) |
| multibrot3 | 579 | 11 | 52.6 | 567 (98%) | 1 (0%) | 0 (0%) |
| multibrot4 | 168 | 19 | 8.8 | 142 (85%) | 7 (4%) | 0 (0%) |
| multibrot5 | 834 | 23 | 36.3 | 801 (96%) | 10 (1%) | 0 (0%) |
| julia:mandelbrot | 1109 | 22 | 50.4 | 1055 (95%) | 28 (3%) | 4 (0%) |
| julia:multibrot3 | 159 | 15 | 10.6 | 129 (81%) | 15 (9%) | 0 (0%) |
| julia:multibrot4 | 1847 | 21 | 88.0 | 1783 (97%) | 39 (2%) | 4 (0%) |
| julia:multibrot5 | 1195 | 14 | 85.4 | 1149 (96%) | 31 (3%) | 1 (0%) |

_Two distinct cost profiles: low-degree c-plane (mandelbrot/mb3/mb5) is cheap-scorer over-admission (canon_not_q3 ~70%, cheap tau_h passes frames the canonical render decodes below q3); julia + multibrot4 is hot-region dup-churn (q3_dup 78–99%), not decode/guard failure. The dup-churn compute is a canonical render spent on a candidate the coord-dup check then rejects — a pre-canonical coord-dup filter would reclaim it._

## 3. Distinct morph-look count over time

_Morph pass skipped (--no-morph or no admissions)._

## 4. Library overlap vs prior admissions

_Prior corpus: 15 ledgers, 801 distinct-q3 places (coord); library embedding store for morph._

- **Coord-dup overlap: 0/168 = 0.0%** of campaign admissions fall inside a prior admission's coord-dup radius (same partition, DEDUP_K=1.5).
  - per partition: julia:mandelbrot 0/22, julia:multibrot3 0/15, julia:multibrot4 0/21, julia:multibrot5 0/14, mandelbrot 0/42, multibrot3 0/11, multibrot4 0/19, multibrot5 0/24
- Morph near-dup: not computed (--no-morph).

_Verdict: coord overlap 0.0% → **NOT worth a cross-run freshness prior yet**._

## 5. Family coverage (zero-admission flags)

| partition | admissions | flag |
|---|--:|---|
| mandelbrot | 42 |  |
| multibrot3 | 11 |  |
| multibrot4 | 19 |  |
| multibrot5 | 24 |  |
| julia:mandelbrot | 22 |  |
| julia:multibrot3 | 15 |  |
| julia:multibrot4 | 21 |  |
| julia:multibrot5 | 14 |  |

