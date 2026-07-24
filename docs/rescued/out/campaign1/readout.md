# Campaign 1 — steered frontier + dive: scheduling readout

_Regenerated from ledgers + state. Breadth `breadth` active **864.0 min** (14.40 h); dive `dive` active **106.5 min** (1.78 h)._

## 1. Admissions/hr over accumulated active time

- **Breadth overall: 21.8 adm/hr** (314 admitted / 14.40 active-h).
- **Dive overall: 143.1 adm/hr** (254 admitted / 1.78 active-h).

Breadth admissions per active-hour bin, rate-normalized by bin width (314/314 admissions time-stamped from stdout×harvest_log):

| active-hr | width (h) | admits | adm/hr |
|--:|--:|--:|--:|
| 0.0–1.0 | 1.00 | 33 | 33.0 |
| 1.0–2.0 | 1.00 | 24 | 24.0 |
| 2.0–3.0 | 1.00 | 19 | 19.0 |
| 3.0–4.0 | 1.00 | 18 | 18.0 |
| 4.0–5.0 | 1.00 | 17 | 17.0 |
| 5.0–6.0 | 1.00 | 33 | 33.0 |
| 6.0–7.0 | 1.00 | 26 | 26.0 |
| 7.0–8.0 | 1.00 | 18 | 18.0 |
| 8.0–9.0 | 1.00 | 22 | 22.0 |
| 9.0–10.0 | 1.00 | 25 | 25.0 |
| 10.0–11.0 | 1.00 | 17 | 17.0 |
| 11.0–12.0 | 1.00 | 16 | 16.0 |
| 12.0–13.0 | 1.00 | 23 | 23.0 |
| 13.0–14.0 | 1.00 | 14 | 14.0 |
| 14.0–14.4 | 0.40 ⚠partial | 9 | 22.5 |

_Verdict: mean **21.8 adm/hr** over 14 full-hour bins, trend **-0.64/hr per hr**, 1/14 bins below 16/hr → floor **HOLDS**. (The naive last-bin count reads as a collapse only because that bin is 0.40h wide; its true rate is 22.5/hr.)_

## 2. Per-family admissions & cost (breadth vs dive)

- Breadth cost/admission: **2.75 active-min** (864 min / 314).
- Dive cost/admission: **0.42 active-min** (107 min / 254).

| partition | breadth adm | dive adm |
|---|--:|--:|
| mandelbrot | 69 | 44 |
| multibrot3 | 89 | 74 |
| multibrot4 | 25 | 20 |
| multibrot5 | 85 | 64 |
| julia:mandelbrot | 2 | 2 |
| julia:multibrot3 | 23 | 27 |
| julia:multibrot4 | 12 | 14 |
| julia:multibrot5 | 9 | 9 |

Breadth compute-fate per family — where each canonical confirmation render goes (checks/admit = renders spent per admission):

| partition | checks | admit | checks/admit | canon_not_q3 | q3_dup | reframe_fail |
|---|--:|--:|--:|--:|--:|--:|
| mandelbrot | 2744 | 69 | 39.8 | 1944 (71%) | 731 (27%) | 0 (0%) |
| multibrot3 | 3887 | 89 | 43.7 | 2813 (72%) | 983 (25%) | 2 (0%) |
| multibrot4 | 141 | 25 | 5.6 | 6 (4%) | 110 (78%) | 0 (0%) |
| multibrot5 | 3126 | 85 | 36.8 | 2265 (72%) | 775 (25%) | 1 (0%) |
| julia:mandelbrot | 5435 | 2 | 2717.5 | 693 (13%) | 4740 (87%) | 0 (0%) |
| julia:multibrot3 | 4116 | 23 | 179.0 | 565 (14%) | 3520 (86%) | 8 (0%) |
| julia:multibrot4 | 4050 | 12 | 337.5 | 81 (2%) | 3954 (98%) | 3 (0%) |
| julia:multibrot5 | 8389 | 9 | 932.1 | 93 (1%) | 8285 (99%) | 2 (0%) |

_Two distinct cost profiles: low-degree c-plane (mandelbrot/mb3/mb5) is cheap-scorer over-admission (canon_not_q3 ~70%, cheap tau_h passes frames the canonical render decodes below q3); julia + multibrot4 is hot-region dup-churn (q3_dup 78–99%), not decode/guard failure. The dup-churn compute is a canonical render spent on a candidate the coord-dup check then rejects — a pre-canonical coord-dup filter would reclaim it._

## 3. Distinct morph-look count over time

- **508 distinct morph looks** among 568 admissions (within-family single-linkage, CLIP≥0.974 on the library morph_gray recipe) = 89% distinct.

| partition | distinct looks | admitted |
|---|--:|--:|
| mandelbrot | 99 | 113 |
| multibrot3 | 158 | 163 |
| multibrot4 | 41 | 45 |
| multibrot5 | 144 | 149 |
| julia:mandelbrot | 4 | 4 |
| julia:multibrot3 | 28 | 50 |
| julia:multibrot4 | 19 | 26 |
| julia:multibrot5 | 15 | 18 |

Cumulative distinct looks vs accumulated active time (breadth-timed admissions):

| active-hr | cum admitted | cum distinct looks |
|--:|--:|--:|
| ≤1 | 33 | 33 |
| ≤2 | 57 | 56 |
| ≤3 | 76 | 75 |
| ≤4 | 94 | 92 |
| ≤5 | 111 | 109 |
| ≤6 | 144 | 140 |
| ≤7 | 170 | 166 |
| ≤8 | 188 | 184 |
| ≤9 | 210 | 205 |
| ≤10 | 235 | 229 |
| ≤11 | 252 | 246 |
| ≤12 | 268 | 261 |
| ≤13 | 291 | 284 |
| ≤14 | 305 | 298 |
| ≤15 | 314 | 307 |
## 4. Library overlap vs prior admissions

_Prior corpus: 14 ledgers, 493 distinct-q3 places (coord); library embedding store for morph._

- **Coord-dup overlap: 78/568 = 13.7%** of campaign admissions fall inside a prior admission's coord-dup radius (same partition, DEDUP_K=1.5).
  - per partition: julia:mandelbrot 3/4, julia:multibrot3 14/50, julia:multibrot4 9/26, julia:multibrot5 5/18, mandelbrot 21/113, multibrot3 7/163, multibrot4 7/45, multibrot5 12/149
- **Morph near-dup overlap: 9/568 = 1.6%** are CLIP≥0.974 near-dups of a library admission (library morph_gray recipe).

_Verdict: coord overlap 13.7% → **WORTH building a cross-run freshness prior**._

## 5. Family coverage (zero-admission flags)

| partition | admissions | flag |
|---|--:|---|
| mandelbrot | 113 |  |
| multibrot3 | 163 |  |
| multibrot4 | 45 |  |
| multibrot5 | 149 |  |
| julia:mandelbrot | 4 |  |
| julia:multibrot3 | 50 |  |
| julia:multibrot4 | 26 |  |
| julia:multibrot5 | 18 |  |
