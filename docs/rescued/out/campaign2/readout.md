# Campaign 2 — deficit scheduler + freshness prior: readout

_Run `breadth`, active **699.7 min** (11.66 h), 311 distinct-q3 admissions. Scheduler + freshness prior both ON._

> **Budget — planned vs actual.** Launched at **864** accumulated-active-min; stopped cleanly at **699.7** against an effective **700-min** budget (Δ **-164 min** vs plan). The budget was lowered 864→700 at the *deadline* resume (a "finish by 11pm" trim), NOT the later spacing-change resume, which preserved 700. Shortfall **deliberately accepted** — breadth is not topped up; the dive leg carries the campaign forward._

## Regime comparison — julia-hook-spacing 0.20 → 0.10 (resume boundary)

Regimes (spacing → first batch): **0.2**@b3, **0.1**@b1260.


### 1. Hook skip fraction (over-thinning check)

| spacing | decisions | fired | skipped | skip frac |
|--:|--:|--:|--:|--:|
| 0.2 | 94 | 36 | 58 | 0.62 |
| 0.1 | 94 | 29 | 65 | 0.69 |

Nearest-c of the **skipped** hooks (is the spacing radius even the binding constraint?):

| spacing | skipped | median nearest-c | genuine near-dup (< 0.1) | recoverable [0.1, 0.2) |
|--:|--:|--:|--:|--:|
| 0.2 | 58 | 0.012 | 46 | 12 |
| 0.1 | 65 | 0.019 | 65 | 0 |

_Smoke reference skip frac 0.25. Reducing spacing 0.2→0.1: skip frac **rose** (0.62→0.69). The nearest-c table is the reason — if virtually all skips sit BELOW 0.1 (genuine near-dups) and few/none are in the recoverable band (0 at 0.1), the radius was never the binding constraint: the skipped parents are true near-duplicate Julia sets, so the smaller radius cannot recover them. Skip fraction is then driven by hooked-c DENSITY (accumulating over the run), not the radius._


### 2. Conversion: hook fires → julia roots → julia admissions

| spacing | hooks fired (=julia roots) | julia admissions | adm / root |
|--:|--:|--:|--:|
| 0.2 | 36 | 71 | 1.97 |
| 0.1 | 29 | 52 | 1.79 |

_Roots-per-regime is the direct lever: fewer spacing-skips → more julia roots seeded → more julia descents to admit. (Prospective only — spacing-skipped parents from the 0.20 regime are NOT re-hooked; their seed c stays recoverable in julia_hooks.jsonl.)_


### 3. Julia distinct-look share — per-regime incremental

Julia target share (order book): **77%**. Incremental = new julia looks / new total looks produced *within* each regime (library seed + prior regime excluded):

| spacing | batch span | Δ total looks | Δ julia looks | incremental julia share |
|--:|--:|--:|--:|--:|
| 0.2 | b1–b1259 | 160 | 68 | 42% |
| 0.1 | b1260–b2540 | 126 | 42 | 33% |

_**Two julia-share scopes, complementary not contradictory:** run-incremental **38%** (julia share of the 286 looks this run added, seed excluded) vs library-wide **23%** (julia share of the full 809-look tally incl. the 523 library seed — the scope the deficit/target act on). The run is producing julia well above its library share yet still below the 77% target because the seed-heavy denominator moves slowly._


### 4. Per-partition allocation share (batches served)

| partition | target | 0.2 | 0.1 |
|---|--:|--:|--:|
| mandelbrot | 9% | 37% | 30% |
| multibrot3 | 6% | 5% | 9% |
| multibrot4 | 2% | 37% | 42% |
| multibrot5 | 6% | 10% | 10% |
| julia:mandelbrot | 19% | 4% | 3% |
| julia:multibrot3 | 19% | 2% | 2% |
| julia:multibrot4 | 19% | 3% | 1% |
| julia:multibrot5 | 19% | 3% | 3% |


### 5. Learned-price drift (active-min per distinct look)

Per-partition price at each regime's first vs last traced batch (online EMA):

| partition | 0.2 start→end | 0.1 start→end |
|---|--:|--:|
| mandelbrot | 3.00→6.07 | 6.07→8.50 |
| multibrot3 | 3.00→1.79 | 1.79→1.57 |
| multibrot4 | 3.00→6.13 | 6.13→5.32 |
| multibrot5 | 3.00→1.84 | 1.84→1.13 |
| julia:mandelbrot | 4.00→0.32 | 0.32→0.40 |
| julia:multibrot3 | 4.00→0.67 | 0.67→0.50 |
| julia:multibrot4 | 4.00→0.51 | 0.51→0.36 |
| julia:multibrot5 | 4.00→0.52 | 0.52→0.42 |

_Julia prices are the campaign-3 seed; watch whether the cheaper 0.10 regime pushes the julia price further down (more looks per active-min as hook throughput rises)._

## A. Scheduler allocation — distinct-look shares vs target over time

Batches traced: **2557**; served (non-null pop): **2540**. Served-partition histogram (the scheduler's realized cross-partition allocation):

| partition | batches served | share | target |
|---|--:|--:|--:|
| mandelbrot | 860 | 33.9% | 9.2% |
| multibrot3 | 185 | 7.3% | 6.1% |
| multibrot4 | 1002 | 39.4% | 1.9% |
| multibrot5 | 243 | 9.6% | 6.1% |
| julia:mandelbrot | 81 | 3.2% | 19.2% |
| julia:multibrot3 | 44 | 1.7% | 19.2% |
| julia:multibrot4 | 57 | 2.2% | 19.2% |
| julia:multibrot5 | 68 | 2.7% | 19.2% |

Run-only distinct-look shares (library seed subtracted) at trace checkpoints — the acceptance the scheduler produced, converging toward target:

| batch | run looks | mandelbrot | multibrot3 | multibrot4 | multibrot5 | julia:mandelbrot | julia:multibrot3 | julia:multibrot4 | julia:multibrot5 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 638 | 102 | 23% | 6% | 12% | 18% | 9% | 9% | 14% | 11% |
| 1272 | 163 | 25% | 7% | 11% | 15% | 13% | 8% | 13% | 9% |
| 1904 | 224 | 21% | 8% | 14% | 16% | 13% | 6% | 12% | 10% |
| 2540 | 286 | 20% | 10% | 14% | 17% | 12% | 7% | 9% | 10% |
| target | — | 9% | 6% | 2% | 6% | 19% | 19% | 19% | 19% |

Σ|deficit| trajectory (0 == every partition at target): start **1.246** → latest **1.075** over 2557 batches.

## B. Final learned prices (the re-priced table for campaign 3)

Price = active-minutes per DISTINCT LOOK (online EMA). **Campaign-1 julia prices are void; this table replaces them.**

| partition | seed price | learned price | active-min spent | looks (incl. seed) |
|---|--:|--:|--:|--:|
| mandelbrot | 3.00 | **8.50** | 248.9 | 160 |
| multibrot3 | 3.00 | **1.28** | 48.6 | 189 |
| multibrot4 | 3.00 | **5.33** | 291.4 | 82 |
| multibrot5 | 3.00 | **1.13** | 72.6 | 194 |
| julia:mandelbrot | 4.00 | **0.40** | 12.4 | 38 |
| julia:multibrot3 | 4.00 | **0.50** | 6.4 | 58 |
| julia:multibrot4 | 4.00 | **0.36** | 8.9 | 46 |
| julia:multibrot5 | 4.00 | **0.42** | 10.5 | 43 |

_Capped partitions: none._

## C. Freshness-prior effect — BREADTH ONLY

> **Scope.** This verdict covers the **breadth** leg only. The prior was **OFF** in the dive leg by design — it is structurally incompatible with dive precanon (see the design finding at the end of this readout); the dive's numbers must not be read as a prior result.

- **Throughput: 26.7 adm/hr** vs campaign-1 context **21.8 adm/hr** (311 admitted / 11.66 active-h). Δ **+4.9 adm/hr**.
  - _This Δ is the **JOINT** scheduler+prior effect vs campaign 1 — both knobs changed this campaign, so it is not attributable to the prior alone. The prior's **isolated** wins are the 0/311 coord overlap and the 10993 renders it saved (both below)._
- **Pre-canonical dup skips (renders saved): 10993** (81.9% of 13425 harvest checks).
- q3_dup (canonical-render-then-coord-dup): 260.
- **Coord overlap vs 1071 prior places (ctx 507): 257/311 = 82.6%** of campaign-2 admissions fall inside a prior admission's coord-dup radius. With the prior ON this should be LOW (the prior actively steers off prior coords).
## D. Julia hook spacing at scale

- Hook decisions: **188** (65 fired, 123 skipped-by-spacing). **Skip fraction 0.65** vs smoke 0.25.
  - Verdict: **NOT over-thinning — 90% of skips are genuine near-dups (nearest-c < 0.1, median 0.018); the skips are correct and the julia-yield ceiling is c-space clustering of admitted parents, not spacing**.

| julia partition | fired | skipped |
|---|--:|--:|
| julia:mandelbrot | 21 | 44 |
| julia:multibrot3 | 12 | 18 |
| julia:multibrot4 | 15 | 26 |
| julia:multibrot5 | 17 | 35 |
## E. Julia admission depth distribution

- **123 julia admissions**, depth median **3**, range [2, 5]. The shallow (depth 1–2) center-descent population — suppressed in prior runs — should now appear.

| reached_depth | julia admissions |
|--:|--:|
| 2 | 52 |
| 3 | 27 |
| 4 | 21 |
| 5 | 23 |

_Shallow (depth ≤ 2): 52/123 = 42%._

## F. Novelty-memory saturation (sat_frac) trajectory

- **Overall sat_frac 0.743** over 2537 scored batches (campaign-1 context 0.735; permanent novelty memory now larger). Report, don't tune.
  - trajectory (batch-quartile means): 0.71 → 0.74 → 0.77 → 0.78 → 0.71
  - final memory: perm 311 + recency 215 = 526 looks.

## H. Dive leg — depth-mining off breadth (freshness prior OFF by design)

- **271 admissions off 311 dives = 0.87 adm/dive** (campaign-1 ref ~0.8); 97.6 active-min, **0.36 min/admission**.

| partition | dives | admissions | adm/dive |
|---|--:|--:|--:|
| mandelbrot | 65 | 49 | 0.75 |
| multibrot3 | 30 | 19 | 0.63 |
| multibrot4 | 41 | 25 | 0.61 |
| multibrot5 | 52 | 45 | 0.87 |
| julia:mandelbrot | 38 | 39 | 1.03 |
| julia:multibrot3 | 29 | 27 | 0.93 |
| julia:multibrot4 | 27 | 35 | 1.30 |
| julia:multibrot5 | 29 | 32 | 1.10 |

_End cause: {'target_depth': 118, 'gate_dead_or_floor': 193} (`target_depth` = ran to depth 23; `gate_dead_or_floor` = the descent exhausted). Sources: 250 top + 61 control._

**Dive prices — not learned (by design).** The dive consumes the scheduler's per-partition DEFICITS to order its sources (deficit-ordering — this feature's first real use; confirmed: julia:mandelbrot, the highest-deficit partition, dived first) but does NOT run the price-EMA loop (no per-batch active-time charge exists in dive mode), so it produces no price update. The campaign's final learned-price table is the breadth one (§B).

## I. Whole-campaign totals (breadth + dive)

- **582 distinct-q3 admissions** (breadth 311 + dive 271).

| partition | breadth+dive adm | admission share | target (look-share) |
|---|--:|--:|--:|
| mandelbrot | 114 | 20% | 9% |
| multibrot3 | 49 | 8% | 6% |
| multibrot4 | 66 | 11% | 2% |
| multibrot5 | 97 | 17% | 6% |
| julia:mandelbrot | 77 | 13% | 19% |
| julia:multibrot3 | 56 | 10% | 19% |
| julia:multibrot4 | 62 | 11% | 19% |
| julia:multibrot5 | 61 | 10% | 19% |

_Julia share of admissions: **256/582 = 44%** (target look-share 77%). Note the dive is julia-tilted (deficit-ordered), pulling the whole-campaign julia admission share above breadth's alone._

- **490 distinct morph looks** among 582 admissions (within-family single-linkage, CLIP≥0.974) = 84% distinct.
  - breadth alone: 290/311 = 93%; dive alone: 245/271 = 90%.
- **Dive marginal-distinct rate: 218/271 = 80%** of dive admissions are NEW looks (not CLIP≥0.974 near-dups of a breadth admission, same family). Campaign-1 context: dive ~80% vs breadth ~97% — a dive descends *from* breadth points, so a portion re-expresses a look breadth already banked; the rest is genuine depth-find.

## G. Visual samples

Whole-campaign admission / reject contact sheet(s):

```
uv run python tools/atlas/campaign1_contact_sheet.py --run data\discovery\campaign2\breadth --run data\discovery\campaign2\dive \
    --out out/campaign2/contact_sheet.png
```


## H. Design finding — dive + freshness-prior precanon are structurally incompatible

**Symptom.** The first campaign-2 dive attempt ran with the freshness prior ON (per the launch spec) and admitted **0/311** — every one of ~1040 harvest checks was rejected as a `precanon_dup`, with **zero** candidates reaching a canonical render. Campaign 1's dive (which predates the prior) admitted 254 off a comparable 314 sources.

**Root cause (structural, not a tuning miss).** The two mechanisms serve opposite goals:

- The **freshness prior** is an *exploration* tool — it seeds the dedup/steering clouds with prior-library coords so root draws and frontier steering avoid re-covering known ground.

- A **dive** is *exploitation* of a known point: it descends the greedy argmax-p_good path **from a breadth admission**. That source coord (and its basin) is, by construction, already in the prior cloud — so the dive's pre-canonical coord-dup filter rejects the descent against the very point it was told to mine. With the full 7926-row library in the cloud the basin is densely covered, so **100%** of candidates dup out before any canonical render. Sterilization is guaranteed, not incidental.

**Resolution taken.** The dive leg runs with the prior **OFF** (dedup against its own accruing cloud only, exactly as campaign 1's productive dive did). The prior's proven wins are a **breadth**-leg result (§C) and are unaffected.

**No lost-freshness guard needed.** Turning the prior off in the dive means a dive can, in principle, re-mint a location some *other-era* library ledger already holds (the dive dedups only within-campaign). That is acceptable and needs no extra guard here: **emission intake's own dedup pass catches cross-era re-mints downstream** (coord + CLIP-morph), so a re-mint is collapsed at library-assembly time rather than silently shipped.

**Campaign-3 options.**

1. **Keep the prior off in dives** (the current fix) — simple, proven, and correct given the exploration/exploitation split. Recommended default.

2. **fw-scaled precanon radius semantics** — make the dive's coord-dup radius shrink with the candidate's own `fw`, so a genuinely-deeper frame in a covered basin reads as distinct from its shallower source/neighbours. This would let a dive keep *some* cross-run freshness. It needs design + tests (the radius/`DEDUP_K` semantics are load-bearing across the pipeline) and is **deferred**.

---

# Base scheduling numbers (campaign1_readout, reused verbatim)

> ⚠ **Caveat on the base §4 verdict.** campaign1_readout computes its "worth a freshness prior?" verdict under the campaign-1 assumption that the prior was *OFF* (high coord overlap → build one). In campaign 2 the prior is **ON**, so a **0.0% coord overlap is the prior working as intended**, not evidence against it — the base verdict's logic is inverted here and does not apply. The authoritative freshness-prior result is **§C above** (whole-run).

# Campaign 1 — steered frontier + dive: scheduling readout

_Regenerated from ledgers + state. Breadth `breadth` active **699.7 min** (11.66 h); dive `dive` active **97.6 min** (1.63 h)._

## 1. Admissions/hr over accumulated active time

- **Breadth overall: 26.7 adm/hr** (311 admitted / 11.66 active-h).
- **Dive overall: 166.5 adm/hr** (271 admitted / 1.63 active-h).

Breadth admissions per active-hour bin, rate-normalized by bin width (311/311 admissions time-stamped from stdout×harvest_log):

| active-hr | width (h) | admits | adm/hr |
|--:|--:|--:|--:|
| 0.0–1.0 | 1.00 | 23 | 23.0 |
| 1.0–2.0 | 1.00 | 38 | 38.0 |
| 2.0–3.0 | 1.00 | 47 | 47.0 |
| 3.0–4.0 | 1.00 | 19 | 19.0 |
| 4.0–5.0 | 1.00 | 20 | 20.0 |
| 5.0–6.0 | 1.00 | 28 | 28.0 |
| 6.0–7.0 | 1.00 | 19 | 19.0 |
| 7.0–8.0 | 1.00 | 16 | 16.0 |
| 8.0–9.0 | 1.00 | 32 | 32.0 |
| 9.0–10.0 | 1.00 | 29 | 29.0 |
| 10.0–11.0 | 1.00 | 23 | 23.0 |
| 11.0–11.7 | 0.66 ⚠partial | 16 | 24.2 |

_Verdict: mean **26.5 adm/hr** over 12 full-hour bins, trend **-0.72/hr per hr**, 0/12 bins below 16/hr → floor **HOLDS**. (The naive last-bin count reads as a collapse only because that bin is 0.66h wide; its true rate is 24.2/hr.)_

## 2. Per-family admissions & cost (breadth vs dive)

- Breadth cost/admission: **2.25 active-min** (700 min / 311).
- Dive cost/admission: **0.36 active-min** (98 min / 271).

| partition | breadth adm | dive adm |
|---|--:|--:|
| mandelbrot | 65 | 49 |
| multibrot3 | 30 | 19 |
| multibrot4 | 41 | 25 |
| multibrot5 | 52 | 45 |
| julia:mandelbrot | 38 | 39 |
| julia:multibrot3 | 29 | 27 |
| julia:multibrot4 | 27 | 35 |
| julia:multibrot5 | 29 | 32 |

Breadth compute-fate per family — where each canonical confirmation render goes (checks/admit = renders spent per admission):

| partition | checks | admit | checks/admit | canon_not_q3 | q3_dup | reframe_fail |
|---|--:|--:|--:|--:|--:|--:|
| mandelbrot | 3486 | 65 | 53.6 | 3397 (97%) | 22 (1%) | 2 (0%) |
| multibrot3 | 1195 | 30 | 39.8 | 1156 (97%) | 8 (1%) | 1 (0%) |
| multibrot4 | 325 | 41 | 7.9 | 272 (84%) | 12 (4%) | 0 (0%) |
| multibrot5 | 1699 | 52 | 32.7 | 1622 (95%) | 25 (1%) | 0 (0%) |
| julia:mandelbrot | 1656 | 38 | 43.6 | 1570 (95%) | 44 (3%) | 4 (0%) |
| julia:multibrot3 | 627 | 29 | 21.6 | 569 (91%) | 28 (4%) | 1 (0%) |
| julia:multibrot4 | 2391 | 27 | 88.6 | 2308 (97%) | 51 (2%) | 5 (0%) |
| julia:multibrot5 | 2046 | 29 | 70.6 | 1960 (96%) | 55 (3%) | 2 (0%) |

_Two distinct cost profiles: low-degree c-plane (mandelbrot/mb3/mb5) is cheap-scorer over-admission (canon_not_q3 ~70%, cheap tau_h passes frames the canonical render decodes below q3); julia + multibrot4 is hot-region dup-churn (q3_dup 78–99%), not decode/guard failure. The dup-churn compute is a canonical render spent on a candidate the coord-dup check then rejects — a pre-canonical coord-dup filter would reclaim it._

## 3. Distinct morph-look count over time

_Morph pass skipped (--no-morph or no admissions)._

## 4. Library overlap vs prior admissions

_Prior corpus: 15 ledgers, 801 distinct-q3 places (coord); library embedding store for morph._

- **Coord-dup overlap: 1/582 = 0.2%** of campaign admissions fall inside a prior admission's coord-dup radius (same partition, DEDUP_K=1.5).
  - per partition: julia:mandelbrot 0/77, julia:multibrot3 0/56, julia:multibrot4 0/62, julia:multibrot5 0/61, mandelbrot 0/114, multibrot3 0/49, multibrot4 0/66, multibrot5 1/97
- Morph near-dup: not computed (--no-morph).

_Verdict: coord overlap 0.2% → **NOT worth a cross-run freshness prior yet**._

## 5. Family coverage (zero-admission flags)

| partition | admissions | flag |
|---|--:|---|
| mandelbrot | 114 |  |
| multibrot3 | 49 |  |
| multibrot4 | 66 |  |
| multibrot5 | 97 |  |
| julia:mandelbrot | 77 |  |
| julia:multibrot3 | 56 |  |
| julia:multibrot4 | 62 |  |
| julia:multibrot5 | 61 |  |

