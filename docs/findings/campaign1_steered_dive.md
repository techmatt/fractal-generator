# Campaign 1 — production-scale steered frontier + dive (scheduling readout)

_2026-07-19. First production-scale discovery campaign. Deliverable = a grown v7-decoded
ledger + a readout that prices future scheduling. No intake/emission — the ledger is the
product. Regenerate the numbers with `tools/atlas/campaign1_readout.py`; artifacts in
`data/discovery/campaign1/{breadth,dive}/` and `out/campaign1/`._

Config = the v1.2 recency production defaults, **no retuning**: `--julia-hook --mem-recency`
(recency_k=8), λ_m=0.5, β=0.02, B=32, the 4 c-plane families + julia twins, Phoenix excluded.
Budget = accumulated **active** runtime (survives every stop/resume): breadth 864 min,
dive 576-min cap (source-limited, did not bind). Dive plan = **all 314** breadth admissions
(n_top 250 + n_control 64) — the plan size is a coverage knob, scaled to spend the dive leg
on the whole harvest rather than a 28-sample.

## Headline

| leg | active | admissions | adm/hr | cost/adm | notes |
|---|--:|--:|--:|--:|---|
| breadth | 864 min (14.4 h) | **314** | 21.8 | 2.75 min | best-first frontier |
| dive | 106.5 min (1.8 h) | **254** | 143.1 | **0.42 min** | 314 single-track descents, 0.81 adm/dive |
| **total** | **970.5 min (16.2 h)** | **568** | — | — | **508 distinct morph looks (89%)** |

## Corrected verdicts

**1. Breadth throughput — the ~16/hr floor HOLDS (corrected).** An earlier read reported the
curve "decays to 9/hr" in the final hour. That was a **partial-bin artifact**: the last active-hour
bin is only 0.40 h wide (active 14.0–14.4 h), so its 9-admission *count* is a 22.5/hr *rate*.
Rate-normalized per bin width, breadth holds **mean 21.8 adm/hr** over 14 full-hour bins with a
shallow **−0.64/hr-per-hr** trend and only **1/14 bins below 16/hr**. Mild late-run softening, no
collapse. → For scheduling, a 14 h breadth leg stays productive; no urgent re-seed cliff, but the
gentle negative slope favors **moderate legs over marathon single runs**.

**2. Dive is the efficiency lever — 6.5× cheaper per admission.** 0.42 vs 2.75 active-min/admission,
143 vs 21.8 adm/hr. Diving the existing harvest grows the ledger far cheaper than extending
breadth. Marginal diversity is good but below breadth: the 254 dive admissions added **+202 distinct
looks (80% marginal-distinct)** vs breadth's 97%, the shortfall concentrated in julia:multibrot3
(28 looks / 50 admits — sibling descents down one lineage). → **Weight future budget toward dives**;
accept that dive growth trades some distinctness for throughput.

> **CORRECTION (2026-07-19, post-audit).** Verdict 3 below attributed julia's churn to
> *hot-region dup-churn* — the julia hook re-firing into already-saturated neighborhoods. A
> subsequent audit ([`julia_dup_metric_audit.md`](julia_dup_metric_audit.md)) found the real
> mechanism is **dup-metric over-kill**: julia distinctness keyed on the z-viewport only, in a
> cloud shared across every distinct seed c of a base family, so **distinct-c Julias annihilated
> each other** (12/16 ledger dups collided against a *different* seed c; 191 distinct-c hooks
> wholly suppressed in breadth). The high `q3_dup` rate is therefore mostly the metric killing
> genuinely distinct Julia sets, not re-visits of the same place. The **multibrot4 churn is
> genuine** hot-region dup-churn (c-plane, unaffected by the julia keying bug) — that part of
> the verdict stands. Fixed in the julia dup-fix package (seed-c-aware key + hook spacing +
> pre-canonical filter + freshness prior); the recomputed julia overlap numbers supersede §4's.

**3. Julia cost is hot-region dup-churn, NOT low quality (attributed).** Julia's alarming
checks/admit (julia:mandelbrot 2717, julia:multibrot5 932) decomposes cleanly by harvest-check fate:

| profile | families | canon_not_q3 | q3_dup (pre-reframe) | reframe_fail |
|---|---|--:|--:|--:|
| cheap-scorer over-admission | mandelbrot / mb3 / mb5 | ~70% | ~25% | 0% |
| **hot-region dup-churn** | **julia:\* + multibrot4** | 1–14% | **78–99%** | 0% |

Julia checks almost never fail at decode or guard — they are canonical-q3 frames that the coord-dup
check then rejects as near-repeats of already-admitted places (the julia hook keeps re-firing into
saturated neighborhoods). The wasted compute is **the canonical confirmation render spent *before*
the coord-dup check**. → The lever is not "avoid julia" but a **pre-canonical candidate-coord dup
filter** (reframe only nudges ≤0.25·fw, so a candidate inside an admitted dup radius cannot escape
it — the same argument that already justifies the pre-*reframe* skip, pushed one render earlier).
That reclaims the bulk of julia + multibrot4 harvest compute at zero admission loss. Reaching julia
via **dives** (cheap, 0.42 min/adm, and they descend *out* of the saturated root neighborhood)
already sidesteps most of this.

**4. Library overlap — build a coord-space freshness prior; looks are fresh.** Against 14 prior
ledgers (493 distinct-q3 places): **coord-dup overlap 78/568 = 13.7%** (a campaign admission inside a
prior admission's dedup radius, same partition) but **morph near-dup only 9/568 = 1.6%** (CLIP≥0.974
vs the library store, library morph_gray recipe). We re-visit coordinate *neighborhoods* of prior
work but almost never reproduce a *look*. → A cheap cross-run coord-space freshness prior (reject
seeds within the prior cloud's dedup radius) would reclaim ~14% of harvest compute with negligible
look-diversity loss. A morph-space freshness prior is not yet worth it (1.6%).

> **CORRECTION (post dup-fix).** The 78/568 = 13.7% coord overlap above was measured with the
> buggy z-only julia keying. Recomputed under the fixed **seed-c-aware** metric (same 14 prior
> ledgers): prior distinct-q3 places **493 → 507** (distinct-c Julias are now counted as separate
> prior places, not collapsed) and overall coord overlap **61/568 = 10.7%**. Per partition:
> julia:mandelbrot 0/4, julia:multibrot3 2/50, julia:multibrot4 5/26, julia:multibrot5 7/18,
> mandelbrot 21/113, multibrot3 7/163, multibrot4 7/45, multibrot5 12/149. The freshness-prior
> verdict is unchanged (worth building — now shipped as the coordinate freshness prior). Cross-
> representation note: prior *production-seeder* julia rows key c on `outcome_cx` while the steered
> campaign keys c on `julia_c_re`, so the fixed metric treats them as non-collidable (you cannot
> compare a z-viewport to a c-coordinate) — those pairs no longer contribute spurious overlap.

**5. Coverage — all 8 partitions populated, no zero-admission family.** multibrot4 (the watched one)
healthy at 45; only julia:mandelbrot low (4, and its breadth leg is the most dup-churned). Combined
per-family: mandelbrot 113, multibrot3 163, multibrot4 45, multibrot5 149, julia:mandelbrot 4,
julia:multibrot3 50, julia:multibrot4 26, julia:multibrot5 18.

## Scheduling takeaways (what this prices)

- Breadth ≈ **2.75 min/admission at ~22/hr, floor-stable to 14 h**; dive ≈ **0.42 min/admission at
  ~143/hr** but ~80% marginal-distinct. A budget-efficient cycle is **breadth to seed the harvest,
  then dive the whole harvest** (dive is ~1/8 the breadth active-time for ~80% of the admission count).
- **Two cheap pipeline wins**, both pre-canonical-render coord-dup filters: a per-partition one kills
  julia/mb4 dup-churn (~80–99% of those checks); a cross-run one against prior ledgers kills ~14% of
  all harvest checks. Neither costs admissions.
- Julia is not low-value; it is dup-churn-expensive **in breadth**. Prefer julia via dives, or fix
  the hook's re-firing into saturated neighborhoods.

## Provenance / crash-safety

Both legs are `active_s`-budgeted and checkpoint every batch/dive; killed and resumed cleanly across
the multi-day window with zero lost/duplicated admissions (cloud rebuilt from the durable ledger).
Contact sheet: `out/campaign1/contact_sheet_all.png` (all 568, cluster-ordered). Tools:
`tools/atlas/campaign1_readout.py` (readout), `tools/atlas/campaign1_contact_sheet.py` (sheet),
runbook `out/campaign1/RUNBOOK.md`.
