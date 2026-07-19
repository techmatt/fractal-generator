# Prospect run 1 — per-family q3 price list

Read-only, cycles 1–8, from `seeder_summaries/*.json` + `timing.jsonl`. Prices are in **active
seconds** (subprocess `duration_s`, not wall-clock). The unit priced is a **distinct v6-q3** (a
coord-deduped location that becomes a library record), not a raw decode — see the attrition note.

**Purpose:** cost an order ("K q3s distributed like *this*"), not to set a budget. Price informs the
bill; it does not choose the order.

---

## The price list

| family | active sec (cyc1–8) | base q3 | twin q3 | distinct q3 | **sec / q3** | confidence |
|---|---:|---:|---:|---:|---:|---|
| **phoenix** | 5266 | — | — | 52 | **101** | moderate (n=52) — but see morph caveat |
| **mandelbrot** | 4000 | 12 | 12 | 24 | **167** | moderate (n=24) |
| **multibrot4** | 4579 | 2 | 4 | 6 | **763** | LOW (n=6) |
| **multibrot3** | 5002 | 0 | 5 | 5 | **1000** | LOW (n=5) |
| **multibrot5** | 4392 | 0 | 0 | 0 | **≥1464 / unfillable** | none (n=0) |

Per-family seconds are the **actual** per-family subprocess times, not a 4-way split of the
discovery total — and they are **not equal** (4000–5002 s), so an even split would misprice by up to
~25%. Each family is its own GPU subprocess (separable). **Within** a family, base-time and twin-time
are **not** separable — the julia hook runs inline in the same subprocess, so base and twin share the
family's seconds. The seconds→twin attribution is therefore clean only for the **zero-base families
(mb3, mb5)**.

**Attrition note (why "distinct", not raw):** ~half of raw v6-q3 decodes are within-run near-dups the
seeder's own cloud drops before anything is priced — mandelbrot 45→24 (47%), mb3 11→5 (55%), mb4
12→6 (50%). Pricing on raw decodes would understate cost ~2×. Downstream store-dedup (the 27 coord-dups
in the reconciliation readout) removes a few more across the pooled cycle.

---

## Twin supply chain (twins are not independently orderable)

Twins are byproducts of c-plane parents. Chain: **sec/c-plane descent → parents/descent →
hook/parent → twins/hook**. `hook/parent = 1.00` for every family (the density gate never fired in 8
cycles — see rebalance readout), so that link is lossless; the loss is all in parents/descent and
twins/hook.

| family | sec/descent (loaded) | parents/desc | hook/parent | twins/hook | twins/desc | sec/twin (backward) |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot | 18.0 | 0.077 | 1.00 | 0.71 | 0.054 | 333 \* |
| multibrot3 | 39.1 | 0.109 | 1.00 | 0.357 | 0.039 | **1000** |
| multibrot4 | 46.3 | 0.121 | 1.00 | 0.333 | 0.040 | 1145 \* |
| multibrot5 | 52.3 | 0.071 | 1.00 | 0.00 | 0.00 | ∞ |

\* mandelbrot/mb4 `sec/twin` **overstates** the marginal twin cost because those descents *also* emit
base q3 (joint product) — the seconds are shared. For mb3/mb5 (0 base) the backward cost is exact.

### Worked example — "I want 10 jm3 twins"

Cost it backward through the chain (mb3, exact because 0 base):

```
10 twins ÷ 0.357 twins/hook   = 28 hook descents
28 hooks ÷ 1.00 hook/parent   = 28 qualifying parents
28 parents ÷ 0.109 par/desc   = 257 c-plane descents
257 descents × 39.1 sec/desc  = 10,050 active sec ≈ 2.8 hr
                              ( = 10 × 1000 sec/twin )
```

**mb3's entire c-plane budget is a twin-acquisition cost: ~1000 active seconds per jm3 twin, 100% of
it overhead with zero base emitted.** The two lossy stages are parents/descent (only 11% of descents
qualify) and twins/hook (each hook of 3 julia walks yields 0.36 distinct twin). Widen either to cut the
twin price; the density gate is already lossless so there's nothing to gain there.

---

## Marginal cost — the 10th q3 costs more than the 1st

coord-dup rose 6% → 57% across the 8 cycles, so cost-per-q3 climbs with cumulative q3 in the seeded
region. Per-family N is too small for per-family marginal curves (except mandelbrot), so the c-plane
families are pooled:

**Pooled c-plane, early vs late half:**

| half | active sec | distinct q3 | sec/q3 |
|---|---:|---:|---:|
| cycles 1–4 | 9416 | 25 | **377** |
| cycles 5–8 | 8558 | 10 | **856** |

**c-plane q3 is ~2.3× more expensive in the back half.** mandelbrot alone (the only c-plane family with
enough N) shows the same: cyc1–4 = 110 sec/q3, cyc5–8 = 304 sec/q3 (**2.8×**). An order priced at the
8-cycle average silently overruns once you push past where these cycles stopped — budget the *marginal*
(back-half) number, not the blend.

**Phoenix does NOT escalate:** sec/q3 bounces 70–321 with no trend (mean ~100), because it descends a
fixed plane and is descent-limited, not coverage-limited. So over a long run phoenix gets *relatively*
cheaper as c-plane escalates — the cost-minimizing drift toward phoenix accelerates.

---

## Unfillable line item — multibrot5

84 descents, 6 parents, 6 hook descents, **0 base and 0 twin** over 8 cycles. Not priced as infinity
and not omitted — bounded:

- Rule-of-three on 0/84 descents → q3-rate ≤ 3/84 at 95%, i.e. ≤ 3 q3 in these 84 descents, so
  **price ≥ 4392 / 3 ≈ 1464 active sec/q3** — already ~9× mandelbrot's blend, and possibly truly
  infinite.
- twins are worse: 0/6 hooks yielded a twin, so twins/hook ≤ 0.5 (rule-of-three on 6) — an extremely
  weak bound.

**Verdict: unfillable at this sample.** Don't accept an mb5 line item without more cycles; treat any
mb5 quantity as "unknown, ≥1464 s each and likely unbuyable."

---

## Phoenix is the cheapest q3 in the system — plainly

**Yes: 101 active sec/q3, cheapest by ~1.65× over mandelbrot (167) and ~10× over the multibrots.** A
cost-minimizing order buys phoenix. **That is exactly how the library reached 43% phoenix with no one
choosing it** — the bill, left to itself, orders phoenix.

**Caveat that must ride with the phoenix price** (from the morph audit): phoenix q3 collapse to ≈one
morphological theme (a hooked serrated spiral, re-framed). Per *record* phoenix is cheapest; per
*distinct morphology* it is not — you are buying the same look repeatedly at 101 s each. If the order is
denominated in cell/morphology coverage rather than record count, the effective phoenix price is far
higher and the c-plane twins (distinct-morph rate 0.89 vs phoenix 0.68) are underpriced by this table.

---

## Honest limits

- **8 cycles; c-plane families have single-digit q3** (mb3 5, mb4 6, mb5 0). The mb3/mb4/mb5 rows and
  their chain rates are **low/none confidence** — directional, not point estimates. mandelbrot (24) and
  phoenix (52) are the only moderately-grounded rows.
- **Prices are region-and-cycle-specific.** The marginal escalation means these averages are already
  drifting up; they price *the next* q3 near where cycle 8 left off, not an arbitrary future one.
- **Base cost is never isolated** for base-producing families — base and twin are a joint product of the
  same c-plane seconds. Only the zero-base families give a clean seconds→twin price.
- No config changed — reporting only.
