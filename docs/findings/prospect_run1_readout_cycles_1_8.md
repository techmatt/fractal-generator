# Prospect run 1 — readout, cycles 1–8

Run `prospect_run1`, live (this readout taken while cycle 9 was in flight; all numbers
are cycles 1–8, committed watermark 968). Read-only; sources: `seeder_summaries/*.json`
(per-family rebalance), `timing.jsonl` (per-phase active time + yields), `orchestrator.log`
(`julia pq+N` batch lines, reconcile lines), run-scoped ledger phoenix rows.

---

## 1. Rebalance verdict

### Per-family, 8-cycle totals

| family | c-plane desc | qual parents | hook desc | q3 base | q3 twin | par/desc | hook/par |
|---|---:|---:|---:|---:|---:|---:|---:|
| mandelbrot | 222 | 17 | 17 | **12** | 12 | 0.08 | **1.00** |
| multibrot3 | 128 | 14 | 14 | **0** | 5 | 0.11 | **1.00** |
| multibrot4 | 99 | 12 | 12 | 2 | 4 | 0.12 | **1.00** |
| multibrot5 | 84 | 6 | 6 | **0** | **0** | 0.07 | **1.00** |
| **all** | **533** | **49** | **49** | **14** | **21** | 0.09 | 1.00 |

### Headline: the hook is 100% parent-supply-limited

`hook_descents == qualifying_parents` for **every** family, every cycle: **hook/par = 1.00**.
The Julia-density gate (`Q3_DENSITY_CAP`) never fired in 8 cycles — every qualifying parent
the c-plane produced was consumed by a hook descent. The hook is not throttled by density;
it is starved of parents. So parent supply is the binding constraint on twin production, and
c-plane descent is the *only* source of parents.

### The main question — c-plane descent yielding neither base q3 nor a qualifying parent

**Yes, and it's the norm: ≥86% of c-plane descents are barren** (produce neither a qualifying
parent nor a base q3): mandelbrot 193/222, mb3 114/128, mb4 85/99, mb5 78/84. Parent yield is
only ~9% of descents (49/533).

**But the barren descents are not a trimmable early-saturation tail.** Intra-cycle arrival
(`julia pq+N` per batch):

- At 7 min/family, mb3/4/5 complete only **1–2 batches per cycle** — mb4/mb5 are usually a
  *single* batch. Sub-cycle temporal resolution is therefore 1–2 points; the first-half/last-half
  split is a single-batch artifact, not signal.
- Where 2 batches ran (mb3 cyc 1–4), parents appear in **both** batches (e.g. cyc1 b1=0/b2=3,
  cyc2 b1=2/b2=2, cyc4 b1=1/b2=1), i.e. arrival is proportional to descents, spread across the
  budget — **no early exhaustion followed by a barren tail.**

**Verdict:** for mandelbrot / mb3 / mb4, c-plane descent *is* the mechanism that finds the
twins — parents arrive throughout, every parent is consumed, and twins come only from these
parents. There is **no early-saturation point to size `--mb-cplane-min` against**; a c-plane
budget cut would reduce parent supply (and thus twins) roughly linearly, not trim overhead.
**Close the `--mb-cplane-min` thread — it isn't the lever these 8 cycles support.**

### Base vs twin — overhead vs mechanism, per family

- **mandelbrot**: 12 base + 12 twin. C-plane pays for itself at the emitted level *and* supplies
  twins. Not overhead.
- **multibrot3**: **0 base** + 5 twin. C-plane is **pure twin-supply** — zero emitted base value;
  its entire worth is the 5 twins via the hook. Cutting it costs those 5.
- **multibrot4**: 2 base + 4 twin. Mostly twin-supply, marginal base.
- **multibrot5**: **0 base + 0 twin over 8 cycles.** 84 descents → 6 parents → 6 hook descents →
  **nothing**. Currently pure overhead.

### The one real lever: throttle mb5, not `--mb-cplane-min`

mb5 is the only family that is barren at *both* levels. The correct knob is **`--mb5-every`**
(throttle, don't drop). **Caveat: the sample is tiny** — 6 hook descents, 0/6 twins — 8 cycles
cannot distinguish "mb5 twins are zero" from "mb5 twins are rare." Recommend a few more cycles
before pulling mb5; if mb5 twin stays 0 through ~cycle 15, set `--mb5-every 2` or 3.

### Parent starvation — answered

hook/par = 1.00 everywhere *is* the starvation signal: qualifying-parents/cycle exactly equals
the hook's consumption rate. The hook wants every parent the c-plane can produce. This argues
against cutting c-plane budget and, if twin volume is the goal, mildly *for* more c-plane (or a
looser parent gate) — not less.

---

## 2. Cycle economics

### Per-phase active seconds (from `timing.jsonl`)

| cyc | disc (4 fam) | phoenix | pool | annotate | cycle | pool/disc | records |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2435 | 666 | 858 | 65 | 4023 | 0.35 | 17 |
| 2 | 2250 | 626 | 838 | 56 | 3770 | 0.37 | 16 |
| 3 | 2350 | 737 | 1262 | 77 | 4427 | 0.54 | 21 |
| 4 | 2381 | 656 | 413 | 19 | 3469 | 0.17 | 4 |
| 5 | 2120 | 643 | 905 | 51 | 3718 | 0.43 | 13 |
| 6 | 2079 | 690 | 472 | 33 | 3274 | 0.23 | 6 |
| 7 | 2198 | 628 | 930 | 67 | 3823 | 0.42 | 13 |
| 8 | 2161 | 619 | 426 | 19 | 3225 | 0.20 | 3 |
| **Σ** | **17974** | **5266** | **6103** | **386** | **29729** | **0.34** | **93** |

**Phase share of active time: discovery 60.5% · pool 20.5% · phoenix 17.7% · annotate 1.3%.**

- **Discovery dominates; pool does NOT.** pool/disc = 0.34 overall (pool is ~⅓ of discovery).
  This **corrects the launch-time estimate**, which assumed the uncapped pool would dominate.
  Discovery (4 families × ~7 min c-plane + the hook descents that run inside each seeder
  subprocess) is the tail that sets cycle length. Pool scales with per-cycle q3 (413–1262 s) but
  never overtakes discovery.
- **Stable, no drift.** Cycle wall 3225–4427 s (~54–74 min, mean ~62 min). Discovery ~2080–2435 s
  with a slight downward drift. Phoenix flat ~620–740 s.

### Records/hr vs accumulated active time — declining

| cyc | cum active h | cum records | records/hr (cum) |
|---:|---:|---:|---:|
| 1 | 1.12 | 17 | 15.2 |
| 3 | 3.39 | 54 | 15.9 |
| 5 | 5.39 | 71 | 13.2 |
| 8 | 8.26 | 93 | **11.3** |

Records/hr is trending **down ~26%** (15.2 → 11.3). Fresh q3/cycle fell 18 → 7 and coord-dup
fraction rose (see §3) — the discovery cloud is saturating within the seeded region, so new
distinct locations get rarer per unit active time. Expected for a fixed-region discovery process;
not a defect.

---

## 3. Reconciliation — clean, expected attrition

**120 fresh q3 = 93 records + 27 coord_dup + 0 field_fail + 0 deferred + 0 unexplained.**

| cyc | q3_found | recorded | coord_dup | field_fail | deferred | unexplained |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 17 | 1 | 0 | 0 | 0 |
| 3 | 27 | 21 | 6 | 0 | 0 | 0 |
| 4 | 8 | 4 | 4 | 0 | 0 | 0 |
| 8 | 7 | 3 | 4 | 0 | 0 | 0 |
| **Σ** | **120** | **93** | **27** | **0** | **0** | **0** |

The 27 are **all coordinate re-discoveries** (3 within-cycle + 24 vs an already-stored record) —
the ONE always-legitimate drop. **0 field_fail, 0 unexplained: the keep-every-q3 invariant holds.**
Not a leak — expected attrition, and rising with saturation (cyc1 coord-dup 6% of q3 → cyc8 57%),
the same signal as the records/hr decline. (Note: the `120` is the pool-admission count
`guard_pass ∧ decoded==3` *before* store-dedup; the seeder's own within-cloud `distinct` flag is a
separate earlier dedup layer — no contradiction.)

Largest single q3 source: **phoenix (52 distinct within watermark)**, vs c-plane families 35
(14 base + 21 twin). Phoenix carries the run's q3 volume.

---

## 4. Phoenix — decode threshold is re-derivable NOW

288 phoenix outcomes logged (through the current ledger), **all guard_pass** (the fixed Ushiki
location is always structured, so the guard never fires), **52 keepers** at the current
`t_good=0.18`. Every row carries `p_notbad` + `p_good` + `t_good`, so the threshold is fully
re-derivable post-hoc **for any candidate t_good** — no new cycles needed for that.

`p_good` over the 288 rows: min 0.032 · p25 0.063 · median 0.093 · p75 0.143 · max 0.593.
Dense sampling across the decision band (149 rows <0.10, 72 in [0.10,0.15), 15 in [0.15,0.18),
7 in [0.18,0.20), 24 in [0.20,0.25), 11 in [0.25,0.30)).

Keeper-count sensitivity (approx `p_good≥t ∧ p_notbad≥0.5`, which reproduces the actual
`corn_decode==3` count of 52 at 0.18 — validating the re-derivation):

| t_good | 0.10 | 0.15 | **0.18** | 0.20 | 0.25 | 0.30 |
|---|---:|---:|---:|---:|---:|---:|
| keepers | 139 | 67 | **52** | 45 | 25 | 10 |

**Verdict: the evidence is there now.** 288 scored outcomes densely sampled around 0.18 is ample
to re-derive the phoenix decode threshold (supersedes the earlier N=36 "undersampled" caveat).
Threshold NOT changed — reporting only.

---

## Bottom line

1. **hook/par = 1.00 across all families** — the Julia-hook is parent-starved, not density-throttled.
   More parents → more twins; the density gate is inert so far.
2. **Close `--mb-cplane-min`** — barren descents dominate (≥86%) but arrive throughout, and every
   parent is consumed; c-plane is the twin-finding mechanism for mandelbrot/mb3/mb4, not trimmable
   overhead. mb3 in particular is pure twin-supply (0 base, 5 twin).
3. **mb5 is the only cut candidate** (0 base + 0 twin / 84 descents) → lever is `--mb5-every`, but
   6 hook descents is too small to commit; revisit ~cycle 15.
4. **Discovery (60.5%), not pool (20.5%), sets cycle length** — corrects the launch estimate.
   Cycle wall stable ~62 min; records/hr declining 15→11 as the cloud saturates.
5. **Reconciliation clean** — 120 = 93 + 27 coord-dup, 0 field_fail, 0 unexplained.
6. **Phoenix threshold re-derivable now** (288 outcomes, full p_good/p_notbad, mapped sensitivity).
