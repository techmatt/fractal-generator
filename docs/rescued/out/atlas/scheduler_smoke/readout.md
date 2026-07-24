# Deficit scheduler (family-level, v1) — smoke readout

Cross-partition allocation for the steered frontier, denominated in **distinct looks** against
a target measure, replacing the single global p_good queue that Campaign 1 showed lets an
un-calibrated cross-family p_good comparison drive the family mix.

- Code: `tools/atlas/deficit_scheduler.py` (pure/torch-free logic) + `--scheduler` wiring in
  `tools/atlas/steered_frontier.py` (default **OFF**; off == byte-identical to pre-change).
- Seed config: `data/atlas/scheduler_prices.json`. Order book: `data/emission/target_measure.json`
  projected to per-partition marginals (here a deliberately-skewed copy,
  `data/discovery/sched_smoke/skewed_target.json`).
- Unit tests: `tools/atlas/test_deficit_scheduler.py` (17 passing).

## Smoke configuration
`--families mandelbrot,multibrot3,multibrot4,multibrot5 --julia-hook --scheduler
--scheduler-target skewed_target.json --lambda-m 0 --beta 0 --seed 0 --budget 30`.
The target boosts every julia twin 8× → a **julia-heavy order book (57.2% julia)**, deliberately
opposite to the c-plane-heavy mix a natural steered run produces, so allocation has to move.

**Run:** 116 batches, **29.8 min active**, 35 distinct-q3 admissions, **28 distinct looks**.

---

## 1. Allocation trace — shares move toward the skewed target
Distinct-look share started at 0 (empty tally) and, over the run, moved to within ~4 pts of the
julia-heavy target in aggregate:

| | julia distinct-look share |
|---|--:|
| target (order book) | **0.572** |
| final (run) | **0.536** |
| start | 0.000 |

Per-partition (target = projected marginal; look = final distinct-look fraction; served =
batches this partition's sub-queue was popped):

| partition | target | look_frac | distinct looks | price seed→final | batches served |
|---|--:|--:|--:|--:|--:|
| mandelbrot        | 0.097 | 0.071 | 2 | 3.0 → 1.84 | 10 |
| multibrot3        | 0.152 | 0.286 | 8 | 3.0 → 2.77 | 47 |
| multibrot4        | 0.040 | 0.000 | 0 | 3.0 → 3.00 | 9 |
| multibrot5        | 0.138 | 0.107 | 3 | 3.0 → 1.86 | 21 |
| julia:mandelbrot  | 0.030 | 0.036 | 1 | 4.0 → 4.00 | 4 |
| julia:multibrot3  | 0.282 | 0.250 | 7 | 4.0 → **0.80** | 20 |
| julia:multibrot4  | 0.145 | 0.000 | 0 | 4.0 → 4.00 | 0 |
| julia:multibrot5  | 0.115 | 0.250 | 7 | 4.0 → 1.66 | 5 |

The trajectory is legible in `scheduler_trace.jsonl`:
- **Batches 1–2** serve `multibrot3` — its *effective* deficit 0.434 is its base 0.152 plus the
  routed `julia:multibrot3` twin deficit 0.282 (twin queue empty). Julia demand buys c-plane work
  from the first batch.
- The instant `multibrot3` banks its first distinct look, its deficit flips **negative**
  (look-denominated: with 1 look total, its look_frac = 1.0 ≫ target) and the scheduler rotates
  to `multibrot5`, `multibrot4`, `mandelbrot`.
- **Batches ~21+** concentrate on `julia:multibrot3` **directly** — the hooks fired earlier have
  grown its own sub-queue (4→14→52 nodes), so the highest remaining deficit is now servable
  without routing.

**Honest limits at 28 looks.** Direction is strongly correct (julia 0 → 53.6%), but per-partition
precision is coarse at this look count: `julia:multibrot5` overshot (0.25 vs 0.115 — it was cheap
and productive), and the entire **multibrot4 lineage stayed at 0** (target 0.04 + 0.145). The
scheduler *did* serve `multibrot4` 9× and routed for its twin, but the family yielded no q3 in
30 min, so no `julia:multibrot4` hook ever fired — a discovery-yield limit of a hard family, not a
scheduling fault. Its price correctly never dropped below the 3.0 seed (it banked no looks).

## 2. Online prices update from their seeds
Seeds were neutral (3.0 c-plane / 4.0 julia); the EMA of active-minutes-per-distinct-look took
over. `julia:multibrot3` fell **4.0 → 0.80** (the best value on the board — many looks per minute),
`julia:multibrot5 → 1.66`, every productive c-plane dropped to ~1.8–2.8. Partitions that banked no
look (`multibrot4`, `julia:multibrot4`, `julia:mandelbrot`≈1 look) held near seed, exactly as the
price definition requires. Full trajectory: `prices` field per batch in `scheduler_trace.jsonl`.

## 3. Julia routing — demand buys c-plane work + hooks
**Design note (the one open point).** A `julia:X` partition can only be *fed* by descending c-plane
`X` and hooking a qualifying parent — you cannot "pop" a julia look into existence. So when
`julia:X` has positive deficit but an **empty own queue**, its (positive) deficit is folded into its
c-plane parent's *effective* deficit (`effective_deficits`, weighted by `julia_route_gain=1.0`).
Serving c-plane `X` then fires the hook, seeding `julia:X` roots that later become directly
poppable and compete on their own (no double-count once the twin has a queue). Root draws use the
same twin-inclusive rule. Chosen for simplicity: it needs no separate julia planner — the existing
price-weighted-deficit pop and the existing hook do all the work; the only new idea is the deficit
fold. Alternatives considered (a dedicated julia budget; spending *proportional* to twin price)
were dropped as more machinery for no clear gain at this scale.

**Evidence.** Of 116 batches, **64 (55%) were routed-for-twin** — a c-plane partition served while
its julia twin had positive deficit and an empty queue (47× multibrot3 on behalf of
julia:multibrot3). **9 julia hooks fired** (4 skipped by c-spacing), producing **24 julia
admissions vs 13 c-plane** — the julia-heavy target realized through c-plane spending.

## 4. Distinct-look tally spot-check (incremental vs post-hoc)
For the most-admitted partition, `julia:multibrot5` (11 admissions), the **live incremental tally**
matched a fresh post-hoc pass exactly:

| method | distinct count |
|---|--:|
| live incremental tally (run) | 7 |
| replay of the tally rule on re-embedded admissions | 7 (gap 0) |
| independent post-hoc medoid clustering (cos 0.974) | 7 (gap 0) |

So 4 of the 11 admissions were near-duplicate *looks* (correctly not counted); the incremental
tally is exact against both its own replay and an independent clustering method.

## 5. Kill mid-run + resume preserves scheduler state exactly
Hard-killed the worker (`taskkill /F`, crash simulation) at the batch-18 checkpoint (4 admissions,
28-look-in-progress state, prices moved), then `--resume`:
- Resume banner: `batch 18, admitted 4 (cloud rebuilt from ledger: 4 places)` — **no loss, no
  double-count** (admissions rebuilt from the durable ledger, not the checkpoint).
- First post-resume decision (batch 19) used prices **loaded from the checkpoint**
  (multibrot3 = 2.1686, multibrot5 = 2.2785), **not** reset to the 3.0/4.0 seeds.
- Distinct-look tally counts at kill `{multibrot3:1, julia:multibrot3:1, multibrot5:2}` persisted
  as a prefix of the final tally (no look lost). Run then completed to budget normally.

## 6. Scheduler-off byte-identity
Ran the **new** binary scheduler-off and the **pre-change HEAD** `steered_frontier.py` at the same
seed / same run-dir basename, and diffed the deterministic append-only artifacts on their common
prefix:

| artifact | new lines | old lines | differing |
|---|--:|--:|--:|
| outcome_ledger.jsonl | 7 | 7 | **0** |
| harvest_log.jsonl | 331 | 331 | **0** |
| prio_terms.jsonl | 938 | 938 | **0** |
| julia_hooks.jsonl | 4 | 4 | **0** |

Identical cloud sizes. The scheduler-off path is provably inert (every scheduler call guards on
`self.scheduler is None`; RNG draw order, gates, and root pipeline are untouched).

## 7. Preference-ranker scope
The preference ranker (`pref_loc_*`) does **not** appear anywhere in scheduling — the pop decision
(`deficit_scheduler.choose_partition`) is a pure function of per-partition deficits and prices, and
a test asserts its signature contains no p_good/node/score parameter. Commented at the seam in
`pop_batch_scheduled`.

## 8. Visual sample
- Admissions (18 of 35): `admissions_sheet.png` — diverse, julia-heavy set across all six active
  partitions, each labeled family + p_good.
- Rejects (sample): `rejects_sheet.png` — harvest checks that failed the canonical/reframe gate.

## Artifacts
- Run: `data/discovery/sched_smoke/smoke_on/` (ledger, `scheduler_trace.jsonl`,
  `distinct_looks.npz`, `state.json`).
- Analysis: `out/atlas/scheduler_smoke/analysis.json`.
- Skewed target: `data/discovery/sched_smoke/skewed_target.json`.

## Runtime estimate (for a production scheduler run)
The smoke burned its 30-min active budget in 116 batches with lambda_m/beta off; wall time was
~40 min (fresh-run startup + the root-draw pipeline are not charged against the active budget).
A production loop with the morph novelty term on (`--lambda-m 0.5`) adds a per-candidate CLIP embed
and runs proportionally slower; budget by active-minutes as usual and background with
checkpoint/resume (validated above). GPU is free.
