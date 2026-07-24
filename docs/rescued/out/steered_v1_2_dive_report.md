# Steered v1.2 — dive run + novelty-memory fix + mandelbrot t_good

Dive run `steered_v1_2_dive` off `steered_run2`'s committed state: **28/28 dives** (20 top + 8 control), target_depth 23, fw floor 2e-09, active 9.2 min. Single-track descents: each rung expands 4 candidates under the existing gates, harvests every survivor at the per-partition tau_h, and continues down the cheap-p_good argmax child. Morphology is the LIBRARY morph_gray recipe (offline, admissions-only) — comparable to the run-2 library yardstick.

## 1. Dive yield

| start group | dives | admissions | adm/dive | median rungs | median end-depth |
|---|---:|---:|---:|---:|---:|
| top | 20 | 17 | 0.85 | 12 | 19 |
| control | 8 | 4 | 0.50 | 14 | 20 |
| all | 28 | 21 | 0.75 | 12 | 20 |

**Total dive admissions: 21** (canonical_q3 seen 225, q3_dup 205). End causes: {'gate_dead_or_floor': 26, 'target_depth': 2}. Every admission carries its `dive_id` + `dive_source_id` in the run ledger.

Top-start dives yield **0.85** admissions/dive vs control **0.50** — a good starting neighborhood produces more deep admissions, but control dives from arbitrary run-2 admissions still reach depth and admit, so deep quality is not exclusive to the best neighborhoods.

### Depth distribution of dive admissions

| depth | top | control | all |
|---:|---:|---:|---:|
| 4 | 0 | 1 | 1 |
| 5 | 2 | 1 | 3 |
| 7 | 6 | 0 | 6 |
| 8 | 1 | 0 | 1 |
| 9 | 1 | 0 | 1 |
| 10 | 2 | 0 | 2 |
| 11 | 4 | 1 | 5 |
| 13 | 1 | 1 | 2 |

Median admitted depth **8**, max **13** (dives seeded from run-2 admissions at depth 3–12).

### Canonical p_good of dive admissions

- **top** (n=17): canon p_good median 0.712, mean 0.690, range [0.438, 0.907].
- **control** (n=4): canon p_good median 0.601, mean 0.631, range [0.544, 0.779].
- **all** (n=21): canon p_good median 0.678, mean 0.679, range [0.438, 0.907].

**Deep-options read:** the top-start vs control comparison tests whether deep quality requires a good starting neighborhood or is reachable from anywhere — the blind manifest (`out/dive_manifest/`) adjudicates it on the human read; the yield numbers above are the classifier-side view.

## 2. Morph novelty of dive admissions vs the run-2 library

Each dive admission's cheap-look CLIP embedding (library morph_gray recipe) vs the 75 run-2 admission embeddings — `cos_max` is its nearest run-2 look (higher = less novel). Yardsticks: library-wide median pairwise cos 0.851, strict near-dup cut cos>0.974.

| group | n | median cos_max vs run-2 | p90 | near-repeat (cos>0.974) |
|---|---:|---:|---:|---:|
| top | 17 | 0.958 | 0.979 | 3 (18%) |
| control | 4 | 0.959 | 0.968 | 0 (0%) |
| all | 21 | 0.958 | 0.978 | 3 (14%) |

Dive admissions cluster to **21 distinct morphs** among themselves (strict cos>0.974, from 21 — no internal collapse). Median cos_max vs the run-2 library is **0.958**: the dives descend FROM run-2 admissions, so their looks are lineage-RELATED to the library (deeper views of the same neighborhoods), which is expected — but only **3/21** cross the near-dup cut (14%), so the deep views are morphologically distinct looks, not re-buys of the run-2 admissions they descend from.

## 3. Novelty-memory saturation before/after the fix

Saturation fraction = candidates whose novelty penalty is within 10% of full (cos_max past 90% of the [lo,hi] ramp) — a high fraction means the penalty is a constant down-shift, not a gradient. run-2 (v1.1, legacy all-permanent memory) saturated at **0.897** with a **10,420-row** memory (see `steered_run2_report.md`). The fix makes memory ADMITTED-only + a rolling window of the last K batches' expanded looks, so |memory| stays bounded and the term stays a live gradient.

| run | memory mode | batches | end |memory| | overall sat_frac |
|---|---|---:|---:|---:|
| steered_run2 (before) | legacy all-permanent | 341 | 10420 | **0.897** |
| shakeout_legacy (legacy) | legacy all-permanent | 16 | 405 | **0.704** |
| shakeout_recency (recency) | recency (k=8) | 17 | 268 | **0.575** |

Per-batch saturation trajectory (early vs late thirds of each shakeout):

| run | early third | mid third | late third |
|---|---:|---:|---:|
| shakeout_legacy (legacy) | 0.277 | 0.839 | 0.844 |
| shakeout_recency (recency) | 0.212 | 0.734 | 0.628 |

At matched budget the legacy shakeout climbs toward run-2's saturated regime (overall 0.704, and still rising — memory unbounded) while the recency window holds memory bounded (268 rows vs run-2's 10,420) and saturation at **0.575**. On the recency shakeout `nov_pen` has mean 0.371 / std **0.183** with **14%** of candidates at zero penalty and the rest spread across (0, 0.50] — a live gradient, versus run-2's near-constant ~0.489 offset.

The residual saturation is intrinsic: a descent produces a chain of morphologically similar views, so a candidate almost always has a near-mate among its own recent lineage in the window — independent of total memory size. The memory fix removes the unbounded-density driver (10,420→bounded) and restores a varying penalty; driving saturation lower would need a per-lineage-excluded novelty or a higher knee (anchors held fixed here per spec).

## 4. Mandelbrot discovery t_good (F0.5 re-derive)

Re-derived precision-weighted (F0.5) with the 16 steered_run2 blind mandelbrot labels folded into the v7 eval slice (n=942, pos=29). The blind read scored **0/16** mandelbrot admissions good.

- **mandelbrot t_good 0.14 (F2) → 0.51 (F0.5)** — applied to `production_seeder.T_GOOD_OVERRIDES`.

- On the 16 steered mandelbrot tiles (all human-not-good): the old bar admitted **16/16**, the new bar admits **0/16**.

- Deliberate, family-specific admission tightening (precedent: phoenix 0.18→0.50); the julia families keep their F2 cuts. Full derivation: `docs/findings/mandelbrot_tgood_steered.md`.
