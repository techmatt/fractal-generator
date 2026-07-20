# Invariant audit — freshness-prior suppression, identity round-trips, reject autopsy

Prompted by the julia dup bug: a metric that was semantically wrong but aggregate-plausible,
caught only by a human eyeballing an *absence*. This audit fixes one active bug (part 0) and
ground-truths the load-bearing identity/presentation assumptions (parts 1–2). One-line verdict
per item; detail below.

## Verdicts

**Part 0 — freshness-prior suppression**
- **Default flipped OFF first** (`--no-freshness-prior` → opt-in `--freshness-prior`). ✅
- **Root cause = the native-seed rejection sampler, not the dedup path.** The prior seeded 184
  mandelbrot q3 places into the cloud `draw_roots` feeds `count_within(radius=0.20) >= cap 5`;
  a productive-region seed has a **median 12 prior neighbours within 0.20**, so **97.8%** of
  seeds reject on arrival → `MAX_SEED_REDRAWS` saturation → **0 roots → 0 admits → 0 julia
  hooks**. The 10.7% overlap prediction was measured on the *fw-scaled dedup path* (median
  radius ~7e-4, ~287× tighter) — a different, benign mechanism. Two-order gap explained. 🔴→✅
- **Fix:** the rejection sampler now reads a **run-only** cloud (`run_clouds`); the prior still
  feeds the dedup + steering clouds (its real coverage purpose). Prior-OFF path is byte-identical
  (prior rows empty ⇒ `run_clouds == clouds`). ✅
- **Seed composition (suspicion 2):** the 7,942 "loaded rows" are raw pre-filter ledger rows;
  `build_cloud` filters to `guard_pass ∧ decoded_class==3 ∧ dedup` → **783 distinct q3 places**
  (mandelbrot 184). Cloud is admitted-coords-only per design, NOT harvest/candidate coords. ✅
- **Acceptance smoke (mandelbrot, julia-hook, 6-min, seed 7) — PASSED.** Identical config, prior
  OFF vs ON:

  | | admits | julia roots | precanon_dup | seeded cloud (mand/julia) |
  |---|--:|--:|--:|---|
  | prior-OFF | 12 | 4 | 368 | 0 / 0 |
  | prior-ON (fixed) | 11 | 3 | 591 | 184 / 102 |

  Roots survive; throughput 11 vs 12 admits is within noise. The prior is demonstrably active
  (cloud seeded to 184/102; precanon_dup rises 368→591 = the *intended* dedup overlap the prior
  exists to create). Contrast the pre-fix behaviour that triggered this audit: **0 admits / 0
  roots**. Bar met. **Ships default-OFF anyway** (the flag makes ON a deliberate opt-in); ON is
  now validated-safe, so `--freshness-prior` can be enabled or the default restored at will.

**Part 1 — identity & presentation round-trips**
- **Location identity is fully keyed per family** (table below); one **latent** gap flagged:
  Phoenix `(c,p)` is absent from the dup key (safe only while `(c,p)` stays the fixed Ushiki
  constant — the same bug class as the julia z-only key, dormant not active). 🟡
- **Round-trip renders (Guard B across all 13 batches, all 9 families):** stamped batches rebuild
  **byte-perfect** (gather_v6, all 9 families, mean|d|=0.000; jm3/jm45/blindspot/mining/prospect_R
  all 0.000). **One RED**: `2026-07-17_prospect_run1_baserate_v1` — 22 c-plane multibrot rows had
  `fractal_type` MISSING from the render block, so `from_render_block` defaulted them to
  **mandelbrot** and they re-rendered the wrong fractal (mean|d| 26–106; rendered as the true
  family → 0.000). Origin: `build_prospect_baserate.render_block` native branch (`extra={}`) — the
  julia-bug's twin (a version-invariant identity field silently absent, aggregate-plausible).
  **Fixed** (origin one-liner + proven data backfill; batch now Guard-B PASS at 0.000). The
  `flat_generate` batch trips the threshold at 5.36 (all-mandelbrot, no-stamp → default palette
  source + JPEG requant; benign, not an identity failure). 🔴→✅
- **Deploy presentation:** the production scoring path (`score_lib.Scorer`, wrapped by
  `guard.make_guarded_scorer`) builds `Transform(train=False)` from the checkpoint cfg, which
  decodes to the documented 1280×720→384×224 **bicubic-stretch + normalize** recipe, **bit-for-bit**
  on sample inputs. Pinned by `classifier/test_deploy_transform_parity.py` (3/3). ✅

**Part 2 — reject autopsy**
- **Rejects were coordless.** `harvest_log` logged no cx/cy/fw, so precanon_dup / canonical-not-q3
  rejects (31,888 rows in campaign-1 breadth alone) were **unrenderable** — the exact blind spot
  the julia bug hid in. Fixed: `_log_harvest` now stamps cx/cy/fw + julia c on every check. ✅
- **Standing habit:** `tools/atlas/reject_autopsy.py` renders fate-stratified sheets (admit_q3 /
  coord_dup / guard_fail / precanon_dup / canon_not_q3) at deploy fidelity; `julia_fix_readout.py`
  emits one by **default** (`--no-visual-sample` to skip). ✅

---

## Part 0 detail — freshness-prior suppression

Diagnosis script: `scratchpad/freshness_prior_diag.py` (loads the real prior library exactly as
`load_prior_library_rows` does, quantifies both mechanisms).

| partition | prior q3 places | median neighbours <0.20 | P(seed rejected) |
|---|--:|--:|--:|
| mandelbrot | 184 | 12 | 97.8% |
| multibrot3 | 149 | 11 | 96.6% |
| multibrot4 | 74 | 6 | 77.0% |
| multibrot5 | 127 | 8 | 86.6% |

The rejection gate (`REJECT_RADIUS=0.20`, `Q3_DENSITY_CAP=5`, fixed radius in (cx,cy)) was tuned
for a run-scoped cloud that **starts empty** and accrues a handful of places; the native seed
source draws from the *same* productive region the prior admits occupy, so a pre-populated prior
cloud saturates it immediately. The fw-scaled dedup path (`near_dup = DEDUP_K·max(fw)`, median
radius ~7e-4) is ~287× tighter and only costs the intended ~10.7% re-coverage — it never
sterilizes. The overlap measurement never exercised the root gate because the run-scoped cloud
was empty at root time.

**Fix seam** (`steered_frontier.py`): `build_run_clouds()` (run-ledger-only) feeds `draw_roots`;
`build_clouds()` (prior ⊕ run) feeds the pre-canonical filter, admission near-dup, and steering.
Both updated on admit and on resume. Prior-OFF ⇒ identical clouds ⇒ zero behaviour change.

## Part 1 detail — identity & dedup map

**Location identity per family** (canonical: `tools/corpus/location.py`):

| family | render identity | dup identity (seeder) |
|---|---|---|
| mandelbrot | family + cx,cy,fw | (cx,cy,fw); c=None |
| multibrot3/4/5 | `--family multibrotN` + cx,cy,fw | (cx,cy,fw); c=None |
| julia:mandelbrot, julia:multibrotN | render_family + **c_re,c_im** + cx,cy,fw | (cx,cy,fw) **AND seed c within 1e-6** ✅ (post-fix) |
| phoenix | family + c_re,c_im + **p_re,p_im** (family_params) + cx,cy,fw | (cx,cy,fw) only — **(c,p) not in the dup key** 🟡 latent |

Pixel-exact crop identity additionally pins maxiter, palette (name), ss, filter, interior_mode —
all carried in the corpus `render` block and consumed by `render_corpus_crop` → `render_one_flags`.

**Dedup / keying points** (what each compares):

| # | point | keys on | seed-c aware |
|---|---|---|---|
| 1 | within-run q3-cloud pre-reframe skip (harvest pre-canonical) | (cx,cy,fw)+c, fw-scaled | ✅ |
| 2 | admission near-dup (post-reframe) | (ocx,ocy,ofw)+c, fw-scaled | ✅ |
| 3 | cross-run freshness prior (build_cloud) | (cx,cy,fw)+c, fw-scaled | ✅ |
| 4 | native-seed rejection sampler (root draw) | (cx,cy) only, **fixed 0.20**, family-partitioned | n/a (c-plane roots) |
| 5 | julia root hook spacing | **seed c**, JULIA_HOOK_SPACING=0.20 | ✅ |
| 6 | emission intake dedup | exact `id` string | — |
| 7 | morphology dedup (curation) | CLIP cosine, within-family | — |

`build_cloud` partitions by `family`, so cross-family outcomes at the same (cx,cy) never interact.
Point 4 is the one that broke under the freshness prior (part 0).

## Part 2 detail — reject autopsy

Fates and their sources: `admit_q3` (ledger distinct-q3), `coord_dup` (ledger post-reframe
distinct=False **∪** harvest pre-reframe q3-dup), `guard_fail` (ledger guard sentinel),
`precanon_dup` and `canon_not_q3` (harvest_log; renderable only for coord-logged runs).

Sheets rendered at 640×360 ss2 deploy fidelity (`out/atlas/smoke_invariant/`):
- `campaign1_breadth_autopsy.png`, `campaign1_dive_autopsy.png` — the `julia:multibrot{3,4}`
  **coord_dup** rejects are visually near-identical to the multibrot admits: the over-kill
  signature, now eyeball-able. Campaign-1 harvest rows are coordless (pre-fix: 31,888 breadth /
  5,965 dive) so precanon/canon-not-q3 can't be re-shown for those runs — the gap the coord
  logging closes going forward.
- `prior_off_autopsy.png`, `prior_on_autopsy.png` — fresh coord-logged smokes; **all five fates
  render** (coordless_harvest=0). Validates the standing feature end-to-end.

The readout emits its sheet by default; `--no-visual-sample` opts out.
