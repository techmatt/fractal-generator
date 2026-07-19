# v7 retrain — scope & plan (PLAN ONLY; nothing trained, no manifest/threshold touched)

**Date:** 2026-07-17 · **Status:** proposal. `data/v6/manifest.jsonl` stays frozen — v6 is
the rollback, exactly as v5 is v6's. No `ACTIVE_CKPT` switch, no `t_good` change.

**One-line thesis:** append the post-freeze labels to a byte-frozen v6 prefix; force the
144-location julia census → **eval** and the 125 band locations → **train**; retrain. The
train-side julia:multibrot positive class goes **13 → 95** (7.3×). This is worth doing and
cheap, but the eval is underpowered — a modest gain will train but may be unprovable at
n=144. Native multibrot changes nothing and stays unmeasurable.

---

## 1. Data — what v7 trains on that v6 didn't

Counts are **location-level** (manifest unit = one render location, `label = max` over its
crops), via the canonical resolver `label_store.resolve_score` (the trainer's own path), on
the **current** store — i.e. **after** the stage-2 R census merge, so these supersede the
5713 figure in `v6_training_coverage_audit.md` (now **5797** labeled locations; the +84 is
the stage-2 R julia:mb census).

| family | labeled locs (now) | q3 (now) | in v6 | v6 q3 | **post-freeze** | **post-freeze q3** |
|---|---:|---:|---:|---:|---:|---:|
| mandelbrot        | 4004 | 387 | 3737 | 377 | 267 | 10 |
| multibrot3        |   64 |   2 |   64 |   2 |   0 |  0 |
| multibrot4        |   57 |   6 |   57 |   6 |   0 |  0 |
| multibrot5        |   68 |   1 |   68 |   1 |   0 |  0 |
| julia (J0)        | 1087 | 252 | 1087 | 252 |   0 |  0 |
| **julia:mb3**     |  186 |  48 |   75 |   1 | **111** | **47** |
| **julia:mb4**     |  147 |  59 |   62 |   8 |  **85** | **51** |
| **julia:mb5**     |  111 |  55 |   38 |   4 |  **73** | **51** |
| phoenix           |   73 |   7 |   73 |   7 |   0 |  0 |
| **TOTAL**         | 5797 | 817 | 5261 | 658 | 536 | 159 |

`in v6` sums to **5261** = the exact v6 manifest row count (join on full identity
`(fractal_type, cx, cy, fw, c_re, c_im)`). Excluded, correctly: `palette_scores`,
`render_mode_*`, `wallpaper_*` — different heads.

**The load-bearing decomposition (julia:mb3/4/5, exact):**

- post-freeze julia:mb = **269 loc / 149 q3** = **125 band** (`jm3_band` 57·26q3 + `jm45_band`
  68·56q3 = 82 q3) **⊎ 144 census** (67 q3). Band ∩ census = ∅ at identity / seed-`c` /
  `parent_oid` (re-verified in the census readout).
- **v7 train julia:mb positives = 13 (in-v6) + 82 (band) = 95.** **v7 eval julia:mb
  positives = 67 (census).** This is the task's "13 → ~95," reproduced to the unit.

The SIDECAR_LABELS registry fix (corpus_reader 6877 → 7006 crops) is what makes the 125
band locations visible to the trainer's reader at all; without it they'd be silently
UNLABELED and never enter training. It is a precondition of this plan, already landed.

**Native multibrot gets zero new positives** — mb3/4/5 are 100% in v6 (post-freeze = 0). Its
positive class stays **9** (2/6/1). See §6.

---

## 2. Split — **append, do not rebuild**

v5→v6 precedent: freeze the prior manifest **verbatim** (every prior location keeps its
split + group_id + row order), append the new batch with a fresh split. v7 follows it:

- **Frozen prefix:** v6's 5261 rows copied byte-for-byte (loc_ids 0…5260). Preserves the
  v5↔v6↔v7 eval-comparability chain **and** — critically for §4 — carries v6's mandelbrot
  eval (942 loc / 29 q3) forward *unchanged*, so "did mandelbrot regress?" is answerable
  against v6's own numbers on the identical set. **Rebuild is rejected**: re-rolling
  `EVAL_FRAC` reshuffles the mandelbrot eval and destroys that paired comparison for a class
  that already works and is 70% of the corpus.

- **Appended 536 post-freeze locations**, split by rule (all standing rules honored:
  location-disjoint, group_id integrity, biased→train, julia children inherit seed's split):
  - **Census 144 → `eval` (forced 100%, not `EVAL_FRAC`).** The census's entire value is
    being the *complete* unbiased-given-descent julia:mb eval; letting 60% leak to train
    would both gut the eval and hand training its only unbiased draw. Identified by batch:
    `2026-07-17_prospect_run1_baserate_{v1(julia rows),R_v1}`.
  - **Band 125 → `train`.** Model-band-selected (`decoded_class=2`) ⇒ biased ⇒ train. Batches
    `2026-07-11_jm3_band_v1`, `2026-07-12_jm45_band_v1`.
  - **All other post-freeze (267 mandelbrot: blindspot 219 + prospect native-plane) → `train`.**
    Blindspot is negative-by-construction (v6-reject, 4 q3) and must never sit in a q3-vs-rest
    eval; prospect native-plane is descent-screened. Both biased ⇒ train. This means the
    appended block contributes **no new eval outside the census** — intentional: eval power
    for mandelbrot/J0 already lives in the frozen v6 eval.
  - **group_id**: assign appended locations via the existing §5 neighborhood union-find
    partitioned by `(family, c-bucket)`, offset above v6's `GATHER_GID_OFFSET` range so ids
    never collide. Because census ∩ band = ∅ at seed-`c`, no census location can share a
    c-bucket (hence a group) with a band/train location → **no group straddles the split by
    construction.** That seed-`c` disjointness is exactly why the readout checked it.

  **Net v7 split ≈ 4109 + (125 band + 267 other) = 4501 train / 1152 + 144 = 1296 eval.**

### Verifiability (same guarantees v6's builder asserts, plus a frozen-prefix gate)

The builder must assert, and abort on any violation:

1. **0 orphans** — every appended row has non-null `label`, `split`, `group_id`.
2. **0 identities straddling** — `(fractal_type,cx,cy,fw,c_re,c_im)` in `train ∩ eval` = ∅.
3. **0 group_ids straddling** — no `group_id` has members in both splits.
4. **0 biased-in-eval** — every eval location is unbiased (census only).
5. **Forced assignments hold** — all 144 census identities ∈ eval; all 125 band ∈ train.
6. **Frozen-prefix byte gate** — regenerate rows 0…5260 and assert byte-identity to
   `data/v6/manifest.jsonl` (the analogue of build_plan's recipe-parity gate; frozen prefix
   must not drift). Extend to the cache manifest: the frozen v5+gather aug rows stay verbatim.
7. **Census→eval disjointness re-assert** — 0 overlap census vs all-train at identity /
   seed-`c` / `parent_oid` (already 0/0/0; re-run as a build gate, not a one-off).

All of these are read-only checks on the built manifest; they are the plan's acceptance test.

---

## 3. Presentation

- **Labels attach to locations, training re-renders from coords — confirmed.** Manifest row
  is a location; `build_plan.py` emits `plan_gather.jsonl` (cx/cy/fw/c/fractal_type) that
  `v4-render-batch` renders fresh into the aug cache. The batches' **stored crops are never
  read for training** — so the census's 640×360 ss2 JPGs do not constrain training. (They
  *are* the eval scoring input; see §4.)

- **Deploy presentation = 640×360 ss2 `twilight_shifted`.** Aug recipe (unchanged v4/v5/v6):
  1280×720, 6 palettes, scale {0.7,1.0,1.3}, shift {center,shifted}, AA {ss1 box, ss4
  lanczos3} = 42 crops/loc. Coverage of the deploy point:
  - **Palette: covered.** `twilight_shifted` is aug palette #0 (`aug_roster.json`).
  - **Geometry: covered.** Both 640×360 and 1280×720 are 16:9 → `Transform` bicubic-stretches
    both to 384×224; the stretch dominates the resolution difference.
  - **AA level: NOT covered — the residual.** Deploy is ss2; the aug set has ss1 and ss4 but
    no ss2. After the 384×224 stretch the ss2 high-frequency signature differs slightly from
    ss1/ss4. This is small (palette + geometry match), but it is **the one piece of the
    otherwise-retired "presentation" hypothesis that survives** — the census AUC result
    pinned the failure on data starvation, but this covariate gap is real and cheap to close.
  - **Recommendation:** add a **640×360 ss2 twilight_shifted** slot to the aug fan-out **for
    the 536 appended locations only** (append-only; the frozen-prefix byte gate in §2.6
    forbids touching the 5261 frozen rows' aug rows, so this cannot be retrofitted — that's
    acceptable, the deploy-geometry match is what matters for the new families). Cost: +1
    crop/loc × 536 ≈ negligible render. If we decline it, state explicitly that the ss2 AA
    gap is a knowingly-accepted second-order covariate shift.

---

## 4. Metric, baseline, success criterion, and the mandelbrot-regression guard

- **Eval / metric:** AUC(q3 vs rest) on the 144-location census, Mann-Whitney with 5000-boot
  95% CI — the same instrument the census readout used, so v6→v7 is like-for-like.
- **Baseline:** v6 on the census = **AUC 0.571 [0.48, 0.66]** (chance is *inside* the CI).

**Detectable effect size — put the number down now.** n = 144 (P=67, N=77). Hanley–McNeil
SE(AUC) ≈ **0.048** at A≈0.5–0.6 (95% half-width ≈ 0.095 — matches the census's [0.48,0.66]).
Consequences:

- For v7's own CI to **clear chance** it needs **AUC ≳ 0.60**.
- To be **credibly better than v6's 0.571**, use the **paired DeLong test** (re-score the
  *same* 144 census locations with both v6 and v7 — maximizes power via score correlation).
  Independent-sample math needs Δ ≳ 1.96·√2·0.048 ≈ **0.13** (⇒ v7 ≳ 0.70); the paired test
  cuts this, plausibly to Δ ≈ 0.07–0.09 (⇒ **v7 ≳ 0.65–0.68**).
- **So: a v7 result of 0.60–0.64 beats chance but is statistically indistinguishable from
  v6's 0.571.** It would *not* be evidence of improvement. Say this **before** the run, not
  after: the credible-win bar is **AUC ≈ 0.68** on the census, via the paired DeLong test.

**Mandelbrot non-regression (70% of corpus, currently works).** The frozen v6 eval carries
**942 mandelbrot loc / 29 q3** forward unchanged. Compute v7's AUC/AP on that identical
subset and require **paired non-inferiority to v6** (same 942 locations, so DeLong-paired,
no re-labeling). Also watch J0 julia (178 loc / 25 q3) the same way. This is the guard that
a julia-targeted retrain didn't quietly cost mandelbrot rank.

---

## 5. Is this enough data? — honest read

**Do the retrain, but pre-register tempered expectations.**

- **Train side is a genuine, cheap bump:** julia:mb positives 13 → 95 (7.3×). v6 literally
  had 1/8/4 positives for jm3/4/5 — below any threshold for learning a class. 95 (≈32/family)
  crosses from "cannot learn" to "can begin to." The data already exists; the retrain is one
  build. Downside is bounded: v6 is the frozen rollback.
- **Eval side is the binding constraint.** SE ≈ 0.048 means a *real* but *modest* v7 gain
  (say true AUC 0.63) **cannot be certified** at n=144 — we could train an improvement and be
  unable to prove it. 95 positives is still small for an aesthetic-ranking task with wide
  within-family variance.
- **Verdict:** retrain now if the goal is a **better deployed gate** (worth it even if
  unprovable). If the goal is a **measurable** win, the actual bottleneck is **more labels**,
  not the model: label **native R** (stratified 300, per the census readout) **and more julia
  census** to (a) grow the positive class past ~95 and (b) build an eval large enough to
  resolve a 0.62-vs-0.57 difference. Recommended sequencing: retrain v7 now (monotone-safe),
  and in parallel start native-R + julia labeling so v8 has both more train signal and a
  powered eval. **A null/ambiguous census result (0.55–0.65) means "label more," not "v7
  failed" — decide that now.**

---

## 6. Known gap (state, don't solve): native multibrot is unmeasurable

Native mb3/4/5 receive **zero** new positives (all 9 already in v6) and have **no labeled
eval** — their 517 R crops are unlabeled. Therefore **v7's native-multibrot performance is as
unmeasurable as v6's**, and the census certifies **nothing** outside julia:mb3/4/5. This is a
direct consequence of the plan (census is julia-only, native R deferred). The decision to
retrain is made with this eyes-open: v7 may leave native multibrot exactly where v6 left it,
and we won't know either way until native R is labeled (§5, §census-readout §6).

---

## Out of scope

Re-deriving `t_good` from the census ROC comes **after** a v7 model exists (and the census is
now the *eval* set, so it cannot also be used to pick the operating threshold without leakage
— a separate held-out or a nested procedure is needed then). Not folded in here.
