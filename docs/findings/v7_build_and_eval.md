# v7 — build + train + eval (julia:multibrot retrain)

**Date:** 2026-07-17 · **Status:** v7 trained and measured. **`ACTIVE_CKPT` NOT switched
(v6 stays deployed); `t_good` NOT set.** Plan: `v7_retrain_scope.md`; gate stop + Option A:
`v7_build_gate_stop.md`; build metadata: `data/v7/build_metadata.json`.

**One-line result:** v7 is a **credible win** on the julia:multibrot census — AUC(q3 vs rest)
**0.705 [0.618, 0.789]** vs v6's **0.577 [0.483, 0.668]**, paired DeLong **Δ +0.128,
p = 0.001** — clears the pre-registered 0.68 bar — and it **does not regress** mandelbrot
(70 % of the corpus) or J0 julia.

---

## 1. Pre-build gate (resolved — Option A)

The gate asked whether v6's frozen eval already contained julia:multibrot. It **did**: 17
UNBIASED julia:mb eval locations (1 q3), because gather_v6's split sent unbiased
`random_eval` draws to eval (the "all gather_v6 julia → train" premise was false). Per the
gate, this was reported (`v7_build_gate_stop.md`) and resolved under **Option A** (user
choice): keep the frozen prefix byte-identical, accept the 17 as a pre-existing remnant, and
**report the julia:mb metric on the census-144 slice only** (never the 161-union). Assert #4
was reworded to what it actually enforces — *no biased location in eval* (holds).

## 2. Build (all 7 gates passed — aborts, not warnings)

`data/v7/manifest.jsonl` = **5261 v6 rows byte-for-byte + 536 appended** = 5797 rows,
**4501 train / 1296 eval**. Appended splits (forced by batch):

| bucket | n | q3 | split | biased |
|---|--:|--:|---|---|
| prospect census (julia:mb) | 144 | 67 | **eval** | no |
| jm3+jm45 band | 125 | 82 | train | yes |
| blindspot v6-reject (mandelbrot) | 219 | 4 | train | yes |
| prospect native-plane (multibrot3/4/5) | 22 | 6 | train | yes |
| loose0_v3 re-labels (mandelbrot) | 26 | 0 | train | no |

Gates: 0 orphans · 0 identity straddle · 0 group straddle · 0 biased-in-eval · forced
assignments hold (census 144→eval, band 125→train) · **frozen-prefix byte gate** (5261 rows
+ 220 962 cache rows byte-identical to v6) · census disjoint from all train at
identity / seed-`c` / parent_oid (0/0/0).

**One recipe fix vs the plan:** the group union-find had to be partitioned by
`(fractal_type, split, c-bucket)`, not just `(family, c-bucket)` — with split forced by
batch, the c-bucket tolerance transitively chained 2 census(eval) `julia_multibrot4` locs to
band(train) locs, straddling the split. Adding split to the partition enforces gate 3;
splits themselves are unchanged.

**Amendments & corrections (in `build_metadata.json`):**
- **Amdt 1 — no ss2 aug slot.** Confined by the byte gate to the appended block (≈all
  julia:mb + blindspot negatives), an ss2 slot would correlate with family+label. Dropped;
  the ss2 AA gap is a **knowingly-accepted second-order covariate shift** (deploy is 640×360
  ss2; aug is ss1+ss4, palette+geometry match).
- **Amdt 2 — census is the eval.** When `t_good` is later fit it will be fit **on the eval**
  (one scalar on 144 pts) — a small, recorded leak. `t_good` is a separate later step.
- **Plan §1 table corrections:** native multibrot post-freeze is **22 locs / 6 q3** (not 0;
  §6's "zero new positives" is stale — native train positives go 9→15, but native stays
  **unmeasurable**, no eval); 26 loose0_v3 mandelbrot re-labels were unenumerated. All → train.
- **julia:mb train positives = 94** (12 v6-train + 82 band), not the plan's 95: the 1 v6
  julia:mb q3 sits in the frozen eval under Option A, not train.

## 3. Train

`classifier/train_v7.py` — v6 recipe **verbatim** (reuses the v4 loop; CORN ordinal,
mobilenetv4, 40 epochs, patience=40, sampler βbiased=0.4/sqrt), only the data changed
(+536 appended) and the compare is v7-vs-v6. Best **epoch 24/40**, val not-bad AP **0.7506**,
wall 2266 s.

> **Resilience note (not a recipe change):** this environment repeatedly reaped the training
> process at random points (epoch 6, ~0, ~0, 32) — no error, no OOM (GPU 1.2/8 GB), while a
> 55-min Rust render survived; cause = an external process-tree reaper we could not pin from
> inside the sandbox. `train_v7.train_resumable` adds **per-epoch checkpointing + exact
> resume** (model+optimizer+scheduler+RNG) so a kill costs ≤1 epoch. The completing run (5th
> launch) ran uninterrupted start→finish, so no resume was exercised in the final model.

## 4. Evaluation (`tools/v7/eval_delong.py`, paired DeLong on frozen scores)

**PRIMARY — census-144 (julia:multibrot, q3 vs rest), score = Σσ(logit):**

| model | AUC | 95 % CI (5000-boot) |
|---|--:|--|
| v6 | 0.577 | [0.483, 0.668] |
| **v7** | **0.705** | **[0.618, 0.789]** |

Paired DeLong (same 144, both re-scored): **Δ +0.128, z 3.29, p = 0.001.**
v6's 0.577 reproduces the census readout's 0.571 → baseline sound.

**Pre-registered bucket rule → `CREDIBLE WIN (≥0.68)`.** 0.705 clears the 0.68 bar and the
paired p = 0.001 confirms v7 > v6. Stated as the rule requires, not argued upward: **the
point estimate's CI is wide (n=144 is still underpowered)** — the *win* is significant
(paired test), but the *magnitude* [0.618, 0.789] is uncertain. A 0.55–0.65 result would
have meant "label more"; this one does not.

**NON-REGRESSION (paired DeLong, same frozen v6-eval locations):**

| family | n | q3 | v6 AUC | v7 AUC | Δ | p | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| mandelbrot | 942 | 29 | 0.924 | 0.904 | −0.020 | 0.345 | non-inferior |
| J0 julia | 178 | 25 | 0.847 | 0.920 | +0.073 | 0.049 | non-inferior (improved) |

Mandelbrot's −0.020 is not significant (p 0.345) — no regression on the 70 %-of-corpus class.
J0 julia improved (marginally significant). Per-family within the census (jm3/4/5) all move
the same way: good-AP v7/v6 = 0.55/0.39, 0.70/0.57, 0.70/0.68.

## 5. What is NOT certified

- **Native multibrot** (mb3/4/5): still **no labeled eval** — v7's native performance is as
  unmeasurable as v6's, despite +6 native train positives. Unchanged from plan §6.
- **The census is now the eval**, and it is the only unbiased-given-descent julia draw that
  exists (run-1 ledger exhausted). Setting `t_good` later fits a threshold **on the eval** —
  do it with a held-out/nested procedure, and know it's fitting not discovering (Amdt 2).
- **ss2 AA covariate gap** (Amdt 1) remains, knowingly.

## 6. Next steps (not done here)

1. Decide whether to promote v7 to `ACTIVE_CKPT` (win is real; mandelbrot safe).
2. Derive `t_good` per the leak-aware procedure above (separate step).
3. To *power* the eval (tighten the wide CI) and grow native measurability: label native-R
   (stratified) + more julia census → v8.

Artifacts: `data/classifier/v7/{model_best.pt, metrics.json, eval_scores_v7.jsonl,
eval_delong.json, config.json}`; builders `tools/v7/{build_manifest,build_plan,eval_delong}.py`,
`classifier/train_v7.py`.
