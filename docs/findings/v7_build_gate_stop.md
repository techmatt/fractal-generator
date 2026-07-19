# v7 build — GATE STOP: v6 frozen eval already contains julia:multibrot

**Date:** 2026-07-17 · **Status:** build **halted at the pre-build gate**. Nothing built,
trained, or written to `data/v7/`. `data/v6/manifest.jsonl` untouched.

The `v7-build-and-train` gate asks: *does v6's frozen eval contain any julia:multibrot
locations?* It predicts **zero** (from the standing rule "all gather_v6 julia → train")
and instructs: **zero ⇒ proceed; nonzero ⇒ stop and report.**

**Result: NONZERO — 17 julia:multibrot locations sit in v6's frozen eval split.** Stopping.

---

## The finding

| family | eval locs in v6 frozen prefix | labels (1/2/3) | q3 |
|---|---:|---|---:|
| julia_multibrot3 | 5 | 4 / 1 / 0 | 0 |
| julia_multibrot4 | 7 | 4 / 2 / 1 | 1 |
| julia_multibrot5 | 5 | 4 / 1 / 0 | 0 |
| **total** | **17** | **12 / 4 / 1** | **1** |

All 17: `biased=False`, `source=gather_v6`, loc_ids 5217–5246 (all in the gather_v6
portion, ≥ N_V5=4622). Verified against `data/v6/manifest.jsonl` (`split=="eval"` ∧
`fractal_type` starts `julia_multibrot`).

## Why the gate's premise is false

The premise "all gather_v6 julia → train (biased→train keeps eval unbiased)" misreads the
actual gather_v6 split. `tools/v6/build_manifest.py:split_gather` does: **biased groups →
train; then among the _unbiased_ groups, `EVAL_FRAC=0.40` → eval** (stratified, seed 0).
The `biased→train` rule only diverts *biased* locations — it never sends *unbiased* julia
to train. Julia:mb locations selected via `random_eval` (the unbiased role) were therefore
eval-eligible, and 40% of them (17) legitimately landed in eval. v6 has been marginally
evaluable on julia:mb all along — just far too underpowered (17 loc / 1 q3) to notice.

**Refinement vs the gate's guess:** the gate expected "a second *biased* julia eval
population." It is the opposite — the 17 are all *unbiased*. So the literal v6 assert
`0 biased-in-eval` still holds. What fails is the plan's *premise* that the census is the
**only / complete** unbiased julia:mb eval.

## What is and isn't broken

- **Primary metric is SAFE.** The headline (AUC q3-vs-rest on the 144-census, paired
  DeLong vs v6) is computed on the 144 census locations only. The census is **disjoint**
  from the 17 — verified 0 overlap at both seed-`c` (9 dp) and full identity
  (fractal_type, cx, cy, fw, c_re, c_im). The 144-census distinct full-id count is exactly
  144, matching the plan. So the pre-registered instrument is intact.
- **`v7_retrain_scope.md` §2 assert #4 "(census only)" is contradicted.** The appended
  census is not the only unbiased julia:mb eval — the frozen prefix carries 17 more. The
  literal "0 biased-in-eval" passes; the *intent* ("census only") does not.
- **§2 / §5 claim "the census's entire value is being the complete unbiased-given-descent
  julia:mb eval" is false** — it is short by 17 (1 q3).
- **Per-family julia:mb eval battery would mix draws.** v7's eval split = frozen v6 eval
  (incl. these 17) + 144 census = **161** julia:mb eval locations. A train-time
  `julia_multibrot* eval` block computed the v6/train_v6 way would report on 161, blending
  gather_v6 `random_eval` with prospect_run1 baserate — not the clean 144.

## Rework options (author's call — not decided here)

- **A — keep frozen prefix byte-identical; report julia:mb on the census slice only.**
  Accept the 17 as a pre-existing, unbiased, negligible eval remnant. Rewrite assert #4 to
  what it actually enforces (no *biased* location in eval — holds) and add an explicit
  reporting rule: the julia:mb metric is sliced to the census batch identity (144), never
  the 161-union. Preserves the §2.6 byte gate and the v5↔v6↔v7 comparability chain. The
  primary metric is already isolated to the census, so this is the cheapest fix and nothing
  in the build changes except the eval-reporting slice. **Recommended.**
- **B — move the 17 out of eval.** Violates the §2.6 frozen-prefix byte gate (changes
  manifest bytes) and the eval-comparability chain. Rejected by the plan's own design
  unless the gate is redefined.
- **C — fold the 17 into a 161-loc julia:mb eval, re-baseline v6 on 161.** Larger n, but
  mixes two different unbiased draws and abandons the pre-registered 144-census instrument
  (v6 AUC 0.571). Deviates from like-for-like.

**Recommendation: Option A.** It keeps every frozen-prefix guarantee, the byte gate, and
the pre-registered census instrument, and requires only a reporting-slice clause — the
headline number is unaffected either way.
