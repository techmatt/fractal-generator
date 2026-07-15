# Julia recall — step 1: raw discovery pool + v6 distribution

**Goal.** Characterize the *raw* yield of Julia discovery — the honest distribution
before any q3-selection — to size/stratify a labeled batch next. Julia was underpowered;
its only labeled data came from the pre-filtered gather pool.

## What was run

Live Julia descent path (`guided-descend --julia`, production config verbatim —
`run_julia_descent` mirror: depth 4–14, node-width 384, σ-band 8..16, per-walk-rng, 3
walks/descent, engine's own `julia_band_defaults`). c's sourced from the **parent
discovery clouds** (the live julia-hook's actual c-distribution): 600 evenly-sampled
parent c's per family × 3 walks. Took **everything the descent emits** (deepest frame per
walk = the walk's discovery outcome; NO q3 post-select), rendered each at the label
geometry (1280×720 ss1, twilight_shifted, auto-maxiter), scored with **v6** in-memory
(`score_lib.Scorer`, crops transient — never persisted). 1200 descents, **0 bad / 0
timeouts**, 65 min, 4 workers.

- **Ledger (durable, append):** `data/discovery/julia_recall/pool.jsonl` — 3600 rows.
  Per row: partition (`julia:{fam}`), c_re/c_im, cx/cy/fw (decimal strings), walk, depth,
  degenerate flag, p_notbad, p_good, score, t_good, q3 verdict. Resumable (c-dedup),
  hard-killed per descent (90s).
- Scripts: `julia_recall_pool.py` (generator), `analyze.py` (report).

## Raw v6 distribution

| partition | n | c's | degenerate (d≤1) | p_good p50 / p90 / p99 / max | **q3 base rate** |
|---|---|---|---|---|---|
| `julia:mandelbrot` (t=0.24) | 1800 | 600 | 177 (9.8%) | 0.005 / 0.027 / 0.358 / 0.916 | **1.33%** (24/1800) |
| `julia:multibrot3` (t=0.30) | 1800 | 600 | 85 (4.7%) | 0.007 / 0.028 / 0.147 / 0.901 | **0.44%** (8/1800) |

- Strict CORN decode (nb≥0.5 ∧ pg≥t) == pg≥t here: every pg≥t location also cleared nb≥0.5.
- Non-degenerate only: julia:mandelbrot **1.48%** (24/1623), julia:multibrot3 **0.47%**
  (8/1715). Degenerate depth≤1 outcomes (base-scale whole-set z-plane views) are ~all
  rejects but only nudge the rate.
- **c-level yield** (≥1 q3 walk per c): julia:mandelbrot **19/600 c's (3.2%)**,
  julia:multibrot3 **6/600 c's (1.0%)**. q3 is concentrated on few c's, not spread thin
  across all — a q3-bearing c tends to yield >1 of its 3 walks.

### p_good histogram (both families: mass jammed at ~0, thin good tail)
```
                 julia:mandelbrot        julia:multibrot3
 [0.00,0.05)     1699                    1723
 [0.05,0.15)       65                      61
 [0.15,0.24)       12                       6
 [0.24,0.30)        4                       2
 [0.30,0.50)        7                       5
 [0.50,1.00)       13                       3
```
Genuine q3 Julia locations DO exist (max p_good 0.916 / 0.901; a handful in [0.7,1.0]) —
just rare.

## vs mandelbrot's raw ~1%

Mandelbrot reference = the flat-draw unbiased pool (finder_step1 STEP 0b): raw v6-good
**~1%**, drawn **flat over the c-plane**.

- **julia:mandelbrot 1.33%** ≈ parity with mandelbrot's ~1% (marginally above).
- **julia:multibrot3 0.44%** ≈ half of mandelbrot's ~1%.

**Julia is not a hidden recall goldmine.** Deg-2 Julia only *matches* mandelbrot's flat
rate and deg-3 is *below* it — and that is with a **favorable c prior** (see honesty box),
so the numbers are an *upper bound*. Raw Julia yield is in the same order of magnitude as
mandelbrot, not better.

## Honesty boundary (read before comparing rates)

The two rates are **not the same sampling**. The mandelbrot ~1% is flat over the c-plane
(unbiased parameter draw). This Julia pool draws c from **qualifying parent outcomes**
(biased toward structure-bearing c), then samples the z-plane unbiasedly given c. So:
- Within-Julia, per-family rates ARE comparable (identical c-source policy + descent).
- vs mandelbrot flat: Julia's rate is an **upper bound** — a flat-c Julia draw would be
  lower (most c give dust/empty sets). The operationally honest read: even c-advantaged,
  Julia deg-2 = mandelbrot flat, deg-3 < it.
- These parent-conditioned rates ARE the right numbers for sizing the **live julia-hook**
  batch (that hook only ever fires on qualifying parents), which is the next step's target.

## For the next step (labeled batch, stratified incl. rejects)

Strata already in the ledger (p_good bands), per family:

| band | julia:mandelbrot | julia:multibrot3 |
|---|---|---|
| reject <0.05 | 1699 | 1723 |
| 0.05–0.15 | 65 | 61 |
| 0.15–0.24 | 12 | 6 |
| q3 ≥ t | 24 | 8 |

Plenty of reject/marginal mass; **q3 exemplars are the bottleneck** (24 / 8). A stratified
label batch should **oversample the high-p_good tail** (all q3 + all [0.15,0.24) marginals)
and subsample the reject bulk, so the labeler sees enough good/marginal Julia to validate
(or move) the 0.24 / 0.30 decode thresholds out-of-sample.

## Status

- Pool + scores persisted: `data/discovery/julia_recall/pool.jsonl` (3600 rows, durable).
- No label crops rendered (per prompt — that's the next batch).
- Higher-degree twins (multibrot4/5) NOT run — untuned + not cheap (~same descent cost);
  can be added as a smaller sample if the batch wants cross-degree coverage.
