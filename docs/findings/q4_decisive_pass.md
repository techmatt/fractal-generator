# q4 DECISIVE — the exemplar conjunction is a structural class (~2%), not a one-off

Measurement pass (no descent / config / `data/` changes). Settles the last open q4 question
by **sample size**. Follow-up to [`q4_dM_property.md`](q4_dM_property.md), whose "0/10
non-exemplar corners" was **underpowered**: the organic base rate is ~1/1000, so 10 draws
cannot separate a real ~1% class from a true one-off. This pass deep-sweeps **534 viable
near-∂M c's** and counts band hits against a pre-registered read.

**Exemplar** — center-view Julia, `c=(0.26103, −0.48932)`, origin-centered,
`twilight_shifted`. Conjunction = interior lakes + busy filament detail + composed rest.
Tool: `tools/studies/q4_decisive_pass.py` (`pool`/`screen`/`measure`/`analyze`/`morph`/`sheets`).

## HEADLINE — CONFIRMED structural class. h=12 / n=534 → 2.25% (Wilson 95% CI [1.29%, 3.89%])

The pre-registered payoff branch (`h ≥ 3` → structural class, not a one-off) fires with room
to spare: **12 non-exemplar viable c's independently reach the exemplar band** (`band_dist ≤
1.5`, the same threshold as the prior passes → directly comparable). The 95% CI **excludes the
"one-off" regime** (its lower bound 1.29% is ~13× the ~0.1% organic rate; the h≤1 bank-branch
would have required an upper bound ≲1%). The conjunction is a **low-frequency-but-real,
targetable class among near-∂M c's**, not a pathological singleton. The prior "c-unique"
verdict was a **sample-size artifact** — 33 hand-picked ladders simply couldn't hit a 2% class.

The verdict is **not threshold-fragile** (sensitivity curve, monotone, no cliff at 1.5):

| band_dist thr | 0.75 | 1.0 | 1.25 | **1.5** | 1.75 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| hits | 8 | 8 | 10 | **12** | 17 | 23 | 42 | 92 |

Even at the strict thr 0.75, **8 c's clear it** — the class is robustly populated, not a pile-up
just under 1.5. `decisive.png` (sensitivity + ECDF + `|dist_dM|` vs J-quality); the 12 hits are
visually verified in `sheet_hits_and_ridge.png` (all 12 show distributed multi-scale busyness
**with** composed black lakes **and** rest — the full conjunction).

## The exemplar is NOT the corner — it sits mid-ridge; several c's beat it

The best non-exemplar, **b0703, reaches `band_dist 0.10` — 4× closer to the band than the
exemplar's 0.42** (`b0453` 0.12, `b0023` 0.20 also beat it). The exemplar-to-best gap is
**negative (−0.32)**: this is a **graded ridge, not a cliff to a special point.** 8 c's land
within 2× the exemplar's band_dist. The exemplar was a *representative sample of its class,
never its apex* — which is exactly why the class is targetable rather than a lucky coordinate.

## Characterization set — the determining property the two scalars missed: the interior-lake channel

The prior pass's "runner-up puzzle" (near-∂M-and-rich `c` that still lands 6× off the band) is
resolved by the n>1 the hit set provides. Group means at each c's band-nearest framing:

| group | n | `\|dist_dM\|` | M_mid_detail | best_fw | **interior_frac** | mid_detail | flat |
|---|---|---|---|---|---|---|---|
| exemplar | 1 | 0.0005 | 0.082 | 0.858 | 0.258 | 0.683 | 0.263 |
| **hits** | 12 | **0.0004** | 0.103 | 1.015 | **0.204** | 0.728 | 0.241 |
| near-miss (1.5–2.5) | 30 | 0.0013 | 0.097 | 0.756 | **0.079** | 0.683 | 0.306 |
| all viable | 534 | 0.0083 | 0.092 | 0.851 | 0.108 | 0.378 | 0.614 |

Two properties separate **hits** from **near-miss** at *matched* mid-detail and richness:
1. **∂M knife-edge proximity** — hits sit at `|dist_dM| ≈ 0.0004` (3× tighter than near-miss's
   0.0013, 20× tighter than the viable-pool mean). Confirms the prior pass's within-viable
   `abs(dist_dM)` signal, now at n=534.
2. **The interior-lake fraction is the missing determinant** — hits carry `interior_frac 0.204`
   vs near-miss's **0.079**, at essentially identical `mid_detail` (0.728 vs 0.683) and identical
   `|dist_dM|`-class. **The near-misses are busy-without-composed-lakes**; being near-∂M-and-busy
   is necessary but *the conjunction additionally requires the interior (composed-black-lake)
   channel to fire.* That channel — `interior_frac`, valid on every pixel and cheap on the julia
   field — is the property scalar Mandelbrot proximity/richness could not see, and it is exactly
   what the two-axis screen was blind to.

Hits also favor **wide framings** (`best_fw ~1.0–1.4`, vs the exemplar's 0.858): the composed
whole-Julia view, not a mid-zoom, is where the class lands — 7 of 12 best at `fw 1.40` (the
grid's widest rung), hinting the sweet spot may sit even wider.

## Motif variety — same class, distinct instances (positive)

The exemplar + 12 hits morph to **13 distinct looks, ZERO near-dups** (morph_clip grayscale /
CLIP: median off-diag cos **0.892**, max **0.967** < the 0.974 near-dup line;
`morph.json`/`morph_sim.npz`). Median cos sits *above* the 0.851 inter-location yardstick —
they are recognizably **one motif family** (composed spiral/lake Julias) — yet every one is a
genuinely different image, not a coordinate duplicate. A real class with internal variety, the
ideal generation target.

## Reusable artifact — the campaign-3 near-∂M c-sampler

Built here and validated end-to-end, reusable regardless of the verdict:
1. **Boundary rejection sampler** (`pool`) — vectorized: draw c in-box, keep where membership
   is non-constant over `{c} ∪ ring(ε=0.02)`. Weights the kept set by ∂M **arc length**, so it
   spreads across the whole boundary (bulbs / seahorse / elephant / dendrites) for c-diversity,
   greedy-deduped to `MIN_SEP 0.006`. 5.9% raw near-∂M yield; 751 distinct c's in ~10s.
2. **Cheap viability screen** (`screen`) — one mid-fw (0.6) center Julia render per candidate →
   blob (`interior>0.85`) / dust (`mid<0.04 ∧ occ<0.06`) / viable. **535/751 viable (71%)**;
   213 blob, 3 dust. The near-∂M shell is mostly viable — the ε=0.02 sampler already does most of
   the degenerate-rejection the prior ladder needed a full sweep for.
3. **Rank survivors** — within viable, minimize `abs(dist_dM)` **and** require the
   `interior_frac` (lake) channel to fire; that pair is what the 2% class shares.

Verdict for sourcing: **exemplar-grade is a ~2% target, not a stumble.** A campaign-3 run of the
sampler → screen → interior-lake-gated ranking should surface exemplar-class Julias at ~1-in-50
of viable near-∂M c's — orders of magnitude above the ~1/1000 organic rate, and the class
carries real motif variety.

## Ops / achieved-vs-planned

Planned ~300 viable; **achieved n = 534** (all viable deep-swept, no time-gate cutoff).
`measure`: 535 viable × 20 framings (`fw∈{0.03,0.15,0.40,0.858,1.40}` × 4 pans, band-calibrated
768×432 ss1) = 10,700 field-dumps in **1167 s** (1.6 s/c median), inside the ~30-min window.
Per-c checkpoint + idempotent resume; per-render 30 s hard-kill backstop; fields purged
per-unit. Exemplar carried as positive control (screened viable; band_dist 0.424 = HIT).

### Artifacts (`out/q4_decisive/`)
`sheet_hits_and_ridge.png` (primary — exemplar + 12 hits + ridge, annotated `band_dist`/`fw`/
`dist_dM`), `decisive.png` (sensitivity curve + band_dist ECDF + `|dist_dM|` vs J-quality),
`exemplar_large.png`, `analysis.json` (verdict + CI + branch + sensitivity + characterization +
ridge + per-c), `pool.json` (751 c's), `screen.jsonl`/`viable.json` (535 viable + ∂M props),
`metrics.jsonl` (10,700 framings), `morph.json`/`morph_sim.npz`. Regenerate:
`uv run python -m tools.studies.q4_decisive_pass {pool,screen,measure,analyze,morph,sheets}`.
