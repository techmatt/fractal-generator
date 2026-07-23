# q4 c-perturbation — which exemplar criteria generalize across Julia c?

Measurement pass (no descent / config / `data/` changes). Follow-up to
[`q4_neighborhood_sweep.md`](q4_neighborhood_sweep.md), which proved the artist-quality
"corner" is a coherent, reachable **region** under pan/zoom *at the exemplar's c* — but at
one `c` it is a single motif family. This pass sweeps **across Julia `c`** to decide whether
any of it becomes a real search bias: do the exemplar-calibrated criteria generalize, does a
target-band corner exist at other `c`, does variety come from `c`, and is variant B reachable.

**Exemplar** — center-view Julia, `c=(0.26103, −0.48932)`, origin-centered, fw≈0.66–0.75,
`twilight_shifted`. See `out/q4_cperturb/exemplar_large.png`.

## Method

- **c sampling** — exemplar + 3 rings around it (radii {0.03, 0.08, 0.16} × 6 angles = 18) +
  4 deliberately-farther c's near ∂M (`far_west_neck (-0.8,0.156)`,
  `far_upper_card (0.285,0.535)`, `far_upper_bulb (-0.4,0.6)`, `far_rabbit (-0.702,-0.384)`).
  **23 c's.**
- **Per c** — center-descent framing sweep (fw log-spaced [0.13, 1.5], 7 rungs × {center + 4
  small ±0.15·fw pans} = **35 framings/c**; center pan preserves the z→−z symmetry bonus).
  **805 field-dumps**, f64 escape-time, 768×432 ss1, colormap-invariant, purged per-unit.
  Measured in ~90s. Colored judge renders (1024×576 ss2) only for displayed framings.
- **Axes** — reuse the neighborhood sweep's calibrated two-scale detail bands
  (`interior_frac`, `deep_frac`, `detail_in_deep`, `flat_frac`, `mid_detail_frac`, `busy_frac`)
  + two NEW: **`busy_near_black`** (fine-scale detail in a 4px dilated ring around interior
  lakes — the distracting speckle that breaks strange modes; penalty) and **`coherent_rest`**
  (largest *connected* low-variance region as frame fraction — the composed smooth sweep,
  distinct from total `flat_frac`). Candidate composite **q4 = −band_dist − `busy_near_black`
  + `coherent_rest`** (band_dist to `mid_detail≈0.73, interior≈0.24, flat≈0.23`).
  Tool: `tools/studies/q4_c_perturbation.py` (`measure`/`analyze`/`morph`/`sheets`); reuses
  `q4_neighborhood_sweep`'s bands + `library_annotate.morph_gray_image` + `colored_clip`.
  Data: `out/q4_cperturb/{metrics.jsonl, analysis.json, morph.json}`.

## HEADLINE — the corner does NOT generalize; it is essentially c-UNIQUE

**A target-band corner exists at only 1 of 10 non-degenerate c's — the exemplar itself.**
The exemplar sits at min band_dist 0.37; the *nearest* other live c (`ring0_a0`, only **0.03**
away in c) has min band_dist **4.57**, and every other live c is 2.7–6.4. The
exemplar-calibrated criteria are **overfit to this one `c`** — they do not pick a same-looking
framing anywhere else.

The reason is not that the good-picking *value* drifts with `c` — it is that **no framing at
other `c` reaches the target band at all**. The exemplar's defining signature is the
**conjunction** of distributed mid-detail **and** composed interior lakes; across the whole
sweep that conjunction occurs only at the exemplar:

| live c | fw | interior | mid_detail | flat | look (see `sheet_best_per_c.png`) |
|---|---|---|---|---|---|
| **exemplar** | 0.66 | **0.25** | **0.69** | 0.27 | busy multi-scale composition **with** black lakes |
| ring0_a0 (Δc 0.03) | 1.50 | 0.35 | 0.26 | 0.70 | one big lake + thin filigree edge — collapsed |
| ring1_a0 / a4 / a5, ring2_a4 / a5, far_rabbit | 1.5 | **0.00** | 0.13–0.28 | 0.72–0.87 | sparse dendrite dust on flat void, no lakes |
| far_west_neck | 0.20 | 0.00 | **0.56** | 0.44 | dense colorful filigree, **zero lakes** |
| far_upper_bulb | 1.50 | 0.00 | 0.48 | 0.52 | dense snowflake filigree, **zero lakes** |

Every other `c` gives **either** interior lakes **or** busy mid-detail, **never both**. The two
farther c's that approach the exemplar's busy-ness (`far_west_neck`, `far_upper_bulb`) do so by
trading away the composed negative space entirely (interior_frac 0.00). This is the exact
`q4_axis_discovery` result — "the joint high-busy × high-interior corner is nearly empty (1 of
1087)" — now shown to be a property of **`c`, not framing**: you cannot pan/zoom into the
corner at a `c` that doesn't already have it.

Per-axis generalization verdict (composite-best axis value across the 10 live c's vs the
exemplar target): all three band axes read **c-SPECIFIC** (`mid_detail` mean 0.31 vs target
0.73; `flat` mean 0.68 vs 0.23) — but this is the corner-non-existence above, not a usable
per-c re-calibration. `generalization_drift.png`.

## The exemplar c sits on a KNIFE-EDGE of ∂M (why 13/23 c's are degenerate)

**13 of 23 sampled c's are degenerate** and were logged + excluded:
- **7 "solid interior"** (median interior_frac > 0.9) — `c` fell *inside* M, so the filled
  Julia set has non-empty interior and a center z-view sits entirely inside it: solid black at
  every depth ≤ fw 1.0, boundary only visible zoomed out to fw 1.5 (and even there mid_detail
  ≤ 0.09, a thin filament).
- **6 "dust"** (max mid_detail < 0.05) — `c` fell *outside* M: interior_frac 0.00 everywhere,
  a disconnected Cantor dust with no set to frame.

**Four of the six radius-0.03 ring c's are already degenerate** (a 0.03 step lands inside M or
in dust). The exemplar `c` straddles ∂M at a point where the boundary carries rich
lake-and-filament structure, and that straddle is fragile: most small perturbations fall off it.
This is itself the core obstacle to using the exemplar as a descent seed — its neighborhood in
`c` is mostly boring solid or dust, not more of the same look.

## Motif variety IS real — and comes from c (positive)

The 10 live per-c bests are **10 distinct looks**: morph_clip (grayscale robust-z / CLIP,
library recipe) median off-diagonal cosine **0.859** (right at the inter-location yardstick
0.851), max **0.973** — **just under** the 0.974 near-dup line, **zero** near-dup pairs
(`morph.json`, `morph_sim.npz`). So the "too similar" worry is answered: different `c` genuinely
gives different looks (busy-with-lakes / spiral-with-one-lake / sparse-dendrite-dust /
dense-lakeless-filigree). Variety must indeed come from `c` — but note the corollary above:
**variety across `c` ≠ the exemplar look at other `c`**. You get different looks, not the same
look re-seeded.

## Variant B — c-perturbation unlocks the deep_frac MAGNITUDE, but toward flat void (honest negative)

The neighborhood sweep capped `deep_frac` (slow-escape "almost-negative") at 0.085 and
predicted c-perturbation as the lever. **It is** — `deep_frac` reaches **0.183** at `ring2_a4`
and 0.086 at `ring2_a5`, both past 0.085. **But `detail_in_deep = 0.000` at every one of those
framings** (flat_frac 1.00, mid 0.00): the large slow-escape region is a **structureless dark
blob**, the exact opposite of variant B's "textured deep, not flat." So the magnitude axis is
now steerable, but it steers into **flat darkness, not composed textured negative space**.
**Literal variant B (large slow-escape basins *with* texture) remains unreachable across all 23
c's.**

## The two new axes — measured, neither rescues generalization

- **`busy_near_black`** stayed low across all live c's (max 0.095, mean 0.026): the
  distracting near-lake speckle failure is **not** reached even under c-perturbation in this
  region (consistent with the neighborhood sweep's `busy_frac` being unreachable). Useful as a
  penalty *if* a c-region produced it; here it never fires. Flag sheet
  `sheet_busy_near_black.png` (the "worst" offenders are mild).
- **`coherent_rest`** cleanly separates composed-sweep from fragmented flat, but on this
  cross-c set it mostly just tracks the sparse/empty far framings (high `coherent_rest` = big
  empty region). It is a real discriminator *within* a structured c; it cannot manufacture a
  corner where none exists.

## Verdict — for the descent-bias question

**The exemplar criteria are not a usable cross-c descent bias.** The composite ranks correctly
*within* the exemplar c (its top pick is exactly the `sheet_A` look), but cross-c it can only
pick each c's least-bad framing, and the target band physically exists at only one `c`. The
generalizing lever is therefore **not** a framing score to carry between c's — it is a
**c-selection** problem: find c on the ∂M knife-edge that (a) straddles the boundary so interior
lakes survive **and** (b) carries distributed mid-detail. That conjunction is what's rare, it
lives in `c`, and the pan/zoom framing score is downstream of it. A search bias that works would
screen *candidate c's* for the boundary-straddle-with-fill property first, then apply the
within-c framing composite (which does work) second.

**Scope caveat.** fw was bounded to [0.13, 1.5] (the band where the exemplar look held). The
neighborhood sweep found the exemplar look *intensifies* zooming below fw 0.13; some ring/far c's
might recover richness deeper than tested here. This pass answers generalization **within the
exemplar's own fw band** — where it clearly fails.

### Artifacts (`out/q4_cperturb/`)
`sheet_best_per_c.png` (primary — the 10 distinct per-c bests), `sheet_by_c_rings.png` /
`sheet_by_c_far.png` (per-c top-4 grouped by c), `sheet_busy_near_black.png` (flag),
`exemplar_large.png` (reference), `generalization_drift.png`, `analysis.json`, `morph.json`,
`morph_sim.npz`, `metrics.jsonl` (805 framings). Regenerate:
`uv run python -m tools.studies.q4_c_perturbation {measure,analyze,morph,sheets}`.
