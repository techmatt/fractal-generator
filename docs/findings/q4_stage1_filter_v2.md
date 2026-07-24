# q4 stage-1 coarse filter v2 — as-framed calibration

Tightened the stage-1 coarse pre-filter against Matt's **107 labels**
(`labels/q4_stage1_windows.json`: 23 accept / 23 reject / 61 filter_leak — a **57%
leak rate**, the problem this pass fixes). Judgment model is **as-framed**: a window
is labeled as the finished frame shown; a good-content-but-badly-framed window (dead
corner eating part of the frame) is a **reject**, because the dense sweep already
presents the well-framed recentered crop as its own separate candidate. So the
ceilings are **plain** — no corner-sparing, no decoration AND-condition.

Tools: `tools/studies/q4_stage1_filter_v2.py` (`analyze` calibrates on the 107,
`apply` drops the 193 unlabeled → `auto_filter_v2`); `tools/studies/q4_stage1_nms_check.py`
(NMS density verification).

## v2 ceilings (auto-drop → `auto_filter_v2`, NOT human labels)

| ceiling | rule | catches (of 61 leak) | note |
|---|---|---|---|
| too-large interior % | `interior_frac >= 0.10` | 29 | accepts/rejects are interior-free (acc max 0.001, rej max 0.009); 0.10 = "eats ≥10% of frame", clears the envelope by 0.09 |
| too-barren | `flat_frac >= 0.88` | 7 net | sits in the `[0.871, 0.884]` gap — spares the lone calm accept at flat=0.871, catches the leak cluster 0.884–0.912. **Plain** ceiling |
| speckle / high-freq | `speckle_ratio >= 0.30 & hf_mean >= 0.012` | 1 unique | **fixed metric** (see below) — accept-max 0.285, catches the mb04 wall (0.327) and pure-speckle leaks up to 1.066, sparing coherent ornate detail |

The accept-guardrail is the safety net: it auto-protects real composed-calm windows
regardless of how "barren"/speckly is measured, so the metrics stay simple.

### Speckle metric fix (Bug 2)

The stored `busy_frac = (fine > 0.30) & (struct_e < 0.05)` was **broken** — it read
the obvious mb04 speckle wall as `0.001` and had caught **0** ("unreachable"). Cause:
`fine = |work − 3×3 lowpass|` with a `0.30` threshold never fires — a dense-banding
wall is a *locally-smooth steep gradient* (residual-from-local-mean ≈ 0), and the
0.5–99.5 percentile stretch compresses its amplitude further. Replaced with a real
fine-scale **frequency** measure computed from the field (`tools/studies/
q4_stage1_filter_v2.py::_field_speckle`, cached to `speckle_scores.json`):

```
speckle_ratio = hf_mean / coarse_std
  hf_mean    = mean |DoG(σ=0.8, 1.8)|   pixel-scale oscillation energy
  coarse_std = std of a σ=6 lowpass      large-FORM variation
```

The discriminator is **high fine energy WITHOUT coarse structure**: coherent ornate
detail (spirals, dendrites) keeps strong coarse-form variation → low ratio (the
top-scoring accept, a symmetric dendrite, is 0.285); granular static has fine energy
over a near-uniform coarse field → high ratio (mb04 wall 0.327, pure-speckle leaks
to 1.066). Per class: accept max 0.285, reject max 0.278, filter_leak to 1.066. The
`0.30` ceiling clears the accept envelope by 0.015 and catches the mb04 wall by
frequency; `hf_mean ≥ 0.012` guards near-flat windows from a spurious tiny/tiny ratio.

## Agreement on the 107

- **filter_leak caught: 37 / 61 (61%)** — interior_heavy 29, barren 7, speckle 1.
- **reject caught: 0 / 23** — Matt's rejects are clean, well-formed windows (not
  q4-worthy but not degenerate); correctly not catchable by degeneracy ceilings.
- **accept dropped: 0 / 23 — guardrail HOLDS.**
- Speckle ceiling **uniquely** catches 1 labeled leak (`mb06_p15_s060_dc725810`,
  interior 0, flat 0.31) that interior/flat miss — a pure speckle wall.
- Residual leak rate on labeled survivors: **34%** (24 leak / 70 survivors), down
  from 57% pre-v2.

The residual 34% does **not** approach zero, and cannot under the 0-accept
guardrail: the remaining leaks sit squarely inside the accept feature-region
(mid decoration, ~0 interior, sub-0.88 flat, sub-0.30 ratio) — metrically
indistinguishable from accepts. Under as-framed this is acceptable: a leak that
survives is fine to leave in the queue **iff** its good recentered crop also
survives as a separate candidate (see NMS check — the load-bearing condition, which
partly fails).

## Apply to the batch (all 300 — Bug 1 fix)

`apply` now filters **all 300 windows**, not just the unlabeled ones. An
already-*labeled* degenerate (e.g. `mb04_p13_s060_c273d415`, interior 0.32, a
`filter_leak`) must also leave the label queue — the earlier unlabeled-only pass
never recorded it, so the UI still showed it. `auto_filter_v2.json` is now a
**queue-exclusion set** over all 300; human labels are untouched (they stay in
`scores.json`).

- `auto_filter_v2` dropped **89** (interior_heavy 59, barren 14, speckle 16) → the
  frequency ceiling removes **16** windows across the batch.
- **0 accepts dropped.**
- Survivors: **141 unlabeled** (the label queue) + 70 labeled = **211 shown**. The
  queue fell from 156 (pre-frequency-fix) to 141.
- `mb04_p13_s060_c273d415` is dropped and caught by **both** ceilings
  (interior 0.32 ≥ 0.10; speckle_ratio 0.327 ≥ 0.30).

### Bug 1 — UI wiring

`q4_window_label.html` loaded the full `windows.jsonl` and ignored the filter.
Fixed: the UI now fetches `auto_filter_v2.json` before the windows and excludes its
`dropped` ids from the queue and overview (header shows "N auto-filtered, M
survivors"). Preserves Matt's labels (localStorage / `scores.json` untouched).

## NMS density spot-check — the one condition, PARTLY FAILS

As-framed is lossless only if the good recentered crop of every dropped
dead-cornered window survives as its own candidate. Reconstructed the **pre-NMS**
candidate set and checked, for each dead-cornered window (interior ≥ 0.20), whether
a well-framed recentered neighbor (same scale, within 1 window-width, lower interior,
decoration ≥ 0.30) exists and whether NMS kept it:

- Good recentered crops **do exist** pre-NMS in **22%** of dead-corner neighborhoods
  → the 0.30-stride sweep is dense enough (where a good crop is geometrically
  possible, it's a candidate).
- But NMS (`NMS_IOU=0.35`, ranked by the occupancy-biased `score_A`) **suppresses
  every good recenter 38% of the time** — the busier dead-cornered window outscores
  its balanced neighbor and evicts it.

Example (the image-4 pattern — dead corner, good crop up-right/low-barren):
`mb01_p07 s0.09 dead@(0.75,0.24) int=0.21 → good recenter (0.83,0.19) int=0.00
SUPPRESSED by NMS`.

**Flag (don't rebuild now):** the *production* sweep needs looser NMS so balanced
framings survive — either raise `NMS_IOU`, or (better) run NMS on a **framing-balanced
score** instead of the occupancy-biased `score_A` that structurally prefers the busy
neighbor. Stride is fine; NMS selection is the leak.

## UI guidance line (added to `tools/viz/q4_window_label.html`)

> Judge the frame **AS SHOWN — no cropping in your head.** accept = a good finished
> frame · reject = bad as-framed (too busy / too barren / dead-cornered / too much
> interior). If a better crop of this view exists, the sweep presents it as its own
> window — accept *that* one when it appears (no crop debt, no seed/descent axis).
