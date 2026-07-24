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
| too-barren | `flat_frac >= 0.88` | 10 (7 net after interior overlap) | sits in the `[0.871, 0.884]` gap — spares the lone calm accept at flat=0.871, catches the leak cluster 0.884–0.912. **Plain** ceiling |
| speckle / high-freq | `busy_frac >= 0.10 & mean_struct < 0.16` | 0 | **INACTIVE** on this corpus — true speckle is unreachable here (busy_frac accept-max 0.007, leak-max 0.008). Kept as a forward guard |

The accept-guardrail is the safety net: it auto-protects real composed-calm windows
regardless of how "barren" is measured, so the metric stays deliberately simple.

## Agreement on the 107

- **filter_leak caught: 36 / 61 (59%)** — interior_heavy 29, barren 7.
- **reject caught: 0 / 23** — Matt's rejects are clean, well-formed windows (not
  q4-worthy but not degenerate); correctly not catchable by degeneracy ceilings.
- **accept dropped: 0 / 23 — guardrail HOLDS.**
- Residual leak rate on labeled survivors: **35%** (25 leak / 71 survivors), down
  from 57% pre-v2.

The residual 35% does **not** approach zero, and cannot under the 0-accept
guardrail: the remaining 25 leaks sit squarely inside the accept feature-region
(mid decoration, ~0 interior, sub-0.88 flat) — metrically indistinguishable from
accepts. Under as-framed this is acceptable: a leak that survives is fine to leave
in the queue **iff** its good recentered crop also survives as a separate candidate
(see NMS check — this is the load-bearing condition, and it partly fails).

## Apply to the 193 unlabeled

- `auto_filter_v2` dropped **37** (interior_heavy 30, barren 7) → **156 survivors**.
- Labeled + survivors = **263** (well above the ~150 floor; no top-up needed).
- Written to `data/q4_window_corpus/batches/2026-07-23_q4_stage1_windows/auto_filter_v2.json`.

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
