# q4 stage-1 first fit — linear (L1) accept-vs-reject goodness field

Labels: p1+p2 union (p2 precedence), `filter_leak` excluded. **228** windows (55 accept / 173 reject) over 30 minibrots. Referee: leave-one-minibrot-out, pooled held-out ranking (never train+test the same minibrot).

## Held-out referee (minibrot-disjoint LOMO)

| tier | C | AUC | AP | accept-recall@G0 | reject-precision@G0 |
|---|---|---|---|---|---|
| T1_global | 0.05 | 0.848 | 0.607 | 0.93 | 0.43 |
| T2_cells | 2.0 | 0.878 | 0.707 | 0.80 | 0.54 |
| T3_laplacian | 0.5 | 0.873 | 0.669 | 0.84 | 0.53 |

**Cells earn their place?** ΔAUC(T2−T1) = +0.030; ΔAUC(T3−T2 Laplacian) = -0.005. Chosen tier (max held-out AUC): **T2_cells**.

## Reading

- **Global-only already ranks well** (AUC 0.848) on a *single* surviving scalar, `g_mid` (mid-detail fraction) — the rough heuristic is essentially "how much of the window is mid-scale ornament." A cell-free `g_mid` threshold is a usable labeling aid on its own.
- **Cell-dispersion earns a modest, real lift** (+0.030 AUC, AP 0.607→0.707). What the cells add is *contrast* structure: `flat_worst`(+) ∧ `detail_worst`(+) ∧ `detail_spread`(−) = a window with **both** a calm anchor cell **and** evenly-distributed detail elsewhere (not one busy spike). `interior_worst`(−) kills any window with a dead cell — the corner-deadness signal.
- **Laplacian does NOT earn its place** (-0.005 AUC vs T2; `lapvar_*` weights are tiny). 2nd-order curvature adds nothing over the struct_e decomposition. Drop it.
- **Priors that did NOT hold** (weights contradict the framing hypotheses): (a) `g_occ` carries a *positive* weight (+0.08) — the "down-weight occupancy" prior was a story; more occupancy reads as accept. (b) `flat_edge_minus_center`(+0.37) has the **opposite** sign to the "flat-in-edge = empty corner = bad" prior: a calmer edge with a busier center (subject-centered composition) reads as accept, not reject. (c) `g_speckle`(+1.31) is positive — *within pre-filter survivors* (pure speckle already gated at ratio≥0.30) a higher fine/coarse ratio is fine ornamentation, not noise.
- **Field visual test passes**: masking the field to pre-filter survivors (the deployed v2 gate) is load-bearing — the *unmasked* linear G extrapolates to huge OOD spikes on the dead-interior blob (the model never trains on interior-heavy windows). Over survivors, G∈[-11,+6] and its position-maxima land on the ornate spiral ring; the rendered maxima crops are exactly the good filigree windows. "Plot G, take maxima" auto-frames correctly.

## Surviving L1 weights (standardized, sorted by |w|)

**T1_global** — 1/6 survive:

| feature | weight |
|---|---|
| g_mid | +0.847 |

**T2_cells** — 11/15 survive:

| feature | weight |
|---|---|
| flat_worst | +1.922 |
| detail_spread | -1.765 |
| interior_worst | -1.660 |
| g_speckle | +1.306 |
| detail_worst | +1.247 |
| g_mid | +0.521 |
| speckle_spread | -0.506 |
| flat_edge_minus_center | +0.371 |
| g_flat | -0.126 |
| g_occ | +0.076 |
| speckle_worst | -0.062 |

**T3_laplacian** — 10/18 survive:

| feature | weight |
|---|---|
| detail_spread | -1.522 |
| g_speckle | +1.136 |
| g_occ | +0.501 |
| flat_worst | +0.481 |
| flat_edge_minus_center | +0.272 |
| speckle_worst | -0.232 |
| lapvar_spread | -0.156 |
| interior_worst | -0.141 |
| speckle_spread | -0.097 |
| g_mid | +0.002 |

## Next-to-label

`out/q4_stage1/linear_fit/next_to_label.json` — margin-uncertain unlabeled survivors (where labels teach the boundary) + a uniform-random control slug (audits confident-and-wrong outside what the model knows). Crops already exist in the store (`data\q4_window_corpus\batches\2026-07-23_q4_stage1_windows\crops`).

## Goodness field

`out/q4_stage1/linear_fit/field_<mb>.png` — G over a dense position×scale grid (G computed directly, score_A NMS bypassed), position-maxima marked, their windows rendered. The visual test of *plot G, take maxima*.
