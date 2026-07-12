# Selector-side family-diversity policy — tradeoff curve

Follow-up to [palette-family-entropy-trace](palette-family-entropy-trace.md).
The trace localized the collapse to stage 3 (pref-v3-gvo's *within-location*
ranking), not generation. Fix lives in the emission **selector** (the portfolio
step), not the pref head. This adds a tunable palette-family dial there and
delivers its calibration curve. **Selector-policy only — no retrain, no re-render.**

## What was built

`emission_selector.select()` gains a keyword-only control
(`palette_family_of` + `palette_family_cap`): a hard cap of ≤`M` emitted renders
per **palette** family, mirroring the existing per-palette reuse cap one level up.
`palette_family_cap=None` (default) is OFF and **byte-identical** to prior
behavior (verified: identical pick lists). The MAP-Elites behavior axis
(fractal-family × Lab-color-cell), ≤1/loc, and palette-reuse cap are untouched.

- **Pool:** the trace's dramatic funnel `2026-07-09_wallpaper_headbatch_dramatic_v1`,
  952 topk candidates.
- **Fitness = pref-v3-gvo score** (from provenance) → cap OFF = pref-greedy
  selection. This is the correct fitness for a *pref-cost* study (the trace used
  head_v2 as fitness; here head_v2 is only the quality gate).
- **Gate = wallpaper head_v2 `p_ge3`** quality floor. Quality is held; the cap
  trades away pref preference, never the gate.
- **Family key = hybrid** (dramatic roster mood-family else hue/chroma bucket),
  identical to the trace, so spread numbers are directly comparable to its stage-4.
- Repro: `tools/wallpaper/selector_family_diversity_sweep.py`; picks JSON with crop
  paths in `scratchpad/family_diversity_picks.json`.

## Tradeoff curve

Swept cap ∈ {off, 5, 4, 3, 2, 1} at two quality floors. **Cost columns:** `mRank`
= mean within-location pref-rank of the emitted set (1.0 = always the pref-best;
higher = pref cost paid); `dScore` = mean pref-score delta vs. the unconstrained
(off) set (pref-score spans ≈4–17).

### Production gate — `p_ge3 > 0.90` (21 survivors; comparable to trace stage-4, N≈7)

| cap | picks | #fam | H | H_norm | top-1 | top-1% | top-3% | mRank | dScore |
|----|------|-----|-----|------|-------|-------|-------|------|-------|
| off | 7 | 5 | 2.236 | 0.963 | violet-purple | 0.286 | 0.714 | 3.43 | 0.0 |
| 5–2 | 7 | 5 | 2.236 | 0.963 | violet-purple | 0.286 | 0.714 | 3.43 | 0.0 |
| 1 | 5 | 5 | 2.322 | 1.00 | pastel-irid. | 0.20 | 0.60 | 3.20 | +0.57 |

At the production gate the dial barely engages: caps ≥2 never bind (max family
count is already 2/7), and only cap=1 changes anything (and it *shrinks* 7→5).

### Richer floor — `p_ge3 > 0.50` (116 survivors, 22-pick baseline) — the calibration curve

| cap | picks | #fam | H | H_norm | top-1% (violet) | top-3% | mRank | dScore |
|----|------|-----|-----|------|-------|-------|------|-------|
| **off** | 22 | 9 | 2.881 | 0.909 | 0.227 | 0.591 | 3.55 | 0.0 |
| 5 | 22 | 9 | 2.881 | 0.909 | 0.227 | 0.591 | 3.55 | 0.0 |
| 4 | 21 | 9 | 2.928 | 0.924 | 0.190 | 0.571 | 3.52 | +0.14 |
| 3 | 21 | 10 | 3.165 | 0.953 | 0.143 | 0.429 | 3.81 | −0.09 |
| **2** | 17 | 10 | 3.264 | 0.983 | 0.118 | 0.353 | 3.94 | −0.09 |
| **1** | 12 | 12 | 3.585 | 1.00 | 0.083 | 0.250 | 4.17 | −0.32 |

```
family spread (top-3 share, ↓=more diverse)     pref cost (mean rank, ↑=costlier)
 off  0.591 ██████████████████████               off  3.55 ██████████████
 4    0.571 █████████████████████                4    3.52 ██████████████
 3    0.429 ████████████████                     3    3.81 ███████████████
 2    0.353 █████████████                        2    3.94 ███████████████▊
 1    0.250 █████████                            1    4.17 ████████████████▋
```

## Reading the curve

- **The dial works and is monotone in spread.** off→1 drives top-3-family share
  0.59→0.25, families 9→12, H_norm 0.909→1.00. The "electric-purple / lava / ice"
  concentration (violet-purple top-1) drops from 23% to 8% of the set.
- **The pref cost is small; the real cost is portfolio *shrinkage*.** Mean pref-rank
  rises only +0.6 (from the ~3.5th-best to ~4.2nd-best palette per location) and
  pref-score falls ≈0.3 (≈2% of range) at the strongest setting. But `picks`
  contracts 22→12: when a family is capped out, cells whose only candidates are all
  in that family are orphaned (there's no other-family palette for that fractal ×
  color cell). **So the cap buys family spread mostly by emitting *fewer*
  wallpapers, not by emitting *worse* ones.** For an operator that reframes the dial:
  it's diversity-vs-*volume* at least as much as diversity-vs-pref.
- **`dScore` is non-monotone (cap=4 is +0.14).** Artifact of a set-mean over a
  shrinking set — dropping one low-pref pick lifts the mean. Treat **mean pref-rank**
  (per-wallpaper, robust to set size) as the headline cost; `dScore` is secondary.
- **A natural knee at cap=2:** H_norm 0.983, top-3 share halved (0.59→0.35), for
  +0.4 mean-rank and 5 fewer wallpapers (22→17). cap=1 forces literal one-per-family
  (H_norm 1.0) but costs another 5 wallpapers. **No operating point chosen** — this
  is the calibration curve.

## The load-bearing caveat (where the collapse actually is)

At the **production gate the emitted set is already fairly diverse** — 7 picks, top
family only 2/7, H_norm 0.963 — so the dial has almost nothing to grip. Two reasons,
both worth flagging:

1. **The selector's cell axis + ≤1/loc already de-concentrate stage 3.** The trace's
   collapse (top-3 share 0.62, 93% of per-location #1 picks amber-gold/violet-purple)
   is a property of the *pref top-K pool*, and MAP-Elites over fractal×color already
   subsamples it into a much flatter emitted set before the family cap ever runs.
2. **The quality gate and pref ranking are substantially orthogonal.** Of 21
   survivors at `p_ge3>0.90`, only **1 is pref-rank-1**; the off-set mean rank is 3.5
   even with the cap OFF. The pref-*best* palette for a location usually **fails** the
   wallpaper quality gate. So "off = pref-#1 everywhere" holds for the ungated pref
   funnel (the trace) but **not** for the gated emission set — gating alone already
   strips most of the attractor-family rank-1 picks.

**Implication:** the family cap is a real, cheap spread lever with a clean curve, but
the residual collapse it targets is largely *already handled* by the existing gate +
cell axis at the production floor. Its value shows up when the portfolio is large
(loose floor, many cells) — there cap=2–3 meaningfully flattens the family mix for a
modest pref-rank cost. It does **not** address the stage-3 root cause (pref-v3-gvo's
learned amber-gold/violet-purple preference); that stays a pref-head problem.

## Low / mid / high emitted sets (0.50 floor — inspect against crops)

Crops: `data/wallpaper_corpus/batches/2026-07-09_wallpaper_headbatch_dramatic_v1/crops/<image_id>.jpg`
(full lists incl. image_ids in `scratchpad/family_diversity_picks.json`).

- **LOW (cap=off, 22 wallpapers, 9 families):** violet-purple ×5, amber-gold ×5,
  pastel-iridescent ×3, high-key-luminous ×3, fire-ice ×2 — the two attractor
  families are 45% of the set.
- **MID (cap=2, 17 wallpapers, 10 families):** violet-purple ×2, amber-gold ×2, each
  family ≤2; sapphire-rose and teal-cyan enter. Top-3 share 0.35.
- **HIGH (cap=1, 12 wallpapers, 12 families):** exactly one per family — amber-gold,
  antique-faded, fire-ice, high-key-luminous, jewel-earth, oceanic, orchid-twilight,
  pastel-iridescent, red-rust, sapphire-rose, teal-cyan, violet-purple. Fully flat
  family mix; five fewer wallpapers than LOW.
