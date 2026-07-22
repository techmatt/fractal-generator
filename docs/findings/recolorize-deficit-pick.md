# Re-colorize: the low-chroma / no-green / no-rainbow cause is the within-flavor PICK

**Verdict: pick bias at the v3-gvo within-flavor palette selector — NOT a supply gap, NOT
recipe muting, NOT the head gate.** Fixed at the colorize pick; no measure edit.

## Diagnosis (cheap, no re-render — from `out/first_release/pool_log.jsonl` + intrinsic palette stats)

The whole-library colorize (1387 loc, one colorize each) picks the concrete palette within a
measure-chosen flavor by **v3-gvo argmax**. That pick systematically collapses variety:

| axis | pool supply (987) | chosen (intrinsic) | realized output |
|---|---|---|---|
| mean_chroma p50 | 0.306 | 0.332 | 0.310 |
| high-chroma (>0.5) | 13% | 11% | — |
| green-present | 21% | 14% | **8.5% hue-share** |
| rainbow (≥6 hue bins) | 18% | 23% | — |
| special:spectral | 57 palettes | — | routed (63 loc) |

Three candidate causes, decided by data:

1. **NOT a supply gap.** The pool holds 128 high-chroma, 205 green-present, 57 spectral, 181
   rainbow. The measure spreads flavors flatly (all 19 flavors get 61–86 locations, incl.
   `special:spectral` 63). Variety is available and reached.
2. **NOT recipe muting.** Realized `mean_chroma` p50 0.310 ≈ chosen-palette intrinsic p50 0.332
   (a ~7% drop). Chroma survives the canonical pct/γ1 recipe.
3. **IT IS the pick.** Within the chosen flavor: **64%** of picks are >0.15 lower-chroma than an
   available flavor-mate (median headroom **0.206**); **86%** pass over a >0.2-greener flavor-mate
   (median green headroom **0.392**). v3-gvo's ranking bias (amber-gold/violet, see
   `palette-family-collapse-localized`) leaves green/high-chroma on the table at zero flavor cost.

**The head gates do not block green/high-chroma** (so a pick fix reaches the gated pool): wallpaper
`corr(green,p_ge3) = −0.07 ≈ 0`, `corr(chroma,p_ge3) = 0.00`; mining `corr(green,p_ge3) = +0.14`
(green passes *more* often). The bias is entirely upstream at palette *selection*, before any render.

## Fix — `tools/emission/palette_deficit.py` (`--palette-pick deficit`)

Within a flavor, choose the palette to fill a **running realized chroma×hue deficit** (target =
uniform hue with a green over-weight + mid-high-chroma aspiration), with **v3-gvo as the
within-deficit tiebreaker**:

```
obj(member) = z(v3-gvo) + λ · z(deficit_gain),   argmax over members with z(v3-gvo) ≥ −1.5
deficit_gain = d_hue·sig_hue + w_c·d_chroma·sig_chroma + w_s·sig_spread
```

- Intrinsic palette signatures use the SAME HSV convention as `realized_palette_stats` (max−min
  chroma, RGB-wheel hue), so a palette's signature predicts its render's hue/chroma.
- The non-uniform target discriminates even at an empty start (green favored from render 1); as
  green fills, its deficit falls and the pull self-balances to the next starved hue.
- Resume-safe: the tracker is a **sum** over realized histograms, rebuilt by replaying the durable
  `pool_log.jsonl`, so a kill/resume continues the exact deficit.
- `pref` mode (v3-gvo argmax) stays the batch-stable default; deficit is opt-in.
- Head floors unchanged (0.75 / 0.05): a deficit pick that is genuinely poor simply fails to gate.

## Smoke validation (30 locations, deficit vs pref on the SAME locations)

| | mean green | mean chroma | green-present frac |
|---|---|---|---|
| pref (baseline) | 0.012 | 0.297 | 0% |
| **deficit** | **0.233** | **0.432** | **43%** |

27/30 palettes changed, all within the same flavor (e.g. `azarn→cmr.tropical` green 0.00→0.65;
`795565→commons_Julia` green 0.00→0.53, chroma 0.18→0.72), and quality held (a tia row's p_ge3 rose
0.098→0.578). Early-run green is the strongest (empty-tracker pull); the full-library mean self-limits
toward the target as the deficit saturates.

## Full-library result (out/recolor_release, 960×540 ss2 pool render)

1387 colorized, **725 gated (52.3% pass — UP from first_release's 47.6%**; the head gates
reward the added green/chroma), 112 release-eligible, 50 released (0 short-fill, 0 errors).
Palette variety measurably up vs the prior pool:

| metric | full pool | gated pool | release-50 |
|---|---|---|---|
| mean green | 0.085 → **0.272** (3.2×) | 0.102 → 0.309 | 0.117 → 0.220 |
| green-present frac | 0.13 → **0.47** | 0.15 → 0.51 | 0.18 → 0.40 |
| mean chroma | 0.326 → 0.435 | 0.339 → 0.467 | 0.337 → 0.390 |
| high-chroma frac (mc>0.4) | 0.30 → **0.54** | 0.34 → 0.63 | 0.32 → 0.44 |
| rainbow frac (≥6 hue bins) | 0.15 → **0.39** | 0.17 → 0.36 | 0.18 → 0.40 |

`special:spectral` FLAVOR share is ~unchanged (0.045→0.048) — flavor allocation is measure-driven
and untouched; the rainbow rise comes from the deficit pick choosing more spread-y palettes WITHIN
each flavor, not from more spectral-flavor routing. Green likewise rises via within-flavor picks.

**Selection (committed fixes):** heads never mixed (disjoint within-head passes), split honest
25 smooth / 25 strange (realized strange-frac 0.50, no dip), coverage **non-inert** — cov.gain<1
on 48/50 picks (median 0.115) via the morph-CLIP kernel (vs first_release's uniform 1.0). Release-50:
50/50 distinct morph clusters, all 9 families, mean pairwise morph-dist 0.165.

**Timings (960×540 ss2):** colorize 3.35 h wall (8.7 s/loc incl. warmup; steady 7.6), release
render 6.2 min for 50 (judge 1024×576 ss2). Two-point per-stage split: fixed pref-pick+score ~5.7 s,
pixel-scaling render ~4.1 s (was ~7.3 s at 1280 ss2). See `pool-render-res-960.md`.
