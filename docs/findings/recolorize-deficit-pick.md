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
