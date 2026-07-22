# Pool render res pinned to 960×540 ss2 (was a leftover 1280×720 ss2 corpus-crop default)

The whole-library colorize's cost is the per-location fractal re-render. It was rendering the
pool at **1280×720 ss2** (= 2560×1440 supersampled, 3.69 M px) — the Stage-2 corpus-crop res,
which **nothing downstream consumes**: both quality heads deploy `Transform(train=False)` =
**384×224 bicubic stretch**, the palette pick scores on the cached 640×360 field, and
realized-stats is a resolution-robust histogram. So the pool render only has to survive a
384×224 downsample.

## Scores-match guard (before relaunch)

Batch-noise floor: fresh-render vs stored score **|Δ| = 0.0 exactly** — render+score is fully
deterministic, so every delta below is real resolution-sensitivity, not noise.

**Mining head (strange styles)** — re-render 25 stopped-run rows at candidate res, mining-score:

| cand vs old 1280 ss2 | median \|Δ\| | max \|Δ\| |
|---|---|---|
| 640×360 **ss2** | 0.007 | 0.052 |
| 640×360 ss1 | 0.019 | 0.167 |

Mining is res-robust at 640 ss2; **ss1 is out** (no-AA aliasing on stripe/composite fields
inflates the tail).

**Wallpaper head (smooth, strict 0.90 release floor)** — 22 smooth rows spanning p_ge3∈[0,0.99]:

| cand vs old 1280 ss2 | median \|Δ\| | max \|Δ\| | spearman | 0.90-floor residual flips |
|---|---|---|---|---|
| **960×540 ss2** | 0.027 | 0.126 | 0.933 | **0/22** (remap 0.90→0.896) |
| 640×360 ss2 | 0.023 | **0.300** | 0.894 | 5/22 |

The wallpaper head was **trained on 1280-sourced crops**, so its 384×224 input carries
1280-derived detail; a 640-sourced downsample is out-of-distribution and the score scrambles
non-monotonically (e.g. em_000021 0.927→0.754 at 640, while others rise) — a fat 0.30 tail that
flips the strict 0.90 release floor. At **960 ss2** the shift collapses to median 0.027,
spearman 0.933, and **zero release-floor flips** (the implied remap 0.90→0.896 is within the
head's own res-noise, so floors are kept unchanged).

## Decision

**Pool render res = 960×540 ss2** — the smallest res that faithfully feeds BOTH heads. 0.56× the
pixels of the old default → ~1.8× faster (13.0 → ~7.3 s/loc, ~4.9 h → ~2.8 h for 1387 loc).
Floors unchanged (0.75 / 0.90 / 0.05 / 0.50); the guard shows decisions match within noise.
Pinned as the `POOL_*` default (standing fix). The release render (judge-quality 1024×576 ss2)
is a separate slot, unchanged.

640 ss2 (1.3 h) was rejected: it fails the wallpaper head. Per-style res (640 for strange, 960
for smooth) would recover most of that speed but violates the single-default pin; deferred.

Not done here (teed up for the profiling pass): **field-cache → recolor** — dump each location's
960-res field once and re-color per palette with zero re-iteration, collapsing the re-render to
sub-second. The durable fix for *recurring* re-colorize; a considered change, not a mid-flow
retrofit.
