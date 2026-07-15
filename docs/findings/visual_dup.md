# Visual-duplicate measurement — is coordinate-distinct-but-visually-identical a real mode?

**Source:** `overnight_20260713_001420` emit, 62 rows → 47 same_fractal-distinct (the
coordinate-dedup fix `96bed12`). **Verdict by eye; no gate built.**

## TL;DR
- **Yes, the mode is real — at the *morphology* level.** ~20 of the 47 "distinct"
  fractals (43%) sit *within the known-dup control band* by palette-blind morphology
  similarity. The best coordinate-distinct pairs (CLIP 0.96–0.978) are **as
  morphologically alike as genuine recolor dups** (control band 0.92–0.985).
- **But it is largely *masked at the wallpaper level*.** The emission selector's
  MAP-Elites color diversification gives same-skeleton pairs different palettes, so
  the shipped color renders read as distinct wallpapers (see contact-sheet color
  columns). A geometry-only dedup gate would over-collapse legitimate color variants.
- **Two concrete redundancy sources:** (1) **phoenix** — all 9 distinct phoenix
  collapse into one morphology group (they share the fixed Ushiki c/p, so every
  "location" is a viewport of one system); (2) recurring **log-spiral** and **radial-
  starburst** morphotypes across julia/multibrot/mandelbrot ("good morphology is
  intrinsically narrow", as anticipated).

## Descriptor fitness (positive controls FIRST)
Canonical render = deterministic robust-z-score (median/MAD tanh, K=2) on the smooth
field → grayscale; palette- and framing-independent. Two descriptors, both on the
grayscale renders.

- **CLIP ViT-B/16 (openai), zero fractal training — FIT.** 18/19 known-dup control
  pairs score ≥0.922 (median 0.955); top coordinate-distinct pairs are 14/16
  same-family; graded (distinct-pair median 0.855, p95 0.942) with a real high tail.
- **v6 backbone pre-logits (in-house, fractal-trained) — UNFIT here.** Saturates on
  palette-stripped grayscale: cross-family pairs hit *exact* cos=1.0000 (julia↔phoenix,
  julia↔mandelbrot), 145 distinct pairs ≥ control-median vs CLIP's 28, and one recolor
  control inverts to 0.42. The recall-study separation was on *color* crops with coarse
  k-means; on fine grayscale pairwise it collapses. **Use CLIP for any gate.**
- **One loose control:** the sole 0.799/0.42 recolor pair is a genuinely wide-framing
  julia merge (fw 1.71 vs 3.00, whole-set views) that the *identity rule* (scale-aware,
  ratio≤4) joined — not a descriptor bug. Both descriptors flag it as the outlier.

## Where distinct pairs fall vs controls (CLIP, single-linkage on the 47 reps)
| threshold | meaning | clusters | locs merged |
|---|---|---|---|
| 0.985 | tightest control | 47 | 0 |
| 0.978 | ~tight dup | 46 | 1 |
| 0.970 | — | 44 | 3 |
| **0.955** | **control median** | **27** | **20** |
| 0.937 | control p25 | 18 | 29 |

Sharp knee 0.97→0.955 (3→20 merged): a dense shell of coordinate-distinct pairs sits
*exactly in* the genuine-dup band. At the control median the 4 merged groups are:
9×phoenix; 5×mixed log-spiral (julia/mb3/mandelbrot); 4×radial-starburst
(julia/mb3/mb5); 6×julia_mb3 + 1×julia_mb4.

## If a gate follows (out of scope here)
1. **Phoenix first** — the clearest win (9→~1–2); a fixed-c/p family artifact, better
   handled by viewport-spread or a phoenix-specific dedup than the general rule.
2. **Mandelbrot/julia**: gate at a *conservative* CLIP threshold (~0.97, merges only
   the 3 tightest) to avoid killing color variants; or add a morphology axis to the
   selector's niche instead of a post-hoc dedup.

## Artifacts (calibration set — promote to `data/` if a gate is built)
`scratchpad/visual_dup/`: `reconstruct.py`/`clusters.json` (62→47 structure),
`render_fields.py`/`canon/`+`fields/` (canonical grayscale + raw fields),
`embed.py`/`embeddings.npz` (v6+CLIP), `measure.py`/`out/` (`report.json`,
`sim_matrices.npz`, `clip_clustering.json`, `contact_sheet.png` [v6-ranked],
`contact_sheet_clip.png` [CLIP-ranked, the primary]).
