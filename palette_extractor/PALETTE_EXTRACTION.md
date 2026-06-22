# Single-Palette Extraction from Fractal Images

Extract **one** well-defined, smooth, closed color palette from a fractal (or
fractal-like) image, in the `clean_colormaps.json` stop format. Everything that
isn't the dominant palette — antialiasing haze, beaded tendrils, secondary color
families — is computed only as intermediate scaffolding and discarded.

`palette_extract.py` is the implementation. This document is the *why*.

---

## 1. Mental model: a colormap is a 1-D curve

A colormap is a map `t ∈ [0,1] → OKLab`. A fractal render is
`color(x,y) = colormap(field(x,y))`, so the set of colors in the image is the
**image of that map — a 1-manifold (a curve) in color space**. "Extract the
palette" literally means "recover that curve." Every difficulty is a *departure*
from the ideal curve, and naming the departure tells you how to defeat it:

| Departure | What it looks like in OKLab | Defeated by |
|---|---|---|
| Antialiasing | low-density **chords** bridging far-apart parts of the curve | density (AA is O(perimeter), the curve is O(area-of-mass)) |
| Sampling / noise | fuzz thickening the curve into a tube | voxel centroiding + smoothing |
| **Junctions** | curve self-touches; at low chroma all hues collapse to a **hub** | min-weight graph routing; tip trimming |
| **Intrinsic dim** | the cloud is a **sheet/fan**, not a rope | accept it — report coverage; one curve can't cover a sheet |

The last row is the one that bites on rich images. **Measure it first** (local
PCA: if λ2 is consistently non-trivial you have a sheet, and a single palette is
an approximation with a real coverage ceiling, not a bug).

---

## 2. Pipeline

```
pixels ─► OKLab ─► density voxels ─► dominant ridge ─► kNN graph
       ─► MST + tree-diameter (principal curve) ─► trim sparse tips
       ─► detect closure | mirror-close ─► periodic smooth ─► uniform-arc resample
```

**OKLab (Ottosson).** Perceptual; ΔE is plain Euclidean distance. Both the cost
metric and the smoothness budget live here.

**Density voxelization.** Density is the single most discriminative signal.
The true palette is a high-density **ridge** (many pixels map to each palette
color); antialiasing is intrinsically low-density (boundaries are 1-D in image
space, interiors 2-D). A single-pass histogram at 48³ gives the density field
for free — no k-means iteration, and crucially it *keeps the mass* that k-means
would throw away. Per-voxel centroids denoise the curve.

**Ridge selection.** Keep the densest voxels accounting for ~90% of pixel mass.
This is what discards the thin tendrils and AA haze: they are numerous but
individually sparse, so they fall below the mass threshold.

**kNN graph + largest component.** Candidate connectivity (k≈8), edge weight =
OKLab ΔE. Keep the most massive connected component.

**MST + tree-diameter = the principal curve.** The MST of a rope is the rope
plus little hairs; its **diameter** (two-sweep farthest-node) is the spine, and
the hairs (branches, including beaded spurs) are pruned automatically. This
beats the alternatives we tried:

- **TSP is wrong.** TSP visits *every* node — the opposite of what you want.
  Robustness here is about *skipping* nodes (AA, beads). The right framing is
  prize-collecting / orienteering (max coverage − λ·length), and on a near-1-D
  ridge the diameter already solves it. (TSP is fine only as a *final
  sequencing* of a pre-cleaned ridge, never as the extractor over raw points.)
- **Spectral unrolling is fragile.** Elegant for a *true* loop (v2,v3 → a
  circle), but when the cloud isn't a ring the two smallest non-trivial
  eigenvalues are degenerate noise and the embedding scrambles — *unstably*
  (different scramble each run). Keep it, if at all, only to confirm a loop.
- **Diameter handles the junction hub for free.** The desaturation hub (where
  chroma→0 and every hue collapses together) is a many-way crossing. The
  diameter routes through it on minimum-weight edges and exits the matching
  arm — no explicit tangent-continuity logic needed. A clean palette passes
  through white in the middle with no detour.

**Trim sparse tips.** The diameter's ends often plunge into near-black bead
tips joined by an over-budget jump. Those tips are the *trap/interior* coloring,
a different layer — trim them off rather than smoothing them in.

**Closure: detect, don't impose.** If the spine's two ends already meet
(gap ≤ `tau_close`) it's a **native loop** — join them. Otherwise the colormap
is genuinely open (e.g. a blue→…→orange arc) and forcing a wrap would *invent*
the discontinuity you're trying to avoid. Verified empirically: `twilight`
closes to a 0.003 gap (cyclic by design); the open fractal arc does not.

**Mirror = the fallback for open maps.** When no good native loop exists, mirror
the spine out-and-back (`c0…cn…c1`) to produce a seamless closed loop. This is
how diverging maps become cyclic. It doubles arc length (so resample with enough
stops — see below) and the turnarounds are C0 cusps, which periodic smoothing
softens.

**Periodic smoothing.** A wrap-mode Gaussian along the loop kills the zig-zag
that the diameter produces on *thick* ridges (parallel strands of a sheet-like
cloud). This is what makes the palette "well-*defined*" rather than jittery. The
cost is a slight rounding of fidelity — worth it.

**Uniform arc-length resample.** Output stops are evenly spaced in OKLab, so the
gradient is perceptually even (the same no-dead-black, smooth-gradient property
good generated palettes already have). With N stops, the per-stop ΔE is
`arc_length / N`, so **N controls smoothness directly** — 256 (a standard LUT
size) keeps steps tiny even for a mirrored (2×-length) loop. For the short
`twilight`-style arcs, 33 also clears the budget.

---

## 3. Coverage and the two thresholds

- **Coverage** = fraction of pixels within `eps` of the palette **curve**
  (point-to-polyline, via a KD-tree on a densely-resampled loop) — *not* the
  discrete stops, which would undercount a coarsely-sampled curve.
- **ε** (coverage tolerance) and **δ** (max step between stops) are different
  scales but should be coupled: **δ ≤ 2ε** makes consecutive ε-balls overlap, so
  the covered region is a connected tube with no gaps along the curve.

Coverage is the honesty metric. A clean palette at 38% coverage is telling you
the image is a multi-family **sheet**, and one curve is doing its honest best —
not that the extractor failed.

---

## 4. Parameters

| Param | Default | Effect |
|---|---|---|
| `n_stops` | 256 | output resolution; also sets per-stop ΔE = arc/N |
| `voxel_res` | 48 | density grid; higher = finer ridge, noisier |
| `mass_fraction` | 0.90 | ridge mass kept; **lower → thinner, more dominant core** |
| `knn_k` | 8 | graph connectivity |
| `trim_delta` | 0.06 | tip-trim threshold (≈ δ budget) |
| `tau_close` | 0.10 | endpoint gap below which a loop is "native" |
| `smooth_frac` | 0.012 | wrap-Gaussian σ as fraction of loop; **higher → cleaner, less faithful** |
| `coverage_eps` | 0.05 | ε for the coverage metric |

Tuning intuition: if the palette zig-zags, raise `smooth_frac` or lower
`mass_fraction`. If coverage is low and the image *is* rope-like, lower
`mass_fraction` to lock onto the dense core.

---

## 5. Validation (the two test images)

| Image | Closure | Coverage | max-step | Note |
|---|---|---|---|---|
| `glowdon` | mirrored | ~57% | 0.015 | open warm→cool arc; blue field + beads discarded |
| `crisis-25` | native | ~38% | 0.006 | sheet-like; clean cream↔blue loop, low ceiling is real |

Both palettes are smooth (max-step ≪ δ) and well-defined. The coverage gap
between them is the rope-vs-sheet distinction, surfaced honestly.

---

## 6. Output format

```json
{
  "name": "crisis-25",
  "source": "extracted",
  "closed": true,
  "closure": "native",          // or "mirrored"
  "coverage": 0.38,             // fraction within eps of the curve
  "max_step_oklab": 0.006,
  "stops": [[0.0, [r,g,b]], [0.0039, [r,g,b]], ...]   // t = i/N, rgb 0-255, cyclic
}
```

`t` runs `i/N` (cyclic — no duplicated endpoint), so a consumer wraps `t=1`→`t=0`.

---

## 7. Known limitations / upgrade paths

- **Diameter maximizes length, not coverage.** For a fan it grabs the longest
  arm-pair. If you want the *max-coverage* path instead, replace the two-sweep
  with a mass-weighted longest-path DP on the MST (orienteering). Often changes
  which arms get chosen on rich images like crisis-25.
- **The discarded layers are recoverable.** If you ever want them: extract,
  mask every pixel within ε of the curve, re-extract on the residual (matching
  pursuit over palettes). Intentionally *not* done here — single palette only.
- **Sheet images have a coverage ceiling.** No single 1-D palette covers a 2-D
  color cloud. The number is the signal; act on it (accept, or switch to
  multi-palette) rather than fighting it.
