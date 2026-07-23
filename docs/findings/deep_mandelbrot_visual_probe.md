# Probe: deep Mandelbrot past the 1e-9 floor — perturbation engagement + q4-class looks

Eyeball-first visual probe (no emission, no classifier — Matt's eye judges). Goal:
descend deg-2 Mandelbrot well past the `1e-9` search floor, confirm the perturbation
tier engages cleanly across the f64→perturbation switch, and see whether deep
Mandelbrot yields artist-quality "q4-class" locations.

Artifacts under `out/deep_probe/`: `sheets/` (per-rung 3-palette contact sheets),
`ladder_anchor.png` / `ladder_scepter.png` (per-seed depth-ladder montages),
`timing/`, `telemetry_*.txt`.

## TL;DR

- **(a) Perturbation is clean across the switch — YES, unqualified.** The auto tier
  crossed f64 (fw 1e-9, spacing 9.8e-13) → perturbation (fw 1e-12…1e-20, spacing
  9.8e-16…9.8e-24) with **zero glitched pixels at every rung**, no seam, no
  stair-stepping, structure fully resolved to fw **1e-20** (~3×10²⁰ magnification,
  prec 141 bits). The engine renders deep Mandelbrot correctly today.
- **(b) Deep Mandelbrot yields q4-class looks — YES**, with a precision caveat that
  is the real story of this probe (below).
- **(c) The descent-geometry anomaly past 1e-9 is structural, not policy.** The
  `guided-descend` walker cannot produce a *valid* location below ~f64 resolution —
  `Frame.center` is `f64` and its node renders are **`BackendChoice::F64`-forced**.
  `--min-fw` mechanically stops the walk, but even with the floor lifted the walk
  descends on numerically-quantized f64 fields and emits f64 centers that cannot be
  re-rendered at depth. Lifting the floor alone does **not** unlock valid deep
  descent.

## What actually gates (answers "which constant")

Two independent floors, per `deep_zoom_precision_audit.md`, plus a third structural
limit this probe surfaced:

1. `--min-fw` (default `1e-9`, `guided_descend.rs:2771`) — the **binding** mechanical
   stop. A depth-≥2 step whose `new_fw < min_fw` dies with `EndCause::MinFwFloor` and
   harvests what it has. Lift this to descend deeper. **This is the only floor you
   have to touch to make the walk *proceed* past 1e-9.**
2. `depth_max` (default `17`, `:2759`) — co-gates: walks step ~0.4×/rung, so from a
   ~1e-2 root, depth 17 only reaches ~1e-7. To reach 1e-20 by walking you also need
   `depth_max` ≈ 50–60. (`FW_FLOOR=1e-9`, `:69`, only guards depth-1 root-start
   widths — never binds a deep descent.)
3. **The structural wall (new finding).** `Frame.center: Complex<f64>` (`render.rs:74`)
   and the descent node render is hardwired to `BackendChoice::F64`
   (`guided_descend.rs:533`), with the perturbation reference (when built at all)
   derived *from the f64 center* (`:529`). So the walker's own data structures top
   out at f64 precision (~2e-16 ulp near |c|~0.75). Below fw ~1e-15 the center is
   uncertain by **thousands of frame-widths** — the location is delocalized. `--min-fw`
   was set at `1e-9` as "bring-up insurance … keeps walks inside the tested f64
   regime" (`:2762`); it is in fact protecting a regime the walker *cannot represent*.

The engine's true precision ceiling (`MIN_FRAME_WIDTH=1e-300`, `main.rs:35`) and the
perturbation tier itself are nowhere near binding — but they live on the **bare
`render`/`sheet` path** (`iterate_location`, decimal-string center → BigFloat →
perturbation), **not** the walker.

## The judge-render path used here

The walker is f64-bound and `render-one` is f64-by-construction. The only
deep-capable, perturbation-selecting path is the **bare `render` / `sheet`**
(`iterate_location`, `main.rs:51`): decimal-string center parsed at full precision,
auto-selects perturbation when spacing ≤ `PERTURB_SPACING=1e-13`. All ladders below
were rendered with `sheet` (one deep iterate, 3 palettes: default / cubehelix /
viridis) at 1024×576 ss2, smooth coloring, `--backend auto`.

## Seeds

Deep ∂M seeds cannot be *guessed* — three separate 16-digit guesses this probe tried
landed in interior black or exterior gradient, because precise ∂M placement at fw=F
requires the center known to ~F. Nor can they be **offset** from a known ∂M point
(anchor + 5e-5 at fw=1e-9 → exterior; the offset is 5×10⁴ frame-widths off the
boundary). Both are the same delocalization lesson. Two published high-precision ∂M
coordinates were validated by render and kept:

| # | name | center (decimal strings, reproducible) | character |
|---|---|---|---|
| S1 | Seahorse valley | `-0.743643887037158704752191506114774` / `0.131825904205311970493132056385139` | Misiurewicz preperiodic → **self-similar** nested double-spirals |
| S2 | West "scepter" | `-1.749721929742338549812172197994` / `-0.000029016647523930041518664712` | **generic** grand log-spiral feature |

Depth ladder per seed: fw ∈ {3e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-20}, maxiter
scaled 3000→30000, one rung each side of the f64→perturbation switch (which sits at
fw≈1e-10 / spacing 1e-13 for a 1024-wide frame).

## (a) Perturbation cleanliness — clean, no caveat

Anchor S1 ladder telemetry (`telemetry_anchor.txt`), glitch count **0** at every rung:

| fw | backend | pixel spacing | prec bits | ref len | iterate |
|---|---|---|---|---|---|
| 3e-3 | f64 | 2.9e-6 | 83 | — | <0.1s |
| 1e-6 | f64 | 9.8e-10 | 94 | — | <0.1s |
| 1e-9 | f64 | 9.8e-13 | 104 | — | ~0.2s |
| 1e-12 | **perturb** | 9.8e-16 | 114 | 8001 | ~8s |
| 1e-15 | perturb | 9.8e-19 | 124 | 12001 | ~32s |
| 1e-18 | perturb | 9.8e-22 | 134 | 20001 | ~29s |
| 1e-20 | perturb | 9.8e-24 | 141 | 30001 | ~32s |

The 1e-9 (f64) → 1e-12 (perturb) transition shows **no visible seam** and no change
in fidelity — the two backends agree across the switch. No underflow/glitch flags
fired anywhere on either seed (`telemetry_scepter.txt` also 0). Timing probe: fw=1e-18
1024×576 ss2 maxiter 20000 = **25.9s** (full delta loop; no series-approximation, so
deep renders pay every iteration).

## (b) q4-class looks at depth — yes, but q4 is *depth-band-local*, not automatic

- **S1 Seahorse (self-similar):** q4-class at most rungs — 3e-3, 1e-9, 1e-12, 1e-18,
  1e-20 all show calm lakes (any tone: blue/gold/pink/green/purple) framing mid-scale
  spiral filaments with bright focal blooms. Because it is a Misiurewicz point it stays
  on ∂M at every scale, so structure persists all the way to 1e-20 (there just gets
  *denser* — trending toward "busy" as more self-similar generations pack in).
- **S2 Scepter (generic):** a **stunning** grand log-spiral at its native band
  (1e-9…1e-12) — arguably the strongest single composition of the probe — but the
  fixed center sits in the *exterior* by 1e-15 (its reference orbit escapes at iter
  569), so 1e-15/1e-18/1e-20 render as **flat featureless blue**. Clean, not glitched
  — just off-structure.

**The finding:** a fixed center holds q4 only over a limited depth band around where
it was chosen. Even the self-similar anchor oscillates — its 1e-6 rung is a sparse
needle against interior and its 1e-15 rung is a dense speckle band (also slightly
under-iterated at maxiter 12000; the maxiter-30000 1e-20 rung read cleaner). Only
truly self-similar (Misiurewicz) centers sustain q4 across many decades; generic
features have a narrow window. See `ladder_anchor.png` vs `ladder_scepter.png` — the
scepter's collapse to flat blue past 1e-12 is the whole story in one strip.

**Consequence for the pipeline:** harvesting q4 deep locations at scale needs a
component that *tracks ∂M at high precision* while descending (iterative
re-centering / Newton on nucleus/Misiurewicz points), producing centers as
decimal strings. Neither the walker nor anything else in the repo does this.

## (c) Descent-geometry past 1e-9

- With `--min-fw 1e-22` + high `depth_max`, the walk **does proceed** past 1e-9 and
  past the 1e-13 switch — but every node render stays on the **F64 backend**
  (perturbation never engages in descent; it is gated behind `iterate_location`, a
  path the walker doesn't call). Past spacing ~1e-13 those f64 fields are quantized /
  stair-stepped, so the foci finder, best-of-N screen, and spread/occupancy/black
  gates all operate on numerically-degraded input. Nothing in the walk flags the
  precision breakdown — there is no guard between `--min-fw` and the f64 quantization
  cliff (3+ orders apart).
- Empirical lifted-floor run (`--min-fw 1e-22`, 3 walks, depth_max 32,
  `walk_deep3/summary.json`): 70 candidates, **156s** (pathologically slow — deep
  steps thrash on best-of-N gate-failure redraws; 24 of 70 candidates screened by the
  occupancy floor over degenerate fields). Crucially: **`min_fw_truncations = 0`** —
  with the floor lifted, `depth_max` bound the walks (2/3 hit terminal depth, 1 died
  on occ-floor), *not* `--min-fw`. Deepest fw reached = **1.18e-14** — the walk sailed
  **past the 1e-13 switch** (node spacing ~1.5e-17), entirely on the F64 backend, with
  the center still f64 (ulp ~2e-16 ≈ 1.7% of that frame width — already delocalized by
  ~13 px), and emitted it as a candidate with no flag. It does not self-terminate on a
  precision signal; it just gets expensive and quietly invalid.
- Candidate centers are stored as `f64` (`guided_descend.rs:765`), so any deep
  candidate the walk *did* emit could not be re-rendered validly by the deep path
  anyway — the decimal precision was never there to carry.

**So:** lifting `--min-fw` is necessary but far from sufficient. Valid deep descent
needs the walker to carry the center as a high-precision decimal/BigFloat and to
render nodes through the perturbation tier — a real change, out of scope for "just
lift the floor."

## Recommendations (not acted on — probe only)

1. If deep Mandelbrot is worth pursuing: give the descent path a high-precision center
   (decimal string / BigFloat in `Frame` or a parallel deep-descent mode) and route
   node renders through `iterate_location`'s auto backend. Until then, keep `--min-fw`
   at `1e-9` — deeper is invalid, not merely untested.
2. Deep q4 harvesting is a **∂M-tracking** problem, not a floor problem. A Newton /
   nucleus-finder that lands centers on ∂M at target precision would let the existing
   (clean, proven-to-1e-20) perturbation render path produce deep wallpapers.
3. Deep renders are correctness-OK but slow (no SA/BLA — 25–32s/frame at 1024×576 ss2
   maxiter 20–30k). Series approximation is the lever if deep becomes a batch concern.
4. Scale maxiter generously with depth — the 1e-15 rung's speckle was partly
   under-iteration at maxiter 12000.
