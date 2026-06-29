# Ultra Fractal Coloring Algorithms — Implementation Reference

Four standard coloring algorithms from Ultra Fractal's `Standard.ucl`: **Gaussian
Integer**, **Exponential Smoothing**, **Direct Orbit Traps**, and **Decomposition**.
For each: what it computes, the math, where it hooks into the orbit, the parameters,
and notes for a vectorized numpy rasterizer. Logic is reconstructed from the
algorithm behavior and paraphrased into framework-neutral pseudocode rather than
copied from the `.ucl` source.

---

## The UF execution model (shared backbone)

Every UF coloring algorithm runs in three sections, which map cleanly onto an
accumulate-then-finalize orbit loop:

- **init** — once per pixel. Allocate accumulators, precompute constants.
- **loop** — once per iteration, *after* the fractal formula's own step, with the
  current `z` visible. This is where orbit-monitoring algorithms accumulate.
- **final** — once, after bail-out. Produces either:
  - `#index` — a float gradient position. The gradient is **periodic over [0,1]**
    (density/transfer/offset are applied afterward), so values outside [0,1] wrap.
  - `#color` — a direct RGBA value (used by "Direct" algorithms instead of an index).
  - optionally `#solid` — flags the pixel to take a flat solid color.

Two classes matter here:

- **Orbit-monitoring** (uses `loop`): builds the index from the *whole* orbit —
  Gaussian Integer, Exponential Smoothing, Direct Orbit Traps.
- **Final-only** (uses only the last `z`): Decomposition.

In numpy terms: orbit-monitoring algorithms need per-pixel accumulator arrays updated
every iteration step on the still-active mask; final-only algorithms just read the
final `z` array once. All four are "outside" colorings by default (computed on escaped
pixels); Gaussian Integer and Exponential Smoothing are also usable inside.

---

## 1. Gaussian Integer

### What it is

Gaussian integers are complex numbers `a + bi` with integer `a, b` — a unit square
lattice in the plane. The algorithm watches the orbit and, each iteration, measures how
close `z` lands to the nearest lattice point. Coloring by that distance (and related
statistics) yields the richly textured fields of circles, dots, and stars UF's docs
describe. It is also exposed as a *trap shape* plug-in, i.e. it is fundamentally a
point-lattice orbit trap with per-iteration distance bookkeeping.

### Math

Let `N` be a complex **normalization factor** (see params). Each iteration:

```
q       = round( z / N )          # nearest Gaussian integer, component-wise
remain  = z - q * N               # residual relative to the scaled lattice
r       = |remain|                # distance to nearest lattice point
```

`round` applies independently to the real and imaginary parts (`trunc`/`floor`/`ceil`
are the alternatives). With `N = 1`, this is just the distance from `z` to the nearest
integer-coordinate point. A non-trivial `N` scales/rotates the lattice (and if
`N = pixel`, the lattice differs per pixel).

Maintained across the orbit:

```
total += r ;  rave = total / iter          # running mean
if r < rmin: rmin = r; zmin = z; itermin = iter
if r > rmax: rmax = r; zmax = z; itermax = iter
```

### Output modes ("Color By")

| Mode | `#index` |
|---|---|
| minimum distance *(default)* | `rmin` |
| average distance | `rave` |
| maximum distance | `rmax` |
| iteration @ min | `0.01 * itermin` |
| iteration @ max | `0.01 * itermax` |
| angle @ min | `normalize_angle(zmin)` |
| angle @ max | `normalize_angle(zmax)` |
| min/mean/max angle | angle of the complex `(rave - rmin) + i(rmax - rave)` |
| max/min ratio | `rmax / (rmin + 1e-12)` |

where `normalize_angle(w) = (atan2(w) / π)`, shifted by `+2` if negative, then `*0.5`
(i.e. atan2 result folded into [0,1)). The `0.01 *` scaling on iteration modes assumes
the gradient is read modulo 1, so iteration counts band the gradient every 100 iters.

### Parameters

- **Integer Type** — `round` (smoothest), `trunc`, `floor`, `ceil`.
- **Color By** — the table above.
- **Normalization** — `none` (`N=1`), `pixel` (`N=#pixel`), `factor` (`N` = a fixed
  complex constant), or `f(z)` (`N` = a function of current `z`).
- **Randomize** — optional. Before rounding, perturb `z` by a small factor driven by a
  logistic map `logfac = 4·logfac·(1−logfac)`, seeded in [0,1]; `z *= (1 − size·logfac)`.
  Each seed gives a different speckle pattern. (As a trap plug-in this is dropped; you'd
  instead jitter the trap position.)

### numpy notes

- Keep `rmin, rmax, total, itermin, itermax, zmin, zmax` as full-grid arrays; update
  under the active mask each step.
- Complex round: `np.round(z.real) + 1j*np.round(z.imag)`. Same idea for trunc/floor/ceil.
- Division by complex `N`: trivial for `N=1`; for `N=#pixel` precompute `1/N` once.
- "minimum distance" with `round` + `N=1` is the canonical look and the cheapest path —
  basically a single `np.minimum(rmin, r)` per step plus the residual computation.
- The randomize logistic recurrence is inherently sequential per pixel but vectorizes
  fine across pixels (it's the *same* scalar recurrence advanced once per iteration, or a
  per-pixel array if you seed per pixel).

---

## 2. Exponential Smoothing

### What it is

A smooth-iteration coloring that, unlike the standard log–log smooth count, needs no
knowledge of the formula's exponent and works for **both divergent and convergent**
fractals. UF's docs note it doesn't map exactly to iteration count but stays close, and
that it pairs with almost any formula. For Mandelbrot/Julia (purely divergent) you only
need the divergent branch.

### Math

Two independent accumulators over the orbit; `zold` is the previous iterate:

```
divergent : sum  += exp( -|z| )            # each iteration
convergent: sum2 += exp( -1 / |zold - z| ) # each iteration
zold = z
```

At bail-out, decide which regime the orbit fell into and emit:

```
if |z - zold| < 0.5:        # orbit converged
    #index = sum2           (or 0 if convergent coloring disabled)
else:                       # orbit diverged
    #index = divergescale * sum
```

**Why it's smooth (divergent case):** while `|z|` is small, `exp(-|z|) ≈ 1`; once the
orbit blows up, terms collapse toward 0 almost immediately. So `sum` ≈ (iterations spent
near the set) plus a smooth fractional tail contributed by the last few growing steps.
That fractional tail is what removes the integer banding — and it's bail-out robust,
so low bail-outs are fine.

**Convergent case:** as the orbit settles, `|zold − z| → 0`, so `1/|·| → ∞` and
`exp(-1/|·|) → 0`; large early steps contribute ≈ 1. So `sum2` smoothly counts steps
before convergence. Relevant for Newton/Nova/Magnet, not plain Mandelbrot/Julia.

### Parameters

- **Color Divergent** (bool) / **Color Convergent** (bool) — enable each branch; enable
  at least one. Only Magnet-type fractals need both. Disabling the unused branch is a
  small speedup.
- **Divergent Density** (`divergescale`) — scales the divergent index so its color
  density can be matched to the convergent side; only meaningful when both regimes coexist.

### numpy notes

- For Mandelbrot/Julia: one accumulator, `sum += np.exp(-np.abs(z))` under the active
  mask each step; final `index = divergescale * sum`. Cheap and trivially vectorized.
- Accumulate up to and including the escaping step; inactive pixels stop contributing.
- This is a strong, formula-agnostic alternative to your existing `smooth` for cases
  where you don't want to hardwire the exponent or you're using aggressive bail-outs.
  The trade-off vs log–log smooth: the index scale is arbitrary (not "fractional iteration
  count"), so you'll re-tune palette density rather than reuse smooth's calibration.
- Guard the convergent branch against `|zold - z| == 0` (exact fixed point) → use a tiny
  epsilon or treat as fully converged.

---

## 3. Direct Orbit Traps

### What it is

The **direct-color** sibling of the general Orbit Traps algorithm. Instead of reducing
the orbit to one scalar `#index` and looking up the gradient once, it computes a color
**every iteration the orbit lands inside the trap**, and composites those samples
together — effectively stacking many semi-transparent layers within a single coloring
pass. Because the samples are taken from the gradient and merged, editing the gradient
recolors the whole structure; the merged result can be hard to predict, which is also
where the lacy, overlapping, layered looks come from.

This is the generalization of the "sample a trap shape every iteration and overlay the
hits" idea — directly relevant to the **`trap_cross` lace composite** concept: a cross
trap, sampled per iteration and alpha-composited, *is* a Direct Orbit Traps pass with
`trapshape = cross`.

### Per-iteration pipeline

```
init:   accumulator = startcolor          # RGBA background
loop:
    z2 = (z - trapcenter) * rot           # translate + rotate
    if aspect != 1: z2 = re(z2) + i*aspect*im(z2)
    d  = trap_distance(z2)                # see shape catalog below
    if d < threshold:
        current = gradient( color_key )   # RGBA sample, key per "Trap Color" mode
        if modifier == "distance":
            current.a *= (1 - d/threshold)        # soft edge: closer = more opaque
        accumulator = composite(accumulator, current, mergemode, order, opacity)
final:  #color = accumulator
```

### Trap Color (what the gradient key is)

- **distance** — `gradient(d / threshold)`
- **magnitude** — `gradient(|z2|)`
- **real** / **imaginary** — `gradient(|re z2|)` / `gradient(|im z2|)`
- **angle to trap** — `gradient( atan2(z2) folded into [0,1) )`
- **angle to origin** — same but on the unrotated `z`
- **angle to origin 2** — `gradient( 0.02 * |atan(im z / re z) * 180/π| )`
- **iteration** — `gradient( iter / maxiter )`

### Compositing controls

- **Merge mode** — normal, multiply, screen, overlay, … (`normal` ⇒ the new sample
  simply over the accumulator).
- **Merge order** — *bottom-up* (new sample blended onto accumulator) vs *top-down*
  (accumulator blended onto new sample). Changes which hits dominate.
- **Merge opacity** — global per-sample opacity.
- **Merge modifier = distance** — multiplies sample alpha by `(1 − d/threshold)`, giving
  anti-aliased / feathered trap edges. The main quality lever.
- **Start color** — the background the stack composites onto (often transparent or black).

`composite` here is alpha-over with the chosen blend mode and opacity; `blend(c1,c2,t)`
is linear color interpolation; for `normal` mode the blended color is just the sample.

### Trap shape catalog (shared with Orbit Traps)

Each shape is a distance function `d = trap_distance(z2)`; `D` = diameter, `k` = trap
order, `f` = trap frequency:

- **point** `|z2|`
- **ring** `| |z2| − D |` ; **ring 2** `| |z2|² − D² |`
- **cross** `min(|re z2|, |im z2|)` ; **hypercross** `|re z2 · im z2|`
- **hyperbola** `|re z2 · im z2 − D|`
- **diamond** `|re z2| + |im z2|` (L1) ; **rectangle** `max(|re z2|, |im z2|)` (L∞)
- **box** `| max(|re z2|,|im z2|) − D |`
- **astroid** `|re z2|^k + |im z2|^k` (reciprocal if `k<0`)
- **lines** `| |im z2| − D |`
- **waves / mirrored waves / radial waves** — `im`/radius modulated by `sin(·f)·k`
- **ring / grid / radial ripples** — cosine ripples inside radius `k`, 0 outside
- **egg, pinch, spiral, heart** — specialized closed curves

For the lace idea, **cross** (`min(|re|, |im|)`) is the one to start with; **hypercross**
(`|re·im|`) gives the alternative "soft asymptote" cross.

### numpy notes

- Accumulator is a float `(H, W, 4)` RGBA array; `startcolor` initializes it.
- You need a vectorized gradient/palette LUT returning RGBA for an array of keys, keys
  wrapped to [0,1).
- Each iteration: compute `d` over the whole grid, build `mask = (d < threshold) & active`,
  sample `current` for the masked pixels, then composite **only** where `mask` is true.
- Over-compositing: for `normal` mode the per-pixel update is the standard
  `out = current·αc + acc·(1−αc)` (premultiply if you care about correctness with the
  background alpha). Apply `opacity` into `αc`. The distance modifier scales `αc` by
  `(1 − d/threshold)` for feathering.
- This is the heaviest of the four (full-grid composite every iteration), but it
  vectorizes cleanly; tile for wallpaper-scale memory. Most cost is the gradient lookups
  and the masked composite, both of which you can restrict to active pixels.
- Determinism: composite order is iteration order, so it's deterministic as long as your
  iteration loop is — no concurrency hazards across pixels.

---

## 4. Decomposition

### What it is

The simplest of the four and final-only: it colors by the **angle of the final `z`** at
bail-out, spread across the whole gradient. This produces the characteristic circular
bands / radiating field-line "petals" around the set boundary. Low bail-outs (e.g. 4)
give the cleanest structure. (The two-color sibling, **Binary Decomposition**, thresholds
the same angle into just two gradient entries — at the left and middle of the gradient —
reproducing the *Beauty of Fractals* look.)

### Math

```
final:
    a = atan2(z)              # angle of final z, in (-π, π]
    if a < 0: a += 2π         # fold to [0, 2π)
    #index = a / (2π)         # → [0, 1)
```

Binary Decomposition variants:

- **Type 1:** `#index = 0.5 if (re z · im z ≥ 0) else 0`
- **Type 2:** `#index = 0.5 if (atan2(z) > 0) else 0`

### numpy notes

- `a = np.angle(z)` returns `(-π, π]`; `index = (a % (2*np.pi)) / (2*np.pi)`. Compute on
  escaped pixels only.
- Often most useful as a **layer** over a smooth/continuous base: decomposition supplies
  the angular field-line structure, the base supplies radial shading.
- It pairs naturally with low bail-out. At high bail-out the angle is still defined and
  the bands still form, but the visual character changes (tighter, more uniform petals).
- Cheap: one `arctan2` over the final-`z` array, no loop accumulation.

---

## How these map to your pipeline

- **Decomposition** is a one-liner final-section addition (`np.angle` on final `z`),
  best used as a layered angular field over your existing `smooth` base — closest in
  spirit to your `tia`/`stripe` "structure over shading" usage.
- **Exponential Smoothing** is a drop-in alternative shading channel to `smooth`:
  formula-agnostic and bail-out robust, at the cost of recalibrating palette density
  since its index isn't a fractional iteration count.
- **Gaussian Integer** (minimum-distance, `round`, `N=1`) is a lattice **point trap** —
  it sits alongside `trap_circle`/`trap_cross` as another orbit-trap keeper candidate,
  with the angle/iteration/ratio modes as extra channels.
- **Direct Orbit Traps** is the general engine behind the pending **lace** concept:
  `trapshape = cross` (or `hypercross`), `Trap Color = distance`, distance-modifier
  feathering on, composited every iteration. If lace works out, this is the framework to
  implement it in generally (any shape, any color key, any merge mode), rather than as a
  one-off.

## Sources

- Ultra Fractal manual (ultrafractal.com/help) — Gaussian Integer, Exponential Smoothing,
  Direct Orbit Traps, Decomposition, Binary Decomposition, Orbit Traps, Writing coloring
  algorithms.
- `Standard.ucl` formula logic (Mitchell / Jones / Slijkerman), for the section structure,
  output modes, trap shape distance functions, and compositing pipeline.
