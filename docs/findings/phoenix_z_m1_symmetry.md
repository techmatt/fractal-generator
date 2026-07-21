# Phoenix `z_{-1}` symmetry — spec correction

**Context.** Phoenix Phase A (`prompts/phoenix_phase_a.md`) opens `z_{-1}` (the
two-state slice coordinate) as a first-class parameter axis and asks for a symmetry
guard "that stops anyone silently re-pinning `z_{-1}`". The governing spec
(`docs/design/phoenix_seed_sampler_spec.md` §3, §7) frames the guard around a **180°
point symmetry**: "With `z₋₁ = 0`, orbits from `z₀` and `−z₀` coincide from step 1,
so the rendered image is exactly 180°-symmetric … Setting `z₋₁ ≠ 0` breaks the
symmetry."

**Finding: the 180° claim is false for this engine's pixel convention.** Our Phoenix
kernel seeds `z₀ = pixel`, `z_{-1} = z_m1` (see `Family::Phoenix`, `PhoenixBackend`).
Under `z_{-1}=0`:

```
z₁ = z₀² + c            (even in z₀ — coincides for ±z₀)
z₂ = z₁² + c + p·z₀     (carries the ODD term p·z₀)
```

so `z₂(z₀) ≠ z₂(−z₀)` whenever `p·z₀ ≠ 0`, and `escape(z₀) ≠ escape(−z₀)` in general.
Measured on a 401² grid at the classic Ushiki spot (`c=0.5667, p=−0.5`, `z_{-1}=0`):
**6904/160801 ≈ 4.3%** of pixels disagree under `z₀ → −z₀`. There is no 180° point
symmetry. (The spec author was likely picturing the textbook Phoenix, which uses a
different pixel-injection convention — e.g. pixel into `z_{-1}` with `z₀=0`, or a
transposed render.)

**What symmetry we actually have: real-axis reflection `Im → −Im`.** The recurrence
`z² + c + p·z_{-1}` has real coefficients, so when `c, p, z_{-1}` are all real,
`orbit(conj z₀) = conj(orbit z₀)` **bit-for-bit** in IEEE-754 (conjugation only flips
the imaginary sign bit; `z²`, `+c`, `+p·z_{-1}` all commute with it). Measured: **0/160801**
mismatches under `Im → −Im`. This holds for any real `z_{-1}` (not just 0).

**The guard we ship** (`render_modes::tests::phoenix_z_m1_symmetry_guard`), matching
reality and preserving the spec's *intent* (prove `z_{-1}` is load-bearing):

1. `z_{-1}=0`, real `(c,p)`: real-axis reflection is **exact** (escaped flag + smooth
   bits bit-identical across a grid).
2. `z_{-1}` with a **non-zero imaginary part** breaks that reflection (>0 pixels flip).
3. `z_{-1}` real but non-zero is still **load-bearing**: it changes the render vs
   `z_{-1}=0` (>0 pixels flip) even though it preserves reflection. This (3) is the
   real anti-re-pinning gate — *any* non-zero `z_{-1}` moves pixels, so a silent
   re-pin to 0 fails the test.

**Consequence for the sampler.** `z_{-1}` genuinely expands the repertoire (not a
no-op), so it stays a first-class proposal axis per the spec. The diversity argument
is unchanged; only the *name* of the broken symmetry changes (real-axis reflection,
not 180° rotation). Sampling a **non-real** `z_{-1}` is what buys the largest visual
departure (it breaks the reflection that all-real parameters preserve).
