# Fractal Generator — Project Handoff

**Purpose of this document.** This is the consolidated context for a fractal-background generation project. It is written to start a fresh chat. **Your first task in that chat is to write the first Claude Code prompt** (see "Prompt sequencing" → Prompt 1 at the end), deliver it as a `.md` file, then stop and wait for results. Do not write ahead. The rest of this document is the reasoning and decisions behind that prompt so you can write it well and answer follow-ups.

---

## 1. What we're building and why

An engine that **automatically generates new, strong fractal images with beautiful palettes**, en masse, under strict quality gates — most candidates are expected to fail aesthetic muster, and the human (Matt) manually selects favorites from the survivors. Mentally this is **rejection sampling with hard quality gates plus human final selection**.

Two things to keep in proportion:

- **Palettes are the thing Matt cares about most.** The color treatment is first-class, not an afterthought.
- **The corpus is a minor assist.** Matt has ~700 existing fractal backgrounds (some "raw" Mandelbrots — a good location with a good palette; some more complex compositions). They serve as a light soft-prior for a few target ranges, *not* as the centerpiece. The generation engine is the spine; the corpus bolts on later.

**Fractal scope:** focus on **orbit-trap fractals** in the Mandelbrot / Julia family. Structure the code so other fractal types (Burning Ship, Newton, etc.) can be added later, but don't build them now.

**Two compositional modes** were observed in reference images and both are targets:
- **Anchored-black:** a near-boundary view where the black set sits in a corner as deliberate negative space (~20–30% of frame).
- **No-black deep zoom:** a structure-dense region (e.g. an embedded-Julia spiral) where the interior is essentially absent from the frame (~0%).

The interior ("black region") is therefore a **compositional variable with a target distribution**, not something to minimize globally.

---

## 2. Language & architecture decisions

- **Rust** for rendering and for pixels→features (rayon for parallelism, SIMD inner loop). **Python** for statistics, modeling, clustering, plotting, and selection tooling.
- **Shared feature extractor in Rust** so the *identical* code measures the corpus (Phase 1) and freshly rendered candidates (Phase 2). No risk of two implementations drifting.
- **Pure-Rust bignum** (`astro-float` or `dashu`) for the high-precision reference orbit — **no C dependency.** Rationale is empirical: the high-precision work is a single reference orbit costing well under 0.5% of render time (measured 0.13–0.24 s even at 200 digits / 30k iterations, vs. tens of seconds for the pixel pass). GMP/`rug` would buy essentially nothing. "Easiest, no C dep" wins because the speed delta lives in a negligible part of the runtime.
- **Precision behind a backend trait** so a deeper tier (floatexp / rescaled iterations, plus BLA) can slot in later without touching the search or coloring code.
- Repo: **`C:\Code\fractal-generator`** (Claude Code is launched inside this git repo).

---

## 3. Precision — findings and the v1 cap

This was tested empirically (Python + mpmath prototype) and the conclusions are firm.

**f64's clean limit is resolution-dependent.** It's governed by pixel spacing (`width / output_width`) versus `eps · |c|` ≈ 1.7e-16, *not* a fixed zoom depth. So higher output resolution breaks earlier:

| output width | f64 clean to ~magnification | breaks below frame width |
|---|---|---|
| 360 px | 1.7e13 | 6.0e-14 |
| 1024 px | 5.8e12 | 1.7e-13 |
| 1920 px | 3.1e12 | 3.2e-13 |
| 3840 px | 1.6e12 | 6.4e-13 |

So at production resolution, **f64 is trustworthy only to ~1e12 magnification.** Past that it produces stair-step coordinate quantization (verified visually).

**Perturbation theory** is the next tier and was validated correct (median difference 2e-13 iterations vs. an mpmath ground-truth orbit, with exact interior/exterior classification):
- Compute **one high-precision reference orbit** at the frame center. The center coordinate needs many digits, but the orbit *values* stay O(1) until escape, so they're stored as `f64`/`complex128`. Every other pixel is computed as a low-precision **delta** relative to that reference, in plain f64.
- **Rebasing (Zhuoran's method)** for glitch-free single-reference rendering: when `|Z_m + z| < |z|`, replace `z := Z_m + z` and reset the reference index `m := 0`. This *avoids* glitches rather than detecting/correcting them (skip Pauldelbrot-style detection entirely). The per-pixel iteration count is unaffected by rebasing; `m` is just the reference alignment.

**v1 cap: ~1e300 magnification.** That's where plain f64 deltas underflow (deltas stay normal down to frame width ~1e-305). This is ~250 orders past f64 and far beyond what backgrounds need. **We detect "too deep"** via (a) a width-threshold guard and (b) a per-pixel underflow flag (`|z|` collapses to 0 while the pixel's `|dc|` > 0). When tripped, refuse or flag rather than render garbage.

**Deferred (not v1):** floatexp / rescaled iterations to extend *range* past 1e300; BLA (bivariate linear approximation) to *skip* iterations for speed (the modern replacement for series approximation — simpler, parallelizable, generalizes to other formulas). Reference implementation and theory: Claude Heiland-Allen's **Fraktaler 3** and his mathr blog "Deep zoom theory and practice (again)."

---

## 4. Coloring decisions

The structural decision that matters most: **separate iteration from coloring.** The per-pixel iteration emits a small record — smooth escape value, distance estimate, orbit-trap minimum (and where on the trap it was hit). A **separable coloring stage** then maps any combination of those channels through a palette. This makes re-coloring cheap (crucial, since palettes are the priority) and keeps escape-time / DE / orbit-trap blends as configuration rather than code.

- **Smooth (normalized) iteration count** is the default and eliminates level-set banding (verified). Formula: `nu = n + 1 − log₂(log|z|)`, with a large bailout (≈1e6) for accuracy.
- **Distance estimation** works and is useful three ways — crisp boundary filaments, an antialiasing aid, and a search signal. `DE = |z|·ln|z| / |dz|`, with the derivative updated `dz := 2·z·dz + 1`.
- **Orbit traps are the primary coloring focus.** Color by the minimum distance the orbit approaches a geometric trap (point / cross / circle to start, extensible). Three distinct aesthetics were demonstrated (pearled beads / thorny organic / overlapping scales). Critically, **orbit traps also color the interior** (a non-escaping orbit still has a trap minimum), so they solve the "dead black" problem.
- **Antialiasing is mandatory, not optional.** Busy fractal regions produce heavy sub-pixel salt-and-pepper aliasing; supersampling (or DE-based AA) must be present from the start, not bolted on.

**Interior ("black region") treatment** — support all three and sample per render for style diversity (Matt's call: ~1/3 each):
1. Keep black, used as a deliberate anchor, with controlled area fraction.
2. Orbit-trap-filled interior (no dead black).
3. Deep frame with ~0% interior.

---

## 5. Palette system (first-class)

- **Representation:** a continuous, cyclic gradient — control points interpolated in **OKLab** (perceptually even spacing, which matters for mapping smooth escape values). Render-time parameters: cycle period/density, offset/rotation, direction.
- **Mapping:** smooth value (or DE / trap value) → `(value · density + offset) mod 1` → gradient lookup.
- **Sources to harvest** (forward task, not v1): UltraFractal public formula database (full pack), the classic `Maps01–11.ugr` set, DeviantArt gradient packs (e.g. Velvet--Glove), jwfsanctuary.club, Apophysis (`.ugr`-compatible). Plus zero-restriction generators: **cubehelix** and the scientific/perceptual colormaps.
- **File formats** (both trivially parseable; a working `.ugr` parser already exists in the prototype):
  - `.ugr` (UltraFractal) — plain text, named blocks of `index=N color=INT` pairs, index range 0–400, color is COLORREF-style `0x00BBGGRR` (R = low byte).
  - `.map` (Fractint) — even simpler, 256 lines of `R G B`.
- **Licensing (not legal advice):** UF public-DB files are copyright their respective authors; the norm is *free to use and to modify for your own use, but not to redistribute without permission*. DeviantArt packs match (e.g. "free to use in your art, no credit required, don't redistribute or claim"). Matt's workflow — parse → render his own fractals → hand-pick backgrounds — is squarely the intended "use," and a gradient is a thin-copyright list of colors regardless. The thing to avoid is **redistributing the `.ugr`/`.map` collections themselves**. For any output that must be clean of all restriction (publishing/selling), prefer cubehelix / scientific maps / corpus-extracted palettes, which carry none.
- **Key review artifact:** decouple location from palette. Find a strong location once (expensive), then render it as a **contact sheet of N palettes** (cheap). Given palettes are the priority, this one-location-×-many-palettes sheet is likely the main thing Matt selects from.

---

## 6. The descent search (generation) — design, not yet built

Parameter selection is **iterative-deepening best-first descent**, not a single boundary sample: render at fixed resolution, score sub-windows for "interestingness," recurse into the best, repeat to target depth.

- **Beam + backtracking, not greedy.** Greedy single-path descent dead-ends into pure black or self-similar tedium. Keep a frontier of top-k windows across the tree; backtrack when a branch collapses.
- **Navigate toward features, not just "the boundary."** The prettiest deep spots are **embedded Julia sets** and the neighborhoods of **minibrots**. Find these deterministically: **Newton's method for nucleus finding** (locate the period-n minibrot center) plus **atom-domain** size estimates to know where periodic structure lives and how deep to frame it. This is far higher-yield than biasing toward small distance-estimate pixels at random.
- **Per-window interest score (to be designed; will iterate):** a blend of escape-time entropy/range (reject all-fast and all-interior), boundary length via DE, and a **black-fraction-in-target-band** term (target a band, don't minimize). The corpus contributes only the target bands and a busyness range.
- **Mandelbrot → Julia for free:** sampling `c` near the Mandelbrot boundary yields good (connected) Julia sets by construction. Julias sometimes at the base symmetric level, sometimes zoomed.

**This is the main open design question still to resolve** (the interest-score details). It is *not* needed for the first few Claude Code prompts and can be deferred.

---

## 7. Quality gates, selection, corpus role (later phases)

- **Cascade cheap→expensive:** render a small thumbnail → apply cheap hard gates (intensity-variance floor, near-black/near-white fraction, boundary-structure fraction) → only render full-res for survivors. Most feature extraction also happens on thumbnails.
- **Don't maximize corpus likelihood** — that regresses to the bland center of the distribution. Treat corpus-fit as a *gate* ("is this in the plausible region at all?") and sample across the support including tails; the human eye does the final aesthetic cut.
- **Diversity-aware final selection:** cluster survivors and sample across clusters (or farthest-point sampling) so review isn't 50 variants of one swirl.
- **Log manual selections from day one.** They're labeled data; eventually a small learned scorer (logistic regression on the feature vector, or a tiny CNN — same shape as the still_extractor face classifier) can replace hand-tuned gates. Not v1, but build the harness so selections are recorded.
- **Corpus feature extractor (Rust, shared):** emits a per-image feature sidecar (JSON); Python does the stats. Features: palette (k-means in OKLab, hue circular statistics), busyness (FFT power-spectrum slope ~1/f^α, compressed-file-size proxy, edge density / local entropy), composition (radial detail distribution, symmetry), and black fraction. Expect the "raw" vs "complex" images to form two clusters — model separately or just as soft target bands.

---

## 8. Output, validation, and working conventions

- **Output is fully resolution/aspect parametric** with high-res defaults — no canonical size baked in. (Reference corpus spans 2:1 up to 10240×5120 and 3:2 around 3300×2300.)
- **Validation pattern:** the f64 renderer is the ground-truth reference for perturbation. Shallow renders from both must match; at depth, perturbation must stay clean where f64 quantizes. (This is exactly how perturbation was validated in the prototype.)
- **Performance note / don't be misled:** the prototype timings are slow single-threaded Python with a Python-level iteration loop (~51 s for 320²/15k iters). Rust with rayon + SIMD should be ~50–200× faster, putting deep production-res renders in seconds. Don't anchor expectations on the Python numbers.
- **Matt's working conventions (apply to every Claude Code interaction):**
  - Claude Code prompts are delivered as `.md` files (in the outputs folder), never printed inline.
  - **Write one prompt, wait for results, then write the next.** Never write ahead.
  - Long-running steps must include a runtime estimate and a background-run instruction.
  - Debug with diagnosis-first prompts that read the actual current code before proposing changes.
  - Matt is expert (Stanford PhD, Adobe Research, ML + graphics). Be terse, precise, iterative; skip basics.

---

## 9. Prompt sequencing for Claude Code

Build the **render core** first and lock it before any search/aesthetic/corpus work. Everything downstream sits on this core.

1. **Minimal runnable f64 skeleton.** Escape-time Mandelbrot + smooth (normalized) coloring + supersampling AA + PNG output + a CLI taking center, frame width, maxiter, resolution, and aspect. One simple built-in palette is fine. This is the validation reference for everything after it. *(This is the prompt to write first.)*
2. **Perturbation + rebasing**, behind a precision backend trait; pure-Rust bignum reference orbit; the too-deep detection (width guard + underflow flag). Validate against Prompt 1 (shallow match) and show a clean deep render where f64 quantizes.
3. **Coloring channels:** distance estimation + orbit traps (point / cross / circle), the separable coloring stage, and the interior-treatment switch (the three modes).
4. **Palette system:** `.ugr` / `.map` loader, OKLab gradient representation, cyclic density/offset parameters, and the one-location-×-N-palettes contact-sheet output.

*Later, beyond this initial sequence:* the descent search (with Newton nucleus-finding); the corpus feature extractor + stats; quality gates + selection UI + selection logging; and floatexp/BLA if deeper zoom is ever wanted.

---

## 10. Immediate task for the new chat

Write **Claude Code Prompt 1** (the f64 skeleton above) as a `.md` file for `C:\Code\fractal-generator`. Keep it scoped to a runnable, validatable skeleton — clean module boundaries that anticipate the precision backend trait and the separable coloring stage, but no perturbation, no orbit traps, no palette files yet. Then stop and wait for results before writing Prompt 2.
