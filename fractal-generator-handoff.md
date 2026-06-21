# Fractal Generator — Handoff

A from-scratch Rust engine that searches the Mandelbrot set for wallpaper-grade deep-zoom locations, renders them (Mandelbrot + matching Julia), and colors them through a first-class palette system. Goal: an automated pipeline that surfaces *many* strong candidates a human selects from. Repo: `C:\Code\fractal-generator`. This doc is the single source of truth for a fresh session — it reflects what is **actually built** (Prompts 1–9), the findings that cost real effort to discover, and the current open problem.

---

## Environment & working conventions

- **OS:** Windows, PowerShell. Long renders run in the **background** (`Start-Job`, or redirect to a log + poll). Always give a runtime estimate for multi-minute steps.
- **Stack:** pure-Rust, dependency-light by deliberate ethos. `astro-float 0.9.5` (arbitrary-precision float; `default-features=false`, `features=["std"]`), `num-complex`, `rayon`, `clap`, `image` (png+jpeg+webp enabled). **Hand-rolled JSON** and a **SplitMix64** RNG — no `serde`, no `rand`. (astro-float has no public `to_f64`; it's reconstructed from the raw mantissa/exponent.)
- **Prompt discipline (how this project is driven):** one `.md` Claude Code prompt at a time, written to `/mnt/user-data/outputs/`, never inline. Wait for results before writing the next. Diagnosis-first; read the actual current code before editing. **Validate primitives in isolation before any expensive run.** Matt commits/pushes constantly — assume everything is committed.
- **Collaboration style:** terse, iterative, precise technical detail. Matt judges all visual quality — the agent presents artifacts and data, makes **no** quality claims.

## Subcommands & modules

Subcommands: `render` (+ `--julia`), `descend`, `navigate`, `search`, `corpus`, `wallpaper`.
Modules: `backend`, `render`, `coloring`, `palette`, `palette_io`, `sheet` (`compose_grid`), `probe` (shared filmstrip/Julia/hp-accum/JSON/label/RNG/circle), `descend`, `navigate`, `search`, `corpus`, `wallpaper`, `font` (5×7 ASCII bitmap), `cli`, `main`, `lib`.

---

## Architecture as built

### Render core (Prompts 1–4)

**Precision backends behind a trait.** f64 is clean while pixel spacing `frame_width/render_width > ~1e-13`; below that, perturbation engages (auto-selected by spacing; `--backend f64|perturb|auto`). Hard width guard refuses `frame_width < 1e-300` (the v1 cap, ~1e300 magnification).

```rust
pub trait FractalBackend: Sync { fn sample(&self, c: Complex<f64>, dc: Complex<f64>) -> PixelSample; }
F64Backend::new(maxiter, bailout)
PerturbationBackend::new(center_re,center_im: &BigFloat, maxiter, bailout, prec_bits)  // builds+stores ref orbit
JuliaBackend                                                                            // f64, z0=pixel, fixed c
```

- **`dc` is computed straight from pixel geometry** (`(px_frac−0.5)·fw`, …), never `c−center` (that's catastrophic cancellation at depth). `c = center+dc` is formed only for the f64 backend.
- **Perturbation:** single high-precision reference orbit at frame center (bignum, stored as `Vec<Complex<f64>>` — values stay O(1)). Per-pixel delta `δ = (2·Z[m]+δ)·δ + dc`; **Zhuoran rebasing** (`if |z|²<|δ|² or m≥L−1 { δ=z; m=0 }`). Full value `z = Z[m]+δ` drives escape/coloring. Reference orbit is <0.5% of render time. `prec_bits = ceil(log2(render_width/frame_width)) + 64`. Center passed as **arbitrary-precision decimal strings**. `glitched` flag on per-pixel underflow.
- **Smooth iteration:** `nu = (n+1) − ln(ln|z|)/ln2` at **bailout 1e6**.
- **Separable coloring** (the spine): iteration produces a cached **supersampled `PixelSample` buffer**; `shade_and_downsample(...)` is a **pure** function over it — re-coloring never re-iterates. AA correctness: shade each subpixel, **then** average in **linear light**, then encode sRGB.

**Current `PixelSample` contract:**
```rust
pub struct PixelSample {
    pub escaped: bool,
    pub smooth_iter: f64,  // exterior only
    pub de: f64,           // exterior only; raw distance estimate (plane units); 0 interior; clamped 0 if dz non-finite
    pub trap_min: f64,     // ALL pixels incl. interior
    pub trap_phase: f64,   // normalized angle [0,1) at the trap minimum
    pub glitched: bool,    // perturbation underflow
    pub atom_period: u32,  // navigation: argmin_n |z_n|, absolute n, rebase-invariant
    pub atom_min: f64,     // navigation: min |z_n|
}
```
`de_px = de / pixel_spacing` (distance-to-boundary in pixels) is the key derived quantity — see the open problem.

**Channels:** distance estimation `DE = |z|·ln|z|/|dz|`, `dz=2·z·dz+1` (`dz_0=0`); in perturbation `dz` is carried on the **full value** `z` (continuous across rebasing). **Orbit traps** point/cross/circle, tracked from `n=1` on the full value, recording `trap_min`+`trap_phase`, interior pixels included.

**Coloring CLI:** `--color smooth|trap|de`, `--interior black|trap`, `--de-shade`, `--trap point|cross|circle`, `--trap-center`, `--trap-radius`, `--trap-curve linear|sqrt|log` (default `sqrt`). The "~0% interior" treatment is a *framing* property, not a coloring branch.

**Palette system:** continuous **cyclic OKLab gradient** (Ottosson matrices) baked to a **4096-entry linear-RGB LUT** (hot path is O(1)). Loaders: `.ugr` (named blocks, `index∈0..400 → pos=index/400`, `color` = COLORREF `0x00BBGGRR`, R=low byte) and `.map` (`R G B` lines, `pos=i/N`). Built-ins: `default`, `cubehelix` (generated), `viridis` (CC0), and `corpus` (`Palette::from_oklab_colors`, built from `targets.json`). `--palette name|path`, `--palette-entry`, `--palette-reverse`. **Contact sheet** = one location × N palettes: iterate once, shade N times (`compose_grid`, gradient swatch + 5×7 index, stdout legend). Sample assets in `assets/palettes/` (`sample.ugr` = Ember/Ocean/Viol; `sample.map`).

### Navigation (Prompts 5–7)

**`descend`** — greedy pixel descent, Mandelbrot|Julia filmstrip + JSON, hp center accumulation (`center += dc` in BigFloat), maxiter schedule `base + per_decade·log10(mag)` (1000, 1500). **Known bad** (see findings).

**Navigation primitives** (validated in isolation first):
- **Atom-domain** period = `argmin_{n≥1} |z_n|`; candidates = local minima of `atom_min` grouped by `atom_period` (free from the render loop; both backends, rebase-invariant absolute `n`).
- **Newton nucleus** = solve `z_p(c)=0` for the period-`p` minibrot center: iterate `z,dz` (`dz=2z·dz+1`) to `p`, `c −= z_p/dz_p`, in **BigFloat**; quadratic. (period-3 nucleus ≈ `−1.754877666…`.)
- **Size estimate** = **`1/(b·l²)`** (Munafo/Heiland-Allen), seed `b=1`, update `l` before `b`. ⚠️ The naive `1/(b·l)` is **8× too large** — this was caught only by empirically framing a known minibrot; verified against the period-2 disk (0.5 = diameter). Use `|size|` only (axis-aligned framing); `l` overflows f64 at high period.

**`navigate`** — single-path feature navigation: atom→Newton→size, adaptive zoom `width = |size|·multiple`, **normalized** busyness (÷maxiter). Beats greedy (targets real nuclei at natural scale) but **single-path converges into a period-doubling cascade** — re-selects the nucleus it came from, `c` pins.

**`search`** — **global best-first frontier** (max-heap by diversity-adjusted score) over a tree; best-first = **implicit backtracking** (a collapsed branch sinks; next pop is the best sibling). **Re-selection filter** (drop a child whose nucleus is within ~k·width of an **ancestor's** — the cascade fix) + diversity penalty `adjusted = score − λ·similarity` (do **not** penalize off-center / high-roff candidates — they're the branch diversity). Outputs: best-path strip + top-N **farthest-point-sampled** contact sheet + node-tree JSON. Wall-clock budget. Validated: explores high-roff siblings, kills the cascade (distinct periods), and **decouples depth from quality** (best spine sat mid-depth at mag 3.5e7 while the tree reached 6.5e20 cleanly).

### Corpus (Prompt 8)

**`corpus --dir C:\Users\techm\Desktop\Wallpapers\`** (top-level only, no recursion). "Looks fractal" heuristic on `edge_density`, `edge_spread` (CoV across 8×8 tiles), `flat_fraction`, and a **chroma-weighted OKLab a×b histogram** for color richness (dark-invariant — the key fix). 746→728 kept, 18 rejected (2.4%).

- **Color targets are EXACT and the real win:** corpus palette (blue/purple darks `#28232f`/`#040307`/`#353f74`, muted greens, amber accents), hue ≈240°, chroma `[0.028,0.117]`, contrast `[0.242,0.636]`, luminance `[0.249,0.638]`. Ready for the (unbuilt) palette-matching step.
- **Structural bands are weak image proxies** (native units via a fitted proxy→native scale): busyness `[0.061,0.318]`, interior_frac `[0,0.431]`, boundary `[0.099,0.224]`. **period** is labels-only (`[3,20000]` default — not recoverable from pixels).
- **Label transition:** `α = n_labels/(n_labels+k)` (k≈20) blends bootstrap→labeled native bands from `search.json` keep/discard. ⚠️ **interior_frac & boundary stay pinned at α=0** because `search.json` emits no native channel for them — labels can only correct busyness & period until that's fixed.
- `targets.json` written; `search` consumes it (`--targets`, default `./targets.json`) and **falls back to built-in constants** when absent or provenance is `default`.

### Cheap wallpaper probe (Prompt 9)

**`wallpaper`** — cheap **f64-only** descent (asserted; no perturbation), ranked by **corpus-band proximity** (rejects too-flat **and** too-noisy), hard-stopped at the f64 floor for the wallpaper width: `floor = wallpaper_width·1e-13·margin` (margin 4 → ~1e-9). Renders the **deepest** level once at 2560×1440 and reshades it into a coloring×palette matrix. Produces a low-res descent strip + JSON (hp strings for re-render).

---

## Validated findings (do not rediscover)

1. **Size estimate is `1/(b·l²)`**, not `1/(b·l)` (8× error). Always validate a navigation primitive by framing a *known* minibrot before trusting it.
2. **Perturbation `dz`/`δ` are carried on the full value `z` and are rebase-invariant** (like the iteration count `n`). `dc` from pixel geometry, never `c−center`.
3. **Deep-zoom trap degeneracy:** origin point/cross traps go flat at deep zoom (every pixel shares the early reference orbit; the trap min lands in the shared trajectory) **and** at any off-origin frame (the orbit never approaches origin). Circle / off-center traps resolve it. (This is why the Prompt 9 `trap` wallpapers were pure black — center ≈ (−0.029, 0.764).)
4. **Crude image metrics cannot separate dark-vivid fractals from non-fractals.** A dark Julia on black is metrically identical to a solid. Luminance-based entropy tossed the *most* fractal images at 17% reject; a chroma-weighted OKLab gate fixed it to 2.4%. Photos mostly pass — not separable without ML; documented, not chased.
5. **The score never distinguished intricacy from noise** — the root cause of the open problem below.
6. **Greedy pixel-descent is the wrong navigator** (dead-ends into self-similar / near-Misiurewicz tedium, c-saturates). Minibrot-targeted framing reaches coherent structure by construction.
7. **Depth ≠ quality.** Best-first search keeps strong locations at modest depth; chasing magnification chases noise.
8. **Throughput is render-bound** (~12 search expansions / 10 min). Mass generation will need a cheap→expensive thumbnail gate (deferred).

---

## ⭐ THE CENTRAL OPEN PROBLEM — high-zoom noise

The first full wallpaper (`wallpaper_smooth_corpus.png`, deepest cheap level, mag 1.07e9) was **unusable — a grey speckle field**. Resolved this session:

- **It is a SELECTION/LOCATION failure, not a coloring one.** No coloring saves it; trap/DE reshades would just re-color the same noise.
- **Mechanism: sub-pixel escape banding.** Across the frame `de_px ≪ 1` — the boundary is finer than one pixel — so escape times fluctuate chaotically pixel-to-pixel and smooth coloring renders pure grain. Corroborated by the **328 s** f64 render at only 14.5k maxiter: most pixels ran to `maxiter`, i.e. a near-interior chaotic field.
- **Root bug:** **busyness (std-dev of `smooth_iter`) measures value-spread, not spatial coherence.** Incoherent speckle scores "busy" identically to a beautiful smooth gradient. Every prior "rich region" claim lacked a metric that tells intricacy from noise.
- **The good structure WAS in frame** (a clean minibrot + dendrite bottom-right, decoration blobs left) — greedy framed the noise field *between* the good parts.
- **The fix (cheap, uses an existing channel):** a **coherence gate at selection time** = fraction of escaped pixels with `de_px < ~1`. A high fraction ⇒ sub-pixel boundary ⇒ guaranteed speckle ⇒ reject. DE is already computed; this is nearly free. This is the missing score term.

---

## Forward path

**NEXT (agreed): cheap-regime, minibrot-targeted framing with the DE-coherence gate built in.** Stay in f64 (cheap floor as in `wallpaper`). Use the Newton/atom-domain primitives to *frame coherent structure* (center the minibrot/embedded-Julia field) instead of greedy-busyness pointing at noise between features. Add the `de_px<1` coherence gate to candidate scoring; as a sanity check, report that the gate rejects the Prompt 9 noise frame and the flat strip levels. Drive toward one human-approved 2560×1440 wallpaper. **No quality claims — Matt judges the output.**

**Then, in roughly this order:**
- **Native `interior_frac` & `boundary` channels into `search.json`** (per-node aggregates the search already computes) — unlocks `α>0` for those corpus bands so labels can correct them.
- **Labeling harness** — a generated **static HTML review page** (grid of cached thumbnails, keep/discard toggles, exports `labels.json`) to produce labels at scale. *Deferred by Matt until candidates are good enough to be worth sorting* — do not build it to triage clear rejects.
- **Palette-matching step** — use the exact corpus color targets to pick palettes whose rendered output matches the folder aesthetic.
- **Cheap→expensive thumbnail generation gate** for mass-generation throughput.

**Deferred (not v1):** floatexp/rescaled iterations (range past ~1e300; also fixes `dz`/`l` overflow at high period), BLA / series approximation (iteration skipping for interactive deep zoom), rotated framing (size estimate gives orientation; render core is axis-aligned), top-N sheet undershoot (budget for ≥N+1 expansions).

## References

Heiland-Allen, *"Deep zoom theory and practice (again)"* (mathr blog) & **Fraktaler 3**; Munafo mu-ency *"Size Estimate"*; **Zhuoran** rebasing; **Ottosson** OKLab.

## Key paths / artifacts

`C:\Code\fractal-generator` · corpus `C:\Users\techm\Desktop\Wallpapers\` (top-level) · `targets.json` · `corpus_features.json` · `search.json` (no native interior/boundary yet) · `wallpaper.json` · `assets/palettes/{sample.ugr,sample.map}`. Deepest Prompt-9 frame for reference: center ≈ `(−0.029, 0.764)`, width `2.794e-9`, mag `1.07e9`, maxiter `14546` (its `wallpaper.json` hp strings re-render any treatment).
