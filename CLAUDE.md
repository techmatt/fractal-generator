# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust engine for generating orbit-trap Mandelbrot/Julia fractal images as wallpapers. The long-term goal (see `fractal-generator-handoff.md`) is mass-generating strong fractals under quality gates with a human picking favorites — palettes are the first-class concern. Only the **render core** (precision backends, separable coloring, palette system) and a diagnostic **descent probe** are built so far; the real beam-search navigation, corpus feature extractor, and selection tooling are deferred.

## Commands

```bash
cargo build --release            # always build release — debug is ~50-200x slower for iteration
cargo test                       # all integration tests (tests/*.rs) + unit tests
cargo test --test perturbation   # one test file
cargo test to_f64_matches        # one test by name substring

# Single render (no subcommand → render one PNG):
cargo run --release -- --center-re -0.743643887 --center-im 0.131825904 \
  --frame-width 1e-6 --maxiter 2000 --width 1920 --output out.png

# Contact sheet: one location, many palettes, iterated once:
cargo run --release -- sheet --builtins "default cubehelix viridis" \
  --palettes assets/palettes/sample.ugr --output sheet.png

# Descent probe (greedy depth-falloff diagnostic, emits filmstrip + JSON):
cargo run --release -- descend --levels 20 --zoom 6 --output descend_strip.png
```

Long renders / descents should be backgrounded; release builds put deep production-res renders in seconds.

## Architecture

Two deliberate seams structure everything (`src/lib.rs` is the module root; `src/main.rs` is a thin CLI wrapper):

**1. Precision behind `backend::FractalBackend`.** The per-pixel `sample(c, dc)` loop is swappable without touching the render driver, coloring, or CLI. Backends are built **per frame** (maxiter/bailout in the constructor; perturbation also computes its reference orbit there). Tiers:
- `F64Backend` — plain f64 escape time. Fast, accurate only while pixel spacing stays clear of f64 epsilon (~1e-13 of |c|, i.e. ~1e12 magnification at production resolution).
- `PerturbationBackend` — single high-precision reference orbit at the frame center (stored as f64 projections, since orbit *values* stay O(1)) plus per-pixel f64 deltas with **Zhuoran rebasing**. Clean far past where f64 quantizes; v1 cap ~1e300 magnification (where f64 deltas underflow). Glitch detection is a per-pixel underflow flag, not Pauldelbrot detection.
- `JuliaBackend` — base-scale Julia (`z₀ = pixel`, fixed `c`); always shallow, so never needs perturbation. Intentionally skips DE (`de = 0`).

Backend auto-selection is by pixel spacing (`PERTURB_SPACING = 1e-13`); `--backend f64|perturb|auto` overrides.

**2. Separable coloring (`coloring::shade`): `PixelSample` → linear-RGB.** Iteration emits a small `PixelSample` record (smooth iter, DE, trap_min, trap_phase, escaped/glitched); coloring is a **pure** map over it, so **re-coloring never re-iterates**. This is what makes the contact sheet and palette experimentation cheap. Channel validity matters: `smooth_iter`/`de` are exterior-only; `trap_min`/`trap_phase` are valid for *every* pixel (interior included), which is how orbit traps fill the interior instead of dead black.

### The two-stage render (`render.rs`)
1. `iterate_samples` — runs the backend over the **supersampled** grid (rayon over rows), caches `Vec<PixelSample>` at SS resolution. The only stage that touches a backend.
2. `shade_and_downsample` — pure: shades each subpixel, averages **in linear light**, then sRGB-encodes. AA is mandatory and must stay correct under re-color, so colors (not pre-shade channel values) are averaged.

Memory: the SS buffer is ~48 B × out_w × out_h × ss² (~470 MB at 1920×1280 ss2). Keep large supersampled frames to modest resolution.

### Critical coordinate rule
**`dc` (a pixel's offset from frame center) is computed straight from pixel geometry, never as `c - center`.** At deep zoom `c_f64 - center_f64` is catastrophic cancellation. `dc` is O(frame_width) and accurate in f64 to ~1e-305; perturbation uses only `dc`. The absolute `c = center + dc` is formed solely for the f64 backend (shallow only). Centers are parsed as **arbitrary-precision decimal strings** (`--center-re`/`--center-im`) because an f64 center is meaningless at depth.

### Supporting modules
- `hp.rs` — high-precision scalar support (astro-float, pure Rust, no C dep). Decimal parse, `prec_bits` (precision sizing), fast `to_f64` projection (hand-rolled, bypasses the slow decimal formatter), and `to_decimal_string` for the descend JSON round-trip. Orbit arithmetic uses `RoundingMode::None` (f64 projection absorbs sub-ulp error); only the input parse rounds correctly.
- `palette.rs` — cyclic gradients interpolated in **OKLab**, baked once into a `LUT_SIZE`-entry linear-RGB LUT so `lookup_linear` (the only coloring contract) is O(1). Built-ins: `default` (Ultra Fractal), `cubehelix`, `viridis`.
- `palette_io.rs` — `.ugr` (UltraFractal, multi-block) and `.map` (Fractint) loaders → sRGB8 stops. The resolver dispatches a `--palette` spec: built-in name or path by extension.
- `sheet.rs` — contact sheet: iterate one location once, re-shade across N palettes (multi-block `.ugr` → one tile per block). Burns a swatch strip + index per tile.
- `descend.rs` — **diagnostic, deliberately naive greedy probe** (not the real search). Greedily scores K×K windows for interest (busyness × boundary × interior-fraction band), descends into the best, emits a Mandelbrot|Julia filmstrip + JSON log. Its collapse into self-similarity/minibrot interiors *is the signal* for where deep-zoom quality falls off; the real Newton/atom-domain navigation must beat it.
- `font.rs` — hand-rolled bitmap font for on-image labels (no font crate).
- `energy.rs` — pixel-space corpus metric (`calibrate` + `rescore` subcommands). Per image: center-crop 16:9 → 2560×1440 → OKLab forward-diff edge-energy → per-area pooling at 4 scales (16/8/4/2) → frozen equal-count (quantile) histograms (`NBINS=12`/scale). Distance = `distance()` = Σ per-scale 1-D EMD (`emd1d` = Σ|CDF₁−CDF₂|). `calibrate` freezes the bins over the corpus and writes the artifact; `rescore` is a **diagnosis-only** re-scorer of the buffet DEEP tiles under candidate rules. **Reuse `Signature`/`FrozenBins`/`distance`/`emd1d`/`kmeans` — don't reimplement.** The artifact's per-image histograms are parseable (hand-rolled `parse_artifact`), so corpus-side experiments load 746 signatures from disk instead of re-running the slow 2560×1440 pass. k-means archetype centroids/membership are **not** persisted — recompute via `kmeans(..., seed)` (default seed 0 matches the calibrate cluster sheet).
- `buffet.rs` — `buffet` subcommand: deliberately un-engineered visual-first sampler (3 sources × offset×scale grid, captions only, no scoring). Its `buffet.json` source-B DEEP tiles are the fixed known-answer candidate set that `calibrate`/`rescore` score against (okay = B1/B2/B4/B5, sparse = B0/B3). Tiles are re-rendered from `buffet.json` on demand (f64 cheap-regime), not stored as images.

## Validation pattern

The f64 backend is the **ground truth** for perturbation: shallow renders from both must match (`tests/perturbation.rs`, run at `maxiter = 300` where f64 orbits stay accurate — deeper, f64 is *not* valid ground truth at chaotic locations). Separability is enforced by tests that wrap a backend in a `sample()` counter and assert re-coloring never re-iterates (`tests/separability.rs`, `tests/sheet.rs`).

## Conventions

> **Generated-output convention.** All generated artifacts — renders, strips, contact sheets, descent/search/corpus/wallpaper JSON, logs, demo fixtures — are written under the single `out/` tree, never the repo root. The root holds only source, config, docs, and committed `assets/`. `out/` is gitignored (except `.gitkeep`), so the entire working corpus wipes with one `rm -r out/*` without touching anything tracked. **New subcommands MUST default their output under `out/<subcommand>/` and MUST NOT write to the repo root.**

The tree is `out/{renders,strips,search,corpus,wallpaper,demos}/`. Use `crate::ensure_parent_dir(path)?` before any top-level `save`/`fs::write` so a no-flag default writes its dir on a fresh checkout.

> **Persistent-store convention (`data/`).** `out/` is *disposable* — anything that must survive `rm -r out/*` lives under `data/` instead (committed, NOT gitignored). Use this for **load-bearing artifacts that are part of a metric's definition** and that you don't want silently regenerated: e.g. `data/calibration/energy_calibration.json` (the `calibrate` frozen quantile bins + per-image histograms — see `energy::ARTIFACT_PATH`). Regenerable *views* (PNG sheets) stay in `out/`. When something reads such an artifact back, expose the default path as a `pub const` (e.g. `energy::ARTIFACT_PATH`) shared by writer and reader rather than re-deriving the string.

- Deps are kept minimal and pure-Rust (no C deps): clap, num-complex, rayon, image (png only), astro-float. The descend JSON is hand-rolled rather than pulling in serde.
- Matt is expert (graphics + ML PhD) — be terse and precise; skip basics.
- Module docs (`//!`) carry the real design rationale; read them before changing a module.
