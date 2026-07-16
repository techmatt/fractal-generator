# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust engine for generating orbit-trap Mandelbrot/Julia fractal images as wallpapers. The long-term goal is mass-generating strong fractals under quality gates with a human picking favorites — palettes are the first-class concern. The **render core** (precision backends, separable coloring, palette system) is settled; the active workstream is the corpus → label → classifier pipeline (see "Corpus & classifier pipeline" below). The early navigation/diagnostic probes (descend, navigate, search, buffet, wallpaper, and the energy-metric scoring experiments) were retired in the P2 subcommand cull once the guided-descend → present/enrich → label flow superseded them.

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
cargo run --release -- sheet --builtins "default cubehelix viridis" --output sheet.png
```

Long renders / descents should be backgrounded; release builds put deep production-res renders in seconds.

**Windows exe-lock note.** While a long run is executing, the OS file-locks
`target/release/fractal-generator.exe`, so a concurrent `cargo build --release`
fails with `Access is denied (os error 5)`. To build/iterate while a background
run holds the exe, build into an isolated target dir:
`CARGO_TARGET_DIR=target-test cargo build --release` (run
`target-test/release/fractal-generator.exe`). `cargo build --release --lib` also
compile-checks without touching the exe. `target-*/` is gitignored.

**Compile-time model + build lanes.** A full incremental `cargo build --release`
is **~100s** on this machine, and that cost is almost entirely codegen, not the
frontend: the `[profile.release]` in `Cargo.toml` is tuned for *render
throughput* (`opt-level=3`, `lto="thin"`, `codegen-units=1`), so every edit
re-optimizes the whole crate as a single LLVM unit with no parallelism. Measured
breakdown of one edit→rebuild: `cargo check --release` ≈ **3s** (type/borrow-check
only, no codegen) vs the full ~100s. Pick the lightest lane that answers your
question:
- **Inner loop — just "does it compile?":** `cargo check --release` (~3s) or
  `cargo build --release --lib`. This is the default during a refactor; reach for
  it before a full build. (Use `--release` check so it shares the release
  dependency artifacts already on disk — a bare `cargo check` rebuilds deps in the
  `dev` profile, ~15s the first time.)
- **Need a runnable binary fast (smoke-test, eyeball a render):**
  `cargo build --profile quick` (~16s incremental; `lto=false`,
  `codegen-units=16`). Binary lands at **`target/quick/fractal-generator.exe`**,
  not `target/release/`. Runtime is ~10-30% slower (the per-pixel kernel loses
  cross-module inlining), so it's for correctness/visual checks, **not** perf
  timing.
- **Production renders, batch reproducibility, perf timing:**
  `cargo build --release` (~100s). The `release` profile is load-bearing for
  render speed — don't relax it; the `quick` lane exists so you don't have to.

`cargo test` builds the test binaries under the `release`/`dev` profile you pass
it (`cargo test --release` reuses the release artifacts). The test suite itself
runs in seconds; the cost is the compile, so the same lane logic applies.

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
- `hp.rs` — high-precision scalar support (astro-float, pure Rust, no C dep). Decimal parse, `prec_bits` (precision sizing), fast `to_f64` projection (hand-rolled, bypasses the slow decimal formatter), and `to_decimal_string` for high-precision decimal serialization. Orbit arithmetic uses `RoundingMode::None` (f64 projection absorbs sub-ulp error); only the input parse rounds correctly.
- `palette.rs` — cyclic gradients interpolated in **OKLab**, baked once into a `LUT_SIZE`-entry linear-RGB LUT so `lookup_linear` (the only coloring contract) is O(1). Built-ins: `default` (Ultra Fractal), `cubehelix`, `viridis`.
- `palette_io.rs` — `.ugr` (UltraFractal, multi-block) and `.map` (Fractint) loaders → sRGB8 stops. The resolver dispatches a `--palette` spec: built-in name or path by extension.
- `sheet.rs` — contact sheet: iterate one location once, re-shade across N palettes (multi-block `.ugr` → one tile per block). Burns a swatch strip + index per tile.
- `font.rs` — hand-rolled bitmap font for on-image labels (no font crate).
- `energy.rs` — pixel-space corpus metric (the `calibrate` subcommand; the six scoring-experiment subcommands it once hosted were retired in the P2 cull). Per image: center-crop 16:9 → 2560×1440 → OKLab forward-diff edge-energy → per-area pooling at 4 scales (16/8/4/2) → frozen equal-count (quantile) histograms (`NBINS=12`/scale). Distance = `distance()` = Σ per-scale 1-D EMD (`emd1d` = Σ|CDF₁−CDF₂|). `calibrate` freezes the bins over the corpus and writes the artifact. **Reuse `Signature`/`FrozenBins`/`distance`/`emd1d`/`kmeans`/`occupancy`/`region_energies`/`tile_energy` — don't reimplement.** The artifact's per-image histograms are parseable (hand-rolled `parse_artifact`, still consumed by `generate`), so corpus-side experiments load 746 signatures from disk instead of re-running the slow 2560×1440 pass. k-means archetype centroids/membership are **not** persisted — recompute via `kmeans(..., seed)` (default seed 0 matches the calibrate cluster sheet).

## Validation pattern

The f64 backend is the **ground truth** for perturbation: shallow renders from both must match (`tests/perturbation.rs`, run at `maxiter = 300` where f64 orbits stay accurate — deeper, f64 is *not* valid ground truth at chaotic locations). Separability is enforced by tests that wrap a backend in a `sample()` counter and assert re-coloring never re-iterates (`tests/separability.rs`, `tests/sheet.rs`).

## Corpus & classifier pipeline

The active workstream (the render core above is "done enough"). Goal: a labeled
corpus that trains an aesthetic classifier across every generator version's output.

**The flow.**
`guided-descend` → `data/guided_descend/<run>/pool.jsonl` (one candidate location
per row: cx/cy/fw + idx + provenance) → **either** `present` (zoom/composition
batches) **or** `enrich` (v2-filtered center batches) → a batch under
`data/label_corpus/batches/<batch_id>/` (schema: `data/label_corpus/CORPUS_SCHEMA.md`)
→ label in `tools/viz/corpus_label.html` (exports `scores.json`, merged into
`images.jsonl` labels by `tools/corpus/merge_scores.py`) → `classifier/` trains by
**unioning every batch blind to provenance**.

**The label corpus contract** (full spec: `CORPUS_SCHEMA.md`). Each `images.jsonl`
row has three independent blocks. `render` is **version-invariant** — the identical
field set across all batches (`RENDER_KEYS` in `tools/corpus/corpus_common.py`),
cx/cy/fw as decimal strings, and is the *only* thing the classifier sees (it's a
pure function → `crops/<image_id>.jpg`, rebuildable via `present`/`render-one`).
`provenance` is **version-tagged**, free to differ/be null across batches
(`PROVENANCE_KEYS`); it feeds the bias loop only and **never enters training**.
`label.score ∈ {null,1,2,3}` (bad/okay/good); `null → value` is the ONE allowed
mutation anywhere in the store — a merge that would change a non-null score warns
and refuses.

**The classifier** (`classifier/`, pkg). Weights/metrics in
`data/classifier/{v2…v6}/` (gitignored under the `data/*` rule). v2+ is a CORN
**ordinal** head (K−1=2 rank-consistent logits) on
`mobilenetv4_conv_medium.e250_r384_in12k`. Deploy transform =
`classifier.data.Transform(train=False)`: the deterministic **1280×720 → 384×224
bicubic stretch + normalize** mirror of `present.rs`'s JPG path (no jitter/flips).
`model.score_from_logits` returns `Σ σ(logit_k)` ∈ [0,2] — the monotone rank score
used for AP. **P(not-bad) = σ(logit₀)** (= P(rank≥1) = P(label≥2)). Black-gate
parity with the Rust render path: accept iff `black_fraction < 0.30`
(`BLACK_THRESH`, strict `<`).

**The in-memory scoring bridge** (`enrich` subcommand, `src/enrich.rs`).
`enrich --mode score` iterates each guided-descend pool location once at the label
geometry, recolors under K seeded score-3 palettes, and streams each recolored RGB
frame to **stdout** as a raw record (16-byte LE header `idx,ki,w,h` then `w*h*3`
RGB bytes); `tools/corpus/enrich_score.py` reads the stream and scores every frame
with v2 through the exact deploy transform — so 10k+ scoring passes never write
crops to disk. Only the ~1.1k selected `(location, argmax-palette)` rows are
rendered to JPG (`enrich --mode render`, full ss4 Lanczos3 wallpaper quality).

## Conventions

> **Generated-output convention.** All generated artifacts — renders, strips, contact sheets, guided-descend/calibration JSON, logs, demo fixtures — are written under the single `out/` tree, never the repo root. The root holds only source, config, docs, and committed `assets/`. `out/` is gitignored (except `.gitkeep`), so the entire working corpus wipes with one `rm -r out/*` without touching anything tracked. **New subcommands MUST default their output under `out/<subcommand>/` and MUST NOT write to the repo root.**

The fixed base defaults are `out/renders/` (bare render) and `out/strips/` (sheet); every other subcommand writes under its own `out/<subcommand>/`. Use `crate::ensure_parent_dir(path)?` before any top-level `save`/`fs::write` so a no-flag default writes its dir on a fresh checkout.

> **Scratchpad is not a dependency tier.** `scratchpad/` is the canonical *disposable temp*
> dir (gitignored). **If a file is imported from outside `scratchpad/`, or it's the only
> thing that produces a durable artifact, it isn't scratch — promote it to `tools/` (or
> delete it).** Findings/analysis text goes to `docs/findings/`, committed. `scratchpad/`
> must never be on anyone's dependency path — nothing tracked may import from it or read a
> non-regenerable artifact out of it. (This rule exists because `scratchpad/visual_dup/embed.py`
> was load-bearing production code — the whole morph_clip dedup axis depended on it — living
> in a dir whose name said it didn't matter; it was never committed, vanished, and cost a
> formula sweep to recover.) **Tests and harnesses belong in the suite, not `scratchpad/`** —
> if it's worth running twice, it's worth committing (default suite for a normal test,
> `slow`-marked for an opt-in / destructive one). A test CI never runs and git never sees is a
> memory of a test, not a test. Mechanically checkable — the two greps below must both stay empty:
>
> ```bash
> # (a) nothing outside scratchpad imports a scratchpad module:
> grep -rn "import" --include="*.py" tools/ classifier/ src/ | grep -i scratchpad
> # (b) no scratchpad file writes a durable data/ artifact:
> grep -rnE "savez|write_text|open\([^)]*['\"]w" --include="*.py" scratchpad/ | grep -iE "data/|STORE"
> ```

> **Persistent-store convention (`data/`).** `out/` is *disposable* — anything that must survive `rm -r out/*` lives under `data/` instead (committed, NOT gitignored). Use this for **load-bearing artifacts that are part of a metric's definition** and that you don't want silently regenerated: e.g. `data/calibration/energy_calibration.json` (the `calibrate` frozen quantile bins + per-image histograms — see `energy::ARTIFACT_PATH`). Regenerable *views* (PNG sheets) stay in `out/`. When something reads such an artifact back, expose the default path as a `pub const` (e.g. `energy::ARTIFACT_PATH`) shared by writer and reader rather than re-deriving the string.

> **Adding a subcommand.** The per-subcommand `Args` struct (+ its `impl
> { resolved_* }` helpers) lives **in the subcommand's own module**, next to its
> `run_*` (the P0 `cli.rs` decomposition moved every struct out of `cli.rs`). Four
> edit sites: (1) the `#[derive(Args)]` struct in the subcommand's module (e.g.
> `EnrichArgs` in `src/enrich.rs`, `CalibrateArgs` in `src/energy.rs`),
> `use`-importing any shared groups it flattens from `cli`
> (`crate::cli::{LocationArgs, ShadeArgs, PaletteSelectArgs, BackendChoice,
> parse_complex}`); (2) a `Command` enum variant in `cli.rs` referencing it by path
> (`Enrich(crate::enrich::EnrichArgs)`); (3) `src/main.rs` `use` + dispatch arm;
> (4) `src/lib.rs` `pub mod`. **`cli.rs` keeps only** the shared cross-cutting types
> (`BackendChoice`, `LocationArgs`, `ShadeArgs`, `PaletteSelectArgs`, `Cli`,
> `Command`, `parse_complex`). New subcommands MUST default outputs under
> `out/<subcommand>/` (disposable) or `data/<subcommand>/` (load-bearing artifacts)
> — never the repo root. Keep `#[derive(Args)]`/`#[arg(...)]` attributes and all
> default values/flag names stable (batch reproducibility depends on them).

- Deps are kept minimal and pure-Rust (no C deps): clap, num-complex, rayon, image (png/jpeg/webp), astro-float. The JSON logs (guided-descend pool, generate manifest, calibration artifact) are hand-rolled rather than pulling in serde.
- **Max 4 workers for multiprocessing.** Any parallel/multiprocessing worker pool (Python `ThreadPoolExecutor`/`ProcessPoolExecutor` `max_workers`, subprocess fan-out, `WORKERS` constants, etc.) MUST cap at 4. Do not exceed 4 concurrent workers.
- Matt is expert (graphics + ML PhD) — be terse and precise; skip basics.
- Module docs (`//!`) carry the real design rationale; read them before changing a module.

## Python / uv

The Rust engine is the core; Python is the ML/analysis side (corpus tooling, the
aesthetic classifier, palette experiments). **Use `uv` for all Python, not bare
`python`/`pip`/conda.** The project env is declared in root `pyproject.toml` +
`uv.lock` (both committed); `.venv/` is gitignored and regenerable with `uv sync`.

- Run things with **`uv run python …`** (or `uv run <tool>`) — never the global
  `python` on PATH (that's base conda, no torch). Add deps with `uv add <pkg>`.
- **GPU stack:** torch is the **cu124** build (`torch==2.6.0`,
  `torchvision==0.21.0`) pulled from the `pytorch-cu124` index pinned in
  `pyproject.toml` — not PyPI's CPU default. CUDA runs on the local RTX 2060 SUPER
  (8 GB). `timm`, `scikit-learn`, `Pillow`, `numpy` round out the classifier stack.
- Versions are pinned to match the `video-to-photo` project so uv's global cache
  hardlinks the wheels (a full `uv sync` is seconds, no multi-GB torch download).
  Keep them in lockstep when bumping.
- The classifier lives in `classifier/`; its weights/metrics go to
  `data/classifier/v1/` (gitignored under the `data/*` rule — expected for
  weights/scratch).
