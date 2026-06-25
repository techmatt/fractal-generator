//! CLI parsing (clap derive) and resolution of aspect → output height.
//!
//! The default (subcommand-less) invocation is **render** — a single PNG, exactly
//! as before. The optional `sheet` subcommand renders one location across many
//! palettes. Location + shading flags are shared via flattened arg groups so the
//! two paths stay in sync.

use clap::{Args, Parser, Subcommand, ValueEnum};
use num_complex::Complex;

use crate::backend::TrapShape;
use crate::coloring::{ColorChannel, InteriorMode, TrapCurve};

/// Precision backend selection.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum BackendChoice {
    /// Plain f64 escape time (fast; accurate only at shallow depth).
    F64,
    /// Single-reference perturbation with rebasing (clean at deep zoom).
    Perturb,
    /// Pick automatically by pixel spacing.
    Auto,
}

/// Frame geometry, precision, and orbit-trap *shape* — everything that feeds
/// iteration. Shared by `render` and `sheet` (one iteration, identical setup).
#[derive(Args, Debug)]
pub struct LocationArgs {
    /// Frame center, real part — arbitrary-precision decimal string (an f64
    /// center is meaningless at depth, so this is parsed at full precision).
    #[arg(long, default_value = "-0.5", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal string.
    #[arg(long, default_value = "0.0", allow_hyphen_values = true)]
    pub center_im: String,

    /// Width of the view in the complex plane.
    #[arg(long, default_value_t = 3.0)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Output image width in pixels.
    #[arg(long, default_value_t = 1920)]
    pub width: u32,

    /// Output image height in pixels. Overrides --aspect if given.
    #[arg(long)]
    pub height: Option<u32>,

    /// Aspect ratio as W:H (used when --height is absent).
    #[arg(long, default_value = "3:2")]
    pub aspect: String,

    /// Linear supersampling factor (S×S box downsample).
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Orbit-trap shape.
    #[arg(long, value_enum, default_value_t = TrapShape::Point)]
    pub trap: TrapShape,

    /// Orbit-trap center as `re,im`.
    #[arg(long, default_value = "0,0")]
    pub trap_center: String,

    /// Orbit-trap radius (circle trap only).
    #[arg(long, default_value_t = 1.0)]
    pub trap_radius: f64,

    /// Precision backend: f64, perturb, or auto (default).
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Render a Julia set instead of the Mandelbrot: `z₀ = pixel`, fixed
    /// parameter `c = (--param-re, --param-im)`. The frame (`--center-*`,
    /// `--frame-width`) is the *view*; for the whole set use center `0` and
    /// width ~3.5. This is the wallpaper re-render path for a descend target.
    #[arg(long, default_value_t = false)]
    pub julia: bool,

    /// Julia parameter `c`, real part — arbitrary-precision decimal (projected
    /// to f64). Ignored unless `--julia`.
    #[arg(long, default_value = "0", allow_hyphen_values = true)]
    pub param_re: String,

    /// Julia parameter `c`, imaginary part — arbitrary-precision decimal.
    /// Ignored unless `--julia`.
    #[arg(long, default_value = "0", allow_hyphen_values = true)]
    pub param_im: String,
}

/// Channel → gradient mapping. Shared by `render` and `sheet` (the sheet applies
/// the same shading to every tile, varying only the palette).
#[derive(Args, Debug)]
pub struct ShadeArgs {
    /// Gradient cycles per unit of the mapped channel value.
    #[arg(long, default_value_t = 0.025)]
    pub density: f64,

    /// Gradient phase offset / rotation in [0,1).
    #[arg(long, default_value_t = 0.0)]
    pub offset: f64,

    /// Primary exterior coloring channel.
    #[arg(long, value_enum, default_value_t = ColorChannel::Smooth)]
    pub color: ColorChannel,

    /// Interior (non-escaping) pixel treatment.
    #[arg(long, value_enum, default_value_t = InteriorMode::Black)]
    pub interior: InteriorMode,

    /// Multiplier applied to the curved trap minimum before mapping (trap channel).
    #[arg(long, default_value_t = 1.0)]
    pub trap_scale: f64,

    /// Curve applied to trap_min before scaling (trap headroom).
    #[arg(long, value_enum, default_value_t = TrapCurve::Sqrt)]
    pub trap_curve: TrapCurve,

    /// Weight of trap phase added as a secondary hue offset (0 = unused).
    #[arg(long, default_value_t = 0.0)]
    pub trap_phase_strength: f64,

    /// DE-shade: brighten thin boundary filaments. Bare flag uses strength 1.0;
    /// pass a value to tune. Omit to disable.
    #[arg(long, num_args = 0..=1, default_missing_value = "1.0")]
    pub de_shade: Option<f64>,

    /// Paint per-pixel glitched (delta-underflow) pixels magenta for diagnosis.
    #[arg(long, default_value_t = false)]
    pub mark_glitches: bool,
}

/// Palette selection shared shape (reverse applies in both modes).
#[derive(Args, Debug)]
pub struct PaletteSelectArgs {
    /// Built-in name (`default`, `cubehelix`, `viridis`) or a path to `.ugr`/`.map`.
    #[arg(long, default_value = "default")]
    pub palette: String,

    /// For a multi-block `.ugr`, the block to use (default: first).
    #[arg(long)]
    pub palette_entry: Option<String>,

    /// Reverse the gradient direction.
    #[arg(long, default_value_t = false)]
    pub palette_reverse: bool,
}

/// Top-level CLI. No subcommand → render (existing behavior).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub location: LocationArgs,

    #[command(flatten)]
    pub shade: ShadeArgs,

    #[command(flatten)]
    pub palette: PaletteSelectArgs,

    /// Output PNG path.
    #[arg(long, default_value = "out/renders/out.png")]
    pub output: String,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// One location × N palettes → a single grid PNG (iterates once).
    Sheet(crate::sheet::SheetArgs),
    /// Discovery location generator (promotes probe 1): draw → cheap screen →
    /// accept band → keep, run to a target keeper count, persisting the full
    /// per-image log vector (`locations.jsonl`) + a run manifest under
    /// `data/generated/<run>/`, plus an annotated keeper contact sheet. Emits
    /// located, logged, single-palette-preview keepers only (3-palette labeling
    /// is a downstream stage).
    Generate(crate::generate::GenerateArgs),
    /// Presentation renderer: takes a `locations.jsonl` from a `generate` run,
    /// zooms in on each seed center, tries three composition offsets (center,
    /// thirds, golden) at cheap resolution, gates on black fraction < 40%, and
    /// renders the accepted composition at full resolution across random palettes.
    /// Emits per-crop PNGs, a contact sheet, and a manifest.json.
    Present(crate::present::PresentArgs),
    /// Stochastic guided descent: many decorrelated root-down walks to random
    /// depth, each step picking the next center by a probabilistic policy (mostly
    /// into a detected μ-focus). Every visited frame is a candidate; emits a
    /// candidate-pool sheet (by-walk ladders + flat grid) + `pool.jsonl` under
    /// `data/guided_descend/<run>/`. Geometric policies only — no CNN, no dedup,
    /// no prefix-sharing. Diagnosis-first; `generate` is left intact as the control.
    GuidedDescend(crate::guided_descend::GuidedDescendArgs),
    /// Deterministic feature navigation (atom-domain + Newton nuclei) filmstrip.
    Navigate(crate::navigate::NavigateArgs),
    /// Corpus feature extractor: decode a wallpaper folder, reject non-fractal
    /// outliers, extract exact color targets + proxy structural priors, and emit
    /// `targets.json` (bootstrap bands, optionally blended toward labeled picks).
    Corpus(crate::corpus::CorpusArgs),
    /// Cheap (f64-only) descent ranked by corpus-band proximity, hard-stopped at
    /// the f64 floor for the wallpaper resolution; emits a descent strip, one
    /// deepest-level wallpaper reshaded across a coloring×palette matrix, and a
    /// JSON log. Tests whether the corpus busyness band's upper bound rejects
    /// high-noise regions.
    Wallpaper(crate::wallpaper::WallpaperArgs),
    /// DE-coherence gate isolation probe: render one frame (f64) and report the
    /// `subpixel_frac` speckle indicator (escaped pixels with `de_px < θ`),
    /// `esc_frac`, and median `de_px`, with `de_px` pinned to the target
    /// wallpaper spacing. Validates the missing selection statistic in isolation.
    Cohere(crate::coherence::CohereArgs),
    /// Coverage-dominance scorer: render one frame f64 and report `coverage` (the
    /// fraction of the escaped frame at the magical few-pixels boundary spacing) plus
    /// the speckle/interior/busy gates; a band-sensitivity table and optional
    /// scale-sweep make it the discrimination + retune tool for the harvest.
    Cover(crate::coherence::CoverArgs),
    /// Corpus energy-histogram metric calibration + eye-check (Prompt
    /// corpus-energy-calibration). Computes a multi-scale OKLab edge-energy
    /// histogram for every corpus wallpaper, freezes equal-count bins per scale,
    /// stores each image's signature, and runs two eye-checks (corpus-internal NN
    /// pairs + buffet source-B DEEP ranking) plus a k-means archetype sheet. No
    /// descent, no search, no candidate scoring beyond the buffet eye-check.
    Calibrate(crate::energy::CalibrateArgs),
    /// Throwaway diagnostic: re-score the buffet source-B DEEP tiles against the
    /// persisted corpus calibration under five candidate scoring rules (nearest-k,
    /// nearest-archetype, global-centroid, tail-pruned nearest-k, two-sided density
    /// band) and print a PASS/FAIL table per rule. Loads everything from disk
    /// (persisted artifact + buffet histogram cache); the only render is a
    /// deterministic re-render of the fixed buffet set if its histograms aren't
    /// cached. No search, no descent, no new-location rendering, no winner picked.
    Rescore(crate::energy::RescoreArgs),
    /// Diagnosis-only: add non-sparse-but-bad over-busy/speckle controls to the
    /// known-answer set, quarantine the degenerate reference cluster (C4) from the
    /// typicality statistics, and re-score the survivor rules (R3 global-centroid,
    /// R5 density band, raw s16-bin0 scalar) against okay + sparse + the controls.
    /// Renders only the fixed control set (a known-answer set, same category as the
    /// buffet re-render). No search, no descent, no winner picked.
    Overbusy(crate::energy::OverbusyArgs),
    /// Diagnosis-only: score the 22-tile known-answer set under nearest-good-archetype
    /// (min EMD to the k centroids of the C4-quarantined corpus), swept over cluster
    /// granularity k ∈ {5,8,12,16}. Loads everything from disk (artifact + both
    /// histogram caches); renders nothing. Prints a per-k ranking + PASS/FAIL plus the
    /// control-match / straddle / sparse-survivor diagnostics. No winner picked.
    Archetype(crate::energy::ArchetypeArgs),
    /// Diagnosis-only adversarial anchor probe: tests the founding axiom
    /// ("good = resembles some real wallpaper") at the level of *individual* corpus
    /// members, not centroids. Calibrates the corpus 1-NN distance distribution
    /// (real wallpaper-to-wallpaper similarity), then finds each known-answer tile's
    /// nearest individual corpus wallpaper and the smallest intrinsic corpus-corpus
    /// pairs, rendering both as side-by-side montages for Matt's eye. EMD on cached
    /// histograms; re-renders only the fixed known-answer set (controls + buffet DEEP)
    /// for the montage images (flagged). Picks no pivot, wires nothing.
    Anchor(crate::energy::AnchorArgs),
    /// Diagnosis-only trivial corpus dedup: find descriptor-near corpus pairs
    /// (EMD < epsilon), confirm each as near-pixel-identical via a 16×16 gray
    /// thumbnail diff, union confirmed pairs into duplicate groups (keep the
    /// lexically-first member), and emit a drop-list plus the corpus 1-NN
    /// distribution before vs after the drop. Reads corpus PNGs + cached
    /// histograms only — no fractal renders. Does NOT mutate the artifact.
    Dedup(crate::energy::DedupArgs),
    /// Diagnosis-only palette-sweep muster: does a corpus-marginal density band
    /// filter the 22-tile known-answer set? Renders each fixed tile's iteration
    /// data ONCE, recolors across a legit palette sweep (+ random/flat degenerate
    /// controls), scores a two-sided busyness scalar (mean fine s16 edge energy,
    /// recovered from the frozen-bin histogram), places each recolor as a corpus
    /// percentile, and sweeps an accept band reporting okay-recall / speckle-leak /
    /// sparse-rejection. Marginal control only — no good-busy vs bad-busy split.
    /// Picks no band, builds no loop. Matt judges the eye-check sheets.
    Muster(crate::energy::MusterArgs),
    /// Locked wallpaper-render default: render ONE (location × palette) at the
    /// settled quality — grid ss4 + Lanczos-3 @ 2560×1440 — to a caller-chosen
    /// stable path, reporting iterate / filter / total wall-clock. An extract of
    /// the verified `aa-filter` f64 path (selective-mirror palette load, the
    /// `ss×`-scaled reconstruction filter); the locked defaults live here only, so
    /// the bare render path's fast-preview defaults are untouched. Shallow f64 by
    /// construction (asserted).
    RenderOne(crate::render_one::RenderOneArgs),
    /// Palette universality probe: pick N label-3 ("great") locations at random
    /// (fixed seed) from the `loose0_v3` labels+manifest, iterate each ONCE at the
    /// `render-one` quality path (grid ss4 + Lanczos-3), and recolor across the
    /// full score-3 palette pool — so universally-bad palettes (bad even on a
    /// proven-good structure) can be spotted and cut. Emits the JPG crops +
    /// `probe_index.json` (palettes carry their corpus not-bad rate, sorted
    /// worst-first) under `data/palette_probe/`. The viewer
    /// (`tools/viz/palette_probe.html`) writes the verdict; this picks nothing.
    PaletteProbe(crate::palette_probe::PaletteProbeArgs),
    /// v2-filtered enrichment batch render bridge (two disjoint modes). `score`
    /// iterates each guided-descend pool location once at the label geometry,
    /// applies the present gates (black<0.30 + occ>=0.321), recolors survivors
    /// under K seeded score-3 palettes, and streams the recolored RGB frames to
    /// stdout for in-memory v2 scoring (no crops to disk) + a `--meta-out` gate
    /// sidecar. `render` renders the ~1100 selected `(location, argmax palette)`
    /// rows at the locked label-crop quality (ss4 + Lanczos-3, q90 JPG). Shallow
    /// f64 (asserted). See `enrich::run_enrich`.
    Enrich(crate::enrich::EnrichArgs),
}

/// Parse a `re,im` pair into a complex number.
pub fn parse_complex(s: &str, what: &str) -> Result<Complex<f64>, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("invalid {what} '{s}', expected re,im"));
    }
    let re: f64 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("invalid {what} real part in '{s}'"))?;
    let im: f64 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("invalid {what} imaginary part in '{s}'"))?;
    Ok(Complex::new(re, im))
}

impl LocationArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        let parts: Vec<&str> = self.trap_center.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid --trap-center '{}', expected re,im",
                self.trap_center
            ));
        }
        let re: f64 = parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("invalid trap-center real part in '{}'", self.trap_center))?;
        let im: f64 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("invalid trap-center imaginary part in '{}'", self.trap_center))?;
        Ok(Complex::new(re, im))
    }

    /// Resolve the output height from `--height` or `--aspect`.
    pub fn resolved_height(&self) -> Result<u32, String> {
        if let Some(h) = self.height {
            if h == 0 {
                return Err("--height must be > 0".into());
            }
            return Ok(h);
        }
        let (wr, hr) = parse_aspect(&self.aspect)?;
        // height = width * (hr / wr), keeping pixels square.
        let h = (self.width as f64 * hr / wr).round() as u32;
        Ok(h.max(1))
    }
}

fn parse_aspect(s: &str) -> Result<(f64, f64), String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("invalid --aspect '{s}', expected W:H"));
    }
    let w: f64 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("invalid aspect width in '{s}'"))?;
    let h: f64 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("invalid aspect height in '{s}'"))?;
    if w <= 0.0 || h <= 0.0 {
        return Err(format!("aspect components must be positive in '{s}'"));
    }
    Ok((w, h))
}
