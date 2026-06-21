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
    Sheet(SheetArgs),
    /// Greedy Mandelbrot→Julia descent filmstrip + JSON (depth-falloff probe).
    Descend(DescendArgs),
    /// Deterministic feature navigation (atom-domain + Newton nuclei) filmstrip.
    Navigate(NavigateArgs),
    /// Best-first frontier search (beam + backtracking + diversity) over a tree
    /// of minibrot locations; emits a best-path strip, a top-N diversity sheet,
    /// and the full node tree as JSON.
    Search(SearchArgs),
    /// Corpus feature extractor: decode a wallpaper folder, reject non-fractal
    /// outliers, extract exact color targets + proxy structural priors, and emit
    /// `targets.json` (bootstrap bands, optionally blended toward labeled picks).
    Corpus(CorpusArgs),
    /// Cheap (f64-only) descent ranked by corpus-band proximity, hard-stopped at
    /// the f64 floor for the wallpaper resolution; emits a descent strip, one
    /// deepest-level wallpaper reshaded across a coloring×palette matrix, and a
    /// JSON log. Tests whether the corpus busyness band's upper bound rejects
    /// high-noise regions.
    Wallpaper(WallpaperArgs),
    /// DE-coherence gate isolation probe: render one frame (f64) and report the
    /// `subpixel_frac` speckle indicator (escaped pixels with `de_px < θ`),
    /// `esc_frac`, and median `de_px`, with `de_px` pinned to the target
    /// wallpaper spacing. Validates the missing selection statistic in isolation.
    Cohere(CohereArgs),
}

/// `wallpaper` subcommand: see the module docs in `wallpaper.rs`. Everything here
/// stays in the f64 regime by construction (the floor is sized so the deepest
/// level renders f64-clean at the wallpaper resolution).
#[derive(Args, Debug)]
pub struct WallpaperArgs {
    #[command(flatten)]
    pub shade: ShadeArgs,

    /// Per-level zoom factor (`width_{i+1} = width_i / zoom`).
    #[arg(long, default_value_t = 4.0)]
    pub zoom: f64,

    /// Start frame center as `re,im` (arbitrary-precision decimals).
    #[arg(long, default_value = "-0.5,0", allow_hyphen_values = true)]
    pub start_center: String,

    /// Start frame width in the complex plane.
    #[arg(long, default_value_t = 3.0)]
    pub start_width: f64,

    /// Final wallpaper width in pixels (height follows 16:9).
    #[arg(long, default_value_t = 2560)]
    pub wallpaper_width: u32,

    /// Low-res descent panel width in pixels (height follows 16:9).
    #[arg(long, default_value_t = 640)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S) for the descent panels and the wallpaper.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// f64-floor safety margin: deepest width must stay ≥ `wallpaper_width ·
    /// 1e-13 · margin`, keeping pixel spacing comfortably above f64's ~1e-13 limit.
    #[arg(long, default_value_t = 4.0)]
    pub margin: f64,

    /// Hard cap on descent levels (the floor normally stops it first).
    #[arg(long, default_value_t = 64)]
    pub max_levels: u32,

    /// Score window size K (K×K window over the feature map).
    #[arg(long, default_value_t = 5)]
    pub window: u32,

    /// RNG seed for sampling a target from each level's top in-band windows.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// maxiter schedule base: `maxiter = round(base + per_decade·log10(mag))`.
    #[arg(long, default_value_t = 1000.0)]
    pub maxiter_base: f64,

    /// maxiter schedule slope (iterations added per decade of magnification).
    #[arg(long, default_value_t = 1500.0)]
    pub per_decade: f64,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Orbit-trap shape (used by the `trap` coloring panel of the matrix).
    #[arg(long, value_enum, default_value_t = TrapShape::Point)]
    pub trap: TrapShape,

    /// Orbit-trap center as `re,im`.
    #[arg(long, default_value = "0,0")]
    pub trap_center: String,

    /// Orbit-trap radius (circle trap only).
    #[arg(long, default_value_t = 1.0)]
    pub trap_radius: f64,

    /// Corpus targets (`targets.json`): supplies the busyness band `[lo,hi]` used
    /// for ranking and the color block used to build the `corpus` palette.
    #[arg(long, default_value = "out/corpus/targets.json")]
    pub targets: String,

    /// Descent strip PNG. Per-level panels go in `<stem>_panels/`.
    #[arg(long, default_value = "out/strips/wallpaper_strip.png")]
    pub strip: String,

    /// Output prefix for the 6 wallpapers (`<prefix>_<coloring>_<palette>.png`).
    #[arg(long, default_value = "out/wallpaper/wallpaper")]
    pub out_prefix: String,

    /// JSON log path.
    #[arg(long, default_value = "out/wallpaper/wallpaper.json")]
    pub json: String,
}

/// `cohere` subcommand: isolation validation of the DE-coherence gate. Renders
/// one frame at a modest probe resolution (f64, asserted) and reports the
/// per-frame coherence statistic — `subpixel_frac` (escaped pixels with
/// `de_px < θ`, the speckle indicator), `esc_frac`, and the median `de_px`,
/// all with `de_px` pinned to the target wallpaper spacing so a cheap probe
/// predicts the full-resolution gate. Pure over the cached buffer; never
/// re-iterates. Does not modify any scoring (the wiring is the follow-up).
#[derive(Args, Debug)]
pub struct CohereArgs {
    /// Frame center, real part — arbitrary-precision decimal string.
    #[arg(long, default_value = "-0.5", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal string.
    #[arg(long, default_value = "0.0", allow_hyphen_values = true)]
    pub center_im: String,

    /// Frame width in the complex plane.
    #[arg(long, default_value_t = 3.0)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Probe render width in pixels (height follows 16:9). Cheap — only the
    /// sampled set changes with this; `de_px` is taken against `--target-width`.
    #[arg(long, default_value_t = 640)]
    pub probe_width: u32,

    /// Target wallpaper width in pixels — `de_px = de / (frame_width /
    /// target_width)`. Pins the gate to the final render's spacing (default the
    /// 2560-wide wallpaper) so the cheap probe is predictive.
    #[arg(long, default_value_t = 2560)]
    pub target_width: u32,

    /// Sub-pixel threshold θ: an escaped pixel with `de_px < θ` is speckle.
    #[arg(long, default_value_t = 1.0)]
    pub theta: f64,

    /// Linear supersampling factor (S×S) for the probe render.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Window size K (K×K) for the windowed-max busyness diagnostic (mirrors the
    /// selector's `--window`, so `busy_win` matches `wallpaper.json`'s
    /// `max_available_busyness`).
    #[arg(long, default_value_t = 5)]
    pub window: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Label for the printed data row / JSON (e.g. `noise`, `flat_L8`, `control`).
    #[arg(long, default_value = "frame")]
    pub label: String,

    /// Optional JSON sidecar path (one frame per file).
    #[arg(long)]
    pub json: Option<String>,
}

impl WallpaperArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Parse `--start-center` (`re,im`) into two decimal strings.
    pub fn resolved_start_center(&self) -> Result<(String, String), String> {
        let parts: Vec<&str> = self.start_center.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid --start-center '{}', expected re,im",
                self.start_center
            ));
        }
        Ok((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
}

/// `corpus` subcommand: bootstrap the search's target bands from a folder of
/// admired reference images, then (with `--labels`/`--search`) blend toward the
/// search's own labeled picks in native units. Exact color/aesthetic targets are
/// recovered directly from the images; structural bands from images are proxies
/// (dark-fraction ≈ interior, edge-density ≈ busyness) that labels later correct.
#[derive(Args, Debug)]
pub struct CorpusArgs {
    /// Folder of reference images (top level only — no recursion).
    #[arg(long)]
    pub dir: String,

    /// Optional `labels.json` (kept/discarded `search` node ids) — enables the
    /// transition from bootstrap proxy bands toward labeled native bands.
    #[arg(long)]
    pub labels: Option<String>,

    /// `search.json` providing the labeled nodes' native structural features.
    /// Required when `--labels` is given.
    #[arg(long)]
    pub search: Option<String>,

    /// Output `targets.json` path (the search consumes this).
    #[arg(long, default_value = "out/corpus/targets.json")]
    pub targets_out: String,

    /// Output per-image features JSON path.
    #[arg(long, default_value = "out/corpus/corpus_features.json")]
    pub features_out: String,

    /// Output rejected-thumbnails contact-sheet PNG (eyeball the ~5% tossed).
    #[arg(long, default_value = "out/corpus/corpus_rejected.png")]
    pub rejected_sheet: String,

    /// Longest-edge pixels to downscale each image to before feature extraction.
    #[arg(long, default_value_t = 1024)]
    pub max_edge: u32,

    /// Smoothing constant `k` in the label-blend weight `α = n/(n+k)`.
    #[arg(long, default_value_t = 20.0)]
    pub blend_k: f64,

    /// Uniform-reject edge floor: below this mean Sobel edge density AND below
    /// `--uniform-spread`, an image is a gradient/solid. Conservative — kept low
    /// so smooth flame fractals (also low-edge) are not swept up.
    #[arg(long, default_value_t = 0.010)]
    pub edge_min: f64,

    /// Uniform-reject spread ceiling: a gradient/solid has *evenly* low detail
    /// (low per-tile CoV); a smooth fractal still varies more. Gates the edge
    /// reject so the two don't get confused.
    #[arg(long, default_value_t = 0.30)]
    pub uniform_spread: f64,

    /// Dead-flat reject: near-flat 8×8 tile fraction above this AND a poor
    /// palette → a sparse non-fractal. High by default (rarely fires; a rich dark
    /// fractal on black also has many flat tiles, so the palette gate protects it).
    #[arg(long, default_value_t = 0.92)]
    pub flat_max: f64,

    /// Text/logo reject: per-tile edge-density CoV above this AND a poor palette →
    /// concentrated detail with few colors. High by default — high spread alone is
    /// unreliable (fractals concentrate detail too), so this rarely fires.
    #[arg(long, default_value_t = 1.60)]
    pub spread_max: f64,

    /// Color-poverty gate: reject if the effective OKLab color count
    /// (chroma-weighted) is below this — near-grayscale / solid. Also the palette
    /// gate (×2) that spares dark-but-vivid fractals from the structural rules.
    #[arg(long, default_value_t = 1.50)]
    pub entropy_min: f64,

    /// Comma-separated filename substrings to force-keep (overrides heuristic).
    #[arg(long, default_value = "")]
    pub include: String,

    /// Comma-separated filename substrings to force-reject (overrides heuristic).
    #[arg(long, default_value = "")]
    pub exclude: String,
}

/// `search` subcommand: a global best-first frontier over a tree of minibrot
/// locations. Each pop renders a frame, finds child minibrots (atom-domain →
/// Newton → size, the `navigate` primitives), filters re-selections (the
/// anti-cascade fix), and pushes diversity-adjusted children. Bounded by a
/// wall-clock budget. Outputs: a best-path filmstrip, a farthest-point-sampled
/// top-N contact sheet of diverse candidate locations, and `search.json`.
#[derive(Args, Debug)]
pub struct SearchArgs {
    #[command(flatten)]
    pub shade: ShadeArgs,

    #[command(flatten)]
    pub palette: PaletteSelectArgs,

    /// Wall-clock budget in seconds (the runtime knob; the search expands
    /// best-first until this elapses, then composes outputs).
    #[arg(long, default_value_t = 600.0)]
    pub time_budget: f64,

    /// Children kept (pushed to the frontier) per expanded node.
    #[arg(long, default_value_t = 6)]
    pub beam_width: usize,

    /// Diversity penalty λ in `adjusted = score − λ·similarity` (frontier
    /// priority). Larger → the frontier spreads across distinct families faster.
    #[arg(long, default_value_t = 0.15)]
    pub diversity: f64,

    /// Number of diverse high-scoring locations in the top-N contact sheet.
    #[arg(long, default_value_t = 12)]
    pub top_n: usize,

    /// Mandelbrot/Julia panel width in pixels (height follows 16:9).
    #[arg(long, default_value_t = 640)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S box downsample) for every panel.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Start frame center as `re,im` (arbitrary-precision decimals).
    #[arg(long, default_value = "-0.5,0", allow_hyphen_values = true)]
    pub start_center: String,

    /// Start frame width in the complex plane (the base set view).
    #[arg(long, default_value_t = 3.0)]
    pub start_width: f64,

    /// RNG seed for deterministic tie-breaks among near-equal priorities.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Frame width as a multiple of the chosen minibrot's `|size|` (the descend
    /// scale; also sets each child's frame).
    #[arg(long, default_value_t = 8.0)]
    pub frame_multiple: f64,

    /// Re-selection filter radius in child-frame-widths: drop a child whose
    /// nucleus is within `k·(child width)` of an ancestor's nucleus (the
    /// anti-cascade fix). Distinct off-position sub-minibrots are preserved.
    #[arg(long, default_value_t = 2.0)]
    pub reselect_k: f64,

    /// maxiter schedule base: `maxiter = round(base + per_decade·log10(mag))`.
    #[arg(long, default_value_t = 1000.0)]
    pub maxiter_base: f64,

    /// maxiter schedule slope (iterations added per decade of magnification).
    #[arg(long, default_value_t = 1500.0)]
    pub per_decade: f64,

    /// Hard cap: skip expanding a node whose scheduled maxiter exceeds this.
    #[arg(long, default_value_t = 250_000)]
    pub maxiter_ceiling: u32,

    /// Hard cap: don't descend into minibrots of higher period than this.
    #[arg(long, default_value_t = 100_000)]
    pub period_cap: u32,

    /// Fixed maxiter for every Julia panel (base-scale, shallow).
    #[arg(long, default_value_t = 3000)]
    pub julia_maxiter: u32,

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

    /// Precision backend: f64, perturb, or auto (default; switches per node).
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Best-path filmstrip PNG. Per-node panels go in `<stem>_panels/`.
    #[arg(long, default_value = "out/strips/search_strip.png")]
    pub strip: String,

    /// Top-N diversity contact-sheet PNG.
    #[arg(long, default_value = "out/strips/search_sheet.png")]
    pub sheet: String,

    /// Node-tree JSON output path.
    #[arg(long, default_value = "out/search/search.json")]
    pub json: String,

    /// Corpus-derived structural target bands (`corpus` subcommand output). When
    /// present, its busyness/period bands replace the hand-tuned score constants
    /// (per-band, only where provenance ≠ "default"); absent → current constants.
    #[arg(long, default_value = "out/corpus/targets.json")]
    pub targets: String,
}

impl SearchArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Parse `--start-center` (`re,im`) into two decimal strings (kept as
    /// strings for arbitrary-precision parsing downstream).
    pub fn resolved_start_center(&self) -> Result<(String, String), String> {
        let parts: Vec<&str> = self.start_center.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid --start-center '{}', expected re,im",
                self.start_center
            ));
        }
        Ok((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
}

/// `navigate` subcommand: deterministic single-path navigation toward minibrot
/// nuclei via atom-domain period detection, Newton refinement, and a size
/// estimate. Same filmstrip/JSON format as `descend` for a direct comparison;
/// the zoom is minibrot-driven (each minibrot framed at `|size|·frame_multiple`)
/// rather than a fixed factor.
#[derive(Args, Debug)]
pub struct NavigateArgs {
    #[command(flatten)]
    pub shade: ShadeArgs,

    #[command(flatten)]
    pub palette: PaletteSelectArgs,

    /// Number of navigation levels (each re-frames at the chosen nucleus).
    #[arg(long, default_value_t = 20)]
    pub levels: u32,

    /// Frame width as a multiple of the chosen minibrot's `|size|`.
    #[arg(long, default_value_t = 8.0)]
    pub frame_multiple: f64,

    /// Mandelbrot/Julia panel width in pixels (height follows 16:9).
    #[arg(long, default_value_t = 640)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S box downsample) for both panels.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Start frame center as `re,im` (arbitrary-precision decimals).
    #[arg(long, default_value = "-0.5,0", allow_hyphen_values = true)]
    pub start_center: String,

    /// Start frame width in the complex plane.
    #[arg(long, default_value_t = 3.0)]
    pub start_width: f64,

    /// RNG seed for tie-breaks among near-equal top candidates.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// maxiter schedule base: `maxiter = round(base + per_decade·log10(mag))`.
    #[arg(long, default_value_t = 1000.0)]
    pub maxiter_base: f64,

    /// maxiter schedule slope (iterations added per decade of magnification).
    #[arg(long, default_value_t = 1500.0)]
    pub per_decade: f64,

    /// Early-stop if the scheduled maxiter would exceed this ceiling.
    #[arg(long, default_value_t = 250_000)]
    pub maxiter_ceiling: u32,

    /// Early-stop if a chosen nucleus period exceeds this cap.
    #[arg(long, default_value_t = 100_000)]
    pub period_cap: u32,

    /// Fixed maxiter for every Julia panel (base-scale, shallow).
    #[arg(long, default_value_t = 3000)]
    pub julia_maxiter: u32,

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

    /// Precision backend: f64, perturb, or auto (default; switches per level).
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Output filmstrip PNG path. Per-level panels go in `<stem>_panels/`.
    #[arg(long, default_value = "out/strips/navigate_strip.png")]
    pub output: String,

    /// Output JSON log path.
    #[arg(long, default_value = "out/strips/navigate.json")]
    pub json: String,
}

impl NavigateArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Parse `--start-center` (`re,im`) into two decimal strings (kept as
    /// strings for arbitrary-precision parsing downstream).
    pub fn resolved_start_center(&self) -> Result<(String, String), String> {
        let parts: Vec<&str> = self.start_center.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid --start-center '{}', expected re,im",
                self.start_center
            ));
        }
        Ok((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
}

/// `descend` subcommand: greedy quality-scored descent emitting a tall
/// Mandelbrot|Julia filmstrip and a JSON log. A diagnostic for *where* deep-zoom
/// quality falls off (and a prototype of the per-window interest score the real
/// beam search will need) — deliberately the naive greedy baseline.
#[derive(Args, Debug)]
pub struct DescendArgs {
    #[command(flatten)]
    pub shade: ShadeArgs,

    #[command(flatten)]
    pub palette: PaletteSelectArgs,

    /// Number of descent levels (each zooms in by `--zoom`).
    #[arg(long, default_value_t = 20)]
    pub levels: u32,

    /// Per-level zoom factor (`width_{i+1} = width_i / zoom`).
    #[arg(long, default_value_t = 6.0)]
    pub zoom: f64,

    /// Mandelbrot/Julia panel width in pixels (height follows 16:9).
    #[arg(long, default_value_t = 640)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S box downsample) for both panels.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Start frame center as `re,im` (arbitrary-precision decimals).
    #[arg(long, default_value = "-0.5,0", allow_hyphen_values = true)]
    pub start_center: String,

    /// Start frame width in the complex plane.
    #[arg(long, default_value_t = 3.0)]
    pub start_width: f64,

    /// RNG seed for sampling a target from each level's top-1% scored windows.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Score window size K (K×K window over the feature map).
    #[arg(long, default_value_t = 5)]
    pub window: u32,

    /// maxiter schedule base: `maxiter = round(base + per_decade·log10(mag))`.
    #[arg(long, default_value_t = 1000.0)]
    pub maxiter_base: f64,

    /// maxiter schedule slope (iterations added per decade of magnification).
    #[arg(long, default_value_t = 1500.0)]
    pub per_decade: f64,

    /// Fixed maxiter for every Julia panel (base-scale, shallow).
    #[arg(long, default_value_t = 3000)]
    pub julia_maxiter: u32,

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

    /// Precision backend: f64, perturb, or auto (default; switches per level).
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Output filmstrip PNG path. Per-level panels go in `<stem>_panels/`.
    #[arg(long, default_value = "out/strips/descend_strip.png")]
    pub output: String,

    /// Output JSON log path.
    #[arg(long, default_value = "out/strips/descend.json")]
    pub json: String,
}

impl DescendArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Parse `--start-center` (`re,im`) into two decimal strings (kept as
    /// strings for arbitrary-precision parsing downstream).
    pub fn resolved_start_center(&self) -> Result<(String, String), String> {
        let parts: Vec<&str> = self.start_center.split(',').collect();
        if parts.len() != 2 {
            return Err(format!(
                "invalid --start-center '{}', expected re,im",
                self.start_center
            ));
        }
        Ok((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
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

/// `sheet` subcommand: same location + shading, multiple palettes.
#[derive(Args, Debug)]
pub struct SheetArgs {
    #[command(flatten)]
    pub location: LocationArgs,

    #[command(flatten)]
    pub shade: ShadeArgs,

    /// Palette file paths (`.ugr`/`.map`). For multi-block `.ugr`, every block
    /// is included as its own tile.
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub palettes: Vec<String>,

    /// Built-in palette names (`default`, `cubehelix`, `viridis`).
    #[arg(long, num_args = 0.., value_delimiter = ' ')]
    pub builtins: Vec<String>,

    /// Grid columns (default: ≈ √N).
    #[arg(long)]
    pub cols: Option<usize>,

    /// Per-tile width in pixels (height follows aspect). Modest, e.g. 320–512.
    #[arg(long, default_value_t = 384)]
    pub tile_width: u32,

    /// Reverse every palette's gradient direction.
    #[arg(long, default_value_t = false)]
    pub palette_reverse: bool,

    /// Output PNG path.
    #[arg(long, default_value = "out/strips/sheet.png")]
    pub output: String,
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
