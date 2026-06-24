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
    /// Discovery location generator (promotes probe 1): draw → cheap screen →
    /// accept band → keep, run to a target keeper count, persisting the full
    /// per-image log vector (`locations.jsonl`) + a run manifest under
    /// `data/generated/<run>/`, plus an annotated keeper contact sheet. Emits
    /// located, logged, single-palette-preview keepers only (3-palette labeling
    /// is a downstream stage).
    Generate(GenerateArgs),
    /// Presentation renderer: takes a `locations.jsonl` from a `generate` run,
    /// zooms in on each seed center, tries three composition offsets (center,
    /// thirds, golden) at cheap resolution, gates on black fraction < 40%, and
    /// renders the accepted composition at full resolution across random palettes.
    /// Emits per-crop PNGs, a contact sheet, and a manifest.json.
    Present(PresentArgs),
    /// Greedy Mandelbrot→Julia descent filmstrip + JSON (depth-falloff probe).
    Descend(DescendArgs),
    /// Stochastic guided descent: many decorrelated root-down walks to random
    /// depth, each step picking the next center by a probabilistic policy (mostly
    /// into a detected μ-focus). Every visited frame is a candidate; emits a
    /// candidate-pool sheet (by-walk ladders + flat grid) + `pool.jsonl` under
    /// `data/guided_descend/<run>/`. Geometric policies only — no CNN, no dedup,
    /// no prefix-sharing. Diagnosis-first; `generate` is left intact as the control.
    GuidedDescend(GuidedDescendArgs),
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
    /// Coverage-dominance scorer: render one frame f64 and report `coverage` (the
    /// fraction of the escaped frame at the magical few-pixels boundary spacing) plus
    /// the speckle/interior/busy gates; a band-sensitivity table and optional
    /// scale-sweep make it the discrimination + retune tool for the harvest.
    Cover(CoverArgs),
    /// Corpus energy-histogram metric calibration + eye-check (Prompt
    /// corpus-energy-calibration). Computes a multi-scale OKLab edge-energy
    /// histogram for every corpus wallpaper, freezes equal-count bins per scale,
    /// stores each image's signature, and runs two eye-checks (corpus-internal NN
    /// pairs + buffet source-B DEEP ranking) plus a k-means archetype sheet. No
    /// descent, no search, no candidate scoring beyond the buffet eye-check.
    Calibrate(CalibrateArgs),
    /// Visual buffet: sample three sources (main-boundary neighborhoods,
    /// off-cusp signature-filtered frames, minibrot control), render each across
    /// an off-boundary-offset × scale grid, label every tile with its metrics,
    /// and compose one sheet per source. No objective / ranking / drift — sample,
    /// render, label, so a human can point at what reads "magical".
    Buffet(BuffetArgs),
    /// Throwaway diagnostic: re-score the buffet source-B DEEP tiles against the
    /// persisted corpus calibration under five candidate scoring rules (nearest-k,
    /// nearest-archetype, global-centroid, tail-pruned nearest-k, two-sided density
    /// band) and print a PASS/FAIL table per rule. Loads everything from disk
    /// (persisted artifact + buffet histogram cache); the only render is a
    /// deterministic re-render of the fixed buffet set if its histograms aren't
    /// cached. No search, no descent, no new-location rendering, no winner picked.
    Rescore(RescoreArgs),
    /// Diagnosis-only: add non-sparse-but-bad over-busy/speckle controls to the
    /// known-answer set, quarantine the degenerate reference cluster (C4) from the
    /// typicality statistics, and re-score the survivor rules (R3 global-centroid,
    /// R5 density band, raw s16-bin0 scalar) against okay + sparse + the controls.
    /// Renders only the fixed control set (a known-answer set, same category as the
    /// buffet re-render). No search, no descent, no winner picked.
    Overbusy(OverbusyArgs),
    /// Diagnosis-only: score the 22-tile known-answer set under nearest-good-archetype
    /// (min EMD to the k centroids of the C4-quarantined corpus), swept over cluster
    /// granularity k ∈ {5,8,12,16}. Loads everything from disk (artifact + both
    /// histogram caches); renders nothing. Prints a per-k ranking + PASS/FAIL plus the
    /// control-match / straddle / sparse-survivor diagnostics. No winner picked.
    Archetype(ArchetypeArgs),
    /// Diagnosis-only adversarial anchor probe: tests the founding axiom
    /// ("good = resembles some real wallpaper") at the level of *individual* corpus
    /// members, not centroids. Calibrates the corpus 1-NN distance distribution
    /// (real wallpaper-to-wallpaper similarity), then finds each known-answer tile's
    /// nearest individual corpus wallpaper and the smallest intrinsic corpus-corpus
    /// pairs, rendering both as side-by-side montages for Matt's eye. EMD on cached
    /// histograms; re-renders only the fixed known-answer set (controls + buffet DEEP)
    /// for the montage images (flagged). Picks no pivot, wires nothing.
    Anchor(AnchorArgs),
    /// Diagnosis-only trivial corpus dedup: find descriptor-near corpus pairs
    /// (EMD < epsilon), confirm each as near-pixel-identical via a 16×16 gray
    /// thumbnail diff, union confirmed pairs into duplicate groups (keep the
    /// lexically-first member), and emit a drop-list plus the corpus 1-NN
    /// distribution before vs after the drop. Reads corpus PNGs + cached
    /// histograms only — no fractal renders. Does NOT mutate the artifact.
    Dedup(DedupArgs),
    /// Diagnosis-only palette-sweep muster: does a corpus-marginal density band
    /// filter the 22-tile known-answer set? Renders each fixed tile's iteration
    /// data ONCE, recolors across a legit palette sweep (+ random/flat degenerate
    /// controls), scores a two-sided busyness scalar (mean fine s16 edge energy,
    /// recovered from the frozen-bin histogram), places each recolor as a corpus
    /// percentile, and sweeps an accept band reporting okay-recall / speckle-leak /
    /// sparse-rejection. Marginal control only — no good-busy vs bad-busy split.
    /// Picks no band, builds no loop. Matt judges the eye-check sheets.
    Muster(MusterArgs),
    /// Diagnosis-only f64 render-path profiler: time the phase breakdown
    /// (setup / iterate / shade+downsample / encode / write) of one f64 render,
    /// build an escape-time histogram (max_iter-pixel fraction, mean/total
    /// iterations, interior-vs-escaper iteration split), and run a thread-scaling
    /// sweep (wall-clock / speedup / efficiency vs 1 thread) over the iteration
    /// pass. Release-build only. Changes no render behavior; picks no optimization.
    Profile(ProfileArgs),
    /// Diagnosis-only audit of the `generate` accept-band detail-floor: draw with
    /// a fresh seed, log EVERY draw (full screen vector + reject clause + per-clause
    /// margins), and render a keeper-res contact sheet of the un-eyeballed SPRD
    /// corridor (interior-ok, not-flat, straddling `spread_min`) marked CUT/KEEP,
    /// plus a few flat / interior-black bulk reps. Reuses `generate`'s screen + band
    /// verbatim; changes no band default, builds no new metric.
    RejectCorridor(RejectCorridorArgs),
    /// Diagnostic palette favorite-picker: iterate one fixed dense field (the
    /// seahorse-valley spiral) ONCE and re-shade it across N palettes sampled
    /// (fixed seed) from the survivor colormap library, into one labeled contact
    /// sheet. Diagnosis-only — no band, no scoring, no render-path change; Matt
    /// picks. Sheet + reproducibility legend land under `data/palette_pick/`.
    PalettePick(PalettePickArgs),
    /// Palette-scoring surface: iterate 4 fixed "good" views ONCE (reusing
    /// `present`'s render config) and recolor them into a clean 2×2 grid under
    /// **each** of the 224 library palettes, so Matt can hand-label palette
    /// quality (1/2/3) decoupled from location quality. Without `--full` it stops
    /// at a `twilight_shifted` preview grid (the views-fixture eyeball gate); with
    /// `--full` it writes one grid PNG per palette + a manifest under
    /// `data/palette_score/`. Selective mirror per `mirror_needed`, exactly as
    /// `present`/`palette-pick`. Deterministic; no scoring (Matt judges).
    PaletteScore(PaletteScoreArgs),
    /// Antialiasing bake-off at the 2560×1440 target: one view × one cyclic
    /// palette, rendered under five AA schemes (ordered grid ss2/ss3/ss4 +
    /// rotated-grid 4-rooks + stratified jitter) so two axes — sample count and
    /// sub-sample placement — can be eyeballed at 1:1. Reuses `present`'s f64
    /// render path; varies only `render::SubsamplePattern`. Per cell: wall-clock,
    /// the full PNG, and a matched 1:1 crop of one auto-picked high-frequency
    /// region. Emits `tools/viz/aa_study.html` + a JSON log (stable path, not
    /// `out/`). Box downsample fixed; the reconstruction-filter study is next.
    AaStudy(AaStudyArgs),
    /// AA reconstruction-filter bake-off: one view × one cyclic palette, rendered
    /// **once** at grid ss4 (16 spp), then downsampled three ways — box vs
    /// Mitchell–Netravali vs Lanczos-3 — over the *same* shaded supersample buffer.
    /// The filter only reweights samples already iterated, so the two extra cells
    /// are ~free on top of the single ~4.2 s iterate. Pins the 1:1 crop to the
    /// aa-study's selected box so all three filters are judged on identical pixels.
    /// Emits `tools/viz/aa_filter_study.html` + a JSON log (stable path, not
    /// `out/`).
    AaFilter(AaFilterArgs),
    /// Locked wallpaper-render default: render ONE (location × palette) at the
    /// settled quality — grid ss4 + Lanczos-3 @ 2560×1440 — to a caller-chosen
    /// stable path, reporting iterate / filter / total wall-clock. An extract of
    /// the verified `aa-filter` f64 path (selective-mirror palette load, the
    /// `ss×`-scaled reconstruction filter); the locked defaults live here only, so
    /// the bare render path's fast-preview defaults are untouched. Shallow f64 by
    /// construction (asserted).
    RenderOne(RenderOneArgs),
    /// Diagnosis-only iteration-cap escalation harness. Auto-selects the
    /// worst-offender crops (grayest spiral cores) from a `present` manifest +
    /// the standard test location, renders each at the locked wallpaper quality
    /// (grid ss4 + Lanczos-3) across an escalating `maxiter` series, and reports
    /// per (crop × cap) the residual pinned-at-cap fraction + wall-time, an HTML
    /// escalation sheet, residual-vs-`frame_width` (depth question), occupancy
    /// drift, a cost multiplier, and — at the auto-detected knee cap — the
    /// no-escape-fraction distribution grounding the new black gate. Reuses
    /// `energy.rs` + the `render-one` path; picks nothing. Stable output under
    /// `data/calibration/maxiter_diag/` (not `out/`). Shallow f64 (asserted).
    MaxiterDiag(MaxiterDiagArgs),
    /// Palette universality probe: pick N label-3 ("great") locations at random
    /// (fixed seed) from the `loose0_v3` labels+manifest, iterate each ONCE at the
    /// `render-one` quality path (grid ss4 + Lanczos-3), and recolor across the
    /// full score-3 palette pool — so universally-bad palettes (bad even on a
    /// proven-good structure) can be spotted and cut. Emits the JPG crops +
    /// `probe_index.json` (palettes carry their corpus not-bad rate, sorted
    /// worst-first) under `data/palette_probe/`. The viewer
    /// (`tools/viz/palette_probe.html`) writes the verdict; this picks nothing.
    PaletteProbe(PaletteProbeArgs),
    /// Measurement-only signal-separation diagnostic for an explicit reject-the-bad
    /// gate. For each distinct (draw_index, composition) geometry in a `present`
    /// manifest, re-render the cheap f64 screen at the stored crop frame (DE channel
    /// on) and compute dynamics signals — `de_small_frac` (escaped pixels whose
    /// DE-in-2560px < k, swept k∈{0.5,1,2,4}), `slow_escape_frac` (escape iter near
    /// `maxiter`), `interior_frac` — plus image-space signals on the representative
    /// JPG via the `energy.rs` descriptor under the frozen corpus bins
    /// (`fine_energy_frac` = s16 density, coarse density, mean edge energy). Writes
    /// one CSV row per geometry (signals + crop frame + manifest occupancy/black).
    /// Builds NO gate, modifies NO gating; the Python side does the AUC/sheet.
    GateDiag(GateDiagArgs),
    /// Measurement-only dynamics-FIELD dumper for the smoothed-escape focal-point +
    /// scale-space-organization exploration (sibling of `gate-diag`, but full 2D
    /// arrays not scalars). For each frame in a JSONL frames file, re-render the
    /// cheap f64 screen (DE channel on) and dump three row-major arrays — `mu`
    /// (smooth escape, NaN interior), `de_px` (DE in 2560-px units), `interior`
    /// mask — plus a manifest. The Python side derives the potential `G ≈ 2^-mu`,
    /// the P/Q/R smoothed focus fields, scale-space persistence, and the
    /// organization scalars. Builds NO gate, modifies NO gating.
    FocusDiag(FocusDiagArgs),
}

/// `focus-diag` subcommand: see `focus_diag::run_focus_diag`. Field-array dumper —
/// renders each frame's f64 dynamics fields (mu / de_px / interior) at a modest
/// res and writes them as raw arrays for the Python scale-space analysis.
#[derive(Args, Debug)]
pub struct FocusDiagArgs {
    /// JSONL frames file (one compact `{name,cx,cy,fw,width}` object per line),
    /// emitted by the Python driver after it picks the contrast + sample frames.
    #[arg(long, default_value = "data/focus_diag/frames.jsonl")]
    pub frames: String,

    /// Iteration cap. Production default 8000 (the manifest's "maxiter 2000" string
    /// is a stale hardcoded note; ignore it).
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// Escape (bailout) radius. ≥1e6 for a stable DE estimate / ideal smooth band;
    /// 1e6 ≈ 2^20 matches `present`/`generate` — do not bump.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Default field width (px) for frames that don't carry their own `width`;
    /// height follows 16:9. Modest by design — we need smooth peaks, not AA.
    #[arg(long, default_value_t = 768)]
    pub width: u32,

    /// Reference width the DE-in-pixels normalization is pinned to (so `de_px`
    /// reads as "DE in final-wallpaper pixels", independent of field res).
    #[arg(long, default_value_t = 2560)]
    pub de_ref_width: u32,

    /// Stable output dir (not under `out/`). Emits `fields/` + `fields_manifest.json`.
    #[arg(long, default_value = "data/focus_diag/")]
    pub out_dir: String,
}

/// Sub-pixel sample placement for `render-one` (maps to [`crate::render::SubsamplePattern`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum PatternChoice {
    /// Ordered grid (the lock; byte-identical historical path). Any `ss`.
    Grid,
    /// Rotated grid / 4-rooks. **ss2 only.**
    Rgss,
    /// Stratified jitter (seeded). Any `ss`.
    Jitter,
}

impl PatternChoice {
    pub fn label(self) -> &'static str {
        match self {
            PatternChoice::Grid => "grid",
            PatternChoice::Rgss => "rgss",
            PatternChoice::Jitter => "jitter",
        }
    }
}

impl From<PatternChoice> for crate::render::SubsamplePattern {
    fn from(p: PatternChoice) -> Self {
        match p {
            PatternChoice::Grid => crate::render::SubsamplePattern::Grid,
            PatternChoice::Rgss => crate::render::SubsamplePattern::Rgss,
            PatternChoice::Jitter => crate::render::SubsamplePattern::Jitter,
        }
    }
}

/// Downsample reconstruction filter for `render-one` (maps to [`crate::render::DownsampleFilter`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FilterChoice {
    /// Flat `ss×ss` average.
    Box,
    /// Mitchell–Netravali cubic.
    Mitchell,
    /// Lanczos-3 windowed sinc (the lock).
    Lanczos3,
}

impl FilterChoice {
    pub fn label(self) -> &'static str {
        match self {
            FilterChoice::Box => "box",
            FilterChoice::Mitchell => "mitchell",
            FilterChoice::Lanczos3 => "lanczos3",
        }
    }
}

impl From<FilterChoice> for crate::render::DownsampleFilter {
    fn from(f: FilterChoice) -> Self {
        match f {
            FilterChoice::Box => crate::render::DownsampleFilter::Box,
            FilterChoice::Mitchell => crate::render::DownsampleFilter::Mitchell,
            FilterChoice::Lanczos3 => crate::render::DownsampleFilter::Lanczos3,
        }
    }
}

/// `render-one` subcommand: see `render_one::run_render_one`. One location ×
/// palette at the locked wallpaper quality. Locked defaults (all overridable):
/// `--width 2560 --height 1440 --ss 4 --pattern grid --filter lanczos3`.
#[derive(Args, Debug)]
pub struct RenderOneArgs {
    /// Frame center, real part (`--cx`) — arbitrary-precision decimal string.
    #[arg(long = "cx", default_value = "-0.746339", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part (`--cy`) — arbitrary-precision decimal string.
    #[arg(long = "cy", default_value = "0.112242", allow_hyphen_values = true)]
    pub center_im: String,

    /// Frame width in the complex plane (`--fw`).
    #[arg(long = "fw", default_value_t = 0.000583)]
    pub frame_width: f64,

    /// Palette name, looked up in `--colormaps` (loaded through the selective-mirror
    /// path, so cyclic and sequential maps both render seam-free).
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Output PNG path — a stable path the caller chooses (not under `out/`).
    #[arg(long, default_value = "render.png")]
    pub out: String,

    /// Output width in pixels (the lock: 2560).
    #[arg(long, default_value_t = 2560)]
    pub width: u32,

    /// Output height in pixels (the lock: 1440).
    #[arg(long, default_value_t = 1440)]
    pub height: u32,

    /// Linear supersampling factor (the lock: 4 → 16 spp).
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Sub-pixel sample placement (the lock: grid).
    #[arg(long, value_enum, default_value_t = PatternChoice::Grid)]
    pub pattern: PatternChoice,

    /// Downsample reconstruction filter (the lock: lanczos3).
    #[arg(long, value_enum, default_value_t = FilterChoice::Lanczos3)]
    pub filter: FilterChoice,

    /// Maximum iterations / orbit cap ("max_orbit"). Raised 2000 → 8000 (the
    /// `maxiter-blackgate` pass, Matt's pick): the escalation sheet's residual
    /// pinned-at-cap fraction asymptotes by ~8k (max-over-crops |Δ| drops below
    /// 0.02 at the 8k→32k step — the measured knee), what remains is genuine
    /// minibrot interior no cap reclaims. 8000 is the knee: ~3.3–3.5× the cap-2000
    /// cost on interior-heavy frames, near-free on filament frames.
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// SplitMix64 seed (consumed only by `--pattern jitter`).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

/// `maxiter-diag` subcommand: see `maxiter_diag::run_maxiter_diag`. Diagnosis-only
/// iteration-cap escalation harness. All caps/crops/resolution overridable; the
/// defaults reproduce the loose0_v3 worst-offender escalation.
#[derive(Args, Debug)]
pub struct MaxiterDiagArgs {
    /// `present` manifest mined for worst-offender crops (highest recorded
    /// `black_fraction` per seed × composition — the grayest spiral cores).
    #[arg(long, default_value = "data/label_crops/loose0_v3/manifest.json")]
    pub manifest: String,

    /// Number of worst-offender crops to escalate (the test location is appended).
    #[arg(long, default_value_t = 3)]
    pub offenders: usize,

    /// Escalating iteration-cap series (comma-separated).
    #[arg(long, default_value = "2000,8000,32000,128000")]
    pub caps: String,

    /// Render width in px (height follows 16:9). The locked quality is otherwise
    /// fixed: grid ss4 + Lanczos-3 (the `render-one` path).
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Linear supersample factor (the lock: 4 → 16 spp).
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Palette name, looked up in `--colormaps` (selective-mirror load).
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Standard test location (minibrot eye), real part. Matches `render-one`.
    #[arg(long, default_value = "-0.746339", allow_hyphen_values = true)]
    pub test_cx: String,

    /// Standard test location, imaginary part.
    #[arg(long, default_value = "0.112242", allow_hyphen_values = true)]
    pub test_cy: String,

    /// Standard test location, frame width.
    #[arg(long, default_value_t = 0.000583)]
    pub test_fw: f64,

    /// Knee-detection epsilon: the first cap past which the max-over-crops residual
    /// change drops below this is the provisional knee.
    #[arg(long, default_value_t = 0.02)]
    pub knee_eps: f32,

    /// Number of manifest crops re-rendered (cheap gate res) at the knee cap to
    /// report the no-escape-fraction distribution. `0` skips gate calibration.
    #[arg(long, default_value_t = 80)]
    pub gate_sample: usize,

    /// Stable output directory (not under `out/`).
    #[arg(long, default_value = "data/calibration/maxiter_diag/")]
    pub out_dir: String,
}

/// `palette-probe` subcommand: see `palette_probe::run_palette_probe`. Picks N
/// label-3 locations at random (fixed seed) from the labels+manifest, iterates
/// each once at the render-one quality path, and recolors across the full score-3
/// palette pool. Shallow f64 by construction (asserted per location).
#[derive(Args, Debug)]
pub struct PaletteProbeArgs {
    /// Location labels (the `draw|comp|palette` → label-1/2/3 map).
    #[arg(long, default_value = "labels/location_labels.json")]
    pub labels: String,

    /// `present` manifest the labels were drawn against (recovers crop geometry).
    #[arg(long, default_value = "data/label_crops/loose0_v3/manifest.json")]
    pub manifest: String,

    /// Palette pool to recolor across (the score-3 survivors).
    #[arg(long, default_value = "data/palettes/score3_colormaps.json")]
    pub colormaps: String,

    /// Number of distinct label-3 locations to sample (uses all if fewer exist).
    #[arg(long, default_value_t = 5)]
    pub n_locations: usize,

    /// SplitMix64 seed for the location pick (logged; fixed for reproducibility).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Output width in px (height follows 16:9). Probe crops are 1280×720.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Output height in px.
    #[arg(long, default_value_t = 720)]
    pub height: u32,

    /// Linear supersample factor (the lock: 4 → 16 spp).
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Maximum iterations / orbit cap (the present/render-one current default).
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// JPEG quality for the output crops.
    #[arg(long, default_value_t = 90)]
    pub jpg_quality: u8,

    /// Stable output directory (not under `out/`).
    #[arg(long, default_value = "data/palette_probe/")]
    pub out_dir: String,
}

/// `gate-diag` subcommand: see `gate_diag::run_gate_diag`. Measurement-only signal
/// extractor for the reject-the-bad gate study. Re-renders each manifest geometry's
/// cheap f64 screen (DE channel on) at the stored crop frame and emits per-geometry
/// dynamics + energy-descriptor signals to a CSV. Builds no gate.
#[derive(Args, Debug)]
pub struct GateDiagArgs {
    /// `present` manifest whose crop geometries (cx/cy/fw per draw_index×comp) are
    /// re-rendered. The label join + AUC analysis happen Python-side off this CSV.
    #[arg(long, default_value = "data/label_crops/loose0_v3/manifest.json")]
    pub manifest: String,

    /// Persisted energy calibration (frozen quantile bins) — the `fine_energy_frac`
    /// signal bins each JPG's region energies under these corpus-frozen edges.
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Per-scale EMD weights are irrelevant here (no distance), but the frozen bins
    /// come from the same artifact; this is unused and kept only for symmetry.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Iteration cap for the diagnostic re-render. Defaults to `present`'s 8000
    /// (the manifest's embedded "maxiter 2000" note is a stale hardcoded string).
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// Escape (bailout) radius. ≥1e6 is required for a stable DE estimate; the
    /// default matches `present`/`generate` (1e6 ≈ 2^20, already in the ideal band).
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Diagnostic cheap-screen width (px); height follows 16:9. The DE field is
    /// smooth, so a sub-full-res screen samples `de_small_frac` faithfully and fast.
    #[arg(long, default_value_t = 640)]
    pub screen_width: u32,

    /// Reference width the DE-in-pixels normalization is pinned to (so `de_px` reads
    /// as "DE in final-wallpaper pixels", independent of `screen_width`).
    #[arg(long, default_value_t = 2560)]
    pub de_ref_width: u32,

    /// Stable output dir (not under `out/`). Emits `signals.csv`.
    #[arg(long, default_value = "data/gate_diag/")]
    pub out_dir: String,
}

/// `aa-filter` subcommand: see `aa_filter::run_aa_filter`. One fixed view + cyclic
/// palette iterated once at grid ss4; the sole axis is the downsample
/// reconstruction filter (box / Mitchell / Lanczos). Shallow f64 (asserted).
#[derive(Args, Debug)]
pub struct AaFilterArgs {
    /// Frame center, real part — arbitrary-precision decimal string. Default is
    /// the aa-study view (so the pinned crop lands on identical pixels).
    #[arg(long, default_value = "-0.746339", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal string.
    #[arg(long, default_value = "0.112242", allow_hyphen_values = true)]
    pub center_im: String,

    /// Width of the view in the complex plane (aa-study default).
    #[arg(long, default_value_t = 0.000583)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 2000)]
    pub maxiter: u32,

    /// Output width in pixels (height follows 16:9). The study's real target.
    #[arg(long, default_value_t = 2560)]
    pub width: u32,

    /// Supersample factor (grid). The study locks this to 4 (16 spp); exposed for
    /// experimentation only.
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Cyclic (mirror-safe) palette name, looked up in `--colormaps`.
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Survivor colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// 1:1 crop width (px). The matched crop is auto-picked at this size only if
    /// `--crop` is empty.
    #[arg(long, default_value_t = 384)]
    pub crop_width: u32,

    /// 1:1 crop height (px).
    #[arg(long, default_value_t = 384)]
    pub crop_height: u32,

    /// Explicit crop box `x,y,w,h`. Default pins the aa-study's selected box so
    /// the three filters are judged on identical pixels; pass an empty string to
    /// auto-pick by edge energy instead.
    #[arg(long, default_value = "928,464,384,384")]
    pub crop: String,

    /// Output directory for the PNGs/crops/JSON (HTML lands in its parent).
    /// Stable path by design — not under `out/`.
    #[arg(long, default_value = "tools/viz/aa_filter_study")]
    pub out_dir: String,
}

impl AaFilterArgs {
    /// Parse `--crop` (`x,y,w,h`). Empty string → auto-pick (returns `None`).
    pub fn resolved_crop(&self) -> Result<Option<(u32, u32, u32, u32)>, String> {
        if self.crop.trim().is_empty() {
            return Ok(None);
        }
        let p: Vec<&str> = self.crop.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --crop '{}', expected x,y,w,h", self.crop));
        }
        let mut v = [0u32; 4];
        for (i, s) in p.iter().enumerate() {
            v[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --crop component '{}'", s.trim()))?;
        }
        Ok(Some((v[0], v[1], v[2], v[3])))
    }
}

/// `aa-study` subcommand: see `aa_study::run_aa_study`. One fixed view + cyclic
/// palette rendered under the five AA schemes; reports per-cell wall-clock and
/// emits the 1:1-crop comparison HTML. Shallow f64 by construction (asserted).
#[derive(Args, Debug)]
pub struct AaStudyArgs {
    /// Frame center, real part — arbitrary-precision decimal string.
    #[arg(long, default_value = "-0.7453", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal string.
    #[arg(long, default_value = "0.1127", allow_hyphen_values = true)]
    pub center_im: String,

    /// Width of the view in the complex plane.
    #[arg(long, default_value_t = 0.0035)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 2000)]
    pub maxiter: u32,

    /// Output width in pixels (height follows 16:9). The study's real target.
    #[arg(long, default_value_t = 2560)]
    pub width: u32,

    /// Cyclic (mirror-safe) palette name, looked up in `--colormaps`.
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Survivor colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// SplitMix64 seed for the stratified-jitter cell (deterministic).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// 1:1 crop width (px). The matched high-frequency crop is auto-picked at
    /// this size unless `--crop` pins an explicit box.
    #[arg(long, default_value_t = 384)]
    pub crop_width: u32,

    /// 1:1 crop height (px).
    #[arg(long, default_value_t = 384)]
    pub crop_height: u32,

    /// Explicit crop box `x,y,w,h` (overrides the auto edge-energy pick).
    #[arg(long)]
    pub crop: Option<String>,

    /// Output directory for the PNGs/crops/JSON (the HTML lands in its parent).
    /// Stable path by design — not under `out/`.
    #[arg(long, default_value = "tools/viz/aa_study")]
    pub out_dir: String,
}

impl AaStudyArgs {
    /// Parse `--crop` (`x,y,w,h`) into an explicit box, or `None` for auto-pick.
    pub fn resolved_crop(&self) -> Result<Option<(u32, u32, u32, u32)>, String> {
        let Some(spec) = &self.crop else { return Ok(None) };
        let p: Vec<&str> = spec.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --crop '{spec}', expected x,y,w,h"));
        }
        let mut v = [0u32; 4];
        for (i, s) in p.iter().enumerate() {
            v[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --crop component '{}'", s.trim()))?;
        }
        Ok(Some((v[0], v[1], v[2], v[3])))
    }
}

/// `palette-pick` subcommand: see `palette_pick::run_palette_pick`. Reproducible
/// for a fixed `--seed`; the field is shallow (f64 cheap-regime, asserted by the
/// auto backend staying f64).
#[derive(Args, Debug)]
pub struct PalettePickArgs {
    /// Field center, real part — the handoff's preferred palette-reading spiral.
    #[arg(long, default_value = "-0.7453", allow_hyphen_values = true)]
    pub center_re: String,

    /// Field center, imaginary part.
    #[arg(long, default_value = "0.1127", allow_hyphen_values = true)]
    pub center_im: String,

    /// Frame width in the complex plane. Default frames a dense spiral; shrink it
    /// (or move the center) if the density read says the field is flat.
    #[arg(long, default_value_t = 0.012)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 2000)]
    pub maxiter: u32,

    /// Per-tile width in pixels (height follows 16:9). Modest diagnostic size.
    #[arg(long, default_value_t = 320)]
    pub tile_width: u32,

    /// Linear supersampling factor (S×S box downsample) per tile.
    #[arg(long, default_value_t = 1)]
    pub supersample: u32,

    /// Number of palettes to sample from the library.
    #[arg(long, default_value_t = 100)]
    pub count: usize,

    /// SplitMix64 seed for the deterministic palette sample (reproducible).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Grid columns (default ≈ √N).
    #[arg(long)]
    pub cols: Option<usize>,

    /// Survivor colormap library (JSON array of name/source/stops objects).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Output directory for the sheet + reproducibility legend.
    #[arg(long, default_value = "out/palette_pick")]
    pub out_dir: String,
}

/// `palette-score` subcommand: see `palette_score::run_palette_score`. Builds the
/// hand-scoring surface — 4 fixed views iterated once, recolored under each of the
/// 224 survivors into clean 2×2 grids. Deterministic; reuses `present`'s coloring
/// config + the selective-mirror path verbatim so scores transfer to the location
/// pass.
#[derive(Args, Debug)]
pub struct PaletteScoreArgs {
    /// The 4-view fixture: `[{name, cx, cy, fw, composition?}, …]` (exactly 4).
    #[arg(long, default_value = "data/palette_score/views.json")]
    pub views: String,

    /// Survivor colormap library (the 224 clean maps, inline cycle/mirror_needed).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub palette_file: String,

    /// Per-cell width in pixels (one view per cell).
    #[arg(long, default_value_t = 480)]
    pub cell_width: u32,

    /// Per-cell height in pixels (keep 16:9 with cell_width).
    #[arg(long, default_value_t = 270)]
    pub cell_height: u32,

    /// Linear supersampling factor (S×S box downsample) per cell.
    #[arg(long, default_value_t = 2)]
    pub ss: u32,

    /// Gutter (px) between/around the 2×2 cells — kept thin, no burned-in text.
    #[arg(long, default_value_t = 6)]
    pub gutter: u32,

    /// Maximum iterations (matches `present`'s default for config parity).
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Diagnostic palette for the preview grid (the views-fixture eyeball gate).
    #[arg(long, default_value = "twilight_shifted")]
    pub diagnostic_palette: String,

    /// Render the full 224-palette sweep. Without it, stop at the preview grid.
    #[arg(long)]
    pub full: bool,

    /// Output directory (outside `out/` — Matt clears `out/`).
    #[arg(long, default_value = "data/palette_score/")]
    pub out_dir: String,
}

/// `profile` subcommand: see `profile::run_profile`. Measure-only — phase
/// breakdown + escape-time histogram + thread-scaling sweep for the f64
/// Mandelbrot render path. Runs one location (shallow-decorative by default);
/// run it twice with different `--center-*/--frame-width/--label` to bracket the
/// cost range. f64-only by construction (asserts the backend stayed f64).
#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Frame center, real part — arbitrary-precision decimal. Default: the
    /// shallow-decorative seahorse-valley spiral (reads palette character, not
    /// interior-dominated).
    #[arg(long, default_value = "-0.7453", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal.
    #[arg(long, default_value = "0.1127", allow_hyphen_values = true)]
    pub center_im: String,

    /// Width of the view in the complex plane.
    #[arg(long, default_value_t = 0.012)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Output image width in pixels (height follows 3:2). Default 1280 gives a
    /// stable phase/scaling signal; pass 384 for the contact-sheet tile size.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Linear supersampling factor (S×S box downsample). Iteration scales with S².
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Repeats per timed measurement (min + median reported; kernel is
    /// deterministic, so spread is system noise).
    #[arg(long, default_value_t = 5)]
    pub runs: usize,

    /// Thread counts for the strong-scaling sweep (comma-separated). Each builds
    /// its own rayon pool and re-times the iteration pass. Empty skips the sweep.
    #[arg(long, default_value = "1,2,4,6,8,12")]
    pub threads: String,

    /// Label for the printed report / JSON (e.g. `shallow`, `interior`).
    #[arg(long, default_value = "shallow")]
    pub label: String,

    /// Built-in palette for the shade/encode phases (`default`, `cubehelix`, `viridis`).
    #[arg(long, default_value = "default")]
    pub palette: String,

    /// Output directory for the profiling JSON.
    #[arg(long, default_value = "out/profile")]
    pub out_dir: String,
}

impl ProfileArgs {
    /// Parse `--threads` (comma-separated) into the scaling sweep counts.
    /// An empty list (or all-zero) means "no sweep".
    pub fn resolved_threads(&self) -> Result<Vec<usize>, String> {
        let mut out = Vec::new();
        for s in self.threads.split(',') {
            let t = s.trim();
            if t.is_empty() {
                continue;
            }
            let n: usize = t
                .parse()
                .map_err(|_| format!("invalid --threads component '{t}'"))?;
            if n > 0 {
                out.push(n);
            }
        }
        Ok(out)
    }
}

/// `muster` subcommand: see `energy::run_muster`. Diagnosis-only — produces the
/// busyness-scalar corpus distribution + non-saturation check, the per-tile
/// per-palette percentile table, the full band sweep, and three eye-check sheets.
/// Selects nothing.
#[derive(Args, Debug)]
pub struct MusterArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Corpus image folder root (for the colocated-pair montage).
    #[arg(long, default_value = "C:/Users/techm/Desktop/Wallpapers")]
    pub corpus_dir: String,

    /// Buffet metrics JSON (source-B DEEP tile centers — the 18 okay/sparse tiles).
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Dedup drop-list (its `descriptor_near_but_distinct` pairs → colocated sheet).
    #[arg(long, default_value = "data/calibration/dedup_droplist.json")]
    pub droplist: String,

    /// Per-tile per-palette percentile table + corpus distribution + band sweep.
    #[arg(long, default_value = "data/calibration/palette_muster.json")]
    pub out_json: String,

    /// The 22-tile × legit-palette recolor grid (+ random/flat control columns).
    #[arg(long, default_value = "out/palette_muster.png")]
    pub out_grid: String,

    /// Any speckle (OB_*) recolor that passed muster (redemption-vs-blind-leak).
    #[arg(long, default_value = "out/speckle_passing_recolors.png")]
    pub out_speckle: String,

    /// Descriptor-near-but-distinct corpus pairs (the within-busy blind spot).
    #[arg(long, default_value = "out/colocated_pairs.png")]
    pub out_colocated: String,

    /// Render width (px) per tile (iterated once, recolored per palette).
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for the per-tile iteration.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Thumbnail width (px) for the sheet tiles (height follows 16:9).
    #[arg(long, default_value_t = 300)]
    pub thumb_width: u32,

    /// Max descriptor-near-but-distinct pairs to montage on the colocated sheet.
    #[arg(long, default_value_t = 12)]
    pub colocated_pairs: usize,

    /// RNG seed for the degenerate random palette (per-entry random LUT).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

/// `dedup` subcommand: see `energy::run_dedup`. Removes accidental near-pixel
/// duplicates from the corpus before it is used as a band reference. Trivial only
/// — descriptor-near is the cheap finder, the pixel check is the verdict; no
/// aesthetic/quality judgment. Filters at use-time via a drop-list; the calibration
/// artifact is left intact.
#[derive(Args, Debug)]
pub struct DedupArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Corpus image folder root (the `name` fields in the artifact resolve here).
    #[arg(long, default_value = "C:/Users/techm/Desktop/Wallpapers")]
    pub corpus_dir: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal). Must match calibrate.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Candidate-pair EMD cutoff: pairs with descriptor distance below this are
    /// pixel-checked. Generous on purpose — the pixel check, not this, decides.
    #[arg(long, default_value_t = 0.13)]
    pub epsilon: f64,

    /// Pixel-confirm threshold: a candidate pair is a duplicate only if the mean
    /// absolute 16×16 gray difference (0..255 scale) is at/below this.
    #[arg(long, default_value_t = 8.0)]
    pub pixel_threshold: f64,

    /// Side length of the gray confirmation thumbnail (S×S, center-cropped 16:9).
    #[arg(long, default_value_t = 16)]
    pub thumb_side: u32,

    /// Drop-list output (dropped filenames + kept representative + pixel distance).
    #[arg(long, default_value = "data/calibration/dedup_droplist.json")]
    pub out_json: String,
}

impl DedupArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}

/// `anchor` subcommand: see `energy::run_anchor`. Diagnosis-only — produces the
/// pairs, distances, and calibration; selects nothing.
#[derive(Args, Debug)]
pub struct AnchorArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Corpus image folder root (the `name` fields in the artifact resolve here).
    #[arg(long, default_value = "C:/Users/techm/Desktop/Wallpapers")]
    pub corpus_dir: String,

    /// Buffet metrics JSON (source-B DEEP tile centers, for the montage re-render).
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Cached buffet DEEP-tile histograms (okay/sparse anchors).
    #[arg(long, default_value = "data/calibration/buffet_histograms.json")]
    pub buffet_hist: String,

    /// Cached over-busy/speckle control histograms.
    #[arg(long, default_value = "data/calibration/control_histograms.json")]
    pub control_hist: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal). Must match calibrate.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Number of smallest intrinsic corpus-corpus pairs to surface (Task B).
    #[arg(long, default_value_t = 20)]
    pub top_pairs: usize,

    /// Task-A montage sheet (tile | nearest individual corpus wallpaper).
    #[arg(long, default_value = "out/adversarial_anchor.png")]
    pub out_sheet_a: String,

    /// Task-B montage sheet (smallest intrinsic corpus-corpus pairs).
    #[arg(long, default_value = "out/corpus_collisions.png")]
    pub out_sheet_b: String,

    /// Per-tile + Task-0 distribution + Task-B pair dump.
    #[arg(long, default_value = "data/calibration/collision_distances.json")]
    pub out_json: String,

    /// Thumbnail width (px) for the montage sheets (height follows 16:9).
    #[arg(long, default_value_t = 384)]
    pub thumb_width: u32,

    /// Render width (px) for re-rendering each known-answer tile (montage image).
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for the known-answer tile re-renders.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,
}

impl AnchorArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}

/// `calibrate` subcommand: see the module docs in `energy.rs`. Calibration +
/// eye-check only — it freezes the metric (bins) and produces the visual gates
/// (NN pairs, buffet ranking, cluster sheet). It proposes no objective and runs
/// no search.
#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// Corpus folder of reference wallpapers (top level only — no recursion).
    #[arg(long, default_value = "C:/Users/techm/Desktop/Wallpapers")]
    pub dir: String,

    /// Output directory for the calibration artifact + eye-check sheets.
    #[arg(long, default_value = "out/calibrate")]
    pub out_dir: String,

    /// Buffet metrics JSON whose source-B DEEP tiles are the candidate eye-check.
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal).
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Number of corpus images sampled for the NN-pair eye-check sheet.
    #[arg(long, default_value_t = 16)]
    pub nn_samples: usize,

    /// k for the buffet EMD-to-nearest-k score.
    #[arg(long, default_value_t = 5)]
    pub knn: usize,

    /// k-means archetype count for the corpus-structure sheet (`<2` disables).
    #[arg(long, default_value_t = 6)]
    pub clusters: usize,

    /// Exemplars per cluster in the archetype sheet (one row each).
    #[arg(long, default_value_t = 6)]
    pub exemplars: usize,

    /// Thumbnail width (px) for the eye-check sheets (height follows 16:9).
    #[arg(long, default_value_t = 384)]
    pub thumb_width: u32,

    /// Render width (px) for each buffet candidate tile (height follows 16:9).
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for candidate renders.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// RNG seed for k-means seeding.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl CalibrateArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}

/// `rescore` subcommand: diagnosis-only re-scoring of the buffet source-B DEEP
/// tiles under several candidate scoring rules, using only what is already on
/// disk (the persisted calibration artifact + a buffet-histogram cache). See
/// `energy::run_rescore`.
#[derive(Args, Debug)]
pub struct RescoreArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    /// Default mirrors `energy::ARTIFACT_PATH`.
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Buffet metrics JSON whose source-B DEEP tiles are the candidates.
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Cached buffet DEEP-tile histograms (written on first run, reused after).
    #[arg(long, default_value = "data/calibration/buffet_histograms.json")]
    pub buffet_hist: String,

    /// Full per-tile per-rule score dump.
    #[arg(long, default_value = "data/calibration/rescore_buffet.json")]
    pub out_json: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal). Must match calibrate.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// k for the nearest-k rules (R1, R4).
    #[arg(long, default_value_t = 5)]
    pub knn: usize,

    /// k-means archetype count (R2). Recomputed here — not stored in the artifact.
    #[arg(long, default_value_t = 6)]
    pub clusters: usize,

    /// RNG seed for the k-means recompute (matches calibrate's default).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Render width (px) for a buffet candidate tile, only used on a cache miss.
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for candidate renders (cache miss only).
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,
}

impl RescoreArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}

/// `overbusy` subcommand: see `energy::run_overbusy`. Adds over-busy/speckle
/// controls to the known-answer set, quarantines the degenerate reference cluster,
/// and re-scores the surviving typicality rules. Diagnosis-only; no winner picked.
#[derive(Args, Debug)]
pub struct OverbusyArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Buffet metrics JSON whose source-B DEEP tiles are the okay/sparse anchors.
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Cached buffet DEEP-tile histograms (reused; rendered once on a cache miss).
    #[arg(long, default_value = "data/calibration/buffet_histograms.json")]
    pub buffet_hist: String,

    /// Cached over-busy/speckle control histograms (written first run, reused).
    #[arg(long, default_value = "data/calibration/control_histograms.json")]
    pub control_hist: String,

    /// Full per-tile per-rule score dump.
    #[arg(long, default_value = "data/calibration/rescore_controls.json")]
    pub out_json: String,

    /// Output dir for the eyeballable control sheet (regenerable view).
    #[arg(long, default_value = "out/controls")]
    pub out_dir: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal). Must match calibrate.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// k-means archetype count (must match calibrate/rescore for stable clusters).
    #[arg(long, default_value_t = 6)]
    pub clusters: usize,

    /// Cluster index to quarantine from the typicality statistics (the degenerate
    /// reference-render cluster; C4 under seed 0 / k=6).
    #[arg(long, default_value_t = 4)]
    pub quarantine: usize,

    /// RNG seed for the k-means recompute (matches calibrate's default).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Render width (px) for each control + buffet tile (height follows 16:9).
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for control / buffet candidate renders.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Thumbnail width (px) for the control sheet (height follows 16:9).
    #[arg(long, default_value_t = 480)]
    pub thumb_width: u32,

    /// Force re-render of the control tiles even if a cache exists.
    #[arg(long, default_value_t = false)]
    pub refresh_controls: bool,
}

impl OverbusyArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}

/// `archetype` subcommand: see `energy::run_archetype`. Scores the 22-tile
/// known-answer set under nearest-good-archetype (min EMD to the k centroids of
/// the C4-quarantined survivor corpus), swept over `--ks`. Diagnosis-only:
/// reuses the cached histograms, frozen bins, EMD, and k-means unchanged;
/// renders nothing.
#[derive(Args, Debug)]
pub struct ArchetypeArgs {
    /// Persisted calibration artifact (frozen bins + per-image histograms).
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Cached buffet DEEP-tile histograms (okay/sparse anchors).
    #[arg(long, default_value = "data/calibration/buffet_histograms.json")]
    pub buffet_hist: String,

    /// Cached over-busy/speckle control histograms.
    #[arg(long, default_value = "data/calibration/control_histograms.json")]
    pub control_hist: String,

    /// Full per-tile per-k score dump.
    #[arg(long, default_value = "data/calibration/rescore_archetype.json")]
    pub out_json: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal). Must match calibrate.
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Cluster granularities to sweep (the re-cluster k values over survivors).
    #[arg(long, default_value = "5,8,12,16")]
    pub ks: String,

    /// k-means archetype count used to FIND the quarantine cluster (must match the
    /// overbusy/calibrate clustering so C4 is the same n=93 degenerate cluster).
    #[arg(long, default_value_t = 6)]
    pub clusters: usize,

    /// Cluster index to quarantine before re-clustering (the degenerate reference
    /// cluster; C4 under seed 0 / k=6).
    #[arg(long, default_value_t = 4)]
    pub quarantine: usize,

    /// RNG seed for both the quarantine clustering and the per-k re-clustering.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl ArchetypeArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }

    /// Parse `--ks` (comma-separated) into the granularity sweep.
    pub fn resolved_ks(&self) -> Result<Vec<usize>, String> {
        let mut out = Vec::new();
        for s in self.ks.split(',') {
            let t = s.trim();
            if t.is_empty() {
                continue;
            }
            out.push(t.parse::<usize>().map_err(|_| format!("invalid --ks component '{t}'"))?);
        }
        if out.is_empty() {
            return Err("--ks parsed to no values".into());
        }
        Ok(out)
    }
}

/// `buffet` subcommand: deliberately un-engineered, visual-first sampling of what
/// "scale-uniform decoration away from a cusp" looks like (Prompt visual-buffet-v2).
/// Reuses `coherence::coverage_stats` purely to **label** tiles (coverage,
/// `subpixel_frac`, `interior_frac`, de_px median + IQR spread); none of it feeds
/// any selection. Three sources — (A) main-set boundary neighborhoods, (B) off-cusp
/// signature-filtered frames (a light two-threshold filter on trial, not a scorer),
/// (C) a small minibrot control block — each rendered over rows = off-boundary
/// offset × columns = scale. f64 cheap-regime throughout (asserted).
#[derive(Args, Debug)]
pub struct BuffetArgs {
    #[command(flatten)]
    pub shade: ShadeArgs,

    #[command(flatten)]
    pub palette: PaletteSelectArgs,

    /// Source-A (main-boundary neighborhood) location count.
    #[arg(long, default_value_t = 6)]
    pub a_count: usize,

    /// Source-B (off-cusp signature-filtered) location count.
    #[arg(long, default_value_t = 6)]
    pub b_count: usize,

    /// Source-C (minibrot control) location count.
    #[arg(long, default_value_t = 4)]
    pub c_count: usize,

    /// Per-tile panel width in px (height follows 16:9).
    #[arg(long, default_value_t = 240)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S box downsample) for every tile.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Target wallpaper width `de_px` is pinned to (resolution-invariant `de`), so
    /// the cheap tiles' labels predict the final render's spacing.
    #[arg(long, default_value_t = 2560)]
    pub target_width: u32,

    /// Sub-pixel threshold θ: an escaped pixel with `de_px < θ` is speckle.
    #[arg(long, default_value_t = 1.0)]
    pub theta: f64,

    /// Window size K (K×K) for the windowed-max busyness label.
    #[arg(long, default_value_t = 5)]
    pub window: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Base frame width for source-A boundary neighborhoods (the `BASE` scale).
    #[arg(long, default_value_t = 0.08)]
    pub base_width_a: f64,

    /// Base frame width for the source-B signature scan and its tiles.
    #[arg(long, default_value_t = 0.05)]
    pub base_width_b: f64,

    /// Source-C minibrot framing: base width = `|size| · frame_multiple`.
    #[arg(long, default_value_t = 8.0)]
    pub frame_multiple: f64,

    /// Source-B filter (on trial): keep only frames with `interior_frac` below this.
    #[arg(long, default_value_t = 0.15)]
    pub b_interior_max: f64,

    /// Source-B filter (on trial): keep only frames whose in-band `[2,14]` de_px
    /// fraction (`coverage`) is above this (tight de_px spread). Default set
    /// relative to the achievable ceiling — the harvest's best whole-frame coverage
    /// was 0.339, so 0.35+ selects nothing; 0.15 picks meaningfully-covered frames.
    #[arg(long, default_value_t = 0.15)]
    pub b_coverage_min: f64,

    /// Light pre-filter: drop a candidate base frame whose `interior_frac` exceeds
    /// this (pure interior). Applied to A/B selection; C is kept regardless (the
    /// minibrot body is the control's whole point).
    #[arg(long, default_value_t = 0.60)]
    pub interior_max: f64,

    /// Coarse-scan candidate budget for the source-B random sample.
    #[arg(long, default_value_t = 800)]
    pub scan_tries: usize,

    /// Broad scan region `re_lo,re_hi,im_lo,im_hi` (the main-set neighborhood the
    /// B sample draws from).
    #[arg(long, default_value = "-1.8,0.45,-1.15,1.15", allow_hyphen_values = true)]
    pub scan_region: String,

    /// maxiter schedule base: `maxiter = round(base + per_decade·log10(3/width))`.
    #[arg(long, default_value_t = 1000.0)]
    pub maxiter_base: f64,

    /// maxiter schedule slope (iterations added per decade of zoom past width 3).
    #[arg(long, default_value_t = 1500.0)]
    pub per_decade: f64,

    /// RNG seed for the source-B random scan (deterministic for a fixed seed).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Orbit-trap shape.
    #[arg(long, value_enum, default_value_t = TrapShape::Point)]
    pub trap: TrapShape,

    /// Orbit-trap center as `re,im`.
    #[arg(long, default_value = "0,0")]
    pub trap_center: String,

    /// Orbit-trap radius (circle trap only).
    #[arg(long, default_value_t = 1.0)]
    pub trap_radius: f64,

    /// Output directory for the per-source sheets.
    #[arg(long, default_value = "out/buffet")]
    pub out_dir: String,

    /// Flat per-tile metrics table (JSON) path.
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub json: String,
}

impl BuffetArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Parse `--scan-region` (`re_lo,re_hi,im_lo,im_hi`) into bounds.
    pub fn resolved_scan_region(&self) -> Result<(f64, f64, f64, f64), String> {
        let p: Vec<&str> = self.scan_region.split(',').collect();
        if p.len() != 4 {
            return Err(format!(
                "invalid --scan-region '{}', expected re_lo,re_hi,im_lo,im_hi",
                self.scan_region
            ));
        }
        let parse = |s: &str, what: &str| -> Result<f64, String> {
            s.trim()
                .parse()
                .map_err(|_| format!("invalid --scan-region {what} in '{}'", self.scan_region))
        };
        let re_lo = parse(p[0], "re_lo")?;
        let re_hi = parse(p[1], "re_hi")?;
        let im_lo = parse(p[2], "im_lo")?;
        let im_hi = parse(p[3], "im_hi")?;
        if re_hi <= re_lo || im_hi <= im_lo {
            return Err(format!("--scan-region bounds must be lo < hi in '{}'", self.scan_region));
        }
        Ok((re_lo, re_hi, im_lo, im_hi))
    }
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

    /// DE-coherence sub-pixel threshold θ: an escaped pixel with `de_px < θ` (at
    /// the wallpaper spacing) is sub-pixel-boundary speckle. The descent rejects
    /// windows over the speckle fraction and soft-penalizes borderline ones.
    #[arg(long, default_value_t = 1.0)]
    pub coherence_theta: f64,

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

/// `cover` subcommand: single-frame **coverage-dominance** scorer (Prompt
/// coverage-dominance). Renders one frame f64 and reports `coverage` (the fraction
/// of the escaped frame whose boundary is a few pixels wide at the target spacing),
/// the speckle / interior / busy gates, and the gate verdict. Always emits a
/// band-sensitivity table (re-scoring the same buffer over several `[lo,hi]` bands);
/// with `--scale-sweep n` it re-renders the same center at `n` log-spaced widths to
/// check whether a different zoom lifts a boundary-packed frame into the band.
#[derive(Args, Debug)]
pub struct CoverArgs {
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

    /// Probe render width in pixels (height follows 16:9). `de_px` is taken against
    /// `--target-width`, not this.
    #[arg(long, default_value_t = 640)]
    pub panel_width: u32,

    /// Linear supersampling factor (S×S) for the probe render.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Target wallpaper width `de_px` is pinned to (resolution-invariant `de`).
    #[arg(long, default_value_t = 2560)]
    pub target_width: u32,

    /// Sub-pixel threshold θ: an escaped pixel with `de_px < θ` is speckle.
    #[arg(long, default_value_t = 1.0)]
    pub theta: f64,

    /// Window size K (K×K) for the windowed-max busyness richness floor.
    #[arg(long, default_value_t = 5)]
    pub window: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Orbit-trap shape (matches the harvest default).
    #[arg(long, value_enum, default_value_t = TrapShape::Point)]
    pub trap: TrapShape,

    /// Orbit-trap center as `re,im`.
    #[arg(long, default_value = "0,0")]
    pub trap_center: String,

    /// Orbit-trap radius (circle trap only).
    #[arg(long, default_value_t = 1.0)]
    pub trap_radius: f64,

    /// Override the coverage band low edge `de_px` (default: the module const 2.0).
    #[arg(long)]
    pub cover_lo: Option<f64>,

    /// Override the coverage band high edge `de_px` (default: the const 14.0).
    #[arg(long)]
    pub cover_hi: Option<f64>,

    /// Override the speckle reject cap `subpixel_frac` (default: 0.12).
    #[arg(long)]
    pub spx_cap: Option<f64>,

    /// Override the interior reject cap `interior_frac` (default: 0.30).
    #[arg(long)]
    pub int_cap: Option<f64>,

    /// Override the coverage floor reject `coverage` (default: 0.45).
    #[arg(long)]
    pub cover_min: Option<f64>,

    /// Override the windowed-busyness richness floor (default: 0.02).
    #[arg(long)]
    pub busy_floor: Option<f64>,

    /// Scale-sweep step count: re-render the same center at `n` log-spaced widths in
    /// `[frame_width·scale_lo, frame_width·scale_hi]`. `0` (default) → no sweep.
    #[arg(long, default_value_t = 0)]
    pub scale_sweep: usize,

    /// Low multiplier of `frame_width` for the scale sweep (zoom *in*; <1).
    #[arg(long, default_value_t = 0.25)]
    pub scale_lo: f64,

    /// High multiplier of `frame_width` for the scale sweep (zoom *out*; >1).
    #[arg(long, default_value_t = 16.0)]
    pub scale_hi: f64,

    /// Label for the printed `COVER` row / JSON.
    #[arg(long, default_value = "frame")]
    pub label: String,

    /// Optional JSON sidecar path.
    #[arg(long)]
    pub json: Option<String>,
}

impl CoverArgs {
    /// Parse `--trap-center` (`re,im`) into a complex number.
    pub fn resolved_trap_center(&self) -> Result<Complex<f64>, String> {
        parse_complex(&self.trap_center, "--trap-center")
    }

    /// Effective coverage params: each field overridden by its flag if present.
    pub fn coverage_params(&self) -> crate::coherence::CoverageParams {
        let d = crate::coherence::CoverageParams::default();
        crate::coherence::CoverageParams {
            cover_lo: self.cover_lo.unwrap_or(d.cover_lo),
            cover_hi: self.cover_hi.unwrap_or(d.cover_hi),
            spx_cap: self.spx_cap.unwrap_or(d.spx_cap),
            int_cap: self.int_cap.unwrap_or(d.int_cap),
            cover_min: self.cover_min.unwrap_or(d.cover_min),
            busy_floor: self.busy_floor.unwrap_or(d.busy_floor),
        }
    }
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

    /// **Broad shallow seed count** (Prompt broad-shallow-harvest). When `> 0`, the
    /// root frontier is seeded by a coarse global atom-domain scan deduped
    /// *spatially* — keeping up to this many distinct low-period nuclei spread
    /// across the base region, instead of `beam_width`. The source of breadth for
    /// the shallow harvest. `0` (default) → legacy single-root-expand behaviour.
    #[arg(long, default_value_t = 0)]
    pub seed_count: usize,

    /// Spatial-dedup cell size (panel px) for the broad seed scan: one nucleus per
    /// `seed_cell_px²` cell. Smaller → more, closer-packed seeds. Only used when
    /// `--seed-count > 0`.
    #[arg(long, default_value_t = 24)]
    pub seed_cell_px: u32,

    /// **Off-nucleus drift drive** (Prompt offnucleus-deband, Phase 4). When set,
    /// each expanded frame is re-centered: render at the nucleus, find the best
    /// contiguous in-band `de_px`-band region (the decoration), drift the frame
    /// center toward its centroid (clamped to `coherence::DRIFT_MAX`), re-render,
    /// and **surface/rank candidates by band reward**, not busyness — framing the
    /// decoration around a minibrot instead of its cusp. Off → the existing
    /// busyness-ranked search is unchanged.
    #[arg(long, default_value_t = false)]
    pub drift: bool,

    /// Target wallpaper width in pixels the DE-coherence gate is pinned to:
    /// `de_px = de / (frame_width / target_width)`. The panels are cheap
    /// thumbnails, but `de` is resolution-invariant, so a thumbnail predicts the
    /// final render's speckle gate. Keep at the eventual wallpaper width (2560).
    #[arg(long, default_value_t = 2560)]
    pub target_width: u32,

    /// DE-coherence sub-pixel threshold θ: an escaped pixel with `de_px < θ` (at
    /// the target spacing) counts as sub-pixel-boundary speckle.
    #[arg(long, default_value_t = 1.0)]
    pub coherence_theta: f64,

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

    /// **Magnification ceiling** (Prompt broad-shallow-harvest): don't expand a
    /// node deeper than this magnification. The depth analog of `--period-cap`: a
    /// chain of low-period descents nests deep even under a low period cap, and the
    /// busyness-priority frontier dives there, wasting budget on sub-pixel deep
    /// frames. Capping mag bounds the harvest to the shallow decoration regime.
    /// `0` (default) → unlimited (default search unchanged).
    #[arg(long, default_value_t = 0.0)]
    pub max_mag: f64,

    /// Also render the parallel base-scale Julia column in the best-path strip.
    /// Off by default: a good Mandelbrot region implies a good Julia, so the
    /// automated render is pure cost. The on-demand `render --julia` is unaffected.
    #[arg(long, default_value_t = false)]
    pub with_julia: bool,

    /// Fixed maxiter for every Julia panel (base-scale, shallow). Only used when
    /// `--with-julia` is set.
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

    /// **Coverage-dominance harvest** (Prompt coverage-dominance). When set, the
    /// whole drive turns away from the boundary: instead of the deep child-nucleus
    /// frontier descent, it broad-spatial-seeds the base frame, **scale-optimizes**
    /// each seed (sweeps `frame_multiple ∈ [scale_min, scale_max]` and keeps the zoom
    /// that maximizes `coverage` — the fraction of the frame at the magical
    /// few-pixels boundary spacing — subject to the speckle/interior/coverage gates),
    /// drifts the center onto the coverage peak, and ranks survivors by `coverage`.
    /// No descent chain; staying shallow is the point. Off → the existing
    /// busyness/band frontier search is unchanged.
    #[arg(long, default_value_t = false)]
    pub coverage: bool,

    /// Coverage band low edge `de_px` (default: module const 2.0). Only in `--coverage`.
    #[arg(long)]
    pub cover_lo: Option<f64>,

    /// Coverage band high edge `de_px` (default: 14.0). Only in `--coverage`.
    #[arg(long)]
    pub cover_hi: Option<f64>,

    /// Speckle reject cap `subpixel_frac` (default: 0.12). Only in `--coverage`.
    #[arg(long)]
    pub spx_cap: Option<f64>,

    /// Interior reject cap `interior_frac` (default: 0.30). Only in `--coverage`.
    #[arg(long)]
    pub int_cap: Option<f64>,

    /// Coverage floor reject `coverage` (default: 0.45). Only in `--coverage`.
    #[arg(long)]
    pub cover_min: Option<f64>,

    /// Windowed-busyness richness floor (default: 0.02). Only in `--coverage`.
    #[arg(long)]
    pub cover_busy_floor: Option<f64>,

    /// Scale-sweep low multiple of the minibrot `|size|` (zoom in; smaller frame).
    /// Only in `--coverage`.
    #[arg(long, default_value_t = 4.0)]
    pub scale_min: f64,

    /// Scale-sweep high multiple of the minibrot `|size|` (zoom out; wider frame).
    /// Only in `--coverage`.
    #[arg(long, default_value_t = 64.0)]
    pub scale_max: f64,

    /// Number of log-spaced scales swept per seed. Only in `--coverage`.
    #[arg(long, default_value_t = 8)]
    pub scale_steps: usize,
}

impl SearchArgs {
    /// Effective coverage params: each field overridden by its flag if present.
    pub fn coverage_params(&self) -> crate::coherence::CoverageParams {
        let d = crate::coherence::CoverageParams::default();
        crate::coherence::CoverageParams {
            cover_lo: self.cover_lo.unwrap_or(d.cover_lo),
            cover_hi: self.cover_hi.unwrap_or(d.cover_hi),
            spx_cap: self.spx_cap.unwrap_or(d.spx_cap),
            int_cap: self.int_cap.unwrap_or(d.int_cap),
            cover_min: self.cover_min.unwrap_or(d.cover_min),
            busy_floor: self.cover_busy_floor.unwrap_or(d.busy_floor),
        }
    }
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

    /// Also render the parallel base-scale Julia column in the filmstrip. Off by
    /// default: a good Mandelbrot region implies a good Julia, so the automated
    /// render is pure cost. The on-demand `render --julia` is unaffected.
    #[arg(long, default_value_t = false)]
    pub with_julia: bool,

    /// Fixed maxiter for every Julia panel (base-scale, shallow). Only used when
    /// `--with-julia` is set.
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

    /// Also render the parallel base-scale Julia column in the filmstrip. Off by
    /// default: a good Mandelbrot region implies a good Julia, so the automated
    /// render is pure cost. The on-demand `render --julia` is unaffected.
    #[arg(long, default_value_t = false)]
    pub with_julia: bool,

    /// Fixed maxiter for every Julia panel (base-scale, shallow). Only used when
    /// `--with-julia` is set.
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

/// `generate` subcommand: see `generate::run_generate`. Promotes probe 1's
/// discovery sampler to a real generator — draw → cheap screen → accept band →
/// keep, run to a target keeper count K, persisting the full per-image log vector
/// (`locations.jsonl`) + a run manifest under `data/generated/<run>/`, plus an
/// annotated keeper contact sheet. The accept band is centralized
/// (`generate::AcceptBand`); its three flags are the deferred boundary-retune knob
/// (FLAG: band not yet eye-validated). Defaults reproduce probe 1's keeper stream.
#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Target keeper count: draw until this many keepers pass the band (capped by
    /// `--max-draws`).
    #[arg(long, default_value_t = 50)]
    pub keepers: usize,

    /// SplitMix64 seed (deterministic). Default = probe 1's seed, so the
    /// probe-1-default config reproduces the probe-1 keeper stream.
    #[arg(long, default_value_t = 20_260_622)]
    pub seed: u64,

    /// Sampling box `re_lo,re_hi,im_lo,im_hi` (uniform center draw).
    #[arg(long = "box", default_value = "-2.0,0.7,-1.2,1.2", allow_hyphen_values = true)]
    pub box_bounds: String,

    /// Log-uniform frame-width range, low edge (shallow cap; f64 cheap regime).
    #[arg(long, default_value_t = 0.003)]
    pub fw_lo: f64,

    /// Log-uniform frame-width range, high edge.
    #[arg(long, default_value_t = 0.05)]
    pub fw_hi: f64,

    /// Cheap-screen render width in px (height follows 16:9; ss1). Every draw.
    #[arg(long, default_value_t = 320)]
    pub screen_width: u32,

    /// Maximum iterations for the screen + keeper renders.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Accept-band override: middle-90% smooth-iter spread floor (default from
    /// the label-retuned `AcceptBand`).
    #[arg(long)]
    pub spread_min: Option<f64>,

    /// Accept-band override: interior (max-iter) fraction cap.
    #[arg(long)]
    pub interior_max: Option<f64>,

    /// Accept-band override: escape-median smooth-iter floor.
    #[arg(long)]
    pub esc_median_min: Option<f64>,

    /// Max-draws safeguard (0 → K×500). The loop stops and reports if it hits this
    /// before reaching K keepers.
    #[arg(long, default_value_t = 0)]
    pub max_draws: usize,

    /// Keeper contact-sheet thumbnail width in px (height follows 16:9).
    #[arg(long, default_value_t = 256)]
    pub thumb_width: u32,

    /// Keeper contact-sheet grid columns.
    #[arg(long, default_value_t = 6)]
    pub cols: usize,

    /// Corpus calibration artifact (frozen bins + per-image signatures) for the
    /// keeper-only descriptor feature.
    #[arg(long, default_value = "data/calibration/energy_calibration.json")]
    pub artifact: String,

    /// Preview colormap name (from `data/palettes/clean_colormaps.json`) for the
    /// keeper thumbnails/sheet only — purely cosmetic, structure-finding is
    /// palette-independent and the 3-palette labeling stage is downstream.
    #[arg(long, default_value = "cubehelix")]
    pub palette: String,

    /// Output directory for the batch (`locations.jsonl`, `manifest.json`,
    /// `keeper_sheet.png`, `thumbs/`). Outside `out/` — a durable location store.
    /// Use a distinct dir per batch.
    #[arg(long, default_value = "data/generated/run0")]
    pub out_dir: String,
}

impl GenerateArgs {
    /// Parse `--box` (`re_lo,re_hi,im_lo,im_hi`) into bounds.
    pub fn resolved_box(&self) -> Result<(f64, f64, f64, f64), String> {
        let p: Vec<&str> = self.box_bounds.split(',').collect();
        if p.len() != 4 {
            return Err(format!(
                "invalid --box '{}', expected re_lo,re_hi,im_lo,im_hi",
                self.box_bounds
            ));
        }
        let parse = |s: &str, what: &str| -> Result<f64, String> {
            s.trim()
                .parse()
                .map_err(|_| format!("invalid --box {what} in '{}'", self.box_bounds))
        };
        let re_lo = parse(p[0], "re_lo")?;
        let re_hi = parse(p[1], "re_hi")?;
        let im_lo = parse(p[2], "im_lo")?;
        let im_hi = parse(p[3], "im_hi")?;
        if re_hi <= re_lo || im_hi <= im_lo {
            return Err(format!("--box bounds must be lo < hi in '{}'", self.box_bounds));
        }
        Ok((re_lo, re_hi, im_lo, im_hi))
    }

    /// Effective accept band: each clause overridden by its flag if present, else
    /// the centralized `generate::AcceptBand` default (FLAG: one-place retune).
    pub fn band(&self) -> crate::generate::AcceptBand {
        let d = crate::generate::AcceptBand::default();
        crate::generate::AcceptBand {
            spread_min: self.spread_min.unwrap_or(d.spread_min),
            interior_max: self.interior_max.unwrap_or(d.interior_max),
            esc_median_min: self.esc_median_min.unwrap_or(d.esc_median_min),
        }
    }
}

/// `guided-descend` subcommand: see `guided_descend::run_guided_descend`.
/// Stochastic guided descent from the fixed base-Mandelbrot root; geometric
/// policy only (no CNN). Reuses the `generate` cheap screen + `AcceptBand` and a
/// freshly-built μ scale-space focus finder.
#[derive(Args, Debug)]
pub struct GuidedDescendArgs {
    /// Number of independent, decorrelated walks (each starts at the root).
    #[arg(long, default_value_t = 80)]
    pub n_walks: usize,

    /// Minimum terminal walk depth (inclusive). rev3: raised 3→4 (the shallow
    /// d3 frames were too zoomed-out / set-dominated to be useful wallpapers).
    #[arg(long, default_value_t = 4)]
    pub depth_min: u32,

    /// Maximum terminal walk depth (inclusive).
    #[arg(long, default_value_t = 10)]
    pub depth_max: u32,

    /// Per-step zoom factor (`child.fw = parent.fw × this`). Applies to the
    /// normal mid-descent steps (depth ≥ 2); the root step uses `--root-zoom`.
    #[arg(long, default_value_t = 0.4)]
    pub zoom_per_step: f64,

    /// Root-step zoom factor (the first jump is its own thing — base fw 3.0 ×
    /// this → depth-1 window). Default 0.08 ⇒ depth-1 fw ≈ 0.24. Much bigger than
    /// a mid-descent step because the root is the whole Mandelbrot view.
    #[arg(long, default_value_t = 0.08)]
    pub root_zoom: f64,

    /// Target-policy weight: descend into a detected μ-focus (renormalized).
    #[arg(long, default_value_t = 0.85)]
    pub w_foci: f64,

    /// Target-policy weight: descend into the energy-weighted density focus.
    #[arg(long, default_value_t = 0.10)]
    pub w_density: f64,

    /// Target-policy weight: descend into a random interior point (explore floor).
    #[arg(long, default_value_t = 0.05)]
    pub w_random: f64,

    /// Placement mixture `center,horizon,random` (where the chosen target lands in
    /// the child frame). Applied to the foci/density branches; the random branch
    /// is always centered. Lowered center vs run0 (was 0.50,0.25,0.25) to
    /// decorrelate the shallow layers — this is the repetition dial.
    #[arg(long, default_value = "0.25,0.40,0.35")]
    pub placement: String,

    /// Focus-finder σ band in field px (comma-separated). Persistence is measured
    /// within this band; the sampling score is peak×isolation (NOT persistence).
    #[arg(long, default_value = "16,20,24,28,32")]
    pub sigma_band: String,

    /// Cheap field/screen render width in px (height follows 16:9, ss1). Drives
    /// both the AcceptBand screen and the μ-field the focus finder reads. Default
    /// 768 so the σ band {16..32} runs at the frame-fraction it was tuned at
    /// (run0 ran 256 → σ ~3× too coarse). Alias `--node-size`.
    #[arg(long, alias = "node-size", default_value_t = 768)]
    pub node_width: u32,

    /// Preview render width in px (height follows 16:9, ss1).
    #[arg(long, default_value_t = 640)]
    pub preview_width: u32,

    /// Preview colormap name (from the colormaps JSON) — diagnostic only.
    /// `twilight_shifted` is cyclic (seam-free, mirror-moot).
    #[arg(long, default_value = "twilight_shifted")]
    pub preview_palette: String,

    /// Colormap library JSON (for the preview palette).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Maximum iterations for the field/screen and preview renders.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Accept-band override: middle-90% smooth-iter spread floor.
    #[arg(long)]
    pub spread_min: Option<f64>,

    /// Accept-band override: interior (max-iter) fraction cap.
    #[arg(long)]
    pub interior_max: Option<f64>,

    /// Accept-band override: escape-median smooth-iter floor.
    #[arg(long)]
    pub esc_median_min: Option<f64>,

    /// Best-of-N candidate count per step (rev3 Change 2). Each step draws this
    /// many candidate next-centers from the per-node policy and two-stage screens
    /// them (cheap interior cap → 768 occupancy floor) before selecting the
    /// least-set survivor. 1 reproduces rev2's accept-first behaviour.
    #[arg(long, default_value_t = 4)]
    pub descent_candidates: usize,

    /// Best-of-N **Stage 1** interior/black cap (rev3, lowered 0.45→0.30 — the
    /// aggressive set-avoidance hard ceiling). Each candidate is probed at ~128px
    /// escape-time and rejected if its `render::black_fraction` (interior counts as
    /// black) is ≥ this; interior fraction is scale-robust so the cheap probe is
    /// fine. Default-on at 0.30; 0 or ≥1.0 disables. Black/interior-fraction ONLY
    /// (busyness is known-unseparable by magnitude).
    #[arg(long, default_value_t = 0.30)]
    pub descent_black_cap: f64,

    /// Best-of-N **Stage 2** occupancy floor (rev3, reuses present's 0.321). Stage-1
    /// survivors are rendered at the 768 node size, shaded, and scored with the
    /// `energy::occupancy` parity scorer; candidates below this are rejected (keeps
    /// walks feature-rich, not empty). Among the rest the **least interior** wins.
    /// Default-on at 0.321; 0 or ≥1.0 disables. CAVEAT: 0.321 was calibrated on the
    /// labeling-crop distribution, not navigation frames — a tunable knob.
    #[arg(long, default_value_t = 0.321)]
    pub descent_occ_floor: f64,

    /// Flat-grid PNG columns.
    #[arg(long, default_value_t = 10)]
    pub cols: usize,

    /// SplitMix64 seed (deterministic).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Output directory (`pool_sheet.html`, `pool.jsonl`, `pool_grid.png`,
    /// `tiles/`). Outside `out/` — durable. Use a distinct dir per run.
    #[arg(long, default_value = "data/guided_descend/run1")]
    pub out_dir: String,
}

impl GuidedDescendArgs {
    /// Effective accept band (each clause flag-overridable; shared default).
    pub fn band(&self) -> crate::generate::AcceptBand {
        let d = crate::generate::AcceptBand::default();
        crate::generate::AcceptBand {
            spread_min: self.spread_min.unwrap_or(d.spread_min),
            interior_max: self.interior_max.unwrap_or(d.interior_max),
            esc_median_min: self.esc_median_min.unwrap_or(d.esc_median_min),
        }
    }

    /// Parse `--placement` (`center,horizon,random`) raw weights (un-normalized).
    pub fn resolved_placement(&self) -> Result<(f64, f64, f64), String> {
        let p: Vec<&str> = self.placement.split(',').collect();
        if p.len() != 3 {
            return Err(format!("invalid --placement '{}', expected center,horizon,random", self.placement));
        }
        let parse = |s: &str| -> Result<f64, String> {
            s.trim().parse::<f64>().map_err(|_| format!("invalid --placement value '{}'", s.trim()))
        };
        let (c, h, r) = (parse(p[0])?, parse(p[1])?, parse(p[2])?);
        if c < 0.0 || h < 0.0 || r < 0.0 || c + h + r <= 0.0 {
            return Err("--placement weights must be non-negative and sum > 0".into());
        }
        Ok((c, h, r))
    }

    /// Parse `--sigma-band` (comma-separated px) into ascending σ values.
    pub fn resolved_sigmas(&self) -> Result<Vec<f64>, String> {
        let mut v: Vec<f64> = Vec::new();
        for s in self.sigma_band.split(',') {
            let x: f64 = s.trim().parse().map_err(|_| format!("invalid --sigma-band value '{}'", s.trim()))?;
            if x <= 0.0 {
                return Err(format!("--sigma-band values must be > 0 (got {x})"));
            }
            v.push(x);
        }
        if v.is_empty() {
            return Err("--sigma-band is empty".into());
        }
        Ok(v)
    }
}

/// `present` subcommand: see `present::run_present`. Takes a `locations.jsonl`
/// from a `generate` run and produces presentation-ready crops. Zooms in on each
/// seed center, tries three composition offsets at cheap 320×180 resolution,
/// picks the one with the lowest black fraction, gates on < 40% black, then
/// renders at full resolution across random palettes.
#[derive(Args, Debug)]
pub struct PresentArgs {
    /// Path to `locations.jsonl` from a `generate` run (required).
    #[arg(long)]
    pub input: String,

    /// Output directory root; a `<run_stem>/` subdirectory is created inside.
    #[arg(long, default_value = "out/present/")]
    pub out_dir: String,

    /// Full-resolution render width in pixels.
    #[arg(long, default_value_t = 1920)]
    pub width: u32,

    /// Full-resolution render height in pixels.
    #[arg(long, default_value_t = 1080)]
    pub height: u32,

    /// Linear supersampling factor for the full-resolution render.
    #[arg(long, default_value_t = 2)]
    pub ss: u32,

    /// Zoom factor: `new_fw = seed_fw × this`.
    #[arg(long, default_value_t = 0.4)]
    pub zoom_factor: f64,

    /// Which composition offsets to try: "center", "thirds", "golden", or "all".
    #[arg(long, default_value = "all")]
    pub compositions: String,

    /// Path to the colormap library JSON.
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub palette_file: String,

    /// Number of random palettes to apply per accepted crop.
    #[arg(long, default_value_t = 3)]
    pub palettes_per_crop: usize,

    /// Emit a distinct crop for **every** composition that passes the black gate
    /// (labeling mode: each (seed × composition) is its own location to judge),
    /// instead of the default pick-the-lowest-black single crop per seed.
    #[arg(long, default_value_t = false)]
    pub all_compositions: bool,

    /// Maximum iterations / orbit cap ("max_orbit") for both cheap-screen and
    /// full-resolution renders. Raised 1000 → 8000 (the `maxiter-blackgate`
    /// pass, Matt's pick = the measured escalation knee): resolves the pinned
    /// spiral-core fringe so the black gate sees true interior, not under-iterated
    /// pixels. The gate (`BLACK_THRESH`) is calibrated against the no-escape
    /// distribution at this cap.
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// SplitMix64 seed for reproducible palette selection.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Detail-occupancy gate threshold applied **after** full-res iteration and
    /// **before** palettes: discard the (seed × composition) crop if its
    /// occupancy (fraction of 32×18 tiles with mean edge energy > 0.010) is below
    /// this. `0` disables the gate (legacy behaviour). The loose0 calibration
    /// floor was 0.23; gate-diag (loose0_v3) raised it to 0.321 (now the
    /// default) — the low occupancy tail's bottom decile is ~96% doomed
    /// (geo_label==1) and holds zero label-3 crops (min label-3 occupancy
    /// 0.4184), so 0.321 rejects ~52 geometries at zero good-crop cost.
    #[arg(long, default_value_t = 0.321)]
    pub occupancy_floor: f64,

    /// Output image format for emitted crops: `png` or `jpg`.
    #[arg(long, default_value = "png")]
    pub format: String,

    /// JPEG quality (1..=100) when `--format jpg`.
    #[arg(long, default_value_t = 90)]
    pub jpg_quality: u8,

    /// Write crops directly into `--out-dir` instead of an `<run_stem>/`
    /// subdirectory (avoids the doubled `loose0/loose0` nesting).
    #[arg(long, default_value_t = false)]
    pub flat_out: bool,

    /// Crop focus: `content` = energy-weighted centroid of a cheap edge-energy
    /// screen over the seed frame (re-frame on where structure actually is, with
    /// a peak-tile void guard + in-frame clamp); `seed-center` = the raw seed
    /// center (legacy / comparison fallback). Both focuses are always rendered
    /// for the side-by-side `focus_compare.html`; this only picks which one the
    /// emitted batch uses.
    #[arg(long, default_value = "content")]
    pub focus: String,
}

/// `reject-corridor` subcommand: see `reject_corridor::run_reject_corridor`.
/// Diagnosis-only audit of the `generate` accept-band detail-floor. Shares
/// `generate`'s regime (box / fw-range / screen / band) so the corridor is
/// defined against the same boundary `generate` ships; only the seed differs by
/// default (fresh seed → reject draws `generate` never logged). Renders the
/// corridor at keeper resolution under the same preview palette.
#[derive(Args, Debug)]
pub struct RejectCorridorArgs {
    /// Total draws to screen (all logged, kept or not). Fresh seed below, so this
    /// is a new draw stream from `generate`'s run0/run1.
    #[arg(long, default_value_t = 2000)]
    pub draws: usize,

    /// SplitMix64 seed — distinct from `generate`'s default so the rejects logged
    /// here are a fresh stream (run1 logged keepers only).
    #[arg(long, default_value_t = 20_260_623)]
    pub seed: u64,

    /// Sampling box `re_lo,re_hi,im_lo,im_hi` (matches `generate`).
    #[arg(long = "box", default_value = "-2.0,0.7,-1.2,1.2", allow_hyphen_values = true)]
    pub box_bounds: String,

    /// Log-uniform frame-width range, low edge (matches `generate`).
    #[arg(long, default_value_t = 0.003)]
    pub fw_lo: f64,

    /// Log-uniform frame-width range, high edge (matches `generate`).
    #[arg(long, default_value_t = 0.05)]
    pub fw_hi: f64,

    /// Cheap-screen render width in px (height follows 16:9; ss1). Every draw.
    #[arg(long, default_value_t = 320)]
    pub screen_width: u32,

    /// Maximum iterations for the screen + corridor renders (matches `generate`).
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Escape radius.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Corridor lower spread edge ("not-flat" floor — the confirmed bad-sparse
    /// SPRD ceiling). Draws below this are the *flat* bulk.
    #[arg(long, default_value_t = 24.0)]
    pub corridor_lo: f64,

    /// Corridor upper spread edge (the confirmed good-anchor SPRD floor). Draws
    /// above this are clear anchors, not corridor.
    #[arg(long, default_value_t = 85.0)]
    pub corridor_hi: f64,

    /// Max corridor tiles to render (evenly sampled by spread across the corridor
    /// if more are found). The all-draw log keeps every corridor draw regardless.
    #[arg(long, default_value_t = 48)]
    pub max_corridor: usize,

    /// Representatives to render per bulk bucket (flat, interior-black).
    #[arg(long, default_value_t = 3)]
    pub bulk_reps: usize,

    /// Accept-band override: spread floor (default from `generate::AcceptBand`).
    /// This is the live floor the corridor's CUT/KEEP split is drawn against.
    #[arg(long)]
    pub spread_min: Option<f64>,

    /// Accept-band override: interior (max-iter) fraction cap.
    #[arg(long)]
    pub interior_max: Option<f64>,

    /// Accept-band override: escape-median smooth-iter floor.
    #[arg(long)]
    pub esc_median_min: Option<f64>,

    /// Corridor / bulk thumbnail width in px (height follows 16:9).
    #[arg(long, default_value_t = 256)]
    pub thumb_width: u32,

    /// Corridor contact-sheet grid columns.
    #[arg(long, default_value_t = 8)]
    pub cols: usize,

    /// Preview colormap name (from `data/palettes/clean_colormaps.json`); default
    /// cubehelix — same preview palette as `generate`.
    #[arg(long, default_value = "cubehelix")]
    pub palette: String,

    /// Output directory (`draws.jsonl`, `manifest.json`, `corridor_sheet.png`,
    /// `bulk_sheet.png`). Outside `out/` — the all-draw log is the durable artifact.
    #[arg(long, default_value = "data/generated/reject_corridor")]
    pub out_dir: String,
}

impl RejectCorridorArgs {
    /// Parse `--box` (`re_lo,re_hi,im_lo,im_hi`) into bounds (same shape as
    /// `GenerateArgs::resolved_box`).
    pub fn resolved_box(&self) -> Result<(f64, f64, f64, f64), String> {
        let p: Vec<&str> = self.box_bounds.split(',').collect();
        if p.len() != 4 {
            return Err(format!(
                "invalid --box '{}', expected re_lo,re_hi,im_lo,im_hi",
                self.box_bounds
            ));
        }
        let parse = |s: &str, what: &str| -> Result<f64, String> {
            s.trim()
                .parse()
                .map_err(|_| format!("invalid --box {what} in '{}'", self.box_bounds))
        };
        let re_lo = parse(p[0], "re_lo")?;
        let re_hi = parse(p[1], "re_hi")?;
        let im_lo = parse(p[2], "im_lo")?;
        let im_hi = parse(p[3], "im_hi")?;
        if re_hi <= re_lo || im_hi <= im_lo {
            return Err(format!("--box bounds must be lo < hi in '{}'", self.box_bounds));
        }
        Ok((re_lo, re_hi, im_lo, im_hi))
    }

    /// Effective accept band (each clause flag-overridable; default = the band
    /// `generate` ships). The corridor is sliced against this same boundary.
    pub fn band(&self) -> crate::generate::AcceptBand {
        let d = crate::generate::AcceptBand::default();
        crate::generate::AcceptBand {
            spread_min: self.spread_min.unwrap_or(d.spread_min),
            interior_max: self.interior_max.unwrap_or(d.interior_max),
            esc_median_min: self.esc_median_min.unwrap_or(d.esc_median_min),
        }
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
