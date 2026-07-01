//! `render-one` subcommand — the **locked wallpaper-render default**.
//!
//! The AA study is settled: **grid ss4 + Lanczos-3 at 2560×1440** is the
//! production wallpaper quality. This gives that path a production home instead of
//! leaving it inside the `aa-filter` study harness. It is an **extract** of the
//! pieces `aa-filter` already built and verified — same f64 render path,
//! [`generate::color_params`], selective-mirror palette load, channel-intent
//! iterate, and [`render::shade_and_downsample_filtered`] with the `ss×`-scaled
//! reconstruction kernel — not new logic.
//!
//! Renders **one** (location × palette) at the locked quality to a caller-chosen
//! **stable** path (not `out/`), reporting iterate / filter / total wall-clock.
//! The locked values (`--width 2560 --height 1440 --ss 4 --pattern grid --filter
//! lanczos3`) live here only; the bare render path keeps its own defaults so fast
//! ss1/ss2 previews and diagnostics are unaffected.
//!
//! Shallow f64 by construction (asserted) — the discovery pipeline that produces
//! wallpaper locations (`generate` → `present`) works in the f64 cheap regime, so
//! deep-zoom perturbation is out of scope here. At 2560×1440 the ss4 transient
//! (~2.8 GB) is fine; tiled supersampling is the escape hatch for higher-res
//! masters (constant buffer regardless of ss/res) — noted, not built.

use std::time::Instant;

use num_complex::Complex;

use crate::backend::{F64Backend, JuliaBackend, Trap, TrapShape};
use clap::{Args, ValueEnum};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::render::{self, Frame};
use crate::render_modes::{self, ColoringParams};
use crate::{coloring, ensure_parent_dir, hp};

/// Resolve `--coloring` to a beautiful [`ColoringParams`], or `None` for the
/// **location profile** (the settled byte-identical path). `None` is returned for
/// an absent flag *and* for an explicit default (`--coloring '{}'` or any spec
/// that resolves to [`ColoringParams::default`]) — both must reproduce the current
/// output exactly. A `@path` prefix reads the JSON from a file.
fn resolve_coloring(arg: &Option<String>) -> Result<Option<ColoringParams>, String> {
    let raw = match arg {
        None => return Ok(None),
        Some(s) => s,
    };
    let text = if let Some(path) = raw.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|e| format!("read --coloring file {path}: {e}"))?
    } else {
        raw.clone()
    };
    let cp = ColoringParams::from_json(&text)?;
    Ok((!cp.is_location_profile()).then_some(cp))
}

/// Escape radius (matches the generate/present/aa-study/aa-filter regime).
const BAILOUT: f64 = 1e6;

pub fn run_render_one(args: &RenderOneArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.supersample.max(1);

    // Julia/parameter coupling: `--c` is required iff `--julia`, and forbidden
    // otherwise (so a stray `--c` on a Mandelbrot render is a loud error, not a
    // silently ignored flag).
    let julia_param = match (args.julia, &args.julia_c) {
        (true, None) => {
            return Err("--julia requires --c <re> <im> (the Julia parameter)".into());
        }
        (false, Some(_)) => {
            return Err("--c given without --julia; --c is the Julia parameter and \
                        is meaningless on a Mandelbrot render"
                .into());
        }
        (false, None) => None,
        (true, Some(c)) => {
            // num_args = 2 guarantees exactly two values; guard anyway.
            if c.len() != 2 {
                return Err(format!("--c expects exactly two values <re> <im>, got {}", c.len()));
            }
            Some((c[0].clone(), c[1].clone()))
        }
    };

    // Center at full precision (parsed exactly as render/aa-filter do, so the
    // f64 projection matches bit-for-bit).
    let prec_bits = hp::prec_bits(args.width, args.frame_width);
    let cx = hp::to_f64(&hp::parse_decimal(&args.center_re, prec_bits)?);
    let cy = hp::to_f64(&hp::parse_decimal(&args.center_im, prec_bits)?);
    let center = Complex::new(cx, cy);

    // Julia parameter `c`, parsed as decimal strings exactly like the center and
    // projected to f64 (Julia is a shallow base-scale render — f64 is accurate).
    let julia_param = match julia_param {
        Some((re, im)) => {
            let pr = hp::to_f64(&hp::parse_decimal(&re, prec_bits)?);
            let pi = hp::to_f64(&hp::parse_decimal(&im, prec_bits)?);
            Some(Complex::new(pr, pi))
        }
        None => None,
    };

    let frame = Frame {
        center,
        frame_width: args.frame_width,
        out_width: args.width,
        out_height: args.height,
    };
    let pixel_spacing = frame.pixel_size();

    // Shallow-regime assertion: the locked path is f64 ground truth. The wallpaper
    // discovery pipeline (generate → present) lives in this regime; deep-zoom
    // perturbation is deferred.
    if pixel_spacing <= 1e-13 {
        return Err(format!(
            "pixel spacing {pixel_spacing:.3e} is inside f64's quantization regime — \
             render-one is the shallow f64 wallpaper path (perturbation/floatexp deferred). \
             Use a shallower frame width."
        ));
    }

    // Rotated-grid (4-rooks) is an ss2-only placement; refuse rather than silently
    // fall back to grid centers.
    if args.pattern == PatternChoice::Rgss && ss != 2 {
        return Err(format!(
            "--pattern rgss is ss2-only (4-rooks); got --ss {ss}. Use grid or jitter."
        ));
    }

    // Palette through the SAME selective-mirror path as present/aa-filter, so any
    // library palette (cyclic or sequential `mirror_needed`) renders seam-free.
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library =
        parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    let cm = library
        .iter()
        .find(|c| c.name == args.palette)
        .ok_or_else(|| format!("palette '{}' not found in {}", args.palette, args.colormaps))?;
    let palette =
        Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed);

    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // Beautiful coloring (non-default `--coloring`) routes to the decoupled
    // field→shade→palette pipeline; absent/default routes to the byte-identical
    // location-profile path below.
    let coloring_params = resolve_coloring(&args.coloring)?;

    // --- field dump branch (serialize the raw smooth field, exit before coloring) ---
    if let Some(dump_path) = &args.dump_field {
        // The dumped field is the beautiful *smooth* field. Its only iterate-stage
        // input is the bailout: take it from an explicit `--coloring` smooth spec,
        // else the beautiful-smooth default (2^16). A non-smooth `--coloring` is a
        // loud error rather than a silently-ignored field.
        let field_params = match &coloring_params {
            Some(cp) if cp.field == render_modes::Field::Smooth => *cp,
            Some(cp) => {
                return Err(format!(
                    "--dump-field serializes the smooth field, but --coloring names field={:?}. \
                     Pass a smooth spec (e.g. '{{\"field\":\"smooth\"}}') or omit --coloring.",
                    cp.field
                ));
            }
            None => ColoringParams::beautiful(render_modes::Field::Smooth),
        };
        let t0 = Instant::now();
        let (field, sub_w, sub_h) = render_modes::smooth_field_supersampled(
            &frame,
            ss,
            args.maxiter,
            julia_param,
            &field_params,
        );
        let secs = t0.elapsed().as_secs_f64();

        // Raw binary: little-endian f32, row-major.
        ensure_parent_dir(dump_path)?;
        let mut bytes = Vec::with_capacity(field.len() * 4);
        for v in &field {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dump_path, &bytes)
            .map_err(|e| format!("write field {dump_path}: {e}"))?;

        // JSON sidecar alongside the binary. `width`/`height` are the *supersampled*
        // array dims (so binary length == width·height); `supersample` folds them to
        // the eval size (out = width/ss). Location is version-invariant render keys.
        let (kind, c_fields) = match &julia_param {
            Some(_) => {
                let c = args
                    .julia_c
                    .as_ref()
                    .expect("julia_param implies --c present");
                (
                    "julia",
                    format!(",\"c_re\":\"{}\",\"c_im\":\"{}\"", c[0], c[1]),
                )
            }
            None => ("mandelbrot", String::new()),
        };
        let sidecar = format!(
            "{{\"width\":{sub_w},\"height\":{sub_h},\"supersample\":{ss},\
             \"field\":\"smooth\",\"dtype\":\"f32\",\"layout\":\"row_major\",\
             \"bailout_b\":{bailout},\
             \"location\":{{\"kind\":\"{kind}\",\"cx\":\"{cx}\",\"cy\":\"{cy}\",\
             \"fw\":\"{fw}\",\"maxiter\":{maxiter}{c_fields}}}}}",
            bailout = field_params.bailout_b,
            cx = args.center_re,
            cy = args.center_im,
            fw = args.frame_width,
            maxiter = args.maxiter,
        );
        let json_path = match dump_path.strip_suffix(".bin") {
            Some(stem) => format!("{stem}.json"),
            None => format!("{dump_path}.json"),
        };
        std::fs::write(&json_path, sidecar).map_err(|e| format!("write sidecar {json_path}: {e}"))?;

        eprintln!(
            "dump-field: {sub_w}x{sub_h} (ss{ss}) smooth field → {dump_path} (+ {json_path}) in {secs:.2}s"
        );
        println!("=== render-one (dump-field) ===");
        println!("field:   {dump_path}");
        println!("sidecar: {json_path}");
        println!("dims:    {sub_w}x{sub_h} (supersampled, ss{ss})");
        println!("total:   {secs:.3}s");
        return Ok(());
    }

    let mode = match julia_param {
        Some(p) => format!("julia c=({:.9}, {:.9})", p.re, p.im),
        None => "mandelbrot".to_string(),
    };
    let coloring_label = match &coloring_params {
        Some(cp) => format!("field={:?} transform={:?} shade={:?} biomorph={:?} B={:.0}",
            cp.field, cp.transform, cp.shade, cp.biomorph, cp.bailout_b),
        None => "location-profile (smooth)".to_string(),
    };
    eprintln!(
        "render-one [{mode}]: center ({cx:.9}, {cy:.9}) fw {:.3e}  {}x{}  {} ss{ss}  {}  maxiter {}  palette '{}'  coloring: {coloring_label}",
        args.frame_width,
        args.width,
        args.height,
        args.pattern.label(),
        args.filter.label(),
        args.maxiter,
        palette.name()
    );

    // --- beautiful pipeline branch (global-normalized field render) ---
    if let Some(cp) = &coloring_params {
        let t0 = Instant::now();
        let img = render_modes::render_beautiful(
            &frame,
            ss,
            args.maxiter,
            julia_param,
            cp,
            &palette,
            args.filter.into(),
        );
        let secs = t0.elapsed().as_secs_f64();
        ensure_parent_dir(&args.out)?;
        let lower = args.out.to_ascii_lowercase();
        if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            render::save_jpeg(&img, std::path::Path::new(&args.out), args.jpg_quality)?;
        } else {
            img.save(&args.out)
                .map_err(|e| format!("failed to write {}: {e}", args.out))?;
        }
        eprintln!("wrote {} in {secs:.2}s (beautiful pipeline)", args.out);
        println!("=== render-one (beautiful) ===");
        println!("out:      {}", args.out);
        println!("coloring: {}", cp.to_json());
        println!("total:    {secs:.3}s");
        return Ok(());
    }

    // --- iterate (the only expensive stage) ---
    // Mandelbrot uses the channel-intent f64 path; Julia uses the JuliaBackend
    // (z₀ = pixel, fixed parameter). `channels` is irrelevant to Julia (it always
    // computes the gated trap and never a `dz`), so it is unused on that branch.
    let t_iter = Instant::now();
    let buf = match julia_param {
        None => {
            let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
            render::iterate_samples_f64_pattern(
                &backend,
                &frame,
                ss,
                channels,
                args.pattern.into(),
                args.seed,
            )
        }
        Some(param) => {
            let backend = JuliaBackend::new(param, args.maxiter, BAILOUT, trap);
            render::iterate_samples_julia_pattern(
                &backend,
                &frame,
                ss,
                args.pattern.into(),
                args.seed,
            )
        }
    };
    let iter_secs = t_iter.elapsed().as_secs_f64();
    if buf.glitched_pixels > 0 {
        eprintln!("warning: {} glitched pixel(s)", buf.glitched_pixels);
    }

    // --- shade + downsample with the locked reconstruction filter ---
    let t_filter = Instant::now();
    let img = render::shade_and_downsample_filtered(
        &buf.samples,
        args.width,
        args.height,
        ss,
        &palette,
        &params,
        pixel_spacing,
        args.filter.into(),
    );
    let filter_secs = t_filter.elapsed().as_secs_f64();
    drop(buf);

    ensure_parent_dir(&args.out)?;
    // Dispatch on extension: `.jpg`/`.jpeg` → quality-controlled JPEG (matches the
    // present/enrich crop writer), anything else → the image-crate default (PNG).
    let lower = args.out.to_ascii_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        render::save_jpeg(&img, std::path::Path::new(&args.out), args.jpg_quality)?;
    } else {
        img.save(&args.out)
            .map_err(|e| format!("failed to write {}: {e}", args.out))?;
    }

    let total = iter_secs + filter_secs;
    eprintln!(
        "wrote {} in {total:.2}s (iterate {iter_secs:.2}s + filter {filter_secs:.3}s)",
        args.out
    );
    println!("=== render-one ===");
    println!("out:     {}", args.out);
    println!("iterate: {iter_secs:.3}s");
    println!("filter:  {filter_secs:.3}s");
    println!("total:   {total:.3}s");
    Ok(())
}


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
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

    /// Render a **Julia** set instead of Mandelbrot. Off ⇒ today's Mandelbrot
    /// behavior exactly (bit-for-bit). When on, the viewport (`--cx`/`--cy`/`--fw`)
    /// addresses the **z-plane** and `--c` supplies the fixed Julia parameter.
    /// Requires `--c`; using `--c` without `--julia` is an error.
    #[arg(long, default_value_t = false)]
    pub julia: bool,

    /// Julia parameter `c` as two arbitrary-precision decimal strings:
    /// `--c <re> <im>` (e.g. `--c -0.8 0.156`). Required iff `--julia`; ignored —
    /// and an error — without `--julia`.
    #[arg(
        long = "c",
        num_args = 2,
        value_names = ["RE", "IM"],
        allow_hyphen_values = true
    )]
    pub julia_c: Option<Vec<String>>,

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

    /// JPEG quality (1..=100) used only when `--out` ends in `.jpg`/`.jpeg`.
    /// Ignored for PNG output. q95 keeps cache renders clean enough that the
    /// train-time JPEG-q jitter (85..95) does not compound artifacts.
    #[arg(long, default_value_t = 95)]
    pub jpg_quality: u8,

    /// Dump the **raw smooth scalar field** (pre-coloring: before percentile-stretch,
    /// transform, gamma, shade, palette) to `<path>` as little-endian `f32`,
    /// row-major, at the supersampled resolution (`--width·ss × --height·ss`);
    /// interior / non-escaped subpixels are `NaN`. Also writes a JSON sidecar
    /// (`<path>` with `.bin`→`.json`, else `<path>.json`) describing dims / ss /
    /// location. Computes the field and **exits before coloring** — no PNG is
    /// written. The field is fixed to `smooth`; its bailout follows `--coloring`
    /// (`{"field":"smooth", ...}`) or, when absent, the beautiful-smooth default
    /// (`2^16`). This is the serialization half of the field⊗Python-coloring split.
    #[arg(long)]
    pub dump_field: Option<String>,

    /// Beautiful coloring params as a JSON object (inline, or `@path` to read from
    /// a file). Omitted — or any spec that resolves to the default (e.g. `{}`) —
    /// renders the **location profile** (current output, byte-identical). A
    /// non-default spec routes to the decoupled field→shade→palette pipeline
    /// (`render_modes`). Keys: `field` (smooth|stripe|tia|curvature|trap_circle|
    /// trap_cross|velocity|de|gaussian_int|exp_smoothing|decomposition|direct_trap),
    /// `bailout_b`, `skip`, `biomorph` (off|epsilon_cross),
    /// `stripe_density`, `trap_radius`, `color_by`
    /// (minimum_distance|average_distance|maximum_distance|iter_min|iter_max|
    /// angle_min|angle_max|mean_angle|ratio — gaussian_int only),
    /// `de_scale`, `direct_threshold`,
    /// `direct_opacity` / `merge_mode` (normal|multiply|screen|overlay) /
    /// `merge_order` (bottom_up|top_down) / `start_color` (black|white|#rrggbb)
    /// (direct_trap), `transform`
    /// (linear|sqrt|log|histeq|scurve),
    /// `gamma`, `shade` (none|normal_map), `light_azimuth`, `light_height`,
    /// `palette_cycles`, `palette_offset`. A spec naming any key seeds from
    /// `beautiful(field)` (so omitted keys follow the field preset, e.g.
    /// `{"field":"stripe"}` ≡ density-6 / linear / 2^16); an empty `{}` (or absent
    /// flag) renders the location profile.
    #[arg(long)]
    pub coloring: Option<String>,
}
