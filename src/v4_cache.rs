//! `v4-render-batch` — bulk render executor for the v4 augmentation cache.
//!
//! The v4 cache is **152k+ renders** (3622 locations × 42 augmentation slots).
//! Driving that through one `render-one` process per render would pay a process
//! spawn + 224-palette colormap parse on every single render — hours of pure
//! overhead. This subcommand instead reads a **plan** (one JSONL row per render,
//! produced by `tools/v4/build_plan.py`), loads the colormap library **once**,
//! builds each distinct [`Palette`] **once**, and renders every row in-process,
//! parallel across rows.
//!
//! Each render is byte-identical to the equivalent `render-one` invocation: same
//! shallow-f64 path, [`color_params`], selective-mirror palette load, grid
//! supersample (`ss` per row), and [`render::shade_and_downsample_filtered`]. The
//! plan, not this code, owns the augmentation scheme (palette × scale × shift ×
//! AA) — so class-balance is a property of the *plan* and is verified Python-side.
//!
//! **Resumable.** A row whose output already exists is skipped, so a killed run
//! resumes by re-invoking with the same plan. **Additive / non-default** — adds no
//! behavior to any existing render path.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use clap::Args;
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{F64Backend, JuliaBackend, Trap, TrapShape};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::render::{self, DownsampleFilter, Frame, SubsamplePattern};
use crate::{coloring, jsonl};

/// Escape radius — identical to render-one/present/enrich.
const BAILOUT: f64 = 1e6;

/// One render row parsed from the plan.
struct Spec {
    cx: String,
    cy: String,
    fw: f64,
    palette: String,
    ss: u32,
    filter: DownsampleFilter,
    out: String,
    /// Julia parameter `c` (decimal strings) when `fractal_type == "julia"`;
    /// `None` ⇒ Mandelbrot (today's behavior, byte-identical). Mirrors the
    /// `render-one --julia --c` coupling: the viewport addresses the z-plane.
    julia_c: Option<(String, String)>,
}

fn parse_filter(s: &str) -> Result<DownsampleFilter, String> {
    match s {
        "box" => Ok(DownsampleFilter::Box),
        "mitchell" => Ok(DownsampleFilter::Mitchell),
        "lanczos3" => Ok(DownsampleFilter::Lanczos3),
        other => Err(format!("unknown filter '{other}' (box|mitchell|lanczos3)")),
    }
}

fn parse_spec(line: &str) -> Result<Spec, String> {
    let cx = jsonl::field_str(line, "cx").ok_or("missing cx")?;
    let cy = jsonl::field_str(line, "cy").ok_or("missing cy")?;
    let fw = jsonl::field_f64(line, "fw").ok_or("missing fw")?;
    let palette = jsonl::field_str(line, "palette").ok_or("missing palette")?;
    let ss = jsonl::field_usize(line, "ss").ok_or("missing ss")? as u32;
    let filter = parse_filter(&jsonl::field_str(line, "filter").ok_or("missing filter")?)?;
    let out = jsonl::field_str(line, "out").ok_or("missing out")?;
    // Optional Julia coupling: `fractal_type:"julia"` requires `c_re`/`c_im`
    // (decimal strings). Absent or `"mandelbrot"` ⇒ Mandelbrot. A `julia` row
    // missing `c` is a loud error, not a silent Mandelbrot fallback (would
    // poison the cache with mis-rendered tiles).
    let julia_c = match jsonl::field_str(line, "fractal_type").as_deref() {
        Some("julia") => {
            let c_re = jsonl::field_str(line, "c_re").ok_or("julia row missing c_re")?;
            let c_im = jsonl::field_str(line, "c_im").ok_or("julia row missing c_im")?;
            Some((c_re, c_im))
        }
        None | Some("mandelbrot") => None,
        Some(other) => return Err(format!("unknown fractal_type '{other}' (mandelbrot|julia)")),
    };
    Ok(Spec { cx, cy, fw, palette, ss, filter, out, julia_c })
}

pub fn run_v4_render_batch(args: &V4RenderBatchArgs) -> Result<(), String> {
    // --- load + parse the plan ---
    let plan_text = std::fs::read_to_string(&args.plan)
        .map_err(|e| format!("read plan {}: {e}", args.plan))?;
    let mut specs: Vec<Spec> = Vec::new();
    for (i, line) in plan_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        specs.push(parse_spec(line).map_err(|e| format!("plan line {}: {e}", i + 1))?);
    }
    let total = specs.len();
    if total == 0 {
        return Err("plan is empty".into());
    }

    // --- load colormap library once, build every distinct palette once ---
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library =
        parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;

    let mut distinct: Vec<String> = specs.iter().map(|s| s.palette.clone()).collect();
    distinct.sort();
    distinct.dedup();
    let mut palettes: std::collections::HashMap<String, Palette> = std::collections::HashMap::new();
    for name in &distinct {
        let cm = library
            .iter()
            .find(|c| &c.name == name)
            .ok_or_else(|| format!("palette '{name}' not in {}", args.colormaps))?;
        palettes.insert(
            name.clone(),
            Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed),
        );
    }

    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    eprintln!(
        "v4-render-batch: {total} renders, {} palettes, maxiter {}, q{}",
        distinct.len(),
        args.maxiter,
        args.jpg_quality
    );

    let done = AtomicU64::new(0);
    let skipped = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let start = Instant::now();
    let last_log = Mutex::new(0u64);
    let log_every = args.log_every.max(1) as u64;

    specs.par_iter().for_each(|spec| {
        // Resume: a completed render is left untouched.
        if Path::new(&spec.out).exists() {
            skipped.fetch_add(1, Ordering::Relaxed);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            maybe_log(n, total, &start, &last_log, log_every, &skipped, &failed);
            return;
        }
        match render_one_spec(
            spec,
            &palettes,
            &params,
            channels,
            trap,
            args.width,
            args.height,
            args.maxiter,
            args.jpg_quality,
        ) {
            Ok(()) => {}
            Err(e) => {
                failed.fetch_add(1, Ordering::Relaxed);
                eprintln!("FAIL {}: {e}", spec.out);
            }
        }
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        maybe_log(n, total, &start, &last_log, log_every, &skipped, &failed);
    });

    let elapsed = start.elapsed().as_secs_f64();
    let nf = failed.load(Ordering::Relaxed);
    let ns = skipped.load(Ordering::Relaxed);
    eprintln!(
        "v4-render-batch DONE: {total} rows, {} rendered, {ns} skipped, {nf} failed in {:.1}s",
        total as u64 - ns - nf,
        elapsed
    );
    println!("=== v4-render-batch ===");
    println!("plan:      {}", args.plan);
    println!("total:     {total}");
    println!("rendered:  {}", total as u64 - ns - nf);
    println!("skipped:   {ns}");
    println!("failed:    {nf}");
    println!("wall:      {elapsed:.1}s");
    if nf > 0 {
        return Err(format!("{nf} render(s) failed"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_one_spec(
    spec: &Spec,
    palettes: &std::collections::HashMap<String, Palette>,
    params: &coloring::ColorParams,
    channels: coloring::ChannelSet,
    trap: Trap,
    width: u32,
    height: u32,
    maxiter: u32,
    jpg_quality: u8,
) -> Result<(), String> {
    use crate::hp;
    let palette = palettes.get(&spec.palette).ok_or("palette not built")?;

    let prec_bits = hp::prec_bits(width, spec.fw);
    let cx = hp::to_f64(&hp::parse_decimal(&spec.cx, prec_bits)?);
    let cy = hp::to_f64(&hp::parse_decimal(&spec.cy, prec_bits)?);
    let frame = Frame {
        center: Complex::new(cx, cy),
        frame_width: spec.fw,
        out_width: width,
        out_height: height,
    };
    let pixel_spacing = frame.pixel_size();
    if pixel_spacing <= 1e-13 {
        return Err(format!("pixel spacing {pixel_spacing:.3e} inside f64 quantization (deep zoom)"));
    }

    let ss = spec.ss.max(1);
    // Mandelbrot: channel-intent f64 path (byte-identical to render-one). Julia:
    // JuliaBackend over the same grid/seed — `channels` is irrelevant to Julia
    // (it always computes the gated trap, never a `dz`), exactly as render-one.
    let buf = match &spec.julia_c {
        None => {
            let backend = F64Backend::new(maxiter, BAILOUT, trap);
            render::iterate_samples_f64_pattern(
                &backend,
                &frame,
                ss,
                channels,
                SubsamplePattern::Grid,
                0,
            )
        }
        Some((c_re, c_im)) => {
            let pr = hp::to_f64(&hp::parse_decimal(c_re, prec_bits)?);
            let pi = hp::to_f64(&hp::parse_decimal(c_im, prec_bits)?);
            let backend =
                JuliaBackend::new(Complex::new(pr, pi), maxiter, BAILOUT, trap);
            render::iterate_samples_julia_pattern(&backend, &frame, ss, SubsamplePattern::Grid, 0)
        }
    };
    let img = render::shade_and_downsample_filtered(
        &buf.samples,
        frame.out_width,
        frame.out_height,
        ss,
        palette,
        params,
        pixel_spacing,
        spec.filter,
    );
    crate::ensure_parent_dir(&spec.out)?;
    render::save_jpeg(&img, Path::new(&spec.out), jpg_quality)
}

#[allow(clippy::too_many_arguments)]
fn maybe_log(
    n: u64,
    total: usize,
    start: &Instant,
    last_log: &Mutex<u64>,
    log_every: u64,
    skipped: &AtomicU64,
    failed: &AtomicU64,
) {
    if n % log_every != 0 && n != total as u64 {
        return;
    }
    let mut last = last_log.lock().unwrap();
    if n <= *last {
        return; // another thread already logged a later count
    }
    *last = n;
    let el = start.elapsed().as_secs_f64();
    let rate = n as f64 / el.max(1e-9);
    let eta = (total as f64 - n as f64) / rate.max(1e-9);
    eprintln!(
        "  [{n}/{total}] {:.0}/s  elapsed {:.0}s  ETA {:.0}s  (skipped {}, failed {})",
        rate,
        el,
        eta,
        skipped.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
    );
}

/// `v4-render-batch` subcommand: render every row of a plan JSONL to JPEG, reusing
/// one colormap-library load. Resumable (skips existing outputs). See module docs.
#[derive(Args, Debug)]
pub struct V4RenderBatchArgs {
    /// Plan JSONL: one render per line `{cx,cy,fw,palette,ss,filter,out}`.
    #[arg(long, default_value = "data/v4/plan.jsonl")]
    pub plan: String,

    /// Colormap library (looked up by `palette` name, selective-mirror loaded).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Output width in pixels (the locked reduced cache resolution).
    #[arg(long, default_value_t = 512)]
    pub width: u32,

    /// Output height in pixels.
    #[arg(long, default_value_t = 288)]
    pub height: u32,

    /// Maximum iterations / orbit cap — matches the render-one wallpaper lock.
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// JPEG quality (1..=100). The cache is locked at **q85** — it sits at the
    /// floor of the train-time JPEG-q jitter range (85..95), so cache reads never
    /// present the net cleaner JPEGs than deploy and on-disk size stays modest.
    #[arg(long, default_value_t = 85)]
    pub jpg_quality: u8,

    /// Progress log cadence (every N completed rows).
    #[arg(long, default_value_t = 2000)]
    pub log_every: u32,
}
