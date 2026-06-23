//! CLI wrapper over the render core.
//!
//! Default invocation renders one PNG (parse params, pick the precision backend
//! by depth with a `--backend` override, build the backend, iterate, shade,
//! save). The `sheet` subcommand iterates one location once and re-shades it
//! across many palettes into a single grid PNG.

use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use num_complex::Complex;

use fractal_generator::backend::{F64Backend, JuliaBackend, PerturbationBackend, Trap};
use fractal_generator::buffet;
use fractal_generator::cli::{BackendChoice, Cli, Command, LocationArgs, ShadeArgs, SheetArgs};
use fractal_generator::coherence;
use fractal_generator::corpus;
use fractal_generator::descend;
use fractal_generator::energy;
use fractal_generator::generate;
use fractal_generator::navigate;
use fractal_generator::palette_pick;
use fractal_generator::present;
use fractal_generator::profile;
use fractal_generator::reject_corridor;
use fractal_generator::search;
use fractal_generator::wallpaper;
use fractal_generator::coloring::{self, ChannelSet, ColorParams};
use fractal_generator::hp;
use fractal_generator::palette::{builtin, Palette};
use fractal_generator::palette_io::{load_palette, load_palette_file};
use fractal_generator::render::{self, Frame};
use fractal_generator::sheet;

/// Pixel spacing below which f64 enters its quantization regime (≈ eps·|c|).
/// At or below this, auto-selection switches to perturbation.
const PERTURB_SPACING: f64 = 1e-13;

/// Frame width below which f64 deltas approach denormals — the v1 cap
/// (~1e300 magnification). Refuse rather than render garbage.
const MIN_FRAME_WIDTH: f64 = 1e-300;

/// A location iterated once: frame, cached samples, and diagnostics.
struct Iterated {
    frame: Frame,
    buf: render::SampleBuffer,
    iter_secs: f64,
}

/// Build the frame + backend for a location and iterate the supersampled grid
/// once. Shared by render and sheet so both set up identically.
///
/// `channels` is the colorer's channel-intent (from
/// [`coloring::required_channels`]): the **f64** backend iterates only those
/// channels via [`render::iterate_samples_f64`]. The perturbation and Julia
/// paths ignore it and use the all-on trait `sample` (untouched).
fn iterate_location(loc: &LocationArgs, channels: ChannelSet) -> Result<Iterated, String> {
    let height = loc.resolved_height()?;
    if loc.width == 0 {
        return Err("--width must be > 0".into());
    }
    if loc.supersample == 0 {
        return Err("--supersample must be > 0".into());
    }
    if loc.frame_width <= 0.0 {
        return Err("--frame-width must be > 0".into());
    }
    if loc.frame_width < MIN_FRAME_WIDTH {
        return Err(format!(
            "frame width {:.3e} is past the v1 magnification cap (~1e300): f64 deltas \
             would underflow to denormals. Deeper zoom needs floatexp (deferred).",
            loc.frame_width
        ));
    }

    let prec_bits = hp::prec_bits(loc.width, loc.frame_width);
    let center_re_hp = hp::parse_decimal(&loc.center_re, prec_bits)?;
    let center_im_hp = hp::parse_decimal(&loc.center_im, prec_bits)?;
    let center = Complex::new(hp::to_f64(&center_re_hp), hp::to_f64(&center_im_hp));

    let frame = Frame {
        center,
        frame_width: loc.frame_width,
        out_width: loc.width,
        out_height: height,
    };

    let trap = Trap {
        shape: loc.trap,
        center: loc.resolved_trap_center()?,
        radius: loc.trap_radius,
    };

    // Julia path: always f64 at base scale, parameter `c` is the f64 projection
    // of the high-precision `--param-*`. Short-circuits the precision tiers — a
    // base-scale Julia never needs perturbation.
    if loc.julia {
        let param_re = hp::parse_decimal(&loc.param_re, prec_bits)?;
        let param_im = hp::parse_decimal(&loc.param_im, prec_bits)?;
        let param = Complex::new(hp::to_f64(&param_re), hp::to_f64(&param_im));
        let backend = JuliaBackend::new(param, loc.maxiter, loc.bailout, trap);
        eprintln!(
            "iterating Julia {}x{} (supersample {}), c = ({:.6}, {:.6}), maxiter {} ...",
            loc.width, height, loc.supersample, param.re, param.im, loc.maxiter,
        );
        let t0 = Instant::now();
        let buf = render::iterate_samples(&backend, &frame, loc.supersample);
        let iter_secs = t0.elapsed().as_secs_f64();
        return Ok(Iterated {
            frame,
            buf,
            iter_secs,
        });
    }

    let spacing = frame.pixel_size();
    let use_perturb = match loc.backend {
        BackendChoice::Auto => spacing <= PERTURB_SPACING,
        BackendChoice::Perturb => true,
        BackendChoice::F64 => false,
    };
    if !use_perturb && spacing < PERTURB_SPACING {
        eprintln!(
            "warning: pixel spacing {spacing:.3e} is inside f64's quantization regime and \
             --backend f64 was selected; expect coordinate stair-stepping. Use perturbation \
             (the auto default at this depth) for a clean render."
        );
    }

    // Build the perturbation reference up front (f64 has none). The f64 backend
    // is built inside the iterate branch below so it can take the channel-intent
    // dispatch; perturbation always uses the all-on trait path.
    let (pb, backend_name, ref_report): (Option<PerturbationBackend>, &str, Option<(f64, usize)>) =
        if use_perturb {
            let t = Instant::now();
            let pb = PerturbationBackend::new(
                &center_re_hp,
                &center_im_hp,
                loc.maxiter,
                loc.bailout,
                prec_bits,
                trap,
            );
            let ref_secs = t.elapsed().as_secs_f64();
            let len = pb.ref_len();
            (Some(pb), "perturbation", Some((ref_secs, len)))
        } else {
            (None, "f64", None)
        };

    eprintln!(
        "iterating {}x{} (supersample {}, {} subsamples/pixel), maxiter {}, backend {} \
         (spacing {:.3e}, prec {} bits) ...",
        loc.width,
        height,
        loc.supersample,
        loc.supersample * loc.supersample,
        loc.maxiter,
        backend_name,
        spacing,
        prec_bits,
    );
    if let Some((ref_secs, len)) = ref_report {
        eprintln!("reference orbit: {len} points in {ref_secs:.4}s");
    }

    let t0 = Instant::now();
    let buf = if let Some(pb) = &pb {
        render::iterate_samples(pb, &frame, loc.supersample)
    } else {
        // f64: compute only the channels the colorer reads (atom always off).
        let backend = F64Backend::new(loc.maxiter, loc.bailout, trap);
        render::iterate_samples_f64(&backend, &frame, loc.supersample, channels)
    };
    let iter_secs = t0.elapsed().as_secs_f64();

    if buf.glitched_pixels > 0 {
        eprintln!(
            "warning: {} pixel(s) glitched (f64 delta underflowed — too deep for this tier). \
             Re-run with --mark-glitches to locate them.",
            buf.glitched_pixels
        );
    }

    Ok(Iterated {
        frame,
        buf,
        iter_secs,
    })
}

/// Map shading CLI args to coloring parameters.
fn color_params(shade: &ShadeArgs) -> ColorParams {
    ColorParams {
        density: shade.density,
        offset: shade.offset,
        channel: shade.color,
        interior: shade.interior,
        trap_scale: shade.trap_scale,
        trap_curve: shade.trap_curve,
        trap_phase_strength: shade.trap_phase_strength,
        de_shade: shade.de_shade,
        mark_glitches: shade.mark_glitches,
    }
}

fn run_render(cli: &Cli) -> Result<(), String> {
    let params = color_params(&cli.shade);
    let it = iterate_location(&cli.location, coloring::required_channels(&params))?;
    let palette = load_palette(
        &cli.palette.palette,
        cli.palette.palette_entry.as_deref(),
        cli.palette.palette_reverse,
    )?;

    let t1 = Instant::now();
    let img = render::shade_and_downsample(
        &it.buf.samples,
        it.frame.out_width,
        it.frame.out_height,
        it.buf.ss,
        &palette,
        &params,
        it.frame.pixel_size(),
    );
    let shade_secs = t1.elapsed().as_secs_f64();

    fractal_generator::ensure_parent_dir(&cli.output)?;
    img.save(&cli.output)
        .map_err(|e| format!("failed to write {}: {e}", cli.output))?;
    eprintln!(
        "wrote {} (palette '{}') in {:.2}s (iterate {:.2}s + shade {:.3}s)",
        cli.output,
        palette.name(),
        it.iter_secs + shade_secs,
        it.iter_secs,
        shade_secs
    );
    Ok(())
}

/// Resolve the sheet's palette set: every `--builtins` name, then every block of
/// every `--palettes` file (a multi-block `.ugr` contributes one tile per block).
fn resolve_sheet_palettes(args: &SheetArgs) -> Result<Vec<Palette>, String> {
    let mut palettes = Vec::new();
    for name in &args.builtins {
        let p = builtin(name, args.palette_reverse)
            .ok_or_else(|| format!("unknown built-in palette '{name}'"))?;
        palettes.push(p);
    }
    for spec in &args.palettes {
        let path = std::path::Path::new(spec);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "ugr" {
            // Expand every block into its own tile.
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read palette '{}': {e}", path.display()))?;
            let grads = fractal_generator::palette_io::parse_ugr(&text)
                .map_err(|e| format!("parsing '{}': {e}", path.display()))?;
            if grads.is_empty() {
                return Err(format!("no gradient blocks in '{}'", path.display()));
            }
            for g in grads {
                palettes.push(Palette::from_srgb8_stops(
                    g.name,
                    &g.stops,
                    args.palette_reverse,
                ));
            }
        } else {
            palettes.push(load_palette_file(path, None, args.palette_reverse)?);
        }
    }
    if palettes.is_empty() {
        return Err("contact sheet needs at least one palette (--palettes / --builtins)".into());
    }
    Ok(palettes)
}

fn run_sheet(args: &SheetArgs) -> Result<(), String> {
    let palettes = resolve_sheet_palettes(args)?;
    eprintln!("contact sheet: {} palettes", palettes.len());

    // The location iterates at the tile resolution: override width with
    // --tile-width (height still follows --aspect).
    let loc = LocationArgs {
        center_re: args.location.center_re.clone(),
        center_im: args.location.center_im.clone(),
        frame_width: args.location.frame_width,
        maxiter: args.location.maxiter,
        width: args.tile_width.max(1),
        height: None,
        aspect: args.location.aspect.clone(),
        supersample: args.location.supersample,
        bailout: args.location.bailout,
        trap: args.location.trap,
        trap_center: args.location.trap_center.clone(),
        trap_radius: args.location.trap_radius,
        backend: args.location.backend,
        julia: false,
        param_re: "0".into(),
        param_im: "0".into(),
    };

    let params = color_params(&args.shade);
    let it = iterate_location(&loc, coloring::required_channels(&params))?;

    let t1 = Instant::now();
    let (grid, legend) =
        sheet::render_contact_sheet(&it.buf, &palettes, &params, it.frame.pixel_size(), args.cols);
    let shade_secs = t1.elapsed().as_secs_f64();

    fractal_generator::ensure_parent_dir(&args.output)?;
    grid.save(&args.output)
        .map_err(|e| format!("failed to write {}: {e}", args.output))?;

    for line in &legend {
        println!("{line}");
    }
    eprintln!(
        "wrote {} ({} tiles) in iterate {:.2}s + {} shadings {:.3}s",
        args.output,
        palettes.len(),
        it.iter_secs,
        palettes.len(),
        shade_secs
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match &cli.command {
        Some(Command::Sheet(args)) => run_sheet(args),
        Some(Command::Generate(args)) => generate::run_generate(args),
        Some(Command::Present(args)) => present::run_present(args),
        Some(Command::Descend(args)) => descend::run_descend(args),
        Some(Command::Navigate(args)) => navigate::run_navigate(args),
        Some(Command::Search(args)) => search::run_search(args),
        Some(Command::Corpus(args)) => corpus::run_corpus(args),
        Some(Command::Wallpaper(args)) => wallpaper::run_wallpaper(args),
        Some(Command::Cohere(args)) => coherence::run_cohere(args),
        Some(Command::Cover(args)) => coherence::run_cover(args),
        Some(Command::Buffet(args)) => buffet::run_buffet(args),
        Some(Command::Calibrate(args)) => energy::run_calibrate(args),
        Some(Command::Rescore(args)) => energy::run_rescore(args),
        Some(Command::Overbusy(args)) => energy::run_overbusy(args),
        Some(Command::Archetype(args)) => energy::run_archetype(args),
        Some(Command::Anchor(args)) => energy::run_anchor(args),
        Some(Command::Dedup(args)) => energy::run_dedup(args),
        Some(Command::Muster(args)) => energy::run_muster(args),
        Some(Command::Profile(args)) => profile::run_profile(args),
        Some(Command::RejectCorridor(args)) => reject_corridor::run_reject_corridor(args),
        Some(Command::PalettePick(args)) => palette_pick::run_palette_pick(args),
        None => run_render(&cli),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
