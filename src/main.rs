//! CLI wrapper over the render core.
//!
//! Parses parameters, picks the precision backend by depth (with a `--backend`
//! override), refuses frames past the v1 magnification cap, builds the backend
//! (computing the perturbation reference orbit when needed), renders, and saves.

use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use num_complex::Complex;

use fractal_generator::backend::{F64Backend, FractalBackend, PerturbationBackend};
use fractal_generator::cli::{BackendChoice, Cli};
use fractal_generator::coloring::ColorParams;
use fractal_generator::hp;
use fractal_generator::palette::Palette;
use fractal_generator::render::{self, Frame, RenderConfig};

/// Pixel spacing below which f64 enters its quantization regime (≈ eps·|c|).
/// At or below this, auto-selection switches to perturbation.
const PERTURB_SPACING: f64 = 1e-13;

/// Frame width below which f64 deltas approach denormals — the v1 cap
/// (~1e300 magnification). Refuse rather than render garbage.
const MIN_FRAME_WIDTH: f64 = 1e-300;

fn run() -> Result<(), String> {
    let args = Cli::parse();

    let height = args.resolved_height()?;
    if args.width == 0 {
        return Err("--width must be > 0".into());
    }
    if args.supersample == 0 {
        return Err("--supersample must be > 0".into());
    }

    // Too-deep refusal (frame-level): below this, f64 deltas underflow.
    if args.frame_width <= 0.0 {
        return Err("--frame-width must be > 0".into());
    }
    if args.frame_width < MIN_FRAME_WIDTH {
        return Err(format!(
            "frame width {:.3e} is past the v1 magnification cap (~1e300): f64 deltas \
             would underflow to denormals. Deeper zoom needs floatexp (deferred).",
            args.frame_width
        ));
    }

    // High-precision center: parse to bignum, keep an f64 projection for geometry
    // and the f64 backend.
    let prec_bits = hp::prec_bits(args.width, args.frame_width);
    let center_re_hp = hp::parse_decimal(&args.center_re, prec_bits)?;
    let center_im_hp = hp::parse_decimal(&args.center_im, prec_bits)?;
    let center = Complex::new(hp::to_f64(&center_re_hp), hp::to_f64(&center_im_hp));

    let frame = Frame {
        center,
        frame_width: args.frame_width,
        out_width: args.width,
        out_height: height,
    };

    // Backend selection by pixel spacing, with --backend override.
    let spacing = frame.pixel_size();
    let use_perturb = match args.backend {
        BackendChoice::Auto => spacing <= PERTURB_SPACING,
        BackendChoice::Perturb => true,
        BackendChoice::F64 => false,
    };

    // Quantization warning ONLY when f64 is (force-)selected past its clean
    // limit; auto-selected perturbation is silent.
    if !use_perturb && spacing < PERTURB_SPACING {
        eprintln!(
            "warning: pixel spacing {spacing:.3e} is inside f64's quantization regime and \
             --backend f64 was selected; expect coordinate stair-stepping. Use perturbation \
             (the auto default at this depth) for a clean render."
        );
    }

    let cfg = RenderConfig {
        frame,
        supersample: args.supersample,
        color: ColorParams {
            density: args.density,
            offset: args.offset,
        },
        mark_glitches: args.mark_glitches,
    };

    // Build the backend; time the reference-orbit construction for perturbation.
    let (backend, backend_name, ref_report): (Box<dyn FractalBackend>, &str, Option<(f64, usize)>) =
        if use_perturb {
            let t = Instant::now();
            let pb = PerturbationBackend::new(
                &center_re_hp,
                &center_im_hp,
                args.maxiter,
                args.bailout,
                prec_bits,
            );
            let ref_secs = t.elapsed().as_secs_f64();
            let len = pb.ref_len();
            (Box::new(pb), "perturbation", Some((ref_secs, len)))
        } else {
            (Box::new(F64Backend::new(args.maxiter, args.bailout)), "f64", None)
        };

    let palette = Palette::ultra_fractal();

    eprintln!(
        "rendering {}x{} (supersample {}, {} subsamples/pixel), maxiter {}, backend {} \
         (spacing {:.3e}, prec {} bits) ...",
        args.width,
        height,
        args.supersample,
        args.supersample * args.supersample,
        args.maxiter,
        backend_name,
        spacing,
        prec_bits,
    );
    if let Some((ref_secs, len)) = ref_report {
        eprintln!("reference orbit: {len} points in {ref_secs:.4}s");
    }

    let t0 = Instant::now();
    let out = render::render(&*backend, &palette, &cfg);
    let render_secs = t0.elapsed().as_secs_f64();

    if let Some((ref_secs, _)) = ref_report {
        let pct = if render_secs > 0.0 {
            100.0 * ref_secs / render_secs
        } else {
            0.0
        };
        eprintln!("reference orbit was {pct:.3}% of render time");
    }

    if out.glitched_pixels > 0 {
        eprintln!(
            "warning: {} pixel(s) glitched (f64 delta underflowed — too deep for this tier). \
             Re-run with --mark-glitches to locate them.",
            out.glitched_pixels
        );
    }

    let img = image::RgbImage::from_raw(args.width, height, out.pixels)
        .ok_or_else(|| "render buffer size mismatch".to_string())?;
    img.save(&args.output)
        .map_err(|e| format!("failed to write {}: {e}", args.output))?;

    eprintln!("wrote {} in {:.2}s", args.output, render_secs);
    Ok(())
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
