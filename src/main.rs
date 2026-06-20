//! f64 Mandelbrot render skeleton.
//!
//! Two architectural seams this skeleton exists to establish:
//!  1. Precision behind [`backend::FractalBackend`] (perturbation slots in,
//!     Prompt 2) — note `dc` is already in the signature.
//!  2. A separable [`coloring`] stage (sample → RGB) so re-coloring never
//!     re-iterates (palette system, Prompt 4).

mod backend;
mod cli;
mod coloring;
mod palette;
mod render;

use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use num_complex::Complex;

use backend::F64Backend;
use cli::Cli;
use coloring::ColorParams;
use palette::Palette;
use render::{Frame, RenderConfig};

fn main() -> ExitCode {
    let args = Cli::parse();

    let height = match args.resolved_height() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if args.width == 0 {
        eprintln!("error: --width must be > 0");
        return ExitCode::FAILURE;
    }
    if args.supersample == 0 {
        eprintln!("error: --supersample must be > 0");
        return ExitCode::FAILURE;
    }

    let frame = Frame {
        center: Complex::new(args.center_re, args.center_im),
        frame_width: args.frame_width,
        out_width: args.width,
        out_height: height,
    };

    // Cheap foreshadowing guard (Prompt 2 lands the real too-deep refusal).
    // Compare per-pixel spacing to f64's relative epsilon regime.
    let pixel_spacing = frame.pixel_size();
    if pixel_spacing < 1e-13 {
        eprintln!(
            "warning: pixel spacing {pixel_spacing:.3e} (frame_width/width) is entering f64's \
             quantization regime; expect coordinate stair-stepping. Perturbation (Prompt 2) is \
             needed for clean renders this deep."
        );
    }

    let cfg = RenderConfig {
        frame,
        maxiter: args.maxiter,
        bailout: args.bailout,
        supersample: args.supersample,
        color: ColorParams {
            density: args.density,
            offset: args.offset,
        },
    };

    let backend = F64Backend; // constructed per-frame (holds a ref orbit later)
    let palette = Palette::ultra_fractal();

    eprintln!(
        "rendering {}x{} (supersample {}, {} subsamples/pixel), maxiter {} ...",
        args.width,
        height,
        args.supersample,
        args.supersample * args.supersample,
        args.maxiter
    );
    let t0 = Instant::now();
    let buf = render::render(&backend, &palette, &cfg);
    let dt = t0.elapsed();

    let img = match image::RgbImage::from_raw(args.width, height, buf) {
        Some(img) => img,
        None => {
            eprintln!("error: render buffer size mismatch");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = img.save(&args.output) {
        eprintln!("error: failed to write {}: {e}", args.output);
        return ExitCode::FAILURE;
    }

    eprintln!("wrote {} in {:.2}s", args.output, dt.as_secs_f64());
    ExitCode::SUCCESS
}
