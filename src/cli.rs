//! CLI parsing (clap derive) and resolution of aspect → output height.

use clap::{Parser, ValueEnum};
use num_complex::Complex;

use crate::backend::TrapShape;
use crate::coloring::{ColorChannel, InteriorMode};

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

/// Escape-time Mandelbrot renderer (f64 + perturbation backends).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Frame center, real part — arbitrary-precision decimal string (an f64
    /// center is meaningless at depth, so this is parsed at full precision).
    #[arg(long, default_value = "-0.5", allow_negative_numbers = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal string.
    #[arg(long, default_value = "0.0", allow_negative_numbers = true)]
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

    /// Gradient cycles per unit smooth-iteration count.
    #[arg(long, default_value_t = 0.025)]
    pub density: f64,

    /// Gradient phase offset in [0,1).
    #[arg(long, default_value_t = 0.0)]
    pub offset: f64,

    /// Primary exterior coloring channel.
    #[arg(long, value_enum, default_value_t = ColorChannel::Smooth)]
    pub color: ColorChannel,

    /// Interior (non-escaping) pixel treatment.
    #[arg(long, value_enum, default_value_t = InteriorMode::Black)]
    pub interior: InteriorMode,

    /// Orbit-trap shape.
    #[arg(long, value_enum, default_value_t = TrapShape::Point)]
    pub trap: TrapShape,

    /// Orbit-trap center as `re,im`.
    #[arg(long, default_value = "0,0")]
    pub trap_center: String,

    /// Orbit-trap radius (circle trap only).
    #[arg(long, default_value_t = 1.0)]
    pub trap_radius: f64,

    /// Multiplier applied to the trap minimum before mapping (trap channel).
    #[arg(long, default_value_t = 1.0)]
    pub trap_scale: f64,

    /// Weight of trap phase added as a secondary hue offset (0 = unused).
    #[arg(long, default_value_t = 0.0)]
    pub trap_phase_strength: f64,

    /// DE-shade: brighten thin boundary filaments. Bare flag uses strength 1.0;
    /// pass a value to tune. Omit to disable.
    #[arg(long, num_args = 0..=1, default_missing_value = "1.0")]
    pub de_shade: Option<f64>,

    /// Precision backend: f64, perturb, or auto (default).
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    pub backend: BackendChoice,

    /// Paint per-pixel glitched (delta-underflow) pixels magenta for diagnosis.
    #[arg(long, default_value_t = false)]
    pub mark_glitches: bool,

    /// Output PNG path.
    #[arg(long, default_value = "out.png")]
    pub output: String,
}

impl Cli {
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
