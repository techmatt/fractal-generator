//! CLI parsing (clap derive) and resolution of aspect → output height.

use clap::Parser;

/// Escape-time Mandelbrot renderer (f64 reference skeleton).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Frame center, real part.
    #[arg(long, default_value_t = -0.5, allow_negative_numbers = true)]
    pub center_re: f64,

    /// Frame center, imaginary part.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub center_im: f64,

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

    /// Output PNG path.
    #[arg(long, default_value = "out.png")]
    pub output: String,
}

impl Cli {
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
