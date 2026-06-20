//! Frame geometry + the rayon render loop with supersampling.
//!
//! Pixels are square: the complex-plane frame *height* is derived from
//! `frame_width * (out_height / out_width)` so aspect never distorts. The
//! center, frame width, and output resolution fully determine the mapping.
//!
//! **`dc` is computed straight from pixel geometry, never as `c - center`.** At
//! deep zoom `c_f64 - center_f64` is catastrophic cancellation (garbage); the
//! pixel offset `dc` is O(frame_width) and stays accurate in f64 down to
//! ~1e-305. The absolute coordinate `c = center + dc` is formed only for the
//! f64 backend, which is used solely at shallow depth where it is accurate.
//!
//! Antialiasing is mandatory: each output pixel is the box average of an
//! `S×S` grid of subsamples, **averaged in linear light** and only then
//! sRGB-encoded — this avoids the dark fringing of averaging gamma-encoded
//! values. Per-pixel independence makes the rayon traversal order irrelevant,
//! so output is deterministic for fixed parameters.

use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::FractalBackend;
use crate::coloring::{self, ColorParams};
use crate::palette::{linear_to_srgb, Palette};

/// Linear-light sentinel for `--mark-glitches` (sRGB magenta).
const GLITCH_LINEAR: [f64; 3] = [1.0, 0.0, 1.0];

/// Everything needed to place pixels in the complex plane.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub center: Complex<f64>,
    pub frame_width: f64,
    pub out_width: u32,
    pub out_height: u32,
}

impl Frame {
    /// Complex-plane frame height, derived to keep pixels square.
    pub fn frame_height(&self) -> f64 {
        self.frame_width * (self.out_height as f64 / self.out_width as f64)
    }

    /// Complex-plane size of one *output* pixel (square).
    pub fn pixel_size(&self) -> f64 {
        self.frame_width / self.out_width as f64
    }
}

/// Render parameters bundled for the driver. `maxiter`/`bailout` now live in the
/// backend (constructed per frame), so they are absent here.
pub struct RenderConfig {
    pub frame: Frame,
    pub supersample: u32,
    pub color: ColorParams,
    /// Paint per-pixel glitched (delta-underflow) subsamples magenta.
    pub mark_glitches: bool,
}

/// Render result: the row-major 8-bit sRGB buffer plus a diagnostic count of
/// output pixels that had at least one glitched (unreliable) subsample.
pub struct RenderOutput {
    pub pixels: Vec<u8>,
    pub glitched_pixels: u64,
}

/// Render to a row-major buffer of 8-bit sRGB RGB triples.
pub fn render(backend: &dyn FractalBackend, palette: &Palette, cfg: &RenderConfig) -> RenderOutput {
    let Frame {
        center,
        out_width: w,
        out_height: h,
        ..
    } = cfg.frame;
    let s = cfg.supersample.max(1);

    let fw = cfg.frame.frame_width;
    let fh = cfg.frame.frame_height();
    let inv_samples = 1.0 / (s * s) as f64;

    // Subsample grid dimensions and reciprocals (fractional pixel position).
    let sub_w = (w * s) as f64;
    let sub_h = (h * s) as f64;

    // Parallelize over output rows. Each row yields its byte slice plus a glitch
    // count; collecting an ordered Vec keeps the output deterministic.
    let rows: Vec<(Vec<u8>, u64)> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut out = Vec::with_capacity(w as usize * 3);
            let mut glitched_in_row: u64 = 0;
            for col in 0..w {
                let mut acc = [0.0f64; 3];
                let mut pixel_glitched = false;
                for sj in 0..s {
                    // Fractional vertical position in [0,1); row 0 (top) is the
                    // largest imaginary value, hence (0.5 - py_frac).
                    let py = (row * s + sj) as f64 + 0.5;
                    let py_frac = py / sub_h;
                    let dc_im = (0.5 - py_frac) * fh;
                    for si in 0..s {
                        let px = (col * s + si) as f64 + 0.5;
                        let px_frac = px / sub_w;
                        let dc_re = (px_frac - 0.5) * fw;

                        let dc = Complex::new(dc_re, dc_im);
                        let c = center + dc; // only the f64 backend reads this
                        let sample = backend.sample(c, dc);
                        if sample.glitched {
                            pixel_glitched = true;
                        }
                        let lin = if cfg.mark_glitches && sample.glitched {
                            GLITCH_LINEAR
                        } else {
                            coloring::shade(&sample, palette, &cfg.color)
                        };
                        acc[0] += lin[0];
                        acc[1] += lin[1];
                        acc[2] += lin[2];
                    }
                }
                if pixel_glitched {
                    glitched_in_row += 1;
                }
                // Average in linear light, then encode sRGB.
                for k in 0..3 {
                    let v = linear_to_srgb(acc[k] * inv_samples);
                    out.push((v * 255.0 + 0.5) as u8);
                }
            }
            (out, glitched_in_row)
        })
        .collect();

    let mut pixels = Vec::with_capacity(w as usize * h as usize * 3);
    let mut glitched_pixels = 0u64;
    for (row_bytes, g) in rows {
        pixels.extend_from_slice(&row_bytes);
        glitched_pixels += g;
    }
    RenderOutput {
        pixels,
        glitched_pixels,
    }
}
