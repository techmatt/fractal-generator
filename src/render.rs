//! Frame geometry + the rayon render loop with supersampling.
//!
//! Pixels are square: the complex-plane frame *height* is derived from
//! `frame_width * (out_height / out_width)` so aspect never distorts. The
//! center, frame width, and output resolution fully determine the mapping.
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

/// Render parameters bundled for the driver.
pub struct RenderConfig {
    pub frame: Frame,
    pub maxiter: u32,
    pub bailout: f64,
    pub supersample: u32,
    pub color: ColorParams,
}

/// Render to a row-major buffer of 8-bit sRGB RGB triples.
pub fn render(
    backend: &dyn FractalBackend,
    palette: &Palette,
    cfg: &RenderConfig,
) -> Vec<u8> {
    let Frame {
        center,
        out_width: w,
        out_height: h,
        ..
    } = cfg.frame;
    let s = cfg.supersample.max(1);

    let fw = cfg.frame.frame_width;
    let fh = cfg.frame.frame_height();
    // Top-left corner of the frame in the complex plane. Image row 0 is the
    // top, which maps to the *largest* imaginary value.
    let left = center.re - 0.5 * fw;
    let top = center.im + 0.5 * fh;

    // Per-subsample step (square), and the half-step that centers the first
    // subsample within its output pixel.
    let step = fw / (w * s) as f64;
    let inv_samples = 1.0 / (s * s) as f64;

    // Parallelize over output rows. flat_map over an ordered parallel iterator
    // preserves row order, so the result is deterministic.
    (0..h)
        .into_par_iter()
        .flat_map_iter(|row| {
            let mut out = Vec::with_capacity(w as usize * 3);
            for col in 0..w {
                let mut acc = [0.0f64; 3];
                for sj in 0..s {
                    let py = (row * s + sj) as f64;
                    let im = top - (py + 0.5) * step;
                    for si in 0..s {
                        let px = (col * s + si) as f64;
                        let re = left + (px + 0.5) * step;
                        let c = Complex::new(re, im);
                        let dc = c - center;
                        let sample = backend.sample(c, dc, cfg.maxiter, cfg.bailout);
                        let lin = coloring::shade(&sample, palette, &cfg.color);
                        acc[0] += lin[0];
                        acc[1] += lin[1];
                        acc[2] += lin[2];
                    }
                }
                // Average in linear light, then encode sRGB.
                for k in 0..3 {
                    let v = linear_to_srgb(acc[k] * inv_samples);
                    out.push((v * 255.0 + 0.5) as u8);
                }
            }
            out
        })
        .collect()
}
