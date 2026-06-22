//! Frame geometry + the two separable render stages.
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
//! Rendering is split into two stages so re-coloring never re-iterates:
//!  1. [`iterate_samples`] runs the backend over the **supersampled** grid and
//!     caches a `Vec<PixelSample>` (SS resolution).
//!  2. [`shade_and_downsample`] is a **pure** function over that cached buffer:
//!     it shades each subpixel, averages in linear light, and sRGB-encodes.
//!
//! Antialiasing is mandatory and stays correct under re-color: we shade each
//! subpixel and average the *colors* in linear light — never the channel values
//! pre-shade, which would break AA when the buffer is re-colored.
//!
//! Memory note: the SS buffer is ~48 B × out_w × out_h × ss² (≈470 MB at
//! 1920×1280 ss2). Fine for single renders; keep large supersampled frames to
//! modest resolution.

use image::RgbImage;
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{F64Backend, FractalBackend, PixelSample, PHASE_GATED};
use crate::coloring::{self, ChannelSet, ColorParams};
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

/// Cached supersampled iteration result. Holding this lets the coloring stage
/// run any number of times without re-iterating.
pub struct SampleBuffer {
    /// Row-major `PixelSample`s at SS resolution (`sub_width × sub_height`).
    pub samples: Vec<PixelSample>,
    pub out_width: u32,
    pub out_height: u32,
    /// Supersampling factor `S` (`sub_width = out_width · S`).
    pub ss: u32,
    /// Output pixels with at least one glitched (unreliable) subsample.
    pub glitched_pixels: u64,
}

impl SampleBuffer {
    /// Subsample grid width (`out_width · ss`).
    pub fn sub_width(&self) -> u32 {
        self.out_width * self.ss
    }
}

/// Stage 1 — iterate the backend over the supersampled grid and cache samples.
/// This is the only stage that touches the backend; everything downstream is
/// pure over the returned buffer.
///
/// Uses the trait `sample` (every channel live), so it is the path for the
/// perturbation and Julia backends, `navigate`, `probe`, and `profile`. The
/// render/sheet driver routes the **f64** backend through
/// [`iterate_samples_f64`] instead, which computes only the channels the
/// colorer reads.
pub fn iterate_samples(backend: &dyn FractalBackend, frame: &Frame, ss: u32) -> SampleBuffer {
    iterate_grid(frame, ss, |c, dc| backend.sample(c, dc))
}

/// Stage 1, **channel-intent dispatch** for the f64 backend (render/sheet only).
///
/// Monomorphizes the f64 kernel to exactly the channels `channels` says the
/// colorer reads — `trap` (orbit-trap distance/phase + its `eval_dist`) and `de`
/// (the distance estimate + its `dz` recurrence) are computed only when set, and
/// fully dead-code-eliminated otherwise. `ATOM` is always `false` here: no
/// coloring path reads the atom-domain channel, so the render path realizes the
/// atom strip for free (the trait `sample` used by `navigate` is untouched).
///
/// The `(trap, de)` match is total, so every config maps to a concrete
/// monomorphization; [`crate::coloring::required_channels`] is conservative, so
/// an unrecognized mode resolves to all-on (correct-but-slow). Trap phase is
/// always [`PHASE_GATED`] — the production strategy, identical to the trait path.
pub fn iterate_samples_f64(
    backend: &F64Backend,
    frame: &Frame,
    ss: u32,
    channels: ChannelSet,
) -> SampleBuffer {
    match (channels.trap, channels.de) {
        (true, true) => {
            iterate_grid(frame, ss, |c, _dc| {
                backend.sample_flags::<true, false, true, PHASE_GATED>(c)
            })
        }
        (true, false) => {
            iterate_grid(frame, ss, |c, _dc| {
                backend.sample_flags::<true, false, false, PHASE_GATED>(c)
            })
        }
        (false, true) => {
            iterate_grid(frame, ss, |c, _dc| {
                backend.sample_flags::<false, false, true, PHASE_GATED>(c)
            })
        }
        (false, false) => {
            iterate_grid(frame, ss, |c, _dc| {
                backend.sample_flags::<false, false, false, PHASE_GATED>(c)
            })
        }
    }
}

/// Shared supersampled-grid iteration: the `dc`-from-pixel-geometry rule and row
/// parallelism live here once, so the trait path ([`iterate_samples`]) and the
/// f64 channel-dispatch path ([`iterate_samples_f64`]) produce identical geometry
/// and differ only in the per-subpixel sampler closure. `sample(c, dc)` receives
/// the absolute coordinate `c = center + dc` (read only by the f64 backend) and
/// the pixel offset `dc` (the only coordinate perturbation needs).
fn iterate_grid<F>(frame: &Frame, ss: u32, sample: F) -> SampleBuffer
where
    F: Fn(Complex<f64>, Complex<f64>) -> PixelSample + Sync,
{
    let w = frame.out_width;
    let h = frame.out_height;
    let s = ss.max(1);
    let sub_w = w * s;
    let sub_h = h * s;

    let fw = frame.frame_width;
    let fh = frame.frame_height();
    let sub_w_f = sub_w as f64;
    let sub_h_f = sub_h as f64;
    let center = frame.center;

    // Parallelize over subsample rows; collecting an ordered Vec keeps the
    // output deterministic for fixed parameters.
    let rows: Vec<Vec<PixelSample>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let mut row = Vec::with_capacity(sub_w as usize);
            // Fractional vertical position in [0,1); row 0 (top) is the largest
            // imaginary value, hence (0.5 - py_frac).
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let dc = Complex::new(dc_re, dc_im);
                let c = center + dc; // only the f64 backend reads this
                row.push(sample(c, dc));
            }
            row
        })
        .collect();

    let mut samples = Vec::with_capacity(sub_w as usize * sub_h as usize);
    for r in rows {
        samples.extend_from_slice(&r);
    }

    // Count output pixels touched by any glitched subsample (diagnostic only).
    let glitched_pixels = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut count = 0u64;
            for col in 0..w {
                let mut g = false;
                for sj in 0..s {
                    let base = ((row * s + sj) * sub_w + col * s) as usize;
                    for si in 0..s as usize {
                        if samples[base + si].glitched {
                            g = true;
                        }
                    }
                }
                if g {
                    count += 1;
                }
            }
            count
        })
        .sum();

    SampleBuffer {
        samples,
        out_width: w,
        out_height: h,
        ss: s,
        glitched_pixels,
    }
}

/// Stage 2 — **pure** shading + linear-light box downsample over a cached
/// buffer. Re-coloring is exactly a re-run of this function with different
/// `palette`/`params`; iteration is never repeated.
///
/// Each output pixel is the box average of its `ss × ss` subsamples, averaged
/// in linear light and only then sRGB-encoded.
pub fn shade_and_downsample(
    samples: &[PixelSample],
    out_w: u32,
    out_h: u32,
    ss: u32,
    palette: &Palette,
    params: &ColorParams,
    pixel_spacing: f64,
) -> RgbImage {
    let s = ss.max(1);
    let sub_w = out_w * s;
    let inv_samples = 1.0 / (s * s) as f64;

    let rows: Vec<Vec<u8>> = (0..out_h)
        .into_par_iter()
        .map(|row| {
            let mut out = Vec::with_capacity(out_w as usize * 3);
            for col in 0..out_w {
                let mut acc = [0.0f64; 3];
                for sj in 0..s {
                    let base = ((row * s + sj) * sub_w + col * s) as usize;
                    for si in 0..s as usize {
                        let lin = coloring::shade(&samples[base + si], palette, params, pixel_spacing);
                        acc[0] += lin[0];
                        acc[1] += lin[1];
                        acc[2] += lin[2];
                    }
                }
                for k in 0..3 {
                    let v = linear_to_srgb(acc[k] * inv_samples);
                    out.push((v * 255.0 + 0.5) as u8);
                }
            }
            out
        })
        .collect();

    let mut pixels = Vec::with_capacity(out_w as usize * out_h as usize * 3);
    for r in rows {
        pixels.extend_from_slice(&r);
    }
    RgbImage::from_raw(out_w, out_h, pixels).expect("buffer dimensions match")
}
