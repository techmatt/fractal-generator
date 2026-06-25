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

use crate::backend::{F64Backend, FractalBackend, PixelSample, Trap, PHASE_GATED};
use crate::coloring::{self, ChannelSet, ColorParams};
use crate::palette::{linear_to_srgb, Palette};
use crate::probe::SplitMix64;

/// Where the supersample sub-points land **within** each output pixel. The box
/// downsample in [`shade_and_downsample`] is an unweighted sum over the `ss²`
/// sub-slots regardless of pattern, so a pattern only changes *where* each
/// sub-slot is sampled in the plane — never the layout of the cached buffer.
///
/// [`SubsamplePattern::Grid`] reproduces the historical ordered-grid path
/// **byte-for-byte** (the `0.5`-centered sub-cells); the other two are the AA
/// study's placement axis at a fixed 4-spp (ss2) budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubsamplePattern {
    /// Ordered grid: sub-cell centers `((si+0.5)/s, (sj+0.5)/s)`. Any `ss`.
    Grid,
    /// Rotated grid / 4-rooks (canonical arctan½ ≈ 26.6°). **ss2 only.**
    Rgss,
    /// Stratified jitter: one seeded SplitMix64 sample per sub-cell. Any `ss`.
    Jitter,
}

/// Canonical 4-rook (rotated-grid) sub-pixel offsets for ss2, returned as the
/// *within-sub-cell* local offset `(lx, ly) ∈ [0,1)²` for sub-cell `(si, sj)`.
/// The four rooks `{(⅛,⅝),(⅜,⅛),(⅝,⅞),(⅞,⅜)}` each fall in a distinct quadrant,
/// so mapping by quadrant places one rook per sub-slot. (Which slot holds which
/// rook is irrelevant — the downsample sums all four — but keeping it canonical
/// keeps the offsets self-documenting.) Non-ss2 sub-cells fall back to center.
#[inline]
fn rgss_offset(si: u32, sj: u32) -> (f64, f64) {
    match (si, sj) {
        (0, 0) => (0.75, 0.25), // rook (3/8,1/8)
        (0, 1) => (0.25, 0.25), // rook (1/8,5/8)
        (1, 0) => (0.75, 0.75), // rook (7/8,3/8)
        (1, 1) => (0.25, 0.75), // rook (5/8,7/8)
        _ => (0.5, 0.5),
    }
}

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
    iterate_grid(frame, ss, SubsamplePattern::Grid, 0, |c, dc| backend.sample(c, dc))
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
    iterate_samples_f64_pattern(backend, frame, ss, channels, SubsamplePattern::Grid, 0)
}

/// As [`iterate_samples_f64`], with a selectable sub-pixel sample **pattern**
/// (AA study). `pattern == Grid` is byte-identical to [`iterate_samples_f64`];
/// `seed` is consumed only by [`SubsamplePattern::Jitter`]. Same channel-intent
/// dispatch — the pattern only moves where each sub-slot is sampled.
pub fn iterate_samples_f64_pattern(
    backend: &F64Backend,
    frame: &Frame,
    ss: u32,
    channels: ChannelSet,
    pattern: SubsamplePattern,
    seed: u64,
) -> SampleBuffer {
    match (channels.trap, channels.de) {
        (true, true) => {
            iterate_grid(frame, ss, pattern, seed, |c, _dc| {
                backend.sample_flags::<true, false, true, PHASE_GATED>(c)
            })
        }
        (true, false) => {
            iterate_grid(frame, ss, pattern, seed, |c, _dc| {
                backend.sample_flags::<true, false, false, PHASE_GATED>(c)
            })
        }
        (false, true) => {
            iterate_grid(frame, ss, pattern, seed, |c, _dc| {
                backend.sample_flags::<false, false, true, PHASE_GATED>(c)
            })
        }
        (false, false) => {
            iterate_grid(frame, ss, pattern, seed, |c, _dc| {
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
fn iterate_grid<F>(
    frame: &Frame,
    ss: u32,
    pattern: SubsamplePattern,
    seed: u64,
    sample: F,
) -> SampleBuffer
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
            if pattern == SubsamplePattern::Grid {
                // Historical ordered-grid path — kept literally identical so its
                // float ops stay byte-for-byte unchanged. Fractional vertical
                // position in [0,1); row 0 (top) is the largest imaginary value,
                // hence (0.5 - py_frac).
                let py = srow as f64 + 0.5;
                let dc_im = (0.5 - py / sub_h_f) * fh;
                for scol in 0..sub_w {
                    let px = scol as f64 + 0.5;
                    let dc_re = (px / sub_w_f - 0.5) * fw;
                    let dc = Complex::new(dc_re, dc_im);
                    let c = center + dc; // only the f64 backend reads this
                    row.push(sample(c, dc));
                }
            } else {
                // Pattern path: the sub-point lands at `scol + lx` / `srow + ly`
                // where `(lx, ly) ∈ [0,1)²` is the within-sub-cell offset. For a
                // grid `(lx, ly) = (0.5, 0.5)` recovers the branch above; the
                // y-offset varies along the row (rgss/jitter depend on `si`), so
                // dc_im is computed per sub-point here.
                let sj = srow % s;
                for scol in 0..sub_w {
                    let si = scol % s;
                    let (lx, ly) = match pattern {
                        SubsamplePattern::Grid => (0.5, 0.5),
                        SubsamplePattern::Rgss => rgss_offset(si, sj),
                        SubsamplePattern::Jitter => {
                            // Deterministic per sub-point: (scol, srow) uniquely
                            // identify it, so order/threading never affects the draw.
                            let mut rng = SplitMix64(
                                seed.wrapping_add(
                                    (scol as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                                )
                                .wrapping_add(
                                    (srow as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                                ),
                            );
                            (rng.unit(), rng.unit())
                        }
                    };
                    let px = scol as f64 + lx;
                    let py = srow as f64 + ly;
                    let dc_re = (px / sub_w_f - 0.5) * fw;
                    let dc_im = (0.5 - py / sub_h_f) * fh;
                    let dc = Complex::new(dc_re, dc_im);
                    let c = center + dc; // only the f64 backend reads this
                    row.push(sample(c, dc));
                }
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

/// Reconstruction filter for the supersample→output downsample.
///
/// [`DownsampleFilter::Box`] is the flat `ss×ss` average — byte-identical to
/// [`shade_and_downsample`]. The others are real separable reconstruction kernels
/// **scaled to the `ss×` minification**: their radius in source-sample units is
/// `radius() · ss`, so they reach into neighbouring output pixels' subsamples. A
/// unit-radius kernel at `ss×` downsample under-blurs and aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownsampleFilter {
    /// Flat `ss×ss` average (current default; byte-identical to box downsample).
    Box,
    /// Mitchell–Netravali cubic, `B = C = ⅓`. Radius 2 (dest px).
    Mitchell,
    /// Lanczos windowed-sinc, `a = 3`. Radius 3 (dest px); has negative lobes.
    Lanczos3,
}

impl DownsampleFilter {
    /// Base kernel radius in *destination* pixels (before the `ss` scaling).
    fn radius(self) -> f64 {
        match self {
            DownsampleFilter::Box => 0.5,
            DownsampleFilter::Mitchell => 2.0,
            DownsampleFilter::Lanczos3 => 3.0,
        }
    }

    /// Evaluate the 1-D kernel at `t` *destination-pixel* units from the center.
    fn eval(self, t: f64) -> f64 {
        let x = t.abs();
        match self {
            DownsampleFilter::Box => {
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            DownsampleFilter::Mitchell => {
                const B: f64 = 1.0 / 3.0;
                const C: f64 = 1.0 / 3.0;
                if x < 1.0 {
                    ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
                        + (-18.0 + 12.0 * B + 6.0 * C) * x * x
                        + (6.0 - 2.0 * B))
                        / 6.0
                } else if x < 2.0 {
                    ((-B - 6.0 * C) * x * x * x
                        + (6.0 * B + 30.0 * C) * x * x
                        + (-12.0 * B - 48.0 * C) * x
                        + (8.0 * B + 24.0 * C))
                        / 6.0
                } else {
                    0.0
                }
            }
            DownsampleFilter::Lanczos3 => {
                if x < 1e-12 {
                    1.0
                } else if x < 3.0 {
                    let pix = std::f64::consts::PI * x;
                    let pix3 = pix / 3.0;
                    (pix.sin() / pix) * (pix3.sin() / pix3)
                } else {
                    0.0
                }
            }
        }
    }
}

/// One output sample's separable kernel: the contiguous run of source samples
/// `[start, start+w.len())` and their (already normalized) weights.
struct FilterTaps {
    start: usize,
    w: Vec<f64>,
}

/// Precompute the per-output-coordinate kernel taps for a 1-D `src_len → dst_len`
/// minification (`dst_len · ss == src_len`). Output coordinate `d` is centered at
/// source position `(d + 0.5)·ss`; source sample `sx` sits at `sx + 0.5`. The tap
/// argument is the destination-unit distance `t = ((sx+0.5) − center)/ss`, and the
/// support radius in source units is `radius()·ss`. Weights are normalized to sum
/// to 1 (kernels integrate to ≈1, but edge clamping + discreteness drift).
fn build_taps(dst_len: usize, src_len: usize, ss: u32, filter: DownsampleFilter) -> Vec<FilterTaps> {
    let ssf = ss as f64;
    let r = filter.radius() * ssf;
    (0..dst_len)
        .map(|d| {
            let center = (d as f64 + 0.5) * ssf;
            let lo = (center - r).floor() as isize;
            let hi = (center + r).ceil() as isize;
            let mut start = None;
            let mut w = Vec::new();
            let mut sum = 0.0;
            for sx in lo..=hi {
                if sx < 0 || sx >= src_len as isize {
                    continue;
                }
                if start.is_none() {
                    start = Some(sx as usize);
                }
                let t = (sx as f64 + 0.5 - center) / ssf;
                let wt = filter.eval(t);
                w.push(wt);
                sum += wt;
            }
            if sum != 0.0 {
                for x in w.iter_mut() {
                    *x /= sum;
                }
            }
            FilterTaps { start: start.unwrap_or(0), w }
        })
        .collect()
}

/// Stage 2, generalized — shade the SS buffer and downsample with a **separable
/// reconstruction filter** (the AA filter study). `Box` delegates to
/// [`shade_and_downsample`] (byte-identical flat `ss×ss` average); the others run
/// two separable passes over the shaded buffer.
///
/// Three correctness cruxes vs. a naive resample: (1) the kernel is scaled to the
/// `ss×` minification (see [`DownsampleFilter`]); (2) filtering happens **in linear
/// light** — same space the box average uses; shading per subpixel then weighting;
/// (3) the result is **clamped to [0,1]** before sRGB8 quantization, since
/// Mitchell/Lanczos negative lobes overshoot on high-contrast edges.
///
/// Passes: horizontal (`sub_w → out_w`, shading each SS row on the fly) into a
/// `out_w × sub_h` f32 intermediate (~0.17 GB at ss4 2560×1440), then vertical
/// (`sub_h → out_h`).
pub fn shade_and_downsample_filtered(
    samples: &[PixelSample],
    out_w: u32,
    out_h: u32,
    ss: u32,
    palette: &Palette,
    params: &ColorParams,
    pixel_spacing: f64,
    filter: DownsampleFilter,
) -> RgbImage {
    if filter == DownsampleFilter::Box {
        return shade_and_downsample(samples, out_w, out_h, ss, palette, params, pixel_spacing);
    }
    let s = ss.max(1);
    let sub_w = (out_w * s) as usize;
    let sub_h = (out_h * s) as usize;
    let ow = out_w as usize;
    let oh = out_h as usize;

    let htaps = build_taps(ow, sub_w, s, filter);
    let vtaps = build_taps(oh, sub_h, s, filter);

    // Horizontal pass — shade each SS row to linear on the fly, then collapse
    // sub_w → out_w. Intermediate is f32 (out_w × sub_h).
    let inter: Vec<Vec<[f32; 3]>> = (0..sub_h)
        .into_par_iter()
        .map(|r| {
            let base = r * sub_w;
            let lin: Vec<[f64; 3]> = (0..sub_w)
                .map(|c| coloring::shade(&samples[base + c], palette, params, pixel_spacing))
                .collect();
            let mut row = vec![[0f32; 3]; ow];
            for (x, tap) in htaps.iter().enumerate() {
                let mut acc = [0.0f64; 3];
                for (k, &w) in tap.w.iter().enumerate() {
                    let p = lin[tap.start + k];
                    acc[0] += w * p[0];
                    acc[1] += w * p[1];
                    acc[2] += w * p[2];
                }
                row[x] = [acc[0] as f32, acc[1] as f32, acc[2] as f32];
            }
            row
        })
        .collect();

    // Vertical pass — collapse sub_h → out_h, clamp, sRGB-encode.
    let out_rows: Vec<Vec<u8>> = (0..oh)
        .into_par_iter()
        .map(|y| {
            let tap = &vtaps[y];
            let mut out = Vec::with_capacity(ow * 3);
            for x in 0..ow {
                let mut acc = [0.0f64; 3];
                for (k, &w) in tap.w.iter().enumerate() {
                    let p = inter[tap.start + k][x];
                    acc[0] += w * p[0] as f64;
                    acc[1] += w * p[1] as f64;
                    acc[2] += w * p[2] as f64;
                }
                for c in 0..3 {
                    let v = linear_to_srgb(acc[c].clamp(0.0, 1.0));
                    out.push((v * 255.0 + 0.5) as u8);
                }
            }
            out
        })
        .collect();

    let mut pixels = Vec::with_capacity(ow * oh * 3);
    for r in out_rows {
        pixels.extend_from_slice(&r);
    }
    RgbImage::from_raw(out_w, out_h, pixels).expect("buffer dimensions match")
}

/// Fraction of samples that did not escape (interior pixels). A fast black-pixel
/// proxy from raw iteration data, without shading.
///
/// Interior pixels render as dead black under `InteriorMode::Black` (the
/// presentation default), so this is an accurate stand-in for OKLab L < 0.08
/// when evaluated on the raw buffer before `shade_and_downsample`.
pub fn black_fraction(samples: &[PixelSample]) -> f32 {
    if samples.is_empty() {
        return 1.0;
    }
    let n = samples.iter().filter(|s| !s.escaped).count();
    n as f32 / samples.len() as f32
}

/// Iterate one shallow-f64 wallpaper frame **once**, returning the cached
/// supersampled buffer and the colorer's pixel spacing. This is the shared crop
/// primitive: present / palette_probe / enrich iterate here and shade the buffer
/// one or many times, so re-coloring never re-iterates. Channels are derived
/// from `params` (palette-independent), so the buffer holds exactly what the
/// colorer reads across every palette in a recolor set.
///
/// **Shallow f64 only** — callers assert the pixel spacing stays clear of the
/// f64 quantization floor (`probe::PERTURB_SPACING`) before calling.
pub fn iterate_crop_buffer_f64(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    bailout: f64,
    trap: Trap,
    params: &ColorParams,
) -> (SampleBuffer, f64) {
    let channels = coloring::required_channels(params);
    let backend = F64Backend::new(maxiter, bailout, trap);
    let buf = iterate_samples_f64(&backend, frame, ss, channels);
    let pixel_spacing = frame.frame_width / frame.out_width as f64;
    (buf, pixel_spacing)
}

/// Iterate + shade one shallow-f64 crop under a single palette — the single-shot
/// convenience over [`iterate_crop_buffer_f64`] (recolor-many callers keep the
/// buffer instead). Byte-identical to the inline iterate→shade the crop writers
/// used.
#[allow(clippy::too_many_arguments)]
pub fn render_crop_f64(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    bailout: f64,
    trap: Trap,
    palette: &Palette,
    params: &ColorParams,
    filter: DownsampleFilter,
) -> RgbImage {
    let (buf, pixel_spacing) = iterate_crop_buffer_f64(frame, ss, maxiter, bailout, trap, params);
    shade_and_downsample_filtered(
        &buf.samples,
        frame.out_width,
        frame.out_height,
        ss,
        palette,
        params,
        pixel_spacing,
        filter,
    )
}

/// Save an `RgbImage` as JPEG at an explicit quality via the explicit encoder
/// (the `image::save` default is 75; the wallpaper crops want q≈90). Shared by
/// the present/palette_probe/enrich crop writers.
pub fn save_jpeg(img: &RgbImage, path: &std::path::Path, quality: u8) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, quality);
    enc.encode(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("encode jpeg {}: {e}", path.display()))
}
