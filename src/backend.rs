//! Precision backend behind a trait.
//!
//! The per-pixel iteration sits behind [`FractalBackend`] so a perturbation
//! backend (Prompt 2) can replace it without the render driver, coloring, or
//! CLI changing. Backends are constructed *per frame* so a future perturbation
//! backend can hold a reference orbit. The `dc` argument (the pixel's complex
//! offset from the frame center) is already in the interface even though the
//! f64 backend ignores it — perturbation needs it.

use num_complex::Complex;

/// Result of iterating a single pixel. Deliberately small: re-coloring must
/// never require re-iterating, so everything coloring needs lives here.
///
/// Prompt 3 will add: distance estimate, orbit-trap minimum + hit position.
#[derive(Clone, Copy, Debug)]
pub struct PixelSample {
    /// Whether the orbit escaped the bailout radius before `maxiter`.
    pub escaped: bool,
    /// Smooth (normalized) iteration count. Valid only when `escaped`.
    pub smooth_iter: f64,
}

impl PixelSample {
    pub const INTERIOR: PixelSample = PixelSample {
        escaped: false,
        smooth_iter: 0.0,
    };
}

/// A per-pixel iteration backend at a fixed precision tier.
pub trait FractalBackend: Sync {
    /// Iterate the pixel at absolute coordinate `c`, whose offset from the
    /// frame center is `dc`. `bailout` is the escape radius (not squared).
    fn sample(&self, c: Complex<f64>, dc: Complex<f64>, maxiter: u32, bailout: f64) -> PixelSample;
}

/// Plain `f64` escape-time backend. Uses `c`, ignores `dc`.
pub struct F64Backend;

impl FractalBackend for F64Backend {
    #[inline]
    fn sample(
        &self,
        c: Complex<f64>,
        _dc: Complex<f64>,
        maxiter: u32,
        bailout: f64,
    ) -> PixelSample {
        let bailout2 = bailout * bailout;
        let mut zr = 0.0f64;
        let mut zi = 0.0f64;
        // Scalar inner loop, autovectorization-friendly: no early Complex
        // abstraction overhead, no branches besides the bailout test.
        for n in 0..maxiter {
            let zr2 = zr * zr;
            let zi2 = zi * zi;
            if zr2 + zi2 > bailout2 {
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zr2 + zi2),
                };
            }
            // z = z^2 + c
            zi = 2.0 * zr * zi + c.im;
            zr = zr2 - zi2 + c.re;
        }
        PixelSample::INTERIOR
    }
}

/// Smooth (normalized) iteration count, given the escape iteration index `n`
/// and `|z|^2` at escape.
///
/// `nu = (n + 1) - log2(ln|z|) = (n + 1) - ln(ln|z|)/ln(2)`.
/// With a large bailout (1e6) `ln|z| ≈ 13.8`, well clear of the degenerate
/// region; the guard below protects against any pathological `|z|`.
#[inline]
fn smooth(n: u32, norm_sqr: f64) -> f64 {
    let log_zn = 0.5 * norm_sqr.ln(); // ln|z|
    // Guard the double-log: ln(log_zn) is only finite for log_zn > 0
    // (i.e. |z| > 1). Anything else falls back to the integer count.
    if log_zn > 0.0 && log_zn.is_finite() {
        (n + 1) as f64 - log_zn.ln() / std::f64::consts::LN_2
    } else {
        (n + 1) as f64
    }
}
