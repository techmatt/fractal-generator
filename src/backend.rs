//! Precision backends behind a trait.
//!
//! The per-pixel iteration sits behind [`FractalBackend`] so the precision tier
//! is swappable without the render driver, coloring, or CLI changing. Two tiers
//! exist now:
//!  - [`F64Backend`] — plain `f64` escape time, fast, accurate while pixel
//!    spacing stays clear of f64's relative epsilon (~1e-13 of `|c|`).
//!  - [`PerturbationBackend`] — single high-precision reference orbit plus
//!    per-pixel `f64` deltas with Zhuoran rebasing; clean far past where `f64`
//!    quantizes (v1 cap ~1e300 magnification).
//!
//! Backends are built **per frame**: `maxiter` and `bailout` live in the
//! constructor, and perturbation also computes and stores its reference orbit
//! there. The `dc` argument (the pixel's complex offset from the frame center,
//! computed straight from pixel geometry — never as `c - center`) is the only
//! coordinate perturbation needs; `c` is the absolute coordinate the f64 backend
//! uses and is accurate only at shallow depth.

use num_complex::Complex;

use astro_float::{BigFloat, RoundingMode};

use crate::hp;

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
    /// The pixel is unreliable: its `f64` delta underflowed to zero while the
    /// pixel had a nonzero offset from the reference (too deep for this tier).
    /// Additive field, defaults false — the f64 backend never sets it.
    pub glitched: bool,
}

impl PixelSample {
    pub const INTERIOR: PixelSample = PixelSample {
        escaped: false,
        smooth_iter: 0.0,
        glitched: false,
    };
}

/// A per-pixel iteration backend at a fixed precision tier. Holds `maxiter`
/// and `bailout` (and, for perturbation, the reference orbit), so the render
/// driver passes only per-pixel geometry.
pub trait FractalBackend: Sync {
    /// Iterate the pixel at absolute coordinate `c`, whose offset from the frame
    /// center is `dc`.
    fn sample(&self, c: Complex<f64>, dc: Complex<f64>) -> PixelSample;
}

/// Plain `f64` escape-time backend. Uses `c`, ignores `dc`.
pub struct F64Backend {
    maxiter: u32,
    bailout2: f64,
}

impl F64Backend {
    pub fn new(maxiter: u32, bailout: f64) -> Self {
        F64Backend {
            maxiter,
            bailout2: bailout * bailout,
        }
    }
}

impl FractalBackend for F64Backend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        let mut zr = 0.0f64;
        let mut zi = 0.0f64;
        // Scalar inner loop, autovectorization-friendly: no early Complex
        // abstraction overhead, no branches besides the bailout test.
        for n in 0..self.maxiter {
            let zr2 = zr * zr;
            let zi2 = zi * zi;
            if zr2 + zi2 > self.bailout2 {
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zr2 + zi2),
                    glitched: false,
                };
            }
            // z = z^2 + c
            zi = 2.0 * zr * zi + c.im;
            zr = zr2 - zi2 + c.re;
        }
        PixelSample::INTERIOR
    }
}

/// Single-reference perturbation backend with Zhuoran rebasing.
///
/// Stores the reference orbit `Z[0..L]` (`Z[0] = 0`) as `f64` projections — the
/// values stay O(1) until escape, so `f64` storage is exact enough. Each pixel
/// iterates a low-precision delta `δ` against the reference; rebasing keeps `δ`
/// well-scaled (glitch-free) without per-pixel high precision.
pub struct PerturbationBackend {
    /// Reference orbit, `Z[0] = 0`, length `L` (may be short if the reference
    /// escaped before `maxiter`).
    orbit: Vec<Complex<f64>>,
    maxiter: u32,
    bailout2: f64,
}

impl PerturbationBackend {
    /// Build the reference orbit at the high-precision center `(center_re,
    /// center_im)` and store its `f64` projection.
    ///
    /// Iterates `Z_{n+1} = Z_n² + C` from `Z_0 = 0` in `prec_bits`-bit floats
    /// until `|Z|² > bailout²` or `n == maxiter`, hand-rolling the three real
    /// ops (no complex-bignum type):
    /// `new_a = a² − b² + Ca`, `new_b = 2ab + Cb`.
    pub fn new(
        center_re: &BigFloat,
        center_im: &BigFloat,
        maxiter: u32,
        bailout: f64,
        prec_bits: usize,
    ) -> Self {
        let p = prec_bits;
        // Per the crate's perf note: skip rounding during the orbit and let the
        // f64 projection absorb the sub-ulp error.
        let rm = RoundingMode::None;
        let two = BigFloat::from_f64(2.0, p);
        let ca = center_re;
        let cb = center_im;

        let bailout2 = bailout * bailout;
        let mut a = BigFloat::from_f64(0.0, p);
        let mut b = BigFloat::from_f64(0.0, p);

        let mut orbit = Vec::with_capacity(maxiter as usize + 1);
        orbit.push(Complex::new(0.0, 0.0)); // Z[0]

        for _ in 0..maxiter {
            // new_a = a*a - b*b + Ca
            let a2 = a.mul(&a, p, rm);
            let b2 = b.mul(&b, p, rm);
            let new_a = a2.sub(&b2, p, rm).add(ca, p, rm);
            // new_b = 2*a*b + Cb
            let ab = a.mul(&b, p, rm);
            let new_b = ab.mul(&two, p, rm).add(cb, p, rm);
            a = new_a;
            b = new_b;

            let fa = hp::to_f64(&a);
            let fb = hp::to_f64(&b);
            orbit.push(Complex::new(fa, fb));
            if fa * fa + fb * fb > bailout2 {
                break; // reference escaped
            }
        }

        PerturbationBackend {
            orbit,
            maxiter,
            bailout2,
        }
    }

    /// Reference orbit length `L` (number of stored `Z` points).
    pub fn ref_len(&self) -> usize {
        self.orbit.len()
    }
}

impl FractalBackend for PerturbationBackend {
    #[inline]
    fn sample(&self, _c: Complex<f64>, dc: Complex<f64>) -> PixelSample {
        let z = &self.orbit;
        let l = z.len();
        let bailout2 = self.bailout2;
        let dc_nonzero = dc.re != 0.0 || dc.im != 0.0;

        // δ (complex f64 delta from the reference at index m), reference index m,
        // per-pixel iteration count n.
        let mut dr = 0.0f64;
        let mut di = 0.0f64;
        let mut m = 0usize;
        let mut n = 0u32;
        let mut glitched = false;

        loop {
            // δ_{n+1} = (2 Z[m] + δ) δ + dc
            let zr = z[m].re;
            let zi = z[m].im;
            let ar = 2.0 * zr + dr; // 2 Z[m] + δ
            let ai = 2.0 * zi + di;
            let nr = ar * dr - ai * di + dc.re; // complex (2Z+δ)·δ + dc
            let ni = ar * di + ai * dr + dc.im;
            dr = nr;
            di = ni;
            m += 1;
            n += 1;

            // Full value z = Z[m] + δ.
            let fr = z[m].re + dr;
            let fi = z[m].im + di;
            let zmag2 = fr * fr + fi * fi;

            if n >= self.maxiter {
                return PixelSample {
                    escaped: false,
                    smooth_iter: 0.0,
                    glitched,
                };
            }
            if zmag2 > bailout2 {
                // Escape test + smooth value use the FULL value, so both
                // backends agree on classification and coloring.
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zmag2),
                    glitched,
                };
            }

            let dmag2 = dr * dr + di * di;
            // Underflow flag: δ collapsed to exactly 0 on a pixel that has a
            // real offset — too deep for f64 deltas, result is unreliable.
            if dmag2 == 0.0 && dc_nonzero {
                glitched = true;
            }
            // Zhuoran rebase: when the full value is smaller than the delta (or
            // we've run off the end of the reference), re-anchor δ := z, m := 0.
            // n is untouched — rebasing is reference alignment only.
            if zmag2 < dmag2 || m >= l - 1 {
                dr = fr;
                di = fi;
                m = 0;
            }
        }
    }
}

/// Smooth (normalized) iteration count, given the escape iteration index `n`
/// and `|z|^2` at escape (from the FULL value, identical formula for both
/// backends so shallow renders match).
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
