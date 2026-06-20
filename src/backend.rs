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

use std::f64::consts::PI;

use num_complex::Complex;

use astro_float::{BigFloat, RoundingMode};

use crate::hp;

/// Result of iterating a single pixel. Deliberately small: re-coloring must
/// never require re-iterating, so everything coloring needs lives here.
///
/// Channel validity:
///  - `smooth_iter`, `de` — **exterior only** (valid when `escaped`).
///  - `trap_min`, `trap_phase` — **all pixels**, interior included. A
///    non-escaping orbit still has a closest approach to the trap, so these
///    fill the interior (the "no dead black" coloring) as well as the exterior.
#[derive(Clone, Copy, Debug)]
pub struct PixelSample {
    /// Whether the orbit escaped the bailout radius before `maxiter`.
    pub escaped: bool,
    /// Smooth (normalized) iteration count. Valid only when `escaped`.
    pub smooth_iter: f64,
    /// Raw distance estimate in plane units: `|z|·ln|z| / |dz|`. Exterior only
    /// (`0.0` when interior). The coloring stage normalizes by pixel spacing.
    pub de: f64,
    /// Minimum orbit-trap distance over the orbit (`n ≥ 1`, skipping `z_0 = 0`).
    /// Valid for every pixel, interior included.
    pub trap_min: f64,
    /// Normalized angle `[0,1)` of `z − trap_center` at the trap-minimizing
    /// iteration. Valid for every pixel.
    pub trap_phase: f64,
    /// The pixel is unreliable: its `f64` delta underflowed to zero while the
    /// pixel had a nonzero offset from the reference (too deep for this tier).
    /// Additive field, defaults false — the f64 backend never sets it.
    pub glitched: bool,
}

/// Geometric orbit trap. The orbit's closest approach to this shape (over all
/// iterations, fed the **full value** `z`) drives trap coloring, so both
/// backends compute it identically.
#[derive(Clone, Copy, Debug)]
pub struct Trap {
    pub shape: TrapShape,
    pub center: Complex<f64>,
    /// Radius (circle trap only; ignored otherwise).
    pub radius: f64,
}

/// Trap geometry. Three shapes give three distinct aesthetics.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum TrapShape {
    /// `d = |z − p|` — pearled beads.
    Point,
    /// `d = min(|Re(z−p)|, |Im(z−p)|)` — thorny / organic.
    Cross,
    /// `d = | |z − p| − r |` — overlapping scales.
    Circle,
}

impl Trap {
    /// Trap distance and normalized phase of `z` relative to the trap center.
    /// Phase is `(atan2(Im, Re) / 2π).rem_euclid(1.0)`.
    #[inline]
    fn eval(&self, zr: f64, zi: f64) -> (f64, f64) {
        let dr = zr - self.center.re;
        let di = zi - self.center.im;
        let dist = match self.shape {
            TrapShape::Point => (dr * dr + di * di).sqrt(),
            TrapShape::Cross => dr.abs().min(di.abs()),
            TrapShape::Circle => ((dr * dr + di * di).sqrt() - self.radius).abs(),
        };
        let phase = (di.atan2(dr) / (2.0 * PI)).rem_euclid(1.0);
        (dist, phase)
    }
}

/// Exterior distance estimate `DE = |z|·ln|z| / |dz|` from the escaped full
/// value `z` (with `|z|² = zmag2`) and its carried derivative `dz`.
///
/// Known v1 limitation: `dz` is carried in plain `f64` and can overflow to a
/// non-finite value at very high `maxiter` / deep zoom (the regime that
/// motivates the deferred floatexp tier). When `dz` is non-finite (or zero), we
/// clamp `de = 0` — treating the pixel as an infinitely thin filament — rather
/// than emitting a NaN that would poison the coloring stage.
#[inline]
fn exterior_de(dzr: f64, dzi: f64, zmag2: f64) -> f64 {
    let dzmag2 = dzr * dzr + dzi * dzi;
    if !dzmag2.is_finite() || dzmag2 == 0.0 {
        return 0.0;
    }
    let zmag = zmag2.sqrt();
    let de = zmag * zmag.ln() / dzmag2.sqrt();
    if de.is_finite() {
        de
    } else {
        0.0
    }
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
    trap: Trap,
}

impl F64Backend {
    pub fn new(maxiter: u32, bailout: f64, trap: Trap) -> Self {
        F64Backend {
            maxiter,
            bailout2: bailout * bailout,
            trap,
        }
    }
}

impl FractalBackend for F64Backend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        // Canonical loop shared with the perturbation backend (minus δ/m/Z, using
        // z directly), so both agree on classification, smooth value, DE, and
        // the set of orbit points that feed the trap.
        let mut zr = 0.0f64; // z_0
        let mut zi = 0.0f64;
        let mut dzr = 0.0f64; // dz_0
        let mut dzi = 0.0f64;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;

        loop {
            // dz_{n+1} = 2·z_n·dz_n + 1 (z still holds z_n from the prior step).
            let ndzr = 2.0 * (zr * dzr - zi * dzi) + 1.0;
            let ndzi = 2.0 * (zr * dzi + zi * dzr);
            dzr = ndzr;
            dzi = ndzi;

            // z = z^2 + c
            let nzr = zr * zr - zi * zi + c.re;
            let nzi = 2.0 * zr * zi + c.im;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            let (d, ph) = self.trap.eval(zr, zi);
            if d < trap_min {
                trap_min = d;
                trap_phase = ph;
            }

            if n >= self.maxiter {
                return PixelSample {
                    escaped: false,
                    smooth_iter: 0.0,
                    de: 0.0,
                    trap_min,
                    trap_phase,
                    glitched: false,
                };
            }
            if zmag2 > self.bailout2 {
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zmag2),
                    de: exterior_de(dzr, dzi, zmag2),
                    trap_min,
                    trap_phase,
                    glitched: false,
                };
            }
        }
    }
}

/// Plain `f64` Julia escape-time backend: `z₀ = pixel`, a **fixed** parameter
/// `c`, iterating `z_{n+1} = z² + c`. Used only at base scale (whole-set view,
/// center `0`, width ~3.5), so f64 is always accurate — no perturbation tier.
///
/// Smooth value and orbit trap are computed exactly as the Mandelbrot backends
/// (full value `z`, trap skips `z₀`), so the same coloring stage applies. The
/// distance estimate is **not** carried (`de = 0`): the descend probe never
/// needs Julia DE, and the simple `dz` recurrence differs for Julia. `de = 0`
/// reads as "infinitely thin filament", which DE-shade/`--color de` would
/// misuse — the probe's default `--color smooth` sidesteps that.
pub struct JuliaBackend {
    /// Fixed Julia parameter `c` (the chosen Mandelbrot target, f64 projection).
    param: Complex<f64>,
    maxiter: u32,
    bailout2: f64,
    trap: Trap,
}

impl JuliaBackend {
    pub fn new(param: Complex<f64>, maxiter: u32, bailout: f64, trap: Trap) -> Self {
        JuliaBackend {
            param,
            maxiter,
            bailout2: bailout * bailout,
            trap,
        }
    }
}

impl FractalBackend for JuliaBackend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        // z₀ is the pixel; the parameter is fixed. (Mandelbrot uses z₀ = 0 and
        // the pixel as the parameter — that's the only structural difference.)
        let mut zr = c.re;
        let mut zi = c.im;
        let cr = self.param.re;
        let ci = self.param.im;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;

        loop {
            // z = z² + c
            let nzr = zr * zr - zi * zi + cr;
            let nzi = 2.0 * zr * zi + ci;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            let (d, ph) = self.trap.eval(zr, zi);
            if d < trap_min {
                trap_min = d;
                trap_phase = ph;
            }

            if n >= self.maxiter {
                return PixelSample {
                    escaped: false,
                    smooth_iter: 0.0,
                    de: 0.0,
                    trap_min,
                    trap_phase,
                    glitched: false,
                };
            }
            if zmag2 > self.bailout2 {
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zmag2),
                    de: 0.0, // Julia DE intentionally skipped.
                    trap_min,
                    trap_phase,
                    glitched: false,
                };
            }
        }
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
    trap: Trap,
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
        trap: Trap,
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
            trap,
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
        // Full value z = Z[m] + δ, carried so the DE derivative and trap use it
        // directly (z_0 = 0). The carried `dz` is continuous across rebasing —
        // it is a function of the full value, which a rebase doesn't change.
        let mut zr = 0.0f64;
        let mut zi = 0.0f64;
        let mut dzr = 0.0f64; // dz_0
        let mut dzi = 0.0f64;
        let mut m = 0usize;
        let mut n = 0u32;
        let mut glitched = false;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;

        loop {
            // dz_{n+1} = 2·z_n·dz_n + 1 (full value z still holds z_n; a rebase
            // never touched it, so this is unaffected by rebasing).
            let ndzr = 2.0 * (zr * dzr - zi * dzi) + 1.0;
            let ndzi = 2.0 * (zr * dzi + zi * dzr);
            dzr = ndzr;
            dzi = ndzi;

            // δ_{n+1} = (2 Z[m] + δ) δ + dc
            let zmr = z[m].re;
            let zmi = z[m].im;
            let ar = 2.0 * zmr + dr; // 2 Z[m] + δ
            let ai = 2.0 * zmi + di;
            let nr = ar * dr - ai * di + dc.re; // complex (2Z+δ)·δ + dc
            let ni = ar * di + ai * dr + dc.im;
            dr = nr;
            di = ni;
            m += 1;
            n += 1;

            // Full value z = Z[m] + δ.
            zr = z[m].re + dr;
            zi = z[m].im + di;
            let zmag2 = zr * zr + zi * zi;

            let (d, ph) = self.trap.eval(zr, zi);
            if d < trap_min {
                trap_min = d;
                trap_phase = ph;
            }

            if n >= self.maxiter {
                return PixelSample {
                    escaped: false,
                    smooth_iter: 0.0,
                    de: 0.0,
                    trap_min,
                    trap_phase,
                    glitched,
                };
            }
            if zmag2 > bailout2 {
                // Escape test + smooth value + DE use the FULL value, so both
                // backends agree on classification and coloring.
                return PixelSample {
                    escaped: true,
                    smooth_iter: smooth(n, zmag2),
                    de: exterior_de(dzr, dzi, zmag2),
                    trap_min,
                    trap_phase,
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
            // n and the carried full value / dz are untouched — rebasing is
            // reference alignment only.
            if zmag2 < dmag2 || m >= l - 1 {
                dr = zr;
                di = zi;
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
