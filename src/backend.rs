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
    /// **Navigation channel, not coloring.** The iteration `n ≥ 1` at which the
    /// orbit made its closest approach to the origin (`z_0 = 0` is skipped). This
    /// is the period of the nearby minibrot — the atom domain. The coloring
    /// stage ignores it (the feature-navigation tooling that consumed it was
    /// retired in the P2 subcommand cull). Both Mandelbrot
    /// backends populate it identically (perturbation feeds the full value
    /// `Z[m]+δ`, so `n` is absolute and unaffected by rebasing); Julia leaves the
    /// default `0`.
    pub atom_period: u32,
    /// **Navigation channel, not coloring.** Minimum `|z_n|` over the orbit
    /// (`n ≥ 1`): how near this pixel passes a nucleus. Pixels with the smallest
    /// `atom_min` of a given `atom_period` sit nearest that period's nucleus.
    /// Default `f64::INFINITY` (Julia).
    pub atom_min: f64,
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
    /// Trap **distance** of `z` to the shape (no phase). This is the only part
    /// computed every iteration: the loop tracks the running minimum distance
    /// (and the minimizing `z`), and the phase is deferred to a single post-loop
    /// [`phase_at`](Trap::phase_at) call on the captured minimizer.
    ///
    /// The per-iteration `atan2` the combined eval used to do dominated the
    /// kernel (~88%) yet its result was consumed only at the one trap-minimizing
    /// iteration — computed ~100× more often than used. Splitting distance from
    /// phase removes it from the hot loop.
    #[inline]
    fn eval_dist(&self, zr: f64, zi: f64) -> f64 {
        let dr = zr - self.center.re;
        let di = zi - self.center.im;
        match self.shape {
            TrapShape::Point => (dr * dr + di * di).sqrt(),
            TrapShape::Cross => dr.abs().min(di.abs()),
            TrapShape::Circle => ((dr * dr + di * di).sqrt() - self.radius).abs(),
        }
    }

    /// Normalized phase `[0,1)` of `z − center`: `(atan2(Im, Re) / 2π).rem_euclid(1.0)`.
    /// Computed **once**, after the loop, on the captured trap-minimizing `z`.
    ///
    /// Byte-identical to the old inline phase: `dr`/`di` are re-derived here from
    /// the captured `zr`/`zi` by the same deterministic subtraction, and `atan2`
    /// is a pure function of its input bits — so a post-loop call on the captured
    /// minimizer returns the exact bits the inline call returned at that
    /// iteration. (The only failure mode is capturing the wrong `z`; the loop
    /// captures it at the identical `<` comparison that selected the minimum.)
    #[inline]
    fn phase_at(&self, zr: f64, zi: f64) -> f64 {
        let dr = zr - self.center.re;
        let di = zi - self.center.im;
        (di.atan2(dr) / (2.0 * PI)).rem_euclid(1.0)
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

/// Trap-phase computation strategy — a `sample_flags` const-generic selector.
/// All three produce **bit-identical** `PixelSample`s (the phase `atan2` is a
/// pure function of its input bits, and all three store the phase of the same
/// trap-minimizing iteration); they differ only in *when*, and how often, the
/// `atan2` runs. The trait `sample` (the real render path) uses
/// [`PHASE_GATED`]; the others exist for the `profile` ceiling sub-ablation.
///
/// Measured ranking (decoration + interior workloads, 12 threads): GATED is
/// fastest. EVERY (the pre-change baseline) wastes an `atan2` on every iteration
/// though the result is consumed only at the trap minimum. DEFER removes the
/// per-iteration `atan2` but is *slower than GATED* — capturing the minimizing
/// `z` keeps two extra f64s live across the register-starved hot loop, and that
/// spill cost exceeds the handful of `atan2`s GATED still pays on min-updates.
/// Gating (compute the `atan2` only on a trap-min improvement) gets the win
/// without the extra live state.
pub const PHASE_GATED: u8 = 0;
/// Capture the minimizing `z`, compute one `atan2` after the loop. Slower than
/// [`PHASE_GATED`] (extra live state); kept only for the profile comparison.
pub const PHASE_DEFER: u8 = 1;
/// Compute the `atan2` every iteration (the pre-change baseline). The profile
/// ceiling reference: GATED's speedup is measured against this.
pub const PHASE_EVERY: u8 = 2;

/// Plain `f64` escape-time backend. Uses `c`, ignores `dc`.
///
/// `degree` is the escape recurrence exponent `d` in `z ← z^d + c` (parameter
/// plane). `degree == 2` is the classic Mandelbrot and takes the byte-identical
/// [`sample_flags`](Self::sample_flags) kernel; `degree ≥ 3` (the multibrot walker
/// / root-field families) takes [`sample_multibrot`](Self::sample_multibrot). Only
/// the trait [`sample`](FractalBackend::sample) dispatches on `degree`; the
/// channel-intent render path ([`crate::render::iterate_samples_f64`]) stays
/// degree-2 (`render`/`sheet` never drive multibrot through this backend —
/// `render-one` renders multibrot through `render_modes`).
pub struct F64Backend {
    maxiter: u32,
    bailout2: f64,
    trap: Trap,
    degree: u32,
}

impl F64Backend {
    /// Degree-2 (Mandelbrot) backend. Bit-for-bit the prior behaviour — `new` is
    /// exactly `new_degree(.., 2)`, and degree 2 routes through `sample_flags`.
    pub fn new(maxiter: u32, bailout: f64, trap: Trap) -> Self {
        F64Backend::new_degree(maxiter, bailout, trap, 2)
    }

    /// Degree-parametric constructor for the parameter-plane multibrot walker /
    /// root field (`z ← z^d + c`, `d ≥ 2`). `degree` is clamped to `≥ 2`; the
    /// trait [`sample`](FractalBackend::sample) routes `d == 2` through the
    /// byte-identical [`sample_flags`](Self::sample_flags) path and `d ≥ 3` through
    /// [`sample_multibrot`](Self::sample_multibrot).
    pub fn new_degree(maxiter: u32, bailout: f64, trap: Trap, degree: u32) -> Self {
        F64Backend {
            maxiter,
            bailout2: bailout * bailout,
            trap,
            degree: degree.max(2),
        }
    }
}

impl F64Backend {
    /// Per-iteration kernel, generic over which **bookkeeping** channels are
    /// computed. The core (`z² + c`, `|z|²` bailout test, iteration count, smooth
    /// value at escape) is always present; the three flags gate the work coloring
    /// *may* read but a given config may not:
    ///  - `DE` — the `dz` derivative recurrence + the escape-time `exterior_de`.
    ///  - `TRAP` — the orbit-trap distance/phase min-tracking.
    ///  - `ATOM` — the atom-domain closest-approach tracking (navigation-only;
    ///    no coloring path reads it).
    ///
    /// These are **const** generics so each combination monomorphizes to true
    /// dead-code-eliminated machine code, and `profile`'s ablation combos give
    /// the genuine "what if we never computed this" cost, not a runtime-branch
    /// approximation. Disabled channels leave their `PixelSample` fields at the
    /// inert defaults (`de = 0`, `trap_min = ∞`, `atom_* = 0/∞`).
    ///
    /// `PHASE` selects the trap-phase strategy ([`PHASE_GATED`] /
    /// [`PHASE_DEFER`] / [`PHASE_EVERY`]) — all bit-identical (see those consts
    /// and [`Trap::phase_at`]); production (`sample`) uses `PHASE_GATED`, the
    /// others let `profile` size the win against the per-iteration baseline.
    #[inline]
    pub fn sample_flags<
        const TRAP: bool,
        const ATOM: bool,
        const DE: bool,
        const PHASE: u8,
    >(
        &self,
        c: Complex<f64>,
    ) -> PixelSample {
        // Canonical loop shared with the perturbation backend (minus δ/m/Z, using
        // z directly), so both agree on classification, smooth value, DE, and
        // the set of orbit points that feed the trap.
        let mut zr = 0.0f64; // z_0
        let mut zi = 0.0f64;
        let mut dzr = 0.0f64; // dz_0
        let mut dzi = 0.0f64;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64; // GATED/EVERY write in-loop; DEFER writes post-loop.
        // Captured trap-minimizing full value (DEFER only; read post-loop).
        let mut trap_zr = 0.0f64;
        let mut trap_zi = 0.0f64;
        // Atom domain: closest approach of the full value to the origin (n ≥ 1).
        let mut atom_min2 = f64::INFINITY;
        let mut atom_period = 0u32;

        let escaped;
        let smooth_iter;
        let de;
        loop {
            // dz_{n+1} = 2·z_n·dz_n + 1 (z still holds z_n from the prior step).
            if DE {
                let ndzr = 2.0 * (zr * dzr - zi * dzi) + 1.0;
                let ndzi = 2.0 * (zr * dzi + zi * dzr);
                dzr = ndzr;
                dzi = ndzi;
            }

            // z = z^2 + c
            let nzr = zr * zr - zi * zi + c.re;
            let nzi = 2.0 * zr * zi + c.im;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            if TRAP {
                let d = self.trap.eval_dist(zr, zi);
                // PHASE selects when the (identical) phase atan2 runs; all const,
                // so only the chosen arm survives monomorphization.
                if PHASE == PHASE_EVERY {
                    let ph = self.trap.phase_at(zr, zi); // every iteration (baseline)
                    if d < trap_min {
                        trap_min = d;
                        trap_phase = ph;
                    }
                } else if PHASE == PHASE_GATED {
                    if d < trap_min {
                        trap_min = d;
                        trap_phase = self.trap.phase_at(zr, zi); // only on a min improvement
                    }
                } else if d < trap_min {
                    // PHASE_DEFER: capture the minimizer, atan2 once post-loop.
                    trap_min = d;
                    trap_zr = zr;
                    trap_zi = zi;
                }
            }
            if ATOM && zmag2 < atom_min2 {
                atom_min2 = zmag2;
                atom_period = n;
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                de = 0.0;
                break;
            }
            if zmag2 > self.bailout2 {
                escaped = true;
                smooth_iter = smooth(n, zmag2);
                de = if DE { exterior_de(dzr, dzi, zmag2) } else { 0.0 };
                break;
            }
        }

        // DEFER: one atan2 on the captured minimizer. The `is_finite` guard
        // reproduces the original default (`trap_phase = 0.0`) for the degenerate
        // case where the min never updated (`trap_min` stays ∞) — in practice the
        // first iteration always updates it.
        if TRAP && PHASE == PHASE_DEFER && trap_min.is_finite() {
            trap_phase = self.trap.phase_at(trap_zr, trap_zi);
        }

        PixelSample {
            escaped,
            smooth_iter,
            de,
            trap_min,
            trap_phase,
            glitched: false,
            atom_period,
            atom_min: atom_min2.sqrt(),
        }
    }

    /// General multibrot kernel: `z ← z^d + c` for `d = self.degree ≥ 3`
    /// (parameter plane, `z_0 = 0`). Every channel is live and its semantics match
    /// the degree-2 [`sample_flags`](Self::sample_flags)`::<true, true, true,
    /// PHASE_GATED>` path — trap distance + gated phase, atom-domain min, exterior
    /// DE via the degree-`d` derivative `dz ← d·z^{d-1}·dz + 1`, and the smooth
    /// value with the `ln d` outer log base. `z^{d-1}` (and `z^d = z·z^{d-1}`) are
    /// formed by repeated real-`f64` complex multiplication ([`cpow_f64`]).
    ///
    /// This is a **separate** kernel from `sample_flags`, not a generalization of
    /// it: keeping the degree-2 kernel textually untouched is what guarantees the
    /// Mandelbrot walker / root-field bytes are unchanged. `d ≥ 3` is a new
    /// capability with no byte-identity constraint, so this path is written for
    /// clarity (all channels, no const-generic ablation).
    fn sample_multibrot(&self, c: Complex<f64>) -> PixelSample {
        let d = self.degree;
        let d_f = d as f64;
        let mut zr = 0.0f64; // z_0
        let mut zi = 0.0f64;
        let mut dzr = 0.0f64; // dz_0
        let mut dzi = 0.0f64;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;
        let mut atom_min2 = f64::INFINITY;
        let mut atom_period = 0u32;

        let escaped;
        let smooth_iter;
        let de;
        loop {
            // z^{d-1} of the current z_n (drives both the derivative and z^d).
            let (pr, pi) = cpow_f64(zr, zi, d - 1);
            // dz_{n+1} = d·z_n^{d-1}·dz_n + 1 (parameter-plane +1). z_n still holds.
            let cr = pr * dzr - pi * dzi;
            let ci = pr * dzi + pi * dzr;
            dzr = d_f * cr + 1.0;
            dzi = d_f * ci;

            // z_{n+1} = z_n^d + c = z_n·z_n^{d-1} + c.
            let nzr = zr * pr - zi * pi + c.re;
            let nzi = zr * pi + zi * pr + c.im;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            // Gated trap phase: atan2 only on a trap-min improvement (as sample_flags).
            let dist = self.trap.eval_dist(zr, zi);
            if dist < trap_min {
                trap_min = dist;
                trap_phase = self.trap.phase_at(zr, zi);
            }
            if zmag2 < atom_min2 {
                atom_min2 = zmag2;
                atom_period = n;
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                de = 0.0;
                break;
            }
            if zmag2 > self.bailout2 {
                escaped = true;
                smooth_iter = smooth_deg(n, zmag2, d);
                de = exterior_de(dzr, dzi, zmag2);
                break;
            }
        }

        PixelSample {
            escaped,
            smooth_iter,
            de,
            trap_min,
            trap_phase,
            glitched: false,
            atom_period,
            atom_min: atom_min2.sqrt(),
        }
    }
}

impl FractalBackend for F64Backend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        // Degree 2 → the byte-identical Mandelbrot kernel (every channel live, trap
        // phase gated); degree ≥ 3 → the general multibrot kernel. The branch is per
        // pixel on a per-backend constant (perfectly predicted), so the maxiter-long
        // inner loop pays nothing and the degree-2 bytes are literally unchanged.
        if self.degree == 2 {
            self.sample_flags::<true, true, true, PHASE_GATED>(c)
        } else {
            self.sample_multibrot(c)
        }
    }
}

/// Plain `f64` Julia escape-time backend: `z₀ = pixel`, a **fixed** parameter
/// `c`, iterating `z_{n+1} = z^d + c`. Used only at base scale (whole-set view,
/// center `0`, width ~3.5), so f64 is always accurate — no perturbation tier.
///
/// `degree` is the dynamical-plane exponent `d` in `z ← z^d + c`. `degree == 2`
/// is the classic quadratic Julia and takes the byte-identical [`sample_deg2`](Self::sample_deg2)
/// kernel; `degree ≥ 3` (the **Julia-multibrot** dynamical families) takes
/// [`sample_multibrot`](Self::sample_multibrot). The trait
/// [`sample`](FractalBackend::sample) dispatches on `degree`.
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
    /// Dynamical-plane degree `d` (`z ← z^d + c`), clamped `≥ 2`.
    degree: u32,
}

impl JuliaBackend {
    /// Quadratic (degree-2) Julia. Bit-for-bit the prior behaviour — `new` is
    /// exactly `new_degree(.., 2)`, and degree 2 routes through `sample_deg2`.
    pub fn new(param: Complex<f64>, maxiter: u32, bailout: f64, trap: Trap) -> Self {
        JuliaBackend::new_degree(param, maxiter, bailout, trap, 2)
    }

    /// Degree-parametric constructor for the dynamical `z^d + c` Julia-multibrot
    /// (`d ≥ 2`). `degree` is clamped to `≥ 2`; the trait
    /// [`sample`](FractalBackend::sample) routes `d == 2` through the byte-identical
    /// [`sample_deg2`](Self::sample_deg2) path and `d ≥ 3` through
    /// [`sample_multibrot`](Self::sample_multibrot).
    pub fn new_degree(param: Complex<f64>, maxiter: u32, bailout: f64, trap: Trap, degree: u32) -> Self {
        JuliaBackend {
            param,
            maxiter,
            bailout2: bailout * bailout,
            trap,
            degree: degree.max(2),
        }
    }

    /// Quadratic Julia kernel (`z ← z² + c`) — textually the prior `sample`, kept
    /// intact so the degree-2 Julia bytes are unchanged.
    #[inline]
    fn sample_deg2(&self, c: Complex<f64>) -> PixelSample {
        // z₀ is the pixel; the parameter is fixed. (Mandelbrot uses z₀ = 0 and
        // the pixel as the parameter — that's the only structural difference.)
        let mut zr = c.re;
        let mut zi = c.im;
        let cr = self.param.re;
        let ci = self.param.im;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        // Gated trap phase: the `atan2` runs only on a trap-min improvement (see
        // `PHASE_GATED` / `Trap::phase_at`), not every iteration.
        let mut trap_phase = 0.0f64;

        let escaped;
        let smooth_iter;
        loop {
            // z = z² + c
            let nzr = zr * zr - zi * zi + cr;
            let nzi = 2.0 * zr * zi + ci;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            let d = self.trap.eval_dist(zr, zi);
            if d < trap_min {
                trap_min = d;
                trap_phase = self.trap.phase_at(zr, zi);
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                break;
            }
            if zmag2 > self.bailout2 {
                escaped = true;
                smooth_iter = smooth(n, zmag2);
                break;
            }
        }

        PixelSample {
            escaped,
            smooth_iter,
            de: 0.0, // Julia DE intentionally skipped.
            trap_min,
            trap_phase,
            glitched: false,
            atom_period: 0, // navigation channel unused for Julia
            atom_min: f64::INFINITY,
        }
    }

    /// Julia-multibrot kernel: `z ← z^d + c`, `z₀ = pixel`, fixed parameter `c`,
    /// for `d = self.degree ≥ 3`. The dynamical-plane twin of
    /// [`F64Backend::sample_multibrot`] — same `z^{d-1}` power primitive
    /// ([`cpow_f64`]) and the same degree-`d` smooth base ([`smooth_deg`]) — but
    /// `z₀ = pixel` and no parameter-plane `+1` (there is no `dz` recurrence at all,
    /// as Julia carries `de = 0`). Gated trap phase, atom channel unused.
    #[inline]
    fn sample_multibrot(&self, c: Complex<f64>) -> PixelSample {
        let d = self.degree;
        let mut zr = c.re; // z_0 = pixel
        let mut zi = c.im;
        let cr = self.param.re;
        let ci = self.param.im;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;

        let escaped;
        let smooth_iter;
        loop {
            // z = z^d + c = z·z^{d-1} + c.
            let (pr, pi) = cpow_f64(zr, zi, d - 1);
            let nzr = zr * pr - zi * pi + cr;
            let nzi = zr * pi + zi * pr + ci;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            let dist = self.trap.eval_dist(zr, zi);
            if dist < trap_min {
                trap_min = dist;
                trap_phase = self.trap.phase_at(zr, zi);
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                break;
            }
            if zmag2 > self.bailout2 {
                escaped = true;
                smooth_iter = smooth_deg(n, zmag2, d);
                break;
            }
        }

        PixelSample {
            escaped,
            smooth_iter,
            de: 0.0, // Julia DE intentionally skipped.
            trap_min,
            trap_phase,
            glitched: false,
            atom_period: 0,
            atom_min: f64::INFINITY,
        }
    }
}

impl FractalBackend for JuliaBackend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        // Degree 2 → the byte-identical quadratic Julia kernel; degree ≥ 3 → the
        // Julia-multibrot kernel. Per-backend constant, perfectly predicted.
        if self.degree == 2 {
            self.sample_deg2(c)
        } else {
            self.sample_multibrot(c)
        }
    }
}

/// Plain `f64` **Phoenix** escape-time backend: the Ushiki two-state dynamical
/// recurrence `z_{n+1} = z_n² + c + p·z_{n-1}`, `z₀ = pixel`, `z_{-1} = 0`, fixed
/// constants `c` (additive) and `p` (the `z_{n-1}` coefficient). Base-scale only
/// (f64 accurate; no perturbation tier).
///
/// **Escape / smooth normalization (derivation).** Near escape the quadratic term
/// dominates: once `|z_n|` is large, `|z_n²| = |z_n|²` swamps the linear memory term
/// `|p·z_{n-1}|` (which is `O(|z|)`), so `|z_{n+1}| ≈ |z_n|²` — the escape is
/// **degree-2**, and the standard `nu = (n+1) − log2(ln|z|)` smooth count applies
/// unchanged ([`smooth`], base `ln 2`). The memory term does not enter the outer-log
/// base. This is why a fast escape-time smooth field is clean here — the two-state
/// coupling reshapes the *set* but not the escape order.
///
/// DE is intentionally **not** carried (`de = 0`, as [`JuliaBackend`]). The Phoenix
/// derivative recurrence gains a `p·dz_{n-1}` term (`dz_{n+1} = 2·z_n·dz_n +
/// p·dz_{n-1}`) — carried by the slow beautiful kernel for the `de` field — but the
/// fast smooth field the guard/score path reads needs only the smooth channel, so
/// this backend skips it (no banding risk).
pub struct PhoenixBackend {
    /// Additive constant `c`.
    param_c: Complex<f64>,
    /// `z_{n-1}` coefficient `p` (Ushiki's `q`).
    param_p: Complex<f64>,
    maxiter: u32,
    bailout2: f64,
    trap: Trap,
}

impl PhoenixBackend {
    pub fn new(
        param_c: Complex<f64>,
        param_p: Complex<f64>,
        maxiter: u32,
        bailout: f64,
        trap: Trap,
    ) -> Self {
        PhoenixBackend {
            param_c,
            param_p,
            maxiter,
            bailout2: bailout * bailout,
            trap,
        }
    }
}

impl FractalBackend for PhoenixBackend {
    #[inline]
    fn sample(&self, c: Complex<f64>, _dc: Complex<f64>) -> PixelSample {
        // z₀ = pixel, z_{-1} = 0. Constants fixed.
        let mut zr = c.re;
        let mut zi = c.im;
        let mut zpr = 0.0f64; // z_{n-1}
        let mut zpi = 0.0f64;
        let cr = self.param_c.re;
        let ci = self.param_c.im;
        let pr = self.param_p.re;
        let pi = self.param_p.im;
        let mut n = 0u32;
        let mut trap_min = f64::INFINITY;
        let mut trap_phase = 0.0f64;

        let escaped;
        let smooth_iter;
        loop {
            // z_{n+1} = z_n² + c + p·z_{n-1}. Compute p·z_{n-1} (complex) then shift.
            let pzr = pr * zpr - pi * zpi;
            let pzi = pr * zpi + pi * zpr;
            let nzr = zr * zr - zi * zi + cr + pzr;
            let nzi = 2.0 * zr * zi + ci + pzi;
            zpr = zr; // z_{n-1} := z_n
            zpi = zi;
            zr = nzr;
            zi = nzi;
            n += 1;

            let zmag2 = zr * zr + zi * zi;
            let dist = self.trap.eval_dist(zr, zi);
            if dist < trap_min {
                trap_min = dist;
                trap_phase = self.trap.phase_at(zr, zi);
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                break;
            }
            if zmag2 > self.bailout2 {
                escaped = true;
                // Quadratic-dominated escape → the standard degree-2 smooth count.
                smooth_iter = smooth(n, zmag2);
                break;
            }
        }

        PixelSample {
            escaped,
            smooth_iter,
            de: 0.0, // Phoenix DE intentionally skipped (fast smooth field only).
            trap_min,
            trap_phase,
            glitched: false,
            atom_period: 0,
            atom_min: f64::INFINITY,
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
        // Gated trap phase: the `atan2` runs only on a trap-min improvement (see
        // `PHASE_GATED` / `Trap::phase_at`), not every iteration.
        let mut trap_phase = 0.0f64;
        // Atom domain: closest approach of the full value Z[m]+δ to the origin.
        // `n` is absolute (a rebase realigns `m`, not the iteration count), so
        // `atom_period` matches the f64 backend exactly.
        let mut atom_min2 = f64::INFINITY;
        let mut atom_period = 0u32;

        let escaped;
        let smooth_iter;
        let de;
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

            let d = self.trap.eval_dist(zr, zi);
            if d < trap_min {
                trap_min = d;
                trap_phase = self.trap.phase_at(zr, zi);
            }
            if zmag2 < atom_min2 {
                atom_min2 = zmag2;
                atom_period = n;
            }

            if n >= self.maxiter {
                escaped = false;
                smooth_iter = 0.0;
                de = 0.0;
                break;
            }
            if zmag2 > bailout2 {
                // Escape test + smooth value + DE use the FULL value, so both
                // backends agree on classification and coloring.
                escaped = true;
                smooth_iter = smooth(n, zmag2);
                de = exterior_de(dzr, dzi, zmag2);
                break;
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

        PixelSample {
            escaped,
            smooth_iter,
            de,
            trap_min,
            trap_phase,
            glitched,
            atom_period,
            atom_min: atom_min2.sqrt(),
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

/// Degree-`d` smooth value: the `smooth` formula with the correct outer log base.
/// Near escape `|z_{n+1}| ≈ |z_n|^d`, so the double-log base is the degree, not
/// always 2. Degree 2 pins the exact [`LN_2`](std::f64::consts::LN_2) constant, so
/// `smooth_deg(n, m², 2)` is bit-for-bit `smooth(n, m²)` — the same convention as
/// `render_modes::smooth_value`'s un-normalized twin (no `ln B` term here: the
/// backend's smooth is un-normalized, and the multibrot walker consumes it in the
/// same units the degree-2 path emits). Used by [`F64Backend::sample_multibrot`].
#[inline]
fn smooth_deg(n: u32, norm_sqr: f64, degree: u32) -> f64 {
    let log_zn = 0.5 * norm_sqr.ln(); // ln|z|
    let ln_d = if degree == 2 {
        std::f64::consts::LN_2
    } else {
        (degree as f64).ln()
    };
    if log_zn > 0.0 && log_zn.is_finite() {
        (n + 1) as f64 - log_zn.ln() / ln_d
    } else {
        (n + 1) as f64
    }
}

/// `z^k` (`k ≥ 1`) by repeated real-`f64` complex multiplication — the multibrot
/// kernel's power primitive (no `powf`/polar). `k = 1` returns `z`. Byte-note: for
/// `z^{d-1}` at `d = 2` this returns `z` unchanged, so the degree-2 recurrence it
/// would reconstruct (`z·z`, `2·z·dz`) matches the inline degree-2 kernel exactly —
/// but the trait `sample` never routes degree 2 here, so that identity is only a
/// design invariant, not a live path.
#[inline]
fn cpow_f64(zr: f64, zi: f64, k: u32) -> (f64, f64) {
    let mut pr = zr; // z^1
    let mut pi = zi;
    for _ in 1..k {
        let nr = pr * zr - pi * zi;
        let ni = pr * zi + pi * zr;
        pr = nr;
        pi = ni;
    }
    (pr, pi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traps() -> [Trap; 3] {
        [
            Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 },
            Trap { shape: TrapShape::Cross, center: Complex::new(0.13, -0.21), radius: 0.5 },
            Trap { shape: TrapShape::Circle, center: Complex::new(-0.31, 0.42), radius: 0.7 },
        ]
    }

    fn sample_eq(a: &PixelSample, b: &PixelSample) -> bool {
        a.escaped == b.escaped
            && a.smooth_iter.to_bits() == b.smooth_iter.to_bits()
            && a.de.to_bits() == b.de.to_bits()
            && a.trap_min.to_bits() == b.trap_min.to_bits()
            && a.trap_phase.to_bits() == b.trap_phase.to_bits()
            && a.atom_period == b.atom_period
            && a.atom_min.to_bits() == b.atom_min.to_bits()
    }

    /// The degree-parametric constructor must not perturb the Mandelbrot path:
    /// `new_degree(.., 2)` (which dispatches through the `degree == 2` branch of the
    /// trait `sample`) must produce **bit-for-bit** identical `PixelSample`s to the
    /// prior `new(..)` across every trap shape, interior + exterior, and a range of
    /// `maxiter`. This is the byte-identity gate for the whole degree change.
    #[test]
    fn degree2_dispatch_is_byte_identical() {
        for trap in traps() {
            for &maxiter in &[1u32, 2, 7, 50, 300, 2000] {
                let base = F64Backend::new(maxiter, 1e6, trap);
                let deg2 = F64Backend::new_degree(maxiter, 1e6, trap, 2);
                let n = 60;
                for iy in 0..n {
                    let ci = -1.3 + 2.6 * (iy as f64 + 0.5) / n as f64;
                    for ix in 0..n {
                        let cr = -2.2 + 3.0 * (ix as f64 + 0.5) / n as f64;
                        let c = Complex::new(cr, ci);
                        let z = Complex::new(0.0, 0.0);
                        assert!(
                            sample_eq(&base.sample(c, z), &deg2.sample(c, z)),
                            "degree-2 dispatch mismatch shape={:?} maxiter={maxiter} c=({cr},{ci})",
                            trap.shape,
                        );
                    }
                }
            }
        }
    }

    /// The multibrot degrees must actually iterate `z^d + c` and produce a
    /// non-trivial set (some escaping, some bounded), with finite smooth values on
    /// escape and the origin bounded for every degree. Not a byte gate — a
    /// "renders correctly" smoke test that the degree is live.
    #[test]
    fn multibrot_degrees_are_live() {
        let trap = traps()[0];
        for d in [3u32, 4, 5] {
            let bk = F64Backend::new_degree(400, 1e6, trap, d);
            // Origin is in every multibrot set (0 → 0^d + 0 = 0 forever).
            let s0 = bk.sample(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            assert!(!s0.escaped, "degree {d}: origin should be bounded");
            let (mut esc, mut bounded) = (0usize, 0usize);
            let n = 48;
            for iy in 0..n {
                let ci = -1.9 + 3.8 * (iy as f64 + 0.5) / n as f64;
                for ix in 0..n {
                    let cr = -1.9 + 3.8 * (ix as f64 + 0.5) / n as f64;
                    let s = bk.sample(Complex::new(cr, ci), Complex::new(0.0, 0.0));
                    if s.escaped {
                        esc += 1;
                        assert!(s.smooth_iter.is_finite(), "degree {d}: non-finite smooth");
                    } else {
                        bounded += 1;
                    }
                }
            }
            assert!(esc > 0 && bounded > 0, "degree {d}: esc={esc} bounded={bounded}");
        }
    }

    /// Julia-multibrot degrees must iterate `z^d + c` dynamically (`z₀ = pixel`,
    /// fixed param) and produce a non-trivial set, and the degree-2 arm must stay
    /// byte-identical to `new` (`new_degree(.., 2)` == `new`, trivially, but assert
    /// the dispatch produces the same bits). Smooth values finite on escape.
    #[test]
    fn julia_multibrot_degrees_are_live() {
        let trap = traps()[0];
        let param = Complex::new(-0.8, 0.156);
        // degree-2 dispatch equals `new` bit-for-bit.
        let base = JuliaBackend::new(param, 400, 1e6, trap);
        let deg2 = JuliaBackend::new_degree(param, 400, 1e6, trap, 2);
        for iy in 0..40 {
            let zi = -1.5 + 3.0 * (iy as f64 + 0.5) / 40.0;
            for ix in 0..40 {
                let zr = -1.5 + 3.0 * (ix as f64 + 0.5) / 40.0;
                let z = Complex::new(zr, zi);
                assert!(sample_eq(&base.sample(z, z), &deg2.sample(z, z)), "julia deg-2 dispatch mismatch");
            }
        }
        // degrees 3/4/5 are live (some escape, some bounded, finite smooth). Use
        // c = 0 (pure z^d): the filled Julia set is the closed unit disk, so a window
        // spanning it guarantees both classes appear at every degree (a generic c can
        // have a tiny/empty set at these degrees, which would make the test brittle).
        for d in [3u32, 4, 5] {
            let bk = JuliaBackend::new_degree(Complex::new(0.0, 0.0), 400, 1e6, trap, d);
            let (mut esc, mut bounded) = (0usize, 0usize);
            for iy in 0..48 {
                let zi = -1.3 + 2.6 * (iy as f64 + 0.5) / 48.0;
                for ix in 0..48 {
                    let zr = -1.3 + 2.6 * (ix as f64 + 0.5) / 48.0;
                    let s = bk.sample(Complex::new(zr, zi), Complex::new(0.0, 0.0));
                    if s.escaped {
                        esc += 1;
                        assert!(s.smooth_iter.is_finite(), "julia degree {d}: non-finite smooth");
                    } else {
                        bounded += 1;
                    }
                }
            }
            assert!(esc > 0 && bounded > 0, "julia degree {d}: esc={esc} bounded={bounded}");
        }
    }

    /// Phoenix must iterate its two-state recurrence and produce a non-trivial set
    /// with finite smooth values on escape. Ushiki `c=0.5667, p=-0.5` is used, whose
    /// interior is well-populated at base scale.
    #[test]
    fn phoenix_kernel_is_live() {
        let trap = traps()[0];
        let bk = PhoenixBackend::new(
            Complex::new(0.5667, 0.0),
            Complex::new(-0.5, 0.0),
            500,
            1e6,
            trap,
        );
        let (mut esc, mut bounded) = (0usize, 0usize);
        let n = 64;
        for iy in 0..n {
            let zi = -1.8 + 3.6 * (iy as f64 + 0.5) / n as f64;
            for ix in 0..n {
                let zr = -1.8 + 3.6 * (ix as f64 + 0.5) / n as f64;
                let s = bk.sample(Complex::new(zr, zi), Complex::new(0.0, 0.0));
                if s.escaped {
                    esc += 1;
                    assert!(s.smooth_iter.is_finite(), "phoenix: non-finite smooth");
                } else {
                    bounded += 1;
                }
            }
        }
        assert!(esc > 0 && bounded > 0, "phoenix: esc={esc} bounded={bounded}");
    }

    /// All three trap-phase strategies — GATED (production), DEFER, and EVERY
    /// (the pre-change baseline) — must produce **bit-for-bit** identical
    /// `PixelSample`s across every trap shape, interior + exterior pixels, and a
    /// range of `maxiter`. This is the byte-identical gate for the optimization:
    /// `atan2` is a pure function of its input bits, so changing *when* it runs
    /// (and storing the phase of the same trap-minimizing iteration) changes
    /// timing, not output.
    #[test]
    fn phase_strategies_are_byte_identical() {
        let eq = |a: &PixelSample, b: &PixelSample| {
            a.escaped == b.escaped
                && a.smooth_iter.to_bits() == b.smooth_iter.to_bits()
                && a.de.to_bits() == b.de.to_bits()
                && a.trap_min.to_bits() == b.trap_min.to_bits()
                && a.trap_phase.to_bits() == b.trap_phase.to_bits()
                && a.atom_period == b.atom_period
                && a.atom_min.to_bits() == b.atom_min.to_bits()
        };
        for trap in traps() {
            for &maxiter in &[1u32, 2, 7, 50, 300, 2000] {
                let b = F64Backend::new(maxiter, 1e6, trap);
                let n = 80;
                for iy in 0..n {
                    let ci = -1.3 + 2.6 * (iy as f64 + 0.5) / n as f64;
                    for ix in 0..n {
                        let cr = -2.2 + 3.0 * (ix as f64 + 0.5) / n as f64;
                        let c = Complex::new(cr, ci);
                        let every = b.sample_flags::<true, true, true, PHASE_EVERY>(c);
                        let gated = b.sample_flags::<true, true, true, PHASE_GATED>(c);
                        let defer = b.sample_flags::<true, true, true, PHASE_DEFER>(c);
                        assert!(
                            eq(&every, &gated) && eq(&every, &defer),
                            "phase strategy mismatch shape={:?} maxiter={maxiter} c=({cr},{ci}): \
                             every.phase={} gated.phase={} defer.phase={}",
                            trap.shape, every.trap_phase, gated.trap_phase, defer.trap_phase
                        );
                    }
                }
            }
        }
    }
}
