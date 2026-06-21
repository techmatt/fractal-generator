//! `navigate` — deterministic feature navigation (atom domains + Newton nuclei).
//!
//! Replaces `descend`'s pixel-greedy scoring with navigation toward *features*:
//! minibrot nuclei and their embedded-Julia neighborhoods. Each level:
//!  1. render the Mandelbrot panel (atom channels come free from the loop),
//!  2. read **atom-domain candidates** from the buffer (Primitive 1),
//!  3. **Newton-refine** each to its true nucleus in BigFloat (Primitive 2),
//!  4. **size-estimate** each minibrot (Primitive 3),
//!  5. rank by a normalized interest score, pick the best,
//!  6. **adaptively** re-frame at the nucleus, width = `|size|·frame_multiple`.
//!
//! The zoom is minibrot-driven, not a fixed factor — each minibrot is framed at
//! its natural scale, so the path plunges deep fast (periods grow, sizes shrink
//! super-exponentially). That is the point of the diagnostic: how deep before it
//! breaks, and does it stay richer than the greedy baseline.
//!
//! The three primitives are `pub` and unit-tested in isolation (see `tests`
//! below) so a wrong size estimate is caught in seconds, not after a long strip.
//! The filmstrip / Julia / JSON machinery is shared with `descend` via
//! [`crate::probe`].

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use astro_float::{BigFloat, RoundingMode};
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::Trap;
use crate::cli::NavigateArgs;
use crate::font;
use crate::hp;
use crate::palette_io::load_palette;
use crate::probe::{self, SplitMix64};
use crate::render::SampleBuffer;

/// Rounding mode for the Newton BigFloat arithmetic. The nucleus *coordinate* is
/// the output we keep, so unlike the reference orbit (which projects to f64 and
/// can use `None`) we round correctly here.
const RM: RoundingMode = RoundingMode::ToEven;

/// Frame width below which f64 deltas underflow (the v1 perturbation cap) — a
/// hard early-stop floor for the navigation depth.
const MIN_WIDTH: f64 = 1e-200;

/// Radius (output px) of the busyness window used to rank candidates.
const BUSY_R: i32 = 3;

// ===========================================================================
// Complex BigFloat scratch type (Newton works in arbitrary precision)
// ===========================================================================

/// A complex number with arbitrary-precision real/imaginary parts. Private; the
/// public primitives take/return `BigFloat` pairs so callers never touch it.
#[derive(Clone)]
struct CBig {
    re: BigFloat,
    im: BigFloat,
}

impl CBig {
    fn zero(p: usize) -> Self {
        CBig {
            re: BigFloat::from_f64(0.0, p),
            im: BigFloat::from_f64(0.0, p),
        }
    }
}

fn cadd(a: &CBig, b: &CBig, p: usize) -> CBig {
    CBig {
        re: a.re.add(&b.re, p, RM),
        im: a.im.add(&b.im, p, RM),
    }
}

fn csub(a: &CBig, b: &CBig, p: usize) -> CBig {
    CBig {
        re: a.re.sub(&b.re, p, RM),
        im: a.im.sub(&b.im, p, RM),
    }
}

/// `(ac − bd) + (ad + bc)i`.
fn cmul(a: &CBig, b: &CBig, p: usize) -> CBig {
    let ac = a.re.mul(&b.re, p, RM);
    let bd = a.im.mul(&b.im, p, RM);
    let ad = a.re.mul(&b.im, p, RM);
    let bc = a.im.mul(&b.re, p, RM);
    CBig {
        re: ac.sub(&bd, p, RM),
        im: ad.add(&bc, p, RM),
    }
}

/// `(re² − im²) + 2·re·im·i` — a cheaper square than `cmul(a, a)`.
fn csqr(a: &CBig, p: usize) -> CBig {
    let re2 = a.re.mul(&a.re, p, RM);
    let im2 = a.im.mul(&a.im, p, RM);
    let reim = a.re.mul(&a.im, p, RM);
    CBig {
        re: re2.sub(&im2, p, RM),
        im: reim.add(&reim, p, RM),
    }
}

/// `|a|² = re² + im²` (real).
fn cabs2(a: &CBig, p: usize) -> BigFloat {
    let re2 = a.re.mul(&a.re, p, RM);
    let im2 = a.im.mul(&a.im, p, RM);
    re2.add(&im2, p, RM)
}

/// `a / b = a·conj(b) / |b|²`.
fn cdiv(a: &CBig, b: &CBig, p: usize) -> CBig {
    let denom = cabs2(b, p);
    let num_re = a.re.mul(&b.re, p, RM).add(&a.im.mul(&b.im, p, RM), p, RM);
    let num_im = a.im.mul(&b.re, p, RM).sub(&a.re.mul(&b.im, p, RM), p, RM);
    CBig {
        re: num_re.div(&denom, p, RM),
        im: num_im.div(&denom, p, RM),
    }
}

// ===========================================================================
// Primitive 2 — Newton nucleus refinement
// ===========================================================================

/// A refined minibrot nucleus (period `period`) and its convergence diagnostics.
pub struct Nucleus {
    pub re: BigFloat,
    pub im: BigFloat,
    pub period: u32,
    /// `|z_p|²` at the converged `c` (≈ 0 at a true nucleus).
    pub final_z2: f64,
    /// `|z_p|²` at the initial guess (the shrink check compares against this).
    pub init_z2: f64,
    pub newton_iters: u32,
}

/// Newton's method on `z_p(c) = 0`: refine `guess` to the true period-`p`
/// nucleus, in BigFloat at `prec` bits. Returns `None` if it fails to converge
/// or `|z_p|` does not shrink (a wrong-period or spurious candidate).
///
/// Convergence is `|Δc| < width·1e-6` (the step falls well below a pixel). The
/// shrink guard rejects candidates whose `|z_p|` did not drop by ≥100× — a real
/// nucleus drives `z_p → 0` regardless of the minibrot's apparent size, so this
/// is scale-robust where an absolute `|z_p|` threshold would bias toward
/// frame-filling minibrots. The caller additionally rejects out-of-frame nuclei.
pub fn newton_nucleus(
    guess_re: &BigFloat,
    guess_im: &BigFloat,
    period: u32,
    width: f64,
    prec: usize,
) -> Option<Nucleus> {
    if period == 0 {
        return None;
    }
    let p = prec;
    let one = BigFloat::from_f64(1.0, p);
    // Convergence threshold² in BigFloat (f64 would underflow `thr²` past
    // ~1e-150; BigFloat's exponent range does not).
    let thr = BigFloat::from_f64(width * 1e-6, p);
    let thr2 = thr.mul(&thr, p, RM);

    let mut c = CBig {
        re: guess_re.clone(),
        im: guess_im.clone(),
    };
    let mut init_z2 = f64::INFINITY;
    let mut converged = false;
    let mut used = 0u32;

    for it in 0..64u32 {
        used = it + 1;
        // Critical orbit + its c-derivative, tracked together.
        let mut z = CBig::zero(p);
        let mut dz = CBig::zero(p);
        for _k in 1..=period {
            // dz = 2·z·dz + 1 (uses z_{k-1}; update before advancing z).
            let zdz = cmul(&z, &dz, p);
            dz = CBig {
                re: zdz.re.add(&zdz.re, p, RM).add(&one, p, RM),
                im: zdz.im.add(&zdz.im, p, RM),
            };
            // z = z² + c (advance to z_k).
            z = cadd(&csqr(&z, p), &c, p);
        }

        let z2 = hp::to_f64(&cabs2(&z, p));
        if it == 0 {
            init_z2 = z2;
        }

        // delta = z_p / (dz_p); c -= delta.
        let delta = cdiv(&z, &dz, p);
        c = csub(&c, &delta, p);

        let d2 = cabs2(&delta, p);
        if d2.is_inf() || d2.is_nan() {
            return None; // diverged
        }
        // astro-float `cmp` returns the sign of the difference: < 0 ⇒ d2 < thr2.
        if matches!(d2.cmp(&thr2), Some(c) if c < 0) {
            converged = true;
            break;
        }
    }

    if !converged {
        return None;
    }

    // |z_p|² at the converged c (recomputed for an honest final value).
    let final_z2 = hp::to_f64(&cabs2(&eval_orbit(&c, period, p), p));
    if !final_z2.is_finite() || !init_z2.is_finite() {
        return None;
    }
    // Shrink guard: a genuine nucleus drives z_p → 0.
    let shrank = final_z2 < init_z2 * 1e-2 || final_z2 < 1e-10;
    if !shrank {
        return None;
    }

    Some(Nucleus {
        re: c.re,
        im: c.im,
        period,
        final_z2,
        init_z2,
        newton_iters: used,
    })
}

/// Iterate `z_{k+1} = z² + c` for `period` steps from `z_0 = 0`, returning `z_p`.
fn eval_orbit(c: &CBig, period: u32, p: usize) -> CBig {
    let mut z = CBig::zero(p);
    for _ in 0..period {
        z = cadd(&csqr(&z, p), c, p);
    }
    z
}

// ===========================================================================
// Primitive 3 — minibrot size estimate (Munafo / Heiland-Allen)
// ===========================================================================

/// Complex minibrot size estimate at nucleus `c` (period `p`): `mag` = scale,
/// `arg` = orientation. Iterated in f64 (orbit values stay O(1)); `overflow` is
/// set when the derivative product `l` exceeds f64 range (high period — the
/// regime of the deferred floatexp tier), in which case `mag` is not usable.
///
/// Indexing is the Munafo / Heiland-Allen `m_d_size` form
/// `size = 1 / (b · l²)`, with `l = ∏ 2z_k` and `b = 1 + Σ 1/l` — the prompt's
/// starting structure was off by a factor of `l` (it dropped the square and used
/// the wrong `b` seed/order). Validated empirically against the period-3 island
/// (the `size_estimate_period3` test + the framing render): `mag ≈ 0.019`, and a
/// multiple of ~4–8 frames the minibrot centered and correctly scaled.
pub struct SizeEstimate {
    pub mag: f64,
    pub arg: f64,
    pub overflow: bool,
}

pub fn size_estimate(c: Complex<f64>, period: u32) -> SizeEstimate {
    let mut z = Complex::new(0.0, 0.0);
    let mut l = Complex::new(1.0, 0.0); // ∏ 2 z_k
    let mut b = Complex::new(1.0, 0.0); // 1 + Σ 1/l
    for _k in 1..period {
        z = z * z + c;
        l *= z.scale(2.0);
        b += l.inv();
    }
    let size = (b * l * l).inv();
    let overflow = !l.re.is_finite()
        || !l.im.is_finite()
        || !b.re.is_finite()
        || !b.im.is_finite()
        || !size.re.is_finite()
        || !size.im.is_finite();
    SizeEstimate {
        mag: size.norm(),
        arg: size.arg(),
        overflow,
    }
}

// ===========================================================================
// Primitive 1 — atom-domain candidates (from the rendered buffer)
// ===========================================================================

/// A raw nucleus candidate read from the atom channels of a rendered panel.
pub struct Candidate {
    pub col: usize,
    pub row: usize,
    /// Period of the nearby minibrot (the atom-domain period at this pixel).
    pub period: u32,
    /// Closest orbit approach to the origin at this pixel.
    pub atom_min: f64,
    /// Pixel offset of the candidate from the frame center (plane units).
    pub dc_re: f64,
    pub dc_im: f64,
    /// Normalized surrounding busyness (std-dev of smooth-iter / maxiter).
    pub busyness: f64,
}

/// Per-output-pixel atom aggregate.
struct AtomPix {
    amin: f64,
    period: u32,
    smooth: f64,
    escaped: bool,
}

/// Read atom-domain candidates from a rendered Mandelbrot buffer: local minima
/// of `atom_min`, deduplicated to the best representative per distinct period,
/// filtered to periods in `[2, maxiter]` and away from the frame edge.
pub fn atom_candidates(
    buf: &SampleBuffer,
    panel_w: u32,
    panel_h: u32,
    width: f64,
    maxiter: u32,
) -> Vec<Candidate> {
    let w = panel_w as usize;
    let h = panel_h as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;

    // Aggregate subpixels: take the subpixel of smallest atom_min (nearest a
    // nucleus) per output pixel; mean smooth over escaped subpixels for busyness.
    let mut pix: Vec<AtomPix> = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let mut amin = f64::INFINITY;
            let mut period = 0u32;
            let mut esc = 0usize;
            let mut sm = 0.0f64;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.atom_min < amin {
                        amin = px.atom_min;
                        period = px.atom_period;
                    }
                    if px.escaped {
                        esc += 1;
                        sm += px.smooth_iter;
                    }
                }
            }
            let escaped = esc * 2 >= s * s;
            let smooth = if esc > 0 { sm / esc as f64 } else { 0.0 };
            pix.push(AtomPix {
                amin,
                period,
                smooth,
                escaped,
            });
        }
    }

    let fw = width;
    let fh = width * (panel_h as f64 / panel_w as f64);
    let margin = BUSY_R.max(2);
    let inv_maxiter = 1.0 / maxiter.max(1) as f64;

    // (period -> best candidate so far) dedup by distinct period.
    let mut best: std::collections::HashMap<u32, Candidate> = std::collections::HashMap::new();
    for row in (margin as usize)..(h - margin as usize) {
        for col in (margin as usize)..(w - margin as usize) {
            let idx = row * w + col;
            let a = pix[idx].amin;
            let period = pix[idx].period;
            if !a.is_finite() || period < 2 || period > maxiter {
                continue;
            }
            // 3×3 strict-ish local minimum of atom_min.
            let mut is_min = true;
            'nb: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nidx = (row as i32 + dy) as usize * w + (col as i32 + dx) as usize;
                    if pix[nidx].amin < a {
                        is_min = false;
                        break 'nb;
                    }
                }
            }
            if !is_min {
                continue;
            }

            // Normalized busyness in a BUSY_R window (std of smooth over escaped).
            let mut vals: Vec<f64> = Vec::new();
            for dy in -BUSY_R..=BUSY_R {
                for dx in -BUSY_R..=BUSY_R {
                    let nidx = (row as i32 + dy) as usize * w + (col as i32 + dx) as usize;
                    if pix[nidx].escaped {
                        vals.push(pix[nidx].smooth);
                    }
                }
            }
            let busyness = stddev(&vals) * inv_maxiter;

            let dc_re = ((col as f64 + 0.5) / panel_w as f64 - 0.5) * fw;
            let dc_im = (0.5 - (row as f64 + 0.5) / panel_h as f64) * fh;

            let cand = Candidate {
                col,
                row,
                period,
                atom_min: a,
                dc_re,
                dc_im,
                busyness,
            };
            best.entry(period)
                .and_modify(|c| {
                    if a < c.atom_min {
                        *c = Candidate {
                            col,
                            row,
                            period,
                            atom_min: a,
                            dc_re,
                            dc_im,
                            busyness,
                        };
                    }
                })
                .or_insert(cand);
        }
    }

    best.into_values().collect()
}

/// Population standard deviation (empty → 0).
fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// Hermite smoothstep clamped to `[0,1]`.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ===========================================================================
// Candidate ranking (Prompt-5 score fixes folded in)
// ===========================================================================

/// A fully-resolved candidate: atom detection + Newton + size estimate + score.
struct Ranked {
    period: u32,
    nucleus_re: BigFloat,
    nucleus_im: BigFloat,
    size_mag: f64,
    size_arg: f64,
    /// Nucleus offset from the frame center (plane units), for the footprint
    /// circle and the in-frame check.
    nuc_dc_re: f64,
    nuc_dc_im: f64,
    busyness: f64,
    final_z2: f64,
    newton_iters: u32,
    score: f64,
}

/// Interest score for a resolved candidate. The Prompt-5 fixes:
///  - **busyness is normalized** (divided by maxiter at detection) so it is O(1)
///    and commensurable with the bounded terms below.
///  - prefer a navigable **period band** (penalize too-low / absurdly-high),
///  - a **frame-able** size (a sane zoom from the current depth), and
///  - high surrounding busyness (the embedded-Julia decoration around the
///    minibrot); centrality is a mild tie-breaker.
fn score_candidate(
    busyness: f64,
    period: u32,
    size_mag: f64,
    frame_multiple: f64,
    width: f64,
    nuc_dc_re: f64,
    nuc_dc_im: f64,
    panel_w: u32,
    panel_h: u32,
) -> f64 {
    let next_width = size_mag * frame_multiple;
    let zoom = width / next_width;
    // Frame-able: must zoom IN, finite, and not plunge past the v1 floor in one
    // jump. Penalize absurd single-step zooms (> ~1e12).
    let framable = if !next_width.is_finite() || next_width <= MIN_WIDTH || zoom <= 1.0 {
        0.0
    } else {
        1.0 - smoothstep(12.0, 16.0, zoom.log10())
    };
    // Period band: drop the degenerate low end; soft-cap very high periods.
    let pf = period as f64;
    let period_band = smoothstep(1.5, 3.0, pf) * (1.0 - smoothstep(20_000.0, 60_000.0, pf));
    // Centrality (mild): closer to the frame center scores a touch higher.
    let half_w = width * 0.5;
    let half_h = width * (panel_h as f64 / panel_w as f64) * 0.5;
    let rel = ((nuc_dc_re / half_w).powi(2) + (nuc_dc_im / half_h).powi(2)).sqrt();
    let centrality = (1.0 - 0.4 * rel.min(1.0)).max(0.0);

    (0.05 + busyness) * framable * period_band * centrality
}

// ===========================================================================
// navigate subcommand
// ===========================================================================

/// One level's logged record.
struct LevelLog {
    level: u32,
    center_re: String,
    center_im: String,
    frame_width: f64,
    magnification: f64,
    maxiter: u32,
    backend: &'static str,
    glitch_count: u64,
    n_candidates: usize,
    period: u32,
    nucleus_re: String,
    nucleus_im: String,
    size_mag: f64,
    size_arg: f64,
    frame_multiple: f64,
    final_z2: f64,
    newton_iters: u32,
    busyness: f64,
    score: f64,
    /// Chosen nucleus distance from frame center in half-extent units (the
    /// nucleus-stall signal; ~0 ⇒ central cascade).
    rel_offcenter: f64,
    c_f64: Complex<f64>,
    stuck: bool,
    stop_reason: Option<String>,
    mandel_panel: String,
    julia_panel: String,
}

/// Entry point for the `navigate` subcommand.
pub fn run_navigate(args: &NavigateArgs) -> Result<(), String> {
    if args.levels == 0 {
        return Err("--levels must be > 0".into());
    }
    if args.panel_width == 0 {
        return Err("--panel-width must be > 0".into());
    }
    if args.frame_multiple <= 0.0 {
        return Err("--frame-multiple must be > 0".into());
    }

    let panel_w = args.panel_width;
    let panel_h = ((panel_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample.max(1);

    let palette = load_palette(
        &args.palette.palette,
        args.palette.palette_entry.as_deref(),
        args.palette.palette_reverse,
    )?;
    let params = probe::color_params(&args.shade);
    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };

    let (start_re, start_im) = args.resolved_start_center()?;
    let mut width = args.start_width;
    // Center carried in high precision; precision grows as we descend (sized off
    // the *next* width each level via the Newton re-refine).
    let init_prec = hp::prec_bits(panel_w, width) + 96;
    let mut center_re = hp::parse_decimal(&start_re, init_prec)?;
    let mut center_im = hp::parse_decimal(&start_im, init_prec)?;

    let strip_path = Path::new(&args.output);
    let panels_dir = probe::panels_dir_for(strip_path);
    fs::create_dir_all(&panels_dir)
        .map_err(|e| format!("failed to create {}: {e}", panels_dir.display()))?;

    let mut rng = SplitMix64(args.seed);
    let mut logs: Vec<LevelLog> = Vec::with_capacity(args.levels as usize);
    let mut mandel_panels: Vec<RgbImage> = Vec::with_capacity(args.levels as usize);
    let mut julia_panels: Vec<RgbImage> = Vec::with_capacity(args.levels as usize);

    // Stuck guard history: chosen period and the nucleus's off-center fraction
    // per level (the two stall signals — see `is_stuck`).
    let mut period_hist: Vec<u32> = Vec::new();
    let mut move_hist: Vec<f64> = Vec::new();

    print_table_header();

    for level in 0..args.levels {
        let mag = args.start_width / width;
        let maxiter = (args.maxiter_base + args.per_decade * mag.log10())
            .round()
            .max(1.0) as u32;

        // Early-stop guards keyed on the *current* frame (the "how deep before it
        // breaks" signal).
        if width < MIN_WIDTH {
            push_stop(&mut logs, level, &center_re, &center_im, width, mag, maxiter,
                "width below f64-delta floor (1e-200)")?;
            break;
        }
        if maxiter > args.maxiter_ceiling {
            push_stop(&mut logs, level, &center_re, &center_im, width, mag, maxiter,
                &format!("maxiter {maxiter} exceeds ceiling {}", args.maxiter_ceiling))?;
            break;
        }

        let prec = hp::prec_bits(panel_w, width) + 32;
        let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

        let panel = probe::render_mandel_panel(
            &center_re, &center_im, center_f64, width, panel_w, panel_h, ss,
            maxiter, args.bailout, prec, trap, args.backend,
        );
        let buf = panel.buf;
        let backend_name = panel.backend_name;
        let spacing = panel.spacing;

        // Primitive 1: atom-domain candidates.
        let cands = atom_candidates(&buf, panel_w, panel_h, width, maxiter);
        let n_candidates = cands.len();

        // Primitives 2+3: Newton-refine + size each candidate; rank.
        let half_w = width * 0.5;
        let half_h = width * (panel_h as f64 / panel_w as f64) * 0.5;
        let mut ranked: Vec<Ranked> = Vec::new();
        for c in &cands {
            let guess_re = center_re.add(&BigFloat::from_f64(c.dc_re, prec), prec, RM);
            let guess_im = center_im.add(&BigFloat::from_f64(c.dc_im, prec), prec, RM);
            let Some(nuc) = newton_nucleus(&guess_re, &guess_im, c.period, width, prec) else {
                continue;
            };
            // In-frame check: the converged nucleus must stay inside the panel.
            let nuc_dc_re = hp::to_f64(&nuc.re.sub(&center_re, prec, RM));
            let nuc_dc_im = hp::to_f64(&nuc.im.sub(&center_im, prec, RM));
            if nuc_dc_re.abs() > half_w || nuc_dc_im.abs() > half_h {
                continue;
            }
            let nuc_f64 = Complex::new(hp::to_f64(&nuc.re), hp::to_f64(&nuc.im));
            let size = size_estimate(nuc_f64, c.period);
            if size.overflow || !(size.mag > 0.0) {
                continue;
            }
            let score = score_candidate(
                c.busyness, c.period, size.mag, args.frame_multiple, width,
                nuc_dc_re, nuc_dc_im, panel_w, panel_h,
            );
            ranked.push(Ranked {
                period: c.period,
                nucleus_re: nuc.re,
                nucleus_im: nuc.im,
                size_mag: size.mag,
                size_arg: size.arg,
                nuc_dc_re,
                nuc_dc_im,
                busyness: c.busyness,
                final_z2: nuc.final_z2,
                newton_iters: nuc.newton_iters,
                score,
            });
        }

        // No valid feature — early stop.
        if ranked.is_empty() {
            push_stop(&mut logs, level, &center_re, &center_im, width, mag, maxiter,
                "no valid nucleus candidate (atom/Newton/size all rejected)")?;
            // Still render the panel so the strip shows where it died.
            let mandel_img = finalize_dead_panel(
                &buf, panel_w, panel_h, ss, &palette, &params, spacing,
                level, mag, maxiter, backend_name,
            );
            let julia_img = probe::render_julia_panel(
                center_f64, args.julia_maxiter, args.bailout, trap,
                panel_w, panel_h, ss, &palette, &params,
            );
            save_panels(&panels_dir, level, &mandel_img, &julia_img)?;
            mandel_panels.push(mandel_img);
            julia_panels.push(julia_img);
            break;
        }

        // Rank; pick the best, tie-breaking among near-equal tops with the seed.
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        let best_score = ranked[0].score;
        let tie_hi = best_score * 0.98;
        let n_tie = ranked.iter().take_while(|r| r.score >= tie_hi).count().max(1);
        let chosen = &ranked[rng.below(n_tie)];

        let next_width = chosen.size_mag * args.frame_multiple;
        let zoom = width / next_width;

        // Re-refine the chosen nucleus at the precision the *next* (deeper) frame
        // needs, so the carried center stays exact as we descend.
        let next_prec = hp::prec_bits(panel_w, next_width.max(MIN_WIDTH)) + 96;
        let refined = newton_nucleus(
            &chosen.nucleus_re, &chosen.nucleus_im, chosen.period, next_width.max(MIN_WIDTH), next_prec,
        );
        let (nucleus_re, nucleus_im) = match refined {
            Some(n) => (n.re, n.im),
            None => (
                chosen.nucleus_re.clone(),
                chosen.nucleus_im.clone(),
            ),
        };
        let c_f64 = Complex::new(hp::to_f64(&nucleus_re), hp::to_f64(&nucleus_im));

        // Stuck guards: period stall + nucleus-pinned-to-center stall.
        let rel_offcenter = ((chosen.nuc_dc_re / half_w).powi(2)
            + (chosen.nuc_dc_im / half_h).powi(2))
        .sqrt();
        period_hist.push(chosen.period);
        move_hist.push(rel_offcenter);
        let stuck = is_stuck(&period_hist, &move_hist);

        // ---- compose the row ----
        let mut mandel_img = render_panel(
            &buf, panel_w, panel_h, ss, &palette, &params, spacing,
        );
        // Footprint circle at the nucleus pixel; radius = next-frame px / 2.
        let circle_x = (chosen.nuc_dc_re / width + 0.5) * panel_w as f64;
        let circle_y = (0.5 - chosen.nuc_dc_im / (width * (panel_h as f64 / panel_w as f64)))
            * panel_h as f64;
        let circle_r = (panel_w as f64 / (2.0 * zoom)).max(2.0);
        probe::draw_circle(&mut mandel_img, circle_x, circle_y, circle_r);

        let label = format!(
            "L{:02} M={:.1e} IT={} {} P={} SZ={:.1e} X{:.0} S={:.2}{}",
            level, mag, maxiter, backend_name, chosen.period, chosen.size_mag,
            args.frame_multiple, chosen.score, if stuck { " STUCK" } else { "" },
        )
        .to_uppercase();
        font::draw_text(&mut mandel_img, &label, 2, 2, 2, Rgb([240, 240, 240]), true);

        let julia_img = probe::render_julia_panel(
            c_f64, args.julia_maxiter, args.bailout, trap,
            panel_w, panel_h, ss, &palette, &params,
        );
        save_panels(&panels_dir, level, &mandel_img, &julia_img)?;

        let mandel_rel = format!("mandel_{level:02}.png");
        let julia_rel = format!("julia_{level:02}.png");

        print_table_row(
            level, width, mag, maxiter, backend_name, buf.glitched_pixels,
            n_candidates, chosen.period, chosen.size_mag, zoom, chosen.score,
            rel_offcenter, stuck,
        );

        logs.push(LevelLog {
            level,
            center_re: hp::to_decimal_string(&center_re)?,
            center_im: hp::to_decimal_string(&center_im)?,
            frame_width: width,
            magnification: mag,
            maxiter,
            backend: backend_name,
            glitch_count: buf.glitched_pixels,
            n_candidates,
            period: chosen.period,
            nucleus_re: hp::to_decimal_string(&nucleus_re)?,
            nucleus_im: hp::to_decimal_string(&nucleus_im)?,
            size_mag: chosen.size_mag,
            size_arg: chosen.size_arg,
            frame_multiple: args.frame_multiple,
            final_z2: chosen.final_z2,
            newton_iters: chosen.newton_iters,
            busyness: chosen.busyness,
            score: chosen.score,
            rel_offcenter,
            c_f64,
            stuck,
            stop_reason: None,
            mandel_panel: probe::path_str(&panels_dir.join(&mandel_rel)),
            julia_panel: probe::path_str(&panels_dir.join(&julia_rel)),
        });
        mandel_panels.push(mandel_img);
        julia_panels.push(julia_img);

        // Score-collapse early-stop: the best candidate is not frame-able (its
        // size exceeds the frame ⇒ a zoom-*out*, or no navigable period). Single
        // path can only oscillate from here — stop at the honest break point.
        if chosen.score <= 1e-6 {
            if let Some(last) = logs.last_mut() {
                last.stop_reason =
                    Some("best candidate not frame-able (score collapsed to 0)".into());
            }
            eprintln!("L{level:02}: early stop — best candidate score collapsed to 0");
            break;
        }
        // Period cap early-stop (after logging the level that reached it).
        if chosen.period > args.period_cap {
            if let Some(last) = logs.last_mut() {
                last.stop_reason =
                    Some(format!("period {} exceeds cap {}", chosen.period, args.period_cap));
            }
            break;
        }
        if stuck {
            if let Some(last) = logs.last_mut() {
                last.stop_reason = Some("nucleus/period stalled (stuck guard)".into());
            }
            eprintln!("L{level:02}: early stop — stuck guard (self-similar cascade)");
            break;
        }

        // Adaptive descend: re-frame at the nucleus, minibrot-driven width. The
        // nucleus was re-refined at `next_prec` (sized for this deeper frame), so
        // the carried center stays exact as the precision requirement grows.
        center_re = nucleus_re;
        center_im = nucleus_im;
        width = next_width;
    }

    if mandel_panels.is_empty() {
        return Err("navigate produced no panels (first level had no candidate)".into());
    }

    let strip = probe::compose_strip(&mandel_panels, &julia_panels, panel_w, panel_h);
    crate::ensure_parent_dir(strip_path)?;
    strip
        .save(strip_path)
        .map_err(|e| format!("failed to write {}: {e}", strip_path.display()))?;

    let json = build_json(&logs, &probe::path_str(strip_path));
    crate::ensure_parent_dir(&args.json)?;
    fs::write(&args.json, json).map_err(|e| format!("failed to write {}: {e}", args.json))?;

    eprintln!(
        "wrote {} ({} levels), per-level panels in {}/, log {}",
        args.output,
        logs.len(),
        panels_dir.display(),
        args.json,
    );
    Ok(())
}

/// The path is stuck when either:
///  - **period stalls** — the latest period is no greater than three levels back
///    (cycling self-similar nuclei), or
///  - **nucleus stalls** — the chosen nucleus has sat near the frame center
///    (`rel_offcenter` small) for the last three levels. This catches the
///    period-*doubling* cascade that the period check misses: period keeps
///    rising while `c` accumulates at a Feigenbaum-type limit point and the
///    embedded Julia field saturates (Δ-nucleus-relative-to-width stalls).
///
/// `rel_offcenter` is the chosen nucleus's distance from the frame center in
/// half-extent units; a genuine jump to an off-center sibling minibrot is
/// ~0.3–0.8, a dive into the central nested copy is ~0.
fn is_stuck(periods: &[u32], moves: &[f64]) -> bool {
    let n = periods.len();
    let period_stall = n >= 4 && periods[n - 1] <= periods[n - 4];
    let nucleus_stall = moves.len() >= 3 && moves[moves.len() - 3..].iter().all(|&r| r < 0.08);
    period_stall || nucleus_stall
}

/// Shade a Mandelbrot panel (no annotation).
fn render_panel(
    buf: &SampleBuffer,
    panel_w: u32,
    panel_h: u32,
    ss: u32,
    palette: &crate::palette::Palette,
    params: &crate::coloring::ColorParams,
    spacing: f64,
) -> RgbImage {
    crate::render::shade_and_downsample(&buf.samples, panel_w, panel_h, ss, palette, params, spacing)
}

/// Shade + label a panel for a dead (no-candidate) level.
#[allow(clippy::too_many_arguments)]
fn finalize_dead_panel(
    buf: &SampleBuffer,
    panel_w: u32,
    panel_h: u32,
    ss: u32,
    palette: &crate::palette::Palette,
    params: &crate::coloring::ColorParams,
    spacing: f64,
    level: u32,
    mag: f64,
    maxiter: u32,
    backend_name: &str,
) -> RgbImage {
    let mut img = render_panel(buf, panel_w, panel_h, ss, palette, params, spacing);
    let label = format!("L{level:02} M={mag:.1e} IT={maxiter} {backend_name} NO-FEATURE").to_uppercase();
    font::draw_text(&mut img, &label, 2, 2, 2, Rgb([255, 120, 120]), true);
    img
}

fn save_panels(
    dir: &Path,
    level: u32,
    mandel: &RgbImage,
    julia: &RgbImage,
) -> Result<(), String> {
    let mp = dir.join(format!("mandel_{level:02}.png"));
    let jp = dir.join(format!("julia_{level:02}.png"));
    mandel
        .save(&mp)
        .map_err(|e| format!("failed to write {}: {e}", mp.display()))?;
    julia
        .save(&jp)
        .map_err(|e| format!("failed to write {}: {e}", jp.display()))?;
    Ok(())
}

/// Record an early-stop level (no panel data beyond center/width) into the log.
#[allow(clippy::too_many_arguments)]
fn push_stop(
    logs: &mut Vec<LevelLog>,
    level: u32,
    center_re: &BigFloat,
    center_im: &BigFloat,
    width: f64,
    mag: f64,
    maxiter: u32,
    reason: &str,
) -> Result<(), String> {
    eprintln!("L{level:02}: early stop — {reason}");
    logs.push(LevelLog {
        level,
        center_re: hp::to_decimal_string(center_re)?,
        center_im: hp::to_decimal_string(center_im)?,
        frame_width: width,
        magnification: mag,
        maxiter,
        backend: "-",
        glitch_count: 0,
        n_candidates: 0,
        period: 0,
        nucleus_re: String::new(),
        nucleus_im: String::new(),
        size_mag: f64::NAN,
        size_arg: f64::NAN,
        frame_multiple: f64::NAN,
        final_z2: f64::NAN,
        newton_iters: 0,
        busyness: f64::NAN,
        score: f64::NAN,
        rel_offcenter: f64::NAN,
        c_f64: Complex::new(f64::NAN, f64::NAN),
        stuck: false,
        stop_reason: Some(reason.to_string()),
        mandel_panel: String::new(),
        julia_panel: String::new(),
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// stdout table
// ---------------------------------------------------------------------------

fn print_table_header() {
    println!(
        "{:>3}  {:>10}  {:>9}  {:>6}  {:>4}  {:>5}  {:>5}  {:>7}  {:>10}  {:>9}  {:>6}  {:>5}  {:>5}",
        "lvl", "width", "mag", "maxit", "bknd", "gltch", "cand", "period", "size", "zoom",
        "score", "roff", "stuck",
    );
}

#[allow(clippy::too_many_arguments)]
fn print_table_row(
    level: u32,
    width: f64,
    mag: f64,
    maxiter: u32,
    backend: &str,
    glitch: u64,
    n_cand: usize,
    period: u32,
    size: f64,
    zoom: f64,
    score: f64,
    rel_offcenter: f64,
    stuck: bool,
) {
    println!(
        "{:>3}  {:>10.3e}  {:>9.2e}  {:>6}  {:>4}  {:>5}  {:>5}  {:>7}  {:>10.3e}  {:>9.2e}  {:>6.3}  {:>5.2}  {:>5}",
        level, width, mag, maxiter, backend, glitch, n_cand, period, size, zoom, score,
        rel_offcenter, if stuck { "YES" } else { "no" },
    );
}

// ---------------------------------------------------------------------------
// JSON log (extends the descend schema with the navigation fields)
// ---------------------------------------------------------------------------

fn build_json(logs: &[LevelLog], strip: &str) -> String {
    use probe::{jf, js};
    let mut s = String::from("[\n");
    for (i, lv) in logs.iter().enumerate() {
        s.push_str("  {\n");
        s.push_str(&format!("    \"level\": {},\n", lv.level));
        s.push_str(&format!(
            "    \"center\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&lv.center_re),
            js(&lv.center_im)
        ));
        s.push_str(&format!("    \"frame_width\": {},\n", jf(lv.frame_width)));
        s.push_str(&format!("    \"magnification\": {},\n", jf(lv.magnification)));
        s.push_str(&format!("    \"maxiter\": {},\n", lv.maxiter));
        s.push_str(&format!("    \"backend\": {},\n", js(lv.backend)));
        s.push_str(&format!("    \"n_candidates\": {},\n", lv.n_candidates));
        s.push_str(&format!("    \"period\": {},\n", lv.period));
        s.push_str(&format!(
            "    \"nucleus\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&lv.nucleus_re),
            js(&lv.nucleus_im)
        ));
        s.push_str(&format!(
            "    \"size_estimate\": {{ \"mag\": {}, \"arg\": {} }},\n",
            jf(lv.size_mag),
            jf(lv.size_arg)
        ));
        s.push_str(&format!("    \"frame_multiple\": {},\n", jf(lv.frame_multiple)));
        s.push_str(&format!("    \"final_z2\": {},\n", jf(lv.final_z2)));
        s.push_str(&format!("    \"newton_iters\": {},\n", lv.newton_iters));
        s.push_str(&format!("    \"busyness\": {},\n", jf(lv.busyness)));
        s.push_str(&format!("    \"score\": {},\n", jf(lv.score)));
        s.push_str(&format!("    \"rel_offcenter\": {},\n", jf(lv.rel_offcenter)));
        s.push_str(&format!(
            "    \"c_f64\": {{ \"re\": {}, \"im\": {} }},\n",
            jf(lv.c_f64.re),
            jf(lv.c_f64.im)
        ));
        s.push_str(&format!("    \"glitch_count\": {},\n", lv.glitch_count));
        s.push_str(&format!("    \"stuck\": {},\n", lv.stuck));
        match &lv.stop_reason {
            Some(r) => s.push_str(&format!("    \"stop_reason\": {},\n", js(r))),
            None => s.push_str("    \"stop_reason\": null,\n"),
        }
        s.push_str(&format!("    \"mandel_panel\": {},\n", js(&lv.mandel_panel)));
        s.push_str(&format!("    \"julia_panel\": {},\n", js(&lv.julia_panel)));
        s.push_str(&format!("    \"strip\": {}\n", js(strip)));
        s.push_str("  }");
        if i + 1 < logs.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("]\n");
    s
}

// ===========================================================================
// Primitive validation tests (run in isolation — `cargo test --test ...` or
// `cargo test navigate`). These prove the primitives before any filmstrip.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{F64Backend, FractalBackend, TrapShape};
    use crate::render::{self, Frame};

    const TRAP: Trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    /// Render a small Mandelbrot frame with the f64 backend.
    fn render_frame(center: Complex<f64>, width: f64, w: u32, h: u32, maxiter: u32) -> SampleBuffer {
        let backend = F64Backend::new(maxiter, 1e6, TRAP);
        let frame = Frame {
            center,
            frame_width: width,
            out_width: w,
            out_height: h,
        };
        render::iterate_samples(&backend, &frame, 2)
    }

    /// Primitive 1: the prominent period-3 real-axis island (≈ −1.7549) must
    /// register atom-domain period 3, with the `atom_min` minimum localized on it.
    #[test]
    fn atom_period3_island() {
        let center = Complex::new(-1.7549, 0.0);
        let width = 0.05;
        let (w, h) = (320u32, 200u32);
        let maxiter = 2000;
        let buf = render_frame(center, width, w, h, maxiter);
        let cands = atom_candidates(&buf, w, h, width, maxiter);
        assert!(!cands.is_empty(), "no atom candidates near the period-3 island");
        // A period-3 candidate must exist and sit near the frame center.
        let p3 = cands
            .iter()
            .find(|c| c.period == 3)
            .expect("no period-3 candidate found");
        let dc = (p3.dc_re.powi(2) + p3.dc_im.powi(2)).sqrt();
        assert!(
            dc < width * 0.5,
            "period-3 candidate not near center: dc={dc:e}"
        );
        println!(
            "atom: {} candidates; period-3 at dc=({:.3e},{:.3e}), atom_min={:.3e}, periods={:?}",
            cands.len(),
            p3.dc_re,
            p3.dc_im,
            p3.atom_min,
            {
                let mut ps: Vec<u32> = cands.iter().map(|c| c.period).collect();
                ps.sort();
                ps
            }
        );
    }

    /// Primitive 2: Newton must converge from a guess near the period-3 island to
    /// its nucleus, driving `|z_p|` to ~0.
    #[test]
    fn newton_period3() {
        let prec = 128;
        let gre = hp::parse_decimal("-1.7549", prec).unwrap();
        let gim = hp::parse_decimal("0.0", prec).unwrap();
        let nuc = newton_nucleus(&gre, &gim, 3, 0.05, prec).expect("period-3 Newton failed");
        let re = hp::to_f64(&nuc.re);
        let im = hp::to_f64(&nuc.im);
        println!(
            "newton p3: nucleus=({re:.15}, {im:.3e}), |z_p|^2: {:.3e} -> {:.3e}, iters={}",
            nuc.init_z2, nuc.final_z2, nuc.newton_iters
        );
        // Known real period-3 nucleus ≈ -1.7548776662...
        assert!((re - (-1.754877_7)).abs() < 1e-4, "nucleus re off: {re}");
        assert!(im.abs() < 1e-9, "nucleus im not on real axis: {im}");
        assert!(nuc.final_z2 < 1e-12, "|z_p|^2 did not vanish: {:.3e}", nuc.final_z2);
    }

    /// Primitive 2, generality: refine a higher-period candidate discovered in a
    /// render (self-consistent — no hard-coded coordinate).
    #[test]
    fn newton_higher_period() {
        // A frame on the real axis west of the main cardioid hosts several
        // islands of period > 3.
        let center = Complex::new(-1.4, 0.0);
        let width = 0.2;
        let (w, h) = (400u32, 250u32);
        let maxiter = 4000;
        let buf = render_frame(center, width, w, h, maxiter);
        let cands = atom_candidates(&buf, w, h, width, maxiter);
        let hp_cand = cands
            .iter()
            .filter(|c| c.period > 3)
            .min_by_key(|c| c.period)
            .expect("no period>3 candidate");
        let prec = 128;
        let cre = hp::parse_decimal("-1.4", prec).unwrap();
        let cim = hp::parse_decimal("0.0", prec).unwrap();
        let gre = cre.add(&BigFloat::from_f64(hp_cand.dc_re, prec), prec, RM);
        let gim = cim.add(&BigFloat::from_f64(hp_cand.dc_im, prec), prec, RM);
        let nuc = newton_nucleus(&gre, &gim, hp_cand.period, width, prec)
            .unwrap_or_else(|| panic!("Newton failed for period {}", hp_cand.period));
        println!(
            "newton p{}: nucleus=({:.12},{:.3e}), |z_p|^2 {:.3e} -> {:.3e}, iters={}",
            hp_cand.period,
            hp::to_f64(&nuc.re),
            hp::to_f64(&nuc.im),
            nuc.init_z2,
            nuc.final_z2,
            nuc.newton_iters
        );
        assert!(nuc.final_z2 < nuc.init_z2 * 1e-2 || nuc.final_z2 < 1e-10);
    }

    /// Primitive 3: the size estimate at the period-3 nucleus must be finite,
    /// positive, and frame the island at a sane multiple. We sanity-check the
    /// magnitude against the island's known real-axis extent.
    #[test]
    fn size_estimate_period3() {
        let prec = 128;
        let gre = hp::parse_decimal("-1.7549", prec).unwrap();
        let gim = hp::parse_decimal("0.0", prec).unwrap();
        let nuc = newton_nucleus(&gre, &gim, 3, 0.05, prec).unwrap();
        let nuc_f64 = Complex::new(hp::to_f64(&nuc.re), hp::to_f64(&nuc.im));
        let size = size_estimate(nuc_f64, 3);
        println!(
            "size p3: mag={:.6e}, arg={:.4} rad, overflow={}",
            size.mag, size.arg, size.overflow
        );
        assert!(!size.overflow, "period-3 size overflowed");
        assert!(size.mag.is_finite() && size.mag > 0.0, "bad size mag");
        // m_d_size at the period-3 nucleus ≈ 0.019 (validated by the framing
        // render: width = mag·{4..8} centers and scales the minibrot).
        assert!(
            (size.mag - 0.019).abs() < 0.004,
            "period-3 size magnitude off reference (~0.019): {:.4e}",
            size.mag
        );
    }

    /// The atom channel must agree between f64 and perturbation backends at a
    /// shallow location (the navigation channel is separable/precision-agnostic).
    #[test]
    fn atom_channel_backends_agree() {
        use crate::backend::PerturbationBackend;
        let prec = hp::prec_bits(300, 0.01);
        let cre = hp::parse_decimal("-0.745", prec).unwrap();
        let cim = hp::parse_decimal("0.113", prec).unwrap();
        let center = Complex::new(hp::to_f64(&cre), hp::to_f64(&cim));
        let maxiter = 300;
        let f64b = F64Backend::new(maxiter, 1e6, TRAP);
        let pb = PerturbationBackend::new(&cre, &cim, maxiter, 1e6, prec, TRAP);
        let width = 0.01;
        let fh = width * (200.0 / 300.0);
        let mut mismatches = 0;
        let mut checked = 0;
        for row in 0..200u32 {
            let dc_im = (0.5 - (row as f64 + 0.5) / 200.0) * fh;
            for col in 0..300u32 {
                let dc_re = ((col as f64 + 0.5) / 300.0 - 0.5) * width;
                let dc = Complex::new(dc_re, dc_im);
                let a = f64b.sample(center + dc, dc);
                let b = pb.sample(center + dc, dc);
                checked += 1;
                if a.atom_period != b.atom_period {
                    mismatches += 1;
                }
            }
        }
        println!("atom channel: {mismatches}/{checked} period mismatches (f64 vs perturb)");
        // Allow a tiny number of boundary ties; the bulk must match exactly.
        assert!(
            (mismatches as f64) < 0.01 * checked as f64,
            "too many atom_period mismatches: {mismatches}/{checked}"
        );
    }
}
