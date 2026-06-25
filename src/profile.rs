//! Throwaway `profile` subcommand — f64 render-path profiling (Prompt
//! profile_f64_render). **Diagnosis only: measure and report, change no render
//! behavior, pick no optimization.** Nothing here feeds coloring or selection;
//! the real render path (`render.rs`, `backend.rs`) is untouched. The output is
//! a profile that decides whether a follow-up optimization prompt is worth
//! writing.
//!
//! Three measurements over the **f64** Mandelbrot path (perturbation is parked):
//!  1. **Phase breakdown** — wall-clock per phase (setup / iterate /
//!     shade+downsample / encode / write) of one representative render, ranked.
//!     Iteration cost is isolated from recolor cost because the contact sheet
//!     iterates once and recolors many — they amortize differently.
//!  2. **Escape-time histogram** — distribution of per-pixel iteration counts,
//!     the fraction of pixels that run to `maxiter` (interior), mean/total
//!     iterations, and the iteration work split interior-vs-escaper. This is the
//!     number that decides whether an interior-skip (cardioid/bulb/periodicity)
//!     optimization would buy anything — measured, not acted on.
//!  3. **Thread-scaling sweep** — fixed workload, the iteration pass re-timed in
//!     a fresh rayon pool per thread count; wall-clock, speedup, and parallel
//!     efficiency vs. 1 thread, against the logical/physical core counts.
//!
//! **Release build only** — a debug profile is meaningless. The runner asserts
//! the auto-selected backend stayed f64 (the cheap regime this prompt scopes).

use std::io::Cursor;
use std::time::Instant;

use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{
    F64Backend, FractalBackend, PixelSample, Trap, TrapShape, PHASE_DEFER, PHASE_EVERY, PHASE_GATED,
};
use clap::Args;
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::hp;
use crate::palette::builtin;
use crate::probe::PERTURB_SPACING;
use crate::render::{self, Frame, SampleBuffer};

/// Min + median of a set of timings (seconds). The kernel is deterministic, so
/// the min is the cleanest estimate and the median–min gap is system noise.
fn min_median(mut xs: Vec<f64>) -> (f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = xs[0];
    let median = xs[xs.len() / 2];
    (min, median)
}

/// Time `f` `runs` times, returning every elapsed-seconds sample.
fn time_runs(runs: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut out = Vec::with_capacity(runs);
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_secs_f64());
    }
    out
}

/// Escape iteration count for one pixel, mirroring `F64Backend`'s loop ordering
/// (interior test takes precedence at `n == maxiter`, so a pixel that would
/// escape exactly at `maxiter` is counted interior, exactly as the real kernel
/// classifies it). Returns `maxiter` for interior / never-escape pixels.
///
/// This is a *stripped* kernel (no trap / dz / atom bookkeeping): it exists only
/// to recover the per-pixel iteration count for the histogram, which the cached
/// `PixelSample` does not expose for interior pixels. The distribution of `n` is
/// independent of the per-iteration work the real kernel does.
#[inline]
fn count_iters(cr: f64, ci: f64, maxiter: u32, bailout2: f64) -> u32 {
    let mut zr = 0.0f64;
    let mut zi = 0.0f64;
    for n in 1..=maxiter {
        let nzr = zr * zr - zi * zi + cr;
        let nzi = 2.0 * zr * zi + ci;
        zr = nzr;
        zi = nzi;
        if zr * zr + zi * zi > bailout2 {
            return n;
        }
    }
    maxiter
}

/// Escape-time statistics over the supersampled grid.
struct EscapeStats {
    total: u64,
    /// Subpixels that reached `maxiter` (interior / never-escape).
    interior: u64,
    /// Σ n over all subpixels.
    total_iters: u128,
    /// Σ n over interior subpixels (= interior · maxiter).
    interior_iters: u128,
    /// log2 buckets of `n`: bucket `b` holds counts with `n` in `[2^b, 2^(b+1))`,
    /// the final implicit bucket being the interior count (`n == maxiter`).
    log2_buckets: Vec<u64>,
}

impl EscapeStats {
    fn interior_frac(&self) -> f64 {
        self.interior as f64 / self.total.max(1) as f64
    }
    fn mean_iters(&self) -> f64 {
        self.total_iters as f64 / self.total.max(1) as f64
    }
    /// Fraction of total iteration *work* spent in interior pixels.
    fn interior_work_frac(&self) -> f64 {
        if self.total_iters == 0 {
            0.0
        } else {
            self.interior_iters as f64 / self.total_iters as f64
        }
    }
}

/// Run the stripped counting kernel over the supersampled grid and accumulate
/// the escape-time distribution. Parallel over rows, matching the real pass.
fn escape_stats(frame: &Frame, ss: u32, maxiter: u32, bailout: f64) -> EscapeStats {
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
    let bailout2 = bailout * bailout;
    let nbuckets = (32 - (maxiter.max(1)).leading_zeros()) as usize + 1; // ⌈log2⌉+1

    // (total, interior, total_iters, interior_iters, buckets) reduced over rows.
    let init = || (0u64, 0u64, 0u128, 0u128, vec![0u64; nbuckets]);
    let (total, interior, total_iters, interior_iters, log2_buckets) = (0..sub_h)
        .into_par_iter()
        .fold(init, |mut acc, srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let cr = center.re + dc_re;
                let ci = center.im + dc_im;
                let n = count_iters(cr, ci, maxiter, bailout2);
                acc.0 += 1;
                acc.2 += n as u128;
                if n >= maxiter {
                    acc.1 += 1;
                    acc.3 += n as u128;
                }
                let b = (32 - (n.max(1)).leading_zeros()) as usize - 1;
                acc.4[b.min(nbuckets - 1)] += 1;
            }
            acc
        })
        .reduce(init, |mut a, b| {
            a.0 += b.0;
            a.1 += b.1;
            a.2 += b.2;
            a.3 += b.3;
            for (x, y) in a.4.iter_mut().zip(b.4.iter()) {
                *x += y;
            }
            a
        });

    EscapeStats {
        total,
        interior,
        total_iters,
        interior_iters,
        log2_buckets,
    }
}

// ===========================================================================
// Cardioid / period-2-bulb interior coverage (prompt
// profile_cardioid_bulb_coverage). **Diagnosis only.** The two algebraic tests
// are pure functions of `c` and are *sufficient* conditions for set membership:
// any pixel passing either is guaranteed interior (the iteration never escapes).
// They catch only the main cardioid and the left period-2 disc — satellite bulbs
// are out of scope by construction, and the residue (uncaught interior pixels) is
// exactly the satellite-bulb question made concrete.
// ===========================================================================

/// Main-cardioid membership test (`c = x + iy`). Sufficient: true ⇒ interior.
/// `q = (x − 1/4)² + y²`; inside iff `q·(q + (x − 1/4)) ≤ y²/4`.
#[inline]
fn in_main_cardioid(x: f64, y: f64) -> bool {
    let xm = x - 0.25;
    let q = xm * xm + y * y;
    q * (q + xm) <= 0.25 * y * y
}

/// Period-2-bulb membership test. Sufficient: true ⇒ interior.
/// Inside iff `(x + 1)² + y² ≤ 1/16`.
#[inline]
fn in_period2_bulb(x: f64, y: f64) -> bool {
    let xp = x + 1.0;
    xp * xp + y * y <= 0.0625
}

/// Per-subpixel coverage classification (mask category encoding).
const ESCAPER: u8 = 0; // not caught, escaped
const CAUGHT_INTERIOR: u8 = 1; // caught by cardioid/bulb AND interior (the win)
const UNCAUGHT_INTERIOR: u8 = 2; // interior but missed by both tests (the residue)
const FALSE_POSITIVE: u8 = 3; // caught by a test but actually escaped — MUST be 0

/// Cross-tabulation of the algebraic tests against the actual iteration verdict,
/// over the supersampled grid (same grid as `escape_stats`, so the fractions
/// compose with the escape-time numbers).
struct CoverageCounts {
    total: u64,
    interior: u64,
    caught_interior: u64,
    /// Algebraically flagged but escaped per iteration. A mis-implemented test;
    /// must be 0 (the tests are *sufficient* conditions for membership).
    false_positive: u64,
    uncaught_interior: u64,
}

impl CoverageCounts {
    /// Fraction of interior pixels the free tests catch. Because every interior
    /// pixel costs full `maxiter`, this equals the fraction of interior *work*
    /// the free test could eliminate.
    fn catchable_frac(&self) -> f64 {
        self.caught_interior as f64 / self.interior.max(1) as f64
    }
}

/// Run the stripped counting kernel once over the supersampled grid, evaluating
/// the cardioid/bulb tests in the **same pass** (pure `c` predicates, computed
/// regardless of the iteration result so the correctness guard can catch any
/// algebraically-flagged escaper). Returns the per-subpixel mask (row-major, SS
/// resolution) plus the cross-tab counts derived from it.
fn coverage_pass(frame: &Frame, ss: u32, maxiter: u32, bailout: f64) -> (Vec<u8>, CoverageCounts) {
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
    let bailout2 = bailout * bailout;

    let mask: Vec<u8> = (0..sub_h)
        .into_par_iter()
        .flat_map_iter(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            (0..sub_w).map(move |scol| {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let cr = center.re + dc_re;
                let ci = center.im + dc_im;
                let caught = in_main_cardioid(cr, ci) || in_period2_bulb(cr, ci);
                // Always iterate — the guard depends on classifying caught pixels too.
                let interior = count_iters(cr, ci, maxiter, bailout2) >= maxiter;
                match (caught, interior) {
                    (true, true) => CAUGHT_INTERIOR,
                    (true, false) => FALSE_POSITIVE,
                    (false, true) => UNCAUGHT_INTERIOR,
                    (false, false) => ESCAPER,
                }
            })
        })
        .collect();

    let mut c = CoverageCounts {
        total: mask.len() as u64,
        interior: 0,
        caught_interior: 0,
        false_positive: 0,
        uncaught_interior: 0,
    };
    for &m in &mask {
        match m {
            CAUGHT_INTERIOR => {
                c.caught_interior += 1;
                c.interior += 1;
            }
            UNCAUGHT_INTERIOR => {
                c.uncaught_interior += 1;
                c.interior += 1;
            }
            FALSE_POSITIVE => c.false_positive += 1,
            _ => {}
        }
    }
    (mask, c)
}

/// Downsample the SS-resolution categorical mask to an output-resolution RGB
/// image by per-block majority vote (ties broken toward the residue so it is
/// never visually under-represented). Three distinct colors — caught-interior /
/// uncaught-interior / escaper — plus an alarm color for any false positive.
fn render_coverage_mask(mask: &[u8], w: u32, h: u32, ss: u32) -> RgbImage {
    let s = ss.max(1);
    let sub_w = w * s;
    let color = |cat: u8| match cat {
        CAUGHT_INTERIOR => Rgb([40u8, 170, 70]),    // green — the win
        UNCAUGHT_INTERIOR => Rgb([210u8, 60, 50]),  // red — the residue
        FALSE_POSITIVE => Rgb([255u8, 0, 255]),     // magenta — alarm (should be absent)
        _ => Rgb([12u8, 12, 16]),                   // near-black — escaper
    };
    let mut img = RgbImage::new(w, h);
    for oy in 0..h {
        for ox in 0..w {
            let mut tally = [0u32; 4];
            for sy in 0..s {
                let srow = oy * s + sy;
                let base = (srow as usize) * (sub_w as usize) + (ox * s) as usize;
                for sx in 0..s as usize {
                    tally[mask[base + sx] as usize] += 1;
                }
            }
            // Priority on ties: false-positive > residue > caught > escaper, so
            // any glitch and the residue character survive the downsample.
            let best = tally.iter().copied().max().unwrap();
            let cat = if tally[FALSE_POSITIVE as usize] == best {
                FALSE_POSITIVE
            } else if tally[UNCAUGHT_INTERIOR as usize] == best {
                UNCAUGHT_INTERIOR
            } else if tally[CAUGHT_INTERIOR as usize] == best {
                CAUGHT_INTERIOR
            } else {
                ESCAPER
            };
            img.put_pixel(ox, oy, color(cat));
        }
    }
    img
}

// ===========================================================================
// Inner-loop cost decomposition (prompt profile_inner_loop_deadweight).
// **Diagnosis only.** Sizes the per-iteration bookkeeping the mint/search
// colorer never reads, by compile-time ablation of the F64 kernel's three
// bookkeeping channels (TRAP / ATOM / DE). The real path is the all-on
// `sample_flags::<true,true,true>` (== the original kernel); the combos below
// are extra monomorphizations the optimizer dead-code-eliminates, giving the
// true "never computed this" cost rather than a runtime-branch approximation.
// ===========================================================================

/// One ablation combo's timing.
struct AblationRow {
    label: &'static str,
    trap: bool,
    atom: bool,
    de: bool,
    wall_min: f64,
    wall_med: f64,
    /// Wall saving vs the all-on baseline (`baseline_min − this_min`), seconds.
    delta_s: f64,
    /// Same as a fraction of the all-on baseline.
    delta_frac: f64,
}

/// Iterate the const-generic F64 kernel over the supersampled grid with the
/// given channel flags, returning the full sample buffer. Every live field is
/// materialized into the `Vec`, so the optimizer cannot drop a channel whose
/// result is written out — the const flag is then the *only* thing gating that
/// channel's computation, which is exactly what makes the per-combo delta the
/// channel's cost. Mirrors `render::iterate_samples`' grid + row parallelism so
/// the timing is comparable to the real iterate pass.
fn iterate_ablation<const TRAP: bool, const ATOM: bool, const DE: bool, const PHASE: u8>(
    backend: &F64Backend,
    frame: &Frame,
    ss: u32,
) -> Vec<PixelSample> {
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

    let rows: Vec<Vec<PixelSample>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let mut row = Vec::with_capacity(sub_w as usize);
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let c = Complex::new(center.re + dc_re, center.im + dc_im);
                row.push(backend.sample_flags::<TRAP, ATOM, DE, PHASE>(c));
            }
            row
        })
        .collect();
    let mut samples = Vec::with_capacity(sub_w as usize * sub_h as usize);
    for r in rows {
        samples.extend_from_slice(&r);
    }
    samples
}

/// One (channel × mint-treatment) byte-equality result for the deadness proof.
struct ProofRow {
    /// Ablated channel (`TRAP` / `ATOM` / `DE`).
    channel: &'static str,
    /// Mint color treatment the comparison was shaded under.
    treatment: &'static str,
    /// `true` ⟺ the channel-off PNG bytes equal the all-on PNG bytes under this
    /// treatment. Unread ⟹ identical; any diff ⟹ the channel is live here.
    identical: bool,
}

/// PNG-encode an image to bytes in memory (the comparison unit for the deadness
/// proof — identical pixels ⟹ identical PNG bytes for this deterministic codec).
fn encode_png(img: &RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("png encode");
    bytes
}

/// The mint/search render matrix: every color treatment the production sweep
/// (`wallpaper.rs`) ever applies to one iterated buffer. The *union* of channels
/// these read is the live set; anything outside it is strippable. `de_shade`
/// strength mirrors `wallpaper::DE_SHADE_STRENGTH`.
const MINT_TREATMENTS: [(&str, ColorChannel, InteriorMode, Option<f64>); 3] = [
    ("smooth", ColorChannel::Smooth, InteriorMode::Black, None),
    ("trap", ColorChannel::Trap, InteriorMode::Trap, None),
    ("smooth_de", ColorChannel::Smooth, InteriorMode::Black, Some(1.0)),
];

/// One thread-count row of the strong-scaling sweep.
struct ScaleRow {
    threads: usize,
    wall_min: f64,
    speedup: f64,
    efficiency: f64,
}

pub fn run_profile(args: &ProfileArgs) -> Result<(), String> {
    if args.frame_width <= 0.0 {
        return Err("--frame-width must be > 0".into());
    }
    if args.width == 0 || args.supersample == 0 {
        return Err("--width and --supersample must be > 0".into());
    }

    let w = args.width;
    let h = ((w as f64) * 2.0 / 3.0).round().max(1.0) as u32; // 3:2
    let ss = args.supersample;
    let sub_w = w * ss;
    let sub_h = h * ss;
    let n_sub = sub_w as u64 * sub_h as u64;

    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    let prec = hp::prec_bits(w, args.frame_width);
    let center_re = hp::parse_decimal(&args.center_re, prec)?;
    let center_im = hp::parse_decimal(&args.center_im, prec)?;
    let center = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

    let frame = Frame {
        center,
        frame_width: args.frame_width,
        out_width: w,
        out_height: h,
    };
    let spacing = frame.pixel_size();
    if spacing <= PERTURB_SPACING {
        return Err(format!(
            "pixel spacing {spacing:.3e} is in the perturbation regime; this profiler is \
             f64-only — use a shallower --frame-width / smaller --width."
        ));
    }

    let palette = builtin(&args.palette, false)
        .ok_or_else(|| format!("unknown built-in palette '{}'", args.palette))?;
    // Representative default shading (smooth/black, matching the render default).
    let params = ColorParams {
        density: 0.025,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    };
    let runs = args.runs.max(1);

    eprintln!(
        "[{}] f64 profile {w}x{h} ss{ss} ({} subpixels), center=({}, {}), width={:.3e}, \
         maxiter={}, spacing={:.3e}, runs={}",
        args.label, n_sub, args.center_re, args.center_im, args.frame_width, args.maxiter,
        spacing, runs,
    );
    eprintln!(
        "cores: {} logical / {} physical (rayon default pool {} threads)",
        num_cpus_logical(),
        physical_hint(),
        rayon::current_num_threads(),
    );

    // ----- Phase: setup (build the f64 backend; trivial, but measured) --------
    let setup = time_runs(runs, || {
        let b = F64Backend::new(args.maxiter, args.bailout, trap);
        std::hint::black_box(&b);
    });
    let f64_backend = F64Backend::new(args.maxiter, args.bailout, trap);
    // Guard: auto-selection would have picked f64 here (asserted via spacing above).
    let backend: &dyn FractalBackend = &f64_backend;

    // ----- Phase: iterate (the only stage that touches the backend) -----------
    let mut buf: Option<SampleBuffer> = None;
    let iterate = time_runs(runs, || {
        buf = Some(render::iterate_samples(backend, &frame, ss));
    });
    let buf = buf.unwrap();

    // ----- Phase: shade + downsample (the recolor cost; pure over the buffer) -
    let mut img = None;
    let shade = time_runs(runs, || {
        img = Some(render::shade_and_downsample(
            &buf.samples, w, h, ss, &palette, &params, spacing,
        ));
    });
    let img = img.unwrap();

    // ----- Phase: encode (RgbImage -> PNG bytes, in memory) -------------------
    let mut png_bytes: Vec<u8> = Vec::new();
    let encode = time_runs(runs, || {
        let mut bytes = Vec::new();
        img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("png encode");
        png_bytes = bytes;
    });

    // ----- Phase: write (encoded bytes -> disk) -------------------------------
    let out_png = format!("{}/profile_{}.png", args.out_dir.trim_end_matches('/'), args.label);
    crate::ensure_parent_dir(&out_png)?;
    let write = time_runs(runs, || {
        std::fs::write(&out_png, &png_bytes).expect("write png");
    });

    // ----- Escape-time histogram ----------------------------------------------
    let es = escape_stats(&frame, ss, args.maxiter, args.bailout);

    // ----- Cardioid / period-2-bulb interior coverage -------------------------
    let (cov_mask, cov) = coverage_pass(&frame, ss, args.maxiter, args.bailout);
    let mask_img = render_coverage_mask(&cov_mask, w, h, ss);
    let mask_png = format!(
        "{}/coverage_mask_{}.png",
        args.out_dir.trim_end_matches('/'),
        args.label
    );
    crate::ensure_parent_dir(&mask_png)?;
    mask_img
        .save(&mask_png)
        .map_err(|e| format!("failed to write {mask_png}: {e}"))?;
    // Projected upper-bound wall-clock saving from the free test alone: the share
    // of interior pixels it catches × the share of total iteration work interior
    // pixels represent.
    let proj_saving = cov.catchable_frac() * es.interior_work_frac();

    // ----- Strong-scaling sweep over the iteration pass -----------------------
    let thread_list = args.resolved_threads()?;
    let mut sweep: Vec<ScaleRow> = Vec::new();
    let mut base_wall = None;
    for &nt in &thread_list {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .map_err(|e| format!("failed to build {nt}-thread pool: {e}"))?;
        let samples = time_runs(runs, || {
            pool.install(|| {
                let b = render::iterate_samples(backend, &frame, ss);
                std::hint::black_box(&b);
            });
        });
        let (wall_min, _) = min_median(samples);
        if base_wall.is_none() {
            base_wall = Some(wall_min);
        }
        let b1 = base_wall.unwrap();
        let speedup = b1 / wall_min;
        sweep.push(ScaleRow {
            threads: nt,
            wall_min,
            speedup,
            efficiency: speedup / nt as f64,
        });
    }

    // ----- Inner-loop cost decomposition (channel ablation) -------------------
    // Time the all-on kernel and each single-channel-off kernel on the rayon
    // default pool (same as the real render). Per-channel cost = baseline −
    // that-channel-off; the all-off combo is the pure-core floor.
    let time_buf = |f: &dyn Fn() -> Vec<PixelSample>| -> (f64, f64) {
        let samples = time_runs(runs, || {
            let b = f();
            std::hint::black_box(&b);
        });
        min_median(samples)
    };
    // Baseline is the PRODUCTION kernel: all channels on, trap phase GATED. The
    // phase-strategy sub-ablation below isolates the atan2 against EVERY/DEFER.
    let (base_min, base_med) = time_buf(&|| iterate_ablation::<true, true, true, PHASE_GATED>(&f64_backend, &frame, ss));
    let combos: [(&'static str, bool, bool, bool, f64, f64); 5] = [
        ("all-on (TRAP+ATOM+DE)", true, true, true, base_min, base_med),
        {
            let (m, d) = time_buf(&|| iterate_ablation::<false, true, true, PHASE_GATED>(&f64_backend, &frame, ss));
            ("TRAP off", false, true, true, m, d)
        },
        {
            let (m, d) = time_buf(&|| iterate_ablation::<true, false, true, PHASE_GATED>(&f64_backend, &frame, ss));
            ("ATOM off  (dead in mint)", true, false, true, m, d)
        },
        {
            let (m, d) = time_buf(&|| iterate_ablation::<true, true, false, PHASE_GATED>(&f64_backend, &frame, ss));
            ("DE off", true, true, false, m, d)
        },
        {
            let (m, d) = time_buf(&|| iterate_ablation::<false, false, false, PHASE_GATED>(&f64_backend, &frame, ss));
            ("all-off (pure core floor)", false, false, false, m, d)
        },
    ];

    // ----- trap-phase strategy sub-ablation (the ceiling) ---------------------
    // The decomposition above measures TRAP as a whole (distance-min + phase). To
    // attribute the cost to the `atan2` specifically — and to justify the chosen
    // production strategy — time the *same* all-on kernel under all three phase
    // strategies. EVERY (atan2 every iteration) is the pre-change baseline; GATED
    // (atan2 only on a trap-min improvement) is production; DEFER (capture the
    // minimizer, one atan2 post-loop) is the alternative. All keep distance-min
    // tracking, so the deltas isolate where/how often the atan2 runs.
    let (every_min, _) =
        time_buf(&|| iterate_ablation::<true, true, true, PHASE_EVERY>(&f64_backend, &frame, ss));
    let (gated_min, _) =
        time_buf(&|| iterate_ablation::<true, true, true, PHASE_GATED>(&f64_backend, &frame, ss));
    let (defer_min, _) =
        time_buf(&|| iterate_ablation::<true, true, true, PHASE_DEFER>(&f64_backend, &frame, ss));
    // Realized win = pre-change baseline (EVERY) vs production (GATED).
    let atan2_delta_s = every_min - gated_min;
    let atan2_delta_frac = atan2_delta_s / every_min.max(1e-12); // share of the OLD kernel
    let atan2_speedup = every_min / gated_min.max(1e-12);
    let defer_speedup = every_min / defer_min.max(1e-12);
    let ablation: Vec<AblationRow> = combos
        .iter()
        .map(|&(label, trap, atom, de, wall_min, wall_med)| AblationRow {
            label,
            trap,
            atom,
            de,
            wall_min,
            wall_med,
            delta_s: base_min - wall_min,
            delta_frac: (base_min - wall_min) / base_min.max(1e-12),
        })
        .collect();

    // ----- Deadness proof (byte-identical PNG under the mint treatments) ------
    // Build the all-on buffer and one buffer per single-channel ablation, shade
    // each under every mint treatment, and compare PNG bytes against all-on.
    // ATOM should be byte-identical everywhere (dead → strippable); TRAP/DE
    // should DIFFER under the treatments that read them (live → not strippable),
    // which also proves the comparison harness can see a removed live channel.
    let buf_on = iterate_ablation::<true, true, true, PHASE_GATED>(&f64_backend, &frame, ss);
    let buf_trap_off = iterate_ablation::<false, true, true, PHASE_GATED>(&f64_backend, &frame, ss);
    let buf_atom_off = iterate_ablation::<true, false, true, PHASE_GATED>(&f64_backend, &frame, ss);
    let buf_de_off = iterate_ablation::<true, true, false, PHASE_GATED>(&f64_backend, &frame, ss);

    // Byte-identical gate for the phase optimization: the pre-change baseline
    // (EVERY, atan2 every iteration) and the production kernel (GATED, == buf_on)
    // must agree bit-for-bit on every field, and — shaded under the full mint
    // matrix — produce identical PNG bytes. Any mismatch is a correctness bug.
    let buf_baseline = iterate_ablation::<true, true, true, PHASE_EVERY>(&f64_backend, &frame, ss);
    let phase_buf_bit_identical = buf_on.len() == buf_baseline.len()
        && buf_on.iter().zip(&buf_baseline).all(|(a, b)| {
            a.escaped == b.escaped
                && a.smooth_iter.to_bits() == b.smooth_iter.to_bits()
                && a.de.to_bits() == b.de.to_bits()
                && a.trap_min.to_bits() == b.trap_min.to_bits()
                && a.trap_phase.to_bits() == b.trap_phase.to_bits()
                && a.atom_period == b.atom_period
                && a.atom_min.to_bits() == b.atom_min.to_bits()
        });

    // Guard: confirm each ablation actually changed its channel's fields (so an
    // "identical" verdict means "unread", not "ablation was a silent no-op").
    let atom_fields_changed = buf_on
        .iter()
        .zip(&buf_atom_off)
        .filter(|(a, b)| a.atom_period != b.atom_period || a.atom_min.to_bits() != b.atom_min.to_bits())
        .count();
    let trap_fields_changed = buf_on
        .iter()
        .zip(&buf_trap_off)
        .filter(|(a, b)| a.trap_min.to_bits() != b.trap_min.to_bits() || a.trap_phase.to_bits() != b.trap_phase.to_bits())
        .count();
    let de_fields_changed = buf_on
        .iter()
        .zip(&buf_de_off)
        .filter(|(a, b)| a.de.to_bits() != b.de.to_bits())
        .count();

    let shade_bytes = |buf: &[PixelSample], ch: ColorChannel, int: InteriorMode, de: Option<f64>| {
        let p = ColorParams { channel: ch, interior: int, de_shade: de, ..params };
        encode_png(&render::shade_and_downsample(buf, w, h, ss, &palette, &p, spacing))
    };
    let mut proof: Vec<ProofRow> = Vec::new();
    for &(channel, buf) in &[
        ("TRAP", &buf_trap_off),
        ("ATOM", &buf_atom_off),
        ("DE", &buf_de_off),
    ] {
        for &(tname, ch, int, de) in &MINT_TREATMENTS {
            let identical = shade_bytes(&buf_on, ch, int, de) == shade_bytes(buf, ch, int, de);
            proof.push(ProofRow {
                channel,
                treatment: tname,
                identical,
            });
        }
    }

    // Trap-phase PNG gate: production (buf_on, GATED) vs baseline (buf_baseline,
    // EVERY) must yield byte-identical PNGs under every mint treatment (the trap
    // treatment is the one that actually reads the phase).
    let phase_png_identical: Vec<(&'static str, bool)> = MINT_TREATMENTS
        .iter()
        .map(|&(tname, ch, int, de)| {
            (tname, shade_bytes(&buf_on, ch, int, de) == shade_bytes(&buf_baseline, ch, int, de))
        })
        .collect();

    // Free the proof buffers before the report.
    drop((buf_on, buf_trap_off, buf_atom_off, buf_de_off, buf_baseline));

    // ----- Report -------------------------------------------------------------
    let (su_min, su_med) = min_median(setup.clone());
    let (it_min, it_med) = min_median(iterate.clone());
    let (sh_min, sh_med) = min_median(shade.clone());
    let (en_min, en_med) = min_median(encode.clone());
    let (wr_min, wr_med) = min_median(write.clone());
    let phase_total = su_min + it_min + sh_min + en_min + wr_min;

    println!("\n=== PHASE BREAKDOWN [{}]  {w}x{h} ss{ss}, maxiter {} ===", args.label, args.maxiter);
    println!("  phase              min(s)     median(s)    %of one-shot");
    let row = |name: &str, mn: f64, md: f64| {
        println!("  {name:<16} {mn:>10.5}  {md:>10.5}     {:>6.2}%", 100.0 * mn / phase_total);
    };
    row("setup", su_min, su_med);
    row("iterate", it_min, it_med);
    row("shade+downsmpl", sh_min, sh_med);
    row("encode(png)", en_min, en_med);
    row("write(disk)", wr_min, wr_med);
    println!("  {:<16} {phase_total:>10.5}", "one-shot total");
    println!(
        "  iterate:recolor ratio = {:.1}x  (iterate {:.5}s vs shade {:.5}s)",
        it_min / sh_min.max(1e-12),
        it_min,
        sh_min
    );

    println!("\n=== ESCAPE-TIME HISTOGRAM [{}] ===", args.label);
    println!("  subpixels            {}", es.total);
    println!(
        "  interior (==maxiter) {}  ({:.2}% of pixels)",
        es.interior,
        100.0 * es.interior_frac()
    );
    println!("  mean iterations/px   {:.1}", es.mean_iters());
    println!("  total iterations     {}", es.total_iters);
    println!(
        "  interior iter-work   {:.2}%   (escapers {:.2}%)",
        100.0 * es.interior_work_frac(),
        100.0 * (1.0 - es.interior_work_frac())
    );
    println!("  log2(n) buckets [2^b, 2^(b+1)):");
    for (b, &c) in es.log2_buckets.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let lo = 1u64 << b;
        let hi = lo.saturating_mul(2).saturating_sub(1);
        let tag = if (lo..=hi).contains(&(args.maxiter as u64)) {
            "  <- includes maxiter (interior)"
        } else {
            ""
        };
        println!(
            "    n in [{lo:>6}, {hi:>6}]  {c:>10}  ({:>5.2}%){tag}",
            100.0 * c as f64 / es.total.max(1) as f64
        );
    }

    println!("\n=== CARDIOID/BULB INTERIOR COVERAGE [{}] ===", args.label);
    println!("  interior pixels        {}", cov.interior);
    println!(
        "  caught (cardioid|bulb) {}  ({:.2}% of interior  <- catchable fraction)",
        cov.caught_interior,
        100.0 * cov.catchable_frac()
    );
    println!(
        "  uncaught interior      {}  ({:.2}% of interior  <- residue)",
        cov.uncaught_interior,
        100.0 * cov.uncaught_interior as f64 / cov.interior.max(1) as f64
    );
    println!(
        "  false positives        {}  {}",
        cov.false_positive,
        if cov.false_positive == 0 {
            "(OK — tests are sound)"
        } else {
            "(!!! BUG — a sufficient test flagged an escaper)"
        }
    );
    println!(
        "  interior iter-work share {:.2}%  =>  projected upper-bound wall saving {:.2}%",
        100.0 * es.interior_work_frac(),
        100.0 * proj_saving
    );
    println!("  (= catchable {:.2}% x interior-work {:.2}%; free test only, no periodicity)",
        100.0 * cov.catchable_frac(),
        100.0 * es.interior_work_frac());
    println!("  residue mask -> {}", mask_png.replace('\\', "/"));

    if !sweep.is_empty() {
        println!("\n=== THREAD SCALING [{}] (iterate pass, fixed workload) ===", args.label);
        println!("  threads   wall_min(s)   speedup   efficiency");
        for r in &sweep {
            println!(
                "  {:>5}    {:>10.5}   {:>6.2}x   {:>6.1}%",
                r.threads,
                r.wall_min,
                r.speedup,
                100.0 * r.efficiency
            );
        }
        let cpu_time = sweep[0].wall_min; // deterministic kernel: 1-thread wall == total CPU-work
        println!(
            "  total CPU-work (1-thread wall) = {cpu_time:.5}s; parallel efficiency = that / (threads x wall)"
        );
        println!(
            "  logical={} physical={}: gains past {} threads are hyperthread, expect sublinear",
            num_cpus_logical(),
            physical_hint(),
            physical_hint()
        );
    }

    // ----- Channel liveness / deadness proof report --------------------------
    // Per-channel verdict: DEAD ⟺ byte-identical under every mint treatment.
    let channel_dead = |ch: &str| proof.iter().filter(|r| r.channel == ch).all(|r| r.identical);
    println!("\n=== CHANNEL LIVENESS [{}]  (mint matrix: smooth/black, trap/trap, smooth+DE) ===", args.label);
    println!("  channel  smooth   trap     smooth_de   verdict");
    for ch in ["TRAP", "ATOM", "DE"] {
        let cell = |t: &str| {
            let r = proof.iter().find(|r| r.channel == ch && r.treatment == t).unwrap();
            if r.identical { "identical" } else { "DIFFERS  " }
        };
        let verdict = if channel_dead(ch) {
            "DEAD (unread → strippable)"
        } else {
            "LIVE (read → keep)"
        };
        println!(
            "  {ch:<7}  {}  {}  {}   {verdict}",
            cell("smooth"),
            cell("trap"),
            cell("smooth_de")
        );
    }
    println!(
        "  ablation-changed subpixels: TRAP={trap_fields_changed} ATOM={atom_fields_changed} DE={de_fields_changed} \
         (all > 0 ⟹ each ablation truly removed its channel; an 'identical' verdict is then 'unread', not a no-op)"
    );
    println!("  consumers: smooth_iter→Smooth | trap_min/phase→Trap+InteriorMode::Trap | de→De+de_shade | atom_*→navigate.rs only (parked)");

    // ----- Inner-loop cost decomposition report ------------------------------
    println!("\n=== INNER-LOOP COST DECOMPOSITION [{}]  (iterate pass, ss{ss}, {} threads) ===",
        args.label, rayon::current_num_threads());
    println!("  combo                       wall_min(s)  wall_med(s)   Δ vs all-on(s)   Δ%");
    for r in &ablation {
        println!(
            "  {:<26}  {:>10.5}   {:>10.5}    {:>+10.5}   {:>+6.2}%",
            r.label, r.wall_min, r.wall_med, r.delta_s, 100.0 * r.delta_frac
        );
    }
    let atom_row = &ablation[2]; // ATOM off — the only dead channel in the mint config
    let alloff_row = &ablation[4];
    let dead_strippable = atom_row.delta_frac.max(0.0); // mint dead set = {atom}
    println!(
        "\n  per-channel cost (all-on − that-off): TRAP {:+.2}%  ATOM {:+.2}%  DE {:+.2}%",
        100.0 * ablation[1].delta_frac,
        100.0 * ablation[2].delta_frac,
        100.0 * ablation[3].delta_frac
    );
    println!(
        "  pure-core floor (all bookkeeping off): {:.2}% of all-on is core; bookkeeping total ≈ {:.2}%",
        100.0 * alloff_row.wall_min / base_min.max(1e-12),
        100.0 * alloff_row.delta_frac
    );
    println!("\n  BOTTOM LINE (mint config, dead set = {{ATOM}}):");
    if channel_dead("ATOM") {
        println!(
            "  strippable fraction of the inner loop = ATOM-off delta = {:.2}%  ({:.5}s of the {:.5}s iterate pass)",
            100.0 * dead_strippable, atom_row.delta_s, base_min
        );
        println!(
            "  projected sweep saving at N locations ≈ {:.2}% of total iterate time \
             (e.g. {:.3}s saved per 1000 frames of this workload)",
            100.0 * dead_strippable,
            1000.0 * atom_row.delta_s
        );
        if dead_strippable < 0.01 {
            println!("  → under ~1%: a follow-up strip prompt is NOT worth writing. Nothing to get, stop.");
        } else {
            println!("  → a follow-up strip prompt (drop the atom channel from the F64 kernel) is worth writing.");
        }
    } else {
        println!("  ATOM is NOT byte-dead under the mint matrix (see liveness table) — nothing safely strippable; stop.");
    }
    println!("  NOTE: this is ONE workload (interior_frac {:.2}%). Run a second invocation at a \
        decoration-heavy center to bracket — bookkeeping runs on escaper iterations too.",
        100.0 * es.interior_frac());

    // ----- Trap-phase strategy: realized speedup vs ceiling + byte gate -------
    let buf_gate = if phase_buf_bit_identical { "PASS (bit-identical buffer)" } else { "FAIL — buffers differ!" };
    let png_gate_pass = phase_png_identical.iter().all(|&(_, ok)| ok);
    println!("\n=== TRAP-PHASE STRATEGY [{}]  (move the per-iteration atan2 off the hot loop; iterate pass, {} threads) ===",
        args.label, rayon::current_num_threads());
    println!("  EVERY  (atan2 every iter, pre-change baseline)  {every_min:.5}s   1.00x");
    println!("  GATED  (atan2 only on a trap-min improvement)   {gated_min:.5}s   {atan2_speedup:.2}x   <- PRODUCTION");
    println!("  DEFER  (capture minimizer, one atan2 post-loop) {defer_min:.5}s   {defer_speedup:.2}x");
    println!(
        "  realized win (EVERY − GATED) = {atan2_delta_s:+.5}s = {:.2}% of the old kernel  →  {atan2_speedup:.2}x",
        100.0 * atan2_delta_frac
    );
    println!("  (this is the CEILING for moving the atan2: distance-min tracking is kept in all three.)");
    if defer_min > gated_min {
        println!(
            "  note: DEFER is {:.2}x SLOWER than GATED — capturing the minimizing z keeps two extra f64s \
             live across the register-starved hot loop; that spill cost exceeds the few atan2s GATED still pays.",
            defer_min / gated_min.max(1e-12)
        );
    }
    println!(
        "  projected sweep saving (EVERY→GATED) ≈ {:.3}s per 1000 frames of this workload",
        1000.0 * atan2_delta_s
    );
    println!("  byte-identical gate (buffer, GATED vs EVERY): {buf_gate}");
    print!("  byte-identical gate (PNG, mint matrix):");
    for (tname, ok) in &phase_png_identical {
        print!("  {tname}={}", if *ok { "identical" } else { "DIFFERS!" });
    }
    println!();
    if !phase_buf_bit_identical || !png_gate_pass {
        println!("  → !!! GATED IS NOT BYTE-IDENTICAL TO THE BASELINE — a correctness bug, not an optimization.");
    } else {
        println!("  → GATED is byte-identical to the pre-change baseline (output unchanged); the speedup is free.");
    }

    // ----- JSON sidecar -------------------------------------------------------
    let json = build_json(args, w, h, ss, spacing, phase_total,
        (su_min, su_med), (it_min, it_med), (sh_min, sh_med), (en_min, en_med), (wr_min, wr_med),
        &es, &cov, proj_saving, &sweep, &ablation, &proof,
        (trap_fields_changed as u64, atom_fields_changed as u64, de_fields_changed as u64),
        base_min);
    let json_path = format!("{}/profile_{}.json", args.out_dir.trim_end_matches('/'), args.label);
    crate::ensure_parent_dir(&json_path)?;
    std::fs::write(&json_path, json).map_err(|e| format!("failed to write {json_path}: {e}"))?;
    eprintln!("\nwrote {} and {}", json_path.replace('\\', "/"), out_png.replace('\\', "/"));

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_json(
    args: &ProfileArgs,
    w: u32,
    h: u32,
    ss: u32,
    spacing: f64,
    phase_total: f64,
    setup: (f64, f64),
    iterate: (f64, f64),
    shade: (f64, f64),
    encode: (f64, f64),
    write: (f64, f64),
    es: &EscapeStats,
    cov: &CoverageCounts,
    proj_saving: f64,
    sweep: &[ScaleRow],
    ablation: &[AblationRow],
    proof: &[ProofRow],
    fields_changed: (u64, u64, u64),
    base_min: f64,
) -> String {
    let jf = crate::probe::jf;
    let js = crate::probe::js;
    let phase = |name: &str, p: (f64, f64)| {
        format!(
            "    {{ \"phase\": {}, \"min_s\": {}, \"median_s\": {} }}",
            js(name),
            jf(p.0),
            jf(p.1)
        )
    };
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"label\": {},\n", js(&args.label)));
    s.push_str(&format!(
        "  \"center\": {{ \"re\": {}, \"im\": {} }},\n",
        js(&args.center_re),
        js(&args.center_im)
    ));
    s.push_str(&format!("  \"frame_width\": {},\n", jf(args.frame_width)));
    s.push_str(&format!("  \"maxiter\": {},\n", args.maxiter));
    s.push_str(&format!("  \"out_width\": {w}, \"out_height\": {h}, \"supersample\": {ss},\n"));
    s.push_str(&format!("  \"subpixels\": {},\n", es.total));
    s.push_str(&format!("  \"pixel_spacing\": {},\n", jf(spacing)));
    s.push_str(&format!("  \"runs\": {},\n", args.runs.max(1)));
    s.push_str(&format!("  \"cores_logical\": {}, \"cores_physical\": {},\n", num_cpus_logical(), physical_hint()));
    s.push_str("  \"phases\": [\n");
    s.push_str(&phase("setup", setup));
    s.push_str(",\n");
    s.push_str(&phase("iterate", iterate));
    s.push_str(",\n");
    s.push_str(&phase("shade_downsample", shade));
    s.push_str(",\n");
    s.push_str(&phase("encode_png", encode));
    s.push_str(",\n");
    s.push_str(&phase("write_disk", write));
    s.push_str("\n  ],\n");
    s.push_str(&format!("  \"one_shot_total_s\": {},\n", jf(phase_total)));
    s.push_str("  \"escape\": {\n");
    s.push_str(&format!("    \"interior\": {},\n", es.interior));
    s.push_str(&format!("    \"interior_frac\": {},\n", jf(es.interior_frac())));
    s.push_str(&format!("    \"mean_iters\": {},\n", jf(es.mean_iters())));
    s.push_str(&format!("    \"total_iters\": {},\n", es.total_iters));
    s.push_str(&format!("    \"interior_work_frac\": {},\n", jf(es.interior_work_frac())));
    let buckets: Vec<String> = es.log2_buckets.iter().map(|c| c.to_string()).collect();
    s.push_str(&format!("    \"log2_buckets\": [{}]\n", buckets.join(", ")));
    s.push_str("  },\n");
    s.push_str("  \"coverage\": {\n");
    s.push_str(&format!("    \"total\": {},\n", cov.total));
    s.push_str(&format!("    \"interior\": {},\n", cov.interior));
    s.push_str(&format!("    \"caught_interior\": {},\n", cov.caught_interior));
    s.push_str(&format!("    \"uncaught_interior\": {},\n", cov.uncaught_interior));
    s.push_str(&format!("    \"false_positive\": {},\n", cov.false_positive));
    s.push_str(&format!("    \"catchable_frac\": {},\n", jf(cov.catchable_frac())));
    s.push_str(&format!("    \"interior_work_frac\": {},\n", jf(es.interior_work_frac())));
    s.push_str(&format!("    \"projected_wall_saving_frac\": {}\n", jf(proj_saving)));
    s.push_str("  },\n");
    s.push_str("  \"scaling\": [\n");
    let rows: Vec<String> = sweep
        .iter()
        .map(|r| {
            format!(
                "    {{ \"threads\": {}, \"wall_min_s\": {}, \"speedup\": {}, \"efficiency\": {} }}",
                r.threads,
                jf(r.wall_min),
                jf(r.speedup),
                jf(r.efficiency)
            )
        })
        .collect();
    s.push_str(&rows.join(",\n"));
    s.push_str("\n  ],\n");

    // Inner-loop cost decomposition + deadness proof.
    let dead = |ch: &str| proof.iter().filter(|r| r.channel == ch).all(|r| r.identical);
    s.push_str("  \"ablation\": {\n");
    s.push_str(&format!("    \"all_on_min_s\": {},\n", jf(base_min)));
    let abl: Vec<String> = ablation
        .iter()
        .map(|r| {
            format!(
                "    {{ \"combo\": {}, \"trap\": {}, \"atom\": {}, \"de\": {}, \"wall_min_s\": {}, \"wall_med_s\": {}, \"delta_s\": {}, \"delta_frac\": {} }}",
                js(r.label), r.trap, r.atom, r.de, jf(r.wall_min), jf(r.wall_med), jf(r.delta_s), jf(r.delta_frac)
            )
        })
        .collect();
    s.push_str("    \"combos\": [\n");
    s.push_str(&abl.join(",\n"));
    s.push_str("\n    ]\n  },\n");
    s.push_str("  \"liveness\": {\n");
    s.push_str(&format!(
        "    \"fields_changed\": {{ \"trap\": {}, \"atom\": {}, \"de\": {} }},\n",
        fields_changed.0, fields_changed.1, fields_changed.2
    ));
    s.push_str(&format!(
        "    \"verdict\": {{ \"trap\": {}, \"atom\": {}, \"de\": {} }},\n",
        js(if dead("TRAP") { "dead" } else { "live" }),
        js(if dead("ATOM") { "dead" } else { "live" }),
        js(if dead("DE") { "dead" } else { "live" })
    ));
    let pr: Vec<String> = proof
        .iter()
        .map(|r| {
            format!(
                "    {{ \"channel\": {}, \"treatment\": {}, \"identical\": {} }}",
                js(r.channel), js(r.treatment), r.identical
            )
        })
        .collect();
    s.push_str("    \"proof\": [\n");
    s.push_str(&pr.join(",\n"));
    s.push_str("\n    ]\n  }\n}\n");
    s
}

/// Logical processor count (rayon's default pool size == available parallelism).
fn num_cpus_logical() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

/// Physical-core hint. `available_parallelism` reports logical processors; with
/// hyperthreading the physical count is half. This is a hint for the report's
/// "past N threads is hyperthreading" note, not an exact topology query.
fn physical_hint() -> usize {
    (num_cpus_logical() / 2).max(1)
}


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// `profile` subcommand: see `profile::run_profile`. Measure-only — phase
/// breakdown + escape-time histogram + thread-scaling sweep for the f64
/// Mandelbrot render path. Runs one location (shallow-decorative by default);
/// run it twice with different `--center-*/--frame-width/--label` to bracket the
/// cost range. f64-only by construction (asserts the backend stayed f64).
#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Frame center, real part — arbitrary-precision decimal. Default: the
    /// shallow-decorative seahorse-valley spiral (reads palette character, not
    /// interior-dominated).
    #[arg(long, default_value = "-0.7453", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part — arbitrary-precision decimal.
    #[arg(long, default_value = "0.1127", allow_hyphen_values = true)]
    pub center_im: String,

    /// Width of the view in the complex plane.
    #[arg(long, default_value_t = 0.012)]
    pub frame_width: f64,

    /// Maximum iterations before a pixel is treated as interior.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Output image width in pixels (height follows 3:2). Default 1280 gives a
    /// stable phase/scaling signal; pass 384 for the contact-sheet tile size.
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Linear supersampling factor (S×S box downsample). Iteration scales with S².
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Repeats per timed measurement (min + median reported; kernel is
    /// deterministic, so spread is system noise).
    #[arg(long, default_value_t = 5)]
    pub runs: usize,

    /// Thread counts for the strong-scaling sweep (comma-separated). Each builds
    /// its own rayon pool and re-times the iteration pass. Empty skips the sweep.
    #[arg(long, default_value = "1,2,4,6,8,12")]
    pub threads: String,

    /// Label for the printed report / JSON (e.g. `shallow`, `interior`).
    #[arg(long, default_value = "shallow")]
    pub label: String,

    /// Built-in palette for the shade/encode phases (`default`, `cubehelix`, `viridis`).
    #[arg(long, default_value = "default")]
    pub palette: String,

    /// Output directory for the profiling JSON.
    #[arg(long, default_value = "out/profile")]
    pub out_dir: String,
}

impl ProfileArgs {
    /// Parse `--threads` (comma-separated) into the scaling sweep counts.
    /// An empty list (or all-zero) means "no sweep".
    pub fn resolved_threads(&self) -> Result<Vec<usize>, String> {
        let mut out = Vec::new();
        for s in self.threads.split(',') {
            let t = s.trim();
            if t.is_empty() {
                continue;
            }
            let n: usize = t
                .parse()
                .map_err(|_| format!("invalid --threads component '{t}'"))?;
            if n > 0 {
                out.push(n);
            }
        }
        Ok(out)
    }
}
