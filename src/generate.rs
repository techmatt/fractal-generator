//! `generate` subcommand — discovery location sampler promoted to a generator.
//!
//! First non-diagnostic build on the location front. Lifts probe 1's throwaway
//! discovery sampler (`location_probe_probe1`, diagnosis-only) into a real
//! subcommand that emits located, logged, single-palette-preview keepers at
//! scale, with the full per-image log vector persisted for the downstream bias
//! loop. **Promotion, not redesign**: the cheap screen (interior% ∧ spread ∧
//! esc_median), the log-uniform shallow scale draw, the escape-distribution
//! logging, and the corpus descriptor as a *keeper-only feature* (not a gate)
//! all carry over unchanged from probe 1.
//!
//! What hardened for a real subcommand vs. the probe:
//!  - **Run-to-target-K loop** (not a fixed draw count): draw → cheap screen →
//!    [`AcceptBand`] → keep, looping until K keepers, capped by a max-draws
//!    safeguard (reported if hit).
//!  - **CLI arg parsing** for K, seed, box, scale range, screen res, and the
//!    band constants (so the deferred boundary retune is a flag, not a recompile).
//!  - **Output layout** outside `out/`: `locations.jsonl` (one keeper/row) +
//!    `manifest.json` under `data/generated/<run>/`, plus an annotated keeper
//!    contact sheet for an eye-pass.
//!
//! The cheap screen + band must match probe 1 numerically: with the probe-1
//! defaults (seed, box, scale range, screen res, band) the first K keepers
//! reproduce the probe-1 keeper stream (the draw order and accept decisions are
//! identical — same three `unit()` draws per candidate, same `screen_stats`,
//! same band).
//!
//! ## Accept band — retuned against run0 hand-labels
//!
//! The accept band (`spread ≥ 50 ∧ interior ≤ 40% ∧ esc_median ≥ 3`) was
//! **retuned against Matt's eye-labels on run0's 50 keepers** (band-retune pass),
//! replacing probe 1's un-eyeballed `8 / 90% / 3`. Two-sided: `interior ≤ 40%`
//! encodes the mostly-black rule; `spread ≥ 50` cuts the thin-filament sparse
//! class (it lands in the confirmed gap between the bad-sparse SPRD ceiling ≈23
//! and the good-anchor SPRD floor ≈86), losing zero good anchors. The band stays
//! a **single, named, centralized [`AcceptBand`]** read in one place (never
//! inlined magic numbers), and each keeper persists the **per-clause pass margin**
//! (`spread − min`, `interior_max − interior`, `esc_median − min`), so the next
//! batch can be straddler-audited from the log and re-retuned (via flags) without
//! touching sampler structure. Thresholds are fit to 50 frames — re-check per batch.

use std::fmt::Write as _;
use std::path::Path;

use astro_float::BigFloat;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::{BackendChoice, GenerateArgs};
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::energy::{self, distance, region_energies, Signature};
use crate::palette::Palette;
use crate::probe::{self, frac_le, load_colormap};
use crate::{font, hp, render, sheet};

// --- fixed (non-exposed) regime constants ------------------------------------

/// Coarse fixed escape histogram bins over smooth_iter ∈ [0, maxiter], escaped
/// pixels only, normalized to escaped count. Fixed edges → comparable across
/// keepers (for re-fitting the band from data later). Matches probe 1.
const ESC_HIST_BINS: usize = 16;

/// Screen supersample. ss1 — moments/histogram only need a representative pixel
/// field, not AA (matches probe 1).
const SCREEN_SS: u32 = 1;

/// Keeper render resolution — the **calibration regime** (`calibrate`'s
/// `--candidate-width 1280 --supersample 2`), so the corpus descriptor sees the
/// same input scale the corpus signatures were frozen at. Held fixed (not a flag)
/// because the descriptor is only comparable at this scale. Matches probe 1.
const KEEP_W: u32 = 1280;
const KEEP_H: u32 = 720; // 16:9
const KEEP_SS: u32 = 2;

/// Equal per-scale EMD weights — the `calibrate` default. Matches probe 1.
const WEIGHTS: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

/// The preview colormap is a *flag* (`--palette`, default `cubehelix`), not a
/// constant — structure-finding is palette-independent, so the single-tile
/// preview palette is purely cosmetic (the 3-palette labeling unit is a
/// downstream stage). Resolved once, in `run_generate`. The colormaps live in:
const COLORMAPS_PATH: &str = "data/palettes/clean_colormaps.json";

// --- the accept band (FLAG: centralized; not yet eye-validated) --------------

/// The cheap-screen keeper decision boundary — the **single, named, centralized**
/// definition (retuned against run0 hand-labels; see module doc). Re-retuning
/// touches *only* this default (via the `generate` band flags), never inlined
/// numbers.
///
/// Two-sided: a low/flat side (spread floor + escape-median floor) rejects flat
/// exterior / instant-escape / filament-graze; a high side (interior cap) rejects
/// interior-black.
#[derive(Clone, Copy, Debug)]
pub struct AcceptBand {
    /// Middle-90% smooth-iter spread below this = flat (no iteration variety).
    pub spread_min: f64,
    /// Interior (max-iter) fraction above this = mostly dead black.
    pub interior_max: f64,
    /// Median escape smooth-iter below this = whole frame escapes in a few
    /// iterations (far exterior / instant-escape).
    pub esc_median_min: f64,
}

impl Default for AcceptBand {
    /// Retuned against Matt's eye-labels on run0's 50 keepers (band-retune pass).
    /// Two-sided: `interior_max` encodes Matt's 40%-black rule; `spread_min`
    /// sits in the confirmed gap (bad-sparse ceiling SPRD≈23, good-anchor floor
    /// SPRD≈86) at 50, cutting the thin-filament class while keeping branching
    /// structure and every good anchor. `esc_median_min` unchanged (inactive on
    /// run0; kept as a far-exterior/instant-escape backstop). Flags still
    /// override each clause. Fit to 50 frames — re-check on each new batch.
    fn default() -> Self {
        Self {
            spread_min: 20.0,  // loose: seed-discovery; presentation filter applied downstream
            interior_max: 0.80, // loose: seed-discovery; presentation filter applied downstream
            esc_median_min: 3.0,
        }
    }
}

/// The result of testing one screen against the band: the accept verdict, the
/// failed-clause names (empty ⇒ accepted), the priority-ordered primary mode, and
/// the per-clause pass margins (positive ⇒ passed that clause by this much).
pub(crate) struct BandVerdict {
    pub(crate) accepted: bool,
    pub(crate) primary: &'static str,
    /// `spread − spread_min` (native smooth-iter units).
    pub(crate) spread_margin: f64,
    /// `interior_max − interior_frac` (fraction; ×100 for the documented "90−%").
    pub(crate) interior_margin: f64,
    /// `esc_median − esc_median_min` (native smooth-iter units).
    pub(crate) esc_median_margin: f64,
}

impl AcceptBand {
    /// Test a screen's three scalars against the band. The *only* place the band
    /// is read (FLAG: one-place decision boundary). Shared with `reject_corridor`,
    /// which reclassifies every draw against this same band.
    pub(crate) fn test(&self, interior_frac: f64, spread: f64, esc_median: f64) -> BandVerdict {
        let mut failed: Vec<&'static str> = Vec::new();
        if interior_frac > self.interior_max {
            failed.push("interior_black");
        }
        if esc_median < self.esc_median_min {
            failed.push("instant_escape");
        }
        if spread < self.spread_min {
            failed.push("flat");
        }
        let accepted = failed.is_empty();
        // Primary mode for reject bucketing (priority: dead interior, then
        // instant escape, then flat/graze).
        let primary = if accepted {
            "ok"
        } else if failed.contains(&"interior_black") {
            "interior_black"
        } else if failed.contains(&"instant_escape") {
            "instant_escape"
        } else {
            "flat"
        };
        BandVerdict {
            accepted,
            primary,
            spread_margin: spread - self.spread_min,
            interior_margin: self.interior_max - interior_frac,
            esc_median_margin: esc_median - self.esc_median_min,
        }
    }
}

/// Low fixed coloring density (same as probe 1): ~1 gradient cycle / 250
/// smooth-iter, so value→color reads as structure, not palette churn. Shared
/// with `reject_corridor` so its tiles shade exactly like keeper previews.
pub(crate) fn color_params() -> ColorParams {
    ColorParams {
        density: 0.004,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    }
}

/// Escape-iteration distribution across one frame (escaped pixels only).
/// Identical shape to probe 1's `EscDist`. Shared with `reject_corridor` (its
/// all-draw log dumps the same vector for every draw, kept or not).
#[derive(Clone)]
pub(crate) struct EscDist {
    pub(crate) count: usize,
    pub(crate) mean: f64,
    pub(crate) median: f64,
    pub(crate) std: f64,
    pub(crate) skew: f64,
    pub(crate) min: f64,
    pub(crate) p5: f64,
    pub(crate) p25: f64,
    pub(crate) p75: f64,
    pub(crate) p95: f64,
    pub(crate) max: f64,
    /// Spread = p95 − p5.
    pub(crate) spread: f64,
    /// Fixed-bin histogram (fractions of escaped pixels), `ESC_HIST_BINS` long.
    pub(crate) hist: Vec<f64>,
}

/// A persisted keeper: the full per-image log vector (over-logged — the bias
/// loop can only use what we store).
struct Keeper {
    /// Keeper ordinal (acceptance position, 0-based).
    keeper_index: usize,
    /// Draw ordinal it was accepted at (recovers draws-per-keeper post hoc).
    draw_index: usize,
    center: Complex<f64>,
    frame_width: f64,
    /// The raw [0,1) unit draw that produced `frame_width` (log-uniform map).
    scale_u: f64,
    interior_frac: f64,
    esc: EscDist,
    /// Per-clause pass margins (see [`BandVerdict`] / FLAG).
    spread_margin: f64,
    interior_margin: f64,
    esc_median_margin: f64,
    glitched_px: u64,
    // keeper-only corpus descriptor features (feature, not gate):
    sparse_score: f64,
    sparse_pct: f64,
    nn_emd: f64,
}

/// `generate` entry point.
pub fn run_generate(args: &GenerateArgs) -> Result<(), String> {
    if args.keepers == 0 {
        return Err("--keepers must be > 0".into());
    }
    let (re_lo, re_hi, im_lo, im_hi) = args.resolved_box()?;
    if args.fw_lo <= 0.0 || args.fw_hi <= args.fw_lo {
        return Err(format!(
            "invalid --fw-lo/--fw-hi: need 0 < fw_lo < fw_hi (got {}, {})",
            args.fw_lo, args.fw_hi
        ));
    }
    let band = args.band();
    let screen_w = args.screen_width.max(1);
    let screen_h = (screen_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let thumb_w = args.thumb_width.max(1);
    let thumb_h = (thumb_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    // Max-draws safeguard: 0 → a generous K×500 (probe-1 yield ~8.8% ⇒ ~11.4
    // draws/keeper, so this is ~44× headroom over the expected draw count).
    let max_draws = if args.max_draws == 0 {
        args.keepers.saturating_mul(500)
    } else {
        args.max_draws
    };

    let out_dir = Path::new(&args.out_dir);
    let thumbs_dir = out_dir.join("thumbs");
    crate::ensure_parent_dir(thumbs_dir.join("x"))?;

    // Held-out colormap (reuse the probe colormap loader).
    // The only place the preview palette is read (single read site). Default
    // cubehelix; any name in `clean_colormaps.json` works via `--palette`.
    let cm_text = std::fs::read_to_string(COLORMAPS_PATH)
        .map_err(|e| format!("read {COLORMAPS_PATH}: {e}"))?;
    let stops = load_colormap(&cm_text, &args.palette)?;
    // Selective seam fix: SEQUENTIAL (mirror_needed) maps bake pre-mirrored.
    let mirror = probe::colormap_mirror_needed(&cm_text, &args.palette);
    let palette = Palette::from_srgb8_stops_mirrored(args.palette.clone(), &stops, false, mirror);
    let params = color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    // Corpus descriptor: frozen bins + per-image signatures (keeper-only use).
    let art = std::fs::read_to_string(&args.artifact)
        .map_err(|e| format!("read {}: {e}", args.artifact))?;
    let (bins, corpus) = energy::parse_artifact(&art)?;
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    let mut corpus_sparse: Vec<f64> = corpus.iter().map(|(_, s)| s.hist[0][0]).collect();
    corpus_sparse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "corpus: {} signatures (descriptor is a keeper-only feature, not a gate)",
        corpus.len()
    );

    eprintln!(
        "generate: target K={} keepers, seed {}, screen {screen_w}x{screen_h} ss{SCREEN_SS}, \
         keepers {KEEP_W}x{KEEP_H} ss{KEEP_SS}, max-draws {max_draws}",
        args.keepers, args.seed
    );
    eprintln!(
        "box re[{re_lo},{re_hi}] im[{im_lo},{im_hi}]  fw log-uniform [{},{}] (geo-center ~ {:.4})",
        args.fw_lo,
        args.fw_hi,
        (args.fw_lo * args.fw_hi).sqrt()
    );
    eprintln!(
        "band: spread>={} AND interior<={:.0}% AND esc_median>={}",
        band.spread_min,
        band.interior_max * 100.0,
        band.esc_median_min
    );

    let ln_lo = args.fw_lo.ln();
    let ln_hi = args.fw_hi.ln();

    let mut rng = probe::SplitMix64(args.seed);
    let mut keepers: Vec<Keeper> = Vec::with_capacity(args.keepers);
    let mut keeper_thumbs: Vec<RgbImage> = Vec::with_capacity(args.keepers);
    // Reject tally per primary mode (for the stdout summary).
    let mut reject_modes = [0usize; 3]; // interior_black, instant_escape, flat
    let mode_idx = |m: &str| match m {
        "interior_black" => 0,
        "instant_escape" => 1,
        _ => 2,
    };

    let mut draw = 0usize;
    let mut max_draws_hit = false;
    while keepers.len() < args.keepers {
        if draw >= max_draws {
            max_draws_hit = true;
            break;
        }
        // Three draws per candidate, deterministic order: re, im, scale (matches
        // probe 1, so the keeper stream reproduces on the probe-1 defaults).
        let re = re_lo + rng.unit() * (re_hi - re_lo);
        let im = im_lo + rng.unit() * (im_hi - im_lo);
        let scale_u = rng.unit();
        let frame_width = (ln_lo + scale_u * (ln_hi - ln_lo)).exp();
        let center = Complex::new(re, im);
        let this_draw = draw;
        draw += 1;

        // --- cheap low-res neighborhood screen (every draw) ---
        let prec = hp::prec_bits(screen_w, frame_width);
        let cre = BigFloat::from_f64(center.re, prec);
        let cim = BigFloat::from_f64(center.im, prec);
        let panel = probe::render_mandel_panel(
            &cre, &cim, center, frame_width, screen_w, screen_h, SCREEN_SS, args.maxiter,
            args.bailout, prec, trap, BackendChoice::F64,
        );
        let (interior_frac, esc) = screen_stats(&panel.buf.samples, args.maxiter);

        // Band test reads esc.spread (p95−p5) and esc.median raw, matching probe 1.
        let v = band.test(interior_frac, esc.spread, esc.median);
        if !v.accepted {
            reject_modes[mode_idx(v.primary)] += 1;
            continue;
        }

        // --- keeper-only: high-res render + corpus descriptor ---
        let kprec = hp::prec_bits(KEEP_W, frame_width);
        let kre = BigFloat::from_f64(center.re, kprec);
        let kim = BigFloat::from_f64(center.im, kprec);
        let kpanel = probe::render_mandel_panel(
            &kre, &kim, center, frame_width, KEEP_W, KEEP_H, KEEP_SS, args.maxiter, args.bailout,
            kprec, trap, BackendChoice::F64,
        );
        let krgb = render::shade_and_downsample(
            &kpanel.buf.samples, KEEP_W, KEEP_H, KEEP_SS, &palette, &params, kpanel.spacing,
        );
        let regions = region_energies(&krgb);
        let sig = bins.signature(&regions);
        let sparse_score = sig.hist[0][0];
        let sparse_pct = frac_le(&corpus_sparse, sparse_score);
        let nn_emd = corpus_sigs
            .iter()
            .map(|cs| distance(&sig, cs, &WEIGHTS))
            .fold(f64::INFINITY, f64::min);

        let keeper_index = keepers.len();
        let mut kth = image::imageops::resize(&krgb, thumb_w, thumb_h, FilterType::Triangle);
        annotate(&mut kth, keeper_index, frame_width, interior_frac, esc.spread, sparse_pct);
        kth.save(thumbs_dir.join(format!("keep_{keeper_index:04}.png")))
            .map_err(|e| format!("save keeper thumb {keeper_index}: {e}"))?;
        keeper_thumbs.push(kth);

        eprintln!(
            "  keeper {keeper_index:04} @ draw {this_draw} c=({re:.5},{im:.5}) fw={frame_width:.4} \
             int={:.0}% spread={:.0} med={:.0} sparse_p{:.0} nn={:.3}",
            interior_frac * 100.0,
            esc.spread.max(0.0),
            esc.median,
            sparse_pct * 100.0,
            nn_emd
        );

        keepers.push(Keeper {
            keeper_index,
            draw_index: this_draw,
            center,
            frame_width,
            scale_u,
            interior_frac,
            esc,
            spread_margin: v.spread_margin,
            interior_margin: v.interior_margin,
            esc_median_margin: v.esc_median_margin,
            glitched_px: panel.buf.glitched_pixels,
            sparse_score,
            sparse_pct,
            nn_emd,
        });
    }

    let total_draws = draw;
    let accept_rate = keepers.len() as f64 / total_draws.max(1) as f64;

    // --- artifact 1: keeper contact sheet ---
    if !keeper_thumbs.is_empty() {
        let grid = sheet::compose_grid(&keeper_thumbs, Some(args.cols.max(1)));
        let p = out_dir.join("keeper_sheet.png");
        grid.save(&p).map_err(|e| format!("save keeper sheet: {e}"))?;
        eprintln!("keeper sheet: {} ({} tiles)", p.display(), keeper_thumbs.len());
    }

    // --- artifact 2: locations.jsonl (one keeper per row) ---
    let jsonl = build_jsonl(&keepers);
    let jsonl_path = out_dir.join("locations.jsonl");
    std::fs::write(&jsonl_path, jsonl).map_err(|e| format!("write locations.jsonl: {e}"))?;

    // --- artifact 3: run manifest (run-level config, fully reproducible) ---
    let manifest = build_manifest(
        args, &band, (re_lo, re_hi, im_lo, im_hi), screen_w, screen_h, max_draws, total_draws,
        keepers.len(), accept_rate, max_draws_hit, corpus.len(),
    );
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest).map_err(|e| format!("write manifest.json: {e}"))?;

    // --- stdout summary ---
    println!("=== generate ===");
    println!("seed={}  K={} keepers", args.seed, keepers.len());
    if max_draws_hit {
        println!(
            "WARNING: max-draws safeguard hit ({total_draws} draws) before reaching K={} — \
             produced {} keepers. Raise --max-draws or loosen the band.",
            args.keepers,
            keepers.len()
        );
    }
    println!(
        "draws={total_draws}  accept_rate={:.1}%  draws-per-keeper~{:.1}",
        accept_rate * 100.0,
        total_draws as f64 / keepers.len().max(1) as f64
    );
    println!(
        "scale-range: fw log-uniform [{},{}] (geo-center ~ {:.4})",
        args.fw_lo,
        args.fw_hi,
        (args.fw_lo * args.fw_hi).sqrt()
    );
    println!(
        "band used: spread>={} AND interior<={:.0}% AND esc_median>={}",
        band.spread_min,
        band.interior_max * 100.0,
        band.esc_median_min
    );
    println!(
        "rejects: interior_black={} instant_escape={} flat={}",
        reject_modes[0], reject_modes[1], reject_modes[2]
    );
    println!("locations: {}", jsonl_path.display());
    println!("manifest : {}", manifest_path.display());
    Ok(())
}

/// Cheap screen stats straight off the sample buffer: interior fraction + the
/// full escape-iteration distribution (moments + fixed-bin histogram). Identical
/// to probe 1's `screen_stats` (so the keeper stream reproduces). Shared with
/// `reject_corridor`.
pub(crate) fn screen_stats(
    samples: &[crate::backend::PixelSample],
    maxiter: u32,
) -> (f64, EscDist) {
    let mut esc: Vec<f64> = Vec::with_capacity(samples.len());
    let mut interior = 0usize;
    for s in samples {
        if s.escaped {
            esc.push(s.smooth_iter);
        } else {
            interior += 1;
        }
    }
    let total = samples.len().max(1);
    let interior_frac = interior as f64 / total as f64;
    esc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = esc.len();
    let q = |t: f64| -> f64 {
        if n == 0 {
            f64::NAN
        } else {
            esc[((t * (n - 1) as f64).round() as usize).min(n - 1)]
        }
    };
    let (mean, std, skew) = if n == 0 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mean = esc.iter().sum::<f64>() / n as f64;
        let var = esc.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        let skew = if std > 0.0 {
            (esc.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n as f64) / std.powi(3)
        } else {
            0.0
        };
        (mean, std, skew)
    };

    // Fixed-bin histogram over [0, maxiter], escaped only.
    let mut hist = vec![0.0f64; ESC_HIST_BINS];
    let span = maxiter as f64;
    for &v in &esc {
        let b = ((v / span) * ESC_HIST_BINS as f64).floor() as usize;
        hist[b.min(ESC_HIST_BINS - 1)] += 1.0;
    }
    if n > 0 {
        for h in hist.iter_mut() {
            *h /= n as f64;
        }
    }

    let p5 = q(0.05);
    let p95 = q(0.95);
    let esc = EscDist {
        count: n,
        mean,
        median: q(0.5),
        std,
        skew,
        min: esc.first().copied().unwrap_or(f64::NAN),
        p5,
        p25: q(0.25),
        p75: q(0.75),
        p95,
        max: esc.last().copied().unwrap_or(f64::NAN),
        spread: p95 - p5,
        hist,
    };
    (interior_frac, esc)
}

/// Annotate a keeper thumbnail with keeper index, frame_width, interior%, spread,
/// and the corpus percentile (mirrors probe 1's keeper annotation).
fn annotate(th: &mut RgbImage, i: usize, fw: f64, interior_frac: f64, spread: f64, sparse_pct: f64) {
    let white = Rgb([240u8, 240, 240]);
    font::draw_text(th, &format!("{i:04} fw{fw:.4}"), 2, 2, 1, white, true);
    font::draw_text(
        th,
        &format!(
            "INT{:.0}% SPRD{:.0} P{:.0}",
            interior_frac * 100.0,
            spread.max(0.0),
            sparse_pct * 100.0
        ),
        2,
        12,
        1,
        white,
        true,
    );
    font::draw_text(th, "OK", th.width().saturating_sub(18), 2, 1, Rgb([120, 255, 120]), true);
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

fn jarr(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (k, x) in v.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        s.push_str(&jnum(*x));
    }
    s.push(']');
    s
}

/// `locations.jsonl`: one keeper per line (newline-delimited JSON objects).
fn build_jsonl(keepers: &[Keeper]) -> String {
    let mut s = String::new();
    for c in keepers {
        let _ = writeln!(
            s,
            "{{ \"keeper_index\": {}, \"draw_index\": {}, \"accept_position\": {}, \
             \"center_re\": {}, \"center_im\": {}, \"frame_width\": {}, \"scale_u\": {}, \
             \"interior_frac\": {}, \"spread\": {}, \"esc_median\": {}, \
             \"esc_count\": {}, \"esc_mean\": {}, \"esc_std\": {}, \"esc_skew\": {}, \
             \"esc_min\": {}, \"esc_p5\": {}, \"esc_p25\": {}, \"esc_p75\": {}, \"esc_p95\": {}, \
             \"esc_max\": {}, \"esc_hist\": {}, \
             \"margin_spread\": {}, \"margin_interior\": {}, \"margin_interior_pct\": {}, \
             \"margin_esc_median\": {}, \
             \"glitched_px\": {}, \"sparse_score\": {}, \"sparse_pct\": {}, \"nn_emd\": {} }}",
            c.keeper_index,
            c.draw_index,
            c.keeper_index,
            jnum(c.center.re),
            jnum(c.center.im),
            jnum(c.frame_width),
            jnum(c.scale_u),
            jnum(c.interior_frac),
            jnum(c.esc.spread),
            jnum(c.esc.median),
            c.esc.count,
            jnum(c.esc.mean),
            jnum(c.esc.std),
            jnum(c.esc.skew),
            jnum(c.esc.min),
            jnum(c.esc.p5),
            jnum(c.esc.p25),
            jnum(c.esc.p75),
            jnum(c.esc.p95),
            jnum(c.esc.max),
            jarr(&c.esc.hist),
            jnum(c.spread_margin),
            jnum(c.interior_margin),
            jnum(c.interior_margin * 100.0),
            jnum(c.esc_median_margin),
            c.glitched_px,
            jnum(c.sparse_score),
            jnum(c.sparse_pct),
            jnum(c.nn_emd),
        );
    }
    s
}

/// `manifest.json`: run-level config so any batch is fully reproducible and
/// self-describing.
#[allow(clippy::too_many_arguments)]
fn build_manifest(
    args: &GenerateArgs,
    band: &AcceptBand,
    bx: (f64, f64, f64, f64),
    screen_w: u32,
    screen_h: u32,
    max_draws: usize,
    total_draws: usize,
    accepted: usize,
    accept_rate: f64,
    max_draws_hit: bool,
    corpus_size: usize,
) -> String {
    let (re_lo, re_hi, im_lo, im_hi) = bx;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"subcommand\": \"generate\",\n");
    s.push_str("  \"lineage\": \"discovery-sampler probe 1 promoted to a generator\",\n");
    let _ = writeln!(s, "  \"seed\": {},", args.seed);
    let _ = writeln!(s, "  \"keepers_target\": {},", args.keepers);
    let _ = writeln!(s, "  \"keepers_produced\": {accepted},");
    let _ = writeln!(s, "  \"max_draws\": {max_draws},");
    let _ = writeln!(s, "  \"total_draws\": {total_draws},");
    let _ = writeln!(s, "  \"accept_rate\": {},", jnum(accept_rate));
    let _ = writeln!(
        s,
        "  \"draws_per_keeper\": {},",
        jnum(total_draws as f64 / accepted.max(1) as f64)
    );
    let _ = writeln!(s, "  \"max_draws_hit\": {max_draws_hit},");
    let _ = writeln!(
        s,
        "  \"box\": {{ \"re_lo\": {re_lo}, \"re_hi\": {re_hi}, \"im_lo\": {im_lo}, \"im_hi\": {im_hi} }},"
    );
    let _ = writeln!(
        s,
        "  \"frame_width_range\": {{ \"lo\": {}, \"hi\": {}, \"sampling\": \"log-uniform\", \"geo_center\": {} }},",
        args.fw_lo,
        args.fw_hi,
        (args.fw_lo * args.fw_hi).sqrt()
    );
    let _ = writeln!(s, "  \"maxiter\": {},", args.maxiter);
    let _ = writeln!(s, "  \"bailout\": {},", args.bailout);
    let _ = writeln!(
        s,
        "  \"screen\": {{ \"w\": {screen_w}, \"h\": {screen_h}, \"ss\": {SCREEN_SS} }},"
    );
    let _ = writeln!(
        s,
        "  \"keeper_render\": {{ \"w\": {KEEP_W}, \"h\": {KEEP_H}, \"ss\": {KEEP_SS} }},"
    );
    let _ = writeln!(s, "  \"colormap\": {},", probe::js(&args.palette));
    s.push_str("  \"color\": { \"channel\": \"smooth\", \"density\": 0.004, \"offset\": 0.0, \"interior\": \"black\" },\n");
    s.push_str("  \"palette\": \"held out (structure-finding only; 3-palette labeling is downstream)\",\n");
    let _ = writeln!(
        s,
        "  \"esc_hist\": {{ \"bins\": {ESC_HIST_BINS}, \"range\": [0, {}], \"note\": \"fractions of escaped pixels, fixed edges\" }},",
        args.maxiter
    );
    let _ = writeln!(
        s,
        "  \"accept_band\": {{ \"spread_min\": {}, \"interior_max\": {}, \"esc_median_min\": {}, \"note\": \"retuned against run0 hand-labels (two-sided: interior<=40% + spread floor in confirmed 23..86 gap); centralized one-place default, flag-overridable; per-clause margins logged per keeper; fit to 50 frames, re-check per batch\" }},",
        band.spread_min, band.interior_max, band.esc_median_min
    );
    s.push_str("  \"descriptor\": \"keeper-only feature (energy s16 sparse + nn_emd to corpus); NOT a gate\",\n");
    let _ = writeln!(s, "  \"artifact\": {},", probe::js(&args.artifact));
    let _ = writeln!(s, "  \"corpus_size\": {corpus_size}");
    s.push_str("}\n");
    s
}
