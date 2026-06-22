//! **Throwaway diagnostic — discovery-sampler probe 1 (diagnosis-only).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Builds directly on
//! the probe-0 scaffold (`location_probe`): reuses its colormap loader
//! (`load_colormap`), percentile helper (`frac_le`), the `probe::*` render
//! surface (f64 cheap regime), `render::shade_and_downsample`, the `energy`
//! corpus descriptor, and `sheet::compose_grid`.
//!
//! **What this decides.** Probe 0 settled feasibility (≈2% colorable at one fixed
//! zoom). This probe measures *diversity coverage*: with a loose escape-time band
//! over a *range* of shallow zooms, do the provisional keepers spread across
//! visually distinct motifs or collapse to one or two? That verdict decides
//! whether plain discovery suffices or we later need region-seeded sampling.
//!
//! Method (no catalog, no optimization): uniform center draw + log-uniform
//! shallow scale draw → cheap low-res neighborhood screen (interior%, smooth-iter
//! spread, full escape-iteration distribution) → loose two-sided escape band →
//! keep. We characterize the *region we'd render*, not a single center orbit. The
//! band is loose by design (probe 0 rejected its own one keeper with too high a
//! floor — not repeated here) and is **reported, not optimized**: render
//! everything, log where keepers sit in escape-stat space, re-fit later.
//!
//! Deliverables under `data/location_probe/probe1/`: a keeper contact sheet, a
//! reject strip (sampled per failure mode), and a full JSON log of all N draws.
//!
//! Run:
//! ```text
//! cargo test --release --lib discovery_probe1 -- --ignored --nocapture
//! ```

use std::fmt::Write as _;
use std::path::Path;

use astro_float::BigFloat;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::BackendChoice;
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::energy::{self, distance, region_energies, Signature};
use crate::location_probe::{frac_le, load_colormap};
use crate::palette::Palette;
use crate::{font, hp, probe, render, sheet};

// --- fixed experiment parameters --------------------------------------------

/// Fixed SplitMix64 seed for probe 1 (probe 0 used `20_260_621`; a distinct
/// stream so the two probes don't share draws).
const SEED: u64 = 20_260_622;
/// Candidates drawn (enough to see a band and judge motif spread).
const N: usize = 400;

/// Non-trivial sampling box (reused from probe 0): real ∈ [RE_LO, RE_HI],
/// imag ∈ [IM_LO, IM_HI].
const RE_LO: f64 = -2.0;
const RE_HI: f64 = 0.7;
const IM_LO: f64 = -1.2;
const IM_HI: f64 = 1.2;

/// Shallow-capped **frame_width range**, sampled log-uniform (uniform in zoom).
/// Bounds chosen so probe 0's fixed `0.012` sits essentially at the geometric
/// center (√(FW_LO·FW_HI) ≈ 0.0122). Both ends stay in the f64 cheap regime:
/// FW_LO=0.003 is ≈1000× magnification from the full set — far above f64 epsilon.
const FW_LO: f64 = 0.003;
const FW_HI: f64 = 0.05;

/// Iteration budget = a normal render's default (`LocationArgs::maxiter`).
const MAXITER: u32 = 1000;
const BAILOUT: f64 = 1e6;

/// Cheap low-res neighborhood screen (every draw). ss1 — moments/histogram only
/// need a representative pixel field, not AA.
const SCREEN_W: u32 = 320;
const SCREEN_H: u32 = 180; // 16:9
/// Keeper render (provisional-keepers only) at the **calibration regime**
/// (`calibrate`'s `--candidate-width 1280 --supersample 2`), so the corpus
/// descriptor sees the same input scale the corpus signatures were frozen at.
const KEEP_W: u32 = 1280;
const KEEP_H: u32 = 720; // 16:9
const KEEP_SS: u32 = 2;

/// Sheet/strip thumbnail size (16:9).
const THUMB_W: u32 = 256;
const THUMB_H: u32 = 144;

/// Equal per-scale EMD weights — the `calibrate` default.
const WEIGHTS: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

// --- one fixed neutral wide-range colormap (palette held out) ----------------

/// Viridis: perceptually uniform, near-monotone luminance — value→color reads as
/// location structure, not palette noise. Palette interaction is a labeling-stage
/// concern, not a structure-finding one.
const COLORMAP: &str = "viridis";
const COLORMAPS_PATH: &str = "data/palettes/clean_colormaps.json";

/// Low fixed density (same as probe 0): ~1 gradient cycle / 250 smooth-iter.
fn color_params() -> ColorParams {
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

// --- loose, two-sided provisional escape band (reported, not enforced) -------
//
// Two-sided: a low/flat side (spread floor + escape-median floor) rejects flat
// exterior / instant-escape / filament-graze; a high side (interior cap) rejects
// interior-black. Deliberately loose — probe-0 #03 (spread ≈14) must pass.

/// Middle-90% smooth-iter spread below this = flat (no iteration variety).
/// Floor below probe-0 #03's ≈14 so it passes (probe 0's 20 rejected its keeper).
const BAND_SPREAD_MIN: f64 = 8.0;
/// Interior (max-iter) fraction above this = mostly dead black.
const BAND_INTERIOR_MAX: f64 = 0.90;
/// Median escape smooth-iter below this = whole frame is far exterior, escaping
/// in a few iterations (flat). Low/loose — just kills instant-escape frames.
const BAND_ESC_MEDIAN_MIN: f64 = 3.0;

/// Coarse fixed escape histogram: NBANDS bins over smooth_iter ∈ [0, MAXITER],
/// escaped pixels only, normalized to escaped count. Fixed edges → comparable
/// across candidates (for re-fitting the band from data later).
const ESC_HIST_BINS: usize = 16;

/// Rejects sampled per primary failure mode for the reject strip.
const REJECTS_PER_MODE: usize = 6;

/// Escape-iteration distribution across one frame (escaped pixels only).
#[derive(Clone)]
struct EscDist {
    count: usize,
    mean: f64,
    median: f64,
    std: f64,
    skew: f64,
    min: f64,
    p5: f64,
    p25: f64,
    p75: f64,
    p95: f64,
    max: f64,
    /// Spread = p95 − p5 (matches probe-0's `spread`).
    spread: f64,
    /// Fixed-bin histogram (fractions of escaped pixels), `ESC_HIST_BINS` long.
    hist: Vec<f64>,
}

/// Per-candidate recorded vector (logged for all N draws).
struct Cand {
    index: usize,
    /// Seed-stream draw ordinal (3 `unit()` draws per candidate: re, im, scale).
    draw_ordinal: usize,
    center: Complex<f64>,
    frame_width: f64,
    /// The raw [0,1) unit draw that produced `frame_width` (log-uniform map).
    scale_u: f64,
    interior_frac: f64,
    esc: EscDist,
    accepted: bool,
    /// Which band clauses failed (empty ⇒ accepted).
    failed: Vec<&'static str>,
    /// Primary failure mode (priority order), or "ok".
    primary: &'static str,
    glitched_px: u64,
    // keeper-only descriptor features (NaN/None when not a keeper):
    sparse_score: f64,
    sparse_pct: f64,
    nn_emd: f64,
}

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn discovery_probe1() {
    run().expect("discovery probe 1");
}

fn run() -> Result<(), String> {
    let out_dir = Path::new("data/location_probe/probe1");
    let thumbs_dir = out_dir.join("thumbs");
    crate::ensure_parent_dir(thumbs_dir.join("x"))?;

    // Fixed colormap from the JSON library (reuse probe-0 loader).
    let cm_text = std::fs::read_to_string(COLORMAPS_PATH)
        .map_err(|e| format!("read {COLORMAPS_PATH}: {e}"))?;
    let stops = load_colormap(&cm_text, COLORMAP)?;
    let palette = Palette::from_srgb8_stops(COLORMAP, &stops, false);
    let params = color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    // Corpus descriptor: frozen bins + per-image signatures (keeper-only use).
    let art = std::fs::read_to_string(energy::ARTIFACT_PATH)
        .map_err(|e| format!("read {}: {e}", energy::ARTIFACT_PATH))?;
    let (bins, corpus) = energy::parse_artifact(&art)?;
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    let mut corpus_sparse: Vec<f64> = corpus.iter().map(|(_, s)| s.hist[0][0]).collect();
    corpus_sparse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("corpus: {} signatures (descriptor is keeper-only here)", corpus.len());

    let mut rng = probe::SplitMix64(SEED);
    let mut cands: Vec<Cand> = Vec::with_capacity(N);
    // Low-res screen thumbnail for every candidate (reject strip draws from these).
    let mut screen_thumbs: Vec<RgbImage> = Vec::with_capacity(N);
    // Keeper high-res thumbnails (the keeper contact sheet).
    let mut keeper_thumbs: Vec<RgbImage> = Vec::new();

    eprintln!(
        "probe 1: {N} draws, screen {SCREEN_W}x{SCREEN_H} ss1, keepers {KEEP_W}x{KEEP_H} ss{KEEP_SS}"
    );
    eprintln!("fw log-uniform [{FW_LO}, {FW_HI}] (geo-center ~ {:.4})", (FW_LO * FW_HI).sqrt());

    let ln_lo = FW_LO.ln();
    let ln_hi = FW_HI.ln();

    for i in 0..N {
        // Three draws per candidate, deterministic order: re, im, scale.
        let re = RE_LO + rng.unit() * (RE_HI - RE_LO);
        let im = IM_LO + rng.unit() * (IM_HI - IM_LO);
        let scale_u = rng.unit();
        let frame_width = (ln_lo + scale_u * (ln_hi - ln_lo)).exp();
        let center = Complex::new(re, im);

        // --- cheap low-res neighborhood screen (every draw) ---
        let prec = hp::prec_bits(SCREEN_W, frame_width);
        let cre = BigFloat::from_f64(center.re, prec);
        let cim = BigFloat::from_f64(center.im, prec);
        let panel = probe::render_mandel_panel(
            &cre, &cim, center, frame_width, SCREEN_W, SCREEN_H, 1, MAXITER, BAILOUT, prec, trap,
            BackendChoice::F64,
        );

        let (interior_frac, esc) = screen_stats(&panel.buf.samples);

        // --- loose two-sided band (reported, not enforced) ---
        let mut failed: Vec<&'static str> = Vec::new();
        if interior_frac > BAND_INTERIOR_MAX {
            failed.push("interior_black");
        }
        if esc.median < BAND_ESC_MEDIAN_MIN {
            failed.push("instant_escape");
        }
        if esc.spread < BAND_SPREAD_MIN {
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

        // Low-res screen thumbnail (for the reject strip / sanity).
        let screen_rgb = render::shade_and_downsample(
            &panel.buf.samples, SCREEN_W, SCREEN_H, 1, &palette, &params, panel.spacing,
        );
        let mut sth = image::imageops::resize(&screen_rgb, THUMB_W, THUMB_H, FilterType::Triangle);
        annotate(&mut sth, i, frame_width, interior_frac, esc.spread, None, primary, accepted);
        screen_thumbs.push(sth);

        // --- keeper-only: high-res render + corpus descriptor ---
        let (mut sparse_score, mut sparse_pct, mut nn_emd) = (f64::NAN, f64::NAN, f64::NAN);
        if accepted {
            let kprec = hp::prec_bits(KEEP_W, frame_width);
            let kre = BigFloat::from_f64(center.re, kprec);
            let kim = BigFloat::from_f64(center.im, kprec);
            let kpanel = probe::render_mandel_panel(
                &kre, &kim, center, frame_width, KEEP_W, KEEP_H, KEEP_SS, MAXITER, BAILOUT, kprec,
                trap, BackendChoice::F64,
            );
            let krgb = render::shade_and_downsample(
                &kpanel.buf.samples, KEEP_W, KEEP_H, KEEP_SS, &palette, &params, kpanel.spacing,
            );
            let regions = region_energies(&krgb);
            let sig = bins.signature(&regions);
            sparse_score = sig.hist[0][0];
            sparse_pct = frac_le(&corpus_sparse, sparse_score);
            nn_emd = corpus_sigs
                .iter()
                .map(|cs| distance(&sig, cs, &WEIGHTS))
                .fold(f64::INFINITY, f64::min);

            let mut kth = image::imageops::resize(&krgb, THUMB_W, THUMB_H, FilterType::Triangle);
            annotate(&mut kth, i, frame_width, interior_frac, esc.spread, Some(sparse_pct), "ok", true);
            kth.save(thumbs_dir.join(format!("keep_{i:03}.png")))
                .map_err(|e| format!("save keeper thumb {i}: {e}"))?;
            keeper_thumbs.push(kth);
        }

        cands.push(Cand {
            index: i,
            draw_ordinal: i,
            center,
            frame_width,
            scale_u,
            interior_frac,
            esc,
            accepted,
            failed,
            primary,
            glitched_px: panel.buf.glitched_pixels,
            sparse_score,
            sparse_pct,
            nn_emd,
        });

        let c = cands.last().unwrap();
        if (i + 1) % 25 == 0 || accepted {
            eprintln!(
                "  #{i:03} c=({re:.4},{im:.4}) fw={frame_width:.4} int={:.0}% spread={:.0} med={:.0} {}",
                interior_frac * 100.0,
                c.esc.spread.max(0.0),
                c.esc.median,
                if accepted {
                    format!("ACCEPT sparse_p{:.0} nn={:.3}", sparse_pct * 100.0, nn_emd)
                } else {
                    format!("reject[{}]", c.primary)
                }
            );
        }
    }

    let accepted_count = cands.iter().filter(|c| c.accepted).count();

    // --- artifact 1: keeper contact sheet ---
    if !keeper_thumbs.is_empty() {
        let sheet = sheet::compose_grid(&keeper_thumbs, Some(6));
        let p = out_dir.join("keeper_sheet.png");
        sheet.save(&p).map_err(|e| format!("save keeper sheet: {e}"))?;
        eprintln!("keeper sheet: {} ({} tiles)", p.display(), keeper_thumbs.len());
    } else {
        eprintln!("keeper sheet: SKIPPED (0 keepers — band too tight?)");
    }

    // --- artifact 2: reject strip (sampled per primary failure mode) ---
    let strip = build_reject_strip(&cands, &screen_thumbs);
    if let Some(strip) = strip {
        let p = out_dir.join("reject_strip.png");
        strip.save(&p).map_err(|e| format!("save reject strip: {e}"))?;
        eprintln!("reject strip: {}", p.display());
    }

    // --- artifact 3: full JSON log (all N draws) ---
    let json = build_json(&cands, corpus.len(), accepted_count);
    let json_path = out_dir.join("probe1.json");
    std::fs::write(&json_path, json).map_err(|e| format!("write json: {e}"))?;

    // --- summary ---
    let modes = ["interior_black", "instant_escape", "flat"];
    eprintln!("\n=== discovery-sampler probe 1 (diversity coverage) ===");
    eprintln!("seed={SEED}  N={N}  box re[{RE_LO},{RE_HI}] im[{IM_LO},{IM_HI}]");
    eprintln!(
        "fw log-uniform [{FW_LO},{FW_HI}] (probe-0's 0.012 ~ geo-center)  maxiter={MAXITER}  colormap={COLORMAP}"
    );
    eprintln!(
        "loose band: spread>={BAND_SPREAD_MIN} AND interior<={:.0}% AND esc_median>={BAND_ESC_MEDIAN_MIN} (NO corpus-sparse clause)",
        BAND_INTERIOR_MAX * 100.0
    );
    eprintln!(
        "ACCEPTED {accepted_count}/{N} = {:.1}%  ->  draws-per-keeper ~ {:.1}  (probe-0 baseline ~2%)",
        100.0 * accepted_count as f64 / N as f64,
        N as f64 / accepted_count.max(1) as f64
    );
    for m in modes {
        let n = cands.iter().filter(|c| !c.accepted && c.primary == m).count();
        eprintln!("  reject[{m}]: {n}");
    }
    // Where keepers sit in escape-stat space (so the band can be re-fit).
    report_keeper_locus(&cands);
    eprintln!("json: {}", json_path.display());
    Ok(())
}

/// Cheap screen stats straight off the sample buffer: interior fraction + the
/// full escape-iteration distribution (moments + fixed-bin histogram).
fn screen_stats(samples: &[crate::backend::PixelSample]) -> (f64, EscDist) {
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

    // Fixed-bin histogram over [0, MAXITER], escaped only.
    let mut hist = vec![0.0f64; ESC_HIST_BINS];
    let span = MAXITER as f64;
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

/// Annotate a thumbnail with index, frame_width, interior%, spread, (corpus pct),
/// and a corner mode/OK marker.
fn annotate(
    th: &mut RgbImage,
    i: usize,
    fw: f64,
    interior_frac: f64,
    spread: f64,
    sparse_pct: Option<f64>,
    primary: &str,
    accepted: bool,
) {
    let white = Rgb([240u8, 240, 240]);
    font::draw_text(th, &format!("{i:03} fw{fw:.4}"), 2, 2, 1, white, true);
    let line2 = match sparse_pct {
        Some(p) => format!("INT{:.0}% SPRD{:.0} P{:.0}", interior_frac * 100.0, spread.max(0.0), p * 100.0),
        None => format!("INT{:.0}% SPRD{:.0}", interior_frac * 100.0, spread.max(0.0)),
    };
    font::draw_text(th, &line2, 2, 12, 1, white, true);
    if accepted {
        font::draw_text(th, "OK", THUMB_W - 18, 2, 1, Rgb([120, 255, 120]), true);
    } else {
        font::draw_text(th, primary, 2, THUMB_H - 10, 1, Rgb([255, 150, 120]), true);
    }
}

/// Reject strip: up to `REJECTS_PER_MODE` rejects per primary failure mode, in
/// candidate order, one row per mode (modes stacked vertically via the grid).
fn build_reject_strip(cands: &[Cand], screen_thumbs: &[RgbImage]) -> Option<RgbImage> {
    let modes = ["interior_black", "instant_escape", "flat"];
    let mut tiles: Vec<RgbImage> = Vec::new();
    for m in modes {
        let picks: Vec<usize> = cands
            .iter()
            .filter(|c| !c.accepted && c.primary == m)
            .take(REJECTS_PER_MODE)
            .map(|c| c.index)
            .collect();
        // Pad the row to REJECTS_PER_MODE so the grid columns line up by mode.
        for k in 0..REJECTS_PER_MODE {
            if let Some(&idx) = picks.get(k) {
                tiles.push(screen_thumbs[idx].clone());
            } else {
                tiles.push(RgbImage::from_pixel(THUMB_W, THUMB_H, Rgb([24, 24, 24])));
            }
        }
    }
    if tiles.is_empty() {
        return None;
    }
    Some(sheet::compose_grid(&tiles, Some(REJECTS_PER_MODE)))
}

/// Print where accepted keepers sit in escape-stat space (min/median/max of the
/// key scalars) so the band can be re-fit from real numbers later.
fn report_keeper_locus(cands: &[Cand]) {
    let keepers: Vec<&Cand> = cands.iter().filter(|c| c.accepted).collect();
    if keepers.is_empty() {
        return;
    }
    let stat = |f: &dyn Fn(&Cand) -> f64| -> (f64, f64, f64) {
        let mut v: Vec<f64> = keepers.iter().map(|c| f(c)).filter(|x| x.is_finite()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (v[0], v[v.len() / 2], v[v.len() - 1])
        }
    };
    eprintln!("keeper locus (min / median / max):");
    let pr = |label: &str, (lo, md, hi): (f64, f64, f64)| {
        eprintln!("  {label:<14} {lo:.3} / {md:.3} / {hi:.3}");
    };
    pr("frame_width", stat(&|c| c.frame_width));
    pr("interior%", stat(&|c| c.interior_frac * 100.0));
    pr("spread", stat(&|c| c.esc.spread));
    pr("esc_median", stat(&|c| c.esc.median));
    pr("esc_mean", stat(&|c| c.esc.mean));
    pr("esc_std", stat(&|c| c.esc.std));
    pr("esc_skew", stat(&|c| c.esc.skew));
    pr("sparse_pct", stat(&|c| c.sparse_pct));
    pr("nn_emd", stat(&|c| c.nn_emd));
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

fn build_json(cands: &[Cand], corpus_size: usize, accepted: usize) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"probe\": \"discovery-sampler probe 1 (diversity coverage)\",\n");
    let _ = writeln!(s, "  \"seed\": {SEED},");
    let _ = writeln!(s, "  \"n\": {N},");
    let _ = writeln!(
        s,
        "  \"box\": {{ \"re_lo\": {RE_LO}, \"re_hi\": {RE_HI}, \"im_lo\": {IM_LO}, \"im_hi\": {IM_HI} }},"
    );
    let _ = writeln!(
        s,
        "  \"frame_width_range\": {{ \"lo\": {FW_LO}, \"hi\": {FW_HI}, \"sampling\": \"log-uniform\", \"geo_center\": {} }},",
        (FW_LO * FW_HI).sqrt()
    );
    let _ = writeln!(s, "  \"maxiter\": {MAXITER},");
    let _ = writeln!(s, "  \"bailout\": {BAILOUT},");
    let _ = writeln!(
        s,
        "  \"screen\": {{ \"w\": {SCREEN_W}, \"h\": {SCREEN_H}, \"ss\": 1 }},"
    );
    let _ = writeln!(
        s,
        "  \"keeper_render\": {{ \"w\": {KEEP_W}, \"h\": {KEEP_H}, \"ss\": {KEEP_SS} }},"
    );
    let _ = writeln!(s, "  \"colormap\": \"{COLORMAP}\",");
    s.push_str("  \"color\": { \"channel\": \"smooth\", \"density\": 0.004, \"offset\": 0.0, \"interior\": \"black\" },\n");
    let _ = writeln!(
        s,
        "  \"esc_hist\": {{ \"bins\": {ESC_HIST_BINS}, \"range\": [0, {MAXITER}], \"note\": \"fractions of escaped pixels, fixed edges\" }},"
    );
    let _ = writeln!(
        s,
        "  \"band\": {{ \"spread_min\": {BAND_SPREAD_MIN}, \"interior_max\": {BAND_INTERIOR_MAX}, \"esc_median_min\": {BAND_ESC_MEDIAN_MIN}, \"note\": \"loose two-sided; reported not enforced; NO corpus-sparse clause\" }},"
    );
    s.push_str("  \"descriptor\": \"keeper-only feature (energy s16 sparse + nn_emd to corpus); demoted from gate\",\n");
    let _ = writeln!(s, "  \"corpus_size\": {corpus_size},");
    let _ = writeln!(s, "  \"accepted_count\": {accepted},");
    let _ = writeln!(
        s,
        "  \"implied_draws_per_keeper\": {},",
        jnum(cands.len() as f64 / accepted.max(1) as f64)
    );
    s.push_str("  \"candidates\": [\n");
    for (k, c) in cands.iter().enumerate() {
        let comma = if k + 1 < cands.len() { "," } else { "" };
        let failed = {
            let mut t = String::from("[");
            for (j, f) in c.failed.iter().enumerate() {
                if j > 0 {
                    t.push_str(", ");
                }
                let _ = write!(t, "\"{f}\"");
            }
            t.push(']');
            t
        };
        let _ = writeln!(
            s,
            "    {{ \"index\": {}, \"draw_ordinal\": {}, \"center_re\": {}, \"center_im\": {}, \"frame_width\": {}, \"scale_u\": {}, \"interior_frac\": {}, \"esc_count\": {}, \"esc_mean\": {}, \"esc_median\": {}, \"esc_std\": {}, \"esc_skew\": {}, \"esc_min\": {}, \"esc_p5\": {}, \"esc_p25\": {}, \"esc_p75\": {}, \"esc_p95\": {}, \"esc_max\": {}, \"spread_p5_p95\": {}, \"esc_hist\": {}, \"accepted\": {}, \"failed\": {}, \"primary\": \"{}\", \"glitched_px\": {}, \"sparse_score\": {}, \"sparse_pct\": {}, \"nn_emd\": {} }}{}",
            c.index,
            c.draw_ordinal,
            jnum(c.center.re),
            jnum(c.center.im),
            jnum(c.frame_width),
            jnum(c.scale_u),
            jnum(c.interior_frac),
            c.esc.count,
            jnum(c.esc.mean),
            jnum(c.esc.median),
            jnum(c.esc.std),
            jnum(c.esc.skew),
            jnum(c.esc.min),
            jnum(c.esc.p5),
            jnum(c.esc.p25),
            jnum(c.esc.p75),
            jnum(c.esc.p95),
            jnum(c.esc.max),
            jnum(c.esc.spread),
            jarr(&c.esc.hist),
            c.accepted,
            failed,
            c.primary,
            c.glitched_px,
            jnum(c.sparse_score),
            jnum(c.sparse_pct),
            jnum(c.nn_emd),
            comma
        );
    }
    s.push_str("  ]\n}\n");
    s
}
