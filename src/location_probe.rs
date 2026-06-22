//! **Throwaway diagnostic — location-source base-rate probe (diagnosis-only).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`) so it never reaches
//! the production binary. It reuses the existing render/descriptor surface
//! rather than reimplementing it: `probe::render_mandel_panel` (the f64 cheap
//! regime), `render::shade_and_downsample`, and the `energy` descriptor
//! (`region_energies` → frozen-bin `signature` → corpus `distance`).
//!
//! Question it answers: at a fixed shallow zoom, what fraction of uniformly
//! random centers land on something with structure to color, and do the cheap
//! reject scalars (value-field spread, interior fraction, corpus sparse
//! percentile) track what the eye sees on the contact sheet? The deliverable is
//! one annotated contact sheet + one JSON sidecar under `data/location_probe/`.
//! No filtering, no quality claims — every candidate is rendered; Matt judges.
//!
//! Run:
//! ```text
//! cargo test --release --lib location_probe -- --ignored --nocapture
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
use crate::palette::Palette;
use crate::{font, hp, probe, render, sheet};

// --- fixed experiment parameters (everything but the center is held constant) ---

/// Fixed SplitMix64 seed (stated for reproducibility). Today's date as a u64.
const SEED: u64 = 20_260_621;
/// Candidates drawn.
const N: usize = 48;
/// Non-trivial sampling box: real ∈ [RE_LO, RE_HI], imag ∈ [IM_LO, IM_HI].
const RE_LO: f64 = -2.0;
const RE_HI: f64 = 0.7;
const IM_LO: f64 = -1.2;
const IM_HI: f64 = 1.2;
/// Scale = the decoration-profile workload's window span (`profile` subcommand,
/// the shallow seahorse-valley spiral). Held fixed; the center is the only free
/// variable.
const FRAME_WIDTH: f64 = 0.012;
/// Iteration budget = a normal render's default (`LocationArgs::maxiter`).
const MAXITER: u32 = 1000;
const BAILOUT: f64 = 1e6;
const SS: u32 = 2;
/// Candidate render resolution — the calibration regime (`calibrate`'s
/// `--candidate-width 1280 --supersample 2`), so the descriptor sees the same
/// input scale the corpus signatures were frozen against.
const CAND_W: u32 = 1280;
const CAND_H: u32 = 720; // 16:9
/// Sheet thumbnail width (height follows 16:9).
const THUMB_W: u32 = 256;
const THUMB_H: u32 = 144;
/// Equal per-scale EMD weights — the `calibrate` default (`--weights 1,1,1,1`).
const WEIGHTS: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

// --- near-monotone coloring (one fixed colormap, no per-candidate cycling) ---

/// Fixed colormap, pulled from `data/palettes/clean_colormaps.json`. Viridis:
/// perceptually uniform, near-monotone luminance, low churn — value→color reads
/// as location structure, not palette noise.
const COLORMAP: &str = "viridis";
const COLORMAPS_PATH: &str = "data/palettes/clean_colormaps.json";

/// Low fixed density: ~1 gradient cycle per 250 smooth-iter, so the visible
/// value range sweeps the gradient a small number of times instead of churning.
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

// --- provisional acceptance rule (reported, not enforced) ---

/// Middle-90% smooth-iter spread below this = flat exterior (nothing to map).
const ACCEPT_SPREAD_MIN: f64 = 20.0;
/// Interior (max-iter) fraction above this = mostly dead black.
const ACCEPT_INTERIOR_MAX: f64 = 0.85;
/// Corpus sparse percentile above this = sparser than most of the corpus.
const ACCEPT_SPARSE_PCT_MAX: f64 = 0.80;

/// Per-candidate recorded scalars.
struct Cand {
    index: usize,
    center: Complex<f64>,
    interior_frac: f64,
    smooth_min: f64,
    smooth_p5: f64,
    smooth_median: f64,
    smooth_mean: f64,
    smooth_p95: f64,
    smooth_max: f64,
    spread: f64,
    sparse_score: f64,
    sparse_pct: f64,
    nn_emd: f64,
    glitched_px: u64,
    accepted: bool,
}

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn location_probe() {
    run().expect("location probe");
}

fn run() -> Result<(), String> {
    let out_dir = Path::new("data/location_probe");
    let thumbs_dir = out_dir.join("thumbs");

    // Fixed colormap from the JSON library (no serde — tolerant hand scan).
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

    // Corpus descriptor: frozen bins + per-image signatures.
    let art = std::fs::read_to_string(energy::ARTIFACT_PATH)
        .map_err(|e| format!("read {}: {e}", energy::ARTIFACT_PATH))?;
    let (bins, corpus) = energy::parse_artifact(&art)?;
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    // Corpus distribution of the sparse scalar (s16 bin-0 fraction), sorted.
    let mut corpus_sparse: Vec<f64> = corpus.iter().map(|(_, s)| s.hist[0][0]).collect();
    corpus_sparse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "corpus: {} signatures (sparse scalar = s16 bin0 fraction)",
        corpus.len()
    );

    let prec = hp::prec_bits(CAND_W, FRAME_WIDTH);
    let mut rng = probe::SplitMix64(SEED);
    let mut cands: Vec<Cand> = Vec::with_capacity(N);
    let mut thumbs: Vec<RgbImage> = Vec::with_capacity(N);
    crate::ensure_parent_dir(thumbs_dir.join("x"))?;

    eprintln!(
        "rendering {N} candidates @ {CAND_W}x{CAND_H} ss{SS}, fw={FRAME_WIDTH}, maxiter={MAXITER} ..."
    );
    for i in 0..N {
        // Two draws per candidate, in deterministic candidate order.
        let re = RE_LO + rng.unit() * (RE_HI - RE_LO);
        let im = IM_LO + rng.unit() * (IM_HI - IM_LO);
        let center = Complex::new(re, im);

        let cre = BigFloat::from_f64(center.re, prec);
        let cim = BigFloat::from_f64(center.im, prec);
        let panel = probe::render_mandel_panel(
            &cre,
            &cim,
            center,
            FRAME_WIDTH,
            CAND_W,
            CAND_H,
            SS,
            MAXITER,
            BAILOUT,
            prec,
            trap,
            BackendChoice::F64,
        );

        // Pre-shade scalars straight off the sample buffer.
        let mut esc: Vec<f64> = Vec::new();
        let mut interior = 0usize;
        for s in &panel.buf.samples {
            if s.escaped {
                esc.push(s.smooth_iter);
            } else {
                interior += 1;
            }
        }
        let total = panel.buf.samples.len();
        let interior_frac = interior as f64 / total as f64;
        esc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |t: f64| -> f64 {
            if esc.is_empty() {
                f64::NAN
            } else {
                let idx = (t * (esc.len() - 1) as f64).round() as usize;
                esc[idx.min(esc.len() - 1)]
            }
        };
        let smooth_min = esc.first().copied().unwrap_or(f64::NAN);
        let smooth_max = esc.last().copied().unwrap_or(f64::NAN);
        let smooth_p5 = q(0.05);
        let smooth_median = q(0.5);
        let smooth_p95 = q(0.95);
        let smooth_mean = if esc.is_empty() {
            f64::NAN
        } else {
            esc.iter().sum::<f64>() / esc.len() as f64
        };
        let spread = smooth_p95 - smooth_p5;

        // Shade once → descriptor input + sheet thumbnail.
        let rgb = render::shade_and_downsample(
            &panel.buf.samples,
            CAND_W,
            CAND_H,
            SS,
            &palette,
            &params,
            panel.spacing,
        );
        let regions = region_energies(&rgb);
        let sig = bins.signature(&regions);
        let sparse_score = sig.hist[0][0];
        let sparse_pct = frac_le(&corpus_sparse, sparse_score);
        let nn_emd = corpus_sigs
            .iter()
            .map(|cs| distance(&sig, cs, &WEIGHTS))
            .fold(f64::INFINITY, f64::min);

        let accepted = spread >= ACCEPT_SPREAD_MIN
            && interior_frac <= ACCEPT_INTERIOR_MAX
            && sparse_pct <= ACCEPT_SPARSE_PCT_MAX;

        // Annotated thumbnail.
        let mut th = image::imageops::resize(&rgb, THUMB_W, THUMB_H, FilterType::Triangle);
        let white = Rgb([240u8, 240, 240]);
        font::draw_text(
            &mut th,
            &format!("{i:02} SPRD{:.0}", spread.max(0.0)),
            2,
            2,
            1,
            white,
            true,
        );
        font::draw_text(
            &mut th,
            &format!("INT{:.0}% SPRS{:.0}%", interior_frac * 100.0, sparse_pct * 100.0),
            2,
            12,
            1,
            white,
            true,
        );
        if accepted {
            // small green tick plate corner marker via text
            font::draw_text(&mut th, "OK", THUMB_W - 18, 2, 1, Rgb([120, 255, 120]), true);
        }
        th.save(thumbs_dir.join(format!("{i:02}.png")))
            .map_err(|e| format!("save thumb {i}: {e}"))?;
        thumbs.push(th);

        cands.push(Cand {
            index: i,
            center,
            interior_frac,
            smooth_min,
            smooth_p5,
            smooth_median,
            smooth_mean,
            smooth_p95,
            smooth_max,
            spread,
            sparse_score,
            sparse_pct,
            nn_emd,
            glitched_px: panel.buf.glitched_pixels,
            accepted,
        });

        eprintln!(
            "  #{i:02} c=({re:.5},{im:.5}) int={:.0}% spread={:.0} sparse={:.3}(p{:.0}) nn={:.3} {}",
            interior_frac * 100.0,
            spread.max(0.0),
            sparse_score,
            sparse_pct * 100.0,
            nn_emd,
            if accepted { "ACCEPT" } else { "reject" }
        );
    }

    let accepted_count = cands.iter().filter(|c| c.accepted).count();

    // Contact sheet (6 cols → 8 rows).
    let sheet = sheet::compose_grid(&thumbs, Some(6));
    let sheet_path = out_dir.join("contact_sheet.png");
    crate::ensure_parent_dir(&sheet_path)?;
    sheet
        .save(&sheet_path)
        .map_err(|e| format!("save sheet: {e}"))?;

    // JSON sidecar (hand-rolled).
    let json = build_json(&cands, corpus.len(), accepted_count);
    let json_path = out_dir.join("probe.json");
    std::fs::write(&json_path, json).map_err(|e| format!("write json: {e}"))?;

    eprintln!("\n=== location-source base-rate probe ===");
    eprintln!("seed={SEED}  N={N}  box re[{RE_LO},{RE_HI}] im[{IM_LO},{IM_HI}]");
    eprintln!("frame_width={FRAME_WIDTH}  maxiter={MAXITER}  ss={SS}  colormap={COLORMAP}");
    eprintln!("color: smooth, density=0.004, offset=0, interior=black");
    eprintln!(
        "provisional accept: spread>={ACCEPT_SPREAD_MIN} AND interior<={:.0}% AND sparse_pct<={:.0}%",
        ACCEPT_INTERIOR_MAX * 100.0,
        ACCEPT_SPARSE_PCT_MAX * 100.0
    );
    eprintln!(
        "ACCEPTED {accepted_count}/{N}  ->  implied draws-per-keeper ~= {:.1}",
        N as f64 / accepted_count.max(1) as f64
    );
    eprintln!("sheet : {}", sheet_path.display());
    eprintln!("json  : {}", json_path.display());
    eprintln!("thumbs: {}/NN.png", thumbs_dir.display());
    Ok(())
}

/// Fraction of a sorted slice `<= v` (the empirical percentile of `v`).
/// `pub(crate)` so probe 1 reuses the same percentile helper.
pub(crate) fn frac_le(sorted: &[f64], v: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    // partition_point: count of elements <= v.
    let c = sorted.partition_point(|&x| x <= v);
    c as f64 / sorted.len() as f64
}

/// Pull one named colormap's stops out of `clean_colormaps.json` without serde.
/// The stops array is `[[pos,[r,g,b]], ...]`; we bracket-match the array span,
/// flatten every numeric token in it, and chunk by 4 → `(pos, [r,g,b])`.
/// `pub(crate)` so probe 1 reuses the same colormap loader.
pub(crate) fn load_colormap(text: &str, name: &str) -> Result<Vec<(f64, [u8; 3])>, String> {
    let needle = format!("\"{name}\"");
    let np = text
        .find(&needle)
        .ok_or_else(|| format!("colormap '{name}' not found"))?;
    let sp = text[np..]
        .find("\"stops\"")
        .map(|p| p + np)
        .ok_or_else(|| format!("colormap '{name}': no stops"))?;
    let open = text[sp..]
        .find('[')
        .map(|p| p + sp)
        .ok_or("stops: no '['")?;
    // Bracket-match to find the end of the stops array.
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut end = open;
    for (k, &b) in bytes[open..].iter().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = open + k;
                    break;
                }
            }
            _ => {}
        }
    }
    let span = &text[open + 1..end];
    // Flatten numeric tokens.
    let mut nums: Vec<f64> = Vec::new();
    for tok in span.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')) {
        if tok.is_empty() {
            continue;
        }
        if let Ok(x) = tok.parse::<f64>() {
            nums.push(x);
        }
    }
    if nums.is_empty() || nums.len() % 4 != 0 {
        return Err(format!(
            "colormap '{name}': parsed {} numbers (not a multiple of 4)",
            nums.len()
        ));
    }
    let mut stops = Vec::with_capacity(nums.len() / 4);
    for ch in nums.chunks_exact(4) {
        let pos = ch[0];
        let rgb = [ch[1] as u8, ch[2] as u8, ch[3] as u8];
        stops.push((pos, rgb));
    }
    Ok(stops)
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

fn build_json(cands: &[Cand], corpus_size: usize, accepted: usize) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"probe\": \"location-source base-rate\",\n");
    let _ = writeln!(s, "  \"seed\": {SEED},");
    let _ = writeln!(
        s,
        "  \"box\": {{ \"re_lo\": {RE_LO}, \"re_hi\": {RE_HI}, \"im_lo\": {IM_LO}, \"im_hi\": {IM_HI} }},"
    );
    let _ = writeln!(s, "  \"frame_width\": {FRAME_WIDTH},");
    let _ = writeln!(s, "  \"maxiter\": {MAXITER},");
    let _ = writeln!(s, "  \"bailout\": {BAILOUT},");
    let _ = writeln!(s, "  \"supersample\": {SS},");
    let _ = writeln!(s, "  \"candidate_width\": {CAND_W},");
    let _ = writeln!(s, "  \"candidate_height\": {CAND_H},");
    let _ = writeln!(s, "  \"colormap\": \"{COLORMAP}\",");
    s.push_str("  \"color\": { \"channel\": \"smooth\", \"density\": 0.004, \"offset\": 0.0, \"interior\": \"black\" },\n");
    let _ = writeln!(s, "  \"weights\": [1, 1, 1, 1],");
    let _ = writeln!(s, "  \"corpus_size\": {corpus_size},");
    s.push_str("  \"sparse_scalar\": \"signature.hist[0][0] (s16 grid, bin-0 fraction; higher = sparser)\",\n");
    let _ = writeln!(
        s,
        "  \"accept_threshold\": {{ \"spread_min\": {ACCEPT_SPREAD_MIN}, \"interior_max\": {ACCEPT_INTERIOR_MAX}, \"sparse_pct_max\": {ACCEPT_SPARSE_PCT_MAX} }},"
    );
    let _ = writeln!(s, "  \"accepted_count\": {accepted},");
    let _ = writeln!(
        s,
        "  \"implied_draws_per_keeper\": {},",
        jnum(cands.len() as f64 / accepted.max(1) as f64)
    );
    s.push_str("  \"candidates\": [\n");
    for (k, c) in cands.iter().enumerate() {
        let comma = if k + 1 < cands.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{ \"index\": {}, \"center_re\": {}, \"center_im\": {}, \"interior_frac\": {}, \"smooth_min\": {}, \"smooth_p5\": {}, \"smooth_median\": {}, \"smooth_mean\": {}, \"smooth_p95\": {}, \"smooth_max\": {}, \"spread_p5_p95\": {}, \"sparse_score\": {}, \"sparse_pct\": {}, \"nn_emd\": {}, \"glitched_px\": {}, \"accepted\": {} }}{}",
            c.index,
            jnum(c.center.re),
            jnum(c.center.im),
            jnum(c.interior_frac),
            jnum(c.smooth_min),
            jnum(c.smooth_p5),
            jnum(c.smooth_median),
            jnum(c.smooth_mean),
            jnum(c.smooth_p95),
            jnum(c.smooth_max),
            jnum(c.spread),
            jnum(c.sparse_score),
            jnum(c.sparse_pct),
            jnum(c.nn_emd),
            c.glitched_px,
            c.accepted,
            comma
        );
    }
    s.push_str("  ]\n}\n");
    s
}
