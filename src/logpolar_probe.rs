//! **Throwaway diagnostic — log-polar characterizer at nucleus centers.**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Answers one question
//! (see `prompts/prompt-02-logpolar-characterizer.md`): **anchored at
//! nucleus-derived centers, does a log-polar readout cleanly recover hub-ness +
//! spiral pitch + fold number + self-similar scaling + R₂/R₄, and does it flag
//! *delicacy*'s X as R₂-strong?** If yes, it's worth wiring as a focus-candidate
//! score later. Diagnosis only — no production wiring, no gate, no quality claim.
//!
//! Division of labor (settled by prompt 1, which retired winding-as-finder and
//! kept reflection):
//!  - **Find** centers → the **nucleus field** (atom-domain/Newton stack reused as
//!    a scoring field; hubs at nucleus peaks, junctions at nucleus-pair midpoints).
//!  - **Characterize** each center → **log-polar transform about it**: a log-spiral
//!    arm → a straight pitched diagonal; n-fold rotation → angular periodicity
//!    2π/n; self-similar scaling → radial periodicity; Rₙ rotation → a pure cyclic
//!    shift by 2π/n along the angular axis (so R₂ = A(π), R₄ = A(π/2), where A is
//!    the energy-normalized angular autocorrelation).
//!
//! Substrate is the **same native smooth-iteration scalar** as `symmetry_probe`
//! (palette-independent), regenerated on the **identical deterministic bench
//! frames** (`symmetry_probe::bench_frames`). Reflection scalars are **consumed,
//! not rebuilt** (`symmetry_probe::reflection_axes`). Output: `data/logpolar_probe/`.
//!
//! Readouts are validated on **analytic synthetic fields with known answers**
//! (step 2) BEFORE the bench; if pitch/fold/scaling don't recover there, the whole
//! thing stops — the bench would mean nothing.
//!
//! Run: `cargo test --release --lib logpolar_probe -- --ignored --nocapture`.

use std::f64::consts::TAU;
use std::fmt::Write as _;

use astro_float::{BigFloat, RoundingMode};
use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{Trap, TrapShape};
use crate::cli::BackendChoice;
use crate::font;
use crate::navigate::{atom_candidates_spatial, newton_nucleus};
use crate::palette::{linear_to_srgb, Palette};
use crate::symmetry_probe as sp; // shared substrate + reflection (prompt 1)
use crate::{hp, probe, render};

const RM: RoundingMode = RoundingMode::ToEven;

// --- substrate (shared with symmetry_probe) ----------------------------------
const RW: u32 = sp::RW; // 1280
const RH: u32 = sp::RH; // 720
const MAXITER: u32 = sp::MAXITER; // 2000
const BAILOUT: f64 = sp::BAILOUT;

const OUT_DIR: &str = "data/logpolar_probe";
const REFS_DIR: &str = "data/symmetry_probe/refs"; // refs placed by prompt 1
const CMAP_FILE: &str = sp::CMAP_FILE;
const SEED_CMAP: &str = sp::SEED_CMAP; // twilight_shifted smooth preview
const SEQ: &str = "inferno"; // L strip / residual / curve background
const DIV: &str = "coolwarm"; // signed detail D strip

// --- nucleus field (centers) -------------------------------------------------
/// Only periods ≤ this are Newton-refined (low-period = the human-scale foci).
const NUC_PERIOD_CAP: u32 = 80;
/// Spatial dedup cell (px) for the broad nucleus scan.
const NUC_CELL_PX: u32 = 8;
/// Min separation between kept hub peaks, as a fraction of the frame **width**
/// (reused intent from `focus_heatmaps::PEAK_DIV_RADIUS_FRAC`) — forces spread.
const PEAK_DIV_RADIUS_FRAC: f64 = 0.15;
/// Up to this many hub peaks and junction midpoints are characterized per frame.
const HUB_CAP: usize = 4;
const JUNC_CAP: usize = 2;
/// A junction is the midpoint of a hub pair no farther apart than this (frame-
/// width fraction) — nearby pairs only (the X is a *neighbouring*-nucleus saddle).
const JUNC_MAX_PAIR_FRAC: f64 = 0.45;

// --- log-polar analysis grid -------------------------------------------------
const NU: usize = 200; // radial samples (u = log r)
const NV: usize = 512; // angular samples (v = θ)
/// The two (r_min, r_max) bands in **pixels** about the center — a scale choice
/// and a measure-trap, so we SHOW two rather than hardcode one. r_min skips the
/// saturated interior; r_max stays inside the frame for a near-center anchor.
const BANDS: [(f64, f64); 2] = [(4.0, 90.0), (10.0, 260.0)];
/// Detail field = band-pass of L: light denoise minus a broad background, so the
/// radial brightness ramp + DC wash → 0 while arms/folds/rings survive (the
/// high-pass is linear, so it preserves a pure sinusoid's autocorrelation exactly,
/// only scaling amplitude). Energy-normalized to unit RMS. flat → 0.
const SD_SMALL: f64 = 1.0;
const SD_BIG: f64 = 48.0;
/// Radon slope scan for pitch (slope = d(iv)/d(iu) in grid-index units).
const N_SLOPE: usize = 161;
/// Below this structure-presence the band is a flat/saturated plateau — the
/// autocorrelations self-correlate to ~1 and are NOT a real symmetry signal.
const PRESENCE_FLOOR: f64 = 0.02;
/// `A(2π/n)` must clear this for n to count as a genuine fold (so a 4-fold isn't
/// reported as merely 2-fold). Below it, the fold falls back to the argmax.
const FOLD_THRESH: f64 = 0.8;

// --- display -----------------------------------------------------------------
const DISP_W: u32 = 460;
const STRIP_W: u32 = 384; // v axis (angular) → horizontal
const STRIP_H: u32 = 150; // u axis (radial) → vertical
const CURVE_W: u32 = 300;
const CURVE_H: u32 = 96;
const TITLE_H: u32 = 14;
const PAD: u32 = 5;
const BG: Rgb<u8> = Rgb([16, 16, 18]);

// ===========================================================================
// Entry
// ===========================================================================

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn logpolar_probe() {
    run().expect("logpolar-probe");
}

fn run() -> Result<(), String> {
    crate::ensure_parent_dir(&format!("{OUT_DIR}/x"))?;
    let cmaps = std::fs::read_to_string(CMAP_FILE).map_err(|e| format!("read {CMAP_FILE}: {e}"))?;
    let pal_seed = sp::load_pal(&cmaps, SEED_CMAP)?;
    let pal_seq = sp::load_pal(&cmaps, SEQ)?;
    let pal_div = sp::load_pal(&cmaps, DIV)?;

    // --- Step 2 FIRST: validate the readouts on analytic fields with known
    //     pitch / fold / scaling. If they don't recover, stop — the bench means
    //     nothing. ---
    eprintln!("=== synthetic readout validation (primitive-in-isolation) ===");
    if !synthetic_validation(&pal_seq, &pal_div)? {
        return Err("synthetic log-polar readouts did NOT recover known pitch/fold/scaling — broken, stopping.".into());
    }
    eprintln!("  synthetics recovered cleanly — proceeding to the bench.\n");

    // --- Step 0/1/3/4: the bench (same deterministic frames as symmetry_probe) ---
    let benches = sp::bench_frames();
    eprintln!(
        "bench: {} frames @ {RW}x{RH} maxiter {MAXITER}; log-polar {NU}x{NV}; bands(px) {:?}",
        benches.len(),
        BANDS
    );
    let t0 = std::time::Instant::now();
    process_bench(&benches[0], &pal_seed, &pal_seq, &pal_div)?;
    let per = t0.elapsed().as_secs_f64();
    eprintln!(
        "  [1/{}] {} done in {:.1}s (≈{:.0}s for the rest)",
        benches.len(),
        benches[0].name,
        per,
        per * (benches.len() - 1) as f64
    );
    for (i, b) in benches.iter().enumerate().skip(1) {
        let t = std::time::Instant::now();
        process_bench(b, &pal_seed, &pal_seq, &pal_div)?;
        eprintln!("  [{}/{}] {} done in {:.1}s", i + 1, benches.len(), b.name, t.elapsed().as_secs_f64());
    }

    // --- Step 5: off-substrate reference sanity (luminance, labeled) ---
    reference_panel(&pal_seq, &pal_div)?;

    eprintln!(
        "\nlogpolar-probe done — panels under {OUT_DIR}/. READ: do nucleus centers land on the \
         visual hubs (step-0 overlay)? do the L/A(Δ)/radial readouts cleanly separate hub vs \
         junction, recover pitch/fold/scaling, and flag delicacy's X as R₂-strong? No quality claim."
    );
    Ok(())
}

// ===========================================================================
// Centers — nucleus field (sub-pixel)
// ===========================================================================

#[derive(Clone)]
struct Center {
    /// Sub-pixel image location (log-polar is acutely center-sensitive — keep the
    /// Newton-refined location, do NOT round to the pixel grid).
    fx: f64,
    fy: f64,
    kind: &'static str, // "hub" | "junc" (display tag only; the signature classifies)
    label: String,
    period: u32, // 0 for junctions
}

#[derive(Clone)]
struct Nuc {
    period: u32,
    fx: f64,
    fy: f64,
    weight: f64,
}

/// Full f64 render (all channels, incl. atom-domain) at the bench center. The
/// nucleus finder needs `atom_min`; the smooth scalar is taken from the same buffer.
fn render_full(center: Complex<f64>, fw: f64) -> render::SampleBuffer {
    let prec = hp::prec_bits(RW, fw);
    let cre = BigFloat::from_f64(center.re, prec);
    let cim = BigFloat::from_f64(center.im, prec);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    probe::render_mandel_panel(&cre, &cim, center, fw, RW, RH, 1, MAXITER, BAILOUT, prec, trap, BackendChoice::F64).buf
}

/// Newton-confirmed low-period nuclei of a rendered seed → sub-pixel image
/// locations. Reuses the parked navigation primitives as a scoring field (none is
/// chased). Mirrors `focus_heatmaps::confirmed_nuclei` but keeps the **sub-pixel**
/// location instead of rounding to a display pixel.
fn nuclei(center: Complex<f64>, fw: f64, buf: &render::SampleBuffer) -> Vec<Nuc> {
    let prec = hp::prec_bits(RW, fw) + 32;
    let cre = BigFloat::from_f64(center.re, prec);
    let cim = BigFloat::from_f64(center.im, prec);
    let half_w = fw * 0.5;
    let aspect = RH as f64 / RW as f64;
    let half_h = fw * aspect * 0.5;

    let cands = atom_candidates_spatial(buf, RW, RH, fw, MAXITER, NUC_CELL_PX);
    {
        let mut periods: Vec<u32> = cands.iter().map(|c| c.period).collect();
        periods.sort_unstable();
        let elig = periods.iter().filter(|&&p| (2..=NUC_PERIOD_CAP).contains(&p)).count();
        eprintln!(
            "      [nuclei] {} raw atom candidates, period range {:?}..{:?}, {} in [2,{NUC_PERIOD_CAP}]",
            periods.len(),
            periods.first(),
            periods.last(),
            elig
        );
    }
    cands
        .par_iter()
        .filter(|c| c.period >= 2 && c.period <= NUC_PERIOD_CAP)
        .filter_map(|c| {
            let guess_re = cre.add(&BigFloat::from_f64(c.dc_re, prec), prec, RM);
            let guess_im = cim.add(&BigFloat::from_f64(c.dc_im, prec), prec, RM);
            let nuc = newton_nucleus(&guess_re, &guess_im, c.period, fw, prec)?;
            let ndr = hp::to_f64(&nuc.re.sub(&cre, prec, RM));
            let ndi = hp::to_f64(&nuc.im.sub(&cim, prec, RM));
            if ndr.abs() > half_w || ndi.abs() > half_h {
                return None;
            }
            let fx = (ndr / fw + 0.5) * RW as f64;
            let fy = (0.5 - ndi / (fw * aspect)) * RH as f64;
            if !(fx.is_finite() && fy.is_finite()) {
                return None;
            }
            Some(Nuc { period: nuc.period, fx, fy, weight: 1.0 / (nuc.period as f64).sqrt() })
        })
        .collect()
}

/// Hub candidates = nucleus peaks, farthest-point suppressed (spread by ≥
/// `PEAK_DIV_RADIUS_FRAC` of the frame width). Junction candidates = midpoints of
/// nearby kept-hub pairs. No hub/junction pre-classification beyond the display
/// tag — the characterizer's signature is what actually decides.
fn centers(nucs: &[Nuc]) -> Vec<Center> {
    let min_sep = PEAK_DIV_RADIUS_FRAC * RW as f64;
    let sep2 = min_sep * min_sep;
    // greedy by weight (lowest period first), farthest-point suppressed
    let mut order: Vec<&Nuc> = nucs.iter().collect();
    order.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
    let mut hubs: Vec<&Nuc> = Vec::new();
    for n in order {
        if hubs.iter().all(|h| {
            let dx = h.fx - n.fx;
            let dy = h.fy - n.fy;
            dx * dx + dy * dy >= sep2
        }) {
            hubs.push(n);
            if hubs.len() >= HUB_CAP {
                break;
            }
        }
    }

    let mut out: Vec<Center> = hubs
        .iter()
        .enumerate()
        .map(|(i, h)| Center {
            fx: h.fx,
            fy: h.fy,
            kind: "hub",
            label: format!("H{}", i + 1),
            period: h.period,
        })
        .collect();

    // junctions: midpoints of nearby hub pairs, deduped by min separation.
    let pair_max = JUNC_MAX_PAIR_FRAC * RW as f64;
    let mut juncs: Vec<(f64, f64)> = Vec::new();
    for i in 0..hubs.len() {
        for j in (i + 1)..hubs.len() {
            let dx = hubs[i].fx - hubs[j].fx;
            let dy = hubs[i].fy - hubs[j].fy;
            if (dx * dx + dy * dy).sqrt() > pair_max {
                continue;
            }
            let mx = 0.5 * (hubs[i].fx + hubs[j].fx);
            let my = 0.5 * (hubs[i].fy + hubs[j].fy);
            if juncs.iter().all(|&(ex, ey)| {
                let ddx = ex - mx;
                let ddy = ey - my;
                ddx * ddx + ddy * ddy >= 0.25 * sep2
            }) {
                juncs.push((mx, my));
            }
        }
    }
    for (i, (mx, my)) in juncs.into_iter().take(JUNC_CAP).enumerate() {
        out.push(Center { fx: mx, fy: my, kind: "junc", label: format!("J{}", i + 1), period: 0 });
    }
    out
}

// ===========================================================================
// Log-polar sampler + detail field
// ===========================================================================

struct LogPolar {
    l: Vec<f64>, // raw smooth scalar sampled on the polar grid (NU rows × NV cols)
    d: Vec<f64>, // unit-RMS detail (band-pass) field
    /// Structure-presence: band-pass RMS / L's value spread. Near 0 ⇒ the band is
    /// a flat/saturated interior plateau, so the (unit-normalized) autocorrelations
    /// self-correlate to ~1 and MUST NOT be read as a real hub. The honest "flat→0".
    presence: f64,
}

/// Bilinearly sample `field` onto the polar grid about `(fx,fy)`: row `iu` = log r
/// over `[ln rmin, ln rmax]`, col `iv` = θ over `[0, 2π)`. Row-major `iu*NV + iv`.
fn logpolar_sample(field: &[f64], w: usize, h: usize, fx: f64, fy: f64, rmin: f64, rmax: f64) -> LogPolar {
    let lr0 = rmin.ln();
    let lr1 = rmax.ln();
    let mut l = vec![0.0; NU * NV];
    l.par_chunks_mut(NV).enumerate().for_each(|(iu, row)| {
        let u = lr0 + (lr1 - lr0) * iu as f64 / (NU - 1) as f64;
        let r = u.exp();
        for (iv, cell) in row.iter_mut().enumerate() {
            let th = iv as f64 / NV as f64 * TAU;
            let (s, c) = th.sin_cos();
            *cell = bilin(field, w, h, fx + r * c, fy + r * s);
        }
    });
    let (d, raw_rms) = detail_field(&l);
    let spread = (pctl(&l, 0.99) - pctl(&l, 0.01)).max(1e-12);
    LogPolar { l, d, presence: raw_rms / spread }
}

/// Band-pass detail of `L`: `blur(small) − blur(big)` (v-axis cyclic, u-axis
/// clamp), unit-RMS normalized for display/readouts. Returns `(D, raw band-pass
/// RMS)` — the raw RMS feeds the structure-presence gate (flat → ~0).
fn detail_field(l: &[f64]) -> (Vec<f64>, f64) {
    let lo = blur_uv(l, SD_SMALL);
    let hi = blur_uv(l, SD_BIG);
    let mut d: Vec<f64> = lo.iter().zip(&hi).map(|(a, b)| a - b).collect();
    let rms = (d.iter().map(|x| x * x).sum::<f64>() / (NU * NV) as f64).sqrt();
    let inv = if rms > 1e-30 { 1.0 / rms } else { 0.0 };
    for x in &mut d {
        *x *= inv;
    }
    (d, rms)
}

/// Separable Gaussian on the `NU×NV` polar grid: **wrap** along v (angular,
/// periodic), **clamp** along u (radial). σ≤0 → identity.
fn blur_uv(src: &[f64], sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let r = (3.0 * sigma).ceil() as i64;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut k: Vec<f64> = (-r..=r).map(|i| (-((i * i) as f64) * inv).exp()).collect();
    let s: f64 = k.iter().sum();
    for x in &mut k {
        *x /= s;
    }
    // along v (wrap)
    let mut tmp = vec![0.0; NU * NV];
    tmp.par_chunks_mut(NV).enumerate().for_each(|(iu, row)| {
        for (iv, cell) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let off = ki as i64 - r;
                let sv = (iv as i64 + off).rem_euclid(NV as i64) as usize;
                acc += kv * src[iu * NV + sv];
            }
            *cell = acc;
        }
    });
    // along u (clamp)
    let mut out = vec![0.0; NU * NV];
    out.par_chunks_mut(NV).enumerate().for_each(|(iu, row)| {
        for (iv, cell) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let off = ki as i64 - r;
                let su = (iu as i64 + off).clamp(0, NU as i64 - 1) as usize;
                acc += kv * tmp[su * NV + iv];
            }
            *cell = acc;
        }
    });
    out
}

// ===========================================================================
// Readouts
// ===========================================================================

struct Readout {
    l: Vec<f64>,
    d: Vec<f64>,
    /// Structure-presence (see `LogPolar`); below `PRESENCE_FLOOR` the readouts are
    /// flat-plateau artifacts, not real symmetry.
    presence: f64,
    /// Radon oriented-energy peak strength (hub/spiral coherence).
    hubness: f64,
    /// Physical pitch dθ/d(log r) at the dominant slope (0 = pure radial arms).
    pitch: f64,
    radon: Vec<f64>, // oriented-energy vs slope index
    ang: Vec<f64>,   // angular autocorr A(Δ), Δ index 0..NV
    r2: f64,         // A(π)
    r4: f64,         // A(π/2)
    fold: u32,       // best n in {2,3,4,5,6,8} by A(2π/n)
    fold_score: f64,
    rad: Vec<f64>,   // radial autocorr R(Δu), lag 0..NU/2
    ratio: f64,      // self-similar scaling ratio e^{Δu*} at the radial peak
    scale_strength: f64,
}

fn compute_readout(field: &[f64], w: usize, h: usize, fx: f64, fy: f64, rmin: f64, rmax: f64) -> Readout {
    let lp = logpolar_sample(field, w, h, fx, fy, rmin, rmax);
    let ang = angular_autocorr(&lp.d);
    let rad = radial_autocorr(&lp.d, NU / 2);

    let smax = 2.0 * NV as f64 / NU as f64;
    let slopes: Vec<f64> = (0..N_SLOPE)
        .map(|i| -smax + 2.0 * smax * i as f64 / (N_SLOPE - 1) as f64)
        .collect();
    let (bi, hubness, radon) = radon_pitch(&lp.d, &slopes);
    let ku = (rmax.ln() - rmin.ln()) / (NU - 1) as f64;
    let kv = TAU / NV as f64;
    let pitch = kv * slopes[bi] / ku;

    let r2 = ang[NV / 2];
    let r4 = ang[NV / 4];
    // Fold = the LARGEST n whose A(2π/n) clears the threshold (an n-fold pattern is
    // also n/2-fold, so a plain argmax of A(2π/n) ties n vs 2n; the *fundamental*
    // fold is the finest one that still holds). Fall back to argmax if none clears.
    let folds = [2u32, 3, 4, 5, 6, 8];
    let argmax = folds
        .iter()
        .map(|&n| (n, ang[NV / n as usize]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let (fold, fold_score) = folds
        .iter()
        .rev()
        .map(|&n| (n, ang[NV / n as usize]))
        .find(|&(_, a)| a > FOLD_THRESH)
        .unwrap_or(argmax);

    // self-similar scaling: skip tiny radial lags (ratio < e^0.05), take the peak.
    let minlag = ((0.05 / ku).round() as usize).max(3);
    let (blag, scale_strength) = (minlag..=NU / 2)
        .map(|lag| (lag, rad[lag]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((minlag, 0.0));
    let ratio = (ku * blag as f64).exp();

    Readout {
        l: lp.l,
        d: lp.d,
        presence: lp.presence,
        hubness,
        pitch,
        radon,
        ang,
        r2,
        r4,
        fold,
        fold_score,
        rad,
        ratio,
        scale_strength,
    }
}

/// Angular autocorrelation `A(Δ) = Σ D·D(u, v+Δ) / Σ D²`, cyclic in v. `A(0)=1`.
fn angular_autocorr(d: &[f64]) -> Vec<f64> {
    let e: f64 = d.iter().map(|x| x * x).sum::<f64>().max(1e-30);
    (0..NV)
        .into_par_iter()
        .map(|lag| {
            let mut acc = 0.0;
            for iu in 0..NU {
                let base = iu * NV;
                for iv in 0..NV {
                    acc += d[base + iv] * d[base + (iv + lag) % NV];
                }
            }
            acc / e
        })
        .collect()
}

/// Radial autocorrelation along u (normalized cross-correlation over the overlap).
/// A peak at `Δu>0` ⇒ structure repeats under scaling by `e^{Δu}`.
fn radial_autocorr(d: &[f64], maxlag: usize) -> Vec<f64> {
    (0..=maxlag)
        .into_par_iter()
        .map(|lag| {
            let (mut num, mut ea, mut eb) = (0.0, 0.0, 0.0);
            for iu in 0..(NU - lag) {
                let ba = iu * NV;
                let bb = (iu + lag) * NV;
                for iv in 0..NV {
                    let a = d[ba + iv];
                    let b = d[bb + iv];
                    num += a * b;
                    ea += a * a;
                    eb += b * b;
                }
            }
            if ea > 1e-30 && eb > 1e-30 {
                num / (ea.sqrt() * eb.sqrt())
            } else {
                0.0
            }
        })
        .collect()
}

/// Discrete Radon over diagonal slopes: project D onto intercept bins along lines
/// `iv = c + s·iu` (c = `(iv − s·iu) mod NV`), oriented energy = `Σ_c P_s(c)² /
/// (Σ D² · NU)`. A straight pitched diagonal (log-spiral arm) concentrates energy
/// → a peak at its slope. Returns `(best slope index, peak strength, full curve)`.
fn radon_pitch(d: &[f64], slopes: &[f64]) -> (usize, f64, Vec<f64>) {
    let e: f64 = d.iter().map(|x| x * x).sum::<f64>().max(1e-30);
    let curve: Vec<f64> = slopes
        .par_iter()
        .map(|&s| {
            let mut proj = vec![0.0f64; NV];
            for iu in 0..NU {
                let base = iu * NV;
                let shift = s * iu as f64;
                for iv in 0..NV {
                    let c = (iv as f64 - shift).rem_euclid(NV as f64);
                    let ci = (c.round() as usize) % NV;
                    proj[ci] += d[base + iv];
                }
            }
            let ss: f64 = proj.iter().map(|p| p * p).sum();
            ss / (e * NU as f64)
        })
        .collect();
    let (bi, bv) = curve
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap();
    (bi, bv, curve)
}

// ===========================================================================
// Step 2 — synthetic validation (known pitch / fold / scaling)
// ===========================================================================

fn synthetic_validation(pal_seq: &Palette, pal_div: &Palette) -> Result<bool, String> {
    let gs = 257usize;
    let c = (gs as f64 - 1.0) * 0.5;
    let (rmin, rmax) = (4.0f64, 120.0f64);
    let ku = (rmax.ln() - rmin.ln()) / (NU - 1) as f64;

    // --- log-spiral, known pitch dθ/dlogr = Q/P = 3 ---
    let spiral = synth(gs, |r, th| (1.0 * th - 3.0 * r.ln()).cos());
    let r_sp = compute_readout(&spiral, gs, gs, c, c, rmin, rmax);
    let pitch_ok = (r_sp.pitch.abs() - 3.0).abs() / 3.0 < 0.2;
    eprintln!("  log-spiral  (expect pitch dθ/dlogr ≈ ±3.0): recovered {:+.3}  hubness {:.3}  [{}]", r_sp.pitch, r_sp.hubness, pass(pitch_ok));

    // --- 4-fold rosette → A(π/2) high, fold=4 ---
    let ros4 = synth(gs, |_r, th| (4.0 * th).cos());
    let r4r = compute_readout(&ros4, gs, gs, c, c, rmin, rmax);
    let fold4_ok = r4r.fold == 4 && r4r.r4 > 0.6;
    eprintln!("  rosette n=4 (expect fold 4, A(π/2)>0.6): fold {} A(π/2)={:.3} A(π)={:.3}  [{}]", r4r.fold, r4r.r4, r4r.r2, pass(fold4_ok));

    // --- 2-fold rosette → A(π) high, fold=2 ---
    let ros2 = synth(gs, |_r, th| (2.0 * th).cos());
    let r2r = compute_readout(&ros2, gs, gs, c, c, rmin, rmax);
    let fold2_ok = r2r.fold == 2 && r2r.r2 > 0.6;
    eprintln!("  rosette n=2 (expect fold 2, A(π)>0.6):   fold {} A(π)={:.3} A(π/2)={:.3}  [{}]", r2r.fold, r2r.r2, r2r.r4, pass(fold2_ok));

    // --- self-similar rings, known ratio 1.5 ---
    let kk = TAU / 1.5f64.ln();
    let rings = synth(gs, |r, _th| (kk * r.ln()).cos());
    let rr = compute_readout(&rings, gs, gs, c, c, rmin, rmax);
    let ratio_ok = (rr.ratio - 1.5).abs() / 1.5 < 0.15;
    eprintln!("  rings ratio (expect e^Δu ≈ 1.5):          recovered {:.4}  strength {:.3}  [{}]  (Δu_phys≈{:.3})", rr.ratio, rr.scale_strength, pass(ratio_ok), 1.5f64.ln());

    // panel: each synthetic's D strip + its diagnostic curve.
    let mark_ang = vec![(NV / 2, Rgb([90, 130, 255])), (NV / 4, Rgb([255, 80, 80]))];
    let mark_rad: Vec<(usize, Rgb<u8>)> = vec![(((1.5f64.ln()) / ku).round() as usize, Rgb([120, 255, 120]))];
    let row_sp = candidate_block("spiral pitch=3", &r_sp, pal_seq, pal_div, &mark_ang, &mark_rad);
    let row_4 = candidate_block("rosette n=4", &r4r, pal_seq, pal_div, &mark_ang, &mark_rad);
    let row_2 = candidate_block("rosette n=2", &r2r, pal_seq, pal_div, &mark_ang, &mark_rad);
    let row_ri = candidate_block("rings r=1.5", &rr, pal_seq, pal_div, &mark_ang, &mark_rad);
    let body = vstack(&[row_sp, row_4, row_2, row_ri]);
    let pass_all = pitch_ok && fold4_ok && fold2_ok && ratio_ok;
    let panel = vstack(&[banner(body.width(), &format!("SYNTHETIC LOG-POLAR VALIDATION  [pass={pass_all}]  pitch / fold / R2 R4 / scaling")), body]);
    save(&panel, &format!("{OUT_DIR}/synthetic_validation.png"))?;
    Ok(pass_all)
}

fn pass(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

/// "[FLAT]" tag when structure-presence is below the floor (readouts untrustworthy).
fn flat_flag(presence: f64) -> &'static str {
    if presence < PRESENCE_FLOOR {
        " [FLAT]"
    } else {
        ""
    }
}

/// Build a `gs×gs` analytic field from `f(r, θ)` about the grid center; the
/// immediate center (r<rmin guard) is zeroed.
fn synth<F: Fn(f64, f64) -> f64 + Sync>(gs: usize, f: F) -> Vec<f64> {
    let c = (gs as f64 - 1.0) * 0.5;
    (0..gs * gs)
        .into_par_iter()
        .map(|i| {
            let x = (i % gs) as f64 - c;
            let y = (i / gs) as f64 - c;
            let r = (x * x + y * y).sqrt();
            if r < 2.0 {
                0.0
            } else {
                f(r, y.atan2(x))
            }
        })
        .collect()
}

// ===========================================================================
// Per-bench frame
// ===========================================================================

fn process_bench(b: &sp::Bench, pal_seed: &Palette, pal_seq: &Palette, pal_div: &Palette) -> Result<(), String> {
    let w = RW as usize;
    let h = RH as usize;
    // Full f64 render (every channel live, incl. atom-domain) — the nucleus finder
    // reads `atom_min`, which the smooth fast path (`sp::render_seed`) omits. The
    // native smooth-iteration scalar is identical (smooth_iter is trap/atom-agnostic).
    let buf = render_full(b.center, b.fw);
    let scalar = sp::smooth_scalar(&buf);

    // --- centers from the nucleus field ---
    let nucs = nuclei(b.center, b.fw, &buf);
    let cents = centers(&nucs);
    eprintln!(
        "    {}: {} nuclei (period≤{NUC_PERIOD_CAP}) → {} centers ({} hub, {} junc)",
        b.name,
        nucs.len(),
        cents.len(),
        cents.iter().filter(|c| c.kind == "hub").count(),
        cents.iter().filter(|c| c.kind == "junc").count()
    );

    // --- step 0: candidate overlay on the smooth preview (finder sanity) ---
    let preview = render::shade_and_downsample(&buf.samples, RW, RH, 1, pal_seed, &sp::preview_params(), b.fw / RW as f64);
    let big_w = 900u32;
    let big_h = big_w * RH / RW;
    let mut overlay = downscale_rgb(&preview, big_w, big_h);
    let sc = big_w as f64 / RW as f64;
    for cn in &cents {
        let col = if cn.kind == "hub" { Rgb([255, 70, 70]) } else { Rgb([90, 160, 255]) };
        draw_marker(&mut overlay, cn.fx * sc, cn.fy * sc, 7.0, col);
        font::draw_text(&mut overlay, &cn.label, (cn.fx * sc + 8.0) as u32, (cn.fy * sc - 4.0) as u32, 1, col, true);
    }
    let overlay = titled(overlay, &format!("{} smooth + nucleus centers (red=hub H#, blue=junc J#)  [land on visual hubs?]", b.name));

    // --- reflection (consumed, not rebuilt) ---
    let tf = sp::structure_tensor(&scalar, w, h, sp::SCALES[0].0, sp::SCALES[0].1);
    let detail: Vec<f64> = tf.energy.iter().map(|&e| e.sqrt()).collect();
    let (axes, rh, rv) = sp::reflection_axes(&detail, w, h);
    let rmax = pctl(&rh, 0.99).max(pctl(&rv, 0.99)).max(1e-12);
    let t_rh = field_tile(&rh, w, h, 0.0, rmax, pal_seq, "frame-axis H reflection residual |D-D.M|", DISP_W);
    let t_rv = field_tile(&rv, w, h, 0.0, rmax, pal_seq, "frame-axis V reflection residual |D-D.M|", DISP_W);
    let refl_row = hstack(&[t_rh, t_rv]);
    eprintln!("    {} frame reflection[H,V,diag1,diag2]={:.3},{:.3},{:.3},{:.3}", b.name, axes[0], axes[1], axes[2], axes[3]);

    // --- step 3: characterize each center over both bands ---
    let mark_ang = vec![(NV / 2, Rgb([90, 130, 255])), (NV / 4, Rgb([255, 80, 80]))];
    let mut cand_blocks: Vec<RgbImage> = Vec::new();
    let mut jbuf = String::from("[");
    for (ci, cn) in cents.iter().enumerate() {
        let mut band_rows: Vec<RgbImage> = Vec::new();
        for (bi, &(rmin, rmax_px)) in BANDS.iter().enumerate() {
            let ro = compute_readout(&scalar, w, h, cn.fx, cn.fy, rmin, rmax_px);
            eprintln!(
                "      {} {} band{} (r {:.0}-{:.0}px): presence {:.3}{} hubness {:.3} pitch {:+.2} | fold {} (A={:.2}) R2 {:.2} R4 {:.2} | scale x{:.3} ({:.2})",
                b.name, cn.label, bi, rmin, rmax_px, ro.presence, flat_flag(ro.presence), ro.hubness, ro.pitch, ro.fold, ro.fold_score, ro.r2, ro.r4, ro.ratio, ro.scale_strength
            );
            let ku = (rmax_px.ln() - rmin.ln()) / (NU - 1) as f64;
            let mark_rad = scale_marks(&ro, ku);
            let title = format!(
                "{} ({}, P{}) band{} r{:.0}-{:.0}px | pres {:.2}{} hub {:.2} pitch {:+.2} fold {} R2 {:.2} R4 {:.2} x{:.2}",
                cn.label, cn.kind, cn.period, bi, rmin, rmax_px, ro.presence, flat_flag(ro.presence), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4, ro.ratio
            );
            band_rows.push(candidate_block(&title, &ro, pal_seq, pal_div, &mark_ang, &mark_rad));
            if bi == 0 {
                let _ = write!(
                    jbuf,
                    "{}{{\"label\":\"{}\",\"kind\":\"{}\",\"period\":{},\"fx\":{:.2},\"fy\":{:.2},\"hubness\":{:.4},\"pitch\":{:.4},\"fold\":{},\"r2\":{:.4},\"r4\":{:.4},\"ratio\":{:.4}}}",
                    if ci > 0 { "," } else { "" },
                    cn.label, cn.kind, cn.period, cn.fx, cn.fy, ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4, ro.ratio
                );
            }
        }
        cand_blocks.push(vstack(&band_rows));
    }
    jbuf.push(']');

    // --- compose the frame panel ---
    let dir = format!("{OUT_DIR}/{}", b.name);
    crate::ensure_parent_dir(&format!("{dir}/x"))?;
    let mut rows = vec![overlay, refl_row];
    rows.extend(cand_blocks);
    let body = vstack(&rows);
    let head = banner(
        body.width(),
        &format!(
            "LOG-POLAR CHARACTERIZER  {}  ({})  center ({:.6},{:.6}) fw {:.3e}  refl[H,V]={:.2},{:.2}  [DIAGNOSIS ONLY]",
            b.name, b.note, b.center.re, b.center.im, b.fw, axes[0], axes[1]
        ),
    );
    let panel = vstack(&[head, body]);
    save(&panel, &format!("{dir}/panel.png"))?;

    let mut j = String::new();
    let _ = write!(
        j,
        "{{\n  \"name\": \"{}\", \"center_re\": {:.12e}, \"center_im\": {:.12e}, \"frame_width\": {:.6e},\n  \"reflection\": {{ \"h\": {:.5}, \"v\": {:.5}, \"diag1\": {:.5}, \"diag2\": {:.5} }},\n  \"n_nuclei\": {}, \"bands_px\": {:?},\n  \"centers\": {}\n}}\n",
        b.name, b.center.re, b.center.im, b.fw, axes[0], axes[1], axes[2], axes[3], nucs.len(), BANDS, jbuf
    );
    std::fs::write(format!("{dir}/logpolar.json"), &j).map_err(|e| format!("write json: {e}"))?;
    Ok(())
}

/// The green radial-autocorr marker at the detected self-similar lag.
fn scale_marks(ro: &Readout, ku: f64) -> Vec<(usize, Rgb<u8>)> {
    let lag = (ro.ratio.ln() / ku).round() as usize;
    vec![(lag.min(NU / 2), Rgb([120, 255, 120]))]
}

/// One candidate's display block: [L strip · D strip · A(Δ) curve · radial curve],
/// titled with the scalar scores.
fn candidate_block(
    title: &str,
    ro: &Readout,
    pal_seq: &Palette,
    pal_div: &Palette,
    mark_ang: &[(usize, Rgb<u8>)],
    mark_rad: &[(usize, Rgb<u8>)],
) -> RgbImage {
    let lmax = pctl(&ro.l, 0.99).max(1e-12);
    let lmin = pctl(&ro.l, 0.01);
    let t_l = strip_tile(&ro.l, lmin, lmax, pal_seq, "L(u,v)  u=logr ↑  v=θ →");
    let dmax = pctl(&ro.d.iter().map(|x| x.abs()).collect::<Vec<_>>(), 0.99).max(1e-12);
    let t_d = strip_tile(&ro.d, -dmax, dmax, pal_div, "detail D(u,v)");
    let t_a = curve_tile(&ro.ang, -0.4, 1.0, "angular A(Δ): R2=blue(π) R4=red(π/2)", mark_ang);
    let t_r = curve_tile(&ro.rad, -0.4, 1.0, "radial autocorr R(Δu): scale=green", mark_rad);
    // Radon oriented-energy vs slope: peak = pitch (orange), s=0 radial-arm (grey).
    let rmax_v = ro.radon.iter().cloned().fold(1e-12, f64::max);
    let peak_i = ro.radon.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
    let mark_radon = vec![((N_SLOPE - 1) / 2, Rgb([110, 110, 120])), (peak_i, Rgb([255, 160, 60]))];
    let t_p = curve_tile(&ro.radon, 0.0, rmax_v, "Radon oriented-E vs slope: pitch=orange s0=grey", &mark_radon);
    let row = hstack(&[t_l, t_d, t_a, t_r, t_p]);
    titled(row, title)
}

// ===========================================================================
// Step 5 — reference sanity panel (off native substrate, luminance)
// ===========================================================================

fn reference_panel(pal_seq: &Palette, pal_div: &Palette) -> Result<(), String> {
    // (file, fractional center (human-marked approx, tunable), expectation)
    let refs = [
        ("delicacy", "delicacy.jpg", 0.520, 0.460, "central X — expect R2 strong; fold may read 2 not 4"),
        ("helping-hands", "helping-hands-25.jpg", 0.120, 0.330, "spiral hub — expect strong pitch peak, R2 modest"),
    ];
    let mark_ang = vec![(NV / 2, Rgb([90, 130, 255])), (NV / 4, Rgb([255, 80, 80]))];
    let mut blocks: Vec<RgbImage> = Vec::new();
    let mut any = false;
    for (name, file, fcx, fcy, expect) in refs {
        let path = format!("{REFS_DIR}/{file}");
        let Some((lum, w, h)) = load_luma(&path) else {
            eprintln!("  reference {path} absent — skipping (graceful).");
            continue;
        };
        any = true;
        let fx = fcx * w as f64;
        let fy = fcy * h as f64;
        let (rmin, rmax) = (6.0, (w.min(h) as f64) * 0.45);
        let ro = compute_readout(&lum, w, h, fx, fy, rmin, rmax);
        let ku = (rmax.ln() - rmin.ln()) / (NU - 1) as f64;
        eprintln!(
            "  ref {name} [SANITY ONLY, palette-contaminated]: center frac ({fcx},{fcy})  presence {:.3}{} hubness {:.3} pitch {:+.2} fold {} R2 {:.3} R4 {:.3} x{:.3}  ({expect})",
            ro.presence, flat_flag(ro.presence), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4, ro.ratio
        );
        let block = candidate_block(
            &format!("{name} [SANITY] pres {:.2}{} hub {:.2} pitch {:+.2} fold {} R2 {:.2} R4 {:.2}", ro.presence, flat_flag(ro.presence), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4),
            &ro,
            pal_seq,
            pal_div,
            &mark_ang,
            &scale_marks(&ro, ku),
        );
        // a small luminance preview with the marked center
        let pw = 300u32;
        let ph = (pw as f64 * h as f64 / w as f64) as u32;
        let lmax = pctl(&lum, 0.99).max(1e-6);
        let mut prev = colorize(&lum, w, h, 0.0, lmax, pal_seq);
        prev = downscale_rgb(&prev, pw, ph);
        draw_marker(&mut prev, fx * pw as f64 / w as f64, fy * ph as f64 / h as f64, 6.0, Rgb([255, 255, 90]));
        let prev = titled(prev, &format!("{name} luminance + center [SANITY]"));
        blocks.push(vstack(&[prev, block]));
    }
    if !any {
        eprintln!("  no reference images under {REFS_DIR}/ — ref panel skipped.");
        return Ok(());
    }
    let body = hstack(&blocks);
    let head = banner(body.width(), "REFERENCE SANITY  (luminance, palette-contaminated, OFF native substrate — sanity only)");
    save(&vstack(&[head, body]), &format!("{OUT_DIR}/reference_sanity.png"))?;
    Ok(())
}

fn load_luma(path: &str) -> Option<(Vec<f64>, usize, usize)> {
    let img = image::open(path).ok()?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let lum: Vec<f64> = img
        .pixels()
        .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
        .collect();
    Some((lum, w, h))
}

// ===========================================================================
// Visualization (self-contained — generic tiles, strips, curves)
// ===========================================================================

fn lut_rgb(pal: &Palette, t: f64) -> Rgb<u8> {
    let lin = pal.lookup_linear(t.clamp(0.0, 1.0));
    Rgb([
        (linear_to_srgb(lin[0]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[1]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[2]) * 255.0 + 0.5) as u8,
    ])
}

fn colorize(field: &[f64], w: usize, h: usize, vmin: f64, vmax: f64, pal: &Palette) -> RgbImage {
    let span = (vmax - vmin).max(1e-12);
    let mut img = RgbImage::new(w as u32, h as u32);
    for (i, px) in img.pixels_mut().enumerate() {
        *px = lut_rgb(pal, (field[i] - vmin) / span);
    }
    img
}

/// A `w×h` field colorized then aspect-fit to `target_w` with a title bar.
fn field_tile(field: &[f64], w: usize, h: usize, vmin: f64, vmax: f64, pal: &Palette, title: &str, target_w: u32) -> RgbImage {
    let img = colorize(field, w, h, vmin, vmax, pal);
    let dw = target_w.min(w as u32);
    let dh = ((h as u32 * dw) / w as u32).max(1);
    titled(downscale_rgb(&img, dw, dh), title)
}

/// A log-polar strip (`NU` rows × `NV` cols) → v horizontal, u vertical, fixed
/// `STRIP_W×STRIP_H`, titled.
fn strip_tile(field: &[f64], vmin: f64, vmax: f64, pal: &Palette, title: &str) -> RgbImage {
    let img = colorize(field, NV, NU, vmin, vmax, pal); // width=NV, height=NU
    titled(downscale_rgb(&img, STRIP_W, STRIP_H), title)
}

/// A 1-D curve over `[ymin,ymax]` with a zero baseline and vertical lag markers.
fn curve_tile(ys: &[f64], ymin: f64, ymax: f64, title: &str, marks: &[(usize, Rgb<u8>)]) -> RgbImage {
    let (w, h) = (CURVE_W, CURVE_H);
    let mut img = RgbImage::from_pixel(w, h, Rgb([24, 24, 28]));
    let span = (ymax - ymin).max(1e-12);
    let yof = |v: f64| -> i64 {
        let t = ((v - ymin) / span).clamp(0.0, 1.0);
        ((1.0 - t) * (h as f64 - 1.0)).round() as i64
    };
    // zero baseline
    if ymin < 0.0 && ymax > 0.0 {
        let y0 = yof(0.0);
        for x in 0..w {
            put(&mut img, x as i64, y0, Rgb([70, 70, 80]));
        }
    }
    // markers (vertical lines)
    for &(idx, col) in marks {
        let x = (idx as f64 / ys.len().max(1) as f64 * w as f64).round() as i64;
        for y in 0..h {
            put(&mut img, x, y as i64, col);
        }
    }
    // curve
    let n = ys.len().max(1);
    let mut prev: Option<(i64, i64)> = None;
    for (i, &v) in ys.iter().enumerate() {
        let x = (i as f64 / n as f64 * w as f64).round() as i64;
        let y = yof(v);
        if let Some((px, py)) = prev {
            // connect with a vertical span for legibility
            let (a, b) = if py <= y { (py, y) } else { (y, py) };
            for yy in a..=b {
                put(&mut img, x, yy, Rgb([245, 230, 150]));
            }
            let _ = px;
        }
        put(&mut img, x, y, Rgb([255, 245, 180]));
        prev = Some((x, y));
    }
    titled(img, title)
}

fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn downscale_rgb(src: &RgbImage, dw: u32, dh: u32) -> RgbImage {
    let (sw, sh) = (src.width(), src.height());
    if dw == sw && dh == sh {
        return src.clone();
    }
    let mut out = RgbImage::new(dw, dh);
    for y in 0..dh {
        let sy0 = (y as u64 * sh as u64 / dh as u64) as u32;
        let sy1 = (((y + 1) as u64 * sh as u64 / dh as u64) as u32).max(sy0 + 1).min(sh);
        for x in 0..dw {
            let sx0 = (x as u64 * sw as u64 / dw as u64) as u32;
            let sx1 = (((x + 1) as u64 * sw as u64 / dw as u64) as u32).max(sx0 + 1).min(sw);
            let mut acc = [0u32; 3];
            let mut n = 0u32;
            for yy in sy0..sy1 {
                for xx in sx0..sx1 {
                    let p = src.get_pixel(xx, yy);
                    acc[0] += p[0] as u32;
                    acc[1] += p[1] as u32;
                    acc[2] += p[2] as u32;
                    n += 1;
                }
            }
            let n = n.max(1);
            out.put_pixel(x, y, Rgb([(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]));
        }
    }
    out
}

fn titled(img: RgbImage, title: &str) -> RgbImage {
    let w = img.width();
    let mut out = RgbImage::from_pixel(w.max(8), img.height() + TITLE_H, BG);
    probe::blit(&mut out, &img, 0, TITLE_H);
    font::draw_text(&mut out, &title.to_uppercase(), 2, 3, 1, Rgb([235, 235, 235]), true);
    out
}

fn banner(width: u32, text: &str) -> RgbImage {
    let mut img = RgbImage::from_pixel(width.max(8), 18, Rgb([10, 10, 12]));
    font::draw_text(&mut img, &text.to_uppercase(), 4, 4, 1, Rgb([255, 230, 150]), true);
    img
}

fn hstack(tiles: &[RgbImage]) -> RgbImage {
    let width: u32 = tiles.iter().map(|t| t.width()).sum::<u32>() + PAD * (tiles.len() as u32 + 1);
    let height = tiles.iter().map(|t| t.height()).max().unwrap_or(0) + 2 * PAD;
    let mut img = RgbImage::from_pixel(width, height, BG);
    let mut x = PAD;
    for t in tiles {
        probe::blit(&mut img, t, x, PAD);
        x += t.width() + PAD;
    }
    img
}

fn vstack(tiles: &[RgbImage]) -> RgbImage {
    let width = tiles.iter().map(|t| t.width()).max().unwrap_or(0) + 2 * PAD;
    let height: u32 = tiles.iter().map(|t| t.height()).sum::<u32>() + PAD * (tiles.len() as u32 + 1);
    let mut img = RgbImage::from_pixel(width, height, BG);
    let mut y = PAD;
    for t in tiles {
        probe::blit(&mut img, t, PAD, y);
        y += t.height() + PAD;
    }
    img
}

/// A ring (radius `r` px) + center dot at `(cx,cy)` with a dark halo, clipped.
fn draw_marker(img: &mut RgbImage, cx: f64, cy: f64, r: f64, col: Rgb<u8>) {
    let x0 = (cx - r - 2.0).floor() as i64;
    let x1 = (cx + r + 2.0).ceil() as i64;
    let y0 = (cy - r - 2.0).floor() as i64;
    let y1 = (cy + r + 2.0).ceil() as i64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cx).powi(2)) + ((y as f64 - cy).powi(2))).sqrt();
            if (d - r).abs() <= 1.6 {
                put(img, x, y, Rgb([0, 0, 0]));
            }
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cx).powi(2)) + ((y as f64 - cy).powi(2))).sqrt();
            if (d - r).abs() <= 0.8 {
                put(img, x, y, col);
            }
        }
    }
    put(img, cx as i64, cy as i64, col);
}

fn save(img: &RgbImage, path: &str) -> Result<(), String> {
    crate::ensure_parent_dir(path)?;
    img.save(path).map_err(|e| format!("save {path}: {e}"))
}

fn bilin(f: &[f64], w: usize, h: usize, x: f64, y: f64) -> f64 {
    let xf = x.clamp(0.0, (w - 1) as f64);
    let yf = y.clamp(0.0, (h - 1) as f64);
    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = xf - x0 as f64;
    let ty = yf - y0 as f64;
    let a = f[y0 * w + x0] * (1.0 - tx) + f[y0 * w + x1] * tx;
    let b = f[y1 * w + x0] * (1.0 - tx) + f[y1 * w + x1] * tx;
    a * (1.0 - ty) + b * ty
}

fn pctl(f: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = f.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize]
}
