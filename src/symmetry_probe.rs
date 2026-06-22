//! **Throwaway diagnostic — symmetry probe (winding + reflection).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Answers one question
//! (see `prompts/prompt-01-symmetry-probe.md`): **can a coherence-weighted
//! winding integral on a structure-tensor director field both FIND and
//! SIGN-CLASSIFY symmetry centers on native hub-bearing smooth-iteration fields,
//! while staying ~0 on empty grey?** This replaces the prior Pearson rotational
//! probe (which scored empty backgrounds ~1.0 — see
//! `focus-heatmaps-three-field.md` / `focus_heatmaps.rs`, *not* extended here).
//!
//! The pipeline, per bench frame:
//!  1. Render the **native smooth-iteration scalar** (palette-independent), fast
//!     smooth path (no trap/DE), interior = max-escape plateau.
//!  2. **Structure tensor** at a sweep of (derivative σ, tensor-smoothing ρ)
//!     scales → per-pixel director **orientation** (mod π, as the doubled-angle
//!     unit vector cos2φ/sin2φ so it interpolates without a branch cut),
//!     **coherence** ((λ₁−λ₂)/(λ₁+λ₂))², **energy** (trace).
//!  3. **Winding charge** s = (1/2π)·Σ Δφ_wrapped around a loop, with Δφ wrapped
//!     into (−π/2, π/2]. A radial/spiral hub → **+1**, a saddle/junction → **−1**,
//!     empty/isotropic → **0**. Swept over a small radius band; the
//!     cross-radius-consistent (same sign, similar magnitude) charge is the
//!     reported defect. Weighted by loop-mean coherence for display.
//!  4. **Reflection** scores `Σ D·D∘M / Σ D²` about the four frame axes on the
//!     detail field D (energy), with H/V residual maps.
//!  5. Off-substrate **reference sanity panel** on two external JPEG luminances.
//!
//! **Convention note.** The prompt states charge = (1/π)·Σ Δθ_wrapped; that is
//! the doubled-angle convention and yields +2 for an aster. We use the nematic
//! index s = (1/2π)·Σ Δφ_wrapped (φ the director angle, mod π), which yields the
//! clean **+1 / −1** the prompt expects for hub / saddle. The synthetic
//! validation (step 2) confirms ±1, so the normalization is pinned, not assumed.
//!
//! Run: `cargo test --release --lib symmetry_probe -- --ignored --nocapture`.

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt::Write as _;

use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::coloring::{ChannelSet, ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::font;
use crate::palette::{linear_to_srgb, Palette};
use crate::probe;
use crate::render::{self, Frame};

// --- fixed regime ------------------------------------------------------------
/// Analysis + render resolution (16:9, focus-window scale, not the 320×180 gen
/// screen). f64 cheap-regime — these benches are shallow (fw ≥ ~2e-3).
pub(crate) const RW: u32 = 1280;
pub(crate) const RH: u32 = 720;
pub(crate) const MAXITER: u32 = 2000;
pub(crate) const BAILOUT: f64 = 1e6;

const OUT_DIR: &str = "data/symmetry_probe";
const REFS_DIR: &str = "data/symmetry_probe/refs";
pub(crate) const CMAP_FILE: &str = "data/palettes/clean_colormaps.json";
pub(crate) const SEED_CMAP: &str = "twilight_shifted"; // smooth preview
const DIVERGING: &str = "coolwarm"; // winding ± (blue −1, white 0, red +1)
const SEQ: &str = "inferno"; // coherence / energy / residual
const CYCLIC: &str = "twilight"; // orientation (angle mod π)

// --- structure-tensor scale sweep -------------------------------------------
/// (derivative σ, tensor-smoothing ρ) in pixels. Too small = noise, too large =
/// washes the hub out — so we sweep and SHOW each rather than hardcode one.
pub(crate) const SCALES: [(f64, f64); 3] = [(1.0, 2.0), (2.0, 4.0), (3.5, 7.0)];

// --- winding loop ------------------------------------------------------------
/// Loop radii (px) for the cross-radius-consistency check. A real scale-stable
/// hub holds its charge across the band; noise does not.
const RADII: [f64; 3] = [5.0, 8.0, 12.0];
/// Loop sample count.
const LOOP_STEPS: usize = 48;
/// A radius set is "consistent" only if all charges share sign and each exceeds
/// this magnitude. Median is then the reported charge.
const CONS_MIN_ABS: f64 = 0.25;
/// |charge| above which a consistent pixel is flagged as a candidate defect in
/// the printed extrema.
const DEFECT_THRESH: f64 = 0.45;

// --- display -----------------------------------------------------------------
const DISP_W: u32 = 460; // tile image width (height derived from field aspect)
const TITLE_H: u32 = 14;
const CBAR_W: u32 = 12;
const GUT: u32 = 44;
const PAD: u32 = 5;
const BG: Rgb<u8> = Rgb([16, 16, 18]);

// ===========================================================================
// Entry
// ===========================================================================

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn symmetry_probe() {
    run().expect("symmetry-probe");
}

pub(crate) struct Bench {
    pub(crate) name: &'static str,
    pub(crate) note: &'static str,
    pub(crate) center: Complex<f64>,
    pub(crate) fw: f64,
}

/// The canonical bench frames (shared with `logpolar_probe` so both probes run
/// the identical deterministic substrate). Seahorse-valley spiral at several
/// zooms plus nearby offsets.
pub(crate) fn bench_frames() -> Vec<Bench> {
    vec![
        Bench { name: "spiral_wide", note: "canonical spiral, wide", center: Complex::new(-0.7453, 0.1127), fw: 0.030 },
        Bench { name: "spiral_mid", note: "canonical spiral, mid", center: Complex::new(-0.7453, 0.1127), fw: 0.010 },
        Bench { name: "spiral_deep", note: "canonical spiral, deep", center: Complex::new(-0.7453, 0.1127), fw: 0.0035 },
        Bench { name: "seahorse_a", note: "valley offset A", center: Complex::new(-0.74540, 0.11320), fw: 0.018 },
        Bench { name: "seahorse_b", note: "valley offset B", center: Complex::new(-0.74300, 0.11400), fw: 0.012 },
    ]
}

fn run() -> Result<(), String> {
    crate::ensure_parent_dir(&format!("{OUT_DIR}/x"))?;

    let cmaps = std::fs::read_to_string(CMAP_FILE).map_err(|e| format!("read {CMAP_FILE}: {e}"))?;
    let pal_div = load_pal(&cmaps, DIVERGING)?;
    let pal_seq = load_pal(&cmaps, SEQ)?;
    let pal_cyc = load_pal(&cmaps, CYCLIC)?;
    let pal_seed = load_pal(&cmaps, SEED_CMAP)?;

    // --- Step 2 FIRST: validate the winding primitive on synthetic fields with
    //     known index. If +1/−1 don't come out clean, stop — nothing downstream
    //     is trustworthy. ---
    eprintln!("=== synthetic winding validation (primitive-in-isolation) ===");
    if !synthetic_validation(&pal_div)? {
        return Err(
            "synthetic winding did NOT recover clean +1/−1 — primitive broken, stopping.".into(),
        );
    }
    eprintln!("  synthetic +1/−1 clean — proceeding to the bench.\n");

    // --- Step 0: bench. Canonical seahorse-valley spiral at several zooms +
    //     nearby offsets. Wide multi-hub compositions; eyeball ground truth. ---
    let benches = bench_frames();

    eprintln!(
        "bench: {} frames @ {RW}x{RH} ss1 maxiter {MAXITER}; tensor scales {:?}; winding radii {:?} ({} steps)",
        benches.len(), SCALES, RADII, LOOP_STEPS
    );

    // time one frame to give the human a backgrounding estimate up front.
    let t_probe = std::time::Instant::now();
    process_bench(&benches[0], &pal_seed, &pal_div, &pal_seq, &pal_cyc)?;
    let per = t_probe.elapsed().as_secs_f64();
    eprintln!(
        "  [1/{}] {} done in {:.1}s (≈{:.0}s remaining for the rest)",
        benches.len(), benches[0].name, per, per * (benches.len() - 1) as f64
    );
    for (i, b) in benches.iter().enumerate().skip(1) {
        let t0 = std::time::Instant::now();
        process_bench(b, &pal_seed, &pal_div, &pal_seq, &pal_cyc)?;
        eprintln!("  [{}/{}] {} done in {:.1}s", i + 1, benches.len(), b.name, t0.elapsed().as_secs_f64());
    }

    // --- Step 5: off-substrate reference sanity panel ---
    reference_panel(&pal_div, &pal_seq)?;

    eprintln!(
        "\nsymmetry-probe done — panels under {OUT_DIR}/. \
         READ: do clean ±1 winding islands land on the marked hubs (+) and junctions (−), \
         and does winding stay ~0 on empty grey? No quality claim."
    );
    Ok(())
}

// ===========================================================================
// Structure tensor
// ===========================================================================

pub(crate) struct TensorField {
    /// Director orientation φ ∈ (−π/2, π/2] (gradient orientation, mod π).
    pub(crate) orient: Vec<f64>,
    /// Doubled-angle unit vector (cos2φ, sin2φ) — interpolates without branch cut.
    pub(crate) c2: Vec<f64>,
    pub(crate) s2: Vec<f64>,
    /// Coherence ((λ₁−λ₂)/(λ₁+λ₂))² ∈ [0,1].
    pub(crate) coh: Vec<f64>,
    /// Energy = trace λ₁+λ₂ (≈ |∇|²; flat regions → ~0).
    pub(crate) energy: Vec<f64>,
    pub(crate) w: usize,
    pub(crate) h: usize,
}

pub(crate) fn structure_tensor(s: &[f64], w: usize, h: usize, sigma_d: f64, rho: f64) -> TensorField {
    let sm = gauss_blur(s, w, h, sigma_d);
    // central-difference gradient of the smoothed scalar
    let grad: Vec<(f64, f64)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let x = i % w;
            let y = i / w;
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            let gx = 0.5 * (sm[y * w + xp] - sm[y * w + xm]);
            let gy = 0.5 * (sm[yp * w + x] - sm[ym * w + x]);
            (gx, gy)
        })
        .collect();
    let jxx0: Vec<f64> = grad.iter().map(|&(gx, _)| gx * gx).collect();
    let jyy0: Vec<f64> = grad.iter().map(|&(_, gy)| gy * gy).collect();
    let jxy0: Vec<f64> = grad.iter().map(|&(gx, gy)| gx * gy).collect();
    let jxx = gauss_blur(&jxx0, w, h, rho);
    let jyy = gauss_blur(&jyy0, w, h, rho);
    let jxy = gauss_blur(&jxy0, w, h, rho);

    let mut orient = vec![0.0; w * h];
    let mut c2 = vec![0.0; w * h];
    let mut s2 = vec![0.0; w * h];
    let mut coh = vec![0.0; w * h];
    let mut energy = vec![0.0; w * h];
    for i in 0..w * h {
        let a = jxx[i];
        let b = jyy[i];
        let d = jxy[i];
        let tr = a + b;
        let diff = a - b;
        let q = (diff * diff + 4.0 * d * d).sqrt(); // λ₁ − λ₂
        orient[i] = 0.5 * (2.0 * d).atan2(diff);
        if q > 1e-300 {
            c2[i] = diff / q;
            s2[i] = 2.0 * d / q;
        }
        coh[i] = if tr > 1e-300 { let r = q / tr; r * r } else { 0.0 };
        energy[i] = tr;
    }
    TensorField { orient, c2, s2, coh, energy, w, h }
}

/// Separable Gaussian blur (clamp-to-edge). σ≤0 → identity.
fn gauss_blur(src: &[f64], w: usize, h: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return src.to_vec();
    }
    let r = (3.0 * sigma).ceil().max(1.0) as i64;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut k: Vec<f64> = (-r..=r).map(|i| (-((i * i) as f64) * inv).exp()).collect();
    let ksum: f64 = k.iter().sum();
    for x in &mut k {
        *x /= ksum;
    }
    let r = r as usize;
    // horizontal
    let mut tmp = vec![0.0; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - r as i64).clamp(0, w as i64 - 1) as usize;
                acc += kv * src[y * w + sx];
            }
            row[x] = acc;
        }
    });
    // vertical
    let mut out = vec![0.0; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - r as i64).clamp(0, h as i64 - 1) as usize;
                acc += kv * tmp[sy * w + x];
            }
            row[x] = acc;
        }
    });
    out
}

// ===========================================================================
// Winding
// ===========================================================================

/// Winding charge s = (1/2π)·Σ Δφ_wrapped around a circle of radius `r` (px)
/// centered at `(px,py)`, plus the loop-mean coherence. Δφ is the wrapped change
/// of the director angle (interpolated via the doubled-angle vector, so no branch
/// cut). Returns `(charge, mean_coh)`.
fn winding_at(tf: &TensorField, px: f64, py: f64, r: f64, n: usize) -> (f64, f64) {
    let mut prev = 0.0f64;
    let mut sum = 0.0;
    let mut cohsum = 0.0;
    for k in 0..=n {
        let th = k as f64 / n as f64 * TAU;
        let (s, c) = th.sin_cos();
        let x = px + r * c;
        let y = py + r * s;
        let cc = bilin(&tf.c2, tf.w, tf.h, x, y);
        let ss = bilin(&tf.s2, tf.w, tf.h, x, y);
        let phi = 0.5 * ss.atan2(cc);
        if k > 0 {
            let mut d = phi - prev;
            while d > FRAC_PI_2 {
                d -= PI;
            }
            while d <= -FRAC_PI_2 {
                d += PI;
            }
            sum += d;
            cohsum += bilin(&tf.coh, tf.w, tf.h, x, y);
        }
        prev = phi;
    }
    (sum / TAU, cohsum / n as f64)
}

struct WindRes {
    /// Per-radius display field (raw charge × loop-mean coherence).
    disp: Vec<Vec<f64>>,
    /// Cross-radius-consistent raw charge (0 where inconsistent).
    cons_raw: Vec<f64>,
    /// Consistent charge × min loop-coherence (display).
    cons_disp: Vec<f64>,
}

fn compute_winding(tf: &TensorField) -> WindRes {
    let w = tf.w;
    let h = tf.h;
    debug_assert_eq!(RADII.len(), 3);
    let margin = (RADII.iter().cloned().fold(0.0, f64::max).ceil() as usize) + 2;

    // per-pixel: (disp[3], cons_raw, cons_disp)
    let out: Vec<([f64; 3], f64, f64)> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let px = i % w;
            let py = i / w;
            if px < margin || py < margin || px + margin >= w || py + margin >= h {
                return ([0.0; 3], 0.0, 0.0);
            }
            let mut chg = [0.0; 3];
            let mut cf = [0.0; 3];
            for (ri, &r) in RADII.iter().enumerate() {
                let (c, mc) = winding_at(tf, px as f64, py as f64, r, LOOP_STEPS);
                chg[ri] = c;
                cf[ri] = mc;
            }
            let s0 = chg[0].signum();
            let consistent = chg.iter().all(|&c| c.signum() == s0 && c.abs() > CONS_MIN_ABS);
            let (cons_raw, cons_disp) = if consistent {
                let mut v = chg;
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = v[1];
                let conf = cf.iter().cloned().fold(f64::INFINITY, f64::min);
                (med, med * conf)
            } else {
                (0.0, 0.0)
            };
            let disp = [chg[0] * cf[0], chg[1] * cf[1], chg[2] * cf[2]];
            (disp, cons_raw, cons_disp)
        })
        .collect();

    let mut disp = vec![vec![0.0; w * h]; RADII.len()];
    let mut cons_raw = vec![0.0; w * h];
    let mut cons_disp = vec![0.0; w * h];
    for (i, (d, cr, cd)) in out.into_iter().enumerate() {
        for ri in 0..RADII.len() {
            disp[ri][i] = d[ri];
        }
        cons_raw[i] = cr;
        cons_disp[i] = cd;
    }
    WindRes { disp, cons_raw, cons_disp }
}

/// Spatially-diverse extrema of a signed field (by |value|), threshold + min
/// separation. Returns `(x, y, value)` sorted by |value| descending.
fn extrema(field: &[f64], w: usize, thresh: f64, min_sep: f64, k: usize) -> Vec<(usize, usize, f64)> {
    let mut c: Vec<(usize, usize, f64)> = field
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.abs() >= thresh)
        .map(|(i, &v)| (i % w, i / w, v))
        .collect();
    c.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
    let sep2 = min_sep * min_sep;
    let mut kept: Vec<(usize, usize, f64)> = Vec::new();
    for cand in c {
        let ok = kept.iter().all(|kp| {
            let dx = kp.0 as f64 - cand.0 as f64;
            let dy = kp.1 as f64 - cand.1 as f64;
            dx * dx + dy * dy >= sep2
        });
        if ok {
            kept.push(cand);
            if kept.len() >= k {
                break;
            }
        }
    }
    kept
}

// ===========================================================================
// Reflection
// ===========================================================================

/// Energy-normalized reflection score `Σ D·D∘M / Σ D²` plus the residual map
/// `|D − D∘M|`, for a mirror `M(x,y) -> Option<(x',y')>` (None = out of bounds).
fn reflection<M>(d: &[f64], w: usize, h: usize, m: M) -> (f64, Vec<f64>)
where
    M: Fn(usize, usize) -> Option<(usize, usize)> + Sync,
{
    let resid: Vec<f64> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            match m(x, y) {
                Some((mx, my)) => (d[i] - d[my * w + mx]).abs(),
                None => 0.0,
            }
        })
        .collect();
    let (num, den) = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            match m(x, y) {
                Some((mx, my)) => (d[i] * d[my * w + mx], d[i] * d[i]),
                None => (0.0, 0.0),
            }
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (if den > 0.0 { num / den } else { 0.0 }, resid)
}

/// The four frame-axis reflection scores `[H, V, diag↘, diag↗]` + H,V residuals.
/// Diagonals are evaluated in centered pixel coords (cover the inscribed square).
pub(crate) fn reflection_axes(d: &[f64], w: usize, h: usize) -> ([f64; 4], Vec<f64>, Vec<f64>) {
    let (sh, rh) = reflection(d, w, h, |x, y| Some((x, h - 1 - y))); // horizontal axis
    let (sv, rv) = reflection(d, w, h, |x, y| Some((w - 1 - x, y))); // vertical axis
    let (cx, cy) = (w as i64 / 2, h as i64 / 2);
    let (sd1, _) = reflection(d, w, h, |x, y| {
        let (u, v) = (x as i64 - cx, y as i64 - cy);
        let (mx, my) = (cx + v, cy + u);
        if mx >= 0 && my >= 0 && (mx as usize) < w && (my as usize) < h {
            Some((mx as usize, my as usize))
        } else {
            None
        }
    });
    let (sd2, _) = reflection(d, w, h, |x, y| {
        let (u, v) = (x as i64 - cx, y as i64 - cy);
        let (mx, my) = (cx - v, cy - u);
        if mx >= 0 && my >= 0 && (mx as usize) < w && (my as usize) < h {
            Some((mx as usize, my as usize))
        } else {
            None
        }
    });
    ([sh, sv, sd1, sd2], rh, rv)
}

// ===========================================================================
// Step 2 — synthetic validation
// ===========================================================================

fn synthetic_validation(pal_div: &Palette) -> Result<bool, String> {
    let gs = 121usize;
    // Build analytic doubled-angle fields. Radial gradient → +1; saddle → −1.
    let radial = analytic_field(gs, false);
    let saddle = analytic_field(gs, true);
    let empty = TensorField {
        orient: vec![0.0; gs * gs],
        c2: vec![0.0; gs * gs],
        s2: vec![0.0; gs * gs],
        coh: vec![0.0; gs * gs],
        energy: vec![0.0; gs * gs],
        w: gs,
        h: gs,
    };
    // parallel/straight lines: constant orientation (φ=0 → c2=1,s2=0), coh=1.
    let parallel = TensorField {
        orient: vec![0.0; gs * gs],
        c2: vec![1.0; gs * gs],
        s2: vec![0.0; gs * gs],
        coh: vec![1.0; gs * gs],
        energy: vec![1.0; gs * gs],
        w: gs,
        h: gs,
    };

    let cc = (gs / 2) as f64;
    let test_radii = [10.0, 20.0, 30.0, 40.0];
    let charge_at_center = |tf: &TensorField| -> Vec<f64> {
        test_radii.iter().map(|&r| winding_at(tf, cc, cc, r, 96).0).collect()
    };
    let cr = charge_at_center(&radial);
    let cs = charge_at_center(&saddle);
    let ce = charge_at_center(&empty);
    let cp = charge_at_center(&parallel);
    eprintln!("  radial (expect +1): {:?}", fmt3(&cr));
    eprintln!("  saddle (expect −1): {:?}", fmt3(&cs));
    eprintln!("  empty  (expect  0): {:?}", fmt3(&ce));
    eprintln!("  parallel(expect 0): {:?}", fmt3(&cp));

    // tolerance: ±1 within 0.1; 0-fields within 0.1.
    let ok_pm1 = |v: &[f64], sign: f64| v.iter().all(|&c| (c - sign).abs() < 0.1);
    let ok_zero = |v: &[f64]| v.iter().all(|&c| c.abs() < 0.1);
    let pass = ok_pm1(&cr, 1.0) && ok_pm1(&cs, -1.0) && ok_zero(&ce) && ok_zero(&cp);

    // synthetic heatmaps (winding per pixel at r=15) for the panel.
    let hm = |tf: &TensorField| -> Vec<f64> {
        (0..gs * gs)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % gs) as f64, (i / gs) as f64);
                if x < 16.0 || y < 16.0 || x >= gs as f64 - 16.0 || y >= gs as f64 - 16.0 {
                    0.0
                } else {
                    winding_at(tf, x, y, 12.0, 64).0
                }
            })
            .collect()
    };
    let t_rad = field_tile(&hm(&radial), gs, gs, -1.2, 1.2, pal_div, "synthetic RADIAL (winding, expect +1 center)", DISP_W);
    let t_sad = field_tile(&hm(&saddle), gs, gs, -1.2, 1.2, pal_div, "synthetic SADDLE (winding, expect -1 center)", DISP_W);
    let panel = vstack(&[banner(t_rad.width().max(t_sad.width()) * 2, &format!("SYNTHETIC WINDING VALIDATION  [pass={pass}]  charge = (1/2pi) sum d-phi-wrapped")), hstack(&[t_rad, t_sad])]);
    save(&panel, &format!("{OUT_DIR}/synthetic_validation.png"))?;
    Ok(pass)
}

/// Analytic doubled-angle field. `saddle=false` → radial gradient (aster, +1);
/// `saddle=true` → saddle (−1). coh=1 except the singular center.
fn analytic_field(gs: usize, saddle: bool) -> TensorField {
    let cc = (gs / 2) as f64;
    let mut c2 = vec![0.0; gs * gs];
    let mut s2 = vec![0.0; gs * gs];
    let mut coh = vec![0.0; gs * gs];
    let mut orient = vec![0.0; gs * gs];
    for y in 0..gs {
        for x in 0..gs {
            let dx = x as f64 - cc;
            let dy = y as f64 - cc;
            let r2 = dx * dx + dy * dy;
            let i = y * gs + x;
            if r2 < 1.0 {
                continue;
            }
            c2[i] = (dx * dx - dy * dy) / r2;
            s2[i] = if saddle { -2.0 * dx * dy / r2 } else { 2.0 * dx * dy / r2 };
            coh[i] = 1.0;
            orient[i] = 0.5 * s2[i].atan2(c2[i]);
        }
    }
    TensorField { orient, c2, s2, coh, energy: vec![1.0; gs * gs], w: gs, h: gs }
}

// ===========================================================================
// Step 0/1/3/4 — per bench frame
// ===========================================================================

fn process_bench(
    b: &Bench,
    pal_seed: &Palette,
    pal_div: &Palette,
    pal_seq: &Palette,
    pal_cyc: &Palette,
) -> Result<(), String> {
    let w = RW as usize;
    let h = RH as usize;
    let buf = render_seed(b.center, b.fw);
    let scalar = smooth_scalar(&buf);

    // smooth preview (markers overlaid after winding, below)
    let preview = render::shade_and_downsample(
        &buf.samples, RW, RH, 1, pal_seed, &preview_params(), b.fw / RW as f64,
    );
    let mut preview_small = downscale_rgb(&preview, DISP_W, DISP_W * RH / RW);

    // structure tensor per scale
    let tfs: Vec<TensorField> = SCALES
        .iter()
        .map(|&(sd, rho)| structure_tensor(&scalar, w, h, sd, rho))
        .collect();

    // orientation + energy at the middle scale; coherence at all scales.
    let mid = 1usize;
    let t_orient = orientation_tile(&tfs[mid], pal_cyc, &format!("orientation s{mid} (sig={},rho={})", SCALES[mid].0, SCALES[mid].1));
    let emax = pctl(&tfs[mid].energy, 0.99).max(1e-12);
    let t_energy = field_tile(&tfs[mid].energy, w, h, 0.0, emax, pal_seq, &format!("energy s{mid} (trace)"), DISP_W);
    let coh_tiles: Vec<RgbImage> = (0..SCALES.len())
        .map(|si| field_tile(&tfs[si].coh, w, h, 0.0, 1.0, pal_seq, &format!("coherence s{si} (sig={})", SCALES[si].0), DISP_W))
        .collect();

    // winding per scale
    let winds: Vec<WindRes> = tfs.iter().map(compute_winding).collect();

    // Per-scale: RAW consistent charge at full ±1.2 saturation (the clean
    // classification — red +1 hub, blue −1 junction, white 0).
    let cons_tiles: Vec<RgbImage> = (0..SCALES.len())
        .map(|si| field_tile(&winds[si].cons_raw, w, h, -1.2, 1.2, pal_div, &format!("winding CONSISTENT s{si} (RAW charge)"), DISP_W))
        .collect();

    // diverging symmetric range across the display (×coherence) winding.
    let vmax = winds
        .iter()
        .flat_map(|wr| wr.cons_disp.iter())
        .fold(0.5f64, |m, &v| m.max(v.abs()))
        .min(1.2);
    // radius sweep at the middle scale (charge×coh display per radius + consistent)
    let mut radius_tiles: Vec<RgbImage> = (0..RADII.len())
        .map(|ri| field_tile(&winds[mid].disp[ri], w, h, -vmax, vmax, pal_div, &format!("winding r={} s{mid} (chg x coh)", RADII[ri]), DISP_W))
        .collect();
    radius_tiles.push(field_tile(&winds[mid].cons_disp, w, h, -vmax, vmax, pal_div, &format!("winding consistent s{mid} (chg x coh)"), DISP_W));

    // Overlay mid-scale detected defects on the smooth preview: red ring = +1
    // hub, blue ring = −1 junction. Lets the eye check placement vs the marked
    // hubs/junctions directly.
    let all_def = extrema(&winds[mid].cons_raw, w, DEFECT_THRESH, 18.0, 20);
    let dscale = DISP_W as f64 / RW as f64;
    for (x, y, v) in &all_def {
        let col = if *v > 0.0 { Rgb([255, 70, 70]) } else { Rgb([90, 130, 255]) };
        draw_marker(&mut preview_small, *x as f64 * dscale, *y as f64 * dscale, 5.0, col);
    }
    let t_preview = titled(preview_small, &format!("{} smooth + defects (red +1, blue -1)", b.name));

    // reflection on the detail field D = sqrt(energy) at scale 0
    let detail: Vec<f64> = tfs[0].energy.iter().map(|&e| e.sqrt()).collect();
    let (axes, rh, rv) = reflection_axes(&detail, w, h);
    let rmax = pctl(&rh, 0.99).max(pctl(&rv, 0.99)).max(1e-12);
    let t_rh = field_tile(&rh, w, h, 0.0, rmax, pal_seq, "H reflection residual |D-D.M|", DISP_W);
    let t_rv = field_tile(&rv, w, h, 0.0, rmax, pal_seq, "V reflection residual |D-D.M|", DISP_W);

    // printed scalars: reflection axes + winding extrema (consistent, mid scale)
    let pos = extrema(&winds[mid].cons_raw, w, DEFECT_THRESH, 24.0, 6)
        .into_iter()
        .filter(|e| e.2 > 0.0)
        .collect::<Vec<_>>();
    let neg = extrema(&winds[mid].cons_raw, w, DEFECT_THRESH, 24.0, 6)
        .into_iter()
        .filter(|e| e.2 < 0.0)
        .collect::<Vec<_>>();
    eprintln!(
        "    {} reflection[H,V,diag1,diag2]={:.3},{:.3},{:.3},{:.3}",
        b.name, axes[0], axes[1], axes[2], axes[3]
    );
    eprintln!(
        "    {} winding(s{mid}) +defects: {}  | -defects: {}",
        b.name,
        fmt_defects(&pos),
        fmt_defects(&neg)
    );

    // compose panel
    let dir = format!("{OUT_DIR}/{}", b.name);
    crate::ensure_parent_dir(&format!("{dir}/x"))?;

    // standalone higher-res inspection pair (placement of defects vs hubs).
    {
        let bw = 900u32;
        let mut pb = downscale_rgb(&preview, bw, bw * RH / RW);
        let bscale = bw as f64 / RW as f64;
        for (x, y, v) in &all_def {
            let col = if *v > 0.0 { Rgb([255, 70, 70]) } else { Rgb([90, 130, 255]) };
            draw_marker(&mut pb, *x as f64 * bscale, *y as f64 * bscale, 8.0, col);
        }
        let mut rows = vec![titled(pb, &format!("{} smooth + defects (red +1, blue -1)", b.name))];
        for si in 0..SCALES.len() {
            rows.push(field_tile(&winds[si].cons_raw, w, h, -1.2, 1.2, pal_div, &format!("winding consistent (raw) s{si} sig={} rho={}", SCALES[si].0, SCALES[si].1), bw));
        }
        let insp = vstack(&rows);
        save(&insp, &format!("{dir}/inspect.png"))?;
    }
    let row1 = hstack(&[t_preview, t_orient, t_energy]);
    let row2 = hstack(&coh_tiles);
    let row3 = hstack(&cons_tiles);
    let row4 = hstack(&radius_tiles);
    let row5 = hstack(&[t_rh, t_rv]);
    let width = [row1.width(), row2.width(), row3.width(), row4.width(), row5.width()].into_iter().max().unwrap();
    let head = banner(
        width,
        &format!(
            "SYMMETRY PROBE  {}  ({})  center ({:.6},{:.6}) fw {:.3e}   refl[H,V,d1,d2]={:.2},{:.2},{:.2},{:.2}   [DIAGNOSIS ONLY]",
            b.name, b.note, b.center.re, b.center.im, b.fw, axes[0], axes[1], axes[2], axes[3]
        ),
    );
    let panel = vstack(&[head, row1, row2, row3, row4, row5]);
    save(&panel, &format!("{dir}/panel.png"))?;

    // per-frame JSON
    let mut j = String::new();
    let _ = write!(
        j,
        "{{\n  \"name\": \"{}\", \"center_re\": {:.12e}, \"center_im\": {:.12e}, \"frame_width\": {:.6e},\n  \"reflection\": {{ \"h\": {:.5}, \"v\": {:.5}, \"diag1\": {:.5}, \"diag2\": {:.5} }},\n  \"winding_pos_defects\": {},\n  \"winding_neg_defects\": {}\n}}\n",
        b.name, b.center.re, b.center.im, b.fw, axes[0], axes[1], axes[2], axes[3],
        defects_json(&pos), defects_json(&neg)
    );
    std::fs::write(format!("{dir}/symmetry.json"), &j).map_err(|e| format!("write json: {e}"))?;
    Ok(())
}

pub(crate) fn render_seed(center: Complex<f64>, fw: f64) -> render::SampleBuffer {
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let backend = F64Backend::new(MAXITER, BAILOUT, trap);
    let frame = Frame { center, frame_width: fw, out_width: RW, out_height: RH };
    // Smooth-only fast path: no trap, no DE.
    render::iterate_samples_f64(&backend, &frame, 1, ChannelSet { trap: false, de: false })
}

/// Native smooth-iteration scalar; interior filled with the max escape value so
/// the set boundary reads as a plateau edge (no spurious interior structure).
pub(crate) fn smooth_scalar(buf: &render::SampleBuffer) -> Vec<f64> {
    let s = &buf.samples;
    let maxesc = s
        .iter()
        .filter(|p| p.escaped)
        .map(|p| p.smooth_iter)
        .fold(f64::NEG_INFINITY, f64::max);
    let fill = if maxesc.is_finite() { maxesc } else { 0.0 };
    s.iter().map(|p| if p.escaped { p.smooth_iter } else { fill }).collect()
}

pub(crate) fn preview_params() -> ColorParams {
    ColorParams {
        density: 0.025,
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

// ===========================================================================
// Step 5 — reference sanity panel (off native substrate)
// ===========================================================================

fn reference_panel(pal_div: &Palette, pal_seq: &Palette) -> Result<(), String> {
    let refs = [
        ("delicacy", "delicacy.jpg", "expect central X-junction = -1"),
        ("helping-hands", "helping-hands-25.jpg", "expect spiral hubs = +1"),
    ];
    let mut tiles: Vec<RgbImage> = Vec::new();
    let mut any = false;
    for (name, file, expect) in refs {
        let path = format!("{REFS_DIR}/{file}");
        let Some((lum, w, h)) = load_luma(&path) else {
            eprintln!("  reference {path} absent — skipping ref panel entry (graceful).");
            continue;
        };
        any = true;
        let (lum, w, h) = fit_downscale(&lum, w, h, 900);
        eprintln!("  reference {name}: {w}x{h} luminance — SANITY ONLY (palette-contaminated, off substrate)");
        let tf = structure_tensor(&lum, w, h, SCALES[1].0, SCALES[1].1);
        let wr = compute_winding(&tf);
        let detail: Vec<f64> = tf.energy.iter().map(|&e| e.sqrt()).collect();
        let (axes, rh, _rv) = reflection_axes(&detail, w, h);
        let pos = extrema(&wr.cons_raw, w, DEFECT_THRESH, 24.0, 6).into_iter().filter(|e| e.2 > 0.0).count();
        let neg = extrema(&wr.cons_raw, w, DEFECT_THRESH, 24.0, 6).into_iter().filter(|e| e.2 < 0.0).count();
        eprintln!("    {name}: refl[H,V]={:.3},{:.3}  +defects={pos} -defects={neg}  ({expect})", axes[0], axes[1]);

        // luminance preview (grey ramp)
        let lmax = pctl(&lum, 0.99).max(1e-6);
        let t_lum = field_tile(&lum, w, h, 0.0, lmax, pal_seq, &format!("{name} luminance [SANITY ONLY]"), DISP_W);
        let vmax = wr.cons_disp.iter().fold(0.5f64, |m, &v| m.max(v.abs())).min(1.2);
        let t_w = field_tile(&wr.cons_disp, w, h, -vmax, vmax, pal_div, &format!("{name} winding [SANITY: {expect}]"), DISP_W);
        let rmax = pctl(&rh, 0.99).max(1e-12);
        let t_rh = field_tile(&rh, w, h, 0.0, rmax, pal_seq, &format!("{name} H reflection resid"), DISP_W);
        tiles.push(vstack(&[t_lum, t_w, t_rh]));
    }
    if !any {
        eprintln!("  no reference images found under {REFS_DIR}/ — ref panel skipped.");
        return Ok(());
    }
    let body = hstack(&tiles);
    let head = banner(body.width(), "REFERENCE SANITY PANEL  (palette-contaminated, OFF native substrate -- sanity only)");
    let panel = vstack(&[head, body]);
    save(&panel, &format!("{OUT_DIR}/reference_sanity.png"))?;
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

/// Box-average downscale of an f64 field so the longer dim ≤ `maxdim`.
fn fit_downscale(f: &[f64], w: usize, h: usize, maxdim: usize) -> (Vec<f64>, usize, usize) {
    let scale = (maxdim as f64 / w.max(h) as f64).min(1.0);
    let dw = ((w as f64 * scale).round() as usize).max(1);
    let dh = ((h as f64 * scale).round() as usize).max(1);
    if dw == w && dh == h {
        return (f.to_vec(), w, h);
    }
    let mut out = vec![0.0; dw * dh];
    for y in 0..dh {
        let sy0 = y * h / dh;
        let sy1 = ((y + 1) * h / dh).max(sy0 + 1).min(h);
        for x in 0..dw {
            let sx0 = x * w / dw;
            let sx1 = ((x + 1) * w / dw).max(sx0 + 1).min(w);
            let mut acc = 0.0;
            let mut n = 0;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    acc += f[sy * w + sx];
                    n += 1;
                }
            }
            out[y * dw + x] = acc / n as f64;
        }
    }
    (out, dw, dh)
}

// ===========================================================================
// Visualization helpers
// ===========================================================================

pub(crate) fn load_pal(cmaps: &str, name: &str) -> Result<Palette, String> {
    Ok(Palette::from_srgb8_stops(name, &probe::load_colormap(cmaps, name)?, false))
}

fn lut_rgb(pal: &Palette, t: f64) -> Rgb<u8> {
    let lin = pal.lookup_linear(t.clamp(0.0, 1.0));
    Rgb([
        (linear_to_srgb(lin[0]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[1]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[2]) * 255.0 + 0.5) as u8,
    ])
}

/// Colorize a `w×h` field through `pal` over `[vmin,vmax]` → same-size image.
fn colorize(field: &[f64], w: usize, h: usize, vmin: f64, vmax: f64, pal: &Palette) -> RgbImage {
    let span = (vmax - vmin).max(1e-12);
    let mut img = RgbImage::new(w as u32, h as u32);
    for (i, px) in img.pixels_mut().enumerate() {
        let t = ((field[i] - vmin) / span).clamp(0.0, 1.0);
        *px = lut_rgb(pal, t);
    }
    img
}

/// A finished field tile: colorize → downscale to `target_w` (aspect-preserved)
/// → title bar + vertical colorbar.
fn field_tile(field: &[f64], w: usize, h: usize, vmin: f64, vmax: f64, pal: &Palette, title: &str, target_w: u32) -> RgbImage {
    let img = colorize(field, w, h, vmin, vmax, pal);
    let dw = target_w.min(w as u32);
    let dh = ((h as u32 * dw) / w as u32).max(1);
    let small = downscale_rgb(&img, dw, dh);
    titled_cbar(small, title, pal, vmin, vmax)
}

/// Orientation tile: hue from φ (cyclic palette), brightness from coherence.
fn orientation_tile(tf: &TensorField, pal: &Palette, title: &str) -> RgbImage {
    let w = tf.w;
    let h = tf.h;
    let mut img = RgbImage::new(w as u32, h as u32);
    for (i, px) in img.pixels_mut().enumerate() {
        let t = (tf.orient[i] + FRAC_PI_2) / PI; // (-pi/2,pi/2] -> [0,1)
        let lin = pal.lookup_linear(t.clamp(0.0, 1.0));
        let b = tf.coh[i].clamp(0.0, 1.0);
        *px = Rgb([
            (linear_to_srgb(lin[0] * b) * 255.0 + 0.5) as u8,
            (linear_to_srgb(lin[1] * b) * 255.0 + 0.5) as u8,
            (linear_to_srgb(lin[2] * b) * 255.0 + 0.5) as u8,
        ]);
    }
    let dw = DISP_W.min(w as u32);
    let dh = ((h as u32 * dw) / w as u32).max(1);
    titled(downscale_rgb(&img, dw, dh), title)
}

/// Box-average downscale of an RgbImage to `dw×dh`.
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

/// Title bar above an image (image width preserved).
fn titled(img: RgbImage, title: &str) -> RgbImage {
    let w = img.width();
    let mut out = RgbImage::from_pixel(w, img.height() + TITLE_H, BG);
    probe::blit(&mut out, &img, 0, TITLE_H);
    font::draw_text(&mut out, &title.to_uppercase(), 2, 3, 1, Rgb([235, 235, 235]), true);
    out
}

/// Title bar + right-side vertical colorbar with min/max labels.
fn titled_cbar(img: RgbImage, title: &str, pal: &Palette, vmin: f64, vmax: f64) -> RgbImage {
    let iw = img.width();
    let ih = img.height();
    let mut out = RgbImage::from_pixel(iw + CBAR_W + GUT, ih + TITLE_H, BG);
    probe::blit(&mut out, &img, 0, TITLE_H);
    font::draw_text(&mut out, &title.to_uppercase(), 2, 3, 1, Rgb([235, 235, 235]), true);
    let bx = iw + 4;
    for y in 0..ih {
        let t = 1.0 - y as f64 / (ih - 1).max(1) as f64;
        let px = lut_rgb(pal, t);
        for x in bx..bx + CBAR_W {
            out.put_pixel(x, TITLE_H + y, px);
        }
    }
    let lx = bx + CBAR_W + 2;
    font::draw_text(&mut out, &fmt_v(vmax), lx, TITLE_H + 1, 1, Rgb([230, 230, 230]), true);
    font::draw_text(&mut out, &fmt_v(vmin), lx, TITLE_H + ih - 8, 1, Rgb([230, 230, 230]), true);
    out
}

/// A dark banner strip of the given width with a one-line caption.
fn banner(width: u32, text: &str) -> RgbImage {
    let mut img = RgbImage::from_pixel(width.max(8), 18, Rgb([10, 10, 12]));
    font::draw_text(&mut img, &text.to_uppercase(), 4, 4, 1, Rgb([255, 230, 150]), true);
    img
}

/// Horizontally stack images (top-aligned), padded.
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

/// Vertically stack images (left-aligned), padded.
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

/// A ring (radius `r` px) + center dot at `(cx,cy)` in `col`, with a 1px dark
/// halo for legibility. Clipped to the image.
fn draw_marker(img: &mut RgbImage, cx: f64, cy: f64, r: f64, col: Rgb<u8>) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let put = |img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>| {
        if x >= 0 && y >= 0 && x < w && y < h {
            img.put_pixel(x as u32, y as u32, c);
        }
    };
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

// ===========================================================================
// Small numeric helpers
// ===========================================================================

/// Value at quantile `q` of a field (copy + sort; fields here are ≤ ~1M).
fn pctl(f: &[f64], q: f64) -> f64 {
    if f.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = f.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    v[idx]
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

fn fmt_v(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 100.0 || v.abs() < 0.01 {
        format!("{v:.1e}")
    } else {
        format!("{v:.2}")
    }
}

fn fmt3(v: &[f64]) -> Vec<String> {
    v.iter().map(|x| format!("{x:+.3}")).collect()
}

fn fmt_defects(d: &[(usize, usize, f64)]) -> String {
    if d.is_empty() {
        return "none".into();
    }
    d.iter().map(|(x, y, v)| format!("({x},{y}):{v:+.2}")).collect::<Vec<_>>().join(" ")
}

fn defects_json(d: &[(usize, usize, f64)]) -> String {
    let mut s = String::from("[");
    for (i, (x, y, v)) in d.iter().enumerate() {
        let _ = write!(s, "{}[{x},{y},{v:.4}]", if i > 0 { "," } else { "" });
    }
    s.push(']');
    s
}
