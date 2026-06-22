//! **Throwaway diagnostic — characterizer-as-finder (prompt 4).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Answers one question
//! (see `prompts/prompt-04-characterizer-as-finder.md`): **can the validated
//! log-polar characterizer be its own finder?** Prompt 2/3 settled that the
//! characterizer *works* (synthetics recover pitch/fold/scaling; center-jitter
//! climbed *delicacy* R₂ 0.08→0.82) and that the *finder* is the open problem
//! (nucleus seeds land on minibrots/bulbs, not on the visual spiral eyes).
//!
//! The plan: make a **guard-aware spiral score `S(c)`** out of the characterizer's
//! own oriented-energy, map it over **candidate center position** (THE decisive
//! artifact — do the visual eyes sit on S-maxima?), then hill-climb it from two
//! seed sources (nucleus vs grid) to see whether climbed centers reach the eyes.
//!
//! `S(c)` realization: on the **precomputed** bench smooth-scalar field (bilinear
//! log-polar resample — **never re-iterate per polar sample**), reuse
//! `logpolar_probe`'s sampler + detail field + Radon oriented-energy, then take the
//! **strongest oriented-energy peak at an OFF-AXIS (diagonal) slope, over total
//! energy, presence-gated** (`= 0` when flat). The Radon transform is the
//! oriented-energy realization of "diagonal energy in the log-polar FFT":
//!   - **log-spiral arm** → ridges run at a nonzero slope → an off-axis Radon peak
//!     (slope = pitch). **This is what S rewards.**
//!   - **n-fold rosette / radial arms** → ridges constant in `u` → all energy on the
//!     **s=0 angular axis**, which the guard **excludes**.
//!   - **concentric / minibrot interior** → zero-mean band-pass rings have ~zero
//!     Radon energy at *every* slope (the structure lives in the radial autocorr,
//!     not the Radon) → **auto-excluded**; the synthetic concentric gate verifies it.
//!   - **flat** → presence < floor → `S = 0`.
//! So S is guard-aware by construction: the `[RADIAL]` minibrot degeneracy that
//! fooled the naive nucleus finder scores low here.
//!
//! Reuses `logpolar_probe`'s primitives (sampler / detail / Radon / guards /
//! readout / viz) `pub(crate)` and `symmetry_probe::bench_frames` + nucleus centers
//! — mirrors how `logpolar_probe` consumed `symmetry_probe`. Output:
//! `data/hubfind_probe/`. Synthetics gate the objective BEFORE the bench. No
//! production wiring, no gate, no quality claim — the human judges whether the eyes
//! are S-maxima and whether climbed centers reach them, seed-source by seed-source.
//!
//! Run: `cargo test --release --lib hubfind_probe -- --ignored --nocapture`.

use std::f64::consts::TAU;
use std::fmt::Write as _;

use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::logpolar_probe as lp;
use crate::palette::Palette;
use crate::render;
use crate::symmetry_probe as sp;

const OUT_DIR: &str = "data/hubfind_probe";
const REFS_DIR: &str = "data/symmetry_probe/refs";
const SEQ: &str = "inferno"; // L strip / preview
const DIV: &str = "coolwarm"; // signed detail D strip
// The heatmap palette the prompt asks for; also the smooth-preview palette.
const HEAT: &str = sp::SEED_CMAP; // "twilight_shifted"

const RW: u32 = sp::RW; // 1280
const RH: u32 = sp::RH; // 720

/// The two log-polar bands (r_min, r_max) in **pixels**, mirrored from
/// `logpolar_probe::BANDS` — scale is a measure-trap, so the S-field is shown at
/// both. The climb runs on the tighter band (sharper basins).
const BANDS: [(f64, f64); 2] = [(4.0, 90.0), (10.0, 260.0)];
const CLIMB_BAND: usize = 0;

/// On-axis exclusion: drop Radon slopes within this many indices of `s=0` (the
/// angular/rosette axis). `N_SLOPE=161` → center index 80, step ≈ 0.064 in slope,
/// so ±4 excises |slope| ≲ 0.26 — wide enough to swallow the s=0 peak's lobe, narrow
/// enough to keep all but the very lowest-pitch spirals.
const ONAXIS_GUARD: usize = 4;

/// S-field grid over candidate center position (coarse is fine — r_min can't resolve
/// below the render pixel scale anyway). ~aspect-matched to 1280×720.
const GRID_NX: usize = 48;
const GRID_NY: usize = 27;

/// Coarse grid of climb seeds spread over the frame (the "where are the basins
/// really" seed source).
const SEED_NX: usize = 4;
const SEED_NY: usize = 3;

/// Pattern-search climb: 5×5 evaluations at radius ρ, recenter on best, shrink ρ.
const CLIMB_PASSES: usize = 11;
const CLIMB_SHRINK: f64 = 0.62;

const COL_NUC: Rgb<u8> = Rgb([60, 200, 255]); // nucleus seeds / climbs (cyan)
const COL_GRID: Rgb<u8> = Rgb([255, 150, 40]); // grid seeds / climbs (orange)
const COL_SMAX: Rgb<u8> = Rgb([255, 240, 90]); // S-field local maxima (yellow)
const COL_EYE: Rgb<u8> = Rgb([255, 60, 200]); // hand-marked / midpoint (magenta)

// ===========================================================================
// Entry
// ===========================================================================

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn hubfind_probe() {
    run().expect("hubfind-probe");
}

fn run() -> Result<(), String> {
    crate::ensure_parent_dir(&format!("{OUT_DIR}/x"))?;
    let cmaps =
        std::fs::read_to_string(sp::CMAP_FILE).map_err(|e| format!("read {}: {e}", sp::CMAP_FILE))?;
    let pal_heat = sp::load_pal(&cmaps, HEAT)?;
    let pal_seq = sp::load_pal(&cmaps, SEQ)?;
    let pal_div = sp::load_pal(&cmaps, DIV)?;

    // --- Step A FIRST: synthetic gate. S must peak on a spiral at its true center,
    //     show TWO maxima for two spirals, and NOT peak on a concentric field
    //     (the guard test). If the concentric case peaks, the on-axis exclusion is
    //     wrong → stop, the bench would mean nothing. ---
    eprintln!("=== Step A — synthetic gate (S-over-center; primitive-in-isolation) ===");
    if !synthetic_gate(&pal_heat, &pal_seq, &pal_div)? {
        return Err("S-over-center synthetic gate FAILED (concentric peaked, or spiral did not) — objective broken, stopping.".into());
    }
    eprintln!("  synthetic gate passed — proceeding to the bench.\n");

    // --- Steps B/C/D: the bench (identical deterministic frames as the other probes) ---
    let benches = sp::bench_frames();
    eprintln!(
        "bench: {} frames @ {RW}x{RH}; S-field {GRID_NX}x{GRID_NY}; bands(px) {:?}; climb band {}",
        benches.len(),
        BANDS,
        CLIMB_BAND
    );
    let t0 = std::time::Instant::now();
    process_bench(&benches[0], &pal_heat, &pal_seq, &pal_div)?;
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
        process_bench(b, &pal_heat, &pal_seq, &pal_div)?;
        eprintln!("  [{}/{}] {} done in {:.1}s", i + 1, benches.len(), b.name, t.elapsed().as_secs_f64());
    }

    // --- Step E: reference check (off-substrate, labeled). ---
    reference_panel(&pal_heat, &pal_seq, &pal_div)?;

    eprintln!(
        "\nhubfind-probe done — panels under {OUT_DIR}/. READ, seed-source by seed-source: \
         do the visual spiral eyes sit on S-field maxima (objective)? do climbed centers reach \
         them (finder)? does the nucleus-vs-grid split say it's seeding or objective that's the \
         gap? No quality claim."
    );
    Ok(())
}

// ===========================================================================
// The objective — guard-aware spiral score S(c)
// ===========================================================================

/// `S(c)` and its pitch: the strongest **off-axis** Radon oriented-energy peak over
/// total energy, presence-gated to 0. Reuses `logpolar_probe`'s validated sampler +
/// detail field + Radon. Returns `(S, pitch, on_axis_energy, presence)` — the extra
/// scalars are for diagnostics (on-axis = the excised s=0 rosette/radial band).
fn s_score(field: &[f64], w: usize, h: usize, fx: f64, fy: f64, rmin: f64, rmax: f64) -> (f64, f64, f64, f64) {
    let lpf = lp::logpolar_sample(field, w, h, fx, fy, rmin, rmax);
    if lpf.presence < lp::PRESENCE_FLOOR {
        return (0.0, 0.0, 0.0, lpf.presence);
    }
    let smax = 2.0 * lp::NV as f64 / lp::NU as f64;
    let slopes: Vec<f64> = (0..lp::N_SLOPE).map(|i| -smax + 2.0 * smax * i as f64 / (lp::N_SLOPE - 1) as f64).collect();
    let (_, _, curve) = lp::radon_pitch(&lpf.d, &slopes);
    let center = (lp::N_SLOPE - 1) / 2;
    let on_axis = curve[center];
    let mut best = (center, f64::NEG_INFINITY);
    for (i, &v) in curve.iter().enumerate() {
        if (i as isize - center as isize).unsigned_abs() <= ONAXIS_GUARD {
            continue;
        }
        if v > best.1 {
            best = (i, v);
        }
    }
    let s = best.1.max(0.0);
    let ku = (rmax.ln() - rmin.ln()) / (lp::NU - 1) as f64;
    let kv = TAU / lp::NV as f64;
    let pitch = kv * slopes[best.0] / ku;
    (s, pitch, on_axis, lpf.presence)
}

/// Evaluate `S` over a `GRID_NX×GRID_NY` lattice of candidate centers spanning the
/// full frame (cell centers; the band clamps at the edges → edge cells go dark, not
/// falsely bright, so no margin needed). Row-major `iy*GRID_NX + ix`.
fn s_field(field: &[f64], w: usize, h: usize, rmin: f64, rmax: f64) -> Vec<f64> {
    (0..GRID_NX * GRID_NY)
        .map(|i| {
            let (fx, fy) = grid_center(i % GRID_NX, i / GRID_NX, w, h);
            s_score(field, w, h, fx, fy, rmin, rmax).0
        })
        .collect()
}

/// Pixel center of grid cell `(ix,iy)` over a `w×h` frame (cell centers).
fn grid_center(ix: usize, iy: usize, w: usize, h: usize) -> (f64, f64) {
    let fx = (ix as f64 + 0.5) * w as f64 / GRID_NX as f64;
    let fy = (iy as f64 + 0.5) * h as f64 / GRID_NY as f64;
    (fx, fy)
}

/// Coarse-to-fine local pattern search on `S`: 5×5 evaluations at radius ρ → recenter
/// on the best → shrink ρ. The generalized center-jitter search. Returns the climbed
/// `(fx, fy, S)`.
fn climb(field: &[f64], w: usize, h: usize, fx0: f64, fy0: f64, rho0: f64, rmin: f64, rmax: f64) -> (f64, f64, f64) {
    let mut cx = fx0;
    let mut cy = fy0;
    let mut rho = rho0;
    let mut bs = s_score(field, w, h, cx, cy, rmin, rmax).0;
    for _ in 0..CLIMB_PASSES {
        let step = rho * 0.5;
        let pts: Vec<(f64, f64)> = (-2i32..=2)
            .flat_map(|jy| (-2i32..=2).map(move |jx| (jx, jy)))
            .map(|(jx, jy)| (cx + jx as f64 * step, cy + jy as f64 * step))
            .collect();
        let best = pts
            .par_iter()
            .map(|&(x, y)| (x, y, s_score(field, w, h, x, y, rmin, rmax).0))
            .reduce(|| (cx, cy, bs), |a, b| if b.2 > a.2 { b } else { a });
        if best.2 > bs {
            cx = best.0;
            cy = best.1;
            bs = best.2;
        } else {
            rho *= 0.5; // no neighbour improved → contract harder
        }
        rho *= CLIMB_SHRINK;
        if rho < 0.5 {
            break;
        }
    }
    (cx, cy, bs)
}

// ===========================================================================
// Step A — synthetic gate
// ===========================================================================

/// Analytic field of size `gs×gs` from `f(r,θ)` about an **arbitrary** center
/// `(cx,cy)` (the immediate core r<2 is zeroed). Off-center on purpose — proves the
/// finder localizes, not just that S is high at the grid center.
fn synth_at<F: Fn(f64, f64) -> f64 + Sync>(gs: usize, cx: f64, cy: f64, f: F) -> Vec<f64> {
    (0..gs * gs)
        .into_par_iter()
        .map(|i| {
            let x = (i % gs) as f64 - cx;
            let y = (i / gs) as f64 - cy;
            let r = (x * x + y * y).sqrt();
            if r < 2.0 {
                0.0
            } else {
                f(r, y.atan2(x))
            }
        })
        .collect()
}

fn synthetic_gate(pal_heat: &Palette, pal_seq: &Palette, pal_div: &Palette) -> Result<bool, String> {
    let gs = 221usize;
    let (rmin, rmax) = (4.0f64, 0.22 * gs as f64);

    // --- 1) single log-spiral, off-center; S-over-center must peak at the true center ---
    let (sx, sy) = (0.58 * gs as f64, 0.42 * gs as f64);
    let spiral = synth_at(gs, sx, sy, |r, th| (1.0 * th - 3.0 * r.ln()).cos());
    let (sf1, max1) = synth_field(&spiral, gs, rmin, rmax);
    let (px1, py1, pk1) = field_argmax(&sf1, gs);
    let off1 = ((px1 - sx).powi(2) + (py1 - sy).powi(2)).sqrt();
    let single_ok = off1 < 0.06 * gs as f64;
    let basin1 = basin_width(&spiral, gs, sx, sy, pk1, rmin, rmax);
    eprintln!(
        "  single spiral @({sx:.0},{sy:.0}): S-peak @({px1:.0},{py1:.0}) S={pk1:.3} off={off1:.1}px  basin≈{basin1:.0}px  [{}]",
        pass(single_ok)
    );

    // --- 2) two log-spirals → the S-field must be bimodal (finder is multi-modal) ---
    let (ax, ay) = (0.30 * gs as f64, 0.34 * gs as f64);
    let (bx, by) = (0.72 * gs as f64, 0.68 * gs as f64);
    let sa = synth_at(gs, ax, ay, |r, th| (1.0 * th - 2.5 * r.ln()).cos());
    let sb = synth_at(gs, bx, by, |r, th| (-1.0 * th - 3.5 * r.ln()).cos());
    let two: Vec<f64> = sa.iter().zip(&sb).map(|(p, q)| p + q).collect();
    let (sf2, _max2) = synth_field(&two, gs, rmin, rmax);
    let peaks2 = local_maxima(&sf2, GRID_NX, GRID_NY, 0.45);
    // map both true centers to grid maxima, count distinct strong maxima near each
    let two_ok = peaks2.len() >= 2;
    eprintln!("  two spirals @({ax:.0},{ay:.0}),({bx:.0},{by:.0}): {} S-maxima found  [{}]", peaks2.len(), pass(two_ok));

    // --- 3) concentric rings → S must NOT peak at the center (the guard test) ---
    let kk = TAU / 1.5f64.ln();
    let (rcx, rcy) = (0.5 * gs as f64, 0.5 * gs as f64);
    let rings = synth_at(gs, rcx, rcy, |r, _th| (kk * r.ln()).cos());
    let (sf3, max3) = synth_field(&rings, gs, rmin, rmax);
    let s_at_ring_center = s_score(&rings, gs, gs, rcx, rcy, rmin, rmax);
    // Guard PASS: the concentric center is NOT a strong S-max (well under the genuine
    // spiral peak), i.e. the on-axis exclusion is doing its job.
    let conc_ok = s_at_ring_center.0 < 0.5 * pk1 && max3 < 0.6 * pk1;
    eprintln!(
        "  concentric @({rcx:.0},{rcy:.0}): S@center={:.3} (on-axis s=0 E={:.3}) field-max={max3:.3}  vs spiral peak {pk1:.3}  [{}]",
        s_at_ring_center.0, s_at_ring_center.2, pass(conc_ok)
    );

    // panel: three S-over-center heatmaps + a D-strip readout of each true center.
    let mk = |name: &str, sf: &[f64], smax: f64, fld: &[f64], cx: f64, cy: f64| -> RgbImage {
        let heat = heatmap_tile(sf, GRID_NX, GRID_NY, 300, 300, smax, pal_heat, &format!("{name}: S over center (twilight_shifted)"));
        let ro = lp::compute_readout(fld, gs, gs, cx, cy, rmin, rmax);
        let ku = (rmax.ln() - rmin.ln()) / (lp::NU - 1) as f64;
        let blk = lp::candidate_block(
            &format!("{name} @true center: S-objective uses Radon (orange=pitch peak, grey=s0 excised)"),
            &ro,
            pal_seq,
            pal_div,
            &[(lp::NV / 2, Rgb([90, 130, 255])), (lp::NV / 4, Rgb([255, 80, 80]))],
            &lp::scale_marks(&ro, ku),
        );
        lp::hstack(&[heat, blk])
    };
    let r1 = mk("single spiral", &sf1, max1, &spiral, sx, sy);
    let r2 = mk("two spirals", &sf2, _max2, &two, ax, ay);
    let r3 = mk("concentric (GUARD)", &sf3, max3, &rings, rcx, rcy);
    let body = lp::vstack(&[r1, r2, r3]);
    let pass_all = single_ok && two_ok && conc_ok;
    let head = lp::banner(
        body.width(),
        &format!("STEP A — S-OVER-CENTER SYNTHETIC GATE  [pass={pass_all}]  single peaks @true · two is bimodal · concentric does NOT peak (guard)"),
    );
    lp::save(&lp::vstack(&[head, body]), &format!("{OUT_DIR}/synthetic_gate.png"))?;
    Ok(pass_all)
}

/// S-field over candidate centers of a `gs×gs` synthetic, plus its max (for the
/// shared heatmap normalization).
fn synth_field(field: &[f64], gs: usize, rmin: f64, rmax: f64) -> (Vec<f64>, f64) {
    let sf = s_field(field, gs, gs, rmin, rmax);
    let mx = sf.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
    (sf, mx)
}

/// Argmax of an S-field in **pixel** coordinates (returns the cell center px + value).
fn field_argmax(sf: &[f64], w: usize) -> (f64, f64, f64) {
    let (bi, &bv) = sf.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    let (fx, fy) = grid_center_of(bi, w, w);
    (fx, fy, bv)
}

/// Grid-cell center for a synthetic field of square side `gs` (uses the same
/// GRID_NX/NY lattice as the bench, scaled to the synthetic).
fn grid_center_of(idx: usize, w: usize, h: usize) -> (f64, f64) {
    grid_center(idx % GRID_NX, idx / GRID_NX, w, h)
}

/// Radial basin width: walk outward from the true center along 8 directions; report
/// the mean radius at which S first falls below half the peak. A wide basin means a
/// far-off nucleus seed is still inside it (predicts whether nucleus seeding works).
fn basin_width(field: &[f64], gs: usize, cx: f64, cy: f64, peak: f64, rmin: f64, rmax: f64) -> f64 {
    let half = 0.5 * peak;
    let mut acc = 0.0;
    let mut n = 0;
    for k in 0..8 {
        let th = k as f64 / 8.0 * TAU;
        let (s, c) = th.sin_cos();
        let mut r = 2.0;
        let mut last = rmax;
        while r < 0.35 * gs as f64 {
            let sv = s_score(field, gs, gs, cx + r * c, cy + r * s, rmin, rmax).0;
            if sv < half {
                last = r;
                break;
            }
            r += 2.0;
        }
        acc += last;
        n += 1;
    }
    acc / n as f64
}

// ===========================================================================
// Steps B/C/D — per-bench frame
// ===========================================================================

fn process_bench(b: &sp::Bench, pal_heat: &Palette, pal_seq: &Palette, pal_div: &Palette) -> Result<(), String> {
    let w = RW as usize;
    let h = RH as usize;
    let buf = lp::render_full(b.center, b.fw);
    let scalar = sp::smooth_scalar(&buf);

    // nucleus centers (the cheap seed source) ----------------------------------
    let nucs = lp::nuclei(b.center, b.fw, &buf);
    let cents = lp::centers(&nucs);
    let nuc_hubs: Vec<(f64, f64)> = cents.iter().filter(|c| c.kind == "hub").map(|c| (c.fx, c.fy)).collect();
    eprintln!("    {}: {} nuclei → {} nucleus-hub seeds", b.name, nucs.len(), nuc_hubs.len());

    // smooth preview (the human's eye-truth for where the spiral eyes are) -------
    let preview = render::shade_and_downsample(&buf.samples, RW, RH, 1, pal_heat, &sp::preview_params(), b.fw / RW as f64);
    let big_w = 760u32;
    let big_h = big_w * RH / RW;
    let base_preview = lp::downscale_rgb(&preview, big_w, big_h);

    // --- Step B: S-field over center, at BOTH bands (THE decisive artifact) ----
    let mut band_tiles: Vec<RgbImage> = Vec::new();
    for (bi, &(rmin, rmax)) in BANDS.iter().enumerate() {
        let sf = s_field(&scalar, w, h, rmin, rmax);
        let smax = sf.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
        let smaxima = local_maxima(&sf, GRID_NX, GRID_NY, 0.5);
        eprintln!(
            "      {} band{} (r {:.0}-{:.0}px): S-field max {:.3}, {} local maxima",
            b.name, bi, rmin, rmax, smax, smaxima.len()
        );

        // heatmap with eyes(=nucleus hubs) + S-maxima marked
        let mut heat = heatmap_img(&sf, GRID_NX, GRID_NY, big_w, big_h, smax, pal_heat);
        overlay_smaxima(&mut heat, &smaxima, big_w, big_h, w, h);
        overlay_points(&mut heat, &nuc_hubs, big_w, w, h, 8.0, COL_NUC, false);
        let heat = lp::titled(heat, &format!("S-field band{} r{:.0}-{:.0}px (twilight_shifted) · yellow=S-max · cyan=nucleus hub", bi, rmin, rmax));

        // matching preview with the same markers (so eyes ↔ maxima compare directly)
        let mut prev = base_preview.clone();
        overlay_smaxima(&mut prev, &smaxima, big_w, big_h, w, h);
        overlay_points(&mut prev, &nuc_hubs, big_w, w, h, 8.0, COL_NUC, false);
        let prev = lp::titled(prev, &format!("{} smooth + same S-maxima(yellow) — do the eyes sit on them?", b.name));
        band_tiles.push(lp::hstack(&[prev, heat]));
    }

    // --- Step C: climb from nucleus seeds AND a coarse grid (climb band only) ---
    let (rmin, rmax) = BANDS[CLIMB_BAND];
    let rho0 = 0.06 * w as f64;
    let nuc_climbs: Vec<(f64, f64, f64, f64, f64)> = nuc_hubs
        .iter()
        .map(|&(sx, sy)| {
            let (fx, fy, s) = climb(&scalar, w, h, sx, sy, rho0, rmin, rmax);
            (sx, sy, fx, fy, s)
        })
        .collect();
    let mut grid_seeds: Vec<(f64, f64)> = Vec::new();
    for iy in 0..SEED_NY {
        for ix in 0..SEED_NX {
            let fx = (ix as f64 + 0.5) * w as f64 / SEED_NX as f64;
            let fy = (iy as f64 + 0.5) * h as f64 / SEED_NY as f64;
            grid_seeds.push((fx, fy));
        }
    }
    let grid_climbs: Vec<(f64, f64, f64, f64, f64)> = grid_seeds
        .iter()
        .map(|&(sx, sy)| {
            let (fx, fy, s) = climb(&scalar, w, h, sx, sy, rho0, rmin, rmax);
            (sx, sy, fx, fy, s)
        })
        .collect();
    eprintln!(
        "      {} climb (band{CLIMB_BAND}): {} nucleus-seed climbs (best S {:.3}), {} grid-seed climbs (best S {:.3})",
        b.name,
        nuc_climbs.len(),
        nuc_climbs.iter().map(|c| c.4).fold(0.0f64, f64::max),
        grid_climbs.len(),
        grid_climbs.iter().map(|c| c.4).fold(0.0f64, f64::max),
    );

    let mut climb_overlay = base_preview.clone();
    for &(sx, sy, fx, fy, _) in &grid_climbs {
        draw_seed_to_climb(&mut climb_overlay, sx, sy, fx, fy, big_w, w, h, COL_GRID);
    }
    for &(sx, sy, fx, fy, _) in &nuc_climbs {
        draw_seed_to_climb(&mut climb_overlay, sx, sy, fx, fy, big_w, w, h, COL_NUC);
    }
    let climb_overlay = lp::titled(
        climb_overlay,
        &format!("{} CLIMBED centers — cyan=nucleus-seed, orange=grid-seed (dim dot=seed, ring=climbed). Do they reach the eyes?", b.name),
    );

    // --- Step D: characterize the climbed centers (distinct, strongest first) ---
    let mut climbed: Vec<(&'static str, f64, f64, f64)> = Vec::new();
    for &(_, _, fx, fy, s) in &nuc_climbs {
        climbed.push(("nuc", fx, fy, s));
    }
    for &(_, _, fx, fy, s) in &grid_climbs {
        climbed.push(("grid", fx, fy, s));
    }
    let distinct = dedup_distinct(&mut climbed, 0.05 * w as f64, 6);
    let mut char_blocks: Vec<RgbImage> = Vec::new();
    for (src, fx, fy, s) in &distinct {
        let ro = lp::compute_readout(&scalar, w, h, *fx, *fy, rmin, rmax);
        let ku = (rmax.ln() - rmin.ln()) / (lp::NU - 1) as f64;
        eprintln!(
            "        climbed[{src}] ({:.0},{:.0}) S={:.3}: presence {:.3} trough {:.2}{} hubness {:.3} pitch {:+.2} fold {} R2 {:.2}",
            fx, fy, s, ro.presence, ro.ang_trough, lp::trust_flag(&ro), ro.hubness, ro.pitch, ro.fold, ro.r2
        );
        let title = format!(
            "climbed[{src}] ({:.0},{:.0}) S={:.2}{} hub {:.2} pitch {:+.2} fold {} R2 {:.2}",
            fx, fy, s, lp::trust_flag(&ro), ro.hubness, ro.pitch, ro.fold, ro.r2
        );
        char_blocks.push(lp::candidate_block(
            &title,
            &ro,
            pal_seq,
            pal_div,
            &[(lp::NV / 2, Rgb([90, 130, 255])), (lp::NV / 4, Rgb([255, 80, 80]))],
            &lp::scale_marks(&ro, ku),
        ));
    }

    // compose the frame panel ----------------------------------------------------
    let dir = format!("{OUT_DIR}/{}", b.name);
    crate::ensure_parent_dir(&format!("{dir}/x"))?;
    let mut rows = band_tiles;
    rows.push(climb_overlay);
    rows.extend(char_blocks);
    let body = lp::vstack(&rows);
    let head = lp::banner(
        body.width(),
        &format!(
            "CHARACTERIZER-AS-FINDER  {}  ({})  center ({:.6},{:.6}) fw {:.3e}  [DIAGNOSIS ONLY — no quality claim]",
            b.name, b.note, b.center.re, b.center.im, b.fw
        ),
    );
    lp::save(&lp::vstack(&[head, body]), &format!("{dir}/panel.png"))?;

    // json: climbed centers (both seed sources) + S
    let mut j = String::new();
    let _ = write!(j, "{{\n  \"name\": \"{}\", \"fw\": {:.6e}, \"climb_band\": {:?},\n  \"nucleus_climbs\": [", b.name, b.fw, BANDS[CLIMB_BAND]);
    for (i, &(sx, sy, fx, fy, s)) in nuc_climbs.iter().enumerate() {
        let _ = write!(j, "{}{{\"seed\":[{:.1},{:.1}],\"climbed\":[{:.1},{:.1}],\"S\":{:.4}}}", if i > 0 { "," } else { "" }, sx, sy, fx, fy, s);
    }
    let _ = write!(j, "],\n  \"grid_climbs\": [");
    for (i, &(sx, sy, fx, fy, s)) in grid_climbs.iter().enumerate() {
        let _ = write!(j, "{}{{\"seed\":[{:.1},{:.1}],\"climbed\":[{:.1},{:.1}],\"S\":{:.4}}}", if i > 0 { "," } else { "" }, sx, sy, fx, fy, s);
    }
    j.push_str("]\n}\n");
    std::fs::write(format!("{dir}/hubfind.json"), &j).map_err(|e| format!("write json: {e}"))?;
    Ok(())
}

/// Greedily keep the strongest-S climbed centers separated by ≥ `min_sep` px, up to
/// `cap`. (Many seeds converge to the same eye — characterize each eye once.)
fn dedup_distinct(climbed: &mut [(&'static str, f64, f64, f64)], min_sep: f64, cap: usize) -> Vec<(&'static str, f64, f64, f64)> {
    climbed.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    let sep2 = min_sep * min_sep;
    let mut out: Vec<(&'static str, f64, f64, f64)> = Vec::new();
    for &c in climbed.iter() {
        if c.3 <= 0.0 {
            continue;
        }
        if out.iter().all(|o| (o.1 - c.1).powi(2) + (o.2 - c.2).powi(2) >= sep2) {
            out.push(c);
            if out.len() >= cap {
                break;
            }
        }
    }
    out
}

// ===========================================================================
// Step E — reference check (off native substrate, luminance, labeled)
// ===========================================================================

fn reference_panel(pal_heat: &Palette, pal_seq: &Palette, pal_div: &Palette) -> Result<(), String> {
    // (name, file, expectation)
    let refs = [
        ("delicacy", "delicacy.jpg", "two flanking 180° spiral hubs are the S-maxima (the central X is a junction, NOT a hub → correctly not an S-max); midpoint of the two climbed hubs = the X → characterize → expect R2≈0.82 from climbed (not hand-placed) centers"),
        ("helping-hands", "helping-hands-25.jpg", "a climb seeded near the hub lands on the hub (strong S, the prompt-2 pitch)"),
    ];
    let mut blocks: Vec<RgbImage> = Vec::new();
    let mut any = false;
    for (name, file, expect) in refs {
        let path = format!("{REFS_DIR}/{file}");
        let Some((lum, w, h)) = lp::load_luma(&path) else {
            eprintln!("  reference {path} absent — skipping (graceful).");
            continue;
        };
        any = true;
        let (rmin, rmax) = (6.0, (w.min(h) as f64) * 0.30);
        let sf = s_field(&lum, w, h, rmin, rmax);
        let smax = sf.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
        let smaxima = local_maxima(&sf, GRID_NX, GRID_NY, 0.5);

        // climb from a coarse grid (no nucleus field off-substrate)
        let rho0 = 0.06 * w as f64;
        let mut seeds: Vec<(f64, f64)> = Vec::new();
        for iy in 0..SEED_NY {
            for ix in 0..(SEED_NX + 1) {
                seeds.push(((ix as f64 + 0.5) * w as f64 / (SEED_NX + 1) as f64, (iy as f64 + 0.5) * h as f64 / SEED_NY as f64));
            }
        }
        let mut climbs: Vec<(&'static str, f64, f64, f64)> = seeds
            .par_iter()
            .map(|&(sx, sy)| {
                let (fx, fy, s) = climb(&lum, w, h, sx, sy, rho0, rmin, rmax);
                ("grid", fx, fy, s)
            })
            .collect();
        let hubs = dedup_distinct(&mut climbs, 0.06 * w as f64, 4);
        eprintln!("  ref {name} [SANITY, palette-contaminated]: S-field max {smax:.3}, {} distinct climbed hubs", hubs.len());
        for (i, (_, fx, fy, s)) in hubs.iter().enumerate() {
            let ro = lp::compute_readout(&lum, w, h, *fx, *fy, rmin, rmax);
            eprintln!("      hub{i} ({:.0},{:.0}) S={:.3}{} hub {:.3} pitch {:+.2} fold {} R2 {:.3} R4 {:.3}", fx, fy, s, lp::trust_flag(&ro), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4);
        }

        // delicacy: the X = midpoint of the two strongest climbed hubs; characterize it.
        let mut midpoint: Option<(f64, f64)> = None;
        if hubs.len() >= 2 {
            let (mx, my) = (0.5 * (hubs[0].1 + hubs[1].1), 0.5 * (hubs[0].2 + hubs[1].2));
            let ro = lp::compute_readout(&lum, w, h, mx, my, rmin, rmax);
            eprintln!(
                "      {name} midpoint of 2 strongest climbed hubs ({:.0},{:.0}): presence {:.3}{} hubness {:.3} pitch {:+.2} fold {} R2 {:.3} R4 {:.3}  <- expect X / R2-strong",
                mx, my, ro.presence, lp::trust_flag(&ro), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4
            );
            midpoint = Some((mx, my));
        }

        // visuals: preview + heatmap, both marked; plus readout of the strongest hub
        // and (for delicacy) the midpoint.
        let pw = 360u32;
        let ph = (pw as f64 * h as f64 / w as f64) as u32;
        let lmax = lp::pctl(&lum, 0.99).max(1e-6);
        let lprev = lp::colorize(&lum, w, h, 0.0, lmax, pal_seq);
        let mut prev = lp::downscale_rgb(&lprev, pw, ph);
        overlay_smaxima(&mut prev, &smaxima, pw, ph, w, h);
        for (_, fx, fy, _) in &hubs {
            lp::draw_marker(&mut prev, *fx * pw as f64 / w as f64, *fy * ph as f64 / h as f64, 6.0, COL_GRID);
        }
        if let Some((mx, my)) = midpoint {
            lp::draw_marker(&mut prev, mx * pw as f64 / w as f64, my * ph as f64 / h as f64, 6.0, COL_EYE);
        }
        let prev = lp::titled(prev, &format!("{name} luminance · orange=climbed hub · magenta=hub-pair midpoint(X) [SANITY]"));

        let mut heat = heatmap_img(&sf, GRID_NX, GRID_NY, pw, ph, smax, pal_heat);
        overlay_smaxima(&mut heat, &smaxima, pw, ph, w, h);
        let heat = lp::titled(heat, &format!("{name} S-field (twilight_shifted) · yellow=S-max"));

        // readout of the X (midpoint) if present, else the strongest hub
        let (cx, cy, lab) = midpoint.map(|(x, y)| (x, y, "X(midpoint)")).unwrap_or((hubs[0].1, hubs[0].2, "hub0"));
        let ro = lp::compute_readout(&lum, w, h, cx, cy, rmin, rmax);
        let ku = (rmax.ln() - rmin.ln()) / (lp::NU - 1) as f64;
        let blk = lp::candidate_block(
            &format!("{name} {lab} ({:.0},{:.0}){} hub {:.2} pitch {:+.2} fold {} R2 {:.2} R4 {:.2}", cx, cy, lp::trust_flag(&ro), ro.hubness, ro.pitch, ro.fold, ro.r2, ro.r4),
            &ro,
            pal_seq,
            pal_div,
            &[(lp::NV / 2, Rgb([90, 130, 255])), (lp::NV / 4, Rgb([255, 80, 80]))],
            &lp::scale_marks(&ro, ku),
        );
        let row = lp::hstack(&[prev, heat, blk]);
        blocks.push(lp::titled(row, &format!("{name} — {expect}")));
    }
    if !any {
        eprintln!("  no reference images under {REFS_DIR}/ — ref panel skipped.");
        return Ok(());
    }
    let body = lp::vstack(&blocks);
    let head = lp::banner(body.width(), "STEP E — REFERENCE CHECK  (luminance, palette-contaminated, OFF native substrate — sanity only; closes the jitter loop)");
    lp::save(&lp::vstack(&[head, body]), &format!("{OUT_DIR}/reference_check.png"))?;
    Ok(())
}

// ===========================================================================
// Visualization helpers (heatmap + overlays; generic tiles reused from logpolar)
// ===========================================================================

/// An S-field grid colorized under `pal` (normalized to `[0,smax]`) and upscaled to
/// `tw×th`. No title (caller overlays markers then titles).
fn heatmap_img(sf: &[f64], nx: usize, ny: usize, tw: u32, th: u32, smax: f64, pal: &Palette) -> RgbImage {
    let img = lp::colorize(sf, nx, ny, 0.0, smax, pal);
    lp::downscale_rgb(&img, tw, th)
}

/// As `heatmap_img` but titled (for the synthetic panel where no markers are drawn).
fn heatmap_tile(sf: &[f64], nx: usize, ny: usize, tw: u32, th: u32, smax: f64, pal: &Palette, title: &str) -> RgbImage {
    lp::titled(heatmap_img(sf, nx, ny, tw, th, smax, pal), title)
}

/// Local maxima of an S-field (≥ all 8 neighbours and ≥ `frac` of the global max),
/// returned as grid `(ix,iy)`.
fn local_maxima(sf: &[f64], nx: usize, ny: usize, frac: f64) -> Vec<(usize, usize)> {
    let mx = sf.iter().cloned().fold(0.0f64, f64::max);
    let thresh = frac * mx;
    let mut out = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let v = sf[iy * nx + ix];
            if v < thresh || v <= 0.0 {
                continue;
            }
            let mut is_max = true;
            'nb: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nxx = ix as i32 + dx;
                    let nyy = iy as i32 + dy;
                    if nxx < 0 || nyy < 0 || nxx >= nx as i32 || nyy >= ny as i32 {
                        continue;
                    }
                    if sf[nyy as usize * nx + nxx as usize] > v {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                out.push((ix, iy));
            }
        }
    }
    out
}

/// Draw S-field local-maxima markers onto a `tw×th` tile (grid cells mapped to the
/// frame `w×h` then to tile pixels).
fn overlay_smaxima(img: &mut RgbImage, maxima: &[(usize, usize)], tw: u32, th: u32, w: usize, h: usize) {
    for &(ix, iy) in maxima {
        let (fx, fy) = grid_center(ix, iy, w, h);
        let tx = fx * tw as f64 / w as f64;
        let ty = fy * th as f64 / h as f64;
        lp::draw_marker(img, tx, ty, 7.0, COL_SMAX);
    }
}

/// Draw a set of frame-pixel points as markers on a tile of width `tw` (height
/// derived from `w,h`). `small` → a dim dot rather than a ring.
fn overlay_points(img: &mut RgbImage, pts: &[(f64, f64)], tw: u32, w: usize, h: usize, r: f64, col: Rgb<u8>, small: bool) {
    let th = (tw as f64 * h as f64 / w as f64) as u32;
    for &(fx, fy) in pts {
        let tx = fx * tw as f64 / w as f64;
        let ty = fy * th as f64 / h as f64;
        lp::draw_marker(img, tx, ty, if small { 3.0 } else { r }, col);
    }
}

/// Seed→climbed: a dim dot at the seed, a bright ring at the climbed center, and a
/// connecting line, on a tile of width `tw`.
fn draw_seed_to_climb(img: &mut RgbImage, sx: f64, sy: f64, fx: f64, fy: f64, tw: u32, w: usize, h: usize, col: Rgb<u8>) {
    let th = (tw as f64 * h as f64 / w as f64) as u32;
    let s = (sx * tw as f64 / w as f64, sy * th as f64 / h as f64);
    let e = (fx * tw as f64 / w as f64, fy * th as f64 / h as f64);
    draw_line(img, s.0, s.1, e.0, e.1, Rgb([col[0] / 2, col[1] / 2, col[2] / 2]));
    lp::draw_marker(img, s.0, s.1, 3.0, Rgb([col[0] / 2, col[1] / 2, col[2] / 2]));
    lp::draw_marker(img, e.0, e.1, 9.0, col);
}

/// Bresenham-ish line (clipped) for seed→climb connectors.
fn draw_line(img: &mut RgbImage, x0: f64, y0: f64, x1: f64, y1: f64, col: Rgb<u8>) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs())).ceil().max(1.0) as i64;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = (x0 + (x1 - x0) * t).round() as i64;
        let y = (y0 + (y1 - y0) * t).round() as i64;
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, col);
        }
    }
}

fn pass(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
