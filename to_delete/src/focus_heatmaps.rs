//! **Throwaway diagnostic — focus heatmaps (three scoring fields + combined).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Answers one question
//! (see `prompts/focus-heatmaps-three-field.md`): can we locate the **points of
//! focus** in a seed — the dense, organized places a human would zoom into — by
//! scoring a sliding window across the frame at three scales under three fields,
//! and do the combined peaks land where Matt's eye says the focus is?
//!
//! **No zoom, no descent, no composition, no band change.** It re-renders six
//! logged seeds (the `generate`/`reject_corridor` surface verbatim), sweeps a
//! 16:9 window at three scales, and emits one annotated panel per seed plus a
//! JSON log under `data/focus_probe/`.
//!
//! The three fields, each scored per window:
//!  1. **Rotational-symmetry correlation** (image-space) — the window's
//!     self-similarity under rotation about its center (max Pearson over a set of
//!     angles, on a downsampled patch, inside the inscribed disk). Correlated on
//!     the **local edge-energy** field (not raw smooth-iter): a smooth empty
//!     background is locally near-radially symmetric and would otherwise score
//!     high everywhere — the edge field is flat there (→ guarded to 0) so the
//!     score reflects *structural* symmetry at spiral hubs / radial foci.
//!  2. **Nucleus / low-period field** (orbit-space) — Newton-confirmed low-period
//!     minibrot nuclei (the parked navigation stack, used here as a *scoring
//!     field*, not a descent target), Gaussian-splatted per pixel and averaged in
//!     the window. High where the mathematical foci sit.
//!  3. **Local escape-n distribution** (orbit-space) — the **raw** std of escape
//!     iterations over the window (variance-is-good is *not* baked in), with a
//!     hard reject mask for near-constant / near-empty windows.
//!
//! Scale is collapsed per field by **max-over-scale** (the winning scale is the
//! future zoom-depth hint). The combined field is the PROVISIONAL normalized
//! product (the real combination rule is Matt's to set after seeing the three).
//!
//! Run: `cargo test --release --lib focus_heatmaps -- --ignored --nocapture`.

use std::f64::consts::PI;
use std::fmt::Write as _;

use astro_float::BigFloat;
use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{Trap, TrapShape};
use crate::cli::BackendChoice;
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::font;
use crate::navigate::{atom_candidates_spatial, newton_nucleus};
use crate::palette::{linear_to_srgb, Palette};
use crate::{hp, probe, render};

// --- fixed regime (matches the logs we re-render from) -----------------------
const MAXITER: u32 = 1000;
const BAILOUT: f64 = 1e6;
/// Analysis + seed render resolution (16:9). f64 cheap-regime (fw ~0.004–0.05).
const RW: u32 = 960;
const RH: u32 = 540;

const CORRIDOR_LOG: &str = "data/generated/reject_corridor/draws.jsonl";
const OUT_DIR: &str = "data/focus_probe";
const HEATMAP_CMAP: &str = "inferno";
const SEED_CMAP: &str = "twilight_shifted";
const CMAP_FILE: &str = "data/palettes/clean_colormaps.json";

// --- the sweep ---------------------------------------------------------------
/// Position grid (window centers). Stride ≈ RW/GX ≈ 24px.
const GX: usize = 40;
const GY: usize = 23;
/// Window widths as a fraction of the frame width; height = width·9/16 (the
/// wallpaper aspect). Three scales: the winning scale is the zoom-depth hint.
const SCALES: [f64; 3] = [0.30, 0.18, 0.11];
/// Downsample size for the rotational-symmetry patch (coarse — symmetry is a
/// low-frequency feature; caps cost regardless of window size).
const SYM_D: usize = 48;
/// Rotation angles (degrees) tested for self-similarity. Covers approximate
/// 2/3/4/6-fold symmetry; the score is the max correlation over them.
const SYM_ANGLES_DEG: [f64; 4] = [60.0, 90.0, 120.0, 180.0];

// --- peak selection (spatial-diversity) --------------------------------------
/// Minimum separation between selected peaks, as a fraction of the frame
/// **width**, measured in aspect-corrected frame space. Replaces the old tiny
/// fixed grid-cell suppression radius: peaks must be ≥ this far apart, which
/// forces spatial spread along the boundary ridge AND out into the interior, so
/// isolated mid-density foci survive instead of being crushed by the densest
/// ridge maxima. Tunable. At 16:9 a radius of 0.15 still admits ≈25 well-spread
/// peaks, so it forces spread without starving genuine distinct foci.
const PEAK_DIV_RADIUS_FRAC: f64 = 0.15;

// --- nucleus field -----------------------------------------------------------
/// Only periods ≤ this are Newton-refined ("low-period", and cheap). Higher
/// periods are not the human-scale foci and the BigFloat refine cost grows.
const NUC_PERIOD_CAP: u32 = 80;
/// Spatial dedup cell (px) for the broad nucleus scan.
const NUC_CELL_PX: u32 = 8;
/// Gaussian splat sigma (px) for each confirmed nucleus.
const NUC_SIGMA: f64 = 6.0;

// --- field 3 reject floor ----------------------------------------------------
/// A window whose escape-n std is below this **fraction of the frame-global**
/// escape-n std is degenerate (near-constant *relative to this frame*) → masked.
/// Frame-relative because the absolute spread of a fast-escape background varies
/// enormously between seeds; a fixed floor either never bites or masks everything.
const REJECT_STD_FRAC: f64 = 0.15;
/// A window with fewer than this fraction of escaped pixels is degenerate (an
/// interior/instant-escape void) → masked.
const REJECT_ESC_FRAC: f64 = 0.10;

// --- display -----------------------------------------------------------------
const HM_W: u32 = 460;
const HM_H: u32 = 259; // 460 * 540/960
const TITLE_H: u32 = 16;
const CBAR_W: u32 = 16;
const CBAR_GUT: u32 = 46; // gutter to the right of the colorbar for min/max labels
const PAD: u32 = 6;

/// A labeled seed to re-render (key on `draw_index` in the corridor log).
struct Seed {
    name: String,
    kind: &'static str, // "cut-good" | "dense-anchor"
    center: Complex<f64>,
    fw: f64,
}

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn focus_heatmaps() {
    run().expect("focus-heatmaps probe");
}

fn run() -> Result<(), String> {
    let corridor =
        std::fs::read_to_string(CORRIDOR_LOG).map_err(|e| format!("read {CORRIDOR_LOG}: {e}"))?;

    // Seeds: four CUT-but-good corridor frames (the hard case) + two dense
    // anchors. All keyed on draw_index in the corridor log.
    let want: [(usize, &str); 6] = [
        (2095, "cut-good"),
        (1295, "cut-good"),
        (1875, "cut-good"),
        (4721, "cut-good"),
        (2361, "dense-anchor"),
        (512, "dense-anchor"),
    ];
    let mut seeds: Vec<Seed> = Vec::new();
    for (di, kind) in want {
        let line = find_line(&corridor, "draw_index", di as f64)
            .ok_or_else(|| format!("draw_index {di} not in {CORRIDOR_LOG}"))?;
        let g = |k: &str| fnum(line, k).ok_or_else(|| format!("D{di}: missing {k}"));
        seeds.push(Seed {
            name: format!("D{di:05}"),
            kind,
            center: Complex::new(g("center_re")?, g("center_im")?),
            fw: g("frame_width")?,
        });
    }

    // Palettes: heatmaps under `inferno` (readable magnitude); seed renders under
    // the chosen diagnostic palette `twilight_shifted`.
    let cmaps = std::fs::read_to_string(CMAP_FILE).map_err(|e| format!("read {CMAP_FILE}: {e}"))?;
    let heat = Palette::from_srgb8_stops(
        HEATMAP_CMAP,
        &probe::load_colormap(&cmaps, HEATMAP_CMAP)?,
        false,
    );
    let seed_pal = Palette::from_srgb8_stops(
        SEED_CMAP,
        &probe::load_colormap(&cmaps, SEED_CMAP)?,
        false,
    );

    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    eprintln!(
        "focus-heatmaps: {} seeds, render {RW}x{RH} ss1 maxiter {MAXITER}, grid {GX}x{GY}, \
         scales {:?} (window widths {:?}px)",
        seeds.len(),
        SCALES,
        SCALES.map(|s| (s * RW as f64).round() as u32),
    );

    // --- diagnosis-first: validate the nucleus primitive in isolation on one
    //     seed (a dense anchor) BEFORE building any field. Report a couple of
    //     detected low-period nuclei; if it is rotted, stop here. ---
    {
        let s = seeds.iter().find(|s| s.kind == "dense-anchor").unwrap();
        eprintln!(
            "\n=== diagnosis-first: nucleus primitive on {} ({}), center ({:.9}, {:.9}) fw {:.4e} ===",
            s.name, s.kind, s.center.re, s.center.im, s.fw
        );
        let buf = render_seed(s, trap);
        let nuclei = confirmed_nuclei(s, &buf);
        if nuclei.is_empty() {
            return Err(format!(
                "nucleus primitive produced NO confirmed low-period nucleus on {} — \
                 navigation stack may be rotted; stopping before building fields.",
                s.name
            ));
        }
        let mut by_period = nuclei.clone();
        by_period.sort_by_key(|n| n.period);
        eprintln!(
            "  {} confirmed low-period nuclei (period ≤ {NUC_PERIOD_CAP}); lowest periods:",
            nuclei.len()
        );
        for n in by_period.iter().take(5) {
            eprintln!(
                "    period {:>3}  px=({:>4},{:>3})  |z_p|^2 {:.2e}  weight {:.3}",
                n.period, n.px, n.py, n.final_z2, n.weight
            );
        }
        eprintln!("  nucleus primitive OK — proceeding to the three-field sweep.\n");
    }

    crate::ensure_parent_dir(&format!("{OUT_DIR}/x"))?;
    let t_all = std::time::Instant::now();
    let mut manifest = String::from("{\n");
    let _ = write!(
        manifest,
        "  \"probe\": \"focus-heatmaps-three-field\",\n  \"render\": {{ \"w\": {RW}, \"h\": {RH}, \"ss\": 1, \"maxiter\": {MAXITER}, \"bailout\": {BAILOUT} }},\n  \"grid\": {{ \"gx\": {GX}, \"gy\": {GY} }},\n  \"scales\": {SCALES:?},\n  \"seeds\": [\n"
    );

    for (si, s) in seeds.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let buf = render_seed(s, trap);
        let result = process_seed(s, &buf, &heat, &seed_pal)?;
        eprintln!(
            "  [{}/{}] {} ({}) -> {} peaks, {:.1}s",
            si + 1,
            seeds.len(),
            s.name,
            s.kind,
            result.peaks.len(),
            t0.elapsed().as_secs_f64()
        );
        let _ = write!(
            manifest,
            "    {{ \"name\": \"{}\", \"kind\": \"{}\", \"center_re\": {:.15e}, \"center_im\": {:.15e}, \"frame_width\": {:.6e}, \"panel\": \"{}/{}/panel.png\", \"peaks\": {} }}{}\n",
            s.name, s.kind, s.center.re, s.center.im, s.fw, OUT_DIR, s.name,
            result.peaks_json,
            if si + 1 < seeds.len() { "," } else { "" }
        );
    }
    manifest.push_str("  ]\n}\n");
    std::fs::write(format!("{OUT_DIR}/manifest.json"), &manifest)
        .map_err(|e| format!("write manifest: {e}"))?;

    eprintln!(
        "\nfocus-heatmaps done in {:.1}s — panels + per-seed JSON under {OUT_DIR}/, manifest {OUT_DIR}/manifest.json",
        t_all.elapsed().as_secs_f64()
    );
    eprintln!(
        "read: do the combined peaks land on the focus points Matt would pick, and which field \
         carries the signal vs. adds noise? (no quality claim, no recommendation)"
    );
    Ok(())
}

/// Render one seed's sample buffer (f64, ss1 — these frames are shallow).
fn render_seed(s: &Seed, trap: Trap) -> render::SampleBuffer {
    let prec = hp::prec_bits(RW, s.fw);
    let cre = BigFloat::from_f64(s.center.re, prec);
    let cim = BigFloat::from_f64(s.center.im, prec);
    probe::render_mandel_panel(
        &cre, &cim, s.center, s.fw, RW, RH, 1, MAXITER, BAILOUT, prec, trap, BackendChoice::F64,
    )
    .buf
}

// ===========================================================================
// Nucleus field primitive (Newton-confirmed low-period nuclei)
// ===========================================================================

#[derive(Clone)]
struct ConfNucleus {
    period: u32,
    px: usize,
    py: usize,
    final_z2: f64,
    weight: f64,
}

/// Newton-confirm the low-period atom-domain minima of a rendered seed. Reuses
/// the parked navigation primitives (`atom_candidates_spatial` + `newton_nucleus`)
/// as a **scoring field** — every confirmed nucleus is marked, none is chased.
fn confirmed_nuclei(s: &Seed, buf: &render::SampleBuffer) -> Vec<ConfNucleus> {
    let prec = hp::prec_bits(RW, s.fw) + 32;
    let cre = BigFloat::from_f64(s.center.re, prec);
    let cim = BigFloat::from_f64(s.center.im, prec);
    let half_w = s.fw * 0.5;
    let half_h = s.fw * (RH as f64 / RW as f64) * 0.5;

    let cands = atom_candidates_spatial(buf, RW, RH, s.fw, MAXITER, NUC_CELL_PX);
    cands
        .par_iter()
        .filter(|c| c.period >= 2 && c.period <= NUC_PERIOD_CAP)
        .filter_map(|c| {
            let guess_re = cre.add(&BigFloat::from_f64(c.dc_re, prec), prec, astro_float::RoundingMode::ToEven);
            let guess_im = cim.add(&BigFloat::from_f64(c.dc_im, prec), prec, astro_float::RoundingMode::ToEven);
            let nuc = newton_nucleus(&guess_re, &guess_im, c.period, s.fw, prec)?;
            // In-frame check on the refined nucleus; map to a display pixel.
            let ndr = hp::to_f64(&nuc.re.sub(&cre, prec, astro_float::RoundingMode::ToEven));
            let ndi = hp::to_f64(&nuc.im.sub(&cim, prec, astro_float::RoundingMode::ToEven));
            if ndr.abs() > half_w || ndi.abs() > half_h {
                return None;
            }
            let fx = (ndr / s.fw + 0.5) * RW as f64;
            let fy = (0.5 - ndi / (s.fw * (RH as f64 / RW as f64))) * RH as f64;
            if !(fx.is_finite() && fy.is_finite()) {
                return None;
            }
            let px = (fx as i64).clamp(0, RW as i64 - 1) as usize;
            let py = (fy as i64).clamp(0, RH as i64 - 1) as usize;
            // Nucleus-ness favours low period (the human-scale foci).
            let weight = 1.0 / (nuc.period as f64).sqrt();
            Some(ConfNucleus { period: nuc.period, px, py, final_z2: nuc.final_z2, weight })
        })
        .collect()
}

/// Splat confirmed nuclei into a per-pixel Gaussian density field.
fn nucleus_field(nuclei: &[ConfNucleus]) -> Vec<f64> {
    let w = RW as usize;
    let h = RH as usize;
    let mut f = vec![0.0f64; w * h];
    let sigma = NUC_SIGMA;
    let rad = (3.0 * sigma).ceil() as i64;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    for n in nuclei {
        let cx = n.px as i64;
        let cy = n.py as i64;
        for dy in -rad..=rad {
            let y = cy + dy;
            if y < 0 || y >= h as i64 {
                continue;
            }
            for dx in -rad..=rad {
                let x = cx + dx;
                if x < 0 || x >= w as i64 {
                    continue;
                }
                let r2 = (dx * dx + dy * dy) as f64;
                f[y as usize * w + x as usize] += n.weight * (-r2 * inv2s2).exp();
            }
        }
    }
    f
}

// ===========================================================================
// The sweep: three fields over (position × scale)
// ===========================================================================

/// Per-scale fields plus the collapsed/combined results for one seed.
struct SweepResult {
    f1: Vec<f64>,         // symmetry, max-over-scale, GX*GY
    f2: Vec<f64>,         // nucleus density, max-over-scale (normalized later)
    f3raw: Vec<f64>,      // escape-n std, max-over-scale (raw)
    reject: Vec<bool>,    // true = degenerate at every scale
    combined: Vec<f64>,   // PROVISIONAL normalized product, max-over-scale
    win_scale: Vec<u8>,   // winning scale index for the combined field
}

fn sweep(s: &Seed, buf: &render::SampleBuffer, nucfield: &[f64]) -> SweepResult {
    let w = RW as usize;
    let h = RH as usize;
    let samples = &buf.samples; // ss1 → one sample per pixel

    // Escaped mask + escape-n value field (interior filled with frame-max so the
    // set boundary reads as a plateau edge, not a spike — same trick as the
    // detail-clause bench).
    let max_esc = samples
        .iter()
        .filter(|s| s.escaped)
        .map(|s| s.smooth_iter)
        .fold(f64::NEG_INFINITY, f64::max);
    let fill = if max_esc.is_finite() { max_esc } else { 0.0 };
    let escaped: Vec<bool> = samples.iter().map(|s| s.escaped).collect();
    let escn: Vec<f64> = samples.iter().map(|s| if s.escaped { s.smooth_iter } else { 0.0 }).collect();
    let sfield: Vec<f64> = samples
        .iter()
        .map(|s| if s.escaped { s.smooth_iter } else { fill })
        .collect();

    // Symmetry is correlated on the **local edge-energy** field, not the raw
    // smooth-iter field: a smooth empty background is locally near-radially
    // symmetric and would otherwise score ~1.0 everywhere (rewarding emptiness,
    // the opposite of "spiral hub"). Edge energy is ~constant (→ guarded to 0) on
    // a flat gradient and structured where there is real detail, so the symmetry
    // score reflects *structural* self-similarity. Forward-diff gradient magnitude.
    let efield: Vec<f64> = {
        let mut e = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let gx = sfield[i.min(y * w + (x + 1).min(w - 1))] - sfield[i];
                let gy = sfield[(y + 1).min(h - 1) * w + x] - sfield[i];
                e[i] = (gx * gx + gy * gy).sqrt();
            }
        }
        e
    };

    // Frame-global escape-n std (over escaped pixels) → the relative reject floor.
    let (gsum, gsum2, gn) = samples.iter().filter(|s| s.escaped).fold(
        (0.0f64, 0.0f64, 0usize),
        |(a, b, c), s| (a + s.smooth_iter, b + s.smooth_iter * s.smooth_iter, c + 1),
    );
    let global_std = if gn >= 2 {
        let m = gsum / gn as f64;
        (gsum2 / gn as f64 - m * m).max(0.0).sqrt()
    } else {
        0.0
    };
    let reject_std_abs = (global_std * REJECT_STD_FRAC).max(1e-6);

    let n_scales = SCALES.len();
    let win_px: Vec<(usize, usize)> = SCALES
        .iter()
        .map(|&fr| {
            let ww = ((fr * RW as f64).round() as usize).max(4);
            let wh = ((ww as f64 * 9.0 / 16.0).round() as usize).max(4);
            (ww, wh)
        })
        .collect();

    // For each grid position, compute (sym, nuc_mean, escn_std, esc_frac) at each
    // scale. Parallelize over positions.
    let per_pos: Vec<Vec<[f64; 5]>> = (0..GX * GY)
        .into_par_iter()
        .map(|gi| {
            let gx = gi % GX;
            let gy = gi / GX;
            let pcx = ((gx as f64 + 0.5) * RW as f64 / GX as f64) as usize;
            let pcy = ((gy as f64 + 0.5) * RH as f64 / GY as f64) as usize;
            let mut out = Vec::with_capacity(n_scales);
            for sc in 0..n_scales {
                let (ww, wh) = win_px[sc];
                let (x0, x1, y0, y1) = clip_window(pcx, pcy, ww, wh, w, h);
                let (corr, medge) = symmetry_score(&efield, w, x0, x1, y0, y1);
                let (nuc_mean, escn_std, esc_frac) =
                    window_stats(nucfield, &escaped, &escn, w, x0, x1, y0, y1);
                out.push([corr, medge, nuc_mean, escn_std, esc_frac]);
            }
            out
        })
        .collect();

    // Frame-max window edge energy → the structure-presence gate for symmetry.
    let mut max_medge = 1e-12f64;
    for p in &per_pos {
        for sc in p {
            max_medge = max_medge.max(sc[1]);
        }
    }
    // F1 = rotational correlation **gated by structure presence** (window edge
    // energy / frame max): Pearson is scale-invariant, so a smooth empty window
    // correlates ~1.0; gating by edge magnitude makes F1 fire on symmetric
    // *structure* (spiral hubs), not on flat backgrounds.
    let sym_of = |c: &[f64; 5]| -> f64 { c[0] * (c[1] / max_medge).clamp(0.0, 1.0) };

    // Global maxima (over pos × scale) for cross-scale normalization.
    let mut max_sym = 1e-12f64;
    let mut max_nuc = 1e-12f64;
    let mut max_std = 1e-12f64;
    for p in &per_pos {
        for sc in p {
            max_sym = max_sym.max(sym_of(sc));
            max_nuc = max_nuc.max(sc[2]);
            max_std = max_std.max(sc[3]);
        }
    }

    // Collapse per field by max-over-scale; combined = max-over-scale of the
    // per-scale normalized product (so the winning scale is well-defined).
    let n = GX * GY;
    let mut f1 = vec![0.0f64; n];
    let mut f2 = vec![0.0f64; n];
    let mut f3raw = vec![0.0f64; n];
    let mut reject = vec![true; n];
    let mut combined = vec![0.0f64; n];
    let mut win_scale = vec![0u8; n];

    for gi in 0..n {
        let scales = &per_pos[gi];
        let mut best_comb = -1.0;
        for (sc, vals) in scales.iter().enumerate() {
            let sym = sym_of(vals);
            let nuc = vals[2];
            let std = vals[3];
            let esc = vals[4];
            f1[gi] = f1[gi].max(sym);
            f2[gi] = f2[gi].max(nuc);
            f3raw[gi] = f3raw[gi].max(std);
            // Reject floor (degenerate windows): near-constant escape-n OR too few
            // escaped pixels.
            let masked = std < reject_std_abs || esc < REJECT_ESC_FRAC;
            if !masked {
                reject[gi] = false;
            }
            // Combined (PROVISIONAL): normalized product (dense AND organized AND
            // non-degenerate). Field 3 enters as its raw value above the reject
            // floor (zero when masked).
            let n1 = sym / max_sym;
            let n2 = nuc / max_nuc;
            let n3 = if masked { 0.0 } else { std / max_std };
            let comb = n1 * n2 * n3;
            if comb > best_comb {
                best_comb = comb;
                win_scale[gi] = sc as u8;
            }
        }
        combined[gi] = best_comb.max(0.0);
    }

    let _ = s; // (kept symmetric with the other helpers' signatures)
    SweepResult { f1, f2, f3raw, reject, combined, win_scale }
}

/// Clip a window of size `ww×wh` centred at `(cx,cy)` to `[0,w)×[0,h)`.
/// Returns inclusive-exclusive bounds `(x0,x1,y0,y1)`.
fn clip_window(
    cx: usize,
    cy: usize,
    ww: usize,
    wh: usize,
    w: usize,
    h: usize,
) -> (usize, usize, usize, usize) {
    let hx = ww / 2;
    let hy = wh / 2;
    let x0 = cx.saturating_sub(hx);
    let y0 = cy.saturating_sub(hy);
    let x1 = (cx + hx + 1).min(w);
    let y1 = (cy + hy + 1).min(h);
    (x0, x1.max(x0 + 1), y0, y1.max(y0 + 1))
}

/// Nucleus-density mean + escape-n std + escaped fraction over a window.
fn window_stats(
    nucfield: &[f64],
    escaped: &[bool],
    escn: &[f64],
    w: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> (f64, f64, f64) {
    let mut nuc_sum = 0.0;
    let mut npix = 0usize;
    let mut esc_n = 0usize;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for y in y0..y1 {
        let row = y * w;
        for x in x0..x1 {
            let i = row + x;
            nuc_sum += nucfield[i];
            npix += 1;
            if escaped[i] {
                let v = escn[i];
                sum += v;
                sum2 += v * v;
                esc_n += 1;
            }
        }
    }
    let nuc_mean = if npix > 0 { nuc_sum / npix as f64 } else { 0.0 };
    let esc_frac = if npix > 0 { esc_n as f64 / npix as f64 } else { 0.0 };
    let std = if esc_n >= 2 {
        let m = sum / esc_n as f64;
        (sum2 / esc_n as f64 - m * m).max(0.0).sqrt()
    } else {
        0.0
    };
    (nuc_mean, std, esc_frac)
}

/// Rotational-symmetry score for a window. Downsamples the given field (the
/// local edge-energy field — see `sweep`) to a `SYM_D×SYM_D` patch, then returns
/// `(corr, mean_edge)`:
///  - `corr` = max Pearson correlation (clamped ≥0) between the patch and its
///    rotated copy over the inscribed disk, across `SYM_ANGLES_DEG`.
///  - `mean_edge` = mean patch (edge-energy) value over the disk — the
///    structure-presence the caller gates `corr` by (Pearson is scale-invariant,
///    so `corr` alone reads high on smooth empty windows).
fn symmetry_score(
    sfield: &[f64],
    w: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> (f64, f64) {
    let d = SYM_D;
    let ww = (x1 - x0) as f64;
    let wh = (y1 - y0) as f64;
    // Area-average downsample window → d×d patch.
    let mut patch = vec![0.0f64; d * d];
    for j in 0..d {
        let sy0 = y0 + (j as f64 / d as f64 * wh) as usize;
        let sy1 = (y0 + ((j + 1) as f64 / d as f64 * wh) as usize).max(sy0 + 1).min(y1);
        for i in 0..d {
            let sx0 = x0 + (i as f64 / d as f64 * ww) as usize;
            let sx1 = (x0 + ((i + 1) as f64 / d as f64 * ww) as usize).max(sx0 + 1).min(x1);
            let mut acc = 0.0;
            let mut cnt = 0usize;
            for sy in sy0..sy1 {
                let row = sy * w;
                for sx in sx0..sx1 {
                    acc += sfield[row + sx];
                    cnt += 1;
                }
            }
            patch[j * d + i] = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
        }
    }

    // Inscribed-disk mask (indices once; shared across angles).
    let c = (d as f64 - 1.0) * 0.5;
    let rdisk = c; // radius = half the patch
    let r2 = rdisk * rdisk;
    let mut disk: Vec<usize> = Vec::with_capacity(d * d);
    for j in 0..d {
        for i in 0..d {
            let dx = i as f64 - c;
            let dy = j as f64 - c;
            if dx * dx + dy * dy <= r2 {
                disk.push(j * d + i);
            }
        }
    }
    if disk.len() < 8 {
        return (0.0, 0.0);
    }
    let mean_edge = disk.iter().map(|&i| patch[i]).sum::<f64>() / disk.len() as f64;

    let mut best = 0.0f64;
    for &deg in &SYM_ANGLES_DEG {
        let th = deg * PI / 180.0;
        let (st, ct) = th.sin_cos();
        // Pearson over the disk between patch and its rotated sample.
        let mut sa = 0.0;
        let mut sb = 0.0;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        let mut sab = 0.0;
        let mut nn = 0.0;
        for &idx in &disk {
            let i = (idx % d) as f64;
            let j = (idx / d) as f64;
            // rotate sampling coords by +θ about the centre
            let dx = i - c;
            let dy = j - c;
            let srx = ct * dx - st * dy + c;
            let sry = st * dx + ct * dy + c;
            let b = bilinear(&patch, d, srx, sry);
            let a = patch[idx];
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
            nn += 1.0;
        }
        let cova = saa - sa * sa / nn;
        let covb = sbb - sb * sb / nn;
        if cova <= 1e-9 || covb <= 1e-9 {
            continue; // flat patch / flat rotation → no symmetry signal
        }
        let cov = sab - sa * sb / nn;
        let corr = cov / (cova.sqrt() * covb.sqrt());
        if corr > best {
            best = corr;
        }
    }
    (best.clamp(0.0, 1.0), mean_edge)
}

/// Bilinear sample of a `d×d` patch at `(x,y)`; out-of-range clamps to edge.
fn bilinear(p: &[f64], d: usize, x: f64, y: f64) -> f64 {
    let xf = x.clamp(0.0, d as f64 - 1.0);
    let yf = y.clamp(0.0, d as f64 - 1.0);
    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(d - 1);
    let y1 = (y0 + 1).min(d - 1);
    let tx = xf - x0 as f64;
    let ty = yf - y0 as f64;
    let a = p[y0 * d + x0] * (1.0 - tx) + p[y0 * d + x1] * tx;
    let b = p[y1 * d + x0] * (1.0 - tx) + p[y1 * d + x1] * tx;
    a * (1.0 - ty) + b * ty
}

// ===========================================================================
// Peaks (non-local-max suppression on a GX×GY field)
// ===========================================================================

#[derive(Clone)]
struct Peak {
    gx: usize,
    gy: usize,
    val: f64,
}

/// Aspect-corrected frame-space position of a grid cell, in units of the frame
/// **width** (x ∈ [0,1], y ∈ [0, RH/RW]). Used so peak separation is measured in
/// real frame geometry, not anisotropic grid cells.
fn grid_frame_xy(gx: usize, gy: usize) -> (f64, f64) {
    let fx = (gx as f64 + 0.5) / GX as f64;
    let fy = (gy as f64 + 0.5) / GY as f64 * (RH as f64 / RW as f64);
    (fx, fy)
}

/// **Spatial-diversity peak selection** (farthest-point-style; reuses the idea
/// from the old `search` diversity selector). Collect local maxima (3×3
/// neighbourhood, value > 0), sort by value descending, then greedily keep the
/// top peak and each next-highest candidate that lies ≥ `min_dist` (frame-width
/// fraction, aspect-corrected) from **all** already-kept peaks, until `cap`.
///
/// Unlike plain top-N suppression with a tiny radius — which spends the whole
/// cap on the densest boundary ridge — the larger frame-space radius forces
/// spread, so isolated mid-density interior foci survive. **Selection only: the
/// score is untouched** (still F2×F3 with F3 raw above the reject floor).
fn find_peaks(field: &[f64], cap: usize, min_dist: f64) -> Vec<Peak> {
    let mut cands: Vec<Peak> = Vec::new();
    for gy in 0..GY {
        for gx in 0..GX {
            let v = field[gy * GX + gx];
            if v <= 0.0 {
                continue;
            }
            let mut is_max = true;
            'nb: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = gx as i64 + dx;
                    let ny = gy as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= GX as i64 || ny >= GY as i64 {
                        continue;
                    }
                    if field[ny as usize * GX + nx as usize] > v {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                cands.push(Peak { gx, gy, val: v });
            }
        }
    }
    cands.sort_by(|a, b| b.val.partial_cmp(&a.val).unwrap_or(std::cmp::Ordering::Equal));
    let r2 = min_dist * min_dist;
    let mut kept: Vec<Peak> = Vec::new();
    for c in cands {
        let (cx, cy) = grid_frame_xy(c.gx, c.gy);
        let too_close = kept.iter().any(|k| {
            let (kx, ky) = grid_frame_xy(k.gx, k.gy);
            let dx = kx - cx;
            let dy = ky - cy;
            dx * dx + dy * dy < r2
        });
        if too_close {
            continue;
        }
        kept.push(c);
        if kept.len() >= cap {
            break;
        }
    }
    kept
}

// ===========================================================================
// Per-seed processing + visualization
// ===========================================================================

struct ProcResult {
    peaks: Vec<Peak>,
    peaks_json: String,
}

fn process_seed(
    s: &Seed,
    buf: &render::SampleBuffer,
    heat: &Palette,
    seed_pal: &Palette,
) -> Result<ProcResult, String> {
    let nuclei = confirmed_nuclei(s, buf);
    let nucfield = nucleus_field(&nuclei);
    let sw = sweep(s, buf, &nucfield);

    // Combined peaks (≤10) + per-field faint peaks, all via spatial-diversity
    // selection (min separation PEAK_DIV_RADIUS_FRAC of frame width).
    let r = PEAK_DIV_RADIUS_FRAC;
    let comb_peaks = find_peaks(&sw.combined, 10, r);
    let f1_peaks = find_peaks(&sw.f1, 10, r);
    let f2_peaks = find_peaks(&sw.f2, 10, r);
    // Field 3 peaks only where not rejected.
    let f3_masked: Vec<f64> = sw
        .f3raw
        .iter()
        .zip(&sw.reject)
        .map(|(&v, &r)| if r { 0.0 } else { v })
        .collect();
    let f3_peaks = find_peaks(&f3_masked, 10, r);

    // --- seed image under twilight_shifted (display size) ---
    let seed_full = render::shade_and_downsample(
        &buf.samples,
        RW,
        RH,
        1,
        seed_pal,
        &seed_params(),
        s.fw / RW as f64,
    );
    let mut seed_disp = downscale(&seed_full, HM_W, HM_H);

    // --- heatmap tiles ---
    let f2_max = sw.f2.iter().cloned().fold(1e-12, f64::max);
    let f3_max = sw.f3raw.iter().cloned().fold(1e-12, f64::max);
    let comb_max = sw.combined.iter().cloned().fold(1e-12, f64::max);

    let tile_f1 = heatmap_tile(&sw.f1, 0.0, 1.0, heat, "F1 rot-symmetry (corr, max/scale)");
    let tile_f2 = heatmap_tile(&sw.f2, 0.0, f2_max, heat, "F2 nucleus density (norm)");
    let tile_f3 = heatmap_tile(&sw.f3raw, 0.0, f3_max, heat, "F3 escape-n std (RAW, max/scale)");
    let tile_mask = mask_tile(&sw.reject, "F3 reject mask (black=degenerate)");

    // Combined heatmap with peaks burned in.
    let mut comb_hm = field_to_heatmap(&sw.combined, 0.0, comb_max, heat);
    // faint per-field peaks first (so combined peaks draw on top)
    draw_faint_peaks(&mut comb_hm, &f1_peaks, Rgb([255, 80, 80])); // symmetry = red
    draw_faint_peaks(&mut comb_hm, &f2_peaks, Rgb([80, 255, 120])); // nucleus = green
    draw_faint_peaks(&mut comb_hm, &f3_peaks, Rgb([90, 160, 255])); // escape-n = blue
    draw_combined_peaks(&mut comb_hm, &comb_peaks, &sw.win_scale);
    let tile_comb = wrap_titled(comb_hm, "COMBINED (PROVISIONAL prod) + peaks");

    // Seed with the same combined peaks (winning scale → footprint radius).
    draw_combined_peaks_on_image(&mut seed_disp, &comb_peaks, &sw.win_scale);
    let tile_seed = wrap_titled(seed_disp, &format!("{} ({}) seed + peaks [twilight_shifted]", s.name, s.kind));

    // --- compose: row A = [F1][F2][F3][mask]; row B = [combined][seed] ---
    let panel = compose_panel(s, &[&tile_f1, &tile_f2, &tile_f3, &tile_mask], &[&tile_comb, &tile_seed]);
    let dir = format!("{OUT_DIR}/{}", s.name);
    crate::ensure_parent_dir(&format!("{dir}/x"))?;
    panel
        .save(format!("{dir}/panel.png"))
        .map_err(|e| format!("save {dir}/panel.png: {e}"))?;

    // --- per-seed JSON ---
    let mut pj = String::from("[");
    for (i, p) in comb_peaks.iter().enumerate() {
        let gi = p.gy * GX + p.gx;
        let sc = sw.win_scale[gi] as usize;
        let (px, py) = grid_to_px(p.gx, p.gy);
        let _ = write!(
            pj,
            "{}{{\"rank\":{},\"gx\":{},\"gy\":{},\"px\":{},\"py\":{},\"win_scale\":{},\"win_frac\":{:.3},\"combined\":{:.5},\"f1\":{:.5},\"f2_norm\":{:.5},\"f3_std\":{:.5},\"rejected\":{}}}",
            if i > 0 { "," } else { "" },
            i + 1, p.gx, p.gy, px, py, sc, SCALES[sc],
            p.val, sw.f1[gi], sw.f2[gi] / f2_max, sw.f3raw[gi], sw.reject[gi]
        );
    }
    pj.push(']');
    let seed_json = format!(
        "{{\n  \"name\": \"{}\", \"kind\": \"{}\",\n  \"center_re\": {:.15e}, \"center_im\": {:.15e}, \"frame_width\": {:.6e},\n  \"n_confirmed_nuclei\": {},\n  \"scales\": {:?},\n  \"combined_peaks\": {}\n}}\n",
        s.name, s.kind, s.center.re, s.center.im, s.fw, nuclei.len(), SCALES, pj
    );
    std::fs::write(format!("{dir}/focus.json"), &seed_json)
        .map_err(|e| format!("write {dir}/focus.json: {e}"))?;

    let mut peaks_json = String::from("[");
    for (i, p) in comb_peaks.iter().enumerate() {
        let (px, py) = grid_to_px(p.gx, p.gy);
        let _ = write!(peaks_json, "{}[{},{},{:.4}]", if i > 0 { "," } else { "" }, px, py, p.val);
    }
    peaks_json.push(']');

    Ok(ProcResult { peaks: comb_peaks, peaks_json })
}

/// Seed coloring: smooth escape-time exterior, trap-filled interior (so the
/// interior is not dead black for focus-judging), under twilight_shifted.
fn seed_params() -> ColorParams {
    ColorParams {
        density: 0.025,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Trap,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    }
}

fn grid_to_px(gx: usize, gy: usize) -> (usize, usize) {
    let px = ((gx as f64 + 0.5) * RW as f64 / GX as f64) as usize;
    let py = ((gy as f64 + 0.5) * RH as f64 / GY as f64) as usize;
    (px.min(RW as usize - 1), py.min(RH as usize - 1))
}

/// Map a grid cell to display (HM) coordinates.
fn grid_to_disp(gx: usize, gy: usize) -> (i64, i64) {
    let dx = ((gx as f64 + 0.5) / GX as f64 * HM_W as f64) as i64;
    let dy = ((gy as f64 + 0.5) / GY as f64 * HM_H as f64) as i64;
    (dx, dy)
}

// ===========================================================================
// Heatmap / tile rendering
// ===========================================================================

/// Render a `GX×GY` field to an `HM_W×HM_H` heatmap (nearest upscale, `heat`
/// colormap over `[vmin,vmax]`).
fn field_to_heatmap(field: &[f64], vmin: f64, vmax: f64, heat: &Palette) -> RgbImage {
    let mut img = RgbImage::new(HM_W, HM_H);
    let span = (vmax - vmin).max(1e-12);
    for y in 0..HM_H {
        let gy = (y as usize * GY / HM_H as usize).min(GY - 1);
        for x in 0..HM_W {
            let gx = (x as usize * GX / HM_W as usize).min(GX - 1);
            let t = ((field[gy * GX + gx] - vmin) / span).clamp(0.0, 1.0);
            img.put_pixel(x, y, lut_rgb(heat, t));
        }
    }
    img
}

/// A finished heatmap tile: title bar + heatmap + a vertical colorbar with
/// min/max labels.
fn heatmap_tile(field: &[f64], vmin: f64, vmax: f64, heat: &Palette, title: &str) -> RgbImage {
    let hm = field_to_heatmap(field, vmin, vmax, heat);
    titled_with_colorbar(hm, title, heat, vmin, vmax)
}

/// Binary reject-mask tile (black = degenerate/masked, white = kept).
fn mask_tile(reject: &[bool], title: &str) -> RgbImage {
    let mut img = RgbImage::new(HM_W, HM_H);
    for y in 0..HM_H {
        let gy = (y as usize * GY / HM_H as usize).min(GY - 1);
        for x in 0..HM_W {
            let gx = (x as usize * GX / HM_W as usize).min(GX - 1);
            let v = if reject[gy * GX + gx] { 20u8 } else { 230 };
            img.put_pixel(x, y, Rgb([v, v, v]));
        }
    }
    wrap_titled(img, title)
}

/// LUT lookup → sRGB8 pixel (Palette is linear-light; encode for the PNG).
fn lut_rgb(pal: &Palette, t: f64) -> Rgb<u8> {
    let lin = pal.lookup_linear(t.clamp(0.0, 1.0));
    Rgb([
        (linear_to_srgb(lin[0]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[1]) * 255.0 + 0.5) as u8,
        (linear_to_srgb(lin[2]) * 255.0 + 0.5) as u8,
    ])
}

/// Wrap an `HM_W`-wide image with a title bar only (no colorbar).
fn wrap_titled(hm: RgbImage, title: &str) -> RgbImage {
    let tw = HM_W + CBAR_W + CBAR_GUT;
    let th = HM_H + TITLE_H;
    let mut img = RgbImage::from_pixel(tw, th, Rgb([24, 24, 28]));
    probe::blit(&mut img, &hm, 0, TITLE_H);
    font::draw_text(&mut img, &title.to_uppercase(), 2, 3, 1, Rgb([235, 235, 235]), true);
    img
}

/// Wrap a heatmap with a title bar and a vertical colorbar (min at bottom, max
/// at top) labelled with the value range.
fn titled_with_colorbar(hm: RgbImage, title: &str, heat: &Palette, vmin: f64, vmax: f64) -> RgbImage {
    let mut img = wrap_titled(hm, title);
    let bx = HM_W + 6;
    for y in 0..HM_H {
        let t = 1.0 - y as f64 / (HM_H - 1) as f64;
        let px = lut_rgb(heat, t);
        for x in bx..bx + CBAR_W {
            img.put_pixel(x, TITLE_H + y, px);
        }
    }
    let lx = bx + CBAR_W + 2;
    font::draw_text(&mut img, &fmt_v(vmax), lx, TITLE_H + 1, 1, Rgb([230, 230, 230]), true);
    font::draw_text(&mut img, &fmt_v(vmin), lx, TITLE_H + HM_H - 8, 1, Rgb([230, 230, 230]), true);
    img
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

/// Small faint marker (filled 3×3 dot) for a per-field peak.
fn draw_faint_peaks(img: &mut RgbImage, peaks: &[Peak], color: Rgb<u8>) {
    for p in peaks {
        let (cx, cy) = grid_to_disp(p.gx, p.gy);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

/// Prominent combined peaks (display coords): the **winning-scale window
/// footprint** as a thin rectangle (the zoom-region hint) + a center dot + an
/// index label. The rectangle directly shows the 16:9 region the scale picks,
/// far more legibly than a frame-dominating circle.
fn draw_combined_peaks(img: &mut RgbImage, peaks: &[Peak], win_scale: &[u8]) {
    for (i, p) in peaks.iter().enumerate() {
        let gi = p.gy * GX + p.gx;
        let sc = win_scale[gi] as usize;
        let (cx, cy) = grid_to_disp(p.gx, p.gy);
        let rw = (SCALES[sc] * HM_W as f64 * 0.5).max(3.0);
        let rh = rw * 9.0 / 16.0;
        draw_rect(img, cx, cy, rw, rh);
        // center dot
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                put(img, cx + dx, cy + dy, Rgb([255, 255, 120]));
            }
        }
        font::draw_text(
            img,
            &format!("{}", i + 1),
            (cx + 3).max(0) as u32,
            (cy + 3).max(0) as u32,
            1,
            Rgb([255, 255, 120]),
            true,
        );
    }
}

/// Same combined peaks on the seed image (display coords).
fn draw_combined_peaks_on_image(img: &mut RgbImage, peaks: &[Peak], win_scale: &[u8]) {
    draw_combined_peaks(img, peaks, win_scale);
}

/// Put a pixel if in-bounds.
fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

/// Thin white rectangle (with a 1px dark halo for legibility) centred at
/// `(cx,cy)` with half-extents `(rw,rh)`, clipped to the image.
fn draw_rect(img: &mut RgbImage, cx: i64, cy: i64, rw: f64, rh: f64) {
    let x0 = cx - rw as i64;
    let x1 = cx + rw as i64;
    let y0 = cy - rh as i64;
    let y1 = cy + rh as i64;
    let halo = Rgb([0u8, 0, 0]);
    let white = Rgb([255u8, 255, 255]);
    for x in x0..=x1 {
        for (yy, col) in [(y0, halo), (y1, halo)] {
            put(img, x, yy - 1, col);
            put(img, x, yy + 1, col);
        }
        put(img, x, y0, white);
        put(img, x, y1, white);
    }
    for y in y0..=y1 {
        for (xx, col) in [(x0, halo), (x1, halo)] {
            put(img, xx - 1, y, col);
            put(img, xx + 1, y, col);
        }
        put(img, x0, y, white);
        put(img, x1, y, white);
    }
}

/// Box-average downscale of an `RgbImage` to `dw×dh`.
fn downscale(src: &RgbImage, dw: u32, dh: u32) -> RgbImage {
    let (sw, sh) = (src.width(), src.height());
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

/// Compose the per-seed panel: a header strip, row A (4 tiles), row B (2 tiles).
fn compose_panel(s: &Seed, row_a: &[&RgbImage], row_b: &[&RgbImage]) -> RgbImage {
    let tw = HM_W + CBAR_W + CBAR_GUT;
    let th = HM_H + TITLE_H;
    let cols_a = row_a.len() as u32;
    let header_h = 22u32;
    let width = cols_a * tw + (cols_a + 1) * PAD;
    let height = header_h + 2 * th + 3 * PAD;
    let mut img = RgbImage::from_pixel(width, height, Rgb([12, 12, 14]));

    let header = format!(
        "FOCUS HEATMAPS  {}  ({})   center ({:.6}, {:.6})  fw {:.3e}   grid {GX}x{GY}  scales {:?}  [DIAGNOSIS ONLY]",
        s.name, s.kind, s.center.re, s.center.im, s.fw, SCALES
    );
    font::draw_text(&mut img, &header.to_uppercase(), 4, 6, 1, Rgb([255, 230, 150]), true);

    for (i, t) in row_a.iter().enumerate() {
        let x0 = PAD + i as u32 * (tw + PAD);
        probe::blit(&mut img, t, x0, header_h);
    }
    for (i, t) in row_b.iter().enumerate() {
        let x0 = PAD + i as u32 * (tw + PAD);
        probe::blit(&mut img, t, x0, header_h + th + PAD);
    }
    img
}

// ===========================================================================
// log parsing (flat JSON line) — mirrors the detail-clause bench helpers
// ===========================================================================

fn find_line<'a>(text: &'a str, key: &str, val: f64) -> Option<&'a str> {
    text.lines().find(|l| fnum(l, key) == Some(val))
}

fn fnum(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = line[i..].trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}
