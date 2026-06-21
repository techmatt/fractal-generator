//! `wallpaper` — cheap (f64-only) descent → one real wallpaper candidate, with
//! the corpus busyness band as a **noise gate**.
//!
//! This is the first subcommand whose output is meant to be *judged*, not
//! diagnosed. It stays entirely in the f64 regime by construction: a fixed-zoom
//! greedy descent is hard-stopped at the deepest level whose width keeps the
//! wallpaper render comfortably above f64's ~1e-13 pixel-spacing limit (the
//! "cheap floor"). No perturbation ever engages — and we assert that.
//!
//! The experiment is in the *ranking*. Unlike `descend` (which maximizes
//! busyness and dead-ends in noise), each level scores its K×K windows by
//! **corpus-band proximity**: the same normalized busyness the search uses
//! (`stddev(smooth)/maxiter`), scored 1.0 inside the corpus band `[lo,hi]` and
//! falling off on **both** sides — penalizing too-flat (`<lo`) *and* too-noisy
//! (`>hi`). Per level we log the chosen busyness, the band, and the
//! **max-available** busyness (what the old maximize-greedy would have picked);
//! when max-available ≫ `hi` and the chosen is in-band, the band is actively
//! steering away from the noisy window — the test of its upper bound.
//!
//! Outputs: a low-res descent strip (smooth + black interior — the noise shown
//! "as is"), the deepest level iterated **once** at the wallpaper resolution and
//! **reshaded** (never re-iterated) into a coloring×palette matrix (6 PNGs), and
//! a JSON log whose hp center strings let any treatment be re-rendered later.
//! The `corpus` palette is built here from `targets.json`'s color block — the
//! corpus color targets' first real use.

use std::fs;
use std::path::Path;

use astro_float::{BigFloat, RoundingMode};
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{F64Backend, Trap};
use crate::cli::WallpaperArgs;
use crate::coloring::{ColorChannel, ColorParams, InteriorMode};
use crate::font;
use crate::hp;
use crate::palette::{self, Palette};
use crate::probe::{self, SplitMix64, PERTURB_SPACING};
use crate::render::{self, Frame, SampleBuffer};

/// `de_px < BOUNDARY_DE` counts a feature pixel as near-boundary structure (the
/// sanity gate's "this window touches real structure" test). Mirrors `descend`.
const BOUNDARY_DE: f64 = 2.0;

/// DE-shade strength for the `smooth + --de-shade` matrix panel.
const DE_SHADE_STRENGTH: f64 = 1.0;

/// Per-output-pixel aggregate of its `ss×ss` subsamples (descent panels).
#[derive(Clone, Copy)]
struct Feature {
    /// Mean smooth-iteration over escaped subpixels (0 if none escaped).
    smooth: f64,
    /// Mean DE in **panel** pixel units over escaped subpixels (`∞` if none
    /// escaped). The panel-relative structure test (`BOUNDARY_DE`).
    de_px: f64,
    /// Mean DE in **target wallpaper** pixel units (`de / (width/wp_w)`). `de` is
    /// resolution-invariant, so this predicts the speckle the full-res render will
    /// see; the DE-coherence gate keys on it, *not* on the panel-relative `de_px`.
    de_px_target: f64,
    /// Pixel reads as exterior (majority of subpixels escaped).
    escaped: bool,
}

/// The chosen target plus the per-level band experiment diagnostics.
struct Pick {
    col: usize,
    row: usize,
    /// Normalized busyness of the chosen window (the number on the panel label).
    chosen_busyness: f64,
    /// Band membership score of the chosen window, `[0,1]`.
    chosen_score: f64,
    /// Max normalized busyness over all valid windows — what maximize-greedy
    /// would have taken. When `≫ hi` and `chosen` is in-band, the band steered.
    max_busyness: f64,
    n_valid: usize,
    /// Chosen window's DE-coherence: sub-pixel-boundary fraction at the target
    /// spacing, and median target `de_px`. The selection now steers away from
    /// speckle windows (hard reject) and de-weights borderline ones (soft penalty).
    chosen_subpixel_frac: f64,
    chosen_de_px_median: f64,
    /// Windows dropped by the coherence hard reject (speckle), for the log.
    n_coherence_rejected: usize,
}

/// One descent level's logged record.
struct LevelLog {
    level: u32,
    center_re: String,
    center_im: String,
    frame_width: f64,
    magnification: f64,
    maxiter: u32,
    backend: &'static str,
    chosen_busyness: f64,
    chosen_score: f64,
    max_busyness: f64,
    n_windows: usize,
    chosen_subpixel_frac: f64,
    chosen_de_px_median: f64,
    n_coherence_rejected: usize,
    target_re: String,
    target_im: String,
}

/// Entry point for the `wallpaper` subcommand.
pub fn run_wallpaper(args: &WallpaperArgs) -> Result<(), String> {
    if args.zoom <= 1.0 {
        return Err("--zoom must be > 1".into());
    }
    if args.panel_width == 0 || args.wallpaper_width == 0 {
        return Err("--panel-width and --wallpaper-width must be > 0".into());
    }
    if args.window < 1 {
        return Err("--window must be ≥ 1".into());
    }
    if args.margin <= 0.0 {
        return Err("--margin must be > 0".into());
    }

    let panel_w = args.panel_width;
    let panel_h = ((panel_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let wp_w = args.wallpaper_width;
    let wp_h = ((wp_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample.max(1);
    let k = args.window as i32;

    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };

    // ---- the cheap floor (compute it; assert it later) ----
    // f64 stays clean while pixel_spacing = width/wp_w > ~1e-13. Stop descending
    // at the deepest level whose width keeps `margin`× headroom above that.
    let floor = wp_w as f64 * PERTURB_SPACING * args.margin;
    let mut n_levels = 0u32;
    {
        let mut w = args.start_width;
        while w >= floor && n_levels < args.max_levels {
            n_levels += 1;
            w /= args.zoom;
        }
    }
    let n_levels = n_levels.max(1);
    let deepest_level = n_levels - 1;
    let deepest_width = args.start_width / args.zoom.powi(deepest_level as i32);
    let wp_spacing = deepest_width / wp_w as f64;
    eprintln!(
        "cheap floor: wp_w={wp_w} margin={} → floor={floor:.3e}; {n_levels} levels, \
         deepest width={deepest_width:.3e}, wallpaper pixel-spacing={wp_spacing:.3e} \
         (f64 limit {PERTURB_SPACING:.0e})",
        args.margin,
    );

    // ---- corpus targets: busyness band + corpus palette ----
    let targets_text = fs::read_to_string(&args.targets)
        .map_err(|e| format!("failed to read targets '{}': {e}", args.targets))?;
    let (band_lo, band_hi) = parse_busyness_band(&targets_text).ok_or_else(|| {
        format!("could not read structural.busyness.band from '{}'", args.targets)
    })?;
    eprintln!("corpus busyness band = [{band_lo:.4}, {band_hi:.4}] (native units)");
    let corpus_pal = build_corpus_palette(&targets_text)?;
    let cubehelix_pal = palette::cubehelix(false);
    // The descent strip is shaded "as is": smooth + black interior.
    let strip_pal = palette::Palette::ultra_fractal();
    let strip_params = ColorParams {
        density: args.shade.density,
        offset: args.shade.offset,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: args.shade.trap_scale,
        trap_curve: args.shade.trap_curve,
        trap_phase_strength: args.shade.trap_phase_strength,
        de_shade: None,
        mark_glitches: false,
    };

    // ---- master precision for the carried center (resolve below the deepest pixel) ----
    let master_prec = hp::prec_bits(panel_w, deepest_width.max(1e-300)) + 64;
    let rm = RoundingMode::ToEven;
    let (start_re, start_im) = args.resolved_start_center()?;
    let mut center_re = hp::parse_decimal(&start_re, master_prec)?;
    let mut center_im = hp::parse_decimal(&start_im, master_prec)?;
    let mut width = args.start_width;

    let strip_path = Path::new(&args.strip);
    let panels_dir = probe::panels_dir_for(strip_path);
    fs::create_dir_all(&panels_dir)
        .map_err(|e| format!("failed to create {}: {e}", panels_dir.display()))?;

    let mut rng = SplitMix64(args.seed);
    let mut logs: Vec<LevelLog> = Vec::with_capacity(n_levels as usize);
    let mut panels: Vec<RgbImage> = Vec::with_capacity(n_levels as usize);

    // Captured when we render the deepest level (the wallpaper frame).
    let mut deepest: Option<(BigFloat, BigFloat, f64, u32)> = None;

    print_table_header();

    for level in 0..n_levels {
        let mag = args.start_width / width;
        let maxiter = (args.maxiter_base + args.per_decade * mag.log10())
            .round()
            .max(1.0) as u32;
        let prec = hp::prec_bits(panel_w, width);
        let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

        // Force f64 — this is the cheap regime by construction.
        let panel = probe::render_mandel_panel(
            &center_re, &center_im, center_f64, width, panel_w, panel_h, ss, maxiter,
            args.bailout, prec, trap, crate::cli::BackendChoice::F64,
        );
        assert_eq!(
            panel.backend_name, "F64",
            "wallpaper descent must stay f64 (level {level}); floor logic is wrong"
        );
        let buf = panel.buf;
        let spacing = panel.spacing;

        if level == deepest_level {
            deepest = Some((center_re.clone(), center_im.clone(), width, maxiter));
        }

        // Feature map → band-ranked window pick. The coherence gate keys on the
        // *target wallpaper* spacing (width/wp_w), not the 640-wide panel spacing,
        // so a window's speckle is judged at the resolution the wallpaper renders.
        let target_spacing = width / wp_w as f64;
        let feats = build_features(&buf, spacing, target_spacing);
        let pick = score_and_pick(
            &feats, panel_w, panel_h, k, maxiter, args.zoom, band_lo, band_hi,
            args.coherence_theta, &mut rng,
        );

        // Pixel offset of the chosen target (straight from geometry), accumulated
        // in full precision so the path stays exact at the deepest level.
        let fw = width;
        let fh = width * (panel_h as f64 / panel_w as f64);
        let dc_re = ((pick.col as f64 + 0.5) / panel_w as f64 - 0.5) * fw;
        let dc_im = (0.5 - (pick.row as f64 + 0.5) / panel_h as f64) * fh;
        let target_re = center_re.add(&BigFloat::from_f64(dc_re, master_prec), master_prec, rm);
        let target_im = center_im.add(&BigFloat::from_f64(dc_im, master_prec), master_prec, rm);

        // Shade the panel "as is" and mark the next target.
        let mut img = render::shade_and_downsample(
            &buf.samples, panel_w, panel_h, ss, &strip_pal, &strip_params, spacing,
        );
        let circle_r = panel_w as f64 / (2.0 * args.zoom);
        probe::draw_circle(&mut img, pick.col as f64, pick.row as f64, circle_r);
        let label = format!(
            "L{:02} W={:.1E} M={:.1E} B={:.3} F64",
            level, width, mag, pick.chosen_busyness,
        )
        .to_uppercase();
        font::draw_text(&mut img, &label, 2, 2, 2, Rgb([240, 240, 240]), true);
        img.save(panels_dir.join(format!("level_{level:02}.png")))
            .map_err(|e| format!("failed to write level panel: {e}"))?;

        print_table_row(level, width, mag, maxiter, band_lo, band_hi, &pick);

        logs.push(LevelLog {
            level,
            center_re: hp::to_decimal_string(&center_re)?,
            center_im: hp::to_decimal_string(&center_im)?,
            frame_width: width,
            magnification: mag,
            maxiter,
            backend: "F64",
            chosen_busyness: pick.chosen_busyness,
            chosen_score: pick.chosen_score,
            max_busyness: pick.max_busyness,
            n_windows: pick.n_valid,
            chosen_subpixel_frac: pick.chosen_subpixel_frac,
            chosen_de_px_median: pick.chosen_de_px_median,
            n_coherence_rejected: pick.n_coherence_rejected,
            target_re: hp::to_decimal_string(&target_re)?,
            target_im: hp::to_decimal_string(&target_im)?,
        });
        panels.push(img);

        // Descend.
        center_re = target_re;
        center_im = target_im;
        width /= args.zoom;
    }

    // ---- descent strip (single column of mandel panels) ----
    let strip = probe::compose_strip_single(&panels, panel_w, panel_h);
    crate::ensure_parent_dir(strip_path)?;
    strip
        .save(strip_path)
        .map_err(|e| format!("failed to write {}: {e}", strip_path.display()))?;
    eprintln!("wrote {} ({n_levels} levels), panels in {}/", args.strip, panels_dir.display());

    // ---- the wallpaper: iterate the deepest level ONCE, reshade 6 ways ----
    let (dre, dim, dwidth, dmaxiter) = deepest.expect("deepest level was rendered");
    let dcenter = Complex::new(hp::to_f64(&dre), hp::to_f64(&dim));
    let frame = Frame {
        center: dcenter,
        frame_width: dwidth,
        out_width: wp_w,
        out_height: wp_h,
    };
    let spacing = frame.pixel_size();
    // Assert the floor logic: the wallpaper render is f64-clean (no perturbation).
    assert!(
        spacing > PERTURB_SPACING,
        "wallpaper pixel spacing {spacing:.3e} is inside f64's quantization regime \
         (limit {PERTURB_SPACING:.0e}); the floor logic failed"
    );
    eprintln!(
        "iterating wallpaper {wp_w}x{wp_h} (ss{ss}) at deepest level {deepest_level}, \
         width={dwidth:.3e}, maxiter={dmaxiter}, spacing={spacing:.3e}, backend F64 ...",
    );
    let t0 = std::time::Instant::now();
    let backend = F64Backend::new(dmaxiter, args.bailout, trap);
    let buf = render::iterate_samples(&backend, &frame, ss);
    eprintln!(
        "  iterated in {:.1}s ({} glitched pixels — must be 0 for f64)",
        t0.elapsed().as_secs_f64(),
        buf.glitched_pixels,
    );

    // Frame-level DE-coherence gate on the actual wallpaper buffer. Here the
    // render IS at wp_w, so `de_px` against `width/wp_w` is exact (no thumbnail
    // extrapolation). The per-window gate already steered the descent away from
    // speckle, so this should pass; report it either way (and warn if it doesn't —
    // we still emit the images, this command is a forced single descent to judge).
    let wp_stats = crate::coherence::coherence_stats(&buf, dwidth, wp_w, args.coherence_theta);
    let wp_gate = crate::coherence::coherence_gate(&wp_stats);
    eprintln!(
        "  wallpaper coherence: subpixel_frac={:.4} de_px_median={:.4} → {}",
        wp_stats.subpixel_frac,
        wp_stats.de_px_median,
        if wp_gate.reject {
            format!("REJECT ({})", wp_gate.reason.unwrap_or("speckle"))
        } else if wp_gate.penalty < 0.999 {
            format!("pass (soft penalty {:.3})", wp_gate.penalty)
        } else {
            "pass (clean)".to_string()
        },
    );
    if wp_gate.reject {
        eprintln!(
            "  WARNING: the deepest frame is sub-pixel speckle at {wp_w}px — emitting anyway \
             (forced single descent), but it is not a wallpaper candidate."
        );
    }

    // The coloring × palette matrix. Base params from --shade; each cell overrides
    // channel / interior / de_shade. One buffer, 6 sequential reshades.
    let base = probe::color_params(&args.shade);
    let treatments: [(&str, ColorChannel, InteriorMode, Option<f64>); 3] = [
        ("smooth", ColorChannel::Smooth, InteriorMode::Black, None),
        ("trap", ColorChannel::Trap, InteriorMode::Trap, None),
        ("smooth_de", ColorChannel::Smooth, InteriorMode::Black, Some(DE_SHADE_STRENGTH)),
    ];
    let palettes: [(&str, &Palette); 2] = [("corpus", &corpus_pal), ("cubehelix", &cubehelix_pal)];

    let mut wallpaper_paths: Vec<(String, String, String)> = Vec::new();
    for (cname, channel, interior, de_shade) in treatments {
        let params = ColorParams { channel, interior, de_shade, ..base };
        for (pname, pal) in palettes {
            let img = render::shade_and_downsample(
                &buf.samples, wp_w, wp_h, ss, pal, &params, spacing,
            );
            let path = format!("{}_{cname}_{pname}.png", args.out_prefix);
            crate::ensure_parent_dir(&path)?;
            img.save(&path).map_err(|e| format!("failed to write {path}: {e}"))?;
            eprintln!("  wrote {path} ({cname} × {pname})");
            wallpaper_paths.push((cname.to_string(), pname.to_string(), probe::path_str(Path::new(&path))));
        }
    }
    // Free the ~1 GB supersampled buffer now that all reshades are done.
    drop(buf);

    // ---- JSON log ----
    let deepest_info = (
        hp::to_decimal_string(&dre)?,
        hp::to_decimal_string(&dim)?,
        dwidth,
        dmaxiter,
        spacing,
    );
    let wp_coherence = (
        wp_stats.subpixel_frac,
        wp_stats.de_px_median,
        wp_gate.penalty,
        wp_gate.reject,
    );
    let json = build_json(
        &logs, args, band_lo, band_hi, floor, &deepest_info, wp_w, wp_h,
        &wallpaper_paths, &probe::path_str(strip_path), wp_coherence,
    );
    crate::ensure_parent_dir(&args.json)?;
    fs::write(&args.json, json).map_err(|e| format!("failed to write {}: {e}", args.json))?;
    eprintln!("wrote {}", args.json);

    Ok(())
}

/// Aggregate the supersampled buffer into one [`Feature`] per output pixel.
/// `spacing` is the panel pixel spacing (for `de_px`); `target_spacing` is the
/// final wallpaper spacing (`width / wp_w`) the coherence gate keys on.
fn build_features(buf: &SampleBuffer, spacing: f64, target_spacing: f64) -> Vec<Feature> {
    let w = buf.out_width as usize;
    let h = buf.out_height as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;
    let total = (s * s) as f64;
    let inv_target = 1.0 / target_spacing;
    let mut feats = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let mut esc = 0usize;
            let mut sm = 0.0f64;
            let mut de = 0.0f64;
            let mut de_t = 0.0f64;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.escaped {
                        esc += 1;
                        sm += px.smooth_iter;
                        de += px.de / spacing;
                        de_t += px.de * inv_target;
                    }
                }
            }
            let esc_frac = esc as f64 / total;
            let (smooth, de_px, de_px_target) = if esc > 0 {
                (sm / esc as f64, de / esc as f64, de_t / esc as f64)
            } else {
                (0.0, f64::INFINITY, f64::INFINITY)
            };
            feats.push(Feature {
                smooth,
                de_px,
                de_px_target,
                escaped: esc_frac >= 0.5,
            });
        }
    }
    feats
}

/// Hermite smoothstep clamped to `[0,1]`.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Band-membership score for a normalized busyness `b` against `[lo,hi]`:
/// 1.0 inside the band, ramping up over `[0,lo]` (penalize too-flat) and ramping
/// down over `[hi, 2·hi]` (penalize too-noisy — the upper-bound noise gate).
fn band_membership(b: f64, lo: f64, hi: f64) -> f64 {
    if b < lo {
        smoothstep(0.0, lo, b)
    } else if b <= hi {
        1.0
    } else {
        1.0 - smoothstep(hi, 2.0 * hi, b)
    }
}

/// Population standard deviation (n<2 → 0).
fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// Score every eligible K×K window by corpus-band proximity of its normalized
/// busyness (`stddev(smooth)/maxiter`), and sample one target from the top-scored
/// windows. Light sanity gate only: reject pure-interior and pure-fast windows.
/// Also returns the max normalized busyness available (maximize-greedy's pick).
#[allow(clippy::too_many_arguments)]
fn score_and_pick(
    feats: &[Feature],
    panel_w: u32,
    panel_h: u32,
    k: i32,
    maxiter: u32,
    zoom: f64,
    band_lo: f64,
    band_hi: f64,
    theta: f64,
    rng: &mut SplitMix64,
) -> Pick {
    let w = panel_w as i32;
    let h = panel_h as i32;
    let r = k / 2;
    // The next frame's footprint must fit inside the panel.
    let mx = (panel_w as f64 / (2.0 * zoom)).ceil() as i32;
    let my = (panel_h as f64 / (2.0 * zoom)).ceil() as i32;
    let margin_x = r.max(mx);
    let margin_y = r.max(my);

    // The exact normalization the search uses (navigate::atom_candidates): the
    // window smooth-std divided by maxiter, so busyness is O(1) and commensurable
    // with the corpus native band.
    let inv_scale = 1.0 / maxiter.max(1) as f64;

    // Per surviving window: position, band score (already coherence-penalized),
    // busyness, and the window's coherence stats (for the chosen-window log).
    struct Win {
        col: usize,
        row: usize,
        score: f64,
        busyness: f64,
        subpixel_frac: f64,
        de_px_median: f64,
    }
    let mut cands: Vec<Win> = Vec::new();
    let mut n_coherence_rejected = 0usize;
    let mut de_t_buf: Vec<f64> = Vec::with_capacity((k * k) as usize);
    for row in margin_y..(h - margin_y) {
        for col in margin_x..(w - margin_x) {
            let mut vals: Vec<f64> = Vec::with_capacity((k * k) as usize);
            de_t_buf.clear();
            let mut interior_c = 0usize;
            let mut boundary_c = 0usize;
            let mut subpixel_c = 0usize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let f = &feats[(row + dy) as usize * w as usize + (col + dx) as usize];
                    if f.escaped {
                        vals.push(f.smooth);
                        if f.de_px < BOUNDARY_DE {
                            boundary_c += 1;
                        }
                        // Coherence keys on the *target*-spacing de_px.
                        de_t_buf.push(f.de_px_target);
                        if f.de_px_target < theta {
                            subpixel_c += 1;
                        }
                    } else {
                        interior_c += 1;
                    }
                }
            }
            // Sanity gate: pure interior (can't measure busyness) → skip.
            if vals.len() < 3 {
                continue;
            }
            // Sanity gate: pure-fast background (all exterior, no interior, no
            // boundary structure) → skip.
            if interior_c == 0 && boundary_c == 0 {
                continue;
            }

            // DE-coherence over the escaped pixels of this window, at the target
            // wallpaper spacing — the shared gate (reject speckle, penalize
            // borderline). de_t_buf is non-empty here (vals.len() ≥ 3 ⇒ escaped).
            let escaped_n = de_t_buf.len();
            let subpixel_frac = subpixel_c as f64 / escaped_n as f64;
            let de_px_median = median(&mut de_t_buf);
            let gate = crate::coherence::gate_from(subpixel_frac, de_px_median);
            if gate.reject {
                n_coherence_rejected += 1;
                continue; // steer away from sub-pixel speckle windows
            }

            let busyness = stddev(&vals) * inv_scale;
            let score = band_membership(busyness, band_lo, band_hi) * gate.penalty;
            cands.push(Win { col: col as usize, row: row as usize, score, busyness, subpixel_frac, de_px_median });
        }
    }

    if cands.is_empty() {
        return Pick {
            col: panel_w as usize / 2,
            row: panel_h as usize / 2,
            chosen_busyness: 0.0,
            chosen_score: 0.0,
            max_busyness: 0.0,
            n_valid: 0,
            chosen_subpixel_frac: f64::NAN,
            chosen_de_px_median: f64::NAN,
            n_coherence_rejected,
        };
    }

    let n_valid = cands.len();
    let max_busyness = cands.iter().fold(0.0f64, |m, c| m.max(c.busyness));

    // Rank by band score; sample one from the top 1% (seeded) for reproducible
    // diversity among the in-band windows.
    let mut order: Vec<usize> = (0..n_valid).collect();
    order.sort_by(|&a, &b| cands[b].score.partial_cmp(&cands[a].score).unwrap());
    let topk = (n_valid / 100).max(1);
    let chosen = order[rng.below(topk)];
    let c = &cands[chosen];

    Pick {
        col: c.col,
        row: c.row,
        chosen_busyness: c.busyness,
        chosen_score: c.score,
        max_busyness,
        n_valid,
        chosen_subpixel_frac: c.subpixel_frac,
        chosen_de_px_median: c.de_px_median,
        n_coherence_rejected,
    }
}

/// Median of a slice (mutates: sorts in place). Empty → NaN.
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// ---------------------------------------------------------------------------
// corpus targets parsing (minimal, hand-rolled — the project keeps deps minimal)
// ---------------------------------------------------------------------------

/// Parse the first `[lo, hi]` numeric array following `key` in `text`.
fn parse_bracket_pair_after(text: &str, key: &str) -> Option<(f64, f64)> {
    let p = text.find(key)? + key.len();
    let rest = &text[p..];
    let lb = rest.find('[')?;
    let rb = rest[lb..].find(']')? + lb;
    let parts: Vec<f64> = rest[lb + 1..rb]
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    if parts.len() >= 2 {
        Some((parts[0], parts[1]))
    } else {
        None
    }
}

/// Read `structural.busyness.band` = `[lo, hi]` from a `targets.json` string.
fn parse_busyness_band(text: &str) -> Option<(f64, f64)> {
    // "busyness" appears once; its "band" array is the next bracket pair.
    let p = text.find("\"busyness\"")?;
    parse_bracket_pair_after(&text[p..], "\"band\"")
}

/// Build the `corpus` palette from `targets.json`'s color block: take the
/// dominant OKLab cluster centers, sort by luminance (OKLab L), and lay them out
/// as a cyclic OKLab gradient. The corpus color targets' first real use.
fn build_corpus_palette(text: &str) -> Result<Palette, String> {
    // Each palette entry is `{ "oklab": [L,a,b], "weight": w }`; "oklab"/"weight"
    // appear only in the color.palette array of targets.json, so scan globally.
    let mut entries: Vec<([f64; 3], f64)> = Vec::new();
    let mut rest = text;
    while let Some(p) = rest.find("\"oklab\"") {
        rest = &rest[p + "\"oklab\"".len()..];
        let Some(lb) = rest.find('[') else { break };
        let Some(rb) = rest[lb..].find(']').map(|x| x + lb) else { break };
        let lab: Vec<f64> = rest[lb + 1..rb]
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if lab.len() < 3 {
            continue;
        }
        let weight = {
            let after = &rest[rb..];
            after
                .find("\"weight\"")
                .and_then(|wp| {
                    let s = after[wp + "\"weight\"".len()..]
                        .trim_start_matches([':', ' ', '\t', '\n']);
                    let end = s
                        .find(|c: char| {
                            !(c.is_ascii_digit()
                                || c == '.'
                                || c == '-'
                                || c == '+'
                                || c == 'e'
                                || c == 'E')
                        })
                        .unwrap_or(s.len());
                    s[..end].parse::<f64>().ok()
                })
                .unwrap_or(0.0)
        };
        entries.push(([lab[0], lab[1], lab[2]], weight));
        rest = &rest[rb..];
    }

    if entries.len() < 2 {
        return Err(format!(
            "corpus palette needs ≥2 OKLab clusters in targets.json (found {})",
            entries.len()
        ));
    }
    // Dominant clusters first (by weight), then order the gradient by luminance.
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap_or(std::cmp::Ordering::Equal));
    let colors: Vec<[f64; 3]> = entries.iter().map(|e| e.0).collect();
    Ok(Palette::from_oklab_colors("corpus", &colors, false))
}

// ---------------------------------------------------------------------------
// stdout table
// ---------------------------------------------------------------------------

fn print_table_header() {
    println!(
        "{:>3}  {:>10}  {:>9}  {:>6}  {:>4}  {:>8}  {:>8}  {:>8}  {:>9}  {:>6}  {:>6}  {:>5}",
        "lvl", "width", "mag", "maxit", "bknd", "chosen_b", "band_lo", "band_hi", "max_avail",
        "wins", "spx", "crej",
    );
}

fn print_table_row(
    level: u32,
    width: f64,
    mag: f64,
    maxiter: u32,
    band_lo: f64,
    band_hi: f64,
    pick: &Pick,
) {
    let steered = if pick.max_busyness > band_hi && pick.chosen_busyness <= band_hi {
        " <-steered"
    } else {
        ""
    };
    println!(
        "{:>3}  {:>10.3e}  {:>9.2e}  {:>6}  {:>4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>9.4}  {:>6}  {:>6.3}  {:>5}{}",
        level, width, mag, maxiter, "F64", pick.chosen_busyness, band_lo, band_hi,
        pick.max_busyness, pick.n_valid, pick.chosen_subpixel_frac, pick.n_coherence_rejected,
        steered,
    );
}

// ---------------------------------------------------------------------------
// JSON log (hand-rolled)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn build_json(
    logs: &[LevelLog],
    args: &WallpaperArgs,
    band_lo: f64,
    band_hi: f64,
    floor: f64,
    deepest: &(String, String, f64, u32, f64),
    wp_w: u32,
    wp_h: u32,
    wallpapers: &[(String, String, String)],
    strip: &str,
    wp_coherence: (f64, f64, f64, bool),
) -> String {
    use probe::{jf, js};
    let mut s = String::from("{\n");

    s.push_str("  \"params\": {\n");
    s.push_str(&format!("    \"start_center\": {},\n", js(&args.start_center)));
    s.push_str(&format!("    \"start_width\": {},\n", jf(args.start_width)));
    s.push_str(&format!("    \"zoom\": {},\n", jf(args.zoom)));
    s.push_str(&format!("    \"wallpaper_width\": {wp_w},\n"));
    s.push_str(&format!("    \"wallpaper_height\": {wp_h},\n"));
    s.push_str(&format!("    \"supersample\": {},\n", args.supersample));
    s.push_str(&format!("    \"margin\": {},\n", jf(args.margin)));
    s.push_str(&format!("    \"f64_floor_width\": {},\n", jf(floor)));
    s.push_str(&format!(
        "    \"busyness_band\": [{}, {}],\n",
        jf(band_lo),
        jf(band_hi)
    ));
    s.push_str(&format!("    \"coherence_theta\": {},\n", jf(args.coherence_theta)));
    s.push_str(&format!("    \"coherence_reject\": {},\n", jf(crate::coherence::COHERENCE_REJECT)));
    s.push_str(&format!("    \"de_px_median_floor\": {},\n", jf(crate::coherence::DE_PX_MEDIAN_FLOOR)));
    s.push_str(&format!("    \"seed\": {}\n", args.seed));
    s.push_str("  },\n");

    // Deepest level = the wallpaper frame.
    s.push_str("  \"wallpaper\": {\n");
    s.push_str(&format!(
        "    \"center\": {{ \"re\": {}, \"im\": {} }},\n",
        js(&deepest.0),
        js(&deepest.1)
    ));
    s.push_str(&format!("    \"width\": {},\n", jf(deepest.2)));
    s.push_str(&format!("    \"maxiter\": {},\n", deepest.3));
    s.push_str(&format!("    \"pixel_spacing\": {},\n", jf(deepest.4)));
    s.push_str("    \"backend\": \"F64\",\n");
    s.push_str("    \"coherence\": {\n");
    s.push_str(&format!("      \"subpixel_frac\": {},\n", jf(wp_coherence.0)));
    s.push_str(&format!("      \"de_px_median\": {},\n", jf(wp_coherence.1)));
    s.push_str(&format!("      \"penalty\": {},\n", jf(wp_coherence.2)));
    s.push_str(&format!("      \"reject\": {}\n", wp_coherence.3));
    s.push_str("    },\n");
    s.push_str("    \"images\": [\n");
    for (i, (coloring, pal, path)) in wallpapers.iter().enumerate() {
        s.push_str(&format!(
            "      {{ \"coloring\": {}, \"palette\": {}, \"path\": {} }}{}\n",
            js(coloring),
            js(pal),
            js(path),
            if i + 1 < wallpapers.len() { "," } else { "" }
        ));
    }
    s.push_str("    ]\n");
    s.push_str("  },\n");

    s.push_str(&format!("  \"strip\": {},\n", js(strip)));

    s.push_str("  \"levels\": [\n");
    for (i, lv) in logs.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"level\": {},\n", lv.level));
        s.push_str(&format!(
            "      \"center\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&lv.center_re),
            js(&lv.center_im)
        ));
        s.push_str(&format!("      \"frame_width\": {},\n", jf(lv.frame_width)));
        s.push_str(&format!("      \"magnification\": {},\n", jf(lv.magnification)));
        s.push_str(&format!("      \"maxiter\": {},\n", lv.maxiter));
        s.push_str(&format!("      \"backend\": {},\n", js(lv.backend)));
        s.push_str(&format!("      \"chosen_busyness\": {},\n", jf(lv.chosen_busyness)));
        s.push_str(&format!("      \"chosen_score\": {},\n", jf(lv.chosen_score)));
        s.push_str(&format!("      \"max_available_busyness\": {},\n", jf(lv.max_busyness)));
        s.push_str(&format!(
            "      \"band\": [{}, {}],\n",
            jf(band_lo),
            jf(band_hi)
        ));
        s.push_str(&format!("      \"n_windows\": {},\n", lv.n_windows));
        s.push_str("      \"coherence\": {\n");
        s.push_str(&format!("        \"chosen_subpixel_frac\": {},\n", jf(lv.chosen_subpixel_frac)));
        s.push_str(&format!("        \"chosen_de_px_median\": {},\n", jf(lv.chosen_de_px_median)));
        s.push_str(&format!("        \"n_windows_rejected\": {}\n", lv.n_coherence_rejected));
        s.push_str("      },\n");
        s.push_str(&format!(
            "      \"chosen_target\": {{ \"re\": {}, \"im\": {} }}\n",
            js(&lv.target_re),
            js(&lv.target_im)
        ));
        s.push_str("    }");
        if i + 1 < logs.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}
