//! `descend` — greedy Mandelbrot→Julia descent filmstrip (depth-falloff probe).
//!
//! Starting from a shallow frame, we repeatedly: iterate the Mandelbrot panel,
//! aggregate subpixels into a feature map, score every K×K window for "interest"
//! (busyness × boundary structure × a target-band on interior fraction), sample
//! one target from the top-1% scored windows with a seeded RNG, render the Julia
//! for that target, compose a `Mandelbrot | Julia` row, then **descend into the
//! sampled target** — accumulating the center in full precision (BigFloat) so the
//! path stays exact well past the f64 floor — and zoom in by `--zoom`.
//!
//! This is the deliberately-naive greedy baseline: it dead-ends into
//! self-similar tedium or dives into a minibrot interior, and *that collapse is
//! the signal* — the row labels and JSON log surface interior fraction, the
//! score range, the `low_signal` flag, and the f64→perturbation handoff so the
//! falloff is legible. The real navigation (Newton / atom domains) must beat it.

use std::fs;
use std::path::{Path, PathBuf};

use astro_float::{BigFloat, RoundingMode};
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{F64Backend, FractalBackend, JuliaBackend, PerturbationBackend, Trap};
use crate::cli::{BackendChoice, DescendArgs};
use crate::coloring::ColorParams;
use crate::font;
use crate::hp;
use crate::palette_io::load_palette;
use crate::render::{self, Frame, SampleBuffer};

/// Pixel spacing at/below which f64 enters its quantization regime — the auto
/// switch to perturbation (mirrors `main`'s constant; the back third of a deep
/// descent crosses it).
const PERTURB_SPACING: f64 = 1e-13;

/// Base-scale Julia view width (whole set, center 0). f64 is always accurate
/// here, so Julia panels never need perturbation.
const JULIA_WIDTH: f64 = 3.5;

/// `de_px < BOUNDARY_DE` counts a feature pixel as near-boundary structure.
const BOUNDARY_DE: f64 = 2.0;

/// Horizontal gap (px) between the Mandelbrot and Julia panels in a row.
const GAP_H: u32 = 4;
/// Vertical gap (px) between rows of the filmstrip.
const GAP_V: u32 = 3;
/// Filmstrip background (near-black).
const STRIP_BG: [u8; 3] = [16, 16, 16];

/// SplitMix64 — a tiny, dependency-free seeded PRNG. Deterministic for a fixed
/// `--seed`, which is what makes a descent reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform index in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Per-output-pixel aggregate of its `ss×ss` subsamples.
#[derive(Clone, Copy)]
struct Feature {
    /// Mean smooth-iteration over escaped subpixels (0 if none escaped).
    smooth: f64,
    /// Mean DE in pixel units over escaped subpixels (`∞` if none escaped).
    de_px: f64,
    /// Pixel reads as exterior (majority of subpixels escaped).
    escaped: bool,
}

/// The chosen target and per-level scoring diagnostics.
struct Pick {
    col: usize,
    row: usize,
    score_min: f64,
    score_mean: f64,
    score_max: f64,
    chosen_score: f64,
    /// Percentile of the chosen window's score among valid windows, `[0,1]`.
    chosen_pct: f64,
    interior_fraction: f64,
    /// The level's structural reward collapsed (top scores ≈ 0) — the falloff.
    low_signal: bool,
    n_valid: usize,
}

/// One level's logged record (drives both the JSON and the stdout table).
struct LevelLog {
    level: u32,
    center_re: String,
    center_im: String,
    frame_width: f64,
    magnification: f64,
    maxiter: u32,
    backend: &'static str,
    glitch_count: u64,
    target_re: String,
    target_im: String,
    c_f64: Complex<f64>,
    pick: Pick,
    mandel_panel: String,
    julia_panel: String,
}

/// Entry point for the `descend` subcommand.
pub fn run_descend(args: &DescendArgs) -> Result<(), String> {
    if args.levels == 0 {
        return Err("--levels must be > 0".into());
    }
    if args.zoom <= 1.0 {
        return Err("--zoom must be > 1".into());
    }
    if args.panel_width == 0 {
        return Err("--panel-width must be > 0".into());
    }
    if args.window < 1 {
        return Err("--window must be ≥ 1".into());
    }

    let panel_w = args.panel_width;
    // 16:9 panel; height derived to keep pixels square at base scale.
    let panel_h = ((panel_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample.max(1);
    let k = args.window as i32;

    let palette = load_palette(
        &args.palette.palette,
        args.palette.palette_entry.as_deref(),
        args.palette.palette_reverse,
    )?;
    let params = color_params(args);
    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };

    let (start_re, start_im) = args.resolved_start_center()?;

    // Master precision must resolve the *deepest* center below a pixel; carrying
    // every level's center at this precision is what keeps the descent exact
    // past the f64 floor (a plain-f64 center would be garbage by the back third).
    let deepest_width = (args.start_width / args.zoom.powi(args.levels as i32)).max(1e-300);
    let master_prec = hp::prec_bits(panel_w, deepest_width) + 64;
    let rm = RoundingMode::ToEven;

    let mut center_re = hp::parse_decimal(&start_re, master_prec)?;
    let mut center_im = hp::parse_decimal(&start_im, master_prec)?;
    let mut width = args.start_width;

    // Per-level output panels live alongside the strip in `<stem>_panels/`.
    let strip_path = Path::new(&args.output);
    let panels_dir = panels_dir_for(strip_path);
    fs::create_dir_all(&panels_dir)
        .map_err(|e| format!("failed to create {}: {e}", panels_dir.display()))?;

    let mut rng = SplitMix64(args.seed);
    let mut logs: Vec<LevelLog> = Vec::with_capacity(args.levels as usize);
    let mut mandel_panels: Vec<RgbImage> = Vec::with_capacity(args.levels as usize);
    let mut julia_panels: Vec<RgbImage> = Vec::with_capacity(args.levels as usize);

    print_table_header();

    for level in 0..args.levels {
        let mag = args.start_width / width;
        let maxiter = (args.maxiter_base + args.per_decade * mag.log10())
            .round()
            .max(1.0) as u32;
        let prec = hp::prec_bits(panel_w, width);

        let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));
        let frame = Frame {
            center: center_f64,
            frame_width: width,
            out_width: panel_w,
            out_height: panel_h,
        };
        let spacing = frame.pixel_size();
        let use_perturb = match args.backend {
            BackendChoice::Auto => spacing <= PERTURB_SPACING,
            BackendChoice::Perturb => true,
            BackendChoice::F64 => false,
        };

        // Mandelbrot panel (the only expensive iteration; perturbation engages
        // automatically on the back third).
        let (backend, backend_name): (Box<dyn FractalBackend>, &'static str) = if use_perturb {
            let pb = PerturbationBackend::new(
                &center_re,
                &center_im,
                maxiter,
                args.bailout,
                prec,
                trap,
            );
            (Box::new(pb), "PERT")
        } else {
            (Box::new(F64Backend::new(maxiter, args.bailout, trap)), "F64")
        };
        let buf = render::iterate_samples(&*backend, &frame, ss);

        // Feature map → score every window → sample a target from the top 1%.
        let feats = build_features(&buf, spacing);
        let pick = score_and_pick(
            &feats, panel_w, panel_h, k, args.zoom, &mut rng,
        );

        // Pixel offset of the chosen target from the current center (straight
        // from geometry — never c − center), accumulated in full precision.
        let fw = width;
        let fh = width * (panel_h as f64 / panel_w as f64);
        let dc_re = ((pick.col as f64 + 0.5) / panel_w as f64 - 0.5) * fw;
        let dc_im = (0.5 - (pick.row as f64 + 0.5) / panel_h as f64) * fh;
        let target_re = center_re.add(&BigFloat::from_f64(dc_re, master_prec), master_prec, rm);
        let target_im = center_im.add(&BigFloat::from_f64(dc_im, master_prec), master_prec, rm);
        let c_f64 = Complex::new(hp::to_f64(&target_re), hp::to_f64(&target_im));

        // Shade the Mandelbrot panel and annotate the chosen target's footprint.
        let mut mandel_img =
            render::shade_and_downsample(&buf.samples, panel_w, panel_h, ss, &palette, &params, spacing);
        let circle_r = panel_w as f64 / (2.0 * args.zoom);
        draw_circle(&mut mandel_img, pick.col, pick.row, circle_r);

        // Julia panel for the chosen target, base scale (whole set), f64.
        let julia_backend = JuliaBackend::new(c_f64, args.julia_maxiter, args.bailout, trap);
        let julia_frame = Frame {
            center: Complex::new(0.0, 0.0),
            frame_width: JULIA_WIDTH,
            out_width: panel_w,
            out_height: panel_h,
        };
        let julia_buf = render::iterate_samples(&julia_backend, &julia_frame, ss);
        let julia_img = render::shade_and_downsample(
            &julia_buf.samples,
            panel_w,
            panel_h,
            ss,
            &palette,
            &params,
            julia_frame.pixel_size(),
        );

        // On-image label (uppercased into the reduced glyph set).
        let label = format!(
            "L{:02} M={:.1e} IT={} {} INT={} SIG={}",
            level,
            mag,
            maxiter,
            backend_name,
            (pick.interior_fraction * 100.0).round() as i64,
            if pick.low_signal { "LOW" } else { "OK" },
        )
        .to_uppercase();
        font::draw_text(&mut mandel_img, &label, 2, 2, 2, Rgb([240, 240, 240]), true);

        // Persist per-level panels (so any level re-renders from the JSON).
        let mandel_rel = format!("mandel_{level:02}.png");
        let julia_rel = format!("julia_{level:02}.png");
        let mandel_path = panels_dir.join(&mandel_rel);
        let julia_path = panels_dir.join(&julia_rel);
        mandel_img
            .save(&mandel_path)
            .map_err(|e| format!("failed to write {}: {e}", mandel_path.display()))?;
        julia_img
            .save(&julia_path)
            .map_err(|e| format!("failed to write {}: {e}", julia_path.display()))?;

        let center_re_str = hp::to_decimal_string(&center_re)?;
        let center_im_str = hp::to_decimal_string(&center_im)?;
        let target_re_str = hp::to_decimal_string(&target_re)?;
        let target_im_str = hp::to_decimal_string(&target_im)?;

        print_table_row(level, width, mag, maxiter, backend_name, buf.glitched_pixels, &pick);

        logs.push(LevelLog {
            level,
            center_re: center_re_str,
            center_im: center_im_str,
            frame_width: width,
            magnification: mag,
            maxiter,
            backend: backend_name,
            glitch_count: buf.glitched_pixels,
            target_re: target_re_str,
            target_im: target_im_str,
            c_f64,
            pick,
            mandel_panel: path_str(&mandel_path),
            julia_panel: path_str(&julia_path),
        });
        mandel_panels.push(mandel_img);
        julia_panels.push(julia_img);

        // Descend: the sampled target becomes the next center; zoom in.
        center_re = target_re;
        center_im = target_im;
        width /= args.zoom;
    }

    // Compose the filmstrip (one row per level: Mandelbrot | Julia).
    let strip = compose_strip(&mandel_panels, &julia_panels, panel_w, panel_h);
    strip
        .save(strip_path)
        .map_err(|e| format!("failed to write {}: {e}", strip_path.display()))?;

    // JSON log.
    let json = build_json(&logs, args.zoom, &path_str(strip_path));
    fs::write(&args.json, json).map_err(|e| format!("failed to write {}: {e}", args.json))?;

    eprintln!(
        "wrote {} ({} levels), per-level panels in {}/, log {}",
        args.output,
        args.levels,
        panels_dir.display(),
        args.json,
    );
    Ok(())
}

/// Map the descend shading args to coloring parameters.
fn color_params(args: &DescendArgs) -> ColorParams {
    ColorParams {
        density: args.shade.density,
        offset: args.shade.offset,
        channel: args.shade.color,
        interior: args.shade.interior,
        trap_scale: args.shade.trap_scale,
        trap_curve: args.shade.trap_curve,
        trap_phase_strength: args.shade.trap_phase_strength,
        de_shade: args.shade.de_shade,
        mark_glitches: args.shade.mark_glitches,
    }
}

/// Aggregate the supersampled buffer into one [`Feature`] per output pixel.
fn build_features(buf: &SampleBuffer, spacing: f64) -> Vec<Feature> {
    let w = buf.out_width as usize;
    let h = buf.out_height as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;
    let total = (s * s) as f64;
    let mut feats = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let mut esc = 0usize;
            let mut sm = 0.0f64;
            let mut de = 0.0f64;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.escaped {
                        esc += 1;
                        sm += px.smooth_iter;
                        de += px.de / spacing;
                    }
                }
            }
            let esc_frac = esc as f64 / total;
            let (smooth, de_px) = if esc > 0 {
                (sm / esc as f64, de / esc as f64)
            } else {
                (0.0, f64::INFINITY)
            };
            feats.push(Feature {
                smooth,
                de_px,
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

/// Target-band on interior fraction `f`: a smooth bump rewarding `f ∈ [0.05,
/// 0.40]`, penalizing `f→0` (bland fast-escape) and `f→1` (dead black). Not
/// interior-minimization — a frame with *some* interior is where the structure
/// lives.
fn band(f: f64) -> f64 {
    smoothstep(0.0, 0.05, f) * (1.0 - smoothstep(0.40, 0.70, f))
}

/// Population standard deviation.
fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// Score every eligible K×K window and sample one target from the top 1%.
///
/// `score = busyness · (0.5 + boundary_frac) · band(interior_fraction)`, where
/// busyness is the std-dev of smooth-iter over escaped pixels (rejected if too
/// few escaped). Windows too near the border to host the next frame, and
/// pure-interior windows, are excluded. If the top scores collapse to ≈0 we
/// still pick the global best (ranked by busyness) and flag `low_signal`.
fn score_and_pick(
    feats: &[Feature],
    panel_w: u32,
    panel_h: u32,
    k: i32,
    zoom: f64,
    rng: &mut SplitMix64,
) -> Pick {
    let w = panel_w as i32;
    let h = panel_h as i32;
    let r = k / 2;
    // The next frame's footprint must fit inside the panel, so keep the target
    // at least its half-extent from each edge (and the window in-bounds).
    let mx = (panel_w as f64 / (2.0 * zoom)).ceil() as i32;
    let my = (panel_h as f64 / (2.0 * zoom)).ceil() as i32;
    let margin_x = r.max(mx);
    let margin_y = r.max(my);

    let interior = feats.iter().filter(|f| !f.escaped).count();
    let interior_fraction = interior as f64 / feats.len() as f64;

    // (col, row, score, busyness) for each valid window.
    let mut cands: Vec<(usize, usize, f64, f64)> = Vec::new();
    for row in margin_y..(h - margin_y) {
        for col in margin_x..(w - margin_x) {
            let mut vals: Vec<f64> = Vec::with_capacity((k * k) as usize);
            let mut interior_c = 0usize;
            let mut boundary_c = 0usize;
            let mut total = 0usize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let f = &feats[(row + dy) as usize * w as usize + (col + dx) as usize];
                    total += 1;
                    if f.escaped {
                        vals.push(f.smooth);
                        if f.de_px < BOUNDARY_DE {
                            boundary_c += 1;
                        }
                    } else {
                        interior_c += 1;
                    }
                }
            }
            if vals.is_empty() {
                continue; // pure interior — excluded
            }
            let frac_int = interior_c as f64 / total as f64;
            let busyness = if vals.len() >= 3 { stddev(&vals) } else { 0.0 };
            let boundary_frac = boundary_c as f64 / total as f64;
            let score = busyness * (0.5 + boundary_frac) * band(frac_int);
            cands.push((col as usize, row as usize, score, busyness));
        }
    }

    if cands.is_empty() {
        // Whole panel excluded (e.g. fully interior minibrot) — descend into the
        // center and flag the collapse.
        return Pick {
            col: panel_w as usize / 2,
            row: panel_h as usize / 2,
            score_min: 0.0,
            score_mean: 0.0,
            score_max: 0.0,
            chosen_score: 0.0,
            chosen_pct: 0.0,
            interior_fraction,
            low_signal: true,
            n_valid: 0,
        };
    }

    let n_valid = cands.len();
    let score_max = cands.iter().fold(f64::NEG_INFINITY, |m, c| m.max(c.2));
    let score_min = cands.iter().fold(f64::INFINITY, |m, c| m.min(c.2));
    let score_mean = cands.iter().map(|c| c.2).sum::<f64>() / n_valid as f64;
    // Collapse: the structural reward vanished everywhere — the falloff signal.
    let low_signal = score_max <= 1e-9;

    // Rank by score; if the scores collapsed, fall back to busyness so we still
    // descend toward whatever residual structure exists.
    let mut order: Vec<usize> = (0..n_valid).collect();
    if low_signal {
        order.sort_by(|&a, &b| cands[b].3.partial_cmp(&cands[a].3).unwrap());
    } else {
        order.sort_by(|&a, &b| cands[b].2.partial_cmp(&cands[a].2).unwrap());
    }

    let topk = (n_valid / 100).max(1);
    let chosen = order[rng.below(topk)];
    let (col, row, chosen_score, _) = cands[chosen];

    // Percentile of the chosen score among all valid windows.
    let le = cands.iter().filter(|c| c.2 <= chosen_score).count();
    let chosen_pct = le as f64 / n_valid as f64;

    Pick {
        col,
        row,
        score_min,
        score_mean,
        score_max,
        chosen_score,
        chosen_pct,
        interior_fraction,
        low_signal,
        n_valid,
    }
}

/// Draw a white circle (radius `r` px) at `(cx, cy)` with a 1px dark halo on
/// each side for legibility over light regions. `r` marks the next frame's
/// footprint.
fn draw_circle(img: &mut RgbImage, cx: usize, cy: usize, r: f64) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    let cxf = cx as f64;
    let cyf = cy as f64;
    let x0 = ((cxf - r - 2.0).floor() as i64).max(0);
    let x1 = ((cxf + r + 2.0).ceil() as i64).min(w - 1);
    let y0 = ((cyf - r - 2.0).floor() as i64).max(0);
    let y1 = ((cyf + r + 2.0).ceil() as i64).min(h - 1);
    let dark = Rgb([0u8, 0, 0]);
    let white = Rgb([255u8, 255, 255]);
    // Halo first (wider), then the white ring inside it.
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cxf).powi(2)) + ((y as f64 - cyf).powi(2))).sqrt();
            if (d - r).abs() <= 2.0 {
                img.put_pixel(x as u32, y as u32, dark);
            }
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cxf).powi(2)) + ((y as f64 - cyf).powi(2))).sqrt();
            if (d - r).abs() <= 1.0 {
                img.put_pixel(x as u32, y as u32, white);
            }
        }
    }
}

/// Compose the tall filmstrip: one row per level, `Mandelbrot | Julia`.
fn compose_strip(
    mandel: &[RgbImage],
    julia: &[RgbImage],
    panel_w: u32,
    panel_h: u32,
) -> RgbImage {
    let n = mandel.len() as u32;
    let width = 2 * panel_w + GAP_H;
    let height = n * panel_h + n.saturating_sub(1) * GAP_V;
    let mut strip = RgbImage::from_pixel(width, height, Rgb(STRIP_BG));
    for i in 0..mandel.len() {
        let y0 = i as u32 * (panel_h + GAP_V);
        blit(&mut strip, &mandel[i], 0, y0);
        blit(&mut strip, &julia[i], panel_w + GAP_H, y0);
    }
    strip
}

/// Paste `src` into `dst` at `(x0, y0)`.
fn blit(dst: &mut RgbImage, src: &RgbImage, x0: u32, y0: u32) {
    for (sx, sy, px) in src.enumerate_pixels() {
        let (dx, dy) = (x0 + sx, y0 + sy);
        if dx < dst.width() && dy < dst.height() {
            dst.put_pixel(dx, dy, *px);
        }
    }
}

/// `<stem>_panels/` directory beside the strip output.
fn panels_dir_for(strip: &Path) -> PathBuf {
    let stem = strip
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("descend");
    let dir = format!("{stem}_panels");
    match strip.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(dir),
        _ => PathBuf::from(dir),
    }
}

/// Forward-slash path string for the JSON (portable, copy-pasteable).
fn path_str(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

// ---------------------------------------------------------------------------
// stdout table
// ---------------------------------------------------------------------------

fn print_table_header() {
    println!(
        "{:>3}  {:>10}  {:>9}  {:>6}  {:>4}  {:>6}  {:>5}  {:>9}  {:>9}  {:>9}  {:>5}  {:>3}",
        "lvl", "width", "mag", "maxit", "bknd", "int%", "gltch", "score_min", "score_avg",
        "score_max", "pct", "sig",
    );
}

fn print_table_row(
    level: u32,
    width: f64,
    mag: f64,
    maxiter: u32,
    backend: &str,
    glitch: u64,
    pick: &Pick,
) {
    println!(
        "{:>3}  {:>10.3e}  {:>9.2e}  {:>6}  {:>4}  {:>5.1}  {:>5}  {:>9.3}  {:>9.3}  {:>9.3}  {:>5.1}  {:>3}",
        level,
        width,
        mag,
        maxiter,
        backend,
        pick.interior_fraction * 100.0,
        glitch,
        pick.score_min,
        pick.score_mean,
        pick.score_max,
        pick.chosen_pct * 100.0,
        if pick.low_signal { "LOW" } else { "ok" },
    );
}

// ---------------------------------------------------------------------------
// JSON log (hand-rolled — the project keeps deps minimal)
// ---------------------------------------------------------------------------

/// Format a finite f64 in scientific form for JSON; non-finite → `null`.
fn jf(x: f64) -> String {
    if x.is_finite() {
        format!("{x:e}")
    } else {
        "null".into()
    }
}

/// JSON-escape a decimal string (only `"`/`\` are possible; defensive).
fn js(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn build_json(logs: &[LevelLog], zoom: f64, strip: &str) -> String {
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
        s.push_str(&format!("    \"zoom\": {},\n", jf(zoom)));
        s.push_str(&format!(
            "    \"chosen_target\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&lv.target_re),
            js(&lv.target_im)
        ));
        s.push_str(&format!(
            "    \"c_f64\": {{ \"re\": {}, \"im\": {} }},\n",
            jf(lv.c_f64.re),
            jf(lv.c_f64.im)
        ));
        s.push_str(&format!(
            "    \"score\": {{ \"min\": {}, \"mean\": {}, \"max\": {}, \"chosen\": {}, \"chosen_percentile\": {}, \"n_windows\": {} }},\n",
            jf(lv.pick.score_min),
            jf(lv.pick.score_mean),
            jf(lv.pick.score_max),
            jf(lv.pick.chosen_score),
            jf(lv.pick.chosen_pct),
            lv.pick.n_valid,
        ));
        s.push_str(&format!(
            "    \"interior_fraction\": {},\n",
            jf(lv.pick.interior_fraction)
        ));
        s.push_str(&format!("    \"glitch_count\": {},\n", lv.glitch_count));
        s.push_str(&format!("    \"low_signal\": {},\n", lv.pick.low_signal));
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
