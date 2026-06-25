//! `aa-study` subcommand — antialiasing bake-off at the real 2560×1440 target.
//!
//! Two axes at one fixed view + palette: **sample count** (ordered grid ss2/ss3/ss4
//! = 4/9/16 spp) and **sub-sample placement** at a fixed 4-spp budget (ordered grid
//! vs rotated-grid 4-rooks vs stratified jitter). Box downsample stays fixed this
//! round (a reconstruction-filter study is the deliberate next step); see
//! `prompts/aa-study-prompt.md`.
//!
//! It reuses `present`'s render path verbatim — f64 backend (the view is shallow,
//! asserted by the auto-regime), `generate::color_params`, and the selective-mirror
//! cyclic palette load — and only varies the [`render::SubsamplePattern`]. Per cell
//! it reports wall-clock (the standalone "single full render timing" answer too),
//! saves the full 2560×1440 PNG, and extracts a **matched 1:1 crop** of one
//! high-frequency region (auto-picked from the ss4 cell by edge energy, identical
//! box for all cells). Output is an `aa_study.html` (thumbnail + 1:1 crop per cell)
//! plus a JSON log, persisted under a stable path (not `out/`).

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use image::imageops::FilterType;
use image::RgbImage;
use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::AaStudyArgs;
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::probe::PERTURB_SPACING;
use crate::render::{self, Frame, SubsamplePattern};
use crate::{coloring, ensure_parent_dir, hp};

/// Escape radius (matches the generate/present regime).
const BAILOUT: f64 = 1e6;

/// One AA scheme to render.
struct Cell {
    /// 1-based index, for filenames + labels.
    index: usize,
    /// Human label (`grid ss2`, `rgss ss2`, …).
    label: &'static str,
    /// Filename-safe scheme tag.
    tag: &'static str,
    pattern: SubsamplePattern,
    ss: u32,
    spp: u32,
}

/// The five-cell matrix: 1/4/5 isolate placement at 4 spp; 1/2/3 isolate count.
fn cells() -> Vec<Cell> {
    vec![
        Cell { index: 1, label: "grid ss2", tag: "grid_ss2", pattern: SubsamplePattern::Grid, ss: 2, spp: 4 },
        Cell { index: 2, label: "grid ss3", tag: "grid_ss3", pattern: SubsamplePattern::Grid, ss: 3, spp: 9 },
        Cell { index: 3, label: "grid ss4", tag: "grid_ss4", pattern: SubsamplePattern::Grid, ss: 4, spp: 16 },
        Cell { index: 4, label: "rgss ss2", tag: "rgss_ss2", pattern: SubsamplePattern::Rgss, ss: 2, spp: 4 },
        Cell { index: 5, label: "jitter ss2", tag: "jitter_ss2", pattern: SubsamplePattern::Jitter, ss: 2, spp: 4 },
    ]
}

struct CellResult {
    index: usize,
    label: &'static str,
    spp: u32,
    wall_secs: f64,
    iter_secs: f64,
    full_png: String,
    thumb_png: String,
    crop_png: String,
}

pub fn run_aa_study(args: &AaStudyArgs) -> Result<(), String> {
    if args.width == 0 {
        return Err("--width must be > 0".into());
    }
    let height = args.width * 9 / 16;

    // Center at full precision (an f64 center is meaningless at depth; here it is
    // shallow, but parse the same way present/render do for consistency).
    let prec_bits = hp::prec_bits(args.width, args.frame_width);
    let cx = hp::to_f64(&hp::parse_decimal(&args.center_re, prec_bits)?);
    let cy = hp::to_f64(&hp::parse_decimal(&args.center_im, prec_bits)?);
    let center = Complex::new(cx, cy);

    let frame = Frame {
        center,
        frame_width: args.frame_width,
        out_width: args.width,
        out_height: height,
    };
    let pixel_spacing = args.frame_width / args.width as f64;

    // Shallow-regime assertion: this study is f64 ground truth, never perturbation.
    if pixel_spacing <= PERTURB_SPACING {
        return Err(format!(
            "pixel spacing {pixel_spacing:.3e} is in f64's quantization regime — the AA \
             study assumes the shallow f64 ground-truth path. Pick a shallower frame width."
        ));
    }

    // Palette: load the named cyclic map through the SAME selective-mirror path as
    // present/palette-score (mirror_needed carries the sequential de-seam; a cyclic
    // map like twilight is single-pass).
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text)
        .map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    let cm = library
        .iter()
        .find(|c| c.name == args.palette)
        .ok_or_else(|| format!("palette '{}' not found in {}", args.palette, args.colormaps))?;
    let palette =
        Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed);

    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    let out_dir = Path::new(&args.out_dir);
    ensure_parent_dir(out_dir.join("x"))?;

    eprintln!(
        "aa-study: center ({cx:.6}, {cy:.6}) fw {:.3e}  {}x{}  maxiter {}  palette '{}'",
        args.frame_width, args.width, height, args.maxiter, palette.name()
    );

    // --- warm once: spin up the rayon pool + caches on a small frame ---
    {
        let warm = Frame { out_width: 256, out_height: 144, ..frame };
        let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
        let buf = render::iterate_samples_f64(&backend, &warm, 2, channels);
        let _ = render::shade_and_downsample(
            &buf.samples, 256, 144, 2, &palette, &params, pixel_spacing,
        );
    }

    // --- render every cell (timed), keep the full image, drop the SS buffer ---
    let mut images: Vec<RgbImage> = Vec::new();
    let mut timings: Vec<(f64, f64)> = Vec::new(); // (wall, iterate)
    for cell in &cells() {
        let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
        let t0 = Instant::now();
        let buf = render::iterate_samples_f64_pattern(
            &backend, &frame, cell.ss, channels, cell.pattern, args.seed,
        );
        let iter_secs = t0.elapsed().as_secs_f64();
        let img = render::shade_and_downsample(
            &buf.samples, args.width, height, cell.ss, &palette, &params, pixel_spacing,
        );
        let wall = t0.elapsed().as_secs_f64();
        drop(buf); // free the (large at ss4) SS buffer before the next cell
        eprintln!(
            "  cell {} {:<10} spp {:>2}  iterate {:>6.2}s  total {:>6.2}s",
            cell.index, cell.label, cell.spp, iter_secs, wall
        );
        images.push(img);
        timings.push((wall, iter_secs));
    }

    // --- one matched 1:1 crop box, picked on the cleanest cell (ss4) by edge energy ---
    let (cw, ch) = (args.crop_width.min(args.width), args.crop_height.min(height));
    let (crop_x, crop_y) = match args.resolved_crop()? {
        Some((x, y, w, h)) => {
            // explicit box wins; clamp the size lever to it
            return finish(args, &images, &timings, x, y, w.min(args.width), h.min(height), out_dir);
        }
        None => pick_crop(&images[2], cw, ch, 16),
    };

    finish(args, &images, &timings, crop_x, crop_y, cw, ch, out_dir)
}

/// Write all per-cell PNGs/thumbs/crops + the JSON log + the HTML viewer.
fn finish(
    args: &AaStudyArgs,
    images: &[RgbImage],
    timings: &[(f64, f64)],
    crop_x: u32,
    crop_y: u32,
    cw: u32,
    ch: u32,
    out_dir: &Path,
) -> Result<(), String> {
    eprintln!("crop box: {cw}x{ch} at ({crop_x},{crop_y})");
    let thumb_w = 640u32;
    let thumb_h = thumb_w * images[0].height() / images[0].width();

    let mut results: Vec<CellResult> = Vec::new();
    for (cell, (img, (wall, iter_secs))) in cells().iter().zip(images.iter().zip(timings.iter())) {
        let full_name = format!("cell{}_{}_spp{}.png", cell.index, cell.tag, cell.spp);
        let thumb_name = format!("cell{}_{}_thumb.png", cell.index, cell.tag);
        let crop_name = format!("cell{}_{}_crop.png", cell.index, cell.tag);

        img.save(out_dir.join(&full_name))
            .map_err(|e| format!("save {full_name}: {e}"))?;

        let thumb = image::imageops::resize(img, thumb_w, thumb_h, FilterType::Triangle);
        thumb.save(out_dir.join(&thumb_name))
            .map_err(|e| format!("save {thumb_name}: {e}"))?;

        let crop = image::imageops::crop_imm(img, crop_x, crop_y, cw, ch).to_image();
        crop.save(out_dir.join(&crop_name))
            .map_err(|e| format!("save {crop_name}: {e}"))?;

        results.push(CellResult {
            index: cell.index,
            label: cell.label,
            spp: cell.spp,
            wall_secs: *wall,
            iter_secs: *iter_secs,
            full_png: full_name,
            thumb_png: thumb_name,
            crop_png: crop_name,
        });
    }

    // JSON log
    let json = build_json(args, crop_x, crop_y, cw, ch, &results);
    std::fs::write(out_dir.join("aa_study.json"), json)
        .map_err(|e| format!("write aa_study.json: {e}"))?;

    // HTML viewer — written alongside tools/viz/*.html, referencing the out_dir
    // basename as the image folder (so paths stay relative to tools/viz/).
    let img_folder = out_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("aa_study");
    let html = build_html(args, cw, ch, crop_x, crop_y, img_folder, &results);
    let html_path = out_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("aa_study.html");
    std::fs::write(&html_path, html).map_err(|e| format!("write {}: {e}", html_path.display()))?;

    println!("=== aa-study ===");
    for r in &results {
        println!(
            "cell {} {:<10} spp {:>2}  {:>6.2}s (iterate {:>6.2}s)",
            r.index, r.label, r.spp, r.wall_secs, r.iter_secs
        );
    }
    println!("html: {}", html_path.display());
    println!("images: {}", out_dir.display());
    Ok(())
}

/// Pick the `cw×ch` window with the highest forward-difference luma edge energy
/// (a "high-frequency region"). Summed-area table → O(1) per candidate window,
/// scanned on a `step`-pixel grid. Returns the window top-left.
pub(crate) fn pick_crop(img: &RgbImage, cw: u32, ch: u32, step: u32) -> (u32, u32) {
    let w = img.width();
    let h = img.height();
    if cw >= w || ch >= h {
        return (0, 0);
    }
    let wn = w as usize;
    let hn = h as usize;

    // Luma, then forward-difference edge energy.
    let mut lum = vec![0f64; wn * hn];
    for y in 0..hn {
        for x in 0..wn {
            let p = img.get_pixel(x as u32, y as u32).0;
            lum[y * wn + x] = 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
        }
    }
    let mut e = vec![0f64; wn * hn];
    for y in 0..hn {
        for x in 0..wn {
            let i = y * wn + x;
            let gx = if x + 1 < wn { (lum[i + 1] - lum[i]).abs() } else { 0.0 };
            let gy = if y + 1 < hn { (lum[i + wn] - lum[i]).abs() } else { 0.0 };
            e[i] = gx + gy;
        }
    }

    // Summed-area table, padded by one row/col of zeros.
    let sw = wn + 1;
    let mut sat = vec![0f64; sw * (hn + 1)];
    for y in 0..hn {
        let mut row_sum = 0.0;
        for x in 0..wn {
            row_sum += e[y * wn + x];
            sat[(y + 1) * sw + (x + 1)] = sat[y * sw + (x + 1)] + row_sum;
        }
    }
    let win = |x0: u32, y0: u32| -> f64 {
        let (x1, y1) = (x0 + cw, y0 + ch);
        let g = |x: u32, y: u32| sat[y as usize * sw + x as usize];
        g(x1, y1) - g(x0, y1) - g(x1, y0) + g(x0, y0)
    };

    let (mut best, mut best_v) = ((0u32, 0u32), -1.0f64);
    let mut y0 = 0;
    while y0 + ch <= h {
        let mut x0 = 0;
        while x0 + cw <= w {
            let v = win(x0, y0);
            if v > best_v {
                best_v = v;
                best = (x0, y0);
            }
            x0 += step;
        }
        y0 += step;
    }
    best
}

fn build_json(
    args: &AaStudyArgs,
    crop_x: u32,
    crop_y: u32,
    cw: u32,
    ch: u32,
    results: &[CellResult],
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"center_re\": \"{}\",", args.center_re);
    let _ = writeln!(s, "  \"center_im\": \"{}\",", args.center_im);
    let _ = writeln!(s, "  \"frame_width\": {},", args.frame_width);
    let _ = writeln!(s, "  \"width\": {}, \"height\": {},", args.width, args.width * 9 / 16);
    let _ = writeln!(s, "  \"maxiter\": {}, \"palette\": \"{}\", \"seed\": {},", args.maxiter, args.palette, args.seed);
    let _ = writeln!(
        s,
        "  \"crop\": {{ \"x\": {crop_x}, \"y\": {crop_y}, \"w\": {cw}, \"h\": {ch} }},"
    );
    s.push_str("  \"cells\": [\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{ \"index\": {}, \"scheme\": \"{}\", \"spp\": {}, \"wall_secs\": {:.4}, \
             \"iterate_secs\": {:.4}, \"full\": \"{}\", \"thumb\": \"{}\", \"crop\": \"{}\" }}{comma}",
            r.index, r.label, r.spp, r.wall_secs, r.iter_secs, r.full_png, r.thumb_png, r.crop_png
        );
    }
    s.push_str("  ]\n}\n");
    s
}

fn build_html(
    args: &AaStudyArgs,
    cw: u32,
    ch: u32,
    crop_x: u32,
    crop_y: u32,
    folder: &str,
    results: &[CellResult],
) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    s.push_str("<title>AA study — sample count × placement (1440p)</title>\n<style>\n");
    s.push_str(
        "body{background:#111;color:#ddd;font:13px/1.5 system-ui,sans-serif;margin:24px;}\n\
         h1{font-size:18px;} h2{font-size:15px;margin-top:28px;border-bottom:1px solid #333;padding-bottom:4px;}\n\
         .meta{color:#999;}\n\
         .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}\n\
         .card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;padding:8px;}\n\
         .card .lab{font-weight:600;margin-bottom:6px;} .card .sub{color:#9a9a9a;}\n\
         img{display:block;image-rendering:pixelated;background:#000;}\n\
         .thumb img{width:480px;height:auto;image-rendering:auto;}\n\
         code{color:#7fd1b9;}\n",
    );
    s.push_str("</style>\n</head>\n<body>\n");
    let _ = writeln!(s, "<h1>AA study — antialiasing bake-off @ 2560×1440</h1>");
    let _ = writeln!(
        s,
        "<p class=\"meta\">center <code>{}, {}</code> · frame_width <code>{}</code> · \
         maxiter {} · palette <code>{}</code> · box downsample · 1:1 crop {cw}×{ch} at \
         ({crop_x},{crop_y}). Cells 1/4/5 isolate <b>placement</b> at 4&nbsp;spp; \
         1/2/3 isolate <b>count</b>.</p>",
        args.center_re, args.center_im, args.frame_width, args.maxiter, args.palette
    );

    // 1:1 crops — the actual comparison, side by side on the identical box.
    s.push_str("<h2>1:1 crops (the comparison — AA is sub-pixel)</h2>\n<div class=\"row\">\n");
    for r in results {
        let _ = writeln!(
            s,
            "  <div class=\"card\"><div class=\"lab\">{}. {} <span class=\"sub\">· {} spp · {:.2}s</span></div>\
             <img src=\"{folder}/{}\" width=\"{cw}\" height=\"{ch}\"></div>",
            r.index, r.label, r.spp, r.wall_secs, r.crop_png
        );
    }
    s.push_str("</div>\n");

    // Full-frame thumbnails.
    s.push_str("<h2>Full frames (thumbnails)</h2>\n<div class=\"row\">\n");
    for r in results {
        let _ = writeln!(
            s,
            "  <div class=\"card thumb\"><div class=\"lab\">{}. {} <span class=\"sub\">· {} spp · {:.2}s \
             (iterate {:.2}s)</span></div><img src=\"{folder}/{}\"><div class=\"sub\">\
             <a style=\"color:#7fd1b9\" href=\"{folder}/{}\">full 2560×1440 PNG</a></div></div>",
            r.index, r.label, r.spp, r.wall_secs, r.iter_secs, r.thumb_png, r.full_png
        );
    }
    s.push_str("</div>\n</body>\n</html>\n");
    s
}
