//! `aa-filter` subcommand — reconstruction-filter bake-off at the 2560×1440
//! target. Placement is settled (grid) and sample count is settled (ss4); the aa-
//! study held the downsample at **box**. This study renders the one combination it
//! never tested: grid ss4 + a *real* reconstruction filter.
//!
//! The frame is iterated **once** (grid ss4, the only expensive stage at ~4.2 s),
//! then the same shaded supersample buffer is collapsed three ways —
//! [`DownsampleFilter::Box`], `Mitchell`, `Lanczos3`. A filter only reweights
//! samples already iterated, so the two non-box cells are ~free; per cell we report
//! **total** (shared iterate + that cell's filter) and **filter time alone**.
//!
//! It reuses `aa_study`'s view/palette/crop machinery: same `present` f64 render
//! path (shallow, asserted), `generate::color_params`, selective-mirror cyclic
//! palette, and the edge-energy crop picker. The 1:1 crop is **pinned** by default
//! to the box the aa-study selected (`--crop`), so all three filters are judged on
//! identical pixels. Output is `tools/viz/aa_filter_study.html` (three crops + full
//! thumbnails) + a JSON log under a stable path (not `out/`).

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use image::imageops::FilterType;
use image::RgbImage;
use num_complex::Complex;

use crate::aa_study::pick_crop;
use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::AaFilterArgs;
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::render::{self, DownsampleFilter, Frame};
use crate::{coloring, ensure_parent_dir, hp};

/// Escape radius (matches the generate/present/aa-study regime).
const BAILOUT: f64 = 1e6;

/// One downsample filter to bake.
struct Cell {
    index: usize,
    label: &'static str,
    tag: &'static str,
    filter: DownsampleFilter,
}

/// The three cells: box baseline, then the two real reconstruction filters.
fn cells() -> Vec<Cell> {
    vec![
        Cell { index: 1, label: "box", tag: "box", filter: DownsampleFilter::Box },
        Cell { index: 2, label: "Mitchell", tag: "mitchell", filter: DownsampleFilter::Mitchell },
        Cell { index: 3, label: "Lanczos-3", tag: "lanczos3", filter: DownsampleFilter::Lanczos3 },
    ]
}

struct CellResult {
    index: usize,
    label: &'static str,
    filter_secs: f64,
    total_secs: f64,
    full_png: String,
    thumb_png: String,
    crop_png: String,
}

pub fn run_aa_filter(args: &AaFilterArgs) -> Result<(), String> {
    if args.width == 0 {
        return Err("--width must be > 0".into());
    }
    let ss = args.supersample.max(1);
    let height = args.width * 9 / 16;

    // Center at full precision (shallow here, but parsed the same way present/
    // render/aa-study do for consistency).
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

    // Shallow-regime assertion: f64 ground truth, never perturbation.
    if pixel_spacing <= 1e-13 {
        return Err(format!(
            "pixel spacing {pixel_spacing:.3e} is in f64's quantization regime — the AA \
             filter study assumes the shallow f64 ground-truth path. Pick a shallower frame width."
        ));
    }

    // Palette through the SAME selective-mirror path as present/aa-study.
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
        "aa-filter: center ({cx:.6}, {cy:.6}) fw {:.3e}  {}x{}  grid ss{ss}  maxiter {}  palette '{}'",
        args.frame_width, args.width, height, args.maxiter, palette.name()
    );

    // --- warm once: spin up the rayon pool + caches on a small frame ---
    {
        let warm = Frame { out_width: 256, out_height: 144, ..frame };
        let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
        let buf = render::iterate_samples_f64(&backend, &warm, 2, channels);
        let _ = render::shade_and_downsample_filtered(
            &buf.samples, 256, 144, 2, &palette, &params, pixel_spacing, DownsampleFilter::Mitchell,
        );
    }

    // --- iterate ONCE (grid ss4); every filter reuses this shaded buffer ---
    let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
    let t_iter = Instant::now();
    let buf = render::iterate_samples_f64(&backend, &frame, ss, channels);
    let iter_secs = t_iter.elapsed().as_secs_f64();
    eprintln!("  iterate (shared) grid ss{ss}  {iter_secs:>6.2}s");

    // --- apply each filter (timed alone) over the same buffer ---
    let mut images: Vec<RgbImage> = Vec::new();
    let mut filter_secs: Vec<f64> = Vec::new();
    for cell in &cells() {
        let t0 = Instant::now();
        let img = render::shade_and_downsample_filtered(
            &buf.samples, args.width, height, ss, &palette, &params, pixel_spacing, cell.filter,
        );
        let fs = t0.elapsed().as_secs_f64();
        eprintln!(
            "  cell {} {:<10} filter {:>6.2}s  total {:>6.2}s",
            cell.index, cell.label, fs, iter_secs + fs
        );
        images.push(img);
        filter_secs.push(fs);
    }
    drop(buf); // free the large ss4 SS buffer

    // --- one matched 1:1 crop box: pinned (default) or auto-picked on box ---
    let (cw, ch) = (args.crop_width.min(args.width), args.crop_height.min(height));
    let (crop_x, crop_y, cw, ch) = match args.resolved_crop()? {
        Some((x, y, w, h)) => (x, y, w.min(args.width), h.min(height)),
        None => {
            let (x, y) = pick_crop(&images[0], cw, ch, 16);
            (x, y, cw, ch)
        }
    };

    finish(args, iter_secs, &images, &filter_secs, crop_x, crop_y, cw, ch, out_dir)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    args: &AaFilterArgs,
    iter_secs: f64,
    images: &[RgbImage],
    filter_secs: &[f64],
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
    for (cell, (img, fs)) in cells().iter().zip(images.iter().zip(filter_secs.iter())) {
        let full_name = format!("cell{}_{}.png", cell.index, cell.tag);
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
            filter_secs: *fs,
            total_secs: iter_secs + *fs,
            full_png: full_name,
            thumb_png: thumb_name,
            crop_png: crop_name,
        });
    }

    let json = build_json(args, iter_secs, crop_x, crop_y, cw, ch, &results);
    std::fs::write(out_dir.join("aa_filter_study.json"), json)
        .map_err(|e| format!("write aa_filter_study.json: {e}"))?;

    let img_folder = out_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("aa_filter_study");
    let html = build_html(args, iter_secs, cw, ch, crop_x, crop_y, img_folder, &results);
    let html_path = out_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("aa_filter_study.html");
    std::fs::write(&html_path, html).map_err(|e| format!("write {}: {e}", html_path.display()))?;

    println!("=== aa-filter ===");
    println!("iterate (shared): {iter_secs:.2}s");
    for r in &results {
        println!(
            "cell {} {:<10} filter {:>6.2}s  total {:>6.2}s",
            r.index, r.label, r.filter_secs, r.total_secs
        );
    }
    println!("html: {}", html_path.display());
    println!("images: {}", out_dir.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_json(
    args: &AaFilterArgs,
    iter_secs: f64,
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
    let _ = writeln!(s, "  \"supersample\": {}, \"spp\": {},", args.supersample, args.supersample * args.supersample);
    let _ = writeln!(s, "  \"maxiter\": {}, \"palette\": \"{}\",", args.maxiter, args.palette);
    let _ = writeln!(s, "  \"iterate_secs\": {iter_secs:.4},");
    let _ = writeln!(
        s,
        "  \"crop\": {{ \"x\": {crop_x}, \"y\": {crop_y}, \"w\": {cw}, \"h\": {ch} }},"
    );
    s.push_str("  \"cells\": [\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{ \"index\": {}, \"filter\": \"{}\", \"filter_secs\": {:.4}, \"total_secs\": {:.4}, \
             \"full\": \"{}\", \"thumb\": \"{}\", \"crop\": \"{}\" }}{comma}",
            r.index, r.label, r.filter_secs, r.total_secs, r.full_png, r.thumb_png, r.crop_png
        );
    }
    s.push_str("  ]\n}\n");
    s
}

#[allow(clippy::too_many_arguments)]
fn build_html(
    args: &AaFilterArgs,
    iter_secs: f64,
    cw: u32,
    ch: u32,
    crop_x: u32,
    crop_y: u32,
    folder: &str,
    results: &[CellResult],
) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    s.push_str("<title>AA filter study — box vs Mitchell vs Lanczos (1440p, ss4)</title>\n<style>\n");
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
    let _ = writeln!(s, "<h1>AA filter study — box vs Mitchell vs Lanczos @ 2560×1440, grid ss{} (16 spp)</h1>", args.supersample);
    let _ = writeln!(
        s,
        "<p class=\"meta\">center <code>{}, {}</code> · frame_width <code>{}</code> · \
         maxiter {} · palette <code>{}</code> · single shared iterate <code>{iter_secs:.2}s</code> · \
         1:1 crop {cw}×{ch} at ({crop_x},{crop_y}). One render set, three downsample filters over the \
         identical ss4 buffer; the filter only reweights samples already iterated.</p>",
        args.center_re, args.center_im, args.frame_width, args.maxiter, args.palette
    );

    // 1:1 crops — the comparison, side by side on the identical box.
    s.push_str("<h2>1:1 crops (the comparison — the filter is sub-pixel)</h2>\n<div class=\"row\">\n");
    for r in results {
        let _ = writeln!(
            s,
            "  <div class=\"card\"><div class=\"lab\">{}. {} <span class=\"sub\">· filter {:.2}s · total {:.2}s</span></div>\
             <img src=\"{folder}/{}\" width=\"{cw}\" height=\"{ch}\"></div>",
            r.index, r.label, r.filter_secs, r.total_secs, r.crop_png
        );
    }
    s.push_str("</div>\n");

    // Full-frame thumbnails.
    s.push_str("<h2>Full frames (thumbnails)</h2>\n<div class=\"row\">\n");
    for r in results {
        let _ = writeln!(
            s,
            "  <div class=\"card thumb\"><div class=\"lab\">{}. {} <span class=\"sub\">· filter {:.2}s · \
             total {:.2}s</span></div><img src=\"{folder}/{}\"><div class=\"sub\">\
             <a style=\"color:#7fd1b9\" href=\"{folder}/{}\">full 2560×1440 PNG</a></div></div>",
            r.index, r.label, r.filter_secs, r.total_secs, r.thumb_png, r.full_png
        );
    }
    s.push_str("</div>\n</body>\n</html>\n");
    s
}
