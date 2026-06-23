//! `palette-score` subcommand — the palette hand-scoring surface.
//!
//! Builds a surface that **decouples palette quality from location quality**: a
//! fixed fixture of 4 diverse "good" views is iterated **once** (sharing
//! `present`'s coloring config), then **recolored** into a clean 2×2 grid under
//! **each** of the 224 survivor palettes. Matt hand-labels each palette
//! `1` (universally bad) / `2` (sometimes good) / `3` (generally good) from the
//! grids; the score-3 set later becomes the random roster for the location pass.
//!
//! ## Why this shape
//! The 4 views are the **controlled axis** — a palette that sings on a spiral but
//! dies on a junction reads as "sometimes" (2). That's the signal, so the views
//! must be diverse structures (Matt fills views 2–4 via the preview gate).
//!
//! ## Iterate-once / recolor-many
//! The per-pixel field a cell consumes (smooth-iter / trap `t`, whatever
//! [`present`]'s [`color_params`] reads) is **palette-independent**; only the LUT
//! lookup (+ selective mirror) varies. So the 4 views are iterated a single time
//! and every one of the 224 palettes is an O(LUT) re-shade
//! ([`render::shade_and_downsample`], pure). The whole sweep is dominated by that
//! one iterate plus PNG encode/IO — recolor is ~3 orders cheaper than iterate.
//!
//! ## Config parity (scores must transfer)
//! Cells render **identically** to `present`'s location crops: same [`Trap`],
//! same [`BAILOUT`], same [`color_params`], same `iterate_samples_f64` /
//! `shade_and_downsample` path, same `pixel_spacing = fw / cell_width`, and the
//! **selective mirror** (pre-mirror SEQUENTIAL `mirror_needed` maps, single-pass
//! cyclic maps) applied exactly as `palette-pick`. A center-composition cyclic
//! cell is byte-identical to a `present` crop of the same `(cx, cy, fw)`.
//!
//! ## Preview gate (visual-first)
//! Without `--full` the run stops after writing a single `twilight_shifted` 2×2
//! `preview.png` — the gate where Matt eyeballs the fixture, edits `views.json`,
//! and re-runs until the 4 views are right. `--full` then renders the 224 grids
//! + a manifest. No quality claims; Matt judges.

use std::fmt::Write as _;
use std::path::Path;

use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::PaletteScoreArgs;
use crate::coloring::{self, ColorParams};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::{parse_colormaps, Colormap, Json, JsonParser};
use crate::present::composition_offset;
use crate::render::{self, Frame, SampleBuffer};
use crate::ensure_parent_dir;

/// Escape radius — matches `present`'s `BAILOUT` (config parity).
const BAILOUT: f64 = 1e6;

/// Grid background (neutral dark gray): a calm surround so palette color is judged
/// honestly, not against white or black.
const BG: Rgb<u8> = Rgb([32, 32, 32]);

/// One fixture view: a center, a frame width, and a named composition offset.
struct View {
    name: String,
    cx: f64,
    cy: f64,
    fw: f64,
    composition: String,
}

/// A view iterated once: its palette-independent sample field + render geometry.
struct CachedView {
    view_name: String,
    /// Composition-resolved frame center (focus + offset·fw).
    center: Complex<f64>,
    fw: f64,
    spacing: f64,
    buf: SampleBuffer,
}

// ---------- views.json parsing (reuses palette_pick's JSON parser) -----------

fn obj_get<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    obj.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn obj_num(obj: &[(String, Json)], key: &str) -> Result<f64, String> {
    match obj_get(obj, key) {
        Some(Json::Num(n)) => Ok(*n),
        _ => Err(format!("view: missing/non-numeric '{key}'")),
    }
}

fn parse_views(text: &str) -> Result<Vec<View>, String> {
    let v = JsonParser::new(text).parse()?;
    let arr = match v {
        Json::Arr(a) => a,
        _ => return Err("views.json top-level must be an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, entry) in arr.iter().enumerate() {
        let obj = match entry {
            Json::Obj(o) => o,
            _ => return Err(format!("views.json entry {i} is not an object")),
        };
        let name = match obj_get(obj, "name") {
            Some(Json::Str(s)) => s.clone(),
            _ => format!("view{i}"),
        };
        let composition = match obj_get(obj, "composition") {
            Some(Json::Str(s)) => s.clone(),
            _ => "center".to_string(),
        };
        out.push(View {
            name,
            cx: obj_num(obj, "cx")?,
            cy: obj_num(obj, "cy")?,
            fw: obj_num(obj, "fw")?,
            composition,
        });
    }
    Ok(out)
}

// ---------- 2×2 grid compositor (thin gutter, no burned-in text) -------------

/// Composite 4 equal-size cells into a 2×2 grid with a `gutter`-px neutral border
/// between and around them. Row-major: `[v0 v1 / v2 v3]`.
fn compose_2x2(cells: &[RgbImage; 4], gutter: u32) -> RgbImage {
    let cw = cells[0].width();
    let ch = cells[0].height();
    let w = 2 * cw + 3 * gutter;
    let h = 2 * ch + 3 * gutter;
    let mut grid = RgbImage::from_pixel(w, h, BG);
    for (i, cell) in cells.iter().enumerate() {
        let c = (i % 2) as u32;
        let r = (i / 2) as u32;
        let x0 = gutter + c * (cw + gutter);
        let y0 = gutter + r * (ch + gutter);
        for y in 0..ch {
            for x in 0..cw {
                grid.put_pixel(x0 + x, y0 + y, *cell.get_pixel(x, y));
            }
        }
    }
    grid
}

// ---------- iterate the 4 views once -----------------------------------------

fn iterate_views(
    views: &[View],
    cell_w: u32,
    cell_h: u32,
    ss: u32,
    maxiter: u32,
    params: &ColorParams,
) -> Vec<CachedView> {
    let channels = coloring::required_channels(params);
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };
    views
        .iter()
        .map(|v| {
            let (dre, dim) = composition_offset(&v.composition);
            let center = Complex::new(v.cx + dre * v.fw, v.cy + dim * v.fw);
            let frame = Frame {
                center,
                frame_width: v.fw,
                out_width: cell_w,
                out_height: cell_h,
            };
            let backend = F64Backend::new(maxiter, BAILOUT, trap);
            let buf = render::iterate_samples_f64(&backend, &frame, ss, channels);
            CachedView {
                view_name: v.name.clone(),
                center,
                fw: v.fw,
                spacing: v.fw / cell_w as f64,
                buf,
            }
        })
        .collect()
}

/// Recolor the 4 cached views under one palette and composite the 2×2 grid.
fn grid_for_palette(
    cached: &[CachedView],
    palette: &Palette,
    cell_w: u32,
    cell_h: u32,
    ss: u32,
    gutter: u32,
    params: &ColorParams,
) -> RgbImage {
    let cells: Vec<RgbImage> = cached
        .iter()
        .map(|cv| {
            render::shade_and_downsample(
                &cv.buf.samples,
                cell_w,
                cell_h,
                ss,
                palette,
                params,
                cv.spacing,
            )
        })
        .collect();
    let arr: [RgbImage; 4] = [
        cells[0].clone(),
        cells[1].clone(),
        cells[2].clone(),
        cells[3].clone(),
    ];
    compose_2x2(&arr, gutter)
}

/// Sanitize a palette name to a filesystem-safe stem (matches `present`).
fn safe_name(name: &str) -> String {
    name.replace(['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|'], "_")
}

// ---------- manifest ---------------------------------------------------------

fn build_manifest(
    views: &[View],
    cached: &[CachedView],
    cell_w: u32,
    cell_h: u32,
    ss: u32,
    maxiter: u32,
    params: &ColorParams,
    rows: &[(String, String, bool, Option<String>)], // (palette, grid_path, mirror_needed, cycle)
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"cell_width\": {cell_w},");
    let _ = writeln!(s, "  \"cell_height\": {cell_h},");
    let _ = writeln!(s, "  \"ss\": {ss},");
    let _ = writeln!(s, "  \"maxiter\": {maxiter},");
    let _ = writeln!(s, "  \"bailout\": {BAILOUT},");
    let _ = writeln!(s, "  \"channel\": \"{:?}\",", params.channel);
    let _ = writeln!(s, "  \"density\": {},", params.density);
    let _ = writeln!(s, "  \"interior\": \"{:?}\",", params.interior);
    // Views (with composition-resolved centers, for provenance).
    s.push_str("  \"views\": [\n");
    for (i, (v, cv)) in views.iter().zip(cached.iter()).enumerate() {
        let comma = if i + 1 < views.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{ \"name\": \"{}\", \"cx\": {}, \"cy\": {}, \"fw\": {}, \
             \"composition\": \"{}\", \"render_cx\": {}, \"render_cy\": {} }}{comma}",
            v.name, v.cx, v.cy, v.fw, v.composition, cv.center.re, cv.center.im,
        );
    }
    s.push_str("  ],\n");
    // Palette rows.
    s.push_str("  \"palettes\": [\n");
    for (i, (pal, path, mirror, cycle)) in rows.iter().enumerate() {
        let comma = if i + 1 < rows.len() { "," } else { "" };
        let cyc = match cycle {
            Some(c) => format!("\"{c}\""),
            None => "null".to_string(),
        };
        let fwd = path.replace('\\', "/");
        let _ = writeln!(
            s,
            "    {{ \"palette\": \"{pal}\", \"grid_path\": \"{fwd}\", \
             \"mirror_needed\": {mirror}, \"cycle\": {cyc} }}{comma}",
        );
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

// ---------- entry point ------------------------------------------------------

pub fn run_palette_score(args: &PaletteScoreArgs) -> Result<(), String> {
    // 1. Load + validate the 4-view fixture.
    let views_text = std::fs::read_to_string(&args.views)
        .map_err(|e| format!("read {}: {e}", args.views))?;
    let views = parse_views(&views_text)?;
    if views.len() != 4 {
        return Err(format!(
            "views.json must hold exactly 4 views, found {}",
            views.len()
        ));
    }

    // 2. Load the survivor palette library (inline cycle / mirror_needed).
    let lib_text = std::fs::read_to_string(&args.palette_file)
        .map_err(|e| format!("read {}: {e}", args.palette_file))?;
    let library: Vec<Colormap> = parse_colormaps(&lib_text)
        .map_err(|e| format!("parse {}: {e}", args.palette_file))?;
    if library.is_empty() {
        return Err(format!("no palettes in {}", args.palette_file));
    }

    let params = color_params();
    let cell_w = args.cell_width.max(1);
    let cell_h = args.cell_height.max(1);

    eprintln!(
        "palette-score: 4 views, library {} palettes, cell {}x{} ss{} maxiter {}",
        library.len(),
        cell_w,
        cell_h,
        args.ss,
        args.maxiter,
    );

    // 3. Iterate the 4 views ONCE (the only backend touch).
    let cached = iterate_views(&views, cell_w, cell_h, args.ss, args.maxiter, &params);

    let out_dir = Path::new(args.out_dir.trim_end_matches('/'));

    // 4. Preview gate — stop here unless --full.
    if !args.full {
        let diag = library
            .iter()
            .find(|c| c.name == args.diagnostic_palette)
            .ok_or_else(|| {
                format!(
                    "diagnostic palette '{}' not found in {}",
                    args.diagnostic_palette, args.palette_file
                )
            })?;
        let palette = Palette::from_srgb8_stops_mirrored(
            diag.name.clone(),
            &diag.stops,
            false,
            diag.mirror_needed,
        );
        let grid = grid_for_palette(
            &cached, &palette, cell_w, cell_h, args.ss, args.gutter, &params,
        );
        let preview_path = out_dir.join("preview.png");
        ensure_parent_dir(&preview_path)?;
        grid.save(&preview_path)
            .map_err(|e| format!("save {}: {e}", preview_path.display()))?;
        println!("=== palette-score (preview gate) ===");
        for cv in &cached {
            println!(
                "  view '{}': center ({}, {}) fw {:.3e}",
                cv.view_name, cv.center.re, cv.center.im, cv.fw
            );
        }
        println!("diagnostic palette: {}", diag.name);
        println!("preview: {}", preview_path.display());
        println!("edit {} and re-run; pass --full to render the 224-palette sweep.", args.views);
        return Ok(());
    }

    // 5. Full sweep — recolor the 4 cached views under every palette.
    let grids_dir = out_dir.join("grids");
    ensure_parent_dir(grids_dir.join("x"))?;

    let mut rows: Vec<(String, String, bool, Option<String>)> = Vec::with_capacity(library.len());
    for (i, cm) in library.iter().enumerate() {
        let palette = Palette::from_srgb8_stops_mirrored(
            cm.name.clone(),
            &cm.stops,
            false,
            cm.mirror_needed,
        );
        let grid = grid_for_palette(
            &cached, &palette, cell_w, cell_h, args.ss, args.gutter, &params,
        );
        let fname = format!("{}.png", safe_name(&cm.name));
        let path = grids_dir.join(&fname);
        grid.save(&path)
            .map_err(|e| format!("save {}: {e}", path.display()))?;
        rows.push((
            cm.name.clone(),
            path.to_string_lossy().into_owned(),
            cm.mirror_needed,
            cm.cycle.clone(),
        ));
        if (i + 1) % 32 == 0 || i + 1 == library.len() {
            eprintln!("  {}/{} grids", i + 1, library.len());
        }
    }

    // 6. Manifest.
    let manifest = build_manifest(
        &views, &cached, cell_w, cell_h, args.ss, args.maxiter, &params, &rows,
    );
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest)
        .map_err(|e| format!("write {}: {e}", manifest_path.display()))?;

    println!("=== palette-score ===");
    println!("palettes: {}  grids: {}", library.len(), rows.len());
    println!("grids dir: {}", grids_dir.display());
    println!("manifest: {}", manifest_path.display());
    Ok(())
}
