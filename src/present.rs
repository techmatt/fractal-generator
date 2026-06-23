//! `present` subcommand — zoom + composition + presentation filter.
//!
//! Takes a `locations.jsonl` from a `generate` run and produces presentation-ready
//! crops. For each seed, tries three composition offsets at cheap 320×180 resolution,
//! picks the one with the lowest black fraction (non-escaped pixel fraction), gates on
//! < 40% black, then renders the accepted composition at full resolution across
//! `--palettes-per-crop` random palettes sampled from the colormap library.
//!
//! Black fraction is computed from raw `PixelSample` escape status — interior pixels
//! (`escaped == false`) are the main black population under `InteriorMode::Black`.
//! This avoids a shading pass just for the composition gate.
//!
//! The three composition offsets (center, thirds, golden) are fixed fw-relative
//! constants, not CLI-tunable. The one with the lowest black fraction wins; if all
//! three exceed the gate, the seed is discarded.

use std::fmt::Write as _;
use std::path::Path;

use image::imageops::FilterType;
use image::RgbImage;
use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::PresentArgs;
use crate::generate::color_params;
use crate::palette::Palette;
use crate::probe::{load_colormap, SplitMix64};
use crate::render::{self, black_fraction, Frame};
use crate::{coloring, ensure_parent_dir, sheet};

/// Cheap render resolution for composition selection.
const CHEAP_W: u32 = 320;
const CHEAP_H: u32 = 180;
const CHEAP_SS: u32 = 1;

/// Gate: discard seed if best composition has black fraction >= this.
const BLACK_THRESH: f32 = 0.40;

/// Thumbnail width for the contact sheet (height 16:9).
const SHEET_THUMB_W: u32 = 240;
const SHEET_THUMB_H: u32 = 135;

/// Escape radius (matches the generate regime).
const BAILOUT: f64 = 1e6;

/// fw-relative offsets for each named composition. Actual center = focus + offset * fw.
/// (dre, dim) in complex-plane units normalised to fw.
const COMP_OFFSETS: &[(&str, f64, f64)] = &[
    ("center", 0.0, 0.0),
    ("thirds", 1.0 / 6.0, 1.0 / 6.0),  // upper-left third
    ("golden", -0.118, -0.118),           // golden-ratio offset
];

struct Seed {
    keeper_index: usize,
    cx: f64,
    cy: f64,
    frame_width: f64,
}

struct CropRecord {
    seed_index: usize,
    cx: f64,
    cy: f64,
    fw: f64,
    composition: &'static str,
    black_fraction: f32,
    coverage: f32,
    palette: String,
    output: String,
}

// ---------- hand-rolled NDJSON field parsers ---------------------------------

fn parse_f64(line: &str, key: &str) -> Result<f64, String> {
    let needle = format!("\"{key}\": ");
    let p = line.find(&needle).ok_or_else(|| format!("missing field '{key}'"))?;
    let rest = &line[p + needle.len()..];
    let end = rest
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("field '{key}': {e}"))
}

fn parse_usize(line: &str, key: &str) -> Result<usize, String> {
    let needle = format!("\"{key}\": ");
    let p = line.find(&needle).ok_or_else(|| format!("missing field '{key}'"))?;
    let rest = &line[p + needle.len()..];
    let end = rest
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<usize>()
        .map_err(|e| format!("field '{key}': {e}"))
}

fn parse_seed(line: &str) -> Result<Seed, String> {
    Ok(Seed {
        keeper_index: parse_usize(line, "keeper_index")?,
        cx: parse_f64(line, "center_re")?,
        cy: parse_f64(line, "center_im")?,
        frame_width: parse_f64(line, "frame_width")?,
    })
}

// ---------- colormap name listing --------------------------------------------

/// Extract all "name" string values from a colormaps JSON array (no serde).
fn list_colormap_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = text;
    while let Some(p) = rest.find("\"name\"") {
        rest = &rest[p + 6..];
        let after = rest.trim_start_matches(|c: char| c == ':' || c.is_ascii_whitespace());
        if let Some(inner) = after.strip_prefix('"') {
            if let Some(end) = inner.find('"') {
                names.push(inner[..end].to_string());
            }
        }
    }
    names
}

// ---------- manifest builder -------------------------------------------------

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

fn build_manifest(
    source: &str,
    zoom_factor: f64,
    total_seeds: usize,
    accepted: usize,
    rejected_black: usize,
    crops: &[CropRecord],
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"source_jsonl\": \"{source}\",");
    let _ = writeln!(s, "  \"zoom_factor\": {zoom_factor},");
    let _ = writeln!(s, "  \"total_seeds\": {total_seeds},");
    let _ = writeln!(s, "  \"accepted\": {accepted},");
    let _ = writeln!(s, "  \"rejected_black\": {rejected_black},");
    s.push_str("  \"crops\": [\n");
    for (i, c) in crops.iter().enumerate() {
        let comma = if i + 1 < crops.len() { "," } else { "" };
        let out_fwd = c.output.replace('\\', "/");
        let _ = writeln!(
            s,
            "    {{ \"seed_index\": {}, \"cx\": {}, \"cy\": {}, \"fw\": {}, \
             \"composition\": \"{}\", \"black_fraction\": {:.4}, \"coverage\": {:.4}, \
             \"palette\": \"{}\", \"output\": \"{out_fwd}\" }}{comma}",
            c.seed_index,
            jnum(c.cx),
            jnum(c.cy),
            jnum(c.fw),
            c.composition,
            c.black_fraction,
            c.coverage,
            c.palette,
        );
    }
    s.push_str("  ]\n");
    s.push('}');
    s.push('\n');
    s
}

// ---------- entry point ------------------------------------------------------

pub fn run_present(args: &PresentArgs) -> Result<(), String> {
    // 1. Parse seeds from locations.jsonl
    let input_text = std::fs::read_to_string(&args.input)
        .map_err(|e| format!("read {}: {e}", args.input))?;
    let seeds: Vec<Seed> = input_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(parse_seed)
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!("present: {} seeds from {}", seeds.len(), args.input);

    // 2. Load palette library
    let cm_text = std::fs::read_to_string(&args.palette_file)
        .map_err(|e| format!("read {}: {e}", args.palette_file))?;
    let all_names = list_colormap_names(&cm_text);
    if all_names.is_empty() {
        return Err(format!("no palettes found in {}", args.palette_file));
    }
    eprintln!("palette library: {} entries", all_names.len());

    let mut rng = SplitMix64(args.seed);
    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // 3. Output directory
    let run_stem = Path::new(&args.input)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("run")
        .to_string();
    let run_dir = Path::new(&args.out_dir).join(&run_stem);
    ensure_parent_dir(run_dir.join("x"))?;

    // Which compositions to try
    let try_comps: &[(&str, f64, f64)] = match args.compositions.as_str() {
        "center" => &COMP_OFFSETS[0..1],
        "thirds" => &COMP_OFFSETS[1..2],
        "golden" => &COMP_OFFSETS[2..3],
        _ => COMP_OFFSETS,
    };

    let mut accepted = 0usize;
    let mut rejected_black = 0usize;
    let mut crops: Vec<CropRecord> = Vec::new();
    let mut sheet_tiles: Vec<RgbImage> = Vec::new();

    for seed in &seeds {
        let new_fw = seed.frame_width * args.zoom_factor;

        // TODO: replace with symmetry-finder focus
        let focus_cx = seed.cx;
        let focus_cy = seed.cy;

        // --- cheap render per composition, pick min black fraction ---
        let mut best: Option<(&'static str, f64, f64, f32)> = None; // (name, cx, cy, bf)
        for &(comp_name, dre, dim) in try_comps {
            let ccx = focus_cx + dre * new_fw;
            let ccy = focus_cy + dim * new_fw;
            let frame = Frame {
                center: Complex::new(ccx, ccy),
                frame_width: new_fw,
                out_width: CHEAP_W,
                out_height: CHEAP_H,
            };
            let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
            let buf = render::iterate_samples_f64(&backend, &frame, CHEAP_SS, channels);
            let bf = black_fraction(&buf.samples);
            if best.is_none() || bf < best.unwrap().3 {
                best = Some((comp_name, ccx, ccy, bf));
            }
        }

        let (comp_name, comp_cx, comp_cy, bf) = best.unwrap();
        let coverage = 1.0 - bf;

        if bf >= BLACK_THRESH {
            eprintln!(
                "  seed {:04} REJECTED  comp={} black_frac={:.3}",
                seed.keeper_index, comp_name, bf
            );
            rejected_black += 1;
            continue;
        }

        eprintln!(
            "  seed {:04} ACCEPTED  comp={} black_frac={:.3} coverage={:.3}",
            seed.keeper_index, comp_name, bf, coverage
        );
        accepted += 1;

        // --- full-res render (iterate once, recolor per palette) ---
        let frame = Frame {
            center: Complex::new(comp_cx, comp_cy),
            frame_width: new_fw,
            out_width: args.width,
            out_height: args.height,
        };
        let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
        let buf = render::iterate_samples_f64(&backend, &frame, args.ss, channels);
        let pixel_spacing = new_fw / args.width as f64;

        for _ in 0..args.palettes_per_crop {
            let pi = rng.below(all_names.len());
            let pal_name = &all_names[pi];
            let stops = match load_colormap(&cm_text, pal_name) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  palette '{pal_name}' load error: {e}");
                    continue;
                }
            };
            let palette = Palette::from_srgb8_stops(pal_name.clone(), &stops, false);
            let img = render::shade_and_downsample(
                &buf.samples,
                args.width,
                args.height,
                args.ss,
                &palette,
                &params,
                pixel_spacing,
            );

            let safe_name =
                pal_name.replace(['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|'], "_");
            let fname = format!("{}_{}.png", seed.keeper_index, safe_name);
            let out_path = run_dir.join(&fname);
            img.save(&out_path)
                .map_err(|e| format!("save {}: {e}", out_path.display()))?;

            let out_str = out_path.to_string_lossy().into_owned();

            let thumb =
                image::imageops::resize(&img, SHEET_THUMB_W, SHEET_THUMB_H, FilterType::Triangle);
            sheet_tiles.push(thumb);

            crops.push(CropRecord {
                seed_index: seed.keeper_index,
                cx: comp_cx,
                cy: comp_cy,
                fw: new_fw,
                composition: comp_name,
                black_fraction: bf,
                coverage,
                palette: pal_name.clone(),
                output: out_str,
            });
        }
    }

    // --- contact sheet ---
    if !sheet_tiles.is_empty() {
        let grid = sheet::compose_grid(&sheet_tiles, Some(8));
        let sheet_path = run_dir.join("present_sheet.png");
        grid.save(&sheet_path).map_err(|e| format!("save sheet: {e}"))?;
        eprintln!("sheet: {}", sheet_path.display());
    }

    // --- manifest ---
    let manifest = build_manifest(
        &args.input,
        args.zoom_factor,
        seeds.len(),
        accepted,
        rejected_black,
        &crops,
    );
    let manifest_path = run_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest)
        .map_err(|e| format!("write manifest: {e}"))?;

    println!("=== present ===");
    println!(
        "seeds={}  accepted={}  rejected_black={}  crops={}",
        seeds.len(),
        accepted,
        rejected_black,
        crops.len()
    );
    println!("manifest: {}", manifest_path.display());
    Ok(())
}
