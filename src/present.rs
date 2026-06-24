//! `present` subcommand — zoom + composition + presentation filter.
//!
//! Takes a `locations.jsonl` from a `generate` run and produces presentation-ready
//! crops. For each seed, tries three composition offsets at cheap 320×180 resolution,
//! picks the one with the lowest black fraction (non-escaped pixel fraction), gates on
//! ≤ 30% black (see `BLACK_THRESH`), then renders the accepted composition at full resolution across
//! `--palettes-per-crop` random palettes sampled from the colormap library.
//!
//! Black fraction is computed from raw `PixelSample` escape status — interior pixels
//! (`escaped == false`) are the main black population under `InteriorMode::Black`.
//! This avoids a shading pass just for the composition gate.
//!
//! The three composition offsets (center, thirds, golden) are fixed fw-relative
//! constants, not CLI-tunable. The one with the lowest black fraction wins; if all
//! three exceed the gate, the seed is discarded.
//!
//! **Full-AA render + occupancy gate.** The full render reuses `render-one`'s
//! locked quality path — grid ss (`--ss`, run at 4) + Lanczos-3 reconstruction
//! (`shade_and_downsample_filtered`). After iteration (once per seed×composition)
//! and **before** the library palettes, an optional **occupancy gate**
//! (`--occupancy-floor`) discards sparse/corner-only crops: it shades the crop
//! once with the `default` palette and scores `energy::occupancy` (the loose0
//! calibration — fraction of 32×18 tiles whose mean OKLab edge energy > 0.010) at
//! native resolution; below the floor, the crop is dropped with no library-palette
//! render (a small default-palette thumbnail is kept for the sanity sheet). Crops
//! emit as PNG or JPEG (`--format`/`--jpg-quality`). `--flat-out` writes directly
//! into `--out-dir`. Outputs: the JPG/PNG crops, `manifest.json` (with per-crop
//! `occupancy`), `sanity_sheet.html` (occupancy-sorted survivors + margin rejects),
//! and the legacy `present_sheet.png` thumbnail grid.

use std::fmt::Write as _;
use std::path::Path;

use image::imageops::FilterType;
use image::RgbImage;
use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::PresentArgs;
use crate::energy::{self, OCC_FLOOR, OCC_GX, OCC_GY};
use crate::generate::color_params;
use crate::palette::{builtin, Palette};
use crate::palette_pick::{parse_colormaps, Colormap};
use crate::probe::SplitMix64;
use crate::render::{self, black_fraction, DownsampleFilter, Frame};
use crate::{coloring, ensure_parent_dir, sheet};

/// Locked wallpaper-quality reconstruction filter (matches `render-one`): the
/// full render uses grid-ss (CLI `--ss`, run at 4) + Lanczos-3 downsample.
const FULL_FILTER: DownsampleFilter = DownsampleFilter::Lanczos3;
/// Reject-thumbnail width for the sanity sheet (default-palette gate image).
const REJECT_THUMB_W: u32 = 480;

/// Cheap render resolution for composition selection.
const CHEAP_W: u32 = 320;
const CHEAP_H: u32 = 180;
const CHEAP_SS: u32 = 1;

/// Gate: discard a (seed × composition) crop when its no-escape (black) fraction
/// is **> this**. Tightened 0.40 → 0.30 (the `maxiter-blackgate` pass) once the
/// iteration cap was raised: raising `maxiter` lowers the no-escape fraction
/// (spiral-core pixels formerly pinned at the cap now escape), so a 0.30 ceiling
/// on the high-iter renders rejects genuinely interior-dominated crops without
/// losing the resolved-core frames. Calibrated on the cap-raised no-escape
/// distribution (see `maxiter-diag`'s gate-calibration report).
const BLACK_THRESH: f32 = 0.30;

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

/// Resolve a named composition (`center` / `thirds` / `golden`) to its
/// fw-relative `(dre, dim)` offset. Unknown names fall back to `center` (0, 0).
/// Shared with `palette-score` so scored crops use the same offsets as `present`.
pub(crate) fn composition_offset(name: &str) -> (f64, f64) {
    for &(n, dre, dim) in COMP_OFFSETS {
        if n == name {
            return (dre, dim);
        }
    }
    (0.0, 0.0)
}

struct Seed {
    keeper_index: usize,
    /// Standardized label key carried from the `generate` draw stream.
    draw_index: usize,
    /// Cheap-screen interior (max-iter) fraction from the seed's draw log.
    interior_frac: f64,
    cx: f64,
    cy: f64,
    frame_width: f64,
}

struct CropRecord {
    seed_index: usize,
    draw_index: usize,
    interior_frac: f64,
    cx: f64,
    cy: f64,
    fw: f64,
    /// Chosen focus center (content-centered or seed-center per `--focus`); the
    /// composition center `cx,cy` is this focus plus the composition offset.
    focus_cx: f64,
    focus_cy: f64,
    /// The content-focus void guard fired for this seed (centroid snapped to peak).
    void_guard: bool,
    composition: &'static str,
    black_fraction: f32,
    coverage: f32,
    occupancy: f64,
    palette: String,
    output: String,
}

/// A (seed × composition) crop that passed the black gate but failed occupancy.
/// Recorded for the survivors-vs-rejects distribution and the sanity sheet.
struct RejectRecord {
    seed_index: usize,
    composition: &'static str,
    coverage: f32,
    occupancy: f64,
    thumb: String, // relative path to the default-palette reject thumbnail
}

/// A seed's content-centered focus + whether the void guard fired.
struct FocusResult {
    cx: f64,
    cy: f64,
    /// The energy-weighted centroid landed in a sub-floor tile (a gap between
    /// blobs); focus was snapped to the peak-energy tile instead.
    void_guard: bool,
}

/// One seed's center-composition rendered at full AA + its occupancy + the
/// (reusable) sample buffer, default-palette gate image, and pixel spacing.
struct GateRender {
    buf: render::SampleBuffer,
    gate_img: RgbImage,
    occ: f64,
    pixel_spacing: f64,
}

/// A row of the seed-center vs content-center comparison sheet.
struct CompareRow {
    seed_index: usize,
    draw_index: usize,
    seed_occ: f64,
    content_occ: f64,
    /// content − seed-center occupancy (the recentering gain; sort key, desc).
    gain: f64,
    void_guard: bool,
    seed_thumb: String,    // path relative to run_dir
    content_thumb: String,
}

/// Save an `RgbImage` as JPEG at the given quality (the `image::save` default is
/// 75; we want q≈90), via the explicit encoder.
fn save_jpeg(img: &RgbImage, path: &Path, quality: u8) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, quality);
    enc.encode(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("encode jpeg {}: {e}", path.display()))
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
        draw_index: parse_usize(line, "draw_index")?,
        interior_frac: parse_f64(line, "interior_frac")?,
        cx: parse_f64(line, "center_re")?,
        cy: parse_f64(line, "center_im")?,
        frame_width: parse_f64(line, "frame_width")?,
    })
}

// ---------- manifest builder -------------------------------------------------

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    source: &str,
    zoom_factor: f64,
    maxiter: u32,
    occupancy_floor: f64,
    total_seeds: usize,
    accepted: usize,
    rejected_black: usize,
    rejected_occupancy: usize,
    crops: &[CropRecord],
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"source_jsonl\": \"{source}\",");
    let _ = writeln!(s, "  \"zoom_factor\": {zoom_factor},");
    let _ = writeln!(
        s,
        "  \"render\": \"grid ss4 + lanczos3, maxiter {maxiter} (render-one quality path)\","
    );
    let _ = writeln!(
        s,
        "  \"occupancy_gate\": {{ \"floor\": {occupancy_floor}, \"edge_floor\": {OCC_FLOOR}, \
         \"tile_grid\": [{OCC_GX}, {OCC_GY}] }},"
    );
    let _ = writeln!(s, "  \"total_seeds\": {total_seeds},");
    let _ = writeln!(s, "  \"accepted\": {accepted},");
    let _ = writeln!(s, "  \"rejected_black\": {rejected_black},");
    let _ = writeln!(s, "  \"rejected_occupancy\": {rejected_occupancy},");
    s.push_str("  \"crops\": [\n");
    for (i, c) in crops.iter().enumerate() {
        let comma = if i + 1 < crops.len() { "," } else { "" };
        let out_fwd = c.output.replace('\\', "/");
        let _ = writeln!(
            s,
            "    {{ \"draw_index\": {}, \"seed_index\": {}, \"cx\": {}, \"cy\": {}, \"fw\": {}, \
             \"focus_cx\": {}, \"focus_cy\": {}, \"void_guard\": {}, \
             \"composition\": \"{}\", \"interior_frac\": {:.6}, \"black_fraction\": {:.4}, \
             \"coverage\": {:.4}, \"occupancy\": {:.4}, \"palette\": \"{}\", \"output\": \"{out_fwd}\" }}{comma}",
            c.draw_index,
            c.seed_index,
            jnum(c.cx),
            jnum(c.cy),
            jnum(c.fw),
            jnum(c.focus_cx),
            jnum(c.focus_cy),
            c.void_guard,
            c.composition,
            c.interior_frac,
            c.black_fraction,
            c.coverage,
            c.occupancy,
            c.palette,
        );
    }
    s.push_str("  ]\n");
    s.push('}');
    s.push('\n');
    s
}

// ---------- focus + full-render helpers --------------------------------------

/// Render the `width×height` crop at full AA (grid-ss + Lanczos-3, the render-one
/// quality path), shade once with the default palette, and score detail
/// occupancy on it. Returns the reusable sample buffer alongside, so a caller
/// can apply the library palettes without re-iterating.
#[allow(clippy::too_many_arguments)]
fn render_and_gate(
    center_re: f64,
    center_im: f64,
    new_fw: f64,
    args: &PresentArgs,
    trap: Trap,
    channels: coloring::ChannelSet,
    gate_palette: &Palette,
    params: &coloring::ColorParams,
) -> GateRender {
    let frame = Frame {
        center: Complex::new(center_re, center_im),
        frame_width: new_fw,
        out_width: args.width,
        out_height: args.height,
    };
    let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
    let buf = render::iterate_samples_f64(&backend, &frame, args.ss, channels);
    let pixel_spacing = new_fw / args.width as f64;
    let gate_img = render::shade_and_downsample_filtered(
        &buf.samples,
        args.width,
        args.height,
        args.ss,
        gate_palette,
        params,
        pixel_spacing,
        FULL_FILTER,
    );
    let occ = energy::occupancy(&gate_img, OCC_GX, OCC_GY, OCC_FLOOR);
    GateRender { buf, gate_img, occ, pixel_spacing }
}

/// Content-centered focus: the energy-weighted centroid of a cheap edge-energy
/// screen over the **seed frame** (width `fw`, before the `zoom_factor`
/// retighten — that wider field is where the structure the tight crop misses
/// lives). The screen is default-palette shaded so `energy::tile_energy` is
/// defined on it, tiled exactly like the occupancy gate (`OCC_GX×OCC_GY`).
///
/// Void guard: if the centroid lands in a sub-`OCC_FLOOR` tile (a gap between two
/// blobs), snap focus to the peak-energy tile. The focus is then clamped so the
/// `fw × zoom_factor` crop stays fully inside the seed frame.
fn content_focus(
    seed: &Seed,
    args: &PresentArgs,
    trap: Trap,
    channels: coloring::ChannelSet,
    gate_palette: &Palette,
    params: &coloring::ColorParams,
) -> FocusResult {
    let fw = seed.frame_width;
    let frame = Frame {
        center: Complex::new(seed.cx, seed.cy),
        frame_width: fw,
        out_width: CHEAP_W,
        out_height: CHEAP_H,
    };
    let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
    let buf = render::iterate_samples_f64(&backend, &frame, CHEAP_SS, channels);
    let pixel_spacing = fw / CHEAP_W as f64;
    let img = render::shade_and_downsample_filtered(
        &buf.samples,
        CHEAP_W,
        CHEAP_H,
        CHEAP_SS,
        gate_palette,
        params,
        pixel_spacing,
        FULL_FILTER,
    );

    let (gx, gy) = (OCC_GX, OCC_GY);
    let tiles = energy::tile_energy(&img, gx, gy);

    // energy-weighted centroid in fractional [0,1]² frame coordinates.
    let mut wsum = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut peak = f64::NEG_INFINITY;
    let mut peak_i = 0usize;
    for ty in 0..gy {
        for tx in 0..gx {
            let i = ty * gx + tx;
            let w = tiles[i];
            sx += w * (tx as f64 + 0.5) / gx as f64;
            sy += w * (ty as f64 + 0.5) / gy as f64;
            wsum += w;
            if w > peak {
                peak = w;
                peak_i = i;
            }
        }
    }
    let (mut fpx, mut fpy) = if wsum > 0.0 { (sx / wsum, sy / wsum) } else { (0.5, 0.5) };

    // void guard: centroid's own tile below the occupancy edge floor → snap to peak.
    let ctx = ((fpx * gx as f64) as usize).min(gx - 1);
    let cty = ((fpy * gy as f64) as usize).min(gy - 1);
    let void_guard = tiles[cty * gx + ctx] < OCC_FLOOR;
    if void_guard {
        fpx = (peak_i % gx) as f64 + 0.5;
        fpx /= gx as f64;
        fpy = (peak_i / gx) as f64 + 0.5;
        fpy /= gy as f64;
    }

    // fractional → complex (row 0 = top = largest imaginary; matches render.rs).
    let fh = fw * CHEAP_H as f64 / CHEAP_W as f64;
    let mut cx = seed.cx + (fpx - 0.5) * fw;
    let mut cy = seed.cy + (0.5 - fpy) * fh;

    // clamp so the fw×zoom_factor crop stays fully inside the seed frame.
    let new_fw = fw * args.zoom_factor;
    let new_fh = new_fw * CHEAP_H as f64 / CHEAP_W as f64;
    let half_re = ((fw - new_fw) * 0.5).max(0.0);
    let half_im = ((fh - new_fh) * 0.5).max(0.0);
    cx = cx.clamp(seed.cx - half_re, seed.cx + half_re);
    cy = cy.clamp(seed.cy - half_im, seed.cy + half_im);

    FocusResult { cx, cy, void_guard }
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

    // 2. Load palette library through the SAME path as palette-score / palette-pick
    //    (carries the inline `mirror_needed` classification, so present's location
    //    crops adopt the selective-mirror construction the scored grids used).
    let cm_text = std::fs::read_to_string(&args.palette_file)
        .map_err(|e| format!("read {}: {e}", args.palette_file))?;
    let library: Vec<Colormap> = parse_colormaps(&cm_text)
        .map_err(|e| format!("parse {}: {e}", args.palette_file))?;
    if library.is_empty() {
        return Err(format!("no palettes found in {}", args.palette_file));
    }
    eprintln!("palette library: {} entries", library.len());

    let mut rng = SplitMix64(args.seed);
    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // 3. Output directory. `--flat-out` writes straight into `--out-dir`
    //    (avoids the doubled `<run>/<run>` nesting); otherwise append the run stem.
    let run_dir = if args.flat_out {
        Path::new(&args.out_dir).to_path_buf()
    } else {
        let run_stem = Path::new(&args.input)
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("run")
            .to_string();
        Path::new(&args.out_dir).join(&run_stem)
    };
    ensure_parent_dir(run_dir.join("x"))?;

    // Output format / extension.
    let as_jpeg = args.format.eq_ignore_ascii_case("jpg") || args.format.eq_ignore_ascii_case("jpeg");
    let ext = if as_jpeg { "jpg" } else { "png" };

    // Occupancy gate config. The gate runs on a default-palette full-AA render of
    // each crop (occupancy is palette-invariant to ~0.7%); we shade the actual
    // library palettes only for survivors. The reject thumbnails reuse that same
    // default-palette image, so a rejected crop never costs a library-palette shade.
    let gate_on = args.occupancy_floor > 0.0;
    let gate_palette = builtin("default", false).expect("default palette");
    let reject_dir = run_dir.join("rejects");
    if gate_on {
        ensure_parent_dir(reject_dir.join("x"))?;
    }

    // Focus mode + the side-by-side comparison sheet's thumbnail dir. Both
    // focuses are rendered for every seed regardless of the active mode.
    let content_focus_on = !args.focus.eq_ignore_ascii_case("seed-center");
    eprintln!(
        "focus: {} (emitted batch); both focuses rendered for focus_compare.html",
        if content_focus_on { "content" } else { "seed-center" }
    );
    let compare_dir = run_dir.join("compare");
    ensure_parent_dir(compare_dir.join("x"))?;

    // Which compositions to try
    let try_comps: &[(&str, f64, f64)] = match args.compositions.as_str() {
        "center" => &COMP_OFFSETS[0..1],
        "thirds" => &COMP_OFFSETS[1..2],
        "golden" => &COMP_OFFSETS[2..3],
        _ => COMP_OFFSETS,
    };

    let mut accepted = 0usize;
    let mut rejected_black = 0usize;
    let mut rejected_occupancy = 0usize;
    let mut black_passed = 0usize;
    let mut crops: Vec<CropRecord> = Vec::new();
    let mut rejects: Vec<RejectRecord> = Vec::new();
    let mut sheet_tiles: Vec<RgbImage> = Vec::new();
    let mut full_renders = 0usize;
    let mut void_guard_fires = 0usize;
    let mut compare_rows: Vec<CompareRow> = Vec::new();
    let mut timing_printed = false;
    let t_start = std::time::Instant::now();

    for seed in &seeds {
        let new_fw = seed.frame_width * args.zoom_factor;

        // --- both focuses: content-centered (energy centroid over the seed
        //     frame) and seed-center. ---
        let content = content_focus(seed, args, trap, channels, &gate_palette, &params);
        if content.void_guard {
            void_guard_fires += 1;
        }

        // Comparison sheet: the CENTER composition rendered both ways at full AA,
        // occupancy scored on each. The active-focus center render is reused by
        // the main batch below (no third render of the same frame).
        let seed_center =
            render_and_gate(seed.cx, seed.cy, new_fw, args, trap, channels, &gate_palette, &params);
        let content_center = render_and_gate(
            content.cx, content.cy, new_fw, args, trap, channels, &gate_palette, &params,
        );
        full_renders += 2;
        let rh = (REJECT_THUMB_W as f64 * args.height as f64 / args.width as f64).round() as u32;
        let st = image::imageops::resize(&seed_center.gate_img, REJECT_THUMB_W, rh, FilterType::Triangle);
        let ct = image::imageops::resize(&content_center.gate_img, REJECT_THUMB_W, rh, FilterType::Triangle);
        let sname = format!("{}_seed.jpg", seed.keeper_index);
        let cname = format!("{}_content.jpg", seed.keeper_index);
        save_jpeg(&st, &compare_dir.join(&sname), 85)?;
        save_jpeg(&ct, &compare_dir.join(&cname), 85)?;
        compare_rows.push(CompareRow {
            seed_index: seed.keeper_index,
            draw_index: seed.draw_index,
            seed_occ: seed_center.occ,
            content_occ: content_center.occ,
            gain: content_center.occ - seed_center.occ,
            void_guard: content.void_guard,
            seed_thumb: format!("compare/{sname}"),
            content_thumb: format!("compare/{cname}"),
        });

        if !timing_printed {
            let dt = t_start.elapsed().as_secs_f64();
            let est = dt / 2.0 * (seeds.len() * 3) as f64; // ~3 full renders/seed avg
            eprintln!(
                "  [first seed: focus screen + 2 compare renders in {dt:.1}s — rough whole-run estimate ~{:.0}s]",
                est
            );
            timing_printed = true;
        }

        // --- active focus drives the emitted batch ---
        let (focus_cx, focus_cy, active_center): (f64, f64, &GateRender) = if content_focus_on {
            (content.cx, content.cy, &content_center)
        } else {
            (seed.cx, seed.cy, &seed_center)
        };

        // --- cheap render per composition (black fraction for the gate) ---
        let mut comp_bf: Vec<(&'static str, f64, f64, f32)> = Vec::new(); // (name, cx, cy, bf)
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
            comp_bf.push((comp_name, ccx, ccy, bf));
        }

        // Which composition crops to emit: in all-compositions mode, every one
        // that clears the black gate becomes a distinct label unit; otherwise the
        // single lowest-black composition (legacy pick-best).
        let chosen: Vec<(&'static str, f64, f64, f32)> = if args.all_compositions {
            comp_bf.iter().copied().filter(|c| c.3 < BLACK_THRESH).collect()
        } else {
            comp_bf
                .iter()
                .copied()
                .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
                .filter(|c| c.3 < BLACK_THRESH)
                .into_iter()
                .collect()
        };

        // Reject accounting is per composition-crop considered (= label units).
        let considered = if args.all_compositions { comp_bf.len() } else { 1 };
        rejected_black += considered - chosen.len();

        for &(comp_name, comp_cx, comp_cy, bf) in &chosen {
            let coverage = 1.0 - bf;
            black_passed += 1;

            // --- full-res render (iterate once at full AA: grid ss + Lanczos-3,
            //     the render-one quality path; recolor per palette is free).
            //     The center composition reuses the active-focus render already
            //     done for the compare sheet (offset (0,0) → same frame). ---
            let owned;
            let gr: &GateRender = if comp_name == "center" {
                active_center
            } else {
                owned = render_and_gate(
                    comp_cx, comp_cy, new_fw, args, trap, channels, &gate_palette, &params,
                );
                full_renders += 1;
                &owned
            };
            let buf = &gr.buf;
            let pixel_spacing = gr.pixel_spacing;
            let gate_img = &gr.gate_img;
            let occ = gr.occ;

            if gate_on && occ < args.occupancy_floor {
                rejected_occupancy += 1;
                eprintln!(
                    "  seed {:04} draw {:04} REJECT-occ comp={} occ={:.3} (<{:.3}) coverage={:.3}",
                    seed.keeper_index, seed.draw_index, comp_name, occ, args.occupancy_floor, coverage
                );
                // Small default-palette thumbnail so the sanity sheet shows the
                // gate's effect at the margin (no library-palette render spent).
                let rh = (REJECT_THUMB_W as f64 * args.height as f64 / args.width as f64).round() as u32;
                let rthumb = image::imageops::resize(gate_img, REJECT_THUMB_W, rh, FilterType::Triangle);
                let rname = format!("{}_{}.jpg", seed.keeper_index, comp_name);
                let rpath = reject_dir.join(&rname);
                save_jpeg(&rthumb, &rpath, 85)?;
                rejects.push(RejectRecord {
                    seed_index: seed.keeper_index,
                    composition: comp_name,
                    coverage,
                    occupancy: occ,
                    thumb: format!("rejects/{rname}"),
                });
                continue;
            }

            eprintln!(
                "  seed {:04} draw {:04} ACCEPTED  comp={} black_frac={:.3} coverage={:.3} occ={:.3}",
                seed.keeper_index, seed.draw_index, comp_name, bf, coverage, occ
            );
            accepted += 1;

            for _ in 0..args.palettes_per_crop {
                let pi = rng.below(library.len());
                let cm = &library[pi];
                let pal_name = &cm.name;
                // Selective seam fix (parity with palette-score): SEQUENTIAL
                // (`mirror_needed`) maps bake pre-mirrored out-and-back; cyclic maps
                // stay single-pass. Density compensation rides on the palette's
                // `density_scale` and is applied in shade.
                let palette = Palette::from_srgb8_stops_mirrored(
                    pal_name.clone(),
                    &cm.stops,
                    false,
                    cm.mirror_needed,
                );
                let img = render::shade_and_downsample_filtered(
                    &buf.samples,
                    args.width,
                    args.height,
                    args.ss,
                    &palette,
                    &params,
                    pixel_spacing,
                    FULL_FILTER,
                );

                let safe_name =
                    pal_name.replace(['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|'], "_");
                // Composition in the filename so (seed × composition × palette)
                // crops never collide in all-compositions mode.
                // NOTE: the leading numeric prefix is the **seed_index**
                // (`keeper_index`), NOT the `draw_index` — e.g. `93_center_*.jpg`
                // is seed 93, whose draw_index may be unrelated (see manifest).
                let fname = format!("{}_{}_{}.{ext}", seed.keeper_index, comp_name, safe_name);
                let out_path = run_dir.join(&fname);
                if as_jpeg {
                    save_jpeg(&img, &out_path, args.jpg_quality)?;
                } else {
                    img.save(&out_path)
                        .map_err(|e| format!("save {}: {e}", out_path.display()))?;
                }

                let out_str = out_path.to_string_lossy().into_owned();

                let thumb = image::imageops::resize(
                    &img,
                    SHEET_THUMB_W,
                    SHEET_THUMB_H,
                    FilterType::Triangle,
                );
                sheet_tiles.push(thumb);

                crops.push(CropRecord {
                    seed_index: seed.keeper_index,
                    draw_index: seed.draw_index,
                    interior_frac: seed.interior_frac,
                    cx: comp_cx,
                    cy: comp_cy,
                    fw: new_fw,
                    focus_cx,
                    focus_cy,
                    void_guard: content.void_guard,
                    composition: comp_name,
                    black_fraction: bf,
                    coverage,
                    occupancy: occ,
                    palette: pal_name.clone(),
                    output: out_str,
                });
            }
        }
    }

    // --- contact sheet (PNG grid, survivors) ---
    if !sheet_tiles.is_empty() {
        let grid = sheet::compose_grid(&sheet_tiles, Some(8));
        let sheet_path = run_dir.join("present_sheet.png");
        grid.save(&sheet_path).map_err(|e| format!("save sheet: {e}"))?;
        eprintln!("sheet: {}", sheet_path.display());
    }

    // --- sanity sheet (HTML, occupancy-sorted survivors + margin rejects) ---
    let html = build_sanity_html(&crops, &rejects, &run_dir, args.occupancy_floor);
    let html_path = run_dir.join("sanity_sheet.html");
    std::fs::write(&html_path, html).map_err(|e| format!("write sanity sheet: {e}"))?;

    // --- manifest ---
    let manifest = build_manifest(
        &args.input,
        args.zoom_factor,
        args.maxiter,
        args.occupancy_floor,
        seeds.len(),
        accepted,
        rejected_black,
        rejected_occupancy,
        &crops,
    );
    let manifest_path = run_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest)
        .map_err(|e| format!("write manifest: {e}"))?;

    // --- funnel / distribution report ---
    let distinct_survivors: std::collections::BTreeSet<(usize, &str)> =
        crops.iter().map(|c| (c.seed_index, c.composition)).collect();
    let surviving_seeds: std::collections::BTreeSet<usize> =
        crops.iter().map(|c| c.seed_index).collect();
    let mut surv_occ: Vec<f64> = distinct_survivors
        .iter()
        .filter_map(|key| {
            crops.iter().find(|c| (c.seed_index, c.composition) == *key).map(|c| c.occupancy)
        })
        .collect();
    let mut rej_occ: Vec<f64> = rejects.iter().map(|r| r.occupancy).collect();
    surv_occ.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rej_occ.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("=== present (occupancy-gated) ===");
    println!("elapsed: {:.1}s  full renders: {full_renders}", t_start.elapsed().as_secs_f64());
    println!(
        "funnel: {} seeds × {} comps = {} comp-crops → {} black-passed → {} occupancy-passed",
        seeds.len(),
        try_comps.len(),
        seeds.len() * try_comps.len(),
        black_passed,
        accepted,
    );
    println!(
        "  rejected_black={rejected_black}  rejected_occupancy={rejected_occupancy} (floor {:.3})",
        args.occupancy_floor
    );
    println!(
        "  → label rows={} (×{} palettes)  distinct crops={}  seeds surviving ≥1 comp={}",
        crops.len(),
        args.palettes_per_crop,
        distinct_survivors.len(),
        surviving_seeds.len(),
    );
    print_occ_summary("survivors", &surv_occ);
    print_occ_summary("rejects  ", &rej_occ);

    // --- focus comparison sheet (seed-center vs content-center, gain-sorted) ---
    let compare_html = build_compare_html(&compare_rows, args.focus.as_str());
    let compare_path = run_dir.join("focus_compare.html");
    std::fs::write(&compare_path, compare_html)
        .map_err(|e| format!("write focus compare sheet: {e}"))?;

    // --- focus report: void guard + occupancy-gain distribution over all seeds ---
    let mut gains: Vec<f64> = compare_rows.iter().map(|r| r.gain).collect();
    gains.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let qg = |p: f64| -> f64 {
        if gains.is_empty() {
            return f64::NAN;
        }
        let i = ((p * (gains.len() - 1) as f64).round() as usize).min(gains.len() - 1);
        gains[i]
    };
    let regressed = compare_rows.iter().filter(|r| r.gain < 0.0).count();
    println!(
        "focus: content-centered for {} seeds; void-guard fired {void_guard_fires}× (centroid snapped to peak)",
        compare_rows.len()
    );
    println!(
        "  occupancy gain (content − seed-center) over {} seeds: p10 {:+.3}  median {:+.3}  p90 {:+.3}  | regressed (gain<0): {regressed}",
        gains.len(),
        qg(0.10),
        qg(0.50),
        qg(0.90),
    );
    println!("manifest: {}", manifest_path.display());
    println!("sheet:    {}", html_path.display());
    println!("compare:  {}", compare_path.display());
    Ok(())
}

/// Seed-center vs content-center comparison sheet: one row per seed, the center
/// composition rendered both ways (left = seed-center, right = content-center),
/// each captioned with its occupancy. Sorted by occupancy **gain** descending so
/// the biggest wins float to the top and regressions / void-guard fires sink to
/// the bottom for fast spotting.
fn build_compare_html(rows: &[CompareRow], active: &str) -> String {
    let mut sorted: Vec<&CompareRow> = rows.iter().collect();
    sorted.sort_by(|a, b| b.gain.partial_cmp(&a.gain).unwrap_or(std::cmp::Ordering::Equal));

    let mut cells = String::new();
    for r in &sorted {
        let cls = if r.void_guard { "row void" } else { "row" };
        let gcol = if r.gain >= 0.0 { "pos" } else { "neg" };
        let _ = write!(
            cells,
            "<div class=\"{cls}\">\
             <div class=meta><b>s{}</b> draw{} <span class=\"g {gcol}\">{:+.3}</span>{}</div>\
             <div class=pair>\
             <figure><img loading=lazy src=\"{}\"><figcaption>seed-center {:.3}</figcaption></figure>\
             <figure><img loading=lazy src=\"{}\"><figcaption>content {:.3}</figcaption></figure>\
             </div></div>",
            r.seed_index,
            r.draw_index,
            r.gain,
            if r.void_guard { " <span class=vg>VOID-GUARD</span>" } else { "" },
            r.seed_thumb.replace('\\', "/"),
            r.seed_occ,
            r.content_thumb.replace('\\', "/"),
            r.content_occ,
        );
    }

    format!(
        "<!doctype html><html><head><meta charset=utf-8><title>loose0_v3 focus compare</title>\
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}\
body{{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;margin:0}}\
header{{position:sticky;top:0;background:#12141a;border-bottom:1px solid #23252e;padding:10px 18px;z-index:5}}\
h1{{font-size:15px;margin:0 0 4px;color:#eee}}.note{{color:#888;font-size:12px}}\
.row{{border-bottom:1px solid #1c1f29;padding:8px 14px}}\
.row.void{{background:#1a1410}}\
.meta{{margin:0 0 4px;color:#9aa}}\
.g{{font-weight:bold;padding:0 5px;border-radius:3px}}.g.pos{{color:#5ec07a}}.g.neg{{color:#e06a6a}}\
.vg{{color:#e0b24a;font-weight:bold}}\
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:1100px}}\
figure{{margin:0;border:1px solid #23252e;border-radius:5px;overflow:hidden;background:#000}}\
figure img{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}}\
figcaption{{padding:2px 6px;font-size:10.5px;color:#9aa;background:#12141a;border-top:1px solid #1c1f29}}\
</style></head><body>\
<header><h1>loose0_v3 — focus comparison (seed-center vs content-center)</h1>\
<div class=note>{n} seeds · center composition both ways · sorted by occupancy gain (content − seed-center) desc · \
emitted batch uses <b>{active}</b> focus · gold rows = void-guard fired</div></header>\
{cells}</body></html>",
        n = sorted.len(),
        active = active,
        cells = cells,
    )
}

/// Print min / p25 / median / p75 / max of a sorted occupancy vector.
fn print_occ_summary(label: &str, sorted: &[f64]) {
    if sorted.is_empty() {
        println!("  occupancy {label}: (none)");
        return;
    }
    let q = |p: f64| {
        let i = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[i]
    };
    println!(
        "  occupancy {label} (n={}): min {:.3}  p25 {:.3}  med {:.3}  p75 {:.3}  max {:.3}",
        sorted.len(),
        sorted[0],
        q(0.25),
        q(0.50),
        q(0.75),
        sorted[sorted.len() - 1],
    );
}

/// Self-contained sanity sheet: survivors (occupancy-ascending) followed by the
/// margin rejects, captioned with occupancy so the gate's effect is visible.
/// Image data is referenced by path relative to the sheet (sibling JPG/PNGs).
fn build_sanity_html(
    crops: &[CropRecord],
    rejects: &[RejectRecord],
    run_dir: &Path,
    floor: f64,
) -> String {
    // One tile per (seed × composition) survivor — use the first palette as the
    // representative image (occupancy is per-crop, not per-palette).
    let mut seen: std::collections::BTreeSet<(usize, &str)> = std::collections::BTreeSet::new();
    let mut surv: Vec<&CropRecord> = Vec::new();
    for c in crops {
        if seen.insert((c.seed_index, c.composition)) {
            surv.push(c);
        }
    }
    surv.sort_by(|a, b| a.occupancy.partial_cmp(&b.occupancy).unwrap());
    let mut rej: Vec<&RejectRecord> = rejects.iter().collect();
    rej.sort_by(|a, b| b.occupancy.partial_cmp(&a.occupancy).unwrap()); // closest-to-floor first

    let rel = |abs: &str| -> String {
        let p = Path::new(abs);
        p.strip_prefix(run_dir)
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| {
                p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
            })
    };

    let mut cell = String::new();
    for c in &surv {
        let _ = write!(
            cell,
            "<div class=cell><img loading=lazy src=\"{}\"><div class=sc>{:.3}</div>\
             <div class=cap>s{} {} cov{:.2}</div></div>",
            rel(&c.output),
            c.occupancy,
            c.seed_index,
            c.composition,
            c.coverage,
        );
    }
    let mut rcell = String::new();
    for r in &rej {
        let _ = write!(
            rcell,
            "<div class=\"cell reject\"><img loading=lazy src=\"{}\"><div class=sc>{:.3}</div>\
             <div class=cap>s{} {} cov{:.2}</div></div>",
            r.thumb.replace('\\', "/"),
            r.occupancy,
            r.seed_index,
            r.composition,
            r.coverage,
        );
    }

    format!(
        "<!doctype html><html><head><meta charset=utf-8><title>loose0_v2 sanity sheet</title>\
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}\
body{{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;margin:0}}\
header{{position:sticky;top:0;background:#12141a;border-bottom:1px solid #23252e;padding:10px 18px;z-index:5}}\
h1{{font-size:15px;margin:0 0 4px;color:#eee}}h2{{font-size:13px;color:#e0b24a;margin:20px 14px 6px}}\
.note{{color:#888;font-size:12px}}\
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;padding:14px}}\
.cell{{position:relative;border:1px solid #23252e;border-radius:5px;overflow:hidden;background:#000}}\
.cell img{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}}\
.cell .cap{{padding:3px 6px;font-size:10.5px;color:#9aa;background:#12141a;border-top:1px solid #1c1f29}}\
.cell .sc{{position:absolute;left:0;top:0;font-size:12px;font-weight:bold;padding:1px 6px;background:rgba(0,0,0,.7);color:#5ec07a;border-bottom-right-radius:4px}}\
.cell.reject{{opacity:.5}}.cell.reject .sc{{color:#e06a6a}}\
.cell.reject::after{{content:'REJECT';position:absolute;right:0;top:0;font-size:9px;font-weight:bold;color:#e06a6a;background:rgba(0,0,0,.7);padding:1px 5px;border-bottom-left-radius:4px}}\
</style></head><body>\
<header><h1>loose0_v2 — occupancy gate sanity sheet</h1>\
<div class=note>occupancy floor <b>{floor:.3}</b> · {ns} survivors (green, sorted boring→busy) · {nr} margin rejects (red, sorted near→far below floor) · tiles are the first palette per (seed×comp)</div></header>\
<h2>survivors ({ns}) — occupancy ascending; the lowest are at the gate margin</h2>\
<div class=grid>{cell}</div>\
<h2>rejects ({nr}) — default-palette gate thumbnails, occupancy below floor</h2>\
<div class=grid>{rcell}</div>\
</body></html>",
        floor = floor,
        ns = surv.len(),
        nr = rej.len(),
        cell = cell,
        rcell = rcell,
    )
}
