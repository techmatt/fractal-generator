//! `palette-probe` subcommand — palette universality probe.
//!
//! Renders a handful of **label-3 ("great") locations** across the **full
//! score-3 palette pool**, holding location + framing fixed and varying only the
//! palette, so universally-dead palettes (bad even on a known-good structure) can
//! be spotted and cut. This is **not** a prune-to-pretty pass: mediocre palettes
//! stay as negatives for the eventual joint structure+palette classifier; only
//! the universally-bad are candidates.
//!
//! It is an **extract** of the established render path, not new logic:
//! - **Selection** joins `location_labels.json` (the `seed|comp|palette` → label
//!   key, here `draw_index|comp|palette`) with the `present` `loose0_v3`
//!   `manifest.json` to recover each label-3 crop's exact geometry (`cx/cy/fw`),
//!   collapses to distinct locations, and picks `--n-locations` at random with a
//!   fixed [`SplitMix64`] seed (logged for reproducibility).
//! - **Render** reuses `present`/`render-one`'s f64 quality path: iterate **once**
//!   per location at grid-ss4 + the channel-intent buffer
//!   ([`render::iterate_samples_f64`]), then **recolor** across all 76 palettes
//!   with [`render::shade_and_downsample_filtered`] (Lanczos-3) — no re-iteration.
//! - **Palettes** load through [`palette_pick::parse_colormaps`] +
//!   [`Palette::from_srgb8_stops_mirrored`], the same selective-mirror path as
//!   `present`, so sequential maps render seam-free.
//!
//! Per-palette **corpus not-bad rate** (fraction of that palette's corpus labels
//! that are not label-1) is computed from the labels and emitted with the index,
//! so the viewer (`tools/viz/palette_probe.html`) can sort palettes worst-first.
//! Outputs land under `data/palette_probe/` (a stable store, **not** `out/`): the
//! JPG crops + `probe_index.json`. The viewer writes the decision artifact
//! (`palette_verdict.json`); this subcommand makes **no** quality claim and cuts
//! nothing.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::PaletteProbeArgs;
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::probe::SplitMix64;
use crate::render::{self, DownsampleFilter, Frame};
use crate::ensure_parent_dir;

/// Escape radius (matches the generate/present/render-one regime).
const BAILOUT: f64 = 1e6;
/// Locked wallpaper-quality reconstruction filter (matches `render-one`/`present`).
const FULL_FILTER: DownsampleFilter = DownsampleFilter::Lanczos3;

/// One chosen location: a label-3 crop's geometry, carried verbatim from the
/// `present` manifest (the framing is already proven great).
struct Location {
    /// Standardized label key first field (= `present` `draw_index`).
    draw_index: usize,
    seed_index: i64,
    cx: f64,
    cy: f64,
    fw: f64,
    composition: String,
    focus_cx: f64,
    focus_cy: f64,
    /// The label-3 palette whose crop's geometry was adopted (provenance).
    label3_palette: String,
}

/// Sanitize a palette name for use in a filename (parity with `present`).
fn safe_name(name: &str) -> String {
    name.replace(['/', '\\', ' ', ':', '*', '?', '"', '<', '>', '|'], "_")
}

// JPEG crop writer shared via `crate::render::save_jpeg`.
use crate::render::save_jpeg;

// ---------- labels parsing ---------------------------------------------------

/// Parse `location_labels.json`: a single flat object `{"draw|comp|palette": N, …}`.
/// Returns the key→label map. Keys carry no `:` or `,` (composition + palette
/// names are bare), so a comma/colon split of the inner object is unambiguous.
fn parse_labels(text: &str) -> Result<BTreeMap<String, u8>, String> {
    let s = text.trim();
    let s = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or("labels: expected a top-level JSON object")?;
    let mut out = BTreeMap::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let colon = tok.rfind(':').ok_or_else(|| format!("labels: no ':' in {tok}"))?;
        let key = tok[..colon].trim().trim_matches('"');
        let val: u8 = tok[colon + 1..]
            .trim()
            .parse()
            .map_err(|e| format!("labels: bad value in {tok}: {e}"))?;
        out.insert(key.to_string(), val);
    }
    Ok(out)
}

// ---------- manifest parsing (hand-rolled NDJSON, parity with present) --------
// Field readers shared via `crate::jsonl` (the canonical copy).
use crate::jsonl::*;

/// One manifest crop's geometry, indexed by its `(draw_index, comp, palette)` key.
struct Crop {
    seed_index: i64,
    cx: f64,
    cy: f64,
    fw: f64,
    focus_cx: f64,
    focus_cy: f64,
}

/// Index the `present` manifest by the label composite key (each crop object is
/// one line). Only the fields the probe needs are parsed.
fn parse_manifest(text: &str) -> BTreeMap<String, Crop> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        if !line.contains("\"draw_index\": ") {
            continue;
        }
        let (Some(di), Some(comp), Some(pal)) = (
            field_usize(line, "draw_index"),
            field_str(line, "composition"),
            field_str(line, "palette"),
        ) else {
            continue;
        };
        let (Some(cx), Some(cy), Some(fw)) =
            (field_f64(line, "cx"), field_f64(line, "cy"), field_f64(line, "fw"))
        else {
            continue;
        };
        let crop = Crop {
            seed_index: field_f64(line, "seed_index").map(|v| v as i64).unwrap_or(-1),
            cx,
            cy,
            fw,
            focus_cx: field_f64(line, "focus_cx").unwrap_or(cx),
            focus_cy: field_f64(line, "focus_cy").unwrap_or(cy),
        };
        out.insert(format!("{di}|{comp}|{pal}"), crop);
    }
    out
}

// ---------- entry point ------------------------------------------------------

pub fn run_palette_probe(args: &PaletteProbeArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.supersample.max(1);

    // --- load labels + manifest, join on the composite key, filter label==3 ---
    let labels_text =
        std::fs::read_to_string(&args.labels).map_err(|e| format!("read {}: {e}", args.labels))?;
    let labels = parse_labels(&labels_text)?;
    let manifest_text = std::fs::read_to_string(&args.manifest)
        .map_err(|e| format!("read {}: {e}", args.manifest))?;
    let manifest = parse_manifest(&manifest_text);

    // Distinct locations (by draw_index) that have ≥1 label-3 crop joinable to the
    // manifest, with the label-3 crop keys sorted for deterministic geometry pick.
    let mut by_loc: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut joined_l3 = 0usize;
    let mut missing = 0usize;
    for (key, &lab) in &labels {
        if lab != 3 {
            continue;
        }
        let mut parts = key.splitn(3, '|');
        let (Some(di), Some(_comp), Some(_pal)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let Ok(di) = di.parse::<usize>() else { continue };
        if manifest.contains_key(key) {
            joined_l3 += 1;
            by_loc.entry(di).or_default().push(key.clone());
        } else {
            missing += 1;
        }
    }
    for v in by_loc.values_mut() {
        v.sort();
    }
    let mut loc_keys: Vec<usize> = by_loc.keys().copied().collect();
    loc_keys.sort_unstable();
    let distinct = loc_keys.len();
    eprintln!(
        "palette-probe: {joined_l3} label-3 crops joined ({missing} unjoinable) → {distinct} distinct locations"
    );

    // --- pick N distinct locations at random with a fixed seed ---
    // Fisher–Yates over the sorted distinct-location list; take the first N.
    let mut rng = SplitMix64(args.seed);
    for i in (1..loc_keys.len()).rev() {
        let j = rng.below(i + 1);
        loc_keys.swap(i, j);
    }
    let want = args.n_locations.min(distinct);
    if want < args.n_locations {
        eprintln!(
            "  only {distinct} distinct label-3 locations exist (< {} requested) — using all",
            args.n_locations
        );
    }
    let mut chosen: Vec<Location> = Vec::new();
    for &di in loc_keys.iter().take(want) {
        let l3_key = by_loc[&di][0].clone(); // first label-3 crop (sorted) → geometry
        let crop = &manifest[&l3_key];
        let comp = l3_key.splitn(3, '|').nth(1).unwrap_or("center").to_string();
        let pal = l3_key.splitn(3, '|').nth(2).unwrap_or("").to_string();
        chosen.push(Location {
            draw_index: di,
            seed_index: crop.seed_index,
            cx: crop.cx,
            cy: crop.cy,
            fw: crop.fw,
            composition: comp,
            focus_cx: crop.focus_cx,
            focus_cy: crop.focus_cy,
            label3_palette: pal,
        });
    }
    // Log the chosen set (geometry + provenance).
    eprintln!("  chosen {} locations (seed {}):", chosen.len(), args.seed);
    for l in &chosen {
        eprintln!(
            "    draw {:>4} seed {:>4} {:>6} cx={:.12} cy={:.12} fw={:.6e} focus=({:.9},{:.9}) [l3:{}]",
            l.draw_index, l.seed_index, l.composition, l.cx, l.cy, l.fw, l.focus_cx, l.focus_cy,
            l.label3_palette
        );
    }

    // --- palette pool (selective-mirror load, parity with present) ---
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text)
        .map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    if library.is_empty() {
        return Err(format!("no palettes in {}", args.colormaps));
    }
    eprintln!("palette pool: {} entries", library.len());

    // --- corpus not-bad rate per palette (from ALL labels) ---
    // not-bad = label != 1; rate = not-bad / total occurrences of that palette.
    let mut corpus_total: BTreeMap<&str, u32> = BTreeMap::new();
    let mut corpus_notbad: BTreeMap<&str, u32> = BTreeMap::new();
    for (key, &lab) in &labels {
        if let Some(pal) = key.splitn(3, '|').nth(2) {
            *corpus_total.entry(pal).or_insert(0) += 1;
            if lab != 1 {
                *corpus_notbad.entry(pal).or_insert(0) += 1;
            }
        }
    }

    // --- output dir ---
    let out_dir = Path::new(&args.out_dir);
    ensure_parent_dir(out_dir.join("x"))?;

    let params = color_params();
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // --- render: iterate once per location, recolor across the whole pool ---
    let t_start = Instant::now();
    let mut total_jpgs = 0usize;
    for (li, loc) in chosen.iter().enumerate() {
        let frame = Frame {
            center: Complex::new(loc.cx, loc.cy),
            frame_width: loc.fw,
            out_width: args.width,
            out_height: args.height,
        };
        let pixel_spacing = loc.fw / args.width as f64;
        if pixel_spacing <= 1e-13 {
            return Err(format!(
                "location draw {} pixel spacing {pixel_spacing:.3e} is inside f64's quantization \
                 regime — palette-probe is the shallow f64 path",
                loc.draw_index
            ));
        }

        let t_iter = Instant::now();
        let (buf, _) = render::iterate_crop_buffer_f64(&frame, ss, args.maxiter, BAILOUT, trap, &params);
        let iter_secs = t_iter.elapsed().as_secs_f64();

        let t_color = Instant::now();
        for cm in &library {
            let palette = Palette::from_srgb8_stops_mirrored(
                cm.name.clone(),
                &cm.stops,
                false,
                cm.mirror_needed,
            );
            let img = render::shade_and_downsample_filtered(
                &buf.samples,
                args.width,
                args.height,
                ss,
                &palette,
                &params,
                pixel_spacing,
                FULL_FILTER,
            );
            let fname = format!("{}_{}.jpg", loc.draw_index, safe_name(&cm.name));
            save_jpeg(&img, &out_dir.join(&fname), args.jpg_quality)?;
            total_jpgs += 1;
        }
        let color_secs = t_color.elapsed().as_secs_f64();
        eprintln!(
            "  [{}/{}] draw {:>4}: iterate {iter_secs:.2}s + recolor×{} {color_secs:.2}s",
            li + 1,
            chosen.len(),
            loc.draw_index,
            library.len()
        );
    }
    let elapsed = t_start.elapsed().as_secs_f64();

    // --- probe_index.json (locations + palettes sorted worst-first) ---
    let mut pal_rows: Vec<(usize, u32, f64)> = Vec::with_capacity(library.len());
    for (i, cm) in library.iter().enumerate() {
        let n = corpus_total.get(cm.name.as_str()).copied().unwrap_or(0);
        let nb = corpus_notbad.get(cm.name.as_str()).copied().unwrap_or(0);
        let rate = if n > 0 { nb as f64 / n as f64 } else { f64::NAN };
        pal_rows.push((i, n, rate));
    }
    // Worst-first: ascending not-bad rate; among ties, more corpus evidence first.
    // Palettes with no corpus labels (NaN) sink to the bottom (no evidence of bad).
    pal_rows.sort_by(|a, b| {
        let ka = if a.2.is_nan() { 2.0 } else { a.2 };
        let kb = if b.2.is_nan() { 2.0 } else { b.2 };
        ka.partial_cmp(&kb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
    });

    let index = build_index_json(args, &chosen, &library, &pal_rows, &corpus_total, &corpus_notbad);
    let index_path = out_dir.join("probe_index.json");
    std::fs::write(&index_path, index).map_err(|e| format!("write index: {e}"))?;

    // --- report ---
    println!("=== palette-probe ===");
    println!(
        "locations: {}  palettes: {}  crops: {total_jpgs}  ({:.1}s)",
        chosen.len(),
        library.len(),
        elapsed
    );
    println!("out:   {}", out_dir.display());
    println!("index: {}", index_path.display());
    println!("viewer: tools/viz/palette_probe.html");
    // Prime deletion candidates = lowest corpus not-bad rate (with evidence).
    println!("prime deletion candidates (lowest corpus not-bad rate):");
    for &(i, n, rate) in pal_rows.iter().take(12) {
        if rate.is_nan() {
            continue;
        }
        let nb = corpus_notbad.get(library[i].name.as_str()).copied().unwrap_or(0);
        println!("  {:>5.0}% ({nb}/{n})  {}", rate * 100.0, library[i].name);
    }
    Ok(())
}

/// Hand-rolled `probe_index.json` (no serde, per project convention). Carries the
/// render config, the chosen locations (geometry + provenance), and the palette
/// list already sorted worst-first with each palette's corpus not-bad rate.
fn build_index_json(
    args: &PaletteProbeArgs,
    chosen: &[Location],
    library: &[crate::palette_pick::Colormap],
    pal_rows: &[(usize, u32, f64)],
    corpus_total: &BTreeMap<&str, u32>,
    corpus_notbad: &BTreeMap<&str, u32>,
) -> String {
    let jnum = |x: f64| -> String {
        if x.is_finite() {
            format!("{x}")
        } else {
            "null".into()
        }
    };
    let mut s = String::new();
    s.push_str("{\n");
    let _ = writeln!(s, "  \"seed\": {},", args.seed);
    let _ = writeln!(
        s,
        "  \"render\": {{ \"width\": {}, \"height\": {}, \"ss\": {}, \"maxiter\": {}, \"filter\": \"lanczos3\", \"jpg_quality\": {} }},",
        args.width, args.height, args.supersample, args.maxiter, args.jpg_quality
    );
    let _ = writeln!(s, "  \"labels\": \"{}\",", args.labels.replace('\\', "/"));
    let _ = writeln!(s, "  \"manifest\": \"{}\",", args.manifest.replace('\\', "/"));
    let _ = writeln!(s, "  \"colormaps\": \"{}\",", args.colormaps.replace('\\', "/"));

    // locations
    s.push_str("  \"locations\": [\n");
    for (i, l) in chosen.iter().enumerate() {
        let comma = if i + 1 < chosen.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{ \"tag\": \"{}\", \"draw_index\": {}, \"seed_index\": {}, \"composition\": \"{}\", \
             \"cx\": {}, \"cy\": {}, \"fw\": {}, \"focus_cx\": {}, \"focus_cy\": {}, \
             \"label3_palette\": \"{}\" }}{comma}",
            l.draw_index,
            l.draw_index,
            l.seed_index,
            l.composition,
            jnum(l.cx),
            jnum(l.cy),
            jnum(l.fw),
            jnum(l.focus_cx),
            jnum(l.focus_cy),
            l.label3_palette,
        );
    }
    s.push_str("  ],\n");

    // palettes (worst-first)
    s.push_str("  \"palettes\": [\n");
    for (row_i, &(i, n, rate)) in pal_rows.iter().enumerate() {
        let comma = if row_i + 1 < pal_rows.len() { "," } else { "" };
        let cm = &library[i];
        let nb = corpus_notbad.get(cm.name.as_str()).copied().unwrap_or(0);
        let tot = corpus_total.get(cm.name.as_str()).copied().unwrap_or(0);
        let _ = writeln!(
            s,
            "    {{ \"name\": \"{}\", \"safe\": \"{}\", \"mirror_needed\": {}, \
             \"corpus_n\": {}, \"corpus_notbad\": {}, \"corpus_notbad_rate\": {} }}{comma}",
            cm.name,
            safe_name(&cm.name),
            cm.mirror_needed,
            tot,
            nb,
            jnum(rate),
        );
        let _ = n; // n carried via corpus_total lookup above
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}
