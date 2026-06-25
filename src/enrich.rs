//! `enrich` subcommand — the v2-filtered enrichment batch render bridge.
//!
//! Two deliberately-disjoint modes share one render path so scoring and the final
//! label crop are produced by the *same* iterate-once / recolor machinery as
//! `palette-probe` / `present` (the wallpaper f64 path):
//!
//! - **`--mode score`** (Stage B). For each location in a guided-descend
//!   `pool.jsonl`, iterate ONCE at the label geometry (center composition, the
//!   stored `cx/cy/fw`, 1280×720), apply the **present gates** (black `< 0.30` +
//!   detail-occupancy `>= 0.321`, parity with `present`), and — for survivors —
//!   recolor under **K seeded score-3 palettes**, streaming each recolored RGB
//!   frame to **stdout** as a raw length-prefixed record. The Python side
//!   (`tools/corpus/enrich_score.py`) scores every frame with v2 *in-memory*
//!   through inference.py's exact transform — so no 10k crops ever touch disk.
//!   A `--meta-out` sidecar JSONL records each location's gate verdict + the K
//!   palette names (one header line then one line per location).
//!
//! - **`--mode render`** (Stage D). Given the ~1100 selected `(location, argmax
//!   palette)` rows (`--selection`), render each at the locked label-crop quality
//!   (grid **ss4** + Lanczos-3, 1280×720, q90 JPG) into `--crops-dir`. Only the
//!   selected crops are ever written.
//!
//! Scoring renders at a cheap `--score-ss` (default 1): the model only ever sees
//! the 1280×720 frame downsampled to 384×224 by PIL bicubic, where ss1-vs-ss4 of
//! a 1280 source is invisible — the parity that matters is the *transform*, driven
//! Python-side, not the supersample of the scoring frame. The final label crops
//! (`render` mode) use the full ss4 path so what Matt judges is wallpaper-quality.
//!
//! Shallow f64 by construction (asserted per location via pixel spacing); the
//! guided-descend pool is shallow-regime.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::{EnrichArgs, EnrichMode};
use crate::energy::{self, OCC_FLOOR, OCC_GX, OCC_GY};
use crate::generate::color_params;
use crate::palette::{builtin, Palette};
use crate::palette_pick::{parse_colormaps, Colormap};
use crate::probe::SplitMix64;
use crate::render::{self, black_fraction, DownsampleFilter, Frame};
use crate::ensure_parent_dir;

const BAILOUT: f64 = 1e6;
const FULL_FILTER: DownsampleFilter = DownsampleFilter::Lanczos3;
/// f64 quantization floor on pixel spacing (matches `palette-probe`/`render-one`).
const PERTURB_SPACING: f64 = 1e-13;

// JPEG crop writer shared via `crate::render::save_jpeg`.
use crate::render::save_jpeg;

// ---------- hand-rolled JSONL field readers ----------------------------------
// Shared canonical copy lives in `crate::jsonl` (this module's variant was the
// source of truth — see jsonl.rs module doc).
use crate::jsonl::*;

// ---------- entry point ------------------------------------------------------

pub fn run_enrich(args: &EnrichArgs) -> Result<(), String> {
    match args.mode {
        EnrichMode::Score => run_score(args),
        EnrichMode::Render => run_render(args),
    }
}

/// One pool location: just the geometry the render path needs (provenance is
/// rejoined Python-side from `pool.jsonl` by `idx`).
struct PoolLoc {
    idx: usize,
    cx: f64,
    cy: f64,
    fw: f64,
}

fn parse_pool(text: &str) -> Vec<PoolLoc> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || !t.contains("\"cx\"") {
            continue;
        }
        let (Some(idx), Some(cx), Some(cy), Some(fw)) = (
            field_usize(t, "idx"),
            field_f64(t, "cx"),
            field_f64(t, "cy"),
            field_f64(t, "fw"),
        ) else {
            continue;
        };
        out.push(PoolLoc { idx, cx, cy, fw });
    }
    out
}

/// Pick K distinct palette indices from the library for a given location via a
/// seeded partial Fisher–Yates. The draw is **per-location** (seed mixed with the
/// pool `idx`) so different locations sample different palettes — a less-biased
/// estimate of each location's best palette than a single fixed set, and it gives
/// the final batch real palette diversity. Deterministic in `(seed, idx)`.
fn pick_palettes_for(n: usize, k: usize, seed: u64, idx: usize) -> Vec<usize> {
    let mut sel: Vec<usize> = (0..n).collect();
    let loc_seed = seed
        .wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ 0xD1B5_4A32_D192_ED03;
    let mut rng = SplitMix64(loc_seed);
    let take = k.min(n);
    for i in 0..take {
        let j = i + rng.below(n - i);
        sel.swap(i, j);
    }
    sel.truncate(take);
    sel
}

fn run_score(args: &EnrichArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.score_ss.max(1);

    let pool_text =
        std::fs::read_to_string(&args.pool).map_err(|e| format!("read {}: {e}", args.pool))?;
    let pool = parse_pool(&pool_text);
    if pool.is_empty() {
        return Err(format!("no locations parsed from {}", args.pool));
    }

    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    if library.is_empty() {
        return Err(format!("no palettes in {}", args.colormaps));
    }
    eprintln!(
        "enrich score: {} locations, K={} per-location palettes from {} roster, gates black<{} occ>={}",
        pool.len(),
        args.k,
        library.len(),
        args.black_cap,
        args.occ_floor
    );

    let params = color_params();
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let gate_palette = builtin("default", false).expect("default palette");

    // --- meta sidecar: header line then one line per location -----------------
    ensure_parent_dir(Path::new(&args.meta_out))?;
    let mut meta = BufWriter::new(
        File::create(&args.meta_out).map_err(|e| format!("create {}: {e}", args.meta_out))?,
    );
    writeln!(
        meta,
        "{{\"kind\":\"header\",\"k\":{},\"roster\":\"{}\",\"roster_size\":{},\"per_location_palettes\":true,\"width\":{},\"height\":{},\"score_ss\":{},\"maxiter\":{},\"black_cap\":{},\"occ_floor\":{},\"seed\":{}}}",
        args.k, args.colormaps.replace('\\', "/").replace('"', "\\\""), library.len(),
        args.width, args.height, ss, args.maxiter, args.black_cap, args.occ_floor, args.seed
    )
    .map_err(|e| format!("write meta: {e}"))?;

    // --- binary stdout image stream ------------------------------------------
    let stdout = io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());

    let t0 = Instant::now();
    let (mut n_gated, mut n_kept, mut n_deep) = (0usize, 0usize, 0usize);
    for (i, loc) in pool.iter().enumerate() {
        let frame = Frame {
            center: Complex::new(loc.cx, loc.cy),
            frame_width: loc.fw,
            out_width: args.width,
            out_height: args.height,
        };
        let pixel_spacing = loc.fw / args.width as f64;
        if pixel_spacing <= PERTURB_SPACING {
            n_deep += 1;
            writeln!(
                meta,
                "{{\"idx\":{},\"cx\":{},\"cy\":{},\"fw\":{},\"gated\":true,\"gate_reason\":\"deep\"}}",
                loc.idx, loc.cx, loc.cy, loc.fw
            )
            .map_err(|e| format!("write meta: {e}"))?;
            continue;
        }

        let (buf, _) = render::iterate_crop_buffer_f64(&frame, ss, args.maxiter, BAILOUT, trap, &params);
        let bf = black_fraction(&buf.samples);
        let gate_img = render::shade_and_downsample_filtered(
            &buf.samples, args.width, args.height, ss, &gate_palette, &params, pixel_spacing,
            FULL_FILTER,
        );
        let occ = energy::occupancy(&gate_img, OCC_GX, OCC_GY, OCC_FLOOR);

        let reason = if (bf as f64) >= args.black_cap {
            "black"
        } else if occ < args.occ_floor {
            "occ"
        } else {
            "ok"
        };
        let gated = reason != "ok";
        if gated {
            n_gated += 1;
            writeln!(
                meta,
                "{{\"idx\":{},\"cx\":{},\"cy\":{},\"fw\":{},\"black_fraction\":{:.5},\"occupancy\":{:.5},\"gated\":true,\"gate_reason\":\"{}\"}}",
                loc.idx, loc.cx, loc.cy, loc.fw, bf, occ, reason
            )
            .map_err(|e| format!("write meta: {e}"))?;
        } else {
            n_kept += 1;
            // per-location seeded K-palette draw (the filter's recolor set).
            let pal_idx = pick_palettes_for(library.len(), args.k, args.seed, loc.idx);
            let pal_json = pal_idx
                .iter()
                .map(|&pi| format!("\"{}\"", library[pi].name.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                meta,
                "{{\"idx\":{},\"cx\":{},\"cy\":{},\"fw\":{},\"black_fraction\":{:.5},\"occupancy\":{:.5},\"gated\":false,\"gate_reason\":\"ok\",\"palettes\":[{}]}}",
                loc.idx, loc.cx, loc.cy, loc.fw, bf, occ, pal_json
            )
            .map_err(|e| format!("write meta: {e}"))?;
            for (ki, &pi) in pal_idx.iter().enumerate() {
                let palette = Palette::from_srgb8_stops_mirrored(
                    library[pi].name.clone(), &library[pi].stops, false, library[pi].mirror_needed,
                );
                let img = render::shade_and_downsample_filtered(
                    &buf.samples, args.width, args.height, ss, &palette, &params, pixel_spacing,
                    FULL_FILTER,
                );
                // record header: idx, ki, w, h (LE u32) then w*h*3 RGB bytes.
                let hdr = [
                    (loc.idx as u32).to_le_bytes(),
                    (ki as u32).to_le_bytes(),
                    img.width().to_le_bytes(),
                    img.height().to_le_bytes(),
                ];
                for b in &hdr {
                    out.write_all(b).map_err(|e| format!("stdout: {e}"))?;
                }
                out.write_all(img.as_raw()).map_err(|e| format!("stdout: {e}"))?;
            }
        }

        if (i + 1) % 200 == 0 {
            meta.flush().ok();
            out.flush().ok();
            eprintln!(
                "  [{}/{}] kept {} gated {} deep {} ({:.0}s)",
                i + 1, pool.len(), n_kept, n_gated, n_deep, t0.elapsed().as_secs_f64()
            );
        }
    }
    out.flush().map_err(|e| format!("stdout flush: {e}"))?;
    meta.flush().map_err(|e| format!("meta flush: {e}"))?;
    eprintln!(
        "enrich score done: {} locations → kept {} (×{} recolors streamed), gated {}, deep-skip {} in {:.0}s",
        pool.len(), n_kept, args.k, n_gated, n_deep, t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One selected crop: geometry + argmax palette + the durable image_id stem.
struct Sel {
    image_id: String,
    cx: f64,
    cy: f64,
    fw: f64,
    palette: String,
}

fn parse_selection(text: &str) -> Vec<Sel> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || !t.contains("\"image_id\"") {
            continue;
        }
        let (Some(image_id), Some(cx), Some(cy), Some(fw), Some(palette)) = (
            field_str(t, "image_id"),
            field_f64(t, "cx"),
            field_f64(t, "cy"),
            field_f64(t, "fw"),
            field_str(t, "palette"),
        ) else {
            continue;
        };
        out.push(Sel { image_id, cx, cy, fw, palette });
    }
    out
}

fn run_render(args: &EnrichArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.render_ss.max(1);

    let sel_text = std::fs::read_to_string(&args.selection)
        .map_err(|e| format!("read {}: {e}", args.selection))?;
    let sel = parse_selection(&sel_text);
    if sel.is_empty() {
        return Err(format!("no selection rows parsed from {}", args.selection));
    }

    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library = parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    let by_name: std::collections::HashMap<&str, &Colormap> =
        library.iter().map(|c| (c.name.as_str(), c)).collect();

    let crops_dir = Path::new(&args.crops_dir);
    std::fs::create_dir_all(crops_dir).map_err(|e| format!("mkdir {}: {e}", crops_dir.display()))?;

    let params = color_params();
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    eprintln!(
        "enrich render: {} selected crops @ {}×{} ss{} lanczos3 q{} → {}",
        sel.len(), args.width, args.height, ss, args.jpg_quality, crops_dir.display()
    );

    let t0 = Instant::now();
    let mut n_done = 0usize;
    let mut n_missing_pal = 0usize;
    for (i, s) in sel.iter().enumerate() {
        let Some(cm) = by_name.get(s.palette.as_str()) else {
            eprintln!("  WARN palette '{}' not in library, skipping {}", s.palette, s.image_id);
            n_missing_pal += 1;
            continue;
        };
        let frame = Frame {
            center: Complex::new(s.cx, s.cy),
            frame_width: s.fw,
            out_width: args.width,
            out_height: args.height,
        };
        let pixel_spacing = s.fw / args.width as f64;
        if pixel_spacing <= PERTURB_SPACING {
            return Err(format!(
                "selection {} pixel spacing {pixel_spacing:.3e} is inside f64 quantization — \
                 enrich render is the shallow f64 path",
                s.image_id
            ));
        }
        let palette = Palette::from_srgb8_stops_mirrored(
            cm.name.clone(), &cm.stops, false, cm.mirror_needed,
        );
        let img = render::render_crop_f64(
            &frame, ss, args.maxiter, BAILOUT, trap, &palette, &params, FULL_FILTER,
        );
        // image_id is already fs-safe (built Python-side in enrich_select.py).
        save_jpeg(&img, &crops_dir.join(format!("{}.jpg", s.image_id)), args.jpg_quality)?;
        n_done += 1;
        if (i + 1) % 100 == 0 {
            eprintln!("  [{}/{}] ({:.0}s)", i + 1, sel.len(), t0.elapsed().as_secs_f64());
        }
    }
    eprintln!(
        "enrich render done: {} crops written ({} missing palette) in {:.0}s",
        n_done, n_missing_pal, t0.elapsed().as_secs_f64()
    );
    Ok(())
}
