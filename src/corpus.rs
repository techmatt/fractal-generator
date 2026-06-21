//! `corpus` — feature extractor that data-shapes the search's score bands.
//!
//! The search's structural features (busyness, interior fraction, boundary) are
//! products of **running the iteration** — `smooth_iter`, escaped-fraction, DE.
//! A finished wallpaper JPG is just RGB: you cannot recover those channels from
//! it. So this tool is deliberately two-honesty-tiered:
//!
//!  - **Color/aesthetic targets are exact** (palette, hue, chroma, contrast,
//!    luminance) — recovered directly from pixels, the valuable part, and the
//!    substrate for the eventual palette-matching step.
//!  - **Structural targets from images are proxies** — dark-pixel fraction for
//!    interior, edge density for busyness/boundary — and dark fraction conflates
//!    dead-black interior with a merely-dark palette. These seed the search's
//!    bands as **weak priors only** (`provenance:"bootstrap"`), to avoid a cold
//!    start. Labels later replace them with the search's **native** units pulled
//!    straight from `search.json` (no proxy gap).
//!
//! `targets.json` always expresses structural bands in the search's native
//! units. Bootstrap maps proxy→native roughly (interior_frac ≈ dark_frac 1:1;
//! busyness ≈ a corpus-fitted scale on edge density — explicitly approximate);
//! the `--labels`/`--search` blend (`α = n/(n+k)`) corrects them as picks
//! accumulate. The point isn't rigor on day one — it's bands that are
//! *data-shaped and improvable* instead of magic numbers.

use std::fs;
use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::cli::CorpusArgs;
use crate::font;
use crate::palette::{linear_srgb_to_oklab, srgb_to_linear};
use crate::probe::{jf, js, SplitMix64};
use crate::sheet::compose_grid;

/// Recognized top-level image extensions (lower-cased).
const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp"];
/// Tile grid for the edge-spread / flat-fraction heuristic.
const TILE_GRID: usize = 8;
/// OKLab L below this counts as "dark" (interior proxy).
const DARK_L: f64 = 0.18;
/// Per-tile mean edge below this → the tile is "flat".
const FLAT_TILE_EDGE: f64 = 0.015;
/// Hue histogram bins (OKLab hue angle).
const HUE_BINS: usize = 12;
/// Per-image dominant-color clusters (k-means in OKLab).
const KMEANS_K: usize = 5;
/// Pixels sampled per image for k-means (stride-subsampled for speed).
const KMEANS_SAMPLE: usize = 12_000;
/// Lloyd iterations for per-image k-means.
const KMEANS_ITERS: usize = 12;
/// Corpus-level palette size (k-means over pooled per-image cluster centers).
const CORPUS_PALETTE_K: usize = 8;
/// Native busyness the corpus *median* edge density is fitted to map onto. This
/// is the proxy→native scale anchor; ~0.15 matches typical `search.json`
/// busyness. Explicitly approximate — labels correct it.
const BUSYNESS_TARGET_MEDIAN: f64 = 0.15;
/// Rejected-thumbnail tile size for the audit sheet (fixed so the grid aligns).
const THUMB_W: u32 = 256;
const THUMB_H: u32 = 160;

// ===========================================================================
// Per-image results
// ===========================================================================

/// The cheap, inspectable "looks fractal" metrics (computed for every image).
#[derive(Clone, Copy)]
struct Metrics {
    edge_density: f64,
    edge_spread: f64,
    flat_fraction: f64,
    color_entropy: f64,
}

/// Exact color targets + proxy structural features (kept images only).
struct ImgFeatures {
    /// Dominant OKLab clusters `(lab, weight)`, weight = pixel fraction.
    palette: Vec<([f64; 3], f64)>,
    hue_hist: [f64; HUE_BINS],
    mean_chroma: f64,
    contrast: f64,
    luminance: f64,
    /// Structural proxies (weak).
    dark_frac: f64,
}

struct ImgResult {
    name: String,
    path: String,
    metrics: Option<Metrics>,
    kept: bool,
    reject_reasons: Vec<String>,
    feat: Option<ImgFeatures>,
    /// Fixed-size letterboxed thumbnail (rejected images only, for the sheet).
    thumb: Option<RgbImage>,
    error: Option<String>,
}

// ===========================================================================
// Entry point
// ===========================================================================

pub fn run_corpus(args: &CorpusArgs) -> Result<(), String> {
    let dir = Path::new(&args.dir);
    if !dir.is_dir() {
        return Err(format!("--dir '{}' is not a directory", args.dir));
    }
    if args.labels.is_some() != args.search.is_some() {
        return Err("--labels and --search must be given together".into());
    }

    let includes = split_list(&args.include);
    let excludes = split_list(&args.exclude);

    // ---- enumerate top-level image files (no recursion) ----
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", args.dir))? {
        let entry = entry.map_err(|e| format!("reading dir entry: {e}"))?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if IMAGE_EXTS.contains(&ext.as_str()) {
            files.push(p);
        }
    }
    files.sort();
    let n_total = files.len();
    if n_total == 0 {
        return Err(format!("no images ({IMAGE_EXTS:?}) at top level of {}", args.dir));
    }
    eprintln!("corpus: {n_total} image(s) in {} — decoding (rayon-parallel) ...", args.dir);
    if n_total > 1500 {
        eprintln!("  (large folder — decode dominates; expect more than a minute)");
    }

    // ---- parallel decode + feature pipeline ----
    let mut results: Vec<ImgResult> = files
        .par_iter()
        .map(|p| process_image(p, args, &includes, &excludes))
        .collect();
    results.sort_by(|a, b| a.name.cmp(&b.name));

    let n_errors = results.iter().filter(|r| r.error.is_some()).count();
    let kept: Vec<&ImgResult> = results.iter().filter(|r| r.kept).collect();
    let rejected: Vec<&ImgResult> = results
        .iter()
        .filter(|r| !r.kept && r.error.is_none())
        .collect();
    let n_kept = kept.len();
    let n_rejected = rejected.len();

    // ---- audit table ----
    print_audit(&results, args, n_total, n_kept, n_rejected, n_errors);

    // ---- rejected-thumbnails sheet ----
    let rejected_sheet_path = if n_rejected > 0 {
        let tiles: Vec<RgbImage> = rejected.iter().filter_map(|r| r.thumb.clone()).collect();
        if tiles.is_empty() {
            None
        } else {
            let grid = compose_grid(&tiles, None);
            grid.save(&args.rejected_sheet)
                .map_err(|e| format!("failed to write {}: {e}", args.rejected_sheet))?;
            Some(args.rejected_sheet.clone())
        }
    } else {
        None
    };

    // ---- aggregate bootstrap bands + color targets ----
    let agg = aggregate(&kept);

    // ---- optional label transition ----
    let label_info = match (&args.labels, &args.search) {
        (Some(lp), Some(sp)) => Some(load_labels_and_nodes(lp, sp)?),
        _ => None,
    };
    let bands = build_bands(&agg, label_info.as_ref(), args.blend_k);
    let n_labels = label_info.as_ref().map(|l| l.matched).unwrap_or(0);

    // ---- write outputs ----
    let targets_json = build_targets_json(
        &bands, &agg, args, n_total, n_kept, n_rejected, n_errors, n_labels,
    );
    fs::write(&args.targets_out, targets_json)
        .map_err(|e| format!("failed to write {}: {e}", args.targets_out))?;

    let features_json = build_features_json(&results, args);
    fs::write(&args.features_out, features_json)
        .map_err(|e| format!("failed to write {}: {e}", args.features_out))?;

    // ---- report ----
    report(&bands, &agg, label_info.as_ref(), args, &rejected_sheet_path);

    Ok(())
}

// ===========================================================================
// Per-image processing
// ===========================================================================

fn process_image(
    path: &Path,
    args: &CorpusArgs,
    includes: &[String],
    excludes: &[String],
) -> ImgResult {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let path_s = path.to_string_lossy().replace('\\', "/");
    let mut blank = ImgResult {
        name: name.clone(),
        path: path_s.clone(),
        metrics: None,
        kept: false,
        reject_reasons: Vec::new(),
        feat: None,
        thumb: None,
        error: None,
    };

    let dynimg = match image::open(path) {
        Ok(i) => i,
        Err(e) => {
            blank.error = Some(format!("decode failed: {e}"));
            return blank;
        }
    };
    // Downscale to longest-edge ~max_edge for feature speed (never upscale).
    let (w0, h0) = (dynimg.width(), dynimg.height());
    let small = if w0.max(h0) > args.max_edge {
        dynimg.resize(args.max_edge, args.max_edge, FilterType::Triangle)
    } else {
        dynimg
    };
    let rgb = small.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    if w < 8 || h < 8 {
        blank.error = Some(format!("too small after downscale ({w}x{h})"));
        return blank;
    }

    // OKLab per pixel; L grid for the Sobel/grayscale stage.
    let mut labs: Vec<[f64; 3]> = Vec::with_capacity(w * h);
    let mut lgrid: Vec<f64> = Vec::with_capacity(w * h);
    for px in rgb.pixels() {
        let lin = [
            srgb_to_linear(px[0] as f64 / 255.0),
            srgb_to_linear(px[1] as f64 / 255.0),
            srgb_to_linear(px[2] as f64 / 255.0),
        ];
        let lab = linear_srgb_to_oklab(lin);
        lgrid.push(lab[0]);
        labs.push(lab);
    }

    let metrics = compute_metrics(&labs, &lgrid, w, h);

    // ---- keep/reject decision ----
    //
    // The trap (which the first naive pass fell into): a dark fractal on black —
    // a flame or off-center Julia — has *low* mean edge density and a *high* flat
    // fraction and *high* edge spread (its detail is concentrated in the bright
    // region), tripping every "obvious" reject. But it is the most fractal image
    // in the folder. What actually separates the degenerate non-fractals is that
    // they are ALSO color-poor or detail-uniform; a dark fractal stays color-rich
    // (high `color_entropy`) and structurally non-uniform (high `edge_spread`).
    // So every structural reject is gated on color-poverty to protect them.
    // Photos remain mostly un-catchable by crude metrics — accepted, documented.
    let m = metrics;
    let mut reasons: Vec<String> = Vec::new();
    let forced_keep = includes.iter().any(|s| name.contains(s.as_str()));
    let forced_reject = excludes.iter().any(|s| name.contains(s.as_str()));
    if !forced_keep {
        let palette_gate = 2.0 * args.entropy_min; // "poor palette" for structural rules
        // R1 uniform smooth: low detail spread *evenly* (gradient / solid). The
        // spread gate spares smooth flame fractals, which vary more across tiles.
        if m.edge_density < args.edge_min && m.edge_spread < args.uniform_spread {
            reasons.push(format!(
                "uniform: edge {:.4}<{:.4} & spread {:.2}<{:.2}",
                m.edge_density, args.edge_min, m.edge_spread, args.uniform_spread
            ));
        }
        // R2 color-poor: too few tones (solid / near-grayscale / banded gradient).
        if m.color_entropy < args.entropy_min {
            reasons.push(format!("color_entropy {:.2}<{:.2}", m.color_entropy, args.entropy_min));
        }
        // R3 text / logo: detail concentrated (high spread) AND a poor palette —
        // a dark fractal also has high spread but a rich palette, so the palette
        // gate keeps it. High spread threshold: high spread alone is NOT a
        // non-fractal signal (fractals concentrate detail too).
        if m.edge_spread > args.spread_max && m.color_entropy < palette_gate {
            reasons.push(format!(
                "text/logo: spread {:.2}>{:.2} & entropy {:.2}<{:.2}",
                m.edge_spread, args.spread_max, m.color_entropy, palette_gate
            ));
        }
        // R4 dead-flat: mostly empty AND color-poor (a sparse non-fractal, not a
        // rich dark fractal on black).
        if m.flat_fraction > args.flat_max && m.color_entropy < palette_gate {
            reasons.push(format!(
                "dead-flat: flat {:.2}>{:.2} & entropy {:.2}<{:.2}",
                m.flat_fraction, args.flat_max, m.color_entropy, palette_gate
            ));
        }
    }
    if forced_reject {
        reasons.push("force-excluded".into());
    }
    let kept = forced_keep || (reasons.is_empty() && !forced_reject);

    let mut out = ImgResult {
        name,
        path: path_s,
        metrics: Some(metrics),
        kept,
        reject_reasons: reasons.clone(),
        feat: None,
        thumb: None,
        error: None,
    };

    if kept {
        out.feat = Some(compute_features(&labs, &lgrid, metrics));
    } else {
        // Letterbox thumbnail + reason label for the audit sheet.
        let thumb = fit_thumb(&rgb, THUMB_W, THUMB_H);
        let mut t = thumb;
        let label = if reasons.is_empty() {
            out.name.clone()
        } else {
            reasons.join("  ")
        };
        let short: String = label.chars().take(40).collect();
        font::draw_text(&mut t, &short.to_uppercase(), 2, 2, 1, Rgb([240, 240, 240]), true);
        out.thumb = Some(t);
    }
    out
}

/// Mean Sobel edge density, per-tile edge-spread CoV, flat-tile fraction (on the
/// OKLab-L grayscale), and effective OKLab color count (chroma-weighted, on the
/// full lab pixels).
fn compute_metrics(labs: &[[f64; 3]], l: &[f64], w: usize, h: usize) -> Metrics {
    // Sobel gradient magnitude per interior pixel; accumulate per 8×8 tile.
    let mut tile_sum = [0.0f64; TILE_GRID * TILE_GRID];
    let mut tile_cnt = [0u32; TILE_GRID * TILE_GRID];
    let mut total = 0.0f64;
    let mut total_cnt = 0u64;
    let at = |x: usize, y: usize| l[y * w + x];
    for y in 1..h - 1 {
        let ty = (y * TILE_GRID) / h;
        for x in 1..w - 1 {
            // Sobel, normalized so a full black/white edge ~ 1.0.
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2.0 * at(x - 1, y) - at(x - 1, y + 1))
                / 4.0;
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1))
                / 4.0;
            let mag = (gx * gx + gy * gy).sqrt();
            let tx = (x * TILE_GRID) / w;
            let ti = ty * TILE_GRID + tx;
            tile_sum[ti] += mag;
            tile_cnt[ti] += 1;
            total += mag;
            total_cnt += 1;
        }
    }
    let edge_density = if total_cnt > 0 { total / total_cnt as f64 } else { 0.0 };

    // Per-tile mean edge → coefficient of variation + flat fraction.
    let mut tile_edges = Vec::with_capacity(TILE_GRID * TILE_GRID);
    for i in 0..TILE_GRID * TILE_GRID {
        if tile_cnt[i] > 0 {
            tile_edges.push(tile_sum[i] / tile_cnt[i] as f64);
        }
    }
    let tmean = tile_edges.iter().sum::<f64>() / tile_edges.len().max(1) as f64;
    let tvar = tile_edges.iter().map(|e| (e - tmean).powi(2)).sum::<f64>()
        / tile_edges.len().max(1) as f64;
    let edge_spread = if tmean > 1e-9 { tvar.sqrt() / tmean } else { 0.0 };
    let flat_fraction = tile_edges.iter().filter(|&&e| e < FLAT_TILE_EDGE).count() as f64
        / tile_edges.len().max(1) as f64;

    let color_entropy = color_entropy(labs);

    Metrics {
        edge_density,
        edge_spread,
        flat_fraction,
        color_entropy,
    }
}

/// Effective number of OKLab colors: `exp(H)` of a **chroma-weighted** 2-D
/// chromaticity histogram (OKLab a×b, 12×12 bins over ±0.3). Weighting by chroma
/// is the crux — a dark fractal is mostly black, but its bright filaments carry
/// vivid, varied hues, so it scores *high*; a solid, a single-hue gradient, or a
/// near-grayscale image scores *low*. This is the palette-richness gate that
/// keeps the reject rules from tossing dark-but-colorful fractals (the luminance
/// version did exactly that). Near-grayscale → ~0 (correctly color-poor for a
/// palette corpus).
fn color_entropy(labs: &[[f64; 3]]) -> f64 {
    const NB: usize = 12;
    const SPAN: f64 = 0.3; // a,b mostly within ±0.3 in OKLab
    let mut hist = [0.0f64; NB * NB];
    let mut wsum = 0.0;
    for p in labs {
        let c = (p[1] * p[1] + p[2] * p[2]).sqrt();
        if c < 1e-3 {
            continue; // neutral pixel carries no palette information
        }
        let ai = (((p[1] + SPAN) / (2.0 * SPAN)) * NB as f64).clamp(0.0, (NB - 1) as f64) as usize;
        let bi = (((p[2] + SPAN) / (2.0 * SPAN)) * NB as f64).clamp(0.0, (NB - 1) as f64) as usize;
        hist[bi * NB + ai] += c;
        wsum += c;
    }
    if wsum <= 1e-9 {
        return 0.0; // essentially grayscale
    }
    let mut ent = 0.0;
    for &v in &hist {
        if v > 0.0 {
            let pr = v / wsum;
            ent -= pr * pr.ln();
        }
    }
    ent.exp()
}

/// Exact color targets + structural proxies for a kept image.
fn compute_features(labs: &[[f64; 3]], l: &[f64], metrics: Metrics) -> ImgFeatures {
    let n = labs.len() as f64;

    // dark fraction (interior proxy) + luminance + contrast.
    let dark = labs.iter().filter(|p| p[0] < DARK_L).count() as f64 / n;
    let luminance = l.iter().sum::<f64>() / n;
    let mut lsort = l.to_vec();
    lsort.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let contrast = percentile(&lsort, 0.9) - percentile(&lsort, 0.1);

    // chroma + hue histogram (chroma-weighted so neutrals don't dominate hue).
    let mut chroma_sum = 0.0;
    let mut hue = [0.0f64; HUE_BINS];
    let mut hue_w = 0.0;
    for p in labs {
        let c = (p[1] * p[1] + p[2] * p[2]).sqrt();
        chroma_sum += c;
        if c > 1e-4 {
            let mut ang = p[2].atan2(p[1]); // [-π, π]
            if ang < 0.0 {
                ang += std::f64::consts::TAU;
            }
            let b = ((ang / std::f64::consts::TAU) * HUE_BINS as f64) as usize % HUE_BINS;
            hue[b] += c;
            hue_w += c;
        }
    }
    let mean_chroma = chroma_sum / n;
    if hue_w > 0.0 {
        for v in hue.iter_mut() {
            *v /= hue_w;
        }
    }

    // dominant clusters via k-means over a strided pixel sample.
    let stride = (labs.len() / KMEANS_SAMPLE).max(1);
    let sample: Vec<[f64; 3]> = labs.iter().step_by(stride).copied().collect();
    let palette = kmeans(&sample, KMEANS_K, KMEANS_ITERS, 0xC0FFEE);

    let _ = metrics; // metrics retained on the ImgResult, not duplicated here.
    ImgFeatures {
        palette,
        hue_hist: hue,
        mean_chroma,
        contrast,
        luminance,
        dark_frac: dark,
    }
}

// ===========================================================================
// k-means in OKLab
// ===========================================================================

fn dist2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Lloyd k-means with k-means++ seeding. Returns `(center, weight)` sorted by
/// descending weight; weight is the assigned-pixel fraction.
fn kmeans(points: &[[f64; 3]], k: usize, iters: usize, seed: u64) -> Vec<([f64; 3], f64)> {
    if points.is_empty() {
        return Vec::new();
    }
    let k = k.min(points.len());
    let mut rng = SplitMix64(seed);
    // k-means++ seeding.
    let mut centers: Vec<[f64; 3]> = vec![points[rng.below(points.len())]];
    while centers.len() < k {
        let d2: Vec<f64> = points
            .iter()
            .map(|p| centers.iter().map(|c| dist2(p, c)).fold(f64::INFINITY, f64::min))
            .collect();
        let sum: f64 = d2.iter().sum();
        if sum <= 0.0 {
            centers.push(points[rng.below(points.len())]);
            continue;
        }
        let mut t = rng.unit() * sum;
        let mut chosen = points.len() - 1;
        for (i, &d) in d2.iter().enumerate() {
            t -= d;
            if t <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.push(points[chosen]);
    }

    let mut assign = vec![0usize; points.len()];
    for _ in 0..iters {
        // assign
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for (j, c) in centers.iter().enumerate() {
                let d = dist2(p, c);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            assign[i] = best;
        }
        // update
        let mut sums = vec![[0.0f64; 3]; k];
        let mut cnt = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let a = assign[i];
            sums[a][0] += p[0];
            sums[a][1] += p[1];
            sums[a][2] += p[2];
            cnt[a] += 1;
        }
        for j in 0..k {
            if cnt[j] > 0 {
                centers[j] = [
                    sums[j][0] / cnt[j] as f64,
                    sums[j][1] / cnt[j] as f64,
                    sums[j][2] / cnt[j] as f64,
                ];
            } else {
                centers[j] = points[rng.below(points.len())];
            }
        }
    }
    // final weights
    let mut cnt = vec![0usize; k];
    for (i, p) in points.iter().enumerate() {
        let mut best = 0;
        let mut bd = f64::INFINITY;
        for (j, c) in centers.iter().enumerate() {
            let d = dist2(p, c);
            if d < bd {
                bd = d;
                best = j;
            }
        }
        assign[i] = best;
        cnt[best] += 1;
    }
    let total = points.len() as f64;
    let mut out: Vec<([f64; 3], f64)> = centers
        .into_iter()
        .zip(cnt.iter())
        .map(|(c, &n)| (c, n as f64 / total))
        .filter(|(_, w)| *w > 0.0)
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out
}

// ===========================================================================
// Aggregation → bootstrap bands + exact color targets
// ===========================================================================

/// Corpus-level aggregates over the kept images.
struct Aggregate {
    /// proxy→native fit: `busyness_native = edge_density * busyness_scale`.
    busyness_scale: f64,
    busyness_band: (f64, f64),
    busyness_reject_below: f64,
    interior_band: (f64, f64),
    interior_reject_above: f64,
    boundary_band: (f64, f64),
    // color targets (exact, image-domain)
    palette: Vec<([f64; 3], f64)>,
    hue_hist: [f64; HUE_BINS],
    chroma: (f64, f64),
    contrast: (f64, f64),
    luminance: (f64, f64),
}

fn aggregate(kept: &[&ImgResult]) -> Aggregate {
    // collect per-image scalars
    let edges: Vec<f64> = kept
        .iter()
        .filter_map(|r| r.metrics.map(|m| m.edge_density))
        .collect();
    let darks: Vec<f64> = kept.iter().filter_map(|r| r.feat.as_ref().map(|f| f.dark_frac)).collect();
    let chromas: Vec<f64> = kept.iter().filter_map(|r| r.feat.as_ref().map(|f| f.mean_chroma)).collect();
    let contrasts: Vec<f64> = kept.iter().filter_map(|r| r.feat.as_ref().map(|f| f.contrast)).collect();
    let lums: Vec<f64> = kept.iter().filter_map(|r| r.feat.as_ref().map(|f| f.luminance)).collect();

    let median_edge = {
        let mut e = edges.clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        percentile(&e, 0.5).max(1e-6)
    };
    let busyness_scale = BUSYNESS_TARGET_MEDIAN / median_edge;
    let busy: Vec<f64> = edges.iter().map(|e| e * busyness_scale).collect();

    let busyness_band = (psorted(&busy, 0.1), psorted(&busy, 0.9));
    let busyness_reject_below = psorted(&busy, 0.1);
    let interior_band = (psorted(&darks, 0.1), psorted(&darks, 0.9));
    let interior_reject_above = psorted(&darks, 0.9);
    // boundary uses the same edge-density proxy (a weaker signal than busyness).
    let boundary_band = (psorted(&busy, 0.25), psorted(&busy, 0.75));

    // corpus palette: pool every kept image's clusters, weighted, re-cluster.
    let mut pool: Vec<[f64; 3]> = Vec::new();
    for r in kept {
        if let Some(f) = &r.feat {
            for (lab, w) in &f.palette {
                // expand by weight (×100) so the corpus k-means sees mass.
                let reps = (w * 100.0).round() as usize;
                for _ in 0..reps.max(1) {
                    pool.push(*lab);
                }
            }
        }
    }
    let palette = if pool.is_empty() {
        Vec::new()
    } else {
        kmeans(&pool, CORPUS_PALETTE_K, KMEANS_ITERS, 0xBEEF)
    };

    // corpus hue histogram: average per-image (already normalized) histograms.
    let mut hue = [0.0f64; HUE_BINS];
    let mut nh = 0.0;
    for r in kept {
        if let Some(f) = &r.feat {
            for i in 0..HUE_BINS {
                hue[i] += f.hue_hist[i];
            }
            nh += 1.0;
        }
    }
    if nh > 0.0 {
        for v in hue.iter_mut() {
            *v /= nh;
        }
    }

    Aggregate {
        busyness_scale,
        busyness_band,
        busyness_reject_below,
        interior_band,
        interior_reject_above,
        boundary_band,
        palette,
        hue_hist: hue,
        chroma: (psorted(&chromas, 0.1), psorted(&chromas, 0.9)),
        contrast: (psorted(&contrasts, 0.1), psorted(&contrasts, 0.9)),
        luminance: (psorted(&lums, 0.1), psorted(&lums, 0.9)),
    }
}

// ===========================================================================
// Label transition (bootstrap → native labeled bands)
// ===========================================================================

/// Labeled native structural distributions pulled from `search.json`.
struct LabelInfo {
    /// number of labeled nodes matched in search.json (keep + discard).
    matched: usize,
    /// native busyness band [p10,p90] from *kept* labeled nodes (if ≥1).
    busyness_band: Option<(f64, f64)>,
    busyness_reject_below: Option<f64>,
    /// native period band [p10,p90] from *kept* labeled nodes (if ≥1).
    period_band: Option<(f64, f64)>,
    n_keep: usize,
    n_discard: usize,
}

fn load_labels_and_nodes(labels_path: &str, search_path: &str) -> Result<LabelInfo, String> {
    let ltext = fs::read_to_string(labels_path)
        .map_err(|e| format!("reading {labels_path}: {e}"))?;
    let labels = parse_labels(&ltext); // Vec<(id, keep)>
    let stext = fs::read_to_string(search_path)
        .map_err(|e| format!("reading {search_path}: {e}"))?;
    let nodes = parse_search_nodes(&stext); // Vec<(id, busyness, period)>

    use std::collections::HashMap;
    let map: HashMap<u64, (f64, u32)> =
        nodes.into_iter().map(|(id, b, p)| (id, (b, p))).collect();

    let mut keep_busy: Vec<f64> = Vec::new();
    let mut keep_period: Vec<f64> = Vec::new();
    let mut discard_busy: Vec<f64> = Vec::new();
    let mut matched = 0usize;
    let mut n_keep = 0usize;
    let mut n_discard = 0usize;
    for (id, keep) in &labels {
        if let Some(&(b, p)) = map.get(id) {
            matched += 1;
            if *keep {
                n_keep += 1;
                if b.is_finite() {
                    keep_busy.push(b);
                }
                keep_period.push(p as f64);
            } else {
                n_discard += 1;
                if b.is_finite() {
                    discard_busy.push(b);
                }
            }
        }
    }

    let busyness_band = if keep_busy.len() >= 1 {
        keep_busy.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some((psorted(&keep_busy, 0.1), psorted(&keep_busy, 0.9)))
    } else {
        None
    };
    // reject_below: midway between the busiest discarded and the least-busy kept,
    // falling back to the kept p10 when no discards inform it.
    let busyness_reject_below = if !keep_busy.is_empty() {
        let keep_lo = psorted(&keep_busy, 0.1);
        if !discard_busy.is_empty() {
            discard_busy.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let disc_hi = psorted(&discard_busy, 0.9);
            Some(0.5 * (keep_lo + disc_hi.min(keep_lo)))
        } else {
            Some(keep_lo)
        }
    } else {
        None
    };
    let period_band = if !keep_period.is_empty() {
        keep_period.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some((psorted(&keep_period, 0.1), psorted(&keep_period, 0.9)))
    } else {
        None
    };

    Ok(LabelInfo {
        matched,
        busyness_band,
        busyness_reject_below,
        period_band,
        n_keep,
        n_discard,
    })
}

/// One structural band, post-blend, ready to serialize.
struct StructBand {
    lo: f64,
    hi: f64,
    reject_below: Option<f64>,
    reject_above: Option<f64>,
    provenance: String,
    n_labels: usize,
    alpha: f64,
}

struct Bands {
    busyness: StructBand,
    interior_frac: StructBand,
    boundary: StructBand,
    period: StructBand,
}

fn blend(b: (f64, f64), l: (f64, f64), a: f64) -> (f64, f64) {
    ((1.0 - a) * b.0 + a * l.0, (1.0 - a) * b.1 + a * l.1)
}

fn build_bands(agg: &Aggregate, label: Option<&LabelInfo>, k: f64) -> Bands {
    // busyness: bootstrap proxy band, blended toward labeled native if present.
    let (busyness, period) = match label {
        Some(li) if li.matched > 0 => {
            let alpha = li.matched as f64 / (li.matched as f64 + k);
            // busyness blend (only where labeled busyness exists)
            let busyness = if let Some(lb) = li.busyness_band {
                let (lo, hi) = blend(agg.busyness_band, lb, alpha);
                let rb = match li.busyness_reject_below {
                    Some(lr) => (1.0 - alpha) * agg.busyness_reject_below + alpha * lr,
                    None => agg.busyness_reject_below,
                };
                StructBand {
                    lo,
                    hi,
                    reject_below: Some(rb),
                    reject_above: None,
                    provenance: "blend".into(),
                    n_labels: li.n_keep + li.n_discard,
                    alpha,
                }
            } else {
                StructBand {
                    lo: agg.busyness_band.0,
                    hi: agg.busyness_band.1,
                    reject_below: Some(agg.busyness_reject_below),
                    reject_above: None,
                    provenance: "bootstrap".into(),
                    n_labels: 0,
                    alpha: 0.0,
                }
            };
            // period: bootstrap has none → labels-only.
            let period = match li.period_band {
                Some(pb) => StructBand {
                    lo: pb.0,
                    hi: pb.1,
                    reject_below: None,
                    reject_above: None,
                    provenance: "labels".into(),
                    n_labels: li.n_keep,
                    alpha: 1.0,
                },
                None => default_period_band(),
            };
            (busyness, period)
        }
        _ => (
            StructBand {
                lo: agg.busyness_band.0,
                hi: agg.busyness_band.1,
                reject_below: Some(agg.busyness_reject_below),
                reject_above: None,
                provenance: "bootstrap".into(),
                n_labels: 0,
                alpha: 0.0,
            },
            default_period_band(),
        ),
    };

    // interior_frac / boundary: image proxies only — search.json carries no
    // native channel for them, so labels can't correct them yet (α stays 0).
    let interior_frac = StructBand {
        lo: agg.interior_band.0,
        hi: agg.interior_band.1,
        reject_below: None,
        reject_above: Some(agg.interior_reject_above),
        provenance: "bootstrap".into(),
        n_labels: 0,
        alpha: 0.0,
    };
    let boundary = StructBand {
        lo: agg.boundary_band.0,
        hi: agg.boundary_band.1,
        reject_below: None,
        reject_above: None,
        provenance: "bootstrap".into(),
        n_labels: 0,
        alpha: 0.0,
    };

    Bands {
        busyness,
        interior_frac,
        boundary,
        period,
    }
}

/// The search's current period band, expressed as a [lo,hi] plateau with
/// `provenance:"default"` so the search keeps its built-in constants.
fn default_period_band() -> StructBand {
    StructBand {
        lo: 3.0,
        hi: 20_000.0,
        reject_below: None,
        reject_above: None,
        provenance: "default".into(),
        n_labels: 0,
        alpha: 0.0,
    }
}

// ===========================================================================
// JSON writers (hand-rolled)
// ===========================================================================

fn band_json(name: &str, b: &StructBand, indent: &str) -> String {
    let mut s = format!("{indent}\"{name}\": {{ \"band\": [{}, {}]", jf(b.lo), jf(b.hi));
    if let Some(r) = b.reject_below {
        s.push_str(&format!(", \"reject_below\": {}", jf(r)));
    }
    if let Some(r) = b.reject_above {
        s.push_str(&format!(", \"reject_above\": {}", jf(r)));
    }
    s.push_str(&format!(
        ", \"units\": \"native\", \"provenance\": {}, \"n_labels\": {}, \"alpha\": {} }}",
        js(&b.provenance),
        b.n_labels,
        jf(b.alpha)
    ));
    s
}

#[allow(clippy::too_many_arguments)]
fn build_targets_json(
    bands: &Bands,
    agg: &Aggregate,
    args: &CorpusArgs,
    n_total: usize,
    n_kept: usize,
    n_rejected: usize,
    n_errors: usize,
    n_labels: usize,
) -> String {
    let mut s = String::from("{\n");
    s.push_str("  \"structural\": {\n");
    s.push_str(&band_json("busyness", &bands.busyness, "    "));
    s.push_str(",\n");
    s.push_str(&band_json("interior_frac", &bands.interior_frac, "    "));
    s.push_str(",\n");
    s.push_str(&band_json("boundary", &bands.boundary, "    "));
    s.push_str(",\n");
    s.push_str(&band_json("period", &bands.period, "    "));
    s.push_str("\n  },\n");

    // color
    s.push_str("  \"color\": {\n");
    s.push_str("    \"palette\": [\n");
    for (i, (lab, w)) in agg.palette.iter().enumerate() {
        s.push_str(&format!(
            "      {{ \"oklab\": [{}, {}, {}], \"weight\": {} }}{}\n",
            jf(lab[0]),
            jf(lab[1]),
            jf(lab[2]),
            jf(*w),
            if i + 1 < agg.palette.len() { "," } else { "" }
        ));
    }
    s.push_str("    ],\n");
    s.push_str(&format!(
        "    \"hue_hist\": [{}],\n",
        agg.hue_hist.iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", ")
    ));
    s.push_str(&format!("    \"chroma\": [{}, {}],\n", jf(agg.chroma.0), jf(agg.chroma.1)));
    s.push_str(&format!("    \"contrast\": [{}, {}],\n", jf(agg.contrast.0), jf(agg.contrast.1)));
    s.push_str(&format!("    \"luminance\": [{}, {}]\n", jf(agg.luminance.0), jf(agg.luminance.1)));
    s.push_str("  },\n");

    // meta
    s.push_str("  \"meta\": {\n");
    s.push_str(&format!("    \"n_total\": {n_total},\n"));
    s.push_str(&format!("    \"n_kept\": {n_kept},\n"));
    s.push_str(&format!("    \"n_rejected\": {n_rejected},\n"));
    s.push_str(&format!("    \"n_errors\": {n_errors},\n"));
    s.push_str(&format!("    \"n_labels\": {n_labels},\n"));
    s.push_str(&format!("    \"busyness_edge_scale\": {},\n", jf(agg.busyness_scale)));
    s.push_str(&format!("    \"dir\": {},\n", js(&args.dir)));
    s.push_str("    \"note\": \"structural bands are search-native; busyness/interior/boundary are image proxies (provenance bootstrap) until labels correct them; period is labels-only\"\n");
    s.push_str("  }\n}\n");
    s
}

fn build_features_json(results: &[ImgResult], args: &CorpusArgs) -> String {
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"dir\": {},\n", js(&args.dir)));
    s.push_str("  \"thresholds\": {\n");
    s.push_str(&format!("    \"edge_min\": {},\n", jf(args.edge_min)));
    s.push_str(&format!("    \"flat_max\": {},\n", jf(args.flat_max)));
    s.push_str(&format!("    \"spread_max\": {},\n", jf(args.spread_max)));
    s.push_str(&format!("    \"entropy_min\": {}\n", jf(args.entropy_min)));
    s.push_str("  },\n");
    s.push_str("  \"images\": [\n");
    for (i, r) in results.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"name\": {},\n", js(&r.name)));
        s.push_str(&format!("      \"path\": {},\n", js(&r.path)));
        s.push_str(&format!("      \"kept\": {},\n", r.kept));
        match &r.error {
            Some(e) => s.push_str(&format!("      \"error\": {},\n", js(e))),
            None => s.push_str("      \"error\": null,\n"),
        }
        match &r.metrics {
            Some(m) => s.push_str(&format!(
                "      \"metrics\": {{ \"edge_density\": {}, \"edge_spread\": {}, \"flat_fraction\": {}, \"color_entropy\": {} }},\n",
                jf(m.edge_density), jf(m.edge_spread), jf(m.flat_fraction), jf(m.color_entropy)
            )),
            None => s.push_str("      \"metrics\": null,\n"),
        }
        s.push_str(&format!(
            "      \"reject_reasons\": [{}],\n",
            r.reject_reasons.iter().map(|x| js(x)).collect::<Vec<_>>().join(", ")
        ));
        match &r.feat {
            Some(f) => {
                s.push_str("      \"features\": {\n");
                s.push_str("        \"palette\": [");
                let pal: Vec<String> = f
                    .palette
                    .iter()
                    .map(|(lab, w)| {
                        format!(
                            "{{ \"oklab\": [{}, {}, {}], \"weight\": {} }}",
                            jf(lab[0]), jf(lab[1]), jf(lab[2]), jf(*w)
                        )
                    })
                    .collect();
                s.push_str(&pal.join(", "));
                s.push_str("],\n");
                s.push_str(&format!(
                    "        \"hue_hist\": [{}],\n",
                    f.hue_hist.iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", ")
                ));
                s.push_str(&format!("        \"mean_chroma\": {},\n", jf(f.mean_chroma)));
                s.push_str(&format!("        \"contrast\": {},\n", jf(f.contrast)));
                s.push_str(&format!("        \"luminance\": {},\n", jf(f.luminance)));
                s.push_str(&format!("        \"dark_frac\": {}\n", jf(f.dark_frac)));
                s.push_str("      }\n");
            }
            None => s.push_str("      \"features\": null\n"),
        }
        s.push_str("    }");
        if i + 1 < results.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

// ===========================================================================
// Minimal JSON readers for labels.json / search.json
// ===========================================================================

/// Parse `labels.json`: `{ "labels": [ {"id": N, "keep": true}, ... ] }`.
/// Tolerant scan: for each `"id"` token, read the integer, then the next
/// `"keep"` boolean.
fn parse_labels(text: &str) -> Vec<(u64, bool)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(p) = find_from(text, "\"id\"", i) {
        let after = p + 4;
        let (id, ni) = read_uint(bytes, after);
        i = ni;
        // find the next "keep" before the next "id"
        let next_id = find_from(text, "\"id\"", i).unwrap_or(text.len());
        if let Some(kp) = find_from(text, "\"keep\"", i) {
            if kp < next_id {
                let keep = read_bool(text, kp + 6);
                if let Some(idv) = id {
                    out.push((idv, keep));
                }
                continue;
            }
        }
        if let Some(idv) = id {
            out.push((idv, true)); // default keep if unspecified
        }
    }
    out
}

/// Parse `search.json` nodes: each node object contains `"id"`, `"busyness"`,
/// `"period"`. Returns `(id, busyness, period)`; busyness NaN if null.
fn parse_search_nodes(text: &str) -> Vec<(u64, f64, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    // node ids appear as `"id": N` inside the nodes array; the same token also
    // never appears elsewhere in our schema. Slice each node block at the next id.
    let mut ids: Vec<usize> = Vec::new();
    while let Some(p) = find_from(text, "\"id\":", i) {
        ids.push(p);
        i = p + 5;
    }
    for (k, &start) in ids.iter().enumerate() {
        let end = ids.get(k + 1).copied().unwrap_or(text.len());
        let block = &text[start..end];
        let bytes = block.as_bytes();
        let (id, _) = read_uint(bytes, 5); // after "id":
        let busyness = field_f64(block, "\"busyness\":").unwrap_or(f64::NAN);
        let period = field_f64(block, "\"period\":").unwrap_or(0.0) as u32;
        if let Some(idv) = id {
            out.push((idv, busyness, period));
        }
    }
    out
}

fn find_from(hay: &str, needle: &str, from: usize) -> Option<usize> {
    hay.get(from..).and_then(|s| s.find(needle)).map(|p| p + from)
}

/// Read an unsigned integer starting at/after byte `pos` (skips ws and `:`).
fn read_uint(b: &[u8], mut pos: usize) -> (Option<u64>, usize) {
    while pos < b.len() && (b[pos] == b' ' || b[pos] == b':' || b[pos] == b'\t') {
        pos += 1;
    }
    let start = pos;
    while pos < b.len() && b[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == start {
        return (None, pos);
    }
    let v = std::str::from_utf8(&b[start..pos]).ok().and_then(|s| s.parse().ok());
    (v, pos)
}

fn read_bool(s: &str, mut pos: usize) -> bool {
    let b = s.as_bytes();
    while pos < b.len() && (b[pos] == b' ' || b[pos] == b':' || b[pos] == b'\t') {
        pos += 1;
    }
    s.get(pos..pos + 4).map(|x| x == "true").unwrap_or(false)
}

/// Read a finite f64 (or scientific) value following `key` in `block`; `null` →
/// None.
fn field_f64(block: &str, key: &str) -> Option<f64> {
    let p = block.find(key)? + key.len();
    let rest = block[p..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ===========================================================================
// Audit table + report
// ===========================================================================

fn print_audit(
    results: &[ImgResult],
    args: &CorpusArgs,
    n_total: usize,
    n_kept: usize,
    n_rejected: usize,
    n_errors: usize,
) {
    println!(
        "\ncorpus audit — thresholds: edge_min={:.3} flat_max={:.2} spread_max={:.2} entropy_min={:.2}",
        args.edge_min, args.flat_max, args.spread_max, args.entropy_min
    );
    println!(
        "{:<38} {:>7} {:>7} {:>6} {:>6}  {}",
        "name", "edge", "spread", "flat", "entrpy", "verdict"
    );

    // Rejected rows (the audit-critical ones).
    println!("--- REJECTED ({n_rejected}) ---");
    for r in results.iter().filter(|r| !r.kept && r.error.is_none()) {
        if let Some(m) = r.metrics {
            println!(
                "{:<38} {:>7.4} {:>7.2} {:>6.2} {:>6.2}  reject: {}",
                trunc(&r.name, 38),
                m.edge_density,
                m.edge_spread,
                m.flat_fraction,
                m.color_entropy,
                r.reject_reasons.join("; ")
            );
        }
    }

    // Borderline kept (within 1.4× of any threshold) for sanity.
    println!("--- BORDERLINE KEPT (near a threshold) ---");
    let mut shown = 0;
    for r in results.iter().filter(|r| r.kept) {
        if let Some(m) = r.metrics {
            let near = m.edge_density < args.edge_min * 1.5
                || m.flat_fraction > args.flat_max * 0.85
                || m.edge_spread > args.spread_max * 0.85
                || m.color_entropy < args.entropy_min * 1.4;
            if near {
                println!(
                    "{:<38} {:>7.4} {:>7.2} {:>6.2} {:>6.2}  keep",
                    trunc(&r.name, 38),
                    m.edge_density,
                    m.edge_spread,
                    m.flat_fraction,
                    m.color_entropy
                );
                shown += 1;
                if shown >= 25 {
                    println!("  ... (more borderline kept; see {})", args.features_out);
                    break;
                }
            }
        }
    }
    if n_errors > 0 {
        println!("--- DECODE ERRORS ({n_errors}) ---");
        for r in results.iter().filter(|r| r.error.is_some()) {
            println!("{:<38} {}", trunc(&r.name, 38), r.error.as_deref().unwrap_or(""));
        }
    }

    // metric percentiles across all decoded images.
    let allm: Vec<Metrics> = results.iter().filter_map(|r| r.metrics).collect();
    let pct = |sel: &dyn Fn(&Metrics) -> f64, p: f64| -> f64 {
        let mut v: Vec<f64> = allm.iter().map(sel).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        psorted(&v, p)
    };
    println!("--- metric distribution (all {} decoded) ---", allm.len());
    for (name, sel) in [
        ("edge_density", &(|m: &Metrics| m.edge_density) as &dyn Fn(&Metrics) -> f64),
        ("edge_spread", &(|m: &Metrics| m.edge_spread) as &dyn Fn(&Metrics) -> f64),
        ("flat_fraction", &(|m: &Metrics| m.flat_fraction) as &dyn Fn(&Metrics) -> f64),
        ("color_entropy", &(|m: &Metrics| m.color_entropy) as &dyn Fn(&Metrics) -> f64),
    ] {
        println!(
            "  {:<14} p05={:.4} p25={:.4} p50={:.4} p75={:.4} p95={:.4}",
            name,
            pct(sel, 0.05),
            pct(sel, 0.25),
            pct(sel, 0.50),
            pct(sel, 0.75),
            pct(sel, 0.95)
        );
    }
    println!(
        "summary: {n_total} total, {n_kept} kept, {n_rejected} rejected, {n_errors} decode errors",
    );
}

fn report(
    bands: &Bands,
    agg: &Aggregate,
    label: Option<&LabelInfo>,
    args: &CorpusArgs,
    rejected_sheet: &Option<String>,
) {
    println!("\n=== bootstrap structural bands (search-native units) ===");
    let show = |name: &str, b: &StructBand| {
        println!(
            "  {:<14} band=[{:.4}, {:.4}] {}{}prov={} n_labels={} α={:.3}",
            name,
            b.lo,
            b.hi,
            b.reject_below.map(|r| format!("reject<{r:.4} ")).unwrap_or_default(),
            b.reject_above.map(|r| format!("reject>{r:.4} ")).unwrap_or_default(),
            b.provenance,
            b.n_labels,
            b.alpha,
        );
    };
    show("busyness", &bands.busyness);
    show("interior_frac", &bands.interior_frac);
    show("boundary", &bands.boundary);
    show("period", &bands.period);
    println!(
        "  (busyness proxy→native scale = {:.3}; these are WEAK priors pending labels)",
        agg.busyness_scale
    );

    println!("\n=== exact color targets (image-domain) ===");
    println!("  corpus palette (OKLab L,a,b @ weight):");
    for (lab, w) in &agg.palette {
        let srgb = oklab_to_srgb8(*lab);
        println!(
            "    L={:.3} a={:+.3} b={:+.3}  w={:.3}  ≈#{:02x}{:02x}{:02x}",
            lab[0], lab[1], lab[2], w, srgb[0], srgb[1], srgb[2]
        );
    }
    println!(
        "  chroma=[{:.3},{:.3}]  contrast=[{:.3},{:.3}]  luminance=[{:.3},{:.3}]",
        agg.chroma.0, agg.chroma.1, agg.contrast.0, agg.contrast.1, agg.luminance.0, agg.luminance.1
    );
    let dom = agg
        .hue_hist
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    println!(
        "  hue_hist (12 bins, chroma-weighted): [{}]  (dominant bin {} ≈ {}°)",
        agg.hue_hist.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "),
        dom,
        dom * 30
    );

    if let Some(li) = label {
        println!("\n=== label transition ===");
        println!(
            "  matched {} labeled node(s) in search.json ({} keep, {} discard); blend k={}",
            li.matched, li.n_keep, li.n_discard, args.blend_k
        );
        let alpha = if li.matched > 0 {
            li.matched as f64 / (li.matched as f64 + args.blend_k)
        } else {
            0.0
        };
        println!("  α = {}/({}+{}) = {:.3}", li.matched, li.matched, args.blend_k, alpha);
        println!(
            "  busyness: provenance now '{}' (band [{:.4},{:.4}])",
            bands.busyness.provenance, bands.busyness.lo, bands.busyness.hi
        );
        println!(
            "  period: provenance '{}' (band [{:.1},{:.1}]) — labels-only feature",
            bands.period.provenance, bands.period.lo, bands.period.hi
        );
        println!(
            "  interior_frac/boundary: stay bootstrap (search.json has no native channel for them yet)"
        );
    }

    println!("\nwrote {} (targets), {} (features){}",
        args.targets_out,
        args.features_out,
        match rejected_sheet {
            Some(p) => format!(", {p} (rejected sheet)"),
            None => String::new(),
        }
    );
}

// ===========================================================================
// small helpers
// ===========================================================================

fn split_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let t: String = s.chars().take(n - 1).collect();
        format!("{t}…")
    }
}

/// Percentile of an *already-sorted* slice (`p` in [0,1]).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Percentile of an *unsorted* slice (sorts a copy). For the small per-corpus
/// scalar vectors (≤ a few thousand) this is cheap.
fn psorted(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&s, p)
}

/// Letterbox `src` into a fixed `tw×th` dark canvas (aspect-preserved), so the
/// rejected sheet's tiles are uniform for [`compose_grid`].
fn fit_thumb(src: &RgbImage, tw: u32, th: u32) -> RgbImage {
    let (sw, sh) = (src.width() as f64, src.height() as f64);
    let scale = (tw as f64 / sw).min(th as f64 / sh);
    let nw = (sw * scale).round().max(1.0) as u32;
    let nh = (sh * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(src, nw, nh, FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(tw, th, Rgb([20, 20, 20]));
    let x0 = (tw - nw) / 2;
    let y0 = (th - nh) / 2;
    for (x, y, px) in resized.enumerate_pixels() {
        canvas.put_pixel(x0 + x, y0 + y, *px);
    }
    canvas
}

/// OKLab → sRGB8 for the human-readable palette swatch hex in the report.
fn oklab_to_srgb8(lab: [f64; 3]) -> [u8; 3] {
    use crate::palette::{linear_to_srgb, oklab_to_linear_srgb};
    let lin = oklab_to_linear_srgb(lab);
    [
        (linear_to_srgb(lin[0]) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        (linear_to_srgb(lin[1]) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        (linear_to_srgb(lin[2]) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
    ]
}
