//! `deband` — Phase-3 calibration of the off-nucleus de_px-band objective.
//!
//! The drive's objective (Prompt offnucleus-deband) maximizes **band proximity**
//! of a window's boundary distance `de_px` to a target center, demoting busyness
//! to a floor gate, then drifts the frame center off the minibrot nucleus toward
//! the in-band *decoration*. Before spending the drive's wall-clock budget we
//! validate the objective on a **known** frame (default: the P17 m6 frame the last
//! drive surfaced): does `de_px_win` actually separate the magenta decoration
//! cells from the white-circle nuclei?
//!
//! This subcommand re-renders that frame f64-clean, computes the per-cell reward
//! map ([`coherence::cell_reward_map`]), and emits three artifacts to
//! `out/coherence_cal/`:
//!  - a **`de_px_win` heatmap** (low de = nucleus, mid = decoration, high = flat),
//!  - a **band-membership mask overlay** on the shaded frame (in-band cells tinted
//!    magenta; the best contiguous in-band centroid — the drift target — circled),
//!  - a **split JSON**: a 1-D 3-means clustering of `log10(de_px_win)` into the
//!    nucleus / decoration / flat regimes, with each cluster's `de_px` and
//!    busyness, and the over-corrected constants those imply.
//!
//! It is a pure post-pass over one cached buffer; it never re-iterates and changes
//! no scoring. The drive (Phase 4) is the follow-up, gated on this eye test.

use std::fs;
use std::path::Path;

use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::Trap;
use crate::cli::{BackendChoice, DebandArgs};
use crate::coherence::{self, CellMap};
use crate::hp;
use crate::probe;
use crate::render;

/// Entry point for the `deband` subcommand (Phase-3 calibration).
pub fn run_deband(args: &DebandArgs) -> Result<(), String> {
    if args.frame_width <= 0.0 {
        return Err("--frame-width must be > 0".into());
    }
    if args.panel_width == 0 || args.target_width == 0 {
        return Err("--panel-width and --target-width must be > 0".into());
    }
    if args.supersample == 0 {
        return Err("--supersample must be > 0".into());
    }

    let panel_w = args.panel_width;
    let panel_h = ((panel_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample;

    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };

    let prec = hp::prec_bits(panel_w, args.frame_width);
    let center_re = hp::parse_decimal(&args.center_re, prec)?;
    let center_im = hp::parse_decimal(&args.center_im, prec)?;
    let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

    let target_spacing = args.frame_width / args.target_width as f64;
    eprintln!(
        "[{}] deband calibration: f64 {panel_w}x{panel_h} ss{ss}, center=({}, {}), \
         width={:.6e}, maxiter={}, K={}; target_width={} → target spacing={:.3e}",
        args.label, args.center_re, args.center_im, args.frame_width, args.maxiter,
        args.window, args.target_width, target_spacing,
    );

    let t0 = std::time::Instant::now();
    let panel = probe::render_mandel_panel(
        &center_re, &center_im, center_f64, args.frame_width, panel_w, panel_h, ss, args.maxiter,
        args.bailout, prec, trap, BackendChoice::F64,
    );
    assert_eq!(
        panel.backend_name, "F64",
        "deband calibration must stay f64 (the known frame is shallow)"
    );
    eprintln!("  iterated in {:.1}s", t0.elapsed().as_secs_f64());

    // The per-cell de_px-band reward map (the Phase-2 objective), under the
    // effective band params (calibrated consts unless overridden per-run).
    let bp = args.band_params();
    eprintln!(
        "  band params: center={:.3} reject_lo={:.3} reject_hi={:.3} busy_floor={:.4}",
        bp.band_center, bp.reject_lo, bp.reject_hi, bp.busy_floor,
    );
    let map = coherence::cell_reward_map(
        &panel.buf, panel_w, panel_h, args.window as i32, args.maxiter, args.frame_width,
        args.target_width, &bp,
    );

    // Three-way split of the de_px_win distribution.
    let split = three_way_split(&map);

    // Shaded base frame (same coloring as the drive panels, so the mask sits over
    // the image Matt judged).
    let palette = crate::palette_io::load_palette(
        &args.palette.palette,
        args.palette.palette_entry.as_deref(),
        args.palette.palette_reverse,
    )?;
    let params = probe::color_params(&args.shade);
    let shaded = render::shade_and_downsample(
        &panel.buf.samples, panel_w, panel_h, ss, &palette, &params, panel.spacing,
    );

    // Artifacts.
    let dir = Path::new(&args.out_dir);
    fs::create_dir_all(dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let heatmap = render_heatmap(&map, &split);
    let heatmap_path = dir.join(format!("{}_de_px_heatmap.png", args.label));
    heatmap
        .save(&heatmap_path)
        .map_err(|e| format!("failed to write {}: {e}", heatmap_path.display()))?;

    let centroid = coherence::best_inband_centroid(&map);
    let mask = render_mask_overlay(&shaded, &map, centroid);
    let mask_path = dir.join(format!("{}_band_mask.png", args.label));
    mask.save(&mask_path)
        .map_err(|e| format!("failed to write {}: {e}", mask_path.display()))?;

    let json = build_json(args, &map, &split, centroid, target_spacing, &bp);
    let json_path = dir.join(format!("{}_split.json", args.label));
    fs::write(&json_path, json).map_err(|e| format!("failed to write {}: {e}", json_path.display()))?;

    // ---- report (data only) ----
    report(args, &map, &split, centroid);

    eprintln!(
        "  wrote {} (heatmap), {} (mask), {} (split)",
        probe::path_str(&heatmap_path),
        probe::path_str(&mask_path),
        probe::path_str(&json_path),
    );
    Ok(())
}

// ===========================================================================
// Three-way split: 1-D 3-means over log10(de_px_win)
// ===========================================================================

/// One cluster of the `de_px_win` split.
struct Cluster {
    /// Cells assigned here.
    count: usize,
    /// `de_px_win` percentiles (p10, p50, p90) over the cluster's cells.
    de_px: (f64, f64, f64),
    /// Busyness percentiles (p10, p50, p90) over the cluster's cells.
    busy: (f64, f64, f64),
}

/// The split + the over-corrected constants it implies.
struct Split {
    /// Valid (≥ min-escaped) window cells used.
    n_valid: usize,
    /// Sorted by `de_px` median: `[nucleus, decoration, flat]`.
    clusters: [Cluster; 3],
    // ---- recommended (over-corrected) consts ----
    band_center: f64,
    reject_lo: f64,
    reject_hi: f64,
    busy_floor: f64,
}

/// Percentile of an already-sorted slice (nearest-rank, clamped).
fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((q * (sorted.len() as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// 1-D 3-means on `log10(de_px_win)` over valid cells, deterministically seeded at
/// the 16/50/84 percentiles. Returns the three clusters sorted by `de_px` median
/// (nucleus → decoration → flat) plus the over-corrected constants they imply.
fn three_way_split(map: &CellMap) -> Split {
    // Collect (de_px, busy, log_de) for valid cells.
    let mut items: Vec<(f64, f64, f64)> = Vec::new();
    for i in 0..map.de_px.len() {
        let d = map.de_px[i];
        if d.is_finite() && d > 0.0 {
            items.push((d, map.busy[i], d.log10()));
        }
    }
    let n_valid = items.len();

    // Seed centroids at percentiles of log_de.
    let mut logs: Vec<f64> = items.iter().map(|t| t.2).collect();
    logs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut centers = [pct(&logs, 0.16), pct(&logs, 0.50), pct(&logs, 0.84)];

    // Lloyd iterations.
    let mut assign = vec![0usize; n_valid];
    for _ in 0..50 {
        let mut changed = false;
        for (k, it) in items.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (c, &cen) in centers.iter().enumerate() {
                let dd = (it.2 - cen).abs();
                if dd < best_d {
                    best_d = dd;
                    best = c;
                }
            }
            if assign[k] != best {
                changed = true;
            }
            assign[k] = best;
        }
        let mut sum = [0.0f64; 3];
        let mut cnt = [0usize; 3];
        for (k, it) in items.iter().enumerate() {
            sum[assign[k]] += it.2;
            cnt[assign[k]] += 1;
        }
        for c in 0..3 {
            if cnt[c] > 0 {
                centers[c] = sum[c] / cnt[c] as f64;
            }
        }
        if !changed {
            break;
        }
    }

    // Build clusters, then sort by de_px median ascending.
    let mut groups: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 3]; // (de_px, busy)
    for (k, it) in items.iter().enumerate() {
        groups[assign[k]].push((it.0, it.1));
    }
    let mut built: Vec<Cluster> = groups
        .into_iter()
        .map(|mut g| {
            let mut de: Vec<f64> = g.iter().map(|x| x.0).collect();
            let mut bz: Vec<f64> = g.iter().map(|x| x.1).collect();
            de.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            bz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            g.clear();
            Cluster {
                count: de.len(),
                de_px: (pct(&de, 0.10), pct(&de, 0.50), pct(&de, 0.90)),
                busy: (pct(&bz, 0.10), pct(&bz, 0.50), pct(&bz, 0.90)),
            }
        })
        .collect();
    built.sort_by(|a, b| a.de_px.1.partial_cmp(&b.de_px.1).unwrap_or(std::cmp::Ordering::Equal));
    let clusters: [Cluster; 3] = [
        built.remove(0),
        built.remove(0),
        built.remove(0),
    ];

    // Over-corrected consts from the split:
    //  - band center = decoration median × 1.5 (push off the boundary).
    //  - reject_lo  = nucleus cluster p90 + margin (hard-exclude the white nuclei).
    //  - reject_hi  = geometric mean of decoration p90 and flat p50 (cut the flat).
    //  - busy_floor = decoration busy p10 (drop smoother-than-decoration cells).
    let nucleus = &clusters[0];
    let decoration = &clusters[1];
    let flat = &clusters[2];
    let band_center = decoration.de_px.1 * 1.5;
    let reject_lo = (nucleus.de_px.2 * 1.25).max(1.0);
    let reject_hi = (decoration.de_px.2 * flat.de_px.1).sqrt();
    let busy_floor = decoration.busy.0;

    Split {
        n_valid,
        clusters,
        band_center,
        reject_lo,
        reject_hi,
        busy_floor,
    }
}

// ===========================================================================
// Visualization
// ===========================================================================

/// Five-stop sRGB ramp over `t∈[0,1]`: dark-purple → blue → green → yellow → red.
/// Low `de` (nucleus) reads cool, the decoration band warm, the flat exterior hot.
fn ramp(t: f64) -> Rgb<u8> {
    const STOPS: [(f64, [f64; 3]); 5] = [
        (0.00, [48.0, 18.0, 86.0]),
        (0.25, [33.0, 120.0, 180.0]),
        (0.50, [40.0, 180.0, 90.0]),
        (0.75, [240.0, 200.0, 40.0]),
        (1.00, [200.0, 40.0, 40.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut i = 0;
    while i + 1 < STOPS.len() && t > STOPS[i + 1].0 {
        i += 1;
    }
    let (t0, c0) = STOPS[i];
    let (t1, c1) = STOPS[(i + 1).min(STOPS.len() - 1)];
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
    Rgb([
        (c0[0] + (c1[0] - c0[0]) * f) as u8,
        (c0[1] + (c1[1] - c0[1]) * f) as u8,
        (c0[2] + (c1[2] - c0[2]) * f) as u8,
    ])
}

/// `de_px_win` heatmap: each cell colored by `log10(de_px_win)` linearly mapped
/// over the observed valid range (so all three regimes are visible). Invalid
/// (too-few-escaped) cells are dark gray. The cell grid is the output resolution.
fn render_heatmap(map: &CellMap, split: &Split) -> RgbImage {
    // Observed log range over valid cells (fall back to the reject window).
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &d in &map.de_px {
        if d.is_finite() && d > 0.0 {
            lo = lo.min(d.log10());
            hi = hi.max(d.log10());
        }
    }
    if !lo.is_finite() || !(hi > lo) {
        lo = split.reject_lo.max(1e-3).log10();
        hi = split.reject_hi.max(lo.exp2() * 10.0).log10();
    }
    let inv = 1.0 / (hi - lo);
    let mut img = RgbImage::from_pixel(map.w as u32, map.h as u32, Rgb([34, 34, 34]));
    for row in 0..map.h {
        for col in 0..map.w {
            let d = map.de_px[row * map.w + col];
            if d.is_finite() && d > 0.0 {
                let t = (d.log10() - lo) * inv;
                img.put_pixel(col as u32, row as u32, ramp(t));
            }
        }
    }
    img
}

/// Band-membership mask overlay: in-band cells tinted magenta over the shaded
/// frame, the best contiguous in-band centroid (the drift target) circled white.
fn render_mask_overlay(
    shaded: &RgbImage,
    map: &CellMap,
    centroid: Option<(f64, f64, f64)>,
) -> RgbImage {
    let mut img = shaded.clone();
    let magenta = [255.0f64, 0.0, 255.0];
    let alpha = 0.45;
    for row in 0..map.h {
        for col in 0..map.w {
            if map.in_band[row * map.w + col] {
                let p = img.get_pixel_mut(col as u32, row as u32);
                for k in 0..3 {
                    p[k] = ((p[k] as f64) * (1.0 - alpha) + magenta[k] * alpha) as u8;
                }
            }
        }
    }
    if let Some((cx, cy, _)) = centroid {
        probe::draw_circle(&mut img, cx, cy, 6.0);
    }
    img
}

// ===========================================================================
// Report + JSON
// ===========================================================================

fn report(args: &DebandArgs, map: &CellMap, split: &Split, centroid: Option<(f64, f64, f64)>) {
    let names = ["nucleus", "decoration", "flat   "];
    let in_band = map.in_band.iter().filter(|&&b| b).count();
    println!(
        "DEBAND  label={}  valid_cells={}  in_band_cells={}  K={}",
        args.label, split.n_valid, in_band, map.k,
    );
    println!(
        "  three-way de_px_win split ({} valid windowed cells):",
        split.n_valid
    );
    for (i, c) in split.clusters.iter().enumerate() {
        println!(
            "    {:<10} n={:>5}  de_px[p10/p50/p90]={:>7.3}/{:>7.3}/{:>7.3}  busy[p10/p50/p90]={:.3}/{:.3}/{:.3}",
            names[i], c.count, c.de_px.0, c.de_px.1, c.de_px.2, c.busy.0, c.busy.1, c.busy.2,
        );
    }
    println!(
        "  recommended (over-corrected) consts: BAND_CENTER={:.3}  REJECT_LO={:.3}  REJECT_HI={:.3}  BUSY_FLOOR={:.3}",
        split.band_center, split.reject_lo, split.reject_hi, split.busy_floor,
    );
    // Regression-relevant: a sample of the in-band reward at the centroid vs. the
    // nucleus-cluster cells (the flip's intent — decoration must beat nucleus).
    match centroid {
        Some((cx, cy, sr)) => println!(
            "  best in-band centroid at cell ({:.1}, {:.1}), summed reward {:.2} (the drift target)",
            cx, cy, sr
        ),
        None => println!("  NO in-band cell — the band did not capture any decoration on this frame"),
    }
    // The gate verdict the prompt asks for.
    let dec = &split.clusters[1];
    let nuc = &split.clusters[0];
    let separates = nuc.de_px.2 < dec.de_px.0 || (dec.de_px.1 / nuc.de_px.1.max(1e-9)) > 2.0;
    println!(
        "  SEPARATION: nucleus p90 de_px={:.3} vs decoration p10 de_px={:.3} → {}",
        nuc.de_px.2,
        dec.de_px.0,
        if separates { "SEPARATES (proceed to Phase 4 pending eye test)" } else { "DOES NOT separate — STOP" },
    );
}

fn build_json(
    args: &DebandArgs,
    map: &CellMap,
    split: &Split,
    centroid: Option<(f64, f64, f64)>,
    target_spacing: f64,
    bp: &coherence::BandParams,
) -> String {
    use probe::{jf, js};
    let names = ["nucleus", "decoration", "flat"];
    let in_band = map.in_band.iter().filter(|&&b| b).count();
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"label\": {},\n", js(&args.label)));
    s.push_str(&format!(
        "  \"center\": {{ \"re\": {}, \"im\": {} }},\n",
        js(&args.center_re),
        js(&args.center_im)
    ));
    s.push_str(&format!("  \"frame_width\": {},\n", jf(args.frame_width)));
    s.push_str(&format!("  \"maxiter\": {},\n", args.maxiter));
    s.push_str(&format!("  \"panel_width\": {},\n", args.panel_width));
    s.push_str(&format!("  \"target_width\": {},\n", args.target_width));
    s.push_str(&format!("  \"target_spacing\": {},\n", jf(target_spacing)));
    s.push_str(&format!("  \"window_k\": {},\n", map.k));
    s.push_str(&format!("  \"valid_cells\": {},\n", split.n_valid));
    s.push_str(&format!("  \"in_band_cells\": {},\n", in_band));
    s.push_str("  \"clusters\": [\n");
    for (i, c) in split.clusters.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"name\": {},\n", js(names[i])));
        s.push_str(&format!("      \"count\": {},\n", c.count));
        s.push_str(&format!(
            "      \"de_px\": {{ \"p10\": {}, \"p50\": {}, \"p90\": {} }},\n",
            jf(c.de_px.0),
            jf(c.de_px.1),
            jf(c.de_px.2)
        ));
        s.push_str(&format!(
            "      \"busy\": {{ \"p10\": {}, \"p50\": {}, \"p90\": {} }}\n",
            jf(c.busy.0),
            jf(c.busy.1),
            jf(c.busy.2)
        ));
        s.push_str("    }");
        if i + 1 < split.clusters.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ],\n");
    s.push_str("  \"recommended_consts\": {\n");
    s.push_str(&format!("    \"DE_PX_BAND_CENTER\": {},\n", jf(split.band_center)));
    s.push_str(&format!("    \"DE_PX_REJECT_LO\": {},\n", jf(split.reject_lo)));
    s.push_str(&format!("    \"DE_PX_REJECT_HI\": {},\n", jf(split.reject_hi)));
    s.push_str(&format!("    \"BUSY_FLOOR\": {}\n", jf(split.busy_floor)));
    s.push_str("  },\n");
    s.push_str("  \"active_consts\": {\n");
    s.push_str(&format!("    \"DE_PX_BAND_CENTER\": {},\n", jf(bp.band_center)));
    s.push_str(&format!("    \"DE_PX_REJECT_LO\": {},\n", jf(bp.reject_lo)));
    s.push_str(&format!("    \"DE_PX_REJECT_HI\": {},\n", jf(bp.reject_hi)));
    s.push_str(&format!("    \"BUSY_FLOOR\": {}\n", jf(bp.busy_floor)));
    s.push_str("  },\n");
    match centroid {
        Some((cx, cy, sr)) => s.push_str(&format!(
            "  \"best_centroid\": {{ \"col\": {}, \"row\": {}, \"summed_reward\": {} }}\n",
            jf(cx),
            jf(cy),
            jf(sr)
        )),
        None => s.push_str("  \"best_centroid\": null\n"),
    }
    s.push_str("}\n");
    s
}
