//! `energy.rs` — multi-scale edge-energy histogram metric (Prompt corpus-energy
//! -calibration). Calibration + eye-check only: no descent, no candidate scoring
//! beyond the buffet eye-check.
//!
//! The metric is **one pixel-space function**, applied *identically* to a corpus
//! PNG and to a rendered candidate. It reads only RGB — never any fractal
//! internal — so a finished wallpaper and a freshly rendered frame are compared
//! on the same footing. This is what lets it eventually rank generated frames
//! against the ~700-wallpaper corpus.
//!
//! Per image:
//!  1. **Canonical resolution.** Center-crop to 16:9, resize to [`WORK_W`]×
//!     [`WORK_H`] (Triangle), so per-pixel edge magnitudes are comparable across
//!     the corpus's mixed native resolutions. Letterbox black is cropped away
//!     before resize so it can't skew the energy.
//!  2. **Per-pixel edge energy** at full canonical res: the OKLab-image gradient
//!     magnitude (forward-difference neighbor ΔE in OKLab). Computed **once**.
//!  3. **Multi-scale region pooling.** The full-res energy is averaged into
//!     fractional-region grids at four scales ([`SCALE_GRID`] = 16×16 → 2×2),
//!     energy **per unit area**. Crucially the coarse scales pool the *fine* edge
//!     map — the image is never re-downsampled (that would destroy the fine edges
//!     the coarse scales must still see).
//!  4. **Histogram per scale** = the distribution of that scale's region
//!     energies, binned under frozen **equal-count (quantile)** edges fitted on
//!     the whole corpus per scale. Four histograms per image — the *signature*.
//!
//! Distance between two signatures = **per-scale 1-D EMD, summed across the four
//! scales** (Wasserstein-1 on equally spaced bin positions = Σ|CDF₁−CDF₂|),
//! equal weight by default (exposed via `--weights`). Quantile bins are defined
//! by the corpus's energy range, so a candidate busier than anything in the
//! corpus saturates the top bin — acceptable for a region-finder (off-distribution
//! *should* score poorly), flagged in the report.
//!
//! Everything is pure-Rust and dependency-light per the project ethos (k-means
//! and EMD are both trivial; JSON is hand-rolled). The candidate render path
//! reuses the f64 cheap-regime panel renderer.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use astro_float::BigFloat;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{Trap, TrapShape};
use clap::Args;
use crate::cli::BackendChoice;
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::hp;
use crate::palette::{builtin, srgb8_to_oklab, Palette};
use crate::probe::{self, jf, js, SplitMix64};
use crate::render;
use crate::sheet::compose_grid;

/// Canonical working resolution (16:9). All edge magnitudes are computed here.
pub const WORK_W: u32 = 2560;
pub const WORK_H: u32 = 1440;
/// Region-grid side length per scale: 16×16 → 8×8 → 4×4 → 2×2 cells.
pub const SCALE_GRID: [usize; 4] = [16, 8, 4, 2];
/// Persistent calibration store (NOT under `out/`, so it survives `out/` clears).
/// The frozen quantile bins are part of the metric — regenerating risks silent
/// drift if the corpus folder changed, so the artifact lives here, committed.
pub const CALIBRATION_DIR: &str = "data/calibration";
/// The load-bearing calibration artifact: frozen bins + per-image histograms.
pub const ARTIFACT_PATH: &str = "data/calibration/energy_calibration.json";
/// Histogram bins per scale (equal-count quantile bins, frozen on the corpus).
pub const NBINS: usize = 12;
/// Recognized corpus image extensions (lower-cased).
const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp"];

// ===========================================================================
// The metric (identical on corpus and candidate)
// ===========================================================================

/// An image's four per-scale region-energy vectors (`SCALE_GRID[i]²` values each).
pub type Regions = [Vec<f64>; 4];

/// Center-crop `img` to 16:9, resize to the canonical resolution, return its
/// OKLab pixels row-major (`WORK_W·WORK_H`).
fn canonical_oklab(img: &RgbImage) -> Vec<[f64; 3]> {
    let cropped = center_crop_16x9(img);
    let resized = image::imageops::resize(&cropped, WORK_W, WORK_H, FilterType::Triangle);
    resized
        .pixels()
        .map(|p| srgb8_to_oklab([p[0], p[1], p[2]]))
        .collect()
}

/// Center-crop to the widest 16:9 window that fits, dropping letterbox bars.
fn center_crop_16x9(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let target = 16.0 / 9.0;
    let cur = w as f64 / h as f64;
    let (cw, ch) = if cur > target {
        ((h as f64 * target).round() as u32, h) // too wide → trim width
    } else {
        (w, (w as f64 / target).round() as u32) // too tall → trim height
    };
    let cw = cw.clamp(1, w);
    let ch = ch.clamp(1, h);
    let x0 = (w - cw) / 2;
    let y0 = (h - ch) / 2;
    image::imageops::crop_imm(img, x0, y0, cw, ch).to_image()
}

/// OKLab Euclidean distance.
#[inline]
fn de(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Per-pixel edge energy = OKLab gradient magnitude (forward-difference neighbor
/// ΔE). Last row/column have no forward neighbor → that axis contributes 0.
fn edge_energy(oklab: &[[f64; 3]], w: usize, h: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let c = &oklab[y * w + x];
            let gx = if x + 1 < w { de(c, &oklab[y * w + x + 1]) } else { 0.0 };
            let gy = if y + 1 < h { de(c, &oklab[(y + 1) * w + x]) } else { 0.0 };
            e[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    e
}

/// Average the full-res energy into an `s×s` region grid (energy per unit area),
/// regions as frame fractions. Returns `s·s` row-major region energies.
fn pool(e: &[f64], w: usize, h: usize, s: usize) -> Vec<f64> {
    let mut sum = vec![0.0f64; s * s];
    let mut cnt = vec![0u32; s * s];
    for y in 0..h {
        let gy = (y * s) / h;
        for x in 0..w {
            let gx = (x * s) / w;
            let gi = gy * s + gx;
            sum[gi] += e[y * w + x];
            cnt[gi] += 1;
        }
    }
    sum.iter()
        .zip(&cnt)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect()
}

/// The whole metric front half: image → four region-energy vectors.
pub fn region_energies(img: &RgbImage) -> Regions {
    let oklab = canonical_oklab(img);
    let e = edge_energy(&oklab, WORK_W as usize, WORK_H as usize);
    std::array::from_fn(|i| pool(&e, WORK_W as usize, WORK_H as usize, SCALE_GRID[i]))
}

// ===========================================================================
// Detail-occupancy gate (loose0 calibration; port of score_complexity.py)
// ===========================================================================
//
// The corpus EMD descriptor rewards sparseness (the wrong objective for a
// sparse-reject floor — see the `energy-metric-nn-rewards-sparse` finding), so
// the loose0 floor is a *different* reduction of the **same** `edge_energy`
// primitive: occupancy. Crucially it runs on the image at its **native
// resolution** — NOT the `WORK_W×WORK_H` canonical resize `region_energies`
// uses — because the Python calibration that chose the 0.23 floor scored the raw
// 1280×720 crops. Tile `gx×gy`, take each tile's MEAN edge energy, occupancy =
// fraction of tiles whose mean exceeds `floor`. A smooth gradient or corner-only
// detail occupies few tiles; a dense filament field occupies most.

/// Tile grid for the occupancy reduction (1280/32 = 720/18 = 40px tiles).
pub const OCC_GX: usize = 32;
pub const OCC_GY: usize = 18;
/// Per-tile edge-energy floor (OKLab ΔE/px tile mean) that defines "occupied".
/// This is the calibration's chosen floor — NOT the gate threshold on the
/// resulting occupancy fraction (that is `present`'s `--occupancy-floor`).
pub const OCC_FLOOR: f64 = 0.010;

/// Detail occupancy of a colored image at its native resolution: fraction of the
/// `gx×gy` tiles whose mean `edge_energy` exceeds `floor`. Ragged remainder
/// rows/cols (`w % gx`, `h % gy`) are dropped, matching the numpy `reshape`
/// the Python scorer used. Byte-for-byte the same primitive as `edge_energy` /
/// `srgb8_to_oklab`, so it reproduces `complexity_scores.json`.
pub fn occupancy(img: &RgbImage, gx: usize, gy: usize, floor: f64) -> f64 {
    let w = img.width() as usize;
    let h = img.height() as usize;
    if w == 0 || h == 0 || gx == 0 || gy == 0 {
        return 0.0;
    }
    let oklab: Vec<[f64; 3]> =
        img.pixels().map(|p| srgb8_to_oklab([p[0], p[1], p[2]])).collect();
    let e = edge_energy(&oklab, w, h);
    let tw = w / gx; // tile width  (px)
    let th = h / gy; // tile height (px)
    if tw == 0 || th == 0 {
        return 0.0;
    }
    let mut occupied = 0usize;
    for ty in 0..gy {
        for tx in 0..gx {
            let mut sum = 0.0f64;
            for yy in 0..th {
                let row = (ty * th + yy) * w + tx * tw;
                for xx in 0..tw {
                    sum += e[row + xx];
                }
            }
            if sum / (tw * th) as f64 > floor {
                occupied += 1;
            }
        }
    }
    occupied as f64 / (gx * gy) as f64
}

/// Per-tile **mean** edge energy of a colored image at its native resolution:
/// the `gx*gy` row-major tile means, computed with the exact same
/// `srgb8_to_oklab`/`edge_energy`/tiling primitive as [`occupancy`] — only the
/// floor reduction is dropped. Used by `present`'s content-centered focus, which
/// takes the energy-weighted centroid of this grid. Degenerate tilings return an
/// all-zero `gx*gy` grid (or empty when `gx*gy == 0`).
pub fn tile_energy(img: &RgbImage, gx: usize, gy: usize) -> Vec<f64> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    if w == 0 || h == 0 || gx == 0 || gy == 0 {
        return vec![0.0; gx * gy];
    }
    let oklab: Vec<[f64; 3]> =
        img.pixels().map(|p| srgb8_to_oklab([p[0], p[1], p[2]])).collect();
    let e = edge_energy(&oklab, w, h);
    let tw = w / gx;
    let th = h / gy;
    if tw == 0 || th == 0 {
        return vec![0.0; gx * gy];
    }
    let mut means = vec![0.0f64; gx * gy];
    for ty in 0..gy {
        for tx in 0..gx {
            let mut sum = 0.0f64;
            for yy in 0..th {
                let row = (ty * th + yy) * w + tx * tw;
                for xx in 0..tw {
                    sum += e[row + xx];
                }
            }
            means[ty * gx + tx] = sum / (tw * th) as f64;
        }
    }
    means
}

/// Bin a region value under quantile `edges` (`NBINS+1` of them); below `edges[0]`
/// → bin 0, at/above the top edge → the last bin.
#[inline]
fn bin_of(v: f64, edges: &[f64]) -> usize {
    let nbins = edges.len() - 1;
    for b in 0..nbins {
        if v < edges[b + 1] {
            return b;
        }
    }
    nbins - 1
}

/// Bin region values into a normalized histogram (sums to 1) under `edges`.
fn histogram(values: &[f64], edges: &[f64]) -> Vec<f64> {
    let nbins = edges.len() - 1;
    let mut h = vec![0.0f64; nbins];
    for &v in values {
        h[bin_of(v, edges)] += 1.0;
    }
    let tot: f64 = h.iter().sum();
    if tot > 0.0 {
        for x in h.iter_mut() {
            *x /= tot;
        }
    }
    h
}

/// Equal-count (quantile) bin edges over `values` (consumed/sorted in place):
/// `nbins+1` edges at quantiles `0, 1/n, … 1`, nudged to be strictly increasing.
fn quantile_edges(values: &mut [f64], nbins: usize) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 0 {
        return vec![0.0; nbins + 1];
    }
    let mut edges = Vec::with_capacity(nbins + 1);
    for b in 0..=nbins {
        let q = b as f64 / nbins as f64;
        let idx = ((q * (n - 1) as f64).round() as usize).min(n - 1);
        edges.push(values[idx]);
    }
    for i in 1..edges.len() {
        if edges[i] <= edges[i - 1] {
            let bump = edges[i - 1].abs().max(1.0) * 1e-9;
            edges[i] = edges[i - 1] + bump;
        }
    }
    edges
}

/// Frozen per-scale quantile edges — part of the metric definition.
pub struct FrozenBins {
    /// `edges[scale]` has `NBINS+1` entries.
    pub edges: [Vec<f64>; 4],
}

impl FrozenBins {
    /// Fit equal-count bins per scale across every image's region energies.
    fn fit(all: &[Regions]) -> FrozenBins {
        let edges = std::array::from_fn(|s| {
            let mut pooled: Vec<f64> = Vec::new();
            for r in all {
                pooled.extend_from_slice(&r[s]);
            }
            quantile_edges(&mut pooled, NBINS)
        });
        FrozenBins { edges }
    }

    /// Bin one image's region energies into its 4-scale signature.
    pub(crate) fn signature(&self, regions: &Regions) -> Signature {
        let hist = std::array::from_fn(|s| histogram(&regions[s], &self.edges[s]));
        Signature { hist }
    }
}

/// An image's calibrated signature: one normalized histogram per scale.
#[derive(Clone)]
pub struct Signature {
    pub hist: [Vec<f64>; 4],
}

impl Signature {
    /// Concatenate the four histograms into one `4·NBINS` feature vector.
    fn concat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(4 * NBINS);
        for s in 0..4 {
            v.extend_from_slice(&self.hist[s]);
        }
        v
    }
}

/// 1-D EMD (Wasserstein-1) between two normalized histograms on equally spaced
/// bin positions: Σ |CDF_a − CDF_b|.
fn emd1d(a: &[f64], b: &[f64]) -> f64 {
    let mut ca = 0.0;
    let mut cb = 0.0;
    let mut acc = 0.0;
    for i in 0..a.len() {
        ca += a[i];
        cb += b[i];
        acc += (ca - cb).abs();
    }
    acc
}

/// Per-scale EMD summed across scales, weighted.
pub(crate) fn distance(x: &Signature, y: &Signature, w: &[f64; 4]) -> f64 {
    (0..4).map(|s| w[s] * emd1d(&x.hist[s], &y.hist[s])).sum()
}

// ===========================================================================
// Corpus image records
// ===========================================================================

struct CorpusImg {
    name: String,
    path: PathBuf,
    regions: Regions,
    sig: Option<Signature>, // filled after the bins are frozen
}

// ===========================================================================
// Entry point
// ===========================================================================

pub fn run_calibrate(args: &CalibrateArgs) -> Result<(), String> {
    let dir = Path::new(&args.dir);
    if !dir.is_dir() {
        return Err(format!("--dir '{}' is not a directory", args.dir));
    }
    let weights = args.resolved_weights()?;

    // ---- enumerate corpus images (top level only) ----
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", args.dir))? {
        let p = entry.map_err(|e| format!("dir entry: {e}"))?.path();
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
    if files.is_empty() {
        return Err(format!("no images ({IMAGE_EXTS:?}) at top level of {}", args.dir));
    }
    eprintln!(
        "calibrate: {} corpus image(s) — computing region energies at {WORK_W}x{WORK_H} \
         (rayon-parallel; this is the slow stage) ...",
        files.len()
    );

    // ---- pass 0: per-image region energies (parallel) ----
    let t0 = std::time::Instant::now();
    let mut imgs: Vec<CorpusImg> = files
        .par_iter()
        .filter_map(|p| {
            let img = image::open(p).ok()?.to_rgb8();
            if img.width() < 16 || img.height() < 16 {
                return None;
            }
            Some(CorpusImg {
                name: p.file_name()?.to_string_lossy().into_owned(),
                path: p.clone(),
                regions: region_energies(&img),
                sig: None,
            })
        })
        .collect();
    imgs.sort_by(|a, b| a.name.cmp(&b.name));
    let n_ok = imgs.len();
    let n_err = files.len() - n_ok;
    eprintln!(
        "  {} image(s) processed ({} decode/size skips) in {:.1}s",
        n_ok,
        n_err,
        t0.elapsed().as_secs_f64()
    );

    // ---- pass 1: freeze equal-count bins per scale ----
    let regions_only: Vec<Regions> = imgs.iter().map(|i| i.regions.clone()).collect();
    let bins = FrozenBins::fit(&regions_only);

    // ---- pass 2: each image's signature under frozen edges ----
    for im in imgs.iter_mut() {
        im.sig = Some(bins.signature(&im.regions));
    }

    fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("failed to create {}: {e}", args.out_dir))?;

    // ---- calibration artifact (frozen edges + per-image histograms) ----
    // Persisted OUTSIDE out/ (ARTIFACT_PATH): the frozen bins are part of the
    // metric and must survive `out/` clears. Only the view sheets stay in out_dir.
    let artifact = build_artifact_json(&bins, &imgs, args);
    let artifact_path = ARTIFACT_PATH.to_string();
    crate::ensure_parent_dir(&artifact_path)?;
    fs::write(&artifact_path, artifact)
        .map_err(|e| format!("failed to write {artifact_path}: {e}"))?;

    // ---- eye-check 1: corpus-internal NN pairs ----
    let nn_path = nn_pair_sheet(&imgs, &weights, args)?;

    // ---- eye-check 2: buffet DEEP ranking ----
    let buffet_report = buffet_ranking(&imgs, &bins, &weights, args)?;

    // ---- phase 5: corpus structure (k-means archetypes) ----
    let cluster_path = if args.clusters >= 2 {
        Some(cluster_exemplars(&imgs, args)?)
    } else {
        None
    };

    // ---- report ----
    report(&bins, &imgs, &buffet_report, &artifact_path, &nn_path, &cluster_path, args, n_err);
    Ok(())
}

// ===========================================================================
// Eye-check 1 — corpus-internal nearest-neighbour pairs
// ===========================================================================

fn nn_pair_sheet(
    imgs: &[CorpusImg],
    weights: &[f64; 4],
    args: &CalibrateArgs,
) -> Result<String, String> {
    let n = imgs.len();
    if n < 2 {
        return Ok(String::new());
    }
    // Deterministic spread of sample indices across the (name-sorted) corpus.
    let m = args.nn_samples.min(n);
    let sample: Vec<usize> = (0..m).map(|k| (k * n) / m).collect();

    let sigs: Vec<&Signature> = imgs.iter().map(|i| i.sig.as_ref().unwrap()).collect();
    let mut tiles: Vec<RgbImage> = Vec::with_capacity(m * 2);
    for &i in &sample {
        // nearest other image by summed EMD
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = distance(sigs[i], sigs[j], weights);
            if d < bd {
                bd = d;
                best = j;
            }
        }
        let mut ta = thumb(&imgs[i].path, args.thumb_width)?;
        let mut tb = thumb(&imgs[best].path, args.thumb_width)?;
        label(&mut ta, &format!("{}", short(&imgs[i].name)));
        label(&mut tb, &format!("NN d={bd:.2} {}", short(&imgs[best].name)));
        tiles.push(ta);
        tiles.push(tb);
    }
    let grid = compose_grid(&tiles, Some(2));
    let path = format!("{}/nn_pairs.png", args.out_dir.trim_end_matches('/'));
    grid.save(&path).map_err(|e| format!("failed to write {path}: {e}"))?;
    Ok(path)
}

// ===========================================================================
// Eye-check 2 — buffet DEEP tile ranking against the frozen corpus
// ===========================================================================

struct ScoredTile {
    id: String,
    loc: String,    // e.g. "B1"
    knn: f64,        // mean EMD to the k nearest corpus images
    nearest: usize,  // index into imgs of the single closest corpus image
    nearest_d: f64,
    saturated: bool, // any scale's top bin carries mass (off-distribution flag)
}

struct BuffetReport {
    tiles: Vec<ScoredTile>,
    loc_score: Vec<(String, f64)>, // per-location mean knn, ascending (best first)
    okay_above_sparse: Option<bool>,
    sheet: Option<String>,
}

fn buffet_ranking(
    imgs: &[CorpusImg],
    bins: &FrozenBins,
    weights: &[f64; 4],
    args: &CalibrateArgs,
) -> Result<BuffetReport, String> {
    let empty = BuffetReport {
        tiles: Vec::new(),
        loc_score: Vec::new(),
        okay_above_sparse: None,
        sheet: None,
    };
    let text = match fs::read_to_string(&args.buffet_json) {
        Ok(t) => t,
        Err(_) => {
            eprintln!(
                "  (no buffet json at {}; skipping the buffet eye-check)",
                args.buffet_json
            );
            return Ok(empty);
        }
    };
    let deep = parse_buffet_deep_b(&text);
    if deep.is_empty() {
        eprintln!("  (no source-B DEEP tiles in {}; skipping)", args.buffet_json);
        return Ok(empty);
    }

    let palette = builtin("default", false).expect("default palette");
    let params = default_color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };
    let k = args.knn.max(1);
    let sigs: Vec<&Signature> = imgs.iter().map(|i| i.sig.as_ref().unwrap()).collect();

    let mut scored: Vec<ScoredTile> = Vec::with_capacity(deep.len());
    let mut tiles_for_sheet: Vec<RgbImage> = Vec::with_capacity(deep.len() * 2);
    eprintln!("  rendering + scoring {} source-B DEEP tile(s) ...", deep.len());
    for t in &deep {
        let cand = render_candidate(
            t.center,
            t.width,
            t.maxiter,
            args.candidate_width,
            args.supersample,
            trap,
            &palette,
            &params,
        );
        let regions = region_energies(&cand);
        let sig = bins.signature(&regions);
        let saturated = (0..4).any(|s| *sig.hist[s].last().unwrap() > 0.0);

        // distances to every corpus image; nearest + mean of k smallest.
        let mut ds: Vec<(f64, usize)> = (0..imgs.len())
            .map(|j| (distance(&sig, sigs[j], weights), j))
            .collect();
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let knn = ds[..k.min(ds.len())].iter().map(|x| x.0).sum::<f64>() / k.min(ds.len()) as f64;
        let (nearest_d, nearest) = ds[0];
        let loc = t.id.split('_').next().unwrap_or(&t.id).to_string();

        // sheet row: rendered candidate | nearest corpus image
        let mut ct = fit_to(&cand, args.thumb_width);
        let mut nt = thumb(&imgs[nearest].path, args.thumb_width)?;
        label(&mut ct, &format!("{} knn{:.2}", t.id, knn));
        label(&mut nt, &format!("near d{nearest_d:.2} {}", short(&imgs[nearest].name)));
        tiles_for_sheet.push(ct);
        tiles_for_sheet.push(nt);

        scored.push(ScoredTile {
            id: t.id.clone(),
            loc,
            knn,
            nearest,
            nearest_d,
            saturated,
        });
    }

    // per-location mean knn (lower = closer to the corpus = better).
    let mut by_loc: HashMap<String, Vec<f64>> = HashMap::new();
    for s in &scored {
        by_loc.entry(s.loc.clone()).or_default().push(s.knn);
    }
    let mut loc_score: Vec<(String, f64)> = by_loc
        .into_iter()
        .map(|(l, v)| (l, v.iter().sum::<f64>() / v.len() as f64))
        .collect();
    loc_score.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // okay (B1/B2/B4/B5) must outrank sparse (B0/B3): worst okay < best sparse.
    let okay = ["B1", "B2", "B4", "B5"];
    let sparse = ["B0", "B3"];
    let score_of = |name: &str| loc_score.iter().find(|(l, _)| l == name).map(|(_, s)| *s);
    let worst_okay = okay.iter().filter_map(|n| score_of(n)).fold(f64::MIN, f64::max);
    let best_sparse = sparse.iter().filter_map(|n| score_of(n)).fold(f64::MAX, f64::min);
    let okay_above_sparse = if worst_okay > f64::MIN && best_sparse < f64::MAX {
        Some(worst_okay < best_sparse)
    } else {
        None
    };

    let sheet = if tiles_for_sheet.is_empty() {
        None
    } else {
        let grid = compose_grid(&tiles_for_sheet, Some(2));
        let path = format!("{}/buffet_ranking.png", args.out_dir.trim_end_matches('/'));
        grid.save(&path).map_err(|e| format!("failed to write {path}: {e}"))?;
        Some(path)
    };

    Ok(BuffetReport {
        tiles: scored,
        loc_score,
        okay_above_sparse,
        sheet,
    })
}

// ===========================================================================
// Phase 5 — corpus structure (k-means archetypes + exemplar sheet)
// ===========================================================================

fn cluster_exemplars(imgs: &[CorpusImg], args: &CalibrateArgs) -> Result<String, String> {
    let points: Vec<Vec<f64>> = imgs.iter().map(|i| i.sig.as_ref().unwrap().concat()).collect();
    let k = args.clusters.min(points.len().max(1));
    let (assign, centers) = kmeans(&points, k, 20, args.seed);

    // exemplars per cluster: nearest points to the centroid.
    let per = args.exemplars.max(1);
    let mut rows: Vec<RgbImage> = Vec::new();
    let mut sizes = vec![0usize; k];
    for a in &assign {
        sizes[*a] += 1;
    }
    for c in 0..k {
        let mut members: Vec<(f64, usize)> = (0..points.len())
            .filter(|&i| assign[i] == c)
            .map(|i| (vdist2(&points[i], &centers[c]), i))
            .collect();
        members.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for slot in 0..per {
            // pad short clusters with a dark blank so the grid stays rectangular.
            if let Some(&(_, idx)) = members.get(slot) {
                let mut th = thumb(&imgs[idx].path, args.thumb_width)?;
                if slot == 0 {
                    label(&mut th, &format!("C{c} n={} {}", sizes[c], short(&imgs[idx].name)));
                }
                rows.push(th);
            } else {
                let h = (args.thumb_width as f64 * 9.0 / 16.0).round() as u32;
                rows.push(RgbImage::from_pixel(args.thumb_width, h, Rgb([20, 20, 20])));
            }
        }
    }
    let grid = compose_grid(&rows, Some(per));
    let path = format!("{}/clusters.png", args.out_dir.trim_end_matches('/'));
    grid.save(&path).map_err(|e| format!("failed to write {path}: {e}"))?;
    Ok(path)
}

/// Squared Euclidean distance between two equal-length vectors.
fn vdist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Lloyd k-means with k-means++ seeding over arbitrary-length vectors. Returns
/// `(assignments, centroids)`.
fn kmeans(points: &[Vec<f64>], k: usize, iters: usize, seed: u64) -> (Vec<usize>, Vec<Vec<f64>>) {
    let n = points.len();
    let dim = points.first().map(|p| p.len()).unwrap_or(0);
    if n == 0 || k == 0 {
        return (vec![0; n], Vec::new());
    }
    let k = k.min(n);
    let mut rng = SplitMix64(seed);
    let mut centers: Vec<Vec<f64>> = vec![points[rng.below(n)].clone()];
    while centers.len() < k {
        let d2: Vec<f64> = points
            .iter()
            .map(|p| centers.iter().map(|c| vdist2(p, c)).fold(f64::INFINITY, f64::min))
            .collect();
        let sum: f64 = d2.iter().sum();
        if sum <= 0.0 {
            centers.push(points[rng.below(n)].clone());
            continue;
        }
        let mut t = rng.unit() * sum;
        let mut chosen = n - 1;
        for (i, &d) in d2.iter().enumerate() {
            t -= d;
            if t <= 0.0 {
                chosen = i;
                break;
            }
        }
        centers.push(points[chosen].clone());
    }

    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut bd = f64::INFINITY;
            for (j, c) in centers.iter().enumerate() {
                let d = vdist2(p, c);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            assign[i] = best;
        }
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut cnt = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let a = assign[i];
            for d in 0..dim {
                sums[a][d] += p[d];
            }
            cnt[a] += 1;
        }
        for j in 0..k {
            if cnt[j] > 0 {
                for d in 0..dim {
                    centers[j][d] = sums[j][d] / cnt[j] as f64;
                }
            } else {
                centers[j] = points[rng.below(n)].clone();
            }
        }
    }
    (assign, centers)
}

// ===========================================================================
// Candidate rendering (f64 cheap regime — the buffet DEEP tiles are shallow)
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn render_candidate(
    center: Complex<f64>,
    width: f64,
    maxiter: u32,
    w: u32,
    ss: u32,
    trap: Trap,
    palette: &Palette,
    params: &ColorParams,
) -> RgbImage {
    let h = (w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let prec = hp::prec_bits(w, width);
    let cre = BigFloat::from_f64(center.re, prec);
    let cim = BigFloat::from_f64(center.im, prec);
    let panel = probe::render_mandel_panel(
        &cre, &cim, center, width, w, h, ss, maxiter, 1e6, prec, trap, BackendChoice::F64,
    );
    render::shade_and_downsample(
        &panel.buf.samples,
        w,
        h,
        ss,
        palette,
        params,
        width / w as f64,
    )
}

/// The buffet/search default shading (smooth iteration, default palette).
fn default_color_params() -> ColorParams {
    ColorParams {
        density: 0.025,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    }
}

// ===========================================================================
// buffet.json parsing (hand-rolled, tolerant)
// ===========================================================================

struct BuffetTile {
    id: String,
    center: Complex<f64>,
    width: f64,
    maxiter: u32,
}

/// Pull the source-B, scale-DEEP tiles out of a `buffet.json`. Each tile object
/// begins with `"id":` and contains `source`, `scale`, `center.{re,im}`,
/// `width`, `maxiter`. Tolerant block scan: slice between consecutive `"id":`.
fn parse_buffet_deep_b(text: &str) -> Vec<BuffetTile> {
    let mut starts = Vec::new();
    let mut i = 0;
    while let Some(p) = text.get(i..).and_then(|s| s.find("\"id\":")).map(|p| p + i) {
        starts.push(p);
        i = p + 5;
    }
    let mut out = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(text.len());
        let block = &text[start..end];
        let source = str_field(block, "\"source\":").unwrap_or_default();
        let scale = str_field(block, "\"scale\":").unwrap_or_default();
        if source != "B" || scale != "DEEP" {
            continue;
        }
        let id = str_field(block, "\"id\":").unwrap_or_default();
        let re = num_field(block, "\"re\":");
        let im = num_field(block, "\"im\":");
        let width = num_field(block, "\"width\":");
        let maxiter = num_field(block, "\"maxiter\":");
        if let (Some(re), Some(im), Some(width), Some(maxiter)) = (re, im, width, maxiter) {
            out.push(BuffetTile {
                id,
                center: Complex::new(re, im),
                width,
                maxiter: maxiter.round() as u32,
            });
        }
    }
    out
}

/// Read the string value following `key` in `block`.
fn str_field(block: &str, key: &str) -> Option<String> {
    let p = block.find(key)? + key.len();
    let rest = block[p..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Read the numeric value (decimal/scientific) following `key` in `block`.
fn num_field(block: &str, key: &str) -> Option<f64> {
    let p = block.find(key)? + key.len();
    let rest = block[p..].trim_start();
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ===========================================================================
// Thumbnails / labels
// ===========================================================================

/// Decode a corpus image, center-crop 16:9, resize to a `w`-wide thumbnail.
fn thumb(path: &Path, w: u32) -> Result<RgbImage, String> {
    let img = image::open(path)
        .map_err(|e| format!("decode {}: {e}", path.display()))?
        .to_rgb8();
    Ok(fit_to(&center_crop_16x9(&img), w))
}

/// Resize an already-16:9 image to a `w`-wide thumbnail (height follows 16:9).
fn fit_to(img: &RgbImage, w: u32) -> RgbImage {
    let h = (w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    image::imageops::resize(img, w, h, FilterType::Triangle)
}

/// Burn a small caption (top-left, dark plate) onto a thumbnail.
fn label(img: &mut RgbImage, text: &str) {
    font::burn(img, text);
}

mod font {
    use super::*;
    use crate::font as f;
    pub fn burn(img: &mut RgbImage, text: &str) {
        let short: String = text.chars().take(34).collect();
        f::draw_text(img, &short.to_uppercase(), 2, 2, 1, Rgb([240, 240, 240]), true);
    }
}

/// Trim a long filename for a caption.
fn short(name: &str) -> &str {
    let n = name.len();
    if n <= 22 {
        name
    } else {
        &name[..22]
    }
}

// ===========================================================================
// JSON artifact + report
// ===========================================================================

fn build_artifact_json(bins: &FrozenBins, imgs: &[CorpusImg], args: &CalibrateArgs) -> String {
    let mut s = String::from("{\n");
    s.push_str("  \"metric\": {\n");
    s.push_str(&format!("    \"work_w\": {WORK_W}, \"work_h\": {WORK_H},\n"));
    s.push_str(&format!(
        "    \"scale_grid\": [{}],\n",
        SCALE_GRID.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    ));
    s.push_str(&format!("    \"nbins\": {NBINS},\n"));
    s.push_str("    \"energy\": \"oklab forward-diff gradient magnitude, pooled per-area\",\n");
    s.push_str("    \"distance\": \"sum over scales of 1-D EMD (cdf-L1) on equal-count bins\"\n");
    s.push_str("  },\n");

    // frozen quantile edges per scale
    s.push_str("  \"frozen_edges\": {\n");
    for (i, &g) in SCALE_GRID.iter().enumerate() {
        s.push_str(&format!(
            "    \"s{g}\": [{}]{}\n",
            bins.edges[i].iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", "),
            if i + 1 < SCALE_GRID.len() { "," } else { "" }
        ));
    }
    s.push_str("  },\n");

    s.push_str("  \"meta\": {\n");
    s.push_str(&format!("    \"dir\": {},\n", js(&args.dir)));
    s.push_str(&format!("    \"n_images\": {}\n", imgs.len()));
    s.push_str("  },\n");

    // per-image histograms (anchored individually, no premature mean/std)
    s.push_str("  \"images\": [\n");
    for (k, im) in imgs.iter().enumerate() {
        let sig = im.sig.as_ref().unwrap();
        s.push_str(&format!("    {{ \"name\": {}, \"hist\": [", js(&im.name)));
        let scales: Vec<String> = (0..4)
            .map(|sc| {
                format!(
                    "[{}]",
                    sig.hist[sc].iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", ")
                )
            })
            .collect();
        s.push_str(&scales.join(", "));
        s.push_str("] }");
        if k + 1 < imgs.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

#[allow(clippy::too_many_arguments)]
fn report(
    bins: &FrozenBins,
    imgs: &[CorpusImg],
    br: &BuffetReport,
    artifact: &str,
    nn_path: &str,
    cluster_path: &Option<String>,
    args: &CalibrateArgs,
    n_err: usize,
) {
    println!("\n=== energy-metric calibration ({} images, {} skipped) ===", imgs.len(), n_err);
    println!(
        "  canonical {WORK_W}x{WORK_H}; scales {:?}; {NBINS} equal-count bins/scale; \
         distance = Σ per-scale 1-D EMD (weights {:?})",
        SCALE_GRID,
        args.resolved_weights().unwrap_or([1.0; 4])
    );
    println!("  frozen quantile edges (min .. median .. max per scale):");
    for (i, &g) in SCALE_GRID.iter().enumerate() {
        let e = &bins.edges[i];
        println!(
            "    {g:>2}x{g:<2} ({:>3} regions/img): [{:.5} .. {:.5} .. {:.5}]",
            g * g,
            e[0],
            e[e.len() / 2],
            e[e.len() - 1]
        );
    }

    if !br.tiles.is_empty() {
        println!("\n=== buffet eye-check (source-B DEEP vs frozen corpus) ===");
        println!("  per-tile EMD-to-nearest-{} (lower = closer to corpus):", args.knn);
        println!("    {:<14} {:>8} {:>8}  {}", "tile", "knn", "near_d", "nearest corpus image");
        let mut rows = br.tiles.iter().collect::<Vec<_>>();
        rows.sort_by(|a, b| a.knn.partial_cmp(&b.knn).unwrap_or(std::cmp::Ordering::Equal));
        for t in rows {
            println!(
                "    {:<14} {:>8.3} {:>8.3}  {}{}",
                t.id,
                t.knn,
                t.nearest_d,
                short(&imgs[t.nearest].name),
                if t.saturated { "  [top-bin saturated]" } else { "" }
            );
        }
        println!("\n  per-location mean knn (ascending = best first):");
        for (loc, sc) in &br.loc_score {
            let tag = match loc.as_str() {
                "B1" | "B2" | "B4" | "B5" => "okay",
                "B0" | "B3" => "sparse",
                _ => "?",
            };
            println!("    {loc:<4} {sc:>8.3}  ({tag})");
        }
        match br.okay_above_sparse {
            Some(true) => println!(
                "  VERDICT: PASS — every okay tile (B1/B2/B4/B5) outranks both sparse tiles (B0/B3)."
            ),
            Some(false) => println!(
                "  VERDICT: FAIL — an okay tile did NOT outrank a sparse one; metric needs fixing \
                 before anything builds on it."
            ),
            None => println!("  VERDICT: indeterminate (missing okay/sparse locations in buffet json)."),
        }
    }

    println!("\nwrote:");
    println!("  {artifact} (frozen bins + per-image histograms)");
    if !nn_path.is_empty() {
        println!("  {nn_path} (NN-pair eye-check sheet)");
    }
    if let Some(p) = &br.sheet {
        println!("  {p} (buffet ranking: candidate | nearest corpus)");
    }
    if let Some(p) = cluster_path {
        println!("  {p} (k-means archetype exemplar sheet)");
    }
    println!(
        "\nnote: equal-count bins are defined by the corpus energy range; a candidate busier than \
         anything in the corpus saturates the top bin (off-distribution scores poorly — expected)."
    );
}

// ----- artifact parsing (frozen edges + per-image histograms) -----

pub(crate) fn parse_artifact(text: &str) -> Result<(FrozenBins, Vec<(String, Signature)>), String> {
    // frozen edges (one float list per scale)
    let fe = text.find("\"frozen_edges\"").ok_or("artifact: no frozen_edges")?;
    let keys = ["\"s16\":", "\"s8\":", "\"s4\":", "\"s2\":"];
    let mut edges: Vec<Vec<f64>> = Vec::with_capacity(4);
    for kkey in keys {
        let kp = text[fe..]
            .find(kkey)
            .map(|p| p + fe)
            .ok_or_else(|| format!("artifact: missing {kkey} in frozen_edges"))?;
        let br = text[kp..].find('[').map(|p| p + kp).ok_or("artifact: no '[' after edge key")?;
        let (vals, _) = read_float_list(text, br)?;
        edges.push(vals);
    }
    let edges: [Vec<f64>; 4] = edges.try_into().map_err(|_| "artifact: edges not 4 scales")?;
    let bins = FrozenBins { edges };

    // per-image histograms
    let imp = text.find("\"images\"").ok_or("artifact: no images array")?;
    let mut corpus = Vec::new();
    let mut i = imp + "\"images\"".len();
    while let Some(np) = text[i..].find("\"name\":").map(|p| p + i) {
        let name = str_field(&text[np..], "\"name\":").unwrap_or_default();
        let Some(hp) = text[np..].find("\"hist\":").map(|p| p + np) else {
            break;
        };
        let (hist, next) = read_hist(text, hp + "\"hist\":".len())?;
        corpus.push((name, Signature { hist }));
        i = next;
    }
    Ok((bins, corpus))
}

/// Read a `[`-delimited list of f64 (no nested brackets). `s[from]` must be `[`.
/// Returns the floats and the byte index just past the matching `]`.
fn read_float_list(s: &str, from: usize) -> Result<(Vec<f64>, usize), String> {
    if s.as_bytes().get(from) != Some(&b'[') {
        return Err("read_float_list: expected '['".into());
    }
    let end = s[from..].find(']').map(|p| p + from).ok_or("read_float_list: unterminated '['")?;
    let mut v = Vec::new();
    for tok in s[from + 1..end].split(',') {
        let t = tok.trim();
        if !t.is_empty() {
            v.push(t.parse::<f64>().map_err(|_| format!("read_float_list: bad float '{t}'"))?);
        }
    }
    Ok((v, end + 1))
}

/// Read a 4-scale histogram `[[…],[…],[…],[…]]` starting after the `"hist":` key.
/// Skips the outer `[` and reads four inner float lists.
fn read_hist(s: &str, after_key: usize) -> Result<([Vec<f64>; 4], usize), String> {
    let outer = s[after_key..].find('[').map(|p| p + after_key).ok_or("read_hist: no '[' after key")?;
    let mut pos = outer + 1;
    let mut scales: Vec<Vec<f64>> = Vec::with_capacity(4);
    for _ in 0..4 {
        let inner = s[pos..].find('[').map(|p| p + pos).ok_or("read_hist: missing inner array")?;
        let (vals, next) = read_float_list(s, inner)?;
        scales.push(vals);
        pos = next;
    }
    let arr: [Vec<f64>; 4] = scales.try_into().map_err(|_| "read_hist: not 4 scales")?;
    Ok((arr, pos))
}

// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// `calibrate` subcommand: see the module docs in `energy.rs`. Calibration +
/// eye-check only — it freezes the metric (bins) and produces the visual gates
/// (NN pairs, buffet ranking, cluster sheet). It proposes no objective and runs
/// no search.
#[derive(Args, Debug)]
pub struct CalibrateArgs {
    /// Corpus folder of reference wallpapers (top level only — no recursion).
    #[arg(long, default_value = "C:/Users/techm/Desktop/Wallpapers")]
    pub dir: String,

    /// Output directory for the calibration artifact + eye-check sheets.
    #[arg(long, default_value = "out/calibrate")]
    pub out_dir: String,

    /// Buffet metrics JSON whose source-B DEEP tiles are the candidate eye-check.
    #[arg(long, default_value = "out/buffet/buffet.json")]
    pub buffet_json: String,

    /// Per-scale EMD weights `w16,w8,w4,w2` (default equal).
    #[arg(long, default_value = "1,1,1,1")]
    pub weights: String,

    /// Number of corpus images sampled for the NN-pair eye-check sheet.
    #[arg(long, default_value_t = 16)]
    pub nn_samples: usize,

    /// k for the buffet EMD-to-nearest-k score.
    #[arg(long, default_value_t = 5)]
    pub knn: usize,

    /// k-means archetype count for the corpus-structure sheet (`<2` disables).
    #[arg(long, default_value_t = 6)]
    pub clusters: usize,

    /// Exemplars per cluster in the archetype sheet (one row each).
    #[arg(long, default_value_t = 6)]
    pub exemplars: usize,

    /// Thumbnail width (px) for the eye-check sheets (height follows 16:9).
    #[arg(long, default_value_t = 384)]
    pub thumb_width: u32,

    /// Render width (px) for each buffet candidate tile (height follows 16:9).
    #[arg(long, default_value_t = 1280)]
    pub candidate_width: u32,

    /// Supersample for candidate renders.
    #[arg(long, default_value_t = 2)]
    pub supersample: u32,

    /// RNG seed for k-means seeding.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl CalibrateArgs {
    /// Parse `--weights` (`w16,w8,w4,w2`) into the per-scale weight array.
    pub fn resolved_weights(&self) -> Result<[f64; 4], String> {
        let p: Vec<&str> = self.weights.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --weights '{}', expected w16,w8,w4,w2", self.weights));
        }
        let mut w = [0.0; 4];
        for (i, s) in p.iter().enumerate() {
            w[i] = s
                .trim()
                .parse()
                .map_err(|_| format!("invalid --weights component '{}'", s.trim()))?;
        }
        Ok(w)
    }
}
