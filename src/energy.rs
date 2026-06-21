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
use crate::cli::{ArchetypeArgs, BackendChoice, CalibrateArgs, OverbusyArgs, RescoreArgs};
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
    fn signature(&self, regions: &Regions) -> Signature {
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
fn distance(x: &Signature, y: &Signature, w: &[f64; 4]) -> f64 {
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

// ===========================================================================
// `rescore` — diagnosis-only re-scoring of the buffet DEEP tiles (5 rules)
// ===========================================================================
//
// Loads the persisted corpus calibration + a buffet-histogram cache (re-rendering
// the fixed buffet set once on a cache miss), then scores each source-B DEEP tile
// under five candidate rules and prints a PASS/FAIL table per rule. Reuses the
// metric's `distance`/`emd1d`/`Signature`/`FrozenBins` and the `kmeans` from the
// calibrate path unchanged. Picks no winner; produces the table and stops.

/// A buffet DEEP tile's calibrated signature (already binned under frozen edges).
struct TileSig {
    id: String,
    loc: String, // location label, e.g. "B0" (the id prefix before the first '_')
    sig: Signature,
}

/// Per-tile scores under every rule (plus the two diagnostic scalars).
struct TileScore {
    id: String,
    loc: String,
    sparse: f64,   // s16 bin0 fraction — the sparseness scalar
    d: f64,        // density scalar = 1 − sparseness
    r1: f64,       // nearest-k mean EMD
    r2: f64,       // nearest-archetype-centroid EMD
    r2_arch: usize,
    r3: f64,       // EMD to global mean histogram
    r4: Vec<f64>,  // tail-pruned nearest-k, per cutoff
    r5: Vec<f64>,  // two-sided band violation, per band
}

/// `id` → location label: the prefix before the first `_` (`B0_ON_DEEP` → `B0`).
fn loc_of(id: &str) -> String {
    id.split('_').next().unwrap_or(id).to_string()
}

pub fn run_rescore(args: &RescoreArgs) -> Result<(), String> {
    let weights = args.resolved_weights()?;

    // ---- load persisted corpus calibration (frozen bins + 746 histograms) ----
    let text = fs::read_to_string(&args.artifact)
        .map_err(|e| format!("reading artifact {}: {e}", args.artifact))?;
    let (bins, corpus) = parse_artifact(&text)?;
    let n = corpus.len();
    if n == 0 {
        return Err("artifact has no per-image histograms".into());
    }
    eprintln!("rescore: loaded {n} corpus signatures + frozen bins from {}", args.artifact);

    // ---- load (or render+cache) the buffet DEEP tile histograms ----
    let (tiles, source) = load_or_render_buffet(args, &bins)?;
    if tiles.is_empty() {
        return Err("no buffet DEEP tiles to score".into());
    }

    // ---- corpus-derived structures ----
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    let global_mean = mean_signature(&corpus);

    // sparseness (s16 bin0) distribution over the corpus → cutoffs/percentiles.
    let mut sparse_corpus: Vec<f64> = corpus_sigs.iter().map(|s| s.hist[0][0]).collect();
    sparse_corpus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let s_p75 = pct(&sparse_corpus, 0.75);
    let s_p90 = pct(&sparse_corpus, 0.90);

    // density d = 1 − s16 bin0 distribution → central bands.
    let mut d_corpus: Vec<f64> = sparse_corpus.iter().map(|&x| 1.0 - x).collect();
    d_corpus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let d_p10 = pct(&d_corpus, 0.10);
    let d_p25 = pct(&d_corpus, 0.25);
    let d_p75 = pct(&d_corpus, 0.75);
    let d_p90 = pct(&d_corpus, 0.90);

    // R4 cutoffs and R5 bands (labels carried for the report).
    let r4_cuts: Vec<(String, f64)> = vec![
        ("none".into(), f64::INFINITY),
        (format!("p90={s_p90:.3}"), s_p90),
        (format!("p75={s_p75:.3}"), s_p75),
        (">=0.30".into(), 0.30),
        (">=0.15".into(), 0.15),
    ];
    let r5_bands: Vec<(String, f64, f64)> = vec![
        (format!("p25-p75 [{d_p25:.3},{d_p75:.3}]"), d_p25, d_p75),
        (format!("p10-p90 [{d_p10:.3},{d_p90:.3}]"), d_p10, d_p90),
    ];

    // R2 archetypes: recompute k-means (not stored in the artifact). seed mirrors
    // calibrate's default so the clustering matches the calibrate cluster sheet.
    let points: Vec<Vec<f64>> = corpus_sigs.iter().map(|s| s.concat()).collect();
    let kk = args.clusters.max(2).min(n);
    let (assign, centers) = kmeans(&points, kk, 20, args.seed);
    let centroid_sigs: Vec<Signature> = centers.iter().map(|c| vec_to_sig(c)).collect();
    let mut arch_size = vec![0usize; centroid_sigs.len()];
    let mut arch_sparse_sum = vec![0.0f64; centroid_sigs.len()];
    for (i, &a) in assign.iter().enumerate() {
        arch_size[a] += 1;
        arch_sparse_sum[a] += corpus_sigs[i].hist[0][0];
    }

    let k = args.knn.max(1);

    // ---- score every tile ----
    let mut scores: Vec<TileScore> = Vec::with_capacity(tiles.len());
    for t in &tiles {
        let sparse = t.sig.hist[0][0];
        let d = 1.0 - sparse;

        // distances to every corpus image, paired with that image's sparseness.
        let mut ds: Vec<(f64, f64)> = corpus_sigs
            .iter()
            .map(|cs| (distance(&t.sig, cs, &weights), cs.hist[0][0]))
            .collect();
        ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // R1 — nearest-k mean.
        let r1 = mean_k(ds.iter().map(|x| x.0), k);

        // R4 — tail-pruned nearest-k, per cutoff.
        let r4: Vec<f64> = r4_cuts
            .iter()
            .map(|&(_, cut)| mean_k(ds.iter().filter(|x| x.1 < cut).map(|x| x.0), k))
            .collect();

        // R2 — nearest archetype centroid.
        let mut r2 = f64::INFINITY;
        let mut r2_arch = 0;
        for (a, cs) in centroid_sigs.iter().enumerate() {
            let dd = distance(&t.sig, cs, &weights);
            if dd < r2 {
                r2 = dd;
                r2_arch = a;
            }
        }

        // R3 — distance to the global mean histogram.
        let r3 = distance(&t.sig, &global_mean, &weights);

        // R5 — two-sided central-density band violation, per band.
        let r5: Vec<f64> = r5_bands
            .iter()
            .map(|&(_, lo, hi)| {
                if d < lo {
                    lo - d
                } else if d > hi {
                    d - hi
                } else {
                    0.0
                }
            })
            .collect();

        scores.push(TileScore {
            id: t.id.clone(),
            loc: t.loc.clone(),
            sparse,
            d,
            r1,
            r2,
            r2_arch,
            r3,
            r4,
            r5,
        });
    }

    rescore_report(
        args, &scores, &r4_cuts, &r5_bands, &arch_size, &arch_sparse_sum, &centroid_sigs, s_p90,
        source, n,
    );
    write_rescore_json(args, &scores, &r4_cuts, &r5_bands)?;
    Ok(())
}

/// Mean of the `k` smallest values from an iterator already in ascending order.
/// `NaN` if the iterator is empty (e.g. a cutoff pruned every corpus survivor).
fn mean_k(sorted: impl Iterator<Item = f64>, k: usize) -> f64 {
    let taken: Vec<f64> = sorted.take(k).collect();
    if taken.is_empty() {
        f64::NAN
    } else {
        taken.iter().sum::<f64>() / taken.len() as f64
    }
}

/// Percentile of an already-sorted slice (`p` in `[0,1]`, nearest-rank).
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Per-scale mean histogram over the corpus (each scale still sums to 1).
fn mean_signature(corpus: &[(String, Signature)]) -> Signature {
    let n = corpus.len().max(1) as f64;
    let hist = std::array::from_fn(|s| {
        let mut acc = vec![0.0f64; NBINS];
        for (_, sig) in corpus {
            for b in 0..NBINS {
                acc[b] += sig.hist[s][b];
            }
        }
        for x in acc.iter_mut() {
            *x /= n;
        }
        acc
    });
    Signature { hist }
}

/// Split a `4·NBINS` concat vector (a k-means centroid) back into a `Signature`.
fn vec_to_sig(v: &[f64]) -> Signature {
    let hist = std::array::from_fn(|s| v[s * NBINS..(s + 1) * NBINS].to_vec());
    Signature { hist }
}

// ----- buffet histogram cache (load, or render the fixed set once) -----

fn load_or_render_buffet(
    args: &RescoreArgs,
    bins: &FrozenBins,
) -> Result<(Vec<TileSig>, &'static str), String> {
    load_or_render_buffet_tiles(
        &args.buffet_hist,
        &args.buffet_json,
        args.candidate_width,
        args.supersample,
        bins,
    )
}

/// Load the fixed source-B DEEP tile signatures from the histogram cache, or
/// render them once (deterministic, f64 cheap-regime) and cache them. Shared by
/// `rescore` and `overbusy`; the only permissible candidate render in either path.
fn load_or_render_buffet_tiles(
    buffet_hist: &str,
    buffet_json: &str,
    candidate_width: u32,
    supersample: u32,
    bins: &FrozenBins,
) -> Result<(Vec<TileSig>, &'static str), String> {
    // Prefer the persisted cache (repeatable after `out/` clears).
    if let Ok(text) = fs::read_to_string(buffet_hist) {
        let tiles = parse_tile_cache(&text)?;
        if !tiles.is_empty() {
            eprintln!("  loaded {} cached buffet tile histogram(s) from {}", tiles.len(), buffet_hist);
            return Ok((tiles, "cache"));
        }
    }

    // Cache miss → deterministically re-render the fixed source-B DEEP set
    // (the only permissible render; flagged).
    let btext = fs::read_to_string(buffet_json)
        .map_err(|e| format!("reading buffet json {buffet_json}: {e}"))?;
    let deep = parse_buffet_deep_b(&btext);
    if deep.is_empty() {
        return Err(format!("no source-B DEEP tiles in {buffet_json}"));
    }
    eprintln!(
        "  buffet histogram cache missing — RE-RENDERING {} fixed source-B DEEP tile(s) \
         at {candidate_width}px ss{supersample} (deterministic; flagged fallback) ...",
        deep.len(),
    );
    let palette = builtin("default", false).expect("default palette");
    let params = default_color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };
    let mut tiles = Vec::with_capacity(deep.len());
    for t in &deep {
        let cand = render_candidate(
            t.center,
            t.width,
            t.maxiter,
            candidate_width,
            supersample,
            trap,
            &palette,
            &params,
        );
        let regions = region_energies(&cand);
        let sig = bins.signature(&regions);
        tiles.push(TileSig {
            loc: loc_of(&t.id),
            id: t.id.clone(),
            sig,
        });
    }
    crate::ensure_parent_dir(buffet_hist)?;
    fs::write(buffet_hist, build_tile_cache_json(&tiles))
        .map_err(|e| format!("writing cache {buffet_hist}: {e}"))?;
    eprintln!("  cached {} tile histogram(s) → {buffet_hist}", tiles.len());
    Ok((tiles, "rendered"))
}

fn build_tile_cache_json(tiles: &[TileSig]) -> String {
    let mut s = String::from("{\n  \"tiles\": [\n");
    for (i, t) in tiles.iter().enumerate() {
        let scales: Vec<String> = (0..4)
            .map(|sc| format!("[{}]", t.sig.hist[sc].iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", ")))
            .collect();
        s.push_str(&format!("    {{ \"id\": {}, \"hist\": [{}] }}", js(&t.id), scales.join(", ")));
        if i + 1 < tiles.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n}\n");
    s
}

fn parse_tile_cache(text: &str) -> Result<Vec<TileSig>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(idp) = text[i..].find("\"id\":").map(|p| p + i) {
        let id = str_field(&text[idp..], "\"id\":").unwrap_or_default();
        let Some(hp) = text[idp..].find("\"hist\":").map(|p| p + idp) else {
            break;
        };
        let (hist, next) = read_hist(text, hp + "\"hist\":".len())?;
        out.push(TileSig {
            loc: loc_of(&id),
            id,
            sig: Signature { hist },
        });
        i = next;
    }
    Ok(out)
}

// ----- artifact parsing (frozen edges + per-image histograms) -----

fn parse_artifact(text: &str) -> Result<(FrozenBins, Vec<(String, Signature)>), String> {
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

// ----- aggregation, report, json -----

/// Per-location mean of a per-tile scalar (selected via `f`), keyed B0..B5.
fn loc_means(scores: &[TileScore], f: impl Fn(&TileScore) -> f64) -> Vec<(String, f64)> {
    let mut m: std::collections::BTreeMap<String, (f64, usize)> = std::collections::BTreeMap::new();
    for t in scores {
        let e = m.entry(t.loc.clone()).or_insert((0.0, 0));
        e.0 += f(t);
        e.1 += 1;
    }
    m.into_iter().map(|(l, (s, c))| (l, s / c as f64)).collect()
}

/// Sort locations ascending by score (lower = better for every rule).
fn ranked(mut v: Vec<(String, f64)>) -> Vec<(String, f64)> {
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// PASS iff every okay location (B1/B2/B4/B5) outranks both sparse ones (B0/B3).
fn pass_of(per_loc: &[(String, f64)]) -> Option<bool> {
    let okay = ["B1", "B2", "B4", "B5"];
    let sparse = ["B0", "B3"];
    let g = |name: &str| per_loc.iter().find(|(l, _)| l == name).map(|(_, s)| *s);
    let worst_okay = okay.iter().filter_map(|n| g(n)).fold(f64::MIN, f64::max);
    let best_sparse = sparse.iter().filter_map(|n| g(n)).fold(f64::MAX, f64::min);
    if worst_okay > f64::MIN && best_sparse < f64::MAX {
        Some(worst_okay < best_sparse)
    } else {
        None
    }
}

/// Print one rule's location ranking (best→worst) with its PASS/FAIL flag.
fn print_rank(label: &str, per_loc: &[(String, f64)]) {
    let sorted = ranked(per_loc.to_vec());
    let tag = match pass_of(&sorted) {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "INDET",
    };
    let body = sorted
        .iter()
        .map(|(l, s)| format!("{l}:{s:.4}"))
        .collect::<Vec<_>>()
        .join("  ");
    println!("  {label:<26} [{tag:>5}]  {body}");
}

#[allow(clippy::too_many_arguments)]
fn rescore_report(
    args: &RescoreArgs,
    scores: &[TileScore],
    r4_cuts: &[(String, f64)],
    r5_bands: &[(String, f64, f64)],
    arch_size: &[usize],
    arch_sparse_sum: &[f64],
    centroid_sigs: &[Signature],
    s_p90: f64,
    source: &str,
    n_corpus: usize,
) {
    println!(
        "\n=== rescore: buffet source-B DEEP vs persisted corpus ({} tiles, {} corpus imgs) ===",
        scores.len(),
        n_corpus
    );
    println!(
        "  tile histograms: {}; weights {:?}; k={}; sparseness scalar = s16 bin0 fraction; density d = 1 − sparseness",
        match source {
            "cache" => "loaded from cache",
            _ => "RE-RENDERED (flagged fallback)",
        },
        args.resolved_weights().unwrap_or([1.0; 4]),
        args.knn
    );

    // per-tile scalars
    println!("\n  per-tile diagnostic scalars:");
    println!("    {:<14} {:>10} {:>10}", "tile", "sparse(s16b0)", "density d");
    for t in scores {
        println!("    {:<14} {:>13.4} {:>10.4}", t.id, t.sparse, t.d);
    }

    // per-location mean scalars
    let loc_sparse = ranked(loc_means(scores, |t| t.sparse));
    println!("\n  per-location mean sparseness (ascending; sparse tiles B0/B3 should top this):");
    println!(
        "    {}",
        loc_sparse.iter().map(|(l, s)| format!("{l}:{s:.4}")).collect::<Vec<_>>().join("  ")
    );

    println!("\n  per-rule location ranking (best→worst; lower=better). PASS = B1/B2/B4/B5 all beat B0/B3:");
    print_rank("R1 nearest-k", &loc_means(scores, |t| t.r1));
    print_rank("R2 nearest-archetype", &loc_means(scores, |t| t.r2));
    print_rank("R3 global-centroid", &loc_means(scores, |t| t.r3));
    for (ci, (clab, _)) in r4_cuts.iter().enumerate() {
        print_rank(&format!("R4 prune {clab}"), &loc_means(scores, |t| t.r4[ci]));
    }
    for (bi, (blab, _, _)) in r5_bands.iter().enumerate() {
        print_rank(&format!("R5 band {blab}"), &loc_means(scores, |t| t.r5[bi]));
    }

    // R2 archetype match per tile + archetype-sparseness inspection
    println!("\n  R2 archetype match per tile (arch idx | EMD):");
    for t in scores {
        println!("    {:<14} arch C{} d={:.4}", t.id, t.r2_arch, t.r2);
    }
    println!("\n  archetype sparseness (s16 bin0; flag if centroid or member-mean ≥ corpus p90={s_p90:.3}):");
    println!("    {:<6} {:>6} {:>14} {:>14}  {}", "arch", "n", "centroid_s16b0", "mean_member", "flag");
    for c in 0..centroid_sigs.len() {
        let csp = centroid_sigs[c].hist[0][0];
        let msp = if arch_size[c] > 0 {
            arch_sparse_sum[c] / arch_size[c] as f64
        } else {
            f64::NAN
        };
        let flag = if csp >= s_p90 || msp >= s_p90 { "SPARSE/degenerate" } else { "" };
        println!("    C{:<5} {:>6} {:>14.4} {:>14.4}  {}", c, arch_size[c], csp, msp, flag);
    }

    println!("\nwrote:");
    println!("  {} (per-tile per-rule scores)", args.out_json);
    if source == "rendered" {
        println!("  {} (buffet tile histogram cache — re-rendered this run)", args.buffet_hist);
    }
    println!(
        "\nnote: diagnosis only — no winning rule selected. R5's `d = 1 − s16-bin0-fraction` is a \
         coarse density proxy (fine-region edge mass would be a richer `d`)."
    );
}

fn write_rescore_json(
    args: &RescoreArgs,
    scores: &[TileScore],
    r4_cuts: &[(String, f64)],
    r5_bands: &[(String, f64, f64)],
) -> Result<(), String> {
    let mut s = String::from("{\n");
    s.push_str(&format!(
        "  \"rules\": {{ \"r4_cutoffs\": [{}], \"r5_bands\": [{}] }},\n",
        r4_cuts.iter().map(|(l, _)| js(l)).collect::<Vec<_>>().join(", "),
        r5_bands.iter().map(|(l, _, _)| js(l)).collect::<Vec<_>>().join(", "),
    ));
    s.push_str("  \"tiles\": [\n");
    for (i, t) in scores.iter().enumerate() {
        s.push_str(&format!(
            "    {{ \"id\": {}, \"loc\": {}, \"sparse_s16b0\": {}, \"density_d\": {}, \
\"r1_nearest_k\": {}, \"r2_archetype\": {}, \"r2_arch_idx\": {}, \"r3_global\": {}, \
\"r4_tail_pruned\": [{}], \"r5_band_violation\": [{}] }}{}\n",
            js(&t.id),
            js(&t.loc),
            jf(t.sparse),
            jf(t.d),
            jf(t.r1),
            jf(t.r2),
            t.r2_arch,
            jf(t.r3),
            t.r4.iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", "),
            t.r5.iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", "),
            if i + 1 < scores.len() { "," } else { "" },
        ));
    }
    s.push_str("  ]\n}\n");
    crate::ensure_parent_dir(&args.out_json)?;
    fs::write(&args.out_json, s).map_err(|e| format!("writing {}: {e}", args.out_json))?;
    Ok(())
}

// ===========================================================================
// `overbusy` — over-busy/speckle controls + C4 quarantine + survivor re-score
// ===========================================================================
//
// Diagnosis-only. Adds non-sparse-but-bad tiles (sub-pixel escape speckle) to the
// known-answer set, quarantines the degenerate reference cluster from the corpus
// typicality statistics, and re-scores the survivor rules (R3 global-centroid, R5
// density band, raw s16-bin0 scalar) against okay + sparse + the controls. Reuses
// the metric, frozen bins, cached buffet histograms, `distance`, and `kmeans`
// unchanged. Renders ONLY the fixed control set. Picks no winner.

/// A fixed over-busy/speckle control candidate (a known-answer-set member). All
/// are rendered f64 cheap-regime (width ≥ ~1e-7) so the noise is genuine fractal
/// speckle, not f64 quantization. `kind` is descriptive only.
struct ControlSpec {
    id: &'static str,
    re: f64,
    im: f64,
    width: f64,
    maxiter: u32,
    kind: &'static str,
}

/// The fixed control set. Speckle = near-boundary dust / okay-center-driven-past-
/// coherence at high maxiter, where local detail is finer than a pixel so adjacent
/// pixels are uncorrelated and the whole frame fills with incoherent high-frequency
/// noise (the `depth ≠ quality` root-bug failure mode).
//
// Empirically located (see out/cprobe screening): these are the rare full-frame
// embedded-Julia regions whose detail is dense edge-to-edge with no flat far-field
// or interior, so d pins at ~1.0. At the 1280px render the fine webbing resolves
// as genuine sub-pixel gray static (uncorrelated pixel noise) between harsh
// coherent filaments — the over-busy / `depth ≠ quality` failure mode descent
// dead-ends in. Self-similar attempts (Misiurewicz dust) stay scale-invariant
// COHERENT and okay-center-driven-deeper exits to FLAT (d→0); both were rejected
// by the eye-check + d gate. Three distinct centers + one depth variant.
const CONTROLS: &[ControlSpec] = &[
    ControlSpec { id: "OB_A1", re: -0.16070135, im: 1.0375665, width: 3.0e-5, maxiter: 20000, kind: "speckle" },
    ControlSpec { id: "OB_A2", re: -0.16070135, im: 1.0375665, width: 1.0e-5, maxiter: 30000, kind: "speckle" },
    ControlSpec { id: "OB_B", re: -0.235125, im: 0.827215, width: 2.0e-5, maxiter: 20000, kind: "speckle" },
    ControlSpec { id: "OB_C", re: 0.432539867, im: 0.226118675, width: 1.0e-5, maxiter: 25000, kind: "speckle" },
];

/// Tile category for the stricter pass criterion.
#[derive(Clone, Copy, PartialEq)]
enum Cat {
    Okay,
    Sparse,
    Control,
}

impl Cat {
    fn of_loc(loc: &str) -> Cat {
        match loc {
            "B1" | "B2" | "B4" | "B5" => Cat::Okay,
            "B0" | "B3" => Cat::Sparse,
            _ => Cat::Control,
        }
    }
    fn tag(self) -> &'static str {
        match self {
            Cat::Okay => "okay",
            Cat::Sparse => "sparse",
            Cat::Control => "control",
        }
    }
}

/// One scored tile in the expanded known-answer set.
struct ObTile {
    id: String,
    cat: Cat,
    d: f64,
    sparse: f64, // s16 bin0 fraction
    r3: f64,     // EMD to the C4-quarantined global centroid
    r5: Vec<f64>, // per-band violation (lower = better)
    r5_branch: Vec<&'static str>, // below / inside / above, per band
    saturated: bool,
}

pub fn run_overbusy(args: &OverbusyArgs) -> Result<(), String> {
    let weights = args.resolved_weights()?;

    // ---- load persisted corpus calibration (frozen bins + 746 histograms) ----
    let text = fs::read_to_string(&args.artifact)
        .map_err(|e| format!("reading artifact {}: {e}", args.artifact))?;
    let (bins, corpus) = parse_artifact(&text)?;
    let n = corpus.len();
    if n == 0 {
        return Err("artifact has no per-image histograms".into());
    }
    eprintln!("overbusy: loaded {n} corpus signatures + frozen bins from {}", args.artifact);

    // ---- recompute k-means to recover cluster membership (not persisted) ----
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    let points: Vec<Vec<f64>> = corpus_sigs.iter().map(|s| s.concat()).collect();
    let kk = args.clusters.max(2).min(n);
    let (assign, _centers) = kmeans(&points, kk, 20, args.seed);
    let q = args.quarantine;
    let mut sizes = vec![0usize; kk];
    let mut sparse_sum = vec![0.0f64; kk];
    for (i, &a) in assign.iter().enumerate() {
        sizes[a] += 1;
        sparse_sum[a] += corpus_sigs[i].hist[0][0];
    }

    // survivors = corpus images NOT in the quarantined cluster.
    let survivors: Vec<(String, Signature)> = corpus
        .iter()
        .enumerate()
        .filter(|(i, _)| assign[*i] != q)
        .map(|(_, c)| c.clone())
        .collect();
    let n_surv = survivors.len();

    // ---- corpus d-distribution, before vs after quarantine ----
    let d_all = sorted_d(&corpus_sigs);
    let surv_sigs: Vec<&Signature> = survivors.iter().map(|(_, s)| s).collect();
    let d_surv = sorted_d(&surv_sigs);
    let edges_all = band_edges(&d_all);
    let edges_surv = band_edges(&d_surv);

    // R3 global centroid over survivors; R5 bands over survivor d-distribution.
    let global_mean = mean_signature(&survivors);
    let bands: Vec<(String, f64, f64)> = vec![
        (format!("p25-p75 [{:.3},{:.3}]", edges_surv.p25, edges_surv.p75), edges_surv.p25, edges_surv.p75),
        (format!("p10-p90 [{:.3},{:.3}]", edges_surv.p10, edges_surv.p90), edges_surv.p10, edges_surv.p90),
    ];

    // ---- load/render the okay+sparse buffet anchors (from cache) ----
    let (buffet, source) = load_or_render_buffet_tiles(
        &args.buffet_hist,
        &args.buffet_json,
        args.candidate_width,
        args.supersample,
        &bins,
    )?;

    // ---- render the fixed control set (the only permissible render here) ----
    let (controls, ctrl_imgs) = render_controls(args, &bins)?;
    write_control_cache(args, &controls)?;
    let sheet = control_sheet(args, &controls, &ctrl_imgs)?;

    // ---- assemble + score the expanded known-answer set ----
    let mut tiles: Vec<ObTile> = Vec::new();
    for t in buffet.iter().chain(controls.iter()) {
        let sparse = t.sig.hist[0][0];
        let d = 1.0 - sparse;
        let r3 = distance(&t.sig, &global_mean, &weights);
        let saturated = (0..4).any(|s| *t.sig.hist[s].last().unwrap() > 0.0);
        let (r5, r5_branch): (Vec<f64>, Vec<&'static str>) = bands
            .iter()
            .map(|&(_, lo, hi)| {
                if d < lo {
                    (lo - d, "below")
                } else if d > hi {
                    (d - hi, "above")
                } else {
                    (0.0, "inside")
                }
            })
            .unzip();
        tiles.push(ObTile {
            id: t.id.clone(),
            cat: Cat::of_loc(&t.loc),
            d,
            sparse,
            r3,
            r5,
            r5_branch,
            saturated,
        });
    }

    overbusy_report(&tiles, &bands, &edges_all, &edges_surv, &sizes, &sparse_sum, q, n, n_surv, source, &sheet, args);
    write_overbusy_json(args, &tiles, &bands)?;
    Ok(())
}

/// d = 1 − s16-bin0 over a signature slice, sorted ascending.
fn sorted_d(sigs: &[&Signature]) -> Vec<f64> {
    let mut d: Vec<f64> = sigs.iter().map(|s| 1.0 - s.hist[0][0]).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d
}

struct BandEdges {
    p10: f64,
    p25: f64,
    p75: f64,
    p90: f64,
}

fn band_edges(sorted_d: &[f64]) -> BandEdges {
    BandEdges {
        p10: pct(sorted_d, 0.10),
        p25: pct(sorted_d, 0.25),
        p75: pct(sorted_d, 0.75),
        p90: pct(sorted_d, 0.90),
    }
}

/// Render the fixed control set; return (signatures, rendered tiles for the sheet).
fn render_controls(
    args: &OverbusyArgs,
    bins: &FrozenBins,
) -> Result<(Vec<TileSig>, Vec<RgbImage>), String> {
    let palette = builtin("default", false).expect("default palette");
    let params = default_color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };
    eprintln!(
        "overbusy: rendering {} fixed control tile(s) at {}px ss{} (f64) ...",
        CONTROLS.len(),
        args.candidate_width,
        args.supersample
    );
    let mut sigs = Vec::with_capacity(CONTROLS.len());
    let mut imgs = Vec::with_capacity(CONTROLS.len());
    for c in CONTROLS {
        let cand = render_candidate(
            Complex::new(c.re, c.im),
            c.width,
            c.maxiter,
            args.candidate_width,
            args.supersample,
            trap,
            &palette,
            &params,
        );
        let regions = region_energies(&cand);
        let sig = bins.signature(&regions);
        let d = 1.0 - sig.hist[0][0];
        eprintln!(
            "  {:<8} [{}] re={:+.6} im={:+.6} w={:.1e} maxiter={} → d={:.3}",
            c.id, c.kind, c.re, c.im, c.width, c.maxiter, d
        );
        sigs.push(TileSig { loc: c.id.to_string(), id: c.id.to_string(), sig });
        imgs.push(cand);
    }
    Ok((sigs, imgs))
}

/// Cache control histograms (repeatable downstream without re-render).
fn write_control_cache(args: &OverbusyArgs, controls: &[TileSig]) -> Result<(), String> {
    crate::ensure_parent_dir(&args.control_hist)?;
    fs::write(&args.control_hist, build_tile_cache_json(controls))
        .map_err(|e| format!("writing control cache {}: {e}", args.control_hist))?;
    eprintln!("overbusy: cached {} control histogram(s) → {}", controls.len(), args.control_hist);
    Ok(())
}

/// Compose the eyeballable control sheet (one labeled tile each) under `out/`.
fn control_sheet(
    args: &OverbusyArgs,
    controls: &[TileSig],
    imgs: &[RgbImage],
) -> Result<String, String> {
    let mut tiles: Vec<RgbImage> = Vec::with_capacity(imgs.len());
    for (c, img) in controls.iter().zip(imgs) {
        let mut th = fit_to(img, args.thumb_width);
        let d = 1.0 - c.sig.hist[0][0];
        let sat = (0..4).any(|s| *c.sig.hist[s].last().unwrap() > 0.0);
        label(&mut th, &format!("{} d={:.3}{}", c.id, d, if sat { " SAT" } else { "" }));
        tiles.push(th);
    }
    let grid = compose_grid(&tiles, Some(3));
    let path = format!("{}/controls.png", args.out_dir.trim_end_matches('/'));
    crate::ensure_parent_dir(&path)?;
    grid.save(&path).map_err(|e| format!("writing {path}: {e}"))?;
    Ok(path)
}

/// Stricter pass: every okay tile must beat (lower score) every sparse tile AND
/// every control. `score` is lower-is-better. `None` if a category is missing.
fn ob_pass(tiles: &[ObTile], score: impl Fn(&ObTile) -> f64) -> Option<bool> {
    let mut worst_okay = f64::MIN;
    let mut best_bad = f64::MAX;
    let (mut has_okay, mut has_bad) = (false, false);
    for t in tiles {
        let s = score(t);
        match t.cat {
            Cat::Okay => {
                worst_okay = worst_okay.max(s);
                has_okay = true;
            }
            _ => {
                best_bad = best_bad.min(s);
                has_bad = true;
            }
        }
    }
    if has_okay && has_bad {
        Some(worst_okay < best_bad)
    } else {
        None
    }
}

/// Print a full best→worst tile ranking under one rule, with PASS/FAIL.
fn ob_rank(label: &str, tiles: &[ObTile], score: impl Fn(&ObTile) -> f64 + Copy) {
    let tag = match ob_pass(tiles, score) {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "INDET",
    };
    let mut rows: Vec<(&ObTile, f64)> = tiles.iter().map(|t| (t, score(t))).collect();
    rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n  {label}  [{tag}]  (best→worst):");
    for (t, s) in rows {
        println!("    {:<14} {:<8} score={:>8.4}  d={:.3}", t.id, t.cat.tag(), s, t.d);
    }
}

#[allow(clippy::too_many_arguments)]
fn overbusy_report(
    tiles: &[ObTile],
    bands: &[(String, f64, f64)],
    edges_all: &BandEdges,
    edges_surv: &BandEdges,
    sizes: &[usize],
    sparse_sum: &[f64],
    q: usize,
    n: usize,
    n_surv: usize,
    source: &str,
    sheet: &str,
    args: &OverbusyArgs,
) {
    println!("\n=== overbusy: expanded known-answer set (okay + sparse + over-busy controls) ===");
    println!(
        "  corpus {n} imgs; quarantined cluster C{q} → {n_surv} survivors; buffet anchors {}; weights {:?}",
        match source { "cache" => "from cache", _ => "RE-RENDERED (flagged)" },
        args.resolved_weights().unwrap_or([1.0; 4]),
    );

    // cluster membership recovery (confirm C{q} is the degenerate n=93 cluster)
    println!("\n  cluster membership (recomputed kmeans seed={} k={}):", args.seed, sizes.len());
    println!("    {:<6} {:>5} {:>16}  {}", "clust", "n", "mean_member_s16b0", "note");
    for c in 0..sizes.len() {
        let msp = if sizes[c] > 0 { sparse_sum[c] / sizes[c] as f64 } else { f64::NAN };
        let note = if c == q { "QUARANTINED (degenerate)" } else { "" };
        println!("    C{:<5} {:>5} {:>16.4}  {}", c, sizes[c], msp, note);
    }

    // before/after quarantine band edges
    println!("\n  corpus density d = 1 − s16-bin0; band edges before vs after C{q} quarantine:");
    println!("    {:<10} {:>9} {:>9} {:>9} {:>9}", "set", "p10", "p25", "p75", "p90");
    println!(
        "    {:<10} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
        "all(746)", edges_all.p10, edges_all.p25, edges_all.p75, edges_all.p90
    );
    println!(
        "    {:<10} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
        "survivors", edges_surv.p10, edges_surv.p25, edges_surv.p75, edges_surv.p90
    );
    let upper_on_ceiling = edges_surv.p75 >= 0.9995 && edges_surv.p90 >= 0.9995;
    println!(
        "    → R5 upper edge after quarantine {} (p75={:.4}, p90={:.4}); {}",
        if upper_on_ceiling { "STILL PINS at the 1.0 ceiling" } else { "moved off 1.0" },
        edges_surv.p75,
        edges_surv.p90,
        if upper_on_ceiling {
            "d saturates at 1.0 by definition → quarantine does NOT restore R5's two-sidedness."
        } else {
            "quarantine moved the upper edge below 1.0."
        }
    );

    // per-tile table
    println!("\n  per-tile scalars + scores (R5 branch per band):");
    println!(
        "    {:<14} {:<8} {:>6} {:>6} {:>8}  {:<22} {:<22}",
        "tile", "cat", "d", "spars", "R3", &bands[0].0, &bands[1].0
    );
    for t in tiles {
        println!(
            "    {:<14} {:<8} {:>6.3} {:>6.3} {:>8.3}  {:<7}v={:<13.3} {:<7}v={:<13.3}",
            t.id,
            t.cat.tag(),
            t.d,
            t.sparse,
            t.r3,
            t.r5_branch[0],
            t.r5[0],
            t.r5_branch[1],
            t.r5[1],
        );
    }

    // rankings + PASS/FAIL per survivor rule. PASS = all okay beat every sparse AND control.
    println!("\n  --- survivor rules vs expanded set (PASS = okay beats ALL sparse + controls) ---");
    ob_rank("R3 global-centroid (C4-quarantined)", tiles, |t| t.r3);
    ob_rank(&format!("R5 band {}", bands[0].0), tiles, |t| t.r5[0]);
    ob_rank(&format!("R5 band {}", bands[1].0), tiles, |t| t.r5[1]);
    // raw s16-bin0 scalar, one-sided "less sparse = better" → rank by sparseness ascending.
    ob_rank("raw s16-bin0 (less-sparse=better)", tiles, |t| t.sparse);

    println!("\nwrote:");
    println!("  {sheet} (control sheet — eyeball for genuine speckle)");
    println!("  {} (control histogram cache)", args.control_hist);
    println!("  {} (per-tile per-rule scores)", args.out_json);
    println!(
        "\nnote: diagnosis only — no winner picked. A rule that ranks a speckle control as \
         good is one-sided; R5's saturated upper edge (d≈1.0) is structurally unable to \
         penalize it."
    );
}

fn write_overbusy_json(
    args: &OverbusyArgs,
    tiles: &[ObTile],
    bands: &[(String, f64, f64)],
) -> Result<(), String> {
    let mut s = String::from("{\n");
    s.push_str(&format!(
        "  \"rules\": {{ \"quarantine_cluster\": {}, \"r5_bands\": [{}] }},\n",
        args.quarantine,
        bands.iter().map(|(l, _, _)| js(l)).collect::<Vec<_>>().join(", "),
    ));
    s.push_str("  \"tiles\": [\n");
    for (i, t) in tiles.iter().enumerate() {
        s.push_str(&format!(
            "    {{ \"id\": {}, \"cat\": {}, \"density_d\": {}, \"sparse_s16b0\": {}, \
\"saturated\": {}, \"r3_global_quarantined\": {}, \"r5_band_violation\": [{}], \
\"r5_branch\": [{}] }}{}\n",
            js(&t.id),
            js(t.cat.tag()),
            jf(t.d),
            jf(t.sparse),
            t.saturated,
            jf(t.r3),
            t.r5.iter().map(|v| jf(*v)).collect::<Vec<_>>().join(", "),
            t.r5_branch.iter().map(|b| js(b)).collect::<Vec<_>>().join(", "),
            if i + 1 < tiles.len() { "," } else { "" },
        ));
    }
    s.push_str("  ]\n}\n");
    crate::ensure_parent_dir(&args.out_json)?;
    fs::write(&args.out_json, s).map_err(|e| format!("writing {}: {e}", args.out_json))?;
    Ok(())
}

// ===========================================================================
// `archetype` — nearest-good-archetype, swept over cluster granularity
// ===========================================================================
//
// Diagnosis-only. Scores the 22-tile known-answer set (18 buffet okay/sparse +
// 4 over-busy/speckle controls, all loaded from the histogram caches — renders
// NOTHING) under one estimator:
//
//     score(tile) = min over the k good centroids of EMD(tile, centroid)
//
// "good" = the k centroids of the corpus re-clustered AFTER quarantining the
// degenerate C4 (the same n=93 cluster overbusy quarantines). Swept over
// k ∈ {5,8,12,16}. This is the missing in-between of nearest-neighbour (rewards
// the sparse tail) and global-centroid (straddles): typical of SOME good mode,
// excluding the degenerate one. Reuses `distance`/`emd1d`/`kmeans`/`Signature`/
// `FrozenBins`/`Cat` unchanged. Picks no winner; produces the table and stops.

/// One tile scored at a single granularity k.
struct ArchScore {
    id: String,
    cat: Cat,
    score: f64,  // min EMD to a good centroid
    arch: usize, // index of the matched centroid
    d: f64,      // 1 − s16-bin0 (context only)
}

/// One re-clustered granularity: centroids (with size + sparseness) and tiles.
struct ArchK {
    k: usize,
    centroid_n: Vec<usize>,
    centroid_s16b0: Vec<f64>,
    tiles: Vec<ArchScore>,
    pass: Option<bool>,
}

pub fn run_archetype(args: &ArchetypeArgs) -> Result<(), String> {
    let weights = args.resolved_weights()?;
    let ks = args.resolved_ks()?;

    // ---- load persisted corpus calibration (frozen bins + 746 histograms) ----
    let text = fs::read_to_string(&args.artifact)
        .map_err(|e| format!("reading artifact {}: {e}", args.artifact))?;
    let (_bins, corpus) = parse_artifact(&text)?;
    let n = corpus.len();
    if n == 0 {
        return Err("artifact has no per-image histograms".into());
    }
    eprintln!("archetype: loaded {n} corpus signatures + frozen bins from {}", args.artifact);

    // ---- corpus s16-bin0 p90 reference (the sparse-survivor bar) ----
    let mut s_corpus: Vec<f64> = corpus.iter().map(|(_, s)| s.hist[0][0]).collect();
    s_corpus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let s_p90 = pct(&s_corpus, 0.90);

    // ---- recover C4 membership, then keep the survivors (renders nothing) ----
    let corpus_sigs: Vec<&Signature> = corpus.iter().map(|(_, s)| s).collect();
    let points: Vec<Vec<f64>> = corpus_sigs.iter().map(|s| s.concat()).collect();
    let kk = args.clusters.max(2).min(n);
    let (assign, _) = kmeans(&points, kk, 20, args.seed);
    let q = args.quarantine;
    let survivors: Vec<Vec<f64>> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| assign[*i] != q)
        .map(|(_, p)| p.clone())
        .collect();
    let n_surv = survivors.len();
    eprintln!("archetype: quarantined C{q} → {n_surv} survivors (re-clustering at k={ks:?})");

    // ---- the 22-tile known-answer set, both caches (no render) ----
    let buffet = parse_tile_cache(
        &fs::read_to_string(&args.buffet_hist)
            .map_err(|e| format!("reading buffet cache {}: {e}", args.buffet_hist))?,
    )?;
    let controls = parse_tile_cache(
        &fs::read_to_string(&args.control_hist)
            .map_err(|e| format!("reading control cache {}: {e}", args.control_hist))?,
    )?;
    if buffet.is_empty() || controls.is_empty() {
        return Err("buffet or control histogram cache is empty".into());
    }
    let tiles: Vec<&TileSig> = buffet.iter().chain(controls.iter()).collect();
    eprintln!(
        "archetype: scoring {} tiles ({} buffet + {} controls)",
        tiles.len(),
        buffet.len(),
        controls.len()
    );

    // ---- per-k: re-cluster survivors, score every tile = min EMD to a centroid ----
    let mut per_k: Vec<ArchK> = Vec::with_capacity(ks.len());
    for &k in &ks {
        let k = k.max(1).min(n_surv);
        let (s_assign, centers) = kmeans(&survivors, k, 20, args.seed);
        let centroid_sigs: Vec<Signature> = centers.iter().map(|c| vec_to_sig(c)).collect();
        let mut centroid_n = vec![0usize; centroid_sigs.len()];
        for &a in &s_assign {
            centroid_n[a] += 1;
        }
        let centroid_s16b0: Vec<f64> = centroid_sigs.iter().map(|s| s.hist[0][0]).collect();

        let mut scored: Vec<ArchScore> = Vec::with_capacity(tiles.len());
        for t in &tiles {
            let mut best = f64::INFINITY;
            let mut arch = 0;
            for (a, cs) in centroid_sigs.iter().enumerate() {
                let dd = distance(&t.sig, cs, &weights);
                if dd < best {
                    best = dd;
                    arch = a;
                }
            }
            scored.push(ArchScore {
                id: t.id.clone(),
                cat: Cat::of_loc(&t.loc),
                score: best,
                arch,
                d: 1.0 - t.sig.hist[0][0],
            });
        }
        let pass = arch_pass(&scored);
        per_k.push(ArchK { k, centroid_n, centroid_s16b0, tiles: scored, pass });
    }

    archetype_report(args, &per_k, &weights, s_p90, n, n_surv);
    write_archetype_json(args, &per_k, &weights, s_p90, n, n_surv)?;
    Ok(())
}

/// Stricter per-tile pass: every okay tile must beat (lower score) every sparse
/// tile AND every control. `None` if a category is missing.
fn arch_pass(scored: &[ArchScore]) -> Option<bool> {
    let mut worst_okay = f64::MIN;
    let mut best_bad = f64::MAX;
    let (mut has_okay, mut has_bad) = (false, false);
    for t in scored {
        match t.cat {
            Cat::Okay => {
                worst_okay = worst_okay.max(t.score);
                has_okay = true;
            }
            _ => {
                best_bad = best_bad.min(t.score);
                has_bad = true;
            }
        }
    }
    if has_okay && has_bad {
        Some(worst_okay < best_bad)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn archetype_report(
    args: &ArchetypeArgs,
    per_k: &[ArchK],
    weights: &[f64; 4],
    s_p90: f64,
    n: usize,
    n_surv: usize,
) {
    println!("\n=== archetype: nearest-good-archetype, swept over cluster granularity ===");
    println!(
        "  corpus {n} imgs; quarantined C{} → {n_surv} survivors; re-clustered at k={:?}; seed={}; weights {:?}",
        args.quarantine,
        per_k.iter().map(|p| p.k).collect::<Vec<_>>(),
        args.seed,
        weights,
    );
    println!(
        "  score(tile) = min over good centroids of EMD(tile, centroid); lower = more typical of some good mode."
    );
    println!("  PASS = every okay (B1/B2/B4/B5) beats every sparse (B0/B3) AND every control (OB_*).");
    println!("  sparse-survivor bar: corpus s16-bin0 p90 = {s_p90:.3} (centroid above ⇒ a sparse archetype).");

    for pk in per_k {
        let tag = match pk.pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "INDET",
        };
        println!("\n  --- k={} [{tag}] full 22-tile ranking (best→worst) ---", pk.k);
        let mut rows: Vec<&ArchScore> = pk.tiles.iter().collect();
        rows.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        for t in &rows {
            println!(
                "    {:<14} {:<8} score={:>8.4}  arch=A{:<2} d={:.3}",
                t.id,
                t.cat.tag(),
                t.score,
                t.arch,
                t.d
            );
        }

        // diagnostic 1 — speckle-rewarding archetype: each control's match.
        println!("    control matches (arch | EMD):");
        for t in pk.tiles.iter().filter(|t| t.cat == Cat::Control) {
            println!("      {:<8} A{:<2} d={:.4}", t.id, t.arch, t.score);
        }
        let worst_control = pk
            .tiles
            .iter()
            .filter(|t| t.cat == Cat::Control)
            .map(|t| t.score)
            .fold(f64::MIN, f64::max);

        // diagnostic 2 — straddle persistence: the R3 straddlers vs worst control.
        println!("    straddle check (R3 straddlers B5_ON_DEEP, B4_FAR_DEEP vs worst control d={worst_control:.4}):");
        for sid in ["B5_ON_DEEP", "B4_FAR_DEEP"] {
            if let Some(t) = pk.tiles.iter().find(|t| t.id == sid) {
                let gap = t.score - worst_control;
                let rel = if gap < 0.0 { "beats all controls" } else { "STILL below a control" };
                println!(
                    "      {:<14} A{:<2} d={:.4}  gap_to_worst_control={:+.4}  ({rel})",
                    t.id, t.arch, t.score, gap
                );
            }
        }

        // diagnostic 3 — surviving sparse mode: any centroid itself sparse?
        let sparse_archs: Vec<usize> = (0..pk.centroid_s16b0.len())
            .filter(|&c| pk.centroid_s16b0[c] >= s_p90)
            .collect();
        println!("    centroid s16-bin0 (n) [SPARSE if ≥ p90={s_p90:.3}]:");
        let body = (0..pk.centroid_s16b0.len())
            .map(|c| {
                format!(
                    "A{}={:.3}(n{}){}",
                    c,
                    pk.centroid_s16b0[c],
                    pk.centroid_n[c],
                    if pk.centroid_s16b0[c] >= s_p90 { "*" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("      {body}");
        if sparse_archs.is_empty() {
            println!("      → no surviving sparse archetype (B0/B3 cannot match a sparse mode cheaply).");
        } else {
            println!(
                "      → {} sparse archetype(s) survive: {:?} — B0/B3 may match these cheaply.",
                sparse_archs.len(),
                sparse_archs
            );
        }
    }

    println!("\nwrote:");
    println!("  {} (per-tile per-k scores + matched centroids)", args.out_json);
    println!(
        "\nnote: diagnosis only — no rule picked, nothing wired. If a straddler stays below a control \
         across all k, the energy histogram is coherence-blind by construction (pooled region-energy \
         bag discards spatial arrangement); that points at a spatial-coherence FEATURE, not another rule."
    );
}

#[allow(clippy::too_many_arguments)]
fn write_archetype_json(
    args: &ArchetypeArgs,
    per_k: &[ArchK],
    weights: &[f64; 4],
    s_p90: f64,
    n: usize,
    n_surv: usize,
) -> Result<(), String> {
    let mut s = String::from("{\n");
    s.push_str(&format!(
        "  \"params\": {{ \"weights\": [{}], \"init_clusters\": {}, \"quarantine_cluster\": {}, \
\"seed\": {}, \"n_corpus\": {}, \"n_survivors\": {}, \"corpus_s16b0_p90\": {} }},\n",
        weights.iter().map(|w| jf(*w)).collect::<Vec<_>>().join(", "),
        args.clusters,
        args.quarantine,
        args.seed,
        n,
        n_surv,
        jf(s_p90),
    ));
    s.push_str("  \"ks\": [\n");
    for (ki, pk) in per_k.iter().enumerate() {
        let pass = match pk.pass {
            Some(b) => b.to_string(),
            None => "null".to_string(),
        };
        s.push_str(&format!("    {{ \"k\": {}, \"pass\": {pass},\n", pk.k));
        let centroids: Vec<String> = (0..pk.centroid_n.len())
            .map(|c| {
                format!(
                    "{{ \"idx\": {}, \"n\": {}, \"s16b0\": {}, \"sparse\": {} }}",
                    c,
                    pk.centroid_n[c],
                    jf(pk.centroid_s16b0[c]),
                    pk.centroid_s16b0[c] >= s_p90
                )
            })
            .collect();
        s.push_str(&format!("      \"centroids\": [{}],\n", centroids.join(", ")));
        s.push_str("      \"tiles\": [\n");
        for (ti, t) in pk.tiles.iter().enumerate() {
            s.push_str(&format!(
                "        {{ \"id\": {}, \"cat\": {}, \"score\": {}, \"arch\": {}, \"density_d\": {} }}{}\n",
                js(&t.id),
                js(t.cat.tag()),
                jf(t.score),
                t.arch,
                jf(t.d),
                if ti + 1 < pk.tiles.len() { "," } else { "" },
            ));
        }
        s.push_str("      ]\n");
        s.push_str(&format!("    }}{}\n", if ki + 1 < per_k.len() { "," } else { "" }));
    }
    s.push_str("  ]\n}\n");
    crate::ensure_parent_dir(&args.out_json)?;
    fs::write(&args.out_json, s).map_err(|e| format!("writing {}: {e}", args.out_json))?;
    Ok(())
}
