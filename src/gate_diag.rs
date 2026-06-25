//! `gate-diag` — **measurement-only** signal extractor for the reject-the-bad
//! gate study (CC prompt: signal-separation diagnostic).
//!
//! This builds nothing and gates nothing. For each distinct `(draw_index,
//! composition)` geometry recorded in a `present` manifest, it re-renders the
//! cheap f64 screen at the **stored crop frame** (the manifest persists the exact
//! final `cx`/`cy`/`fw` per crop — no composition math to reconstruct) and emits a
//! row of candidate gate signals to `signals.csv`. The label join, ROC/AUC
//! analysis, and the visual sheet all happen Python-side off that CSV.
//!
//! Two signal families (both palette-independent geometry/dynamics):
//!
//! **Dynamics-space** (from a fresh DE render — reuses the production f64 kernel
//! via [`render::iterate_samples_f64`] with the `de` channel on, `trap` off):
//!  - `interior_frac` — non-escaping pixel fraction (sanity-checks against the
//!    manifest's black fraction).
//!  - `de_small_frac[k]` — fraction of *escaped* pixels whose distance estimate,
//!    expressed in **final-wallpaper pixels** (`de_px = de / (fw / de_ref_width)`,
//!    pinned to 2560 so it is independent of the diagnostic screen size), is below
//!    `k ∈ {0.5, 1, 2, 4}`. This is the rigorous form of "huge chunks of
//!    high-escape boundary points": a thicket sits *on* the boundary, so its
//!    escaped pixels have a tiny DE.
//!  - `slow_escape_frac` — derivative-free proxy: fraction of escaped pixels whose
//!    escape iteration is near `maxiter` (`smooth_iter ≥ 0.9·maxiter`).
//!
//! **Image-space** (the `energy.rs` descriptor on the representative JPG, binned
//! under the **frozen corpus quantile bins** — no reinvented filter):
//!  - `fine_energy_frac` — finest-scale (s16) **density** = `1 − hist[s16][bin0]`:
//!    the fraction of 16×16 region tiles above the corpus's lowest energy quantile.
//!    This is exactly `rescore`'s `d` scalar, read at the finest scale.
//!  - `coarse_density` — the same at the coarsest scale (s2), for a fine-vs-coarse
//!    contrast.
//!  - `energy_mean` — global mean OKLab edge energy (raw busyness).
//!
//! DE needs a large bailout (≥~1e6) for a stable estimate; the default 1e6 (≈2^20)
//! matches `present`/`generate` and is already in the ideal band.

use std::fs;
use std::path::Path;

use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::GateDiagArgs;
use crate::coloring::ChannelSet;
use crate::energy::{self, region_energies};
use crate::render::{self, Frame};

/// DE-in-pixels thresholds swept for `de_small_frac` (final-wallpaper px units).
const DE_K: [f64; 4] = [0.5, 1.0, 2.0, 4.0];

/// One labeled crop geometry (a distinct `(draw_index, composition)`), carrying
/// the stored crop frame and the manifest scalars passed through to the CSV.
struct Geometry {
    draw_index: usize,
    seed_index: usize,
    composition: String,
    cx: f64,
    cy: f64,
    fw: f64,
    occupancy: f64,
    black_fraction: f64,
    interior_frac_manifest: f64,
    /// Representative JPG (the first palette per geometry — occupancy/dynamics are
    /// palette-invariant, so any palette's crop has the identical structure).
    jpg: String,
}

/// All computed signals for one geometry.
struct Signals {
    interior_frac: f64,
    n_escaped: usize,
    de_small_frac: [f64; 4],
    slow_escape_frac: f64,
    fine_energy_frac: f64,
    coarse_density: f64,
    energy_mean: f64,
}

// ---------- hand-rolled manifest field parsers (one crop record per line) -----
// Field readers shared via `crate::jsonl` (the canonical copy).
use crate::jsonl::*;

/// Parse the manifest's `crops` array into one [`Geometry`] per distinct
/// `(draw_index, composition)` (first palette row kept as the representative).
fn parse_manifest(text: &str) -> Result<Vec<Geometry>, String> {
    let mut seen: std::collections::BTreeSet<(usize, String)> = std::collections::BTreeSet::new();
    let mut out: Vec<Geometry> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.contains("\"draw_index\":") || !line.contains("\"output\":") {
            continue;
        }
        let draw_index = field_usize(line, "draw_index")
            .ok_or_else(|| format!("manifest: bad draw_index in: {line}"))?;
        let composition = field_str(line, "composition")
            .ok_or_else(|| format!("manifest: bad composition in: {line}"))?;
        if !seen.insert((draw_index, composition.clone())) {
            continue; // already have this geometry (another palette row)
        }
        out.push(Geometry {
            draw_index,
            seed_index: field_usize(line, "seed_index").unwrap_or(usize::MAX),
            composition,
            cx: field_f64(line, "cx").ok_or("manifest: bad cx")?,
            cy: field_f64(line, "cy").ok_or("manifest: bad cy")?,
            fw: field_f64(line, "fw").ok_or("manifest: bad fw")?,
            occupancy: field_f64(line, "occupancy").unwrap_or(f64::NAN),
            black_fraction: field_f64(line, "black_fraction").unwrap_or(f64::NAN),
            interior_frac_manifest: field_f64(line, "interior_frac").unwrap_or(f64::NAN),
            jpg: field_str(line, "output").ok_or("manifest: bad output")?,
        });
    }
    Ok(out)
}

// ---------- per-geometry signal computation ----------------------------------

fn compute_signals(g: &Geometry, args: &GateDiagArgs, bins: &energy::FrozenBins) -> Signals {
    // --- dynamics: fresh DE render at the stored crop frame ---
    let height = (args.screen_width as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let frame = Frame {
        center: Complex::new(g.cx, g.cy),
        frame_width: g.fw,
        out_width: args.screen_width,
        out_height: height,
    };
    // Trap channel off (not read); DE on. Trap value is unused but the ctor needs one.
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let backend = F64Backend::new(args.maxiter, args.bailout, trap);
    let channels = ChannelSet { trap: false, de: true };
    let buf = render::iterate_samples_f64(&backend, &frame, 1, channels);

    // DE pinned to the final-wallpaper pixel scale: de_px = de / (fw / de_ref_width).
    let ref_spacing = g.fw / args.de_ref_width as f64;
    let slow_iter_thresh = 0.9 * args.maxiter as f64;

    let total = buf.samples.len();
    let mut n_escaped = 0usize;
    let mut de_below = [0usize; 4];
    let mut slow = 0usize;
    for s in &buf.samples {
        if !s.escaped {
            continue;
        }
        n_escaped += 1;
        let de_px = s.de / ref_spacing;
        for (i, &k) in DE_K.iter().enumerate() {
            if de_px < k {
                de_below[i] += 1;
            }
        }
        if s.smooth_iter >= slow_iter_thresh {
            slow += 1;
        }
    }
    let interior_frac = if total > 0 {
        (total - n_escaped) as f64 / total as f64
    } else {
        f64::NAN
    };
    let denom = n_escaped.max(1) as f64;
    let de_small_frac = std::array::from_fn(|i| {
        if n_escaped == 0 { f64::NAN } else { de_below[i] as f64 / denom }
    });
    let slow_escape_frac = if n_escaped == 0 { f64::NAN } else { slow as f64 / denom };

    // --- image-space: energy descriptor on the representative JPG ---
    let (fine_energy_frac, coarse_density, energy_mean) = match image::open(&g.jpg) {
        Ok(im) => {
            let rgb = im.to_rgb8();
            let regions = region_energies(&rgb);
            let sig = bins.signature(&regions);
            // density = 1 − fraction in the lowest-energy quantile bin (bin 0).
            let fine = 1.0 - sig.hist[0][0]; // s16 (finest)
            let coarse = 1.0 - sig.hist[3][0]; // s2 (coarsest)
            // global mean edge energy = mean over any scale's per-area region means.
            let s2 = &regions[3];
            let mean = if s2.is_empty() { f64::NAN } else { s2.iter().sum::<f64>() / s2.len() as f64 };
            (fine, coarse, mean)
        }
        Err(e) => {
            eprintln!("  warn: decode {} failed ({e}); energy signals = NaN", g.jpg);
            (f64::NAN, f64::NAN, f64::NAN)
        }
    };

    Signals {
        interior_frac,
        n_escaped,
        de_small_frac,
        slow_escape_frac,
        fine_energy_frac,
        coarse_density,
        energy_mean,
    }
}

// ---------- entry point ------------------------------------------------------

pub fn run_gate_diag(args: &GateDiagArgs) -> Result<(), String> {
    let manifest_text = fs::read_to_string(&args.manifest)
        .map_err(|e| format!("read {}: {e}", args.manifest))?;
    let geometries = parse_manifest(&manifest_text)?;
    if geometries.is_empty() {
        return Err("no crop geometries parsed from manifest".into());
    }
    eprintln!(
        "gate-diag: {} distinct (draw,comp) geometries from {}",
        geometries.len(),
        args.manifest
    );

    let artifact_text = fs::read_to_string(&args.artifact)
        .map_err(|e| format!("read {}: {e}", args.artifact))?;
    let (bins, _corpus) = energy::parse_artifact(&artifact_text)?;
    eprintln!("gate-diag: frozen energy bins loaded from {}", args.artifact);
    eprintln!(
        "gate-diag: rendering DE screens at {}px (maxiter {}, bailout {:.0}, de pinned to {}px) + \
         energy descriptor per JPG ...",
        args.screen_width, args.maxiter, args.bailout, args.de_ref_width
    );

    let t0 = std::time::Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let n = geometries.len();
    let results: Vec<Signals> = geometries
        .par_iter()
        .map(|g| {
            let s = compute_signals(g, args, &bins);
            let c = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if c % 50 == 0 || c == n {
                eprintln!("  {c}/{n} geometries ({:.1}s)", t0.elapsed().as_secs_f64());
            }
            s
        })
        .collect();
    eprintln!("gate-diag: signals computed in {:.1}s", t0.elapsed().as_secs_f64());

    // --- write signals.csv ---
    let out_dir = Path::new(&args.out_dir);
    crate::ensure_parent_dir(out_dir.join("x"))?;
    let csv_path = out_dir.join("signals.csv");
    let mut csv = String::new();
    csv.push_str(
        "draw_index,seed_index,composition,cx,cy,fw,occupancy,black_fraction,\
         interior_frac_manifest,interior_frac,n_escaped,\
         de_small_frac_0p5,de_small_frac_1,de_small_frac_2,de_small_frac_4,\
         slow_escape_frac,fine_energy_frac,coarse_density,energy_mean,jpg\n",
    );
    let g = |x: f64| if x.is_finite() { format!("{x:.6}") } else { "nan".to_string() };
    for (geo, s) in geometries.iter().zip(&results) {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            geo.draw_index,
            geo.seed_index,
            geo.composition,
            // full-precision frame so the geometry can be re-rendered exactly.
            geo.cx,
            geo.cy,
            geo.fw,
            g(geo.occupancy),
            g(geo.black_fraction),
            g(geo.interior_frac_manifest),
            g(s.interior_frac),
            s.n_escaped,
            g(s.de_small_frac[0]),
            g(s.de_small_frac[1]),
            g(s.de_small_frac[2]),
            g(s.de_small_frac[3]),
            g(s.slow_escape_frac),
            g(s.fine_energy_frac),
            g(s.coarse_density),
            g(s.energy_mean),
            geo.jpg.replace('\\', "/"),
        ));
    }
    fs::write(&csv_path, csv).map_err(|e| format!("write {}: {e}", csv_path.display()))?;

    // --- quick distribution sanity (so the run reports something useful) ---
    let mut de1: Vec<f64> = results.iter().map(|s| s.de_small_frac[1]).filter(|x| x.is_finite()).collect();
    de1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |v: &[f64], p: f64| -> f64 {
        if v.is_empty() { return f64::NAN; }
        v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
    };
    println!("=== gate-diag (measurement only) ===");
    println!("geometries: {n}  elapsed: {:.1}s", t0.elapsed().as_secs_f64());
    println!(
        "de_small_frac(k=1px) over geometries: min {:.3}  p25 {:.3}  med {:.3}  p75 {:.3}  p90 {:.3}  max {:.3}",
        q(&de1, 0.0), q(&de1, 0.25), q(&de1, 0.5), q(&de1, 0.75), q(&de1, 0.9), q(&de1, 1.0),
    );
    println!("wrote: {}", csv_path.display());
    Ok(())
}
