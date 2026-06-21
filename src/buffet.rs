//! `buffet` — visual-first sampling of what "magical density" looks like.
//!
//! Deliberately **un-engineered** (Prompt visual-buffet-v2). The coverage-dominance
//! harvest produced a structural finding: a minibrot-centered frame is necessarily
//! black body + near-cusp speckle annulus + flat far-field, so in-band coverage is
//! capped low *by construction* — nucleus-centering can't reach the target. The
//! diagnostic also told us what the target *is*: **scale-uniform decoration, away
//! from any cusp** — low `interior_frac`, `de_px` tightly spread (not bimodal).
//! This subcommand goes and finds what that looks like, by eye.
//!
//! **No objective, no frontier, no ranking, no drift.** It samples three sources,
//! renders each across a small grid, labels every tile with its metrics, and
//! composes one sheet per source. The metrics are captions only — nothing here
//! selects or scores beyond the source-B signature filter, which is itself on trial.
//!
//! Sources:
//!  - **(A) Main-set boundary neighborhoods** — a coarse scan of the *main* set's
//!    boundary (the smallest-`de` exterior pixels of one wide probe, spatially
//!    spread), where shallow scale-uniform decoration lives. No Newton, no descent.
//!  - **(B) Off-cusp signature-filtered sample** — a broad random scan kept by a
//!    light two-threshold filter (`interior_frac` low, in-band `de_px` fraction
//!    high). Not a scorer — "show me frames with the signature the diagnostic
//!    pointed at" so it can be confirmed or rejected visually.
//!  - **(C) Minibrot neighborhoods** — a handful via the `navigate` primitives
//!    (atom → Newton → size), the visual control expected to look *worse*.
//!
//! Each location is rendered over two axes: **off-boundary offset** (stepped along
//! the outward normal of the DE field, on-boundary → mid-field → flat) and **scale**
//! (a few log-spaced widths). Everything is f64 cheap-regime (asserted); the labels
//! reuse [`crate::coherence::coverage_stats`] / [`crate::coherence::windowed_busyness_max`]
//! unchanged — the scorers are read, never edited.

use std::fs;

use astro_float::BigFloat;
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::Trap;
use crate::cli::{BackendChoice, BuffetArgs};
use crate::coherence::{coverage_stats, windowed_busyness_max, CoverageStats, COVER_HI, COVER_LO};
use crate::coloring::ColorParams;
use crate::font;
use crate::hp;
use crate::navigate::{atom_candidates, newton_nucleus, size_estimate};
use crate::palette::Palette;
use crate::palette_io::load_palette;
use crate::probe::{self, SplitMix64};
use crate::render::{self, SampleBuffer};

/// Off-boundary offsets along the outward normal, in units of the tile's **own**
/// half-width. `ON` ≈ boundary-centered, `MID` ≈ mid-field, `FAR` ≈ flat exterior.
/// Using the tile's own half-width keeps the progression coherent at every scale.
const OFFSETS: [(&str, f64); 3] = [("ON", 0.0), ("MID", 0.5), ("FAR", 1.2)];

/// Scale multipliers on each location's base width (the columns), wide → deep.
const SCALES: [(&str, f64); 3] = [("WIDE", 4.0), ("BASE", 1.0), ("DEEP", 0.25)];

/// Probe width for the per-location outward-normal estimate (cheap; one render).
const NORMAL_PROBE_W: u32 = 160;

/// Reference width the maxiter schedule is anchored at (the base-set view width).
const REF_WIDTH: f64 = 3.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    A,
    B,
    C,
}

impl Source {
    fn tag(self) -> &'static str {
        match self {
            Source::A => "A",
            Source::B => "B",
            Source::C => "C",
        }
    }
}

/// A sampled location: a base center + base width, plus an optional note (the
/// minibrot period for source C). The offset × scale grid is rendered around it.
struct Location {
    source: Source,
    idx: usize,
    center: Complex<f64>,
    base_width: f64,
    note: String,
}

/// Shared per-run context threaded into the tile renderer (keeps arg lists sane).
struct Ctx<'a> {
    pw: u32,
    ph: u32,
    ss: u32,
    target_width: u32,
    theta: f64,
    window: u32,
    bailout: f64,
    maxiter_base: f64,
    per_decade: f64,
    palette: &'a Palette,
    params: &'a ColorParams,
    trap: Trap,
}

/// One tile's flat metrics record (the per-id table row, captions, and JSON).
struct TileRecord {
    id: String,
    source: &'static str,
    loc_idx: usize,
    off: &'static str,
    scale: &'static str,
    note: String,
    center_re: f64,
    center_im: f64,
    width: f64,
    maxiter: u32,
    coverage: f64,
    subpixel_frac: f64,
    interior_frac: f64,
    esc_frac: f64,
    de_px_median: f64,
    de_px_iqr: f64,
    busy_win: f64,
}

// ===========================================================================
// Entry point
// ===========================================================================

pub fn run_buffet(args: &BuffetArgs) -> Result<(), String> {
    if args.panel_width == 0 {
        return Err("--panel-width must be > 0".into());
    }
    if args.supersample == 0 {
        return Err("--supersample must be > 0".into());
    }
    if args.target_width == 0 {
        return Err("--target-width must be > 0".into());
    }

    let pw = args.panel_width;
    let ph = ((pw as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample.max(1);

    let palette = load_palette(
        &args.palette.palette,
        args.palette.palette_entry.as_deref(),
        args.palette.palette_reverse,
    )?;
    let params = probe::color_params(&args.shade);
    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };
    let region = args.resolved_scan_region()?;
    let mut rng = SplitMix64(args.seed);

    let ctx = Ctx {
        pw,
        ph,
        ss,
        target_width: args.target_width,
        theta: args.theta,
        window: args.window,
        bailout: args.bailout,
        maxiter_base: args.maxiter_base,
        per_decade: args.per_decade,
        palette: &palette,
        params: &params,
        trap,
    };

    // --- scan the three sources (selection only; no ranking) ---
    eprintln!("buffet: scanning sources (f64 cheap-regime)...");
    let t_scan = std::time::Instant::now();
    let a = scan_boundary(&ctx, args.a_count, args.base_width_a, args.interior_max);
    let b = scan_signature(
        &ctx,
        &mut rng,
        args.b_count,
        args.base_width_b,
        region,
        args.scan_tries,
        args.b_interior_max,
        args.b_coverage_min,
        args.interior_max,
    );
    let c = scan_minibrots(&ctx, args.c_count, args.frame_multiple);
    eprintln!(
        "  scanned in {:.1}s → A={} B={} C={} locations",
        t_scan.elapsed().as_secs_f64(),
        a.len(),
        b.len(),
        c.len()
    );

    let per_loc = OFFSETS.len() * SCALES.len();
    let total = (a.len() + b.len() + c.len()) * per_loc;
    eprintln!(
        "  rendering {total} tiles ({pw}x{ph} ss{ss}, {} offsets × {} scales/loc) — \
         background recommended",
        OFFSETS.len(),
        SCALES.len(),
    );

    fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("failed to create {}: {e}", args.out_dir))?;

    // --- render + compose one sheet per source ---
    let mut records: Vec<TileRecord> = Vec::with_capacity(total);
    let t_render = std::time::Instant::now();
    for locs in [&a, &b, &c] {
        if locs.is_empty() {
            continue;
        }
        let tag = locs[0].source.tag();
        let mut tiles: Vec<RgbImage> = Vec::with_capacity(locs.len() * per_loc);
        for loc in locs {
            let (mut imgs, mut recs) = render_location(&ctx, loc);
            tiles.append(&mut imgs);
            records.append(&mut recs);
        }
        let grid = crate::sheet::compose_grid(&tiles, Some(SCALES.len()));
        let path = format!("{}/buffet_{}.png", args.out_dir.trim_end_matches('/'), tag);
        crate::ensure_parent_dir(&path)?;
        grid.save(&path)
            .map_err(|e| format!("failed to write {path}: {e}"))?;
        eprintln!(
            "  wrote {path} ({} tiles, rows=offset × cols=scale)",
            locs.len() * per_loc
        );
    }
    eprintln!("  rendered in {:.1}s", t_render.elapsed().as_secs_f64());

    // --- flat per-tile metrics table: stdout + JSON ---
    print_table(&records);
    let json = build_json(args, &records);
    crate::ensure_parent_dir(&args.json)?;
    fs::write(&args.json, json).map_err(|e| format!("failed to write {}: {e}", args.json))?;
    eprintln!("wrote {} ({} tiles)", args.json, records.len());

    Ok(())
}

// ===========================================================================
// Per-location offset × scale grid
// ===========================================================================

/// Render one location's offset × scale grid: estimate the outward normal once,
/// then render `OFFSETS × SCALES` tiles (row-major: offset outer, scale inner, so
/// `compose_grid(cols = SCALES.len())` lays out rows = offset, columns = scale).
fn render_location(ctx: &Ctx, loc: &Location) -> (Vec<RgbImage>, Vec<TileRecord>) {
    // Outward normal from a cheap probe at the base frame (DE field: exterior vs
    // interior centroid points from the set body out into the exterior).
    let normal = {
        let nph = ((NORMAL_PROBE_W as f64) * 9.0 / 16.0).round().max(1.0) as u32;
        let mi = sched_maxiter(ctx.maxiter_base, ctx.per_decade, loc.base_width);
        let buf = render_buf(loc.center, loc.base_width, NORMAL_PROBE_W, nph, ctx.ss, mi, ctx.bailout, ctx.trap);
        outward_normal(&buf, loc.base_width).unwrap_or_else(|| Complex::new(1.0, 0.0))
    };

    let mut imgs = Vec::with_capacity(OFFSETS.len() * SCALES.len());
    let mut recs = Vec::with_capacity(OFFSETS.len() * SCALES.len());
    for (oname, ofrac) in OFFSETS {
        for (sname, smult) in SCALES {
            let width = loc.base_width * smult;
            let center = loc.center + normal.scale(ofrac * width * 0.5);
            let maxiter = sched_maxiter(ctx.maxiter_base, ctx.per_decade, width);
            let buf = render_buf(center, width, ctx.pw, ctx.ph, ctx.ss, maxiter, ctx.bailout, ctx.trap);

            let cov = coverage_stats(&buf, width, ctx.target_width, ctx.theta, COVER_LO, COVER_HI);
            let busy = windowed_busyness_max(&buf, ctx.pw, ctx.ph, ctx.window as i32, maxiter);
            let iqr = de_px_iqr(&buf, width, ctx.target_width);

            let mut img = render::shade_and_downsample(
                &buf.samples,
                ctx.pw,
                ctx.ph,
                ctx.ss,
                ctx.palette,
                ctx.params,
                width / ctx.pw as f64,
            );
            draw_caption(&mut img, loc, oname, sname, &cov, busy, iqr);

            recs.push(TileRecord {
                id: format!("{}{}_{}_{}", loc.source.tag(), loc.idx, oname, sname),
                source: loc.source.tag(),
                loc_idx: loc.idx,
                off: oname,
                scale: sname,
                note: loc.note.clone(),
                center_re: center.re,
                center_im: center.im,
                width,
                maxiter,
                coverage: cov.coverage,
                subpixel_frac: cov.subpixel_frac,
                interior_frac: cov.interior_frac,
                esc_frac: cov.escaped as f64 / cov.total.max(1) as f64,
                de_px_median: cov.de_px_median,
                de_px_iqr: iqr,
                busy_win: busy,
            });
            imgs.push(img);
        }
    }
    (imgs, recs)
}

/// Three-line caption: id/offset/scale, then two metric lines.
fn draw_caption(
    img: &mut RgbImage,
    loc: &Location,
    off: &str,
    scale: &str,
    cov: &CoverageStats,
    busy: f64,
    iqr: f64,
) {
    let note = if loc.note.is_empty() {
        String::new()
    } else {
        format!(" {}", loc.note)
    };
    let l1 = format!("{}{} {} {}{}", loc.source.tag(), loc.idx, off, scale, note);
    let l2 = format!("COV{} DE{} IQ{}", cn(cov.coverage, 2), cn(cov.de_px_median, 1), cn(iqr, 1));
    let l3 = format!("SPX{} INT{} B{}", cn(cov.subpixel_frac, 2), cn(cov.interior_frac, 2), cn(busy, 3));
    let col = Rgb([235u8, 235, 235]);
    font::draw_text(img, &l1.to_uppercase(), 1, 1, 1, col, true);
    font::draw_text(img, &l2.to_uppercase(), 1, 13, 1, col, true);
    font::draw_text(img, &l3.to_uppercase(), 1, 25, 1, col, true);
}

// ===========================================================================
// Source A — main-set boundary neighborhoods (coarse small-de scan)
// ===========================================================================

/// Coarse-scan the *main* set boundary: render one wide probe, take the
/// smallest-`de` exterior pixels (nearest the boundary), spatially dedup so the
/// kept points spread across distinct valleys, and frame each at `base_width`.
/// No Newton, no descent — just "where does the main boundary live".
fn scan_boundary(ctx: &Ctx, n: usize, base_width: f64, interior_max: f64) -> Vec<Location> {
    if n == 0 {
        return Vec::new();
    }
    // Wide 3:2 probe of the whole set (center -0.6 frames re∈[-2.1,0.9]).
    let scan_w = 480u32;
    let scan_h = (scan_w as f64 * 2.0 / 3.0).round() as u32;
    let width = 3.0;
    let center = Complex::new(-0.6, 0.0);
    let maxiter = 800;
    let buf = render_buf(center, width, scan_w, scan_h, ctx.ss, maxiter, ctx.bailout, ctx.trap);

    // Per-output-pixel min de over escaped subsamples + escaped flag.
    let w = scan_w as usize;
    let h = scan_h as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;
    let mut min_de = vec![f64::INFINITY; w * h];
    let mut escaped = vec![false; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut esc = 0usize;
            let mut de = f64::INFINITY;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.escaped {
                        esc += 1;
                        if px.de < de {
                            de = px.de;
                        }
                    }
                }
            }
            let idx = row * w + col;
            escaped[idx] = esc * 2 >= s * s;
            min_de[idx] = de;
        }
    }

    // Spatial dedup: keep the smallest-de exterior pixel per grid cell.
    let cell = (scan_w / 14).max(8) as usize;
    let grid_w = w.div_ceil(cell);
    use std::collections::HashMap;
    let mut best: HashMap<usize, (f64, usize, usize)> = HashMap::new(); // cell -> (de, col, row)
    let margin = 3usize;
    for row in margin..h - margin {
        for col in margin..w - margin {
            let idx = row * w + col;
            if !escaped[idx] || !min_de[idx].is_finite() || min_de[idx] <= 0.0 {
                continue;
            }
            let key = (row / cell) * grid_w + (col / cell);
            let de = min_de[idx];
            best.entry(key)
                .and_modify(|b| {
                    if de < b.0 {
                        *b = (de, col, row);
                    }
                })
                .or_insert((de, col, row));
        }
    }

    // Smallest-de cells first (closest to the boundary), then take `n`.
    let mut cells: Vec<(f64, usize, usize)> = best.into_values().collect();
    cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let fw = width;
    let fh = width * (scan_h as f64 / scan_w as f64);
    let mut out = Vec::new();
    for (_de, col, row) in cells {
        if out.len() >= n {
            break;
        }
        let cr = center.re + ((col as f64 + 0.5) / w as f64 - 0.5) * fw;
        let ci = center.im + (0.5 - (row as f64 + 0.5) / h as f64) * fh;
        let loc_center = Complex::new(cr, ci);
        // Light pre-filter: drop a pure-interior base frame (boundary points
        // normally clear this, but a point that landed inside a bay would not).
        if pre_filter_reject(ctx, loc_center, base_width, interior_max) {
            continue;
        }
        out.push(Location {
            source: Source::A,
            idx: out.len(),
            center: loc_center,
            base_width,
            note: String::new(),
        });
    }
    out
}

// ===========================================================================
// Source B — off-cusp signature-filtered random sample
// ===========================================================================

/// Broad random scan over the main-set neighborhood; keep frames matching the
/// target *signature* via a light two-threshold filter (`interior_frac` low,
/// in-band de_px fraction `coverage` high). This is the filter on trial — it
/// selects what to *show*, it does not rank anything.
#[allow(clippy::too_many_arguments)]
fn scan_signature(
    ctx: &Ctx,
    rng: &mut SplitMix64,
    n: usize,
    base_width: f64,
    region: (f64, f64, f64, f64),
    tries: usize,
    interior_max_b: f64,
    coverage_min: f64,
    interior_max: f64,
) -> Vec<Location> {
    if n == 0 {
        return Vec::new();
    }
    let (re_lo, re_hi, im_lo, im_hi) = region;
    let pw = 160u32;
    let ph = (pw as f64 * 9.0 / 16.0).round() as u32;
    let maxiter = sched_maxiter(ctx.maxiter_base, ctx.per_decade, base_width);

    let mut out: Vec<Location> = Vec::new();
    for _ in 0..tries {
        if out.len() >= n {
            break;
        }
        let re = re_lo + rng.unit() * (re_hi - re_lo);
        let im = im_lo + rng.unit() * (im_hi - im_lo);
        let center = Complex::new(re, im);
        let buf = render_buf(center, base_width, pw, ph, ctx.ss, maxiter, ctx.bailout, ctx.trap);
        let cov = coverage_stats(&buf, base_width, ctx.target_width, ctx.theta, COVER_LO, COVER_HI);

        // Pre-filter (pure interior) then the signature filter (on trial).
        if cov.interior_frac > interior_max {
            continue;
        }
        if cov.interior_frac >= interior_max_b || cov.coverage < coverage_min {
            continue;
        }
        // Spatial dedup so the sample spreads instead of clustering on one valley.
        if out.iter().any(|l| (l.center - center).norm() < base_width * 2.0) {
            continue;
        }
        out.push(Location {
            source: Source::B,
            idx: out.len(),
            center,
            base_width,
            note: String::new(),
        });
    }
    out
}

// ===========================================================================
// Source C — minibrot control (navigate primitives)
// ===========================================================================

/// A handful of minibrot neighborhoods via the `navigate` primitives (atom →
/// Newton → size), framed at `|size| · frame_multiple`. The visual control,
/// expected to look *worse* (black body + near-cusp speckle + flat far-field).
fn scan_minibrots(ctx: &Ctx, n: usize, frame_multiple: f64) -> Vec<Location> {
    if n == 0 {
        return Vec::new();
    }
    let scan_w = 600u32;
    let scan_h = (scan_w as f64 * 2.0 / 3.0).round() as u32;
    let width = 3.0;
    let center = Complex::new(-0.5, 0.0);
    let maxiter = 2000u32;
    let prec = hp::prec_bits(scan_w, width) + 32;
    let buf = render_buf(center, width, scan_w, scan_h, ctx.ss, maxiter, ctx.bailout, ctx.trap);

    let cands = atom_candidates(&buf, scan_w, scan_h, width, maxiter);

    // Resolve every candidate to (size, nucleus, period), then keep only genuine
    // **minibrot islands** by `size.mag`. The size-band filter is what separates an
    // island from a period-p *bulb* attached to the cardioid: a bulb's size estimate
    // is O(1) (l stays O(1)), so framing it at `size·multiple` yields a huge flat
    // frame with no minibrot in it — exactly the degenerate the ascending-period
    // pick produced. Islands sit at `size.mag ≲ 0.05`.
    let mut resolved: Vec<(f64, Complex<f64>, u32)> = Vec::new();
    for c in &cands {
        if c.period < 3 || c.period > 20_000 {
            continue;
        }
        let gre = BigFloat::from_f64(center.re + c.dc_re, prec);
        let gim = BigFloat::from_f64(center.im + c.dc_im, prec);
        let Some(nuc) = newton_nucleus(&gre, &gim, c.period, width, prec) else {
            continue;
        };
        let nf = Complex::new(hp::to_f64(&nuc.re), hp::to_f64(&nuc.im));
        let size = size_estimate(nf, c.period);
        if size.overflow || !(size.mag > 0.0) || !size.mag.is_finite() {
            continue;
        }
        // Island band: small enough to exclude bulbs, large enough to stay f64-safe
        // at the DEEP (0.25×) scale (size·multiple·0.25 ≫ the ~1e-13 floor).
        if size.mag > 0.05 || size.mag < 1e-6 {
            continue;
        }
        resolved.push((size.mag, nf, c.period));
    }
    // Largest framable islands first — the clearest, most prominent minibrots for
    // the control (period 3/4 class), then spatially distinct picks.
    resolved.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut out: Vec<Location> = Vec::new();
    for (mag, nf, period) in resolved {
        if out.len() >= n {
            break;
        }
        let bw = mag * frame_multiple;
        if out.iter().any(|l| (l.center - nf).norm() < bw * 4.0) {
            continue;
        }
        out.push(Location {
            source: Source::C,
            idx: out.len(),
            center: nf,
            base_width: bw,
            note: format!("P{period}"),
        });
    }
    out
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Render one f64 Mandelbrot tile. Centers are shallow (well above the f64
/// floor), so the BigFloat center is just the f64 projection.
#[allow(clippy::too_many_arguments)]
fn render_buf(
    center: Complex<f64>,
    width: f64,
    pw: u32,
    ph: u32,
    ss: u32,
    maxiter: u32,
    bailout: f64,
    trap: Trap,
) -> SampleBuffer {
    let prec = hp::prec_bits(pw, width);
    let cre = BigFloat::from_f64(center.re, prec);
    let cim = BigFloat::from_f64(center.im, prec);
    let panel = probe::render_mandel_panel(
        &cre, &cim, center, width, pw, ph, ss, maxiter, bailout, prec, trap, BackendChoice::F64,
    );
    debug_assert_eq!(panel.backend_name, "F64", "buffet stays in the f64 cheap regime");
    panel.buf
}

/// maxiter schedule anchored at [`REF_WIDTH`] (matches the search/cover shape).
fn sched_maxiter(base: f64, per_decade: f64, width: f64) -> u32 {
    let mag = (REF_WIDTH / width).max(1.0);
    (base + per_decade * mag.log10()).round().max(300.0) as u32
}

/// Outward normal of the boundary in this frame: the direction from the interior
/// (non-escaped) centroid to the exterior (escaped) centroid, in plane units,
/// normalized. This is the DE field's outward direction integrated over the frame
/// (`de = 0` inside, `de > 0` outside), robust where a local gradient is noisy.
/// `None` when either set is empty or the two centroids coincide.
fn outward_normal(buf: &SampleBuffer, width: f64) -> Option<Complex<f64>> {
    let ss = buf.ss as usize;
    let sub_w = buf.out_width as usize * ss;
    let sub_h = buf.out_height as usize * ss;
    let fw = width;
    let fh = width * (buf.out_height as f64 / buf.out_width as f64);
    let (sub_w_f, sub_h_f) = (sub_w as f64, sub_h as f64);

    let mut ext = Complex::new(0.0, 0.0);
    let mut int = Complex::new(0.0, 0.0);
    let (mut n_ext, mut n_int) = (0usize, 0usize);
    for (i, s) in buf.samples.iter().enumerate() {
        let scol = (i % sub_w) as f64 + 0.5;
        let srow = (i / sub_w) as f64 + 0.5;
        let dc = Complex::new((scol / sub_w_f - 0.5) * fw, (0.5 - srow / sub_h_f) * fh);
        if s.escaped {
            ext += dc;
            n_ext += 1;
        } else {
            int += dc;
            n_int += 1;
        }
    }
    if n_ext == 0 || n_int == 0 {
        return None;
    }
    let dir = ext.scale(1.0 / n_ext as f64) - int.scale(1.0 / n_int as f64);
    let m = dir.norm();
    if m < 1e-12 * width {
        return None;
    }
    Some(dir.scale(1.0 / m))
}

/// Interquartile range of `de_px` (target-spacing) among escaped subsamples — the
/// de_px **spread** label ("uniform" vs "bimodal"). `NaN` with < 4 escaped.
fn de_px_iqr(buf: &SampleBuffer, width: f64, target_width: u32) -> f64 {
    let inv = target_width.max(1) as f64 / width;
    let mut v: Vec<f64> = buf
        .samples
        .iter()
        .filter(|s| s.escaped)
        .map(|s| s.de * inv)
        .collect();
    if v.len() < 4 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 {
        let idx = (((v.len() - 1) as f64) * p).round() as usize;
        v[idx]
    };
    q(0.75) - q(0.25)
}

/// Render a base frame just to check the pre-filter (pure-interior reject).
fn pre_filter_reject(ctx: &Ctx, center: Complex<f64>, width: f64, interior_max: f64) -> bool {
    let pw = 128u32;
    let ph = (pw as f64 * 9.0 / 16.0).round() as u32;
    let mi = sched_maxiter(ctx.maxiter_base, ctx.per_decade, width);
    let buf = render_buf(center, width, pw, ph, ctx.ss, mi, ctx.bailout, ctx.trap);
    let cov = coverage_stats(&buf, width, ctx.target_width, ctx.theta, COVER_LO, COVER_HI);
    cov.interior_frac > interior_max
}

/// Compact caption number: `NA` for non-finite, scientific for very large/small,
/// else fixed precision. Keeps every label short enough for the 5×7 font.
fn cn(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return "NA".into();
    }
    let a = x.abs();
    if a != 0.0 && (a >= 1.0e4 || a < 1.0e-3) {
        return format!("{x:.0e}");
    }
    format!("{x:.prec$}")
}

// ===========================================================================
// Reporting
// ===========================================================================

fn print_table(records: &[TileRecord]) {
    println!(
        "{:<12} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "id", "coverage", "subpix", "interior", "de_med", "de_iqr", "busy", "esc", "maxit",
    );
    for r in records {
        println!(
            "BUFFET {:<12} {:>9.4} {:>9.4} {:>8.4} {:>8.3} {:>8.3} {:>8.4} {:>8.4} {:>6}",
            r.id, r.coverage, r.subpixel_frac, r.interior_frac, r.de_px_median, r.de_px_iqr,
            r.busy_win, r.esc_frac, r.maxiter,
        );
    }
}

fn build_json(args: &BuffetArgs, records: &[TileRecord]) -> String {
    use probe::{jf, js};
    let mut s = String::from("{\n");
    s.push_str(&format!(
        "  \"params\": {{ \"target_width\": {}, \"theta\": {}, \"window\": {}, \
\"cover_band\": [{}, {}], \"offsets\": [\"ON\",\"MID\",\"FAR\"], \
\"scales\": [\"WIDE\",\"BASE\",\"DEEP\"], \"b_interior_max\": {}, \"b_coverage_min\": {} }},\n",
        args.target_width,
        jf(args.theta),
        args.window,
        jf(COVER_LO),
        jf(COVER_HI),
        jf(args.b_interior_max),
        jf(args.b_coverage_min),
    ));
    s.push_str("  \"tiles\": [\n");
    for (i, r) in records.iter().enumerate() {
        s.push_str(&format!(
            "    {{ \"id\": {}, \"source\": {}, \"loc\": {}, \"offset\": {}, \"scale\": {}, \
\"note\": {}, \"center\": {{ \"re\": {}, \"im\": {} }}, \"width\": {}, \"maxiter\": {}, \
\"coverage\": {}, \"subpixel_frac\": {}, \"interior_frac\": {}, \"esc_frac\": {}, \
\"de_px_median\": {}, \"de_px_iqr\": {}, \"busy_win\": {} }}{}\n",
            js(&r.id),
            js(r.source),
            r.loc_idx,
            js(r.off),
            js(r.scale),
            js(&r.note),
            jf(r.center_re),
            jf(r.center_im),
            jf(r.width),
            r.maxiter,
            jf(r.coverage),
            jf(r.subpixel_frac),
            jf(r.interior_frac),
            jf(r.esc_frac),
            jf(r.de_px_median),
            jf(r.de_px_iqr),
            jf(r.busy_win),
            if i + 1 < records.len() { "," } else { "" },
        ));
    }
    s.push_str("  ]\n}\n");
    s
}
