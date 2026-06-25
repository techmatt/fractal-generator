//! `reject-corridor` subcommand — diagnosis-only audit of the `generate` accept
//! band's **detail-floor** decision (the `spread_min` clause).
//!
//! The band rejects ~97% of draws. The bulk of those rejects are uncontroversial
//! — *flat* (no iteration variety) or *interior-black* (mostly dead interior) —
//! and Matt has already endorsed cutting them. But between the confirmed
//! bad-sparse SPRD ceiling (~23) and the confirmed good-anchor SPRD floor (~86)
//! lies an **un-eyeballed corridor**: frames that pass the interior cap, clear
//! the not-flat floor, yet straddle the detail-floor that `generate` placed at
//! `spread_min = 50`. Nobody has looked at what that floor cuts — it was set on a
//! data gap, not on labels. This subcommand renders the corridor so Matt can.
//!
//! ## What it does (no new metric, no band change)
//!  - **Fresh seed, all-draw logging.** `generate` (run1) logged keepers only, so
//!    its rejects are gone. Here *every* draw is logged (`draws.jsonl`): the full
//!    screen vector, the band verdict, which clause rejected it, and the
//!    per-clause margins — so the corridor can be re-sliced later without
//!    re-rendering.
//!  - **Corridor contact sheet** at keeper resolution under the preview palette:
//!    every draw with `interior < interior_max ∧ esc_median ≥ esc_median_min ∧
//!    spread ∈ [corridor_lo, corridor_hi]`. Sorted by spread, each tile marked
//!    **CUT** (below the live floor — what `generate` rejects) or **KEEP** (above
//!    it). This is exactly "what floor-at-50 is cutting that isn't obvious junk".
//!  - **Bulk representatives.** A few *flat* (`spread < corridor_lo`) and a few
//!    *interior-black* (`interior > interior_max`) tiles — Matt's already endorsed
//!    these, so they're shown only as a handful of reps, not a full sheet.
//!
//! It **reuses** `generate`'s screen + band verbatim (`screen_stats`,
//! `AcceptBand::test`, `EscDist`, `color_params`) so the corridor is defined
//! against the *same* decision boundary `generate` ships. It changes no band
//! default, touches no render path, and computes no corpus descriptor — the
//! "bench a coverage measure against new labels" follow-up is noted in the prompt,
//! not built here.

use std::fmt::Write as _;
use std::path::Path;

use astro_float::BigFloat;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use clap::Args;
use crate::cli::BackendChoice;
use crate::generate::{color_params, screen_stats, AcceptBand, EscDist};
use crate::palette::Palette;
use crate::probe::{self, load_colormap};
use crate::{font, hp, render, sheet};

/// Screen supersample — moments/histogram only need a representative field
/// (matches `generate`).
const SCREEN_SS: u32 = 1;
/// Keeper render resolution (the `generate` keeper regime), so corridor tiles
/// look exactly like keeper previews.
const KEEP_W: u32 = 1280;
const KEEP_H: u32 = 720;
const KEEP_SS: u32 = 2;
const COLORMAPS_PATH: &str = "data/palettes/clean_colormaps.json";

/// Which render bucket a draw falls in (for sheet selection — orthogonal to the
/// band's accept/reject verdict, which is logged separately).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bucket {
    /// `interior > interior_max` — dead interior (endorsed reject; bulk rep).
    InteriorBlack,
    /// `esc_median < esc_median_min` — far-exterior / instant escape (counted).
    InstantEscape,
    /// `spread < corridor_lo` — genuinely flat (endorsed reject; bulk rep).
    Flat,
    /// `interior ok ∧ esc ok ∧ spread ∈ [corridor_lo, corridor_hi]` — the
    /// audited corridor (the main artifact).
    Corridor,
    /// `spread > corridor_hi` — clear good anchor above the corridor (counted).
    Anchor,
}

impl Bucket {
    fn label(self) -> &'static str {
        match self {
            Bucket::InteriorBlack => "interior_black",
            Bucket::InstantEscape => "instant_escape",
            Bucket::Flat => "flat",
            Bucket::Corridor => "corridor",
            Bucket::Anchor => "anchor",
        }
    }
}

/// One logged draw (every draw, kept or not).
struct DrawRow {
    draw_index: usize,
    center: Complex<f64>,
    frame_width: f64,
    scale_u: f64,
    interior_frac: f64,
    esc: EscDist,
    /// Band verdict (against the same band `generate` ships).
    accepted: bool,
    reject_clause: &'static str, // "ok" if accepted
    spread_margin: f64,
    interior_margin: f64,
    esc_median_margin: f64,
    bucket: Bucket,
}

/// `reject-corridor` entry point.
pub fn run_reject_corridor(args: &RejectCorridorArgs) -> Result<(), String> {
    if args.draws == 0 {
        return Err("--draws must be > 0".into());
    }
    let (re_lo, re_hi, im_lo, im_hi) = args.resolved_box()?;
    if args.fw_lo <= 0.0 || args.fw_hi <= args.fw_lo {
        return Err(format!(
            "invalid --fw-lo/--fw-hi: need 0 < fw_lo < fw_hi (got {}, {})",
            args.fw_lo, args.fw_hi
        ));
    }
    if args.corridor_hi <= args.corridor_lo {
        return Err(format!(
            "invalid --corridor-lo/--corridor-hi: need lo < hi (got {}, {})",
            args.corridor_lo, args.corridor_hi
        ));
    }
    let band = args.band();
    let screen_w = args.screen_width.max(1);
    let screen_h = (screen_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let thumb_w = args.thumb_width.max(1);
    let thumb_h = (thumb_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;

    let out_dir = Path::new(&args.out_dir);
    crate::ensure_parent_dir(out_dir.join("x"))?;

    // Preview palette (single read site; default cubehelix). Same loader as
    // `generate`, so the corridor shades identically to keeper previews.
    let cm_text = std::fs::read_to_string(COLORMAPS_PATH)
        .map_err(|e| format!("read {COLORMAPS_PATH}: {e}"))?;
    let stops = load_colormap(&cm_text, &args.palette)?;
    // Selective seam fix: SEQUENTIAL (mirror_needed) maps bake pre-mirrored.
    let mirror = probe::colormap_mirror_needed(&cm_text, &args.palette);
    let palette = Palette::from_srgb8_stops_mirrored(args.palette.clone(), &stops, false, mirror);
    let params = color_params();
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    eprintln!(
        "reject-corridor: draws={} seed={} screen {screen_w}x{screen_h} ss{SCREEN_SS}, \
         keeper-res {KEEP_W}x{KEEP_H} ss{KEEP_SS}, palette {}",
        args.draws, args.seed, args.palette
    );
    eprintln!(
        "band (same as generate ships): spread>={} AND interior<={:.0}% AND esc_median>={}",
        band.spread_min,
        band.interior_max * 100.0,
        band.esc_median_min
    );
    eprintln!(
        "corridor = interior<={:.0}% AND esc_median>={} AND spread in [{}, {}]; live floor at {} \
         splits CUT (<floor) from KEEP (>=floor)",
        band.interior_max * 100.0,
        band.esc_median_min,
        args.corridor_lo,
        args.corridor_hi,
        band.spread_min
    );

    let ln_lo = args.fw_lo.ln();
    let ln_hi = args.fw_hi.ln();
    let mut rng = probe::SplitMix64(args.seed);
    let mut rows: Vec<DrawRow> = Vec::with_capacity(args.draws);
    // Bucket tallies (in Bucket enum order for the summary).
    let mut counts = [0usize; 5];
    let bidx = |b: Bucket| match b {
        Bucket::InteriorBlack => 0,
        Bucket::InstantEscape => 1,
        Bucket::Flat => 2,
        Bucket::Corridor => 3,
        Bucket::Anchor => 4,
    };

    for draw in 0..args.draws {
        // Same three-draw order as generate (re, im, scale) — keeps the regime
        // identical; only the seed differs.
        let re = re_lo + rng.unit() * (re_hi - re_lo);
        let im = im_lo + rng.unit() * (im_hi - im_lo);
        let scale_u = rng.unit();
        let frame_width = (ln_lo + scale_u * (ln_hi - ln_lo)).exp();
        let center = Complex::new(re, im);

        let prec = hp::prec_bits(screen_w, frame_width);
        let cre = BigFloat::from_f64(center.re, prec);
        let cim = BigFloat::from_f64(center.im, prec);
        let panel = probe::render_mandel_panel(
            &cre, &cim, center, frame_width, screen_w, screen_h, SCREEN_SS, args.maxiter,
            args.bailout, prec, trap, BackendChoice::F64,
        );
        let (interior_frac, esc) = screen_stats(&panel.buf.samples, args.maxiter);

        let v = band.test(interior_frac, esc.spread, esc.median);
        let bucket = if interior_frac > band.interior_max {
            Bucket::InteriorBlack
        } else if esc.median < band.esc_median_min {
            Bucket::InstantEscape
        } else if esc.spread < args.corridor_lo {
            Bucket::Flat
        } else if esc.spread <= args.corridor_hi {
            Bucket::Corridor
        } else {
            Bucket::Anchor
        };
        counts[bidx(bucket)] += 1;

        rows.push(DrawRow {
            draw_index: draw,
            center,
            frame_width,
            scale_u,
            interior_frac,
            esc,
            accepted: v.accepted,
            reject_clause: if v.accepted { "ok" } else { v.primary },
            spread_margin: v.spread_margin,
            interior_margin: v.interior_margin,
            esc_median_margin: v.esc_median_margin,
            bucket,
        });
    }

    // --- corridor: render (capped, evenly sampled by spread) ---
    let mut corridor: Vec<&DrawRow> = rows.iter().filter(|r| r.bucket == Bucket::Corridor).collect();
    corridor.sort_by(|a, b| a.esc.spread.partial_cmp(&b.esc.spread).unwrap_or(std::cmp::Ordering::Equal));
    let corridor_total = corridor.len();
    let max_c = args.max_corridor.max(1);
    let rendered: Vec<&&DrawRow> = if corridor_total <= max_c {
        corridor.iter().collect()
    } else {
        // Even stride across the spread-sorted corridor, so the sheet spans the
        // whole [lo, hi] band rather than one end.
        (0..max_c)
            .map(|k| &corridor[k * (corridor_total - 1) / (max_c - 1)])
            .collect()
    };

    let mut corridor_tiles: Vec<RgbImage> = Vec::with_capacity(rendered.len());
    for r in &rendered {
        let cut = r.esc.spread < band.spread_min;
        let mut th = render_thumb(r, &palette, &params, thumb_w, thumb_h, args.maxiter, args.bailout, trap);
        annotate(&mut th, r, "corridor", Some(cut));
        corridor_tiles.push(th);
    }
    let cut_rendered = rendered.iter().filter(|r| r.esc.spread < band.spread_min).count();

    if !corridor_tiles.is_empty() {
        let grid = sheet::compose_grid(&corridor_tiles, Some(args.cols.max(1)));
        let p = out_dir.join("corridor_sheet.png");
        grid.save(&p).map_err(|e| format!("save corridor sheet: {e}"))?;
        eprintln!(
            "corridor sheet: {} ({} tiles of {} corridor draws; {} CUT, {} KEEP)",
            p.display(),
            corridor_tiles.len(),
            corridor_total,
            cut_rendered,
            corridor_tiles.len() - cut_rendered
        );
    } else {
        eprintln!("corridor: 0 draws landed in the corridor — nothing to render (raise --draws or widen the corridor)");
    }

    // --- bulk reps: a few flat + a few interior-black ---
    let reps = args.bulk_reps.max(1);
    let flat_reps = sample_even(&rows, Bucket::Flat, reps, |r| r.esc.spread, true);
    let int_reps = sample_even(&rows, Bucket::InteriorBlack, reps, |r| r.interior_frac, false);
    let mut bulk_tiles: Vec<RgbImage> = Vec::new();
    for r in flat_reps.iter().chain(int_reps.iter()) {
        let mut th = render_thumb(r, &palette, &params, thumb_w, thumb_h, args.maxiter, args.bailout, trap);
        annotate(&mut th, r, r.bucket.label(), None);
        bulk_tiles.push(th);
    }
    if !bulk_tiles.is_empty() {
        let grid = sheet::compose_grid(&bulk_tiles, Some(reps));
        let p = out_dir.join("bulk_sheet.png");
        grid.save(&p).map_err(|e| format!("save bulk sheet: {e}"))?;
        eprintln!(
            "bulk reps: {} ({} flat + {} interior-black; endorsed rejects, shown as reps only)",
            p.display(),
            flat_reps.len(),
            int_reps.len()
        );
    }

    // --- all-draw log + manifest ---
    let jsonl = build_jsonl(&rows);
    let jsonl_path = out_dir.join("draws.jsonl");
    std::fs::write(&jsonl_path, jsonl).map_err(|e| format!("write draws.jsonl: {e}"))?;

    let manifest = build_manifest(
        args, &band, (re_lo, re_hi, im_lo, im_hi), screen_w, screen_h, &counts, corridor_total,
        corridor_tiles.len(), cut_rendered,
    );
    let manifest_path = out_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest).map_err(|e| format!("write manifest.json: {e}"))?;

    // --- stdout summary ---
    println!("=== reject-corridor ===");
    println!("seed={} draws={}", args.seed, args.draws);
    println!(
        "buckets: interior_black={} instant_escape={} flat={} corridor={} anchor={}",
        counts[0], counts[1], counts[2], counts[3], counts[4]
    );
    let accepted = rows.iter().filter(|r| r.accepted).count();
    println!(
        "band verdict: accepted={} ({:.1}%)  rejected={}",
        accepted,
        accepted as f64 / args.draws as f64 * 100.0,
        args.draws - accepted
    );
    println!(
        "corridor [{}, {}] (interior<={:.0}% , esc_median>={}): {} draws; \
         rendered {} (CUT<{}={}, KEEP={})",
        args.corridor_lo,
        args.corridor_hi,
        band.interior_max * 100.0,
        band.esc_median_min,
        corridor_total,
        corridor_tiles.len(),
        band.spread_min,
        cut_rendered,
        corridor_tiles.len() - cut_rendered
    );
    println!("draws log: {}", jsonl_path.display());
    println!("manifest : {}", manifest_path.display());
    println!(
        "READ: the CUT tiles are what floor-at-{} rejects in the corridor — eyeball whether any are good frames (→ then bench a coverage measure against new labels) or all filaments-on-empty.",
        band.spread_min
    );
    Ok(())
}

/// Render one draw at keeper resolution and downsample to a thumbnail (no
/// annotation — the caller annotates).
#[allow(clippy::too_many_arguments)]
fn render_thumb(
    r: &DrawRow,
    palette: &Palette,
    params: &crate::coloring::ColorParams,
    thumb_w: u32,
    thumb_h: u32,
    maxiter: u32,
    bailout: f64,
    trap: Trap,
) -> RgbImage {
    let kprec = hp::prec_bits(KEEP_W, r.frame_width);
    let kre = BigFloat::from_f64(r.center.re, kprec);
    let kim = BigFloat::from_f64(r.center.im, kprec);
    let kpanel = probe::render_mandel_panel(
        &kre, &kim, r.center, r.frame_width, KEEP_W, KEEP_H, KEEP_SS, maxiter, bailout, kprec,
        trap, BackendChoice::F64,
    );
    let krgb = render::shade_and_downsample(
        &kpanel.buf.samples, KEEP_W, KEEP_H, KEEP_SS, palette, params, kpanel.spacing,
    );
    image::imageops::resize(&krgb, thumb_w, thumb_h, FilterType::Triangle)
}

/// Even-stride sample of `reps` rows from a bucket, ordered by `key` (`asc` =
/// ascending). Spans the bucket's value range deterministically.
fn sample_even(
    rows: &[DrawRow],
    bucket: Bucket,
    reps: usize,
    key: impl Fn(&DrawRow) -> f64,
    asc: bool,
) -> Vec<&DrawRow> {
    let mut v: Vec<&DrawRow> = rows.iter().filter(|r| r.bucket == bucket).collect();
    v.sort_by(|a, b| {
        let (ka, kb) = (key(a), key(b));
        let o = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
        if asc { o } else { o.reverse() }
    });
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let take = reps.min(n);
    if take == 1 {
        return vec![v[0]];
    }
    (0..take).map(|k| v[k * (n - 1) / (take - 1)]).collect()
}

/// Annotate a tile: index, fw, interior%, spread, bucket, and (corridor only)
/// the CUT/KEEP verdict against the live floor.
fn annotate(th: &mut RgbImage, r: &DrawRow, tag: &str, cut: Option<bool>) {
    let white = Rgb([240u8, 240, 240]);
    font::draw_text(th, &format!("d{:05} fw{:.4}", r.draw_index, r.frame_width), 2, 2, 1, white, true);
    font::draw_text(
        th,
        &format!("INT{:.0}% SPRD{:.0}", r.interior_frac * 100.0, r.esc.spread.max(0.0)),
        2,
        12,
        1,
        white,
        true,
    );
    font::draw_text(th, tag, 2, 22, 1, Rgb([180, 200, 255]), true);
    if let Some(cut) = cut {
        let (txt, col) = if cut {
            ("CUT", Rgb([255u8, 110, 110]))
        } else {
            ("KEEP", Rgb([120u8, 255, 120]))
        };
        font::draw_text(th, txt, th.width().saturating_sub(34), 2, 1, col, true);
    }
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

fn jarr(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (k, x) in v.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        s.push_str(&jnum(*x));
    }
    s.push(']');
    s
}

/// `draws.jsonl`: one row per draw (the all-draw log — every draw, kept or not).
fn build_jsonl(rows: &[DrawRow]) -> String {
    let mut s = String::new();
    for r in rows {
        let _ = writeln!(
            s,
            "{{ \"draw_index\": {}, \"center_re\": {}, \"center_im\": {}, \"frame_width\": {}, \
             \"scale_u\": {}, \"interior_frac\": {}, \"spread\": {}, \"esc_median\": {}, \
             \"esc_count\": {}, \"esc_mean\": {}, \"esc_std\": {}, \"esc_skew\": {}, \
             \"esc_min\": {}, \"esc_p5\": {}, \"esc_p25\": {}, \"esc_p75\": {}, \"esc_p95\": {}, \
             \"esc_max\": {}, \"esc_hist\": {}, \
             \"accepted\": {}, \"reject_clause\": \"{}\", \"bucket\": \"{}\", \
             \"margin_spread\": {}, \"margin_interior\": {}, \"margin_interior_pct\": {}, \
             \"margin_esc_median\": {} }}",
            r.draw_index,
            jnum(r.center.re),
            jnum(r.center.im),
            jnum(r.frame_width),
            jnum(r.scale_u),
            jnum(r.interior_frac),
            jnum(r.esc.spread),
            jnum(r.esc.median),
            r.esc.count,
            jnum(r.esc.mean),
            jnum(r.esc.std),
            jnum(r.esc.skew),
            jnum(r.esc.min),
            jnum(r.esc.p5),
            jnum(r.esc.p25),
            jnum(r.esc.p75),
            jnum(r.esc.p95),
            jnum(r.esc.max),
            jarr(&r.esc.hist),
            r.accepted,
            r.reject_clause,
            r.bucket.label(),
            jnum(r.spread_margin),
            jnum(r.interior_margin),
            jnum(r.interior_margin * 100.0),
            jnum(r.esc_median_margin),
        );
    }
    s
}

/// `manifest.json`: run config + bucket counts so the audit is reproducible and
/// self-describing.
#[allow(clippy::too_many_arguments)]
fn build_manifest(
    args: &RejectCorridorArgs,
    band: &AcceptBand,
    bx: (f64, f64, f64, f64),
    screen_w: u32,
    screen_h: u32,
    counts: &[usize; 5],
    corridor_total: usize,
    corridor_rendered: usize,
    cut_rendered: usize,
) -> String {
    let (re_lo, re_hi, im_lo, im_hi) = bx;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"subcommand\": \"reject-corridor\",\n");
    s.push_str("  \"purpose\": \"diagnosis-only audit of the generate accept-band detail-floor (spread_min clause); no band change, no new metric\",\n");
    let _ = writeln!(s, "  \"seed\": {},", args.seed);
    let _ = writeln!(s, "  \"draws\": {},", args.draws);
    let _ = writeln!(
        s,
        "  \"box\": {{ \"re_lo\": {re_lo}, \"re_hi\": {re_hi}, \"im_lo\": {im_lo}, \"im_hi\": {im_hi} }},"
    );
    let _ = writeln!(
        s,
        "  \"frame_width_range\": {{ \"lo\": {}, \"hi\": {}, \"sampling\": \"log-uniform\", \"geo_center\": {} }},",
        args.fw_lo,
        args.fw_hi,
        (args.fw_lo * args.fw_hi).sqrt()
    );
    let _ = writeln!(s, "  \"maxiter\": {},", args.maxiter);
    let _ = writeln!(s, "  \"bailout\": {},", args.bailout);
    let _ = writeln!(
        s,
        "  \"screen\": {{ \"w\": {screen_w}, \"h\": {screen_h}, \"ss\": {SCREEN_SS} }},"
    );
    let _ = writeln!(
        s,
        "  \"keeper_render\": {{ \"w\": {KEEP_W}, \"h\": {KEEP_H}, \"ss\": {KEEP_SS} }},"
    );
    let _ = writeln!(s, "  \"palette\": {},", probe::js(&args.palette));
    let _ = writeln!(
        s,
        "  \"accept_band\": {{ \"spread_min\": {}, \"interior_max\": {}, \"esc_median_min\": {}, \"note\": \"same band generate ships; not modified here\" }},",
        band.spread_min, band.interior_max, band.esc_median_min
    );
    let _ = writeln!(
        s,
        "  \"corridor\": {{ \"lo\": {}, \"hi\": {}, \"definition\": \"interior<=interior_max AND esc_median>=esc_median_min AND spread in [lo,hi]\", \"floor\": {}, \"max_render\": {} }},",
        args.corridor_lo, args.corridor_hi, band.spread_min, args.max_corridor
    );
    let _ = writeln!(
        s,
        "  \"buckets\": {{ \"interior_black\": {}, \"instant_escape\": {}, \"flat\": {}, \"corridor\": {}, \"anchor\": {} }},",
        counts[0], counts[1], counts[2], counts[3], counts[4]
    );
    let _ = writeln!(
        s,
        "  \"corridor_render\": {{ \"total\": {corridor_total}, \"rendered\": {corridor_rendered}, \"cut\": {cut_rendered}, \"keep\": {} }}",
        corridor_rendered.saturating_sub(cut_rendered)
    );
    s.push_str("}\n");
    s
}


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// `reject-corridor` subcommand: see `reject_corridor::run_reject_corridor`.
/// Diagnosis-only audit of the `generate` accept-band detail-floor. Shares
/// `generate`'s regime (box / fw-range / screen / band) so the corridor is
/// defined against the same boundary `generate` ships; only the seed differs by
/// default (fresh seed → reject draws `generate` never logged). Renders the
/// corridor at keeper resolution under the same preview palette.
#[derive(Args, Debug)]
pub struct RejectCorridorArgs {
    /// Total draws to screen (all logged, kept or not). Fresh seed below, so this
    /// is a new draw stream from `generate`'s run0/run1.
    #[arg(long, default_value_t = 2000)]
    pub draws: usize,

    /// SplitMix64 seed — distinct from `generate`'s default so the rejects logged
    /// here are a fresh stream (run1 logged keepers only).
    #[arg(long, default_value_t = 20_260_623)]
    pub seed: u64,

    /// Sampling box `re_lo,re_hi,im_lo,im_hi` (matches `generate`).
    #[arg(long = "box", default_value = "-2.0,0.7,-1.2,1.2", allow_hyphen_values = true)]
    pub box_bounds: String,

    /// Log-uniform frame-width range, low edge (matches `generate`).
    #[arg(long, default_value_t = 0.003)]
    pub fw_lo: f64,

    /// Log-uniform frame-width range, high edge (matches `generate`).
    #[arg(long, default_value_t = 0.05)]
    pub fw_hi: f64,

    /// Cheap-screen render width in px (height follows 16:9; ss1). Every draw.
    #[arg(long, default_value_t = 320)]
    pub screen_width: u32,

    /// Maximum iterations for the screen + corridor renders (matches `generate`).
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Escape radius.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Corridor lower spread edge ("not-flat" floor — the confirmed bad-sparse
    /// SPRD ceiling). Draws below this are the *flat* bulk.
    #[arg(long, default_value_t = 24.0)]
    pub corridor_lo: f64,

    /// Corridor upper spread edge (the confirmed good-anchor SPRD floor). Draws
    /// above this are clear anchors, not corridor.
    #[arg(long, default_value_t = 85.0)]
    pub corridor_hi: f64,

    /// Max corridor tiles to render (evenly sampled by spread across the corridor
    /// if more are found). The all-draw log keeps every corridor draw regardless.
    #[arg(long, default_value_t = 48)]
    pub max_corridor: usize,

    /// Representatives to render per bulk bucket (flat, interior-black).
    #[arg(long, default_value_t = 3)]
    pub bulk_reps: usize,

    /// Accept-band override: spread floor (default from `generate::AcceptBand`).
    /// This is the live floor the corridor's CUT/KEEP split is drawn against.
    #[arg(long)]
    pub spread_min: Option<f64>,

    /// Accept-band override: interior (max-iter) fraction cap.
    #[arg(long)]
    pub interior_max: Option<f64>,

    /// Accept-band override: escape-median smooth-iter floor.
    #[arg(long)]
    pub esc_median_min: Option<f64>,

    /// Corridor / bulk thumbnail width in px (height follows 16:9).
    #[arg(long, default_value_t = 256)]
    pub thumb_width: u32,

    /// Corridor contact-sheet grid columns.
    #[arg(long, default_value_t = 8)]
    pub cols: usize,

    /// Preview colormap name (from `data/palettes/clean_colormaps.json`); default
    /// cubehelix — same preview palette as `generate`.
    #[arg(long, default_value = "cubehelix")]
    pub palette: String,

    /// Output directory (`draws.jsonl`, `manifest.json`, `corridor_sheet.png`,
    /// `bulk_sheet.png`). Outside `out/` — the all-draw log is the durable artifact.
    #[arg(long, default_value = "data/generated/reject_corridor")]
    pub out_dir: String,
}

impl RejectCorridorArgs {
    /// Parse `--box` (`re_lo,re_hi,im_lo,im_hi`) into bounds (same shape as
    /// `GenerateArgs::resolved_box`).
    pub fn resolved_box(&self) -> Result<(f64, f64, f64, f64), String> {
        let p: Vec<&str> = self.box_bounds.split(',').collect();
        if p.len() != 4 {
            return Err(format!(
                "invalid --box '{}', expected re_lo,re_hi,im_lo,im_hi",
                self.box_bounds
            ));
        }
        let parse = |s: &str, what: &str| -> Result<f64, String> {
            s.trim()
                .parse()
                .map_err(|_| format!("invalid --box {what} in '{}'", self.box_bounds))
        };
        let re_lo = parse(p[0], "re_lo")?;
        let re_hi = parse(p[1], "re_hi")?;
        let im_lo = parse(p[2], "im_lo")?;
        let im_hi = parse(p[3], "im_hi")?;
        if re_hi <= re_lo || im_hi <= im_lo {
            return Err(format!("--box bounds must be lo < hi in '{}'", self.box_bounds));
        }
        Ok((re_lo, re_hi, im_lo, im_hi))
    }

    /// Effective accept band (each clause flag-overridable; default = the band
    /// `generate` ships). The corridor is sliced against this same boundary.
    pub fn band(&self) -> crate::generate::AcceptBand {
        let d = crate::generate::AcceptBand::default();
        crate::generate::AcceptBand {
            spread_min: self.spread_min.unwrap_or(d.spread_min),
            interior_max: self.interior_max.unwrap_or(d.interior_max),
            esc_median_min: self.esc_median_min.unwrap_or(d.esc_median_min),
        }
    }
}
