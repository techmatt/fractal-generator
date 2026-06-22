//! **Throwaway diagnostic — detail-clause coverage bench (diagnosis-only).**
//!
//! Not a generator, not a subcommand, not on any render path. Compiled solely
//! under `cargo test` (declared `#[cfg(test)]` in `lib.rs`). Reuses the
//! `generate`/`reject_corridor` render surface verbatim (`probe::*`,
//! `generate::screen_stats`, `hp::prec_bits`) so the screen buffer it scores is
//! the *same* one the band decides on.
//!
//! ## The question (see `prompts/detail-clause-coverage-bench.md`)
//!
//! The accept band's detail floor is `spread = p95 − p5` of escaped smooth-iter —
//! a **value-range** measure, not a **spatial-coverage** one. The worry: a
//! coherent dendrite whose detail sits in a narrow value band over a dark field
//! reads as low-spread and gets CUT even though it is richly covered. This bench
//! scores candidate **coverage** measures against Matt's hand-labels and reports
//! which (if any) separates CUT-but-good from genuinely-sparse-bad with a wider,
//! cleaner margin than `spread`.
//!
//! Measures, all oriented *higher = more detail* (per the prompt):
//!  1. **spread** — `p95 − p5` of escaped smooth-iter (the current floor; baseline).
//!  2. **edge_energy** — mean Sobel gradient magnitude over the smooth-iter field,
//!     per pixel (the primary hypothesis: lights up on every dendrite edge
//!     regardless of value range). Adds one Sobel pass over the screen buffer.
//!  3. **edge_cov** — fraction of pixels whose Sobel magnitude ≥ 1 smooth-iter/px
//!     (a direct spatial "detail-pixel" coverage; the scale-free variant).
//!  4. **hist_outside_dom** — `1 − max(esc_hist)`: fraction of escaped pixels
//!     outside the dominant histogram bin (free — already logged).
//!  5. **hist_populated** — count of esc_hist bins holding ≥1% of escaped pixels.
//!  6. **hist_entropy** — Shannon entropy of esc_hist, normalized by `ln(NBINS)`.
//!
//! Labels are pulled from the committed logs (re-render from logged params; key on
//! `draw_index` / `keeper_index`):
//!  - **good (CUT-but-should-keep):** reject-corridor draws D02095, D01295,
//!    D01875, D04721.
//!  - **bad (genuinely too sparse):** run0 keepers 036, 038, 040, 042, 048, 049.
//!  - **good anchors (dense, already KEEP):** the top-3-spread corridor draws
//!    (printed by `draw_index` at run time).
//!
//! Diagnosis only: builds no gate, changes no band default, writes no artifact —
//! the table goes to stdout. Run:
//! ```text
//! cargo test --release --lib detail_clause_bench -- --ignored --nocapture
//! ```

use astro_float::BigFloat;
use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::BackendChoice;
use crate::generate::screen_stats;
use crate::{hp, probe};

// --- fixed regime (must match the logs we read) ------------------------------
const MAXITER: u32 = 1000;
const BAILOUT: f64 = 1e6;
const SCREEN_W: u32 = 320;
const SCREEN_H: u32 = 180; // 16:9, ss1
const ESC_HIST_BINS: usize = 16;
/// Sobel magnitude (smooth-iter per pixel) at/above which a pixel counts as a
/// "detail pixel" for `edge_cov`.
const EDGE_PX_THRESH: f64 = 1.0;

const CORRIDOR_LOG: &str = "data/generated/reject_corridor/draws.jsonl";
const RUN0_LOG: &str = "data/generated/run0/locations.jsonl";

/// Hand-label class.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    /// CUT by the live floor, but Matt says keep (the frames the floor mis-cuts).
    GoodCut,
    /// Genuinely too sparse — faint threads on empty (correctly unwanted).
    BadSparse,
    /// Dense, already KEEP — positive anchor (sanity, must score high).
    GoodAnchor,
}

impl Class {
    fn tag(self) -> &'static str {
        match self {
            Class::GoodCut => "good-CUT",
            Class::BadSparse => "bad-sparse",
            Class::GoodAnchor => "good-anchor",
        }
    }
}

/// A labeled frame: its params (to re-render) + class + the spread the log saw.
struct Labeled {
    name: String,
    class: Class,
    center: Complex<f64>,
    fw: f64,
    logged_spread: f64,
}

/// All six measures computed on one re-rendered screen buffer.
struct Measures {
    spread: f64,
    edge_energy: f64,
    edge_cov: f64,
    hist_outside_dom: f64,
    hist_populated: f64,
    hist_entropy: f64,
}

#[test]
#[ignore = "throwaway diagnostic; run explicitly with --ignored --nocapture"]
fn detail_clause_bench() {
    run().expect("detail-clause coverage bench");
}

fn run() -> Result<(), String> {
    let corridor = std::fs::read_to_string(CORRIDOR_LOG)
        .map_err(|e| format!("read {CORRIDOR_LOG}: {e}"))?;
    let run0 = std::fs::read_to_string(RUN0_LOG).map_err(|e| format!("read {RUN0_LOG}: {e}"))?;

    let mut frames: Vec<Labeled> = Vec::new();

    // --- good (CUT-but-should-keep): corridor draws by draw_index ---
    for di in [2095usize, 1295, 1875, 4721] {
        let line = find_line(&corridor, "draw_index", di as f64)
            .ok_or_else(|| format!("draw_index {di} not in {CORRIDOR_LOG}"))?;
        frames.push(labeled_from(&format!("D{di:05}"), Class::GoodCut, line)?);
    }

    // --- bad (genuinely too sparse): run0 keepers by keeper_index ---
    for ki in [36usize, 38, 40, 42, 48, 49] {
        let line = find_line(&run0, "keeper_index", ki as f64)
            .ok_or_else(|| format!("keeper_index {ki} not in {RUN0_LOG}"))?;
        frames.push(labeled_from(&format!("K{ki:03}"), Class::BadSparse, line)?);
    }

    // --- good anchors: top-3-spread corridor draws (dense KEEP), stated below ---
    let mut corridor_rows: Vec<(usize, f64, &str)> = corridor
        .lines()
        .filter(|l| field_str(l, "bucket").as_deref() == Some("corridor"))
        .filter_map(|l| Some((fnum(l, "draw_index")? as usize, fnum(l, "spread")?, l)))
        .collect();
    corridor_rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let anchor_ids: Vec<usize> = corridor_rows.iter().take(3).map(|r| r.0).collect();
    eprintln!(
        "anchors (top-3-spread corridor KEEP draws): {}",
        anchor_ids
            .iter()
            .zip(corridor_rows.iter())
            .map(|(id, r)| format!("D{id:05}(spread {:.1})", r.1))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (di, sp, line) in corridor_rows.iter().take(3) {
        frames.push(labeled_from(&format!("D{di:05}"), Class::GoodAnchor, line).map(
            |mut f| {
                f.logged_spread = *sp;
                f
            },
        )?);
    }

    // --- score each labeled frame on the same screen buffer ---
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };
    let mut scored: Vec<(usize, Measures)> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        let m = measure(f, trap);
        scored.push((i, m));
    }

    // --- table: frame x measure x value x label ---
    eprintln!("\n=== detail-clause coverage bench ===");
    eprintln!(
        "screen {SCREEN_W}x{SCREEN_H} ss1, maxiter {MAXITER}, bailout {BAILOUT} \
         (matches the logs); all measures oriented higher = more detail\n"
    );
    eprintln!(
        "{:<8} {:<12} {:>9} {:>11} {:>9} {:>10} {:>6} {:>8}",
        "frame", "label", "spread", "edge_enrg", "edge_cov", "h_outdom", "h_pop", "h_entr"
    );
    eprintln!("{}", "-".repeat(80));
    for (i, m) in &scored {
        let f = &frames[*i];
        eprintln!(
            "{:<8} {:<12} {:>9.2} {:>11.4} {:>9.4} {:>10.4} {:>6.0} {:>8.4}",
            f.name, f.class.tag(), m.spread, m.edge_energy, m.edge_cov, m.hist_outside_dom,
            m.hist_populated, m.hist_entropy
        );
    }

    // Cross-check: re-rendered spread vs logged spread (should match — same f64
    // center, same panel).
    eprintln!("\nspread cross-check (re-rendered vs logged):");
    for (i, m) in &scored {
        let f = &frames[*i];
        let d = (m.spread - f.logged_spread).abs();
        eprintln!(
            "  {:<8} re-render {:>7.3}  logged {:>7.3}  |Δ| {:.2e}",
            f.name, m.spread, f.logged_spread, d
        );
    }

    // --- per-measure separation summary ---
    eprintln!("\n=== separation per measure (higher = better) ===");
    let getters: [(&str, fn(&Measures) -> f64); 6] = [
        ("spread", |m| m.spread),
        ("edge_energy", |m| m.edge_energy),
        ("edge_cov", |m| m.edge_cov),
        ("hist_outside_dom", |m| m.hist_outside_dom),
        ("hist_populated", |m| m.hist_populated),
        ("hist_entropy", |m| m.hist_entropy),
    ];

    let mut best: Option<(&str, f64)> = None; // (name, normalized cut-margin), clean only
    for (name, get) in getters {
        // partition values by class
        let val = |c: Class| -> Vec<f64> {
            scored
                .iter()
                .filter(|(i, _)| frames[*i].class == c)
                .map(|(_, m)| get(m))
                .collect()
        };
        let cut = val(Class::GoodCut);
        let bad = val(Class::BadSparse);
        let anchor = val(Class::GoodAnchor);
        let goods: Vec<f64> = cut.iter().chain(anchor.iter()).copied().collect();

        let max_bad = bad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_bad = bad.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_cut = cut.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_good = goods.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_anchor = anchor.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // value range over all labeled frames (for a unit-free margin).
        let all: Vec<f64> = scored.iter().map(|(_, m)| get(m)).collect();
        let span = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - all.iter().cloned().fold(f64::INFINITY, f64::min);
        let span = if span > 0.0 { span } else { 1.0 };

        // The hard test: do the four CUT-but-good sit above every sparse-bad?
        let gap_cut = min_cut - max_bad; // >0 ⇒ clean for the critical four
        let clean_cut = gap_cut > 0.0;
        // The full test: all goods (incl. anchors) above all bads?
        let gap_all = min_good - max_bad;
        let clean_all = gap_all > 0.0;
        let thresh = if clean_cut {
            format!("{:.4}", 0.5 * (min_cut + max_bad))
        } else {
            "none".into()
        };

        eprintln!(
            "\n[{name}]\n  bad-sparse [{:.4}, {:.4}]   cut-good [{:.4}, .. ]   anchors up to {:.4}",
            min_bad, max_bad, min_cut, max_anchor
        );
        eprintln!(
            "  cut-vs-bad gap = {:+.4} (norm {:+.3})  -> {}",
            gap_cut,
            gap_cut / span,
            if clean_cut { "CLEAN (4 cut frames above all bad)" } else { "OVERLAP" }
        );
        eprintln!(
            "  all-good-vs-bad gap = {:+.4} (norm {:+.3})  -> {}   best threshold: {}",
            gap_all,
            gap_all / span,
            if clean_all { "CLEAN" } else { "OVERLAP" },
            thresh
        );

        if clean_cut {
            let norm = gap_cut / span;
            if best.map(|(_, b)| norm > b).unwrap_or(true) {
                best = Some((name, norm));
            }
        }
    }

    eprintln!("\n=== verdict ===");
    match best {
        Some((name, norm)) => eprintln!(
            "Widest CLEAN cut-vs-bad separation: [{name}] (normalized margin {norm:+.3}).\n\
             (diagnosis only — promotion of a coverage measure to the AcceptBand detail clause,\n\
             replacing spread_min at its single read site, is a later build, not this prompt.)"
        ),
        None => eprintln!("No measure cleanly separated the four CUT-good from the six sparse-bad."),
    }
    Ok(())
}

/// Re-render the screen buffer for a labeled frame and compute all six measures.
fn measure(f: &Labeled, trap: Trap) -> Measures {
    let prec = hp::prec_bits(SCREEN_W, f.fw);
    let cre = BigFloat::from_f64(f.center.re, prec);
    let cim = BigFloat::from_f64(f.center.im, prec);
    let panel = probe::render_mandel_panel(
        &cre, &cim, f.center, f.fw, SCREEN_W, SCREEN_H, 1, MAXITER, BAILOUT, prec, trap,
        BackendChoice::F64,
    );
    let samples = &panel.buf.samples;
    let (_interior, esc) = screen_stats(samples, MAXITER);

    // Escape-iteration value field for the Sobel pass. Interior pixels (rare in
    // these frames, ~0.1%) are filled with the frame's max escaped value so the
    // set boundary reads as a plateau edge, not a spike to/from zero.
    let max_esc = esc.max;
    let fill = if max_esc.is_finite() { max_esc } else { 0.0 };
    let w = SCREEN_W as usize;
    let h = SCREEN_H as usize;
    let field: Vec<f64> = samples
        .iter()
        .map(|s| if s.escaped { s.smooth_iter } else { fill })
        .collect();

    let (edge_energy, edge_cov) = sobel_stats(&field, w, h);

    // Histogram coverage from the (re-derived) esc_hist.
    let max_bin = esc.hist.iter().cloned().fold(0.0, f64::max);
    let hist_outside_dom = 1.0 - max_bin;
    let hist_populated = esc.hist.iter().filter(|&&p| p >= 0.01).count() as f64;
    let hist_entropy = {
        let ent: f64 = esc
            .hist
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        ent / (ESC_HIST_BINS as f64).ln()
    };

    Measures {
        spread: esc.spread,
        edge_energy,
        edge_cov,
        hist_outside_dom,
        hist_populated,
        hist_entropy,
    }
}

/// Sobel pass over a row-major scalar field. Returns (mean gradient magnitude
/// over interior pixels, fraction of interior pixels with magnitude ≥ thresh).
fn sobel_stats(f: &[f64], w: usize, h: usize) -> (f64, f64) {
    if w < 3 || h < 3 {
        return (0.0, 0.0);
    }
    let at = |x: usize, y: usize| f[y * w + x];
    let mut sum = 0.0;
    let mut count = 0usize;
    let mut detail = 0usize;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            let mag = (gx * gx + gy * gy).sqrt();
            sum += mag;
            count += 1;
            if mag >= EDGE_PX_THRESH {
                detail += 1;
            }
        }
    }
    (sum / count as f64, detail as f64 / count as f64)
}

/// Build a `Labeled` from a log line (parses the params we need to re-render).
fn labeled_from(name: &str, class: Class, line: &str) -> Result<Labeled, String> {
    let g = |k: &str| fnum(line, k).ok_or_else(|| format!("{name}: missing field {k}"));
    Ok(Labeled {
        name: name.to_string(),
        class,
        center: Complex::new(g("center_re")?, g("center_im")?),
        fw: g("frame_width")?,
        logged_spread: g("spread")?,
    })
}

/// First line in `text` whose integer field `key` equals `val`.
fn find_line<'a>(text: &'a str, key: &str, val: f64) -> Option<&'a str> {
    text.lines().find(|l| fnum(l, key) == Some(val))
}

/// Parse a flat-JSON numeric field `"key": <number>` from one line. Leading-quote
/// match avoids `"esc_median"` colliding with `"margin_esc_median"` etc.
fn fnum(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = line[i..].trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].trim().parse::<f64>().ok()
}

/// Parse a flat-JSON string field `"key": "<value>"` from one line.
fn field_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let i = line.find(&pat)? + pat.len();
    let rest = line[i..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}
