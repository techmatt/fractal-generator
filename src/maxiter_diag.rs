//! `maxiter-diag` subcommand — **diagnostic** iteration-cap escalation harness.
//!
//! Two forward-default changes ride on its output: (1) raise the iteration/orbit
//! cap (`maxiter`) so spiral cores resolve instead of pinning gray, with the value
//! **chosen by eye** from an escalation sheet; (2) recalibrate the `present` black
//! gate on renders made at the raised cap. This builds the evidence for both —
//! it picks nothing and changes no render behaviour.
//!
//! The coupling that orders the two changes: **raising `maxiter` lowers the
//! no-escape fraction** (pixels pinned at the cap and bucketed as interior escape
//! at higher iterations). So the cap is escalated first; the gate distribution is
//! then measured *at* the chosen cap.
//!
//! What it does:
//!  - **Auto-selects** the worst-offender crops from a `present` manifest
//!    (`data/label_crops/loose0_v3/manifest.json`) — the distinct (seed ×
//!    composition) crops with the highest recorded `black_fraction` (the grayest
//!    spiral cores) — and appends the standard test location (the minibrot eye)
//!    for continuity.
//!  - Renders each crop at the **locked wallpaper quality** (grid ss4 + Lanczos-3,
//!    1280×720 by default — exactly the `render-one` path) across an **escalating
//!    cap series** (`2000 → 8000 → 32000 → 128000`), recording per (crop × cap)
//!    the **residual pinned-at-cap fraction** ([`render::black_fraction`] on the
//!    supersample buffer — the same statistic the gate sees) and the render
//!    wall-time.
//!  - Emits a **visual escalation sheet** (`escalation.html`) under a stable path
//!    (`data/calibration/maxiter_diag/`, never `out/`): one row per crop, the cap
//!    series side by side, residual-pinned + render-time under each.
//!  - Reports **residual-pinned vs `frame_width`** so a fixed vs depth-adaptive cap
//!    can be judged, **occupancy drift** (old vs new cap, reusing [`energy::occupancy`]),
//!    a **cost multiplier** (top cap vs 2000 on the non-offender test crop), and —
//!    at the auto-detected knee cap — the **no-escape-fraction distribution** over a
//!    sample of manifest crops re-rendered at the cheap gate resolution, grounding
//!    the new 0.30 black gate.
//!
//! Reuses `energy.rs` (occupancy) and the `render-one` f64 quality path verbatim;
//! no new metric. Shallow f64 by construction (the loose0 fw range + the minibrot
//! eye are all in the cheap regime; asserted per crop).

use std::path::PathBuf;
use std::time::Instant;

use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use clap::Args;
use crate::energy::{occupancy, OCC_FLOOR, OCC_GX, OCC_GY};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::probe::PERTURB_SPACING;
use crate::render::{self, black_fraction, DownsampleFilter, Frame, SubsamplePattern};
use crate::{coloring, ensure_parent_dir, hp};

const BAILOUT: f64 = 1e6;

/// One crop to escalate (a worst offender or the test location).
struct Crop {
    label: String,
    cx: f64,
    cy: f64,
    fw: f64,
    /// `black_fraction` recorded in the source manifest (cheap-screen, old cap).
    src_black: f64,
    is_test: bool,
}

/// One (crop × cap) render result.
struct Cell {
    cap: u32,
    /// Residual pinned-at-cap fraction on the supersample buffer (gate statistic).
    residual: f32,
    /// Detail occupancy of the downsampled image (reuses `energy::occupancy`).
    occ: f64,
    secs: f64,
    png: String, // relative filename for the HTML sheet
}

pub fn run_maxiter_diag(args: &MaxiterDiagArgs) -> Result<(), String> {
    let caps = parse_caps(&args.caps)?;
    if caps.is_empty() {
        return Err("--caps is empty".into());
    }
    let out_dir = PathBuf::from(&args.out_dir);
    ensure_parent_dir(out_dir.join("x"))?;

    // --- assemble the crop set: worst offenders + the test location ---
    let mut crops = select_offenders(&args.manifest, args.offenders)?;
    crops.push(Crop {
        label: "test_minibrot_eye".into(),
        cx: parse_dec(&args.test_cx, args.width, args.test_fw)?,
        cy: parse_dec(&args.test_cy, args.width, args.test_fw)?,
        fw: args.test_fw,
        src_black: f64::NAN,
        is_test: true,
    });

    eprintln!(
        "maxiter-diag: {} crops × {} caps {:?} @ {}×{} grid ss{} lanczos3",
        crops.len(),
        caps.len(),
        caps,
        args.width,
        height_16x9(args.width),
        args.supersample
    );

    // --- cost estimate from a single cheap probe (test crop @ smallest cap) ---
    let height = height_16x9(args.width);
    let palette = load_palette(&args.colormaps, &args.palette)?;
    let probe = render_cell(&crops[crops.len() - 1], caps[0], args, height, &palette, &out_dir, true)?;
    // The full job's interior pixels dominate at the top cap; estimate from the
    // probe scaled by the cap ratio over the offenders (interior-heavy) + the
    // already-paid probe. Rough, printed so the background run is not a black box.
    let top = *caps.iter().max().unwrap();
    let est_offender_top = probe.secs * (top as f64 / caps[0] as f64) * 0.5; // interior-frac discount
    let rough_total: f64 = crops.len() as f64
        * caps.iter().map(|&c| probe.secs * (c as f64 / caps[0] as f64) * 0.4).sum::<f64>()
        + est_offender_top;
    eprintln!(
        "  [probe: test crop cap {} in {:.1}s → rough whole-escalation estimate ~{:.0}s ({:.1} min)]",
        caps[0],
        probe.secs,
        rough_total,
        rough_total / 60.0
    );

    // --- the escalation: every (crop × cap) ---
    let mut grid: Vec<(usize, Vec<Cell>)> = Vec::new(); // (crop_idx, cells)
    for (ci, crop) in crops.iter().enumerate() {
        let mut cells = Vec::new();
        for &cap in &caps {
            // Reuse the already-rendered probe cell (test crop, smallest cap).
            let cell = if crop.is_test && cap == caps[0] {
                Cell { cap, residual: probe.residual, occ: probe.occ, secs: probe.secs, png: probe.png.clone() }
            } else {
                render_cell(crop, cap, args, height, &palette, &out_dir, false)?
            };
            eprintln!(
                "  {:24} cap {:>6}: residual_pinned {:.4}  occ {:.3}  {:.1}s",
                crop.label, cap, cell.residual, cell.occ, cell.secs
            );
            cells.push(cell);
        }
        grid.push((ci, cells));
    }

    // --- knee detection: smallest cap past which max-over-crops residual stops
    //     visibly changing (|Δ| < knee_eps vs previous cap) ---
    let knee = detect_knee(&caps, &grid, args.knee_eps);

    // --- gate calibration sample @ knee cap (cheap gate resolution) ---
    let gate = run_gate_sample(&args.manifest, knee, args, &palette)?;

    // --- reports ---
    print_reports(&crops, &caps, &grid, knee, &gate, &probe);

    // --- visual escalation sheet + JSON ---
    let html = build_html(&crops, &grid, knee, &gate);
    let html_path = out_dir.join("escalation.html");
    std::fs::write(&html_path, html).map_err(|e| format!("write {}: {e}", html_path.display()))?;
    let json = build_json(&crops, &caps, &grid, knee, &gate);
    let json_path = out_dir.join("maxiter_diag.json");
    std::fs::write(&json_path, json).map_err(|e| format!("write {}: {e}", json_path.display()))?;

    eprintln!("wrote {}", html_path.display());
    eprintln!("wrote {}", json_path.display());
    println!("escalation sheet: {}", html_path.display());
    println!("PROVISIONAL knee cap = {knee} (confirm by eye from the sheet)");
    Ok(())
}

/// Render one (crop × cap) cell on the locked render-one path; returns the gate
/// statistic, occupancy, wall-time, and the saved PNG filename.
fn render_cell(
    crop: &Crop,
    cap: u32,
    args: &MaxiterDiagArgs,
    height: u32,
    palette: &Palette,
    out_dir: &PathBuf,
    _is_probe: bool,
) -> Result<Cell, String> {
    let ss = args.supersample.max(1);
    let frame = Frame {
        center: Complex::new(crop.cx, crop.cy),
        frame_width: crop.fw,
        out_width: args.width,
        out_height: height,
    };
    let pixel_spacing = frame.pixel_size();
    if pixel_spacing <= PERTURB_SPACING {
        return Err(format!(
            "crop {} pixel spacing {pixel_spacing:.3e} is in f64's quantization regime — \
             maxiter-diag is the shallow f64 path",
            crop.label
        ));
    }

    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let backend = F64Backend::new(cap, BAILOUT, trap);

    let t = Instant::now();
    let buf = render::iterate_samples_f64_pattern(&backend, &frame, ss, channels, SubsamplePattern::Grid, 0);
    let residual = black_fraction(&buf.samples); // non-escaped = pinned at this cap
    let img = render::shade_and_downsample_filtered(
        &buf.samples, args.width, height, ss, palette, &params, pixel_spacing, DownsampleFilter::Lanczos3,
    );
    let secs = t.elapsed().as_secs_f64();
    let occ = occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR);

    let fname = format!("{}_cap{cap}.png", crop.label);
    let path = out_dir.join(&fname);
    img.save(&path).map_err(|e| format!("save {}: {e}", path.display()))?;
    Ok(Cell { cap, residual, occ, secs, png: fname })
}

/// Re-render a sample of manifest crops at `cap` at the **cheap gate resolution**
/// (320×180 ss1 — exactly what `present`'s black gate evaluates) and return the
/// resulting no-escape fractions, so the 0.30 threshold is grounded in what the
/// gate now sees post-cap-raise.
struct GateSample {
    cap: u32,
    /// no-escape fraction per sampled crop, at the new cap.
    new: Vec<f64>,
    /// the source-manifest black_fraction for the same crops (old cap).
    old: Vec<f64>,
}

fn run_gate_sample(
    manifest: &str,
    cap: u32,
    args: &MaxiterDiagArgs,
    palette: &Palette,
) -> Result<GateSample, String> {
    if args.gate_sample == 0 {
        return Ok(GateSample { cap, new: vec![], old: vec![] });
    }
    let all = read_crops(manifest)?;
    if all.is_empty() {
        return Ok(GateSample { cap, new: vec![], old: vec![] });
    }
    // Even sample across the full black_fraction range (sorted), so the
    // distribution is not just survivors near the old 0.40 ceiling.
    let mut sorted = all;
    sorted.sort_by(|a, b| a.black.partial_cmp(&b.black).unwrap());
    let n = args.gate_sample.min(sorted.len());
    let step = sorted.len() as f64 / n as f64;
    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let backend = F64Backend::new(cap, BAILOUT, trap);
    let _ = palette; // gate stat is pre-shade; palette unused here

    eprintln!("gate calibration: {n} crops re-rendered @ cap {cap}, cheap gate res 320×180 ss1");
    let (cw, ch) = (320u32, 180u32);
    let mut new = Vec::with_capacity(n);
    let mut old = Vec::with_capacity(n);
    for i in 0..n {
        let c = &sorted[(i as f64 * step) as usize];
        let frame = Frame { center: Complex::new(c.cx, c.cy), frame_width: c.fw, out_width: cw, out_height: ch };
        let buf = render::iterate_samples_f64(&backend, &frame, 1, channels);
        new.push(black_fraction(&buf.samples) as f64);
        old.push(c.black);
    }
    Ok(GateSample { cap, new, old })
}

// ----------------------------------------------------------------------------
// crop selection / manifest parsing
// ----------------------------------------------------------------------------

/// A parsed manifest crop center (deduped by seed × composition upstream).
struct ManCrop {
    label: String,
    cx: f64,
    cy: f64,
    fw: f64,
    black: f64,
}

fn select_offenders(manifest: &str, n: usize) -> Result<Vec<Crop>, String> {
    let mut crops = read_crops(manifest)?;
    crops.sort_by(|a, b| b.black.partial_cmp(&a.black).unwrap());
    Ok(crops
        .into_iter()
        .take(n)
        .map(|c| Crop { label: c.label, cx: c.cx, cy: c.cy, fw: c.fw, src_black: c.black, is_test: false })
        .collect())
}

/// Parse the manifest crops, deduped by (seed_index, composition) — the same
/// (center, fw, black_fraction) repeats once per palette in the manifest.
fn read_crops(manifest: &str) -> Result<Vec<ManCrop>, String> {
    let text = std::fs::read_to_string(manifest).map_err(|e| format!("read {manifest}: {e}"))?;
    let mut seen: std::collections::HashSet<(i64, String)> = std::collections::HashSet::new();
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.contains("\"seed_index\"") {
            continue;
        }
        let seed = field(line, "seed_index").ok_or("missing seed_index")? as i64;
        let comp = field_str(line, "composition").unwrap_or_else(|| "center".into());
        if !seen.insert((seed, comp.clone())) {
            continue;
        }
        out.push(ManCrop {
            label: format!("s{seed:03}_{comp}"),
            cx: field(line, "cx").ok_or("missing cx")?,
            cy: field(line, "cy").ok_or("missing cy")?,
            fw: field(line, "fw").ok_or("missing fw")?,
            black: field(line, "black_fraction").ok_or("missing black_fraction")?,
        });
    }
    Ok(out)
}

/// Extract a numeric JSON field value by key from a single-line object.
fn field(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = line.find(&pat)? + pat.len();
    let rest = &line[i..];
    let colon = rest.find(':')? + 1;
    let tail = rest[colon..].trim_start();
    let end = tail
        .find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'))
        .unwrap_or(tail.len());
    tail[..end].parse().ok()
}

// String JSON field reader shared via `crate::jsonl` (the canonical copy).
use crate::jsonl::field_str;

// ----------------------------------------------------------------------------
// knee, reports, sheet
// ----------------------------------------------------------------------------

/// Smallest cap past which the **max-over-crops** residual-pinned stops visibly
/// changing (the per-step max |Δ| drops below `eps`). Falls back to the top cap.
fn detect_knee(caps: &[u32], grid: &[(usize, Vec<Cell>)], eps: f32) -> u32 {
    for k in 1..caps.len() {
        let max_delta = grid
            .iter()
            .map(|(_, cells)| (cells[k - 1].residual - cells[k].residual).abs())
            .fold(0.0f32, f32::max);
        if max_delta < eps {
            return caps[k - 1]; // the cap *before* the change went sub-eps
        }
    }
    *caps.last().unwrap()
}

fn print_reports(
    crops: &[Crop],
    caps: &[u32],
    grid: &[(usize, Vec<Cell>)],
    knee: u32,
    gate: &GateSample,
    probe: &Cell,
) {
    println!("\n=== maxiter-diag ===");
    println!("\n# residual pinned-at-cap fraction  (and render wall-time, s)");
    print!("{:24}  fw        ", "crop");
    for &c in caps {
        print!("| cap{:<6}     ", c);
    }
    println!();
    for (ci, cells) in grid {
        let crop = &crops[*ci];
        print!("{:24}  {:.2e} ", crop.label, crop.fw);
        for cell in cells {
            print!("| {:.4} {:>5.0}s ", cell.residual, cell.secs);
        }
        println!();
    }

    println!("\n# depth question — residual pinned vs frame_width (at top cap {})", caps.last().unwrap());
    let mut rows: Vec<(f64, f32, &str)> = grid
        .iter()
        .map(|(ci, cells)| (crops[*ci].fw, cells.last().unwrap().residual, crops[*ci].label.as_str()))
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (fw, res, label) in &rows {
        println!("  fw {:.3e}  residual {:.4}  {}", fw, res, label);
    }
    println!(
        "  → if residual-at-top-cap is flat across fw, one FIXED cap covers loose0's range;\n    \
         if it rises as fw shrinks, the cap should scale with zoom depth (see proposal below)."
    );

    println!("\n# occupancy drift — old cap {} vs new cap {} (energy::occupancy, floor {})", caps[0], knee, OCC_FLOOR);
    let knee_idx = caps.iter().position(|&c| c == knee).unwrap_or(caps.len() - 1);
    for (ci, cells) in grid {
        if crops[*ci].is_test {
            continue;
        }
        println!(
            "  {:24}  occ {:.3} → {:.3}  (Δ {:+.3})",
            crops[*ci].label, cells[0].occ, cells[knee_idx].occ, cells[knee_idx].occ - cells[0].occ
        );
    }
    println!("  → 0.23 occupancy-gate floor was calibrated at cap 2000; surfaced only, not acted on.");

    // cost multiplier: bracket the batch-time hit. The test (filament, ~0 interior)
    // is the lower bound — escapers exit fast so the cap barely costs more; the
    // worst offender (interior-heavy) is the upper bound — interior pixels run the
    // full cap. A typical loose0 crop sits between.
    let top_cap = *caps.last().unwrap();
    println!("\n# cost — render-time multiplier of cap {top_cap} vs cap {} (batch-time hit)", caps[0]);
    if let Some((_, cells)) = grid.iter().find(|(ci, _)| crops[*ci].is_test) {
        let base = cells[0].secs.max(1e-6);
        let topc = cells.last().unwrap();
        println!(
            "  test/filament (lower bound): {:.2}s → {:.2}s  = {:.1}×",
            base, topc.secs, topc.secs / base
        );
    }
    let offender_mult = grid
        .iter()
        .filter(|(ci, _)| !crops[*ci].is_test)
        .map(|(_, cells)| cells.last().unwrap().secs / cells[0].secs.max(1e-6))
        .fold(0.0f64, f64::max);
    println!("  worst offender (upper bound, interior-heavy): {:.1}×", offender_mult);
    let _ = probe;

    if !gate.new.is_empty() {
        println!("\n# black-gate calibration — no-escape fraction @ cap {} ({} crops, cheap gate res)", gate.cap, gate.new.len());
        let pct = |v: &[f64], p: f64| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[((p * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)]
        };
        println!(
            "  new cap : p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}",
            pct(&gate.new, 0.10), pct(&gate.new, 0.50), pct(&gate.new, 0.90),
            gate.new.iter().cloned().fold(0.0, f64::max)
        );
        println!(
            "  old cap : p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}  (manifest black_fraction)",
            pct(&gate.old, 0.10), pct(&gate.old, 0.50), pct(&gate.old, 0.90),
            gate.old.iter().cloned().fold(0.0, f64::max)
        );
        let over30 = gate.new.iter().filter(|&&v| v > 0.30).count();
        let over40 = gate.new.iter().filter(|&&v| v > 0.40).count();
        println!(
            "  at new cap: {}/{} crops exceed 0.30 ({:.0}%), {}/{} exceed 0.40 — grounds the 0.30 gate.",
            over30, gate.new.len(), 100.0 * over30 as f64 / gate.new.len() as f64, over40, gate.new.len()
        );
    }
}

fn build_html(
    crops: &[Crop],
    grid: &[(usize, Vec<Cell>)],
    knee: u32,
    gate: &GateSample,
) -> String {
    let mut s = String::new();
    s.push_str("<!doctype html><meta charset=utf-8><title>maxiter escalation</title>");
    s.push_str("<style>body{background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif;margin:16px}\
h1{font-size:18px}h2{font-size:14px;color:#9cf;margin:18px 0 6px}\
.row{display:flex;gap:8px;align-items:flex-start;margin:10px 0;flex-wrap:nowrap;overflow-x:auto}\
.cell{flex:0 0 auto}.cell img{display:block;width:320px;height:180px;border:1px solid #333}\
.cap{font-weight:bold}.knee img{border:2px solid #5c5}.cap.kneecap{color:#5c5}\
.cm{color:#888}.lbl{width:170px;flex:0 0 auto;color:#bbb}.note{color:#888;margin:4px 0 14px}</style>");
    s.push_str(&format!(
        "<h1>maxiter escalation</h1><div class=note>locked quality grid ss4 + Lanczos-3 · \
         residual = pinned-at-cap (no-escape) fraction · green = PROVISIONAL knee cap {knee} \
         (confirm by eye — pick where the cores go clean)</div>"
    ));
    for (ci, cells) in grid {
        let crop = &crops[*ci];
        let src = if crop.src_black.is_nan() {
            "test location".to_string()
        } else {
            format!("manifest black_frac {:.3}", crop.src_black)
        };
        s.push_str("<div class=row>");
        s.push_str(&format!(
            "<div class=lbl><div class=cap>{}</div><div class=cm>fw {:.3e}<br>{}</div></div>",
            crop.label, crop.fw, src
        ));
        for cell in cells {
            let knee_cls = if cell.cap == knee { " knee" } else { "" };
            let cap_cls = if cell.cap == knee { "cap kneecap" } else { "cap" };
            s.push_str(&format!(
                "<div class='cell{knee_cls}'><img loading=lazy src='{}'>\
                 <div class='{cap_cls}'>cap {}</div>\
                 <div class=cm>resid {:.4}<br>occ {:.3} · {:.0}s</div></div>",
                cell.png, cell.cap, cell.residual, cell.occ, cell.secs
            ));
        }
        s.push_str("</div>");
    }
    if !gate.new.is_empty() {
        let over30 = gate.new.iter().filter(|&&v| v > 0.30).count();
        s.push_str(&format!(
            "<h2>black-gate calibration @ cap {}</h2><div class=note>{} sampled manifest crops \
             re-rendered at the cheap gate resolution; {}/{} now exceed the proposed 0.30 gate.</div>",
            gate.cap, gate.new.len(), over30, gate.new.len()
        ));
    }
    s
}

fn build_json(
    crops: &[Crop],
    caps: &[u32],
    grid: &[(usize, Vec<Cell>)],
    knee: u32,
    gate: &GateSample,
) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"caps\": {:?},\n", caps));
    s.push_str(&format!("  \"knee_provisional\": {knee},\n"));
    s.push_str("  \"crops\": [\n");
    for (i, (ci, cells)) in grid.iter().enumerate() {
        let crop = &crops[*ci];
        s.push_str(&format!(
            "    {{ \"label\": \"{}\", \"cx\": {}, \"cy\": {}, \"fw\": {}, \"is_test\": {}, \"cells\": [",
            crop.label, crop.cx, crop.cy, crop.fw, crop.is_test
        ));
        let cell_strs: Vec<String> = cells
            .iter()
            .map(|c| format!("{{\"cap\":{},\"residual\":{:.6},\"occ\":{:.6},\"secs\":{:.3}}}", c.cap, c.residual, c.occ, c.secs))
            .collect();
        s.push_str(&cell_strs.join(", "));
        s.push_str("] }");
        s.push_str(if i + 1 < grid.len() { ",\n" } else { "\n" });
    }
    s.push_str("  ],\n");
    s.push_str(&format!("  \"gate_sample_cap\": {},\n", gate.cap));
    s.push_str(&format!("  \"gate_new\": {:?},\n", gate.new));
    s.push_str(&format!("  \"gate_old\": {:?}\n", gate.old));
    s.push_str("}\n");
    s
}

// ----------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------

fn parse_caps(spec: &str) -> Result<Vec<u32>, String> {
    let mut v = Vec::new();
    for p in spec.split(',') {
        let t = p.trim();
        if t.is_empty() {
            continue;
        }
        v.push(t.parse::<u32>().map_err(|_| format!("invalid --caps component '{t}'"))?);
    }
    Ok(v)
}

fn height_16x9(width: u32) -> u32 {
    (width as f64 * 9.0 / 16.0).round() as u32
}

fn parse_dec(s: &str, width: u32, fw: f64) -> Result<f64, String> {
    let prec = hp::prec_bits(width, fw);
    Ok(hp::to_f64(&hp::parse_decimal(s, prec)?))
}

fn load_palette(colormaps: &str, name: &str) -> Result<Palette, String> {
    let text = std::fs::read_to_string(colormaps).map_err(|e| format!("read {colormaps}: {e}"))?;
    let library = parse_colormaps(&text).map_err(|e| format!("parse {colormaps}: {e}"))?;
    let cm = library
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| format!("palette '{name}' not found in {colormaps}"))?;
    Ok(Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed))
}


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// `maxiter-diag` subcommand: see `maxiter_diag::run_maxiter_diag`. Diagnosis-only
/// iteration-cap escalation harness. All caps/crops/resolution overridable; the
/// defaults reproduce the loose0_v3 worst-offender escalation.
#[derive(Args, Debug)]
pub struct MaxiterDiagArgs {
    /// `present` manifest mined for worst-offender crops (highest recorded
    /// `black_fraction` per seed × composition — the grayest spiral cores).
    #[arg(long, default_value = "data/label_crops/loose0_v3/manifest.json")]
    pub manifest: String,

    /// Number of worst-offender crops to escalate (the test location is appended).
    #[arg(long, default_value_t = 3)]
    pub offenders: usize,

    /// Escalating iteration-cap series (comma-separated).
    #[arg(long, default_value = "2000,8000,32000,128000")]
    pub caps: String,

    /// Render width in px (height follows 16:9). The locked quality is otherwise
    /// fixed: grid ss4 + Lanczos-3 (the `render-one` path).
    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    /// Linear supersample factor (the lock: 4 → 16 spp).
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Palette name, looked up in `--colormaps` (selective-mirror load).
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Standard test location (minibrot eye), real part. Matches `render-one`.
    #[arg(long, default_value = "-0.746339", allow_hyphen_values = true)]
    pub test_cx: String,

    /// Standard test location, imaginary part.
    #[arg(long, default_value = "0.112242", allow_hyphen_values = true)]
    pub test_cy: String,

    /// Standard test location, frame width.
    #[arg(long, default_value_t = 0.000583)]
    pub test_fw: f64,

    /// Knee-detection epsilon: the first cap past which the max-over-crops residual
    /// change drops below this is the provisional knee.
    #[arg(long, default_value_t = 0.02)]
    pub knee_eps: f32,

    /// Number of manifest crops re-rendered (cheap gate res) at the knee cap to
    /// report the no-escape-fraction distribution. `0` skips gate calibration.
    #[arg(long, default_value_t = 80)]
    pub gate_sample: usize,

    /// Stable output directory (not under `out/`).
    #[arg(long, default_value = "data/calibration/maxiter_diag/")]
    pub out_dir: String,
}
