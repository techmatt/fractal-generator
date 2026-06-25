//! `focus-diag` — **measurement-only** dynamics-field dumper for the
//! smoothed-escape focal-point + scale-space-organization exploration (CC prompt:
//! focus-field-exploration).
//!
//! This builds nothing and gates nothing. It is the field-array sibling of
//! [`crate::gate_diag`]: where `gate-diag` reduces each crop to a CSV row of
//! scalars, `focus-diag` dumps the **full 2D dynamics fields** the scale-space
//! analysis needs, so all the smoothing / maxima / persistence / organization
//! scalars live Python-side (scipy) off these arrays.
//!
//! For each frame in a JSONL frames file (one compact `{name,cx,cy,fw,width}`
//! object per line — emitted by the Python driver), it renders the cheap f64
//! screen at the stored crop frame (reusing the production kernel via
//! [`render::iterate_samples_f64`] with the `de` channel on, `trap` off) and
//! writes three row-major arrays plus a manifest:
//!  - `<name>_mu.f32`     — smooth escape `mu = (n+1) − log2 ln|z|` (the
//!    `PixelSample::smooth_iter`), `NaN` on interior/non-escaping pixels.
//!  - `<name>_depx.f32`   — distance estimate in **final-wallpaper pixels**
//!    (`de_px = de / (fw / de_ref_width)`, pinned to 2560 like gate-diag so it is
//!    independent of the field screen size), `NaN` on interior.
//!  - `<name>_interior.u8`— 1 where the orbit did not escape (the set), else 0.
//!    The explicit mask drives the Python minibrot-exclusion dilation.
//!
//! The potential `G ≈ 2^-mu` (bounded, →0 at the boundary) and the smoothed
//! candidate focus fields P/Q/R are all derived Python-side from these three
//! arrays — Rust only owns the kernel.
//!
//! DE needs a large bailout (≥~1e6) for a stable estimate; the default 1e6
//! (≈2^20) matches `present`/`generate` and is already in the ideal band. The
//! manifest's embedded `"maxiter 2000"` string is **stale** — the production
//! `present`/`render-one` default is 8000, which is what this defaults to.

use std::fs;
use std::path::Path;

use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::FocusDiagArgs;
use crate::coloring::ChannelSet;
use crate::render::{self, Frame};

/// One frame to dump: a name tag plus the stored crop geometry and field res.
struct FrameSpec {
    name: String,
    cx: f64,
    cy: f64,
    fw: f64,
    width: u32,
}

// ---------- hand-rolled JSONL field parsers (one frame object per line) -------
// Field readers shared via `crate::jsonl` (no serde dep).
use crate::jsonl::*;

fn parse_frames(text: &str, default_width: u32) -> Result<Vec<FrameSpec>, String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.contains("\"name\"") || !line.contains("\"cx\"") {
            continue;
        }
        out.push(FrameSpec {
            name: field_str(line, "name").ok_or_else(|| format!("frames: bad name in: {line}"))?,
            cx: field_f64(line, "cx").ok_or("frames: bad cx")?,
            cy: field_f64(line, "cy").ok_or("frames: bad cy")?,
            fw: field_f64(line, "fw").ok_or("frames: bad fw")?,
            width: field_f64(line, "width").map(|w| w as u32).unwrap_or(default_width),
        });
    }
    Ok(out)
}

/// Render one frame's dynamics fields and write the three arrays + return the
/// manifest fragment line. Row-major, row 0 = top (matches the render grid).
fn dump_frame(spec: &FrameSpec, args: &FocusDiagArgs, out_dir: &Path) -> Result<String, String> {
    let width = spec.width.max(1);
    let height = (width as f64 * 9.0 / 16.0).round().max(1.0) as u32; // crops are 16:9
    let frame = Frame {
        center: Complex::new(spec.cx, spec.cy),
        frame_width: spec.fw,
        out_width: width,
        out_height: height,
    };
    // DE on, trap off (unused). Trap ctor still needs a shape.
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    let backend = F64Backend::new(args.maxiter, args.bailout, trap);
    let channels = ChannelSet { trap: false, de: true };
    // ss = 1: the fields are smoothed downstream; AA would only blur the peaks we
    // are about to detect. Row-major, deterministic.
    let buf = render::iterate_samples_f64(&backend, &frame, 1, channels);

    let ref_spacing = spec.fw / args.de_ref_width as f64; // de_px = de / ref_spacing
    let n = buf.samples.len();
    let mut mu = Vec::with_capacity(n * 4);
    let mut depx = Vec::with_capacity(n * 4);
    let mut interior = Vec::with_capacity(n);
    for s in &buf.samples {
        if s.escaped {
            mu.extend_from_slice(&(s.smooth_iter as f32).to_le_bytes());
            depx.extend_from_slice(&((s.de / ref_spacing) as f32).to_le_bytes());
            interior.push(0u8);
        } else {
            mu.extend_from_slice(&f32::NAN.to_le_bytes());
            depx.extend_from_slice(&f32::NAN.to_le_bytes());
            interior.push(1u8);
        }
    }

    let fields = out_dir.join("fields");
    let mu_path = fields.join(format!("{}_mu.f32", spec.name));
    let de_path = fields.join(format!("{}_depx.f32", spec.name));
    let in_path = fields.join(format!("{}_interior.u8", spec.name));
    fs::write(&mu_path, &mu).map_err(|e| format!("write {}: {e}", mu_path.display()))?;
    fs::write(&de_path, &depx).map_err(|e| format!("write {}: {e}", de_path.display()))?;
    fs::write(&in_path, &interior).map_err(|e| format!("write {}: {e}", in_path.display()))?;

    let interior_frac = interior.iter().filter(|&&b| b == 1).count() as f64 / n.max(1) as f64;
    Ok(format!(
        "{{ \"name\": \"{}\", \"cx\": {}, \"cy\": {}, \"fw\": {}, \"width\": {}, \"height\": {}, \
         \"interior_frac\": {:.6} }}",
        spec.name, spec.cx, spec.cy, spec.fw, width, height, interior_frac
    ))
}

pub fn run_focus_diag(args: &FocusDiagArgs) -> Result<(), String> {
    let text = fs::read_to_string(&args.frames)
        .map_err(|e| format!("read {}: {e}", args.frames))?;
    let frames = parse_frames(&text, args.width)?;
    if frames.is_empty() {
        return Err(format!("no frames parsed from {} (expect one JSONL object per line)", args.frames));
    }
    let out_dir = Path::new(&args.out_dir);
    crate::ensure_parent_dir(out_dir.join("fields").join("x"))?;
    eprintln!(
        "focus-diag: {} frames, f64 fields (maxiter {}, bailout {:.0}, de pinned to {}px) ...",
        frames.len(), args.maxiter, args.bailout, args.de_ref_width
    );

    let t0 = std::time::Instant::now();
    let mut manifest_lines = Vec::with_capacity(frames.len());
    for (i, spec) in frames.iter().enumerate() {
        let line = dump_frame(spec, args, out_dir)?;
        manifest_lines.push(line);
        eprintln!("  [{}/{}] {} ({:.1}s)", i + 1, frames.len(), spec.name, t0.elapsed().as_secs_f64());
    }

    let manifest = format!(
        "{{\n  \"maxiter\": {},\n  \"bailout\": {},\n  \"de_ref_width\": {},\n  \"frames\": [\n    {}\n  ]\n}}\n",
        args.maxiter, args.bailout, args.de_ref_width, manifest_lines.join(",\n    ")
    );
    let mpath = out_dir.join("fields_manifest.json");
    fs::write(&mpath, manifest).map_err(|e| format!("write {}: {e}", mpath.display()))?;

    println!("=== focus-diag (measurement only) ===");
    println!("frames: {}  elapsed: {:.1}s", frames.len(), t0.elapsed().as_secs_f64());
    println!("wrote fields/ + {}", mpath.display());
    Ok(())
}
