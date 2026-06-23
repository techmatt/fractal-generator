//! `render-one` subcommand — the **locked wallpaper-render default**.
//!
//! The AA study is settled: **grid ss4 + Lanczos-3 at 2560×1440** is the
//! production wallpaper quality. This gives that path a production home instead of
//! leaving it inside the `aa-filter` study harness. It is an **extract** of the
//! pieces `aa-filter` already built and verified — same f64 render path,
//! [`generate::color_params`], selective-mirror palette load, channel-intent
//! iterate, and [`render::shade_and_downsample_filtered`] with the `ss×`-scaled
//! reconstruction kernel — not new logic.
//!
//! Renders **one** (location × palette) at the locked quality to a caller-chosen
//! **stable** path (not `out/`), reporting iterate / filter / total wall-clock.
//! The locked values (`--width 2560 --height 1440 --ss 4 --pattern grid --filter
//! lanczos3`) live here only; the bare render path keeps its own defaults so fast
//! ss1/ss2 previews and diagnostics are unaffected.
//!
//! Shallow f64 by construction (asserted) — the discovery pipeline that produces
//! wallpaper locations (`generate` → `present`) works in the f64 cheap regime, so
//! deep-zoom perturbation is out of scope here. At 2560×1440 the ss4 transient
//! (~2.8 GB) is fine; tiled supersampling is the escape hatch for higher-res
//! masters (constant buffer regardless of ss/res) — noted, not built.

use std::time::Instant;

use num_complex::Complex;

use crate::backend::{F64Backend, Trap, TrapShape};
use crate::cli::{PatternChoice, RenderOneArgs};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::render::{self, Frame};
use crate::{coloring, ensure_parent_dir, hp};

/// Escape radius (matches the generate/present/aa-study/aa-filter regime).
const BAILOUT: f64 = 1e6;

pub fn run_render_one(args: &RenderOneArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.supersample.max(1);

    // Center at full precision (parsed exactly as render/aa-filter do, so the
    // f64 projection matches bit-for-bit).
    let prec_bits = hp::prec_bits(args.width, args.frame_width);
    let cx = hp::to_f64(&hp::parse_decimal(&args.center_re, prec_bits)?);
    let cy = hp::to_f64(&hp::parse_decimal(&args.center_im, prec_bits)?);
    let center = Complex::new(cx, cy);

    let frame = Frame {
        center,
        frame_width: args.frame_width,
        out_width: args.width,
        out_height: args.height,
    };
    let pixel_spacing = frame.pixel_size();

    // Shallow-regime assertion: the locked path is f64 ground truth. The wallpaper
    // discovery pipeline (generate → present) lives in this regime; deep-zoom
    // perturbation is deferred.
    if pixel_spacing <= 1e-13 {
        return Err(format!(
            "pixel spacing {pixel_spacing:.3e} is inside f64's quantization regime — \
             render-one is the shallow f64 wallpaper path (perturbation/floatexp deferred). \
             Use a shallower frame width."
        ));
    }

    // Rotated-grid (4-rooks) is an ss2-only placement; refuse rather than silently
    // fall back to grid centers.
    if args.pattern == PatternChoice::Rgss && ss != 2 {
        return Err(format!(
            "--pattern rgss is ss2-only (4-rooks); got --ss {ss}. Use grid or jitter."
        ));
    }

    // Palette through the SAME selective-mirror path as present/aa-filter, so any
    // library palette (cyclic or sequential `mirror_needed`) renders seam-free.
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let library =
        parse_colormaps(&cm_text).map_err(|e| format!("parse {}: {e}", args.colormaps))?;
    let cm = library
        .iter()
        .find(|c| c.name == args.palette)
        .ok_or_else(|| format!("palette '{}' not found in {}", args.palette, args.colormaps))?;
    let palette =
        Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed);

    let params = color_params();
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    eprintln!(
        "render-one: center ({cx:.9}, {cy:.9}) fw {:.3e}  {}x{}  {} ss{ss}  {}  maxiter {}  palette '{}'",
        args.frame_width,
        args.width,
        args.height,
        args.pattern.label(),
        args.filter.label(),
        args.maxiter,
        palette.name()
    );

    // --- iterate (the only expensive stage) ---
    let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
    let t_iter = Instant::now();
    let buf = render::iterate_samples_f64_pattern(
        &backend,
        &frame,
        ss,
        channels,
        args.pattern.into(),
        args.seed,
    );
    let iter_secs = t_iter.elapsed().as_secs_f64();
    if buf.glitched_pixels > 0 {
        eprintln!("warning: {} glitched pixel(s)", buf.glitched_pixels);
    }

    // --- shade + downsample with the locked reconstruction filter ---
    let t_filter = Instant::now();
    let img = render::shade_and_downsample_filtered(
        &buf.samples,
        args.width,
        args.height,
        ss,
        &palette,
        &params,
        pixel_spacing,
        args.filter.into(),
    );
    let filter_secs = t_filter.elapsed().as_secs_f64();
    drop(buf);

    ensure_parent_dir(&args.out)?;
    img.save(&args.out)
        .map_err(|e| format!("failed to write {}: {e}", args.out))?;

    let total = iter_secs + filter_secs;
    eprintln!(
        "wrote {} in {total:.2}s (iterate {iter_secs:.2}s + filter {filter_secs:.3}s)",
        args.out
    );
    println!("=== render-one ===");
    println!("out:     {}", args.out);
    println!("iterate: {iter_secs:.3}s");
    println!("filter:  {filter_secs:.3}s");
    println!("total:   {total:.3}s");
    Ok(())
}
