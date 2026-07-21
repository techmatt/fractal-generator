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

use crate::backend::{F64Backend, JuliaBackend, Trap, TrapShape};
use clap::{Args, ValueEnum};
use crate::generate::color_params;
use crate::palette::Palette;
use crate::palette_pick::parse_colormaps;
use crate::render::{self, Frame};
use crate::render_modes::{self, ColoringParams, Family};
use crate::{coloring, ensure_parent_dir, hp};

/// Resolve `--coloring` to a beautiful [`ColoringParams`], or `None` for the
/// **location profile** (the settled byte-identical path). `None` is returned for
/// an absent flag *and* for an explicit default (`--coloring '{}'` or any spec
/// that resolves to [`ColoringParams::default`]) — both must reproduce the current
/// output exactly. A `@path` prefix reads the JSON from a file.
fn resolve_coloring(arg: &Option<String>) -> Result<Option<ColoringParams>, String> {
    let raw = match arg {
        None => return Ok(None),
        Some(s) => s,
    };
    let text = if let Some(path) = raw.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|e| format!("read --coloring file {path}: {e}"))?
    } else {
        raw.clone()
    };
    let cp = ColoringParams::from_json(&text)?;
    Ok((!cp.is_location_profile()).then_some(cp))
}

/// Escape radius (matches the generate/present/aa-study/aa-filter regime).
const BAILOUT: f64 = 1e6;

pub fn run_render_one(args: &RenderOneArgs) -> Result<(), String> {
    if args.width == 0 || args.height == 0 {
        return Err("--width and --height must be > 0".into());
    }
    let ss = args.supersample.max(1);

    // Flag-combo validation (string level, before parsing constants):
    //  - `--julia` flips a `z^d+c` parameter plane into its dynamical (fixed-c,
    //    z₀=pixel) twin: valid with `mandelbrot` (quadratic Julia) and
    //    `multibrot3|4|5` (Julia-multibrot). NOT with `phoenix` (its own dynamical
    //    two-state plane).
    //  - `--p` (Phoenix's z_{n-1} coefficient) is valid only for `--family phoenix`.
    if args.julia && args.family == FamilyChoice::Phoenix {
        return Err(
            "--julia (fixed-c dynamical plane) is incompatible with --family phoenix, which is \
             already its own dynamical two-state plane"
                .into(),
        );
    }
    if args.phoenix_p.is_some() && args.family != FamilyChoice::Phoenix {
        return Err("--p is the Phoenix z_{n-1} coefficient; valid only with --family phoenix".into());
    }
    if args.phoenix_z1.is_some() && args.family != FamilyChoice::Phoenix {
        return Err("--phoenix-z1 is the Phoenix z_{-1} slice coordinate; valid only with --family phoenix".into());
    }

    // Center at full precision (parsed exactly as render/aa-filter do, so the
    // f64 projection matches bit-for-bit).
    let prec_bits = hp::prec_bits(args.width, args.frame_width);
    let cx = hp::to_f64(&hp::parse_decimal(&args.center_re, prec_bits)?);
    let cy = hp::to_f64(&hp::parse_decimal(&args.center_im, prec_bits)?);
    let center = Complex::new(cx, cy);

    // Parse a fixed complex constant (two decimal strings) → f64, exactly like the
    // center (these are shallow base-scale renders — f64 is accurate).
    let parse_const = |v: &[String], name: &str| -> Result<Complex<f64>, String> {
        if v.len() != 2 {
            return Err(format!("--{name} expects exactly two values <re> <im>, got {}", v.len()));
        }
        let re = hp::to_f64(&hp::parse_decimal(&v[0], prec_bits)?);
        let im = hp::to_f64(&hp::parse_decimal(&v[1], prec_bits)?);
        Ok(Complex::new(re, im))
    };

    // Resolve the render family + its fixed constants. `c_strings`/`p_strings` carry
    // the original decimal strings for the dump-field sidecar (dynamical families).
    let (family, c_strings, p_strings, z1_strings): (
        Family,
        Option<(String, String)>,
        Option<(String, String)>,
        Option<(String, String)>,
    ) = match args.family {
        FamilyChoice::Mandelbrot => {
            if args.julia {
                let c = args
                    .julia_c
                    .as_ref()
                    .ok_or("--julia requires --c <re> <im> (the Julia parameter)")?;
                (
                    Family::Julia { c: parse_const(c, "c")?, degree: 2 },
                    Some((c[0].clone(), c[1].clone())),
                    None,
                    None,
                )
            } else {
                if args.julia_c.is_some() {
                    return Err("--c is the Julia parameter and is meaningless on a plain \
                                Mandelbrot render (did you mean --julia?)"
                        .into());
                }
                (Family::Mandelbrot, None, None, None)
            }
        }
        FamilyChoice::Multibrot3 | FamilyChoice::Multibrot4 | FamilyChoice::Multibrot5 => {
            let degree = match args.family {
                FamilyChoice::Multibrot3 => 3,
                FamilyChoice::Multibrot4 => 4,
                _ => 5,
            };
            if args.julia {
                // Julia-multibrot: dynamical z^d + c at the fixed parameter `c`.
                let c = args
                    .julia_c
                    .as_ref()
                    .ok_or("--julia --family multibrot* requires --c <re> <im> (the fixed parameter)")?;
                (
                    Family::Julia { c: parse_const(c, "c")?, degree },
                    Some((c[0].clone(), c[1].clone())),
                    None,
                    None,
                )
            } else {
                if args.julia_c.is_some() {
                    return Err("--c is the (Julia) fixed parameter; pass --julia to render the \
                                dynamical z^d+c plane, or drop --c for the parameter-plane multibrot"
                        .into());
                }
                (Family::Multibrot { degree }, None, None, None)
            }
        }
        FamilyChoice::Phoenix => {
            // Ushiki Phoenix: `--c` (additive const), `--p` (z_{n-1} coeff), and
            // `--phoenix-z1` (z_{-1} slice coordinate) all optional, defaulting to the
            // classic real-valued spot c≈0.5667, p≈-0.5, z_{-1}=0.
            let cs = args
                .julia_c
                .clone()
                .unwrap_or_else(|| vec!["0.5667".into(), "0".into()]);
            let ps = args
                .phoenix_p
                .clone()
                .unwrap_or_else(|| vec!["-0.5".into(), "0".into()]);
            let zs = args
                .phoenix_z1
                .clone()
                .unwrap_or_else(|| vec!["0".into(), "0".into()]);
            let c = parse_const(&cs, "c")?;
            let p = parse_const(&ps, "p")?;
            let z_m1 = parse_const(&zs, "phoenix-z1")?;
            (
                Family::Phoenix { c, p, z_m1 },
                Some((cs[0].clone(), cs[1].clone())),
                Some((ps[0].clone(), ps[1].clone())),
                Some((zs[0].clone(), zs[1].clone())),
            )
        }
    };
    // Location-profile (settled byte-identical) backends exist only for the two
    // degree-2 families; the new families (Julia-multibrot d≥3, Multibrot, Phoenix)
    // always render through the beautiful path.
    let has_location_profile =
        matches!(family, Family::Mandelbrot | Family::Julia { degree: 2, .. });

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

    // Beautiful coloring (non-default `--coloring`) routes to the decoupled
    // field→shade→palette pipeline; absent/default routes to the byte-identical
    // location-profile path below. Resolved before the palette build so the beautiful
    // `reverse` knob can flip the baked LUT (field-independent; the location-profile
    // path keeps `reverse = false`, preserving byte-identity).
    let coloring_params = resolve_coloring(&args.coloring)?;
    let want_reverse = coloring_params.as_ref().map_or(false, |cp| cp.reverse);

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
        Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, want_reverse, cm.mirror_needed);

    let mut params = color_params();
    params.density *= args.n_cycles; // n_cycles = palette band-repeat multiplier
    let channels = coloring::required_channels(&params);
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // --- field dump branch (serialize the raw scalar field, exit before coloring) ---
    if let Some(dump_path) = &args.dump_field {
        // The dumped field is whichever single scalar coloring mode `--coloring`
        // names (default: the beautiful smooth field); its iterate-stage inputs
        // (bailout, biomorph, stripe_density, …) come from the same spec. This is the
        // serialization half of the field⊗Python-coloring split — the Python tail
        // (`colormap.py`) is field-agnostic, so a dumped tia/stripe/curvature/… field
        // inherits the full colormap param set (reverse/transfer/log_premap/…).
        // `direct_trap` is colour-valued (no scalar reduction) and rejected below.
        let field_params = match &coloring_params {
            Some(cp) => *cp,
            None => ColoringParams::beautiful(render_modes::Field::Smooth),
        };
        let dumped_field = field_params.field;
        if dumped_field == render_modes::Field::DirectTrap {
            return Err("--dump-field: direct_trap is a colour-valued composite, not a scalar \
                        field — nothing to serialize. Use a scalar field (smooth/tia/stripe/\
                        curvature/trap_circle/…) or the Rust `--coloring` render path."
                .into());
        }
        // `beautiful` runs the generic beautiful kernel and reduces `params.field`
        // per subpixel (byte-identical to the smooth render for smooth). `f64` sources
        // from the fast escape-time backend's smooth channel — same geometry / NaN
        // seam, un-normalized value at the render path's `1e6` escape radius; for the
        // mask/std-only guard, and smooth-only (no fast twin for the other fields).
        // `report_bailout` records the escape radius actually used, so the sidecar's
        // `bailout_b` matches the field.
        let t0 = Instant::now();
        let ((field, sub_w, sub_h), report_bailout) = match args.dump_field_source {
            FieldSourceChoice::Beautiful => (
                render_modes::single_field_supersampled(
                    &frame,
                    ss,
                    args.maxiter,
                    family,
                    &field_params,
                ),
                field_params.bailout_b,
            ),
            FieldSourceChoice::F64 => {
                if dumped_field != render_modes::Field::Smooth {
                    return Err(format!(
                        "--dump-field-source f64 serializes only the smooth field, but \
                         --coloring names field={:?}. Drop --dump-field-source (use the \
                         default beautiful source) for non-smooth fields.",
                        dumped_field
                    ));
                }
                (
                    render_modes::smooth_field_f64_supersampled(
                        &frame,
                        ss,
                        args.maxiter,
                        family,
                        BAILOUT,
                    )?,
                    BAILOUT,
                )
            }
        };
        let secs = t0.elapsed().as_secs_f64();

        // Raw binary: little-endian f32, row-major.
        ensure_parent_dir(dump_path)?;
        let mut bytes = Vec::with_capacity(field.len() * 4);
        for v in &field {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dump_path, &bytes)
            .map_err(|e| format!("write field {dump_path}: {e}"))?;

        // JSON sidecar alongside the binary. `width`/`height` are the *supersampled*
        // array dims (so binary length == width·height); `supersample` folds them to
        // the eval size (out = width/ss). Location is version-invariant render keys.
        // Dynamical families (julia/phoenix) emit their fixed constant as `c_re/c_im`;
        // Phoenix additionally emits its `p_re/p_im` coefficient (extra keys the
        // Python `load_field` reader ignores). `kind` follows the family.
        let kind = family.kind_str();
        let mut c_fields = String::new();
        if let Some((re, im)) = &c_strings {
            c_fields.push_str(&format!(",\"c_re\":\"{re}\",\"c_im\":\"{im}\""));
        }
        if let Some((re, im)) = &p_strings {
            c_fields.push_str(&format!(",\"p_re\":\"{re}\",\"p_im\":\"{im}\""));
        }
        if let Some((re, im)) = &z1_strings {
            c_fields.push_str(&format!(",\"zm1_re\":\"{re}\",\"zm1_im\":\"{im}\""));
        }
        let sidecar = format!(
            "{{\"width\":{sub_w},\"height\":{sub_h},\"supersample\":{ss},\
             \"field\":\"{field_name}\",\"dtype\":\"f32\",\"layout\":\"row_major\",\
             \"bailout_b\":{bailout},\
             \"location\":{{\"kind\":\"{kind}\",\"cx\":\"{cx}\",\"cy\":\"{cy}\",\
             \"fw\":\"{fw}\",\"maxiter\":{maxiter}{c_fields}}}}}",
            field_name = dumped_field.as_str(),
            bailout = report_bailout,
            cx = args.center_re,
            cy = args.center_im,
            fw = args.frame_width,
            maxiter = args.maxiter,
        );
        let json_path = match dump_path.strip_suffix(".bin") {
            Some(stem) => format!("{stem}.json"),
            None => format!("{dump_path}.json"),
        };
        std::fs::write(&json_path, sidecar).map_err(|e| format!("write sidecar {json_path}: {e}"))?;

        eprintln!(
            "dump-field: {sub_w}x{sub_h} (ss{ss}) {field} field → {dump_path} (+ {json_path}) in {secs:.2}s",
            field = dumped_field.as_str()
        );
        println!("=== render-one (dump-field) ===");
        println!("field:   {dump_path}");
        println!("sidecar: {json_path}");
        println!("dims:    {sub_w}x{sub_h} (supersampled, ss{ss})");
        println!("total:   {secs:.3}s");
        return Ok(());
    }

    // Render coloring: an explicit `--coloring` wins; absent, the two degree-2
    // families take the settled location-profile path (`None`), while the new
    // families (no location-profile backend) default to the beautiful smooth field.
    let coloring_params: Option<ColoringParams> = match (coloring_params, has_location_profile) {
        (Some(cp), _) => Some(cp),
        (None, true) => None,
        (None, false) => Some(ColoringParams::beautiful(render_modes::Field::Smooth)),
    };
    // n_cycles also multiplies the beautiful path's gradient wraps.
    let coloring_params = coloring_params.map(|mut cp| {
        cp.palette_cycles *= args.n_cycles;
        cp
    });

    let mode = match family {
        Family::Mandelbrot => "mandelbrot".to_string(),
        Family::Julia { c, degree: 2 } => format!("julia c=({:.9}, {:.9})", c.re, c.im),
        Family::Julia { c, degree } => {
            format!("julia-multibrot d={degree} c=({:.9}, {:.9})", c.re, c.im)
        }
        Family::Multibrot { degree } => format!("multibrot d={degree}"),
        Family::Phoenix { c, p, z_m1 } => {
            format!(
                "phoenix c=({:.4}, {:.4}) p=({:.4}, {:.4}) z_-1=({:.4}, {:.4})",
                c.re, c.im, p.re, p.im, z_m1.re, z_m1.im
            )
        }
    };
    let coloring_label = match &coloring_params {
        Some(cp) => format!("field={:?} transform={:?} shade={:?} biomorph={:?} B={:.0}",
            cp.field, cp.transform, cp.shade, cp.biomorph, cp.bailout_b),
        None => "location-profile (smooth)".to_string(),
    };
    eprintln!(
        "render-one [{mode}]: center ({cx:.9}, {cy:.9}) fw {:.3e}  {}x{}  {} ss{ss}  {}  maxiter {}  palette '{}'  coloring: {coloring_label}",
        args.frame_width,
        args.width,
        args.height,
        args.pattern.label(),
        args.filter.label(),
        args.maxiter,
        palette.name()
    );

    // --- beautiful pipeline branch (global-normalized field render) ---
    if let Some(cp) = &coloring_params {
        let t0 = Instant::now();
        let img = render_modes::render_beautiful(
            &frame,
            ss,
            args.maxiter,
            family,
            cp,
            &palette,
            args.filter.into(),
        );
        let secs = t0.elapsed().as_secs_f64();
        ensure_parent_dir(&args.out)?;
        let lower = args.out.to_ascii_lowercase();
        if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
            render::save_jpeg(&img, std::path::Path::new(&args.out), args.jpg_quality)?;
        } else {
            img.save(&args.out)
                .map_err(|e| format!("failed to write {}: {e}", args.out))?;
        }
        eprintln!("wrote {} in {secs:.2}s (beautiful pipeline)", args.out);
        println!("=== render-one (beautiful) ===");
        println!("out:      {}", args.out);
        println!("coloring: {}", cp.to_json());
        println!("total:    {secs:.3}s");
        return Ok(());
    }

    // --- iterate (the only expensive stage) ---
    // Location-profile path: only the two degree-2 families reach here (the new
    // families were routed to the beautiful branch above). Mandelbrot uses the
    // channel-intent f64 path; Julia uses the JuliaBackend (z₀ = pixel, fixed
    // parameter). `channels` is irrelevant to Julia (it always computes the gated
    // trap and never a `dz`), so it is unused on that branch.
    let t_iter = Instant::now();
    let buf = match family {
        Family::Mandelbrot => {
            let backend = F64Backend::new(args.maxiter, BAILOUT, trap);
            render::iterate_samples_f64_pattern(
                &backend,
                &frame,
                ss,
                channels,
                args.pattern.into(),
                args.seed,
            )
        }
        Family::Julia { c, degree: 2 } => {
            let backend = JuliaBackend::new(c, args.maxiter, BAILOUT, trap);
            render::iterate_samples_julia_pattern(
                &backend,
                &frame,
                ss,
                args.pattern.into(),
                args.seed,
            )
        }
        // Unreachable: new families always set `coloring_params = Some(..)` above,
        // so they take the beautiful branch and never fall through here.
        other => {
            return Err(format!(
                "internal: {} has no location-profile backend (should have routed to beautiful)",
                other.kind_str()
            ));
        }
    };
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
    // Dispatch on extension: `.jpg`/`.jpeg` → quality-controlled JPEG (matches the
    // present/enrich crop writer), anything else → the image-crate default (PNG).
    let lower = args.out.to_ascii_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        render::save_jpeg(&img, std::path::Path::new(&args.out), args.jpg_quality)?;
    } else {
        img.save(&args.out)
            .map_err(|e| format!("failed to write {}: {e}", args.out))?;
    }

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


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====
/// Fractal family selector for `render-one`. `Mandelbrot` (default) is the classic
/// `z²+c` parameter plane (pair with `--julia`/`--c` for the Julia dynamical plane);
/// `Multibrot3/4/5` are `z^d+c` parameter planes; `Phoenix` is the Ushiki two-state
/// dynamical plane. The two degree-2 families keep the settled location-profile
/// render path; the new families render through the beautiful smooth pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FamilyChoice {
    /// `z → z² + c`, `z₀ = 0` (or Julia with `--julia`/`--c`).
    Mandelbrot,
    /// `z → z³ + c`, parameter plane.
    Multibrot3,
    /// `z → z⁴ + c`, parameter plane.
    Multibrot4,
    /// `z → z⁵ + c`, parameter plane.
    Multibrot5,
    /// Ushiki Phoenix `z_{n+1} = z_n² + c + p·z_{n-1}`, dynamical.
    Phoenix,
}

/// Field source for `render-one --dump-field`. `Beautiful` (default) runs the
/// generic beautiful smooth kernel ([`render_modes::smooth_field_supersampled`]) —
/// the **byte-identical** field the field⊗colormap reproduction path requires.
/// `F64` sources the field from the fast escape-time [`crate::backend::F64Backend`]
/// (Mandelbrot) / [`crate::backend::JuliaBackend`] (Julia) instead: same geometry
/// and NaN-interior seam, but the un-normalized smooth value (differs from
/// `Beautiful` by the constant `ln(ln B)/ln d`). Intended for **mask/statistic**
/// consumers that read only the escape mask and a std (the degenerate-outcome
/// guard) — not for byte-faithful recoloring.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FieldSourceChoice {
    /// Generic beautiful smooth kernel (byte-identical field; colormap-split source).
    Beautiful,
    /// Fast escape-time backend smooth channel (mask/std-faithful, offset field).
    F64,
}

/// Sub-pixel sample placement for `render-one` (maps to [`crate::render::SubsamplePattern`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum PatternChoice {
    /// Ordered grid (the lock; byte-identical historical path). Any `ss`.
    Grid,
    /// Rotated grid / 4-rooks. **ss2 only.**
    Rgss,
    /// Stratified jitter (seeded). Any `ss`.
    Jitter,
}

impl PatternChoice {
    pub fn label(self) -> &'static str {
        match self {
            PatternChoice::Grid => "grid",
            PatternChoice::Rgss => "rgss",
            PatternChoice::Jitter => "jitter",
        }
    }
}

impl From<PatternChoice> for crate::render::SubsamplePattern {
    fn from(p: PatternChoice) -> Self {
        match p {
            PatternChoice::Grid => crate::render::SubsamplePattern::Grid,
            PatternChoice::Rgss => crate::render::SubsamplePattern::Rgss,
            PatternChoice::Jitter => crate::render::SubsamplePattern::Jitter,
        }
    }
}

/// Downsample reconstruction filter for `render-one` (maps to [`crate::render::DownsampleFilter`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum FilterChoice {
    /// Flat `ss×ss` average.
    Box,
    /// Mitchell–Netravali cubic.
    Mitchell,
    /// Lanczos-3 windowed sinc (the lock).
    Lanczos3,
}

impl FilterChoice {
    pub fn label(self) -> &'static str {
        match self {
            FilterChoice::Box => "box",
            FilterChoice::Mitchell => "mitchell",
            FilterChoice::Lanczos3 => "lanczos3",
        }
    }
}

impl From<FilterChoice> for crate::render::DownsampleFilter {
    fn from(f: FilterChoice) -> Self {
        match f {
            FilterChoice::Box => crate::render::DownsampleFilter::Box,
            FilterChoice::Mitchell => crate::render::DownsampleFilter::Mitchell,
            FilterChoice::Lanczos3 => crate::render::DownsampleFilter::Lanczos3,
        }
    }
}

/// `render-one` subcommand: see `render_one::run_render_one`. One location ×
/// palette at the locked wallpaper quality. Locked defaults (all overridable):
/// `--width 2560 --height 1440 --ss 4 --pattern grid --filter lanczos3`.
#[derive(Args, Debug)]
pub struct RenderOneArgs {
    /// Frame center, real part (`--cx`) — arbitrary-precision decimal string.
    #[arg(long = "cx", default_value = "-0.746339", allow_hyphen_values = true)]
    pub center_re: String,

    /// Frame center, imaginary part (`--cy`) — arbitrary-precision decimal string.
    #[arg(long = "cy", default_value = "0.112242", allow_hyphen_values = true)]
    pub center_im: String,

    /// Frame width in the complex plane (`--fw`).
    #[arg(long = "fw", default_value_t = 0.000583)]
    pub frame_width: f64,

    /// Fractal **family**. `mandelbrot` (default) is the classic `z²+c` parameter
    /// plane; `multibrot3`/`multibrot4`/`multibrot5` are `z^d+c` (parameter plane);
    /// `phoenix` is the Ushiki two-state dynamical plane. `--julia` pairs only with
    /// `mandelbrot`; `phoenix` reuses `--c` (additive constant) and adds `--p`.
    #[arg(long, value_enum, default_value_t = FamilyChoice::Mandelbrot)]
    pub family: FamilyChoice,

    /// Render a **dynamical** (Julia) plane instead of the parameter plane. Off ⇒
    /// today's parameter-plane behavior exactly (bit-for-bit). When on, the viewport
    /// (`--cx`/`--cy`/`--fw`) addresses the **z-plane** (`z₀ = pixel`) and `--c`
    /// supplies the fixed parameter. Pairs with `--family mandelbrot` (quadratic
    /// Julia) or `--family multibrot3|4|5` (**Julia-multibrot**, dynamical `z^d+c`).
    /// Requires `--c`; using `--c` without `--julia` is an error. Incompatible with
    /// `--family phoenix` (already its own dynamical plane).
    #[arg(long, default_value_t = false)]
    pub julia: bool,

    /// Fixed constant `c` as two arbitrary-precision decimal strings:
    /// `--c <re> <im>` (e.g. `--c -0.8 0.156`). The **Julia parameter** (required iff
    /// `--julia`, error without it) or the **Phoenix additive constant** (optional
    /// under `--family phoenix`, defaults to the classic `0.5667 0`).
    #[arg(
        long = "c",
        num_args = 2,
        value_names = ["RE", "IM"],
        allow_hyphen_values = true
    )]
    pub julia_c: Option<Vec<String>>,

    /// Phoenix second constant `p` (the `z_{n-1}` coefficient / Ushiki's `q`) as two
    /// decimal strings `--p <re> <im>`. Valid only with `--family phoenix`; defaults
    /// to the classic `-0.5 0`. Carried through for future exploration even though
    /// the classic instance is the only known-good point today.
    #[arg(
        long = "p",
        num_args = 2,
        value_names = ["RE", "IM"],
        allow_hyphen_values = true
    )]
    pub phoenix_p: Option<Vec<String>>,

    /// Phoenix slice coordinate `z_{-1}` as two decimal strings `--phoenix-z1 <re> <im>`.
    /// Valid only with `--family phoenix`; defaults to the legacy `0 0`. A non-zero
    /// value breaks the slice symmetry and yields a different set from the same
    /// `(c, p)` (see `docs/design/phoenix_seed_sampler_spec.md` §3).
    #[arg(
        long = "phoenix-z1",
        num_args = 2,
        value_names = ["RE", "IM"],
        allow_hyphen_values = true
    )]
    pub phoenix_z1: Option<Vec<String>>,

    /// Palette name, looked up in `--colormaps` (loaded through the selective-mirror
    /// path, so cyclic and sequential maps both render seam-free).
    #[arg(long, default_value = "twilight")]
    pub palette: String,

    /// Colormap library (carries the inline `mirror_needed` flag).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Output PNG path — a stable path the caller chooses (not under `out/`).
    #[arg(long, default_value = "render.png")]
    pub out: String,

    /// Output width in pixels (the lock: 2560).
    #[arg(long, default_value_t = 2560)]
    pub width: u32,

    /// Output height in pixels (the lock: 1440).
    #[arg(long, default_value_t = 1440)]
    pub height: u32,

    /// Linear supersampling factor (the lock: 4 → 16 spp).
    #[arg(long, default_value_t = 4)]
    pub supersample: u32,

    /// Sub-pixel sample placement (the lock: grid).
    #[arg(long, value_enum, default_value_t = PatternChoice::Grid)]
    pub pattern: PatternChoice,

    /// Downsample reconstruction filter (the lock: lanczos3).
    #[arg(long, value_enum, default_value_t = FilterChoice::Lanczos3)]
    pub filter: FilterChoice,

    /// Maximum iterations / orbit cap ("max_orbit"). Raised 2000 → 8000 (the
    /// `maxiter-blackgate` pass, Matt's pick): the escalation sheet's residual
    /// pinned-at-cap fraction asymptotes by ~8k (max-over-crops |Δ| drops below
    /// 0.02 at the 8k→32k step — the measured knee), what remains is genuine
    /// minibrot interior no cap reclaims. 8000 is the knee: ~3.3–3.5× the cap-2000
    /// cost on interior-heavy frames, near-free on filament frames.
    #[arg(long, default_value_t = 8000)]
    pub maxiter: u32,

    /// SplitMix64 seed (consumed only by `--pattern jitter`).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Palette **cycle count**: the number of times the gradient wraps across the
    /// escape range (band-repeat multiplier). `1.0` = the single-pass default we've
    /// always previewed; `N` tiles the palette `N` times, producing concentric
    /// bands. Location-profile path scales the smooth `density`; the beautiful path
    /// scales `palette_cycles`. Cyclic (endpoint-matched) palettes tile seamlessly.
    #[arg(long = "n-cycles", default_value_t = 1.0)]
    pub n_cycles: f64,

    /// JPEG quality (1..=100) used only when `--out` ends in `.jpg`/`.jpeg`.
    /// Ignored for PNG output. q95 keeps cache renders clean enough that the
    /// train-time JPEG-q jitter (85..95) does not compound artifacts.
    #[arg(long, default_value_t = 95)]
    pub jpg_quality: u8,

    /// Dump the **raw scalar field of a single coloring mode** (pre-coloring: before
    /// percentile-stretch, transform, gamma, shade, palette) to `<path>` as
    /// little-endian `f32`, row-major, at the supersampled resolution
    /// (`--width·ss × --height·ss`); interior / non-escaped subpixels are `NaN`. Also
    /// writes a JSON sidecar (`<path>` with `.bin`→`.json`, else `<path>.json`)
    /// describing dims / ss / location / **field name**. Computes the field and
    /// **exits before coloring** — no PNG is written. The field is whichever single
    /// scalar mode `--coloring` names (`smooth` default, or `tia`/`stripe`/`curvature`/
    /// `trap_circle`/…); `direct_trap` is colour-valued and rejected. Its bailout /
    /// iterate knobs follow the same `--coloring` spec, else the beautiful default
    /// (`2^16`). This is the serialization half of the field⊗Python-coloring split —
    /// the Python `colormap.py` tail is field-agnostic, so any dumped field inherits
    /// the full colormap param set (reverse / transfer / log_premap / n_cycles / phase).
    #[arg(long)]
    pub dump_field: Option<String>,

    /// Source kernel for `--dump-field`. `beautiful` (default) dumps the
    /// byte-identical beautiful smooth field (the field⊗colormap reproduction path
    /// depends on this). `f64` dumps the fast escape-time backend's smooth channel
    /// (Mandelbrot/Julia only) — same geometry and NaN-interior seam, but an
    /// un-normalized smooth value (a constant offset from `beautiful`). Use `f64`
    /// for the degenerate-outcome guard, which reads only the escape mask
    /// (`interior_frac`) and a std (`field_std`), both invariant to that offset; the
    /// `f64` bailout is fixed to the render path's `1e6` escape radius, ignoring any
    /// `--coloring` bailout. No effect without `--dump-field`.
    #[arg(long, value_enum, default_value_t = FieldSourceChoice::Beautiful)]
    pub dump_field_source: FieldSourceChoice,

    /// Beautiful coloring params as a JSON object (inline, or `@path` to read from
    /// a file). Omitted — or any spec that resolves to the default (e.g. `{}`) —
    /// renders the **location profile** (current output, byte-identical). A
    /// non-default spec routes to the decoupled field→shade→palette pipeline
    /// (`render_modes`). Keys: `field` (smooth|stripe|tia|curvature|trap_circle|
    /// trap_cross|velocity|de|gaussian_int|exp_smoothing|decomposition|direct_trap),
    /// `bailout_b`, `skip`, `biomorph` (off|epsilon_cross),
    /// `stripe_density`, `trap_radius`, `color_by`
    /// (minimum_distance|average_distance|maximum_distance|iter_min|iter_max|
    /// angle_min|angle_max|mean_angle|ratio — gaussian_int only),
    /// `de_scale`, `direct_threshold`,
    /// `direct_opacity` / `merge_mode` (normal|multiply|screen|overlay) /
    /// `merge_order` (bottom_up|top_down) / `start_color` (black|white|#rrggbb)
    /// (direct_trap), `transform`
    /// (linear|sqrt|log|histeq|scurve),
    /// `gamma`, `shade` (none|normal_map), `light_azimuth`, `light_height`,
    /// `palette_cycles`, `palette_offset`. A spec naming any key seeds from
    /// `beautiful(field)` (so omitted keys follow the field preset, e.g.
    /// `{"field":"stripe"}` ≡ density-6 / linear / 2^16); an empty `{}` (or absent
    /// flag) renders the location profile.
    #[arg(long)]
    pub coloring: Option<String>,
}
