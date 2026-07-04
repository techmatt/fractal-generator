//! Shared machinery for the depth-probe subcommands (`descend`, `navigate`).
//!
//! Both probes walk a Mandelbrot→Julia path one level at a time, accumulate the
//! center in high precision, render a `Mandelbrot | Julia` row per level, and
//! emit a tall filmstrip plus a hand-rolled JSON log. The only thing that
//! differs between them is *how the next target is chosen* — `descend` scores
//! K×K windows greedily; `navigate` finds minibrot nuclei deterministically.
//! Everything else (panel rendering, the seeded RNG, the footprint circle,
//! strip composition, path/JSON string helpers) lives here so the two
//! subcommands share one implementation.

use std::path::{Path, PathBuf};

use astro_float::BigFloat;
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::{F64Backend, FractalBackend, JuliaBackend, PerturbationBackend, Trap};
use crate::cli::{BackendChoice, ShadeArgs};
use crate::coloring::ColorParams;
use crate::palette::Palette;
use crate::render::{self, Frame, SampleBuffer};

/// Pixel spacing at/below which f64 enters its quantization regime — the auto
/// switch to perturbation (mirrors `main`'s constant).
pub const PERTURB_SPACING: f64 = 1e-13;

/// Base-scale Julia view width (whole set, center 0). f64 is always accurate
/// here, so Julia panels never need perturbation.
pub const JULIA_WIDTH: f64 = 3.5;

/// Horizontal gap (px) between the Mandelbrot and Julia panels in a row.
pub const GAP_H: u32 = 4;
/// Vertical gap (px) between rows of the filmstrip.
pub const GAP_V: u32 = 3;
/// Filmstrip background (near-black).
pub const STRIP_BG: [u8; 3] = [16, 16, 16];

/// SplitMix64 — a tiny, dependency-free seeded PRNG. Deterministic for a fixed
/// `--seed`, which is what makes a probe reproducible.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform index in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform `f64` in `[0,1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Map shared shading args to coloring parameters (identical mapping for every
/// subcommand — the probes only ever vary the palette/channel via these args).
pub fn color_params(shade: &ShadeArgs) -> ColorParams {
    ColorParams {
        density: shade.density,
        offset: shade.offset,
        channel: shade.color,
        interior: shade.interior,
        trap_scale: shade.trap_scale,
        trap_curve: shade.trap_curve,
        trap_phase_strength: shade.trap_phase_strength,
        de_shade: shade.de_shade,
        mark_glitches: shade.mark_glitches,
    }
}

/// A rendered Mandelbrot panel plus the diagnostics a probe logs per level.
pub struct MandelPanel {
    pub buf: SampleBuffer,
    pub backend_name: &'static str,
    /// Output-pixel spacing (the coloring/DE normalization constant).
    pub spacing: f64,
}

/// Iterate one Mandelbrot panel at the high-precision center, picking f64 or
/// perturbation by pixel spacing (or the explicit `--backend` override). This is
/// the only expensive stage and the sole place a probe touches a backend.
#[allow(clippy::too_many_arguments)]
pub fn render_mandel_panel(
    center_re: &BigFloat,
    center_im: &BigFloat,
    center_f64: Complex<f64>,
    width: f64,
    panel_w: u32,
    panel_h: u32,
    ss: u32,
    maxiter: u32,
    bailout: f64,
    degree: u32,
    prec: usize,
    trap: Trap,
    backend: BackendChoice,
) -> MandelPanel {
    let frame = Frame {
        center: center_f64,
        frame_width: width,
        out_width: panel_w,
        out_height: panel_h,
    };
    let spacing = frame.pixel_size();
    let use_perturb = match backend {
        BackendChoice::Auto => spacing <= PERTURB_SPACING,
        BackendChoice::Perturb => true,
        BackendChoice::F64 => false,
    };
    let (backend, backend_name): (Box<dyn FractalBackend>, &'static str) = if use_perturb {
        // Perturbation is degree-2 only (the deep-zoom tier, out of multibrot scope);
        // callers with degree != 2 always force `BackendChoice::F64` at these shallow
        // root/descend scales, so this arm is never reached with a higher degree.
        let pb = PerturbationBackend::new(center_re, center_im, maxiter, bailout, prec, trap);
        (Box::new(pb), "PERT")
    } else {
        (Box::new(F64Backend::new_degree(maxiter, bailout, trap, degree)), "F64")
    };
    let buf = render::iterate_samples(&*backend, &frame, ss);
    MandelPanel {
        buf,
        backend_name,
        spacing,
    }
}

/// Render + shade a base-scale Julia panel for the parameter `c` (whole set,
/// center 0, width [`JULIA_WIDTH`]). Always f64 — a base-scale Julia is shallow.
#[allow(clippy::too_many_arguments)]
pub fn render_julia_panel(
    c_f64: Complex<f64>,
    julia_maxiter: u32,
    bailout: f64,
    trap: Trap,
    panel_w: u32,
    panel_h: u32,
    ss: u32,
    palette: &Palette,
    params: &ColorParams,
) -> RgbImage {
    let backend = JuliaBackend::new(c_f64, julia_maxiter, bailout, trap);
    let frame = Frame {
        center: Complex::new(0.0, 0.0),
        frame_width: JULIA_WIDTH,
        out_width: panel_w,
        out_height: panel_h,
    };
    let buf = render::iterate_samples(&backend, &frame, ss);
    render::shade_and_downsample(
        &buf.samples,
        panel_w,
        panel_h,
        ss,
        palette,
        params,
        frame.pixel_size(),
    )
}

/// Draw a white circle (radius `r` px) at `(cx, cy)` with a 1px dark halo on
/// each side for legibility over light regions. `r` marks the next frame's
/// footprint inside the current panel.
pub fn draw_circle(img: &mut RgbImage, cx: f64, cy: f64, r: f64) {
    let w = img.width() as i64;
    let h = img.height() as i64;
    let x0 = ((cx - r - 2.0).floor() as i64).max(0);
    let x1 = ((cx + r + 2.0).ceil() as i64).min(w - 1);
    let y0 = ((cy - r - 2.0).floor() as i64).max(0);
    let y1 = ((cy + r + 2.0).ceil() as i64).min(h - 1);
    let dark = Rgb([0u8, 0, 0]);
    let white = Rgb([255u8, 255, 255]);
    // Halo first (wider), then the white ring inside it.
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cx).powi(2)) + ((y as f64 - cy).powi(2))).sqrt();
            if (d - r).abs() <= 2.0 {
                img.put_pixel(x as u32, y as u32, dark);
            }
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (((x as f64 - cx).powi(2)) + ((y as f64 - cy).powi(2))).sqrt();
            if (d - r).abs() <= 1.0 {
                img.put_pixel(x as u32, y as u32, white);
            }
        }
    }
}

/// Compose the tall filmstrip: one row per level, `Mandelbrot | Julia`.
pub fn compose_strip(
    mandel: &[RgbImage],
    julia: &[RgbImage],
    panel_w: u32,
    panel_h: u32,
) -> RgbImage {
    let n = mandel.len() as u32;
    let width = 2 * panel_w + GAP_H;
    let height = n * panel_h + n.saturating_sub(1) * GAP_V;
    let mut strip = RgbImage::from_pixel(width, height, Rgb(STRIP_BG));
    for i in 0..mandel.len() {
        let y0 = i as u32 * (panel_h + GAP_V);
        blit(&mut strip, &mandel[i], 0, y0);
        blit(&mut strip, &julia[i], panel_w + GAP_H, y0);
    }
    strip
}

/// Compose a single-column strip of just the Mandelbrot panels (no Julia column).
/// The default for the automated drives now that Julia is opt-in: a Mandelbrot
/// region being good implies its Julia is good, so the parallel render is pure
/// cost. [`compose_strip`] still builds the two-column layout under `--with-julia`.
pub fn compose_strip_single(mandel: &[RgbImage], panel_w: u32, panel_h: u32) -> RgbImage {
    let n = mandel.len() as u32;
    let height = n * panel_h + n.saturating_sub(1) * GAP_V;
    let mut strip = RgbImage::from_pixel(panel_w, height, Rgb(STRIP_BG));
    for (i, p) in mandel.iter().enumerate() {
        blit(&mut strip, p, 0, i as u32 * (panel_h + GAP_V));
    }
    strip
}

/// Paste `src` into `dst` at `(x0, y0)`.
pub fn blit(dst: &mut RgbImage, src: &RgbImage, x0: u32, y0: u32) {
    for (sx, sy, px) in src.enumerate_pixels() {
        let (dx, dy) = (x0 + sx, y0 + sy);
        if dx < dst.width() && dy < dst.height() {
            dst.put_pixel(dx, dy, *px);
        }
    }
}

/// `<stem>_panels/` directory beside the strip output.
pub fn panels_dir_for(strip: &Path) -> PathBuf {
    let stem = strip
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("probe");
    let dir = format!("{stem}_panels");
    match strip.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(dir),
        _ => PathBuf::from(dir),
    }
}

/// Forward-slash path string for the JSON (portable, copy-pasteable).
pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Format a finite f64 in scientific form for JSON; non-finite → `null`.
pub fn jf(x: f64) -> String {
    if x.is_finite() {
        format!("{x:e}")
    } else {
        "null".into()
    }
}

/// JSON-escape a string (only `"`/`\` are possible in our decimals; defensive).
pub fn js(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Fraction of a sorted slice `<= v` (the empirical percentile of `v`).
///
/// Shared by the location probes and the `generate` subcommand (the corpus
/// descriptor is a keeper-only feature there). Lifted here from the test-only
/// `location_probe` module so a production subcommand can reuse it unchanged.
pub fn frac_le(sorted: &[f64], v: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    // partition_point: count of elements <= v.
    let c = sorted.partition_point(|&x| x <= v);
    c as f64 / sorted.len() as f64
}

/// Pull one named colormap's stops out of `clean_colormaps.json` without serde.
/// The stops array is `[[pos,[r,g,b]], ...]`; we bracket-match the array span,
/// flatten every numeric token in it, and chunk by 4 → `(pos, [r,g,b])`.
///
/// Shared by the location probes and the `generate` subcommand (one fixed
/// held-out colormap for the structure-finding previews). Lifted here from the
/// test-only `location_probe` module so a production subcommand can reuse it.
pub fn load_colormap(text: &str, name: &str) -> Result<Vec<(f64, [u8; 3])>, String> {
    let needle = format!("\"{name}\"");
    let np = text
        .find(&needle)
        .ok_or_else(|| format!("colormap '{name}' not found"))?;
    let sp = text[np..]
        .find("\"stops\"")
        .map(|p| p + np)
        .ok_or_else(|| format!("colormap '{name}': no stops"))?;
    let open = text[sp..]
        .find('[')
        .map(|p| p + sp)
        .ok_or("stops: no '['")?;
    // Bracket-match to find the end of the stops array.
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut end = open;
    for (k, &b) in bytes[open..].iter().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = open + k;
                    break;
                }
            }
            _ => {}
        }
    }
    let span = &text[open + 1..end];
    // Flatten numeric tokens.
    let mut nums: Vec<f64> = Vec::new();
    for tok in span.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')) {
        if tok.is_empty() {
            continue;
        }
        if let Ok(x) = tok.parse::<f64>() {
            nums.push(x);
        }
    }
    if nums.is_empty() || nums.len() % 4 != 0 {
        return Err(format!(
            "colormap '{name}': parsed {} numbers (not a multiple of 4)",
            nums.len()
        ));
    }
    let mut stops = Vec::with_capacity(nums.len() / 4);
    for ch in nums.chunks_exact(4) {
        let pos = ch[0];
        let rgb = [ch[1] as u8, ch[2] as u8, ch[3] as u8];
        stops.push((pos, rgb));
    }
    Ok(stops)
}

/// Read the inline `mirror_needed` flag for a named colormap in
/// `clean_colormaps.json` (the SEQUENTIAL/pre-mirror classification written by
/// `classify.py`). Substring-scoped to the named entry's object — `mirror_needed`
/// follows `name`/`stops` within the same `{...}`, ahead of the next entry's
/// `"name"`. Absent / non-sequential / unparsable → `false` (safe: no mirror).
/// The companion to [`load_colormap`]; both back the `generate` and
/// `reject-corridor` colormap previews.
pub fn colormap_mirror_needed(text: &str, name: &str) -> bool {
    let needle = format!("\"{name}\"");
    let np = match text.find(&needle) {
        Some(p) => p + needle.len(),
        None => return false,
    };
    // Bound the search to this entry: the next `"name"` key starts the next object.
    let bound = text[np..]
        .find("\"name\"")
        .map(|p| np + p)
        .unwrap_or(text.len());
    let region = &text[np..bound];
    match region.find("\"mirror_needed\"") {
        Some(mp) => {
            let tail = &region[mp + "\"mirror_needed\"".len()..];
            // First boolean token after the colon.
            let t = tail.find("true");
            let f = tail.find("false");
            match (t, f) {
                (Some(ti), Some(fi)) => ti < fi,
                (Some(_), None) => true,
                _ => false,
            }
        }
        None => false,
    }
}

#[cfg(test)]
mod mirror_flag_tests {
    use super::colormap_mirror_needed;

    const LIB: &str = r#"[
        {"name": "seq", "source": "x", "stops": [[0.0,[0,0,4]]], "cycle": "sequential", "mirror_needed": true},
        {"name": "cyc", "source": "y", "stops": [[0.0,[1,2,3]]], "cycle": "cyclic", "mirror_needed": false}
    ]"#;

    #[test]
    fn reads_inline_mirror_needed_scoped_to_entry() {
        assert!(colormap_mirror_needed(LIB, "seq"));
        assert!(!colormap_mirror_needed(LIB, "cyc"));
        // Unknown / absent -> false (safe: no mirror).
        assert!(!colormap_mirror_needed(LIB, "missing"));
    }
}
