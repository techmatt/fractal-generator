//! `root_field` — the durable 8192×8192 smooth-iteration field that seeds the
//! `guided-descend` rev4 root mixture (the 8k-field root, Part A1).
//!
//! One **continuous (smooth) iteration** f32 field over the full-set bounding box
//! is rendered **once** and cached under `data/root_field/` (raw f32 blob + a JSON
//! header); every run reuses it. Interior (non-escaped) pixels are stored as
//! `f32::NAN` — that *is* the black mask. All window statistics are taken over the
//! **escaped** pixels only (interior pins at maxiter and would corrupt the moments).
//!
//! ## The [`RootField::score`] seam — a one-function swap
//!
//! The window scan ([`RootField::passing_windows`]) calls [`RootField::score`] over
//! each sub-rectangle. The **hand criterion lives entirely inside `score`**: a
//! window passes iff `black ≤ black_max` **and** escaped smooth-iteration
//! `mean ∈ [lo, hi]` **and** escaped `variance ≥ floor` (two-sided by construction —
//! low mean = empty exterior, high mean = set-dominated, variance floor = not flat).
//! Swapping this for a learned "cool regions" heat field later (a CNN logit + one
//! threshold) is a single-function change, not a refactor — keep the seam clean.
//!
//! The field is rendered in horizontal **strips** rather than one buffer: at 8192²
//! a single ss1 `SampleBuffer` is ~3.2 GB of `PixelSample`; per-strip we keep only
//! the f32 smooth-iter projection (268 MB total) and drop each strip's samples.

use std::path::{Path, PathBuf};

use astro_float::BigFloat;
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::Trap;
use crate::cli::BackendChoice;
use crate::{hp, probe};

/// Full-set bounding box (re∈[-2.5,1.0], im∈[-1.75,1.75]) — square 3.5×3.5 so the
/// 8192×8192 grid keeps pixels square. Chosen to contain the whole connected set
/// plus a margin of exterior on every side.
pub const RE_LO: f64 = -2.5;
pub const RE_HI: f64 = 1.0;
pub const IM_LO: f64 = -1.75;
pub const IM_HI: f64 = 1.75;

/// Field resolution. 8192² — at root-zoom-8k 0.10 a depth-1 window is ~234×132 px,
/// adequately resolved. (fw < ~0.03 would want a denser cache; rev4 stays at 0.10.)
pub const FIELD_W: u32 = 8192;
pub const FIELD_H: u32 = 8192;

/// Strip height (rows) for the build pass — bounds peak `PixelSample` memory to
/// ~`FIELD_W × STRIP_H × 48 B` (≈ 400 MB at 1024).
const STRIP_H: u32 = 1024;

/// Sentinel marking an interior (non-escaped) pixel in the f32 field.
const INTERIOR: f32 = f32::NAN;

/// Where the durable cache lives (gitignored under `data/*`).
pub const CACHE_DIR: &str = "data/root_field";

/// The cached smooth-iteration field. `data[y*w + x]` is the smooth iteration count
/// for escaped pixels, [`INTERIOR`] (`NaN`) for interior pixels.
pub struct RootField {
    pub w: usize,
    pub h: usize,
    pub re_lo: f64,
    pub re_hi: f64,
    pub im_lo: f64,
    pub im_hi: f64,
    pub maxiter: u32,
    pub data: Vec<f32>,
}

/// The hand criterion's tunables (exposed as `guided-descend` flags). Replacing the
/// whole criterion with a learned scorer touches only [`RootField::score`].
#[derive(Clone, Copy, Debug)]
pub struct ScoreCfg {
    /// Max interior (NaN) fraction of the window.
    pub black_max: f64,
    /// Escaped smooth-iteration mean must lie in `[mean_lo, mean_hi]`.
    pub mean_lo: f64,
    pub mean_hi: f64,
    /// Escaped smooth-iteration variance floor (not-flat).
    pub var_floor: f64,
}

/// One window's score breakdown (returned by the seam; the scalar `score` is the
/// future swap point — uniform sampling among passers ignores it for now).
#[derive(Clone, Copy)]
pub struct WindowScore {
    pub score: f32,
    pub black: f64,
    pub mean: f64,
    pub var: f64,
}

/// A passing window: the complex-plane center its depth-1 node is placed at, plus
/// the score breakdown (logged, not yet used for weighting).
#[derive(Clone, Copy)]
pub struct PassWindow {
    pub center: Complex<f64>,
    pub score: WindowScore,
}

impl RootField {
    /// Plane coords of pixel center `(px, py)` (row 0 = top = max im).
    pub fn pixel_to_complex(&self, px: f64, py: f64) -> Complex<f64> {
        let fx = (px + 0.5) / self.w as f64;
        let fy = (py + 0.5) / self.h as f64;
        Complex::new(
            self.re_lo + fx * (self.re_hi - self.re_lo),
            self.im_hi - fy * (self.im_hi - self.im_lo),
        )
    }

    /// **The score seam.** Hand criterion over the window `[x0, x0+wpx) × [y0, y0+hpx)`
    /// (clamped to the field). All moments over escaped (non-NaN) pixels only.
    /// Returns `Some(score)` iff the window passes, else `None`.
    pub fn score(&self, x0: usize, y0: usize, wpx: usize, hpx: usize, cfg: &ScoreCfg) -> Option<WindowScore> {
        let x1 = (x0 + wpx).min(self.w);
        let y1 = (y0 + hpx).min(self.h);
        if x1 <= x0 || y1 <= y0 {
            return None;
        }
        let total = ((x1 - x0) * (y1 - y0)) as f64;
        let (mut n, mut sum, mut sumsq) = (0.0f64, 0.0f64, 0.0f64);
        for y in y0..y1 {
            let row = y * self.w;
            for x in x0..x1 {
                let v = self.data[row + x];
                if v.is_nan() {
                    continue;
                }
                let v = v as f64;
                n += 1.0;
                sum += v;
                sumsq += v * v;
            }
        }
        let black = (total - n) / total;
        if n < 1.0 || black > cfg.black_max {
            return None;
        }
        let mean = sum / n;
        let var = (sumsq / n - mean * mean).max(0.0);
        if mean < cfg.mean_lo || mean > cfg.mean_hi || var < cfg.var_floor {
            return None;
        }
        // Bootstrap scalar: variance (the "interestingness" proxy). Swap point for a
        // learned logit — sampling currently ignores it (uniform among passers).
        Some(WindowScore { score: var as f32, black, mean, var })
    }

    /// Scan `win_w × win_h` windows on a `stride` grid, keep every passer. The
    /// scan is the only caller of [`RootField::score`] (the swap seam). Window
    /// origins are enumerated then scored in parallel.
    pub fn passing_windows(&self, win_w: usize, win_h: usize, stride: usize, cfg: &ScoreCfg) -> Vec<PassWindow> {
        let stride = stride.max(1);
        let mut origins: Vec<(usize, usize)> = Vec::new();
        let y_last = self.h.saturating_sub(win_h);
        let x_last = self.w.saturating_sub(win_w);
        let mut y = 0;
        while y <= y_last {
            let mut x = 0;
            while x <= x_last {
                origins.push((x, y));
                x += stride;
            }
            y += stride;
        }
        origins
            .par_iter()
            .filter_map(|&(x0, y0)| {
                self.score(x0, y0, win_w, win_h, cfg).map(|s| PassWindow {
                    center: self.pixel_to_complex((x0 + win_w / 2) as f64, (y0 + win_h / 2) as f64),
                    score: s,
                })
            })
            .collect()
    }

    /// Load the cached field if a matching header exists, else render + cache it.
    /// Matching = same dims, maxiter, **and degree** (bailout is recorded but not
    /// keyed — the smooth field is insensitive to it once large). `degree` selects
    /// the `z^d + c` recurrence and the origin-symmetric bounding box for `d ≥ 3`
    /// (see [`degree_bbox`]); `d = 2` keeps the exact historical Mandelbrot box.
    pub fn load_or_build(maxiter: u32, bailout: f64, trap: Trap, degree: u32) -> Result<RootField, String> {
        let (blob, header) = cache_paths(FIELD_W, FIELD_H, maxiter, degree);
        if let Some(f) = try_load(&blob, &header, maxiter, degree)? {
            eprintln!(
                "  root_field: loaded cache {} ({}x{}, maxiter {}, degree {})",
                blob.display(), f.w, f.h, f.maxiter, degree
            );
            return Ok(f);
        }
        let f = build(maxiter, bailout, trap, degree);
        write_cache(&f, &blob, &header, bailout, degree)?;
        Ok(f)
    }
}

/// Per-degree root bounding box. `d = 2` keeps the exact historical Mandelbrot box
/// (asymmetric, main-cardioid framed) so the degree-2 field is byte-identical.
/// Multibrot sets (`d ≥ 3`) are origin-symmetric, so use an origin-centered square
/// of half-width `2^(1/(d−1))·MARGIN` — `2^(1/(d−1))` is the `|c|` escape bound (the
/// radius that contains the connected set), and `MARGIN` adds an exterior frame.
fn degree_bbox(degree: u32) -> (f64, f64, f64, f64) {
    if degree == 2 {
        (RE_LO, RE_HI, IM_LO, IM_HI)
    } else {
        const MARGIN: f64 = 1.2;
        let r = 2f64.powf(1.0 / (degree as f64 - 1.0)) * MARGIN;
        (-r, r, -r, r)
    }
}

/// `(blob, header)` cache paths keyed by dims + maxiter + degree. Degree 2 keeps
/// the historical suffix-free filename so existing caches still load; `d ≥ 3` adds
/// a `_d{degree}` suffix (its box differs, so it must never collide with `d = 2`).
fn cache_paths(w: u32, h: u32, maxiter: u32, degree: u32) -> (PathBuf, PathBuf) {
    let stem = if degree == 2 {
        format!("field_{w}x{h}_m{maxiter}")
    } else {
        format!("field_{w}x{h}_m{maxiter}_d{degree}")
    };
    let dir = Path::new(CACHE_DIR);
    (dir.join(format!("{stem}.f32")), dir.join(format!("{stem}.json")))
}

/// Attempt to load a cached field; `Ok(None)` if absent or header mismatch.
/// Keyed on dims, maxiter, and degree. The bbox is read back from the header (so a
/// multibrot field restores its origin-square box); legacy degree-2 headers lack a
/// `"degree"` key (defaults to 2) but always carry the bbox floats.
fn try_load(blob: &Path, header: &Path, maxiter: u32, degree: u32) -> Result<Option<RootField>, String> {
    if !blob.exists() || !header.exists() {
        return Ok(None);
    }
    let txt = std::fs::read_to_string(header).map_err(|e| format!("read {}: {e}", header.display()))?;
    let w = json_u(&txt, "w").unwrap_or(0) as usize;
    let h = json_u(&txt, "h").unwrap_or(0) as usize;
    let m = json_u(&txt, "maxiter").unwrap_or(0) as u32;
    let deg = json_u(&txt, "degree").unwrap_or(2) as u32;
    if w != FIELD_W as usize || h != FIELD_H as usize || m != maxiter || deg != degree {
        return Ok(None);
    }
    let bytes = std::fs::read(blob).map_err(|e| format!("read {}: {e}", blob.display()))?;
    if bytes.len() != w * h * 4 {
        return Ok(None);
    }
    let mut data = vec![0.0f32; w * h];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        data[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    // Restore the box from the header (fallback to the degree's canonical box if a
    // field float is somehow missing — keeps old headers robust).
    let (be_lo, be_hi, bi_lo, bi_hi) = degree_bbox(degree);
    Ok(Some(RootField {
        w, h,
        re_lo: json_f(&txt, "re_lo").unwrap_or(be_lo),
        re_hi: json_f(&txt, "re_hi").unwrap_or(be_hi),
        im_lo: json_f(&txt, "im_lo").unwrap_or(bi_lo),
        im_hi: json_f(&txt, "im_hi").unwrap_or(bi_hi),
        maxiter, data,
    }))
}

/// Persist the field (raw little-endian f32 blob + JSON header).
fn write_cache(f: &RootField, blob: &Path, header: &Path, bailout: f64, degree: u32) -> Result<(), String> {
    crate::ensure_parent_dir(blob)?;
    let mut bytes = Vec::with_capacity(f.data.len() * 4);
    for &v in &f.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(blob, &bytes).map_err(|e| format!("write {}: {e}", blob.display()))?;
    let hdr = format!(
        "{{\n  \"w\": {}, \"h\": {}, \"maxiter\": {}, \"degree\": {}, \"bailout\": {},\n  \
         \"re_lo\": {}, \"re_hi\": {}, \"im_lo\": {}, \"im_hi\": {},\n  \
         \"interior_sentinel\": \"NaN\", \"format\": \"row-major little-endian f32 smooth_iter\"\n}}\n",
        f.w, f.h, f.maxiter, degree, bailout, f.re_lo, f.re_hi, f.im_lo, f.im_hi,
    );
    std::fs::write(header, hdr).map_err(|e| format!("write {}: {e}", header.display()))?;
    eprintln!("  root_field: cached {} ({:.0} MB)", blob.display(), bytes.len() as f64 / 1e6);
    Ok(())
}

/// Render the full field in horizontal strips, projecting each strip's escape
/// result to f32 (NaN = interior) and dropping its samples. `degree` selects the
/// `z^d + c` recurrence and the [`degree_bbox`] (origin-square for `d ≥ 3`).
fn build(maxiter: u32, bailout: f64, trap: Trap, degree: u32) -> RootField {
    let w = FIELD_W as usize;
    let h = FIELD_H as usize;
    let (re_lo, re_hi, im_lo, im_hi) = degree_bbox(degree);
    let mut data = vec![INTERIOR; w * h];
    let full_h_plane = im_hi - im_lo;
    let t0 = std::time::Instant::now();
    eprintln!(
        "  root_field: building {FIELD_W}x{FIELD_H} smooth field (maxiter {maxiter}, degree {degree}, \
         box re[{re_lo},{re_hi}] im[{im_lo},{im_hi}]) in strips ..."
    );

    let mut y0 = 0u32;
    while y0 < FIELD_H {
        let sh = STRIP_H.min(FIELD_H - y0);
        // Strip center: full width, sub-height; im at the strip's vertical midpoint.
        let center_im = im_hi - ((y0 as f64 + sh as f64 / 2.0) / FIELD_H as f64) * full_h_plane;
        let center = Complex::new((re_lo + re_hi) / 2.0, center_im);
        let frame_width = re_hi - re_lo;
        let prec = hp::prec_bits(FIELD_W, frame_width);
        let cre = BigFloat::from_f64(center.re, prec);
        let cim = BigFloat::from_f64(center.im, prec);
        let panel = probe::render_mandel_panel(
            &cre, &cim, center, frame_width, FIELD_W, sh, 1, maxiter, bailout, degree, prec, trap,
            BackendChoice::F64,
        );
        let base = y0 as usize * w;
        for (i, s) in panel.buf.samples.iter().enumerate() {
            data[base + i] = if s.escaped { s.smooth_iter as f32 } else { INTERIOR };
        }
        eprintln!(
            "    strip rows [{},{}) done ({:.1}s)",
            y0, y0 + sh, t0.elapsed().as_secs_f64()
        );
        y0 += sh;
    }
    eprintln!("  root_field: built in {:.1}s", t0.elapsed().as_secs_f64());
    RootField { w, h, re_lo, re_hi, im_lo, im_hi, maxiter, data }
}

/// Minimal unsigned-int field reader for the tiny hand-rolled JSON header
/// (`"key": 123`). Returns the integer part; good enough for w/h/maxiter.
fn json_u(txt: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let i = txt.find(&pat)? + pat.len();
    let rest = &txt[i..];
    let colon = rest.find(':')? + 1;
    let rest = &rest[colon..];
    let digits: String = rest.trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Minimal signed-float field reader for the header (`"key": -1.75`). Consumes the
/// float token after the colon (sign / digits / `.` / exponent). Used to restore
/// the per-degree bounding box on load.
fn json_f(txt: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\"");
    let i = txt.find(&pat)? + pat.len();
    let rest = &txt[i..];
    let colon = rest.find(':')? + 1;
    let tok: String = rest[colon..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
        .collect();
    tok.parse().ok()
}
