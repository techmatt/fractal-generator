//! Beautiful rendering modes — the decoupled `iterate → field → shade → palette`
//! pipeline (v1: single fields + generic normal-map emboss).
//!
//! This is **Stage-2 substrate** (the coloring-method axis the aesthetic sweep
//! will enumerate) and is **purely additive**: it lives entirely beside the
//! settled location-profile path and is reached only when `render-one` is given a
//! non-default `--coloring`. The default (no `--coloring`, or `--coloring '{}'`)
//! routes to the untouched location-profile render, so its byte-identity (which
//! the v5 cache and the frozen location classifier depend on) is preserved by
//! construction — see [`ColoringParams::is_location_profile`] and
//! `render_one::run_render_one`.
//!
//! ## Why this is a separate pipeline (not [`crate::coloring::shade`])
//!
//! The location-profile coloring is a **per-pixel-pure** map ([`crate::coloring`])
//! — re-coloring never re-iterates, which is the whole project's separability
//! win. The beautiful path is **not** per-pixel pure: the doc's percentile-stretch
//! (and histogram-equalization) normalize each field against the **whole frame's**
//! value distribution, a global reduction. So this pipeline shades up front into a
//! linear-RGB supersample buffer and hands it to
//! [`crate::render::downsample_linear_filtered`] (the linear-buffer twin of the
//! smooth path's reconstruction filter), rather than threading through
//! `shade_and_downsample`.
//!
//! ## Architecture (doc §2)
//!
//! ```text
//! iterate() -> OrbitAccum         one pass; accumulates every channel a field/shade can read
//! field()   -> scalar             the "style": smooth | stripe | tia | curvature | trap_*
//! (percentile-stretch / histeq + transform + gamma) -> gray in [0,1]
//! shade()   -> gray               optional Lambert emboss (normal_map), multiplies over the field
//! palette   -> linear RGB         last, fully swappable (palette_cycles / palette_offset)
//! ```
//!
//! [`OrbitAccum`] holds the **union** of per-orbit channels, and
//! [`OrbitAccum::field`] is a pure reduction to any one field — so a future
//! Stage-2 sweep can iterate once and read many fields off one buffer. `render-one`
//! itself renders a single field, so it reduces each accumulator to one scalar
//! immediately (the per-subpixel store stays small).
//!
//! ## Conventions vs. the prompt
//!
//! The prompt asks for a "serde `ColoringParams`". The project deliberately avoids
//! serde (CLAUDE.md — JSON logs are hand-rolled); this honors the *intent* (JSON
//! round-tripping so params travel into provenance / the sweep can enumerate them)
//! with hand-rolled [`ColoringParams::to_json`] / [`from_json`], matching the
//! `jsonl` module's tolerant readers. Omitted JSON keys fall back to defaults.

use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::Trap;
use crate::jsonl;
use crate::palette::Palette;
use crate::render::{DownsampleFilter, Frame};

/// Large bailout for the beautiful profiles: `2^16`. Stripe / TIA / normal-map
/// assume `|z| ≫ |c|` at escape for a clean escape angle (doc §3). The
/// location-profile default keeps the low bailout (`1e6`).
pub const BEAUTIFUL_BAILOUT: f64 = 65536.0; // 2^16

// ===========================================================================
// Enums
// ===========================================================================

/// The coloring method / "style" — the field stage (doc §4). Each maps an orbit
/// to a scalar, then percentile-stretch + transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Field {
    /// `nu = n + 1 - log2(log|z|)` — classic smooth escape time. Exterior only.
    Smooth,
    /// Mean over `n ≥ skip` of `0.5 + 0.5·sin(s·arg z)`; last term lerped by the
    /// fractional iteration (deband). Exterior only. Reads clean at density 3–5.
    Stripe,
    /// Triangle-inequality average (TIA); averaged like stripe. Exterior only.
    Tia,
    /// Mean of `|arg((zₙ−zₙ₋₁)/(zₙ₋₁−zₙ₋₂))|` (niche). Exterior only.
    Curvature,
    /// `min‖z|−r|` over the orbit (niche). Valid for every pixel (fills interior).
    TrapCircle,
    /// Pickover stalks: `min(|Re z|,|Im z|)` over the orbit; pairs with `Log` +
    /// biomorph + normal_map for the lace look. Valid for every pixel.
    TrapCross,
}

impl Field {
    fn as_str(self) -> &'static str {
        match self {
            Field::Smooth => "smooth",
            Field::Stripe => "stripe",
            Field::Tia => "tia",
            Field::Curvature => "curvature",
            Field::TrapCircle => "trap_circle",
            Field::TrapCross => "trap_cross",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "smooth" => Field::Smooth,
            "stripe" => Field::Stripe,
            "tia" => Field::Tia,
            "curvature" => Field::Curvature,
            "trap_circle" => Field::TrapCircle,
            "trap_cross" => Field::TrapCross,
            _ => return Err(format!("unknown field '{s}'")),
        })
    }
}

/// Compression / transfer transform applied to the percentile-normalized field
/// (doc §4). `gamma` is the final power applied after the curve.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Identity.
    Linear,
    /// `√x` — gentle low-band expansion.
    Sqrt,
    /// `ln(1+x)/ln2` — stronger compression of the high tail.
    Log,
    /// Histogram-equalization (rank): every band equal screen area. Replaces the
    /// percentile-stretch with a rank map over the frame's valid values.
    Histeq,
    /// Soft S-curve transfer (`smoothstep`).
    Scurve,
}

impl Transform {
    fn as_str(self) -> &'static str {
        match self {
            Transform::Linear => "linear",
            Transform::Sqrt => "sqrt",
            Transform::Log => "log",
            Transform::Histeq => "histeq",
            Transform::Scurve => "scurve",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "linear" => Transform::Linear,
            "sqrt" => Transform::Sqrt,
            "log" => Transform::Log,
            "histeq" => Transform::Histeq,
            "scurve" => Transform::Scurve,
            _ => return Err(format!("unknown transform '{s}'")),
        })
    }
}

/// Optional lighting layer multiplied over the field (doc §5).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Shade {
    /// No shading (the field passes through unchanged).
    None,
    /// Lambert-slope emboss `u = z/z'` (normalized), `t = (u·light + h)/(1+h)`.
    NormalMap,
}

impl Shade {
    fn as_str(self) -> &'static str {
        match self {
            Shade::None => "none",
            Shade::NormalMap => "normal_map",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "none" => Shade::None,
            "normal_map" => Shade::NormalMap,
            _ => return Err(format!("unknown shade '{s}'")),
        })
    }
}

/// Escape rule (doc §2/§7). `EpsilonCross` is Pickover's biomorph: bail when
/// `|Re z| > B` **or** `|Im z| > B` (instead of `|z| > B`), carving the organic
/// perforations of the lace path. This changes `n`, so it is an iterate-stage knob.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Biomorph {
    /// Standard `|z| > B` escape.
    Off,
    /// `|Re z| > B || |Im z| > B` escape (epsilon cross).
    EpsilonCross,
}

impl Biomorph {
    fn as_str(self) -> &'static str {
        match self {
            Biomorph::Off => "off",
            Biomorph::EpsilonCross => "epsilon_cross",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "off" => Biomorph::Off,
            "epsilon_cross" => Biomorph::EpsilonCross,
            _ => return Err(format!("unknown biomorph '{s}'")),
        })
    }
}

// ===========================================================================
// ColoringParams
// ===========================================================================

/// The full parameterization of the beautiful pipeline. JSON-round-trippable
/// (hand-rolled, see module docs) so it travels into provenance and the Stage-2
/// sweep can enumerate it.
///
/// [`ColoringParams::default`] is the **location profile** sentinel: when params
/// equal the default, `render-one` renders through the settled location-profile
/// path (byte-identity guaranteed) instead of this pipeline. The default's field
/// values mirror that profile (`Smooth`, low bailout, no shade) but are not
/// otherwise load-bearing — the equality check is what routes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColoringParams {
    // --- iterate stage ---
    /// Escape radius `B`. Location profile = `1e6`; beautiful = `2^16`.
    pub bailout_b: f64,
    /// First iteration included in the averaging fields (stripe/tia/curvature).
    pub skip: u32,
    /// Escape rule (standard or epsilon-cross biomorph).
    pub biomorph: Biomorph,
    // --- field stage ---
    /// The coloring method.
    pub field: Field,
    /// Stripe sine density `s` (stripe field).
    pub stripe_density: f64,
    /// Trap radius `r` (trap_circle field).
    pub trap_radius: f64,
    // --- colorize stage ---
    /// Compression / transfer transform.
    pub transform: Transform,
    /// Final power applied after the transform.
    pub gamma: f64,
    /// Optional emboss layer.
    pub shade: Shade,
    /// Light azimuth in radians (normal_map).
    pub light_azimuth: f64,
    /// Light height `h` (normal_map); lower = sharper relief.
    pub light_height: f64,
    /// Gradient cycles across the `[0,1]` field (palette stage).
    pub palette_cycles: f64,
    /// Gradient phase offset in `[0,1)` (palette stage).
    pub palette_offset: f64,
}

impl Default for ColoringParams {
    fn default() -> Self {
        // The location-profile sentinel. `render-one` renders the settled path
        // when params == this; the values mirror that profile but only the
        // equality check is load-bearing.
        ColoringParams {
            bailout_b: 1e6,
            skip: 1,
            biomorph: Biomorph::Off,
            field: Field::Smooth,
            stripe_density: 4.0,
            trap_radius: 1.0,
            transform: Transform::Linear,
            gamma: 1.0,
            shade: Shade::None,
            light_azimuth: std::f64::consts::FRAC_PI_4, // 45°
            light_height: 1.0,
            palette_cycles: 1.0,
            palette_offset: 0.0,
        }
    }
}

impl ColoringParams {
    /// True when these params are the location-profile sentinel (== default).
    /// `render-one` routes these to the settled byte-identical path.
    pub fn is_location_profile(&self) -> bool {
        *self == ColoringParams::default()
    }

    /// A beautiful preset: `2^16` bailout + the chosen field, with a transform
    /// sensible for that field (`Log` for the trap stalk fields, `Sqrt`
    /// otherwise). All other knobs stay at defaults; tune via JSON overrides.
    pub fn beautiful(field: Field) -> Self {
        let transform = match field {
            Field::TrapCircle | Field::TrapCross => Transform::Log,
            _ => Transform::Sqrt,
        };
        ColoringParams {
            bailout_b: BEAUTIFUL_BAILOUT,
            field,
            transform,
            ..ColoringParams::default()
        }
    }

    /// Serialize to a compact JSON object (all fields, stable key order).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"bailout_b\":{},\"skip\":{},\"biomorph\":\"{}\",\"field\":\"{}\",\
             \"stripe_density\":{},\"trap_radius\":{},\"transform\":\"{}\",\"gamma\":{},\
             \"shade\":\"{}\",\"light_azimuth\":{},\"light_height\":{},\
             \"palette_cycles\":{},\"palette_offset\":{}}}",
            self.bailout_b,
            self.skip,
            self.biomorph.as_str(),
            self.field.as_str(),
            self.stripe_density,
            self.trap_radius,
            self.transform.as_str(),
            self.gamma,
            self.shade.as_str(),
            self.light_azimuth,
            self.light_height,
            self.palette_cycles,
            self.palette_offset,
        )
    }

    /// Parse a JSON object; **omitted keys fall back to [`Default`]** (so a partial
    /// `{"field":"stripe"}` is a beautiful-stripe-on-defaults spec). Uses the
    /// tolerant `jsonl` readers (flat object, first-match per key).
    pub fn from_json(s: &str) -> Result<Self, String> {
        let mut p = ColoringParams::default();
        if let Some(v) = jsonl::field_f64(s, "bailout_b") {
            p.bailout_b = v;
        }
        if let Some(v) = jsonl::field_usize(s, "skip") {
            p.skip = v as u32;
        }
        if let Some(v) = jsonl::field_str(s, "biomorph") {
            p.biomorph = Biomorph::parse(&v)?;
        }
        if let Some(v) = jsonl::field_str(s, "field") {
            p.field = Field::parse(&v)?;
        }
        if let Some(v) = jsonl::field_f64(s, "stripe_density") {
            p.stripe_density = v;
        }
        if let Some(v) = jsonl::field_f64(s, "trap_radius") {
            p.trap_radius = v;
        }
        if let Some(v) = jsonl::field_str(s, "transform") {
            p.transform = Transform::parse(&v)?;
        }
        if let Some(v) = jsonl::field_f64(s, "gamma") {
            p.gamma = v;
        }
        if let Some(v) = jsonl::field_str(s, "shade") {
            p.shade = Shade::parse(&v)?;
        }
        if let Some(v) = jsonl::field_f64(s, "light_azimuth") {
            p.light_azimuth = v;
        }
        if let Some(v) = jsonl::field_f64(s, "light_height") {
            p.light_height = v;
        }
        if let Some(v) = jsonl::field_f64(s, "palette_cycles") {
            p.palette_cycles = v;
        }
        if let Some(v) = jsonl::field_f64(s, "palette_offset") {
            p.palette_offset = v;
        }
        Ok(p)
    }
}

// ===========================================================================
// Iterate stage — OrbitAccum
// ===========================================================================

/// The union of per-orbit channels captured in one iteration pass. Reduce to any
/// one field with [`OrbitAccum::field`] (a pure function over the accumulators),
/// so a single pass can serve many fields in a sweep.
///
/// Averaging fields (stripe/tia/curvature) store `(sum, count, last)` so the
/// post-loop deband can lerp the last term by the fractional iteration.
#[derive(Clone, Copy, Debug)]
pub struct OrbitAccum {
    pub escaped: bool,
    /// Smooth iteration count (valid when `escaped`).
    pub smooth: f64,
    /// Final value `z` and derivative `z'` (for normal_map `u = z/z'`).
    pub z: Complex<f64>,
    pub dz: Complex<f64>,
    // Averaging accumulators (n ≥ skip): running sum, count, last term.
    stripe: (f64, u32, f64),
    tia: (f64, u32, f64),
    curv: (f64, u32, f64),
    /// `min‖z|−r|` over the orbit.
    trap_circle_min: f64,
    /// `min(|Re z|,|Im z|)` over the orbit.
    trap_cross_min: f64,
}

/// Deband an averaging accumulator: `result = d·A + (1−d)·A_prev`, where `A` is
/// the mean of all terms and `A_prev` excludes the last (doc §3). `d` is the
/// fractional iteration. `None` if no terms accumulated.
fn deband((sum, count, last): (f64, u32, f64), d: f64) -> Option<f64> {
    match count {
        0 => None,
        1 => Some(sum),
        _ => {
            let a = sum / count as f64;
            let a_prev = (sum - last) / (count - 1) as f64;
            Some(d * a + (1.0 - d) * a_prev)
        }
    }
}

impl OrbitAccum {
    /// Reduce to one field's raw scalar (pre-normalization). `None` for an
    /// exterior-only field on an interior pixel (→ rendered black).
    pub fn field(&self, field: Field) -> Option<f64> {
        // Fractional iteration for the deband lerp (exterior fields only).
        let d = if self.escaped {
            self.smooth.fract()
        } else {
            0.0
        };
        match field {
            Field::Smooth => self.escaped.then_some(self.smooth),
            Field::Stripe => self.escaped.then(|| deband(self.stripe, d)).flatten(),
            Field::Tia => self.escaped.then(|| deband(self.tia, d)).flatten(),
            Field::Curvature => self.escaped.then(|| deband(self.curv, d)).flatten(),
            Field::TrapCircle => Some(self.trap_circle_min),
            Field::TrapCross => Some(self.trap_cross_min),
        }
    }

    /// Normalized emboss vector `u = z/z'`, `|u| = 1` (doc §5). `(0,0)` when `z'`
    /// is zero / non-finite (yields the flat ambient term in normal_map).
    pub fn ushade(&self) -> Complex<f64> {
        let d2 = self.dz.norm_sqr();
        if !d2.is_finite() || d2 == 0.0 {
            return Complex::new(0.0, 0.0);
        }
        let u = self.z / self.dz;
        let un = u.norm();
        if !un.is_finite() || un == 0.0 {
            Complex::new(0.0, 0.0)
        } else {
            u / un
        }
    }
}

/// Iterate one orbit, accumulating the full channel union. `z0` is `0` for
/// Mandelbrot (`c` = pixel) and the pixel for Julia (`c` = fixed parameter); the
/// `z'` recurrence differs accordingly (`dz0 = 0`, `+1` for Mandelbrot; `dz0 = 1`,
/// no `+1` for Julia — doc §0). Order matches [`crate::backend::F64Backend`]: `z'`
/// updates from `zₙ` before `z` advances.
// The orbit-history bindings (`zprev2`, `escaped`) are initialized then
// unconditionally overwritten on the first loop pass before they're read — the
// loop always runs ≥1 iteration. That's the carried-history idiom, not a bug.
#[allow(unused_assignments)]
pub fn iterate_orbit(
    z0: Complex<f64>,
    c: Complex<f64>,
    maxiter: u32,
    params: &ColoringParams,
    julia: bool,
) -> OrbitAccum {
    let b = params.bailout_b;
    let b2 = b * b;
    let skip = params.skip;
    let s_density = params.stripe_density;
    let r = params.trap_radius;
    let cabs = c.norm();

    let mut z = z0;
    let mut dz = if julia {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(0.0, 0.0)
    };
    // Orbit history for curvature: zprev1 = zₙ₋₁, zprev2 = zₙ₋₂.
    let mut zprev1 = z0;
    let mut zprev2 = Complex::new(0.0, 0.0);

    let mut stripe = (0.0f64, 0u32, 0.0f64);
    let mut tia = (0.0f64, 0u32, 0.0f64);
    let mut curv = (0.0f64, 0u32, 0.0f64);
    let mut trap_circle_min = f64::INFINITY;
    let mut trap_cross_min = f64::INFINITY;

    let mut n = 0u32;
    let mut escaped = false;
    let mut smooth = 0.0f64;

    loop {
        // z' first (uses zₙ), then z advances. |zₙ²| feeds tia's lo/hi.
        let dz_next = if julia {
            Complex::new(2.0, 0.0) * z * dz
        } else {
            Complex::new(2.0, 0.0) * z * dz + Complex::new(1.0, 0.0)
        };
        let zn_sq = z * z;
        let zn_sq_abs = zn_sq.norm(); // |zₙ²| = |z_prev²| for tia
        let z_next = zn_sq + c;

        zprev2 = zprev1;
        zprev1 = z;
        z = z_next;
        dz = dz_next;
        n += 1;

        let zabs2 = z.norm_sqr();
        let zabs = zabs2.sqrt();

        // Orbit traps: min over the whole orbit (every n ≥ 1, independent of skip).
        let tc = (zabs - r).abs();
        if tc < trap_circle_min {
            trap_circle_min = tc;
        }
        let tx = z.re.abs().min(z.im.abs());
        if tx < trap_cross_min {
            trap_cross_min = tx;
        }

        // Averaging fields, n ≥ skip.
        if n >= skip {
            // stripe: 0.5 + 0.5·sin(s·arg z)
            let st = 0.5 + 0.5 * (s_density * z.im.atan2(z.re)).sin();
            stripe.0 += st;
            stripe.1 += 1;
            stripe.2 = st;

            // tia: (|z| − lo)/(hi − lo), lo = ‖z_prev²|−|c‖, hi = |z_prev²|+|c|
            let lo = (zn_sq_abs - cabs).abs();
            let hi = zn_sq_abs + cabs;
            let denom = hi - lo;
            let ti = if denom > 1e-300 {
                ((zabs - lo) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            tia.0 += ti;
            tia.1 += 1;
            tia.2 = ti;

            // curvature: |arg((zₙ−zₙ₋₁)/(zₙ₋₁−zₙ₋₂))|, needs three points (n ≥ 2).
            if n >= 2 {
                let num = z - zprev1;
                let den = zprev1 - zprev2;
                if den.norm_sqr() > 1e-300 {
                    let ang = (num / den).arg().abs();
                    curv.0 += ang;
                    curv.1 += 1;
                    curv.2 = ang;
                }
            }
        }

        if n >= maxiter {
            escaped = false;
            break;
        }
        let bail = match params.biomorph {
            Biomorph::Off => zabs2 > b2,
            Biomorph::EpsilonCross => z.re.abs() > b || z.im.abs() > b,
        };
        if bail {
            escaped = true;
            smooth = smooth_value(n, zabs2);
            break;
        }
    }

    OrbitAccum {
        escaped,
        smooth,
        z,
        dz,
        stripe,
        tia,
        curv,
        trap_circle_min,
        trap_cross_min,
    }
}

/// `nu = (n+1) − log2(ln|z|)` — the same smooth formula both production backends
/// use ([`crate::backend`]); the doc's `−log2(log B)` offset is an additive
/// constant the percentile-stretch removes.
#[inline]
fn smooth_value(n: u32, zabs2: f64) -> f64 {
    let log_zn = 0.5 * zabs2.ln();
    if log_zn > 0.0 && log_zn.is_finite() {
        (n + 1) as f64 - log_zn.ln() / std::f64::consts::LN_2
    } else {
        (n + 1) as f64
    }
}

// ===========================================================================
// Colorize stage
// ===========================================================================

/// Percentile clip bounds (low / high) for the field-value stretch. Trims the
/// long tails so the gradient spans the bulk of the distribution.
const PCT_LO: f64 = 0.5;
const PCT_HI: f64 = 99.5;

/// Hermite smoothstep on `[0,1]`.
#[inline]
fn smoothstep01(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Apply the transform curve to a `[0,1]`-normalized value, then `gamma`.
/// (Histeq is handled upstream — the value arrives already rank-equalized.)
#[inline]
fn apply_transform(x: f64, transform: Transform, gamma: f64) -> f64 {
    let y = match transform {
        Transform::Linear | Transform::Histeq => x,
        Transform::Sqrt => x.max(0.0).sqrt(),
        Transform::Log => x.max(0.0).ln_1p() / std::f64::consts::LN_2, // ln(1+x)/ln2: [0,1]→[0,1]
        Transform::Scurve => smoothstep01(x),
    };
    y.clamp(0.0, 1.0).powf(gamma)
}

/// `p`-th percentile (p in [0,100]) of a slice via `select_nth_unstable`. The
/// slice is partially reordered in place (caller owns a scratch copy).
fn percentile(scratch: &mut [f64], p: f64) -> f64 {
    if scratch.is_empty() {
        return 0.0;
    }
    let n = scratch.len();
    let idx = (((p / 100.0) * (n - 1) as f64).round() as usize).min(n - 1);
    let (_, nth, _) = scratch.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *nth
}

/// A per-subpixel reduction of an orbit: the raw field scalar (if valid) and the
/// emboss vector. Kept small so the supersample buffer stays modest.
#[derive(Clone, Copy)]
struct ShadePix {
    value: f64,
    valid: bool,
    ushade: Complex<f64>,
}

/// Render one location through the beautiful pipeline → sRGB image.
///
/// `julia_param = Some(c)` renders a Julia (viewport is the z-plane, `z0 = pixel`);
/// `None` renders the Mandelbrot. Grid-centered supersampling (the rgss/jitter
/// placements are a smooth-path AA study; beautiful v1 uses grid). The trap fields
/// fill the interior; exterior-only fields render interior pixels black.
#[allow(clippy::too_many_arguments)]
pub fn render_beautiful(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    julia_param: Option<Complex<f64>>,
    params: &ColoringParams,
    palette: &Palette,
    filter: DownsampleFilter,
) -> image::RgbImage {
    let s = ss.max(1);
    let sub_w = (frame.out_width * s) as usize;
    let sub_h = (frame.out_height * s) as usize;
    let fw = frame.frame_width;
    let fh = frame.frame_height();
    let sub_w_f = sub_w as f64;
    let sub_h_f = sub_h as f64;
    let center = frame.center;
    let julia = julia_param.is_some();
    let want_shade = params.shade == Shade::NormalMap;

    // --- iterate + reduce to (value, valid, ushade) per subpixel ---
    let rows: Vec<Vec<ShadePix>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                // Mandelbrot: z0 = 0, c = pixel. Julia: z0 = pixel, c = param.
                let (z0, c) = match julia_param {
                    Some(p) => (pixel, p),
                    None => (Complex::new(0.0, 0.0), pixel),
                };
                let acc = iterate_orbit(z0, c, maxiter, params, julia);
                let value = acc.field(params.field);
                row.push(ShadePix {
                    value: value.unwrap_or(0.0),
                    valid: value.is_some(),
                    ushade: if want_shade {
                        acc.ushade()
                    } else {
                        Complex::new(0.0, 0.0)
                    },
                });
            }
            row
        })
        .collect();
    let pix: Vec<ShadePix> = rows.into_iter().flatten().collect();

    // --- global normalization over valid values ---
    let mut valids: Vec<f64> = pix
        .iter()
        .filter(|p| p.valid && p.value.is_finite())
        .map(|p| p.value)
        .collect();

    // Histeq builds a sorted table for rank lookup; the stretch transforms use
    // percentile bounds. Both are global reductions over the frame's valid values.
    let histeq = params.transform == Transform::Histeq;
    let (lo, span, sorted) = if histeq {
        valids.sort_unstable_by(f64::total_cmp);
        (0.0, 1.0, Some(valids))
    } else {
        let lo = percentile(&mut valids, PCT_LO);
        let hi = percentile(&mut valids, PCT_HI);
        let span = if hi > lo { hi - lo } else { 1.0 };
        (lo, span, None)
    };

    let (saz, caz) = params.light_azimuth.sin_cos();
    let h = params.light_height;
    let cycles = params.palette_cycles;
    let offset = params.palette_offset;
    let transform = params.transform;
    let gamma = params.gamma;

    // --- shade each subpixel to linear RGB ---
    let linear: Vec<[f64; 3]> = pix
        .par_iter()
        .map(|p| {
            if !p.valid || !p.value.is_finite() {
                return [0.0, 0.0, 0.0];
            }
            // Normalize: histeq → rank fraction; else percentile-stretch.
            let x = match &sorted {
                Some(tab) => {
                    if tab.len() <= 1 {
                        0.0
                    } else {
                        let rank = tab.partition_point(|&v| v < p.value);
                        rank as f64 / (tab.len() - 1) as f64
                    }
                }
                None => ((p.value - lo) / span).clamp(0.0, 1.0),
            };
            let mut gray = apply_transform(x, transform, gamma);

            // normal_map emboss multiplied over the field.
            if want_shade {
                let u = p.ushade;
                let dot = u.re * caz + u.im * saz;
                let t = ((dot + h) / (1.0 + h)).clamp(0.0, 1.0);
                gray *= t;
            }

            let tt = (gray * cycles + offset).rem_euclid(1.0);
            palette.lookup_linear(tt)
        })
        .collect();

    crate::render::downsample_linear_filtered(
        &linear,
        frame.out_width,
        frame.out_height,
        s,
        filter,
    )
}

/// Convenience: the trap shape the location-profile path uses (point trap at the
/// origin). Beautiful fields don't read [`Trap`] (they accumulate their own trap
/// minima), but callers that share setup may want a default.
pub fn default_trap() -> Trap {
    Trap {
        shape: crate::backend::TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_location_profile() {
        assert!(ColoringParams::default().is_location_profile());
        assert!(!ColoringParams::beautiful(Field::TrapCross).is_location_profile());
    }

    #[test]
    fn json_roundtrip() {
        let p = ColoringParams::beautiful(Field::Stripe);
        let p2 = ColoringParams::from_json(&p.to_json()).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn json_partial_falls_back_to_default() {
        // Omitted keys default; only `field` overridden.
        let p = ColoringParams::from_json("{\"field\":\"tia\"}").unwrap();
        assert_eq!(p.field, Field::Tia);
        assert_eq!(p.bailout_b, ColoringParams::default().bailout_b);
        assert_eq!(p.transform, ColoringParams::default().transform);
    }

    #[test]
    fn empty_json_is_default() {
        let p = ColoringParams::from_json("{}").unwrap();
        assert!(p.is_location_profile());
    }

    #[test]
    fn unknown_enum_errors() {
        assert!(ColoringParams::from_json("{\"field\":\"bogus\"}").is_err());
    }

    /// Mandelbrot z'₀ = 0 with the `+1` recurrence; Julia z'₀ = 1 without it.
    /// A smooth-field escape should produce a finite smooth value and a usable
    /// emboss vector for both fractal types.
    #[test]
    fn iterate_both_fractal_types() {
        let p = ColoringParams::beautiful(Field::Smooth);
        // Mandelbrot exterior point.
        let m = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            500,
            &p,
            false,
        );
        assert!(m.escaped && m.smooth.is_finite());
        // Julia exterior point (c = -0.8 + 0.156i, z0 far out).
        let j = iterate_orbit(
            Complex::new(2.0, 2.0),
            Complex::new(-0.8, 0.156),
            500,
            &p,
            true,
        );
        assert!(j.escaped && j.smooth.is_finite());
        assert!(j.ushade().norm() <= 1.0 + 1e-9);
    }

    /// Trap fields are valid for interior pixels (fill), exterior fields are not.
    #[test]
    fn trap_fields_fill_interior() {
        let p = ColoringParams::beautiful(Field::TrapCross);
        // Deep interior of the main cardioid stays bounded → interior pixel.
        let acc = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.0),
            500,
            &p,
            false,
        );
        assert!(!acc.escaped);
        assert!(acc.field(Field::TrapCross).is_some());
        assert!(acc.field(Field::Smooth).is_none());
    }
}
