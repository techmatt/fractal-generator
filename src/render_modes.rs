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
    /// Mean over `n ≥ skip` of `0.5 + 0.5·sin(s·arg z)`, last term lerped by the
    /// bailout-normalized fractional iteration (deband). Exterior only. Reads clean
    /// at density 3–5.
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

/// Field-modulates-field combine op (v2 composite, doc §3). Both operands are
/// already independently normalized to `[0,1]`; every op maps `[0,1]²→[0,1]`.
/// `Multiply` is the named "lace" target / default.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Combine {
    /// `b·t` — the lace path: texture's flat (→0) regions darken the base.
    Multiply,
    /// `1−(1−b)(1−t)` — inverse-multiply; texture's bright regions lift the base.
    Screen,
    /// Multiply in the shadows, screen in the highlights (pivot at `b=0.5`).
    Overlay,
    /// `min(b,t)` — texture clamps the base from above.
    Min,
}

impl Combine {
    fn as_str(self) -> &'static str {
        match self {
            Combine::Multiply => "multiply",
            Combine::Screen => "screen",
            Combine::Overlay => "overlay",
            Combine::Min => "min",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "multiply" => Combine::Multiply,
            "screen" => Combine::Screen,
            "overlay" => Combine::Overlay,
            "min" => Combine::Min,
            _ => return Err(format!("unknown combine '{s}'")),
        })
    }
    /// Combine two `[0,1]` operands → `[0,1]`.
    #[inline]
    fn apply(self, b: f64, t: f64) -> f64 {
        match self {
            Combine::Multiply => b * t,
            Combine::Screen => 1.0 - (1.0 - b) * (1.0 - t),
            Combine::Overlay => {
                if b < 0.5 {
                    2.0 * b * t
                } else {
                    1.0 - 2.0 * (1.0 - b) * (1.0 - t)
                }
            }
            Combine::Min => b.min(t),
        }
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
    // --- v2 composite (field-modulates-field) ---
    /// Optional **texture** field that modulates the base ([`Self::field`]).
    /// `None` → the v1 single-field path (byte-identical, separate branch). When
    /// `Some`, base and texture each normalize **independently** to `[0,1]`, then
    /// [`Self::combine`] merges them and [`Self::texture_weight`] lerps base↔op.
    ///
    /// Field-shape params (`stripe_density`/`trap_radius`) are **shared** with the
    /// base — one orbit pass accumulates a single stripe/trap channel, so a
    /// stripe-texture-over-stripe-base with differing densities is not expressible
    /// (none of the v2 targets need it; the texture schema carries no shape knob).
    pub texture_field: Option<Field>,
    /// Texture's own transform (independent normalization).
    pub texture_transform: Transform,
    /// Texture's own post-transform gamma.
    pub texture_gamma: f64,
    /// Field-modulates-field combine op (composite only).
    pub combine: Combine,
    /// Texture-strength dial: `lerp(base_n, combine(base_n,tex_n), w)`. `0` → base
    /// only; `1` → full combine op (composite only).
    pub texture_weight: f64,
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
            texture_field: None,
            texture_transform: Transform::Linear,
            texture_gamma: 1.0,
            combine: Combine::Multiply,
            texture_weight: 1.0,
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
    /// sensible for that field (`Log` for the trap stalk fields, `Linear` for
    /// stripe — Matt's validated sweep verdict — `Sqrt` otherwise). All other
    /// knobs stay at defaults; tune via JSON overrides.
    pub fn beautiful(field: Field) -> Self {
        let transform = match field {
            Field::TrapCircle | Field::TrapCross => Transform::Log,
            // Validated stripe default (sweep verdict): linear, not sqrt.
            Field::Stripe => Transform::Linear,
            _ => Transform::Sqrt,
        };
        let mut p = ColoringParams {
            bailout_b: BEAUTIFUL_BAILOUT,
            field,
            transform,
            ..ColoringParams::default()
        };
        // Validated stripe default: density 6 (sweep usable band 4–8).
        if matches!(field, Field::Stripe) {
            p.stripe_density = 6.0;
        }
        p
    }

    /// Serialize to a compact JSON object (all fields, stable key order).
    /// `texture_field` is emitted as `"none"` when absent (single-field).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"bailout_b\":{},\"skip\":{},\"biomorph\":\"{}\",\"field\":\"{}\",\
             \"stripe_density\":{},\"trap_radius\":{},\"transform\":\"{}\",\"gamma\":{},\
             \"shade\":\"{}\",\"light_azimuth\":{},\"light_height\":{},\
             \"palette_cycles\":{},\"palette_offset\":{},\
             \"texture_field\":\"{}\",\"texture_transform\":\"{}\",\"texture_gamma\":{},\
             \"combine\":\"{}\",\"texture_weight\":{}}}",
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
            self.texture_field.map_or("none", Field::as_str),
            self.texture_transform.as_str(),
            self.texture_gamma,
            self.combine.as_str(),
            self.texture_weight,
        )
    }

    /// Parse a JSON object (tolerant `jsonl` readers, flat object, first-match per
    /// key). **§0 seeding contract:** a spec that names *any* recognized key seeds
    /// from [`beautiful`](Self::beautiful)`(field)` (the field-appropriate preset),
    /// then overlays the explicitly-present keys — so `{"field":"stripe"}` ≡
    /// `beautiful(Stripe)` and unspecified `bailout_b`/`transform` follow the field
    /// instead of the sentinel's `1e6`/`linear`. An **empty** spec (`{}`, no
    /// recognized key) returns [`Default`] unchanged, so the
    /// `resolve_coloring → is_location_profile` location dispatch stays
    /// byte-for-byte identical. A fully-pinned spec is unaffected (every key wins
    /// over the seed).
    pub fn from_json(s: &str) -> Result<Self, String> {
        // Overlay every present JSON key onto `p`; returns whether the spec named
        // *any* recognized key. Presence (not value-equality with the sentinel) is
        // what distinguishes an empty `{}` from a `{"field":"smooth"}` that happens
        // to equal `default()`.
        fn overlay(p: &mut ColoringParams, s: &str) -> Result<bool, String> {
            let mut named = false;
            if let Some(v) = jsonl::field_f64(s, "bailout_b") {
                p.bailout_b = v;
                named = true;
            }
            if let Some(v) = jsonl::field_usize(s, "skip") {
                p.skip = v as u32;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "biomorph") {
                p.biomorph = Biomorph::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "field") {
                p.field = Field::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "stripe_density") {
                p.stripe_density = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "trap_radius") {
                p.trap_radius = v;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "transform") {
                p.transform = Transform::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "gamma") {
                p.gamma = v;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "shade") {
                p.shade = Shade::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "light_azimuth") {
                p.light_azimuth = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "light_height") {
                p.light_height = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "palette_cycles") {
                p.palette_cycles = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "palette_offset") {
                p.palette_offset = v;
                named = true;
            }
            // v2 composite. `texture_field` absent or "none" → single-field (None).
            if let Some(v) = jsonl::field_str(s, "texture_field") {
                p.texture_field = if v == "none" {
                    None
                } else {
                    Some(Field::parse(&v)?)
                };
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "texture_transform") {
                p.texture_transform = Transform::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "texture_gamma") {
                p.texture_gamma = v;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "combine") {
                p.combine = Combine::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "texture_weight") {
                p.texture_weight = v;
                named = true;
            }
            Ok(named)
        }

        // Probe pass onto the sentinel: discover the named field (if any) and
        // whether the spec is empty.
        let mut probe = ColoringParams::default();
        if !overlay(&mut probe, s)? {
            // Empty / absent-equivalent spec → location sentinel, untouched.
            return Ok(probe); // == ColoringParams::default()
        }
        // Named spec → re-seed from the field's beautiful preset, then re-overlay
        // the explicit keys (they always win over the seed).
        let mut p = ColoringParams::beautiful(probe.field);
        overlay(&mut p, s)?;
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
/// post-loop deband can lerp the last term by the fractional iteration ([`deband`]).
#[derive(Clone, Copy, Debug)]
pub struct OrbitAccum {
    pub escaped: bool,
    /// Smooth iteration count (valid when `escaped`). Bailout-normalized
    /// (`nu = (n+1) − log2(ln|z|/ln B)` → 0 at `|z| = B`), so its fraction is the
    /// correct deband weight — see [`smooth_value`].
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
/// the mean of all terms and `A_prev` excludes the last term (doc §3). `d` is the
/// bailout-normalized fractional iteration (→ 0 at the escape boundary), which is
/// what de-terraces the bands — the previous attempt fed an un-normalized `d`
/// (omitting `−log2(ln B)`), phase-shifting the lerp by ~0.47 band and producing
/// the "brick / fish-scale" terrace. `None` if no terms; plain `sum` for one term.
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
        // Fractional iteration for the deband lerp (exterior fields only). `smooth`
        // is bailout-normalized, so its fraction → 0 at the escape boundary.
        let d = if self.escaped {
            self.smooth.fract().clamp(0.0, 1.0)
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
            smooth = smooth_value(n, zabs2, b);
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

/// `nu = (n+1) − log2(ln|z| / ln B)` — the **bailout-normalized** smooth iteration,
/// which → 0 at the escape boundary (`|z| = B`). Its fraction is therefore the
/// correct deband weight ([`field`](OrbitAccum::field) / [`deband`]).
///
/// For the *smooth field itself* the `−log2(ln B)` term is an additive constant the
/// percentile-stretch absorbs (so the smooth render is invariant to it — verified
/// by pixel-diff against the un-normalized formula). But in the **deband weight**
/// path the constant is load-bearing: omitting it (the previous shortcut) phase-
/// shifts the lerp by `log2(ln B) mod 1` and terraces the bands. We normalize here
/// once so both paths share the correct value. `B` is threaded from
/// `params.bailout_b` — never hardcoded — so the normalization tracks the bailout.
#[inline]
fn smooth_value(n: u32, zabs2: f64, bailout_b: f64) -> f64 {
    let log_zn = 0.5 * zabs2.ln(); // ln|z|
    let log_b = bailout_b.ln(); // ln B
    let ratio = log_zn / log_b;
    if log_zn > 0.0 && log_b > 0.0 && ratio.is_finite() && ratio > 0.0 {
        (n + 1) as f64 - ratio.ln() / std::f64::consts::LN_2
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

/// One field's **global normalization** to `[0,1]` — the v1 two-pass stat
/// machinery factored out so it can run *independently per field* (the v2 fix: a
/// texture stretches on its own distribution, not a shared one). Built from the
/// field's valid finite raw values; `map01` reproduces the v1 single-field math
/// exactly (histeq rank fraction or percentile stretch).
enum FieldNorm {
    /// Percentile stretch: `clamp((v−lo)/span, 0, 1)`.
    Stretch { lo: f64, span: f64 },
    /// Histogram-equalization: rank fraction over the sorted valid values.
    Histeq { sorted: Vec<f64> },
}

impl FieldNorm {
    /// Build from the field's valid finite raw values (consumed/reordered).
    fn build(mut valids: Vec<f64>, transform: Transform) -> Self {
        if transform == Transform::Histeq {
            valids.sort_unstable_by(f64::total_cmp);
            FieldNorm::Histeq { sorted: valids }
        } else {
            let lo = percentile(&mut valids, PCT_LO);
            let hi = percentile(&mut valids, PCT_HI);
            let span = if hi > lo { hi - lo } else { 1.0 };
            FieldNorm::Stretch { lo, span }
        }
    }
    /// Raw field value → `x ∈ [0,1]` (pre-transform).
    #[inline]
    fn map01(&self, value: f64) -> f64 {
        match self {
            FieldNorm::Histeq { sorted } => {
                if sorted.len() <= 1 {
                    0.0
                } else {
                    let rank = sorted.partition_point(|&v| v < value);
                    rank as f64 / (sorted.len() - 1) as f64
                }
            }
            FieldNorm::Stretch { lo, span } => ((value - lo) / span).clamp(0.0, 1.0),
        }
    }
}

/// A per-subpixel reduction of an orbit: the raw field scalar (if valid) and the
/// emboss vector. Kept small so the supersample buffer stays modest.
#[derive(Clone, Copy)]
struct ShadePix {
    value: f64,
    valid: bool,
    ushade: Complex<f64>,
}

/// Per-subpixel reduction for the **composite** path: base + texture raw scalars
/// (each with its own validity) and the shared emboss vector.
#[derive(Clone, Copy)]
struct CompositePix {
    base: f64,
    base_valid: bool,
    tex: f64,
    tex_valid: bool,
    ushade: Complex<f64>,
}

/// Render one location through the beautiful pipeline → sRGB image.
///
/// `julia_param = Some(c)` renders a Julia (viewport is the z-plane, `z0 = pixel`);
/// `None` renders the Mandelbrot. Grid-centered supersampling (the rgss/jitter
/// placements are a smooth-path AA study; beautiful v1 uses grid). The trap fields
/// fill the interior; exterior-only fields render interior pixels black.
///
/// Dispatches on `params.texture_field`: `None` → the **v1 single-field** path
/// (byte-identical, [`render_beautiful_single`]); `Some` → the v2
/// **field-modulates-field composite** ([`render_beautiful_composite`]). The split
/// is a hard branch — texture-absent never touches the composite lerp/ops, so its
/// float bytes are preserved by construction.
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
    if params.texture_field.is_some() {
        return render_beautiful_composite(frame, ss, maxiter, julia_param, params, palette, filter);
    }
    render_beautiful_single(frame, ss, maxiter, julia_param, params, palette, filter)
}

/// The **v1 single-field** path — verbatim. Do not modify: its output bytes are a
/// reference the montage/SHA guard pins.
#[allow(clippy::too_many_arguments)]
fn render_beautiful_single(
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

/// The **v2 composite** path: `base` field modulated by a `texture` field. One
/// orbit pass feeds both fields (the [`OrbitAccum`] union already carries every
/// channel — no per-field gating to extend); each field then normalizes
/// **independently** (its own [`FieldNorm`] + transform + gamma) to `[0,1]`,
/// `params.combine` merges them, and `params.texture_weight` lerps base↔op. Shade
/// (`normal_map`) and palette apply POST-combine to the final scalar, exactly as v1.
///
/// Validity is **graceful per-field**: a subpixel renders if *either* field is
/// valid, and where one field is absent its operand drops out — `smooth × trap`
/// on a Julia therefore keeps the interior lace (texture-only, since smooth is
/// exterior-only there) instead of blacking it out, and the composite reads only
/// where the base actually exists (the exterior). Black only where *neither* is
/// valid.
#[allow(clippy::too_many_arguments)]
fn render_beautiful_composite(
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
    let texture_field = params
        .texture_field
        .expect("composite branch requires texture_field");

    // --- iterate + reduce to (base, tex, ushade) per subpixel ---
    let rows: Vec<Vec<CompositePix>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                let (z0, c) = match julia_param {
                    Some(p) => (pixel, p),
                    None => (Complex::new(0.0, 0.0), pixel),
                };
                let acc = iterate_orbit(z0, c, maxiter, params, julia);
                // One pass, two fields off the shared channel union.
                let bv = acc.field(params.field);
                let tv = acc.field(texture_field);
                row.push(CompositePix {
                    base: bv.unwrap_or(0.0),
                    base_valid: bv.is_some(),
                    tex: tv.unwrap_or(0.0),
                    tex_valid: tv.is_some(),
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
    let pix: Vec<CompositePix> = rows.into_iter().flatten().collect();

    // --- per-field global normalization (independent stat passes) ---
    let base_norm = FieldNorm::build(
        pix.iter()
            .filter(|p| p.base_valid && p.base.is_finite())
            .map(|p| p.base)
            .collect(),
        params.transform,
    );
    let tex_norm = FieldNorm::build(
        pix.iter()
            .filter(|p| p.tex_valid && p.tex.is_finite())
            .map(|p| p.tex)
            .collect(),
        params.texture_transform,
    );

    let (saz, caz) = params.light_azimuth.sin_cos();
    let h = params.light_height;
    let cycles = params.palette_cycles;
    let offset = params.palette_offset;
    let base_t = params.transform;
    let base_g = params.gamma;
    let tex_t = params.texture_transform;
    let tex_g = params.texture_gamma;
    let combine = params.combine;
    let weight = params.texture_weight;

    // --- shade each subpixel to linear RGB ---
    let linear: Vec<[f64; 3]> = pix
        .par_iter()
        .map(|p| {
            // Graceful per-field validity: combine where both exist; fall back to
            // the surviving field alone where one is absent; black only if neither.
            let bok = p.base_valid && p.base.is_finite();
            let tok = p.tex_valid && p.tex.is_finite();
            let mut gray = match (bok, tok) {
                (false, false) => return [0.0, 0.0, 0.0],
                (true, false) => apply_transform(base_norm.map01(p.base), base_t, base_g),
                (false, true) => apply_transform(tex_norm.map01(p.tex), tex_t, tex_g),
                (true, true) => {
                    let base_n = apply_transform(base_norm.map01(p.base), base_t, base_g);
                    let tex_n = apply_transform(tex_norm.map01(p.tex), tex_t, tex_g);
                    let op = combine.apply(base_n, tex_n);
                    base_n + (op - base_n) * weight // lerp(base_n, op, weight)
                }
            };

            // normal_map emboss over the composite scalar (post-combine, v1 semantics).
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

    /// The averaging family (stripe/tia/curvature) is the **deband lerp**
    /// `d·A + (1−d)·A_prev`, with `d = smooth.fract()` (smooth is bailout-normalized).
    #[test]
    fn averaging_fields_are_deband_lerp() {
        // smooth = 12.7 → d = 0.7. Lerp = d·(sum/cnt) + (1−d)·((sum−last)/(cnt−1)).
        let acc = OrbitAccum {
            escaped: true,
            smooth: 12.7,
            z: Complex::new(3.0, 4.0),
            dz: Complex::new(1.0, 0.0),
            stripe: (8.0, 5, 1.5),
            tia: (3.0, 6, 0.4),
            curv: (2.5, 4, 0.9),
            trap_circle_min: 0.3,
            trap_cross_min: 0.1,
        };
        let d = 12.7f64.fract().clamp(0.0, 1.0); // == field()'s d (not exactly 0.7)
        let lerp = |sum: f64, cnt: u32, last: f64| {
            let a = sum / cnt as f64;
            let a_prev = (sum - last) / (cnt - 1) as f64;
            d * a + (1.0 - d) * a_prev
        };
        assert_eq!(acc.field(Field::Stripe), Some(lerp(8.0, 5, 1.5)));
        assert_eq!(acc.field(Field::Tia), Some(lerp(3.0, 6, 0.4)));
        assert_eq!(acc.field(Field::Curvature), Some(lerp(2.5, 4, 0.9)));
        // Exterior-only: a non-escaped orbit yields None for the averaging family.
        let interior = OrbitAccum { escaped: false, ..acc };
        assert_eq!(interior.field(Field::Stripe), None);
        assert_eq!(interior.field(Field::Tia), None);
        assert_eq!(interior.field(Field::Curvature), None);
        // count == 0 → None even when escaped (no terms accumulated).
        let empty = OrbitAccum { stripe: (0.0, 0, 0.0), ..acc };
        assert_eq!(empty.field(Field::Stripe), None);
        // count == 1 → plain sum (deband needs ≥2 terms).
        let one = OrbitAccum { stripe: (0.42, 1, 0.42), ..acc };
        assert_eq!(one.field(Field::Stripe), Some(0.42));
    }

    #[test]
    fn json_roundtrip() {
        let p = ColoringParams::beautiful(Field::Stripe);
        let p2 = ColoringParams::from_json(&p.to_json()).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn json_partial_seeds_from_beautiful() {
        // §0 fix: a partial spec that names a field seeds from beautiful(field), so
        // unspecified bailout/transform follow the field preset — NOT the sentinel's
        // 1e6/linear.
        let p = ColoringParams::from_json("{\"field\":\"tia\"}").unwrap();
        assert_eq!(p, ColoringParams::beautiful(Field::Tia));
        assert_eq!(p.bailout_b, BEAUTIFUL_BAILOUT);
        assert_eq!(p.transform, Transform::Sqrt);

        // `{"field":"stripe"}` ≡ the validated beautiful(Stripe) default.
        let s = ColoringParams::from_json("{\"field\":\"stripe\"}").unwrap();
        assert_eq!(s, ColoringParams::beautiful(Field::Stripe));

        // Spot-check a non-stripe field seeds its beautiful preset (log transform).
        let tc = ColoringParams::from_json("{\"field\":\"trap_cross\"}").unwrap();
        assert_eq!(tc, ColoringParams::beautiful(Field::TrapCross));
        assert_eq!(tc.transform, Transform::Log);

        // An explicit key still wins over the seed; unspecified keys keep the seed.
        let pin =
            ColoringParams::from_json("{\"field\":\"stripe\",\"transform\":\"log\"}").unwrap();
        assert_eq!(pin.transform, Transform::Log);
        assert_eq!(pin.stripe_density, 6.0); // unspecified → beautiful(Stripe) seed
    }

    #[test]
    fn beautiful_stripe_default_is_density6_linear() {
        let p = ColoringParams::beautiful(Field::Stripe);
        assert_eq!(p.stripe_density, 6.0);
        assert_eq!(p.transform, Transform::Linear);
        assert_eq!(p.bailout_b, BEAUTIFUL_BAILOUT);
    }

    #[test]
    fn empty_json_is_default() {
        // `{}` names no key → location sentinel preserved (the §3 dispatch contract).
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

    /// Composite JSON round-trips, including `texture_field: None → "none"`.
    #[test]
    fn composite_json_roundtrip() {
        let mut p = ColoringParams::beautiful(Field::Smooth);
        p.texture_field = Some(Field::TrapCross);
        p.texture_transform = Transform::Linear;
        p.combine = Combine::Screen;
        p.texture_weight = 0.5;
        let p2 = ColoringParams::from_json(&p.to_json()).unwrap();
        assert_eq!(p, p2);
        // Absent texture survives the round-trip as None (single-field).
        let single = ColoringParams::beautiful(Field::TrapCross);
        assert!(single.texture_field.is_none());
        assert_eq!(single, ColoringParams::from_json(&single.to_json()).unwrap());
        // `{}` is still the location-profile sentinel after the v2 additions.
        assert!(ColoringParams::from_json("{}").unwrap().is_location_profile());
    }

    /// Combine ops map `[0,1]² → [0,1]` and hit their defining values.
    #[test]
    fn combine_ops_in_unit_range() {
        let ops = [
            Combine::Multiply,
            Combine::Screen,
            Combine::Overlay,
            Combine::Min,
        ];
        for op in ops {
            for bi in 0..=10 {
                for ti in 0..=10 {
                    let b = bi as f64 / 10.0;
                    let t = ti as f64 / 10.0;
                    let o = op.apply(b, t);
                    assert!((0.0..=1.0).contains(&o), "{op:?}({b},{t}) = {o}");
                }
            }
        }
        assert_eq!(Combine::Multiply.apply(0.5, 0.5), 0.25);
        assert_eq!(Combine::Screen.apply(0.5, 0.5), 0.75);
        assert_eq!(Combine::Min.apply(0.3, 0.8), 0.3);
    }

    /// A multiply composite on a small exterior patch produces finite, in-gamut
    /// pixels and is not all-black (the smoke patch escapes).
    #[test]
    fn composite_smoke_patch_in_range() {
        let mut p = ColoringParams::beautiful(Field::Smooth);
        p.texture_field = Some(Field::TrapCross);
        p.combine = Combine::Multiply;
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.02,
            out_width: 16,
            out_height: 12,
        };
        let palette = crate::palette::builtin("default", false).unwrap();
        let img = render_beautiful(
            &frame,
            1,
            300,
            None,
            &p,
            &palette,
            DownsampleFilter::Box,
        );
        assert_eq!(img.dimensions(), (16, 12));
        let any_lit = img.pixels().any(|px| px.0 != [0, 0, 0]);
        assert!(any_lit, "composite smoke patch rendered all black");
    }

    /// Texture-absent ≡ v1 single field: a composite-capable param with
    /// `texture_field: None` renders byte-identically to the same param stripped of
    /// all composite knobs (the separate-branch guard, scalar parity on a patch).
    #[test]
    fn texture_absent_equals_single_field() {
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.02,
            out_width: 24,
            out_height: 16,
        };
        let palette = crate::palette::builtin("default", false).unwrap();
        // A trap_cross/log/normal_map single-field spec.
        let single = ColoringParams::beautiful(Field::TrapCross);
        assert!(single.texture_field.is_none());
        // Same base, but with composite knobs populated AND texture_field None —
        // must still take the single branch and match bit-for-bit.
        let mut dressed = single;
        dressed.combine = Combine::Screen;
        dressed.texture_weight = 0.7;
        dressed.texture_transform = Transform::Histeq;
        let a = render_beautiful(&frame, 2, 400, None, &single, &palette, DownsampleFilter::Lanczos3);
        let b = render_beautiful(&frame, 2, 400, None, &dressed, &palette, DownsampleFilter::Lanczos3);
        assert_eq!(a.into_raw(), b.into_raw());
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
