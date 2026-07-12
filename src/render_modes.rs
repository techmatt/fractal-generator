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

use crate::backend::{
    F64Backend, FractalBackend, JuliaBackend, PhoenixBackend, Trap, TrapShape, PHASE_DEFER,
};
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

/// The orbit-escape **family** — which recurrence the iterate stage runs and how
/// the viewport plane seeds it. This is the generalization of the old
/// Mandelbrot-vs-Julia `julia: bool` (whose entire semantics were: dynamical
/// families fix the constant and sweep `z₀ = pixel`, parameter-plane families fix
/// `z₀ = 0` and sweep `c = pixel`).
///
///  - [`Mandelbrot`](Self::Mandelbrot) / [`Julia`](Self::Julia) — `z → z² + c`,
///    degree 2 (parameter / dynamical). Reproduce the prior two paths byte-for-byte.
///  - [`Multibrot`](Self::Multibrot) — `z → z^d + c`, parameter plane (`z₀ = 0`),
///    `d ∈ {3,4,5}`. `z^d` by repeated complex multiplication.
///  - [`Phoenix`](Self::Phoenix) — Ushiki `z_{n+1} = z_n² + c + p·z_{n-1}`,
///    dynamical (`z₀ = pixel`, `z_{-1} = 0`), degree 2. `c` is the additive constant
///    (Julia's role); `p` is the `z_{n-1}` coefficient (Ushiki's `q`). Both fixed
///    per render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Family {
    /// `z → z² + c`, `z₀ = 0` (parameter plane).
    Mandelbrot,
    /// `z → z^d + c`, `z₀ = pixel`, `c` fixed (dynamical / seed plane). `degree == 2`
    /// is the classic quadratic Julia (byte-identical path); `d ∈ {3,4,5}` are the
    /// **Julia-multibrot** dynamical planes (`--julia --family multibrot{d}`).
    Julia { c: Complex<f64>, degree: u32 },
    /// `z → z^d + c`, `z₀ = 0` (parameter plane), `d ∈ {3,4,5}`.
    Multibrot { degree: u32 },
    /// `z_{n+1} = z_n² + c + p·z_{n-1}`, `z₀ = pixel`, `z_{-1} = 0` (dynamical).
    Phoenix { c: Complex<f64>, p: Complex<f64> },
}

impl Family {
    /// Escape degree `d`: the exponent in `z^d` that dominates near escape, so the
    /// smooth-field outer log base is `ln d` (Phoenix's `z²` dominates → 2).
    pub fn degree(self) -> u32 {
        match self {
            Family::Multibrot { degree } | Family::Julia { degree, .. } => degree,
            Family::Mandelbrot | Family::Phoenix { .. } => 2,
        }
    }

    /// Dynamical (seed-plane) families fix the constant and sweep `z₀ = pixel`;
    /// parameter-plane families fix `z₀ = 0` and sweep `c = pixel`. This also sets
    /// the `dz` seed (`1` dynamical, `0` parameter) and whether the `dz` recurrence
    /// carries the parameter-plane `+1`.
    pub fn is_dynamical(self) -> bool {
        matches!(self, Family::Julia { .. } | Family::Phoenix { .. })
    }

    /// `(z₀, c)` for a pixel: dynamical → `(pixel, fixed_const)`, parameter-plane →
    /// `(0, pixel)`. The single source for the per-pixel seed the render fns used to
    /// inline as `match julia_param`.
    pub fn seed(self, pixel: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
        match self {
            Family::Julia { c, .. } => (pixel, c),
            Family::Phoenix { c, .. } => (pixel, c),
            Family::Mandelbrot | Family::Multibrot { .. } => (Complex::new(0.0, 0.0), pixel),
        }
    }

    /// Version-invariant `location.kind` string for the `--dump-field` sidecar.
    /// Julia-multibrot (`d ≥ 3`) reads back as `julia_multibrot{d}` so a sidecar
    /// round-trips to the same dynamical z^d plane; the quadratic Julia stays `julia`.
    pub fn kind_str(self) -> &'static str {
        match self {
            Family::Mandelbrot => "mandelbrot",
            Family::Julia { degree: 2, .. } => "julia",
            Family::Julia { degree: 3, .. } => "julia_multibrot3",
            Family::Julia { degree: 4, .. } => "julia_multibrot4",
            Family::Julia { degree: 5, .. } => "julia_multibrot5",
            Family::Julia { .. } => "julia_multibrot",
            Family::Multibrot { degree: 3 } => "multibrot3",
            Family::Multibrot { degree: 4 } => "multibrot4",
            Family::Multibrot { degree: 5 } => "multibrot5",
            Family::Multibrot { .. } => "multibrot",
            Family::Phoenix { .. } => "phoenix",
        }
    }
}

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
    /// Mean discrete orbit velocity `mean(|z_{n+1} − z_n|)` over the **full**
    /// orbit (interior iterates included). Returned unconditionally — interior-
    /// valued like the trap fields, NOT gated to `escaped` like the averaging
    /// family (stripe/tia/curvature). The standard Siegel-disk interior coloring.
    Velocity,
    /// Exterior distance estimate `de = |z|·ln|z|/|dz|`, normalized to pixel scale
    /// (`de_px = de/(fw/width)`) and soft-mapped `tanh(de_px·de_scale)` → 0 at the
    /// boundary, 1 in the open exterior. Exterior only (interior → black, like
    /// curvature). Reuses the `normal_map` derivative recurrence; `de_scale` is the
    /// only DE-specific knob (filament thickness). Needs pixel scale, so it is
    /// reduced via [`OrbitAccum::de_value`], not [`OrbitAccum::field`].
    De,
    /// **Gaussian Integer** lattice trap: `min‖z − round(z/N)·N‖` over the orbit,
    /// with `N = 1` (the unit integer lattice). A min-distance point-lattice trap —
    /// same shape as the trap fields, valid for every pixel (fills the interior).
    /// "Color By = minimum distance" — the canonical look.
    GaussianInt,
    /// **Exponential Smoothing**: `Σ exp(−|zₙ|)` over the divergent orbit (an
    /// averaging-family member; escaped-gated like smooth/stripe). A drop-in
    /// alternative shading to smooth. `divergescale = 1.0`, so `#index` is the raw
    /// sum (the percentile-stretch absorbs the scale).
    ///
    /// **Redundant with `smooth` — `niche`, deprecated for render-mode exploration.**
    /// Added with the UF-algorithm reconstruction (`uf_coloring_algorithms.md`) as a
    /// formula-agnostic smooth-alternative, *before* `smooth` was promoted the canonical
    /// base carrier, so the overlap is by design. Empirically this field is monotone with
    /// `smooth` (Spearman ≥ 0.999 across all 8 pilot families ⇒ beam-equivalent under the
    /// gamma/transfer/n_cycles freedom), so its render-mode-pilot rasters were pixel-dupes
    /// of their smooth counterpart (ΔE76 < 5, all flagged `too_close_to_smooth`). Its one
    /// nominal knob `divergescale` is hardcoded 1.0 and, as a constant scale *before* the
    /// percentile-stretch, is absorbed to a no-op (no non-stretched/fixed-palette index
    /// path exists to make it live). Keep `smooth` as the base carrier.
    ExpSmoothing,
    /// **Decomposition**: final-only, escaped pixels only — `atan2(z_final)` folded
    /// to `[0,1)`. No accumulation; reads the escape-point angle. Pairs with a low
    /// bail-out radius (escape radius 2 ⇒ `bailout_b = 4`) for the cleanest petals.
    Decomposition,
    /// **Direct Orbit Traps** — *not a scalar field*. A direct-colour algorithm that
    /// composites a gradient sample every iteration the orbit lands inside the trap
    /// and emits `#color` directly, bypassing scalar-index normalization. Routed to
    /// the parallel colour-valued path [`render_direct_trap`]; [`OrbitAccum::field`]
    /// returns `None` for it (the scalar pipeline never sees this field).
    DirectTrap,
}

impl Field {
    pub fn as_str(self) -> &'static str {
        match self {
            Field::Smooth => "smooth",
            Field::Stripe => "stripe",
            Field::Tia => "tia",
            Field::Curvature => "curvature",
            Field::TrapCircle => "trap_circle",
            Field::TrapCross => "trap_cross",
            Field::Velocity => "velocity",
            Field::De => "de",
            Field::GaussianInt => "gaussian_int",
            Field::ExpSmoothing => "exp_smoothing",
            Field::Decomposition => "decomposition",
            Field::DirectTrap => "direct_trap",
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
            "velocity" => Field::Velocity,
            "de" => Field::De,
            "gaussian_int" => Field::GaussianInt,
            "exp_smoothing" => Field::ExpSmoothing,
            "decomposition" => Field::Decomposition,
            "direct_trap" => Field::DirectTrap,
            _ => return Err(format!("unknown field '{s}'")),
        })
    }
}

/// **Gaussian Integer "Color By"** — which reduction of the per-iteration lattice
/// orbit-trap (`round`, `N=1`) becomes the `#index`. All nine modes read the *same*
/// orbit accumulator ([`GaussTrap`]: `rmin/zmin/itermin`, `rmax/zmax/itermax`,
/// `total`/`count`), so this is a free reduction axis on one iterate pass (doc §1
/// "Color By" table). [`MinimumDistance`](Self::MinimumDistance) is the canonical
/// default and reproduces the prior single-mode behaviour.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GaussianColorBy {
    /// `rmin` — distance to the nearest lattice point over the orbit (default look).
    MinimumDistance,
    /// `rave = total/count` — mean lattice distance.
    AverageDistance,
    /// `rmax` — farthest lattice approach over the orbit.
    MaximumDistance,
    /// `0.01·itermin` — iteration index of the closest approach (folded mod 1).
    IterMin,
    /// `0.01·itermax` — iteration index of the farthest approach (folded mod 1).
    IterMax,
    /// `normalize_angle(zmin)` — angle of `z` at the closest approach.
    AngleMin,
    /// `normalize_angle(zmax)` — angle of `z` at the farthest approach.
    AngleMax,
    /// `normalize_angle((rave−rmin) + i(rmax−rave))` — min/mean/max spread angle.
    MeanAngle,
    /// `rmax / (rmin + 1e-12)` — max/min ratio (high-contrast field).
    Ratio,
}

impl GaussianColorBy {
    fn as_str(self) -> &'static str {
        match self {
            GaussianColorBy::MinimumDistance => "minimum_distance",
            GaussianColorBy::AverageDistance => "average_distance",
            GaussianColorBy::MaximumDistance => "maximum_distance",
            GaussianColorBy::IterMin => "iter_min",
            GaussianColorBy::IterMax => "iter_max",
            GaussianColorBy::AngleMin => "angle_min",
            GaussianColorBy::AngleMax => "angle_max",
            GaussianColorBy::MeanAngle => "mean_angle",
            GaussianColorBy::Ratio => "ratio",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "minimum_distance" => GaussianColorBy::MinimumDistance,
            "average_distance" => GaussianColorBy::AverageDistance,
            "maximum_distance" => GaussianColorBy::MaximumDistance,
            "iter_min" => GaussianColorBy::IterMin,
            "iter_max" => GaussianColorBy::IterMax,
            "angle_min" => GaussianColorBy::AngleMin,
            "angle_max" => GaussianColorBy::AngleMax,
            "mean_angle" => GaussianColorBy::MeanAngle,
            "ratio" => GaussianColorBy::Ratio,
            _ => return Err(format!("unknown color_by '{s}'")),
        })
    }
    /// Iteration modes are designed for a **direct mod-1 gradient read** (the `0.01·`
    /// scaling bands every 100 iters). They are folded mod 1 at reduction and bypass
    /// the percentile-stretch, which would otherwise flatten the banding into a ramp
    /// (prompt §Normalization).
    fn is_iteration(self) -> bool {
        matches!(self, GaussianColorBy::IterMin | GaussianColorBy::IterMax)
    }
}

/// `normalize_angle(w) = atan2(w)/π`, folded into `[0,1)` (shift `+2` if negative,
/// then `·0.5`) — the doc §1 angle reduction shared by the angle Color-By modes.
#[inline]
fn normalize_angle(w: Complex<f64>) -> f64 {
    let mut a = w.im.atan2(w.re) / std::f64::consts::PI; // [-1, 1]
    if a < 0.0 {
        a += 2.0; // [0, 2)
    }
    (a * 0.5).rem_euclid(1.0) // [0, 1)
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

/// **Highlight rolloff** — a luminance-domain tone-compression applied to the final
/// linear-RGB color right before downsample/sRGB, gated to the clipping-prone screen
/// family (screen composites + additive `direct_trap` screen merge). The screen
/// operators drive their output toward white (the additive direct-trap accumulator
/// asymptotes to `1` as trap hits stack); large near-white plateaus read as blown.
/// A rolloff compresses the top of the tone range so residual color re-emerges from
/// the wash, while leaving midtones/shadows near-identity.
///
/// **Luminance-domain, not per-channel:** the operator maps the pixel's luminance
/// `L → L'` and rescales all three channels by `L'/L`, preserving hue/chroma. A
/// per-channel curve would pull the brightest channel down faster than the others and
/// desaturate the highlight toward gray/white — the opposite of the goal.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Rolloff {
    /// Identity — no rolloff (default; byte-identical to the pre-rolloff path).
    None,
    /// Extended Reinhard on luminance with white point `w = strength`
    /// (`L' = L(1 + L/w²)/(1+L)`): `w` maps to `1`. Compresses the whole range (also
    /// darkens midtones/shadows somewhat), so it's the "global" option.
    Reinhard,
    /// Narkowicz ACES filmic on exposure-scaled luminance (`strength` = pre-exposure).
    /// Filmic shoulder + slight toe; the punchiest recovery, mild saturation shift.
    Aces,
    /// Soft-knee shoulder: identity below the knee `k = strength`, `tanh` shoulder
    /// above (`L' = k + (1-k)·tanh((L-k)/(1-k))`, C¹ at the knee, → 1 as `L → ∞`).
    /// Near-identity by construction on in-range content — bends only the highlights.
    SoftKnee,
}

impl Rolloff {
    fn as_str(self) -> &'static str {
        match self {
            Rolloff::None => "none",
            Rolloff::Reinhard => "reinhard",
            Rolloff::Aces => "aces",
            Rolloff::SoftKnee => "soft_knee",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "none" => Rolloff::None,
            "reinhard" => Rolloff::Reinhard,
            "aces" => Rolloff::Aces,
            "soft_knee" | "softknee" => Rolloff::SoftKnee,
            _ => return Err(format!("unknown rolloff '{s}'")),
        })
    }
    /// Map a single luminance value `l ≥ 0` under this operator. `strength` is the
    /// operator's one knob (white point / exposure / knee — see the variant docs).
    #[inline]
    fn map_luma(self, l: f64, strength: f64) -> f64 {
        match self {
            Rolloff::None => l,
            Rolloff::Reinhard => {
                let w = strength.max(1e-6);
                l * (1.0 + l / (w * w)) / (1.0 + l)
            }
            Rolloff::Aces => {
                let x = l * strength;
                let n = x * (2.51 * x + 0.03);
                let d = x * (2.43 * x + 0.59) + 0.14;
                (n / d).clamp(0.0, 1.0)
            }
            Rolloff::SoftKnee => {
                let k = strength.clamp(0.0, 0.999);
                if l <= k {
                    l
                } else {
                    k + (1.0 - k) * ((l - k) / (1.0 - k)).tanh()
                }
            }
        }
    }
}

/// Rec.709 linear luminance of a linear-RGB pixel.
#[inline]
fn luma709(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Apply a highlight rolloff to one linear-RGB pixel in the **luminance domain**:
/// map `L → L'` and rescale all channels by `L'/L` (chroma-preserving), clamping the
/// result to `[0,1]`. `None` is the exact identity (early return → byte-identical).
/// Black pixels (`L ≈ 0`) pass through untouched.
#[inline]
fn apply_rolloff(lin: [f64; 3], op: Rolloff, strength: f64) -> [f64; 3] {
    if op == Rolloff::None {
        return lin;
    }
    let l = luma709(lin);
    if l <= 1e-9 {
        return lin;
    }
    let s = op.map_luma(l, strength) / l;
    [
        (lin[0] * s).clamp(0.0, 1.0),
        (lin[1] * s).clamp(0.0, 1.0),
        (lin[2] * s).clamp(0.0, 1.0),
    ]
}

/// Direct-orbit-traps **merge mode** (doc §3 compositing). Each per-iteration trap
/// hit blends its gradient sample RGB against the accumulator RGB through this mode,
/// then alpha-overs with the sample's `α = opacity·(1−d/threshold)`. With `Normal`
/// the blend is just the sample (last-hit-over), so feathered low-α samples average
/// to mud; the multiplicative/inverse modes are what build layered lace.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// Blend = sample (`s`) — the new sample simply over the accumulator.
    Normal,
    /// Blend = `a·s` — darkens toward black as layers stack.
    Multiply,
    /// Blend = `1−(1−a)(1−s)` — inverse-multiply; layers lift brightness.
    Screen,
    /// Blend = `a<0.5 ? 2as : 1−2(1−a)(1−s)` — multiply in shadows, screen in
    /// highlights; pushes midtone contrast.
    Overlay,
}

impl MergeMode {
    fn as_str(self) -> &'static str {
        match self {
            MergeMode::Normal => "normal",
            MergeMode::Multiply => "multiply",
            MergeMode::Screen => "screen",
            MergeMode::Overlay => "overlay",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "normal" => MergeMode::Normal,
            "multiply" => MergeMode::Multiply,
            "screen" => MergeMode::Screen,
            "overlay" => MergeMode::Overlay,
            _ => return Err(format!("unknown merge_mode '{s}'")),
        })
    }
    /// Blend the back operand `a` (accumulator) against the front operand `s`
    /// (sample), per channel, returning the blended channel value in `[0,1]`. The
    /// caller alpha-overs the result; merge order picks which operand is `a` vs `s`.
    #[inline]
    fn blend(self, a: f64, s: f64) -> f64 {
        match self {
            MergeMode::Normal => s,
            MergeMode::Multiply => a * s,
            MergeMode::Screen => 1.0 - (1.0 - a) * (1.0 - s),
            MergeMode::Overlay => {
                if a < 0.5 {
                    2.0 * a * s
                } else {
                    1.0 - 2.0 * (1.0 - a) * (1.0 - s)
                }
            }
        }
    }
}

/// Direct-orbit-traps **merge order** (doc §3). `BottomUp` blends the new sample
/// onto the accumulator (`blend(acc, sample)`); `TopDown` blends the accumulator
/// onto the new sample (`blend(sample, acc)`) so earlier hits dominate. Symmetric
/// for multiply/screen; only `Overlay`/`Normal` actually see the swap. The alpha-
/// over step is identical either way — only the blend operand order changes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MergeOrder {
    /// New sample over accumulator — `blend(acc, sample)`.
    BottomUp,
    /// Accumulator over new sample — `blend(sample, acc)`.
    TopDown,
}

impl MergeOrder {
    fn as_str(self) -> &'static str {
        match self {
            MergeOrder::BottomUp => "bottom_up",
            MergeOrder::TopDown => "top_down",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "bottom_up" => MergeOrder::BottomUp,
            "top_down" => MergeOrder::TopDown,
            _ => return Err(format!("unknown merge_order '{s}'")),
        })
    }
}

/// Direct-orbit-traps **shape** (doc §"Direct Orbit Traps"). Each variant is a
/// different per-orbit-point trap-*distance* function evaluated at the hardcoded
/// `trapcenter = 0` (so `z2 = z`), `rot = identity`, `aspect = 1` — i.e. a different
/// "norm" of the iterate `z` whose sub-threshold contour the composite paints. The
/// colour-key / feather / merge / start-color stages are shape-agnostic.
///
/// Distances live on **different natural scales** (L1 ≥ min-axis, L∞ ≥ min-axis,
/// astroid ≥ L1, …), so a fixed `direct_threshold` does not read the same across
/// shapes — see [`DirectShape::default_threshold`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DirectShape {
    /// `d = |z|` (L2 to origin) — radial pearled beads.
    Point,
    /// `d = | |z| − r |` — concentric overlapping scales. `r = trap_radius`.
    Ring,
    /// `d = min(|Re z|, |Im z|)` — axis "+" thorns. **The baked default**
    /// (reproduces the pre-shape behaviour byte-for-byte at `direct_threshold:0.1`).
    Cross,
    /// `d = min(axis-cross, diagonal-cross)` — an 8-ray "✳" asterisk: the union of
    /// the axis cross and its 45°-rotated twin (diagonal distance `=
    /// min(|Re−Im|,|Re+Im|)/√2`).
    Hypercross,
    /// `d = |Re z| + |Im z|` (L1) — rotated-square / diamond contours.
    Diamond,
    /// `d = max(|Re z|, |Im z|)` (L∞) — axis-aligned square contours.
    Box,
    /// `d = (|Re z|^{2/3} + |Im z|^{2/3})^{3/2}` (astroid norm) — concave 4-point star.
    Astroid,
    /// `d = |Im z|` — distance to the real axis only: anisotropic horizontal bands
    /// (the single-line, directional degenerate of the cross).
    Lines,
}

impl DirectShape {
    fn as_str(self) -> &'static str {
        match self {
            DirectShape::Point => "point",
            DirectShape::Ring => "ring",
            DirectShape::Cross => "cross",
            DirectShape::Hypercross => "hypercross",
            DirectShape::Diamond => "diamond",
            DirectShape::Box => "box",
            DirectShape::Astroid => "astroid",
            DirectShape::Lines => "lines",
        }
    }
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "point" => DirectShape::Point,
            "ring" => DirectShape::Ring,
            "cross" => DirectShape::Cross,
            "hypercross" => DirectShape::Hypercross,
            "diamond" => DirectShape::Diamond,
            "box" => DirectShape::Box,
            "astroid" => DirectShape::Astroid,
            "lines" => DirectShape::Lines,
            _ => return Err(format!("unknown shape '{s}'")),
        })
    }

    /// Per-orbit-point trap distance of the iterate `z = (zr, zi)` to the shape at
    /// `trapcenter = 0`. `radius` (= `trap_radius`) is consumed only by `Ring`.
    #[inline]
    fn dist(self, zr: f64, zi: f64, radius: f64) -> f64 {
        let ar = zr.abs();
        let ai = zi.abs();
        match self {
            DirectShape::Point => (zr * zr + zi * zi).sqrt(),
            DirectShape::Ring => ((zr * zr + zi * zi).sqrt() - radius).abs(),
            // Byte-identical to the pre-shape baked `z.re.abs().min(z.im.abs())`.
            DirectShape::Cross => ar.min(ai),
            DirectShape::Hypercross => {
                let axis = ar.min(ai);
                let diag =
                    (zr - zi).abs().min((zr + zi).abs()) * std::f64::consts::FRAC_1_SQRT_2;
                axis.min(diag)
            }
            DirectShape::Diamond => ar + ai,
            DirectShape::Box => ar.max(ai),
            DirectShape::Astroid => {
                (ar.powf(2.0 / 3.0) + ai.powf(2.0 / 3.0)).powf(1.5)
            }
            DirectShape::Lines => ai,
        }
    }

    /// Per-shape default `direct_threshold`, **coverage-anchored** to the cross's
    /// settled `0.1`. At the working anchor (Julia c=(-0.0781,-0.6515),
    /// cx/cy/fw≈0.410/0.210/0.562, maxiter 1500) the `FRACTAL_DT_STATS`
    /// instrumentation in [`render_direct_trap`] shows cross@0.1 paints **95.4 %** of
    /// pixels (its closest-approach distribution is the most concentrated near 0 — it
    /// is the easiest norm to minimize). Equalizing *painted fraction* — not the raw
    /// distance scale, which differs wildly (L1 ≥ min-axis, astroid ≫ L1) — is what
    /// makes each shape read with a comparable stroke weight, so each default is the
    /// shape's measured **p95** closest-approach (cross's own p95 is 0.094 ≈ 0.1,
    /// confirming the anchor). These are *defaults*; `direct_threshold` in a spec
    /// always overrides. `Ring` is measured at `trap_radius = 1.0`.
    #[allow(dead_code)]
    pub fn default_threshold(self) -> f64 {
        match self {
            DirectShape::Point => 0.60,
            DirectShape::Ring => 0.078,
            DirectShape::Cross => 0.10,
            DirectShape::Hypercross => 0.074,
            DirectShape::Diamond => 0.80,
            DirectShape::Box => 0.55,
            DirectShape::Astroid => 1.06,
            DirectShape::Lines => 0.127,
        }
    }
}

/// Parse a direct_trap `start_color` spec → linear-RGB `[f64; 3]`. Accepts the names
/// `black`/`white` and a `#rrggbb` (or bare `rrggbb`) sRGB hex string, gamma-decoded
/// per channel so the stored accumulator background is linear (matching the
/// gradient samples it composites against).
fn parse_start_color(s: &str) -> Result<[f64; 3], String> {
    let s = s.trim();
    match s {
        "black" => return Ok([0.0, 0.0, 0.0]),
        "white" => return Ok([1.0, 1.0, 1.0]),
        _ => {}
    }
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() != 6 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid start_color '{s}' (want black|white|#rrggbb)"
        ));
    }
    let comp = |i: usize| {
        let v = u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
        crate::palette::srgb_to_linear(v as f64 / 255.0)
    };
    Ok([comp(0), comp(2), comp(4)])
}

/// Serialize a linear-RGB `start_color` back to a `#rrggbb` sRGB hex string (the
/// `to_json` form). Round-trips black/white exactly; arbitrary colors quantize to
/// 8-bit sRGB, which is lossless enough for a background swatch.
fn start_color_to_hex(c: [f64; 3]) -> String {
    let enc = |x: f64| (crate::palette::linear_to_srgb(x) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", enc(c[0]), enc(c[1]), enc(c[2]))
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
    /// Gaussian-integer **Color By** reduction (gaussian_int field): which orbit
    /// statistic of the lattice trap becomes the index. Default `MinimumDistance`
    /// (the canonical look). Ignored by every other field.
    pub gaussint_color_by: GaussianColorBy,
    /// Boundary-filament thickness `de_scale` (de field): the only DE-specific knob.
    /// The DE map is `tanh(de_px · de_scale)`, so larger values hug the boundary
    /// (thinner gradient band), smaller values spread a wider glow. Default `1.0`.
    pub de_scale: f64,
    /// Trap entry radius `threshold` (direct_trap field): the orbit composites a
    /// gradient sample on every iteration with `trap_distance(z) < threshold`.
    /// Default `0.1` (the faithful UF starting point). Larger → fatter, more-filled
    /// strokes; smaller → sparser hits.
    pub direct_threshold: f64,
    /// Layer opacity for the direct_trap composite (the `normal` merge mode's base
    /// opacity, before the distance feather). Default `0.5` so layers show through.
    pub direct_opacity: f64,
    /// Direct_trap **merge mode**: how each per-iteration trap sample blends against
    /// the accumulator before the alpha-over. Default `Normal` (last-hit-over).
    pub merge_mode: MergeMode,
    /// Direct_trap **merge order**: which operand is the blend's back vs front.
    /// Default `BottomUp` (new sample over accumulator).
    pub merge_order: MergeOrder,
    /// Direct_trap **shape**: the per-orbit-point trap-distance function painted by
    /// the composite. Default `Cross` (= the pre-shape baked behaviour; byte-identical
    /// at `direct_threshold:0.1`). See [`DirectShape`]. `Ring` reads `trap_radius`.
    pub direct_shape: DirectShape,
    /// Direct_trap **start color**: the linear-RGB background the per-iteration trap
    /// samples composite onto (`init: accumulator = startcolor`, doc §3). Default
    /// `[0,0,0]` (black) — the prior hardcoded behaviour, so a black start reproduces
    /// it byte-for-byte. A white start `[1,1,1]` is what lets `multiply` (dead on a
    /// black, absorbing start) build the inverse dark-lace-on-light look. JSON-encoded
    /// as a `#rrggbb` sRGB hex string (or the names `black`/`white`).
    pub start_color: [f64; 3],
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
    /// Reverse the palette LUT direction (field-independent flip about the seam,
    /// matching [`crate::palette::Palette::from_srgb8_stops_mirrored`]'s `reverse`).
    /// The one approved colormap knob the beautiful path lacked; needed so a
    /// `reverse=true` approved coloring (e.g. Winner B) renders faithfully through
    /// `--coloring` on the composite modes, which cannot go through the Python
    /// dump→recolor tail. Default `false` (byte-identical to the prior path).
    pub reverse: bool,
    // --- highlight rolloff (screen-family blowout recovery) ---
    /// Luminance-domain highlight compression applied to the final linear RGB, before
    /// downsample/sRGB. Gated to the clipping-prone screen family (composite
    /// `combine=screen` and `direct_trap` `merge_mode=screen`): only those two paths
    /// consult it. Default [`Rolloff::None`] (identity → byte-identical to the prior
    /// path). See [`Rolloff`].
    pub rolloff: Rolloff,
    /// The rolloff operator's one knob (white point / exposure / knee — see the
    /// [`Rolloff`] variant docs). Default `1.0`.
    pub rolloff_strength: f64,
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
            gaussint_color_by: GaussianColorBy::MinimumDistance,
            de_scale: 1.0,
            direct_threshold: 0.1,
            direct_opacity: 0.5,
            merge_mode: MergeMode::Normal,
            merge_order: MergeOrder::BottomUp,
            direct_shape: DirectShape::Cross,
            start_color: [0.0, 0.0, 0.0],
            transform: Transform::Linear,
            gamma: 1.0,
            shade: Shade::None,
            light_azimuth: std::f64::consts::FRAC_PI_4, // 45°
            light_height: 1.0,
            palette_cycles: 1.0,
            palette_offset: 0.0,
            reverse: false,
            rolloff: Rolloff::None,
            rolloff_strength: 1.0,
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
    /// stripe — Matt's validated sweep verdict — as well as Smooth/Curvature and
    /// the UF-mode/Tia/Velocity seeds; `Sqrt` otherwise). All other knobs stay at
    /// defaults; tune via JSON overrides.
    pub fn beautiful(field: Field) -> Self {
        let transform = match field {
            Field::TrapCircle | Field::TrapCross => Transform::Log,
            // Validated stripe default (sweep verdict): linear, not sqrt.
            Field::Stripe => Transform::Linear,
            // Validated tia default (sweep verdict): linear, not sqrt.
            Field::Tia => Transform::Linear,
            // DE seed (§1): log — the soft-mapped distance still has wide dynamic
            // range, and log is the sane viewing default until palette spacing exists.
            Field::De => Transform::Log,
            // Velocity seed: linear (identity spacing) — raw mean step length, no
            // retuned spacing yet. We can revisit once we've seen the interior.
            Field::Velocity => Transform::Linear,
            // UF-mode seeds: linear (the prompt's explicit per-new-field seed). The
            // percentile-stretch handles spacing; DirectTrap ignores the transform
            // (it bypasses the scalar normalize stage entirely).
            Field::GaussianInt
            | Field::ExpSmoothing
            | Field::Decomposition
            | Field::DirectTrap => Transform::Linear,
            // Smooth + Curvature default to linear (specs pinned this by hand
            // until now; make it the source default). With these two added the
            // match is exhaustive — no field falls through to Sqrt anymore (Sqrt
            // survives only as an explicit per-spec transform).
            Field::Smooth | Field::Curvature => Transform::Linear,
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
        // Validated DE default (flat-field sweep sweet spot): de_scale 0.25 — it
        // needed spread (a wider glow band) over the sentinel's 1.0.
        if matches!(field, Field::De) {
            p.de_scale = 0.25;
        }
        p
    }

    /// Serialize to a compact JSON object (all fields, stable key order).
    /// `texture_field` is emitted as `"none"` when absent (single-field).
    pub fn to_json(&self) -> String {
        format!(
            "{{\"bailout_b\":{},\"skip\":{},\"biomorph\":\"{}\",\"field\":\"{}\",\
             \"stripe_density\":{},\"trap_radius\":{},\"color_by\":\"{}\",\"de_scale\":{},\
             \"direct_threshold\":{},\"direct_opacity\":{},\"merge_mode\":\"{}\",\
             \"merge_order\":\"{}\",\"shape\":\"{}\",\"start_color\":\"{}\",\"transform\":\"{}\",\"gamma\":{},\
             \"shade\":\"{}\",\"light_azimuth\":{},\"light_height\":{},\
             \"palette_cycles\":{},\"palette_offset\":{},\"reverse\":{},\
             \"rolloff\":\"{}\",\"rolloff_strength\":{},\
             \"texture_field\":\"{}\",\"texture_transform\":\"{}\",\"texture_gamma\":{},\
             \"combine\":\"{}\",\"texture_weight\":{}}}",
            self.bailout_b,
            self.skip,
            self.biomorph.as_str(),
            self.field.as_str(),
            self.stripe_density,
            self.trap_radius,
            self.gaussint_color_by.as_str(),
            self.de_scale,
            self.direct_threshold,
            self.direct_opacity,
            self.merge_mode.as_str(),
            self.merge_order.as_str(),
            self.direct_shape.as_str(),
            start_color_to_hex(self.start_color),
            self.transform.as_str(),
            self.gamma,
            self.shade.as_str(),
            self.light_azimuth,
            self.light_height,
            self.palette_cycles,
            self.palette_offset,
            self.reverse,
            self.rolloff.as_str(),
            self.rolloff_strength,
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
            if let Some(v) = jsonl::field_str(s, "color_by") {
                p.gaussint_color_by = GaussianColorBy::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "de_scale") {
                p.de_scale = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "direct_threshold") {
                p.direct_threshold = v;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "direct_opacity") {
                p.direct_opacity = v;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "merge_mode") {
                p.merge_mode = MergeMode::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "merge_order") {
                p.merge_order = MergeOrder::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "shape") {
                p.direct_shape = DirectShape::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "start_color") {
                p.start_color = parse_start_color(&v)?;
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
            if let Some(v) = jsonl::field_bool(s, "reverse") {
                p.reverse = v;
                named = true;
            }
            if let Some(v) = jsonl::field_str(s, "rolloff") {
                p.rolloff = Rolloff::parse(&v)?;
                named = true;
            }
            if let Some(v) = jsonl::field_f64(s, "rolloff_strength") {
                p.rolloff_strength = v;
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

/// Gaussian-integer lattice-trap accumulator (`round`, `N=1`) over the full orbit:
/// the running statistics every "Color By" mode reduces from (doc §1). `zmin`/`zmax`
/// are the iterates at the closest/farthest lattice approach (for the angle modes);
/// `total`/`count` give the running mean `rave`.
#[derive(Clone, Copy, Debug)]
pub struct GaussTrap {
    /// `rmin` — closest lattice distance over the orbit.
    pub rmin: f64,
    /// `zmin` — iterate at the closest approach.
    pub zmin: Complex<f64>,
    /// `itermin` — iteration index of the closest approach.
    pub itermin: u32,
    /// `rmax` — farthest lattice distance over the orbit.
    pub rmax: f64,
    /// `zmax` — iterate at the farthest approach.
    pub zmax: Complex<f64>,
    /// `itermax` — iteration index of the farthest approach.
    pub itermax: u32,
    /// `total = Σ r` — sum of lattice distances (→ `rave = total/count`).
    pub total: f64,
    /// `count` — number of orbit points accumulated.
    pub count: u32,
}

impl GaussTrap {
    /// Reduce to one "Color By" index. Iteration modes are folded mod 1 here (they
    /// bypass the percentile-stretch downstream); distance/ratio/angle modes return
    /// their raw value. `None` only on an empty orbit (`count == 0`).
    fn index(&self, color_by: GaussianColorBy) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let rave = self.total / self.count as f64;
        use GaussianColorBy::*;
        Some(match color_by {
            MinimumDistance => self.rmin,
            AverageDistance => rave,
            MaximumDistance => self.rmax,
            IterMin => (0.01 * self.itermin as f64).rem_euclid(1.0),
            IterMax => (0.01 * self.itermax as f64).rem_euclid(1.0),
            AngleMin => normalize_angle(self.zmin),
            AngleMax => normalize_angle(self.zmax),
            MeanAngle => normalize_angle(Complex::new(rave - self.rmin, self.rmax - rave)),
            Ratio => self.rmax / (self.rmin + 1e-12),
        })
    }
}

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
    /// Gaussian-integer unit-lattice trap statistics over the orbit (every Color By
    /// mode reduces from this; see [`GaussTrap`]).
    gauss: GaussTrap,
    /// Exponential-smoothing accumulator: `(Σ exp(−|zₙ|), count)` over the orbit.
    /// Escaped-gated at reduction (divergent branch only).
    exp_sum: (f64, u32),
    /// Discrete-velocity accumulator: `(Σ|z_{n+1}−z_n|, count)` over the full
    /// orbit (interior included). Reduced to the mean step length by `field`.
    velocity: (f64, u32),
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
            // Mean discrete velocity over the full orbit. Returned unconditionally
            // (interior-valued); `max(1)` guards the empty-orbit divide.
            Field::Velocity => Some(self.velocity.0 / self.velocity.1.max(1) as f64),
            // DE needs the pixel scale (+ de_scale) to normalize, which `field` has
            // no access to — it is reduced via [`Self::de_value`] in the render stage.
            Field::De => None,
            // Gaussian-integer lattice trap: the canonical min-distance reduction
            // (interior-valued, like the traps). The Color-By-aware single-field path
            // routes through [`Self::gaussint_value`] instead; this default keeps the
            // composite/texture path (which has no Color By knob) on `minimum_distance`.
            Field::GaussianInt => self.gauss.index(GaussianColorBy::MinimumDistance),
            // Exponential smoothing: the raw sum (divergescale = 1.0), escaped-gated
            // like the averaging family. `None` on a non-escaping / empty orbit.
            Field::ExpSmoothing => {
                let (sum, count) = self.exp_sum;
                (self.escaped && count > 0).then_some(sum)
            }
            // Decomposition: escape-point angle atan2(z_final) folded to [0,1).
            // Escaped only (no escape point on a bounded orbit).
            Field::Decomposition => self.escaped.then(|| {
                (self.z.im.atan2(self.z.re) / std::f64::consts::TAU).rem_euclid(1.0)
            }),
            // Direct orbit traps are colour-valued — handled by `render_direct_trap`,
            // never reduced to a scalar here.
            Field::DirectTrap => None,
        }
    }

    /// Gaussian-integer trap reduced under a chosen "Color By" mode (doc §1). Routes
    /// to [`GaussTrap::index`]; the single-field render path calls this (instead of
    /// the default [`Self::field`] reduction) so all nine modes are reachable.
    /// Interior-valued (the lattice trap fills the interior); `None` only on an empty
    /// orbit. Iteration modes arrive folded mod 1 and are rendered direct (the caller
    /// bypasses the percentile-stretch for them — see [`GaussianColorBy::is_iteration`]).
    pub fn gaussint_value(&self, color_by: GaussianColorBy) -> Option<f64> {
        self.gauss.index(color_by)
    }

    /// Exterior distance estimate, rendered to a `[0,1]` field value (the `de`
    /// field). `de = |z|·ln|z|/|dz|` is the boundary distance in complex-plane
    /// units (reusing the `normal_map` derivative `dz`); dividing by `pixel_size`
    /// (`fw/width`) makes the thickness **zoom-invariant** (distance in output
    /// pixels), and `tanh(de_px · de_scale)` is the standard soft DE ramp — `0` at
    /// the boundary, `→1` in the open exterior, with `de_scale` setting filament
    /// thickness. `None` for interior / non-finite orbits (→ black, like the
    /// exterior-only fields). The downstream percentile-stretch absorbs any *linear*
    /// scale, so the nonlinear `tanh` (not a bare `de_px·de_scale`) is what makes
    /// `de_scale` a real, render-visible knob.
    pub fn de_value(&self, pixel_size: f64, de_scale: f64) -> Option<f64> {
        if !self.escaped {
            return None;
        }
        let zabs = self.z.norm();
        let dzabs = self.dz.norm();
        let lnz = zabs.ln();
        if !(zabs.is_finite() && dzabs.is_finite() && lnz.is_finite() && dzabs > 0.0 && lnz > 0.0) {
            return None;
        }
        let de_px = (zabs * lnz / dzabs) / pixel_size;
        if !de_px.is_finite() {
            return None;
        }
        Some((de_px * de_scale).tanh())
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

/// Iterate one orbit, accumulating the full channel union. `z0` is `0` for the
/// parameter-plane families (`c` = pixel) and the pixel for the dynamical families
/// (`c` = fixed parameter); the `z'` recurrence differs accordingly (`dz0 = 0`,
/// `+1` for parameter plane; `dz0 = 1`, no `+1` for dynamical — doc §0). `family`
/// selects the recurrence: `z^d + c` for Mandelbrot/Julia/Multibrot (degree 2 or
/// `d ∈ {3,4,5}`, by repeated complex multiplication), or the two-state Phoenix
/// `z_{n+1} = z_n² + c + p·z_{n-1}`. Order matches [`crate::backend::F64Backend`]:
/// `z'` updates from `zₙ` before `z` advances.
///
/// **Degree byte-identity:** for degree 2 the `z^d`/`dz` ops and the smooth base
/// reduce to exactly the prior Mandelbrot/Julia float sequence (`z^1 = z`,
/// `z^2 = z·z`, base [`LN_2`](std::f64::consts::LN_2)), so a degree-2 render is
/// bit-for-bit unchanged; only `d ∈ {3,4,5}` take the new terms.
// The orbit-history bindings (`zprev2`, `escaped`) are initialized then
// unconditionally overwritten on the first loop pass before they're read — the
// loop always runs ≥1 iteration. That's the carried-history idiom, not a bug.
#[allow(unused_assignments)]
pub fn iterate_orbit(
    z0: Complex<f64>,
    c: Complex<f64>,
    maxiter: u32,
    params: &ColoringParams,
    family: Family,
) -> OrbitAccum {
    let b = params.bailout_b;
    let b2 = b * b;
    let skip = params.skip;
    let s_density = params.stripe_density;
    let r = params.trap_radius;
    let cabs = c.norm();

    let degree = family.degree();
    let dynamical = family.is_dynamical();
    // Phoenix's second constant `p` (the z_{n-1} coefficient); zero / unused for the
    // z^d families.
    let phoenix_p = match family {
        Family::Phoenix { p, .. } => p,
        _ => Complex::new(0.0, 0.0),
    };

    let mut z = z0;
    let mut dz = if dynamical {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(0.0, 0.0)
    };
    // Phoenix two-state: z_{n-1} and dz_{n-1}, both seeded 0 (z_{-1} = 0). Distinct
    // from the curvature history below (which seeds z0) so the first Phoenix step
    // couples against z_{-1} = 0, not z_0.
    let mut ph_zprev = Complex::new(0.0, 0.0);
    let mut ph_dzprev = Complex::new(0.0, 0.0);
    // Orbit history for curvature: zprev1 = zₙ₋₁, zprev2 = zₙ₋₂.
    let mut zprev1 = z0;
    let mut zprev2 = Complex::new(0.0, 0.0);

    let mut stripe = (0.0f64, 0u32, 0.0f64);
    let mut tia = (0.0f64, 0u32, 0.0f64);
    let mut curv = (0.0f64, 0u32, 0.0f64);
    let mut trap_circle_min = f64::INFINITY;
    let mut trap_cross_min = f64::INFINITY;
    // Gaussian-integer lattice-trap statistics over the orbit (all Color By modes).
    let mut g_rmin = f64::INFINITY;
    let mut g_zmin = Complex::new(0.0, 0.0);
    let mut g_itermin = 0u32;
    let mut g_rmax = 0.0f64;
    let mut g_zmax = Complex::new(0.0, 0.0);
    let mut g_itermax = 0u32;
    let mut g_total = 0.0f64;
    let mut g_count = 0u32;
    // Exponential-smoothing accumulator (Σ exp(−|z|), count) over the orbit.
    let mut exp_sum = (0.0f64, 0u32);
    // Discrete-velocity accumulator (Σ step length, count) over the full orbit.
    let mut velocity = (0.0f64, 0u32);

    let mut n = 0u32;
    let mut escaped = false;
    let mut smooth = 0.0f64;

    loop {
        // z' first (uses zₙ), then z advances. |zₙ²| feeds tia's lo/hi (kept for
        // every family — tia is a niche exterior field, not a degree-correctness
        // target; only the smooth base tracks the degree).
        let zn_sq = z * z;
        let zn_sq_abs = zn_sq.norm(); // |zₙ²| = |z_prev²| for tia

        // Recurrence + derivative, per family. Degree 2 reduces to the prior
        // `z·z + c` / `2·z·z' (+1)` float sequence exactly (z^{d-1} = z, so
        // `cpow_deriv` returns `(z·z, z)`); d ∈ {3,4,5} take the base-d power and
        // degree-scaled derivative. Phoenix carries the two-state z_{n-1} coupling.
        let (z_next, dz_next) = match family {
            Family::Phoenix { .. } => {
                // z_{n+1} = z_n² + c + p·z_{n-1}; z'_{n+1} = 2·z_n·z'_n + p·z'_{n-1}
                // (dynamical — no +1). Shift the two-state after reading it.
                let z_next = zn_sq + c + phoenix_p * ph_zprev;
                let dz_next = Complex::new(2.0, 0.0) * z * dz + phoenix_p * ph_dzprev;
                ph_zprev = z;
                ph_dzprev = dz;
                (z_next, dz_next)
            }
            _ => {
                // z^d and z^{d-1} by repeated complex multiplication (exact, fast).
                let (zd, zd_minus_1) = cpow_deriv(z, degree);
                let z_next = zd + c;
                // dz' = d·z^{d-1}·z' (+1 on the parameter plane only). Branch on the
                // `+1` (not `+ 0`) so the dynamical arm never adds a `+0.0` that
                // could flip a signed zero vs the prior degree-2 Julia bytes.
                let core = Complex::new(degree as f64, 0.0) * zd_minus_1 * dz;
                let dz_next = if dynamical {
                    core
                } else {
                    core + Complex::new(1.0, 0.0)
                };
                (z_next, dz_next)
            }
        };

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
        // Gaussian-integer trap: distance to the nearest unit-lattice point
        // (N = 1), q = round(z). Track rmin/zmin/itermin, rmax/zmax/itermax, and the
        // running total/count over the whole orbit (the Color By accumulators).
        let q = Complex::new(z.re.round(), z.im.round());
        let gi = (z - q).norm();
        g_total += gi;
        g_count += 1;
        if gi < g_rmin {
            g_rmin = gi;
            g_zmin = z;
            g_itermin = n;
        }
        if gi > g_rmax {
            g_rmax = gi;
            g_zmax = z;
            g_itermax = n;
        }
        // Exponential smoothing: Σ exp(−|z|) over the orbit (escaped-gated at
        // reduction). Accumulated every iteration, independent of skip.
        exp_sum.0 += (-zabs).exp();
        exp_sum.1 += 1;

        // Discrete velocity: |z_{n+1} − z_n|. After the advance above `zprev1`
        // holds z_n and `z` holds z_{n+1}, so this is the step just taken. Runs
        // every iteration (full bounded orbit for interior points), unconditional.
        velocity.0 += (z - zprev1).norm();
        velocity.1 += 1;

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
            smooth = smooth_value(n, zabs2, b, degree);
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
        gauss: GaussTrap {
            rmin: g_rmin,
            zmin: g_zmin,
            itermin: g_itermin,
            rmax: g_rmax,
            zmax: g_zmax,
            itermax: g_itermax,
            total: g_total,
            count: g_count,
        },
        exp_sum,
        velocity,
    }
}

/// `nu = (n+1) − log_d(ln|z| / ln B)` — the **bailout-normalized** smooth iteration,
/// which → 0 at the escape boundary (`|z| = B`). Its fraction is therefore the
/// correct deband weight ([`field`](OrbitAccum::field) / [`deband`]).
///
/// For the *smooth field itself* the `−log_d(ln B)` term is an additive constant the
/// percentile-stretch absorbs (so the smooth render is invariant to it — verified
/// by pixel-diff against the un-normalized formula). But in the **deband weight**
/// path the constant is load-bearing: omitting it (the previous shortcut) phase-
/// shifts the lerp by `log_d(ln B) mod 1` and terraces the bands. We normalize here
/// once so both paths share the correct value. `B` is threaded from
/// `params.bailout_b` — never hardcoded — so the normalization tracks the bailout.
///
/// **Outer log base = `ln d`, the degree correctness landmine.** Near escape
/// `|z_{n+1}| ≈ |z_n|^d`, so the double-log base is the family degree, not always 2.
/// Only this base changes with degree; every other term (the `+1`, the `ln B`
/// normalization) is identical. Degree 2 uses the exact [`LN_2`](std::f64::consts::LN_2)
/// constant (not `2.0.ln()`), so the degree-2 output is bit-for-bit the prior value;
/// `d ∈ {3,4,5}` use `(d as f64).ln()` and converge instead of banding.
#[inline]
fn smooth_value(n: u32, zabs2: f64, bailout_b: f64, degree: u32) -> f64 {
    let log_zn = 0.5 * zabs2.ln(); // ln|z|
    let log_b = bailout_b.ln(); // ln B
    let ratio = log_zn / log_b;
    // Degree 2 pins the exact LN_2 const so d=2 is byte-identical to before.
    let ln_d = if degree == 2 {
        std::f64::consts::LN_2
    } else {
        (degree as f64).ln()
    };
    if log_zn > 0.0 && log_b > 0.0 && ratio.is_finite() && ratio > 0.0 {
        (n + 1) as f64 - ratio.ln() / ln_d
    } else {
        (n + 1) as f64
    }
}

/// `(z^d, z^{d-1})` by repeated complex multiplication, for `d ≥ 2`. Exact and fast
/// (no `powf`/polar). Returns the derivative-power `z^{d-1}` alongside `z^d` so the
/// caller's `dz` recurrence (`d·z^{d-1}·z'`) shares the work. For `d = 2` this is
/// `(z·z, z)` — the exact float sequence the degree-2 path had inline.
#[inline]
fn cpow_deriv(z: Complex<f64>, d: u32) -> (Complex<f64>, Complex<f64>) {
    let mut zd_minus_1 = z; // z^1
    for _ in 2..d {
        zd_minus_1 *= z; // build up to z^{d-1}
    }
    let zd = zd_minus_1 * z; // z^d
    (zd, zd_minus_1)
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
/// `family` selects the recurrence and how the viewport seeds it (see [`Family`]):
/// dynamical families (Julia/Phoenix) address the z-plane with `z0 = pixel`;
/// parameter-plane families (Mandelbrot/Multibrot) sweep `c = pixel`.
/// Grid-centered supersampling (the rgss/jitter
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
    family: Family,
    params: &ColoringParams,
    palette: &Palette,
    filter: DownsampleFilter,
) -> image::RgbImage {
    // Direct orbit traps are colour-valued (composite-during-iteration), not a
    // scalar field — they take their own output path, ignoring texture/normalize.
    if params.field == Field::DirectTrap {
        return render_direct_trap(frame, ss, maxiter, family, params, palette, filter);
    }
    // Fast smooth path: a smooth render on a **new** family (Multibrot,
    // Julia-multibrot at degree ≥ 3, or Phoenix) sources its scalar from the fast
    // escape-time backend smooth channel instead of the slow degree-parametric
    // beautiful kernel (~20-45x faster, and it doesn't blow up with depth). Correct
    // for smooth mode only — colour is palette-over-smooth-scalar, so the
    // f64↔beautiful constant offset is absorbed by the percentile-stretch / histeq
    // normalization and the crop is visually equivalent. Gated OFF the degree-2
    // Mandelbrot/Julia families so their beautiful path stays byte-identical; any
    // knob this path can't honor (texture composite, normal_map emboss, biomorph
    // escape) falls through to the beautiful kernel below.
    let fast_smooth_family = match family {
        Family::Multibrot { .. } | Family::Phoenix { .. } => true,
        // Julia-multibrot (d ≥ 3) fast-routes; quadratic Julia (d = 2) stays on the
        // byte-identical beautiful path.
        Family::Julia { degree, .. } => degree >= 3,
        Family::Mandelbrot => false,
    };
    if fast_smooth_family
        && params.field == Field::Smooth
        && params.texture_field.is_none()
        && params.shade != Shade::NormalMap
        && params.biomorph == Biomorph::Off
    {
        return render_smooth_f64_fast(frame, ss, maxiter, family, params, palette, filter);
    }
    if params.texture_field.is_some() {
        return render_beautiful_composite(frame, ss, maxiter, family, params, palette, filter);
    }
    render_beautiful_single(frame, ss, maxiter, family, params, palette, filter)
}

/// Fast smooth-mode colored render sourced from the escape-time
/// [`F64Backend`]/[`JuliaBackend`] smooth channel
/// ([`smooth_field_f64_supersampled`]) rather than the slow beautiful
/// [`iterate_orbit`] kernel. Reproduces [`render_beautiful_single`]'s smooth-mode
/// tail — same grid geometry, same percentile-stretch / histeq normalization, same
/// transform / gamma / palette-cycle shade — over the fast field. **Not**
/// byte-identical to the beautiful render: the backend smooth value is un-normalized
/// (differs from beautiful by the constant `ln(ln B)/ln d`), but at the shared
/// `bailout_b` the escape mask is identical and the constant shift is removed by the
/// normalization, so the colored crop is visually equivalent (see the fast-field doc
/// on [`smooth_field_f64_supersampled`]). Callers gate this to the smooth/no-texture/
/// no-normal_map/no-biomorph case in [`render_beautiful`]; the `normal_map` emboss and
/// biomorph escape have no fast-field analogue and stay on the beautiful path.
#[allow(clippy::too_many_arguments)]
fn render_smooth_f64_fast(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
    params: &ColoringParams,
    palette: &Palette,
    filter: DownsampleFilter,
) -> image::RgbImage {
    let s = ss.max(1);
    // Fast escape-time smooth field at the render's bailout (matches the beautiful
    // kernel's escape mask; the un-normalized value offset washes out under stretch).
    let (field, _sub_w, _sub_h) =
        smooth_field_f64_supersampled(frame, s, maxiter, family, params.bailout_b)
            .expect("render_smooth_f64_fast gated to families with an escape-time backend");

    // --- global normalization over valid (escaped, finite) values ---
    // NaN encodes interior / non-escaped, exactly as `render_beautiful_single`.
    let mut valids: Vec<f64> =
        field.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();

    // Smooth is never an iteration Color-By field, so no `direct_map` branch — this is
    // strictly `render_beautiful_single`'s histeq-or-stretch reduction.
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

    let cycles = params.palette_cycles;
    let offset = params.palette_offset;
    let transform = params.transform;
    let gamma = params.gamma;

    // --- shade each subpixel to linear RGB (no emboss — gated out) ---
    let linear: Vec<[f64; 3]> = field
        .par_iter()
        .map(|&v| {
            if !v.is_finite() {
                return [0.0, 0.0, 0.0];
            }
            let value = v as f64;
            let x = match &sorted {
                Some(tab) => {
                    if tab.len() <= 1 {
                        0.0
                    } else {
                        let rank = tab.partition_point(|&t| t < value);
                        rank as f64 / (tab.len() - 1) as f64
                    }
                }
                None => ((value - lo) / span).clamp(0.0, 1.0),
            };
            let gray = apply_transform(x, transform, gamma);
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

/// Compute the **raw smooth scalar field** over the supersampled grid — the most
/// upstream scalar the smooth render consumes, *before* percentile-stretch,
/// transform, gamma, shade, and palette. Interior / non-escaped subpixels are
/// `f32::NAN` (the mask rides in the data). Row-major, length `(out_h·ss)·(out_w·ss)`.
///
/// This is the serialization source for the field⊗Python-coloring split
/// (`render-one --dump-field`): the returned values are **bit-for-bit** the same
/// `Field::Smooth` scalars [`render_beautiful_single`] reduces per subpixel (same
/// grid geometry, same [`iterate_orbit`], same [`OrbitAccum::field`] reduction) —
/// so a Python coloring tail fed this field reproduces the Rust smooth render.
/// `params` supplies only the iterate-stage knobs the smooth field reads
/// (`bailout_b`, `biomorph`); the field is fixed to `Smooth` regardless of
/// `params.field`. Returns `(field, sub_w, sub_h)`.
pub fn smooth_field_supersampled(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
    params: &ColoringParams,
) -> (Vec<f32>, u32, u32) {
    let s = ss.max(1);
    let sub_w = (frame.out_width * s) as usize;
    let sub_h = (frame.out_height * s) as usize;
    let fw = frame.frame_width;
    let fh = frame.frame_height();
    let sub_w_f = sub_w as f64;
    let sub_h_f = sub_h as f64;
    let center = frame.center;

    // Grid geometry identical to `render_beautiful_single` (grid-centered SS).
    let rows: Vec<Vec<f32>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                let (z0, c) = family.seed(pixel);
                let acc = iterate_orbit(z0, c, maxiter, params, family);
                // NaN encodes interior / non-escaped (Field::Smooth is exterior-only).
                let v = acc.field(Field::Smooth).map_or(f32::NAN, |x| x as f32);
                row.push(v);
            }
            row
        })
        .collect();

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    (flat, sub_w as u32, sub_h as u32)
}

/// Compute the **raw scalar field of an arbitrary single coloring mode** over the
/// supersampled grid — the general-field twin of [`smooth_field_supersampled`], for
/// the field⊗Python-coloring split's non-smooth keeper modes (`tia`, `stripe`,
/// `curvature`, `trap_circle`, …). Same grid geometry, same [`iterate_orbit`]
/// accumulate pass, same `NaN`-encodes-interior / row-major layout — it differs only
/// in the per-subpixel reduction: `params.field` instead of a hardcoded
/// [`Field::Smooth`]. `de`/`gaussian_int` route through their pixel-/Color-By-aware
/// reducers ([`OrbitAccum::de_value`] / [`OrbitAccum::gaussint_value`]) exactly as
/// the composite path's `eval` closure, so every scalar field dumps faithfully.
///
/// Because the reduction is the *same* one [`render_beautiful_single`] applies before
/// its percentile-stretch, a Python coloring tail fed this field reproduces the Rust
/// single-field render (up to the shared stretch/transform, which the tail mirrors) —
/// and inherits the full colormap param set (reverse / transfer / log_premap /
/// n_cycles / phase) for free. [`Field::DirectTrap`] is colour-valued (no scalar
/// reduction) and rejected by the caller before this is reached; on the off chance it
/// arrives, its subpixels fall to `NaN`. Returns `(field, sub_w, sub_h)`.
pub fn single_field_supersampled(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
    params: &ColoringParams,
) -> (Vec<f32>, u32, u32) {
    let s = ss.max(1);
    let sub_w = (frame.out_width * s) as usize;
    let sub_h = (frame.out_height * s) as usize;
    let fw = frame.frame_width;
    let fh = frame.frame_height();
    let sub_w_f = sub_w as f64;
    let sub_h_f = sub_h as f64;
    let center = frame.center;
    let the_field = params.field;
    // pixel scale for the `de` reducer (zoom-invariant filament thickness).
    let pixel_size = fw / frame.out_width as f64;

    // Grid geometry identical to `smooth_field_supersampled` (grid-centered SS).
    let rows: Vec<Vec<f32>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                let (z0, c) = family.seed(pixel);
                let acc = iterate_orbit(z0, c, maxiter, params, family);
                // Same per-field reduction the composite `eval` closure uses; `de` and
                // `gaussian_int` need their pixel-/Color-By-aware reducers.
                let v = match the_field {
                    Field::De => acc.de_value(pixel_size, params.de_scale),
                    Field::GaussianInt => acc.gaussint_value(params.gaussint_color_by),
                    _ => acc.field(the_field),
                };
                // NaN encodes interior / non-escaped (the seam every dumped field writes).
                row.push(v.map_or(f32::NAN, |x| x as f32));
            }
            row
        })
        .collect();

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    (flat, sub_w as u32, sub_h as u32)
}

/// Fast smooth-field twin of [`smooth_field_supersampled`], sourced from the
/// **escape-time [`F64Backend`]** (Mandelbrot degree 2 + Multibrot degree ≥ 3) /
/// [`JuliaBackend`] (Julia) rather than the generic beautiful [`iterate_orbit`]
/// kernel. Multibrot dispatches through the trait `sample` (→ `sample_multibrot`);
/// the degree-2 Mandelbrot path is textually untouched. Same grid geometry, same
/// row-major layout, same `NaN`-encodes-interior convention — the field array is a
/// drop-in for the beautiful one for **mask/statistic** consumers (the degenerate-
/// outcome guard reads only `interior_frac` = NaN fraction and `field_std` = std
/// over the escaped pixels).
///
/// It is **not** byte-identical to [`smooth_field_supersampled`]: the backend's
/// smooth value is un-normalized (`(n+1) − ln(ln|z|)/ln d`) while the beautiful
/// one carries the bailout normalization (`… − ln(ln|z|/ln B)/ln d`), so the two
/// fields differ by the constant `ln(ln B)/ln d`. A standard deviation is invariant
/// to that constant shift, and the escape mask is bailout-driven, not magnitude-
/// driven — so both guard statistics carry over (proven by the guard's calibration
/// control). Do **not** feed this to the field⊗colormap reproduction path, which
/// requires the byte-identical beautiful field.
///
/// **Every** family now has an escape-time f64 backend: Mandelbrot (degree 2) +
/// Multibrot (`F64Backend`), quadratic + Julia-multibrot (`JuliaBackend`), and
/// Ushiki Phoenix (`PhoenixBackend`, quadratic-dominated escape). The `Result` is
/// retained for API stability but no longer errors on any family. `bailout` is the
/// backend's escape radius (render-one passes its `1e6` production constant).
/// Returns `(field, sub_w, sub_h)`.
pub fn smooth_field_f64_supersampled(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
    bailout: f64,
) -> Result<(Vec<f32>, u32, u32), String> {
    // The trap is unused (the smooth-only fast path computes no trap channel), but
    // the backend constructors require one.
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
    enum Fast {
        Mandel(F64Backend),
        // Degree ≥ 3 parameter-plane multibrot; dispatched through the trait
        // `sample` (→ `sample_multibrot`), not the const-generic degree-2 kernel.
        Multi(F64Backend),
        // Any-degree dynamical Julia (`z^d + c`): degree 2 → byte-identical quadratic
        // kernel, degree ≥ 3 → Julia-multibrot, both via the trait `sample`.
        Julia(JuliaBackend),
        // Ushiki Phoenix two-state (`z² + c + p·z_{n-1}`); quadratic-dominated escape.
        Phoenix(PhoenixBackend),
    }
    let fast = match family {
        Family::Mandelbrot => Fast::Mandel(F64Backend::new(maxiter, bailout, trap)),
        Family::Multibrot { degree } => {
            Fast::Multi(F64Backend::new_degree(maxiter, bailout, trap, degree))
        }
        Family::Julia { c, degree } => {
            Fast::Julia(JuliaBackend::new_degree(c, maxiter, bailout, trap, degree))
        }
        Family::Phoenix { c, p } => Fast::Phoenix(PhoenixBackend::new(c, p, maxiter, bailout, trap)),
    };

    let s = ss.max(1);
    let sub_w = (frame.out_width * s) as usize;
    let sub_h = (frame.out_height * s) as usize;
    let fw = frame.frame_width;
    let fh = frame.frame_height();
    let sub_w_f = sub_w as f64;
    let sub_h_f = sub_h as f64;
    let center = frame.center;

    // Grid geometry identical to `smooth_field_supersampled` (grid-centered SS).
    let rows: Vec<Vec<f32>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                // Both backends take the pixel as `c`; each applies its own z₀
                // (Mandelbrot 0, Julia = pixel). The smooth-only kernel skips
                // trap/atom/DE (all const-false). `dc` is unused by these shallow
                // f64 backends.
                let smp = match &fast {
                    Fast::Mandel(b) => b.sample_flags::<false, false, false, PHASE_DEFER>(pixel),
                    // Trait `sample` routes degree ≥ 3 through `sample_multibrot`.
                    Fast::Multi(b) => b.sample(pixel, Complex::new(0.0, 0.0)),
                    Fast::Julia(b) => b.sample(pixel, Complex::new(0.0, 0.0)),
                    Fast::Phoenix(b) => b.sample(pixel, Complex::new(0.0, 0.0)),
                };
                // NaN encodes interior / non-escaped (smooth is exterior-only), the
                // same seam `smooth_field_supersampled` writes.
                let v = if smp.escaped { smp.smooth_iter as f32 } else { f32::NAN };
                row.push(v);
            }
            row
        })
        .collect();

    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Ok((flat, sub_w as u32, sub_h as u32))
}

/// Source-side saturation cap for the `direct_trap_screen` mode (**cross shape +
/// `screen` merge only**). The additive screen accumulator asymptotes to white as
/// trap hits stack, and the cross shape `min(|Re z|,|Im z|)` traps a large fraction
/// of the orbit, so cross+screen drives fully to `(1,1,1)` once opacity/threshold
/// climb. A fully-saturated raster is **unrecoverable** post-hoc — the `soft_knee`
/// rolloff only rescues the partially-blown tier. Measured on worst-case saturating
/// locations (controlled per-location opacity sweep, threshold 0.08): hard-blowout
/// (min-channel > 0.996) is ~0% at opacity ≤ 0.08, ~6% at 0.15, then takes off —
/// 14% at 0.20, 39% at 0.30, 71% at 0.45. On the threshold axis at opacity 0.15:
/// 0% at ≤ 0.05, 6% at 0.08, 19% at 0.12. So the non-saturating regime is
/// **opacity ≤ 0.15, threshold ≤ 0.08** (the spec default 0.15 / 0.05 is the clean
/// corner: 0% hard-blowout even worst-case). Cross+screen is clamped to this regime
/// so the mode can never emit a blown raster; the clamp is byte-identical at/below
/// the cap. `ring`/`lines`/`multiply` don't saturate (fewer hits / non-brightening
/// merge) and are left untouched — low opacity is deploy-exploration guidance for
/// them, not a hard cap.
pub const DTS_SCREEN_OPACITY_CAP: f64 = 0.15;
pub const DTS_SCREEN_THRESHOLD_CAP: f64 = 0.08;

/// **Direct Orbit Traps** — the one colour-valued path (doc §"Direct Orbit Traps").
/// Unlike the scalar fields, this composites a gradient sample into a per-pixel
/// RGBA accumulator on every iteration the orbit enters the trap, and emits the
/// accumulator as `#color` directly — **no** percentile-stretch / histeq / palette-
/// index step. The accumulator is therefore already linear RGB and is downsampled
/// straight away (no [`ShadePix`] reduction, no global stat pass).
///
/// Per iteration (faithful UF config — `trapcenter = 0`, `rot = identity`, shape =
/// cross): `d = min(|Re z|,|Im z|)`; if `d < threshold`, sample the gradient at the
/// colour key `d/threshold` (Trap Color = distance), feather the alpha by the same
/// distance (`1 − d/threshold`, the "distance" merge modifier), and composite with
/// `normal` merge at `direct_opacity`. Order-dependent in iteration order, hence
/// deterministic. Black (`start color`) where the orbit never enters the trap.
#[allow(clippy::too_many_arguments)]
fn render_direct_trap(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
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
    let b = params.bailout_b;
    let b2 = b * b;
    let mode = params.merge_mode;
    let order = params.merge_order;
    let start_color = params.start_color;
    let shape = params.direct_shape;
    // Source-side saturation cap: clamp cross+screen (the `direct_trap_screen` mode)
    // to the confirmed non-saturating regime so it never emits an unrecoverable
    // (1,1,1) raster. Byte-identical at/below the cap; other shapes/merges untouched.
    // See [`DTS_SCREEN_OPACITY_CAP`] / [`DTS_SCREEN_THRESHOLD_CAP`].
    let (opacity_cap, threshold_cap) = if mode == MergeMode::Screen && shape == DirectShape::Cross {
        (DTS_SCREEN_OPACITY_CAP, DTS_SCREEN_THRESHOLD_CAP)
    } else {
        (1.0, f64::INFINITY)
    };
    let threshold = params.direct_threshold.max(1e-12).min(threshold_cap);
    let opacity = params.direct_opacity.clamp(0.0, 1.0).min(opacity_cap);
    let radius = params.trap_radius;
    // Catalog instrumentation (off in normal renders): when `FRACTAL_DT_STATS` is
    // set, also collect each pixel's *closest approach* (min trap distance over the
    // orbit) so the per-shape distance scale can be measured — see
    // [`DirectShape::default_threshold`]. Adds one `min` per iteration; no effect on
    // the rendered accumulator, so output stays byte-identical when off.
    let stats = std::env::var_os("FRACTAL_DT_STATS").is_some();

    // Iterate one orbit and composite the direct-trap accumulator → linear RGB. The
    // second return is the orbit's closest-approach distance (for `stats`).
    let degree = family.degree();
    let phoenix_p = match family {
        Family::Phoenix { p, .. } => p,
        _ => Complex::new(0.0, 0.0),
    };
    let composite = |z0: Complex<f64>, c: Complex<f64>| -> ([f64; 3], f64) {
        let mut z = z0;
        // Phoenix two-state (z_{n-1}, seeded 0); unused for the z^d families.
        let mut ph_zprev = Complex::new(0.0, 0.0);
        // Linear-RGB accumulator initialized to the `start color` background (doc §3).
        // Black (default) is the absorbing background the multiplicative modes darken
        // from; a white start lets `multiply` build dark lace down from light.
        let mut acc = start_color;
        let mut min_d = f64::INFINITY;
        let mut n = 0u32;
        loop {
            z = match family {
                Family::Phoenix { .. } => {
                    let z_next = z * z + c + phoenix_p * ph_zprev;
                    ph_zprev = z;
                    z_next
                }
                _ => cpow_deriv(z, degree).0 + c,
            };
            n += 1;
            // Trap distance of the iterate to the chosen shape (trapcenter 0, rot
            // identity, aspect 1 → z2 = z). `Cross` reproduces the prior baked
            // `z.re.abs().min(z.im.abs())` exactly.
            let d = shape.dist(z.re, z.im, radius);
            if d < min_d {
                min_d = d;
            }
            if d < threshold {
                let key = (d / threshold).clamp(0.0, 1.0);
                let src = palette.lookup_linear(key); // Trap Color = distance
                let feather = 1.0 - key; // distance merge modifier (feathering on)
                let a = opacity * feather;
                // Blend sample vs accumulator through the merge mode; merge order
                // picks which operand is the blend's back (`a`) vs front (`s`), then
                // alpha-over with the sample's α (held bottom-up for this sweep).
                for ch in 0..3 {
                    let blended = match order {
                        MergeOrder::BottomUp => mode.blend(acc[ch], src[ch]),
                        MergeOrder::TopDown => mode.blend(src[ch], acc[ch]),
                    };
                    acc[ch] = blended * a + acc[ch] * (1.0 - a);
                }
            }
            if n >= maxiter {
                break;
            }
            let zabs2 = z.norm_sqr();
            let bail = match params.biomorph {
                Biomorph::Off => zabs2 > b2,
                Biomorph::EpsilonCross => z.re.abs() > b || z.im.abs() > b,
            };
            if bail {
                break;
            }
        }
        (acc, min_d)
    };

    let rows: Vec<Vec<([f64; 3], f64)>> = (0..sub_h)
        .into_par_iter()
        .map(|srow| {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            let mut row = Vec::with_capacity(sub_w);
            for scol in 0..sub_w {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let pixel = Complex::new(center.re + dc_re, center.im + dc_im);
                let (z0, c) = family.seed(pixel);
                row.push(composite(z0, c));
            }
            row
        })
        .collect();
    // Highlight rolloff — gated to the additive `screen` merge (the accumulator that
    // asymptotes to white as trap hits stack). Off / byte-identical for every other
    // merge mode or when `rolloff == None`.
    let (rolloff, rolloff_strength) = if mode == MergeMode::Screen {
        (params.rolloff, params.rolloff_strength)
    } else {
        (Rolloff::None, 1.0)
    };
    let linear: Vec<[f64; 3]> = rows
        .iter()
        .flatten()
        .map(|(acc, _)| apply_rolloff(*acc, rolloff, rolloff_strength))
        .collect();

    if stats {
        let mut mins: Vec<f64> = rows
            .iter()
            .flatten()
            .map(|(_, d)| *d)
            .filter(|d| d.is_finite())
            .collect();
        mins.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |q: f64| -> f64 {
            if mins.is_empty() {
                return f64::NAN;
            }
            let i = ((mins.len() - 1) as f64 * q).round() as usize;
            mins[i]
        };
        let painted = mins.iter().filter(|d| **d < threshold).count();
        let frac = painted as f64 / mins.len().max(1) as f64;
        eprintln!(
            "[DT_STATS] shape={} thr={:.4} painted_frac={:.4} closest-approach: min={:.4e} p50={:.4e} p90={:.4e} p95={:.4e} p99={:.4e} max={:.4e}",
            shape.as_str(),
            threshold,
            frac,
            pct(0.0),
            pct(0.50),
            pct(0.90),
            pct(0.95),
            pct(0.99),
            pct(1.0),
        );
    }

    crate::render::downsample_linear_filtered(
        &linear,
        frame.out_width,
        frame.out_height,
        s,
        filter,
    )
}

/// The **v1 single-field** path — verbatim. Do not modify: its output bytes are a
/// reference the montage/SHA guard pins.
#[allow(clippy::too_many_arguments)]
fn render_beautiful_single(
    frame: &Frame,
    ss: u32,
    maxiter: u32,
    family: Family,
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
    let want_shade = params.shade == Shade::NormalMap;
    // Output-pixel size in the complex plane — the zoom-invariant unit the `de`
    // field normalizes against (`fw/width`). `field()` can't see this; `de_value`
    // takes it explicitly.
    let pixel_size = fw / frame.out_width as f64;

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
                // Parameter plane: z0 = 0, c = pixel. Dynamical: z0 = pixel, c = param.
                let (z0, c) = family.seed(pixel);
                let acc = iterate_orbit(z0, c, maxiter, params, family);
                let value = match params.field {
                    Field::De => acc.de_value(pixel_size, params.de_scale),
                    // Color-By-aware reduction (the default `field()` is min-distance only).
                    Field::GaussianInt => acc.gaussint_value(params.gaussint_color_by),
                    _ => acc.field(params.field),
                };
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

    // Iteration Color-By modes are pre-folded mod 1 and read directly off the
    // gradient — the `0.01·iter` banding must NOT be re-stretched (it would flatten
    // into a ramp). Identity normalization (lo=0, span=1, no histeq) passes the
    // folded value straight through (prompt §Normalization).
    let direct_map = params.field == Field::GaussianInt && params.gaussint_color_by.is_iteration();

    // Histeq builds a sorted table for rank lookup; the stretch transforms use
    // percentile bounds. Both are global reductions over the frame's valid values.
    let histeq = params.transform == Transform::Histeq && !direct_map;
    let (lo, span, sorted) = if direct_map {
        (0.0, 1.0, None)
    } else if histeq {
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
    family: Family,
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
    let want_shade = params.shade == Shade::NormalMap;
    let texture_field = params
        .texture_field
        .expect("composite branch requires texture_field");
    let pixel_size = fw / frame.out_width as f64;
    // Reduce one field, routing `de` through the pixel-aware estimator and
    // `gaussian_int` through the Color-By-aware reduction.
    let eval = |acc: &OrbitAccum, f: Field| -> Option<f64> {
        match f {
            Field::De => acc.de_value(pixel_size, params.de_scale),
            Field::GaussianInt => acc.gaussint_value(params.gaussint_color_by),
            _ => acc.field(f),
        }
    };

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
                let (z0, c) = family.seed(pixel);
                let acc = iterate_orbit(z0, c, maxiter, params, family);
                // One pass, two fields off the shared channel union.
                let bv = eval(&acc, params.field);
                let tv = eval(&acc, texture_field);
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
    // Highlight rolloff — gated to the screen combine (the blowout-prone op). Off for
    // every other combine (and byte-identical when `rolloff == None`).
    let (rolloff, rolloff_strength) = if combine == Combine::Screen {
        (params.rolloff, params.rolloff_strength)
    } else {
        (Rolloff::None, 1.0)
    };

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
            apply_rolloff(palette.lookup_linear(tt), rolloff, rolloff_strength)
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
            gauss: GaussTrap {
                rmin: 0.2,
                zmin: Complex::new(0.0, 0.0),
                itermin: 1,
                rmax: 0.5,
                zmax: Complex::new(0.0, 0.0),
                itermax: 2,
                total: 1.0,
                count: 5,
            },
            exp_sum: (1.5, 5),
            velocity: (5.0, 4),
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
        // Velocity is interior-valued (unconditional, like the trap fields): the
        // mean step length, returned even on a non-escaping orbit.
        assert_eq!(interior.field(Field::Velocity), Some(5.0 / 4.0));
        assert_eq!(acc.field(Field::Velocity), Some(5.0 / 4.0));
        // count == 0 → None even when escaped (no terms accumulated).
        let empty = OrbitAccum { stripe: (0.0, 0, 0.0), ..acc };
        assert_eq!(empty.field(Field::Stripe), None);
        // count == 1 → plain sum (deband needs ≥2 terms).
        let one = OrbitAccum { stripe: (0.42, 1, 0.42), ..acc };
        assert_eq!(one.field(Field::Stripe), Some(0.42));
    }

    /// The dumped smooth field has the supersampled dims, and its NaN mask is
    /// exactly the interior (non-escaped) mask — i.e. a NaN entry iff
    /// `iterate_orbit(...).field(Smooth)` is `None` at that grid point, and the
    /// finite entries equal that scalar (the pre-coloring value the render consumes).
    #[test]
    fn dump_field_geometry_and_nan_mask() {
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.02,
            out_width: 20,
            out_height: 12,
        };
        let ss = 2u32;
        let maxiter = 400u32;
        let params = ColoringParams::beautiful(Field::Smooth);
        let (field, w, h) = smooth_field_supersampled(&frame, ss, maxiter, Family::Mandelbrot, &params);
        assert_eq!((w, h), (40, 24));
        assert_eq!(field.len(), (w * h) as usize);

        // Recompute the reference per-subpixel scalar the same way the render does
        // and confirm the dump matches it bit-for-bit (NaN iff interior).
        let sub_w_f = w as f64;
        let sub_h_f = h as f64;
        let fw = frame.frame_width;
        let fh = frame.frame_height();
        let mut any_nan = false;
        let mut any_finite = false;
        for srow in 0..h as usize {
            let py = srow as f64 + 0.5;
            let dc_im = (0.5 - py / sub_h_f) * fh;
            for scol in 0..w as usize {
                let px = scol as f64 + 0.5;
                let dc_re = (px / sub_w_f - 0.5) * fw;
                let c = Complex::new(frame.center.re + dc_re, frame.center.im + dc_im);
                let acc = iterate_orbit(Complex::new(0.0, 0.0), c, maxiter, &params, Family::Mandelbrot);
                let got = field[srow * w as usize + scol];
                match acc.field(Field::Smooth) {
                    Some(v) => {
                        any_finite = true;
                        assert_eq!(got, v as f32, "finite mismatch at ({scol},{srow})");
                    }
                    None => {
                        any_nan = true;
                        assert!(got.is_nan(), "expected NaN at interior ({scol},{srow})");
                    }
                }
            }
        }
        assert!(any_finite, "patch had no escaped pixels");
        assert!(any_nan, "patch had no interior pixels (pick a mixed patch)");
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
        assert_eq!(p.transform, Transform::Linear);

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
            Family::Mandelbrot,
        );
        assert!(m.escaped && m.smooth.is_finite());
        // Julia exterior point (c = -0.8 + 0.156i, z0 far out).
        let j = iterate_orbit(
            Complex::new(2.0, 2.0),
            Complex::new(-0.8, 0.156),
            500,
            &p,
            Family::Julia { c: Complex::new(-0.8, 0.156), degree: 2 },
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
            Family::Mandelbrot,
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
        let a = render_beautiful(&frame, 2, 400, Family::Mandelbrot, &single, &palette, DownsampleFilter::Lanczos3);
        let b = render_beautiful(&frame, 2, 400, Family::Mandelbrot, &dressed, &palette, DownsampleFilter::Lanczos3);
        assert_eq!(a.into_raw(), b.into_raw());
    }

    /// DE seeds from `beautiful(De)`: 2^16 bailout, log transform, de_scale 0.25
    /// (the validated flat-field sweet spot), and round-trips through JSON
    /// (including the `de_scale` key).
    #[test]
    fn de_beautiful_and_roundtrip() {
        let p = ColoringParams::beautiful(Field::De);
        assert_eq!(p.field, Field::De);
        assert_eq!(p.bailout_b, BEAUTIFUL_BAILOUT);
        assert_eq!(p.transform, Transform::Log);
        assert_eq!(p.de_scale, 0.25);
        assert_eq!(p, ColoringParams::from_json(&p.to_json()).unwrap());
        // A partial spec seeds the De preset, then overlays explicit keys.
        let s = ColoringParams::from_json("{\"field\":\"de\",\"de_scale\":4.0}").unwrap();
        assert_eq!(s.transform, Transform::Log);
        assert_eq!(s.de_scale, 4.0);
        // `{}` is still the location sentinel after the de_scale addition.
        assert!(ColoringParams::from_json("{}").unwrap().is_location_profile());
    }

    /// `de_value` is exterior-only, lands in `[0,1)`, increases away from the
    /// boundary, and `field(De)` returns `None` (DE is reduced via `de_value`).
    #[test]
    fn de_value_exterior_and_monotone() {
        let p = ColoringParams::beautiful(Field::De);
        let ps = 1e-4; // a representative output-pixel size
        // Exterior point: escapes, finite de in [0,1).
        let ext = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(ext.escaped);
        assert!(ext.field(Field::De).is_none());
        let v = ext.de_value(ps, 1.0).expect("exterior de");
        assert!((0.0..=1.0).contains(&v), "de_value {v} out of [0,1]");
        // Interior point: no escape → None (black).
        let interior = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(!interior.escaped);
        assert!(interior.de_value(ps, 1.0).is_none());
        // tanh is monotone in de_scale·de_px, so a larger de_scale never lowers the
        // value for a fixed orbit (the knob genuinely moves output).
        let lo = ext.de_value(ps, 0.5).unwrap();
        let hi = ext.de_value(ps, 4.0).unwrap();
        assert!(hi >= lo);
    }

    /// Gaussian-integer trap on a known short orbit: `c = 1` gives the integer
    /// orbit 1, 2, 5, 26, … — every iterate is exactly a unit-lattice point, so the
    /// min lattice distance over the orbit is exactly 0.
    #[test]
    fn gaussian_int_known_orbit() {
        let p = ColoringParams::beautiful(Field::GaussianInt);
        assert_eq!(p.transform, Transform::Linear); // seeded linear per the prompt
        let acc = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            50,
            &p,
            Family::Mandelbrot,
        );
        let v = acc.field(Field::GaussianInt).expect("interior-valued");
        assert!(v.abs() < 1e-12, "integer orbit lattice distance {v} ≠ 0");
        // A generic point: min distance is in [0, √2/2] and finite, and is valid
        // even on a bounded (interior) orbit (the trap fills the interior).
        let interior = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        let iv = interior.field(Field::GaussianInt).expect("fills interior");
        assert!(iv.is_finite() && (0.0..=0.7072).contains(&iv));
    }

    /// Gaussian-integer Color-By modes: distance modes ordered `rmin ≤ rave ≤ rmax`,
    /// iteration modes folded into `[0,1)`, angle modes in `[0,1)`, ratio ≥ 1.
    #[test]
    fn gaussian_color_by_modes() {
        // A generic bounded orbit (interior-valued; the trap fills the interior).
        let p = ColoringParams::beautiful(Field::GaussianInt);
        let acc = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.3),
            500,
            &p,
            Family::Mandelbrot,
        );
        let g = |m| acc.gaussint_value(m).expect("interior-valued");
        let rmin = g(GaussianColorBy::MinimumDistance);
        let rave = g(GaussianColorBy::AverageDistance);
        let rmax = g(GaussianColorBy::MaximumDistance);
        assert!(rmin <= rave && rave <= rmax, "rmin {rmin} ≤ rave {rave} ≤ rmax {rmax}");
        assert!((0.0..=0.7072).contains(&rmin));
        // Iteration modes: folded into [0,1).
        for m in [GaussianColorBy::IterMin, GaussianColorBy::IterMax] {
            let v = g(m);
            assert!((0.0..1.0).contains(&v), "iter mode {} = {v} ∉ [0,1)", m.as_str());
            assert!(m.is_iteration());
        }
        // Angle modes: in [0,1).
        for m in [
            GaussianColorBy::AngleMin,
            GaussianColorBy::AngleMax,
            GaussianColorBy::MeanAngle,
        ] {
            let v = g(m);
            assert!((0.0..1.0).contains(&v), "angle mode {} = {v} ∉ [0,1)", m.as_str());
            assert!(!m.is_iteration());
        }
        // Ratio: rmax/(rmin+eps) ≥ 1.
        let ratio = g(GaussianColorBy::Ratio);
        assert!(ratio >= 1.0 - 1e-9, "ratio {ratio} < 1");
        // Default field() == minimum_distance.
        assert_eq!(acc.field(Field::GaussianInt), Some(rmin));
    }

    /// `color_by` survives JSON round-trip and seeds independently of the field default.
    #[test]
    fn gaussian_color_by_json_roundtrip() {
        let p = ColoringParams::from_json(
            "{\"field\":\"gaussian_int\",\"color_by\":\"mean_angle\"}",
        )
        .unwrap();
        assert_eq!(p.field, Field::GaussianInt);
        assert_eq!(p.gaussint_color_by, GaussianColorBy::MeanAngle);
        let p2 = ColoringParams::from_json(&p.to_json()).unwrap();
        assert_eq!(p, p2);
        // Default is minimum_distance.
        assert_eq!(
            ColoringParams::beautiful(Field::GaussianInt).gaussint_color_by,
            GaussianColorBy::MinimumDistance
        );
    }

    /// Decomposition is escaped-only and folds the escape-point angle into `[0,1)`.
    #[test]
    fn decomposition_angle_in_unit() {
        let p = ColoringParams::from_json("{\"field\":\"decomposition\",\"bailout_b\":4}").unwrap();
        // Exterior point escapes → angle in [0,1).
        let ext = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(ext.escaped);
        let a = ext.field(Field::Decomposition).expect("escaped angle");
        assert!((0.0..1.0).contains(&a), "decomposition angle {a} ∉ [0,1)");
        // Interior point: no escape angle → None.
        let interior = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(!interior.escaped);
        assert!(interior.field(Field::Decomposition).is_none());
    }

    /// Exponential smoothing accumulates a finite, positive sum on a divergent
    /// orbit and is escaped-gated (interior → None).
    #[test]
    fn exp_smoothing_sum_finite_and_gated() {
        let p = ColoringParams::beautiful(Field::ExpSmoothing);
        assert_eq!(p.transform, Transform::Linear);
        let ext = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(ext.escaped);
        let s = ext.field(Field::ExpSmoothing).expect("escaped sum");
        assert!(s.is_finite() && s > 0.0, "exp-smoothing sum {s} not finite-positive");
        // Each term is exp(−|z|) ∈ (0,1], so the sum is bounded by the count.
        assert!(s <= ext.exp_sum.1 as f64 + 1e-9);
        // Interior orbit: escaped-gated → None.
        let interior = iterate_orbit(
            Complex::new(0.0, 0.0),
            Complex::new(-0.2, 0.0),
            500,
            &p,
            Family::Mandelbrot,
        );
        assert!(!interior.escaped);
        assert!(interior.field(Field::ExpSmoothing).is_none());
    }

    /// Direct orbit traps: the colour-valued path emits non-background colour where
    /// the orbit enters the trap, and stays at the background (`start color` = black)
    /// where it doesn't. We drive "doesn't enter" by shrinking `direct_threshold` to
    /// effectively zero (no iterate satisfies `d < threshold` → all black); a real
    /// threshold lights pixels up.
    #[test]
    fn direct_trap_lights_only_on_entry() {
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.03,
            out_width: 24,
            out_height: 16,
        };
        let palette = crate::palette::builtin("default", false).unwrap();
        // Reachable trap (threshold 0.5): some pixels composite a gradient sample.
        let mut lit = ColoringParams::beautiful(Field::DirectTrap);
        lit.direct_threshold = 0.5;
        let img_lit = render_beautiful(&frame, 1, 300, Family::Mandelbrot, &lit, &palette, DownsampleFilter::Box);
        assert!(
            img_lit.pixels().any(|px| px.0 != [0, 0, 0]),
            "direct trap rendered all background despite a reachable trap"
        );
        // Unreachable trap (threshold ≈ 0): no iterate enters → pure background.
        let mut dark = ColoringParams::beautiful(Field::DirectTrap);
        dark.direct_threshold = 1e-300;
        let img_dark =
            render_beautiful(&frame, 1, 300, Family::Mandelbrot, &dark, &palette, DownsampleFilter::Box);
        assert!(
            img_dark.pixels().all(|px| px.0 == [0, 0, 0]),
            "direct trap lit pixels with an unreachable trap"
        );
    }

    /// Merge-mode blend formulas on a known accumulator/sample pair: screen
    /// brightens, multiply darkens, overlay matches the piecewise formula, normal
    /// returns the sample. Checked per the doc §3 backstop formulas.
    #[test]
    fn merge_mode_blend_formulas() {
        let a = 0.4; // accumulator (back)
        let s = 0.6; // sample (front)
        assert_eq!(MergeMode::Normal.blend(a, s), s);
        // multiply darkens below either operand.
        let m = MergeMode::Multiply.blend(a, s);
        assert!((m - 0.24).abs() < 1e-12);
        assert!(m < a && m < s, "multiply should darken");
        // screen brightens above either operand.
        let sc = MergeMode::Screen.blend(a, s);
        assert!((sc - (1.0 - 0.6 * 0.4)).abs() < 1e-12); // 1-(1-.4)(1-.6)=0.76
        assert!(sc > a && sc > s, "screen should brighten");
        // overlay: a<0.5 branch → 2*a*s.
        assert!((MergeMode::Overlay.blend(0.4, 0.6) - 2.0 * 0.4 * 0.6).abs() < 1e-12);
        // overlay: a>=0.5 branch → 1-2(1-a)(1-s).
        assert!(
            (MergeMode::Overlay.blend(0.6, 0.6) - (1.0 - 2.0 * 0.4 * 0.4)).abs() < 1e-12
        );
    }

    /// `merge_mode`/`merge_order` round-trip through JSON and seed cleanly from a
    /// direct_trap spec; the composite stays deterministic in iteration order
    /// (identical params → byte-identical render).
    #[test]
    fn merge_params_roundtrip_and_deterministic() {
        let mut p = ColoringParams::beautiful(Field::DirectTrap);
        p.merge_mode = MergeMode::Screen;
        p.merge_order = MergeOrder::TopDown;
        let json = p.to_json();
        assert!(json.contains("\"merge_mode\":\"screen\""));
        assert!(json.contains("\"merge_order\":\"top_down\""));
        let rt = ColoringParams::from_json(&json).unwrap();
        assert_eq!(rt, p);
        // Spec naming only the merge keys seeds from beautiful(DirectTrap).
        let seeded = ColoringParams::from_json(
            "{\"field\":\"direct_trap\",\"merge_mode\":\"multiply\"}",
        )
        .unwrap();
        assert_eq!(seeded.merge_mode, MergeMode::Multiply);
        assert_eq!(seeded.merge_order, MergeOrder::BottomUp);

        // Determinism: same params render byte-identically (composite order = iter
        // order, no cross-pixel hazards).
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.03,
            out_width: 24,
            out_height: 16,
        };
        let palette = crate::palette::builtin("default", false).unwrap();
        let mut q = ColoringParams::beautiful(Field::DirectTrap);
        q.merge_mode = MergeMode::Multiply;
        q.direct_threshold = 0.5;
        let a = render_beautiful(&frame, 2, 300, Family::Mandelbrot, &q, &palette, DownsampleFilter::Box);
        let b = render_beautiful(&frame, 2, 300, Family::Mandelbrot, &q, &palette, DownsampleFilter::Box);
        assert_eq!(a.into_raw(), b.into_raw());
    }

    /// `start_color`: parses (names + hex), JSON-round-trips, a black start
    /// reproduces the prior hardcoded-black render byte-for-byte, and a white start
    /// changes the `multiply` result (and isn't all-black).
    #[test]
    fn start_color_param() {
        // Parse: names and hex → linear RGB; round-trips through the hex form.
        assert_eq!(parse_start_color("black").unwrap(), [0.0, 0.0, 0.0]);
        assert_eq!(parse_start_color("white").unwrap(), [1.0, 1.0, 1.0]);
        assert_eq!(parse_start_color("#ffffff").unwrap(), [1.0, 1.0, 1.0]);
        assert_eq!(parse_start_color("ff0000").unwrap()[0], 1.0);
        assert!(parse_start_color("nope").is_err());
        assert_eq!(start_color_to_hex([0.0, 0.0, 0.0]), "#000000");
        assert_eq!(start_color_to_hex([1.0, 1.0, 1.0]), "#ffffff");

        let mut p = ColoringParams::beautiful(Field::DirectTrap);
        p.merge_mode = MergeMode::Screen;
        p.start_color = [1.0, 1.0, 1.0];
        let json = p.to_json();
        assert!(json.contains("\"start_color\":\"#ffffff\""));
        assert_eq!(ColoringParams::from_json(&json).unwrap(), p);

        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.03,
            out_width: 24,
            out_height: 16,
        };
        let palette = crate::palette::builtin("default", false).unwrap();

        // Black start (explicit) reproduces the default (hardcoded-black) screen
        // render byte-for-byte.
        let mut screen_default = ColoringParams::beautiful(Field::DirectTrap);
        screen_default.merge_mode = MergeMode::Screen;
        screen_default.direct_threshold = 0.5;
        let mut screen_black = screen_default;
        screen_black.start_color = [0.0, 0.0, 0.0];
        let img_default =
            render_beautiful(&frame, 2, 300, Family::Mandelbrot, &screen_default, &palette, DownsampleFilter::Box);
        let img_black =
            render_beautiful(&frame, 2, 300, Family::Mandelbrot, &screen_black, &palette, DownsampleFilter::Box);
        assert_eq!(img_default.clone().into_raw(), img_black.into_raw());

        // Multiply on a white start differs from multiply on black, and isn't
        // all-black (black start is the absorbing/degenerate case for multiply).
        let mut mul_black = ColoringParams::beautiful(Field::DirectTrap);
        mul_black.merge_mode = MergeMode::Multiply;
        mul_black.direct_threshold = 0.5;
        let mut mul_white = mul_black;
        mul_white.start_color = [1.0, 1.0, 1.0];
        let img_mb =
            render_beautiful(&frame, 2, 300, Family::Mandelbrot, &mul_black, &palette, DownsampleFilter::Box);
        let img_mw =
            render_beautiful(&frame, 2, 300, Family::Mandelbrot, &mul_white, &palette, DownsampleFilter::Box);
        assert_ne!(img_mb.into_raw(), img_mw.clone().into_raw());
        assert!(
            img_mw.pixels().any(|px| px.0 != [0, 0, 0]),
            "multiply on white start collapsed to all-black"
        );
    }

    /// Source-side saturation cap: `direct_trap_screen` (cross + screen) is clamped
    /// to the non-saturating regime, so an over-cap spec renders identically to the
    /// cap. Below-cap params are untouched (distinct params → distinct output), and
    /// the cap is gated strictly to cross+screen — ring+screen is NOT clamped.
    #[test]
    fn direct_trap_screen_saturation_cap() {
        let frame = Frame {
            center: Complex::new(-0.74, 0.13),
            frame_width: 0.03,
            out_width: 24,
            out_height: 16,
        };
        let palette = crate::palette::builtin("default", false).unwrap();
        let render = |p: &ColoringParams| {
            render_beautiful(&frame, 2, 400, Family::Mandelbrot, p, &palette, DownsampleFilter::Box)
                .into_raw()
        };
        let cross_screen = |op: f64, th: f64| {
            let mut p = ColoringParams::beautiful(Field::DirectTrap);
            p.merge_mode = MergeMode::Screen;
            p.direct_shape = DirectShape::Cross;
            p.direct_opacity = op;
            p.direct_threshold = th;
            p
        };

        // Over-cap (op 0.45 / th 0.12) is clamped to (0.15 / 0.08) byte-for-byte.
        assert_eq!(
            render(&cross_screen(0.45, 0.12)),
            render(&cross_screen(DTS_SCREEN_OPACITY_CAP, DTS_SCREEN_THRESHOLD_CAP)),
            "cross+screen above the cap must render identically to the cap"
        );
        // Below the cap is untouched: distinct sub-cap opacities render differently.
        assert_ne!(
            render(&cross_screen(0.05, 0.05)),
            render(&cross_screen(DTS_SCREEN_OPACITY_CAP, 0.05)),
            "sub-cap cross+screen opacity must not be flattened by the clamp"
        );
        // Gated to cross+screen only: ring+screen is NOT capped (op 0.45 ≠ op 0.15).
        let ring_screen = |op: f64| {
            let mut p = ColoringParams::beautiful(Field::DirectTrap);
            p.merge_mode = MergeMode::Screen;
            p.direct_shape = DirectShape::Ring;
            p.trap_radius = 1.0;
            p.direct_opacity = op;
            p.direct_threshold = 0.12;
            p
        };
        assert_ne!(
            render(&ring_screen(0.45)),
            render(&ring_screen(0.15)),
            "ring+screen must be untouched by the cross+screen cap"
        );
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
            Family::Mandelbrot,
        );
        assert!(!acc.escaped);
        assert!(acc.field(Field::TrapCross).is_some());
        assert!(acc.field(Field::Smooth).is_none());
    }

    // === New families: multibrot d=3,4,5 + classic Phoenix ===

    /// Degree-2 reduction is the exact prior float sequence: `cpow_deriv(z, 2)` is
    /// `(z·z, z)` bit-for-bit (so Mandelbrot/Julia `z^d`/`dz` are unchanged), and the
    /// higher degrees are plain repeated multiplication.
    #[test]
    fn cpow_deriv_reduces_exactly() {
        for &(re, im) in &[(0.3, -0.7), (1.5, 0.0), (-0.9, 1.1), (0.0, 0.4)] {
            let z = Complex::new(re, im);
            let (z2, z1) = cpow_deriv(z, 2);
            assert_eq!(z2.re.to_bits(), (z * z).re.to_bits());
            assert_eq!(z2.im.to_bits(), (z * z).im.to_bits());
            assert_eq!(z1.re.to_bits(), z.re.to_bits()); // z^1 = z
            assert_eq!(z1.im.to_bits(), z.im.to_bits());
            // z^3 = z·z·z, derivative-power z^2; z^5 derivative-power z^4.
            let (z3, z3d) = cpow_deriv(z, 3);
            assert_eq!(z3.re.to_bits(), (z * z * z).re.to_bits());
            assert_eq!(z3d.re.to_bits(), (z * z).re.to_bits());
            let (z5, z5d) = cpow_deriv(z, 5);
            assert_eq!(z5.re.to_bits(), (z * z * z * z * z).re.to_bits());
            assert_eq!(z5d.re.to_bits(), (z * z * z * z).re.to_bits());
        }
    }

    /// Degree-2 smooth value pins the exact `LN_2` constant (not `2.0.ln()`), so a
    /// degree-2 render is bit-for-bit the prior value.
    #[test]
    fn smooth_value_degree2_uses_ln2() {
        let (n, zabs2, b) = (12u32, 1.0e9f64, 65536.0f64);
        let got = smooth_value(n, zabs2, b, 2);
        let log_zn = 0.5 * zabs2.ln();
        let want = (n + 1) as f64 - (log_zn / b.ln()).ln() / std::f64::consts::LN_2;
        assert_eq!(got.to_bits(), want.to_bits());
        // Degree 3 uses ln(3) — a different, converging base.
        let got3 = smooth_value(n, zabs2, b, 3);
        let want3 = (n + 1) as f64 - (log_zn / b.ln()).ln() / 3.0f64.ln();
        assert_eq!(got3.to_bits(), want3.to_bits());
        assert_ne!(got.to_bits(), got3.to_bits());
    }

    /// Multibrot `z^d + c` is invariant under `c → ω·c` with `ω^{d-1} = 1`. For the
    /// two degrees where `ω` is exactly representable — d=3 (`ω = -1`) and d=5
    /// (`ω = i`) — the rotated orbit is the exact float negation/`i`-multiple of the
    /// original, so the escape classification and smooth value are **byte-identical**.
    #[test]
    fn multibrot_rotational_symmetry_exact() {
        let p = ColoringParams::beautiful(Field::Smooth);
        let z0 = Complex::new(0.0, 0.0);
        for &(re, im) in &[(0.4, 0.3), (-0.6, 0.8), (1.1, -0.2), (0.05, 1.0)] {
            let c = Complex::new(re, im);
            // d=3: ω = -1 (d-1 = 2).
            let a = iterate_orbit(z0, c, 400, &p, Family::Multibrot { degree: 3 });
            let b = iterate_orbit(z0, -c, 400, &p, Family::Multibrot { degree: 3 });
            assert_eq!(a.escaped, b.escaped);
            assert_eq!(a.smooth.to_bits(), b.smooth.to_bits(), "d=3 c=({re},{im})");
            // d=5: ω = i (d-1 = 4). i·c = (-im, re).
            let ic = Complex::new(-im, re);
            let a5 = iterate_orbit(z0, c, 400, &p, Family::Multibrot { degree: 5 });
            let b5 = iterate_orbit(z0, ic, 400, &p, Family::Multibrot { degree: 5 });
            assert_eq!(a5.escaped, b5.escaped);
            assert_eq!(a5.smooth.to_bits(), b5.smooth.to_bits(), "d=5 c=({re},{im})");
        }
    }

    /// Phoenix with `p = 0` drops the `z_{n-1}` coupling and reduces to a plain Julia
    /// (`z_{n+1} = z_n² + c`, same `z₀ = pixel`, same `dz₀ = 1`, same `dz` recurrence
    /// with no `+1`), so the two paths must agree **bit-for-bit** on every channel.
    #[test]
    fn phoenix_p_zero_is_julia() {
        let p = ColoringParams::beautiful(Field::Smooth);
        let c = Complex::new(-0.8, 0.156);
        for &(re, im) in &[(0.5, 0.5), (-1.2, 0.3), (0.0, -0.9), (2.0, 2.0)] {
            let pixel = Complex::new(re, im);
            let ph = iterate_orbit(
                pixel,
                c,
                500,
                &p,
                Family::Phoenix { c, p: Complex::new(0.0, 0.0) },
            );
            let ju = iterate_orbit(pixel, c, 500, &p, Family::Julia { c, degree: 2 });
            assert_eq!(ph.escaped, ju.escaped);
            assert_eq!(ph.smooth.to_bits(), ju.smooth.to_bits(), "pixel=({re},{im})");
            assert_eq!(ph.z.re.to_bits(), ju.z.re.to_bits());
            assert_eq!(ph.z.im.to_bits(), ju.z.im.to_bits());
            assert_eq!(ph.dz.re.to_bits(), ju.dz.re.to_bits());
            assert_eq!(ph.dz.im.to_bits(), ju.dz.im.to_bits());
        }
    }

    /// Family metadata: degree threads the exponent; dynamical families seed the
    /// z-plane (`z₀ = pixel`), parameter-plane families seed `z₀ = 0`.
    #[test]
    fn family_seed_and_degree() {
        let pixel = Complex::new(0.37, -0.11);
        assert_eq!(Family::Mandelbrot.degree(), 2);
        assert_eq!(Family::Multibrot { degree: 4 }.degree(), 4);
        assert_eq!(Family::Phoenix { c: pixel, p: pixel }.degree(), 2);
        // Parameter plane: z0 = 0, c = pixel.
        assert_eq!(Family::Mandelbrot.seed(pixel), (Complex::new(0.0, 0.0), pixel));
        assert_eq!(
            Family::Multibrot { degree: 3 }.seed(pixel),
            (Complex::new(0.0, 0.0), pixel)
        );
        // Dynamical: z0 = pixel, c = fixed const.
        let k = Complex::new(0.5667, 0.0);
        assert_eq!(Family::Julia { c: k, degree: 2 }.seed(pixel), (pixel, k));
        // Julia-multibrot threads its degree while staying dynamical (z0 = pixel).
        assert_eq!(Family::Julia { c: k, degree: 4 }.degree(), 4);
        assert_eq!(Family::Julia { c: k, degree: 4 }.seed(pixel), (pixel, k));
        assert!(Family::Julia { c: k, degree: 3 }.is_dynamical());
        assert_eq!(
            Family::Phoenix { c: k, p: Complex::new(-0.5, 0.0) }.seed(pixel),
            (pixel, k)
        );
        assert!(!Family::Mandelbrot.is_dynamical());
        assert!(Family::Phoenix { c: k, p: k }.is_dynamical());
    }
}
