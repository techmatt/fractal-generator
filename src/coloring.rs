//! Separable coloring stage: `PixelSample` → linear-light RGB.
//!
//! This stage never re-iterates; it consumes only the [`PixelSample`] record,
//! so re-coloring a render is just re-running this map (the separability the
//! whole project leans on — see [`crate::render::shade_and_downsample`]). Core
//! mapping is unchanged from Prompt 1: `t = (value·density + offset).rem_euclid(1)`
//! → cyclic gradient. Prompt 3 adds the channel/interior/DE-shade switches.

use crate::backend::PixelSample;
use crate::palette::Palette;

/// Linear-light sentinel for `--mark-glitches` (sRGB magenta).
const GLITCH_LINEAR: [f64; 3] = [1.0, 0.0, 1.0];

/// Width (in output pixels) of the DE-shade falloff: the boundary-proximity
/// brightening fades out by the time the estimated distance reaches this many
/// pixels.
const DE_SHADE_WIDTH_PX: f64 = 2.0;

/// Primary **exterior** channel: which iteration product maps to the gradient.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ColorChannel {
    /// Smooth (normalized) iteration count — the classic escape-time look.
    Smooth,
    /// Orbit-trap minimum distance (optionally scaled), with `trap_phase` as a
    /// secondary hue offset.
    Trap,
    /// Distance-estimate filament index (log of normalized DE).
    De,
}

/// Treatment of non-escaping (interior) pixels.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum InteriorMode {
    /// Dead black — interior as deliberate negative space.
    Black,
    /// Palette via the orbit-trap channel — interior fill, no dead black.
    Trap,
}

/// Parameters controlling the channel → gradient-position mapping. All fields
/// are pure coloring inputs; none require re-iteration.
#[derive(Clone, Copy, Debug)]
pub struct ColorParams {
    /// Cycles per unit of the mapped channel value.
    pub density: f64,
    /// Phase offset / rotation into the gradient, in `[0, 1)`.
    pub offset: f64,
    /// Primary exterior channel.
    pub channel: ColorChannel,
    /// Interior treatment.
    pub interior: InteriorMode,
    /// Multiplier applied to `trap_min` before mapping (trap channel / interior).
    pub trap_scale: f64,
    /// Weight of `trap_phase` added as a secondary hue offset (trap channel /
    /// interior). `0.0` = phase unused.
    pub trap_phase_strength: f64,
    /// Optional DE-shade strength; brightens thin boundary filaments. `None` =
    /// off. Composes with any primary channel.
    pub de_shade: Option<f64>,
    /// Paint glitched (delta-underflow) subsamples magenta for diagnosis.
    pub mark_glitches: bool,
}

/// Map a sample to linear-light RGB. Output is averaged in linear light by the
/// render stage, then sRGB-encoded for the PNG. `pixel_spacing` is the frame
/// constant used to normalize the raw DE into pixel units.
#[inline]
pub fn shade(
    sample: &PixelSample,
    palette: &Palette,
    params: &ColorParams,
    pixel_spacing: f64,
) -> [f64; 3] {
    if params.mark_glitches && sample.glitched {
        return GLITCH_LINEAR;
    }

    let mut color = if sample.escaped {
        let value = match params.channel {
            ColorChannel::Smooth => sample.smooth_iter,
            ColorChannel::Trap => sample.trap_min * params.trap_scale,
            // Filament index: normalized DE, log-compressed so the wide exterior
            // range folds into a usable gradient sweep.
            ColorChannel::De => (sample.de / pixel_spacing).ln_1p(),
        };
        let mut t = value * params.density + params.offset;
        if matches!(params.channel, ColorChannel::Trap) {
            t += sample.trap_phase * params.trap_phase_strength;
        }
        palette.lookup_linear(t.rem_euclid(1.0))
    } else {
        match params.interior {
            InteriorMode::Black => [0.0, 0.0, 0.0],
            InteriorMode::Trap => {
                let t = sample.trap_min * params.trap_scale * params.density
                    + params.offset
                    + sample.trap_phase * params.trap_phase_strength;
                palette.lookup_linear(t.rem_euclid(1.0))
            }
        }
    };

    // DE-shade: brighten pixels close to the boundary (small DE in pixel units).
    // Exterior-only — interior DE is 0, which would otherwise read as "on the
    // boundary" and wrongly brighten the fill.
    if let Some(strength) = params.de_shade {
        if sample.escaped {
            let de_px = sample.de / pixel_spacing;
            let prox = 1.0 - smoothstep(0.0, DE_SHADE_WIDTH_PX, de_px);
            let factor = 1.0 + strength * prox;
            color[0] *= factor;
            color[1] *= factor;
            color[2] *= factor;
        }
    }

    color
}

/// Hermite smoothstep, clamped to `[0,1]`.
#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
