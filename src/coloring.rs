//! Separable coloring stage: `PixelSample` → linear-light RGB.
//!
//! This stage never re-iterates; it consumes only the [`PixelSample`] record.
//! Re-coloring a render is therefore just re-running this map. v1 mapping:
//! `t = (smooth_iter * density + offset).rem_euclid(1.0)` → cyclic gradient.
//! Interior (non-escaped) pixels are black for now; orbit-trap interior fill
//! arrives in Prompt 3.

use crate::backend::PixelSample;
use crate::palette::Palette;

/// Parameters controlling the escape-value → gradient-position mapping.
#[derive(Clone, Copy, Debug)]
pub struct ColorParams {
    /// Cycles per unit of smooth iteration count.
    pub density: f64,
    /// Phase offset / rotation into the gradient, in `[0, 1)`.
    pub offset: f64,
}

/// Map a sample to linear-light RGB. Output is averaged in linear light by the
/// render stage, then sRGB-encoded for the PNG.
#[inline]
pub fn shade(sample: &PixelSample, palette: &Palette, params: &ColorParams) -> [f64; 3] {
    if !sample.escaped {
        // Interior: black for now (Prompt 3 adds orbit-trap interior fill).
        return [0.0, 0.0, 0.0];
    }
    let t = (sample.smooth_iter * params.density + params.offset).rem_euclid(1.0);
    palette.lookup_linear(t)
}
