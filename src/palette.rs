//! One built-in cyclic gradient.
//!
//! v1 keeps this deliberately simple: a fixed set of control points
//! interpolated linearly **in linear light**, looked up cyclically. The
//! OKLab gradient system and palette-file loaders are Prompt 4. The only
//! contract coloring depends on is [`Palette::lookup_linear`].

/// A cyclic gradient: control colors at parametric positions in `[0, 1)`,
/// stored in linear-light RGB. Position `1.0` wraps back to position `0.0`.
pub struct Palette {
    /// (position, linear-RGB) stops, sorted by position ascending.
    stops: Vec<(f64, [f64; 3])>,
}

impl Palette {
    /// The classic "Ultra Fractal" exterior gradient (deep blue → blue →
    /// white → orange → near-black), the recognizable default Mandelbrot
    /// coloring. Stops are specified in 8-bit sRGB and converted to linear.
    pub fn ultra_fractal() -> Self {
        let srgb_stops: &[(f64, [u8; 3])] = &[
            (0.0, [0, 7, 100]),
            (0.16, [32, 107, 203]),
            (0.42, [237, 255, 255]),
            (0.6425, [255, 170, 0]),
            (0.8575, [0, 2, 0]),
        ];
        let stops = srgb_stops
            .iter()
            .map(|&(pos, rgb)| {
                (
                    pos,
                    [
                        srgb_to_linear(rgb[0] as f64 / 255.0),
                        srgb_to_linear(rgb[1] as f64 / 255.0),
                        srgb_to_linear(rgb[2] as f64 / 255.0),
                    ],
                )
            })
            .collect();
        Palette { stops }
    }

    /// Look up the cyclic gradient at `t`, returning linear-light RGB.
    /// `t` is taken modulo 1.0 (callers already pass `rem_euclid(1.0)`, this
    /// is defensive). Interpolates linearly between the two surrounding stops,
    /// wrapping from the last stop back to the first.
    pub fn lookup_linear(&self, t: f64) -> [f64; 3] {
        let t = t.rem_euclid(1.0);
        let stops = &self.stops;
        let n = stops.len();

        // Find the segment [a, b) containing t. Because the gradient is
        // cyclic, the final segment runs from the last stop (wrapping past
        // 1.0) back to the first stop at position+1.0.
        for i in 0..n {
            let (pa, ca) = stops[i];
            let (pb_raw, cb) = if i + 1 < n {
                stops[i + 1]
            } else {
                let (p0, c0) = stops[0];
                (p0 + 1.0, c0)
            };
            if t >= pa && t < pb_raw {
                let f = (t - pa) / (pb_raw - pa);
                return lerp3(ca, cb, f);
            }
        }

        // t is below the first stop's position: it lives in the wrap segment
        // (last stop → first stop). Reconstruct that interpolation.
        let (plast, clast) = stops[n - 1];
        let (p0, c0) = stops[0];
        let span = (p0 + 1.0) - plast;
        let f = (t + 1.0 - plast) / span;
        lerp3(clast, c0, f)
    }
}

#[inline]
fn lerp3(a: [f64; 3], b: [f64; 3], f: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

/// sRGB transfer function (gamma-decode): sRGB component → linear.
#[inline]
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer function (gamma-encode): linear → sRGB.
#[inline]
pub fn linear_to_srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
