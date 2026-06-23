//! Cyclic gradients interpolated in **OKLab**, baked to a linear-RGB LUT.
//!
//! A palette is a set of control points `(position ∈ [0,1), color)` interpolated
//! perceptually (OKLab) and **cyclically** (the last control point wraps to the
//! first across the 1→0 seam). Perceptual interpolation matters because the
//! coloring stage maps a smooth, evenly-spaced channel value onto the gradient;
//! linear-light interpolation would bunch perceived hue/lightness unevenly.
//!
//! To keep the per-subpixel hot path O(1), each palette is **baked once** into a
//! cyclic LUT of [`LUT_SIZE`] linear-RGB entries at construction. `lookup_linear`
//! — the only contract the coloring stage depends on — is then a LUT index plus a
//! cheap lerp, never an OKLab interpolation.
//!
//! Conversions use the standard Ottosson linear-sRGB↔OKLab matrices (LMS +
//! cube-root). Source colors (sRGB8 from files / generators) are converted
//! sRGB8 → linear sRGB → OKLab on load; the bake converts back OKLab → linear
//! RGB to feed the existing linear-light shade/downsample pipeline.

/// Number of baked LUT entries. 4096 makes the LUT quantization step far finer
/// than 8-bit output, so the lerp between entries is visually exact.
pub const LUT_SIZE: usize = 4096;

/// Density multiplier applied by the coloring stage when a palette was built with
/// **pre-mirror** (selective seam fix for SEQUENTIAL maps). Pre-mirror folds the
/// gradient into an out-and-back, doubling the spatial band frequency at a fixed
/// density; scaling the effective density by this factor keeps a mirrored
/// sequential map's band count ~matched to the un-mirrored original — just
/// de-seamed. `1.0` for un-mirrored palettes (no change). Matched in
/// `coloring.MIRROR_DENSITY_SCALE`.
pub const MIRROR_DENSITY_SCALE: f64 = 0.5;

/// A cyclic gradient baked to a linear-RGB lookup table.
pub struct Palette {
    /// Display name (built-in name, `.map` filename, or `.ugr` block title).
    name: String,
    /// Cyclic LUT of linear-light RGB, `LUT_SIZE` entries. Index `i` is the
    /// color at `t = i / LUT_SIZE`; entry `LUT_SIZE-1` wraps to entry `0`.
    lut: Vec<[f64; 3]>,
    /// Coloring-stage density multiplier (see [`MIRROR_DENSITY_SCALE`]).
    /// `MIRROR_DENSITY_SCALE` for a pre-mirrored palette, else `1.0`. Does **not**
    /// affect the baked LUT (so the byte-match invariant is unaffected) — only the
    /// `value·density` mapping in [`crate::coloring::shade`].
    density_scale: f64,
}

/// An OKLab control point: parametric position and OKLab color.
#[derive(Clone, Copy)]
struct OklabStop {
    pos: f64,
    lab: [f64; 3],
}

impl Palette {
    /// The classic "Ultra Fractal" exterior gradient — the recognizable default
    /// Mandelbrot coloring. Kept as the built-in `default` so existing
    /// invocations don't change palette (the interpolation is now OKLab).
    pub fn ultra_fractal() -> Self {
        const STOPS: &[(f64, [u8; 3])] = &[
            (0.0, [0, 7, 100]),
            (0.16, [32, 107, 203]),
            (0.42, [237, 255, 255]),
            (0.6425, [255, 170, 0]),
            (0.8575, [0, 2, 0]),
        ];
        Palette::from_srgb8_stops("default", STOPS, false)
    }

    /// Build a palette from sRGB8 control points (file loaders / authored
    /// built-ins). `reverse` flips the gradient direction.
    pub fn from_srgb8_stops(name: impl Into<String>, stops: &[(f64, [u8; 3])], reverse: bool) -> Self {
        Palette::from_srgb8_stops_mirrored(name, stops, reverse, false)
    }

    /// As [`from_srgb8_stops`], but `mirror=true` first reflects the stops into a
    /// seamless out-and-back via [`mirror_stops`] — the **selective** seam fix for
    /// SEQUENTIAL (`mirror_needed`) palettes. Pass `mirror=false` for cyclic maps
    /// (unchanged single-pass bake). See [`mirror_stops`] for the construction.
    pub fn from_srgb8_stops_mirrored(
        name: impl Into<String>,
        stops: &[(f64, [u8; 3])],
        reverse: bool,
        mirror: bool,
    ) -> Self {
        let mirrored;
        let stops: &[(f64, [u8; 3])] = if mirror {
            mirrored = mirror_stops(stops);
            &mirrored
        } else {
            stops
        };
        let oklab: Vec<OklabStop> = stops
            .iter()
            .map(|&(pos, rgb)| OklabStop {
                pos,
                lab: srgb8_to_oklab(rgb),
            })
            .collect();
        let mut pal = Palette::from_oklab_stops(name, oklab, reverse);
        if mirror {
            // Pre-mirror doubled the band frequency; compensate so the mirrored
            // sequential map keeps the original's band count (just de-seamed).
            pal.density_scale = MIRROR_DENSITY_SCALE;
        }
        pal
    }

    /// Build a cyclic gradient from a set of OKLab colors placed at **evenly
    /// spaced** positions (`i / n`). The corpus color block (Prompt 9) is a set of
    /// dominant OKLab cluster centers with no inherent parametric position; the
    /// caller orders them (e.g. by luminance) and this lays them out uniformly.
    /// At least two colors are required.
    pub fn from_oklab_colors(name: impl Into<String>, colors: &[[f64; 3]], reverse: bool) -> Self {
        let n = colors.len();
        let stops: Vec<OklabStop> = colors
            .iter()
            .enumerate()
            .map(|(i, &lab)| OklabStop {
                pos: i as f64 / n as f64,
                lab,
            })
            .collect();
        Palette::from_oklab_stops(name, stops, reverse)
    }

    /// Build from already-OKLab control points. Stops are sorted by position;
    /// at least two distinct stops are required.
    fn from_oklab_stops(name: impl Into<String>, mut stops: Vec<OklabStop>, reverse: bool) -> Self {
        assert!(
            stops.len() >= 2,
            "a palette needs at least two control points"
        );
        // Normalize positions into [0,1) and sort. Stable so duplicate-position
        // stops keep input order.
        for s in &mut stops {
            s.pos = s.pos.rem_euclid(1.0);
        }
        stops.sort_by(|a, b| a.pos.partial_cmp(&b.pos).unwrap());

        let mut lut = vec![[0.0f64; 3]; LUT_SIZE];
        for (i, entry) in lut.iter_mut().enumerate() {
            let t = i as f64 / LUT_SIZE as f64;
            let lab = interp_oklab_cyclic(&stops, t);
            *entry = oklab_to_linear_srgb(lab);
        }
        if reverse {
            // Reverse direction about t=0: new[i] = old[(N - i) mod N], so the
            // seam stays continuous (new[0] == old[0]).
            let src = lut.clone();
            for i in 0..LUT_SIZE {
                lut[i] = src[(LUT_SIZE - i) % LUT_SIZE];
            }
        }

        Palette {
            name: name.into(),
            lut,
            density_scale: 1.0,
        }
    }

    /// Display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Coloring-stage density multiplier (see [`MIRROR_DENSITY_SCALE`]): `0.5`
    /// for a pre-mirrored sequential palette, `1.0` otherwise. The shade stage
    /// multiplies the configured density by this so a mirrored map keeps its
    /// band count.
    #[inline]
    pub fn density_scale(&self) -> f64 {
        self.density_scale
    }

    /// Look up the cyclic gradient at `t`, returning linear-light RGB. `t` is
    /// taken modulo 1.0 (defensive; callers already pass `rem_euclid(1.0)`).
    /// O(1): a LUT index plus a lerp between adjacent (cyclic) entries.
    #[inline]
    pub fn lookup_linear(&self, t: f64) -> [f64; 3] {
        let t = t.rem_euclid(1.0);
        let x = t * LUT_SIZE as f64;
        let i0 = x.floor() as usize;
        let f = x - i0 as f64;
        let i0 = i0 % LUT_SIZE; // guard the t==1.0-epsilon rounding edge
        let i1 = (i0 + 1) % LUT_SIZE;
        lerp3(self.lut[i0], self.lut[i1], f)
    }
}

/// Interpolate an OKLab color at cyclic position `t ∈ [0,1)` between the
/// bracketing control points. The final segment wraps the last stop (at its
/// position) to the first stop (at position + 1.0).
fn interp_oklab_cyclic(stops: &[OklabStop], t: f64) -> [f64; 3] {
    let n = stops.len();
    for i in 0..n {
        let a = stops[i];
        let (pb, cb) = if i + 1 < n {
            (stops[i + 1].pos, stops[i + 1].lab)
        } else {
            (stops[0].pos + 1.0, stops[0].lab)
        };
        if t >= a.pos && t < pb {
            let f = (t - a.pos) / (pb - a.pos);
            return lerp3(a.lab, cb, f);
        }
    }
    // t is below the first stop's position: it lives in the wrap segment
    // (last stop → first stop). Reconstruct that interpolation.
    let last = stops[n - 1];
    let first = stops[0];
    let span = (first.pos + 1.0) - last.pos;
    let f = (t + 1.0 - last.pos) / span;
    lerp3(last.lab, first.lab, f)
}

/// Pre-mirror a stop list into a symmetric out-and-back (triangle wave).
///
/// For a SEQUENTIAL (`mirror_needed`) palette the raw cyclic bake compresses the
/// endpoint (last→first color) transition into the tiny wrap segment, producing a
/// visible seam band on trap renders (and spurious dark/light stops when the
/// extractor recovers that band). Reflecting the stops removes the seam: the
/// forward gradient occupies positions `[0, 0.5]`, its reflection occupies
/// `(0.5, 1)`, and the cyclic wrap (last→first) mirrors the opening segment, so
/// endpoints meet on the same color. Density `d` then yields `d` out-and-back
/// passes.
///
/// Matched byte-for-byte to `coloring.mirror_stops` (Python port): same
/// normalize-into-`[0,1)` + stable sort, same `u = (p−p0)/span` remap, same
/// `0.5·u` forward / `1−0.5·u` reflected positions. The reflection drops the two
/// endpoints (`i=0` lands on the seam at 0, already present; `i=n−1` is the
/// turning point at 0.5, already present), giving `2n−2` stops.
fn mirror_stops(stops: &[(f64, [u8; 3])]) -> Vec<(f64, [u8; 3])> {
    let mut s: Vec<(f64, [u8; 3])> =
        stops.iter().map(|&(p, c)| (p.rem_euclid(1.0), c)).collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = s.len();
    let p0 = s[0].0;
    let span = s[n - 1].0 - p0;
    if !(span > 0.0) {
        return s; // degenerate: all stops coincide — nothing to mirror.
    }
    let u = |i: usize| (s[i].0 - p0) / span;
    let mut out = Vec::with_capacity(2 * n - 2);
    for i in 0..n {
        out.push((0.5 * u(i), s[i].1)); // forward → [0, 0.5]
    }
    for i in (1..n - 1).rev() {
        out.push((1.0 - 0.5 * u(i), s[i].1)); // reflection → (0.5, 1)
    }
    out
}

#[inline]
fn lerp3(a: [f64; 3], b: [f64; 3], f: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

// ---------------------------------------------------------------------------
// sRGB transfer function
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OKLab (Ottosson) — linear sRGB ↔ OKLab
// ---------------------------------------------------------------------------

/// Linear sRGB → OKLab.
#[inline]
pub fn linear_srgb_to_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    let l = 0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b;
    let m = 0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b;
    let s = 0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    [
        0.210_454_255_3 * l_ + 0.793_617_785_0 * m_ - 0.004_072_046_8 * s_,
        1.977_998_495_1 * l_ - 2.428_592_205_0 * m_ + 0.450_593_709_9 * s_,
        0.025_904_037_1 * l_ + 0.782_771_766_2 * m_ - 0.808_675_766_0 * s_,
    ]
}

/// OKLab → linear sRGB.
#[inline]
pub fn oklab_to_linear_srgb(lab: [f64; 3]) -> [f64; 3] {
    let [ll, aa, bb] = lab;
    let l_ = ll + 0.396_337_777_4 * aa + 0.215_803_757_3 * bb;
    let m_ = ll - 0.105_561_345_8 * aa - 0.063_854_172_8 * bb;
    let s_ = ll - 0.089_484_177_5 * aa - 1.291_485_548_0 * bb;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
        4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s,
        -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s,
        -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s,
    ]
}

/// Convenience: sRGB8 → OKLab (decode gamma, then to OKLab).
#[inline]
pub fn srgb8_to_oklab(rgb: [u8; 3]) -> [f64; 3] {
    linear_srgb_to_oklab([
        srgb_to_linear(rgb[0] as f64 / 255.0),
        srgb_to_linear(rgb[1] as f64 / 255.0),
        srgb_to_linear(rgb[2] as f64 / 255.0),
    ])
}

// ---------------------------------------------------------------------------
// Generated, license-free palettes
// ---------------------------------------------------------------------------

/// Number of control points sampled from a parametric generator before baking.
const GEN_STOPS: usize = 256;

/// **Cubehelix** (Green 2011): a perceptually monotonic-lightness helix through
/// RGB. The zero-restriction default for licensing-clean output. Defaults:
/// `start = 0.5`, `rotations = -1.5`, `hue = 1.0`, `gamma = 1.0`. Outputs are
/// treated as sRGB and converted like any source color.
pub fn cubehelix(reverse: bool) -> Palette {
    cubehelix_with("cubehelix", 0.5, -1.5, 1.0, 1.0, reverse)
}

/// Cubehelix with explicit parameters (kept generic for future CLI exposure).
pub fn cubehelix_with(
    name: impl Into<String>,
    start: f64,
    rotations: f64,
    hue: f64,
    gamma: f64,
    reverse: bool,
) -> Palette {
    use std::f64::consts::PI;
    let stops: Vec<OklabStop> = (0..GEN_STOPS)
        .map(|i| {
            let fract = i as f64 / GEN_STOPS as f64;
            let angle = 2.0 * PI * (start / 3.0 + 1.0 + rotations * fract);
            let fg = fract.powf(gamma);
            let amp = hue * fg * (1.0 - fg) / 2.0;
            let (sa, ca) = angle.sin_cos();
            let r = fg + amp * (-0.14861 * ca + 1.78277 * sa);
            let g = fg + amp * (-0.29227 * ca - 0.90649 * sa);
            let b = fg + amp * (1.97294 * ca);
            // Cubehelix output is sRGB (display-referred); clamp then decode.
            let lin = [
                srgb_to_linear(r.clamp(0.0, 1.0)),
                srgb_to_linear(g.clamp(0.0, 1.0)),
                srgb_to_linear(b.clamp(0.0, 1.0)),
            ];
            OklabStop {
                pos: fract,
                lab: linear_srgb_to_oklab(lin),
            }
        })
        .collect();
    Palette::from_oklab_stops(name, stops, reverse)
}

/// **Viridis** (Matplotlib, CC0): perceptually-uniform sequential map. Embedded
/// as a compact control-point set (sampled from the published data) — used
/// cyclically here, so expect a wrap from yellow back to purple (inherent to a
/// sequential map cycled).
pub fn viridis(reverse: bool) -> Palette {
    // 11 evenly spaced samples of the canonical viridis colormap (sRGB8).
    const STOPS: &[(f64, [u8; 3])] = &[
        (0.0, [68, 1, 84]),
        (0.1, [72, 36, 117]),
        (0.2, [65, 68, 135]),
        (0.3, [53, 95, 141]),
        (0.4, [42, 120, 142]),
        (0.5, [33, 145, 140]),
        (0.6, [34, 168, 132]),
        (0.7, [68, 191, 112]),
        (0.8, [122, 209, 81]),
        (0.9, [189, 223, 38]),
        (0.95, [253, 231, 37]),
    ];
    Palette::from_srgb8_stops("viridis", STOPS, reverse)
}

/// Resolve a built-in palette by name. Returns `None` for unknown names (the
/// caller then treats the spec as a file path).
pub fn builtin(name: &str, reverse: bool) -> Option<Palette> {
    match name {
        "default" => Some(Palette::ultra_fractal()),
        "cubehelix" => Some(cubehelix(reverse)),
        "viridis" => Some(viridis(reverse)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sRGB → OKLab → sRGB must round-trip within tolerance across the cube.
    #[test]
    fn oklab_srgb_roundtrip() {
        let mut max_err = 0.0f64;
        for &r in &[0u8, 17, 64, 128, 200, 255] {
            for &g in &[0u8, 17, 64, 128, 200, 255] {
                for &b in &[0u8, 17, 64, 128, 200, 255] {
                    let lin = [
                        srgb_to_linear(r as f64 / 255.0),
                        srgb_to_linear(g as f64 / 255.0),
                        srgb_to_linear(b as f64 / 255.0),
                    ];
                    let lab = linear_srgb_to_oklab(lin);
                    let back = oklab_to_linear_srgb(lab);
                    for k in 0..3 {
                        let s0 = linear_to_srgb(lin[k]);
                        let s1 = linear_to_srgb(back[k]);
                        max_err = max_err.max((s0 - s1).abs());
                    }
                }
            }
        }
        // Round-trip is exact up to float round-off; a half-LSB of 8-bit output
        // is ~0.002. Hold well under that.
        assert!(max_err < 1e-4, "OKLab round-trip max sRGB error {max_err:e}");
    }

    /// The baked LUT is cyclic: the wrap step (last→first) is no larger than the
    /// largest interior step, i.e. there is no seam discontinuity at 1→0.
    #[test]
    fn cyclic_gradient_has_no_seam() {
        for pal in [Palette::ultra_fractal(), cubehelix(false)] {
            let lut = &pal.lut;
            let n = lut.len();
            let step = |a: [f64; 3], b: [f64; 3]| {
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
            };
            let mut max_interior = 0.0f64;
            for i in 0..n - 1 {
                max_interior = max_interior.max(step(lut[i], lut[i + 1]));
            }
            let wrap = step(lut[n - 1], lut[0]);
            // The wrap segment is interpolated identically to interior segments,
            // so its step must be of the same order — no jump.
            assert!(
                wrap <= max_interior * 1.5 + 1e-9,
                "seam at 1→0 for '{}': wrap step {wrap:e} vs max interior {max_interior:e}",
                pal.name()
            );
        }
    }

    /// Control-point colors are reproduced at their positions (OKLab interp
    /// passes through the stops).
    #[test]
    fn stops_are_reproduced() {
        let stops: &[(f64, [u8; 3])] = &[
            (0.0, [10, 20, 200]),
            (0.5, [240, 250, 250]),
            (0.75, [255, 160, 0]),
        ];
        let pal = Palette::from_srgb8_stops("t", stops, false);
        for &(pos, rgb) in stops {
            let got = pal.lookup_linear(pos);
            let want = [
                srgb_to_linear(rgb[0] as f64 / 255.0),
                srgb_to_linear(rgb[1] as f64 / 255.0),
                srgb_to_linear(rgb[2] as f64 / 255.0),
            ];
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 5e-3,
                    "stop at {pos}: got {got:?} want {want:?}"
                );
            }
        }
    }

    /// Reverse keeps the seam continuous and flips direction.
    #[test]
    fn reverse_flips_direction() {
        let stops: &[(f64, [u8; 3])] = &[(0.0, [255, 0, 0]), (0.5, [0, 255, 0])];
        let fwd = Palette::from_srgb8_stops("f", stops, false);
        let rev = Palette::from_srgb8_stops("r", stops, true);
        // new[0] == old[0] (seam fixed point); new[t] == old[1-t] elsewhere.
        let a = fwd.lookup_linear(0.0);
        let b = rev.lookup_linear(0.0);
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-6);
        }
        let c = fwd.lookup_linear(0.25);
        let d = rev.lookup_linear(0.75);
        for k in 0..3 {
            assert!((c[k] - d[k]).abs() < 1e-3, "reverse mismatch at 0.25/0.75");
        }
    }
}
