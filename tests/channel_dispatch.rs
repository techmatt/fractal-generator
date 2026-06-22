//! Channel-intent dispatch correctness: the f64 render path computes only the
//! channels the colorer reads ([`render::iterate_samples_f64`] +
//! [`coloring::required_channels`]) — this proves the channel-requirement map is
//! **complete**.
//!
//! For every color mode in the matrix and two bracketing workloads, we shade two
//! buffers:
//!  - the **all-on** buffer (trait `sample`, every channel live), and
//!  - the **dispatched** buffer (only `required_channels(params)` computed),
//!
//! both with the *same* `params`, and assert the **PNG bytes are byte-identical**.
//! A disabled channel leaves its `PixelSample` field at the inert default
//! (`trap_min = ∞`, `de = 0`); if the map wrongly skipped a channel the mode
//! actually reads, the dispatched render would diverge — that's the failure
//! signal. All must match.

use std::io::Cursor;

use num_complex::Complex;

use fractal_generator::backend::{F64Backend, Trap, TrapShape};
use fractal_generator::coloring::{self, ColorChannel, ColorParams, InteriorMode, TrapCurve};
use fractal_generator::palette::Palette;
use fractal_generator::render::{self, Frame};

/// PNG-encode an image to bytes in memory (the byte-equality unit — identical
/// pixels ⟹ identical PNG for this deterministic codec).
fn encode_png(img: &image::RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("png encode");
    bytes
}

/// The full color-mode matrix: every (exterior channel × interior × de_shade ×
/// trap_phase) combination that exercises a distinct channel consumer. Covers
/// all four `(trap, de)` dispatch combos and *both* readers of each channel
/// (trap: exterior `Trap` channel **and** `InteriorMode::Trap` fill; de:
/// exterior `De` channel **and** the de_shade overlay).
fn mode_matrix() -> Vec<(&'static str, ColorParams)> {
    let base = ColorParams {
        density: 0.05,
        offset: 0.1,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        // Nonzero so trap_phase is actually read — a skipped phase would diverge.
        trap_phase_strength: 0.37,
        de_shade: None,
        mark_glitches: false,
    };
    vec![
        // (trap=false, de=false)
        ("smooth/black", ColorParams { ..base }),
        // (trap=false, de=true): de via exterior De channel, and via de_shade.
        (
            "de/black",
            ColorParams { channel: ColorChannel::De, density: 0.4, ..base },
        ),
        (
            "smooth/black+deshade",
            ColorParams { de_shade: Some(2.0), ..base },
        ),
        // (trap=true, de=false): trap via exterior channel, and via interior fill.
        (
            "trap/black",
            ColorParams { channel: ColorChannel::Trap, density: 2.0, ..base },
        ),
        (
            "smooth/trap-interior",
            ColorParams { interior: InteriorMode::Trap, ..base },
        ),
        (
            "trap/trap-interior",
            ColorParams {
                channel: ColorChannel::Trap,
                interior: InteriorMode::Trap,
                density: 2.0,
                ..base
            },
        ),
        // (trap=true, de=true): both channels, both reader kinds engaged.
        (
            "trap/trap-interior+deshade",
            ColorParams {
                channel: ColorChannel::Trap,
                interior: InteriorMode::Trap,
                density: 2.0,
                de_shade: Some(1.5),
                ..base
            },
        ),
        (
            "de/trap-interior",
            ColorParams {
                channel: ColorChannel::De,
                interior: InteriorMode::Trap,
                density: 0.4,
                ..base
            },
        ),
    ]
}

/// Two bracketing workloads: a mostly-exterior decoration frame (escapers,
/// filaments — exercises smooth/de/de_shade) and an interior-heavy frame near a
/// minibrot (non-escaping pixels — exercises the interior-trap fill). A
/// non-point trap with an off-origin center makes `trap_min`/`trap_phase`
/// non-degenerate.
fn workloads() -> [(&'static str, Frame); 2] {
    [
        (
            "decoration",
            Frame {
                center: Complex::new(-0.745, 0.113),
                frame_width: 0.02,
                out_width: 200,
                out_height: 140,
            },
        ),
        (
            "interior",
            Frame {
                center: Complex::new(-0.75, 0.0),
                frame_width: 0.6,
                out_width: 200,
                out_height: 140,
            },
        ),
    ]
}

#[test]
fn dispatch_is_byte_identical_across_mode_matrix() {
    let ss = 2u32;
    let trap = Trap {
        shape: TrapShape::Circle,
        center: Complex::new(0.13, -0.21),
        radius: 0.5,
    };
    let palette = Palette::ultra_fractal();
    let modes = mode_matrix();

    for (wl_name, frame) in workloads() {
        let backend = F64Backend::new(2000, 1e6, trap);
        let spacing = frame.pixel_size();

        // The all-on reference buffer (trait `sample`, every channel live).
        let all_on = render::iterate_samples(&backend, &frame, ss);

        for (mode_name, params) in &modes {
            let channels = coloring::required_channels(params);
            let dispatched =
                render::iterate_samples_f64(&backend, &frame, ss, channels);

            let shade = |samples: &[_]| {
                encode_png(&render::shade_and_downsample(
                    samples,
                    frame.out_width,
                    frame.out_height,
                    ss,
                    &palette,
                    params,
                    spacing,
                ))
            };
            let ref_png = shade(&all_on.samples);
            let disp_png = shade(&dispatched.samples);

            assert_eq!(
                ref_png, disp_png,
                "workload '{wl_name}' mode '{mode_name}' (channels {channels:?}): \
                 dispatched render diverged from all-on — the channel map skipped a \
                 channel this mode reads"
            );
        }
        println!(
            "workload '{wl_name}': {} modes byte-identical (all-on vs channel-dispatch)",
            modes.len()
        );
    }
}

/// The map must be the conservative single source of truth: `Trap` anywhere
/// (exterior channel or interior fill) ⟹ `trap`; `De` channel or any de_shade ⟹
/// `de`; nothing else turns a channel on.
#[test]
fn required_channels_covers_every_consumer() {
    let base = ColorParams {
        density: 1.0,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    };

    let c = coloring::required_channels(&base);
    assert!(!c.trap && !c.de, "smooth/black reads neither optional channel");

    let c = coloring::required_channels(&ColorParams { channel: ColorChannel::Trap, ..base });
    assert!(c.trap && !c.de, "Trap exterior channel requires trap");

    let c = coloring::required_channels(&ColorParams { interior: InteriorMode::Trap, ..base });
    assert!(c.trap && !c.de, "Trap interior fill requires trap");

    let c = coloring::required_channels(&ColorParams { channel: ColorChannel::De, ..base });
    assert!(!c.trap && c.de, "De exterior channel requires de");

    let c = coloring::required_channels(&ColorParams { de_shade: Some(1.0), ..base });
    assert!(!c.trap && c.de, "de_shade overlay requires de");
}
