//! Separability proof: iteration runs **once**, then the cached sample buffer
//! is shaded into several PNGs with different coloring parameters.
//!
//! A `CountingBackend` wraps the real backend and tallies `sample()` calls. We
//! iterate once, assert the call count equals the subsample grid size, then run
//! `shade_and_downsample` three times with different `--color`/palette params
//! and assert the count never moves — i.e. re-coloring never re-iterated. The
//! three PNGs are written to `target/test-out/` for visual inspection.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex;

use fractal_generator::backend::{
    F64Backend, FractalBackend, PixelSample, Trap, TrapShape,
};
use fractal_generator::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use fractal_generator::palette::Palette;
use fractal_generator::render::{self, Frame};

/// Backend wrapper that counts how many times `sample()` is invoked.
struct CountingBackend<'a> {
    inner: &'a dyn FractalBackend,
    count: AtomicUsize,
}

impl FractalBackend for CountingBackend<'_> {
    fn sample(&self, c: Complex<f64>, dc: Complex<f64>) -> PixelSample {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.inner.sample(c, dc)
    }
}

#[test]
fn shade_is_pure_iteration_runs_once() {
    let frame = Frame {
        center: Complex::new(-0.745, 0.113),
        frame_width: 0.02,
        out_width: 240,
        out_height: 160,
    };
    let ss = 2u32;
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    let inner = F64Backend::new(2000, 1e6, trap);
    let counting = CountingBackend {
        inner: &inner,
        count: AtomicUsize::new(0),
    };

    // Stage 1: iterate exactly once.
    let buf = render::iterate_samples(&counting, &frame, ss);
    let after_iter = counting.count.load(Ordering::Relaxed);
    let expected = (frame.out_width * ss) as usize * (frame.out_height * ss) as usize;
    assert_eq!(
        after_iter, expected,
        "iteration should sample each subpixel exactly once"
    );

    let palette = Palette::ultra_fractal();
    let spacing = frame.pixel_size();

    // Stage 2: three different colorings from the SAME buffer.
    let variants = [
        (
            "sep-smooth",
            ColorParams {
                density: 0.03,
                offset: 0.0,
                channel: ColorChannel::Smooth,
                interior: InteriorMode::Black,
                trap_scale: 1.0,
                trap_curve: TrapCurve::Sqrt,
                trap_phase_strength: 0.0,
                de_shade: None,
                mark_glitches: false,
            },
        ),
        (
            "sep-trap",
            ColorParams {
                density: 2.0,
                offset: 0.2,
                channel: ColorChannel::Trap,
                interior: InteriorMode::Trap,
                trap_scale: 1.0,
                trap_curve: TrapCurve::Sqrt,
                trap_phase_strength: 0.0,
                de_shade: None,
                mark_glitches: false,
            },
        ),
        (
            "sep-de",
            ColorParams {
                density: 0.5,
                offset: 0.5,
                channel: ColorChannel::De,
                interior: InteriorMode::Black,
                trap_scale: 1.0,
                trap_curve: TrapCurve::Sqrt,
                trap_phase_strength: 0.0,
                de_shade: Some(2.0),
                mark_glitches: false,
            },
        ),
    ];

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-out");
    std::fs::create_dir_all(&out_dir).unwrap();

    for (name, params) in &variants {
        let img = render::shade_and_downsample(
            &buf.samples,
            frame.out_width,
            frame.out_height,
            buf.ss,
            &palette,
            params,
            spacing,
        );
        let path = out_dir.join(format!("{name}.png"));
        img.save(&path).unwrap();
        println!("wrote {}", path.display());
    }

    // The crux: shading three times did not invoke the backend again.
    let after_shade = counting.count.load(Ordering::Relaxed);
    assert_eq!(
        after_shade, after_iter,
        "shade_and_downsample must not re-iterate (sample count moved)"
    );
    println!(
        "iteration ran once: {after_iter} subpixel samples, then 3 colorings with 0 further samples"
    );
}
