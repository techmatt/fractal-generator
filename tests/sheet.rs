//! Contact-sheet separability: iterate one location **once**, then shade it
//! across N palettes with zero further `sample()` calls. Also exercises the
//! `.ugr`/`.map` asset loaders end-to-end (parse → OKLab bake → lookup).

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex;

use fractal_generator::backend::{F64Backend, FractalBackend, PixelSample, Trap, TrapShape};
use fractal_generator::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use fractal_generator::palette::{cubehelix, viridis, Palette};
use fractal_generator::palette_io::load_palette_file;
use fractal_generator::render::{self, Frame};
use fractal_generator::sheet;

/// Backend wrapper that counts `sample()` invocations.
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

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/palettes")
}

#[test]
fn contact_sheet_iterates_once_over_n_palettes() {
    let frame = Frame {
        center: Complex::new(-0.745, 0.113),
        frame_width: 0.02,
        out_width: 200,
        out_height: 134,
    };
    let ss = 2u32;
    let trap = Trap {
        shape: TrapShape::Circle,
        center: Complex::new(0.0, 0.0),
        radius: 0.5,
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
    assert_eq!(after_iter, expected, "iteration should sample each subpixel once");

    // Assemble a mixed palette set: .ugr block + .map + cubehelix + viridis + default.
    let dir = assets_dir();
    let ember = load_palette_file(&dir.join("sample.ugr"), Some("Ember"), false).unwrap();
    let map = load_palette_file(&dir.join("sample.map"), None, false).unwrap();
    let palettes = vec![
        Palette::ultra_fractal(),
        ember,
        map,
        cubehelix(false),
        viridis(false),
    ];

    let params = ColorParams {
        density: 2.0,
        offset: 0.1,
        channel: ColorChannel::Trap,
        interior: InteriorMode::Trap,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    };

    // Stage 2: N colorings from the SAME buffer.
    let (grid, legend) =
        sheet::render_contact_sheet(&buf, &palettes, &params, frame.pixel_size(), None);

    // The crux: composing the sheet did not re-invoke the backend.
    let after_shade = counting.count.load(Ordering::Relaxed);
    assert_eq!(
        after_shade, after_iter,
        "contact sheet must not re-iterate (sample count moved)"
    );

    assert_eq!(legend.len(), palettes.len());
    assert!(grid.width() > frame.out_width && grid.height() > frame.out_height);

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let path = out_dir.join("contact-sheet.png");
    grid.save(&path).unwrap();
    println!(
        "iteration ran once: {after_iter} subpixel samples, then {} colorings with 0 further samples",
        palettes.len()
    );
    for l in &legend {
        println!("{l}");
    }
    println!("wrote {}", path.display());
}

/// The asset loaders bake usable palettes and reproduce a known stop color.
#[test]
fn loaders_bake_known_colors() {
    let dir = assets_dir();
    // Ember index=200 → pos 0.5 → sRGB (255,140,0).
    let ember = load_palette_file(&dir.join("sample.ugr"), Some("Ember"), false).unwrap();
    let lin = ember.lookup_linear(0.5);
    let want = [
        fractal_generator::palette::srgb_to_linear(255.0 / 255.0),
        fractal_generator::palette::srgb_to_linear(140.0 / 255.0),
        fractal_generator::palette::srgb_to_linear(0.0 / 255.0),
    ];
    for k in 0..3 {
        assert!(
            (lin[k] - want[k]).abs() < 5e-3,
            "Ember@0.5 channel {k}: got {} want {}",
            lin[k],
            want[k]
        );
    }
    assert_eq!(ember.name(), "Ember");

    // .map loads with 256 stops and a tight seam (cyclic cosine palette).
    let map = load_palette_file(&dir.join("sample.map"), None, false).unwrap();
    let a = map.lookup_linear(0.0);
    let b = map.lookup_linear(0.999);
    let d: f64 = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
    assert!(d < 0.05, "map seam jump {d}");
    assert_eq!(map.name(), "sample");
}
