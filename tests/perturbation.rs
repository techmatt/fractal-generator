//! Cross-backend validation: the perturbation backend must reproduce the
//! ground-truth f64 backend at a shallow location *where f64 is trustworthy*.
//!
//! We iterate pixels directly (rather than through the full render path) so we
//! can compare the raw `smooth_iter` channel and the escaped/interior
//! classification pixel-by-pixel, computing `dc` from geometry exactly as the
//! render loop does.
//!
//! Important nuance discovered while building this: at the seahorse-valley
//! location `-0.745 + 0.113i` the orbit boundary is chaotically ill-conditioned
//! past a few hundred iterations. Verified against a 60-digit mpmath reference,
//! *both* f64 and perturbation diverge by hundreds of iterations from the true
//! value for pixels that escape after ~400+ iterations — f64 is simply not a
//! valid ground truth there (in several spot checks perturbation was the closer
//! of the two). So the agreement test is run at `maxiter = 300`, the regime
//! where f64 orbits stay accurate; deeper pixels become interior in both
//! backends and are excluded. The deep renders (CLI validation #2) are where
//! perturbation's real advantage over f64 shows.

use num_complex::Complex;

use fractal_generator::backend::{F64Backend, FractalBackend, PerturbationBackend};
use fractal_generator::hp;

/// Shallow location and resolution for the agreement test.
const CENTER_RE: &str = "-0.745";
const CENTER_IM: &str = "0.113";
const FRAME_WIDTH: f64 = 0.01;
const OUT_W: u32 = 300;
const OUT_H: u32 = 200;
const BAILOUT: f64 = 1e6;
/// f64 is an accurate ground truth at this location only below this depth.
const MAXITER: u32 = 300;

#[test]
fn shallow_backends_agree() {
    let (median, max, disagreements, both_escaped, ref_len) = shallow_stats(MAXITER);

    println!(
        "shallow match @ ({CENTER_RE}, {CENTER_IM}) fw={FRAME_WIDTH:e}, {OUT_W}x{OUT_H}, \
         maxiter={MAXITER}, ref_len={ref_len}: both_escaped={both_escaped}, \
         disagreements={disagreements}, median |Δsmooth|={median:.3e}, max |Δsmooth|={max:.3e}"
    );

    assert_eq!(disagreements, 0, "classification disagreements: {disagreements}");
    assert!(both_escaped > 1000, "too few escaped pixels to be meaningful");
    assert!(median < 1e-8, "median |Δsmooth| {median:.3e} >= 1e-8");
    assert!(max < 1e-2, "max |Δsmooth| {max:.3e} >= 1e-2");
}

/// Returns `(median, max, disagreements, both_escaped, ref_len)` for the
/// f64-vs-perturbation comparison over the shallow frame at `maxiter`.
fn shallow_stats(maxiter: u32) -> (f64, f64, u64, u64, usize) {
    let prec_bits = hp::prec_bits(OUT_W, FRAME_WIDTH);
    let cre = hp::parse_decimal(CENTER_RE, prec_bits).unwrap();
    let cim = hp::parse_decimal(CENTER_IM, prec_bits).unwrap();
    let center = Complex::new(hp::to_f64(&cre), hp::to_f64(&cim));

    let f64b = F64Backend::new(maxiter, BAILOUT);
    let pb = PerturbationBackend::new(&cre, &cim, maxiter, BAILOUT, prec_bits);

    let fh = FRAME_WIDTH * (OUT_H as f64 / OUT_W as f64);

    let mut diffs: Vec<f64> = Vec::new();
    let mut disagreements: u64 = 0;
    let mut both_escaped: u64 = 0;

    for row in 0..OUT_H {
        let py_frac = (row as f64 + 0.5) / OUT_H as f64;
        let dc_im = (0.5 - py_frac) * fh;
        for col in 0..OUT_W {
            let px_frac = (col as f64 + 0.5) / OUT_W as f64;
            let dc_re = (px_frac - 0.5) * FRAME_WIDTH;
            let dc = Complex::new(dc_re, dc_im);
            let c = center + dc;

            let a = f64b.sample(c, dc);
            let b = pb.sample(c, dc);

            if a.escaped != b.escaped {
                disagreements += 1;
                continue;
            }
            if a.escaped && b.escaped {
                both_escaped += 1;
                diffs.push((a.smooth_iter - b.smooth_iter).abs());
            }
        }
    }

    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = diffs.get(diffs.len() / 2).copied().unwrap_or(0.0);
    let max = diffs.iter().cloned().fold(0.0_f64, f64::max);
    (median, max, disagreements, both_escaped, pb.ref_len())
}
