//! High-precision scalar support for the perturbation reference orbit.
//!
//! The only arbitrary-precision work in the engine is a *single* reference
//! orbit at the frame center (see [`crate::backend::PerturbationBackend`]).
//! The center coordinate needs many bits, but the orbit *values* stay O(1)
//! until escape, so everything outside this module is plain `f64`.
//!
//! Crate choice: **astro-float** (pure Rust, no C dependency). We need only
//! decimal parsing, the three field ops (handled inline by the backend), and a
//! fast projection back to `f64` — implemented here without the heavy decimal
//! formatter so it is cheap enough to call twice per reference iteration.

use astro_float::{BigFloat, Consts, Radix, RoundingMode, Sign};

/// Rounding mode for parsing the center. Correct rounding of the *input* matters
/// (it is the ground truth the whole orbit derives from); orbit arithmetic uses
/// `RoundingMode::None` for speed since results are projected to `f64` anyway.
const PARSE_RM: RoundingMode = RoundingMode::ToEven;

/// Mantissa precision (bits) for the reference orbit and center.
///
/// We need enough bits that the center coordinate is resolved well below one
/// pixel: `log2(out_width / frame_width)` bits separate the frame edge from a
/// pixel, plus a 64-bit guard for accumulated round-off over the orbit. Floored
/// at f64's 53 bits so shallow frames never ask for *less* than `f64`.
pub fn prec_bits(out_width: u32, frame_width: f64) -> usize {
    let ratio = out_width as f64 / frame_width;
    let bits = ratio.log2().ceil() as i64 + 64;
    bits.max(53) as usize
}

/// Parse an arbitrary-precision decimal string into a `BigFloat` at `prec_bits`.
///
/// Accepts ordinary decimal (`-0.743643887...`) and scientific (`1.3e-2`) forms.
/// Returns an error string on malformed input rather than a silent NaN.
pub fn parse_decimal(s: &str, prec_bits: usize) -> Result<BigFloat, String> {
    let mut cc = Consts::new().map_err(|e| format!("astro-float init failed: {e:?}"))?;
    let v = BigFloat::parse(s.trim(), Radix::Dec, prec_bits, PARSE_RM, &mut cc);
    if v.is_nan() || v.is_inf() {
        return Err(format!("could not parse high-precision decimal '{s}'"));
    }
    Ok(v)
}

/// Project a `BigFloat` to the nearest (toward-zero) `f64`.
///
/// Replicates astro-float's internal `to_f64` from the public raw-parts view:
/// the last mantissa word holds the most-significant 64 bits (bit 63 is the
/// implicit leading 1 for normal numbers), and the stored exponent is unbiased.
/// This avoids the decimal formatter entirely, which would dominate the
/// reference-orbit cost. Conversion truncates (< 1 ulp), which is irrelevant
/// against the f64 delta arithmetic that follows.
pub fn to_f64(x: &BigFloat) -> f64 {
    let (m, _n, sign, e, _inexact) = match x.as_raw_parts() {
        Some(parts) => parts,
        None => return f64::NAN, // Inf / NaN
    };
    let top = match m.last() {
        Some(&w) => w as u64, // most-significant word (64-bit target: Word = u64)
        None => return 0.0,
    };
    if top == 0 {
        return 0.0;
    }
    let neg = matches!(sign, Sign::Neg);
    let sign_bit = if neg { 1u64 << 63 } else { 0 };

    let eb: i64 = e as i64 + 1023; // f64 exponent bias
    if eb >= 0x7ff {
        return if neg { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if eb <= 0 {
        // Gradual underflow toward zero — a safety net; orbit values are O(1).
        let shift = (-eb) as u64;
        if shift < 52 {
            return f64::from_bits(sign_bit | (top >> (shift + 12)));
        }
        return f64::from_bits(sign_bit); // ±0
    }
    // Normal: drop the implicit leading 1, keep the top 52 fraction bits.
    let mant = top << 1;
    f64::from_bits(sign_bit | ((eb as u64 - 1) << 52) | (mant >> 12))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `to_f64` must agree with a decimal round-trip across magnitudes and signs.
    #[test]
    fn to_f64_matches_decimal_roundtrip() {
        let p = 256;
        let cases = [
            "0.0",
            "1.0",
            "-1.0",
            "0.7436438870371587",
            "-0.743643887037158704752191506114774",
            "0.131825904205311970493132056385139",
            "123456.789",
            "-2.5e-3",
            "9.99e5",
            "3.141592653589793238462643383279502884",
        ];
        for s in cases {
            let bf = parse_decimal(s, p).unwrap();
            let got = to_f64(&bf);
            let want: f64 = s.parse().unwrap();
            // Toward-zero truncation of ~256-bit mantissa: within a few ulp of
            // the directly-parsed f64.
            let tol = want.abs() * 4.0 * f64::EPSILON + f64::MIN_POSITIVE;
            assert!(
                (got - want).abs() <= tol,
                "to_f64('{s}') = {got:e}, want {want:e}, tol {tol:e}"
            );
        }
    }
}
