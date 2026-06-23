//! Dump a baked palette LUT to a file so the Python `coloring.bake_lut` port can
//! be checked byte-for-byte against the Rust reference (the verbatim-port
//! invariant). Ignored by default; driven by env vars from
//! `palette_extractor/check_bytematch.py`:
//!
//!   DUMP_STOPS = "pos,r,g,b;pos,r,g,b;..."   (sRGB8 stops)
//!   DUMP_MIRROR = "0" | "1"                  (selective pre-mirror)
//!   DUMP_OUT   = path                        (LUT_SIZE*3 little-endian f64, linear RGB)
//!
//! The dump samples `lookup_linear(i / LUT_SIZE)` for i in 0..LUT_SIZE, which
//! returns LUT entry `i` exactly (fractional part 0), i.e. the baked table.

use fractal_generator::palette::{Palette, LUT_SIZE};
use std::io::Write;

#[test]
#[ignore = "driven by check_bytematch.py via env vars"]
fn dump_lut() {
    let spec = match std::env::var("DUMP_STOPS") {
        Ok(s) => s,
        Err(_) => return, // not invoked as a dump
    };
    let mirror = std::env::var("DUMP_MIRROR").map(|v| v == "1").unwrap_or(false);
    let out = std::env::var("DUMP_OUT").expect("DUMP_OUT not set");

    let stops: Vec<(f64, [u8; 3])> = spec
        .split(';')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let f: Vec<&str> = t.split(',').collect();
            (
                f[0].parse::<f64>().unwrap(),
                [
                    f[1].parse::<u8>().unwrap(),
                    f[2].parse::<u8>().unwrap(),
                    f[3].parse::<u8>().unwrap(),
                ],
            )
        })
        .collect();

    let pal = Palette::from_srgb8_stops_mirrored("bytematch", &stops, false, mirror);

    let mut buf = Vec::with_capacity(LUT_SIZE * 3 * 8);
    for i in 0..LUT_SIZE {
        let rgb = pal.lookup_linear(i as f64 / LUT_SIZE as f64);
        for c in rgb {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(&out).expect("create DUMP_OUT");
    f.write_all(&buf).expect("write DUMP_OUT");
}
