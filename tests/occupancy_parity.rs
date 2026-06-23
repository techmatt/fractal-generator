//! Rust↔Python parity for the loose0 detail-occupancy gate.
//!
//! `present`'s occupancy gate ports the Python `score_complexity.py` reduction
//! (which chose the 0.23 floor) onto `energy::edge_energy`. The 0.23 floor only
//! transfers if the Rust occupancy reproduces the Python `complexity_scores.json`
//! scores. This test runs `energy::occupancy` (floor 0.010, 32×18) on the
//! existing loose0 crops and compares to the persisted JSON, printing max/median
//! abs difference. Ignored by default (depends on the loose0 corpus on disk);
//! run explicitly:
//!
//!   cargo test --release --test occupancy_parity -- --ignored --nocapture

use std::path::Path;

use fractal_generator::energy::{occupancy, OCC_FLOOR, OCC_GX, OCC_GY};

const CROP_DIR: &str = "data/label_crops/loose0/loose0";
const SCORES: &str = "data/label_crops/loose0/loose0/complexity_scores.json";

/// (filename, score-at-0.010) pairs pulled from the compact JSON. Each object is
/// `..."score":X,...,"file":"NAME.png"...`; score precedes its file with no other
/// score between, so pairing score→next-file is unambiguous.
fn parse_scores(text: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(p) = text[i..].find("\"score\":").map(|p| p + i) {
        let after = p + "\"score\":".len();
        let rest = &text[after..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+'))
            .unwrap_or(rest.len());
        let score: f64 = match rest[..end].parse() {
            Ok(v) => v,
            Err(_) => {
                i = after;
                continue;
            }
        };
        // the file belonging to this score is the next "file":"..."
        if let Some(fp) = rest.find("\"file\":\"") {
            let fstart = after + fp + "\"file\":\"".len();
            if let Some(fend) = text[fstart..].find('"') {
                out.push((text[fstart..fstart + fend].to_string(), score));
            }
        }
        i = after + end;
    }
    out
}

#[test]
#[ignore = "depends on the loose0 corpus + complexity_scores.json on disk"]
fn occupancy_matches_python_calibration() {
    let text = std::fs::read_to_string(SCORES)
        .unwrap_or_else(|e| panic!("read {SCORES}: {e}"));
    let pairs = parse_scores(&text);
    assert!(!pairs.is_empty(), "no (file,score) pairs parsed from {SCORES}");

    // Anchors named in the prompt must be in the comparison.
    let anchors = [
        "3_", "7_", "14_", "20_", "26_", "0_", "11_", "27_",
    ];
    let dir = Path::new(CROP_DIR);

    let mut diffs: Vec<(f64, String, f64, f64)> = Vec::new(); // (absdiff, file, py, rs)
    let mut anchor_rows: Vec<(String, f64, f64)> = Vec::new();
    let mut missing = 0usize;

    for (file, py) in &pairs {
        let path = dir.join(file);
        let img = match image::open(&path) {
            Ok(im) => im.to_rgb8(),
            Err(_) => {
                missing += 1;
                continue;
            }
        };
        let rs = occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR);
        let d = (rs - py).abs();
        if anchors.iter().any(|a| file.starts_with(a)) {
            anchor_rows.push((file.clone(), *py, rs));
        }
        diffs.push((d, file.clone(), *py, rs));
    }

    assert!(!diffs.is_empty(), "no crops decoded under {CROP_DIR}");
    diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = diffs.len();
    let median = diffs[n / 2].0;
    let (maxd, ref maxfile, maxpy, maxrs) = diffs[n - 1];

    println!("\n=== occupancy parity (floor {OCC_FLOOR}, {OCC_GX}x{OCC_GY}) ===");
    println!("compared {n} crops ({missing} missing on disk)");
    println!("median abs diff = {median:.6}");
    println!("max    abs diff = {maxd:.6}  ({maxfile}: py={maxpy:.4} rs={maxrs:.4})");
    println!("\nanchors (py vs rs):");
    anchor_rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (f, py, rs) in &anchor_rows {
        println!("  {f:<48} py={py:.4} rs={rs:.4} |Δ|={:.5}", (rs - py).abs());
    }

    // edge_energy + srgb8_to_oklab are matched byte-for-byte; allow only
    // PNG-decode / float-order noise.
    assert!(maxd < 0.01, "max parity diff {maxd:.6} exceeds 0.01 — port diverges from the calibration");
}
