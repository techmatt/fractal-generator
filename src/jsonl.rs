//! Shared hand-rolled JSONL / manifest field readers.
//!
//! The project deliberately avoids serde (see CLAUDE.md), so every module that
//! reads a `.jsonl` / manifest line hand-rolls the same scalar extractors. This
//! is the single canonical copy (adopted from the `enrich.rs` variant, the most
//! robust of the six that had drifted).
//!
//! Tolerances: the `"{key}":` needle carries a **leading quote** so a key is
//! never matched as the suffix of a longer key (`re` won't hit `centre_re`).
//! `trim_start` after the colon accepts both compact `"k":v` and pretty
//! `"k": v` spacing; numeric extraction also `trim_matches('"')` so a quoted
//! scalar (`"k":"1.5"`) parses. All readers find the **first** occurrence of the
//! key in the line/block and return `None` on a missing or unparseable field.

/// Numeric field: `"key": 1.5` / `"key":1.5` / `"key": "1.5"` → `Some(1.5)`.
pub fn field_f64(line: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\":");
    let p = line.find(&needle)?;
    let rest = line[p + needle.len()..].trim_start();
    let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().trim_matches('"').parse::<f64>().ok()
}

/// Integer field via [`field_f64`] (tolerates a quoted or float-formatted int).
pub fn field_usize(line: &str, key: &str) -> Option<usize> {
    field_f64(line, key).map(|v| v as usize)
}

/// String field: the quoted value following `"key":` (no escape handling).
pub fn field_str(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let p = line.find(&needle)?;
    let rest = line[p + needle.len()..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Boolean field: `"key": true` / `"key":false` → `Some(bool)`.
pub fn field_bool(line: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{key}\":");
    let p = line.find(&needle)?;
    let rest = line[p + needle.len()..].trim_start();
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}
