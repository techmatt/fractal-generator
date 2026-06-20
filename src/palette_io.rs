//! Palette file loaders (`.ugr`, `.map`) and the name/path resolver.
//!
//! Both formats decode to sRGB8 control points, which [`crate::palette`] then
//! lifts to OKLab and bakes. The resolver dispatches a `--palette` spec: a
//! built-in name (`default`, `cubehelix`, `viridis`) or a path whose extension
//! picks the parser.

use std::path::Path;

use crate::palette::{builtin, Palette};

/// A parsed gradient: a block name and its sRGB8 control points.
pub struct RawGradient {
    pub name: String,
    pub stops: Vec<(f64, [u8; 3])>,
}

/// Resolve a `--palette` spec to a baked [`Palette`].
///
/// - A built-in name (`default` / `cubehelix` / `viridis`) → generated palette.
/// - Otherwise a path; `.ugr` selects a named block (`entry`, default first),
///   `.map` is a single gradient.
pub fn load_palette(spec: &str, entry: Option<&str>, reverse: bool) -> Result<Palette, String> {
    if let Some(p) = builtin(spec, reverse) {
        return Ok(p);
    }
    load_palette_file(Path::new(spec), entry, reverse)
}

/// Load a palette from a file path (extension dispatch).
pub fn load_palette_file(
    path: &Path,
    entry: Option<&str>,
    reverse: bool,
) -> Result<Palette, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read palette '{}': {e}", path.display()))?;

    match ext.as_str() {
        "ugr" => {
            let grads = parse_ugr(&text)
                .map_err(|e| format!("parsing '{}': {e}", path.display()))?;
            if grads.is_empty() {
                return Err(format!("no gradient blocks in '{}'", path.display()));
            }
            let chosen = match entry {
                Some(name) => grads
                    .iter()
                    .find(|g| g.name.eq_ignore_ascii_case(name))
                    .ok_or_else(|| {
                        let names: Vec<&str> = grads.iter().map(|g| g.name.as_str()).collect();
                        format!(
                            "no gradient '{name}' in '{}' (have: {})",
                            path.display(),
                            names.join(", ")
                        )
                    })?,
                None => &grads[0],
            };
            Ok(Palette::from_srgb8_stops(
                chosen.name.clone(),
                &chosen.stops,
                reverse,
            ))
        }
        "map" => {
            let stops =
                parse_map(&text).map_err(|e| format!("parsing '{}': {e}", path.display()))?;
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("map")
                .to_string();
            Ok(Palette::from_srgb8_stops(name, &stops, reverse))
        }
        other => Err(format!(
            "unknown palette extension '.{other}' for '{}' (expected .ugr or .map)",
            path.display()
        )),
    }
}

/// Parse a Fractint `.map`: lines of `R G B` (0–255). Blank lines and `;`/`#`
/// comments are skipped; the first three integers on a line are the color.
/// `position = i / N` for the `i`-th color (tolerates `N != 256`).
pub fn parse_map(text: &str) -> Result<Vec<(f64, [u8; 3])>, String> {
    let mut colors: Vec<[u8; 3]> = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.split([';', '#']).next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let nums: Vec<&str> = line.split_whitespace().collect();
        if nums.len() < 3 {
            return Err(format!("line {}: expected 'R G B', got '{raw}'", lineno + 1));
        }
        let parse = |s: &str| -> Result<u8, String> {
            let v: i64 = s
                .parse()
                .map_err(|_| format!("line {}: bad channel '{s}'", lineno + 1))?;
            Ok(v.clamp(0, 255) as u8)
        };
        colors.push([parse(nums[0])?, parse(nums[1])?, parse(nums[2])?]);
    }
    if colors.len() < 2 {
        return Err(format!("need ≥2 colors, found {}", colors.len()));
    }
    let n = colors.len();
    Ok(colors
        .into_iter()
        .enumerate()
        .map(|(i, c)| (i as f64 / n as f64, c))
        .collect())
}

/// Parse an UltraFractal `.ugr`: one or more named blocks `NAME { ... }`. Inside
/// a block, `index=N color=INT` pairs define stops (`index` 0–400 →
/// `position = index/400`). `color` is a COLORREF `0x00BBGGRR` (R is the low
/// byte), usually written in decimal. Other keys (`title=`, `smooth=`,
/// `rotation=`, the `opacity:` section) are ignored.
pub fn parse_ugr(text: &str) -> Result<Vec<RawGradient>, String> {
    // Tokenize on whitespace but keep `{` / `}` as their own tokens so block
    // boundaries are unambiguous regardless of layout.
    let spaced = text.replace('{', " { ").replace('}', " } ");
    let mut tokens = spaced.split_whitespace().peekable();

    let mut grads: Vec<RawGradient> = Vec::new();
    // The block name is the identifier immediately preceding `{`. Track the last
    // non-`{`/`}` token to recover it.
    let mut last_ident: Option<String> = None;

    while let Some(tok) = tokens.next() {
        match tok {
            "{" => {
                let name = last_ident.take().unwrap_or_else(|| format!("gradient{}", grads.len()));
                let stops = parse_ugr_block(&mut tokens)?;
                if !stops.is_empty() {
                    grads.push(RawGradient { name, stops });
                }
            }
            "}" => { /* unbalanced close; ignore defensively */ }
            other => last_ident = Some(other.to_string()),
        }
    }
    Ok(grads)
}

/// Consume tokens until the matching `}`; collect `index=`/`color=` stops.
/// `index` without a following `color` (e.g. the `opacity:` section) is dropped.
fn parse_ugr_block<'a, I>(tokens: &mut std::iter::Peekable<I>) -> Result<Vec<(f64, [u8; 3])>, String>
where
    I: Iterator<Item = &'a str>,
{
    let mut stops: Vec<(f64, [u8; 3])> = Vec::new();
    let mut pending_index: Option<f64> = None;

    for tok in tokens.by_ref() {
        if tok == "}" {
            break;
        }
        if tok == "{" {
            // Nested brace inside a block is unexpected; skip rather than error.
            continue;
        }
        if let Some(val) = tok.strip_prefix("index=") {
            let idx: f64 = val
                .parse()
                .map_err(|_| format!("bad index '{val}'"))?;
            pending_index = Some((idx / 400.0).rem_euclid(1.0));
        } else if let Some(val) = tok.strip_prefix("color=") {
            // COLORREF, usually decimal; tolerate 0x-prefixed hex too.
            let colorref: i64 = if let Some(hex) = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X")) {
                i64::from_str_radix(hex, 16).map_err(|_| format!("bad color '{val}'"))?
            } else {
                val.parse().map_err(|_| format!("bad color '{val}'"))?
            };
            if let Some(pos) = pending_index.take() {
                let c = colorref as u64;
                let r = (c & 0xFF) as u8;
                let g = ((c >> 8) & 0xFF) as u8;
                let b = ((c >> 16) & 0xFF) as u8;
                stops.push((pos, [r, g, b]));
            }
        }
        // Any other token (title=, smooth=, opacity=, section labels) is ignored.
    }
    Ok(stops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_parses_positions() {
        let txt = "; a comment\n255 0 0\n0 255 0\n0 0 255\n255 255 0\n";
        let stops = parse_map(txt).unwrap();
        assert_eq!(stops.len(), 4);
        assert_eq!(stops[0], (0.0, [255, 0, 0]));
        assert!((stops[1].0 - 0.25).abs() < 1e-9);
        assert_eq!(stops[3].1, [255, 255, 0]);
    }

    #[test]
    fn ugr_parses_blocks_and_colorref() {
        // R=200, G=30, B=0 → COLORREF 200 + 30*256 = 7880.
        let txt = "\
Ember {
gradient:
  title=\"Ember\" smooth=yes
  index=0 color=10
  index=200 color=7880
opacity:
  smooth=no index=0 opacity=255
}
Ocean {
gradient:
  index=0 color=3937290
  index=130 color=13130270
}
";
        let grads = parse_ugr(txt).unwrap();
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].name, "Ember");
        assert_eq!(grads[0].stops.len(), 2);
        // index=200 → pos 0.5; color 7880 → (200,30,0).
        assert!((grads[0].stops[1].0 - 0.5).abs() < 1e-9);
        assert_eq!(grads[0].stops[1].1, [200, 30, 0]);
        // Opacity section's index=0 must not leak in as a stop.
        assert_eq!(grads[1].name, "Ocean");
        assert_eq!(grads[1].stops.len(), 2);
    }
}
