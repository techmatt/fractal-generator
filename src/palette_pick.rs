//! `palette-pick` subcommand: diagnostic palette favorite-picker.
//!
//! Renders **one fixed dense fractal field** (the seahorse-valley spiral, the
//! handoff's preferred palette-reading location) **once**, then re-shades that
//! single iterated buffer across N palettes sampled from the survivor colormap
//! library (`clean_colormaps.json`). Pure diagnosis: no band, no scoring, no
//! render-path change — Matt eyeballs the contact sheet and picks a favorite
//! diagnostic palette.
//!
//! Separability is the whole point: the expensive iteration runs one time and
//! every palette is an O(LUT) re-shade ([`sheet::render_contact_sheet`]). The
//! sample is deterministic for a fixed `--seed` (SplitMix64 partial Fisher-Yates
//! over the fixed library order), so the sheet is reproducible and every tile is
//! identifiable by its burned-in index via the printed/saved legend.

use std::fmt::Write as _;

use num_complex::Complex;

use crate::backend::{Trap, TrapShape};
use crate::cli::{BackendChoice, PalettePickArgs};
use crate::coloring::{ColorChannel, ColorParams, InteriorMode, TrapCurve};
use crate::hp;
use crate::palette::Palette;
use crate::probe::{render_mandel_panel, SplitMix64};
use crate::sheet;

pub fn run_palette_pick(args: &PalettePickArgs) -> Result<(), String> {
    // 1. Load + parse the survivor colormap library.
    let text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("failed to read '{}': {e}", args.colormaps))?;
    let library = parse_colormaps(&text)
        .map_err(|e| format!("parsing '{}': {e}", args.colormaps))?;
    if library.is_empty() {
        return Err(format!("no colormaps in '{}'", args.colormaps));
    }

    // 2. Deterministic sample of `--count` distinct palettes (partial
    //    Fisher-Yates over the fixed library order, seeded). Selection order is
    //    the tile order — reproducible for a fixed seed.
    let want = args.count.min(library.len());
    if args.count > library.len() {
        eprintln!(
            "note: requested {} palettes but library has only {}; using all {}",
            args.count,
            library.len(),
            library.len()
        );
    }
    let mut rng = SplitMix64(args.seed);
    let mut idx: Vec<usize> = (0..library.len()).collect();
    for i in 0..want {
        let j = i + rng.below(library.len() - i);
        idx.swap(i, j);
    }
    idx.truncate(want);

    let palettes: Vec<Palette> = idx
        .iter()
        .map(|&i| {
            let cm = &library[i];
            // Selective seam fix: SEQUENTIAL (mirror_needed) maps bake pre-mirrored
            // (out-and-back); cyclic maps stay single-pass.
            Palette::from_srgb8_stops_mirrored(cm.name.clone(), &cm.stops, false, cm.mirror_needed)
        })
        .collect();

    // 3. Iterate the fixed field ONCE (f64 cheap-regime — shallow spiral).
    let panel_w = args.tile_width.max(1);
    let panel_h = (panel_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let prec = hp::prec_bits(panel_w, args.frame_width);
    let center_re = hp::parse_decimal(&args.center_re, prec)?;
    let center_im = hp::parse_decimal(&args.center_im, prec)?;
    let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));
    let trap = Trap {
        shape: TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    eprintln!(
        "palette-pick: field ({}, {}) width {:.3e}, maxiter {}, tile {}x{} ss{}, {} palettes (seed {})",
        args.center_re,
        args.center_im,
        args.frame_width,
        args.maxiter,
        panel_w,
        panel_h,
        args.supersample,
        palettes.len(),
        args.seed,
    );

    let panel = render_mandel_panel(
        &center_re,
        &center_im,
        center_f64,
        args.frame_width,
        panel_w,
        panel_h,
        args.supersample,
        args.maxiter,
        args.bailout,
        prec,
        trap,
        BackendChoice::Auto,
    );

    // Density sanity read: a flat field (mostly interior or mostly fast-escape)
    // reads palette character poorly. Report the escaped fraction so a sparse
    // field is visible without opening the sheet.
    let total = panel.buf.samples.len();
    let escaped = panel.buf.samples.iter().filter(|s| s.escaped).count();
    let esc_frac = escaped as f64 / total.max(1) as f64;
    eprintln!(
        "field density: {:.1}% escaped ({} backend); <~15% or >~95% reads flat — \
         if so, swap to a denser corridor keeper via --center-*/--frame-width",
        esc_frac * 100.0,
        panel.backend_name,
    );

    // 4. Re-shade across every palette into one contact sheet (default coloring,
    //    matching `render`'s defaults — smooth escape time).
    let params = ColorParams {
        density: 0.025,
        offset: 0.0,
        channel: ColorChannel::Smooth,
        interior: InteriorMode::Black,
        trap_scale: 1.0,
        trap_curve: TrapCurve::Sqrt,
        trap_phase_strength: 0.0,
        de_shade: None,
        mark_glitches: false,
    };
    let (grid, legend) =
        sheet::render_contact_sheet(&panel.buf, &palettes, &params, panel.spacing, args.cols);

    // 5. Persist the sheet + a reproducibility legend OUTSIDE out/ (load-bearing
    //    pick artifact — survives `rm -r out/*`).
    let sheet_path = format!("{}/palette_pick.png", args.out_dir.trim_end_matches('/'));
    let legend_path = format!("{}/palette_pick_legend.txt", args.out_dir.trim_end_matches('/'));
    crate::ensure_parent_dir(&sheet_path)?;
    grid.save(&sheet_path)
        .map_err(|e| format!("failed to write {sheet_path}: {e}"))?;

    let mut legend_txt = String::new();
    writeln!(legend_txt, "# diagnostic-palette pick").ok();
    writeln!(
        legend_txt,
        "# field: center ({}, {}), frame_width {:.6e}, maxiter {}",
        args.center_re, args.center_im, args.frame_width, args.maxiter
    )
    .ok();
    writeln!(
        legend_txt,
        "# tile {panel_w}x{panel_h} ss{}, escaped {:.1}%",
        args.supersample,
        esc_frac * 100.0
    )
    .ok();
    writeln!(
        legend_txt,
        "# library: {} ({} entries), sampled {} with seed {}",
        args.colormaps,
        library.len(),
        palettes.len(),
        args.seed
    )
    .ok();
    writeln!(legend_txt, "# tile order is deterministic (selection order).").ok();
    for line in &legend {
        writeln!(legend_txt, "{line}").ok();
    }
    std::fs::write(&legend_path, &legend_txt)
        .map_err(|e| format!("failed to write {legend_path}: {e}"))?;

    for line in &legend {
        println!("{line}");
    }
    eprintln!("wrote {sheet_path} ({} tiles) + {legend_path}", palettes.len());
    Ok(())
}

/// A colormap parsed out of `clean_colormaps.json`: a name, its sRGB8 stops, the
/// inline `mirror_needed` classification (SEQUENTIAL → pre-mirror at bake), and
/// the `cycle` label (provenance only). Shared with `palette_score`.
pub(crate) struct Colormap {
    pub(crate) name: String,
    pub(crate) stops: Vec<(f64, [u8; 3])>,
    pub(crate) mirror_needed: bool,
    /// Inline `cycle` classification string (e.g. "cyclic" / "sequential"), if present.
    pub(crate) cycle: Option<String>,
}

/// Parse the survivor colormap library: a JSON array of
/// `{"name": str, "source": str, "stops": [[pos, [r,g,b]], ...]}` objects.
/// Hand-rolled (the project avoids serde) via a tiny generic JSON parser, then a
/// schema projection — `source` and any other keys are ignored.
pub(crate) fn parse_colormaps(text: &str) -> Result<Vec<Colormap>, String> {
    let v = JsonParser::new(text).parse()?;
    let arr = match v {
        Json::Arr(a) => a,
        _ => return Err("top-level JSON is not an array".into()),
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, entry) in arr.into_iter().enumerate() {
        let obj = match entry {
            Json::Obj(o) => o,
            _ => return Err(format!("entry {i} is not an object")),
        };
        let mut name = None;
        let mut stops_json = None;
        let mut mirror_needed = false;
        let mut cycle = None;
        for (k, val) in obj {
            match k.as_str() {
                "name" => {
                    if let Json::Str(s) = val {
                        name = Some(s);
                    }
                }
                "stops" => stops_json = Some(val),
                "mirror_needed" => {
                    if let Json::Bool(b) = val {
                        mirror_needed = b;
                    }
                }
                "cycle" => {
                    if let Json::Str(s) = val {
                        cycle = Some(s);
                    }
                }
                _ => {}
            }
        }
        let name = name.ok_or_else(|| format!("entry {i} has no string 'name'"))?;
        let stops_arr = match stops_json {
            Some(Json::Arr(a)) => a,
            _ => return Err(format!("colormap '{name}' has no 'stops' array")),
        };
        let mut stops = Vec::with_capacity(stops_arr.len());
        for s in stops_arr {
            let pair = match s {
                Json::Arr(p) if p.len() == 2 => p,
                _ => return Err(format!("colormap '{name}': stop is not [pos, [r,g,b]]")),
            };
            let pos = match &pair[0] {
                Json::Num(n) => *n,
                _ => return Err(format!("colormap '{name}': stop position is not a number")),
            };
            let rgb = match &pair[1] {
                Json::Arr(c) if c.len() == 3 => {
                    let chan = |j: &Json| -> Result<u8, String> {
                        match j {
                            Json::Num(n) => Ok(n.round().clamp(0.0, 255.0) as u8),
                            _ => Err(format!("colormap '{name}': non-numeric color channel")),
                        }
                    };
                    [chan(&c[0])?, chan(&c[1])?, chan(&c[2])?]
                }
                _ => return Err(format!("colormap '{name}': color is not [r,g,b]")),
            };
            stops.push((pos, rgb));
        }
        if stops.len() < 2 {
            return Err(format!("colormap '{name}': need ≥2 stops, found {}", stops.len()));
        }
        out.push(Colormap { name, stops, mirror_needed, cycle });
    }
    Ok(out)
}

/// Minimal JSON value (only what the colormap schema needs; objects keep insertion order).
/// Shared with `palette_score` (views.json parsing).
#[allow(dead_code)] // Null/Bool are parsed for completeness but the schema never reads them.
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// A tiny recursive-descent JSON parser over byte input. Sufficient for the
/// well-formed colormap library; not a general-purpose validator.
pub(crate) struct JsonParser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> JsonParser<'a> {
    pub(crate) fn new(s: &'a str) -> Self {
        JsonParser { b: s.as_bytes(), i: 0 }
    }

    pub(crate) fn parse(&mut self) -> Result<Json, String> {
        self.ws();
        let v = self.value()?;
        self.ws();
        if self.i != self.b.len() {
            return Err(format!("trailing data at byte {}", self.i));
        }
        Ok(v)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected byte {other:?} at {}", self.i)),
        }
    }

    fn literal(&mut self, lit: &str, val: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(val)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // {
        let mut obj = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(obj));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err(format!("expected ':' at byte {}", self.i));
            }
            self.i += 1;
            let val = self.value()?;
            obj.push((key, val));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(obj));
                }
                other => return Err(format!("expected ',' or '}}' at byte {}, got {other:?}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // [
        let mut arr = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(arr));
        }
        loop {
            let val = self.value()?;
            arr.push(val);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(arr));
                }
                other => return Err(format!("expected ',' or ']' at byte {}, got {other:?}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(format!("expected '\"' at byte {}", self.i));
        }
        self.i += 1;
        let mut s = String::new();
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'u' => {
                            let hex = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or("truncated \\u escape")?;
                            let cp = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| "bad \\u hex")?,
                                16,
                            )
                            .map_err(|_| "bad \\u hex")?;
                            self.i += 4;
                            s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        other => return Err(format!("bad escape '\\{}'", other as char)),
                    }
                }
                _ => {
                    // Pass through raw UTF-8 bytes (names are ASCII in practice).
                    s.push(c as char);
                }
            }
        }
        Err("unterminated string".into())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while let Some(&c) = self.b.get(self.i) {
            if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.i += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "bad number utf8")?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number '{s}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_colormap_schema() {
        let txt = r#"[
            {"name": "a", "source": "x", "stops": [[0.0, [0, 0, 4]], [1.0, [252, 253, 191]]]},
            {"name": "b", "source": "y", "stops": [[0.0, [1, 2, 3]], [0.5, [10, 20, 30]], [1.0, [255, 255, 255]]]}
        ]"#;
        let cms = parse_colormaps(txt).unwrap();
        assert_eq!(cms.len(), 2);
        assert_eq!(cms[0].name, "a");
        assert_eq!(cms[0].stops.len(), 2);
        assert_eq!(cms[0].stops[0], (0.0, [0, 0, 4]));
        assert_eq!(cms[1].stops[2].1, [255, 255, 255]);
        // Absent mirror_needed defaults to false (no mirror).
        assert!(!cms[0].mirror_needed && !cms[1].mirror_needed);
    }

    /// The inline `mirror_needed` label is read, and a sequential map bakes
    /// pre-mirrored (density_scale 0.5); a cyclic map stays single-pass (1.0).
    #[test]
    fn reads_mirror_needed_and_compensates_density() {
        use crate::palette::{Palette, MIRROR_DENSITY_SCALE};
        let txt = r#"[
            {"name": "seq", "stops": [[0.0,[0,0,4]],[1.0,[252,253,191]]], "cycle": "sequential", "mirror_needed": true},
            {"name": "cyc", "stops": [[0.0,[10,20,30]],[0.5,[200,180,90]]], "cycle": "cyclic", "mirror_needed": false}
        ]"#;
        let cms = parse_colormaps(txt).unwrap();
        assert!(cms[0].mirror_needed, "sequential entry must read mirror_needed=true");
        assert!(!cms[1].mirror_needed, "cyclic entry must read mirror_needed=false");
        let seq = Palette::from_srgb8_stops_mirrored("seq", &cms[0].stops, false, cms[0].mirror_needed);
        let cyc = Palette::from_srgb8_stops_mirrored("cyc", &cms[1].stops, false, cms[1].mirror_needed);
        assert_eq!(seq.density_scale(), MIRROR_DENSITY_SCALE);
        assert_eq!(cyc.density_scale(), 1.0);
    }

    #[test]
    fn rejects_non_array_root() {
        assert!(parse_colormaps(r#"{"name": "a"}"#).is_err());
    }
}
