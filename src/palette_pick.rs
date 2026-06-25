//! Survivor colormap library loader (`clean_colormaps.json`).
//!
//! Hosts the shared `.json` colormap parser — [`parse_colormaps`] → [`Colormap`]
//! (name, sRGB8 stops, the inline `mirror_needed` label) — built on a tiny
//! hand-rolled JSON reader ([`Json`] / [`JsonParser`]; the project avoids serde).
//! Loaded by `present`, `enrich`, `render_one`, and `palette_probe` to bake the
//! survivor palettes through the selective-mirror path. The diagnostic
//! `palette-pick` favorite-picker subcommand that originally lived here was
//! retired in the P2 subcommand cull; only the loader remains.

/// A colormap parsed out of `clean_colormaps.json`: a name, its sRGB8 stops, the
/// inline `mirror_needed` classification (SEQUENTIAL → pre-mirror at bake), and
/// Shared across the corpus render path (`present` / `enrich` / `render_one` /
/// `palette_probe`).
pub(crate) struct Colormap {
    pub(crate) name: String,
    pub(crate) stops: Vec<(f64, [u8; 3])>,
    pub(crate) mirror_needed: bool,
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
        out.push(Colormap { name, stops, mirror_needed });
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
