//! `search` — global best-first frontier (beam + backtracking + diversity).
//!
//! Replaces `navigate`'s single argmax path with a **global max-priority
//! frontier** over a tree of minibrot locations. Each pop takes the highest
//! adjusted-score location *anywhere* in the tree, renders its frame, finds the
//! child minibrots (the `navigate` primitives — atom domains → Newton nuclei →
//! Munafo size), filters, scores, and pushes the survivors. When a branch
//! collapses (children filtered or scoring low) the next pop is simply the best
//! sibling from elsewhere — that *is* the backtrack, with no explicit control
//! flow. The whole thing is bounded by a **wall-clock budget**.
//!
//! This is the fix for the Prompt 6 failure (single-path dived into a
//! period-doubling cascade and pinned `c`). Two mechanisms force exploration off
//! that spine:
//!  - **Re-selection filter** (anti-cascade): a child whose nucleus sits within
//!    `k·(child width)` of an *ancestor's* nucleus is the self-targeting central
//!    nested copy — dropped. Distinct off-position sub-minibrots survive.
//!  - **Diversity-adjusted priority** `adjusted = score − λ·similarity`: keeps
//!    the frontier from filling with near-duplicates of one dive, so the
//!    off-center sibling minibrots `navigate` discarded get explored.
//!
//! Two outputs come from the cached tree: a **best-path strip** (the spine the
//! search preferred, in the `navigate`/`descend` filmstrip format) and a
//! **top-N diversity contact sheet** (farthest-point sampled over the node
//! feature vectors — the wallpaper-candidate set), plus `search.json`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use astro_float::{BigFloat, RoundingMode};
use image::{Rgb, RgbImage};
use num_complex::Complex;

use crate::backend::Trap;
use crate::cli::SearchArgs;
use crate::coloring::ColorParams;
use crate::font;
use crate::hp;
use crate::navigate::{atom_candidates, atom_candidates_spatial, newton_nucleus, size_estimate};
use crate::palette::Palette;
use crate::palette_io::load_palette;
use crate::probe::{self, SplitMix64};
use crate::sheet;

/// Rounding mode for the hp center/nucleus arithmetic (we keep these coordinates,
/// so round correctly — matches `navigate`).
const RM: RoundingMode = RoundingMode::ToEven;

/// Frame width below which f64 deltas underflow (the v1 perturbation cap) — a
/// hard safety floor regardless of budget.
const MIN_WIDTH: f64 = 1e-200;

/// Gaussian bandwidth in the diversity feature space (see [`Feature`]). Fixed:
/// the user-facing knob is λ (`--diversity`), which scales the *penalty*.
const DIVERSITY_SIGMA: f64 = 0.35;

/// K×K window for the per-node `busy_win` reported alongside the DE-coherence
/// stats (mirrors the selector / `cohere` window so the numbers are comparable).
const COHERENCE_WINDOW: i32 = 5;

// ===========================================================================
// Diversity feature vector
// ===========================================================================

/// A node's position in a small, scale-invariant feature space used for both the
/// frontier diversity penalty and the top-N farthest-point sampling.
///
/// Components (all ~O(1) so Euclidean distance is meaningful):
///  - `lp` — `log10(period)/5`, separating distinct period families.
///  - `bz` — normalized surrounding busyness, the embedded-Julia richness.
///  - `px`,`py` — the nucleus's offset **within its parent frame**, in
///    half-extent units (`[-1,1]`). Parent-relative (not absolute) on purpose:
///    deep siblings differ by ~1e-15 in absolute coordinates yet are genuinely
///    distinct, and absolute position would collapse them to one point and
///    penalize exactly the exploration we want.
#[derive(Clone, Copy)]
struct Feature {
    lp: f64,
    bz: f64,
    px: f64,
    py: f64,
}

impl Feature {
    fn new(period: u32, busyness: f64, px: f64, py: f64) -> Self {
        Feature {
            lp: (period.max(1) as f64).log10() / 5.0,
            bz: (busyness * 5.0).min(1.0),
            px: px.clamp(-1.0, 1.0),
            py: py.clamp(-1.0, 1.0),
        }
    }
    fn dist2(&self, o: &Feature) -> f64 {
        let dl = self.lp - o.lp;
        let db = self.bz - o.bz;
        let dx = self.px - o.px;
        let dy = self.py - o.py;
        dl * dl + db * db + dx * dx + dy * dy
    }
}

/// Max similarity (Gaussian over feature distance) of `f` to any already-seen
/// node — the `tree ∪ frontier` set. `1` when a near-duplicate exists, `→0` far
/// from everything.
fn similarity(f: &Feature, seen: &[Feature]) -> f64 {
    let s2 = DIVERSITY_SIGMA * DIVERSITY_SIGMA;
    seen.iter()
        .map(|s| (-f.dist2(s) / s2).exp())
        .fold(0.0, f64::max)
}

// ===========================================================================
// Node tree
// ===========================================================================

/// One node in the search tree: a minibrot location, its score/feature, and —
/// once expanded — its rendered panel and children. The root is the start
/// location (no minibrot feature).
struct Node {
    id: usize,
    parent: Option<usize>,
    depth: u32,
    is_root: bool,

    /// Frame rendered when this node is expanded (center = nucleus for non-root).
    center_re: BigFloat,
    center_im: BigFloat,
    width: f64,

    /// The minibrot feature (defaults for root).
    period: u32,
    nucleus_re: BigFloat,
    nucleus_im: BigFloat,
    size_mag: f64,
    size_arg: f64,
    busyness: f64,
    /// Nucleus offset from the *parent* frame center, in half-extent units.
    roff: f64,
    final_z2: f64,
    /// f64 projection of the nucleus — the Julia parameter and a JSON convenience.
    c_f64: Complex<f64>,
    score: f64,
    adjusted: f64,
    feat: Feature,

    // ---- filled on expansion ----
    expanded: bool,
    magnification: f64,
    maxiter: u32,
    backend: &'static str,
    glitch_count: u64,
    n_candidates: usize,
    collapse_reason: Option<String>,

    // ---- DE-coherence gate (filled on expansion; see `coherence`) ----
    /// Fraction of escaped subsamples whose boundary is finer than a pixel at the
    /// target resolution — the speckle indicator. `NaN` until expanded.
    coherence_subpixel_frac: f64,
    /// Median `de_px` over escaped subsamples (the co-gate). `NaN` until expanded.
    coherence_de_px_median: f64,
    /// Windowed-max busyness of this frame (what the selector keys on).
    busy_win: f64,
    /// Soft penalty multiplier already folded into `score` (`1.0` = unpenalized).
    coherence_penalty: f64,
    /// Hard-rejected as sub-pixel speckle → excluded from the surfaced candidates
    /// and its children are not explored.
    coherence_reject: bool,

    // ---- off-nucleus drift drive (filled on expansion when `cfg.drift`) ----
    /// Band-reward density of the surfaced (drifted or nucleus) frame — the
    /// off-nucleus objective the candidates are ranked by. `NaN` when drift is off.
    band_score: f64,
    /// The frame center was drifted off the nucleus toward the decoration centroid.
    drifted: bool,
    /// Drift offset applied to the frame center, in plane units `(re, im)`.
    drift_dc: (f64, f64),
    /// `|offset|` as a fraction of the frame half-extent (`0` when not drifted).
    drift_frac: f64,
    /// `de_px_win` at the surfaced frame's center cell (how off-boundary the new
    /// center sits). `NaN` when drift off / undefined.
    de_px_center: f64,
    /// Drifted frame center hp strings (the JSON-reported center when drifted).
    drift_re: BigFloat,
    drift_im: BigFloat,

    children: Vec<usize>,
    /// Clean shaded Mandelbrot panel (child footprint circles drawn), cached for
    /// the strip / sheet so neither output re-iterates.
    panel: Option<RgbImage>,
    panel_path: String,
}

/// A scored, filtered, Newton-refined child ready to become a [`Node`].
struct ChildBuild {
    period: u32,
    nucleus_re: BigFloat,
    nucleus_im: BigFloat,
    size_mag: f64,
    size_arg: f64,
    width: f64,
    busyness: f64,
    roff: f64,
    score: f64,
    feat: Feature,
    final_z2: f64,
    c_f64: Complex<f64>,
    /// Footprint circle in the parent panel (px, px, radius).
    circle: (f64, f64, f64),
}

/// Frontier entry: priority `key` (= adjusted score with a tiny seed jitter) and
/// the node id. Max-heap pops the largest key; exact ties break by smaller id
/// (earlier creation) for determinism.
#[derive(Clone, Copy)]
struct PqItem {
    key: f64,
    id: usize,
}
impl PartialEq for PqItem {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key && self.id == o.id
    }
}
impl Eq for PqItem {}
impl PartialOrd for PqItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for PqItem {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key
            .partial_cmp(&o.key)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.id.cmp(&self.id))
    }
}

// ===========================================================================
// Scoring (the navigate score minus the centrality bias)
// ===========================================================================

/// Hermite smoothstep clamped to `[0,1]`.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The hand-tuned score constants the Prompt 7 report flagged — now overridable
/// by the corpus's `targets.json` (Prompt 8). Defaults reproduce the original
/// constants exactly, so an absent / all-`default` targets file is a no-op.
#[derive(Clone, Copy)]
struct ScoreBands {
    /// Floor added to busyness so a low-busyness location isn't zeroed outright.
    busyness_floor: f64,
    /// A child with surrounding busyness below this scores 0 (corpus reject).
    busyness_reject_below: f64,
    busyness_from_corpus: bool,
    /// Period-band smoothstep rise edges (lower shoulder).
    period_rise: (f64, f64),
    /// Period-band smoothstep fall edges (upper shoulder).
    period_fall: (f64, f64),
    period_from_corpus: bool,
}

impl Default for ScoreBands {
    fn default() -> Self {
        ScoreBands {
            busyness_floor: 0.05,
            busyness_reject_below: 0.0,
            busyness_from_corpus: false,
            period_rise: (1.5, 3.0),
            period_fall: (20_000.0, 60_000.0),
            period_from_corpus: false,
        }
    }
}

/// Period band value `∈[0,1]` for period `pf`, using the (possibly corpus-set)
/// shoulders. Factored out so the candidate pre-filter and [`base_score`] agree.
fn period_band_value(pf: f64, b: &ScoreBands) -> f64 {
    smoothstep(b.period_rise.0, b.period_rise.1, pf)
        * (1.0 - smoothstep(b.period_fall.0, b.period_fall.1, pf))
}

/// Base interest score for a child: normalized surrounding busyness × a
/// frame-able term (a sane single-step zoom) × a period band. **No centrality
/// term** — off-center (high-`roff`) candidates are the branch diversity the
/// argmax wrongly discarded, so centrality is not a virtue here.
fn base_score(
    busyness: f64,
    period: u32,
    size_mag: f64,
    frame_multiple: f64,
    width: f64,
    bands: &ScoreBands,
) -> f64 {
    if busyness < bands.busyness_reject_below {
        return 0.0; // below the corpus busyness floor
    }
    let next_width = size_mag * frame_multiple;
    let zoom = width / next_width;
    let framable = if !next_width.is_finite() || next_width <= MIN_WIDTH || zoom <= 1.0 {
        0.0
    } else {
        1.0 - smoothstep(12.0, 16.0, zoom.log10())
    };
    let period_band = period_band_value(period as f64, bands);
    (bands.busyness_floor + busyness) * framable * period_band
}

/// Load corpus-derived score bands from `targets.json` (Prompt 8). Per band,
/// the corpus value is applied only when its `provenance` is not `"default"`;
/// anything missing falls back to the built-in constants. Logs what it applied.
fn load_score_bands(path: &str) -> ScoreBands {
    let mut bands = ScoreBands::default();
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("targets: '{path}' absent — using built-in score constants (no corpus bands)");
            return bands;
        }
    };
    let Some(structural) = json_object(&text, "structural") else {
        eprintln!("targets: '{path}' has no \"structural\" block — using built-in constants");
        return bands;
    };

    // busyness: apply band lo as a reject floor (the native busyness threshold).
    if let Some(b) = json_object(structural, "busyness") {
        let prov = json_string(b, "provenance").unwrap_or_default();
        if prov != "default" {
            if let Some(r) = json_field(b, "reject_below") {
                bands.busyness_reject_below = r;
                bands.busyness_from_corpus = true;
            }
            if let Some((lo, hi)) = json_pair(b, "band") {
                bands.busyness_from_corpus = true;
                eprintln!(
                    "targets: busyness band=[{lo:.4},{hi:.4}] reject_below={:.4} (provenance {prov})",
                    bands.busyness_reject_below
                );
            }
        } else {
            eprintln!("targets: busyness provenance=default → built-in busyness constants");
        }
    }

    // period: replace the smoothstep shoulders when the band is label-derived.
    if let Some(p) = json_object(structural, "period") {
        let prov = json_string(p, "provenance").unwrap_or_default();
        if prov != "default" {
            if let Some((lo, hi)) = json_pair(p, "band") {
                bands.period_rise = (lo * 0.75, lo);
                bands.period_fall = (hi, hi * 1.5);
                bands.period_from_corpus = true;
                eprintln!("targets: period band=[{lo:.1},{hi:.1}] (provenance {prov})");
            }
        } else {
            eprintln!("targets: period provenance=default → built-in period band");
        }
    }

    // interior_frac / boundary are present in targets.json for future palette /
    // structural work but the current score has no native channel for them.
    if json_object(structural, "interior_frac").is_some() {
        eprintln!("targets: interior_frac/boundary present but not consumed (no native score channel yet)");
    }

    if !bands.busyness_from_corpus && !bands.period_from_corpus {
        eprintln!("targets: all bands provenance=default/absent → built-in constants unchanged");
    }
    bands
}

// --- minimal read-only JSON helpers (hand-rolled, like the rest of the project) ---

/// The substring of the balanced `{...}` object that follows `"key"` in `text`.
fn json_object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let p = text.find(&needle)? + needle.len();
    let rest = &text[p..];
    let open = rest.find('{')?;
    let bytes = rest.as_bytes();
    let mut depth = 0i32;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[open..=i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Numeric scalar field `"key": <number|null>` within `obj`.
fn json_field(obj: &str, key: &str) -> Option<f64> {
    let needle = format!("\"{key}\"");
    let p = obj.find(&needle)? + needle.len();
    let rest = obj[p..].trim_start_matches([':', ' ', '\t', '\n']);
    if rest.starts_with("null") {
        return None;
    }
    let end = rest
        .find(|c: char| {
            !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')
        })
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Two-element numeric array field `"key": [lo, hi]` within `obj`.
fn json_pair(obj: &str, key: &str) -> Option<(f64, f64)> {
    let needle = format!("\"{key}\"");
    let p = obj.find(&needle)? + needle.len();
    let rest = &obj[p..];
    let lb = rest.find('[')?;
    let rb = rest[lb..].find(']')? + lb;
    let inner = &rest[lb + 1..rb];
    let parts: Vec<f64> = inner
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    if parts.len() >= 2 {
        Some((parts[0], parts[1]))
    } else {
        None
    }
}

/// String field `"key": "value"` within `obj`.
fn json_string(obj: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let p = obj.find(&needle)? + needle.len();
    let rest = obj[p..].trim_start_matches([':', ' ', '\t', '\n']);
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Total in-band reward over the whole frame — the drift-drive's frame objective:
/// "how much resolved decoration is framed." Total (not the largest contiguous
/// region) so it responds monotonically to re-centering — drifting the frame
/// toward the decoration pulls more in-band cells into view, raising the sum,
/// where a single-region metric would jitter as the blob fragments at the edge.
fn total_reward(map: &crate::coherence::CellMap) -> f64 {
    map.reward.iter().sum()
}

/// `de_px_win` at a cell map's center cell (how off-boundary the frame center is).
fn center_de_px(map: &crate::coherence::CellMap, w: usize, h: usize) -> f64 {
    map.de_px[(h / 2) * w + (w / 2)]
}

/// Tiny deterministic jitter from `(seed, id)` to break exact priority ties
/// reproducibly per `--seed` without perturbing genuine score differences.
fn jitter(seed: u64, id: usize) -> f64 {
    let mut r = SplitMix64(seed ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (r.unit() - 0.5) * 2e-9
}

// ===========================================================================
// search subcommand
// ===========================================================================

/// Shared per-run configuration threaded into expansion (immutable).
struct Cfg<'a> {
    panel_w: u32,
    panel_h: u32,
    ss: u32,
    palette: &'a Palette,
    params: &'a ColorParams,
    trap: Trap,
    start_width: f64,
    frame_multiple: f64,
    reselect_k: f64,
    beam_width: usize,
    /// Broad shallow seed count: when `> 0`, the **root** is seeded by a coarse
    /// global atom-domain scan deduped *spatially* (see `atom_candidates_spatial`),
    /// keeping up to this many distinct low-period nuclei across the plane instead
    /// of `beam_width`. `0` → legacy single-root-expand (root keeps `beam_width`).
    seed_count: usize,
    /// Spatial-dedup cell size (panel px) for the broad seed scan.
    seed_cell_px: u32,
    diversity: f64,
    seed: u64,
    maxiter_base: f64,
    per_decade: f64,
    maxiter_ceiling: u32,
    period_cap: u32,
    /// Magnification ceiling: don't expand nodes deeper than this (`0` = unlimited).
    /// The depth analog of `period_cap` for the broad-shallow harvest.
    max_mag: f64,
    bailout: f64,
    backend: crate::cli::BackendChoice,
    panels_dir: std::path::PathBuf,
    bands: ScoreBands,
    /// Wallpaper width the DE-coherence gate is pinned to (resolution-invariant
    /// `de` → predicts the final render's speckle from a cheap thumbnail).
    target_width: u32,
    coherence_theta: f64,
    /// Off-nucleus drift drive (Prompt offnucleus-deband): re-center each frame on
    /// the in-band decoration centroid and rank by band reward, not busyness.
    drift: bool,
    /// Band objective params (the calibrated `coherence` consts).
    band: crate::coherence::BandParams,
}

/// Entry point for the `search` subcommand.
pub fn run_search(args: &SearchArgs) -> Result<(), String> {
    if args.panel_width == 0 {
        return Err("--panel-width must be > 0".into());
    }
    if args.beam_width == 0 {
        return Err("--beam-width must be > 0".into());
    }
    if args.frame_multiple <= 0.0 {
        return Err("--frame-multiple must be > 0".into());
    }
    if args.time_budget <= 0.0 {
        return Err("--time-budget must be > 0".into());
    }

    let panel_w = args.panel_width;
    let panel_h = ((panel_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample.max(1);

    let palette = load_palette(
        &args.palette.palette,
        args.palette.palette_entry.as_deref(),
        args.palette.palette_reverse,
    )?;
    let params = probe::color_params(&args.shade);
    let trap = Trap {
        shape: args.trap,
        center: args.resolved_trap_center()?,
        radius: args.trap_radius,
    };

    // Load corpus-derived score bands (Prompt 8 seam); fall back to constants.
    let bands = load_score_bands(&args.targets);

    let (start_re_s, start_im_s) = args.resolved_start_center()?;
    let init_prec = hp::prec_bits(panel_w, args.start_width) + 96;
    let start_re = hp::parse_decimal(&start_re_s, init_prec)?;
    let start_im = hp::parse_decimal(&start_im_s, init_prec)?;
    let start_f64 = Complex::new(hp::to_f64(&start_re), hp::to_f64(&start_im));

    let strip_path = Path::new(&args.strip);
    let panels_dir = probe::panels_dir_for(strip_path);
    fs::create_dir_all(&panels_dir)
        .map_err(|e| format!("failed to create {}: {e}", panels_dir.display()))?;

    let cfg = Cfg {
        panel_w,
        panel_h,
        ss,
        palette: &palette,
        params: &params,
        trap,
        start_width: args.start_width,
        frame_multiple: args.frame_multiple,
        reselect_k: args.reselect_k,
        beam_width: args.beam_width,
        seed_count: args.seed_count,
        seed_cell_px: args.seed_cell_px,
        diversity: args.diversity,
        seed: args.seed,
        maxiter_base: args.maxiter_base,
        per_decade: args.per_decade,
        maxiter_ceiling: args.maxiter_ceiling,
        period_cap: args.period_cap,
        max_mag: args.max_mag,
        bailout: args.bailout,
        backend: args.backend,
        panels_dir: panels_dir.clone(),
        bands,
        target_width: args.target_width,
        coherence_theta: args.coherence_theta,
        drift: args.drift,
        band: crate::coherence::BandParams::default(),
    };
    if cfg.seed_count > 0 {
        eprintln!(
            "broad shallow seeding ON: root seeded by spatial atom scan (≤{} seeds, cell {}px); \
             period-cap {}, max-mag {}",
            cfg.seed_count, cfg.seed_cell_px, cfg.period_cap,
            if cfg.max_mag > 0.0 { format!("{:.1e}", cfg.max_mag) } else { "unlimited".into() },
        );
    }
    if cfg.drift {
        eprintln!(
            "off-nucleus drift drive ON: band objective (center={:.2} reject=[{:.2},{:.2}] \
             busy_floor={:.4}), DRIFT_MAX={:.2}; candidates ranked by band reward, not busyness",
            cfg.band.band_center, cfg.band.reject_lo, cfg.band.reject_hi, cfg.band.busy_floor,
            crate::coherence::DRIFT_MAX,
        );
    }

    // ---- tree state ----
    let mut nodes: Vec<Node> = Vec::new();
    let mut seen: Vec<Feature> = Vec::new();
    let mut frontier: BinaryHeap<PqItem> = BinaryHeap::new();

    // Root = the start location (no minibrot feature).
    nodes.push(Node {
        id: 0,
        parent: None,
        depth: 0,
        is_root: true,
        center_re: start_re.clone(),
        center_im: start_im.clone(),
        width: args.start_width,
        period: 0,
        nucleus_re: start_re,
        nucleus_im: start_im,
        size_mag: f64::NAN,
        size_arg: f64::NAN,
        busyness: f64::NAN,
        roff: f64::NAN,
        final_z2: f64::NAN,
        c_f64: start_f64,
        score: f64::NAN,
        adjusted: f64::INFINITY,
        feat: Feature::new(1, 0.0, 0.0, 0.0),
        expanded: false,
        magnification: 1.0,
        maxiter: 0,
        backend: "-",
        glitch_count: 0,
        n_candidates: 0,
        collapse_reason: None,
        coherence_subpixel_frac: f64::NAN,
        coherence_de_px_median: f64::NAN,
        busy_win: f64::NAN,
        coherence_penalty: 1.0,
        coherence_reject: false,
        band_score: f64::NAN,
        drifted: false,
        drift_dc: (0.0, 0.0),
        drift_frac: 0.0,
        de_px_center: f64::NAN,
        drift_re: BigFloat::from_f64(0.0, 64),
        drift_im: BigFloat::from_f64(0.0, 64),
        children: Vec::new(),
        panel: None,
        panel_path: String::new(),
    });

    print_table_header();

    // Expand the root first (untimed, per the frontier algorithm), then start
    // the clock and pop best-first until the budget elapses.
    expand_node(0, &mut nodes, &mut frontier, &mut seen, &cfg)?;
    let mut expanded_count = 1usize;
    let mut last_expanded_id = Some(0usize);
    let mut branch_points = if nodes[0].children.len() >= 2 { 1 } else { 0 };
    let mut backtracks = 0usize;
    let mut budget_hit = false;

    let clock = Instant::now();
    loop {
        if clock.elapsed().as_secs_f64() >= args.time_budget {
            budget_hit = true;
            break;
        }
        let Some(item) = frontier.pop() else { break };
        let id = item.id;
        // Backtrack = the pop jumped to a node whose parent is not the node we
        // just expanded (i.e. a different branch than the depth-first spine).
        if let Some(le) = last_expanded_id {
            if nodes[id].parent != Some(le) {
                backtracks += 1;
            }
        }
        expand_node(id, &mut nodes, &mut frontier, &mut seen, &cfg)?;
        expanded_count += 1;
        if nodes[id].children.len() >= 2 {
            branch_points += 1;
        }
        last_expanded_id = Some(id);
    }
    let elapsed = clock.elapsed().as_secs_f64();

    // ---- summary ----
    let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    let deepest_mag = nodes
        .iter()
        .filter(|n| n.expanded)
        .map(|n| n.magnification)
        .fold(0.0f64, f64::max);

    eprintln!(
        "\nsearch done: {} nodes ({} expanded), max depth {}, deepest mag {:.2e}, \
         {} branch points, {} backtracks, {:.1}s / {:.0}s budget{}",
        nodes.len(),
        expanded_count,
        max_depth,
        deepest_mag,
        branch_points,
        backtracks,
        elapsed,
        args.time_budget,
        if budget_hit { " (budget reached)" } else { " (frontier emptied)" },
    );

    // ---- outputs ----
    let best_path = best_path_ids(&nodes, cfg.drift);
    write_best_path_strip(&nodes, &best_path, args, &cfg, strip_path)?;

    let top_ids = top_n_ids(&nodes, args.top_n, cfg.drift);
    write_top_sheet(&nodes, &top_ids, args, &cfg)?;

    // Regression check (Prompt offnucleus-deband): under the band objective the
    // surfaced candidates are off-nucleus decoration frames, the inverse of the
    // old busyness objective that ranked the boundary cusps highest.
    if cfg.drift {
        let drifted = nodes.iter().filter(|n| n.expanded && n.drifted).count();
        let surfaced = top_ids.len();
        let best_band = top_ids
            .iter()
            .map(|&i| nodes[i].band_score)
            .fold(0.0f64, f64::max);
        eprintln!(
            "drift drive: {drifted} frame(s) re-centered off-nucleus; {surfaced} candidate(s) \
             surfaced by band reward (best band_score {best_band:.4}). Decoration framed, not \
             the nucleus cusp — the objective flip is in effect."
        );
    }

    let summary = Summary {
        n_nodes: nodes.len(),
        n_expanded: expanded_count,
        max_depth,
        deepest_mag,
        branch_points,
        backtracks,
        elapsed,
        budget: args.time_budget,
        budget_hit,
    };
    let json = build_json(&nodes, &best_path, &top_ids, args, &summary, strip_path);
    crate::ensure_parent_dir(&args.json)?;
    fs::write(&args.json, json).map_err(|e| format!("failed to write {}: {e}", args.json))?;

    eprintln!(
        "wrote {} (best path, {} levels), {} (top-{}), node panels in {}/, log {}",
        args.strip,
        best_path.len(),
        args.sheet,
        top_ids.len(),
        panels_dir.display(),
        args.json,
    );
    Ok(())
}

/// Expand one node: render its frame, find/filter/score child minibrots, push
/// the top `beam_width`, and cache the shaded panel. Hard safety caps (width
/// floor, maxiter ceiling) skip the render entirely (logged); the period cap
/// renders the panel but descends no further.
fn expand_node(
    id: usize,
    nodes: &mut Vec<Node>,
    frontier: &mut BinaryHeap<PqItem>,
    seen: &mut Vec<Feature>,
    cfg: &Cfg,
) -> Result<(), String> {
    // --- read what we need (clone the hp coords; drop the borrow before mutating) ---
    let center_re = nodes[id].center_re.clone();
    let center_im = nodes[id].center_im.clone();
    let width = nodes[id].width;
    let depth = nodes[id].depth;
    let is_root = nodes[id].is_root;
    let node_period = nodes[id].period;

    let mag = cfg.start_width / width;
    let maxiter = (cfg.maxiter_base + cfg.per_decade * mag.log10())
        .round()
        .max(1.0) as u32;

    // Hard safety caps — skip the (expensive / impossible) render, log, return.
    if width < MIN_WIDTH {
        nodes[id].collapse_reason = Some("width below f64-delta floor (1e-200)".into());
        nodes[id].magnification = mag;
        nodes[id].maxiter = maxiter;
        print_skip_row(id, depth, mag, "width<floor");
        return Ok(());
    }
    // Magnification ceiling (the depth analog of `period_cap`): a chain of
    // low-period descents nests deep even under a low period cap, and the frontier
    // (priority = busyness base-score) dives toward those busy-but-sub-pixel deep
    // boundaries — exactly the budget waste the broad-shallow harvest exists to
    // avoid. Capping mag bounds the harvest to the shallow regime where decoration
    // resolves at the target width, converting deep-descent pops into near-free
    // skips so the budget covers breadth. `0` = unlimited (default search).
    if cfg.max_mag > 0.0 && !is_root && mag > cfg.max_mag {
        nodes[id].collapse_reason =
            Some(format!("magnification {mag:.2e} exceeds max-mag {:.2e}", cfg.max_mag));
        nodes[id].magnification = mag;
        nodes[id].maxiter = maxiter;
        print_skip_row(id, depth, mag, "mag>max");
        return Ok(());
    }
    if maxiter > cfg.maxiter_ceiling {
        nodes[id].collapse_reason =
            Some(format!("maxiter {maxiter} exceeds ceiling {}", cfg.maxiter_ceiling));
        nodes[id].magnification = mag;
        nodes[id].maxiter = maxiter;
        print_skip_row(id, depth, mag, "maxiter>ceil");
        return Ok(());
    }

    // Gather ancestor nuclei (non-root) for the re-selection filter.
    let mut ancestor_nuclei: Vec<(BigFloat, BigFloat)> = Vec::new();
    {
        let mut cur = Some(id);
        while let Some(ci) = cur {
            if !nodes[ci].is_root {
                ancestor_nuclei.push((nodes[ci].nucleus_re.clone(), nodes[ci].nucleus_im.clone()));
            }
            cur = nodes[ci].parent;
        }
    }

    let prec = hp::prec_bits(cfg.panel_w, width) + 32;
    let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

    let panel = probe::render_mandel_panel(
        &center_re, &center_im, center_f64, width, cfg.panel_w, cfg.panel_h, cfg.ss, maxiter,
        cfg.bailout, prec, cfg.trap, cfg.backend,
    );
    let backend_name = panel.backend_name;
    let spacing = panel.spacing;
    let glitch_count = panel.buf.glitched_pixels;

    // DE-coherence gate over this rendered frame, pinned to the *target wallpaper*
    // spacing (the panels are cheap thumbnails, but `de` is resolution-invariant,
    // so the thumbnail predicts the final render's speckle). The root (whole-set
    // entry view) is never gated. A hard reject drops the node from the surfaced
    // candidates and stops descent here (its children are deeper, hence worse
    // speckle); a soft penalty ramps the node's score down toward the threshold.
    let cstats = crate::coherence::coherence_stats(
        &panel.buf, width, cfg.target_width, cfg.coherence_theta,
    );
    let cgate = crate::coherence::coherence_gate(&cstats);
    let busy_win = crate::coherence::windowed_busyness_max(
        &panel.buf, cfg.panel_w, cfg.panel_h, COHERENCE_WINDOW, maxiter,
    );
    let coherence_reject = !is_root && cgate.reject;

    // Shade the panel now; we draw child footprint circles after finding them.
    let mut shaded = crate::render::shade_and_downsample(
        &panel.buf.samples, cfg.panel_w, cfg.panel_h, cfg.ss, cfg.palette, cfg.params, spacing,
    );

    // --- find children (unless a hard gate stops descent here) ---
    let half_w = width * 0.5;
    let half_h = width * (cfg.panel_h as f64 / cfg.panel_w as f64) * 0.5;
    let mut builds: Vec<ChildBuild> = Vec::new();
    let mut n_candidates = 0usize;
    let mut collapse_reason: Option<String> = None;

    if coherence_reject {
        collapse_reason = Some(format!(
            "coherence reject: {} (subpixel_frac={:.3}, de_px_median={:.3})",
            cgate.reason.unwrap_or("speckle"),
            cstats.subpixel_frac,
            cstats.de_px_median,
        ));
    } else if !is_root && node_period > cfg.period_cap {
        collapse_reason = Some(format!("period {node_period} exceeds cap {}", cfg.period_cap));
    } else {
        // Broad shallow seeding: at the root, scan the whole base frame and dedup
        // *spatially* (many distinct same-period nuclei across the plane) and keep
        // up to `seed_count` of them — the source of breadth. Every other node uses
        // the per-period descent scan and keeps `beam_width`.
        let broad_seed = is_root && cfg.seed_count > 0;
        let keep = if broad_seed { cfg.seed_count } else { cfg.beam_width };
        let mut cands = if broad_seed {
            atom_candidates_spatial(
                &panel.buf, cfg.panel_w, cfg.panel_h, width, maxiter, cfg.seed_cell_px,
            )
        } else {
            atom_candidates(&panel.buf, cfg.panel_w, cfg.panel_h, width, maxiter)
        };
        n_candidates = cands.len();
        // Newton-refining every candidate just to score it dominates the cost at
        // depth (thousands of distinct periods). The score is
        // `(0.05+busyness)·framable·period_band`, so first drop candidates whose
        // period band is zero (free — period is known from the atom channel),
        // then Newton only the most promising by busyness. `framable` still needs
        // the size (post-Newton), so evaluate a generous multiple of beam_width.
        cands.retain(|c| {
            let pf = c.period as f64;
            c.period >= 2
                && c.period <= cfg.period_cap
                && period_band_value(pf, &cfg.bands) > 0.0
                && c.busyness >= cfg.bands.busyness_reject_below
        });
        cands.sort_by(|a, b| b.busyness.partial_cmp(&a.busyness).unwrap_or(Ordering::Equal));
        let n_eval = (keep * 4).max(cfg.beam_width * 8).max(24);
        cands.truncate(n_eval);
        for c in &cands {
            let guess_re = center_re.add(&BigFloat::from_f64(c.dc_re, prec), prec, RM);
            let guess_im = center_im.add(&BigFloat::from_f64(c.dc_im, prec), prec, RM);
            let Some(nuc) = newton_nucleus(&guess_re, &guess_im, c.period, width, prec) else {
                continue;
            };
            // In-frame check.
            let nuc_dc_re = hp::to_f64(&nuc.re.sub(&center_re, prec, RM));
            let nuc_dc_im = hp::to_f64(&nuc.im.sub(&center_im, prec, RM));
            if nuc_dc_re.abs() > half_w || nuc_dc_im.abs() > half_h {
                continue;
            }
            let nuc_f64 = Complex::new(hp::to_f64(&nuc.re), hp::to_f64(&nuc.im));
            let size = size_estimate(nuc_f64, c.period);
            if size.overflow || !(size.mag > 0.0) {
                continue;
            }
            let child_width = size.mag * cfg.frame_multiple;
            if !child_width.is_finite() || child_width < MIN_WIDTH {
                continue;
            }
            if c.period > cfg.period_cap {
                continue;
            }
            let score =
                base_score(c.busyness, c.period, size.mag, cfg.frame_multiple, width, &cfg.bands);
            if score <= 1e-6 {
                continue; // not frame-able (zoom-out) or period out of band
            }
            // Re-selection filter: drop the self-targeting central nested copy
            // (nucleus ≈ an ancestor's), preserve distinct off-position children.
            let mut reselect = false;
            for (anc_re, anc_im) in &ancestor_nuclei {
                let dre = hp::to_f64(&nuc.re.sub(anc_re, prec, RM));
                let dim = hp::to_f64(&nuc.im.sub(anc_im, prec, RM));
                if (dre * dre + dim * dim).sqrt() < cfg.reselect_k * child_width {
                    reselect = true;
                    break;
                }
            }
            if reselect {
                continue;
            }

            // Re-refine the nucleus at the precision the child's (deeper) frame
            // needs, so the carried center stays exact as we descend.
            let next_prec = hp::prec_bits(cfg.panel_w, child_width.max(MIN_WIDTH)) + 96;
            let (nre, nim) =
                match newton_nucleus(&nuc.re, &nuc.im, c.period, child_width.max(MIN_WIDTH), next_prec) {
                    Some(n) => (n.re, n.im),
                    None => (nuc.re.clone(), nuc.im.clone()),
                };
            let c_f64 = Complex::new(hp::to_f64(&nre), hp::to_f64(&nim));
            let roff = ((nuc_dc_re / half_w).powi(2) + (nuc_dc_im / half_h).powi(2)).sqrt();
            let feat = Feature::new(c.period, c.busyness, nuc_dc_re / half_w, nuc_dc_im / half_h);

            // Footprint circle (next frame's extent inside this panel).
            let zoom = width / child_width;
            let cx = (nuc_dc_re / width + 0.5) * cfg.panel_w as f64;
            let cy = (0.5 - nuc_dc_im / (width * (cfg.panel_h as f64 / cfg.panel_w as f64)))
                * cfg.panel_h as f64;
            let cr = (cfg.panel_w as f64 / (2.0 * zoom)).max(2.0);

            builds.push(ChildBuild {
                period: c.period,
                nucleus_re: nre,
                nucleus_im: nim,
                size_mag: size.mag,
                size_arg: size.arg,
                width: child_width,
                busyness: c.busyness,
                roff,
                score,
                feat,
                final_z2: nuc.final_z2,
                c_f64,
                circle: (cx, cy, cr),
            });
        }
        // Keep the top `keep` by base score (seed_count at the root in broad-seed
        // mode, beam_width otherwise).
        builds.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        builds.truncate(keep);
        if builds.is_empty() && collapse_reason.is_none() {
            collapse_reason =
                Some("no valid child (atom/Newton/size/re-selection all rejected)".into());
        }
    }

    // --- off-nucleus drift drive (Prompt offnucleus-deband, Phase 4) ---
    // Anchors (nuclei) are boundary points; the busyness gradient runs uphill to
    // them. Instead of surfacing the nucleus cusp, re-center on the surrounding
    // *decoration*: find the best contiguous in-band `de_px`-band region, drift
    // the frame center toward its centroid (clamped), re-render, and keep the
    // drift iff its band reward improved AND it clears the coherence gate.
    let mut band_score = f64::NAN;
    let mut drifted = false;
    let mut drift_dc = (0.0f64, 0.0f64);
    let mut drift_frac = 0.0f64;
    let mut de_px_center = f64::NAN;
    let mut drift_re = center_re.clone();
    let mut drift_im = center_im.clone();
    let cells = (cfg.panel_w as f64) * (cfg.panel_h as f64);
    if cfg.drift && !is_root && !coherence_reject {
        let map0 = crate::coherence::cell_reward_map(
            &panel.buf, cfg.panel_w, cfg.panel_h, COHERENCE_WINDOW, maxiter, width,
            cfg.target_width, &cfg.band,
        );
        let reward0 = total_reward(&map0);
        // Phase-3 fix: drift toward the **densest in-band reward lobe**, not the
        // centroid. Decoration is a ring of filigree around the cusp; the centroid
        // of a ring is its hole (back on the boundary, de_px → 0), which is why the
        // centroid-drift landed at de_px ~0.06. The peak sits on an actual lobe.
        let target0 = crate::coherence::best_inband_peak(&map0).map(|(cx, cy, _)| (cx, cy));
        band_score = reward0 / cells;
        de_px_center = center_de_px(&map0, cfg.panel_w as usize, cfg.panel_h as usize);

        if let Some((cx, cy)) = target0 {
            // Peak cell coords → plane offset from the frame center.
            let frame_h = width * (cfg.panel_h as f64 / cfg.panel_w as f64);
            let mut dre = (cx / cfg.panel_w as f64 - 0.5) * width;
            let mut dim = (0.5 - cy / cfg.panel_h as f64) * frame_h;
            // Clamp so the drift stays within DRIFT_MAX of the half-extent on each
            // axis (keeps the drifted frame inside the rendered region).
            let half_w = width * 0.5;
            let half_h = frame_h * 0.5;
            let m = (dre / half_w).abs().max((dim / half_h).abs());
            if m > crate::coherence::DRIFT_MAX {
                let s = crate::coherence::DRIFT_MAX / m;
                dre *= s;
                dim *= s;
            }
            // Drifted center in hp; re-render at the same width.
            let dcenter_re = center_re.add(&BigFloat::from_f64(dre, prec), prec, RM);
            let dcenter_im = center_im.add(&BigFloat::from_f64(dim, prec), prec, RM);
            let dcenter_f64 = Complex::new(hp::to_f64(&dcenter_re), hp::to_f64(&dcenter_im));
            let dpanel = probe::render_mandel_panel(
                &dcenter_re, &dcenter_im, dcenter_f64, width, cfg.panel_w, cfg.panel_h, cfg.ss,
                maxiter, cfg.bailout, prec, cfg.trap, cfg.backend,
            );
            let dgate = crate::coherence::coherence_gate(&crate::coherence::coherence_stats(
                &dpanel.buf, width, cfg.target_width, cfg.coherence_theta,
            ));
            let map1 = crate::coherence::cell_reward_map(
                &dpanel.buf, cfg.panel_w, cfg.panel_h, COHERENCE_WINDOW, maxiter, width,
                cfg.target_width, &cfg.band,
            );
            let reward1 = total_reward(&map1);
            // Keep the drift only if it framed more decoration and the drifted
            // frame itself is coherent (not speckle).
            if reward1 > reward0 && !dgate.reject {
                shaded = crate::render::shade_and_downsample(
                    &dpanel.buf.samples, cfg.panel_w, cfg.panel_h, cfg.ss, cfg.palette, cfg.params,
                    dpanel.spacing,
                );
                drifted = true;
                band_score = reward1 / cells;
                drift_dc = (dre, dim);
                drift_frac = (dre / half_w).hypot(dim / half_h);
                de_px_center = center_de_px(&map1, cfg.panel_w as usize, cfg.panel_h as usize);
                drift_re = dcenter_re;
                drift_im = dcenter_im;
            }
        }
    }

    // Draw the kept children's footprint circles on the surfaced panel — but only
    // when it is the nucleus-centered frame (a drifted frame's geometry no longer
    // matches the nucleus-relative footprints).
    if !drifted {
        for b in &builds {
            probe::draw_circle(&mut shaded, b.circle.0, b.circle.1, b.circle.2);
        }
    }

    // --- mutate: create child nodes, push to frontier, finalize this node ---
    let mut child_ids = Vec::with_capacity(builds.len());
    for b in builds {
        let cid = nodes.len();
        // Diversity penalty vs everything seen so far (incl. earlier siblings of
        // this batch), so a batch can't be all near-duplicates of one another.
        let sim = similarity(&b.feat, seen);
        let adjusted = b.score - cfg.diversity * sim + jitter(cfg.seed, cid);
        seen.push(b.feat);

        nodes.push(Node {
            id: cid,
            parent: Some(id),
            depth: depth + 1,
            is_root: false,
            center_re: b.nucleus_re.clone(),
            center_im: b.nucleus_im.clone(),
            width: b.width,
            period: b.period,
            nucleus_re: b.nucleus_re,
            nucleus_im: b.nucleus_im,
            size_mag: b.size_mag,
            size_arg: b.size_arg,
            busyness: b.busyness,
            roff: b.roff,
            final_z2: b.final_z2,
            c_f64: b.c_f64,
            score: b.score,
            adjusted,
            feat: b.feat,
            expanded: false,
            magnification: cfg.start_width / b.width,
            maxiter: 0,
            backend: "-",
            glitch_count: 0,
            n_candidates: 0,
            collapse_reason: None,
            coherence_subpixel_frac: f64::NAN,
            coherence_de_px_median: f64::NAN,
            busy_win: f64::NAN,
            coherence_penalty: 1.0,
            coherence_reject: false,
            band_score: f64::NAN,
            drifted: false,
            drift_dc: (0.0, 0.0),
            drift_frac: 0.0,
            de_px_center: f64::NAN,
            drift_re: BigFloat::from_f64(0.0, 64),
            drift_im: BigFloat::from_f64(0.0, 64),
            children: Vec::new(),
            panel: None,
            panel_path: String::new(),
        });
        frontier.push(PqItem { key: adjusted, id: cid });
        child_ids.push(cid);
    }

    // Save a labeled per-node PNG for the JSON; keep the clean panel in memory.
    let label = node_label(id, node_period, mag, nodes[id].score, is_root);
    let mut labeled = shaded.clone();
    font::draw_text(&mut labeled, &label, 2, 2, 2, Rgb([240, 240, 240]), true);
    let panel_file = cfg.panels_dir.join(format!("node_{id:04}.png"));
    labeled
        .save(&panel_file)
        .map_err(|e| format!("failed to write {}: {e}", panel_file.display()))?;

    let n = &mut nodes[id];
    n.expanded = true;
    n.magnification = mag;
    n.maxiter = maxiter;
    n.backend = backend_name;
    n.glitch_count = glitch_count;
    n.n_candidates = n_candidates;
    n.collapse_reason = collapse_reason;
    n.coherence_subpixel_frac = cstats.subpixel_frac;
    n.coherence_de_px_median = cstats.de_px_median;
    n.busy_win = busy_win;
    n.coherence_penalty = cgate.penalty;
    n.coherence_reject = coherence_reject;
    // Fold the soft penalty into the node's own surfacing score (best-path /
    // top-N rank on `score`). Non-root only: the root score is NaN by design.
    // A rejected node keeps `score` for the record but is excluded by the flag.
    if !n.is_root && n.score.is_finite() {
        n.score *= cgate.penalty;
    }
    n.band_score = band_score;
    n.drifted = drifted;
    n.drift_dc = drift_dc;
    n.drift_frac = drift_frac;
    n.de_px_center = de_px_center;
    n.drift_re = drift_re;
    n.drift_im = drift_im;
    n.children = child_ids;
    n.panel = Some(shaded);
    n.panel_path = probe::path_str(&panel_file);

    print_expand_row(&nodes[id], n_candidates, cfg.drift);
    Ok(())
}

/// Per-node tile/strip label.
fn node_label(id: usize, period: u32, mag: f64, score: f64, is_root: bool) -> String {
    if is_root {
        format!("N{id} ROOT M={mag:.1e}").to_uppercase()
    } else {
        format!("N{id} P={period} M={mag:.1e} S={score:.2}").to_uppercase()
    }
}

// ===========================================================================
// Best-path strip
// ===========================================================================

/// The surfacing metric a node is ranked by: the off-nucleus **band reward** in
/// drift mode, the busyness base-score otherwise.
fn rank_metric(n: &Node, drift: bool) -> f64 {
    if drift {
        n.band_score
    } else {
        n.score
    }
}

/// Whether a node may be surfaced (best-path / top-N): expanded, non-root, not
/// coherence-rejected, has a cached panel, and carries a usable rank metric (in
/// drift mode the band reward must be strictly positive — a frame with no in-band
/// decoration is not a candidate).
fn surfaceable(n: &Node, drift: bool) -> bool {
    if !(n.expanded && !n.is_root && !n.coherence_reject && n.panel.is_some()) {
        return false;
    }
    let m = rank_metric(n, drift);
    m.is_finite() && (!drift || m > 0.0)
}

/// Trace the highest-scoring expanded leaf to the root via parent pointers. The
/// "best leaf" is the highest-metric node that has a cached panel; its ancestors
/// are all expanded (they produced it), so every strip row has a Mandelbrot panel.
fn best_path_ids(nodes: &[Node], drift: bool) -> Vec<usize> {
    let best = nodes
        .iter()
        .filter(|n| surfaceable(n, drift))
        .max_by(|a, b| rank_metric(a, drift).partial_cmp(&rank_metric(b, drift)).unwrap_or(Ordering::Equal))
        .map(|n| n.id)
        // Fallback: root only (e.g. nothing past the start expanded).
        .unwrap_or(0);
    let mut path = Vec::new();
    let mut cur = Some(best);
    while let Some(ci) = cur {
        path.push(ci);
        cur = nodes[ci].parent;
    }
    path.reverse();
    path
}

/// Compose the best-path filmstrip (Mandelbrot panel + base-scale Julia per
/// level), reusing the `probe` strip machinery. Julia panels render on demand
/// for just this short path (Mandelbrot frames are never re-iterated).
fn write_best_path_strip(
    nodes: &[Node],
    path: &[usize],
    args: &SearchArgs,
    cfg: &Cfg,
    strip_path: &Path,
) -> Result<(), String> {
    if path.is_empty() {
        return Err("search produced no nodes to strip".into());
    }
    let mut mandel: Vec<RgbImage> = Vec::with_capacity(path.len());
    // Julia is opt-in (`--with-julia`): the parallel render only costs time, since
    // a good Mandelbrot region implies a good Julia.
    let mut julia: Vec<RgbImage> = Vec::with_capacity(if args.with_julia { path.len() } else { 0 });
    for &id in path {
        let n = &nodes[id];
        let mut m = match &n.panel {
            Some(p) => p.clone(),
            None => RgbImage::from_pixel(cfg.panel_w, cfg.panel_h, Rgb(probe::STRIP_BG)),
        };
        let label = node_label(id, n.period, n.magnification, n.score, n.is_root);
        font::draw_text(&mut m, &label, 2, 2, 2, Rgb([240, 240, 240]), true);
        mandel.push(m);
        if args.with_julia {
            julia.push(probe::render_julia_panel(
                n.c_f64, args.julia_maxiter, args.bailout, cfg.trap, cfg.panel_w, cfg.panel_h,
                cfg.ss, cfg.palette, cfg.params,
            ));
        }
    }
    let strip = if args.with_julia {
        probe::compose_strip(&mandel, &julia, cfg.panel_w, cfg.panel_h)
    } else {
        probe::compose_strip_single(&mandel, cfg.panel_w, cfg.panel_h)
    };
    crate::ensure_parent_dir(strip_path)?;
    strip
        .save(strip_path)
        .map_err(|e| format!("failed to write {}: {e}", strip_path.display()))
}

// ===========================================================================
// Top-N diversity contact sheet (farthest-point sampling)
// ===========================================================================

/// Surfaceable pool with **same-nucleus duplicates collapsed** (highest rank
/// metric kept). The same minibrot is often reached by two paths (and re-found
/// drifted/undrifted), producing redundant tiles; the broad spatial seed makes
/// this common. Conjugate nuclei (`im` sign flip) are genuinely distinct frames
/// and are *not* collapsed — only nuclei equal to f64 precision. Greedy: sort by
/// rank desc, keep a node unless its `c_f64` matches an already-kept node's.
fn dedup_by_nucleus(nodes: &[Node], drift: bool) -> Vec<usize> {
    let mut ids: Vec<usize> = nodes
        .iter()
        .filter(|nd| surfaceable(nd, drift))
        .map(|nd| nd.id)
        .collect();
    ids.sort_by(|&a, &b| {
        rank_metric(&nodes[b], drift)
            .partial_cmp(&rank_metric(&nodes[a], drift))
            .unwrap_or(Ordering::Equal)
    });
    let mut kept: Vec<usize> = Vec::new();
    for id in ids {
        let c = nodes[id].c_f64;
        let dup = kept.iter().any(|&k| {
            let o = nodes[k].c_f64;
            let scale = 1.0 + c.re.abs().max(c.im.abs());
            (c.re - o.re).hypot(c.im - o.im) < 1e-9 * scale
        });
        if !dup {
            kept.push(id);
        }
    }
    kept
}

/// Pick up to `n` diverse high-scoring nodes by farthest-point sampling over the
/// feature vectors, seeded with the top scorer. Pool = surfaceable non-root nodes
/// (with same-nucleus duplicates collapsed) that have a cached panel.
fn top_n_ids(nodes: &[Node], n: usize, drift: bool) -> Vec<usize> {
    let pool = dedup_by_nucleus(nodes, drift);
    if pool.is_empty() || n == 0 {
        return Vec::new();
    }
    let seed = *pool
        .iter()
        .max_by(|a, b| {
            rank_metric(&nodes[**a], drift)
                .partial_cmp(&rank_metric(&nodes[**b], drift))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap();
    let mut chosen = vec![seed];
    while chosen.len() < n && chosen.len() < pool.len() {
        let mut best: Option<usize> = None;
        let mut best_d = -1.0f64;
        for &p in &pool {
            if chosen.contains(&p) {
                continue;
            }
            let mind = chosen
                .iter()
                .map(|&c| nodes[p].feat.dist2(&nodes[c].feat))
                .fold(f64::INFINITY, f64::min);
            if mind > best_d {
                best_d = mind;
                best = Some(p);
            }
        }
        match best {
            Some(b) => chosen.push(b),
            None => break,
        }
    }
    chosen
}

/// Compose + write the top-N contact sheet from cached panels (labeled clones)
/// via the shared [`sheet::compose_grid`] machinery.
fn write_top_sheet(
    nodes: &[Node],
    top_ids: &[usize],
    args: &SearchArgs,
    cfg: &Cfg,
) -> Result<(), String> {
    if top_ids.is_empty() {
        eprintln!("warning: no nodes to compose a top-N sheet (search expanded nothing past root)");
        return Ok(());
    }
    let tiles: Vec<RgbImage> = top_ids
        .iter()
        .map(|&id| {
            let n = &nodes[id];
            let mut t = n.panel.clone().unwrap();
            // First line: identity + the primary rank metric (band reward in drift
            // mode, busyness base-score otherwise).
            let label = if cfg.drift {
                format!("N{id} P={} M={:.1e} BAND={:.4}{}",
                    n.period, n.magnification, n.band_score, if n.drifted { " D" } else { "" })
            } else {
                format!("N{id} P={} M={:.1e} S={:.2}", n.period, n.magnification, n.score)
            }
            .to_uppercase();
            font::draw_text(&mut t, &label, 2, 2, 2, Rgb([240, 240, 240]), true);
            // Second line: the DE-coherence stats + (drift mode) the drift fraction
            // and de_px at the new center, so Matt sees how far off-nucleus each
            // surfaced candidate sits and how close to the speckle floor.
            let cline = if cfg.drift {
                format!(
                    "SPX={:.3} DEM={:.2} DR={:.2} DPC={:.2}",
                    n.coherence_subpixel_frac, n.coherence_de_px_median, n.drift_frac, n.de_px_center
                )
            } else {
                format!(
                    "SPX={:.3} DEM={:.2} BW={:.3}",
                    n.coherence_subpixel_frac, n.coherence_de_px_median, n.busy_win
                )
            }
            .to_uppercase();
            font::draw_text(&mut t, &cline, 2, 22, 2, Rgb([180, 220, 180]), true);
            t
        })
        .collect();
    let grid = sheet::compose_grid(&tiles, None);
    crate::ensure_parent_dir(&args.sheet)?;
    grid.save(&args.sheet)
        .map_err(|e| format!("failed to write {}: {e}", args.sheet))?;
    Ok(())
}

// ===========================================================================
// stdout table
// ===========================================================================

fn print_table_header() {
    println!(
        "{:>4}  {:>4}  {:>3}  {:>9}  {:>6}  {:>4}  {:>5}  {:>7}  {:>6}  {:>6}  {:>5}  {:>6}  {:>5}  {:>5}",
        "id", "par", "dep", "mag", "maxit", "bknd", "cand", "period", "roff", "score", "adj",
        "child", "spx", "gate",
    );
}

fn print_expand_row(n: &Node, n_candidates: usize, drift: bool) {
    let par = n.parent.map(|p| p as i64).unwrap_or(-1);
    let gate = if n.coherence_reject {
        "REJ"
    } else if n.coherence_penalty < 0.999 {
        "pen"
    } else {
        "ok"
    };
    print!(
        "{:>4}  {:>4}  {:>3}  {:>9.2e}  {:>6}  {:>4}  {:>5}  {:>7}  {:>6.3}  {:>6.3}  {:>5.3}  {:>6}  {:>5.3}  {:>5}",
        n.id,
        par,
        n.depth,
        n.magnification,
        n.maxiter,
        n.backend,
        n_candidates,
        n.period,
        n.roff,
        n.score,
        n.adjusted,
        n.children.len(),
        n.coherence_subpixel_frac,
        gate,
    );
    // Drift-drive columns: band reward, drift fraction, de_px at the new center,
    // and a `*` when the frame was re-centered off the nucleus.
    if drift {
        print!(
            "  band={:.4} dr={:.2} dpc={:.2}{}",
            n.band_score,
            n.drift_frac,
            n.de_px_center,
            if n.drifted { " *" } else { "" },
        );
    }
    println!();
}

fn print_skip_row(id: usize, depth: u32, mag: f64, why: &str) {
    println!(
        "{:>4}  {:>4}  {:>3}  {:>9.2e}  skip ({})",
        id, "-", depth, mag, why
    );
}

// ===========================================================================
// JSON
// ===========================================================================

struct Summary {
    n_nodes: usize,
    n_expanded: usize,
    max_depth: u32,
    deepest_mag: f64,
    branch_points: usize,
    backtracks: usize,
    elapsed: f64,
    budget: f64,
    budget_hit: bool,
}

fn build_json(
    nodes: &[Node],
    best_path: &[usize],
    top_ids: &[usize],
    args: &SearchArgs,
    sum: &Summary,
    strip_path: &Path,
) -> String {
    use probe::{jf, js};
    let mut s = String::from("{\n");

    // params
    s.push_str("  \"params\": {\n");
    s.push_str(&format!("    \"start_center\": {},\n", js(&args.start_center)));
    s.push_str(&format!("    \"start_width\": {},\n", jf(args.start_width)));
    s.push_str(&format!("    \"time_budget\": {},\n", jf(args.time_budget)));
    s.push_str(&format!("    \"beam_width\": {},\n", args.beam_width));
    s.push_str(&format!("    \"seed_count\": {},\n", args.seed_count));
    s.push_str(&format!("    \"seed_cell_px\": {},\n", args.seed_cell_px));
    s.push_str(&format!("    \"diversity\": {},\n", jf(args.diversity)));
    s.push_str(&format!("    \"frame_multiple\": {},\n", jf(args.frame_multiple)));
    s.push_str(&format!("    \"reselect_k\": {},\n", jf(args.reselect_k)));
    s.push_str(&format!("    \"period_cap\": {},\n", args.period_cap));
    s.push_str(&format!("    \"max_mag\": {},\n", jf(args.max_mag)));
    s.push_str(&format!("    \"panel_width\": {},\n", args.panel_width));
    s.push_str(&format!("    \"coherence_target_width\": {},\n", args.target_width));
    s.push_str(&format!("    \"coherence_theta\": {},\n", jf(args.coherence_theta)));
    s.push_str(&format!("    \"coherence_reject\": {},\n", jf(crate::coherence::COHERENCE_REJECT)));
    s.push_str(&format!("    \"de_px_median_floor\": {},\n", jf(crate::coherence::DE_PX_MEDIAN_FLOOR)));
    s.push_str(&format!("    \"drift\": {},\n", args.drift));
    if args.drift {
        let bp = crate::coherence::BandParams::default();
        s.push_str(&format!("    \"band_center\": {},\n", jf(bp.band_center)));
        s.push_str(&format!("    \"band_reject_lo\": {},\n", jf(bp.reject_lo)));
        s.push_str(&format!("    \"band_reject_hi\": {},\n", jf(bp.reject_hi)));
        s.push_str(&format!("    \"band_busy_floor\": {},\n", jf(bp.busy_floor)));
        s.push_str(&format!("    \"drift_max\": {},\n", jf(crate::coherence::DRIFT_MAX)));
    }
    s.push_str(&format!("    \"seed\": {}\n", args.seed));
    s.push_str("  },\n");

    // summary
    s.push_str("  \"summary\": {\n");
    s.push_str(&format!("    \"n_nodes\": {},\n", sum.n_nodes));
    s.push_str(&format!("    \"n_expanded\": {},\n", sum.n_expanded));
    s.push_str(&format!("    \"max_depth\": {},\n", sum.max_depth));
    s.push_str(&format!("    \"deepest_magnification\": {},\n", jf(sum.deepest_mag)));
    s.push_str(&format!("    \"branch_points\": {},\n", sum.branch_points));
    s.push_str(&format!("    \"backtracks\": {},\n", sum.backtracks));
    s.push_str(&format!("    \"elapsed_secs\": {},\n", jf(sum.elapsed)));
    s.push_str(&format!("    \"budget_secs\": {},\n", jf(sum.budget)));
    s.push_str(&format!("    \"budget_hit\": {}\n", sum.budget_hit));
    s.push_str("  },\n");

    s.push_str(&format!(
        "  \"best_path\": [{}],\n",
        best_path.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
    ));
    s.push_str(&format!(
        "  \"top_n\": [{}],\n",
        top_ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
    ));
    s.push_str(&format!("  \"strip\": {},\n", js(&probe::path_str(strip_path))));
    s.push_str(&format!("  \"sheet\": {},\n", js(&args.sheet)));

    // nodes
    s.push_str("  \"nodes\": [\n");
    for (i, n) in nodes.iter().enumerate() {
        s.push_str("    {\n");
        s.push_str(&format!("      \"id\": {},\n", n.id));
        match n.parent {
            Some(p) => s.push_str(&format!("      \"parent\": {p},\n")),
            None => s.push_str("      \"parent\": null,\n"),
        }
        s.push_str(&format!("      \"depth\": {},\n", n.depth));
        s.push_str(&format!("      \"is_root\": {},\n", n.is_root));
        s.push_str(&format!("      \"backend\": {},\n", js(n.backend)));
        // center / nucleus hp strings.
        s.push_str(&format!(
            "      \"center\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&dec(&n.center_re)),
            js(&dec(&n.center_im))
        ));
        s.push_str(&format!(
            "      \"nucleus\": {{ \"re\": {}, \"im\": {} }},\n",
            js(&dec(&n.nucleus_re)),
            js(&dec(&n.nucleus_im))
        ));
        s.push_str(&format!("      \"width\": {},\n", jf(n.width)));
        s.push_str(&format!("      \"magnification\": {},\n", jf(n.magnification)));
        s.push_str(&format!("      \"maxiter\": {},\n", n.maxiter));
        s.push_str(&format!("      \"period\": {},\n", n.period));
        s.push_str(&format!(
            "      \"size_estimate\": {{ \"mag\": {}, \"arg\": {} }},\n",
            jf(n.size_mag),
            jf(n.size_arg)
        ));
        s.push_str(&format!("      \"busyness\": {},\n", jf(n.busyness)));
        s.push_str(&format!("      \"roff\": {},\n", jf(n.roff)));
        s.push_str(&format!("      \"score\": {},\n", jf(n.score)));
        s.push_str(&format!("      \"adjusted\": {},\n", jf(n.adjusted)));
        s.push_str("      \"coherence\": {\n");
        s.push_str(&format!("        \"subpixel_frac\": {},\n", jf(n.coherence_subpixel_frac)));
        s.push_str(&format!("        \"de_px_median\": {},\n", jf(n.coherence_de_px_median)));
        s.push_str(&format!("        \"busy_win\": {},\n", jf(n.busy_win)));
        s.push_str(&format!("        \"penalty\": {},\n", jf(n.coherence_penalty)));
        s.push_str(&format!("        \"reject\": {}\n", n.coherence_reject));
        s.push_str("      },\n");
        // Off-nucleus drift drive: the band objective + the applied drift. The
        // drifted center (hp) is the frame actually surfaced when `drifted`.
        s.push_str("      \"band\": {\n");
        s.push_str(&format!("        \"band_score\": {},\n", jf(n.band_score)));
        s.push_str(&format!("        \"drifted\": {},\n", n.drifted));
        s.push_str(&format!(
            "        \"drift_dc\": {{ \"re\": {}, \"im\": {} }},\n",
            jf(n.drift_dc.0),
            jf(n.drift_dc.1)
        ));
        s.push_str(&format!("        \"drift_frac\": {},\n", jf(n.drift_frac)));
        s.push_str(&format!("        \"de_px_center\": {},\n", jf(n.de_px_center)));
        if n.drifted {
            s.push_str(&format!(
                "        \"drifted_center\": {{ \"re\": {}, \"im\": {} }}\n",
                js(&dec(&n.drift_re)),
                js(&dec(&n.drift_im))
            ));
        } else {
            s.push_str("        \"drifted_center\": null\n");
        }
        s.push_str("      },\n");
        s.push_str(&format!("      \"final_z2\": {},\n", jf(n.final_z2)));
        s.push_str(&format!(
            "      \"c_f64\": {{ \"re\": {}, \"im\": {} }},\n",
            jf(n.c_f64.re),
            jf(n.c_f64.im)
        ));
        s.push_str(&format!("      \"glitch_count\": {},\n", n.glitch_count));
        s.push_str(&format!("      \"n_candidates\": {},\n", n.n_candidates));
        match &n.collapse_reason {
            Some(r) => s.push_str(&format!("      \"collapse_reason\": {},\n", js(r))),
            None => s.push_str("      \"collapse_reason\": null,\n"),
        }
        s.push_str(&format!(
            "      \"children\": [{}],\n",
            n.children.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", ")
        ));
        s.push_str(&format!("      \"panel_path\": {}\n", js(&n.panel_path)));
        s.push_str("    }");
        if i + 1 < nodes.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}

/// Decimal string for an hp coordinate, falling back to empty on the (shouldn't
/// happen) format error so a single bad node can't abort the whole log.
fn dec(x: &BigFloat) -> String {
    hp::to_decimal_string(x).unwrap_or_default()
}
