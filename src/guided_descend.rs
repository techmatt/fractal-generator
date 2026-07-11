//! `guided-descend` — stochastic guided descent → candidate-pool sheet.
//!
//! Replaces flat shallow loose-seed generation (`generate`, left intact as the
//! control) with **guided stochastic descent**: from the single base-Mandelbrot
//! root, run many **fully independent, decorrelated** walks, each to a **random
//! depth**, where each step picks the next center by a probabilistic policy
//! (mostly into a detected μ-focus). Every frame visited along a walk is a
//! candidate. **Geometric policies only — no CNN, no dedup, no prefix-sharing,
//! no memoization.** Hand-set weights, output a candidate-pool sheet judged by
//! eye. **No quality claims.**
//!
//! Reuse: the cheap screen ([`probe::render_mandel_panel`] +
//! [`generate::screen_stats`] + [`generate::AcceptBand`]), the energy-weighted
//! content focus ([`energy::tile_energy`], same primitive `present`'s
//! `content_focus` uses), and `present`'s focus→frame composition math
//! (child center = focus placed inside `parent.fw × zoom_per_step`).
//!
//! ## The focus finder (built here — `focus_diag` is dump-only)
//!
//! `focus_diag` writes the raw `mu`/interior field arrays to disk; the
//! scale-space maxima/persistence/isolation analysis lived in an uncommitted
//! scipy notebook. So the 0.85 foci branch needs a **live Rust** finder, built
//! in [`find_foci`]: render the cheap μ-field, Gaussian-smooth (3-box approx,
//! normalized convolution over the exterior) at σ ∈ {16,20,24,28,32} px, dilate
//! the interior mask by ~σ to exclude minibrot halos, take local maxima, and
//! score each by **peak-response × isolation (peak ÷ local field mean) —
//! explicitly NOT raw persistence** (persistence rewards mass → biases toward
//! thickets, per the `focus-field-concentration` finding; we keep it only as a
//! reported attribute). Foci are sampled in proportion to that score.
//!
//! Root step is a **50/50 sampler mixture** (rev4 Part A, `--root-mix`): with
//! probability `root_mix` the depth-1 window is drawn from the durable 8192² smooth
//! field ([`root_field`]) — uniformly among windows passing a two-sided escaped-μ
//! criterion (the `score` seam) at `--root-zoom-8k` — and otherwise from the flat
//! `generate` sampler verbatim (uniform-in-plane center × log-uniform shallow fw →
//! cheap screen → `AcceptBand`), keeping its own sampled fw. **Both roots are
//! permissive (≤80% black) and skip the occupancy floor** — the depth ≥ 2 set-
//! avoidance guards do the tightening. This grafts the flat method's planar
//! measure-uniformity onto descent at the root; rev4 Part B grafts it per-step
//! (random near-boundary draws, FPS-spread foci, zoom jitter). Decorrelation comes
//! from the RNG stream; the 8k field is shared across all runs.

use std::fmt::Write as _;
use std::path::Path;

use astro_float::BigFloat;
use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{JuliaBackend, PhoenixBackend, Trap, TrapShape};
use clap::Args;
use crate::cli::BackendChoice;
use crate::energy::{self, OCC_FLOOR, OCC_GX, OCC_GY};
use crate::generate::{self, color_params};
use crate::palette::Palette;
use crate::probe::{self, SplitMix64};
use crate::render::{self, Frame};
use crate::root_field::{PassWindow, RootField};
use crate::{hp, sheet};

/// Conservative safe-f64 zoom floor on the depth-1 **root start** frame width. At
/// the finest sampling any location ever sees (wallpaper 2560×1440 ss4, ~1e4
/// samples/axis), `fw = 1e-9` still leaves ~100–1000 ULPs per sample even near the
/// worst-case `|c|≈2` — comfortably clear of the f64 precision wall (~5e-12) with no
/// guard required. No real deep-zoom handling (perturbation is the search's job).
/// The **descent** floor (depth ≥ 2 step truncation) is the tunable `--min-fw` arg,
/// not this constant.
const FW_FLOOR: f64 = 1e-9;

/// Per-scale local-maxima percentile floor (over the exterior smoothed field):
/// a candidate maximum must sit above this quantile of its scale to count.
const MAX_FLOOR_PCT: f64 = 0.85;

/// Cap on foci returned per frame (top by sampling score).
const TOP_FOCI: usize = 16;

/// Stage-1 cheap interior-screen render width (height 16:9, ss1). Small + fast:
/// interior fraction is scale-robust, so a ~128px escape-time panel is enough to
/// reject set-dominated candidates before paying for the 768 node render.
const PROBE_W: u32 = 128;

/// Near-boundary band cut (rev4 B2 + flat-root screens are elsewhere): exterior
/// pixels whose DE sits in the bottom this quantile of exterior DE (small DE ⇒
/// close to the set, not interior). Traces a thin boundary band the per-step
/// `random` branch samples from. (The rev1–3 5-feature exclusion list is retired:
/// the ≤80%-black root gate subsumes it.)
const BOUNDARY_DE_PCT: f64 = 0.12;

/// Window-scan stride as a fraction of window height (8k root). Half-window stride
/// gives broad spatial coverage of passing windows without an integral image.
const SCAN_STRIDE_FRAC: f64 = 0.5;

/// Max redraw attempts for either root sampler before a walk's root step dies.
const ROOT_MAX_TRIES: usize = 4000;

/// Parse an injected-seed list (`--seed-list`): one JSON object per line carrying at
/// least `"cx"`, `"cy"`, `"fw"` (extra keys, e.g. an `exploit`/`explore` tag, are
/// ignored — the atlas proposer keeps provenance in its own emitted file). Hand-rolled
/// to match the repo's serde-free JSON convention. Returns the depth-1 (cx,cy,fw) per
/// walk, in file order (walk `w` ← row `w`).
fn load_seed_list(path: &str) -> Result<Vec<(f64, f64, f64)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let get = |key: &str| -> Result<f64, String> {
            let pat = format!("\"{key}\"");
            let i = line
                .find(&pat)
                .ok_or_else(|| format!("seed-list line {}: missing key {key}", ln + 1))?;
            let rest = &line[i + pat.len()..];
            let colon = rest.find(':').ok_or_else(|| format!("seed-list line {}: no ':' after {key}", ln + 1))?;
            // number token = chars after ':' up to the next comma/brace/space
            let tok: String = rest[colon + 1..]
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
                .collect();
            tok.parse::<f64>()
                .map_err(|_| format!("seed-list line {}: bad {key} value '{tok}'", ln + 1))
        };
        let (cx, cy, fw) = (get("cx")?, get("cy")?, get("fw")?);
        if !(fw > 0.0) || !cx.is_finite() || !cy.is_finite() {
            return Err(format!("seed-list line {}: need finite cx/cy and fw>0", ln + 1));
        }
        out.push((cx, cy, fw));
    }
    if out.is_empty() {
        return Err(format!("seed-list {path} produced no rows"));
    }
    Ok(out)
}

/// Which policy branch chose a step's target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Branch {
    Foci,
    Density,
    Random,
}
impl Branch {
    fn name(self) -> &'static str {
        match self {
            Branch::Foci => "foci",
            Branch::Density => "density",
            Branch::Random => "random",
        }
    }
}

/// Why a walk stopped (rev3 — surfaces the best-of-N two-stage screen tradeoffs).
#[derive(Clone, Copy, PartialEq, Eq)]
enum EndCause {
    /// Walk ran all `target` depths.
    ReachedTerminalDepth,
    /// All N candidates were ≥ the Stage-1 black cap (set-dominated; no survivor
    /// even reached the 768 render).
    BlackCapExhausted,
    /// Some candidate cleared the black cap + band but all were below the Stage-2
    /// occupancy floor (drifted feature-poor / empty).
    OccFloorExhausted,
    /// No candidate cleared the band screen (flat / instant-escape).
    DegenerateExhausted,
    /// The next step's frame width would fall below the `--min-fw` f64-reliable
    /// floor; the walk is truncated here and its accumulated candidates harvested
    /// (bring-up insurance against precision-degraded forced-F64 frames at extreme
    /// zoom — expected to fire near-zero times at current walk depths).
    MinFwFloor,
}
impl EndCause {
    fn name(self) -> &'static str {
        match self {
            EndCause::ReachedTerminalDepth => "terminal",
            EndCause::BlackCapExhausted => "black_cap",
            EndCause::OccFloorExhausted => "occ_floor",
            EndCause::DegenerateExhausted => "degenerate",
            EndCause::MinFwFloor => "min_fw_floor",
        }
    }
}

/// Where the chosen target lands inside the child frame.
#[derive(Clone, Copy)]
enum Placement {
    Center,
    Horizon,
    Random,
}
impl Placement {
    fn name(self) -> &'static str {
        match self {
            Placement::Center => "center",
            Placement::Horizon => "horizon",
            Placement::Random => "random",
        }
    }
}

/// One scale-space focus: location (field px), survival across σ, scale-normalized
/// peak response, isolation (peak ÷ local field mean), and the sampling score
/// (peak × isolation — NOT persistence).
#[derive(Clone, Copy)]
struct Focus {
    px: f64,
    py: f64,
    persistence: u32,
    peak_norm: f64,
    isolation: f64,
    score: f64,
}

/// One drawn best-of-N candidate next-center plus its policy provenance (before
/// any screening). The generator closures (root boundary sampler / per-node
/// policy) produce these; [`best_of_n_step`] screens + selects among them.
struct StepCand {
    center: Complex<f64>,
    branch: &'static str,
    placement: &'static str,
    fscore: f64,
}

/// Outcome of one best-of-N step.
enum StepResult {
    /// Winning node: frame, its 768 buffer (reused — no re-render), provenance,
    /// its interior fraction (the selection key, logged for the drift check), and
    /// the **chosen child's occupancy** (the value the occ floor 0.321 gates
    /// against; `NaN` for the depth-1 root steps, which are not descent children).
    Accepted(Frame, render::SampleBuffer, &'static str, &'static str, f64, f64, f64),
    /// No survivor across the N draws; the binding constraint.
    Died(EndCause),
}

/// One emitted candidate frame (a frame visited along a walk).
struct Candidate {
    idx: usize,
    walk: usize,
    depth: u32,
    target_depth: u32,
    /// Which root sampler seeded this walk ("8k" | "flat") — the attribution facet.
    root_src: &'static str,
    branch: &'static str,
    placement: &'static str,
    /// Sampling score of the focus that produced this frame (NaN for density/random).
    focus_score: f64,
    cx: f64,
    cy: f64,
    fw: f64,
    /// Chosen-child occupancy at this node's admission point (the value the occ
    /// floor 0.321 gates against, from [`energy::occupancy`]). `NaN` → null for the
    /// depth-1 root nodes (occupancy is a descent-child concept, not a root one).
    occ: f64,
    /// Preview PNG filename (relative to the run dir).
    png: String,
}

/// `guided-descend` entry point.
pub fn run_guided_descend(args: &GuidedDescendArgs) -> Result<(), String> {
    if args.n_walks == 0 {
        return Err("--n-walks must be > 0".into());
    }
    if args.depth_min == 0 || args.depth_max < args.depth_min {
        return Err(format!(
            "need 1 <= depth_min <= depth_max (got {}, {})",
            args.depth_min, args.depth_max
        ));
    }
    if !(args.min_fw > 0.0) {
        return Err(format!("--min-fw must be > 0 (got {})", args.min_fw));
    }
    // Policy mix (legacy finder): `--branch-weights f,d,r` overrides the `--w-*` trio.
    let (bw_f, bw_d, bw_r) = args.resolved_branch_weights()?;
    let w_foci = bw_f.max(0.0);
    let w_density = bw_d.max(0.0);
    let w_random = bw_r.max(0.0);
    let wsum = w_foci + w_density + w_random;
    if wsum <= 0.0 {
        return Err("target weights sum to zero".into());
    }
    let (p_foci, p_density) = (w_foci / wsum, w_density / wsum); // random = remainder

    // Descent-ablation seam resolution (defaults reproduce prior behaviour).
    let finder = args.finder;
    let selection = args.selection;
    let pct_band = args.resolved_pct_band()?;
    let pct_schedule = args.resolved_pct_schedule()?;

    // Parse a fixed complex constant from two decimal strings. Shallow base-scale →
    // modest precision is plenty for the f64 projection (exactly like render-one).
    let parse_c2 = |v: &[String], name: &str| -> Result<Complex<f64>, String> {
        if v.len() != 2 {
            return Err(format!("--{name} expects exactly two values <re> <im>, got {}", v.len()));
        }
        let re = hp::to_f64(&hp::parse_decimal(&v[0], 64)?);
        let im = hp::to_f64(&hp::parse_decimal(&v[1], 64)?);
        Ok(Complex::new(re, im))
    };

    // --- Dynamical z-plane descent modes: Julia (`z^d+c`) and Phoenix
    //     (`z²+c+p·z_{n-1}`). Both fix their constant(s) and descend the z-plane
    //     (z₀ = pixel) on the same fractal-agnostic policy; only the recurrence and
    //     root step differ from the c-plane Mandelbrot/multibrot walk. ---
    if args.julia && args.phoenix {
        return Err("--julia and --phoenix are mutually exclusive dynamical modes".into());
    }
    // Degree drives the escape recurrence (`z^d+c`) — for the c-plane multibrot AND,
    // under `--julia`, the dynamical Julia-multibrot. Phoenix is always degree 2.
    let degree = args.family.degree();

    // Phoenix mode: `--c` = additive constant (default 0.5667,0), `--p` = z_{n-1}
    // coefficient (default -0.5,0), both classic Ushiki.
    let phoenix_cp: Option<(Complex<f64>, Complex<f64>)> = if args.phoenix {
        if degree != 2 {
            return Err(
                "--phoenix is the degree-2 two-state plane; incompatible with --family multibrot*"
                    .into(),
            );
        }
        let cs = args.julia_c.clone().unwrap_or_else(|| vec!["0.5667".into(), "0".into()]);
        let ps = args.phoenix_p.clone().unwrap_or_else(|| vec!["-0.5".into(), "0".into()]);
        Some((parse_c2(&cs, "c")?, parse_c2(&ps, "p")?))
    } else {
        if args.phoenix_p.is_some() {
            return Err("--p is the Phoenix z_{n-1} coefficient; valid only with --phoenix".into());
        }
        None
    };

    // Julia mode: quadratic (`--family mandelbrot`) or Julia-multibrot
    // (`--family multibrot3|4|5`). `--c` is the fixed parameter.
    let julia_c: Option<Complex<f64>> = match (args.julia, &args.julia_c) {
        (true, None) => return Err("--julia requires --c <re> <im> (the fixed parameter)".into()),
        (true, Some(c)) => Some(parse_c2(c, "c")?),
        // `--c` without `--julia` is only meaningful in Phoenix mode (consumed above).
        (false, Some(_)) if !args.phoenix => {
            return Err("--c given without --julia/--phoenix; it is the fixed dynamical parameter \
                        and is meaningless on the c-plane"
                .into())
        }
        _ => None,
    };

    // Either dynamical mode roots at the base-scale z-plane view.
    let dynamical = julia_c.is_some() || phoenix_cp.is_some();
    if dynamical && args.julia_root_fw <= 0.0 {
        return Err(format!("--julia-root-fw must be > 0 (got {})", args.julia_root_fw));
    }
    // Center-descend is a Julia-only descent shape (a straight centered z-plane zoom).
    if args.julia_center && julia_c.is_none() {
        return Err("--julia-center is valid only with --julia".into());
    }
    // The boundary band (rev4 B2 random branch) reads DE; the dynamical kernels carry
    // none, so gate it off (the branch then draws a DE-free interior point).
    let random_boundary = args.random_boundary && !dynamical;

    // --- Injected-seed mode (`--seed-list`): pin the depth-1 frame from an external
    //     proposer (the atlas round-1 acceptance harness) instead of the internal
    //     8k/flat draw. All depth>=2 walk mechanics are byte-identical to a native
    //     run — ONLY the depth-1 seed source changes — so multiple arms differing
    //     only in their seed lists are directly comparable. `--n-walks` is overridden
    //     by the list length. Mandelbrot only; incompatible with --julia. ---
    let injected_seeds: Option<Vec<(f64, f64, f64)>> = match &args.seed_list {
        Some(p) => {
            if dynamical {
                return Err("--seed-list is c-plane-only; drop --julia/--phoenix".into());
            }
            Some(load_seed_list(p)?)
        }
        None => None,
    };
    // Effective walk count: the injected list length, else --n-walks.
    let n_walks = injected_seeds.as_ref().map_or(args.n_walks, |s| s.len());

    let (pl_center, pl_horizon, pl_random) = args.resolved_placement()?;
    let plsum = pl_center + pl_horizon + pl_random;
    let sigmas = args.resolved_sigmas()?;
    let band = args.band();
    // rev4 root-mixture + Part-B graft config.
    let root_mix = args.root_mix.clamp(0.0, 1.0);
    // Decoupled-start (2x2 experiment seam): `--root-start-fw` forces the depth-1
    // node width independently of each proposer's native center-selection scale
    // (8k window scan stays at `--root-zoom-8k`; flat screen stays at its sampled
    // fw). 0 (default) = coupled/native behaviour — byte-identical to prior runs.
    let root_start_fw = args.resolved_root_start_fw()?;
    let start_fw_8k = root_start_fw.unwrap_or(args.root_zoom_8k);
    let start_fw_flat = root_start_fw; // None ⇒ flat keeps its own sampled fw
    let (zoom_lo, zoom_hi) = args.resolved_zoom_band()?;
    let flat_box = args.resolved_flat_box()?;
    if args.flat_fw_lo <= 0.0 || args.flat_fw_hi <= args.flat_fw_lo {
        return Err(format!(
            "need 0 < flat_fw_lo < flat_fw_hi (got {}, {})",
            args.flat_fw_lo, args.flat_fw_hi
        ));
    }
    let score_cfg = args.root8k_score_cfg();

    let node_w = args.node_width.max(16);
    let node_h = (node_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    // rev4 B3 diversity radius in node px (0 ⇒ disabled).
    let foci_div_px = (args.foci_diversity_radius.max(0.0)) * node_w as f64;
    let prev_w = args.preview_width.max(16);
    let prev_h = (prev_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;

    let out_dir = Path::new(&args.out_dir);
    let tiles_dir = out_dir.join("tiles");
    crate::ensure_parent_dir(tiles_dir.join("x"))?;

    // Preview palette (diagnostic only — structure-finding is palette-independent).
    let cm_text = std::fs::read_to_string(&args.colormaps)
        .map_err(|e| format!("read {}: {e}", args.colormaps))?;
    let stops = probe::load_colormap(&cm_text, &args.preview_palette)?;
    let mirror = probe::colormap_mirror_needed(&cm_text, &args.preview_palette);
    let palette =
        Palette::from_srgb8_stops_mirrored(args.preview_palette.clone(), &stops, false, mirror);
    let params = color_params();
    let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };

    // Descent acceptance band: like the screen band but with the interior/black
    // occupancy clause DISABLED (rev1 Change 2 — black is inevitable in the first
    // couple of descends and is a *presentation* filter, not a navigation cull).
    // Keep the genuine-degenerate culls: flat (low spread) + instant-escape.
    let descent_band = crate::generate::AcceptBand { interior_max: 1.0, ..band };

    // Best-of-N two-stage screen (rev3 Change 2). Stage 1: cheap PROBE_W interior
    // cap (`render::black_fraction`, interior counts as black) — the aggressive
    // set-avoidance ceiling. Stage 2: 768 occupancy floor (`energy::occupancy`
    // parity scorer on the shaded node) — the feature-rich floor. Survivors are
    // selected by least interior fraction. Each gate enabled in (0,1); 0 or ≥1.0
    // disables. Black/interior + occupancy ONLY — no busyness axis (unseparable).
    let black_cap = args.descent_black_cap;
    let black_cap_on = black_cap > 0.0 && black_cap < 1.0;
    let occ_floor = args.descent_occ_floor;
    let occ_on = occ_floor > 0.0 && occ_floor < 1.0;
    // Default-palette gate render for the occupancy probe (occupancy is
    // palette-invariant to <1%; same palette `density_focus`/`present` shade with).
    let gate_palette = crate::palette::builtin("default", false).expect("default palette");

    // The conceptual root frame (full Mandelbrot view) — only the initial `parent`
    // placeholder; rev4's depth-1 step ignores it (the root mixture sets its own
    // center+fw). Per-node renders use `node_w`.
    let root = Frame {
        center: Complex::new(-0.5, 0.0),
        frame_width: 3.0,
        out_width: node_w,
        out_height: node_h,
    };
    let root_desc = if let Some((pc, pp)) = phoenix_cp {
        format!(
            "PHOENIX z-plane descent: c=({:.4},{:.4}) p=({:.4},{:.4}), root base-scale fw {} @ center 0",
            pc.re, pc.im, pp.re, pp.im, args.julia_root_fw
        )
    } else {
        match julia_c {
        Some(c) => format!(
            "JULIA{} z-plane descent: c=({:.6},{:.6}), root base-scale fw {} @ symmetry center 0",
            if degree == 2 { String::new() } else { format!("-multibrot{degree}") },
            c.re, c.im, args.julia_root_fw
        ),
        None => format!(
            "root-mix {root_mix:.2} (8k vs flat); 8k root-zoom {} + criterion black<={} mean[{},{}] var>={}; flat box {flat_box:?} fw[{},{}] screen {}px",
            args.root_zoom_8k, score_cfg.black_max, score_cfg.mean_lo, score_cfg.mean_hi, score_cfg.var_floor,
            args.flat_fw_lo, args.flat_fw_hi, args.flat_screen_width,
        ),
        }
    };
    eprintln!(
        "guided-descend (rev4): {} walks, depth [{},{}], zoom/step log-uniform [{},{}], seed {}\n  \
         {}\n  \
         weights foci/density/random = {:.2}/{:.2}/{:.2}, placement {:.2}/{:.2}/{:.2}, sigma {:?}, \
         foci-diversity {:.0}px, random-boundary {}\n  \
         node {}x{} ss1, preview {}x{} ({}), maxiter {}\n  \
         descent culls (depth>=2): flat spread>={} + instant-escape esc_med>={} + min-fw floor {:e}\n  \
         best-of-{}: Stage-1 interior-cap {} (probe {}px), Stage-2 occ-floor {} (@{}px), select min-interior",
        n_walks, args.depth_min, args.depth_max, zoom_lo, zoom_hi, args.seed,
        root_desc,
        p_foci, p_density, 1.0 - p_foci - p_density,
        pl_center / plsum, pl_horizon / plsum, pl_random / plsum, sigmas,
        foci_div_px, random_boundary,
        node_w, node_h, prev_w, prev_h, args.preview_palette, args.maxiter,
        band.spread_min, band.esc_median_min, args.min_fw,
        args.descent_candidates.max(1),
        if black_cap_on { format!("black_frac<{black_cap}") } else { "OFF".into() }, PROBE_W,
        if occ_on { format!("occ>={occ_floor}") } else { "OFF".into() }, node_w,
    );
    if degree != 2 && !dynamical {
        // d3/d4/d5 all carry per-family band defaults now; only a future untuned
        // degree would fall back to the Mandelbrot values. (Julia-multibrot descends
        // the z-plane and skips the c-plane 8k/flat root apparatus this describes.)
        let tuned = matches!(
            args.family,
            WalkFamily::Multibrot3 | WalkFamily::Multibrot4 | WalkFamily::Multibrot5
        );
        eprintln!(
            "  FAMILY multibrot{degree}: c-plane recurrence z^{degree}+c, origin-square root box; \
             8k band mean[{},{}] var>={}, flat spread>={} — {}",
            score_cfg.mean_lo, score_cfg.mean_hi, score_cfg.var_floor, band.spread_min,
            if tuned {
                "per-family DEFAULTS tuned by eye for this degree"
            } else {
                "pre-gate/seed-bias thresholds left at Mandelbrot values (expected to mis-fire — tune later)"
            }
        );
    }

    let render_node = |frame: &Frame| -> render::SampleBuffer {
        if let Some((pc, pp)) = phoenix_cp {
            // Phoenix: two-state dynamical, frame addresses the z-plane (z₀ = pixel).
            let backend = PhoenixBackend::new(pc, pp, args.maxiter, args.bailout, trap);
            render::iterate_samples(&backend, frame, 1)
        } else if let Some(c) = julia_c {
            // Julia / Julia-multibrot: fixed parameter, z-plane (z₀ = pixel), degree
            // from `--family`. Shallow base-scale → f64.
            let backend = JuliaBackend::new_degree(c, args.maxiter, args.bailout, trap, degree);
            render::iterate_samples(&backend, frame, 1)
        } else {
            let prec = hp::prec_bits(frame.out_width, frame.frame_width);
            let cre = BigFloat::from_f64(frame.center.re, prec);
            let cim = BigFloat::from_f64(frame.center.im, prec);
            probe::render_mandel_panel(
                &cre, &cim, frame.center, frame.frame_width, frame.out_width, frame.out_height, 1,
                args.maxiter, args.bailout, degree, prec, trap, BackendChoice::F64,
            )
            .buf
        }
    };

    let t0 = std::time::Instant::now();
    // --- rev4 A1 (Mandelbrot only): load (or build+cache) the durable 8k smooth
    //     field, then scan it once for windows passing the score seam at the 8k
    //     root zoom. Julia mode roots at the deterministic base-scale z-plane view,
    //     so it skips the c-plane field entirely. ---
    let windows: Vec<PassWindow> = if dynamical {
        eprintln!("  dynamical mode: skipping c-plane 8k field (root is the base-scale z-plane view)");
        Vec::new()
    } else if injected_seeds.is_some() {
        eprintln!(
            "  injected-seed mode: {} seeds from {} — skipping 8k field (depth-1 pinned)",
            n_walks,
            args.seed_list.as_deref().unwrap_or("?")
        );
        Vec::new()
    } else {
        let rf: RootField = RootField::load_or_build(args.maxiter, args.bailout, trap, degree)?;
        // 8k window footprint at root-zoom-8k (16:9 to match the node aspect).
        let win_w = ((args.root_zoom_8k / (rf.re_hi - rf.re_lo)) * rf.w as f64).round().max(1.0) as usize;
        let win_h = ((args.root_zoom_8k * node_h as f64 / node_w as f64 / (rf.im_hi - rf.im_lo)) * rf.h as f64)
            .round()
            .max(1.0) as usize;
        let stride = ((win_h as f64) * SCAN_STRIDE_FRAC).round().max(1.0) as usize;
        let windows = rf.passing_windows(win_w, win_h, stride, &score_cfg);
        eprintln!(
            "  8k field ready in {:.2}s; scanned {}x{} windows (stride {}) → {} passing the score seam",
            t0.elapsed().as_secs_f64(), win_w, win_h, stride, windows.len()
        );
        if windows.is_empty() && root_mix > 0.0 {
            return Err("8k root window scan left nothing — loosen --root8k-* criterion".into());
        }
        match root_start_fw {
            Some(fw) => eprintln!(
                "  decoupled start-fw {} (8k selects windows @ {} → content-focus center, flat screens @ sampled fw; both START at {})",
                fw, args.root_zoom_8k, fw
            ),
            None => eprintln!("  start-fw coupled to native proposer scale (no --root-start-fw override)"),
        }
        windows
    };

    // Best-of-N screen config (shared across every step of every walk).
    let screen = StepScreen {
        n_cand: args.descent_candidates.max(1),
        node_w,
        node_h,
        maxiter: args.maxiter,
        band: &descent_band,
        black_cap,
        black_cap_on,
        occ_floor,
        occ_on,
        gate_palette: &gate_palette,
        params: &params,
    };

    // Part-0: the occ floor over-fires at the depth-1→2 transition. Unless the
    // legacy `--descent-occ-at-d1d2` flag is set, the d1→d2 step uses a screen
    // variant with the occupancy floor disabled (Stage-1 interior cap and
    // least-interior selection are still in force); d≥3 keeps the full screen.
    let screen_d1d2 = StepScreen { occ_on: false, ..screen };
    let skip_occ_d1d2 = !args.descent_occ_at_d1d2;
    if skip_occ_d1d2 && occ_on {
        eprintln!("  occ floor SKIPPED at d1→d2 step (Part-0 fix); active for d>=3");
    }

    let mut rng = SplitMix64(args.seed);
    let mut cands: Vec<Candidate> = Vec::new();
    // Per-walk reached depth + intended target (for the ladder view + died-early count).
    let mut walk_reached: Vec<(u32, u32)> = Vec::with_capacity(n_walks);
    // Per-walk terminal cause (instrumentation: the field that didn't survive run4 —
    // persisted to walks.jsonl for the per-cell generated-fate breakdown).
    let mut walk_cause: Vec<EndCause> = Vec::with_capacity(n_walks);
    let mut branch_counts = [0usize; 3]; // foci, density, random
    let mut root8k_count = 0usize; // depth-1 nodes seeded by the 8k field
    let mut rootflat_count = 0usize; // depth-1 nodes seeded by the flat sampler
    // Per-walk root sampler ("8k" | "flat" | "" if the root died) for attribution.
    let mut walk_root_src: Vec<&'static str> = Vec::with_capacity(n_walks);
    let mut died_early = 0usize;
    // End-of-walk cause: [ReachedTerminalDepth, BlackCapExhausted, OccFloorExhausted, DegenerateExhausted, MinFwFloor].
    let mut cause_counts = [0usize; 5];
    let mut black_rejects = 0usize; // total best-of-N candidates killed by the Stage-1 black cap
    let mut occ_rejects = 0usize; // total best-of-N candidates killed by the Stage-2 occupancy floor
    // Per-step chosen interior fraction (the drift check — does min-interior pull toward empty?).
    let mut chosen_interiors: Vec<f64> = Vec::new();

    // Per-walk deterministic sub-seed source (only consulted when --per-walk-rng).
    // Derive well-mixed, walk-independent sub-seeds by running a SplitMix64 keyed on
    // the global seed and taking one output per walk index.
    let mut walk_seed_src = SplitMix64(args.seed);
    for w in 0..n_walks {
        // Paired-study mode: reseed this walk's stream from (seed, walk_index) so its
        // depth-1 seed does not depend on prior walks' (config-dependent) draw counts.
        if args.per_walk_rng {
            rng = SplitMix64(walk_seed_src.next_u64());
        }
        let target = args.depth_min + (rng.below((args.depth_max - args.depth_min + 1) as usize) as u32);
        let mut parent = root;
        // `None` until the depth-1 root step renders the first node.
        let mut parent_buf: Option<render::SampleBuffer> = None;
        let mut reached = 0u32;
        // Walk completed unless a step dies; the dying step overwrites this.
        let mut end_cause = EndCause::ReachedTerminalDepth;
        // The root sampler this walk drew (set on the depth-1 step; "" if it died).
        let mut cur_root_src: &'static str = "";

        for d in 1..=target {
            let result = if d == 1 {
                if let Some(seeds) = &injected_seeds {
                    // --- INJECTED ROOT STEP: pin depth-1 to the proposer's seed.
                    //     Consumes NO rng (unlike the native samplers' screen-redraw
                    //     loops), so depth>=2 shares one rng stream across arms. ---
                    cur_root_src = "injected";
                    root_step_injected(seeds[w], node_w, node_h, args.maxiter, &render_node)
                } else if dynamical {
                    // --- DYNAMICAL ROOT STEP (Julia / Julia-multibrot / Phoenix):
                    //     deterministic base-scale z-plane view at center 0. Descent
                    //     then leaves it. ---
                    cur_root_src = if phoenix_cp.is_some() { "phoenix" } else { "julia" };
                    root_step_julia(args.julia_root_fw, node_w, node_h, args.maxiter, &render_node)
                } else if rng.unit() < root_mix {
                    // --- ROOT STEP (rev4 Part A): 50/50 sampler mixture, permissive
                    //     (≤80% black), occupancy floor NOT applied. ---
                    cur_root_src = "8k";
                    root_step_8k(
                        &windows, args.root_zoom_8k, start_fw_8k, root_start_fw.is_some(),
                        node_w, node_h, args.maxiter,
                        score_cfg.black_max, &render_node, &mut rng,
                    )
                } else {
                    cur_root_src = "flat";
                    root_step_flat(
                        flat_box, args.flat_fw_lo, args.flat_fw_hi, start_fw_flat,
                        args.flat_screen_width,
                        &band, node_w, node_h, args.maxiter, &render_node, &mut rng,
                    )
                }
            } else if args.julia_center {
                // --- CENTER-DESCEND STEP (depth ≥ 2, `--julia-center`): pure centered
                //     zoom. Shrink fw by the SAME per-step ratio the normal walk uses,
                //     honour the SAME --min-fw floor, keep the window at (0,0). No
                //     finder / best-of-N / placement — every rung is emitted as a
                //     candidate in the normal shape. ---
                let new_fw = parent.frame_width * sample_log_uniform(zoom_lo, zoom_hi, &mut rng);
                if new_fw < args.min_fw {
                    StepResult::Died(EndCause::MinFwFloor)
                } else {
                    center_step_julia(new_fw, node_w, node_h, args.maxiter, &render_node)
                }
            } else {
                // --- NORMAL STEP (depth ≥ 2): per-node finder + placement, unchanged
                //     rev3 best-of-N set-avoidance; rev4 B4 jitters the zoom. ---
                let new_fw = parent.frame_width * sample_log_uniform(zoom_lo, zoom_hi, &mut rng);
                if new_fw < args.min_fw {
                    // f64-reliable descent floor: stop before crossing below it and
                    // harvest what the walk already collected (all visited frames are
                    // already emitted candidates). Distinct cause so it is counted.
                    StepResult::Died(EndCause::MinFwFloor)
                } else {
                    let new_fh = new_fw * parent.out_height as f64 / parent.out_width as f64;
                    // parent_buf is always Some here (depth-1 root step set it).
                    let parent_samples = &parent_buf.as_ref().unwrap().samples;
                    // Part-0: skip the occ floor at the depth-1→2 step.
                    let step_screen = if d == 2 && skip_occ_d1d2 { &screen_d1d2 } else { &screen };
                    match finder {
                        FinderMode::Legacy => {
                            let mut gen = |rng: &mut SplitMix64| -> Option<StepCand> {
                                let (focus, branch, fscore) = pick_target(
                                    &parent, parent_samples, node_w as usize, node_h as usize, &sigmas,
                                    (p_foci, p_density), foci_div_px, random_boundary, rng,
                                );
                                let placement = if branch == Branch::Random {
                                    Placement::Center
                                } else {
                                    pick_placement((pl_center, pl_horizon, pl_random), plsum, rng)
                                };
                                let center = child_center(focus, placement, new_fw, new_fh, rng);
                                Some(StepCand { center, branch: branch.name(), placement: placement.name(), fscore })
                            };
                            best_of_n_step(
                                step_screen, new_fw, selection, &render_node, &mut gen, &mut rng,
                                &mut black_rejects, &mut occ_rejects,
                            )
                        }
                        FinderMode::Percentile => {
                            let pband = pct_band_for_depth(&pct_schedule, pct_band, d);
                            let mut gen = |rng: &mut SplitMix64| -> Option<StepCand> {
                                percentile_gen(
                                    &parent, parent_samples, node_w as usize, node_h as usize, new_fw,
                                    pband, args.pct_interior_cap, args.pct_max_tries, rng,
                                )
                            };
                            best_of_n_step(
                                step_screen, new_fw, selection, &render_node, &mut gen, &mut rng,
                                &mut black_rejects, &mut occ_rejects,
                            )
                        }
                    }
                }
            };

            match result {
                StepResult::Accepted(child, buf, branch, placement, fscore, interior, occ) => {
                    match branch {
                        "foci" => branch_counts[0] += 1,
                        "density" => branch_counts[1] += 1,
                        "random" => branch_counts[2] += 1,
                        "root8k" => root8k_count += 1,
                        // Percentile-finder steps use no foci/density/random bucket
                        // (the per-candidate `branch` in pool.jsonl carries the tag).
                        "percentile" => {}
                        // Center-descend rungs used no policy branch — count them in
                        // none of the buckets (roots still count as rootjulia below).
                        "center" => {}
                        _ => rootflat_count += 1, // "rootflat" / "rootjulia"
                    }
                    chosen_interiors.push(interior);
                    cands.push(Candidate {
                        idx: cands.len(),
                        walk: w,
                        depth: d,
                        target_depth: target,
                        root_src: cur_root_src,
                        branch,
                        placement,
                        focus_score: fscore,
                        cx: child.center.re,
                        cy: child.center.im,
                        fw: child.frame_width,
                        occ,
                        png: String::new(), // filled after the parallel preview pass
                    });
                    reached = d;
                    parent = child;
                    parent_buf = Some(buf);
                }
                StepResult::Died(cause) => {
                    end_cause = cause;
                    break;
                }
            }
        }
        walk_root_src.push(cur_root_src);

        cause_counts[match end_cause {
            EndCause::ReachedTerminalDepth => 0,
            EndCause::BlackCapExhausted => 1,
            EndCause::OccFloorExhausted => 2,
            EndCause::DegenerateExhausted => 3,
            EndCause::MinFwFloor => 4,
        }] += 1;

        if reached < target {
            died_early += 1;
        }
        walk_reached.push((reached, target));
        walk_cause.push(end_cause);
        if (w + 1) % 20 == 0 || w + 1 == n_walks {
            eprintln!(
                "  walk {}/{}: {} candidates so far ({:.1}s)",
                w + 1,
                n_walks,
                cands.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    if cands.is_empty() {
        return Err("no candidates produced (every walk died on the first step?)".into());
    }

    // Preview filenames are purely idx-derived, so set them up front. This lets the
    // durable pool.jsonl/walks.jsonl be written BEFORE the best-effort preview render
    // — a kill or OOM in the cosmetic preview/grid stage (which collects every
    // preview in memory, ~4 GB at 600 walks) can no longer discard the run's data.
    for c in cands.iter_mut() {
        c.png = format!("tiles/tile_{:04}.png", c.idx);
    }

    // --- pool.jsonl (one row per candidate) ---
    let mut jsonl = String::new();
    for c in &cands {
        let _ = writeln!(
            jsonl,
            "{{ \"idx\": {}, \"walk\": {}, \"depth\": {}, \"target_depth\": {}, \
             \"root_src\": \"{}\", \"branch\": \"{}\", \"placement\": \"{}\", \"focus_score\": {}, \
             \"cx\": {}, \"cy\": {}, \"fw\": {}, \"occ\": {}, \"png\": \"{}\" }}",
            c.idx, c.walk, c.depth, c.target_depth, c.root_src, c.branch, c.placement,
            jnum(c.focus_score), jnum(c.cx), jnum(c.cy), jnum(c.fw), jnum(c.occ), c.png,
        );
    }
    std::fs::write(out_dir.join("pool.jsonl"), jsonl)
        .map_err(|e| format!("write pool.jsonl: {e}"))?;

    // --- walks.jsonl: one row per walk (the per-walk fate instrumentation that
    //     run4 lacked). cause = terminal cause; death_depth = depth of the step
    //     that died (= reached+1) or null if the walk reached its target. Root
    //     center features (cx/cy/fw) are the walk's depth-1 node (null if the walk
    //     died at the root step). band_energy + finer center features are computed
    //     post-hoc in Python from (cx,cy,fw) — recorded here, not baked in. ---
    let mut root_of_walk: std::collections::HashMap<usize, &Candidate> = std::collections::HashMap::new();
    // Depth-2 node per walk: its occupancy is the seed's chosen-child occupancy at
    // the admission point the seed-admission occ floor (0.321) is calibrated for —
    // exactly what the depth-2 descendability probe interrogates. null if the walk
    // died before reaching depth 2.
    let mut child2_of_walk: std::collections::HashMap<usize, &Candidate> = std::collections::HashMap::new();
    for c in &cands {
        if c.depth == 1 {
            root_of_walk.insert(c.walk, c);
        } else if c.depth == 2 {
            child2_of_walk.insert(c.walk, c);
        }
    }
    let mut walks_jsonl = String::new();
    for w in 0..n_walks {
        let (reached, target) = walk_reached[w];
        let cause = walk_cause[w];
        let death_depth = if reached < target {
            format!("{}", reached + 1)
        } else {
            "null".to_string()
        };
        let (rcx, rcy, rfw) = match root_of_walk.get(&w) {
            Some(c) => (jnum(c.cx), jnum(c.cy), jnum(c.fw)),
            None => ("null".into(), "null".into(), "null".into()),
        };
        let child_occ = match child2_of_walk.get(&w) {
            Some(c) => jnum(c.occ),
            None => "null".to_string(),
        };
        let _ = writeln!(
            walks_jsonl,
            "{{ \"walk\": {}, \"root_src\": \"{}\", \"target_depth\": {}, \"reached_depth\": {}, \
             \"cause\": \"{}\", \"death_depth\": {}, \"root_cx\": {}, \"root_cy\": {}, \"root_fw\": {}, \
             \"child_occ\": {} }}",
            w, walk_root_src[w], target, reached, cause.name(), death_depth, rcx, rcy, rfw, child_occ,
        );
    }
    std::fs::write(out_dir.join("walks.jsonl"), walks_jsonl)
        .map_err(|e| format!("write walks.jsonl: {e}"))?;

    // --- summary.json: run-level roll-up. Surfaces the --min-fw floor fire count
    //     (min_fw_truncations) as bring-up telemetry — expected ~0 at current walk
    //     depths. Durable, written before the cosmetic preview stage. Hand-rolled to
    //     match the repo's serde-free JSON convention. ---
    let mut summary = String::new();
    let _ = write!(
        summary,
        "{{ \"seed\": {}, \"n_walks\": {}, \"candidates\": {}, \"died_early\": {}, \
         \"min_fw\": {}, \"min_fw_truncations\": {}, \
         \"cause_counts\": {{ \"terminal\": {}, \"black_cap\": {}, \"occ_floor\": {}, \"degenerate\": {}, \"min_fw_floor\": {} }} }}\n",
        args.seed, n_walks, cands.len(), died_early,
        jnum(args.min_fw), cause_counts[4],
        cause_counts[0], cause_counts[1], cause_counts[2], cause_counts[3], cause_counts[4],
    );
    std::fs::write(out_dir.join("summary.json"), summary)
        .map_err(|e| format!("write summary.json: {e}"))?;

    // --- preview renders (parallel; the per-candidate frame is independent) ---
    // Each worker renders a full preview, SAVES it to tiles/, and returns only the
    // small 240x135 thumbnail. Peak memory is thus ~thumbnails (~0.5 GB at 600 walks)
    // rather than every full 640x360 preview at once (~4 GB) — which OOMed the
    // cosmetic stage. Runs AFTER pool.jsonl/walks.jsonl, so data is already durable.
    eprintln!("rendering {} previews at {}x{} ...", cands.len(), prev_w, prev_h);
    let tp = std::time::Instant::now();
    let mut grid_thumbs: Vec<RgbImage> = cands
        .par_iter()
        .map(|c| -> Result<RgbImage, String> {
            let frame = Frame {
                center: Complex::new(c.cx, c.cy),
                frame_width: c.fw,
                out_width: prev_w,
                out_height: prev_h,
            };
            let (samples, spacing) = if let Some((pc, pp)) = phoenix_cp {
                let backend = PhoenixBackend::new(pc, pp, args.maxiter, args.bailout, trap);
                let buf = render::iterate_samples(&backend, &frame, 1);
                (buf.samples, frame.pixel_size())
            } else if let Some(jc) = julia_c {
                let backend = JuliaBackend::new_degree(jc, args.maxiter, args.bailout, trap, degree);
                let buf = render::iterate_samples(&backend, &frame, 1);
                (buf.samples, frame.pixel_size())
            } else {
                let prec = hp::prec_bits(prev_w, c.fw);
                let cre = BigFloat::from_f64(c.cx, prec);
                let cim = BigFloat::from_f64(c.cy, prec);
                let panel = probe::render_mandel_panel(
                    &cre, &cim, frame.center, c.fw, prev_w, prev_h, 1, args.maxiter, args.bailout,
                    degree, prec, trap, BackendChoice::F64,
                );
                (panel.buf.samples, panel.spacing)
            };
            let img = render::shade_and_downsample(&samples, prev_w, prev_h, 1, &palette, &params, spacing);
            img.save(tiles_dir.join(format!("tile_{:04}.png", c.idx)))
                .map_err(|e| format!("save preview {}: {e}", c.idx))?;
            Ok(image::imageops::resize(&img, 240, 135, image::imageops::FilterType::Triangle))
        })
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!("  previews in {:.2}s", tp.elapsed().as_secs_f64());

    // --- a quick flat PNG contact grid (sibling to the HTML) ---
    for (c, th) in cands.iter().zip(grid_thumbs.iter_mut()) {
        annotate(th, c);
    }
    let grid = sheet::compose_grid(&grid_thumbs, Some(args.cols.max(1)));
    grid.save(out_dir.join("pool_grid.png"))
        .map_err(|e| format!("save pool grid: {e}"))?;

    // --- pool_sheet.html (the deliverable Matt judges) ---
    let ci = interior_summary(&chosen_interiors);
    let html = build_html(
        &cands, &walk_reached, &walk_root_src, &branch_counts, root8k_count, rootflat_count,
        died_early, &cause_counts, black_rejects, black_cap_on, black_cap, occ_rejects, occ_on,
        occ_floor, &ci, args, n_walks, &sigmas, zoom_lo, zoom_hi,
    );
    let html_path = out_dir.join("pool_sheet.html");
    std::fs::write(&html_path, html).map_err(|e| format!("write pool_sheet.html: {e}"))?;

    // --- depth histogram (over emitted candidates) ---
    let mut depth_hist = vec![0usize; (args.depth_max + 1) as usize];
    for c in &cands {
        depth_hist[c.depth as usize] += 1;
    }

    // --- root-window spread: how diverse are the depth-1 boundary samples? ---
    let roots: Vec<&Candidate> = cands.iter().filter(|c| c.depth == 1).collect();
    let (rsx, rsy) = root_spread(&roots);
    // --- repetition: most-repeated emitted frame (round center+fw). ---
    let (top_mult, n_unique) = repetition(&cands);

    // Realized root-mix split (attempted sampler per walk) + productive (reached≥1).
    let n_8k = walk_root_src.iter().filter(|s| **s == "8k").count();
    let n_flat = walk_root_src.iter().filter(|s| **s == "flat").count();
    let (mut prod_8k, mut prod_flat) = (0usize, 0usize);
    for (src, &(reached, _)) in walk_root_src.iter().zip(walk_reached.iter()) {
        if reached >= 1 {
            match *src {
                "8k" => prod_8k += 1,
                "flat" => prod_flat += 1,
                _ => {}
            }
        }
    }

    println!("=== guided-descend (rev4) ===");
    println!(
        "seed={}  walks={}  candidates={}  best-of-{}",
        args.seed, n_walks, cands.len(), screen.n_cand
    );
    println!(
        "realized root-mix (target {:.2}): 8k={}/{} walks (productive {}), flat={}/{} walks (productive {})",
        root_mix, n_8k, n_walks, prod_8k, n_flat, n_walks, prod_flat
    );
    println!(
        "branch breakdown: root8k={} rootflat={} foci={} density={} random={} (target foci/density/random {:.2}/{:.2}/{:.2})",
        root8k_count, rootflat_count, branch_counts[0], branch_counts[1], branch_counts[2],
        p_foci, p_density, 1.0 - p_foci - p_density,
    );
    print!("depth histogram (emitted, all visited frames):");
    for (d, n) in depth_hist.iter().enumerate().skip(1) {
        print!(" d{d}={n}");
    }
    println!();
    // Terminal-depth histogram (the target each walk drew) — this is where the
    // depth_min=4 floor is visible (the emitted histogram always carries d1..d3
    // pass-through frames).
    let mut target_hist = vec![0usize; (args.depth_max + 1) as usize];
    for &(_reached, target) in &walk_reached {
        target_hist[target as usize] += 1;
    }
    print!("terminal-depth histogram (target drawn, floor={}):", args.depth_min);
    for (d, n) in target_hist.iter().enumerate().skip(1) {
        if *n > 0 || (d as u32 >= args.depth_min && d as u32 <= args.depth_max) {
            print!(" d{d}={n}");
        }
    }
    println!();
    // Reached-depth histogram (how deep walks actually got after early deaths).
    let mut reached_hist = vec![0usize; (args.depth_max + 1) as usize];
    for &(reached, _target) in &walk_reached {
        reached_hist[reached as usize] += 1;
    }
    print!("reached-depth histogram (actual leaf):");
    for (d, n) in reached_hist.iter().enumerate() {
        if *n > 0 {
            print!(" d{d}={n}");
        }
    }
    println!();
    println!(
        "root-window spread: {} depth-1 samples, center std (re,im) = ({:.3}, {:.3}) over the full set view",
        roots.len(), rsx, rsy
    );
    println!(
        "repetition: {} unique frames / {} candidates; most-repeated frame ×{}",
        n_unique, cands.len(), top_mult
    );
    println!(
        "walks died early (terminated before target depth): {}/{}",
        died_early, n_walks
    );
    println!(
        "end-of-walk cause: terminal={} black_cap_exhausted={} occ_floor_exhausted={} degenerate_exhausted={} min_fw_floor={}",
        cause_counts[0], cause_counts[1], cause_counts[2], cause_counts[3], cause_counts[4],
    );
    println!(
        "min-fw floor (--min-fw {:e}): {} walk(s) truncated below the f64-reliable depth (expected ~0)",
        args.min_fw, cause_counts[4],
    );
    println!(
        "best-of-N rejects: black-cap {} ({}), occ-floor {} ({}) — total candidates screened out",
        black_rejects, if black_cap_on { format!("<{black_cap}") } else { "OFF".into() },
        occ_rejects, if occ_on { format!(">={occ_floor}") } else { "OFF".into() },
    );
    println!(
        "chosen interior fraction (drift check): n={} min={:.3} p25={:.3} med={:.3} p75={:.3} max={:.3} mean={:.3}",
        ci.n, ci.min, ci.p25, ci.med, ci.p75, ci.max, ci.mean,
    );
    println!("elapsed: {:.1}s", t0.elapsed().as_secs_f64());
    println!("pool.jsonl + pool_grid.png + tiles/ under {}", out_dir.display());
    println!("sheet: {}", html_path.display());
    Ok(())
}

// ===========================================================================
// Best-of-N set-avoidant step selection (rev3 Change 2)
// ===========================================================================

/// Shared best-of-N screen config (constant across the whole run).
struct StepScreen<'a> {
    n_cand: usize,
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    band: &'a crate::generate::AcceptBand,
    black_cap: f64,
    black_cap_on: bool,
    occ_floor: f64,
    occ_on: bool,
    gate_palette: &'a Palette,
    params: &'a crate::coloring::ColorParams,
}

/// Draw up to `cfg.n_cand` candidates from `gen` and select a winning survivor
/// (per `selection`: **random-survivor** by default, else least-interior) of a
/// two-stage screen:
/// - **Stage 1 (cheap):** probe at [`PROBE_W`] escape-time, reject interior ≥ cap.
///   Interior fraction is scale-robust, so the cheap probe is a sound ceiling.
/// - **Stage 2 (768):** render the node, cull degenerate (band: flat/instant-escape),
///   then shade + reject occupancy < floor. The winner's 768 buffer is reused.
///
/// With no survivor, [`StepResult::Died`] reports the furthest-reached binding
/// constraint: occ-floor (a candidate cleared the cap+band) > black-cap (all too
/// black) > degenerate (none cleared the band).
fn best_of_n_step(
    cfg: &StepScreen,
    new_fw: f64,
    selection: SelectionMode,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
    gen: &mut dyn FnMut(&mut SplitMix64) -> Option<StepCand>,
    rng: &mut SplitMix64,
    black_rejects: &mut usize,
    occ_rejects: &mut usize,
) -> StepResult {
    let mut saw_black = false; // a candidate failed the Stage-1 interior cap
    let mut saw_degen = false; // a candidate cleared the cap but failed the band
    let mut saw_occ = false; // a candidate cleared the band but failed the occ floor
    // Winner among full survivors: least-interior keeps the min-interior-fraction
    // survivor (draws NO rng); random-survivor reservoir-samples uniformly on `rng`.
    let mut best: Option<(Frame, render::SampleBuffer, &'static str, &'static str, f64, f64)> = None;
    let mut n_surv = 0usize;

    for _ in 0..cfg.n_cand.max(1) {
        let Some(sc) = gen(rng) else { break };
        let frame = Frame {
            center: sc.center,
            frame_width: new_fw,
            out_width: cfg.node_w,
            out_height: cfg.node_h,
        };

        // --- Stage 1: cheap interior screen (~PROBE_W escape-time). ---
        if cfg.black_cap_on {
            let probe_h = (PROBE_W as f64 * cfg.node_h as f64 / cfg.node_w as f64).round().max(1.0) as u32;
            let probe = Frame { out_width: PROBE_W, out_height: probe_h, ..frame };
            let pbuf = render_node(&probe);
            if render::black_fraction(&pbuf.samples) as f64 >= cfg.black_cap {
                saw_black = true;
                *black_rejects += 1;
                continue;
            }
        }

        // --- Stage 2: 768 node render → band cull → occupancy floor. ---
        let buf = render_node(&frame);
        let (int_frac, esc) = generate::screen_stats(&buf.samples, cfg.maxiter);
        if !cfg.band.test(int_frac, esc.spread, esc.median).accepted {
            saw_degen = true;
            continue;
        }
        if cfg.occ_on {
            let img = render::shade_and_downsample(
                &buf.samples, cfg.node_w, cfg.node_h, 1, cfg.gate_palette, cfg.params, frame.pixel_size(),
            );
            if energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR) < cfg.occ_floor {
                saw_occ = true;
                *occ_rejects += 1;
                continue;
            }
        }

        // Full survivor. least-interior: keep iff least-set so far (no rng).
        // random-survivor: reservoir-sample (replace with prob 1/n_surv).
        n_surv += 1;
        let take = match selection {
            SelectionMode::LeastInterior => best.as_ref().map_or(true, |b| int_frac < b.5),
            SelectionMode::RandomSurvivor => rng.below(n_surv) == 0,
        };
        if take {
            best = Some((frame, buf, sc.branch, sc.placement, sc.fscore, int_frac));
        }
    }

    match best {
        Some((f, b, br, pl, fs, ifr)) => {
            // Emit the chosen child's occupancy — the value the occ floor (0.321)
            // gates against. Computed ONCE, on the winner's already-rendered buffer,
            // via the shared `energy::occupancy` primitive (reused, never
            // reimplemented). This makes the floor *observable* even at steps where
            // it is not gating (e.g. the d1→d2 descendability probe, where `occ_on`
            // is false), and is byte-identical to the gate's value when `occ_on`. It
            // consumes no RNG and does not affect selection, so walk generation is
            // unperturbed.
            let occ = {
                let img = render::shade_and_downsample(
                    &b.samples, cfg.node_w, cfg.node_h, 1, cfg.gate_palette, cfg.params, f.pixel_size(),
                );
                energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR)
            };
            StepResult::Accepted(f, b, br, pl, fs, ifr, occ)
        }
        None if saw_occ => StepResult::Died(EndCause::OccFloorExhausted),
        None if saw_black => StepResult::Died(EndCause::BlackCapExhausted),
        None if saw_degen => StepResult::Died(EndCause::DegenerateExhausted),
        None => StepResult::Died(EndCause::DegenerateExhausted),
    }
}

// ===========================================================================
// rev4 Part A: root sampler mixture (both permissive, occ floor NOT applied)
// ===========================================================================

/// 8k-field root: sample a passing window uniformly, render the depth-1 node, and
/// accept the first whose node-resolution black fraction is permissive (≤ `black_max`).
/// The score-seam scan already enforced the criterion at 8k resolution, so this
/// node-level recheck only catches the rare resolution-mismatch case.
///
/// Two scales decouple here (2x2 experiment). The window is selected at `select_fw`
/// (the field's native `--root-zoom-8k`, the spatial-selection scale, unchanged);
/// the depth-1 node starts descent at `start_fw` (the experiment factor). When
/// `recenter` is set (decoupled mode, `--root-start-fw > 0`) the node is placed on
/// the window's **energy-weighted content focus** (rendered at `select_fw`, found
/// via [`density_focus`] — the same primitive the descent's density branch uses)
/// rather than the geometric window center: the field selects good *windows* but
/// their geometric center frequently lands in bland exterior, so a sharply-narrowed
/// start there dies at depth 1. Centering on the window's structure makes the field
/// proposal scale-stable so wide and narrow share one center (clean A-vs-B). When
/// `recenter` is false (native, `--root-start-fw 0`) the geometric window center is
/// used — byte-identical to prior runs.
#[allow(clippy::too_many_arguments)]
fn root_step_8k(
    windows: &[PassWindow],
    select_fw: f64,
    start_fw: f64,
    recenter: bool,
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    black_max: f64,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
    rng: &mut SplitMix64,
) -> StepResult {
    if windows.is_empty() || start_fw < FW_FLOOR {
        return StepResult::Died(EndCause::DegenerateExhausted);
    }
    for _ in 0..ROOT_MAX_TRIES {
        let wn = windows[rng.below(windows.len())];
        // The proposer's emitted center: geometric window center (native), or the
        // window's content focus rendered at the selection scale (decoupled mode).
        let center = if recenter {
            let sel = Frame {
                center: wn.center,
                frame_width: select_fw,
                out_width: node_w,
                out_height: node_h,
            };
            let sel_buf = render_node(&sel);
            density_focus(&sel, &sel_buf.samples, node_w as usize, node_h as usize)
        } else {
            wn.center
        };
        let frame = Frame { center, frame_width: start_fw, out_width: node_w, out_height: node_h };
        let buf = render_node(&frame);
        if render::black_fraction(&buf.samples) as f64 <= black_max {
            let (int_frac, _esc) = generate::screen_stats(&buf.samples, maxiter);
            return StepResult::Accepted(frame, buf, "root8k", "center", f64::NAN, int_frac, f64::NAN);
        }
    }
    StepResult::Died(EndCause::BlackCapExhausted)
}

/// Flat-sampler root (the prior `generate` method verbatim): draw a uniform-in-box
/// center × log-uniform shallow fw, cheap-screen it against the (real) `AcceptBand`,
/// and keep the first pass — that `(center, fw)` is the depth-1 start, with its own
/// sampled fw. The node is then rendered as the depth-1 parent buffer.
#[allow(clippy::too_many_arguments)]
fn root_step_flat(
    flat_box: (f64, f64, f64, f64),
    fw_lo: f64,
    fw_hi: f64,
    start_fw: Option<f64>,
    screen_w: u32,
    band: &crate::generate::AcceptBand,
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
    rng: &mut SplitMix64,
) -> StepResult {
    let (re_lo, re_hi, im_lo, im_hi) = flat_box;
    let (ln_lo, ln_hi) = (fw_lo.ln(), fw_hi.ln());
    let screen_h = (screen_w as f64 * node_h as f64 / node_w as f64).round().max(1.0) as u32;
    for _ in 0..ROOT_MAX_TRIES {
        // Three draws per candidate (re, im, scale) — same order as `generate`.
        let re = re_lo + rng.unit() * (re_hi - re_lo);
        let im = im_lo + rng.unit() * (im_hi - im_lo);
        let fw = (ln_lo + rng.unit() * (ln_hi - ln_lo)).exp();
        if fw < FW_FLOOR {
            continue;
        }
        let center = Complex::new(re, im);
        let screen = Frame { center, frame_width: fw, out_width: screen_w, out_height: screen_h };
        let sbuf = render_node(&screen);
        let (int_frac, esc) = generate::screen_stats(&sbuf.samples, maxiter);
        if !band.test(int_frac, esc.spread, esc.median).accepted {
            continue;
        }
        // Passed the screen at its native sampled `fw` (the flat proposer's center
        // selection). The depth-1 node starts at the decoupled `start_fw` if the
        // experiment overrides it, else the native sampled `fw`.
        let node_fw = start_fw.unwrap_or(fw);
        if node_fw < FW_FLOOR {
            continue;
        }
        let frame = Frame { center, frame_width: node_fw, out_width: node_w, out_height: node_h };
        let buf = render_node(&frame);
        let (node_int, _esc) = generate::screen_stats(&buf.samples, maxiter);
        return StepResult::Accepted(frame, buf, "rootflat", "center", f64::NAN, node_int, f64::NAN);
    }
    StepResult::Died(EndCause::DegenerateExhausted)
}

/// Julia root: the deterministic base-scale z-plane view (center 0 = the z→−z
/// symmetry point, width `root_fw`). No sampling — every walk shares this root and
/// decorrelates downstream via the stochastic per-node policy. Permissive by
/// construction (a base-scale Julia is well-formed at any in-set `c`); the depth-≥2
/// best-of-N screen does the tightening from there.
fn root_step_julia(
    root_fw: f64,
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
) -> StepResult {
    let frame = Frame {
        center: Complex::new(0.0, 0.0),
        frame_width: root_fw,
        out_width: node_w,
        out_height: node_h,
    };
    let buf = render_node(&frame);
    let (int_frac, _esc) = generate::screen_stats(&buf.samples, maxiter);
    StepResult::Accepted(frame, buf, "rootjulia", "center", f64::NAN, int_frac, f64::NAN)
}

/// Center-descend rung (`--julia-center`): a pure centered zoom step at the (0,0)
/// z-plane symmetry center. The caller has already applied the normal per-step zoom
/// ratio and the `--min-fw` floor to `fw`; this just renders the centered node and
/// emits it as an ordinary accepted candidate. No finder, no best-of-N, no placement
/// — the walk output is the same shape as a normal walk's, so downstream scoring and
/// harvest are mode-agnostic. Shares the Julia `render_node` (picks the kernel from
/// `julia_c`). Interior fraction is logged like the other roots; occupancy is NaN
/// (no best-of-N selection produced it).
fn center_step_julia(
    fw: f64,
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
) -> StepResult {
    let frame = Frame {
        center: Complex::new(0.0, 0.0),
        frame_width: fw,
        out_width: node_w,
        out_height: node_h,
    };
    let buf = render_node(&frame);
    let (int_frac, _esc) = generate::screen_stats(&buf.samples, maxiter);
    StepResult::Accepted(frame, buf, "center", "center", f64::NAN, int_frac, f64::NAN)
}

/// Injected root: pin the depth-1 node to an externally proposed `(cx, cy, fw)`
/// (`--seed-list`). No sampling, no gate — the proposer already selected the
/// location; the depth>=2 best-of-N screen does all tightening from here. Renders the
/// node so the finder has a parent buffer, exactly like the native roots. Permissive
/// by construction: a proposer seed into bland/set-dominated territory is *kept* at
/// depth 1 (and typically dies at depth 2), so a weak proposer honestly shows up as
/// low yield rather than being silently filtered.
fn root_step_injected(
    seed: (f64, f64, f64),
    node_w: u32,
    node_h: u32,
    maxiter: u32,
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
) -> StepResult {
    let (cx, cy, fw) = seed;
    let frame = Frame {
        center: Complex::new(cx, cy),
        frame_width: fw,
        out_width: node_w,
        out_height: node_h,
    };
    let buf = render_node(&frame);
    let (int_frac, _esc) = generate::screen_stats(&buf.samples, maxiter);
    StepResult::Accepted(frame, buf, "rootinjected", "center", f64::NAN, int_frac, f64::NAN)
}

/// Log-uniform draw in `[lo, hi]` (rev4 B4 per-step zoom jitter). `lo == hi` ⇒ `lo`.
fn sample_log_uniform(lo: f64, hi: f64, rng: &mut SplitMix64) -> f64 {
    if hi <= lo {
        lo
    } else {
        (lo.ln() + rng.unit() * (hi.ln() - lo.ln())).exp()
    }
}

// ===========================================================================
// Policy: target selection
// ===========================================================================

/// Pick the next descent target on `parent`, returning the complex focus point,
/// the branch that chose it, and (for the foci branch) the focus's sampling score.
///
/// rev4 grafts: foci candidates are value-ordered & distance-thresholded to a
/// spatially-spread set (B3, `foci_div_px`) before score-weighted sampling; the
/// `random` branch draws from the frame's near-boundary band rather than a uniform
/// interior point (B2, `random_boundary`).
#[allow(clippy::too_many_arguments)]
fn pick_target(
    parent: &Frame,
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
    sigmas: &[f64],
    (p_foci, p_density): (f64, f64),
    foci_div_px: f64,
    random_boundary: bool,
    rng: &mut SplitMix64,
) -> (Complex<f64>, Branch, f64) {
    let r = rng.unit();
    let mut branch = if r < p_foci {
        Branch::Foci
    } else if r < p_foci + p_density {
        Branch::Density
    } else {
        Branch::Random
    };

    if branch == Branch::Foci {
        let foci = find_foci(samples, w, h, sigmas);
        let spread = spread_foci(&foci, foci_div_px);
        if let Some(f) = sample_focus(&spread, rng) {
            return (pixel_to_complex(parent, f.px, f.py), Branch::Foci, f.score);
        }
        branch = Branch::Density; // foci empty → density fallthrough
    }

    match branch {
        Branch::Density => (density_focus(parent, samples, w, h), Branch::Density, f64::NAN),
        _ => {
            let pt = if random_boundary {
                frame_boundary_point(parent, samples, w, h, rng)
            } else {
                random_interior_point(parent, rng)
            };
            (pt, Branch::Random, f64::NAN)
        }
    }
}

/// rev4 B3: value-ordered, distance-thresholded suppression. `foci` arrive sorted
/// by score (desc); keep each only if it is ≥ `radius_px` from every higher-scoring
/// kept focus, so the densest-ridge peak stops dominating the score-weighted draw.
/// `radius_px ≤ 0` disables (returns the input order). Node pixels are square in the
/// plane, so field-px Euclidean distance is already aspect-correct.
fn spread_foci(foci: &[Focus], radius_px: f64) -> Vec<Focus> {
    if radius_px <= 0.0 || foci.len() < 2 {
        return foci.to_vec();
    }
    let r2 = radius_px * radius_px;
    let mut kept: Vec<Focus> = Vec::new();
    for &f in foci {
        if kept.iter().all(|k| (k.px - f.px).powi(2) + (k.py - f.py).powi(2) > r2) {
            kept.push(f);
        }
    }
    kept
}

/// rev4 B2: draw a uniform point from the frame's near-boundary band (exterior
/// pixels in the bottom [`BOUNDARY_DE_PCT`] of exterior DE). Falls back to a random
/// interior point if the frame has no usable boundary band (e.g. all-interior).
fn frame_boundary_point(
    parent: &Frame,
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
    rng: &mut SplitMix64,
) -> Complex<f64> {
    let band = build_boundary_band(samples, w, h, parent);
    match sample_boundary(&band, rng) {
        Some(c) => c,
        None => random_interior_point(parent, rng),
    }
}

/// Sample one focus weighted by its sampling score (peak × isolation). Returns
/// `None` when the list is empty or all scores are non-positive.
fn sample_focus(foci: &[Focus], rng: &mut SplitMix64) -> Option<Focus> {
    if foci.is_empty() {
        return None;
    }
    let total: f64 = foci.iter().map(|f| f.score.max(0.0)).sum();
    if !(total > 0.0) {
        return None;
    }
    let mut t = rng.unit() * total;
    for f in foci {
        t -= f.score.max(0.0);
        if t <= 0.0 {
            return Some(*f);
        }
    }
    Some(*foci.last().unwrap())
}

/// Density-optimal focus: the energy-weighted centroid of a default-shaded
/// `OCC_GX×OCC_GY` edge-energy grid over the parent frame, with the same void
/// guard `present`'s `content_focus` uses (centroid tile < `OCC_FLOOR` → snap to
/// peak tile). Reuses the parent's already-rendered samples (no re-render).
fn density_focus(
    parent: &Frame,
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
) -> Complex<f64> {
    // Shade the parent samples with the built-in default palette so tile_energy
    // is defined (occupancy is palette-invariant to <1%).
    let gate = crate::palette::builtin("default", false).expect("default palette");
    let params = color_params();
    let img = render::shade_and_downsample(
        samples, w as u32, h as u32, 1, &gate, &params, parent.pixel_size(),
    );
    let tiles = energy::tile_energy(&img, OCC_GX, OCC_GY);
    let (mut wsum, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut peak, mut peak_i) = (f64::NEG_INFINITY, 0usize);
    for ty in 0..OCC_GY {
        for tx in 0..OCC_GX {
            let e = tiles[ty * OCC_GX + tx];
            sx += e * (tx as f64 + 0.5) / OCC_GX as f64;
            sy += e * (ty as f64 + 0.5) / OCC_GY as f64;
            wsum += e;
            if e > peak {
                peak = e;
                peak_i = ty * OCC_GX + tx;
            }
        }
    }
    let (mut fx, mut fy) = if wsum > 0.0 { (sx / wsum, sy / wsum) } else { (0.5, 0.5) };
    let (ctx, cty) = (((fx * OCC_GX as f64) as usize).min(OCC_GX - 1), ((fy * OCC_GY as f64) as usize).min(OCC_GY - 1));
    if tiles[cty * OCC_GX + ctx] < OCC_FLOOR {
        fx = ((peak_i % OCC_GX) as f64 + 0.5) / OCC_GX as f64;
        fy = ((peak_i / OCC_GX) as f64 + 0.5) / OCC_GY as f64;
    }
    let fh = parent.frame_height();
    Complex::new(parent.center.re + (fx - 0.5) * parent.frame_width, parent.center.im + (0.5 - fy) * fh)
}

/// **PARKED — not for production.** Targeting a band of escape *value* is not a
/// diversity axis: it reaches the same morphology vocabulary at every distance-band
/// (verified null vs the random-survivor baseline). Kept flag-gated as the
/// finder/selection seam for future *structure*-targeting work (|∇field|/curvature/
/// winding), not to be selected on real runs. Do not delete.
///
/// Percentile-band finder (`--finder percentile`): draw one child-window center from
/// the parent frame's escaped smooth-iter percentile band `[quantile(lo), quantile(hi)]`,
/// window forced to CENTER (placement `center`). The child window's parent-space
/// rectangle (width `new_fw`, 16:9) must have interior fraction `< interior_cap`
/// (measured on the parent escaped mask) or the pixel is redrawn (up to `max_tries`).
/// Returns `None` on an empty band or all-tries-fail — falling through exactly like an
/// empty foci set. Consumes rng only (one `below` per redraw). Screened downstream by
/// the same best-of-N band/occupancy/black gates as the legacy finder.
#[allow(clippy::too_many_arguments)]
fn percentile_gen(
    parent: &Frame,
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
    new_fw: f64,
    (lo, hi): (f64, f64),
    interior_cap: f64,
    max_tries: usize,
    rng: &mut SplitMix64,
) -> Option<StepCand> {
    let n = w * h;
    if samples.len() < n || n == 0 {
        return None;
    }
    // Escaped smooth-iter values → percentile thresholds.
    let mut esc: Vec<f64> = samples[..n].iter().filter(|s| s.escaped).map(|s| s.smooth_iter).collect();
    if esc.len() < 16 {
        return None;
    }
    esc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| esc[((p * (esc.len() - 1) as f64) as usize).min(esc.len() - 1)];
    let (v_lo, v_hi) = (q(lo), q(hi));
    // Candidate pixel indices: escaped, μ inside the band.
    let band: Vec<usize> = (0..n)
        .filter(|&i| {
            let s = &samples[i];
            s.escaped && s.smooth_iter >= v_lo && s.smooth_iter <= v_hi
        })
        .collect();
    if band.is_empty() {
        return None;
    }
    // Child window footprint in parent pixels (parent pixels are square in-plane).
    let child_w_px = w as f64 * (new_fw / parent.frame_width);
    let (hw, hh) = (child_w_px * 0.5, child_w_px * 9.0 / 16.0 * 0.5);
    for _ in 0..max_tries.max(1) {
        let i = band[rng.below(band.len())];
        let (px, py) = ((i % w) as f64, (i / w) as f64);
        // Parent-space child rect (clamped to the frame); interior frac on the mask.
        let x0 = (px - hw).floor().max(0.0) as usize;
        let x1 = ((px + hw).ceil() as i64).clamp(0, w as i64 - 1) as usize;
        let y0 = (py - hh).floor().max(0.0) as usize;
        let y1 = ((py + hh).ceil() as i64).clamp(0, h as i64 - 1) as usize;
        let (mut interior, mut total) = (0usize, 0usize);
        for yy in y0..=y1 {
            let row = yy * w;
            for xx in x0..=x1 {
                total += 1;
                if !samples[row + xx].escaped {
                    interior += 1;
                }
            }
        }
        let int_frac = if total > 0 { interior as f64 / total as f64 } else { 1.0 };
        if int_frac < interior_cap {
            let center = pixel_to_complex(parent, px, py);
            return Some(StepCand { center, branch: "percentile", placement: "center", fscore: f64::NAN });
        }
    }
    None
}

/// A uniformly random interior point of the frame, ≥ 20 % from any edge.
fn random_interior_point(parent: &Frame, rng: &mut SplitMix64) -> Complex<f64> {
    let u = 0.2 + 0.6 * rng.unit();
    let v = 0.2 + 0.6 * rng.unit();
    let fh = parent.frame_height();
    Complex::new(parent.center.re + (u - 0.5) * parent.frame_width, parent.center.im + (0.5 - v) * fh)
}

/// Build a frame's near-boundary band: complex coords of exterior pixels whose DE
/// sits in the bottom [`BOUNDARY_DE_PCT`] of exterior DE (close to the set but not
/// interior). The rev4 B2 `random` branch draws uniformly from this band — uniform
/// over a small-DE band is already boundary-biased, so no extra DE weighting. (The
/// rev1–3 principal-feature exclusion list is retired with the boundary-sampler root.)
fn build_boundary_band(
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
    frame: &Frame,
) -> Vec<Complex<f64>> {
    // Exterior DE threshold (bottom quantile).
    let mut des: Vec<f64> =
        samples.iter().filter(|s| s.escaped && s.de > 0.0).map(|s| s.de).collect();
    if des.is_empty() {
        return Vec::new();
    }
    des.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let thresh = des[((BOUNDARY_DE_PCT * (des.len() - 1) as f64) as usize).min(des.len() - 1)];

    let mut band = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let s = &samples[y * w + x];
            if s.escaped && s.de > 0.0 && s.de <= thresh {
                band.push(pixel_to_complex(frame, x as f64, y as f64));
            }
        }
    }
    band
}

/// Draw one center uniformly from a near-boundary band.
fn sample_boundary(band: &[Complex<f64>], rng: &mut SplitMix64) -> Option<Complex<f64>> {
    if band.is_empty() {
        None
    } else {
        Some(band[rng.below(band.len())])
    }
}

/// Sample a placement from the (center, horizon, random) mixture.
fn pick_placement(weights: (f64, f64, f64), sum: f64, rng: &mut SplitMix64) -> Placement {
    let t = rng.unit() * sum;
    if t < weights.0 {
        Placement::Center
    } else if t < weights.0 + weights.1 {
        Placement::Horizon
    } else {
        Placement::Random
    }
}

/// Place `focus` inside the child frame per the placement mixture and return the
/// child's center. Center → focus at (0.5,0.5); horizon → focus at (u,0.5),
/// u∈[0.2,0.8]; random → focus at (u,v), u,v∈[0.2,0.8].
fn child_center(
    focus: Complex<f64>,
    placement: Placement,
    new_fw: f64,
    new_fh: f64,
    rng: &mut SplitMix64,
) -> Complex<f64> {
    match placement {
        Placement::Center => focus,
        Placement::Horizon => {
            let u = 0.2 + 0.6 * rng.unit();
            Complex::new(focus.re - (u - 0.5) * new_fw, focus.im)
        }
        Placement::Random => {
            let u = 0.2 + 0.6 * rng.unit();
            let v = 0.2 + 0.6 * rng.unit();
            // focus at fractional (u,v): re = center + (u-0.5)fw, im = center + (0.5-v)fh
            Complex::new(focus.re - (u - 0.5) * new_fw, focus.im - (0.5 - v) * new_fh)
        }
    }
}

/// Field pixel → complex. Pixel center at `(px+0.5, py+0.5)`; row 0 = top = max im.
fn pixel_to_complex(frame: &Frame, px: f64, py: f64) -> Complex<f64> {
    let w = frame.out_width as f64;
    let h = frame.out_height as f64;
    let fx = (px + 0.5) / w;
    let fy = (py + 0.5) / h;
    let fh = frame.frame_height();
    Complex::new(frame.center.re + (fx - 0.5) * frame.frame_width, frame.center.im + (0.5 - fy) * fh)
}

// ===========================================================================
// The focus finder (μ scale-space; built here — focus_diag is dump-only)
// ===========================================================================

/// Scale-space μ-foci of a cheap field: smooth the smooth-escape `mu` at each σ
/// (normalized convolution over the exterior), exclude pixels within ~σ of the
/// interior (minibrot-halo filter), take local maxima above a per-scale floor,
/// and merge across σ. Each focus scored by **peak-response × isolation** (NOT
/// persistence — persistence is reported but never used as the sampling weight).
fn find_foci(samples: &[crate::backend::PixelSample], w: usize, h: usize, sigmas: &[f64]) -> Vec<Focus> {
    let n = w * h;
    if samples.len() < n || n == 0 {
        return Vec::new();
    }
    // mu (0 where interior), validity mask, interior mask.
    let mut mu = vec![0.0f64; n];
    let mut valid = vec![0.0f64; n];
    let mut interior = vec![0.0f64; n];
    let mut ext_mu: Vec<f64> = Vec::new();
    for i in 0..n {
        let s = &samples[i];
        if s.escaped {
            mu[i] = s.smooth_iter;
            valid[i] = 1.0;
            ext_mu.push(s.smooth_iter);
        } else {
            interior[i] = 1.0;
        }
    }
    if ext_mu.len() < 16 {
        return Vec::new();
    }
    // Clip μ at the exterior p90 before smoothing: μ diverges at the boundary, so
    // un-clipped smoothed-μ is monotone toward the set (no interior maxima — the
    // very degeneracy the focus exploration clipped to avoid). Clipping saturates
    // the near-boundary band so isolated exterior spiral-eye ridges can win.
    ext_mu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mu_cap = ext_mu[((0.90 * (ext_mu.len() - 1) as f64) as usize).min(ext_mu.len() - 1)];
    for v in mu.iter_mut() {
        if *v > mu_cap {
            *v = mu_cap;
        }
    }

    // All per-scale detections: (px, py, sigma_idx, peak_norm, isolation).
    struct Det {
        x: usize,
        y: usize,
        si: usize,
        resp: f64,
        iso: f64,
    }
    let mut dets: Vec<Det> = Vec::new();

    for (si, &sigma) in sigmas.iter().enumerate() {
        let r = sigma.round().max(1.0) as usize;
        // Normalized convolution: smoothed = gauss(mu·valid) / gauss(valid).
        let num = gauss_approx(&mu, w, h, r);
        let den = gauss_approx(&valid, w, h, r);
        let mut sm = vec![f64::NAN; n];
        for i in 0..n {
            if den[i] > 1e-6 {
                sm[i] = num[i] / den[i];
            }
        }
        // Interior dilation by ~σ (exclude minibrot halos).
        let dil = dilate(&interior, w, h, r);
        // Local-mean field at 2σ for the isolation ratio.
        let sm0: Vec<f64> = sm.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
        let smvalid: Vec<f64> = sm.iter().map(|&v| if v.is_nan() { 0.0 } else { 1.0 }).collect();
        let lnum = gauss_approx(&sm0, w, h, 2 * r);
        let lden = gauss_approx(&smvalid, w, h, 2 * r);

        // Per-scale exterior stats (mean/std) + percentile floor.
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..n {
            if !sm[i].is_nan() && dil[i] == 0.0 {
                vals.push(sm[i]);
            }
        }
        if vals.len() < 8 {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        let std = var.sqrt().max(1e-9);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let floor = sorted[((MAX_FLOOR_PCT * (sorted.len() - 1) as f64) as usize).min(sorted.len() - 1)];

        // Local maxima (radius mr) above the floor, exterior, away from interior.
        let mr = (sigma * 0.4).round().max(2.0) as i64;
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let i = (y as usize) * w + x as usize;
                if sm[i].is_nan() || dil[i] != 0.0 || sm[i] < floor {
                    continue;
                }
                let v = sm[i];
                let mut is_max = true;
                // In-region max only: skip invalid/dilated-interior neighbors, else a
                // pixel just outside the exclusion ring always sees a higher excluded
                // neighbor toward the set and never wins.
                'nbr: for dy in -mr..=mr {
                    let yy = y + dy;
                    if yy < 0 || yy >= h as i64 {
                        continue;
                    }
                    for dx in -mr..=mr {
                        let xx = x + dx;
                        if xx < 0 || xx >= w as i64 || (dx == 0 && dy == 0) {
                            continue;
                        }
                        let j = (yy as usize) * w + xx as usize;
                        if sm[j].is_nan() || dil[j] != 0.0 {
                            continue;
                        }
                        if sm[j] > v {
                            is_max = false;
                            break 'nbr;
                        }
                    }
                }
                if !is_max {
                    continue;
                }
                let lm = if lden[i] > 1e-6 { lnum[i] / lden[i] } else { v };
                let iso = if lm.abs() > 1e-9 { v / lm } else { 1.0 };
                dets.push(Det {
                    x: x as usize,
                    y: y as usize,
                    si,
                    resp: (v - mean) / std,
                    iso,
                });
            }
        }
    }

    if dets.is_empty() {
        return Vec::new();
    }
    // Merge across scales: greedy by response, link by nearest location.
    dets.sort_by(|a, b| b.resp.partial_cmp(&a.resp).unwrap_or(std::cmp::Ordering::Equal));
    let mean_sigma = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
    let merge_r2 = (mean_sigma * 0.75).powi(2);
    let mut foci: Vec<(Focus, std::collections::HashSet<usize>)> = Vec::new();
    for d in &dets {
        let (dx, dy) = (d.x as f64, d.y as f64);
        let mut hit = None;
        for (k, (f, _)) in foci.iter().enumerate() {
            if (f.px - dx).powi(2) + (f.py - dy).powi(2) <= merge_r2 {
                hit = Some(k);
                break;
            }
        }
        match hit {
            Some(k) => {
                foci[k].1.insert(d.si);
                let np = foci[k].1.len() as u32;
                let f = &mut foci[k].0;
                f.persistence = np;
                if d.iso > f.isolation {
                    f.isolation = d.iso;
                }
                // peak_norm/location stay at the strongest detection (dets are sorted).
            }
            None => {
                let mut set = std::collections::HashSet::new();
                set.insert(d.si);
                foci.push((
                    Focus {
                        px: dx,
                        py: dy,
                        persistence: 1,
                        peak_norm: d.resp.max(0.0),
                        isolation: d.iso,
                        score: 0.0,
                    },
                    set,
                ));
            }
        }
    }
    let mut out: Vec<Focus> = foci
        .into_iter()
        .map(|(mut f, _)| {
            f.score = f.peak_norm.max(0.0) * f.isolation.max(0.0);
            f
        })
        .filter(|f| f.score > 0.0)
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(TOP_FOCI);
    out
}

/// Separable 3-box approximation of a Gaussian (σ ≈ box radius `r`), edge-clamped.
fn gauss_approx(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut a = box_blur_sep(src, w, h, r);
    a = box_blur_sep(&a, w, h, r);
    box_blur_sep(&a, w, h, r)
}

/// One separable box-mean pass (window `2r+1`), edge-clamped, via running sums.
fn box_blur_sep(src: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; w * h];
    let win = (2 * r + 1) as f64;
    // horizontal
    for y in 0..h {
        let row = y * w;
        let mut acc = 0.0;
        for k in 0..=r.min(w - 1) {
            acc += src[row + k];
        }
        // seed window [−r, r] with clamping: left clamps to src[row]
        acc += src[row] * r as f64; // the (-r..0) clamp contributions
        for x in 0..w {
            tmp[row + x] = acc / win;
            let add = (x + r + 1).min(w - 1);
            let sub = if x >= r { x - r } else { 0 };
            acc += src[row + add] - src[row + sub];
        }
    }
    let mut out = vec![0.0f64; w * h];
    // vertical
    for x in 0..w {
        let mut acc = 0.0;
        for k in 0..=r.min(h - 1) {
            acc += tmp[k * w + x];
        }
        acc += tmp[x] * r as f64;
        for y in 0..h {
            out[y * w + x] = acc / win;
            let add = (y + r + 1).min(h - 1);
            let sub = if y >= r { y - r } else { 0 };
            acc += tmp[add * w + x] - tmp[sub * w + x];
        }
    }
    out
}

/// Boolean dilation of `mask` (1.0 = set) by radius `r`: result `1.0` where any
/// set pixel lies within the `(2r+1)²` window. Separable max filter.
fn dilate(mask: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let mut m = 0.0;
            for k in lo..=hi {
                if mask[row + k] > m {
                    m = mask[row + k];
                }
            }
            tmp[row + x] = m;
        }
    }
    let mut out = vec![0.0f64; w * h];
    for x in 0..w {
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let mut m = 0.0;
            for k in lo..=hi {
                if tmp[k * w + x] > m {
                    m = tmp[k * w + x];
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

// ===========================================================================
// Output helpers
// ===========================================================================

/// Summary of the per-step chosen interior fractions (the min-interior selection's
/// drift check: does it pull walks toward the empty/thin floor?).
struct InteriorSummary {
    n: usize,
    min: f64,
    p25: f64,
    med: f64,
    p75: f64,
    max: f64,
    mean: f64,
}

/// Five-number + mean summary of the chosen interior fractions (empty → all zero).
fn interior_summary(xs: &[f64]) -> InteriorSummary {
    if xs.is_empty() {
        return InteriorSummary { n: 0, min: 0.0, p25: 0.0, med: 0.0, p75: 0.0, max: 0.0, mean: 0.0 };
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| s[((p * (s.len() - 1) as f64).round() as usize).min(s.len() - 1)];
    InteriorSummary {
        n: s.len(),
        min: s[0],
        p25: q(0.25),
        med: q(0.5),
        p75: q(0.75),
        max: s[s.len() - 1],
        mean: s.iter().sum::<f64>() / s.len() as f64,
    }
}

/// Std-dev of depth-1 root-window centers in (re, im) — a coarse read on whether
/// the boundary sampler spread walks across the fw-3.0 view or piled up.
fn root_spread(roots: &[&Candidate]) -> (f64, f64) {
    let n = roots.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let (mx, my) = (
        roots.iter().map(|c| c.cx).sum::<f64>() / n,
        roots.iter().map(|c| c.cy).sum::<f64>() / n,
    );
    let vx = roots.iter().map(|c| (c.cx - mx).powi(2)).sum::<f64>() / n;
    let vy = roots.iter().map(|c| (c.cy - my).powi(2)).sum::<f64>() / n;
    (vx.sqrt(), vy.sqrt())
}

/// Most-repeated emitted frame and unique-frame count. Frames are keyed by center
/// + fw rounded to ~1/10 of their own width, so near-identical centers (the run0
/// shared-root repetition signature) collapse to one key.
fn repetition(cands: &[Candidate]) -> (usize, usize) {
    use std::collections::HashMap;
    let mut counts: HashMap<(i64, i64, i64), usize> = HashMap::new();
    for c in cands {
        let q = (c.fw * 0.1).abs().max(1e-300); // ~1/10 of frame width
        let key = (
            (c.cx / q).round() as i64,
            (c.cy / q).round() as i64,
            c.fw.log10().round() as i64,
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    (top, counts.len())
}

fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

/// Burn provenance onto a flat-grid thumbnail.
fn annotate(th: &mut RgbImage, c: &Candidate) {
    let white = Rgb([240u8, 240, 240]);
    crate::font::draw_text(th, &format!("w{} d{}/{}", c.walk, c.depth, c.target_depth), 2, 2, 1, white, true);
    let tag = if c.focus_score.is_finite() {
        format!("{} {:.1}", c.branch, c.focus_score)
    } else {
        format!("{} {}", c.branch, c.placement)
    };
    crate::font::draw_text(th, &tag, 2, 12, 1, white, true);
}

/// The candidate-pool sheet: header stats + a by-walk ladder view + a flat grid.
#[allow(clippy::too_many_arguments)]
fn build_html(
    cands: &[Candidate],
    walk_reached: &[(u32, u32)],
    walk_root_src: &[&'static str],
    branch_counts: &[usize; 3],
    root8k_count: usize,
    rootflat_count: usize,
    died_early: usize,
    cause_counts: &[usize; 5],
    black_rejects: usize,
    black_cap_on: bool,
    black_cap: f64,
    occ_rejects: usize,
    occ_on: bool,
    occ_floor: f64,
    ci: &InteriorSummary,
    args: &GuidedDescendArgs,
    n_walks: usize,
    sigmas: &[f64],
    zoom_lo: f64,
    zoom_hi: f64,
) -> String {
    let n_8k = walk_root_src.iter().filter(|s| **s == "8k").count();
    let n_flat = walk_root_src.iter().filter(|s| **s == "flat").count();
    let roots: Vec<&Candidate> = cands.iter().filter(|c| c.depth == 1).collect();
    let (rsx, rsy) = root_spread(&roots);
    let (top_mult, n_unique) = repetition(cands);
    // depth histogram
    let mut depth_hist = vec![0usize; (args.depth_max + 1) as usize];
    for c in cands {
        depth_hist[c.depth as usize] += 1;
    }
    let mut dh = String::new();
    for (d, n) in depth_hist.iter().enumerate().skip(1) {
        let _ = write!(dh, "d{d}:{n} ");
    }

    // by-walk ladders
    let mut nwalks = 0usize;
    let mut ladders = String::new();
    let mut w = 0usize;
    while w < n_walks {
        let row: Vec<&Candidate> = cands.iter().filter(|c| c.walk == w).collect();
        if !row.is_empty() {
            nwalks += 1;
            let (reached, target) = walk_reached[w];
            let died = if reached < target { " <span class=died>DIED</span>" } else { "" };
            let src = walk_root_src.get(w).copied().unwrap_or("");
            let _ = write!(
                ladders,
                "<div class=ladder><div class=lmeta>walk {w} · root {src} · reached {reached}/{target}{died}</div><div class=lrow>"
            );
            for c in &row {
                let _ = write!(
                    ladders,
                    "<figure><img loading=lazy src=\"{}\"><figcaption>d{} {} {}{}</figcaption></figure>",
                    c.png, c.depth, c.branch, c.placement,
                    if c.focus_score.is_finite() { format!(" {:.1}", c.focus_score) } else { String::new() },
                );
            }
            let _ = write!(ladders, "</div></div>");
        }
        w += 1;
    }

    // flat pool grid
    let mut flat = String::new();
    for c in cands {
        let fs = if c.focus_score.is_finite() { format!(" fs{:.1}", c.focus_score) } else { String::new() };
        let _ = write!(
            flat,
            "<div class=cell><img loading=lazy src=\"{}\"><div class=cap>w{} d{}/{} <b>{}</b> {}{}</div></div>",
            c.png, c.walk, c.depth, c.target_depth, c.branch, c.placement, fs,
        );
    }

    format!(
        "<!doctype html><html><head><meta charset=utf-8><title>guided-descend pool</title>\
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}\
body{{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;margin:0}}\
header{{position:sticky;top:0;background:#12141a;border-bottom:1px solid #23252e;padding:10px 18px;z-index:5}}\
h1{{font-size:15px;margin:0 0 4px;color:#eee}}h2{{font-size:13px;color:#e0b24a;margin:18px 14px 6px}}\
.note{{color:#9aa;font-size:12px}}\
.ladder{{border-bottom:1px solid #1c1f29;padding:6px 14px}}\
.lmeta{{color:#9aa;margin-bottom:3px}}.died{{color:#e06a6a;font-weight:bold}}\
.lrow{{display:flex;gap:6px;overflow-x:auto}}\
.lrow figure{{margin:0;flex:0 0 auto;width:200px;border:1px solid #23252e;border-radius:4px;overflow:hidden;background:#000}}\
.lrow img{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}}\
figcaption{{padding:2px 5px;font-size:10px;color:#9aa;background:#12141a}}\
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;padding:14px}}\
.cell{{border:1px solid #23252e;border-radius:4px;overflow:hidden;background:#000}}\
.cell img{{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}}\
.cap{{padding:2px 6px;font-size:10px;color:#9aa;background:#12141a}}\
.cap b{{color:#5ec07a}}\
</style></head><body>\
<header><h1>guided-descend rev4 — candidate pool ({ncand} candidates)</h1>\
<div class=note>{nwalks} walks emitting · best-of-{ncand_n} · \
root-mix target {rmix:.2}: 8k {n8k}/{total} vs flat {nflat}/{total} walks · \
branch root8k={r8} rootflat={rf} foci={bf} density={bd} random={br} (target {pf:.2}/{pd:.2}/{pr:.2}) · \
depth: {dh}· walks died early {died}/{total} · \
end-cause: terminal={ct} black-cap={cb} occ-floor={co} degenerate={cd} min-fw-floor={cmf} · \
best-of-N rejects: black-cap {blk} ×{brk}, occ-floor {olk} ×{ork} · \
chosen interior (drift check): med={cimed:.3} [{cimin:.3}..{cimax:.3}] p25/p75 {cip25:.3}/{cip75:.3} · \
preview {pal} · 8k root-zoom {rz8} · zoom/step jitter [{zlo},{zhi}] · \
root-window std (re,im)=({rsx:.3},{rsy:.3}) · unique frames {nuniq}/{ncand} max-rep ×{tmult} · \
σ {sig:?} · seed {seed} · gates on black/interior + occupancy ONLY (no busyness axis) · \
<b>no quality claims</b> — eyeball the sampling behaviour</div></header>\
<h2>by walk — descent ladders (root→leaf left→right)</h2>{ladders}\
<h2>flat pool ({ncand})</h2><div class=grid>{flat}</div>\
</body></html>",
        ncand = cands.len(),
        ncand_n = args.descent_candidates.max(1),
        nwalks = nwalks,
        rmix = args.root_mix.clamp(0.0, 1.0),
        n8k = n_8k,
        nflat = n_flat,
        r8 = root8k_count,
        rf = rootflat_count,
        bf = branch_counts[0],
        bd = branch_counts[1],
        br = branch_counts[2],
        pf = args.w_foci.max(0.0) / (args.w_foci.max(0.0) + args.w_density.max(0.0) + args.w_random.max(0.0)),
        pd = args.w_density.max(0.0) / (args.w_foci.max(0.0) + args.w_density.max(0.0) + args.w_random.max(0.0)),
        pr = args.w_random.max(0.0) / (args.w_foci.max(0.0) + args.w_density.max(0.0) + args.w_random.max(0.0)),
        dh = dh,
        died = died_early,
        total = n_walks,
        ct = cause_counts[0],
        cb = cause_counts[1],
        co = cause_counts[2],
        cd = cause_counts[3],
        cmf = cause_counts[4],
        brk = black_rejects,
        blk = if black_cap_on { format!("<{black_cap}") } else { "OFF".into() },
        ork = occ_rejects,
        olk = if occ_on { format!(">={occ_floor}") } else { "OFF".into() },
        cimed = ci.med,
        cimin = ci.min,
        cimax = ci.max,
        cip25 = ci.p25,
        cip75 = ci.p75,
        pal = args.preview_palette,
        rz8 = args.root_zoom_8k,
        zlo = zoom_lo,
        zhi = zoom_hi,
        rsx = rsx,
        rsy = rsy,
        nuniq = n_unique,
        tmult = top_mult,
        sig = sigmas,
        seed = args.seed,
        ladders = ladders,
        flat = flat,
    )
}


// ===== Args structs relocated from cli.rs (P0 cli decomposition) =====

/// Parameter-plane escape family the walker descends on: Mandelbrot (`z²+c`) or a
/// multibrot (`z^d+c`, `d ∈ {3,4,5}`). Mandelbrot is the default and is byte-
/// identical to prior runs. Multibrot is parameter-plane only (the fixed-`c`
/// dynamical Julia mode is a separate `--julia` axis and is incompatible with it).
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum WalkFamily {
    /// `z ← z² + c` (degree 2).
    Mandelbrot,
    /// `z ← z³ + c`.
    Multibrot3,
    /// `z ← z⁴ + c`.
    Multibrot4,
    /// `z ← z⁵ + c`.
    Multibrot5,
}

impl WalkFamily {
    /// Escape recurrence degree `d`.
    pub fn degree(self) -> u32 {
        match self {
            WalkFamily::Mandelbrot => 2,
            WalkFamily::Multibrot3 => 3,
            WalkFamily::Multibrot4 => 4,
            WalkFamily::Multibrot5 => 5,
        }
    }

    /// Per-family 8k-root window band `(mean_lo, mean_hi, var_floor)` — the
    /// smooth-iteration criterion applied by [`crate::root_field::RootField::score`].
    /// `z^d` (d≥3) escapes faster than `z²`, so the escaped smooth-iter statistics
    /// drift with degree and the Mandelbrot values mis-fire (a d3 seed passing the
    /// Mandelbrot band lands on set-thicket / dead-exterior windows that can't
    /// descend). Values are the *default* for each family, still flag-overridable.
    ///
    /// - **d2 (Mandelbrot):** the exact historical values — byte-identical to every
    ///   prior run.
    /// - **d3 (multibrot3):** tuned by eye against the d3 8k field
    ///   (`prompts/cc-d3-band-tuning`). `mean_lo` drops 8→6 (`z³`'s escaped mean is
    ///   ~25% lower); `var_floor` rises 6→20 (`z³`'s boundary escape is sharper, so
    ///   window variance runs *higher* — the floor climbs to hold the same ~9%
    ///   window-pass selectivity the Mandelbrot band had). `mean_hi` stays 120 (an
    ///   inactive upper backstop; the d3 mean never reaches it).
    /// - **d4 (multibrot4):** `mean_lo` 8→7.2, `var_floor` 6→209 — the projected
    ///   `z⁴` shift confirmed by eye against the d4 8k field. `var_floor` explodes
    ///   with degree (`z⁴`'s boundary escape is sharper still than `z³`, so window
    ///   variance runs an order of magnitude higher); `mean_lo` barely moves.
    /// - **d5 (multibrot5):** `mean_lo` 8→8.4, `var_floor` 6→502 — same picture, an
    ///   even steeper `var_floor`. (For d5 the escaped mean drifts slightly *up*, so
    ///   `mean_lo` rises rather than falls.)
    /// - **mean_hi** stays 120 for every family (inactive upper backstop).
    pub fn root8k_band_defaults(self) -> (f64, f64, f64) {
        match self {
            WalkFamily::Multibrot3 => (6.0, 120.0, 20.0),
            WalkFamily::Multibrot4 => (7.2, 120.0, 209.0),
            WalkFamily::Multibrot5 => (8.4, 120.0, 502.0),
            // d2 (byte-identical Mandelbrot).
            WalkFamily::Mandelbrot => (8.0, 120.0, 6.0),
        }
    }

    /// Per-family flat-sampler / descent spread floor (the `AcceptBand::spread_min`
    /// clause, middle-90% smooth-iter spread `p95−p5`). This gate is read twice: the
    /// flat-sampler depth-1 root screen, and — via `descent_band` — the degenerate
    /// "flat" cull at every descent step. `z³`'s compressed smooth-iter range yields
    /// a ~0.8× smaller frame spread, so the Mandelbrot floor of 20 over-rejects d3
    /// (structured d3 frames read as "flat") and starves descents. d3 → 15. d4/d5
    /// recover toward the historical floor (d4 → 16, d5 → 17); d2 stays at 20.
    pub fn flat_spread_min_default(self) -> f64 {
        match self {
            WalkFamily::Multibrot3 => 15.0,
            WalkFamily::Multibrot4 => 16.0,
            WalkFamily::Multibrot5 => 17.0,
            WalkFamily::Mandelbrot => 20.0,
        }
    }

    /// Julia-specific descent band defaults `(esc_median_min, flat_spread_min)` — the
    /// loosened, degree-aware bands applied to **dynamical Julia descents only**
    /// (`--julia`, NOT Phoenix and NOT the parameter plane). Promoted from the
    /// `--gather` Julia-only overrides after the overnight harvest validated them by
    /// yield (`julia:mandelbrot` 222/288 q3, 287/288 guard-pass; multibrot-Julia
    /// families q3-rich). The c-plane `z^d` descent escapes such that a shared
    /// Mandelbrot band (`esc_median_min = 3.0`, the per-family `flat_spread_min`)
    /// starves higher-degree Julia sets: real filled/dendritic `z^d` Julia structure
    /// reads as instant-escape / flat and is culled. These values sit below the
    /// assessment's per-degree real-set medians / lower tails, so genuine Julia
    /// structure passes while true far-exterior instant-escape is still excluded.
    /// The c-plane keeps its calibrated `root8k_band_defaults` / `flat_spread_min_default`
    /// — this table is Julia-only. (Mirror of `JULIA_GATHER_BANDS` in
    /// `tools/atlas/production_seeder.py`.)
    ///
    /// - **d2 (quadratic Julia):** `esc_median_min` 3.0 (unchanged), `flat_spread_min`
    ///   20 → 14.
    /// - **d3 (Julia-multibrot3):** `esc` 3.0 → 2.0, `flat` 15 → 10.
    /// - **d4 (Julia-multibrot4):** `esc` 3.0 → 2.0, `flat` 16 → 13.
    /// - **d5 (Julia-multibrot5):** `esc` 3.0 → 1.8, `flat` 17 → 13.
    pub fn julia_band_defaults(self) -> (f64, f64) {
        match self {
            WalkFamily::Mandelbrot => (3.0, 14.0),
            WalkFamily::Multibrot3 => (2.0, 10.0),
            WalkFamily::Multibrot4 => (2.0, 13.0),
            WalkFamily::Multibrot5 => (1.8, 13.0),
        }
    }

    /// Per-family flat-sampler root box `(re_lo, re_hi, im_lo, im_hi)`. The flat
    /// sampler draws uniform centers in this box; the Mandelbrot default is the
    /// historical asymmetric cardioid frame, but multibrot sets (`d ≥ 3`) are
    /// origin-symmetric, so an asymmetric cardioid box wastes half its draws on dead
    /// exterior. For `d ≥ 3` use the same origin-centered square the 8k root field
    /// uses — half-width `2^(1/(d−1))·1.2` (see [`crate::root_field::degree_bbox`]),
    /// so the flat arm and 8k arm frame the identical region. d2 keeps its exact
    /// historical box (byte-identical).
    pub fn flat_box_default(self) -> (f64, f64, f64, f64) {
        match self {
            WalkFamily::Mandelbrot => (-2.0, 0.7, -1.2, 1.2),
            _ => crate::root_field::degree_bbox(self.degree()),
        }
    }
}

/// Which next-center generator the descent uses at depth ≥ 2 (ablation seam).
/// `Legacy` = the settled foci/density/random policy (`pick_target` + placement);
/// `Percentile` = the smooth-iter percentile-band finder (draw a pixel from an
/// escaped-μ quantile band, window forced center) — **PARKED**, not for production
/// use (escape-value banding is not a diversity axis; verified null vs the
/// random-survivor baseline). Default `Legacy` is byte-identical to prior runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FinderMode {
    Legacy,
    Percentile,
}

/// How `best_of_n_step` picks the winner among full survivors (ablation seam).
/// `RandomSurvivor` (the shipped default) = a uniform draw among the survivors via
/// reservoir sampling on the walk rng; `LeastInterior` (opt-in via `--selection`) =
/// the min-interior-fraction objective (draws NO rng — byte-identical to prior runs).
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SelectionMode {
    LeastInterior,
    RandomSurvivor,
}

/// `dump-julia-bands` subcommand: see `guided_descend::run_dump_julia_bands`.
/// Read-only; prints the per-family Julia descent band table as JSON. No render.
#[derive(Args, Debug)]
pub struct DumpJuliaBandsArgs {}

/// Emit the canonical per-family Julia band table (`(esc_median_min, spread_min)`
/// from [`WalkFamily::julia_band_defaults`]) as JSON to stdout, keyed by the
/// `production_seeder.py` partition name. This is the single source of truth for the
/// Rust↔Python parity guard (`tools/atlas/check_julia_bands.py`), which compares
/// this output against `JULIA_GATHER_BANDS` — the guard never parses Rust source.
/// Runtime render behavior is unchanged; this just prints what the engine already
/// computes.
pub fn run_dump_julia_bands(_args: &DumpJuliaBandsArgs) -> Result<(), String> {
    // (partition key as used in tools/atlas/production_seeder.py, family).
    let table = [
        ("mandelbrot", WalkFamily::Mandelbrot),
        ("multibrot3", WalkFamily::Multibrot3),
        ("multibrot4", WalkFamily::Multibrot4),
        ("multibrot5", WalkFamily::Multibrot5),
    ];
    let mut out = String::from("{\n");
    for (i, (key, fam)) in table.iter().enumerate() {
        let (esc, spread) = fam.julia_band_defaults();
        let comma = if i + 1 < table.len() { "," } else { "" };
        let _ = writeln!(out, "  \"{key}\": [{esc}, {spread}]{comma}");
    }
    out.push_str("}\n");
    print!("{out}");
    Ok(())
}

/// `guided-descend` subcommand: see `guided_descend::run_guided_descend`.
/// Stochastic guided descent from the fixed base-Mandelbrot root; geometric
/// policy only (no CNN). Reuses the `generate` cheap screen + `AcceptBand` and a
/// freshly-built μ scale-space focus finder.
#[derive(Args, Debug)]
pub struct GuidedDescendArgs {
    /// Number of independent, decorrelated walks (each starts at the root).
    #[arg(long, default_value_t = 80)]
    pub n_walks: usize,

    /// Minimum terminal walk depth (inclusive). rev3: raised 3→4 (the shallow
    /// d3 frames were too zoomed-out / set-dominated to be useful wallpapers).
    #[arg(long, default_value_t = 4)]
    pub depth_min: u32,

    /// Maximum terminal walk depth (inclusive). Raised 10→17 so the deepest-
    /// starting roots can descend meaningfully deep: walks step ~0.4×/step, so from a
    /// ~0.003 root depth 10 only reaches ~5e-7 and the depth cap bound the walk long
    /// before the `--min-fw` f64 floor (1e-9) ever would.
    #[arg(long, default_value_t = 17)]
    pub depth_max: u32,

    /// f64-reliable descent floor on the frame width (bring-up insurance). A walk is
    /// truncated — and its already-collected candidates harvested — before any step
    /// whose frame width would fall below this, keeping walks clear of the untested
    /// forced-F64/perturbation regime at extreme zoom. Default `1e-9` matches the
    /// `FW_FLOOR` root-start floor, so descent and root floors sit at the same
    /// known-good f64 cutoff — well above the ~1e-13 f64 cliff and far below anything
    /// current walks reach, so it should fire near-zero times (the count is surfaced in
    /// `summary.json` / `min_fw_floor`). Tunable/liftable later. (The separate
    /// `FW_FLOOR` constant still guards the depth-1 root start widths.)
    #[arg(long, default_value_t = 1e-9)]
    pub min_fw: f64,

    /// LEGACY fixed per-step zoom (rev1–3). rev4 B4 samples the per-step zoom from
    /// a log-uniform band `[--zoom-lo, --zoom-hi]` instead; this value is reported
    /// as the nominal but no longer drives stepping. Set `--zoom-lo`=`--zoom-hi`=x
    /// to ablate B4 back to a fixed x.
    #[arg(long, default_value_t = 0.4)]
    pub zoom_per_step: f64,

    /// rev4 B4: per-step zoom-jitter band, low edge. Each depth-≥2 step draws its
    /// zoom log-uniform in `[zoom_lo, zoom_hi]` (default [0.35,0.50], centered near
    /// the old fixed 0.4) — restores the flat method's scale diversity.
    #[arg(long, default_value_t = 0.35)]
    pub zoom_lo: f64,

    /// rev4 B4: per-step zoom-jitter band, high edge.
    #[arg(long, default_value_t = 0.50)]
    pub zoom_hi: f64,

    /// LEGACY (rev1–3 boundary-sampler root, retired in rev4). The root is now a
    /// 50/50 sampler mixture (Part A); the 8k root uses `--root-zoom-8k` and the
    /// flat root uses its own sampled fw. Kept only so old invocations still parse.
    #[arg(long, default_value_t = 0.08)]
    pub root_zoom: f64,

    // --- rev4 Part A: root sampler mixture ------------------------------------
    /// `P(8k-field root)` vs `P(flat-sampler root)` at depth-1 (rev4 Part A). 1.0 =
    /// always the 8k field; 0.0 = always the flat sampler. Both emit a permissive
    /// (≤80% black) depth-1 node, then hand to the same per-node descent loop.
    #[arg(long, default_value_t = 0.5)]
    pub root_mix: f64,

    /// 8k-field root depth-1 frame width (rev4 A1, significant — past the old 0.24,
    /// folds in "push past base"). At 0.10 the 8k field gives a ~234px window.
    #[arg(long, default_value_t = 0.10)]
    pub root_zoom_8k: f64,

    /// 8k-root window criterion: max interior (black) fraction (rev4 A1, permissive;
    /// subsumes the retired 5-feature exclusion list).
    #[arg(long, default_value_t = 0.80)]
    pub root8k_black_max: f64,

    /// 8k-root window criterion: escaped smooth-iter mean lower bound (rejects empty
    /// far-exterior). Stats over escaped pixels only. Unset ⇒ the per-family default
    /// (`WalkFamily::root8k_band_defaults`): d2=8.0, d3=6.0, d4=7.2, d5=8.4 (`z^d`
    /// escape drifts the mean; the shift is small — `var_floor` is the live knob).
    #[arg(long)]
    pub root8k_mean_lo: Option<f64>,

    /// 8k-root window criterion: escaped smooth-iter mean upper bound (rejects
    /// set-dominated thickets). Unset ⇒ per-family default (all families = 120.0, an
    /// inactive upper backstop).
    #[arg(long)]
    pub root8k_mean_hi: Option<f64>,

    /// 8k-root window criterion: escaped smooth-iter variance floor (not-flat). Unset
    /// ⇒ per-family default: d2=6.0, d3=20.0, d4=209.0, d5=502.0. The dominant
    /// per-degree knob — `z^d`'s boundary escape sharpens with degree, so window
    /// variance explodes and the floor rises steeply to hold selectivity.
    #[arg(long)]
    pub root8k_var_floor: Option<f64>,

    // --- rev4 Part A2: flat-sampler root (the prior `generate` method) ---------
    /// Flat-root center box `re_lo,re_hi,im_lo,im_hi` (rev4 A2 — reuses `generate`'s
    /// uniform-in-plane box verbatim). Unset ⇒ the per-family default
    /// (`WalkFamily::flat_box_default`): d2 = the historical cardioid frame
    /// `-2.0,0.7,-1.2,1.2`; d≥3 = the origin-square root box (half-width
    /// `2^(1/(d−1))·1.2`), so a multibrot walk needs no explicit `--flat-box`.
    #[arg(long = "flat-box", allow_hyphen_values = true)]
    pub flat_box: Option<String>,

    /// Flat-root log-uniform fw range, low edge (reuses `generate`'s shallow range).
    #[arg(long, default_value_t = 0.003)]
    pub flat_fw_lo: f64,

    /// Flat-root log-uniform fw range, high edge.
    #[arg(long, default_value_t = 0.05)]
    pub flat_fw_hi: f64,

    /// Flat-root cheap-screen render width (height 16:9, ss1) — reuses `generate`'s
    /// screen + `AcceptBand`.
    #[arg(long, default_value_t = 320)]
    pub flat_screen_width: u32,

    /// Target-policy weight: descend into a detected μ-focus (renormalized).
    /// rev4 B1: lowered 0.85→0.70 (raise measure-uniformity via `random`).
    #[arg(long, default_value_t = 0.70)]
    pub w_foci: f64,

    /// Target-policy weight: descend into the energy-weighted density focus.
    #[arg(long, default_value_t = 0.10)]
    pub w_density: f64,

    /// Target-policy weight: descend into a near-boundary-band point (rev4 B2 —
    /// the per-step measure-uniformity injection). rev4 B1: raised 0.05→0.20.
    #[arg(long, default_value_t = 0.20)]
    pub w_random: f64,

    /// Placement mixture `center,horizon,random` (where the chosen target lands in
    /// the child frame). Applied to the foci/density branches; the random branch
    /// is always centered. Lowered center vs run0 (was 0.50,0.25,0.25) to
    /// decorrelate the shallow layers — this is the repetition dial.
    #[arg(long, default_value = "0.25,0.40,0.35")]
    pub placement: String,

    /// Focus-finder σ band in field px (comma-separated). Persistence is measured
    /// within this band; the sampling score is peak×isolation (NOT persistence).
    #[arg(long, default_value = "16,20,24,28,32")]
    pub sigma_band: String,

    /// rev4 B3: foci-diversity radius as a fraction of the node frame width. Before
    /// score-weighted sampling, foci are value-ordered and distance-thresholded (a
    /// focus within this radius of an already-kept, higher-scoring focus is dropped)
    /// so the densest-ridge peak stops dominating every step. 0 disables (ablate B3
    /// → plain top-score sampling).
    #[arg(long, default_value_t = 0.12)]
    pub foci_diversity_radius: f64,

    /// rev4 B2: random branch draws from the current frame's near-boundary band
    /// (exterior pixels near the set) instead of a uniform interior point. Default
    /// on; `--random-boundary=false` ablates back to the rev3 interior point.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub random_boundary: bool,

    /// Cheap field/screen render width in px (height follows 16:9, ss1). Drives
    /// both the AcceptBand screen and the μ-field the focus finder reads. Default
    /// 768 so the σ band {16..32} runs at the frame-fraction it was tuned at
    /// (run0 ran 256 → σ ~3× too coarse). Alias `--node-size`.
    #[arg(long, alias = "node-size", default_value_t = 768)]
    pub node_width: u32,

    /// Preview render width in px (height follows 16:9, ss1).
    #[arg(long, default_value_t = 640)]
    pub preview_width: u32,

    /// Preview colormap name (from the colormaps JSON) — diagnostic only.
    /// `twilight_shifted` is cyclic (seam-free, mirror-moot).
    #[arg(long, default_value = "twilight_shifted")]
    pub preview_palette: String,

    /// Colormap library JSON (for the preview palette).
    #[arg(long, default_value = "data/palettes/clean_colormaps.json")]
    pub colormaps: String,

    /// Maximum iterations for the field/screen and preview renders.
    #[arg(long, default_value_t = 1000)]
    pub maxiter: u32,

    /// Escape radius. Large (1e6) for smooth-coloring accuracy.
    #[arg(long, default_value_t = 1e6)]
    pub bailout: f64,

    /// Accept-band override: middle-90% smooth-iter spread floor.
    #[arg(long)]
    pub spread_min: Option<f64>,

    /// Accept-band override: interior (max-iter) fraction cap.
    #[arg(long)]
    pub interior_max: Option<f64>,

    /// Accept-band override: escape-median smooth-iter floor.
    #[arg(long)]
    pub esc_median_min: Option<f64>,

    /// Best-of-N candidate count per step (rev3 Change 2). Each step draws this
    /// many candidate next-centers from the per-node policy and two-stage screens
    /// them (cheap interior cap → 768 occupancy floor) before selecting the
    /// least-set survivor. 1 reproduces rev2's accept-first behaviour.
    #[arg(long, default_value_t = 4)]
    pub descent_candidates: usize,

    /// Best-of-N **Stage 1** interior/black cap (rev3, lowered 0.45→0.30 — the
    /// aggressive set-avoidance hard ceiling). Each candidate is probed at ~128px
    /// escape-time and rejected if its `render::black_fraction` (interior counts as
    /// black) is ≥ this; interior fraction is scale-robust so the cheap probe is
    /// fine. Default-on at 0.30; 0 or ≥1.0 disables. Black/interior-fraction ONLY
    /// (busyness is known-unseparable by magnitude).
    #[arg(long, default_value_t = 0.30)]
    pub descent_black_cap: f64,

    /// Best-of-N **Stage 2** occupancy floor (rev3, reuses present's 0.321). Stage-1
    /// survivors are rendered at the 768 node size, shaded, and scored with the
    /// `energy::occupancy` parity scorer; candidates below this are rejected (keeps
    /// walks feature-rich, not empty). Among the rest the **least interior** wins.
    /// Default-on at 0.321; 0 or ≥1.0 disables. CAVEAT: 0.321 was calibrated on the
    /// labeling-crop distribution, not navigation frames — a tunable knob.
    #[arg(long, default_value_t = 0.321)]
    pub descent_occ_floor: f64,

    /// Apply the Stage-2 occupancy floor at the **depth-1→2** step too. The occ
    /// floor over-fires at that first descent transition (12 of 17 run4 early
    /// deaths were occ-floor kills at d1→d2 — the wide depth-1 root frame is still
    /// resolving structure the tight depth-2 node hasn't entered yet). **Default
    /// off:** the occ floor is skipped at d1→d2 and kept for every d≥3 step. Pass
    /// this flag to restore the legacy behaviour (occ floor at every depth). The
    /// Stage-1 interior cap and least-interior selection are unaffected at every
    /// depth; the *presentation* occ floor (`present --occupancy-floor`) is a
    /// separate, correctly-calibrated gate and is untouched by this flag.
    #[arg(long, default_value_t = false)]
    pub descent_occ_at_d1d2: bool,

    /// Decoupled-start override (2x2 scale experiment seam). When > 0, the depth-1
    /// node starts descent at exactly this frame width for BOTH proposers, while
    /// each proposer's native center-SELECTION scale is held fixed (the 8k field
    /// still scans windows at `--root-zoom-8k`; the flat sampler still screens at
    /// its own log-uniform sampled fw). This separates "which center" from "at what
    /// start scale" so the field-vs-flat spatial selection and the wide-vs-narrow
    /// start scale are independent factors. 0 (default) = coupled/native behaviour,
    /// byte-identical to prior runs. Affects ONLY the depth-1 root node; every
    /// depth>=2 descent/sampler/gate step is unchanged.
    #[arg(long, default_value_t = 0.0)]
    pub root_start_fw: f64,

    /// Parameter-plane escape family: `mandelbrot` (default, `z²+c`) or
    /// `multibrot3|multibrot4|multibrot5` (`z^d+c`). Multibrot descends on the
    /// same geometric policy / screen / best-of-N machinery with the recurrence and
    /// per-degree root box swapped in. Each degree carries **per-family band / box
    /// defaults** tuned by eye (`z^d` escapes faster, so a shared Mandelbrot band
    /// mis-fires): see `WalkFamily::root8k_band_defaults` / `flat_spread_min_default`
    /// / `flat_box_default`. Under `--julia`, `--family multibrot{d}` instead descends
    /// the **dynamical** Julia-multibrot z^d plane (z-plane root, no c-plane apparatus).
    #[arg(long, value_enum, default_value_t = WalkFamily::Mandelbrot)]
    pub family: WalkFamily,

    // --- Julia descent mode (z-plane descent at fixed c) ----------------------
    /// Descend in the **z-plane** at a fixed parameter `c` instead of the c-plane.
    /// The descent geometry (foci / density / placement / best-of-N screen) is
    /// fractal-agnostic and reused verbatim; only the root step changes (deterministic
    /// base-scale z-plane view, not the 8k/flat c-plane samplers) and the DE-dependent
    /// boundary branch is gated off (the dynamical kernels carry no DE). Pairs with
    /// `--family mandelbrot` (quadratic Julia) or `--family multibrot{d}`
    /// (**Julia-multibrot**, dynamical `z^d+c`). Requires `--c`; `--c` without
    /// `--julia`/`--phoenix` is an error.
    #[arg(long, default_value_t = false)]
    pub julia: bool,

    /// Fixed dynamical parameter `c` as two arbitrary-precision decimal strings:
    /// `--c <re> <im>` (e.g. `--c -0.8 0.156`). Required iff `--julia` (the Julia /
    /// Julia-multibrot parameter); under `--phoenix` it is the additive constant
    /// (optional, defaults to the classic `0.5667 0`). Parsed and f64-projected
    /// exactly like `render-one`'s `--c`.
    #[arg(long = "c", num_args = 2, value_names = ["RE", "IM"], allow_hyphen_values = true)]
    pub julia_c: Option<Vec<String>>,

    /// **Phoenix** dynamical z-plane descent (Ushiki `z_{n+1} = z_n² + c + p·z_{n-1}`,
    /// `z₀ = pixel`, `z_{-1} = 0`). Descends the z-plane like `--julia` (same
    /// fractal-agnostic policy + base-scale root at center 0), with the two-state
    /// recurrence swapped in. `--c` (additive const) and `--p` (z_{n-1} coeff) both
    /// optional, defaulting to the classic real-valued spot. Mutually exclusive with
    /// `--julia`; incompatible with `--family multibrot*` (Phoenix is degree 2).
    #[arg(long, default_value_t = false)]
    pub phoenix: bool,

    /// Phoenix second constant `p` (the `z_{n-1}` coefficient / Ushiki's `q`) as two
    /// decimal strings `--p <re> <im>`. Valid only with `--phoenix`; defaults to the
    /// classic `-0.5 0`.
    #[arg(long = "p", num_args = 2, value_names = ["RE", "IM"], allow_hyphen_values = true)]
    pub phoenix_p: Option<Vec<String>>,

    /// Julia/Phoenix z-plane root frame width (the depth-1 base-scale view, centered
    /// at 0). Descent then leaves this center.
    #[arg(long, default_value_t = 3.0)]
    pub julia_root_fw: f64,

    /// **Center-descend** the Julia z-plane (valid only with `--julia`): every rung
    /// stays pinned at the (0,0) symmetry center, shrinking `fw` from `--julia-root-fw`
    /// by the normal per-step zoom ratio — no foci, no best-of-N, no content search, a
    /// straight centered zoom. Rung count emerges from the same depth cap / `--min-fw`
    /// bounds a normal walk uses; the walk output shape is identical so downstream
    /// scoring/harvest is mode-agnostic. Default off = ordinary off-center descend.
    #[arg(long, default_value_t = false)]
    pub julia_center: bool,

    /// Flat-grid PNG columns.
    #[arg(long, default_value_t = 10)]
    pub cols: usize,

    /// SplitMix64 seed (deterministic).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Reseed each walk's RNG deterministically from `(seed, walk_index)` at the top
    /// of the walk (instead of drawing every walk from one shared global stream).
    /// This makes each walk's depth-1 seed a function of ONLY `(seed, walk_index)` —
    /// independent of how many RNG draws prior walks consumed. That is required for
    /// paired cross-configuration studies (e.g. the descent-resolution efficiency
    /// study): the per-step focus-finder consumes a *resolution-dependent* number of
    /// draws (a Foci step draws 2 when foci are found, 1 when none — `sample_focus`
    /// returns early without drawing on an empty list), so under the shared stream
    /// only walk 0 keeps an identical depth-1 seed across resolutions; every later
    /// walk desyncs. With this flag the depth-1 seeds are bit-identical across
    /// configurations and the walks then legitimately diverge at depth≥2 (the thing
    /// under test). **Default off** — a shared-stream run is byte-identical to prior
    /// runs. The per-walk seed = `SplitMix64(seed).next` advanced `walk_index+1`
    /// times, i.e. a distinct, well-mixed sub-seed per walk.
    #[arg(long, default_value_t = false)]
    pub per_walk_rng: bool,

    /// Injected depth-1 seed list (atlas round-1 acceptance harness). A JSONL file
    /// with one `{"cx":..,"cy":..,"fw":..}` object per line (extra keys ignored). When
    /// set, the internal 8k/flat root draw is bypassed: walk `w` pins its depth-1 frame
    /// to row `w`, `--n-walks` is overridden by the list length, and the root step
    /// consumes NO rng — so multiple runs differing only in their seed list (or vs a
    /// native run's own seeds re-injected) share one depth>=2 rng stream and are
    /// directly comparable. Every depth>=2 sampler/screen/gate is unchanged. Pair with
    /// `--per-walk-rng`. Mandelbrot only (incompatible with `--julia`).
    #[arg(long)]
    pub seed_list: Option<String>,

    // --- Descent-ablation seam (finder / selection / percentile-band) ---------
    /// Depth-≥2 next-center finder: `legacy` (foci/density/random policy, default,
    /// byte-identical to prior runs) or `percentile` (smooth-iter percentile-band
    /// finder — window centered on a pixel drawn from an escaped-μ quantile band).
    /// **`percentile` is PARKED** (escape-value banding is not a diversity axis —
    /// verified null vs the random-survivor baseline); the flag is retained as the
    /// finder seam for future structure-targeting work, not for production use.
    /// The percentile finder ignores the foci/density/random weights, `--sigma-band`,
    /// `--foci-diversity-radius`, `--random-boundary`, and placement (window forced
    /// center); it is screened by the same best-of-N band/occupancy/black gates.
    #[arg(long, value_enum, default_value_t = FinderMode::Legacy)]
    pub finder: FinderMode,

    /// Convenience override of the normalized foci/density/random policy mix as
    /// `f,d,r` (e.g. `0.10,0.20,0.70`). When set, replaces `--w-foci/--w-density/
    /// --w-random`. **Legacy-finder ablation knob** (the percentile finder has no
    /// policy mix). Unset ⇒ the individual `--w-*` flags apply.
    #[arg(long)]
    pub branch_weights: Option<String>,

    /// Best-of-N winner objective among full survivors: `random-survivor` (default,
    /// uniform among survivors via reservoir sampling on the walk rng) or
    /// `least-interior` (the pre-ship default, retained opt-in: min interior fraction
    /// — draws NO rng, byte-identical to prior runs). Reject/EndCause logic is
    /// identical either way.
    #[arg(long, value_enum, default_value_t = SelectionMode::RandomSurvivor)]
    pub selection: SelectionMode,

    /// Percentile finder band `lo,hi` (quantiles over escaped smooth-iter). The
    /// candidate pixel set is the escaped pixels whose μ falls in `[quantile(lo),
    /// quantile(hi)]`; a pixel is drawn uniformly from it and the child window is
    /// centered there. Default `0.60,0.80`. (All `--pct-*` flags only bite under
    /// `--finder percentile`, which is **PARKED** — see `--finder`.)
    #[arg(long, default_value = "0.60,0.80")]
    pub pct_band: String,

    /// Percentile finder interior cap: the child window's parent-space rectangle must
    /// have interior fraction `< cap` (measured on the parent mask) or the pixel is
    /// redrawn. Default 0.30 (matches the Stage-1 black cap spirit).
    #[arg(long, default_value_t = 0.30)]
    pub pct_interior_cap: f64,

    /// Percentile finder max redraw attempts before a single `gen` call returns None
    /// (falls through like an empty foci set). Default 32.
    #[arg(long, default_value_t = 32)]
    pub pct_max_tries: usize,

    /// Optional percentile band schedule: piecewise-constant by depth, as
    /// `d0:lo0,hi0;d1:lo1,hi1;...` (the band for the largest breakpoint depth ≤ the
    /// current descent depth wins). Unset ⇒ the constant `--pct-band`.
    #[arg(long)]
    pub pct_band_schedule: Option<String>,

    /// Output directory (`pool_sheet.html`, `pool.jsonl`, `pool_grid.png`,
    /// `tiles/`). Outside `out/` — durable. Use a distinct dir per run.
    #[arg(long, default_value = "data/guided_descend/run4")]
    pub out_dir: String,
}

impl GuidedDescendArgs {
    /// Effective accept band (each clause flag-overridable; shared default).
    ///
    /// Band defaults are **descent-mode-aware**. A dynamical **Julia** descent
    /// (`--julia`, NOT Phoenix) resolves its `esc_median_min` / `spread_min` from the
    /// loosened, degree-aware [`WalkFamily::julia_band_defaults`] table — the `z^d`
    /// Julia plane escapes such that the c-plane Mandelbrot/multibrot floors starve
    /// real filled/dendritic Julia structure. Parameter-plane descents (and Phoenix,
    /// which is degree-2 and descends cleanly on the Mandelbrot defaults) keep the
    /// calibrated `esc_median_min = 3.0` + per-family `flat_spread_min_default`. In
    /// every mode an explicit CLI `--esc-median-min` / `--spread-min` still wins.
    pub fn band(&self) -> crate::generate::AcceptBand {
        let d = crate::generate::AcceptBand::default();
        // Julia-only loosened defaults (`--julia`; Phoenix stays on the c-plane
        // defaults). d2 esc default == d.esc_median_min (3.0), so only the spread
        // floor moves for quadratic Julia.
        let (esc_default, spread_default) = if self.julia {
            self.family.julia_band_defaults()
        } else {
            (d.esc_median_min, self.family.flat_spread_min_default())
        };
        crate::generate::AcceptBand {
            // spread_min drifts with degree (`z^d` compresses the smooth-iter range);
            // resolve against the per-mode default. c-plane d2 default == d.spread_min,
            // so parameter-plane Mandelbrot stays byte-identical.
            spread_min: self.spread_min.unwrap_or(spread_default),
            interior_max: self.interior_max.unwrap_or(d.interior_max),
            esc_median_min: self.esc_median_min.unwrap_or(esc_default),
        }
    }

    /// Decoupled depth-1 start fw (2x2 seam). `> 0` ⇒ `Some(fw)` (force both
    /// proposers to start descent at `fw`); `0` ⇒ `None` (native coupled scale).
    pub fn resolved_root_start_fw(&self) -> Result<Option<f64>, String> {
        if self.root_start_fw == 0.0 {
            Ok(None)
        } else if self.root_start_fw > 0.0 && self.root_start_fw.is_finite() {
            Ok(Some(self.root_start_fw))
        } else {
            Err(format!("--root-start-fw must be 0 (native) or > 0 (got {})", self.root_start_fw))
        }
    }

    /// Parse `--placement` (`center,horizon,random`) raw weights (un-normalized).
    pub fn resolved_placement(&self) -> Result<(f64, f64, f64), String> {
        let p: Vec<&str> = self.placement.split(',').collect();
        if p.len() != 3 {
            return Err(format!("invalid --placement '{}', expected center,horizon,random", self.placement));
        }
        let parse = |s: &str| -> Result<f64, String> {
            s.trim().parse::<f64>().map_err(|_| format!("invalid --placement value '{}'", s.trim()))
        };
        let (c, h, r) = (parse(p[0])?, parse(p[1])?, parse(p[2])?);
        if c < 0.0 || h < 0.0 || r < 0.0 || c + h + r <= 0.0 {
            return Err("--placement weights must be non-negative and sum > 0".into());
        }
        Ok((c, h, r))
    }

    /// Parse `--flat-box` (`re_lo,re_hi,im_lo,im_hi`) for the flat-sampler root.
    /// Unset ⇒ the per-family default box ([`WalkFamily::flat_box_default`]).
    pub fn resolved_flat_box(&self) -> Result<(f64, f64, f64, f64), String> {
        let spec = match &self.flat_box {
            Some(s) => s,
            None => return Ok(self.family.flat_box_default()),
        };
        let p: Vec<&str> = spec.split(',').collect();
        if p.len() != 4 {
            return Err(format!("invalid --flat-box '{spec}', expected re_lo,re_hi,im_lo,im_hi"));
        }
        let parse = |s: &str, what: &str| -> Result<f64, String> {
            s.trim().parse().map_err(|_| format!("invalid --flat-box {what} in '{spec}'"))
        };
        let (re_lo, re_hi, im_lo, im_hi) =
            (parse(p[0], "re_lo")?, parse(p[1], "re_hi")?, parse(p[2], "im_lo")?, parse(p[3], "im_hi")?);
        if re_hi <= re_lo || im_hi <= im_lo {
            return Err(format!("--flat-box bounds must be lo < hi in '{spec}'"));
        }
        Ok((re_lo, re_hi, im_lo, im_hi))
    }

    /// The 8k-root window score config (the hand criterion's tunables).
    pub fn root8k_score_cfg(&self) -> crate::root_field::ScoreCfg {
        // Per-family band defaults (d2 byte-identical), each clause flag-overridable.
        let (mean_lo, mean_hi, var_floor) = self.family.root8k_band_defaults();
        crate::root_field::ScoreCfg {
            black_max: self.root8k_black_max,
            mean_lo: self.root8k_mean_lo.unwrap_or(mean_lo),
            mean_hi: self.root8k_mean_hi.unwrap_or(mean_hi),
            var_floor: self.root8k_var_floor.unwrap_or(var_floor),
        }
    }

    /// Resolved per-step zoom-jitter band `(lo, hi)` (rev4 B4). `lo==hi` ⇒ fixed.
    pub fn resolved_zoom_band(&self) -> Result<(f64, f64), String> {
        if self.zoom_lo <= 0.0 || self.zoom_hi <= 0.0 || self.zoom_hi < self.zoom_lo {
            return Err(format!(
                "need 0 < zoom_lo <= zoom_hi (got {}, {})",
                self.zoom_lo, self.zoom_hi
            ));
        }
        Ok((self.zoom_lo, self.zoom_hi))
    }

    /// Parse `--sigma-band` (comma-separated px) into ascending σ values.
    pub fn resolved_sigmas(&self) -> Result<Vec<f64>, String> {
        let mut v: Vec<f64> = Vec::new();
        for s in self.sigma_band.split(',') {
            let x: f64 = s.trim().parse().map_err(|_| format!("invalid --sigma-band value '{}'", s.trim()))?;
            if x <= 0.0 {
                return Err(format!("--sigma-band values must be > 0 (got {x})"));
            }
            v.push(x);
        }
        if v.is_empty() {
            return Err("--sigma-band is empty".into());
        }
        Ok(v)
    }

    /// Resolved policy weights `(foci, density, random)` (un-normalized). `--branch-weights
    /// f,d,r` overrides the individual `--w-*` flags when set (legacy finder only).
    pub fn resolved_branch_weights(&self) -> Result<(f64, f64, f64), String> {
        match &self.branch_weights {
            None => Ok((self.w_foci, self.w_density, self.w_random)),
            Some(s) => {
                let p: Vec<&str> = s.split(',').collect();
                if p.len() != 3 {
                    return Err(format!("invalid --branch-weights '{s}', expected f,d,r"));
                }
                let parse = |x: &str| -> Result<f64, String> {
                    x.trim().parse::<f64>().map_err(|_| format!("invalid --branch-weights value '{}'", x.trim()))
                };
                let (f, d, r) = (parse(p[0])?, parse(p[1])?, parse(p[2])?);
                if f < 0.0 || d < 0.0 || r < 0.0 || f + d + r <= 0.0 {
                    return Err("--branch-weights must be non-negative and sum > 0".into());
                }
                Ok((f, d, r))
            }
        }
    }

    /// Parse a `lo,hi` percentile-band spec into two quantiles in `[0,1]`, lo < hi.
    fn parse_pct_pair(spec: &str) -> Result<(f64, f64), String> {
        let p: Vec<&str> = spec.split(',').collect();
        if p.len() != 2 {
            return Err(format!("invalid percentile band '{spec}', expected lo,hi"));
        }
        let parse = |x: &str| -> Result<f64, String> {
            x.trim().parse::<f64>().map_err(|_| format!("invalid percentile value '{}'", x.trim()))
        };
        let (lo, hi) = (parse(p[0])?, parse(p[1])?);
        if !(0.0..=1.0).contains(&lo) || !(0.0..=1.0).contains(&hi) || hi <= lo {
            return Err(format!("percentile band '{spec}' must satisfy 0<=lo<hi<=1"));
        }
        Ok((lo, hi))
    }

    /// Resolved constant percentile band `(lo, hi)` from `--pct-band`.
    pub fn resolved_pct_band(&self) -> Result<(f64, f64), String> {
        Self::parse_pct_pair(&self.pct_band)
    }

    /// Resolved percentile band schedule as ascending `(depth, lo, hi)` breakpoints.
    /// Empty ⇒ constant band (`--pct-band` used directly). Format:
    /// `d0:lo0,hi0;d1:lo1,hi1;...`.
    pub fn resolved_pct_schedule(&self) -> Result<Vec<(u32, f64, f64)>, String> {
        let Some(spec) = &self.pct_band_schedule else { return Ok(Vec::new()) };
        let mut out: Vec<(u32, f64, f64)> = Vec::new();
        for seg in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let (d, band) = seg
                .split_once(':')
                .ok_or_else(|| format!("invalid --pct-band-schedule segment '{seg}', expected d:lo,hi"))?;
            let depth: u32 = d.trim().parse().map_err(|_| format!("invalid schedule depth '{}'", d.trim()))?;
            let (lo, hi) = Self::parse_pct_pair(band.trim())?;
            out.push((depth, lo, hi));
        }
        if out.is_empty() {
            return Err("--pct-band-schedule parsed to no breakpoints".into());
        }
        out.sort_by_key(|(d, _, _)| *d);
        Ok(out)
    }
}

/// The percentile band for descent depth `d`: the schedule breakpoint with the
/// largest depth ≤ `d` (or the first breakpoint if `d` precedes all), else the
/// constant `fallback` when the schedule is empty.
fn pct_band_for_depth(schedule: &[(u32, f64, f64)], fallback: (f64, f64), d: u32) -> (f64, f64) {
    if schedule.is_empty() {
        return fallback;
    }
    let mut band = (schedule[0].1, schedule[0].2);
    for &(bd, lo, hi) in schedule {
        if bd <= d {
            band = (lo, hi);
        } else {
            break;
        }
    }
    band
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chosen-child occupancy emitted by `best_of_n_step` (the value that lands
    /// in `walks.jsonl`'s `child_occ` and `pool.jsonl`'s `occ`) must equal the
    /// admission-time occupancy — i.e. the `energy::occupancy` of the winner's node,
    /// shaded with the same gate palette/params. This exercises the **occ-off** path
    /// (the d1→d2 descendability probe, previously logged null), so it proves the
    /// emit is not a reimplementation but the primitive's own value on the winner's
    /// already-rendered buffer.
    #[test]
    fn chosen_child_occ_equals_admission_occupancy() {
        let node_w = 256u32;
        let node_h = (node_w as f64 * 9.0 / 16.0).round() as u32;
        let maxiter = 300u32;
        let bailout = 4.0f64;
        let trap = Trap { shape: TrapShape::Point, center: Complex::new(0.0, 0.0), radius: 1.0 };
        // Shallow, deterministic Julia render (no RNG, no perturbation) so the node
        // is reproducible and its occupancy well-defined.
        let julia_c = Complex::new(-0.8, 0.156);
        let render_node = |frame: &Frame| -> render::SampleBuffer {
            let backend = JuliaBackend::new(julia_c, maxiter, bailout, trap);
            render::iterate_samples(&backend, frame, 1)
        };

        let gate_palette = crate::palette::builtin("default", false).expect("default palette");
        let params = color_params();
        // Descent band: interior clause off (as in the real run), keep the degenerate culls.
        let band = crate::generate::AcceptBand { interior_max: 1.0, ..Default::default() };
        // Probe-like screen: occ floor OFF (the d1→d2 case), black cap OFF.
        let cfg = StepScreen {
            n_cand: 1,
            node_w,
            node_h,
            maxiter,
            band: &band,
            black_cap: 0.0,
            black_cap_on: false,
            occ_floor: 0.321,
            occ_on: false,
            gate_palette: &gate_palette,
            params: &params,
        };

        let new_fw = 3.0f64;
        let mut gen = |_rng: &mut SplitMix64| -> Option<StepCand> {
            Some(StepCand {
                center: Complex::new(0.0, 0.0),
                branch: "random",
                placement: "center",
                fscore: f64::NAN,
            })
        };
        let mut black_rejects = 0usize;
        let mut occ_rejects = 0usize;
        let mut sel_rng = SplitMix64(0);
        let result = best_of_n_step(
            &cfg, new_fw, SelectionMode::LeastInterior, &render_node, &mut gen, &mut sel_rng,
            &mut black_rejects, &mut occ_rejects,
        );

        let (frame, emitted_occ) = match result {
            StepResult::Accepted(f, _b, _br, _pl, _fs, _ifr, occ) => (f, occ),
            StepResult::Died(cause) => panic!("known-good Julia node was rejected: {}", cause.name()),
        };
        assert!(emitted_occ.is_finite(), "emitted occ must be finite, got {emitted_occ}");
        assert!(emitted_occ > 0.0, "structured Julia node should have nonzero occupancy");

        // Independently recompute the admission-time occupancy on a fresh render of
        // the winner frame — must match the emitted value bit-for-bit (same primitive).
        let buf = render_node(&frame);
        let img = render::shade_and_downsample(
            &buf.samples, node_w, node_h, 1, &gate_palette, &params, frame.pixel_size(),
        );
        let recomputed = energy::occupancy(&img, OCC_GX, OCC_GY, OCC_FLOOR);
        assert_eq!(
            emitted_occ, recomputed,
            "emitted child occupancy ({emitted_occ}) must equal admission-time energy::occupancy ({recomputed})"
        );
    }
}
