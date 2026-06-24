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
//! Root step is **special-cased** (rev1): foci at base scale are degenerate (the
//! μ-finder has nothing to lock onto on the whole set), so the root→depth-1 jump
//! is a **boundary-window sampler** instead — a high-res base field (`center
//! (-0.5, 0)`, `fw 3.0`) rendered once, its near-boundary band (small-DE exterior
//! pixels) taken, principal features excluded, and the depth-1 window center
//! drawn from that band and placed at frame center with its own `--root-zoom`.
//! Subsequent steps (depth ≥ 2) use the per-node finder + placement mixture.
//! Decorrelation comes from the RNG stream; the base field is shared.

use std::fmt::Write as _;
use std::path::Path;

use astro_float::BigFloat;
use image::{Rgb, RgbImage};
use num_complex::Complex;
use rayon::prelude::*;

use crate::backend::{Trap, TrapShape};
use crate::cli::{BackendChoice, GuidedDescendArgs};
use crate::energy::{self, OCC_FLOOR, OCC_GX, OCC_GY};
use crate::generate::{self, color_params};
use crate::palette::Palette;
use crate::probe::{self, SplitMix64};
use crate::render::{self, Frame};
use crate::{hp, sheet};

/// f64-safety sanity guard on the descended frame width. Depth ≤ 6 at 0.4×/step
/// keeps `fw` ≈ 0.012 — nowhere near this — so it only ever fires on a
/// misconfigured deep run. No real deep-zoom handling (that is the search's job).
const FW_FLOOR: f64 = 1e-7;

/// Per-scale local-maxima percentile floor (over the exterior smoothed field):
/// a candidate maximum must sit above this quantile of its scale to count.
const MAX_FLOOR_PCT: f64 = 0.85;

/// Cap on foci returned per frame (top by sampling score).
const TOP_FOCI: usize = 16;

/// Stage-1 cheap interior-screen render width (height 16:9, ss1). Small + fast:
/// interior fraction is scale-robust, so a ~128px escape-time panel is enough to
/// reject set-dominated candidates before paying for the 768 node render.
const PROBE_W: u32 = 128;

/// Base-field render width for the root boundary sampler (height 16:9). A few
/// thousand px — enough to resolve a clean near-boundary band. Rendered once,
/// shared across all walks; shallow f64 (fw 3.0) so it is cheap.
const BASE_FIELD_WIDTH: u32 = 2048;

/// Near-boundary band cut: exterior pixels whose DE sits in the bottom this
/// quantile of exterior DE (small DE ⇒ close to the set, not interior). Traces a
/// thin boundary band the root window is sampled from.
const BOUNDARY_DE_PCT: f64 = 0.12;

/// Known principal features excluded from root sampling (Matt wants fresh
/// regions, not the clichés): main-cardioid cusp, period-2 cusp / seahorse neck,
/// west antenna tip, the two period-3 bulb cusps. **Starting point — refine after
/// seeing the pool.**
const EXCLUDE_FEATURES: &[(f64, f64)] = &[
    (0.25, 0.0),     // main-cardioid cusp
    (-0.75, 0.0),    // period-2 cusp / seahorse neck
    (-2.0, 0.0),     // west antenna tip
    (-0.125, 0.74),  // period-3 bulb cusp (upper)
    (-0.125, -0.74), // period-3 bulb cusp (lower)
];

/// Plane-unit radius of the exclusion disk around each principal feature.
const EXCLUDE_R: f64 = 0.12;

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
    /// and its interior fraction (the selection key, logged for the drift check).
    Accepted(Frame, render::SampleBuffer, &'static str, &'static str, f64, f64),
    /// No survivor across the N draws; the binding constraint.
    Died(EndCause),
}

/// One emitted candidate frame (a frame visited along a walk).
struct Candidate {
    idx: usize,
    walk: usize,
    depth: u32,
    target_depth: u32,
    branch: &'static str,
    placement: &'static str,
    /// Sampling score of the focus that produced this frame (NaN for density/random).
    focus_score: f64,
    cx: f64,
    cy: f64,
    fw: f64,
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
    let w_foci = args.w_foci.max(0.0);
    let w_density = args.w_density.max(0.0);
    let w_random = args.w_random.max(0.0);
    let wsum = w_foci + w_density + w_random;
    if wsum <= 0.0 {
        return Err("target weights sum to zero".into());
    }
    let (p_foci, p_density) = (w_foci / wsum, w_density / wsum); // random = remainder
    let (pl_center, pl_horizon, pl_random) = args.resolved_placement()?;
    let plsum = pl_center + pl_horizon + pl_random;
    let sigmas = args.resolved_sigmas()?;
    let band = args.band();

    let node_w = args.node_width.max(16);
    let node_h = (node_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
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

    // The fixed root: full Mandelbrot view (conceptual parent of every walk; its
    // fw seeds the root jump). Per-node renders use `node_w`.
    let root = Frame {
        center: Complex::new(-0.5, 0.0),
        frame_width: 3.0,
        out_width: node_w,
        out_height: node_h,
    };
    eprintln!(
        "guided-descend (rev3): {} walks, depth [{},{}], zoom/step {}, root-zoom {}, seed {}\n  \
         weights foci/density/random = {:.2}/{:.2}/{:.2}, placement {:.2}/{:.2}/{:.2}, \
         sigma {:?}\n  node {}x{} ss1, preview {}x{} ({}), maxiter {}\n  \
         descent culls: flat spread>={} + instant-escape esc_med>={}\n  \
         best-of-{}: Stage-1 interior-cap {} (probe {}px), Stage-2 occ-floor {} (@{}px), select min-interior",
        args.n_walks, args.depth_min, args.depth_max, args.zoom_per_step, args.root_zoom, args.seed,
        p_foci, p_density, 1.0 - p_foci - p_density,
        pl_center / plsum, pl_horizon / plsum, pl_random / plsum, sigmas,
        node_w, node_h, prev_w, prev_h, args.preview_palette, args.maxiter,
        band.spread_min, band.esc_median_min,
        args.descent_candidates.max(1),
        if black_cap_on { format!("black_frac<{black_cap}") } else { "OFF".into() }, PROBE_W,
        if occ_on { format!("occ>={occ_floor}") } else { "OFF".into() }, node_w,
    );

    let render_node = |frame: &Frame| -> render::SampleBuffer {
        let prec = hp::prec_bits(frame.out_width, frame.frame_width);
        let cre = BigFloat::from_f64(frame.center.re, prec);
        let cim = BigFloat::from_f64(frame.center.im, prec);
        probe::render_mandel_panel(
            &cre, &cim, frame.center, frame.frame_width, frame.out_width, frame.out_height, 1,
            args.maxiter, args.bailout, prec, trap, BackendChoice::F64,
        )
        .buf
    };

    let t0 = std::time::Instant::now();
    // --- root boundary band: render the base field once at high res, take the
    //     near-boundary (small-DE exterior) band, exclude principal features. ---
    let base_w = BASE_FIELD_WIDTH;
    let base_h = (base_w as f64 * 9.0 / 16.0).round().max(1.0) as u32;
    let base_frame = Frame {
        center: root.center,
        frame_width: root.frame_width,
        out_width: base_w,
        out_height: base_h,
    };
    let base_buf = render_node(&base_frame);
    let boundary = build_boundary_band(&base_buf.samples, base_w as usize, base_h as usize, &base_frame);
    eprintln!(
        "  base field {}x{} + boundary band in {:.2}s ({} band px after exclusion)",
        base_w, base_h, t0.elapsed().as_secs_f64(), boundary.len()
    );
    if boundary.is_empty() {
        return Err("root boundary band empty (DE cut + exclusion left nothing)".into());
    }

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

    let mut rng = SplitMix64(args.seed);
    let mut cands: Vec<Candidate> = Vec::new();
    // Per-walk reached depth + intended target (for the ladder view + died-early count).
    let mut walk_reached: Vec<(u32, u32)> = Vec::with_capacity(args.n_walks);
    let mut branch_counts = [0usize; 3]; // foci, density, random
    let mut root_count = 0usize; // depth-1 boundary-sampled steps
    let mut died_early = 0usize;
    // End-of-walk cause: [ReachedTerminalDepth, BlackCapExhausted, OccFloorExhausted, DegenerateExhausted].
    let mut cause_counts = [0usize; 4];
    let mut black_rejects = 0usize; // total best-of-N candidates killed by the Stage-1 black cap
    let mut occ_rejects = 0usize; // total best-of-N candidates killed by the Stage-2 occupancy floor
    // Per-step chosen interior fraction (the drift check — does min-interior pull toward empty?).
    let mut chosen_interiors: Vec<f64> = Vec::new();

    for w in 0..args.n_walks {
        let target = args.depth_min + (rng.below((args.depth_max - args.depth_min + 1) as usize) as u32);
        let mut parent = root;
        // `None` until the depth-1 root step renders the first node.
        let mut parent_buf: Option<render::SampleBuffer> = None;
        let mut reached = 0u32;
        // Walk completed unless a step dies; the dying step overwrites this.
        let mut end_cause = EndCause::ReachedTerminalDepth;

        for d in 1..=target {
            // Best-of-N: draw `screen.n_cand` candidates from the per-node policy
            // (root step = boundary sampler; depth ≥ 2 = focus/density/random), two-
            // stage screen each, and select the least-set survivor. Provenance
            // (branch/placement/focus_score) rides on each StepCand.
            let result = if d == 1 {
                // --- ROOT STEP (rev1): boundary-window sampler, own zoom, centered. ---
                let new_fw = parent.frame_width * args.root_zoom;
                if new_fw < FW_FLOOR {
                    StepResult::Died(EndCause::DegenerateExhausted)
                } else {
                    let mut gen = || -> Option<StepCand> {
                        let center = sample_boundary(&boundary, &mut rng)?;
                        Some(StepCand { center, branch: "root", placement: "center", fscore: f64::NAN })
                    };
                    best_of_n_step(&screen, new_fw, &render_node, &mut gen, &mut black_rejects, &mut occ_rejects)
                }
            } else {
                // --- NORMAL STEP (depth ≥ 2): per-node finder + placement. ---
                let new_fw = parent.frame_width * args.zoom_per_step;
                if new_fw < FW_FLOOR {
                    StepResult::Died(EndCause::DegenerateExhausted)
                } else {
                    let new_fh = new_fw * parent.out_height as f64 / parent.out_width as f64;
                    // parent_buf is always Some here (depth-1 root step set it).
                    let parent_samples = &parent_buf.as_ref().unwrap().samples;
                    let mut gen = || -> Option<StepCand> {
                        let (focus, branch, fscore) = pick_target(
                            &parent, parent_samples, node_w as usize, node_h as usize, &sigmas,
                            (p_foci, p_density), &mut rng,
                        );
                        let placement = if branch == Branch::Random {
                            Placement::Center
                        } else {
                            pick_placement((pl_center, pl_horizon, pl_random), plsum, &mut rng)
                        };
                        let center = child_center(focus, placement, new_fw, new_fh, &mut rng);
                        Some(StepCand { center, branch: branch.name(), placement: placement.name(), fscore })
                    };
                    best_of_n_step(&screen, new_fw, &render_node, &mut gen, &mut black_rejects, &mut occ_rejects)
                }
            };

            match result {
                StepResult::Accepted(child, buf, branch, placement, fscore, interior) => {
                    match branch {
                        "foci" => branch_counts[0] += 1,
                        "density" => branch_counts[1] += 1,
                        "random" => branch_counts[2] += 1,
                        _ => root_count += 1, // "root"
                    }
                    chosen_interiors.push(interior);
                    cands.push(Candidate {
                        idx: cands.len(),
                        walk: w,
                        depth: d,
                        target_depth: target,
                        branch,
                        placement,
                        focus_score: fscore,
                        cx: child.center.re,
                        cy: child.center.im,
                        fw: child.frame_width,
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

        cause_counts[match end_cause {
            EndCause::ReachedTerminalDepth => 0,
            EndCause::BlackCapExhausted => 1,
            EndCause::OccFloorExhausted => 2,
            EndCause::DegenerateExhausted => 3,
        }] += 1;

        if reached < target {
            died_early += 1;
        }
        walk_reached.push((reached, target));
        if (w + 1) % 20 == 0 || w + 1 == args.n_walks {
            eprintln!(
                "  walk {}/{}: {} candidates so far ({:.1}s)",
                w + 1,
                args.n_walks,
                cands.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    if cands.is_empty() {
        return Err("no candidates produced (every walk died on the first step?)".into());
    }

    // --- preview renders (parallel; the per-candidate frame is independent) ---
    eprintln!("rendering {} previews at {}x{} ...", cands.len(), prev_w, prev_h);
    let tp = std::time::Instant::now();
    let imgs: Vec<RgbImage> = cands
        .par_iter()
        .map(|c| {
            let frame = Frame {
                center: Complex::new(c.cx, c.cy),
                frame_width: c.fw,
                out_width: prev_w,
                out_height: prev_h,
            };
            let prec = hp::prec_bits(prev_w, c.fw);
            let cre = BigFloat::from_f64(c.cx, prec);
            let cim = BigFloat::from_f64(c.cy, prec);
            let panel = probe::render_mandel_panel(
                &cre, &cim, frame.center, c.fw, prev_w, prev_h, 1, args.maxiter, args.bailout,
                prec, trap, BackendChoice::F64,
            );
            render::shade_and_downsample(&panel.buf.samples, prev_w, prev_h, 1, &palette, &params, panel.spacing)
        })
        .collect();
    for (c, img) in cands.iter_mut().zip(imgs.iter()) {
        let fname = format!("tile_{:04}.png", c.idx);
        img.save(tiles_dir.join(&fname))
            .map_err(|e| format!("save preview {}: {e}", c.idx))?;
        c.png = format!("tiles/{fname}");
    }
    eprintln!("  previews in {:.2}s", tp.elapsed().as_secs_f64());

    // --- pool.jsonl (one row per candidate) ---
    let mut jsonl = String::new();
    for c in &cands {
        let _ = writeln!(
            jsonl,
            "{{ \"idx\": {}, \"walk\": {}, \"depth\": {}, \"target_depth\": {}, \
             \"branch\": \"{}\", \"placement\": \"{}\", \"focus_score\": {}, \
             \"cx\": {}, \"cy\": {}, \"fw\": {}, \"png\": \"{}\" }}",
            c.idx, c.walk, c.depth, c.target_depth, c.branch, c.placement,
            jnum(c.focus_score), jnum(c.cx), jnum(c.cy), jnum(c.fw), c.png,
        );
    }
    std::fs::write(out_dir.join("pool.jsonl"), jsonl)
        .map_err(|e| format!("write pool.jsonl: {e}"))?;

    // --- a quick flat PNG contact grid (sibling to the HTML) ---
    let mut grid_thumbs: Vec<RgbImage> =
        imgs.iter().map(|i| image::imageops::resize(i, 240, 135, image::imageops::FilterType::Triangle)).collect();
    for (c, th) in cands.iter().zip(grid_thumbs.iter_mut()) {
        annotate(th, c);
    }
    let grid = sheet::compose_grid(&grid_thumbs, Some(args.cols.max(1)));
    grid.save(out_dir.join("pool_grid.png"))
        .map_err(|e| format!("save pool grid: {e}"))?;

    // --- pool_sheet.html (the deliverable Matt judges) ---
    let ci = interior_summary(&chosen_interiors);
    let html = build_html(
        &cands, &walk_reached, &branch_counts, root_count, died_early, &cause_counts,
        black_rejects, black_cap_on, black_cap, occ_rejects, occ_on, occ_floor, &ci,
        args, &sigmas,
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

    println!("=== guided-descend (rev3) ===");
    println!(
        "seed={}  walks={}  candidates={}  best-of-{}",
        args.seed, args.n_walks, cands.len(), screen.n_cand
    );
    println!(
        "branch breakdown: root={} foci={} density={} random={}",
        root_count, branch_counts[0], branch_counts[1], branch_counts[2]
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
        "root-window spread: {} depth-1 samples, center std (re,im) = ({:.3}, {:.3}) of fw-3.0 view",
        roots.len(), rsx, rsy
    );
    println!(
        "repetition: {} unique frames / {} candidates; most-repeated frame ×{}",
        n_unique, cands.len(), top_mult
    );
    println!(
        "walks died early (terminated before target depth): {}/{}",
        died_early, args.n_walks
    );
    println!(
        "end-of-walk cause: terminal={} black_cap_exhausted={} occ_floor_exhausted={} degenerate_exhausted={}",
        cause_counts[0], cause_counts[1], cause_counts[2], cause_counts[3],
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

/// Draw up to `cfg.n_cand` candidates from `gen` and select the **least-interior**
/// survivor of a two-stage screen:
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
    render_node: &impl Fn(&Frame) -> render::SampleBuffer,
    gen: &mut dyn FnMut() -> Option<StepCand>,
    black_rejects: &mut usize,
    occ_rejects: &mut usize,
) -> StepResult {
    let mut saw_black = false; // a candidate failed the Stage-1 interior cap
    let mut saw_degen = false; // a candidate cleared the cap but failed the band
    let mut saw_occ = false; // a candidate cleared the band but failed the occ floor
    // Winner so far = smallest interior fraction among full survivors.
    let mut best: Option<(Frame, render::SampleBuffer, &'static str, &'static str, f64, f64)> = None;

    for _ in 0..cfg.n_cand.max(1) {
        let Some(sc) = gen() else { break };
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

        // Full survivor: keep it iff it is the least-set seen so far.
        if best.as_ref().map_or(true, |b| int_frac < b.5) {
            best = Some((frame, buf, sc.branch, sc.placement, sc.fscore, int_frac));
        }
    }

    match best {
        Some((f, b, br, pl, fs, ifr)) => StepResult::Accepted(f, b, br, pl, fs, ifr),
        None if saw_occ => StepResult::Died(EndCause::OccFloorExhausted),
        None if saw_black => StepResult::Died(EndCause::BlackCapExhausted),
        None if saw_degen => StepResult::Died(EndCause::DegenerateExhausted),
        None => StepResult::Died(EndCause::DegenerateExhausted),
    }
}

// ===========================================================================
// Policy: target selection
// ===========================================================================

/// Pick the next descent target on `parent`, returning the complex focus point,
/// the branch that chose it, and (for the foci branch) the focus's sampling score.
#[allow(clippy::too_many_arguments)]
fn pick_target(
    parent: &Frame,
    samples: &[crate::backend::PixelSample],
    w: usize,
    h: usize,
    sigmas: &[f64],
    (p_foci, p_density): (f64, f64),
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
        if let Some(f) = sample_focus(&foci, rng) {
            return (pixel_to_complex(parent, f.px, f.py), Branch::Foci, f.score);
        }
        branch = Branch::Density; // foci empty → density fallthrough
    }

    match branch {
        Branch::Density => (density_focus(parent, samples, w, h), Branch::Density, f64::NAN),
        _ => (random_interior_point(parent, rng), Branch::Random, f64::NAN),
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

/// A uniformly random interior point of the frame, ≥ 20 % from any edge.
fn random_interior_point(parent: &Frame, rng: &mut SplitMix64) -> Complex<f64> {
    let u = 0.2 + 0.6 * rng.unit();
    let v = 0.2 + 0.6 * rng.unit();
    let fh = parent.frame_height();
    Complex::new(parent.center.re + (u - 0.5) * parent.frame_width, parent.center.im + (0.5 - v) * fh)
}

/// Build the root near-boundary band: complex coords of exterior pixels whose DE
/// sits in the bottom [`BOUNDARY_DE_PCT`] of exterior DE (close to the set but not
/// interior), excluding disks around the known principal features. The depth-1
/// window center is drawn uniformly from this band — uniform over a small-DE band
/// is already boundary-biased, so no extra DE weighting.
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
                let c = pixel_to_complex(frame, x as f64, y as f64);
                if !excluded_feature(c) {
                    band.push(c);
                }
            }
        }
    }
    band
}

/// Is `c` within an exclusion disk of any principal feature?
fn excluded_feature(c: Complex<f64>) -> bool {
    let r2 = EXCLUDE_R * EXCLUDE_R;
    EXCLUDE_FEATURES
        .iter()
        .any(|&(fx, fy)| (c.re - fx).powi(2) + (c.im - fy).powi(2) <= r2)
}

/// Draw one root-window center uniformly from the boundary band.
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
    branch_counts: &[usize; 3],
    root_count: usize,
    died_early: usize,
    cause_counts: &[usize; 4],
    black_rejects: usize,
    black_cap_on: bool,
    black_cap: f64,
    occ_rejects: usize,
    occ_on: bool,
    occ_floor: f64,
    ci: &InteriorSummary,
    args: &GuidedDescendArgs,
    sigmas: &[f64],
) -> String {
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
    while w < args.n_walks {
        let row: Vec<&Candidate> = cands.iter().filter(|c| c.walk == w).collect();
        if !row.is_empty() {
            nwalks += 1;
            let (reached, target) = walk_reached[w];
            let died = if reached < target { " <span class=died>DIED</span>" } else { "" };
            let _ = write!(
                ladders,
                "<div class=ladder><div class=lmeta>walk {w} · reached {reached}/{target}{died}</div><div class=lrow>"
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
<header><h1>guided-descend rev3 — candidate pool ({ncand} candidates)</h1>\
<div class=note>{nwalks} walks emitting · best-of-{ncand_n} · branch root={rt} foci={bf} density={bd} random={br} · \
depth: {dh}· walks died early {died}/{total} · \
end-cause: terminal={ct} black-cap={cb} occ-floor={co} degenerate={cd} · \
best-of-N rejects: black-cap {blk} ×{brk}, occ-floor {olk} ×{ork} · \
chosen interior (drift check): med={cimed:.3} [{cimin:.3}..{cimax:.3}] p25/p75 {cip25:.3}/{cip75:.3} · \
preview {pal} · root-zoom {rz} (depth-1 fw≈{d1fw:.3}) · \
root-window std (re,im)=({rsx:.3},{rsy:.3}) · unique frames {nuniq}/{ncand} max-rep ×{tmult} · \
σ {sig:?} · zoom/step {zps} · seed {seed} · gates on black/interior + occupancy ONLY (no busyness axis) · \
<b>no quality claims</b> — eyeball the sampling behaviour</div></header>\
<h2>by walk — descent ladders (root→leaf left→right)</h2>{ladders}\
<h2>flat pool ({ncand})</h2><div class=grid>{flat}</div>\
</body></html>",
        ncand = cands.len(),
        ncand_n = args.descent_candidates.max(1),
        nwalks = nwalks,
        rt = root_count,
        bf = branch_counts[0],
        bd = branch_counts[1],
        br = branch_counts[2],
        dh = dh,
        died = died_early,
        total = args.n_walks,
        ct = cause_counts[0],
        cb = cause_counts[1],
        co = cause_counts[2],
        cd = cause_counts[3],
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
        rz = args.root_zoom,
        d1fw = 3.0 * args.root_zoom,
        rsx = rsx,
        rsy = rsy,
        nuniq = n_unique,
        tmult = top_mult,
        sig = sigmas,
        zps = args.zoom_per_step,
        seed = args.seed,
        ladders = ladders,
        flat = flat,
    )
}
