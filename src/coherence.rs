//! DE-coherence gate — the missing selection statistic.
//!
//! The central open problem (handoff ⭐): deep frames score "busy" on the
//! std-dev of `smooth_iter`, which cannot tell a coherent gradient from
//! sub-pixel escape speckle. Two frames can carry identical `stddev(smooth)`
//! while one is a beautiful filament sweep and the other is grey grain — the
//! difference is *spatial coherence of the boundary*, which the value-spread
//! busyness term never measures.
//!
//! The fix is nearly free because the distance estimate is already on every
//! `PixelSample`: the boundary sits a distance `de` (plane units) from the
//! pixel, so `de_px = de / pixel_spacing` is the boundary's distance **in
//! pixels**. When `de_px < ~1` the boundary is finer than one pixel, so escape
//! times alias chaotically pixel-to-pixel ⇒ guaranteed speckle. The
//! **coherence/speckle indicator** is therefore the fraction of *escaped*
//! pixels whose `de_px < θ` (θ default 1.0): high ⇒ noise ⇒ reject.
//!
//! [`coherence_stats`] is a **pure** map over an already-iterated supersampled
//! [`SampleBuffer`] — it never re-iterates (the separability the whole project
//! leans on), exactly like the coloring stage.
//!
//! **Critical pixel-spacing detail.** `de_px` is computed against the *target
//! wallpaper* pixel spacing (`frame_width / target_render_width`), **not** the
//! probe's sampling resolution. `de` is in plane units and resolution-invariant;
//! lowering the probe resolution would inflate the probe's own pixel spacing and
//! fake a higher speckle fraction. Pinning the normalization to the 2560-wide
//! target makes a cheap 640-wide probe predict the *actual* wallpaper-resolution
//! gate. See the module-level `target_render_width` argument.
//!
//! This module computes the statistic ([`coherence_stats`], the `cohere` probe)
//! **and** owns the gate it feeds ([`coherence_gate`] / [`gate_from`]): the
//! shared reject/penalty contract both selectors route their rendered frames
//! through. `search` gates each expanded node's frame (hard reject ⇒ dropped from
//! the surfaced candidates and its children pruned; soft penalty folded into the
//! node score); `wallpaper` gates per K×K window during the descent and again on
//! the final wallpaper buffer. `de_px` is always taken against the *target render
//! width*, never the probe/panel width (the spacing trap — see below).

use std::fs;
use std::path::Path;

use num_complex::Complex;

use crate::backend::Trap;
use crate::cli::{BackendChoice, CohereArgs};
use crate::hp;
use crate::probe;
use crate::render::SampleBuffer;

/// Per-frame coherence statistic over a supersampled `PixelSample` buffer.
///
/// All fractions are over **subsamples** (the buffer is at SS resolution); `de`
/// being resolution-invariant, the subsample granularity does not bias the
/// result. `subpixel_frac` and `de_px_median` are `NaN` when no subsample
/// escaped (a fully interior frame — caught upstream by the too-flat / esc_frac
/// gate, not here).
#[derive(Clone, Copy, Debug)]
pub struct CoherenceStats {
    /// Total subsamples examined.
    pub total: usize,
    /// Escaped (exterior) subsamples.
    pub escaped: usize,
    /// `escaped / total` — exterior fraction.
    pub esc_frac: f64,
    /// Escaped subsamples with `de_px < θ` (the sub-pixel-boundary speckle set).
    pub subpixel: usize,
    /// `subpixel / escaped` — **the coherence/speckle indicator**. High ⇒ the
    /// boundary is finer than a pixel across most of the exterior ⇒ grain.
    pub subpixel_frac: f64,
    /// Median `de_px` among escaped subsamples (diagnostic — a coherent frame
    /// has a fat tail of large `de_px`, pushing the median well above θ).
    pub de_px_median: f64,
    /// θ threshold used for the sub-pixel test.
    pub theta: f64,
    /// `frame_width / target_render_width` — the spacing `de_px` is taken
    /// against (the *wallpaper* spacing, independent of probe resolution).
    pub target_spacing: f64,
}

/// Compute [`CoherenceStats`] over a cached supersampled buffer. Pure: no
/// iteration, no allocation beyond the `de_px` vector needed for the median.
///
/// `target_render_width` pins `de_px` to the final wallpaper's pixel spacing
/// (see the module note), so a cheap low-res probe predicts the gate that the
/// full-resolution render will see.
pub fn coherence_stats(
    buf: &SampleBuffer,
    frame_width: f64,
    target_render_width: u32,
    theta: f64,
) -> CoherenceStats {
    let target_spacing = frame_width / target_render_width.max(1) as f64;
    let inv_spacing = 1.0 / target_spacing;

    let total = buf.samples.len();
    let mut de_px: Vec<f64> = Vec::new();
    let mut subpixel = 0usize;
    for s in &buf.samples {
        if s.escaped {
            let v = s.de * inv_spacing;
            if v < theta {
                subpixel += 1;
            }
            de_px.push(v);
        }
    }
    let escaped = de_px.len();
    let esc_frac = escaped as f64 / total.max(1) as f64;

    let (subpixel_frac, de_px_median) = if escaped == 0 {
        (f64::NAN, f64::NAN)
    } else {
        de_px.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if escaped % 2 == 1 {
            de_px[escaped / 2]
        } else {
            0.5 * (de_px[escaped / 2 - 1] + de_px[escaped / 2])
        };
        (subpixel as f64 / escaped as f64, median)
    };

    CoherenceStats {
        total,
        escaped,
        esc_frac,
        subpixel,
        subpixel_frac,
        de_px_median,
        theta,
        target_spacing,
    }
}

// ===========================================================================
// The gate — the shared reject/penalty contract wired into the selectors.
// ===========================================================================

/// Speckle-fraction reject threshold: an escaped-pixel sub-pixel-boundary
/// fraction above this is grain, not a coherent boundary. Validated empirically
/// (Prompt `de-coherence-gate`): noise frames 0.72–0.94, coherent control 0.03,
/// empty gap `[0.04, 0.72]`. Matt may retune.
pub const COHERENCE_REJECT: f64 = 0.5;

/// Median-`de_px` co-gate floor (pixels). A coherent frame has a fat tail of
/// large `de_px` (control median ~159); a speckle frame's median sits well below
/// one pixel (noise 0.003). Co-gating on this guards the `N=1`-control risk in
/// the `subpixel_frac` ceiling (the indicator could in principle saturate for a
/// single odd frame); both must agree to *pass*, either fires to *reject*.
pub const DE_PX_MEDIAN_FLOOR: f64 = 1.0;

/// Soft-penalty onset. Below this `subpixel_frac` the frame is unpenalized; from
/// here the score ramps down to 0 at [`COHERENCE_REJECT`], so a borderline frame
/// degrades gracefully instead of passing at full score then falling off a cliff.
pub const COHERENCE_SOFT_LO: f64 = 0.25;

/// The gate's verdict for one frame: a hard `reject`, or a soft `penalty`
/// multiplier in `[0,1]` to fold into the existing score (1.0 = unpenalized).
#[derive(Clone, Copy, Debug)]
pub struct CoherenceGate {
    /// Hard drop: this frame is sub-pixel speckle at the target resolution.
    pub reject: bool,
    /// Multiplier on the candidate score (`1.0` when clean / rejected-handled).
    pub penalty: f64,
    /// Which rule fired (for logs / JSON); `None` when clean.
    pub reason: Option<&'static str>,
}

/// Hermite smoothstep clamped to `[0,1]` (matches the band ramps in the selectors).
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Map a frame's [`CoherenceStats`] to the gate verdict — **the shared contract**
/// both `search` and `wallpaper` route their rendered frames through.
///
/// - **Hard reject** when `subpixel_frac > COHERENCE_REJECT` **or**
///   `de_px_median < DE_PX_MEDIAN_FLOOR` (the co-gate).
/// - **Soft penalty** otherwise: `1 − smoothstep(SOFT_LO, REJECT, subpixel_frac)`,
///   so the score ramps down as speckle rises toward the reject threshold.
/// - A fully-interior frame (`escaped == 0`, NaN stats) is a **no-op** here
///   (`reject = false`, `penalty = 1.0`): emptiness is the too-flat / `esc_frac`
///   gate's job, not the speckle gate's. Guard the NaN explicitly so the
///   smoothstep/comparisons never propagate it.
pub fn coherence_gate(stats: &CoherenceStats) -> CoherenceGate {
    gate_from(stats.subpixel_frac, stats.de_px_median)
}

/// The gate on the two raw scalars — the actual contract. `search` passes a
/// whole-frame [`CoherenceStats`] via [`coherence_gate`]; `wallpaper` passes a
/// single window's `subpixel_frac` / median `de_px` directly, so the descent
/// steers per window. Either way the same thresholds decide.
pub fn gate_from(subpixel_frac: f64, de_px_median: f64) -> CoherenceGate {
    // Interior / no-exterior frame (or window): nothing to say about coherence.
    if !subpixel_frac.is_finite() {
        return CoherenceGate { reject: false, penalty: 1.0, reason: None };
    }
    if subpixel_frac > COHERENCE_REJECT {
        return CoherenceGate { reject: true, penalty: 0.0, reason: Some("subpixel_frac>reject") };
    }
    // de_px_median is NaN only when subpixel_frac is too (handled above), so this
    // comparison is well-defined here.
    if de_px_median < DE_PX_MEDIAN_FLOOR {
        return CoherenceGate { reject: true, penalty: 0.0, reason: Some("de_px_median<floor") };
    }
    let penalty = 1.0 - smoothstep(COHERENCE_SOFT_LO, COHERENCE_REJECT, subpixel_frac);
    CoherenceGate { reject: false, penalty, reason: None }
}

/// Population standard deviation (n<2 → 0).
fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// Frame-wide normalized busyness — the *existing* selection term taken over the
/// whole frame (`stddev(smooth_iter) / maxiter` over escaped subsamples; see
/// `wallpaper::score_and_pick` / `navigate::atom_candidates`). A single
/// aggregate; it *understates* a structured frame in which a small feature sits
/// in a wide smooth surround, which is why [`windowed_busyness_max`] exists.
/// Reported only for comparison — this module changes no scoring.
fn frame_busyness(buf: &SampleBuffer, maxiter: u32) -> f64 {
    let vals: Vec<f64> = buf
        .samples
        .iter()
        .filter(|s| s.escaped)
        .map(|s| s.smooth_iter)
        .collect();
    stddev(&vals) / maxiter.max(1) as f64
}

/// Maximum windowed normalized busyness — what the actual selector sees. Mirrors
/// `wallpaper::score_and_pick`: aggregate each output pixel's escaped subsamples
/// to a mean `smooth`, slide a `k×k` window, and take the largest
/// `stddev(smooth)/maxiter` over windows with ≥3 escaped pixels. This is the
/// `max_available_busyness` reported per level in `wallpaper.json`, so a
/// structured frame's best in-band window is represented honestly (frame-wide
/// busyness would wrongly read it as too-flat).
pub(crate) fn windowed_busyness_max(buf: &SampleBuffer, probe_w: u32, probe_h: u32, k: i32, maxiter: u32) -> f64 {
    let w = probe_w as usize;
    let h = probe_h as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;

    // Per-output-pixel mean smooth over escaped subsamples + escaped flag.
    let mut smooth = vec![0.0f64; w * h];
    let mut escaped = vec![false; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut esc = 0usize;
            let mut sm = 0.0f64;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.escaped {
                        esc += 1;
                        sm += px.smooth_iter;
                    }
                }
            }
            let idx = row * w + col;
            escaped[idx] = esc * 2 >= s * s;
            smooth[idx] = if esc > 0 { sm / esc as f64 } else { 0.0 };
        }
    }

    let r = k / 2;
    let inv_scale = 1.0 / maxiter.max(1) as f64;
    let mut max_b = 0.0f64;
    for row in r..(h as i32 - r) {
        for col in r..(w as i32 - r) {
            let mut vals: Vec<f64> = Vec::with_capacity((k * k) as usize);
            for dy in -r..=r {
                for dx in -r..=r {
                    let idx = (row + dy) as usize * w + (col + dx) as usize;
                    if escaped[idx] {
                        vals.push(smooth[idx]);
                    }
                }
            }
            if vals.len() < 3 {
                continue;
            }
            let b = stddev(&vals) * inv_scale;
            if b > max_b {
                max_b = b;
            }
        }
    }
    max_b
}

// ===========================================================================
// Off-nucleus de_px-band objective (Prompt offnucleus-deband)
//
// The coherence *gate* above only walls off the sub-pixel cliff edge; it does
// not reverse the busyness gradient, which runs *uphill toward the boundary* —
// the white minibrot nuclei. Candidate generation (atom → Newton) can only point
// at nuclei, and nuclei are boundary points, so survivors pile up pressed against
// the gate. This objective inverts the target: reward a cell whose boundary is
// *resolved and a few pixels wide* (`de_px` inside a band around a target center)
// and demote busyness to a floor gate (maximizing busyness inside the band would
// re-create boundary-seeking). Maximizing band proximity frames the *decoration*
// that surrounds a minibrot instead of descending into its cusp.
//
// Default constants — **set in Phase 3 from the m6 calibration split** (a 1-D
// 3-means over `log10(de_px_win)`: nucleus de_px p50 0.055, decoration 1.97, flat
// 63; nucleus busy p50 0.026 vs decoration 0.011 — busyness anti-correlates with
// the band, confirming it must be a floor gate, not a reward term). Biased toward
// decoration / away from boundary by design (over-correction is intended). The
// `deband` subcommand can override any of these per-run via flags.
// ===========================================================================

/// `de_px` (a window's boundary distance in target-resolution pixels) at which
/// the band reward peaks. Over-corrected *off* the boundary: decoration cluster
/// median (~2.0) × ~1.5.
pub const DE_PX_BAND_CENTER: f64 = 3.0;

/// Hard reject below this `de_px` — near-boundary / sub-pixel speckle (the white
/// nuclei). Well above the nucleus cluster ceiling (p90 0.17): a deliberately wide
/// margin to guarantee survivors are off the white.
pub const DE_PX_REJECT_LO: f64 = 1.0;

/// Hard reject above this `de_px` — flat smooth exterior. Geometric mean of the
/// decoration p90 (8.0) and flat p50 (63) — cuts the flat while keeping the band.
pub const DE_PX_REJECT_HI: f64 = 22.5;

/// Busyness floor a cell must clear to count. **Floor gate only** — busyness is
/// deliberately *not* added to the reward (the maximized quantity is band
/// proximity alone; folding busyness in would re-create boundary-seeking). Just
/// under the decoration cells' busyness so the dead-flat exterior (busy ~0) drops
/// out but the structured band is kept.
pub const BUSY_FLOOR: f64 = 0.01;

/// Max center drift as a fraction of the frame half-extent, so the drifted frame
/// stays inside the originally-rendered region.
pub const DRIFT_MAX: f64 = 0.45;

/// Minimum escaped output pixels in a window for its stats to be defined (matches
/// the ≥3 rule in [`windowed_busyness_max`]).
const MIN_WIN_ESCAPED: usize = 3;

/// The four tunable knobs of the de_px-band objective. [`Default`] is the
/// calibrated module consts; the `deband` CLI overrides them per-run so the band
/// can be swept without recompiling, and the drive passes the default.
#[derive(Clone, Copy, Debug)]
pub struct BandParams {
    pub band_center: f64,
    pub reject_lo: f64,
    pub reject_hi: f64,
    pub busy_floor: f64,
}

impl Default for BandParams {
    fn default() -> Self {
        BandParams {
            band_center: DE_PX_BAND_CENTER,
            reject_lo: DE_PX_REJECT_LO,
            reject_hi: DE_PX_REJECT_HI,
            busy_floor: BUSY_FLOOR,
        }
    }
}

/// Band reward `∈[0,1]` under `bp`: a bump peaking at `band_center`, rising via
/// smoothstep from `reject_lo` and falling to `reject_hi`. Hard 0 (reject) outside
/// `[reject_lo, reject_hi]` and for a non-finite `de_px` (a window with too few
/// escaped pixels). The maximized quantity is *only* this (busyness is a gate).
pub fn band_reward_p(de_px: f64, bp: &BandParams) -> f64 {
    if !de_px.is_finite() || de_px < bp.reject_lo || de_px > bp.reject_hi {
        return 0.0;
    }
    if de_px <= bp.band_center {
        smoothstep(bp.reject_lo, bp.band_center, de_px)
    } else {
        1.0 - smoothstep(bp.band_center, bp.reject_hi, de_px)
    }
}

/// [`band_reward_p`] with the default (calibrated) [`BandParams`].
pub fn band_reward(de_px: f64) -> f64 {
    band_reward_p(de_px, &BandParams::default())
}

/// Per-output-pixel-cell windowed objective map over a cached supersampled buffer.
/// All vectors are row-major at the **output** (post-downsample) resolution; cells
/// within `k/2` of an edge are never window-centers and stay `NaN`/`0`/`false`.
pub struct CellMap {
    pub w: usize,
    pub h: usize,
    pub k: i32,
    /// Windowed median `de_px` per cell (`NaN` where `< MIN_WIN_ESCAPED` escaped).
    pub de_px: Vec<f64>,
    /// Windowed normalized busyness per cell (`stddev(smooth)/maxiter`; `0` where
    /// undefined). The floor-gate input — never folded into [`reward`](Self::reward).
    pub busy: Vec<f64>,
    /// The maximized objective: [`band_reward`]`(de_px)` gated by `busy ≥ BUSY_FLOOR`.
    pub reward: Vec<f64>,
    /// `reward > 0` — band membership **and** busy-floor pass.
    pub in_band: Vec<bool>,
}

impl CellMap {
    fn idx(&self, col: usize, row: usize) -> usize {
        row * self.w + col
    }
}

/// Median of a slice (caller guarantees non-empty); sorts a local copy.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Compute the [`CellMap`]: the **pure** per-K×K-window de_px-band reward over a
/// cached buffer. Mirrors [`windowed_busyness_max`]'s per-output-pixel aggregation
/// (mean `smooth` and escaped flag) and additionally aggregates mean `de_px`
/// pinned to the **target wallpaper spacing** (the same resolution-invariant `de`
/// normalization the gate uses, so a cheap probe predicts the final render).
///
/// `de_px_win` is the median of escaped cells' mean `de_px` in the window (robust,
/// parallels the frame stat's median choice); `busy` is `stddev(smooth)/maxiter`.
pub fn cell_reward_map(
    buf: &SampleBuffer,
    probe_w: u32,
    probe_h: u32,
    k: i32,
    maxiter: u32,
    frame_width: f64,
    target_render_width: u32,
    bp: &BandParams,
) -> CellMap {
    let w = probe_w as usize;
    let h = probe_h as usize;
    let s = buf.ss as usize;
    let sub_w = w * s;
    let inv_spacing = target_render_width.max(1) as f64 / frame_width;

    // Per-output-pixel mean smooth, mean de_px (target-spacing), escaped flag.
    let mut smooth = vec![0.0f64; w * h];
    let mut de_px = vec![0.0f64; w * h];
    let mut escaped = vec![false; w * h];
    for row in 0..h {
        for col in 0..w {
            let mut esc = 0usize;
            let mut sm = 0.0f64;
            let mut de = 0.0f64;
            for sj in 0..s {
                let base = (row * s + sj) * sub_w + col * s;
                for si in 0..s {
                    let px = &buf.samples[base + si];
                    if px.escaped {
                        esc += 1;
                        sm += px.smooth_iter;
                        de += px.de * inv_spacing;
                    }
                }
            }
            let idx = row * w + col;
            escaped[idx] = esc * 2 >= s * s;
            if esc > 0 {
                smooth[idx] = sm / esc as f64;
                de_px[idx] = de / esc as f64;
            }
        }
    }

    let r = k / 2;
    let inv_scale = 1.0 / maxiter.max(1) as f64;
    let mut map = CellMap {
        w,
        h,
        k,
        de_px: vec![f64::NAN; w * h],
        busy: vec![0.0; w * h],
        reward: vec![0.0; w * h],
        in_band: vec![false; w * h],
    };

    for row in r..(h as i32 - r) {
        for col in r..(w as i32 - r) {
            let mut smooth_vals: Vec<f64> = Vec::with_capacity((k * k) as usize);
            let mut de_vals: Vec<f64> = Vec::with_capacity((k * k) as usize);
            for dy in -r..=r {
                for dx in -r..=r {
                    let idx = (row + dy) as usize * w + (col + dx) as usize;
                    if escaped[idx] {
                        smooth_vals.push(smooth[idx]);
                        de_vals.push(de_px[idx]);
                    }
                }
            }
            if de_vals.len() < MIN_WIN_ESCAPED {
                continue;
            }
            let center = map.idx(col as usize, row as usize);
            let busy = stddev(&smooth_vals) * inv_scale;
            let dpx = median(&mut de_vals);
            map.busy[center] = busy;
            map.de_px[center] = dpx;
            let reward = if busy >= bp.busy_floor { band_reward_p(dpx, bp) } else { 0.0 };
            map.reward[center] = reward;
            map.in_band[center] = reward > 0.0;
        }
    }
    map
}

/// Reward-weighted centroid (in output-pixel cell coordinates `(col, row)`) of the
/// **best contiguous in-band, busy-floor-passing region** — 4-connected components
/// over `in_band`, the component maximizing summed `reward`, then its
/// reward-weighted centroid. A single-cell argmax would chase flukes; the
/// contiguous-region centroid frames the body of the decoration. `None` when no
/// cell is in band. Also returns the component's summed reward (region strength).
pub fn best_inband_centroid(map: &CellMap) -> Option<(f64, f64, f64)> {
    let w = map.w;
    let h = map.h;
    let mut visited = vec![false; w * h];
    let mut best: Option<(f64, f64, f64)> = None; // (cx, cy, sum_reward)
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if visited[start] || !map.in_band[start] {
            continue;
        }
        // Flood-fill this component, accumulating the reward-weighted centroid.
        stack.clear();
        stack.push(start);
        visited[start] = true;
        let (mut sx, mut sy, mut sr) = (0.0f64, 0.0f64, 0.0f64);
        while let Some(p) = stack.pop() {
            let col = p % w;
            let row = p / w;
            let rw = map.reward[p];
            sx += col as f64 * rw;
            sy += row as f64 * rw;
            sr += rw;
            let neighbors = [
                (col >= 1).then(|| p - 1),
                (col + 1 < w).then(|| p + 1),
                (row >= 1).then(|| p - w),
                (row + 1 < h).then(|| p + w),
            ];
            for n in neighbors.into_iter().flatten() {
                if !visited[n] && map.in_band[n] {
                    visited[n] = true;
                    stack.push(n);
                }
            }
        }
        if sr > 0.0 {
            let cand = (sx / sr, sy / sr, sr);
            if best.map(|b| cand.2 > b.2).unwrap_or(true) {
                best = Some(cand);
            }
        }
    }
    best
}

/// Smoothing radius (cells) for the reward-density peak: the lobe metric is the
/// summed `reward` over a `(2r+1)²` window, so a single fluke cell can't win — the
/// peak lands on the body of the densest decoration cluster.
const PEAK_SMOOTH_R: i32 = 2;

/// Location of the **densest in-band reward lobe** (output-pixel cell coords
/// `(col, row)`), plus that lobe's windowed reward density. For each in-band cell
/// it sums `reward` over a `±PEAK_SMOOTH_R` window and returns the argmax cell.
///
/// This is the drift target that **replaces** [`best_inband_centroid`]: decoration
/// is typically a *ring* of filigree around the minibrot cusp, and the centroid of
/// a ring is its hole — back on the boundary (`de_px → 0`), which is why the
/// centroid-drift landed at `de_px` ~0.06. The density peak sits on an actual lobe
/// of the ring, so the drifted frame centers on resolved decoration and the move
/// never no-ops on a symmetric halo. `None` when no cell is in band.
pub fn best_inband_peak(map: &CellMap) -> Option<(f64, f64, f64)> {
    let w = map.w as i32;
    let h = map.h as i32;
    let r = PEAK_SMOOTH_R;
    let mut best: Option<(f64, f64, f64)> = None; // (cx, cy, density)
    for row in 0..h {
        for col in 0..w {
            let center = (row as usize) * map.w + col as usize;
            if !map.in_band[center] {
                continue;
            }
            // Windowed reward density around this in-band cell.
            let mut density = 0.0f64;
            for dy in -r..=r {
                let ry = row + dy;
                if ry < 0 || ry >= h {
                    continue;
                }
                for dx in -r..=r {
                    let rx = col + dx;
                    if rx < 0 || rx >= w {
                        continue;
                    }
                    density += map.reward[(ry as usize) * map.w + rx as usize];
                }
            }
            if best.map(|b| density > b.2).unwrap_or(true) {
                best = Some((col as f64, row as f64, density));
            }
        }
    }
    best
}

/// `cohere` subcommand — isolation validation of the coherence statistic.
///
/// Renders **one** frame at a modest probe resolution with the f64 backend
/// (asserted — this is a cheap-regime diagnostic, like `wallpaper`), computes
/// [`coherence_stats`] against the 2560-wide target spacing, and prints a single
/// data row (plus the existing frame-wide busyness, for comparison). Run it once
/// per test frame; the report is assembled from the rows.
pub fn run_cohere(args: &CohereArgs) -> Result<(), String> {
    if args.frame_width <= 0.0 {
        return Err("--frame-width must be > 0".into());
    }
    if args.probe_width == 0 || args.target_width == 0 {
        return Err("--probe-width and --target-width must be > 0".into());
    }
    if args.supersample == 0 {
        return Err("--supersample must be > 0".into());
    }

    let probe_w = args.probe_width;
    let probe_h = ((probe_w as f64) * 9.0 / 16.0).round().max(1.0) as u32;
    let ss = args.supersample;

    // de is trap-independent; a point trap at the origin suffices for the render.
    let trap = Trap {
        shape: crate::backend::TrapShape::Point,
        center: Complex::new(0.0, 0.0),
        radius: 1.0,
    };

    let prec = hp::prec_bits(probe_w, args.frame_width);
    let center_re = hp::parse_decimal(&args.center_re, prec)?;
    let center_im = hp::parse_decimal(&args.center_im, prec)?;
    let center_f64 = Complex::new(hp::to_f64(&center_re), hp::to_f64(&center_im));

    let target_spacing = args.frame_width / args.target_width as f64;
    eprintln!(
        "[{}] f64 probe {probe_w}x{probe_h} ss{ss}, center=({}, {}), width={:.3e}, \
         maxiter={}; target_render_width={} → target spacing={:.3e}",
        args.label, args.center_re, args.center_im, args.frame_width, args.maxiter,
        args.target_width, target_spacing,
    );

    let t0 = std::time::Instant::now();
    let panel = probe::render_mandel_panel(
        &center_re, &center_im, center_f64, args.frame_width, probe_w, probe_h, ss,
        args.maxiter, args.bailout, prec, trap, BackendChoice::F64,
    );
    assert_eq!(
        panel.backend_name, "F64",
        "cohere must stay f64 (this is the cheap diagnostic regime)"
    );
    let buf = panel.buf;
    let secs = t0.elapsed().as_secs_f64();

    let stats = coherence_stats(&buf, args.frame_width, args.target_width, args.theta);
    let busyness = frame_busyness(&buf, args.maxiter);
    let busyness_win = windowed_busyness_max(&buf, probe_w, probe_h, args.window as i32, args.maxiter);

    eprintln!(
        "  iterated in {secs:.1}s ({} subsamples, {} escaped)",
        stats.total, stats.escaped,
    );

    // One parseable data row. `subpixel_frac` is the coherence/speckle indicator;
    // `busy_win` is the windowed-max busyness the real selector keys on.
    println!(
        "COHERE  label={:<14}  esc_frac={:.4}  subpixel_frac={:.4}  de_px_median={:.4}  busy_frame={:.4}  busy_win={:.4}  theta={}  maxiter={}",
        args.label, stats.esc_frac, stats.subpixel_frac, stats.de_px_median, busyness,
        busyness_win, args.theta, args.maxiter,
    );

    // Optional JSON sidecar (one frame per file) for re-tabulation.
    if let Some(path) = &args.json {
        let json = build_json(args, &stats, busyness, busyness_win, secs);
        crate::ensure_parent_dir(path)?;
        fs::write(path, json).map_err(|e| format!("failed to write {path}: {e}"))?;
        eprintln!("  wrote {}", probe::path_str(Path::new(path)));
    }

    Ok(())
}

fn build_json(
    args: &CohereArgs,
    stats: &CoherenceStats,
    busyness: f64,
    busyness_win: f64,
    secs: f64,
) -> String {
    use probe::{jf, js};
    let mut s = String::from("{\n");
    s.push_str(&format!("  \"label\": {},\n", js(&args.label)));
    s.push_str(&format!(
        "  \"center\": {{ \"re\": {}, \"im\": {} }},\n",
        js(&args.center_re),
        js(&args.center_im)
    ));
    s.push_str(&format!("  \"frame_width\": {},\n", jf(args.frame_width)));
    s.push_str(&format!("  \"maxiter\": {},\n", args.maxiter));
    s.push_str(&format!("  \"probe_width\": {},\n", args.probe_width));
    s.push_str(&format!("  \"target_render_width\": {},\n", args.target_width));
    s.push_str(&format!("  \"theta\": {},\n", jf(args.theta)));
    s.push_str(&format!("  \"target_spacing\": {},\n", jf(stats.target_spacing)));
    s.push_str(&format!("  \"total_subsamples\": {},\n", stats.total));
    s.push_str(&format!("  \"escaped\": {},\n", stats.escaped));
    s.push_str(&format!("  \"esc_frac\": {},\n", jf(stats.esc_frac)));
    s.push_str(&format!("  \"subpixel\": {},\n", stats.subpixel));
    s.push_str(&format!("  \"subpixel_frac\": {},\n", jf(stats.subpixel_frac)));
    s.push_str(&format!("  \"de_px_median\": {},\n", jf(stats.de_px_median)));
    s.push_str(&format!("  \"frame_busyness\": {},\n", jf(busyness)));
    s.push_str(&format!("  \"windowed_busyness_max\": {},\n", jf(busyness_win)));
    s.push_str(&format!("  \"window\": {},\n", args.window));
    s.push_str(&format!("  \"iterate_secs\": {}\n", jf(secs)));
    s.push_str("}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::PixelSample;

    fn px(escaped: bool, de: f64) -> PixelSample {
        PixelSample {
            escaped,
            smooth_iter: 0.0,
            de,
            trap_min: 0.0,
            trap_phase: 0.0,
            glitched: false,
            atom_period: 0,
            atom_min: f64::INFINITY,
        }
    }

    fn buf(samples: Vec<PixelSample>, out_w: u32, out_h: u32) -> SampleBuffer {
        SampleBuffer {
            samples,
            out_width: out_w,
            out_height: out_h,
            ss: 1,
            glitched_pixels: 0,
        }
    }

    /// Core arithmetic: esc_frac, subpixel_frac (de_px < θ over escaped), and the
    /// de_px median, with de_px taken against the *target* spacing.
    #[test]
    fn stats_counts_and_median() {
        // frame_width=1, target=10 → target_spacing=0.1, de_px = de*10.
        let samples = vec![
            px(true, 0.005), // de_px 0.05  < 1  speckle
            px(true, 0.50),  // de_px 5.0   ≥ 1
            px(true, 0.02),  // de_px 0.2   < 1  speckle
            px(false, 0.0),  // interior
        ];
        let s = coherence_stats(&buf(samples, 2, 2), 1.0, 10, 1.0);
        assert_eq!(s.total, 4);
        assert_eq!(s.escaped, 3);
        assert!((s.esc_frac - 0.75).abs() < 1e-12);
        assert_eq!(s.subpixel, 2);
        assert!((s.subpixel_frac - 2.0 / 3.0).abs() < 1e-12);
        // sorted de_px = [0.05, 0.2, 5.0] → median 0.2.
        assert!((s.de_px_median - 0.2).abs() < 1e-12);
    }

    /// The critical pixel-spacing detail: `de_px` must depend on
    /// `target_render_width`, **never** on the probe's own resolution. The same
    /// buffer scored against two target widths gives proportionally different
    /// de_px (and thus speckle counts), independent of `out_width`.
    #[test]
    fn de_px_pinned_to_target_not_probe_resolution() {
        let mk = || {
            vec![
                px(true, 0.05), // de_px = 0.05 / target_spacing
                px(true, 0.05),
            ]
        };
        // Probe "resolution" (out_width) differs, target width identical → identical.
        let coarse = coherence_stats(&buf(mk(), 4, 1), 1.0, 1000, 1.0);
        let fine = coherence_stats(&buf(mk(), 400, 1), 1.0, 1000, 1.0);
        assert_eq!(coarse.subpixel, fine.subpixel, "probe resolution must not move the gate");
        assert!((coarse.de_px_median - fine.de_px_median).abs() < 1e-12);

        // de=0.05, target_spacing = 1/1000 = 1e-3 → de_px = 50 ≥ 1 → not speckle.
        assert_eq!(coarse.subpixel, 0);
        assert!((coarse.de_px_median - 50.0).abs() < 1e-9);

        // Deeper target (finer spacing) makes the same de sub-pixel.
        // target=100000 → spacing 1e-5 → de_px = 5000? no: de/spacing = 0.05/1e-5=5000.
        // Use a target where de_px < 1: target small. width 1, target=10 → spacing .1 → de_px .5 <1.
        let deep = coherence_stats(&buf(mk(), 4, 1), 1.0, 10, 1.0);
        assert_eq!(deep.subpixel, 2, "smaller target spacing should flag sub-pixel");
    }

    /// A `CoherenceStats` carrying only the two fields the gate keys on (the rest
    /// are irrelevant to [`coherence_gate`]).
    fn stats_with(subpixel_frac: f64, de_px_median: f64) -> CoherenceStats {
        CoherenceStats {
            total: 1,
            escaped: 1,
            esc_frac: 1.0,
            subpixel: 0,
            subpixel_frac,
            de_px_median,
            theta: 1.0,
            target_spacing: 1.0,
        }
    }

    /// Phase-3 wiring validation: the **cached** isolation frames (`out/cohere/`
    /// sidecars) fed through the augmented gate must reject the speckle frames (A
    /// noise, the deep flat L8/L9) and pass the coherent control C at full score.
    /// The gate consumes the per-frame statistic, so this drives it with the exact
    /// cached `subpixel_frac` / `de_px_median` — proving the term composes into the
    /// selector pipeline before the expensive drive, not just the standalone probe.
    #[test]
    fn gate_rejects_cached_noise_passes_control() {
        // (label, subpixel_frac, de_px_median) straight from the cached sidecars.
        let a_noise = coherence_gate(&stats_with(0.93659114829612, 2.645664870252327e-3));
        let b_l8 = coherence_gate(&stats_with(0.720150880079176, 5.94462370288494e-2));
        let b_l9 = coherence_gate(&stats_with(0.7424875053982322, 4.766830046372818e-2));
        let c_control = coherence_gate(&stats_with(2.6053421714347682e-2, 1.5914030317454007e2));

        // Speckle frames: hard reject (both the subpixel_frac and the co-gate fire).
        assert!(a_noise.reject, "A noise must be rejected");
        assert!(b_l8.reject, "B flat L8 must be rejected");
        assert!(b_l9.reject, "B flat L9 must be rejected");

        // Coherent control: survives, and at full score (well below the soft onset).
        assert!(!c_control.reject, "C control must survive the gate");
        assert!(
            (c_control.penalty - 1.0).abs() < 1e-9,
            "C control penalty should be ~1.0 (subpixel_frac far below soft onset), got {}",
            c_control.penalty
        );
        assert!(c_control.reason.is_none());
    }

    /// The co-gate guards the `N=1` ceiling risk: a frame with an *acceptable*
    /// `subpixel_frac` but a sub-pixel median `de_px` is still rejected.
    #[test]
    fn co_gate_rejects_on_median_alone() {
        let g = coherence_gate(&stats_with(0.1, 0.5)); // frac ok, median < floor
        assert!(g.reject);
        assert_eq!(g.reason, Some("de_px_median<floor"));
    }

    /// The soft penalty ramps monotonically over `[SOFT_LO, REJECT]` and is a no-op
    /// below the onset.
    #[test]
    fn soft_penalty_ramps() {
        let clean = coherence_gate(&stats_with(0.10, 100.0));
        let mid = coherence_gate(&stats_with(0.375, 100.0)); // midpoint of [0.25, 0.5]
        let hot = coherence_gate(&stats_with(0.49, 100.0));
        assert!((clean.penalty - 1.0).abs() < 1e-9);
        assert!(mid.penalty < clean.penalty && mid.penalty > hot.penalty);
        assert!(hot.penalty < 0.1, "near the reject edge the penalty should be small");
        assert!(!clean.reject && !mid.reject && !hot.reject);
    }

    /// Band reward shape: 0 (reject) outside `[REJECT_LO, REJECT_HI]` and for a
    /// non-finite de_px, peaks at the center, rises and falls monotonically.
    #[test]
    fn band_reward_bump_shape() {
        assert_eq!(band_reward(f64::NAN), 0.0);
        assert_eq!(band_reward(DE_PX_REJECT_LO - 0.1), 0.0, "below LO is rejected (the white nuclei)");
        assert_eq!(band_reward(DE_PX_REJECT_HI + 1.0), 0.0, "above HI is rejected (flat)");
        let peak = band_reward(DE_PX_BAND_CENTER);
        assert!((peak - 1.0).abs() < 1e-9, "reward peaks at the band center");
        // Monotone rise on the low skirt, fall on the high skirt.
        let lo_mid = 0.5 * (DE_PX_REJECT_LO + DE_PX_BAND_CENTER);
        let hi_mid = 0.5 * (DE_PX_BAND_CENTER + DE_PX_REJECT_HI);
        assert!(band_reward(lo_mid) > 0.0 && band_reward(lo_mid) < peak);
        assert!(band_reward(hi_mid) > 0.0 && band_reward(hi_mid) < peak);
    }

    /// The objective flip in miniature: a window whose `de_px` sits in the band
    /// (decoration) outscores one whose `de_px` is near zero (the nucleus cusp),
    /// even when the nucleus window is *busier* — busyness is a floor gate only.
    #[test]
    fn band_beats_nucleus_even_when_busier() {
        let decoration = if 1.0 >= BUSY_FLOOR { band_reward(DE_PX_BAND_CENTER) } else { 0.0 };
        let nucleus = if 5.0 >= BUSY_FLOOR { band_reward(DE_PX_REJECT_LO * 0.1) } else { 0.0 };
        assert!(decoration > nucleus, "in-band decoration must outscore the busier near-zero-de nucleus");
        assert_eq!(nucleus, 0.0);
    }

    /// `best_inband_centroid` picks the larger (higher summed-reward) contiguous
    /// component and returns its reward-weighted centroid, not a stray single cell.
    #[test]
    fn centroid_picks_largest_region() {
        let w = 8usize;
        let h = 4usize;
        let mut map = CellMap {
            w,
            h,
            k: 1,
            de_px: vec![f64::NAN; w * h],
            busy: vec![0.0; w * h],
            reward: vec![0.0; w * h],
            in_band: vec![false; w * h],
        };
        // A fluke single cell at (0,0); a 2x2 block at cols 4..6, rows 1..3.
        let set = |m: &mut CellMap, c: usize, r: usize, rw: f64| {
            let i = r * w + c;
            m.reward[i] = rw;
            m.in_band[i] = true;
        };
        set(&mut map, 0, 0, 1.0);
        for c in 4..6 {
            for r in 1..3 {
                set(&mut map, c, r, 1.0);
            }
        }
        let (cx, cy, sr) = best_inband_centroid(&map).expect("a region exists");
        assert!((sr - 4.0).abs() < 1e-9, "the 2x2 block (sum 4) beats the single fluke (sum 1)");
        assert!((cx - 4.5).abs() < 1e-9 && (cy - 1.5).abs() < 1e-9, "centroid is the block center");
    }

    /// Phase-3 drift-target fix: on a *ring* of in-band reward the centroid lands
    /// in the hole (back on the boundary), but `best_inband_peak` lands on the ring
    /// itself (the densest lobe). This is the bug the peak target fixes.
    #[test]
    fn peak_lands_on_ring_centroid_in_hole() {
        let w = 11usize;
        let h = 11usize;
        let mut map = CellMap {
            w,
            h,
            k: 1,
            de_px: vec![f64::NAN; w * h],
            busy: vec![0.0; w * h],
            reward: vec![0.0; w * h],
            in_band: vec![false; w * h],
        };
        // A reward ring at radius ~3 from the grid center (5,5); empty hole inside.
        let (cx0, cy0) = (5i32, 5i32);
        for row in 0..h as i32 {
            for col in 0..w as i32 {
                let d = (((col - cx0).pow(2) + (row - cy0).pow(2)) as f64).sqrt();
                if (d - 3.0).abs() < 1.0 {
                    let i = row as usize * w + col as usize;
                    map.reward[i] = 1.0;
                    map.in_band[i] = true;
                }
            }
        }
        // Centroid of the symmetric ring is its center (the hole).
        let (ccx, ccy, _) = best_inband_centroid(&map).expect("ring is one component");
        assert!((ccx - 5.0).abs() < 1e-6 && (ccy - 5.0).abs() < 1e-6, "centroid is the hole center");
        assert!(!map.in_band[5 * w + 5], "the hole center is itself out of band");

        // The density peak lands on the ring — an in-band cell, not the hole.
        let (pcx, pcy, dens) = best_inband_peak(&map).expect("ring has in-band cells");
        let pi = pcy as usize * w + pcx as usize;
        assert!(map.in_band[pi], "peak cell must be in band (on the ring)");
        let dist = (((pcx as i32 - cx0).pow(2) + (pcy as i32 - cy0).pow(2)) as f64).sqrt();
        assert!(dist > 1.5, "peak sits out on the ring, not in the hole: dist={dist}");
        assert!(dens > 0.0);
    }

    /// Dive-agnostic boundary (Prompt julia-off-dive-agnostic, Phase 3/4): the
    /// field-space objective chain — `cell_reward_map` → `best_inband_centroid` —
    /// is callable from a **bare `PixelSample` buffer + frame geometry alone**, with
    /// no nucleus / period / size-estimate / atom-domain context. A future Julia
    /// dive supplies its own buffer and calls these unchanged; only candidate
    /// framing (atom → Newton → nuclei) stays Mandelbrot-specific.
    #[test]
    fn band_objective_runs_on_bare_buffer() {
        // 6x6 single-sample grid, all escaped: de=0.3 plane units everywhere and a
        // per-column smooth_iter gradient. Nothing here is Mandelbrot-derived.
        let (w, h) = (6u32, 6u32);
        let mut samples = Vec::with_capacity((w * h) as usize);
        for _row in 0..h {
            for col in 0..w {
                let mut s = px(true, 0.3);
                s.smooth_iter = col as f64 * 3.0; // intra-window spread → clears BUSY_FLOOR
                samples.push(s);
            }
        }
        let b = buf(samples, w, h);

        // frame_width=1, target=10 → de_px = de*10 = 3.0 = DE_PX_BAND_CENTER (peak).
        let bp = BandParams::default();
        let map = cell_reward_map(&b, w, h, 3, 100, 1.0, 10, &bp);
        assert_eq!((map.w, map.h), (w as usize, h as usize));

        // Interior windows land at the band peak and clear the busy floor.
        let center = map.idx(3, 3);
        assert!((map.de_px[center] - 3.0).abs() < 1e-9, "de_px median at band center");
        assert!(map.busy[center] >= bp.busy_floor, "busy clears the floor");
        assert!((map.reward[center] - 1.0).abs() < 1e-9, "reward peaks in-band");
        assert!(map.in_band[center]);

        // The whole interior is one in-band component; its centroid is the grid
        // center — computed without any Mandelbrot parameter.
        let (cx, cy, sr) = best_inband_centroid(&map).expect("an in-band region exists");
        assert!((cx - 2.5).abs() < 1e-6 && (cy - 2.5).abs() < 1e-6);
        assert!(sr > 0.0);
    }

    /// Fully interior frame: no escaped subsamples → NaN coherence (handled
    /// upstream by the too-flat/esc_frac gate, not here).
    #[test]
    fn all_interior_is_nan() {
        let samples = vec![px(false, 0.0), px(false, 0.0)];
        let s = coherence_stats(&buf(samples, 2, 1), 1.0, 10, 1.0);
        assert_eq!(s.escaped, 0);
        assert_eq!(s.esc_frac, 0.0);
        assert!(s.subpixel_frac.is_nan());
        assert!(s.de_px_median.is_nan());
        // The gate is a no-op on an interior frame (emptiness is another gate's job).
        let g = coherence_gate(&s);
        assert!(!g.reject && (g.penalty - 1.0).abs() < 1e-12 && g.reason.is_none());
    }
}
