//! Fractal-generator render core.
//!
//! Exposed as a library so the precision backends, render driver, and coloring
//! stage are testable in-process (see `tests/perturbation.rs`, which renders the
//! same location through both backends and asserts they agree). The `main`
//! binary is a thin CLI wrapper over these modules.
//!
//! Two architectural seams:
//!  1. Precision behind [`backend::FractalBackend`] — f64 and perturbation tiers.
//!  2. A separable [`coloring`] stage (sample → RGB) so re-coloring never
//!     re-iterates (palette system, Prompt 4).

use std::path::Path;

/// Ensure the parent directory of an output path exists, creating it if needed.
///
/// Generated artifacts default under the gitignored `out/` tree (see the
/// generated-output convention in `CLAUDE.md`). Its subdirs are gitignored and
/// absent on a fresh checkout, so a no-flag invocation writing its default path
/// (e.g. `out/renders/out.png`) must create the dir first. Call this before any
/// top-level `save`/`fs::write` of an output file.
pub fn ensure_parent_dir(path: impl AsRef<Path>) -> Result<(), String> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }
    Ok(())
}

pub mod backend;
pub mod buffet;
pub mod cli;
pub mod coherence;
pub mod coloring;
pub mod corpus;
/// Throwaway diagnostic (detail-clause coverage bench: spread vs coverage
/// measures against Matt's CUT-vs-sparse labels). Test-only, compiled solely
/// under `cargo test`. Run explicitly:
/// `cargo test --release --lib detail_clause_bench -- --ignored --nocapture`.
#[cfg(test)]
mod detail_clause_bench;
pub mod deband;
pub mod descend;
pub mod energy;
/// Throwaway diagnostic (focus heatmaps: three scoring fields + combined — can we
/// locate a seed's points of focus?). Test-only, compiled solely under
/// `cargo test`, never into the production binary. Run explicitly:
/// `cargo test --release --lib focus_heatmaps -- --ignored --nocapture`.
#[cfg(test)]
mod focus_heatmaps;
pub mod font;
pub mod generate;
pub mod hp;
/// Throwaway diagnostic (log-polar characterizer at nucleus centers: does a
/// nucleus-anchored log-polar readout cleanly recover hub-ness / spiral pitch /
/// fold number / self-similar scaling / Rₙ point-symmetry, and flag delicacy's X
/// as R₂-strong?). Test-only, compiled solely under `cargo test`. Reuses the
/// `symmetry_probe` substrate + reflection. Run explicitly:
/// `cargo test --release --lib logpolar_probe -- --ignored --nocapture`.
#[cfg(test)]
mod logpolar_probe;
/// Throwaway diagnostic (characterizer-as-finder: prompt 4). Builds a guard-aware
/// spiral score S(c) on the precomputed bench field (off-axis Radon oriented-energy
/// over total, presence-gated), maps it over candidate centers (the decisive
/// S-field artifact), then hill-climbs it from nucleus and grid seeds to test
/// whether climbed centers reach the visual spiral eyes. Test-only; reuses
/// `logpolar_probe`'s sampler/detail/oriented-energy + guards and `symmetry_probe`'s
/// bench. Run explicitly:
/// `cargo test --release --lib hubfind_probe -- --ignored --nocapture`.
#[cfg(test)]
mod hubfind_probe;
/// Throwaway diagnostic (location-source base-rate probe). Test-only: compiled
/// solely under `cargo test`, never into the production binary. Run explicitly:
/// `cargo test --release --lib location_probe -- --ignored --nocapture`.
#[cfg(test)]
mod location_probe;
/// Throwaway diagnostic (discovery-sampler probe 1: diversity coverage). Test-only,
/// builds on the probe-0 scaffold. Run explicitly:
/// `cargo test --release --lib discovery_probe1 -- --ignored --nocapture`.
#[cfg(test)]
mod location_probe_probe1;
pub mod navigate;
pub mod palette;
pub mod palette_io;
pub mod palette_pick;
pub mod probe;
pub mod profile;
pub mod reject_corridor;
pub mod render;
pub mod search;
pub mod sheet;
/// Throwaway diagnostic (symmetry probe: structure-tensor winding + reflection —
/// can a topological winding integral find & sign-classify symmetry centers on
/// native smooth-iteration fields without lighting up on empty space?).
/// Test-only, compiled solely under `cargo test`. Run explicitly:
/// `cargo test --release --lib symmetry_probe -- --ignored --nocapture`.
#[cfg(test)]
mod symmetry_probe;
pub mod wallpaper;
