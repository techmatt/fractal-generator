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
pub mod descend;
pub mod energy;
pub mod font;
pub mod generate;
pub mod hp;
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
pub mod wallpaper;
