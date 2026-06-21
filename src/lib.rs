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

pub mod backend;
pub mod cli;
pub mod coloring;
pub mod descend;
pub mod font;
pub mod hp;
pub mod navigate;
pub mod palette;
pub mod palette_io;
pub mod probe;
pub mod render;
pub mod sheet;
