# Render-path config audit: `render` vs `render-one` vs `colormap.py`

**Trigger.** The q4 stage-1 label crops came out flat/washed-out under the vivid UF
`default` palette, even though the approved fair montage (`out/fair_rerender/`) is
vividly banded. Same palette name, same location, same field — different image. This
was not a palette bug; it was a **coloring-path** bug. This report maps the render
config surface, names the real divergence, and proposes the (small) changes worth
making. Matt's read that "there's a concerning divergence and these should be more
unified" is **correct** — but the fix is not "deprecate `render-one`". The duplication
that bites is elsewhere.

## TL;DR

There are **two different coloring algorithms** in the engine and **three disjoint
palette namespaces**, and the mapping between an entry point and which
algorithm/namespace it uses is implicit and inconsistent:

1. **Two colorings, both called "the render".** *Location-profile* (`coloring::shade`
   — raw smooth-iteration × `density`, cycled → the classic banded UF look) vs
   *beautiful* (`render_modes` / `colormap.py` — percentile-stretch → transform →
   single palette pass → a smooth gradient). Same field + same palette → **visibly
   different images at their respective defaults.** The q4 fields were dumped and
   recolored through the *beautiful* tail (`colormap.py`), which **structurally cannot
   reproduce** the location-profile banding the montage was rendered with.

2. **Three palette namespaces.** The UF `default` (blue→white→orange→black — the
   signature Mandelbrot palette) exists **only** as a Rust built-in. It is **absent**
   from both colormap-library JSONs. So `--palette default` works on `render`/`sheet`
   and **errors** on `render-one`, and is unreachable from `colormap.py`.

3. **Even one coloring has two default densities.** Location-profile `density` is
   **0.025** from the bare-`render`/`sheet` CLI but **0.004** from
   `generate::color_params()` (every corpus path, including `render-one`). ~6× denser
   banding depending on entry point.

None of this is documented in one place, and nothing fails loudly — you get a
different-looking image, not an error. That's the hazard.

## The map

| Entry point | Palette resolver | Default palette | Coloring algorithm | Cycle/density default | AA default |
|---|---|---|---|---|---|
| bare `render` (`main.rs::run_render`) | `load_palette`: **built-in OR file/path** | `default` (UF) | **location-profile** (`shade_and_downsample`) | `density` **0.025** (`ShadeArgs`) | ss2, box |
| `sheet` (`main.rs::run_sheet`) | `--builtins`→`palette::builtin`; `--palettes`→file | *(required, none)* | **location-profile** (`render_contact_sheet`) | `density` **0.025** (`ShadeArgs`) | ss2, box |
| `render-one` deg-2, no `--coloring` (`render_one.rs`) | `parse_colormaps(--colormaps)`: **file only** | `twilight` | **location-profile** (`generate::color_params`) | `density` **0.004** | ss4, lanczos3 |
| `render-one --coloring …` / new families | file only | `twilight` | **beautiful** (`render_modes::render_beautiful`) | `palette_cycles` **1** + pct-stretch | ss4, lanczos3 |
| `colormap.py` (recolor tail: dump-field → `enrich`/keeper/label/q4 crops) | file only (`score3_colormaps.json`) | *(per-config)* | **beautiful smooth** (pct-stretch, pinned to `render_modes`) | `n_cycles` **1** + pct-stretch | box/lanczos3 |

Palette namespaces:
- **N1 — Rust built-ins** (`palette::builtin`, `palette.rs:409`): `default`, `cubehelix`,
  `viridis`. **`default` lives only here.** Reachable from `render` (`load_palette`) and
  `sheet --builtins`.
- **N2 — `clean_colormaps.json`** (224 maps): `render-one`/`present` default library. Has
  `cubehelix`/`viridis`, **not** `default`.
- **N3 — `score3_colormaps.json`** (76 maps): `colormap.py` default library. **Not**
  `default`, **not** `cubehelix`.

## Why the q4 crops broke (the concrete failure)

The stage-1 pipeline: `render-one --dump-field` writes the **beautiful smooth** field;
`colormap.py` colors it through the **beautiful** pipeline (percentile-stretch 0.5/99.5
→ single palette pass). At `n_cycles=1` that maps the whole escape range onto **one**
traverse of the gradient → a smooth blue→white→orange ramp with the mid-tone filigree
compressed into a narrow band → the flat, washed-out look Matt flagged.

The approved montage was rendered by `sheet --builtins default` = **location-profile**:
`smooth_iter × 0.025`, cycled → the gradient repeats every ~40 iterations → the
concentric **banding** with bright filigree that reads as "vivid".

So the two paths were **never** going to match. The field⊗colormap split reproduces the
*beautiful* coloring faithfully (that's its contract) — but the q4 target look is the
*location-profile* coloring, which the split cannot express. **Fix applied:** q4 capture
now re-renders each field full-frame via bare `render --palette default` (location-profile,
density 0.025 — byte-consistent with the `sheet` montage) and crops from that, rather than
recoloring the dumped field in Python.

## Is `render-one` redundant / deprecated-worthy?

**No.** `render-one` is the locked **wallpaper-quality** path and carries real,
non-duplicated responsibility the bare `render` path does not: ss4 + Lanczos-3 lock,
`.jpg` quality-controlled output, multi-family support (multibrot/Julia/phoenix), the
`--coloring` beautiful pipeline, and `--dump-field`. Bare `render` is deliberately the
**fast ss2 preview/diagnostic** path (`render_one.rs:13` says as much). Collapsing them
would either slow previews or bloat the preview path.

The redundancy that actually hurts is **not** `render` vs `render-one` — it is:
- **the same coloring reachable two ways with different defaults** (density 0.025 vs
  0.004), and
- **a Python re-implementation of *one* of the two colorings** (`colormap.py` =
  beautiful only) presented as "the coloring tail," with no in-repo statement that it
  cannot produce the location-profile look, and
- **a palette name (`default`) that resolves on some paths and errors on others.**

## Proposals

Ranked by value-per-unit-churn. Batch reproducibility (frozen flag names/defaults) is a
hard constraint (see CLAUDE.md), so none of these silently changes an existing render's
output.

### P1 — Unify palette resolution (small, high value). **Recommended.**
Give every palette-name resolver the same fallback order: **library file → built-in
registry (N1) → path**. Concretely, in `render_one.rs` (and any file-only resolver),
when `parse_colormaps` has no match for the name, fall back to `palette::builtin(name)`
before erroring. Effect: `render-one --palette default` works instead of failing; the
"works here, errors there" foot-gun that cost this session disappears. ~15 lines,
additive, changes no existing output (only turns former errors into renders).

*Alternative / complement:* bake the N1 built-ins (`default`, `cubehelix`) into the
colormap-library JSONs so N2/N3 are supersets of N1. Cleaner conceptually, but mutating
the committed curated libraries has provenance cost (`score3_colormaps.json` is the
*score-3 curated subset* — `default` doesn't belong in it). Prefer the code fallback.

### P2 — Name the two colorings and document the split (small, high value). **Recommended.**
Add a short "Two colorings" section to the render module docs (or a top-level doc) stating
plainly: *location-profile* (`shade`, banded, the `render`/`sheet`/`render-one`-default
look) vs *beautiful* (`render_modes`/`colormap.py`, percentile-stretched field look), that
they differ at their defaults, and that **the field⊗colormap (dump-field → `colormap.py`)
path reproduces only the *beautiful* coloring.** One paragraph would have pre-empted this
bug. The `colormap.py` header already says it is "pinned to the Rust `render_modes.rs`
smooth path" — but it never says that path is *not* what bare `render`/`sheet` produce.

### P3 — Reconcile the two location-profile density defaults (small, medium value).
`density` = 0.025 (CLI) vs 0.004 (`generate::color_params`) is a real, undocumented ~6×
split. Don't silently change either (repro), but **name them** at the definition sites
("preview density" / "corpus density") and add a one-line comment cross-referencing the
other. Consider whether the corpus really wants 0.004 vs the previewed 0.025 — if the
label crops are colored at a density the human never previews, that's a subtler version
of today's bug.

### P4 — Audit the corpus coloring for the same mismatch (investigate, flag only).
Today's failure mode is general: **if any training/label crops are produced through
`colormap.py` (beautiful) while the human judges the location-profile look — or vice
versa — the classifier is learning a coloring the deploy path won't reproduce.** Worth a
focused check of which coloring each committed batch under `data/label_corpus/` /
`data/q4_window_corpus/` was rendered with, and whether that matches the intended deploy
render. Out of scope here; logged as the highest-value follow-up.

### P5 — (Optional, larger) Single render entry with a quality switch.
Longer term, fold bare `render` into `render-one` as `--quality preview|wallpaper` (or
make `render` a thin alias that sets ss2/box). Removes the dual-entry location-profile
duplication and the density-default split at the source. Invasive, touches every render
call site and batch script — **not worth doing now**; revisit only if a third quality
tier appears.

## What I would not change
- The **location-profile vs beautiful** distinction itself is legitimate — they are two
  real aesthetics, both wanted. The problem is discoverability/defaults, not that both
  exist.
- **Frozen flag names and default values** — batch reproducibility depends on them. Every
  proposal above is additive (new fallbacks, docs, comments), not a mutation of existing
  render output.

## One-line verdict
Not "`render-one` is redundant" — rather **one palette name resolves inconsistently, two
colorings share the word "default", and the Python recolor tail silently implements only
one of them.** P1+P2 (a palette-resolver fallback + one doc paragraph) remove the sharp
edges for a few dozen lines and zero output churn; P4 (corpus coloring audit) is the
follow-up that matters most.
