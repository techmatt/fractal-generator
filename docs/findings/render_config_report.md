# Render-path config audit: `render` vs `render-one` vs `colormap.py`

**Trigger.** The q4 stage-1 label crops came out flat/washed-out under the vivid UF
`default` palette, even though the approved fair montage (`out/fair_rerender/`) is
vividly banded. Same palette name, same location, same field — different image. This
was not a palette bug; it was a **coloring-path** bug. This report maps the render
config surface, names the real divergence, and tracks the changes made / still open.
Matt's read that "there's a concerning divergence and these should be more unified" is
**correct** — but the fix is not "deprecate `render-one`". The duplication that bites is
elsewhere.

**Status:** P1 (palette-resolver fallback) and P2 (document the two colorings) are
**DONE** (this pass). P3–P5 are open TODOs below, with concrete next steps for whoever
picks them up.

## TL;DR

There are **two different coloring algorithms** in the engine and **three disjoint
palette namespaces**, and the mapping between an entry point and which
algorithm/namespace it uses is implicit and easy to get wrong:

1. **Two colorings, both called "the render".** *Location-profile* (`coloring::shade`
   — raw smooth-iteration × `density`, cycled → the classic banded UF look) vs
   *beautiful* (`render_modes` / `colormap.py` — percentile-stretch → transform →
   single palette pass → a smooth gradient). Same field + same palette → **visibly
   different images at their respective defaults.** The q4 fields were dumped and
   recolored through the *beautiful* tail (`colormap.py`), which **structurally cannot
   reproduce** the location-profile banding the montage was rendered with.

2. **Three palette namespaces.** The UF `default` (blue→white→orange→black — the
   signature Mandelbrot palette) exists **only** as a Rust built-in. It is **absent**
   from both colormap-library JSONs. *(P1 fixed the resolver asymmetry — see below —
   but the namespaces are still disjoint.)*

3. **Even one coloring has two default densities.** Location-profile `density` is
   **0.025** from the bare-`render`/`sheet` CLI but **0.004** from
   `generate::color_params()` (every corpus path, including `render-one`). ~6× denser
   banding depending on entry point. **Still open (P3).**

Nothing fails loudly — you get a different-looking image, not an error. That's the
hazard, and why P2 (documentation at the code sites) matters as much as the code fix.

## The map

| Entry point | Palette resolver | Default palette | Coloring algorithm | Cycle/density default | AA default |
|---|---|---|---|---|---|
| bare `render` (`main.rs::run_render`) | `load_palette`: **built-in OR file/path** | `default` (UF) | **location-profile** (`shade_and_downsample`) | `density` **0.025** (`ShadeArgs`) | ss2, box |
| `sheet` (`main.rs::run_sheet`) | `--builtins`→`palette::builtin`; `--palettes`→file | *(required, none)* | **location-profile** (`render_contact_sheet`) | `density` **0.025** (`ShadeArgs`) | ss2, box |
| `render-one` deg-2, no `--coloring` (`render_one.rs`) | library file, **then built-in fallback (P1)** | `twilight` | **location-profile** (`generate::color_params`) | `density` **0.004** (× `--n-cycles`) | ss4, lanczos3 |
| `render-one --coloring …` / new families | library file, then built-in fallback (P1) | `twilight` | **beautiful** (`render_modes::render_beautiful`) | `palette_cycles` **1** + pct-stretch | ss4, lanczos3 |
| `colormap.py` (recolor tail: dump-field → `enrich`/keeper/label/q4 crops) | file only (`score3_colormaps.json`) | *(per-config)* | **beautiful smooth** (pct-stretch, pinned to `render_modes`) | `n_cycles` **1** + pct-stretch | box/lanczos3 |

Palette namespaces:
- **N1 — Rust built-ins** (`palette::builtin`, `palette.rs:409`): `default`, `cubehelix`,
  `viridis`. **`default` lives only here.**
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
density 0.025 — byte-consistent with the `sheet` montage) and crops from that
(`tools/studies/q4_stage1_labelset.py::stage_capture`).

## Is `render-one` redundant / deprecated-worthy?

**No.** `render-one` is the locked **wallpaper-quality** path and carries real,
non-duplicated responsibility bare `render` does not: ss4 + Lanczos-3 lock, `.jpg`
quality-controlled output, multi-family support (multibrot/Julia/phoenix), the
`--coloring` beautiful pipeline, and `--dump-field`. Bare `render` is deliberately the
**fast ss2 preview/diagnostic** path (`render_one.rs:13`). Collapsing them would either
slow previews or bloat the preview path. The redundancy that hurts is not `render` vs
`render-one`; it is the two-colorings / three-namespaces / split-density surface above.

---

## Changes made this pass

### P1 — Palette-resolver fallback (DONE)
`render_one.rs`: the palette lookup now tries the colormap library file **first**
(unchanged for any name already present there), then falls back to `palette::builtin`.
`render-one --palette default` now renders instead of erroring; unknown names still error,
now with a message that names the built-ins. Purely additive — no name that already
resolved changes output. Verified: `--palette default` renders, `--palette twilight`
unchanged, `--palette nonesuch` errors.

> **Gotcha it exposes (feeds P3):** `render-one --palette default` renders at the
> location-profile **density 0.004**, i.e. *looser* banding than the `sheet`/bare-`render`
> montage (0.025). Same palette, same coloring, different density. To reproduce the
> montage's tight banding from `render-one`, pass **`--n-cycles ~6.25`** (0.025 / 0.004;
> `render-one` does `density *= n_cycles`).

### P2 — Document the two colorings (DONE)
Caveat paragraphs added at all four code sites so the split is discoverable from the code,
not just this report:
- `src/render_modes.rs` (`//!`) — "They also LOOK different at default"; names the banded
  vs pct-stretched distinction and that `colormap.py` mirrors *this* path.
- `src/coloring.rs` (`//!`) — labels itself the *location-profile* path, distinct from
  `render_modes`/`colormap.py`.
- `tools/colormap.py` (header) — CAVEAT: reproduces the *beautiful* coloring only; a
  `--dump-field` recolor cannot produce the location-profile banding.
- `render_one.rs` `--palette` arg doc — notes the built-in fallback.

---

## Open TODOs (for future CC instances)

Ranked by value. All are additive; **frozen flag names/defaults are a hard constraint**
(batch reproducibility — CLAUDE.md), so none may silently change an existing render's
output.

### TODO P3 — Reconcile / name the two location-profile density defaults (small, medium value)
`density` = 0.025 (`ShadeArgs`, `cli.rs`) vs 0.004 (`generate::color_params`,
`generate.rs:189`) is a real, undocumented ~6× split. **Next steps:**
1. Don't silently change either value (repro). **Do** add a comment at each definition
   site naming them ("preview density 0.025" / "corpus density 0.004") and
   cross-referencing the other + this report.
2. Decide, with Matt, whether the corpus/label crops *should* be colored at 0.004 when the
   human previews 0.025 — if the labeled look ≠ the previewed look, that is a subtler
   instance of the q4 bug (see P4). If they should match, the fix is a deliberate,
   documented density change on the corpus path (a repro-breaking migration — treat as
   its own task with a re-render of affected batches).

### TODO P4 — Audit corpus coloring for the same mismatch (investigate; **highest value**)
Today's failure generalizes: **if label/training crops are produced through one coloring
(e.g. `colormap.py` beautiful) while the human judges — or the deploy path renders — the
other (location-profile banded), the classifier learns a look the pipeline won't
reproduce.** **Next steps:**
1. For each committed batch under `data/label_corpus/` and `data/wallpaper_corpus/`,
   determine which coloring produced its crops (grep the batch generator / `meta.json` for
   `colormap.py` vs a Rust `render`/`render-one` call; check `render.palette` + whether a
   dumped field was recolored).
2. Cross-check against the intended **deploy** render for that head.
3. Report per-batch: coloring-used vs coloring-judged vs coloring-deployed. Any row where
   these disagree is a candidate re-render. Start with the heads currently in use.
   *(q4_window_corpus is now consistent — location-profile end to end.)*

### TODO P5 — (Optional, larger) Single render entry with a quality switch
Fold bare `render` into `render-one` as `--quality preview|wallpaper` (or make `render` a
thin alias setting ss2/box), removing the dual-entry location-profile duplication and the
density-default split at the source. **Invasive** — touches every render call site and
batch script, and risks repro churn. **Do not do speculatively;** revisit only if a third
quality tier appears or the density split (P3) forces a unification anyway.

### TODO P6 — Consider unifying the palette namespaces (small–medium)
P1 made the *resolvers* consistent, but N1/N2/N3 are still three disjoint sets. Options,
in preference order: (a) leave as-is now that every resolver falls back to N1 — cheapest,
already removes the foot-gun; (b) if a file-based path (e.g. `colormap.py`, which has **no**
built-in fallback) needs `default`/`cubehelix`, add a built-in fallback there too rather
than polluting the curated JSONs; (c) do **not** inject built-ins into
`score3_colormaps.json` — it is the *score-3 curated subset*, and `default` has no
provenance in it. Note `colormap.py` producing `default` would still be the *beautiful*
look, not the banded one, so this is lower value than it looks.

## For future CC instances — quick reference

- **"Why does my recolored crop look flat/washed-out vs the montage?"** You are almost
  certainly coloring a dumped field via `colormap.py` (beautiful, pct-stretched) and
  comparing to a location-profile (`render`/`sheet`) banded render. They will never match.
  Render full-frame via bare `render`/`sheet` for the banded look.
- **Want the banded UF `default` look for crops/thumbnails:** bare
  `render --palette default` (density 0.025) or `sheet --builtins default`, then crop.
  Not `colormap.py`, not `render-one --coloring`.
- **`render-one --palette default`** now works (P1) but renders at density **0.004**
  (looser bands than the montage) — add `--n-cycles 6.25` to match 0.025.
- **Two colorings, by name:** *location-profile* = `coloring::shade` (banded; bare
  `render`/`sheet`/`render-one`-default). *beautiful* = `render_modes` + `tools/colormap.py`
  (pct-stretched gradient; `render-one --coloring`, and all dump-field recolors).
- **`default`/`cubehelix` are Rust built-ins only** — absent from the colormap JSONs.
  `viridis` is in all three namespaces.
- **Before adding a new render/label/recolor path:** decide which of the two colorings it
  uses and confirm it matches the look the human will judge and the deploy path will
  render (P4). Write it down.

## What I would not change
- The **location-profile vs beautiful** distinction itself is legitimate — two real
  aesthetics, both wanted. The problem is discoverability/defaults, not that both exist.
- **Frozen flag names and default values** — batch reproducibility depends on them.

## One-line verdict
Not "`render-one` is redundant" — rather **one palette name resolved inconsistently (now
fixed, P1), two colorings share the word "default" (now documented, P2), the density
default silently forks 0.025/0.004 (P3), and the corpus may be labeled in a coloring the
deploy path won't reproduce (P4 — the follow-up that matters most).**
