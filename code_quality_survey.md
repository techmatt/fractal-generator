# Code-quality survey — guidance for a future Claude Code run

Written after building the labeling-query assembler (`tools/queries/`,
commit `0786331`). These are concrete refactors that would have made *that* task
faster and will help the next person touching the palette / corpus / coloring code.
Each item is backed by friction actually hit this run. Ordered by payoff.

**Guardrails for every item below:**
- `tools/colormap.py` is Rust-parity-validated. After any change to it, run
  `uv run python tools/colormap_acceptance.py` and confirm `PASS` (≤1 LSB). This is
  the real gate — see next item, pytest can't run.
- Batch reproducibility depends on `render-one` flag names/defaults and on the
  `sample_candidate` / `compose_query` param ranges. Don't rename flags or change a
  default value without saying so loudly; keep named constants as the single knob.

---

## 1. Unify the colormap-file schema — `mirror_needed` is missing from the pool

**Evidence.** Three overlapping colormap files with *inconsistent* schemas:
- `data/palettes/score3_colormaps.json` (76) — has `mirror_needed`
- `data/palettes/clean_colormaps.json` (224, the `render-one` default) — has `mirror_needed`
- `data/palettes/pool_colormaps.json` (777, the assembler's pool) — **no `mirror_needed`** (has `score` instead)

`colormap.PaletteLibrary.lut()` reads `cm.get("mirror_needed", False)`, so feeding it
the pool silently renders every sequential palette **seamed**. This run had to inject
`mirror_needed = (cycle == 'sequential')` in `query_sampler.load_pool_library()` — a rule
re-derived by reading score3 and eyeballing that it held. That derivation is now
duplicated knowledge sitting in a helper, easy to get wrong next time.

**Fix (pick one):**
- Preferred: have `tools/palettes/build_pool.py` emit `mirror_needed` into
  `pool_colormaps.json` so all three files share one schema. Then delete
  `load_pool_library`'s injection loop and let callers use `PaletteLibrary` directly.
- Or: move the `cycle=='sequential'` fallback *into* `PaletteLibrary.lut()` (mirror it
  on `palette_type`'s existing "fall back to `cycle`" pattern) so no caller ever injects.

Verify with `colormap_acceptance.py` and by re-rendering one sequential palette
(e.g. `magma`) seam-free.

## 2. Make the test suite runnable — `pytest` is not installed

**Evidence.** `tools/test_colormap.py` and `tools/palettes/test_palette_features.py`
both `import pytest`, but `pytest` is in neither `pyproject.toml` nor `uv.lock`
(`grep -i pytest` is empty). `uv run python -m pytest` → `No module named pytest`.
The only reason coloring correctness could be checked this run was the standalone
`colormap_acceptance.py` script.

**Fix.** `uv add --dev pytest` (keep it out of the runtime stack). Then wire the
acceptance check as a test so `uv run pytest tools/` is the one command that gates
coloring changes. If pytest is deliberately excluded, add a one-line note at the top
of each `test_*.py` pointing to the runnable acceptance script, so a future run
doesn't waste a cycle discovering it can't run the tests.

## 3. Two sources of truth for "is this palette cyclic?" — document/centralize

**Evidence.** Cyclic-ness is decided by two different fields that disagree in count:
- `palette_features.json` `type` ∈ {cyclic, non_cyclic}: **615 / 162**. Drives
  `phase`/`n_cycles` applicability (`colormap.validate_config`).
- colormap-file `cycle` ∈ {cyclic, sequential}: **619 / 158**. Drives `mirror_needed`.

They differ because 2 palettes are `type=non_cyclic` but `cycle=cyclic`. This run had
to reason through which governs what to avoid handing a cyclic-only knob to a palette
the coloring path considers non-cyclic (would raise). A future edit could easily use
the wrong one.

**Fix.** A short docstring/table in `colormap.py` (or `palette_features.py`) stating:
*type governs coloring knobs; cycle governs the mirror seam-fix; they are not
interchangeable.* Optionally assert the two never disagree in a way that breaks a
render, or reduce to one field if the 2-palette discrepancy is spurious.

## 4. Reuse the existing corpus reader instead of hand-rolling batch scans

**Evidence.** `query_sampler.LocationPool.from_corpus()` hand-rolls a
`glob('batches/*/images.jsonl')` + JSON-parse + score-filter loop. But
`tools/corpus/corpus_reader.py` already has `iter_labeled()` yielding
`LabeledCrop(crop_path, score, image_id, batch_id, render)` for exactly the non-null
labeled rows, plus `count_pairs()` for a per-batch census. The pool loader is a filter
(`score in {2,3}`) + dedup on top of that.

**Fix.** Rebuild `LocationPool.from_corpus` on `corpus_reader.iter_labeled` so batch-
reading lives in one place. Watch one gap: `iter_labeled` yields `render` but confirm
it surfaces `fractal_type`/`c_re`/`c_im` (needed for Julia locations) — if not, extend
it there rather than bypassing it. This also means future schema changes to
`images.jsonl` are absorbed in one reader, not N.

## 5. Kill the `sys.path.insert` bootstrap boilerplate across `tools/`

**Evidence.** Nearly every module under `tools/` opens with 1–3
`sys.path.insert(0, ...)` lines for `tools/`, `tools/palettes/`, `tools/corpus/`,
`tools/mining/` (see `mining/*.py`, `eda/*.py`, `palettes/build_features.py`). There's
no package structure, so imports are position-dependent. This run introduced a
duplicate `import query_sampler` while wrestling with path setup (caught and removed).

**Fix.** Add `tools/__init__.py` (+ subpackage `__init__.py`s) and/or a tiny
`tools/_bootstrap.py` that does the path setup once, imported first. Or declare
`tools` as a package in `pyproject.toml` so `from tools.corpus import corpus_reader`
just works under `uv run`. Low risk, high readability win; do it opportunistically
when next editing a `tools/` module, not as a big-bang rename.

## 6. (Perf, optional) An f32 lookup/downsample lane for the sweep

**Evidence.** ss2/1024 recolor is ~785ms mean, dominated by `lookup_linear` (~280ms,
float64 LUT gathers) and the box downsample (~220ms, float64). The prompt wanted
"well under a second" and the sweep will do *thousands* of recolors per field. This run
added the correct config-independent cache seam (`colormap.stretch_field` /
`StretchedField`) but did **not** touch precision, because the box path is documented
byte-identical to Rust and the parity test runs in float64.

**Fix (only if the sweep demands it).** Add an opt-in f32 path for `lookup_linear` +
downsample (roughly halves both, → ~450ms), guarded by a flag so the default float64
Rust-parity path is untouched. Gate the change on `colormap_acceptance.py` still
passing *for the float64 default*, and add a separate looser tolerance check for the
f32 lane. Alternatively the sweep just uses ss1 (already ~220ms per the field-colormap
notes). Don't do this speculatively — it's precision-sensitive.

---

### What was already good (don't "fix")
- The field⊗colormap split (`render-one --dump-field` → `colormap.render_candidate`)
  made "one coloring path, cache the field, recolor cheaply" fall out naturally. The
  assembler reused it with zero changes to the render core.
- `palette_features.distance_matrix` / `farthest_point_order` were exactly the diversity
  primitives the stratified sampler needed — reused, not reinvented.
- Named-constant discipline in the existing subcommands made the sampler's tunables
  (`QUERY_SPLIT`, `CURATED_WEIGHT`, gamma ranges) an obvious pattern to follow.
