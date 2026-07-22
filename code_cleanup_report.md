# Code-cleanup implementation report (2026-07-22)

Acting on the improvement candidates in `code_cleanup.md`. The guiding constraint was the
repo's reproducibility contract (CLAUDE.md: flag names, defaults, and pipeline behavior are
load-bearing for batch reproducibility), so every change here is **additive and
behavior-preserving on existing runs**: new opt-in flags default to the old behavior,
docstrings add no runtime effect, the new assertion fires only on genuine corruption, and the
one new consumer kwarg is output-identical (just faster). `cargo`-side code is untouched.

## Done

### #2 — Enforce global id-uniqueness (assert)
`EmissionDiversity._load_all_admitted` (`tools/emission/build_emission_diversity_v1.py`)
previously deduped every duplicate id silently — the exact silent-corruption landmine the
retrospective flags (run-scoped `st_<fam>_<arm>_<seq>` ids are reused across campaigns for
*different* locations; a union-by-id drops distinct wallpapers). It now compares the row's
**location identity** (`outcome_cx/cy/fw` + `julia_c_re/im`) on a duplicate id:
- **same id + same location** → legitimate cross-ledger overlap, deduped first-wins (unchanged).
- **same id + different location** → `SystemExit` naming both locations and pointing at the
  `c1__`-prefix remedy in `stage_first_release.py`.

This is the precise guard: it never fires on the intended dedup (the pre-existing
`test_multi_ledger_intake_dedup_and_source_tag` still passes), only on real collisions.
New test: `test_intake_raises_on_run_scoped_id_collision`.

Note on the schema pointer: the retrospective suggested a note in
`data/label_corpus/CORPUS_SCHEMA.md`, but that schema governs the label-corpus `image_id`
space (declared *batch*-unique), not the discovery **ledger** id space where the collision
lives. The invariant is instead documented where the union actually happens — the
`_load_all_admitted` docstring — and enforced by the assertion above, which is the substantive
half of the suggestion.

### #3 — Document the driver's snapshot-reuse contract
Added a docstring to `EmissionDiversity.intake()` spelling out the previously-undocumented
reuse contract: the `{cluster_tags, fields, n_admitted}` shape of `intake.json`, the
`descriptor._save_embs` shape of `morph_embs.npz`, that `cluster_tags` is the authoritative
membership set (newer frontier admits are deferred), and that `stage_first_release.py` writes
exactly these two files. Removes the reverse-engineering step the retrospective spent time on.

### #4 — Resume reuses the run's own ranker cache
`_score_intake_with_ranker` constructed `LocationRanker()` with only the global
`DEFAULT_FEATURES` cache, so every resume re-embedded all ~1.4k tiles. It now passes the run's
own persisted `ranker_feats.npz` as an additional cache
(`feature_caches=(DEFAULT_FEATURES, self.ranker_feats_path)`). `_load_cache` already skips a
missing path, so first runs are unaffected; on resume it is a pure cache hit. Output-identical
(features are deterministic and the cache was written from the same compute) — this only saves
minutes per resume.

### #5 — First-class `--cover-all` mode
Added `--cover-all` (default off): colorize every admitted location exactly once, then stop.
It replaces the awkward `--target-gated 100000000 --max-attempts 1387` encoding that invited
the round-robin double-dip. Semantics are explicit and self-terminating: the surplus-building
target/attempt/time cutoffs are bypassed, and the loop stops the instant `pick_location`
(fewest-attempts-first) returns an already-attempted row — i.e. every location has one pass.
Guaranteed to terminate in ≤ N iterations regardless of backstops.

### #7 — Selection cross-head mixing guard (documentation)
The concrete failure (82 release-eligible strange tiles shut out) comes from `greedy_select`
degenerating to top-N-by-absolute-score under singleton niches, comparing the wallpaper and
mining heads' incommensurable `p_ge3`. Making the tie-break head-aware would change selection
output (reproducibility), so this is the retrospective's "at minimum, document" option: a
CAVEAT block in `selection.py`'s module docstring explaining the incommensurability and the
partition-by-head + quota remedy, plus a cross-reference in `EmissionDiversity.select_release`.

### #8 — Run-control helper (`tools/kill_run.py`)
New standalone tool that ends the `pwsh`-self-match footgun. It enumerates processes
(`Get-CimInstance Win32_Process` on Windows, `ps` on POSIX), keeps only interpreter processes
(`python`/`uv`; `pwsh`/`powershell` only with `--include-shells`) whose command line contains
the pattern, and **excludes this process and its entire ancestor chain** before killing.
Dry-run by default; `--apply` kills. The selection/ancestor-walk logic is a pure function
(`select_targets` / `ancestor_pids`) with a committed unit test (`tools/test_kill_run.py`,
4 tests, no real processes spawned or killed), per the "tests belong in the suite" convention.

## Not done (deliberate)

### #1 — Canonical "current library" object
The biggest lever, but a genuine **design decision**, not a mechanical cleanup. A
`tools/emission/library.py` exposing `admitted_rows()`/`cluster_tags()`/`snapshot()` would
have to re-encode the exact union + `c1__`-prefix + per-family cluster-offset logic that
`stage_first_release.py` currently owns; building a second source of that truth risks silent
divergence from the working, already-run staging path. It's worth doing deliberately when the
*next* release is planned (so the abstraction is validated against a real second consumer),
not speculatively now. Left for that.

### #6 — De-duplicate the two intake modules
`library_intake_2.py` reuses `campaign1_intake.py` by mutating its module globals
(`c1i.LEDGERS = ...; c1i.OUT = ...`) — genuinely fragile. But both modules are "done": their
outputs are committed and the library is already built. Extracting an `intake_core` that takes
an explicit config is a medium refactor of reproducibility-critical code with **no existing
test coverage of their end-to-end output**, so the downside (breaking committed artifacts)
outweighs the forward value (they are unlikely to run again as-is). Left until a change forces
one of them to be re-run, at which point the refactor can be validated against the prior output.

### #9 — `render_release` resume
Already fixed in the session that wrote the retrospective — per-file resume (reuse a
verified-complete PNG, re-render only truncated/missing tiles) is live at
`render_release` in `build_emission_diversity_v1.py`. Nothing to do; verified present.

## Verification

- `uv run pytest tools/test_kill_run.py tools/emission/test_emission_diversity.py -q` →
  **24 passed** (includes the 2 new tests: id-collision guard, kill_run selection).
- `--cover-all` present in `build_emission_diversity_v1.py --help`; all touched modules import
  cleanly (`build_emission_diversity_v1`, `selection`, `stage_first_release`, `score_locations`,
  `tools.kill_run`).
- No Rust code touched.
