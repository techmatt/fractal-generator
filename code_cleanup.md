# Retrospective — first_release production run (2026-07-22)

Session task: execute `prompts/first_release.md` — a library-scale production run of the
five-stage emission pipeline. No new engine features; the work was almost entirely
*orchestration + understanding*, which is exactly where the friction showed up.

## Where the time went

**1. Understanding existing behavior dominated — by a wide margin.** I read ten modules
before writing a line (`build_emission_diversity_v1.py`, `descriptor.py`,
`campaign1_intake.py`, `library_intake_2.py`, `pool.py`, `cells.py`, `selection.py`,
`report.py`, `score_locations.py`, plus three memory files). The hard part was never any
single module — it was reconstructing *how they compose at run scale*:
  - How the driver's `intake()` reuse branch consumes a pre-staged snapshot (`intake.json`
    `{cluster_tags, fields, n_admitted}` + `morph_embs.npz`). This is nowhere documented; I
    reverse-engineered the exact dict shape from the ~15 lines of the reuse `if` branch, and
    the whole staging script is downstream of that reconstruction.
  - The colorize loop's stop conditions (`target_gated` vs `max_attempts` vs `exhausted` vs
    time budget) and how "colorize the whole library" maps onto them. I reasoned through the
    `pick_location` round-robin / `rec is None` / double-dip interaction **three separate
    times** before settling on `max_attempts = n_admitted` + huge `target_gated`.
  - That the measure (`target_measure.json`) is *pinned* to `library_intake_2`'s cluster ids
    (`phoenix#196..236`) — the single fact that determined the entire library-scoping and
    namespacing strategy. It's captured in the measure's `_phoenix_note`, which I only found
    by reading the JSON in full.

**2. Locating code was cheap; discovering *data* facts was expensive.** grep/read found the
modules fast. But the load-bearing facts lived on disk, not in code, and each needed a bespoke
probe: the two intake passes' admitted counts, the **60-id collision** (and that the colliding
ids are *different locations*, not dups), that `features.npz` covers 0/1327 library rows, that
all six ledgers are v7 (so no natural stale row for the acceptance proof). Every one of these
changed the plan, and none were discoverable without writing throwaway Python against the
ledgers/intake JSONs.

**3. The id-collision discovery forced a mid-course redesign.** I had a complete staging plan
(union-by-id) written in my head and half-drafted before the disk probe revealed the 60
collisions are distinct locations. That invalidated union-by-id and forced the `c1__` prefix
scheme (prefixed ledger copies + prefixed snapshot + per-family cluster-index offsets). Real
rework, entirely because global id-uniqueness is assumed but not enforced.

**4. Running & babysitting was a long tail of small frictions.** The actual edits (staging
script, readout, the `render_release` resume fix) were quick once understood. What ate cycles:
process management (the `pwsh`-matches-its-own-command-line gotcha killed my first cleanup
call; two python children where I expected one), monitoring a ~hours-long run whose rate I
mis-estimated 4× (CUDA warm-up inflated my first sample to 11 s/attempt), and re-arming
completion watchers that the harness kept sweeping.

## What I re-derived more than once

- **The six ledger paths.** Typed out three times — in `stage_first_release.py`, the launch
  command, and `supervise.sh` — with no canonical source. The two intake modules each hardcode
  their own `LEDGERS` list too, so the "current library" is defined in four places.
- **The admitted-set predicate + intake artifact layout.** `campaign1_intake.py` and
  `library_intake_2.py` are near-duplicates (the latter literally redirects the former's module
  globals: `c1i.LEDGERS = ...; c1i.OUT = ...`). I read both nearly end-to-end to confirm they
  cluster identically, because the near-duplication meant I couldn't trust one to stand for both.
- **The field-stem recipe.** `store.field_stem(loc, "smooth", la.W, la.H, la.SS)` — grepped for
  it, then had to confirm `la.W/H/SS = 640/360/2` separately. This same tuple is re-established
  in `descriptor.embed_locations`, both intake modules, and my staging script.
- **The colorize stop-condition semantics** (noted above) — re-reasoned three times.
- **The process-kill filter.** The `CommandLine -like '*build_emission_diversity_v1*'` pattern
  matches the *querying* shell too; I hit that once, fixed it with a `Name -in` guard, then had
  to re-apply the same guard in the final cleanup call.

## Improvement candidates (small → medium)

1. **A canonical "current library" object** — *the biggest lever.* The entire
   `stage_first_release.py` exists because there is no first-class notion of "the library." Add
   `tools/emission/library.py` (or `data/library/manifest.json`) that names the constituent
   intake passes + their ledgers and exposes `admitted_rows()`, `cluster_tags()` (merged +
   namespaced), and `snapshot()` in the driver's reuse format. Every downstream consumer (this
   run, the next release, any audit) stops re-deriving the union. Medium.

2. **Enforce or assert global id-uniqueness.** Run-scoped ids (`st_<fam>_<arm>_<seq>`) collide
   across campaigns for *different* locations — a silent-corruption landmine for any union-by-id
   (the driver's `_load_all_admitted` dedups by id and would have dropped 60 distinct wallpapers).
   Fix at the source (prefix ids with a run/campaign token at ledger-write time), or at minimum
   add an assertion in the union path and a one-line invariant note in `CORPUS_SCHEMA.md`. Small
   (assert) to medium (re-id).

3. **Document the driver's snapshot-reuse contract.** The `{cluster_tags, fields, n_admitted}`
   + `morph_embs.npz` shape is load-bearing and undocumented. A short docstring on
   `EmissionDiversity.intake()` (or a tiny `Snapshot` loader) would remove the reverse-engineering
   step. Small.

4. **Resume should reuse the run's own ranker cache.** `LocationRanker()` defaults
   `feature_caches=(DEFAULT_FEATURES,)` and never loads the run's persisted
   `out/<run>/ranker_feats.npz`, so every resume re-embeds all ~1387 tiles. Have the driver pass
   its own `ranker_feats_path` as an additional cache. Small, saves minutes per resume.

5. **A first-class "cover the whole library" mode.** "Colorize every location once, then stop"
   had to be encoded as `--target-gated 100000000 --max-attempts 1387`, which invites the
   round-robin double-dip edge case. A `--cover-all` flag with explicit one-pass semantics would
   be clearer and self-documenting. Small.

6. **De-duplicate the two intake modules.** `library_intake_2.py` mutating
   `campaign1_intake`'s module globals to reuse its primitives is fragile (import-order-sensitive
   global state). Extract the shared machinery into an `intake_core` that takes an explicit
   config object; both callers pass a config instead of monkey-patching. Medium.

7. **Selection cross-head mixing needs a guard.** Under singleton niches (the norm at one
   colorize/location) `greedy_select` degenerates to top-N-by-absolute-score, which compares the
   two heads' incommensurable `p_ge3` and shut out all 82 release-eligible strange tiles.
   Either add per-head selection quotas or make the tie-break head-aware; at minimum document
   that the niche-percentile + coverage machinery only bites with >1 colorize per cell. Small
   (doc/quota) to medium (redesign).

8. **A tiny run-control helper.** A `tools/kill_run.py` (or documented snippet) that filters
   processes by `Name in {python,uv}` AND commandline, excluding the caller, would end the
   `pwsh`-self-match footgun. The supervisor `until`-loop pattern I wrote is reusable too — worth
   promoting to a small `tools/supervise.sh` template rather than inlining per run. Small.

9. **`render_release` resume — fixed this session.** It re-rendered all N on any relaunch (a
   reaper kill mid-render could loop forever). Now per-file (reuse a verified-complete PNG). Noting
   it here because it was a latent gap in a "resume-safe" pipeline that only surfaced under a real
   multi-hour render tail.
