# Restructure — execution plan (read before anything moves)

Rebased on repo state at `ca24896`. Ordered by **when a mistake surfaces**
(loud-early → loud-late → cold tail), not by blast radius.

**Verification defects settled (this revision).**
- **#1 — the `--collect-only` arithmetic couldn't fail.** All three test-glob files
  in scope (`corpus/smoke_test.py`, `palette_extractor/test_roundtrip.py`,
  `test_wallpaper_extraction.py`) define **0 `test_` functions and 0 top-level
  asserts** (verified) — invisible 0-item false-greens. Relocating/renaming them
  changes collection behaviour not at all, so there is no positive signal to
  assert. **Dropped the arithmetic and the `--ignore=tools/studies` config**
  (it fixes an invisible non-problem, can't be verified). Verification is now a
  **regression guard**: `pytest --collect-only` exits 0 with a node-ID set
  **identical** to the pre-move baseline (expected delta: 0).
- **#2 — import-smoke only verifies module-level path math.** Checked per file.
  **All 11 studies-roster files are depth-2 `tools/<dir>/x.py` origins**, so a
  **flat `tools/studies/x.py` destination preserves depth** — `parents[1]`/`parents[2]`
  still resolve to repo root, unchanged. So **no studies file needs a `parents[N]`
  edit; the whole set moves byte-identical.** 9 of 11 carry module-level path math
  (import-smoke genuinely exercises the unchanged resolution); the other 2
  (`compare_v1_v2_diversity`, `morphology_dedup`) have none, so import-smoke just
  confirms they load. The `parents[N]` fix is eliminated by the depth-preserving
  destination — gap B and defect #2 are both moot for studies. Files that *would*
  need an edit are pulled out (below), not edited.

## Already done — dropped
- **T0-a** — `data/library/`, `data/library_embeddings/` committed `9b2f7b5`.
- **T0-b** — the three pref-labels restored from `ef719a6^`, tracked.

## Cut — gone
- **T3-a** (four Oklab copies → silent drift) → note N-c.
- **T2-d** (enrich bridge = two contracts, `HDR` owned by `enrich.rs`) → note N-b.
- **T2-e** (`colormap.py` → `tools/color/`: ~40 importers, cosmetic).

---

## 1 — Notes & docstrings (loud-early or n/a; independent, do first)
- **N-a. Header note** on `tools/palettes/build_features.py` (pool file now owned by
  `build_pool.py`). Dropped the `corpus_data.py` note — the re-audit shows it's on
  the **live v6 path via `train_v2`**, not superseded. *Class:* n/a.
- **N-b. T2-d cross-ref notes** in `corpus/enrich_score.py` and `mining/score_lib.py`.
- **N-c. T3-a contract note** at `tools/palettes/color.py`.
- **N-d. Fix stale Rust docstrings** in `src/probe.rs`, `src/palette_pick.rs`.
  *Class:* loud-early. *Verify:* `cargo build --release --lib`.
- **N-e. Relocate `beautiful_locations.json`** out of repo root (no code reader).
  *Class:* loud-early. *Verify:* grep (empty) → move.

## 2 — Mechanical (loud-early)
- **T1-a (reduced to two renames).** `palette_extractor/test_roundtrip.py`,
  `palette_extractor/test_wallpaper_extraction.py` → `harness_*.py`. They stay in
  place (repo-root subsystem, not `tools/`-intermixed) but shed the misleading
  `test_` prefix. (`smoke_test.py`, the third original target, is dead → archive
  in §3, so it's handled by the move, not a rename.) *Buys:* the tree stops naming
  non-tests `test_*`. *Class:* loud-early. *Verify:* grep no importers +
  `pytest --collect-only` node-set unchanged.
- **T1-c. Delete the two vanished-scratchpad comment refs** (`production_seeder.py`,
  `build_categories.py`). *Class:* n/a.

## 3 — The quarantine move (T2-a-asymmetric ⊕ T3-b, ONE operation)
Flat `tools/studies/` for closed studies, `tools/studies/archive/` for confirmed-dead.
**The entire move is byte-identical** (no content edits either tier); the tiers
differ only in how they verify. Archiving is not deleting — needs no authorization,
leaves the deletion pass unopened.

### → `archive/` — the 8 confirmed-dead (`inventory_followup.md` §2.3)
Move byte-identical. Depth increases (`tools/<dir>/x.py` → `tools/studies/archive/x.py`,
+2 levels), so their `ROOT` resolves wrong — **moot, nobody runs them** (three are
already broken: `smoke_test` fails on run, `selector_family_diversity_sweep` crashes
at import, `exp_vs_smooth_rankcorr` acted-on). Depth delta recorded in
`tools/studies/archive/README.md`. *Verify:* `git log --follow` shows rename-only.

`eda/eda_common.py`, `corpus/smoke_test.py`, `wallpaper/selector_family_diversity_sweep.py`,
`wallpaper/emission_dryrun_v2gate.py`, `palettes/cliff_diag.py`, `palettes/softcliff.py`,
`palettes/render_v2_batch.py`, `render_mode_pilot/exp_vs_smooth_rankcorr.py`.

(Corrects the prior archive-9: removed the 4 §2.4-**ambiguous** files → studies;
added the 3 §2.3-**dead** ones previously misfiled/omitted.)

### → `studies/` — 7 ambiguous closed one-shots (§2.4), flat, byte-identical
Depth preserved (depth-2 origin → `tools/studies/x.py` depth-2), so path math
resolves unchanged. *Verify:* **per-file fresh-process** import-smoke (see the
methodology note). Final set: **7/7 clean in isolation.** No collisions.

`queries/compare_v1_v2_diversity.py`, `curation/{conditioned_colorize,soft_spread_calibrate,morphology_dedup}.py`,
`atlas/gather_overnight.py`, `looksee/looksee_27_walks.py`, `palettes/viz_transfer.py`.

**Verification-methodology correction (this mattered).** The first import-smoke
loaded all files in ONE process, so an earlier file's `sys.path.insert` /
`sys.modules` pollution gave later files false OKs — it reported 9/9 when 3 were
actually broken. Redone **one fresh process per file**, three files failed on
`import <live sibling>` and were pulled out:
- `cross_family_shakeout` → `import prescreen` (live atlas sibling; via `HERE`)
- `verify_v6_gate` → `import guard` (live atlas sibling; via `HERE`)
- `v5_v6_anchor_diff` → (transitively) `guard`
All three reach a live sibling through `sys.path.insert(0, HERE)`, which depth
preservation does NOT fix (the sibling stays in `tools/atlas`, `HERE` moves). Same
class as `gvo_experiment`. `looksee_27_walks` also `import prescreen` but resolves
it via a `ROOT`-based insert (`ROOT` preserved) — genuinely fine, confirmed in
isolation.

### Pulled OUT of the move (would break or need an unverifiable edit)
- **`atlas_probe/` cluster — LEFT IN PLACE, reclassify `step0_reanalysis`+`step0` → live.**
  The running seeder imports `step0_reanalysis` (§gap-A). Moving it breaks the live
  reward path.
- **`queries/scorer/gvo_experiment.py` — LEFT IN PLACE.** Its `sys.path.insert(0, HERE)`
  + `import data/train/surfacing_eval` reaches **live sibling modules in `scorer/`**;
  moving it breaks those imports regardless of depth. Reaching back into live would
  need a fragile edit for ~0 gain.
- **`atlas/{cross_family_shakeout,verify_v6_gate,v5_v6_anchor_diff}.py` — LEFT IN
  PLACE.** Each reaches a **live** atlas sibling (`prescreen`/`guard`) via
  `sys.path.insert(0, HERE)`; depth preservation can't fix a sibling that stays
  behind. Caught by fresh-process import-smoke; moved, failed, reverted. (Their
  T2-b `reframe_probe`→`scoring` edit is correct wherever they live and rides
  along.)
- **`palette_extractor/` ambiguous subset — LEFT IN PLACE.** Repo-root subsystem
  (depth 1), already demarcated and not `tools/`-intermixed; moving into
  `tools/studies/` changes depth for no tier-locality gain.
- **classifier leaves, `coevo/`, `descent_ablation/`, `reframe_probe/speed.py`,
  `render_mode_pilot` (non-dead), `v4/v5` montages — DEFERRED**, unadjudicated.

### Sweep & cross-references (self-contained; live tree untouched)
- **No `parents[N]` sweep needed** — depth preserved (studies) / moot (archive).
- **`emission_selector` — fully dissolved.** All its live consumers stay put and
  `emission_selector.py` stays → **zero live-tree edits**. Its one archived referrer
  (`emission_dryrun_v2gate`) moves byte-identical/cold; its one moved-studies
  "referrer" (`morphology_dedup`) is a **prose comment** (line 4), not an import.
- **Prose-pointer note (note-only):** `production_seeder` comments ("mirrors
  `cross_family_shakeout.*`") and three findings docs name moved files. Prose, not
  deps — leave a forwarding note.

**Whole-move verify:** full `uv run pytest` (green ⇒ live tree untouched: no live
module imports the *moved* set, atlas_probe/gvo excluded) + per-file import-smoke
over `studies/` + `git log --follow` over `archive/` + `pytest --collect-only`
node-set == baseline (regression guard, defect #1).

## 4 — T2-b: promote `probe.py` (loud-early)
`tools/reframe_probe/probe.py` → `tools/scoring/active_ckpt.py`. **After** the
quarantine so relocated consumers are rewritten once (not because the sweep shrinks
— it doesn't). *Class:* loud-early (`test_production_seeder`/`test_guard` import the
two heaviest consumers). *Sweep:* all 14 `sys.path.insert(...reframe_probe)` +
`from probe import` sites repo-wide, incl. the studies files now at `tools/studies/`.
*Verify:* full `uv run pytest`.

## 5 — T2-c: lift `top_k_pool` (loud-late) — LANDED
`build_humanq3.top_k_pool` → new neutral `tools/wallpaper/pool_rule.py`, imported
by `build_fresh_discovery`, `build_humanq3`, `build_headbatch_dramatic`.
*Verified:* direct proof that `build_fresh_discovery.top_k_pool IS
pool_rule.top_k_pool` with behaviour intact (`['a','b']`, ranks 1/2); `build_humanq3`
imports `pool_rule` clean with its dir on path (real-usage condition). No residual
`from build_humanq3 import top_k_pool` anywhere.
*Out-of-scope finding:* `build_headbatch_dramatic:112` reads `BFD.DEG2_FAMILIES`,
but `build_fresh_discovery` has **no `DEG2_FAMILIES` (0 hits at HEAD)** — that
builder was already stale against the current API, independent of this work
(my diff to it is import-only). Deferred NIBK, not test-imported.

---

## Verification ledger
| Step | Class | Verify |
|---|---|---|
| N-a/b/c | n/a | — |
| N-d | loud-early | `cargo build --release --lib` |
| N-e | loud-early | grep readers (empty) → move |
| T1-a | loud-early | grep no importers + `pytest --collect-only` node-set == baseline |
| T1-c | n/a | — |
| Quarantine `studies/` | loud-early | full `pytest` + per-file import-smoke |
| Quarantine `archive/` | cold | `git log --follow` rename-only; README depth-delta |
| T2-b | loud-early | full `pytest` |
| T2-c | loud-late | orchestrator `--mini` |

Execution begins now; each numbered step reported as it lands. Nothing staged.
