# Code inventory & restructuring proposal

**Read-only report.** Nothing was edited, moved, or deleted. A long GPU run
(`prospect_orchestrator.py --run`) was live throughout; only cheap read-only
commands (`git log`/`ls`/`grep`/reads) and parallel read-only exploration agents
were used.

**Method.** Reachability backbone (import edges, subprocess/`-m` edges, git
last-touched, `git ls-files`/`git check-ignore` for commit status) was computed
centrally; per-file purpose was gathered by area. **253 tracked source files**
(222 `.py` + 31 `.rs`) plus specs/config/docs. Buckets follow the prompt:
`live` / `test-only` / `not-imported-but-keep` / `dead` / `ambiguous`, with the
`not-imported-but-keep` bucket reserved for files that carry non-import value
(sole producer of a durable artifact, diagnostic/harness, irreplaceable text,
CLI/subprocess/CLAUDE-documented).

---

## 1. Headlines — act on these

1. **⚠ Two load-bearing `data/` artifacts are gitignored (the exact shape of the
   prior 3 losses), and one is the *already-once-lost* morph_clip data.** The live
   prospecting loop reads/writes them; neither is committed:
   - `data/library_embeddings/embeddings.npz` (2.3 MB) — the **recovered**
     morph_clip embeddings, appended per-cycle by the running loop.
   - `data/library/library_records.jsonl` (458 KB) — the durable location library.
   Root cause is a `.gitignore` allowlist gap, not intent (see §3). **Fix is one
   safe edit.**

2. **Almost nothing is confidently dead.** Exactly **one** file is a plausible
   delete (`tools/eda/eda_common.py`, ambiguous-leaning-dead). The repo is
   dominated by *sole-producer* scripts that look dead by import-graph but are
   the only regeneration path for a committed artifact — deleting on "nothing
   imports it" would repeat the prior losses. The correct lever here is
   **reorganize + commit-the-at-risk-data**, not prune.

3. **Your entry-point list is incomplete in a few load-bearing ways** — the
   emission *orchestrator*, the disk-audit purge, `render-one` as a
   heavily-subprocessed engine entry, the Rust↔Python parity guard, the model
   trainers, and the palette-library regen path are all additional roots (see §2).

4. **Structural debt is concentrated in three patterns:** (a) `tools/` mixes live
   pipeline, per-version sole-producers, and closed one-shot studies with no tier
   separation; (b) heavy *copy-forward* version lineages (v4→v5→v6, classifier
   v2→v6, scorer train→v3→v3_gvo) read as duplication but are intentional
   provenance; (c) the Oklab/coloring math exists in **four** hand-synced copies.

5. **Two committed scripts cite scratchpad producers that no longer exist**
   (`production_seeder.py` → `scratchpad/jm{3,45}_tgood_sweep.py`;
   `build_categories.py` → `scratchpad/palette_categories/stability.py`). The
   committed *outputs* survive under `labels/`, but the provenance scripts are
   gone — the recurring anti-pattern, already realized.

---

## 2. Entry points & reachability

### Confirmed live roots (your list)
- **Prospecting loop (RUNNING):** `tools/wallpaper/prospect_orchestrator.py --run`.
  Per cycle it `import overnight_orchestrator as oo` (shared helpers) and
  subprocess-spawns `production_seeder.py` (`--run`/`--run-phoenix`) →
  `build_fresh_discovery.py` → `library_annotate.py` (annotate/persist tail).
- **Emission loop:** `overnight_orchestrator.py --run` → `production_seeder.py` →
  `build_fresh_discovery.py` → `emit_v1.py`; post-emission tail `deploy_tail.py`.
- **Labeling bridge:** Rust `enrich --mode score` → `tools/corpus/enrich_score.py`.
- **Rust engine:** `src/main.rs` dispatch (10 subcommands + bare render).
- **pytest:** `tests/*.rs` + `test_*.py`.

### Additional roots found (your list was incomplete here)
- **`overnight_orchestrator.py` is itself a root**, not just the chain of stages —
  it owns the GPU-decoupled-phase / wall-cap / purge machinery that
  `prospect_orchestrator` imports wholesale. (The running process is the
  *prospect* variant, which has **no emit phase** — a newcomer must not assume
  the live run produces wallpapers.)
- **`tools/audit/disk_audit.py`** — invoked by `overnight_orchestrator` (the
  SCRATCH/REGEN purge); CLAUDE/memory-documented. Live.
- **`render-one` (Rust subcommand)** — not just a CLI convenience; it is
  subprocess-invoked by ~a dozen tools (`deploy_tail`, `colored_clip`,
  `render_mode_pilot/*`, `descent_ablation`, palette viz, explorer). A primary
  engine entry point.
- **`dump-julia-bands` (Rust) ↔ `tools/atlas/test_julia_bands_parity.py`** — a
  CI-style Rust↔Python band-parity guard.
- **Model-producing roots** (each the sole producer of a weights dir, run via
  `-m`): `classifier.train_v6`, `classifier.train_wallpaper_v3`,
  `classifier.train_mining_head`, `tools/queries/scorer/train_v3_gvo.py`
  (the **ACTIVE** pref head).
- **Palette-library regen root:** `python -m palette_lib.build_sheet` — the sole
  regeneration path for the committed `data/palettes/clean_colormaps.json`
  library that ~18 live tools read.
- **Labeling UIs:** static `tools/viz/{corpus,location,wallpaper}_label.html`
  (export `scores.json` → `merge_scores.py`) and the Flask
  `tools/queries/launch_query_label_server.py` (+ `query_label.html`) — the human
  label sources.
- **Interactive viewer:** `tools/explorer/app.py` via `explorer.cmd` (shells to
  `render-one`).
- **Secondary/legacy discovery drivers** (standalone, still runnable):
  `tools/atlas/gather_overnight.py`, `tools/v6/monitored_harvest.py`,
  `tools/coevo/coevo_round.py`, `tools/descent_ablation/run_campaign.py`.
- **Manual curation ops:** `tools/mining/deploy_tail.py`,
  `tools/curation/recolor_pass.py`, `tools/curation/morphology_dedup.py`.

### Hub modules (many importers ⇒ live if any importer is live)
`tools/corpus/location.py` (~34 importers), `tools/corpus/corpus_common.py`
(~15), `tools/colormap.py` (~40), `tools/queries/query_sampler.py` (~15),
`tools/corpus/{corpus_reader,label_store}.py`, `tools/mining/{score_lib,mining_gate}.py`,
`classifier/{data,model,inference}.py`, `tools/reframe_probe/probe.py`
(the `ACTIVE_CKPT` + scorer hub — badly misnamed), `tools/queries/scorer/{data,train}.py`,
`palette_lib/coloring.py` (hub **only within** the palette-extractor subsystem —
the live side does not use it).

---

## 3. ⚠ Critical: uncommitted load-bearing `data/` artifacts

The `.gitignore` uses `/data/*` (disposable by default) plus an explicit
`!`-allowlist for every load-bearing store (palettes, `label_corpus`,
`wallpaper_corpus`, `queries/labels/*.json`, `atlas`, `discovery`, calibration,
test fixtures). **The Phase-1 library dirs and the curation dir were added after
that allowlist and have no exception**, so they are silently swept into the
ignore:

| Artifact | Size | Sole/primary producer | Committed? | Regenerable? |
|---|---|---|---|---|
| `data/library_embeddings/embeddings.npz` | 2.3 MB | `tools/curation/colored_clip.py` (`np.savez`), appended by LIVE `library_annotate.py`/`library_store.py`; tagged by `morph_producer_tag.py` | **NO** (gitignored via `/data/*`) | Only by re-running GPU CLIP over every location |
| `data/library/library_records.jsonl` | 458 KB | `tools/wallpaper/library_records_build.py` | **NO** | Needs a `morph_uids`/`uids` key remap on rebuild (its own docstring) |
| `data/curation/recolor_pass/recolor_assignments.jsonl` | — | `tools/curation/recolor_pass.py` | **NO** | Regenerable runtime state (weaker case) |

`git check-ignore -v` confirms all three match the `/data/*` rule at
`.gitignore:67`. This is precisely the "sole producer of a durable `data/`
artifact that isn't committed" shape that CLAUDE.md says caused the last three
losses — and `embeddings.npz` **is** the morph_clip data that was already lost
once and reconstructed by a formula sweep (per `docs/findings/morph_parity.md`
and MEMORY).

**Recommended fix (safe, one edit):** add allowlist lines mirroring the existing
convention, e.g.
```
!/data/library/
!/data/library_embeddings/
# (decide: commit recolor_assignments.jsonl, or accept it as regenerable runtime state)
```
then `git add` the two artifacts. Verify: `git check-ignore` returns empty for
both; `git status` shows them staged. This does not touch the running loop.

**Adjacent provenance gap (not the same flag).** The three allowlisted human
pref-label files (`data/queries/labels/{coldstart_v2,warmstart_v1,prefv2_dramatic_v1}.json`)
are **neither on disk nor committed in this checkout** (the `labels/` dir is
absent; `git ls-files data/queries` is empty). Those are the irreplaceable human
labels the ACTIVE pref-v3-gvo scorer trained on. If they exist only on another
machine, they are a single-copy human-work asset with no backup here — worth
confirming they're archived somewhere.

---

## 4. Classification summary

- **live (~65 files):** all 22 `src/*.rs`; the corpus hubs (`location`,
  `corpus_common`, `corpus_reader`, `label_store`, `enrich_score`,
  `gather_select`, `verify_render_path`); `classifier/{__init__,data,model,inference}`;
  the wallpaper orchestrators + annotate/pool tail (10); atlas `guard`/`prescreen`/`production_seeder`;
  mining `mining_gate`/`score_lib`/`tail_alloc`/`deploy_tail`; the queries inference
  layer (`query_sampler`/`sample_location`/`query_batch_gen`/`assemble_queries`/`scorer.data`/`scorer.train`);
  `reframe`/`reframe_probe.probe`; `colormap`/`colormap_acceptance`/`_bootstrap`;
  `disk_audit`; palettes `color`/`palette_features`; curation `colored_clip`.
- **test-only (~23):** 7 `tests/*.rs` + the `test_*.py` suite. Caveats: two are
  `slow`-gated real crash/tripwire harnesses (`test_shard_crash.py`,
  `test_guard_tripwire.py`); two `#[ignore]` Rust tests
  (`occupancy_parity.rs`, `palette_bytematch.rs`) are env/corpus-driven.
- **not-imported-but-keep (~163):** the bulk — per-version campaign builders (each
  the sole producer of a committed `data/vN` manifest/plan or a `label_corpus`
  batch), model trainers, closed one-shot studies whose findings are in
  `docs/findings/` or MEMORY, and diagnostics/harnesses worth re-running.
- **dead (0 confident) / ambiguous (1):** `tools/eda/eda_common.py` only (§8).
- **Near-broken but kept:** `tools/wallpaper/selector_family_diversity_sweep.py`
  crashes at import — it hard-loads `scratchpad/_stage4_cells.json`, which no
  longer exists (scratchpad-on-dependency-path anti-pattern).

Full per-file detail in §5.

---

## 5. Full per-file inventory

Format: `path` **[class]** — one-line _(touched; reach/evidence)_. `NIBK` =
not-imported-but-keep. Flags called out inline.

### src/ (Rust engine) — all **live**
- `lib.rs` — crate root; `ensure_parent_dir` + 22 `pub mod`. _(2026-06-26)_
- `main.rs` — CLI entry; dispatches 10 subcommands + bare render. _(2026-07-09)_
- `cli.rs` — clap arg groups, `Cli`/`Command` enum, parsers. _(2026-07-09)_
- `backend.rs` — `FractalBackend` trait + F64/Perturbation/Julia tiers, `PixelSample`. _(2026-07-04)_
- `coloring.rs` — separable `shade`: PixelSample→linear-RGB; `required_channels`. _(2026-06-22)_
- `energy.rs` — multi-scale OKLab edge-energy metric + `calibrate`; occupancy primitives reused by present/guided_descend. _(2026-07-04)_
- `enrich.rs` — `enrich` score/render bridge; streams RGB to stdout. _(2026-06-25)_
- `font.rs` — hand-rolled bitmap font for on-image labels. _(2026-06-20)_
- `generate.rs` — `generate` sampler; `AcceptBand`/`screen_stats`/`color_params` reused widely. _(2026-07-12)_
- `guided_descend.rs` — `guided-descend`+`dump-julia-bands`; the running seeder's engine. _(2026-07-10)_
- `hp.rs` — astro-float high-precision scalar (decimal parse, `to_f64`). _(2026-06-20)_
- `jsonl.rs` — single serde-free JSONL reader (consolidated the 6 drifted copies). _(2026-06-25)_
- `palette.rs` — OKLab cyclic gradients → linear-RGB LUT; builtins. _(2026-06-22)_
- `palette_io.rs` — `.ugr`/`.map` loaders + resolver. _(2026-06-20)_
- `palette_pick.rs` — survivor colormap-library loader; **stale docstring** advertises retired `palette-pick` subcommand (loader kept, used by 5 modules). _(2026-06-25)_
- `palette_probe.rs` — `palette-probe` subcommand. _(2026-06-25)_
- `present.rs` — `present` zoom+composition+black/occupancy gate render. _(2026-06-25)_
- `probe.rs` — shared depth-probe plumbing (`PERTURB_SPACING`, `render_mandel_panel`); **stale docstring** advertises retired `descend`/`navigate`. _(2026-07-04)_
- `render.rs` — `Frame` geometry + two-stage render; the `dc`-from-geometry rule. _(2026-06-26)_
- `render_modes.rs` — "beautiful modes" pipeline; 16 modes cataloged by `specs/*.json` (metadata only, not read at runtime). _(2026-07-11)_
- `render_one.rs` — `render-one` locked wallpaper quality; `--dump-field`, `--coloring`. _(2026-07-10)_
- `root_field.rs` — durable 8192² smooth field under `data/root_field/`; seeds guided-descend. _(2026-07-04)_
- `sheet.rs` — contact-sheet compositor. _(2026-06-25)_
- `v4_cache.rs` — `v4-render-batch` bulk executor. _(2026-07-05)_

### tests/ (Rust) — all **test-only**
`channel_dispatch.rs` (channel-intent dispatch complete), `modes_registry.rs`
(specs↔registry lockstep), `occupancy_parity.rs` (`#[ignore]`, loose0 corpus),
`palette_bytematch.rs` (`#[ignore]`, env-driven by `check_bytematch.py`),
`perturbation.rs` (perturb vs f64 ground truth), `separability.rs`
(re-color never re-iterates), `sheet.rs` (sheet separability + loader coverage).

### specs/ — all **NIBK** (registry is airtight: 16 spec files == 16 registry keys, enforced by `modes_registry.rs`)
`modes_registry.json` (source of truth) + `REGISTRY.md` (generated view) + 16
one-line mode specs (8 promoted: smooth, tia, stripe, smooth_mean_angle,
smooth_angle_min, composite_c7/c13/c17; 8 niche/deprecated incl. `exp_smoothing`,
kept as a documented pixel-dupe of smooth). Consumed by `tests/modes_registry.rs`,
`tools/specs/gen_registry.py`, and (as `render-one --coloring` args) by
`deploy_tail.py` + `render_mode_pilot/render_batch.py`.

### tools/corpus/
- `location.py` **[live]** — canonical `Location`, `location_key`, `render_one_flags`; ~34 importers incl. running seeder. _(2026-07-11)_
- `corpus_common.py` **[live]** — `RENDER_KEYS` row shape, `render_corpus_crop`, `is_v6_decoded`. _(2026-07-09)_
- `corpus_reader.py` **[live]** — version-blind cross-batch reader (`iter_labeled`). _(2026-07-01)_
- `label_store.py` **[live]** — label resolution SOT (`resolve_score`, sidecars). _(2026-07-01)_
- `enrich_score.py` **[live]** — Rust `enrich --mode score` → v2 scoring (no crops to disk). _(2026-06-24)_
- `gather_select.py` **[live]** — v6 gather selector; `render_family`/`outcome_geometry` used by live `build_fresh_discovery`. _(2026-07-05)_
- `verify_render_path.py` **[live]** — Guard B, rebuild-K-crops parity; `check_batch` called by live gather path. _(2026-07-05)_
- `build_enrich_batch.py`, `build_rev4_batch.py`, `enrich_select.py`, `import_loose0_v3.py`, `pool_to_locations.py`, `recolor_gather_v6.py` **[NIBK]** — per-batch one-shot builders; each sole producer of one committed `data/label_corpus/batches/<id>`. _(2026-06-24 / 07-05)_
- `merge_scores.py` **[NIBK]** — CLAUDE-documented `scores.json → images.jsonl` merge (null→value only). _(2026-07-05)_
- `rev4_sanity_sheet.py` **[NIBK]** — pre-label QA sheet. _(2026-06-24)_
- `smoke_test.py` **[NIBK]** — union-invariant harness; **misnamed** (not a pytest) and stale at the loose0/rev4 two-batch era. _(2026-06-24)_
- `test_corpus_reader.py`, `test_location.py` **[test-only]**.

### classifier/
- `__init__.py`, `data.py`, `model.py`, `inference.py` **[live]** — package + deploy transform/black-gate, MobileNetV4+CORN head, `load_scorer`. _(→2026-07-05)_
- `corpus_data.py` **[NIBK]** — crop-level loader for v2/v3 (**superseded** by `data_v4.py`). _(2026-06-24)_
- `data_v4.py` **[NIBK]** — location-level loader over the 42-render cache; **current** (v4→v6). _(2026-06-25)_
- `eval.py`, `eval_v2.py`, `sheet.py`, `diagnose.py`, `diagnose_v2.py` **[NIBK]** — shared metric libs + read-only census/report tools. _(2026-06-23/24)_
- `train.py`, `train_v2.py`, `train_v3.py`, `train_v4.py`, `train_v5.py`, `train_v6.py` **[NIBK]** — each sole producer of `data/classifier/v{1..6}`; **inheritance chain** (v5/v6 import v4's `train()` verbatim — v4 is load-bearing beyond its own weights). v6 = ACTIVE location gate. _(→2026-07-05)_
- `train_mining_head.py` **[NIBK]** — produces `data/render_mode_head/v1/` (the live `mining_v1` gate weights). _(2026-07-11)_
- `train_wallpaper_v1.py`, `train_wallpaper_v2.py`, `train_wallpaper_v3.py`, `train_wallpaper_resdiag.py` **[NIBK]** — wallpaper-head lineage; funnel through `train_wallpaper_v2.split_rows`; v3 = current. _(→2026-07-09)_

### tools/wallpaper/
- `prospect_orchestrator.py` **[live]** — THE RUNNING PROCESS (prospecting, no emit). _(2026-07-15, edited in working tree)_
- `overnight_orchestrator.py` **[live]** — emission loop + shared spine. _(2026-07-15)_
- `build_fresh_discovery.py` **[live]** — fresh-discovery pool front-end (POOL_BUILDER). _(2026-07-15)_
- `build_humanq3.py` **[live]** — human-q3 batch **and** home of the shared `top_k_pool` rule (mislocated in a batch builder). _(2026-07-09)_
- `label_crop.py` **[live]** — locked label-crop spec (`render_label_crop`). _(2026-07-06)_
- `emission_selector.py` **[live]** — Stage-2d MAP-Elites diversity selector (unit-tested). _(2026-07-13)_
- `emit_v1.py` **[live]** — emission back-end (v2 gate → selector → full-res render); idle in the prospect loop. _(2026-07-13)_
- `library_annotate.py` **[live]** — ANNOTATOR tail; folds the **recovered** `morph_gray_image` producer; writes to `library_store`. _(2026-07-15)_
- `library_dedup.py` **[live]** — identity-aware coordinate dedup. _(2026-07-15)_
- `library_store.py` **[live]** — crash-safe library persistence (tmp+atomic-rename shards). _(2026-07-15)_
- `build_bootstrap.py`, `build_headbatch_dramatic.py` **[NIBK]** — sole producers of committed wallpaper-head training batches. _(2026-07-08/09)_
- `library_records_build.py` **[NIBK]** — sole producer of **uncommitted** `data/library/library_records.jsonl` (§3). _(2026-07-15)_
- `morph_producer_tag.py` **[NIBK]** — tags `morph_producer` into the **uncommitted** `embeddings.npz` (§3). _(2026-07-15)_
- `prelabel_score_v2.py`, `rerender_bootstrap_ss2.py` **[NIBK]** — pre-label preprocess / completed one-shot migration. _(2026-07-08/09)_
- `emission_dryrun_v2gate.py`, `family_entropy_trace.py`, `selector_montage.py` **[NIBK]** — selector analysis/diagnostics. _(2026-07-06/11)_
- `selector_family_diversity_sweep.py` **[NIBK, NEAR-BROKEN]** — crashes at import; needs missing `scratchpad/_stage4_cells.json`. _(2026-07-11)_
- `test_emission_selector.py`, `test_pool_rebalance.py`, `test_prospect.py`, `test_shard_crash.py` (`slow`) **[test-only]**.

### tools/atlas/
- `production_seeder.py` **[live]**, `guard.py` **[live]**, `prescreen.py` **[live]** — the standing seeder + its degenerate-outcome gate + depth-2 pre-screen. _(→2026-07-15)_
- `gather_overnight.py` **[NIBK]** — legacy overnight gather driver (superseded as *the* loop by the orchestrators; still runnable). _(2026-07-05)_
- `confirm_report.py`, `cross_family_shakeout.py`, `check_ledger_decode_version.py` **[NIBK]** — post-run/observation diagnostics. _(→2026-07-09)_
- `verify_v6_gate.py`, `v5_v6_anchor_diff.py` **[NIBK]** — v5→v6 swap verifiers (investigation closed; strongest dead-*candidates* in the dir, but reproducible verifications — keep). _(2026-07-05)_
- `test_guard.py`, `test_guard_tripwire.py` (`slow`), `test_julia_bands_parity.py`, `test_production_seeder.py` **[test-only]**.

### tools/mining/
- `mining_gate.py` **[live]** (strange-mode gate SOT), `score_lib.py` **[live]** (v3/CORN scoring hub), `tail_alloc.py` **[live]**, `deploy_tail.py` **[live]** (post-emission tail).
- `calibrate_t2.py`, `lock_mining_gate.py`, `palette_families.py` **[NIBK]** — each sole producer of a committed `data/mining/*.json` operating-point/roster artifact.
- `harvest.py`, `dedup.py` **[NIBK]** — the manual harvest island (harvest has "no automated driver"; dedup is its pHash dep).
- `test_tail_alloc.py` **[test-only]**.
- Note: `score_lib.run_enrich_score` **reimplements** the `enrich_score.py` bridge — a drift risk (§6).

### tools/queries/ (+ scorer/)
- **Live inference layer:** `query_sampler.py`, `sample_location.py`, `query_batch_gen.py`, `assemble_queries.py`, `scorer/data.py` (holds `ACTIVE_SCORER_DIR`), `scorer/train.py`.
- **Batch-gen CLIs [NIBK]:** `regenerate_coldstart_v2.py`, `warmstart_v1.py`, `prefv2_dramatic_v1.py`, `launch_query_label_server.py` (Flask label server — sole producer of the human pref-label store, §3 note).
- **Diagnostics [NIBK]:** `diversity_diagnostic.py` → `compare_v1_v2_diversity.py`, `validate_coarse_score.py`, `color_metrics.py` (ΔE lib, has a committed test).
- **scorer trainers/evals [NIBK]:** `train_v3.py`, `train_v3_gvo.py` (**sole producer of the ACTIVE pref head** — highest-value keeper here), `compare_v3_gvo.py`, `surfacing_eval.py`, `gvo_experiment.py` (v1-era A/B, dead-candidate if closed).
- `test_location_pool.py` **[test-only]** (guards the "0 Julia" regression), plus root `tools/test_color_metrics.py`.

### tools/palettes/
- `color.py` **[live]** (Oklab↔sRGB), `palette_features.py` **[live]** (trajectory features + diversity API, imported by the emit query path).
- `build_categories.py`, `build_pool.py`, `build_features.py` **[NIBK]** — sole producers of committed `data/palettes/{palette_categories,pool_colormaps,palette_features}.json`. **Gotcha:** `build_pool` regenerates `palette_features.json` over the full pool; `build_features` (76-entry) is **superseded** for the artifact — running it silently regresses the durable file.
- `densify_authored.py`, `preview_render.py` **[NIBK]** — shared authored-palette bridge + viz-helper hub.
- `cliff_diag.py`, `softcliff.py`, `render_v2_batch.py` **[NIBK, prune-on-review]** — closed cliff/soft-cliff studies (W=0.08 shipped).
- `viz_batches.py`, `viz_render.py`, `viz_render_winners.py`, `viz_transfer.py` **[NIBK]** — incremental eyeball tools; `viz_transfer` carries a load-bearing bit-parity assertion.
- `test_palette_features.py` **[test-only]**.

### tools/curation/ (active colorizer increment, all 2026-07-15)
- `colored_clip.py` **[live]** — CLIP descriptor producer; **sole producer of the uncommitted `embeddings.npz`** (§3). Imported by live `library_annotate`.
- `colored_clip_spread.py`, `colorize_assign.py` **[NIBK]** — maximin/share algorithms + collision-aware placement ("NOT wired into emit yet").
- `recolor_pass.py` **[NIBK]** — productionized corpus-scale recolor; writes uncommitted `data/curation/recolor_pass/…` (§3).
- `conditioned_colorize.py`, `soft_spread_calibrate.py`, `morphology_dedup.py` **[NIBK]** — realizability study / tau-calibration harness / non-destructive dedup pass.

### tools/v4, v5, v6 (per-version campaign builders) — all **NIBK**
- `v4/{assemble,build_plan,build_roster}.py`, `v5/{build_manifest,build_plan}.py`, `v6/{build_manifest,build_plan}.py` — each sole producer of a committed `data/vN/` manifest/plan; v5/v6 assert byte-identity vs the prior (copy-forward lineage, §6). `v6/build_manifest` feeds the ACTIVE v6 gate.
- `v4/{montage,pilot,verify}.py`, `v5/montage_render.py`, `v6/{harvest_report,threshold_sweep,verify_t_good}.py` — diagnostics/one-shots. `v6/monitored_harvest.py` is a runnable bounded-harvest driver. (`verify_t_good`/`threshold_sweep` are assert/decode checks that belong in the test suite.)

### tools/render_mode_pilot/ — all **NIBK**
`integrate_dataset.py` is the one on a real dependency path (sole producer of
`data/render_mode_corpus/dataset_v1`, consumed by `train_mining_head`). The rest
are pilot-vs-scale twins: `build_sample`/`render_batch`/`smooth_pass`/`smooth_report`/`signal_read`
(500 pilot) paralleled by `build_scale_sample`/`render_scale_batch` (1000 scale;
`render_scale_batch` ≈ verbatim copy of `render_batch`). `exp_vs_smooth_rankcorr`
produced the exp_smoothing-dupe verdict.

### tools/atlas_probe/, reframe*, coevo/, descent_ablation/
- `reframe/reframe.py` **[live]** (`reframe_location` on the running reward path); `reframe_probe/probe.py` **[live]** (the `ACTIVE_CKPT`+scorer hub — misnamed "probe").
- `reframe_probe/speed.py` **[NIBK]** (finding promoted into reframe).
- `atlas_probe/{step0,step0_coverage,step0_reanalysis,efficiency_depth,efficiency_res}.py` **[NIBK]** — one study cluster over the shared 600-walk pool; findings in MEMORY.
- `coevo/{coevo_round,analyze_round}.py`, `descent_ablation/{run_campaign,finalize}.py` **[NIBK]** — re-runnable campaign harnesses (findings shipped; write only under `out/`).

### palette_extractor/, palette_lib/, dramatic_palettes/ (settled subsystem)
- `palette_lib/build_sheet.py` + its chain (`coloring`, `classify`, `download`, `field`, `importer`, `sampler`, `__init__`) **[NIBK]** — `-m palette_lib.build_sheet` is the sole regen path for the committed colormap library (~18 live readers).
- `palette_extractor/palette_extract.py` **[NIBK]** — core image→palette extractor + its bench/harvest tree (`bench_*` ×6, `harvest_*` ×5, `eval_palette`, `check_bytematch`, `phase0_mirror_visual`, `build_*_manifest`). Self-contained settled research subsystem; findings in MEMORY. `test_roundtrip.py`/`test_wallpaper_extraction.py` are **misnamed harnesses** (match pytest glob, no `test_*` fns).
- `dramatic_palettes/{grid_runner,validate_palettes}.py` **[NIBK]** — explicitly PARKED API-generation path; its committed `results/*.json` are still live inputs (folded into wallpaper-v3 head, read by viz + selector sweep).

### tools/ root + viz/ + misc
- `colormap.py` **[live]** (Python coloring tail, ~40-importer hub), `colormap_acceptance.py` **[live]** (Rust↔Python ≤1-LSB gate), `_bootstrap.py` **[live]** (sys.path shim used by live scripts).
- `label_sheet.py` **[NIBK]** — orphaned present-manifest→HTML viewer at root; overlaps `viz/enrich_sanity_sheet.py`.
- `test_acceptance.py`, `test_color_metrics.py`, `test_colormap.py` **[test-only]**.
- `viz/enrich_sanity_sheet.py`, `viz/guided_descend_scatter.py` **[NIBK]** — pre-label QA / c-plane scatter (references older run layouts).
- `readout/morning_readout.py` **[NIBK]** — overnight diversity readout (recent, current workstream).
- `looksee/looksee_27_walks.py` **[NIBK]** — cross-family health-check.
- `julia_ladder/build_j0.py` **[NIBK]** — sole producer of committed `data/label_corpus/batches/julia_ladder_j0/`.
- `explorer/app.py` **[NIBK]** — Flask explorer (interactive viewer via `explorer.cmd`).
- `audit/disk_audit.py` **[live]** — purge tool invoked by `overnight_orchestrator`.
- `specs/gen_registry.py` **[NIBK]** — sole producer of committed `specs/REGISTRY.md`; parity enforced by `modes_registry.rs`.
- **HTML** (all committed): labeling harnesses `viz/{corpus,location,wallpaper}_label.html`, `queries/query_label.html`; `explorer/templates/index.html`; ~16 palette/AA/harvest **bench** views under `viz/` (mostly from the retired extractor sweep) — `viz/` conflates active labeling infra with frozen benches.

### Root config/docs
`Cargo.toml`/`Cargo.lock`/`pyproject.toml`/`uv.lock` **[live]**; `CLAUDE.md`
(touched 2026-07-16, most-recent), `README.md`, `uf_coloring_algorithms.md`,
`explorer.cmd`, `.gitattributes` (LFS for `v5/model_best.pt`), 8
`docs/findings/*.md` **[NIBK]**. `beautiful_locations.json` **[NIBK]** — curated
reference JSON **at repo root, untracked+gitignored**, regenerable from a
committed batch; violates the "generated output under `out/`" convention (§6).

---

## 6. Structural problems

1. **`tools/` has no tier separation.** Live pipeline code, per-version
   sole-producers, and closed one-shot studies sit intermixed — most visibly in
   `tools/atlas/` (live seeder next to v6-swap verifiers and observation drivers)
   and `tools/queries/` (live inference layer wrapped in batch-gen CLIs and
   diagnostics). A newcomer tracing "what runs in production" can't tell tiers
   apart by location.

2. **Two orchestrators, one spine, one trap.** `prospect_orchestrator` imports
   `overnight_orchestrator` and reuses its discovery/pool/wall-cap helpers
   verbatim; only the tail differs (`library_annotate` vs `emit_v1`). The running
   process does **not** emit wallpapers — easy to misread.

3. **Copy-forward version lineages read as duplication.**
   `v4→v5→v6` `build_plan`/`build_manifest` (byte-identity gated),
   classifier `train_v2..v6` + `train_wallpaper_v1..v3` (import-the-prior chains),
   scorer `train→train_v3→train_v3_gvo`, `render_mode_pilot` pilot-vs-scale twins,
   `atlas_probe` step0 family, `palette_extractor` bench family. All are
   *intentional provenance* (each sole-produces one committed artifact), but the
   naming implies obsolescence — deleting an "old" `train_v4` would break v5/v6.

4. **Oklab/coloring math exists in four hand-synced copies:**
   `tools/palettes/color.py` (reference), `tools/colormap.py` (matrices copied,
   with a comment saying so), the Rust colorer, and `palette_lib/coloring.py` (a
   separate numpy LUT bake). Only the last is guarded by `check_bytematch.py`.
   Any Oklab change must touch all four; drift is silent everywhere else.

5. **Duplicated seams:** the enrich scoring bridge exists twice
   (`corpus/enrich_score.py` vs `mining/score_lib.run_enrich_score`); the
   shared live pool rule `top_k_pool` lives inside the one-shot builder
   `build_humanq3`; `emission_selector` is imported three different ways
   (plain import, `importlib.spec_from_file_location`) so importer-grep under-reports.

6. **Misnaming / mislocation newcomers trip on:**
   `reframe_probe/probe.py` is the live `ACTIVE_CKPT` hub, not a probe;
   root `tools/{colormap,colormap_acceptance,label_sheet}.py` predate the
   subdir convention (the colormap hub lives at root while its sibling
   `color_metrics.py` is under `queries/`); `corpus/smoke_test.py` and the two
   `palette_extractor/test_*.py` match test globs but aren't tests;
   Rust `probe.rs`/`palette_pick.rs` docstrings advertise retired subcommands;
   `beautiful_locations.json` is a generated artifact at repo root.

7. **`data/` persistence policy isn't applied to new stores.** The `/data/*` +
   `!`-allowlist convention (CLAUDE "Persistent-store convention") was not
   extended when `data/library/`, `data/library_embeddings/`, `data/curation/`
   were added — so load-bearing artifacts are gitignored (§3). The policy is
   sound; its application lags new subsystems.

8. **Two committed scripts depend on vanished scratchpad producers** (§1 item 5)
   — the exact anti-pattern CLAUDE warns about, already realized in prose
   references.

---

## 7. Target structure — proposed changes, risk-ordered

Mechanical/safe first, semantically risky last. Nothing here is applied.

### Tier 0 — config/docs only, ~zero code risk
- **T0-a (do first). Un-ignore the at-risk data.** Add allowlist lines for
  `data/library/`, `data/library_embeddings/` (and decide on `data/curation/`),
  then `git add` the artifacts. *Breaks:* nothing (allowlist-only). *Verify:*
  `git check-ignore` empty for both; `git status` shows them staged; live loop
  unaffected.
- **T0-b. Back up / confirm the human pref-labels** (`data/queries/labels/*.json`)
  exist somewhere — they're absent in this checkout (§3 note).
- **T0-c. Header notes on superseded producers:** `build_features.py` ("pool
  file now owned by `build_pool.py`"), `corpus_data.py` ("v2/v3 only; v4+ uses
  `data_v4.py`"). *Breaks:* nothing.
- **T0-d. Fix stale docstrings** in `src/probe.rs`, `src/palette_pick.rs`
  (retired subcommands). *Verify:* `cargo build --release --lib` (~3s check).
- **T0-e. Relocate `beautiful_locations.json`** under `out/` or commit it
  intentionally under `data/`. *Verify:* grep shows no code reads it (confirmed).

### Tier 1 — mechanical renames, low risk (greppable call sites)
- **T1-a. Rename off the `test_` prefix** the non-tests:
  `palette_extractor/test_roundtrip.py`, `test_wallpaper_extraction.py`
  (→ `harness_*`); consider renaming `corpus/smoke_test.py`. *Breaks:* pytest
  collection (currently collects them, finds nothing). *Verify:*
  `uv run pytest --collect-only`.
- **T1-b. Repair or quarantine `selector_family_diversity_sweep.py`** (crashes at
  import on a missing scratchpad file). Either regenerate the input under `out/`
  or move the script to an archive. *Verify:* `python -c "import ast"` parse +
  run.
- **T1-c. Delete the two vanished-scratchpad references** in comments
  (`production_seeder.py`, `build_categories.py`) or point them at the committed
  `labels/*` outputs. *Breaks:* nothing (comments).

### Tier 2 — structural regrouping, medium risk (touches live import/subprocess paths)
- **T2-a. Introduce a tier split under `tools/`:** e.g. `tools/pipeline/` (live —
  corpus hubs, wallpaper orchestrators, atlas seeder+guard+prescreen, mining
  gate/score_lib/tail/deploy, queries inference layer, reframe, colormap,
  disk_audit) vs `tools/studies/` (closed one-shots — atlas_probe,
  reframe_probe/speed, render_mode_pilot, eda, coevo, descent_ablation,
  palette_extractor benches, v4/v5 montages). Per-version *builders* stay beside
  their committed `data/vN`. *Breaks:* `_bootstrap.py` sys.path assumptions,
  subprocess path constants in both orchestrators, `import`-by-basename. High
  blast radius — do as a scripted move with a grep sweep. *Verify:* full
  `uv run pytest`; a `prospect_orchestrator.py --mini` and
  `overnight_orchestrator.py --mini` dry run; `production_seeder.py --smoke`.
- **T2-b. Promote `reframe_probe/probe.py`** to a neutrally-named scoring module
  (e.g. `tools/scoring/active_ckpt.py`) exporting `ACTIVE_CKPT`/`make_scorer`.
  *Breaks:* ~10 `import probe` sites across the live seeder/guard/reframe chain.
  *Verify:* grep clean + `production_seeder.py --smoke`.
- **T2-c. Lift `top_k_pool`** out of `build_humanq3` into a neutral pool module
  imported by `build_fresh_discovery`. *Verify:* `--mini` prospect run.
- **T2-d. Unify the enrich scoring bridge** (`enrich_score.py` vs
  `score_lib.run_enrich_score`) to one implementation. *Breaks:* labeling path +
  harvest path. *Verify:* an `enrich --mode score` smoke + a `harvest.py` dry run.
- **T2-e. Consolidate root `tools/*.py`** (`colormap`, `colormap_acceptance`,
  `label_sheet`, root tests) into a `tools/color/` package. *Breaks:* `colormap`
  is a ~40-importer hub — largest import blast radius in the repo; script the
  rename. *Verify:* full pytest incl. `test_acceptance.py` (the ≤1-LSB gate).

### Tier 3 — semantically risky, do last (byte-exactness is load-bearing)
- **T3-a. Collapse the Oklab/coloring copies:** make `tools/colormap.py` import
  its matrices from `tools/palettes/color.py`; decide whether
  `palette_lib/coloring.py` should share the Rust-parity bake or stay an
  explicitly-frozen extractor-only port. *Breaks:* LUT byte-exactness is asserted
  by `check_bytematch.py`, `tests/palette_bytematch.rs`, `viz_transfer.py`'s
  parity assert, and `colormap_acceptance.py`. Any change must preserve bytes.
  *Verify:* run all four before/after; they must stay green.
- **T3-b. Archive (don't delete) confirmed-closed studies** only after
  confirming each finding is captured in `docs/findings/` or MEMORY and the
  script is not the sole producer of a committed artifact. Move to
  `tools/studies/archive/` rather than `rm`. The **only** genuine delete
  candidate is `tools/eda/eda_common.py` (§8) — and only after settling its
  ambiguity.

---

## 8. Dead / ambiguous / near-broken (the short, high-scrutiny list)

- **`tools/eda/eda_common.py` — ambiguous, leans dead.** Imported by *nothing*
  (zero hits repo-wide, including within `tools/eda/`). It's a cohort loader
  written for "Panel A–E" scripts + `prompts/provenance-label-eda.md` that are
  **not present** in the repo. *Settles it:* find those panel scripts or the
  prompt; if they never existed outside scratchpad, it's dead. Until then, don't
  delete — this is exactly the pattern that burned the repo before.
- **`tools/wallpaper/selector_family_diversity_sweep.py` — near-broken.** Crashes
  at import (missing `scratchpad/_stage4_cells.json`). Keep but repair/quarantine
  (T1-b).
- **Weak dead-candidates (keep — reproducible verifications of closed
  investigations):** `atlas/verify_v6_gate.py` + `atlas/v5_v6_anchor_diff.py`
  (v5→v6 swap), `queries/scorer/gvo_experiment.py` (v1-era gvo A/B),
  `queries/compare_v1_v2_diversity.py` (coldstart A/B),
  `palettes/{cliff_diag,softcliff,render_v2_batch}.py` (shipped cliff studies).
  None imported; each documents a closed decision. Archive-on-review, not delete.

*No file in the repo is a sole producer of a committed `data/` artifact while
being itself uncommitted (all producer scripts are tracked). The risk is the
inverse and is real: committed producers whose **output** is uncommitted — §3.*
