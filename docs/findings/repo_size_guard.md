# Repo-size guard — large-in-tree inventory + relocation worklist

The standing constraint: the working tree stays ~what git tracks — source +
irreplaceable metadata + `out/`. Nothing large lives in-tree without a written-down
reason. This is now **enforced and self-documenting**:

- **Scan** — `tools/audit/size_guard.py` walks the *filesystem* (not `git ls-files`;
  a gitignored file can bloat the tree while invisible to git — the whole point).
  Flags (a) every file ≥ **1 MiB** (matches the pre-commit blob hook) and (b) every
  many-small-file directory whose small-file subtree ≥ **~100 MB** (catches crop /
  cache dirs no single-file rule sees). Excludes `{out/, .venv/, target/,
  target-test/, .git/}` from flagging.
- **Registry** — the `REGISTRY` allowlist in that module. Every current violator is
  covered by exactly one line at a stable path-prefix granularity, with a
  disposition. This table is that registry, grouped.
- **Test** — `tests/test_repo_size_guard.py` fails on any flagged violator with no
  registry line (new bloat caught from today), and warns (does not fail) on a line
  that no longer covers over-threshold content (nudge to delete it). Proven to go
  red on purpose (temp ≥1 MiB file outside the excluded prefixes → fail naming it →
  remove → green).

**This is the guard-backed, live successor to `repo_size_audit.md`'s table** — same
worklist, now mechanically checked. Nothing here has been moved, deleted, or
committed; the `RELOCATE` lines are the worklist for the phases that follow.

## Snapshot

| | |
|---|---:|
| flagged violators | **1147** (1126 files + 21 small-file dirs) |
| total flagged bytes | **13.2 GB** |
| `.git` (FYI — history-rewrite target, not flagged) | 352.3 MB |

Dispositions: **KEEP** = legitimately in-tree (irreplaceable tracked metadata, no
smaller form). **RELOCATE → `<tier>`** = pending a move; the tier is a disposition
label only (no dirs created / paths wired — the precious-store *location* is still
undecided): `artifacts` = regenerable bulk, `precious-store` = irreplaceable trained
binaries, `trash` = dead/superseded. Delete a `RELOCATE` line when its move lands;
when only `KEEP` lines remain, the guard is fully enforcing.

## KEEP — stays in-tree (26.5 MB)

| size | tracked | path | reason |
|---:|:--|:--|:--|
| 24.2 MB | mixed | `data/palettes/` | committed palette definitions (harvested 746-palette pool + features); load-bearing palette-system config, no smaller form |
| 2.2 MB | tracked | `data/library_embeddings/` | prospect-library CLIP embeddings (`embeddings.npz`); unregenerable except value-approximate under a verdict-sensitive threshold. **CANARY** |

## RELOCATE → artifacts — regenerable bulk (12.4 GB)

Rebuildable render / cache output. Where a prefix is `mixed`, the tiny tracked
metadata inside it (labels, ledgers, pools, records) **stays in-tree** — only the
regenerable bulk relocates.

| size | tracked | path | reason |
|---:|:--|:--|:--|
| 5.7 GB | mixed | `data/label_corpus/` | batch crops (regenerable via `present`/`render-one`) + dead `_work/` preview & crop-staging; tracked `scores.json`/`images.jsonl` labels stay |
| 1.0 GB | ignored | `data/root_field/` | root8k f32 score-field cache (4× 256 MB); regenerable via the **Rust** dump (`src/root_field.rs` `CACHE_DIR`) — needs a Rust-side artifacts resolver first |
| 930.2 MB | mixed | `data/wallpaper_corpus/` | wallpaper batch crops (regenerable); tracked `images.jsonl`/ledgers stay |
| 844.8 MB | mixed | `data/render_mode_corpus/` | render-mode batch crops (regenerable via `present`); tracked manifests stay |
| 808.6 MB | mixed | `data/library/` | `field_cache` render bulk (regenerable); tracked `library_records.jsonl` stays |
| 754.2 MB | mixed | `data/queries/` | query-assembler field/colormap renders + scorer caches (regenerable via `tools/queries`); tracked `queries/labels/*.json` preference tiers stay |
| 650.5 MB | ignored | `data_large/label_crops/` | loose0 crop feed; regenerable render output (tracked `data_large/README` stays) |
| 632.9 MB | mixed | `dramatic_palettes/` | `viz_render` + `viz_render_winners` render sheets (regenerable); tracked palette definitions stay |
| 548.0 MB | ignored | `data/label_crops/` | early loose label-crop feed (`loose0_v2`/`v3`); regenerable render output |
| 195.6 MB | mixed | `data/discovery/` | regenerable run-state overlays (`campaign*`/`steered*`/`shakeout*` renders, logs); tracked ledgers/pools/`outcome_feats` provenance stays |
| 97.4 MB | ignored | `data/v7/` | v7 build cache-manifest + plan (regenerable via `build_plan`; active version) |
| 91.0 MB | ignored | `data/v4/` | v4 build cache-manifest + plan + montage (regenerable; superseded) |
| 90.0 MB | ignored | `data/v6/` | v6 build cache-manifest + gather plan (regenerable) |
| 84.5 MB | ignored | `data/v5/` | v5 build cache-manifest + julia plan (regenerable) |
| 19.6 MB | ignored | `data/calibration/maxiter_diag/` | maxiter diagnostic renders (regenerable); the frozen `energy_calibration.json` metric bins are tiny and stay tracked |
| 10.2 MB | mixed | `data/mining/` | mining prospect renders (`run1`); regenerable via `tools/mining` |
| 4.7 MB | mixed | `data/guided_descend/` | render/field caches (`atlas_probe_step0`, `run5`, `julia_test_bulb`); regenerable via `present`/`enrich` (tiny pools stay) |
| 3.3 MB | ignored | `data/ranker/` | frozen-feature location-ranker fits + feature caches (`pref_loc_v0`/`v1`, `campaign1`); regenerable — logistic on committed frozen features |

*Running total: 12.4 GB.*

## RELOCATE → precious-store — irreplaceable trained binaries (599.2 MB)

Trained `.pt` weights; not GPU-reproducible (float nondeterminism), so no rebuild
path. Active + rollback anchors go to the precious store. The `data/classifier/v{5,6,7}`
weights are **CANARY paths** (`tests/test_tracked_artifacts.py`) — their eventual
move needs a deliberate canary update; **change nothing there now**.

| size | tracked | path | reason |
|---:|:--|:--|:--|
| 343.1 MB | ignored | `data/wallpaper_head/` | trained wallpaper-quality heads (v1/v2/v3 `.pt`); active + rollback → precious-store, older versions curate to trash at move |
| 67.5 MB | tracked | `data/classifier/v5/` | v5 `model_best.pt` — deeper rollback anchor. **CANARY** |
| 65.2 MB | tracked | `data/classifier/v7/` | v7 `model_best.pt` — LIVE deployed discovery-gate weight. **CANARY** |
| 65.2 MB | tracked | `data/classifier/v6/` | v6 `model_best.pt` — one-flip rollback anchor. **CANARY** |
| 58.2 MB | ignored | `data/render_mode_head/` | trained render-mode (strange-mode gate) head v1 `.pt` |

*Running total: 13.0 GB.*

## RELOCATE → trash — dead / superseded (277.1 MB)

| size | tracked | path | reason |
|---:|:--|:--|:--|
| 65.2 MB | ignored | `data/classifier/v4/` | superseded classifier v4 weight — won't be retrained |
| 65.2 MB | ignored | `data/classifier/v3/` | superseded classifier v3 weight |
| 65.2 MB | ignored | `data/classifier/v5_seed1/` | v5 seed-1 diagnostic variant — not the live checkpoint |
| 65.2 MB | ignored | `data/classifier/v2/` | superseded classifier v2 weight |
| 15.2 MB | ignored | `data/focus_diag/` | focus-diagnostic scratch (orbit-space field `.npy` dumps); dead, regenerable |
| 1.3 MB | ignored | `scratchpad/` | canonical disposable temp dir — nothing large should persist here |

*Running total: 13.2 GB (= total flagged).*

## `.git` — FYI (352.3 MB, not flagged)

`.git` is a history-**rewrite** target (`git filter-repo` to drop the ~358 MB of
dead committed binary weight the earlier audit found), not a relocation one. The
pre-commit blob hook already prevents recurrence. Reported here for completeness;
the guard never flags it.

## How the buckets net out

- **~12.4 GB → artifacts** is the whole prize: label/wallpaper/render-mode crops,
  field caches, viz sheets, build manifests — all regenerable, all read through
  hardcoded paths (several in Rust, one HTTP-served to the label UI), so each needs
  the resolver-wire-then-verify discipline the earlier audit's phases describe, not
  a blind `mv`.
- **~0.6 GB → precious-store** is the trained-weight tier: needs a precious-store
  *location* decision, and the three classifier canaries need a coordinated
  `test_tracked_artifacts.py` update on the day they move.
- **~0.28 GB → trash** is dead weight (superseded classifier versions, diag scratch)
  — the cheapest win, deletable outright once confirmed.
- **26.5 MB stays.** Two `KEEP` lines. Once every `RELOCATE` line above is retired,
  those are the only large things in-tree, each with a stated reason, and the guard
  is fully enforcing.
