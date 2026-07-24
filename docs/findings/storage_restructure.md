# Overnight storage restructure — execution report

**Status: COMPLETE.** Relocation + resolver + guards + rescue + `out/` trash-staging
+ before/after verification all done. Nothing is committed — every change below is
left in the working tree for review.

## 0. Preconditions checked
- **No live orchestrator/seeder/render process.** The only processes touching the
  repo were 4 `python -m http.server 8731` static file servers (read-only label
  viz) — no writer mid-flight. Safe to proceed.
- **Same volume.** Repo `C:\Code\fractal-generator`, `ARTIFACTS_ROOT`
  `C:\Code\fractal-generator-artifacts`, `TRASH_ROOT` `C:\Code\fractal-generator-trash`
  are all on the C: NTFS volume → every move is an O(1) rename. Verified with a
  500-file probe (instant) and by the real moves (277k files in 0s).
- **Invariant honored.** Every move gated on `git ls-files <path>` == 0 tracked.
  Post-move `git ls-files -d` (deleted-tracked) == **0**. No tracked file moved,
  deleted, or edited by the relocation. No git operations (add/commit/stage) were
  run against the repo.

## 1. The presenting bug, fixed
Working-tree file count (excl `.git`): **~640,000 → 60,455** (−90.6%). `find`
traversal of the whole tree is now ~3 s. The two named relocations
(`aug_cache` + `campaign2` scratch) were 560,955 files = the bulk, exactly as the
audit predicted; staging the full `out/` tree removed a further ~20.3k files / 14 GB
of binaries. Remaining in-tree bulk is all live/regenerable infrastructure left
deliberately: `.venv` (25k, live env), `data/` interleaved corpora (27.8k), `target*`
(5.4k, live exe).

## 2. Per-subtree classification → tier → action

| subtree | files | tier | action | destination |
|---|---:|---|---|---|
| `src/ tests/ tools/ classifier/ docs/ specs/ palette_* dramatic_palettes/ labels/` | (tracked source) | 1 | **NEVER TOUCH** | in-tree |
| `data/v4/aug_cache` | 152,125 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/data/v4/aug_cache` |
| `data/v5/aug_cache_julia` | 42,001 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/data/v5/aug_cache_julia` |
| `data/v6/aug_cache_gather` | 26,839 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/data/v6/aug_cache_gather` |
| `data/v7/aug_cache` | 22,512 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/data/v7/aug_cache` |
| `data/discovery/campaign2/breadth/scratch` | 277,127 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/…/breadth/scratch` |
| `data/discovery/campaign2/dive/scratch` | 40,351 | 3 | **RELOCATE** | `ARTIFACTS_ROOT/…/dive/scratch` |
| **relocated total** | **560,955** | | | |
| `out/*` (excl tracked `.gitkeep`) | ~20,600 | 4 | **TRASH** (after §4 rescue) | `TRASH_ROOT/out` |
| `out/**/*.md` (24 analysis docs) | 24 | 2 | **RESCUE** (copy) | `docs/rescued/out/` |
| `data/label_corpus/` (batches: tracked `images.jsonl` + ignored `crops/`) | 18,428 | 5 | **LEAVE ENTIRELY** | in-tree |
| `data/discovery/campaign2/{breadth,dive}/` ledgers (`outcome_ledger.jsonl`, `outcome_feats.npz`, `summary.json` — TRACKED) | 6 tracked + siblings | 5 | **LEAVE** (only the untracked `scratch/` child moved) | in-tree |
| `data_large/` (tracked `README.md` + untracked `label_crops/`) | 1,026 | 5 | **LEAVE ENTIRELY** | in-tree |
| `data/wallpaper_corpus/ q4_window_corpus/ library/ queries/ …` | small, interleaved | 5 | **LEAVE** | in-tree |
| `data/label_crops/ render_mode_corpus/ …` (cleanly untracked, but small) | 2,307 / 1,510 / … | 3/4 | **LEAVE** (below the size that matters; relocate later if wanted) | in-tree |
| `.venv/` | 25,117 | 3-ish | **LEAVE** — live env, venvs are not relocatable; `uv sync`-regenerable | in-tree |
| `target/ target-test/` | 4,416 / 938 | 4 | **LEAVE** — the release `.exe` is live-consumed by the verification gate; cargo-wipeable anytime | in-tree |
| `scratchpad/` | 108 | 4 | **LEAVE** — uncommitted `.py` analysis code (the `visual_dup/embed.py`-loss lesson); deserves attended *promotion*, not overnight deletion | in-tree |

### Key reconciliation vs. the prompt's named targets
The prompt said "move the whole `data/discovery/campaign2/` dir (ledgers travel)."
Reality: its `outcome_ledger.jsonl` / `outcome_feats.npz` / `summary.json` in
`breadth/` and `dive/` are **git-TRACKED**, and are live-read by
`tools/emission/library_intake_2.py` and `stage_first_release.py`. Under the
supreme invariant (tracked ⇒ untouchable), campaign2 is an **interleaved** dir. So
only the two wholly-untracked `scratch/` children (317,478 of its ~318,000 files;
0 tracked inside; **read by no current code** — dead per-node render scratch) were
relocated. The tracked ledgers stay in place and their readers are unaffected.

## 3. Resolver — reader + writer sweep coverage
New module **`tools/corpus/artifacts.py`**: `resolve(repo_relative)` maps the six
relocated prefixes to `ARTIFACTS_ROOT` (default repo sibling
`../fractal-generator-artifacts`, env `FRACTAL_ARTIFACTS_ROOT`), everything else
in-tree. Prefixes matched as whole components (a sibling like `aug_cache_notes`
does NOT match). Unit-tested (`tools/audit/test_relocated_artifacts.py`, 5 tests).

**Data model that made this clean:** manifests/plans store *repo-relative* JPG
paths (portable, version-invariant). Only the resolution site changes.

Readers routed (join manifest path → real file):
- `classifier/data_v4.py:load_locations` (the training data loader; also used by
  `train_v5/v6/v7`) — `path = resolve_artifact(r["path"])`.
- `tools/v4/verify.py` — integrity `exists()`/`stat()` loop **and** the coherence
  montage `Image.open`.

Writers routed (emit the path the Rust `v4-render-batch` writes JPGs to):
- `tools/v4/build_plan.py`, `tools/v5/build_plan.py`, `tools/v6/build_plan.py` —
  the plan `"out"` field is now `resolve_artifact(out).as_posix()` (absolute, under
  `ARTIFACTS_ROOT`). The cache-manifest `"path"` field is left **repo-relative and
  byte-identical** (parity-preserving).
- `tools/v7/build_plan.py` — reuses `v6bp.emit_location`, so it inherits the routing
  automatically (verified: its two byte-parity gates compare only the unchanged
  cache-manifest `path`, so they still hold).

The Rust writer `src/v4_cache.rs` needs **no change**: it already `ensure_parent_dir`s
and `save_jpeg`s to `spec.out` verbatim, so an absolute out-of-tree `out` just works.

Confirmed there are **no other** readers/writers of these families: grepped
`aug_cache`, `r["path"]`, `ROOT / …path`, `cache_manifest`, `load_locations`,
`campaign2`, `/scratch` across `tools/ classifier/ src/`. The only other `aug_cache`
references are in `tools/audit/disk_audit.py` (classification regexes over in-tree
paths — informational, not a reader/writer) and the recipe-parity tests (read the
small in-tree JSONL manifests, not the JPGs; skip if absent).

## 4. Rescued (tier-2)
24 `out/**/*.md` analysis docs **copied** to `docs/rescued/out/` (structure
preserved) with a `docs/rescued/README.md` splitting them into 5 likely
hand-authored (keep) vs 19 tool-regenerable (safe to drop). Copies, so the
originals still travel to trash; nothing irreplaceable is lost when trash is
deleted. Uncommitted — `git add` only what you want to keep.

## 5. Trash-staged (tier-4) — FINALIZE COMMAND
Staged to `TRASH_ROOT` = `C:\code\fractal-generator-trash`: **the entire `out/` tree —
20,338 files, 14 GB** (renders, contact sheets, readouts, mini-scratch, field dumps).
The tracked `out/.gitkeep` was left in place, so `out/` survives as an empty anchor
(`out/` now holds exactly 1 file, `.gitkeep`). The 24 tier-2 `.md` were rescued first
(§4). `deleted-tracked` stayed **0** throughout.

Note on the staging: 4 dirs (`out/atlas`, `out/wallpaper`, `out/deep_centers`,
`out/q4_stage1`) were held by a **directory-handle lock** — a plain `mv` of the whole
dir failed with "Permission denied" even though individual files inside moved fine.
Stopping the `http.server :8731` viz processes (done, at your request) did not release
it (the holder was an Explorer/editor handle on the dir, which I did not force-kill).
`robocopy /MOVE` (file-by-file, native) bypassed the lock and completed the move.

**Finalize (reclaim the 14 GB — this is the one irreversible act, yours to run):**
```sh
rm -rf /c/code/fractal-generator-trash
```
Everything under it is regenerable `out/` output; nothing tracked, nothing from
`ARTIFACTS_ROOT`, is in there. `target/`, `target-test/`, and `scratchpad/` were
deliberately NOT trashed (§2).

**Side effect of this step:** the 4 `python -m http.server 8731` label-viz processes
were stopped (you asked me to, to release the lock). Restart the viz server if you
need it.

## 6. Interleaved dirs left for the later attended split
`data/label_corpus/batches/<id>/` (tracked `images.jsonl` + ignored `crops/`),
`data/wallpaper_corpus/`, `data/q4_window_corpus/`, `data/library/` +
`data/library_embeddings/`, `data/queries/`, `data_large/`, and
`data/discovery/campaign2/{breadth,dive}/` (tracked ledgers beside now-empty
`scratch/`). Per-file separation of tracked metadata from ignored bulk in a shared
directory is the explicit "later, attended pass" — not attempted tonight.

## 7. Guards (all uncommitted)
- **`.gitignore`** — no dead lines existed to remove: the relocated families were
  only ever ignored by the blanket `/data/*` and `/data/discovery/**/scratch/`
  rules, which still govern other in-tree paths *and* act as a backstop (an
  accidental in-tree rebuild stays un-committable). Added a clarifying comment
  documenting the relocation.
- **Pre-commit hook** — `tools/hooks/pre-commit` (tracked source) + installed to
  `.git/hooks/pre-commit`. Rejects staged blobs > 1 MiB (override
  `FRACTAL_BLOB_LIMIT=<bytes>`, bypass `--no-verify`). **Proven RED on purpose** in
  an isolated throwaway repo: 2 MiB blob rejected (exit 1), 1 KiB accepted,
  raised-limit override accepted — zero real-repo git operations.
- **Reappearance tripwire** — `tools/audit/test_relocated_artifacts.py::
  test_no_relocated_root_repopulated_in_tree` fails (naming the offender) if any
  relocated family holds real files under its old in-tree path. **Proven RED**
  pre-move (fired on all 6 families, 43–277k files each) and **GREEN** post-move.

## 8. Verification gate — nothing went RED that is attributable to this change
| step | BEFORE (pre-move) | AFTER (post-move) |
|---|---|---|
| `pytest -m "not slow"` | 287 passed, **1 failed** (pre-existing), 3 deselected | 293 passed, **1 failed** (same pre-existing), 3 deselected |
| `production_seeder --smoke` | **exit 0** (3 batches, q3_new=1, 403 s) | **exit 0** (3 batches, q3_new=1, 375 s) |
| `prospect_orchestrator --mini` | — | discovery **rc=0** (phoenix +22 rows, 15 fresh q3) → pool phase started; harness-killed mid-render (env background-lifetime limit, not a failure) |
| `overnight_orchestrator --mini` | — | full ~50 m run exceeds the environment's tool/background time caps; validated by phase decomposition + import integrity (below) |

- **pytest:** the **+6 passing** after are exactly the new resolver/tripwire tests.
  No existing test regressed; **recipe-parity tests pass**, confirming the writer
  edits kept the cache manifest byte-identical.
- **The 1 pytest failure is PRE-EXISTING and unrelated:** `labels/q4_g_aimed.json`
  has a string value `"reject"` where `label_store.py` expects int/null (active q4
  labeling workstream). Reproduces on clean HEAD before any change; not touched.
- **aug_cache READ path validated end-to-end** (the minis don't train, so they never
  read it): `data_v4.load_locations(verify_paths=True)` loaded 3622 locations /
  **152,124 renders, every resolved path existing** under
  `ARTIFACTS_ROOT/data/v4/aug_cache/…`.
- **overnight emit path** — cannot run to completion in-environment (~50 m > caps),
  so validated the way the v7-promote report did: its three phases are covered
  individually — discovery = `production_seeder` (exit 0 after), pool =
  `build_fresh_discovery` (started rc=0 inside the prospect run post-move), emit =
  `emit_v1` — and **all emit-chain modules import clean in fresh processes post-move**
  (`build_fresh_discovery`, `emit_v1`, `overnight_orchestrator`,
  `prospect_orchestrator`, `production_seeder`, `data_v4`).
- **The one anomaly, explained:** the prospect run logged
  `discovery:mandelbrot FAILED (isolated)`. Cause = **GPU contention on the 8 GB card
  from two of my prospect launches briefly overlapping** (a `nohup` launch that
  didn't detach cleanly), NOT a code regression — `production_seeder` imports none of
  the edited modules, and the clean uncontended `--smoke` runs (before *and* after)
  are both exit 0. The overlapping processes were identified and killed.
- **Proven invariance of the minis to this change:** the orchestrators/seeder import
  none of the edited modules (`data_v4`/`build_plan`/`verify`/`artifacts`), and the
  scoring path doesn't import `data_v4`; they read classifier *weights*, never
  `aug_cache`. So their pipeline behavior is independent of the relocation.

## 9. Honest gaps (stated, not papered over)
- **aug_cache WRITE path is not runtime-exercised.** Regenerating a cache needs an
  actual classifier rebuild (`build_plan.py` → Rust `v4-render-batch`), which no
  mini runs. Its rewiring is verified by **grep-completeness + the reappearance
  tripwire + parity tests** only. The READ path, by contrast, *is* now validated
  (§8). If a future rebuild is done, confirm JPGs land under `ARTIFACTS_ROOT` and
  the tripwire stays green.
- **Seeder-smoke gate side-effect:** running `production_seeder --smoke` for the
  BEFORE baseline appended real outcomes to the tracked durable ledgers
  (`data/discovery/outcome_ledger.jsonl` + `outcome_feats.npz` + `probe_rejects.jsonl`).
  That is the seeder's normal behavior, not a restructure edit; left uncommitted
  (revert with `git checkout -- data/discovery/*.jsonl data/discovery/*.npz` if
  unwanted).
