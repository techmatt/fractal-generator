# Repo structure audit

A critical read of how this repo is laid out, triggered by a mundane symptom: a
`grep -rl` over the tree ran **>120 s and had to be backgrounded**, while the
ripgrep-backed Grep tool answered the same query in well under a second. That gap
is not a tooling accident — it is a direct readout of a structural decision. This
doc chases it to root and audits what else the same decision costs.

## The measurement

| metric | value |
|---|---|
| Git-**tracked** files | **1,194** (6.2 MB) |
| Files **in the working tree** | **~640,000** (~**85 GB**) |
| ├─ `data/` | **588,661** files, **72 GB** |
| ├─ `.venv/` | 25,117 files |
| ├─ `out/` | 18,137 files, 10 GB |
| ├─ `target/` + `target-test/` | 5,354 files, 2.1 GB |
| └─ `data_large/` | 1,026 files, 653 MB |
| `.git/` history size | **364 MB** (for a 6.2 MB tracked tree — 58:1) |
| Generated-to-tracked file ratio in-tree | **~530 : 1** |

Where the 588k `data/` files actually live:

| dir | files | what it is |
|---|---|---|
| `data/discovery/campaign2` | **317,500** | one discovery run's per-node render scratch + logs |
| `data/v4/aug_cache` | 152,125 | classifier augmentation JPGs (42/location) |
| `data/v5,v6,v7/aug_cache*` | ~91,000 | same, per classifier version |
| `data/label_corpus/batches` | 18,426 | label crops (rebuildable) |
| everything else | ~9,000 | corpora, ledgers, caches |

**~99.98% of the files a recursive tool traverses here are regenerable ML
scratch.** `grep -r`/`find`/`du`/editor indexers/file-watchers/antivirus/cloud-sync
all walk inodes without consulting `.gitignore`, so they pay the full ~640k cost.
ripgrep is fast *only because it reads `.gitignore` and skips them.* The repo's
speed under a given tool is now a coin-flip on whether that tool happens to be
gitignore-aware. That is fragile by construction.

## Root cause: disposable artifacts live *inside* the source tree, interleaved with committed metadata

The single decision under everything below: **generated output shares a filesystem
subtree with source, and committed metadata shares directories with gigantic
regenerable artifacts.** Concretely, `data/label_corpus/batches/<id>/` holds both
`images.jsonl` (tiny, irreplaceable, committed) and `crops/` (thousands of
rebuildable JPGs, ignored) *in the same directory*.

That interleaving forces the `.gitignore` to become a **~150-line per-experiment
include/exclude negation machine** — re-include a dir, re-exclude its contents,
negate the one durable file, repeat for `library/`, `library_embeddings/`,
`queries/`, `sampler_eval/`, `discovery/`, `atlas/`, `emission/`, each corpus… The
file even documents its own traps ("a bare `!/data/library_embeddings/` would
re-include the per-cycle shards"). Every new experiment must hand-author another
stanza correctly or it either commits scratch or loses metadata. This is a
standing tax on every future change and a live source of "oops, committed 300 MB"
and "oops, lost the only copy" incidents.

Symptoms that all trace back to this one cause:

1. **Traversal cost** (the presenting bug) — 640k inodes in-tree.
2. **`.gitignore` complexity** — 150 lines that should be ~3.
3. **Git history bloat** — `.git` is 364 MB against a 6.2 MB tree; large artifacts
   were committed and later rewritten/removed, but history keeps the weight. Every
   clone pays it.
4. **Storage-tier sprawl** — because "inside the tree" was the default, disposable
   data accreted into **four parallel tiers** with overlapping semantics:
   `out/` (disposable), `data/` (mostly-ignored-but-partly-committed),
   `data_large/` (653 MB, only a README tracked), and `scratchpad/` (temp). Three
   of the four are prose-governed in CLAUDE.md. The `data/` vs `data_large/` naming
   is actively confusing. That the rules had to be written — and that load-bearing
   code once *vanished* from `scratchpad/` (the `visual_dup/embed.py` incident,
   still cited in CLAUDE.md), and that an overnight run once dumped 100 GB+ — is
   evidence the model is confusing enough to cause real losses, not just friction.

## The file-count bomb is a serialization choice, not a data-volume problem

The two worst offenders — `discovery/campaign2` (317k files) and `vN/aug_cache`
(243k files) — are large in **file count**, which is what kills traversal, and the
count comes from **one-file-per-item serialization**: one JPG per augmentation
(42 × N locations), one scratch file per descent node. The same bytes packed into
a handful of shard files (tar / webdataset / a single `.npy` stack per version)
would cut the tree from ~640k files to a few thousand with *zero* loss of
information and make every traversal fast regardless of gitignore-awareness. The
augmentation cache in particular is a classic ML anti-pattern: materializing
deterministic augmentations to loose files instead of generating them in the
`DataLoader` or reading them from one packed archive.

## The Python experiment layer has no live/dead boundary

`tools/` is **249 Python files across ~35 subdirectories**, many named for a
now-superseded model version or a one-shot probe: `v4/ v5/ v6/ v7/ atlas/
atlas_probe/ phoenix/ coevo/ render_mode_pilot/ descent_ablation/ reframe_probe/
queries/ mining/ …`. This is an append-only research ledger masquerading as a
source tree. Nothing distinguishes the *current* pipeline drivers from retired
scaffolding, so every "what produces X?" question is an archaeology dig, and dead
code silently rots (imports that no longer resolve, caches no one regenerates).
The knowledge side mirrors it: **161 `docs/findings/*.md`** plus **100+ memory
entries**, flat and append-only. Supersession is tracked *inline in prose*
("SUPERSEDES", "CORRECTS above") rather than structurally, so the reader must load
the whole pile to know what's still true.

## What is actually well-built (so the critique is calibrated)

The **Rust render core is disciplined and should not be touched by any of the
above**: 24 files / ~17.8k LOC behind two deliberate seams (`FractalBackend`
precision tiering; the pure `PixelSample → RGB` coloring map that makes re-color
free), documented in real `//!` module rationale, validated by f64-vs-perturbation
ground-truth tests, minimal pure-Rust deps by choice. The `out/<subcommand>/`
output convention and the "expose the artifact path as a `pub const` shared by
reader and writer" rule are good instincts. The problem is **not** the engine or
the *stated* conventions — it's that the artifact-storage model those conventions
try to police is fighting the filesystem instead of using it.

## Recommendations (highest leverage first)

1. **Move all regenerable artifacts out of the working tree.** Introduce one
   `ARTIFACTS_ROOT` (env var, default to a *sibling* dir e.g. `../fractal-artifacts/`,
   or a junction). Source tree then contains only source → `grep`/`find`/`du`/
   watchers are fast *by construction*, gitignore-awareness stops mattering, and
   `.gitignore` collapses to a handful of lines. This single change dissolves
   symptoms 1–4.
2. **If artifacts must stay in-tree, enforce a hard split:** *all* disposable
   output under one ignored root (`out/`), *all* committed metadata under one
   small tracked root (`meta/` or `store/`) — never interleaved in the same
   directory. Then `.gitignore` is `/out/` + `/data_large/` + venv/target, not 150
   lines of negations. Retire `data_large/` into the same scheme (kill the
   `data/`↔`data_large/` name clash).
3. **Pack the file-count bombs.** One archive per augmentation set and per
   discovery run instead of hundreds of thousands of loose files; better yet,
   generate augmentations in the loader. Cuts the tree ~100× in file count.
4. **Rewrite git history** (`git filter-repo` / BFG) to shed the ~358 MB of dead
   binary weight, then add a pre-commit guard that rejects blobs over ~1 MB so it
   never recurs.
5. **Give `tools/` a live/dead boundary** — an `archive/` (or delete) for retired
   version-drivers, so the current pipeline is legible at a glance. Add a
   findings index with explicit supersession, so 161 docs are navigable without
   reading all of them.

The through-line: this is a research repo that has been **using git and the
working tree as its experiment database.** Separating "the code + the irreplaceable
metadata" (tiny, versioned) from "the generated experiment data" (huge,
regenerable, *out of the tree*) is the one move that makes almost every listed
problem — starting with the slow grep — disappear at once.
