# Repo size audit + relocation plan

Companion to `repo_structure_audit.md` (which chased the *file-count* bomb behind
the slow `grep`). This one chases **bytes**: after the file-count restructure the
working tree is 60k files but still **~21 GB on disk**. Where that weight lives, what
must stay, and a phased plan to get the rest out — under one rule from the owner:

> **Except `out/` (disposable → trash) and the trained deep-network checkpoints
> (`.pt`, precious → keep), most large files should live outside the source tree —
> in `fractal-generator-artifacts`, or wherever they actually belong.**

## Current footprint (three trees, one volume)

| tree | size | what |
|---|---:|---|
| `C:\Code\fractal-generator` (repo) | **21 GB** | source + the weight below |
| `C:\Code\fractal-generator-artifacts` | **60 GB** | relocated: aug_cache ×4 (the real byte bomb) + campaign2 scratch + viz sheets |
| `C:\Code\fractal-generator-trash` | **14 GB** | staged `out/`; delete to reclaim (`rm -rf`) |

**Already relocated (this session):** the aug_cache families were ~560k *files* but
also **~60 GB of JPGs** — moving them out was the single biggest byte win, it just
landed in the sibling tree. `out/` (14 GB) is staged to trash. Reader-free viz sheets
(`*_grid.png`/`*_sheet.png`, 860 files / **497 MB**) were relocated to artifacts.

## Where the remaining 21 GB is

| dir | size | disposition | blocker to moving |
|---|---:|---|---|
| `.venv/` | 4.8 GB | **KEEP** | live env; `uv sync`-regenerable, not a data artifact |
| `data/label_corpus/` (crops = **3.1 GB**) | 6.1 GB | **RELOCATE crops** | crops are HTTP-served to the label UI **and** read by training via `images.jsonl` repo-relative paths |
| `target/` + `target-test/` | 2.1 GB | KEEP (or redirect) | live `.exe`; `CARGO_TARGET_DIR` could point outside the tree |
| `data/root_field/*.f32` | 1.0 GB | **RELOCATE** | **Rust** `src/root_field.rs:55` `CACHE_DIR` — live cache for `guided_descend` (root8k scoring) |
| `data/wallpaper_head/*.pt` (346M) + `data/classifier/*.pt` (462M) | **918 MB (38 `.pt`)** | **KEEP** | trained checkpoints — the owner's explicit exception |
| `data/render_mode_corpus/` | 851 MB | **RELOCATE** | ~10 Python readers (`render_mode_pilot/*`, `train_mining_head`, `lock_mining_gate`) + `dataset_v1` crop paths |
| `data/library/` | 814 MB | **RELOCATE bulk** | interleaved: tracked `library_records.jsonl` (keep) + ignored shards/tiles read by emission |
| `data/queries/` | 757 MB | **RELOCATE crops** | interleaved (tracked labels + ignored query crops) |
| `data_large/` | 653 MB | **RELOCATE** | interleaved: tracked `README.md` + untracked `label_crops/` bulk |
| `dramatic_palettes/` | 640 MB | RELOCATE viz | tracked palettes (keep) + ignored `viz_render/`/`densified/` (regenerable) |
| `data/cache_manifest.jsonl` ×4 + `plan.jsonl` | ~360 MB | **RELOCATE** | hardcoded `ROOT/data/vN/cache_manifest.jsonl` in `data_v4`/`train_vN`/`verify` |
| `.git/` | 364 MB | **history rewrite** | 58:1 vs the 6 MB tracked tree — old committed binaries; `git filter-repo` |
| `data/discovery/` | 234 MB | **RELOCATE run-state** | 173 MB of regenerable overlays (`prio_terms`/`morph_mem`/`node_embs`/`*.log`); intake reads only `outcome_ledger`/`harvest_log` |
| `data/guided_descend/` | ~150 MB | RELOCATE caches | tiny `pool.jsonl` pools (keep, live) + field/render caches |

**The shape of the problem:** unlike the file-count bomb (one reader, portable
manifest — trivially relocatable), the byte-weight is **woven into live readers via
hardcoded paths**, several in **Rust**, and one (label_corpus crops) is **served over
HTTP to the label UI**. None is a blind `mv`. Each needs the same discipline the
aug_cache move used: sweep every reader+writer, route through a resolver, verify.

## The one architectural gap to close first

The `tools/corpus/artifacts.py` resolver is **Python-only**. Three of the biggest
targets (`root_field.f32`, keeper/pool viz already handled, any future Rust cache)
are written by **Rust** to hardcoded `data/...` constants, which a Python resolver
can't intercept. Before Phase 2, add a **Rust-side twin**: read `FRACTAL_ARTIFACTS_ROOT`
(same env var, same default sibling) in `src/lib.rs`, and route the handful of
Rust cache/output path constants (`root_field::CACHE_DIR`, and audit `generate.rs`/
`present.rs`/`guided_descend.rs` output roots) through it. One small module; makes the
Rust and Python halves agree on where artifacts live.

## Phased plan (each phase: sweep → wire resolver → relocate → verify gate)

**Phase 0 — done.** aug_cache ×4 + campaign2 scratch → artifacts; `out/` → trash;
reader-free viz sheets → artifacts. Resolver + tripwire + >1 MB pre-commit hook in
place. Net: ~74 GB of bytes and ~580k files out of the tree.

**Phase 1 — Python-only, low risk (~1.5 GB).** Extend `RELOCATED_PREFIXES` and wire
the Python readers for: `data/cache_manifest.jsonl`+`plan.jsonl` (~360 MB; change the
4 hardcoded `DEFAULT_CACHE`/`V*_CACHE` constants to `resolve()`), `data/discovery`
regenerable run-state (~173 MB; intake never reads it — verify then relocate),
`data/guided_descend` non-pool caches, `dramatic_palettes/viz_render`. Each has a
bounded, greppable reader set and no Rust/HTTP entanglement.

**Phase 2 — Rust cache (~1 GB).** After the Rust-side resolver (above), relocate
`data/root_field/*.f32` by routing `root_field::CACHE_DIR` through it; rebuild;
confirm `guided-descend` still finds/writes the field under artifacts. Add the prefix
to the tripwire.

**Phase 3 — live corpora (~2.5 GB).** `render_mode_corpus`, `library` bulk,
`queries` crops, `data_large/label_crops`. Wire the ~10 Python readers (and the
`dataset_v1`/`images.jsonl` repo-relative crop paths, aug_cache-style) through the
resolver. These are exercised by mining/emission tools not in the mini gate, so verify
each tool directly after wiring.

**Phase 4 — label_corpus crops (3.1 GB, most involved).** The crops are (a) read by
training via `images.jsonl` and (b) **served to the label UI over HTTP**. Relocation
requires the label viz server to serve from artifacts too (point its document root at
`ARTIFACTS_ROOT`, or run a second static mount), plus routing `present.rs` crop-rebuild
and the classifier corpus loader through the resolver. Biggest single win; do it last,
attended.

**Phase 5 — history (~360 MB, one-time).** `git filter-repo` to drop the old committed
binary weight from `.git`; the >1 MB pre-commit hook (already installed) prevents
recurrence. Rewrites history — coordinate before doing it.

## What stays in-tree by design
Source; committed metadata (tracked `.jsonl`/`.json` — ledgers, manifests, schemas,
labels, pools); the **38 `.pt` checkpoints (918 MB)**; `out/` as an empty `.gitkeep`
anchor; `.venv` and `target*` (live, regenerable infra — relocate only via
`CARGO_TARGET_DIR`/env if desired, not via the artifacts resolver).

## Net if fully executed
Repo drops from ~21 GB to **~7–8 GB** (`.venv` 4.8 + `target*` 2.1 + checkpoints 0.9
+ source/metadata ~0.3), with all regenerable bulk in `fractal-generator-artifacts`
and disposable output in trash. The source tree becomes "code + irreplaceable
metadata + trained weights," nothing else.
