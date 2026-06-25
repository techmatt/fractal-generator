# code_report.md — refactor / cleanup guide for a follow-up Claude Code instance

Written 2026-06-24 after building the `enrich` (v2-filtered enrichment) pipeline.
Scope: the files that pipeline touched — `src/cli.rs`, `src/main.rs`,
`src/{present,palette_probe,enrich,render,energy,generate,probe}.rs`, the
`tools/corpus/*.py` batch tooling, `tools/viz/corpus_label.html`, and `CLAUDE.md`.

These are observations "at a glance," prioritized. Each item has **Why**, **Where**
(concrete paths/lines), **Do**, and **Risk**. Do the high-priority, low-risk ones
first. **Verify after each change**: `cargo build --release && cargo test`, and for
Python, `uv run python <script> --help` (the corpus scripts have no unit tests).

---

## P0 — `src/cli.rs` is a 3,416-line monolith (biggest friction for future prompts)

**Why.** Every subcommand's clap `Args` struct + its parse helpers (`resolved_*`)
live in one file. Adding a subcommand currently means editing **five** places:
`cli.rs` (the `Command` enum variant + a new `Args` struct + impl), `main.rs`
(`use` + dispatch arm), and `lib.rs` (`pub mod`). The enum has ~30 subcommands and
the file is the single most-edited, most-merge-conflicted file in the repo. Reading
it to find one struct costs a paginated Read every time (it blew the 25k-token Read
cap during this task).

**Where.** `src/cli.rs` lines ~387–3220 are almost entirely per-subcommand `Args`
structs (`FocusDiagArgs`, `AaFilterArgs`, `MusterArgs`, `EnrichArgs`, …) and their
`impl … { resolved_* }` helpers. The shared, genuinely-cross-cutting types are only
at the top: `BackendChoice`, `LocationArgs`, `ShadeArgs`, `PaletteSelectArgs`,
`Cli`, `Command`, `parse_complex`.

**Do (incremental, low-risk).** Move each subcommand's `Args` struct + its `impl`
into that subcommand's own module (e.g. `EnrichArgs` → `src/enrich.rs`,
`MusterArgs` → `src/energy.rs`). clap derive works fine on structs defined anywhere;
`cli.rs` keeps only `Cli`, `Command` (variants reference the moved structs via
`crate::enrich::EnrichArgs`), the shared arg groups, and `parse_complex`. Do it a
few subcommands per commit, `cargo build` between each. End state: cli.rs ~400
lines; adding a subcommand touches the module + `Command` + `main.rs` + `lib.rs`
(four spots, and the bulky struct lives where its logic is).

**Risk.** Low — purely mechanical moves; the compiler catches any missed path.
Keep the `#[derive(Args)]` and `#[arg(...)]` attributes verbatim. Do NOT change any
default values or flag names (batch reproducibility + `batch.json`
`sampling_metaparameters` depend on the documented defaults).

---

## P1 — Duplicated hand-rolled JSONL field readers (6 copies)

**Why.** The project deliberately avoids serde (see CLAUDE.md), so every module
that reads a `.jsonl`/manifest hand-rolls the same three functions. They have
drifted slightly (e.g. `palette_probe::field_f64` matches `"key": ` with a space;
`enrich::field_f64` uses `trim_start()` to tolerate both — the enrich version is
the more robust one).

**Where.** `fn field_f64 / field_str / field_usize` duplicated in:
`src/enrich.rs`, `src/focus_diag.rs`, `src/gate_diag.rs`,
`src/maxiter_diag.rs`, `src/palette_probe.rs`.

**Do.** Add a tiny `src/jsonl.rs` (`pub fn field_f64/field_usize/field_str/field_bool`
+ a one-line module doc) using the **enrich.rs** implementation (the `trim_start`
one, tolerant of `"k":` and `"k": `). Replace the six private copies with
`use crate::jsonl::*;`. Add `pub mod jsonl;` to `lib.rs`.

**Risk.** Low, but these parsers are positional/whitespace-sensitive. After
swapping, re-run any subcommand that reads a manifest (e.g. `palette-probe`,
`gate-diag`) and diff one output against a pre-change run to confirm identical
parsing. The enrich version was validated this task.

---

## P1 — Duplicated "render one wallpaper crop" + `save_jpeg`

**Why.** Three modules independently build the identical f64 iterate-once →
`shade_and_downsample_filtered(..., Lanczos3)` crop, the shallow-spacing assert,
and a quality-90 `save_jpeg`. This is the de-facto wallpaper render path and should
be one helper.

**Where.**
- `fn save_jpeg` duplicated in `src/present.rs`, `src/palette_probe.rs`, `src/enrich.rs` (byte-identical bodies).
- The render body (`F64Backend::new` → `iterate_samples_f64` → `shade_and_downsample_filtered` with `FULL_FILTER = Lanczos3`, `BAILOUT = 1e6`) is repeated in `present::render_and_gate`, `palette_probe::run_palette_probe`, `enrich::{run_score,run_render}`, and `render_one`/`aa_filter`.

**Do.** In `render.rs` (or a new `src/crop.rs`), add:
- `pub fn save_jpeg(img, path, quality) -> Result<(),String>` (move one copy).
- `pub fn render_crop(cx, cy, fw, w, h, ss, maxiter, trap, palette, params, filter) -> RgbImage` that does the iterate+shade and is reused by present/palette_probe/enrich/render_one. Optionally `render_crop_with_buffer` returning `(SampleBuffer, RgbImage)` for the recolor-many callers (present's gate, enrich's K recolors) so they iterate once.

**Risk.** Medium — `render_one`/`aa_filter` are "locked, byte-identical" paths
(see the `render-one wallpaper default` and `aa-filter` memories). If you fold them
in, assert byte-identical PNG/JPG output against a saved reference before/after.
Safer first step: only de-dupe `save_jpeg` (trivially safe) and the
present/palette_probe/enrich recolor path; leave `render_one`/`aa_filter` alone.

---

## P2 — Canonicalize `PERTURB_SPACING` and `color_params`

**Why.** `probe::PERTURB_SPACING` is `pub const = 1e-13` (the canonical f64-floor),
but `main.rs`, `enrich.rs`, `profile.rs` each redefine a private `const
PERTURB_SPACING = 1e-13`, and `aa_filter.rs`, `aa_study.rs`, `maxiter_diag.rs`,
`palette_probe.rs` use the bare `1e-13` literal. One canonical const avoids drift.

Similarly `color_params` exists three times: `generate::color_params()` (no-arg,
the canonical wallpaper coloring — density 0.004, Smooth, Black interior; used by
present/palette_probe/enrich) and two `color_params(shade: &ShadeArgs)` (in
`main.rs` and `probe.rs`) for the CLI render path. The no-arg one is the de-facto
"corpus coloring standard" but its name doesn't say so and it's `pub(crate)` in
`generate`, an odd home.

**Do.** (a) Replace the local `PERTURB_SPACING` consts + `1e-13` literals with
`use crate::probe::PERTURB_SPACING;`. (b) Consider moving `generate::color_params()`
to `coloring.rs` as `pub fn corpus_color_params()` (or document clearly that
`generate::color_params` IS the shared wallpaper standard). Leave the
`color_params(shade)` CLI variants.

**Risk.** Low for (a). For (b), it's a rename across present/palette_probe/enrich —
mechanical; compiler-checked.

---

## P2 — Diagnostic subcommand audit (binary + cli bloat)

**Why.** ~Half the `Command` enum is one-shot diagnostics that have already
produced their finding (recorded in `~/.claude/.../memory/MEMORY.md`):
`rescore`, `overbusy`, `archetype`, `anchor`, `dedup`, `muster`, `cohere`, `cover`,
`buffet`, `reject-corridor`, `gate-diag`, `focus-diag`, `palette-pick`,
`palette-score`, `aa-study`, `aa-filter`, `maxiter-diag`, `palette-probe`,
`profile`, `wallpaper`, `descend`, `navigate`, `search`. Each carries an `Args`
struct (the bulk of cli.rs) and compiles into the binary.

**Do — AUDIT, do not bulk-delete.** Cross-reference each against MEMORY.md. Many
are "diagnosis-only, picks nothing, finding recorded" and could be retired. Propose
a tiered list to the user:
- **Keep (load-bearing):** `render` (default), `sheet`, `generate`, `present`,
  `guided-descend`, `enrich`, `calibrate` (writes the persisted artifact),
  `render-one` (the locked wallpaper default).
- **Likely retire (finding captured in memory, superseded):** `rescore`,
  `overbusy`, `archetype`, `anchor`, `dedup`, `muster`, `cohere`, `cover`,
  `reject-corridor`, `palette-pick`, `palette-score`, `aa-study`, `aa-filter`,
  `maxiter-diag`, `profile`, `gate-diag`, `focus-diag`.
- **Discuss:** `descend`, `navigate`, `search`, `buffet`, `wallpaper` (older
  navigation experiments; CLAUDE.md still references `descend`).

**Risk.** HIGH if done blindly. These are the evidence trail behind the memories,
and CLAUDE.md's architecture section documents `descend`. Removing a subcommand
also drops its `energy.rs`/`coherence.rs` `run_*` entry points (energy.rs hosts
many: `run_rescore/overbusy/archetype/anchor/dedup/muster`). Recommend: get
explicit user sign-off on the retire list, remove in one reviewable commit per
subcommand, and keep `calibrate` + the `Signature/FrozenBins/distance/emd1d/kmeans`
core in energy.rs intact (CLAUDE.md says reuse, don't reimplement).

---

## P2 — `CLAUDE.md` additions (would have saved the most time this task)

**Why.** The corpus/classifier/labeling pipeline is now the active workstream but
CLAUDE.md only documents the render core + descent. A future prompt on this area
has to rediscover the whole flow (this task spent most of its reading budget on it).

**Do — add a "Corpus & classifier pipeline" section** documenting:
1. **The flow:** `guided-descend` → `data/guided_descend/<run>/pool.jsonl` →
   (`present` for zoom/composition batches **or** `enrich` for v2-filtered center
   batches) → `data/label_corpus/batches/<batch_id>/` (schema:
   `data/label_corpus/CORPUS_SCHEMA.md`) → label in `tools/viz/corpus_label.html`
   → `classifier/` trains unioning all batches blind to provenance.
2. **The label corpus contract** (point to CORPUS_SCHEMA.md): `render` block is
   version-invariant (the only thing the classifier sees), `provenance` is
   version-tagged + free to differ, `label.score` is the only `null→value`
   mutation. The shared row shape lives in `tools/corpus/corpus_common.py`
   (`RENDER_KEYS`, `PROVENANCE_KEYS`).
3. **The classifier:** lives in `classifier/` (pkg), weights in
   `data/classifier/{v1,v2}/`. v2 is a CORN **ordinal** head on
   `mobilenetv4_conv_medium.e250_r384_in12k`. Deploy transform = `classifier.data.Transform(train=False)`:
   the deterministic **1280×720 → 384×224 bicubic stretch + normalize** mirror of
   `present.rs`'s JPG path. **P(not-bad) = σ(logit₀)** (= P(rank≥1)=P(label≥2));
   the summed `score_from_logits` is the monotone rank score used for AP. Black
   gate parity: accept iff `black_fraction < 0.30`.
4. **The in-memory scoring bridge** (new this task): `enrich --mode score` streams
   raw RGB frames to stdout (16-byte LE header `idx,ki,w,h` + `w*h*3` bytes);
   `tools/corpus/enrich_score.py` reads the stream and scores with v2 through the
   exact deploy transform — so 10k+ scoring passes never write crops to disk. Only
   the ~1.1k selected crops are rendered (`enrich --mode render`).

**Do — add an "Adding a subcommand" checklist** (until the P0 cli.rs split lands):
`Args` struct in `cli.rs` (or the module) → `Command` variant → `main.rs` `use` +
dispatch arm → `lib.rs` `pub mod`. New subcommands MUST default outputs under
`out/<subcommand>/` (disposable) or `data/<subcommand>/` (load-bearing artifacts);
never the repo root.

**Do — add a Windows note:** the release binary is **file-locked while a long run
is executing**, so `cargo build --release` fails with "Access is denied (os error
5)". To build/iterate while a background run holds `target/release/*.exe`, build
into an isolated dir: `CARGO_TARGET_DIR=target-test cargo build --release` and run
`target-test/release/fractal-generator.exe`. Add `target-test/` to `.gitignore`
(see cleanup below). `cargo build --release --lib` also works for a compile-check
without touching the exe.

**Risk.** None (docs only).

---

## P3 — Repo-root cleanup + `.gitignore`

**Why.** CLAUDE.md's generated-output convention says the root holds only source,
config, docs, committed `assets/`. `git status` shows strays at root:
- `beautiful_locations.json` — **pre-existing, not from this task**; violates the
  convention (a generated artifact at root). Investigate origin; move under `out/`
  or `data/` or delete.
- `target-test/` and `target-test-build.log` — created this task for the
  exe-lock workaround; **transient, delete** (`rm -rf target-test target-test-build.log`)
  and add `target-test/` + `target-test-build.log` (or `target-*/`) to `.gitignore`
  so the documented workaround doesn't dirty the tree. `.gitignore` currently
  ignores `target` and `/target` but not `target-test`.

**Do.** Delete the transient build dir/log; add the gitignore rule; decide on
`beautiful_locations.json` with the user. The new `tools/corpus/build_enrich_batch.py`,
`tools/corpus/enrich_select.py`, `tools/corpus/enrich_score.py`,
`tools/viz/enrich_sanity_sheet.py`, and `src/enrich.rs` are intended to be committed.

**Risk.** Low. Confirm `beautiful_locations.json` isn't an input something reads
before deleting (grep the tree for the filename).

---

## P3 — Minor dead code / smells (low value, fix opportunistically)

- **`palette_probe.rs` `build_index_json`**: carries `let _ = n;` (a param read only
  via a lookup) — vestigial; clean when touching the file.
- **ATOM trap channel is dead in iteration** (already a finding:
  `inner-loop-deadweight` memory — ATOM is 2.4–3.3% of iterate, byte-identical PNG
  without it). Not "delete now" (the const-generic selector is intentional), but if
  a perf pass happens, strip the ATOM path from `iterate_samples_f64`'s dispatch.
- **`enrich::EnrichArgs` mixes score-mode and render-mode fields** (disjoint per
  mode) and bakes `run5` into defaults (`--pool data/guided_descend/run5/...`,
  `--meta-out data/enrich/run5/...`). Fine for one run; a future `run6` must pass
  overrides. If `enrich` becomes routine, consider clap subcommands
  (`enrich score` / `enrich render`) so each gets only its own args, and drop the
  run-number from defaults.
- **`PROVENANCE_KEYS` allowlist** (`tools/corpus/corpus_common.py`): `provenance_block`
  raises on any key not in the tuple, so every new batch type must edit the tuple
  (this task added 6 keys for the v2 bias). Since the schema explicitly allows
  provenance to differ across batches, consider relaxing to "warn on unknown key"
  or accepting arbitrary keys, to lower the friction of new batch types. Keep the
  `RENDER_KEYS` check strict (that block IS the version-invariant contract).
- **Two batch builders** (`build_rev4_batch.py`, `build_enrich_batch.py`) duplicate
  the big `sampling_metaparameters` rev4 dict. Factor a
  `rev4_sampling_metaparameters()` helper in `corpus_common.py` if a third rev4
  batch type appears.

---

## Suggested execution order

1. P3 cleanup (delete `target-test/`, gitignore, resolve `beautiful_locations.json`) — instant, unblocks a clean tree.
2. P2 CLAUDE.md additions — pure docs, highest leverage for the next prompt.
3. P1 `src/jsonl.rs` shared parser — small, compiler-checked, removes 6 copies.
4. P1 `save_jpeg` dedupe + present/palette_probe/enrich render helper (leave the locked render_one/aa_filter paths).
5. P2 canonicalize `PERTURB_SPACING` / `color_params`.
6. P0 `cli.rs` decomposition — biggest win, do last (most churn), a few subcommands per commit.
7. P2 diagnostic-subcommand audit — **only with user sign-off**.
