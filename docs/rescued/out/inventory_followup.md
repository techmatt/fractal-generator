# Inventory follow-up — parity surface, NIBK re-audit, embeddings regenerability

**Read-only.** Companion to `out/code_inventory.md`. Nothing moved, edited, or
deleted. A long GPU run was live; only cheap read-only commands and read-only
exploration agents were used. Every commit-status claim below was checked with
`git ls-files` on the actual artifact path.

The three parts, up front:

1. **Parity surface** — the map of every byte-identity / reproduces-stored-values
   guarantee, and which ones are unenforced or off-by-default. One systemic hazard
   dominates: **on a checkout without a built release binary, the entire Rust↔Python
   numeric parity layer silently skips green** — a passing `pytest` is *not* evidence
   of parity.
2. **NIBK re-audit** — **70 of ~151 adjudicated `.py`/`.rs` files move out of the
   bucket** (3→live, 59→ambiguous, 8→dead). The bucket's core defect: ~24 files were
   justified as "sole producer of a *committed* `data/` artifact" and that claim is
   **false** — those `data/` trees are gitignored. They survive only on the weaker
   "regen path for an *uncommitted* live artifact," which doubles as a loss-risk list.
3. **`embeddings.npz` is NOT fully regenerable** — one array (`morph_v6`) has no
   producer at all and survives only inside that file. Commit the frozen base; keep
   the per-cycle shard stream ignored.

Part 4 ties 2+3 together: the expanded §3 loss-risk surface.

---

## Part 1 — Parity surface (the byte-identity acceptance bar)

### 1.0 The systemic gating hazard (read this first)

- `cargo test` (default) runs every parity gate **except** the two
  `#[ignore]` tests (`tests/palette_bytematch.rs`, `tests/occupancy_parity.rs`).
  The `src/backend.rs` `#[cfg(test)]` byte-identity gates **do** run by default.
- `pytest`: `pyproject.toml:36-41` registers a `slow` marker but sets **no
  `addopts`**, so a bare `pytest` does *not* auto-exclude slow. **But** every
  Rust↔Python numeric test is additionally guarded by
  `@pytest.mark.skipif(not acc.BIN.exists())` / an in-body `pytest.skip`. **With no
  `target/release/fractal-generator.exe` on disk, the whole numeric-parity layer
  SKIPS and the suite passes green with those assertions never run.** This is the
  single biggest way a restructure hides drift.
- Several `test_*.py` are collected by pytest but contain **zero `def test_*`**
  (assertions live in a `main()` pytest never calls) — effectively non-tests
  (§1.4).

### 1.1 HIGHEST VALUE — guaranteed but UNENFORCED, or enforced-but-OFF-by-default

Ranked. The top two are the ones to fix before any restructure touches coloring or
the classifier I/O boundary.

1. **Deploy-transform parity — enforced by NOTHING.** The classifier's entire
   premise is that `classifier.data.Transform(train=False)` (1280×720 → 384×224
   bicubic stretch + black-gate + normalize) reproduces what Rust `present.rs`
   emits on its JPG path. **No test renders through `present.rs` and compares
   pixel-for-pixel** — the guarantee lives only in prose (`classifier/data.py:6-11,
   31-32,74`; `corpus_data.py:34,100`; `diagnose.py:14`; `inference.py:5`).
   `colormap_acceptance.py` checks a *different* seam (Rust render ↔ Python color at
   640×360), not the deploy stretch. A refactor to the Rust JPG geometry/encoder or
   the PIL resize/normalize drifts training-vs-deploy inputs **silently**. (MEMORY's
   "deploy scorer parity max|Δ|=2e-7" is the *mining head*, item 1.7 — not this.)
2. **Rust LUT bake ↔ Python `palette_lib.coloring.bake_lut`** — `tests/palette_bytematch.rs`
   is `#[ignore = "driven by check_bytematch.py via env vars"]`. Enforced **only** by
   the standalone `palette_extractor/check_bytematch.py` (`max|Δ| < 1e-12`), which is
   neither in `cargo test` default nor pytest-collected. Someone must remember to run
   it. This is exactly the T3-a collapse the restructure risks.
3. **`colormap_acceptance.py` ≤1-LSB gate** (Rust smooth render ↔ Python
   `colormap.render_candidate`, `TOL_MAX=2`, `TOL_FRAC_GT1=1e-4`). Wired into pytest
   (`tools/test_acceptance.py:24`, `tools/test_colormap.py:188`) but **both
   `skipif(not acc.BIN.exists())`** — green-skips on a fresh checkout.
4. **v5/v6 frozen-cache byte-identity** (`tools/v5/build_plan.py:118-139`,
   `tools/v6/build_plan.py:131-151`): regenerate the frozen Mandelbrot/Julia
   cache-manifest rows and assert byte-equality vs committed `data/v{4,5}/cache_manifest.jsonl`.
   **Runs only when someone manually runs `build_plan.py`** — never in any test/CI.
   Every location-classifier build depends on the frozen `aug_cache` JPGs matching
   the recipe.
5. **`tools/occupancy_parity.rs`** (Rust `energy::occupancy` ↔ Python
   `complexity_scores.json`, `maxd < 0.01`) — `#[ignore]`, needs
   `data_large/label_crops/loose0/…` on disk. Off by default.
6. **`viz_transfer.py` bit-parity assert** (`transfer='pct'` == `transfer='grad'@gamma=0`,
   diff must be `0`) — standalone, **not pytest-collected**, needs cached fields on
   disk. NOTE: this assertion is **redundant** with items 2/3 — it's not the sole
   guard of anything (relevant to the NIBK verdict on `viz_transfer.py`, Part 2).
7. **Mining-gate deploy parity** (`tools/mining/lock_mining_gate.py:127-174`,
   `p_ge3 max|Δ| < 1e-4` — MEMORY's "max|Δ|=2e-7, 625/625"). Runs only when someone
   runs `lock_mining_gate.py` (skippable with `--no-parity`).
8. **Julia band-table hand-sync** (`production_seeder.JULIA_GATHER_BANDS` ↔ Rust
   `julia_band_defaults`) — the table is kept "bit-identical" **by a code comment**
   (`production_seeder.py:172`); machine-checked only by
   `tools/atlas/test_julia_bands_parity.py`, which `pytest.skip`s if the binary is
   absent.
9. **Guard tripwire** (`test_guard_tripwire.py`, 81 pinned f64 verdicts) — `slow`
   **and** `skipif` binary — doubly off by default.
10. **Julia kernel "byte-identity proven" (build-fold)** — the proof was one-time,
    **not committed as a test**. Going forward only the *band table* (item 8) and the
    `backend.rs` degree-2 dispatch arm (item 1.11-B) guard it.

### 1.2 ENFORCED by default `cargo test` (the solid Rust core)

- **A. Perturbation ↔ f64 ground truth** (`tests/perturbation.rs:44`) — *tolerance*,
  not byte (`|Δsmooth|` median <1e-8 / max <1e-2, 0 class disagreements) at
  `maxiter=300` only. Silent-break: a `dc`-geometry/rebasing change drifts *deep*
  renders while this shallow test still passes (it caps where f64 is valid).
- **B. Degree-2 dispatch byte-identity** (`backend.rs:1019`, Julia arm `1078`) —
  `new_degree(..,2)` == `new(..)` bit-for-bit.
- **C. Trap-phase strategy byte-identity** (`backend.rs:1155`) — `PHASE_EVERY/GATED/DEFER`
  produce bit-identical `PixelSample` (`.to_bits()` on all channels).
- **D. Channel-intent dispatch byte-identity** (`tests/channel_dispatch.rs:134`) —
  dispatched-shade **PNG bytes** == all-channels-on, 8-mode × 2-workload matrix.
  Silent-break: a new color mode reading a channel not added to `required_channels`
  *and* not added to `mode_matrix()` escapes this guard.
- **E. Separability** (`tests/separability.rs`, `tests/sheet.rs`) — `sample()` count
  == subpixel grid, invariant across N re-shadings.
- **F. Modes registry ↔ specs lockstep** (`tests/modes_registry.rs`) — structural,
  16==16.

### 1.3 ENFORCED by default `pytest` (pure-Python, no binary — these DON'T skip)

- **G. Field-cache key byte-identity** (`tools/corpus/test_location.py`) — the
  cache-hit-correctness linchpin. Frozen-literal oracles
  (`mandelbrot_90d714081a89180f`, `…_2560x1440ss4`) pin `_field_key` /
  `field_mode_token` / `CANDIDATE_SS` / `EVAL_{WIDTH,HEIGHT}`. Any change to the key
  inputs orphans every cached field (or returns a foreign field for a new mode) —
  **this one is defended** by default. (The on-disk-corpus variant
  `test_existing_keys_byte_identical` silently skips content if the corpus is absent,
  but the frozen literals don't.)
- **H. Python LUT memo purity** (`tools/test_colormap.py`) — `build_lut` (memoized)
  == `_bake_lut` (uncached), memo-warm == cold-rebake, `np.array_equal`. Guards the
  memo key against dropping `reverse`/`mirror`/palette identity. Defended by default.
- **I. Recipe/location bridge byte-identity** (`test_location.py:224`,
  `test_pool_rebalance.py:83`) — default.
- **J. morph_gray robust-z transfer formula pin** (`tools/wallpaper/test_prospect.py:277`)
  — GPU-free `np.array_equal` pin of `MORPH_K=2.0` / `MORPH_MAD_SCALE≈1.4826` / tanh
  form. This is the **committed** enforcement of the recovered morph_clip producer;
  the *stored-row cosine reproduction* is script-only (`morph_producer_tag.py`,
  item 1.1-adjacent). Defended by default. **Directly relevant to Part 3.**

### 1.4 Misnamed non-tests (pytest collects, ZERO assertions run)

`palette_extractor/test_roundtrip.py`, `palette_extractor/test_wallpaper_extraction.py`
(hardcodes `C:/Users/techm/Desktop/Wallpapers`), `tools/corpus/smoke_test.py` — all
`main()`-only, no `def test_*`. They pass green because pytest finds nothing to run;
any invariant they "check" is enforced only via `python <file>`. (These feed the
Part-2 verdicts — a "test" that never asserts is not a keep-criterion.)

**Restructure takeaway:** a green suite proves the §1.2/§1.3 core only. Items 1.1.1–1.1.10
are what a byte-identity acceptance bar must run *manually*, and #1 (deploy transform)
isn't runnable at all yet — it has no oracle in the repo.

---

## Part 2 — NIBK re-audit (the amnesty bucket)

Adjudicated **every** NIBK file against the four criteria with `git ls-files`
evidence. Legend: **C1** = sole producer of a *committed* artifact; **C1-fail→C2** =
producer of an *uncommitted* live artifact (regen-path keep, flagged at-risk);
**C4** = import/CLI/subprocess/docs-invoked.

### 2.1 The bucket's structural defect

The single biggest correction to `code_inventory.md`: **~24 files were kept as
"sole producer of a committed `data/vN | queries | mining | render_mode` artifact,"
and that is false.** Verified `git ls-files`:

| tree | tracked files | verdict |
|---|---|---|
| `data/palettes` | 7 | **committed** — C1 holds here |
| `data/wallpaper_corpus` | 9 | **committed** — C1 holds |
| `data/label_corpus/batches` | 31 | **committed** — C1 holds |
| `data/classifier` | 1 (`v5/model_best.pt`) | only v5 committed |
| `data/mining` | 0 | **gitignored** |
| `data/queries` | 0 | **gitignored** |
| `data/v4` / `data/v5` / `data/v6` | 0 / 0 / 0 | **gitignored** |
| `data/render_mode_corpus` / `render_mode_head` | 0 / 0 | **gitignored** |
| `data/wallpaper_head` | 0 | **gitignored** |
| `data/atlas_probe` | 0 | **gitignored** |

So the honest justification for the v4/v5/v6 builders, the scorer/mining trainers,
`integrate_dataset`, `calibrate_t2`, etc. is **not** C1 — it's C2 (regen path for an
*uncommitted* live artifact). That is a real keep, but a weaker one, and it means each
of those files is also a marker on the loss-risk map (Part 4). Where a file's data
tree is gitignored *and* its finding is fully captured in MEMORY *and* nothing imports
it, it meets no criterion → ambiguous/dead.

### 2.2 Corrected counts (`.py`/`.rs`; specs-JSON, `.md`, HTML tallied separately)

~151 NIBK code files adjudicated across three area sweeps:

| | stay NIBK | → live | → ambiguous | → dead |
|---|---|---|---|---|
| Area 1 (corpus, classifier, wallpaper, atlas, mining) — 48 | 24 | 0 | 21 | 3 |
| Area 2 (queries, palettes, curation, v4/v5/v6, render_mode_pilot) — 55 | 28 | 3 | 20 | 4 |
| Area 3 (specs, atlas_probe, reframe_probe, coevo, descent_ablation, palette_extractor, palette_lib, dramatic_palettes, eda, misc) — 48 code | 29 | 0 | 18 | 1 |
| **Total code** | **81** | **3** | **59** | **8** |

**70 of 151 move out of NIBK (~46%).** Separately, Area 3 promoted **5 label/viewer
HTML files to live** (`corpus/location/wallpaper_label.html`, `query_label.html`,
`explorer/templates/index.html` — human-label infra, C4), and the 16 mode specs +
`modes_registry.json` + `REGISTRY.md` + the two subsystem `REPORT/PALETTE_EXTRACTION.md`
+ `descent_ablation/README.md` legitimately stay NIBK (committed config/text with
lockstep tests or C1 producers).

### 2.3 The 8 confirmed-dead (finding captured elsewhere / broken / orphaned)

Deletion remains a separate, unauthorized pass — dead ⇒ archive-and-commit-first,
never `rm`.

| file | why dead |
|---|---|
| `tools/eda/eda_common.py` | Imported by **nothing**; built for `prompts/provenance-label-eda.md` + "Panel A–E" scripts that **never existed in-repo**; `__main__` prints a retired enrich taxonomy. (Was the report's lone ambiguous; now settled dead.) |
| `tools/corpus/smoke_test.py` | Asserts "rev4 batch exists with **0 labels**" — false today (rev4 is labeled) ⇒ **fails on run**; stale two-batch era; superseded by committed `test_corpus_reader.py`. |
| `tools/wallpaper/selector_family_diversity_sweep.py` | **Crashes at import**: hard-loads absent `scratchpad/_stage4_cells.json`. Scratchpad-on-dependency-path. |
| `tools/wallpaper/emission_dryrun_v2gate.py` | v2-era dry-run; superseded by v3 head + shipped `emit_v1`; finding in MEMORY (*Emission dry-run (v2 gate)*). |
| `tools/palettes/cliff_diag.py` | Closed cliff study; finding in MEMORY (*Cliff-jarring diagnostic*); report already tagged "prune-on-review." |
| `tools/palettes/softcliff.py` | Closed soft-cliff study (W=0.08 shipped into densifier default). |
| `tools/palettes/render_v2_batch.py` | Cliff-study render/composite; closed; out/-only. |
| `tools/render_mode_pilot/exp_vs_smooth_rankcorr.py` | Finding *fully captured AND acted on* — MEMORY (*exp_smoothing duplicates smooth*) and `integrate_dataset.py:4` drops exp_smoothing. |

### 2.4 Notable ambiguous (the honest "nobody established this needs to be here")

The 59 ambiguous are mostly eyeball montages, closed one-shot studies whose findings
live in MEMORY, and superseded version leaves. The ones worth naming:

- **`looksee/looksee_27_walks.py`** — the canonical bad-keep. Its own docstring says
  "a health check, **not a harness**," no metrics, out/-only, no finding, not
  invoked. The report's "cross-family health-check" was a filename restatement. → ambiguous.
- **classifier version leaves `train.py`(v1), `train_v3.py`, `train_wallpaper_v1.py`,
  `train_wallpaper_resdiag.py`, `sheet.py`, `diagnose.py`, `diagnose_v2.py`** — the
  copy-forward chain is real for v4→v5→v6 (`from .train_v4 import train`; `train_v2`
  is the shared hub pulling in `corpus_data`/`eval_v2`), but these are **dead-end
  leaves** nothing imports, producing uncommitted superseded weights. Provenance
  lineage ≠ a criterion. (NOTE: the report's "`corpus_data.py` superseded, NIBK" is
  *wrong the other way* — it's on the live v6 import path via `train_v2`, so it stays
  a keep.)
- **`tools/atlas_probe/{step0,step0_coverage,step0_reanalysis,efficiency_depth,efficiency_res}.py`**
  — comments call `data/atlas_probe/` "DURABLE" but it's **uncommitted** (0 tracked);
  findings in MEMORY. Whole cluster → ambiguous.
- **`palette_extractor` split is now evidence-backed:** the 6 `bench_*` + 4
  `harvest_*/phase0` scripts each sole-produce a **committed** `tools/viz/*.html`
  (C1 holds — stay NIBK); but `harvest_stats/harvest_wallpapers/build_*_manifest` +
  the two misnamed `test_*.py` write only to **uncommitted** `data/wallpaper_harvest/`,
  `data/palette_viz/`, `data/palette_roundtrip/` → ambiguous.
- **`viz_transfer.py`** → ambiguous — its "load-bearing bit-parity assert" is
  redundant with `check_bytematch.py` + `tests/palette_bytematch.rs` +
  `colormap_acceptance.py` (Part 1 item 1.1.6), so it guards nothing unique.
- **Coldstart/gvo A-B closers** (`compare_v1_v2_diversity.py`, `scorer/gvo_experiment.py`),
  **cross-family/v6-swap closers** (`atlas/{cross_family_shakeout,verify_v6_gate,
  v5_v6_anchor_diff,gather_overnight}.py`), **curation studies**
  (`conditioned_colorize`, `soft_spread_calibrate`, `morphology_dedup`) — findings in
  MEMORY, uncommitted or out/-only outputs, not invoked.

### 2.5 What legitimately stays NIBK (the defensible core)

- **C1 — committed-artifact producers (solid):** the ~13 batch builders mapping 1:1
  to a committed `data/label_corpus/batches/<id>` / `data/wallpaper_corpus/batches/<id>`
  (`build_enrich_batch`, `build_rev4_batch`, `import_loose0_v3`, `recolor_gather_v6`,
  `harvest`, `build_bootstrap`, `build_headbatch_dramatic`, `julia_ladder/build_j0`);
  the 3 palette producers (`build_categories`, `build_pool`, `build_features` — the
  **only** committed-data producers in the queries/palettes/mining surface);
  `train_v5` (→ the one committed weight); `gen_registry` (→ committed `REGISTRY.md`);
  the `palette_lib.build_sheet` chain (→ committed `clean_colormaps.json`, ~18
  readers); the `bench_*` HTML producers.
- **C2 — regen path for an uncommitted **live** artifact (keep, but flag at-risk):**
  `train_v6` / `train_mining_head` / `train_wallpaper_v3` / `scorer/train_v3_gvo`
  (live gates/heads, weights gitignored **by design** per CLAUDE); the v4/v5/v6
  `build_manifest`/`build_plan`/`assemble`/`build_roster` lineage (frozen-cache regen
  + the byte-identity asserts of Part 1 item 1.1.4); the `render_mode_pilot`
  build→render→smooth→signal→`integrate_dataset` chain (→ `dataset_v1` → live mining
  gate); `warmstart_v1`/`prefv2_dramatic_v1` (sole generators of the pref training
  batches); `recolor_pass` + its `colored_clip_spread`/`colorize_assign` algo deps.
- **C2 — re-runnable parity/QA on the *current* pipeline:** `check_bytematch.py`
  (drives `palette_bytematch.rs`), `lock_mining_gate.py`, `confirm_report.py`,
  `check_ledger_decode_version.py`, `validate_palettes.py`, `descent_ablation`,
  `morning_readout.py`.
- **C4 — import/CLI/docs-invoked:** `merge_scores.py`, `explorer/app.py` (via
  `explorer.cmd`), `launch_query_label_server.py`, `dedup.py`, `palette_families.py`,
  `surfacing_eval.py`, plus the classifier import hubs (`train_v2/train_v4/eval/
  eval_v2/data_v4/corpus_data`).
- **Reclassified NIBK→live** (module-level imported by the live seeder path):
  `regenerate_coldstart_v2.py`, `diversity_diagnostic.py`, `color_metrics.py`
  (imported by `sample_location.py` → `build_fresh_discovery.py`), and the 5 label/
  viewer HTMLs.

---

## Part 3 — Is `data/library_embeddings/embeddings.npz` regenerable?

**Verdict: NO — commit the frozen base now; keep the shard stream ignored.** Byte-
identical regeneration is impossible for any array; one array can't be regenerated at
all.

The file (verified on disk) is three tiers:

| array | shape | regenerability |
|---|---|---|
| `colored_clip` | 564×768 | **value-approximate.** `colored_clip.py` takes only coords+recipe from `library_records.jsonl` (full Location + complete coloring recipe per candidate), re-dumps the smooth field via `render-one` (deterministic; `out/curation/morph_fields` is a regenerable cache), recolors through **committed** `data/palettes/{pool_colormaps,palette_features}.json`, embeds with a name-pinned timm CLIP (`vit_base_patch16_clip_224.openai`, `is_training=False`). No wiped input — **but** CLIP inference is GPU-float-nondeterministic and weights resolve by *name* not hash ⇒ cosine ≈1.0, **not bytes**. |
| `morph_clip` | 62×768 | **value-approximate, and the as-written producer is broken.** In `colored_clip.py` these are promoted verbatim from `data/library_embeddings/gray_embeddings.npz`, which is **absent on disk** — so `colored_clip.py build()` crashes today at `np.load(GRAY_NPZ)`. Regen is possible only via the *recovered* producer (`library_annotate.morph_gray_image`, robust-z tanh `K=2, MAD≈1.4826`; `docs/findings/morph_parity.md` self-cos `0.99999988–1.00000024`). That finding explicitly calls the exact triple a **"sharp optimum, load-bearing"** — the next-best variant drops to 0.97 and *flips the 0.974 dedup verdicts*. |
| `morph_v6` | 62×1280 | **NOT regenerable.** Sourced verbatim from the wiped `gray_embeddings.npz`; `library_annotate` **deliberately skips v6 for fresh locations** ("not free → skipped") and flags it `UNFIT (saturates)`. **Its only surviving copy is inside this `embeddings.npz`.** This single array makes the file irreplaceable regardless of the CLIP story. |

Determinism note ties to Part 1 item J: the morph *formula* is pinned by default
pytest (`test_prospect.py:277`), so a rebuild is value-reproducible — but under the
verdict-sensitive 0.974 dedup threshold even ~1e-6 drift is decision-relevant, and the
`morph_v6` array has no formula to pin.

**Commit policy (follows directly):**
1. **Commit the frozen base now** — `data/library_embeddings/embeddings.npz` +
   `data/library/library_records.jsonl`. Add `.gitignore` `!`-negations mirroring the
   existing allowlist (`!/data/library_embeddings/embeddings.npz`,
   `!/data/library/library_records.jsonl`). Justification: sole surviving copy of
   `morph_v6`; the CLIP arrays regenerate only value-approximate under a
   verdict-sensitive threshold; the direct promotion source (`gray_embeddings.npz`) is
   already lost.
2. **Do NOT commit the growing shard stream** — `data/library_embeddings/shards/*.npz`
   (dir not yet created; no cycle has appended). By design (`library_store.py`) shards
   hold only `morph_uids` + `morph_clip` via the recovered deterministic-from-coords
   producer, are atomic/crash-safe, dedup-idempotent on `location_id`, and the loader
   concatenates base+shards last-writer-wins. They're a regenerable overlay; blanket-
   allowlisting the directory would put a new binary blob in git every cycle — which is
   the plausible reason the `/data/*` ignore was left in place. Keep `shards/` ignored;
   negate the base file **by exact path**, not the directory.

This resolves the open decision in `code_inventory.md` §3: the base is *not*
regenerable, so it is committed regardless; the answer additionally tells us the
directory-level allowlist the report sketched (`!/data/library_embeddings/`) is the
**wrong** grain — it would sweep in the shard stream. Negate the file.

---

## Part 4 — Expanded §3 loss-risk surface (what Parts 2+3 surface)

`code_inventory.md` §3 named 3 uncommitted at-risk artifacts. The NIBK audit's
C1-fail set is a superset: **every gitignored `data/` tree with a live consumer is a
loss-risk node.** Triaged by how recoverable it is:

**Tier A — irreplaceable, no regeneration path (fix first):**
- `data/queries/labels/{coldstart_v2,warmstart_v1,prefv2_dramatic_v1}.json` — the
  **human pref-labels** that trained the ACTIVE pref-v3-gvo head. **Absent on disk in
  this checkout AND uncommitted** (confirmed: `git ls-files data/queries/labels`
  empty, dir absent). Single-copy human work with no backup here — this is the worst
  node on the map, worse than the library data.
- `data/library_embeddings/embeddings.npz` `morph_v6` array — Part 3.

**Tier B — regenerable only value-approximate / with a byte-identity recipe that
isn't run in CI:**
- `data/library_embeddings/embeddings.npz` (CLIP arrays), `data/library/library_records.jsonl`
  — Part 3.
- `data/v{4,5,6}/` manifests + `aug_cache*` frozen JPGs — regen recipe is the
  `build_plan`/`build_manifest` scripts, but the byte-identity asserts (Part 1 item
  1.1.4) run only when invoked manually; the frozen JPGs themselves are uncommitted.
- `data/render_mode_corpus/dataset_v1` — trains the live mining gate.

**Tier C — gitignored by design (CLAUDE: "weights/scratch expected"), regenerable IF
Tier A/B inputs survive:**
- `data/classifier/v6`, `data/render_mode_head/v1`, `data/wallpaper_head/v3`,
  `data/queries/scorer/v3_gvo` — live model weights. Safe to leave ignored **only
  because** a committed trainer + committed training data can rebuild them — which is
  exactly what fails if the Tier-A pref-labels stay lost. The weights aren't the risk;
  their uncommitted *inputs* are.

The through-line: the report's "0 confident dead" fell out of the bucket's design, and
its "3 at-risk artifacts" undercounted for the same reason — a gitignored `data/` tree
was silently read as "fine." It isn't. The corrected picture is **8 dead, 59 ambiguous,
and a loss-risk surface whose apex is the absent human pref-labels**, not the library
embeddings.

### Suggested order of operations (still nothing applied)
1. **Confirm/restore the Tier-A pref-labels** off-machine before anything else — they
   have no regeneration path at all.
2. **Commit the library frozen base by exact path** (Part 3), not directory.
3. **Build a `present.rs` ↔ `Transform` deploy-parity oracle** (Part 1 item 1.1.1) —
   it's the one crown-jewel guarantee with no test in the repo, and the restructure
   will touch that boundary.
4. Only then proceed to the T2/T3 moves in `code_inventory.md` §7, running the
   off-by-default parity gates (Part 1 §1.1) manually before/after each byte-sensitive
   move.
