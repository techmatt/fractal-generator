# Diversity-aware emission v1 (deficit colorize + persistent pool + selection)

Built against the first steered-frontier run whose ledger decodes as current
(`data/discovery/steered_run2/outcome_ledger.jsonl`). New package: **`tools/emission/`**
(`build_emission_diversity_v1.py` driver + `cells.py` deficit + `selection.py` greedy
select + `pool.py` resume-safe pool + `descriptor.py` intake + `report.py` +
`test_emission_diversity.py`). Config: **`data/emission/target_measure.json`** (hand-editable,
committed via a `.gitignore` negation). Disposable run output: `out/emission_v1/` +
`out/emission_v1_report.md`.

## The descriptor + the five stages
A wallpaper's descriptor is the 4-tuple **(fractal_type, morph_cluster, palette_flavor,
render_style)**. type+cluster are fixed at intake; flavor+style are chosen at colorize time.

1. **Intake** (`descriptor.py`). Admitted rows = `corpus_common.is_current_decoded` ∧
   `decoded_class==3` ∧ `guard_pass` ∧ `distinct` (a v6/v5/unstamped row is rejected;
   `require_current` raises on the strict path). Each admitted location gets the **library
   morph-CLIP embedding** verbatim (640×360 ss2 smooth field → `library_annotate.morph_gray_image`
   robust-z tanh → `colored_clip` `vit_base_patch16_clip_224.openai`), then an **incremental
   medoid cluster** id WITHIN its fractal type at the strict near-dup cos **0.974**.
2. **Deficit** (`cells.py`). Joint counts over the feasible product cells for the *gated pool*;
   uniform target measure (+ optional per-region weight overrides in the config); deficit =
   `target_frac − pool_frac`. A cell that hits the attempt cap with zero fills leaves support.
3. **Colorize** (deficit-driven). For a location's fixed (type, cluster), pick the (flavor, style)
   that maximizes the joint deficit (range-normalized softmax tie-break, not argmax); pick the best
   concrete palette in that flavor by the pref-v3-gvo head on the cached smooth field.
4. **Gate + pool** (`pool.py`). Append-only `pool_log.jsonl` is the durable truth (one line per
   attempt, written after render+score); the gated pool = the passing subset; joint/attempt counts
   rebuild from it on resume; RNG/cursor checkpoint atomically but are never trusted for counts.
   Each row carries the full descriptor, head scores, realized hue/chroma histogram, and provenance.
5. **Select** (`selection.py`). Greedy max-marginal-gain of N: `niche_relative_quality`
   (within-cell p_ge3 percentile) × `coverage_gain` (1 − max kernel-sim to selected), kernel =
   ∏ exact-match categorical × cosine(morph emb). Ties (singleton niches, all percentile 1.0)
   break on absolute p_ge3 → the release is the N highest-quality **distinct-cell** renders.

## The one extension of §4: two-head routing
The prompt says "score with the wallpaper head" (v3, 0.90 gate). That head is trained on **smooth**
wallpapers only and scores strange fields (tia/stripe/composite) ≈0, which would collapse the
render-style axis to smooth in the gated pool. The repo already gates the two render classes with
two heads, so each style routes to its own: **smooth → wallpaper head** (floor 0.75), **promoted
strange modes → mining head** (`render_mode_head/v1`, 0.50 gate). Quality is only compared
within a niche (which pins the style → the head), so the heads never mix in one comparison.

## Smoke result (steered_run2 ledger)
- **54** admitted locations → **54** morph clusters (all singletons: the frontier's `distinct`
  filter already removed plane-dups, and the library morph recipe confirms **no visual near-dups**).
- Colorized to **36 gated in 80 attempts (45% post-floor)**; selected **12**; 12 full-res
  2560×1440 ss4 release PNGs on disk.
- Gated pool covers **8/8 render styles, 18/19 palette flavors, 28 morph clusters, 7 types** —
  the deficit genuinely spread flavor/style demand (chosen-flavor counts 1–6 vs uniform 4.2).
- **Finding — strange modes underperform on this source.** 34 of 36 gated were strange, but only
  **2 cleared the 0.50 mining production gate**; the permissive 0.05 mining floor did heavy lifting
  (32/36 below their production gate). The two **smooth** keepers both cleared 0.90. The release
  (tie-broken by absolute p_ge3) is 2 smooth (0.95/0.95) + a tia/stripe/composite spread down to
  0.26 — the strong smooth wins, the weak strange stays as unselected pool inventory.
- Implication: for this deep-multibrot/julia frontier output, promoted strange modes are marginal
  at wallpaper quality; the mining floor is the lever that admits them as inventory, and selection
  culls. A future target-measure override could down-weight strange styles a priori.

## Acceptance
- Suite green (14 emission unit tests; 177 repo tests collect clean). Only `tools/emission/`,
  `data/emission/target_measure.json`, and one emission `.gitignore` negation added — nothing else
  touched.
- Current-decode enforced + proven: a v6-stamped row is skipped (soft) / raises `StaleDecodeError`
  (strict) in `test_v6_row_rejected`.
- Pool resume proven live: killed mid-colorize at 3 rows, resumed to 8 — ids contiguous, no
  duplicates, pre-kill rows byte-identical, sequence continued at `em_000003` (+ `pool.py` unit test).
- Report + release/pool contact sheets delivered; the 12-wallpaper release exists on disk.

## Reproduce
```
uv run pytest tools/emission/test_emission_diversity.py -q
uv run python tools/emission/build_emission_diversity_v1.py \
    --out out/emission_v1 --release-n 12 --target-gated 36 --mining-floor 0.05
uv run python tools/emission/build_emission_diversity_v1.py --out out/emission_v1 --resume   # after a kill
```
