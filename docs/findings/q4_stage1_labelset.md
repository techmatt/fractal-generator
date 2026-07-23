# q4 stage-1 labeling harness — the stratified window set

**Status:** windows ready to label + capture wired. **Stops before fitting** (the
basic-function fit + per-scale heatmap is the next prompt, after labels exist).

Producer `tools/studies/q4_stage1_labelset.py` (stages minibrots → fields → sweep →
stratify → capture). Store `data/q4_window_corpus/` (README = schema contract).
Reader `tools/corpus/q4_window_reader.py`. Label UI `tools/viz/q4_window_label.html`.

## What it does

Find ~30 varied minibrots, render each at 4×size (the island framed by its
decoration ring), sweep 16:9 windows at 3 scales, drop **obvious** garbage, then
**stratify survivors by the current composite score** into equal-count bands and
sample ~300 across bands. Present in a fast three-way accept / reject / filter-leak
flow, captured to a registered, single-reader store SEPARATE from the v7 corpus.

**Why stratify-by-score is the whole point.** Matt's hand-drawn good picks ranked
*mid-pack* (22–82) on the label-free `score_A` composite
(`docs/findings/q4_sweep_validation.md`), so uniform or top-K sampling never surfaces
the mid-band good windows the re-fit must learn to promote. Equal-count score bands
force the mid-band into the labeled set.

## Pipeline

1. **Minibrots** — `deep_center_finder` Newton-refines a grammar of valley anchor
   seeds × periods 4–65 to nucleus centers; keep converged, minimal-period,
   f64-dumpable-depth, dedup by center, select ~30 spanning the period range. Each
   emits `fw = 4×size` (frames the ring — established) + a depth-scaled maxiter.
2. **Fields** — `render-one --dump-field` per minibrot at **2176×1224 ss1** (16:9),
   so the smallest window (scale 0.06) is ~130 px. Resumable; run detached.
3. **Sweep** — 16:9 windows at scales **[0.06, 0.09, 0.14]** (spanning Matt's box
   widths 0.057–0.099 with headroom), stride 0.30w; `compute_metrics` + `score_A`
   transferred verbatim from `q4_neighborhood_sweep`; greedy NMS (IoU>0.35) per
   minibrot for spatial de-dup.
4. **Pre-filter — obvious rejects only** (loose; keep borderline): `interior_frac >
   0.85` (dead black) | `flat_frac > 0.92 ∧ occupancy < 0.06` (barren/sparse) |
   `busy_frac > 0.15` (speckle). Reject counts are reported per reason.
5. **Stratify** — equal-count (quantile) score bands over the survivors; sample to
   ~300 with **round-robin across minibrots within each band** for spatial/period
   diversity; top up from leftovers to hit the target.
6. **Capture** — colorize each field **once** with `tools/colormap.py` (field⊗colormap
   split — no second deep render) and crop the selected windows from that single
   medium render into `crops/<window_id>.jpg`; write `windows.jsonl`.

Each window records: minibrot id · frame-normalized rect · scale · band ·
`score_composite` · the `compute_metrics` **feature vector** (fitting-ready) · thumbnail.

## Three-way label (accept / reject / filter-leak)

- **accept** — "worth stage-2's time" (**high-recall** gate; err toward accept —
  stage-2 filters).
- **reject** — "clean window, just not q4-worthy".
- **filter_leak** — "dead/noisy/barren garbage the step-3 pre-filter should have
  dropped". This is **feedback on the pre-filter, not a quality judgment**: excluded
  from the accept-vs-reject fit (`iter_labeled`) and surfaced only as the pre-filter
  **leak-rate diagnostic** (`prefilter_leak_rate`). It is the rare exception tag, so
  accept/reject stay one-key fast paths.

## Store separation

The store is **distribution-bound** (score-stratified, pre-filtered — biased by
construction). It is intentionally NOT registered in
`tools/corpus/label_store.py::SIDECAR_LABELS`; pooling it blind into the version-blind
v7 union would poison that union. The single canonical reader
`tools/corpus/q4_window_reader.py` owns the layout + the three-way resolution rule.
`label.klass: null → value` is the only allowed mutation.

## Caveats

1. **Scale bias.** Windows are crops of ONE medium render, so small-window frequency
   stats (`fine`/`struct_e` bands in `compute_metrics`) are **scale-biased** — a
   window at scale 0.06 sees the same pixel grid as one at 0.14, not its own true-scale
   render. A per-window true-scale re-render is a **stage-2 refinement**, not now.
2. **f64-dumpable depth floor.** `--dump-field` exists only on the f64 `render-one`
   (the perturbation paths can't dump a field), so the sourced minibrots are floored
   at `size ≥ 1e-10` (`fw ≥ 4e-10`, spacing > 1e-13 at 2176 px). Minibrots are
   self-similar, so a moderate-depth period-p nucleus is a valid **compositional**
   proxy for a deep one (period governs decoration density, not the island shape);
   deep-specific precision behavior is exactly what the stage-2 true-scale re-render
   validates. Extending `--dump-field` to the perturbation backend would lift this
   floor — deferred, out of scope for "reuse, don't rebuild".

## Results (batch `2026-07-23_q4_stage1_windows`)

- **Minibrots:** 30, periods **4–65** (30 distinct), `fw` 4.67e-10 … 7.90e-2. The
  finder reproduced Matt's boxed centers exactly — `mb19_p35` at fw 8.069624e-10 and
  `mb27_p58` match the `q4_sweep_validation` p35/p58 frames. Field dumps: 30/30 clean
  in ~36 min (deepest ~107 s @ maxiter ~14 k).
- **Sweep:** 25 281 windows (3 scales × NMS IoU>0.35 per minibrot).
- **Pre-filter (obvious rejects only):** 6 180 dropped — `interior_heavy` 4 395,
  `barren` 1 785, `speckle` 0 (minibrot renders aren't speckly). **19 101 survivors.**
- **Stratified sample:** **300** windows, exactly **50 per score band** across 6
  equal-count bands; **all 30 minibrots** covered (7/10/12 per-mb min/median/max).
  Scale mix 0.06/0.09/0.14 = 244/17/39 (small windows dominate, matching Matt's
  0.057–0.099 picks).
- **Band gradient (mean over each band's 50 windows):** score −0.50 → +0.33 with
  `mid_detail_frac` rising monotonically 0.156 → 0.658 — the quality spread the
  re-fit must learn. Accept-worthy compositions (spirals, island-in-ring, distributed
  decoration) appear across bands 2–5, confirming the mid-pack spread that makes
  uniform/top sampling inadequate.
- **Store:** 300 crops, 0 missing / 0 orphan vs `windows.jsonl`; all `label.klass`
  null. Band-stratified montage `out/q4_stage1/_bands_montage.png`.

## Launch labeling

Open `tools/viz/q4_window_label.html` in a browser (auto-loads
`data/q4_window_corpus/batches/2026-07-23_q4_stage1_windows/windows.jsonl`). Keys:
**a** accept · **r** reject · **f** filter-leak · ←/→ move · **u** next unlabeled.
Export → `scores.json`; drop it in the batch dir; `q4_window_reader` joins it.
