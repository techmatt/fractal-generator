# q4 window label store

A **separate, distribution-bound** label store for the q4 stage-1 gate — NOT the v7
location corpus (`data/label_corpus`) and deliberately never pooled into it. It holds
16:9 **windows** swept over a set of minibrot renders, score-stratified and
pre-filtered by construction, so its label distribution is biased on purpose. Pooling
it blind into the version-invariant v7 union would poison that union.

- **Producer:** `tools/studies/q4_stage1_labelset.py` (stages: minibrots → fields →
  sweep → stratify → capture).
- **Canonical reader (single source of truth):** `tools/corpus/q4_window_reader.py`.
  Every consumer of q4-window labels routes through it. It is intentionally absent
  from `tools/corpus/label_store.py::SIDECAR_LABELS`.
- **Label UI:** `tools/viz/q4_window_label.html` (three-way, one-key accept/reject).

## Layout

```
data/q4_window_corpus/batches/<batch_id>/
  windows.jsonl     one row per window (schema below)
  crops/<window_id>.jpg   the window thumbnail (crop of the parent medium render)
  meta.json         batch provenance: geometry, scales, prefilter, stratification
  scores.json       (added by the label export) {window_id: class-string}
```

## `windows.jsonl` row

| field             | meaning |
|-------------------|---------|
| `window_id`       | stable id (`<minibrot_id>_s<scale×1000>_<rect-hash>`) == `crops/<id>.jpg` |
| `minibrot_id`     | parent nucleus render id (`mbNN_pPP`) |
| `period`          | nucleus period |
| `render`          | parent MEDIUM-render geometry: `cx`/`cy`/`fw` (decimal strings), `maxiter`, `family`, `width`, `height`, `aspect`, `palette` |
| `window`          | frame-normalized rect `{u,v,w,h}` within the parent render |
| `scale`           | window width as a fraction of frame width |
| `band`            | composite-score stratification band (`0..N_BANDS-1`) |
| `score_composite` | the `score_A` composite the stratification used |
| `features`        | field-stat feature vector (`compute_metrics` keys) — **fitting-ready** |
| `label.klass`     | `null` \| `"accept"` \| `"reject"` \| `"filter_leak"` |

## Three-way labels

- **accept** — "worth stage-2's time". A **high-recall** gate — err toward accept;
  stage-2 filters.
- **reject** — "clean window, just not q4-worthy".
- **filter_leak** — "dead / noisy / barren garbage the step-3 pre-filter should have
  dropped". This is **feedback on the pre-filter, not a quality judgment**. It is
  **excluded from the accept-vs-reject fit** (`iter_labeled`) and surfaced only as a
  leak-rate diagnostic (`prefilter_leak_rate`). Keeping it a rare exception tag is
  what lets accept/reject stay one-key fast paths.

## Mutation rule

`label.klass: null → value` is the ONLY allowed mutation. A merge that would change a
non-null class must warn and refuse (same contract as the v7 corpus `label.score`).

## Caveats (see `docs/findings/q4_stage1_labelset.md`)

1. Windows are crops of **one medium render**, so small-window frequency stats are
   **scale-biased**. A true-scale per-window re-render is a stage-2 refinement.
2. `--dump-field` lives only on the f64 `render-one`, so the sourced minibrots are
   floored at an **f64-dumpable depth**. Minibrots are self-similar, so a
   moderate-depth period-p nucleus is a valid compositional proxy for a deep one;
   deep-specific precision behavior is exactly what stage-2's true-scale re-render
   validates.
