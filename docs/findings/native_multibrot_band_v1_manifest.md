# Native multibrot band batch — `2026-07-22_native_multibrot_band_v1`

**Generator version:** `native_multibrot_band_v1`  ·  **Total rows:** 300  ·  **Render:** 640×360 ss2 `twilight_shifted` (the scored presentation)  ·  **Score axis:** v7 p_good, t_good measured = 0.5

Labels-input only. Native mb3/mb4/mb5 have never been human-measured; this batch calibrates their t_good (currently an uncalibrated 0.50). Threshold analysis comes after labeling.

## Per-family × per-band counts (requested vs realized)

Target 20/band → 100/family. Bands are v7 p_good; the split at 0.50 is the measured operating point.

### multibrot3  (pool 767, selected 100)

| band (p_good) | available | picked | source: campaign_v7 / gather_v6→v7 |
|---|---|---|---|
| 0.00-0.20 | 253 | 20 | 20 / 0 |
| 0.20-0.35 | 175 | 20 | 20 / 0 |
| 0.35-0.50 | 106 | 20 | 20 / 0 |
| 0.50-0.65 | 106 | 20 | 20 / 0 |
| 0.65-1.00 | 127 | 20 | 20 / 0 |

### multibrot4  (pool 729, selected 100)

| band (p_good) | available | picked | source: campaign_v7 / gather_v6→v7 |
|---|---|---|---|
| 0.00-0.20 | 483 | 20 | 0 / 20 |
| 0.20-0.35 | 69 | 20 | 1 / 20 | (+1 filled into other bands)
| 0.35-0.50 | 19 | 19 | 4 / 15 |
| 0.50-0.65 | 18 | 18 | 5 / 13 |
| 0.65-1.00 | 140 | 20 | 21 / 1 | (+2 filled into other bands)

### multibrot5  (pool 925, selected 100)

| band (p_good) | available | picked | source: campaign_v7 / gather_v6→v7 |
|---|---|---|---|
| 0.00-0.20 | 231 | 20 | 20 / 0 |
| 0.20-0.35 | 243 | 20 | 20 / 0 |
| 0.35-0.50 | 162 | 20 | 20 / 0 |
| 0.50-0.65 | 143 | 20 | 20 / 0 |
| 0.65-1.00 | 146 | 20 | 20 / 0 |

## Source ledgers

- **Admissions** (score = `p_good`): `outcome_ledger.jsonl` in campaign1/breadth, campaign1/dive, campaign2/breadth, campaign2/dive (native `multibrot{3,4,5}`, `julia_c` null).
- **Canon-rejects / sub-cut** (score = `canon_pgood`): `harvest_log.jsonl` in campaign2/breadth, campaign2/dive (`admitted=False`; campaign1 harvest carries no coordinates → excluded).
- **mb4 sub-cut fill** (score = v7-rescored `p_good`): `gather/multibrot4/outcome_ledger.jsonl`, every distinct candidate re-rendered at 640×360 ss2 and re-scored under v7 (`tools/corpus/rescore_gather_mb4_v7.py`; cache `batches/2026-07-22_native_multibrot_band_v1/mb4_gather_v7_rescore.jsonl`). v7-decoded native mb4 has only ~7 sub-0.5 rows, so its below-threshold bands come entirely from this current-decoded re-score.

Each row records its exact `provenance.src`/`lineage`/`ledger_id`/`scorer_version` (per-item source is recoverable from `images.jsonl`).

## Provenance / pool-safety

`gather_v6_rescored_v7` rows are LABEL-BATCH material only (`source=gather_v6_rescored_v7`, `scorer=v7`, `lineage=gather`). They are **not** ledger admissions and were **not** written to any discovery ledger or generation/pool path.

## Registration

- Batch registered in the store: `data/label_corpus/batches/2026-07-22_native_multibrot_band_v1/` (`images.jsonl` + `batch.json` + `crops/`). Never-delete.
- Empty label sidecar created: `labels/native_multibrot_band_v1.json` (`{}`).
- **Sidecar registry entry is DEFERRED** — adding an empty sidecar to `label_store.SIDECAR_LABELS` now makes `assert_sidecars_joined` raise for the whole corpus (the batch joins 0 rows until labeled). After labels are exported, add this one line to `tools/corpus/label_store.py`:

```python
    "2026-07-22_native_multibrot_band_v1": "native_multibrot_band_v1.json",
```

## Labeling

- View: `tools/viz/corpus_label.html` (blind — no scores shown; family may be visible). `image_id` collides across batches, so calibration reads must key on `render.cx/cy/fw` + `fractal_type`, all stamped per row.
- `label.score ∈ {null,1,2,3}` (bad/okay/good). Export → the sidecar above.
