# Integrate + analyze v6 labels — no training

## Goal
Labeling of the gather_v6 batch is complete; labels are at `location_labels_gather_v6.json`. Fold them into the corpus, commit, and run basic analysis. **No training and no manifest/split build** — that's the next session. This is integrate + commit + analyze only.

## 1. Inspect the label file
Read `location_labels_gather_v6.json`: schema, keying (image_id vs location), score values (expect 1/2/3), count, and **coverage vs the batch's 640 crops** (labeled / skipped). If it's location-keyed rather than image_id-keyed, note how it maps onto crops.

## 2. Fold into the corpus
Fold the labels into `data/label_corpus/batches/2026-07-05_gather_v6/images.jsonl` via the canonical merge (`merge_scores.py --batch 2026-07-05_gather_v6 …`, or the correct tool if the file isn't the `scores.json` shape — inspect first). Enforce **null→value only**; report any non-null conflicts refused. **Dry-run first, then apply.** Report folded count and final labeled/total.

## 3. Commit
Commit the merged `images.jsonl` + the label file with a clear message. (Explicitly requested — go ahead and commit.)

## 4. Analysis
Join the folded labels with each crop's provenance (class, `selection_role`, `guard_verdict`, `decoded_class`, `k3`). Report as tables.

**Core distributions**
- Overall label distribution (1/2/3 counts + fractions).
- **Per class** (the 9: mandelbrot, multibrot3/4/5, julia:mandelbrot, julia:multibrot3/4/5, phoenix): distribution + good-rate. Which families yield good locations.
- Per `selection_role` (best / random_eval / disagreement): distribution. Did `best` beat `random_eval` on good-rate (selector working)? How did `disagreement` (guard-fail, high-k3) get labeled — mostly bad (guard was right) or some good (v5 was right)?

**Model-vs-human diagnostics** (the labels are now ground truth)
- **v5 decode precision:** of `decoded_class==3` ("good"), what fraction labeled good / okay / bad — **broken out per family** (this is the "does v5 generalize OOD to the new families" question v6 exists to answer). Same for `decoded_class==2`.
- **k3 as a ranker:** rank correlation / AUC of continuous `k3` vs the human label. Is k3 a good good-predictor (it's the v6 selection ranker)?
- **guard verdict vs human:** how were guard-flagged (flat / interior) crops labeled? Do the degeneracy flags align with human "bad"? (Feeds the open new-family guard-calibration question.)

**Training-readiness flags** (report, don't act)
- Per-class label balance — flag any class that's nearly all-good (few negatives, e.g. the julia families) or too sparse to train well.
- Quick sanity: the ~55 previously-dark, since-recolored crops — were they labeled (i.e. labelable after recolor)? How?

## Report
The tables above, the fold + commit confirmation, and a one-paragraph training-readiness note: the batch is folded + committed and `assemble.py` would pick it up; the union-add + location-disjoint manifest/split bump remain for the next-session train. No training, no manifest build here.
