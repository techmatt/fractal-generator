# Emission-corpus curation → the 47 distinct locations

**Context.** The `overnight_20260713_001420` emission produced **62 emitted rows**. The location
library is built on the *distinct* subset, not the raw 62 (recolors and viewport-dups would
otherwise inflate every downstream count). `scratchpad/curate_emission.py` was the one-shot that
collapsed 62 → 47, materialized as `deploy_tail`'s input manifest under
`out/wallpaper/overnight/overnight_20260713_001420/emit/curated/`. It is hardcoded to that single
run and writes only under `out/` (disposable), so it is not a general tool — but its curation
*decisions* are non-obvious and are recorded here.

**Dedup layers (in order applied):**

1. **Coordinate** (`reconstruct.py` / `clusters.json`): 7 non-phoenix RECOLOR groups always
   collapse + 4 phoenix cross-cycle groups flagged. → 10 non-phoenix recolor members drop.
2. **Phoenix viewport re-check** (this pass): the coordinate rule *over-merges* phoenix — fixed
   Ushiki `c`/`p`, `fw` spans decades, so identical `(c,p)` ≠ identical image. Read the z-plane
   `(cx,cy,fw)` viewport instead: identical viewport → true recolor-dup (drop); distinct viewport
   → keep back.
   - G1 (rep `c1/wfd_014_05`): 2 drops are DISTINCT (0.56× & 2.8× zoom apart) → **KEEP BACK**.
   - G3 (rep `c2/wfd_003_10`): 1 drop DISTINCT (0.76 frame / 2× zoom) → **KEEP BACK**.
   - G2 (rep `c1/wfd_015_00`): 1 drop SAME viewport (7% pan / same zoom) → drop.
   - G4 (rep `c4/wfd_008_06`): 1 drop SAME viewport (8% pan / 5% zoom) → drop.
   → 2 phoenix true-dups drop, 3 distinct phoenix kept back.
3. **Morphology** (`morphology_dedup.json`, phoenix-excluded): 1 julia + 2 julia_mb3 drop.

**Keeper rule.** Per collapsed group, keeper = **argmax head `p_ge3`** (the explicit task rule).
Note this diverges from `clusters.json`'s fitness-picked representative in 3 recolor groups, and
matches what the morphology report already used.

**Arithmetic.** `62 − 10 recolor − 2 phoenix-dup − 3 morphology = 47 distinct.`

**Non-destructive.** The raw 62 stay in `cycle_*/manifest.jsonl`; the curated set is a NEW manifest
under `emit/curated/` (image_id was made cycle-unique since it collides across cycles). The
per-group decisions + id map are persisted in `emit/curated/curation_report.json`.
