# Too-close-to-smooth — distance pass + calibration montage

Flag pilot rasters that are effectively their smooth counterpart, so they can be disregarded when we build the render-mode head corpus. Runs independent of labeling.

Batch: `data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1/` (`images.jsonl` + `crops/`).

**1. Smooth counterparts.** For each of the 500 rasters, render its smooth counterpart — same location, palette, and approved color params, 1280×720 ss2, smooth field (the location's real deployable smooth form). Dedup by (location, palette, color) so shared counterparts render once.

**2. Distance-to-smooth**, per raster — compute **both** mean Lab ΔE and **1−SSIM** (structural). Keep both; they catch different kinds of sameness (color/brightness vs structure).

**3. Calibration montage.** Rasters sorted closest→farthest from smooth, each shown beside its smooth counterpart with both distance numbers, so we can eyeball where "basically smooth" ends. Anchor the flagged purple raster in the strip if it's identifiable.

**4. Propose, don't write.** Suggest a cutoff on each metric from the sorted order; do **not** set flags yet. Report the montage, proposed thresholds, how many rasters fall below each, and the per-mode breakdown of what would be flagged.
