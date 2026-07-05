# Recolor gather_v6 label batch from the 76 curated palettes + make the 76 the location-crop default

## Goal
The `2026-07-05_gather_v6` label batch was rendered from the 777 pool, which introduced too-dark, unlabelable crops. Recolor the whole batch from the **76 curated q3 palettes** (`score3_colormaps.json`) — bright by construction, so no dark-palette crop can remain — and make the 76 the **default** palette source for the location label-crop render path so this can't recur. Location quality is palette-invariant, so recoloring changes appearance, not label validity; `image_id` is preserved throughout.

## 1. Recolor all 640 crops
For each row in `data/label_corpus/batches/2026-07-05_gather_v6/images.jsonl`:
- Read its render block for coords / family / viewport. **Julia rows:** reconstruct the viewport from `julia_z_cx/cy/fw` + `c`, with `render_family` mapped (`julia`, `julia_multibrotN`) — the same reconstruction the bootstrap driver uses.
- Draw a random palette from the **76** (`score3_colormaps.json`), carrying its palette-type metadata.
- Render at the locked label spec via the **general path validated in the bootstrap**: `render-one --dump-field` (1280×720 ss4) + `colormap.render_candidate` (`filter=lanczos3`) → q90 JPG, `interior_mode=black`, at the row's own coords/viewport (which preserves the existing framing). **Not `enrich --mode render`** — it's mandelbrot-only / named-palette and can't express the non-mandelbrot families.
- Overwrite `crops/<image_id>.jpg` (`image_id` unchanged). Update the row's provenance palette to the 76-palette actually rendered; leave every other provenance field and the null `label` untouched.

Rendering the same coords/viewport with a new palette means the only thing that changes per crop is the coloring. The high-interior "dark by location" crops (bright palette, `interior_frac≈0.99`) will still read mostly black — that's honest location structure under `interior_mode=black`, a labeler judgment, not something recoloring should or will change.

## 2. Make the 76 the default for location label-crop renders
Find where the location label-crop render path (the v6 selection / `enrich_select` render step) picks palettes — it drew from the 777 `pool_colormaps.json`. Change the default palette source to the **76** (`score3_colormaps.json`) so future location batches render from the 76, not the pool.

## 3. Verify the training-cache render source
Check `build_plan.py`'s per-location render cache (the 42-render/location augmentation). If it draws palettes from the 777 pool or anything broader than the 76, switch it to the 76 for consistency. If it already uses the 76 (or an intentional set), report and leave it.

## Operational
~640 label-spec renders — estimate runtime up front and **background** it; ≤3 renderer processes at low priority (same discipline as the batch build).

## Report
- Count recolored (expect 640/640), all crops 1280×720, `image_id`s unchanged, labels still null.
- Confirm the label-crop palette default is now the 76, and `build_plan.py`'s status (already-76 vs switched).
