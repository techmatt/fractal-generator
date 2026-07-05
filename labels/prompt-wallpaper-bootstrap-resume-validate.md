# Resume wallpaper bootstrap — validate remaining render paths, prep the fresh full run

## Context
The driver is built (`tools/wallpaper/build_bootstrap.py` + the `retain_all` path in `tools/queries/sample_location.py`); mandelbrot end-to-end and the julia render path are validated. **Not yet validated:** phoenix + multibrot render paths, and any end-to-end run beyond mandelbrot. The palette pool is now clean (661, luma floor 70) — this run must draw from it.

Do the remaining validation now. **Do NOT launch the full run in this pass** — the v6 recolor is currently rendering; leave the fleet to it. Print the launch command for Matt to trigger separately.

## 0. Confirm the clean pool
The driver reads `pool_colormaps.json` / `palette_features.json` at runtime — assert it now sees **661** palettes (post-luma-floor) with features pruned 1:1. No renders.

## 1. Validate the remaining render paths (cheap, direct)
Via the driver's render function (`render-one --dump-field` 1280×720 ss4 + `render_candidate` lanczos3 → q90 JPG, `interior_mode=black`), render one crop each for: **phoenix, multibrot3, multibrot4, multibrot5**, and one **julia:multibrot (deg ≥3)**. Source a sane location per family from the gather ledgers — a `decoded_class ≥ 2` **outcome** each, not a center-descent shallow frame. Confirm each yields real structure (not black, not featureless — sane `field_std`, low `black_frac`). Write to a **scratch dir**, not the batch.

If any family's field breaks (phoenix two-state kernel, multibrot degree threading), **diagnose first** — read the render function's per-family handling — before proposing a fix.

## 2. One end-to-end smoke beyond mandelbrot
Run the full driver path — retain-all beam → strata-sample ~8 → re-render — on **one non-mandelbrot location** (phoenix or a multibrot), to a scratch dir. Confirms the beam/strata/emit integration is family-generic past the mandelbrot smoke. **One beam only** (keep it cheap).

## Fleet
The v6 recolor is rendering — keep this validation to **≤2 renderer processes** so it doesn't contend. It's short.

## 3. Report + prep (do NOT launch)
Report per family: render-path OK, field stats, and the smoke result. Then **print but do NOT execute** the full-run launch command:
- **wipe** `data/wallpaper_corpus/batches/2026-07-05_wallpaper_bootstrap_v1/` (clears the dirty pre-floor-pool mandelbrot smoke crops so the whole batch is from the clean 661),
- then run the driver in full mode (~63 locations / ~504 crops), backgrounded, ≤3 procs low priority.

Matt runs it once the recolor frees the fleet.
