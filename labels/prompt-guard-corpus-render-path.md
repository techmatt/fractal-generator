# Guard the location-corpus render path against the off-recipe coloring tail

## Why
Two coloring recipes exist and produce **different images** (measured mean Δ 16.2, max 209 — not parity):
- `render-one --palette --colormaps <file>` — native Rust colorer. **The canonical path for location-corpus label crops** (what gather_select uses; fast, in-process).
- `render-one --dump-field` + `colormap.render_candidate` — Python pct-stretch→LUT tail. For **arbitrary-param** coloring (gamma/phase/cycles/reverse).

A location-corpus crop must be a pure function of its render block through **`render-one --palette`** (the "crops are rebuildable" contract). The dump-field tail is ~5–10× slower (59 MB field dump + GIL-serialized numpy) **and** off-recipe for corpus crops — it breaks reproducibility and cross-batch coloring consistency. It nearly shipped in the gather_v6 recolor. Add guards so a corpus batch can't be built with the wrong path again.

**Scope: location-corpus render path only** (gather_select / the recolor / future location batches). **Do NOT touch the wallpaper-bootstrap / preference path** — it *correctly* uses `render_candidate` for arbitrary-param coloring; that's its own canonical recipe (reproduced via render_candidate + stored params), not a bug.

## 1. Read first
Confirm the canonical corpus crop render (`render-one --palette --colormaps`), find every site that renders location-corpus label crops, and note whether a shared render entry point already exists or each site hand-rolls the call.

## 2. Guard A — make the wrong path unreachable for corpus crops
Route all location-corpus label-crop rendering through **one canonical helper** (e.g. `render_label_crop(render_block, palette_source) -> jpg`) that uses `render-one --palette` internally. Corpus code calls only this; no raw `dump-field`/`render_candidate` for corpus crops. Consolidate any hand-rolled corpus render onto it.

## 3. Guard B — reproducibility check (the durable backstop)
Add a corpus-batch reproducibility check: sample K crops from a batch, **rebuild each from its render block via `render-one --palette`**, and assert pixel match within JPEG-quantization noise. Anchor the threshold on the measured gap — legit rebuilds are ~0–3 mean |Δ| (same deterministic recipe), the off-recipe path was 16.2, so a threshold around ~5 separates cleanly. This is the enforceable form of "crops are rebuildable" and fires regardless of how a bad batch was produced. Wire it two ways:
- an assertion at the end of location-batch emission (auto-verify each new/modified batch on a K-sample), and
- a standalone check runnable on any existing batch (point it at the corrected gather_v6 batch to confirm it now passes).

Keep K small — `render-one --palette` is ~2 s/crop, so a handful is cheap.

## 4. Optional, if cheap
Stamp the render recipe used (path + palette source) into `batch.json` at emission, and have Guard B confirm it's the canonical one — lightweight self-identifying provenance.

## Report
Sites consolidated onto the canonical helper; Guard B's K-sample result on the corrected gather_v6 batch (should pass now that CC switched it to `render-one --palette`); and confirm the bootstrap/preference path was left untouched.
