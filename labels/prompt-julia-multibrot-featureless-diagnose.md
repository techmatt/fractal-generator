# Investigate 3 featureless julia_multibrot frames in the v6 batch (diagnose, read-only)

## Symptom
Three crops in `2026-07-05_gather_v6` render as smooth, near-featureless **d-fold star/cross gradients with essentially no fractal detail** (pure exterior escape-lobe smoothness, no boundary). All three are **julia_multibrot** and all are **trajectory frames** (`_fN`) — the shared signature suggests something specific to the julia_multibrot trajectory-frame render path, not a location-agnostic issue.

- `r_jmb4_20260705_030514_000007_f6`
- `r_jmb4_20260705_055707_000008_f6`
- `r_jmb3_20260705_052458_000016_f18`

**Diagnose only — no fixes.** A handful of renders is expected and fine (can't see a render bug without rendering).

## 1. Pull provenance + the render block (likely the smoking gun)
For each id, from the batch `images.jsonl`, dump the full **render block** and provenance: `family` / render_family / degree, julia `c`, z-viewport (`julia_z_cx/cy/fw` or the frame's `cx/cy/fw`), `fw`, `auto_maxiter` used, `decoded_class`, `k3`, `guard_verdict`, `selection_role`, `descend_mode`, `parent_oid`, frame index, lineage.

Look immediately for the obvious failure: does the render block record a **julia render** (julia flag set, `c` present, correct degree jmb3→3 / jmb4→4), or did it render as a **plain multibrot/mandelbrot with no `c`**? A missing `c` or wrong family/degree here is the whole story.

Then find the source frame in the gather `walks.jsonl` (the trajectory frame this was re-rendered from) and dump its raw coords/occupancy + the walk's `c`.

## 2. Guard verdict (the mundane check)
What did the logged guard verdict say for these? Compute `field_std` / `interior_frac` at the recorded params and compare. If they're genuinely `flat` (field_std < 6), the field really is featureless — the guard already flags them and they're only in the pool because gather ran guard-off (`random_eval` samples uniform trajectory frames by design). That would make this **not a renderer bug**. Distinguish this from §3.

## 3. Reproduce vs reconstruct (the discriminator)
For each frame:
- **(a) Reproduce:** render at the **recorded** params exactly. Confirm it reproduces the featureless crop.
- **(b) Reconstruct intended:** independently build the intended julia params from the walk frame — `family = julia` at the **correct degree**, `c` = the walk's `c`, z-viewport = the frame's own z-plane `cx/cy/fw`, and the correct `auto_maxiter(fw)` for that width — and render that.
- **(c) Compare.** If (b) shows real julia structure and (a) is featureless → the re-render path is threading wrong params; **report exactly which param diverges** (c dropped? degree wrong? viewport wrong? maxiter mis-scaled for high degree?). If both are featureless → the frame is a genuinely boring shallow/exterior sample, not a renderer bug.

## 4. Read the re-render code
Read the code that re-renders `random_eval` trajectory frames in the selection driver — how it threads family / degree / julia `c` / z-viewport for julia_multibrot frames. Is the walk's `c` correctly joined to each frame, or can it be lost (frame stores z-coords only, `c` lives at walk level)? State whether the suspected path is **julia_multibrot-specific** or would equally hit plain-julia / parameter-plane frames.

## 5. One context render
For one of the three, render the parent outcome (or a late high-`k3` frame from the same walk) to confirm the **walk itself found detail** — isolating "these specific frames are wrong/boring" from "the whole walk is degenerate."

## Report
Per frame: recorded params, render-block family/c/degree, guard verdict + field stats, the reproduce-vs-reconstruct verdict (and the diverging param if it's a bug), and whether it's julia_multibrot-specific. If it's the mundane case, say so plainly. No fixes.
