# Profile one beam — attribute the ~1.2 s/candidate (measure only)

## Goal
The redundancy audit confirmed there's no wasted field render — the ~1.2 s/candidate is recolor + GPU score of genuinely distinct colorings — but it's never been *attributed*. Profile **one** location's beam and break per-candidate wall time into stages, so we know the real cost split (and whether there's hidden per-candidate overhead). **Measure only — no optimization, no change to beam behavior.**

## Measure
Run one **representative** location's beam (mid-cost — not the cheapest shallow julia, not the deepest multibrot) and attribute wall time across:
- **field dump** (`ensure_field`) — once/location; confirm it's amortized (1 : N).
- **stretch prep** (`stretch_field`) — once/location.
- **recolor / candidate** (`render_candidate` over the ss2 field).
- **score-prep / candidate** (the PIL resize to 384×224 / deploy transform).
- **GPU forward / candidate** (MobileNet).
- **residual** — anything not in the above (per-candidate overhead).

Report total wall, per-stage totals, per-candidate averages, the **recolor : score** ratio, and the candidate count (~348/location expected).

## Honest GPU timing (or the profile lies)
CUDA is async — wrap the forward in `torch.cuda.synchronize()` (or CUDA events) so GPU time isn't misattributed to the next Python call. Use synchronized timing for the GPU stage; `cProfile`/`perf_counter` is fine for the CPU stages (recolor, resize). Don't report a bare `cProfile` number for the forward.

## Watch for (report if present, don't fix)
- GPU forward run **one candidate at a time** (unbatched) — if score dominates, batching a generation into one forward is the future lever.
- Per-candidate sync stall (`.item()`/`.cpu()`), redundant host↔device copy, or model/inputs re-moved to GPU each call.
- Recolor operating on the **full ss2 field then downsampling** — i.e. higher res than the score needs.

## Output
A per-stage wall-time breakdown for the one location, the recolor-vs-score split, and any overhead smoking gun. This tells us which lever (if any) is worth pulling later at emission scale — no edits, no optimization now. One location only (cheap, ~one beam).
