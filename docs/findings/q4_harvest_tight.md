# q4 stage-1 harvest (tight) — first real q4 candidates

The delivery: the harvest-ready goodness field **G** (refit T2 model) run over every
minibrot, masked position-maxima, gated at a **label-derived** high-precision cutoff,
rendered vivid. Palette-free **locations** — Matt's eyeball is the test. No new labels,
no aimed batch.

## What ran

1. **Field over 33 corpus minibrots** — dense position×scale G, masked to v2 pre-filter
   survivors (G extrapolates OOD on dead interior; maxima only within the manifold).
2. **Position-maxima** — local maxima of the masked G per minibrot, collapsed to
   distinct framings by elliptical center-separation NMS (a peak within 1 window-width
   of a higher one is the same composition; this also merges the same spot seen across
   adjacent scales), then the top **4 per minibrot** by G.
3. **Gate from labels, tight** — cutoff on the deployment model's own G scale (so it
   transfers directly to the harvested windows), set at the high-precision operating
   point on the 340 labeled windows. **Not** p=0.5 (p's 0.5 bucket is only ~14% accept).
4. **Render** survivors at 1920×1080 ss2 in one vivid palette (UF `default`).

## Label-derived cutoff

| gate | G cutoff | labeled precision | recall | n labeled above | held-out prec @ same recall |
|---|---|---|---|---|---|
| **tight (used)** | **1.390** | **0.85** | 0.53 | 55 | 0.76 |
| loose | 1.181 | 0.79 | 0.55 | 62 | — |

The tight cutoff hits the ~0.85 labeled-precision target. In-sample precision (0.85)
sits above the held-out estimate at the same recall (0.76) — the expected optimism of
scoring the training labels; the refit already showed held-out **ranking** holds
(LOMO AUC 0.86), which is what auto-framing relies on.

## Candidates

**116 rendered** across **29 minibrots** (4 distinct framings each). 4 of the 33
minibrots contributed nothing above the tight cutoff (their best framing scored
< 1.39) — dropped, not hidden. Rendered G range **1.62 … 4.62**.

> **Not silently capped:** 116 is the per-minibrot **cap (4×29)**, not the full
> above-cutoff set. The ornate ring is a *continuum* of high-G windows (~2.1k clear
> the cutoff corpus-wide), so this is a diversity-curated "best-few-per-minibrot"
> first look, not an exhaustive harvest. Raise `PER_MB_CAP` for more depth per
> location; the gate, not the cap, guarantees precision.

Per-minibrot: uniform 4 across all 29 covered minibrots (every one has ≥4 framings
clearing the cutoff).

## Contact sheets

`out/q4_stage1/harvest_tight/sheet_{01..05}.png` — ordered by G, tight-gate label in
the header. `candidates.json` carries each survivor's absolute geometry
(`cx_win`/`cy_win`/`fw_win`/maxiter) + minibrot + scale + G — these are palette-free
**locations** (emission/color wiring is a later era).

## Eyeball notes (for the verdict)

- **Top page (G 4.0–4.6):** strong — composed spirals, seahorse valleys, dense
  filigree, all auto-framed, no dead corners, no pure speckle. Composition test passes.
- **Tail (near the 1.39 cutoff):** mostly holds (dendrites/spirals/filigree). The
  weakest tiles are the `mb28_p61` grey-studded ones — a textural
  minibrot-in-frame the model over-likes; these are the honest borderline cases you'd
  expect right at the cutoff. If they read as misses, nudge the cutoff up (fewer, purer).

Regenerate: `uv run python -m tools.studies.q4_harvest_tight all`.
