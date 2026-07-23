# q4 sweep validation — a zero-label heuristic sweep does NOT recover Matt's deep-center picks

Measurement pass (no net / training / production; no config or `data/` changes).
Question: on the frames Matt hand-boxed, does a **label-free** composite window-sweep
recover his boxes in its top-K? Answer decides how much calibration/labeling the
deep-center hand-curation path owes. Tool: `tools/studies/q4_sweep_validation.py`
(`boxes`/`fields`/`sweep`/`overlays`/`diagnose`).

## HEADLINE — uncalibrated recall ≈ 0. The transferred julia q4 composite does NOT transfer to deep Mandelbrot.

| recall @K | K=5 | K=10 | K=20 |
|---|---|---|---|
| center-in-box | **1/7** | 1/7 | 1/7 |
| IoU ≥ 0.30 | **0/7** | 0/7 | 0/7 |

The lone "hit" (mis box0) is a **big-box artifact**, not a match: Matt's 0.23-wide box
is large enough to *contain* the sweep's top busy-window, so a top-ranked window's center
lands inside it — yet that box's own composite score (0.138) is **far below** the top-10
median (0.399). No box is recovered on the strength of the composite ranking its geometry.

**The composite can SEE Matt's features but ranks them mid-pack.** 5 of 7 boxes have a
sweep window overlapping them at IoU ≥ 0.30 *somewhere* in the NMS-ranked list — p58 box3
at **0.73**, box0 0.51, box4 0.40; mis 0.59; p35 0.60 — but at ranks **22–82**, never top-20.
The mechanism finds the right pixels; the score buries them under the busy bands.

## Ground truth — 3 frames, 7 boxes (reconciled from disk)

The spec named `out/deep_centers/annotated/` (absent) and "two frames, n≈6". On disk the
magenta rectangles are burned on **subframe 0** of three sheets: `preview_p58.png` (5 boxes),
`ladder_mis/fw_1e_8.png` (1), `ladder_p35/fw_8p07e_10.png` (1) = **n=7**. Color-keyed →
normalized frame windows via the `sheet.rs` layout (2×2 grid, PAD=6, tile 0 at origin;
p58/mis tiles 16:9, p35 tile 3:2). Centers/fw/maxiter from `pool.jsonl` + the
[deep-center sourcer](deep_center_sourcer.md) ladders.

**Matt's boxes are small.** 6 of 7 are **0.057–0.099 of frame width** — *below* the spec's
≥1/6 (0.167) window floor; only the mis box (0.233w) clears it. The sweep floor was
deliberately relaxed to `scales = {0.10, 0.16, 0.25}` to span his actual boxes; honoring
1/6 would make IoU recovery structurally impossible (a 0.167w window vs a 0.06w box tops out
at IoU ≈ 0.13). Even so, recovery fails — so the negative verdict is not a scale artifact.

## Method (composite transferred VERBATIM, weights NOT fit)

Fields dumped once per frame via `render-one --dump-field` at the pool/ladder geometry,
768-wide (matches the julia q4 `MEAS` pixel-scale so the struct-band thresholds transfer),
**default `beautiful` source** so the deep frames (fw 1.6e-10 / 8e-10) auto-select the
**perturbation** backend — `--dump-field-source f64` would be garbage past ~1e-13.
Orientation guard: dumped field ⇔ rendered subframe 0, no flip (`orient_*.png`). Sweep:
16:9 windows × 3 scales → `score_A` (from `q4_neighborhood_sweep.py`, the exemplar-built
q4 composite: `mid_detail_frac + 0.3·distributed_interior − 0.4·flat_frac − 2·band_pen
− 5·busy_frac`, `band_pen` wanting interior ∈ [0.10,0.35]) → greedy NMS (IoU>0.30) → top-K.
No weight touched Matt's boxes.

## Per-miss diagnosis — the named failing feature

`diagnose` compares each box's window metrics to the top-10 window distribution:

| | interior | mid_detail | flat | busy | distrib_int |
|---|---|---|---|---|---|
| p58 TOP-10 median | **0.000** | 0.785 | 0.215 | 0.006 | **0.000** |
| p58 misses (box0/3/4) | 0.000 | **0.64–0.67** ↓ | **0.33–0.36** ↑ | 0.006 | 0.000 |
| mis box0 | 0.000 | **0.538** ↓ | **0.462** ↑ | 0.003 | 0.000 |
| p35 box0 | 0.000 | 0.582 | 0.418 | 0.010 | 0.000 |

**Named failing feature: `mid_detail_frac` (LOW) + `flat_frac` (HIGH).** Matt's picks are
**composed spiral hubs with calm surround** — a compact detailed spiral eye set against
smooth negative space (the blue/deep-basin expanse). They carry *less* wall-to-wall ornate
fill and *more* calm than the top windows. `score_A` maximizes ornate fill and penalizes
flat, so it top-ranks the densest busy-filigree bands (the "money-shot" decoration ring
around the central island) and ranks Matt's figure-with-breathing-room hubs mid-pack. His
aesthetic and the composite's objective are **inverted at the motif scale** (`overlay_all.png`:
cyan top-10 packs the busy rings; yellow best-overlap-per-box sits right on each magenta pick).

## Root cause — the interior-lake channel is DEAD on deep Mandelbrot fields

The q4 determinant (per [`q4_decisive_pass.md`](q4_decisive_pass.md)) was the
**interior-lake channel** — composed black lakes distinguishing hits from busy-without-lakes.
On these frames `interior_frac` and `distributed_interior` are **≈ 0.000 for every
decoration-scale window** (top picks *and* Matt's boxes): the only true non-escaping interior
is the central minibrot island, which no small window on the surrounding decorations touches.
**Stripped of its lake term, `score_A` collapses to a busy-ness maximizer** — precisely the
axis [`q4_axis_discovery.md`](q4_axis_discovery.md) found *anti*-quality (high occupancy →
0.47× good-rate). The composite was validated on shallow origin-centered **Julia** fields
where large NaN-interior lakes exist and carry the signal; deep Mandelbrot minibrot
*exteriors* have no such interior, so the load-bearing channel simply never fires.

## Verdict + implication for the hand-curation path

A zero-label heuristic sweep **does not** recover Matt's deep-center picks (recall ~0). The
deep-center hand-curation path **does owe real calibration/labeling** — the transferred julia
q4 composite is insufficient for two independent reasons: (1) its lake channel is absent on
deep Mandelbrot fields, and (2) Matt's motif-scale figure-ground preference is orthogonal to,
indeed inverted from, the busy-ness the composite maximizes.

**Actionable (not implemented):** at deep Mandelbrot the "lake"/calm is the smooth
**deep-basin exterior** (high `flat_frac`), not NaN interior. A composite that *rewards*
moderate flat (calm surround) and adds a **concentrated-detail** term (anti-`distributed`) —
i.e. the **opposite sign on `flat_frac`** from the julia composite — would rank isolated
spiral hubs up. That sign flip is exactly what labels would have to teach; a hand-set prior
transferred from the julia work cannot guess it. This is the concrete evidence that deep
hand-curation needs its own small labeled seed, not a borrowed heuristic.

### Artifacts (`out/q4_sweep_val/`)
`overlay_all.png` (primary — 3 frames, Matt vs top-10 vs best-per-box), `overlay_{p58,mis,p35}.png`,
`orient_{p58,p35}.png` (flip guard), `recall.json` (per-box ranks + pooled), `boxes.json`
(ground-truth windows), `fields/*.bin` (dumped smooth fields).
Regenerate: `uv run python -m tools.studies.q4_sweep_validation all`.
