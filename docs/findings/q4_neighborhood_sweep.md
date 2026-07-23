# q4 exemplar-neighborhood sweep — is the "corner" a reachable region or an isolated spike?

Measurement pass (no descent / config / `data/` changes). Follow-up to
[`q4_axis_discovery.md`](q4_axis_discovery.md), which found the exemplar's look sits
alone (1 of 1087) at high busy-ness × high composed-negative-space in the *labeled
corpus*. This pass asks whether that corner is a **reachable region** under small
framing changes around the exemplar, or a fragile isolated spike — and surfaces the
first q4 label candidates.

**Exemplar** — center-view Julia, `c=(0.26103, −0.48932)`, origin-centered
(cx=cy=0), fw=0.75, `twilight_shifted`. See `out/q4_sweep/exemplar_large.png`.

## Method

- **Sweep** — fixed `c`, center-view family. zoom = fw log-spaced ±2.5 octaves
  around 0.75 (0.133 → 4.243, 11 steps) × pan = (cx,cy) on a 5×5 grid, radius up to
  0.5·fw at each zoom (deliberately breaks the origin z→−z symmetry). **275 framings.**
- **Substrate** — `render-one --dump-field --dump-field-source f64` at 768×432 ss1
  per framing, escape-time field, NaN interior, colormap-invariant. Field bins purged
  per-unit. Full grid measured in **31s** (0.07s/dump); colored judge renders
  (1024×576 ss2, `twilight_shifted`) only for the contact-sheet framings.
- **Detail metric** — speckle-vs-ornate is a **scale** distinction, not magnitude, so
  the normalized [0,1] escape field (interior→deepest) is split into `fine`
  (pixel-scale high-freq residual) and `struct_e` (5×5 std of the 3px–11px band =
  mid-scale ornate structure). Bands calibrated on exemplar/deep/void references:
  `flat` = struct_e<0.030, `mid_detail` = [0.030,0.180), `busy`(speckle) =
  fine>0.30 ∧ struct_e<0.05. Quiet-region classes: `interior_frac` (NaN),
  `deep_frac` (finite, normalized escape ≥0.80 = slow-escape "almost-negative"),
  fast-exterior (rest). Tool: `tools/studies/q4_neighborhood_sweep.py`
  (`measure`/`analyze`/`sheets`); data `out/q4_sweep/metrics.jsonl` + `bins.json`.

## Headline: the corner is a COHERENT, REACHABLE REGION — but one-sided

**Not an isolated spike.** 172 of 275 framings sit within ~2 normalized units of the
exemplar in (interior_frac × deep_frac × mid_detail_frac) space
(`out/q4_sweep/fragility.png`, left). The exemplar is at the *top of a dense
continuous band*, not alone — a descent could trivially hunt this. Persistence is
**strongly asymmetric in zoom**:

| direction | behavior |
|---|---|
| zoom **IN** (fw 0.53 → 0.13) | look **intensifies** — mid_detail_frac rises to ~0.60–0.68 (self-similar spiral structure; the exemplar's fw=0.75 is not even the richest rung) |
| mid zoom (fw 0.53 → 1.06) | holds; mid_detail 0.47–0.61 |
| zoom **OUT** (fw ≥ 1.5) | collapses — mid_detail 0.42 → 0.05, flat_frac → 1.0. The whole Julia set shrinks to one blob in a growing void |

**Pan is robust where the structure is dense.** Within the exemplar-like fw band
(≤1.5), the look survives panning to the **corner** (0.5·fw offset) at every zoom —
max pan-ring within threshold = 2 (full grid) for fw 0.13–1.5. Only past fw ≥ 2.1
does panning fall off the set into the void (pan-reach → −1). So: **pan-robust at
deep/mid zoom, pan-fragile only once zoomed out.**

**Reach:** exemplar-like fw ∈ **[0.133, 1.5]** (lower bound = sweep floor; likely
extends deeper), full pan grid throughout. A descent seeded near this `c` would find
the look easily and could push *deeper* for richer variants.

## Variant B is NOT reachable here (honest negative)

Bin B was defined as "less interior, **high deep_frac**, textured" — quiet regions
deep/dark yet not flat. **`deep_frac` caps at 0.085 across all 275 framings**
(exemplar 0.052); there is no framing where slow-escape basins occupy a large area.
This Julia's escape-time histogram simply doesn't produce big "almost-negative"
regions — the slow-escape fraction is a near-constant ~5% texture, not a steerable
axis. The `sheet_B.png` best-first picks therefore **overlap bin A 5/9** and differ
only marginally (slightly less interior, the fw 0.133 deepest rung is the closest
B-flavor: denser texture, less composed negative space). **Literal variant B does not
exist as a distinct look around this exemplar** — it is the same visual family at
higher density.

## Speckle failure mode also unreachable (finding)

`busy_frac` (speckle) caps at **0.0013** across the whole sweep. The pan/zoom family
around this `c` cannot produce the saturated-high-freq failure — the structure stays
ornate and coherent at every zoom, including the deepest. The *only* reachable failure
direction is **flatness** (`flat_frac` → 1.0), from zooming out or panning into the
surrounding void. `sheet_busy.png` is consequently a near-duplicate of bin A and is
labeled as such. (True speckle likely requires **c-perturbation** — the separate
follow-up, not this pass.)

## Bins (label candidates) → `out/q4_sweep/`

- **`sheet_A.png` — BIN A (exemplar-like), 9 framings, best-first.** Composed black
  lakes + distributed ornate mid-detail, balanced negative space. All genuine, all
  strong. Defining ranges: **fw 0.375–0.53**, interior_frac 0.24–0.26,
  mid_detail_frac 0.73–0.77, flat_frac 0.19–0.24, busy≈0, distributed_interior≈1.0.
  (Best-first favors slightly deeper than the exemplar's 0.75 — better composition.)
  **These are the primary q4 label candidates.**
- **`sheet_B.png` — BIN B (variant target), 9 framings.** Best-available slow-escape-
  textured/low-interior framings; **overlaps A 5/9**. fw 0.133–0.375. Ship for
  labeling but note above: not a distinct look.
- **`sheet_flat.png` — FAILURE (flat/boring), 9 framings.** All fw 4.243, corner
  pans; huge void + a detail speck on one edge. Clean illustration of the one real
  failure direction.
- **`sheet_busy.png` — FAILURE (speckle) — UNREACHABLE.** Labeled; ≈bin A.
- **`exemplar_large.png`** — reference.
- **`fragility.png`** — scatter (coherent band) + persistence-vs-zoom.

## Verdict

The q4 corner is a **reachable, coherent region** a descent can hunt — asymmetric
(deepen to intensify, zoom out to lose it), pan-robust at depth. But of the three
directions the prompt hoped to steer, **only detail-density is steerable**:
`deep_frac` (variant B) and `busy_frac` (speckle) are both effectively **constant**
around this exemplar. To reach genuinely *different* q4 looks — high slow-escape
negative space, or controlled busy-ness — the lever is **c-perturbation**, not
framing. Bin A is a solid batch of first q4 label candidates; bin B is a marginal
density variant of the same family.
