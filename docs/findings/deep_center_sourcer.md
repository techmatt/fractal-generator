# Sourcer: high-precision deep-center finder (hand-curation track)

Built the sourcing component the deep probe proved missing
(`deep_mandelbrot_visual_probe.md`): the render tier is deep-clean to 1e-20, but
the walker is f64-bound (`Frame.center: Complex<f64>`) and structurally can't
localize a center below f64 resolution. This finder produces **valid deep
centers** — points on ∂M whose neighborhoods sustain structure at depth — as
high-precision **decimal-string** centers that flow straight into the proven
perturbation render tier. No discovery/scoring/emission, no classifier: it feeds
**hand-curation**, Matt's eye picks the beautiful ones.

**Deliverable:** `tools/sourcing/deep_center_finder.py` (reusable finder, mpmath
HP Newton — `nucleus` / `misiurewicz` / `scan` subcommands, also importable) +
`tools/sourcing/emit_deep_pool.py` (batch a seed list → `pool.jsonl`). Validation
ladders under `out/deep_centers/`.

## TL;DR

- **Both finders work and recover known centers exactly.** Nucleus Newton on
  z_p(c)=0 recovers the canonical nuclei to full precision (period-2 → −1,
  period-3 west → −1.75487766624669276…, period-4 → −1.94079980652948475…,
  Douady rabbit → −0.12256116687665362+0.74486176661974424i). Misiurewicz Newton
  on z_{k+n}=z_k recovers the exact antenna tip c=−2 (preperiod 2 / period 1) to
  residual **1e-116**.
- **The probe's "known center" is NOT a special point** — the load-bearing
  correction of this task (below). The published Seahorse coordinate cannot be
  *recovered* by any nucleus/Misiurewicz finder because it is not a low-order
  root of either equation. The probe's "Misiurewicz preperiodic" label was an
  unverified assumption.
- **Fresh deep centers render on-structure and hold a band** through the
  perturbation tier, crossing the f64→perturbation switch with **0 glitched
  pixels** — matching the probe's clean regime. **10 fresh valid centers**
  produced (`out/deep_centers/pool.jsonl`): 6 minibrot nuclei (period 4…59) + 4
  self-similar Misiurewicz spirals.

## The load-bearing correction: the probe's Seahorse center is not special

The recovery gate said: *recover the probe's self-similar center (Seahorse) from
an f64 seed; if the finder can't recover a center we already have, stop.* Running
the finder revealed the gate's premise is false.

Evaluating both residuals **at the full published 33-digit coordinate**
(`-0.743643887037158704752191506114774 + 0.131825904205311970493132056385139i`):

| search | range | global min |z_p| or |z_{k+n}−z_k| |
|---|---|---|
| nucleus | period 1…260 | ~1e-2.6 (period 39) |
| Misiurewicz | preperiod 1…70 × period 1…70 | **1e-5.45** (period 39) |

No dip toward zero anywhere. A genuine order-K special point rounded to 33 digits
would show a residual plunging to ~1e-15…1e-33 at its true (k,n); the floor here
is **1e-5.45**, i.e. the coordinate is merely in a period-39-ish *slow* region.
**It is a generic near-∂M point placed to ~1e-20 precision** (enough to render
clean at fw 1e-20) — not an algebraic special point. So "recover THIS center"
is unsatisfiable by construction, and that is itself the finding: the probe
labelled the point Misiurewicz on the strength of its self-similar *look*, but it
is not one of low order.

The gate's **intent** (don't trust the finder blindly) is met instead by the
exact canonical recoveries above plus the render validations below.

## The finder

`tools/sourcing/deep_center_finder.py` — mpmath high precision (correctly
rounded; a Newton solver needs accurate division, unlike `hp.rs`'s
projection-absorbed `RoundingMode::None` orbit arithmetic). Two roots:

- **Nucleus** (period-p component center): Newton on z_p(c)=0 with the dz/dc
  recurrence `d_{n+1}=2 z_n d_n + 1`. Rock-solid — this is the **workhorse**.
- **Misiurewicz** (pre-periodic z_{k+n}=z_k): Newton on that residual.
  Caveat below.

Precision is sized from the target fw (`dps_for_fw`); centers serialize as decimal
strings with enough digits for the deepest fw (`emit_digits_for_fw`). The
`scan` subcommand Newtons a grid of periods / (k,n) from an f64 seed and reports
which converge minimally near it — the way to *identify* an unknown coordinate's
type rather than guess.

### Misiurewicz has spurious periodic roots — filter by minimality

z_{k+n}−z_k = 0 is satisfied by **every periodic parameter** (c=0, c=−1, c=−2,
every nucleus), because at a periodic point z_{k+n}=z_k trivially. Newton from a
loose seed frequently falls into those large basins (observed: seahorse seeds
converging to c=0 / c=−1). `is_minimal_misiurewicz` screens them (checks the
orbit isn't already periodic one step earlier, and the eventual period is
minimal). Harvesting genuine spirals = seed near ∂M, small (k,n), **keep only
minimal roots with residual→0**. This yielded 5 genuine elephant-valley spirals
of preperiod 7–9 / period 4–5.

### Composition rule: a nucleus centered at fw=size is interior black

The sharp practical finding from the render ladders. A nucleus sits in the
minibrot's **interior** (black). Centered on it:

- fw ≈ size, or deeper → the frame is dominated by black interior (dead).
- **fw ≈ 4× size → the money shot**: the whole minibrot frames as a small island
  ringed by dense radial spiral decorations (see `ladder_p35/fw_8p07e_10.png`,
  `preview_p58.png` — gorgeous across all 3 palettes).

So for a **nucleus-centered** frame the compositionally-valid band is roughly
`[~40× size (context) … ~2× size (island fills frame)]`, and the finder now
suggests **fw = 4× size** (was: =size, which rendered mostly black). Going
*deeper* on-structure needs **offsetting** the center onto a decoration —
out of scope here (that's a search; hand-curation offsets by eye). Misiurewicz
points have no interior-black problem: they sit **on** ∂M, so structure fills the
frame at every scale.

## Validation renders (perturbation tier, 1024×576 ss2, default/cubehelix/viridis)

All rungs auto-selected the backend by spacing; **0 glitched pixels** everywhere.

**Nucleus — seahorse-valley minibrot p35** (c=−0.749774832723653428…,
+0.107617243526536783…, size 2.0e-10), `out/deep_centers/ladder_p35/`:

| fw | backend | look |
|---|---|---|
| 8.07e-10 (4× size) | f64 | **money shot** — island + full radial spiral ring |
| 2.02e-10 (= size) | f64 | mostly interior black (the composition finding) |
| 5e-11 | **perturb** | clean, decoration filaments |
| 1e-11 | **perturb** | clean |

**Nucleus — p58** (size 4.0e-11), money shot at fw 1.6e-10 (perturb-adjacent,
spacing 1.55e-13): island framed by dense spirals, on-structure, clean
(`preview_p58.png`). ~6×10⁹ magnification.

**Misiurewicz — elephant spiral M(7,5)** (c=0.32187663879025893206…,
+0.033260752306371290736…), `out/deep_centers/ladder_mis/`: **self-similar band
hold** — the same nested log-spiral tendrils at fw 1e-5 (f64), 1e-8 (f64), and
1e-11 (**perturbation**). Structure is identical in character across the switch,
confirming the class sustains q4-style looks at depth exactly as the probe's
(mislabelled) anchor did. Composition sits at the filament/exterior junction
(half the frame is exterior gradient) — a framing offset for hand-curation, not a
validity issue.

## Produced centers (`out/deep_centers/pool.jsonl`)

10 fresh valid centers, all Newton-converged to genuine roots (residual
1e-76…1e-80), render-ready via each row's `render_cmd`:

- **6 minibrot nuclei**: seahorse p35/p47/p58/p59, elephant p29, north-bulb p4.
  Sizes 8.5e-3 … 4.0e-11 (deepest money-shots into the perturbation tier).
- **4 Misiurewicz spirals**: elephant-valley M(7,5), M(8,4), M(8,5), M(9,5) —
  self-similar, any depth.

## Reuse / next

- Single center: `deep_center_finder.py nucleus --seed RE IM --period P` (or
  `misiurewicz --preperiod K --period N`, or `scan` to identify a coordinate).
- Batch: edit `SEEDS` in `emit_deep_pool.py` and re-run.
- **Deeper nuclei on-structure need an offset step** (center → onto a decoration
  spiral at fw ≪ size). The Misiurewicz path already reaches arbitrary depth
  on-structure and is the better vein for genuinely deep self-similar wallpapers;
  worth better seeding (branch-tip seeds) to harvest more of them.
- Render cost is the gate, not the finder (Newton is milliseconds): deep rungs are
  25–60s at 1024×576 ss2 (no SA/BLA). Background large ladders.
