# Steered vs baseline frontier descent — pilot A/B

Two fresh-generation arms, same families, separate run-scoped ledgers. **Steered** = classifier best-first frontier (`tools/atlas/steered_frontier.py`, `guided-descend --expand`); **baseline** = the current production walk (`production_seeder.py --run`), byte-untouched.

- steered  : `C:\Code\fractal-generator\data\discovery\steered_pilot\steered`  — active 7.8 min
- baseline : `C:\Code\fractal-generator\data\discovery\steered_pilot\baseline`  — active 8.0 min (**NOMINAL** — see caveats)

## Read this first — interpretation & caveats

- **Timing is not cleanly controlled for the baseline; the primary comparison is the RAW distinct-q3 counts and per-family / depth breakdowns, not the per-hour rate.** `production_seeder --run` checks its wall-clock budget only at batch boundaries, and with `--julia-hook` on, one 24-seed batch spawns dozens of 3-walk julia sub-descents — so a single mandelbrot batch ran ~15 min against a 2-min budget. The baseline was therefore run in segments (mandelbrot + multibrot3 at `--batch 4`; multibrot4/5 topped up at `--batch 1` to bound cost) and its "8.0 min active" is the *target* budget, not measured — true wall-clock was substantially larger. The steered arm's 7.8 min is real batch time (it excludes the uncounted NativeSeeder root-generation warmup, which both arms share). **Treat the per-hour rates as indicative only; the counts are the comparison.**
- **Headline finding — steering reaches SHALLOWER, not deeper.** Neither arm admitted anything at depth > 14. But the steered frontier admitted **13 of 16** q3 at **depth 2–3** — *below* the walk's `depth_min = 4` floor. A frontier can admit a q3 the instant a root's first rungs produce one; the baseline walk only harvests its terminal region at `reached_depth ∈ [4,14]`. So the fixed [4,14] draw truncates good shallow locations *from below*, and steering surfaces them. This is the clearest behavioral difference in the pilot — and the obvious question for the blind human read (do the shallow steered keepers hold up?).
- **Coverage:** steered admitted across **all 8 partitions**; the (truncated) baseline reached 3 with distinct q3. Steered's low coord-dup at admission (11% vs 53%) reflects the priority dup-penalty + density rejection + the pre-reframe dedup short-circuit.
- **Harvest / τ_h transfers to live steering:** 881/936 cheap-gated frames (94.1%) canonically decoded q3, confirming the fidelity-study τ_h derivation holds online. The frontier **concentrated** on `julia:multibrot5` (552 of 936 harvest checks) — a rich region best-first exploited hard; only 18 of the 881 canonical-q3 survived reframe+guard as distinct (the rest were coord-dups of the 16 admitted places, correctly deduped, not re-counted).
- **M-cap is soft at batch granularity:** the `< M` check is at pop time, but a batch can pop several nodes sharing one root, so a root can overshoot 40 (max 66). It bound 8 roots — genuinely binding, just not to an exact ceiling.
- **Coord-dup "over time"** is reported as the run-total rate (the segmented baseline timing makes a clean time-series unreliable); the steered `harvest_log.jsonl` carries per-batch `admitted`/dup if a time-series is wanted.

## Primary — distinct q3 per active hour (coord-deduped)

| arm | distinct q3 | active min | q3 / active hour |
|---|---|---|---|
| steered | 16 | 7.8 | 123.1 |
| baseline | 7 | 8.0 | 52.5 |

### Per-family distinct q3

| family | steered | baseline |
|---|---|---|
| julia:mandelbrot | 3 | 2 |
| julia:multibrot3 | 1 | 3 |
| julia:multibrot4 | 1 | 0 |
| julia:multibrot5 | 2 | 0 |
| mandelbrot | 2 | 2 |
| multibrot3 | 2 | 0 |
| multibrot4 | 1 | 0 |
| multibrot5 | 4 | 0 |

## Coord-dup rate (dup q3 / all decoded-q3)

| arm | q3 dup | decoded q3 total | dup rate |
|---|---|---|---|
| steered | 2 | 18 | 11.11% |
| baseline | 8 | 15 | 53.33% |

## Depth distribution of admitted q3 (does steering exceed the fixed [4,14] draw?)

| depth | steered | baseline |
|---|---|---|
| 2 | 5 | 0 |
| 3 | 8 | 0 |
| 4 | 2 | 1 |
| 5 | 1 | 0 |
| 6 | 0 | 1 |
| 8 | 0 | 2 |
| 11 | 0 | 1 |
| 13 | 0 | 2 |

Admitted q3 at depth > 14 (beyond the walk's terminal draw): **steered 0**, baseline 0 (baseline caps at 14 by construction).

## Harvest confusion (steered) — cheap-said-harvest x canonical decode

Every candidate whose **cheap p_good >= tau_h** got one canonical 640x360 ss2 render + decode; the confusion is how those canonical decodes landed.

| cheap-said-harvest -> canonical decode | count |
|---|---|
| class 3 | 881 |
| class 2 | 50 |
| class 1 | 5 |
| class None | 0 |
| **total harvest checks** | 936 |

Canonical-q3 confirmations: **881** / 936 harvest checks (**realized tau_h precision** = fraction of cheap-gated frames that canonically decode q3 = 94.1%). Of those, 18 survived reframe+guard as q3 and 16 were admitted distinct.

### Per-family harvest checks -> canonical q3

| family | tau_h | checks | canonical q3 | precision |
|---|---|---|---|---|
| julia:mandelbrot | 0.184 | 13 | 6 | 46.2% |
| julia:multibrot3 | 0.311 | 170 | 159 | 93.5% |
| julia:multibrot4 | 0.207 | 157 | 154 | 98.1% |
| julia:multibrot5 | 0.186 | 552 | 550 | 99.6% |
| mandelbrot | 0.067 | 22 | 3 | 13.6% |
| multibrot3 | 0.199 | 7 | 4 | 57.1% |
| multibrot4 | 0.774 | 1 | 1 | 100.0% |
| multibrot5 | 0.199 | 14 | 4 | 28.6% |

## Per-root expansion histogram + M-cap (steered)

- roots expanded at least once: **127**
- expansions/root: max 66, median 1, mean 4.5
- roots that hit the M=40 cap: **8** (M is BINDING); batch-level cap_hits telemetry = 6


| expansions/root bucket | roots |
|---|---|
| 0-4 | 115 |
| 5-9 | 3 |
| 15-19 | 1 |
| 40-44 | 8 |

## Paired sample manifest (blind-read set — built, NOT labeled)

Even draw of 19 admitted locations (12/arm requested), coords + canonical 640x360 ss2 renders under `out\steered_pilot_manifest/` (`manifest.json` lists arm/id/coords/render). **Not labeled** — this is the blind-read set for a later human pass; arm identity is in the manifest, not the filenames' visible order.
