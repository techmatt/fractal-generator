# v7 per-partition `t_good` derivation

**Report-only.** This pass re-derives the per-partition q3 decode threshold table for the
v7 classifier and reports the evidence. **Nothing in `production_seeder.py` was edited and
nothing was flipped.** v7 is still not the deployed scorer; this is the threshold table it
*would* ship with.

Reproduce: `uv run python tools/v7/derive_t_good.py` (CPU-only, ~2 s). Machine-readable
dump: `data/v7/t_good_derivation.json`.

Decode contract (unchanged from v6): a location is q3 iff `p_good >= t_good AND
p_notbad >= 0.5`. All signal is in `p_good`; `p_notbad >= 0.5` holds for 41% of eval
locations and gates out only 4 of 122 human-q3 locations regardless of `t`. Selection
objective is **F2** (recall-weighted β=2) per the prompt — this table feeds a prospecting
loop whose downstream population is roughly half human-good, so discarding a good location
costs more than admitting a bad one.

## Proposed table (copy-pasteable)

```python
T_GOOD_BASELINE = 0.50
T_GOOD_OVERRIDES = {
    "mandelbrot":       0.14,   # v7 F2 sweep, n=942 pos=29
    "julia:mandelbrot": 0.22,   # v7 F2 sweep, n=178 pos=25
    "julia:multibrot3": 0.25,   # v7 F2 sweep, census-144 slice, n=54 pos=21
    "julia:multibrot4": 0.17,   # v7 F2 sweep, census-144 slice, n=51 pos=24
    "julia:multibrot5": 0.10,   # v7 F2 sweep, census-144 slice, n=39 pos=22
    # phoenix: DROPPED — v6 carried 0.18; undecidable under v7 (0 eval positives) -> baseline.
    # native multibrot3/4/5: uncalibrated, no eval in either direction -> baseline.
}
```

Every value is **lower** than its v6 counterpart. This is the whole reason the table had
to be re-derived: v7's `p_good` distribution is markedly more compressed than v6's, so the
v6 numbers (0.24 / 0.30) sit far up v7's tail and silently starve recall. The v6 table is
not "slightly stale" under v7 — it is calibrated to a different score scale.

## Per-partition evidence

Derived at the v7 F2-argmax `t` on the full labeled slice; out-of-fold (OOF) is
leave-one-out (t re-selected on the other n−1 rows, scored on the held row, confusion
aggregated). "disc q3" = human-q3 locations discarded at that threshold. v6 columns score
the **same v7 slice** at the old threshold, so the delta is apples-to-apples on scores,
only the cut moves.

| partition | n | pos | **t\*** | F2 in | F2 oof | gap | P@t\* | R@t\* | admit | disc q3 | v6 t | v6 F2 | v6 R | v6 disc q3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mandelbrot | 942 | 29 | **0.14** | 0.569 | 0.483 | +0.086 | 0.267 | 0.793 | 86 | 6 | 0.24 | 0.471 | 0.552 | 13 |
| julia:mandelbrot | 178 | 25 | **0.22** | 0.755 | 0.693 | +0.062 | 0.538 | 0.840 | 39 | 4 | 0.24 | 0.704 | 0.760 | 6 |
| julia:multibrot3 | 54 | 21 | **0.25** | 0.868 | 0.833 | +0.034 | 0.568 | 1.000 | 37 | 0 | 0.30 | 0.739 | 0.810 | 4 |
| julia:multibrot4 | 51 | 24 | **0.17** | 0.839 | 0.809 | +0.031 | 0.561 | 0.958 | 41 | 1 | 0.30 | 0.714 | 0.750 | 6 |
| julia:multibrot5 | 39 | 22 | **0.10** | 0.894 | 0.861 | +0.034 | 0.629 | 1.000 | 35 | 0 | 0.30 | 0.702 | 0.727 | 6 |

Reading it:

- **Every partition improves in-sample F2 over its v6 cut, and recovers real recall.** The
  julia:multibrot degrees go from discarding 4–6 human-q3 each (v6 cut) to discarding 0–1;
  jm3 and jm5 reach R=1.0 with precision still ~0.57–0.63. This is the julia-hook
  population the loop exists to stop throwing away.
- **The optima sit on plateaus, not knife-edges** (F2 within 0.01 of max over):
  mandelbrot t∈[0.13,0.18], julia:mandelbrot [0.13,0.22], jm3 [0.22,0.25], jm4 [0.10,0.17],
  jm5 [0.06,0.10]. The tie-break picks the **highest** t on the plateau (equal F2, fewer
  false admits), which is why the shipped values land at the top of each band. A rounder
  unified julia:multibrot pick (e.g. 0.15–0.20 for all three) stays inside every plateau if
  a single knob is preferred later.
- **mandelbrot's P=0.267 looks alarming but is a base-rate artifact, not a bad cut.** Its
  eval slice is 3% positive (29/942, the frozen v5-era flat mandelbrot eval), nothing like
  the ~50% post-julia-hook population the objective is tuned for. Even so, v7@0.14 strictly
  dominates v6@0.24 on this slice: +0.10 F2, +0.24 recall, half the q3 discarded (6 vs 13).
  Treat mandelbrot's *threshold* as sound and its *precision number* as a property of an
  unrepresentative eval base rate.
- **OOF gaps are small (+0.03 to +0.09)** — the optimism from fitting the cut on the same
  slice is real but bounded; the julia:multibrot gaps (+0.03) are the tightest, the
  low-positive-count mandelbrot gap (+0.086) the loosest.

## Undecidable / uncalibrated partitions

- **phoenix** — 3 eval locations, **0 positive**. Below the 15-positive floor; no
  derivation. It **falls to baseline 0.50** under v7. Note this is a behavior change: v6
  carried `phoenix: 0.18`, but that value was fit on a v6-era "take-the-best" study against
  v6's `p_good` scale and is meaningless under v7. Re-deriving would need a phoenix eval
  slice that does not exist. Consequence: phoenix q3 admission tightens from 0.18→0.50 until
  a phoenix eval is built. Flagged, deliberate.
- **native multibrot3 / multibrot4 / multibrot5** — 3 / 5 / 4 eval locations, **0 positive
  each**. Per the prompt these get **no invented value**: uncalibrated, no eval exists in
  either direction, held at baseline 0.50.

## Routing check

Enumerated every partition string the seeder can emit (`resolve_family` in
`production_seeder.py`: native c-plane `mandelbrot` / `multibrot{3,4,5}`, `--julia`/hook
`julia:mandelbrot` / `julia:multibrot{3,4,5}`, and the 9th family `phoenix`) and confirmed
each resolves deliberately — no partition silently drops to 0.50 on a name mismatch:

| emittable partition | resolution |
|---|---|
| mandelbrot | DERIVED 0.14 |
| julia:mandelbrot | DERIVED 0.22 |
| julia:multibrot3 | DERIVED 0.25 |
| julia:multibrot4 | DERIVED 0.17 |
| julia:multibrot5 | DERIVED 0.10 |
| multibrot3 / multibrot4 / multibrot5 | native, uncalibrated → baseline 0.50 |
| phoenix | undecidable (0 eval pos) → baseline 0.50 |

All 9 covered. (`tools/v6/verify_t_good.py` is the v6 equivalent of this check; the routing
block in `tools/v7/derive_t_good.py` is its v7 adaptation.)

## Gates (all passed)

1. **Coverage.** The julia:multibrot census resolves to **144 locations, 67 positive**
   (jm3 54/21, jm4 51/24, jm5 39/22) — exactly the expected instrument. Labels were
   reconstructed **only** through `label_store.resolve_score` (crops→location, label=max)
   and cross-checked against the manifest label for all 1296 eval locations: 0 mismatches,
   0 unresolved.
2. **Eval hygiene.** All 1296 scored locations are `split=eval` **and** `biased=False` in
   `data/v7/manifest.jsonl` (checked against the manifest, not batch names). The 17
   unbiased frozen-v6 julia:multibrot eval remnants (source `gather_v6`, 1 q3) were
   **excluded** from the julia:multibrot slices per the build's Option A ("report the
   census-144 slice only, never the 161-union"); they are reported here but not used.
3. **Sufficiency.** Derived only where the labeled slice has ≥15 positives (mandelbrot 29,
   julia:mandelbrot 25, jm3 21, jm4 24, jm5 22). phoenix (0) and native multibrot (0) fall
   below and are left at baseline.

## The leak, stated plainly

The julia:multibrot census is simultaneously the v7 **eval set** and the **only**
unbiased-given-descent julia draw that exists (run-1 ledger exhausted; recorded in the v7
build metadata, Amendment 2). So the three julia:multibrot thresholds are **fit on the same
144 locations they are then reported against** — this is fitting, not discovering. The
leave-one-out column exists to make the size of that optimism visible: the OOF F2 sits
0.03–0.09 below the in-sample F2, and the OOF number is the honest estimate of how these
cuts generalize. mandelbrot and julia:mandelbrot are drawn from the frozen v5/v6 eval and
are not part of the census, but they carry the same in-sample-selection caveat on their own
slices. Until an independent unbiased julia:multibrot draw exists, prefer the OOF figures
when reasoning about expected field performance.
