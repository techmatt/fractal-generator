# Mandelbrot discovery t_good — re-derivation with steered labels (F0.5)

The shipped mandelbrot discovery `t_good = 0.14` is the v7 **F2** (recall-weighted) sweep
over the v7 eval slice (`tools/v7/derive_t_good.py`; n=942, pos=29). The steered_run2
blind human read scored **0/16** mandelbrot admissions good — the human uniformly
rejects what the 0.14 bar admits on steered mandelbrot output (see
`docs/findings/steered_run2_keeper_calibration.md` §E). That is direct evidence the bar
**over-admits on this family**, so — unlike the julia families, whose blind slices were tiny
and not similarly one-sided — mandelbrot is re-derived here **precision-weighted (F0.5)** with
the 16 newly-committed steered labels folded in. Same precedent as phoenix 0.18→0.50: a
deliberate admission tightening backed by a labeled read.

## Slices

| slice | n | positives (human/label==3) |
|---|---:|---:|
| v7 eval (mandelbrot) | 942 | 29 |
| steered_run2 blind | 16 | 0 |
| **combined** | **958** | **29** |

## Sweep

| objective | slice | t\* | F |
|---|---|---:|---:|
| F0.5 | eval only | 0.51 | 0.427 |
| F0.5 | combined | **0.51** | 0.427 |
| F2 (shipped objective) | combined | 0.18 | 0.536 |

**New mandelbrot t_good = 0.51** (F0.5, combined), up from 0.14 (F2).

## What the move buys (on the combined slice)

| cut | precision | recall | F0.5 | admit (TP+FP) | discarded q3 (FN) |
|---|---:|---:|---:|---:|---:|
| old t=0.14 (F2) | 0.226 | 0.793 | 0.263 | 102 | 6 |
| new t=0.51 (F0.5) | 0.455 | 0.345 | 0.427 | 22 | 19 |

On the **16 steered mandelbrot tiles specifically** (all human-not-good, so every admission
is a false positive): the old bar admitted **16/16**; the new bar admits
**0/16**.

## Verdict

The blind read makes mandelbrot's over-admission concrete, and F0.5 acts on it: the bar moves
0.14→0.51, cutting the steered-mandelbrot false-positive admissions from 16
to 0 of 16. This is a deliberate, family-specific tightening of the discovery
bar — the julia families keep their existing t_good (their blind slices do not justify a
similar move). Applied to `production_seeder.T_GOOD_OVERRIDES["mandelbrot"]`.
