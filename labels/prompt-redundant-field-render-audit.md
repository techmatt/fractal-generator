# Diagnose redundant field re-renders (colormap-invariant re-rendering) — read-only

## Principle
The smooth field is **invariant** to every colormap-tail param (palette, gamma, phase, cycles, reverse) — those apply to field values, not the iteration. So any loop that holds **location × resolution × render-mode** fixed and varies only colormap params should dump the field **once** and apply all colormaps to that cached array (the field⊗colormap seam already exists, validated ≤1 LSB). Re-rendering the field per candidate is pure waste. Find every place this happens.

**Diagnose only — no edits.** Determine behavior from reading the code; only instrument (and pay a render) if a path is genuinely ambiguous. This is the map; fixes are the follow-up.

## 1. Confirm the beam (known hot spot)
Read `run_location`'s candidate-evaluation loop (`tools/queries/sample_location.py`). For its N candidates on one held-constant location at one working resolution: does it dump the field once and reuse it across candidates, or render the field per candidate? The timing (397 s / 324 ≈ 1.2 s/candidate ≈ one field render) implies per-candidate — confirm from the code and report the field-render : candidate ratio.

## 2. Audit ALL field-render call sites
Trace every place that renders or dumps a field (`render-one` invocations, `render_candidate`, the field-dump path). For each, decide whether it sits inside a loop varying only colormap params over a fixed (location, resolution, mode) — i.e., whether the field is invariant across that loop. Check these explicitly:
- **beam** (§1);
- **bootstrap strata re-render** (`build_bootstrap.py`) — the ~8 label-spec picks per location: one field dump reused across 8 colormaps, or 8 field dumps?
- **preference query generation** (`query_sampler` / `assemble_queries`) — 6 candidates on one held-constant location;
- any **per-location coloring / emission** sampler.

Also answer the fix-enabling question: does **`render_candidate` accept a precomputed field array**, or does it render the field internally each call? (This determines whether the fix is just hoisting the dump out of the loop, or a small signature change.)

## 3. Quantify + fix shape (do not implement)
Per redundant site: current behavior (dump-once vs per-candidate), redundancy factor, the fix shape (dump once per location×resolution×mode → apply colormaps to the cached array), and the expected speedup. For the beam specifically, give the projected overnight-run time after the fix vs the current ~8 h.

## Report
A table of field-render sites: `location/res/mode-varying?` · `current behavior` · `redundant?` · `redundancy factor` · `fix shape` · `speedup`. Plus the `render_candidate` field-input answer. No edits.
