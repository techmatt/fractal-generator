# Cache the baked LUT per (palette, reverse) — recolor speedup (byte-identical)

## Goal
The beam profile found recolor is ~96% of beam wall, and ~21% of recolor is the OKLab palette→LUT bake (`build_lut` / `_interp_oklab_cyclic`), recomputed nearly per candidate. The baked LUT is a **pure function of (palette, reverse)** — invariant across a lineage's gamma/phase/cycle variants, which apply *after* the LUT — so ~7/8 of those bakes are redundant. Cache the baked LUT so each distinct (palette, reverse) is baked once per process. This is **pure memoization: output must be byte-identical.** Measure the speedup.

## 1. Verify the LUT's true dependencies first (the whole correctness of this change)
Read `build_lut` / the OKLab bake in `colormap.py`. Confirm the baked LUT depends **only** on (palette, reverse) — i.e. gamma, phase, n_cycles, log_premap, transform all apply to the LUT's *output* during recolor, not to the bake itself. If the bake depends on anything beyond (palette, reverse), the cache key **must** include it. Get this exactly right; a wrong key silently corrupts colors.

## 2. Add the cache
Memoize the baked LUT keyed on **(palette identity, reverse)** — palette identity = its pool name/id (or a hash of the stops if names aren't guaranteed unique). A module-level dict is fine: bounded by pool size (~hundreds max), LUTs are tiny, no eviction needed (or a trivial cap). `render_candidate` (and `build_lut`) look up the cache before baking. A redundant concurrent bake under the beam's threads is harmless (same result), so no lock is needed. This is **universal** — it benefits every `render_candidate` loop (beam, bootstrap, query gen), not just the beam. Change nothing else about recolor.

## 3. Byte-parity gate (must pass exactly)
Since this is pure memoization, output must be **identical**. Render a sample of candidates spanning varied palette + gamma + phase + cycles, **with and without** the cache, and assert pixel-identical (mean |Δ| == 0, exact). Any nonzero delta means the cache key is wrong — stop and report, don't ship it.

## 4. Measure
Re-run the one-location beam timing (or a recolor microbench) with the cache on; report recolor ms/candidate and total wall, before vs after. Expectation: ~7/8 of the ~166 ms bake removed (~145 ms/cand), recolor ~773 → ~628 ms/cand, ~18% off beam wall — report the actual numbers.

## Report
The verified cache key (the LUT's true dependencies), the exact byte-parity result, and the measured recolor/wall speedup. No other behavior change.
