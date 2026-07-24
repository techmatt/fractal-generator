# Fair re-render: "30 useless" was a PALETTE artifact — and mb19_p35 IS ladder_p35

Measurement pass (no net / training / production; no `data/` changes). Settles two
things about the 30-minibrot stage-1 sheet (`out/q4_stage1/minibrots_sheet.png`,
rendered in one dark-purple `twilight_shifted` ramp on near-black): (1) the
coordinate identity of mb19_p35 / mb27_p58 against the known deep-center controls,
and (2) whether the "useless" verdict survives a **fair** render in the target
vivid palette. Tools: `tools/studies/q4_field_richness.py` (palette-invariant
stat), `q4_fair_rerender.py` (target-style sheets), `q4_richness_grid.py` +
`q4_fair_montage.py` (ranked eye-sheets).

## 1. Coordinate identity — settled at the data level, no pixels needed

| claim | verdict | evidence |
|---|---|---|
| **mb19_p35 == ladder_p35 / fw_8p07e_10** | **TRUE — byte-identical** | cx/cy/fw strings match character-for-character between `minibrots.json` mb19_p35 and `deep_centers/pool.jsonl` line 1 (p35 nucleus). Δcx=Δcy=0, fw `8.069624e-10` identical, same maxiter 13640. |
| **mb27_p58 == preview_p58** | **FALSE — different nuclei** | Both are period-58 seahorse-valley nuclei but distinct points: Δcx=**+5.007e-3**, Δcy=**−1.673e-2**, fw ratio **1.68e7** (mb27 fw 2.668e-3 is a shallow context frame; preview_p58 fw 1.588e-10 is the deep money-shot). |

So mb19_p35's "known-rich" claim was **correct**; the mb27_p58 ↔ preview_p58 pairing
was **wrong** (same period, same valley, different coordinate).

### Visual confirm (`identity_side_by_side.png`)
mb19_p35 re-rendered at the ladder's exact framing (3:2, tile 1024, ss2, default/
cubehelix/viridis) vs the on-disk `fw_8p07e_10.png`: **same fractal, structural
NCC 0.977, no spatial shift, per-channel mean diff ~2.4/255 (~1%).** The only
differences are (a) the on-disk money-shot used a different **palette arrangement**
per tile and (b) it has Matt's **magenta annotation box** burned into subframe 0
(from `q4_sweep_validation`). The structure is pixel-for-pixel the same location.
**Yes — mb19_p35 is ladder_p35, at the coordinate level and visually.**

## 2. Fair re-render — "30 useless" was overwhelmingly a palette artifact

All 30 + the preview_p58 control re-rendered in the **target vivid style**
(`sheet --builtins "default cubehelix viridis"`, 4×size framing = each row's stored
fw, tile 1024 / 16:9 / ss2, auto backend), 0.9 min total. `sheets/*.png` +
rank-ordered `fair_montage_ranked.png` (default tile) + palette-invariant
`richness_grid.png` (held-constant turbo).

**The `twilight_shifted` purple-on-black ramp was crushing the mid-tone filigree to
invisible noise.** Under blue/orange, essentially every one of the 30 shows real,
substantial decoration. Rich vs barren, judged by eye across both sheets:

- **~25/30 genuinely rich** — dense composed spiral/filigree crowns around the
  central island (the classic seahorse & elephant money-shot look).
- **~5 compositionally weak** — mb28_p61, mb27_p58, mb15_p31, mb06_p15 (a big
  dead-black minibrot wedge eats ~half the frame) and the sparsest, e.g. mb23_p43 /
  mb13_p29 (a thin decorated rim in a large smooth basin).
- **0 intrinsically barren.** The weak ones fail on **framing**, not structure: a
  nucleus-centered frame at fw=4×size can still leave a dominant island or basin.
  This is exactly the [deep_center_sourcer](deep_center_sourcer.md) composition
  rule — deeper on-structure needs an **offset** onto a decoration, not a rejection.

## 3. Field-richness stat — palette-invariant, ranks decoration DENSITY sensibly, but is NOT an aesthetic ranker

`q4_field_richness.py` on the 30 dumped smooth fields (already at the 4×size frame).
Per center: `L = log(smooth)` (NaN interior → median; log because escape structure
is log-periodic), robust range = p99−p1(L); 4-octave DoG band-pass (σ∈{1,2,4,8});
**R_occ** = mean-over-scales fraction with |DoG|/range > 0.01 (decoration area);
**R_energy** = threshold-free mean |DoG|/range. Nothing fit. R_occ vs R_energy
rank-correlation **0.971** (robust to the threshold). Full ranking in
`richness.json`; top/bottom:

```
#0  mb26_p55  Rocc 0.528     ...     #25 mb05_p14  Rocc 0.138
#1  mb25_p47  0.490                  #26 mb21_p38  0.138
#2  mb24_p45  0.456                  #27 mb22_p40  0.134
...                                  #28 mb13_p29  0.134
#12 mb19_p35  0.315  (money shot)    #29 mb23_p43  0.124
```

**Does it rank sensibly?** Partly. On the palette-invariant grid the top (dense
wall-to-wall filigree) and bottom (thin rim in a large smooth basin) are visually
correct — it cleanly separates decoration-**dense** from decoration-**sparse**. But
it is a **busy-ness / area** measure, not an aesthetic one, with two known failures:

1. **It ranks the known-best money-shot mb19_p35 mid-pack (#12).** mb19's large
   **calm surround** (the smooth deep-basin expanse Matt likes) reads as
   low-occupancy and counts *against* it. Composed figure-with-breathing-room
   scores below wall-to-wall busy.
2. **Big dead-black interiors inflate the rank** (mb28_p61 #7, mb27_p58 #8,
   mb15_p31 #10 all ~46–50% interior) because the non-black remainder is dense.

Both are the **same inversion** [`q4_sweep_validation`](q4_sweep_validation.md)
found on Matt's hand-boxes, now reproduced on the full 30: occupancy/busy-ness is
orthogonal to (indeed partly inverted from) the composed-hub-with-calm aesthetic. A
richness stat that tracks Matt's taste needs the **opposite sign on flat/calm** and
a **concentrated-detail** (anti-distributed) term — which only labels can teach.

## Verdict

(a) **mb19_p35 == ladder_p35** — byte-identical coordinates, structurally identical
render (NCC 0.977). The unverified claim was right.
(b) Under fair rendering **~25/30 are genuinely rich, ~5 are compositionally weak
(framing, fixable by offset), 0 are barren.** "30 useless" was a palette artifact of
the `twilight_shifted` ramp.
(c) The field-richness stat **ranks decoration density sensibly** (dense↔sparse,
robust to threshold) but is a **busy-ness measure, not an aesthetic one** — it buries
the composed money-shot mid-pack and is inflated by big interiors. Report-only; do
not gate on it.

### Artifacts (`out/fair_rerender/`)
`sheets/*.png` (31 target-style 3-palette re-renders), `fair_montage_ranked.png`
(default tile, rank-ordered), `richness_grid.png` (held-constant turbo, rank-ordered),
`identity_side_by_side.png` (mb19 vs ladder + 8× diff), `richness.json` (full ranking),
`identity/mb19_p35_3x2.png`. Regenerate: `uv run python -m tools.studies.q4_fair_rerender`
then `q4_field_richness` / `q4_richness_grid` / `q4_fair_montage`.
