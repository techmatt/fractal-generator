# Gap report — library record v0.1 against the 47-location instance

What the schema gets **free** from existing artifacts vs what needs a **new
producer**. Built by `build.py`; instance is `library_records.jsonl` (47 rows,
428 KB).

## Free today — every field below populated with 0 misses across 47 records
| block | source artifact | notes |
|---|---|---|
| `identity` (c_plane/julia) | `curated/manifest.jsonl` `location` | decimal-string cx/cy/fw + c already carried |
| `location_potential` | `fresh_runs/…/outcome_ledger.jsonl` via `source_oid` | k3 / p_good / decoded_class / t_good all present |
| `palette_candidates` (K≈12) | `pools/<cycle>/images.jsonl` provenance | pref rank/score + full recolor recipe already there |
| `color_category` | `data/palettes/palette_categories.json` | **564/564** candidate palettes resolve k8/k12/k16 — the just-committed map covers the whole live pool |
| `mode_candidacy` | `out/mining/deploy_tail/report.json` `per_location` | 47/47 keyed by loc_id; 12/47 locs have a kept mode |
| `descriptors` (CLIP + v6) | `scratchpad/visual_dup/embeddings.npz` | uid == curated_from; stored by-reference |
| `wallpaper_quality` | `curated/manifest.jsonl` | emitted recipe + v3 gate |

Coverage spot-checks from the instance:
- `coord_kind`: 25 julia_c_fixed, 13 z_viewport (phoenix), 9 c_plane.
- `t_good` in force: 20×0.24, 13×0.18 (phoenix), 12×0.30, 2×0.50 — the per-family
  bands land in the record automatically.
- palette-candidate sources: 375 dramatic / 79 extracted / 64 q3 / 46 q2.

## Needs work

### A. Free-but-not-stamped (value known, producer trivial)
1. **Phoenix `identity.c` / `identity.p`** — null in manifest *and* provenance
   because they are the **fixed Ushiki constants** `c=(0.5667,0)`, `p=(-0.5,0)`
   (`src/v4_cache.rs:44-45`). The builder stamps them from that constant. **Action:**
   the record producer must materialize implicit family constants, not copy nulls —
   otherwise every phoenix record looks parameterless and the dedup key is wrong.
   `coord_kind: z_viewport` is the flag that these are viewport coords of one system.

### B. Real gaps — no producer exists (emitted as `null`)
2. **`wallpaper_quality.actual_p_ge3`** (Fork #1). Only the *predicted* pre-render
   gate score exists. A post-render rescore of the shipped 2560×1440 wallpaper is
   *possible* (the v5→v6 rescore convention: re-score the committed frame at reframe
   fidelity, |Δ|≈2e-4) but is **not run or stored**. Producer = a rescore pass over
   the emitted PNGs. Cheap; not yet wired.
3. **`descriptors.colored_clip`** (Fork #2, soft substrate). Present descriptors are
   **palette-blind grayscale** morphology (correct for dedup). A *colored*-image CLIP
   descriptor — needed if selection wants to diversify on delivered appearance, not
   just skeleton — has no producer. Slot reserved. Producer = CLIP over the shipped
   color crops (embed.py already does grayscale; a color variant is a small edit).
4. **`palette_candidates[].mood`** — no `mood` field exists anywhere.
   `palette_features.json` carries structural signals (cycle type, chroma, seam) and
   `palette_categories.json` carries a `special` tag (chromatic/neutral/spectral/
   outlier) — the closest existing categorical, already surfaced in `color_category`.
   A semantic "mood" (warm/cold/electric/earthy…) would need a new labeling or
   derivation pass. Left `null`; `color_category.special` is the usable stand-in.

### C. Design flags for the reviewer (not gaps — decisions to ratify)
5. **Descriptor storage = reference, not inline.** Record stays ~9 KB by pointing at
   `embeddings.npz`. That npz is under `scratchpad/` (calibration set). If the record
   becomes load-bearing, promote embeddings to `data/` (FINDINGS.md already flags
   this) — otherwise the descriptor reference dangles on `rm -r out`/scratch cleanup.
6. **`v6_prelogits` retained but marked UNFIT** for fine grayscale dedup (saturates;
   FINDINGS.md). Kept for the grid/recall lineage; CLIP is the primary morphology axis.
7. **44 vs 47.** Instance is all **47** emitted locations. The "44" is the CLIP-0.97
   conservative merge (3 tightest morphology dups collapse); the merge is a
   *consumer* of `descriptors`, not baked into the record. No dedup gate is applied
   here — the record carries the substrate, the gate stays a downstream policy.

## One-line verdict
The record is **~95% free** from existing artifacts: identity, Stage-1 potential,
palette beam + color category, mode candidacy, and morphology descriptors all
populate with zero misses. The only true new producers are the two Fork-slots
(`actual_p_ge3` post-render rescore, `colored_clip`) and the optional `mood` label —
each small and independently addable. Phoenix constants need stamping, not producing.
