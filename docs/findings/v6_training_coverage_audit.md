# v6 training-coverage audit

**Date:** 2026-07-17 · **Scope:** read-only. Nothing retrained, no manifest/split touched.
**Verdict:** v6 trained on **5261 of 5713** distinct human-labeled locations = **92.1%** of the
labeled corpus. Every location it *didn't* see is accounted for with a concrete reason; there is
no "unknown" bucket. **The premise's ~18% figure was a unit error** (see below).

---

## The unit — settled first, because everything hangs on it

**A v6 manifest row is a LOCATION, not a crop.** This is not inferred; it is stated by the
builder and confirmed in the data:

- `tools/v6/build_manifest.py` reduces "crops → base locations (**label = max over the location's
  crops**)", keyed by `(family, cx, cy, fw, c)`. `tools/v5/build_manifest.py` and
  `tools/v4/assemble.py` do the same. The augmentation fan-out (palettes × scales × shifts) lives
  in `data/v6/cache_manifest.jsonl` / `aug_cache_*`, **not** as manifest rows.
- `data/v6/manifest.jsonl`: 5261 rows, **5261 unique `(cx,cy,fw,c)` identities** — one row = one
  distinct render location. Each row carries a single `label`, `split`, `group_id`, `fractal_type`.

So "5261 rows ≈ 658 locations @8 crops/loc" is wrong by construction: **5261 rows = 5261
locations.** The "639 locations / 9 families" in the v6 description is *only the gather_v6 fold*
(the newest batch), not the whole manifest.

### Where the premise's arithmetic went wrong (all three compounded)

| premise claim | reality |
|---|---|
| rows are crops @≈8/loc → ~658 locations | rows **are** locations → **5261 locations** |
| "87 julia" | 87 gather `julia:mandelbrot` **+ 1000 `julia_ladder_j0` J0 Julia** the premise omitted entirely |
| "~364 across the six mb/jm degree families" | that is the **in-v6** count for those families; **571** are labeled in total (the extra 207 are post-freeze batches) |
| "~3100 mandelbrot" | **3982** labeled mandelbrot locations (3737 in v6) |
| labeled total ~3624 → 18% | labeled total **5713** → **92.1%** |

---

## Coverage table — per family, labeled locations × in-v6 vs not

Location identity = `(family, cx, cy, fw, c)`; label = max over that location's human-labeled
crops. Join to `data/v6/manifest.jsonl` on `(cx,cy,fw)`. **Human labels only** — `decoded_class`
never consulted.

| family | labeled locs | in v6 | not: post-freeze | not: black-gate |
|---|---:|---:|---:|---:|
| mandelbrot        | 3982 | 3737 | 219 | 26 |
| julia (J0)        | 1000 | 1000 | 0 | 0 |
| julia:mandelbrot  |   87 |   87 | 0 | 0 |
| julia:multibrot3  |  163 |   75 | 88 | 0 |
| julia:multibrot4  |  115 |   62 | 53 | 0 |
| julia:multibrot5  |   82 |   38 | 44 | 0 |
| multibrot3        |   73 |   64 | 9 | 0 |
| multibrot4        |   66 |   57 | 9 | 0 |
| multibrot5        |   72 |   68 | 4 | 0 |
| phoenix           |   73 |   73 | 0 | 0 |
| **TOTAL**         | **5713** | **5261** | **426** | **26** |

`in v6` sums to **5261 = the exact manifest row count** — i.e. every manifest row is a genuine
human-labeled location, and the coverage denominator is fully partitioned with no residue.

---

## The 452 not-in-v6, categorized (no "unknown")

**Freeze point:** `data/v6/manifest.jsonl` was built **2026-07-05 18:55**, immediately after the
`2026-07-05_gather_v6` fold. That timestamp is the entire dividing line.

### (1) Post-dates the freeze — 426 locations, 4 batches

Every one of these batches was **labeled after 2026-07-05**, so no build could have included them.
Not a defect — just newer than the frozen manifest.

| batch | labeled | families |
|---|---:|---|
| `2026-07-11_jm3_band_v1`             |  57 | julia:multibrot3 |
| `2026-07-12_jm45_band_v1`            |  68 | julia:multibrot4/5 |
| `2026-07-12_blindspot_v6reject_v1`  | 219 | mandelbrot |
| `2026-07-17_prospect_run1_baserate_v1` | 82 | all six mb/jm degree-3/4/5 |

(3 of these post-freeze locations coincidentally share a `(cx,cy,fw)` with a manifest row, but the
manifest row is a **different family and different c** — spurious coordinate collisions, correctly
counted as *not covered*.)

### (2) Black-gate exclusion — 26 locations, loose0 only

`tools/v4/assemble.py` mirrors the Rust render accept gate: a crop is kept iff
`black_fraction < 0.30`, and a location survives only if ≥1 crop passes. **26
`2026-06-23_flat_generate_loose0_v3` locations had *every* crop at `black_fraction ≥ 0.30`**
(verified: all 26 have all crops in `[0.305, 0.367]`; 0 had any passing crop; and every in-v6
location has ≥1 passing crop). These are too-black to be valid wallpapers — a deliberate quality
gate, not a lost label. loose0 is the **only** pre-freeze batch with any all-crops-gated location.

### Categories explicitly checked and found empty

- **unjoinable:** 0. Every location-quality batch joined (see storage note below).
- **duplicate identity superseded:** `2026-06-25_scale_controlled_2x2` (713 crops) is a re-export
  of the `scale_2x2_labelset` label set — its 600 labeled coords are identical to
  `2026-06-25_scale_2x2_labelset` (all in v6); its 113 extra coords are unlabeled. Adds **0** new
  labeled locations, so it's not a coverage gap.
- **never ingested / unknown:** 0.

---

## Storage heterogeneity (the trap that produced wrong counts before)

Labels do **not** live in one place. Enumerated, not globbed:

- **In-store** (`images.jsonl` `label.score`): loose0, rev4, rev4occ, gather_v6, blindspot, prospect.
- **Sidecar `labels/*.json`, keyed by `image_id`** (batch `scores.json` is empty `{}`): mining,
  scale_2x2, julia_ladder_j0, jm3_band, jm45_band.
- **Legacy key scheme:** `labels/location_labels.json` (loose0) keys as `idx|comp|palette` and does
  **not** join to the batch's `image_id`s — but loose0's labels are fully present in-store, so this
  is moot. Used the store.
- **`image_id` collision:** the `A_<n>_<comp>_<palette>` scheme collides across the two scale
  batches, which is why the coordinate join (not the id join) is authoritative for membership.

## Out of scope — a *different labeling axis*, correctly absent from v6

These `labels/*.json` are **not** location-aesthetic labels (v6's `score∈{1,2,3}` = bad/okay/good
*location*) and legitimately do not enter the location classifier: `palette_scores` (224, palette
quality), `render_mode_pilot` (500) + `render_mode_scale` (1000, strange-coloring-mode quality →
the mining/render-mode head), `wallpaper_bootstrap` (504) + `wallpaper_headbatch_dramatic` (1000) +
`wallpaper_humanq3` (994, wallpaper quality on a 1–4 scale → the wallpaper head). Their absence
from v6 is by design, not a coverage gap.

---

## Bearing on the split verdict

None. The 2026-07-15 disjointness result stands and is orthogonal. This audit found no train/eval
straddle; it only measured how much of the labeled corpus the (perfectly disjoint) manifest omits.
The answer is **7.9%**, fully explained by freeze timing (426) and the black-gate (26).
