# Location-library record schema — proposal v0.1 (design-first)

The artifact the next-era pipeline reads/writes: **one record per curated location.**
Proposed here and validated by populating it against the existing 44/47-location
instance — every field is sourced from an artifact that already exists (see
`build.py`; worked instance in `library_records.jsonl`, 47 rows). **No pipeline
wiring, no Phase-1 run.** Gaps (fields with no producer yet) are listed in
`library_gap_report.md` and emitted as explicit `null`.

## Design principle
The classifier already draws a hard line: `render` (version-invariant identity →
crop) is the *only* thing training sees; `provenance` (version-tagged) feeds the
bias loop and **never** enters training (`CORPUS_SCHEMA.md`). This record keeps that
seam and adds two axes the era needs: **selection substrate** (hard-cell tags *and*
soft embeddings, so descriptor-primary vs grid-primary selection both run off one
record — Fork #2) and **quality provenance** (predicted vs actual — Fork #1).

**References, not copies.** Palettes are stored as `{name, source, type}` + a
`color_category` lookup, never as baked stops (stops live in
`data/palettes/*_colormaps.json`). Embeddings are stored as an `{npz, row}`
reference, not 2048 inlined floats — keeps a record ~9 KB instead of ~50 KB and
keeps the vectors in their canonical store.

## Dense vs sparse (Fork #3, resolved by field)
| block | density | guaranteed when |
|---|---|---|
| `identity` | **dense** | always (the dedup + coverage key) |
| `location_potential` | **dense** | always (Stage-1 score exists for every seeded location) |
| `descriptors` | **dense** | always (morphology CLIP is palette-blind, computable from the field alone) |
| `wallpaper_quality` | **dense** | present once a location is emitted (all 47 are) |
| `palette_candidates` | **sparse** | populated only where the beam colored the location (K per emitted loc) |
| `mode_candidacy` | **sparse** | populated only where the mining tail probed the location |

## The record

```
{
  record_version: "0.1",
  location_id:   str,        # curated image_id, e.g. "cycle_001_wfd_000_02" — primary key
  curated_from:  str,        # "<cycle>/<variant>" — join key into pools + embeddings uid

  # ---- DENSE: identity (the dedup + coverage key) --------------------------
  identity: {
    family:        str,       # julia | mandelbrot | multibrot{3,4,5} | julia_multibrot{3,4,5} | phoenix
    fractal_type:  str,       # render-engine family tag
    cx, cy, fw:    str,       # DECIMAL STRINGS (f64 center is meaningless at depth)
    maxiter:       int,
    c:  {re,im} | null,       # dynamical additive const. julia: the fractal itself.
                              #   phoenix: fixed Ushiki (0.5667, 0) — stamped, was implicit-null.
    p:  {re,im} | null,       # phoenix z_{n-1} coeff, fixed Ushiki (-0.5, 0); null elsewhere.
    coord_kind:    str,       # c_plane | julia_c_fixed | z_viewport
                              #   z_viewport ⇒ cx/cy/fw is a viewport of ONE fixed phoenix system
    source_oid:    str,       # discovery-ledger id — lineage + cross-run dedup anchor
  },

  # ---- DENSE: location-potential (Stage-1 quality, pre-color) --------------
  location_potential: {
    scorer_version:        str,          # "v6"
    k3:                    float,        # v6 top-3 mean logit (the descent objective)
    raw_top3:              [float,3],
    decoded_class:         int,          # 1|2|3
    p_good, p_notbad:      float,        # v6 marginal probs
    t_good:                float,        # per-family gate in force at discovery (julia .24, phoenix .18, …)
    reached_depth:         int,
    guard_pass:            bool,
    seeder_decoded_class:  int,          # seeder-time snapshot (may predate ledger rescore)
    seeder_p_good:         float,
    source_ledger:         str,          # path — provenance of the above
  },

  # ---- SPARSE: palette candidates (the K beam; references, not copies) -----
  palette_candidates: [ {
    variant_id:      str,     # "wfd_000_03"
    emitted:         bool,    # is this the shipped variant?
    pref_rank:       int,     # pref-v2/-v3 rank within the beam
    pref_score:      float,
    selection_role:  str,     # machine_q3 | …
    palette_ref:  { name, source, type },       # REFERENCE (stops live in data/palettes/)
    color_category: { k8, k12, k16, special, leaf_pos } | null,   # committed nested ward cut
    mood:            null,    # GAP — no producer (see gap report)
    coloring:     { reverse, log_premap, gamma, phase, n_cycles,
                    transfer, transfer_gamma, interior_color },    # the recolor recipe
  }, … ]                                                           # K≈12 per emitted loc

  # ---- SPARSE: mode-candidacy (mining deploy-tail) -------------------------
  mode_candidacy: [ { mode, kind, p_ge3, p_ge2, E_ord, passed, kept }, … ] | null,

  # ---- DENSE: descriptors (soft selection substrate; by-reference) --------
  descriptors: {
    npz:                str,   # scratchpad/visual_dup/embeddings.npz
    uid:                str,   # row selector == curated_from
    clip_vitb16_row:    int,   clip_dim: 768,   # palette-blind grayscale CLIP — PRIMARY
    v6_prelogits_row:   int,   v6_dim: 1280,    # in-house; UNFIT for fine grayscale dedup
    colored_clip:       null,  # GAP — colored-image descriptor slot, no producer
  },

  # ---- DENSE: wallpaper-quality (post-selection render outcome) -----------
  wallpaper_quality: {
    emitted_variant:   str,
    emitted_palette:   str,
    emitted_coloring:  { … },              # shipped recipe
    render_mode:       str,                # smooth | tia | stripe | …
    render_spec:       { … },  wallpaper_canon: { … },
    gate:              { head, p_ge3, fitness, threshold },
    predicted_p_ge3:   float,              # FORK #1: gate score is PREDICTED (pre-render, smooth crop)
    actual_p_ge3:      null,               # GAP — post-render rescore of the 2560×1440 wallpaper
  },
}
```

## The three forks — surfaced, not silently resolved

1. **Predicted vs actual quality (q3-vs-q4).** `wallpaper_quality.gate.p_ge3` /
   `predicted_p_ge3` is the head scoring the *pre-render smooth crop* — a prediction.
   The record reserves `actual_p_ge3` (null now) for a post-render rescore of the
   shipped 2560×1440 wallpaper (the rescore convention exists; the value is not
   stored). Two slots, both first-class; the schema does not assume the prediction
   is the truth.
2. **Descriptor-primary vs grid-primary selection.** Both substrates are stored side
   by side: hard cells (`identity.family`, `palette_candidates[].color_category`,
   `mode_candidacy[].mode`) **and** soft embeddings (`descriptors`). Either selection
   strategy runs off the same record; the schema enables the fork, does not decide it.
3. **Dense vs sparse.** Resolved per-field in the table above — identity /
   location-potential / descriptors / wallpaper-quality guaranteed; palette-candidates
   and mode-candidacy populated only where colored / probed.

## Join graph (how the instance is populated)
```
curated/manifest.jsonl ─ location_id, identity, wallpaper_quality
   │ curated_from = "<cycle>/<variant>"
   ├─▶ pools/<cycle>/images.jsonl ─ palette_candidates (K siblings), provenance.source_oid
   │      │ source_oid
   │      └─▶ fresh_runs/…/outcome_ledger.jsonl ─ location_potential
   ├─▶ data/palettes/palette_categories.json[palette] ─ color_category
   ├─▶ out/mining/deploy_tail/report.json[loc_id] ─ mode_candidacy
   └─▶ scratchpad/visual_dup/embeddings.npz[uid=curated_from] ─ descriptors
```
All five joins resolve for all 47 records (0 misses); every candidate palette (564)
resolves a `color_category`.
