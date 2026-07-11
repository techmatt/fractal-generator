# Render-mode registry

**Generated from `specs/modes_registry.json` — do not hand-edit.**
Regenerate: `uv run python tools/specs/gen_registry.py`. Consistency (keys match specs/, valid tiers, stamped-tier parity) is enforced by `cargo test --test modes_registry`.

Counts: **8 promoted**, **8 niche** (16 total). Deletion candidates: **0**.

## Promoted — standard / reference render modes

- **`smooth`** — Canonical smooth-iteration gradient / base carrier all composites build on; fully palette-respecting and robust across locations.
- **`tia`** — Triangle-inequality-average banding; strong standalone, skip 1 (drops the unstable first term), linear transform (log/sqrt are coloring-curve options, not separate modes).
- **`stripe`** — Stripe-average banding; combed flowing striations; density 6, linear transform (the standalone form of the c13 composite base).
- **`smooth_mean_angle`** — Smooth base + gaussian_int texture (color_by mean_angle), screen combine, weight 0.85.
- **`smooth_angle_min`** — Smooth base + gaussian_int texture (color_by angle_min), screen combine, weight 0.85.
- **`composite_c7_smooth_trap_circle`** — C7 flagship: smooth base + trap_circle texture, screen combine, weight 0.85.
- **`composite_c13_smooth_stripe`** — C13 flagship: smooth base + stripe (density 6) texture, screen combine, weight 0.85.
- **`composite_c17_smooth_curvature`** — C17 flagship: smooth base + curvature texture, screen combine, weight 0.85.

## Niche — location-specialists, composite textures, exploration scaffolding

- **`exp_smoothing`** — Exponential-smoothing smooth-escape field (Σ exp(−|z|)), linear stretch. _Averaging-family drop-in alternative to smooth, added with the UF-algorithm reconstruction before smooth was promoted as the canonical base carrier — redundant with it by design. Its field is rank-corr ≈1 (Spearman ≥0.999 across all 8 families) with smooth, so the pilot's 34 rasters were pixel-dupes (ΔE76<5, all flagged too_close_to_smooth). Its one nominal knob divergescale is hardcoded 1.0 and would be absorbed by the percentile-stretch anyway (no live parameter; no fixed-palette path exists). Deprecated for render-mode exploration; smooth is the canonical carrier._
- **`gaussian_int`** — Pure gaussian-integer min-distance lattice-trap field, linear. _Min-distance field reads as sparse dots (heavily-peaked distribution); palette-poor solo. Its promoted form is the composites that use it as a texture (c7 / mean_angle / angle_min)._
- **`direct_trap_ring`** — Direct-trap composite, ring shape (r=1), screen over black start. _Cleanest of the direct-trap family (floor <2%) but palette-indifferent by construction (raw d/threshold key, no whole-image normalization). Demoted with the family; genuine palette-respect needs two-pass whole-image key-normalization — parked as not worth it._
- **`direct_trap_screen`** — Direct-trap composite, cross shape, screen over black start. _Scalar twin trap_cross floor-clusters ~22%; location-specialist, palette-indifferent. Family palette-respect needs two-pass whole-image key-normalization — parked._
- **`direct_trap_multiply`** — Direct-trap composite, cross shape, multiply over white start (dark lace on light). _The dark-lace-on-light look depends on start_color + multiply, not the palette; location-specialist. Family palette-respect needs two-pass whole-image key-normalization — parked._
- **`direct_trap_lines`** — Direct-trap composite, anisotropic |Im z| lines, screen over black start. _Narrowest / most directional of the family; palette-indifferent by construction. Family palette-respect needs two-pass whole-image key-normalization — parked._
- **`trap_circle`** — Solo trap_circle field (min of ||z|-r|), no shade. _Not viable solo at high-complexity locations; its promoted home is the C7 composite texture._
- **`curv_linear`** — Curvature single-field, linear transform. _Curvature single-field transform probe; curvature's promoted home is the C17 composite texture._

## Deletion candidates (flagged, not deleted)

_(none)_
