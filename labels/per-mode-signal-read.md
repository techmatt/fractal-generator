# Per-mode signal read — render_mode_pilot_v1

Pure analysis on the labels — no rendering. Load `render_mode_pilot_v1.json` and join `images.jsonl` provenance (`render_mode`, `mode_params`, `transfer_dropped`, family).

1. **Per mode:** q1/q2/q3 counts, q3-rate and (q2+q3)-rate. Sort by q3-rate. This is the headline — which modes yield wallpapers.
2. **`transfer_dropped` cross-tab** (Rust-path modes): q3-rate among dropped vs not-dropped. Does losing grad actually hurt the score?
3. **Direct-trap param grid:** q3-rate by `opacity`×`threshold` cell — which cells produce good rasters (informs deploy-time param exploration).
4. **Family × mode:** coarse q3-rate table where counts support it; mark thin cells.
5. **Overall** q1/q2/q3 split.

Flag in the output (not yet applied — so we read them right): near-smooth rasters not excluded → inflates exp_smoothing and any near-smooth-prone mode; additive screen-composite blowout not yet rolloff-fixed → deflates those.

Report the tables.
