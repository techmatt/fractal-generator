# Rescued analysis docs (uncommitted — for review)

These `.md` files lived under the gitignored `out/` tree, which the overnight
storage restructure staged to `TRASH_ROOT` (`C:\code\fractal-generator-trash`).
They were **copied** here first so nothing irreplaceable is lost when the trash is
deleted. Originals still exist in the trash until you run the finalizing `rm`.

Decide per file: keep (move into `docs/findings/` or wherever it belongs, and
commit) or drop (delete from here — the trash copy goes with the trash).

## Likely hand-authored (no generating script found — KEEP unless you know otherwise)
- `out/campaign1/readout_fixed_metric.md`
- `out/campaign1/RUNBOOK.md`
- `out/campaign2/readout_midcycle.md`
- `out/code_inventory.md`
- `out/emission_v2_report.md`

## Regenerable (a tool writes this path — safe to drop; re-run the tool to recreate)
- `out/atlas/scheduler_smoke/readout.md`, `out/campaign1/readout.md`,
  `out/campaign2/readout.md`  ← `tools/atlas/campaign1_readout.py`
- `out/descent_algorithm_current.md`, `out/descent_score_fidelity.md`  ← `tools/atlas/steered_frontier.py`
- `out/emission/campaign1_intake.md`, `out/emission/library_intake_2.md`  ← `tools/emission/*`
- `out/emission_v1_report.md`  ← `tools/emission/build_emission_diversity_v1.py`
- `out/first_release_readout.md`, `out/first_release_report.md`  ← `tools/emission/first_release_readout.py`
- `out/first_release_reselect_readout.md`  ← `tools/emission/reselect_readout.py`
- `out/phoenix_grid/readout.md`  ← `tools/phoenix/phoenix_readout.py`
- `out/pref_loc_v0_report.md`  ← `tools/ranker/report.py`
- `out/recolor_release/report.md`, `out/steered_pilot_report.md`  ← `tools/atlas/run_steered_pilot.sh`
- `out/steered_pilot_morph.md`, `out/steered_run2_report.md`  ← `tools/atlas/steered_run2_report.py`
- `out/steered_v1_2_dive_report.md`  ← `tools/atlas/steered_v1_2_dive_report.py`
- `out/inventory_followup.md`  ← referenced by `tools/studies/archive/README.md`

The regenerable/hand-authored split is a heuristic (grep for the output path in
`tools/`); when unsure the file was kept on the rescue side. Nothing here has been
committed — `git add` only what you want to keep.
