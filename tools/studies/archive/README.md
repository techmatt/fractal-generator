# tools/studies/archive/ — confirmed-dead scripts (archived, not deleted)

These are the 8 files classified **dead** in the code re-audit
(`out/inventory_followup.md` §2.3): finding captured elsewhere, broken, or
orphaned. They were **archived, not deleted** — deletion remains a separate,
unauthorized pass. Nothing tracked imports any of them (verified repo-wide at
archive time).

## Move-only, byte-identical — do NOT trust their path math
Each file was moved **byte-identical, with zero edits**. A `compileall` parse-check
can't verify runtime path resolution, so editing a cold file would be an
unverified change; instead they are preserved exactly as they last worked, and any
defect is recoverable via `git log --follow <file>`.

**The one known defect is depth.** Each file computes its repo-root as
`Path(__file__)...parents[N]` (or `sys.path.insert` off it), keyed to its *original*
location. Archiving moved every file **one level deeper**
(`tools/<dir>/x.py` → `tools/studies/archive/x.py`), so every such `parents[N]`
resolves one level short of repo root. To revive one, bump its `N` by **+1** (and,
for the three that import a same-dir sibling, note the sibling moved with it).

| archived file | original path | `parents[N]` fix to revive |
|---|---|---|
| `eda_common.py` | `tools/eda/` | `parents[N]` → `parents[N+1]` |
| `smoke_test.py` | `tools/corpus/` | (also stale: asserts rev4 has 0 labels — false today) |
| `selector_family_diversity_sweep.py` | `tools/wallpaper/` | (also crashes: absent `scratchpad/_stage4_cells.json`) |
| `emission_dryrun_v2gate.py` | `tools/wallpaper/` | `spec_from_file_location("…/emission_selector.py")` path also needs repointing |
| `cliff_diag.py` | `tools/palettes/` | `parents[2]` → `parents[3]` |
| `softcliff.py` | `tools/palettes/` | `parents[2]` → `parents[3]`; imports `cliff_diag` (moved here too) |
| `render_v2_batch.py` | `tools/palettes/` | `parents[2]` → `parents[3]`; imports `cliff_diag` (moved here too) |
| `exp_vs_smooth_rankcorr.py` | `tools/render_mode_pilot/` | `parents[N]` → `parents[N+1]` |

Findings for the closed studies live in MEMORY / `docs/findings/`; see the re-audit
for the per-file rationale.
