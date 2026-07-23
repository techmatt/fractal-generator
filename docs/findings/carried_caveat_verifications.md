# Two carried-caveat verifications (2026-07-22)

Verification pass on two standing caveats (`prompts/caveat_verifications.md`).
Read-only investigation; **no code changes** to either subject.

## 1. `build_sheet` import red (`requests` missing) — KEEP (needs network)

`python -m palette_lib.build_sheet` fails at import
(`build_sheet.py:23 → download.py:22 import requests → ModuleNotFoundError`).
`requests` is genuinely absent from the env.

The proposed lazy-import fix was conditioned on the sheet path *never calling*
the harvester. It does call it:

- `build_sheet.main():156` → `harvest_gnofract4d()`.
- `download.harvest_gnofract4d()` makes an **unconditional** network GET to the
  GitHub tree API (`download.py:46`, `_get(TREE_URL).json()`) *before* any cache
  check — even a fully-cached run hits the network to enumerate `maps/`.

So the code path needs network regardless of import laziness. **Change nothing.**
Record the red as "needs network," not "fixable."

## 2. Walk-era julia ledger schema — no live reader; caveat retirable

Two julia ledger schemas coexist on disk:

- **walk-era / gather-standing shape:** `julia_z_cx/cy/fw` = z-plane viewport,
  `outcome_cx/cy` = parameter c.
- **campaign-era shape:** `outcome_cx/cy` = viewport, `julia_c_re/im` = parameter c.

On-disk inventory (sampled julia rows per family):

| Ledger file(s) | Schema | Read by (live?) |
|---|---|---|
| `campaign1/{breadth,dive}`, `campaign2/{breadth,dive}` | campaign-era | `campaign1_intake`, `library_intake_2`, `stage_first_release`, `build_emission_diversity_v1`, `descriptor` — **LIVE emission/release/library** |
| `phoenix_grid/grid`, `classic_phoenix` | no julia rows (grid/classic) | same live intake path |
| `discovery/gather/<class>/` | walk-era | `gather_select` (one-shot `gather_v6` batch, folded→frozen), `build_bootstrap` (one-shot wallpaper batch, frozen), `check_ledger_decode_version` (diagnostic; reads only `decoded_class`/`guard`, never julia coords), `library_records_build` (scalar potential only) |
| `discovery/outcome_ledger.jsonl` (main standing) | walk-era | `stage_first_release` — **only as a legacy negative control** proving admission *rejects* it (lines 195–207); not a source |
| `fresh_runs/prospect_run1`, `fresh_runs/overnight_*` | walk-era | `build_prospect_baserate`(+stage2) ("FROZEN … ledger"), `descent_score_fidelity` (study), `library_records_build` (scalar potential) |
| `steered_run2`, `steered_v1_2_dive`, `shakeout_*` | campaign-era | pilot/study reports (`steered_pilot_report`, `steered_pilot_morph` handle both schemas) |

**Verdict:** No live pipeline reads walk-era julia *geometry*. The
emission→release→library path reads exclusively campaign-era (`julia_c_re/im`) +
grid/classic (no-julia) ledgers; every walk-era-file reader is a frozen one-shot
batch producer, a study, or a schema-agnostic diagnostic. The caveat is safe to
retire.

**Caveat-text correction (fold in when retiring):** the walk-era *schema* is not
dead — `production_seeder` (the gather/standing seeder) still **writes**
`julia_z_*` hook rows to `gather/` + the main ledger. "Walk-era" is more
precisely "gather/standing-seeder schema," now frozen as pipeline *input*
(current discovery moved to campaign/`steered_frontier`), not an ancient
walk-only file family. Phrase the retirement as "walk-era-schema ledgers (gather
+ main standing + prospect/overnight) are read only by frozen/study/diagnostic
code," not "the schema no longer occurs."
