# v7 classifier promotion — change + verification report

The v7 location-quality classifier is now the live discovery/guard/reframe scorer. Three
coupled changes landed together (any one alone leaves the pipeline incoherent — v6
thresholds against v7 scores would reject nearly everything). **Not committed.**

## 1. Decode-version predicate — made version-general

`tools/corpus/corpus_common.py`: the v6-hardcoded guard was replaced with a version-general
one. `is_v6_decoded` / `V6_SCORER_VERSION` / `v6_rows_only` / `require_v6` / `V5DecodeError`
are **deleted (no alias)** and replaced by:

- `is_decoded_by(row, version)` — explicit-version primitive.
- `is_current_decoded(row)` — compares `row.scorer_version` against the active checkpoint's
  version, resolved from `tools/scoring/active_ckpt.ACTIVE_VERSION` (the one source of truth
  for "current"). New helper `active_scorer_version()` does the resolution.
- `current_rows_only(rows)` / `require_current(row)` / `StaleDecodeError` — the quarantine
  helpers, renamed and re-keyed on "current" (behavior unchanged in kind; only the version
  it keys on is now dynamic). None had external callers.

`tools/scoring/active_ckpt.py` now exposes `ACTIVE_VERSION = Path(ACTIVE_CKPT).parent.name`.

**Callsites migrated to `is_current_decoded` (want "decoded by the model that is live now"):**

| file | site |
|---|---|
| `tools/wallpaper/overnight_orchestrator.py` | emit-selection filter (L382) |
| `tools/wallpaper/build_fresh_discovery.py` | fresh-q3 emit filter (L215) + docstrings/`filter` label |
| `tools/wallpaper/build_headbatch_dramatic.py` | dramatic-head emit filter (L274) + `filter` label |
| `tools/coevo/analyze_round.py` | per-round current-stamp confirmation (L170) + report keys |
| `tools/coevo/coevo_round.py` | recap counter — was an **open-coded** `== "v6"` (L98), now routed through the helper |

**Kept v6-specific, flagged (calls `is_decoded_by(row, "v6")`):**

- `tools/atlas/check_ledger_decode_version.py` (L66). This is a **historical v5-vs-v6
  migration audit** whose `"v5"`/`"v6"` columns mean exactly those versions. Tracking the
  active checkpoint here would relabel every genuine v6 row as `"v5"` after the v7 flip and
  corrupt the readout, so it pins the explicit version. It is the one real "v6 specifically"
  distinction. (Consequence: the tool now audits the v5→v6 era specifically and no longer
  reflects "what the live head can draw" — that role moved to the `is_current_decoded`
  emit selectors above. Left as-is; re-scope it if a v6→v7 ledger audit is wanted.)

The stored stamp is **not** hand-changed: `production_seeder.SCORER_VERSION =
Path(SCORER_PATH).parent.name` already derives from `ACTIVE_CKPT`, so new rows now stamp
`"v7"` automatically. Passthrough sites that merely copy `scorer_version` were left alone,
as were the unrelated `v3_gvo` preference-scorer `scorer_version` fields and all
`data/classifier/v6` path strings.

## 2. Threshold table

`tools/atlas/production_seeder.py::T_GOOD_OVERRIDES` replaced with the v7 F2-derived values
from `docs/findings/v7_t_good.md`:

```python
T_GOOD_BASELINE = 0.50
T_GOOD_OVERRIDES = {
    "mandelbrot": 0.14, "julia:mandelbrot": 0.22,
    "julia:multibrot3": 0.25, "julia:multibrot4": 0.17, "julia:multibrot5": 0.10,
}
```

phoenix (0 v7 eval positives — undecidable; the v6-era 0.18 was on v6's p_good scale) and
native multibrot3/4/5 (uncalibrated, no eval either way) deliberately fall to baseline 0.50.
No speculative entries.

**Routing check** — every emittable partition resolves deliberately, none falls to baseline
on a name mismatch, no stray table keys:

```
mandelbrot           t_good=0.14  DERIVED
julia:mandelbrot     t_good=0.22  DERIVED
julia:multibrot3     t_good=0.25  DERIVED
julia:multibrot4     t_good=0.17  DERIVED
julia:multibrot5     t_good=0.10  DERIVED
multibrot3/4/5       t_good=0.50  baseline (deliberate, uncalibrated)
phoenix              t_good=0.50  baseline (deliberate, undecidable)
```

## 3. Flip

`tools/scoring/active_ckpt.py`: `ACTIVE_CKPT = "data/classifier/v7/model_best.pt"`.
Added `V6_CKPT_ROLLBACK` (the one-flip rollback anchor, role v5 held); v5 stays as deeper
rollback. All `# currently v6` comments in the gate path (production_seeder, guard,
cross_family_shakeout, reframe) updated to v7.

## 4. Canary

`tests/test_tracked_artifacts.py`: added `data/classifier/v7/model_best.pt` (live) and
`data/classifier/v6/model_best.pt` (rollback anchor); v5 left in place. Both weights
force-added to the git index (staged, not committed) so the canary has something to assert.

## Verification

- **Canary proven red on purpose.** Temporarily pointed the v7 path at a nonexistent file →
  `test_canary_tracked` FAILED naming the file (`... model_best_CANARY_REDPROOF_NONEXISTENT.pt`,
  git: "did not match any file(s)"). Reverted; all 28 green.
- **Import smoke, one fresh process per edited file.** 11/12 import clean (corpus_common,
  active_ckpt, production_seeder, guard, cross_family_shakeout, check_ledger_decode_version,
  reframe, overnight_orchestrator, build_fresh_discovery, analyze_round, coevo_round). The
  one failure — `build_headbatch_dramatic` (`AttributeError: module 'build_fresh_discovery'
  has no attribute 'DEG2_FAMILIES'`) — is **PRE-EXISTING**: it reproduces on HEAD with all
  my changes stashed, and it is NOT on the overnight emit path. My edit to that file is
  before the crash line and byte-compiles. Flagged, not fixed (out of scope).
- **Loud-late subprocess surface:**
  - `production_seeder --smoke` — ran 2 full batches under `data/classifier/v7/model_best.pt,
    v7`; decode + v7 stamp working; q3+4/batch; zero guarded/import errors.
  - `prospect_orchestrator --mini` — spawned the discovery child (`--family mandelbrot
    --julia-hook`), which returned **rc=0** (c-plane 6 / hook 3 / q3 base 2 twin 2). No emit
    phase (as documented).
  - `overnight_orchestrator --mini` — discovery child rc=0; both emit-phase spawned modules
    (`build_fresh_discovery`, `emit_v1`) import clean in fresh processes; and the **real emit
    build step** run against the mini run's fresh v7 ledger (18 rows, all `scorer_version=v7`;
    8 guard-pass q3, all current-decoded) printed `filter: scorer_version==v7 & decoded_class==3
    & guard_pass`, selected 8→7 locations, and rendered crops — the `is_current_decoded`
    selection path exercised for real under v7.
- **Predicate semantics** (functional): `active_scorer_version()=="v7"`;
  `is_current_decoded` False on a v6 row / True on a v7 row / False on unstamped;
  `is_decoded_by(v6row,"v6")` True; `require_current(v6row)` raises `StaleDecodeError`;
  `current_rows_only` keeps 1 / excludes 2 of {v6,v7,unstamped}.
- **Run-1 ledger** (`data/discovery/fresh_runs/prospect_run1/`): 1616 v6-stamped rows,
  **0** read as current under v7 — correct and expected. Read-only; not modified/migrated
  (git still sees it as untracked, no staged/modified entries).
- **Default pytest green: 176 passed, 3 deselected (slow).** One fixture fix was required:
  `tools/wallpaper/test_prospect.py::_ledger_row` hardcoded `scorer_version="v6"`, which the
  fresh-q3 harvest (now `is_current_decoded`) correctly reads as stale under v7 → "0 fresh
  q3" → 8 loop tests failed. This is the intended new behavior surfacing in a stale fixture,
  not a code bug: the real seeder always stamps the *active* version, so the fixture now uses
  `active_scorer_version()` and stays correct across future flips. Confirmed the 8 failures
  are caused by the promotion (37/37 pass on HEAD with changes stashed) and fixed by the
  fixture (37/37 pass after).

## Behavior changes beyond the three edits

- **phoenix q3 admission tightens 0.18 → 0.50.** The v6 provisional 0.18 was fit on v6's
  p_good scale; v7 has no phoenix eval to re-derive it, so it falls to baseline. Deliberate;
  documented in `v7_t_good.md`.
- **All existing v6-stamped ledger rows now read as not-current**, so the fresh-discovery /
  wallpaper-head emit selectors admit nothing until a v7 discovery run produces v7 rows.
  This is the intended consequence of the flip, not a regression.
- **`tools/atlas/verify_v6_gate.py`** contains `assert "v6" in ACTIVE_CKPT` (L85) and will
  now FAIL if run — it is a v6-specific pin verifier, superseded by the v7 promotion. Not on
  any smoke path or in the pytest collection, so it turns nothing red. Left as-is (flagging,
  not fixing — out of scope).
