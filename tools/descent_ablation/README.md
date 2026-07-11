# Descent-comparison rig

Reusable **paired-arm comparison** for `guided-descend` finder/selection knobs. Answers
"does knob X change *where* descents land, or just RNG?" without a classifier in the loop.

## What it is

- **Paired design.** One frozen root seed-list (native 8k/flat draw, harvested once) is
  reused by every arm, together with a shared `--seed` + `--per-walk-rng` (matched roots
  *and* matched per-walk sub-seeds). So the only thing that differs across arms is the knob
  under test — not the RNG draw.
- **Geometry-only contact sheets.** Terminal tiles are rendered by the Rust binary at a
  single fixed neutral palette (`twilight_shifted`, Recipe 1), node/thumb res
  (`--preview-width 320`), **never full-res, pre-classifier**. Color is held constant across
  arms so the eye compares *structure*.
- **Eye is the verdict.** The proxies (pairwise-L2 diversity, occupancy, reached-depth,
  endcause) only point the eye at which sheet to open; they do not decide.

## GUARDRAIL (load-bearing — do not compress out)

**Any diversity proxy over terminal features (pairwise-L2 etc.) MUST gate shallow deaths —
restrict to `d≥5` terminals — before you read it.** Un-gated, bland-shallow vs busy-deep
**bimodality inflates the proxy** and can *flip the arm ranking*. This was **observed, not
hypothetical**: the percentile-revive run's raw proxy said "percentile diversifies," but the
`d≥5`-gated proxy reversed it — the shipped random-survivor baseline was the *most* diverse
arm, and the raw gain was 100% shallow-death artifact (~33/129 walks dying at d1–d2). The
raw→gated collapse scales with the shallow-death rate.

**Always report raw and `d≥5`-gated side by side** so the correction is visible. Report the
per-arm shallow-death (d1–d2) rate as a first-class go/no-go stat.

## Pointers

- `run_campaign.py` — overnight ablation driver: harvests+freezes the seed-list, runs the
  A0–A7 arm matrix (legacy/percentile finder × weights × selection × pct band), budget-gated
  with a wall cap. `--finalize-only` re-runs the report over an existing dir.
- `finalize.py` — consumes the durable per-arm `pool.jsonl`/`walks.jsonl` + tile PNGs →
  per-arm & overview contact sheets, `probes.json`, `overview.html`, `report.md`. **Reuse its
  helpers** (`terminals`, `mean_pairwise_l2`, `stratified_sample`, `compose_grid`) rather than
  reimplementing.
- **Revive variant** (`run_revive.py` / `finalize_revive.py`) exists only as `__pycache__`
  (left uncommitted) — it re-tested the percentile finder against the shipped baseline and
  added the `d≥5` gating. Rebuild from `run_campaign.py` + `finalize.py` if needed; the gating
  logic is the guardrail above.

Output lands under `out/descent_ablation/<timestamp>/` (per-arm ledgers, `terminals_*.png`,
`overview.html`, `probes.json`, `report.md`).
