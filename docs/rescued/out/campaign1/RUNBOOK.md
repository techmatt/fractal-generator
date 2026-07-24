# Campaign 1 — runbook (multi-day, active-time budgeted)

Budget is **accumulated active runtime**, not wall clock. State (`active_s`) survives every
resume, so stop/restart freely across days. 24h total split 60/40: **breadth 864 min**,
**dive 576 min**.

## Breadth (running now)
```bash
uv run python tools/atlas/steered_frontier.py \
  --run-dir data/discovery/campaign1/breadth \
  --julia-hook --mem-recency --budget 864 \
  >> data/discovery/campaign1/breadth_stdout.log 2>&1
```
Production defaults, unchanged from the v1.2 recency shakeout: `--julia-hook --mem-recency`
(recency_k=8), lambda_m=0.5, beta=0.02, B=32, families = the 4 c-plane. Phoenix excluded.

**Monitor**
```bash
tail -f data/discovery/campaign1/breadth_stdout.log        # live batch lines
```
Each batch prints `admitted(cum)=… frontier=… | <dt>s active=<mins>m`.

**Stop gracefully** (halts at the next batch boundary, checkpoints, exits clean):
```bash
touch data/discovery/campaign1/breadth/STOP
```
Then delete the sentinel before resuming: `rm data/discovery/campaign1/breadth/STOP`.
(Or just `kill` the process — the per-batch checkpoint means at most one batch is re-done,
and the ledger-rebuild dedup guarantees no lost/duplicate admission.)

**Resume** (same command + `--resume`; continues from `active_s`, re-checks the 864-min cap):
```bash
uv run python tools/atlas/steered_frontier.py \
  --run-dir data/discovery/campaign1/breadth \
  --julia-hook --mem-recency --budget 864 --resume \
  >> data/discovery/campaign1/breadth_stdout.log 2>&1
```
The run self-stops when `active_s + est_batch ≥ budget` (`[budget] … — stopping.`), writing
`summary.json`. To extend, resume with a larger `--budget`.

## Dive (start AFTER breadth finishes)
Single-track descents off the breadth run's admissions. 576-min budget.
```bash
uv run python tools/atlas/steered_frontier.py \
  --run-dir data/discovery/campaign1/dive \
  --dive --dive-source data/discovery/campaign1/breadth \
  --budget 576 \
  >> data/discovery/campaign1/dive_stdout.log 2>&1
```
STOP / resume identical (uses `dive_state.json`; resume rebuilds the admitted count from the
ledger). `--n-top 20 --n-control 8` (defaults) picks the dive plan; raise `--n-top` to spend
more budget. Same graceful-stop + budget gate at dive granularity.

## Readout (regenerable any time from ledgers + state)
```bash
# full (GPU morph pass) — run when NOT contending with a live discovery run:
uv run python tools/atlas/campaign1_readout.py \
  --breadth data/discovery/campaign1/breadth \
  --dive data/discovery/campaign1/dive
# cheap only (no morph; safe to run mid-campaign):
uv run python tools/atlas/campaign1_readout.py \
  --breadth data/discovery/campaign1/breadth --no-morph
```
→ `out/campaign1/readout.md`. Prior-ledger corpus for library-overlap defaults to every
`data/**/outcome_ledger.jsonl` except `campaign1`.

**Note:** run the *full* (morph) readout only when no breadth/dive run is active — the render +
CLIP pass contends for CPU/GPU and would inflate a live run's per-batch `active_s`, corrupting
the throughput accounting. Mid-campaign, use `--no-morph`.

## Crash-safety guarantees (verified)
- `active_s` accumulates per batch, checkpointed to `state.json`/`dive_state.json`, reloaded on
  `--resume` → budget is true accumulated active time across all restarts.
- Won't start a batch/dive it can't finish in the remaining budget (`active_s + est > cap`).
- Hard-kill backstop on a hung expand subprocess: `EXPAND_TIMEOUT_S=900`.
- Admission cloud is rebuilt from the durable ledger on resume, so a kill between
  ledger-append and checkpoint can neither lose nor duplicate an admission.
