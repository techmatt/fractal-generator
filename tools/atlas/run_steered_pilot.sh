#!/usr/bin/env bash
# Pilot A/B: classifier-steered frontier vs the current production walk.
#   Same families both arms (mandelbrot + multibrot3/4/5, julia hook on), same active-time
#   budget per arm, separate fresh run-scoped ledgers. Then builds out/steered_pilot_report.md.
#
#   tools/atlas/run_steered_pilot.sh <BUDGET_MIN_PER_ARM> <RUN_ROOT>
# e.g. tools/atlas/run_steered_pilot.sh 45 data/discovery/steered_pilot
set -euo pipefail
cd "$(dirname "$0")/../.."

BUDGET="${1:-45}"
ROOT_DIR="${2:-data/discovery/steered_pilot}"
FAMILIES="mandelbrot,multibrot3,multibrot4,multibrot5"
STEERED="$ROOT_DIR/steered"
BASELINE="$ROOT_DIR/baseline"
SEED=0

rm -rf "$STEERED" "$BASELINE"
mkdir -p "$STEERED" "$BASELINE"
echo "=== PILOT: budget ${BUDGET}min/arm, families ${FAMILIES}, julia hook ON ==="

# --- Arm A: steered frontier (single mixed frontier, budget = BUDGET active minutes) ---
echo "--- [arm STEERED] ---"
uv run python tools/atlas/steered_frontier.py --run-dir "$STEERED" \
    --families "$FAMILIES" --julia-hook --budget "$BUDGET" --seed $SEED \
    > "$STEERED/run.log" 2>&1 || echo "steered arm exited nonzero (continuing to report)"

# --- Arm B: baseline production walk, same families, budget split evenly, shared ledger ---
echo "--- [arm BASELINE] ---"
PERFAM=$(python -c "print(f'{$BUDGET/4:.3f}')")
BL_START=$(date +%s)
IFS=',' read -ra FAMS <<< "$FAMILIES"
for fam in "${FAMS[@]}"; do
  echo "  baseline family=$fam budget=${PERFAM}min"
  uv run python tools/atlas/production_seeder.py --run \
      --discovery-dir "$BASELINE" --family "$fam" --julia-hook \
      --budget "$PERFAM" --seed $SEED \
      >> "$BASELINE/run.log" 2>&1 || echo "  baseline $fam exited nonzero (continuing)"
done
BL_END=$(date +%s)
BL_MIN=$(python -c "print(round(($BL_END-$BL_START)/60,2))")
python -c "import json,pathlib; pathlib.Path('$BASELINE/summary.json').write_text(json.dumps({'mode':'baseline','active_min':$BL_MIN,'families':'$FAMILIES'.split(',')}))"
echo "  baseline arm active ${BL_MIN}min"

# --- Report ---
echo "--- [report] ---"
uv run python tools/atlas/steered_pilot_report.py \
    --steered "$STEERED" --baseline "$BASELINE" --budget-min "$BUDGET"
echo "=== PILOT DONE -> out/steered_pilot_report.md ==="
