#!/usr/bin/env bash
# Atlas round-1 3-arm acceptance: run all three arms' descents through the IDENTICAL
# injected path (only the depth-1 seed list differs), then k3-harvest each. Same
# --seed 0 + --per-walk-rng across arms => byte-identical depth>=2 rng streams.
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=./target/release/fractal-generator.exe
D=data/atlas/round1
ARMS=(arm1 arm2 arm3)

descend () {
  local arm=$1
  echo "=== DESCEND $arm ==="
  $BIN guided-descend \
    --seed-list "$D/${arm}_seeds.jsonl" --per-walk-rng \
    --node-width 384 --sigma-band 8,10,12,14,16 --depth-min 4 --depth-max 14 \
    --descent-occ-floor 0.321 --descent-black-cap 0.30 \
    --preview-width 240 --seed 0 \
    --out-dir "$D/${arm}_pool"
}

harvest () {
  local arm=$1
  echo "=== HARVEST $arm ==="
  uv run python tools/atlas/round1_harvest.py \
    --pool "$D/${arm}_pool" --out "$D/${arm}_table.jsonl" --workers 6 --resume
}

for a in "${ARMS[@]}"; do descend "$a"; done
for a in "${ARMS[@]}"; do harvest "$a"; done
echo "=== ALL ARMS DESCENDED + HARVESTED ==="
