#!/usr/bin/env bash
# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
# Phase 1 (store) — route each 2x2 cell's pool through present to 1280x720 JPG
# crops, mirroring the rev4 corpus recipe (score-3 palettes, content focus, zoom
# 0.4, occ-floor 0.321) EXCEPT single best-composition + 1 palette per candidate
# (clean per-cell unit: one location -> one crop -> one v3 score -> one label).
# present's black(0.30)/occupancy(0.321) gates ARE the present-reject instrumentation
# (manifest + rejects/). Backgroundable. Run after run_2x2.sh completes.
set -u
BIN=target/release/fractal-generator.exe
BASE=data/guided_descend/scale_2x2
PRESENT_BASE=out/present/scale_2x2
PAL=data/palettes/score3_colormaps.json
LOG=$PRESENT_BASE/present_progress.log
mkdir -p "$PRESENT_BASE"
: > "$LOG"

present_cell () {
  local cell=$1 seed=$2
  local bridge=$PRESENT_BASE/cell_${cell}/locations.jsonl   # written by scale_2x2_cap_locations.py
  local out=$PRESENT_BASE/cell_${cell}
  echo "[$(date '+%H:%M:%S')] present cell $cell ($(wc -l < "$bridge") capped seeds)" | tee -a "$LOG"
  "$BIN" present \
    --input "$bridge" --out-dir "$out" --flat-out \
    --width 1280 --height 720 --ss 2 --maxiter 8000 \
    --format jpg --jpg-quality 90 \
    --palette-file "$PAL" --palettes-per-crop 1 \
    --occupancy-floor 0.321 --seed "$seed" \
    > "${out}.present.stdout.txt" 2> "${out}.present.stderr.txt"
  local rc=$?
  local n; n=$(ls "$out"/*.jpg 2>/dev/null | wc -l)
  echo "[$(date '+%H:%M:%S')] DONE cell $cell rc=$rc crops=$n" | tee -a "$LOG"
}

T0=$(date +%s)
present_cell a 2001
present_cell b 2002
present_cell c 2003
present_cell d 2004
T1=$(date +%s)
echo "[$(date '+%H:%M:%S')] ALL PRESENT DONE in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s" | tee -a "$LOG"
