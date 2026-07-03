#!/usr/bin/env bash
# Atlas round-2: descendability pre-screen + discovery-focused 3-arm re-run
# (prompts/atlas-round2-prescreen-discovery-prompt.md).
#
#   arm1  current seeder  — native draws (the "same areas" baseline reference)
#   arm2  atlas exploit   — pure high-conf theta_hat targeting  (re-mining reference)
#   arm3  atlas explore   — pure low-conf/uncertainty targeting  (the DISCOVERY arm)
#
# Arms 2/3 are pre-screened for descendability (guided-descend depth-2 probe inside
# propose.py) so all arms approach the native ~96% productivity — the round-1 confound.
# All three then descend through the IDENTICAL injected path (only the depth-1 seed
# list differs; same --seed 0 --per-walk-rng => byte-identical depth>=2 rng streams).
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=./target/release/fractal-generator.exe
D=data/atlas/round2
R1=data/atlas/round1
mkdir -p "$D"

N=250            # emitted seeds per injected arm
# in-domain candidate cloud (pre-screened down to survivors @ ~0.26s/walk depth-2
# probe; 2500 -> ~1400 survivors at the ~58% pass rate, ample for n=250/arm + the
# explore low-conf half). Larger just enriches the survivor pool; the science is
# unchanged (every emitted seed is screened either way).
CLOUD=2500
SEEDGEN_WALKS=320
# Efficient descent config (shared by the native seedgen, the pre-screen probe, and
# the full descents).
CFG_NODE=384
CFG_SIGMA=8,10,12,14,16
CFG_OCC=0.321
CFG_BLACK=0.30

stage () { echo ""; echo "======== $* ========"; }

# --- Stage A: native seedgen (arm1) -> the current seeder's own depth-1 draws. ------
seedgen () {
  stage "A. native seedgen (arm1 current seeder, depth-1 x $SEEDGEN_WALKS)"
  $BIN guided-descend \
    --n-walks "$SEEDGEN_WALKS" --per-walk-rng --seed 0 \
    --depth-min 1 --depth-max 1 \
    --node-width "$CFG_NODE" --sigma-band "$CFG_SIGMA" \
    --descent-occ-floor "$CFG_OCC" --descent-black-cap "$CFG_BLACK" \
    --preview-width 64 --cols 40 \
    --out-dir "$D/arm1_native_seedgen"
  # Extract N productive (reached>=1) native roots -> arm1 seed list + fw pool source.
  uv run python - "$D/arm1_native_seedgen/walks.jsonl" "$D/arm1_seeds.jsonl" "$N" <<'PY'
import json, sys
walks_path, out_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for line in open(walks_path, encoding="utf-8"):
    line = line.strip()
    if not line: continue
    w = json.loads(line)
    if w["reached_depth"] >= 1 and w["root_cx"] is not None:
        rows.append({"cx": w["root_cx"], "cy": w["root_cy"], "fw": w["root_fw"],
                     "arm": "current", "tag": w["root_src"]})
rows = rows[:n]
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"extracted {len(rows)} native root seeds -> {out_path}")
PY
}

# --- Stage B: propose arms 2/3 (pre-screened exploit + explore). ---------------------
propose () {
  stage "B. propose arms 2/3 (pre-screened exploit + explore, cloud=$CLOUD)"
  uv run python tools/atlas/propose.py --mode round2 \
    --arm1-seeds "$D/arm1_seeds.jsonl" --n "$N" --cloud "$CLOUD" --seed 0 \
    --node-width "$CFG_NODE" --occ-floor "$CFG_OCC" --black-cap "$CFG_BLACK" \
    --workdir "$D/prescreen" \
    --out-exploit "$D/arm2_seeds.jsonl" --out-explore "$D/arm3_seeds.jsonl" \
    --meta-out "$D/prescreen_meta.json"
}

# --- Stage C: full injected descent (all three arms, identical path). ----------------
descend () {
  local arm=$1
  stage "C. descend $arm (injected, depth 4..14)"
  $BIN guided-descend \
    --seed-list "$D/${arm}_seeds.jsonl" --per-walk-rng --seed 0 \
    --node-width "$CFG_NODE" --sigma-band "$CFG_SIGMA" --depth-min 4 --depth-max 14 \
    --descent-occ-floor "$CFG_OCC" --descent-black-cap "$CFG_BLACK" \
    --preview-width 240 \
    --out-dir "$D/${arm}_pool"
}

# --- Stage D: k3 best-over-walk harvest (reuses round-1 harvester). -------------------
harvest () {
  local arm=$1
  stage "D. harvest $arm (k3 best-over-walk, v5)"
  uv run python tools/atlas/round1_harvest.py \
    --pool "$D/${arm}_pool" --out "$D/${arm}_table.jsonl" --workers 6 --resume
}

# --- Stage E: v5 penultimate outcome embeddings. -------------------------------------
embed () {
  local arm=$1
  stage "E. embed $arm (v5 penultimate)"
  uv run python tools/atlas/round2_embed.py --arm "$arm" --workers 6
}

case "${1:-all}" in
  seedgen) seedgen ;;
  propose) propose ;;
  descend) descend "$2" ;;
  harvest) harvest "$2" ;;
  embed)   embed "$2" ;;
  time)    # time 5 walks of a full descent to project the run
    stage "TIMING: 5-walk projection (arm1)"
    head -5 "$D/arm1_seeds.jsonl" > "$D/_time5_seeds.jsonl"
    $BIN guided-descend --seed-list "$D/_time5_seeds.jsonl" --per-walk-rng --seed 0 \
      --node-width "$CFG_NODE" --sigma-band "$CFG_SIGMA" --depth-min 4 --depth-max 14 \
      --descent-occ-floor "$CFG_OCC" --descent-black-cap "$CFG_BLACK" \
      --preview-width 240 --out-dir "$D/_time5_pool" 2>&1 | grep -E "elapsed|candidates" || true
    ;;
  rest)   # everything after seedgen+propose: descend -> harvest -> embed -> analyze
    for a in arm1 arm2 arm3; do descend "$a"; done
    for a in arm1 arm2 arm3; do harvest "$a"; done
    for a in arm1 arm2 arm3; do embed "$a"; done
    stage "F. analyze"
    uv run python tools/atlas/round2_analyze.py
    echo "=== ROUND 2 (rest) COMPLETE ==="
    ;;
  all)
    seedgen
    propose
    for a in arm1 arm2 arm3; do descend "$a"; done
    for a in arm1 arm2 arm3; do harvest "$a"; done
    for a in arm1 arm2 arm3; do embed "$a"; done
    stage "F. analyze"
    uv run python tools/atlas/round2_analyze.py
    echo "=== ROUND 2 COMPLETE ==="
    ;;
  *) echo "usage: $0 {all|rest|seedgen|propose|descend <arm>|harvest <arm>|embed <arm>|time}"; exit 1 ;;
esac
