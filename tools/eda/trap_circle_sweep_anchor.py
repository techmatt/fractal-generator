"""Select a random label-3 Julia anchor for the trap_circle render-mode sweep.

The label_corpus julia batch (julia_ladder_j0) is still unlabeled on disk
(scores.json == {}). The only place the J0 human labels survive is the v5
training manifest, where they were folded in for the unified Julia classifier.
So that is the source of truth for "label-3 Julia" locations.
"""
import json
import random
import sys

sys.path.insert(0, "tools/corpus")
from corpus_common import read_jsonl  # noqa: E402

MANIFEST = "data/v5/manifest.jsonl"
SEED = 20260627

rows = read_jsonl(MANIFEST)
print(f"loaded {len(rows)} rows from {MANIFEST}")

julia3 = [
    r for r in rows
    if r.get("fractal_type") == "julia" and r.get("label") == 3
]
print(f"label-3 julia records: {len(julia3)}")

# Confirm keys on one raw record before filtering downstream use.
print("\n=== RAW SAMPLE LABEL-3 JULIA RECORD ===")
print(json.dumps(julia3[0], indent=2))

rng = random.Random(SEED)
pick = rng.choice(julia3)
print(f"\nseed = {SEED}")
print("\n=== FULL SELECTED RECORD ===")
print(json.dumps(pick, indent=2))

print("\n=== ANCHOR LOCATION ===")
print(f"c_re = {pick['c_re']}")
print(f"c_im = {pick['c_im']}")
print(f"cx   = {pick['cx']}")
print(f"cy   = {pick['cy']}")
print(f"fw   = {pick['fw']}")
print(f"image_id    = {pick.get('image_id')}")
print(f"mode        = {pick.get('mode')}")
print(f"rung_index  = {pick.get('rung_index')}")
print("=== END ANCHOR LOCATION ===")
