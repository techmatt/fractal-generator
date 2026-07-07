#!/usr/bin/env python3
"""
validate_palettes.py -- mechanical checker for the Dramatic Fractal Palette Generator (v3.1).

SINGLE SOURCE OF TRUTH for the mechanical rules. The v3.1 prompt duplicates these numbers in
prose for the model's benefit, but THIS SCRIPT is authoritative: if a threshold moves, edit the
CONFIG block below and update the prompt text to match.

Two severities:
  ERROR   -- hard mechanical rules (render-breaking or spec-defined). Affects exit code.
  WARN    -- heuristic / aesthetic-adjacent guards. Advisory only; NEVER affects exit code.

The split is deliberately PERMISSIVE. The failure mode for this generator has been *samey*
palettes, not malformed ones -- so only genuinely broken things are errors. Unusual-but-valid
palettes (deep hued darks, narrow ranges, off value-key) warn at most, so the checker never
talks you out of a good, weird palette.

Terse v3.1 schema
  palette : {name, mood, architecture, skeleton, value_key, complexity, stops}
            (value_key/complexity also accepted nested under "axes" for back-compat)
  stop    : {pos, oklch:[L,C,H], role, [segment], [keypoint]}
            segment defaults to "smooth" when absent; keypoint is optional.
  The last stop repeats the first (closed loop).

Usage
  python3 validate_palettes.py batch.json      # one emitted batch (a JSON array)
  python3 validate_palettes.py                 # defaults to ./batch.json
  python3 validate_palettes.py --dir results   # LOCAL sweep of a folder of batches (convenience)
Exit code 0 iff zero ERRORS (warnings never change it).

Importable
  from validate_palettes import validate_batch, validate_palette, batch_spread
"""

import json, sys, glob, os
from collections import Counter

# ============================ CONFIG (authoritative) ==========================
# complexity -> station-count band (INCLUSIVE, counts the closing loop stop)
BAND = {1: (4, 5), 2: (5, 6), 3: (6, 7), 4: (7, 9), 5: (8, 10), 6: (11, 15)}

# cliffs carry only a SMALL step (ERROR if exceeded)
CLIFF_HUE_MAX    = 30      # deg  -- max hue step a cliff may carry
CLIFF_DL_FRAC    = 1/3     # |dL| cap for a cliff = this * palette L-range

# big saturated hue jumps on smooth/ease segments must route through black/white (ERROR)
JUMP_HUE_MIN     = 55      # deg  -- a hue step this large counts as "big"
JUMP_CHROMA_MIN  = 0.04    # both endpoints must exceed this to count as "saturated"
NEAR_BLACK       = 0.20    # a big jump is fine if an endpoint L is below this ...
NEAR_WHITE       = 0.88    # ... or above this (i.e. it passes through black/white)

# WARN-level heuristics (advisory; never block)
# skeletons whose canonical shape has <2 interior L extrema by DESIGN -> the
# ">=2 extrema" heuristic is a guaranteed false-positive on them, so exempt them
# (an inverted-arc has one trough; a cliff replaces a smooth extremum with a step).
FEW_EXTREMA_SKELETONS = {"inverted-arc", "cliff-in-mids"}
MIN_VALUE_RANGE  = 0.55    # "wide value range, always"
HIGH_KEY_MIN_L   = 0.14    # value_key:high => darkest stop should stay above
LOW_KEY_MAX_DARK = 0.12    # value_key:low  => should reach ~true black
EXTREME_L_LO     = 0.08    # near-black/white are ~achromatic ...
EXTREME_L_HI     = 0.94
EXTREME_C_MAX    = 0.10    # ... warn only past this (relaxed -- hued darks are legitimate)

# batch spread report
VIVID_PEAKC      = 0.15    # "vivid" if a palette's peak chroma >= this
MUTED_PEAKC      = 0.13    # "muted" if a palette's peak chroma <  this
# =============================================================================


# ------------------------------- helpers -------------------------------------
def huedist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

def interior_extrema(Ls):
    n = 0
    for i in range(1, len(Ls) - 1):
        if (Ls[i] - Ls[i-1]) * (Ls[i+1] - Ls[i]) < -1e-9:
            n += 1
    return n

def axis(p, key):
    if "axes" in p and isinstance(p["axes"], dict) and key in p["axes"]:
        return p["axes"][key]
    return p.get(key)

def seg(s):
    return s.get("segment", "smooth")

def is_glow(s):
    kp = s.get("keypoint")
    return s.get("role") == "glow" or (isinstance(kp, dict) and kp.get("type") == "glow_band")


# ------------------------------ core check -----------------------------------
def validate_palette(p):
    """Return (errors, warnings) as two lists of strings."""
    errs, warns = [], []
    stops = p["stops"]
    Ls = [s["oklch"][0] for s in stops]
    Cs = [s["oklch"][1] for s in stops]
    Hs = [s["oklch"][2] for s in stops]
    pos = [s["pos"] for s in stops]
    arch = p.get("architecture", "?")
    cx = axis(p, "complexity")
    cx = int(cx) if cx is not None else None
    vk = axis(p, "value_key")
    Lrange = max(Ls) - min(Ls)
    Lmax_i, Lmin_i = Ls.index(max(Ls)), Ls.index(min(Ls))

    # --- loop / ordering (ERROR) ---
    if stops[0]["oklch"] != stops[-1]["oklch"]:
        errs.append(f"loop not closed: {stops[0]['oklch']} != {stops[-1]['oklch']}")
    if abs(pos[0]) > 1e-9 or abs(pos[-1] - 1.0) > 1e-9:
        errs.append(f"pos does not run 0->1 ({pos[0]}..{pos[-1]})")
    if any(pos[i+1] <= pos[i] for i in range(len(pos)-1)):
        errs.append("pos not strictly increasing")

    # --- station count vs complexity band (ERROR) ---
    if cx in BAND:
        lo, hi = BAND[cx]
        if not (lo <= len(stops) <= hi):
            errs.append(f"{len(stops)} stops outside cx{cx} band {lo}-{hi}")

    # --- glow feathered: smooth on BOTH sides (ERROR) ---
    for i, s in enumerate(stops):
        if is_glow(s):
            out_seg = seg(s)
            in_seg = seg(stops[i-1]) if i > 0 else "smooth"
            if out_seg != "smooth" or in_seg != "smooth":
                errs.append(f"glow at pos {s['pos']} not feathered (in={in_seg}, out={out_seg})")

    # --- cliffs: small hue/value step only, never brightest<->darkest (ERROR) ---
    for i in range(len(stops)-1):
        if seg(stops[i]) == "cliff":
            dL = abs(Ls[i+1] - Ls[i]); dH = huedist(Hs[i], Hs[i+1])
            if dL > Lrange * CLIFF_DL_FRAC + 1e-9:
                errs.append(f"cliff |dL| {dL:.2f} > {CLIFF_DL_FRAC:.2f}*range "
                            f"({Lrange*CLIFF_DL_FRAC:.2f}) at pos {pos[i]}")
            if dH > CLIFF_HUE_MAX:
                errs.append(f"cliff hue step {dH:.0f}deg > {CLIFF_HUE_MAX} at pos {pos[i]}")
            if {i, i+1} == {Lmax_i, Lmin_i}:
                errs.append(f"cliff spans brightest<->darkest at pos {pos[i]}")

    # --- big saturated hue jump on smooth/ease must route thru black/white (ERROR) ---
    for i in range(len(stops)-1):
        if seg(stops[i]) in ("smooth", "ease"):
            dH = huedist(Hs[i], Hs[i+1])
            if dH > JUMP_HUE_MIN and Cs[i] > JUMP_CHROMA_MIN and Cs[i+1] > JUMP_CHROMA_MIN:
                routed = (min(Ls[i], Ls[i+1]) < NEAR_BLACK) or (max(Ls[i], Ls[i+1]) > NEAR_WHITE)
                if not routed:
                    errs.append(f"unrouted hue jump {dH:.0f}deg between mid-value stops "
                                f"pos {pos[i]}->{pos[i+1]} (L {Ls[i]:.2f}->{Ls[i+1]:.2f}); "
                                f"route it through black or white")

    # --- WARN heuristics (advisory) ---
    ec = interior_extrema(Ls)
    skel = p.get("skeleton", "?")
    if (arch != "mono-temperature-ramp" and skel not in FEW_EXTREMA_SKELETONS
            and cx is not None and cx >= 3 and ec < 2):
        warns.append(f"only {ec} interior L extrema (want >=2 for cx>=3)")
    if Lrange < MIN_VALUE_RANGE:
        warns.append(f"narrow value range {Lrange:.2f} (< {MIN_VALUE_RANGE})")
    if vk == "high" and min(Ls) < HIGH_KEY_MIN_L:
        warns.append(f"value_key high but darkest L={min(Ls):.2f} (< {HIGH_KEY_MIN_L})")
    if vk == "low" and min(Ls) > LOW_KEY_MAX_DARK:
        warns.append(f"value_key low but no true black (darkest L={min(Ls):.2f})")
    for L, C in zip(Ls, Cs):
        if (L < EXTREME_L_LO or L > EXTREME_L_HI) and C > EXTREME_C_MAX:
            warns.append(f"chroma C={C} high at extreme L={L}")

    return errs, warns


def validate_batch(palettes):
    """Return a list of {name, architecture, skeleton, errors, warnings} per palette."""
    out = []
    for p in palettes:
        e, w = validate_palette(p)
        out.append({"name": p.get("name", "?"),
                    "architecture": p.get("architecture", "?"),
                    "skeleton": p.get("skeleton", "?"),
                    "errors": e, "warnings": w})
    return out


def batch_spread(palettes):
    def col(key):
        return dict(Counter(axis(p, key) if key in ("value_key", "complexity") else p.get(key)
                            for p in palettes))
    peakC = [(p.get("name", "?"), round(max(s["oklch"][1] for s in p["stops"]), 2)) for p in palettes]
    return {"architecture": col("architecture"), "skeleton": col("skeleton"),
            "value_key": col("value_key"), "complexity": col("complexity"),
            "vivid": sum(1 for _, c in peakC if c >= VIVID_PEAKC),
            "muted": sum(1 for _, c in peakC if c < MUTED_PEAKC),
            "peakC": peakC}


# --------------------------------- CLI ---------------------------------------
def _report(path):
    P = json.load(open(path, encoding="utf-8"))
    if isinstance(P, dict):
        P = P.get("palettes", P.get("data", [P]))
    res = validate_batch(P)
    nerr = sum(len(r["errors"]) for r in res)
    nwarn = sum(len(r["warnings"]) for r in res)
    print(f"=== {path}: {len(P)} palettes | {nerr} error(s), {nwarn} warning(s) ===")
    for r in res:
        if r["errors"] or r["warnings"]:
            print(f"\n[{r['name']}]  ({r['architecture']} / {r['skeleton']})")
            for x in r["errors"]:   print(f"   ERROR  {x}")
            for x in r["warnings"]: print(f"   warn   {x}")
    sp = batch_spread(P)
    print("\n--- batch spread ---")
    print("  architecture:", sp["architecture"])
    print("  skeleton    :", sp["skeleton"])
    print("  value_key   :", sp["value_key"])
    print("  complexity  :", sp["complexity"])
    print(f"  chroma      : {sp['vivid']} vivid (peakC>={VIVID_PEAKC}), "
          f"{sp['muted']} muted (peakC<{MUTED_PEAKC})")
    return nerr


def main():
    args = sys.argv[1:]
    if args and args[0] == "--dir":
        d = args[1] if len(args) > 1 else "results"
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        tot = 0
        for f in files:
            tot += _report(f); print()
        print(f">>> TOTAL ERRORS across {len(files)} file(s): {tot}")
        sys.exit(0 if tot == 0 else 1)
    path = args[0] if args else "batch.json"
    tot = _report(path)
    print(f"\n>>> TOTAL ERRORS: {tot}")
    sys.exit(0 if tot == 0 else 1)


if __name__ == "__main__":
    main()
