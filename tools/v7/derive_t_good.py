#!/usr/bin/env python
"""Re-derive the per-partition q3 `t_good` table for v7 (report-only; no seeder edits).

The shipped table (production_seeder.T_GOOD_OVERRIDES) is calibrated to v6's p_good
distribution and is meaningless under v7. This pass re-derives it from the v7 labeled
eval slice and writes docs/findings/v7_t_good.md. It does NOT edit production_seeder.

  uv run python tools/v7/derive_t_good.py

CPU-only, seconds. Gates are aborts (SystemExit), not asserts-in-prose.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

import label_store as ls                                    # noqa: E402
from score_lib import corn_decode                           # noqa: E402
from production_seeder import (                              # noqa: E402
    t_good_for, T_GOOD_OVERRIDES, T_GOOD_BASELINE, julia_partition,
)

MANIFEST = ROOT / "data" / "v7" / "manifest.jsonl"
EVAL = ROOT / "data" / "classifier" / "v7" / "eval_scores_v7.jsonl"
BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")

# fractal_type (Rust kind_str) -> ledger partition key (what t_good_for is keyed on)
FT2FAM = {
    "mandelbrot": "mandelbrot",
    "julia": "julia:mandelbrot",
    "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
    "julia_multibrot3": "julia:multibrot3",
    "julia_multibrot4": "julia:multibrot4",
    "julia_multibrot5": "julia:multibrot5",
    "phoenix": "phoenix",
}
FAM2FT = {v: k for k, v in FT2FAM.items()}

MIN_POS = 15                    # gate 3 sufficiency floor
GRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]   # [0.02, 0.98]


def die(msg: str) -> None:
    print(f"\n*** ABORT: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Labels — reconstructed ONLY through label_store.resolve_score (crops->location,
# label = max). Cross-checked against the manifest label so we never silently trust
# a hand-rolled reader.
# --------------------------------------------------------------------------- #
def ftype_of(row):
    fam = (row.get("provenance") or {}).get("family")
    if fam and fam in FT2FAM:
        return fam if fam in FAM2FT.values() and False else FAM2FT.get(fam)  # noqa
    return None


def resolve_location_labels():
    """{(ft,cx,cy,fw,c_re,c_im): max_score} over the whole corpus via label_store."""
    # partition/family -> fractal_type map used by the manifest builder (authoritative).
    FAM2FT_LOCAL = {
        "mandelbrot": "mandelbrot",
        "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
        "julia:mandelbrot": "julia",
        "julia:multibrot3": "julia_multibrot3",
        "julia:multibrot4": "julia_multibrot4",
        "julia:multibrot5": "julia_multibrot5",
        "phoenix": "phoenix",
    }
    locs = defaultdict(list)
    joined = defaultdict(int)
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            score = ls.resolve_score(row, sidecar)
            if score is None:
                continue
            joined[batch_id] += 1
            fam = (row.get("provenance") or {}).get("family")
            ft = FAM2FT_LOCAL.get(fam) if fam else None
            if ft is None:
                ft = row["render"].get("fractal_type") or "mandelbrot"
            rd = row["render"]
            key = (ft, rd["cx"], rd["cy"], rd["fw"], rd.get("c_re"), rd.get("c_im"))
            locs[key].append(int(score))
    ls.assert_sidecars_joined(joined)     # loud on a broken sidecar join
    return {k: max(v) for k, v in locs.items()}


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def confusion(rows, t):
    """rows: list of (p_notbad, p_good, is_pos). Predicted-q3 iff corn_decode==3 at t."""
    tp = fp = fn = 0
    for nb, g, pos in rows:
        pred = corn_decode(nb, g, t) == 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return tp, fp, fn


def prf2(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) else 0.0
    return prec, rec, f2


def best_t(rows):
    """argmax F2 over GRID; tie-break toward HIGHER t (equal F2, fewer FPs)."""
    best = None
    for t in GRID:
        _, _, f2 = prf2(*confusion(rows, t))
        if best is None or f2 > best[1] + 1e-12 or (abs(f2 - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f2)
    return best[0]


def loo_f2(rows):
    """Leave-one-out: t selected on the other n-1 rows, prediction scored on the held row;
    aggregate confusion over all held rows -> OOF P/R/F2."""
    tp = fp = fn = 0
    for i in range(len(rows)):
        rest = rows[:i] + rows[i + 1:]
        t = best_t(rest)
        nb, g, pos = rows[i]
        pred = corn_decode(nb, g, t) == 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return prf2(tp, fp, fn)


# --------------------------------------------------------------------------- #
def main():
    man = read_jsonl(MANIFEST)
    esc = read_jsonl(EVAL)

    # ---- join eval_scores -> manifest by location_id (== manifest row index). ----
    for r in esc:
        lid = r["location_id"]
        if lid >= len(man):
            die(f"eval location_id {lid} out of manifest range")
        mr = man[lid]
        if mr["split"] != "eval":
            die(f"eval location_id {lid} is split={mr['split']} in manifest (gate 2)")
        if mr.get("biased"):
            die(f"eval location_id {lid} is biased in manifest (gate 2)")
        if mr["fractal_type"] != r["fractal_type"] or mr["label"] != r["label"]:
            die(f"eval location_id {lid} manifest/eval ft/label mismatch")
        r["_man"] = mr

    # ---- labels ONLY via label_store; cross-check vs manifest label. ----
    print("resolving labels via label_store.resolve_score (crops->location, max) ...")
    lab = resolve_location_labels()
    mism = 0
    for r in esc:
        mr = r["_man"]
        key = (mr["fractal_type"], mr["cx"], mr["cy"], mr["fw"], mr.get("c_re"), mr.get("c_im"))
        ls_score = lab.get(key)
        if ls_score is None:
            die(f"eval loc {r['location_id']} ({mr['fractal_type']}) unresolved via "
                f"label_store — hand-rolled reader would have dropped it")
        if ls_score != mr["label"]:
            mism += 1
    if mism:
        die(f"{mism} eval locations: label_store label != manifest label")
    print(f"  OK — all {len(esc)} eval labels resolve via label_store and match the manifest.\n")

    # ---- assemble per-partition slices. julia:multibrot -> CENSUS-only (Option A). ----
    parts = defaultdict(list)      # partition -> list[(nb,g,is_pos)]
    remnant = defaultdict(list)    # julia:mb unbiased frozen remnant (reported, not used)
    for r in esc:
        mr = r["_man"]
        part = FT2FAM[mr["fractal_type"]]
        row = (r["v7_p_not_bad"], r["v7_p_good"], mr["label"] == 3)
        if part.startswith("julia:multibrot"):
            if mr.get("source") == "prospect_census":
                parts[part].append(row)
            else:
                remnant[part].append(row)
        else:
            parts[part].append(row)

    # =================================================================== #
    # GATE 1 — census coverage.
    # =================================================================== #
    census = [r for r in esc if r["_man"].get("source") == "prospect_census"]
    census_pos = sum(1 for r in census if r["label"] == 3)
    print("=" * 72)
    print("GATE 1  census coverage")
    print("=" * 72)
    print(f"  julia:multibrot census: n={len(census)}  positives={census_pos}")
    for p in ("julia:multibrot3", "julia:multibrot4", "julia:multibrot5"):
        n = len(parts[p]); pos = sum(1 for _, _, x in parts[p] if x)
        rn = len(remnant[p]); rpos = sum(1 for _, _, x in remnant[p] if x)
        print(f"    {p:20s} census n={n:3d} pos={pos:3d}   (+frozen remnant n={rn} pos={rpos}, excluded)")
    if not (140 <= len(census) <= 148 and 60 <= census_pos <= 74):
        die(f"census coverage off expectation (~144 loc / ~67 pos): got {len(census)}/{census_pos}")
    print("  OK — census fully covered.\n")

    # =================================================================== #
    # GATE 2 — eval hygiene (checked inline above during the join).
    # =================================================================== #
    print("=" * 72)
    print("GATE 2  eval hygiene")
    print("=" * 72)
    print(f"  all {len(esc)} scored locations are split=eval & biased=False in the v7 manifest.")
    print("  OK.\n")

    # =================================================================== #
    # GATE 3 + derivation, per partition.
    # =================================================================== #
    print("=" * 72)
    print("GATE 3  sufficiency (>=15 positives) + F2 derivation")
    print("=" * 72)

    NO_DERIVE = {"multibrot3", "multibrot4", "multibrot5"}   # native: no eval either way
    results = {}
    undecidable = {}
    for part in sorted(set(list(parts) + list(FT2FAM.values()))):
        if part in NO_DERIVE:
            continue
        rows = parts.get(part, [])
        n = len(rows); pos = sum(1 for _, _, x in rows if x)
        v6t = T_GOOD_OVERRIDES.get(part, T_GOOD_BASELINE)
        if pos < MIN_POS:
            undecidable[part] = (n, pos, v6t)
            print(f"  {part:20s} n={n:4d} pos={pos:3d}  UNDECIDABLE (<{MIN_POS} pos) -> baseline {T_GOOD_BASELINE}")
            continue
        t = best_t(rows)
        p_t, r_t, f2_t = prf2(*confusion(rows, t))
        p6, r6, f26 = prf2(*confusion(rows, v6t))
        tp_t, fp_t, fn_t = confusion(rows, t)
        tp6, fp6, fn6 = confusion(rows, v6t)
        oof_p, oof_r, oof_f2 = loo_f2(rows)
        results[part] = dict(
            n=n, pos=pos, t=t, v6t=v6t,
            f2_in=f2_t, f2_oof=oof_f2, gap=f2_t - oof_f2,
            rec=r_t, prec=p_t, admit=tp_t + fp_t, discarded=fn_t,
            v6_rec=r6, v6_prec=p6, v6_f2=f26, v6_admit=tp6 + fp6, v6_discarded=fn6,
            oof_p=oof_p, oof_r=oof_r,
        )
        print(f"  {part:20s} n={n:4d} pos={pos:3d}  t*={t:.2f}  "
              f"F2_in={f2_t:.3f} F2_oof={oof_f2:.3f} (gap {f2_t-oof_f2:+.3f})  "
              f"P={p_t:.3f} R={r_t:.3f} admit={tp_t+fp_t} disc_q3={fn_t}   "
              f"[v6 t={v6t:.2f}: F2={f26:.3f} P={p6:.3f} R={r6:.3f} disc_q3={fn6}]")

    # =================================================================== #
    # ROUTING — every partition the seeder can emit is covered.
    # =================================================================== #
    print("\n" + "=" * 72)
    print("ROUTING  every emittable partition resolves deliberately")
    print("=" * 72)
    emittable = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
                 julia_partition("mandelbrot"),
                 julia_partition("multibrot3"), julia_partition("multibrot4"),
                 julia_partition("multibrot5"), "phoenix"]
    proposed = {p: results[p]["t"] for p in results}
    for part in emittable:
        if part in proposed:
            where = f"DERIVED  t={proposed[part]:.2f}"
        elif part in undecidable:
            where = f"undecidable -> baseline {T_GOOD_BASELINE}"
        elif part in NO_DERIVE:
            where = f"native, uncalibrated -> baseline {T_GOOD_BASELINE}"
        else:
            where = f"baseline {T_GOOD_BASELINE}"
        print(f"  {part:20s} {where}")

    # ---- dump machine-readable summary for the doc writer ----
    out = dict(results=results, undecidable=undecidable,
               native=sorted(NO_DERIVE), emittable=emittable,
               census_n=len(census), census_pos=census_pos,
               proposed={p: results[p]["t"] for p in results})
    (ROOT / "data" / "v7" / "t_good_derivation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote data/v7/t_good_derivation.json")


if __name__ == "__main__":
    main()
