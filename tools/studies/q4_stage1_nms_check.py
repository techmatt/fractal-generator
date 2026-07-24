#!/usr/bin/env python
"""NMS density spot-check for the q4 stage-1 sweep (as-framed verification).

As-framed labeling is lossless ONLY if, for every dead-cornered / barren window we
now DROP, the well-framed *recentered* crop genuinely survives as its own candidate.
The sweep's NMS (NMS_IOU=0.35) ran on the occupancy-biased `score_A` composite — the
score that prefers a busier neighbor over a balanced one — so it may have SUPPRESSED
the good recentered crop that overlaps a higher-scoring dead-cornered window.

This reconstructs the PRE-NMS candidate set per minibrot and asks, for each
dead-cornered window (v2 `interior_heavy`): does a well-framed recentered candidate
(passes v2, lower interior, near it, same scale) exist pre-NMS, and did NMS KEEP it
or SUPPRESS it? A high suppression rate => production sweep needs looser NMS / finer
stride. Verify + flag; do not rebuild.

Run: uv run python -m tools.studies.q4_stage1_nms_check
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.studies import q4_stage1_labelset as H
from tools.studies.q4_stage1_filter_v2 import filter_v2


def all_candidates(mb_id):
    """Every window pre-NMS: (box, cx, cy, scale, score, metrics)."""
    field, fw, fh = H.load_field_values(mb_id)
    cands = []
    for s in H.SCALES:
        Wp = max(8, int(round(s * fw)))
        Hp = max(8, int(round(Wp * 9 / 16)))
        if Hp >= fh or Wp >= fw:
            continue
        st = max(4, int(round(H.STRIDE_FRAC * Wp)))
        for y in range(0, fh - Hp + 1, st):
            for x in range(0, fw - Wp + 1, st):
                m = H.compute_metrics(field[y:y+Hp, x:x+Wp])
                cands.append(dict(box=(x/fw, y/fh, Wp/fw, Hp/fh),
                                  cx=(x+Wp/2)/fw, cy=(y+Hp/2)/fh, scale=s,
                                  score=float(H.score_A(m)), m=m))
    return cands


def nms_keep_flags(cands):
    """Greedy NMS exactly as the harness runs it; return per-cand kept bool."""
    order = sorted(range(len(cands)), key=lambda i: cands[i]["score"], reverse=True)
    kept_boxes, kept = [], [False]*len(cands)
    for i in order:
        b = cands[i]["box"]
        if all(H._iou(b, kb) <= H.NMS_IOU for kb in kept_boxes):
            kept_boxes.append(b); kept[i] = True
    return kept


def is_well_framed(m):
    """A survivor of v2 that is genuinely composed: passes v2 AND has real
    decoration (not the calm/near-empty tail). The 'good recentered crop' target."""
    from tools.studies.q4_stage1_filter_v2 import decoration
    return filter_v2(m) is None and decoration(m) >= 0.30


def main():
    mbs = H.load_minibrots()
    n_dead = 0            # dead-cornered windows examined (pre-NMS, interior-heavy)
    n_has_good = 0        # ... that have a well-framed recentered neighbor pre-NMS
    n_good_kept = 0       # ... where >=1 such neighbor SURVIVED NMS
    supp_examples = []
    for mb in mbs:
        if not (H.FIELDS / f"{mb['id']}.bin").exists():
            continue
        cands = all_candidates(mb["id"])
        kept = nms_keep_flags(cands)
        for i, c in enumerate(cands):
            if c["m"]["interior_frac"] < 0.20:      # only clearly dead-cornered ones
                continue
            n_dead += 1
            # recentered neighbors: same scale, center within 1.0 window-width,
            # lower interior, well-framed.
            wsz = c["box"][2]
            neigh = [(j, d) for j, d in enumerate(cands)
                     if d["scale"] == c["scale"] and j != i
                     and abs(d["cx"]-c["cx"]) <= wsz and abs(d["cy"]-c["cy"]) <= wsz
                     and d["m"]["interior_frac"] < c["m"]["interior_frac"] - 0.05
                     and is_well_framed(d["m"])]
            if not neigh:
                continue
            n_has_good += 1
            kept_neigh = [j for j, _ in neigh if kept[j]]
            if kept_neigh:
                n_good_kept += 1
            elif len(supp_examples) < 6:
                best = max(neigh, key=lambda t: t[1]["score"])[1]
                supp_examples.append(
                    f"{mb['id']} s{c['scale']} dead@({c['cx']:.2f},{c['cy']:.2f}) "
                    f"int={c['m']['interior_frac']:.2f} -> good recenter "
                    f"({best['cx']:.2f},{best['cy']:.2f}) int={best['m']['interior_frac']:.2f} "
                    f"SUPPRESSED by NMS")

    print("=== NMS density spot-check (as-framed losslessness) ===")
    print(f"dead-cornered windows examined (pre-NMS, interior>=0.20): {n_dead}")
    print(f"  ... with a well-framed recentered neighbor pre-NMS: {n_has_good} "
          f"({n_has_good/n_dead:.0%})" if n_dead else "")
    if n_has_good:
        print(f"  ... where that good recenter SURVIVED NMS:          {n_good_kept} "
              f"({n_good_kept/n_has_good:.0%})")
        print(f"  ... where NMS SUPPRESSED every good recenter:       "
              f"{n_has_good - n_good_kept} ({1-n_good_kept/n_has_good:.0%})")
    print("\nsuppression examples (good recentered crop existed but NMS dropped it):")
    for e in supp_examples:
        print("  " + e)
    if not supp_examples:
        print("  (none — good recenters consistently survive NMS)")


if __name__ == "__main__":
    main()
