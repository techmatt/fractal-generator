#!/usr/bin/env python
"""q4 stage-1 coarse filter v2 — tighten the pre-filter, calibrated on the 107.

AS-FRAMED judgment: a window is labeled as the finished frame SHOWN. A good-content-
but-badly-framed window (dead corner eating part of the frame) is a REJECT, not a
croppable-accept — the dense sweep already presents the well-framed recentered crop
as its OWN candidate window, so dropping the badly-framed one loses nothing. This
means the ceilings are PLAIN (no corner-sparing, no decoration AND-condition): a
dead-cornered / barren window SHOULD drop.

Matt's ~193 remaining windows contain too many obvious rejects (57% of the 107
labeled so far are `filter_leak`). Three plain ceilings, calibrated against the 107
so borderline survives but the leak class is auto-dropped, under a HARD guardrail:
drop ZERO `accept`s. The accept-guardrail is the safety net — it auto-protects real
composed-calm windows regardless of how "barren" is measured, so the metric stays
simple.

v2 ceilings (auto-drop = record `auto_filter_v2`, NOT a human label):
  1. speckle / high-freq   busy_frac >= C_busy  with no larger coherent structure
                           (mean_struct < C_struct)   [INACTIVE on this corpus]
  2. too-large interior %  interior_frac >= C_interior
  3. too-barren            flat_frac >= C_flat         (PLAIN — a dead-cornered window
                           drops; its recentered crop is a separate low-flat candidate)

Both `analyze` (calibrate + report on the 107) and `apply` (drop failures from
the 193 unlabeled queue -> auto_filter_v2). Reads the SEPARATE q4 window store via
its canonical reader; labels from labels/q4_stage1_windows.json.

Run:  uv run python -m tools.studies.q4_stage1_filter_v2 analyze
      uv run python -m tools.studies.q4_stage1_filter_v2 apply
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.corpus import q4_window_reader as qr

BATCH_ID = "2026-07-23_q4_stage1_windows"
LABELS = ROOT / "labels" / "q4_stage1_windows.json"
STORE = qr.batch_dir(BATCH_ID)


# --------------------------------------------------------------------------- #
# v2 ceilings — SET HERE after calibration (see analyze output / findings doc).
# Chosen so: (a) 0 human `accept`s are dropped (hard guardrail, with margin);
# (b) as many `filter_leak` (+ clean `reject`) as possible are caught.
# --------------------------------------------------------------------------- #
# Calibrated on the 107 (labels/q4_stage1_windows.json); see
# docs/findings/q4_stage1_filter_v2.md for the sweep + gap analysis.
# AS-FRAMED: plain ceilings, no corner-sparing — a dead-cornered window drops.
C_INTERIOR = 0.10     # too-large dead-black interior ("eats >=10% of the frame").
                      # accepts/rejects are interior-free (acc max 0.001, rej max
                      # 0.009); 0.10 clears that envelope by 0.09 and catches 29 leaks.
C_BUSY = 0.10         # speckle detail energy ... (INACTIVE on this corpus: true
C_STRUCT = 0.16       # ... AND no coherent structure. speckle unreachable here,
                      # busy_frac accept-max=0.007; kept as a guard for future sweeps.)
C_FLAT = 0.88         # too-barren: dead/flat-exterior fraction. Sits in the
                      # [0.871, 0.884] gap — spares the lone calm accept at flat=0.871,
                      # catches the 10 leaks clustered 0.884..0.912. PLAIN ceiling.


def decoration(m):
    return m["mid_detail_frac"] + m["high_struct_frac"]


def filter_v2(m):
    """Return the v2 drop reason (str) or None to survive (as-framed, plain ceilings)."""
    if m["interior_frac"] >= C_INTERIOR:
        return "interior_heavy"
    if m["busy_frac"] >= C_BUSY and m["mean_struct"] < C_STRUCT:
        return "speckle"
    if m["flat_frac"] >= C_FLAT:
        return "barren"
    return None


# --------------------------------------------------------------------------- #
def load_joined():
    """[(window_id, features, klass_or_None)] over all 300 windows."""
    labels = json.loads(LABELS.read_text())
    rows = []
    for row, _ in qr.iter_windows(BATCH_ID):
        wid = row["window_id"]
        rows.append((wid, row["features"], labels.get(wid)))
    return rows


def _dist(name, vals_by_class):
    print(f"  {name:>18}: ", end="")
    for c in ("accept", "reject", "filter_leak"):
        v = np.array(vals_by_class[c])
        if len(v):
            print(f"{c[:3]} min{v.min():.3f} med{np.median(v):.3f} "
                  f"p90{np.percentile(v,90):.3f} max{v.max():.3f}  ", end="")
    print()


def analyze():
    rows = load_joined()
    labeled = [(w, f, k) for w, f, k in rows if k]
    print(f"labeled: {len(labeled)} / {len(rows)}")
    from collections import Counter
    print("  class counts:", dict(Counter(k for _, _, k in labeled)))

    # per-class feature distributions for the ceiling features
    keys = ["interior_frac", "busy_frac", "mean_struct", "flat_frac",
            "mid_detail_frac", "high_struct_frac", "occupancy"]
    print("\nper-class feature distributions (the ceiling features):")
    for key in keys:
        byc = {c: [f[key] for _, f, k in labeled if k == c]
               for c in ("accept", "reject", "filter_leak")}
        _dist(key, byc)
    # decoration mass composite
    byc = {c: [decoration(f) for _, f, k in labeled if k == c]
           for c in ("accept", "reject", "filter_leak")}
    _dist("decoration", byc)

    # accept guardrail envelope — the values v2 must NOT cross
    acc = [f for _, f, k in labeled if k == "accept"]
    print("\naccept envelope (hard guardrail — ceilings must clear these):")
    print(f"  max interior_frac  = {max(f['interior_frac'] for f in acc):.3f}")
    print(f"  max busy_frac      = {max(f['busy_frac'] for f in acc):.3f}")
    print(f"  min mean_struct    = {min(f['mean_struct'] for f in acc):.3f}")
    print(f"  max flat_frac      = {max(f['flat_frac'] for f in acc):.3f}")
    print(f"  interior>=C accepts: {sum(1 for f in acc if f['interior_frac']>=C_INTERIOR)}")
    print(f"  flat>=C     accepts: {sum(1 for f in acc if f['flat_frac']>=C_FLAT)}")

    # apply v2 to the labeled set -> agreement
    print(f"\nv2 ceilings (as-framed, plain): interior>={C_INTERIOR} | "
          f"(busy>={C_BUSY} & mean_struct<{C_STRUCT}) | flat>={C_FLAT}")
    catch = {c: {} for c in ("accept", "reject", "filter_leak")}
    dropped = {c: 0 for c in ("accept", "reject", "filter_leak")}
    for _, f, k in labeled:
        r = filter_v2(f)
        if r:
            dropped[k] += 1
            catch[k][r] = catch[k].get(r, 0) + 1
    n = {c: sum(1 for _, _, k in labeled if k == c) for c in dropped}
    print("\nAGREEMENT on the 107:")
    for c in ("filter_leak", "reject", "accept"):
        print(f"  {c:>12}: v2 drops {dropped[c]:>2}/{n[c]:<2}  by-reason {catch[c]}")
    print(f"\n  >>> ACCEPTS DROPPED = {dropped['accept']}  "
          f"({'PASS — guardrail holds' if dropped['accept']==0 else 'FAIL — LOOSEN'})")
    leak_caught = dropped["filter_leak"] / n["filter_leak"] if n["filter_leak"] else 0
    print(f"  >>> filter_leak caught = {dropped['filter_leak']}/{n['filter_leak']} "
          f"({leak_caught:.0%})")

    # residual leak rate on the labeled SURVIVORS (the prompt's success metric proxy;
    # the true target is the unlabeled survivors, unmeasurable — this is the estimate).
    surv = [(k) for _, f, k in labeled if not filter_v2(f)]
    from collections import Counter
    sc = Counter(surv)
    tot = len(surv)
    print(f"\nlabeled survivors: {tot}  {dict(sc)}")
    if tot:
        print(f"  residual filter_leak rate on survivors = {sc['filter_leak']}/{tot} "
              f"= {sc['filter_leak']/tot:.0%}  (was {n['filter_leak']}/{sum(n.values())} "
              f"= {n['filter_leak']/sum(n.values()):.0%} pre-v2)")
    return rows


def apply():
    """Apply v2 to the 193 UNLABELED windows; report survivors. Records dropped
    ids to auto_filter_v2.json in the store (NOT as human labels)."""
    rows = load_joined()
    unlabeled = [(w, f) for w, f, k in rows if not k]
    dropped, survivors = {}, []
    reasons = {}
    for w, f in unlabeled:
        r = filter_v2(f)
        if r:
            dropped[w] = r
            reasons[r] = reasons.get(r, 0) + 1
        else:
            survivors.append(w)
    print(f"apply v2 to {len(unlabeled)} unlabeled windows:")
    print(f"  auto_filter_v2 dropped: {len(dropped)}  by-reason {reasons}")
    print(f"  survivors (stay in label queue): {len(survivors)}")

    labeled_n = sum(1 for _, _, k in rows if k)
    total_target = labeled_n + len(survivors)
    print(f"  labeled so far: {labeled_n}  -> labeled + survivors = {total_target}")

    out = STORE / "auto_filter_v2.json"
    out.write_text(json.dumps(
        {"batch_id": BATCH_ID,
         "ceilings": {"C_INTERIOR": C_INTERIOR, "C_BUSY": C_BUSY,
                      "C_STRUCT": C_STRUCT, "C_FLAT": C_FLAT},
         "dropped": dropped,
         "n_dropped": len(dropped), "n_survivors": len(survivors)}, indent=0))
    print(f"  -> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if stage == "analyze":
        analyze()
    elif stage == "apply":
        apply()
    else:
        print("usage: q4_stage1_filter_v2 [analyze|apply]")
