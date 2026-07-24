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
C_FLAT = 0.88         # too-barren: dead/flat-exterior fraction. Sits in the
                      # [0.871, 0.884] gap — spares the lone calm accept at flat=0.871,
                      # catches the 10 leaks clustered 0.884..0.912. PLAIN ceiling.
# --- speckle (Bug 2 fix) --------------------------------------------------- #
# The stored `busy_frac` was broken: fine = |work - 3x3 lowpass| with FINE_SPECKLE
# =0.30 never fires (dense banding is a locally-smooth steep gradient, residual ~0;
# amplitude also compressed by the 0.5-99.5 percentile stretch), so the old speckle
# ceiling caught 0 ("unreachable"). Replaced by a real fine-scale FREQUENCY measure
# computed from the FIELD, not the stored features:
#   speckle_ratio = hf_mean / coarse_std
#     hf_mean    = mean |DoG(0.8,1.8)|   (pixel-scale oscillation energy)
#     coarse_std = std of a sigma-6 lowpass (large-FORM variation)
# The discriminator is high fine energy WITHOUT coarse structure: coherent ornate
# detail (spirals) keeps strong coarse-form variation (low ratio) and survives;
# granular static has fine energy but a near-uniform coarse field (high ratio).
C_SPECKLE = 0.30      # speckle_ratio ceiling. accept-max=0.285 (a coherent dendrite),
                      # rej-max=0.278; the mb04 speckle wall=0.327, pure-speckle leaks
                      # up to 1.066. 0.30 clears the accept envelope by 0.015.
HF_FLOOR = 0.012      # require real fine energy present (accept hf_mean min); guards
                      # near-flat windows from a spurious ratio (tiny/tiny).


def decoration(m):
    return m["mid_detail_frac"] + m["high_struct_frac"]


def filter_v2(m, spk=None):
    """Return the v2 drop reason (str) or None to survive (as-framed, plain ceilings).
    `spk` = (hf_mean, speckle_ratio) from the field; None disables the speckle rule."""
    if m["interior_frac"] >= C_INTERIOR:
        return "interior_heavy"
    if spk is not None and spk[0] >= HF_FLOOR and spk[1] >= C_SPECKLE:
        return "speckle"
    if m["flat_frac"] >= C_FLAT:
        return "barren"
    return None


# --------------------------------------------------------------------------- #
# Field-based speckle score (Bug 2). Loads the dumped fields (out/q4_stage1/
# fields/), crops each window, computes (hf_mean, speckle_ratio). Cached to the
# STORE (committed) so the filter runs without regenerating fields.
# --------------------------------------------------------------------------- #
SPECKLE_CACHE = STORE / "speckle_scores.json"


def _field_speckle(vals):
    from scipy.ndimage import gaussian_filter
    finite = np.isfinite(vals)
    vv = vals[finite]
    if vv.size < 64:
        return (0.0, 0.0)
    lo, hi = np.percentile(vv, [0.5, 99.5])
    span = max(hi - lo, 1e-9)
    norm = np.clip((vals - lo) / span, 0.0, 1.0)
    work = np.where(finite, norm, float(np.median(norm[finite])))
    hf = np.abs(gaussian_filter(work, 0.8) - gaussian_filter(work, 1.8))
    coarse_std = float(gaussian_filter(work, 6.0)[finite].std())
    hf_mean = float(hf[finite].mean())
    return (hf_mean, hf_mean / max(coarse_std, 1e-4))


def compute_speckle_scores(*, use_cache=True):
    """{window_id: [hf_mean, speckle_ratio]} over all windows. Cached to the store."""
    if use_cache and SPECKLE_CACHE.exists():
        return {k: tuple(v) for k, v in json.loads(SPECKLE_CACHE.read_text()).items()}
    from tools.studies import q4_stage1_labelset as H
    by_mb = {}
    for row, _ in qr.iter_windows(BATCH_ID):
        by_mb.setdefault(row["minibrot_id"], []).append(row)
    out = {}
    for mbid, wins in by_mb.items():
        field, fw, fh = H.load_field_values(mbid)
        for r in wins:
            u, v, w, h = (r["window"][k] for k in ("u", "v", "w", "h"))
            x0, y0 = int(round(u*fw)), int(round(v*fh))
            x1, y1 = int(round((u+w)*fw)), int(round((v+h)*fh))
            out[r["window_id"]] = _field_speckle(field[y0:y1, x0:x1])
    SPECKLE_CACHE.write_text(json.dumps({k: [round(a, 5), round(b, 5)]
                                         for k, (a, b) in out.items()}))
    return out


# --------------------------------------------------------------------------- #
def load_joined():
    """[(window_id, features, klass_or_None, speckle)] over all 300 windows.
    `speckle` = (hf_mean, speckle_ratio) from the field."""
    labels = json.loads(LABELS.read_text())
    spk = compute_speckle_scores()
    rows = []
    for row, _ in qr.iter_windows(BATCH_ID):
        wid = row["window_id"]
        rows.append((wid, row["features"], labels.get(wid), spk.get(wid)))
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
    labeled = [(w, f, k, s) for w, f, k, s in rows if k]
    print(f"labeled: {len(labeled)} / {len(rows)}")
    from collections import Counter
    print("  class counts:", dict(Counter(k for _, _, k, _ in labeled)))

    # accept guardrail envelope — the values v2 must NOT cross
    acc = [(f, s) for _, f, k, s in labeled if k == "accept"]
    print("\naccept envelope (hard guardrail — ceilings must clear these):")
    print(f"  max interior_frac     = {max(f['interior_frac'] for f, _ in acc):.3f}")
    print(f"  max flat_frac         = {max(f['flat_frac'] for f, _ in acc):.3f}")
    print(f"  max speckle_ratio     = {max(s[1] for _, s in acc):.3f}  "
          f"(ceiling {C_SPECKLE})")

    # per-class speckle_ratio distribution (the Bug-2 fix)
    print("\nspeckle_ratio (hf_mean/coarse_std) per class — the fixed frequency metric:")
    byc = {c: [s[1] for _, _, k, s in labeled if k == c]
           for c in ("accept", "reject", "filter_leak")}
    _dist("speckle_ratio", byc)

    # apply v2 to the labeled set -> agreement
    print(f"\nv2 ceilings (as-framed, plain): interior>={C_INTERIOR} | "
          f"speckle_ratio>={C_SPECKLE} (hf>={HF_FLOOR}) | flat>={C_FLAT}")
    catch = {c: {} for c in ("accept", "reject", "filter_leak")}
    dropped = {c: 0 for c in ("accept", "reject", "filter_leak")}
    for _, f, k, s in labeled:
        r = filter_v2(f, s)
        if r:
            dropped[k] += 1
            catch[k][r] = catch[k].get(r, 0) + 1
    n = {c: sum(1 for _, _, k, _ in labeled if k == c) for c in dropped}
    print("\nAGREEMENT on the 107:")
    for c in ("filter_leak", "reject", "accept"):
        print(f"  {c:>12}: v2 drops {dropped[c]:>2}/{n[c]:<2}  by-reason {catch[c]}")
    print(f"\n  >>> ACCEPTS DROPPED = {dropped['accept']}  "
          f"({'PASS — guardrail holds' if dropped['accept']==0 else 'FAIL — LOOSEN'})")
    leak_caught = dropped["filter_leak"] / n["filter_leak"] if n["filter_leak"] else 0
    print(f"  >>> filter_leak caught = {dropped['filter_leak']}/{n['filter_leak']} "
          f"({leak_caught:.0%})")

    # speckle ceiling's UNIQUE contribution (leaks it catches that interior/flat miss)
    only_spk = [w for w, f, k, s in labeled
                if k == "filter_leak" and f["interior_frac"] < C_INTERIOR
                and f["flat_frac"] < C_FLAT and s and s[0] >= HF_FLOOR and s[1] >= C_SPECKLE]
    print(f"  >>> speckle ceiling UNIQUELY catches (not interior/flat): {len(only_spk)} "
          f"leak {only_spk}")

    # residual leak rate on the labeled SURVIVORS (success-metric proxy).
    surv = [k for _, f, k, s in labeled if not filter_v2(f, s)]
    sc = Counter(surv)
    tot = len(surv)
    print(f"\nlabeled survivors: {tot}  {dict(sc)}")
    if tot:
        print(f"  residual filter_leak rate on survivors = {sc['filter_leak']}/{tot} "
              f"= {sc['filter_leak']/tot:.0%}  (was {n['filter_leak']}/{sum(n.values())} "
              f"= {n['filter_leak']/sum(n.values()):.0%} pre-v2)")
    return rows


def apply():
    """Apply v2 to ALL 300 windows -> the auto_filter_v2 drop set the UI excludes
    from the label queue. Dropping ALL windows (not just unlabeled) is the Bug-1 fix:
    an already-LABELED degenerate (e.g. mb04..c273 = interior 0.32) must also leave
    the queue. Human labels are untouched — they stay in scores.json; this is a
    queue-exclusion set, NOT a label mutation."""
    rows = load_joined()
    dropped, reasons = {}, {}
    surv_unlab = surv_lab = 0
    acc_dropped = []
    for w, f, k, s in rows:
        r = filter_v2(f, s)
        if r:
            dropped[w] = r
            reasons[r] = reasons.get(r, 0) + 1
            if k == "accept":
                acc_dropped.append(w)
        elif k:
            surv_lab += 1
        else:
            surv_unlab += 1
    print(f"apply v2 to all {len(rows)} windows:")
    print(f"  auto_filter_v2 dropped: {len(dropped)}  by-reason {reasons}")
    print(f"  accepts dropped: {len(acc_dropped)} {acc_dropped}  "
          f"({'PASS' if not acc_dropped else 'FAIL'})")
    print(f"  survivors: {surv_unlab} unlabeled (the label QUEUE) + {surv_lab} labeled "
          f"= {surv_unlab + surv_lab} shown")

    out = STORE / "auto_filter_v2.json"
    out.write_text(json.dumps(
        {"batch_id": BATCH_ID,
         "ceilings": {"C_INTERIOR": C_INTERIOR, "C_SPECKLE": C_SPECKLE,
                      "HF_FLOOR": HF_FLOOR, "C_FLAT": C_FLAT},
         "scope": "all-300 (queue-exclusion; human labels untouched in scores.json)",
         "dropped": dropped,
         "n_dropped": len(dropped),
         "n_queue": surv_unlab, "n_shown": surv_unlab + surv_lab}, indent=0))
    print(f"  -> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if stage == "analyze":
        analyze()
    elif stage == "apply":
        apply()
    else:
        print("usage: q4_stage1_filter_v2 [analyze|apply]")
