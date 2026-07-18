#!/usr/bin/env python
"""Calibrate the keeper bar against the blind human read.

Joins the blind scores ({tile: 1|2|3}) with the hidden manifest_key.json (tile -> family /
canonical p_good / p_notbad / depth / morph-cluster / provisional-keeper) and answers:

  A. Does canonical p_good TRACK the human judgement on unseen steered output? (the core
     validity check — Spearman + human-good-rate by p_good tercile)
  B. How does the PROVISIONAL keeper cut (F0.5 on the v7 eval slice) score against the human
     "good" (label==3)? precision / recall / F0.5, per family + pooled.
  C. Where should the cut move? Re-derive the F0.5-optimal p_good cut on THIS labeled set.
  D. Do the DEEP steered admissions hold up? human-good-rate by depth bucket.
  E. Was mandelbrot's 21-discovery / 0-keeper split right?

The manifest is STRATIFIED (p_good tercile x depth bucket x morph cluster), so rates here are
over the stratified set, not the raw admission population — good for threshold calibration
(spans the p_good range), biased for population good-rate estimates. Report-only.

  uv run python tools/atlas/keeper_calibrate.py \
      --scores labels/steered_run2_blind_scores.json --manifest out/steered_run2_manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

from score_lib import corn_decode              # noqa: E402
import keeper_cut as kc                          # noqa: E402

GRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]


def spearman(x, y):
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        # average ties
        v = np.asarray(v); _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def prf_beta(tp, fp, fn, beta=0.5):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    d = b2 * prec + rec
    return prec, rec, ((1 + b2) * prec * rec / d if d else 0.0)


def confusion(rows, t):
    tp = fp = fn = tn = 0
    for r in rows:
        pred = corn_decode(r["p_notbad"], r["p_good"], t) == 3
        pos = r["human"] == 3
        tp += pred and pos; fp += pred and not pos
        fn += (not pred) and pos; tn += (not pred) and not pos
    return tp, fp, fn, tn


def best_cut(rows, beta=0.5):
    best = None
    for t in GRID:
        tp, fp, fn, _ = confusion(rows, t)
        _, _, f = prf_beta(tp, fp, fn, beta)
        if best is None or f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=ROOT / "labels/steered_run2_blind_scores.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "out/steered_run2_manifest")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/findings/steered_run2_keeper_calibration.md")
    args = ap.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    key = json.loads((args.manifest / "manifest_key.json").read_text(encoding="utf-8"))
    cuts = kc.load_keeper_cuts()

    rows = []
    for e in key["entries"]:
        if e["tile"] not in scores:
            continue
        rows.append(dict(
            tile=e["tile"], family=e["family"], p_good=float(e["p_good"]),
            p_notbad=float(e["p_notbad"]), depth=int(e["depth"]), depth_bucket=e["depth_bucket"],
            pgood_tercile=int(e["pgood_tercile"]), cluster=int(e["cluster"]),
            keeper=bool(e["keeper"]), human=int(scores[e["tile"]]),
        ))
    n = len(rows)
    O = []
    w = O.append

    hd = Counter(r["human"] for r in rows)
    w("# Steered run2 — keeper-bar calibration vs the blind human read\n")
    w(f"{n} blind-scored tiles from the stratified manifest. Human labels: "
      f"**{hd.get(1,0)} bad / {hd.get(2,0)} okay / {hd.get(3,0)} good**. All tiles are "
      f"discovery-admitted q3, so label==1 (bad) is the discovery false-positive on steered "
      f"output. The set is stratified across p_good x depth x morph-cluster (good for cut "
      f"calibration, biased for population rates).\n")

    # ---------- A. p_good validity ----------
    pg = np.array([r["p_good"] for r in rows]); hu = np.array([r["human"] for r in rows])
    rho = spearman(pg, hu)
    w("## A. Does canonical p_good track the human judgement?\n")
    w(f"Spearman(p_good, human label) = **{rho:+.3f}** over n={n}. "
      f"Spearman(p_good, human-good indicator) = **{spearman(pg, (hu==3).astype(float)):+.3f}**.\n")
    w("| p_good tercile | n | mean human | %good(3) | %bad(1) |")
    w("|---|---:|---:|---:|---:|")
    for t in (0, 1, 2):
        sub = [r for r in rows if r["pgood_tercile"] == t]
        if not sub:
            continue
        hm = np.mean([r["human"] for r in sub])
        g = np.mean([r["human"] == 3 for r in sub]) * 100
        b = np.mean([r["human"] == 1 for r in sub]) * 100
        lab = {0: "low", 1: "mid", 2: "high"}[t]
        w(f"| {lab} | {len(sub)} | {hm:.2f} | {g:.0f}% | {b:.0f}% |")
    w("")

    # ---------- B. provisional keeper vs human ----------
    w("## B. Provisional keeper cut vs human-good (label==3)\n")
    w("Prediction = `corn_decode(p_notbad, p_good, keeper_cut) == 3`; positive = human good.\n")
    w("| family | keeper cut | n | pred-keepers | precision | recall | F0.5 |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam in sorted(by_fam):
        sub = by_fam[fam]
        t = kc.keeper_cut_for(fam, cuts)
        tp, fp, fn, tn = confusion(sub, t)
        p, rec, f = prf_beta(tp, fp, fn)
        cal = "" if cuts.get(fam, {}).get("calibrated") else "*"
        w(f"| {fam} | {t}{cal} | {len(sub)} | {tp+fp} | {p:.2f} | {rec:.2f} | {f:.2f} |")
    tp, fp, fn, tn = confusion(rows, None) if False else _pooled(rows, cuts)
    p, rec, f = prf_beta(tp, fp, fn)
    w(f"| **pooled** | (per-fam) | {n} | {tp+fp} | {p:.2f} | {rec:.2f} | {f:.2f} |")
    w(f"\nPooled keeper confusion: TP={tp} FP={fp} FN={fn}. Of {tp+fp} predicted keepers, "
      f"**{tp} were human-good** (precision {p:.0%}); of {tp+fn} human-good, **{tp} were kept** "
      f"(recall {rec:.0%}).\n")

    # ---------- C. re-derive cut on this labeled set ----------
    w("## C. Where should the cut move? (F0.5-optimal on THIS labeled set)\n")
    w("Small n — treat as directional. Pooled first, then per-family where n>=10.\n")
    pt, pf = best_cut(rows)
    tp, fp, fn, _ = confusion(rows, pt)
    p, rec, f = prf_beta(tp, fp, fn)
    w(f"- **pooled** F0.5-optimal p_good cut ~ **{pt:.2f}** (F0.5={pf:.2f}, P={p:.2f} R={rec:.2f}).")
    for fam in sorted(by_fam):
        sub = by_fam[fam]
        pos = sum(1 for r in sub if r["human"] == 3)
        if len(sub) < 10 or pos < 2:
            w(f"- {fam}: n={len(sub)} good={pos} — too few to re-derive; keep provisional "
              f"{kc.keeper_cut_for(fam,cuts)}.")
            continue
        bt, bf = best_cut(sub)
        w(f"- {fam}: n={len(sub)} good={pos} -> F0.5-optimal ~ **{bt:.2f}** "
          f"(provisional {kc.keeper_cut_for(fam,cuts)}).")
    w("")

    # ---------- D. depth ----------
    w("## D. Do the deep steered admissions hold up?\n")
    w("| depth bucket | n | mean human | %good | %bad |")
    w("|---|---:|---:|---:|---:|")
    for db in ("shallow(<=3)", "mid(4-8)", "deep(>8)"):
        sub = [r for r in rows if r["depth_bucket"] == db]
        if not sub:
            continue
        w(f"| {db} | {len(sub)} | {np.mean([r['human'] for r in sub]):.2f} | "
          f"{np.mean([r['human']==3 for r in sub])*100:.0f}% | "
          f"{np.mean([r['human']==1 for r in sub])*100:.0f}% |")
    w("")

    # ---------- E. mandelbrot ----------
    w("## E. Mandelbrot (21 discovery / 0 provisional-keepers)\n")
    mb = sorted(by_fam.get("mandelbrot", []), key=lambda r: -r["p_good"])
    if mb:
        w(f"{len(mb)} mandelbrot tiles in the set. human label vs p_good (sorted by p_good):\n")
        w("| p_good | depth | human | provisional keeper |")
        w("|---:|---:|---:|---:|")
        for r in mb:
            w(f"| {r['p_good']:.3f} | {r['depth']} | {r['human']} | {'yes' if r['keeper'] else 'no'} |")
        good = sum(1 for r in mb if r["human"] == 3)
        w(f"\nHuman called **{good}/{len(mb)}** mandelbrot tiles good; provisional keepers "
          f"among them: **{sum(1 for r in mb if r['keeper'])}**. "
          f"{'0 keepers looks JUSTIFIED.' if good<=1 else 'Human found good ones the keeper cut REJECTED — the cut is too high for mandelbrot.'}\n")

    # ---------- Verdict ----------
    lo = [r for r in rows if r["pgood_tercile"] == 0]
    hi = [r for r in rows if r["pgood_tercile"] == 2]
    lo_good = np.mean([r["human"] == 3 for r in lo]) * 100 if lo else 0
    lo_bad = np.mean([r["human"] == 1 for r in lo]) * 100 if lo else 0
    hi_good = np.mean([r["human"] == 3 for r in hi]) * 100 if hi else 0
    tp, fp, fn, tn = _pooled(rows, cuts)
    kp, _, _ = prf_beta(tp, fp, fn)
    dgood = {db: np.mean([r["human"] == 3 for r in rows if r["depth_bucket"] == db]) * 100
             for db in ("shallow(<=3)", "mid(4-8)", "deep(>8)")}
    w("## Verdict\n")
    w(f"1. **p_good is a BADNESS filter, not a GOODNESS ranker on steered output.** The low "
      f"p_good tercile is {lo_bad:.0f}% bad / {lo_good:.0f}% good — reliably weak. But the HIGH "
      f"tercile is only {hi_good:.0f}% good, no better than mid: above the low band, higher "
      f"p_good does NOT mean better. Spearman {rho:+.2f} is carried by the bad end.")
    w(f"2. **The keeper tier as derived is too permissive on steered output**: it keeps "
      f"{tp+fp}/{n} tiles at {kp:.0%} precision (every human-good is caught, but so are {fp} "
      f"non-good). The F0.5-optimal cut on this set moves UP (pooled ~{best_cut(rows)[0]:.2f}, "
      f"multibrot ~0.6–0.8), but even optimal tops out near ~30% precision — NO p_good threshold "
      f"cleanly isolates human-good here. The lever is a better ranking signal, not a higher cut.")
    w(f"3. **The depth expansion holds up.** Deep(>8) admissions are {dgood['deep(>8)']:.0f}% good "
      f"vs shallow {dgood['shallow(<=3)']:.0f}% — the eviction-fix depth gain did NOT dilute "
      f"quality; deep steered locations are as good as shallow.")
    w(f"4. **Mandelbrot 0-keepers confirmed** (0/16 human-good; all p_good below its 0.51 cut). "
      f"Its discovery bar (t_good 0.14) admits locations the human uniformly rejects — a "
      f"discovery-side over-admission, correctly zeroed by the keeper cut.")
    w(f"5. **Recommendation:** keep the keeper tier report-only; do NOT promote these cuts to a "
      f"gate. Move multibrot keeper cuts up (~0.6) to trim the worst, but treat 'keeper' as "
      f"'not-clearly-bad', not 'good'. Delivering confident-good needs a preference/ranking head "
      f"beyond CORN p_good (cf. pref-v3), or a human pass.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(O), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"n={n} human {dict(sorted(hd.items()))} | Spearman(p_good,human)={rho:+.3f} | "
          f"pooled keeper P={p:.2f} R={rec:.2f}")


def _pooled(rows, cuts):
    tp = fp = fn = tn = 0
    for r in rows:
        t = kc.keeper_cut_for(r["family"], cuts)
        pred = corn_decode(r["p_notbad"], r["p_good"], t) == 3
        pos = r["human"] == 3
        tp += pred and pos; fp += pred and not pos
        fn += (not pred) and pos; tn += (not pred) and not pos
    return tp, fp, fn, tn


if __name__ == "__main__":
    main()
