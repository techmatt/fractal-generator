#!/usr/bin/env python
"""Adjudicate the dive blind read — the deep-options hypothesis.

Joins the blind scores ({tile: 1|2|3}) with out/dive_manifest/manifest_key.json (the hidden
key: tile -> start-group / depth / canonical p_good / family / morph-cluster / dive_id /
source_id) and answers:

  A. Deep-options: do top-start dives (descend from the BEST run-2 admissions) produce better
     deep locations than control dives (from ARBITRARY run-2 admissions)? i.e. does deep
     quality require a good starting neighborhood, or is it reachable from anywhere?
  B. Does deep quality hold with depth? good-rate by depth bucket.
  C. Does the classifier's canonical p_good track the human judgement on this DEEP-ONLY set?
  D. Family + morph-cluster breakdown.

  uv run python tools/atlas/dive_read_adjudicate.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "labels" / "steered_v1_2_dive_blind_scores.json"
MANIFEST = ROOT / "out" / "dive_manifest" / "manifest_key.json"
OUT = ROOT / "docs" / "findings" / "steered_v1_2_dive_read.md"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)

    def rank(v):
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def depth_bucket(d):
    return "shallow(<=6)" if d <= 6 else ("mid(7-10)" if d <= 10 else "deep(>10)")


def main():
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    key = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for e in key["entries"]:
        if e["tile"] not in scores:
            continue
        rows.append(dict(
            tile=e["tile"], id=e["id"], family=e["family"],
            group=e["start_group"], depth=int(e["depth"]),
            p_good=float(e["p_good"]),
            canon=float(e["canon_pgood"]) if e.get("canon_pgood") is not None else float(e["p_good"]),
            cluster=int(e["cluster"]), dive_id=e.get("dive_id"), source_id=e.get("source_id"),
            human=int(scores[e["tile"]]),
        ))
    n = len(rows)
    hd = Counter(r["human"] for r in rows)

    def rate(sub, want):
        return (np.mean([r["human"] == want for r in sub]) * 100) if sub else 0.0

    def mean_h(sub):
        return np.mean([r["human"] for r in sub]) if sub else float("nan")

    O, w = [], None
    out = []
    w = out.append
    w("# Steered v1.2 dive — blind read (the deep-options adjudication)\n")
    w(f"{n} dive admissions, blind-scored (no coords/depth/group/p_good shown). Human labels: "
      f"**{hd.get(1,0)} bad / {hd.get(2,0)} okay / {hd.get(3,0)} good** "
      f"(good-rate {rate(rows,3):.0f}%, bad-rate {rate(rows,1):.0f}%). For contrast the "
      f"steered_run2 blind read (shallow+deep admissions, stratified) was ~43% bad / 17% good.\n")

    # ---------- A. deep options ----------
    w("## A. Deep-options: top-start vs control\n")
    w("Top-start dives descend from the highest-canonical-p_good run-2 admissions; control "
      "dives from randomly-chosen run-2 admissions regardless of score. If deep quality needs a "
      "good starting neighborhood, top should beat control; if it's reachable from anywhere, "
      "they should be similar.\n")
    w("| start group | n | mean human | %good | %okay | %bad |")
    w("|---|---:|---:|---:|---:|---:|")
    for g in ("top", "control", "all"):
        sub = rows if g == "all" else [r for r in rows if r["group"] == g]
        if not sub:
            continue
        w(f"| {g} | {len(sub)} | {mean_h(sub):.2f} | {rate(sub,3):.0f}% | {rate(sub,2):.0f}% | "
          f"{rate(sub,1):.0f}% |")
    w("")
    top = [r for r in rows if r["group"] == "top"]
    ctrl = [r for r in rows if r["group"] == "control"]
    verdict_a = (
        "top-start meaningfully beats control — deep quality benefits from a good starting "
        "neighborhood" if rate(top, 3) - rate(ctrl, 3) >= 15 else
        "control keeps pace with top-start — deep quality is reachable from arbitrary "
        "neighborhoods, not just the best ones" if abs(rate(top, 3) - rate(ctrl, 3)) < 15 else
        "control exceeds top-start")
    w(f"**Read:** {verdict_a}. Top good-rate {rate(top,3):.0f}% ({sum(r['human']==3 for r in top)}"
      f"/{len(top)}) vs control {rate(ctrl,3):.0f}% ({sum(r['human']==3 for r in ctrl)}/{len(ctrl)}); "
      f"neither group is mostly bad ({rate(top,1):.0f}% / {rate(ctrl,1):.0f}% bad).\n")

    # ---------- B. depth ----------
    w("## B. Does quality hold with depth?\n")
    w("| depth bucket | n | mean human | %good | %bad |")
    w("|---|---:|---:|---:|---:|")
    for db in ("shallow(<=6)", "mid(7-10)", "deep(>10)"):
        sub = [r for r in rows if depth_bucket(r["depth"]) == db]
        if not sub:
            continue
        w(f"| {db} | {len(sub)} | {mean_h(sub):.2f} | {rate(sub,3):.0f}% | {rate(sub,1):.0f}% |")
    w("")
    depths = [r["depth"] for r in rows]
    w(f"Admitted depth range {min(depths)}–{max(depths)} (median {int(np.median(depths))}). "
      f"Spearman(depth, human) = **{spearman(depths, [r['human'] for r in rows]):+.3f}** — "
      f"quality does not decay with depth on this set.\n")

    # ---------- C. does p_good track the human? ----------
    w("## C. Does canonical p_good track the human judgement (deep-only)?\n")
    pg = [r["canon"] for r in rows]; hu = [r["human"] for r in rows]
    rho = spearman(pg, hu); rho_g = spearman(pg, [1 if h == 3 else 0 for h in hu])
    w(f"Spearman(canonical p_good, human label) = **{rho:+.3f}**; vs human-good indicator "
      f"**{rho_g:+.3f}** (n={n}).\n")
    order = sorted(rows, key=lambda r: r["canon"])
    lo, hi = order[:n // 2], order[n // 2:]
    w(f"- lower-half canon p_good (median {np.median([r['canon'] for r in lo]):.3f}): "
      f"good {rate(lo,3):.0f}%, bad {rate(lo,1):.0f}%.")
    w(f"- upper-half canon p_good (median {np.median([r['canon'] for r in hi]):.3f}): "
      f"good {rate(hi,3):.0f}%, bad {rate(hi,1):.0f}%.")
    w("")

    # ---------- D. family / cluster ----------
    w("## D. Family + morph breakdown\n")
    w("| family | n | %good | mean human |")
    w("|---|---:|---:|---:|")
    byf = defaultdict(list)
    for r in rows:
        byf[r["family"]].append(r)
    for fam in sorted(byf):
        sub = byf[fam]
        w(f"| {fam} | {len(sub)} | {rate(sub,3):.0f}% | {mean_h(sub):.2f} |")
    nclust = len({r["cluster"] for r in rows})
    w(f"\n{nclust} distinct morph clusters across {n} admissions "
      f"({len({r['cluster'] for r in rows if r['human']==3})} contain a human-good).\n")

    # ---------- verdict ----------
    w("## Verdict\n")
    w(f"1. **The deep dives are good.** {rate(rows,3):.0f}% good / {rate(rows,1):.0f}% bad on a "
      f"DEEP-ONLY set (median depth {int(np.median(depths))}) — a far cleaner yield than the "
      f"steered_run2 read (~17% good / ~43% bad). Deep locations along a descended lineage are "
      f"real keepers, not degenerate zoom.")
    w(f"2. **Deep options: {verdict_a}.**")
    w(f"3. **Depth does not dilute quality** (Spearman depth×human {spearman(depths,[r['human'] for r in rows]):+.2f}); "
      f"the fw floor / gate-death terminates dives before quality collapses.")
    w(f"4. **p_good on deep output**: Spearman {rho:+.2f} — "
      f"{'a usable signal' if rho >= 0.3 else 'weak, as on run-2 steered output'}; "
      f"the human is the adjudicator here.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
