#!/usr/bin/env python
r"""Mini-readout for a julia-dup-fix steered breadth run — the verification report for the
seed-c-aware dup metric + hook spacing + pre-canonical filter + freshness prior package
(docs/findings/julia_dup_metric_audit.md). Reads only the durable run artifacts (summary.json,
harvest_log.jsonl, outcome_ledger.jsonl, julia_hooks.jsonl); no render, no GPU.

Reports, for the julia dup-fix acceptance:
  * julia admissions per partition + their reached-depth distribution (shallow / center-descent
    views should now appear — they were the over-killed population);
  * julia checks/admit this run vs campaign 1 (the buggy-metric baseline);
  * canonical confirmation renders SAVED by the pre-canonical coord-dup filter (precanon_dup);
  * julia-hook decisions: fired vs spacing-skipped.

  uv run python tools/atlas/julia_fix_readout.py --run out/atlas/julia_fix_smoke/breadth
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Campaign-1 breadth+dive julia checks/admit under the BUGGY z-only metric (recomputed
# 2026-07-19; the baseline this fix is measured against). checks, admits.
CAMPAIGN1_JULIA = {
    "julia:mandelbrot": (5435, 2),
    "julia:multibrot3": (4116, 23),
    "julia:multibrot4": (4050, 12),
    "julia:multibrot5": (8389, 9),
}
JULIA_PARTS = list(CAMPAIGN1_JULIA)


def load_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build(run: Path) -> str:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) \
        if (run / "summary.json").exists() else {}
    rows = load_jsonl(run / "outcome_ledger.jsonl")
    harvest = load_jsonl(run / "harvest_log.jsonl")
    hooks = load_jsonl(run / "julia_hooks.jsonl")
    tot = summary.get("totals", {})

    L = []
    w = L.append
    w(f"# Julia dup-fix mini-readout — `{run.name}`\n")
    am = summary.get("active_min", "?")
    w(f"_Active **{am} min**, {summary.get('batches','?')} batches. julia_hook_spacing="
      f"{summary.get('julia_hook_spacing')}, freshness_prior={summary.get('freshness_prior')} "
      f"(prior_rows={summary.get('prior_rows')})._\n")

    # --- admissions this run (from the ledger: distinct==True q3 rows) ---
    adm = [r for r in rows if r.get("distinct") and r.get("decoded_class") == 3]
    adm_by_part = Counter(r.get("family", "mandelbrot") for r in adm)
    w(f"**Admissions (distinct q3): {len(adm)}** — "
      + ", ".join(f"{p} {adm_by_part.get(p,0)}" for p in
                  sorted(adm_by_part, key=lambda k: (-adm_by_part[k], k))) + "\n")

    # --- 1. julia admissions + reached-depth distribution (the over-kill tell) ---
    w("## 1. Julia admissions & reached-depth distribution\n")
    w("Shallow (depth ≤ 3) / center-descent julia views were the population the z-only metric "
      "over-killed; they should now appear.\n")
    w("| partition | admits | depth min | median | max | depth histogram |")
    w("|---|--:|--:|--:|--:|---|")
    jadm = [r for r in adm if r.get("family", "").startswith("julia:")]
    for part in JULIA_PARTS:
        ds = sorted(int(r["reached_depth"]) for r in jadm if r.get("family") == part)
        if not ds:
            w(f"| {part} | 0 | – | – | – | – |")
            continue
        med = ds[len(ds) // 2]
        hist = Counter(ds)
        hs = " ".join(f"d{d}:{hist[d]}" for d in sorted(hist))
        w(f"| {part} | {len(ds)} | {ds[0]} | {med} | {ds[-1]} | {hs} |")
    n_shallow = sum(1 for r in jadm if int(r["reached_depth"]) <= 3)
    w(f"\n_Julia admissions total **{len(jadm)}**, of which **{n_shallow}** are shallow "
      f"(depth ≤ 3)._\n")

    # --- 2. julia checks/admit vs campaign 1 ---
    w("## 2. Julia checks/admit vs campaign 1 (buggy-metric baseline)\n")
    checks = defaultdict(int); admits = defaultdict(int)
    for h in harvest:
        p = h["partition"]
        checks[p] += 1
        if h.get("admitted"):
            admits[p] += 1
    w("| partition | checks | admit | checks/admit | campaign1 checks/admit |")
    w("|---|--:|--:|--:|--:|")
    for part in JULIA_PARTS:
        c, a = checks.get(part, 0), admits.get(part, 0)
        cpa = f"{c/a:.1f}" if a else ("∞" if c else "–")
        c1c, c1a = CAMPAIGN1_JULIA[part]
        c1 = f"{c1c/c1a:.1f}" if c1a else "∞"
        w(f"| {part} | {c} | {a} | {cpa} | {c1} |")
    w("\n_Lower checks/admit = less dup-churn per julia admission than campaign 1._\n")

    # --- 3. pre-canonical filter: renders saved ---
    w("## 3. Canonical renders saved by the pre-canonical coord-dup filter\n")
    precanon = [h for h in harvest if h.get("precanon_dup") is not None]
    precanon_j = sum(1 for h in precanon if h["partition"].startswith("julia:"))
    w(f"- **precanon_dup (renders skipped before the confirmation render): "
      f"{tot.get('precanon_dup', len(precanon))}** "
      f"(julia {precanon_j}, c-plane {len(precanon) - precanon_j}).\n")
    by_part = Counter(h["partition"] for h in precanon)
    if by_part:
        w("  per partition: " + ", ".join(f"{p} {by_part[p]}" for p in sorted(by_part)) + "\n")

    # --- 4. hook decisions ---
    w("## 4. Julia-hook decisions (fire vs spacing-skip)\n")
    fired = sum(1 for h in hooks if h.get("hooked"))
    skipped = sum(1 for h in hooks if not h.get("hooked"))
    w(f"- Hooks fired: **{fired}** (summary julia_roots={tot.get('julia_roots')}); "
      f"spacing-skipped: **{skipped}** (summary julia_hooks_skipped={tot.get('julia_hooks_skipped')}).")
    w(f"- Durable hook log: {len(hooks)} decisions recorded (every hooked seed c is now "
      f"recoverable, incl. zero-admit roots).\n")
    hk_by_part = Counter(h["jpart"] for h in hooks)
    if hk_by_part:
        w("  per partition (decisions): " +
          ", ".join(f"{p} {hk_by_part[p]}" for p in sorted(hk_by_part)) + "\n")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="steered breadth run dir")
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout only)")
    # Standing habit (part-2 finding): every readout drops a fate-stratified visual sample of
    # admissions AND rejects beside the numbers, so a wrong-but-plausible reject is eyeballable.
    ap.add_argument("--no-visual-sample", action="store_true",
                    help="skip the reject-autopsy contact sheet (default: render it)")
    ap.add_argument("--visual-n", type=int, default=10, help="tiles per fate on the sheet")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    md = build(run)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"wrote {out}\n")
    print(md)
    if not args.no_visual_sample:
        from reject_autopsy import render_autopsy_sheet
        res = render_autopsy_sheet(run, run / "reject_autopsy.png", n_per_fate=args.visual_n)
        print(f"\n[visual-sample] {run.name}/reject_autopsy.png  "
              f"rendered={res['rendered']} counts={res['counts']}"
              + (f"  ({res['coordless_harvest']} coordless harvest rows skipped)"
                 if res['coordless_harvest'] else ""))


if __name__ == "__main__":
    main()
