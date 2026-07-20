#!/usr/bin/env python
r"""campaign2_readout.py — scheduler + freshness-prior campaign readout (extends campaign 1).

Campaign 2 is the first full-scale trial of the family-level deficit SCHEDULER (`--scheduler`)
and the coordinate FRESHNESS PRIOR (`--freshness-prior`) together. The base scheduling numbers
(admissions/hr, per-family cost, distinct-look count, library overlap, zero-admission flags) are
reused verbatim from `campaign1_readout.build`; this module adds the seven campaign-2-specific
dimensions the launch spec requires, all from the run's DURABLE artifacts:

  A. Distinct-look SHARES per partition vs target marginals over time + the allocation trace
     (scheduler_trace.jsonl: per-batch chosen partition, deficits, prices, look counts).
  B. Final LEARNED prices per partition (the re-priced table — the new price seed for campaign 3;
     campaign-1 julia prices are void). From summary.json / state.json scheduler block.
  C. FRESHNESS-PRIOR effect: throughput vs the 21.8 adm/hr campaign-1 context, precanon/dup
     savings attributable to the prior, coord overlap vs the prior places.
  D. Hook spacing at scale (julia_hooks.jsonl): decisions fired/skipped, skip fraction
     (smoke was 4/16 = 0.25 — verify 0.20 isn't over-thinning at scale).
  E. Julia DEPTH distribution of admissions (the suppressed shallow center-descent population).
  F. sat_frac trajectory (saturation.jsonl): novelty-memory saturation over the run.
  G. Visual samples — delegated to campaign1_contact_sheet.py (pointer only).

Nothing is reimplemented — campaign1_readout is imported and its loaders/sections reused.

  uv run python tools/atlas/campaign2_readout.py --breadth data/discovery/campaign2/breadth
  uv run python tools/atlas/campaign2_readout.py --breadth data/discovery/campaign2/breadth --no-morph
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "atlas"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import campaign1_readout as c1                          # noqa: E402  (loaders + base sections)
import deficit_scheduler as dsched                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Campaign-1 reference context (spec-supplied; do NOT recompute — they anchor the comparison).
C1_ADM_PER_HR = 21.8            # campaign-1 breadth throughput
C1_PRIOR_PLACES = 507          # campaign-1 prior-coord overlap denominator context
SMOKE_SKIP_FRAC = 0.25         # julia-hook smoke skip fraction (4/16) — watch for over-thinning
PARTITIONS = list(c1.C_FAMILIES) + [dsched.julia_partition(f) for f in c1.C_FAMILIES]


def load_scheduler_summary(run_dir: Path) -> dict:
    """The scheduler block from summary.json (finished) or state.json (mid-run)."""
    for name in ("summary.json", "state.json"):
        p = run_dir / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "scheduler" in d:
            return d["scheduler"]
    return {}


def load_totals(run_dir: Path) -> dict:
    for name in ("summary.json", "state.json"):
        p = run_dir / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "totals" in d:
            return d["totals"]
    return {}


# --------------------------------------------------------------------------- #
# A. Allocation trace + distinct-look shares over time.
# --------------------------------------------------------------------------- #
def scheduler_section(run_dir: Path, sched: dict) -> list:
    w = [].append
    L: list = []
    w = L.append
    trace = c1.load_jsonl(run_dir / "scheduler_trace.jsonl")
    w("## A. Scheduler allocation — distinct-look shares vs target over time\n")
    if not trace:
        w("_No scheduler_trace.jsonl yet (scheduler off, or no batch has been served)._\n")
        return L

    target = sched.get("target_frac") or (trace[-1].get("deficits") and None)
    # allocation trace: how many batches each partition was actually served.
    served = Counter(t["chosen"] for t in trace if t.get("chosen"))
    n_served = sum(served.values())
    w(f"Batches traced: **{len(trace)}**; served (non-null pop): **{n_served}**. "
      "Served-partition histogram (the scheduler's realized cross-partition allocation):\n")
    w("| partition | batches served | share | target |")
    w("|---|--:|--:|--:|")
    for p in PARTITIONS:
        tf = (target or {}).get(p)
        tfs = f"{tf:.1%}" if tf is not None else "—"
        share = served.get(p, 0) / n_served if n_served else 0.0
        w(f"| {p} | {served.get(p,0)} | {share:.1%} | {tfs} |")

    # distinct-look shares over time: sample the trace at a few checkpoints. `looks` is the tally
    # count per partition INCLUSIVE of the 523-look library seed; the run-DELTA (looks now minus the
    # seed baseline) is this run's own accepted distinct looks, which is what the scheduler steered.
    first = trace[0].get("looks", {})
    seed_base = {p: int(first.get(p, 0)) for p in PARTITIONS}   # ~ the library seed per partition
    def shares(looks, base=None):
        if base is not None:
            looks = {p: max(0, int(looks.get(p, 0)) - base.get(p, 0)) for p in PARTITIONS}
        tot = sum(int(looks.get(p, 0)) for p in PARTITIONS)
        return {p: (int(looks.get(p, 0)) / tot if tot else 0.0) for p in PARTITIONS}, tot

    marks = [trace[i] for i in sorted(set([0, len(trace)//4, len(trace)//2,
                                           3*len(trace)//4, len(trace)-1]))]
    w("\nRun-only distinct-look shares (library seed subtracted) at trace checkpoints — the "
      "acceptance the scheduler produced, converging toward target:\n")
    hdr = "| batch | run looks | " + " | ".join(PARTITIONS) + " |"
    w(hdr)
    w("|--:|--:|" + "|".join(["--:"] * len(PARTITIONS)) + "|")
    for t in marks:
        sh, tot = shares(t.get("looks", {}), seed_base)
        w(f"| {t['batch']} | {tot} | " + " | ".join(f"{sh[p]:.0%}" for p in PARTITIONS) + " |")
    if target:
        w("| target | — | " + " | ".join(f"{target.get(p,0):.0%}" for p in PARTITIONS) + " |")

    # deficit convergence: |deficit| trajectory (toward 0 == target met).
    def absdef(t):
        d = t.get("deficits", {})
        return sum(abs(float(d.get(p, 0.0))) for p in PARTITIONS)
    w(f"\nΣ|deficit| trajectory (0 == every partition at target): "
      f"start **{absdef(trace[0]):.3f}** → latest **{absdef(trace[-1]):.3f}** "
      f"over {len(trace)} batches.\n")
    return L


# --------------------------------------------------------------------------- #
# B. Final learned prices.
# --------------------------------------------------------------------------- #
def prices_section(sched: dict) -> list:
    L: list = []
    w = L.append
    w("## B. Final learned prices (the re-priced table for campaign 3)\n")
    prices = sched.get("prices", {})
    if not prices:
        w("_No scheduler price table yet._\n")
        return L
    seed = dsched.SEED_PRICE_MIN
    try:
        seedcfg = json.loads((ROOT / "data" / "atlas" / "scheduler_prices.json").read_text("utf-8"))
        seeds = seedcfg.get("prices", {})
    except Exception:
        seeds = {}
    min_spent = sched.get("min_spent", {})
    looks = sched.get("looks", {})
    w("Price = active-minutes per DISTINCT LOOK (online EMA). **Campaign-1 julia prices are void; "
      "this table replaces them.**\n")
    w("| partition | seed price | learned price | active-min spent | looks (incl. seed) |")
    w("|---|--:|--:|--:|--:|")
    for p in PARTITIONS:
        lp = prices.get(p)
        if lp is None:
            continue
        sp = seeds.get(p, seed)
        w(f"| {p} | {sp:.2f} | **{lp:.2f}** | {min_spent.get(p,0):.1f} | {looks.get(p,0)} |")
    w(f"\n_Capped partitions: {sched.get('capped') or 'none'}._\n")
    return L


# --------------------------------------------------------------------------- #
# C. Freshness-prior effect.
# --------------------------------------------------------------------------- #
def freshness_section(run_dir: Path, totals: dict, all_adm: list, active_min: float) -> list:
    L: list = []
    w = L.append
    w("## C. Freshness-prior effect\n")
    rate = len(all_adm) / (active_min / 60.0) if active_min else 0.0
    w(f"- **Throughput: {rate:.1f} adm/hr** vs campaign-1 context **{C1_ADM_PER_HR} adm/hr** "
      f"({len(all_adm)} admitted / {active_min/60:.2f} active-h). "
      f"Δ **{rate - C1_ADM_PER_HR:+.1f} adm/hr**.")
    # precanon/dup savings attributable to the prior (prior seeds the dedup clouds -> more early
    # rejects, fewer wasted canonical renders). Counters are cumulative in totals.
    pc = totals.get("precanon_dup", 0)
    q3d = totals.get("q3_dup", 0)
    hc = totals.get("harvest_checks", 0)
    w(f"- **Pre-canonical dup skips (renders saved): {pc}** "
      f"({pc/hc:.1%} of {hc} harvest checks)." if hc else f"- Pre-canonical dup skips: {pc}.")
    w(f"- q3_dup (canonical-render-then-coord-dup): {q3d}.")
    # coord overlap vs prior places (the freshness prior's target: don't re-cover prior coords).
    prior_ledgers = [p for p in sorted((ROOT / "data").rglob("outcome_ledger.jsonl"))
                     if run_dir.resolve() not in p.resolve().parents]
    priors = c1.prior_clouds(prior_ledgers, PARTITIONS)
    n_prior = sum(len(v) for v in priors.values())
    ch, ct, cpf = c1.coord_overlap(all_adm, priors)
    if ct:
        w(f"- **Coord overlap vs {n_prior} prior places (ctx {C1_PRIOR_PLACES}): "
          f"{ch}/{ct} = {ch/ct:.1%}** of campaign-2 admissions fall inside a prior admission's "
          f"coord-dup radius. With the prior ON this should be LOW (the prior actively steers off "
          f"prior coords).")
    return L


# --------------------------------------------------------------------------- #
# D. Hook spacing at scale.
# --------------------------------------------------------------------------- #
def hook_section(run_dir: Path) -> list:
    L: list = []
    w = L.append
    w("## D. Julia hook spacing at scale\n")
    hooks = c1.load_jsonl(run_dir / "julia_hooks.jsonl")
    if not hooks:
        w("_No julia_hooks.jsonl yet._\n")
        return L
    fired = sum(1 for h in hooks if h.get("hooked"))
    skipped = sum(1 for h in hooks if not h.get("hooked"))
    tot = fired + skipped
    frac = skipped / tot if tot else 0.0
    w(f"- Hook decisions: **{tot}** ({fired} fired, {skipped} skipped-by-spacing). "
      f"**Skip fraction {frac:.2f}** vs smoke {SMOKE_SKIP_FRAC:.2f}.")
    verdict = ("over-thinning (skip fraction well above the smoke's 0.25 — spacing may be too wide)"
               if frac > 0.40 else "consistent with the smoke — not over-thinning")
    w(f"  - Verdict: **{verdict}**.")
    per = defaultdict(lambda: [0, 0])
    for h in hooks:
        per[h["jpart"]][0 if h.get("hooked") else 1] += 1
    w("\n| julia partition | fired | skipped |")
    w("|---|--:|--:|")
    for p in sorted(per):
        w(f"| {p} | {per[p][0]} | {per[p][1]} |")
    return L


# --------------------------------------------------------------------------- #
# E. Julia admission depth distribution.
# --------------------------------------------------------------------------- #
def julia_depth_section(all_adm: list) -> list:
    L: list = []
    w = L.append
    w("## E. Julia admission depth distribution\n")
    jd = [int(r.get("reached_depth", 0)) for r in all_adm
          if str(r.get("family", "")).startswith("julia:")]
    if not jd:
        w("_No julia admissions yet._\n")
        return L
    hist = Counter(jd)
    w(f"- **{len(jd)} julia admissions**, depth median **{int(np.median(jd))}**, "
      f"range [{min(jd)}, {max(jd)}]. The shallow (depth 1–2) center-descent population — "
      f"suppressed in prior runs — should now appear.\n")
    w("| reached_depth | julia admissions |")
    w("|--:|--:|")
    for d in sorted(hist):
        w(f"| {d} | {hist[d]} |")
    shallow = sum(n for d, n in hist.items() if d <= 2)
    w(f"\n_Shallow (depth ≤ 2): {shallow}/{len(jd)} = {shallow/len(jd):.0%}._\n")
    return L


# --------------------------------------------------------------------------- #
# F. sat_frac trajectory.
# --------------------------------------------------------------------------- #
def saturation_section(run_dir: Path, summary: dict) -> list:
    L: list = []
    w = L.append
    w("## F. Novelty-memory saturation (sat_frac) trajectory\n")
    sat = c1.load_jsonl(run_dir / "saturation.jsonl")
    if not sat:
        w("_No saturation.jsonl yet (lambda_m=0, or no candidate scored)._\n")
        return L
    fracs = [float(s["frac"]) for s in sat if "frac" in s]
    overall = (sum(int(s["sat"]) for s in sat) / max(1, sum(int(s["n"]) for s in sat)))
    w(f"- **Overall sat_frac {overall:.3f}** over {len(sat)} scored batches "
      f"(campaign-1 context 0.735; permanent novelty memory now larger). Report, don't tune.")
    if fracs:
        q = len(fracs) // 4 or 1
        w(f"  - trajectory (batch-quartile means): "
          + " → ".join(f"{np.mean(fracs[i:i+q]):.2f}" for i in range(0, len(fracs), q)))
    mem = sat[-1] if sat else {}
    w(f"  - final memory: perm {mem.get('mem_perm','?')} + recency {mem.get('mem_recency','?')} "
      f"= {mem.get('mem_total','?')} looks.\n")
    return L


def build(args) -> str:
    breadth = Path(args.breadth).resolve()
    sched = load_scheduler_summary(breadth)
    totals = load_totals(breadth)
    b_rows = c1.load_jsonl(breadth / "outcome_ledger.jsonl")
    all_adm = c1.admissions(b_rows)
    active_min = c1.active_min_of(breadth)

    L: list = []
    w = L.append
    w("# Campaign 2 — deficit scheduler + freshness prior: readout\n")
    w(f"_Run `{breadth.name}`, active **{active_min:.1f} min** ({active_min/60:.2f} h), "
      f"{len(all_adm)} distinct-q3 admissions. Scheduler + freshness prior both ON._\n")

    L += scheduler_section(breadth, sched)
    L += prices_section(sched)
    L += freshness_section(breadth, totals, all_adm, active_min)
    L += hook_section(breadth)
    L += julia_depth_section(all_adm)
    L += saturation_section(breadth, sched)

    w("## G. Visual samples\n")
    w("Admission / reject contact sheets: "
      "`uv run python tools/atlas/campaign1_contact_sheet.py --run-dir "
      f"{breadth.relative_to(ROOT)}` (reused; run-dir-agnostic).\n")

    # base campaign-1 numbers (throughput/cost/distinct-look/overlap/coverage) appended verbatim.
    w("---\n\n# Base scheduling numbers (campaign1_readout, reused verbatim)\n")
    c1_args = argparse.Namespace(
        breadth=str(breadth), dive=None,
        out=str(ROOT / "out" / "campaign2" / "_base.md"),
        no_morph=args.no_morph,
        prior_ledgers=[str(p) for p in sorted((ROOT / "data").rglob("outcome_ledger.jsonl"))
                       if "campaign2" not in p.parts])
    try:
        w(c1.build(c1_args))
    except Exception as e:
        w(f"_(base campaign-1 section failed: {e})_\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--breadth", required=True, help="campaign-2 breadth run dir")
    ap.add_argument("--out", default=str(ROOT / "out" / "campaign2" / "readout.md"))
    ap.add_argument("--no-morph", action="store_true", help="skip the GPU morph pass (cheap only)")
    args = ap.parse_args()
    md = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"\nwrote {out}\n")
    print(md)


if __name__ == "__main__":
    main()
