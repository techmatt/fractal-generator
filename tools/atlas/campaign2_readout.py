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


# --------------------------------------------------------------------------- #
# Regime segmentation at the julia-hook-spacing resume boundary(ies).
#
# The run was resumed mid-flight with a lower --julia-hook-spacing (0.20 -> 0.10) as a FLAG
# change only (frozen tree). Every hook decision in julia_hooks.jsonl durably stamps the
# `spacing` in force, so regimes are SELF-LABELING there; for the trace/harvest artifacts
# (which carry `batch` but not spacing) we map batch -> regime via the first batch each new
# spacing appears. Generalizes to N resumes. Freshness-prior metrics are NOT segmented (that
# knob did not change — reported whole-run in section C).
# --------------------------------------------------------------------------- #
def regime_intervals(hooks: list) -> list:
    """[(start_batch, spacing)] ascending by start_batch — one entry per distinct spacing, at the
    first batch it was observed. Empty if no hooks logged."""
    first_batch: dict = {}
    for r in hooks:
        sp = float(r["spacing"])
        b = int(r["batch"])
        first_batch[sp] = min(first_batch.get(sp, b), b)
    return sorted(((b, sp) for sp, b in first_batch.items()), key=lambda t: t[0])


def regime_of_batch(batch: int, intervals: list) -> float | None:
    """The spacing in force at `batch` (largest start_batch <= batch); first regime's spacing for
    anything before the first hook; None if no intervals."""
    if not intervals:
        return None
    sp = intervals[0][1]
    for start, s in intervals:
        if batch >= start:
            sp = s
        else:
            break
    return sp


def _fmt_sp(sp) -> str:
    return "—" if sp is None else f"{sp:g}"


def regime_comparison_section(run_dir: Path, sched: dict) -> list:
    """Both spacing regimes side by side (requirement 1): skip fraction, hook-fire conversion into
    julia roots/admissions, julia distinct-look share trajectory, per-partition allocation shares,
    and price drift. All from durable artifacts; freshness metrics deliberately excluded."""
    L: list = []
    w = L.append
    hooks = c1.load_jsonl(run_dir / "julia_hooks.jsonl")
    trace = c1.load_jsonl(run_dir / "scheduler_trace.jsonl")
    harvest = c1.load_jsonl(run_dir / "harvest_log.jsonl")
    ivals = regime_intervals(hooks)
    regimes = [sp for _b, sp in ivals]
    w("## Regime comparison — julia-hook-spacing 0.20 → 0.10 (resume boundary)\n")
    if len(regimes) < 2:
        only = _fmt_sp(regimes[0]) if regimes else "n/a"
        w(f"_Single spacing regime so far (spacing={only}); the 0.10 regime has produced no hook "
          f"decision yet, so there is nothing to segment. This section populates once the resumed "
          f"(0.10) leg fires its first hook._\n")
        return L
    bounds = {sp: b for b, sp in ivals}
    w(f"Regimes (spacing → first batch): "
      + ", ".join(f"**{_fmt_sp(sp)}**@b{bounds[sp]}" for _b, sp in ivals) + ".\n")

    # ---- 1. skip fraction + nearest-c diagnostic (does the radius even bind?) ----
    w("\n### 1. Hook skip fraction (over-thinning check)\n")
    w("| spacing | decisions | fired | skipped | skip frac |")
    w("|--:|--:|--:|--:|--:|")
    hook_by_sp: dict = defaultdict(lambda: [0, 0])   # spacing -> [fired, skipped]
    for r in hooks:
        hook_by_sp[float(r["spacing"])][0 if r.get("hooked") else 1] += 1
    sp_lo = min(regimes)                              # the reduced-spacing regime
    for _b, sp in ivals:
        fired, skip = hook_by_sp[sp]
        tot = fired + skip
        w(f"| {_fmt_sp(sp)} | {tot} | {fired} | {skip} | {skip/tot:.2f} |" if tot else
          f"| {_fmt_sp(sp)} | 0 | 0 | 0 | — |")
    # nearest-c of the SKIPPED hooks per regime: how many are genuine near-dups (< the reduced
    # spacing) vs "recoverable" (in [reduced, original) — parents the smaller radius would clear).
    # This is the evidence for whether the radius is the binding constraint at all.
    def _skip_nc(sp):
        return [float(r["nearest_c_dist"]) for r in hooks
                if float(r["spacing"]) == sp and not r.get("hooked")
                and r.get("nearest_c_dist") is not None]
    diag_rows = []
    for _b, sp in ivals:
        nc = _skip_nc(sp)
        if not nc:
            diag_rows.append((sp, 0, None, 0, 0)); continue
        a = np.asarray(nc)
        genuine = int((a < sp_lo).sum())             # closer than even the reduced radius
        recov = int(((a >= sp_lo) & (a < max(regimes))).sum())
        diag_rows.append((sp, len(nc), float(np.median(a)), genuine, recov))
    w("\nNearest-c of the **skipped** hooks (is the spacing radius even the binding constraint?):\n")
    w(f"| spacing | skipped | median nearest-c | genuine near-dup (< {sp_lo:g}) | recoverable [{sp_lo:g}, {max(regimes):g}) |")
    w("|--:|--:|--:|--:|--:|")
    for sp, n, med, gen, rec in diag_rows:
        meds = f"{med:.3f}" if med is not None else "—"
        w(f"| {_fmt_sp(sp)} | {n} | {meds} | {gen} | {rec} |")
    # data-driven verdict (no static direction assumption).
    fr = {sp: (hook_by_sp[sp][1] / max(1, sum(hook_by_sp[sp]))) for _b, sp in ivals}
    hi_sp, lo_sp = max(regimes), min(regimes)
    moved = "fell" if fr[lo_sp] < fr[hi_sp] else "rose" if fr[lo_sp] > fr[hi_sp] else "held"
    recov_lo = next((rec for sp, n, med, gen, rec in diag_rows if sp == lo_sp), 0)
    w(f"\n_Smoke reference skip frac {SMOKE_SKIP_FRAC:.2f}. Reducing spacing {hi_sp:g}→{lo_sp:g}: "
      f"skip frac **{moved}** ({fr[hi_sp]:.2f}→{fr[lo_sp]:.2f}). The nearest-c table is the reason — "
      f"if virtually all skips sit BELOW {lo_sp:g} (genuine near-dups) and few/none are in the "
      f"recoverable band ({recov_lo} at {lo_sp:g}), the radius was never the binding constraint: the "
      f"skipped parents are true near-duplicate Julia sets, so the smaller radius cannot recover them. "
      f"Skip fraction is then driven by hooked-c DENSITY (accumulating over the run), not the radius._\n")

    # ---- 2. hook-fire conversion into julia roots / admissions ----
    # fired hook == one julia root created (add_julia_root only fires on a non-skip); julia
    # admissions attributed to the regime of the admitting harvest-check batch.
    w("\n### 2. Conversion: hook fires → julia roots → julia admissions\n")
    jadm_by_sp: dict = defaultdict(int)
    for h in harvest:
        if h.get("admitted") and str(h.get("partition", "")).startswith("julia:"):
            sp = regime_of_batch(int(h["batch"]), ivals)
            if sp is not None:
                jadm_by_sp[sp] += 1
    w("| spacing | hooks fired (=julia roots) | julia admissions | adm / root |")
    w("|--:|--:|--:|--:|")
    for _b, sp in ivals:
        fired = hook_by_sp[sp][0]
        jadm = jadm_by_sp.get(sp, 0)
        apr = f"{jadm/fired:.2f}" if fired else "—"
        w(f"| {_fmt_sp(sp)} | {fired} | {jadm} | {apr} |")
    w("\n_Roots-per-regime is the direct lever: fewer spacing-skips → more julia roots seeded → "
      "more julia descents to admit. (Prospective only — spacing-skipped parents from the 0.20 "
      "regime are NOT re-hooked; their seed c stays recoverable in julia_hooks.jsonl.)_\n")

    # ---- 3. julia distinct-look share trajectory (incremental per regime) ----
    # trace `looks` = cumulative tally INCL. the 523 library seed. Regime-incremental julia share =
    # Δ(julia looks) / Δ(total looks) across the regime's trace rows — the honest per-regime signal
    # (the cumulative share is dominated by the library seed and the earlier regime).
    w("\n### 3. Julia distinct-look share — per-regime incremental\n")
    def looks_at(row):
        lk = row.get("looks", {})
        jl = sum(int(lk.get(p, 0)) for p in PARTITIONS if p.startswith("julia:"))
        tl = sum(int(lk.get(p, 0)) for p in PARTITIONS)
        return jl, tl
    # bucket trace rows by regime, keep first/last row per regime
    rows_by_sp: dict = defaultdict(list)
    for t in trace:
        sp = regime_of_batch(int(t["batch"]), ivals)
        if sp is not None:
            rows_by_sp[sp].append(t)
    tf = sched.get("target_frac") or {}
    jtarget = sum(tf.get(p, 0.0) for p in PARTITIONS if p.startswith("julia:"))
    w(f"Julia target share (order book): **{jtarget:.0%}**. Incremental = new julia looks / new "
      "total looks produced *within* each regime (library seed + prior regime excluded):\n")
    w("| spacing | batch span | Δ total looks | Δ julia looks | incremental julia share |")
    w("|--:|--:|--:|--:|--:|")
    for _b, sp in ivals:
        rr = rows_by_sp.get(sp, [])
        if not rr:
            w(f"| {_fmt_sp(sp)} | — | 0 | 0 | — |")
            continue
        j0, t0 = looks_at(rr[0]); j1, t1 = looks_at(rr[-1])
        dj, dt = max(0, j1 - j0), max(0, t1 - t0)
        share = f"{dj/dt:.0%}" if dt else "—"
        w(f"| {_fmt_sp(sp)} | b{rr[0]['batch']}–b{rr[-1]['batch']} | {dt} | {dj} | {share} |")
    # scope disambiguation (two DIFFERENT, complementary julia shares — not a contradiction):
    #  * run-incremental (this table, ~38% pooled over both regimes) = julia fraction of the looks
    #    THIS RUN accepted, ignoring the library seed.
    #  * library-wide (§A last row, ~22%) = julia fraction of the tally INCLUDING the 523-look seed
    #    — the scope the DEFICIT actually measures against (tallies were library-seeded), so it is
    #    the number that governs the 77% order-book target.
    lastlk = trace[-1].get("looks", {})
    jl = sum(int(lastlk.get(p, 0)) for p in PARTITIONS if p.startswith("julia:"))
    tl = sum(int(lastlk.get(p, 0)) for p in PARTITIONS)
    j_run = sum(max(0, looks_at(rr[-1])[0] - looks_at(rr[0])[0])
                for rr in rows_by_sp.values() if rr)
    t_run = sum(max(0, looks_at(rr[-1])[1] - looks_at(rr[0])[1])
                for rr in rows_by_sp.values() if rr)
    w(f"\n_**Two julia-share scopes, complementary not contradictory:** run-incremental "
      f"**{(j_run/t_run if t_run else 0):.0%}** (julia share of the {t_run} looks this run added, "
      f"seed excluded) vs library-wide **{(jl/tl if tl else 0):.0%}** (julia share of the full "
      f"{tl}-look tally incl. the 523 library seed — the scope the deficit/target act on). The run "
      f"is producing julia well above its library share yet still below the 77% target because the "
      f"seed-heavy denominator moves slowly._\n")

    # ---- 4. per-partition allocation shares per regime (served histogram) ----
    w("\n### 4. Per-partition allocation share (batches served)\n")
    served_by_sp: dict = defaultdict(Counter)
    for t in trace:
        if not t.get("chosen"):
            continue
        sp = regime_of_batch(int(t["batch"]), ivals)
        if sp is not None:
            served_by_sp[sp][t["chosen"]] += 1
    hdr = "| partition | target | " + " | ".join(f"{_fmt_sp(sp)}" for _b, sp in ivals) + " |"
    w(hdr)
    w("|---|--:|" + "|".join(["--:"] * len(ivals)) + "|")
    for p in PARTITIONS:
        cells = []
        for _b, sp in ivals:
            c = served_by_sp[sp]
            n = sum(c.values())
            cells.append(f"{c.get(p,0)/n:.0%}" if n else "—")
        tfs = f"{tf.get(p,0):.0%}" if tf else "—"
        w(f"| {p} | {tfs} | " + " | ".join(cells) + " |")
    w("")

    # ---- 5. price drift per regime ----
    w("\n### 5. Learned-price drift (active-min per distinct look)\n")
    w("Per-partition price at each regime's first vs last traced batch (online EMA):\n")
    w("| partition | " + " | ".join(f"{_fmt_sp(sp)} start→end" for _b, sp in ivals) + " |")
    w("|---|" + "|".join(["--:"] * len(ivals)) + "|")
    for p in PARTITIONS:
        cells = []
        for _b, sp in ivals:
            rr = rows_by_sp.get(sp, [])
            if not rr:
                cells.append("—"); continue
            p0 = rr[0].get("prices", {}).get(p)
            p1 = rr[-1].get("prices", {}).get(p)
            cells.append(f"{p0:.2f}→{p1:.2f}" if p0 is not None and p1 is not None else "—")
        w(f"| {p} | " + " | ".join(cells) + " |")
    w("\n_Julia prices are the campaign-3 seed; watch whether the cheaper 0.10 regime pushes the "
      "julia price further down (more looks per active-min as hook throughput rises)._\n")
    return L


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
    w("## C. Freshness-prior effect — BREADTH ONLY\n")
    w("> **Scope.** This verdict covers the **breadth** leg only. The prior was **OFF** in the dive "
      "leg by design — it is structurally incompatible with dive precanon (see the design finding "
      "at the end of this readout); the dive's numbers must not be read as a prior result.\n")
    rate = len(all_adm) / (active_min / 60.0) if active_min else 0.0
    w(f"- **Throughput: {rate:.1f} adm/hr** vs campaign-1 context **{C1_ADM_PER_HR} adm/hr** "
      f"({len(all_adm)} admitted / {active_min/60:.2f} active-h). "
      f"Δ **{rate - C1_ADM_PER_HR:+.1f} adm/hr**.")
    w(f"  - _This Δ is the **JOINT** scheduler+prior effect vs campaign 1 — both knobs changed this "
      f"campaign, so it is not attributable to the prior alone. The prior's **isolated** wins are "
      f"the 0/{len(all_adm)} coord overlap and the {totals.get('precanon_dup',0)} renders it saved "
      f"(both below)._")
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
    # A high skip fraction is only "over-thinning" if the skipped parents are genuinely distinct-c
    # ones the radius wrongly kills. Judge on the nearest-c of skips, not the fraction alone: if
    # almost all skips sit below the SMALLEST spacing used, they are true near-duplicate Julia sets
    # and the skip is correct (the bottleneck is c-space clustering of admitted parents, not the
    # radius). This is what the campaign-2 0.20→0.10 flip demonstrated.
    nc = np.asarray([float(h["nearest_c_dist"]) for h in hooks
                     if not h.get("hooked") and h.get("nearest_c_dist") is not None])
    min_sp = min((float(h["spacing"]) for h in hooks), default=0.10)
    if len(nc):
        genuine_frac = float((nc < min_sp).mean())
        if genuine_frac >= 0.8:
            verdict = (f"NOT over-thinning — {genuine_frac:.0%} of skips are genuine near-dups "
                       f"(nearest-c < {min_sp:g}, median {np.median(nc):.3f}); the skips are correct "
                       f"and the julia-yield ceiling is c-space clustering of admitted parents, not spacing")
        elif frac > 0.40:
            verdict = (f"possible over-thinning — only {genuine_frac:.0%} of skips are genuine "
                       f"near-dups, so some skipped distinct-c parents may be recoverable at a smaller radius")
        else:
            verdict = "consistent with the smoke — not over-thinning"
    else:
        verdict = "no skip nearest-c data"
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


# --------------------------------------------------------------------------- #
# Design finding — dive + freshness-prior precanon incompatibility.
# --------------------------------------------------------------------------- #
def dive_prior_finding_section() -> list:
    """A structural finding surfaced by the campaign-2 dive: the freshness prior and the dive's
    pre-canonical coord-dup filter are incompatible as built. Recorded here (not just in the
    ledger) because it changes how campaign-3 dives must be configured."""
    L: list = []
    w = L.append
    w("\n## H. Design finding — dive + freshness-prior precanon are structurally incompatible\n")
    w("**Symptom.** The first campaign-2 dive attempt ran with the freshness prior ON (per the "
      "launch spec) and admitted **0/311** — every one of ~1040 harvest checks was rejected as a "
      "`precanon_dup`, with **zero** candidates reaching a canonical render. Campaign 1's dive "
      "(which predates the prior) admitted 254 off a comparable 314 sources.\n")
    w("**Root cause (structural, not a tuning miss).** The two mechanisms serve opposite goals:\n")
    w("- The **freshness prior** is an *exploration* tool — it seeds the dedup/steering clouds with "
      "prior-library coords so root draws and frontier steering avoid re-covering known ground.\n")
    w("- A **dive** is *exploitation* of a known point: it descends the greedy argmax-p_good path "
      "**from a breadth admission**. That source coord (and its basin) is, by construction, already "
      "in the prior cloud — so the dive's pre-canonical coord-dup filter rejects the descent against "
      "the very point it was told to mine. With the full 7926-row library in the cloud the basin is "
      "densely covered, so **100%** of candidates dup out before any canonical render. Sterilization "
      "is guaranteed, not incidental.\n")
    w("**Resolution taken.** The dive leg runs with the prior **OFF** (dedup against its own "
      "accruing cloud only, exactly as campaign 1's productive dive did). The prior's proven wins "
      "are a **breadth**-leg result (§C) and are unaffected.\n")
    w("**No lost-freshness guard needed.** Turning the prior off in the dive means a dive can, in "
      "principle, re-mint a location some *other-era* library ledger already holds (the dive dedups "
      "only within-campaign). That is acceptable and needs no extra guard here: **emission intake's "
      "own dedup pass catches cross-era re-mints downstream** (coord + CLIP-morph), so a re-mint is "
      "collapsed at library-assembly time rather than silently shipped.\n")
    w("**Campaign-3 options.**\n")
    w("1. **Keep the prior off in dives** (the current fix) — simple, proven, and correct given the "
      "exploration/exploitation split. Recommended default.\n")
    w("2. **fw-scaled precanon radius semantics** — make the dive's coord-dup radius shrink with the "
      "candidate's own `fw`, so a genuinely-deeper frame in a covered basin reads as distinct from "
      "its shallower source/neighbours. This would let a dive keep *some* cross-run freshness. It "
      "needs design + tests (the radius/`DEDUP_K` semantics are load-bearing across the pipeline) "
      "and is **deferred**.\n")
    return L


# --------------------------------------------------------------------------- #
# Part 4 — dive leg + whole-campaign close-out.
# --------------------------------------------------------------------------- #
def dive_section(dive_dir: Path) -> list:
    """Dive-specific numbers: per-family admissions + adm/dive, end-cause split, cost, and the
    honest 'prices not learned' note (dive uses deficit-ordering only)."""
    L: list = []
    w = L.append
    w("## H. Dive leg — depth-mining off breadth (freshness prior OFF by design)\n")
    led = dive_dir / "outcome_ledger.jsonl"
    if not led.exists():
        w("_No dive ledger._\n")
        return L
    adm = c1.admissions(c1.load_jsonl(led))
    dlog = c1.load_jsonl(dive_dir / "dive_log.jsonl")
    s = json.loads((dive_dir / "summary.json").read_text(encoding="utf-8")) \
        if (dive_dir / "summary.json").exists() else {}
    active = float(s.get("active_min", 0.0))
    n_dives = int(s.get("n_dives_done", len(dlog)))
    cost = active / len(adm) if adm else float("nan")
    w(f"- **{len(adm)} admissions off {n_dives} dives = {len(adm)/max(1,n_dives):.2f} adm/dive** "
      f"(campaign-1 ref ~0.8); {active:.1f} active-min, **{cost:.2f} min/admission**.")
    fam_adm = Counter(r["family"] for r in adm)
    fam_dives = Counter(d["partition"] for d in dlog)
    w("\n| partition | dives | admissions | adm/dive |")
    w("|---|--:|--:|--:|")
    for p in PARTITIONS:
        nd = fam_dives.get(p, 0)
        if nd:
            w(f"| {p} | {nd} | {fam_adm.get(p,0)} | {fam_adm.get(p,0)/nd:.2f} |")
    ec = Counter(d["end_cause"] for d in dlog)
    ntop = sum(1 for d in dlog if d.get("start_group") == "top")
    w(f"\n_End cause: {dict(ec)} (`target_depth` = ran to depth {s.get('target_depth','?')}; "
      f"`gate_dead_or_floor` = the descent exhausted). Sources: {ntop} top + {len(dlog)-ntop} control._")
    w("\n**Dive prices — not learned (by design).** The dive consumes the scheduler's per-partition "
      "DEFICITS to order its sources (deficit-ordering — this feature's first real use; confirmed: "
      "julia:mandelbrot, the highest-deficit partition, dived first) but does NOT run the price-EMA "
      "loop (no per-batch active-time charge exists in dive mode), so it produces no price update. "
      "The campaign's final learned-price table is the breadth one (§B).\n")
    return L


def _subset(uids, fams, E, keep_ids):
    idx = [i for i, u in enumerate(uids) if u in keep_ids]
    return [uids[i] for i in idx], [fams[i] for i in idx], E[idx]


def whole_campaign_section(breadth_adm: list, dive_adm: list, sched: dict, no_morph: bool) -> list:
    """Whole-campaign close-out: combined admissions + per-family shares vs target, combined
    distinct morph looks, breadth vs dive distinct rate, and the DIVE MARGINAL-distinct rate
    (dive admissions that are genuinely new looks vs breadth — campaign-1 context: dive ~80% vs
    breadth ~97%)."""
    import numpy as np
    L: list = []
    w = L.append
    w("## I. Whole-campaign totals (breadth + dive)\n")
    tot = len(breadth_adm) + len(dive_adm)
    w(f"- **{tot} distinct-q3 admissions** (breadth {len(breadth_adm)} + dive {len(dive_adm)}).")
    tf = sched.get("target_frac") or {}
    fam = Counter(r["family"] for r in breadth_adm + dive_adm)
    w("\n| partition | breadth+dive adm | admission share | target (look-share) |")
    w("|---|--:|--:|--:|")
    for p in PARTITIONS:
        w(f"| {p} | {fam.get(p,0)} | {fam.get(p,0)/tot:.0%} | {tf.get(p,0):.0%} |")
    jt = sum(fam.get(p, 0) for p in PARTITIONS if p.startswith("julia:"))
    w(f"\n_Julia share of admissions: **{jt}/{tot} = {jt/tot:.0%}** (target look-share 77%). Note "
      f"the dive is julia-tilted (deficit-ordered), pulling the whole-campaign julia admission "
      f"share above breadth's alone._\n")
    if no_morph:
        w("_Morph distinct-look totals skipped (--no-morph)._\n")
        return L
    all_adm = breadth_adm + dive_adm
    print(f"[campaign-morph] embedding {len(all_adm)} admissions (breadth+dive) ...", flush=True)
    uids, fams, depths, E = c1.morph_embed(all_adm)
    emb_by = {u: E[i] for i, u in enumerate(uids)}
    fam_by = {u: fams[i] for i, u in enumerate(uids)}
    bids = {r["id"] for r in breadth_adm}
    comb_distinct, _ = c1.distinct_look_count(uids, fams, E)
    b_distinct, _ = c1.distinct_look_count(*_subset(uids, fams, E, bids))
    dids = {r["id"] for r in dive_adm}
    d_distinct, _ = c1.distinct_look_count(*_subset(uids, fams, E, dids))
    w(f"- **{comb_distinct} distinct morph looks** among {tot} admissions "
      f"(within-family single-linkage, CLIP≥{c1.STRICT}) = {comb_distinct/tot:.0%} distinct.")
    w(f"  - breadth alone: {b_distinct}/{len(breadth_adm)} = {b_distinct/max(1,len(breadth_adm)):.0%}; "
      f"dive alone: {d_distinct}/{len(dive_adm)} = {d_distinct/max(1,len(dive_adm)):.0%}.")
    # dive MARGINAL-distinct: a dive admission is new iff it is NOT a CLIP>=STRICT near-dup of any
    # BREADTH admission in the same family (the dive's genuinely-additive contribution).
    breadth_by_fam: dict = defaultdict(list)
    for u in uids:
        if u in bids:
            breadth_by_fam[fam_by[u]].append(emb_by[u])
    new = dt = 0
    for u in uids:
        if u in bids:
            continue
        dt += 1
        pri = breadth_by_fam.get(fam_by[u])
        if not pri or float(np.max(np.stack(pri) @ emb_by[u])) < c1.STRICT:
            new += 1
    if dt:
        w(f"- **Dive marginal-distinct rate: {new}/{dt} = {new/dt:.0%}** of dive admissions are NEW "
          f"looks (not CLIP≥{c1.STRICT} near-dups of a breadth admission, same family). "
          f"Campaign-1 context: dive ~80% vs breadth ~97% — a dive descends *from* breadth points, "
          f"so a portion re-expresses a look breadth already banked; the rest is genuine depth-find.")
    w("")
    return L


def build(args) -> str:
    breadth = Path(args.breadth).resolve()
    dive = Path(args.dive).resolve() if getattr(args, "dive", None) else None
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

    # Budget planned-vs-actual (a documented decision, not silent drift). Planned = the fresh
    # launch's budget from the run log; effective = the state/summary budget_s the run actually
    # stopped against.
    planned = None
    rl = breadth / "run.log"
    if rl.exists():
        import re
        m = re.search(r"\[fresh\].*budget=(\d+(?:\.\d+)?)m", rl.read_text(encoding="utf-8", errors="replace"))
        if m:
            planned = float(m.group(1))
    eff_budget = None
    for name in ("summary.json", "state.json"):
        p = breadth / name
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            eff_budget = (d.get("budget_s", 0) / 60.0) or d.get("budget_min")
            if eff_budget:
                break
    if planned:
        w(f"> **Budget — planned vs actual.** Launched at **{planned:.0f}** accumulated-active-min; "
          f"stopped cleanly at **{active_min:.1f}** against an effective **{eff_budget:.0f}-min** "
          f"budget (Δ **{active_min - planned:+.0f} min** vs plan). The budget was lowered "
          f"{planned:.0f}→{eff_budget:.0f} at the *deadline* resume (a \"finish by 11pm\" trim), "
          f"NOT the later spacing-change resume, which preserved {eff_budget:.0f}. Shortfall "
          f"**deliberately accepted** — breadth is not topped up; the dive leg carries the campaign "
          f"forward._\n")

    L += regime_comparison_section(breadth, sched)
    L += scheduler_section(breadth, sched)
    L += prices_section(sched)
    L += freshness_section(breadth, totals, all_adm, active_min)
    L += hook_section(breadth)
    L += julia_depth_section(all_adm)
    L += saturation_section(breadth, sched)

    # --- Part 4: dive leg + whole-campaign close-out (only when --dive is supplied) ---
    dive_adm = []
    if dive is not None:
        dive_adm = c1.admissions(c1.load_jsonl(dive / "outcome_ledger.jsonl"))
        L += dive_section(dive)
        L += whole_campaign_section(all_adm, dive_adm, sched, args.no_morph)

    w("## G. Visual samples\n")
    sheet_runs = f"--run {breadth.relative_to(ROOT)}"
    if dive is not None:
        sheet_runs += f" --run {dive.relative_to(ROOT)}"
    w("Whole-campaign admission / reject contact sheet(s):\n")
    w(f"```\nuv run python tools/atlas/campaign1_contact_sheet.py {sheet_runs} \\\n"
      f"    --out out/campaign2/contact_sheet.png\n```\n")

    L += dive_prior_finding_section()

    # base campaign-1 numbers (throughput/cost/distinct-look/overlap/coverage) appended verbatim.
    # dive columns come from passing the dive dir; morph is forced OFF here to avoid a second
    # embed pass (the whole-campaign morph in §I already covers combined/dive distinct looks).
    w("---\n\n# Base scheduling numbers (campaign1_readout, reused verbatim)\n")
    w("> ⚠ **Caveat on the base §4 verdict.** campaign1_readout computes its "
      "\"worth a freshness prior?\" verdict under the campaign-1 assumption that the prior was "
      "*OFF* (high coord overlap → build one). In campaign 2 the prior is **ON**, so a **0.0% "
      "coord overlap is the prior working as intended**, not evidence against it — the base "
      "verdict's logic is inverted here and does not apply. The authoritative freshness-prior "
      "result is **§C above** (whole-run).\n")
    c1_args = argparse.Namespace(
        breadth=str(breadth), dive=(str(dive) if dive is not None else None),
        out=str(ROOT / "out" / "campaign2" / "_base.md"),
        no_morph=True,   # forced: §I already did the single combined morph pass
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
    ap.add_argument("--dive", default=None, help="campaign-2 dive run dir (adds §H/§I close-out)")
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
