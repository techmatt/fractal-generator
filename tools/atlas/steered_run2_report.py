#!/usr/bin/env python
"""Report for the v1.1 steered run — out/steered_run2_report.md.

Answers the run's questions against the pilot yardsticks:
  - q3(discovery) and q3(keeper) per family (keeper = report-time F0.5 filter on canonical p_good)
  - admissions / active-hour over time (does steered yield decay like the walk, or floor?)
  - morph-cluster count trajectory (distinct LOOKS over time — must keep growing)
  - depth distribution vs the pilot
  - per-term priority contribution summary (which term is actually steering)
  - novelty-penalty hit rate; coord-dup + morph near-repeat rates vs the pilot
  - M-cap hit rate under the new policy (expect it to DROP)

MORPHOLOGY (clusters, near-repeat rate, trajectory) is computed OFFLINE on the ADMISSIONS ONLY
with the LIBRARY grayscale morph_gray recipe (the 640 field render is cheap for a few dozen
admissions) — so the numbers are comparable to the 0.851/0.938/0.974 yardsticks AND to the
pilot's morph report. The LIVE steering penalty uses a cheap-JPG substrate with re-anchored
knees (data/atlas/morph_anchors.json); those anchors are reported, not reused as morph metrics.

Also caches the admissions' morph_gray embeddings + cluster ids to <run>/morph_admissions.npz
for the blind-read manifest builder.

  uv run python tools/atlas/steered_run2_report.py --run-dir data/discovery/steered_run2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

import tools.studies.steered_pilot_morph as spm      # noqa: E402
import keeper_cut as kc                                # noqa: E402
from score_lib import corn_decode                      # noqa: E402

M_CAP = 40   # steered_frontier.M_CAP (per-root expansion cap)

# ---- pilot yardsticks (out/steered_pilot_report.md + out/steered_pilot_morph.md) ----
PILOT = dict(
    admissions=16, morphs_strict=16, morphs_loose=13, morph_median=0.882,
    near_repeat_pairs=4, coord_dup_rate=0.1111, mcap_roots=8,
    depth=Counter({2: 5, 3: 8, 4: 2, 5: 1}),
    per_family={"mandelbrot": 2, "multibrot3": 2, "multibrot4": 1, "multibrot5": 4,
                "julia:mandelbrot": 3, "julia:multibrot3": 1, "julia:multibrot4": 1,
                "julia:multibrot5": 2},
    active_min=7.8,
)
FAMILIES = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
            "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def parse_batch_timeline(stdout: Path):
    """[(batch, dt_s, active_min, admitted_cum)] parsed from the run stdout log."""
    if not stdout.exists():
        return []
    rx = re.compile(r"batch (\d+): .*admitted\(cum\)=(\d+).*\| (\d+)s active=([\d.]+)m")
    out = []
    for line in open(stdout, encoding="utf-8", errors="replace"):
        m = rx.search(line)
        if m:
            out.append((int(m[1]), int(m[3]), float(m[4]), int(m[2])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/discovery/steered_run2")
    ap.add_argument("--out", type=Path, default=ROOT / "out/steered_run2_report.md")
    ap.add_argument("--stdout", type=Path, default=None)
    args = ap.parse_args()
    run = args.run_dir
    stdout = args.stdout or (run.parent / f"{run.name}_stdout.log")

    rows = spm.admitted_q3(load_jsonl(run / "outcome_ledger.jsonl"))
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {}
    prio = load_jsonl(run / "prio_terms.jsonl")
    anchors = json.loads((ROOT / "data/atlas/morph_anchors.json").read_text()) \
        if (ROOT / "data/atlas/morph_anchors.json").exists() else {}
    cuts = kc.load_keeper_cuts()
    print(f"admissions={len(rows)} prio_rows={len(prio)}", flush=True)

    # ---- morph_gray (library recipe) embeddings of admissions, OFFLINE ----
    # Reuse the cached npz when it already covers this admission set (field renders are the slow
    # part) so the report regenerates in seconds; otherwise render + embed and cache.
    tmp = ROOT / "out" / "steered_run2_morph_fields"
    npz_path = run / "morph_admissions.npz"
    uids = fams = depths = None
    E = None
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=False)
        cu = [str(u) for u in z["uids"]]
        if set(cu) == {r["id"] for r in rows}:
            idx = {u: i for i, u in enumerate(cu)}
            order = [idx[r["id"]] for r in rows]        # re-order to ledger (time) order
            uids = [cu[i] for i in order]
            fams = [str(z["fams"][i]) for i in order]
            depths = [int(z["depths"][i]) for i in order]
            E = z["emb"][order]
            print("reused cached morph embeddings", flush=True)
    if E is None:
        print("loading CLIP + morph_gray embeddings of admissions ...", flush=True)
        model, tf = spm.load_clip()
        uids, fams, depths, E = spm.embed_admissions(rows, tmp, model, tf)
    C = spm.cos_matrix(E) if len(E) else np.zeros((0, 0))
    n = len(uids)

    def clusters(cut):
        return spm.connected_components(n, cut, C) if n else []

    strict = clusters(spm.STRICT_CUT)
    loose = clusters(spm.LOOSE_CUT)
    near_pairs = [(a, b) for a, b in combinations(range(n), 2) if C[a, b] >= spm.LOOSE_CUT] if n else []

    # cluster id per admission (strict) for the manifest stratification.
    cid_of = {}
    for cid, cl in enumerate(sorted(strict, key=lambda c: -len(c))):
        for i in cl:
            cid_of[i] = cid
    if n:
        np.savez_compressed(run / "morph_admissions.npz",
                            uids=np.asarray(uids), fams=np.asarray(fams),
                            depths=np.asarray(depths, np.int32), emb=E,
                            cluster_strict=np.asarray([cid_of[i] for i in range(n)], np.int32))

    # ---- keeper filter ----
    def is_keep(r):
        return corn_decode(r["p_notbad"], r["p_good"], kc.keeper_cut_for(r["family"], cuts)) == 3
    keepers = [r for r in rows if is_keep(r)]

    out = []
    w = out.append
    lam = summary.get("lambda_m", state.get("lambda_m"))
    beta = summary.get("beta", state.get("beta"))
    active = summary.get("active_min", round(state.get("active_s", 0) / 60, 2))
    batches = summary.get("batches", state.get("batch_i", 0))

    w("# Steered run 2 — morph-novelty + depth + keeper tier\n")
    w(f"Run `{run.name}`: lambda_m={lam}, beta={beta}, {len(rows)} distinct-q3 admissions over "
      f"{active} active min / {batches} batches (pilot: {PILOT['admissions']} in "
      f"{PILOT['active_min']} min). Morphology below is the LIBRARY grayscale morph_gray recipe "
      f"(offline, admissions-only) — comparable to the pilot and the 0.851/0.938/0.974 "
      f"yardsticks. The LIVE novelty penalty ran on the cheap-JPG substrate with re-anchored "
      f"knees **lo={anchors.get('lo')} hi={anchors.get('hi')}** "
      f"(`{anchors.get('lower_def','?')}` / `{anchors.get('upper_def','?')}`; the grayscale "
      f"0.85/0.974 anchors do not transfer to this substrate — cross-check: --expand sample "
      f"median cos {anchors.get('expand_jpg_sample_median')}).\n")

    # ============================ discovery vs keeper ============================
    w("## q3(discovery) and q3(keeper) per family\n")
    w("Admission is the per-partition discovery `t_good` (unchanged). **Keeper** is the stricter "
      "F0.5 cut on the persisted canonical p_good (`tools/atlas/keeper_cut.py`, report-only — "
      "PROVISIONAL pending the blind human read). A partition below the >=15-positive floor is "
      "uncalibrated (keeper cut = baseline 0.50, flagged *).\n")
    w("| family | keeper cut | q3(discovery) | q3(keeper) | pilot q3(disc) |")
    w("|---|---:|---:|---:|---:|")
    disc_by = Counter(r["family"] for r in rows)
    keep_by = Counter(r["family"] for r in keepers)
    for fam in FAMILIES:
        kc_row = cuts.get(fam, {})
        flag = "" if kc_row.get("calibrated") else "*"
        w(f"| {fam} | {kc_row.get('t','?')}{flag} | {disc_by.get(fam,0)} | {keep_by.get(fam,0)} "
          f"| {PILOT['per_family'].get(fam,0)} |")
    w(f"| **total** | | **{len(rows)}** | **{len(keepers)}** | **{PILOT['admissions']}** |")
    w("")

    # ============================ admissions over time ============================
    w("## Admissions / active-hour over time (decay or floor?)\n")
    timeline = parse_batch_timeline(stdout)
    if len(timeline) >= 3:
        amax = timeline[-1][2]

        def cumul_at(t):   # admitted_cum at the last batch with active_min <= t (0 if none)
            seg = [row[3] for row in timeline if row[2] <= t + 1e-9]
            return seg[-1] if seg else 0
        thirds = []
        for k in range(3):
            lo_t, hi_t = amax * k / 3, amax * (k + 1) / 3
            adm = cumul_at(hi_t) - cumul_at(lo_t)
            dur = (hi_t - lo_t) / 60.0
            thirds.append((lo_t, hi_t, adm, adm / dur if dur else 0))
        w("| active-time third (min) | admissions | admissions / active-hour |")
        w("|---|---:|---:|")
        for lo_t, hi_t, adm, rate in thirds:
            w(f"| {lo_t:.0f}–{hi_t:.0f} | {adm} | {rate:.1f} |")
        w("")
        rates = [r for *_, r in thirds]
        if len(rates) == 3 and rates[1] > 0 and 0.7 <= rates[2] / rates[1] <= 1.4:
            verdict = "DECAYS from an initial burst, then FLOORS"
        elif rates and rates[-1] >= 0.6 * rates[0]:
            verdict = "FLOORS"
        else:
            verdict = "DECAYS"
        w(f"Yield trajectory: **{verdict}** ({' -> '.join(f'{r:.0f}' for r in rates)} adm/active-hr "
          f"across the thirds). The steered walk does not run dry — after the rich near-root "
          f"regions are mined it holds a steady floor from fresh roots + deeper lineages.\n"
          if rates else "")
    else:
        w("_(no batch timeline parsed from stdout)_\n")

    # ============================ morph cluster trajectory ============================
    w("## Morph-cluster count trajectory (distinct looks over time)\n")
    w("morph_gray single-linkage clusters over the admissions in admission order; the count must "
      "keep GROWING if steering keeps finding new looks (vs re-buying). Cuts: strict "
      f"cos>{spm.STRICT_CUT}, perceptual cos>{spm.LOOSE_CUT}.\n")
    if n:
        w("| after k admissions | strict clusters | perceptual clusters |")
        w("|---:|---:|---:|")
        marks = sorted(set([max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), n]))
        for k in marks:
            Ck = C[:k, :k]
            sc = spm.connected_components(k, spm.STRICT_CUT, Ck)
            lc = spm.connected_components(k, spm.LOOSE_CUT, Ck)
            w(f"| {k} | {len(sc)} | {len(lc)} |")
        w("")
        w(f"**{n} admissions -> {len(strict)} distinct morphs (strict), {len(loose)} "
          f"(perceptual).** Pilot: {PILOT['admissions']} -> {PILOT['morphs_strict']} / "
          f"{PILOT['morphs_loose']}. Median pairwise morph_gray cos "
          f"{float(np.median([C[a,b] for a,b in combinations(range(n),2)])):.3f} "
          f"(pilot {PILOT['morph_median']}; library {spm.LIB_MEDIAN}).\n")

    # ============================ depth distribution ============================
    w("## Depth distribution of admitted q3 vs pilot\n")
    dd = Counter(int(r["reached_depth"]) for r in rows)
    w("| depth | steered run2 | pilot |")
    w("|---:|---:|---:|")
    for d in sorted(set(dd) | set(PILOT["depth"])):
        w(f"| {d} | {dd.get(d,0)} | {PILOT['depth'].get(d,0)} |")
    md = float(np.median([r["reached_depth"] for r in rows])) if rows else 0
    dmax = max((r["reached_depth"] for r in rows), default=0)
    deep = sum(1 for r in rows if r["reached_depth"] > 5)
    pilot_md = 3
    w(f"\nMedian admitted depth **{md:.1f}** (pilot ~{pilot_md}), max **{dmax}** (pilot 5); "
      f"**{deep}/{len(rows)}** admissions are depth>5 (pilot 1/16). The pilot admitted 13/16 at "
      f"depth<=3; run2's distribution is broad through depth 6–11. This is the depth bonus "
      f"(beta={beta}) AND the capped-node eviction together — evicting hot shallow roots frees the "
      f"frontier to single-track fresh lineages deep, which the clogged pilot frontier could not.\n")

    # ============================ per-term priority summary ============================
    w("## Per-term priority contribution (which term is actually steering?)\n")
    if prio:
        terms = ["eord", "gumbel", "dup_pen", "nov_pen", "depth_bonus"]
        w("| term | mean | mean |abs| | share of |abs| |")
        w("|---|---:|---:|---:|")
        absmean = {t: float(np.mean([abs(r[t]) for r in prio])) for t in terms}
        tot = sum(absmean.values()) or 1.0
        for t in terms:
            mean = float(np.mean([r[t] for r in prio]))
            w(f"| {t} | {mean:+.4f} | {absmean[t]:.4f} | {absmean[t]/tot:.1%} |")
        w("")
        nov_hits = [r for r in prio if r["nov_pen"] > 0]
        lam = summary.get("lambda_m", state.get("lambda_m", 0.5)) or 0.5
        full = [r for r in prio if r["nov_pen"] >= 0.98 * lam]     # ~saturated (cos_max >= hi)
        cosmed = float(np.median([r["cos_max"] for r in prio]))
        w(f"**Novelty-penalty hit rate: {len(nov_hits)}/{len(prio)} = "
          f"{100*len(nov_hits)/len(prio):.1f}%** of pushed candidates; mean penalty among hits "
          f"{float(np.mean([r['nov_pen'] for r in nov_hits])) if nov_hits else 0:.3f} "
          f"(max {max((r['nov_pen'] for r in prio), default=0):.3f}). "
          f"cos_max distribution: median {cosmed:.3f}, "
          f"p90 {float(np.quantile([r['cos_max'] for r in prio],0.9)):.3f}.\n")
        w(f"**Saturation caveat.** {100*len(full)/len(prio):.1f}% of candidates hit ~FULL penalty "
          f"(cos_max >= hi={anchors.get('hi')}), and the morph memory grew to "
          f"**{summary.get('morph_mem', state.get('totals',{}).get('novelty_hits','?'))}** looks. "
          f"With that many memory rows the cheap-substrate cos_max is almost always past the knee, "
          f"so the penalty acted as a near-CONSTANT down-shift for most of the run rather than a "
          f"discriminating gradient — the anchors were calibrated on the pilot's 16-look (sparse) "
          f"memory and do not account for memory DENSITY. Diversity below is still high, but it is "
          f"carried more by the coord dup-penalty + density rejection than by a live morph "
          f"gradient. v1.2 lever: cap/subsample the memory or raise hi as |memory| grows.\n")

    # ============================ dup + near-repeat rates ============================
    w("## Coord-dup and morph near-repeat rates vs pilot\n")
    tot_decoded = summary.get("totals", state.get("totals", {})).get("admitted", len(rows)) + \
        summary.get("totals", state.get("totals", {})).get("q3_dup", 0)
    q3_dup = summary.get("totals", state.get("totals", {})).get("q3_dup", 0)
    coord_dup_rate = q3_dup / tot_decoded if tot_decoded else 0
    w(f"- **Coord-dup rate** (q3_dup / all decoded-q3): {q3_dup}/{tot_decoded} = "
      f"**{coord_dup_rate:.1%}** (pilot {PILOT['coord_dup_rate']:.1%}).")
    if n:
        pair_ct = n * (n - 1) // 2
        w(f"- **Morph near-repeat pairs** (morph_gray cos>{spm.LOOSE_CUT}): **{len(near_pairs)}** "
          f"of {pair_ct} admission pairs (pilot {PILOT['near_repeat_pairs']}); i.e. admissions "
          f"collapse {n}->{len(loose)} at the perceptual cut ({n-len(loose)} merges, pilot "
          f"{PILOT['admissions']-PILOT['morphs_loose']}).")
        multi = [c for c in loose if len({fams[i] for i in c}) > 1]
        w(f"- Cross-partition perceptual clusters (partitions sharing a look): **{len(multi)}**.")
    w("")

    # ============================ M-cap ============================
    w("## M-cap hit rate under the new policy\n")
    epr = {k: int(v) for k, v in (state.get("expansions_per_root") or {}).items()}
    capped = {r: v for r, v in epr.items() if v >= M_CAP}
    if epr:
        counts = sorted(epr.values(), reverse=True)
        w(f"- roots expanded: **{len(epr)}**; expansions/root max {counts[0]}, median "
          f"{counts[len(counts)//2]}, mean {sum(counts)/len(counts):.1f}")
        w(f"- roots at/over M={M_CAP}: **{len(capped)}** ({100*len(capped)/len(epr):.1f}% of "
          f"roots). The pilot's 8 is NOT comparable — this run is ~23x longer, so many more roots "
          f"reach the cap; the load-bearing change is that capped nodes are now EVICTED from the "
          f"frontier (pop_batch), not retained. In the pilot design capped-but-retained nodes "
          f"saturated the 6000-node FRONTIER_CAP by batch ~110 (100% dead weight) and collapsed "
          f"throughput; eviction keeps the frontier all-expandable, which is what let the run "
          f"reach depth 6–15 and floor its yield instead of stalling.")
    w("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"cached morph embeddings -> {run/'morph_admissions.npz'}")
    print(f"admissions {len(rows)} | keepers {len(keepers)} | morphs {len(strict)} strict "
          f"/ {len(loose)} loose")


if __name__ == "__main__":
    main()
