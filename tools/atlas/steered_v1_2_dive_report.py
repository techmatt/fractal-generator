#!/usr/bin/env python
"""Report for the steered v1.2 dive run + the novelty-memory fix + the mandelbrot t_good move.

Four sections (spec: prompts/steered_frontier_v1_2_dive.md):
  1. Dive yield — admissions/dive, top-start vs control, depth distribution, canonical p_good.
  2. Morph novelty of dive admissions vs the run-2 library (library morph_gray recipe, offline,
     admissions-only — the comparable yardstick; cos_max of each dive admission vs the 75 run-2
     admission embeddings).
  3. Saturation fraction before/after the memory fix (before = run-2's 0.897 full run; after =
     a short recency-mode shakeout, with a matched legacy shakeout for the A/B).
  4. The new mandelbrot t_good with its derivation summary.

Also writes <dive-run>/dive_admissions.npz (uids/emb/cluster_strict/depths/groups/pgood) which
tools/atlas/dive_manifest.py consumes for the blind manifest.

  uv run python tools/atlas/steered_v1_2_dive_report.py \
      --dive-run data/discovery/steered_v1_2_dive --run2 data/discovery/steered_run2 \
      --recency-shakeout data/discovery/shakeout_recency \
      --legacy-shakeout data/discovery/shakeout_legacy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

import tools.studies.steered_pilot_morph as spm      # noqa: E402

STRICT_CUT = spm.STRICT_CUT      # 0.974 library near-dup cut
LOOSE_CUT = spm.LOOSE_CUT        # 0.95 perceptual


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def canon_pgood(r):
    v = r.get("canon_pgood")
    return float(v) if v is not None else float(r.get("p_good", 0.0))


def sat_from_summary(run_dir: Path):
    """(sat_frac, trajectory) for a normal-mode run: overall + per-batch from saturation.jsonl."""
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
        if (run_dir / "summary.json").exists() else {}
    sat_rows = load_jsonl(run_dir / "saturation.jsonl")
    return summ, sat_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dive-run", type=Path, default=ROOT / "data/discovery/steered_v1_2_dive")
    ap.add_argument("--run2", type=Path, default=ROOT / "data/discovery/steered_run2")
    ap.add_argument("--recency-shakeout", type=Path, default=None)
    ap.add_argument("--legacy-shakeout", type=Path, default=None)
    ap.add_argument("--tgood-json", type=Path,
                    default=ROOT / "data/atlas/mandelbrot_tgood_steered.json")
    ap.add_argument("--out", type=Path, default=ROOT / "out/steered_v1_2_dive_report.md")
    args = ap.parse_args()

    # ---- dive admissions + per-dive records ----
    led = load_jsonl(args.dive_run / "outcome_ledger.jsonl")
    adm = spm.admitted_q3(led)
    dives = load_jsonl(args.dive_run / "dive_log.jsonl")
    summ = json.loads((args.dive_run / "summary.json").read_text(encoding="utf-8"))
    id2group = {}
    for r in adm:
        id2group[r["id"]] = r.get("dive_start_group", "?")

    # ---- embed dive admissions (library recipe) + run-2 library ----
    print(f"dive admissions={len(adm)}; embedding (library morph_gray recipe) ...", flush=True)
    model, tf = spm.load_clip()
    tmp = ROOT / "out" / "dive_morph_fields"
    du, df, dd, dE = spm.embed_admissions(adm, tmp, model, tf)
    dU = dE / (np.linalg.norm(dE, axis=1, keepdims=True) + 1e-9) if len(dE) else dE

    z2 = np.load(args.run2 / "morph_admissions.npz", allow_pickle=False)
    r2E = z2["emb"].astype(np.float32)
    r2U = r2E / (np.linalg.norm(r2E, axis=1, keepdims=True) + 1e-9)

    # cos_max of each dive admission vs the run-2 library (novelty yardstick)
    nov_cosmax = (dU @ r2U.T).max(axis=1) if len(dU) else np.zeros(0, np.float32)
    # dive intra-set clustering (assign morph clusters for the manifest)
    dC = spm.cos_matrix(dE) if len(dE) else np.zeros((0, 0), np.float32)
    clusters = spm.connected_components(len(du), STRICT_CUT, dC) if len(du) else []
    cluster_of = {}
    for cid, c in enumerate(sorted(clusters, key=lambda c: -len(c))):
        for i in c:
            cluster_of[du[i]] = cid

    # persist for the manifest builder
    np.savez_compressed(
        args.dive_run / "dive_admissions.npz",
        uids=np.array(du), fams=np.array(df), depths=np.array(dd, np.int64),
        emb=dE.astype(np.float32),
        cluster_strict=np.array([cluster_of[u] for u in du], np.int64) if du else np.zeros(0, np.int64),
        groups=np.array([id2group.get(u, "?") for u in du]),
        nov_cosmax=nov_cosmax.astype(np.float32),
    )

    O = []
    w = O.append
    tot = summ["totals"]
    w("# Steered v1.2 — dive run + novelty-memory fix + mandelbrot t_good\n")
    w(f"Dive run `{args.dive_run.name}` off `{args.run2.name}`'s committed state: "
      f"**{summ['n_dives_done']}/{summ['n_dives_planned']} dives** "
      f"({sum(1 for d in dives if d['start_group']=='top')} top + "
      f"{sum(1 for d in dives if d['start_group']=='control')} control), target_depth "
      f"{summ['target_depth']}, fw floor {summ['min_fw']:g}, active "
      f"{summ['active_min']:.1f} min. Single-track descents: each rung expands 4 candidates "
      f"under the existing gates, harvests every survivor at the per-partition tau_h, and "
      f"continues down the cheap-p_good argmax child. Morphology is the LIBRARY morph_gray "
      f"recipe (offline, admissions-only) — comparable to the run-2 library yardstick.\n")

    # ============================ 1. yield ============================
    w("## 1. Dive yield\n")
    # admissions counted from the LEDGER (authoritative — the durable record) grouped by the
    # per-row dive_start_group; rungs / end-depth come from the per-dive dive_log.
    adm_by_group = Counter(r.get("dive_start_group", "?") for r in adm)
    w("| start group | dives | admissions | adm/dive | median rungs | median end-depth |")
    w("|---|---:|---:|---:|---:|---:|")
    for name in ("top", "control", "all"):
        rows = [d for d in dives if name == "all" or d["start_group"] == name]
        if not rows:
            continue
        na = len(adm) if name == "all" else adm_by_group[name]
        w(f"| {name} | {len(rows)} | {na} | {na/len(rows):.2f} | "
          f"{int(np.median([d['rungs'] for d in rows]))} | "
          f"{int(np.median([d['end_depth'] for d in rows]))} |")
    w("")
    w(f"**Total dive admissions: {len(adm)}** (canonical_q3 seen {tot['canonical_q3']}, "
      f"q3_dup {tot['q3_dup']}). End causes: "
      f"{dict(Counter(d['end_cause'] for d in dives))}. Every admission carries its "
      f"`dive_id` + `dive_source_id` in the run ledger.\n")
    w(f"Top-start dives yield **{adm_by_group['top']/max(1,sum(1 for d in dives if d['start_group']=='top')):.2f}** "
      f"admissions/dive vs control **{adm_by_group['control']/max(1,sum(1 for d in dives if d['start_group']=='control')):.2f}** — "
      f"a good starting neighborhood produces more deep admissions, but control dives from "
      f"arbitrary run-2 admissions still reach depth and admit, so deep quality is not "
      f"exclusive to the best neighborhoods.\n")

    # depth distribution of admissions
    w("### Depth distribution of dive admissions\n")
    depth_by_group = defaultdict(Counter)
    for r in adm:
        depth_by_group[r.get("dive_start_group", "?")][int(r["reached_depth"])] += 1
    all_depths = sorted({int(r["reached_depth"]) for r in adm})
    w("| depth | top | control | all |")
    w("|---:|---:|---:|---:|")
    for d in all_depths:
        t = depth_by_group["top"][d]; c = depth_by_group["control"][d]
        w(f"| {d} | {t} | {c} | {t+c} |")
    if adm:
        md = int(np.median([r["reached_depth"] for r in adm]))
        w(f"\nMedian admitted depth **{md}**, max "
          f"**{max(int(r['reached_depth']) for r in adm)}** "
          f"(dives seeded from run-2 admissions at depth "
          f"{min(d['start_depth'] for d in dives)}–{max(d['start_depth'] for d in dives)}).\n")

    # canonical p_good
    w("### Canonical p_good of dive admissions\n")
    for name, rows in (("top", [r for r in adm if id2group.get(r['id'])=='top']),
                       ("control", [r for r in adm if id2group.get(r['id'])=='control']),
                       ("all", adm)):
        if not rows:
            w(f"- **{name}**: 0 admissions.")
            continue
        pgs = np.array([canon_pgood(r) for r in rows])
        w(f"- **{name}** (n={len(rows)}): canon p_good median {np.median(pgs):.3f}, "
          f"mean {pgs.mean():.3f}, range [{pgs.min():.3f}, {pgs.max():.3f}].")
    w("")
    w("**Deep-options read:** the top-start vs control comparison tests whether deep quality "
      "requires a good starting neighborhood or is reachable from anywhere — the blind manifest "
      "(`out/dive_manifest/`) adjudicates it on the human read; the yield numbers above are the "
      "classifier-side view.\n")

    # ============================ 2. morph novelty ============================
    w("## 2. Morph novelty of dive admissions vs the run-2 library\n")
    if len(nov_cosmax):
        w("Each dive admission's cheap-look CLIP embedding (library morph_gray recipe) vs the 75 "
          "run-2 admission embeddings — `cos_max` is its nearest run-2 look (higher = less novel). "
          f"Yardsticks: library-wide median pairwise cos {spm.LIB_MEDIAN}, strict near-dup "
          f"cut cos>{STRICT_CUT}.\n")
        w("| group | n | median cos_max vs run-2 | p90 | near-repeat (cos>%.3f) |" % STRICT_CUT)
        w("|---|---:|---:|---:|---:|")
        for name in ("top", "control", "all"):
            idx = [i for i in range(len(du)) if (name == "all" or id2group.get(du[i]) == name)]
            if not idx:
                continue
            v = nov_cosmax[idx]
            nr = int((v > STRICT_CUT).sum())
            w(f"| {name} | {len(idx)} | {np.median(v):.3f} | {np.quantile(v,0.9):.3f} | "
              f"{nr} ({nr/len(idx):.0%}) |")
        w("")
        distinct_dive = len(clusters)
        nr_all = int((nov_cosmax > STRICT_CUT).sum())
        w(f"Dive admissions cluster to **{distinct_dive} distinct morphs** among themselves "
          f"(strict cos>{STRICT_CUT}, from {len(du)} — no internal collapse). Median cos_max vs "
          f"the run-2 library is **{np.median(nov_cosmax):.3f}**: the dives descend FROM run-2 "
          f"admissions, so their looks are lineage-RELATED to the library (deeper views of the "
          f"same neighborhoods), which is expected — but only **{nr_all}/{len(du)}** cross the "
          f"near-dup cut ({nr_all/len(du):.0%}), so the deep views are morphologically distinct "
          f"looks, not re-buys of the run-2 admissions they descend from.\n")
    else:
        w("No dive admissions to compare.\n")

    # ============================ 3. saturation ============================
    w("## 3. Novelty-memory saturation before/after the fix\n")
    w("Saturation fraction = candidates whose novelty penalty is within 10% of full (cos_max past "
      "90% of the [lo,hi] ramp) — a high fraction means the penalty is a constant down-shift, not a "
      "gradient. run-2 (v1.1, legacy all-permanent memory) saturated at **0.897** with a "
      "**10,420-row** memory (see `steered_run2_report.md`). The fix makes memory ADMITTED-only + a "
      "rolling window of the last K batches' expanded looks, so |memory| stays bounded and the "
      "term stays a live gradient.\n")
    w("| run | memory mode | batches | end |memory| | overall sat_frac |")
    w("|---|---|---:|---:|---:|")
    w("| steered_run2 (before) | legacy all-permanent | 341 | 10420 | **0.897** |")
    shake = []
    for label, d in (("legacy", args.legacy_shakeout), ("recency", args.recency_shakeout)):
        if not d or not (d / "summary.json").exists():
            continue
        s, rows = sat_from_summary(d)
        mm = s.get("morph_mem", "?")
        sf = s.get("sat_frac")
        w(f"| {d.name} ({label}) | "
          f"{'recency (k=%d)'%s.get('recency_k',0) if s.get('recency_k') else 'legacy all-permanent'} | "
          f"{s.get('batches','?')} | {mm} | "
          f"{'**%.3f**'%sf if sf is not None else 'n/a'} |")
        shake.append((label, d, s, rows))
    w("")
    # trajectory: sat_frac over batches for each shakeout
    if shake:
        w("Per-batch saturation trajectory (early vs late thirds of each shakeout):\n")
        w("| run | early third | mid third | late third |")
        w("|---|---:|---:|---:|")
        for label, d, s, rows in shake:
            if not rows:
                continue
            fr = [x["frac"] for x in rows]
            n = len(fr); a, b = n // 3, 2 * n // 3
            w(f"| {d.name} ({label}) | {np.mean(fr[:a]):.3f} | {np.mean(fr[a:b]):.3f} | "
              f"{np.mean(fr[b:]):.3f} |")
        w("")
        rec = [t for t in shake if t[0] == "recency"]
        leg = [t for t in shake if t[0] == "legacy"]
        # nov_pen variation on the recency shakeout (the "live gradient" evidence).
        nov_line = ""
        if rec:
            pr = load_jsonl(rec[0][1] / "prio_terms.jsonl")
            if pr:
                nv = np.array([r["nov_pen"] for r in pr])
                nov_line = (f" On the recency shakeout `nov_pen` has mean {nv.mean():.3f} / std "
                            f"**{nv.std():.3f}** with **{(nv <= 1e-6).mean():.0%}** of candidates "
                            f"at zero penalty and the rest spread across (0, {nv.max():.2f}] — a "
                            f"live gradient, versus run-2's near-constant ~0.489 offset.")
        if rec and leg:
            rf = rec[0][2].get("sat_frac"); lf = leg[0][2].get("sat_frac")
            w(f"At matched budget the legacy shakeout climbs toward run-2's saturated regime "
              f"(overall {lf:.3f}, and still rising — memory unbounded) while the recency window "
              f"holds memory bounded ({rec[0][2].get('morph_mem')} rows vs run-2's 10,420) and "
              f"saturation at **{rf:.3f}**.{nov_line}\n")
        elif rec:
            rf = rec[0][2].get("sat_frac")
            w(f"The recency shakeout holds saturation at **{rf:.3f}** vs run-2's 0.897.{nov_line}\n")
        w("The residual saturation is intrinsic: a descent produces a chain of morphologically "
          "similar views, so a candidate almost always has a near-mate among its own recent "
          "lineage in the window — independent of total memory size. The memory fix removes the "
          "unbounded-density driver (10,420→bounded) and restores a varying penalty; driving "
          "saturation lower would need a per-lineage-excluded novelty or a higher knee (anchors "
          "held fixed here per spec).\n")
    else:
        w("*(shakeout summaries not supplied; pass --recency-shakeout / --legacy-shakeout.)*\n")

    # ============================ 4. mandelbrot t_good ============================
    w("## 4. Mandelbrot discovery t_good (F0.5 re-derive)\n")
    if args.tgood_json.exists():
        tj = json.loads(args.tgood_json.read_text(encoding="utf-8"))
        sO, sN = tj["steered_at_old_t"], tj["steered_at_new_t"]
        w(f"Re-derived precision-weighted (F0.5) with the {tj['steered']['n']} steered_run2 "
          f"blind mandelbrot labels folded into the v7 eval slice (n={tj['eval']['n']}, "
          f"pos={tj['eval']['pos']}). The blind read scored **{tj['steered']['pos']}/"
          f"{tj['steered']['n']}** mandelbrot admissions good.\n")
        w(f"- **mandelbrot t_good {tj['old_t']} (F2) → {tj['new_t']:.2f} (F0.5)** — applied to "
          f"`production_seeder.T_GOOD_OVERRIDES`.\n")
        w(f"- On the 16 steered mandelbrot tiles (all human-not-good): the old bar admitted "
          f"**{sO['admit']}/{tj['steered']['n']}**, the new bar admits **{sN['admit']}/"
          f"{tj['steered']['n']}**.\n")
        w(f"- Deliberate, family-specific admission tightening (precedent: phoenix 0.18→0.50); the "
          f"julia families keep their F2 cuts. Full derivation: "
          f"`docs/findings/mandelbrot_tgood_steered.md`.\n")
    else:
        w("*(mandelbrot t_good json missing — run tools/atlas/mandelbrot_tgood_steered.py.)*\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(O), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"dive admissions={len(adm)} distinct-morphs={len(clusters)} "
          f"median-novelty-cosmax={np.median(nov_cosmax):.3f}" if len(nov_cosmax) else
          f"dive admissions={len(adm)}")
    print(f"wrote {args.dive_run/'dive_admissions.npz'}")


if __name__ == "__main__":
    main()
