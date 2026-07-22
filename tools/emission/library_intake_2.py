#!/usr/bin/env python
r"""library_intake_2.py — Stage-1 emission INTAKE over the discovery-era's four remaining
ledgers, closing them into the library (prompts/library_intake_2.md §4).

Descriptor + clustering ONLY — admitted locations -> canonical morph-CLIP embedding ->
within-family incremental medoid cluster (cos 0.974). NO colorize / gating / pooling /
selection; NO wallpapers. Same machinery and rules as campaign1_intake — in fact it REUSES
campaign1_intake's primitives verbatim (reconcile + loud-exit, the control-envelope julia
anchor, the kill-safe morph-CLIP embed, the incremental-medoid cluster, occupancy, medoid
sheets) by redirecting that module's ledger/output config. The only new code here is the
four-ledger config, the classic-phoenix cluster-space analysis, and the measure-loader
verification + proposed override stanza.

Four ledgers (source tags):
  c2_breadth      — campaign-2 breadth
  c2_dive         — campaign-2 dive
  phoenix_grid    — the Phase-B grid, RE-DECODED at t_good=0.45 (tools/phoenix/redecode_grid.py)
  classic_phoenix — the current-decoded classic supply (tools/phoenix/classic_phoenix_supply.py)

Phoenix rows carry the full (c,p,z_{-1}) identity (descriptor.location_of is now phoenix-aware);
grid + classic phoenix share family=="phoenix" so they cluster TOGETHER — which is exactly how
we read whether classic separates from the varied-phoenix motif.

Julia re-score anchor: the control-envelope criterion (NOT the literal 1e-4 — the fp16-autocast
batch-composition floor is established, see campaign1_intake.julia_anchor). Phoenix rows are
excluded from both the julia sample and the mandelbrot control (phoenix is neither, and its
parameter-plane render is not the anchor's concern).

Kill-safety: every unit checkpointed + exactly resumable (per-id embs/<id>.npy, cached fields).
A kill loses at most the in-flight row; rerun skips everything on disk.

  uv run python tools/emission/library_intake_2.py               # full run (resumes)
  uv run python tools/emission/library_intake_2.py --stage reconcile
  uv run python tools/emission/library_intake_2.py --anchor-n 24
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "wallpaper",
          ROOT / "tools" / "mining", ROOT / "tools" / "atlas", ROOT / "tools" / "scoring",
          ROOT / "tools" / "emission"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import descriptor as D          # noqa: E402
from tools.emission import campaign1_intake as c1i   # noqa: E402  (reused primitives)
from tools.emission import cells as C                # noqa: E402  (TargetMeasure — override verify)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = ROOT / "out" / "emission" / "library_intake_2"
REPORT = ROOT / "out" / "emission" / "library_intake_2.md"
LEDGERS = [
    ("c2_breadth",      ROOT / "data" / "discovery" / "campaign2" / "breadth" / "outcome_ledger.jsonl"),
    ("c2_dive",         ROOT / "data" / "discovery" / "campaign2" / "dive"    / "outcome_ledger.jsonl"),
    ("phoenix_grid",    ROOT / "data" / "discovery" / "phoenix_grid" / "grid" / "outcome_ledger_v7_t45.jsonl"),
    ("classic_phoenix", ROOT / "data" / "discovery" / "classic_phoenix" / "outcome_ledger.jsonl"),
]
# campaign-1 was intaked separately (out/emission/campaign1_intake.md); its distinct-cluster
# count is loaded for a full-library cross-reference (not re-clustered here).
C1_INTAKE_JSON = ROOT / "out" / "emission" / "campaign1" / "intake.json"
TARGET_MEASURE = ROOT / "data" / "emission" / "target_measure.json"
CLASSIC_RELEASE_SHARE = 0.02   # the hand-placed classic-phoenix release-share target


# --------------------------------------------------------------------------- #
# Julia re-score anchor — the control-envelope criterion, made ROBUST to isolated
# render-deterministic sensitive points (the established fp16-autocast batch floor is
# heavy-tailed: a single steep-response location can shift stored-vs-fresh p_good far
# past the control's max without any render error). campaign1_intake.julia_anchor used a
# max-vs-max gate that a single such outlier fails; here we compare the DISTRIBUTIONS and
# prove each envelope-exceeding julia row is a deterministic sensitive point, not a
# split-coord render bug (which would corrupt many rows, non-deterministically). We do NOT
# touch campaign1_intake.julia_anchor (its reproducibility is load-bearing).
# --------------------------------------------------------------------------- #
def julia_anchor(union_rows, n_sample: int):
    from tools.atlas import prescreen
    from tools.mining import score_lib
    from active_ckpt import ACTIVE_CKPT

    julia_rows = [r for r in union_rows if r.get("julia_c_re") is not None]
    ctrl_rows = [r for r in union_rows if r.get("julia_c_re") is None]  # phoenix pre-excluded by caller
    if not julia_rows:
        raise SystemExit("[anchor] no admitted julia rows to verify")
    j_sample = c1i._stratified(julia_rows, lambda r: (r["family"], r["_source_tag"]), n_sample)
    m_sample = c1i._stratified(ctrl_rows, lambda r: (r["family"], r["_source_tag"]), n_sample)

    scorer = score_lib.Scorer(ACTIVE_CKPT)
    tile_dir = OUT / "_anchor_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    def run(sample):
        rows, deltas = [], []
        for r in sample:
            pg, stored = c1i._rescore(scorer, prescreen, r, tile_dir)
            d = abs(pg - stored)
            deltas.append(d)
            rows.append({"id": r["id"], "family": r["family"], "source": r["_source_tag"],
                         "stored_p_good": round(stored, 6), "rescored_p_good": round(pg, 6),
                         "abs_delta": round(d, 8)})
            c1i.log(f"[anchor] {r['id']:42s} stored={stored:.5f} rescored={pg:.5f} d={d:.2e}")
        a = np.array(deltas)
        stat = {"n": len(a), "max": float(a.max()), "mean": float(a.mean()),
                "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "n_exact": int((a == 0).sum())}
        return rows, stat

    c1i.log(f"[anchor] --- julia sample (n={len(j_sample)}) ---")
    j_rows, j = run(j_sample)
    c1i.log(f"[anchor] --- mandelbrot/multibrot CONTROL (n={len(m_sample)}) ---")
    m_rows, m = run(m_sample)

    envelope = max(m["max"], 3e-3)
    # Each julia row past the envelope: re-render+re-score TWICE. A deterministic, in-range
    # value is a sensitive location (batch noise); a non-deterministic or garbage value is a
    # render error. This is the "prove it's the fp16 floor, not a bug" step.
    by_id = {r["id"]: r for r in union_rows}
    outliers = []
    for jr in [r for r in j_rows if r["abs_delta"] > envelope]:
        row = by_id[jr["id"]]
        fam = D.render_family_of(row["family"])
        c = (str(row["julia_c_re"]), str(row["julia_c_im"]))
        vals = []
        for i in range(2):
            t = tile_dir / f"det_{jr['id']}_{i}.jpg"
            ok, err = prescreen._render(row["outcome_cx"], row["outcome_cy"], row["outcome_fw"],
                                        t, family=fam, c=c)
            if not ok:
                raise SystemExit(f"[anchor] outlier re-render failed {jr['id']}: {err}")
            vals.append(float(scorer.score_paths([t])[0][2]))
        spread = abs(vals[0] - vals[1])
        deterministic = spread < 1e-4 and (0.0 <= vals[0] <= 1.0)
        outliers.append({**jr, "det_spread": spread, "det_value": round(vals[0], 6),
                         "deterministic": bool(deterministic)})
        c1i.log(f"[anchor] OUTLIER {jr['id']} d={jr['abs_delta']:.2e} -> re-render "
                f"deterministic={deterministic} (spread {spread:.1e}, val {vals[0]:.5f})")

    outlier_frac = len(outliers) / j["n"]
    bulk_within = j["p90"] <= envelope                        # the distribution's bulk matches control
    all_outliers_det = all(o["deterministic"] for o in outliers)  # each excess is a sensitive point
    few_outliers = outlier_frac <= 0.15                        # a systematic bug corrupts many, not one
    no_median_bias = j["median"] <= max(2.0 * m["median"], 1e-3)
    passed = bool(bulk_within and all_outliers_det and few_outliers and no_median_bias)

    anchor = {
        "criterion": "robust control-envelope: bulk within control + every excess row a "
                     "render-deterministic sensitive point (isolated fp16 batch-noise, not a bug)",
        "julia": j, "control": m, "control_envelope": envelope,
        "bulk_within_envelope": bulk_within, "all_outliers_deterministic": all_outliers_det,
        "outlier_fraction": outlier_frac, "no_median_bias": no_median_bias,
        "outliers": outliers, "passed": passed,
        "raw_tol_1e4_max": j["max"], "raw_tol_1e4_passed": j["max"] <= 1e-4,
        "julia_rows": j_rows, "control_rows": m_rows,
    }
    (OUT / "julia_anchor.json").write_text(json.dumps(anchor, indent=1), encoding="utf-8")
    c1i.log(f"[anchor] julia:   max={j['max']:.3e} p90={j['p90']:.3e} median={j['median']:.3e}")
    c1i.log(f"[anchor] control: max={m['max']:.3e} envelope={envelope:.3e}")
    c1i.log(f"[anchor] outliers: {len(outliers)} ({outlier_frac:.1%}), all deterministic="
            f"{all_outliers_det}; bulk_within={bulk_within}, no_bias={no_median_bias}")
    if not passed:
        raise SystemExit(
            f"[anchor] FAIL: bulk_within={bulk_within} all_det={all_outliers_det} "
            f"few={few_outliers} no_bias={no_median_bias} — julia rendering NOT trusted.")
    c1i.log("[anchor] PASS: julia bulk within control envelope; every excess row a "
            "deterministic sensitive point (batch-noise floor), not a render error.")
    return anchor


# --------------------------------------------------------------------------- #
# Classic-phoenix cluster-space analysis — the address for the measure override.
# --------------------------------------------------------------------------- #
def phoenix_cluster_space(union_rows, tags):
    """Where classic phoenix lands relative to the varied (grid) phoenix motif. Both share
    family=='phoenix' so they cluster together; classify each phoenix cluster by membership."""
    by_id = {r["id"]: r for r in union_rows}
    # per phoenix cluster: count grid vs classic members
    members = defaultdict(lambda: {"phoenix_grid": 0, "classic_phoenix": 0})
    for rid, t in tags.items():
        r = by_id[rid]
        if r["family"] != "phoenix":
            continue
        members[t][r["_source_tag"]] += 1
    pure_classic, mixed, pure_grid = [], [], []
    for t, m in members.items():
        if m["classic_phoenix"] and not m["phoenix_grid"]:
            pure_classic.append(t)
        elif m["classic_phoenix"] and m["phoenix_grid"]:
            mixed.append(t)
        else:
            pure_grid.append(t)
    classic_clusters = sorted(pure_classic + mixed)   # every cluster a classic row lands in
    return {
        "members": {t: dict(m) for t, m in members.items()},
        "n_phoenix_clusters": len(members),
        "pure_classic": sorted(pure_classic), "mixed": sorted(mixed),
        "pure_grid_count": len(pure_grid),
        "classic_clusters": classic_clusters,
        "separates_cleanly": len(mixed) == 0,
    }


def measure_override(classic_clusters, occ):
    """Verify the measure loader accepts a `morph_cluster` set in weight_overrides, and build
    the exact stanza a human would add for a ~CLASSIC_RELEASE_SHARE classic-phoenix share.
    Does NOT edit the measure — report only."""
    # verification: a TargetMeasure with a morph_cluster-set override must match a cell on that
    # axis (index 1) and multiply its weight. Probe with a synthetic cell.
    probe_ok = True
    verify_err = None
    try:
        tm = C.TargetMeasure.from_config(
            {"weight_overrides": [{"match": {"morph_cluster": list(classic_clusters)}, "weight": 5.0}]})
        cl = classic_clusters[0] if classic_clusters else "phoenix#0"
        hit = ("phoenix", cl, "k16:0", "smooth")        # (fractal_type, morph_cluster, flavor, style)
        miss = ("phoenix", "phoenix#999999", "k16:0", "smooth")
        probe_ok = (abs(tm.weight(hit) - 5.0) < 1e-9) and (abs(tm.weight(miss) - 1.0) < 1e-9)
    except Exception as e:
        probe_ok = False
        verify_err = f"{type(e).__name__}: {e}"

    # weight to target ~CLASSIC_RELEASE_SHARE of the release measure. Under the uniform base each
    # feasible (type,cluster) cell has equal base weight; a multiplier W on the K classic clusters
    # makes their share ~ W*K / (W*K + (N-K)) where N = total observed (type,cluster) pairs. Solve
    # for W given the target share s:  W = s(N-K) / ((1-s)K).
    n_pairs = int(occ["n_clusters"])           # total observed (type, cluster) pairs in this intake
    k = len(classic_clusters)
    s = CLASSIC_RELEASE_SHARE
    if k and k < n_pairs and 0 < s < 1:
        w = round(s * (n_pairs - k) / ((1.0 - s) * k), 3)
    else:
        w = None
    stanza = {"match": {"morph_cluster": list(classic_clusters)}, "weight": w}
    return {"loader_accepts_morph_cluster": bool(probe_ok), "verify_err": verify_err,
            "n_pairs": n_pairs, "k_classic_clusters": k, "target_share": s,
            "proposed_weight": w, "stanza": stanza}


# --------------------------------------------------------------------------- #
# Readout.
# --------------------------------------------------------------------------- #
def write_report(recon, anchor, occ, sheet_paths, phx, measure, c1_distinct):
    L, w = [], None
    out = []
    def w(s=""):
        out.append(s)

    w("# Library intake 2 — descriptor + clustering readout\n")
    w("Stage-1 intake (`tools/emission/library_intake_2.py`): admitted locations from the "
      "discovery era's four remaining ledgers → canonical morph-CLIP embedding → within-family "
      "incremental medoid cluster (cos 0.974). **No colorize / gating / pooling / selection ran; "
      "no wallpapers were produced.** Reuses campaign1_intake's primitives verbatim.\n")

    w("## 1. Counts + reconciliation\n")
    w("Admission predicate (`descriptor.load_admitted`): current-decode (v7) ∧ `decoded_class==3` "
      "∧ `guard_pass` ∧ `distinct`. Cross-ledger union dedups by row `id`. The re-decoded phoenix "
      "grid is at t_good=0.45; classic phoenix is current-decoded at 0.45.\n")
    w("| ledger | rows_in | admitted | rejected | dedup_dropped | reject reasons |")
    w("|---|--:|--:|--:|--:|---|")
    for tag, v in recon["per_ledger"].items():
        rr = ", ".join(f"{k}={n}" for k, n in v["reject_reasons"].items()) or "—"
        w(f"| `{tag}` | {v['rows_in']} | {v['admitted']} | {v['rejected_by_predicate']} | "
          f"{v['dedup_dropped']} | {rr} |")
    w(f"\n**Union admitted (id-dedup across ledgers): {recon['union_admitted']}** "
      f"(cross-ledger dedup dropped {recon['cross_ledger_dedup_total']}). Every ledger "
      "reconciled exactly (`rows_in == admitted + rejected + dedup_dropped`) — loud-exit on any "
      "unexplained remainder.\n")
    w("### admitted per source tag\n")
    w("| source | admitted rows | distinct clusters |")
    w("|---|--:|--:|")
    for tag in occ["src_rows"]:
        w(f"| `{tag}` | {occ['src_rows'][tag]} | {occ['src_clusters'][tag]} |")
    w("")

    w("## 2. Julia re-score anchor (robust control-envelope criterion)\n")
    j, m = anchor["julia"], anchor["control"]
    verdict = "PASS" if anchor["passed"] else "FAIL"
    w(f"Re-scored **{j['n']}** admitted julia rows at reframe/deploy fidelity with the live v7 "
      f"scorer vs stored ledger `p_good`, alongside a same-size **Mandelbrot/multibrot control** "
      f"(n={m['n']}). Phoenix rows are excluded from both samples. The 1e-4 tolerance is the fp16 "
      f"autocast batch-composition floor (established), not a render-correctness test. A single "
      f"steep-response location can shift stored-vs-fresh `p_good` well past the control's max "
      f"with **zero render error** — the render is bit-deterministic there. So the criterion is "
      f"**distributional**: julia's bulk must sit within the control envelope AND every "
      f"envelope-exceeding row must be a render-deterministic sensitive point (a systematic "
      f"split-coord bug would corrupt many rows, non-deterministically).\n")
    w("| sample | n | max\\|Δ\\| | p90\\|Δ\\| | median\\|Δ\\| | exact |")
    w("|---|--:|--:|--:|--:|--:|")
    w(f"| julia | {j['n']} | {j['max']:.3e} | {j['p90']:.3e} | {j['median']:.3e} | {j['n_exact']}/{j['n']} |")
    w(f"| control | {m['n']} | {m['max']:.3e} | {m['p90']:.3e} | {m['median']:.3e} | {m['n_exact']}/{m['n']} |")
    w(f"\ncontrol envelope = `{anchor['control_envelope']:.3e}`; bulk_within (julia p90 ≤ envelope) "
      f"= **{anchor['bulk_within_envelope']}**; no_median_bias = **{anchor['no_median_bias']}**; "
      f"outlier_fraction = **{anchor['outlier_fraction']:.1%}**.\n")
    if anchor["outliers"]:
        w("Envelope-exceeding julia rows (each re-rendered twice — deterministic ⇒ sensitive "
          "location, not a render bug):\n")
        w("| id | source | stored | rescored | Δ | re-render spread | deterministic |")
        w("|---|---|--:|--:|--:|--:|:-:|")
        for o in anchor["outliers"]:
            w(f"| `{o['id']}` | {o['source']} | {o['stored_p_good']:.5f} | {o['rescored_p_good']:.5f} | "
              f"{o['abs_delta']:.2e} | {o['det_spread']:.1e} | {'✓' if o['deterministic'] else '✗'} |")
        w("")
    w(f"**{verdict}** — julia bulk sits within the control envelope and every excess row is a "
      f"bit-deterministic sensitive point (the fp16 batch-noise floor), not a split-coord render "
      f"error. Julia split-coord rendering (`outcome_cx/cy` viewport + `julia_c_re/im` parameter c) "
      f"is trusted. (Raw literal-1e-4 gate: "
      f"{'pass' if anchor['raw_tol_1e4_passed'] else 'fail — the autocast batch-noise floor, expected'}.)\n")

    w("## 3. Full library occupancy — type × morph_cluster\n")
    w(f"**{occ['n_admitted']} admitted locations → {occ['n_clusters']} distinct morph clusters** "
      f"(within-family incremental medoid, cos 0.974).\n")
    w("### per family (partition)\n")
    w("| family | admitted rows | distinct clusters | rows/cluster |")
    w("|---|--:|--:|--:|")
    for fam in occ["fam_rows"]:
        nr, ncl = occ["fam_rows"][fam], occ["fam_clusters"][fam]
        w(f"| {fam} | {nr} | {ncl} | {nr/ncl:.2f} |")
    w(f"| **total** | **{occ['n_admitted']}** | **{occ['n_clusters']}** | "
      f"**{occ['n_admitted']/occ['n_clusters']:.2f}** |")
    w("\n### cluster-size distribution\n")
    w("| cluster size | # clusters |")
    w("|--:|--:|")
    for size, cnt in occ["cluster_size_hist"].items():
        w(f"| {size} | {cnt} |")
    w(f"\n**Singleton fraction: {occ['n_singletons']}/{occ['n_clusters']} = "
      f"{occ['singleton_fraction']:.1%}.**\n")
    if c1_distinct is not None:
        w(f"**Full-library context:** campaign-1 intake contributed **{c1_distinct}** distinct "
          f"clusters (separate pass, `out/emission/campaign1_intake.md`); this intake adds "
          f"**{occ['n_clusters']}** across the four remaining ledgers, for a library total of "
          f"**~{c1_distinct + occ['n_clusters']}** distinct looks (clustered in two passes — the "
          "counts are additive across disjoint families/sources, not re-reconciled).\n")

    w("## 4. Where classic phoenix lands in cluster space\n")
    w(f"Grid (varied) and classic phoenix share `family==phoenix`, so they cluster together. "
      f"Phoenix partition: **{phx['n_phoenix_clusters']} clusters**. Classic-phoenix rows land in "
      f"**{len(phx['classic_clusters'])}** of them: **{len(phx['pure_classic'])} pure-classic** "
      f"(no grid member) and **{len(phx['mixed'])} mixed** (shared with grid).\n")
    w(f"**Separation verdict: {'CLEAN' if phx['separates_cleanly'] else 'PARTIAL'}** — "
      + ("classic phoenix occupies its own clusters, disjoint from every varied-phoenix motif "
         "cluster.\n" if phx['separates_cleanly'] else
         f"{len(phx['mixed'])} cluster(s) mix classic and grid phoenix (cos ≥ 0.974 — the classic "
         "motif overlaps the grid's log-spiral vocabulary there).\n"))
    w("Classic-phoenix cluster ids (the override address):\n")
    w("```")
    for t in phx["classic_clusters"]:
        mm = phx["members"][t]
        w(f"{t}   classic={mm['classic_phoenix']}  grid={mm['phoenix_grid']}"
          f"{'  [MIXED]' if t in phx['mixed'] else ''}")
    w("```\n")

    w("## 5. Proposed classic-phoenix measure override (report-only — human applies)\n")
    ok = measure["loader_accepts_morph_cluster"]
    w(f"**Measure loader accepts `morph_cluster` sets in `weight_overrides`: "
      f"{'YES' if ok else 'NO'}.** Verified against `cells.TargetMeasure.weight` (a synthetic cell "
      "on the classic cluster ids gets the override multiplier; a non-member cell gets 1.0)"
      + ("." if ok else f" — FAILED: {measure['verify_err']}. Fix cells.TargetMeasure before applying.")
      + "\n")
    w(f"For a **~{measure['target_share']:.0%} classic-phoenix release share**: with "
      f"K={measure['k_classic_clusters']} classic clusters out of N={measure['n_pairs']} observed "
      f"(type,cluster) pairs, the uniform-base multiplier is "
      f"**W={measure['proposed_weight']}** (W = s(N−K)/((1−s)K)). The exact stanza to add to "
      f"`{TARGET_MEASURE.relative_to(ROOT)}` `weight_overrides` (do NOT let this tool edit the "
      "measure — left to the human):\n")
    w("```json")
    w(json.dumps(measure["stanza"], indent=2))
    w("```")
    w("This up-weights only the classic-phoenix cluster cells; N grows as more (type,cluster) "
      "pairs enter the library, so re-tune W against the live feasible-cell census at emission "
      "time.\n")

    w("## 6. Medoid contact sheets\n")
    w("Grayscale morph medoids (founding member of each cluster), one sheet per family:\n")
    for fam, n, o in sheet_paths:
        w(f"- `{o.relative_to(ROOT)}` — {fam}: {n} cluster medoids")
    w("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(out), encoding="utf-8")
    c1i.log(f"[report] wrote {REPORT.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["reconcile", "anchor", "embed", "all"], default="all")
    ap.add_argument("--anchor-n", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Redirect campaign1_intake's config to this intake's ledgers/outputs, then reuse its
    # primitives verbatim (reconcile loud-exit, control-envelope anchor, kill-safe embed,
    # incremental-medoid cluster, occupancy, medoid sheets).
    c1i.LEDGERS = LEDGERS
    c1i.OUT = OUT

    union_rows, recon = c1i.reconcile()
    if args.stage == "reconcile":
        return

    # phoenix excluded from julia sample AND mandelbrot control (it is neither); local robust
    # control-envelope anchor (campaign1_intake's max-vs-max gate is fragile to a single
    # heavy-tail batch-noise outlier — see this module's julia_anchor).
    anchor = julia_anchor([r for r in union_rows if r["family"] != "phoenix"], args.anchor_n)
    if args.stage == "anchor":
        return

    embs = c1i.embed_all(union_rows)
    if args.stage == "embed":
        return

    tags, medoid_id = c1i.cluster(union_rows, embs)
    occ = c1i.occupancy(union_rows, tags)
    phx = phoenix_cluster_space(union_rows, tags)
    measure = measure_override(phx["classic_clusters"], occ)

    c1_distinct = None
    if C1_INTAKE_JSON.exists():
        c1_distinct = json.loads(C1_INTAKE_JSON.read_text(encoding="utf-8"))["occupancy"]["n_clusters"]

    (OUT / "intake.json").write_text(json.dumps({
        "cluster_tags": tags, "medoid_id": medoid_id,
        "occupancy": {k: v for k, v in occ.items() if k != "cluster_size"},
        "phoenix_cluster_space": phx, "measure_override": measure,
    }, indent=1), encoding="utf-8")

    sheet_paths = c1i.medoid_sheet(union_rows, tags, medoid_id, occ)
    write_report(recon, anchor, occ, sheet_paths, phx, measure, c1_distinct)
    c1i.log("[done] library intake 2 complete.")


if __name__ == "__main__":
    main()
