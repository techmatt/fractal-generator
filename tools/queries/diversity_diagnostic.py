#!/usr/bin/env python3
"""Read-only pixel-space candidate-diversity diagnostic for a query batch.

For each query, the 6 candidates share the *exact same field/framing* (only the
coloring recipe differs), so the PNGs are pixel-aligned and per-pixel color
difference is directly meaningful with no registration or re-rendering.

We downsample each candidate, convert to CIELAB, compute the 15 pairwise
per-pixel color-difference fields (ΔE), and summarize each pair by mean and p90.
Per-query we emit the 6x6 mean-ΔE matrix, the min pairwise ΔE, and an
effective-distinct count (single-linkage clusters) at several reference
thresholds. Near-duplicate pairs are recipe-joined to split *sampler param-dup*
(it drew two ~identical recipe tuples) from *perceptual collapse* (distinct
recipe the field just doesn't resolve).

STRICTLY READ-ONLY except for the report files under data/queries/diagnostics/.

ΔE metric: CIEDE2000, from the shared `color_metrics` module (validated at import
against the Sharma et al. 2005 reference test vectors) to avoid pulling
scikit-image/scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from color_metrics import THUMB_WIDTH, ciede2000, srgb_to_lab, _validate_ciede2000

# ---- Named constants (printed at run) ---------------------------------------
REF_THRESHOLDS = [2.0, 5.0, 10.0]   # ΔE cuts for single-linkage effective-distinct count
WORST_N = 20                # size of the ranked worst-queries list / HTML montage
NEAR_DUP_THRESH = min(REF_THRESHOLDS)   # a pair is "near-dup" if min-side ΔE < this
GAMMA_DUP_EPS = 0.10        # |Δγ| below this counts as a near-identical gamma for recipe-join

QUERIES_ROOT = Path("data/queries")
OUT_DIR = QUERIES_ROOT / "diagnostics"


# ---- per-query processing ---------------------------------------------------
def load_thumb_lab(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    th = max(1, round(THUMB_WIDTH * h / w))
    img = img.resize((THUMB_WIDTH, th), Image.BOX)
    return srgb_to_lab(np.asarray(img))


def single_linkage_clusters(dmat: np.ndarray, thresh: float) -> int:
    """Count connected components where an edge exists iff pairwise dist < thresh."""
    n = dmat.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] < thresh:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(n)})


def recipe_tuple(cand: dict) -> dict:
    return {
        "palette": cand.get("palette"),
        "reverse": bool(cand.get("reverse")),
        "gamma": float(cand.get("gamma")),
        "n_cycles": cand.get("n_cycles"),
        "phase": cand.get("phase"),
        "log_premap": cand.get("log_premap"),
    }


def classify_pair(ra: dict, rb: dict) -> str:
    """sampler_dup vs perceptual_collapse for a flagged near-dup pair."""
    same_discrete = (
        ra["palette"] == rb["palette"]
        and ra["reverse"] == rb["reverse"]
        and ra["n_cycles"] == rb["n_cycles"]
        and ra["phase"] == rb["phase"]
        and ra["log_premap"] == rb["log_premap"]
    )
    close_gamma = abs(ra["gamma"] - rb["gamma"]) <= GAMMA_DUP_EPS
    return "sampler_dup" if (same_discrete and close_gamma) else "perceptual_collapse"


def main():
    ap = argparse.ArgumentParser(description="Candidate-diversity diagnostic (read-only).")
    ap.add_argument("--batch", default="coldstart_v2",
                    help="batch id under data/queries/ (default coldstart_v2)")
    args = ap.parse_args()
    batch_id = args.batch
    batch_dir = QUERIES_ROOT / batch_id

    print("=== candidate-diversity diagnostic (read-only) ===")
    print(f"THUMB_WIDTH={THUMB_WIDTH}  REF_THRESHOLDS={REF_THRESHOLDS}  "
          f"WORST_N={WORST_N}  NEAR_DUP_THRESH={NEAR_DUP_THRESH}  "
          f"GAMMA_DUP_EPS={GAMMA_DUP_EPS}")
    err2nd, errworst = _validate_ciede2000()
    print(f"CIEDE2000 self-test PASS (30/31 Sharma refs exact, 2nd-worst err "
          f"{err2nd:.2e}; 1 hue-quadrant boundary pair off by {errworst:.3f}, documented)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_paths = sorted((batch_dir / "records").glob("q*.json"))
    print(f"batch={batch_dir}  queries={len(rec_paths)}")

    per_query = []
    flagged_pairs = []   # near-dup pair records for the recipe-join split
    for rp in rec_paths:
        rec = json.loads(rp.read_text())
        qid = rec["query_id"]
        qtype = rec["query_type"]
        cands = rec["candidates"]
        assert len(cands) == 6, f"{qid} has {len(cands)} candidates"
        labs = [load_thumb_lab(batch_dir / c["image"]) for c in cands]

        mean_mat = np.zeros((6, 6))
        p90_mat = np.zeros((6, 6))
        for i in range(6):
            for j in range(i + 1, 6):
                de = ciede2000(labs[i], labs[j]).ravel()
                mean_mat[i, j] = mean_mat[j, i] = float(de.mean())
                p90_mat[i, j] = p90_mat[j, i] = float(np.percentile(de, 90))

        iu = np.triu_indices(6, 1)
        pair_means = mean_mat[iu]
        min_idx = int(np.argmin(pair_means))
        i_min, j_min = iu[0][min_idx], iu[1][min_idx]
        min_de = float(pair_means[min_idx])

        eff = {str(t): single_linkage_clusters(mean_mat, t) for t in REF_THRESHOLDS}

        # recipe-join for every pair below the near-dup threshold
        recs = [recipe_tuple(c) for c in cands]
        pair_flags = []
        for k in range(len(iu[0])):
            i, j = int(iu[0][k]), int(iu[1][k])
            if mean_mat[i, j] < NEAR_DUP_THRESH:
                cls = classify_pair(recs[i], recs[j])
                fp = {
                    "qid": qid, "qtype": qtype, "i": i, "j": j,
                    "mean_de": float(mean_mat[i, j]), "p90_de": float(p90_mat[i, j]),
                    "class": cls,
                    "recipe_i": recs[i], "recipe_j": recs[j],
                    "dgamma": abs(recs[i]["gamma"] - recs[j]["gamma"]),
                    "img_i": cands[i]["image"], "img_j": cands[j]["image"],
                }
                pair_flags.append(fp)
                flagged_pairs.append(fp)

        per_query.append({
            "qid": qid, "qtype": qtype,
            "family": rec["location"]["family"],
            "mean_matrix": mean_mat.tolist(),
            "p90_matrix": p90_mat.tolist(),
            "min_pair_de": min_de,
            "min_pair": [int(i_min), int(j_min)],
            "min_pair_images": [cands[i_min]["image"], cands[j_min]["image"]],
            "mean_of_pair_means": float(pair_means.mean()),
            "eff_distinct": eff,
            "n_flagged_pairs": len(pair_flags),
        })

    # ---- batch-level aggregation --------------------------------------------
    types = ["palette", "param", "joint"]
    min_des = np.array([q["min_pair_de"] for q in per_query])

    def pct(a, ps=(0, 5, 10, 25, 50, 75, 90, 100)):
        return {str(p): float(np.percentile(a, p)) for p in ps}

    # (1) min pairwise ΔE distribution
    hist_edges = [0, 1, 2, 3, 5, 8, 12, 20, 1e9]
    hist_counts, _ = np.histogram(min_des, bins=hist_edges)

    # (2) effective-distinct by type & threshold
    eff_by_type = {}
    for t in types:
        qs = [q for q in per_query if q["qtype"] == t]
        eff_by_type[t] = {"n_queries": len(qs)}
        for thr in REF_THRESHOLDS:
            vals = np.array([q["eff_distinct"][str(thr)] for q in qs], dtype=float)
            eff_by_type[t][str(thr)] = {
                "mean": float(vals.mean()) if len(vals) else None,
                "dist": {str(k): int((vals == k).sum()) for k in range(1, 7)},
            }
    eff_overall = {}
    for thr in REF_THRESHOLDS:
        vals = np.array([q["eff_distinct"][str(thr)] for q in per_query], dtype=float)
        eff_overall[str(thr)] = {
            "mean": float(vals.mean()),
            "dist": {str(k): int((vals == k).sum()) for k in range(1, 7)},
        }

    # (3) near-dup recipe-join split, overall and by type
    def split_counts(flist):
        s = sum(1 for f in flist if f["class"] == "sampler_dup")
        p = sum(1 for f in flist if f["class"] == "perceptual_collapse")
        return {"sampler_dup": s, "perceptual_collapse": p, "total": len(flist)}

    split_overall = split_counts(flagged_pairs)
    split_by_type = {t: split_counts([f for f in flagged_pairs if f["qtype"] == t]) for t in types}
    # queries touched by >=1 near-dup pair
    q_with_dup = {t: len({f["qid"] for f in flagged_pairs if f["qtype"] == t}) for t in types}
    q_with_dup["overall"] = len({f["qid"] for f in flagged_pairs})

    # (4) worst queries
    worst = sorted(per_query, key=lambda q: q["min_pair_de"])[:WORST_N]
    worst_list = [{
        "qid": q["qid"], "qtype": q["qtype"], "family": q["family"],
        "min_pair_de": q["min_pair_de"], "min_pair": q["min_pair"],
        "min_pair_images": q["min_pair_images"],
        "eff_distinct": q["eff_distinct"],
    } for q in worst]

    report = {
        "constants": {
            "THUMB_WIDTH": THUMB_WIDTH, "REF_THRESHOLDS": REF_THRESHOLDS,
            "WORST_N": WORST_N, "NEAR_DUP_THRESH": NEAR_DUP_THRESH,
            "GAMMA_DUP_EPS": GAMMA_DUP_EPS,
        },
        "de_metric": "CIEDE2000 (numpy, validated vs Sharma et al. 2005 refs)",
        "batch": str(batch_dir),
        "n_queries": len(per_query),
        "type_counts": {t: sum(1 for q in per_query if q["qtype"] == t) for t in types},
        "min_pair_de": {
            "percentiles": pct(min_des),
            "histogram": {"edges": hist_edges[:-1] + ["inf"],
                          "counts": hist_counts.tolist()},
        },
        "eff_distinct_overall": eff_overall,
        "eff_distinct_by_type": eff_by_type,
        "near_dup_split_overall": split_overall,
        "near_dup_split_by_type": split_by_type,
        "queries_with_near_dup": q_with_dup,
        "worst_queries": worst_list,
        "per_query": per_query,
        "flagged_pairs": flagged_pairs,
    }

    json_path = OUT_DIR / f"{batch_id}_diversity.json"
    json_path.write_text(json.dumps(report, indent=2))

    csv_path = OUT_DIR / f"{batch_id}_diversity_per_query.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "qtype", "family", "min_pair_de", "mean_of_pair_means",
                    "eff_distinct_2", "eff_distinct_5", "eff_distinct_10",
                    "n_flagged_pairs", "min_pair_i", "min_pair_j"])
        for q in per_query:
            w.writerow([q["qid"], q["qtype"], q["family"],
                        f"{q['min_pair_de']:.4f}", f"{q['mean_of_pair_means']:.4f}",
                        q["eff_distinct"]["2.0"], q["eff_distinct"]["5.0"],
                        q["eff_distinct"]["10.0"], q["n_flagged_pairs"],
                        q["min_pair"][0], q["min_pair"][1]])

    # (5) worst-N HTML montage (references existing files by relative path)
    html_path = OUT_DIR / f"{batch_id}_worst.html"
    rows = []
    for q in worst_list:
        ii, jj = q["min_pair"]
        imgs = "".join(
            f'<img src="../{batch_id}/{q["min_pair_images"][k]}" '
            f'style="height:120px;margin:2px;border:2px solid #888" '
            f'title="cand {[ii, jj][k]}">'
            for k in range(2))
        rows.append(
            f'<tr><td>{q["qid"]}<br><small>{q["qtype"]}/{q["family"]}</small></td>'
            f'<td>minΔE={q["min_pair_de"]:.2f}<br>cands {ii}&{jj}<br>'
            f'eff@2={q["eff_distinct"]["2.0"]}</td>'
            f'<td>{imgs}</td></tr>')
    html = (
        f"<html><head><meta charset='utf-8'><title>{batch_id} worst-diversity queries</title>"
        "<style>body{font-family:sans-serif;background:#222;color:#ddd}"
        "table{border-collapse:collapse}td{border:1px solid #444;padding:6px;vertical-align:top}"
        "</style></head><body>"
        f"<h2>Worst {WORST_N} queries by min pairwise CIEDE2000 (the closest candidate pair)</h2>"
        f"<p>Constants: THUMB_WIDTH={THUMB_WIDTH}, REF_THRESHOLDS={REF_THRESHOLDS}, "
        f"NEAR_DUP_THRESH={NEAR_DUP_THRESH}. Images are the two most-similar candidates.</p>"
        "<table><tr><th>query</th><th>metric</th><th>closest pair</th></tr>"
        + "".join(rows) + "</table></body></html>")
    html_path.write_text(html, encoding="utf-8")

    # ---- console summary ----------------------------------------------------
    print("\n--- (1) min pairwise ΔE distribution across", len(per_query), "queries ---")
    for p in (0, 5, 10, 25, 50, 75, 90, 100):
        print(f"  p{p:<3} = {np.percentile(min_des, p):6.2f}")
    print("  histogram (min ΔE bins):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist_counts):
        hi_s = "inf" if hi > 1e8 else f"{hi:g}"
        print(f"    [{lo:>4g}, {hi_s:>4}) : {c}")

    print("\n--- (2) effective-distinct count by query type ---")
    for thr in REF_THRESHOLDS:
        print(f"  @ΔE<{thr:g}: overall mean={eff_overall[str(thr)]['mean']:.2f}  "
              + "  ".join(f"{t} mean={eff_by_type[t][str(thr)]['mean']:.2f}" for t in types))

    print("\n--- (3) near-dup recipe-join split (pairs with mean ΔE <",
          NEAR_DUP_THRESH, ") ---")
    print(f"  overall: {split_overall}  (queries touched: {q_with_dup['overall']})")
    for t in types:
        print(f"  {t:>7}: {split_by_type[t]}  (queries touched: {q_with_dup[t]})")

    print(f"\n--- (4) worst {WORST_N} queries (lowest min pairwise ΔE) ---")
    for q in worst_list:
        print(f"  {q['qid']} {q['qtype']:>7}/{q['family']:<10} "
              f"minΔE={q['min_pair_de']:6.2f} pair={tuple(q['min_pair'])} "
              f"eff@2/5/10={q['eff_distinct']['2.0']}/{q['eff_distinct']['5.0']}/{q['eff_distinct']['10.0']}")

    print("\n--- report files ---")
    for p in (json_path, csv_path, html_path):
        print(" ", p)


if __name__ == "__main__":
    sys.exit(main())
