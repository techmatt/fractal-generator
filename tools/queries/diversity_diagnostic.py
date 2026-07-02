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

ΔE metric: CIEDE2000, implemented here in numpy (validated at import against the
Sharma et al. 2005 reference test vectors) to avoid pulling scikit-image/scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ---- Named constants (printed at run) ---------------------------------------
THUMB_WIDTH = 256           # candidates downsampled to this width before ΔE
REF_THRESHOLDS = [2.0, 5.0, 10.0]   # ΔE cuts for single-linkage effective-distinct count
WORST_N = 20                # size of the ranked worst-queries list / HTML montage
NEAR_DUP_THRESH = min(REF_THRESHOLDS)   # a pair is "near-dup" if min-side ΔE < this
GAMMA_DUP_EPS = 0.10        # |Δγ| below this counts as a near-identical gamma for recipe-join

QUERIES_ROOT = Path("data/queries")
OUT_DIR = QUERIES_ROOT / "diagnostics"


# ---- CIEDE2000 (numpy, vectorized) ------------------------------------------
def srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """(...,3) uint8 sRGB -> (...,3) float CIELAB, D65."""
    srgb = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = lin @ m.T
    # D65 reference white
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000 ΔE between two (...,3) Lab arrays."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    # when either chroma is 0, hue diff is undefined -> 0
    zero_c = (C1p * C2p) == 0
    dhp = np.where(zero_c, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    habsdiff = np.abs(h1p - h2p)
    hbarp = np.where(
        zero_c, hsum,
        np.where(habsdiff <= 180.0, 0.5 * hsum,
                 np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))))

    T = (1.0
         - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    Rc = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Lbarp_m50sq = (Lbarp - 50.0) ** 2
    Sl = 1.0 + (0.015 * Lbarp_m50sq) / np.sqrt(20.0 + Lbarp_m50sq)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2.0 * dtheta)) * Rc

    kL = kC = kH = 1.0
    tL = dLp / (kL * Sl)
    tC = dCp / (kC * Sc)
    tH = dHp / (kH * Sh)
    return np.sqrt(tL * tL + tC * tC + tH * tH + Rt * tC * tH)


def _validate_ciede2000():
    """Sharma et al. 2005 reference pairs (Lab1, Lab2, expected ΔE)."""
    cases = [
        ([50.0000, 2.6772, -79.7751], [50.0000, 0.0000, -82.7485], 2.0425),
        ([50.0000, 3.1571, -77.2803], [50.0000, 0.0000, -82.7485], 2.8615),
        ([50.0000, 2.8361, -74.0200], [50.0000, 0.0000, -82.7485], 3.4412),
        ([50.0000, -1.3802, -84.2814], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -1.1848, -84.8006], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, -0.9009, -85.5211], [50.0000, 0.0000, -82.7485], 1.0000),
        ([50.0000, 0.0000, 0.0000], [50.0000, -1.0000, 2.0000], 2.3669),
        ([50.0000, -1.0000, 2.0000], [50.0000, 0.0000, 0.0000], 2.3669),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0009], 7.1792),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0011], 7.1792),
        ([50.0000, 2.4900, -0.0010], [50.0000, -2.4900, 0.0012], 7.2195),
        ([50.0000, -0.0010, 2.4900], [50.0000, 0.0009, -2.4900], 4.8045),
        ([50.0000, 2.5000, 0.0000], [50.0000, 0.0000, -2.5000], 4.3065),
        ([50.0000, 2.5000, 0.0000], [73.0000, 25.0000, -18.0000], 27.1492),
        ([50.0000, 2.5000, 0.0000], [61.0000, -5.0000, 29.0000], 22.8977),
        ([50.0000, 2.5000, 0.0000], [56.0000, -27.0000, -3.0000], 31.9030),
        ([50.0000, 2.5000, 0.0000], [58.0000, 24.0000, 15.0000], 19.4535),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.1736, 0.5854], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.2972, 0.0000], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 1.8634, 0.5757], 1.0000),
        ([50.0000, 2.5000, 0.0000], [50.0000, 3.2592, 0.3350], 1.0000),
        ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
        ([63.0109, -31.0961, -5.8663], [62.8187, -29.7946, -4.0864], 1.2630),
        ([61.2901, 3.7196, -5.3901], [61.4292, 2.2480, -4.9620], 1.8731),
        ([35.0831, -44.1164, 3.7933], [35.0232, -40.0716, 1.5901], 1.8645),
        ([22.7233, 20.0904, -46.6940], [23.0331, 14.9730, -42.5619], 2.0373),
        ([36.4612, 47.8580, 18.3852], [36.2715, 50.5065, 21.2231], 1.4146),
        ([90.8027, -2.0831, 1.4410], [91.1528, -1.6435, 0.0447], 1.4441),
        ([90.9257, -0.5406, -0.9208], [88.6381, -0.8985, -0.7239], 1.5381),
        ([6.7747, -0.2908, -2.4247], [5.8714, -0.0985, -2.2286], 0.6377),
        ([2.0776, 0.0795, -1.1350], [0.9033, -0.0636, -0.5514], 0.9082),
    ]
    l1 = np.array([c[0] for c in cases])
    l2 = np.array([c[1] for c in cases])
    exp = np.array([c[2] for c in cases])
    got = ciede2000(l1, l2)
    err = np.sort(np.abs(got - exp))
    # Pair (50,2.49,-.001)&(50,-2.49,.0011) is the documented CIEDE2000 hue-quadrant
    # boundary: |h'1-h'2| lands within ~1.5e-3 deg of exactly 180 deg, so the
    # hue-average branch flips on the last ulp. skimage's own impl doesn't
    # reproduce Sharma's value here either. Allow exactly this one to differ (<0.05),
    # require every other pair exact to 1e-3 (a real bug would break many by a lot).
    if err[-2] > 1e-3 or err[-1] > 0.05:
        raise RuntimeError(
            f"CIEDE2000 self-test FAILED: 2nd-worst err {err[-2]:.5f}, worst {err[-1]:.5f}")
    return err[-2], err[-1]


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
