"""v1-vs-v2 candidate-diversity before/after table.

Reads the two diagnostic JSONs written by diversity_diagnostic.py
(data/queries/diagnostics/coldstart_v{1,2}_diversity.json) and emits the before/after
comparison the coldstart_v2 regeneration is judged on: min-ΔE distribution,
effective-distinct by query type, near-dup count with the sampler-dup vs
perceptual-collapse split, and the worst-N lists. Read-only.

    uv run python tools/queries/compare_v1_v2_diversity.py
"""
from __future__ import annotations

import json
from pathlib import Path

DIAG = Path("data/queries/diagnostics")
TYPES = ["palette", "param", "joint"]


def load(v):
    return json.loads((DIAG / f"{v}_diversity.json").read_text())


def fmt(a, b, pct=False):
    def s(x):
        return f"{x*100:5.1f}%" if pct else f"{x:6.2f}"
    return f"{s(a):>8} -> {s(b):>8}   (Δ {(b-a)*100:+5.1f}pp)" if pct else \
           f"{s(a):>8} -> {s(b):>8}   (Δ {b-a:+6.2f})"


def main():
    v1, v2 = load("coldstart_v1"), load("coldstart_v2")

    print("=" * 78)
    print("  coldstart  v1  ->  v2   candidate-diversity before/after")
    print("=" * 78)
    print(f"queries: v1={v1['n_queries']} v2={v2['n_queries']}   "
          f"type_counts v1={v1['type_counts']} v2={v2['type_counts']}")

    # (1) min pairwise ΔE distribution
    print("\n--- (1) min-pairwise-ΔE distribution (per-query closest pair) ---")
    p1, p2 = v1["min_pair_de"]["percentiles"], v2["min_pair_de"]["percentiles"]
    for p in ("0", "5", "10", "25", "50", "75", "90", "100"):
        print(f"  p{p:<3} {fmt(p1[p], p2[p])}")
    print("  histogram [lo,hi) counts  (v1 / v2):")
    e = v1["min_pair_de"]["histogram"]["edges"]
    c1 = v1["min_pair_de"]["histogram"]["counts"]
    c2 = v2["min_pair_de"]["histogram"]["counts"]
    for i in range(len(c1)):
        hi = e[i + 1] if i + 1 < len(e) else "inf"
        print(f"    [{str(e[i]):>4},{str(hi):>4}) : {c1[i]:>4} / {c2[i]:>4}")

    # (2) effective-distinct by type & threshold
    print("\n--- (2) effective-distinct (mean of 6) by query type & ΔE threshold ---")
    for thr in ("2.0", "5.0", "10.0"):
        print(f"  ΔE<{thr}:")
        for t in TYPES + ["overall"]:
            if t == "overall":
                a = v1["eff_distinct_overall"][thr]["mean"]
                b = v2["eff_distinct_overall"][thr]["mean"]
            else:
                a = v1["eff_distinct_by_type"][t][thr]["mean"]
                b = v2["eff_distinct_by_type"][t][thr]["mean"]
            print(f"    {t:>8}: {fmt(a, b)}")

    # (3) near-dup recipe-join split
    print("\n--- (3) near-dup pairs (mean ΔE < NEAR_DUP_THRESH), recipe-join split ---")
    for scope in ("overall",) + tuple(TYPES):
        s1 = v1["near_dup_split_overall"] if scope == "overall" else v1["near_dup_split_by_type"][scope]
        s2 = v2["near_dup_split_overall"] if scope == "overall" else v2["near_dup_split_by_type"][scope]
        print(f"  {scope:>8}: sampler_dup {s1['sampler_dup']:>3} -> {s2['sampler_dup']:>3}   "
              f"perceptual_collapse {s1['perceptual_collapse']:>3} -> {s2['perceptual_collapse']:>3}   "
              f"total {s1['total']:>3} -> {s2['total']:>3}")
    print(f"  queries touched by a near-dup: v1={v1['queries_with_near_dup']} "
          f"v2={v2['queries_with_near_dup']}")

    # headline: sampler-dup -> 0 and residual perceptual collapse
    sd2 = v2["near_dup_split_overall"]["sampler_dup"]
    pc2 = v2["near_dup_split_overall"]["perceptual_collapse"]
    print(f"\n  >> sampler-dup pairs v2 = {sd2}  ({'PASS: 0' if sd2 == 0 else 'FAIL: expected 0'})")
    print(f"  >> residual perceptual-collapse pairs v2 = {pc2}  "
          f"(deferred low-DR-location routing; report only)")

    # (4) worst-N
    print("\n--- (4) worst queries by min-pair ΔE ---")
    for v, tag in ((v1, "v1"), (v2, "v2")):
        w = v["worst_queries"][:10]
        print(f"  {tag} worst-10: " + ", ".join(
            f"{q['qid']}({q['qtype'][:4]},{q['min_pair_de']:.1f})" for q in w))


if __name__ == "__main__":
    main()
