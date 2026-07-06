#!/usr/bin/env python
"""v6 unified location manifest: freeze v5 (Mandelbrot + J0 Julia) + fold gather_v6.

v6 extends v5 with the 2026-07-05_gather_v6 label batch — the first **multi-family**
harvest (mandelbrot, multibrot3/4/5, julia:{mandelbrot,multibrot3/4/5}, phoenix). The
frozen v5 assignment is carried VERBATIM (every v5 location keeps its split + group_id +
loc order) so the v5<->v6 eval compare stays clean; only the ~639 NEW gather_v6
locations get a fresh split.

  * v5 rows  : copied byte-for-byte from data/v5/manifest.jsonl (Mandelbrot loc_ids
               0..3621, Julia 3622..4621). Nothing touched.
  * gather_v6: crops -> base locations (label = max over the location's crops, the v4
               "there EXISTS a good rendering" semantics), keyed by
               (family, cx, cy, fw, c). Neighborhood group_id via the §5 union-find
               partitioned by (family, c) exactly as v5 partitions Julia; ids offset by
               GATHER_GID_OFFSET so they never collide with Mandelbrot (<2328) or Julia
               (1e6-range) groups. SPLIT: selection_role best/disagreement are model-
               selected (BIASED -> train-locked); random_eval is the only eval-eligible
               source. Among eligible groups: EVAL_FRAC=0.40, seed=0, stratified by group
               max-label. Assert location-disjoint + no group spans split + no biased in
               eval — the same guarantees v4/v5 enforce.

  uv run python tools/v6/build_manifest.py

Output: data/v6/manifest.jsonl (unified, build_plan-ready).
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V5_MANIFEST = ROOT / "data" / "v5" / "manifest.jsonl"
GATHER_BATCH = ROOT / "data" / "label_corpus" / "batches" / "2026-07-05_gather_v6" / "images.jsonl"
OUT = ROOT / "data" / "v6" / "manifest.jsonl"

# §5 neighborhood-clustering predicate (faithful to assemble.py / build_manifest.py).
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5
C_TOL_FRAC = 0.05                 # c-cluster tolerance (fixed-c partition), as v5
GATHER_GID_OFFSET = 2_000_000     # > Julia offset (1e6) + Julia group span; no collision

EVAL_FRAC = 0.40
SEED = 0
# selection_role -> biased (train-locked). best/disagreement are model-selected;
# random_eval is uniform mid-walk (the only unbiased, eval-eligible source).
BIASED_ROLES = {"best", "disagreement"}


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def cluster(cx, cy, fw):
    """§5 neighborhood union-find on (cx,cy,fw). Returns dense local group ids."""
    n = len(cx)
    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            ratio = fw[i] / fw[j]
            if ratio < SCALE_LO or ratio > SCALE_HI:
                continue
            tol = SHIFT_FRAC * min(fw[i], fw[j])
            dx, dy = cx[i] - cx[j], cy[i] - cy[j]
            if dx * dx + dy * dy <= tol * tol:
                uf.union(i, j)
    roots, out = {}, []
    for i in range(n):
        out.append(roots.setdefault(uf.find(i), len(roots)))
    return out


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt_hist(h, n):
    return (f"n={n:4d}  1:{h.get(1,0):4d} ({100*h.get(1,0)/max(n,1):4.1f}%)  "
            f"2:{h.get(2,0):4d} ({100*h.get(2,0)/max(n,1):4.1f}%)  "
            f"3:{h.get(3,0):4d} ({100*h.get(3,0)/max(n,1):4.1f}%)")


# --------------------------------------------------------------------------- #
# gather_v6 -> base locations
# --------------------------------------------------------------------------- #
def build_gather_locations(rows):
    """Reduce gather_v6 crops to base locations keyed by (family, cx, cy, fw, c)."""
    groups = defaultdict(list)
    for r in rows:
        rd, pv = r["render"], r["provenance"]
        fam = pv["family"]                       # e.g. julia:multibrot3
        ftype = rd["fractal_type"]               # e.g. julia_multibrot3 (Rust kind_str)
        key = (fam, rd["cx"], rd["cy"], rd["fw"], rd.get("c_re"), rd.get("c_im"))
        score = r.get("label", {}).get("score")
        groups[key].append((rd, pv, ftype, score))
    locs = []
    for (fam, cx, cy, fw, c_re, c_im), items in groups.items():
        labs = [it[3] for it in items if it[3] is not None]
        if not labs:
            continue                              # unlabeled location (none expected here)
        roles = {it[1].get("selection_role") for it in items}
        biased = any(rl in BIASED_ROLES for rl in roles)
        ftype = items[0][2]
        locs.append(dict(
            family=fam, fractal_type=ftype,
            cx=cx, cy=cy, fw=fw, c_re=c_re, c_im=c_im,
            label=max(labs), biased=biased, n_crops=len(items),
            roles=roles,
        ))
    return locs


def assign_groups(locs):
    """Neighborhood group_id per gather_v6 location: §5 union-find partitioned by
    (family, c-bucket) — the v5 Julia recipe, generalized to every family. c-bucket
    tolerance = C_TOL_FRAC * fw (families without a c collapse to one bucket)."""
    by_famc = defaultdict(list)
    for l in locs:
        by_famc[l["family"]].append(l)
    next_gid = GATHER_GID_OFFSET
    for fam, group in by_famc.items():
        # c-cluster within family (fixed-c partition); no-c families -> single bucket.
        has_c = group[0]["c_re"] is not None
        if has_c:
            fw0 = float(group[0]["fw"])
            ctol = C_TOL_FRAC * fw0
            cre = [float(l["c_re"]) for l in group]
            cim = [float(l["c_im"]) for l in group]
            uf = UF(len(group))
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    dx, dy = cre[i] - cre[j], cim[i] - cim[j]
                    if dx * dx + dy * dy <= ctol * ctol:
                        uf.union(i, j)
            buckets = defaultdict(list)
            for i, l in enumerate(group):
                buckets[uf.find(i)].append(l)
        else:
            buckets = {0: group}
        for cc in buckets.values():
            sub = cluster([float(l["cx"]) for l in cc],
                          [float(l["cy"]) for l in cc],
                          [float(l["fw"]) for l in cc])
            local = {}
            for l, k in zip(cc, sub):
                if k not in local:
                    local[k] = next_gid
                    next_gid += 1
                l["group_id"] = local[k]
    return next_gid - GATHER_GID_OFFSET


def split_gather(locs):
    """Group-level split: any group with a biased member -> TRAIN; among eligible
    (all-unbiased) groups, EVAL_FRAC eval, stratified by group max-label, seed=0."""
    g_members = defaultdict(list)
    for l in locs:
        g_members[l["group_id"]].append(l)
    biased_groups = {g for g, ms in g_members.items() if any(m["biased"] for m in ms)}
    eligible = [g for g in g_members if g not in biased_groups]
    strata = defaultdict(list)
    for g in eligible:
        strata[max(m["label"] for m in g_members[g])].append(g)
    rng = random.Random(SEED)
    eval_groups = set()
    for lbl in sorted(strata):
        gids = sorted(strata[lbl])
        rng.shuffle(gids)
        k = round(len(gids) * EVAL_FRAC)
        eval_groups.update(gids[:k])
    for l in locs:
        l["split"] = "eval" if l["group_id"] in eval_groups else "train"
    return g_members, biased_groups


# --------------------------------------------------------------------------- #
def main():
    v5 = read_jsonl(V5_MANIFEST)
    grows = read_jsonl(GATHER_BATCH)

    print("=" * 78)
    print("v6 COMPOSITION SUMMARY")
    print("=" * 78)

    # ---- frozen v5 portion ----
    v5_ft = Counter(r["fractal_type"] for r in v5)
    print(f"\n-- frozen v5 (carried verbatim): {len(v5)} locations --")
    print(f"   fractal_type: {dict(v5_ft)}")
    for ft in ("mandelbrot", "julia"):
        rr = [r for r in v5 if r["fractal_type"] == ft]
        print(f"     {ft:10s} {fmt_hist(Counter(r['label'] for r in rr), len(rr))}")

    # ---- gather_v6 crop-level composition ----
    print(f"\n-- 2026-07-05_gather_v6: {len(grows)} labeled crops --")
    fam_lab = defaultdict(Counter)
    role_ct = Counter()
    for r in grows:
        pv = r["provenance"]; s = r.get("label", {}).get("score")
        fam_lab[pv["family"]][s] += 1
        role_ct[pv.get("selection_role")] += 1
    print(f"   selection_role (crops): {dict(role_ct)}")
    print("   per-family crop label distribution (score 1/2/3):")
    for fam in sorted(fam_lab):
        c = fam_lab[fam]; n = sum(c.values())
        print(f"     {fam:20s} {fmt_hist(c, n)}")

    # ---- gather_v6 -> locations ----
    glocs = build_gather_locations(grows)
    ngroups = assign_groups(glocs)
    g_members, biased_groups = split_gather(glocs)

    print(f"\n-- gather_v6 base locations: {len(glocs)} (from {len(grows)} crops), "
          f"{ngroups} neighborhood groups --")
    fam_loc = defaultdict(list)
    for l in glocs:
        fam_loc[l["fractal_type"]].append(l)
    print("   per-family LOCATION label distribution + score-3 (positive) count:")
    SPARSE = {"multibrot5", "julia_multibrot3", "multibrot3"}
    for ft in sorted(fam_loc):
        ls = fam_loc[ft]
        h = Counter(l["label"] for l in ls)
        flag = "  <-- POSITIVE-SPARSE" if ft in SPARSE else ""
        print(f"     {ft:20s} {fmt_hist(h, len(ls))}  score3={h.get(3,0)}{flag}")
    tot3 = Counter(l["label"] for l in glocs)
    print(f"     {'COMBINED':20s} {fmt_hist(tot3, len(glocs))}")

    # ---- split summary ----
    print("\n" + "=" * 78)
    print(f"v6 SPLIT SUMMARY  (gather_v6 only; v5 frozen)  "
          f"EVAL_FRAC={EVAL_FRAC} seed={SEED}")
    print("=" * 78)
    tr = [l for l in glocs if l["split"] == "train"]
    ev = [l for l in glocs if l["split"] == "eval"]
    print(f"  gather_v6 biased-containing groups (forced TRAIN): {len(biased_groups)}")
    print(f"  gather_v6 train locations = {len(tr)}   eval locations = {len(ev)}")
    print(f"    TRAIN {fmt_hist(Counter(l['label'] for l in tr), len(tr))}")
    print(f"    EVAL  {fmt_hist(Counter(l['label'] for l in ev), len(ev))}")
    print("  per-family train/eval (locations):")
    for ft in sorted(fam_loc):
        t = sum(1 for l in fam_loc[ft] if l["split"] == "train")
        e = sum(1 for l in fam_loc[ft] if l["split"] == "eval")
        e3 = sum(1 for l in fam_loc[ft] if l["split"] == "eval" and l["label"] == 3)
        print(f"     {ft:20s} train={t:3d}  eval={e:3d}  (eval score3={e3})")

    # ---- asserts ----
    evb = [l for l in ev if l["biased"]]
    assert not evb, f"BIASED LEAK INTO EVAL: {len(evb)}"
    span = [g for g in g_members if len({l["split"] for l in g_members[g]}) > 1]
    assert not span, f"GROUP SPANS SPLIT: {span[:5]}"
    # location-disjoint (train vs eval) on the (family,coords,c) identity
    def ident(l): return (l["family"], l["cx"], l["cy"], l["fw"], l["c_re"], l["c_im"])
    tr_ids, ev_ids = {ident(l) for l in tr}, {ident(l) for l in ev}
    assert not (tr_ids & ev_ids), "LOCATION IN BOTH SPLITS"
    print(f"\n  [assert OK] biased-in-eval=0, groups-spanning-split=0, "
          f"train/eval location-disjoint ({len(tr_ids)}/{len(ev_ids)})")

    # ---- write unified manifest: v5 verbatim, then gather_v6 ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in v5:                                      # v5 rows byte-for-byte
            f.write(json.dumps(r) + "\n")
        for l in glocs:
            row = {
                "cx": l["cx"], "cy": l["cy"], "fw": l["fw"],
                "label": l["label"], "source": "gather_v6", "biased": l["biased"],
                "split": l["split"], "group_id": l["group_id"],
                "fractal_type": l["fractal_type"],
            }
            if l["c_re"] is not None:
                row["c_re"] = l["c_re"]; row["c_im"] = l["c_im"]
            f.write(json.dumps(row) + "\n")

    total = len(v5) + len(glocs)
    print(f"\n=== unified v6 manifest -> {OUT} ===")
    print(f"total rows: {total}  (v5 frozen {len(v5)} + gather_v6 {len(glocs)})")
    all_ft = Counter(r["fractal_type"] for r in v5)
    for l in glocs:
        all_ft[l["fractal_type"]] += 1
    print(f"unified fractal_type counts: {dict(sorted(all_ft.items()))}")


if __name__ == "__main__":
    main()
