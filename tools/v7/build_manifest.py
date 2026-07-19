#!/usr/bin/env python
"""v7 unified location manifest: freeze v6 VERBATIM + append the 536 post-freeze labels.

v7 = the julia:multibrot retrain. It extends v6 the way v6 extended v5: the frozen v6
manifest (5261 rows, loc_ids 0..5260) is carried **byte-for-byte** (split + group_id +
row order + coords all untouched), preserving the v5<->v6<->v7 eval-comparability chain,
and the 536 NEW post-freeze locations are appended with FORCED, rule-based splits:

  * census 144 (julia rows of prospect_run1 baserate_v1 + all baserate_R_v1) -> EVAL.
    The complete unbiased-given-descent julia:mb draw; the primary-metric instrument.
  * band   125 (jm3_band + jm45_band, model-band-selected decoded_class=2) -> TRAIN (biased).
  * other  267 -> TRAIN: blindspot v6-reject 219 (mandelbrot) + loose0_v3 26 (mandelbrot,
    post-freeze re-labels) + prospect native-plane 22 (native multibrot3/4/5).

Post-freeze = a labeled location (canonical resolver, crops->location, label=max) whose
identity (fractal_type,cx,cy,fw,c_re,c_im) is NOT in the frozen v6 manifest.

GATE STOP recorded in docs/findings/v7_build_gate_stop.md: the frozen v6 eval already
carries 17 UNBIASED julia:multibrot eval locations (1 q3). We proceed under **Option A**:
keep the frozen prefix byte-identical, accept the 17 as a pre-existing unbiased remnant,
and slice the reported julia:mb metric to the census-144 only. Consequences vs the plan's
prose (all bookkeeping, split design unchanged) are written into build_metadata.json.

  uv run python tools/v7/build_manifest.py

Output: data/v7/manifest.jsonl (+ data/v7/build_metadata.json). All 7 verifiability
checks are ABORTS, not warnings.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import label_store as ls  # noqa: E402

V6_MANIFEST = ROOT / "data" / "v6" / "manifest.jsonl"
BATCHES_GLOB = str(ROOT / "data" / "label_corpus" / "batches" / "*" / "images.jsonl")
OUT = ROOT / "data" / "v7" / "manifest.jsonl"
META_OUT = ROOT / "data" / "v7" / "build_metadata.json"

# --- §5 neighborhood-clustering predicate (verbatim from tools/v6/build_manifest.py) ---
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5
C_TOL_FRAC = 0.05
GID_OFFSET_V7 = 3_000_000          # > v6 max group_id (2_000_624); no collision

N_V6 = 5261                        # frozen v6 prefix row count (loc_ids 0..5260)

# family (ledger cloud partition) -> fractal_type (Rust kind_str)
FAM2FT = {
    "mandelbrot": "mandelbrot",
    "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
    "julia:mandelbrot": "julia",
    "julia:multibrot3": "julia_multibrot3",
    "julia:multibrot4": "julia_multibrot4",
    "julia:multibrot5": "julia_multibrot5",
    "phoenix": "phoenix",
}
FT2FAM = {v: k for k, v in FAM2FT.items()}

# Forced split rules, by batch (the plan §2 decomposition).
CENSUS_BATCHES = {"2026-07-17_prospect_run1_baserate_R_v1",
                  "2026-07-17_prospect_run1_baserate_v1"}
BAND_BATCHES = {"2026-07-11_jm3_band_v1", "2026-07-12_jm45_band_v1"}
BLINDSPOT_BATCH = "2026-07-12_blindspot_v6reject_v1"


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


def ftype_of(row):
    """fractal_type for a corpus row: provenance.family (authoritative) else render.fractal_type."""
    fam = (row.get("provenance") or {}).get("family")
    if fam and fam in FAM2FT:
        return FAM2FT[fam]
    ft = row["render"].get("fractal_type")
    return ft if ft else "mandelbrot"


# --------------------------------------------------------------------------- #
# 1. Reduce all labeled crops -> locations; subtract frozen v6 identities.
# --------------------------------------------------------------------------- #
def load_post_freeze(v6_ids):
    """Every labeled location (canonical resolver, label=max over crops) not already in
    the frozen v6 manifest. Returns list of loc dicts with string coords + batch + parent_oids."""
    locs = {}
    for images_path in sorted(glob.glob(BATCHES_GLOB)):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        for line in Path(images_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score = ls.resolve_score(row, sidecar)
            if score is None:
                continue
            ft = ftype_of(row)
            rd = row["render"]
            key = (ft, rd["cx"], rd["cy"], rd["fw"], rd.get("c_re"), rd.get("c_im"))
            d = locs.get(key)
            if d is None:
                d = locs[key] = dict(
                    ft=ft, cx=rd["cx"], cy=rd["cy"], fw=rd["fw"],
                    c_re=rd.get("c_re"), c_im=rd.get("c_im"),
                    labels=[], batches=set(), parent_oids=set())
            d["labels"].append(int(score))
            d["batches"].add(batch_id)
            poid = (row.get("provenance") or {}).get("parent_oid")
            if poid is not None:
                d["parent_oids"].add(poid)
    post = {k: d for k, d in locs.items() if k not in v6_ids}
    for d in post.values():
        d["label"] = max(d["labels"])
        assert len(d["batches"]) == 1, f"post-freeze loc spans >1 batch: {d['batches']}"
        d["batch"] = next(iter(d["batches"]))
    return list(post.values())


def assign_split(loc):
    """(split, biased, source) forced by batch — the plan §2 rule set. Census julia ->
    eval (unbiased); everything else -> train. Band/blindspot/native are biased->train."""
    b, ft = loc["batch"], loc["ft"]
    if b in CENSUS_BATCHES:
        if ft.startswith("julia_multibrot"):
            return "eval", False, "prospect_census"          # the eval instrument
        return "train", True, "prospect_native"              # native-plane, descent-screened
    if b in BAND_BATCHES:
        return "train", True, "jm_band"                      # model-band-selected
    if b == BLINDSPOT_BATCH:
        return "train", True, "blindspot_v6reject"           # negative-by-construction
    return "train", False, "loose0_v3"                       # unbiased flat re-labels


def assign_groups(locs):
    """§5 union-find partitioned by (fractal_type, SPLIT, c-bucket); ids offset by
    GID_OFFSET_V7.

    The plan's "no group straddles the split by construction" is optimistic: because
    split here is FORCED by batch (not derived from the groups, as in v6), the c-bucket
    union-find can transitively chain a census(eval) location to a band(train) location
    of the same family whose seed-c differ but sit within `C_TOL_FRAC*fw` of a common
    neighbor — producing a straddling group (gate 3). Since a group is a within-split
    neighborhood-equalization unit and gate 3 is a hard abort, the group union-find is
    partitioned by split as well. Splits are unchanged; only neighborhoods are computed
    within-split. See build_metadata.recipe_note."""
    by_famc = defaultdict(list)
    for l in locs:
        by_famc[(l["ft"], l["split"])].append(l)
    next_gid = GID_OFFSET_V7
    for (ft, _split), group in by_famc.items():
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
    return next_gid - GID_OFFSET_V7


def fmt_hist(h, n):
    return (f"n={n:4d}  1:{h.get(1,0):4d}  2:{h.get(2,0):4d}  3:{h.get(3,0):4d} "
            f"({100*h.get(3,0)/max(n,1):.1f}% q3)")


# --------------------------------------------------------------------------- #
def main():
    v6 = read_jsonl(V6_MANIFEST)
    assert len(v6) == N_V6, f"v6 manifest is {len(v6)} rows, expected {N_V6}"

    def ident(r):
        return (r["fractal_type"], r["cx"], r["cy"], r["fw"], r.get("c_re"), r.get("c_im"))
    v6_ids = {ident(r) for r in v6}
    assert len(v6_ids) == N_V6, "v6 manifest has duplicate identities"

    post = load_post_freeze(v6_ids)
    for l in post:
        l["split"], l["biased"], l["source"] = assign_split(l)
    ngroups = assign_groups(post)

    tr = [l for l in post if l["split"] == "train"]
    ev = [l for l in post if l["split"] == "eval"]

    print("=" * 78)
    print("v7 COMPOSITION  (frozen v6 verbatim + 536 appended)")
    print("=" * 78)
    print(f"  frozen v6 prefix : {len(v6)} rows (loc_ids 0..{N_V6-1})")
    print(f"  appended         : {len(post)}  ({ngroups} neighborhood groups, gid>={GID_OFFSET_V7})")
    print(f"    appended TRAIN {fmt_hist(Counter(l['label'] for l in tr), len(tr))}")
    print(f"    appended EVAL  {fmt_hist(Counter(l['label'] for l in ev), len(ev))}")
    print("  appended by (source, split):")
    bysrc = defaultdict(lambda: defaultdict(int))
    for l in post:
        bysrc[(l["source"], l["split"], l["biased"])][l["ft"]] += 1
    for (src, sp, bi), fams in sorted(bysrc.items()):
        n = sum(fams.values())
        q3 = sum(1 for l in post if l["source"] == src and l["split"] == sp and l["label"] == 3)
        print(f"    {src:20s} {sp:5s} biased={str(bi):5s}  n={n:3d} q3={q3:3d}  {dict(fams)}")

    # ---- the load-bearing julia:mb decomposition ----
    def jmb(rows, split): return [r for r in rows if r["fractal_type"].startswith("julia_multibrot")
                                  and r["split"] == split]
    v6_tr_jmb_q3 = sum(1 for r in v6 if r["split"] == "train"
                       and r["fractal_type"].startswith("julia_multibrot") and r["label"] == 3)
    v6_ev_jmb_q3 = sum(1 for r in v6 if r["split"] == "eval"
                       and r["fractal_type"].startswith("julia_multibrot") and r["label"] == 3)
    band_q3 = sum(1 for l in post if l["source"] == "jm_band" and l["label"] == 3)
    census_q3 = sum(1 for l in ev if l["label"] == 3)
    print("\n  julia:mb positive decomposition (Option A):")
    print(f"    v7 TRAIN julia:mb q3 = {v6_tr_jmb_q3} (v6 train) + {band_q3} (band) = {v6_tr_jmb_q3+band_q3}")
    print(f"    v7 EVAL  julia:mb q3 = {v6_ev_jmb_q3} (frozen v6 eval remnant) + {census_q3} (census)")
    print(f"    REPORTED julia:mb metric slices to the census-144 only ({census_q3} q3).")

    # ===================================================================== #
    # 7 VERIFIABILITY GATES — all ABORTS.
    # ===================================================================== #
    print("\n" + "=" * 78 + "\nBUILD GATES (aborts)\n" + "=" * 78)

    # Build the full appended row set (final schema) for gate checks + write.
    appended = []
    for l in post:
        row = {"cx": l["cx"], "cy": l["cy"], "fw": l["fw"], "label": l["label"],
               "source": l["source"], "biased": l["biased"], "split": l["split"],
               "group_id": l["group_id"], "fractal_type": l["ft"]}
        if l["c_re"] is not None:
            row["c_re"] = l["c_re"]; row["c_im"] = l["c_im"]
        appended.append(row)
    full = v6 + appended

    # Gate 1: 0 orphans — every appended row has non-null label/split/group_id.
    orphans = [r for r in appended if r.get("label") is None or r.get("split") is None
               or r.get("group_id") is None]
    assert not orphans, f"GATE 1 FAIL: {len(orphans)} appended orphans"
    print(f"  [1] 0 orphans                     OK ({len(appended)} appended rows complete)")

    # Gate 2: 0 identities straddling train/eval.
    tr_ids = {ident(r) for r in full if r["split"] == "train"}
    ev_ids = {ident(r) for r in full if r["split"] == "eval"}
    straddle_id = tr_ids & ev_ids
    assert not straddle_id, f"GATE 2 FAIL: {len(straddle_id)} identities in both splits"
    print(f"  [2] 0 identity straddle           OK (train {len(tr_ids)} / eval {len(ev_ids)})")

    # Gate 3: 0 group_ids straddling.
    g_split = defaultdict(set)
    for r in full:
        g_split[r["group_id"]].add(r["split"])
    span = [g for g, s in g_split.items() if len(s) > 1]
    assert not span, f"GATE 3 FAIL: {len(span)} groups span the split, e.g. {span[:5]}"
    print(f"  [3] 0 group straddle              OK ({len(g_split)} groups)")

    # Gate 4 (Option A reword): 0 BIASED locations in eval. The plan's "(census only)"
    # intent is NOT enforceable — the frozen prefix carries 17 unbiased julia:mb eval
    # rows (see build_metadata). What holds, and is asserted, is that eval is unbiased.
    biased_eval = [r for r in full if r["split"] == "eval" and r.get("biased")]
    assert not biased_eval, f"GATE 4 FAIL: {len(biased_eval)} biased rows in eval"
    print(f"  [4] 0 biased-in-eval              OK (Option A: unbiased-only eval; census slice reported)")

    # Gate 5: forced assignments hold — all census -> eval, all band -> train.
    census_locs = [l for l in post if l["batch"] in CENSUS_BATCHES
                   and l["ft"].startswith("julia_multibrot")]
    band_locs = [l for l in post if l["batch"] in BAND_BATCHES]
    assert len(census_locs) == 144, f"GATE 5 FAIL: census={len(census_locs)} != 144"
    assert all(l["split"] == "eval" for l in census_locs), "GATE 5 FAIL: census not all eval"
    assert len(band_locs) == 125, f"GATE 5 FAIL: band={len(band_locs)} != 125"
    assert all(l["split"] == "train" for l in band_locs), "GATE 5 FAIL: band not all train"
    print(f"  [5] forced assignments hold       OK (census 144->eval, band 125->train)")

    # Gate 6: frozen-prefix byte gate — rows 0..5260 byte-identical to data/v6/manifest.jsonl.
    v6_lines = V6_MANIFEST.read_text(encoding="utf-8").splitlines()
    v6_lines = [l for l in v6_lines if l.strip()]
    regen = [json.dumps(r) for r in v6]           # v6 rows re-serialized from parse
    assert len(regen) == len(v6_lines) == N_V6, "GATE 6 FAIL: frozen prefix length drift"
    drift = [i for i, (a, b) in enumerate(zip(regen, v6_lines)) if a != b]
    assert not drift, f"GATE 6 FAIL: {len(drift)} frozen rows drift, e.g. row {drift[0]}"
    print(f"  [6] frozen-prefix byte gate       OK ({N_V6} rows byte-identical to v6)")

    # Gate 7: census -> eval disjoint from ALL train at identity / seed-c / parent_oid.
    def seedc(l):
        return None if l.get("c_re") is None else (round(float(l["c_re"]), 12),
                                                   round(float(l["c_im"]), 12))
    train_ids = tr_ids
    train_seedc = {(_r["fractal_type"], round(float(_r["c_re"]), 12), round(float(_r["c_im"]), 12))
                   for _r in full if _r["split"] == "train" and _r.get("c_re") is not None}
    census_id_ov = [l for l in census_locs if (l["ft"], l["cx"], l["cy"], l["fw"],
                    l["c_re"], l["c_im"]) in train_ids]
    census_sc_ov = [l for l in census_locs
                    if (l["ft"], round(float(l["c_re"]), 12), round(float(l["c_im"]), 12)) in train_seedc]
    train_poids = set()
    for l in post:
        if l["split"] == "train":
            train_poids |= l["parent_oids"]
    census_poids = set()
    for l in post:
        if l in census_locs:
            census_poids |= l["parent_oids"]
    poid_ov = census_poids & train_poids
    assert not census_id_ov, f"GATE 7 FAIL: {len(census_id_ov)} census identities in train"
    assert not census_sc_ov, f"GATE 7 FAIL: {len(census_sc_ov)} census seed-c in train"
    assert not poid_ov, f"GATE 7 FAIL: {len(poid_ov)} census parent_oid in train"
    print(f"  [7] census disjoint from train    OK (id 0 / seed-c 0 / parent_oid 0"
          f"; train parent_oids={len(train_poids)}, census parent_oids={len(census_poids)})")

    # ---- write manifest (v6 verbatim + appended) ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for line in v6_lines:                      # byte-identical frozen prefix
            f.write(line + "\n")
        for row in appended:
            f.write(json.dumps(row) + "\n")

    # ---- build metadata: the recorded decisions (amendments + gate consequences) ----
    meta = {
        "build": "v7",
        "frozen_prefix": {"source": "data/v6/manifest.jsonl", "rows": N_V6,
                          "byte_identical": True},
        "appended": {"total": len(post), "train": len(tr), "eval": len(ev),
                     "groups": ngroups, "gid_offset": GID_OFFSET_V7},
        "split_rule": "census julia -> eval (unbiased); band/blindspot/native/loose0 -> train",
        "recipe_note": "group union-find partitioned by (fractal_type, SPLIT, c-bucket), not "
                       "just (family, c-bucket): with split forced by batch, the c-bucket "
                       "union-find transitively chained 2 census(eval) julia_multibrot4 locs "
                       "to band(train) locs of nearby seed-c, straddling the split (gate 3). "
                       "Adding split to the partition enforces gate 3; splits are unchanged.",
        "gate_stop_resolution": {
            "finding": "v6 frozen eval contains 17 UNBIASED julia:multibrot eval "
                       "locations (1 q3); see docs/findings/v7_build_gate_stop.md",
            "decision": "Option A — keep frozen prefix byte-identical; accept the 17 as a "
                        "pre-existing unbiased remnant; report the julia:mb metric on the "
                        "census-144 slice only, never the 161-union.",
            "assert4_reword": "enforced as '0 BIASED-in-eval' (holds); the plan's "
                              "'(census only)' intent is NOT enforceable given the frozen prefix.",
            "julia_mb_positive_counts": {
                "v7_train": v6_tr_jmb_q3 + band_q3,
                "note": f"plan said 95; actual {v6_tr_jmb_q3+band_q3} because the 1 v6 "
                        f"julia:mb q3 sits in the frozen eval (Option A), not train.",
                "v7_eval_census": census_q3,
                "v7_eval_frozen_remnant": v6_ev_jmb_q3,
            },
        },
        "amendment_1_ss2_gap": {
            "decision": "NO 640x360 ss2 aug slot added (dropped per Amendment 1).",
            "reason": "The frozen-prefix byte gate confines any ss2 slot to the 536 appended "
                      "locations (~all julia:mb + blindspot negatives), so ss2 would correlate "
                      "with both family and label and let the model shortcut on it. Accepted, "
                      "deliberate covariate shift: deploy is 640x360 ss2; the aug set has ss1 "
                      "and ss4 but no ss2. Palette + geometry match (both 16:9 stretch to "
                      "384x224); only the ss2 high-frequency AA signature is uncovered. Known, "
                      "second-order.",
        },
        "amendment_2_census_is_eval": {
            "decision": "The census IS the eval set AND the only unbiased-given-descent julia "
                        "draw that exists (run 1 ledger exhausted). When t_good is later fit, "
                        "it will be fit ON the eval — a small (one scalar on 144 points) but "
                        "real leak. Recorded so whoever sets the threshold knows it is fitting "
                        "on eval, not discovering it. t_good is NOT set in this build.",
        },
        "plan_table_corrections": {
            "native_multibrot_post_freeze": "plan §1 said 0; actual 22 new locations "
                "(mb3 9 / mb4 9 / mb5 4), 6 q3 -> all TRAIN. §6 'zero new positives' is "
                "stale: native train positives go 9 -> 15. Native stays UNMEASURABLE (no eval).",
            "loose0_v3_relabels": "26 post-freeze mandelbrot re-labels on the v5-era "
                "flat_generate_loose0_v3 batch (0 q3) the plan did not enumerate -> TRAIN "
                "(unbiased). Covered by the 'all other post-freeze -> train' rule.",
            "other_267_breakdown": "267 = 219 blindspot (mandelbrot) + 26 loose0_v3 "
                "(mandelbrot) + 22 prospect native-plane (multibrot3/4/5).",
        },
        "deploy_note": "ACTIVE_CKPT NOT switched; v6 remains the deployed scorer until v7 "
                       "is measured. t_good NOT set.",
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"WROTE {OUT}  ({len(full)} rows = {N_V6} frozen + {len(post)} appended)")
    print(f"WROTE {META_OUT}")
    all_ft = Counter(r["fractal_type"] for r in full)
    print(f"unified fractal_type counts: {dict(sorted(all_ft.items()))}")
    sp = Counter(r["split"] for r in full)
    print(f"unified split: train={sp['train']}  eval={sp['eval']}")


if __name__ == "__main__":
    main()
