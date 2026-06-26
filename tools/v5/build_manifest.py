#!/usr/bin/env python
"""v5 unified location manifest: freeze v4 Mandelbrot + fold J0 Julia labels.

Part 1 (merge) + Part 2 (split + dedup) of prompts/v5_unified_classifier_train.md.

  * Mandelbrot rows: copied VERBATIM from data/v4/manifest.jsonl (frozen split,
    frozen group_id) with `fractal_type:"mandelbrot"` stamped on. Nothing about
    the v4 Mandelbrot assignment is touched -> the v4<->v5 Mandelbrot-AP compare
    stays clean.
  * Julia rows: one per labeled, in-batch J0 rung. Recovers render params + c from
    the J0 batch (data/label_corpus/batches/julia_ladder_j0/images.jsonl), the hand
    label from labels/location_labels_julia_ladder_j0.json.
      - SPLIT inheritance: each Julia rung inherits its seed Mandelbrot location's
        split, matched by provenance.src_(cx,cy,fw) -> v4 manifest row. (A Julia
        child can NEVER land in eval while its parent neighborhood was in train.)
      - GROUP id (neighborhood equalization unit): the v4 §5 union-find partitioned
        by (fractal_type, c). Within a seed_group we first c-cluster (tol =
        0.05*src_fw, faithful to build_j0.dedup), then §5-cluster on (cx,cy,fw).
        Julia group ids are offset by JULIA_GID_OFFSET so they never collide with
        (or dedup against) Mandelbrot groups.

  uv run python tools/v5/build_manifest.py

Output: data/v5/manifest.jsonl (unified location manifest, build_plan-ready).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4_MANIFEST = ROOT / "data" / "v4" / "manifest.jsonl"
J0_BATCH = ROOT / "data" / "label_corpus" / "batches" / "julia_ladder_j0" / "images.jsonl"
J0_LABELS = ROOT / "labels" / "location_labels_julia_ladder_j0.json"
OUT = ROOT / "data" / "v5" / "manifest.jsonl"

# §5 union-find predicate constants (faithful to tools/julia_ladder/build_j0.py / assemble.py)
SHIFT_FRAC = 0.5
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5
C_TOL_FRAC = 0.05
JULIA_GID_OFFSET = 1_000_000   # > max Mandelbrot group_id (2327); no cross-type collision


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster(cx, cy, fw):
    """§5 neighborhood union-find on (cx,cy,fw). Returns dense group ids."""
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
        r = uf.find(i)
        out.append(roots.setdefault(r, len(roots)))
    return out


def main() -> None:
    man = [json.loads(l) for l in V4_MANIFEST.read_text().splitlines() if l.strip()]
    split_idx = {(r["cx"], r["cy"], r["fw"]): r["split"] for r in man}
    labels = json.loads(J0_LABELS.read_text())
    batch = [json.loads(l) for l in J0_BATCH.read_text().splitlines() if l.strip()]

    # --- join labels x batch ---
    by_id = {r["image_id"]: r for r in batch}
    julia = []          # collected Julia location dicts (pre group-id)
    skipped = []
    for iid, lab in labels.items():
        row = by_id.get(iid)
        if row is None or lab not in (1, 2, 3):
            skipped.append(iid)
            continue
        rd, pv = row["render"], row["provenance"]
        key = (pv["src_cx"], pv["src_cy"], pv["src_fw"])
        split = split_idx.get(key)
        if split is None:
            skipped.append(iid)     # seed not found in v4 manifest -> unusable
            continue
        julia.append({
            "image_id": iid, "label": int(lab), "split": split,
            "cx": rd["cx"], "cy": rd["cy"], "fw": rd["fw"],
            "c_re": rd["c_re"], "c_im": rd["c_im"],
            "seed_group": pv["seed_group"], "seed_label": pv["seed_label"],
            "seed_source": pv["seed_source"],
            "src_cx": pv["src_cx"], "src_cy": pv["src_cy"], "src_fw": pv["src_fw"],
            "mode": pv["mode"], "rung_index": pv["rung_index"],
        })

    # --- group_id: §5 partitioned by (fractal_type, c) within each seed_group ---
    # bucket by seed_group; c-cluster (tol=0.05*src_fw); then §5 on (cx,cy,fw).
    by_seed = defaultdict(list)
    for r in julia:
        by_seed[r["seed_group"]].append(r)
    next_gid = JULIA_GID_OFFSET
    for sg, group in by_seed.items():
        src_fw = float(group[0]["src_fw"])
        ctol = C_TOL_FRAC * src_fw
        cre = [float(r["c_re"]) for r in group]
        cim = [float(r["c_im"]) for r in group]
        uf = UF(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dx, dy = cre[i] - cre[j], cim[i] - cim[j]
                if dx * dx + dy * dy <= ctol * ctol:
                    uf.union(i, j)
        cclusters = defaultdict(list)
        for i, r in enumerate(group):
            cclusters[uf.find(i)].append(r)
        for cc in cclusters.values():
            sub = cluster([float(r["cx"]) for r in cc],
                          [float(r["cy"]) for r in cc],
                          [float(r["fw"]) for r in cc])
            local = {}
            for r, k in zip(cc, sub):
                if k not in local:
                    local[k] = next_gid
                    next_gid += 1
                r["group_id"] = local[k]

    # --- write unified manifest ---
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in man:                                    # Mandelbrot: verbatim + type stamp
            o = dict(r)
            o["fractal_type"] = "mandelbrot"
            f.write(json.dumps(o) + "\n")
        for r in julia:                                  # Julia: type + c + inherited split/group
            f.write(json.dumps({
                "cx": r["cx"], "cy": r["cy"], "fw": r["fw"],
                "label": r["label"], "source": "julia_ladder_j0", "biased": False,
                "split": r["split"], "group_id": r["group_id"],
                "fractal_type": "julia", "c_re": r["c_re"], "c_im": r["c_im"],
                # provenance carried for traceability (NOT model inputs)
                "image_id": r["image_id"], "seed_group": r["seed_group"],
                "seed_label": r["seed_label"], "mode": r["mode"],
                "rung_index": r["rung_index"],
            }) + "\n")

    # ----------------------------------------------------------------- report
    j_lab = Counter(r["label"] for r in julia)
    j_split = Counter(r["split"] for r in julia)
    j_groups = len({r["group_id"] for r in julia})
    print(f"=== J0 merge ===")
    print(f"labeled J0 rows: {len(labels)}  joined+usable: {len(julia)}  skipped: {len(skipped)}")
    print(f"J0 label hist: {{1:{j_lab[1]}, 2:{j_lab[2]}, 3:{j_lab[3]}}}")
    print(f"Julia split: train={j_split['train']} eval={j_split['eval']}")
    print(f"Julia seed_groups: {len({r['seed_group'] for r in julia})}  "
          f"-> §5 group_ids: {j_groups}")

    # split x (type,label) breakdown
    print(f"\n=== unified manifest -> {OUT} ===")
    print(f"total rows: {len(man)+len(julia)}  (mandelbrot {len(man)}, julia {len(julia)})")
    for ft, rows, splitter in [
        ("mandelbrot", man, lambda r: r["split"]),
        ("julia", julia, lambda r: r["split"]),
    ]:
        for sp in ("train", "eval"):
            c = Counter(r["label"] for r in rows if splitter(r) == sp)
            n = sum(c.values())
            frac = {k: f"{c[k]/n*100:.0f}%" for k in (1, 2, 3)} if n else {}
            print(f"  {ft:10s} {sp:5s} n={n:5d}  "
                  f"label {{1:{c[1]}, 2:{c[2]}, 3:{c[3]}}}  {frac}")
    # smallest-class flag
    tot = Counter()
    for r in man:
        tot[("mandelbrot", r["label"])] += 1
    for r in julia:
        tot[("julia", r["label"])] += 1
    N = len(man) + len(julia)
    for (ft, lab), n in sorted(tot.items()):
        if n / N < 0.05:
            print(f"  [flag] ({ft}, label {lab}) = {n} ({n/N*100:.1f}%) < 5% of corpus")


if __name__ == "__main__":
    main()
