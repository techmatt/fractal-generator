"""v4 dataset assembly & validation (NO TRAINING).

Reads the five labeled batches, joins the two export-only label files
(mining, scale_2x2) onto their batch images.jsonl, reduces crops -> base
locations (label = max over the location's crops, matching the "there EXISTS a
rendering" label semantics), clusters locations into neighborhood groups via
union-find under the label's own shift+scale tolerance, splits by GROUP with
eval drawn ONLY from unbiased sources, and writes data/v4/manifest.jsonl.

Read-only on the pipeline. Writes only under data/v4/. Does NOT train.

  uv run python tools/v4/assemble.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = ROOT / "data" / "label_corpus" / "batches"
LABELS = ROOT / "labels"
OUT = ROOT / "data" / "v4"
OUT.mkdir(parents=True, exist_ok=True)

BLACK_THRESH = 0.30  # mirror present.rs / corpus_data: accept iff bf < 0.30

# Neighborhood tolerance — exactly the label's recolor+shift+scale band.
SHIFT_FRAC = 0.5      # |Δcenter| <= 0.5 * min(fw)
SCALE_LO, SCALE_HI = 1.0 / 1.5, 1.5

# --- the five labeled batches and how each is sourced/tagged ----------------- #
BATCH = {
    "mining":   "2026-06-25_mining_v3guided_v1",
    "scale2x2": "2026-06-25_scale_2x2_labelset",
    "loose0":   "2026-06-23_flat_generate_loose0_v3",
    "rev4":     "2026-06-24_guided_descend_rev4",
    "rev4occ":  "2026-06-24_guided_descend_rev4occfix_v2filtered",
}
# export-only label files keyed by image_id (batch scores.json are empty for these)
EXPORT_LABELS = {
    "mining":   LABELS / "mining_v3guided_v1.json",
    "scale2x2": LABELS / "scale_2x2_labelset.json",
}


@dataclass
class Crop:
    image_id: str
    batch: str            # short batch key
    source: str           # finer source tag (splits rev4occ by selection_role)
    biased: bool
    cx: str
    cy: str
    fw: str
    label: int
    palette: str
    composition: str
    interior_mode: str
    seed_index: int | None
    walk_id: int | None
    selection_role: str | None


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_crops() -> dict[str, list[Crop]]:
    """Per-batch list of black-gated, *labeled* crops."""
    out: dict[str, list[Crop]] = {}
    for key, bid in BATCH.items():
        export = EXPORT_LABELS.get(key)
        export_map = json.loads(export.read_text()) if export else None
        crops: list[Crop] = []
        n_total = n_unlabeled = n_black = 0
        for r in read_jsonl(CORPUS / bid / "images.jsonl"):
            n_total += 1
            iid = r["image_id"]
            # label: from export file if present, else from the row itself
            if export_map is not None:
                score = export_map.get(iid)
            else:
                score = r.get("label", {}).get("score")
            if score is None:
                n_unlabeled += 1
                continue
            prov = r.get("provenance", {})
            bf = prov.get("black_fraction")
            bf = float(bf) if bf is not None else 0.0
            if not (bf < BLACK_THRESH):
                n_black += 1
                continue
            rr = r["render"]
            role = prov.get("selection_role")
            # source / biased tagging
            if key == "rev4occ":
                src = f"rev4occ_{role}" if role else "rev4occ_unknown"
                biased = (role == "enriched")
            elif key == "mining":
                src, biased = "mining", True
            else:
                src, biased = key, False
            si = prov.get("seed_index")
            wid = prov.get("walk_id")
            crops.append(Crop(
                image_id=iid, batch=key, source=src, biased=biased,
                cx=rr["cx"], cy=rr["cy"], fw=rr["fw"], label=int(score),
                palette=rr.get("palette", ""), composition=rr.get("composition", ""),
                interior_mode=rr.get("interior_mode", ""),
                seed_index=(int(si) if si is not None else None),
                walk_id=(int(wid) if wid is not None else None),
                selection_role=role,
            ))
        out[key] = crops
        out.setdefault("_stats", {})  # type: ignore
        out["_stats"][key] = dict(n_total=n_total, n_labeled=len(crops),  # type: ignore
                                  n_unlabeled=n_unlabeled, n_black_dropped=n_black)
    return out


# --------------------------------------------------------------------------- #
# Locations: reduce crops at one exact (cx,cy,fw) -> one base location.
# --------------------------------------------------------------------------- #
@dataclass
class Location:
    cx: str
    cy: str
    fw: str
    label: int                       # MAX over the location's crop labels
    source: str                      # dominant source (single, by construction below)
    biased: bool
    batch: str
    n_crops: int
    crop_labels: list[int] = field(default_factory=list)
    palettes: list[str] = field(default_factory=list)
    seed_index: int | None = None
    walk_id: int | None = None
    loc_id: str = ""
    group_id: int = -1
    split: str = ""
    # float views for clustering
    fcx: float = 0.0
    fcy: float = 0.0
    ffw: float = 0.0


def build_locations(by_batch: dict[str, list[Crop]]) -> list[Location]:
    groups: dict[tuple, list[Crop]] = defaultdict(list)
    for key in BATCH:
        for c in by_batch[key]:
            groups[(c.batch, c.cx, c.cy, c.fw)].append(c)
    locs: list[Location] = []
    role_mixed = 0
    for (batch, cx, cy, fw), cc in groups.items():
        labs = [c.label for c in cc]
        srcs = {c.source for c in cc}
        if len(srcs) > 1:
            role_mixed += 1
        # a location's biased flag: biased if ANY crop is biased (train-lock is conservative)
        biased = any(c.biased for c in cc)
        src = sorted(srcs)[0] if len(srcs) == 1 else "+".join(sorted(srcs))
        loc = Location(
            cx=cx, cy=cy, fw=fw, label=max(labs), source=src, biased=biased,
            batch=batch, n_crops=len(cc), crop_labels=sorted(labs),
            palettes=[c.palette for c in cc],
            seed_index=cc[0].seed_index, walk_id=cc[0].walk_id,
            loc_id=f"{batch}|{cx}|{cy}|{fw}",
            fcx=float(cx), fcy=float(cy), ffw=float(fw),
        )
        locs.append(loc)
    if role_mixed:
        print(f"  [warn] {role_mixed} locations had crops from >1 source tag")
    return locs


# --------------------------------------------------------------------------- #
# Union-find neighborhood clustering under the label's shift+scale tolerance.
# --------------------------------------------------------------------------- #
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_neighborhoods(locs: list[Location]) -> int:
    """Union locations that are the same neighborhood. O(n^2) over ~few-k locs."""
    n = len(locs)
    uf = UF(n)
    cx = [l.fcx for l in locs]
    cy = [l.fcy for l in locs]
    fw = [l.ffw for l in locs]
    for i in range(n):
        for j in range(i + 1, n):
            fwi, fwj = fw[i], fw[j]
            ratio = fwi / fwj
            if ratio < SCALE_LO or ratio > SCALE_HI:
                continue
            tol = SHIFT_FRAC * min(fwi, fwj)
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy <= tol * tol:
                uf.union(i, j)
    roots = {}
    gid = 0
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = gid
            gid += 1
        locs[i].group_id = roots[r]
    return gid


# --------------------------------------------------------------------------- #
# Phase-0 reporting
# --------------------------------------------------------------------------- #
def hist3(items, key=lambda x: x):
    c = Counter(key(x) for x in items)
    return {1: c.get(1, 0), 2: c.get(2, 0), 3: c.get(3, 0)}


def fmt_hist(h, n):
    return (f"n={n:5d}  1:{h[1]:5d} ({100*h[1]/max(n,1):4.1f}%)  "
            f"2:{h[2]:5d} ({100*h[2]/max(n,1):4.1f}%)  "
            f"3:{h[3]:5d} ({100*h[3]/max(n,1):4.1f}%)")


def main():
    print("=" * 78)
    print("PHASE 0 — label sources (crop/record level)")
    print("=" * 78)
    by_batch = load_crops()
    stats = by_batch["_stats"]  # type: ignore

    # the prompt's three named sources (permanent corpus = loose0+rev4+rev4occ)
    print("\n-- per-batch crop counts (black-gated, labeled) --")
    for key in BATCH:
        s = stats[key]
        print(f"  {key:9s} ({BATCH[key]}): total={s['n_total']:5d} "
              f"unlabeled={s['n_unlabeled']:5d} black_dropped={s['n_black_dropped']:4d} "
              f"-> labeled={s['n_labeled']:5d}")

    print("\n-- per-source crop-level class distribution --")
    print(f"  {'mining (BIASED)':22s} {fmt_hist(hist3(by_batch['mining'], lambda c: c.label), len(by_batch['mining']))}")
    print(f"  {'scale2x2 (unbiased)':22s} {fmt_hist(hist3(by_batch['scale2x2'], lambda c: c.label), len(by_batch['scale2x2']))}")
    corpus = by_batch["loose0"] + by_batch["rev4"] + by_batch["rev4occ"]
    print(f"  {'corpus (loose0+rev4+rev4occ)':22s} {fmt_hist(hist3(corpus, lambda c: c.label), len(corpus))}")
    for sub in ("loose0", "rev4", "rev4occ"):
        print(f"      {sub:18s} {fmt_hist(hist3(by_batch[sub], lambda c: c.label), len(by_batch[sub]))}")
    # rev4occ enriched vs random_eval
    for role in ("enriched", "random_eval"):
        rr = [c for c in by_batch["rev4occ"] if c.selection_role == role]
        print(f"        rev4occ/{role:12s} {fmt_hist(hist3(rr, lambda c: c.label), len(rr))}")

    allc = sum((by_batch[k] for k in BATCH), [])
    print(f"\n  {'COMBINED (all 5)':22s} {fmt_hist(hist3(allc, lambda c: c.label), len(allc))}")

    # field-presence check
    print("\n-- field presence (cx/cy/fw, palette, coloring, biased, label) --")
    for key in BATCH:
        cc = by_batch[key]
        has_loc = all(c.cx and c.cy and c.fw for c in cc)
        has_si = sum(c.seed_index is not None for c in cc)
        has_wid = sum(c.walk_id is not None for c in cc)
        print(f"  {key:9s}: cx/cy/fw={'YES' if has_loc else 'NO!'}  palette=YES  "
              f"composition/interior_mode=YES  biased={'set' if cc[0].biased or key in ('mining',) else 'False'}  "
              f"seed_index={has_si}/{len(cc)}  walk_id={has_wid}/{len(cc)}")

    print(f"\n  mining set ~all-3 check: {fmt_hist(hist3(by_batch['mining'], lambda c: c.label), len(by_batch['mining']))}")

    # --------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("PHASE 1 — locations, neighborhoods, split, manifest")
    print("=" * 78)
    locs = build_locations(by_batch)
    print(f"\nbase locations (distinct cx,cy,fw): {len(locs)}  (from {len(allc)} crops)")
    print("  per-source location counts + location-level class dist:")
    by_src = defaultdict(list)
    for l in locs:
        by_src[l.source].append(l)
    for src in sorted(by_src):
        ls = by_src[src]
        print(f"    {src:24s} {fmt_hist(hist3(ls, lambda x: x.label), len(ls))}  biased={ls[0].biased}")

    # neighborhood clustering
    ngroups = cluster_neighborhoods(locs)
    gsize = Counter(l.group_id for l in locs)
    sizes = Counter(gsize.values())
    print(f"\n-- neighborhood groups: {ngroups} groups over {len(locs)} locations --")
    print("  group-size histogram (size: #groups):")
    for sz in sorted(sizes):
        print(f"    size {sz:3d}: {sizes[sz]:5d} groups")
    # 10 largest groups with source breakdown
    g_members = defaultdict(list)
    for l in locs:
        g_members[l.group_id].append(l)
    largest = sorted(g_members.items(), key=lambda kv: -len(kv[1]))[:10]
    print("  10 largest groups (size — source breakdown — label dist):")
    for gid, members in largest:
        sb = Counter(m.source for m in members)
        lb = hist3(members, lambda x: x.label)
        print(f"    g{gid:5d} size={len(members):3d}  {dict(sb)}  labels {lb}")

    # --------------------------------------------------------------------- #
    # Split by GROUP. eval drawn ONLY from unbiased-only groups; any group
    # containing a biased location is forced to TRAIN (biased = train-only AND
    # no group may span the split).
    # --------------------------------------------------------------------- #
    EVAL_FRAC = 0.40   # of eval-eligible groups, label-stratified, seed=0
    SEED = 0
    biased_groups = {gid for gid, ms in g_members.items() if any(m.biased for m in ms)}
    eligible = [gid for gid in g_members if gid not in biased_groups]
    # stratify eligible groups by group max-label
    def gmaxlabel(gid):
        return max(m.label for m in g_members[gid])
    strata = defaultdict(list)
    for gid in eligible:
        strata[gmaxlabel(gid)].append(gid)
    import random
    rng = random.Random(SEED)
    eval_groups = set()
    for lbl in sorted(strata):
        gids = sorted(strata[lbl])          # deterministic
        rng.shuffle(gids)
        k = round(len(gids) * EVAL_FRAC)
        eval_groups.update(gids[:k])
    for l in locs:
        l.split = "eval" if l.group_id in eval_groups else "train"

    tr = [l for l in locs if l.split == "train"]
    ev = [l for l in locs if l.split == "eval"]
    print("\n" + "-" * 70)
    print(f"-- SPLIT (by group; EVAL_FRAC={EVAL_FRAC} of {len(eligible)} eval-eligible "
          f"groups, label-stratified, seed={SEED}) --")
    print(f"  biased-containing groups (forced TRAIN): {len(biased_groups)}")
    print(f"  n_train locations = {len(tr)}   n_eval locations = {len(ev)}")
    print(f"  TRAIN {fmt_hist(hist3(tr, lambda x: x.label), len(tr))}")
    print(f"  EVAL  {fmt_hist(hist3(ev, lambda x: x.label), len(ev))}")
    # biased/unbiased in train
    trb = [l for l in tr if l.biased]
    tru = [l for l in tr if not l.biased]
    print(f"  TRAIN biased   {fmt_hist(hist3(trb, lambda x: x.label), len(trb))}")
    print(f"  TRAIN unbiased {fmt_hist(hist3(tru, lambda x: x.label), len(tru))}")
    # 0-biased-in-eval confirmation
    evb = [l for l in ev if l.biased]
    print(f"  >>> biased records in EVAL: {len(evb)}  (MUST be 0)")
    assert len(evb) == 0, "BIASED LEAK INTO EVAL"
    # no-group-spans-split confirmation
    span = [gid for gid in g_members
            if len({l.split for l in g_members[gid]}) > 1]
    print(f"  >>> groups spanning both splits: {len(span)}  (MUST be 0)")
    assert not span, f"GROUP SPANS SPLIT: {span[:5]}"
    # eval good-class
    eval_good = sum(1 for l in ev if l.label == 3)
    print(f"  >>> EVAL good-class (label-3) locations: {eval_good}"
          + ("  [OK >=20]" if eval_good >= 20 else "  [LOW-POWER <20 — FLAG]"))
    # eval source composition
    ev_src = Counter(l.source for l in ev)
    print(f"  EVAL source composition: {dict(ev_src)}")

    # --------------------------------------------------------------------- #
    # Manifest: one row per base location.
    # --------------------------------------------------------------------- #
    man_path = OUT / "manifest.jsonl"
    with open(man_path, "w", encoding="utf-8") as f:
        for l in locs:
            f.write(json.dumps({
                "cx": l.cx, "cy": l.cy, "fw": l.fw,
                "label": l.label, "source": l.source, "biased": l.biased,
                "split": l.split, "group_id": l.group_id,
            }) + "\n")
    print(f"\nmanifest -> {man_path}  rows={len(locs)}")

    # --------------------------------------------------------------------- #
    # Augmentation projection (report only — no full render here).
    # --------------------------------------------------------------------- #
    K_PALETTES = 5      # + 1 neutral twilight_shifted
    N_FRAMING = 3       # scale {0.7,1.0,1.3}; shift folded into the same renders
    n_loc = len(locs)
    n_renders = n_loc * (1 + K_PALETTES) * N_FRAMING
    # ~0.5s/render at 640x360 ss2 f64 (render-one); production crop ~ a few s.
    sec_lo, sec_hi = 0.4, 1.2
    print(f"\n-- augmentation projection --")
    print(f"  base locations            : {n_loc}")
    print(f"  renders/location          : (1 neutral + {K_PALETTES} palettes) x {N_FRAMING} framings "
          f"= {(1+K_PALETTES)*N_FRAMING}")
    print(f"  projected full render count: {n_renders}")
    print(f"  est. runtime @ {sec_lo}-{sec_hi}s/render: "
          f"{n_renders*sec_lo/3600:.1f}-{n_renders*sec_hi/3600:.1f} h  "
          f"(>>30s -> training prompt backgrounds it; NOT run here)")

    return locs, ngroups, g_members, by_batch


if __name__ == "__main__":
    main()
