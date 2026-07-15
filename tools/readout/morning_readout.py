#!/usr/bin/env python
"""Morning diversity readout over an overnight emit run's manifest.jsonl.

v1 slice: emitted-diversity readout. Pure manifest/pool analysis, no GPU,
no re-render. Answers two questions over the emitted recipes:

  A. Family x Lab-color-cell composition of the emitted set.
  B. Near-duplicate / same-walk audit: spatial-viewport dedup + walk lineage,
     with the mechanism (location_key / color-cell / palette / walk sharing)
     for each surviving near-dup cluster.

Extensible: axes are small functions; render-mode diversity is deferred until
deploy_tail produces alternates.

Usage:
  uv run python tools/readout/morning_readout.py \
      out/wallpaper/overnight/overnight_20260713_001420
"""
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def load_manifest(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return rows


def load_pool_provenance(run_dir: Path):
    """Map (cycle:int, image_id) -> provenance dict from every pool cycle."""
    prov = {}
    for pool in sorted((run_dir / "pools").glob("cycle_*")):
        cyc = int(pool.name.split("_")[1])
        f = pool / "images.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prov[(cyc, r["image_id"])] = r.get("provenance", {})
    return prov


def join(rows, prov):
    """Attach provenance to each manifest row via (cycle, image_id)."""
    out = []
    for r in rows:
        key = (int(r["cycle"]), r["image_id"])
        p = prov.get(key, {})
        loc = r["location"]
        out.append({
            "emit_index": r["emit_index"],
            "image_id": r["image_id"],
            "cycle": int(r["cycle"]),
            "family": r["family"],
            "palette": r["palette"],
            "color_cell": r["color_cell"],
            "cx": Decimal(str(loc["cx"])),
            "cy": Decimal(str(loc["cy"])),
            "fw": Decimal(str(loc["fw"])),
            "cre": Decimal(str(loc["c_re"])) if loc.get("c_re") is not None else None,
            "cim": Decimal(str(loc["c_im"])) if loc.get("c_im") is not None else None,
            "fractal_type": loc.get("fractal_type"),
            "source_oid": p.get("source_oid"),
            "beam_lineage": p.get("beam_lineage"),
            "source_generation": p.get("source_generation"),
            # location identity = the per-cycle location index prefix wfd_LLL_*
            "loc_key": f"{r['cycle']}:{'_'.join(r['image_id'].split('_')[:2])}",
        })
    return out


# ---------------- A. family x color-cell ----------------
def section_a(items):
    n = len(items)
    print("=" * 68)
    print(f"A. FAMILY x COLOR-CELL COMPOSITION  (n={n} emitted)")
    print("=" * 68)

    fam = Counter(i["family"] for i in items)
    print("\nBy family:")
    for k, v in fam.most_common():
        print(f"  {k:<14} {v:>3}   ({100*v/n:4.1f}%)")

    cell = Counter(i["color_cell"] for i in items)
    print(f"\nBy Lab color-cell ({len(cell)} distinct cells filled):")
    for k in sorted(cell):
        print(f"  cell {k:<3} {cell[k]:>3}")

    pairs = Counter((i["family"], i["color_cell"]) for i in items)
    print(f"\nDistinct (family, color-cell) niches filled: {len(pairs)}")

    hi = [f for f in fam if f in ("multibrot3", "multibrot4", "multibrot5")]
    hi_n = sum(fam[f] for f in hi)
    print(f"\nHigh-degree families (multibrot3/4/5): {hi_n}/{n} emitted")
    for f in ("multibrot3", "multibrot4", "multibrot5"):
        cells = sorted({i["color_cell"] for i in items if i["family"] == f})
        print(f"  {f:<12} {fam.get(f,0):>2}   cells={cells}")


# ---------------- B. near-dup / same-walk ----------------
JULIA_FAMILIES = {"julia", "julia_multibrot3", "julia_multibrot4", "julia_multibrot5"}
C_EPS = Decimal("1e-9")  # julia-c identity tolerance


def spatial_clusters(group):
    """Union-find over one family's emitted viewports: link i,j if centers
    within 1.5*max(fw_i,fw_j) (seeder dedup metric). For julia families the
    z-plane viewport is only comparable when the c-parameter matches (c IS
    the fractal identity), so require |dc| <= C_EPS before comparing centers.

    Caveat this does NOT fix: 1.5*max(fw) is asymmetric+transitive, so a
    single wide-fw member can chain in a far deep-zoom neighbour. Families
    with a large fw spread and no c to disambiguate (phoenix) over-merge;
    treat their collapse count as a soft lower bound on distinct locations."""
    n = len(group)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = group[a], group[b]
            if ia["family"] in JULIA_FAMILIES and ia["cre"] is not None:
                dc = ((ia["cre"] - ib["cre"]) ** 2 + (ia["cim"] - ib["cim"]) ** 2).sqrt()
                if dc > C_EPS:
                    continue
            d = ((ia["cx"] - ib["cx"]) ** 2 + (ia["cy"] - ib["cy"]) ** 2).sqrt()
            if d <= Decimal("1.5") * max(ia["fw"], ib["fw"]):
                union(a, b)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(group[i])
    return list(groups.values())


def section_b(items):
    n = len(items)
    print("\n" + "=" * 68)
    print("B. NEAR-DUPLICATE / SAME-WALK AUDIT")
    print("=" * 68)

    byfam = defaultdict(list)
    for it in items:
        byfam[it["family"]].append(it)

    all_clusters, distinct, in_dup = [], 0, 0
    print("\nSpatial dedup, per-family (centers within 1.5*max(fw); "
          "julia families c-matched):")
    for fam, grp in sorted(byfam.items(), key=lambda x: -len(x[1])):
        cs = spatial_clusters(grp)
        multi = [c for c in cs if len(c) > 1]
        distinct += len(cs)
        in_dup += sum(len(c) for c in multi)
        all_clusters += multi
        soft = "  (soft: no-c + wide fw spread over-merges)" if fam == "phoenix" else ""
        print(f"  {fam:<18} n={len(grp):>2} -> {len(cs):>2} distinct loc, "
              f"{sum(len(c) for c in multi)} in a multi-cluster{soft}")
    print(f"\n  => {n} emitted collapse to {distinct} distinct locations")
    print(f"  => {in_dup}/{n} sit within dedup radius of a same-family "
          f"same-c neighbour ({n - distinct} redundant)")

    # walk lineage
    walks = Counter(i["source_oid"] for i in items)
    shared_walk = sum(v for v in walks.values() if v > 1)
    print(f"\nWalk lineage (source_oid):")
    print(f"  distinct source walks               : {len(walks)}")
    print(f"  emitted sharing a walk with another : {shared_walk}/{n}")

    print(f"\nNear-dup cluster mechanism (why each survived the selector):")
    all_clusters.sort(key=len, reverse=True)
    for ci, c in enumerate(all_clusters, 1):
        c = sorted(c, key=lambda x: x["emit_index"])
        cells = {x["color_cell"] for x in c}
        pals = {x["palette"] for x in c}
        oids = [x["source_oid"] for x in c]
        fam = c[0]["family"]
        print(f"\n  cluster {ci}: {len(c)}x [{fam}]")
        for x in c:
            cinfo = (f" c=({float(x['cre']):+.4f},{float(x['cim']):+.4f})"
                     if x["cre"] is not None else "")
            print(f"    emit#{x['emit_index']:>2} {x['image_id']} cyc{x['cycle']} "
                  f"cell={x['color_cell']:<3} fw={float(x['fw']):.2e}{cinfo} "
                  f"pal={x['palette'][:22]:<22} oid={x['source_oid']}")
        print(f"    -> color-cell: {'SAME '+str(sorted(cells)) if len(cells)==1 else f'{len(cells)} DISTINCT niches '+str(sorted(cells))}"
              f"  (MAP-Elites {'collision' if len(cells)==1 else 'kept all as separate niches'})")
        print(f"    -> palette: {len(pals)} distinct")
        print(f"    -> walk: {len(set(oids))} distinct oids"
              f"{' (consecutive siblings, one seeder run)' if len(set(oids))>1 else ''}")


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
                   "out/wallpaper/overnight/overnight_20260713_001420")
    rows = load_manifest(run_dir)
    prov = load_pool_provenance(run_dir)
    items = join(rows, prov)
    unmatched = [i for i in items if i["source_oid"] is None]
    print(f"run: {run_dir}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} emitted rows had no pool provenance match")
    section_a(items)
    section_b(items)


if __name__ == "__main__":
    main()
