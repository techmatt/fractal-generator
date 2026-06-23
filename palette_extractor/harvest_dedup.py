"""Phase 1 dedup — harvested palettes against EACH OTHER ONLY (no library compare).

Palette-space distance = symmetric chamfer between uniform-arc-resampled OKLab curves
(reuse bench_consistency.chamfer; curves precomputed in dedup_curves.npz at N=32).
Chamfer is a set distance (rotation/reflection/phase-invariant) so it catches
re-orderings of the same colors. Threshold is EYEBALLED from a sorted strip of the
closest pairs (written to a viewer); nothing is dropped without Matt's eye.

Usage:  python palette_extractor/harvest_dedup.py [--thresh 0.0]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from bench_consistency import chamfer
from palette_lib.coloring import bake_lut, lookup_linear, linear_to_srgb

OUT = ROOT / "data" / "wallpaper_harvest"
VIZ = ROOT / "tools" / "viz" / "harvest_dedup.html"
STRIP_DIR = ROOT / "out" / "wallpaper_harvest" / "dedup_strips"


def strip_png(name: str, path: Path, w=480, h=34):
    cmap = json.loads((OUT / "palettes" / f"{name}.json").read_text())
    lut = bake_lut(cmap["stops"], mirror=cmap.get("mirror_needed", False))
    from PIL import Image
    t = (np.arange(w) / w) % 1.0
    if cmap.get("mirror_needed", False):
        t = t * 0.5
    row = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.tile(row[None], (h, 1, 1))).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.0,
                    help="if >0, count dups below this chamfer and persist a drop-list")
    ap.add_argument("--show", type=int, default=40, help="closest pairs to show")
    args = ap.parse_args()

    npz = np.load(OUT / "dedup_curves.npz")
    names = sorted(npz.files)
    C = np.stack([npz[n] for n in names]).astype(np.float64)   # (P, 32, 3)
    P = len(names)
    print(f"{P} palettes, computing pairwise chamfer ...")

    # pairwise chamfer, vectorized over j per row i (upper triangle)
    pairs = []
    for i in range(P):
        # dist matrix R[i] (32,3) vs all R[j>i]
        a = C[i]                                                # (32,3)
        rest = C[i + 1:]                                        # (Q,32,3)
        if len(rest) == 0:
            continue
        # (Q,32,32): a points vs each rest's points
        d = np.linalg.norm(a[None, :, None, :] - rest[:, None, :, :], axis=3)
        ch = 0.5 * (d.min(axis=2).mean(axis=1) + d.min(axis=1).mean(axis=1))
        for q, c in enumerate(ch):
            pairs.append((float(c), names[i], names[i + 1 + q]))
    pairs.sort(key=lambda x: x[0])

    dists = np.array([p[0] for p in pairs])
    print(f"chamfer pairs: min={dists.min():.4f} p1={np.percentile(dists,1):.4f} "
          f"p50={np.percentile(dists,50):.4f}")

    # strips for the closest `show` pairs
    closest = pairs[: args.show]
    shown_names = set()
    for _, a, b in closest:
        shown_names |= {a, b}
    for nm in shown_names:
        strip_png(nm, STRIP_DIR / f"{nm}.png")

    dup_count = int((dists < args.thresh).sum()) if args.thresh > 0 else 0
    rows = [{"d": round(c, 4), "a": a, "b": b} for c, a, b in closest]
    payload = {"rows": rows, "P": P, "thresh": args.thresh, "dup_pairs": dup_count,
               "strip_dir": "../../out/wallpaper_harvest/dedup_strips/",
               "hist": np.histogram(dists, bins=40)[0].tolist(),
               "hist_edges": [round(x, 4) for x in np.histogram(dists, bins=40)[1].tolist()]}

    if args.thresh > 0:
        # transitive dedup: keep first of each near-dup cluster
        import collections
        adj = collections.defaultdict(set)
        for c, a, b in pairs:
            if c < args.thresh:
                adj[a].add(b); adj[b].add(a)
        seen, drop = set(), []
        for nm in names:
            if nm in seen:
                continue
            stack, cluster = [nm], []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x); cluster.append(x)
                stack.extend(adj[x] - seen)
            for d in sorted(cluster)[1:]:
                drop.append(d)
        (OUT / "dedup_droplist.json").write_text(json.dumps(
            {"thresh": args.thresh, "n_drop": len(drop), "drop": sorted(drop)}, indent=1))
        print(f"thresh={args.thresh}: {dup_count} dup pairs -> drop {len(drop)} palettes")

    write_viewer(payload)
    print(f"wrote {VIZ}")


def write_viewer(payload):
    html = r"""<!doctype html><html><head><meta charset="utf-8"><title>harvest dedup</title>
<style>:root{color-scheme:dark}body{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px}.sub{color:#888;font-size:11px;max-width:1000px;margin-bottom:10px}
.pair{background:#15171d;border:1px solid #23252e;border-radius:6px;padding:8px 12px;margin-bottom:8px}
.d{color:#e8a05a;font-weight:bold}.nm{color:#aaa;font-size:11px}
img{height:30px;width:420px;display:block;image-rendering:pixelated;border-radius:2px;margin:2px 0}</style>
</head><body><h1>Harvest dedup — closest palette pairs (chamfer, each-other only)</h1>
<div class="sub" id="meta"></div><div id="root"></div>
<script>const M=__PAYLOAD__,SB=M.strip_dir,root=document.getElementById('root');
document.getElementById('meta').innerHTML=`${M.P} palettes. Closest ${M.rows.length} pairs by symmetric OKLab chamfer (N=32 uniform-arc). `+
 (M.thresh>0?`thresh=${M.thresh} -> ${M.dup_pairs} dup pairs.`:'No threshold set — eyeball where pairs stop being the same palette.');
for(const r of M.rows){root.insertAdjacentHTML('beforeend',
 `<div class=pair><span class=d>d=${r.d}</span> <span class=nm>${r.a} | ${r.b}</span>`+
 `<img src="${SB}${r.a}.png"><img src="${SB}${r.b}.png"></div>`);}
</script></body></html>"""
    VIZ.parent.mkdir(parents=True, exist_ok=True)
    VIZ.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


if __name__ == "__main__":
    main()
