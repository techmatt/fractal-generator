"""Phase 2a — measure + show the failure-gate candidates, then PAUSE.

Computes nothing new beyond the harvest manifest; it SURFACES the candidate
"bad palette" signals for Matt to set cutoffs against the eye:

  - coverage      : directed/image-coverage (1-D rope through a 2-D sheet -> may be
                    legitimately low on reals; surfaced but suspect as a "bad" signal)
  - extent        : palette OKLab gyration = color range  (low => washed-out / few colors)
  - arclen        : palette curve length (complexity partner of extent)
  - complexity    : near-constant proxy = (low extent AND low arclen) = degenerate
  - branch_drop   : traversal loss (context; reals are traversal-bound)

For each measure: a distribution histogram (PNG) + the worst-N palette strips, baked
through the SAME post-mirror path Phase 4 renders (so the strip is what you'd get).
Also an extent-vs-arclen scatter to validate the complexity proxy. Everything to a
single viewer. NO threshold is wired and NOTHING is dropped here.

Usage:  python palette_extractor/harvest_gate_measures.py [--worst 24]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))
from palette_lib.coloring import bake_lut, lookup_linear, linear_to_srgb

OUT = ROOT / "data" / "wallpaper_harvest"
VDIR = ROOT / "out" / "wallpaper_harvest" / "gate"
STRIPS = VDIR / "strips"
HISTS = VDIR / "hists"
VIZ = ROOT / "tools" / "viz" / "harvest_gate.html"


def strip_png(name: str, path: Path, w=440, h=30):
    cmap = json.loads((OUT / "palettes" / f"{name}.json").read_text())
    mir = cmap.get("mirror_needed", False)
    lut = bake_lut(cmap["stops"], mirror=mir)
    t = (np.arange(w) / w) % 1.0
    if mir:
        t = t * 0.5
    row = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.tile(row[None], (h, 1, 1))).save(path)


def hist_png(vals, title, path, vline=None):
    fig, ax = plt.subplots(figsize=(4.2, 2.4), facecolor="#15171d")
    ax.set_facecolor("#15171d")
    ax.hist(vals, bins=40, color="#5a9bd8", edgecolor="#15171d")
    ax.set_title(title, color="#ddd", fontsize=10)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#999", labelsize=8)
    if vline is not None:
        ax.axvline(vline, color="#e8a05a", ls="--", lw=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=92, facecolor="#15171d")
    plt.close(fig)


def scatter_png(x, y, path, names=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="#15171d")
    ax.set_facecolor("#15171d")
    ax.scatter(x, y, s=8, c="#5a9bd8", alpha=0.5, edgecolors="none")
    ax.set_xlabel("extent (palette gyration)", color="#ccc")
    ax.set_ylabel("arclen (palette curve length)", color="#ccc")
    ax.set_title("complexity proxy: bottom-left corner = near-constant / degenerate",
                 color="#ddd", fontsize=10)
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#999", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=92, facecolor="#15171d")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst", type=int, default=24)
    args = ap.parse_args()

    man = json.loads((OUT / "manifest.json").read_text())
    E = [e for e in man["entries"] if not e.get("error")]
    print(f"{len(E)} ok palettes ({man['errors']} errors)")

    cov = np.array([e["coverage"] for e in E])
    ext = np.array([e["extent"] for e in E])
    arc = np.array([e["arclen"] for e in E])
    bd = np.array([e["branch_drop_frac"] for e in E])
    de = np.array([e["dropped_extent"] for e in E])

    # complexity proxy: rank by normalized (extent, arclen) corner distance to origin
    en = (ext - ext.min()) / (np.ptp(ext) + 1e-9)
    an = (arc - arc.min()) / (np.ptp(arc) + 1e-9)
    degen = np.sqrt(en ** 2 + an ** 2)            # small => degenerate

    HISTS.mkdir(parents=True, exist_ok=True)
    hist_png(cov, "coverage (image-coverage)", HISTS / "coverage.png")
    hist_png(ext, "extent (color range / gyration)", HISTS / "extent.png")
    hist_png(arc, "arclen (palette curve length)", HISTS / "arclen.png")
    hist_png(bd, "branch_drop_frac (traversal loss)", HISTS / "branch_drop.png")
    hist_png(de, "dropped_extent (lost color vs AA fuzz)", HISTS / "dropped_extent.png")
    scatter_png(ext, arc, HISTS / "scatter.png")

    def worst_by(arr, ascending=True, key=None):
        order = np.argsort(arr if key is None else key)
        if not ascending:
            order = order[::-1]
        return [E[i]["name"] for i in order[: args.worst]], \
               [round(float((arr if key is None else key)[i]), 4) for i in order[: args.worst]]

    measures = {
        "coverage":   worst_by(cov, ascending=True),
        "extent":     worst_by(ext, ascending=True),
        "arclen":     worst_by(arc, ascending=True),
        "complexity": worst_by(degen, ascending=True),    # smallest = most degenerate
    }
    # also a random reference row (good-ish), seeded
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(len(E), min(args.worst, len(E)), replace=False)
    measures["random_reference"] = ([E[i]["name"] for i in ref_idx],
                                    [round(float(cov[i]), 4) for i in ref_idx])

    allnames = {n for v in measures.values() for n in v[0]}
    print(f"baking {len(allnames)} strips ...")
    for nm in allnames:
        strip_png(nm, STRIPS / f"{nm}.png")

    payload = {
        "n": len(E),
        "stats": {k: {"min": round(float(a.min()), 4), "p10": round(float(np.percentile(a, 10)), 4),
                      "p50": round(float(np.percentile(a, 50)), 4),
                      "p90": round(float(np.percentile(a, 90)), 4),
                      "max": round(float(a.max()), 4)}
                  for k, a in {"coverage": cov, "extent": ext, "arclen": arc,
                               "branch_drop": bd, "dropped_extent": de}.items()},
        "measures": {k: [{"name": n, "v": v} for n, v in zip(*pair)]
                     for k, pair in measures.items()},
        "strip_dir": "../../out/wallpaper_harvest/gate/strips/",
        "hist_dir": "../../out/wallpaper_harvest/gate/hists/",
    }
    write_viewer(payload)
    print(f"wrote {VIZ}")


def write_viewer(payload):
    html = r"""<!doctype html><html><head><meta charset="utf-8"><title>harvest gate measures</title>
<style>:root{color-scheme:dark}body{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px}h2{font-size:14px;color:#eee;margin:22px 0 6px}.sub{color:#888;font-size:11px;max-width:1100px;margin-bottom:10px}
.hists{display:flex;flex-wrap:wrap;gap:10px}.hists img{border:1px solid #23252e;border-radius:6px}
.row{display:flex;align-items:center;gap:10px;margin:2px 0}.row img{height:26px;width:380px;image-rendering:pixelated;border-radius:2px}
.v{color:#e8a05a;width:64px;text-align:right}.nm{color:#999;font-size:11px}
table{border-collapse:collapse;margin:6px 0}td,th{padding:3px 10px;text-align:right;font-size:11px}th{color:#8aa}td:first-child,th:first-child{text-align:left;color:#ccc}
.scatter img{border:1px solid #23252e;border-radius:6px}</style>
</head><body><h1>Phase 2a — failure-gate candidate measures (THRESHOLD PAUSE)</h1>
<div class="sub" id="meta"></div>
<h2>Distributions</h2><div class="hists" id="hists"></div>
<h2>Complexity proxy (extent × arclen)</h2><div class="scatter"><img id="scatter"></div>
<h2>Distribution stats</h2><div id="stats"></div>
<div id="root"></div>
<script>const M=__PAYLOAD__,SB=M.strip_dir,HB=M.hist_dir,root=document.getElementById('root');
document.getElementById('meta').innerHTML=`${M.n} harvested palettes. Worst-N strips by each candidate measure, baked through the post-mirror render path. `+
 'Decide BY EYE which measure(s) track "really bad palette" and where each cutoff sits. Coverage may be legitimately low on reals (rope-through-sheet) — suspect it. Nothing is wired or dropped.';
for(const h of ['coverage','extent','arclen','branch_drop','dropped_extent'])
 document.getElementById('hists').insertAdjacentHTML('beforeend',`<img src="${HB}${h}.png">`);
document.getElementById('scatter').src=HB+'scatter.png';
// stats table
let st='<table><tr><th>measure</th><th>min</th><th>p10</th><th>p50</th><th>p90</th><th>max</th></tr>';
for(const [k,s] of Object.entries(M.stats))st+=`<tr><td>${k}</td><td>${s.min}</td><td>${s.p10}</td><td>${s.p50}</td><td>${s.p90}</td><td>${s.max}</td></tr>`;
document.getElementById('stats').innerHTML=st+'</table>';
const LABELS={coverage:'lowest coverage',extent:'lowest extent (color range)',arclen:'lowest arclen',complexity:'most degenerate (low extent AND low arc)',random_reference:'random reference (not worst)'};
for(const [k,rows] of Object.entries(M.measures)){
 root.insertAdjacentHTML('beforeend',`<h2>${LABELS[k]||k}</h2>`);
 for(const r of rows)root.insertAdjacentHTML('beforeend',
  `<div class=row><span class=v>${r.v}</span><img src="${SB}${r.name}.png"><span class=nm>${r.name}</span></div>`);
}
</script></body></html>"""
    VIZ.parent.mkdir(parents=True, exist_ok=True)
    VIZ.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


if __name__ == "__main__":
    main()
