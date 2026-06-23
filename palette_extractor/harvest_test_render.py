"""Phase 4 — render harvested palettes on the canonical test location.

Draws 20 random GATE SURVIVORS + a few GATE REJECTS, renders each at
SPIRAL_CENTER (-0.7453, 0.1127) through the post-mirror render path
(palette_lib.coloring, ported byte-for-byte from the Rust engine), and shows
strip + render thumbnail side-by-side. The rejected set validates the gate
against the REAL use: do rejected palettes actually render badly?

Harvested palettes are stored already-closed/seamless (extract_palette_cycles
pre-mirrors sequential maps), so they render with mirror=False — the post-mirror
path is satisfied without re-mirroring. Visual-first; Matt judges.

Usage:  python palette_extractor/harvest_test_render.py [--n 20] [--rejects 6] [--seed 0]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from palette_extract import gate_quality, EXTENT_FLOOR, ARCLEN_FLOOR
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from palette_lib.coloring import bake_lut, lookup_linear, linear_to_srgb, colorize

OUT = ROOT / "data" / "wallpaper_harvest"
RDIR = ROOT / "out" / "wallpaper_harvest" / "test_render"
VIZ = ROOT / "tools" / "viz" / "harvest_test_render.html"

RENDER_W, RENDER_H = 520, 340
N_CYCLES = 3.0
MAX_ITER = 600


def load_pal(name):
    return json.loads((OUT / "palettes" / f"{name}.json").read_text())


def strip_img(stops, w=520, h=30):
    lut = bake_lut(stops, mirror=False)
    t = (np.arange(w) / w) % 1.0
    row = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    return np.tile(row[None], (h, 1, 1))


def render_img(field, stops):
    lut = bake_lut(stops, mirror=False)             # post-mirror path; palette already closed
    rgb_lin = colorize(field, lut, density=N_CYCLES, mirror=False)
    return (linear_to_srgb(rgb_lin) * 255).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rejects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    man = json.loads((OUT / "manifest.json").read_text())
    E = [e for e in man["entries"] if not e.get("error")]
    for e in E:
        e["_flags"] = gate_quality(e["extent"], e["arclen"])
    survivors = [e for e in E if not e["_flags"]]
    rejects = [e for e in E if e["_flags"]]
    print(f"{len(survivors)} survivors, {len(rejects)} rejects")

    rng = np.random.default_rng(args.seed)
    pick_s = [survivors[i] for i in rng.choice(len(survivors), args.n, replace=False)]
    pick_r = [rejects[i] for i in rng.choice(len(rejects), min(args.rejects, len(rejects)),
                                             replace=False)]

    print(f"rendering field {RENDER_W}x{RENDER_H} maxiter={MAX_ITER} at {SPIRAL_CENTER} ...")
    field = render_mandelbrot(RENDER_W, RENDER_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)

    RDIR.mkdir(parents=True, exist_ok=True)

    def do(group, tag):
        rows = []
        for e in group:
            nm = e["name"]
            cmap = load_pal(nm)
            Image.fromarray(strip_img(cmap["stops"])).save(RDIR / f"{nm}_strip.png")
            Image.fromarray(render_img(field, cmap["stops"])).save(RDIR / f"{nm}_render.png")
            rows.append({"name": nm, "extent": e["extent"], "arclen": e["arclen"],
                         "coverage": e["coverage"], "cycle_label": e["cycle_label"],
                         "flags": e["_flags"]})
            print(f"  [{tag}] {nm:40s} ext={e['extent']:.3f} arc={e['arclen']:.2f} {e['_flags']}")
        return rows

    surv_rows = do(pick_s, "S")
    rej_rows = do(pick_r, "R")

    payload = {"survivors": surv_rows, "rejects": rej_rows,
               "center": list(SPIRAL_CENTER), "extent_floor": EXTENT_FLOOR,
               "arclen_floor": ARCLEN_FLOOR,
               "img_dir": "../../out/wallpaper_harvest/test_render/"}
    write_viewer(payload)
    print(f"\nwrote {VIZ}")


def write_viewer(payload):
    html = r"""<!doctype html><html><head><meta charset="utf-8"><title>harvest test render</title>
<style>:root{color-scheme:dark}body{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px}h2{font-size:14px;color:#eee;margin:22px 0 8px}.sub{color:#888;font-size:11px;max-width:1100px;margin-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:10px}
.card.rej{border-color:#6b3a10}
.nm{color:#e0e0e0;font-weight:bold;font-size:12px}.meta{color:#888;font-size:11px;margin:2px 0 6px}
.flag{color:#e8a05a}.strip{width:100%;height:22px;display:block;image-rendering:pixelated;border-radius:2px;margin-bottom:6px}
.render{width:100%;display:block;border-radius:4px;image-rendering:auto}</style>
</head><body><h1>Phase 4 — harvested palettes on the test render</h1>
<div class="sub" id="meta"></div>
<h2 id="sh"></h2><div class="grid" id="surv"></div>
<h2 id="rh"></h2><div class="grid" id="rej"></div>
<script>const M=__PAYLOAD__,IB=M.img_dir;
document.getElementById('meta').innerHTML=`Spiral render at (${M.center}) through the post-mirror path. Gate: extent<${M.extent_floor} (low_range) OR arclen<${M.arclen_floor} (degenerate). `+
 'Rejects included to validate the gate against the REAL use — do flagged palettes render badly? Visual-first; you judge.';
document.getElementById('sh').textContent=`Survivors (${M.survivors.length})`;
document.getElementById('rh').textContent=`Gate rejects (${M.rejects.length})`;
function card(r,rej){const f=r.flags.length?`<span class=flag>${r.flags.join(',')}</span>`:'pass';
 return `<div class="card${rej?' rej':''}"><div class=nm>${r.name}</div>`+
  `<div class=meta>ext=${r.extent} arc=${r.arclen} cov=${r.coverage} ${r.cycle_label} · ${f}</div>`+
  `<img class=strip src="${IB}${r.name}_strip.png"><img class=render src="${IB}${r.name}_render.png"></div>`;}
document.getElementById('surv').innerHTML=M.survivors.map(r=>card(r,false)).join('');
document.getElementById('rej').innerHTML=M.rejects.map(r=>card(r,true)).join('');
</script></body></html>"""
    VIZ.parent.mkdir(parents=True, exist_ok=True)
    VIZ.write_text(html.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")


if __name__ == "__main__":
    main()
