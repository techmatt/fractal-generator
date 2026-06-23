"""Phase-0 gate visual for the selective-mirror seam fix.

Produces, for a few high-wj SEQUENTIAL (`mirror_needed`) maps:
  - LUT strips swept by t over [0,1): no-mirror vs mirror, at density 1 and 2
    (the density=2 strip is THE gating image — confirms the multiplied
    out-and-back is intended).
  - field renders (the synthetic spiral) no-mirror vs mirror, so the seam band
    is visible in fractal context and seen to clear.

Also renders one CYCLIC control map mirror-OFF (selective branch must leave it
untouched). Views only — written under out/, gated viewer under tools/viz/.
NO quality claim; Matt judges the strips.

Usage:  python palette_extractor/phase0_mirror_visual.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from palette_lib.coloring import bake_lut, lookup_linear, linear_to_srgb
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from bench_cycle_closure import stops_to_list, true_wrap_jump

CMAPS = json.loads((ROOT / "data" / "palettes" / "clean_colormaps.json").read_text())
BY_NAME = {e["name"]: e for e in CMAPS}

OUT = ROOT / "out" / "palette_mirror_phase0"
STRIPS = OUT / "strips"
FIELDS = OUT / "fields"
VIZ = ROOT / "tools" / "viz" / "palette_mirror_phase0.html"

# high-wj sequential maps flagged in the prompt + a couple more, plus a cyclic control
SEQ_MAPS = ["gist_stern", "cmr.ghostlight", "YlOrBr", "YlOrRd", "viridis", "magma"]
CYC_CONTROL = next(e["name"] for e in CMAPS if e["cycle"] == "cyclic")

STRIP_W, STRIP_H = 1024, 56
FIELD_W, FIELD_H = 640, 440
MAX_ITER = 600


def strip(stops, mirror: bool, density: float) -> np.ndarray:
    lut = bake_lut(stops_to_list(stops), mirror=mirror)
    t = (np.arange(STRIP_W) / STRIP_W * density) % 1.0
    row = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    return np.tile(row[None, :, :], (STRIP_H, 1, 1))


def field_render(field, stops, mirror: bool, density: float) -> np.ndarray:
    lut = bake_lut(stops_to_list(stops), mirror=mirror)
    t = (field * density) % 1.0
    return (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)


def save(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    STRIPS.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)

    print(f"Rendering synthetic spiral field {FIELD_W}x{FIELD_H} maxiter={MAX_ITER} ...")
    field = render_mandelbrot(FIELD_W, FIELD_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)

    rows = []
    names = SEQ_MAPS + [CYC_CONTROL]
    for nm in names:
        e = BY_NAME[nm]
        stops = e["stops"]
        wj = true_wrap_jump(stops)
        is_cyc = e["cycle"] == "cyclic"
        # LUT strips
        variants = {
            "nomir_d1": (False, 1.0), "mir_d1": (True, 1.0),
            "nomir_d2": (False, 2.0), "mir_d2": (True, 2.0),
        }
        for key, (mir, d) in variants.items():
            save(strip(stops, mir, d), STRIPS / f"{nm}_{key}.png")
        # field renders (density 1)
        save(field_render(field, stops, False, 1.0), FIELDS / f"{nm}_nomir.png")
        save(field_render(field, stops, True, 1.0), FIELDS / f"{nm}_mir.png")
        rows.append({"name": nm, "wj": round(wj, 4), "cycle": e["cycle"],
                     "mirror_needed": e["mirror_needed"]})
        print(f"  {nm:18s} wj={wj:.3f}  {e['cycle']}")

    manifest = {"rows": rows, "cyc_control": CYC_CONTROL,
                "strip_dir": "../../out/palette_mirror_phase0/strips/",
                "field_dir": "../../out/palette_mirror_phase0/fields/"}
    write_viewer(manifest)
    print(f"\nWrote viewer: {VIZ}")


def write_viewer(manifest):
    payload = json.dumps(manifest)
    html = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 0 — selective mirror seam fix</title>
<style>
:root{color-scheme:dark}body{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px;color:#eee}h2{font-size:14px;color:#eee;margin:20px 0 6px}
.sub{color:#888;font-size:11px;max-width:1100px;margin-bottom:8px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:12px 16px;margin-bottom:14px}
.strip{height:40px;display:block;border-radius:2px;image-rendering:pixelated;width:512px}
.field{width:320px;display:block;border-radius:3px;image-rendering:pixelated}
table{border-collapse:collapse}td{padding:4px 10px;vertical-align:middle}
th{color:#8aa;text-align:left;padding:4px 10px;font-size:11px}
.nm{color:#e0e0e0;font-weight:bold}.tag{font-size:10px;padding:1px 6px;border-radius:999px;border:1px solid #2c2f3a}
.seq{background:#2a1a08;border-color:#6b3a10;color:#e8a05a}.cyc{background:#10223a;border-color:#1f4a6b;color:#7ac8f0}
.lbl{color:#777;font-size:11px}
</style></head><body>
<h1>Phase 0 — selective pre-mirror seam fix (gate)</h1>
<div class="sub" id="meta"></div>
<div id="root"></div>
<script>
const M=__PAYLOAD__, SB=M.strip_dir, FB=M.field_dir;
const root=document.getElementById('root');
document.getElementById('meta').innerHTML=
 'LUT strips swept by t over [0,1). <b>density=2 with mirror</b> is the gating image: pre-mirror makes a sequential map an out-and-back triangle, so density d gives d passes. '+
 'Cyclic control (<b>'+M.cyc_control+'</b>) must look identical mirror on/off (selective branch keys only on mirror_needed). NO quality claim — judge the seam.';
function h2(t){const e=document.createElement('h2');e.textContent=t;root.append(e);}
function card(html){const d=document.createElement('div');d.className='card';d.innerHTML=html;root.append(d);}

h2('LUT strips — no-mirror vs mirror, density 1 and 2');
for(const r of M.rows){
 const tag=r.mirror_needed?'<span class="tag seq">sequential</span>':'<span class="tag cyc">cyclic</span>';
 card('<div class="nm">'+r.name+' '+tag+' <span class=lbl>wj='+r.wj+'</span></div>'+
 '<table><tr><th></th><th>density 1</th><th>density 2</th></tr>'+
 '<tr><td class=lbl>no-mirror</td><td><img class=strip src="'+SB+r.name+'_nomir_d1.png"></td><td><img class=strip src="'+SB+r.name+'_nomir_d2.png"></td></tr>'+
 '<tr><td class=lbl>mirror</td><td><img class=strip src="'+SB+r.name+'_mir_d1.png"></td><td><img class=strip src="'+SB+r.name+'_mir_d2.png"></td></tr>'+
 '</table>');
}
h2('Field renders (synthetic spiral, density 1) — no-mirror vs mirror');
for(const r of M.rows){
 const tag=r.mirror_needed?'<span class="tag seq">sequential</span>':'<span class="tag cyc">cyclic</span>';
 card('<div class="nm">'+r.name+' '+tag+'</div>'+
 '<table><tr><td class=lbl>no-mirror (seam)</td><td class=lbl>mirror</td></tr>'+
 '<tr><td><img class=field src="'+FB+r.name+'_nomir.png"></td><td><img class=field src="'+FB+r.name+'_mir.png"></td></tr></table>');
}
</script></body></html>"""
    VIZ.parent.mkdir(parents=True, exist_ok=True)
    VIZ.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


if __name__ == "__main__":
    main()
