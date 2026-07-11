"""Too-close-to-smooth: analysis + calibration montage (smooth-distance-pass.md step 3+4).

Reads out/render_mode_pilot/smooth_pass/distances.json (from smooth_pass.py), the batch
images.jsonl, and the human labels, then:
  * prints the sorted low tail + gap structure so a cutoff can be proposed per metric,
  * writes a self-contained-by-reference local HTML montage (rasters sorted closest->
    farthest from their smooth counterpart, each beside that counterpart w/ dE + 1-SSIM,
    sortable live by either metric), opened directly from the repo (no CSP / no base64).

    uv run python tools/render_mode_pilot/smooth_report.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict, Counter

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1"
OUT = REPO / "out/render_mode_pilot/smooth_pass"
DIST = OUT / "distances.json"
LABELS = REPO / "labels/render_mode_pilot_v1.json"
HTML = OUT / "montage.html"

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass


def pct(vals, p):
    v = sorted(vals); i = (len(v) - 1) * p; lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (i - lo) * (v[hi] - v[lo])


def main():
    dist = json.load(open(DIST))
    labels = json.load(open(LABELS))
    for d in dist:
        d["score"] = labels.get(d["image_id"])

    de = [d["de76"] for d in dist]
    om = [d["one_minus_ssim"] for d in dist]
    n = len(dist)
    print(f"n={n}")
    for name, vals in (("dE76", de), ("1-SSIM", om)):
        vs = sorted(vals)
        print(f"\n== {name} ==  min={vs[0]:.4f}  p5={pct(vals,.05):.4f}  p10={pct(vals,.10):.4f} "
              f"p25={pct(vals,.25):.4f}  median={pct(vals,.5):.4f}  max={vs[-1]:.4f}")

    # low-tail listing + gap structure per metric (for cutoff proposal)
    for key, name in (("de76", "dE76"), ("one_minus_ssim", "1-SSIM")):
        srt = sorted(dist, key=lambda d: d[key])
        print(f"\n---- closest 40 by {name} ----")
        prev = None
        for i, d in enumerate(srt[:40]):
            v = d[key]
            gap = "" if prev is None else f"  +{v-prev:.4f}"
            print(f"{i:3d}  {d[key]:.4f}  {d['mode']:30s} {d['image_id']}  lbl={d['score']}{gap}")
            prev = v

    # per-mode summary
    bymode = defaultdict(list)
    for d in dist:
        bymode[d["mode"]].append(d)
    print("\n---- per-mode medians ----")
    for m in sorted(bymode, key=lambda m: sorted(x["de76"] for x in bymode[m])[len(bymode[m])//2]):
        g = bymode[m]
        mde = sorted(x["de76"] for x in g)[len(g)//2]
        mom = sorted(x["one_minus_ssim"] for x in g)[len(g)//2]
        print(f"{m:30s} n={len(g):2d}  med dE={mde:6.2f}  med 1-SSIM={mom:.4f}  "
              f"minE={min(x['de76'] for x in g):5.2f}")

    write_html(dist)
    print(f"\nwrote {HTML}")


def write_html(dist):
    # relative paths from OUT/montage.html to the crop jpgs
    raster_rel = "../../../data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1/crops"
    smooth_rel = "smooth_crops"
    rows = []
    for d in dist:
        rows.append({
            "id": d["image_id"], "sid": d["smooth_id"], "mode": d["mode"],
            "fam": d["family"], "kind": d["mode_kind"], "score": d["score"],
            "de": round(d["de76"], 2), "om": round(d["one_minus_ssim"], 4),
        })
    payload = json.dumps(rows)
    de_sorted = sorted(r["de"] for r in rows)
    om_sorted = sorted(r["om"] for r in rows)
    html = HTML_TMPL.replace("__DATA__", payload) \
        .replace("__RASTER__", raster_rel).replace("__SMOOTH__", smooth_rel) \
        .replace("__DEMAX__", f"{de_sorted[-1]:.2f}").replace("__OMMAX__", f"{om_sorted[-1]:.4f}")
    HTML.write_text(html, encoding="utf-8")


HTML_TMPL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Too-close-to-smooth · render-mode pilot</title>
<style>
:root{
  --bg:#12100f; --panel:#1b1917; --ink:#f0ece6; --dim:#a49b90; --line:#332e2a;
  --accent:#e0663c; --good:#6fa86f; --warn:#d8a24a; --bad:#b8564a;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:"Inter",system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5}
header{padding:26px 30px 18px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:20px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;max-width:70ch}
.controls{display:flex;gap:22px;align-items:center;flex-wrap:wrap;
  padding:14px 30px;border-bottom:1px solid var(--line);position:sticky;top:0;
  background:var(--bg);z-index:5}
.controls label{color:var(--dim);text-transform:uppercase;font-size:11px;
  letter-spacing:.08em;margin-right:8px}
button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent)}
input[type=range]{vertical-align:middle;accent-color:var(--accent)}
.cut{font-variant-numeric:tabular-nums;color:var(--accent);font-family:var(--mono)}
.count{color:var(--dim);font-family:var(--mono);font-size:13px}
.grid{padding:18px 30px 60px;display:grid;gap:14px}
.pair{display:grid;grid-template-columns:34px 1fr 1fr 190px;gap:10px;align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:8px;
  transition:opacity .12s}
.pair.below{outline:2px solid var(--accent);outline-offset:0}
.pair img{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:5px;display:block;
  background:#000}
.rank{font-family:var(--mono);color:var(--dim);font-size:12px;text-align:right}
.cap{font-family:var(--mono);color:var(--dim);font-size:10px;text-align:center;
  margin-top:3px;letter-spacing:.04em}
.meta{font-family:var(--mono);font-size:12px;line-height:1.7}
.meta .mode{color:var(--ink);font-size:11px;display:block;margin-bottom:4px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.num{display:flex;justify-content:space-between}
.num b{color:var(--accent);font-weight:600}
.k{color:var(--dim)}
.lbl{display:inline-block;width:16px;height:16px;border-radius:3px;text-align:center;
  line-height:16px;font-size:10px;color:#000;font-weight:700}
.l1{background:var(--bad)} .l2{background:var(--warn)} .l3{background:var(--good)}
.l0{background:#4a453f;color:var(--dim)}
.faded{opacity:.32}
.hint{color:var(--dim);font-size:12px}
</style></head><body>
<header>
  <h1>Too-close-to-smooth · render-mode pilot</h1>
  <div class="sub">Each render-mode raster beside its <b>smooth counterpart</b> (same location,
  palette, approved color params). Sorted closest&rarr;farthest. Two distances: mean CIELAB
  &Delta;E<sub>76</sub> (color/brightness sameness) and 1&minus;SSIM (structural sameness).
  Drag a cutoff slider to see what falls below (outlined). Label chip: <span class="lbl l1">1</span>bad
  <span class="lbl l2">2</span>ok <span class="lbl l3">3</span>good.</div>
</header>
<div class="controls">
  <span><label>sort</label>
    <button id="s_de" class="on" onclick="setSort('de')">&Delta;E&nbsp;76</button>
    <button id="s_om" onclick="setSort('om')">1&minus;SSIM</button></span>
  <span><label>&Delta;E cutoff</label>
    <input id="cde" type="range" min="0" max="__DEMAX__" step="0.5" value="0"
      oninput="render()"> <span id="cdev" class="cut">off</span></span>
  <span><label>1&minus;SSIM cutoff</label>
    <input id="com" type="range" min="0" max="__OMMAX__" step="0.005" value="0"
      oninput="render()"> <span id="comv" class="cut">off</span></span>
  <span class="count" id="cnt"></span>
</div>
<div class="grid" id="grid"></div>
<script>
const DATA=__DATA__, RASTER="__RASTER__", SMOOTH="__SMOOTH__";
let SORT="de";
function setSort(k){SORT=k;document.getElementById('s_de').classList.toggle('on',k=='de');
  document.getElementById('s_om').classList.toggle('on',k=='om');render();}
function render(){
  const cde=parseFloat(document.getElementById('cde').value);
  const com=parseFloat(document.getElementById('com').value);
  document.getElementById('cdev').textContent=cde>0?('< '+cde.toFixed(1)):'off';
  document.getElementById('comv').textContent=com>0?('< '+com.toFixed(3)):'off';
  const rows=[...DATA].sort((a,b)=>a[SORT]-b[SORT]);
  let below=0;
  const g=document.getElementById('grid');g.innerHTML='';
  rows.forEach((r,i)=>{
    const isBelow=(cde>0&&r.de<cde)||(com>0&&r.om<com);
    if(isBelow)below++;
    const lc=r.score?('l'+r.score):'l0';
    const el=document.createElement('div');
    el.className='pair'+(isBelow?' below':'');
    el.innerHTML=
      `<div class="rank">${i+1}</div>
       <div><img loading="lazy" src="${RASTER}/${r.id}.jpg"><div class="cap">${r.mode} · ${r.id}</div></div>
       <div><img loading="lazy" src="${SMOOTH}/${r.sid}.jpg"><div class="cap">smooth counterpart</div></div>
       <div class="meta">
         <span class="mode">${r.fam} · ${r.kind}</span>
         <div class="num"><span class="k">&Delta;E76</span><b>${r.de.toFixed(2)}</b></div>
         <div class="num"><span class="k">1&minus;SSIM</span><b>${r.om.toFixed(4)}</b></div>
         <div class="num"><span class="k">label</span><span class="lbl ${lc}">${r.score||'&middot;'}</span></div>
       </div>`;
    g.appendChild(el);
  });
  document.getElementById('cnt').textContent=
    (cde>0||com>0)?(below+' / '+rows.length+' below cutoff'):(rows.length+' rasters');
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
