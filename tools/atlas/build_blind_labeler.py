#!/usr/bin/env python
"""Build a self-contained blind-read labeling page from a manifest dir.

Reads `<manifest>/blind_index.json` + `<manifest>/tiles/*.jpg`, embeds every tile as a data URI
(so the page is a single file that works offline with no server / no external hosts), and writes
`<manifest>/blind_label.html`: a keyboard-driven 1/2/3 scorer with autosave (localStorage) and a
"Download scores.json" export ({tile: score}). NO metadata (coords / p_good / depth / cluster /
keeper) is embedded — the read stays blind. Feed the exported scores.json + the hidden
manifest_key.json to the keeper-bar calibration.

  uv run python tools/atlas/build_blind_labeler.py --manifest-dir out/steered_run2_manifest
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind read — __RUN__</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#111214;color:#e8e8ea;font:15px/1.4 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;background:#17181b;border-bottom:1px solid #2a2b30;padding:10px 16px;
  display:flex;align-items:center;gap:16px;flex-wrap:wrap;z-index:5}
h1{font-size:15px;margin:0;font-weight:600;color:#cfcfd4}
.spacer{flex:1}
.stat{font-variant-numeric:tabular-nums;color:#9a9aa2}
button{font:inherit;background:#25262b;color:#e8e8ea;border:1px solid #35363c;border-radius:8px;
  padding:8px 14px;cursor:pointer}
button:hover{background:#2f3037}
button.primary{background:#2b6cb0;border-color:#2b6cb0}
button:disabled{opacity:.4;cursor:default}
main{max-width:1180px;margin:0 auto;padding:18px 16px 120px}
.stage{display:flex;flex-direction:column;align-items:center;gap:14px}
.imgwrap{width:100%;max-width:1120px;aspect-ratio:16/9;background:#000;border-radius:10px;
  overflow:hidden;border:1px solid #26272c;display:flex;align-items:center;justify-content:center}
.imgwrap img{width:100%;height:100%;object-fit:contain;image-rendering:auto}
.scores{display:flex;gap:12px}
.scores button{min-width:150px;padding:14px 10px;font-size:16px}
.b1{border-color:#7a3b3b}.b1.sel{background:#8f3b3b;border-color:#8f3b3b}
.b2{border-color:#7a6a3b}.b2.sel{background:#8f7a3b;border-color:#8f7a3b}
.b3{border-color:#3b7a52}.b3.sel{background:#2f8f57;border-color:#2f8f57}
.nav{display:flex;gap:10px;align-items:center}
.hint{color:#7d7d85;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(56px,1fr));gap:5px;margin-top:22px}
.cell{aspect-ratio:16/9;border-radius:4px;overflow:hidden;border:2px solid #26272c;cursor:pointer;position:relative}
.cell img{width:100%;height:100%;object-fit:cover;opacity:.5}
.cell.cur{border-color:#8ab4f8}
.cell .dot{position:absolute;right:2px;bottom:2px;width:9px;height:9px;border-radius:50%;border:1px solid #0008}
.d1{background:#e06666}.d2{background:#e0c266}.d3{background:#66c68a}
.done{background:#193b26;border:1px solid #2f8f57;border-radius:8px;padding:10px 14px;display:none}
</style></head><body>
<header>
  <h1>Blind read — __RUN__</h1>
  <span class="stat" id="counter">–/–</span>
  <span class="stat" id="scored">0 scored</span>
  <div class="spacer"></div>
  <button id="jump">Next unscored</button>
  <button id="export" class="primary">Download scores.json</button>
</header>
<main>
  <div class="done" id="donebar">All __N__ tiles scored — click <b>Download scores.json</b> and send it back.</div>
  <p class="hint" id="instr"></p>
  <div class="stage">
    <div class="imgwrap"><img id="img" alt="tile"></div>
    <div class="scores">
      <button class="b1" data-s="1">1 · bad</button>
      <button class="b2" data-s="2">2 · okay</button>
      <button class="b3" data-s="3">3 · good</button>
    </div>
    <div class="nav">
      <button id="prev">← Prev</button>
      <span class="stat" id="pos"></span>
      <button id="next">Next →</button>
    </div>
    <p class="hint">Keys: <b>1/2/3</b> score &amp; advance · <b>←/→</b> navigate · <b>u</b> next unscored. Progress autosaves in this browser.</p>
  </div>
  <div class="grid" id="grid"></div>
</main>
<script>
const RUN=__RUN_JSON__, TILES=__TILES__, KEY="blind_"+RUN;
document.getElementById("instr").textContent=__INSTR__;
let i=0, scores=JSON.parse(localStorage.getItem(KEY)||"{}");
const img=document.getElementById("img"), grid=document.getElementById("grid");
function nScored(){return TILES.filter(t=>scores[t.tile]).length}
function save(){localStorage.setItem(KEY,JSON.stringify(scores))}
function build(){grid.innerHTML="";TILES.forEach((t,k)=>{const c=document.createElement("div");
  c.className="cell";c.dataset.k=k;const im=new Image();im.src=t.data;c.appendChild(im);
  const d=document.createElement("div");d.className="dot";c.appendChild(d);
  c.onclick=()=>{i=k;render()};grid.appendChild(c)})}
function render(){const t=TILES[i];img.src=t.data;
  document.getElementById("counter").textContent=(i+1)+"/"+TILES.length;
  document.getElementById("pos").textContent="tile "+(i+1)+" of "+TILES.length;
  const s=nScored();document.getElementById("scored").textContent=s+" scored";
  document.getElementById("donebar").style.display=s===TILES.length?"block":"none";
  document.querySelectorAll(".scores button").forEach(b=>b.classList.toggle("sel",+b.dataset.s===scores[t.tile]));
  [...grid.children].forEach((c,k)=>{c.classList.toggle("cur",k===i);
    const d=c.querySelector(".dot");d.className="dot"+(scores[TILES[k].tile]?" d"+scores[TILES[k].tile]:"");
    c.querySelector("img").style.opacity=scores[TILES[k].tile]?"1":".5"});
  document.getElementById("prev").disabled=i===0;document.getElementById("next").disabled=i===TILES.length-1;}
function setScore(s){scores[TILES[i].tile]=s;save();if(i<TILES.length-1)i++;render()}
function nextUnscored(){for(let k=1;k<=TILES.length;k++){const j=(i+k)%TILES.length;
  if(!scores[TILES[j].tile]){i=j;break}}render()}
document.querySelectorAll(".scores button").forEach(b=>b.onclick=()=>setScore(+b.dataset.s));
document.getElementById("prev").onclick=()=>{if(i>0)i--;render()};
document.getElementById("next").onclick=()=>{if(i<TILES.length-1)i++;render()};
document.getElementById("jump").onclick=nextUnscored;
document.getElementById("export").onclick=()=>{const blob=new Blob([JSON.stringify(scores,null,2)],
  {type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download=RUN+"_blind_scores.json";a.click()};
addEventListener("keydown",e=>{if(e.key==="1")setScore(1);else if(e.key==="2")setScore(2);
  else if(e.key==="3")setScore(3);else if(e.key==="ArrowLeft"){if(i>0)i--;render()}
  else if(e.key==="ArrowRight"){if(i<TILES.length-1)i++;render()}else if(e.key==="u")nextUnscored()});
build();render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    args = ap.parse_args()
    md = args.manifest_dir
    idx = json.loads((md / "blind_index.json").read_text(encoding="utf-8"))
    run = idx["run"]
    tiles = []
    for name in idx["tiles"]:
        b = (md / "tiles" / name).read_bytes()
        tiles.append({"tile": name, "data": "data:image/jpeg;base64," + base64.b64encode(b).decode()})
    html = (PAGE
            .replace("__RUN_JSON__", json.dumps(run))
            .replace("__TILES__", json.dumps(tiles))
            .replace("__INSTR__", json.dumps(idx.get("instructions", "")))
            .replace("__RUN__", run)
            .replace("__N__", str(len(tiles))))
    out = md / "blind_label.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({mb:.1f} MB, {len(tiles)} tiles embedded)")


if __name__ == "__main__":
    main()
