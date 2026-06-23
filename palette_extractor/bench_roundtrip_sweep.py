"""Palette extractor — roundtrip fidelity sweep (voxel_res x support_floor).

CHARACTERIZATION sweep, NOT a tuning decision. Maps how extraction fidelity
(alignment between the extracted palette and the true palette) moves across the
two continuous fidelity levers on the ground-truth roundtrip path, per population.
Matt judges the strips; this script promotes nothing and declares no winner.

Path is the exact-GT synthetic field (identical to bench_synthetic_eval):
  field (Mandelbrot, rendered ONCE) -> color through palette LUT -> PNG
  -> ground_truth_lab (exercised arc, OKLab) -> extract_palette_cycles -> metrics.
Render-once / re-extract-many: one PNG per palette, every grid cell re-extracts it.

REUSES (does not reimplement): extract_palette_cycles; render_mandelbrot/SPIRAL_*;
bake_lut/lookup_linear/linear_to_srgb; ground_truth_lab/chamfer/hausdorff/
aligned_residual/exercised_fraction (bench_consistency); true_wrap_jump/lab_strip_png/
stops_strip_png/stops_to_list (bench_cycle_closure); directed_coverage (bench_imbalanced);
classify_palette (palette_lib.classify).

Grid (extraction only; field/GT held fixed across every cell):
  voxel_res    in {24,32,40,48,56,64}           (main axis)
  support_floor in {OFF, 0.5x, 1x, 2x} x 30.0   (= {0,15,30,60}; bench de-facto on-value 30)
  + one mf=0.995 anchor cell at voxel_res=48 (reproduces the def->mf995 chamfer drop)
mass_fraction=0.90 fixed everywhere except the anchor. lam/arc_retain/seam_seq match
bench_synthetic_eval exactly so cell (vr48, sf0) == that eval's "def".

Floor probe (followup #3): coverage_eps does NOT enter extract_palette_cycles or
chamfer -- it only scales the directed_coverage diagnostic radius. So chamfer is
structurally eps-independent (the ~0.004 floor cannot be a coverage_eps artifact),
and we answer the probe for free by storing directed_cov at TWO eps on the SAME
extracted stops per cell. The real quantization-floor question is answered by the
voxel_res axis itself (does chamfer keep dropping past 48, or plateau).

Usage (from repo root):
  python palette_extractor/bench_roundtrip_sweep.py              # full 224 x 25 (background it)
  python palette_extractor/bench_roundtrip_sweep.py --quick      # ~16-palette smoke
  python palette_extractor/bench_roundtrip_sweep.py --workers 8  # parallel extraction
"""
from __future__ import annotations
import sys, json, time, argparse, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from palette_extract import extract_palette_cycles
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from palette_lib.coloring import (
    bake_lut, lookup_linear, linear_to_srgb, linear_srgb_to_oklab, srgb8_to_oklab,
)
from bench_consistency import (
    ground_truth_lab, chamfer, hausdorff, aligned_residual, exercised_fraction, GT_DENSE,
)
from bench_cycle_closure import true_wrap_jump, lab_strip_png, stops_strip_png, stops_to_list
from bench_imbalanced import directed_coverage

# -- paths ---------------------------------------------------------------------
PALETTES_JSON  = ROOT / "data" / "palettes" / "clean_colormaps.json"
PRIOR_MANIFEST = ROOT / "data" / "palette_synthetic_eval" / "manifest.json"   # population seed
DATA_DIR       = ROOT / "data" / "palette_roundtrip_sweep"     # load-bearing (persisted)
STRIPS_DIR     = DATA_DIR / "strips"
SCATTER_DIR    = DATA_DIR / "scatter"
RENDERS_DIR    = ROOT / "out"  / "palette_roundtrip_sweep"     # regenerable views
VIZ_HTML       = ROOT / "tools" / "viz" / "palette_roundtrip_sweep.html"

# -- fixed bench parameters (reported, NOT tuned; match bench_synthetic_eval) --
RENDER_W, RENDER_H = 960, 640
MAX_ITER     = 600
DENSITY      = 1.0
COV_EPS      = 0.05      # coverage radius (matches extractor coverage_eps default)
COV_EPS_HALF = 0.025     # floor-probe second eps (0.5x)
CYCLIC_THR   = 0.05      # true_wrap_jump <= this -> genuinely cyclic GT
SEAM_SEQ_THR = 0.10
LAM          = 2.0
ARC_RETAIN   = 0.5
SEED         = 42

# fidelity levers -------------------------------------------------------------
VOXEL_RES_AXIS    = [24, 32, 40, 48, 56, 64]
SUPPORT_FLOOR_REF = 30.0                       # bench de-facto on-value (~p10 voxel mass)
SUPPORT_FLOOR_AXIS = [0.0, 15.0, 30.0, 60.0]   # OFF, 0.5x, 1x, 2x
DEFAULT_VOXEL_RES = 48                          # extractor shipped default
MF_ANCHOR         = 0.995                       # high-admission anchor column


def make_cells() -> list[dict]:
    """Grid cells: each {key, axis tags, extract kwargs}. mass_fraction 0.90 fixed
    except the single mf995 anchor at the default voxel_res."""
    cells = []
    for vr in VOXEL_RES_AXIS:
        for sf in SUPPORT_FLOOR_AXIS:
            cells.append({
                "key": f"vr{vr}_sf{int(sf)}", "voxel_res": vr, "support_floor": sf,
                "mass_fraction": 0.90, "axis": "primary",
            })
    cells.append({
        "key": f"vr{DEFAULT_VOXEL_RES}_mf995", "voxel_res": DEFAULT_VOXEL_RES,
        "support_floor": 0.0, "mass_fraction": MF_ANCHOR, "axis": "anchor",
    })
    return cells


# ============================================================================ #
# Populations  (frozen once; tagged per palette, never a single global median)
# ============================================================================ #

HARD_CYCLIC_RE = re.compile(r"cyclic.*(grey|wrwbw|wrkbw)")
RESID_B_THR    = 0.03   # prior-manifest residual(mf995) > this & sequential -> pop B


def build_populations(entries: list[dict]) -> dict:
    """Partition the 224 survivors into A/B/C/D from construction GT + the prior
    synthetic_eval residual. Reproducible rule (documented in the artifact):

      C hard-cyclic  : is_cyclic_gt AND name ~ cyclic.*(grey|wrwbw|wrkbw)
                       -- achromatic-axis cyclic, no closeable 2-D loop; irreducible.
                       EXCLUDED from the "did fidelity improve" aggregate.
      A admission tail: is_cyclic_gt AND not C -- ridge-prune-amputated loops that
                       admission recovers (def cyc_seq -> mf995 cyc_cyc).
      B voxel residual: sequential AND prior residual(mf995) > 0.03 -- smooth ramp
                       tails that high admission fragments; voxel_res is the lever.
      D easy baseline : everything else (near the floor at defaults). Watch regressions.
    """
    prior = {}
    if PRIOR_MANIFEST.exists():
        for r in json.loads(PRIOR_MANIFEST.read_text())["build1"]:
            prior[r["name"]] = r.get("residual")
    tags = {}
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for e in entries:
        nm = e["name"]
        cyclic = true_wrap_jump(e["stops"]) <= CYCLIC_THR
        resid = prior.get(nm)
        if cyclic and HARD_CYCLIC_RE.search(nm):
            t = "C"
        elif cyclic:
            t = "A"
        elif (resid is not None) and (resid > RESID_B_THR):
            t = "B"
        else:
            t = "D"
        tags[nm] = t
        counts[t] += 1
    return {
        "tags": tags, "counts": counts,
        "rule": {
            "A_admission_tail": "is_cyclic_gt (true_wrap_jump<=%.2f) AND not C" % CYCLIC_THR,
            "B_voxel_residual": "sequential AND prior synthetic_eval residual(mf995) > %.2f" % RESID_B_THR,
            "C_hard_cyclic":    "is_cyclic_gt AND name ~ /cyclic.*(grey|wrwbw|wrkbw)/ (excluded from aggregate)",
            "D_easy_baseline":  "remainder",
        },
        "resid_seed_thr": RESID_B_THR, "cyclic_thr": CYCLIC_THR,
    }


# ============================================================================ #
# Extraction worker (one palette, all cells)  -- module-level for multiprocessing
# ============================================================================ #

def _gt_from_lut(lut: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """ground_truth_lab without the field: it only needs t's exercised [min,max]."""
    ts = np.linspace(t_min, t_max, GT_DENSE)
    return linear_srgb_to_oklab(lookup_linear(lut, ts))


def _sub(a: np.ndarray, n: int) -> list:
    """Subsample rows of a to <=n, rounded to 3 dp, for the viewer scatter."""
    if len(a) > n:
        a = a[np.linspace(0, len(a) - 1, n).astype(int)]
    return np.round(a, 3).tolist()


def process_palette(args: tuple) -> dict:
    """Extract one palette across every grid cell; write strips + scatter; return row."""
    (name, stops, t_min, t_max, cells, write_strips,
     strips_dir, scatter_dir, png_path) = args
    strips_dir = Path(strips_dir); scatter_dir = Path(scatter_dir); png_path = Path(png_path)

    lut = bake_lut(stops_to_list(stops))
    gt = _gt_from_lut(lut, t_min, t_max)
    pixels_lab = srgb8_to_oklab(np.asarray(Image.open(png_path).convert("RGB"))
                                .reshape(-1, 3).astype(np.float64))
    scatter = {"pixels": _sub(pixels_lab, 400), "gt": _sub(gt, 160), "cells": {}}

    if write_strips:
        stops_strip_png(stops, strips_dir / f"{name}_src.png")
        lab_strip_png(gt, strips_dir / f"{name}_gt.png")

    per = {}
    for c in cells:
        kw = {"voxel_res": c["voxel_res"], "support_floor": c["support_floor"],
              "mass_fraction": c["mass_fraction"]}
        try:
            r = extract_palette_cycles(png_path, lam=LAM, arc_retain=ARC_RETAIN,
                                       seam_seq_threshold=SEAM_SEQ_THR, **kw)
        except Exception as exc:
            per[c["key"]] = {"error": str(exc)}
            continue
        cyc_lab = r.stops_cycle_lab
        clo = "native" if r.cycle_label == "native" else "mirrored"
        per[c["key"]] = {
            "chamfer_cycle": round(chamfer(gt, cyc_lab), 5),
            "chamfer_open":  round(chamfer(gt, r.stops_open_lab), 5),
            "hausdorff":     round(hausdorff(gt, cyc_lab), 5),
            "aligned":       round(aligned_residual(gt, cyc_lab, clo), 5),
            "directed_cov":      round(directed_coverage(gt, cyc_lab, COV_EPS), 4),
            "directed_cov_half": round(directed_coverage(gt, cyc_lab, COV_EPS_HALF), 4),
            "cycle_label":   r.cycle_label,
            "seam_cycle":    round(r.seam_cycle, 4),
            "n_ridge":       int(r.n_ridge),
            "n_chosen":      int(r.n_chosen),
        }
        scatter["cells"][c["key"]] = _sub(cyc_lab, 96)
        if write_strips:
            lab_strip_png(cyc_lab, strips_dir / f"{name}_{c['key']}.png")

    (scatter_dir / f"{name}.json").write_text(json.dumps(scatter))
    return {"name": name, "true_wrap_jump": round(true_wrap_jump(stops), 4),
            "is_cyclic_gt": bool(true_wrap_jump(stops) <= CYCLIC_THR),
            "exercised_fraction": None, "cells": per}


# ============================================================================ #
# Aggregation (per population x cell; NEVER a single global median)
# ============================================================================ #

def _med_spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"median": None, "p25": None, "p75": None, "n": 0}
    a = np.array(xs)
    return {"median": round(float(np.median(a)), 5),
            "p25": round(float(np.percentile(a, 25)), 5),
            "p75": round(float(np.percentile(a, 75)), 5), "n": len(xs)}


def summarize(rows, tags, cells) -> dict:
    """Per (population, cell): chamfer median+spread, hausdorff, aligned median.
    Also a global-improves-but-population-regresses tripwire vs the (vr48,sf0) def cell."""
    pops = ["A", "B", "C", "D"]
    by_pop = {p: [r for r in rows if tags.get(r["name"]) == p] for p in pops}
    # aggregate set excludes C (irreducible)
    by_pop["agg(A+B+D)"] = [r for r in rows if tags.get(r["name"]) in ("A", "B", "D")]
    by_pop["all"] = rows

    def cell_metric(group, ckey, field):
        return [g["cells"].get(ckey, {}).get(field) for g in group]

    table = {}
    for pname, group in by_pop.items():
        table[pname] = {"n": len(group), "cells": {}}
        for c in cells:
            ck = c["key"]
            table[pname]["cells"][ck] = {
                "chamfer": _med_spread(cell_metric(group, ck, "chamfer_cycle")),
                "hausdorff": _med_spread(cell_metric(group, ck, "hausdorff"))["median"],
                "aligned": _med_spread(cell_metric(group, ck, "aligned"))["median"],
                "directed_cov": _med_spread(cell_metric(group, ck, "directed_cov"))["median"],
                "directed_cov_half": _med_spread(cell_metric(group, ck, "directed_cov_half"))["median"],
            }

    # regression tripwire: cell improves agg median vs def but regresses a population
    DEF = f"vr{DEFAULT_VOXEL_RES}_sf0"
    agg = table["agg(A+B+D)"]["cells"]
    def_agg = agg[DEF]["chamfer"]["median"]
    tripwire = []
    for c in cells:
        ck = c["key"]
        if ck == DEF:
            continue
        cell_agg = agg[ck]["chamfer"]["median"]
        if cell_agg is None or def_agg is None or cell_agg >= def_agg:
            continue                       # only cells that improve the aggregate
        regressed = []
        for p in ("A", "B", "D"):
            base = table[p]["cells"][DEF]["chamfer"]["median"]
            here = table[p]["cells"][ck]["chamfer"]["median"]
            if base is not None and here is not None and here > base + 0.0005:
                regressed.append({"pop": p, "def": base, "cell": round(here, 5),
                                  "delta": round(here - base, 5)})
        if regressed:
            tripwire.append({"cell": ck, "agg_def": def_agg, "agg_cell": round(cell_agg, 5),
                             "regressed_pops": regressed})
    return {"def_cell": DEF, "table": table, "regression_tripwire": tripwire}


def floor_probe(rows, tags, probe_cell: str) -> dict:
    """Followup #3: does the chamfer floor move with coverage_eps quantization?
    chamfer is eps-independent BY CONSTRUCTION (it never reads coverage_eps), so we
    report the (eps, 0.5x eps) directed_cov pair at one cell as the only thing eps
    actually moves, and the chamfer median (which cannot move) for contrast."""
    grp = [r for r in rows if tags.get(r["name"]) in ("A", "B", "D")]  # exclude C
    ch = [r["cells"].get(probe_cell, {}).get("chamfer_cycle") for r in grp]
    cov = [r["cells"].get(probe_cell, {}).get("directed_cov") for r in grp]
    covh = [r["cells"].get(probe_cell, {}).get("directed_cov_half") for r in grp]
    return {
        "probe_cell": probe_cell,
        "coverage_eps": COV_EPS, "coverage_eps_half": COV_EPS_HALF,
        "chamfer_median": _med_spread(ch)["median"],
        "directed_cov_median": _med_spread(cov)["median"],
        "directed_cov_half_median": _med_spread(covh)["median"],
        "note": ("chamfer is structurally independent of coverage_eps (never read by "
                 "extract_palette_cycles or chamfer); the ~0.004 floor is a voxel/admission "
                 "limit, NOT a coverage_eps artifact. The voxel_res axis is the real "
                 "quantization-floor probe (does chamfer keep dropping past vr48 or plateau)."),
    }


# ============================================================================ #
# Viewer
# ============================================================================ #

def write_viewer(meta: dict, out_html: Path) -> None:
    html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Palette roundtrip fidelity sweep — voxel_res x support_floor</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box;margin:0;padding:0}
body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#0e0f13;color:#ccc;display:flex;height:100vh;overflow:hidden}
#side{width:300px;min-width:220px;display:flex;flex-direction:column;border-right:1px solid #23252e;overflow:hidden}
#side h1{font-size:12px;color:#eee;padding:10px 12px;border-bottom:1px solid #23252e}
.ctl{padding:8px 12px;border-bottom:1px solid #23252e;display:flex;flex-direction:column;gap:6px}
.ctl label{font-size:10px;color:#778}
.row{display:flex;gap:4px;flex-wrap:wrap}
button{font:10px ui-monospace;padding:2px 7px;border-radius:5px;cursor:pointer;background:#1d2029;border:1px solid #2c2f3a;color:#aaa}
button.on{background:#2a3050;border-color:#4a6af4;color:#9fb6ff}
#list{flex:1;overflow-y:auto}
.it{padding:5px 12px;cursor:pointer;border-left:3px solid transparent;display:flex;justify-content:space-between;gap:6px}
.it:hover{background:#1a1c24}.it.on{border-left-color:#7a9fff;background:#161a2a}
.it .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px}
.it .sc{color:#777;font-size:10px;font-variant-numeric:tabular-nums}
.pp{font-size:9px;padding:0 4px;border-radius:3px;border:1px solid #2c2f3a}
.pA{color:#7ac8f0;border-color:#1f4a6b}.pB{color:#e8a05a;border-color:#6b3a10}
.pC{color:#c08af0;border-color:#4a2f6b}.pD{color:#888}
#main{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:16px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:12px 16px}
h2{font-size:13px;color:#eee;margin-bottom:8px}.sub{color:#777;font-size:11px;margin-bottom:8px;max-width:1100px}
table{border-collapse:collapse;font-size:11px}
th,td{padding:3px 7px;border-bottom:1px solid #1a1c20;text-align:right;font-variant-numeric:tabular-nums}
th{color:#8aa;text-align:right;border-bottom:1px solid #23252e}
th.l,td.l{text-align:left}
.strip{height:16px;display:block;image-rendering:pixelated;width:150px;border-radius:2px}
.gt{outline:1px solid #4a6af4}
.bug{color:#f87171}.good{color:#6ee7b7}.mut{color:#666}
canvas{background:#0c0d10;border:1px solid #23252e;border-radius:4px}
.flex{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.heat td{text-align:center;cursor:pointer}
.tag{font-size:9px;padding:1px 5px;border-radius:999px;border:1px solid #2c2f3a}
.tag.native{color:#7af0b8;border-color:#1f6b46}.tag.sequential{color:#e8a05a;border-color:#6b3a10}
</style></head><body>
<div id="side"><h1>roundtrip sweep</h1>
 <div class="ctl"><label>population filter</label><div class="row" id="popf"></div>
   <label>sort</label><div class="row" id="sortf"></div></div>
 <div id="list"></div></div>
<div id="main"><div class="card" id="meta">loading…</div><div id="detail" class="card">pick a palette</div></div>
<script>
const META=__META__;
const DATA='../../data/palette_roundtrip_sweep/';
const SB=DATA+'strips/', SC=DATA+'scatter/';
const f=(x,d=4)=>(x==null?'-':Number(x).toFixed(d));
let manifest=null, pops=null, cells=null, curPop='all', sortk='chamfer_def', selCell=null, scatterCache={};

const VR=META.voxel_res_axis, SF=META.support_floor_axis, DEFC=META.def_cell;

async function init(){
  manifest=await (await fetch(DATA+'manifest.json')).json();
  pops=manifest.populations; cells=manifest.cells;
  selCell=DEFC;
  const m=manifest.meta;
  document.getElementById('meta').innerHTML=
   `<b>${m.n}</b> survivors · render ${m.render.join('×')} maxiter ${m.max_iter} density ${m.density} · `+
   `GT = exercised-arc OKLab (exact). Fidelity = <b>chamfer(GT, best_cycle)</b>, lower=better. `+
   `<b>NO winner promoted — judge the strips.</b><br>`+
   `grid: voxel_res {${VR.join(',')}} × support_floor {${SF.join(',')}} (OFF/0.5×/1×/2× of ${m.support_floor_ref}) `+
   `+ mf995 anchor @vr${m.default_voxel_res}. mass_fraction 0.90 fixed (anchor 0.995).<br>`+
   `populations: <span class=pp>A</span> admission-tail ${pops.counts.A} · `+
   `<span class=pp>B</span> voxel-residual ${pops.counts.B} · <span class=pp>C</span> hard-cyclic ${pops.counts.C} (excl.) · `+
   `<span class=pp>D</span> easy ${pops.counts.D}`;
  popButtons(); sortButtons(); renderSummary(); renderList();
}
function popButtons(){const e=document.getElementById('popf');
  ['all','A','B','C','D','agg(A+B+D)'].forEach(p=>{const b=document.createElement('button');
   b.textContent=p;b.className=p==curPop?'on':'';b.onclick=()=>{curPop=p;popButtons();renderList();};e.append(b);});
  e.replaceChildren(...e.querySelectorAll('button'));}
function sortButtons(){const e=document.getElementById('sortf');e.innerHTML='';
  [['chamfer_def','chamfer@def'],['span','vr-span'],['name','name'],['wj','wrap-jump']].forEach(([k,l])=>{
   const b=document.createElement('button');b.textContent=l;b.className=k==sortk?'on':'';
   b.onclick=()=>{sortk=k;sortButtons();renderList();};e.append(b);});}

function popOf(n){return pops.tags[n];}
function chDef(r){return r.cells[DEFC]?.chamfer_cycle;}
function vrSpan(r){const v=VR.map(vr=>r.cells[`vr${vr}_sf0`]?.chamfer_cycle).filter(x=>x!=null);
  return v.length?Math.max(...v)-Math.min(...v):0;}

function renderList(){
  let rows=manifest.build1.filter(r=>{if(curPop=='all')return true;
    if(curPop=='agg(A+B+D)')return ['A','B','D'].includes(popOf(r.name));
    return popOf(r.name)==curPop;});
  rows.sort((a,b)=>{ if(sortk=='name')return a.name.localeCompare(b.name);
    if(sortk=='wj')return b.true_wrap_jump-a.true_wrap_jump;
    if(sortk=='span')return vrSpan(b)-vrSpan(a);
    return (chDef(b)??-1)-(chDef(a)??-1);});
  const L=document.getElementById('list');L.innerHTML='';
  rows.forEach(r=>{const d=document.createElement('div');d.className='it';d.dataset.n=r.name;
   d.innerHTML=`<span class="nm"><span class="pp p${popOf(r.name)}">${popOf(r.name)}</span> ${r.name}</span>`+
     `<span class="sc">${f(chDef(r))}</span>`;
   d.onclick=()=>showDetail(r.name);L.append(d);});
}

function renderSummary(){
  const S=manifest.summary, T=S.table;
  let h=`<h2>Per-population × cell — median chamfer (lower=better). Click a cell to set the scatter/strip cell.</h2>`+
    `<div class="sub">support_floor columns: OFF / 0.5× / 1× / 2× (of ${manifest.meta.support_floor_ref}). `+
    `Aggregate excludes C (irreducible). def cell = <b>${DEFC}</b>.</div>`;
  for(const p of ['A','B','D','agg(A+B+D)','C']){
    const t=T[p];h+=`<div style="margin:6px 0"><b class="p${p[0]=='a'?'D':p}">pop ${p}</b> (n=${t.n})`+
     `<table class=heat style="margin-top:3px"><tr><th class=l>vr ↓ / sf →</th>`+
     SF.map(s=>`<th>sf${s}</th>`).join('')+`<th>mf995</th></tr>`;
    for(const vr of VR){h+=`<tr><td class=l>${vr}</td>`+
      SF.map(s=>{const c=t.cells[`vr${vr}_sf${s}`];const v=c?.chamfer?.median;
        const sel=`vr${vr}_sf${s}`==selCell?'outline:1px solid #4a6af4':'';
        return `<td style="${sel};color:${heat(v)}" onclick="setCell('vr${vr}_sf${s}')">${f(v,4)}</td>`;}).join('')+
      (vr==manifest.meta.default_voxel_res?`<td style="color:${heat(t.cells[DEFC.replace('sf0','mf995')]?.chamfer?.median)}" onclick="setCell('vr${vr}_mf995')">${f(t.cells['vr'+vr+'_mf995']?.chamfer?.median,4)}</td>`:`<td class=mut>·</td>`)+
      `</tr>`;}
    h+=`</table></div>`;}
  // tripwire
  const tw=S.regression_tripwire;
  h+=`<div style="margin-top:10px"><b>Regression tripwire</b> — cells that improve the A+B+D aggregate but regress a population:`;
  h+= tw.length? `<table style="margin-top:3px"><tr><th class=l>cell</th><th>agg def</th><th>agg cell</th><th class=l>regressed</th></tr>`+
     tw.map(t=>`<tr><td class=l class=bug>${t.cell}</td><td>${f(t.agg_def)}</td><td>${f(t.agg_cell)}</td>`+
       `<td class=l>${t.regressed_pops.map(x=>`${x.pop}:+${f(x.delta)}`).join(', ')}</td></tr>`).join('')+`</table>`
     : ` <span class=good>none</span>`;
  h+=`</div>`;
  // floor probe
  const fp=manifest.floor_probe;
  h+=`<div style="margin-top:10px"><b>Floor probe</b> (followup #3) @${fp.probe_cell}: `+
   `chamfer median <b>${f(fp.chamfer_median)}</b> (eps-independent) · directed_cov eps=${fp.coverage_eps}→<b>${f(fp.directed_cov_median,3)}</b> · `+
   `eps=${fp.coverage_eps_half}→<b>${f(fp.directed_cov_half_median,3)}</b><br><span class=mut>${fp.note}</span></div>`;
  document.getElementById('main').firstElementChild.insertAdjacentHTML('afterend',`<div class="card">${h}</div>`);
}
function heat(v){if(v==null)return '#666';const t=Math.min(1,v/0.06);
  return `rgb(${Math.round(110+t*145)},${Math.round(231-t*150)},${Math.round(183-t*120)})`;}
window.setCell=(ck)=>{selCell=ck;document.querySelectorAll('.heat td').forEach(td=>td.style.outline='');
  const cur=document.querySelector('.it.on');renderSummaryRefresh();if(cur)showDetail(cur.dataset.n);};
function renderSummaryRefresh(){const c=document.querySelectorAll('#main .card');if(c[1])c[1].remove();renderSummary();}

async function showDetail(name){
  document.querySelectorAll('.it').forEach(e=>e.classList.toggle('on',e.dataset.n==name));
  const r=manifest.build1.find(x=>x.name==name);
  if(!scatterCache[name])scatterCache[name]=await (await fetch(SC+name+'.json')).json();
  const sc=scatterCache[name];
  const det=document.getElementById('detail');
  // strips across all cells
  let strips=`<div class="strip-label mut">GT</div><img class="strip gt" src="${SB}${name}_gt.png">`;
  let grid=`<table><tr><th class=l>cell</th><th>chamfer</th><th>haus</th><th>aligned</th><th>cov</th><th class=l>label</th><th class=l>strip</th></tr>`;
  for(const c of cells){const m=r.cells[c.key]||{};const isSel=c.key==selCell;
   grid+=`<tr style="${isSel?'background:#161a2a':''}"><td class=l>${c.key}${isSel?' ◀':''}</td>`+
    `<td style="color:${heat(m.chamfer_cycle)}">${f(m.chamfer_cycle)}</td><td>${f(m.hausdorff)}</td>`+
    `<td>${f(m.aligned)}</td><td>${f(m.directed_cov,3)}</td>`+
    `<td class=l><span class="tag ${m.cycle_label||''}">${m.cycle_label||'-'}</span></td>`+
    `<td class=l><img class="strip" src="${SB}${name}_${c.key}.png" onclick="setCell('${c.key}')" style="cursor:pointer"></td></tr>`;}
  grid+=`</table>`;
  det.innerHTML=`<h2><span class="pp p${popOf(name)}">${popOf(name)}</span> ${name} `+
    `<span class=mut>wrap-jump ${f(r.true_wrap_jump)} · ${r.is_cyclic_gt?'cyclic GT':'sequential GT'}</span></h2>`+
    `<div class="flex"><div><div class=sub>GT LUT strip + extracted across cells (click a strip to select its scatter cell)</div>${grid}</div>`+
    `<div><div class=sub>OKLab a–b · selected cell <b>${selCell}</b></div>`+
    `<canvas id="cv" width="260" height="260"></canvas>`+
    `<div class=sub style="margin-top:4px"><span style="color:#3aa">●</span> LUT(GT) · <span style="color:#888">●</span> pixels · <span style="color:#f83">●</span> extracted</div></div></div>`;
  drawScatter(sc, selCell);
}
function drawScatter(sc,ck){const cv=document.getElementById('cv');if(!cv)return;const x=cv.getContext('2d');
  x.fillStyle='#0c0d10';x.fillRect(0,0,260,260);x.strokeStyle='#222';
  x.beginPath();x.moveTo(130,0);x.lineTo(130,260);x.moveTo(0,130);x.lineTo(260,130);x.stroke();
  const P=p=>[p[1]*340+130,130-p[2]*340];
  const dot=(p,c,r)=>{const[a,b]=P(p);x.beginPath();x.arc(a,b,r,0,7);x.fillStyle=c;x.fill();};
  (sc.pixels||[]).forEach(p=>dot(p,'rgba(140,140,140,.35)',1.6));
  (sc.gt||[]).forEach(p=>dot(p,'rgba(50,200,200,.65)',2.2));
  (sc.cells[ck]||[]).forEach(p=>dot(p,'rgba(255,130,50,.9)',2.6));
}
init();
</script></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html.replace("__META__", json.dumps(meta)), encoding="utf-8")
    print(f"Wrote viewer: {out_html}")


# ============================================================================ #
# main
# ============================================================================ #

def main() -> None:
    ap = argparse.ArgumentParser(description="Palette roundtrip fidelity sweep")
    ap.add_argument("--quick", action="store_true", help="~16-palette smoke (spans populations)")
    ap.add_argument("--workers", type=int, default=1, help="parallel extraction processes")
    ap.add_argument("--no-strips", action="store_true", help="skip strip PNGs (metrics only)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for d in (DATA_DIR, STRIPS_DIR, SCATTER_DIR, RENDERS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    entries = json.loads(PALETTES_JSON.read_text())
    print(f"Loaded {len(entries)} survivor palettes")
    populations = build_populations(entries)
    print(f"Populations: {populations['counts']}")
    (DATA_DIR / "populations.json").write_text(json.dumps(populations, indent=2))

    if args.quick:
        # 4 per population, spanning, for a smoke run
        per_pop = {p: [e for e in entries if populations["tags"][e["name"]] == p] for p in "ABCD"}
        pick = []
        for p in "ABCD":
            g = sorted(per_pop[p], key=lambda e: e["name"])
            idx = np.linspace(0, len(g) - 1, min(4, len(g))).astype(int)
            pick += [g[i] for i in idx]
        entries = pick
        print(f"  --quick: {len(entries)} palettes across populations")

    cells = make_cells()
    print(f"Grid: {len(cells)} cells/palette × {len(entries)} palettes = {len(cells)*len(entries)} extracts")

    print(f"Rendering field {RENDER_W}×{RENDER_H} maxiter={MAX_ITER} …")
    t0 = time.monotonic()
    field = render_mandelbrot(RENDER_W, RENDER_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)
    t_arr = (field * DENSITY) % 1.0
    t_min, t_max = float(t_arr.min()), float(t_arr.max())
    ex_frac = round(float(exercised_fraction(t_arr)), 4)
    print(f"  field {time.monotonic()-t0:.1f}s  t∈[{t_min:.4f},{t_max:.4f}] exercised={ex_frac}")

    # render one colored PNG per palette (the only render stage; extraction re-reads it)
    print("Coloring per-palette PNGs …")
    for e in entries:
        lut = bake_lut(stops_to_list(e["stops"]))
        srgb = (linear_to_srgb(lookup_linear(lut, t_arr)) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(srgb).save(RENDERS_DIR / f"{e['name']}.png")

    write_strips = not args.no_strips
    tasks = [(e["name"], e["stops"], t_min, t_max, cells, write_strips,
              str(STRIPS_DIR), str(SCATTER_DIR), str(RENDERS_DIR / f"{e['name']}.png"))
             for e in entries]

    print(f"Extracting ({args.workers} worker(s)) …")
    t0 = time.monotonic()
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_palette, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs)):
                rows.append(fut.result())
                if (i + 1) % 20 == 0 or i == len(tasks) - 1:
                    print(f"  [{i+1}/{len(tasks)}] {time.monotonic()-t0:.0f}s")
    else:
        for i, t in enumerate(tasks):
            rows.append(process_palette(t))
            if (i + 1) % 20 == 0 or i == len(tasks) - 1:
                print(f"  [{i+1}/{len(tasks)}] {time.monotonic()-t0:.0f}s")
    for r in rows:
        r["exercised_fraction"] = ex_frac
    print(f"  extraction {time.monotonic()-t0:.0f}s ({(time.monotonic()-t0)/max(1,len(tasks)*len(cells))*1000:.0f}ms/extract)")

    rows.sort(key=lambda r: r["name"])
    summary = summarize(rows, populations["tags"], cells)
    fp = floor_probe(rows, populations["tags"], probe_cell=summary["def_cell"])

    meta = {
        "n": len(entries), "render": [RENDER_W, RENDER_H], "max_iter": MAX_ITER,
        "density": DENSITY, "cov_eps": COV_EPS, "cov_eps_half": COV_EPS_HALF,
        "cyclic_thr": CYCLIC_THR, "seam_seq_thr": SEAM_SEQ_THR, "lam": LAM,
        "arc_retain": ARC_RETAIN, "seed": SEED, "exercised_fraction": ex_frac,
        "voxel_res_axis": VOXEL_RES_AXIS, "support_floor_axis": SUPPORT_FLOOR_AXIS,
        "support_floor_ref": SUPPORT_FLOOR_REF, "default_voxel_res": DEFAULT_VOXEL_RES,
        "mf_anchor": MF_ANCHOR, "def_cell": summary["def_cell"],
    }
    manifest = {
        "meta": meta, "cells": cells, "populations": populations,
        "build1": rows, "summary": summary, "floor_probe": fp,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest))
    print(f"Wrote {DATA_DIR/'manifest.json'} ({(DATA_DIR/'manifest.json').stat().st_size/1e6:.1f} MB)")
    write_viewer(meta, VIZ_HTML)

    # console table
    print("\n== per-population median chamfer across voxel_res (support OFF) ==")
    T = summary["table"]
    hdr = "  pop  n   " + " ".join(f"vr{vr:<5}" for vr in VOXEL_RES_AXIS) + "  mf995"
    print(hdr)
    for p in ("A", "B", "D", "agg(A+B+D)", "C"):
        t = T[p]
        vals = " ".join(f"{(t['cells'][f'vr{vr}_sf0']['chamfer']['median'] or 0):.4f}"
                        for vr in VOXEL_RES_AXIS)
        mf = t["cells"][f"vr{DEFAULT_VOXEL_RES}_mf995"]["chamfer"]["median"]
        print(f"  {p:11s} n={t['n']:3d}  {vals}  {mf}")
    print(f"\n  regression tripwire cells: {len(summary['regression_tripwire'])}")
    for tw in summary["regression_tripwire"]:
        print(f"    {tw['cell']}: agg {tw['agg_def']}->{tw['agg_cell']} but "
              + ", ".join(f"{x['pop']}+{x['delta']}" for x in tw["regressed_pops"]))
    print(f"  floor probe @{fp['probe_cell']}: chamfer {fp['chamfer_median']} (eps-indep) "
          f"cov {fp['directed_cov_median']}/{fp['directed_cov_half_median']}")


if __name__ == "__main__":
    main()
