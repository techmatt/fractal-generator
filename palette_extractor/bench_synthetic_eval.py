"""Palette extractor — synthetic ground-truth reconstruction eval (probe-first).

Answers "how good is extraction, really" on the **synthetic library** (224 survivors
in clean_colormaps.json, where construction ground truth exists): reconstruction
fidelity, what limits it, and mirror/cyclic misclassification — structured as a
two-(three-)setting *admission probe* so the delta attributes error to admission
BEFORE any per-stage ablation. Reals are explicitly out of scope (no ground truth).

The mass_fraction finding (90%-mass prune amputates the loop tail; native-close
climbed 11%->84% as admission opened) is the prime suspect for all three questions,
so the spine of the eval is:

  Build 1  reconstruction fidelity (chamfer + directed GT->ext coverage), stratified
           by cycle class, at THREE admission settings:
             def   = mf 0.90  (shipped default)
             mf995 = mf 0.995 (blunt high-admission)
             sf    = mf 0.90 + support_floor on (selective high-admission)
           derived: admission-attributable delta (def -> mf995) and residual (mf995).
  Build 2  mirror/cyclic confusion: true class (true_wrap_jump) x extracted class
           (cycle_label), at def and mf995; does the off-diagonal shrink with admission?
  Build 3  CONDITIONAL residual attribution (voxel_res / trim_delta / smooth_frac),
           only if the mf995 residual is non-trivial; else "admission is the whole story".
  Build 4  populate a dedicated viewer (GT + best_open/best_cycle strips, chamfer,
           true vs extracted class, admission toggle, sort-by-chamfer).

REUSES (does not rebuild): extract_palette_cycles + classify; true_wrap_jump &
strip helpers from bench_cycle_closure; ground_truth_lab/chamfer/exercised_fraction
from bench_consistency; directed_coverage from bench_imbalanced.

NO promoted defaults, NO quality claims — Matt judges fidelity by eye in the viewer.

Usage (from repo root):
  python palette_extractor/bench_synthetic_eval.py            # full 224-palette run (background it)
  python palette_extractor/bench_synthetic_eval.py --quick    # ~24-palette smoke run
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from palette_extract import extract_palette_cycles
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from palette_lib.coloring import bake_lut, lookup_linear, linear_to_srgb
# --- reuse existing bench machinery (don't reimplement) ----------------------
from bench_cycle_closure import (
    true_wrap_jump, lab_strip_png, stops_strip_png, stops_to_list,
)
from bench_consistency import ground_truth_lab, chamfer, exercised_fraction
from bench_imbalanced import directed_coverage

# -- paths ---------------------------------------------------------------------
PALETTES_JSON = ROOT / "data" / "palettes" / "clean_colormaps.json"
DATA_DIR      = ROOT / "data" / "palette_synthetic_eval"     # load-bearing manifest
RENDERS_DIR   = ROOT / "out"  / "palette_synthetic_eval"     # regenerable views
STRIPS_DIR    = RENDERS_DIR / "strips"
VIZ_HTML      = ROOT / "tools" / "viz" / "palette_synthetic_eval.html"

# -- fixed bench parameters (reported, NOT tuned) ------------------------------
RENDER_W, RENDER_H = 960, 640
MAX_ITER     = 600
DENSITY      = 1.0
GT_DENSE     = 512
COV_EPS      = 0.05      # OKLab coverage radius (matches extractor coverage_eps)
CYCLIC_THR   = 0.05      # true_wrap_jump <= this -> "genuinely cyclic" GT label
SEAM_SEQ_THR = 0.10      # extractor's native-close seam threshold (cycle_label)
LAM          = 2.0       # soft-seam lam (matches cycle_closure illustrative pass)
ARC_RETAIN   = 0.5
SUPPORT_ON   = 30.0      # selective high-admission floor (~p10 voxel mass @ 960x640)
RESID_THR    = 0.03      # mf995 chamfer above this = "materially imperfect" (Build-3 gate)
RESID_MIN_N  = 8         # need at least this many residual palettes to bother ablating
SEED         = 42

# admission settings: key -> extract_palette_cycles kwargs
SETTINGS = {
    "def":   {"mass_fraction": 0.90,  "support_floor": 0.0},
    "mf995": {"mass_fraction": 0.995, "support_floor": 0.0},
    "sf":    {"mass_fraction": 0.90,  "support_floor": SUPPORT_ON},
}
HIGH = "mf995"   # the canonical "high admission" setting for residual / Build-2 / Build-3


# -- helpers -------------------------------------------------------------------

def render_field_with_palette(field: np.ndarray, stops_raw: list, out_png: Path):
    """Render the Mandelbrot field through the palette LUT; return (lut, t_array)."""
    lut = bake_lut(stops_to_list(stops_raw))
    t = (field * DENSITY) % 1.0
    srgb = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(srgb).save(out_png)
    return lut, t


def extract_at(png: Path, kw: dict):
    return extract_palette_cycles(png, lam=LAM, arc_retain=ARC_RETAIN,
                                  seam_seq_threshold=SEAM_SEQ_THR, **kw)


# ============================================================================ #
# Build 1 — fidelity at three admission settings, stratified by cycle class
# ============================================================================ #

def run_build1(entries, field, write_strips=True) -> list[dict]:
    print(f"\n== Build 1: {len(entries)} palettes x {len(SETTINGS)} admission settings ==")
    rows = []
    for i, e in enumerate(entries):
        nm = e["name"]
        wj = true_wrap_jump(e["stops"])
        cyclic = wj <= CYCLIC_THR
        png = RENDERS_DIR / f"{nm}.png"
        lut, t = render_field_with_palette(field, e["stops"], png)
        gt = ground_truth_lab(lut, t)                  # exercised arc, OKLab (==full LUT @ density 1)
        if write_strips:
            stops_strip_png(e["stops"], STRIPS_DIR / f"{nm}_src.png")
            lab_strip_png(gt, STRIPS_DIR / f"{nm}_gt.png")

        per = {}
        for key, kw in SETTINGS.items():
            try:
                r = extract_at(png, kw)
            except Exception as exc:
                print(f"  [{i+1}/{len(entries)}] {nm:36s} {key} FAILED: {exc}")
                per[key] = {"error": str(exc)}
                continue
            # best_cycle is the class-adaptive reconstruction: native loop when the
            # seam closes (cyclic), mirror out-and-back otherwise (sequential).
            ch_cyc = chamfer(gt, r.stops_cycle_lab)
            ch_open = chamfer(gt, r.stops_open_lab)
            cov = directed_coverage(gt, r.stops_cycle_lab, COV_EPS)
            per[key] = {
                "chamfer_cycle": round(ch_cyc, 5),
                "chamfer_open": round(ch_open, 5),
                "directed_cov": round(cov, 4),
                "cycle_label": r.cycle_label,
                "seam_cycle": round(r.seam_cycle, 4),
                "n_ridge": int(r.n_ridge),
                "n_chosen": int(r.n_chosen),
            }
            if write_strips:
                lab_strip_png(r.stops_open_lab,  STRIPS_DIR / f"{nm}_{key}_open.png")
                lab_strip_png(r.stops_cycle_lab, STRIPS_DIR / f"{nm}_{key}_cycle.png")

        # derived: admission-attributable improvement (def -> mf995), residual (mf995)
        adm_delta = res_resid = None
        if "chamfer_cycle" in per.get("def", {}) and "chamfer_cycle" in per.get(HIGH, {}):
            adm_delta = round(per["def"]["chamfer_cycle"] - per[HIGH]["chamfer_cycle"], 5)
            res_resid = per[HIGH]["chamfer_cycle"]
        rows.append({
            "name": nm, "true_wrap_jump": round(wj, 4), "is_cyclic_gt": cyclic,
            "exercised_fraction": round(float(exercised_fraction(t)), 4),
            "settings": per,
            "admission_delta": adm_delta, "residual": res_resid,
        })
        if (i + 1) % 25 == 0 or i == len(entries) - 1:
            d = per.get("def", {}); h = per.get(HIGH, {})
            print(f"  [{i+1}/{len(entries)}] {nm:36s} wj={wj:.3f} "
                  f"ch def={d.get('chamfer_cycle','-')} {HIGH}={h.get('chamfer_cycle','-')} "
                  f"({d.get('cycle_label','?')}->{h.get('cycle_label','?')})")
    return rows


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 5) if xs else None


def summarize_build1(rows) -> dict:
    cyc = [r for r in rows if r["is_cyclic_gt"]]
    seq = [r for r in rows if not r["is_cyclic_gt"]]

    def stat(group, key, field):
        return _med([r["settings"].get(key, {}).get(field) for r in group])

    per_class = {}
    for cls_name, group in [("cyclic", cyc), ("sequential", seq), ("all", rows)]:
        per_class[cls_name] = {
            "n": len(group),
            "chamfer_cycle_median": {k: stat(group, k, "chamfer_cycle") for k in SETTINGS},
            "directed_cov_median":  {k: stat(group, k, "directed_cov") for k in SETTINGS},
            "admission_delta_median": _med([r["admission_delta"] for r in group]),
            "residual_median":        _med([r["residual"] for r in group]),
        }

    # class asymmetry: cyclic vs sequential chamfer gap at def and at high admission
    asym = {}
    for key in ("def", HIGH):
        c = stat(cyc, key, "chamfer_cycle")
        s = stat(seq, key, "chamfer_cycle")
        asym[key] = {"cyclic": c, "sequential": s,
                     "gap": round(c - s, 5) if (c is not None and s is not None) else None}

    # parked question: does support_floor match mf=0.995's recovery?
    sf_vs_mf = {
        "chamfer_median_all": {k: per_class["all"]["chamfer_cycle_median"][k] for k in SETTINGS},
        "sf_matches_mf995": None,
    }
    a, b = per_class["all"]["chamfer_cycle_median"]["sf"], per_class["all"]["chamfer_cycle_median"]["mf995"]
    if a is not None and b is not None:
        sf_vs_mf["sf_matches_mf995"] = bool(abs(a - b) <= 0.005)

    return {"per_class": per_class, "class_asymmetry": asym,
            "support_floor_vs_mf995": sf_vs_mf}


# ============================================================================ #
# Build 2 — true class x extracted class confusion, at def and high admission
# ============================================================================ #

def run_build2(rows) -> dict:
    out = {}
    for key in ("def", HIGH):
        # rows: true cyclic (wj<=thr) x extracted cyclic (cycle_label=="native")
        cm = {"cyc_cyc": 0, "cyc_seq": 0, "seq_cyc": 0, "seq_seq": 0}
        offdiag = []
        for r in rows:
            s = r["settings"].get(key, {})
            if "cycle_label" not in s:
                continue
            true_cyc = r["is_cyclic_gt"]
            ext_cyc = s["cycle_label"] == "native"
            if true_cyc and ext_cyc:   cm["cyc_cyc"] += 1
            elif true_cyc and not ext_cyc:
                cm["cyc_seq"] += 1
                offdiag.append({"name": r["name"], "kind": "true_cyc_ext_seq",
                                "true_wrap_jump": r["true_wrap_jump"],
                                "seam_cycle": s["seam_cycle"],
                                "admission_delta": r["admission_delta"]})
            elif not true_cyc and ext_cyc:
                cm["seq_cyc"] += 1
                offdiag.append({"name": r["name"], "kind": "true_seq_ext_cyc",
                                "true_wrap_jump": r["true_wrap_jump"],
                                "seam_cycle": s["seam_cycle"],
                                "admission_delta": r["admission_delta"]})
            else: cm["seq_seq"] += 1
        n = sum(cm.values())
        off = cm["cyc_seq"] + cm["seq_cyc"]
        out[key] = {"matrix": cm, "n": n,
                    "misclass_rate": round(off / n, 4) if n else None,
                    "off_diagonal": sorted(offdiag, key=lambda d: -(d["true_wrap_jump"])),
                    "n_off": off}
    # does the off-diagonal shrink with admission?
    out["off_shrinks"] = (out[HIGH]["n_off"] < out["def"]["n_off"]) if (
        out["def"]["n"] and out[HIGH]["n"]) else None
    return out


# ============================================================================ #
# Build 3 — CONDITIONAL residual attribution (only if mf995 residual non-trivial)
# ============================================================================ #

ABLATE_GRID = {
    "voxel_res":   [32, 48, 64],
    "trim_delta":  [0.03, 0.06, 0.10],
    "smooth_frac": [0.006, 0.012, 0.025],
}


def run_build3(rows, entries_by_name, field) -> dict:
    resid = [r for r in rows if (r["residual"] is not None and r["residual"] > RESID_THR)]
    decision = {
        "resid_thr": RESID_THR, "n_residual": len(resid),
        "residual_names": [r["name"] for r in resid],
        "residual_median_chamfer": _med([r["residual"] for r in rows]),
    }
    if len(resid) < RESID_MIN_N:
        decision["ran_ablation"] = False
        decision["conclusion"] = (
            f"high-admission ({HIGH}) residual is trivial "
            f"(median chamfer {decision['residual_median_chamfer']}, only {len(resid)} palettes "
            f"> {RESID_THR}) -> ADMISSION IS THE WHOLE STORY; per-stage ablation skipped.")
        print(f"\n== Build 3 SKIPPED: {decision['conclusion']}")
        return decision

    print(f"\n== Build 3: residual non-trivial ({len(resid)} palettes > {RESID_THR}); ablating ==")
    decision["ran_ablation"] = True
    # re-render not needed (PNGs exist from Build 1); ablate one knob at a time at mf995
    base = {"mass_fraction": 0.995, "support_floor": 0.0}
    ablation = {}
    for knob, vals in ABLATE_GRID.items():
        ablation[knob] = []
        for v in vals:
            chs = []
            for r in resid:
                e = entries_by_name[r["name"]]
                png = RENDERS_DIR / f"{r['name']}.png"
                lut = bake_lut(stops_to_list(e["stops"]))
                t = (field * DENSITY) % 1.0
                gt = ground_truth_lab(lut, t)
                kw = dict(base); kw[knob] = v
                try:
                    res = extract_at(png, kw)
                    chs.append(chamfer(gt, res.stops_cycle_lab))
                except Exception:
                    pass
            med = round(float(np.median(chs)), 5) if chs else None
            ablation[knob].append({"value": v, "median_chamfer": med, "n": len(chs)})
            print(f"  ablate {knob:12s}={str(v):<6} median_chamfer={med}")
    # which knob moves the residual the most (range across its values)?
    spans = {k: (max(a["median_chamfer"] for a in ablation[k]) -
                 min(a["median_chamfer"] for a in ablation[k]))
             for k in ablation if all(a["median_chamfer"] is not None for a in ablation[k])}
    decision["ablation"] = ablation
    decision["knob_spans"] = {k: round(v, 5) for k, v in spans.items()}
    decision["dominant_knob"] = max(spans, key=spans.get) if spans else None
    decision["conclusion"] = (
        f"residual non-trivial; dominant residual knob = {decision['dominant_knob']} "
        f"(span {decision['knob_spans'].get(decision['dominant_knob'])}).")
    return decision


# ============================================================================ #
# Build 4 — viewer
# ============================================================================ #

def write_viewer(manifest: dict, out_html: Path) -> None:
    payload = json.dumps(manifest)
    html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Palette extractor — synthetic GT reconstruction eval</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box;margin:0;padding:0}
body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px;color:#eee;margin-bottom:4px}h2{font-size:14px;color:#eee;margin:22px 0 8px}
.sub{color:#777;font-size:11px;margin-bottom:10px;max-width:1180px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:12px 16px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:11px}
th{color:#8aa;text-align:left;padding:4px 8px;border-bottom:1px solid #23252e;cursor:pointer;position:sticky;top:0;background:#0e0f13}
td{padding:3px 8px;border-bottom:1px solid #181a20;font-variant-numeric:tabular-nums}
.strip{height:18px;display:block;border-radius:2px;image-rendering:pixelated;width:160px}
.mut{color:#666}.bug{color:#f87171;font-weight:bold}.good{color:#6ee7b7}
.tag{font-size:10px;padding:1px 6px;border-radius:999px;border:1px solid #2c2f3a}
.tag.native{background:#122a1e;border-color:#1f6b46;color:#7af0b8}
.tag.sequential{background:#2a1a08;border-color:#6b3a10;color:#e8a05a}
.tag.cyc{background:#10223a;border-color:#1f4a6b;color:#7ac8f0}
.heat td{text-align:center}
button{font:11px ui-monospace;padding:3px 9px;border-radius:5px;cursor:pointer;background:#1d2029;border:1px solid #2c2f3a;color:#aaa;margin-right:5px}
button.active{background:#2a3050;border-color:#4a6af4;color:#9fb6ff}
b{color:#e0e0e0}
</style></head><body>
<h1>Palette extractor — synthetic ground-truth reconstruction eval</h1>
<div class="sub" id="meta"></div>
<div id="root"></div>
<script>
const M = __PAYLOAD__;
const SB = '../../out/palette_synthetic_eval/strips/';
const f=(x,d=4)=>(x==null?'-':Number(x).toFixed(d));
const root=document.getElementById('root');
const SET=['def','mf995','sf'], SLAB={def:'mf 0.90 (default)',mf995:'mf 0.995',sf:'mf 0.90 + support_floor'};
let curSet='def', sortk='residual', asc=false;
const HIGH='mf995';

document.getElementById('meta').innerHTML =
 `<b>${M.meta.n}</b> survivors (clean_colormaps.json) · render ${M.meta.render.join('×')} maxiter ${M.meta.max_iter} `+
 `density ${M.meta.density} · cyclic GT = true_wrap_jump ≤ <b>${M.meta.cyclic_thr}</b> · `+
 `native-close seam ≤ <b>${M.meta.seam_seq_thr}</b> · cov eps ${M.meta.cov_eps} · support_floor ON=${M.meta.support_on}`+
 `<br><b>Ground truth</b> = exercised arc OKLab (≈ full construction LUT at density 1). `+
 `<b>Fidelity</b> = chamfer(GT, best_cycle) — best_cycle closes native (cyclic) or mirrors (sequential). `+
 `NO promoted defaults — Matt judges by eye.`;

function sec(t,sub){const h=document.createElement('h2');h.textContent=t;root.append(h);
 if(sub){const d=document.createElement('div');d.className='sub';d.innerHTML=sub;root.append(d);}}
function card(h){const d=document.createElement('div');d.className='card';d.innerHTML=h;root.append(d);return d;}

// ---- Build 1 summary ----
const S=M.build1_summary, pc=S.per_class;
sec('Build 1 — fidelity by class × admission setting (median chamfer, lower=better)');
{let t='<table><tr><th>class</th><th>n</th>'+SET.map(k=>`<th>chamfer ${k}</th>`).join('')+
   SET.map(k=>`<th>cov ${k}</th>`).join('')+'<th>adm Δ (def→mf995)</th><th>residual (mf995)</th></tr>';
 for(const cls of ['cyclic','sequential','all']){const c=pc[cls];
   t+=`<tr><td><b>${cls}</b></td><td>${c.n}</td>`+
     SET.map(k=>`<td>${f(c.chamfer_cycle_median[k])}</td>`).join('')+
     SET.map(k=>`<td>${f(c.directed_cov_median[k],3)}</td>`).join('')+
     `<td class=good>${f(c.admission_delta_median)}</td><td>${f(c.residual_median)}</td></tr>`;}
 t+='</table>';card(t);}
sec('Class-asymmetry test',
 'Do cyclic palettes (need the whole loop) reconstruct worse than sequential at default, '+
 'and does the gap close at high admission?');
{const a=S.class_asymmetry;let t='<table><tr><th>setting</th><th>cyclic</th><th>sequential</th><th>gap (cyc−seq)</th></tr>';
 for(const k of ['def','mf995'])t+=`<tr><td>${k}</td><td>${f(a[k].cyclic)}</td><td>${f(a[k].sequential)}</td>`+
   `<td class=${a[k].gap>0?'bug':'good'}>${f(a[k].gap)}</td></tr>`;
 t+='</table>';card(t);}
sec('Parked question — does support_floor match mf=0.995 recovery?');
{const v=S.support_floor_vs_mf995;
 card(`median chamfer (all): `+SET.map(k=>`${k}=<b>${f(v.chamfer_median_all[k])}</b>`).join(' · ')+
   `<br>support_floor matches mf=0.995 (|Δmedian|≤0.005): <b class=${v.sf_matches_mf995?'good':'bug'}>`+
   `${v.sf_matches_mf995}</b> — if yes, support_floor is the unified admission fix and mass_fraction stays 0.90.`);}

// ---- Build 2 confusion ----
sec('Build 2 — true class (true_wrap_jump) × extracted class (cycle_label)',
 'Off-diagonal: true-cyclic read as sequential (ridge-prune suspect → should recover at high admission), '+
 'or true-sequential read as native (false closure). Does the off-diagonal shrink with admission?');
for(const k of ['def','mf995']){const b=M.build2[k];const m=b.matrix;
 let t=`<b>${SLAB[k]}</b> — misclass rate <b>${f(b.misclass_rate,3)}</b> (${b.n_off}/${b.n})`+
  `<table class=heat style='max-width:420px;margin-top:6px'><tr><th>true ↓ / ext →</th><th>native(cyc)</th><th>sequential</th></tr>`+
  `<tr><td>cyclic</td><td class=good>${m.cyc_cyc}</td><td class=bug>${m.cyc_seq}</td></tr>`+
  `<tr><td>sequential</td><td class=bug>${m.seq_cyc}</td><td class=good>${m.seq_seq}</td></tr></table>`;
 card(t);}
card(`off-diagonal shrinks with admission (def→mf995): <b class=${M.build2.off_shrinks?'good':'bug'}>${M.build2.off_shrinks}</b>`);
{const off=M.build2.mf995.off_diagonal;if(off.length){let t='<b>Residual off-diagonal at mf995</b> (genuine ambiguity / not ridge-prune)'+
  '<table><tr><th>palette</th><th>kind</th><th>true_wrap_jump</th><th>seam_cycle</th><th>adm Δ</th></tr>';
  for(const o of off)t+=`<tr><td>${o.name}</td><td>${o.kind}</td><td>${f(o.true_wrap_jump)}</td>`+
    `<td>${f(o.seam_cycle)}</td><td>${f(o.admission_delta)}</td></tr>`;
  t+='</table>';card(t);}}

// ---- Build 3 ----
sec('Build 3 — residual attribution (conditional)');
{const b=M.build3;let h=`<b>${b.conclusion}</b><br>residual set: ${b.n_residual} palettes > ${b.resid_thr} `+
  `(median ${HIGH=='mf995'?'mf995':HIGH} chamfer ${f(b.residual_median_chamfer)})`;
 if(b.ran_ablation){h+='<table style="margin-top:8px"><tr><th>knob</th><th>values → median chamfer</th><th>span</th></tr>';
   for(const k of Object.keys(b.ablation))h+=`<tr><td>${k}</td><td>`+
     b.ablation[k].map(a=>`${a.value}:${f(a.median_chamfer)}`).join(' · ')+`</td><td>${f(b.knob_spans[k])}</td></tr>`;
   h+='</table>';}
 card(h);}

// ---- Build 4 per-palette table with admission toggle + sort ----
sec('Per palette — GT vs best_open / best_cycle (sortable; admission toggle)',
 'Strips: source · GT (exercised arc) · best_open · best_cycle. Toggle the admission setting; '+
 'click a header to sort. Worst cases surface by sorting residual/chamfer descending.');
const ctrl=card('');
SET.forEach(k=>{const b=document.createElement('button');b.textContent=SLAB[k];
 b.className=(k==curSet?'active':'');b.onclick=()=>{curSet=k;[...ctrl.children].forEach(c=>c.classList.toggle('active',c.textContent==SLAB[k]));draw();};ctrl.append(b);});
const wrap=document.createElement('div');root.append(wrap);
const cols=[['name','name'],['true_wrap_jump','wj'],['chamfer_cycle','chamfer'],['chamfer_open','ch_open'],
  ['directed_cov','cov'],['cycle_label','ext'],['residual','resid (mf995)']];
function val(r,k){if(k=='name'||k=='true_wrap_jump'||k=='residual')return r[k]??-1;
  const s=r.settings[curSet]||{};return s[k]; }
function draw(){
 const rows=M.build1.slice().sort((a,b)=>{let x=val(a,sortk),y=val(b,sortk);
   if(typeof x==='string')return asc?x.localeCompare(y):y.localeCompare(x);
   x=x??-1;y=y??-1;return asc?x-y:y-x;});
 let t='<table><tr>'+cols.map(c=>`<th data-k='${c[0]}'>${c[1]}</th>`).join('')+'<th>src / GT / open / cycle</th></tr>';
 for(const r of rows){const s=r.settings[curSet]||{};
  t+=`<tr><td>${r.is_cyclic_gt?'<span class="tag cyc">cyc</span> ':''}${r.name}</td>`+
     `<td>${f(r.true_wrap_jump)}</td><td>${f(s.chamfer_cycle)}</td><td>${f(s.chamfer_open)}</td>`+
     `<td>${f(s.directed_cov,3)}</td>`+
     `<td><span class='tag ${s.cycle_label||""}'>${s.cycle_label||'-'}</span></td>`+
     `<td>${f(r.residual)}</td>`+
     `<td><img class=strip src='${SB}${r.name}_src.png'><img class=strip src='${SB}${r.name}_gt.png'>`+
     `<img class=strip src='${SB}${r.name}_${curSet}_open.png'><img class=strip src='${SB}${r.name}_${curSet}_cycle.png'></td></tr>`;}
 t+='</table>';wrap.innerHTML=t;
 wrap.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
   if(k==sortk)asc=!asc;else{sortk=k;asc=(k=='name');}draw();});}
draw();
</script></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Wrote viewer: {out_html}")


# ============================================================================ #
# main
# ============================================================================ #

def main() -> None:
    ap = argparse.ArgumentParser(description="Palette synthetic GT reconstruction eval")
    ap.add_argument("--quick", action="store_true", help="~24-palette smoke run")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    STRIPS_DIR.mkdir(parents=True, exist_ok=True)

    entries = json.loads(PALETTES_JSON.read_text())
    print(f"Loaded {len(entries)} survivor palettes")

    if args.quick:
        scored = sorted(entries, key=lambda e: true_wrap_jump(e["stops"]))
        idx = np.linspace(0, len(scored) - 1, 24).astype(int)
        entries = [scored[i] for i in idx]
        print(f"  --quick: {len(entries)} palettes spanning the wrap-jump range")

    print(f"Rendering field {RENDER_W}×{RENDER_H} maxiter={MAX_ITER} …")
    t0 = time.monotonic()
    field = render_mandelbrot(RENDER_W, RENDER_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)
    print(f"  field done in {time.monotonic()-t0:.1f}s")

    t0 = time.monotonic()
    build1 = run_build1(entries, field)
    print(f"  Build 1 pass {time.monotonic()-t0:.1f}s")
    b1_summary = summarize_build1(build1)
    build2 = run_build2(build1)
    entries_by_name = {e["name"]: e for e in entries}
    build3 = run_build3(build1, entries_by_name, field)

    manifest = {
        "meta": {
            "n": len(entries), "render": [RENDER_W, RENDER_H], "max_iter": MAX_ITER,
            "density": DENSITY, "cyclic_thr": CYCLIC_THR, "seam_seq_thr": SEAM_SEQ_THR,
            "cov_eps": COV_EPS, "support_on": SUPPORT_ON, "lam": LAM, "seed": SEED,
            "settings": {k: v for k, v in SETTINGS.items()},
        },
        "build1": build1, "build1_summary": b1_summary,
        "build2": build2, "build3": build3,
    }
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}")
    write_viewer(manifest, VIZ_HTML)

    # console summary
    pc = b1_summary["per_class"]; asym = b1_summary["class_asymmetry"]
    print("\n== SUMMARY ==")
    for cls in ("cyclic", "sequential", "all"):
        c = pc[cls]
        print(f"  {cls:11s} n={c['n']:3d}  chamfer "
              + " ".join(f"{k}={c['chamfer_cycle_median'][k]}" for k in SETTINGS)
              + f"  adm_delta={c['admission_delta_median']} residual={c['residual_median']}")
    print(f"  class asymmetry gap (cyc-seq): def={asym['def']['gap']} {HIGH}={asym[HIGH]['gap']}")
    print(f"  support_floor matches mf995: {b1_summary['support_floor_vs_mf995']['sf_matches_mf995']}")
    for k in ("def", "mf995"):
        b = build2[k]
        print(f"  confusion {k:6s}: misclass={b['misclass_rate']} off={b['n_off']}/{b['n']} "
              f"matrix={b['matrix']}")
    print(f"  off-diagonal shrinks with admission: {build2['off_shrinks']}")
    print(f"  Build 3: {build3['conclusion']}")


if __name__ == "__main__":
    main()
