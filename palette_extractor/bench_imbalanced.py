"""Palette-extractor bench — mass-imbalanced regime (support-gate + branch-drop).

The consistency bench (`bench_consistency.py`) ran at exercised-fraction 1.0 with
balanced pixel mass, where the support-gate never bites. The real failure is
*mass-imbalanced*: a dominant dark blob + a thin colorful subject. We manufacture
that regime with known ground truth via a field t-remap (`t = field**gamma`,
gamma>1 concentrates pixel mass into low-t), so on a dark-low/bright-high palette
the dark voxel dominates density while the bright arc becomes a thin subject.

What it does:
  - Build 1 regression: support-gate ON vs OFF, directed GT->ext arc-coverage.
  - Build 2 diagnostic: branch_drop_frac / dropped_extent distribution.
  - Real-failure visual: taladee + wallhaven-odxj5p, ON vs OFF (strips + coverage%).
  - Sweeps: OAT (mass_fraction, support_floor, smooth_frac, voxel_res),
    2-D grid mass_fraction x support_floor, branch-drop vs knn_k.

Headline metric = directed ground-truth->extracted arc-coverage within eps:
fraction of the true exercised arc with an extracted point within eps in OKLab.
Symmetric chamfer under-weights the thin subject (few GT points), so it is
secondary alongside directed Hausdorff (gt->ext, worst-missed color).

NO promoted defaults, NO quality claims — surfaces numbers + crops; Matt judges.

Usage (from repo root):
  python palette_extractor/bench_imbalanced.py            # full run (background it)
  python palette_extractor/bench_imbalanced.py --quick    # tiny smoke run
"""
from __future__ import annotations
import sys, json, time, argparse, shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
sys.path.insert(0, str(ROOT))

from palette_extract import extract_palette, resample_closed
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from palette_lib.coloring import (
    bake_lut, lookup_linear, linear_to_srgb, linear_srgb_to_oklab,
)

# -- paths ---------------------------------------------------------------------
PALETTES_JSON = ROOT / "data" / "palettes" / "clean_colormaps.json"
DATA_DIR      = ROOT / "data" / "palette_imbalanced"          # load-bearing JSON
RENDERS_DIR   = ROOT / "out"  / "palette_imbalanced"          # regenerable views
STRIPS_DIR    = RENDERS_DIR / "strips"
VIZ_HTML      = ROOT / "tools" / "viz" / "palette_imbalanced.html"
WALLPAPER_DIR = Path("C:/Users/techm/Desktop/Wallpapers")
REAL_CASES    = ["taladee", "wallhaven_wallhaven-odxj5p"]

# -- fixed bench parameters ----------------------------------------------------
RENDER_W, RENDER_H = 960, 640
MAX_ITER   = 600
GT_DENSE   = 512        # ground-truth points over the full exercised arc
EXT_DENSE  = 2048       # extracted-curve dense resample for coverage queries
COV_EPS    = 0.05       # OKLab coverage radius (matches extractor coverage_eps)
SUPPORT_ON = 30.0       # conservative "ON" floor (~p10 of voxel mass; not tuned)
REG_GAMMAS = [2.0, 3.0, 5.0]      # in-regime (conc90 ~0.32 / 0.18 / 0.06)
SWEEP_GAMMA = 3.0                 # representative imbalanced point for sweeps
N_REG_PAL  = 16         # palettes in the synthetic regression set
N_SWEEP_PAL = 6         # palettes per sweep cell (subset of regression set)
SEED       = 42


# -- palette / render helpers --------------------------------------------------

def stops_to_list(stops_raw: list) -> list[tuple[float, tuple]]:
    return [(float(t), tuple(int(v) for v in rgb)) for t, rgb in stops_raw]


def remap_t(field: np.ndarray, gamma: float) -> np.ndarray:
    """field in [0,1) -> t = field**gamma (gamma>1 concentrates mass into low-t)."""
    return (field ** gamma) % 1.0


def render_remap(field: np.ndarray, stops_raw: list, gamma: float,
                 out_png: Path) -> tuple[np.ndarray, np.ndarray]:
    """Render the gamma-remapped field through the palette LUT. Returns (lut, t)."""
    lut = bake_lut(stops_to_list(stops_raw))
    t   = remap_t(field, gamma)
    srgb = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(srgb).save(out_png)
    return lut, t


def ground_truth_arc(lut: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Full exercised arc (all colors touched, even thinly) in OKLab."""
    ts = np.linspace(float(t.min()), float(t.max()), GT_DENSE)
    return linear_srgb_to_oklab(lookup_linear(lut, ts))


def mass_concentration(t: np.ndarray, nb: int = 50) -> float:
    """Fraction of the t-range holding 90% of pixel mass (small = imbalanced)."""
    h, _ = np.histogram(t.ravel(), bins=nb, range=(0.0, 1.0))
    sc = np.sort(h)[::-1]
    cum = np.cumsum(sc) / h.sum()
    return float((np.searchsorted(cum, 0.90) + 1) / nb)


# -- metrics -------------------------------------------------------------------

def directed_coverage(gt_lab: np.ndarray, ext_stops: np.ndarray, eps: float) -> float:
    """Fraction of GT arc points with an extracted curve point within eps (OKLab).
    Directed gt->ext: 'did we recover the thin subject'."""
    curve = resample_closed(ext_stops, EXT_DENSE)
    d, _ = cKDTree(curve).query(gt_lab, k=1)
    return float((d <= eps).mean())


def directed_hausdorff(gt_lab: np.ndarray, ext_stops: np.ndarray) -> float:
    """Worst single missed GT color: max over GT of nearest extracted distance."""
    curve = resample_closed(ext_stops, EXT_DENSE)
    d, _ = cKDTree(curve).query(gt_lab, k=1)
    return float(d.max())


def chamfer(A: np.ndarray, B: np.ndarray) -> float:
    return float((cKDTree(B).query(A, k=1)[0].mean()
                  + cKDTree(A).query(B, k=1)[0].mean()) / 2.0)


def lab_strip_png(lab: np.ndarray, out_path: Path, w: int = 512, h: int = 32) -> None:
    from palette_lib.coloring import oklab_to_linear_srgb
    idx = np.linspace(0, len(lab) - 1, w).astype(int)
    srgb = (linear_to_srgb(oklab_to_linear_srgb(lab[idx])) * 255).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.tile(srgb[None], (h, 1, 1))).save(out_path)


# -- palette selection (dark-low / bright-high -> imbalance bites) -------------

def select_dark_low_palettes(name_to_stops: dict, n: int) -> list[str]:
    """Palettes whose low-t is dark and high-t bright: clear dark blob + bright arc.
    Spread across the (name-sorted) qualifying list to avoid one near-duplicate family."""
    qual = []
    for nm in sorted(name_to_stops):
        lut = bake_lut(stops_to_list(name_to_stops[nm]))
        L = linear_srgb_to_oklab(lut)[:, 0]
        l_lo, l_hi = L[:50].mean(), L[-50:].mean()
        if l_lo < 0.30 and (l_hi - l_lo) > 0.12:
            qual.append((nm, l_lo, l_hi))
    if len(qual) <= n:
        return [q[0] for q in qual]
    idx = np.linspace(0, len(qual) - 1, n).astype(int)
    return [qual[i][0] for i in idx]


# -- core: one extraction with metrics ----------------------------------------

def extract_and_score(out_png: Path, lut: np.ndarray, t: np.ndarray,
                      kw: dict) -> dict:
    gt = ground_truth_arc(lut, t)
    t0 = time.monotonic()
    res = extract_palette(out_png, verbose=False, **kw)
    dt = time.monotonic() - t0
    return {
        "directed_cov": round(directed_coverage(gt, res.stops_lab, COV_EPS), 4),
        "directed_hd":  round(directed_hausdorff(gt, res.stops_lab), 4),
        "chamfer":      round(chamfer(gt, res.stops_lab), 5),
        "self_cov":     round(float(res.coverage), 4),
        "closure":      res.closure,
        "n_ridge":      int(res.n_ridge),
        "branch_drop":  round(float(res.branch_drop_frac), 4),
        "dropped_ext":  round(float(res.dropped_extent), 4),
        "n_chosen":     int(res.n_chosen),
        "n_path":       int(res.n_path),
        "extract_s":    round(dt, 2),
        "_gt": gt, "_res": res,
    }


# ============================================================================ #
# Build 1 + 2 : synthetic imbalanced regression (ON vs OFF) + branch-drop
# ============================================================================ #

def run_regression(reg_pals, name_to_stops, field) -> list[dict]:
    print(f"\n== Regression: {len(reg_pals)} palettes x {len(REG_GAMMAS)} gammas x ON/OFF ==")
    rows = []
    for nm in reg_pals:
        for g in REG_GAMMAS:
            png = RENDERS_DIR / f"reg_{nm}_g{g:g}.png"
            lut, t = render_remap(field, name_to_stops[nm], g, png)
            conc = mass_concentration(t)
            cell = {"name": nm, "gamma": g, "conc90": round(conc, 3)}
            for label, sf in [("off", 0.0), ("on", SUPPORT_ON)]:
                m = extract_and_score(png, lut, t, {"support_floor": sf})
                # strips for the viewer
                lab_strip_png(m["_gt"],      STRIPS_DIR / f"reg_{nm}_g{g:g}_gt.png")
                lab_strip_png(m["_res"].stops_lab, STRIPS_DIR / f"reg_{nm}_g{g:g}_{label}.png")
                cell[label] = {k: v for k, v in m.items() if not k.startswith("_")}
            cell["cov_delta"] = round(cell["on"]["directed_cov"] - cell["off"]["directed_cov"], 4)
            cell["cham_delta"] = round(cell["on"]["chamfer"] - cell["off"]["chamfer"], 5)
            rows.append(cell)
            print(f"  {nm:20s} g={g:g} conc90={conc:.3f}  "
                  f"cov OFF={cell['off']['directed_cov']:.3f} ON={cell['on']['directed_cov']:.3f} "
                  f"(D={cell['cov_delta']:+.3f})  "
                  f"branch_drop OFF={cell['off']['branch_drop']:.2f} ON={cell['on']['branch_drop']:.2f}")
    return rows


# ============================================================================ #
# Real-failure visual (no ground truth -> strips + self-reported coverage)
# ============================================================================ #

def run_real_cases() -> list[dict]:
    print(f"\n== Real failures (visual-first): {REAL_CASES} ==")
    rows = []
    for stem in REAL_CASES:
        src = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = WALLPAPER_DIR / f"{stem}{ext}"
            if p.exists():
                src = p; break
        if src is None:  # fall back to committed thumb
            tp = ROOT / "data" / "palette_viz" / "test" / f"{stem}.thumb.jpg"
            src = tp if tp.exists() else None
        if src is None:
            print(f"  SKIP {stem}: no image found"); continue
        # thumbnail copy for the viewer
        thumb = RENDERS_DIR / f"real_{stem}.jpg"
        im = Image.open(src).convert("RGB"); im.thumbnail((720, 720))
        thumb.parent.mkdir(parents=True, exist_ok=True); im.save(thumb, quality=88)
        cell = {"name": stem}
        # OFF + a floor sweep (full-res reals live at a different mass scale than the
        # synthetic renders, so SUPPORT_ON=30 may not bite — sweep to find where it does).
        for label, sf in [("off", 0.0), ("on", SUPPORT_ON), ("hi", 150.0), ("xhi", 600.0)]:
            res = extract_palette(src, support_floor=sf, verbose=False)
            lab_strip_png(res.stops_lab, STRIPS_DIR / f"real_{stem}_{label}.png")
            cell[label] = {
                "support_floor": sf,
                "self_cov": round(float(res.coverage), 4),
                "closure": res.closure, "n_ridge": int(res.n_ridge),
                "branch_drop": round(float(res.branch_drop_frac), 4),
                "dropped_ext": round(float(res.dropped_extent), 4),
                "max_step": round(float(res.max_step), 4),
            }
        print(f"  {stem:34s} self_cov OFF={cell['off']['self_cov']*100:.1f}% "
              f"ON={cell['on']['self_cov']*100:.1f}% hi={cell['hi']['self_cov']*100:.1f}% "
              f"xhi={cell['xhi']['self_cov']*100:.1f}%  "
              f"n_ridge OFF={cell['off']['n_ridge']} -> xhi={cell['xhi']['n_ridge']}")
        rows.append(cell)
    return rows


# ============================================================================ #
# Sweeps
# ============================================================================ #

OAT_GRID = {
    "mass_fraction": [0.80, 0.85, 0.90, 0.95, 0.99],
    "support_floor": [0.0, 15.0, 30.0, 60.0, 120.0, 240.0],
    "smooth_frac":   [0.004, 0.008, 0.012, 0.018, 0.025],
    "voxel_res":     [32, 40, 48, 56, 64],
}
BASE_KW = {"mass_fraction": 0.90, "support_floor": SUPPORT_ON,
           "smooth_frac": 0.012, "voxel_res": 48, "knn_k": 8}
GRID_MF = [0.80, 0.90, 0.95, 0.99]
GRID_SF = [0.0, 15.0, 30.0, 60.0, 120.0]
KNN_VALUES = [5, 8, 12, 16, 20]


def _prep_sweep(sweep_pals, name_to_stops, field):
    """Render each sweep palette once at SWEEP_GAMMA; cache (png, lut, t)."""
    cache = {}
    for nm in sweep_pals:
        png = RENDERS_DIR / f"sweep_{nm}_g{SWEEP_GAMMA:g}.png"
        lut, t = render_remap(field, name_to_stops[nm], SWEEP_GAMMA, png)
        cache[nm] = (png, lut, t)
    return cache


def _median_cov(cache, kw) -> tuple[float, float, float]:
    covs, drops, ridges = [], [], []
    for png, lut, t in cache.values():
        m = extract_and_score(png, lut, t, kw)
        covs.append(m["directed_cov"]); drops.append(m["branch_drop"]); ridges.append(m["n_ridge"])
    return float(np.median(covs)), float(np.median(drops)), float(np.median(ridges))


def run_sweeps(sweep_pals, name_to_stops, field) -> dict:
    print(f"\n== Sweeps (subset of {len(sweep_pals)} palettes, gamma={SWEEP_GAMMA:g}) ==")
    cache = _prep_sweep(sweep_pals, name_to_stops, field)

    # OAT
    oat = {}
    for knob, vals in OAT_GRID.items():
        oat[knob] = []
        for v in vals:
            kw = dict(BASE_KW); kw[knob] = v
            cov, drop, ridge = _median_cov(cache, kw)
            oat[knob].append({"value": v, "median_cov": round(cov, 4),
                              "median_drop": round(drop, 4), "median_ridge": round(ridge, 1)})
            print(f"  OAT {knob:14s}={str(v):<6} median_cov={cov:.4f} drop={drop:.3f} ridge={ridge:.0f}")

    # 2-D grid mass_fraction x support_floor
    grid = []
    for mf in GRID_MF:
        for sf in GRID_SF:
            kw = dict(BASE_KW); kw["mass_fraction"] = mf; kw["support_floor"] = sf
            cov, drop, ridge = _median_cov(cache, kw)
            grid.append({"mass_fraction": mf, "support_floor": sf,
                         "median_cov": round(cov, 4), "median_drop": round(drop, 4)})
        print(f"  grid mf={mf:.2f}: " +
              " ".join(f"sf{sf:g}={[g['median_cov'] for g in grid if g['mass_fraction']==mf and g['support_floor']==sf][0]:.3f}"
                       for sf in GRID_SF))

    # branch-drop vs knn_k (topology lever)
    knn = []
    for k in KNN_VALUES:
        kw = dict(BASE_KW); kw["knn_k"] = k
        cov, drop, ridge = _median_cov(cache, kw)
        knn.append({"knn_k": k, "median_cov": round(cov, 4), "median_drop": round(drop, 4)})
        print(f"  knn_k={k:2d} median_drop={drop:.4f} median_cov={cov:.4f}")

    return {"oat": oat, "grid": grid, "knn": knn,
            "base_kw": {k: v for k, v in BASE_KW.items()}}


# ============================================================================ #
# Viewer
# ============================================================================ #

def write_viewer(manifest: dict, out_html: Path) -> None:
    payload = json.dumps(manifest)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Palette bench — imbalanced regime</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box;margin:0;padding:0}
body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 24px}
h1{font-size:16px;color:#eee;margin-bottom:4px}h2{font-size:14px;color:#eee;margin:22px 0 8px}
.sub{color:#666;font-size:11px;margin-bottom:10px}
table{border-collapse:collapse;width:100%;font-size:11px;margin-bottom:8px}
th{color:#888;text-align:left;padding:4px 8px;border-bottom:1px solid #23252e;position:sticky;top:0;background:#0e0f13}
td{padding:3px 8px;border-bottom:1px solid #181a20}
.pos{color:#6ee7b7}.neg{color:#f87171}.mut{color:#666}
.strip{height:20px;display:block;border-radius:2px;image-rendering:pixelated;width:180px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:12px 14px;margin-bottom:14px}
.flex{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
img.thumb{max-width:320px;border-radius:4px;border:1px solid #23252e}
.lbl{font-size:10px;text-transform:uppercase;color:#555;margin:4px 0 2px}
.heat td{text-align:center;font-variant-numeric:tabular-nums}
canvas{border:1px solid #23252e;border-radius:4px;background:#111}
.tag{font-size:10px;padding:1px 6px;border-radius:999px;background:#1d2029;border:1px solid #2c2f3a;color:#aaa}
</style></head><body>
<h1>Palette extractor — mass-imbalanced regime</h1>
<div class="sub" id="meta"></div>
<div id="root"></div>
<script>
const M = __PAYLOAD__;
const SB = '../../out/palette_imbalanced/strips/';
const RB = '../../out/palette_imbalanced/';
const f = (x,d=3)=> (x==null?'-':Number(x).toFixed(d));
const cls = d => Math.abs(d)<0.002?'mut':(d>0?'pos':'neg');
const root = document.getElementById('root');
document.getElementById('meta').innerHTML =
  `support_floor ON=${M.meta.support_on} · gammas=${M.meta.reg_gammas.join(', ')} · `+
  `cov eps=${M.meta.cov_eps} · render ${M.meta.render.join('×')} · maxiter ${M.meta.max_iter}`+
  `<br>Headline = directed GT→extracted arc-coverage within eps (higher=recovered thin subject). `+
  `NO promoted defaults.`;

function sec(t,sub){const h=document.createElement('h2');h.textContent=t;root.append(h);
  if(sub){const s=document.createElement('div');s.className='sub';s.innerHTML=sub;root.append(s);}}
function tbl(html){const d=document.createElement('div');d.innerHTML=html;root.append(d);}

// ---- Regression ----
sec('Build 1 — support-gate ON vs OFF (directed coverage)',
  'cov_Δ>0 = ON recovers more of the thin subject. branch_drop = chosen-comp nodes off the diameter path.');
let r='<table><tr><th>palette</th><th>γ</th><th>conc90</th>'+
  '<th>cov OFF</th><th>cov ON</th><th>cov Δ</th>'+
  '<th>cham OFF</th><th>cham ON</th>'+
  '<th>drop OFF</th><th>drop ON</th><th>dropExt ON</th>'+
  '<th>n_ridge OFF→ON</th><th>strips OFF / ON / GT</th></tr>';
for(const e of M.regression){
  const g=`reg_${e.name}_g${(+e.gamma)}`;
  r+=`<tr><td>${e.name}</td><td>${(+e.gamma)}</td><td>${f(e.conc90)}</td>`+
   `<td>${f(e.off.directed_cov)}</td><td>${f(e.on.directed_cov)}</td>`+
   `<td class=${cls(e.cov_delta)}>${e.cov_delta>0?'+':''}${f(e.cov_delta)}</td>`+
   `<td>${f(e.off.chamfer,4)}</td><td>${f(e.on.chamfer,4)}</td>`+
   `<td>${f(e.off.branch_drop)}</td><td>${f(e.on.branch_drop)}</td><td>${f(e.on.dropped_ext)}</td>`+
   `<td>${e.off.n_ridge}→${e.on.n_ridge}</td>`+
   `<td><img class=strip src='${SB}${g}_off.png'><img class=strip src='${SB}${g}_on.png'>`+
   `<img class=strip src='${SB}${g}_gt.png'></td></tr>`;
}
r+='</table>';tbl(r);

// summary of deltas
const cd=M.regression.map(e=>e.cov_delta);
const med=a=>{const s=[...a].sort((x,y)=>x-y);return s[Math.floor(s.length/2)];};
sec('Coverage-delta summary','');
tbl(`<div class=card>median cov_Δ = <b>${f(med(cd))}</b> · `+
  `min ${f(Math.min(...cd))} · max ${f(Math.max(...cd))} · `+
  `n with Δ>0.02: ${cd.filter(x=>x>0.02).length}/${cd.length}</div>`);

// ---- branch-drop distribution ----
sec('Build 2 — branch_drop_frac distribution (diagnostic only)',
  'Fraction of chosen-component voxels dropped by tree-diameter. dropped_extent = OKLab gyration of dropped set '+
  '(large = a real color excursion was discarded; small = fuzz near the path).');
{const on=M.regression.map(e=>e.on.branch_drop), de=M.regression.map(e=>e.on.dropped_ext);
 tbl(`<div class=card>ON: median branch_drop=<b>${f(med(on))}</b> max=${f(Math.max(...on))} · `+
  `median dropped_extent=<b>${f(de.filter(x=>x>0).sort((a,b)=>a-b)[Math.floor(de.filter(x=>x>0).length/2)]||0)}</b><br>`+
  `worst by branch_drop: `+
  M.regression.slice().sort((a,b)=>b.on.branch_drop-a.on.branch_drop).slice(0,5)
   .map(e=>`${e.name}@γ${+e.gamma} drop=${f(e.on.branch_drop)}/ext=${f(e.on.dropped_ext)}`).join(' · ')+
  `</div>`);}

// ---- Real cases ----
sec('Real failures — visual-first (no ground truth)',
  'Eyeball rope-vs-sheet; self_cov is self-reported. Floor swept OFF/30/150/600 because full-res reals '+
  'sit at a different mass scale than the synthetic renders.');
const RLAB={off:'OFF (sf=0)',on:'ON (sf=30)',hi:'sf=150',xhi:'sf=600'};
for(const e of M.real){
  let strips='';
  for(const k of ['off','on','hi','xhi']){const c=e[k];
   strips+=`<div class=lbl>${RLAB[k]} — cov ${f(c.self_cov*100,1)}% · ridge ${c.n_ridge} · `+
    `${c.closure} · drop ${f(c.branch_drop)} · dropExt ${f(c.dropped_ext)}</div>`+
    `<img class=strip style='width:340px' src='${SB}real_${e.name}_${k}.png'>`;}
  root.insertAdjacentHTML('beforeend',
  `<div class=card><div class=flex>
    <div><div class=lbl>original</div><img class=thumb src='${RB}real_${e.name}.jpg'></div>
    <div>${strips}</div></div><div class=sub>${e.name}</div></div>`);
}

// ---- Sweeps ----
sec('OAT sweeps (median directed coverage)','base = '+JSON.stringify(M.sweeps.base_kw));
for(const knob of Object.keys(M.sweeps.oat)){
  let t=`<b>${knob}</b><table><tr><th>value</th><th>median_cov</th><th>median_drop</th><th>median_ridge</th></tr>`;
  for(const row of M.sweeps.oat[knob])
    t+=`<tr><td>${row.value}</td><td>${f(row.median_cov)}</td><td>${f(row.median_drop)}</td><td>${f(row.median_ridge,1)}</td></tr>`;
  t+='</table>';tbl(t);
}

// ---- grid ----
sec('2-D grid: mass_fraction × support_floor (median directed coverage)','the coupled pair.');
{const mfs=[...new Set(M.sweeps.grid.map(g=>g.mass_fraction))];
 const sfs=[...new Set(M.sweeps.grid.map(g=>g.support_floor))];
 let t='<table class=heat><tr><th>mf ↓ / sf →</th>'+sfs.map(s=>`<th>${s}</th>`).join('')+'</tr>';
 const vals=M.sweeps.grid.map(g=>g.median_cov);const lo=Math.min(...vals),hi=Math.max(...vals);
 for(const mf of mfs){t+=`<tr><td>${mf}</td>`;
  for(const sf of sfs){const c=M.sweeps.grid.find(g=>g.mass_fraction===mf&&g.support_floor===sf);
   const v=c.median_cov;const a=hi>lo?(v-lo)/(hi-lo):0.5;
   t+=`<td style='background:rgba(74,106,244,${0.10+0.55*a})'>${f(v)}</td>`;}
  t+='</tr>';}
 t+='</table>';tbl(t);}

// ---- knn branch-drop ----
sec('branch_drop vs knn_k (topology lever)','does branch_drop_frac respond to graph density?');
{let t='<table><tr><th>knn_k</th><th>median_drop</th><th>median_cov</th></tr>';
 for(const row of M.sweeps.knn)
  t+=`<tr><td>${row.knn_k}</td><td>${f(row.median_drop)}</td><td>${f(row.median_cov)}</td></tr>`;
 t+='</table>';tbl(t);}
</script></body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(f"Wrote viewer: {out_html}")


# ============================================================================ #
# main
# ============================================================================ #

def main() -> None:
    ap = argparse.ArgumentParser(description="Palette bench — imbalanced regime")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    STRIPS_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = json.loads(PALETTES_JSON.read_text())
    name_to_stops = {e["name"]: e["stops"] for e in all_entries}
    print(f"Loaded {len(all_entries)} palettes")

    n_reg = 4 if args.quick else N_REG_PAL
    n_sw  = 2 if args.quick else N_SWEEP_PAL
    reg_pals = select_dark_low_palettes(name_to_stops, n_reg)
    sweep_pals = reg_pals[:n_sw]
    print(f"Regression palettes ({len(reg_pals)}): {reg_pals}")
    print(f"Sweep palettes ({len(sweep_pals)}): {sweep_pals}")

    gammas = [3.0] if args.quick else REG_GAMMAS
    globals()["REG_GAMMAS"] = gammas

    print(f"Rendering field {RENDER_W}×{RENDER_H} maxiter={MAX_ITER} …")
    t0 = time.monotonic()
    field = render_mandelbrot(RENDER_W, RENDER_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)
    print(f"  field done in {time.monotonic()-t0:.1f}s")

    regression = run_regression(reg_pals, name_to_stops, field)
    real = run_real_cases()
    sweeps = {} if args.quick else run_sweeps(sweep_pals, name_to_stops, field)
    if args.quick:
        sweeps = run_sweeps(sweep_pals, name_to_stops, field)

    manifest = {
        "meta": {
            "support_on": SUPPORT_ON, "reg_gammas": gammas, "cov_eps": COV_EPS,
            "render": [RENDER_W, RENDER_H], "max_iter": MAX_ITER,
            "reg_palettes": reg_pals, "sweep_palettes": sweep_pals,
            "n_reg": len(regression), "seed": SEED,
        },
        "regression": regression, "real": real, "sweeps": sweeps,
    }
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out}")
    write_viewer(manifest, VIZ_HTML)

    # console summary
    cd = [e["cov_delta"] for e in regression]
    on_drop = [e["on"]["branch_drop"] for e in regression]
    print(f"\n== SUMMARY ==")
    print(f"  cov_delta (ON-OFF): median={np.median(cd):+.4f} "
          f"min={min(cd):+.4f} max={max(cd):+.4f}  n(>0.02)={sum(d>0.02 for d in cd)}/{len(cd)}")
    print(f"  branch_drop ON: median={np.median(on_drop):.4f} max={max(on_drop):.4f}")


if __name__ == "__main__":
    main()
