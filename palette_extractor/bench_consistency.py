"""Palette-extractor consistency bench.

Render -> extract -> compare: color a render with a known palette (via palette_lib
LUT), extract a palette back from the render, score how well extraction recovers
the known ground-truth curve.

Coloring model: palette_lib.coloring.bake_lut (4096-entry linear-RGB LUT) is the
exclusive path — NOT eval_palette.build_lut (1024-entry sRGB LUT, no gamma).
The two differ in entry count, stored color space, and gamma handling.

Usage (from repo root):
  python palette_extractor/bench_consistency.py            # full run
  python palette_extractor/bench_consistency.py --spot     # 5-palette visual check only
"""
from __future__ import annotations
import sys, json, time, argparse
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
    oklab_to_linear_srgb,
)

# -- paths ---------------------------------------------------------------------

PALETTES_JSON  = ROOT / "data" / "palettes" / "clean_colormaps.json"
BENCH_DATA_DIR = ROOT / "data" / "palette_consistency"
STRIPS_DIR     = BENCH_DATA_DIR / "strips"
RENDERS_DIR    = ROOT / "out" / "palette_consistency"
VIZ_HTML       = ROOT / "tools" / "viz" / "palette_consistency.html"

# -- fixed bench parameters ----------------------------------------------------

RENDER_W, RENDER_H = 960, 640
MAX_ITER     = 600
DENSITY      = 1.0      # cycle rate: t = (field * DENSITY) % 1
N_BASELINE   = 75
N_SENSITIVITY = 20
BENCH_SEED   = 42
GT_DENSE     = 512      # ground-truth points over exercised arc


# -- helpers -------------------------------------------------------------------

def stops_to_list(stops_raw: list) -> list[tuple[float, tuple]]:
    return [(float(t), tuple(int(v) for v in rgb)) for t, rgb in stops_raw]


def render_with_palette(field: np.ndarray, stops_raw: list, out_png: Path) -> np.ndarray:
    """Render Mandelbrot field through palette_lib LUT, write PNG. Returns LUT."""
    lut     = bake_lut(stops_to_list(stops_raw))         # (4096,3) linear-RGB
    t_arr   = (field * DENSITY) % 1.0
    rgb_lin = lookup_linear(lut, t_arr)                   # (H,W,3) linear-RGB
    srgb    = (linear_to_srgb(rgb_lin) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(srgb).save(out_png)
    return lut


def ground_truth_lab(lut: np.ndarray, t_array: np.ndarray) -> np.ndarray:
    """OKLab ground-truth curve restricted to the exercised [t_min, t_max] arc.

    Sampled directly from the linear-RGB LUT (skipping uint8 round-trip for a
    cleaner baseline).  At DENSITY=1 the exercised range ≈ [0, 1), so this is
    essentially the full palette.
    """
    t_min = float(t_array.min())
    t_max = float(t_array.max())
    ts      = np.linspace(t_min, t_max, GT_DENSE)
    rgb_lin = lookup_linear(lut, ts)                      # (GT_DENSE, 3) linear-RGB
    return linear_srgb_to_oklab(rgb_lin)                  # (GT_DENSE, 3) OKLab


def exercised_fraction(t_array: np.ndarray, n_bins: int = 50) -> float:
    """Fraction of [0,1) palette-parameter bins that appear in the render."""
    hist, _ = np.histogram(t_array.ravel(), bins=n_bins, range=(0.0, 1.0))
    return float((hist > 0).sum()) / n_bins


def lab_strip_png(lab: np.ndarray, out_path: Path, w: int = 512, h: int = 32) -> None:
    """OKLab curve -> sRGB strip PNG."""
    n    = len(lab)
    idx  = (np.linspace(0, n - 1, w)).astype(int)
    lin  = oklab_to_linear_srgb(lab[idx])
    srgb = (linear_to_srgb(lin) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.tile(srgb[None], (h, 1, 1))).save(out_path)


# -- metrics -------------------------------------------------------------------

def chamfer(A: np.ndarray, B: np.ndarray) -> float:
    d_AB = cKDTree(B).query(A, k=1)[0].mean()
    d_BA = cKDTree(A).query(B, k=1)[0].mean()
    return float((d_AB + d_BA) / 2.0)


def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    d_AB = cKDTree(B).query(A, k=1)[0].max()
    d_BA = cKDTree(A).query(B, k=1)[0].max()
    return float(max(d_AB, d_BA))


def aligned_residual(gt_lab: np.ndarray, ext_stops: np.ndarray,
                     closure: str, N: int = 256) -> float:
    """Min mean OKLab distance over all cyclic shifts × {fwd, rev}.

    Mirrored closure: fold ext to its unique (first) half before the search.
    Fully vectorised: builds all N cyclic shifts at once via index tiling.
    """
    gt_r = resample_closed(gt_lab, N)                     # (N,3) uniform-arc

    if closure == "mirrored":
        half  = len(ext_stops) // 2
        ext_r = resample_closed(ext_stops[:half], N)
    else:
        ext_r = resample_closed(ext_stops, N)

    # All-shifts matrix: tile sequence and slice every offset
    # idx[i, j] = j+i -> ext_r[j+i] for all i,j in [0,N)
    idx       = (np.arange(N)[:, None] + np.arange(N)[None, :]) % N  # (N,N)

    best = np.inf
    for seq in (ext_r, ext_r[::-1]):
        all_shifts = seq[idx]                              # (N, N, 3)
        dists      = np.linalg.norm(gt_r[None] - all_shifts, axis=2).mean(axis=1)  # (N,)
        if dists.min() < best:
            best = float(dists.min())
    return best


# -- single-palette run --------------------------------------------------------

def run_one(name: str, stops_raw: list, field: np.ndarray, t_array: np.ndarray,
            out_png: Path, gt_strip: Path, ext_strip: Path,
            extract_kwargs: dict | None = None) -> dict:
    """Render, extract, score one palette. Returns a metric dict."""
    kw = extract_kwargs or {}

    lut    = render_with_palette(field, stops_raw, out_png)
    gt_lab = ground_truth_lab(lut, t_array)

    t0  = time.monotonic()
    res = extract_palette(out_png, **kw)
    dt  = time.monotonic() - t0

    ext_lab = res.stops_lab                               # (n_stops, 3) OKLab

    gt_strip.parent.mkdir(parents=True, exist_ok=True)
    lab_strip_png(gt_lab,  gt_strip)
    lab_strip_png(ext_lab, ext_strip)

    ch = chamfer(gt_lab, ext_lab)
    hd = hausdorff(gt_lab, ext_lab)
    ar = aligned_residual(gt_lab, ext_lab, res.closure)
    ex = exercised_fraction(t_array)

    return {
        "name":               name,
        "chamfer":            round(ch, 5),
        "hausdorff":          round(hd, 5),
        "aligned_residual":   round(ar, 5),
        "coverage":           round(float(res.coverage), 4),
        "closure":            res.closure,
        "exercised_fraction": round(ex, 4),
        "extract_s":          round(dt, 2),
        # subsampled OKLab for viewer scatter
        "gt_lab":  gt_lab[::max(1, GT_DENSE // 128)].tolist(),
        "ext_lab": ext_lab[::2].tolist(),
    }


# -- spot-check page -----------------------------------------------------------

def write_spot_html(entries: list[dict], out_html: Path) -> None:
    rows = []
    for e in entries:
        render_rel = f"../../out/palette_consistency/{e['name']}.png"
        gt_rel     = f"../../data/palette_consistency/strips/{e['name']}_gt.png"
        ext_rel    = f"../../data/palette_consistency/strips/{e['name']}_ext.png"
        scatter_js = (
            f"drawScatter('sc_{e['name']}', "
            f"{json.dumps(e['gt_lab'])}, "
            f"{json.dumps(e['ext_lab'])});"
        )
        rows.append(f"""
<section class="card">
  <h2>{e['name']}</h2>
  <p class="meta">chamfer={e['chamfer']:.4f}  hausdorff={e['hausdorff']:.4f}
     aligned={e['aligned_residual']:.4f}  closure={e['closure']}
     coverage={e['coverage']*100:.1f}%  exercised={e['exercised_fraction']*100:.0f}%</p>
  <div class="row">
    <figure><img src="{render_rel}"><figcaption>render (palette_lib LUT, density={DENSITY})</figcaption></figure>
    <div class="strips">
      <div><div class="strip-label">ground-truth (exercised arc)</div>
           <img src="{gt_rel}" class="strip" onerror="this.style.opacity=.2"></div>
      <div><div class="strip-label">extracted</div>
           <img src="{ext_rel}" class="strip" onerror="this.style.opacity=.2"></div>
    </div>
    <canvas id="sc_{e['name']}" width="220" height="220" class="scatter"></canvas>
  </div>
</section>
<script>{scatter_js}</script>""")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>palette bench — spot check</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#111;color:#ccc;font:13px/1.5 ui-monospace,monospace;padding:20px}}
h1{{font-size:15px;margin-bottom:8px}}
p.info{{font-size:11px;color:#666;margin-bottom:16px}}
.card{{background:#181818;border:1px solid #2a2a2a;border-radius:8px;padding:16px;margin-bottom:20px}}
h2{{font-size:13px;margin:0 0 4px}}
.meta{{font-size:11px;color:#888;margin-bottom:10px}}
.row{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}}
.row figure{{margin:0;max-width:480px;flex:1}}
.row figure img{{width:100%;border-radius:4px;display:block;border:1px solid #2a2a2a}}
figcaption{{font-size:10px;color:#666;margin-top:4px}}
.strips{{display:flex;flex-direction:column;gap:8px;min-width:200px;flex:0.6}}
.strip-label{{font-size:10px;text-transform:uppercase;color:#555;margin-bottom:2px}}
.strip{{width:100%;height:28px;display:block;border-radius:3px;image-rendering:pixelated}}
.scatter{{border:1px solid #2a2a2a;border-radius:4px}}
</style></head>
<body>
<h1>Palette bench — spot check ({len(entries)} palettes)</h1>
<p class="info">density={DENSITY} · {RENDER_W}×{RENDER_H} · maxiter={MAX_ITER}<br>
OKLab a–b scatter: <span style="color:rgba(80,200,120,0.9)">●</span> ground-truth exercised arc &nbsp;
<span style="color:rgba(255,130,60,0.9)">●</span> extracted stops<br>
<b>Check:</b> ground-truth strip colors should match the colors visible in the render.
If they differ, the exercised-mask logic is wrong — stop before running the full bench.</p>

{''.join(rows)}

<script>
function drawScatter(id, gt, ext) {{
  var c   = document.getElementById(id);
  var ctx = c.getContext("2d");
  ctx.fillStyle = "#111"; ctx.fillRect(0,0,220,220);
  ctx.strokeStyle = "#2a2a2a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(110,0); ctx.lineTo(110,220); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,110); ctx.lineTo(220,110); ctx.stroke();
  function dot(p, col, r) {{
    var x = p[1]*320+110, y = 110-p[2]*320;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2*Math.PI);
    ctx.fillStyle = col; ctx.fill();
  }}
  gt.forEach(p  => dot(p, "rgba(80,200,120,0.55)", 2.5));
  ext.forEach(p => dot(p, "rgba(255,130,60,0.80)", 3.5));
  var leg = [["rgba(80,200,120,0.8)","ground truth"],["rgba(255,130,60,0.8)","extracted"]];
  leg.forEach(function(l,i) {{
    ctx.fillStyle=l[0]; ctx.fillRect(6,6+i*16,9,9);
    ctx.fillStyle="#ccc"; ctx.font="11px monospace"; ctx.fillText(l[1],19,15+i*16);
  }});
}}
</script>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
    print(f"  -> spot check: {out_html}")


# -- sensitivity pass ----------------------------------------------------------

KNOB_VARIANTS: list[tuple[str, dict, dict]] = [
    ("voxel_res",     {"voxel_res": 32},        {"voxel_res": 64}),
    ("mass_fraction", {"mass_fraction": 0.80},   {"mass_fraction": 0.95}),
    ("knn_k",         {"knn_k": 5},             {"knn_k": 12}),
    ("trim_delta",    {"trim_delta": 0.03},      {"trim_delta": 0.10}),
    ("tau_close",     {"tau_close": 0.05},       {"tau_close": 0.20}),
    ("smooth_frac",   {"smooth_frac": 0.005},    {"smooth_frac": 0.025}),
]


def run_sensitivity(
    subset_names: list[str],
    name_to_stops: dict[str, list],
    field: np.ndarray,
    t_array: np.ndarray,
    renders_dir: Path,
) -> list[dict]:
    print(f"\n-- Sensitivity ({len(subset_names)} palettes × {len(KNOB_VARIANTS)} knobs × 2 dirs) --")

    # Baseline chamfers on subset (renders already exist)
    baseline_chs: list[float] = []
    for name in subset_names:
        out_png = renders_dir / f"{name}.png"
        if not out_png.exists():
            render_with_palette(field, name_to_stops[name], out_png)
        lut = bake_lut(stops_to_list(name_to_stops[name]))
        gt  = ground_truth_lab(lut, t_array)
        res = extract_palette(out_png)
        baseline_chs.append(chamfer(gt, res.stops_lab))

    baseline_med = float(np.median(baseline_chs))
    print(f"  baseline median chamfer (subset): {baseline_med:.5f}")

    rows: list[dict] = []
    for knob, low_kw, high_kw in KNOB_VARIANTS:
        for direction, kw in [("low", low_kw), ("high", high_kw)]:
            kw_key, kw_val = list(kw.items())[0]
            chs: list[float] = []
            for name in subset_names:
                out_png = renders_dir / f"{name}.png"
                try:
                    lut = bake_lut(stops_to_list(name_to_stops[name]))
                    gt  = ground_truth_lab(lut, t_array)
                    res = extract_palette(out_png, **kw)
                    chs.append(chamfer(gt, res.stops_lab))
                except Exception as exc:
                    print(f"    SKIP {name} ({knob}={kw_val}): {exc}")
            med   = float(np.median(chs)) if chs else float("nan")
            delta = med - baseline_med
            print(f"  {knob:15s} {direction:4s} {str(kw_val):<8}  "
                  f"median={med:.5f}  D={delta:+.5f}  n={len(chs)}")
            rows.append({
                "knob":           knob,
                "direction":      direction,
                "value":          kw_val,
                "median_chamfer": round(med, 5),
                "delta_chamfer":  round(delta, 5),
                "n_ok":           len(chs),
            })
    return rows


# -- viewer HTML ---------------------------------------------------------------

def write_viewer_html(out_html: Path) -> None:
    html = """\
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Palette consistency bench</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box;margin:0;padding:0}
body{font:13px/1.5 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
     background:#0e0f13;color:#ccc;display:flex;height:100vh;overflow:hidden}

#sidebar{width:240px;min-width:180px;display:flex;flex-direction:column;
         border-right:1px solid #23252e;overflow:hidden}
#sidebar-header{padding:10px 12px;border-bottom:1px solid #23252e}
#sidebar-header h1{font-size:12px;font-weight:600;color:#eee;margin-bottom:6px}
.sort-row{display:flex;gap:5px;flex-wrap:wrap}
.sort-btn{font-size:11px;padding:2px 7px;border-radius:4px;cursor:pointer;
          background:#1d2029;border:1px solid #2c2f3a;color:#aaa}
.sort-btn.active{background:#2a3050;border-color:#4a6af4;color:#7a9fff}
#pal-list{flex:1;overflow-y:auto;padding:4px 0}
.pal-item{padding:6px 12px;cursor:pointer;border-left:3px solid transparent;
          display:flex;flex-direction:column;gap:1px}
.pal-item:hover{background:#1a1c24}
.pal-item.active{border-left-color:#7a9fff;background:#161a2a}
.pal-name{font-size:11px;color:#ddd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pal-score{font-size:10px;color:#666}
.pal-item.err .pal-name{color:#7a4040}

#main{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:18px}
#main.empty{align-items:center;justify-content:center;color:#444;font-size:14px}

#meta-box{background:#15171d;border:1px solid #23252e;border-radius:8px;
          padding:12px 16px;font-size:11px;color:#888;line-height:1.7}
#meta-box b{color:#aaa}

.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:14px 16px}
.card h2{font-size:13px;color:#eee;margin-bottom:6px}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;background:#1d2029;
       border:1px solid #2c2f3a;color:#aaa}
.badge b{color:#e0e0e0}
.badge.native{background:#122a1e;border-color:#1f6b46;color:#7af0b8}
.badge.mirrored{background:#2a1a08;border-color:#6b3a10;color:#e8a05a}

.strip-row{display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap}
.strip-col{display:flex;flex-direction:column;gap:3px;flex:1;min-width:160px}
.strip-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#555}
.strip-img{width:100%;height:28px;display:block;border-radius:3px;image-rendering:pixelated}

.render-scatter-row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.render-box{flex:1;min-width:200px;max-width:480px}
.render-box img{width:100%;display:block;border-radius:4px;border:1px solid #23252e}
figcaption{font-size:10px;color:#555;margin-top:4px}

.scatter-box{display:flex;flex-direction:column;gap:4px}
.scatter-label{font-size:10px;text-transform:uppercase;color:#555}
.scatter-legend{font-size:10px;color:#555;margin-top:3px}
canvas.scatter{border:1px solid #23252e;border-radius:4px}

#sen-box{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:14px 16px;display:none}
#sen-box h2{font-size:13px;color:#eee;margin-bottom:8px}
table{border-collapse:collapse;width:100%;font-size:11px}
th{color:#888;font-weight:600;text-align:left;padding:4px 10px;border-bottom:1px solid #23252e}
td{padding:4px 10px;color:#ccc;border-bottom:1px solid #1a1c24}
.pos{color:#f87171}.neg{color:#6ee7b7}
</style></head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Palette consistency</h1>
    <div class="sort-row">
      <button class="sort-btn active" data-key="chamfer">chamfer ↑</button>
      <button class="sort-btn" data-key="hausdorff">hausdorff</button>
      <button class="sort-btn" data-key="aligned_residual">aligned</button>
      <button class="sort-btn" data-key="name">name</button>
    </div>
  </div>
  <div id="pal-list"></div>
</div>

<div id="main" class="empty">
  <div id="meta-box">loading…</div>
  <div id="sen-box">
    <h2>Sensitivity — one-at-a-time knob Δ(median chamfer)</h2>
    <table><thead><tr>
      <th>knob</th><th>direction</th><th>value</th>
      <th>median chamfer</th><th>Δ vs default</th><th>n_ok</th>
    </tr></thead><tbody id="sen-tbody"></tbody></table>
  </div>
  <div id="detail"></div>
</div>

<script>
const MANIFEST_URL    = '../../data/palette_consistency/manifest.json';
const SENSITIVITY_URL = '../../data/palette_consistency/sensitivity.json';
const RENDERS_BASE    = '../../out/palette_consistency/';
const STRIPS_BASE     = '../../data/palette_consistency/strips/';

let allEntries = [];
let sortKey    = 'chamfer';

document.querySelectorAll('.sort-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    sortKey = btn.dataset.key;
    document.querySelectorAll('.sort-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.key === sortKey));
    renderList();
  });
});

function renderList() {
  const sorted = [...allEntries].sort((a, b) => {
    if (sortKey === 'name') return a.name.localeCompare(b.name);
    return (a[sortKey] ?? 1e9) - (b[sortKey] ?? 1e9);
  });
  const list = document.getElementById('pal-list');
  list.innerHTML = '';
  sorted.forEach(e => {
    const div = document.createElement('div');
    div.className = 'pal-item' + (e.error ? ' err' : '');
    div.dataset.name = e.name;
    div.innerHTML = `<div class="pal-name">${e.name}</div>
      <div class="pal-score">${e.error ? 'ERROR' : `ch=${e.chamfer.toFixed(4)} · ${e.closure}`}</div>`;
    div.addEventListener('click', () => showDetail(e));
    list.appendChild(div);
  });
}

function showDetail(e) {
  document.querySelectorAll('.pal-item').forEach(el =>
    el.classList.toggle('active', el.dataset.name === e.name));
  document.getElementById('main').className = '';
  const det = document.getElementById('detail');
  if (e.error) {
    det.innerHTML = `<div class="card"><h2>${e.name}</h2>
      <p style="color:#f87171">${e.error}</p></div>`;
    return;
  }
  const closureCls = e.closure || '';
  det.innerHTML = `
<div class="card">
  <h2>${e.name}</h2>
  <div class="badges">
    <span class="badge ${closureCls}">${e.closure}</span>
    <span class="badge">chamfer <b>${e.chamfer.toFixed(5)}</b></span>
    <span class="badge">hausdorff <b>${e.hausdorff.toFixed(5)}</b></span>
    <span class="badge">aligned_residual <b>${e.aligned_residual.toFixed(5)}</b></span>
    <span class="badge">coverage <b>${(e.coverage*100).toFixed(1)}%</b></span>
    <span class="badge">exercised <b>${(e.exercised_fraction*100).toFixed(0)}%</b></span>
  </div>

  <div class="strip-row">
    <div class="strip-col">
      <div class="strip-label">ground-truth (exercised arc)</div>
      <img src="${STRIPS_BASE}${e.name}_gt.png" class="strip-img" onerror="this.style.opacity=.2">
    </div>
    <div class="strip-col">
      <div class="strip-label">extracted</div>
      <img src="${STRIPS_BASE}${e.name}_ext.png" class="strip-img" onerror="this.style.opacity=.2">
    </div>
  </div>

  <div class="render-scatter-row">
    <figure class="render-box">
      <img src="${RENDERS_BASE}${e.name}.png" onerror="this.style.opacity=.2">
      <figcaption>render · palette_lib LUT · density=${e.density||1}</figcaption>
    </figure>
    <div class="scatter-box">
      <div class="scatter-label">OKLab a–b scatter</div>
      <canvas id="scatter-cv" width="220" height="220" class="scatter"></canvas>
      <div class="scatter-legend">
        <span style="color:rgba(80,200,120,0.9)">●</span> ground truth &nbsp;
        <span style="color:rgba(255,130,60,0.9)">●</span> extracted
      </div>
    </div>
  </div>
</div>`;

  drawScatter('scatter-cv', e.gt_lab || [], e.ext_lab || []);
}

function drawScatter(id, gt, ext) {
  const c   = document.getElementById(id);
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, 220, 220);
  ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 1;
  [[110,0,110,220],[0,110,220,110]].forEach(([x1,y1,x2,y2]) => {
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  });
  const toXY = p => [p[1]*320+110, 110-p[2]*320];
  const dot  = (p, col, r) => {
    const [x,y] = toXY(p);
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2*Math.PI);
    ctx.fillStyle = col; ctx.fill();
  };
  gt.forEach(p  => dot(p, 'rgba(80,200,120,0.55)', 2.5));
  ext.forEach(p => dot(p, 'rgba(255,130,60,0.80)', 3.5));
}

async function loadSensitivity() {
  try {
    const r = await fetch(SENSITIVITY_URL);
    if (!r.ok) return;
    const d = await r.json();
    const box = document.getElementById('sen-box');
    const tb  = document.getElementById('sen-tbody');
    box.style.display = '';
    d.rows.forEach(row => {
      const tr = document.createElement('tr');
      const delta = row.delta_chamfer;
      const dc = Math.abs(delta) < 0.0005 ? '' : delta > 0 ? 'pos' : 'neg';
      tr.innerHTML = `<td>${row.knob}</td><td>${row.direction}</td><td>${row.value}</td>
        <td>${row.median_chamfer.toFixed(5)}</td>
        <td class="${dc}">${delta > 0 ? '+' : ''}${delta.toFixed(5)}</td>
        <td>${row.n_ok}</td>`;
      tb.appendChild(tr);
    });
  } catch(e) { console.warn('sensitivity not loaded:', e); }
}

async function init() {
  try {
    const r = await fetch(MANIFEST_URL);
    const d = await r.json();
    allEntries = d.entries || [];
    const m   = d.meta || {};
    document.getElementById('meta-box').innerHTML =
      `<b>n</b>=${m.n_baseline} &nbsp; <b>seed</b>=${m.seed} &nbsp; ` +
      `<b>density</b>=${m.density} &nbsp; ` +
      `<b>render</b>=${(m.render_wh||[]).join('×')} &nbsp; ` +
      `<b>maxiter</b>=${m.max_iter}<br>` +
      `<b>chamfer</b>: median=${m.chamfer_median?.toFixed(5)} &nbsp; ` +
      `p25=${m.chamfer_p25?.toFixed(5)} &nbsp; p75=${m.chamfer_p75?.toFixed(5)} &nbsp; ` +
      `max=${m.chamfer_max?.toFixed(5)}<br>` +
      `<b>closure</b>: native=${m.n_native} / mirrored=${m.n_mirrored}<br>` +
      `<b>exercised_fraction</b>: mean=${m.exercised_fraction_mean?.toFixed(4)} ` +
      `± ${m.exercised_fraction_std?.toFixed(4)} ` +
      `<em>(sanity: should be ~constant across all palettes)</em>`;
    document.getElementById('main').className = '';
    renderList();
    await loadSensitivity();
  } catch(e) {
    document.getElementById('meta-box').textContent = 'Error loading manifest: ' + e;
  }
}

init();
</script>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote viewer: {out_html}")


# -- main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Palette extractor consistency bench")
    ap.add_argument("--spot", action="store_true", help="5-palette visual check only")
    ap.add_argument("--n-baseline",   type=int, default=N_BASELINE)
    ap.add_argument("--n-sensitivity", type=int, default=N_SENSITIVITY)
    ap.add_argument("--seed", type=int, default=BENCH_SEED)
    args = ap.parse_args()

    BENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STRIPS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    # Load palette library
    all_entries: list[dict] = json.loads(PALETTES_JSON.read_text())
    name_to_stops = {e["name"]: e["stops"] for e in all_entries}
    print(f"Loaded {len(all_entries)} palettes from clean_colormaps.json")

    rng   = np.random.default_rng(args.seed)
    idxs  = rng.choice(len(all_entries), size=min(args.n_baseline, len(all_entries)), replace=False)
    baseline_pairs = [(all_entries[i]["name"], all_entries[i]["stops"]) for i in idxs]

    # Render field once (same for all palettes — palette-independent)
    print(f"Rendering Mandelbrot field ({RENDER_W}×{RENDER_H}, maxiter={MAX_ITER}) …")
    t0_field = time.monotonic()
    field   = render_mandelbrot(RENDER_W, RENDER_H,
                                center=SPIRAL_CENTER, half_w=SPIRAL_HALF_W,
                                max_iter=MAX_ITER)
    t_array = (field * DENSITY) % 1.0
    ex_frac = exercised_fraction(t_array)
    print(f"  done in {time.monotonic()-t0_field:.1f}s  "
          f"t in [{t_array.min():.4f}, {t_array.max():.4f}]  "
          f"exercised fraction: {ex_frac*100:.1f}%")

    # -- Spot check (first 5 palettes) --
    print(f"\n-- Spot check (5 palettes) --")
    spot_results: list[dict] = []
    for name, stops_raw in baseline_pairs[:5]:
        print(f"  {name} … ", end="", flush=True)
        try:
            e = run_one(name, stops_raw, field, t_array,
                        RENDERS_DIR / f"{name}.png",
                        STRIPS_DIR  / f"{name}_gt.png",
                        STRIPS_DIR  / f"{name}_ext.png")
            spot_results.append(e)
            print(f"chamfer={e['chamfer']:.4f}  closure={e['closure']}  "
                  f"cov={e['coverage']*100:.0f}%  ex={e['exercised_fraction']*100:.0f}%  "
                  f"[{e['extract_s']:.1f}s]")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"FAILED: {exc}")

    write_spot_html(spot_results, RENDERS_DIR / "spot_check.html")

    if args.spot:
        print("\n--spot mode: done. Check spot_check.html before proceeding.")
        return

    # -- Full baseline --
    print(f"\n-- Full baseline ({len(baseline_pairs)} palettes) --")
    results: list[dict] = list(spot_results)

    for name, stops_raw in baseline_pairs[5:]:
        print(f"  {name} … ", end="", flush=True)
        try:
            e = run_one(name, stops_raw, field, t_array,
                        RENDERS_DIR / f"{name}.png",
                        STRIPS_DIR  / f"{name}_gt.png",
                        STRIPS_DIR  / f"{name}_ext.png")
            results.append(e)
            print(f"chamfer={e['chamfer']:.4f}  "
                  f"closure={e['closure']}  [{e['extract_s']:.1f}s]")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"name": name, "error": str(exc)})

    ok = [e for e in results if "chamfer" in e]
    chamfers = np.array([e["chamfer"] for e in ok])
    closures = [e["closure"] for e in ok]
    ex_fracs = [e["exercised_fraction"] for e in ok]

    print(f"\n-- Baseline summary ({len(ok)}/{len(results)} OK) --")
    print(f"  chamfer:  median={np.median(chamfers):.5f}  "
          f"p25={np.percentile(chamfers,25):.5f}  "
          f"p75={np.percentile(chamfers,75):.5f}  "
          f"max={chamfers.max():.5f}")
    print(f"  closure:  native={sum(c=='native' for c in closures)}  "
          f"mirrored={sum(c=='mirrored' for c in closures)}")
    print(f"  exercised fraction: mean={np.mean(ex_fracs):.4f} ± {np.std(ex_fracs):.4f}")
    print("  worst 5 by chamfer:")
    for e in sorted(ok, key=lambda x: -x["chamfer"])[:5]:
        print(f"    {e['name']:30s}  chamfer={e['chamfer']:.5f}  closure={e['closure']}")

    manifest = {
        "meta": {
            "n_baseline":             len(ok),
            "seed":                   args.seed,
            "density":                DENSITY,
            "render_wh":              [RENDER_W, RENDER_H],
            "max_iter":               MAX_ITER,
            "spiral_center":          list(SPIRAL_CENTER),
            "chamfer_median":         round(float(np.median(chamfers)), 5),
            "chamfer_p25":            round(float(np.percentile(chamfers, 25)), 5),
            "chamfer_p75":            round(float(np.percentile(chamfers, 75)), 5),
            "chamfer_max":            round(float(chamfers.max()), 5),
            "n_native":               int(sum(c == "native"   for c in closures)),
            "n_mirrored":             int(sum(c == "mirrored" for c in closures)),
            "exercised_fraction_mean": round(float(np.mean(ex_fracs)), 4),
            "exercised_fraction_std":  round(float(np.std(ex_fracs)), 4),
        },
        "entries": sorted(ok, key=lambda e: e["chamfer"]),
    }
    manifest_path = BENCH_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")

    # -- Sensitivity pass --
    sen_rng = np.random.default_rng(args.seed + 1)
    sen_idx = sen_rng.choice(len(ok), size=min(args.n_sensitivity, len(ok)), replace=False)
    sen_names = [ok[i]["name"] for i in sen_idx]
    sen_rows  = run_sensitivity(sen_names, name_to_stops, field, t_array, RENDERS_DIR)

    sen_path = BENCH_DATA_DIR / "sensitivity.json"
    sen_path.write_text(json.dumps({"rows": sen_rows}, indent=2))
    print(f"Wrote {sen_path}")

    # -- Viewer --
    write_viewer_html(VIZ_HTML)


if __name__ == "__main__":
    main()
