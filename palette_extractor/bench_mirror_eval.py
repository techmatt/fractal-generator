"""Phase-2 verification: does selective pre-mirror clear the seam-sourced spurious
extreme stops on SEQUENTIAL maps?

Re-runs the established synthetic-field roundtrip (same render -> extract -> GT
path as bench_synthetic_eval.py) at the DEFAULT admission (sf0: mass_fraction=0.90,
support_floor=0.0), comparing mirror OFF vs ON on every sequential (mirror_needed)
map plus a cyclic control set.

Spurious-extreme-stop diagnostic (per map, OFF vs ON):
  count extracted control points at OKLab L < DARK_L or > LIGHT_L whose nearest
  GT *defining stop* is farther than COV_EPS in OKLab (extracted-but-not-GT).
  The seam hypothesis predicts this DROPS on sequential maps after mirror, scaling
  with wj; near-0-wj cyclic maps don't have a seam to clear.

Mirror ON renders at the product's compensated density (DENSITY * MIRROR_DENSITY_SCALE),
matching coloring::shade. GT defining-stop color set is mirror-invariant (reflection
adds positions, not colors), so the OFF->ON delta is purely the extracted stops moving.

Residual: spurious extremes that do NOT clear after mirror are NOT seam — they are
admission-sourced sparse-tail spurs (population B); listed separately, not acted on.

Usage (from repo root):
  python palette_extractor/bench_mirror_eval.py            # full run (background it)
  python palette_extractor/bench_mirror_eval.py --quick    # sequential subset + controls
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

from palette_extract import extract_palette_cycles
from eval_palette import render_mandelbrot, SPIRAL_CENTER, SPIRAL_HALF_W
from palette_lib.coloring import (
    bake_lut, lookup_linear, linear_to_srgb, srgb8_to_oklab, MIRROR_DENSITY_SCALE,
)
from bench_cycle_closure import true_wrap_jump, stops_to_list, lab_strip_png
from bench_consistency import ground_truth_lab

PALETTES_JSON = ROOT / "data" / "palettes" / "clean_colormaps.json"
DATA_DIR      = ROOT / "data" / "palette_mirror_eval"      # load-bearing manifest
RENDERS_DIR   = ROOT / "out"  / "palette_mirror_eval"      # regenerable views
STRIPS_DIR    = RENDERS_DIR / "strips"
VIZ_HTML      = ROOT / "tools" / "viz" / "palette_mirror_eval.html"

# fixed bench params (reported, NOT tuned) ------------------------------------
RENDER_W, RENDER_H = 960, 640
MAX_ITER  = 600
DENSITY   = 1.0
COV_EPS   = 0.05      # OKLab "nearby GT stop" radius (matches extractor coverage_eps)
DARK_L    = 0.30     # OKLab L below this = "dark extreme"
LIGHT_L   = 0.90     # OKLab L above this = "light extreme"
# default admission (sf0) — the shipped default; admission is NOT tuned here.
ADMISSION = {"mass_fraction": 0.90, "support_floor": 0.0}
EXTRACT_KW = dict(lam=2.0, arc_retain=0.5, seam_seq_threshold=0.10, **ADMISSION)
STRIP_WJ_THR = 0.5   # high-wj sequential subset for before/after strips


def gt_stops_lab(stops_raw) -> np.ndarray:
    """OKLab of the palette's DEFINING stops (mirror-invariant color set)."""
    return srgb8_to_oklab(np.array([c for _, c in stops_to_list(stops_raw)], float))


def render_through(field, stops_raw, mirror: bool, out_png: Path):
    """Render the field through the (optionally pre-mirrored) palette; product
    density compensation applied when mirror. Returns (lut, t, oklab_image_unused)."""
    lut = bake_lut(stops_to_list(stops_raw), mirror=mirror)
    density = DENSITY * (MIRROR_DENSITY_SCALE if mirror else 1.0)
    t = (field * density) % 1.0
    srgb = (linear_to_srgb(lookup_linear(lut, t)) * 255).clip(0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(srgb).save(out_png)
    return lut, t


def spurious_extremes(ext_lab: np.ndarray, gt_lab: np.ndarray) -> dict:
    """Count extracted control points at extreme OKLab L with no nearby GT stop."""
    if ext_lab is None or len(ext_lab) == 0:
        return {"dark": 0, "light": 0, "total": 0, "names_L": []}
    L = ext_lab[:, 0]
    extreme = (L < DARK_L) | (L > LIGHT_L)
    tree = cKDTree(gt_lab)
    d = tree.query(ext_lab, k=1)[0]
    far = d > COV_EPS
    spur = extreme & far
    dark = int(np.count_nonzero(spur & (L < DARK_L)))
    light = int(np.count_nonzero(spur & (L > LIGHT_L)))
    return {"dark": dark, "light": light, "total": int(np.count_nonzero(spur)),
            "spur_L": [round(float(x), 3) for x in L[spur]]}


def eval_map(field, e, write_strips: bool) -> dict:
    nm = e["name"]
    stops = e["stops"]
    gt_stops = gt_stops_lab(stops)
    wj = true_wrap_jump(stops)
    is_seq = bool(e["mirror_needed"])
    row = {"name": nm, "wj": round(wj, 4), "is_sequential": is_seq, "per": {}}

    for tag, mirror in [("off", False), ("on", True)]:
        # selective: a CYCLIC map never mirrors -> 'on' == 'off' by construction.
        eff_mirror = mirror and is_seq
        png = RENDERS_DIR / f"{nm}_{tag}.png"
        lut, t = render_through(field, stops, eff_mirror, png)
        gt_arc = ground_truth_lab(lut, t)            # exercised arc (for strips)
        try:
            r = extract_palette_cycles(png, **EXTRACT_KW)
            sp_open  = spurious_extremes(r.stops_open_lab,  gt_stops)
            sp_cycle = spurious_extremes(r.stops_cycle_lab, gt_stops)
            row["per"][tag] = {
                "spur_open": sp_open["total"], "spur_cycle": sp_cycle["total"],
                "spur_open_dark": sp_open["dark"], "spur_open_light": sp_open["light"],
                "cycle_label": r.cycle_label,
                "n_open": int(len(r.stops_open_lab)), "n_cycle": int(len(r.stops_cycle_lab)),
            }
            if write_strips:
                lab_strip_png(gt_arc, STRIPS_DIR / f"{nm}_{tag}_gt.png")
                lab_strip_png(r.stops_open_lab,  STRIPS_DIR / f"{nm}_{tag}_open.png")
                lab_strip_png(r.stops_cycle_lab, STRIPS_DIR / f"{nm}_{tag}_cycle.png")
        except Exception as exc:
            row["per"][tag] = {"error": str(exc)}

    o, n = row["per"].get("off", {}), row["per"].get("on", {})
    if "spur_open" in o and "spur_open" in n:
        row["drop_open"]  = o["spur_open"]  - n["spur_open"]
        row["drop_cycle"] = o["spur_cycle"] - n["spur_cycle"]
        row["resid_open"] = n["spur_open"]   # spurious remaining after mirror (population B)
    return row


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="sequential subset + controls")
    ap.add_argument("--resummarize", action="store_true",
                    help="rebuild summary+viewer from existing manifest (no re-render)")
    args = ap.parse_args()

    if args.resummarize:
        resummarize()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STRIPS_DIR.mkdir(parents=True, exist_ok=True)

    entries = json.loads(PALETTES_JSON.read_text())
    seq = [e for e in entries if e["mirror_needed"]]
    cyc = [e for e in entries if not e["mirror_needed"]]
    print(f"Loaded {len(entries)}: {len(seq)} sequential, {len(cyc)} cyclic")

    if args.quick:
        seq_sorted = sorted(seq, key=lambda e: -true_wrap_jump(e["stops"]))
        idx = np.linspace(0, len(seq_sorted) - 1, 16).astype(int)
        run_seq = [seq_sorted[i] for i in idx]
        run_cyc = cyc[:4]
    else:
        run_seq, run_cyc = seq, cyc
    run = run_seq + run_cyc
    print(f"Evaluating {len(run)} maps ({len(run_seq)} sequential, {len(run_cyc)} cyclic control)")

    print(f"Rendering field {RENDER_W}x{RENDER_H} maxiter={MAX_ITER} ...")
    t0 = time.monotonic()
    field = render_mandelbrot(RENDER_W, RENDER_H, center=SPIRAL_CENTER,
                              half_w=SPIRAL_HALF_W, max_iter=MAX_ITER)
    print(f"  field {time.monotonic()-t0:.1f}s")

    rows = []
    t0 = time.monotonic()
    for i, e in enumerate(run):
        wj = true_wrap_jump(e["stops"])
        write_strips = (e["mirror_needed"] and wj >= STRIP_WJ_THR) or (not e["mirror_needed"] and i < len(run_seq) + 4)
        rows.append(eval_map(field, e, write_strips))
        if (i + 1) % 20 == 0 or i == len(run) - 1:
            print(f"  [{i+1}/{len(run)}] {e['name']:30s} done")
    print(f"  eval pass {time.monotonic()-t0:.1f}s")

    manifest = build_manifest(rows)
    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    write_viewer(manifest)
    print_summary(manifest, out)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 3) if xs else None


def build_manifest(rows) -> dict:
    """Aggregate per-map rows into the manifest. PRIMARY metric is the CYCLE
    reconstruction (the class-adaptive output a sequential map actually uses
    downstream — closes native if cyclic, mirrors if sequential); the OPEN path
    is reported as a seam-robust secondary (it ignores the thin wrap band)."""
    seq_rows = sorted([r for r in rows if r["is_sequential"]], key=lambda r: -r["wj"])
    cyc_rows = [r for r in rows if not r["is_sequential"]]

    def drop_cycle(r):
        o, n = r["per"].get("off", {}), r["per"].get("on", {})
        if "spur_cycle" in o and "spur_cycle" in n:
            return o["spur_cycle"] - n["spur_cycle"]
        return None

    def resid_cycle(r):
        return r["per"].get("on", {}).get("spur_cycle")

    summary = {
        "n_sequential": len(seq_rows), "n_cyclic": len(cyc_rows),
        # PRIMARY: cycle reconstruction
        "seq_spur_cycle_off_median": _med([r["per"]["off"].get("spur_cycle") for r in seq_rows]),
        "seq_spur_cycle_on_median":  _med([r["per"]["on"].get("spur_cycle") for r in seq_rows]),
        "seq_spur_cycle_off_total":  int(np.nansum([r["per"]["off"].get("spur_cycle", 0) for r in seq_rows])),
        "seq_spur_cycle_on_total":   int(np.nansum([r["per"]["on"].get("spur_cycle", 0) for r in seq_rows])),
        "seq_drop_cycle_total":      int(np.nansum([drop_cycle(r) or 0 for r in seq_rows])),
        "n_seq_with_cycle_spur_off": int(sum(1 for r in seq_rows if r["per"]["off"].get("spur_cycle", 0) > 0)),
        "n_seq_with_cycle_spur_on":  int(sum(1 for r in seq_rows if r["per"]["on"].get("spur_cycle", 0) > 0)),
        # SECONDARY: open path (seam-robust)
        "seq_spur_open_off_total":   int(np.nansum([r["per"]["off"].get("spur_open", 0) for r in seq_rows])),
        "seq_spur_open_on_total":    int(np.nansum([r["per"]["on"].get("spur_open", 0) for r in seq_rows])),
        # cyclic control: should be unchanged (selective never mirrors them)
        "cyc_spur_cycle_off_total":  int(np.nansum([r["per"]["off"].get("spur_cycle", 0) for r in cyc_rows])),
        "cyc_spur_cycle_on_total":   int(np.nansum([r["per"]["on"].get("spur_cycle", 0) for r in cyc_rows])),
    }
    # residual = spurious cycle stops that did NOT clear after mirror (population B)
    residual = [{"name": r["name"], "wj": r["wj"], "resid_cycle": resid_cycle(r),
                 "drop_cycle": drop_cycle(r)}
                for r in seq_rows if (resid_cycle(r) or 0) > 0]
    residual.sort(key=lambda d: -d["resid_cycle"])

    return {
        "meta": {
            "render": [RENDER_W, RENDER_H], "max_iter": MAX_ITER, "density": DENSITY,
            "mirror_density_scale": MIRROR_DENSITY_SCALE, "admission": ADMISSION,
            "cov_eps": COV_EPS, "dark_L": DARK_L, "light_L": LIGHT_L,
            "strip_wj_thr": STRIP_WJ_THR,
        },
        "summary": summary, "residual_admission_sourced": residual,
        "seq_rows": seq_rows, "cyc_rows": cyc_rows,
    }


def print_summary(manifest, out):
    s = manifest["summary"]
    seq_rows = manifest["seq_rows"]
    print("\n== SUMMARY (sequential, default admission sf0; PRIMARY = cycle reconstruction) ==")
    print(f"  spurious CYCLE extreme stops  median  OFF={s['seq_spur_cycle_off_median']}  "
          f"ON={s['seq_spur_cycle_on_median']}   total OFF={s['seq_spur_cycle_off_total']} ON={s['seq_spur_cycle_on_total']} "
          f"(dropped {s['seq_drop_cycle_total']})")
    print(f"  maps with any cycle spurious: OFF={s['n_seq_with_cycle_spur_off']} ON={s['n_seq_with_cycle_spur_on']} "
          f"of {s['n_sequential']}")
    print(f"  open-path (seam-robust) total OFF={s['seq_spur_open_off_total']} ON={s['seq_spur_open_on_total']}")
    print(f"  cyclic control total OFF={s['cyc_spur_cycle_off_total']} ON={s['cyc_spur_cycle_on_total']} (must be equal)")
    print("\n  movers (cycle spur OFF->ON, |drop|>0), by wj:")
    movers = [r for r in seq_rows if (r['per']['off'].get('spur_cycle',0) - r['per']['on'].get('spur_cycle',0)) != 0]
    for r in movers[:20]:
        o, n = r["per"]["off"], r["per"]["on"]
        print(f"   {r['name']:30s} wj={r['wj']:.3f}  cycle {o.get('spur_cycle','-')}->{n.get('spur_cycle','-')}"
              f"   open {o.get('spur_open','-')}->{n.get('spur_open','-')}")
    if not movers:
        print("   (none — no map's cycle spurious count changes OFF->ON)")
    if manifest["residual_admission_sourced"]:
        print(f"\n  RESIDUAL (cycle spurious surviving mirror — admission-sourced, NOT seam): "
              + ", ".join(f"{d['name']}({d['resid_cycle']})" for d in manifest["residual_admission_sourced"][:15]))
    print(f"\nWrote {out}\nWrote viewer {VIZ_HTML}")


def resummarize():
    """Rebuild summary + viewer from an existing manifest's per-map rows (no
    re-render). Used to re-aggregate after a full eval pass."""
    out = DATA_DIR / "manifest.json"
    m = json.loads(out.read_text())
    rows = m["seq_rows"] + m["cyc_rows"]
    manifest = build_manifest(rows)
    out.write_text(json.dumps(manifest, indent=2))
    write_viewer(manifest)
    print_summary(manifest, out)


def write_viewer(manifest):
    payload = json.dumps(manifest)
    html = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 2 — selective mirror seam verification</title>
<style>
:root{color-scheme:dark}body{font:13px/1.5 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;padding:18px 26px}
h1{font-size:16px;color:#eee}h2{font-size:14px;color:#eee;margin:20px 0 6px}
.sub{color:#888;font-size:11px;max-width:1150px;margin-bottom:8px}
.card{background:#15171d;border:1px solid #23252e;border-radius:8px;padding:12px 16px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:11px}
th{color:#8aa;text-align:left;padding:4px 8px;border-bottom:1px solid #23252e;cursor:pointer}
td{padding:3px 8px;border-bottom:1px solid #181a20;font-variant-numeric:tabular-nums}
.strip{height:18px;display:block;image-rendering:pixelated;width:200px;border-radius:2px}
.good{color:#6ee7b7}.bug{color:#f87171}.mut{color:#666}
.tag{font-size:10px;padding:1px 6px;border-radius:999px;border:1px solid #2c2f3a}
.native{background:#122a1e;border-color:#1f6b46;color:#7af0b8}.sequential{background:#2a1a08;border-color:#6b3a10;color:#e8a05a}
</style></head><body>
<h1>Phase 2 — selective pre-mirror: seam-sourced spurious-stop verification</h1>
<div class="sub" id="meta"></div>
<div id="root"></div>
<script>
const M=__PAYLOAD__, SB='../../out/palette_mirror_eval/strips/';
const f=(x,d=3)=>(x==null?'-':Number(x).toFixed(d));
const root=document.getElementById('root');
const m=M.meta,s=M.summary;
document.getElementById('meta').innerHTML=
 `render ${m.render.join('x')} maxiter ${m.max_iter} · admission sf0 (mf ${m.admission.mass_fraction}, support_floor ${m.admission.support_floor}) · `+
 `extreme L &lt; <b>${m.dark_L}</b> or &gt; <b>${m.light_L}</b> · nearby GT &le; <b>${m.cov_eps}</b> OKLab · mirror density x${m.mirror_density_scale}. `+
 `Spurious = extracted control point at extreme L with no nearby GT defining stop. NO quality claim — judge the strips.`;
function h2(t){const e=document.createElement('h2');e.textContent=t;root.append(e);}
function card(h){const d=document.createElement('div');d.className='card';d.innerHTML=h;root.append(d);}

h2('Summary — sequential maps, mirror OFF vs ON (PRIMARY = cycle reconstruction)');
card(`<b>Cycle reconstruction</b> (class-adaptive output): spurious extreme stops total OFF <b>${s.seq_spur_cycle_off_total}</b> -> ON <b class=good>${s.seq_spur_cycle_on_total}</b> `+
 `(dropped <b class=good>${s.seq_drop_cycle_total}</b>) · median OFF ${f(s.seq_spur_cycle_off_median)} -> ON ${f(s.seq_spur_cycle_on_median)}<br>`+
 `maps with any cycle spurious: OFF <b>${s.n_seq_with_cycle_spur_off}</b> -> ON <b class=good>${s.n_seq_with_cycle_spur_on}</b> (of ${s.n_sequential})<br>`+
 `<span class=mut>Open path (seam-robust secondary): total OFF ${s.seq_spur_open_off_total} -> ON ${s.seq_spur_open_on_total}. `+
 `Cyclic control (selective never mirrors): OFF ${s.cyc_spur_cycle_off_total} = ON ${s.cyc_spur_cycle_on_total}.</span>`);

if(M.residual_admission_sourced.length){
 let t='<b>Residual cycle-spurious after mirror</b> — admission-sourced sparse-tail (population B), NOT seam; candidate for a SEPARATE admission decision, not this prompt.'+
  '<table><tr><th>map</th><th>wj</th><th>resid_cycle</th><th>drop_cycle</th></tr>';
 for(const r of M.residual_admission_sourced)t+=`<tr><td>${r.name}</td><td>${f(r.wj)}</td><td>${r.resid_cycle}</td><td>${f(r.drop_cycle,0)}</td></tr>`;
 card(t+'</table>');
}

h2('Per-map spurious table (sequential, ordered by wj desc)');
{let t='<table><tr><th>map</th><th>wj</th><th>spur_open OFF->ON</th><th>spur_cycle OFF->ON</th><th>label OFF->ON</th></tr>';
 for(const r of M.seq_rows){const o=r.per.off,n=r.per.on;
  const drop=(o.spur_open||0)-(n.spur_open||0);
  t+=`<tr><td>${r.name}</td><td>${f(r.wj)}</td>`+
   `<td class=${drop>0?'good':''}>${o.spur_open}->${n.spur_open}</td>`+
   `<td>${o.spur_cycle}->${n.spur_cycle}</td>`+
   `<td><span class="tag ${o.cycle_label}">${o.cycle_label}</span> -> <span class="tag ${n.cycle_label}">${n.cycle_label}</span></td></tr>`;}
 card(t+'</table>');}

h2('Before/after strips (high-wj sequential) — GT · extracted open · extracted cycle');
{let t='<table><tr><th>map</th><th>mirror OFF (gt / open / cycle)</th><th>mirror ON (gt / open / cycle)</th></tr>';
 for(const r of M.seq_rows){if(r.wj < M.meta.strip_wj_thr)continue;
  const g=(tag,k)=>`<img class=strip src="${SB}${r.name}_${tag}_${k}.png">`;
  t+=`<tr><td>${r.name}<br><span class=mut>wj=${f(r.wj)}</span></td>`+
   `<td>${g('off','gt')}${g('off','open')}${g('off','cycle')}</td>`+
   `<td>${g('on','gt')}${g('on','open')}${g('on','cycle')}</td></tr>`;}
 card(t+'</table>');}
</script></body></html>"""
    VIZ_HTML.parent.mkdir(parents=True, exist_ok=True)
    VIZ_HTML.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


if __name__ == "__main__":
    main()
