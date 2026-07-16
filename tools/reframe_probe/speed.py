#!/usr/bin/env python
"""Coarse-reframe speed & quick-wins exploration (see prompts/coarse_reframe_speed_prompt.md).

One-off diagnostic. Coarse reframing runs on EVERY discovery survivor, so it must be
cheap. This finds the cheapest render fidelity + simplest search that produce
essentially the SAME coarse-reframe crop as a full-fidelity dense search over the
bounded window. It is a SIBLING of reframe_probe/probe.py and reuses its machinery
verbatim (anchors, auto_maxiter, render-one path incl. --julia --c, fixed
classifier-native palette/coloring, score_lib v5 CORN decode).

Metric = REGRET, not argmax-index agreement:
  For a cheap setting, crop* = its argmax framing. Render crop* at the reference
  fidelity (1280x720 ss4) and score it. regret = ref_best_score - ref_score(crop*).
  Because the reference setting already renders+scores all 36 grid framings, and
  every cheap setting's argmax IS one of those 36 framings, ref_score(crop*) is a
  pure LOOKUP into the reference grid -> no extra GPU for regret, and Axis B
  (search simplification) is entirely post-hoc over the reference grid.

Read within-anchor only; never compare absolute scores across fidelities.

Bounded window (strict): recenter dx,dy in [-0.25,+0.25]*fw; fw in [x0.5, x1.41].
Reference grid per anchor: fw in {x0.5, x0.71, x1.0, x1.41} x recenter {-0.25,0,+0.25}^2
= 4 x 9 = 36 framings. Only cx/cy/fw vary within a grid; only fidelity varies across
settings; coloring held fixed to the classifier-native palette.

Anchors (reuse probe's, matched):
  tracking (framing varies): m3_shallow, j0_deep, j1_shallow
  controls (saturated):      m1_mid, m2_mid  (spurious-winner check)

Usage (uv):
  uv run python tools/reframe_probe/speed.py --anchors-only
  uv run python tools/reframe_probe/speed.py --time-only
  uv run python tools/reframe_probe/speed.py            # full run
  uv run python tools/reframe_probe/speed.py --rebuild-sheets
  uv run python tools/reframe_probe/speed.py --analyze
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

# Guard the Windows cp1252 console (probe.py crashed on a stray unicode char); keep
# all our own output ASCII regardless.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scoring"))  # active_ckpt (formerly the reframe_probe sibling `probe`)

# Reuse the active_ckpt machinery verbatim.
from active_ckpt import (  # noqa: E402
    BIN, ROOT, PALETTE, JPG_Q, auto_maxiter, make_scorer, select_anchors,
)

OUT_DIR = ROOT / "out" / "reframe_speed"

# --- bounded-window grid (LOCKED, see prompt) ---
FW_FACS = [0.5, 2.0 ** -0.5, 1.0, 2.0 ** 0.5]   # x0.5 .. x1.41 (2-sided, no zoom-max)
RECENTER = [-0.25, 0.0, 0.25]                    # dx,dy in fractions of the fw step (3x3=9)
NCOL, NROW = len(FW_FACS), len(RECENTER) ** 2    # 4 x 9 = 36

# --- Axis A: render-fidelity ladder ---
# All 16:9 so field-of-view is identical across settings (fw = horizontal extent;
# vertical = fw*h/w). The classifier deploy input is 384x224 (stretch of 1280x720);
# the floor is rendered 384x216 (16:9) which the deploy transform then stretches to
# 384x224 -- the exact mirror of the reference 1280x720->384x224 path, no aspect
# confound. Reference is first.
SETTINGS = [
    {"key": "ref_1280x720_ss4", "w": 1280, "h": 720, "ss": 4},   # REFERENCE
    {"key": "1280x720_ss2",     "w": 1280, "h": 720, "ss": 2},
    {"key": "640x360_ss2",      "w": 640,  "h": 360, "ss": 2},
    {"key": "640x360_ss1",      "w": 640,  "h": 360, "ss": 1},
    {"key": "384x216_ss1",      "w": 384,  "h": 216, "ss": 1},    # floor ~ classifier input res
]
REF_KEY = SETTINGS[0]["key"]

# --- anchors (reuse probe's exact keys, matched) ---
TRACKING = ["m3_shallow", "j0_deep", "j1_shallow"]
CONTROL = ["m1_mid", "m2_mid"]
WANT_KEYS = TRACKING + CONTROL


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def build_framings(anchor: dict) -> list[dict]:
    cx0 = Decimal(anchor["cx"])
    cy0 = Decimal(anchor["cy"])
    fw0 = float(anchor["fw"])
    frames = []
    for ci, fac in enumerate(FW_FACS):
        fw = fw0 * fac
        for ri, dy in enumerate(RECENTER):
            for di, dx in enumerate(RECENTER):
                cx = cx0 + Decimal(repr(dx * fw))
                cy = cy0 + Decimal(repr(dy * fw))
                frames.append({
                    "idx": ci * 9 + ri * 3 + di,
                    "col": ci, "row": ri * 3 + di,
                    "fw_factor": fac, "dx": dx, "dy": dy,
                    "fw": fw, "cx": str(cx), "cy": str(cy),
                    "maxiter": auto_maxiter(fw),
                    "is_anchor": (fac == 1.0 and dx == 0.0 and dy == 0.0),
                })
    return frames


def select_wanted() -> list[dict]:
    all_anchors = {a["anchor_key"]: a for a in select_anchors()}
    missing = [k for k in WANT_KEYS if k not in all_anchors]
    if missing:
        raise SystemExit(f"anchor keys missing from probe selection: {missing}")
    out = []
    for k in WANT_KEYS:
        a = dict(all_anchors[k])
        a["role"] = "tracking" if k in TRACKING else "control"
        out.append(a)
    return out


def print_anchors(anchors: list[dict]):
    print(f"\n=== anchors (n={len(anchors)}, reused from reframe_probe) ===")
    print(f"{'key':<12} {'role':<9} {'family':<11} {'fw':>13}  center (+ julia c)")
    for a in anchors:
        c = "" if a["c_re"] is None else f"  c=({a['c_re']}, {a['c_im']})"
        print(f"{a['anchor_key']:<12} {a['role']:<9} {a['family']:<11} "
              f"{float(a['fw']):>13.6e}  ({a['cx']}, {a['cy']}){c}")


# --------------------------------------------------------------------------- #
# render (parametrized on w/h/ss; otherwise identical to probe.render_one)
# --------------------------------------------------------------------------- #
def tile_path(anchor_dir: Path, setting_key: str, f: dict) -> Path:
    return anchor_dir / "tiles" / setting_key / f"c{f['col']}_r{f['row']}.jpg"


def render_frame(anchor: dict, f: dict, out: Path, w: int, h: int, ss: int) -> tuple[bool, str]:
    import location as loc_mod
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one",
        "--cx", f["cx"], "--cy", f["cy"], "--fw", repr(f["fw"]),
        "--width", str(w), "--height", str(h),
        "--supersample", str(ss), "--maxiter", str(f["maxiter"]),
        "--palette", PALETTE, "--jpg-quality", str(JPG_Q),
        "--out", str(out),
    ]
    cmd += loc_mod.render_one_flags(loc_mod.Location(
        family=anchor["family"], cx=f["cx"], cy=f["cy"], fw=f["fw"],
        c_re=anchor.get("c_re"), c_im=anchor.get("c_im"),
        family_params=anchor.get("family_params") or {}))
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists()
    return ok, ("" if ok else r.stderr[-300:])


def render_setting(anchor: dict, anchor_dir: Path, frames: list[dict], s: dict, workers: int):
    todo = [(f, tile_path(anchor_dir, s["key"], f)) for f in frames]
    fails = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_frame, anchor, f, out, s["w"], s["h"], s["ss"]): (f, out)
                for f, out in todo}
        for fut in cf.as_completed(futs):
            f, out = futs[fut]
            ok, err = fut.result()
            if not ok:
                fails.append((f, err))
    if fails:
        for f, err in fails[:3]:
            sys.stderr.write(f"[render FAIL {anchor['anchor_key']} {s['key']} "
                             f"c{f['col']}r{f['row']}] {err}\n")
        raise SystemExit(f"{len(fails)} render failures in {anchor['anchor_key']}/{s['key']}")


def score_setting(scorer, anchor_dir: Path, setting_key: str, frames: list[dict]) -> list[float]:
    paths = [tile_path(anchor_dir, setting_key, f) for f in frames]
    triples = scorer.score_paths(paths)     # [(score, p_notbad, p_good)]
    return [float(s) for s, _nb, _g in triples]


# --------------------------------------------------------------------------- #
# Axis B: search simplifications (pure post-hoc over the reference-fidelity grid)
# --------------------------------------------------------------------------- #
# recenter row layout (row = ry*3 + rx over RECENTER=[-0.25,0,+0.25]):
#   0:(-,-) 1:(-,0) 2:(-,+) 3:(0,-) 4:(0,0)=CENTER 5:(0,+) 6:(+,-) 7:(+,0) 8:(+,+)
ROW_CENTER = 4
ROWS_CORNERS_CENTER = [0, 2, 4, 6, 8]           # 5 recenters
ROWS_DIAG3 = [0, 4, 8]                          # 3 recenters
COLS_ALL = [0, 1, 2, 3]
COLS_COARSE3 = [0, 2, 3]                        # fw x0.5, x1.0, x1.41


def _argmax_over(ref_scores: list[float], indices: list[int]) -> int:
    return max(indices, key=lambda i: ref_scores[i])


def simplification_picks(ref_scores: list[float]) -> dict:
    """All configs pick an index into the 36-grid; regret computed by caller vs ref_best."""
    def prod(cols, rows):
        return [c * 9 + r for c in cols for r in rows]

    out = {}
    # dense reference search (baseline; regret == 0 by construction)
    out["full_36"] = {"n": 36, "pick": _argmax_over(ref_scores, prod(COLS_ALL, list(range(9))))}
    # coarser grids
    out["coarse_20 (4fw x 5rc)"] = {
        "n": len(COLS_ALL) * len(ROWS_CORNERS_CENTER),
        "pick": _argmax_over(ref_scores, prod(COLS_ALL, ROWS_CORNERS_CENTER)),
    }
    out["grid_9 (3fw x 3rc)"] = {
        "n": len(COLS_COARSE3) * len(ROWS_DIAG3),
        "pick": _argmax_over(ref_scores, prod(COLS_COARSE3, ROWS_DIAG3)),
    }
    # separable: best fw at center, then best recenter at that fw
    fw_cands = [c * 9 + ROW_CENTER for c in COLS_ALL]
    best_col = _argmax_over(ref_scores, fw_cands) // 9
    rc_cands = [best_col * 9 + r for r in range(9)]
    out["separable_13 (fw@ctr,then rc)"] = {
        "n": len(COLS_ALL) + 9,   # 4 fw evals + 9 recenter evals (center shared)
        "pick": _argmax_over(ref_scores, rc_cands),
    }
    # fw-only: center-locked best fw (drop recentering entirely)
    out["fw_only_4 (center-locked)"] = {
        "n": len(COLS_ALL),
        "pick": _argmax_over(ref_scores, [c * 9 + ROW_CENTER for c in COLS_ALL]),
    }
    return out


def colrow(idx: int) -> tuple[int, int]:
    return idx // 9, idx % 9


def frame_tag(frames: list[dict], idx: int) -> str:
    f = next(x for x in frames if x["idx"] == idx)
    return f"fwx{f['fw_factor']:.3g} d{f['dx']:+.2f},{f['dy']:+.2f}"


# --------------------------------------------------------------------------- #
# records.json
# --------------------------------------------------------------------------- #
def compute_record(anchor: dict, frames: list[dict], scores: dict, scorer_cfg: dict) -> dict:
    ref = scores[REF_KEY]
    ref_best_idx = max(range(NCOL * NROW), key=lambda i: ref[i])
    ref_best = ref[ref_best_idx]

    fidelity = {}
    for s in SETTINGS:
        sc = scores[s["key"]]
        am = max(range(NCOL * NROW), key=lambda i: sc[i])
        fidelity[s["key"]] = {
            "argmax_idx": am, "argmax_colrow": list(colrow(am)),
            "argmax_tag": frame_tag(frames, am),
            "regret": round(ref_best - ref[am], 6),
            "own_min": round(min(sc), 6), "own_max": round(max(sc), 6),
            "own_spread": round(max(sc) - min(sc), 6),
            "argmax_agrees_ref": (am == ref_best_idx),
        }

    simp = {}
    for name, d in simplification_picks(ref).items():
        pk = d["pick"]
        simp[name] = {
            "n_framings": d["n"], "pick_idx": pk, "pick_colrow": list(colrow(pk)),
            "pick_tag": frame_tag(frames, pk),
            "regret": round(ref_best - ref[pk], 6),
        }

    a = anchor
    return {
        "anchor_key": a["anchor_key"], "role": a["role"], "family": a["family"],
        "anchor": {"cx": a["cx"], "cy": a["cy"], "fw": a["fw"],
                   "c_re": a["c_re"], "c_im": a["c_im"],
                   "example_image_id": a.get("example_image_id"),
                   "batch_id": a.get("batch_id")},
        "config": {
            "palette": PALETTE, "jpg_quality": JPG_Q,
            "fw_facs": FW_FACS, "recenter": RECENTER,
            "settings": SETTINGS, "ref_key": REF_KEY,
            "classifier_deploy": {
                "geometry": scorer_cfg.get("geometry"),
                "interpolation": scorer_cfg.get("interpolation"),
                "src": "1280x720", "target": "384x224",
                "note": "floor rendered 384x216 (16:9), transform stretches to 384x224",
            },
            "score": "expected_ordinal = sigma(l0)+sigma(l1) in [0,2] (within-anchor only)",
        },
        "framings": [{
            "idx": f["idx"], "col": f["col"], "row": f["row"],
            "fw_factor": f["fw_factor"], "dx": f["dx"], "dy": f["dy"],
            "is_anchor": f["is_anchor"],
            "cx": f["cx"], "cy": f["cy"], "fw": f["fw"], "maxiter": f["maxiter"],
            "family": a["family"], "c_re": a["c_re"], "c_im": a["c_im"],
            "scores": {sk: round(scores[sk][f["idx"]], 6) for sk in scores},
        } for f in frames],
        "ref_best_idx": ref_best_idx, "ref_best_colrow": list(colrow(ref_best_idx)),
        "ref_best_score": round(ref_best, 6),
        "fidelity": fidelity,
        "simplification": simp,
    }


# --------------------------------------------------------------------------- #
# sheet (reuses the reference ss4 tiles: every pick IS a grid framing)
# --------------------------------------------------------------------------- #
TW, TH = 300, 169
PAD = 8
LBL_H = 66
PER_ROW = 6
GUT_T = 34
SEC_H = 22


def _pick_cells(rec: dict) -> list[dict]:
    """Ordered cells: ref-best (gold), then each non-ref fidelity pick, then each
    simplification pick. Every cell references a reference-grid tile."""
    cells = []
    rb = rec["ref_best_idx"]
    cells.append({"section": "REF", "name": "REF-BEST", "idx": rb,
                  "regret": 0.0, "tag": frame_tag(rec["framings"], rb),
                  "role": rec["role"], "gold": True})
    for s in SETTINGS:
        if s["key"] == REF_KEY:
            continue
        fd = rec["fidelity"][s["key"]]
        cells.append({"section": "FIDELITY", "name": s["key"], "idx": fd["argmax_idx"],
                      "regret": fd["regret"], "tag": fd["argmax_tag"],
                      "spread": fd["own_spread"], "agrees": fd["argmax_agrees_ref"]})
    for name, d in rec["simplification"].items():
        if name == "full_36":
            continue
        cells.append({"section": "SIMPLIFY", "name": name, "idx": d["pick_idx"],
                      "regret": d["regret"], "tag": d["pick_tag"], "n": d["n_framings"]})
    return cells


def build_sheet(anchor_dir: Path, rec: dict) -> Path:
    from PIL import Image, ImageDraw
    cells = _pick_cells(rec)
    n = len(cells)
    nrow = (n + PER_ROW - 1) // PER_ROW
    cell_w = TW + 2 * PAD
    cell_h = TH + LBL_H + 2 * PAD
    W = PER_ROW * cell_w
    H = GUT_T + nrow * (cell_h + SEC_H)

    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    a = rec["anchor"]
    ctxt = "" if a["c_re"] is None else f"  c=({a['c_re']}, {a['c_im']})"
    draw.text((8, 8), f"{rec['anchor_key']}  [{rec['family']} / {rec['role']}]  "
              f"anchor fw={float(a['fw']):.4e}{ctxt}   ref_best E={rec['ref_best_score']:.3f} @ "
              f"{frame_tag(rec['framings'], rec['ref_best_idx'])}   "
              f"(all crops shown at ss4-ref; regret = ref_best - ref(pick))",
              fill=(235, 235, 235))

    ref_tiles = anchor_dir / "tiles" / REF_KEY
    for i, cell in enumerate(cells):
        r, c = divmod(i, PER_ROW)
        x = c * cell_w + PAD
        y = GUT_T + r * (cell_h + SEC_H) + SEC_H + PAD
        col, row = colrow(cell["idx"])
        tp = ref_tiles / f"c{col}_r{row}.jpg"
        if tp.exists():
            im = Image.open(tp).convert("RGB").resize((TW, TH))
            sheet.paste(im, (x, y))
        # label block
        draw.rectangle([x, y + TH, x + TW, y + TH + LBL_H], fill=(28, 28, 32))
        reg = cell["regret"]
        reg_col = (90, 230, 110) if reg <= 0.02 else ((235, 200, 60) if reg <= 0.08 else (235, 90, 90))
        draw.text((x + 4, y + TH + 3), cell["name"][:34], fill=(225, 225, 232))
        draw.text((x + 4, y + TH + 20), f"regret {reg:+.3f}", fill=reg_col)
        line3 = cell["tag"]
        if "spread" in cell:
            line3 += f"   spr {cell['spread']:.3f}"
        if "n" in cell:
            line3 += f"   n={cell['n']}"
        draw.text((x + 4, y + TH + 37), line3[:40], fill=(190, 190, 200))
        if not cell.get("gold") and "agrees" in cell and not cell["agrees"]:
            draw.text((x + 4, y + TH + 52), "argmax != ref", fill=(235, 150, 60))
        # border: gold for ref-best, else regret-colored
        border = (245, 215, 40) if cell.get("gold") else reg_col
        for t in range(3):
            draw.rectangle([x - 1 - t, y - 1 - t, x + TW + t, y + TH + LBL_H + t], outline=border)

    out = anchor_dir / "sheet.png"
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- #
# summary + console tables
# --------------------------------------------------------------------------- #
def write_summary(recs: list[dict]) -> dict:
    keys = [r["anchor_key"] for r in recs]
    track = [r for r in recs if r["role"] == "tracking"]
    ctrl = [r for r in recs if r["role"] == "control"]

    def mean(xs):
        return round(sum(xs) / len(xs), 6) if xs else None

    fid = {}
    for s in SETTINGS:
        row = {r["anchor_key"]: r["fidelity"][s["key"]]["regret"] for r in recs}
        row["_mean_tracking"] = mean([r["fidelity"][s["key"]]["regret"] for r in track])
        row["_mean_control"] = mean([r["fidelity"][s["key"]]["regret"] for r in ctrl])
        row["_max_control"] = round(max((r["fidelity"][s["key"]]["regret"] for r in ctrl), default=0.0), 6)
        fid[s["key"]] = row

    simp_names = list(recs[0]["simplification"].keys())
    simp = {}
    for name in simp_names:
        row = {r["anchor_key"]: r["simplification"][name]["regret"] for r in recs}
        row["_n"] = recs[0]["simplification"][name]["n_framings"]
        row["_mean_tracking"] = mean([r["simplification"][name]["regret"] for r in track])
        row["_mean_control"] = mean([r["simplification"][name]["regret"] for r in ctrl])
        simp[name] = row

    return {"anchors": keys, "tracking": TRACKING, "control": CONTROL,
            "fidelity_regret": fid, "simplification_regret": simp,
            "settings": SETTINGS}


def print_tables(summary: dict, recs: list[dict]):
    keys = summary["anchors"]
    print("\n=== AXIS A: fidelity x regret (regret = ref_best - ref(cheap-argmax); within-anchor) ===")
    hdr = f"{'setting':<20}" + "".join(f"{k:>12}" for k in keys) + f"{'trk_mean':>10}{'ctl_max':>9}"
    print(hdr)
    for s in SETTINGS:
        row = summary["fidelity_regret"][s["key"]]
        line = f"{s['key']:<20}" + "".join(f"{row[k]:>12.3f}" for k in keys)
        line += f"{row['_mean_tracking']:>10.3f}{row['_max_control']:>9.3f}"
        print(line)
    print("  (ref row is 0 by construction. ctl_max = worst control regret = spurious-winner risk.)")

    # spurious-winner detail on controls
    print("\n  controls -- own score spread & argmax agreement per setting:")
    for r in recs:
        if r["role"] != "control":
            continue
        print(f"   {r['anchor_key']} (ref_best E={r['ref_best_score']:.3f}):")
        for s in SETTINGS:
            fd = r["fidelity"][s["key"]]
            flag = "" if fd["argmax_agrees_ref"] else "  <-- DIFF PICK"
            print(f"     {s['key']:<20} spread={fd['own_spread']:.3f} "
                  f"regret={fd['regret']:+.3f} argmax={fd['argmax_tag']}{flag}")

    print("\n=== AXIS B: search simplification x regret (all at reference fidelity) ===")
    names = list(summary["simplification_regret"].keys())
    hdr = f"{'config':<32}{'n':>4}" + "".join(f"{k:>12}" for k in keys) + f"{'trk_mean':>10}"
    print(hdr)
    for name in names:
        row = summary["simplification_regret"][name]
        line = f"{name:<32}{row['_n']:>4}" + "".join(f"{row[k]:>12.3f}" for k in keys)
        line += f"{row['_mean_tracking']:>10.3f}"
        print(line)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_full(args):
    anchors = select_wanted()
    print_anchors(anchors)
    scorer = make_scorer(args.model)
    print(f"\nclassifier deploy input: geometry={scorer.cfg.get('geometry')} "
          f"interp={scorer.cfg.get('interpolation')}  1280x720 -> 384x224")
    print(f"fidelity floor: 384x216 ss1 (16:9; transform stretches -> 384x224, no aspect confound)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    t0 = time.time()
    for a in anchors:
        ta = time.time()
        frames = build_framings(a)
        anchor_dir = OUT_DIR / a["anchor_key"]
        scores = {}
        for s in SETTINGS:
            render_setting(a, anchor_dir, frames, s, args.workers)
            scores[s["key"]] = score_setting(scorer, anchor_dir, s["key"], frames)
        rec = compute_record(a, frames, scores, scorer.cfg)
        (anchor_dir / "records.json").write_text(json.dumps(rec, indent=2))
        sheet = build_sheet(anchor_dir, rec)
        recs.append(rec)
        fid = rec["fidelity"]
        print(f"[{a['anchor_key']:<12}] ref_best E={rec['ref_best_score']:.3f}  "
              f"regret: " + " ".join(f"{s['key'].split('_')[-1] if s['key']!=REF_KEY else 'ref'}="
                                     f"{fid[s['key']]['regret']:+.3f}" for s in SETTINGS)
              + f"  ({time.time()-ta:.0f}s) -> {sheet.name}")
    summary = write_summary(recs)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print_tables(summary, recs)
    dt = time.time() - t0
    (OUT_DIR / "COMPLETE").write_text(f"done {len(anchors)} anchors in {dt:.0f}s\n")
    print(f"\nDONE {len(anchors)} anchors in {dt:.0f}s -> {OUT_DIR}")


def run_time_only(args):
    anchors = select_wanted()
    print_anchors(anchors)
    deep = min(anchors, key=lambda a: float(a["fw"]))
    shal = max(anchors, key=lambda a: float(a["fw"]))
    tmp = OUT_DIR / "_timing"
    print("\n=== timing 1 framing (center) x all settings, on deepest & shallowest anchor ===")
    per_setting_avg = {}
    for s in SETTINGS:
        ts = []
        for a in (deep, shal):
            frames = build_framings(a)
            f = next(x for x in frames if x["idx"] == 0 * 9 + 4)  # x0.5 fw, center
            out = tmp / f"{a['anchor_key']}_{s['key']}.jpg"
            t = time.time()
            ok, err = render_frame(a, f, out, s["w"], s["h"], s["ss"])
            el = time.time() - t
            if not ok:
                raise SystemExit(f"timing render failed {a['anchor_key']} {s['key']}: {err}")
            ts.append(el)
        per_setting_avg[s["key"]] = sum(ts) / len(ts)
        print(f"  {s['key']:<20} {s['w']}x{s['h']} ss{s['ss']}: "
              f"deep {ts[0]:.2f}s / shallow {ts[1]:.2f}s  avg {per_setting_avg[s['key']]:.2f}s")
    scorer = make_scorer(args.model)
    tsc = time.time()
    scorer.score_paths([tmp / f"{deep['anchor_key']}_{REF_KEY}.jpg",
                        tmp / f"{shal['anchor_key']}_{REF_KEY}.jpg"])
    sc_each = (time.time() - tsc) / 2
    n_anchors = len(anchors)
    # wall-clock: renders parallelize over workers; score is trivial.
    render_wall = sum(per_setting_avg.values()) * NCOL * NROW * n_anchors / max(1, args.workers)
    score_wall = sc_each * len(SETTINGS) * NCOL * NROW * n_anchors
    total = render_wall + score_wall
    print(f"\n  per-frame score {sc_each:.3f}s")
    print(f"  grid: {NCOL}x{NROW}={NCOL*NROW} framings x {len(SETTINGS)} settings x "
          f"{n_anchors} anchors = {NCOL*NROW*len(SETTINGS)*n_anchors} renders")
    print(f"  PROJECTED wall: render ~{render_wall:.0f}s (workers={args.workers}) + "
          f"score ~{score_wall:.0f}s = ~{total:.0f}s (~{total/60:.1f} min)")
    print(f"  -> {'BACKGROUND recommended' if total > 120 else 'foreground OK'}")


def _load_recs() -> list[dict]:
    if not OUT_DIR.exists():
        raise SystemExit(f"no {OUT_DIR}")
    dirs = [OUT_DIR / k for k in WANT_KEYS if (OUT_DIR / k / "records.json").exists()]
    if not dirs:
        raise SystemExit(f"no records.json under {OUT_DIR}")
    return [json.loads((d / "records.json").read_text()) for d in dirs]


def run_rebuild(args):
    recs = _load_recs()
    for rec in recs:
        d = OUT_DIR / rec["anchor_key"]
        sheet = build_sheet(d, rec)
        print(f"  rebuilt {sheet}")
    summary = write_summary(recs)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print_tables(summary, recs)


def run_analyze(args):
    recs = _load_recs()
    summary = write_summary(recs)
    print_tables(summary, recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors-only", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--rebuild-sheets", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="data/classifier/v5/model_best.pt")
    args = ap.parse_args()

    if args.anchors_only:
        print_anchors(select_wanted())
    elif args.time_only:
        run_time_only(args)
    elif args.rebuild_sheets:
        run_rebuild(args)
    elif args.analyze:
        run_analyze(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
