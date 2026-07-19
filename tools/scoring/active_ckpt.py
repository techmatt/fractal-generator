#!/usr/bin/env python
"""Reframing probe — does the location-quality classifier track framing crispness?

One-off diagnostic (see prompts/reframe_probe_prompt.md). Holds a score-3 anchor
location fixed and sweeps ONLY the frame (center cx/cy + frame width fw), rendering
each framing through the classifier's native render path and scoring it with the
location-quality CORN head. Output: per-anchor thumbnail sheets + records.json.

Interfaces reused verbatim (NOT the preference scorer / query_batch_gen path):
  * render : `render-one` (Rust), the canonical "rebuild a location the way the
             classifier expects" helper — Mandelbrot f64 path, or `--julia --c`
             for Julia. Palette held fixed to `twilight_shifted` (the v4/v5
             deploy-canonical palette); only cx/cy/fw vary. jpg q90 = corpus crop.
  * maxiter: `auto_maxiter(fw)` — the native fw-dependent iteration policy
             (mirror of tools/explorer/app.py; fw_home=3.0, base 500, k 0.30,
             clamp [200,8000]). Follows fw, NOT the anchor, so a deep frame is not
             penalised for under-iteration.
  * score  : classifier.model CORN decode via tools/mining/score_lib.Scorer
             (data.Transform(train=False) deploy mirror). The recorded signal is
             the CONTINUOUS expected-ordinal score = sigma(l0)+sigma(l1) in [0,2],
             not the argmax tier — we want within-tier-3 gradation.
  * anchors: tools/corpus/corpus_reader.iter_labeled() — version-blind labeled
             iterator; family/cx/cy/fw/c read off the version-invariant `render`.

Usage (uv):
  uv run python tools/reframe_probe/probe.py --anchors-only
  uv run python tools/reframe_probe/probe.py --time-only
  uv run python tools/reframe_probe/probe.py            # full run
  uv run python tools/reframe_probe/probe.py --rebuild-sheets
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

BIN = ROOT / "target" / "release" / "fractal-generator.exe"
OUT_DIR = ROOT / "out" / "reframe_probe"
PALETTE = "twilight_shifted"           # v4/v5 deploy-canonical palette
# --- Active discovery/guard/reframe classifier checkpoint (SINGLE SOURCE OF TRUTH) ---
# Every discovery-path scorer (production_seeder, guard, reframe) resolves the live
# checkpoint from here. Flip ACTIVE_CKPT and the whole gate moves; nothing else hardcodes
# a version. The load path is version-agnostic (score_lib.Scorer reads mean/std/head from
# the checkpoint's own config), so only this string changes between versions.
ACTIVE_CKPT = "data/classifier/v7/model_best.pt"    # v7 unified location classifier (LIVE)
# Rollback: point ACTIVE_CKPT back at v6 (the one-flip rollback anchor, the role v5 held)
# to restore the prior gate; v5 remains the deeper rollback.
V6_CKPT_ROLLBACK = "data/classifier/v6/model_best.pt"
V5_CKPT_ROLLBACK = "data/classifier/v5/model_best.pt"
DEFAULT_MODEL = ACTIVE_CKPT             # unified location-quality model (== ACTIVE_CKPT)
# Version token of the live checkpoint ("v6"/"v7"...), parsed off the checkpoint dir. This
# is the SINGLE SOURCE OF TRUTH for what "current" means: corpus_common.is_current_decoded
# and production_seeder.SCORER_VERSION both resolve the decode-stamp version from here, so
# flipping ACTIVE_CKPT moves the whole notion of "current-decoded" with it.
ACTIVE_VERSION = Path(ACTIVE_CKPT).parent.name   # "v7"
JPG_Q = 90                              # match corpus crop quality
DEFAULT_SS = 4                          # ss4 = v4/v5 deploy-canonical antialiased view

# --- sweep grid (LOCKED, see prompt §3) ---
FW_LOG2_STEPS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # 0.5x .. 2x, anchor centered (5 cols)
RECENTER = [-0.25, 0.0, 0.25]                 # dx,dy in fractions of the fw step (3x3=9 rows)

# --- auto_maxiter: native fw-dependent policy (mirror tools/explorer/app.py) ---
FW_HOME = 3.0
MAXITER_BASE, MAXITER_K, MAXITER_MIN, MAXITER_MAX = 500, 0.30, 200, 8000


def auto_maxiter(fw: float) -> int:
    ratio = FW_HOME / fw if fw > 0 else 1.0
    lz = math.log2(ratio) if ratio > 0 else 0.0
    val = MAXITER_BASE * (1.0 + MAXITER_K * lz)
    return int(max(MAXITER_MIN, min(MAXITER_MAX, val)))


# --------------------------------------------------------------------------- #
# Step 2 — anchor selection (deterministic).
# --------------------------------------------------------------------------- #
def _unique_score3_locations():
    """Dedup score==3 crops into unique locations (family + geometry + family_params).

    Family-general: the dedup key and the returned row carry the per-family extra
    constants (Phoenix's `p_re/p_im`) via `location.location_key` / the params slot,
    so two Phoenix locations differing only in `p` are distinct and `p` survives into
    the reframe Location."""
    import corpus_reader as cr
    import location as loc_mod
    seen: dict = {}
    for lc in cr.iter_labeled():
        if lc.score != 3:
            continue
        r = lc.render
        if r.get("cx") is None or r.get("cy") is None or r.get("fw") is None:
            continue
        canon = loc_mod.from_render_block(r)
        key = canon.key()
        if key in seen:
            continue
        seen[key] = {
            "family": canon.family, "cx": str(canon.cx), "cy": str(canon.cy),
            "fw": str(canon.fw),
            "c_re": None if canon.c_re is None else str(canon.c_re),
            "c_im": None if canon.c_im is None else str(canon.c_im),
            "family_params": canon.params,
            "example_image_id": lc.image_id, "batch_id": lc.batch_id,
        }
    return list(seen.values())


def select_anchors() -> list[dict]:
    """6 quality-3 anchors: 4 mandelbrot spanning zoom (1 deep, 2 mid, 1 shallow)
    + 2 julia spanning zoom. Deterministic via stable sort on (fw, cx, cy)."""
    locs = _unique_score3_locations()

    def stable(rows):
        return sorted(rows, key=lambda r: (float(r["fw"]), r["cx"], r["cy"]))

    mands = stable([r for r in locs if r["family"] == "mandelbrot"])
    julias = stable([r for r in locs if r["family"] == "julia"])
    if len(mands) < 4:
        raise SystemExit(f"need >=4 score-3 mandelbrot, have {len(mands)}")
    if len(julias) < 2:
        raise SystemExit(f"need >=2 score-3 julia, have {len(julias)}")

    n = len(mands)
    m_pick = [
        (mands[0], "deep"),
        (mands[n // 3], "mid"),
        (mands[(2 * n) // 3], "mid"),
        (mands[-1], "shallow"),
    ]
    nj = len(julias)
    j_pick = [(julias[nj // 4], "deep"), (julias[(3 * nj) // 4], "shallow")]

    anchors = []
    for i, (loc, tag) in enumerate(m_pick):
        anchors.append({**loc, "anchor_key": f"m{i}_{tag}", "zoom_tag": tag})
    for i, (loc, tag) in enumerate(j_pick):
        anchors.append({**loc, "anchor_key": f"j{i}_{tag}", "zoom_tag": tag})
    return anchors


def print_anchors(anchors: list[dict]):
    print(f"\n=== selected anchors (n={len(anchors)}) ===")
    print(f"{'key':<12} {'family':<11} {'zoom':<8} {'fw':>13}  center (+ julia c)")
    for a in anchors:
        c = "" if a["c_re"] is None else f"  c=({a['c_re']}, {a['c_im']})"
        print(f"{a['anchor_key']:<12} {a['family']:<11} {a['zoom_tag']:<8} "
              f"{float(a['fw']):>13.6e}  ({a['cx']}, {a['cy']}){c}")


# --------------------------------------------------------------------------- #
# Step 3 — sweep grid per anchor (45 framings).
# --------------------------------------------------------------------------- #
def build_framings(anchor: dict) -> list[dict]:
    cx0 = Decimal(anchor["cx"])
    cy0 = Decimal(anchor["cy"])
    fw0 = float(anchor["fw"])
    frames = []
    for ci, l2 in enumerate(FW_LOG2_STEPS):
        fw = fw0 * (2.0 ** l2)
        for ri, dy in enumerate(RECENTER):
            for di, dx in enumerate(RECENTER):
                cx = cx0 + Decimal(repr(dx * fw))
                cy = cy0 + Decimal(repr(dy * fw))
                frames.append({
                    "col": ci, "fw_factor": 2.0 ** l2,
                    "row": ri * 3 + di, "dx": dx, "dy": dy,
                    "fw": fw, "cx": str(cx), "cy": str(cy),
                    "maxiter": auto_maxiter(fw),
                    "is_anchor": (l2 == 0.0 and dx == 0.0 and dy == 0.0),
                })
    return frames


# --------------------------------------------------------------------------- #
# Step 4 — render (render-one) + score (v5 CORN).
# --------------------------------------------------------------------------- #
def tile_path(anchor_dir: Path, f: dict) -> Path:
    return anchor_dir / "tiles" / f"c{f['col']}_r{f['row']}.jpg"


def render_one(anchor: dict, f: dict, out: Path, ss: int) -> tuple[bool, str]:
    import location as loc_mod
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one",
        "--cx", f["cx"], "--cy", f["cy"], "--fw", repr(f["fw"]),
        "--width", "1280", "--height", "720",
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


def render_anchor(anchor: dict, frames: list[dict], ss: int, workers: int):
    anchor_dir = OUT_DIR / anchor["anchor_key"]
    todo = [(f, tile_path(anchor_dir, f)) for f in frames]
    fails = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_one, anchor, f, out, ss): (f, out) for f, out in todo}
        for fut in cf.as_completed(futs):
            f, out = futs[fut]
            ok, err = fut.result()
            if not ok:
                fails.append((f, err))
    if fails:
        for f, err in fails[:3]:
            sys.stderr.write(f"[render FAIL {anchor['anchor_key']} c{f['col']}r{f['row']}] {err}\n")
        raise SystemExit(f"{len(fails)} render failures in {anchor['anchor_key']}")
    return anchor_dir


def make_scorer(model_path: str):
    from score_lib import Scorer
    return Scorer(model_path=model_path)


def score_frames(scorer, anchor_dir: Path, frames: list[dict]):
    paths = [tile_path(anchor_dir, f) for f in frames]
    triples = scorer.score_paths(paths)   # [(score, p_notbad, p_good)]
    for f, (s, nb, g) in zip(frames, triples):
        f["score"] = float(s)
        f["p_notbad"] = float(nb)
        f["p_good"] = float(g)


# --------------------------------------------------------------------------- #
# Step 5 — records.json + sheet.
# --------------------------------------------------------------------------- #
def write_records(anchor: dict, frames: list[dict], anchor_dir: Path, ss: int, model_path: str):
    anchor_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "anchor_key": anchor["anchor_key"],
        "family": anchor["family"],
        "zoom_tag": anchor["zoom_tag"],
        "anchor": {
            "cx": anchor["cx"], "cy": anchor["cy"], "fw": anchor["fw"],
            "c_re": anchor["c_re"], "c_im": anchor["c_im"],
            "example_image_id": anchor["example_image_id"], "batch_id": anchor["batch_id"],
        },
        "config": {
            "palette": PALETTE, "supersample": ss, "width": 1280, "height": 720,
            "jpg_quality": JPG_Q, "model": model_path, "score": "expected_ordinal = p_notbad+p_good in [0,2]",
            "fw_log2_steps": FW_LOG2_STEPS, "recenter": RECENTER,
            "maxiter_policy": {"fw_home": FW_HOME, "base": MAXITER_BASE, "k": MAXITER_K,
                               "min": MAXITER_MIN, "max": MAXITER_MAX},
        },
        "framings": [{
            "col": f["col"], "row": f["row"], "fw_factor": f["fw_factor"],
            "dx": f["dx"], "dy": f["dy"], "is_anchor": f["is_anchor"],
            "family": anchor["family"], "c_re": anchor["c_re"], "c_im": anchor["c_im"],
            "cx": f["cx"], "cy": f["cy"], "fw": f["fw"], "maxiter": f["maxiter"],
            "score": f.get("score"), "p_notbad": f.get("p_notbad"), "p_good": f.get("p_good"),
            "tile": str(tile_path(anchor_dir, f).relative_to(anchor_dir)),
        } for f in frames],
    }
    (anchor_dir / "records.json").write_text(json.dumps(rec, indent=2))
    return rec


# thumbnail sheet geometry
TW, TH = 224, 126           # thumbnail (16:9)
GUT_L, GUT_T = 132, 56      # left row-label gutter, top band (title + col-header)
PAD, LBL_H = 6, 20          # cell padding, per-cell score bar height


def build_sheet(anchor_dir: Path, rec: dict):
    from PIL import Image, ImageDraw
    frames = rec["framings"]
    by_cr = {(f["col"], f["row"]): f for f in frames}
    ncol = len(FW_LOG2_STEPS)
    nrow = len(RECENTER) ** 2
    cell_w, cell_h = TW + 2 * PAD, TH + LBL_H + 2 * PAD
    W = GUT_L + ncol * cell_w
    H = GUT_T + nrow * cell_h + 40

    scores = [f["score"] for f in frames if f.get("score") is not None]
    argmax_key = None
    if scores:
        best = max(frames, key=lambda f: (f.get("score") is not None, f.get("score", -1)))
        argmax_key = (best["col"], best["row"])

    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)

    a = rec["anchor"]
    ctxt = "" if a["c_re"] is None else f"  c=({a['c_re']}, {a['c_im']})"
    draw.text((8, 8), f"{rec['anchor_key']}  [{rec['family']} / {rec['zoom_tag']}]  "
              f"anchor fw={float(a['fw']):.4e}{ctxt}   "
              f"(green=anchor  yellow=argmax  score=E[ord] in [0,2])", fill=(235, 235, 235))

    # column headers (fw factor) — sit in the top band, below the title line
    for ci, l2 in enumerate(FW_LOG2_STEPS):
        x = GUT_L + ci * cell_w
        fac = 2.0 ** l2
        tag = "tight" if fac < 1 else ("LOOSE" if fac > 1 else "anchor")
        draw.text((x + 6, GUT_T - 20), f"fw x{fac:.3g} ({tag})", fill=(200, 200, 210))
    draw.text((8, GUT_T - 20), "recenter", fill=(200, 200, 210))

    for ci in range(ncol):
        for ri in range(nrow):
            f = by_cr.get((ci, ri))
            x = GUT_L + ci * cell_w + PAD
            y = GUT_T + ri * cell_h + PAD
            if ci == 0:  # one clean row label per recenter
                dy, dx = RECENTER[ri // 3], RECENTER[ri % 3]
                draw.text((8, y + TH // 2 - 12),
                          f"r{ri}\ndx{dx:+.2f}\ndy{dy:+.2f}", fill=(175, 175, 185))
            if f is None:
                continue
            tp = anchor_dir / f["tile"]
            if tp.exists():
                im = Image.open(tp).convert("RGB").resize((TW, TH))
                sheet.paste(im, (x, y))
            s = f.get("score")
            stxt = "--" if s is None else f"{s:.3f}"
            nb = f.get("p_notbad")
            nbtxt = "" if nb is None else f"  nb {nb:.2f}"
            draw.rectangle([x, y + TH, x + TW, y + TH + LBL_H], fill=(30, 30, 34))
            draw.text((x + 4, y + TH + 3), f"E {stxt}{nbtxt}", fill=(230, 230, 235))
            # highlight anchor (green) + argmax (yellow)
            border = None
            if f["is_anchor"]:
                border = (60, 220, 90)
            if argmax_key == (ci, ri):
                border = (245, 215, 40)
            if f["is_anchor"] and argmax_key == (ci, ri):
                border = (60, 220, 90)  # anchor wins the tie visually; note below
            if border:
                for t in range(3):
                    draw.rectangle([x - 1 - t, y - 1 - t, x + TW + t, y + TH + LBL_H + t], outline=border)

    out = anchor_dir / "sheet.png"
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_full(args):
    anchors = select_anchors()
    print_anchors(anchors)
    if args.limit_anchors:
        anchors = anchors[:args.limit_anchors]
        print(f"\n(limiting to first {len(anchors)} anchors)")

    scorer = make_scorer(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    t0 = time.time()
    for a in anchors:
        ta = time.time()
        frames = build_framings(a)
        anchor_dir = render_anchor(a, frames, args.ss, args.workers)
        score_frames(scorer, anchor_dir, frames)
        rec = write_records(a, frames, anchor_dir, args.ss, args.model)
        sheet = build_sheet(anchor_dir, rec)
        anc = next(f for f in frames if f["is_anchor"])
        best = max(frames, key=lambda f: f["score"])
        print(f"[{a['anchor_key']}] anchor E={anc['score']:.3f}  "
              f"argmax E={best['score']:.3f} @ c{best['col']}r{best['row']} "
              f"(fwx{best['fw_factor']:.3g} dx{best['dx']:+.2f} dy{best['dy']:+.2f})  "
              f"range[{min(f['score'] for f in frames):.3f},{max(f['score'] for f in frames):.3f}]  "
              f"({time.time()-ta:.0f}s)  -> {sheet.name}")
        summary.append(rec)
    print(f"\nDONE {len(anchors)} anchors in {time.time()-t0:.0f}s -> {OUT_DIR}")
    (OUT_DIR / "COMPLETE").write_text(f"done {len(anchors)} anchors in {time.time()-t0:.0f}s\n")
    _print_analysis(summary)


MID_FACS = (2 ** -0.5, 1.0, 2 ** 0.5)   # sensible middle fw band (exclude 0.5x/2x extremes)


def _analyze_frames(frames):
    band = [f for f in frames if f["fw_factor"] in MID_FACS]  # middle fw AND all recenters
    bmin = min(band, key=lambda f: f["score"])
    bmax = max(band, key=lambda f: f["score"])
    anc = next(f for f in frames if f["is_anchor"])
    best = max(frames, key=lambda f: f["score"])
    best_is_extreme = (best["fw_factor"] in (0.5, 2.0)
                       or abs(best["dx"]) == 0.25 or abs(best["dy"]) == 0.25)
    return {
        "anchorE": anc["score"], "bandmin": bmin["score"], "bandmax": bmax["score"],
        "d_band": bmax["score"] - bmin["score"],
        "d_full": max(f["score"] for f in frames) - min(f["score"] for f in frames),
        "best": best, "best_is_extreme": best_is_extreme,
    }


def _print_analysis(recs):
    print("\n=== quick characterization (within-anchor rankings) ===")
    for rec in recs:
        a = _analyze_frames(rec["framings"])
        b = a["best"]
        print(f" {rec['anchor_key']:<12} [{rec['family'][:4]}/{rec['zoom_tag']:<7}] "
              f"anchorE={a['anchorE']:.3f}  "
              f"band[{a['bandmin']:.3f}..{a['bandmax']:.3f}] d={a['d_band']:.3f}  "
              f"full_d={a['d_full']:.3f}  argmax@{'EXTREME' if a['best_is_extreme'] else 'middle '} "
              f"(fwx{b['fw_factor']:.3g} d{b['dx']:+.2f},{b['dy']:+.2f} E={b['score']:.3f})")
    print("\n(agnostic: full_d small ~<0.05 | sensitive-but-wrong: full_d large, argmax@EXTREME |"
          " tracking: argmax@middle & anchor near band-max — eyeball sheets to confirm framing quality)")


def run_time_only(args):
    anchors = select_anchors()
    print_anchors(anchors)
    # deepest + shallowest anchor, each at its tightest (0.5x) and anchor (1x) frame
    deep = min(anchors, key=lambda a: float(a["fw"]))
    shal = max(anchors, key=lambda a: float(a["fw"]))
    probe = []
    for a in (deep, shal):
        frames = build_framings(a)
        tight = next(f for f in frames if f["col"] == 0 and f["row"] == 4)   # 0.5x, center
        probe.append((a, tight))
    print("\n=== timing 2 framings (deepest.tight, shallowest.tight) ===")
    tmp = OUT_DIR / "_timing"
    per = []
    for a, f in probe:
        out = tmp / f"{a['anchor_key']}.jpg"
        t = time.time()
        ok, err = render_one(a, f, out, args.ss)
        el = time.time() - t
        if not ok:
            raise SystemExit(f"timing render failed {a['anchor_key']}: {err}")
        per.append(el)
        print(f"  {a['anchor_key']:<12} fw={f['fw']:.3e} maxiter={f['maxiter']:>5}  "
              f"render {el:.2f}s (ss{args.ss})")
    scorer = make_scorer(args.model)
    ts = time.time()
    scorer.score_paths([tmp / f"{a['anchor_key']}.jpg" for a, _ in probe])
    sc_each = (time.time() - ts) / len(probe)
    avg = sum(per) / len(per)
    total = 6 * 45 * (avg + sc_each)
    print(f"\n  avg render {avg:.2f}s + score {sc_each:.3f}s/frame")
    print(f"  PROJECTED 6x45=270 frames: {total:.0f}s (~{total/60:.1f} min) at ss{args.ss}, "
          f"workers={args.workers} would divide render wall-clock")
    print(f"  -> {'BACKGROUND recommended' if total > 120 else 'foreground OK'}")


def run_analyze(args):
    dirs = sorted(d for d in OUT_DIR.iterdir() if (d / "records.json").exists()) if OUT_DIR.exists() else []
    if not dirs:
        raise SystemExit(f"no records.json under {OUT_DIR}")
    recs = [json.loads((d / "records.json").read_text()) for d in dirs]
    _print_analysis(recs)


def run_rebuild(args):
    anchors_dirs = sorted(d for d in OUT_DIR.iterdir() if (d / "records.json").exists()) if OUT_DIR.exists() else []
    if not anchors_dirs:
        raise SystemExit(f"no records.json under {OUT_DIR} to rebuild from")
    for d in anchors_dirs:
        rec = json.loads((d / "records.json").read_text())
        # re-render any missing tiles (CPU only, no scoring) so the sheet is complete
        anchor = {"family": rec["family"], "c_re": rec["anchor"]["c_re"], "c_im": rec["anchor"]["c_im"],
                  "anchor_key": rec["anchor_key"]}
        ss = rec["config"]["supersample"]
        missing = [f for f in rec["framings"] if not (d / f["tile"]).exists()]
        if missing:
            print(f"  {rec['anchor_key']}: re-rendering {len(missing)} missing tiles (no scoring)")
            for f in missing:
                render_one(anchor, f, d / f["tile"], ss)
        sheet = build_sheet(d, rec)
        print(f"  rebuilt {sheet}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors-only", action="store_true", help="print selected anchors and exit")
    ap.add_argument("--time-only", action="store_true", help="time 2 framings, project total, exit")
    ap.add_argument("--rebuild-sheets", action="store_true", help="rebuild sheets from records.json (no scoring)")
    ap.add_argument("--analyze", action="store_true", help="print characterization from records.json (no render/score)")
    ap.add_argument("--ss", type=int, default=DEFAULT_SS, help="supersample (default 4 = deploy-canonical)")
    ap.add_argument("--workers", type=int, default=6, help="parallel render-one workers")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="classifier checkpoint")
    ap.add_argument("--limit-anchors", type=int, default=0, help="cap #anchors (debug)")
    args = ap.parse_args()

    if args.anchors_only:
        print_anchors(select_anchors())
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
