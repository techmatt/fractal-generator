#!/usr/bin/env python
"""Coarse reframing — the reframing step of discovery (promoted module).

`reframe_location(loc) -> ReframeResult`: given a quality-passed location, return
its best coarse reframe within a bounded window, PLUS that reframe's classifier
score. This sits BETWEEN the quality filter and coloring in the discovery pipeline
and is designed to be MAPPED OVER discovery survivors (build the `Scorer` once, pass
it in). It is the promotion of `tools/reframe_probe/speed.py`'s validated separable
logic out of the diagnostic into permanent code (cf. the query-batch library rename)
-- hence a proper module name, no probe/speed/diagnostic in it.

FAMILY-AGNOSTIC: `family`/`c` are passed straight to `render-one` (`--julia --c`),
so this applies unchanged to Julia today and to the new orbit fractals later.

GEOMETRY ONLY: outputs the crop coords (cx/cy/fw) + score. It does NOT render ss4
and does NOT pick a palette -- the final ss4 wallpaper render happens downstream at
the chosen crop with the chosen palette. Reframing stays coloring-agnostic.

The validated procedure (pinned config, FIRST-PASS-THAT-WORKED on 5 anchors --
provisional, not swept; see prompts/coarse_reframe_module_prompt.md and the speed
run out/reframe_speed/summary.json):

  * Bounded window (<= half a window; single bounded search, NOT iterated):
      recenter dx,dy in {-0.25, 0, +0.25}*fw ; zoom fw in {x0.5, x0.707, x1.0, x1.414}.
  * Separable search (the validated structure):
      1. fw ladder at center: score the 4 fw candidates at recenter (0,0); pick best fw.
      2. recenter at best fw: score the 9 recenters at that fw; pick best recenter.
      The (best_fw, recenter (0,0)) render is SHARED between the two steps and
      cached/deduped -> 12 distinct renders, not 13.
  * Search fidelity: 640x360 ss2, kept 16:9 so the classifier's 384x224 stretch is
    mirrored (no FoV confound); auto_maxiter(fw) per candidate.
  * Scoring: v5 location-quality CORN head, continuous E[ord] in [0,2], within-location
    argmax.

MONOTONE-NON-DECREASING by construction: the original framing (fw x1.0, recenter
(0,0)) is always in the search space (it is the center rung of the fw ladder), so the
returned reframe can never score below the input crop -- the returned score is a valid
cross-location re-ranking key for discovery survivors (the location classifier is the
one model whose scores are legitimately cross-location comparable).

Reused VERBATIM from the probe/speed tools (do not rewire): `render-one` (incl.
`--julia --c`), `auto_maxiter`, `score_lib` Scorer / v5 CORN decode, and the classifier
deploy input (384x224 stretch) read from cfg.

Validation entry points (script-run checks; the GPU reproduce-check is deliberately
NOT in the pytest gate):
  uv run python tools/reframe/reframe.py --gate            # V1: reproduce speed.py 640ss2 separable pick on 5 anchors
  uv run python tools/reframe/reframe.py --time-only       # project batch cost
  uv run python tools/reframe/reframe.py --batch           # V2: ~40 quality-3 locs, paired before/after sheet
  uv run python tools/reframe/reframe.py --rebuild-sheets  # rebuild batch sheet from records.json (GPU-free)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from collections import namedtuple
from decimal import Decimal
from pathlib import Path

# Guard the Windows cp1252 console; keep our own output ASCII regardless.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # tools/reframe -> tools -> repo root
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

# Classifier native paths reused verbatim (same imports speed.py uses).
from active_ckpt import (  # noqa: E402
    BIN, PALETTE, JPG_Q, auto_maxiter, make_scorer, _unique_score3_locations, ACTIVE_CKPT,
)
import location as loc_mod  # noqa: E402  (the one render-one flag builder + family_params)

OUT_DIR = ROOT / "out" / "reframe"
SPEED_DIR = ROOT / "out" / "reframe_speed"    # source of the V1 reference picks
DEFAULT_MODEL = ACTIVE_CKPT   # single source of truth (probe.ACTIVE_CKPT — currently v6)

# --- degenerate-outcome guard hook (opt-in; OFF => byte-identical to today) ---
# When DUMP_GUARD_FIELD is set, `_render` ALSO dumps the raw smooth field co-located
# with each tile (`<tile>.field.bin` + sidecar) via a second `render-one --dump-field`
# at the SAME geometry/fidelity as the scored JPG. A guarded scorer
# (tools/atlas/guard.py) reads that field and gates the tile — so reframe's own
# candidate scoring inherits the guard (it won't climb toward a black/flat crop when
# a passing framing exists, and if every framing fails the reframe score collapses to
# the guard sentinel). The scored JPG is untouched, so a passing crop scores exactly
# as before. Every existing caller leaves this OFF and is unaffected. The suffix must
# equal guard.FIELD_SIDECAR_SUFFIX (asserted at wire time in production_seeder).
DUMP_GUARD_FIELD = False
GUARD_FIELD_SUFFIX = ".field.bin"

# ---- module constants (VALIDATED, provisional -- see module docstring) ----
FW_FACS = (0.5, 2.0 ** -0.5, 1.0, 2.0 ** 0.5)   # geometric sqrt2 ladder, x0.5 .. x1.41
RECENTER = (-0.25, 0.0, 0.25)                    # dx,dy in fractions of fw, per axis (3x3=9)
RENDER_W, RENDER_H, RENDER_SS = 640, 360, 2      # 16:9 search fidelity; mirrors 384x224 stretch
ORIG_COL = FW_FACS.index(1.0)                    # the fw x1.0 rung = the input framing

# The 5 speed.py anchors (V1 gate is against these, in this order).
GATE_ANCHORS = ["m3_shallow", "j0_deep", "j1_shallow", "m1_mid", "m2_mid"]


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
# `family_params` carries the per-family extra-constant slot (empty for mandelbrot/
# julia/multibrot; {p_re,p_im} for phoenix) so a discovery-produced Phoenix location
# reframes with its `p` intact instead of silently dropping it. Defaults empty, so all
# existing (family, c_re, c_im, cx, cy, fw) call sites are unchanged.
Location = namedtuple("Location", "family c_re c_im cx cy fw family_params",
                      defaults=({},))
ReframeResult = namedtuple("ReframeResult", "cx cy fw score trace")


def as_location(loc) -> Location:
    """Accept a Location or a bare (family, c_re, c_im, cx, cy, fw[, family_params]) tuple."""
    return loc if isinstance(loc, Location) else Location(*loc)


def _candidate(loc: Location, fac: float, dx: float, dy: float) -> dict:
    """Geometry for one framing candidate (fw factor + recenter fractions)."""
    cx0, cy0, fw0 = Decimal(str(loc.cx)), Decimal(str(loc.cy)), float(loc.fw)
    fw = fw0 * fac
    cx = cx0 + Decimal(repr(dx * fw))
    cy = cy0 + Decimal(repr(dy * fw))
    return {"fw_factor": fac, "dx": dx, "dy": dy, "fw": fw,
            "cx": str(cx), "cy": str(cy), "maxiter": auto_maxiter(fw)}


def _tile_name(c: dict) -> str:
    ci = FW_FACS.index(c["fw_factor"])
    return f"c{ci}_dx{c['dx']:+.2f}_dy{c['dy']:+.2f}.jpg"


def _render(loc: Location, c: dict, out: Path, w: int, h: int, ss: int) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one",
        "--cx", c["cx"], "--cy", c["cy"], "--fw", repr(c["fw"]),
        "--width", str(w), "--height", str(h),
        "--supersample", str(ss), "--maxiter", str(c["maxiter"]),
        "--palette", PALETTE, "--jpg-quality", str(JPG_Q),
        "--out", str(out),
    ]
    cmd += loc_mod.render_one_flags(loc)   # --family (+ --julia/--c, --p) via the one builder
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and out.exists()
    if not ok:
        return ok, r.stderr[-300:]
    if DUMP_GUARD_FIELD:
        fok, ferr = _dump_guard_field(loc, c, out, w, h, ss)
        if not fok:
            return False, f"guard field: {ferr}"
    return True, ""


def _dump_guard_field(loc: Location, c: dict, out: Path, w: int, h: int, ss: int) -> tuple[bool, str]:
    """Dump the raw smooth field co-located with a tile (`<out>.field.bin`) at the SAME
    geometry/fidelity, so a guarded scorer can gate the tile. `render-one --dump-field`
    exits before coloring, so this touches no colored output. Only called when
    DUMP_GUARD_FIELD is set (the guard hook).

    Sourced from the fast escape-time F64Backend smooth channel
    (`--dump-field-source f64`), not the slow beautiful kernel: the guard reads only
    interior_frac (escape mask) + field_std (a std), both invariant to the
    bailout-normalization offset between the two kernels, so verdicts are unchanged
    (proven by out/atlas/gate_f64_field.py — union-of-20 reproduced exactly). This
    deletes the redundant beautiful second-render that dominated each tile's wall."""
    fbin = Path(str(out) + GUARD_FIELD_SUFFIX)
    # The fast f64 escape-time smooth-channel source exists for every escape-time
    # family — mandelbrot/julia (degree 2) and multibrot (degree >= 3, dispatched
    # through the trait `sample` -> `sample_multibrot`). Only Phoenix (no escape-time
    # backend) would still reject; no descendable phoenix path dumps a guard field.
    # The guard reads only interior_frac (escape mask) + field_std, both invariant to
    # the bailout-normalization offset between the f64 and beautiful kernels, so
    # verdicts are unchanged (same offset-invariant reasoning that validated the
    # mandelbrot/julia f64 source).
    src = "f64"
    cmd = [
        str(BIN), "render-one",
        "--cx", c["cx"], "--cy", c["cy"], "--fw", repr(c["fw"]),
        "--width", str(w), "--height", str(h),
        "--supersample", str(ss), "--maxiter", str(c["maxiter"]),
        "--dump-field", str(fbin), "--dump-field-source", src,
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and fbin.exists()
    return ok, ("" if ok else r.stderr[-300:])


def reframe_location(
    loc,
    *,
    scorer,
    seed: int = 0,
    workdir: Path | None = None,
    workers: int = 4,
    w: int = RENDER_W,
    h: int = RENDER_H,
    ss: int = RENDER_SS,
) -> ReframeResult:
    """Best coarse reframe of `loc` within the bounded window (see module docstring).

    loc     = (family, c_re, c_im, cx, cy, fw); cx/cy/fw as strings (decimal geometry).
    scorer  = a shared score_lib.Scorer (v5 CORN); build once, map over survivors.
    seed    = recorded in the trace for reproducibility; the bounded search itself is
              deterministic (no stochastic component -- seed is reserved for the
              downstream palette/emission step that is out of scope here).

    Returns ReframeResult(cx, cy, fw, score, trace): geometry only + the reframed
    E[ord] + the fw-ladder / recenter scores for sheets/debug.
    """
    loc = as_location(loc)
    tiles = Path(workdir if workdir is not None
                 else OUT_DIR / "_scratch" / f"loc_{seed}") / "tiles"
    score_cache: dict[str, tuple] = {}   # tile_name -> (score, p_notbad, p_good); dedups shared center

    def ensure_scored(cands: list[dict]) -> list[tuple]:
        need = [c for c in cands if _tile_name(c) not in score_cache]
        to_render = [c for c in need if not (tiles / _tile_name(c)).exists()]
        if to_render:
            fails = []
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_render, loc, c, tiles / _tile_name(c), w, h, ss): c
                        for c in to_render}
                for fut in cf.as_completed(futs):
                    ok, err = fut.result()
                    if not ok:
                        fails.append((futs[fut], err))
            if fails:
                c, err = fails[0]
                raise SystemExit(f"reframe render failed ({len(fails)}) "
                                 f"[{_tile_name(c)}]: {err}")
        if need:
            triples = scorer.score_paths([tiles / _tile_name(c) for c in need])
            for c, t in zip(need, triples):
                score_cache[_tile_name(c)] = tuple(float(x) for x in t)
        return [score_cache[_tile_name(c)] for c in cands]

    # Step 1 -- fw ladder at center; pick best fw.
    fw_cands = [_candidate(loc, fac, 0.0, 0.0) for fac in FW_FACS]
    fw_scores = ensure_scored(fw_cands)
    best_ci = max(range(len(FW_FACS)), key=lambda i: fw_scores[i][0])
    best_fac = FW_FACS[best_ci]

    # Step 2 -- recenter at best fw; the (best_fw, center) tile is cached from step 1.
    rc_cands = [_candidate(loc, best_fac, dx, dy) for dy in RECENTER for dx in RECENTER]
    rc_scores = ensure_scored(rc_cands)
    best_j = max(range(len(rc_cands)), key=lambda i: rc_scores[i][0])
    chosen, chosen_sc = rc_cands[best_j], rc_scores[best_j]

    trace = {
        "seed": seed,
        "render": {"w": w, "h": h, "ss": ss, "palette": PALETTE, "jpg_quality": JPG_Q,
                   "n_renders": len(score_cache)},
        "fw_ladder": [{"fw_factor": FW_FACS[i], "fw": fw_cands[i]["fw"],
                       "score": fw_scores[i][0], "p_notbad": fw_scores[i][1],
                       "p_good": fw_scores[i][2]} for i in range(len(FW_FACS))],
        "best_fw_factor": best_fac,
        "recenter": [{"dx": c["dx"], "dy": c["dy"], "score": s[0],
                      "p_notbad": s[1], "p_good": s[2]}
                     for c, s in zip(rc_cands, rc_scores)],
        "chosen": {"fw_factor": best_fac, "dx": chosen["dx"], "dy": chosen["dy"]},
        "original_score": fw_scores[ORIG_COL][0],
    }
    return ReframeResult(cx=chosen["cx"], cy=chosen["cy"], fw=chosen["fw"],
                         score=chosen_sc[0], trace=trace)


# --------------------------------------------------------------------------- #
# Validation 1 -- reproduce speed.py's 640x360-ss2 separable pick (GATE)
# --------------------------------------------------------------------------- #
def _sep_pick_from_scores(score_of) -> tuple[float, float, float]:
    """Separable pick as (fw_factor, dx, dy) given a score(fw_factor, dx, dy) lookup."""
    best_fac = max(FW_FACS, key=lambda f: score_of(f, 0.0, 0.0))
    best = max(((best_fac, dx, dy) for dy in RECENTER for dx in RECENTER),
               key=lambda t: score_of(*t))
    return best


def _speed_reference(anchor_key: str) -> tuple[Location, tuple, float]:
    """Load a speed anchor -> (Location, reference separable pick, ref pick score),
    the pick recomputed from the stored 640x360_ss2 grid scores."""
    rp = SPEED_DIR / anchor_key / "records.json"
    if not rp.exists():
        raise SystemExit(f"missing {rp}; run tools/reframe_probe/speed.py first")
    rec = json.loads(rp.read_text())
    a = rec["anchor"]
    loc = Location(family=rec["family"], c_re=a["c_re"], c_im=a["c_im"],
                   cx=a["cx"], cy=a["cy"], fw=a["fw"])
    # framing scores at 640x360 ss2, keyed by (fw_factor, dx, dy)
    grid = {(f["fw_factor"], f["dx"], f["dy"]): f["scores"]["640x360_ss2"]
            for f in rec["framings"]}

    def score_of(fac, dx, dy):
        return grid[(fac, dx, dy)]

    pick = _sep_pick_from_scores(score_of)
    return loc, pick, score_of(*pick)


def run_gate(args):
    scorer = make_scorer(args.model)
    print(f"\n=== V1 GATE: module pick vs speed.py 640x360-ss2 separable pick ({len(GATE_ANCHORS)} anchors) ===")
    print("(gate on the PICK, not the score -- scores drift within GPU nondeterminism)")
    print(f"{'anchor':<12} {'family':<11} {'ref pick (fw,dx,dy)':<26} {'module pick':<26} {'match':<6} {'module E':>9}")
    n_ok = 0
    for key in GATE_ANCHORS:
        loc, ref_pick, ref_sc = _speed_reference(key)
        res = reframe_location(loc, scorer=scorer, seed=0,
                               workdir=OUT_DIR / "_gate" / key, workers=args.workers)
        ch = res.trace["chosen"]
        mod_pick = (ch["fw_factor"], ch["dx"], ch["dy"])
        match = (mod_pick == ref_pick)
        n_ok += match

        def fmt(p):
            return f"x{p[0]:.3g} d{p[1]:+.2f},{p[2]:+.2f}"
        print(f"{key:<12} {loc.family:<11} {fmt(ref_pick):<26} {fmt(mod_pick):<26} "
              f"{'OK' if match else 'FAIL':<6} {res.score:>9.4f}")
    verdict = "PASS" if n_ok == len(GATE_ANCHORS) else "FAIL"
    print(f"\nGATE {verdict}: {n_ok}/{len(GATE_ANCHORS)} picks reproduced "
          f"(reference = out/reframe_speed/<key>/records.json 640x360_ss2 separable).")
    if verdict != "PASS":
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Validation 2 -- broad before/after batch (visual, paired)
# --------------------------------------------------------------------------- #
def sample_quality3(n_total: int, seed: int) -> list[Location]:
    """~n_total unique quality-3 locations, mandelbrot+julia mix spanning zoom depth.
    Deterministic: even-spaced over each family's fw-sorted list (spans zoom), split
    ~proportional to the pool. `seed` only rotates the even-spacing phase."""
    locs = _unique_score3_locations()
    fams = {"mandelbrot": [], "julia": []}
    for l in locs:
        fams.setdefault(l["family"], []).append(l)
    for k in fams:
        fams[k].sort(key=lambda r: (float(r["fw"]), r["cx"], r["cy"]))

    total_pool = sum(len(v) for v in fams.values())
    out: list[Location] = []
    for fam, rows in fams.items():
        if not rows:
            continue
        want = max(1, round(n_total * len(rows) / total_pool))
        want = min(want, len(rows))
        # even-spaced indices across the fw-sorted list (spans zoom), phase by seed
        step = len(rows) / want
        phase = (seed % max(1, want)) * (step / max(1, want))
        idxs = sorted({min(len(rows) - 1, int(phase + i * step)) for i in range(want)})
        for i in idxs:
            r = rows[i]
            out.append(Location(family=r["family"], c_re=r["c_re"], c_im=r["c_im"],
                                cx=r["cx"], cy=r["cy"], fw=r["fw"],
                                family_params=r.get("family_params") or {}))
    out.sort(key=lambda l: (l.family, float(l.fw)))
    return out


def _batch_workdir(i: int, loc: Location) -> Path:
    return OUT_DIR / "batch" / f"{i:02d}_{loc.family}"


def run_batch(args):
    scorer = make_scorer(args.model)
    locs = sample_quality3(args.n, args.seed)
    print(f"\n=== V2 BATCH: reframing {len(locs)} quality-3 locations (seed={args.seed}) ===")
    print(f"classifier deploy input: geometry={scorer.cfg.get('geometry')} "
          f"interp={scorer.cfg.get('interpolation')}  1280x720 -> 384x224")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for i, loc in enumerate(locs):
        wd = _batch_workdir(i, loc)
        res = reframe_location(loc, scorer=scorer, seed=args.seed,
                               workdir=wd, workers=args.workers)
        orig = res.trace["original_score"]
        ch = res.trace["chosen"]
        rec = {
            "i": i, "family": loc.family,
            "loc": {"family": loc.family, "c_re": loc.c_re, "c_im": loc.c_im,
                    "cx": loc.cx, "cy": loc.cy, "fw": loc.fw},
            "result": {"cx": res.cx, "cy": res.cy, "fw": res.fw, "score": res.score},
            "original_score": orig, "delta": res.score - orig,
            "chosen": ch, "workdir": str(wd.relative_to(OUT_DIR)),
            "trace": res.trace,
        }
        records.append(rec)
        print(f"[{i:02d} {loc.family:<10} fw={float(loc.fw):.3e}] "
              f"orig E={orig:.3f} -> reframed E={res.score:.3f} "
              f"(d={res.score - orig:+.3f})  fw x{ch['fw_factor']:.3g} "
              f"d{ch['dx']:+.2f},{ch['dy']:+.2f}")
    (OUT_DIR / "records.json").write_text(json.dumps(records, indent=2))
    sheet = build_batch_sheet(records)
    dt = time.time() - t0
    (OUT_DIR / "COMPLETE").write_text(f"done {len(locs)} locations in {dt:.0f}s\n")
    _print_batch_summary(records)
    print(f"\nDONE {len(locs)} locations in {dt:.0f}s ({dt/max(1,len(locs)):.1f}s/loc) "
          f"-> {sheet}")


def _print_batch_summary(records):
    deltas = [r["delta"] for r in records]
    changed = [r for r in records if r["chosen"] != {"fw_factor": 1.0, "dx": 0.0, "dy": 0.0}]
    print(f"\n=== batch summary ===")
    print(f"  locations: {len(records)}  (reframe changed the crop on {len(changed)})")
    print(f"  delta E[ord]: mean {sum(deltas)/len(deltas):+.3f}  "
          f"max {max(deltas):+.3f}  min {min(deltas):+.3f}  (min is 0 by construction)")
    # zoom-preference tallies -- where the provisional config likes to move
    from collections import Counter
    fw_pref = Counter(round(r["chosen"]["fw_factor"], 3) for r in records)
    print("  chosen fw factor: " + "  ".join(f"x{k:.3g}:{v}" for k, v in sorted(fw_pref.items())))


# ---- paired before/after sheet (tiles are the 640x360 search renders, reused) ----
TW, TH = 384, 216
PAD, LBL_H, GAP = 8, 60, 6
PAIRS_PER_ROW = 2
GUT_T = 34


def _orig_tile(wd: Path) -> Path:
    return wd / "tiles" / _tile_name({"fw_factor": 1.0, "dx": 0.0, "dy": 0.0})


def _chosen_tile(wd: Path, ch: dict) -> Path:
    return wd / "tiles" / _tile_name(ch)


def build_batch_sheet(records: list[dict]) -> Path:
    from PIL import Image, ImageDraw
    n = len(records)
    nrow = (n + PAIRS_PER_ROW - 1) // PAIRS_PER_ROW
    pair_w = 2 * TW + GAP + 2 * PAD
    cell_h = TH + LBL_H + 2 * PAD
    W = PAIRS_PER_ROW * pair_w
    H = GUT_T + nrow * cell_h

    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"coarse reframe -- before | after (paired, same location; "
              f"crops at 640x360 search fidelity)   n={n}   "
              f"delta = reframed E[ord] - original E[ord] (>=0 by construction)",
              fill=(235, 235, 235))

    for i, r in enumerate(records):
        wd = OUT_DIR / r["workdir"]
        row, col = divmod(i, PAIRS_PER_ROW)
        x0 = col * pair_w + PAD
        y0 = GUT_T + row * cell_h + PAD
        for j, (tp, tag) in enumerate([(_orig_tile(wd), "BEFORE (x1.0)"),
                                       (_chosen_tile(wd, r["chosen"]), "AFTER")]):
            x = x0 + j * (TW + GAP)
            if tp.exists():
                im = Image.open(tp).convert("RGB").resize((TW, TH))
                sheet.paste(im, (x, y0))
            draw.rectangle([x, y0 + TH, x + TW, y0 + TH + LBL_H], fill=(28, 28, 32))
            draw.text((x + 4, y0 + TH + 3), tag, fill=(210, 210, 218))
        # shared label block under the AFTER tile: delta + chosen framing
        ch = r["chosen"]
        d = r["delta"]
        dcol = (90, 230, 110) if d >= 0.05 else ((200, 200, 210) if d < 1e-6 else (235, 210, 90))
        lo = r["loc"]
        ctxt = "" if lo["c_re"] is None else f" c=({lo['c_re']},{lo['c_im']})"
        draw.text((x0, y0 + TH + 20),
                  f"{r['family']} fw={float(lo['fw']):.3e}{ctxt}", fill=(180, 180, 190))
        draw.text((x0, y0 + TH + 36),
                  f"orig E={r['original_score']:.3f} -> {r['result']['score']:.3f}   "
                  f"d={d:+.3f}   chosen fw x{ch['fw_factor']:.3g} "
                  f"d{ch['dx']:+.2f},{ch['dy']:+.2f}", fill=dcol)
        # gold border on the AFTER tile when the reframe actually moved
        if ch != {"fw_factor": 1.0, "dx": 0.0, "dy": 0.0}:
            ax = x0 + (TW + GAP)
            for t in range(2):
                draw.rectangle([ax - 1 - t, y0 - 1 - t, ax + TW + t, y0 + TH + t],
                               outline=(245, 215, 40))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "batch_sheet.png"
    sheet.save(out)
    return out


def run_rebuild(args):
    rp = OUT_DIR / "records.json"
    if not rp.exists():
        raise SystemExit(f"no {rp}; run --batch first")
    records = json.loads(rp.read_text())
    sheet = build_batch_sheet(records)
    _print_batch_summary(records)
    print(f"  rebuilt {sheet}")


# --------------------------------------------------------------------------- #
# --time-only
# --------------------------------------------------------------------------- #
def run_time_only(args):
    scorer = make_scorer(args.model)
    locs = sample_quality3(args.n, args.seed)
    probe = [min(locs, key=lambda l: float(l.fw)), max(locs, key=lambda l: float(l.fw))]
    print(f"\n=== timing reframe_location on deepest + shallowest of {len(locs)} sampled ===")
    per = []
    for loc in probe:
        wd = OUT_DIR / "_timing" / loc.family
        t = time.time()
        res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=args.workers)
        el = time.time() - t
        per.append(el)
        print(f"  {loc.family:<10} fw={float(loc.fw):.3e}: {el:.2f}s "
              f"({res.trace['render']['n_renders']} renders + scores)  "
              f"chosen fw x{res.trace['chosen']['fw_factor']:.3g}")
    avg = sum(per) / len(per)
    total = avg * len(locs)
    print(f"\n  avg {avg:.2f}s/location  ->  PROJECTED {len(locs)} locations: "
          f"~{total:.0f}s (~{total/60:.1f} min) at workers={args.workers}")
    print(f"  -> {'BACKGROUND recommended' if total > 120 else 'foreground OK'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true", help="V1: reproduce speed.py 640ss2 separable pick")
    ap.add_argument("--batch", action="store_true", help="V2: reframe ~40 quality-3 locs + paired sheet")
    ap.add_argument("--time-only", action="store_true", help="project batch cost, exit")
    ap.add_argument("--rebuild-sheets", action="store_true", help="rebuild batch sheet from records.json (GPU-free)")
    ap.add_argument("--n", type=int, default=40, help="batch size (V2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="parallel render-one workers")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="classifier checkpoint")
    args = ap.parse_args()

    if args.gate:
        run_gate(args)
    elif args.time_only:
        run_time_only(args)
    elif args.rebuild_sheets:
        run_rebuild(args)
    elif args.batch:
        run_batch(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
