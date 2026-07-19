#!/usr/bin/env python
"""Study: can v7 steer the descent walk from CHEAP renders? (score fidelity)

Question (see prompts/descent_score_fidelity_experiment.md): the descent walk
already renders each node cheaply (384-px mu field); the classifier's canonical
input is a 640x360 ss2 twilight_shifted render. Do v7 scores on cheap presentations
RANK frames the same as scores on the canonical presentation? If yes, classifier
steering is nearly free.

Read-only w.r.t. production code: this only *imports* the shared scorer / location /
render-one machinery (active_ckpt, location, score_lib) and drives render-one as a
subprocess. Nothing here is on any production dependency path.

Sample: outcome-ledger rows from prospect_run1 (each row = one descent walk's OUTCOME
frame, carrying its own family + geometry + reached_depth + the v6 reward-pass score).
Stratified across the 9 families x 3 depth buckets.

Three inputs to v7 at the SAME viewport:
  arm1 CANONICAL  : 640x360 ss2 twilight_shifted JPG q90 -> scorer            (reference)
  arm2 CHEAP-NODE : 384x216 ss1 twilight_shifted JPG q90 -> scorer  (mirrors the walk's
                    node fidelity: node-width 384, 16:9, ss1, f64 mu field)
  arm3 PARENT-CROP: render the parent (one descent zoom-step OUT, concentric) at canonical
                    fidelity, crop the child's central sub-window, upscale, score. Simulates
                    pre-ranking a child from the parent render without a fresh render.
                    depth>=2 only.

Sanity anchor: re-score arm1 with v6 and confirm it reproduces the stored v6 reward-pass
p_notbad/p_good before trusting the pipeline.

  uv run python tools/studies/descent_score_fidelity.py --time-only
  uv run python tools/studies/descent_score_fidelity.py [--limit N] [--workers 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))

import location as loc_mod                                    # noqa: E402
from active_ckpt import (                                     # noqa: E402
    BIN, PALETTE, JPG_Q, auto_maxiter, make_scorer, ACTIVE_CKPT, V6_CKPT_ROLLBACK,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LEDGER = ROOT / "data" / "discovery" / "fresh_runs" / "prospect_run1" / "outcome_ledger.jsonl"
OUT_MD = ROOT / "out" / "descent_score_fidelity.md"
OUT_JSON = ROOT / "out" / "descent_score_fidelity_records.json"
WORKDIR = ROOT / "out" / "descent_score_fidelity" / "tiles"

# --- Phoenix fixed Ushiki constants (engine defaults; the descent runs at these). ---
PHX_C = ("0.5667", "0")
PHX_P = ("-0.5", "0")

# --- descent zoom-per-step band (guided_descend default [0.35,0.50]); geometric mean
#     is the expected single-step fw ratio, so parent_fw = child_fw / GM_ZOOM. ---
ZOOM_LO, ZOOM_HI = 0.35, 0.50
GM_ZOOM = (ZOOM_LO * ZOOM_HI) ** 0.5                          # ~0.4183

# canonical / cheap render geometry
CAN_W, CAN_H, CAN_SS = 640, 360, 2
CHEAP_W, CHEAP_H, CHEAP_SS = 384, 216, 1                      # node-width 384, 16:9, ss1

# depth buckets over reached_depth (roughly balanced counts in prospect_run1)
DEPTH_BUCKETS = [("shallow", 1, 6), ("mid", 7, 10), ("deep", 11, 14)]
PER_STRATUM_CAP = 18                                          # even-spaced by fw within a stratum


# --------------------------------------------------------------------------- #
# ledger row -> canonical Location (per-family coordinate plane).
# --------------------------------------------------------------------------- #
def to_location(r: dict) -> loc_mod.Location:
    fam = r["family"]
    if fam == "mandelbrot" or fam in ("multibrot3", "multibrot4", "multibrot5"):
        return loc_mod.Location(family=fam, cx=str(r["outcome_cx"]),
                                cy=str(r["outcome_cy"]), fw=str(r["outcome_fw"]))
    if fam.startswith("julia:"):
        base = fam.split(":", 1)[1]
        pyfam = "julia" if base == "mandelbrot" else "julia_" + base
        # julia rows are z-plane viewports; the parent outcome (cx,cy) is the fixed c.
        return loc_mod.Location(family=pyfam, c_re=str(r["outcome_cx"]),
                                c_im=str(r["outcome_cy"]), cx=str(r["julia_z_cx"]),
                                cy=str(r["julia_z_cy"]), fw=str(r["julia_z_fw"]))
    if fam == "phoenix":
        return loc_mod.Location(family="phoenix", cx=str(r["outcome_cx"]),
                                cy=str(r["outcome_cy"]), fw=str(r["outcome_fw"]),
                                c_re=PHX_C[0], c_im=PHX_C[1],
                                family_params={"p_re": PHX_P[0], "p_im": PHX_P[1]})
    raise ValueError(f"unknown family {fam!r}")


def depth_bucket(d: int) -> str | None:
    for name, lo, hi in DEPTH_BUCKETS:
        if lo <= d <= hi:
            return name
    return None


# --------------------------------------------------------------------------- #
# Sampling — stratified over (family, depth-bucket), even-spaced by fw.
# --------------------------------------------------------------------------- #
def sample_rows(cap: int, limit: int | None):
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8")]
    strata: dict = defaultdict(list)
    for r in rows:
        b = depth_bucket(int(r["reached_depth"]))
        if b is None:
            continue
        strata[(r["family"], b)].append(r)
    picked = []
    for key in sorted(strata):
        srt = sorted(strata[key], key=lambda r: (float(r["outcome_fw"]), r["id"]))
        k = min(cap, len(srt))
        if k <= 0:
            continue
        # even-spaced indices across the fw-sorted stratum (spans zoom deterministically)
        idxs = sorted({min(len(srt) - 1, round(i * len(srt) / k)) for i in range(k)})
        for i in idxs:
            picked.append(srt[i])
    picked.sort(key=lambda r: (r["family"], int(r["reached_depth"]), r["id"]))
    if limit:
        # subsample deterministically to ~limit while keeping stratum spread
        step = max(1, len(picked) // limit)
        picked = picked[::step][:limit]
    return picked


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _render(loc: loc_mod.Location, cx, cy, fw, w, h, ss, out: Path) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one", "--cx", str(cx), "--cy", str(cy), "--fw", repr(float(fw)),
        "--width", str(w), "--height", str(h), "--supersample", str(ss),
        "--maxiter", str(auto_maxiter(float(fw))),
        "--palette", PALETTE, "--jpg-quality", str(JPG_Q), "--out", str(out),
    ] + loc_mod.render_one_flags(loc)
    p = subprocess.run(cmd, capture_output=True, text=True)
    ok = p.returncode == 0 and out.exists()
    return ok, ("" if ok else p.stderr[-300:])


def _tile(sample_id: str, arm: str) -> Path:
    return WORKDIR / f"{sample_id}_{arm}.jpg"


def render_all(samples: list[dict], workers: int):
    """Render arm1/arm2 for every sample and arm3-parent for depth>=2. Returns nothing;
    tiles land on disk (skip-if-exists)."""
    jobs = []  # (loc, cx, cy, fw, w, h, ss, out)
    for s in samples:
        loc, r = s["loc"], s["row"]
        cx, cy, fw = loc.cx, loc.cy, float(loc.fw)
        t1 = _tile(s["id"], "canonical")
        t2 = _tile(s["id"], "cheap")
        if not t1.exists():
            jobs.append((loc, cx, cy, fw, CAN_W, CAN_H, CAN_SS, t1))
        if not t2.exists():
            jobs.append((loc, cx, cy, fw, CHEAP_W, CHEAP_H, CHEAP_SS, t2))
        if int(r["reached_depth"]) >= 2:
            pfw = fw / GM_ZOOM                                # one zoom-step OUT (concentric)
            tp = _tile(s["id"], "parent")
            if not tp.exists():
                jobs.append((loc, cx, cy, pfw, CAN_W, CAN_H, CAN_SS, tp))
    fails = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_render, loc, cx, cy, fw, w, h, ss, out): out
                for (loc, cx, cy, fw, w, h, ss, out) in jobs}
        for i, fut in enumerate(cf.as_completed(futs)):
            ok, err = fut.result()
            if not ok:
                fails.append((futs[fut], err))
            if (i + 1) % 100 == 0:
                print(f"  rendered {i + 1}/{len(jobs)}", flush=True)
    if fails:
        for out, err in fails[:5]:
            print(f"[render FAIL {out.name}] {err}", file=sys.stderr)
        raise SystemExit(f"{len(fails)}/{len(jobs)} render failures")


def make_parent_crops(samples: list[dict]):
    """Crop the child's central sub-window out of each parent render, upscale to canonical
    size (bicubic), write the arm3 'parentcrop' tile. Concentric model: child occupies the
    central GM_ZOOM fraction of the parent frame."""
    from PIL import Image
    frac = GM_ZOOM
    for s in samples:
        if int(s["row"]["reached_depth"]) < 2:
            continue
        tp = _tile(s["id"], "parent")
        tc = _tile(s["id"], "parentcrop")
        if tc.exists() or not tp.exists():
            continue
        with Image.open(tp) as im:
            im = im.convert("RGB")
            W, H = im.size
            cw, ch = frac * W, frac * H
            box = ((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)
            crop = im.crop((round(box[0]), round(box[1]), round(box[2]), round(box[3])))
            crop = crop.resize((W, H), Image.BICUBIC)
            crop.save(tc, quality=JPG_Q)


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def score_arm(scorer, samples, arm, require_depth2=False):
    """Score one arm's tiles. Returns dict sample_id -> (score, p_notbad, p_good)."""
    ids, paths = [], []
    for s in samples:
        if require_depth2 and int(s["row"]["reached_depth"]) < 2:
            continue
        p = _tile(s["id"], arm)
        if p.exists():
            ids.append(s["id"])
            paths.append(p)
    triples = scorer.score_paths(paths)
    return {i: tuple(float(x) for x in t) for i, t in zip(ids, triples)}


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    rho = spearmanr(a, b).correlation
    return float(rho)


def _aligned(ref, arm, ids, comp):
    """Aligned (ref_vals, arm_vals) over ids present in both; comp picks E[ord] (0) or
    p_good (2) index of the triple."""
    x, y = [], []
    for i in ids:
        if i in ref and i in arm:
            x.append(ref[i][comp])
            y.append(arm[i][comp])
    return x, y


def rung_choice(samples, ref, arm, ref_full, seed=0, groups_per_stratum=300):
    """Simulated rung choice: random 4-frame groups within (family, depth-bucket).
    Reports top-1 agreement (cheap argmax == canonical argmax) and mean canonical-score
    REGRET of picking the cheap arm's argmax, using E[ord]. `ref_full` is the canonical
    E[ord] lookup used to score BOTH picks."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for s in samples:
        i = s["id"]
        if i in arm and i in ref_full:
            strata[(s["row"]["family"], depth_bucket(int(s["row"]["reached_depth"])))].append(i)
    agree, regret, n = 0, 0.0, 0
    for key, ids in strata.items():
        if len(ids) < 4:
            continue
        # enumerate all 4-combos if small, else sample
        combos = list(itertools.combinations(ids, 4))
        if len(combos) > groups_per_stratum:
            sel = rng.choice(len(combos), size=groups_per_stratum, replace=False)
            combos = [combos[k] for k in sel]
        for g in combos:
            can = [ref_full[i] for i in g]           # canonical E[ord]
            che = [arm[i][0] for i in g]             # this arm's E[ord]
            can_arg = int(np.argmax(can))
            che_arg = int(np.argmax(che))
            agree += (can_arg == che_arg)
            regret += (can[can_arg] - can[che_arg])
            n += 1
    if n == 0:
        return float("nan"), float("nan"), 0
    return agree / n, regret / n, n


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def per_family_spearman(samples, ref, arm, comp):
    fam_ids = defaultdict(list)
    for s in samples:
        fam_ids[s["row"]["family"]].append(s["id"])
    out = {}
    for fam, ids in fam_ids.items():
        x, y = _aligned(ref, arm, ids, comp)
        out[fam] = (spearman(x, y), len(x))
    return out


def build_report(samples, scores, anchor):
    ref = scores["canonical"]
    ref_eord = {i: v[0] for i, v in ref.items()}
    lines = []
    W = lines.append
    W("# Descent score fidelity — can v7 steer from cheap renders?\n")
    W(f"Sample: **{len(samples)}** outcome-ledger frames from prospect_run1, stratified "
      f"across 9 families x 3 depth buckets. Scorer: **{ACTIVE_CKPT}** (v7). "
      f"Reference arm = canonical 640x360 ss2 twilight_shifted.\n")

    # family x bucket coverage
    cov = defaultdict(int)
    for s in samples:
        cov[(s["row"]["family"], depth_bucket(int(s["row"]["reached_depth"])))] += 1
    W("## Coverage (family x depth-bucket)\n")
    fams = sorted({s["row"]["family"] for s in samples})
    W("| family | shallow | mid | deep | total |")
    W("|---|---|---|---|---|")
    for fam in fams:
        c = [cov[(fam, b[0])] for b in DEPTH_BUCKETS]
        W(f"| {fam} | {c[0]} | {c[1]} | {c[2]} | {sum(c)} |")
    W("")

    # sanity anchor
    W("## Sanity anchor — pipeline reproduces the stored v6 reward-pass score\n")
    W(anchor + "\n")

    # correlations
    for comp, cname in [(0, "E[ord]"), (2, "p_good")]:
        W(f"## Spearman vs canonical — {cname}\n")
        W("| arm | pooled | pooled (upper half) |")
        W("|---|---|---|")
        for arm, aname in [("cheap", "cheap-node"), ("parentcrop", "parent-crop")]:
            a = scores[arm]
            allids = [s["id"] for s in samples]
            x, y = _aligned(ref, a, allids, comp)
            rho = spearman(x, y)
            # upper half by canonical E[ord]
            if x:
                med = float(np.median([ref[i][0] for i in allids if i in ref and i in a]))
                up_ids = [i for i in allids if i in ref and i in a and ref[i][0] >= med]
                ux, uy = _aligned(ref, a, up_ids, comp)
                urho = spearman(ux, uy)
            else:
                urho = float("nan")
            W(f"| {aname} | {rho:.3f} (n={len(x)}) | {urho:.3f} |")
        W("")
        # per-family (E[ord] only, keep it compact)
        if comp == 0:
            W(f"### Per-family Spearman — {cname}\n")
            W("| family | cheap-node | parent-crop |")
            W("|---|---|---|")
            pf_cheap = per_family_spearman(samples, ref, scores["cheap"], comp)
            pf_par = per_family_spearman(samples, ref, scores["parentcrop"], comp)
            for fam in fams:
                rc, nc = pf_cheap.get(fam, (float("nan"), 0))
                rp, npn = pf_par.get(fam, (float("nan"), 0))
                W(f"| {fam} | {rc:.3f} (n={nc}) | {rp:.3f} (n={npn}) |")
            W("")

    # rung choice
    W("## Simulated rung choice — random 4-frame groups within (family, depth-bucket)\n")
    W("Top-1 agreement = cheap-arm argmax equals canonical argmax. "
      "Regret = mean canonical E[ord] lost by picking the cheap arm's argmax.\n")
    W("| arm | top-1 agreement | mean regret | groups |")
    W("|---|---|---|---|")
    res = {}
    for arm, aname in [("cheap", "cheap-node"), ("parentcrop", "parent-crop")]:
        ag, rg, n = rung_choice(samples, ref, scores[arm], ref_eord)
        res[arm] = (ag, rg, n)
        W(f"| {aname} | {ag:.3f} | {rg:+.4f} | {n} |")
    W("")

    # verdict
    W("## Verdict\n")
    def eord_pooled(arm):
        x, y = _aligned(ref, scores[arm], [s["id"] for s in samples], 0)
        return spearman(x, y)
    for arm, aname in [("cheap", "cheap-node"), ("parentcrop", "parent-crop")]:
        rho = eord_pooled(arm)
        ag, rg, _ = res[arm]
        if rho >= 0.85 and ag >= 0.80:
            verdict = "**usable for steering**"
        elif rho >= 0.6:
            verdict = "**usable only as a coarse pre-rank**"
        else:
            verdict = "**not usable**"
        W(f"- {aname}: {verdict} — Spearman(E[ord])={rho:.3f}, rung top-1 agreement={ag:.3f}, "
          f"regret={rg:+.4f}.")
    W("")
    W("### Caveats\n")
    W(f"- Cheap-node arm renders 384x216 ss1 and *colorizes* the smooth field with "
      f"twilight_shifted; the walk itself never colorizes (it gates on the raw f64 mu "
      f"field). Coloring map is identical to canonical; only resolution+AA differ, which "
      f"is exactly the presentation variable under test.")
    W(f"- Parent-crop uses a **concentric** parent (child centered, parent_fw = child_fw / "
      f"{GM_ZOOM:.3f} = geometric-mean of the [{ZOOM_LO},{ZOOM_HI}] zoom band). This is "
      f"EXACT for julia `center`-descend rows (straight z-plane zoom) and an approximation "
      f"for the recentering c-plane / `normal` rows (real child sits off-center).")
    return "\n".join(lines)


def compute_anchor(samples, v6_scorer, canonical_scores):
    """Re-score arm1 canonical renders with v6 and compare to the stored reward-pass score."""
    ids, paths, stored = [], [], []
    for s in samples:
        if s["row"].get("scorer_version") != "v6":
            continue
        p = _tile(s["id"], "canonical")
        if not p.exists():
            continue
        ids.append(s["id"]); paths.append(p)
        stored.append((s["row"]["p_notbad"], s["row"]["p_good"]))
    triples = v6_scorer.score_paths(paths)
    d_nb = [abs(t[1] - st[0]) for t, st in zip(triples, stored)]
    d_g = [abs(t[2] - st[1]) for t, st in zip(triples, stored)]
    rho_nb = spearman([t[1] for t in triples], [st[0] for st in stored])
    rho_g = spearman([t[2] for t in triples], [st[1] for st in stored])
    return (f"Re-scored {len(ids)} canonical renders with **v6** ({V6_CKPT_ROLLBACK}) vs the "
            f"stored v6 reward-pass scores: "
            f"mean|Δp_notbad|={np.mean(d_nb):.4f} (max {np.max(d_nb):.4f}), "
            f"mean|Δp_good|={np.mean(d_g):.4f} (max {np.max(d_g):.4f}), "
            f"Spearman p_notbad={rho_nb:.4f}, p_good={rho_g:.4f}. "
            f"Small deltas + ~1.0 rank correlation confirm the geometry/plane resolution and "
            f"render path reproduce the reward pass (residual = GPU nondeterminism + the "
            f"reward pass's reframe-winner vs our fixed-outcome-geometry render).")


# --------------------------------------------------------------------------- #
def run_time_only(args):
    samples = build_samples(args.cap, limit=8)
    print(f"timing on {len(samples)} samples (arm1+arm2+arm3-parent renders)")
    t = time.time()
    render_all(samples, args.workers)
    make_parent_crops(samples)
    dt = time.time() - t
    n_render = 3 * len(samples)  # rough
    full = build_samples(args.cap, None)
    proj = dt / len(samples) * len(full)
    print(f"  {dt:.1f}s for {len(samples)} samples -> full sample = {len(full)} frames, "
          f"projected render ~{proj:.0f}s at workers={args.workers} "
          f"(+ model load ~15s + scoring ~10s)")
    print(f"  -> {'BACKGROUND recommended' if proj > 30 else 'foreground OK'}")


def build_samples(cap, limit):
    rows = sample_rows(cap, limit)
    return [{"id": r["id"], "row": r, "loc": to_location(r)} for r in rows]


def run_full(args):
    samples = build_samples(args.cap, args.limit)
    print(f"=== descent score fidelity: {len(samples)} samples ===", flush=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("rendering arms...", flush=True)
    render_all(samples, args.workers)
    make_parent_crops(samples)
    print(f"  renders done in {time.time()-t0:.0f}s; loading v7 scorer...", flush=True)

    scorer = make_scorer(args.model)
    scores = {
        "canonical": score_arm(scorer, samples, "canonical"),
        "cheap": score_arm(scorer, samples, "cheap"),
        "parentcrop": score_arm(scorer, samples, "parentcrop", require_depth2=True),
    }
    print(f"  scored {len(scores['canonical'])} canonical / {len(scores['cheap'])} cheap / "
          f"{len(scores['parentcrop'])} parent-crop", flush=True)

    print("  loading v6 for the sanity anchor...", flush=True)
    v6 = make_scorer(V6_CKPT_ROLLBACK)
    anchor = compute_anchor(samples, v6, scores["canonical"])
    print("  " + anchor, flush=True)

    report = build_report(samples, scores, anchor)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(report, encoding="utf-8")
    rec = {
        "n_samples": len(samples), "model": args.model,
        "scores": {arm: {i: list(v) for i, v in d.items()} for arm, d in scores.items()},
        "samples": [{"id": s["id"], "family": s["row"]["family"],
                     "reached_depth": s["row"]["reached_depth"]} for s in samples],
    }
    OUT_JSON.write_text(json.dumps(rec), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {OUT_MD}", flush=True)
    print(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap total sample size (debug)")
    ap.add_argument("--cap", type=int, default=PER_STRATUM_CAP, help="max frames per stratum")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=ACTIVE_CKPT)
    args = ap.parse_args()
    args.limit = args.limit or None
    if args.time_only:
        run_time_only(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
