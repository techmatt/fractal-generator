#!/usr/bin/env python
r"""Cross-family shakeout — timings + visual sanity on the fast (guarded) reward path.

Observation run, NOT discovery. Now that the guard field is sourced from the fast
F64Backend smooth channel (~20-45x faster), run a small shakeout across the
parameter-plane descendable families (mandelbrot + multibrot3/4/5) to (a) measure
real per-walk wallclock on the fast path and (b) eyeball that each family's descents
land on real boundary detail. (Fixed-anchor Julia was dropped — the Julia discovery
hook now descends the z-plane per outcome c, superseding a single hardcoded anchor.)
NO
density-rejection / atlas apparatus -- just ~4 seed-varied walks per family through
the normal guarded reward pipeline:

    guided-descend  ->  raw-score (v5)  ->  reframe top-3 (v5)  ->  guard  ->  k3

Reuse (located, not reinvented):
  * guided-descend engine (Rust) w/ --family / --julia --c        prescreen.BIN
  * per-walk frame loader                                         step0_reanalysis.load_frames_by_walk
  * guarded raw-screen (family passthrough: loc_of)               step0_reanalysis.raw_screen_walk
  * reframe path (family-agnostic via render_one_flags)           reframe.reframe_location
  * degenerate-outcome guard                                      tools/atlas/guard.py
  * v5 CORN scorer bridge                                         guard.make_guarded_scorer

The ONLY code change in scope is the `loc_of` passthrough added to
`raw_screen_walk` (default `_mand_location`, byte-identical for every existing
Mandelbrot caller); reframe was already family-agnostic. This driver owns its own
family-aware k3 reward loop so it needs no change to production_seeder.

  uv run python tools/atlas/cross_family_shakeout.py            # full shakeout (BACKGROUND)
  uv run python tools/atlas/cross_family_shakeout.py --walks 2  # quicker smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "atlas_probe"))
sys.path.insert(0, str(ROOT / "tools" / "reframe"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

try:
    # line_buffering=True so progress lands in a redirected log immediately (default
    # block buffering hides all output until flush/exit when stdout is not a tty).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import prescreen  # noqa: E402  (BIN)
import reframe  # noqa: E402  (reframe_location + DUMP_GUARD_FIELD hook + Location + _tile_name)
from reframe import reframe_location, Location, _tile_name, RENDER_W, RENDER_H, RENDER_SS  # noqa: E402
import step0_reanalysis as sr  # noqa: E402  (module-global SCRATCH holds the raw tiles)
from step0_reanalysis import load_frames_by_walk, raw_screen_walk, KRAW  # noqa: E402
from active_ckpt import auto_maxiter, PALETTE, JPG_Q, ACTIVE_CKPT  # noqa: E402
import location as loc_mod  # noqa: E402  (render_one_flags for the microbench)
import guard  # noqa: E402

# --------------------------------------------------------------------------- #
# Config -- the production walk config (matches production_seeder so per-walk wall
# reflects the real run), but only a handful of seed-varied walks per family.
# --------------------------------------------------------------------------- #
NODE_WIDTH = 384
SIGMA_BAND = "8,10,12,14,16"
DEPTH_MIN = 4
DEPTH_MAX = 14
OCC_FLOOR = 0.321
BLACK_CAP = 0.30
WORKERS = 4                # project rule: max 4
SCORER_PATH = ACTIVE_CKPT  # single source of truth (active_ckpt.ACTIVE_CKPT — currently v7)

FAMILIES = [
    # (key, guided-descend extra flags, reframe-Location family, (c_re,c_im)|None)
    # Fixed-anchor Julia was removed: the single hardcoded JULIA_C anchor (z-crop
    # scatter) is superseded by the Julia discovery hook, which descends the z-plane
    # per parameter-plane outcome c. The remaining parameter-plane families
    # (mandelbrot, multibrot3/4/5) stay valid observation/timing paths.
    ("mandelbrot", [], "mandelbrot", None),
    ("multibrot3", ["--family", "multibrot3"], "multibrot3", None),
    ("multibrot4", ["--family", "multibrot4"], "multibrot4", None),
    ("multibrot5", ["--family", "multibrot5"], "multibrot5", None),
]

OUT_ROOT = ROOT / "out" / "atlas" / "cross_family_shakeout"


def make_loc_of(family: str, c):
    """Per-family reframe.Location factory for raw-screen + reframe."""
    c_re, c_im = (c if c is not None else (None, None))

    def loc_of(cx, cy, fw) -> Location:
        return Location(family=family, c_re=c_re, c_im=c_im,
                        cx=str(cx), cy=str(cy), fw=str(fw), family_params={})
    return loc_of


def run_descent(key: str, flags: list, seed: int, pool_dir: Path, n_walks: int):
    """One guided-descend run (n_walks seed-varied walks) at production config. Returns
    the wall seconds for the descent subprocess."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(prescreen.BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool_dir),
    ] + flags
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    if r.returncode != 0:
        raise SystemExit(f"[{key}] guided-descend failed:\n{r.stderr[-2000:]}")
    return dt


def k3_reward_walk(scorer, wid, frames, loc_of, scratch: Path):
    """Family-aware k3 best-over-walk reward for one walk (the guarded reward loop).

    raw-score every frame (guarded), reframe the top-3 by raw score (guarded), and
    take reward_k3 = max reframed. Returns a dict with the k3 score, the guard verdict,
    the reframed outcome geometry, and the chosen reframed tile path (for the sheet)."""
    raws = raw_screen_walk(scorer, wid, frames, WORKERS, loc_of=loc_of)
    order = sorted(range(len(frames)), key=lambda i: raws[i], reverse=True)
    topk = order[:KRAW]

    best = None  # (score, res, tile_path, frame)
    for rank, i in enumerate(topk):
        fr = frames[i]
        loc = loc_of(fr["cx"], fr["cy"], fr["fw"])
        wd = scratch / f"walk_{wid:04d}" / f"reframe_top{rank}"
        res = reframe_location(loc, scorer=scorer, seed=0, workdir=wd, workers=WORKERS)
        if res.score < res.trace["original_score"] - 1e-4:
            print(f"    WARN monotonicity walk {wid} idx {fr['idx']}: "
                  f"{res.score:.4f} < {res.trace['original_score']:.4f}")
        tile = wd / "tiles" / _tile_name(res.trace["chosen"])
        if best is None or res.score > best[0]:
            best = (float(res.score), res, tile, fr)

    reward_k3, res, tile, fr = best
    guard_pass = reward_k3 > guard.GUARD_SENTINEL + 1e-6
    reached = max(int(f["depth"]) for f in frames)
    return {
        "walk_id": int(wid), "n_frames": len(frames), "reached_depth": reached,
        "reward_k3": reward_k3, "guard_pass": bool(guard_pass),
        "raw_max": float(max(raws)), "raw_mean": float(sum(raws) / len(raws)),
        "outcome_cx": res.cx, "outcome_cy": res.cy, "outcome_fw": res.fw,
        "k3_frame_depth": int(fr["depth"]),
        "chosen_tile": str(tile),
    }


def guard_field_microbench(loc_of, cx, cy, fw) -> dict:
    """Time the two render-one calls a single scored tile pays: the beautiful JPG
    (the classifier's view) and the f64 guard-field dump, at the reframe/deploy
    fidelity. Confirms the guard field is now a small fraction of one tile's render
    wall (it was ~85-93% before the f64-source fix)."""
    loc = loc_of(cx, cy, fw)
    mi = auto_maxiter(float(fw))
    flags = loc_mod.render_one_flags(loc)
    # Same guard-field source reframe uses: f64 for every escape-time family
    # (mandelbrot/julia + multibrot, now that the fast f64 smooth channel is
    # degree-parametric).
    fsrc = "f64"
    scratch = OUT_ROOT / "_microbench"
    scratch.mkdir(parents=True, exist_ok=True)
    jpg = scratch / "bench.jpg"
    fld = scratch / "bench.field.bin"

    jpg_cmd = [str(prescreen.BIN), "render-one", "--cx", str(cx), "--cy", str(cy),
               "--fw", repr(float(fw)), "--width", str(RENDER_W), "--height", str(RENDER_H),
               "--supersample", str(RENDER_SS), "--maxiter", str(mi),
               "--palette", PALETTE, "--jpg-quality", str(JPG_Q), "--out", str(jpg)] + flags
    fld_cmd = [str(prescreen.BIN), "render-one", "--cx", str(cx), "--cy", str(cy),
               "--fw", repr(float(fw)), "--width", str(RENDER_W), "--height", str(RENDER_H),
               "--supersample", str(RENDER_SS), "--maxiter", str(mi),
               "--dump-field", str(fld), "--dump-field-source", fsrc] + flags

    def timed(cmd):
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        return time.time() - t, r.returncode == 0

    jpg_s, jok = timed(jpg_cmd)
    fld_s, fok = timed(fld_cmd)
    return {"jpg_s": jpg_s, "field_s": fld_s, "ok": jok and fok, "field_source": fsrc,
            "field_frac_of_tile": fld_s / max(1e-9, jpg_s + fld_s)}


def build_family_sheet(key: str, results: list[dict], out_png: Path, descent_s: float):
    """One row of walk outcome crops (the k3-winner reframed frames) for eyeballing."""
    from PIL import Image, ImageDraw
    TW, TH, PAD, LBL, GUT = 384, 216, 8, 46, 30
    items = [r for r in results if r.get("chosen_tile")]
    ncol = max(1, len(items))
    cell_w, cell_h = TW + 2 * PAD, TH + LBL + 2 * PAD
    W, H = ncol * cell_w, GUT + cell_h
    sheet = Image.new("RGB", (W, max(H, GUT + cell_h)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), f"{key} -- {len(items)} walk outcomes (k3-winner reframed crop, "
                      f"640x360 search fidelity)   descent {descent_s:.1f}s/{len(results)} walks",
              fill=(235, 235, 235))
    for k, r in enumerate(items):
        x, y = k * cell_w + PAD, GUT + PAD
        tp = Path(r["chosen_tile"])
        if tp.exists():
            im = Image.open(tp).convert("RGB").resize((TW, TH))
            sheet.paste(im, (x, y))
        col = (90, 230, 110) if r["guard_pass"] else (235, 90, 90)
        for t in range(2):
            draw.rectangle([x - 1 - t, y - 1 - t, x + TW + t, y + TH + t], outline=col)
        draw.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        tag = "" if r["guard_pass"] else "  GUARDED"
        draw.text((x + 4, y + TH + 3),
                  f"walk {r['walk_id']} reached d{r['reached_depth']} (k3 frame d{r['k3_frame_depth']})",
                  fill=(210, 210, 218))
        draw.text((x + 4, y + TH + 24),
                  f"k3={r['reward_k3']:.3f}  raw_max={r['raw_max']:.3f}{tag}",
                  fill=col)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walks", type=int, default=4, help="seed-varied walks per family")
    ap.add_argument("--seed", type=int, default=7, help="base engine seed (offset per family)")
    ap.add_argument("--only", default=None, help="comma-separated family keys to run (default all)")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    fams = [f for f in FAMILIES if only is None or f[0] in only]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Guarded v5 scorer -- raw-frame scoring AND reframe candidate scoring inherit the
    # model-free field guard; wire the field-dump hook so every tile drops its sidecar.
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX, "guard field suffix drift"
    reframe.DUMP_GUARD_FIELD = True
    scorer = guard.make_guarded_scorer(SCORER_PATH)
    print(f"=== cross-family shakeout ({args.walks} walks/family) ===")
    print(f"scorer: GUARDED CORN ({SCORER_PATH})  geometry={scorer.cfg.get('geometry')}")
    print(f"walk cfg: node={NODE_WIDTH} sigma={SIGMA_BAND} depth[{DEPTH_MIN},{DEPTH_MAX}] "
          f"occ={OCC_FLOOR} black={BLACK_CAP} workers={WORKERS}")
    print(f"guard: interior_frac>={guard.INTERIOR_CAP} | field_std<{guard.FIELD_STD_FLOOR} "
          f"@ {guard.GUARD_STAT_RES}\n")

    summary = []
    t_all = time.time()
    for fi, (key, flags, loc_family, c) in enumerate(fams):
        seed = args.seed + 1000 * fi
        print(f"--- {key} (seed {seed}) ---")
        try:
            summary.append(run_family(scorer, key, flags, loc_family, c, seed, args.walks))
        except Exception as e:
            print(f"  !! {key} FAILED: {e}\n")
            summary.append(dict(family=key, error=str(e), n_walks=0, results=[]))

    _timing_table(summary)
    OUT_ROOT.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    dt = time.time() - t_all
    print(f"\nDONE in {dt:.0f}s ({dt/60:.1f} min)  ->  {OUT_ROOT}")


def run_family(scorer, key, flags, loc_family, c, seed, n_walks) -> dict:
    """Descend + guarded-reward one family; returns its summary dict. Raises on a hard
    engine failure (caught per-family by main so the other families still report)."""
    loc_of = make_loc_of(loc_family, c)
    fam_dir = OUT_ROOT / key
    pool_dir = fam_dir / "pool"

    descent_s = run_descent(key, flags, seed, pool_dir, n_walks)
    by_walk = load_frames_by_walk(pool_dir)
    n_walks = len(by_walk)
    n_frames = sum(len(v) for v in by_walk.values())
    print(f"  descent {descent_s:.1f}s -> {n_walks} walks / {n_frames} frames")
    if n_walks == 0:
        print("  (no walks with frames; skipping reward)")
        return dict(family=key, descent_s=descent_s, n_walks=0, results=[])

    scratch = fam_dir / "_reward"
    # raw_screen_walk keys raw tiles by walk/idx under the module-global SCRATCH and
    # reuses on-disk hits -- point it at a PER-FAMILY dir so mandelbrot walk0/idx0
    # never gets reused for julia walk0/idx0 (different fractal, same key).
    sr.SCRATCH = fam_dir / "_raw"
    results = []
    t_rew = time.time()
    for wid in sorted(by_walk):
        tw = time.time()
        row = k3_reward_walk(scorer, wid, by_walk[wid], loc_of, scratch)
        row["reward_s"] = time.time() - tw
        results.append(row)
        print(f"    walk {wid:>3} nframes={row['n_frames']:>2} reached d{row['reached_depth']:>2}: "
              f"reward {row['reward_s']:.1f}s  k3={row['reward_k3']:.3f} "
              f"{'PASS' if row['guard_pass'] else 'GUARDED'}")
    reward_total_s = time.time() - t_rew

    # guard-field micro-bench on the best outcome geometry (single tile's two renders).
    best = max(results, key=lambda r: r["reward_k3"])
    mb = guard_field_microbench(loc_of, best["outcome_cx"], best["outcome_cy"], best["outcome_fw"])

    sheet = build_family_sheet(key, results, fam_dir / "outcomes.png", descent_s)
    out = dict(family=key, descent_s=descent_s, reward_total_s=reward_total_s,
               n_walks=n_walks, microbench=mb, results=results, sheet=str(sheet))
    (fam_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"  reward {reward_total_s:.1f}s ({reward_total_s/n_walks:.1f}s/walk)  "
          f"guard-field microbench ({mb['field_source']}): jpg {mb['jpg_s']:.2f}s + "
          f"field {mb['field_s']:.2f}s -> field {mb['field_frac_of_tile']*100:.0f}% of tile render wall")
    print(f"  sheet -> {sheet}\n")
    return out


def _timing_table(summary: list[dict]):
    print("\n================= TIMING TABLE =================")
    print(f"{'family':<12} {'walks':>5} {'descent/walk':>13} {'reward/walk':>12} "
          f"{'per-walk':>9} {'k3_med':>7} {'guarded':>8} {'fld_src':>10} {'fld%tile':>9}")
    for s in summary:
        if s.get("n_walks", 0) == 0 or not s.get("results"):
            tag = f"ERROR: {s['error'][:40]}" if s.get("error") else "(no walks)"
            print(f"{s['family']:<12} {'0':>5}  {tag}")
            continue
        res = s["results"]
        dpw = s["descent_s"] / s["n_walks"]
        rpw = s["reward_total_s"] / s["n_walks"]
        ks = sorted(r["reward_k3"] for r in res)
        kmed = ks[len(ks) // 2]
        ngu = sum(1 for r in res if not r["guard_pass"])
        mb = s["microbench"]
        print(f"{s['family']:<12} {s['n_walks']:>5} {dpw:>12.1f}s {rpw:>11.1f}s "
              f"{dpw+rpw:>8.1f}s {kmed:>7.3f} {ngu:>4}/{len(res):<3} {mb['field_source']:>10} "
              f"{mb['field_frac_of_tile']*100:>8.0f}%")
    print("================================================")
    print("note: reward/walk = raw-score every frame + reframe top-3 (all guarded); "
          "fld%tile = guard-field render as % of one tile's two-render wall (was ~85-93%).")
    print("      fld_src=f64 for every escape-time family (mandelbrot + multibrot).")


if __name__ == "__main__":
    main()
