"""viz_render, retargeted from whq3_000 onto 3 emit_v1 winners, across every batch.

Reuses the `viz_render` sheet harness **verbatim** — tile size, preview
resolution/supersample, downsample filter, `densify_palette` call, layout
constants, and the dump-ONCE / recolor-many cached-field pattern (`viz_render.recolor`
is imported and called unchanged). Behavioural changes vs viz_render:

  1. Location set: instead of the single fixed whq3_000, pick **3 emit_v1 winners**
     (seeded RNG, default seed 0). The winners are emit_v1's selected picks
     (`emit_v1.build_and_select`) — the same set its manifest.jsonl records; each
     carries its own coords / render block, which is all we borrow. A **diversity
     pass** drops near-duplicate winners (same family + adjacent c-plane center) so
     the three sheets show visually distinct fractals, topping up from the pool.
  2. Palette set: sweep **every** `results/*.json` batch (viz_render's own input),
     not just fire-ice. Each (batch × winner) pair emits its own sheet, filename
     **prefixed with the batch stem** (`<batch>_winner<i>.png`) so the cross product
     never collides.
  3. Cycle set: sweep **n_cycles in {1, 2, 3, 4}** (viz_render does {2, 3, 4}).

The smooth field is a pure function of (location, geometry, maxiter) and is invariant
to both palette and n_cycles, so it is dumped **once per winner** up front and reused
across every batch × palette × cycle — never re-dumped per batch (tracked by a dump
counter that must stay at N_WINNERS).

Usage:
    uv run python -u tools/palettes/viz_render_winners.py
    uv run python -u tools/palettes/viz_render_winners.py --seed 0 --filter lanczos3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))

import viz_render as vr                                # noqa: E402  reuse constants + recolor VERBATIM
import colormap as cm                                 # noqa: E402  field⊗colormap seam
import location as loc_mod                            # noqa: E402  render_one_flags / from_render_block / key
from densify_authored import densify_palette          # noqa: E402  OKLCH densifier (W=0.08 default)
from preview_render import font, lut_strip            # noqa: E402  shared label helpers

# ---- the changes ----------------------------------------------------------- #
CYCLES = (1, 2, 3, 4)                                  # change 3: add the single-pass column
RESULTS_DIR = ROOT / "dramatic_palettes" / "results"  # change 2: sweep every batch
N_WINNERS = 3                                          # change 1: 3 emit_v1 winners
DIVERSITY_TOL = 0.05                                   # c-plane center dist below which same-family winners are dupes

VIZ_DIR = ROOT / "dramatic_palettes" / "viz_render_winners"
FIELDS_DIR = ROOT / "out" / "palette_viz_render" / "fields"   # shared disposable field cache

# field spec reused from viz_render (same res / ss / mode → same cache contract).
PREVIEW_W, PREVIEW_H, PREVIEW_SS, RENDER_MODE = vr.PREVIEW_W, vr.PREVIEW_H, vr.PREVIEW_SS, vr.RENDER_MODE


def ensure_field(loc, dump_counter: list) -> cm.FieldData:
    """Dump (or reuse) the smooth field for `loc` at preview resolution — ONE dump per
    winner, keyed on (location, resolution, render-mode). Generalised over family via
    `render_one_flags` (viz_render only handled the mandelbrot whq3 location). Appends
    to `dump_counter` on a cache MISS so the caller can assert n_cycles never re-dumps."""
    FIELDS_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(loc.key().encode()).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{PREVIEW_W}x{PREVIEW_H}ss{PREVIEW_SS}_{RENDER_MODE}"
    bin_path = FIELDS_DIR / f"{stem}.bin"
    json_path = FIELDS_DIR / f"{stem}.json"
    if not (bin_path.exists() and json_path.exists()):
        print(f"  field cache MISS -> dumping {stem} (render-one --dump-field, smooth) ...")
        cmd = [
            str(vr.EXE), "render-one",
            "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
            "--maxiter", str(loc.maxiter),
            "--width", str(PREVIEW_W), "--height", str(PREVIEW_H),
            "--supersample", str(PREVIEW_SS),
            "--dump-field", str(bin_path),
        ]
        cmd += loc_mod.render_one_flags(loc)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"dump-field failed for {stem}:\n{r.stderr[-800:]}")
        dump_counter.append(stem)
        print(f"  field dumped in {time.time()-t0:.1f}s -> {bin_path.relative_to(ROOT)}")
    else:
        print(f"  field cache HIT  {stem}")
    return cm.load_field(str(bin_path), str(json_path))


def sheet_for_winner(palettes: list[dict], field: cm.FieldData, prep: cm.StretchedField,
                     out: Path, filt: str, title: str) -> None:
    """viz_render.sheet_for_batch, re-parameterised on `title` + the wider CYCLES. Same
    layout constants, same `densify_palette`/`build_lut`/`vr.recolor` calls verbatim."""
    n = len(palettes)
    W = vr.PAD + vr.LABEL_W + len(CYCLES) * (vr.CELL_W + vr.PAD) + vr.PAD
    H = vr.HEADER_H + n * (vr.CELL_H + vr.ROW_PAD) + vr.PAD
    img = Image.new("RGB", (W, H), vr.BG)
    draw = ImageDraw.Draw(img)
    f_title, f_name, f_meta, f_hdr = font(19), font(16), font(13), font(15)

    draw.text((vr.PAD, 9), title, fill=(240, 240, 240), font=f_title)
    for ci, cyc in enumerate(CYCLES):
        cx = vr.PAD + vr.LABEL_W + ci * (vr.CELL_W + vr.PAD) + vr.CELL_W // 2
        label = f"n_cycles = {cyc}"
        w = draw.textlength(label, font=f_hdr)
        draw.text((cx - w / 2, 11), label, fill=(210, 210, 210), font=f_hdr)

    for k, p in enumerate(palettes):
        dense = densify_palette(p.get("stops", []))               # W=0.08 default
        lut = cm.build_lut(dense, reverse=False, mirror=False)    # baked ONCE, reused across cycles
        y0 = vr.HEADER_H + k * (vr.CELL_H + vr.ROW_PAD)
        for ci, cyc in enumerate(CYCLES):
            rgb = vr.recolor(field, prep, lut, cyc, filt)         # reused verbatim (cycle-agnostic)
            cell = Image.fromarray(rgb).resize((vr.CELL_W, vr.CELL_H), Image.LANCZOS)
            x = vr.PAD + vr.LABEL_W + ci * (vr.CELL_W + vr.PAD)
            img.paste(cell, (x, y0))
        draw.text((vr.PAD, y0 + 2), p.get("name", "?"), fill=(240, 240, 240), font=f_name)
        draw.text((vr.PAD, y0 + 24), f"skeleton: {p.get('skeleton', '?')}",
                  fill=(185, 185, 190), font=f_meta)
        vk = p.get("value_key", p.get("axes", {}).get("value_key", "?"))
        cxk = p.get("complexity", p.get("axes", {}).get("complexity", "?"))
        draw.text((vr.PAD, y0 + 42), f"value: {vk}   ·   cx: {cxk}", fill=(150, 150, 155), font=f_meta)
        img.paste(lut_strip(dense, vr.LABEL_W - vr.PAD, vr.LUT_STRIP_H), (vr.PAD, y0 + vr.CELL_H - vr.LUT_STRIP_H))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _loc_key(c):
    """(family, cx, cy) for a winner candidate — the diversity-filter coordinate."""
    l = loc_mod.from_render_block(c.meta["row"]["render"])
    return l.family, float(l.cx), float(l.cy)


def _too_similar(a, b) -> bool:
    """Near-duplicate winners: same family AND c-plane centers within DIVERSITY_TOL."""
    (fa, ax, ay), (fb, bx, by) = a, b
    return fa == fb and ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < DIVERSITY_TOL


def pick_winners(seed: int, n: int):
    """emit_v1's selected picks == its manifest winners. Seeded random n-subset, then a
    diversity pass drops near-duplicate locations (same family + adjacent center) so the
    sheets show visually distinct fractals — topping up from the rest of the seeded pool.
    The first non-duplicate picks are preserved, so only colliding winners get swapped."""
    import emit_v1
    picks, _res, _n_pass = emit_v1.build_and_select(emit_v1.POOL_DEFAULT, emit_v1.GATE_THRESHOLD)
    if len(picks) < n:
        raise RuntimeError(f"only {len(picks)} winners available, need {n}")
    rng = random.Random(seed)
    base = rng.sample(picks, n)                          # original seeded pick (winner0/1 preserved)
    chosen, keys = [], []
    for c in base:                                       # keep first occurrence, drop near-dups
        k = _loc_key(c)
        if any(_too_similar(k, kk) for kk in keys):
            continue
        chosen.append(c); keys.append(k)
    if len(chosen) < n:                                  # top up with distinct pool locations
        seen = {c.location_id for c in base}
        for c in rng.sample(picks, len(picks)):
            if c.location_id in seen:
                continue
            k = _loc_key(c)
            if any(_too_similar(k, kk) for kk in keys):
                continue
            chosen.append(c); keys.append(k); seen.add(c.location_id)
            if len(chosen) == n:
                break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for winner selection")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR, help="dir of palette batch JSONs")
    ap.add_argument("--viz", type=Path, default=VIZ_DIR)
    ap.add_argument("--filter", default="box", choices=["box", "mitchell", "lanczos3"],
                    help="downsample AA filter (default box; lanczos3 = production emit)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    srcs = sorted(args.results.glob("*.json"))
    if not srcs:
        print(f"no results files under {args.results}")
        return
    print(f"batches: {len(srcs)} under {args.results.name}  ·  cycles {CYCLES}  ·  filter {args.filter}\n")

    winners = pick_winners(args.seed, N_WINNERS)
    print(f"\n=== winner selection ===\nseed = {args.seed}  (diversity tol = {DIVERSITY_TOL})")
    for i, c in enumerate(winners):
        print(f"  winner[{i}]  key = {c.location_id}")
        print(f"             image_id={c.image_id}  family={c.family}  p_ge3={c.meta['p_ge3']:.3f}")
    print()

    # Dump each winner's field ONCE up front, reused across every batch × palette × cycle.
    dump_counter: list = []
    wf = []
    for i, c in enumerate(winners):
        loc = loc_mod.from_render_block(c.meta["row"]["render"])
        field = ensure_field(loc, dump_counter)
        prep = cm.stretch_field(field)                            # one percentile sort per winner
        wf.append((loc, field, prep, c))
    assert len(dump_counter) <= N_WINNERS, "more field dumps than winners — cache key drift"

    built = 0
    for src in srcs:
        palettes = json.loads(src.read_text())
        for i, (loc, field, prep, c) in enumerate(wf):
            stem = f"{src.stem}_winner{i}"                        # batch-prefixed → unique per (batch, winner)
            out = args.viz / f"{stem}.png"
            title = (f"{stem}   ·   {loc.family} {c.image_id}   ·   {args.filter}"
                     f"   ·   n_cycles {'/'.join(map(str, CYCLES))}")
            sheet_for_winner(palettes, field, prep, out, args.filter, title)
            built += 1
        print(f"  BUILT {src.stem}: {len(wf)} winner sheet(s) x {len(palettes)} palettes x {len(CYCLES)} cycles")

    print(f"\ndone: {built} sheet(s) -> {args.viz.relative_to(ROOT)}  "
          f"({len(dump_counter)} field dump(s) total)")


if __name__ == "__main__":
    main()
