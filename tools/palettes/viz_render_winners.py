"""viz_render, retargeted from whq3_000 onto 3 random emit_v1 winners.

Reuses the `viz_render` sheet harness **verbatim** — tile size, preview
resolution/supersample, downsample filter, `densify_palette` call, layout
constants, and the dump-ONCE / recolor-many cached-field pattern (`viz_render.recolor`
is imported and called unchanged). Exactly two behavioural changes vs viz_render:

  1. Location set: instead of the single fixed whq3_000, pick **3 emit_v1 winners at
     random** (seeded RNG, default seed 0) and emit **one sheet file per winner**.
     The winners are emit_v1's selected picks (`emit_v1.build_and_select`) — the same
     set its manifest.jsonl records; each carries its own coords / render block, which
     is all we borrow (the palettes tested are the fire-ice batch, not the winner's).
  2. Cycle set: sweep **n_cycles in {1, 2, 3, 4}** (viz_render does {2, 3, 4}).

The smooth field is a pure function of (location, geometry, maxiter) and is invariant
to both palette and n_cycles, so it is dumped **once per winner** and reused across
all 20 palettes x 4 cycles — the n_cycles=1 addition is just one more `recolor` call
over the already-cached field, never a re-dump (asserted below via a dump counter).

Palettes: dramatic_palettes/results/fire-ice_c3-4_v3_1.json (whole batch), through the
same densify path viz_render uses.

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

# ---- the two changes ------------------------------------------------------- #
CYCLES = (1, 2, 3, 4)                                  # change 2: add the single-pass column
PALETTES_JSON = ROOT / "dramatic_palettes" / "results" / "fire-ice_c3-4_v3_1.json"
N_WINNERS = 3                                          # change 1: 3 emit_v1 winners

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


def pick_winners(seed: int, n: int):
    """emit_v1's selected picks == its manifest winners. Seeded random n-subset."""
    import emit_v1
    picks, _res, _n_pass = emit_v1.build_and_select(emit_v1.POOL_DEFAULT, emit_v1.GATE_THRESHOLD)
    if len(picks) < n:
        raise RuntimeError(f"only {len(picks)} winners available, need {n}")
    chosen = random.Random(seed).sample(picks, n)
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for winner selection")
    ap.add_argument("--palettes", type=Path, default=PALETTES_JSON)
    ap.add_argument("--viz", type=Path, default=VIZ_DIR)
    ap.add_argument("--filter", default="box", choices=["box", "mitchell", "lanczos3"],
                    help="downsample AA filter (default box; lanczos3 = production emit)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    palettes = json.loads(args.palettes.read_text())
    print(f"palettes: {args.palettes.name} ({len(palettes)} palettes)  ·  cycles {CYCLES}  ·  filter {args.filter}\n")

    winners = pick_winners(args.seed, N_WINNERS)
    print(f"\n=== winner selection ===\nseed = {args.seed}")
    for i, c in enumerate(winners):
        print(f"  winner[{i}]  key = {c.location_id}")
        print(f"             image_id={c.image_id}  family={c.family}  p_ge3={c.meta['p_ge3']:.3f}")
    print()

    dump_counter: list = []
    for i, c in enumerate(winners):
        loc = loc_mod.from_render_block(c.meta["row"]["render"])
        n_before = len(dump_counter)
        field = ensure_field(loc, dump_counter)
        prep = cm.stretch_field(field)                            # one percentile sort, reused across all recolors
        # confirm the n_cycles=1 addition did not force a second dump for this winner.
        assert len(dump_counter) - n_before <= 1, "winner triggered >1 field dump"
        stem = f"winner{i}_{loc.family}_{c.image_id}"
        out = args.viz / f"{stem}.png"
        title = f"{stem}   ·   {loc.family}   ·   {args.filter}   ·   n_cycles {'/'.join(map(str, CYCLES))}"
        sheet_for_winner(palettes, field, prep, out, args.filter, title)
        print(f"  BUILT {out.relative_to(ROOT)}  ({len(palettes)} palettes x {len(CYCLES)} cycles, 1 field dump)")

    print(f"\ndone: {len(winners)} winner sheet(s) -> {args.viz.relative_to(ROOT)}  "
          f"({len(dump_counter)} field dump(s) total)")


if __name__ == "__main__":
    main()
