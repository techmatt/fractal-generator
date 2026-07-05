#!/usr/bin/env python
"""Look-see: 27 walks across all 9 fractal families (a health check, not a harness).

Runs the NORMAL sampler process to produce 3 walks per family for the 9 families
(4 degrees x {param-plane, Julia} + Phoenix), renders one outcome frame per walk at
768x432 to eyeball quality, and reports per-family wall-time.

Ungated: no density rejection, no depth-2 probe filtering -- just draw and descend,
at production walk depth (DEPTH_MIN..DEPTH_MAX). No metrics, no scorer, no analysis
machinery. The outcome frame of a walk is its deepest descended node.

Reuses production_seeder's thin engine wrappers verbatim (generate_native_seeds,
run_full_walks, run_julia_descent) + the render-one wallpaper path (quick-render
overrides). Touches nothing in the discovery pipeline.
"""
import sys
import json
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("tools/atlas", "tools/atlas_probe", "tools/reframe", "tools/corpus"):
    sys.path.insert(0, str(ROOT / p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import prescreen  # noqa: E402  (BIN)
from production_seeder import (  # noqa: E402  (thin engine wrappers + walk config)
    generate_native_seeds, run_full_walks, run_julia_descent,
    NODE_WIDTH, SIGMA_BAND, OCC_FLOOR, BLACK_CAP, DEPTH_MIN, DEPTH_MAX,
)

BIN = prescreen.BIN
OUT = ROOT / "out" / "looksee"
SCRATCH = OUT / "scratch"
FRAMES = OUT / "frames"

DEGREES = (2, 3, 4, 5)
N_NATIVE_DRAW = 16   # native depth-1 draws per degree; take the first 3 as the shared seeds
RENDER_W, RENDER_H, RENDER_SS = 768, 432, 2
PALETTE = "twilight"


def cfamily(d: int) -> str:
    return "mandelbrot" if d == 2 else f"multibrot{d}"


def run_phoenix_descent(seed: int, workdir: Path, n_walks: int) -> Path:
    """Native Phoenix z-plane descent (no parameter plane -> seed-varied native draw),
    mirroring run_julia_descent but with --phoenix (default c/p). Production depth."""
    workdir.mkdir(parents=True, exist_ok=True)
    pool = workdir / "pool"
    cmd = [
        str(BIN), "guided-descend",
        "--n-walks", str(n_walks), "--seed", str(seed), "--per-walk-rng",
        "--depth-min", str(DEPTH_MIN), "--depth-max", str(DEPTH_MAX),
        "--node-width", str(NODE_WIDTH), "--sigma-band", SIGMA_BAND,
        "--descent-occ-floor", str(OCC_FLOOR), "--descent-black-cap", str(BLACK_CAP),
        "--preview-width", "48", "--cols", "40",
        "--phoenix", "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"phoenix descent failed:\n{r.stderr[-2000:]}")
    return pool


def outcomes_of(pool_dir: Path) -> list[dict]:
    """Per walk, the deepest descended node (the walk outcome). Returns rows sorted by
    walk id; each row carries cx/cy/fw."""
    by_walk: dict[int, list[dict]] = {}
    for line in open(pool_dir / "pool.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by_walk.setdefault(r["walk"], []).append(r)
    out = []
    for w in sorted(by_walk):
        rows = sorted(by_walk[w], key=lambda r: r["depth"])
        out.append(rows[-1])
    return out


def render_frame(cx, cy, fw, family: str, julia_c, out_png: Path):
    """Quick 768x432 ss2 render of one outcome location (smooth mode, twilight)."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BIN), "render-one",
        "--cx", repr(float(cx)), "--cy", repr(float(cy)), "--fw", repr(float(fw)),
        "--family", family,
        "--width", str(RENDER_W), "--height", str(RENDER_H), "--supersample", str(RENDER_SS),
        "--palette", PALETTE, "--out", str(out_png),
    ]
    if julia_c is not None:
        cmd += ["--julia", "--c", repr(float(julia_c[0])), repr(float(julia_c[1]))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"render failed ({out_png.name}):\n{r.stderr[-1500:]}")


def montage(rows: list[tuple[str, list[Path]]], out_png: Path):
    """Contact sheet: one family per row, 3 frames across, with a family label strip."""
    from PIL import Image, ImageDraw
    tw, th = RENDER_W // 2, RENDER_H // 2   # 384 x 216 thumbnails
    lab = 22                                # label strip height
    pad = 6
    cols = 3
    W = cols * tw + (cols + 1) * pad
    H = len(rows) * (th + lab) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    y = pad
    for label, frames in rows:
        draw.text((pad, y + 4), label, fill=(210, 210, 220))
        yy = y + lab
        for i in range(cols):
            x = pad + i * (tw + pad)
            if i < len(frames) and frames[i].exists():
                im = Image.open(frames[i]).convert("RGB").resize((tw, th))
                sheet.paste(im, (x, yy))
            else:
                draw.rectangle([x, yy, x + tw, yy + th], fill=(40, 40, 46))
        y += th + lab
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def main():
    FRAMES.mkdir(parents=True, exist_ok=True)
    timing: list[dict] = []          # per-family {family, n, total_s}
    sheet_rows: list[tuple[str, list[Path]]] = []

    for d in DEGREES:
        fam = cfamily(d)
        fam_flags = [] if d == 2 else ["--family", fam]
        base = 4000 + d * 100

        # --- 3 shared native seeds (same (cx,cy) reused for both planes at this degree) ---
        print(f"[d{d}/{fam}] drawing native seeds ...", flush=True)
        seeds = generate_native_seeds(N_NATIVE_DRAW, base, SCRATCH / f"native_d{d}", fam_flags)
        seeds = seeds[:3]
        if len(seeds) < 3:
            raise SystemExit(f"d{d}: native draw yielded only {len(seeds)} seeds")

        # --- parameter-plane walks: normal seed-list descent (one engine run, 3 walks) ---
        survivors = [{"seed_cx": s["cx"], "seed_cy": s["cy"], "fw": s["fw"]} for s in seeds]
        print(f"[d{d}/{fam}] param-plane descend (3 walks) ...", flush=True)
        t0 = time.perf_counter()
        pool = run_full_walks(survivors, SCRATCH / f"param_d{d}", base + 1, fam_flags)
        t_param = time.perf_counter() - t0
        pframes = []
        for i, row in enumerate(outcomes_of(pool)):
            fp = FRAMES / f"{fam}_param_{i}.png"
            render_frame(row["cx"], row["cy"], row["fw"], fam, None, fp)
            pframes.append(fp)
        timing.append({"family": fam, "n": len(pframes), "total_s": t_param})
        sheet_rows.append((f"{fam}  (param plane)", pframes))

        # --- Julia-d walks: the seed's (cx,cy) IS the Julia c; normal native descent ---
        jfam = "julia" if d == 2 else f"julia_multibrot{d}"
        j_total = 0.0
        jframes = []
        for i, s in enumerate(seeds):
            c = (s["cx"], s["cy"])
            print(f"[d{d}/{jfam}] julia descend c=({c[0]:.4f},{c[1]:.4f}) ...", flush=True)
            t0 = time.perf_counter()
            jpool = run_julia_descent(c, "normal", base + 10 + i,
                                      SCRATCH / f"julia_d{d}_{i}", 1, fam)
            j_total += time.perf_counter() - t0
            jouts = outcomes_of(jpool)
            row = jouts[0]
            fp = FRAMES / f"{jfam}_{i}.png"
            render_frame(row["cx"], row["cy"], row["fw"], fam, c, fp)
            jframes.append(fp)
        timing.append({"family": jfam, "n": len(jframes), "total_s": j_total})
        sheet_rows.append((f"{jfam}  (c = degree-{d} seed)", jframes))

    # --- Phoenix: 3 native --phoenix walks (no parameter plane) ---
    print("[phoenix] descend (3 walks) ...", flush=True)
    t0 = time.perf_counter()
    ppool = run_phoenix_descent(9000, SCRATCH / "phoenix", 3)
    t_phx = time.perf_counter() - t0
    pxframes = []
    for i, row in enumerate(outcomes_of(ppool)):
        fp = FRAMES / f"phoenix_{i}.png"
        render_frame(row["cx"], row["cy"], row["fw"], "phoenix", None, fp)
        pxframes.append(fp)
    timing.append({"family": "phoenix", "n": len(pxframes), "total_s": t_phx})
    sheet_rows.append(("phoenix  (native z-plane)", pxframes))

    # --- contact sheet ---
    sheet_path = OUT / "looksee_sheet.png"
    montage(sheet_rows, sheet_path)

    # --- report ---
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print("\n" + "=" * 58)
    print("PER-FAMILY WALL-TIME  (3 walks each, production depth)")
    print("=" * 58)
    print(f"{'family':<22} {'n':>3} {'total_s':>10} {'mean_s/walk':>12}")
    print("-" * 58)
    grand = 0.0
    for t in timing:
        mean = t["total_s"] / t["n"] if t["n"] else 0.0
        grand += t["total_s"]
        print(f"{t['family']:<22} {t['n']:>3} {t['total_s']:>10.1f} {mean:>12.1f}")
    print("-" * 58)
    print(f"{'TOTAL':<22} {sum(t['n'] for t in timing):>3} {grand:>10.1f}")
    print("=" * 58)
    print(f"\nframes : {FRAMES}")
    print(f"sheet  : {sheet_path}")


if __name__ == "__main__":
    main()
