"""Fair re-render of the 30 q4 stage-1 minibrots + controls in the TARGET vivid
style (default/cubehelix/viridis 3-up), 4x-size framing (each row's stored fw is
already 4x size), judge-quality res (tile 1024, 16:9, ss2). Sequential; each
render uses all cores via rayon, so we do NOT fan out processes.

mb19_p35's sheet is rendered with the exact ladder params, so it doubles as the
identity reproduction of out/deep_centers/ladder_p35/fw_8p07e_10.png.
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "target" / "release" / "fractal-generator.exe"
MINIS = json.load(open(ROOT / "out" / "q4_stage1" / "minibrots.json"))
POOL = [json.loads(l) for l in
        open(ROOT / "out" / "deep_centers" / "pool.jsonl")]
OUT = ROOT / "out" / "fair_rerender" / "sheets"
OUT.mkdir(parents=True, exist_ok=True)

COMMON = ["--builtins", "default cubehelix viridis",
          "--tile-width", "1024", "--aspect", "16:9",
          "--supersample", "2", "--backend", "auto"]


def render(rid, cx, cy, fw, maxiter, out):
    cmd = [str(EXE), "sheet",
           "--center-re", str(cx), "--center-im", str(cy),
           "--frame-width", str(fw), "--maxiter", str(maxiter),
           *COMMON, "--output", str(out)]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    ok = r.returncode == 0 and out.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {rid:<12} fw={fw:<14} {dt:6.1f}s", flush=True)
    if not ok:
        print(r.stderr[-500:], flush=True)
    return ok, dt


def main():
    jobs = [(m["id"], m["cx"], m["cy"], m["fw"], m["maxiter"],
             OUT / f"{m['id']}.png") for m in MINIS]
    # preview_p58 control (pool line 2 = p58 nucleus, money-shot fw)
    p58 = next(p for p in POOL if p["period"] == 58 and p["kind"] == "nucleus")
    jobs.append(("preview_p58", p58["cx"], p58["cy"], p58["fw_suggest"],
                 p58["render_maxiter"], OUT / "preview_p58.png"))
    # ladder_p35 == mb19 (byte-identical); mb19 sheet is its reproduction. No
    # separate render needed, but emit an identity copy for the side-by-side step.

    print(f"{len(jobs)} renders -> {OUT}", flush=True)
    t0 = time.time()
    n_ok = 0
    for rid, cx, cy, fw, mi, out in jobs:
        ok, _ = render(rid, cx, cy, fw, mi, out)
        n_ok += ok
    print(f"\nDONE {n_ok}/{len(jobs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
