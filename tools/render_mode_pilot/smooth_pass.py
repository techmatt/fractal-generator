"""Too-close-to-smooth distance pass (smooth-distance-pass.md).

For each of the 500 render-mode-pilot rasters, render its SMOOTH counterpart at the
same location / palette / approved color params (1280x720 ss2, pure smooth field via
`render-one --dump-field` -> colormap.render_candidate, bit-faithful incl. transfer=grad),
deduped by (location_key, palette, color_params) so shared counterparts render once.
Then measure raster<->smooth distance two ways: mean CIELAB dE76 + (1 - SSIM).

    uv run python -u tools/render_mode_pilot/smooth_pass.py --render   # render + measure
    uv run python -u tools/render_mode_pilot/smooth_pass.py            # measure only (crops on disk)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))
sys.path.insert(0, str(REPO / "tools" / "queries"))

EXE = str(REPO / "target/release/fractal-generator.exe")

BATCH_ID = "2026-07-10_render_mode_pilot_v1"
BATCH_DIR = REPO / "data/render_mode_corpus/batches" / BATCH_ID
CROPS = BATCH_DIR / "crops"
IMAGES = BATCH_DIR / "images.jsonl"

OUT = REPO / "out/render_mode_pilot/smooth_pass"
SMOOTH_CROPS = OUT / "smooth_crops"
FIELDS = OUT / "_fields"
RESULTS = OUT / "distances.json"

W, H, SS, FILT, JPG_Q = 1280, 720, 2, "lanczos3", 95
CK_KEYS = ["reverse", "log_premap", "gamma", "phase", "n_cycles", "transfer", "transfer_gamma"]


def color_key(cp):
    return "|".join(f"{k}={cp.get(k)}" for k in CK_KEYS)


def dedup_key(row):
    pv = row["provenance"]
    return (pv["location_key"], row["render"]["palette"], color_key(pv["color_params"]))


def smooth_id(key):
    return "sm_" + hashlib.sha1("||".join(key).encode()).hexdigest()[:16]


# --- per-worker globals -----------------------------------------------------
_LIB = _CM = _LOC = None


def _init_worker():
    global _LIB, _CM, _LOC
    os.environ["RAYON_NUM_THREADS"] = "3"
    import colormap as cm
    import location as loc_mod
    import query_sampler as qs
    _CM, _LOC = cm, loc_mod
    _LIB = qs.load_pool_library()


def _locflags(loc):
    return _LOC.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
            "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def _run(cmd):
    env = dict(os.environ, RAYON_NUM_THREADS="3")
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def render_smooth(job):
    """job: {sid, render(block), palette, color_params}. -> smooth crop jpg."""
    cm = _CM
    crop_path = SMOOTH_CROPS / f"{job['sid']}.jpg"
    if crop_path.exists():
        return {"sid": job["sid"], "cached": True}
    loc = _LOC.from_render_block(job["render"])
    binp = FIELDS / f"{job['sid']}.bin"
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(W), "--height", str(H),
             "--supersample", str(SS), "--coloring", json.dumps({"field": "smooth"}),
             "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        p = job["color_params"]
        ptype = _LIB.palette_type(job["palette"])
        phase = p["phase"] if ptype == "cyclic" else 0.0
        ncyc = p["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=job["palette"], location=fld.location,
            eval_width=ow, eval_height=oh, reverse=bool(p["reverse"]),
            log_premap=p["log_premap"], gamma=float(p["gamma"]), phase=phase, n_cycles=ncyc,
            transfer=p["transfer"], transfer_gamma=float(p["transfer_gamma"]), filter=FILT)
        prep = cm.stretch_field(fld)
        prof = cm.gradient_transfer_profile(fld, prep) if p["transfer"] == "grad" else None
        img = cm.render_candidate(fld, cfg, _LIB, prep=prep, profile=prof)
        Image.fromarray(img).save(crop_path, quality=JPG_Q)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)
    return {"sid": job["sid"], "cached": False}


# --- distance metrics -------------------------------------------------------
def srgb_to_lab(rgb):
    """uint8 sRGB HxWx3 -> CIELAB (D65). Standard sRGB->XYZ->Lab."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def mean_de76(lab1, lab2):
    return float(np.mean(np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))))


def ssim(g1, g2, sigma=1.5, L=255.0):
    """Grayscale SSIM, Gaussian-windowed (skimage defaults: sigma 1.5, gaussian_weights)."""
    g1 = g1.astype(np.float64); g2 = g2.astype(np.float64)
    k1, k2 = 0.01, 0.03
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    filt = lambda x: gaussian_filter(x, sigma, truncate=3.5)
    mu1, mu2 = filt(g1), filt(g2)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = filt(g1 * g1) - mu1s
    s2 = filt(g2 * g2) - mu2s
    s12 = filt(g1 * g2) - mu12
    num = (2 * mu12 + C1) * (2 * s12 + C2)
    den = (mu1s + mu2s + C1) * (s1 + s2 + C2)
    smap = num / den
    # skimage crops the border by the filter pad; approximate by trimming 3.5*sigma.
    pad = int(round(3.5 * sigma))
    return float(smap[pad:-pad, pad:-pad].mean())


def rgb_to_gray(rgb):
    return rgb.astype(np.float64) @ np.array([0.2125, 0.7154, 0.0721])


def measure(row, sid):
    a = np.asarray(Image.open(CROPS / f"{row['image_id']}.jpg").convert("RGB"))
    b = np.asarray(Image.open(SMOOTH_CROPS / f"{sid}.jpg").convert("RGB"))
    de = mean_de76(srgb_to_lab(a), srgb_to_lab(b))
    s = ssim(rgb_to_gray(a), rgb_to_gray(b))
    return de, 1.0 - s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="render smooth counterparts first")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass

    SMOOTH_CROPS.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in IMAGES.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # dedup -> one job per unique (location, palette, color)
    jobs = {}
    row_sid = {}
    for r in rows:
        k = dedup_key(r)
        sid = smooth_id(k)
        row_sid[r["image_id"]] = sid
        if sid not in jobs:
            jobs[sid] = {"sid": sid, "render": r["render"], "palette": r["render"]["palette"],
                         "color_params": r["provenance"]["color_params"]}
    print(f"[smooth] {len(rows)} rasters -> {len(jobs)} unique smooth counterparts")

    if args.render:
        todo = [j for j in jobs.values() if not (SMOOTH_CROPS / f"{j['sid']}.jpg").exists()]
        print(f"[smooth] rendering {len(todo)} (rest cached) w/ {args.workers}x3 threads")
        t0 = time.time(); n = 0
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
            futs = {ex.submit(render_smooth, j): j for j in todo}
            for fut in as_completed(futs):
                j = futs[fut]
                try:
                    fut.result(); n += 1
                except Exception as exc:
                    print(f"[ERR] {j['sid']}: {str(exc)[:200]}")
                if n and n % 20 == 0:
                    el = time.time() - t0
                    print(f"[smooth] {n}/{len(todo)}  {n/el*60:.1f}/min  ETA {(len(todo)-n)/(n/el)/60:.1f}m")
        print(f"[smooth] rendered {n} in {(time.time()-t0)/60:.1f}m")

    # measure every raster against its (deduped) smooth counterpart
    print("[smooth] measuring distances ...")
    out = []
    for i, r in enumerate(rows):
        sid = row_sid[r["image_id"]]
        de, oms = measure(r, sid)
        out.append({
            "image_id": r["image_id"], "smooth_id": sid,
            "mode": r["render"]["render_mode"], "family": r["provenance"]["family"],
            "mode_kind": r["provenance"]["mode_kind"],
            "de76": de, "one_minus_ssim": oms,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"[smooth] wrote {RESULTS.relative_to(REPO)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
