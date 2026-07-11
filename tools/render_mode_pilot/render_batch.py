"""Render-mode pilot (500 stratified labels) — Step 3+4: render the sampled 500
rasters at 1280x720 ss2 Lanczos-3, each mode in its native canonical recipe over the
inherited (gate-approved) palette + color params.

Render paths (see keeper-color-fidelity):
  * pure  : render-one --dump-field <mode-field> -> colormap.render_candidate with the
            FULL approved color params (reverse/log_premap/gamma/phase/n_cycles/
            transfer=grad+transfer_gamma). Bit-faithful colouring.
  * composite / direct : render-one --coloring <spec> --palette --colormaps --out.
            Honors reverse/log_premap->transform/gamma/palette_cycles/offset, but CANNOT
            express transfer=grad -> that knob is dropped (`transfer_dropped` stamped).

Ops: ProcessPoolExecutor(max_workers=4), RAYON_NUM_THREADS=3/worker (fills 12 cores).
Durable append ledger `_progress_ledger.jsonl` -> resume: rasters with a ledger row AND
an on-disk crop are skipped. Final `images.jsonl` is (re)assembled from the ledger.

    uv run python -u tools/render_mode_pilot/render_batch.py --limit 4   # smoke
    uv run python -u tools/render_mode_pilot/render_batch.py             # full 500
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))
sys.path.insert(0, str(REPO / "tools" / "queries"))

EXE = str(REPO / "target/release/fractal-generator.exe")
POOL_CMAPS = str(REPO / "data/palettes/pool_colormaps.json")
POOL_FEATS = str(REPO / "data/palettes/palette_features.json")
PLAN = REPO / "scratchpad/rmp_sample_plan.jsonl"

BATCH_ID = "2026-07-10_render_mode_pilot_v1"
OUT_DIR = REPO / "data/render_mode_corpus/batches" / BATCH_ID
CROPS = OUT_DIR / "crops"
FIELDS = OUT_DIR / "_fields"
LEDGER = OUT_DIR / "_progress_ledger.jsonl"
IMAGES = OUT_DIR / "images.jsonl"

W, H, SS, FILT = 1280, 720, 2, "lanczos3"
JPG_Q = 95

PURE_FIELD_SPEC = {
    "tia": {"field": "tia", "skip": 1},
    "stripe": {"field": "stripe", "stripe_density": 6},
    "exp_smoothing": {"field": "exp_smoothing"},
    "gaussian_int": {"field": "gaussian_int"},
    "trap_circle": {"field": "trap_circle"},
    "curv_linear": {"field": "curvature"},
}
SPEC_FILE = {  # composite + direct modes -> specs/<file>.json
    "smooth_mean_angle": "smooth_mean_angle",
    "smooth_angle_min": "smooth_angle_min",
    "composite_c7_smooth_trap_circle": "composite_c7_smooth_trap_circle",
    "composite_c13_smooth_stripe": "composite_c13_smooth_stripe",
    "composite_c17_smooth_curvature": "composite_c17_smooth_curvature",
    "direct_trap_ring": "direct_trap_ring",
    "direct_trap_screen": "direct_trap_screen",
    "direct_trap_multiply": "direct_trap_multiply",
    "direct_trap_lines": "direct_trap_lines",
}

# per-worker globals (loaded once in the initializer)
_LIB = None
_CM = None
_LOC = None


def _init_worker():
    global _LIB, _CM, _LOC
    os.environ["RAYON_NUM_THREADS"] = "3"
    import colormap as cm
    import location as loc_mod
    import query_sampler as qs
    _CM, _LOC = cm, loc_mod
    _LIB = qs.load_pool_library()


def _loc_of(render):
    return _LOC.from_render_block(render)


def _locflags(loc):
    return _LOC.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
            "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def _run(cmd):
    env = dict(os.environ, RAYON_NUM_THREADS="3")
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-700:])


def _render_pure(entry, loc, crop_path):
    cm = _CM
    spec = dict(PURE_FIELD_SPEC[entry["mode"]])
    binp = FIELDS / f"{entry['image_id']}.bin"
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(W), "--height", str(H),
             "--supersample", str(SS), "--coloring", json.dumps(spec), "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        p = entry["color_params"]
        ptype = _LIB.palette_type(entry["palette"])
        phase = p["phase"] if ptype == "cyclic" else 0.0
        ncyc = p["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=entry["palette"], location=fld.location,
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
    return False  # transfer never dropped on the pure path


def _render_rust(entry, loc, crop_path):
    spec = json.loads((REPO / "specs" / f"{SPEC_FILE[entry['mode']]}.json").read_text())
    spec.pop("tier", None)
    spec.update(entry.get("mode_params", {}))          # direct: threshold/opacity override
    p = entry["color_params"]
    ptype = _LIB.palette_type(entry["palette"])
    spec["transform"] = "log" if p["log_premap"] == "log" else "linear"
    spec["gamma"] = float(p["gamma"])
    spec["reverse"] = bool(p["reverse"])
    if ptype == "cyclic":
        spec["palette_cycles"] = float(p["n_cycles"])
        spec["palette_offset"] = float(p["phase"])
    transfer_dropped = p["transfer"] == "grad"          # Rust can't express grad transfer
    tmp_png = FIELDS / f"{entry['image_id']}.png"
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(W), "--height", str(H),
             "--supersample", str(SS), "--filter", FILT, "--palette", entry["palette"],
             "--colormaps", POOL_CMAPS, "--coloring", json.dumps(spec), "--out", str(tmp_png)])
        with Image.open(tmp_png) as im:
            im.convert("RGB").save(crop_path, quality=JPG_Q)
    finally:
        tmp_png.unlink(missing_ok=True)
    return transfer_dropped


def render_one(entry):
    """Worker entry: render `entry` -> crop jpg, return the assembled images.jsonl row."""
    t0 = time.time()
    loc = _loc_of(entry["render"])
    crop_path = CROPS / f"{entry['image_id']}.jpg"
    if entry["kind"] == "pure":
        transfer_dropped = _render_pure(entry, loc, crop_path)
    else:
        transfer_dropped = _render_rust(entry, loc, crop_path)
    r = entry["render"]
    p = entry["color_params"]
    row = {
        "image_id": entry["image_id"],
        "render": {
            "cx": r["cx"], "cy": r["cy"], "fw": r["fw"], "maxiter": r["maxiter"],
            "fractal_type": r.get("fractal_type"), "c_re": r.get("c_re"), "c_im": r.get("c_im"),
            "palette": entry["palette"], "composition": "center",
            "width": W, "height": H, "ss": SS, "filter": FILT, "interior_mode": "black",
            "render_mode": entry["mode"],
        },
        "provenance": {
            "generator_version": "render_mode_pilot_v1",
            "batch_id": BATCH_ID,
            "lineage": "render_mode_pilot",
            "family": entry["family"],
            "location_key": entry["location_key"],
            "c_re": r.get("c_re"), "c_im": r.get("c_im"),
            "render_mode": entry["mode"],
            "mode_kind": entry["kind"],
            "mode_params": entry.get("mode_params", {}),
            "color_params": {
                "palette_type": _LIB.palette_type(entry["palette"]),
                "reverse": p["reverse"], "log_premap": p["log_premap"], "gamma": p["gamma"],
                "phase": p["phase"], "n_cycles": p["n_cycles"], "transfer": p["transfer"],
                "transfer_gamma": p["transfer_gamma"],
            },
            "transfer_dropped": transfer_dropped,
            "source": {"batch_id": "2026-07-09_wallpaper_headbatch_dramatic_v1",
                       "image_id": entry["src_image_id"], "p_ge3": entry["p_ge3"],
                       "gate": "wallpaper_head_v3 p_ge3>0.90"},
        },
        "label": {"score": None, "labeler": None, "labeled_at": None},
    }
    return {"image_id": entry["image_id"], "mode": entry["mode"], "family": entry["family"],
            "secs": time.time() - t0, "row": row}


def assemble_images(rows_by_id, plan):
    order = [e["image_id"] for e in plan if e["image_id"] in rows_by_id]
    IMAGES.write_text("\n".join(json.dumps(rows_by_id[i]) for i in order) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="render only the first N plan rows")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass

    CROPS.mkdir(parents=True, exist_ok=True)
    FIELDS.mkdir(parents=True, exist_ok=True)
    plan = [json.loads(l) for l in PLAN.read_text().splitlines() if l.strip()]
    if args.limit:
        plan = plan[:args.limit]

    # resume: a raster is done iff it has a ledger row AND its crop exists on disk
    done_rows = {}
    if LEDGER.exists():
        for l in LEDGER.read_text().splitlines():
            if not l.strip():
                continue
            rec = json.loads(l)
            if (CROPS / f"{rec['image_id']}.jpg").exists():
                done_rows[rec["image_id"]] = rec["row"]
    todo = [e for e in plan if e["image_id"] not in done_rows]
    print(f"[rmp] plan {len(plan)}  ·  done {len(done_rows)}  ·  todo {len(todo)}  "
          f"·  {args.workers} workers x 3 threads  ->  {OUT_DIR.relative_to(REPO)}")

    rows_by_id = dict(done_rows)
    errors = []
    ledger_fh = open(LEDGER, "a", encoding="utf-8")
    t_start = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(render_one, e): e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:
                errors.append({"image_id": e["image_id"], "mode": e["mode"], "err": str(exc)[:400]})
                print(f"[ERR] {e['image_id']} {e['mode']}: {str(exc)[:200]}")
                continue
            rows_by_id[res["image_id"]] = res["row"]
            ledger_fh.write(json.dumps({"image_id": res["image_id"], "row": res["row"]}) + "\n")
            ledger_fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t_start
                rate = n / el
                eta = (len(todo) - n) / rate if rate else 0
                print(f"[rmp] {n}/{len(todo)}  last {res['mode']:22s} {res['secs']:5.1f}s  "
                      f"·  {rate*60:.1f}/min  ETA {eta/60:.1f}m")
    ledger_fh.close()

    assemble_images(rows_by_id, plan)
    print(f"[rmp] DONE  rendered {n} this run, {len(rows_by_id)} total  ·  "
          f"{(time.time()-t_start)/60:.1f}m  ·  errors {len(errors)}")
    if errors:
        for e in errors[:20]:
            print(f"   ERR {e['image_id']} {e['mode']}: {e['err'][:160]}")
    # batch.json
    (OUT_DIR / "batch.json").write_text(json.dumps({
        "created": BATCH_ID.split("_")[0] + "T00:00:00", "batch_id": BATCH_ID,
        "generator_version": "render_mode_pilot_v1",
        "n_images": len(rows_by_id), "n_errors": len(errors),
        "render": {"width": W, "height": H, "ss": SS, "filter": FILT},
        "schema_note": ("Render-mode quality-head pilot: 500 stratified rasters over the "
            "401 v3-gate-passers (p_ge3>0.90) x 15 registered modes (ex-smooth), direct-trap "
            "palette-deduped w/ opacity x threshold grid. Each mode in its native canonical "
            "recipe over the inherited approved palette+color params; pure modes bit-faithful "
            "(incl. transfer=grad), composite/direct drop transfer=grad (transfer_dropped). "
            "render.render_mode + provenance.mode_params define the crop beyond the standard "
            "render block. label.score null (bad/okay/good to be labeled)."),
        "source_gate": "scratchpad/gate_passers_v3.json (wallpaper_head_v3, p_ge3>0.90)",
    }, indent=2))
    print(f"[rmp] wrote {IMAGES.relative_to(REPO)} + batch.json")


if __name__ == "__main__":
    main()
