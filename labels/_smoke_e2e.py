"""Scratch: one end-to-end driver-path smoke on a NON-mandelbrot location.
Exercises the family-generic integration: retain-all beam -> strata-sample ~8 ->
label-spec re-render -> emit row dicts (render + provenance blocks). ONE beam only.
Writes crops to a scratch dir; never touches the real batch namespace."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "reframe_probe"))

import build_bootstrap as BB
import query_sampler as qs

SCRATCH = ROOT / "out" / "wallpaper_validate" / "smoke_e2e"
(SCRATCH / "crops").mkdir(parents=True, exist_ok=True)

CLS = "phoenix"          # non-mandelbrot; two-state kernel is the riskiest integration


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass

    spec = {s[0]: s for s in BB.CLASS_SPECS}[CLS]
    rows = BB._load_ledger(spec)
    q3 = BB._spatial_dedup([r for r in rows if r["decoded_class"] == 3])
    rng0 = np.random.default_rng(BB.SEED); rng0.shuffle(q3)
    srow = q3[0]
    loc = BB.to_location(spec, srow)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(BB.SEED)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = BB.SL.load_v2(device)
    print(f"[smoke] {CLS} loc fw={loc.fw[:10]} mi={loc.maxiter}  v2 epoch={epoch} on {device.type}  pool={len(lib.colormaps)}")

    t0 = time.time()
    res = BB.SL.run_location(f"{CLS}_smoke", loc, lib, sampler, model, device, BB.SEED, retain_all=True)
    n_cand = len(res["all_candidates"])
    rng = np.random.default_rng(BB.SEED)
    picks = BB.strata_sample(res["all_candidates"], rng)
    print(f"[smoke] beam done [{time.time()-t0:.0f}s]  all_candidates={n_cand}  picks={len(picks)}")

    field = BB.ensure_label_field(loc)
    emitted = []
    for pi, pick in enumerate(picks):
        image_id = f"smoke_{pi:02d}"
        w, h = BB.render_label_crop(field, pick["config"], lib, SCRATCH / "crops" / f"{image_id}.jpg")
        assert (w, h) == (BB.LABEL_W, BB.LABEL_H)
        emitted.append({
            "image_id": image_id,
            "render": BB.render_block(loc, pick["config"].palette),
            "provenance": BB.provenance_block(spec, loc, "q3_core", srow, pick),
            "label": {"score": None, "labeler": None, "labeled_at": None},
        })
    (SCRATCH / "images.jsonl").write_text("\n".join(json.dumps(r) for r in emitted) + "\n")

    scores = sorted(p["score"] for p in picks)
    strata = {}
    for p in picks:
        strata[p.get("stratum")] = strata.get(p.get("stratum"), 0) + 1
    pals = {p["config"].palette for p in picks}
    print(f"[smoke] emitted {len(emitted)} rows  distinct palettes={len(pals)}  "
          f"pref-v2 range [{scores[0]:.3f},{scores[-1]:.3f}]  strata={strata}")
    # sanity: render block carries phoenix family + params
    rb = emitted[0]["render"]
    print(f"[smoke] render block: fractal_type={rb['fractal_type']} p_re={rb.get('p_re')} p_im={rb.get('p_im')} "
          f"palette={rb['palette'][:24]}")
    print(f"[smoke] -> {SCRATCH}")


if __name__ == "__main__":
    main()
