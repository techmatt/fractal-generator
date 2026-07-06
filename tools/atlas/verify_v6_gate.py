#!/usr/bin/env python
"""Verify the v6 discovery gate loads through the EXACT deploy path and behaves sanely.

This is the step-2 load-contract check for the v5->v6 discovery swap (see
prompts/deploy_v6_discovery_gate.md). It does NOT re-implement scoring; it builds the
SAME guarded scorer the seeder builds (`guard.make_guarded_scorer(guard.SCORER_PATH)`,
which resolves to probe.ACTIVE_CKPT) and asserts:

  1. cfg["target"] == "ordinal" and the CORN head emits K-1 == 2 logits (dummy forward).
  2. mean/std come from the v6 checkpoint (match data/classifier/v6/config.json), NOT a
     v5-hardcoded constant.
  3. A few fixed anchor frames render, score, and decode sanely, and the guard still
     sentinels the known degenerate (guard is model-free, so the swap shouldn't touch it).

  uv run python tools/atlas/verify_v6_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "reframe_probe", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import guard                                  # noqa: E402
import location as loc_mod                    # noqa: E402
from probe import BIN, PALETTE, ACTIVE_CKPT, auto_maxiter  # noqa: E402
from score_lib import corn_decode             # noqa: E402
import subprocess                             # noqa: E402

CROP_W, CROP_H, CROP_SS = 1280, 720, 2        # label/deploy render geometry (Transform stretches to 384x224)
SCRATCH = ROOT / "out" / "atlas" / "verify_v6_gate"

# Fixed anchor frames.  (label, family, cx, cy, fw, c_re, c_im, expect_degenerate)
ANCHORS = [
    # Known-good Julia c from the prompt (full z-plane set, centered at origin).
    ("julia_known_good", "julia", "0", "0", "3.0",
     "-0.07810228973371881", "-0.6514609012382414", False),
    # A known-good Mandelbrot view (classic seahorse-valley spiral) — should score well.
    ("mandelbrot_seahorse", "mandelbrot", "-0.743643887037151", "0.13182590420533",
     "1.0e-4", None, None, False),
    # A real discovered multibrot3 q3 outcome from the durable ledger.
    ("multibrot3_ledger_q3", "multibrot3", "-0.5385741162430248", "0.45328854576385386",
     "0.00013675544721103756", None, None, False),
    # Known degenerate: deep inside the main cardioid -> all-interior black -> guard 'interior'.
    ("degenerate_cardioid", "mandelbrot", "-0.5", "0.0", "0.5", None, None, True),
]


def render_crop(label, family, cx, cy, fw, c_re, c_im) -> Path:
    """render-one at the deploy crop geometry -> JPG. Returns the jpg path."""
    outdir = SCRATCH / label
    outdir.mkdir(parents=True, exist_ok=True)
    jpg = outdir / "crop.jpg"
    fam_params = {}
    loc = loc_mod.Location(family=family, cx=cx, cy=cy, fw=fw, c_re=c_re, c_im=c_im,
                           family_params=fam_params)
    mi = auto_maxiter(float(fw))
    cmd = [
        str(BIN), "render-one", "--cx", cx, "--cy", cy, "--fw", repr(float(fw)),
        "--width", str(CROP_W), "--height", str(CROP_H), "--supersample", str(CROP_SS),
        "--maxiter", str(mi), "--palette", PALETTE, "--out", str(jpg),
    ] + loc_mod.render_one_flags(loc)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not jpg.exists():
        raise SystemExit(f"render failed [{label}]: {r.stderr[-400:]}")
    return jpg


def main() -> int:
    print(f"=== verify v6 discovery gate ===")
    print(f"ACTIVE_CKPT (probe) : {ACTIVE_CKPT}")
    print(f"guard.SCORER_PATH   : {guard.SCORER_PATH}")
    assert guard.SCORER_PATH == ACTIVE_CKPT, "guard not routed through the single source of truth"
    assert "v6" in ACTIVE_CKPT, f"expected v6 checkpoint, got {ACTIVE_CKPT}"

    # --- Build the EXACT deploy scorer the seeder builds. --- #
    scorer = guard.make_guarded_scorer(guard.SCORER_PATH)
    cfg = scorer.cfg
    print(f"\n-- load contract --")
    print(f"device      : {scorer.device}")
    print(f"target      : {cfg['target']}   geometry={cfg['geometry']}  interp={cfg['interpolation']}")
    print(f"ckpt mean   : {tuple(cfg['mean'])}")
    print(f"ckpt std    : {tuple(cfg['std'])}")

    # (1) ordinal + CORN K-1 == 2 head via a dummy forward at the real deploy input shape.
    assert cfg["target"] == "ordinal", f"expected ordinal head, got {cfg['target']!r}"
    dummy = Image.new("RGB", (CROP_W, CROP_H), (40, 60, 90))
    x = scorer.transform(dummy).unsqueeze(0).to(scorer.device)
    with torch.no_grad():
        logits = scorer.model(x)
    assert logits.shape[-1] == 2, f"CORN head must emit K-1=2 logits, got {tuple(logits.shape)}"
    print(f"deploy input: {tuple(x.shape)}   head logits: {tuple(logits.shape)}  (CORN K-1=2 OK)")

    # (2) mean/std come from the v6 CHECKPOINT, not a hardcoded constant.
    v6_cfg = json.loads((ROOT / "data" / "classifier" / "v6" / "config.json").read_text())
    assert list(cfg["mean"]) == list(v6_cfg["mean"]), "mean does not match v6 config.json"
    assert list(cfg["std"]) == list(v6_cfg["std"]), "std does not match v6 config.json"
    print("mean/std match data/classifier/v6/config.json  (came from the v6 checkpoint) OK")

    # --- (3) Anchor frames: render, score, decode, guard. --- #
    print(f"\n-- anchor frames (k3 == E[ord] of the single deploy crop) --")
    hdr = f"{'anchor':<24} {'guard':>9} {'E[ord]':>8} {'p_nb':>6} {'p_good':>7} {'decode':>7} {'int_frac':>9} {'fld_std':>8}"
    print(hdr)
    print("-" * len(hdr))
    ok_all = True
    for label, family, cx, cy, fw, c_re, c_im, expect_degen in ANCHORS:
        jpg = render_crop(label, family, cx, cy, fw, c_re, c_im)
        # bare (unguarded) score of the crop.
        with Image.open(jpg) as im:
            s, nb, g = scorer.score_pils([im.convert("RGB")])
        e_ord, p_nb, p_good = float(s[0]), float(nb[0]), float(g[0])
        decode = corn_decode(p_nb, p_good)
        # guard: render the model-free field + measure (signature: cx, cy, fw, out_bin, **kw).
        gf = SCRATCH / label / "field.bin"
        gs = guard.measure_location(cx, cy, fw, gf, family=family, c_re=c_re, c_im=c_im)
        verdict = guard.guard_fail(gs.interior_frac, gs.field_std)
        gtag = verdict or "pass"
        print(f"{label:<24} {gtag:>9} {e_ord:>8.3f} {p_nb:>6.3f} {p_good:>7.3f} {decode:>7} "
              f"{gs.interior_frac:>9.3f} {gs.field_std:>8.2f}")
        if expect_degen and verdict is None:
            print(f"   !! expected {label} to FAIL the guard but it passed")
            ok_all = False
        if (not expect_degen) and verdict is not None:
            print(f"   ?? {label} unexpectedly failed the guard ({verdict}) — eyeball this")

    print(f"\nverdict: {'PASS — v6 loads through the deploy path; guard sentinels the degenerate' if ok_all else 'CHECK ABOVE'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
