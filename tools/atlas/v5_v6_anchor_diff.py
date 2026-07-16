#!/usr/bin/env python
"""v5-vs-v6 anchor decode diff — resolve the julia_known_good anchor loose end.

The v6 deploy verification (verify_v6_gate.py) left one loose end: the
`julia_known_good` anchor decoded to 1 under v6 (p_notbad 0.493 — coin-flip on the
bad side). This script resolves whether that is a NON-DIAGNOSTIC anchor frame or a
localized deg-2 Julia decode SHIFT by scoring the *same* frames under v5 and v6 and
diffing.

Reuses the anchor set + render harness from verify_v6_gate (identical crop geometry).
Each anchor is rendered ONCE and scored TWICE — once under each checkpoint, each
through its own deploy config (mean/std/head read from the checkpoint's own config by
score_lib.Scorer). Forward passes only.

  uv run python tools/atlas/v5_v6_anchor_diff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "tools" / "scoring", ROOT / "tools" / "corpus",
           ROOT / "tools" / "mining"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from score_lib import Scorer, corn_decode          # noqa: E402
from verify_v6_gate import ANCHORS, render_crop     # noqa: E402

V5_CKPT = "data/classifier/v5/model_best.pt"
V6_CKPT = "data/classifier/v6/model_best.pt"


def main() -> int:
    print("=== v5-vs-v6 anchor decode diff ===")
    print(f"v5 ckpt : {V5_CKPT}")
    print(f"v6 ckpt : {V6_CKPT}")

    v5 = Scorer(model_path=V5_CKPT)
    v6 = Scorer(model_path=V6_CKPT)
    print(f"device  : {v5.device}")
    # Each scorer normalizes through its OWN checkpoint's mean/std (the deploy config).
    print(f"v5 mean/std : {tuple(v5.cfg['mean'])} / {tuple(v5.cfg['std'])}")
    print(f"v6 mean/std : {tuple(v6.cfg['mean'])} / {tuple(v6.cfg['std'])}")

    hdr = (f"{'anchor':<24} | {'v5 E':>6} {'v5 pnb':>7} {'v5 pgd':>7} {'v5 dec':>6} "
           f"| {'v6 E':>6} {'v6 pnb':>7} {'v6 pgd':>7} {'v6 dec':>6} | {'dDec':>5} {'dpnb':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for label, family, cx, cy, fw, c_re, c_im, _expect_degen in ANCHORS:
        # Render ONCE (identical crop to both nets), score TWICE.
        jpg = render_crop(label, family, cx, cy, fw, c_re, c_im)
        with Image.open(jpg) as im:
            rgb = im.convert("RGB")
            s5, nb5, g5 = v5.score_pils([rgb])
            s6, nb6, g6 = v6.score_pils([rgb])
        e5, p5, q5 = float(s5[0]), float(nb5[0]), float(g5[0])
        e6, p6, q6 = float(s6[0]), float(nb6[0]), float(g6[0])
        d5, d6 = corn_decode(p5, q5), corn_decode(p6, q6)
        print(f"{label:<24} | {e5:>6.3f} {p5:>7.3f} {q5:>7.3f} {d5:>6} "
              f"| {e6:>6.3f} {p6:>7.3f} {q6:>7.3f} {d6:>6} "
              f"| {d6 - d5:>+5} {p6 - p5:>+7.3f}")

    print("\n(dec = rank-consistent hard class in {1,2,3}; dDec = v6 - v5; dpnb = v6 - v5 P(not-bad))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
