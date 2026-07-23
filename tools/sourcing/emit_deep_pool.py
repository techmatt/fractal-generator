"""Batch companion to deep_center_finder: refine a curated seed list into a
deep-center pool.jsonl (one render-ready center per row) for hand-curation.

Reusable — edit SEEDS (or import `emit_pool`) and re-run. The single-center path
is `deep_center_finder.py <nucleus|misiurewicz|scan>`; this just fans that over a
list and writes a pool + ready-to-run render commands.

    uv run python tools/sourcing/emit_deep_pool.py            # -> out/deep_centers/pool.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import mpmath as mp

sys.path.insert(0, os.path.dirname(__file__))
import deep_center_finder as f  # noqa: E402


# (kind, seed_re, seed_im, period, preperiod, note). Seeds are rough f64 near ∂M;
# Newton refines to the exact HP nucleus / Misiurewicz point.
SEEDS = [
    ("nucleus", -0.7453, 0.1127, 35, 0, "seahorse-valley minibrot p35"),
    ("nucleus", -0.7453, 0.1127, 58, 0, "seahorse-valley minibrot p58"),
    ("nucleus", -0.7453, 0.1127, 47, 0, "seahorse-valley minibrot p47"),
    ("nucleus", -0.7453, 0.1127, 59, 0, "seahorse-valley minibrot p59"),
    ("nucleus",  0.2925, 0.0149, 29, 0, "elephant-valley minibrot p29"),
    ("nucleus", -0.1568, 1.0322,  4, 0, "north-bulb minibrot p4"),
    ("misiurewicz", 0.322,  0.0333, 5, 7, "elephant spiral M(7,5)"),
    ("misiurewicz", 0.3228, 0.0330, 4, 8, "elephant spiral M(8,4)"),
    ("misiurewicz", 0.3115, 0.0257, 5, 8, "elephant spiral M(8,5)"),
    ("misiurewicz", 0.3032, 0.0202, 5, 9, "elephant spiral M(9,5)"),
]


def emit_pool(seeds=SEEDS, *, dps=80, out_path="out/deep_centers/pool.jsonl",
              preview_dir="out/deep_centers"):
    """Refine each seed to an exact center and write a pool.jsonl. Returns rows."""
    mp.mp.dps = dps
    rows = []
    for kind, sr, si, per, pre, note in seeds:
        seed = mp.mpc(sr, si)
        r = (f.newton_nucleus(seed, per) if kind == "nucleus"
             else f.newton_misiurewicz(seed, pre, per))
        r.newton_residual_log10 = r.residual
        if not r.converged:
            print(f"SKIP (no converge): {note}", file=sys.stderr)
            continue
        dc = f.make_deep_center(r)
        d = asdict(dc)
        d["note"] = note
        d["family"] = "mandelbrot"
        d["render_cmd"] = dc.render_cmd(out=f"{preview_dir}/preview_{len(rows):02d}.png")
        rows.append(d)
        print(f"{note:32s} iters={r.iters:<4d} log10res={r.residual:6.1f} "
              f"fw={dc.fw_suggest} size={dc.size_estimate}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for d in rows:
            fh.write(json.dumps(d) + "\n")
    print(f"\nwrote {len(rows)} centers -> {out_path}")
    return rows


if __name__ == "__main__":
    emit_pool()
