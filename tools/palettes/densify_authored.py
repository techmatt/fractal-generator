"""Densify sparse *authored* palettes into dense render-native colormap libraries.

An authored palette is a small set of hand-placed stops with per-segment
transition semantics (smooth / hard / ease), authored for readability, not for
the render path. The Rust colorer wants a dense, even-spaced `[pos, [r,g,b]]`
stop list (sRGB8) it can bake into its OKLab LUT. This util is that bridge and is
intended to be reusable in production (not just the preview harness).

Segment semantics (the `segment` field on stop *i* describes the span to *i+1*):
  * ``smooth`` — linear interpolation in OKLab.
  * ``ease``   — smoothstep (slow-in / slow-out) easing of the interp parameter,
                 still interpolated in OKLab.
  * ``hard``   — cliff: by default softened to a smoothstep ramp of width
                 ``DEFAULT_SOFT_CLIFF`` ending at stop *i+1* (crisp but not a
                 wall — see the cliff-jarring diagnostic). ``--hard-step`` /
                 ``soft_cliff=0`` restores the raw snap (hold across the span,
                 snap to stop *i+1* within one dense step).

Interpolation is perceptual (OKLab, Ottosson) via ``tools/palettes/color.py`` —
the same formulation the Rust colorer uses — then converted back to sRGB8 and
gamut-clamped. The loop (first stop == last stop) is preserved so the cyclic
render path seams cleanly.

Output is a colormap-library JSON (the `--colormaps` format render-one reads):
``[{"name", "stops": [[pos,[r,g,b]],...], "cycle": "cyclic", "mirror_needed": false}, ...]``.

Usage:
    uv run python tools/palettes/densify_authored.py IN.json OUT.json [--dense 512]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import color as C  # noqa: E402  srgb<->oklab (Ottosson), matches the Rust colorer


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Classic Hermite smoothstep 3u^2 - 2u^3 on [0,1]."""
    return u * u * (3.0 - 2.0 * u)


DEFAULT_SOFT_CLIFF = 0.08  # default hard-cliff ramp width (cycle-position units)


def densify_palette(stops: list[dict], dense: int = 512,
                    segments_override: str | None = None,
                    soft_cliff: float = DEFAULT_SOFT_CLIFF) -> list[list]:
    """Sparse authored stops -> `dense` even-spaced [pos,[r,g,b]] sRGB8 stops.

    `stops` must be ordered with first pos==0.0, last pos==1.0. Returns a list of
    ``[pos, [r,g,b]]`` at pos = i/(dense-1), i in 0..dense (endpoints inclusive so
    the loop closes exactly).

    ``hard`` cliffs are realized by DEFAULT as a smoothstep ease ramp of width
    ``soft_cliff`` (cycle-position units, default ``DEFAULT_SOFT_CLIFF=0.08``)
    ending exactly at the next stop's position — crisp but not a wall (see the
    cliff-jarring diagnostic). The ramp overrides the held span and bleeds into
    the preceding span when the width exceeds the authored hard span.

    Diagnostic overrides:
      * ``segments_override="smooth"`` — treat *every* segment as ``smooth``
        (linear OKLab), ignoring authored ``hard``/``ease``. Removes cliffs
        entirely.
      * ``soft_cliff=0`` (or negative) — restore the raw ~1/512 instantaneous
        step (hold stop *i*'s color, snap at the boundary). The pre-diagnostic
        behavior, kept for comparison. Ignored when ``segments_override`` is set.
    """
    pos = np.array([s["pos"] for s in stops], dtype=np.float64)
    rgb = np.array([s["rgb"] for s in stops], dtype=np.float64) / 255.0
    # `cliff` (v2 authored schema) is an alias for `hard`. A per-stop optional
    # `width` overrides the global soft_cliff ramp width for that cliff only.
    seg = [("hard" if s.get("segment") == "cliff" else s.get("segment", "smooth"))
           for s in stops]
    widths = [s.get("width", None) for s in stops]
    if pos[0] != 0.0 or pos[-1] != 1.0:
        raise ValueError(f"stops must span [0,1] exactly, got [{pos[0]}, {pos[-1]}]")
    if segments_override not in (None, "smooth"):
        raise ValueError(f"segments_override must be None or 'smooth', got {segments_override!r}")

    lab = C.srgb_to_oklab(rgb)  # (n,3)
    t = np.linspace(0.0, 1.0, dense)  # even-spaced, endpoints inclusive
    # Segment index for each t: last stop whose pos <= t (clamped so t==1 -> last span).
    idx = np.clip(np.searchsorted(pos, t, side="right") - 1, 0, len(pos) - 2)

    out_lab = np.empty((dense, 3), dtype=np.float64)
    for k in range(dense):
        i = int(idx[k])
        span = pos[i + 1] - pos[i]
        u = 0.0 if span <= 0 else (t[k] - pos[i]) / span
        mode = "smooth" if segments_override == "smooth" else seg[i]
        if mode == "hard":
            out_lab[k] = lab[i]  # hold; cliff resolves at the next span's start
        else:
            if mode == "ease":
                u = float(_smoothstep(np.array(u)))
            out_lab[k] = (1.0 - u) * lab[i] + u * lab[i + 1]

    # soft-cliff post-pass: replace each hard step with a width-W smoothstep ramp
    # ending at the boundary. Overrides whatever the hold produced in [p-W, p),
    # bleeding into the preceding span when W exceeds the authored hard span.
    if soft_cliff and segments_override != "smooth" and soft_cliff > 0:
        for i in range(len(pos) - 1):
            if seg[i] != "hard":
                continue
            w = soft_cliff if widths[i] is None else float(widths[i])
            if w <= 0:
                continue  # per-stop hard snap
            p = pos[i + 1]
            c_before, c_after = lab[i], lab[i + 1]
            win = (t >= p - w) & (t < p)
            u = (t[win] - (p - w)) / w  # 0 at ramp start, ->1 at p
            s = _smoothstep(u)[:, None]
            out_lab[win] = (1.0 - s) * c_before + s * c_after

    srgb = C.oklab_to_srgb(out_lab)  # clipped to [0,1]
    rgb8 = np.clip(np.rint(srgb * 255.0), 0, 255).astype(int)
    return [[float(t[k]), [int(rgb8[k, 0]), int(rgb8[k, 1]), int(rgb8[k, 2])]] for k in range(dense)]


def densify_library(palettes: list[dict], dense: int = 512,
                    segments_override: str | None = None,
                    soft_cliff: float = DEFAULT_SOFT_CLIFF) -> list[dict]:
    """Densify a list of authored palette objects into colormap-library entries."""
    out = []
    for p in palettes:
        out.append({
            "name": p["name"],
            "stops": densify_palette(p["stops"], dense, segments_override, soft_cliff),
            "cycle": "cyclic",       # authored palettes loop (first==last)
            "mirror_needed": False,  # cyclic -> no pre-mirror at bake
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="authored palette JSON (list of palette objects)")
    ap.add_argument("output", type=Path, help="densified colormap-library JSON")
    ap.add_argument("--dense", type=int, default=512, help="dense stop count (default 512)")
    ap.add_argument("--segments", choices=["smooth"], default=None,
                    help="override all segment semantics (only 'smooth' supported): "
                         "treat every segment as smooth, ignoring hard/ease (removes cliffs)")
    ap.add_argument("--soft-cliff", type=float, default=DEFAULT_SOFT_CLIFF, metavar="W",
                    help=f"width of the default hard-cliff smoothstep ramp, in cycle-position "
                         f"units, ending at the boundary (default {DEFAULT_SOFT_CLIFF}); "
                         f"pass 0 or use --hard-step for the raw instantaneous cliff")
    ap.add_argument("--hard-step", action="store_true",
                    help="diagnostic: restore the raw ~1/512 instantaneous hard cliff "
                         "(equivalent to --soft-cliff 0)")
    args = ap.parse_args()

    soft_cliff = 0.0 if args.hard_step else args.soft_cliff
    palettes = json.loads(args.input.read_text())
    lib = densify_library(palettes, args.dense, args.segments, soft_cliff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lib, indent=1))
    print(f"densified {len(lib)} palette(s) x {args.dense} stops -> {args.output}")


if __name__ == "__main__":
    main()
