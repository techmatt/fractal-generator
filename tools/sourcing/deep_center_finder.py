"""High-precision deep-center finder (hand-curation track).

Produces **valid deep centers** — points on ∂M whose neighborhoods sustain
structure at depth — as high-precision **decimal-string** centers that flow
straight into the proven perturbation render tier (bare `render` / `sheet`,
`iterate_location` auto-selects perturbation at spacing ≤ 1e-13).

This is the sourcing component the deep probe proved missing
(`docs/findings/deep_mandelbrot_visual_probe.md`): the guided-descend walker is
f64-bound (`Frame.center: Complex<f64>`) and structurally cannot localize a
center below ~f64 resolution, so deep q4 harvesting needs a component that
*tracks ∂M at high precision*. This does exactly that, two ways:

  * **Nucleus** (period-p hyperbolic-component center): Newton on z_p(c)=0,
    the critical orbit returning to 0. Lands minibrot centers — self-similar
    over a band around the component's own size.
  * **Misiurewicz** (pre-periodic z_{k+n}=z_k): Newton on that residual. Lands
    points that stay *on* ∂M at every scale, so structure persists across many
    decades (the probe's sustained-q4 Seahorse center is one of these).

All Newton arithmetic is mpmath high precision (correctly rounded — a Newton
solver needs accurate division, unlike `hp.rs`'s projection-absorbed orbit
arithmetic). Coordinates leave as decimal strings; nothing here trusts a
classifier, scores, or emits — Matt's eye picks the beautiful ones.

CLI (also importable as a library — hand-curation calls it repeatedly):

    # Identify what an f64 seed converges to (period / preperiod scan):
    uv run python tools/sourcing/deep_center_finder.py scan \
        --seed -0.743643887 0.131825904 --max-period 24

    # Refine a nucleus and emit a render-ready deep center:
    uv run python tools/sourcing/deep_center_finder.py nucleus \
        --seed -0.1592 1.0317 --period 3

    # Refine a Misiurewicz point:
    uv run python tools/sourcing/deep_center_finder.py misiurewicz \
        --seed -0.743643887 0.131825904 --preperiod 5 --period 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import mpmath as mp


# ---------------------------------------------------------------------------
# Precision sizing. A center that localizes a frame of width `fw` must be known
# to well below `fw` relative to |c|~O(1); Newton converges quadratically so we
# just run generously above what any emitted depth needs.
# ---------------------------------------------------------------------------
def dps_for_fw(fw: float, guard: int = 30) -> int:
    """Decimal working precision sufficient to localize a frame of width `fw`."""
    import math
    need = int(math.ceil(-math.log10(fw))) if fw > 0 else 20
    return max(50, need + guard)


def emit_digits_for_fw(fw: float, guard: int = 15) -> int:
    """How many significant digits to serialize a center for a frame of width `fw`."""
    import math
    need = int(math.ceil(-math.log10(fw))) if fw > 0 else 20
    return need + guard


# ---------------------------------------------------------------------------
# Core orbit + derivative recurrences (all in the ambient mpmath precision).
#   z_0 = 0,  z_{n+1} = z_n^2 + c
#   d_n = dz_n/dc:  d_0 = 0,  d_{n+1} = 2 z_n d_n + 1
# ---------------------------------------------------------------------------
def _orbit(c, n):
    """Return (z_n, d_n) after n steps of the critical orbit at parameter c."""
    z = mp.mpc(0)
    d = mp.mpc(0)
    for _ in range(n):
        d = 2 * z * d + 1
        z = z * z + c
    return z, d


def _orbit_at(c, k, n):
    """Return (z_k, d_k, z_{k+n}, d_{k+n}) — orbit + derivative captured at step
    k and at step k+n. For the Misiurewicz residual z_{k+n} - z_k."""
    z = mp.mpc(0)
    d = mp.mpc(0)
    zk = dk = None
    for i in range(k + n):
        if i == k:
            zk, dk = z, d
        d = 2 * z * d + 1
        z = z * z + c
    if k == 0:
        zk, dk = mp.mpc(0), mp.mpc(0)
    return zk, dk, z, d


# ---------------------------------------------------------------------------
# Newton solvers.
# ---------------------------------------------------------------------------
@dataclass
class NewtonResult:
    c: object            # mpmath.mpc — refined parameter
    converged: bool
    iters: int
    residual: float      # |g(c)| at the final iterate (log10-ish scale)
    kind: str            # "nucleus" | "misiurewicz"
    period: int
    preperiod: int = 0


def newton_nucleus(c0, period, *, max_steps=200, tol_dps_margin=6):
    """Newton on z_p(c) = 0 (period-p nucleus). Returns a NewtonResult."""
    c = mp.mpc(c0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    residual = mp.inf
    it = 0
    for it in range(1, max_steps + 1):
        z, d = _orbit(c, period)
        residual = abs(z)
        if d == 0:
            break
        step = z / d
        c = c - step
        if abs(step) < tol and residual < tol:
            break
    z, _ = _orbit(c, period)
    residual = abs(z)
    conv = residual < tol
    return NewtonResult(c=c, converged=bool(conv), iters=it,
                        residual=float(mp.log10(residual)) if residual > 0 else -999.0,
                        kind="nucleus", period=period)


def newton_misiurewicz(c0, preperiod, period, *, max_steps=200, tol_dps_margin=6):
    """Newton on z_{k+n}(c) - z_k(c) = 0 (pre-periodic Misiurewicz point,
    preperiod k, eventual period n). Returns a NewtonResult."""
    c = mp.mpc(c0)
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    residual = mp.inf
    it = 0
    for it in range(1, max_steps + 1):
        zk, dk, zkn, dkn = _orbit_at(c, preperiod, period)
        g = zkn - zk
        gp = dkn - dk
        residual = abs(g)
        if gp == 0:
            break
        step = g / gp
        c = c - step
        if abs(step) < tol and residual < tol:
            break
    zk, _, zkn, _ = _orbit_at(c, preperiod, period)
    residual = abs(zkn - zk)
    conv = residual < tol
    return NewtonResult(c=c, converged=bool(conv), iters=it,
                        residual=float(mp.log10(residual)) if residual > 0 else -999.0,
                        kind="misiurewicz", period=period, preperiod=preperiod)


def is_minimal_misiurewicz(c, preperiod, period, *, tol_dps_margin=6):
    """A Misiurewicz solution is *minimal* (genuinely preperiod-k / period-n)
    only if the orbit is not already periodic one step earlier and the eventual
    period does not divide to something smaller. Cheap sanity screen so `scan`
    reports the minimal (k,n), not a multiple."""
    tol = mp.mpf(10) ** (-(mp.mp.dps - tol_dps_margin))
    # Not already satisfied at preperiod k-1 (would mean true preperiod < k).
    if preperiod >= 1:
        zk1, _, zkn1, _ = _orbit_at(c, preperiod - 1, period)
        if abs(zkn1 - zk1) < tol:
            return False
    # Eventual period is minimal: no proper divisor q|n also closes.
    for q in range(1, period):
        if period % q == 0:
            zk, _, zkq, _ = _orbit_at(c, preperiod, q)
            if abs(zkq - zk) < tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Minibrot size estimate (Munafo / Kalles-Fraktaler). |size| ~ atom radius,
# arg(size)/2 ~ orientation. Used to suggest an fw band for a nucleus.
# ---------------------------------------------------------------------------
def nucleus_size_estimate(c, period):
    """Return a complex size estimate for the period-p minibrot at nucleus c."""
    l = mp.mpc(1)
    b = mp.mpc(1)
    z = mp.mpc(0)
    for _ in range(1, period):
        z = z * z + c
        l = 2 * z * l
        if l == 0:
            return mp.mpc(0)
        b = b + 1 / l
    denom = b * l * l
    if denom == 0:
        return mp.mpc(0)
    return 1 / denom


# ---------------------------------------------------------------------------
# Emission — a NewtonResult -> render-ready deep center (decimal strings).
# ---------------------------------------------------------------------------
@dataclass
class DeepCenter:
    kind: str                 # "nucleus" | "misiurewicz"
    period: int
    preperiod: int
    cx: str                   # decimal-string center (render-tier native)
    cy: str
    fw_suggest: str           # a single suggested frame width for a first look
    fw_band: list             # [hi, lo] suggested band (decimal strings)
    self_similar: bool        # Misiurewicz => structure holds across all depths
    size_estimate: Optional[str]   # |minibrot size| (nucleus only), else None
    newton_converged: bool
    newton_iters: int
    newton_residual_log10: float
    render_maxiter: int       # a sensible maxiter for fw_suggest

    def render_cmd(self, exe="target/release/fractal-generator.exe",
                   width=1024, ss=2, out="out/deep_centers/preview.png"):
        return [exe, "sheet", "--builtins", "default cubehelix viridis",
                "--center-re", self.cx, "--center-im", self.cy,
                "--frame-width", self.fw_suggest, "--maxiter", str(self.render_maxiter),
                "--tile-width", str(width), "--aspect", "16:9", "--supersample", str(ss),
                "--backend", "auto", "--output", out]


def _maxiter_for_fw(fw: float) -> int:
    """Scale maxiter with depth (probe: fw 1e-20 wanted ~30k; shallow ~3k)."""
    import math
    d = -math.log10(fw) if fw > 0 else 3
    # ~1500 iters per decade of depth, floored at 3000, capped at 40000
    # (matches the probe ladder: fw 1e-20 -> ~30k).
    return int(max(3000, min(40000, round(1500 * d))))


def make_deep_center(res: NewtonResult, *, fw_suggest=None, emit_fw_floor=1e-20) -> DeepCenter:
    """Turn a converged NewtonResult into a render-ready DeepCenter with a
    suggested fw band and enough serialized digits for the deepest fw."""
    digits = emit_digits_for_fw(emit_fw_floor)
    cx = mp.nstr(res.c.real, digits, strip_zeros=False)
    cy = mp.nstr(res.c.imag, digits, strip_zeros=False)

    if res.kind == "misiurewicz":
        # Self-similar: structure holds at every scale. Suggest a mid-band first
        # look; band spans down to the proven perturbation depth.
        fw_hi = 1e-3
        fw_lo = emit_fw_floor
        fw0 = fw_suggest if fw_suggest is not None else 1e-9
        size_s = None
        self_sim = True
    else:
        size = nucleus_size_estimate(res.c, res.period)
        size_abs = float(abs(size)) if size != 0 else 0.0
        size_s = f"{size_abs:.6e}"
        # A nucleus sits in the minibrot's *interior* (black). Centered on it, the
        # money shot frames the whole minibrot as a small island ringed by its
        # radial spiral decorations — empirically fw ~ 4x size (validated: fw=size
        # is mostly interior black; fw < size on-nucleus is pure black). So the
        # compositionally-valid band for a nucleus-CENTERED frame is roughly
        # [~40x size (lots of context) .. ~2x size (minibrot fills frame)]. Going
        # deeper on-structure needs OFFSETTING onto a decoration, not the nucleus.
        fw_hi = size_abs * 40 if size_abs > 0 else 1e-3
        fw_lo = size_abs * 2 if size_abs > 0 else emit_fw_floor
        fw0 = fw_suggest if fw_suggest is not None else (size_abs * 4 if size_abs > 0 else 1e-6)
        self_sim = False

    return DeepCenter(
        kind=res.kind, period=res.period, preperiod=res.preperiod,
        cx=cx, cy=cy,
        fw_suggest=f"{fw0:.6e}",
        fw_band=[f"{fw_hi:.6e}", f"{fw_lo:.6e}"],
        self_similar=self_sim,
        size_estimate=size_s,
        newton_converged=res.converged,
        newton_iters=res.iters,
        newton_residual_log10=res.newton_residual_log10 if hasattr(res, "newton_residual_log10") else res.residual,
        render_maxiter=_maxiter_for_fw(fw0),
    )


# ---------------------------------------------------------------------------
# Scan — identify what an f64 seed converges to (period / preperiod).
# ---------------------------------------------------------------------------
def scan(seed, *, max_period=24, max_preperiod=12, do_nucleus=True, do_misiurewicz=True,
         near=1e-3):
    """From an f64 seed, try nucleus periods and Misiurewicz (k,n) combos; report
    the ones that converge within `near` of the seed, minimal ones first.

    `near` sets the use: a tight value (~1e-9) *identifies* an already-precise
    coordinate's type (does it Newton straight back to itself?); a looser value
    (~1e-2) *explores* which roots a rough seed's basin reaches."""
    hits = []
    c0 = mp.mpc(seed[0], seed[1])
    near = mp.mpf(str(near))
    if do_nucleus:
        for p in range(1, max_period + 1):
            r = newton_nucleus(c0, p)
            if r.converged and abs(r.c - c0) < near:
                # Minimal period only (skip p that's a multiple of a smaller hit).
                z, _ = _orbit(r.c, p)
                minimal = all(not (p % q == 0 and abs(_orbit(r.c, q)[0]) <
                                   mp.mpf(10) ** (-(mp.mp.dps - 6)))
                              for q in range(1, p))
                hits.append(("nucleus", p, 0, r, minimal, float(abs(r.c - c0))))
    if do_misiurewicz:
        for k in range(1, max_preperiod + 1):
            for n in range(1, max_period + 1):
                r = newton_misiurewicz(c0, k, n)
                if r.converged and abs(r.c - c0) < near:
                    minimal = is_minimal_misiurewicz(r.c, k, n)
                    hits.append(("misiurewicz", n, k, r, minimal, float(abs(r.c - c0))))
    # Rank: minimal first, then closest to seed, then smallest (k+n).
    hits.sort(key=lambda h: (not h[4], h[5], h[2] + h[1]))
    return hits


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _print_center(dc: DeepCenter, *, as_json=False):
    if as_json:
        print(json.dumps(asdict(dc)))
        return
    tag = f"{dc.kind} period={dc.period}" + (f" preperiod={dc.preperiod}" if dc.preperiod else "")
    print(f"# {tag}  |  Newton: converged={dc.newton_converged} "
          f"iters={dc.newton_iters} log10|res|={dc.newton_residual_log10:.1f}")
    if dc.size_estimate:
        print(f"# minibrot size estimate ~ {dc.size_estimate}")
    if dc.self_similar:
        print("# self-similar (Misiurewicz): structure holds across all depths")
    print(f"cx = {dc.cx}")
    print(f"cy = {dc.cy}")
    print(f"fw_suggest = {dc.fw_suggest}   band = [{dc.fw_band[0]} .. {dc.fw_band[1]}]")
    print(f"render_maxiter = {dc.render_maxiter}")
    print("render:  " + " ".join(
        (f'"{a}"' if " " in a else a) for a in dc.render_cmd()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--seed", nargs=2, type=float, required=True,
                       metavar=("RE", "IM"), help="rough f64 seed near ∂M")
        p.add_argument("--fw-floor", type=float, default=1e-20,
                       help="deepest fw to serialize digits for (default 1e-20)")
        p.add_argument("--fw-suggest", type=float, default=None,
                       help="override the suggested first-look frame width")
        p.add_argument("--json", action="store_true", help="emit JSON")

    pn = sub.add_parser("nucleus", help="Newton to a period-p nucleus")
    add_common(pn)
    pn.add_argument("--period", type=int, required=True)

    pm = sub.add_parser("misiurewicz", help="Newton to a Misiurewicz point")
    add_common(pm)
    pm.add_argument("--preperiod", type=int, required=True)
    pm.add_argument("--period", type=int, required=True)

    ps = sub.add_parser("scan", help="identify what an f64 seed converges to")
    ps.add_argument("--seed", nargs=2, type=float, required=True, metavar=("RE", "IM"))
    ps.add_argument("--max-period", type=int, default=24)
    ps.add_argument("--max-preperiod", type=int, default=12)
    ps.add_argument("--nucleus-only", action="store_true")
    ps.add_argument("--misiurewicz-only", action="store_true")
    ps.add_argument("--near", type=float, default=1e-3,
                    help="max |c-seed| to accept (tight ~1e-9 identifies a precise "
                         "coordinate; loose ~1e-2 explores a rough seed's basins)")
    ps.add_argument("--top", type=int, default=12)

    args = ap.parse_args(argv)

    if args.cmd == "scan":
        mp.mp.dps = 60
        hits = scan(tuple(args.seed),
                    max_period=args.max_period, max_preperiod=args.max_preperiod,
                    do_nucleus=not args.misiurewicz_only,
                    do_misiurewicz=not args.nucleus_only, near=args.near)
        if not hits:
            print("no convergent roots near the seed — widen --max-period/--max-preperiod "
                  "or check the seed is near ∂M", file=sys.stderr)
            return 1
        print(f"# {len(hits)} convergent root(s) near seed "
              f"({args.seed[0]}, {args.seed[1]}) — minimal first:")
        for kind, per, pre, r, minimal, dist in hits[:args.top]:
            tag = f"{kind:12s} period={per:<3d}" + (f" preperiod={pre:<3d}" if kind == "misiurewicz" else "          ")
            print(f"  {tag}  minimal={minimal!s:5s}  |c-seed|={dist:.2e}  "
                  f"iters={r.iters:<3d} log10|res|={r.residual:.1f}")
        return 0

    mp.mp.dps = max(dps_for_fw(args.fw_floor), dps_for_fw(args.fw_suggest or args.fw_floor))
    if args.cmd == "nucleus":
        r = newton_nucleus(mp.mpc(args.seed[0], args.seed[1]), args.period)
    else:
        r = newton_misiurewicz(mp.mpc(args.seed[0], args.seed[1]), args.preperiod, args.period)

    if not r.converged:
        print(f"# NOT CONVERGED (log10|res|={r.residual:.1f} after {r.iters} iters) — "
              f"seed may be too far, or wrong period/preperiod", file=sys.stderr)
    # Attach residual under the name make_deep_center expects.
    r.newton_residual_log10 = r.residual
    dc = make_deep_center(r, fw_suggest=args.fw_suggest, emit_fw_floor=args.fw_floor)
    _print_center(dc, as_json=args.json)
    return 0 if r.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
