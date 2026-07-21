"""Phoenix seed proposal sampler — the closed-form stability skeleton + near-boundary
sampler (design: docs/design/phoenix_seed_sampler_spec.md, Phase A: prompts/phoenix_phase_a.md).

A phoenix *seed* is a point in parameter space `(c, p, z_{-1})`. The Julia recipe
("sample near ∂M") has no literal analog — phoenix is an invertible complex Hénon map
with no critical point — but the fixed-point / period-2 **neutral-stability skeleton**
is an exact closed-form boundary to sample near, per `p`. Diversity comes from OPENING
the axes `p` (complex), `c` (complex), and especially `z_{-1}` (the slice coordinate),
not from sample count.

This module is the CPU-side sampler only (Phase A). The spec §5 surrogate / fertility
memory loop is DEFERRED — nothing here caches or ranks seeds; §5.2 cheap features are
COMPUTED and LOGGED per proposal for a later surrogate, consumed by nothing yet.

Closed forms (all collapse to the corresponding Mandelbrot feature at `p = 0`):
  * fixed points solve `z² + (p-1)z + c = 0`; multipliers solve `λ² - 2zλ - p = 0`
    (so `λ₁λ₂ = -p`, `λ₁+λ₂ = 2z`).
  * cardioid analog   `z(θ)=½(e^{iθ} - p e^{-iθ})`, `c(θ)=z(1 - p - z)`.
  * period-2 analog   `c(θ)=¼(e^{iθ} + p² e^{-iθ} - 2p) - (p-1)²`.
  * bulb/root points  cardioid `c` at rational `θ = 2π·q` (offset 0).
Sample near the skeleton: `c = c(θ) + offset · n̂(θ)`, `n̂` the outward unit normal.

Reproducible via a seeded `numpy` Generator throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Classic Ushiki phoenix — the legacy fixed instance + the sampler's identity anchor.
# c ≈ 0.5667 sits ~+0.0042 past the p=-0.5 cardioid cusp (9/16 = 0.5625) along the real
# axis: the phoenix equivalent of `c = 0.25 + ε` just outside ∂M's main cardioid.
# --------------------------------------------------------------------------- #
USHIKI_C = complex(0.5667, 0.0)
USHIKI_P = complex(-0.5, 0.0)
USHIKI_Z_M1 = complex(0.0, 0.0)

BRANCHES = ("cardioid", "period2", "root")
_BRANCH_ID = {b: i for i, b in enumerate(BRANCHES)}


# --------------------------------------------------------------------------- #
# §2.1 Fixed points and multipliers.
# --------------------------------------------------------------------------- #
def fixed_points(c: complex, p: complex) -> tuple[complex, complex]:
    """The two fixed points z (of both the 1-var recurrence and the Hénon map's diagonal),
    solving `z² + (p-1)z + c = 0`."""
    b = p - 1.0
    disc = np.sqrt(complex(b * b - 4.0 * c))
    return ((-b + disc) / 2.0, (-b - disc) / 2.0)


def fixed_point_multipliers(z: complex, p: complex) -> tuple[complex, complex]:
    """The two multipliers λ at a fixed point z (eigenvalues of the Hénon Jacobian
    `[[2z, p],[1, 0]]`), solving `λ² - 2zλ - p = 0`. Their product is `-p`."""
    root = np.sqrt(complex(z * z + p))
    return (complex(z + root), complex(z - root))


# --------------------------------------------------------------------------- #
# §2.2 / §2.3 the closed-form skeleton curves. Each takes `θ` (boundary phase, the
# neutral multiplier `e^{iθ}`) and returns the boundary parameter `c`.
# --------------------------------------------------------------------------- #
def cardioid_z(p: complex, theta: float) -> complex:
    e = complex(math.cos(theta), math.sin(theta))
    return 0.5 * (e - p / e)


def cardioid_c(p: complex, theta: float) -> complex:
    """Primary-component (cardioid-analog) boundary. p=0 -> `μ/2 - μ²/4`, μ=e^{iθ}
    (M's main cardioid). θ=0 cusp -> `¼(1-p)²`."""
    z = cardioid_z(p, theta)
    return complex(z * (1.0 - p - z))


def period2_c(p: complex, theta: float) -> complex:
    """Period-2-analog boundary (the '-1 disk'). p=0 -> `e^{iθ}/4 - 1` (center -1, r ¼)."""
    e = complex(math.cos(theta), math.sin(theta))
    return complex(0.25 * (e + p * p / e - 2.0 * p) - (p - 1.0) ** 2)


def cusp(p: complex) -> complex:
    """Cardioid cusp `c = ¼(1-p)²` (θ=0, λ=1). p=-0.5 -> 9/16."""
    return cardioid_c(p, 0.0)


def root_point(p: complex, q: float) -> complex:
    """Root point of the period-`k` bulb at rational `q = m/k`: the cardioid boundary at
    `θ = 2π q` (the neutral multiplier `e^{2πi q}`), offset 0. `q=½` is the cardioid /
    period-2 tangency (bulb root; p=0 -> c=-¾)."""
    return cardioid_c(p, 2.0 * math.pi * q)


def _branch_fn(branch: str) -> Callable[[complex, float], complex]:
    if branch in ("cardioid", "root"):
        return cardioid_c
    if branch == "period2":
        return period2_c
    raise ValueError(f"unknown branch {branch!r}")


# --------------------------------------------------------------------------- #
# §2.5 outward unit normal. The curve `c(θ)` for a fixed `p` is a simple closed loop;
# its centroid lies inside the component, so `c(θ) - centroid` points outward and picks
# the sign of the tangent-rotated-90° normal. (Rigorous enough for the near-boundary
# regime; a self-intersecting curve at large |p| is out of the Phase-A envelope.)
# --------------------------------------------------------------------------- #
def _curve_centroid(cfun: Callable[[complex, float], complex], p: complex, m: int = 512) -> complex:
    ths = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
    pts = np.array([cfun(p, float(t)) for t in ths])
    return complex(pts.mean())


def outward_normal(branch: str, p: complex, theta: float, delta: float = 1e-5,
                   centroid: Optional[complex] = None) -> complex:
    """Outward unit normal `n̂(θ)` to the branch curve at `θ`. Tangent by central
    difference, rotated 90°, sign chosen to agree with `c(θ) - centroid` (outward)."""
    cfun = _branch_fn(branch)
    if centroid is None:
        centroid = _curve_centroid(cfun, p)
    tan = cfun(p, theta + delta) - cfun(p, theta - delta)
    tabs = abs(tan)
    if tabs == 0.0:                       # degenerate (cusp): fall back to radial-from-centroid
        radial = cardioid_c(p, theta) if branch != "period2" else period2_c(p, theta)
        radial = radial - centroid
        return radial / abs(radial) if abs(radial) > 0 else complex(1.0, 0.0)
    n = complex(tan.imag, -tan.real) / tabs      # rotate tangent by -90°
    here = cfun(p, theta)
    if ((here - centroid).real * n.real + (here - centroid).imag * n.imag) < 0.0:
        n = -n
    return n


# --------------------------------------------------------------------------- #
# §4 / §5.2 cheap features. The `mandphoenix` (z₀=z_{-1}=0) escape field boundary
# distance is a closed-form-adjacent fertility PRIOR — logged, never a hard gate.
# --------------------------------------------------------------------------- #
def mandphoenix_boundary_distance(c: complex, p: complex, maxiter: int = 256,
                                  bailout: float = 1e3) -> tuple[bool, float, int]:
    """The pseudo-Mandelbrot proxy (§4): iterate `z_{n+1}=z_n²+c+p z_{n-1}` from
    `z₀=z_{-1}=0`, carrying `dz=∂z/∂c`. Returns `(escaped, de, n)` where `de` is the
    exterior distance estimate `|z|·ln|z|/|dz|` at escape (0.0 for a bounded/interior
    point). NOT a connectedness locus (z=0 is not critical) — a heuristic prior only."""
    z = zprev = 0j
    dz = dzprev = 0j
    for n in range(1, maxiter + 1):
        z_next = z * z + c + p * zprev
        dz_next = 2.0 * z * dz + 1.0 + p * dzprev   # ∂/∂c (parameter plane, carries the +1)
        zprev, dzprev = z, dz
        z, dz = z_next, dz_next
        az = abs(z)
        if az > bailout:
            de = 0.0 if abs(dz) == 0.0 else az * math.log(az) / abs(dz)
            return True, float(de), n
    return False, 0.0, maxiter


def nearest_root_distance(c: complex, p: complex,
                          qs: tuple[float, ...] = (0.5, 1.0 / 3.0, 2.0 / 3.0,
                                                   0.25, 0.75, 0.2, 0.4, 0.6, 0.8)) -> float:
    """Distance to the nearest sampled bulb root point (§5.2 feature)."""
    return float(min(abs(c - root_point(p, q)) for q in qs))


# --------------------------------------------------------------------------- #
# Seed record + cheap-feature vector.
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    c: complex
    p: complex
    z_m1: complex
    branch: str
    theta: float
    offset: float
    classic: bool = False
    features: dict = field(default_factory=dict)


def cheap_features(c: complex, p: complex, z_m1: complex, branch: str, theta: float,
                   offset: float) -> dict:
    """The §5.2 feature vector for a proposal — geometry-only + the mandphoenix prior.
    LOGGED for a later surrogate; consumed by nothing in Phase A."""
    escaped, de, n = mandphoenix_boundary_distance(c, p)
    return {
        "mandphoenix_escaped": bool(escaped),
        "mandphoenix_de": de,
        "mandphoenix_iters": int(n),
        "root_dist": nearest_root_distance(c, p),
        "abs_offset": abs(offset),
        "abs_p": abs(p),
        "arg_p": math.atan2(p.imag, p.real),
        "theta": theta,
        "branch_id": _BRANCH_ID[branch],
        "abs_z_m1": abs(z_m1),
    }


# --------------------------------------------------------------------------- #
# The proposal sampler. `(p, branch, θ, offset, z_{-1}) -> (c, p, z_{-1})`.
# --------------------------------------------------------------------------- #
def _draw_p(rng: np.random.Generator, classic: bool) -> complex:
    if classic:
        # classic real-p sub-mode: p on the real axis in (-1, 0) (the Ushiki regime).
        return complex(float(rng.uniform(-0.95, -0.05)), 0.0)
    # mostly |p| < 1, uniform in the disk (radius^0.5 for area-uniformity); the necessary
    # condition for an attracting fixed point. Complex is the biggest untapped diversity axis.
    r = math.sqrt(float(rng.uniform(0.0, 1.0))) * 0.95
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    return complex(r * math.cos(ang), r * math.sin(ang))


def _draw_offset(rng: np.random.Generator, scale: float, heavy_tail_p: float) -> float:
    """Half-normal near 0 (the 'just past the boundary' dial) with an occasional heavier
    excursion for deeper exploration. Always >= 0 (outward)."""
    if float(rng.uniform(0.0, 1.0)) < heavy_tail_p:
        return abs(float(rng.normal(0.0, scale * 6.0)))
    return abs(float(rng.normal(0.0, scale)))


def _draw_z_m1(rng: np.random.Generator, scale: float, excursion_p: float) -> complex:
    """z_{-1} as small complex offsets from 0, with an occasional larger excursion. A
    non-zero (esp. non-real) z_{-1} breaks the slice symmetry (see the render-side guard
    docs/findings/phoenix_z_m1_symmetry.md) — the single largest per-(c,p) variety lever."""
    s = scale * 8.0 if float(rng.uniform(0.0, 1.0)) < excursion_p else scale
    return complex(float(rng.normal(0.0, s)), float(rng.normal(0.0, s)))


def propose_seed(rng: np.random.Generator, *, classic_p: float = 0.05,
                 root_p: float = 0.12, period2_p: float = 0.25,
                 offset_scale: float = 0.02, offset_heavy_tail_p: float = 0.15,
                 z_m1_scale: float = 0.05, z_m1_excursion_p: float = 0.2,
                 z_m1_zero_p: float = 0.15) -> Seed:
    """Draw one seed. The classic real-c/real-p/z_{-1}=0 case is ONE named low-probability
    sub-mode (`classic_p`), never the bulk of the draw. Branch mix: root points (offset 0,
    exact rational multipliers) `root_p`, period-2 `period2_p`, else cardioid."""
    classic = float(rng.uniform(0.0, 1.0)) < classic_p
    p = _draw_p(rng, classic)

    u = float(rng.uniform(0.0, 1.0))
    if u < root_p:
        branch = "root"
    elif u < root_p + period2_p:
        branch = "period2"
    else:
        branch = "cardioid"

    if branch == "root":
        # exact root point: a rational q = m/k, offset 0.
        k = int(rng.integers(2, 8))
        m = int(rng.integers(1, k))
        theta = 2.0 * math.pi * (m / k)
        offset = 0.0
    else:
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        offset = 0.0 if classic else _draw_offset(rng, offset_scale, offset_heavy_tail_p)

    cfun = _branch_fn(branch)
    c0 = cfun(p, theta)
    if offset != 0.0:
        c = c0 + offset * outward_normal(branch, p, theta)
    else:
        c = c0

    if classic:
        z_m1 = 0j
    elif float(rng.uniform(0.0, 1.0)) < z_m1_zero_p:
        z_m1 = 0j                          # keep a slice of exact-symmetry z_{-1}=0 draws
    else:
        z_m1 = _draw_z_m1(rng, z_m1_scale, z_m1_excursion_p)

    feats = cheap_features(c, p, z_m1, branch, theta, offset)
    return Seed(c=c, p=p, z_m1=z_m1, branch=branch, theta=theta, offset=offset,
                classic=classic, features=feats)


def propose_batch(seed: int, n: int, **kw) -> list[Seed]:
    """`n` reproducible proposals from a seeded RNG (the sole entropy source)."""
    rng = np.random.default_rng(seed)
    return [propose_seed(rng, **kw) for _ in range(n)]


def seed_to_record(s: Seed) -> dict:
    """Flat jsonl record for a proposal (identity + provenance + logged cheap features)."""
    return {
        "phoenix_c_re": s.c.real, "phoenix_c_im": s.c.imag,
        "phoenix_p_re": s.p.real, "phoenix_p_im": s.p.imag,
        "phoenix_zm1_re": s.z_m1.real, "phoenix_zm1_im": s.z_m1.imag,
        "branch": s.branch, "theta": s.theta, "offset": s.offset, "classic": s.classic,
        "features": s.features,
    }


# --------------------------------------------------------------------------- #
# CLI: a seeded dry run + diagnostic summary (the Phase-A acceptance eyeball).
# --------------------------------------------------------------------------- #
def _summary(seeds: list[Seed]) -> dict:
    import collections
    ps = np.array([[s.p.real, s.p.imag] for s in seeds])
    zs = np.array([[s.z_m1.real, s.z_m1.imag] for s in seeds])
    offs = np.array([s.offset for s in seeds])
    thetas = np.array([s.theta for s in seeds])
    absp = np.hypot(ps[:, 0], ps[:, 1])
    des = np.array([s.features["mandphoenix_de"] for s in seeds])
    esc = np.array([s.features["mandphoenix_escaped"] for s in seeds])
    return {
        "n": len(seeds),
        "branch_counts": dict(collections.Counter(s.branch for s in seeds)),
        "classic_frac": float(np.mean([s.classic for s in seeds])),
        "p": {"abs_min": float(absp.min()), "abs_max": float(absp.max()),
              "frac_in_unit_disk": float(np.mean(absp < 1.0)),
              "frac_complex": float(np.mean(np.abs(ps[:, 1]) > 1e-9))},
        "z_m1": {"abs_max": float(np.hypot(zs[:, 0], zs[:, 1]).max()),
                 "frac_zero": float(np.mean(np.hypot(zs[:, 0], zs[:, 1]) < 1e-12)),
                 "frac_nonreal": float(np.mean(np.abs(zs[:, 1]) > 1e-9))},
        "offset": {"min": float(offs.min()), "max": float(offs.max()),
                   "frac_zero": float(np.mean(offs < 1e-12))},
        "theta": {"min": float(thetas.min()), "max": float(thetas.max())},
        "mandphoenix": {"frac_escaped": float(esc.mean()),
                        "de_min": float(des.min()), "de_max": float(des.max())},
    }


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Phoenix seed sampler dry run + diagnostic.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=str, default=None,
                    help="write proposals as jsonl (default: stdout summary only)")
    args = ap.parse_args(argv)
    seeds = propose_batch(args.seed, args.n)
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for s in seeds:
                f.write(json.dumps(seed_to_record(s)) + "\n")
    print(json.dumps(_summary(seeds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
