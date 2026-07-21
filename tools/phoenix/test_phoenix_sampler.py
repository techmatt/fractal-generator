"""Spec §7 correctness net for the phoenix seed sampler + a seeded dry-run sanity gate.

  uv run pytest tools/phoenix/test_phoenix_sampler.py

Covers: the p=0 collapses to the Mandelbrot cardioid / period-2 disk (~1e-12), the p=-0.5
cardioid cusp = 9/16 with the classic Ushiki seed ~+0.0042 outside it, the multiplier-
product identities (λ₁λ₂=-p at a fixed point; period-2 cycle-multiplier product = p²), the
outward-normal direction, and that a seeded batch explores the axes rather than pinning
them. The z_{-1} symmetry guard is a RENDER-level property and lives in the Rust suite
(`render_modes::tests::phoenix_z_m1_symmetry_guard`); a binary-backed check here mirrors it
when the release binary is present (see docs/findings/phoenix_z_m1_symmetry.md).
"""
from __future__ import annotations

import cmath
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import phoenix_sampler as psamp  # noqa: E402

TWO_PI = 2.0 * math.pi
THETAS = [0.0, 0.3, 1.0, 2.0, math.pi, 4.0, 5.5, 6.1]


# --------------------------------------------------------------------------- #
# §7: p=0 collapses to the Mandelbrot features (~1e-12).
# --------------------------------------------------------------------------- #
def test_p_zero_cardioid_is_main_cardioid():
    for th in THETAS:
        mu = cmath.exp(1j * th)
        assert abs(psamp.cardioid_c(0.0, th) - (mu / 2.0 - mu * mu / 4.0)) < 1e-12


def test_p_zero_period2_is_unit_disk():
    for th in THETAS:
        mu = cmath.exp(1j * th)
        assert abs(psamp.period2_c(0.0, th) - (mu / 4.0 - 1.0)) < 1e-12


def test_p_zero_bulb_root_is_minus_three_quarters():
    # cardioid/period-2 tangency (q=½, λ=-1) at p=0 is c = -¾, M's period-2 root.
    assert abs(psamp.root_point(0.0, 0.5) - (-0.75)) < 1e-12


# --------------------------------------------------------------------------- #
# §7: p=-0.5 cardioid cusp = 9/16, classic Ushiki ~+0.0042 outside along the real axis.
# --------------------------------------------------------------------------- #
def test_cusp_at_p_minus_half_is_nine_sixteenths():
    assert abs(psamp.cusp(-0.5) - 0.5625) < 1e-12
    assert abs(psamp.cusp(-0.5).imag) < 1e-12
    # general cusp closed form c = ¼(1-p)².
    for p in (0.0, -0.5, 0.3 + 0.2j, -0.7j):
        assert abs(psamp.cusp(p) - 0.25 * (1.0 - p) ** 2) < 1e-12


def test_ushiki_seed_is_just_outside_the_cusp():
    cusp = psamp.cusp(-0.5).real                       # 0.5625
    ushiki = psamp.USHIKI_C.real                       # 0.5667
    assert abs((ushiki - cusp) - 0.0042) < 1e-9        # ~+0.0042 along the real axis
    assert ushiki > cusp                               # OUTSIDE the component (larger real c)
    # and a small positive real offset from the cusp reconstructs the Ushiki c.
    assert abs((cusp + 0.0042) - ushiki) < 1e-9


# --------------------------------------------------------------------------- #
# §7: multiplier-product identities.
# --------------------------------------------------------------------------- #
def test_fixed_point_multiplier_product_is_minus_p():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
        p = complex(rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9))
        for z in psamp.fixed_points(c, p):
            # z solves z²+(p-1)z+c=0.
            assert abs(z * z + (p - 1.0) * z + c) < 1e-9
            l1, l2 = psamp.fixed_point_multipliers(z, p)
            assert abs(l1 * l2 - (-p)) < 1e-9          # λ₁λ₂ = -p
            assert abs((l1 + l2) - 2.0 * z) < 1e-9     # λ₁+λ₂ = 2z


def test_period2_cycle_multiplier_product_is_p_squared():
    # Build a genuine 2-cycle {z1,z2} (z1+z2=p-1, z1z2=c+(p-1)²), form the cycle Jacobian
    # J = DF(z2)·DF(z1) with DF(z)=[[2z,p],[1,0]], and check its eigenvalue product = det J
    # = p² and trace = 4 z1z2 + 2p (the spec §2.3 relation Λ + p²/Λ = 4z1z2 + 2p).
    rng = np.random.default_rng(2)
    for _ in range(50):
        c = complex(rng.uniform(-1.5, 0.2), rng.uniform(-0.6, 0.6))
        p = complex(rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9))
        s = p - 1.0                                    # z1 + z2
        prod = c + (p - 1.0) ** 2                      # z1 z2
        disc = cmath.sqrt(s * s - 4.0 * prod)
        z1, z2 = (s + disc) / 2.0, (s - disc) / 2.0
        if abs(z1 - z2) < 1e-6:
            continue                                   # skip the degenerate (fixed-point) case
        DF = lambda z: np.array([[2.0 * z, p], [1.0, 0.0]], dtype=complex)
        J = DF(z2) @ DF(z1)
        eig = np.linalg.eigvals(J)
        assert abs(np.prod(eig) - p * p) < 1e-7        # Λ₁Λ₂ = p²
        assert abs(np.trace(J) - (4.0 * prod + 2.0 * p)) < 1e-9


# --------------------------------------------------------------------------- #
# Outward normal — points away from the component (Ushiki direction).
# --------------------------------------------------------------------------- #
def test_outward_normal_points_outward_at_ushiki():
    # near the p=-0.5 cusp (θ small), the outward normal has a positive real component
    # (the Ushiki seed sits at the cusp + a positive real offset).
    n = psamp.outward_normal("cardioid", -0.5, 0.05)
    assert n.real > 0.0
    assert abs(abs(n) - 1.0) < 1e-9                    # unit


def test_outward_offset_increases_mandphoenix_escape():
    # a point pushed OUTWARD past the boundary should be no closer to bounded than the
    # boundary point: escapes at least as readily (sanity on the outward sign, statistically).
    rng = np.random.default_rng(3)
    outward_escapes = 0
    trials = 40
    for _ in range(trials):
        th = float(rng.uniform(0.2, TWO_PI - 0.2))
        p = complex(rng.uniform(-0.6, 0.0), 0.0)
        c_b = psamp.cardioid_c(p, th)
        c_out = c_b + 0.05 * psamp.outward_normal("cardioid", p, th)
        esc, _, _ = psamp.mandphoenix_boundary_distance(c_out, p)
        outward_escapes += int(esc)
    assert outward_escapes >= int(0.7 * trials)        # outward mostly lands in the exterior


# --------------------------------------------------------------------------- #
# Seeded dry run — reproducibility + axis coverage (the acceptance eyeball, asserted).
# --------------------------------------------------------------------------- #
def test_batch_is_reproducible():
    a = psamp.propose_batch(7, 64)
    b = psamp.propose_batch(7, 64)
    assert [psamp.seed_to_record(s) for s in a] == [psamp.seed_to_record(s) for s in b]
    c = psamp.propose_batch(8, 64)
    assert [s.c for s in a] != [s.c for s in c]


def test_batch_explores_the_axes():
    seeds = psamp.propose_batch(0, 400)
    s = psamp._summary(seeds)
    assert set(s["branch_counts"]) == {"cardioid", "period2", "root"}   # all branches drawn
    assert 0.0 < s["classic_frac"] < 0.2                                # a low-prob sub-mode
    assert s["p"]["frac_complex"] > 0.5                                 # p opened to complex
    assert s["p"]["frac_in_unit_disk"] == 1.0                           # mostly |p|<1
    assert s["z_m1"]["frac_nonreal"] > 0.5                              # z_{-1} opened (non-real)
    assert s["z_m1"]["frac_zero"] > 0.0                                 # ...but keeps some z_{-1}=0
    assert s["offset"]["frac_zero"] > 0.0                               # exact root points present
    assert s["theta"]["max"] - s["theta"]["min"] > 5.0                  # θ spans the circle


# --------------------------------------------------------------------------- #
# z_{-1} symmetry guard — RENDER-backed, mirrors the Rust unit test. Skipped when the
# release binary is absent (like the corpus phoenix acceptance gate). z_{-1}=0 (real c,p)
# renders with EXACT real-axis reflection; a non-real z_{-1} breaks it. This is what stops
# anyone silently re-pinning z_{-1}. See docs/findings/phoenix_z_m1_symmetry.md.
# --------------------------------------------------------------------------- #
def _bin():
    exe = ROOT / "target" / "release" / ("fractal-generator.exe" if os.name == "nt"
                                         else "fractal-generator")
    return exe if exe.exists() else None


def _render_gray(tmp, z1):
    from PIL import Image
    out = tmp / f"ph_{z1[0]}_{z1[1]}.png"
    cmd = [str(_bin()), "render-one", "--family", "phoenix",
           "--c", "0.5667", "0", "--p", "-0.5", "0", "--phoenix-z1", str(z1[0]), str(z1[1]),
           "--cx", "0", "--cy", "0", "--fw", "3.0",
           "--width", "160", "--height", "160", "--supersample", "1", "--maxiter", "300",
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return np.asarray(Image.open(out).convert("L"), dtype=np.int16)


def test_render_z_m1_symmetry_guard(tmp_path):
    if _bin() is None:
        import pytest
        pytest.skip("release binary not built")
    try:
        import PIL  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("Pillow not available")
    # z_{-1}=0: exact real-axis (top-bottom) reflection symmetry.
    g0 = _render_gray(tmp_path, (0.0, 0.0))
    assert np.array_equal(g0, g0[::-1, :]), "z_{-1}=0 must render with real-axis symmetry"
    # z_{-1} with a non-zero imaginary part BREAKS the reflection (and is load-bearing).
    gz = _render_gray(tmp_path, (0.0, 0.15))
    assert not np.array_equal(gz, gz[::-1, :]), "non-real z_{-1} must break the symmetry"
    assert not np.array_equal(g0, gz), "z_{-1} must change the render (not a no-op)"
