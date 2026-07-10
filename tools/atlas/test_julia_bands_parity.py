#!/usr/bin/env python
"""Rust <-> Python parity guard for the per-family Julia descent bands.

The band table lives twice and MUST stay bit-identical:
  - Rust:   `WalkFamily::julia_band_defaults` (src/guided_descend.rs) -- canonical.
  - Python: `production_seeder.JULIA_GATHER_BANDS` (the gather CLI override table).

Drift is a silent footgun (a `--julia` gather would over/under-reject vs the engine
defaults). This test asks the ENGINE for its table (the read-only `dump-julia-bands`
subcommand -> JSON) and compares per-degree against the Python dict. It never parses
Rust source text -- the binary is the single source of truth, so there is no third
drift surface.

  uv run pytest tools/atlas/test_julia_bands_parity.py

Requires a built release binary (target/release/fractal-generator.exe). If absent the
test SKIPS loudly (build first); on genuine value drift it FAILS with a per-key diff.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402

BIN = ROOT / "target" / "release" / "fractal-generator.exe"

# repr noise only (all real per-degree gaps are >= 0.2); anything larger is drift.
ABS_TOL = 1e-12


def _rust_bands() -> dict[str, tuple[float, float]]:
    if not BIN.exists():
        pytest.skip(f"release binary not built ({BIN}); run `cargo build --release` first")
    out = subprocess.run(
        [str(BIN), "dump-julia-bands"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, f"dump-julia-bands failed (rc={out.returncode}): {out.stderr}"
    table = json.loads(out.stdout)
    return {k: tuple(v) for k, v in table.items()}


def test_julia_bands_rust_python_parity():
    rust = _rust_bands()
    py = {k: tuple(map(float, v)) for k, v in ps.JULIA_GATHER_BANDS.items()}

    # Same set of degrees on both sides (a missing/extra family is drift too).
    assert set(rust) == set(py), (
        f"family-key mismatch: rust-only={set(rust) - set(py)} "
        f"python-only={set(py) - set(rust)}"
    )

    diffs = []
    for key in sorted(py):
        (r_esc, r_spread) = rust[key]
        (p_esc, p_spread) = py[key]
        if not (math.isclose(r_esc, p_esc, rel_tol=0.0, abs_tol=ABS_TOL)
                and math.isclose(r_spread, p_spread, rel_tol=0.0, abs_tol=ABS_TOL)):
            diffs.append(f"  {key}: rust=({r_esc}, {r_spread}) != python=({p_esc}, {p_spread})")

    assert not diffs, (
        "Julia band drift between Rust `julia_band_defaults` and Python "
        "`JULIA_GATHER_BANDS` -- retune BOTH:\n" + "\n".join(diffs)
    )
