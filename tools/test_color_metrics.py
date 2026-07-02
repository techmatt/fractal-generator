"""Tests for the shared perceptual-color primitives (tools/queries/color_metrics.py).

Run:  uv run pytest tools/test_color_metrics.py -v

The load-bearing check is the CIEDE2000 self-test against the Sharma et al. 2005
reference vectors (30/31 exact to 1e-3, with the one documented hue-quadrant
boundary pair allowed <0.05). This gates the relocation: a broken ΔE would
silently corrupt render-space candidate selection.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "queries"))
import color_metrics as cmet  # noqa: E402


def test_ciede2000_sharma_reference():
    """30/31 Sharma refs exact to 1e-3; the one hue-quadrant boundary pair <0.05."""
    err2nd, errworst = cmet._validate_ciede2000()
    assert err2nd <= 1e-3, f"2nd-worst err {err2nd:.5f} exceeds 1e-3"
    assert errworst <= 0.05, f"worst err {errworst:.5f} exceeds 0.05 boundary allowance"


if __name__ == "__main__":
    e2, ew = cmet._validate_ciede2000()
    print(f"CIEDE2000 self-test PASS (2nd-worst {e2:.2e}, worst {ew:.3f})")
