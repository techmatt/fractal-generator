"""The coloring gate, wired as a pytest test.

`tools/colormap_acceptance.py` is the load-bearing empirical check that the Python
coloring tail (`colormap.render_candidate`) reproduces the Rust smooth render within
<=1 LSB. It has always been runnable standalone (`uv run python
tools/colormap_acceptance.py`); this test makes `uv run pytest tools/` the ONE command
that gates a coloring change — it shells out to the release binary exactly as the
standalone script does and asserts the same PASS.

Skipped (not failed) when the release binary isn't built — build it and re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colormap_acceptance as acc  # noqa: E402


@pytest.mark.skipif(not acc.BIN.exists(), reason="release binary not built")
def test_colormap_acceptance_passes():
    """The default acceptance gate (test_01, twilight, box) matches Rust within tol."""
    m = acc.run_gate()  # same defaults as `colormap_acceptance.main`
    assert m["passed"], m
