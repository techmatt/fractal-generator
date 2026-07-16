"""The Rust↔Python LUT-bake byte-match, wired as a pytest gate.

`palette_extractor/check_bytematch.py` is the load-bearing check that the Python
`palette_lib.coloring.bake_lut` port reproduces the Rust palette bake byte-for-byte
(`max|Δ| < 1e-12` on the linear-RGB LUT). It has always been runnable standalone
(`python palette_extractor/check_bytematch.py`); this test makes `uv run pytest` the
one command that gates a coloring/bake change. It drives the `#[ignore]`d Rust dump
test (`tests/palette_bytematch.rs`) via env vars exactly as the standalone script
does — so wiring this in also exercises that otherwise-never-run dump helper.

Skipped (not failed) when the release binary isn't built — the `test_release_binary`
canary converts that skip into one loud red line, so a no-build checkout can't hide
the fact that this gate didn't run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_bytematch as cbm  # noqa: E402

# Same skip proxy as the sibling parity gates: the release binary standing in for
# "a release build exists" (check_bytematch shells `cargo test --release`).
_EXE = "fractal-generator.exe" if sys.platform == "win32" else "fractal-generator"
BIN = ROOT / "target" / "release" / _EXE


@pytest.mark.skipif(not BIN.exists(), reason="release binary not built — run `cargo build --release`")
def test_bake_bytematch():
    """Python `bake_lut` == Rust palette bake, byte-for-byte, across cyclic +
    sequential(mirrored) maps. Fails the moment either bake drifts."""
    worst = cbm.run()
    assert worst < 1e-12, (
        f"Rust<->Python LUT bake drift: worst max|d| = {worst:.3e} exceeds 1e-12.\n"
        f"The Python palette_lib.coloring.bake_lut port no longer byte-matches the "
        f"Rust palette::Palette bake -- reconcile the two before shipping."
    )
