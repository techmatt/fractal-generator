"""Canary: the release binary MUST be present, so the parity layer is armed.

Much of the Rust↔Python parity suite (`tools/test_acceptance.py`,
`tools/test_colormap.py`, `tools/atlas/test_julia_bands_parity.py`,
`tools/corpus/test_location.py`, …) `skipif`s itself when
`target/release/fractal-generator.exe` is absent. That is correct *per test* —
those tests genuinely cannot run without a build, and forcing every one of them
to fail would make the suite unrunnable on a checkout that legitimately doesn't
need a binary. But it means a no-build checkout produces N green skips: the
parity layer measures nothing while reporting all-clear.

This single test converts that fleet of invisible skips into ONE loud red line
that says *the parity layer is off, build first* — without touching any of the
per-test `skipif`s. It is the widest version of the same failure the tracked-
artifact canary guards: a guard that passes while checking nothing.

It cannot itself skip (that would reintroduce the silence), and it guards its own
target-path constant so a typo can't quietly make it vacuous.

Runs under default `pytest`: no GPU, no corpus. Only checks a file exists.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The parity tests hardcode the .exe name (Windows); mirror that, but resolve the
# platform suffix so this canary is correct if the suite is ever run elsewhere.
_EXE_NAME = "fractal-generator.exe" if sys.platform == "win32" else "fractal-generator"
RELEASE_BINARY = REPO_ROOT / "target" / "release" / _EXE_NAME


def test_release_binary_path_under_target():
    """Guard the guard: if the path constant is edited to something outside the
    release tree, the presence check below stops meaning anything."""
    assert RELEASE_BINARY.parent == REPO_ROOT / "target" / "release", (
        f"RELEASE_BINARY is not under target/release/: {RELEASE_BINARY}"
    )
    assert RELEASE_BINARY.name.startswith("fractal-generator"), (
        f"RELEASE_BINARY is not the engine binary: {RELEASE_BINARY.name}"
    )


def test_release_binary_present():
    assert RELEASE_BINARY.is_file(), (
        f"PARITY LAYER OFF: release binary not built:\n"
        f"    {RELEASE_BINARY}\n"
        f"Every Rust↔Python parity test skips without it. Build it:\n"
        f"    cargo build --release\n"
        f"(This canary is intentionally un-skippable — a green skip here is the "
        f"failure it exists to catch.)"
    )
