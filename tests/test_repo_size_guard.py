"""Repo-size guard: no large-in-tree bloat without an explicit registry entry.

The standing constraint is that the working tree stays ~what git tracks — source +
irreplaceable metadata + `out/`. Anything large in-tree needs a written-down reason.
`tools/audit/size_guard.py` scans the *filesystem* (not `git ls-files`: a gitignored
file can bloat the tree while invisible to git) and flags every file >= 1 MiB and
every many-small-file directory >= ~100 MB, then checks each flagged violator against
the `REGISTRY` allowlist.

Two assertions with different severities:

  * HARD FAIL — any flagged violator not covered by a registry entry. This catches
    NEW bloat the moment it lands: add 300 MB of un-registered crops and this goes
    red, naming the path. To make it green you either delete the bloat or add a
    deliberate registry line stating why it stays. (Proven to go red on purpose:
    drop a >=1 MiB file outside the excluded prefixes → fail → remove → green.)

  * SOFT REPORT — a registry entry that no longer covers any over-threshold content
    (its bulk relocated / was deleted). Emitted as a warning, NOT a failure: it's a
    nudge to delete the stale line, and the guard fully enforces only once every
    RELOCATE line is gone and just KEEP lines remain.

This runs under default `pytest`: filesystem walk + `git` only, no release binary,
no GPU, no corpus reads. Companion to `test_tracked_artifacts.py` (which guards
*de-tracking* of a static canary list) — this guards *bloat* of the live tree.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_MOD_PATH = REPO_ROOT / "tools" / "audit" / "size_guard.py"

# Load size_guard.py by path (tools/ is not an installed package). Register in
# sys.modules before exec so its @dataclass field annotations resolve.
_spec = importlib.util.spec_from_file_location("size_guard", _MOD_PATH)
size_guard = importlib.util.module_from_spec(_spec)
sys.modules["size_guard"] = size_guard
_spec.loader.exec_module(size_guard)


@pytest.fixture(scope="module")
def result():
    return size_guard.check_registry(size_guard.scan(REPO_ROOT))


def test_registry_nonempty():
    """Guard the guard: an emptied REGISTRY would make the coverage assertion pass
    vacuously (nothing to fail against)."""
    assert size_guard.REGISTRY, "REGISTRY is empty — the size guard would pass vacuously"


def test_no_uncovered_violators(result):
    """HARD: every flagged large file / dir is covered by a registry entry."""
    if result.uncovered:
        lines = "\n".join(
            f"    {size_guard.human(v.size):>9}  {v.rel}" for v in result.uncovered
        )
        pytest.fail(
            "REPO-SIZE GUARD TRIPPED: large-in-tree content with no registry entry:\n"
            f"{lines}\n"
            "New bulk landed in the working tree. Either move it out (regenerable ->\n"
            "artifacts, trained binary -> precious-store, dead -> trash) or, if it truly\n"
            "belongs in-tree, add a deliberate KEEP line to REGISTRY in\n"
            "tools/audit/size_guard.py stating why. Do NOT widen an existing prefix\n"
            "just to silence this."
        )


def test_report_stale_registry_entries(result):
    """SOFT: warn (do not fail) on registry entries that no longer cover any
    over-threshold content — a nudge to delete the line as relocations land."""
    if result.stale:
        stale = ", ".join(e.prefix for e in result.stale)
        warnings.warn(
            f"{len(result.stale)} stale size-guard registry entr"
            f"{'y' if len(result.stale) == 1 else 'ies'} (no over-threshold content — "
            f"delete the line(s)): {stale}",
            stacklevel=2,
        )
