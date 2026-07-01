"""One place that puts the tools/ subdirs on sys.path.

There is no package structure under tools/: every module is run BOTH as a
`uv run python tools/<sub>/<mod>.py` script AND collected by pytest, so imports are
position-dependent and each module has historically opened with 1-3 hand-rolled
`sys.path.insert` lines re-deriving the same sibling dirs. This module holds that list
in ONE place. A caller that already has `tools/` on sys.path just does `import
_bootstrap` (idempotent) to make `tools/{palettes,corpus,queries}` importable by bare
name.

Deliberately NOT a package `__init__.py`: converting to true packages would force
`python -m tools...` invocation and break the `python tools/x/y.py` script entry
points the CLI docs + tests rely on. This is the low-risk middle ground. Untouched
modules (mining/, eda/, v4/, v5/, …) still carry their own inserts — migrating them is
a follow-up.
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "palettes", "corpus", "queries"):
    _d = os.path.join(_TOOLS, _sub) if _sub else _TOOLS
    if _d not in sys.path:
        sys.path.insert(0, _d)
