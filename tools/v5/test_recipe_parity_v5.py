"""The v5 recipe-parity gate, wired as a pytest test.

`tools/v5/build_plan.py`'s RECIPE-PARITY GATE regenerates the Mandelbrot cache rows
from the committed recipe and asserts they are byte-identical to the on-disk
`data/v4/cache_manifest.jsonl` — the guarantee that every v5/v6 location-classifier
build reuses the frozen v4 `aug_cache` JPGs verbatim. It used to fire only when
someone manually ran `build_plan.py`; this makes `pytest` run it.

Skipped (not failed) when the v4/v5 manifest inputs aren't on disk — they live under
the gitignored `data/` trees, so a fresh checkout legitimately can't run this. On a
machine with the corpus it runs, and any recipe drift goes red naming the drifted key.

The sibling `build_plan.py` is loaded under a unique module name (not `import
build_plan`) so it can't collide in `sys.modules` with the v4/v6 modules of the same
basename.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load_build_plan():
    spec = importlib.util.spec_from_file_location("v5_build_plan", _HERE / "build_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load_build_plan()
_INPUTS = [bp.ROSTER, bp.MANIFEST, bp.V4_CACHE_MANIFEST]


@pytest.mark.skipif(
    not all(p.exists() for p in _INPUTS),
    reason="v4/v5 manifest inputs absent (gitignored data/ corpus) — build the corpus to run",
)
def test_v5_recipe_parity():
    """Regenerated Mandelbrot cache rows == committed data/v4/cache_manifest.jsonl,
    byte-for-byte. Raises inside verify_recipe_parity() on any drift."""
    n = bp.verify_recipe_parity()
    assert n > 0, "recipe parity produced zero rows — inputs empty?"
