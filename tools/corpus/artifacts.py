"""Resolver for relocated regenerable bulk artifacts.

Why this exists
---------------
~99.98% of the files a recursive tool traverses in this repo were regenerable ML
scratch (augmentation caches, per-node discovery scratch) living *inside* the
source tree, which is what made a plain ``grep -r`` take >120 s. Those two file-
count bombs were physically moved OUT of the working tree to a sibling directory
so traversal (grep/find/editor-indexers/watchers) is fast *by construction*,
independent of gitignore-awareness.

This module is the single seam that maps a **repo-relative** artifact path (the
portable, version-invariant string stored in manifests/plans) to its **real**
on-disk location. Every reader AND writer of a relocated family MUST route
through :func:`resolve` so the data is found where it now lives, and so a rebuild
never re-materializes the bomb in-tree.

ARTIFACTS_ROOT
--------------
Defaults to a *sibling* of the repo (``../fractal-generator-artifacts``), so a
fresh checkout on any machine resolves without configuration. Override with the
``FRACTAL_ARTIFACTS_ROOT`` environment variable (e.g. to point at a different
volume). The relocated tree mirrors the repo-relative layout exactly:
``<ARTIFACTS_ROOT>/data/v4/aug_cache/...`` etc.

Non-relocated paths resolve in-tree, unchanged. This module is deliberately
additive and narrow: it changes *where four specific families live*, nothing
else.
"""
from __future__ import annotations

import os
from pathlib import Path

# tools/corpus/artifacts.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_ENV = "FRACTAL_ARTIFACTS_ROOT"

# Repo-relative prefixes whose contents were relocated to ARTIFACTS_ROOT. Each is
# matched as a whole path component (exact, or followed by "/") so a sibling like
# ``data/v4/aug_cache_notes`` would NOT accidentally match ``data/v4/aug_cache``.
# Keep this list in lockstep with the reappearance tripwire
# (tools/audit/test_relocated_artifacts.py) and the .gitignore stanzas.
RELOCATED_PREFIXES = (
    "data/v4/aug_cache",
    "data/v5/aug_cache_julia",
    "data/v6/aug_cache_gather",
    "data/v7/aug_cache",
    "data/discovery/campaign2/breadth/scratch",
    "data/discovery/campaign2/dive/scratch",
)


def artifacts_root() -> Path:
    """Root under which relocated artifacts live (env override or repo sibling)."""
    env = os.environ.get(ARTIFACTS_ENV)
    if env:
        return Path(env)
    return REPO_ROOT.parent / "fractal-generator-artifacts"


def _norm(rel) -> str:
    """Normalize to a forward-slash, repo-relative string (no leading ./ or /)."""
    s = str(rel).replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def is_relocated(rel) -> bool:
    """True iff ``rel`` (repo-relative) belongs to a relocated family."""
    r = _norm(rel)
    return any(r == p or r.startswith(p + "/") for p in RELOCATED_PREFIXES)


def resolve(rel) -> Path:
    """Map a repo-relative artifact path to its real on-disk location.

    Relocated families -> ``ARTIFACTS_ROOT/<rel>``; every other path ->
    ``REPO_ROOT/<rel>`` (i.e. unchanged in-tree behavior). Accepts str or Path.
    """
    r = _norm(rel)
    base = artifacts_root() if is_relocated(r) else REPO_ROOT
    return base / r
