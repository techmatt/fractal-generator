"""Reappearance tripwire + resolver unit tests for the relocated-artifacts move.

The overnight storage restructure moved four regenerable file-count bombs
(aug_cache x4 versions, campaign2 breadth/dive scratch) OUT of the working tree
to ARTIFACTS_ROOT, routing readers/writers through ``tools/corpus/artifacts.py``.

Two guarantees are pinned here:

1. **Reappearance tripwire** (`test_no_relocated_root_repopulated_in_tree`): if a
   missed *writer* silently re-materializes a relocated family under its old
   in-tree path, this goes RED and names the offender. This is the backstop the
   grep-completeness sweep can't provide.

2. **Resolver correctness**: relocated prefixes map under ARTIFACTS_ROOT, sibling
   look-alikes and every other path stay in-tree, and the env override works.

Run: ``uv run pytest tools/audit/test_relocated_artifacts.py``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "corpus"))
import artifacts as A  # noqa: E402


def _count_files(p: Path, cap: int = 5) -> int:
    """Count up to `cap` regular files under p (cheap; we only need >0 vs 0)."""
    if not p.exists():
        return 0
    n = 0
    for _root, _dirs, files in os.walk(p):
        n += len(files)
        if n >= cap:
            break
    return n


def test_no_relocated_root_repopulated_in_tree():
    """No relocated family may hold real files under its OLD in-tree path.

    Empty leftover dirs are tolerated (a move can leave the parent behind); real
    files mean a writer bypassed the resolver and re-bombed the tree."""
    offenders = []
    for prefix in A.RELOCATED_PREFIXES:
        in_tree = REPO_ROOT / prefix
        n = _count_files(in_tree)
        if n:
            offenders.append(f"{prefix} has {n}+ files in-tree at {in_tree}")
    assert not offenders, (
        "Relocated artifact family repopulated in the working tree (a writer "
        "bypassed tools/corpus/artifacts.resolve): " + "; ".join(offenders)
    )


def test_relocated_prefixes_map_under_artifacts_root():
    root = A.artifacts_root()
    for prefix in A.RELOCATED_PREFIXES:
        sample = f"{prefix}/sub/file.jpg"
        assert A.is_relocated(sample)
        resolved = A.resolve(sample)
        assert resolved == root / sample, (prefix, resolved)


def test_sibling_lookalike_stays_in_tree():
    # a sibling that merely shares a prefix string must NOT be relocated
    assert not A.is_relocated("data/v4/aug_cache_notes/x.txt")
    assert A.resolve("data/v4/aug_cache_notes/x.txt") == A.REPO_ROOT / "data/v4/aug_cache_notes/x.txt"


def test_non_relocated_paths_resolve_in_tree():
    for p in ["data/v4/cache_manifest.jsonl",
              "data/label_corpus/batches/b/images.jsonl",
              "data/discovery/campaign2/breadth/outcome_ledger.jsonl"]:
        assert not A.is_relocated(p)
        assert A.resolve(p) == A.REPO_ROOT / p


def test_backslash_and_dotslash_normalized():
    assert A.is_relocated("data\\v4\\aug_cache\\1\\x.jpg")
    assert A.is_relocated("./data/v4/aug_cache/1/x.jpg")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(A.ARTIFACTS_ENV, str(tmp_path))
    assert A.artifacts_root() == tmp_path
    assert A.resolve("data/v4/aug_cache/1/x.jpg") == tmp_path / "data/v4/aug_cache/1/x.jpg"
