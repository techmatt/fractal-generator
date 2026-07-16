"""Canary: irreplaceable human-authored artifacts MUST stay git-tracked.

Every path below satisfies two conjuncts:

  1. human-authored judgment  — a person sat and made a decision (a label, a
     hand-picked fixture) that no code can reproduce; and
  2. no regeneration path     — there is no script that rebuilds it from
     committed inputs.

Such a file is uniquely fragile: nothing *breaks* when it stops being tracked
(a `.gitignore` edit that widens a rule, an `rm` in the wrong tree), so the loss
is silent and permanent. This test converts that silence into one loud red line.
It asserts each path is tracked via `git ls-files --error-unmatch`; if a path is
untracked or newly ignored, git exits non-zero naming the path and the assertion
fails.

Scope note — this guards *deletion / de-tracking* of a static list. It does NOT
discover newly-added irreplaceable files (that needs a glob, which by
construction cannot detect a file that is already gone). Adding a batch of human
labels is therefore a conscious edit to `TRACKED_CANARIES` below.

Runs under default `pytest`: no release binary, no GPU, no corpus reads — only
`git`. See `test_release_binary.py` for the binary-presence canary.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Irreplaceable, human-authored artifacts (human judgment ∧ no regeneration path).
#
# Deliberately excluded, and why:
#   - batch.json / probe_manifest.jsonl / schema docs — machine-emitted run
#     provenance or hand-written spec text (re-writable), not label data.
#   - jm3 / jm45 label-corpus batches — carry ZERO committed human labels in any
#     file yet (unlabeled); their images.jsonl is a machine coord record only, so
#     it fails conjunct 1. Add them here the moment they are labeled.
#   - wallpaper_corpus/*/images.jsonl — all label blocks are null; "humanq3"
#     names human-*seeded* generation, not committed human labels.
#   - GPU eval records (queries/sampler_eval), discovery ledgers, classifier
#     weights — irreplaceable but MACHINE-authored, so outside conjunct 1.
TRACKED_CANARIES = [
    # Hand-picked reference fixtures (the test locations + palette selection).
    "data/test_renders.json",
    "data/test_palettes.json",
    # Hand-labeled palette-preference tier stores.
    "data/queries/labels/coldstart_v2.json",
    "data/queries/labels/warmstart_v1.json",
    "data/queries/labels/prefv2_dramatic_v1.json",
    # Label-corpus human q3 labels. For each labeled batch we guard BOTH the
    # human labels (scores.json) AND images.jsonl — the latter holds the render
    # coords those labels dereference (scores.json keys by image_id alone; the
    # cx/cy/fw live only in images.jsonl, and the guided-descend pool that
    # produced them is not committed). Labels without their referent are useless,
    # so the pair is canaried together.
    "data/label_corpus/batches/2026-06-23_flat_generate_loose0_v3/scores.json",
    "data/label_corpus/batches/2026-06-23_flat_generate_loose0_v3/images.jsonl",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4/scores.json",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4/images.jsonl",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered/scores.json",
    "data/label_corpus/batches/2026-06-24_guided_descend_rev4occfix_v2filtered/images.jsonl",
    "data/label_corpus/batches/2026-06-25_mining_v3guided_v1/scores.json",
    "data/label_corpus/batches/2026-06-25_mining_v3guided_v1/images.jsonl",
    "data/label_corpus/batches/2026-06-25_scale_2x2_labelset/scores.json",
    "data/label_corpus/batches/2026-06-25_scale_2x2_labelset/images.jsonl",
    "data/label_corpus/batches/2026-06-25_scale_controlled_2x2/scores.json",
    "data/label_corpus/batches/2026-06-25_scale_controlled_2x2/images.jsonl",
    "data/label_corpus/batches/2026-07-05_gather_v6/scores.json",
    "data/label_corpus/batches/2026-07-05_gather_v6/images.jsonl",
    "data/label_corpus/batches/julia_ladder_j0/scores.json",
    "data/label_corpus/batches/julia_ladder_j0/images.jsonl",
    # blindspot: labels live ONLY in images.jsonl (no scores.json exists).
    "data/label_corpus/batches/2026-07-12_blindspot_v6reject_v1/images.jsonl",
]


def _git_tracked(path: str) -> tuple[bool, str]:
    """(is_tracked, stderr). `--error-unmatch` exits non-zero naming an
    untracked/ignored pathspec."""
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()


def test_guard_list_nonempty():
    """If the list is ever emptied (a bad refactor), the parametrized guard below
    would pass vacuously — so guard the guard."""
    assert TRACKED_CANARIES, "TRACKED_CANARIES is empty — the tracking guard would pass vacuously"


@pytest.mark.parametrize("path", TRACKED_CANARIES)
def test_canary_tracked(path):
    tracked, stderr = _git_tracked(path)
    assert tracked, (
        f"CANARY TRIPPED: irreplaceable human-authored artifact is not git-tracked:\n"
        f"    {path}\n"
        f"git: {stderr}\n"
        f"This file has no regeneration path. Check for a .gitignore rule that "
        f"swept it, or a deletion — do NOT delete it from the canary list to "
        f"make this green."
    )
