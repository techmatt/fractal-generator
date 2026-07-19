"""Canary: irreplaceable artifacts MUST stay git-tracked.

Every path below satisfies two conjuncts:

  1. unregenerable            — there is no script that rebuilds it from
     committed inputs (or the only "rebuild" is value-approximate under a
     verdict-sensitive threshold, which is not a rebuild); and
  2. tracked                  — it is currently in the git index, so the canary
     has something to assert about.

Earlier revisions scoped conjunct 1 to *human-authored* judgment. That was a
proxy: what actually makes a file irreplaceable is that nothing reproduces it,
not who authored it. Human labels qualify because a person's judgment has no
regen path — but so do a handful of MACHINE-authored artifacts whose producer
is gone or whose output is only value-approximate on re-run (the CLIP library
embeddings, the one committed v5 weight). The old criterion silently dropped
that whole tier, so the list now keys on unregenerability directly.

The project's own line is that *regenerable at compute cost* does NOT qualify —
discovery ledgers, GPU eval records, and every classifier weight except the v5
rollback anchor rebuild deterministically-enough from committed inputs. The
canary's value is that every entry is a deliberate opt-in; a canary guarding
everything is one nobody maintains.

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

# Unregenerable ∧ tracked artifacts (no rebuild path ∧ currently in the index).
#
# Deliberately excluded, and why:
#   - batch.json / probe_manifest.jsonl / schema docs — machine-emitted run
#     provenance or hand-written spec text, re-writable from committed inputs.
#   - jm3 / jm45 label-corpus batches — carry ZERO committed human labels in any
#     file yet (unlabeled); their images.jsonl is a machine coord record that the
#     `present`/`render-one` path reproduces, so it is regenerable. Add them here
#     the moment they carry labels.
#   - wallpaper_corpus/*/images.jsonl — all label blocks are null; "humanq3"
#     names human-*seeded* generation, not committed human labels.
#   - GPU eval records (queries/sampler_eval), discovery ledgers, and every
#     classifier weight EXCEPT the v5 rollback anchor below — regenerable at
#     compute cost from committed inputs, which is the project's own line.
#   - atlas round embeds (data/atlas/round{1,2}/*_embed.npz) and discovery
#     outcome_feats.npz — machine features with committed producers; regenerable
#     at compute cost. (Called out because "unregenerable" could be read to sweep
#     them in — they stay out. If any lacks a live producer, promote it.)
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
    # Committed classifier weights (force-tracked; not reproducible under GPU float
    # nondeterminism, so no rebuild path). v7 is the LIVE deployed model; v6 is the
    # one-flip rollback anchor (the role v5 held before the v7 promotion); v5 stays as
    # the deeper rollback. Every other v{2..4} weight is gitignored under data/*.
    "data/classifier/v7/model_best.pt",
    "data/classifier/v6/model_best.pt",
    "data/classifier/v5/model_best.pt",
    # The prospect location library. Both are unregenerable: morph_v6 has no
    # producer and the CLIP arrays only regenerate value-approximate under a
    # verdict-sensitive threshold. (.gitignore negates these two exact paths; the
    # regenerable shards/*.npz overlay stays ignored.)
    "data/library_embeddings/embeddings.npz",
    "data/library/library_records.jsonl",
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
