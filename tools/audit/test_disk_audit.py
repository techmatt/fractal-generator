"""Guards for disk_audit.py's classification rules.

The load-bearing property under test: everything under the repo-root `labels/`
directory classifies as NEVER-delete. That directory is the sole home of several
thousand human labels (empty-scores.json batch sidecars + the legacy label store);
human labels are the one unregenerable artifact this tool can touch. The guard is a
LOCATION rule, deliberately NOT keyed off label_store.SIDECAR_LABELS — an
unregistered sidecar has already happened twice, and a registry-keyed guard would
call exactly that file deletable.

Run either way:
  uv run pytest tools/audit/test_disk_audit.py
  uv run python tools/audit/test_disk_audit.py     # prints PASS/FAIL summary
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import disk_audit as da  # noqa: E402


LABELS_PATHS = [
    "labels/location_labels.json",                       # legacy label store
    "labels/location_labels_julia_ladder_j0.json",       # sidecar, empty scores.json
    "labels/jm45_band_v1.json",                          # unmerged revival sidecar
    "labels/julia-census-readout.md",                    # loose readout in the dir
    "labels/subdir/nested_labels.json",                  # covers the whole subtree
]


def _classify_without_labels_rule(rel):
    """First-match category using RULES with the ^labels/ rule removed — i.e. what the
    tool WOULD have decided before this guard existed. Mirrors da.classify()'s logic."""
    for r in da.RULES:
        if r.pattern == r"^labels/":
            continue
        if r._rx.search(rel):
            return r.category
    return da.DEFAULT_UNMATCHED.category


def test_labels_paths_are_never():
    """Every path under labels/ classifies NEVER. Fails if the rule is removed or its
    path is changed (both flip these to a non-NEVER category)."""
    for rel in LABELS_PATHS:
        assert da.classify(rel).category == da.NEVER, rel


def test_labels_rule_sits_above_safe_rules():
    """RED ON PURPOSE, at the rule level: a stray log/render inside labels/ hits a
    *safe, deletable* rule with the guard removed, and is forced NEVER with it. This
    is the whole point of a blanket location rule vs. a per-file content guard — the
    content guard only knows images.jsonl/scores.json, so it would miss it."""
    rel = "labels/run.log"                                 # matches the generic .log SCRATCH rule
    without = _classify_without_labels_rule(rel)
    assert without in da.SAFE_CATEGORIES, (rel, without)   # would have been deleted (SCRATCH)
    assert da.classify(rel).category == da.NEVER, rel      # now blocked by ^labels/


def test_labels_rule_is_location_not_registry():
    """The guard must not depend on label_store.SIDECAR_LABELS: an UNREGISTERED sidecar
    (an arbitrary new filename never added to the registry) is still NEVER."""
    assert da.classify("labels/some_brand_new_unregistered_batch_v9.json").category == da.NEVER


def test_labels_rule_present_and_placed_before_safe_rules():
    """Structural guard: the ^labels/ rule exists, is NEVER, and precedes every
    regenerable/scratch rule (first-match-wins ordering is what makes it a blanket)."""
    idxs = [i for i, r in enumerate(da.RULES) if r.pattern == r"^labels/"]
    assert idxs, "the ^labels/ never-delete rule is gone"
    labels_i = idxs[0]
    assert da.RULES[labels_i].category == da.NEVER
    first_safe = next(i for i, r in enumerate(da.RULES) if r.category in da.SAFE_CATEGORIES)
    assert labels_i < first_safe, "^labels/ must precede safe rules or a safe file leaks through"


def test_audit_end_to_end_forces_never_and_excludes_from_manifest(tmp_path):
    """RED ON PURPOSE, end-to-end: build a tree with labels/ holding a real sidecar,
    a legacy store, AND a stray .log (a would-be SCRATCH delete). Run the full audit()
    and assert the labels/ dir node is NEVER and nothing under it reaches the safe
    delete-manifest. Without the guard the .log is a live deletion candidate."""
    repo = tmp_path
    lab = repo / "labels"
    (lab / "subdir").mkdir(parents=True)
    # A sidecar whose *filename* is not images.jsonl/scores.json, so the per-file
    # content guard cannot see its labels — only the path rule protects it.
    (lab / "location_labels_julia_ladder_j0.json").write_text('{"img_0001": 3, "img_0002": 1}', "utf-8")
    (lab / "subdir" / "nested_labels.json").write_text('{"x": 2}', "utf-8")
    (lab / "stray.log").write_text("would-be scratch delete\n", "utf-8")

    _total, _cat_totals, _counts, items, per_dir = da.audit(repo, [lab])

    assert per_dir["labels/"].category == da.NEVER
    assert per_dir["labels/"].reason.startswith("repo-root human label store")

    # The safe delete-manifest is exactly what --apply would remove. Nothing under
    # labels/ may appear there — most pointedly not the .log.
    safe_items = [n for n in items if n.category in da.SAFE_CATEGORIES and not n.synthetic]
    leaked = [n.rel for n in safe_items if n.rel.startswith("labels/")]
    assert not leaked, f"deletable items leaked under labels/: {leaked}"

    # And the stray .log specifically resolves NEVER through the real classifier.
    assert da.classify("labels/stray.log").category == da.NEVER


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import inspect, tempfile
    from pathlib import Path
    failed = 0
    for fn in fns:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
