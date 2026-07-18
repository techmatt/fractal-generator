"""Every human label on disk must stay reachable through the canonical reader.

The label store's sidecar path depends on a registry (`label_store.SIDECAR_LABELS`)
that has silently gone incomplete TWICE: a batch labeled after the registry was last
touched went unregistered, its labels were never merged in-row, and it resolved to
ZERO through `label_store.resolve_score` — silently dropping the whole batch from any
retrain. This test fails the instant a `labels/*.json` sidecar carries labels that the
canonical resolver cannot reach.

The invariant is BEHAVIORAL, not registry membership (do NOT rewrite this as "every
file appears in SIDECAR_LABELS" — that false-fires on the in-row-merged batches):

  For every sidecar on disk whose keys are label_corpus `image_id`s, every one of
  those labels must resolve non-null through the SAME path the reader uses —
  `resolve_score(row, sidecar_for(batch_id))` = merged `label.score` ELSE the
  REGISTERED sidecar join. A batch merged in-row (blindspot, prospect) reconciles
  because its labels live in `label.score`; a registered sidecar-only batch
  (jm3/jm45/mining/scale/julia_ladder_j0) reconciles because `sidecar_for()` returns
  its map. The failure this catches is an unregistered, unmerged sidecar — reachable
  count 0 while the file holds N.

Out-of-scope sidecars (0 keys match any label_corpus `image_id`) are a DIFFERENT
store, not a drop, and are recorded + skipped rather than asserted against this reader:
the legacy `location_labels.json` (composite `idx|framing|palette` keys, labels live
in-store), `palette_scores.json` (palette labels), the wallpaper_corpus sidecars, and
the render-mode head sidecars. Asserting them here would false-fire on stores the
label_corpus reader never reads.

Reconciliation is on COUNT reachability, NOT score identity: 616 `image_id`s collide
across sibling batches (e.g. loose0 vs rev4 share `0_center_...`), so a colliding key
can legitimately carry a different in-row score in another batch. The drop we guard is
"label vanished to null", which is a count invariant.

Run either way:
  uv run pytest tools/corpus/test_label_reachability.py
  uv run python tools/corpus/test_label_reachability.py     # prints the reconciliation table
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # tools/corpus

import corpus_common as cc  # noqa: E402
import label_store as ls  # noqa: E402


def _scan_corpus():
    """Walk every label_corpus batch ONCE, mirroring corpus_reader.iter_labeled's
    per-row resolution (same ls.resolve_score, same ls.sidecar_for registry lookup).

    Returns (owners, reachable):
      owners    : {image_id: set(batch_id)}  — every id on disk, labeled or not.
      reachable : {image_id: int score}      — ids the CANONICAL path resolves non-null.
    `reachable` prefers a non-null hit, so a colliding id labeled in one batch and null
    in another still counts as reachable (the store holds it somewhere)."""
    owners: dict[str, set] = {}
    reachable: dict[str, int] = {}
    for images_path in sorted(glob.glob(os.path.join(cc.BATCHES_DIR, "*", "images.jsonl"))):
        batch_id = os.path.basename(os.path.dirname(images_path))
        sidecar = ls.sidecar_for(batch_id)
        with open(images_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = row["image_id"]
                owners.setdefault(iid, set()).add(batch_id)
                sc = ls.resolve_score(row, sidecar)
                if sc is not None and iid not in reachable:
                    reachable[iid] = int(sc)
    return owners, reachable


def reconcile():
    """Per-sidecar reconciliation table. For each labels/*.json:
      {sidecar: {"disk": N, "in_scope": M, "reachable": R, "out_of_scope": bool}}
    In-scope = keys that are a label_corpus image_id; the store's reachability
    invariant is reachable == in_scope for every in-scope sidecar."""
    owners, reachable = _scan_corpus()
    table = {}
    for fn in sorted(os.listdir(ls.LABELS_DIR)):
        if not fn.endswith(".json"):
            continue
        labels = ls.load_sidecar(fn)                       # {image_id: int}, nulls dropped
        in_scope = [k for k in labels if k in owners]
        reach = sum(1 for k in in_scope if k in reachable)
        table[fn] = {
            "disk": len(labels),
            "in_scope": len(in_scope),
            "reachable": reach,
            "out_of_scope": len(in_scope) == 0,
        }
    return owners, table


def test_every_label_corpus_sidecar_is_fully_reachable():
    """Every sidecar keyed on label_corpus image_ids resolves ALL its labels through
    the canonical reader. A sidecar-only batch that went unregistered (and was never
    merged in-row) reads as reachable=0 here and fails, naming the batch."""
    owners, table = reconcile()
    assert owners, "no label_corpus image_ids scanned — reader/glob broke, test is vacuous"

    in_scope_files = [fn for fn, r in table.items() if not r["out_of_scope"]]
    # Guard against a silently vacuous pass: the registered sidecar-only batches MUST
    # be present and in-scope, else there is nothing meaningful to reconcile.
    assert in_scope_files, "no in-scope sidecars found — the reachability check ran on nothing"

    unreachable = {
        fn: r for fn, r in table.items()
        if not r["out_of_scope"] and r["reachable"] != r["in_scope"]
    }
    assert not unreachable, (
        "label store UNREACHABLE through resolve_score — labels present on disk that the "
        "canonical reader drops to zero (likely an unregistered sidecar-only batch; "
        "register it in tools/corpus/label_store.SIDECAR_LABELS or merge it in-row):\n  "
        + "\n  ".join(
            f"{fn}: {r['reachable']}/{r['in_scope']} in-scope labels reachable "
            f"({r['disk']} on disk)"
            for fn, r in sorted(unreachable.items())
        )
    )


def main():
    owners, table = reconcile()
    print("=== labels/ sidecar reachability (disk / in-scope / reachable) ===")
    width = max(len(fn) for fn in table)
    bad = 0
    for fn, r in table.items():
        if r["out_of_scope"]:
            print(f"  {fn:<{width}}  disk={r['disk']:<5}  OUT-OF-SCOPE "
                  f"(0 keys match any label_corpus image_id)")
            continue
        ok = r["reachable"] == r["in_scope"]
        bad += not ok
        flag = "OK " if ok else "!! "
        print(f"  {flag}{fn:<{width}}  disk={r['disk']:<5} in_scope={r['in_scope']:<5} "
              f"reachable={r['reachable']}")
    print(f"\n{len([r for r in table.values() if not r['out_of_scope']])} in-scope sidecars, "
          f"{bad} unreachable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
