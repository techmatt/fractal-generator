"""Tests for the version-blind reader's shared label resolution.

Guards against the latent sidecar-drop: `corpus_reader.iter_labeled` used to read
`label.score` alone, so the sidecar-only batches (Julia/mining/scale) silently
resolved to no label. It now routes through `label_store` — the SAME primitive the
query sampler uses — so (a) it recovers those labels and (b) the two consumers can
never disagree on a row.

Run either way:
  uv run pytest tools/corpus/test_corpus_reader.py
  uv run python tools/corpus/test_corpus_reader.py     # prints PASS/FAIL summary
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # tools/corpus
sys.path.insert(0, os.path.join(HERE, "..", "queries"))   # tools/queries

import corpus_reader as cr  # noqa: E402
import label_store as ls  # noqa: E402


# The (b)-case batches whose labels live ONLY in a labels/*.json sidecar — the ones a
# `label.score`-only reader dropped. julia_ladder_j0 is the whole Julia family; the
# other three are Mandelbrot mining/scale.
SIDECAR_BATCHES = list(ls.SIDECAR_LABELS)


def _reader_labels_by_batch():
    """{batch_id: {image_id: score}} as corpus_reader.iter_labeled resolves them."""
    out = {}
    for lc in cr.iter_labeled():
        out.setdefault(lc.batch_id, {})[lc.image_id] = lc.score
    return out


def _independent_join(batch_id):
    """Reconstruct a sidecar batch's labels WITHOUT label_store: raw sidecar file
    JOINED to the batch's images.jsonl image_ids. This is the external ground truth
    that the shared resolver must reproduce."""
    sidecar_file = ls.SIDECAR_LABELS[batch_id]
    raw = json.loads(open(os.path.join(ls.LABELS_DIR, sidecar_file), encoding="utf-8").read())
    body = raw["labels"] if isinstance(raw.get("labels"), dict) else raw
    sidecar = {k: int(v) for k, v in body.items() if v is not None}
    jl = os.path.join(cr.cc.BATCHES_DIR, batch_id, "images.jsonl")
    ids = set()
    with open(jl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["image_id"])
    return {iid: sc for iid, sc in sidecar.items() if iid in ids}


def test_recovers_sidecar_only_labels():
    """iter_labeled now sees the previously-dropped sidecar batches (non-zero each),
    and its recovered labels match an independent sidecar x images.jsonl join."""
    by_batch = _reader_labels_by_batch()
    for bid in SIDECAR_BATCHES:
        got = by_batch.get(bid, {})
        assert got, f"{bid}: iter_labeled recovered 0 labels (sidecar drop regressed)"
        assert got == _independent_join(bid), (
            f"{bid}: reader labels diverge from the raw sidecar join")


def test_both_consumers_share_the_resolver():
    """corpus_reader and query_sampler resolve through the SAME label_store object —
    the structural guarantee that they cannot drift. Also assert they agree row-for-row
    on the sidecar batches (the concrete cross-consumer check)."""
    sys.path.insert(0, os.path.join(HERE, "..", "palettes"))
    import query_sampler as qs  # noqa: E402  (heavy import: colormap + numpy)

    # Same primitive, one registry.
    assert cr.ls is qs.ls, "corpus_reader and query_sampler bound different label_store"
    assert qs.SIDECAR_LABELS is ls.SIDECAR_LABELS

    reader = _reader_labels_by_batch()
    # Reproduce the sampler's per-row resolution (its from_corpus loop calls the SAME
    # ls.resolve_score) and confirm it agrees with the reader on the q2/q3 rows.
    for bid in SIDECAR_BATCHES:
        sidecar = ls.sidecar_for(bid)
        jl = os.path.join(cr.cc.BATCHES_DIR, bid, "images.jsonl")
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sc = ls.resolve_score(row, sidecar)
                if sc in (2, 3):   # q2/q3: the sampler's keep-set
                    assert reader.get(bid, {}).get(row["image_id"]) == sc, (
                        f"{bid}/{row['image_id']}: reader != sampler resolution")


def main():
    tests = [
        test_recovers_sidecar_only_labels,
        test_both_consumers_share_the_resolver,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (t.__name__, e))
    print("\n%d/%d tests passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
