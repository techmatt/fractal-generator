"""Canonical single reader for the q4 WINDOW label store.

This store is SEPARATE from the v7 location corpus (`data/label_corpus`) on purpose:
it is **distribution-bound** — a stratified sweep of 16:9 windows over ~30 minibrot
renders, biased by construction (score-stratified, prefiltered), for the q4 stage-1
gate. Pooling it into the version-blind v7 training union would poison that union's
distribution, so it is NOT registered in
`tools/corpus/label_store.py::SIDECAR_LABELS`. Every consumer of q4-window labels
routes through THIS module — the one place that knows the store layout and the
three-way resolution rule.

Row schema (`data/q4_window_corpus/batches/<batch>/windows.jsonl`, one per window):
    window_id        stable id (minibrot_id + scale + rect hash) == crops/<id>.jpg
    minibrot_id      parent nucleus render id
    period           nucleus period
    render           parent MEDIUM-render geometry: cx/cy/fw (decimal strings),
                     maxiter, family, width, height, aspect, palette
    window           frame-normalized rect {u,v,w,h} within the parent render
    scale            window width as fraction of frame width
    band             composite-score stratification band (0..N_BANDS-1)
    score_composite  the score_A composite the stratification used
    features         field-stat feature vector (compute_metrics keys) — fitting-ready
    label.klass      null | "accept" | "reject" | "filter_leak"
                     (null->value is the ONLY allowed mutation)

THREE-WAY labels:
    accept       "worth stage-2's time"   -- a HIGH-RECALL gate; stage-2 filters
    reject       "clean window, just not q4-worthy"
    filter_leak  "dead/noisy/barren garbage the step-3 pre-filter should have dropped"
                 -- this is FEEDBACK ON THE PRE-FILTER, not a quality judgment. It is
                 EXCLUDED from the accept-vs-reject fit (`iter_labeled`) and surfaced
                 only as a leak-rate diagnostic (`prefilter_leak_rate`). Keeping it a
                 rare exception tag is what lets accept/reject stay one-key fast paths.

Labels: `label.klass` is authoritative when non-null; else the accept/reject/leak flow
exports a `scores.json` sidecar keyed by window_id (string class) — `resolve_klass`
joins the two, sidecar filling nulls only.

CLI:  uv run python -m tools.corpus.q4_window_reader stats
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE_ROOT = ROOT / "data" / "q4_window_corpus" / "batches"

# The registered q4-window batches (single source of truth). NEW batches append here.
REGISTERED_BATCHES = [
    "2026-07-23_q4_stage1_windows",
    "2026-07-23_q4_g_aimed",
]

CLASSES = ("accept", "reject", "filter_leak")
# tolerated legacy int forms in a scores.json (the tools/viz convention): 3->accept,
# 1->reject, 2->filter_leak. Native form is the string class.
_INT_TO_CLASS = {3: "accept", 1: "reject", 2: "filter_leak"}


def batch_dir(batch_id):
    d = STORE_ROOT / batch_id
    if not d.exists():
        raise FileNotFoundError(f"q4-window batch not on disk: {d}")
    return d


def _norm_klass(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v if v in CLASSES else None
    return _INT_TO_CLASS.get(int(v))


def load_scores_sidecar(batch_id):
    """{window_id: class-string} from scores.json if present, else {}."""
    p = batch_dir(batch_id) / "scores.json"
    if not p.exists():
        return {}
    out = {}
    for k, v in json.loads(p.read_text()).items():
        c = _norm_klass(v)
        if c is not None:
            out[k] = c
    return out


def resolve_klass(row, sidecar):
    """A window's class: in-row `label.klass` (authoritative when non-null) ELSE the
    scores.json sidecar join. Returns one of CLASSES, or None if unlabeled in both.
    The ONE resolution rule; every consumer uses it."""
    k = _norm_klass((row.get("label") or {}).get("klass"))
    if k is not None:
        return k
    return sidecar.get(row["window_id"])


def iter_windows(batch_id=None, *, labeled_only=False):
    """Yield (row, klass) over one batch or all REGISTERED_BATCHES.
    `klass` is a CLASSES value or None. `labeled_only` drops None."""
    batches = [batch_id] if batch_id else REGISTERED_BATCHES
    for bid in batches:
        d = batch_dir(bid)
        sidecar = load_scores_sidecar(bid)
        for line in (d / "windows.jsonl").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            k = resolve_klass(row, sidecar)
            if labeled_only and k is None:
                continue
            yield row, k


def iter_labeled(batch_id=None):
    """The accept-vs-reject FIT view: yield (row, accept_bool) over accept/reject rows
    ONLY. filter_leak (and unlabeled) are excluded — filter_leak is pre-filter feedback,
    never a quality target. This is the canonical training/eval view."""
    for row, k in iter_windows(batch_id, labeled_only=True):
        if k == "accept":
            yield row, True
        elif k == "reject":
            yield row, False
        # filter_leak: skip (excluded from the fit by contract)


def prefilter_leak_rate(batch_id=None):
    """Diagnostic: (n_filter_leak, n_labeled, rate) over labeled windows. A high rate
    means the step-3 pre-filter is letting garbage through and needs tightening."""
    n_leak = n_lab = 0
    for _, k in iter_windows(batch_id, labeled_only=True):
        n_lab += 1
        if k == "filter_leak":
            n_leak += 1
    return n_leak, n_lab, (n_leak / n_lab if n_lab else 0.0)


def crop_path(batch_id, window_id):
    return batch_dir(batch_id) / "crops" / f"{window_id}.jpg"


def _stats():
    for bid in REGISTERED_BATCHES:
        try:
            rows = list(iter_windows(bid))
        except FileNotFoundError as e:
            print(f"{bid}: NOT ON DISK ({e})")
            continue
        n = len(rows)
        acc = sum(1 for _, k in rows if k == "accept")
        rej = sum(1 for _, k in rows if k == "reject")
        leak = sum(1 for _, k in rows if k == "filter_leak")
        unl = sum(1 for _, k in rows if k is None)
        mbs = len({r["minibrot_id"] for r, _ in rows})
        bands = {}
        for r, _ in rows:
            bands[r.get("band")] = bands.get(r.get("band"), 0) + 1
        nleak, nlab, rate = prefilter_leak_rate(bid)
        print(f"{bid}: {n} windows  {mbs} minibrots")
        print(f"    accept={acc} reject={rej} filter_leak={leak} unlabeled={unl}")
        print(f"    prefilter leak rate (diagnostic): {nleak}/{nlab} = {rate:.1%}")
        print(f"    per-band counts: {dict(sorted(bands.items(), key=lambda x:(x[0] is None,x[0])))}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        _stats()
    else:
        print(__doc__)
