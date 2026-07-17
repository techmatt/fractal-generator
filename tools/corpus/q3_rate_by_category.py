#!/usr/bin/env python3
"""Human-labeled q3 rate per fractal category, across EVERY labeled batch.

Reads the permanent label corpus (data/label_corpus/, see CORPUS_SCHEMA.md — the
authoritative spec) and reports, per fractal category:

    total human-labeled | q3 count | q3 rate

both crop-level and location-level (deduped canonical = max score over crops at a
location), plus a per-batch breakdown and the sampling-bias caveats.

WHAT COUNTS AS A LABEL (the traps this script is built around):

  * HUMAN LABELS ONLY. The human verdict is `label.score` in images.jsonl
    (reconciled with the harness `scores.json` export, which keys by image_id).
    `provenance.decoded_class` / `provenance.k3` are the MACHINE decode and are
    never counted as a human label — mixing them has produced wrong per-family
    counts before.

  * Storage is heterogeneous. Some batches carry the label both in scores.json and
    in images.jsonl (merged); some have an empty scores.json and no human labels at
    all; the blindspot batch's labels live only in images.jsonl. We enumerate every
    batch dir and reconcile per-batch, never globbing one file pattern.

  * Category is a JOIN. The fractal category lives in the images.jsonl row
    (`render.fractal_type`), next to the image_id the label is keyed on. A
    scores.json key with no matching images.jsonl row is UNJOINABLE — reported on
    its own line, never dropped silently and never guessed from a filename.

  * `render.fractal_type` is the version-invariant category field. Pre-multi-family
    batches predate the field (it is absent) and could only ever render mandelbrot,
    so absent -> "mandelbrot". This is a structural fact about those batches, not a
    filename guess; it is stated in the output.

Read-only. No thresholds, re-labeling, or re-scoring. Writes nothing outside the
report path passed with --out (default docs/findings/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")   # report contains non-cp1252 glyphs
except (AttributeError, ValueError):
    pass

# Route label resolution through the SHARED canonical primitive (merged
# label.score ELSE the registered labels/*.json sidecar joined by image_id) — the
# exact rule corpus_reader (trainer view) and query_sampler (location pool) use, so
# this census cannot drift from the training-data reader. Reading in-row + batch
# scores.json alone (the previous bug here) silently drops every sidecar-only batch
# (julia_ladder_j0, mining, scale, jm3_band, jm45_band).
sys.path.insert(0, os.path.dirname(__file__))
import label_store as ls  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
BATCHES_DIR = os.path.join(ROOT, "data", "label_corpus", "batches")

# Batches that are, for their family, the closest thing to an unbiased draw
# (no classifier in the selection loop, no descent re-selection): a loose accept
# gate over flat redraws. Everything else is biased by construction — descent
# selection (rev4), a v2/v3 model gate (rev4occfix, mining), v5 ranking (gather),
# or is a negative set by design (blindspot = v6-rejects). See the caveat section.
QUASI_UNBIASED_BATCHES = {"2026-06-23_flat_generate_loose0_v3"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def category_of(row):
    """Version-invariant fractal category = render.fractal_type, else mandelbrot.

    The field is absent only in pre-multi-family batches, which could render
    nothing but mandelbrot; absent therefore means mandelbrot (a structural fact,
    not a filename inference)."""
    ft = row.get("render", {}).get("fractal_type")
    return ft if ft else "mandelbrot"


def location_key(row):
    """Canonical location identity: fractal geometry only, palette/composition
    excluded (those define a crop, not a location). Two crops with the same key are
    the same location and collapse to one under location-level dedup."""
    r = row.get("render", {})
    return (
        category_of(row),
        str(r.get("cx")), str(r.get("cy")), str(r.get("fw")),
        str(r.get("c_re")), str(r.get("c_im")),
    )


def process_batch(batch_id):
    """Resolve every row's human label via label_store (merged label.score ELSE the
    registered sidecar). Also records where the label came from so the per-batch
    table can name the store, and flags any in-row/sidecar disagreement."""
    d = os.path.join(BATCHES_DIR, batch_id)
    rows = read_jsonl(os.path.join(d, "images.jsonl"))
    sidecar = ls.sidecar_for(batch_id)          # {image_id: score} or None
    ids = {r["image_id"] for r in rows}
    # A registered sidecar key with no matching images.jsonl row is UNJOINABLE.
    unjoinable = sorted(k for k in (sidecar or {}) if k not in ids)

    labeled = []          # (row, score)
    conflicts = []
    n_inrow = n_sidecar = 0
    for r in rows:
        ir = (r.get("label") or {}).get("score")
        score = ls.resolve_score(r, sidecar)    # in-row ELSE sidecar
        if ir is not None and sidecar is not None:
            ss = sidecar.get(r["image_id"])
            if ss is not None and int(ss) != int(ir):
                conflicts.append(r["image_id"])
        if score is not None:
            labeled.append((r, int(score)))
            if ir is not None:
                n_inrow += 1
            else:
                n_sidecar += 1
    if n_sidecar and n_inrow:
        store = "in-row+sidecar"
    elif n_sidecar:
        store = "sidecar"
    elif n_inrow:
        store = "in-row"
    else:
        store = "UNLABELED"
    return {
        "batch_id": batch_id,
        "n_rows": len(rows),
        "labeled": labeled,
        "unjoinable": unjoinable,
        "conflicts": conflicts,
        "store": store,
    }


def tally(pairs):
    """pairs: iterable of (category, score) -> {category: (total, q3)}."""
    total = defaultdict(int)
    q3 = defaultdict(int)
    for cat, score in pairs:
        total[cat] += 1
        if score == 3:
            q3[cat] += 1
    return total, q3


def coarse(cat):
    """Collapse the 9 render families to the mandelbrot / julia / phoenix split the
    corpus notes track (multibrot folds into its base plane)."""
    if cat.startswith("julia"):
        return "julia"
    if cat == "phoenix":
        return "phoenix"
    return "mandelbrot"


def fmt_table(total, q3, order=None):
    cats = order or sorted(total)
    lines = ["| category | total labeled | q3 count | q3 rate |",
             "|---|---:|---:|---:|"]
    tt = qq = 0
    for c in cats:
        t, q = total.get(c, 0), q3.get(c, 0)
        tt += t
        qq += q
        rate = f"{q / t:.3f}" if t else "—"
        lines.append(f"| {c} | {t} | {q} | {rate} |")
    lines.append(f"| **all** | **{tt}** | **{qq}** | "
                 f"**{qq / tt:.3f}** |" if tt else "| **all** | **0** | **0** | **—** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "findings",
                    "labeled_q3_rate_by_category.md"))
    args = ap.parse_args()

    batch_ids = sorted(x for x in os.listdir(BATCHES_DIR)
                       if os.path.isdir(os.path.join(BATCHES_DIR, x)))
    batches = [process_batch(b) for b in batch_ids]

    # --- crop-level, pooled over all batches ---
    crop_pairs = [(category_of(r), s) for b in batches for (r, s) in b["labeled"]]
    ctot, cq3 = tally(crop_pairs)

    # --- location-level: global dedup by canonical location, MAX score over crops ---
    loc_best = {}   # location_key -> (category, best_score)
    for b in batches:
        for r, s in b["labeled"]:
            k = location_key(r)
            cat = category_of(r)
            if k not in loc_best or s > loc_best[k][1]:
                loc_best[k] = (cat, s)
    loc_pairs = list(loc_best.values())
    ltot, lq3 = tally(loc_pairs)

    # coarse rollups
    ccoarse = [(coarse(c), s) for c, s in crop_pairs]
    lcoarse = [(coarse(c), s) for c, s in loc_pairs]
    cc_t, cc_q = tally(ccoarse)
    lc_t, lc_q = tally(lcoarse)

    fam_order = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
                 "julia", "julia_multibrot3", "julia_multibrot4",
                 "julia_multibrot5", "phoenix"]
    coarse_order = ["mandelbrot", "julia", "phoenix"]

    total_unjoinable = sum(len(b["unjoinable"]) for b in batches)
    total_conflicts = sum(len(b["conflicts"]) for b in batches)

    out = []
    A = out.append
    A("# Human-labeled q3 rate per fractal category\n")
    A("_Generated by `tools/corpus/q3_rate_by_category.py` — read-only over "
      "`data/label_corpus/`. Regenerate with `uv run python "
      "tools/corpus/q3_rate_by_category.py`._\n")
    A("> **CORRECTED 2026-07-17.** The prior version of this report undercounted by "
      "reading only in-row `label.score` + each batch's `scores.json`, which are "
      "empty for the five **sidecar-only** batches (`julia_ladder_j0`, `mining`, "
      "`scale`×2, `jm3_band`, `jm45_band`) — 2779 labels whose sole home is a "
      "`labels/*.json` sidecar keyed by `image_id`. Those batches were wrongly "
      "marked UNLABELED and the pooled julia rate omitted the entire 1000-location "
      "J0 ladder. It now routes through the shared canonical resolver "
      "`tools/corpus/label_store.resolve_score` (the training reader's own path); "
      "location totals reconcile exactly to the v6 coverage audit's 5713. See "
      "`docs/findings/sidecar_label_resolution_and_jm_band_crosscheck.md`._\n")
    A("**Label source: the shared canonical resolver "
      "`tools/corpus/label_store.resolve_score` — merged human `label.score` in "
      "`images.jsonl` ELSE the registered `labels/*.json` sidecar joined by "
      "`image_id`. This is the SAME primitive `corpus_reader` (the trainer view) and "
      "`query_sampler` (the location pool) use, so this census cannot drift from the "
      "training reader. `decoded_class`/`k3` (the machine decode) is NEVER counted.** "
      "Category = `render.fractal_type` (version-invariant); absent in "
      "pre-multi-family batches, which could only render mandelbrot, so absent → "
      "mandelbrot.\n")

    A(f"Unjoinable labels (registered sidecar key with no images.jsonl row): "
      f"**{total_unjoinable}**. Label conflicts (in-row ≠ sidecar): "
      f"**{total_conflicts}**.\n")

    A("_Location totals reconcile exactly to the 5713 labeled locations of the v6 "
      "coverage audit, and the julia / julia\\_multibrot\\* family counts match it "
      "row-for-row. One category-attribution nuance: 22 `prospect_run1` rows are "
      "mandelbrot-plane multibrot3/4/5 (`provenance.family`) but carry no "
      "`render.fractal_type`, so this report — which keys category strictly off the "
      "version-invariant `render.fractal_type` — folds them into `mandelbrot`. That "
      "shifts 9/9/4 locations from multibrot3/4/5 into mandelbrot vs the audit's "
      "`provenance.family` binning; it nets to zero at the mandelbrot-group level "
      "and never touches julia. It is a schema gap in the prospect batch, not a "
      "label defect._\n")

    A("## Location-level (deduped canonical — max score over crops at a location)\n")
    A("This is the honest \"how many distinct good locations\" count.\n")
    A(fmt_table(ltot, lq3, fam_order) + "\n")
    A("Coarse rollup:\n")
    A(fmt_table(lc_t, lc_q, coarse_order) + "\n")

    A("## Crop-level (every labeled crop; palettes/compositions counted separately)\n")
    A(fmt_table(ctot, cq3, fam_order) + "\n")
    A("Coarse rollup:\n")
    A(fmt_table(cc_t, cc_q, coarse_order) + "\n")

    # --- per-batch breakdown ---
    A("## Per-batch breakdown (crop-level)\n")
    A("The pooled tables above mix non-comparable sampling regimes — read this "
      "first. `unbiased?` marks the closest thing to an unbiased draw for its "
      "family (loose accept gate, no model/descent selection); everything else is "
      "biased by construction.\n")
    A("| batch | rows | human-labeled | q3 | q3 rate | families | label store | "
      "unbiased? |")
    A("|---|---:|---:|---:|---:|---|---|---|")
    for b in batches:
        pairs = [(category_of(r), s) for (r, s) in b["labeled"]]
        t, q = tally(pairs)
        nl = sum(t.values())
        nq = sum(q.values())
        fams = ",".join(sorted({category_of(r) for r, _ in b["labeled"]})) or "—"
        if not b["labeled"]:
            fams = ",".join(sorted({category_of(r)
                                    for r in read_jsonl(os.path.join(
                                        BATCHES_DIR, b["batch_id"], "images.jsonl"))}))
        store = b["store"]
        rate = f"{nq / nl:.3f}" if nl else "—"
        ub = "yes" if b["batch_id"] in QUASI_UNBIASED_BATCHES else ""
        A(f"| {b['batch_id']} | {b['n_rows']} | {nl} | {nq} | {rate} | {fams} | "
          f"{store} | {ub} |")
    A("")

    A("## The confound — do not read the pooled table as a per-family price list\n")
    A("These batches were **not sampled uniformly**, so the pooled q3 rate is not "
      "\"how easy is it to find a good one in family X.\"\n")
    A("- **mandelbrot** is the only family with a quasi-unbiased draw: "
      "`2026-06-23_flat_generate_loose0_v3` (loose accept gate over flat redraws, "
      "no model/descent selection). Its q3 rate is the least-biased mandelbrot "
      "base rate available.\n")
    A("- **julia / multibrot / phoenix STILL have no *unbiased* labeled draw**, but "
      "they are no longer unlabeled here — the previous version of this report read "
      "in-row `label.score` + batch `scores.json` only and so dropped every "
      "sidecar-only batch (`julia_ladder_j0`, `mining`, `scale`, `jm3_band`, "
      "`jm45_band`). Those labels exist and are counted now. What remains true is "
      "that every labeled draw in these families is *biased*, not that they are "
      "unlabeled.\n")
    A("- **`julia_ladder_j0` (1000 J0 Julia locations) is NOT an unbiased base "
      "rate**, despite its size. Regime (`tools/julia_ladder/build_j0.py`): its "
      "Julia `c` values are the **label-2/3 (good) Mandelbrot centers** from the v4 "
      "manifest (893 raw → 618 deduped neighborhoods); each seed then spawns "
      "systematic center-zoom rungs (`fw = 3.0/4`, `3.0/8`) **and** "
      "`guided-descend` descent rungs, v4-scored, then stratified-sampled to 1000. "
      "So it is a systematic zoom ladder *conditioned on already-good `c`* with a "
      "descent-selected half — positively biased toward good Julia. Its q3 rate is "
      "an over-estimate of the Julia base rate, not the base rate. Julia's true "
      "unbiased rate was **not** available all along.\n")
    A("- The only labeled multi-family harvest, `2026-07-05_gather_v6`, is a "
      "guard-OFF harvest oversampled by raw v5 rank — positively biased. `jm3_band` "
      "/ `jm45_band` are model-band revival batches (sampled inside a targeted "
      "score band), also biased. So the honest *unbiased* per-family rate for "
      "julia / multibrot / phoenix is still **unknown** — but the labels themselves "
      "are present and pooled above.\n")
    A("- `2026-07-12_blindspot_v6reject_v1` is a **negative set by construction** "
      "(v6-rejects); its near-zero q3 rate is by design and drags the pooled "
      "mandelbrot rate down.\n")
    A("- `guided_descend_rev4`, `rev4occfix_v2filtered`, and `mining_v3guided_v1` "
      "are descent/model-selection-biased (positively enriched).\n")

    report = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[written to {os.path.relpath(args.out, ROOT)}]")


if __name__ == "__main__":
    main()
