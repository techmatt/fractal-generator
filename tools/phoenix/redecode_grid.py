#!/usr/bin/env python
r"""redecode_grid.py — re-decode the Phase-B phoenix grid ledger at the CURRENT production
phoenix t_good (production_seeder.t_good_for("phoenix"), now 0.45 — the label-derived F2
value; docs/findings/phoenix_grid_labels.md §2).

The grid ran at a hardcoded provisional t_good=0.18 that nobody ordered (a stale copy of the
retired v6-era value; see phoenix_grid.py's T_GOOD_DEFAULT, now sourced from the production
table). Because every admitted grid outcome stores its raw p_good/p_notbad and every row is
already guard-clean and scorer_version=="v7", a re-decode is pure arithmetic: recompute
decoded_class = corn_decode(p_notbad, p_good, t) and re-stamp t_good. NO re-render, NO
re-score, NO GPU.

Since 0.45 > 0.18 the new q3 set is a strict subset of the 656 (the notbad gate is unchanged;
only the p_good>=t cut tightens), so re-decoding the admitted ledger is complete. The rows that
fall below 0.45 are simply not written to the re-decoded ledger — they become inadmissible by
the decode predicate, nothing is hand-deleted (the original 0.18 ledger stays on disk untouched).

Distinct-look count at the new threshold = the stored run-global distinct flag intersected with
the new q3 set: a row that was distinct vs all earlier 0.18-admissions is still distinct vs the
0.45 subset (a strict subset can only drop would-be founders, never introduce a new near-dup
collision for a surviving distinct row). This reproduces the docs/findings §2 table exactly.

Outputs (under the grid run dir):
  outcome_ledger_v7_t45.jsonl   the re-decoded admitted q3 ledger (intake source for library_intake_2)
  outcome_feats_v7_t45.npz      the 1280-D feature subset for the surviving ids
  redecode_t45.json             counts + corrected min-per-look price (readout)

  uv run python tools/phoenix/redecode_grid.py            # re-decode data/discovery/phoenix_grid/grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "mining",
          ROOT / "tools" / "scoring", ROOT / "tools" / "reframe", ROOT / "tools" / "corpus",
          ROOT / "tools" / "wallpaper", ROOT / "tools" / "curation",
          ROOT / "tools" / "atlas_probe", ROOT / "tools" / "phoenix"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import production_seeder as ps          # noqa: E402  (t_good_for)
from score_lib import corn_decode        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEFAULT_RUN = ROOT / "data" / "discovery" / "phoenix_grid" / "grid"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--t-good", type=float, default=None,
                    help="override the decode threshold (default: t_good_for('phoenix'))")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    t = args.t_good if args.t_good is not None else ps.t_good_for("phoenix")
    src_ledger = run / "outcome_ledger.jsonl"
    rows = [json.loads(l) for l in src_ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_in = len(rows)

    # Integrity: the source is the admitted q3-at-0.18 ledger — every row must be guard-clean,
    # v7-stamped, and pass the (unchanged) notbad gate. If not, the subset assumption is void.
    for r in rows:
        assert r.get("guard_pass") and r.get("scorer_version") == "v7" and r["p_notbad"] >= 0.5, \
            f"unexpected non-admitted/notbad-fail/non-v7 row {r.get('id')}"

    kept = []
    for r in rows:
        dc = corn_decode(r["p_notbad"], r["p_good"], t)
        if dc == 3:
            r2 = dict(r, decoded_class=dc, t_good=t)   # re-stamp; scorer_version/distinct preserved
            kept.append(r2)

    n_q3 = len(kept)
    n_distinct = sum(1 for r in kept if r.get("distinct"))

    # corrected min-per-look price = cumulative active-min / distinct looks at the new threshold
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    active_min = float(summary["active_minutes"])
    price = round(active_min / n_distinct, 3) if n_distinct else None
    old_distinct = int(summary.get("distinct_looks_phoenix", 0))
    old_price = summary.get("realized_min_per_look_phoenix")

    # write re-decoded ledger
    out_ledger = run / "outcome_ledger_v7_t45.jsonl"
    with open(out_ledger, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    # subset the 1280-D features to the surviving ids (keep the intake unit self-consistent)
    kept_ids = {r["id"] for r in kept}
    feats_src = run / "outcome_feats.npz"
    out_feats = run / "outcome_feats_v7_t45.npz"
    n_feats = 0
    if feats_src.exists():
        with np.load(feats_src, allow_pickle=False) as z:
            sub = {k: z[k] for k in z.files if k in kept_ids}
        np.savez_compressed(out_feats, **sub)
        n_feats = len(sub)

    readout = {
        "run_dir": str(run.relative_to(ROOT)),
        "source_ledger": src_ledger.name, "rows_in": n_in,
        "t_good_old": 0.18, "t_good_new": t,
        "admissions_q3": n_q3, "distinct_looks": n_distinct,
        "dropped_below_t": n_in - n_q3,
        "active_minutes": active_min,
        "min_per_look_new": price, "min_per_look_old": old_price,
        "distinct_looks_old": old_distinct,
        "out_ledger": out_ledger.name, "out_feats": out_feats.name, "n_feats": n_feats,
    }
    (run / "redecode_t45.json").write_text(json.dumps(readout, indent=2), encoding="utf-8")

    print(f"=== phoenix grid re-decode  t_good {0.18} -> {t} ===")
    print(f"  rows_in (q3@0.18)     : {n_in}")
    print(f"  admissions (q3@{t})   : {n_q3}   (dropped {n_in - n_q3} below {t})")
    print(f"  distinct looks        : {n_distinct}   (was {old_distinct} @0.18)")
    print(f"  active-minutes        : {active_min:.2f}")
    print(f"  min-per-look (CORRECTED): {price}   (was {old_price} @0.18)")
    print(f"  wrote {out_ledger.relative_to(ROOT)}  ({n_q3} rows)")
    print(f"  wrote {out_feats.relative_to(ROOT)}  ({n_feats} feats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
