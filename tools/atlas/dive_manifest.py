#!/usr/bin/env python
"""Blind manifest for the v1.2 dive run — the read that adjudicates the deep-options hypothesis.

EVERY dive admission (no pre-filtering beyond standard admission), shuffled arm-free, each
rendered at the deploy-canonical 640x360 ss2 twilight presentation. Writes a HIDDEN key
(`manifest_key.json`: tile -> id / start-group / depth / canonical p_good / morph-cluster /
family / coords) plus the blind index the human scores from (`blind_index.json`). Same format
as the run-2 manifest (tools/atlas/steered_run2_manifest.py), so the same self-contained
labeler (tools/atlas/build_blind_labeler.py) drives the read.

Morph clusters + the group tag come from `<dive-run>/dive_admissions.npz` — run
tools/atlas/steered_v1_2_dive_report.py first.

  uv run python tools/atlas/dive_manifest.py --dive-run data/discovery/steered_v1_2_dive
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))


def main():
    import tools.studies.steered_pilot_morph as spm      # noqa: E402  (heavy; render helpers)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dive-run", type=Path, default=ROOT / "data/discovery/steered_v1_2_dive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/dive_manifest")
    args = ap.parse_args()
    run = args.dive_run

    rows = spm.admitted_q3([json.loads(l) for l in
                            open(run / "outcome_ledger.jsonl", encoding="utf-8") if l.strip()])
    npz = run / "dive_admissions.npz"
    if not npz.exists():
        raise SystemExit(f"missing {npz} — run tools/atlas/steered_v1_2_dive_report.py first")
    z = np.load(npz, allow_pickle=False)
    cluster_of = {str(u): int(c) for u, c in zip(z["uids"], z["cluster_strict"])}
    group_of = {str(u): str(g) for u, g in zip(z["uids"], z["groups"])}

    items = []
    for r in rows:
        pg = float(r["p_good"])
        cpg = r.get("canon_pgood")
        items.append(dict(
            tile=None, id=r["id"], family=r["family"],
            cx=r["outcome_cx"], cy=r["outcome_cy"], fw=r["outcome_fw"],
            c=([r["julia_c_re"], r["julia_c_im"]] if r.get("julia_c_re") is not None else None),
            p_good=pg, p_notbad=float(r["p_notbad"]),
            canon_pgood=(float(cpg) if cpg is not None else None),
            depth=int(r["reached_depth"]),
            start_group=group_of.get(r["id"], r.get("dive_start_group", "?")),
            dive_id=r.get("dive_id"), source_id=r.get("dive_source_id"),
            cluster=cluster_of.get(r["id"], -1),
        ))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)

    tiles = args.out_dir / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    key = []
    for i, x in enumerate(items):
        tile = tiles / f"blind_{i:03d}.jpg"
        spm.render_colored(spm.loc_of_row(by_id[x["id"]]), tile)
        x["tile"] = tile.name
        key.append(x)

    (args.out_dir / "manifest_key.json").write_text(json.dumps(dict(
        run=run.name, n=len(items), n_admissions=len(rows), seed=args.seed,
        note="HIDDEN KEY — do not show the human labeler; maps blind tile -> truth "
             "(start-group, depth, canonical p_good, morph cluster).",
        entries=key,
    ), indent=2), encoding="utf-8")
    (args.out_dir / "blind_index.json").write_text(json.dumps(dict(
        run=run.name, n=len(items),
        instructions="Score each tile 1(bad)/2(okay)/3(good). Tiles are shuffled; no coords/"
                     "scores/group shown. Return {tile: score}.",
        tiles=[e["tile"] for e in key],
    ), indent=2), encoding="utf-8")

    gc = Counter(e["start_group"] for e in key)
    dc = Counter(e["depth"] for e in key)
    print(f"dive manifest: {len(items)} admissions (ALL, no pre-filter), "
          f"{len(set(e['cluster'] for e in key))} morph clusters")
    print(f"  by start-group: {dict(gc)}")
    print(f"  by depth: {dict(sorted(dc.items()))}")
    print(f"wrote {args.out_dir/'manifest_key.json'} (hidden) + blind_index.json + tiles/")


if __name__ == "__main__":
    main()
