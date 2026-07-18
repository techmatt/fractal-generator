#!/usr/bin/env python
"""Blind-read calibration manifest for the steered run's keeper bar.

Draws ~N admitted locations stratified across (canonical p_good tercile x depth bucket x morph
cluster), shuffles them arm-free (single-arm run), renders each at the deploy-canonical 640x360
ss2 twilight presentation, and writes a HIDDEN key (`manifest_key.json`: tile -> id / coords /
canonical p_good / depth / morph-cluster / family / keeper-status) plus a blind index the human
scores from (`blind_index.json`: shuffled tile list, no metadata). This is the labeled set that
CALIBRATES the provisional keeper cut against human judgement — so it must span the p_good range
(does high-p_good really read better?), the depth range (do the shallow steered keepers hold?),
and the morph clusters (are the near-repeats really the same look?).

Morph clusters come from `<run>/morph_admissions.npz` (grayscale morph_gray, written by
steered_run2_report.py) — run the report first.

  uv run python tools/atlas/steered_run2_manifest.py --run-dir data/discovery/steered_run2 --n 60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(ROOT / "tools" / "mining"))

import tools.studies.steered_pilot_morph as spm      # noqa: E402
import keeper_cut as kc                                # noqa: E402
from score_lib import corn_decode                      # noqa: E402


def pgood_tercile(pg, edges):
    return 0 if pg <= edges[0] else (1 if pg <= edges[1] else 2)


def depth_bucket(d):
    return "shallow(<=3)" if d <= 3 else ("mid(4-8)" if d <= 8 else "deep(>8)")


def allocate(strata: dict, n: int, rng) -> list:
    """Largest-remainder allocation of n slots across strata (by size), then within a stratum
    pick round-robin across morph clusters (cluster diversity first), shuffled inside a cluster."""
    total = sum(len(v) for v in strata.values())
    n = min(n, total)
    raw = {k: len(v) / total * n for k, v in strata.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    rem = n - sum(alloc.values())
    for k in sorted(strata, key=lambda k: -(raw[k] - alloc[k]))[:rem]:
        alloc[k] += 1
    picked = []
    for k, members in strata.items():
        want = min(alloc[k], len(members))
        by_cluster = defaultdict(list)
        for m in members:
            by_cluster[m["cluster"]].append(m)
        for cl in by_cluster:
            rng.shuffle(by_cluster[cl])
        order = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))
        while want > 0 and any(by_cluster[c] for c in order):
            for c in order:
                if by_cluster[c] and want > 0:
                    picked.append(by_cluster[c].pop())
                    want -= 1
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=ROOT / "data/discovery/steered_run2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/steered_run2_manifest")
    args = ap.parse_args()
    run = args.run_dir

    rows = spm.admitted_q3([json.loads(l) for l in
                            open(run / "outcome_ledger.jsonl", encoding="utf-8") if l.strip()])
    npz_path = run / "morph_admissions.npz"
    if not npz_path.exists():
        raise SystemExit(f"missing {npz_path} — run tools/atlas/steered_run2_report.py first")
    z = np.load(npz_path, allow_pickle=False)
    cluster_of = {str(u): int(c) for u, c in zip(z["uids"], z["cluster_strict"])}
    cuts = kc.load_keeper_cuts()

    # attach metadata
    items = []
    for r in rows:
        pg = float(r["p_good"])
        d = int(r["reached_depth"])
        items.append(dict(
            id=r["id"], family=r["family"], cx=r["outcome_cx"], cy=r["outcome_cy"],
            fw=r["outcome_fw"], c=([r["julia_c_re"], r["julia_c_im"]] if r.get("julia_c_re") is not None else None),
            p_good=pg, p_notbad=float(r["p_notbad"]), depth=d,
            cluster=cluster_of.get(r["id"], -1),
            keeper=bool(corn_decode(r["p_notbad"], pg, kc.keeper_cut_for(r["family"], cuts)) == 3),
        ))
    pgs = sorted(x["p_good"] for x in items)
    edges = (np.quantile(pgs, 1 / 3), np.quantile(pgs, 2 / 3)) if len(pgs) >= 3 else (0.33, 0.66)
    for x in items:
        x["pgood_tercile"] = int(pgood_tercile(x["p_good"], edges))
        x["depth_bucket"] = depth_bucket(x["depth"])

    strata = defaultdict(list)
    for x in items:
        strata[(x["pgood_tercile"], x["depth_bucket"])].append(x)

    rng = np.random.default_rng(args.seed)
    picked = allocate(strata, args.n, rng)
    rng.shuffle(picked)

    # render blind tiles + write hidden key
    tiles = args.out_dir / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    key = []
    for i, x in enumerate(picked):
        tile = tiles / f"blind_{i:03d}.jpg"
        loc = spm.loc_of_row(next(r for r in rows if r["id"] == x["id"]))
        spm.render_colored(loc, tile)
        key.append(dict(tile=tile.name, **x))

    (args.out_dir / "manifest_key.json").write_text(json.dumps(dict(
        run=run.name, n=len(picked), n_admissions=len(items), seed=args.seed,
        pgood_tercile_edges=[round(float(e), 5) for e in edges],
        keeper_cuts={k: v.get("t") for k, v in cuts.items()},
        note="HIDDEN KEY — do not show the human labeler; maps blind tile -> truth.",
        entries=key,
    ), indent=2), encoding="utf-8")
    (args.out_dir / "blind_index.json").write_text(json.dumps(dict(
        run=run.name, n=len(picked),
        instructions="Score each tile 1(bad)/2(okay)/3(good). Tiles are shuffled, single-arm; "
                     "no coords/scores are shown. Return {tile: score}.",
        tiles=[e["tile"] for e in key],
    ), indent=2), encoding="utf-8")

    # coverage readout
    cov = defaultdict(int)
    for e in key:
        cov[(e["pgood_tercile"], e["depth_bucket"])] += 1
    print(f"manifest: {len(picked)} / {len(items)} admissions, {len(set(e['cluster'] for e in key))} "
          f"morph clusters covered; keepers in set: {sum(e['keeper'] for e in key)}")
    print("strata coverage (pgood_tercile, depth_bucket) -> count:")
    for k in sorted(cov):
        print(f"  {k} -> {cov[k]}")
    print(f"wrote {args.out_dir/'manifest_key.json'} (hidden) + blind_index.json + tiles/")


if __name__ == "__main__":
    main()
