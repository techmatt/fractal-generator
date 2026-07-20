#!/usr/bin/env python
"""Stratified blind-read manifest for the campaign-1 admissions (breadth + dive legs).

Builds the ~300-tile blind labeling batch the campaign-1 quality read wants:
  * ALL julia admissions (julia:mandelbrot + jm3/4/5) — a census (scarce commodity).
  * ~200 across mandelbrot/mb3/mb4/mb5, stratified by family x leg x pref_loc_v0 tercile,
    with the ranker's TOP slice and its UNCERTAINTY band forced in beside a stratified-
    random slice per family.

Every admission (both legs, 568) is scored ONCE with the deployed pref_loc_v0 ranker
(`tools/ranker/score_locations.LocationRanker`) — which renders the deploy-canonical
640x360 ss2 twilight tile per row (the render pass) and computes v7 + colored-CLIP,
persisting to `data/ranker/campaign1/features.npz`. Terciles/percentiles are then over
the full per-family population, so the sample can target the ranker's operating range.

Writes under `--out-dir` (default out/campaign1_blind):
  tiles/blind_NNN.jpg   shuffled canonical tiles (the labeling units)
  blind_index.json      {run, n, instructions, tiles[]}  — what the human sees (NO metadata)
  manifest_key.json     HIDDEN key: tile -> id / coords / family / leg / pref_loc score+pct+
                        tercile / p_good / depth / sel_reason, plus per-(family,leg,tercile)
                        population + sampled counts so Phase B can stratum-weight the rates.

Then run tools/atlas/build_blind_labeler.py --manifest-dir <out-dir> to bake blind_label.html.

  uv run python tools/atlas/campaign1_manifest.py --n-nonjulia 200 --workers 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools" / "atlas", ROOT / "tools" / "mining", ROOT / "tools" / "scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tools.studies.steered_pilot_morph as spm                # noqa: E402
from tools.ranker.score_locations import LocationRanker, rank_percentiles  # noqa: E402

LEGS = ("breadth", "dive")


def load_admissions(base: Path) -> list[dict]:
    """Admission rows = the ids present in each leg's outcome_feats.npz (the admitted set),
    joined to the leg's outcome_ledger for coords/family/p_good/depth. Adds a `leg` tag."""
    adm = []
    for leg in LEGS:
        ledger = {}
        for line in open(base / leg / "outcome_ledger.jsonl", encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                ledger[r["id"]] = r
        feats = np.load(base / leg / "outcome_feats.npz", allow_pickle=True)
        for rid in feats.files:
            row = ledger.get(rid)
            if row is None:
                raise SystemExit(f"admission {rid} in {leg}/outcome_feats.npz has no ledger row")
            adm.append({**row, "leg": leg})
    return adm


def render_all(adm: list[dict], tile_dir: Path, workers: int) -> None:
    tile_dir.mkdir(parents=True, exist_ok=True)

    def one(row):
        spm.render_colored(spm.loc_of_row(row), tile_dir / f"{row['id']}.jpg")

    todo = [r for r in adm if not (tile_dir / f"{r['id']}.jpg").exists()]
    print(f"render: {len(todo)}/{len(adm)} tiles to render ({workers} workers) ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    have = sum((tile_dir / f"{r['id']}.jpg").exists() for r in adm)
    print(f"render: {have}/{len(adm)} tiles present", flush=True)


def terciles(scores: list[float]) -> tuple[float, float]:
    return (float(np.quantile(scores, 1 / 3)), float(np.quantile(scores, 2 / 3))) \
        if len(scores) >= 3 else (0.0, 0.0)


def tercile_of(s: float, edges: tuple[float, float]) -> int:
    return 0 if s <= edges[0] else (1 if s <= edges[1] else 2)


def largest_remainder(sizes: dict, n: int) -> dict:
    """Allocate n across keys proportional to sizes (largest-remainder), capped at each size."""
    total = sum(sizes.values())
    n = min(n, total)
    if total == 0:
        return {k: 0 for k in sizes}
    raw = {k: v / total * n for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    rem = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: -(raw[k] - alloc[k]))[:rem]:
        alloc[k] += 1
    return {k: min(alloc[k], sizes[k]) for k in sizes}


def sample_family(members: list[dict], quota: int, rng) -> dict:
    """Pick `quota` from one family's members. Guarantees a TOP slice (highest ranker score)
    and an UNCERTAINTY band (nearest the family score median), then fills the remainder with a
    stratified-random draw over (leg, tercile). Returns id -> sel_reason. All picks disjoint."""
    n = len(members)
    quota = min(quota, n)
    picks: dict[str, str] = {}
    if quota == 0:
        return picks
    frac = 0.15
    k_side = max(3, round(frac * quota)) if quota >= 8 else max(1, quota // 4)

    for m in sorted(members, key=lambda m: -m["pref_score"]):     # TOP slice
        if len(picks) >= min(k_side, quota):
            break
        picks[m["id"]] = "top"

    med = float(np.median([m["pref_score"] for m in members]))    # UNCERTAINTY band
    for m in sorted((m for m in members if m["id"] not in picks),
                    key=lambda m: abs(m["pref_score"] - med)):
        if len(picks) >= min(2 * k_side, quota):
            break
        picks[m["id"]] = "uncertainty"

    remaining = [m for m in members if m["id"] not in picks]      # stratified-random fill
    R = quota - len(picks)
    if R > 0 and remaining:
        strata = defaultdict(list)
        for m in remaining:
            strata[(m["leg"], m["tercile"])].append(m)
        alloc = largest_remainder({k: len(v) for k, v in strata.items()}, R)
        for k, cnt in alloc.items():
            pool = strata[k]
            rng.shuffle(pool)
            for m in pool[:cnt]:
                picks[m["id"]] = "stratified"
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=ROOT / "data/discovery/campaign1")
    ap.add_argument("--n-nonjulia", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/campaign1_blind")
    ap.add_argument("--features", type=Path, default=ROOT / "data/ranker/campaign1/features.npz")
    ap.add_argument("--render-only", action="store_true",
                    help="render the 568 canonical tiles and exit (the long pass; no torch)")
    args = ap.parse_args()

    adm = load_admissions(args.base)
    tiles_full = args.out_dir / "tiles_full"
    render_all(adm, tiles_full, max(1, min(4, args.workers)))
    if args.render_only:
        print("render-only: done"); return

    # --- score every admission with the deployed pref_loc_v0 ranker ---------- #
    print(f"scoring {len(adm)} admissions with pref_loc_v0 (tiles pre-rendered) ...", flush=True)
    ranker = LocationRanker()
    scores = ranker.score_rows(adm, tiles_full, persist_npz=args.features)
    for a in adm:
        a["pref_score"] = float(scores[a["id"]])

    # per-family tercile + percentile; global percentile
    gpct = rank_percentiles({a["id"]: a["pref_score"] for a in adm})
    by_fam = defaultdict(list)
    for a in adm:
        by_fam[a["family"]].append(a)
    for fam, members in by_fam.items():
        edges = terciles([m["pref_score"] for m in members])
        fpct = rank_percentiles({m["id"]: m["pref_score"] for m in members})
        for m in members:
            m["tercile"] = tercile_of(m["pref_score"], edges)
            m["pref_pct_family"] = round(fpct[m["id"]], 4)
            m["pref_pct_global"] = round(gpct[m["id"]], 4)
        by_fam[fam] = members

    julia = [a for a in adm if a["family"].startswith("julia:")]
    nonjulia_fams = {f: m for f, m in by_fam.items() if not f.startswith("julia:")}

    # --- sample: julia census + stratified non-julia ------------------------- #
    rng = np.random.default_rng(args.seed)
    quotas = largest_remainder({f: len(m) for f, m in nonjulia_fams.items()}, args.n_nonjulia)
    picked: list[dict] = []
    for a in julia:
        a["sel_reason"] = "census"
        picked.append(a)
    for fam, members in nonjulia_fams.items():
        chosen = sample_family(members, quotas[fam], rng)
        for m in members:
            if m["id"] in chosen:
                m["sel_reason"] = chosen[m["id"]]
                picked.append(m)

    rng.shuffle(picked)

    # --- render blind tiles (copy from tiles_full) + write key + index ------- #
    out_tiles = args.out_dir / "tiles"
    out_tiles.mkdir(parents=True, exist_ok=True)
    key = []
    for i, a in enumerate(picked):
        name = f"blind_{i:03d}.jpg"
        shutil.copyfile(tiles_full / f"{a['id']}.jpg", out_tiles / name)
        key.append(dict(
            tile=name, id=a["id"], family=a["family"], leg=a["leg"],
            cx=a["outcome_cx"], cy=a["outcome_cy"], fw=a["outcome_fw"],
            c=([a["julia_c_re"], a["julia_c_im"]] if a.get("julia_c_re") is not None else None),
            p_good=float(a.get("p_good", float("nan"))),
            p_notbad=float(a.get("p_notbad", float("nan"))),
            depth=int(a.get("reached_depth", 0)),
            pref_score=round(a["pref_score"], 6),
            pref_pct_family=a["pref_pct_family"], pref_pct_global=a["pref_pct_global"],
            pref_tercile=int(a["tercile"]), sel_reason=a["sel_reason"],
        ))

    # population + sampled census per (family, leg, tercile) for Phase B weighting
    pop = defaultdict(int)
    for a in adm:
        pop[(a["family"], a["leg"], int(a["tercile"]))] += 1
    smp = defaultdict(int)
    for e in key:
        smp[(e["family"], e["leg"], e["pref_tercile"])] += 1
    strata_census = [dict(family=f, leg=l, tercile=t, population=pop[(f, l, t)],
                          sampled=smp[(f, l, t)]) for (f, l, t) in sorted(pop)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest_key.json").write_text(json.dumps(dict(
        run="campaign1_blind", n=len(key), n_admissions=len(adm), seed=args.seed,
        n_nonjulia_target=args.n_nonjulia,
        palette=spm.PALETTE, tile="640x360 ss2",
        note="HIDDEN KEY — do not show the human labeler; maps blind tile -> truth.",
        family_quotas={f: quotas[f] for f in nonjulia_fams},
        strata_census=strata_census,
        entries=key,
    ), indent=2), encoding="utf-8")
    (args.out_dir / "blind_index.json").write_text(json.dumps(dict(
        run="campaign1_blind", n=len(key),
        instructions="Score each tile 1(bad) / 2(okay) / 3(good) on wallpaper quality. Tiles "
                     "are shuffled; no coords, scores, family, or leg are shown. Return {tile: score}.",
        tiles=[e["tile"] for e in key],
    ), indent=2), encoding="utf-8")

    # --- coverage readout ---------------------------------------------------- #
    print(f"\nmanifest: {len(key)} tiles  ({len(julia)} julia census + "
          f"{len(key) - len(julia)} non-julia stratified) / {len(adm)} admissions")
    print("per-family sampled:")
    fam_cnt = defaultdict(int)
    for e in key:
        fam_cnt[e["family"]] += 1
    for f in sorted(fam_cnt, key=lambda f: -fam_cnt[f]):
        tot = len(by_fam[f])
        print(f"  {f:22s} {fam_cnt[f]:3d} / {tot:3d}")
    reason = defaultdict(int)
    for e in key:
        reason[e["sel_reason"]] += 1
    print("sel_reason:", dict(reason))
    print(f"\nwrote {args.out_dir/'manifest_key.json'} (HIDDEN) + blind_index.json + tiles/")
    print(f"features -> {args.features}")


if __name__ == "__main__":
    main()
