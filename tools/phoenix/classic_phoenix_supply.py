#!/usr/bin/env python
r"""classic_phoenix_supply.py — mint a CURRENT-decoded classic-phoenix supply for the library.

The Phase-B grid is VARIED phoenix (swept c/p/z_{-1}); classic phoenix is the original motif
(fixed Ushiki plane, z_{-1}=0). The legacy phoenix ledgers hold classic q3 coordinates decoded
under v6 — inadmissible now (scorer_version=="v6" fails is_current_decoded). This tool mints
current rows AT THE SAME PLACES: it re-renders each legacy q3 coordinate at reframe/deploy
fidelity carrying the classic Ushiki identity, re-scores under the guarded v7 CORN scorer, and
re-decodes at the production phoenix t_good (t_good_for("phoenix")==0.45). Legacy v6 rows are
untouched; this writes a fresh `classic_phoenix`-tagged ledger.

  Stage A  collect   — q3 (decoded_class==3) phoenix coords from the legacy ledgers, coord-dedup.
  Stage B  rescore   — per-coord guarded v7 render+score+decode@0.45 (reframe fidelity, no search),
                       morph-embed distinct tally + 1280-D feature for each q3. Per-coord resume.
  Stage C  topup     — if < MIN_DISTINCT distinct looks survive, a short fixed-classic descent leg
                       (guided-descend --phoenix at Ushiki; cap ~TOPUP_BUDGET_MIN active-min).
  Stage D  finalize  — write outcome_ledger.jsonl (+ feats + summary).

Identity resolves via the Phase-A legacy Ushiki defaults (production_seeder.PHOENIX_*_DEFAULT):
c=(0.5667,0), p=(-0.5,0), z_{-1}=(0,0). Every legacy coord is classic, so all rows share that
identity — distinctness is a MORPHOLOGY property (cos 0.974), not a parameter property.

Durable outputs under data/discovery/classic_phoenix/ (survive rm -r out/*):
  coords.jsonl           collected legacy q3 coords (source + orig id + viewport)
  rescored.jsonl         per-coord v7 rescore result (q3 AND sub-threshold) — resume key
  outcome_ledger.jsonl   admitted q3 classic_phoenix rows (intake-ready)
  outcome_feats.npz      id -> 1280-D v7 feature for each admitted q3
  distinct_looks.npz     run-global morph distinct-look tally (cos 0.974), resumable
  summary.json           counts + config

  uv run python tools/phoenix/classic_phoenix_supply.py --limit 5   # smoke (5 coords, no topup)
  uv run python tools/phoenix/classic_phoenix_supply.py             # full (backgrounded)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "tools", ROOT / "tools" / "atlas", ROOT / "tools" / "atlas_probe",
          ROOT / "tools" / "reframe", ROOT / "tools" / "scoring", ROOT / "tools" / "mining",
          ROOT / "tools" / "corpus", ROOT / "tools" / "wallpaper", ROOT / "tools" / "curation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import phoenix_sampler as psamp          # noqa: E402
import phoenix_grid as pg                 # noqa: E402  (MorphEmbedder, phoenix_outcome_feature, loc factory)
import production_seeder as ps            # noqa: E402
import prescreen                          # noqa: E402
import guard                              # noqa: E402
import reframe                            # noqa: E402
from score_lib import corn_decode         # noqa: E402
from step0_reanalysis import load_frames_by_walk  # noqa: E402
from deficit_scheduler import DistinctLookTally    # noqa: E402

# --- classic Ushiki identity (Phase-A legacy defaults) --------------------------------------- #
CLASSIC_SEED = psamp.Seed(
    c=complex(*ps.PHOENIX_C_DEFAULT), p=complex(*ps.PHOENIX_P_DEFAULT),
    z_m1=complex(*ps.PHOENIX_ZM1_DEFAULT), branch="classic", theta=0.0, offset=0.0, classic=True)

LEGACY_LEDGERS = [
    ("gather_phoenix", ROOT / "data" / "discovery" / "gather" / "phoenix" / "outcome_ledger.jsonl"),
    ("prospect_run1",  ROOT / "data" / "discovery" / "fresh_runs" / "prospect_run1" / "outcome_ledger.jsonl"),
    ("overnight_0713", ROOT / "data" / "discovery" / "fresh_runs" / "overnight_20260713_001420" / "outcome_ledger.jsonl"),
]
RUN_DIR = ROOT / "data" / "discovery" / "classic_phoenix"
SCRATCH = ROOT / "out" / "phoenix" / "classic_supply"
NEAR_DUP = pg.NEAR_DUP_THRESHOLD
MIN_DISTINCT = 15
TOPUP_BUDGET_MIN = 20.0
RENDER_W, RENDER_H, RENDER_SS = reframe.RENDER_W, reframe.RENDER_H, reframe.RENDER_SS


def log(msg: str):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Stage A — collect legacy q3 coords (coord-dedup).
# --------------------------------------------------------------------------- #
def collect_coords():
    seen, coords = set(), []
    per_src = {}
    for tag, path in LEGACY_LEDGERS:
        n_q3 = n_new = 0
        if not path.exists():
            per_src[tag] = {"q3": 0, "new": 0, "missing": True}
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("family") != "phoenix" or r.get("decoded_class") != 3:
                continue
            n_q3 += 1
            key = (str(r["outcome_cx"]), str(r["outcome_cy"]), str(r["outcome_fw"]))
            if key in seen:
                continue
            seen.add(key)
            n_new += 1
            coords.append({"id": f"clphx_{len(coords):04d}", "legacy_source": tag,
                           "legacy_id": r.get("id"), "outcome_cx": key[0], "outcome_cy": key[1],
                           "outcome_fw": key[2], "reached_depth": int(r.get("reached_depth", 0))})
        per_src[tag] = {"q3": n_q3, "new": n_new}
    return coords, per_src


# --------------------------------------------------------------------------- #
# Stage B — single-coord guarded v7 rescore at reframe fidelity (NO reframe search).
# --------------------------------------------------------------------------- #
def _score_coord(scorer, cx, cy, fw, tile: Path):
    """Render the classic-phoenix outcome frame at 640x360 ss2 with the co-located guard field,
    then guarded-score it. Returns (guard_pass, p_notbad, p_good)."""
    loc = reframe.Location(family="phoenix", c_re=repr(CLASSIC_SEED.c.real),
                           c_im=repr(CLASSIC_SEED.c.imag), cx=str(cx), cy=str(cy), fw=str(fw),
                           family_params=pg._fp(CLASSIC_SEED))
    cand = {"cx": str(cx), "cy": str(cy), "fw": float(fw),
            "maxiter": prescreen.auto_maxiter(float(fw))}
    ok, err = reframe._render(loc, cand, tile, RENDER_W, RENDER_H, RENDER_SS)
    if not ok:
        raise RuntimeError(f"render failed: {err}")
    score, notbad, good = scorer.score_paths([tile])[0]
    guard_pass = float(score) > guard.GUARD_SENTINEL + 1e-6
    return guard_pass, float(notbad) if guard_pass else 0.0, float(good) if guard_pass else 0.0


def rescore(coords, scorer, embedder, tally, feats, rescored_path, feats_path, t, limit=None):
    done = set()
    if rescored_path.exists():
        for line in rescored_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [c for c in coords if c["id"] not in done]
    if limit:
        todo = todo[:limit]
    log(f"[rescore] {len(coords)} coords; {len(done)} done, {len(todo)} to score (limit={limit})")
    fh = open(rescored_path, "a", encoding="utf-8")
    t0 = time.time()
    for k, c in enumerate(todo):
        cid = c["id"]
        dwork = SCRATCH / "rescore" / cid
        try:
            gp, nb, g = _score_coord(scorer, c["outcome_cx"], c["outcome_cy"], c["outcome_fw"],
                                     dwork / "tile.jpg")
        except Exception as e:
            log(f"  WARN {cid} render/score failed: {type(e).__name__}: {str(e)[:120]}")
            shutil.rmtree(dwork, ignore_errors=True)
            continue
        decoded = corn_decode(nb, g, t) if gp else None
        is_q3 = gp and decoded == 3
        row = {**c, "family": "phoenix", "decoded_class": decoded, "p_notbad": nb, "p_good": g,
               "t_good": t, "guard_pass": gp, "guard_fail": None if gp else "sentinel",
               **ps.phoenix_ident_fields(), "mix_source": "classic_phoenix", "canon_pgood": g,
               "scorer_version": ps.SCORER_VERSION}
        if is_q3:
            emb = embedder.embed(c["outcome_cx"], c["outcome_cy"], c["outcome_fw"],
                                 CLASSIC_SEED, dwork / "morph")
            distinct = tally.add("phoenix", emb)
            feat = pg.phoenix_outcome_feature(scorer, c["outcome_cx"], c["outcome_cy"],
                                              c["outcome_fw"], CLASSIC_SEED, dwork / "feat.jpg")
            feats[cid] = np.asarray(feat, np.float32)
            row["distinct"] = bool(distinct)
            row["dup_of"] = None
        else:
            row["distinct"] = False
            row["dup_of"] = None
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        tally.save()
        pg._save_feats(feats_path, feats)
        shutil.rmtree(dwork, ignore_errors=True)
        if (k + 1) % 10 == 0 or k + 1 == len(todo):
            el = time.time() - t0
            rate = (k + 1) / el if el else 0
            log(f"[rescore] {len(done)+k+1}/{len(coords)}  q3-distinct={tally.count('phoenix')} "
                f"({rate:.2f} coord/s, ETA {(len(todo)-k-1)/rate/60:.1f}m)" if rate else "")
    fh.close()


# --------------------------------------------------------------------------- #
# Stage C — fixed-classic descent top-up (guided-descend --phoenix at Ushiki).
# --------------------------------------------------------------------------- #
def topup(scorer, embedder, tally, feats, rescored_path, feats_path, t, seed_base, budget_min):
    reframe.DUMP_GUARD_FIELD = True
    loc_of = pg.make_phoenix_loc_of(CLASSIC_SEED)
    fh = open(rescored_path, "a", encoding="utf-8")
    t0 = time.time()
    leg = 0
    n_appended = 0
    while tally.count("phoenix") < MIN_DISTINCT and (time.time() - t0) / 60.0 < budget_min:
        leg += 1
        dwork = SCRATCH / "topup" / f"leg{leg:03d}"
        rng_seed = seed_base + leg
        log(f"[topup] leg {leg} (distinct={tally.count('phoenix')}/{MIN_DISTINCT}, "
            f"elapsed {(time.time()-t0)/60:.1f}/{budget_min}m)")
        try:
            pool = ps.run_phoenix_descent(rng_seed, dwork, n_walks=12)
            by_walk = load_frames_by_walk(pool)
        except Exception as e:
            log(f"  WARN topup leg {leg} descent failed: {type(e).__name__}: {str(e)[:150]}")
            shutil.rmtree(dwork, ignore_errors=True)
            continue
        for wid in sorted(by_walk):
            try:
                rew = ps.harvest_walk_reward(scorer, wid, by_walk[wid], pg.WORKERS,
                                             dwork / "reward", loc_of)
            except Exception as e:
                log(f"    walk {wid} skipped: {type(e).__name__}: {str(e)[:100]}")
                continue
            gp = rew["reward_k3"] > guard.GUARD_SENTINEL + 1e-6
            decoded = corn_decode(rew["p_notbad"], rew["p_good"], t) if gp else None
            if not (gp and decoded == 3):
                continue
            cid = f"clphx_topup_{leg:03d}_{wid:03d}"
            emb = embedder.embed(rew["outcome_cx"], rew["outcome_cy"], rew["outcome_fw"],
                                 CLASSIC_SEED, dwork / "morph")
            distinct = tally.add("phoenix", emb)
            feat = pg.phoenix_outcome_feature(scorer, rew["outcome_cx"], rew["outcome_cy"],
                                              rew["outcome_fw"], CLASSIC_SEED, dwork / "feat.jpg")
            feats[cid] = np.asarray(feat, np.float32)
            row = {"id": cid, "legacy_source": "topup_descent", "legacy_id": None,
                   "outcome_cx": rew["outcome_cx"], "outcome_cy": rew["outcome_cy"],
                   "outcome_fw": rew["outcome_fw"], "reached_depth": int(rew["reached_depth"]),
                   "family": "phoenix", "decoded_class": decoded,
                   "p_notbad": float(rew["p_notbad"]), "p_good": float(rew["p_good"]), "t_good": t,
                   "guard_pass": gp, "guard_fail": None, **ps.phoenix_ident_fields(),
                   "mix_source": "classic_phoenix", "canon_pgood": float(rew["p_good"]),
                   "scorer_version": ps.SCORER_VERSION, "distinct": bool(distinct), "dup_of": None}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_appended += 1
            tally.save()
            pg._save_feats(feats_path, feats)
        shutil.rmtree(dwork, ignore_errors=True)
    fh.close()
    log(f"[topup] done: {leg} legs, +{n_appended} q3 appended, "
        f"distinct now {tally.count('phoenix')}")
    return n_appended


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="cap coords scored (smoke)")
    ap.add_argument("--no-topup", action="store_true", help="skip the descent top-up leg")
    ap.add_argument("--topup-budget", type=float, default=TOPUP_BUDGET_MIN)
    ap.add_argument("--seed", type=int, default=71000, help="descent RNG base for the top-up")
    args = ap.parse_args(argv)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = ps.t_good_for("phoenix")

    coords, per_src = collect_coords()
    (RUN_DIR / "coords.jsonl").write_text(
        "\n".join(json.dumps(c) for c in coords) + "\n", encoding="utf-8")
    log(f"=== classic phoenix supply  (t_good={t}) ===")
    log(f"[collect] legacy q3: {per_src} -> {len(coords)} unique coords")

    reframe.DUMP_GUARD_FIELD = True
    assert reframe.GUARD_FIELD_SUFFIX == guard.FIELD_SIDECAR_SUFFIX
    scorer = guard.make_guarded_scorer(ps.SCORER_PATH)
    embedder = pg.MorphEmbedder(SCRATCH)
    rescored_path = RUN_DIR / "rescored.jsonl"
    feats_path = RUN_DIR / "outcome_feats.npz"
    looks_path = RUN_DIR / "distinct_looks.npz"
    tally = DistinctLookTally(looks_path, threshold=NEAR_DUP)
    feats = {}
    if feats_path.exists():
        with np.load(feats_path, allow_pickle=False) as z:
            feats = {k: z[k].copy() for k in z.files}

    rescore(coords, scorer, embedder, tally, feats, rescored_path, feats_path, t, limit=args.limit)

    n_distinct = tally.count("phoenix")
    topup_added = 0
    if not args.no_topup and not args.limit and n_distinct < MIN_DISTINCT:
        log(f"[topup] {n_distinct} < {MIN_DISTINCT} distinct — running fixed-classic descent leg")
        topup_added = topup(scorer, embedder, tally, feats, rescored_path, feats_path, t,
                            args.seed, args.topup_budget)

    # Stage D — finalize: ledger = q3 rows from rescored.jsonl
    rows = [json.loads(l) for l in rescored_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    q3 = [r for r in rows if r.get("guard_pass") and r.get("decoded_class") == 3]
    with open(RUN_DIR / "outcome_ledger.jsonl", "w", encoding="utf-8") as f:
        for r in q3:
            f.write(json.dumps(r) + "\n")
    n_distinct = sum(1 for r in q3 if r.get("distinct"))
    summary = {
        "t_good": t, "scorer_version": ps.SCORER_VERSION, "near_dup_threshold": NEAR_DUP,
        "legacy_sources": per_src, "coords_unique": len(coords),
        "rescored": len(rows), "admissions_q3": len(q3), "distinct_looks": n_distinct,
        "topup_added": topup_added, "min_distinct_target": MIN_DISTINCT,
        "classic_identity": {"c": ps.PHOENIX_C_DEFAULT, "p": ps.PHOENIX_P_DEFAULT,
                             "z_m1": ps.PHOENIX_ZM1_DEFAULT},
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\n=== CLASSIC PHOENIX SUPPLY ===")
    log(f"  rescored {len(rows)} coords -> {len(q3)} q3 admissions, {n_distinct} distinct looks")
    log(f"  topup added {topup_added} | durable -> {RUN_DIR}")
    if n_distinct < MIN_DISTINCT and not args.limit:
        log(f"  WARNING: {n_distinct} < {MIN_DISTINCT} distinct target (top-up budget exhausted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
