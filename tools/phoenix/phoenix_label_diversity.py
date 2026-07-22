#!/usr/bin/env python
r"""Phoenix grid labels — §3 theme diversity + §4 ranker transfer (render/CLIP side).

Complements phoenix_label_analysis.py (the compute-only §0/§1/§2/§4a pass). This pass needs
renders + CLIP, so it is separate:

  §3  Morph diversity of the HUMAN-GOOD images. Embed each good image's grayscale morphology
      (library recipe: 640x360 ss2 smooth field -> robust-z tanh -> CLIP), count DISTINCT good
      looks (greedy cos-0.974 near-dup, the library/scheduler definition) across seeds and
      parameter strata, and render a medoid contact sheet of the distinct good looks.

  §4b Ranker transfer. Score the labeled admitted images with the DEPLOYED pref_loc_v1 head
      (v7 + colored_clip), held-out by construction (zero phoenix in its training), and report
      held-out Spearman vs human labels.

Analysis only. Reuses the exact library morph recipe (library_annotate) + the ranker scorer;
retrains nothing. Writes data/discovery/phoenix_grid/diversity_ranker.json + the medoid sheet.

  uv run python tools/phoenix/phoenix_label_diversity.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in ("tools/corpus", "tools/mining", "tools/scoring", "tools/wallpaper",
          "tools/curation", "tools/studies", "tools/ranker"):
    sys.path.insert(0, str(ROOT / p))

import label_store as ls                              # noqa: E402
import location as loc_mod                            # noqa: E402
import library_annotate as la                         # noqa: E402
from colored_clip import load_clip, embed_clip        # noqa: E402
from active_ckpt import auto_maxiter, PALETTE, BIN, JPG_Q  # noqa: E402
from tools.ranker.scorer import RankerScorer          # noqa: E402

RUN = ROOT / "data" / "discovery" / "phoenix_grid" / "grid"
BATCH_ID = "2026-07-21_phoenix_grid"
BATCH = ROOT / "data" / "label_corpus" / "batches" / BATCH_ID
SCRATCH = ROOT / "out" / "phoenix_grid" / "label_analysis"
FCACHE = ROOT / "data" / "library" / "field_cache"
OUT = ROOT / "data" / "discovery" / "phoenix_grid" / "diversity_ranker.json"
SHEET = ROOT / "out" / "phoenix_grid" / "good_look_medoids.png"

NEAR_DUP = 0.974        # library distinct-look cosine
WORKERS = 4             # project cap


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def loc_of(row):
    r = row["render"]
    return loc_mod.Location(
        family="phoenix", cx=r["cx"], cy=r["cy"], fw=r["fw"],
        maxiter=int(r.get("maxiter") or auto_maxiter(float(r["fw"]))),
        c_re=r["c_re"], c_im=r["c_im"],
        family_params={"p_re": r["p_re"], "p_im": r["p_im"],
                       "zm1_re": r["zm1_re"], "zm1_im": r["zm1_im"]})


def load_rows():
    batch = read_jsonl(BATCH / "images.jsonl")
    sidecar = ls.sidecar_for(BATCH_ID)
    allo = {r["id"]: r for r in read_jsonl(RUN / "all_outcomes.jsonl")}
    rows = []
    for r in batch:
        iid = r["image_id"]
        rows.append({"id": iid, "score": ls.resolve_score(r, sidecar), "loc": loc_of(r),
                     "seed": int(iid.split("_")[1]), "branch": r["provenance"]["branch"],
                     "z_class": r["provenance"]["z_class"], "stratum": r["provenance"]["stratum"],
                     "p_good": float((allo.get(iid) or {}).get("p_good",
                                     r["provenance"].get("p_good") or 0.0)),
                     "admitted": iid in allo and (allo[iid].get("decoded_class") == 3
                                                  and allo[iid].get("guard_pass")),
                     "render": r["render"]})
    return rows


# --------------------------------------------------------------------------- #
# Morph embeddings (library recipe) — parallel field render, then batch CLIP.
# --------------------------------------------------------------------------- #
def morph_grays(rows):
    """Grayscale morphology PIL images for `rows`, rendering/caching the smooth field."""
    def one(row):
        try:
            field = la.ensure_field(row["loc"], retain=True, tmp_dir=SCRATCH / "fld",
                                    cache_root=FCACHE)
            return row["id"], la.morph_gray_image(field)
        except Exception as e:
            print(f"  WARN morph {row['id']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return row["id"], None
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for iid, im in ex.map(one, rows):
            out[iid] = im
    return out


def colored_tiles(rows):
    """Twilight_shifted colored tiles (640x360 ss2) for `rows` -> {id: PIL image}."""
    import subprocess
    from PIL import Image

    def one(row):
        tile = SCRATCH / "colored" / f"{row['id']}.jpg"
        tile.parent.mkdir(parents=True, exist_ok=True)
        if not tile.exists():
            loc = row["loc"]
            cmd = [str(BIN), "render-one", "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
                   "--width", "640", "--height", "360", "--supersample", "2",
                   "--maxiter", str(loc.maxiter), "--palette", PALETTE,
                   "--jpg-quality", str(JPG_Q), "--out", str(tile)] + loc_mod.render_one_flags(loc)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  WARN colored {row['id']}: {r.stderr[-120:]}", flush=True)
                return row["id"], None
        try:
            return row["id"], Image.open(tile).convert("RGB")
        except Exception:
            return row["id"], None
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for iid, im in ex.map(one, rows):
            out[iid] = im
    return out


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def greedy_distinct(embs, thr=NEAR_DUP):
    """Greedy near-dup grouping (library DistinctLookTally semantics): a look is distinct if its
    max cosine to all kept reps < thr. Returns (rep_indices, assignment) over rows of `embs`."""
    reps, assign = [], np.full(len(embs), -1)
    for i, e in enumerate(embs):
        if reps:
            sims = np.stack([embs[r] for r in reps]) @ e
            j = int(np.argmax(sims))
            if sims[j] >= thr:
                assign[i] = j
                continue
        assign[i] = len(reps)
        reps.append(i)
    return reps, assign


# --------------------------------------------------------------------------- #
# §3  theme diversity of the good images
# --------------------------------------------------------------------------- #
def diversity(rows, model, tf):
    goods = [r for r in rows if r["score"] == 3]
    grays = morph_grays(goods)
    ok = [r for r in goods if grays[r["id"]] is not None]
    E = l2(embed_clip(model, tf, [grays[r["id"]] for r in ok]).astype(np.float32))

    reps, assign = greedy_distinct(E, NEAR_DUP)
    # coarser theme cuts (agglomerative, average linkage on cosine distance)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    D = pdist(E, metric="cosine")
    Z = linkage(D, method="average")
    theme_cuts = {}
    for cos_thr in (0.90, 0.85, 0.80, 0.70):
        lab = fcluster(Z, t=1.0 - cos_thr, criterion="distance")
        theme_cuts[f"cos{cos_thr:.2f}"] = int(lab.max())

    # distinct-look coverage across seeds / strata
    look_seeds = defaultdict(set)
    look_strata = defaultdict(set)
    look_branch = defaultdict(set)
    for i, r in enumerate(ok):
        g = assign[i]
        look_seeds[g].add(r["seed"]); look_strata[g].add(r["stratum"]); look_branch[g].add(r["branch"])

    # medoid per distinct look (min mean cosine distance within group)
    medoids = []
    groups = defaultdict(list)
    for i in range(len(ok)):
        groups[assign[i]].append(i)
    for g in sorted(groups):
        idx = groups[g]
        sub = E[idx]
        sims = sub @ sub.T
        med = idx[int(np.argmax(sims.mean(1)))]
        medoids.append({"look": int(g), "medoid_id": ok[med]["id"], "size": len(idx),
                        "n_seeds": len(look_seeds[g]), "branches": sorted(look_branch[g]),
                        "seeds": sorted(look_seeds[g])})

    return {
        "n_good": len(goods), "n_good_embedded": len(ok),
        "n_distinct_good_looks": len(reps),
        "n_good_seeds": len({r["seed"] for r in ok}),
        "n_good_branches": sorted({r["branch"] for r in ok}),
        "theme_cuts_coarse": theme_cuts,
        "looks_multi_seed": int(sum(1 for g in groups if len(look_seeds[g]) >= 2)),
        "largest_look_size": max((len(v) for v in groups.values()), default=0),
        "medoids": sorted(medoids, key=lambda m: -m["size"]),
    }, ok, grays, medoids


# --------------------------------------------------------------------------- #
# §4b  ranker transfer
# --------------------------------------------------------------------------- #
def ranker_transfer(rows, model, tf):
    feats = np.load(RUN / "outcome_feats.npz", allow_pickle=False)
    have_v7 = [r for r in rows if r["id"] in feats.files and r["score"] is not None]
    tiles = colored_tiles(have_v7)
    scored = [r for r in have_v7 if tiles[r["id"]] is not None]
    V7 = np.stack([feats[r["id"]] for r in scored]).astype(np.float64)
    COL = embed_clip(model, tf, [tiles[r["id"]] for r in scored]).astype(np.float64)
    scorer = RankerScorer.load()
    rank = scorer.score_matrix({"v7": V7, "colored": COL})
    human = np.array([r["score"] for r in scored], float)
    pg = np.array([r["p_good"] for r in scored], float)
    sp = stats.spearmanr(rank, human)
    sp_pg = stats.spearmanr(pg, human)
    y = (human == 3).astype(int)
    r1 = stats.rankdata(rank)
    auc = float((r1[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))
    return {
        "n_scored": len(scored), "sets": list(scorer.sets), "head": scorer.head,
        "spearman_rank_vs_human": {"rho": float(sp.statistic), "p": float(sp.pvalue)},
        "spearman_pgood_vs_human": {"rho": float(sp_pg.statistic), "p": float(sp_pg.pvalue)},
        "auc_rank_good_vs_rest": auc,
        "note": "held-out by construction: pref_loc_v1 trained on run2+dive+campaign1, zero phoenix.",
    }


# --------------------------------------------------------------------------- #
def build_sheet(ok, grays, medoids, colored_by_id):
    from PIL import Image, ImageDraw
    meds = [m for m in medoids]
    TW, TH, PAD, LBL, GUT, NCOL = 240, 135, 4, 26, 40, 8
    n = len(meds)
    nrow = (n + NCOL - 1) // NCOL
    cw, ch = TW + 2 * PAD, TH + LBL + 2 * PAD
    sheet = Image.new("RGB", (NCOL * cw, GUT + nrow * ch), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 12), f"phoenix grid — {n} DISTINCT good-look medoids (cos {NEAR_DUP}); "
                    f"morphology (twilight tile inset)", fill=(235, 235, 235))
    for k, m in enumerate(meds):
        rr, cc = divmod(k, NCOL)
        x, y = cc * cw + PAD, GUT + rr * ch + PAD
        im = colored_by_id.get(m["medoid_id"]) or grays.get(m["medoid_id"])
        if im is not None:
            sheet.paste(im.resize((TW, TH)), (x, y))
        d.rectangle([x, y + TH, x + TW, y + TH + LBL], fill=(28, 28, 32))
        d.text((x + 2, y + TH + 1),
               f"L{m['look']} n{m['size']} s{m['n_seeds']} {','.join(m['branches'])[:14]}",
               fill=(210, 210, 218))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET)
    return SHEET


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} rows; goods={sum(1 for r in rows if r['score']==3)} "
          f"admitted={sum(1 for r in rows if r['admitted'])}")
    model, tf = load_clip()

    div, ok, grays, medoids = diversity(rows, model, tf)
    print("\n=== §3 diversity ===")
    print(f"  good={div['n_good']} embedded={div['n_good_embedded']} "
          f"DISTINCT good looks (cos {NEAR_DUP})={div['n_distinct_good_looks']}")
    print(f"  across {div['n_good_seeds']} seeds, branches {div['n_good_branches']}")
    print(f"  coarse theme cuts: {div['theme_cuts_coarse']}")
    print(f"  multi-seed looks={div['looks_multi_seed']} largest look size={div['largest_look_size']}")

    # colored tiles for medoids (nicer sheet than gray)
    med_rows = [next(r for r in ok if r["id"] == m["medoid_id"]) for m in medoids]
    col = colored_tiles(med_rows)
    sheet = build_sheet(ok, grays, medoids, col)
    print(f"  medoid sheet -> {sheet}")

    rk = ranker_transfer(rows, model, tf)
    print("\n=== §4b ranker transfer (pref_loc_v1) ===")
    print(f"  n_scored={rk['n_scored']} sets={rk['sets']}")
    print(f"  Spearman(rank, human)={rk['spearman_rank_vs_human']['rho']:.3f} "
          f"(p={rk['spearman_rank_vs_human']['p']:.2e})")
    print(f"  Spearman(p_good, human)={rk['spearman_pgood_vs_human']['rho']:.3f}  "
          f"AUC(rank)={rk['auc_rank_good_vs_rest']:.3f}")

    OUT.write_text(json.dumps({"diversity": div, "ranker": rk,
                               "sheet": str(sheet.relative_to(ROOT))}, indent=2,
                              default=lambda o: int(o) if isinstance(o, np.integer)
                              else float(o) if isinstance(o, np.floating) else str(o)),
                   encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
