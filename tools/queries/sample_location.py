"""Per-location render sampler: for one location, produce a ranked top-18 "good pool"
of rendered-colorings (palette + params) under the v2 scorer.

This builds the PER-LOCATION stage only. No meta layer, no cross-location diversity
selection (that's a later layer's job — global diversity is deferred by design). The
within-location v2 ranking is the honest signal here: every render in a location's
search shares the location, so v2's within-query ranking is valid and score comparisons
*within a location* are meaningful (the ranking-only head is non-comparable only ACROSS
locations). No absolute good/bad cutoff exists (the head is uncalibrated) so the pool is
just top-18 by within-location score.

Everything reuses the warmstart generator / pilot / colormap seams — no coloring path is
re-implemented and no fractal is recomputed per candidate:
  * once-per-location ss2 field dump + cache (assemble_queries.ensure_field, out/fields/),
  * cheap recolor of the cached field (colormap.render_candidate + stretch_field prep),
  * the sampler's palette/param draw (query_sampler.sample_candidate — correct cyclic vs
    non-cyclic knob handling),
  * feature-space farthest-point over the 777 pool (palette_features.distance_matrix +
    farthest_point_order),
  * v2 scoring via the scorer's own data.build_transform(train=False) + train.build_model
    loading data/queries/scorer/v2/model_best.pt (reuses query_batch_gen.score_frames).

Algorithm (per location):
  Stage 0  gen-0: draw 60 palettes by feature-space FP (diverse-by-construction — a bad
           random roll can't starve palette variety; palette exploration happens ONLY
           here, refinement never discovers a new palette), one random param draw per
           palette -> 60 renders, score all 60.
  Stage 1  beam: top-18 by v2 -> 18 palette-distinct lineages (fixed palette + best param).
  Stage 2  shallow refinement, SWEPT over R rounds coarse-to-fine: each round draw K=8
           param variants around each lineage's current best (round 1 broad, later rounds
           tighter — annealed width), within the fixed palette, clamped to valid ranges
           (perturb rev/gamma; phase/n_cycles for cyclic only). Per-lineage elitism
           (keep parent-or-best-variant) + per-lineage early-stop (freeze on a
           no-improvement round). One R=2 pass is run; R=0 (gen-0 top-18, the baseline)
           and R=1 are its snapshots.
  Stage 3  output: the 18 refined lineage-bests, ranked by v2, as re-renderable config
           records + rendered images.

The run is partly to MEASURE whether refinement earns its keep: the report prints
score-lift per round AND a visual before/after (the arbiter — is lift visible, or is v2
chasing its own variance on a weak within-palette axis?), plus a render-space movement
metric (gen-0 -> R=2 mean CIEDE2000) so numeric-only lift is distinguishable from real
change. No conclusions drawn here — Matt eyeballs the strips and sets R.

    uv run python tools/queries/sample_location.py --estimate   # runtime est + exit
    uv run python tools/queries/sample_location.py              # full run (4 locations)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_sampler as qs                 # noqa: E402  (pool, sampler, ranges, sample_candidate)
import assemble_queries as aq              # noqa: E402  (ensure_field, _field_key)
import regenerate_coldstart_v2 as rc       # noqa: E402  (thumb_lab, render_space_dmat — render-space CIEDE2000)
import query_batch_gen as P                # noqa: E402  (score_frames — v2/v1-agnostic deploy-transform scorer)
import diversity_diagnostic as dd          # noqa: E402  (single_linkage_clusters)
sys.path.insert(0, str(qs.ROOT / "tools" / "palettes"))
import palette_features as pf              # noqa: E402  (distance_matrix, farthest_point_order — FEATURE space)
sys.path.insert(0, str(HERE / "scorer"))
import train as ST                         # noqa: E402  (build_model)

cm = qs.cm
ROOT = qs.ROOT
V2_DIR = ROOT / "data" / "queries" / "scorer" / "v2"
OUT_DIR = ROOT / "out" / "sampler_eval"

# --- pipeline constants ----------------------------------------------------
N_GEN0 = 60          # gen-0 palettes drawn by feature-space FP over the 777 pool
TOP_KEEP = 18        # beam width -> lineages -> final pool size
K_VARIANTS = 8       # param variants drawn per lineage per refinement round
R_MAX = 2            # deepest refinement swept (R in {0,1,2}; 0/1 are snapshots of this pass)
DEFAULT_SEED = 7     # --seed default; threads gen-0 param draws + refinement variants (repro)

# Refinement perturbation schedule (coarse-to-fine). Round r (1-indexed) uses
# scale = ANNEAL**(r-1): round 1 broad, round 2 tighter. gamma perturbed in LOG space.
ANNEAL = 0.5
GAMMA_LOG_SIGMA0 = 0.40   # 1-sigma log-gamma step at round 1 (exp(0.4)~=1.49, ~+-50% gamma)
PHASE_SIGMA0 = 0.20       # cyclic phase gaussian step, round 1
REV_FLIP_P0 = 0.30        # non-cyclic reverse flip prob, round 1
NCYC_FLIP_P0 = 0.30       # cyclic n_cycles {1<->2} flip prob, round 1

# --- the four test locations ----------------------------------------------
# Stored test anchor: label-3 Julia (c fixed; cx/cy/fw is the z-plane viewport). maxiter
# 800 matches this c's julia_ladder_j0 rung.
ANCHOR = cm.LocationRef(
    kind="julia",
    cx="0.4104135054546244", cy="0.20967482476903096", fw="0.5622541254857749",
    maxiter=800,
    c_re="-0.07810228973371881", c_im="-0.6514609012382414",
)


# ===========================================================================
# v2 scorer.
# ===========================================================================

def load_v2(device):
    model = ST.build_model().to(device)
    ck = torch.load(V2_DIR / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("epoch")


# ===========================================================================
# Location selection: anchor + 3 varied score-{2,3} corpus locations.
# ===========================================================================

def select_locations(pool, seed):
    """The stored anchor plus 3 varied corpus locations (mix mandelbrot/julia, varied
    complexity via a spread over fw). Deterministic from `seed`."""
    by_fam = pool.by_family()
    rng = np.random.default_rng(seed)

    def pick_varied(cands, k):
        """k locations spread across the fw (zoom/complexity) range: sort by fw, cut into
        k quantile bins, draw one per bin."""
        cands = sorted(cands, key=lambda pl: float(pl.ref.fw))
        out = []
        n = len(cands)
        for b in range(k):
            lo, hi = b * n // k, (b + 1) * n // k
            j = int(rng.integers(lo, max(lo + 1, hi)))
            out.append(cands[min(j, n - 1)])
        return out

    mand = pick_varied(by_fam.get("mandelbrot", []), 2)
    jul = pick_varied(by_fam.get("julia", []), 1)
    corpus = mand + jul

    locs = [("anchor_julia", ANCHOR)]
    for i, pl in enumerate(corpus):
        locs.append((f"corpus{i}_{pl.ref.kind}", pl.ref))
    return locs


# ===========================================================================
# Candidate config + perturbation.
# ===========================================================================

def _cfg(ref, palette, reverse, log_premap, gamma, phase, n_cycles):
    return cm.CandidateConfig(
        palette=palette, location=qs.loc_mod.to_location_ref(ref),
        eval_width=qs.EVAL_WIDTH, eval_height=qs.EVAL_HEIGHT,
        reverse=reverse, log_premap=log_premap, gamma=gamma,
        phase=phase, n_cycles=n_cycles, filter=qs.CANDIDATE_FILTER,
    )


def perturb(cfg, ptype, rng, scale):
    """Draw one param variant around `cfg` within its fixed palette. gamma perturbed in
    log space; cyclic gets phase/n_cycles jitter, non-cyclic gets reverse flips. All
    clamped to valid ranges. `scale` anneals the widths across rounds."""
    gamma = cfg.gamma * math.exp(rng.normal(0.0, GAMMA_LOG_SIGMA0 * scale))
    gamma = float(min(max(gamma, qs.GAMMA_LO), qs.GAMMA_HI))
    if ptype == "cyclic":
        phase = float((cfg.phase + rng.normal(0.0, PHASE_SIGMA0 * scale)) % 1.0)
        n_cycles = cfg.n_cycles
        if rng.random() < NCYC_FLIP_P0 * scale:
            n_cycles = 2 if cfg.n_cycles == 1 else 1
        return _cfg(cfg.location, cfg.palette, False, cfg.log_premap, gamma, phase, n_cycles)
    reverse = cfg.reverse
    if rng.random() < REV_FLIP_P0 * scale:
        reverse = not cfg.reverse
    return _cfg(cfg.location, cfg.palette, reverse, cfg.log_premap, gamma, 0.0, 1)


def recolor(fld, cfg, lib, prep):
    return cm.render_candidate(fld, cfg, lib, prep=prep)


# ===========================================================================
# gen-0 palette draw — feature-space farthest-point over the 777 pool.
# ===========================================================================

def gen0_palettes(sampler, k):
    """`k` palette names spread by farthest-point over palette-FEATURE space (OKLab
    trajectory distance_matrix, NOT render space). Deterministic (FP seeds on the two
    most-distant), so gen-0 palette coverage is identical across locations by design."""
    names = [n for n in sampler.feats if n in sampler.library.colormaps]
    D = pf.distance_matrix(sampler.feats, names)
    return pf.farthest_point_order(names, k=k, dmat=D), names, D


def gen0_spread(sel_names, all_names, D):
    """Feature-space spread sanity of the drawn gen-0 set: min/mean nearest-neighbor
    distance + effective-distinct (single-linkage at the sampler's near-dup eps)."""
    idx = [all_names.index(n) for n in sel_names]
    sub = D[np.ix_(idx, idx)]
    n = len(idx)
    iu = np.triu_indices(n, 1)
    pair = sub[iu]
    off = sub + np.eye(n) * 1e9
    nn = off.min(axis=1)   # nearest-neighbor distance per palette
    eff = dd.single_linkage_clusters(sub, qs.DEDUP_EPS)
    return {
        "n": n,
        "min_pairwise": float(pair.min()),
        "mean_pairwise": float(pair.mean()),
        "min_nn": float(nn.min()),
        "mean_nn": float(nn.mean()),
        "dedup_eps": qs.DEDUP_EPS,
        "effective_distinct": int(eff),
    }


# ===========================================================================
# The per-location run: gen-0 -> beam -> swept refinement.
# ===========================================================================

def run_location(label, ref, lib, sampler, model, device, seed, retain_all=False):
    """Per-location gen-0 -> beam -> swept refinement.

    `retain_all` (opt-in; default False leaves the validated return untouched) adds
    `res["all_candidates"]`: EVERY evaluated candidate — all 60 gen-0 draws plus every
    refinement variant across all rounds — as re-renderable records
    `{palette, palette_type, gen, lineage, score, survivor, config}` (`config` is the
    live `cm.CandidateConfig`; `gen` 0 == gen-0, r == refinement round r; `lineage` ==
    the palette, which is lineage-distinct). This is the full within-location pref-v2
    gradient the wallpaper-quality bootstrap strata-samples over — no images retained."""
    stem = aq._field_key(ref)
    fld, _ = aq.ensure_field(ref)
    prep = cm.stretch_field(fld)
    rng = np.random.default_rng(int(hashlib.sha1(f"{stem}|{seed}".encode()).hexdigest()[:16], 16))

    # --- Stage 0: gen-0 (60 palettes FP + one random param each) ---
    pal_names, all_names, Dfeat = gen0_palettes(sampler, N_GEN0)
    gen0_cfgs = [qs.sample_candidate(ref, rng, sampler, palette=p, canonical=False) for p in pal_names]
    gen0_imgs = [recolor(fld, c, lib, prep) for c in gen0_cfgs]
    gen0_scores = P.score_frames(model, gen0_imgs, device)

    # --- Stage 1: beam -> top-18 lineages ---
    order = sorted(range(len(gen0_scores)), key=lambda i: -gen0_scores[i])[:TOP_KEEP]

    all_candidates = None
    if retain_all:
        keep = set(order)
        all_candidates = [
            {"palette": gen0_cfgs[i].palette, "palette_type": lib.palette_type(gen0_cfgs[i].palette),
             "gen": 0, "lineage": gen0_cfgs[i].palette, "score": float(gen0_scores[i]),
             "survivor": i in keep, "config": gen0_cfgs[i]}
            for i in range(len(gen0_cfgs))
        ]
    lineages = []
    for i in order:
        lineages.append({
            "palette": gen0_cfgs[i].palette,
            "ptype": lib.palette_type(gen0_cfgs[i].palette),
            "gen0_cfg": gen0_cfgs[i], "gen0_score": gen0_scores[i], "gen0_img": gen0_imgs[i],
            "best_cfg": gen0_cfgs[i], "best_score": gen0_scores[i], "best_img": gen0_imgs[i],
            "frozen": False,
            "round_scores": [gen0_scores[i]],   # index r == score after round r (0 == gen-0)
        })

    # snapshots[r] = [best_score per lineage after round r]; round 0 == gen-0.
    def snapshot():
        return [l["best_score"] for l in lineages]

    snapshots = [snapshot()]

    # --- Stage 2: swept coarse-to-fine refinement (per-lineage elitism + early-stop) ---
    for r in range(1, R_MAX + 1):
        scale = ANNEAL ** (r - 1)
        active = [l for l in lineages if not l["frozen"]]
        # Draw + recolor every variant across all active lineages, score in ONE batch.
        variants = []   # (lineage, cfg, img)
        for l in active:
            for _ in range(K_VARIANTS):
                cfg = perturb(l["best_cfg"], l["ptype"], rng, scale)
                variants.append((l, cfg, recolor(fld, cfg, lib, prep)))
        if variants:
            vscores = P.score_frames(model, [v[2] for v in variants], device)
            if retain_all:
                for (l, cfg, _img), s in zip(variants, vscores):
                    all_candidates.append(
                        {"palette": l["palette"], "palette_type": l["ptype"],
                         "gen": r, "lineage": l["palette"], "score": float(s),
                         "survivor": True, "config": cfg})
            per_lin = {}
            for (l, cfg, img), s in zip(variants, vscores):
                cur = per_lin.get(id(l))
                if cur is None or s > cur[0]:
                    per_lin[id(l)] = (s, cfg, img)
            for l in active:
                s, cfg, img = per_lin[id(l)]
                if s > l["best_score"]:          # strict improvement -> elitism keeps variant
                    l["best_score"], l["best_cfg"], l["best_img"] = s, cfg, img
                else:                            # no improvement -> freeze lineage
                    l["frozen"] = True
        for l in lineages:
            l["round_scores"].append(l["best_score"])
        snapshots.append(snapshot())

    # --- render-space movement gen-0 -> R2 (did the picture visibly change?) ---
    moves = []
    for l in lineages:
        d = float(rc.render_space_dmat([rc.thumb_lab(l["gen0_img"]), rc.thumb_lab(l["best_img"])])[0, 1])
        l["move_de"] = d
        moves.append((d, l["best_score"] > l["gen0_score"]))

    return {
        "label": label, "ref": ref, "stem": stem,
        "pal_names": pal_names, "gen0_scores": gen0_scores,
        "spread": gen0_spread(pal_names, all_names, Dfeat),
        "lineages": lineages, "snapshots": snapshots, "moves": moves,
        "all_candidates": all_candidates,
    }


# ===========================================================================
# Reporting: lift table + before/after strip + R=2 top-18 contact sheet.
# ===========================================================================

def lift_table(res):
    """Per-round: beam best + mean/median lineage-best (valid within-location)."""
    rows = []
    for r, snap in enumerate(res["snapshots"]):
        a = np.asarray(snap, dtype=np.float64)
        rows.append({"R": r, "beam_best": float(a.max()),
                     "mean": float(a.mean()), "median": float(np.median(a))})
    return rows


def _thumb(img, w):
    im = Image.fromarray(img).convert("RGB")
    h = round(w * im.height / im.width)
    return im.resize((w, h), Image.BILINEAR), h


def before_after_strip(res, path, tw=300, pad=6, bar=30, head=22):
    """Two rows: gen-0 top-6 (by gen-0 score) over R=2 top-6 (by R=2 score). The visual
    arbiter for whether refinement produces VISIBLE improvement."""
    lin = res["lineages"]
    gen0_top = sorted(lin, key=lambda l: -l["gen0_score"])[:6]
    r2_top = sorted(lin, key=lambda l: -l["best_score"])[:6]
    rows = [("gen-0 top-6", gen0_top, "gen0_img", "gen0_score"),
            ("R=2 refined top-6", r2_top, "best_img", "best_score")]
    _, th = _thumb(lin[0]["gen0_img"], tw)
    W = 6 * tw + 7 * pad
    H = head + 2 * (head + th + bar + pad) + pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), f"{res['label']}  ({res['ref'].kind})  gen-0 vs R=2  [v2 within-location score]",
              fill=(235, 235, 235))
    y = head
    for title, lins, imk, sk in rows:
        draw.text((pad, y + 2), title, fill=(210, 210, 160))
        yy = y + head
        for c, l in enumerate(lins):
            x = pad + c * (tw + pad)
            th_im, _ = _thumb(l[imk], tw)
            sheet.paste(th_im, (x, yy))
            draw.text((x + 2, yy + th + 2), f"{l['palette'][:26]}", fill=(220, 220, 160))
            mv = f" dE{l['move_de']:.1f}" if imk == "best_img" else ""
            draw.text((x + 2, yy + th + 15), f"s={l[sk]:.3f}{mv}", fill=(160, 200, 230))
        y = yy + th + bar + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def contact_18(res, path, tw=300, pad=6, bar=30, head=22, cols=6):
    """R=2 top-18 pool, ranked by v2 score (score-sorted)."""
    lin = sorted(res["lineages"], key=lambda l: -l["best_score"])
    rows = (len(lin) + cols - 1) // cols
    _, th = _thumb(lin[0]["best_img"], tw)
    W = cols * tw + (cols + 1) * pad
    H = head + rows * (th + bar + pad) + pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), f"{res['label']}  R=2 top-18 (ranked by v2 within-location score)",
              fill=(235, 235, 235))
    for i, l in enumerate(lin):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = head + r * (th + bar + pad)
        th_im, _ = _thumb(l["best_img"], tw)
        sheet.paste(th_im, (x, y))
        cfg = l["best_cfg"]
        draw.text((x + 2, y + th + 2), f"#{i+1} {l['palette'][:22]}", fill=(220, 220, 160))
        sub = (f"s={l['best_score']:.3f} g{cfg.gamma:.2f}"
               f"{' ph%.2f' % cfg.phase if l['ptype']=='cyclic' else ''}"
               f"{' n2' if cfg.n_cycles==2 else ''}{' rev' if cfg.reverse else ''}")
        draw.text((x + 2, y + th + 15), sub, fill=(160, 200, 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_records(res, out_loc, seed):
    """Ranked, re-renderable config records per R snapshot (R=2 ranked = the pool)."""
    lin = res["lineages"]
    def rec(l, score, cfg):
        return {"palette": l["palette"], "palette_type": l["ptype"],
                "score": score, "config": json.loads(cfg.to_json())}
    r2_ranked = sorted(lin, key=lambda l: -l["best_score"])
    out = {
        "label": res["label"],
        "location": {"family": res["ref"].kind, "cx": res["ref"].cx, "cy": res["ref"].cy,
                     "fw": res["ref"].fw, "maxiter": res["ref"].maxiter,
                     "c_re": res["ref"].c_re, "c_im": res["ref"].c_im,
                     "family_params": qs.loc_mod.params_of(res["ref"])},
        "constants": {"N_GEN0": N_GEN0, "TOP_KEEP": TOP_KEEP, "K_VARIANTS": K_VARIANTS,
                      "R_MAX": R_MAX, "seed": seed, "eval": [qs.EVAL_WIDTH, qs.EVAL_HEIGHT],
                      "ss": qs.CANDIDATE_SS},
        "gen0_spread": res["spread"],
        "lift": lift_table(res),
        "pool_R2_ranked": [rec(l, l["best_score"], l["best_cfg"]) for l in r2_ranked],
        "pool_R0_ranked": [rec(l, l["gen0_score"], l["gen0_cfg"])
                           for l in sorted(lin, key=lambda l: -l["gen0_score"])],
        "movement": {
            "gen0_to_R2_de": [l["move_de"] for l in lin],
            "median_de": float(np.median([l["move_de"] for l in lin])),
            "median_de_improved": float(np.median([d for d, imp in res["moves"] if imp]))
                                  if any(imp for _, imp in res["moves"]) else 0.0,
            "n_improved": int(sum(1 for _, imp in res["moves"] if imp)),
        },
    }
    (out_loc / "records.json").write_text(json.dumps(out, indent=1))
    return out


# ===========================================================================
# Paired per-lineage before/after sheet (cheap re-render from records.json).
#
# The set-vs-set before_after.png entangles within-lineage refinement (what we judge)
# with reranking (which palette floats up). This pairs the SAME palette's gen-0 render
# (top) against its R=2 render (bottom) in one column, so a column shows exactly what
# refinement did to one lineage. Pure re-render: records.json already carries each
# lineage's gen-0 config+score (pool_R0_ranked) and R=2 config+score (pool_R2_ranked),
# joinable by palette (lineages are palette-distinct).
# ===========================================================================

def _ref_from_records(rec):
    """Rebuild the canonical Location from a records.json location block. Carries
    family_params (Phoenix's `p`) so the re-dumped field / render args are exact."""
    L = rec["location"]
    return qs.loc_mod.Location(
        family=L["family"], cx=L["cx"], cy=L["cy"], fw=L["fw"],
        maxiter=int(L["maxiter"]), c_re=L.get("c_re"), c_im=L.get("c_im"),
        family_params=L.get("family_params") or {},
    )


def paired_before_after(records_path, out_path, lib, tw=260, pad=6, bar=44, head=22,
                        grp=18, left=44, n_top=8, n_movers=4):
    """Rebuild the paired before/after sheet for one location from its records.json.

    Columns = lineages, two rows each (gen-0 render on top, R=2 render below, SAME
    palette). Shows the R=2 top-`n_top` by final score, plus the `n_movers` highest-dE
    movers not already in that set (labeled as a second group — big movers are where
    noise-climbing would show most). Recolors at the sampler's exact settings from the
    stored config (byte-identical re-render). Returns the built row dicts."""
    rec = json.loads(Path(records_path).read_text())
    ref = _ref_from_records(rec)
    fld, _ = aq.ensure_field(ref)                      # cache hit -> 0s
    prep = cm.stretch_field(fld)

    r0 = {e["palette"]: e for e in rec["pool_R0_ranked"]}
    lines = []                                          # per lineage: joined gen-0 + R2
    for e2 in rec["pool_R2_ranked"]:
        pal = e2["palette"]
        e0 = r0[pal]
        cfg0 = cm.CandidateConfig.from_json(json.dumps(e0["config"]))
        cfg2 = cm.CandidateConfig.from_json(json.dumps(e2["config"]))
        img0 = recolor(fld, cfg0, lib, prep)
        img2 = recolor(fld, cfg2, lib, prep)
        de = float(rc.render_space_dmat([rc.thumb_lab(img0), rc.thumb_lab(img2)])[0, 1])
        lines.append({"palette": pal, "s0": e0["score"], "s2": e2["score"], "de": de,
                      "cfg2": cfg2, "ptype": e2["palette_type"], "img0": img0, "img2": img2})

    # Selection: R=2 top-n_top by final score (rec order is already R2-score-desc), then
    # the highest-dE movers not already shown.
    top = lines[:n_top]
    top_pals = {l["palette"] for l in top}
    movers = sorted((l for l in lines if l["palette"] not in top_pals),
                    key=lambda l: -l["de"])[:n_movers]
    cols = [("top", l) for l in top] + [("mover", l) for l in movers]

    _, th = _thumb(lines[0]["img0"], tw)
    W = left + len(cols) * (tw + pad) + pad
    H = head + grp + 2 * th + bar + 3 * pad
    sheet = Image.new("RGB", (W, H), (22, 22, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 4), f"{rec['label']}  ({ref.kind})  PAIRED per-lineage gen-0 (top) vs R=2 "
                        f"(bottom), same palette  [v2 within-location score]", fill=(235, 235, 235))
    yA = head + grp
    yB = yA + th + pad
    draw.text((4, yA + th // 2 - 6), "gen-0", fill=(200, 200, 160))
    draw.text((4, yB + th // 2 - 6), "R=2", fill=(200, 200, 160))

    # group header spans
    n_top_shown = sum(1 for g, _ in cols if g == "top")
    x_top0 = left + pad
    draw.text((x_top0, head + 2), f"R=2 top-{n_top_shown} (by final score)", fill=(150, 220, 160))
    if movers:
        x_mv0 = left + pad + n_top_shown * (tw + pad)
        draw.line([(x_mv0 - pad // 2, head), (x_mv0 - pad // 2, H - pad)], fill=(90, 90, 100), width=2)
        draw.text((x_mv0, head + 2), f"top-{len(movers)} dE movers (not in top-{n_top_shown})",
                  fill=(230, 190, 140))

    for c, (grp_name, l) in enumerate(cols):
        x = left + pad + c * (tw + pad)
        tA, _ = _thumb(l["img0"], tw)
        tB, _ = _thumb(l["img2"], tw)
        sheet.paste(tA, (x, yA))
        sheet.paste(tB, (x, yB))
        cfg = l["cfg2"]
        knob = (f"g{cfg.gamma:.2f}"
                f"{' ph%.2f' % cfg.phase if l['ptype'] == 'cyclic' else ''}"
                f"{' n2' if cfg.n_cycles == 2 else ''}{' rev' if cfg.reverse else ''}")
        yC = yB + th + 2
        draw.text((x + 2, yC), l["palette"][:24], fill=(220, 220, 160))
        draw.text((x + 2, yC + 12), f"s {l['s0']:.2f} -> {l['s2']:.2f}  dE{l['de']:.1f}",
                  fill=(160, 200, 230))
        draw.text((x + 2, yC + 24), knob, fill=(140, 175, 205))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return cols


def rebuild_paired_sheets():
    """Re-render before_after_paired.png for every location dir under OUT_DIR that has a
    records.json. No sampler re-run — gen-0 configs are already persisted."""
    lib = qs.load_pool_library()
    built = []
    for rp in sorted(OUT_DIR.glob("*/records.json")):
        out_loc = rp.parent
        out_path = out_loc / "before_after_paired.png"
        cols = paired_before_after(rp, out_path, lib)
        n_mv = sum(1 for g, _ in cols if g == "mover")
        print(f"[paired] {out_loc.name}: {len(cols)} columns ({len(cols)-n_mv} top + {n_mv} movers)"
              f" -> {out_path}")
        built.append(out_path)
    if not built:
        print(f"[paired] no records.json found under {OUT_DIR} — run the sampler first")
    return built


# ===========================================================================
# Runtime estimate.
# ===========================================================================

def estimate(locs, lib, sampler, model, device):
    ref0 = locs[0][1]
    fld0, dump0 = aq.ensure_field(ref0)
    prep0 = cm.stretch_field(fld0)
    p0 = sampler.sample_palette(np.random.default_rng(0))[0]
    c0 = _cfg(ref0, p0, False, "none", 1.0, 0.0, 1)
    t0 = time.time(); img0 = recolor(fld0, c0, lib, prep0); recolor_s = time.time() - t0
    t0 = time.time(); P.score_frames(model, [img0], device); score_s = time.time() - t0

    need_dump = sum(
        0 if ((aq.OUT_FIELDS / f"{aq._field_key(r)}.bin").exists()
               and (aq.OUT_FIELDS / f"{aq._field_key(r)}.json").exists()) else 1
        for _, r in locs)
    per_loc = N_GEN0 + R_MAX * TOP_KEEP * K_VARIANTS      # worst case (no early-stop)
    total = per_loc * len(locs)
    est_s = need_dump * (dump0 if dump0 > 0 else 20.0) + total * (recolor_s + score_s / 64) * 1.15
    print(f"[sampler] est: {need_dump} field dumps + <= {total} recolors "
          f"(<= {per_loc}/loc x {len(locs)} locs) @~{recolor_s*1000:.0f}ms recolor "
          f"=> <= ~{est_s/60:.1f} min (early-stop typically cuts refinement)")
    return (fld0, prep0)


# ===========================================================================
# Driver.
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Per-location render sampler (top-18 good pool), R sweep.")
    ap.add_argument("--estimate", action="store_true", help="print runtime estimate and exit")
    ap.add_argument("--rebuild-sheets", action="store_true",
                    help="re-render before_after_paired.png from existing records.json (no sampler run)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"seed for gen-0 param draws + refinement variants (default {DEFAULT_SEED}); "
                         "a given seed reproduces the draw stream (GPU scoring is still nondeterministic)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if args.rebuild_sheets:
        rebuild_paired_sheets()
        return

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    pool = qs.LocationPool.from_corpus()
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = load_v2(device)
    print(f"[sampler] {pool.report()}")
    print(f"[sampler] loaded v2 model_best.pt (epoch {epoch}) on {device.type}  seed={seed}")

    locs = select_locations(pool, seed)
    print(f"[sampler] locations ({len(locs)}):")
    for label, r in locs:
        print(f"    {label:18} {r.kind:10} cx={r.cx[:16]} fw={r.fw[:10]} maxiter={r.maxiter}")

    estimate(locs, lib, sampler, model, device)
    if args.estimate:
        return

    all_summ = []
    t_wall = time.time()
    for label, ref in locs:
        t_loc = time.time()
        res = run_location(label, ref, lib, sampler, model, device, seed)
        out_loc = OUT_DIR / label
        out_loc.mkdir(parents=True, exist_ok=True)
        rec = write_records(res, out_loc, seed)
        before_after_strip(res, out_loc / "before_after.png")
        contact_18(res, out_loc / "top18_R2.png")
        paired_before_after(out_loc / "records.json", out_loc / "before_after_paired.png", lib)

        lt = rec["lift"]
        print(f"\n=== {label} ({ref.kind})  [{time.time()-t_loc:.0f}s] ===")
        print(f"  gen-0 palette spread: eff-distinct {res['spread']['effective_distinct']}/{N_GEN0}  "
              f"min-pair {res['spread']['min_pairwise']:.3f}  mean-nn {res['spread']['mean_nn']:.3f}  "
              f"(dedup eps {qs.DEDUP_EPS})")
        print(f"  score-lift per round (beam-best | mean | median of 18 lineage-bests):")
        for row in lt:
            print(f"    R={row['R']}   beam {row['beam_best']:+.3f}   "
                  f"mean {row['mean']:+.3f}   median {row['median']:+.3f}")
        d10, d21 = lt[1]["mean"] - lt[0]["mean"], lt[2]["mean"] - lt[1]["mean"]
        mv = rec["movement"]
        print(f"  lift mean dR0->R1 {d10:+.3f}  dR1->R2 {d21:+.3f}  "
              f"(R2 beats R1: {'yes' if d21 > 1e-4 else 'flat'})")
        print(f"  render-space movement gen-0->R2: median dE {mv['median_de']:.2f} "
              f"(improved-only {mv['median_de_improved']:.2f}, {mv['n_improved']}/18 improved) "
              f"-- large dE + lift = visible; ~0 dE + lift = noise-climb")
        print(f"  sheets: {out_loc / 'before_after.png'}")
        print(f"          {out_loc / 'top18_R2.png'}")
        all_summ.append(rec)

    (OUT_DIR / "summary.json").write_text(json.dumps(
        {"seed": seed, "n_locations": len(locs), "wall_seconds": time.time() - t_wall,
         "locations": [{"label": r["label"], "lift": r["lift"], "gen0_spread": r["gen0_spread"],
                        "movement": r["movement"]} for r in all_summ]},
        indent=2))
    print(f"\n[done] {len(locs)} locations -> {OUT_DIR}  ({(time.time()-t_wall)/60:.1f} min)")
    print(f"[sampler] R is Matt's call: eyeball before_after.png (visible improvement -> R=2; "
          f"numeric-only lift -> R=0/1).")


if __name__ == "__main__":
    main()
