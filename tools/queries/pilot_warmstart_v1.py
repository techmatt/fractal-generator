"""PILOT the v1-seeded labeling batch generator (warmstart_v1_pilot).

Small cross-type sample of the type-dependent, v1-pre-concentrated candidate
pipeline. Extends the coldstart candidate-gen front end (query_sampler +
assemble_queries.ensure_field/candidate_record/contact_sheet + the
regenerate_coldstart_v2 render-space FPS primitives) — the ONLY new machinery is
the v1-scoring insertion and the per-type flow. Ranges/palette-pool/locations are
kept in-distribution with coldstart_v2 (same rev/gamma ranges, same 777-palette
pool; NEW corpus locations, disjoint from coldstart_v2's 188).

For each location N_POOL=48 candidates are drawn, scored where v1 has purchase,
and thinned to 6 by render-space CIEDE2000 farthest-point selection:

  palette : vary palette @ anchor params. Score 48 with v1 (strong axis), keep
            top TOP_KEEP=18, FP-select 6 by render-space dE.
  param   : v1 pre-selects the palette (top-1 of a 48-palette anchor draw), then
            48 param-variations of THAT palette (coldstart ranges + gamma guard),
            FP-select 6 by render-space dE. NO v1 filtering of the param axis.
  joint   : vary palette x param. Score 48 (moderate axis), keep top 18, FP-6.

Cost discipline: the ss2 field is dumped ONCE per location (cached in out/fields)
and every candidate is a cheap Python recolor of that cached field. Only the final
6 per query are persisted as batch images.

This is a PILOT: 6 queries (2 palette / 2 param / 2 joint), NEW distinct
locations, seed 2. It writes CandidateConfig records + contact sheets + a
diagnostics report to data/queries/warmstart_v1_pilot/ and STOPS. No label-store
wiring, no batch_meta/label-server. The full warmstart_v1 is a separate run.

    uv run python tools/queries/pilot_warmstart_v1.py            # full pilot
    uv run python tools/queries/pilot_warmstart_v1.py --estimate # est + exit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_sampler as qs                       # noqa: E402  (pool, sampler, ranges, draw_param_pool)
import assemble_queries as aq                     # noqa: E402  (ensure_field, candidate_record, query_record, contact_sheet)
import regenerate_coldstart_v2 as rc              # noqa: E402  (thumb_lab, render_space_dmat — validated render-space FPS)
import diversity_diagnostic as dd                 # noqa: E402  (single_linkage_clusters, REF/NEAR-dup constants)
sys.path.insert(0, str(qs.ROOT / "tools" / "palettes"))
import palette_features as pf                     # noqa: E402  (farthest_point_order — reused w/ render-space dmat)

# v1 scorer harness — reuse the training pipeline exactly as surfacing_eval does.
sys.path.insert(0, str(HERE / "scorer"))
import data as SD                                 # noqa: E402  (build_transform train=False)
import train as ST                                # noqa: E402  (build_model)

ROOT = qs.ROOT
V1_DIR = ROOT / "data" / "queries" / "scorer" / "v1"
COLDSTART_V2 = ROOT / "data" / "queries" / "coldstart_v2"
PILOT_DIR = ROOT / "data" / "queries" / "warmstart_v1_pilot"

# --- pilot constants -------------------------------------------------------
N_POOL = 48                                   # candidates drawn per location
TOP_KEEP = 18                                 # v1 top-k kept before FP (palette/joint)
K = qs.CANDIDATES_PER_QUERY                   # final candidates per query (6)
QUERY_PLAN = ("palette", "palette", "param", "param", "joint", "joint")
DEFAULT_SEED = 2

# Collapse-flag regime: the known coldstart_v2 singleton (q001_0163) sits at
# min-dE ~1.79 with effective-distinct@10 == 1. Flag the same regime — a final-6
# whose closest pair is inside the near-dup band, or that collapses to one cluster.
NEAR_DUP_THRESH = dd.NEAR_DUP_THRESH          # 2.0
EFF_THRESH = 10.0                             # dd.REF_THRESHOLDS[-1]


# ===========================================================================
# Location selection — NEW locations disjoint from coldstart_v2.
# ===========================================================================

def loc_key(ref):
    return (ref.kind, ref.cx, ref.cy, ref.fw, ref.c_re, ref.c_im)


def coldstart_v2_locations():
    """The 188 (family,cx,cy,fw,c_re,c_im) location keys used by coldstart_v2."""
    excl = set()
    for rp in sorted((COLDSTART_V2 / "records").glob("q*.json")):
        loc = json.loads(rp.read_text())["location"]
        excl.add((loc["family"], loc["cx"], loc["cy"], loc["fw"],
                  loc.get("c_re"), loc.get("c_im")))
    return excl


def select_new_locations(pool, seed, n):
    """`n` distinct q2+q3 locations, sorted for reproducibility then sampled without
    replacement, excluding every coldstart_v2 location."""
    excl = coldstart_v2_locations()
    cand = [pl for pl in pool.locations if loc_key(pl.ref) not in excl]
    cand.sort(key=lambda pl: tuple("" if v is None else str(v) for v in loc_key(pl.ref)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cand), size=n, replace=False)
    return [cand[int(i)] for i in idx], len(cand), len(excl)


# ===========================================================================
# v1 scorer — load once, score a batch of recolored uint8 frames.
# ===========================================================================

def load_v1(device):
    model = ST.build_model().to(device)
    ck = torch.load(V1_DIR / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("epoch")


_TF = SD.build_transform(train=False)   # deploy transform: 1024x576 -> 224 bicubic squash + normalize


@torch.no_grad()
def score_frames(model, imgs_u8, device, chunk=64):
    """v1 utility scores for a list of (H,W,3) uint8 recolors (higher = better)."""
    tens = [_TF(Image.fromarray(im).convert("RGB")) for im in imgs_u8]
    out = []
    for i in range(0, len(tens), chunk):
        batch = torch.stack(tens[i:i + chunk]).to(device)
        out.extend(model(batch).view(-1).cpu().tolist())
    return out


# ===========================================================================
# Candidate configs + render.
# ===========================================================================

def anchor_config(ref, palette):
    """The one canonical linear spec (gamma=1, no log, no reverse/phase/cycles)."""
    return qs.cm.CandidateConfig(
        palette=palette, location=ref,
        eval_width=qs.EVAL_WIDTH, eval_height=qs.EVAL_HEIGHT,
        reverse=False, log_premap="none", gamma=1.0,
        phase=0.0, n_cycles=1, filter=qs.CANDIDATE_FILTER,
    )


def per_query_rng(qid, seed):
    h = hashlib.sha1(f"{qid}|warmstart_v1_pilot|{seed}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


def recolor_all(fld, cfgs, lib, prep):
    return [qs.cm.render_candidate(fld, c, lib, prep=prep) for c in cfgs]


# ===========================================================================
# Render-space selection + diagnostics helpers.
# ===========================================================================

def fp_select(pool_imgs, indices, k):
    """FP-select k of `indices` (into pool_imgs) by render-space mean-CIEDE2000."""
    labs = [rc.thumb_lab(pool_imgs[i]) for i in indices]
    D = rc.render_space_dmat(labs)
    order = pf.farthest_point_order(list(range(len(indices))), k=k, dmat=D)
    return [indices[i] for i in order]


def set_stats(imgs):
    """Render-space diversity of an image set: min/mean pairwise dE + eff-distinct."""
    labs = [rc.thumb_lab(im) for im in imgs]
    D = rc.render_space_dmat(labs)
    n = len(imgs)
    iu = np.triu_indices(n, 1)
    pm = D[iu]
    return {
        "min_de": float(pm.min()),
        "mean_de": float(pm.mean()),
        "eff2": dd.single_linkage_clusters(D, 2.0),
        "eff5": dd.single_linkage_clusters(D, 5.0),
        "eff10": dd.single_linkage_clusters(D, EFF_THRESH),
    }


def collapse_flag(stats):
    return bool(stats["min_de"] < NEAR_DUP_THRESH or stats["eff10"] == 1)


# ===========================================================================
# Per-type query pipelines. Each returns (final_cfgs, final_imgs, diag).
# ===========================================================================

def run_palette(ref, fld, lib, prep, sampler, model, device, rng):
    names = [n for n, _ in sampler.sample_distinct(rng, N_POOL)]
    cfgs = [anchor_config(ref, n) for n in names]
    imgs = recolor_all(fld, cfgs, lib, prep)
    scores = score_frames(model, imgs, device)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])   # v1 desc
    naive6 = order[:K]
    top18 = order[:TOP_KEEP]
    sel = fp_select(imgs, top18, K)

    final_cfgs = [cfgs[i] for i in sel]
    final_imgs = [imgs[i] for i in sel]
    diag = {
        "n_pool": len(cfgs),
        "v1_kept": TOP_KEEP,
        "v1_score": {"max": max(scores), "median": float(np.median(scores)), "min": min(scores)},
        "naive_top6_stats": set_stats([imgs[i] for i in naive6]),
        "fp6_stats": set_stats(final_imgs),
        "fp_vs_naive_overlap": len(set(sel) & set(naive6)),
        "fp_moved": K - len(set(sel) & set(naive6)),
    }
    return final_cfgs, final_imgs, diag


def run_param(ref, fld, lib, prep, sampler, model, device, rng):
    # 1. v1 pre-selects the palette from a 48-palette anchor draw (top-1).
    names = [n for n, _ in sampler.sample_distinct(rng, N_POOL)]
    pre_cfgs = [anchor_config(ref, n) for n in names]
    pre_imgs = recolor_all(fld, pre_cfgs, lib, prep)
    pre_scores = score_frames(model, pre_imgs, device)
    order = sorted(range(len(pre_scores)), key=lambda i: -pre_scores[i])
    top1 = order[0]
    fixed_palette = names[top1]

    # 2. 48 param-variations of that fixed palette (coldstart ranges + gamma guard).
    pool = qs.draw_param_pool(ref, rng, sampler, palette=fixed_palette, pool_size=N_POOL)
    pool_imgs = recolor_all(fld, pool, lib, prep)

    # 3. FP-select 6 in render space; NO v1 filtering of the param axis.
    sel = fp_select(pool_imgs, list(range(len(pool))), K)
    final_cfgs = [pool[i] for i in sel]
    final_imgs = [pool_imgs[i] for i in sel]

    s2 = pre_scores[order[1]]
    diag = {
        "n_pool": len(pool),
        "pool_short": len(pool) < N_POOL,
        "preselect": {
            "palette": fixed_palette,
            "source": sampler.source_of(fixed_palette),
            "type": sampler.library.palette_type(fixed_palette),
            "v1_top1_score": pre_scores[top1],
            "v1_2nd_score": s2,
            "margin_over_2nd": pre_scores[top1] - s2,
            "margin_over_median": pre_scores[top1] - float(np.median(pre_scores)),
        },
        "fp6_stats": set_stats(final_imgs),
        "gamma_range": [min(c.gamma for c in final_cfgs), max(c.gamma for c in final_cfgs)],
    }
    return final_cfgs, final_imgs, diag


def run_joint(ref, fld, lib, prep, sampler, model, device, rng):
    cfgs = [qs.sample_candidate(ref, rng, sampler, canonical=False) for _ in range(N_POOL)]
    imgs = recolor_all(fld, cfgs, lib, prep)
    scores = score_frames(model, imgs, device)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    naive6 = order[:K]
    top18 = order[:TOP_KEEP]
    sel = fp_select(imgs, top18, K)

    final_cfgs = [cfgs[i] for i in sel]
    final_imgs = [imgs[i] for i in sel]
    diag = {
        "n_pool": len(cfgs),
        "v1_kept": TOP_KEEP,
        "v1_score": {"max": max(scores), "median": float(np.median(scores)), "min": min(scores)},
        "naive_top6_stats": set_stats([imgs[i] for i in naive6]),
        "fp6_stats": set_stats(final_imgs),
        "fp_vs_naive_overlap": len(set(sel) & set(naive6)),
        "fp_moved": K - len(set(sel) & set(naive6)),
    }
    return final_cfgs, final_imgs, diag


RUNNERS = {"palette": run_palette, "param": run_param, "joint": run_joint}


# ===========================================================================
# Driver.
# ===========================================================================

def query_is_done(qid):
    rec_ok = (PILOT_DIR / "records" / f"{qid}.json").exists()
    imgs_ok = all((PILOT_DIR / f"images/{qid}_{k}.png").exists() for k in range(K))
    diag_ok = (PILOT_DIR / "diag" / f"{qid}.json").exists()
    return rec_ok and imgs_ok and diag_ok


def main():
    ap = argparse.ArgumentParser(description="Pilot the v1-seeded labeling batch generator.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--estimate", action="store_true", help="print runtime estimate and exit")
    ap.add_argument("--no-resume", action="store_true", help="regenerate completed queries")
    args = ap.parse_args()

    for sub in ("images", "records", "diag"):
        (PILOT_DIR / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    pool = qs.LocationPool.from_corpus()
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)

    chosen, n_new_avail, n_excl = select_new_locations(pool, args.seed, len(QUERY_PLAN))
    plan = list(zip([f"q{args.seed:03d}_{i:04d}" for i in range(len(QUERY_PLAN))],
                    QUERY_PLAN, chosen))

    print(f"[pilot] {pool.report()}")
    print(f"[pilot] coldstart_v2 excluded locations: {n_excl}; "
          f"new-location pool available: {n_new_avail}; selected: {len(chosen)} (seed {args.seed})")
    for qid, qt, loc in plan:
        r = loc.ref
        print(f"[pilot]   {qid} [{qt:7}] {r.kind:10} cx={r.cx[:14]} fw={r.fw[:10]} "
              f"maxiter={r.maxiter} scores={sorted(loc.scores)}")

    # --- runtime estimate: recolor count per plan + a calibrated recolor/dump time ---
    per_type_recolors = {"palette": N_POOL, "param": 2 * N_POOL, "joint": N_POOL}
    total_recolors = sum(per_type_recolors[qt] for _, qt, _ in plan)
    n_dumps = len({aq._field_key(loc.ref) for _, _, loc in plan})

    # Calibrate on the first location: dump its field (needed anyway; cached after)
    # and time one recolor. Fields are per-location; recolor time is location-agnostic.
    fld0, dump0 = aq.ensure_field(chosen[0].ref)
    prep0 = qs.cm.stretch_field(fld0)
    t0 = time.time()
    _ = qs.cm.render_candidate(fld0, anchor_config(chosen[0].ref, sampler.sample_palette(
        np.random.default_rng(0))[0]), lib, prep=prep0)
    recolor_s = time.time() - t0
    dump_est = dump0 if dump0 > 0 else 20.0     # cache hit -> fall back to a nominal ss2 dump
    est_s = n_dumps * dump_est + total_recolors * recolor_s * 1.25   # +25% thumb/dE/FPS/score/IO
    print(f"\n[pilot] est: {n_dumps} field dumps @~{dump_est:.0f}s + {total_recolors} recolors "
          f"@~{recolor_s*1000:.0f}ms => ~{est_s/60:.1f} min "
          f"(recolors: palette 2x{N_POOL}, param 2x{2*N_POOL}, joint 2x{N_POOL})")
    if args.estimate:
        return

    t_wall = time.time()
    field_cache = {aq._field_key(chosen[0].ref): (fld0, prep0)}
    per_query_diag = []

    for qid, qtype, loc in plan:
        if not args.no_resume and query_is_done(qid):
            per_query_diag.append(json.loads((PILOT_DIR / "diag" / f"{qid}.json").read_text()))
            print(f"[pilot] {qid} [{qtype}] resumed (already complete)")
            continue

        ref = loc.ref
        stem = aq._field_key(ref)
        if stem not in field_cache:
            fld, _ = aq.ensure_field(ref)
            field_cache[stem] = (fld, qs.cm.stretch_field(fld))
        fld, prep = field_cache[stem]

        rng = per_query_rng(qid, args.seed)
        t_q = time.time()
        final_cfgs, final_imgs, diag = RUNNERS[qtype](
            ref, fld, lib, prep, sampler, MODEL, device, rng)

        # persist 6 images + record + contact sheet (coldstart schema, atomically)
        image_rels = []
        for ci, im in enumerate(final_imgs):
            rel = f"images/{qid}_{ci}.png"
            Image.fromarray(im).save(PILOT_DIR / rel)
            image_rels.append(rel)
        rec = aq.query_record(qid, loc, qtype, final_cfgs, sampler, image_rels)
        (PILOT_DIR / "records" / f"{qid}.json").write_text(json.dumps(rec, indent=1))
        aq.contact_sheet(final_imgs, final_cfgs, qid, qtype, PILOT_DIR / f"{qid}.png")

        final_stats = set_stats(final_imgs)
        qdiag = {
            "qid": qid, "query_type": qtype,
            "family": ref.kind, "cx": ref.cx, "cy": ref.cy, "fw": ref.fw,
            "final6": final_stats,
            "collapse_flag": collapse_flag(final_stats),
            "detail": diag,
            "contact_sheet": f"{qid}.png",
        }
        (PILOT_DIR / "diag" / f"{qid}.json").write_text(json.dumps(qdiag, indent=1))
        per_query_diag.append(qdiag)
        print(f"[pilot] {qid} [{qtype}] done in {time.time()-t_q:.1f}s  "
              f"final6 minDE={final_stats['min_de']:.2f} eff@10={final_stats['eff10']}"
              f"{'  *COLLAPSE*' if qdiag['collapse_flag'] else ''}")

    wall = time.time() - t_wall
    write_report(per_query_diag, args.seed, wall)
    print(f"\n[done] pilot -> {PILOT_DIR}   ({wall:.1f}s)")


# v1 is loaded once at startup (see __main__) into this module global so the
# per-type runners receive it as an argument without re-loading it per query.
MODEL = None


def write_report(per_query_diag, seed, wall):
    """Print the diagnostics report and persist pilot_report.json + SUMMARY.md."""
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("\n" + "=" * 78)
    emit("WARMSTART_V1 PILOT — DIAGNOSTICS")
    emit("=" * 78)
    emit(f"seed={seed}  queries={len(per_query_diag)}  N_POOL={N_POOL}  TOP_KEEP={TOP_KEEP}  "
         f"collapse-flag regime: minDE<{NEAR_DUP_THRESH} or eff@10==1")

    # (A) diversity of the final 6, per query type
    emit("\n--- (A) diversity of the final 6 (render-space CIEDE2000) ---")
    emit(f"{'qid':<12}{'type':<9}{'family':<11}{'minDE':>7}{'meanDE':>8}"
         f"{'eff@2':>7}{'eff@5':>7}{'eff@10':>8}  flag")
    for q in per_query_diag:
        f6 = q["final6"]
        emit(f"{q['qid']:<12}{q['query_type']:<9}{q['family']:<11}"
             f"{f6['min_de']:>7.2f}{f6['mean_de']:>8.2f}"
             f"{f6['eff2']:>7}{f6['eff5']:>7}{f6['eff10']:>8}"
             f"  {'COLLAPSE' if q['collapse_flag'] else 'ok'}")

    # (B) param queries — preselected palette + spread
    emit("\n--- (B) param queries: v1-preselected palette + param spread ---")
    for q in [q for q in per_query_diag if q["query_type"] == "param"]:
        p = q["detail"]["preselect"]
        f6 = q["final6"]
        gr = q["detail"]["gamma_range"]
        emit(f"  {q['qid']}: palette='{p['palette']}' ({p['source']}, {p['type']})")
        emit(f"      v1 top-1 score={p['v1_top1_score']:.3f}  "
             f"margin over 2nd={p['margin_over_2nd']:.3f}  over median={p['margin_over_median']:.3f}")
        emit(f"      final-6 spread: minDE={f6['min_de']:.2f} meanDE={f6['mean_de']:.2f} "
             f"eff@10={f6['eff10']}  gamma in [{gr[0]:.2f},{gr[1]:.2f}]"
             f"{'   (POOL SHORT)' if q['detail'].get('pool_short') else ''}")

    # (C) palette/joint — FP de-clumping vs naive v1-top-6
    emit("\n--- (C) palette/joint: FP-6 (from v1 top-18) vs naive v1-top-6 ---")
    emit(f"{'qid':<12}{'type':<9}{'overlap':>8}{'moved':>7}"
         f"{'naive minDE':>13}{'fp minDE':>10}{'naive eff10':>13}{'fp eff10':>10}")
    for q in [q for q in per_query_diag if q["query_type"] in ("palette", "joint")]:
        d = q["detail"]
        n6, f6 = d["naive_top6_stats"], d["fp6_stats"]
        emit(f"{q['qid']:<12}{q['query_type']:<9}{d['fp_vs_naive_overlap']:>8}{d['fp_moved']:>7}"
             f"{n6['min_de']:>13.2f}{f6['min_de']:>10.2f}{n6['eff10']:>13}{f6['eff10']:>10}")

    n_collapse = sum(1 for q in per_query_diag if q["collapse_flag"])
    emit(f"\n[summary] collapse-flagged queries: {n_collapse}/{len(per_query_diag)}")
    emit(f"[summary] contact sheets:")
    for q in per_query_diag:
        emit(f"   {PILOT_DIR / q['contact_sheet']}")

    report = {
        "seed": seed, "n_queries": len(per_query_diag),
        "constants": {"N_POOL": N_POOL, "TOP_KEEP": TOP_KEEP, "K": K,
                      "NEAR_DUP_THRESH": NEAR_DUP_THRESH, "EFF_THRESH": EFF_THRESH},
        "wall_seconds": wall,
        "per_query": per_query_diag,
    }
    (PILOT_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2))
    (PILOT_DIR / "SUMMARY.md").write_text("```\n" + "\n".join(lines) + "\n```\n")


if __name__ == "__main__":
    # Load v1 once before the loop; RUNNERS read the module global MODEL.
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL, _ep = load_v1(_dev)
    print(f"[pilot] loaded v1 model_best.pt (epoch {_ep}) on {_dev.type}")
    main()
