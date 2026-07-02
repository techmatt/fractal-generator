"""Generate the full `warmstart_v1` labeling batch (200 queries) + wire it for labeling.

Promotes the validated pilot (`query_batch_gen.py`) to the authoritative 200-query
labeling substrate. The per-type pipeline is REUSED VERBATIM from the pilot module
(`run_palette`/`run_param`/`run_joint`, the v1-scoring insertion via the scorer's own
`data.build_transform(train=False)` + `train.build_model` from
`data/queries/scorer/v1/model_best.pt`, the once-per-location ss2 field dump, the
cheap-recolor candidate pools, and the render-space CIEDE2000 farthest-point select) —
this file only supplies the full-run driver: the 200-location selection, the
param-heavy query-type plan, atomic per-query checkpointing/resume, the
coldstart_v2-parity batch artifacts (durable records, 1200 images, 200 contact sheets,
batch_meta.json with the location list), the re-render byte-exact spot-check, and the
population-level diagnostics report.

Pipeline constants (N_POOL=48, TOP_KEEP=18, K=6, GAMMA_MIN_SPACING=0.15, the 777-palette
pool, the rev/gamma ranges) are inherited from the pilot module unchanged.

Full-run parameters: 200 queries, seed 2, split 0.35/0.50/0.15 => 70 palette / 100 param
/ 30 joint (param-heavy to stress the near-collapse watch item), 200 distinct NEW
locations disjoint from coldstart_v2's 188.

    uv run python tools/queries/warmstart_v1.py --estimate   # runtime estimate + exit
    uv run python tools/queries/warmstart_v1.py              # full run (background it)
    uv run python tools/queries/warmstart_v1.py --report-only  # rebuild report from diag/
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
import query_sampler as qs                 # noqa: E402
import assemble_queries as aq              # noqa: E402  (ensure_field, query_record, contact_sheet, candidate_record)
import query_batch_gen as P                # noqa: E402  (the validated pipeline — reused verbatim)

ROOT = qs.ROOT
BATCH_ID = "warmstart_v1"
BATCH_DIR = ROOT / "data" / "queries" / BATCH_ID

# --- full-run parameters ----------------------------------------------------
SEED = 2
N_QUERIES = 200
# split 0.35/0.50/0.15 (palette/param/joint) -> exact integer counts summing to 200.
QUERY_SPLIT = (0.35, 0.50, 0.15)
QUERY_COUNTS = {"palette": 70, "param": 100, "joint": 30}
assert sum(QUERY_COUNTS.values()) == N_QUERIES

# per-type recolor budget (for the runtime estimate): palette scores 48; param scores 48
# for the v1 pre-select THEN recolors a 48-member param pool; joint scores 48.
RECOLORS_PER_TYPE = {"palette": P.N_POOL, "param": 2 * P.N_POOL, "joint": P.N_POOL}


# ===========================================================================
# Query-type plan — exact counts, deterministically interspersed from the seed.
# ===========================================================================

def build_plan(chosen, seed):
    """Pair the 200 distinct locations with query types at the exact split counts.

    The type multiset (70/100/30) is shuffled with a seed-derived RNG (independent of
    the location-selection stream) and zipped onto qids q{seed:03d}_0000..0199. Order is
    cosmetic (the label server reshuffles presentation), but fixed for reproducibility."""
    types = (["palette"] * QUERY_COUNTS["palette"]
             + ["param"] * QUERY_COUNTS["param"]
             + ["joint"] * QUERY_COUNTS["joint"])
    np.random.default_rng(seed).shuffle(types)
    qids = [f"q{seed:03d}_{i:04d}" for i in range(len(chosen))]
    return list(zip(qids, types, chosen))


def per_query_rng(qid, seed):
    """Per-query independent RNG — same derivation as the pilot, warmstart_v1 namespace."""
    h = hashlib.sha1(f"{qid}|{BATCH_ID}|{seed}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


# ===========================================================================
# Per-query persistence + resume.
# ===========================================================================

def query_is_done(qid):
    rec_ok = (BATCH_DIR / "records" / f"{qid}.json").exists()
    imgs_ok = all((BATCH_DIR / f"images/{qid}_{k}.png").exists() for k in range(P.K))
    diag_ok = (BATCH_DIR / "diag" / f"{qid}.json").exists()
    return rec_ok and imgs_ok and diag_ok


def persist_query(qid, qtype, loc, final_cfgs, final_imgs, detail, sampler):
    """Write the 6 images, the durable record, the contact sheet, then the diag file
    LAST as the completion marker (so a crash mid-query leaves query_is_done() False)."""
    ref = loc.ref
    image_rels = []
    for ci, im in enumerate(final_imgs):
        rel = f"images/{qid}_{ci}.png"
        Image.fromarray(im).save(BATCH_DIR / rel)
        image_rels.append(rel)
    rec = aq.query_record(qid, loc, qtype, final_cfgs, sampler, image_rels)
    (BATCH_DIR / "records" / f"{qid}.json").write_text(json.dumps(rec, indent=1))
    aq.contact_sheet(final_imgs, final_cfgs, qid, qtype, BATCH_DIR / f"{qid}.png")

    final_stats = P.set_stats(final_imgs)
    qdiag = {
        "qid": qid, "query_type": qtype,
        "family": ref.kind, "cx": ref.cx, "cy": ref.cy, "fw": ref.fw,
        "final6": final_stats,
        "collapse_flag": P.collapse_flag(final_stats),
        "detail": detail,
        "contact_sheet": f"{qid}.png",
    }
    (BATCH_DIR / "diag" / f"{qid}.json").write_text(json.dumps(qdiag, indent=1))
    return qdiag


# ===========================================================================
# Runtime estimate.
# ===========================================================================

def estimate(plan, chosen, lib, sampler):
    total_recolors = sum(RECOLORS_PER_TYPE[qt] for _, qt, _ in plan)
    # dumps needed = locations whose ss2 field is not already cached in out/fields.
    need_dump = 0
    for _, _, loc in plan:
        stem = aq._field_key(loc.ref)
        if not ((aq.OUT_FIELDS / f"{stem}.bin").exists()
                and (aq.OUT_FIELDS / f"{stem}.json").exists()):
            need_dump += 1

    # calibrate one recolor on the first location (dump it now — needed anyway, cached after).
    fld0, dump0 = aq.ensure_field(chosen[0].ref)
    prep0 = qs.cm.stretch_field(fld0)
    t0 = time.time()
    _ = qs.cm.render_candidate(
        fld0, P.anchor_config(chosen[0].ref, sampler.sample_palette(np.random.default_rng(0))[0]),
        lib, prep=prep0)
    recolor_s = time.time() - t0
    dump_est = dump0 if dump0 > 0 else 20.0     # first loc may be cached -> nominal ss2 dump
    # +25% overhead for thumb/dE/FPS/v1-score/IO (matches the pilot's calibration factor).
    est_s = need_dump * dump_est + total_recolors * recolor_s * 1.25
    cnt = QUERY_COUNTS
    print(f"[warmstart] est: {need_dump} field dumps @~{dump_est:.0f}s + "
          f"{total_recolors} recolors @~{recolor_s*1000:.0f}ms => ~{est_s/3600:.2f} h "
          f"(~{est_s/60:.0f} min)")
    print(f"[warmstart]   recolors: palette {cnt['palette']}x{P.N_POOL} + "
          f"param {cnt['param']}x{2*P.N_POOL} + joint {cnt['joint']}x{P.N_POOL}")
    return {"need_dump": need_dump, "total_recolors": total_recolors,
            "recolor_s": recolor_s, "est_s": est_s}, (fld0, prep0)


# ===========================================================================
# Batch meta.
# ===========================================================================

def write_batch_meta(plan, n_avail, n_excl, results, v1_epoch):
    locations = []
    for qid, qt, loc in plan:
        r = loc.ref
        locations.append({
            "query_id": qid, "query_type": qt, "family": r.kind,
            "cx": r.cx, "cy": r.cy, "fw": r.fw, "maxiter": r.maxiter,
            "c_re": r.c_re, "c_im": r.c_im,
        })
    meta = {
        "batch_id": BATCH_ID,
        "purpose": ("v1-seeded palette-preference labeling batch: for each of 200 NEW "
                    "locations, the v1 scorer pre-concentrates a candidate pool where it "
                    "has purchase, then render-space CIEDE2000 farthest-point-select 6. "
                    "The authoritative substrate a human tiers into good-vs-good ranking "
                    "signal. Promoted verbatim from the warmstart_v1 pilot."),
        "derived_from": ("v1 scorer data/queries/scorer/v1/model_best.pt; NEW corpus "
                         "locations disjoint from coldstart_v2's 188"),
        "invocation": "uv run python tools/queries/warmstart_v1.py",
        "pipeline_module": "tools/queries/query_batch_gen.py (reused verbatim)",
        "n": N_QUERIES,
        "seed": SEED,
        "candidate_ss": qs.CANDIDATE_SS,
        "eval": [qs.EVAL_WIDTH, qs.EVAL_HEIGHT],
        "query_split": list(QUERY_SPLIT),
        "query_type_counts": QUERY_COUNTS,
        "pipeline": {
            "N_POOL": P.N_POOL, "TOP_KEEP": P.TOP_KEEP, "K": P.K,
            "GAMMA_MIN_SPACING": qs.GAMMA_MIN_SPACING,
            "palette_pool": "data/palettes/pool_colormaps.json (777)",
            "selector": "render-space mean-CIEDE2000 farthest_point_order",
            "per_type": {
                "palette": "48 palettes @anchor -> v1 top-18 -> FP-6",
                "param": "v1 top-1 palette of 48-anchor draw -> 48 param-variations -> FP-6 (NO v1 on param axis)",
                "joint": "48 palette x param -> v1 top-18 -> FP-6",
            },
        },
        "coldstart_exclusion": {
            "excluded_batch": "coldstart_v2",
            "n_excluded_locations": n_excl,
            "new_pool_available": n_avail,
        },
        "v1_model": {"path": "data/queries/scorer/v1/model_best.pt", "epoch": v1_epoch},
        "results": results,
        "locations": locations,
    }
    (BATCH_DIR / "batch_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ===========================================================================
# Re-render byte-exact spot-check.
# ===========================================================================

def spotcheck(n_check, lib, seed):
    """Re-render `n_check` random candidates from their stored recipe and diff the PNG."""
    rng = np.random.default_rng(seed)
    all_cands = []
    for rp in sorted((BATCH_DIR / "records").glob("q*.json")):
        rec = json.loads(rp.read_text())
        for ci, cand in enumerate(rec["candidates"]):
            all_cands.append((rec, ci, cand))
    pick = rng.choice(len(all_cands), size=min(n_check, len(all_cands)), replace=False)
    field_cache = {}
    ok = 0
    fails = []
    for i in pick:
        rec, ci, cand = all_cands[int(i)]
        loc = rec["location"]
        ref = qs.cm.LocationRef(
            kind=loc["family"], cx=loc["cx"], cy=loc["cy"], fw=loc["fw"],
            maxiter=int(loc["maxiter"]), c_re=loc.get("c_re"), c_im=loc.get("c_im"))
        stem = aq._field_key(ref)
        if stem not in field_cache:
            fld, _ = aq.ensure_field(ref)          # cache hit -> 0s
            field_cache[stem] = fld
        fld = field_cache[stem]
        cfg = qs.cm.CandidateConfig.from_json(json.dumps(cand["config"]))
        im = qs.cm.render_candidate(fld, cfg, lib)
        saved = np.asarray(Image.open(BATCH_DIR / cand["image"]))
        if np.array_equal(im, saved):
            ok += 1
        else:
            fails.append(f"{rec['query_id']}_{ci}")
    return ok, len(pick), fails


# ===========================================================================
# Population-level diagnostics.
# ===========================================================================

def _quantiles(xs):
    a = np.asarray(xs, dtype=np.float64)
    qs_ = [0, 10, 25, 50, 75, 90, 100]
    return {f"p{q}": float(np.percentile(a, q)) for q in qs_}


def build_report(per_query_diag, wall):
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    by_type = {t: [q for q in per_query_diag if q["query_type"] == t]
               for t in ("palette", "param", "joint")}

    emit("\n" + "=" * 82)
    emit("WARMSTART_V1 — POPULATION DIAGNOSTICS")
    emit("=" * 82)
    emit(f"batch={BATCH_ID}  seed={SEED}  queries={len(per_query_diag)}  "
         f"N_POOL={P.N_POOL}  TOP_KEEP={P.TOP_KEEP}")
    emit(f"split 0.35/0.50/0.15 -> palette={len(by_type['palette'])} "
         f"param={len(by_type['param'])} joint={len(by_type['joint'])}")
    emit(f"collapse-flag regime: minDE<{P.NEAR_DUP_THRESH} OR eff@10==1   "
         f"near-collapse band: minDE<10")

    report = {"batch_id": BATCH_ID, "seed": SEED, "n_queries": len(per_query_diag),
              "wall_seconds": wall, "per_type": {}}

    # (A) min-dE + eff@10 distributions per type, with band/regime counts -------------
    emit("\n--- (A) final-6 min-dE + eff@10 distribution, per query type ---")
    emit(f"{'type':<9}{'n':>4}{'minDE p0':>10}{'p10':>7}{'p25':>7}{'p50':>7}"
         f"{'p75':>7}{'p90':>7}{'p100':>7}{'  <10':>7}{'  coll':>7}")
    for t in ("palette", "param", "joint"):
        qd = by_type[t]
        mins = [q["final6"]["min_de"] for q in qd]
        effs = [q["final6"]["eff10"] for q in qd]
        near = sum(1 for m in mins if m < 10.0)
        coll = sum(1 for q in qd if q["collapse_flag"])
        qm = _quantiles(mins)
        emit(f"{t:<9}{len(qd):>4}{qm['p0']:>10.2f}{qm['p10']:>7.2f}{qm['p25']:>7.2f}"
             f"{qm['p50']:>7.2f}{qm['p75']:>7.2f}{qm['p90']:>7.2f}{qm['p100']:>7.2f}"
             f"{near:>7}{coll:>7}")
        eff_hist = {k: int(sum(1 for e in effs if e == k)) for k in range(1, P.K + 1)}
        report["per_type"][t] = {
            "n": len(qd),
            "min_de_quantiles": qm,
            "near_collapse_band_lt10": near,
            "collapse_regime": coll,
            "eff10_hist": eff_hist,
            "eff10_quantiles": _quantiles(effs),
        }
    emit("\n  eff@10 histogram (count of queries by #effective-distinct @ dE<10):")
    emit(f"  {'type':<9}" + "".join(f"eff={k:<2}" for k in range(1, P.K + 1)))
    for t in ("palette", "param", "joint"):
        h = report["per_type"][t]["eff10_hist"]
        emit(f"  {t:<9}" + "".join(f"{h[k]:<6}" for k in range(1, P.K + 1)))

    # (B) param arm — v1 palette pre-select margin distribution -----------------------
    emit("\n--- (B) param arm: v1 palette-preselect top-1 margin-over-2nd distribution ---")
    margins = [q["detail"]["preselect"]["margin_over_2nd"] for q in by_type["param"]]
    med_margins = [q["detail"]["preselect"]["margin_over_median"] for q in by_type["param"]]
    thin = sum(1 for m in margins if m < 3.0)
    mq = _quantiles(margins)
    emit(f"  margin-over-2nd:   p0={mq['p0']:.3f} p25={mq['p25']:.3f} p50={mq['p50']:.3f} "
         f"p75={mq['p75']:.3f} p100={mq['p100']:.3f}")
    emit(f"  thin margins (<3): {thin}/{len(margins)} "
         f"({100.0*thin/max(1,len(margins)):.1f}%)  [accepted — good-enough palette, not best]")
    emit(f"  margin-over-median p50={_quantiles(med_margins)['p50']:.3f}")
    report["param_preselect"] = {
        "margin_over_2nd_quantiles": mq,
        "thin_lt3": thin, "n": len(margins),
        "margin_over_median_quantiles": _quantiles(med_margins),
    }

    # (C) FP lift — naive-v1-top-6 vs top-18->FP-6, per type (palette/joint only) ------
    emit("\n--- (C) FP lift: naive v1-top-6 vs FP-6 (from v1 top-18), per type ---")
    emit("    (param has no v1-naive baseline — its FP-6 selects from the raw 48-pool)")
    emit(f"  {'type':<9}{'naive minDE p50':>17}{'fp minDE p50':>14}"
         f"{'naive<10':>10}{'fp<10':>8}{'moved p50':>11}")
    for t in ("palette", "joint"):
        qd = by_type[t]
        naive_min = [q["detail"]["naive_top6_stats"]["min_de"] for q in qd]
        fp_min = [q["detail"]["fp6_stats"]["min_de"] for q in qd]
        moved = [q["detail"]["fp_moved"] for q in qd]
        n_naive_lt = sum(1 for m in naive_min if m < 10.0)
        n_fp_lt = sum(1 for m in fp_min if m < 10.0)
        emit(f"  {t:<9}{np.median(naive_min):>17.2f}{np.median(fp_min):>14.2f}"
             f"{n_naive_lt:>10}{n_fp_lt:>8}{np.median(moved):>11.1f}")
        report["per_type"][t]["fp_lift"] = {
            "naive_min_de_quantiles": _quantiles(naive_min),
            "fp_min_de_quantiles": _quantiles(fp_min),
            "naive_lt10": n_naive_lt, "fp_lt10": n_fp_lt,
            "moved_median": float(np.median(moved)),
        }

    # summary line ---------------------------------------------------------------------
    n_coll = sum(1 for q in per_query_diag if q["collapse_flag"])
    n_near = sum(1 for q in per_query_diag if q["final6"]["min_de"] < 10.0)
    emit(f"\n[summary] collapse-flagged: {n_coll}/{len(per_query_diag)}   "
         f"near-collapse (minDE<10): {n_near}/{len(per_query_diag)}")
    report["totals"] = {"collapse_flagged": n_coll, "near_collapse_lt10": n_near,
                        "n_queries": len(per_query_diag)}

    (BATCH_DIR / "warmstart_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (BATCH_DIR / "SUMMARY.md").write_text("```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    return report


# ===========================================================================
# Driver.
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate the full warmstart_v1 labeling batch.")
    ap.add_argument("--estimate", action="store_true", help="print runtime estimate and exit")
    ap.add_argument("--no-resume", action="store_true", help="regenerate completed queries")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the diagnostics report from existing diag/ (no generation)")
    ap.add_argument("--spotcheck", type=int, default=20,
                    help="re-render N random candidates from recipe and diff (0 to skip)")
    args = ap.parse_args()

    # Windows console/redirected stdout defaults to cp1252, which can't encode some
    # report glyphs (e.g. em-dash); force UTF-8 so prints + files never crash/mojibake.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    for sub in ("images", "records", "diag"):
        (BATCH_DIR / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    pool = qs.LocationPool.from_corpus()
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)

    chosen, n_avail, n_excl = P.select_new_locations(pool, SEED, N_QUERIES)
    plan = build_plan(chosen, SEED)

    print(f"[warmstart] {pool.report()}")
    print(f"[warmstart] coldstart_v2 excluded: {n_excl}; new-location pool available: "
          f"{n_avail}; selected: {len(chosen)} distinct (seed {SEED})")
    from collections import Counter
    print(f"[warmstart] selected by family: {dict(Counter(l.ref.kind for l in chosen))}")
    print(f"[warmstart] query-type counts: {QUERY_COUNTS} (split {QUERY_SPLIT})")

    if args.report_only:
        diag = [json.loads(p.read_text()) for p in sorted((BATCH_DIR / "diag").glob("q*.json"))]
        if not diag:
            sys.exit("[warmstart] no diag/ files to report on")
        build_report(diag, wall=0.0)
        return

    est, first_field = estimate(plan, chosen, lib, sampler)
    if args.estimate:
        return

    model, v1_epoch = P.load_v1(device)
    print(f"[warmstart] loaded v1 model_best.pt (epoch {v1_epoch}) on {device.type}")

    t_wall = time.time()
    field_cache = {aq._field_key(chosen[0].ref): first_field}   # reuse the calibration dump
    per_query_diag = []
    n_gen = 0

    for i, (qid, qtype, loc) in enumerate(plan):
        if not args.no_resume and query_is_done(qid):
            per_query_diag.append(json.loads((BATCH_DIR / "diag" / f"{qid}.json").read_text()))
            continue

        ref = loc.ref
        stem = aq._field_key(ref)
        if stem not in field_cache:
            fld, _ = aq.ensure_field(ref)
            field_cache[stem] = (fld, qs.cm.stretch_field(fld))
        fld, prep = field_cache[stem]
        # free the field cache after use — locations are distinct, no cross-query reuse.
        # (keep the entry key so a resumed rerun of the same qid still recomputes cleanly)

        rng = per_query_rng(qid, SEED)
        t_q = time.time()
        final_cfgs, final_imgs, detail = P.RUNNERS[qtype](
            ref, fld, lib, prep, sampler, model, device, rng)
        qdiag = persist_query(qid, qtype, loc, final_cfgs, final_imgs, detail, sampler)
        per_query_diag.append(qdiag)
        n_gen += 1

        if stem in field_cache and stem != aq._field_key(chosen[0].ref):
            del field_cache[stem]                    # cap memory: one field held at a time

        el = time.time() - t_wall
        rate = el / max(1, n_gen)
        remaining = sum(1 for q, _, _ in plan[i + 1:] if not query_is_done(q))
        print(f"[warmstart] {i+1}/{N_QUERIES} {qid} [{qtype:7}] done {time.time()-t_q:.1f}s  "
              f"minDE={qdiag['final6']['min_de']:.2f} eff@10={qdiag['final6']['eff10']}"
              f"{'  *COLLAPSE*' if qdiag['collapse_flag'] else ''}  "
              f"eta ~{rate*remaining/60:.0f}min", flush=True)

    wall = time.time() - t_wall
    print(f"\n[warmstart] generation complete: {n_gen} generated, "
          f"{N_QUERIES - n_gen} resumed  ({wall/60:.1f} min)")

    # spot-check re-render byte-exactness
    sc = None
    if args.spotcheck > 0:
        ok, tot, fails = spotcheck(args.spotcheck, lib, SEED)
        sc = {"ok": ok, "checked": tot, "fails": fails}
        print(f"[warmstart] byte-exact spot-check: {ok}/{tot} re-render-from-recipe matches"
              f"{'  FAILS: ' + ','.join(fails) if fails else ''}")

    results = {
        "queries": len(per_query_diag),
        "candidates": len(per_query_diag) * P.K,
        "generated_this_run": n_gen,
        "wall_seconds": wall,
        "recolor_estimate": est,
        "spotcheck": sc,
    }
    write_batch_meta(plan, n_avail, n_excl, results, v1_epoch)
    build_report(per_query_diag, wall)

    print(f"\n[done] {BATCH_ID} -> {BATCH_DIR}")
    print(f"[label] launch:  uv run python tools/queries/launch_query_label_server.py "
          f"--batch {BATCH_ID}")


if __name__ == "__main__":
    main()
