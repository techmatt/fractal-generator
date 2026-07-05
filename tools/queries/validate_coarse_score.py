"""Ranking-parity gate for the coarse beam-scoring recolor (prompt-recolor-at-score-res).

Beam scoring recolors the ss2 field (2048x1152) purely to feed the ~224 pref-v2 scorer;
`colormap.render_candidate_coarse` colors a pre-downsampled coarse field at the scorer's
input geometry instead. This is a SCORING-ONLY speedup — the gate is not pixel parity
(the coarse image is a throwaway) but *decision* parity: do full-res vs coarse coloring
pick the same beam survivors / winners under the EXISTING v2?

Per location (a few, spanning families) we compare, with the same v2:
  * gen-0 (identical 60 configs both ways): top-18 survivor overlap (Jaccard) +
    Spearman rank correlation over all 60 scores.
  * full beam both ways (coarse_score False vs True): final winner palette agreement +
    Spearman over the R2 lineage-best scores on the shared palettes.
Also times per-candidate recolor full vs coarse and projects the beam/bootstrap speedup.

    uv run python tools/queries/validate_coarse_score.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path("out/coarse_gate")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "wallpaper"))

import sample_location as SL     # noqa: E402
import query_sampler as qs       # noqa: E402
import query_batch_gen as P      # noqa: E402
import assemble_queries as aq    # noqa: E402
import build_bootstrap as BB     # noqa: E402

cm = qs.cm


# --- rank stats (no scipy) -------------------------------------------------
def _ranks(a):
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort()
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def jaccard(sa, sb):
    sa, sb = set(sa), set(sb)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def best_by_pal(all_candidates):
    """palette -> its best (max) score across every evaluated candidate — the exact
    'representative per palette' strata_sample selects on."""
    m = {}
    for c in all_candidates:
        p = c["palette"]
        if p not in m or c["score"] > m[p]:
            m[p] = float(c["score"])
    return m


# --- gen-0 config reconstruction (mirrors run_location exactly) ------------
def gen0_configs(ref, sampler, seed):
    import hashlib
    stem = aq._field_key(ref)
    rng = np.random.default_rng(int(hashlib.sha1(f"{stem}|{seed}".encode()).hexdigest()[:16], 16))
    pal_names, _, _ = SL.gen0_palettes(sampler, SL.N_GEN0)
    return [qs.sample_candidate(ref, rng, sampler, palette=p, canonical=False) for p in pal_names]


def select_locations(seed, per_family=1):
    """One (or more) source location per family, spanning mandelbrot / multibrot* /
    phoenix / julia:* — drawn from the gather ledgers via build_bootstrap's selector."""
    want = ["mandelbrot", "multibrot4", "phoenix", "julia:mandelbrot", "julia:multibrot3"]
    sources, _ = BB.select_sources(seed, per_class=6)
    by_cls = {}
    for spec, role, row in sources:
        by_cls.setdefault(spec[0], []).append((spec, row))
    out = []
    for cls in want:
        for spec, row in by_cls.get(cls, [])[:per_family]:
            out.append((cls, BB.to_location(spec, row)))
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    seed = 7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    model, epoch = SL.load_v2(device)
    print(f"[validate] v2 epoch {epoch} on {device.type}  coarse grid "
          f"{cm.SCORE_COARSE_W}x{cm.SCORE_COARSE_H}  scorer input 224x224")

    locs = select_locations(seed)
    print(f"[validate] {len(locs)} locations: {[c for c, _ in locs]}\n")

    SCRATCH.mkdir(parents=True, exist_ok=True)
    cand_dump = {}
    t_full_all, t_coarse_all, n_recolor = 0.0, 0.0, 0
    rows = []
    for cls, loc in locs:
        ref = qs.loc_mod.to_location_ref(loc)
        fld, _ = aq.ensure_field(ref)
        prep = cm.stretch_field(fld)
        coarse = cm.coarse_field(prep)

        # --- gen-0 both ways (identical configs) ---
        cfgs = gen0_configs(ref, sampler, seed)

        t0 = time.time()
        imgs_full = [cm.render_candidate(fld, c, lib, prep=prep) for c in cfgs]
        t_full = time.time() - t0
        t0 = time.time()
        imgs_coarse = [cm.render_candidate_coarse(coarse, c, lib) for c in cfgs]
        t_coarse = time.time() - t0
        t_full_all += t_full; t_coarse_all += t_coarse; n_recolor += len(cfgs)

        s_full = np.asarray(P.score_frames(model, imgs_full, device))
        s_coarse = np.asarray(P.score_frames(model, imgs_coarse, device))

        top_full = np.argsort(-s_full)[:SL.TOP_KEEP]
        top_coarse = np.argsort(-s_coarse)[:SL.TOP_KEEP]
        jac = jaccard(top_full.tolist(), top_coarse.tolist())
        rho = spearman(s_full, s_coarse)

        # --- full beam both ways -> winner + R2 ranking + STRATA-BUCKET parity ---
        res_full = SL.run_location(f"{cls}_full", loc, lib, sampler, model, device,
                                   seed, retain_all=True, coarse_score=False)
        res_coarse = SL.run_location(f"{cls}_coarse", loc, lib, sampler, model, device,
                                     seed, retain_all=True, coarse_score=True)
        win_full = max(res_full["lineages"], key=lambda l: l["best_score"])["palette"]
        win_coarse = max(res_coarse["lineages"], key=lambda l: l["best_score"])["palette"]

        # Persist the beam candidate scores (best-per-palette, both ways) so the strata
        # gate can be re-tuned offline WITHOUT another 20-min beam rerun.
        full_best = best_by_pal(res_full["all_candidates"])    # palette -> full-scorer best
        coarse_best = best_by_pal(res_coarse["all_candidates"])
        cand_dump[cls] = {"full_best": full_best, "coarse_best": coarse_best}

        # --- the actual bootstrap consumer: strata_sample over all_candidates ---
        rng = np.random.default_rng(seed)   # strata_sample's rng is vestigial (deterministic)
        picks_full = BB.strata_sample([dict(c) for c in res_full["all_candidates"]], rng)
        picks_coarse = BB.strata_sample([dict(c) for c in res_coarse["all_candidates"]], rng)
        pf_band = {c["palette"]: c["stratum"] for c in picks_full}
        pc_band = {c["palette"]: c["stratum"] for c in picks_coarse}

        pick_jac = jaccard(pf_band.keys(), pc_band.keys())      # overlap of the 8 chosen palettes
        common = set(pf_band) & set(pc_band)
        band_agree = float(np.mean([pf_band[p] == pc_band[p] for p in common])) if common else float("nan")

        # Quality-range recovery: grade EACH path's picks by the TRUSTED full scorer, and
        # ask what fraction of the location's full-scorer quality range they span. This is
        # the bootstrap's real requirement (a broad low->high spread), independent of which
        # exact palette fills each band.
        full_lo, full_hi = min(full_best.values()), max(full_best.values())
        rng_span = max(full_hi - full_lo, 1e-9)
        def range_cov(pick_pals):
            fs = [full_best[p] for p in pick_pals if p in full_best]
            return (max(fs) - min(fs)) / rng_span if fs else 0.0
        cov_full = range_cov(pf_band.keys())
        cov_coarse = range_cov(pc_band.keys())

        rows.append({"cls": cls, "gen0_jac": jac, "gen0_rho": rho,
                     "win_agree": win_full == win_coarse,
                     "pick_jac": pick_jac, "band_agree": band_agree,
                     "cov_full": cov_full, "cov_coarse": cov_coarse})
        print(f"[{cls:18}] gen-0 Spearman {rho:.3f}  | STRATA picks Jaccard {pick_jac:.2f}  "
              f"band-agree {band_agree:.2f}  | quality-range coverage full {cov_full:.2f} "
              f"vs coarse {cov_coarse:.2f}  (winner {'AGREE' if win_full==win_coarse else 'SHIFT'})")

    (SCRATCH / "coarse_gate_candidates.json").write_text(json.dumps(cand_dump, indent=1))

    # --- timing / projection ---
    per_full = t_full_all / n_recolor
    per_coarse = t_coarse_all / n_recolor
    per_loc_recolors = SL.N_GEN0 + SL.R_MAX * SL.TOP_KEEP * SL.K_VARIANTS   # worst case
    n_locs_run = 63
    print("\n" + "=" * 74)
    print("STRATA-BUCKET PARITY SUMMARY  (the bootstrap's real consumer)")
    print("=" * 74)
    mean_rho = np.mean([r["gen0_rho"] for r in rows])
    mean_pick = np.mean([r["pick_jac"] for r in rows])
    mean_band = np.nanmean([r["band_agree"] for r in rows])
    mean_cov_full = np.mean([r["cov_full"] for r in rows])
    mean_cov_coarse = np.mean([r["cov_coarse"] for r in rows])
    agree = sum(r["win_agree"] for r in rows)
    print(f"strata picks Jaccard:      mean {mean_pick:.3f}  (per-loc {[round(r['pick_jac'],2) for r in rows]})")
    print(f"same-band agreement:       mean {mean_band:.3f}  (per-loc {[round(r['band_agree'],2) for r in rows]})")
    print(f"quality-range coverage:    full {mean_cov_full:.3f}  vs  coarse {mean_cov_coarse:.3f}  "
          f"(coarse per-loc {[round(r['cov_coarse'],2) for r in rows]})")
    print(f"  -> coarse picks graded by the TRUSTED full scorer still span "
          f"{mean_cov_coarse*100:.0f}% of the location's full quality range")
    print(f"gen-0 Spearman(60):        mean {mean_rho:.3f}")
    print(f"beam winner agreement: {agree}/{len(rows)} (single argmax — the strictest, "
          f"least bootstrap-relevant view)")
    print("\nTIMING (per scoring recolor):")
    print(f"  full ss2 : {per_full*1000:.1f} ms")
    print(f"  coarse   : {per_coarse*1000:.1f} ms   ({per_full/per_coarse:.1f}x faster)")
    print(f"\nPROJECTION (<= {per_loc_recolors} recolors/loc, {n_locs_run} locs):")
    print(f"  full recolor wall   : {per_loc_recolors*n_locs_run*per_full/60:.1f} min")
    print(f"  coarse recolor wall : {per_loc_recolors*n_locs_run*per_coarse/60:.1f} min")


if __name__ == "__main__":
    main()
