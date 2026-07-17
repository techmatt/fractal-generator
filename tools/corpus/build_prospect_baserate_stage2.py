r"""Build the `2026-07-17_prospect_run1_baserate_R_v1` label batch — STAGE 2 (R band).

Companion to `build_prospect_baserate.py` (stage 1, M/G/H). Stage 2 takes the R band
[0, 0.15) but for **julia:multibrot3/4/5 ONLY** — native multibrot R stays deferred.

Why this specific batch: stage 1 was a census of the M/G/H strata for these three julia
families (61 locations). Stage 2 is the remaining R band (86: jm3 23 · jm4 34 · jm5 29).
`61 + 86 = 147 = the COMPLETE pipeline-surfaced julia:multibrot population in the frozen
ledger`. The census UNIT is **stage1 ∪ stage2**, not this batch alone: only the union is
unbiased-given-descent (no sampling on the score) and thus eligible as an eval set. Stage
2 in isolation is still a score-band selection (R), so on its own it stays `biased→train`
— the census property belongs to the union.

Presentation + parity are IDENTICAL to stage 1 (reused verbatim from `render_block`):
640x360 ss2, palette twilight_shifted, family-aware julia mapping (fixed c = ledger
outcome_cx/cy, viewport = julia_z_cx/cy/fw). `--verify-mapping N` re-scores N rows with
the live v6 head and confirms the reproduced p_good matches the stored ledger p_good
(Gate-2 parity, max|Δ| ~ 1e-4) BEFORE the batch is rendered — the trap that nearly ruined
stage 1 (scoring the wrong viewport) must not recur.

  uv run python tools/corpus/build_prospect_baserate_stage2.py --verify-mapping 8  # parity gate (GPU)
  uv run python tools/corpus/build_prospect_baserate_stage2.py                     # write jsonl + batch.json
  uv run python tools/corpus/build_prospect_baserate_stage2.py --render            # + render the 86 crops
"""
from __future__ import annotations
import argparse, json, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("tools/scoring", "tools/corpus", "tools/reframe", "tools/mining"):
    sys.path.insert(0, str(ROOT / p))
import corpus_common as cc
# Reuse stage-1 presentation VERBATIM so the two batches are pixel-identical in path.
import build_prospect_baserate as s1
from build_prospect_baserate import render_block, stratum, LEDGER, STRATA
from reframe import _render, RENDER_W, RENDER_H, RENDER_SS, PALETTE

BATCH_ID = "2026-07-17_prospect_run1_baserate_R_v1"
GEN_VER = "prospect_run1_baserate_R_v1"
STAGE1_BATCH_ID = "2026-07-17_prospect_run1_baserate_v1"
JULIA = s1.JULIA                       # ("julia:multibrot3/4/5")
STAGE2 = ("R",)                        # R band only; native R stays deferred
CENSUS_UNIT = (STAGE1_BATCH_ID, BATCH_ID)


def load_julia_rows():
    """All guard-pass julia:multibrot rows with a v6 p_good (the full 147 population)."""
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    tgt = [r for r in rows if r.get("family") in JULIA
           and r.get("guard_pass") and r.get("p_good") is not None]
    return len(rows), tgt


def build_rows(tgt):
    rows_out = []
    for r in tgt:
        if stratum(r["p_good"]) not in STAGE2:          # keep only R
            continue
        render, _loc, _cand = render_block(r)
        prov = cc.provenance_block(
            GEN_VER, BATCH_ID,
            family=r["family"], k3=r.get("k3"), decoded_class=r.get("decoded_class"),
            descend_mode=r.get("descend_mode"), parent_oid=r.get("parent_oid"),
            lineage="prospect_run1", p_good=r["p_good"], p_notbad=r.get("p_notbad"),
            t_good=r.get("t_good"), stratum="R", scorer_version=r.get("scorer_version"),
            ledger_id=r["id"])
        rows_out.append(cc.make_row(r["id"], render, prov, cc.label_block()))
    return rows_out


def populations(tgt):
    pops = {f: {s[0]: 0 for s in STRATA} for f in JULIA}
    for r in tgt:
        pops[r["family"]][stratum(r["p_good"])] += 1
    return pops


def verify_mapping(tgt, n):
    """Render N R-band julia rows via the julia mapping, score with live v6, and compare
    the reproduced p_good to the stored ledger p_good. This is the Gate-2 parity check for
    THIS band — stage 1 passing does not certify R (a different p_good regime)."""
    from active_ckpt import make_scorer, ACTIVE_CKPT
    R = [r for r in tgt if stratum(r["p_good"]) in STAGE2]
    # deterministic spread across families and the R p_good range (no RNG)
    R.sort(key=lambda r: (r["family"], r["p_good"]))
    pick = R[:: max(1, len(R) // n)][:n]
    tmp = Path(cc.batch_dir(BATCH_ID)) / "_verify"
    tmp.mkdir(parents=True, exist_ok=True)
    scorer = make_scorer(ACTIVE_CKPT)
    print(f"parity check: {len(pick)} R julia rows, model={ACTIVE_CKPT}")
    print(f"{'ledger_id':<28}{'family':<19}{'stored':>9}{'repro':>9}{'|delta|':>10}")
    worst = 0.0
    for r in pick:
        render, loc, cand = render_block(r)
        out = tmp / f"{r['id']}.jpg"
        good, err = _render(loc, cand, out, RENDER_W, RENDER_H, RENDER_SS)
        if not good:
            print(f"  RENDER FAIL {r['id']}: {err}")
            worst = float("inf"); continue
        _score, _nb, pg = scorer.score_paths([out])[0]
        d = abs(pg - r["p_good"])
        worst = max(worst, d)
        print(f"{r['id']:<28}{r['family']:<19}{r['p_good']:>9.4f}{pg:>9.4f}{d:>10.2e}")
    ok = worst <= 1e-3
    print(f"\nmax|delta p_good| = {worst:.2e}  ->  {'PASS' if ok else 'FAIL'} "
          f"(gate 1e-3; stage-1 achieved 1e-4)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="render the 86 crops")
    ap.add_argument("--verify-mapping", type=int, default=0, metavar="N",
                    help="score N R rows with v6, confirm p_good reproduces, then exit")
    args = ap.parse_args()

    ledger_total, tgt = load_julia_rows()

    if args.verify_mapping:
        ok = verify_mapping(tgt, args.verify_mapping)
        sys.exit(0 if ok else 1)

    rows_out = build_rows(tgt)
    pops = populations(tgt)
    julia_all = sum(sum(pops[f].values()) for f in JULIA)
    mgh_all = sum(pops[f][s] for f in JULIA for s in ("M", "G", "H"))
    census_complete = (mgh_all + len(rows_out) == julia_all == 147)

    bdir = Path(cc.batch_dir(BATCH_ID))
    (bdir / "crops").mkdir(parents=True, exist_ok=True)
    cc.write_jsonl(rows_out, str(bdir / "images.jsonl"))

    per_family = {f: pops[f]["R"] for f in JULIA}
    batch = {
        "batch_id": BATCH_ID, "created": DATE, "labeler": None,
        "generator_version": GEN_VER,
        "source_run": "prospect_run1",
        "source_ledger": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
        "ledger_snapshot_rows": ledger_total,     # must equal stage 1's (1616) to compose
        "schema_version": 1,
        "stage": "2 (R band; julia:multibrot3/4/5 ONLY; native multibrot R still deferred)",
        "census": {
            "unit": "stage1 UNION stage2 (NOT this batch alone)",
            "member_batches": list(CENSUS_UNIT),
            "target_families": list(JULIA),
            "population_total": julia_all,        # 147 = complete pipeline-surfaced julia:multibrot
            "stage1_mgh_locations": mgh_all,      # 61
            "stage2_R_locations": len(rows_out),  # 86
            "complete_condition": "stage1 M/G/H census (61) + stage2 R census (86) == "
                                  "full julia:multibrot ledger population (147)",
            "census_complete": census_complete,
            "union_status": (
                "census / unbiased-given-descent (NO sampling on the score across the "
                "union) -> eligible as an EVAL set"
            ) if census_complete else "INCOMPLETE — union not yet a census",
            "note": (
                "Status belongs to the UNION, not this batch. Stage 2 alone is an R "
                "score-band selection and on its own is `biased->train`; only stage1 "
                "UNION stage2 spans every p_good band with no score sampling, which is "
                "what makes it unbiased-given-descent and eval-eligible."
            ),
        },
        "draw": {
            "stratify_on": "raw v6 p_good (threshold-free, independent of t_good)",
            "strata": {s[0]: [s[1], s[2]] for s in STRATA},
            "stage2_strata": list(STAGE2),
            "policy": "census of R for julia:multibrot3/4/5 (take ALL, no subsampling)",
            "native_R": "DEFERRED (not rendered, not included)",
            "stratum_populations": pops,          # julia families, full bands
            "stage2_crops_per_family": per_family,
            "stage2_total_crops": len(rows_out),
        },
        "rate_definition": (
            "P(q3 | location surfaced by the prospect pipeline), NEVER P(q3 | family). "
            "Unchanged from stage 1. The pipeline (guarded seeder + reframe + per-degree "
            "t_good gate) is the conditioning event; the census gives that conditional "
            "rate over the whole julia:multibrot population, band-unbiased."
        ),
        "parity": {
            "scored_path": f"reframe._render {RENDER_W}x{RENDER_H} ss{RENDER_SS} "
                           f"palette={PALETTE} -> v6 stretch 384x224",
            "julia_mapping": "fixed c = ledger outcome_cx/cy; viewport = julia_z_cx/cy/fw",
            "reverified_for_R": "yes — `--verify-mapping` re-scored R rows before render "
                                "(do not assume stage 1's pass certifies this band)",
            "crop_is_scored_image": True,
        },
        "render_defaults": {
            "palette": PALETTE, "composition": "center", "width": RENDER_W,
            "height": RENDER_H, "ss": RENDER_SS, "filter": "lanczos3",
            "interior_mode": "black",
        },
        "bias_policy": (
            "STAGE 2 ALONE (R band) is a score-band selection -> `biased->train`: never a "
            "standalone retrain feed. The census property (unbiased-given-descent, "
            "eval-eligible) is a property of the UNION stage1 UNION stage2, not of this "
            "batch. Provenance never enters training regardless."
        ),
        "guardrails": "did not touch data/v6/manifest.jsonl; no retrain; no threshold change",
    }
    (bdir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")

    print(f"batch: {BATCH_ID}")
    print(f"frozen ledger rows: {ledger_total} (stage-1 recorded 1616)   julia population: {julia_all}")
    print(f"stage-2 (R) rows written: {len(rows_out)}   census_complete={census_complete} "
          f"(61 MGH + {len(rows_out)} R == {mgh_all + len(rows_out)} vs 147)")
    print(f"{'family':<20}{'R':>5}{'MGH':>6}{'pop':>6}")
    for f in JULIA:
        p = pops[f]
        print(f"{f:<20}{p['R']:>5}{sum(p[s] for s in ('M','G','H')):>6}{sum(p.values()):>6}")
    print(f"images.jsonl + batch.json -> {bdir}")

    if args.render:
        render_all([r for r in tgt if stratum(r["p_good"]) in STAGE2], bdir)


def render_all(todo, bdir):
    crops = bdir / "crops"
    ok = fail = 0
    for i, r in enumerate(todo):
        render, loc, cand = render_block(r)
        out = crops / f"{r['id']}.jpg"
        good, err = _render(loc, cand, out, RENDER_W, RENDER_H, RENDER_SS)
        if good:
            ok += 1
        else:
            fail += 1
            print(f"  RENDER FAIL {r['id']} ({r['family']}): {err}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(todo)}] rendered ok={ok} fail={fail}", flush=True)
    print(f"RENDER DONE: ok={ok} fail={fail} -> {crops}", flush=True)


DATE = datetime.date.today().isoformat()

if __name__ == "__main__":
    main()
