r"""Build the `2026-07-17_prospect_run1_baserate_v1` label batch — STAGE 1 (M/G/H).

A stratified-on-`p_good` draw from the FROZEN prospect_run1 discovery ledger, built
to (a) validate the v6 machine `decoded_class`/`t_good` against human labels and
(b) estimate the per-family q3 rate. Stage 1 takes ALL of the M/G/H bands
(p_good >= 0.15) for the six target families; the R band [0,0.15) is deferred (its
exact populations are recorded here so a later R draw reweights correctly).

Parity: crops are rendered by the EXACT path the v6 scorer used — reframe._render at
640x360 ss2, palette twilight_shifted, family-aware (Julia: fixed c = ledger
outcome_cx/cy, viewport = julia_z_cx/cy/fw). Verified reproduces stored p_good to
max|Δ|=1e-4 (Gate 2). So the labeled crop IS the image v6 scored.

  uv run python tools/corpus/build_prospect_baserate.py            # write jsonl + batch.json
  uv run python tools/corpus/build_prospect_baserate.py --render   # + render the crops
"""
from __future__ import annotations
import argparse, json, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("tools/scoring", "tools/corpus", "tools/reframe"):
    sys.path.insert(0, str(ROOT / p))
import corpus_common as cc
import reframe
from reframe import Location, _render, RENDER_W, RENDER_H, RENDER_SS, PALETTE, auto_maxiter

LEDGER = ROOT / "data/discovery/fresh_runs/prospect_run1/outcome_ledger.jsonl"
BATCH_ID = "2026-07-17_prospect_run1_baserate_v1"
GEN_VER = "prospect_run1_baserate_v1"
NATIVE = ("multibrot3", "multibrot4", "multibrot5")
JULIA = ("julia:multibrot3", "julia:multibrot4", "julia:multibrot5")
TARGET = NATIVE + JULIA
STRATA = (("R", 0.0, 0.15), ("M", 0.15, 0.40), ("G", 0.40, 0.70), ("H", 0.70, 1.01))
STAGE1 = ("M", "G", "H")   # R deferred


def stratum(p):
    for name, lo, hi in STRATA:
        if lo <= p < hi:
            return name
    return "H"


def load_target_rows():
    rows = [json.loads(l) for l in open(LEDGER, encoding="utf-8") if l.strip()]
    total = len(rows)
    tgt = [r for r in rows if r.get("family") in TARGET
           and r.get("guard_pass") and r.get("p_good") is not None]
    return rows, total, tgt


def render_block(r):
    fam = r["family"]
    if fam in NATIVE:
        vfw = float(r["outcome_fw"])
        block = {"cx": cc.hp_str(r["outcome_cx"]), "cy": cc.hp_str(r["outcome_cy"]),
                 "fw": cc.hp_str(vfw)}
        extra = {}
        loc = Location(family=fam, c_re=None, c_im=None,
                       cx=block["cx"], cy=block["cy"], fw=vfw, family_params={})
    else:                                   # julia:multibrotN
        rf = "julia_" + fam.split(":")[1]   # -> julia_multibrot4
        vfw = float(r["julia_z_fw"])
        block = {"cx": cc.hp_str(r["julia_z_cx"]), "cy": cc.hp_str(r["julia_z_cy"]),
                 "fw": cc.hp_str(vfw)}
        extra = {"fractal_type": rf, "c_re": cc.hp_str(r["outcome_cx"]),
                 "c_im": cc.hp_str(r["outcome_cy"])}
        loc = Location(family=rf, c_re=extra["c_re"], c_im=extra["c_im"],
                       cx=block["cx"], cy=block["cy"], fw=vfw, family_params={})
    mit = int(auto_maxiter(vfw))
    render = {**block, "maxiter": mit, "palette": PALETTE, "composition": "center",
              "width": RENDER_W, "height": RENDER_H, "ss": RENDER_SS,
              "filter": "lanczos3", "interior_mode": "black", **extra}
    cand = {"cx": block["cx"], "cy": block["cy"], "fw": vfw, "maxiter": mit}
    return render, loc, cand


def build_rows(tgt):
    rows_out, deferred_R = [], []
    for r in tgt:
        st = stratum(r["p_good"])
        if st not in STAGE1:
            deferred_R.append(r)
            continue
        render, _loc, _cand = render_block(r)
        prov = cc.provenance_block(
            GEN_VER, BATCH_ID,
            family=r["family"], k3=r.get("k3"), decoded_class=r.get("decoded_class"),
            descend_mode=r.get("descend_mode"), parent_oid=r.get("parent_oid"),
            lineage="prospect_run1", p_good=r["p_good"], p_notbad=r.get("p_notbad"),
            t_good=r.get("t_good"), stratum=st, scorer_version=r.get("scorer_version"),
            ledger_id=r["id"])
        rows_out.append(cc.make_row(r["id"], render, prov, cc.label_block()))
    return rows_out, deferred_R


def populations(tgt):
    pops = {f: {s[0]: 0 for s in STRATA} for f in TARGET}
    for r in tgt:
        pops[r["family"]][stratum(r["p_good"])] += 1
    return pops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    all_rows, ledger_total, tgt = load_target_rows()
    rows_out, deferred_R = build_rows(tgt)
    pops = populations(tgt)
    bdir = Path(cc.batch_dir(BATCH_ID))
    (bdir / "crops").mkdir(parents=True, exist_ok=True)

    cc.write_jsonl(rows_out, str(bdir / "images.jsonl"))

    per_family_stage1 = {f: sum(pops[f][s] for s in STAGE1) for f in TARGET}
    batch = {
        "batch_id": BATCH_ID, "created": DATE, "labeler": None,
        "generator_version": GEN_VER,
        "source_run": "prospect_run1",
        "source_ledger": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
        "ledger_snapshot_rows": ledger_total,   # frozen ledger row count drawn from
        "schema_version": 1,
        "stage": "1 (M/G/H only; R band deferred)",
        "draw": {
            "stratify_on": "raw v6 p_good (threshold-free, independent of t_good)",
            "strata": {s[0]: [s[1], s[2]] for s in STRATA},
            "stage1_strata": list(STAGE1),
            "policy": "census of M/G/H (take ALL, no subsampling)",
            "target_families": list(TARGET),
            "stratum_populations": pops,      # FULL populations incl R (for reweighting)
            "stage1_crops_per_family": per_family_stage1,
            "stage1_total_crops": len(rows_out),
            "deferred_R_total": len(deferred_R),
        },
        "rate_definition": (
            "This draw is unbiased GIVEN DESCENT: it estimates "
            "P(q3 | location surfaced by the prospect pipeline), NOT P(q3 | family). "
            "The pipeline (guarded seeder + reframe + per-degree t_good gate) is the "
            "conditioning event; the reweighted whole gives that conditional rate, the "
            "enriched top gives yield. Do NOT quote as a bare per-family base rate."
        ),
        "parity": {
            "scored_path": f"reframe._render {RENDER_W}x{RENDER_H} ss{RENDER_SS} "
                           f"palette={PALETTE} -> v6 stretch 384x224",
            "gate2_max_abs_dpgood": 1e-4,
            "julia_mapping": "fixed c = ledger outcome_cx/cy; viewport = julia_z_cx/cy/fw",
            "crop_is_scored_image": True,
        },
        "render_defaults": {
            "palette": PALETTE, "composition": "center", "width": RENDER_W,
            "height": RENDER_H, "ss": RENDER_SS, "filter": "lanczos3",
            "interior_mode": "black",
        },
        "bias_policy": (
            "STRATIFIED ON SCORE -> biased->train. Analysis/validation ONLY. "
            "Never a retrain feed. Provenance never enters training regardless."
        ),
        "guardrails": "did not touch data/v6/manifest.jsonl; no retrain; no threshold change",
    }
    (bdir / "batch.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")

    print(f"batch: {BATCH_ID}")
    print(f"frozen ledger rows: {ledger_total}   target guard-pass rows: {len(tgt)}")
    print(f"stage-1 (M/G/H) rows written: {len(rows_out)}   R deferred: {len(deferred_R)}")
    print(f"{'family':<20}{'M':>5}{'G':>5}{'H':>5}{'MGH':>6}{'R':>6}{'pop':>6}")
    for f in TARGET:
        p = pops[f]
        mgh = sum(p[s] for s in STAGE1)
        print(f"{f:<20}{p['M']:>5}{p['G']:>5}{p['H']:>5}{mgh:>6}{p['R']:>6}{sum(p.values()):>6}")
    print(f"images.jsonl + batch.json -> {bdir}")

    if args.render:
        render_all(tgt, bdir)


def render_all(tgt, bdir):
    crops = bdir / "crops"
    todo = [r for r in tgt if stratum(r["p_good"]) in STAGE1]
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


# module-constant date (Date.now unavailable in some contexts; stamp explicitly)
DATE = datetime.date.today().isoformat()

if __name__ == "__main__":
    main()
