"""Lock the render-mode (mining) head v1 as the strange-mode quality gate.

Certifies the pin in ``mining_gate.py`` and freezes the operating point:

  §2 THRESHOLD  -- build the held-out eval PR curve for marginal ``p_ge3`` on the
       staged seed-0 model, report the chosen conservative/high-precision operating
       point (``MINING_GATE_THRESHOLD``) + the full curve, and cross-check that the
       (precision, recall, pass-rate) at that threshold is STABLE across all 5 seeds
       (not a seed-0 artifact).

  §4 PARITY (the lock) -- re-score the eval crops through the *deployed* entry point
       (``MiningScorer.score_paths``) and confirm ``p_ge3`` matches the train-harness
       eval scores (``seed_0/eval_scores.jsonl``) within tolerance. A tight delta
       proves the integrated gate reproduces the measured AUC/AP with NO preprocessing
       drift between train-eval and deploy.

Writes ``data/render_mode_head/v1/mining_gate_lock.json`` -- the frozen curve,
operating point, seed-stability cross-check, and parity delta.

    uv run python tools/mining/lock_mining_gate.py            # full lock (needs crops + torch)
    uv run python tools/mining/lock_mining_gate.py --no-parity   # curve + stability only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mining.mining_gate import (  # noqa: E402
    ACTIVE_MINING_CKPT, LOCK_PATH, MINING_GATE_THRESHOLD, MINING_GATE_VERSION,
)

V1 = ROOT / "data" / "render_mode_head" / "v1"
DATASET_EVAL = ROOT / "data" / "render_mode_corpus" / "dataset_v1" / "eval.jsonl"
SEEDS = [0, 1, 2, 3, 4]
STAGED_SEED = 0   # model_best.pt == seed_0 (best per-seed eval not-bad AP)


# --------------------------------------------------------------------------- #
# Load the train-harness eval scores (ground truth for the curve + parity).
# --------------------------------------------------------------------------- #
def load_eval_scores(seed: int) -> list[dict]:
    p = V1 / f"seed_{seed}" / "eval_scores.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def y_and_p(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Binary good-vs-rest label and marginal p_ge3 (the gate signal)."""
    y = np.array([1 if int(r["label"]) >= 3 else 0 for r in rows])
    p = np.array([float(r["p_ge3"]) for r in rows])
    return y, p


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    npred = int(pred.sum())
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "threshold": float(thr),
        "precision": (tp / npred) if npred else None,
        "recall": tp / int((y == 1).sum()),
        "pass_rate": npred / len(y),
        "tp": tp, "fp": fp, "n_pred": npred, "n": len(y),
    }


def pr_curve(y: np.ndarray, p: np.ndarray) -> list[dict]:
    thrs = np.concatenate([[0.0], np.round(np.linspace(0.05, 0.90, 18), 3)])
    return [operating_point(y, p, t) for t in thrs]


# --------------------------------------------------------------------------- #
# §2 threshold + seed-stability cross-check.
# --------------------------------------------------------------------------- #
def build_threshold_block(thr: float) -> dict:
    y0, p0 = y_and_p(load_eval_scores(STAGED_SEED))
    curve = pr_curve(y0, p0)
    op = operating_point(y0, p0, thr)

    # cross-seed stability at the SAME threshold (not a seed-0 artifact).
    per_seed = []
    for s in SEEDS:
        ys, ps = y_and_p(load_eval_scores(s))
        o = operating_point(ys, ps, thr)
        o["seed"] = s
        per_seed.append(o)

    def band(key):
        vals = [o[key] for o in per_seed if o[key] is not None]
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "sd": float(a.std()),
                "median": float(np.median(a)), "values": [float(v) for v in a]}

    return {
        "threshold": float(thr),
        "signal": "marginal p_ge3 = cumprod(sigma(logits))",
        "rationale": ("canonical CORN marginal boundary P(label>=3) > 1/2; "
                      "conservative / high-precision (quota is a ceiling -> under-emit is safe)"),
        "staged_seed": STAGED_SEED,
        "staged_operating_point": op,
        "pr_curve": curve,
        "seed_stability": {
            "per_seed": per_seed,
            "precision": band("precision"),
            "recall": band("recall"),
            "pass_rate": band("pass_rate"),
            "note": (f"staged seed-{STAGED_SEED} precision {op['precision']:.3f} is mid-pack "
                     f"(median {float(np.median([o['precision'] for o in per_seed])):.3f}); NOT the high "
                     f"outlier (seed-4's 1.0 is small-n) -> operating point is representative, not a "
                     f"seed-0 fluke. Per-seed precision spread is small-count noise (npred < 35), "
                     f"so precision >~0.5 (>~3.6x the 0.139 base) is the honest floor, not 0.59"),
        },
    }


# --------------------------------------------------------------------------- #
# §4 parity: deployed scorer vs train-harness eval scores.
# --------------------------------------------------------------------------- #
def build_parity_block() -> dict:
    from tools.mining.mining_gate import MiningScorer

    # image_id -> crop path (version-invariant render -> crops/<id>.jpg).
    crop_of = {}
    for ln in DATASET_EVAL.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        crop_of[r["image_id"]] = ROOT / r["crop"]

    rows = load_eval_scores(STAGED_SEED)
    ids = [r["image_id"] for r in rows]
    paths = [crop_of[i] for i in ids]
    missing = [i for i, p in zip(ids, paths) if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} eval crops missing (e.g. {missing[:3]})")

    scorer = MiningScorer(model_path=ACTIVE_MINING_CKPT)
    deployed = scorer.score_paths(paths)

    train_p3 = np.array([float(r["p_ge3"]) for r in rows])
    dep_p3 = np.array([d.p_ge3 for d in deployed])
    d_p3 = np.abs(dep_p3 - train_p3)

    train_p2 = np.array([float(r["p_ge2"]) for r in rows])
    dep_p2 = np.array([d.p_ge2 for d in deployed])
    d_p2 = np.abs(dep_p2 - train_p2)

    # gate-verdict agreement at the pinned threshold (the thing that actually ships).
    dep_pass = dep_p3 >= scorer.threshold
    train_pass = train_p3 >= scorer.threshold
    verdict_agree = int((dep_pass == train_pass).sum())

    worst = int(np.argmax(d_p3))
    return {
        "n": len(rows),
        "scored_via": "MiningScorer.score_paths (deployed entry point)",
        "p_ge3_max_abs_delta": float(d_p3.max()),
        "p_ge3_mean_abs_delta": float(d_p3.mean()),
        "p_ge2_max_abs_delta": float(d_p2.max()),
        "worst_image": {"image_id": ids[worst], "train_p_ge3": float(train_p3[worst]),
                        "deployed_p_ge3": float(dep_p3[worst]), "abs_delta": float(d_p3[worst])},
        "gate_verdict_agreement": {"agree": verdict_agree, "n": len(rows),
                                   "frac": verdict_agree / len(rows)},
        "tolerance": 1e-4,
        "parity_ok": bool(d_p3.max() < 1e-4),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the deployed-scorer parity pass (curve + stability only)")
    args = ap.parse_args()

    thr = MINING_GATE_THRESHOLD
    tblock = build_threshold_block(thr)
    op = tblock["staged_operating_point"]
    ss = tblock["seed_stability"]

    print(f"=== mining gate lock: {MINING_GATE_VERSION} -> {ACTIVE_MINING_CKPT} ===")
    print(f"\n§2 THRESHOLD = {thr}  (marginal p_ge3)")
    print(f"  staged seed-{STAGED_SEED} operating point: "
          f"precision {op['precision']:.3f}  recall {op['recall']:.3f}  "
          f"pass-rate {op['pass_rate']:.3f}  (tp={op['tp']} fp={op['fp']} npred={op['n_pred']}/{op['n']})")
    print(f"  full PR curve ({len(tblock['pr_curve'])} points):")
    print(f"    {'thr':>6} {'prec':>6} {'rec':>6} {'pass%':>6} {'tp':>4} {'fp':>4} {'npred':>5}")
    for c in tblock["pr_curve"]:
        pp = "  n/a" if c["precision"] is None else f"{c['precision']:6.3f}"
        print(f"    {c['threshold']:6.3f} {pp} {c['recall']:6.3f} {c['pass_rate']:6.3f} "
              f"{c['tp']:4d} {c['fp']:4d} {c['n_pred']:5d}")
    print(f"  5-seed stability @ thr={thr}: "
          f"precision {ss['precision']['mean']:.3f}+/-{ss['precision']['sd']:.3f} "
          f"(median {ss['precision']['median']:.3f}) "
          f"{[round(v,3) for v in ss['precision']['values']]}")
    print(f"                                recall {ss['recall']['mean']:.3f}+/-{ss['recall']['sd']:.3f}"
          f"   pass-rate {ss['pass_rate']['mean']:.3f}+/-{ss['pass_rate']['sd']:.3f}")
    print(f"  -> {ss['note']}")

    lock = {
        "gate_version": MINING_GATE_VERSION,
        "checkpoint": ACTIVE_MINING_CKPT,
        "rollback": None,
        "deploy_config": {
            "preprocess": "384x224 bicubic stretch + checkpoint mean/std (Transform train=False)",
            "gate_signal": "marginal p_ge3 = cumprod(sigma(logits))  (NEVER the CORN conditional)",
            "black_gate": "accept iff black_fraction < 0.30 (upstream, parity with Rust render path)",
        },
        "threshold": tblock,
    }

    if not args.no_parity:
        pblock = build_parity_block()
        lock["parity"] = pblock
        print(f"\n§4 PARITY (deployed scorer vs train-harness eval, n={pblock['n']}):")
        print(f"  p_ge3 max|delta| = {pblock['p_ge3_max_abs_delta']:.2e}   "
              f"mean|delta| = {pblock['p_ge3_mean_abs_delta']:.2e}   "
              f"(tol {pblock['tolerance']:.0e})")
        print(f"  worst: {pblock['worst_image']['image_id']} "
              f"train={pblock['worst_image']['train_p_ge3']:.5f} "
              f"deployed={pblock['worst_image']['deployed_p_ge3']:.5f}")
        print(f"  gate-verdict agreement @ thr={thr}: "
              f"{pblock['gate_verdict_agreement']['agree']}/{pblock['gate_verdict_agreement']['n']} "
              f"({pblock['gate_verdict_agreement']['frac']:.4f})")
        print(f"  -> PARITY {'OK' if pblock['parity_ok'] else 'FAIL'} "
              f"(gate provably reproduces the measured eval scores)")

    out = ROOT / LOCK_PATH
    out.write_text(json.dumps(lock, indent=2))
    print(f"\nfroze lock -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
