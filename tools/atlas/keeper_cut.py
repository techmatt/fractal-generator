#!/usr/bin/env python
"""Per-family KEEPER cut — a stricter, precision-weighted (F0.5) q3 bar for reporting only.

The discovery bar is `production_seeder.t_good_for` (per-partition, F2 / recall-weighted — it
casts wide so the frontier surfaces candidates). The KEEPER bar is its precision-weighted twin:
the `p_good` cut that maximizes **F0.5** against the human labels, so a "keeper" is a location we
are confident a human would call good. NOTHING gates on it — admission stays at the discovery
`t_good`. Keeper status is a *report-time* filter on the persisted canonical `p_good`:

    keeper(row) := corn_decode(row.p_notbad, row.p_good, keeper_cut_for(partition)) == 3

Derived exactly like the discovery table (`tools/v7/derive_t_good.py`), from the frozen v7 eval
slice `data/classifier/v7/eval_scores_v7.jsonl` (label/​fractal_type/​v7 probs inline, already
label_store-resolved + cross-checked by that script), with two changes: the objective is F0.5
(beta=0.5) instead of F2, and the julia:multibrot* slices use the census (`source ==
"prospect_census"`) per Option A. A partition with < MIN_POS positives is UNCALIBRATED and falls
back to the discovery baseline 0.50, flagged. Prediction uses `corn_decode` (the fixed
`p_notbad>=0.5` gate AND `p_good>=t`), matching how an admitted keeper decodes.

  uv run python tools/atlas/keeper_cut.py            # print table + write data/atlas/keeper_cuts.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

from score_lib import corn_decode                    # noqa: E402
from production_seeder import T_GOOD_BASELINE         # noqa: E402

EVAL = ROOT / "data" / "classifier" / "v7" / "eval_scores_v7.jsonl"
OUT = ROOT / "data" / "atlas" / "keeper_cuts.json"

# fractal_type (Rust kind_str) -> ledger partition key (mirrors derive_t_good.FT2FAM).
FT2FAM = {
    "mandelbrot": "mandelbrot",
    "julia": "julia:mandelbrot",
    "multibrot3": "multibrot3", "multibrot4": "multibrot4", "multibrot5": "multibrot5",
    "julia_multibrot3": "julia:multibrot3",
    "julia_multibrot4": "julia:multibrot4",
    "julia_multibrot5": "julia:multibrot5",
    "phoenix": "phoenix",
}
MIN_POS = 15                                          # sufficiency floor (== discovery derivation)
BETA = 0.5                                            # precision-weighted (keeper) objective
GRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]   # [0.02, 0.98]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_triples(eval_path: Path = EVAL) -> dict:
    """{partition: [(p_notbad, p_good, is_pos)]}. julia:multibrot* -> census-only (Option A)."""
    parts: dict = defaultdict(list)
    for r in read_jsonl(eval_path):
        part = FT2FAM.get(r["fractal_type"])
        if part is None:
            continue
        if part.startswith("julia:multibrot") and r.get("source") != "prospect_census":
            continue
        parts[part].append((r["v7_p_not_bad"], r["v7_p_good"], r["label"] == 3))
    return parts


def confusion(rows, t):
    tp = fp = fn = 0
    for nb, g, pos in rows:
        pred = corn_decode(nb, g, t) == 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return tp, fp, fn


def prf_beta(tp, fp, fn, beta=BETA):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * prec + rec
    f = ((1 + b2) * prec * rec / denom) if denom else 0.0
    return prec, rec, f


def best_t(rows):
    """argmax F0.5 over GRID; tie-break toward HIGHER t (equal F, fewer FPs = the keeper intent)."""
    best = None
    for t in GRID:
        _, _, f = prf_beta(*confusion(rows, t))
        if best is None or f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f)
    return best[0]


def loo_f(rows):
    """Leave-one-out OOF F0.5 (honest generalization estimate; small n)."""
    tp = fp = fn = 0
    for i in range(len(rows)):
        rest = rows[:i] + rows[i + 1:]
        t = best_t(rest)
        nb, g, pos = rows[i]
        pred = corn_decode(nb, g, t) == 3
        if pred and pos:
            tp += 1
        elif pred and not pos:
            fp += 1
        elif (not pred) and pos:
            fn += 1
    return prf_beta(tp, fp, fn)


def derive(eval_path: Path = EVAL) -> dict:
    """{partition: {t, calibrated, n, pos, prec, rec, f, oof_f}}. Uncalibrated => baseline, flagged."""
    parts = load_triples(eval_path)
    out = {}
    for part in sorted(set(list(parts) + list(FT2FAM.values()))):
        rows = parts.get(part, [])
        n = len(rows); pos = sum(1 for _, _, x in rows if x)
        if pos < MIN_POS:
            out[part] = dict(t=T_GOOD_BASELINE, calibrated=False, n=n, pos=pos,
                             prec=None, rec=None, f=None, oof_f=None)
            continue
        t = best_t(rows)
        p_t, r_t, f_t = prf_beta(*confusion(rows, t))
        _, _, oof = loo_f(rows)
        out[part] = dict(t=t, calibrated=True, n=n, pos=pos,
                         prec=round(p_t, 4), rec=round(r_t, 4), f=round(f_t, 4),
                         oof_f=round(oof, 4))
    return out


def load_keeper_cuts(path: Path = OUT) -> dict:
    """Read the persisted table; derive-and-write it if absent. Returns {partition: {...}}."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["cuts"]
    cuts = derive()
    write(cuts, path)
    return cuts


def keeper_cut_for(partition: str, cuts: dict) -> float:
    row = cuts.get(partition)
    return float(row["t"]) if row else T_GOOD_BASELINE


def is_keeper(partition: str, p_notbad: float, p_good: float, cuts: dict) -> bool:
    return corn_decode(p_notbad, p_good, keeper_cut_for(partition, cuts)) == 3


def write(cuts: dict, path: Path = OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(
        objective="F0.5", beta=BETA, min_pos=MIN_POS, baseline=T_GOOD_BASELINE,
        eval=str(EVAL.relative_to(ROOT)).replace("\\", "/"), cuts=cuts,
    ), indent=2), encoding="utf-8")


def main():
    cuts = derive()
    print("=" * 78)
    print("KEEPER cut (F0.5 / precision-weighted) — report-only; nothing gates on it")
    print("=" * 78)
    print(f"{'partition':20s} {'n':>4s} {'pos':>4s} {'t_keep':>7s} {'F0.5':>6s} "
          f"{'oof':>6s} {'P':>5s} {'R':>5s}  status")
    for part in sorted(cuts):
        d = cuts[part]
        if d["calibrated"]:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {d['f']:6.3f} "
                  f"{d['oof_f']:6.3f} {d['prec']:5.2f} {d['rec']:5.2f}  calibrated")
        else:
            print(f"{part:20s} {d['n']:4d} {d['pos']:4d} {d['t']:7.2f} {'--':>6s} "
                  f"{'--':>6s} {'--':>5s} {'--':>5s}  UNCALIBRATED -> baseline {d['t']}")
    write(cuts)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
