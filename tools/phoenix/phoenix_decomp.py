#!/usr/bin/env python
r"""Phoenix Phase B — the between/within-seed variance decomposition (spec §5.1, step 0).

Reads a grid run's `descent_records.jsonl` (one row per (seed, repeat) descent) and splits the
descent-quality variance of each decomposition variable into a **between-seed** component (does
seed choice matter — is fertility a structural, cacheable property?) and a **within-seed**
component (descent stochasticity at a fixed seed). Two variables per spec:
  * distinct_looks_within — distinct morph-looks a descent mints (cos-0.974, within the descent)
  * max_p_good            — the descent's best canonical p_good

Method: one-way **random-effects ANOVA** (unbalanced-safe), seed = grouping factor. The variance
components are the classic ANOVA estimators; ICC = σ²_between / (σ²_between + σ²_within) is the
share of variance that is between-seed. CIs are **nonparametric cluster bootstrap** — resample
SEEDS (whole descent-clusters) with replacement, recompute, percentile interval — so the CI
respects the nesting. Between-seed variance can estimate negative (MSB < MSW); we clamp the
point estimate to 0 and report the raw value + the fraction of bootstrap draws that went
negative (a signal that within dominates).

**PROVISIONAL** — every metric here rests on v7 scoring a phoenix population it has zero training
coverage on. This decomposition is a design signal (build the surrogate vs propose-fresh, spec
§5.1); the human labels adjudicate. The readout must not state a surrogate go/no-go as final.

  uv run python tools/phoenix/phoenix_decomp.py --run data/discovery/phoenix_grid/grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIABLES = ("distinct_looks_within", "max_p_good")


def _groups(records: list[dict], var: str) -> list[np.ndarray]:
    """Per-seed arrays of `var`, dropping seeds with <2 repeats (no within signal)."""
    by_seed: dict[int, list[float]] = {}
    for r in records:
        by_seed.setdefault(int(r["seed_idx"]), []).append(float(r[var]))
    return [np.array(v, float) for v in by_seed.values() if len(v) >= 2]


def anova_components(groups: list[np.ndarray]) -> dict:
    """One-way random-effects variance components (unbalanced). Returns σ²_between (raw + clamped),
    σ²_within, ICC, MSB, MSW, and the design constant n0."""
    a = len(groups)
    ns = np.array([len(g) for g in groups], float)
    N = float(ns.sum())
    grand = float(np.concatenate(groups).mean())
    means = np.array([g.mean() for g in groups])
    ss_between = float((ns * (means - grand) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df_b, df_w = a - 1, N - a
    msb = ss_between / df_b if df_b > 0 else 0.0
    msw = ss_within / df_w if df_w > 0 else 0.0
    n0 = (N - (ns ** 2).sum() / N) / df_b if df_b > 0 else float(ns.mean())
    var_b_raw = (msb - msw) / n0 if n0 > 0 else 0.0
    var_b = max(0.0, var_b_raw)
    var_w = msw
    denom = var_b + var_w
    icc = var_b / denom if denom > 0 else 0.0
    return {"a_seeds": a, "N": N, "n0": round(n0, 3), "grand_mean": grand,
            "MSB": msb, "MSW": msw, "var_between_raw": var_b_raw,
            "var_between": var_b, "var_within": var_w, "icc": icc}


def cluster_bootstrap(groups: list[np.ndarray], n_boot: int, seed: int) -> dict:
    """Resample whole seed-clusters with replacement; percentile CIs on ICC and the variance
    components. Reports the fraction of draws with a negative raw between-variance."""
    rng = np.random.default_rng(seed)
    a = len(groups)
    iccs, vbs, vws, neg = [], [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, a, size=a)
        bs = [groups[i] for i in idx]
        c = anova_components(bs)
        iccs.append(c["icc"]); vbs.append(c["var_between"]); vws.append(c["var_within"])
        if c["var_between_raw"] < 0:
            neg += 1

    def ci(x):
        x = np.array(x, float)
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"n_boot": n_boot, "icc_ci95": ci(iccs), "var_between_ci95": ci(vbs),
            "var_within_ci95": ci(vws), "frac_between_negative": neg / max(1, n_boot)}


def decompose(records: list[dict], n_boot: int, seed: int) -> dict:
    out = {}
    for var in VARIABLES:
        groups = _groups(records, var)
        if len(groups) < 2:
            out[var] = {"error": "too few multi-repeat seeds for a decomposition"}
            continue
        comp = anova_components(groups)
        boot = cluster_bootstrap(groups, n_boot, seed)
        # verdict heuristic (PROVISIONAL): between dominates iff ICC>0.5 and its CI floor stays >0.
        between_dominates = comp["icc"] > 0.5 and boot["icc_ci95"][0] > 0.0
        out[var] = {**comp, **boot,
                    "verdict_provisional": ("between-seed dominates" if between_dominates
                                            else "within-seed dominates / inconclusive")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="grid run dir (holds descent_records.jsonl)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON here (default: <run>/decomposition.json)")
    args = ap.parse_args(argv)
    run = Path(args.run)
    recs = [json.loads(l) for l in open(run / "descent_records.jsonl", encoding="utf-8") if l.strip()]
    res = {"run": str(run), "n_records": len(recs),
           "n_seeds": len({r["seed_idx"] for r in recs}),
           "decomposition": decompose(recs, args.n_boot, args.seed)}
    out = Path(args.out) if args.out else run / "decomposition.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["decomposition"], indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
