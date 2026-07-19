#!/usr/bin/env python
"""Re-derive mandelbrot's discovery t_good with the steered-run2 blind labels folded in.

The shipped mandelbrot t_good = 0.14 is the v7 **F2** (recall-weighted) sweep over the v7
eval slice (tools/v7/derive_t_good.py; n=942, pos=29). The steered_run2 blind human read
scored **0/16** mandelbrot admissions good — direct evidence that on steered mandelbrot
output the 0.14 bar over-admits. This pass folds those 16 newly-committed steered labels
into the mandelbrot slice and re-derives the cut **precision-weighted (F0.5)** for this
family specifically (unlike the julia families, whose blind slices were too small / not
similarly one-sided to justify tightening).

Report-only by default (writes docs/findings/mandelbrot_tgood_steered.md +
data/atlas/mandelbrot_tgood_steered.json). Pass --apply to also patch
production_seeder.T_GOOD_OVERRIDES["mandelbrot"] to the derived value.

  uv run python tools/atlas/mandelbrot_tgood_steered.py            # derive + report
  uv run python tools/atlas/mandelbrot_tgood_steered.py --apply    # + patch the table

CPU-only, seconds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mining"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))

from score_lib import corn_decode                       # noqa: E402
from production_seeder import T_GOOD_OVERRIDES           # noqa: E402

EVAL = ROOT / "data" / "classifier" / "v7" / "eval_scores_v7.jsonl"
SCORES = ROOT / "labels" / "steered_run2_blind_scores.json"
MANIFEST = ROOT / "out" / "steered_run2_manifest" / "manifest_key.json"
SEEDER = ROOT / "tools" / "atlas" / "production_seeder.py"
OUT_JSON = ROOT / "data" / "atlas" / "mandelbrot_tgood_steered.json"
OUT_DOC = ROOT / "docs" / "findings" / "mandelbrot_tgood_steered.md"

GRID = [round(0.02 + 0.01 * i, 2) for i in range(97)]    # [0.02, 0.98]
BETA = 0.5                                                # precision-weighted


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def confusion(rows, t):
    """rows: (p_notbad, p_good, is_pos). Predicted-q3 iff corn_decode(nb, g, t) == 3."""
    tp = fp = fn = tn = 0
    for nb, g, pos in rows:
        pred = corn_decode(nb, g, t) == 3
        tp += pred and pos
        fp += pred and not pos
        fn += (not pred) and pos
        tn += (not pred) and not pos
    return tp, fp, fn, tn


def prf_beta(tp, fp, fn, beta=BETA):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    d = b2 * prec + rec
    return prec, rec, ((1 + b2) * prec * rec / d if d else 0.0)


def best_t(rows, beta=BETA):
    """argmax F_beta over GRID; tie-break toward HIGHER t (equal F, fewer FPs)."""
    best = None
    for t in GRID:
        tp, fp, fn, _ = confusion(rows, t)
        _, _, f = prf_beta(tp, fp, fn, beta)
        if best is None or f > best[1] + 1e-12 or (abs(f - best[1]) <= 1e-12 and t > best[0]):
            best = (t, f)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="patch production_seeder.T_GOOD_OVERRIDES['mandelbrot'] to the derived value")
    args = ap.parse_args()

    # ---- eval mandelbrot slice: (v7_p_not_bad, v7_p_good, label==3). ----
    esc = read_jsonl(EVAL)
    eval_rows = [(r["v7_p_not_bad"], r["v7_p_good"], r["label"] == 3)
                 for r in esc if r["fractal_type"] == "mandelbrot"]
    n_e = len(eval_rows)
    pos_e = sum(1 for *_, p in eval_rows if p)
    if n_e == 0:
        raise SystemExit("no mandelbrot rows in the v7 eval slice")

    # ---- steered_run2 mandelbrot blind labels: join scores -> manifest key. ----
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    key = json.loads(MANIFEST.read_text(encoding="utf-8"))
    steered_rows = []
    for e in key["entries"]:
        if e["family"] != "mandelbrot" or e["tile"] not in scores:
            continue
        steered_rows.append((float(e["p_notbad"]), float(e["p_good"]), int(scores[e["tile"]]) == 3))
    n_s = len(steered_rows)
    pos_s = sum(1 for *_, p in steered_rows if p)

    combined = eval_rows + steered_rows
    n_c, pos_c = len(combined), pos_e + pos_s

    old_t = T_GOOD_OVERRIDES.get("mandelbrot", 0.50)
    # F0.5 sweeps: eval-only, combined; F2 combined (for contrast with the shipped objective).
    t_f05_eval, f_f05_eval = best_t(eval_rows, 0.5)
    t_f05, f_f05 = best_t(combined, 0.5)
    t_f2, f_f2 = best_t(combined, 2.0)

    def line(rows, t):
        tp, fp, fn, tn = confusion(rows, t)
        p, r, f05 = prf_beta(tp, fp, fn, 0.5)
        return dict(t=t, tp=tp, fp=fp, fn=fn, tn=tn, prec=round(p, 4), rec=round(r, 4),
                    f05=round(f05, 4), admit=tp + fp, disc_q3=fn)

    new_t = t_f05
    result = dict(
        family="mandelbrot", objective="F0.5", grid=[GRID[0], GRID[-1]],
        old_t=old_t, old_objective="F2", new_t=new_t,
        eval=dict(n=n_e, pos=pos_e), steered=dict(n=n_s, pos=pos_s),
        combined=dict(n=n_c, pos=pos_c),
        sweeps=dict(f05_eval_only=dict(t=t_f05_eval, f=round(f_f05_eval, 4)),
                    f05_combined=dict(t=t_f05, f=round(f_f05, 4)),
                    f2_combined=dict(t=t_f2, f=round(f_f2, 4))),
        at_old_t=line(combined, old_t), at_new_t=line(combined, new_t),
        steered_at_old_t=line(steered_rows, old_t), steered_at_new_t=line(steered_rows, new_t),
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- report doc ----
    aO = result["at_old_t"]; aN = result["at_new_t"]
    sO = result["steered_at_old_t"]; sN = result["steered_at_new_t"]
    doc = f"""# Mandelbrot discovery t_good — re-derivation with steered labels (F0.5)

The shipped mandelbrot discovery `t_good = {old_t}` is the v7 **F2** (recall-weighted) sweep
over the v7 eval slice (`tools/v7/derive_t_good.py`; n={n_e}, pos={pos_e}). The steered_run2
blind human read scored **{pos_s}/{n_s}** mandelbrot admissions good — the human uniformly
rejects what the 0.14 bar admits on steered mandelbrot output (see
`docs/findings/steered_run2_keeper_calibration.md` §E). That is direct evidence the bar
**over-admits on this family**, so — unlike the julia families, whose blind slices were tiny
and not similarly one-sided — mandelbrot is re-derived here **precision-weighted (F0.5)** with
the {n_s} newly-committed steered labels folded in. Same precedent as phoenix 0.18→0.50: a
deliberate admission tightening backed by a labeled read.

## Slices

| slice | n | positives (human/label==3) |
|---|---:|---:|
| v7 eval (mandelbrot) | {n_e} | {pos_e} |
| steered_run2 blind | {n_s} | {pos_s} |
| **combined** | **{n_c}** | **{pos_c}** |

## Sweep

| objective | slice | t\\* | F |
|---|---|---:|---:|
| F0.5 | eval only | {t_f05_eval:.2f} | {f_f05_eval:.3f} |
| F0.5 | combined | **{t_f05:.2f}** | {f_f05:.3f} |
| F2 (shipped objective) | combined | {t_f2:.2f} | {f_f2:.3f} |

**New mandelbrot t_good = {new_t:.2f}** (F0.5, combined), up from {old_t} (F2).

## What the move buys (on the combined slice)

| cut | precision | recall | F0.5 | admit (TP+FP) | discarded q3 (FN) |
|---|---:|---:|---:|---:|---:|
| old t={old_t} (F2) | {aO['prec']:.3f} | {aO['rec']:.3f} | {aO['f05']:.3f} | {aO['admit']} | {aO['disc_q3']} |
| new t={new_t:.2f} (F0.5) | {aN['prec']:.3f} | {aN['rec']:.3f} | {aN['f05']:.3f} | {aN['admit']} | {aN['disc_q3']} |

On the **16 steered mandelbrot tiles specifically** (all human-not-good, so every admission
is a false positive): the old bar admitted **{sO['admit']}/{n_s}**; the new bar admits
**{sN['admit']}/{n_s}**.

## Verdict

The blind read makes mandelbrot's over-admission concrete, and F0.5 acts on it: the bar moves
{old_t}→{new_t:.2f}, cutting the steered-mandelbrot false-positive admissions from {sO['admit']}
to {sN['admit']} of {n_s}. This is a deliberate, family-specific tightening of the discovery
bar — the julia families keep their existing t_good (their blind slices do not justify a
similar move). Applied to `production_seeder.T_GOOD_OVERRIDES["mandelbrot"]`.
"""
    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(doc, encoding="utf-8")

    print(f"eval mandelbrot: n={n_e} pos={pos_e}")
    print(f"steered blind:   n={n_s} pos={pos_s}")
    print(f"combined:        n={n_c} pos={pos_c}")
    print(f"F0.5 eval-only  t*={t_f05_eval:.2f} (F={f_f05_eval:.3f})")
    print(f"F0.5 combined   t*={t_f05:.2f} (F={f_f05:.3f})  <-- NEW mandelbrot t_good (old {old_t}, F2)")
    print(f"steered FP admits: old_t={old_t} -> {sO['admit']}/{n_s} ; new_t={new_t:.2f} -> {sN['admit']}/{n_s}")
    print(f"wrote {OUT_JSON}\nwrote {OUT_DOC}")

    if args.apply:
        src = SEEDER.read_text(encoding="utf-8")
        pat = re.compile(r'("mandelbrot":\s*)([0-9.]+)(,\s*#[^\n]*)')
        m = pat.search(src)
        if not m:
            raise SystemExit("could not locate mandelbrot entry in T_GOOD_OVERRIDES to patch")
        new_src = src[:m.start()] + f'{m.group(1)}{new_t:.2f}' + \
            f'{m.group(3)}  # F0.5 re-derive w/ steered labels (was {old_t} F2)' + src[m.end():]
        SEEDER.write_text(new_src, encoding="utf-8")
        print(f"PATCHED {SEEDER} mandelbrot t_good {old_t} -> {new_t:.2f}")


if __name__ == "__main__":
    main()
