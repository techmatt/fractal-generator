"""v6 discovery-gate threshold sweep — set the q3 (rank-3) operating point on the
labeled eval split, before the v6 harvest. No retrain, no render.

Decode rule: frame decodes q3 iff  p_notbad >= T_nb (0.5, fixed)  AND  p_good >= T_gd.
T_gd is the knob; sweep it. Reports overall + per-family recall/precision/volume.
"""
import json
from collections import Counter, defaultdict

PATH = "data/classifier/v6/eval_scores_v6.jsonl"
T_NB = 0.5
BASELINE_TGD = 0.50

rows = [json.loads(l) for l in open(PATH)]
for r in rows:
    r["pnb"] = r["v6_p_not_bad"]
    r["pg"] = r["v6_p_good"]
    r["is3"] = r["label"] == 3
    r["fam"] = r["fractal_type"]

N = len(rows)
n_true3 = sum(r["is3"] for r in rows)

# --- Non-monotone check: p_good >= T_gd while p_notbad < 0.5 ---
# Worst case is the smallest T_gd we sweep (0.20): most permissive on p_good.
print("=" * 78)
print("MONOTONICITY CHECK (frames with p_good >= T_gd but p_notbad < T_nb)")
print("=" * 78)
for tg in (0.20, 0.30, 0.40, 0.50):
    bad = [r for r in rows if r["pg"] >= tg and r["pnb"] < T_NB]
    print(f"  T_gd={tg:.2f}: {len(bad)} frames (p_good>=T_gd, p_notbad<0.5)")
print("  -> these are excluded from decode by the AND rule; report only.\n")

# --- Per-family good counts (power) ---
fam_goods = Counter(r["fam"] for r in rows if r["is3"])
fam_tot = Counter(r["fam"] for r in rows)
print("=" * 78)
print("POWER: eval good (label==3) counts per family")
print("=" * 78)
print(f"  {'family':<20} {'n_good':>6} {'n_total':>8}")
DEG2 = {"mandelbrot", "julia"}
for fam in sorted(fam_tot, key=lambda f: -fam_goods[f]):
    tag = "  <-- POWERED" if fam in DEG2 else "  (unpowered, ~noise)"
    print(f"  {fam:<20} {fam_goods[fam]:>6} {fam_tot[fam]:>8}{tag}")
print(f"  {'TOTAL':<20} {sum(fam_goods.values()):>6} {sum(fam_tot.values()):>8}")
print()


def stats(subset, tg):
    """recall, precision, decoded volume, n_true3 in subset."""
    t3 = [r for r in subset if r["is3"]]
    decoded = [r for r in subset if r["pnb"] >= T_NB and r["pg"] >= tg]
    tp = sum(r["is3"] for r in decoded)
    recall = tp / len(t3) if t3 else float("nan")
    prec = tp / len(decoded) if decoded else float("nan")
    return recall, prec, len(decoded), tp, len(t3)


TGRID = [round(0.20 + 0.02 * i, 2) for i in range(16)]  # 0.20..0.50


def sweep_table(subset, title):
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'T_gd':>5} {'recall':>7} {'prec':>7} {'dec_vol':>8} {'tp':>4} {'true3':>6}")
    for tg in TGRID:
        rec, prec, vol, tp, t3 = stats(subset, tg)
        mark = "  <-- 0.50 baseline" if abs(tg - BASELINE_TGD) < 1e-9 else ""
        rs = f"{rec:.3f}" if rec == rec else "  n/a"
        ps = f"{prec:.3f}" if prec == prec else "  n/a"
        print(f"  {tg:>5.2f} {rs:>7} {ps:>7} {vol:>8} {tp:>4} {t3:>6}{mark}")
    print()


sweep_table(rows, "OVERALL SWEEP")

deg2 = [r for r in rows if r["fam"] in DEG2]
sweep_table(deg2, "DEG-2 SWEEP (mandelbrot + julia) — the POWERED slice")

for fam in ["mandelbrot", "julia"]:
    sweep_table([r for r in rows if r["fam"] == fam], f"PER-FAMILY: {fam}")

# Unpowered families lumped — just to show over-call behavior, not for tuning.
hi = [r for r in rows if r["fam"] not in DEG2]
sweep_table(hi, "HIGH-DEGREE FAMILIES (lumped, UNPOWERED — do not tune on this)")
