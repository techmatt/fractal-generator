"""Phase 3 — stats over the harvest, on its OWN terms (not vs the library).

Sequential/cyclic split, OKLab extent/coverage distributions, branch-drop
distribution (validates "reals fail by traversal" at corpus scale), dropped_extent
(lost color vs AA fuzz), quarantine count. Gate pass/fail by reason is added once
the gate is wired (Phase 2b) and thresholds are set; this prints what is
threshold-independent now and accepts --gate-thresholds for the rest.

Usage:  python palette_extractor/harvest_stats.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "palette_extractor"))
from palette_extract import gate_quality, EXTENT_FLOOR, ARCLEN_FLOOR

OUT = ROOT / "data" / "wallpaper_harvest"


def pct(a, ps=(0, 10, 25, 50, 75, 90, 100)):
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    man = json.loads((OUT / "manifest.json").read_text())
    E = [e for e in man["entries"] if not e.get("error")]
    n = len(E)
    print(f"=== Harvest stats: {n} ok / {man['errors']} errors / {man['total']} total ===\n")

    cyc = sum(e["classify_cycle"] == "cyclic" for e in E)
    seqlabel = sum(e["cycle_label"] == "sequential" for e in E)
    quar = sum(e["quarantine"] for e in E)
    print(f"closure (classify): cyclic={cyc} ({cyc/n*100:.1f}%)  "
          f"sequential={n-cyc} ({(n-cyc)/n*100:.1f}%)")
    print(f"cycle_label (extractor): sequential={seqlabel} native={n-seqlabel}")
    print(f"quarantine (n_jump>=3): {quar} ({quar/n*100:.1f}%)\n")

    for key, label in [("coverage", "coverage (image)"), ("extent", "extent (color range)"),
                       ("arclen", "arclen (curve len)"),
                       ("branch_drop_frac", "branch_drop_frac"),
                       ("dropped_extent", "dropped_extent"),
                       ("seam_cycle", "seam_cycle")]:
        a = np.array([e[key] for e in E])
        p = pct(a)
        print(f"{label:22s} " + "  ".join(f"{k}={v}" for k, v in p.items()))

    # branch-drop is the headline: reals fail by traversal
    bd = np.array([e["branch_drop_frac"] for e in E])
    print(f"\nbranch_drop_frac: mean={bd.mean():.3f}  frac>0.5={float((bd>0.5).mean()):.3f}  "
          f"frac>0.8={float((bd>0.8).mean()):.3f}")

    # --- failure gate pass/fail by reason (Matt's cutoffs) ---
    gate = {}
    n_low_range = n_degen = n_both = n_reject = 0
    for e in E:
        f = gate_quality(e["extent"], e["arclen"])
        e["_flags"] = f
        if "low_range" in f:
            n_low_range += 1
        if "degenerate" in f:
            n_degen += 1
        if len(f) == 2:
            n_both += 1
        if f:
            n_reject += 1
    print(f"\n=== Failure gate (extent<{EXTENT_FLOOR} -> low_range; "
          f"arclen<{ARCLEN_FLOOR} -> degenerate) ===")
    print(f"  low_range : {n_low_range} ({n_low_range/n*100:.1f}%)")
    print(f"  degenerate: {n_degen} ({n_degen/n*100:.1f}%)")
    print(f"  both flags: {n_both}")
    print(f"  REJECTED  : {n_reject} ({n_reject/n*100:.1f}%)   survivors: {n-n_reject}")
    gate = {"extent_floor": EXTENT_FLOOR, "arclen_floor": ARCLEN_FLOOR,
            "low_range": n_low_range, "degenerate": n_degen, "both": n_both,
            "rejected": n_reject, "survivors": n - n_reject}

    summary = {
        "n_ok": n, "n_err": man["errors"], "total": man["total"],
        "cyclic": cyc, "sequential": n - cyc, "quarantine": quar,
        "cycle_label_sequential": seqlabel,
        "dist": {k: pct(np.array([e[k] for e in E])) for k in
                 ["coverage", "extent", "arclen", "branch_drop_frac",
                  "dropped_extent", "seam_cycle"]},
        "gate": gate,
    }
    (OUT / "stats.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT/'stats.json'}")


if __name__ == "__main__":
    main()
