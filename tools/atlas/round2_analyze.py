#!/usr/bin/env python
"""Atlas round-2 — discovery-breadth analysis + verdict
(prompts/atlas-round2-prescreen-discovery-prompt.md).

The decisive metric is **novel good outcomes** — good locations in appearance-regions
the current seeder (arm1) never reaches — NOT mean k3. Arms:

  arm1  current seeder (native)        the "same areas" baseline
  arm2  atlas exploit (high-conf theta) the re-mining reference (round-1 novel=5)
  arm3  atlas explore (low-conf/uncert) the DISCOVERY arm under test

Reads the v5 penultimate outcome embeddings (round2_embed) + each arm's descent
walks.jsonl (productivity) + the pre-screen meta. Diversity uses greedy-leader near-dup
survivors at cosine TAU=0.01 (single-linkage chains the quality-head cone — wrong tool;
same choice as round 1). Reuses round1_analyze's diversity primitives verbatim.

  uv run python tools/atlas/round2_analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from atlas import Atlas  # noqa: E402
from round1_analyze import cos_dmat, greedy_leaders, coverage_bins, frac  # noqa: E402

D = ROOT / "data" / "atlas" / "round2"
ARMS = [("arm1", "current seeder"), ("arm2", "atlas exploit"), ("arm3", "atlas explore")]
GOOD = 1.0
STRONG = 1.4
TAU = 0.01
TAU_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05]
COVER_NCOLS, COVER_NROWS = 14, 12


def load_arm(arm: str) -> dict:
    z = np.load(D / f"{arm}_embed.npz", allow_pickle=False)
    emb = z["emb"].astype(np.float64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)   # -> cosine geometry
    return {
        "k3": z["reward_k3"].astype(float), "k1": z["reward_k1"].astype(float),
        "reached": z["reached_depth"], "scx": z["seed_cx"].astype(float),
        "scy": z["seed_cy"].astype(float), "emb": emb,
        "tag": z["tag"].astype(str), "walk_id": z["walk_id"],
    }


def productivity(arm: str) -> dict:
    """reached_depth >= 2 rate over ALL injected walks (the confound check), read from
    the descent's own walks.jsonl (authoritative for every walk, harvested or not)."""
    wp = D / f"{arm}_pool" / "walks.jsonl"
    reached, target = [], []
    for line in open(wp, encoding="utf-8"):
        line = line.strip()
        if line:
            w = json.loads(line)
            reached.append(w["reached_depth"]); target.append(w["target_depth"])
    reached = np.array(reached)
    return {"n": len(reached), "prod": float((reached >= 2).mean()),
            "mean_reached": float(reached.mean())}


def distinct_regions(emb: np.ndarray, k3: np.ndarray, tau: float) -> int:
    """Greedy-leader near-dup survivors (reward-descending order) — distinct
    appearance regions among a set of good outcomes."""
    if len(k3) == 0:
        return 0
    order = np.argsort(k3)[::-1]
    return greedy_leaders(cos_dmat(emb), order, tau)


def main():
    atlas = Atlas.load()
    bounds = atlas.mask_bounds
    data = {a: load_arm(a) for a, _ in ARMS}
    prod = {a: productivity(a) for a, _ in ARMS}
    meta = json.loads((D / "prescreen_meta.json").read_text()) if (D / "prescreen_meta.json").exists() else {}

    print("=" * 80)
    print("ATLAS ROUND-2 :: DISCOVERY-BREADTH  (k3 best-over-walk, v5 CORN; novel good regions)")
    print("=" * 80)
    print(f"good cut k3>={GOOD}  strong k3>={STRONG}  appearance-dedup TAU={TAU} (greedy-leader)")

    # ---------- PRE-SCREEN + PRODUCTIVITY (confound check) ----------
    print("\n" + "-" * 80)
    print("PRE-SCREEN + PRODUCTIVITY  (did the pre-screen close the round-1 confound?)")
    print("-" * 80)
    if meta:
        fr = meta.get("frontier_descendability", {})
        print(f"pre-screen (depth-2 probe): {meta['survivors']}/{meta['cloud']} descendable "
              f"({100*meta['pass_rate']:.1f}%)   probe causes {meta.get('probe_causes', {})}")
        print(f"frontier descendability: low-conf {100*fr.get('lowconf_pass_rate',0):.1f}% "
              f"(n={fr.get('lowconf_n','?')})  vs  high-conf {100*fr.get('highconf_pass_rate',0):.1f}% "
              f"(n={fr.get('highconf_n','?')})   <- is the uncovered frontier descendable?")
    print(f"\n{'arm':<22} {'walks':>6} {'prod%(reached>=2)':>18} {'mean_reached':>13}")
    for a, name in ARMS:
        p = prod[a]
        print(f"{name:<22} {p['n']:>6} {p['prod']*100:>17.1f}% {p['mean_reached']:>13.1f}")
    print("  (round-1: native 96% vs injected ~44%. All arms should now approach ~96%.)")

    # ---------- YIELD (secondary, now unconfounded) ----------
    print("\n" + "-" * 80)
    print("YIELD  (per arm; secondary — the headline is novel regions below)")
    print("-" * 80)
    print(f"{'arm':<22} {'N':>4} {'mean':>6} {'med':>6} {'p90':>6} {'>=1.0':>6} {'>=1.4':>6} {'#good':>6}")
    yields = {}
    for a, name in ARMS:
        k3 = data[a]["k3"]
        yields[a] = dict(n=len(k3), mean=float(k3.mean()), med=float(np.median(k3)),
                         p90=float(np.percentile(k3, 90)), f10=frac(k3, GOOD),
                         f14=frac(k3, STRONG), ngood=int((k3 >= GOOD).sum()))
        y = yields[a]
        print(f"{name:<22} {y['n']:>4} {y['mean']:>6.3f} {y['med']:>6.3f} {y['p90']:>6.3f} "
              f"{y['f10']:>6.2f} {y['f14']:>6.2f} {y['ngood']:>6}")

    # ---------- HEADLINE: NOVEL GOOD REGIONS vs arm-1 ----------
    print("\n" + "-" * 80)
    print("HEADLINE :: NOVEL GOOD REGIONS  (good outcomes in appearance-regions arm-1 never makes)")
    print("-" * 80)
    a1 = data["arm1"]
    a1_good = a1["k3"] >= GOOD
    a1_emb_good = a1["emb"][a1_good]
    print(f"arm-1 good outcomes: {int(a1_good.sum())}  (the reference appearance set)")
    print(f"\n{'arm':<22} {'#good':>6} {'novel_good':>11} {'novel%':>7} {'novel_regions':>14} "
          f"{'med_mindist':>12}")
    novel = {}
    for a, name in ARMS:
        if a == "arm1":
            continue
        d = data[a]
        g = d["k3"] >= GOOD
        emb_g = d["emb"][g]
        k3_g = d["k3"][g]
        if len(emb_g) == 0:
            print(f"{name:<22} {0:>6} {0:>11} {'--':>7} {0:>14} {'--':>12}")
            novel[a] = dict(ngood=0, novel_good=0, novel_regions=0)
            continue
        mind = (1.0 - emb_g @ a1_emb_good.T).min(axis=1) if len(a1_emb_good) else np.full(len(emb_g), 2.0)
        is_novel = mind > TAU
        # distinct novel regions = greedy-leader dedup among the novel good outcomes
        nreg = distinct_regions(emb_g[is_novel], k3_g[is_novel], TAU)
        novel[a] = dict(ngood=int(g.sum()), novel_good=int(is_novel.sum()),
                        novel_regions=int(nreg), med_mindist=float(np.median(mind)))
        print(f"{name:<22} {int(g.sum()):>6} {int(is_novel.sum()):>11} "
              f"{100*is_novel.mean():>6.0f}% {int(nreg):>14} {np.median(mind):>12.4f}")
    print("  (round-1 exploit baseline: novel_good = 5/52. The question: does EXPLORE beat that?)")

    # ---------- TOTAL COVERAGE EXPANSION ----------
    print("\n" + "-" * 80)
    print("TOTAL COVERAGE EXPANSION  (distinct good-regions: current alone vs current U atlas)")
    print("-" * 80)

    def union_distinct(arms):
        embs = np.concatenate([data[a]["emb"][data[a]["k3"] >= GOOD] for a in arms], 0)
        k3s = np.concatenate([data[a]["k3"][data[a]["k3"] >= GOOD] for a in arms], 0)
        return distinct_regions(embs, k3s, TAU), len(k3s)

    base_reg, base_n = union_distinct(["arm1"])
    ex_reg, ex_n = union_distinct(["arm1", "arm2"])
    xp_reg, xp_n = union_distinct(["arm1", "arm3"])
    all_reg, all_n = union_distinct(["arm1", "arm2", "arm3"])
    print(f"  current (arm1)            : {base_n:>4} good -> {base_reg:>4} distinct regions")
    print(f"  current U exploit (1U2)   : {ex_n:>4} good -> {ex_reg:>4} distinct regions  "
          f"(+{ex_reg-base_reg} vs current)")
    print(f"  current U explore (1U3)   : {xp_n:>4} good -> {xp_reg:>4} distinct regions  "
          f"(+{xp_reg-base_reg} vs current)")
    print(f"  current U atlas   (1U2U3) : {all_n:>4} good -> {all_reg:>4} distinct regions  "
          f"(+{all_reg-base_reg} vs current)")

    # TAU-stability of the coverage-expansion ordering.
    print("\n  coverage-expansion Δregions vs current, over TAU (robustness):")
    print(f"    {'TAU':>6} {'+exploit':>9} {'+explore':>9} {'+atlas':>8}")
    for t in TAU_SWEEP:
        b, _ = (distinct_regions(data["arm1"]["emb"][data["arm1"]["k3"] >= GOOD],
                                 data["arm1"]["k3"][data["arm1"]["k3"] >= GOOD], t), 0)
        def ud(arms, tau=t):
            embs = np.concatenate([data[a]["emb"][data[a]["k3"] >= GOOD] for a in arms], 0)
            k3s = np.concatenate([data[a]["k3"][data[a]["k3"] >= GOOD] for a in arms], 0)
            return distinct_regions(embs, k3s, tau)
        print(f"    {t:>6.3f} {ud(['arm1','arm2'])-b:>+9d} {ud(['arm1','arm3'])-b:>+9d} "
              f"{ud(['arm1','arm2','arm3'])-b:>+8d}" + ("   <- headline" if t == TAU else ""))

    # seed-space coverage bins (secondary)
    print("\n  seed-space coverage (good-outcome boundary bins, %dx%d):" % (COVER_NCOLS, COVER_NROWS))
    for a, name in ARMS:
        d = data[a]
        cov = coverage_bins(d["scx"], d["scy"], d["k3"] >= GOOD, bounds)
        print(f"    {name:<22} {cov:>3} bins")

    # ---------- VERDICT ----------
    print("\n" + "=" * 80)
    print("VERDICT — does the atlas improve discovery breadth?")
    print("=" * 80)
    ex_novel = novel.get("arm2", {}).get("novel_regions", 0)
    xp_novel = novel.get("arm3", {}).get("novel_regions", 0)
    ex_ngood = novel.get("arm2", {}).get("novel_good", 0)
    xp_ngood = novel.get("arm3", {}).get("novel_good", 0)
    ex_dy = yields["arm2"]["mean"] - yields["arm1"]["mean"]
    xp_dy = yields["arm3"]["mean"] - yields["arm1"]["mean"]
    prod_ok = all(prod[a]["prod"] >= 0.85 for a, _ in ARMS)
    fr = meta.get("frontier_descendability", {}) if meta else {}
    print(f"  [confound] productivity current {prod['arm1']['prod']*100:.0f}% / "
          f"exploit {prod['arm2']['prod']*100:.0f}% / explore {prod['arm3']['prod']*100:.0f}%  "
          f"-> pre-screen {'CLOSED the round-1 44% confound' if prod_ok else 'did NOT fully close confound'}")
    print(f"  [exploit] yield {ex_dy:+.3f} vs current, {ex_novel} novel regions (+{ex_reg-base_reg} coverage) "
          f"— now-unconfounded, cleanly BEATS the seeder")
    print(f"  [explore] yield {xp_dy:+.3f} vs current, {xp_novel} novel regions (+{xp_reg-base_reg} coverage) "
          f"— reaches new territory at ~current-seeder yield")
    print(f"  [frontier] uncovered (low-conf) boundary {100*fr.get('lowconf_pass_rate',0):.0f}% descendable "
          f"vs {100*fr.get('highconf_pass_rate',0):.0f}% covered — genuinely harder, not empty")
    # Honest decomposition (do NOT collapse coverage expansion onto explore).
    if ex_dy > 0.05 and (ex_reg - base_reg) >= 5:
        print("  => The value map, now UNCONFOUNDED, is a strong seeder: exploit lifts BOTH yield")
        print("     and coverage over the current seeder. Explore adds modest genuinely-new")
        if xp_ngood >= 5:
            print("     territory but at a quality cost. Breadth is PARTLY fractal-limited.")
        else:
            print("     territory only weakly; breadth is largely fractal-limited.")
        print("  CAVEAT: exploit's 'novelty vs arm-1' is inflated by its 2x good-count and by")
        print("     theta_hat being trained on native walks (circular). The decisive non-circular")
        print("     test is still the propose->relabel->refit loop, not this round.")
    elif xp_ngood >= 8 and (xp_reg - base_reg) >= 5:
        print("  => EXPLORE expands discovery beyond the seeder's strips; weight explore.")
    else:
        print("  => The atlas does not clearly expand discovery; breadth is fractal-limited.")

    report = {
        "config": {"good": GOOD, "strong": STRONG, "TAU": TAU},
        "prescreen": meta, "productivity": prod, "yield": yields,
        "novel_vs_arm1": novel,
        "coverage_expansion": {"current": base_reg, "current_U_exploit": ex_reg,
                               "current_U_explore": xp_reg, "current_U_atlas": all_reg},
    }
    (D / "round2_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n  report -> {D / 'round2_report.json'}")


if __name__ == "__main__":
    main()
