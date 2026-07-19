#!/usr/bin/env python
"""Report + ranked contact sheet + next-read manifest for the location preference ranker v0.

Consumes the deployed head (`model.npz`), the eval grid (`metrics.json`), and the feature matrix
(`features.npz`). Emits:
  * out/pref_loc_v0_report.md           -- eval grid, winner, per-family, verdict, uncertainty.
  * out/ranker/pref_loc_v0_ranked_sheet.png  -- all 96 admissions sorted by ranker score, human
                                           labels marked (G/O/B, '-' unlabeled).
  * out/ranker_next_read/               -- ~20-tile next read: top-ranked unlabeled + a
                                           max-uncertainty band, shuffled, with a HIDDEN key.
                                           Seeds the standing loop (every read = fresh eval+train).

    uv run python -m tools.ranker.report
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ranker.scorer import RankerScorer          # noqa: E402

ARTDIR = ROOT / "data/ranker/pref_loc_v0"
REPORT = ROOT / "out/pref_loc_v0_report.md"
SHEET = ROOT / "out/ranker/pref_loc_v0_ranked_sheet.png"
NEXT = ROOT / "out/ranker_next_read"
LABEL = {0: "-", 1: "B", 2: "O", 3: "G"}
LABEL_RGB = {0: (110, 110, 110), 1: (200, 60, 60), 2: (210, 180, 60), 3: (70, 200, 90)}
N_NEXT_TOP = 8            # top-ranked unlabeled
N_NEXT_UNC = 7           # max-uncertainty band


def load():
    z = np.load(ARTDIR / "features.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    s = RankerScorer.load()
    d["pred"] = s.score_matrix({b: d[b] for b in s.sets})
    metrics = json.loads((ARTDIR / "metrics.json").read_text())
    return d, s, metrics


def boundary(d):
    """Threshold in deployed score space midway between labeled good and labeled not-good means."""
    lab = d["score"] > 0
    good = d["pred"][lab & (d["score"] == 3)]
    rest = d["pred"][lab & (d["score"] < 3)]
    return float((good.mean() + rest.mean()) / 2) if len(good) and len(rest) else float(np.median(d["pred"]))


# --------------------------------------------------------------------------- #
# ranked contact sheet
# --------------------------------------------------------------------------- #
def ranked_sheet(d):
    order = np.argsort(-d["pred"])
    tw, th, pad, cols = 232, 150, 4, 8
    n = len(order)
    rows = (n + cols - 1) // cols
    W, H = cols * (tw + pad) + pad, rows * (th + pad) + pad
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    for rank, i in enumerate(order):
        r, c = divmod(rank, cols)
        x, y = pad + c * (tw + pad), pad + r * (th + pad)
        try:
            im = Image.open(d["tiles"][i]).convert("RGB").resize((tw, th - 18))
        except Exception:
            im = Image.new("RGB", (tw, th - 18), (40, 40, 40))
        canvas.paste(im, (x, y))
        hs = int(d["score"][i])
        bar = (LABEL_RGB[hs][0] // 2 + 9, LABEL_RGB[hs][1] // 2 + 9, LABEL_RGB[hs][2] // 2 + 9)
        draw.rectangle([x, y + th - 18, x + tw, y + th], fill=bar)
        fam = str(d["family"][i]).replace("multibrot", "mb").replace("julia:", "j:")
        draw.text((x + 3, y + th - 16), f"#{rank+1} {LABEL[hs]} {d['pred'][i]:+.2f} {fam}",
                  fill=LABEL_RGB[hs])
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(SHEET)
    return SHEET


# --------------------------------------------------------------------------- #
# next-read manifest
# --------------------------------------------------------------------------- #
def next_read(d, seed=0):
    """Next blind read: candidate keepers (pred above the good/not-good boundary) + a
    max-uncertainty band (closest to the boundary) + a confident-low control. Bands are defined by
    boundary position so the split stays meaningful whatever the pool's shape (this steered_run2
    remainder happens to sit mostly below the boundary; the standing loop refills the top band as
    fresh runs land)."""
    rng = np.random.default_rng(seed)
    unl = np.where(d["score"] == 0)[0]
    thr = boundary(d)
    pred = d["pred"]
    top = sorted([i for i in unl if pred[i] >= thr], key=lambda i: -pred[i])[:N_NEXT_TOP]
    rest = [i for i in unl if i not in top]
    unc = sorted(rest, key=lambda i: abs(pred[i] - thr))[:N_NEXT_UNC]
    control = [i for i in rest if i not in unc]
    tag = {**{i: "top" for i in top}, **{i: "uncertain" for i in unc},
           **{i: "control" for i in control}}
    picks = top + unc + control
    rng.shuffle(picks)

    if NEXT.exists():
        shutil.rmtree(NEXT)
    (NEXT / "tiles").mkdir(parents=True)
    blind, key = [], []
    for bi, i in enumerate(picks):
        tile = f"blind_{bi:03d}.jpg"
        shutil.copy(d["tiles"][i], NEXT / "tiles" / tile)
        blind.append(tile)
        key.append(dict(tile=tile, id=str(d["ids"][i]), family=str(d["family"][i]),
                        depth=int(d["depth"][i]), pred=float(d["pred"][i]),
                        canon_pgood=float(d["canon_pgood"][i]), tag=tag[i]))
    (NEXT / "blind_index.json").write_text(json.dumps(dict(
        n=len(blind), note="Blind next-read for pref_loc_v0. Score each tile 1/2/3 (bad/okay/good). "
        "Do not open manifest_key.json.", tiles=blind), indent=2))
    (NEXT / "manifest_key.json").write_text(json.dumps(dict(
        note="HIDDEN KEY — do not show the labeler. tag=top (pred above the good/not-good "
        "boundary; candidate keeper) | uncertain (closest to the boundary; max classifier "
        "uncertainty) | control (confidently below the boundary; negative confirmation).",
        pool_unlabeled=int(len(unl)), boundary=thr,
        band_counts={"top": len(top), "uncertain": len(unc), "control": len(control)},
        entries=key), indent=2))
    return len(blind), len(unl), {"top": len(top), "uncertain": len(unc), "control": len(control)}


# --------------------------------------------------------------------------- #
# markdown report
# --------------------------------------------------------------------------- #
def fmt_row(name, r):
    fd, fr = r["folds"]["dive"], r["folds"]["run2"]
    return (f"| {name} | {r['mean_spearman']:+.3f} | {r['mean_auc']:.3f} | {r['mean_p_at_10']:.3f} "
            f"| {fd['spearman']:+.3f} | {fd['auc']:.3f} | {fr['spearman']:+.3f} | {fr['auc']:.3f} |")


def verdict(m):
    win = m["results"][m["winner"]]
    b_pg = m["baselines"]["canon_pgood"]["mean_spearman"]
    ci = m["pooled_ci"]
    beats = win["mean_spearman"] > b_pg and win["mean_spearman"] > m["baselines"]["random"]["mean_spearman"]
    ci_pos = ci[0] > 0
    if beats and ci_pos and m["perm_p"] < 0.05:
        return ("**SHIPS AS KEEPER-RANKER (provisional).** Beats both baselines on mean-LOBO "
                "Spearman with a CI excluding zero; consume in keeper ranking / emission feed / "
                "dive sorting only. n=81 is small — the standing next-read loop must keep feeding it.")
    if beats and win["mean_spearman"] > 0.1:
        return ("**NEEDS MORE LABELS.** Edges past the baselines but the CI/permutation p do not "
                "clear the bar at n=81. Per the pre-registered rule this means *label more*, not "
                "*the approach failed* — run the next-read manifest and re-eval.")
    return ("**FEATURE SET INSUFFICIENT (at n=81).** Does not clear canonical p_good on mean-LOBO "
            "Spearman. Label more via the next-read manifest before concluding; if it holds after "
            "another read, the frozen-feature/light-head route is the wrong lever.")


def main():
    d, s, m = load()
    sheet = ranked_sheet(d)
    n_next, n_unl, bands = next_read(d)

    L = []
    w = L.append
    w("# Location preference ranker v0 — report\n")
    w("Ranks locations by human quality on steered/dive output — the missing third leg beside "
      "canonical p_good (a *badness* filter, not a goodness ranker; see "
      "`docs/findings/steered_run2_keeper_calibration.md`). **Scope: ranks the not-bad; never "
      "steers.** Consumers = keeper ranking, emission feed, dive-result sorting ONLY — never "
      "frontier priority, dive-start selection, or any discovery-side decision.\n")
    w(f"- Labeled admissions: **{m['n_labeled']}** (run2 60 + dive 21), 96 decoded total.")
    w("- Frozen features: morph-CLIP (768), v7 penultimate (1280), colored-CLIP (768).")
    w("- Heads: Bradley-Terry pairwise (bt), ridge on score, logistic good-vs-rest (logi); "
      "reg by inner 5-fold CV on train (leak-free).")
    w("- Prior corpus (v7-only, 360 balanced rows) available in both folds via `+prior`; never in "
      "eval. Colored/morph CLIP priors omitted — corpus crops carry varied delivered palettes, so "
      "they do not share the target's uniform twilight_shifted appearance space.\n")

    w(f"## Winner: `{m['winner']}`  (reg={m['deploy']['reg']:g})\n")
    w(f"Pooled Spearman (within-fold pct-normalized) = **{m['pooled_spearman']:+.3f}**, "
      f"95% bootstrap CI [{m['pooled_ci'][0]:+.3f}, {m['pooled_ci'][1]:+.3f}], "
      f"permutation p = **{m['perm_p']:.4f}**  (n={m['n_labeled']}).\n")
    w(verdict(m) + "\n")

    w("## Leave-one-batch-out grid\n")
    w("Mean = average over the two eval folds. Fold columns: **dive** = train run2/eval dive, "
      "**run2** = train dive/eval run2.\n")
    w("| config | meanSp | meanAUC | meanP@10 | dive Sp | dive AUC | run2 Sp | run2 AUC |")
    w("|---|---|---|---|---|---|---|---|")
    for name in sorted(m["results"], key=lambda k: -m["results"][k]["mean_spearman"]):
        w(fmt_row("`" + name + "`", m["results"][name]))
    for bn in ("canon_pgood", "random"):
        w(fmt_row("**BASE " + bn + "**", m["baselines"][bn]))
    w("")

    w("## Per-family breakdown (winner, pooled over eval folds)\n")
    w("`mean_pct_good`/`mean_pct_bad` = mean within-fold rank-percentile of that family's human-good "
      "/ human-bad tiles (1.0 = ranker put them top). A useful ranker pushes good high, bad low.\n")
    w("| family | n | n_good | mean_pct_good | mean_pct_bad | Spearman |")
    w("|---|---|---|---|---|---|")
    for f, v in sorted(m["family_breakdown"].items()):
        pg = "—" if v["mean_pct_good"] is None else f"{v['mean_pct_good']:.2f}"
        pb = "—" if v["mean_pct_bad"] is None else f"{v['mean_pct_bad']:.2f}"
        sp = "—" if v["spearman"] is None else f"{v['spearman']:+.2f}"
        w(f"| {f} | {v['n']} | {v['n_good']} | {pg} | {pb} | {sp} |")
    mb4 = m["family_breakdown"].get("multibrot4")
    if mb4:
        pg = mb4["mean_pct_good"]
        learns = pg is not None and pg >= 0.6
        w(f"\n**multibrot4 callout.** n={mb4['n']}, n_good={mb4['n_good']} "
          f"(the dive read had 0/6 multibrot4-good — its 2 goods here are both from run2). "
          + (f"The ranker is **not blind** to it: its good tiles land at mean percentile "
             f"{pg:.2f}, its bad at {mb4['mean_pct_bad']:.2f}. "
             if learns else
             f"mean_pct_good={'—' if pg is None else round(pg,2)}. ")
          + "In the dive fold specifically there are no multibrot4 goods to elevate, so there it "
            "is judged only on not over-ranking the family (neutral).\n")

    # Honest read of WHERE the signal is: cross-family vs within-family.
    fam = m["family_breakdown"]
    w("\n**Where the signal comes from (honest).** Much of the mean-LOBO lift is *cross-family*: "
      f"steered mandelbrot is almost all bad (n_good=0) and the ranker pushes its bad tiles to "
      f"percentile {fam['mandelbrot']['mean_pct_bad']:.2f}, and julia families read strong — so a "
      "large slice of the Spearman is the head learning family-level quality priors, which is "
      "legitimate for a *pooled* keeper set but is not fine within-family taste. Within the "
      "good-rich families it is uneven: multibrot5 orders sensibly (Sp "
      f"{fam['multibrot5']['spearman']:+.2f}, good {fam['multibrot5']['mean_pct_good']:.2f} > bad "
      f"{fam['multibrot5']['mean_pct_bad']:.2f}), but julia:multibrot3 **inverts** (Sp "
      f"{fam['julia:multibrot3']['spearman']:+.2f}, ranks its bad {fam['julia:multibrot3']['mean_pct_bad']:.2f} "
      f"*above* its good {fam['julia:multibrot3']['mean_pct_good']:.2f}) on n=7. This is the n=81 "
      "story: real cross-family separation, noisy within-family ordering. Label more to firm up "
      "within-family taste.\n")

    # Prior finding — it hurt; record it so nobody re-adds it blind.
    pr_r = m["results"].get("v7:logi+prior")
    if pr_r:
        w(f"**Corpus prior verdict: rejected.** `v7:logi+prior` scored meanSp "
          f"{pr_r['mean_spearman']:+.3f} (dive fold {pr_r['folds']['dive']['spearman']:+.3f}) — the "
          "older label corpus *degrades* the ranker, collapsing the dive fold. Different sampling "
          "distribution + delivered-palette appearance; the v7 penultimate does not carry it across "
          "cleanly. Do not fold the prior in without a distribution-matched re-derivation.\n")

    w("## Deliverables\n")
    w(f"- Ranked contact sheet (all 96, sorted by ranker score, human label marked): `{sheet.relative_to(ROOT)}`")
    w(f"- Next-read manifest: `{NEXT.relative_to(ROOT)}/` — {n_next} tiles "
      f"({bands['top']} top-ranked/candidate + {bands['uncertain']} max-uncertainty + "
      f"{bands['control']} confident-low control), shuffled, hidden key. "
      f"Unread pool = {n_unl} unlabeled admissions"
      + (f" (< 20 requested — this is the entire steered_run2 non-blind remainder, and it sits mostly "
         "below the good/not-good boundary, so only "
         f"{bands['top']} read as candidate keepers; the standing loop refills the top band as new "
         "runs land)." if n_unl < 20 else "."))
    w(f"- Model artifact: `{(ARTDIR/'model.npz').relative_to(ROOT)}`; scorer "
      "`tools/ranker/scorer.py` (`RankerScorer.load()`).")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"wrote {sheet}")
    print(f"wrote next-read manifest: {NEXT} ({n_next} tiles, pool {n_unl})")


if __name__ == "__main__":
    main()
