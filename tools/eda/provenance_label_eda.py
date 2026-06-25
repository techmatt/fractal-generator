"""Provenance x label EDA — panels A-E + findings stats.

See prompts/provenance-label-eda.md. Read-only, visual-first. Writes figures + a
findings stats JSON to data/eda/. Run: `uv run python tools/eda/provenance_label_eda.py`.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eda_common as C

OUT = C.EDA_DIR
os.makedirs(OUT, exist_ok=True)

# Label colors (1 bad / 2 okay / 3 good).
LCOL = {1: "#444444", 2: "#f2a900", 3: "#e0301e"}
FINDINGS = {}


# ----------------------------------------------------------------------------- A
def panel_A(cohorts):
    rev4 = cohorts["rev4"]
    unbiased = cohorts["unbiased"]
    field, extent = C.load_field_downsampled(1100)

    # walk -> depth-1 entry center (from run4 pool), and walk -> best label (rev4 labels).
    pool = C.load_pool(C.POOL_RUN4)
    entry = {}
    for r in pool:
        if r["depth"] == 1:
            entry[r["walk"]] = (r["cx"], r["cy"])
    best = defaultdict(int)
    for r in rev4:
        if r["score"]:
            best[r["walk_id"]] = max(best[r["walk_id"]], r["score"])

    fig, axes = plt.subplots(1, 2, figsize=(17, 9), constrained_layout=True)
    bg = np.log1p(np.nan_to_num(field, nan=0.0))
    for ax in axes:
        cmap = plt.cm.bone.copy()
        ax.imshow(bg, extent=extent, origin="upper", cmap=cmap,
                  vmin=np.nanpercentile(bg, 2), vmax=np.nanpercentile(bg, 99),
                  aspect="auto", alpha=0.85)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Re(c)"); ax.set_ylabel("Im(c)")

    # (i) walk entry points colored by walk best label
    ax = axes[0]
    for wid, (cx, cy) in entry.items():
        b = best.get(wid, 0)
        if b == 0:
            ax.scatter(cx, cy, s=22, c="#2266cc", marker="x", linewidths=0.8, alpha=0.5)
        else:
            ax.scatter(cx, cy, s=70 if b == 3 else 45, c=LCOL[b],
                       edgecolors="white", linewidths=0.6, zorder=3 + b)
    ax.set_title("(i) Walk entry points (depth-1 root windows)\ncolored by walk's BEST label "
                 "(x = no labeled positive)")

    # (ii) labeled frames cx/cy colored by label (unbiased)
    ax = axes[1]
    for sc in (1, 2, 3):
        pts = [(r["cx"], r["cy"]) for r in unbiased if r["score"] == sc]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=14 if sc == 1 else (40 if sc == 2 else 75),
                       c=LCOL[sc], edgecolors="white" if sc > 1 else "none",
                       linewidths=0.5, alpha=0.55 if sc == 1 else 0.95,
                       zorder=2 + sc, label=f"{sc} (n={len(pts)})")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("(ii) Labeled frame centers (unbiased descent)\ncolored by label")

    fig.suptitle("Panel A — base-set goodness map (unbiased descent; 8k smooth-iter field backdrop)",
                 fontsize=14)
    fig.savefig(os.path.join(OUT, "panelA_baseset_goodness.png"), dpi=110)
    plt.close(fig)

    # Coarse "where" stat: positive entry walks vs all, by real-axis half.
    pos_walks = [w for w in entry if best.get(w, 0) >= 2]
    FINDINGS["A"] = {
        "n_walks_total": len(entry),
        "n_walks_with_notbad": len(pos_walks),
        "n_walks_with_good": len([w for w in entry if best.get(w, 0) == 3]),
    }


# ----------------------------------------------------------------------------- B
def panel_B(cohorts):
    unbiased = cohorts["unbiased"]
    rev4 = cohorts["rev4"]

    # rate vs depth
    depths = sorted({r["depth"] for r in unbiased if r["depth"]})
    nb_rate, g_rate, ns = [], [], []
    for d in depths:
        sub = [r for r in unbiased if r["depth"] == d and r["score"]]
        n = len(sub)
        ns.append(n)
        nb_rate.append(sum(C.notbad(r) for r in sub) / n if n else 0)
        g_rate.append(sum(C.good(r) for r in sub) / n if n else 0)

    # walk terminal length (run4 pool) -> P(walk yields >=1 positive)
    pool = C.load_pool(C.POOL_RUN4)
    term_len = defaultdict(int)
    for r in pool:
        term_len[r["walk"]] = max(term_len[r["walk"]], r["depth"])
    walk_best = defaultdict(int)
    labeled_walks = set()
    for r in rev4:
        if r["score"]:
            labeled_walks.add(r["walk_id"])
            walk_best[r["walk_id"]] = max(walk_best[r["walk_id"]], r["score"])
    # only walks that actually got labeled frames
    buckets = defaultdict(list)  # terminal_len -> list of best label (>=2 = positive)
    for w in labeled_walks:
        buckets[term_len.get(w, 0)].append(walk_best[w])
    blens = sorted(buckets)
    p_pos = [sum(1 for b in buckets[L] if b >= 2) / len(buckets[L]) for L in blens]
    bns = [len(buckets[L]) for L in blens]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    ax = axes[0]
    x = np.array(depths)
    ax.plot(x, nb_rate, "-o", color=LCOL[2], label="not-bad (>=2) rate")
    ax.plot(x, g_rate, "-o", color=LCOL[3], label="good (3) rate")
    for xi, nbi, ni in zip(x, nb_rate, ns):
        ax.annotate(f"n={ni}", (xi, nbi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="gray")
    ax.set_xlabel("frame depth"); ax.set_ylabel("rate"); ax.set_xticks(x)
    ax.legend(); ax.set_title("Label rate vs frame depth (unbiased)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(blens, p_pos, "-o", color=LCOL[2])
    for xi, yi, ni in zip(blens, p_pos, bns):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="gray")
    ax.set_xlabel("walk terminal length (max depth reached)")
    ax.set_ylabel("P(walk yields >=1 not-bad)")
    ax.set_title("Walk length vs walk productivity (rev4)")
    ax.grid(alpha=0.3)

    fig.suptitle("Panel B — goodness vs depth & walk length", fontsize=14)
    fig.savefig(os.path.join(OUT, "panelB_depth_walklen.png"), dpi=110)
    plt.close(fig)

    FINDINGS["B"] = {
        "depths": depths, "n_per_depth": ns,
        "notbad_rate_per_depth": [round(v, 3) for v in nb_rate],
        "good_rate_per_depth": [round(v, 3) for v in g_rate],
        "walk_len_buckets": blens, "n_walks_per_len": bns,
        "p_walk_positive_per_len": [round(v, 3) for v in p_pos],
    }


# ----------------------------------------------------------------------------- C
def panel_C(cohorts):
    unbiased = cohorts["unbiased"]

    def by(key, vals):
        out = {}
        for v in vals:
            sub = [r for r in unbiased if r.get(key) == v and r["score"]]
            n = len(sub)
            out[v] = (n,
                      sum(C.notbad(r) for r in sub) / n if n else 0,
                      sum(C.good(r) for r in sub) / n if n else 0)
        return out

    root_src = by("root_src", ["8k", "flat"])
    branches = ["foci", "density", "random", "rootflat", "root8k"]
    bydict = by("branch", branches)

    # draw weight vs positive share (target branches only, depth>=2)
    target = ["foci", "density", "random"]
    draws = Counter(); pos = Counter()
    for r in unbiased:
        b = r.get("branch")
        if b in target and r.get("depth", 0) and r["depth"] >= 2 and r["score"]:
            draws[b] += 1
            if C.notbad(r):
                pos[b] += 1
    tot_draw = sum(draws.values()); tot_pos = sum(pos.values())
    lift = {b: ((pos[b] / tot_pos if tot_pos else 0), (draws[b] / tot_draw if tot_draw else 0),
                C.TARGET_MIX[b]) for b in target}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    def barpanel(ax, d, title, order):
        keys = [k for k in order if k in d]
        xs = np.arange(len(keys)); w = 0.38
        ax.bar(xs - w / 2, [d[k][1] for k in keys], w, color=LCOL[2], label="not-bad rate")
        ax.bar(xs + w / 2, [d[k][2] for k in keys], w, color=LCOL[3], label="good rate")
        for i, k in enumerate(keys):
            ax.annotate(f"n={d[k][0]}", (i, max(d[k][1], d[k][2])),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(keys, rotation=20)
        ax.set_ylabel("rate"); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3, axis="y")

    barpanel(axes[0], root_src, "by root_src", ["8k", "flat"])
    barpanel(axes[1], bydict, "by branch (placing branch of frame)", branches)

    ax = axes[2]
    xs = np.arange(len(target)); w = 0.38
    ax.bar(xs - w / 2, [lift[b][2] for b in target], w, color="#888", label="draw weight (target_mix)")
    ax.bar(xs + w / 2, [lift[b][0] for b in target], w, color=LCOL[2], label="share of not-bad positives")
    ax.set_xticks(xs); ax.set_xticklabels(target)
    ax.set_ylabel("share"); ax.set_title("branch draw-weight vs positive-share\n(depth>=2 target branches)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Panel C — provenance categoricals (unbiased)", fontsize=14)
    fig.savefig(os.path.join(OUT, "panelC_provenance.png"), dpi=110)
    plt.close(fig)

    FINDINGS["C"] = {
        "root_src": {k: {"n": v[0], "notbad": round(v[1], 3), "good": round(v[2], 3)}
                     for k, v in root_src.items()},
        "branch": {k: {"n": v[0], "notbad": round(v[1], 3), "good": round(v[2], 3)}
                   for k, v in bydict.items()},
        "branch_lift_depth2plus": {b: {"pos_share": round(lift[b][0], 3),
                                       "draw_share": round(lift[b][1], 3),
                                       "target_weight": lift[b][2]} for b in target},
    }


# ----------------------------------------------------------------------------- D
def panel_D(cohorts):
    unbiased = cohorts["unbiased"]
    fields = ["interior_frac", "occupancy", "black_fraction"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    stats = {}
    for ax, f in zip(axes, fields):
        per = {1: [], 2: [], 3: []}
        for r in unbiased:
            v = r.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if r["score"] in per:
                per[r["score"]].append(v)
        # strip plot with jitter
        for sc in (1, 2, 3):
            ys = np.array(per[sc], dtype=float)
            if len(ys) == 0:
                continue
            xs = sc + (np.random.default_rng(sc).random(len(ys)) - 0.5) * 0.6
            ax.scatter(xs, ys, s=12 if sc == 1 else 30, c=LCOL[sc],
                       alpha=0.35 if sc == 1 else 0.8, edgecolors="none")
            # median bar
            ax.plot([sc - 0.35, sc + 0.35], [np.median(ys)] * 2, color="black", lw=2)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["bad", "okay", "good"])
        ax.set_title(f); ax.set_xlabel("label"); ax.grid(alpha=0.3, axis="y")
        if f == "black_fraction":
            ax.axhline(C.GUARD_BLACK_CAP, color="red", ls="--", lw=1.2, label="black cap 0.30")
            ax.legend(fontsize=8)
        if f == "occupancy":
            ax.axhline(C.GUARD_OCC_FLOOR, color="red", ls="--", lw=1.2, label="occ floor 0.321")
            ax.legend(fontsize=8)
        if f == "interior_frac":
            ax.set_yscale("symlog", linthresh=1e-4)
        stats[f] = {sc: {"n": len(per[sc]),
                         "median": round(float(np.median(per[sc])), 5) if per[sc] else None,
                         "p25": round(float(np.percentile(per[sc], 25)), 5) if per[sc] else None,
                         "p75": round(float(np.percentile(per[sc], 75)), 5) if per[sc] else None}
                    for sc in (1, 2, 3)}
    fig.suptitle("Panel D — per-frame stats vs label (unbiased; guards overlaid)", fontsize=14)
    fig.savefig(os.path.join(OUT, "panelD_frame_stats.png"), dpi=110)
    plt.close(fig)
    FINDINGS["D"] = stats


# ----------------------------------------------------------------------------- E
def _warmth_cache(cohorts):
    """Realized palette warmth = mean(R-B)/255 over central 60% of the rendered crop.

    Measures warmth as Matt actually saw it (palette x geometry), not a name heuristic.
    Cached to data/eda/warmth_cache.json keyed by 'batch/image_id'.
    """
    from PIL import Image
    cache_path = os.path.join(OUT, "warmth_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    need = []
    for name in ("rev4", "enriched", "random_eval"):
        for r in cohorts[name]:
            key = f"{r['batch']}/{r['image_id']}"
            if key not in cache:
                need.append((key, r["batch"], r["image_id"]))
    for i, (key, batch, iid) in enumerate(need):
        try:
            im = Image.open(C.crop_path(batch, iid)).convert("RGB")
            im = im.resize((96, 54))
            a = np.asarray(im, dtype=float)
            h, w, _ = a.shape
            c = a[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            cache[key] = float((c[..., 0].mean() - c[..., 2].mean()) / 255.0)
        except Exception:
            cache[key] = None
        if (i + 1) % 400 == 0:
            print(f"  warmth {i+1}/{len(need)}")
    if need:
        json.dump(cache, open(cache_path, "w"))
    return cache


def panel_E(cohorts):
    warm = _warmth_cache(cohorts)

    def w_of(r):
        return warm.get(f"{r['batch']}/{r['image_id']}")

    # (i) within-location (rev4): geometry fixed = same draw_index, palette varies.
    rev4 = cohorts["rev4"]
    by_loc = defaultdict(list)
    for r in rev4:
        if r["score"]:
            by_loc[r["draw_index"]].append(r)
    paired = []  # (delta_warm, delta_score) for every within-location ordered pair where score differs
    n_multi = 0
    agree = disagree = 0
    for loc, rs in by_loc.items():
        if len(rs) < 2:
            continue
        n_multi += 1
        for i in range(len(rs)):
            for j in range(len(rs)):
                if i == j:
                    continue
                a, b = rs[i], rs[j]
                if a["score"] == b["score"]:
                    continue
                wa, wb = w_of(a), w_of(b)
                if wa is None or wb is None:
                    continue
                dw = wa - wb
                ds = a["score"] - b["score"]
                paired.append((dw, ds))
                if dw == 0:
                    continue
                if (dw > 0) == (ds > 0):
                    agree += 1
                else:
                    disagree += 1

    # (ii) warmth distribution of positives: rev4 (unbiased) vs enriched (biased).
    rev4_pos_w = [w_of(r) for r in rev4 if C.notbad(r) and w_of(r) is not None]
    enr_pos_w = [w_of(r) for r in cohorts["enriched"] if C.notbad(r) and w_of(r) is not None]
    rev4_all_w = [w_of(r) for r in rev4 if r["score"] and w_of(r) is not None]
    enr_all_w = [w_of(r) for r in cohorts["enriched"] if r["score"] and w_of(r) is not None]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # E(i) scatter of within-location paired deltas
    ax = axes[0]
    if paired:
        dw, ds = zip(*paired)
        jit = (np.random.default_rng(0).random(len(ds)) - 0.5) * 0.25
        ax.scatter(dw, np.array(ds) + jit, s=18, alpha=0.4, color="#7733aa")
        ax.axvline(0, color="gray", lw=0.8); ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("delta warmth (palette A - B, same location)")
    ax.set_ylabel("delta label (A - B)")
    ax.set_title(f"E(i) within-location palette effect\n{n_multi} multi-palette locs; "
                 f"warm->better agree={agree} disagree={disagree}")
    ax.grid(alpha=0.3)

    # E(ii) warmth distributions
    ax = axes[1]
    bins = np.linspace(-0.6, 0.8, 30)
    ax.hist(rev4_all_w, bins=bins, density=True, alpha=0.35, color="#888", label=f"rev4 all (n={len(rev4_all_w)})")
    ax.hist(rev4_pos_w, bins=bins, density=True, histtype="step", lw=2, color=LCOL[2],
            label=f"rev4 positives (n={len(rev4_pos_w)})")
    ax.hist(enr_pos_w, bins=bins, density=True, histtype="step", lw=2, color=LCOL[3],
            label=f"enriched positives (n={len(enr_pos_w)})")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("realized crop warmth (R-B)/255"); ax.set_ylabel("density")
    ax.set_title("E(ii) warmth of positives:\nunbiased vs v2-enriched"); ax.legend(fontsize=8)

    # E(ii') warmth of ALL enriched draws vs unbiased draws (shows v2's selection skew)
    ax = axes[2]
    ax.hist(rev4_all_w, bins=bins, density=True, alpha=0.4, color="#3377cc", label=f"rev4 draws (n={len(rev4_all_w)})")
    ax.hist(enr_all_w, bins=bins, density=True, alpha=0.4, color="#cc5533", label=f"enriched draws (n={len(enr_all_w)})")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("realized crop warmth (R-B)/255"); ax.set_ylabel("density")
    ax.set_title("v2 selection skew:\nwhat v2 chose to render (warm?)"); ax.legend(fontsize=8)

    fig.suptitle("Panel E — palette warmth vs labels (only place enriched is used)", fontsize=14)
    fig.savefig(os.path.join(OUT, "panelE_palette_warmth.png"), dpi=110)
    plt.close(fig)

    def med(x):
        return round(float(np.median(x)), 4) if len(x) else None
    FINDINGS["E"] = {
        "within_loc_multi_palette_locs": n_multi,
        "warm_better_agree": agree, "warm_better_disagree": disagree,
        "median_warmth_rev4_all": med(rev4_all_w),
        "median_warmth_rev4_positives": med(rev4_pos_w),
        "median_warmth_enriched_positives": med(enr_pos_w),
        "median_warmth_enriched_all_draws": med(enr_all_w),
    }
    # palette frequency among positives (names)
    def top_pal(recs):
        c = Counter(r["palette"] for r in recs if C.notbad(r))
        return c.most_common(12)
    FINDINGS["E"]["top_palettes_rev4_positives"] = top_pal(rev4)
    FINDINGS["E"]["top_palettes_enriched_positives"] = top_pal(cohorts["enriched"])


def main():
    cohorts = C.load_cohorts()
    # cohort summary
    summ = {}
    for name in ("rev4", "random_eval", "enriched", "loose0", "unbiased"):
        lab = C.labeled(cohorts[name])
        summ[name] = {"labeled": len(lab),
                      "notbad": sum(C.notbad(r) for r in lab),
                      "good": sum(C.good(r) for r in lab)}
    FINDINGS["cohorts"] = summ

    panel_A(cohorts); print("A done")
    panel_B(cohorts); print("B done")
    panel_C(cohorts); print("C done")
    panel_D(cohorts); print("D done")
    panel_E(cohorts); print("E done")

    json.dump(FINDINGS, open(os.path.join(OUT, "findings_stats.json"), "w"), indent=2)
    print("wrote findings_stats.json")


if __name__ == "__main__":
    main()
