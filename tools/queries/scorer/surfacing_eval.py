"""Measurement-only eval: does v1 surface human-good candidates to the top of a
within-query ranking? Loads v1 model_best.pt as-is and scores. No retraining.

Reuses the training pipeline verbatim:
- data.py: load_queries (pass-1 only), QueryDataset transform (train=False),
  cross-tier pair helpers, TIER_RANK.
- train.py: build_model (timm construction path).
- split_manifest.json: val_locations used verbatim (no split recompute).

Every metric is within-query; raw scalars are never pooled across queries.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import train as TR  # noqa: E402

V1_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v1")
OUT_DIR = os.path.join(V1_DIR, "surfacing_eval")
MONT_DIR = os.path.join(OUT_DIR, "montages")

TIER_COLOR = {"good": (60, 200, 90), "okay": (230, 190, 40), "bad": (225, 70, 70)}


def load_val_queries():
    """Filter all pass-1 queries down to the val locations from split_manifest.json."""
    queries = D.load_queries()
    manifest = json.load(open(os.path.join(V1_DIR, "split_manifest.json")))
    val_locs = set(manifest["val_locations"])
    val_q = [q for q in queries if q.location_key in val_locs]
    # sanity: manifest says how many val queries to expect
    assert len(val_q) == manifest["n_queries_val"], (
        f"val query count {len(val_q)} != manifest {manifest['n_queries_val']}"
    )
    return val_q


@torch.no_grad()
def score_queries(model, val_q, device):
    """Score every candidate in every val query through the exact deploy transform.
    Returns per-query list of dicts (aligned with candidate order 0..5)."""
    ds = D.QueryDataset(val_q, train=False)  # train=False -> no flips, deploy transform
    model.eval()
    per_query = []
    for i in range(len(ds)):
        imgs, ranks, idx = ds[i]  # imgs [6,3,224,224]
        q = val_q[idx]
        scores = model(imgs.to(device)).view(-1).cpu().tolist()
        cands = []
        for ci in range(len(q.tiers)):
            cands.append({
                "candidate_id": f"{q.query_id}_{ci}",
                "score": scores[ci],
                "tier": q.tiers[ci],
                "image_path": q.image_paths[ci],
            })
        per_query.append({
            "query_id": q.query_id,
            "query_type": q.query_type,
            "location_key": q.location_key,
            "candidates": cands,
        })
    return per_query


# ---- metrics -------------------------------------------------------------
def compute_metrics(per_query):
    n_q = len(per_query)
    top1_tier_dist = {"good": 0, "okay": 0, "bad": 0}
    top1_good = top1_bad = 0
    topk_good = {2: 0, 3: 0}
    random_good_expect = []
    # good recall in top-3
    total_good = 0
    good_in_top3 = 0
    # normalized within-query rank by tier
    rank_by_tier = {"good": [], "okay": [], "bad": []}
    # pair-direction accuracy by type
    per_type = {"good_vs_bad": [0, 0], "good_vs_okay": [0, 0], "okay_vs_bad": [0, 0]}
    tier_name = {v: k for k, v in D.TIER_RANK.items()}

    for pq in per_query:
        cands = pq["candidates"]
        n = len(cands)
        # sort by score desc; tie-break stable by candidate order
        order = sorted(range(n), key=lambda ci: (-cands[ci]["score"], ci))
        sorted_tiers = [cands[ci]["tier"] for ci in order]

        # top-1
        top1_tier = sorted_tiers[0]
        top1_tier_dist[top1_tier] += 1
        if top1_tier == "good":
            top1_good += 1
        if top1_tier == "bad":
            top1_bad += 1

        # top-k contains-a-good
        for k in (2, 3):
            if "good" in sorted_tiers[:k]:
                topk_good[k] += 1

        # random baseline for top-1-good
        n_good = sum(1 for t in [c["tier"] for c in cands] if t == "good")
        random_good_expect.append(n_good / n)

        # good recall in top-3
        for pos, ci in enumerate(order):
            t = cands[ci]["tier"]
            norm_rank = pos / (n - 1)  # 0 = top, 1 = bottom
            rank_by_tier[t].append(norm_rank)
            if t == "good":
                total_good += 1
                if pos < 3:
                    good_in_top3 += 1

        # pair-direction accuracy (mirror eval_pair_accuracy)
        ranks = torch.tensor([D.TIER_RANK[c["tier"]] for c in cands])
        score_t = torch.tensor([c["score"] for c in cands])
        for hi, lo in TR.query_pairs_from_ranks(ranks):
            correct = int((score_t[hi] - score_t[lo]).item() > 0)
            name = D.pair_type_name(tier_name[ranks[hi].item()], tier_name[ranks[lo].item()])
            per_type[name][1] += 1
            per_type[name][0] += correct

    metrics = {
        "n_val_queries": n_q,
        "primary": {
            "top1_good_rate": top1_good / n_q,
            "top1_tier_distribution": top1_tier_dist,
            "top1_bad_rate": top1_bad / n_q,
            "random_top1_good_baseline": sum(random_good_expect) / n_q,
        },
        "secondary": {
            "top2_contains_good_rate": topk_good[2] / n_q,
            "top3_contains_good_rate": topk_good[3] / n_q,
            "good_recall_in_top3": (good_in_top3 / total_good) if total_good else None,
            "total_good_candidates": total_good,
            "mean_norm_rank_by_tier": {
                t: (sum(v) / len(v) if v else None) for t, v in rank_by_tier.items()
            },
            "n_by_tier": {t: len(v) for t, v in rank_by_tier.items()},
        },
        "pair_direction_accuracy": {
            k: {"acc": (v[0] / v[1] if v[1] else None), "n": v[1]}
            for k, v in per_type.items()
        },
    }
    return metrics


# ---- montages ------------------------------------------------------------
def make_montage(pq, path):
    cands = pq["candidates"]
    n = len(cands)
    order = sorted(range(n), key=lambda ci: (-cands[ci]["score"], ci))

    thumb_w, thumb_h = 240, 135  # 16:9
    label_h = 46
    pad = 6
    cell_w = thumb_w + pad
    W = cell_w * n + pad
    H = thumb_h + label_h + 2 * pad + 24  # +header

    canvas = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 4), f"{pq['query_id']}  ({pq['query_type']})  - sorted by v1 score desc",
              fill=(210, 210, 210))

    y0 = 24 + pad
    for rank, ci in enumerate(order):
        c = cands[ci]
        x0 = pad + rank * cell_w
        try:
            im = Image.open(c["image_path"]).convert("RGB").resize((thumb_w, thumb_h), Image.BICUBIC)
        except Exception as e:  # noqa
            im = Image.new("RGB", (thumb_w, thumb_h), (60, 0, 0))
        canvas.paste(im, (x0, y0))
        ly = y0 + thumb_h + 2
        draw.text((x0 + 2, ly), f"#{rank + 1}  s={c['score']:.3f}", fill=(225, 225, 225))
        tier = c["tier"]
        draw.text((x0 + 2, ly + 16), tier.upper(), fill=TIER_COLOR[tier])
    canvas.save(path)


def fmt_table(m):
    p = m["primary"]; s = m["secondary"]; pa = m["pair_direction_accuracy"]
    L = []
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| val queries | {m['n_val_queries']} |")
    L.append(f"| **top-1-good rate** | **{p['top1_good_rate']:.3f}** |")
    L.append(f"| random top-1-good baseline | {p['random_top1_good_baseline']:.3f} |")
    td = p["top1_tier_distribution"]
    L.append(f"| top-1 tier dist (good/okay/bad) | {td['good']} / {td['okay']} / {td['bad']} |")
    L.append(f"| **top-1-bad rate** | **{p['top1_bad_rate']:.3f}** |")
    L.append(f"| top-2 contains-a-good | {s['top2_contains_good_rate']:.3f} |")
    L.append(f"| top-3 contains-a-good | {s['top3_contains_good_rate']:.3f} |")
    L.append(f"| good recall in top-3 | {s['good_recall_in_top3']:.3f} ({s['total_good_candidates']} goods) |")
    mr = s["mean_norm_rank_by_tier"]
    L.append(f"| mean norm rank good/okay/bad (0=top) | {mr['good']:.3f} / {mr['okay']:.3f} / {mr['bad']:.3f} |")
    for k in ("good_vs_bad", "good_vs_okay", "okay_vs_bad"):
        L.append(f"| pair acc {k} | {pa[k]['acc']:.3f} (n={pa[k]['n']}) |")
    return "\n".join(L)


def main():
    os.makedirs(MONT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_q = load_val_queries()
    print(f"[surfacing_eval] {len(val_q)} val queries")

    model = TR.build_model().to(device)
    ck = torch.load(os.path.join(V1_DIR, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"[surfacing_eval] loaded model_best.pt (epoch {ck.get('epoch')})")

    per_query = score_queries(model, val_q, device)
    json.dump({"queries": per_query}, open(os.path.join(OUT_DIR, "scores.json"), "w"), indent=2)

    metrics = compute_metrics(per_query)
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)

    # montages
    for pq in per_query:
        make_montage(pq, os.path.join(MONT_DIR, f"{pq['query_id']}.png"))
    print(f"[surfacing_eval] wrote {len(per_query)} montages")

    # SUMMARY.md
    table = fmt_table(metrics)
    p = metrics["primary"]
    gvb = metrics["pair_direction_accuracy"]["good_vs_bad"]["acc"]
    harness_ok = abs(gvb - 0.773) < 0.02
    prose = (
        f"v1 surfaces a human-good candidate as the top-scored render in "
        f"**{p['top1_good_rate']:.0%}** of val queries, against a random baseline of "
        f"{p['random_top1_good_baseline']:.0%}. Its argmax is a human-**bad** candidate in "
        f"**{p['top1_bad_rate']:.0%}** of queries. The good_vs_bad pair-direction accuracy is "
        f"{gvb:.3f}, which {'reproduces' if harness_ok else 'DIVERGES FROM'} v1's reported val "
        f"0.773 ({'harness faithful' if harness_ok else 'PIPELINE MISMATCH -- investigate'})."
    )
    summary = f"# v1 surfacing eval\n\n{table}\n\n{prose}\n"
    open(os.path.join(OUT_DIR, "SUMMARY.md"), "w").write(summary)

    print("\n" + summary)
    print("=== metrics.json ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
