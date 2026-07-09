"""Step C: side-by-side eval of pref-v3 vs pref-v3-gvo on the SAME union val split.

Both checkpoints are scored on the identical val queries (partitioned verbatim from
v3/split_manifest.json). Reports, for each model:
  - good_vs_okay held-out accuracy (union + dramatic), read against the Step-A
    gvo human ceiling (NOT the 95.6% overall).
  - confident pairs (good_vs_bad, okay_vs_bad) -- the cost side (union + dramatic).
  - overall ranking fidelity (all pairs, union + dramatic + old-slice).
  - within_dramatic specifically (the head-curation-relevant slice).
  - top-region read: per val query, top-1-is-good rate (when a good exists) +
    good-above-okay concordance -- the property top-K curation actually uses.

Read-only on both checkpoints; writes a report JSON to v3_gvo/compare_vs_v3.json.

Run:
    uv run python tools/queries/scorer/compare_v3_gvo.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import train as TV2  # noqa: E402
import train_v3 as TV3  # noqa: E402  (collect_pair_records, _acc, _slice)

V3_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v3")
V3GVO_DIR = os.path.join(D.REPO, "data", "queries", "scorer", "v3_gvo")
V3_MANIFEST = os.path.join(V3_DIR, "split_manifest.json")

# Step-A gvo human self-consistency ceilings (both-passes-strict, gvo pairs only).
# Computed from the pass-1/pass-2 repeats; see report. Read the gvo slice against THESE.
GVO_CEIL_DRAMATIC = 158 / 174   # 0.9080  (prefv2_dramatic_v1 repeats)
GVO_CEIL_UNION = (158 + 69 + 57) / (174 + 74 + 67)  # 0.9016 (pooled 3-batch repeats)


def load_model(ckpt_dir, device):
    model = TV2.build_model().to(device)
    ck = torch.load(os.path.join(ckpt_dir, "model_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("epoch")


def partition_val(queries, manifest_path):
    m = json.load(open(manifest_path))
    val_locs = set(m["val_locations"])
    return [q for q in queries if q.location_key in val_locs]


# ---- top-region read (per-query) ----
@torch.no_grad()
def top_region_read(model, qlist, device, nw):
    """Per query: (a) top-1-is-good when >=1 good exists; (b) good-above-okay
    concordance = frac of (good,okay) candidate pairs with score(good)>score(okay),
    averaged per-query (query with such a pair) then over queries. Also the pooled
    pair-level good>okay rate. Returns dict."""
    ds = D.QueryDataset(qlist, train=False)
    ld = DataLoader(ds, batch_size=TV2.BATCH_QUERIES, shuffle=False,
                    collate_fn=D.collate_queries, num_workers=nw)
    top1_good_hits = top1_good_denom = 0
    per_query_conc = []          # per-query mean good>okay
    pool_conc_hit = pool_conc_n = 0
    for imgs, ranks, idxs in ld:
        B, C = imgs.shape[0], imgs.shape[1]
        flat = imgs.view(B * C, *imgs.shape[2:]).to(device)
        scores = model(flat).view(B, C).cpu()
        for b in range(B):
            r = ranks[b]
            sc = scores[b]
            goods = [i for i in range(C) if r[i].item() == D.TIER_RANK["good"]]
            okays = [i for i in range(C) if r[i].item() == D.TIER_RANK["okay"]]
            if goods:
                top1_good_denom += 1
                if int(sc.argmax().item()) in goods:
                    top1_good_hits += 1
            if goods and okays:
                hits = sum(1 for g in goods for o in okays if sc[g].item() > sc[o].item())
                n = len(goods) * len(okays)
                per_query_conc.append(hits / n)
                pool_conc_hit += hits
                pool_conc_n += n
    return {
        "top1_good_rate": (top1_good_hits / top1_good_denom) if top1_good_denom else None,
        "top1_good_denom": top1_good_denom,
        "good_above_okay_per_query": (sum(per_query_conc) / len(per_query_conc)) if per_query_conc else None,
        "n_queries_with_good_and_okay": len(per_query_conc),
        "good_above_okay_pooled": (pool_conc_hit / pool_conc_n) if pool_conc_n else None,
        "pooled_pairs": pool_conc_n,
    }


def slices(recs):
    """All Step-C slices from a pair-record list."""
    dram = [r for r in recs if r["batch"] == D.DRAMATIC.name]
    old = [r for r in recs if r["batch"] != D.DRAMATIC.name]
    def acc_of(rs, pt=None, qt=None):
        sub = [r for r in rs if (pt is None or r["pair_type"] == pt) and (qt is None or r["query_type"] == qt)]
        return TV3._acc(sub)
    return {
        "overall_union": TV3._acc(recs),
        "overall_dramatic": TV3._acc(dram),
        "overall_old": TV3._acc(old),
        "gvo_union": acc_of(recs, "good_vs_okay"),
        "gvo_dramatic": acc_of(dram, "good_vs_okay"),
        "gvb_union": acc_of(recs, "good_vs_bad"),
        "gvb_dramatic": acc_of(dram, "good_vs_bad"),
        "ovb_union": acc_of(recs, "okay_vs_bad"),
        "ovb_dramatic": acc_of(dram, "okay_vs_bad"),
        "confident_union": TV3._acc([r for r in recs if r["pair_type"] in ("good_vs_bad", "okay_vs_bad")]),
        "confident_dramatic": TV3._acc([r for r in dram if r["pair_type"] in ("good_vs_bad", "okay_vs_bad")]),
        "within_dramatic": acc_of(dram, qt="within_dramatic"),
        "cross_source": acc_of(dram, qt="cross_source"),
        "param_variation": acc_of(dram, qt="param_variation"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = 4 if device.type == "cuda" else 0

    queries = D.load_combined_queries()
    val_q = partition_val(queries, V3_MANIFEST)
    dram_val = [q for q in val_q if q.batch == D.DRAMATIC.name]
    print(f"[compare] val queries={len(val_q)} (dramatic={len(dram_val)}) on v3 split")

    v3, e3 = load_model(V3_DIR, device)
    vg, eg = load_model(V3GVO_DIR, device)
    print(f"[compare] v3 best_epoch={e3}   v3_gvo best_epoch={eg}")

    recs_v3 = TV3.collect_pair_records(v3, val_q, device, nw)
    recs_vg = TV3.collect_pair_records(vg, val_q, device, nw)
    s3, sg = slices(recs_v3), slices(recs_vg)

    tr3 = top_region_read(v3, val_q, device, nw)
    trg = top_region_read(vg, val_q, device, nw)
    tr3_d = top_region_read(v3, dram_val, device, nw)
    trg_d = top_region_read(vg, dram_val, device, nw)

    def fa(x):
        a, n = x
        return f"{a:.3f} (n={n})" if a is not None else f"n/a (n={n})"

    def row(name, k, ceil=None):
        a3, n3 = s3[k]; ag, ng = sg[k]
        d = (ag - a3) if (a3 is not None and ag is not None) else None
        c = f"  [ceil {ceil:.3f}]" if ceil else ""
        ds = f"  d{d:+.3f}" if d is not None else ""
        print(f"  {name:22s} v3 {fa(s3[k]):>16s}   v3-gvo {fa(sg[k]):>16s}{ds}{c}")

    print("\n=== STEP-A GVO HUMAN CEILING (both-passes-strict, gvo pairs only) ===")
    print(f"  dramatic-batch gvo ceiling : {GVO_CEIL_DRAMATIC:.4f}  (158/174)")
    print(f"  pooled 3-batch gvo ceiling : {GVO_CEIL_UNION:.4f}  (284/315)")
    print(f"  (overall both-strict ceiling was 0.956 dramatic / 0.943 old -- NOT the gvo read)")

    print("\n=== STEP-C  v3  vs  v3-gvo  (same union val split) ===")
    print("-- good_vs_okay (the target; read vs gvo ceiling, NOT 0.956) --")
    row("gvo union", "gvo_union", GVO_CEIL_UNION)
    row("gvo dramatic", "gvo_dramatic", GVO_CEIL_DRAMATIC)
    print("-- confident pairs (the COST side -- did easy separations blur?) --")
    row("good_vs_bad union", "gvb_union")
    row("okay_vs_bad union", "ovb_union")
    row("confident union", "confident_union")
    row("good_vs_bad dramatic", "gvb_dramatic")
    row("okay_vs_bad dramatic", "ovb_dramatic")
    row("confident dramatic", "confident_dramatic")
    print("-- overall ranking fidelity (all pairs) --")
    row("overall union", "overall_union")
    row("overall dramatic", "overall_dramatic")
    row("overall old-slice", "overall_old")
    print("-- dramatic query-type slices (within_dramatic = head-curation-relevant) --")
    row("within_dramatic", "within_dramatic")
    row("cross_source", "cross_source")
    row("param_variation", "param_variation")

    print("\n=== TOP-REGION READ (per-query; the property top-K curation uses) ===")
    def tr_row(name, a, b):
        print(f"  {name:26s} v3 top1-good {a['top1_good_rate']:.3f} "
              f"good>okay/q {a['good_above_okay_per_query']:.3f}   "
              f"v3-gvo top1-good {b['top1_good_rate']:.3f} "
              f"good>okay/q {b['good_above_okay_per_query']:.3f}")
    tr_row("union val", tr3, trg)
    tr_row("dramatic val", tr3_d, trg_d)
    print(f"  (union: n_queries_with_good={tr3['top1_good_denom']}, "
          f"n_with_good&okay={tr3['n_queries_with_good_and_okay']}; "
          f"dramatic: {tr3_d['top1_good_denom']}/{tr3_d['n_queries_with_good_and_okay']})")

    report = {
        "gvo_ceiling_dramatic": GVO_CEIL_DRAMATIC,
        "gvo_ceiling_union_pooled": GVO_CEIL_UNION,
        "v3_best_epoch": e3, "v3_gvo_best_epoch": eg,
        "n_val_queries": len(val_q), "n_dramatic_val_queries": len(dram_val),
        "slices": {"v3": {k: {"acc": v[0], "n": v[1]} for k, v in s3.items()},
                   "v3_gvo": {k: {"acc": v[0], "n": v[1]} for k, v in sg.items()}},
        "top_region": {"v3_union": tr3, "v3_gvo_union": trg,
                       "v3_dramatic": tr3_d, "v3_gvo_dramatic": trg_d},
    }
    json.dump(report, open(os.path.join(V3GVO_DIR, "compare_vs_v3.json"), "w"), indent=2)
    print(f"\n[compare] report -> {os.path.join(V3GVO_DIR, 'compare_vs_v3.json')}")


if __name__ == "__main__":
    main()
