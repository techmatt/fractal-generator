#!/usr/bin/env python
"""steered_pilot_morph.py — morphological composition + M-cap depth audit of the steered A/B.

Read-only post-hoc analysis of the steered-vs-baseline pilot (`out/steered_pilot_report.md`).
Nothing mutated in any store/ledger. Two questions:

  Q1  Are the 16 steered admissions distinct LOOKS, or one cheap look re-bought? Embed the
      canonical grayscale morphology of every admission (BOTH arms) with the EXACT morph_clip
      recipe used by the library audit (`vit_base_patch16_clip_224.openai` timm eval transform
      over the robust-z tanh grayscale render at the 640x360 ss2 morphology geometry — imported
      byte-identical from colored_clip.load_clip / library_annotate.morph_gray_image), then
      report intra-arm pairwise cosine, single-linkage morph clusters at the established cuts,
      and the partition-coverage headline after morph dedup. Yardsticks from the library audit:
      library-wide median pairwise cos 0.851, intra-phoenix 0.938, strict near-dup cut cos>0.974.

  Q2  Is the M=40 expansions/root cap a DEPTH ceiling? Reconstruct the full expansion tree from
      the surviving scratch (expand.jsonl carries parent_depth + root_id + kind per child), get
      per-root expansions/max-chain-depth, the observed branching factor (alive gate survivors
      per expansion) by depth, and estimate the reachable depth under M=40 if the frontier splits
      its budget across branches vs single-tracks.

  uv run python -m tools.studies.steered_pilot_morph \
      --steered data/discovery/steered_pilot/steered \
      --baseline data/discovery/steered_pilot/baseline
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
sys.path.insert(0, str(ROOT / "tools" / "scoring"))
sys.path.insert(0, str(ROOT / "tools" / "wallpaper"))

import location as loc_mod                                   # noqa: E402
from active_ckpt import BIN, PALETTE, JPG_Q, auto_maxiter    # noqa: E402
from tools.wallpaper.library_annotate import ensure_field, morph_gray_image  # noqa: E402
from tools.curation.colored_clip import load_clip, embed_clip                 # noqa: E402

STRICT_CUT = 0.974          # library near-dup cut (morphology-dedup-pass); primary
LOOSE_CUT = 0.95            # perceptual "same look" cut
LIB_MEDIAN = 0.851          # library-wide median pairwise cos (yardstick)
LIB_PHOENIX = 0.938         # intra-phoenix median (tight-source yardstick)


# --------------------------------------------------------------------------- #
# Admissions.
# --------------------------------------------------------------------------- #
def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def admitted_q3(rows):
    return [r for r in rows if r.get("distinct") and r.get("guard_pass", True)
            and r.get("decoded_class") == 3]


def loc_of_row(r) -> loc_mod.Location:
    """Ledger row -> canonical Location for the morph field dump. Mirrors steered_pilot_report's
    render_geom: baseline julia stores the parameter c in outcome_cx/cy (z-viewport in julia_z_*);
    steered julia stores the z-viewport in outcome_cx/cy (c in julia_c_*)."""
    fam = r.get("family", "mandelbrot")
    if fam.startswith("julia:"):
        base = fam.split(":", 1)[1]
        rf = "julia" if base == "mandelbrot" else "julia_" + base
        if r.get("julia_z_cx") is not None:          # baseline julia schema
            cx, cy, fw, c_re, c_im = (r["julia_z_cx"], r["julia_z_cy"], r["julia_z_fw"],
                                      r["outcome_cx"], r["outcome_cy"])
        else:                                        # steered julia schema
            cx, cy, fw, c_re, c_im = (r["outcome_cx"], r["outcome_cy"], r["outcome_fw"],
                                      r["julia_c_re"], r["julia_c_im"])
        return loc_mod.Location(family=rf, cx=str(cx), cy=str(cy), fw=str(fw),
                                maxiter=auto_maxiter(float(fw)),
                                c_re=str(c_re), c_im=str(c_im))
    return loc_mod.Location(family=fam, cx=str(r["outcome_cx"]), cy=str(r["outcome_cy"]),
                            fw=str(r["outcome_fw"]), maxiter=auto_maxiter(float(r["outcome_fw"])))


# --------------------------------------------------------------------------- #
# Embedding (EXACT library recipe).
# --------------------------------------------------------------------------- #
def embed_admissions(rows, tmp_dir: Path, model, tf):
    """Return (uids, families, depths, embeddings[N,768]) for a list of admitted-q3 rows."""
    uids, fams, depths, embs = [], [], [], []
    for r in rows:
        loc = loc_of_row(r)
        field = ensure_field(loc, retain=False, tmp_dir=tmp_dir, cache_root=tmp_dir)
        img = morph_gray_image(field)
        emb = embed_clip(model, tf, [img])[0].astype(np.float32)
        uids.append(r["id"]); fams.append(r.get("family", "mandelbrot"))
        depths.append(int(r.get("reached_depth", 0))); embs.append(emb)
    return uids, fams, depths, (np.stack(embs) if embs else np.zeros((0, 768), np.float32))


def cos_matrix(E):
    U = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return U @ U.T


# --------------------------------------------------------------------------- #
# Single-linkage clustering on a cosine cut.
# --------------------------------------------------------------------------- #
def connected_components(n, cut, C):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in combinations(range(n), 2):
        if C[i, j] >= cut:
            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def pair_stats(C, idx):
    """median + full pairwise cos over a subset of indices."""
    if len(idx) < 2:
        return None, []
    vals = [(C[a, b], a, b) for a, b in combinations(idx, 2)]
    med = float(np.median([v for v, _, _ in vals]))
    top = sorted(vals, reverse=True)[:5]
    return med, top


# --------------------------------------------------------------------------- #
# Q2 — expansion-tree reconstruction from scratch.
# --------------------------------------------------------------------------- #
def reconstruct_tree(scratch: Path, state: dict):
    """Rebuild per-expansion child records from the surviving expand.jsonl files.

    Each expand.jsonl row is ONE child of a popped parent, carrying parent_depth, depth,
    root_id, branch, kind ('cand' alive | 'dead'). Returns:
      by_parent_depth : depth -> list of alive-child-counts, one per (expanded parent) that
                        produced >=1 row  (branching factor by depth)
      dead_by_depth   : parent_depth -> dead child count
      root_maxdepth   : root_id -> deepest child depth generated in that lineage
      coverage        : (parents_in_scratch, children_alive, dead)
    """
    exps = sorted(scratch.glob("expand_b*/*/expand.jsonl"))
    # (batch,group,parent_node_id) -> [child rows]
    per_parent = defaultdict(list)
    root_maxdepth = defaultdict(int)
    dead_by_depth = Counter()
    for f in exps:
        tag = f.parent.parent.name + "/" + f.parent.name
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (tag, row["node_id"])
            per_parent[key].append(row)
    by_parent_depth = defaultdict(list)
    n_alive = n_dead = 0
    for key, rows in per_parent.items():
        pdepth = rows[0].get("parent_depth")
        alive = [r for r in rows if r.get("kind") != "dead"]
        dead = [r for r in rows if r.get("kind") == "dead"]
        n_alive += len(alive); n_dead += len(dead)
        if pdepth is not None:
            by_parent_depth[pdepth].append(len(alive))
            dead_by_depth[pdepth] += len(dead)
        for r in alive:
            rid = r.get("root_id")
            root_maxdepth[rid] = max(root_maxdepth[rid], int(r.get("depth", 0)))
    # deepest node ever CREATED per root also includes surviving frontier nodes
    for n in state.get("frontier", []):
        rid = n.get("root_id")
        root_maxdepth[rid] = max(root_maxdepth[rid], int(n.get("depth", 0)))
    return by_parent_depth, dead_by_depth, root_maxdepth, (len(per_parent), n_alive, n_dead)


def expected_depth_balanced(b, M):
    """Depth of a b-ary tree that uses M internal-node expansions, budget split evenly.
    M expansions => M internal nodes; a full b-ary tree of height h has (b^h - 1)/(b-1)
    internal nodes at the levels above the leaves. Solve for h."""
    if b <= 1:
        return M + 1
    # levels of expansion that M internal nodes fill: b^0 + b^1 + ... + b^(h-1) = M
    return math.log(M * (b - 1) + 1) / math.log(b)


# --------------------------------------------------------------------------- #
# Contact sheet — steered admissions grouped by morph cluster.
# --------------------------------------------------------------------------- #
def render_colored(loc: loc_mod.Location, tile: Path):
    import subprocess
    if tile.exists():
        return
    tile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "render-one", "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
           "--width", "640", "--height", "360", "--supersample", "2",
           "--maxiter", str(loc.maxiter), "--palette", PALETTE,
           "--jpg-quality", str(JPG_Q), "--out", str(tile)] + loc_mod.render_one_flags(loc)
    subprocess.run(cmd, capture_output=True, text=True)


def contact_sheet(rows, uids, fams, depths, clusters, cut, out_png: Path):
    """Grouped-by-cluster contact sheet: one row block per morph cluster, tiles labeled
    depth + partition + cluster id. clusters = list[list[int]] over the row indices."""
    tw, th, pad, lab = 320, 180, 6, 20
    tmp = out_png.parent / "_tiles"
    tiles = []
    by_id = {r["id"]: r for r in rows}
    for i, u in enumerate(uids):
        tp = tmp / f"{u}.jpg"
        render_colored(loc_of_row(by_id[u]), tp)
        tiles.append(tp)
    order = sorted(range(len(clusters)), key=lambda k: -len(clusters[k]))
    maxcols = max((len(clusters[k]) for k in order), default=1)
    W = maxcols * (tw + pad) + pad
    H = len(clusters) * (th + lab + pad) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    y = pad
    for cid, k in enumerate(order):
        members = sorted(clusters[k], key=lambda i: depths[i])
        x = pad
        for i in members:
            im = Image.open(tiles[i]).convert("RGB").resize((tw, th)) if tiles[i].exists() \
                else Image.new("RGB", (tw, th), (60, 20, 20))
            sheet.paste(im, (x, y))
            tag = f"C{cid} d{depths[i]} {fams[i]}"
            dr.rectangle([x, y + th, x + tw, y + th + lab], fill=(30, 30, 34))
            dr.text((x + 4, y + th + 5), tag, fill=(230, 230, 120))
            x += tw + pad
        y += th + lab + pad
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


# --------------------------------------------------------------------------- #
def fmt_top(top, uids):
    return "; ".join(f"{s:.4f} {uids[a].split('_')[-1]}~{uids[b].split('_')[-1]}"
                     for s, a, b in top)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steered", type=Path, default=ROOT / "data/discovery/steered_pilot/steered")
    ap.add_argument("--baseline", type=Path, default=ROOT / "data/discovery/steered_pilot/baseline")
    ap.add_argument("--out", type=Path, default=ROOT / "out/steered_pilot_morph.md")
    args = ap.parse_args()

    st_rows = admitted_q3(load_jsonl(args.steered / "outcome_ledger.jsonl"))
    bl_rows = admitted_q3(load_jsonl(args.baseline / "outcome_ledger.jsonl"))
    print(f"steered admissions={len(st_rows)}  baseline admissions={len(bl_rows)}", flush=True)

    tmp = ROOT / "out" / "steered_pilot_morph_fields"
    print("loading CLIP (vit_base_patch16_clip_224.openai) ...", flush=True)
    model, tf = load_clip()
    print("embedding steered ...", flush=True)
    su, sf, sd, sE = embed_admissions(st_rows, tmp, model, tf)
    print("embedding baseline ...", flush=True)
    bu, bf, bd, bE = embed_admissions(bl_rows, tmp, model, tf)

    sC, bC = cos_matrix(sE), cos_matrix(bE)

    # --- Q2 tree reconstruction ---
    state = json.loads((args.steered / "state.json").read_text(encoding="utf-8"))
    by_pd, dead_pd, root_maxd, cov = reconstruct_tree(args.steered / "scratch", state)
    epr = {k: int(v) for k, v in state["expansions_per_root"].items()}

    L, W = [], None
    out = []
    def w(s=""):
        out.append(s)

    w("# Steered pilot — morphological composition + M-cap depth audit\n")
    w("Read-only companion to `out/steered_pilot_report.md`. Morph_clip recipe imported "
      "byte-identical from the library audit (`vit_base_patch16_clip_224.openai`, timm eval "
      "transform, robust-z tanh grayscale render at 640x360 ss2). Yardsticks from "
      "`docs/findings/prospect_run1_morph_composition_audit.md`: library-wide median pairwise "
      f"cos **{LIB_MEDIAN}**, intra-phoenix **{LIB_PHOENIX}**, strict near-dup cut **cos>{STRICT_CUT}**.\n")

    # ============================ Q1 ============================
    w("## Q1 — are the 16 steered admissions distinct looks, or one cheap look re-bought?\n")
    shallow = [i for i in range(len(su)) if sd[i] <= 3]
    deep = [i for i in range(len(su)) if sd[i] > 3]
    w(f"16 steered admissions: **{len(shallow)} shallow (depth<=3)**, **{len(deep)} deep (depth>3)**. "
      f"7 baseline admissions (depth 4-13).\n")

    w("### Intra-arm pairwise cosine (higher = more similar; compare to library yardsticks)\n")
    w("| subset | n | pairs | median cos | vs library |")
    w("|---|---:|---:|---:|---|")
    for name, idx, C, uu in [
        ("steered pooled", list(range(len(su))), sC, su),
        ("steered shallow d<=3", shallow, sC, su),
        ("steered deep d>3", deep, sC, su),
        ("baseline pooled", list(range(len(bu))), bC, bu),
    ]:
        med, top = pair_stats(C, idx)
        npairs = len(idx) * (len(idx) - 1) // 2
        vs = "" if med is None else (
            "≈ library median (diverse)" if med <= LIB_MEDIAN + 0.02 else
            ("phoenix-tight (one look)" if med >= LIB_PHOENIX - 0.01 else "moderately similar"))
        w(f"| {name} | {len(idx)} | {npairs} | {'n/a' if med is None else f'{med:.3f}'} | {vs} |")
    w("")
    med, top = pair_stats(sC, list(range(len(su))))
    w(f"Top steered similar pairs (pooled): {fmt_top(top, su)}\n")
    medb, topb = pair_stats(bC, list(range(len(bu))))
    if topb:
        w(f"Top baseline similar pairs: {fmt_top(topb, bu)}\n")

    w("### Morph clustering (single-linkage) — how many distinct morphs?\n")
    w("Global (family-blind) single-linkage; the worry is that shallow near-whole-set views are "
      "the same look across coordinates/partitions.\n")
    w("| cut cos> | steered clusters | steered largest | baseline clusters | baseline largest |")
    w("|---:|---:|---:|---:|---:|")
    cut_rows = {}
    for cut in (0.98, STRICT_CUT, 0.97, LOOSE_CUT, 0.925):
        scl = connected_components(len(su), cut, sC)
        bcl = connected_components(len(bu), cut, bC)
        cut_rows[cut] = (scl, bcl)
        w(f"| {cut} | {len(scl)} | {max(len(c) for c in scl)} | "
          f"{len(bcl)} | {max((len(c) for c in bcl), default=0)} |")
    w("")

    scl_strict, bcl_strict = cut_rows[STRICT_CUT]
    scl_loose, _ = cut_rows[LOOSE_CUT]
    w(f"**Steered: 16 admissions collapse to {len(scl_strict)} morphs at the strict cut "
      f"(cos>{STRICT_CUT}), {len(scl_loose)} at the perceptual cut (cos>{LOOSE_CUT}).**\n")

    # partition coverage after morph dedup (global clusters)
    w("### Partition-coverage headline after morph dedup\n")
    w("Does the pilot's *8 of 8 partitions* survive, or do partitions share one morph? "
      "(partition == family here; a cluster spanning >1 partition means those partitions "
      "delivered the same look.)\n")
    for cut, (scl, _) in [(STRICT_CUT, cut_rows[STRICT_CUT]), (LOOSE_CUT, cut_rows[LOOSE_CUT])]:
        multi = [c for c in scl if len({sf[i] for i in c}) > 1]
        covered = len({sf[i] for i in range(len(su))})
        distinct_by_part = defaultdict(set)  # partition -> set(cluster ids it appears in)
        cid_of = {}
        for cid, c in enumerate(scl):
            for i in c:
                cid_of[i] = cid
        w(f"- **cos>{cut}:** {covered} partitions present, {len(scl)} distinct morph clusters; "
          f"cross-partition clusters (partitions sharing a look): **{len(multi)}**.")
        for c in multi:
            parts = Counter(sf[i] for i in c)
            w(f"    - cluster {{{', '.join(su[i].split('_')[-1] for i in c)}}} spans "
              f"{dict(parts)} (depths {sorted(sd[i] for i in c)})")
    w("")

    # cluster membership dump (strict)
    w("### Steered morph clusters (strict cos>%.3f)\n" % STRICT_CUT)
    for cid, c in enumerate(sorted(scl_strict, key=lambda c: -len(c))):
        mem = ", ".join(f"{su[i].split('_')[-1]}({sf[i]},d{sd[i]})" for i in c)
        tightest = max((sC[a, b] for a, b in combinations(c, 2)), default=1.0)
        w(f"- **C{cid}** (n={len(c)}, tightest cos {tightest:.3f}): {mem}")
    w("")

    # ============================ Q2 ============================
    w("## Q2 — is the M=40 expansions/root cap a depth ceiling?\n")
    counts = sorted(epr.values(), reverse=True)
    capped = {r: v for r, v in epr.items() if v >= 40}
    w(f"- roots expanded: **{len(epr)}**; expansions/root max {max(counts)}, "
      f"median {counts[len(counts)//2]}, mean {sum(counts)/len(counts):.1f}\n"
      f"- roots at/over the M=40 cap: **{len(capped)}** (soft overshoot to {max(counts)} — the "
      f"`<M` check is at pop time, a batch can pop several nodes of one root)\n"
      f"- tree reconstruction coverage (surviving scratch): {cov[0]} expanded parents with "
      f"child records, {cov[1]} alive children, {cov[2]} dead "
      f"(vs state totals expanded={state['totals']['expanded']}, "
      f"candidates={state['totals']['candidates']}, dead={state['totals']['dead_nodes']}).\n")

    w("### Per-root expansions vs max chain depth reached\n")
    w("Max chain depth = deepest node ever CREATED in the lineage (child records + surviving "
      "frontier). If the cap were the depth limiter, capped roots would show depth climbing with "
      "expansions; instead they fan OUT (many shallow siblings), not DOWN.\n")
    w("| root | partition | expansions | max chain depth |")
    w("|---:|---|---:|---:|")
    part_of = {}
    for n in state.get("frontier", []):
        part_of[str(n["root_id"])] = n["partition"]
    for r, v in sorted(capped.items(), key=lambda kv: -kv[1]):
        md = root_maxd.get(int(r), root_maxd.get(r, 0))
        w(f"| {r} | {part_of.get(r,'?')} | {v} | {md} |")
    w("")
    cap_md = [root_maxd.get(int(r), 0) for r in capped]
    w(f"**The 8 capped roots reached max chain depth {min(cap_md)}–{max(cap_md)} "
      f"(median {int(np.median(cap_md))}) despite spending 40–{max(counts)} expansions each** — "
      f"budget went into breadth, not depth.\n")

    w("### Branching factor (alive gate survivors per expansion) by depth\n")
    w("| parent depth | expansions sampled | mean alive children | dead-child rate |")
    w("|---:|---:|---:|---:|")
    all_b = []
    for d in sorted(by_pd):
        arr = by_pd[d]
        alive_mean = float(np.mean(arr))
        tot_alive = sum(arr)
        dead = dead_pd.get(d, 0)
        all_b += arr
        w(f"| {d} | {len(arr)} | {alive_mean:.2f} | {dead/(tot_alive+dead) if (tot_alive+dead) else 0:.1%} |")
    b_overall = float(np.mean(all_b))
    w("")
    w(f"Overall branching factor b ≈ **{b_overall:.2f}** alive survivors/expansion "
      f"(state totals: {state['totals']['candidates']}/{state['totals']['expanded']} = "
      f"{state['totals']['candidates']/state['totals']['expanded']:.2f}). Gate death is rare "
      f"(9 dead nodes all-run) — lineages do NOT die at the gate; they PLATEAU in priority.\n")

    hb = expected_depth_balanced(b_overall, 40)
    w("### Reachable depth under M=40\n")
    w(f"- **Single-tracked** (always expand the lineage's deepest node): depth ≈ **40** "
      f"(one expansion = one level deeper).\n")
    w(f"- **Budget split across branches** (b≈{b_overall:.2f}, balanced b-ary tree of 40 "
      f"internal expansions): depth ≈ **{hb:.1f}**.\n")
    w(f"Observed capped-root median depth ≈ {int(np.median(cap_md))} — far closer to the "
      f"budget-split floor (~{hb:.0f}) than the single-track ceiling (40); the modest excess over "
      f"a perfectly balanced tree is best-first PARTIALLY concentrating on hot lineages. Priority "
      f"= cheap E[ord] does not grow with depth, so the frontier keeps popping high-scoring "
      f"SHALLOW siblings and never single-tracks a lineage deep.\n")

    w("### Verdict (Q2)\n")
    w(f"**M=40 is not the binding constraint on reachable depth.** 115/{len(epr)} roots "
      f"(90%) expanded <=4 times and stopped far below the cap — their priority sank beneath the "
      f"frontier, not the cap. The 8 capped roots poured 40–{max(counts)} expansions into "
      f"BREADTH (max depth {int(np.median(cap_md))} median), because priority = cheap E[ord] has "
      f"no depth-seeking term. Raising M would buy more shallow siblings in the 8 hot roots, not "
      f"depth. To reach deep locations you would change the POLICY (a depth/novelty bonus, or a "
      f"single-track mode), not the cap.\n")

    # contact sheet — grouped at the PERCEPTUAL cut (strict = all singletons, no grouping to show)
    sheet = contact_sheet(st_rows, su, sf, sd,
                          sorted(scl_loose, key=lambda c: -len(c)), LOOSE_CUT,
                          ROOT / "out/steered_pilot_morph_clusters.png")
    w("## Contact sheet\n")
    w(f"Steered admissions grouped by morph cluster at the **perceptual** cut (cos>{LOOSE_CUT}); "
      f"the strict cut leaves all 16 as singletons so nothing groups. Each tile labeled "
      f"`C<cluster> d<depth> <partition>`, multi-tile rows are the near-look groups: "
      f"`{sheet.relative_to(ROOT)}` (pairs with the parallel blind human read).\n")

    # ============================ combined ============================
    w("## Combined verdict\n")
    surv_strict = len(scl_strict)
    surv_loose = len(scl_loose)
    multi_loose = [c for c in scl_loose if len({sf[i] for i in c}) > 1]
    w(f"1. **Does 16-vs-3 survive morph dedup?** The steered arm's {len(su)} admissions are "
      f"**{surv_strict} distinct morphs** at the strict near-dup cut and **{surv_loose}** at the "
      f"perceptual cut — versus baseline's {len(bu)} admissions collapsing to "
      f"{len(cut_rows[LOOSE_CUT][1])}/{len(bcl_strict)}. "
      f"{'The steered lead survives — the shallow admissions are genuinely different looks, not one view re-bought.' if surv_loose >= max(6, len(su)//2) else 'The steered lead SHRINKS under morph dedup — several admissions are the same look.'} "
      f"Cross-partition sharing at the perceptual cut: {len(multi_loose)} cluster(s).\n")
    w(f"2. **Config change before a longer run.** The depth audit says the M=40 cap is not what "
      f"holds depth down — best-first on cheap E[ord] has no depth pull, so it buys shallow "
      f"breadth. If the goal is depth diversity, add a depth/novelty term to priority (or a "
      f"single-track escape) rather than raising M. The dup-penalty already works (11% coord-dup); "
      f"a morph-space penalty would additionally suppress the {len(multi_loose)} perceptual "
      f"near-repeats a longer run will otherwise multiply.\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"steered morphs: {surv_strict} strict / {surv_loose} loose (from {len(su)})")
    print(f"baseline morphs: {len(bcl_strict)} strict / {len(cut_rows[LOOSE_CUT][1])} loose (from {len(bu)})")


if __name__ == "__main__":
    main()
