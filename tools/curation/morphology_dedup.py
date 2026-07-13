"""Morphology dedup — collapse near-identical skeletons the coordinate-dedup can't catch.

A conservative, NON-DESTRUCTIVE curation pass over an emission corpus. The coordinate
`same_fractal` identity (emission_selector) already collapses same-location recolors; it
CANNOT see two coordinate-distinct emissions that render the same skeleton (the
"good morphology is intrinsically narrow" mode measured in
`scratchpad/visual_dup/FINDINGS.md`). This pass catches only the tightest tier of that:
palette-blind CLIP morphology similarity on the canonical grayscale renders, thresholded
so ONLY the ~0.978 tier trips.

Design (calibrated by eye on the overnight_20260713 contact sheet):
  * WITHIN-FAMILY ONLY. Family is a deliberate MAP-Elites diversity axis — never collapse
    a julia spiral into a mandelbrot spiral even if CLIP says they match.
  * PHOENIX EXCLUDED. Its high self-similarity is variety-poverty (one morphotype, distinct
    viewports of the fixed Ushiki c/p), a separate share-limit question — not duplication.
  * CONSERVATIVE THRESHOLD (default 0.974). The MUST-collapse anchor (jm3 wfd_013_07 /
    wfd_010_08, CLIP 0.9776) trips; the MUST-stay-distinct 0.960-0.969 band (incl. all
    phoenix and the lower jm3 pairs, top = 0.9692) survives. The 0.9776->0.9692 gap is the
    margin.
  * COLLAPSE = keep the highest-head-`p_ge3` member of each above-threshold group; the rest
    drop. Single-linkage within family (transitive: a-b, b-c tight => {a,b,c} one group).

Non-destructive: reports the collapse set, a borderline band for eye-review, and the
resulting distinct count. It does NOT mutate any manifest, lock, or launch. Slots into
morning curation alongside `tools/mining/deploy_tail.py`.

The reference control band is cleaned to RECOLOR-ONLY (same geometry, different palette) —
the phoenix cross_cycle pairs are dropped from the control set because they are distinct
viewports of one Ushiki plane mis-grouped by the coordinate metric, not real dups.

Run (near-free, reuses the saved CLIP matrix — no re-render, no re-embed):
    uv run python -m tools.curation.morphology_dedup \
        --artifacts scratchpad/visual_dup --threshold 0.974
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD = 0.974
DEFAULT_EXCLUDE = ("phoenix",)
# Eye-review band floor: pairs in [BORDERLINE_LO, threshold) are surfaced, not decided.
BORDERLINE_LO = 0.969


@dataclass
class Corpus:
    """The distinct-fractal representatives of an emission corpus + their CLIP sim matrix.

    reps      : ordered list of representative uids (one per coordinate-distinct fractal).
    sim(a, b) : palette-blind CLIP cosine on the canonical grayscale renders.
    family    : uid -> family (MAP-Elites diversity axis; comparison is within-family).
    p_ge3     : uid -> head P(label>=3); the collapse tiebreak (keep the max).
    recolor_clusters : list of known same-geometry-different-palette uid groups (the
                       genuine control band; phoenix cross_cycle deliberately excluded).
    """

    reps: list[str]
    _sim: dict[tuple[str, str], float]
    family: dict[str, str]
    p_ge3: dict[str, float]
    fitness: dict[str, float]
    recolor_clusters: list[list[str]] = field(default_factory=list)

    def sim(self, a: str, b: str) -> float:
        return self._sim[(a, b)] if (a, b) in self._sim else self._sim[(b, a)]


@dataclass
class CollapseGroup:
    family: str
    kept: str
    dropped: list[str]
    members: list[str]
    max_sim: float  # tightest edge in the group
    edges: list[tuple[str, str, float]]


@dataclass
class DedupResult:
    threshold: float
    exclude_families: tuple[str, ...]
    n_reps_total: int
    n_reps_considered: int  # after excluding families
    collapse_groups: list[CollapseGroup]
    borderline_pairs: list[tuple[str, str, str, float]]  # family, a, b, sim
    n_dropped: int
    n_distinct_after: int  # over the FULL corpus (excluded families kept as-is)
    control_band: dict


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return [g for g in groups.values() if len(g) > 1]


def morphology_dedup(
    corpus: Corpus,
    threshold: float = DEFAULT_THRESHOLD,
    exclude_families: tuple[str, ...] = DEFAULT_EXCLUDE,
    borderline_lo: float = BORDERLINE_LO,
) -> DedupResult:
    """Conservative within-family morphology dedup. Pure; mutates nothing."""
    considered = [r for r in corpus.reps if corpus.family[r] not in exclude_families]

    # within-family pairs, ranked
    by_family: dict[str, list[str]] = {}
    for r in considered:
        by_family.setdefault(corpus.family[r], []).append(r)

    collapse_edges: dict[str, list[tuple[str, str]]] = {}
    all_edges: list[tuple[str, str, float]] = []  # (a, b, sim) within-family
    borderline: list[tuple[str, str, str, float]] = []
    for fam, members in by_family.items():
        for a, b in combinations(members, 2):
            s = corpus.sim(a, b)
            all_edges.append((a, b, s))
            if s >= threshold:
                collapse_edges.setdefault(fam, []).append((a, b))
            elif s >= borderline_lo:
                borderline.append((fam, a, b, s))

    groups: list[CollapseGroup] = []
    n_dropped = 0
    for fam, edges in collapse_edges.items():
        nodes = sorted({n for e in edges for n in e})
        for comp in _connected_components(nodes, edges):
            kept = max(comp, key=lambda u: (corpus.p_ge3[u], corpus.fitness[u]))
            dropped = sorted(u for u in comp if u != kept)
            comp_edges = [
                (a, b, corpus.sim(a, b))
                for a, b in combinations(sorted(comp), 2)
                if corpus.sim(a, b) >= threshold
            ]
            groups.append(CollapseGroup(
                family=fam, kept=kept, dropped=dropped, members=sorted(comp),
                max_sim=max(corpus.sim(a, b) for a, b in edges
                            if a in comp and b in comp),
                edges=comp_edges,
            ))
            n_dropped += len(dropped)

    groups.sort(key=lambda g: -g.max_sim)
    borderline.sort(key=lambda x: -x[3])

    return DedupResult(
        threshold=threshold,
        exclude_families=exclude_families,
        n_reps_total=len(corpus.reps),
        n_reps_considered=len(considered),
        collapse_groups=groups,
        borderline_pairs=borderline,
        n_dropped=n_dropped,
        n_distinct_after=len(corpus.reps) - n_dropped,
        control_band=_control_band(corpus),
    )


def _control_band(corpus: Corpus) -> dict:
    """Recolor-only control band: within-cluster pairwise CLIP over the known same-geometry-
    different-palette groups. Phoenix cross_cycle is NOT in recolor_clusters by construction.
    """
    sims = []
    for grp in corpus.recolor_clusters:
        for a, b in combinations(grp, 2):
            sims.append(corpus.sim(a, b))
    if not sims:
        return {}
    arr = np.array(sims)
    # the single loose framing outlier (FINDINGS: wide-framing julia merge) drags the min
    loose = float(arr.min())
    bulk = arr[arr >= 0.9]
    return dict(
        n_pairs=len(sims),
        loose_min=loose,
        bulk_min=float(bulk.min()) if len(bulk) else float("nan"),
        p25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        max=float(arr.max()),
        note="recolor-only; phoenix cross_cycle dropped; loose_min is the one wide-framing "
             "julia merge (see FINDINGS).",
    )


# --------------------------------------------------------------------------- #
# Loader for the near-free path: reuse the already-saved CLIP matrix + cluster structure.
# --------------------------------------------------------------------------- #
def load_corpus_from_artifacts(artifacts: Path) -> Corpus:
    """Build a Corpus from `scratchpad/visual_dup`-style artifacts.

    Reads `clusters.json` (62->47 structure + full per-row manifest fields incl. p_ge3)
    and `out/sim_matrices.npz` (the saved CLIP cosine matrix). No GPU, no re-embed.
    """
    data = json.loads((artifacts / "clusters.json").read_text())
    z = np.load(artifacts / "out" / "sim_matrices.npz", allow_pickle=True)
    uids = list(z["uids"])
    clip = z["clip"]
    idx = {u: i for i, u in enumerate(uids)}

    rows = data["rows"]
    clusters = data["clusters"]
    reps = [c["rep"] for c in clusters]
    family = {c["rep"]: c["family"] for c in clusters}
    p_ge3 = {c["rep"]: float(rows[c["rep"]]["p_ge3"]) for c in clusters}
    fitness = {c["rep"]: float(rows[c["rep"]]["fitness"]) for c in clusters}

    # genuine control band = recolor clusters only (drop phoenix cross_cycle)
    recolor_clusters = [c["members"] for c in clusters
                        if c.get("kind") == "recolor" and c["size"] > 1]

    # sparse sim dict over the reps (both orders not needed; sim() falls back)
    sim: dict[tuple[str, str], float] = {}
    all_uids = set(reps) | {u for grp in recolor_clusters for u in grp}
    all_uids = [u for u in all_uids if u in idx]
    for a, b in combinations(all_uids, 2):
        sim[(a, b)] = float(clip[idx[a], idx[b]])

    return Corpus(reps=reps, _sim=sim, family=family, p_ge3=p_ge3,
                  fitness=fitness, recolor_clusters=recolor_clusters)


def result_to_dict(res: DedupResult) -> dict:
    return dict(
        threshold=res.threshold,
        exclude_families=list(res.exclude_families),
        borderline_lo=BORDERLINE_LO,
        n_reps_total=res.n_reps_total,
        n_reps_considered=res.n_reps_considered,
        n_dropped=res.n_dropped,
        n_distinct_after=res.n_distinct_after,
        control_band_recolor_only=res.control_band,
        collapse_groups=[dict(
            family=g.family, kept=g.kept, dropped=g.dropped, members=g.members,
            max_sim=g.max_sim,
            edges=[dict(a=a, b=b, sim=s) for a, b, s in g.edges],
        ) for g in res.collapse_groups],
        borderline_pairs=[dict(family=f, a=a, b=b, sim=s)
                          for f, a, b, s in res.borderline_pairs],
    )


def print_report(res: DedupResult, corpus: Corpus):
    W = 74
    print("=" * W)
    print(f"MORPHOLOGY DEDUP  (threshold={res.threshold}, exclude={list(res.exclude_families)})")
    print("=" * W)
    cb = res.control_band
    if cb:
        print("\nControl band (RECOLOR-ONLY reference, phoenix cross_cycle dropped):")
        print(f"  n_pairs={cb['n_pairs']}  loose_min={cb['loose_min']:.3f}  "
              f"bulk_min={cb['bulk_min']:.3f}  p25={cb['p25']:.3f}  "
              f"median={cb['median']:.3f}  max={cb['max']:.3f}")

    print(f"\nAUTO-COLLAPSE at >= {res.threshold} ({len(res.collapse_groups)} group(s), "
          f"{res.n_dropped} dropped):")
    if not res.collapse_groups:
        print("  (none)")
    for g in res.collapse_groups:
        print(f"  [{g.family}]  max_sim={g.max_sim:.4f}")
        print(f"     KEEP  {g.kept}  (p_ge3={corpus.p_ge3[g.kept]:.3f})")
        for d in g.dropped:
            print(f"     drop  {d}  (p_ge3={corpus.p_ge3[d]:.3f})")
        for a, b, s in g.edges:
            print(f"       edge {s:.4f}  {a} <-> {b}")

    print(f"\nBORDERLINE eye-review band [{BORDERLINE_LO}, {res.threshold}) "
          f"({len(res.borderline_pairs)} pair(s) — surfaced, NOT decided):")
    if not res.borderline_pairs:
        print("  (none)")
    for f, a, b, s in res.borderline_pairs:
        print(f"  {s:.4f}  [{f}]  {a} <-> {b}")

    print(f"\nDistinct corpus: {res.n_reps_total} -> {res.n_distinct_after}  "
          f"(-{res.n_dropped}).  NON-DESTRUCTIVE: no manifest mutated, no lock, no launch.")
    print("=" * W)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("scratchpad/visual_dup"),
                    help="dir with clusters.json + out/sim_matrices.npz")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON report path (default: <artifacts>/out/morphology_dedup.json)")
    args = ap.parse_args()

    corpus = load_corpus_from_artifacts(args.artifacts)
    res = morphology_dedup(corpus, threshold=args.threshold,
                           exclude_families=tuple(args.exclude))
    print_report(res, corpus)

    out = args.out or (args.artifacts / "out" / "morphology_dedup.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result_to_dict(res), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
