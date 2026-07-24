# Steered pilot — morphological composition + M-cap depth audit

Read-only companion to `out/steered_pilot_report.md`. Morph_clip recipe imported byte-identical from the library audit (`vit_base_patch16_clip_224.openai`, timm eval transform, robust-z tanh grayscale render at 640x360 ss2). Yardsticks from `docs/findings/prospect_run1_morph_composition_audit.md`: library-wide median pairwise cos **0.851**, intra-phoenix **0.938**, strict near-dup cut **cos>0.974**.

## Q1 — are the 16 steered admissions distinct looks, or one cheap look re-bought?

16 steered admissions: **13 shallow (depth<=3)**, **3 deep (depth>3)**. 7 baseline admissions (depth 4-13).

### Intra-arm pairwise cosine (higher = more similar; compare to library yardsticks)

| subset | n | pairs | median cos | vs library |
|---|---:|---:|---:|---|
| steered pooled | 16 | 120 | 0.882 | moderately similar |
| steered shallow d<=3 | 13 | 78 | 0.884 | moderately similar |
| steered deep d>3 | 3 | 3 | 0.877 | moderately similar |
| baseline pooled | 7 | 21 | 0.849 | ≈ library median (diverse) |

Top steered similar pairs (pooled): 0.9696 000004~000006; 0.9608 000016~000017; 0.9578 000005~000016; 0.9560 000005~000017; 0.9478 000001~000002

Top baseline similar pairs: 0.9613 000005~000012; 0.9548 000001~000012; 0.9356 000006~000007; 0.9185 000001~000005; 0.9057 000007~000012

### Morph clustering (single-linkage) — how many distinct morphs?

Global (family-blind) single-linkage; the worry is that shallow near-whole-set views are the same look across coordinates/partitions.

| cut cos> | steered clusters | steered largest | baseline clusters | baseline largest |
|---:|---:|---:|---:|---:|
| 0.98 | 16 | 1 | 7 | 1 |
| 0.974 | 16 | 1 | 7 | 1 |
| 0.97 | 16 | 1 | 7 | 1 |
| 0.95 | 13 | 3 | 5 | 3 |
| 0.925 | 6 | 5 | 4 | 3 |

**Steered: 16 admissions collapse to 16 morphs at the strict cut (cos>0.974), 13 at the perceptual cut (cos>0.95).**

### Partition-coverage headline after morph dedup

Does the pilot's *8 of 8 partitions* survive, or do partitions share one morph? (partition == family here; a cluster spanning >1 partition means those partitions delivered the same look.)

- **cos>0.974:** 8 partitions present, 16 distinct morph clusters; cross-partition clusters (partitions sharing a look): **0**.
- **cos>0.95:** 8 partitions present, 13 distinct morph clusters; cross-partition clusters (partitions sharing a look): **1**.
    - cluster {000005, 000016, 000017} spans {'julia:multibrot3': 1, 'julia:mandelbrot': 2} (depths [2, 3, 4])

### Steered morph clusters (strict cos>0.974)

- **C0** (n=1, tightest cos 1.000): 000000(mandelbrot,d2)
- **C1** (n=1, tightest cos 1.000): 000001(multibrot5,d3)
- **C2** (n=1, tightest cos 1.000): 000002(multibrot5,d3)
- **C3** (n=1, tightest cos 1.000): 000003(multibrot3,d2)
- **C4** (n=1, tightest cos 1.000): 000004(julia:multibrot5,d2)
- **C5** (n=1, tightest cos 1.000): 000005(julia:multibrot3,d2)
- **C6** (n=1, tightest cos 1.000): 000006(julia:multibrot5,d3)
- **C7** (n=1, tightest cos 1.000): 000007(multibrot5,d3)
- **C8** (n=1, tightest cos 1.000): 000008(multibrot5,d3)
- **C9** (n=1, tightest cos 1.000): 000009(julia:mandelbrot,d3)
- **C10** (n=1, tightest cos 1.000): 000011(multibrot4,d5)
- **C11** (n=1, tightest cos 1.000): 000012(julia:multibrot4,d2)
- **C12** (n=1, tightest cos 1.000): 000013(multibrot3,d4)
- **C13** (n=1, tightest cos 1.000): 000014(mandelbrot,d3)
- **C14** (n=1, tightest cos 1.000): 000016(julia:mandelbrot,d4)
- **C15** (n=1, tightest cos 1.000): 000017(julia:mandelbrot,d3)

## Q2 — is the M=40 expansions/root cap a depth ceiling?

- roots expanded: **127**; expansions/root max 66, median 1, mean 4.5
- roots at/over the M=40 cap: **8** (soft overshoot to 66 — the `<M` check is at pop time, a batch can pop several nodes of one root)
- tree reconstruction coverage (surviving scratch): 475 expanded parents with child records, 1672 alive children, 9 dead (vs state totals expanded=576, candidates=2064, dead=9).

### Per-root expansions vs max chain depth reached

Max chain depth = deepest node ever CREATED in the lineage (child records + surviving frontier). If the cap were the depth limiter, capped roots would show depth climbing with expansions; instead they fan OUT (many shallow siblings), not DOWN.

| root | partition | expansions | max chain depth |
|---:|---|---:|---:|
| 1316 | julia:multibrot3 | 66 | 8 |
| 579 | julia:multibrot5 | 59 | 8 |
| 578 | julia:multibrot5 | 54 | 6 |
| 393 | julia:multibrot3 | 52 | 7 |
| 63 | multibrot4 | 48 | 11 |
| 392 | julia:multibrot5 | 45 | 5 |
| 953 | julia:multibrot4 | 41 | 5 |
| 391 | julia:multibrot5 | 40 | 8 |

**The 8 capped roots reached max chain depth 5–11 (median 7) despite spending 40–66 expansions each** — budget went into breadth, not depth.

### Branching factor (alive gate survivors per expansion) by depth

| parent depth | expansions sampled | mean alive children | dead-child rate |
|---:|---:|---:|---:|
| 1 | 125 | 2.74 | 2.0% |
| 2 | 48 | 3.42 | 0.6% |
| 3 | 35 | 3.83 | 0.0% |
| 4 | 77 | 3.88 | 0.3% |
| 5 | 73 | 3.88 | 0.0% |
| 6 | 63 | 3.81 | 0.0% |
| 7 | 32 | 3.81 | 0.0% |
| 8 | 11 | 4.00 | 0.0% |
| 9 | 8 | 4.00 | 0.0% |
| 10 | 3 | 4.00 | 0.0% |

Overall branching factor b ≈ **3.52** alive survivors/expansion (state totals: 2064/576 = 3.58). Gate death is rare (9 dead nodes all-run) — lineages do NOT die at the gate; they PLATEAU in priority.

### Reachable depth under M=40

- **Single-tracked** (always expand the lineage's deepest node): depth ≈ **40** (one expansion = one level deeper).

- **Budget split across branches** (b≈3.52, balanced b-ary tree of 40 internal expansions): depth ≈ **3.7**.

Observed capped-root median depth ≈ 7 — far closer to the budget-split floor (~4) than the single-track ceiling (40); the modest excess over a perfectly balanced tree is best-first PARTIALLY concentrating on hot lineages. Priority = cheap E[ord] does not grow with depth, so the frontier keeps popping high-scoring SHALLOW siblings and never single-tracks a lineage deep.

### Verdict (Q2)

**M=40 is not the binding constraint on reachable depth.** 115/127 roots (90%) expanded <=4 times and stopped far below the cap — their priority sank beneath the frontier, not the cap. The 8 capped roots poured 40–66 expansions into BREADTH (max depth 7 median), because priority = cheap E[ord] has no depth-seeking term. Raising M would buy more shallow siblings in the 8 hot roots, not depth. To reach deep locations you would change the POLICY (a depth/novelty bonus, or a single-track mode), not the cap.

## Contact sheet

Steered admissions grouped by morph cluster at the **perceptual** cut (cos>0.95); the strict cut leaves all 16 as singletons so nothing groups. Each tile labeled `C<cluster> d<depth> <partition>`, multi-tile rows are the near-look groups: `out\steered_pilot_morph_clusters.png` (pairs with the parallel blind human read).

## Combined verdict

1. **Does 16-vs-3 survive morph dedup?** The steered arm's 16 admissions are **16 distinct morphs** at the strict near-dup cut and **13** at the perceptual cut — versus baseline's 7 admissions collapsing to 5/7. The steered lead survives — the shallow admissions are genuinely different looks, not one view re-bought. Cross-partition sharing at the perceptual cut: 1 cluster(s).

2. **Config change before a longer run.** The depth audit says the M=40 cap is not what holds depth down — best-first on cheap E[ord] has no depth pull, so it buys shallow breadth. If the goal is depth diversity, add a depth/novelty term to priority (or a single-track escape) rather than raising M. The dup-penalty already works (11% coord-dup); a morph-space penalty would additionally suppress the 1 perceptual near-repeats a longer run will otherwise multiply.
