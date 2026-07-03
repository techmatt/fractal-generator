# Atlas v1 — round 2: descendability pre-screen + discovery-focused re-run

**Question.** Round 1 showed the θ̂ value map is real but left discovery unanswered:
its exploit win was **confounded** (injected seeds descended only ~44% vs the current
seeder's ~96%) and its explore arm was a dead 27-seed sliver. Round 2 fixes both — adds
a real descendability pre-screen so all arms are ~equally productive, and gives explore
a full-budget test. **The decisive metric is novel good outcomes** — good locations in
appearance-regions the current seeder never reaches — not mean k3.

**Design.** Three arms, 250 walks each, through the **identical** injected guided-descend
path (`--seed-list`, `--seed 0 --per-walk-rng`). Only the depth-1 seed source differs:

| arm | seed source | role |
|-----|-------------|------|
| 1 | current seeder — native draws (169 8k-field + 151 flat → 250) | baseline "same areas" |
| 2 | atlas **exploit** — pure high-conf θ̂ targeting (acq = conf·θ̂_norm, top-20% + FPS) | re-mining reference |
| 3 | atlas **explore** — pure low-conf/uncertainty (bottom-50% conf survivors + FPS) | the discovery arm |

Reward = **k3 best-over-walk** (v5 CORN E[ord] ∈ [0,2]) — the exact reward θ̂ was fit on.

## Build 1 — the descendability pre-screen (the round-2 methodology win)

The pre-screen the prompt described (128px black-cap → band → occupancy 0.321 on the
seed *frame*) turned out to be **wrong**: it rejects **97.5% of the native seeds** that
descend fine, because descendability is a property of the depth-2 *child*, not the wide
root frame — which is exactly why guided-descend **skips the occupancy floor at the
d1→d2 step**. So the faithful "would this seed survive step-1" screen is an actual
**depth-2 descent probe**: inject the candidate cloud as `--seed-list`, run `--depth-min
2 --depth-max 2 --per-walk-rng` at the efficient config, keep `reached_depth ≥ 2`. This
reuses the descent machinery verbatim (zero parity risk) and directly targets the
productivity metric. (The dead seed-frame `screen-seeds` Rust subcommand was removed.)

- Cloud 2500 → **1422 descendable (56.9%)**.
- **Frontier descendability (the key diagnostic):** the uncovered, **low-conf** boundary
  is **38.8%** descendable vs **75.0%** for the covered, high-conf region. The uncovered
  frontier is genuinely *harder* — the current seeder clusters for a reason — but ~39%
  descendable is **far from empty**: real new territory exists out there.

## Productivity — the round-1 confound is closed

| arm | prod % (reached ≥ 2) | mean reached |
|-----|----------------------|--------------|
| 1 current seeder | **95.6%** | 8.2 |
| 2 atlas exploit | 90.4% | 8.0 |
| 3 atlas explore | 88.8% | 7.6 |

Round-1 injected arms were **44%**; pre-screened, all three now sit at **89–96%**. The
`(exploit)v(current)` comparison is finally unconfounded. (The ~5–7% gap below native is
the expected rng-reshuffle: survivors descend at fresh walk indices, so the depth-2 step
draws a different best-of-4 than the probe did — descendability transfers strongly, not
perfectly.)

## Yield (secondary, now unconfounded)

| arm | mean | med | p90 | ≥1.0 | ≥1.4 | #good |
|-----|------|-----|-----|------|------|-------|
| 1 current seeder | 0.792 | 0.689 | 1.388 | 0.24 | 0.10 | 59 |
| 2 atlas exploit | **1.120** | **0.979** | **1.733** | **0.48** | **0.30** | **119** |
| 3 atlas explore | 0.791 | 0.667 | 1.350 | 0.19 | 0.09 | 48 |

- **Exploit, now unconfounded, cleanly beats the seeder:** Δmean **+0.328**, 2× the good
  outcomes (119 vs 59), 3× the strong tail (≥1.4: 0.30 vs 0.10). This is the clean
  end-to-end atlas-vs-baseline win round 1 could not show (its 44% productivity capped
  the good-count).
- **Explore ≈ current-seeder yield** (Δmean −0.002). Spreading into the low-conf frontier
  costs the yield exploit gains — consistent with the frontier being ~half as descendable
  and θ-agnostic by construction.

## Headline — novel good regions (appearance-regions arm-1 never makes)

Good outcomes (k3 ≥ 1.0) embedded via **v5 penultimate features** (1280-D); *novel* =
min cosine distance to **every** arm-1 good outcome > TAU = 0.01; *regions* = greedy-leader
near-dup survivors among the novel set.

| arm | #good | novel good | novel % | novel regions | med min-dist |
|-----|-------|-----------|---------|---------------|--------------|
| 2 exploit | 119 | 35 | 29% | **26** | 0.0059 |
| 3 explore | 48 | 9 | 19% | **8** | 0.0046 |

Round-1 exploit baseline was **5** novel_good. Both arms clear it — but **exploit
dominates explore here too** (26 vs 8 novel regions).

## Total coverage expansion (distinct good-regions)

| set | #good | distinct regions | Δ vs current |
|-----|-------|------------------|--------------|
| current (arm1) | 59 | 25 | — |
| current ∪ exploit | 178 | 59 | **+34** |
| current ∪ explore | 107 | 44 | +19 |
| current ∪ atlas | 226 | 70 | **+45** |

Ordering (exploit > explore) is **stable across TAU ∈ [0.005, 0.05]**. Seed-space
good-bin coverage is flat across arms (28 / 27 / 25 of 168 bins) — no arm collapses.

---

## Verdict

**The pre-screen closed the confound (the methodological result), and the value map —
now unconfounded — is a strong seeder.** Exploit lifts **both** yield (+0.33 mean, 2×
good) **and** coverage (+34 distinct regions) over the current seeder. Explore reaches
**modest genuinely-new territory** (8 novel regions, +19 coverage) but at
**current-seeder-level yield**.

On the prompt's two honest outcomes, the answer is **between them, leaning fractal-limited**:

- Discovery breadth is **partly fractal-limited**: the uncovered frontier is real
  (explore does find new descendable good regions) but **genuinely harder** (39% vs 75%
  descendable) and **lower-quality** (explore yield = current, exploit yield ≫). The good
  stuff concentrates where the seeder already looks; "avoiding the same areas" costs
  quality — as the prompt anticipated.
- The prompt warned against collapsing to pure exploit, but the data says the **value
  map is the lever, not spreading**: weight **exploit**, not explore.

**Two caveats keep this from being production-final:**

1. **Exploit's "novelty" is partly circular / sample-inflated.** "Novel vs arm-1" is
   measured against one 250-walk native sample (59 good); exploit produces 2× the good
   outcomes, so it lands outside that sample by count alone, and θ̂ was trained on native
   walks (it targets where the seeder already does well). This is *coverage*, not proven
   *unreachable-by-the-seeder* territory.
2. **The decisive test is still the non-circular `propose → relabel → refit` loop** —
   propose with the atlas, hand-label the outcomes blind, refit θ̂, and check it predicts
   *held-out* new regions. That is the next round, not this one.

**Recommendation:** promote the atlas **exploit** proposer (high-conf θ̂ targeting +
depth-2 pre-screen) as a seeding arm blended with the current seeder — it is a clean,
now-unconfounded yield+coverage win. Keep a small explore weight for breadth insurance,
not as the primary. Gate production wiring on the non-circular relabel→refit loop.

## Artifacts (`data/atlas/round2/`)

- `arm{1,2,3}_seeds.jsonl` — the three seed lists (arm 2/3 carry θ̂/conf/term provenance)
- `prescreen_meta.json` — pre-screen pass rate + frontier descendability
- `arm{1,2,3}_pool/` — full guided-descend pools (durable); `arm1_native_seedgen/` = the
  depth-1 native draws arm-1 was extracted from
- `arm{1,2,3}_table.jsonl` — k3 best-over-walk harvest (+ reframed geometry)
- `arm{1,2,3}_embed.npz` — v5 penultimate outcome embeddings + rewards + tags
- `round2_report.json` — machine-readable metrics
- `out/atlas/round2/embed_tiles/arm{1,2,3}/` — per-outcome best-frame tiles (eyeball check)
