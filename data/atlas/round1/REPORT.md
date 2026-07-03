# Atlas v1 — round 1: proposer + 3-arm acceptance test

**Question.** Does atlas-guided seeding produce higher-value, more diverse harvests
than the current clustered seeder — and if so, *where does the win come from*
(de-clustering, value-targeting, or genuine discovery)?

**Design.** Three arms, 250 injected walks each, run through the **identical**
injected guided-descend path (new `--seed-list` flag; same `--seed 0 --per-walk-rng`
→ byte-identical depth≥2 RNG stream). Only the depth-1 seed source differs:

| arm | seed source | role |
|-----|-------------|------|
| 1 | current seeder's own native depth-1 draws (134 8k-field + 116 flat) | baseline "bad proposer" |
| 2 | uniform-over-domain, FPS-spread | de-clustering control |
| 3 | atlas acquisition `a = conf·θ̂_norm + λ(1−conf)`, λ=0.5, threshold+FPS | value + uncertainty targeting |

Reward = **k3 best-over-walk** (v5 CORN E[ord] ∈ [0,2]) — the exact reward the atlas
was fit on. `(2)v(1)` = de-clustering benefit; `(3)v(2)` = whether θ̂ value-targeting
adds anything on top of spreading.

---

## Yield (per arm, matched N=250)

| arm | mean | med | p90 | ≥1.0 | ≥1.4 | reached | prod% |
|-----|------|-----|-----|------|------|---------|-------|
| 1 current seeder | 0.792 | 0.689 | 1.388 | 0.24 | 0.10 | 8.2 | **96%** |
| 2 uniform-over-domain | 0.729 | 0.561 | 1.120 | 0.11 | 0.05 | 3.9 | 39% |
| 3 atlas acquisition | **0.880** | **0.861** | **1.528** | 0.21 | **0.12** | 4.3 | 44% |

- **(2)v(1) de-clustering: HURTS** (Δmean −0.063, Δ≥1.0 −0.124). Spreading uniformly
  over the boundary band lands seeds in low-value / undescendable territory.
- **(3)v(2) value-targeting: HELPS** (Δmean **+0.151**, Δmed **+0.299**, Δ≥1.0 +0.096,
  Δ≥1.4 +0.072). Clean, unconfounded comparison (both arms inject un-screened points).
- **(3)v(1) total:** Δmean +0.088, Δmed +0.171, but Δ≥1.0 **−0.028** — arm 1 produces
  *more* good outcomes (59 vs 52) purely on its 96% productivity edge (see confound).

## Spread — outcome-appearance diversity (decisive, at matched yield)

Good outcomes (k3≥1.0) embedded via **v5 penultimate features** (1280-D, forward-hook
on the backbone); distinct = **greedy-leader near-dup survivors** at cosine TAU=0.01
(single-linkage chains the tight quality-head cone into one blob — wrong tool).

| arm | good | cover bins | distinct | /good | randM (matched M=28) |
|-----|------|-----------|----------|-------|-----------------------|
| 1 | 59 | 28 | 25 | 0.42 | 15.9 ± 1.5 |
| 2 | 28 | 21 | 14 | 0.50 | 14.0 ± 0.0 |
| 3 | 52 | 31 | 25 | 0.48 | **17.1 ± 1.6** |

- **(3)v(2) at matched count: +3.1 distinct survivors.** Arm 3's yield win is *not*
  one location over-mined — at equal good-count it yields **more** distinct
  appearances than uniform. Ordering (arm1≈arm3 > arm2) is stable across TAU∈[0.005,0.05].
- Per-good ratios (~0.42–0.50) are flat across arms → no arm collapses to a single spot.

## Attribution (atlas arm: exploit vs explore)

| tag | n | mean k3 | ≥1.0 | good outcomes | novel vs arm-1 |
|-----|---|---------|------|---------------|-----------------|
| exploit | 223 | 0.915 | 0.23 | 52 | **5 / 52 (10%)** |
| explore | 27 | 0.592 | 0.00 | **0** | — |

- **Explore produced ZERO good outcomes.** The low-conf / uncertainty-driven term is
  currently dead weight — the signal says move **λ down** toward pure exploit.
- **Exploit re-mines arm-1's known-good regions:** 90% of its good outcomes sit within
  TAU of an arm-1 outcome (median min-dist 0.0032 < 0.01). The exploit win is real but
  **partly circular** — θ̂ was trained on arm-1's own walks.

---

## Verdict

**The core thesis PASSES: θ̂ value-targeting is not worthless.** `(3) ≫ (2)` on both
yield (+0.151 mean) and matched-yield diversity (+3.1) — the value map adds real
signal *beyond* spreading, which was the decisive thing the 3-arm test was built to
detect. De-clustering **alone** hurts (arm 2 < arm 1); the win is value-targeting.

**But the atlas does not cleanly beat the current seeder yet, and the win is entirely
exploit:**

1. **λ=0.5 explore is dead** — 0 good outcomes from explore seeds. Lower λ (toward pure
   exploit) before this is production seeding.
2. **The exploit win is circular** — 90% of good outcomes re-find appearance regions
   arm 1 already generated. Genuine *new-territory* discovery did **not** happen. Proving
   the atlas expands beyond the current seeder's reach needs a non-circular test
   (propose → relabel → refit), not this round.
3. **Productivity confound:** arm 1's native seeds pass the current seeder's root
   black-gate → 96% descend past depth 1; arms 2–3 inject **raw un-screened** proposer
   points → ~44%. `(3)v(2)` is clean (both un-screened); `(3)v(1)` is confounded. A
   production atlas proposer needs a cheap **descendability pre-screen** to recover the
   productivity — and thus the good-outcome *count* — the raw current seeder gets for free.

**Recommendation:** conditional keep as a value-targeter, **not** wired into production
seeding as-is. Next: drop/shrink λ, add a root-descendability pre-screen to proposed
seeds, and design a non-circular discovery test. The proposer *works in principle*
(value map is real); it has not yet earned production seeding.

## Artifacts (`data/atlas/round1/`)

- `arm{1,2,3}_seeds.jsonl` — the three seed lists (arm 2/3 carry θ̂/conf/acq/tag provenance)
- `arm{1,2,3}_pool/` — full guided-descend pools (durable)
- `arm{1,2,3}_table.jsonl` — k3 best-over-walk harvest (+ reframed geometry)
- `arm{1,2,3}_embed.npz` — v5 penultimate outcome embeddings + rewards + tags
- `round1_report.json` — machine-readable metrics
- `out/atlas/round1/arm{1,2,3}_good_sheet.png` — good-outcome contact sheets (eyeball check)
