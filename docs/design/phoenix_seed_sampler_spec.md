# Phoenix seed proposal sampler — design spec

**Audience:** an implementer (Claude Code or another instance) with no access to the
originating conversation. Everything needed is here.

**Goal.** Build the phoenix analog of "sample points near ∂M as good Julia seeds."
A *seed* is a point in phoenix **parameter space** `(c, p, z₋₁)`. Each proposed seed
is later handed to an existing **descent** algorithm that zooms into that phoenix's
z-plane looking for a high-quality renderable location. The sampler's job is to
propose seeds that are *fertile* — that the descent can turn into keepers — the way
"near ∂M" reliably produces interesting Julia sets. This doc gives (1) the geometry
that replaces the ∂M recipe, (2) the closed forms to sample from, (3) the parameter
axes and their traps, and (4) how to wrap it in a fertility-aware loop, since a seed
is scored by an *expensive, stochastic* descent rather than a cheap check.

---

## 1. Why the Julia recipe does not port directly

Phoenix iteration (standard form):

```
z_{n+1} = z_n² + c + p·z_{n-1},   z₀ = pixel,  z₋₁ = 0
```

Lift to the pair `(zₙ, z_{n-1})` to make it a genuine map:

```
F(z, y) = ( z² + c + p·y ,  z )
```

This is a **complex Hénon map**. Its Jacobian is `[[2z, p], [1, 0]]`, so
**det F ≡ −p** (constant). Consequences that break the Julia intuition:

- `F` is **invertible** for `p ≠ 0` ⇒ it has **no critical point** ⇒ the theorems the
  Mandelbrot recipe rests on (connectedness ⇔ bounded critical orbit) **do not apply**.
  There is no honest "connectedness locus" to sit on the boundary of.
- What you render in the z-plane is **not a Julia set**. It is the `y = z₋₁` **slice**
  of a filled Julia set `K⁺ ⊂ ℂ²`. The familiar "bird" is partly a slice artifact.
- For small `|p|` it is a perturbation of `z²+c` (Hubbard–Oberste-Vorth,
  Fornæss–Sibony), so the Mandelbrot intuition *degrades gracefully* rather than
  vanishing. The classic `p = −0.5` is **not** small — do not assume M-like structure.

**The replacement.** You cannot sample "near a connectedness locus," but you *can*
sample near the **exact, closed-form stability skeleton** — the bifurcation curves
where the fixed point / period-2 cycle go neutral. These are rigorous boundaries
(they don't depend on criticality), and near-boundary is exactly where the
near-parabolic filigree lives — the same reason near-∂M Julia sets are the good ones.
The classic Ushiki phoenix seed is literally a point just past this skeleton's cusp.

---

## 2. The closed-form skeleton (the core of the sampler)

All derived from `F` above; each collapses to the corresponding Mandelbrot feature at
`p = 0`. **Assert these `p = 0` collapses as unit tests.**

### 2.1 Fixed points and their multiplier

Fixed points solve `z² + (p − 1)z + c = 0`. The multiplier `λ` at a fixed point solves

```
λ² − 2z·λ − p = 0        ⇒   λ₁·λ₂ = −p
```

Because both eigenvalues must have `|λ| < 1` for an attracting fixed point and their
product has magnitude `|p|`, **`|p| < 1` is necessary** for any attracting fixed point.
Keep proposals mostly in `|p| < 1`; treat `|p| ≥ 1` as a separate exploratory mode.

### 2.2 Cardioid analog (primary component boundary)

Invert the multiplier relation: pick a boundary phase `θ`, set the neutral multiplier
`λ = e^{iθ}`, and read off the parameter directly.

```
z(θ)  = ½ ( e^{iθ} − p·e^{−iθ} )
c(θ)  = z · ( 1 − p − z )
```

- **`p = 0` collapse:** `z = ½e^{iθ}`, `c = z(1−z) = μ/2 − μ²/4` with `μ = e^{iθ}` — the
  **main cardioid** of M, exactly. ✓
- **Cusp** (`θ = 0`, `λ = 1`): `c = ¼(1 − p)²`. For `p = −0.5` this is **`9/16 = 0.5625`**.
  The classic Ushiki seed `c ≈ 0.5667` is **~0.0042 past the cusp** — the phoenix
  equivalent of sitting at `c = 0.25 + ε` on M. **Assert cusp = 0.5625 at p = −0.5.**

### 2.3 Period-2 analog (the "−1 disk" boundary)

The 2-cycle `{z₁, z₂}` satisfies `z₁ + z₂ = p − 1`, `z₁z₂ = c + (p−1)²`, and its cycle
multiplier `Λ` obeys `Λ + p²/Λ = 4·z₁z₂ + 2p`. Setting `Λ = e^{iθ}` (neutral):

```
c(θ) = ¼ ( e^{iθ} + p²·e^{−iθ} − 2p ) − (p − 1)²
```

- **`p = 0` collapse:** `c = e^{iθ}/4 − 1` — the **period-2 disk** (center −1, radius ¼). ✓
- The cardioid-analog and period-2-analog curves are **tangent at the bulb root**, as on M.

### 2.4 Higher bulbs / root points

Root points of period-`k` bulbs sit where the relevant multiplier is `λ = e^{2πi·q}` for
rational `q = m/k`. These are the Misiurewicz-flavored, high-yield spots (dense
filigree, spiral centers). For a first version, the cardioid + period-2 curves plus
their root points are plenty; add higher bulbs by root-finding on the period-`k`
multiplier condition if you want more variety per `p`.

### 2.5 How to sample *near* the skeleton

For a fixed `p`, `c(θ)` traces a closed curve. Propose:

```
c_proposed = c(θ) + offset · n̂(θ)
```

where `n̂(θ)` is the **outward** unit normal (compute numerically: tangent =
`(c(θ+δ) − c(θ−δ))`, rotate 90°, normalize; pick the sign pointing away from the
component interior) and `offset` is a small displacement. `offset` is the key knob —
it is the "just past the boundary" dial. Small positive offsets land in the
near-parabolic filament zone (the Ushiki regime). Draw `offset` from a distribution
concentrated near 0 (e.g. half-normal, scale ~1e-2 to 1e-1 in c-units) with a heavier
tail for occasional deeper excursions. Also emit exact root points (`offset = 0` at
rational `q`) as their own proposal class.

---

## 3. Parameter axes and their traps

A phoenix seed has more freedom than a Julia seed. **Diversity comes from opening
these axes, not from sample count.** The historically observed "variety-poor narrow
multi-spiral band" is the symptom of pinning them.

- **`p` (complex).** The classic plane pins `p` real (often at one engine default). `p`
  is the single biggest untapped diversity axis. Sweep `p` across the `|p| < 1` disk,
  including complex values. Each `p` gives a *different* skeleton to sample near.
- **`c` (complex).** The classic plane also pins `c` real. The closed forms above are
  fully complex; use them that way.
- **`z₋₁` (the slice coordinate) — load-bearing.** With `z₋₁ = 0`, orbits from `z₀` and
  `−z₀` coincide from step 1, so **the rendered image is exactly 180°-symmetric** — a
  slice artifact that halves the effective variety. **Setting `z₋₁ ≠ 0` breaks the
  symmetry and yields a visibly different set from the same `(c, p)`.** Treat `z₋₁` as a
  first-class proposal axis (small complex offsets from 0, plus larger excursions),
  not a constant.

A full proposal is therefore parametrized by `(p, branch, θ, offset, z₋₁)` →
deterministic `(c, p, z₋₁)` via the closed forms. Keep the classic real-`c`, real-`p`,
`z₋₁ = 0` case as one named sub-mode for reproducing known results, but do not let it
dominate the draw.

---

## 4. The pseudo-Mandelbrot proxy (`mandphoenix`) — one legitimate use

Setting `z₀ = z₋₁ = 0` and sweeping `(c, p)` with escape-time gives Fractint's
`mandphoenix` figure. **Caveat:** because `z = 0` is *not* a critical point here, this
is a **heuristic proxy**, not a connectedness locus — do not treat its boundary as
rigorous. It equals M exactly at `p = 0` and deforms as `|p| → 1`.

Its one solid use: the **boundary distance of `(c,p)` in the `z₀=0` escape field is a
cheap, closed-form-adjacent *fertility prior feature*** — "is there structure near
here" without running a descent. Feed it to the surrogate in §5. Do **not** use it to
gate proposals hard.

---

## 5. Wrapping it in a fertility-aware loop (critical — not optional)

**The evaluation reality:** a seed is scored by running a full **descent** on it, which
is expensive and **stochastic** (the descent has per-walk RNG). This changes the
architecture. Two naive designs are both wrong:

- **Frozen "precompute 10,000 good seeds" list — reject.** Scoring each candidate with a
  *single* noisy descent means a frozen list caches seeds that got **one lucky descent**,
  not seeds that are reliably fertile. Re-descended in production they regress to the
  mean. A large count *hides* this as survivorship noise dressed up as curated
  diversity. (This is the same winner's-curse / distribution-bound failure seen when
  ranking on a selector's own tail-error-concentrated output.)
- **Blind resample every time — wastes the expensive signal** you paid descents to learn.

### 5.1 Step 0 — variance decomposition (do this first; it decides everything)

Take `N` seeds spread across the skeleton (vary `p`, `θ`, branch, `z₋₁`). Descend each
`K` times. Split descent-quality variance into **between-seed** vs **within-seed**.

- **Between-seed dominates** → fertility is a real, structural, *cacheable* property →
  build the surrogate + memory below, and expect to need **many distinct** fertile
  seeds (see next point).
- **Within-seed dominates** → seed choice barely matters beyond "near the skeleton" →
  just propose fresh from §2 each time and let descent stochasticity carry variety; a
  fertility cache would be modeling noise.

**Prior expectation for phoenix: between-seed dominates.** A single phoenix has a thin
z-plane repertoire (the symmetry + narrow-band behavior), so most variety lives
*across* seeds, and you **cannot amortize a fertile seed by re-descending it** (few
distinct good looks per seed). That makes the outer parameter-space search the whole
game, with every probe costing a descent — which is exactly what the surrogate buys
down.

### 5.2 If fertility is cacheable — the sampler is a surrogate-ranked, memory-backed loop

1. **Propose** candidates cheaply from §2 (skeleton + offset + `z₋₁` + `p`).
2. **Rank by a fertility surrogate** on cheap features — spend descents only on
   predicted-fertile candidates. Feature vector per candidate:
   - `mandphoenix` boundary distance at `z₀=0` (§4) — closed-form prior.
   - distance to nearest bulb root; `|offset|`; `|p|`, `arg p`; `θ`, branch id; `|z₋₁|`.
   - *(optional)* one **shallow descent probe** as a cheap feature, if pure-geometry
     features underpredict. (Analogously, inside the descent a cheap 384px colorized
     field predicts the canonical score at Spearman ≈ 0.95 — a cheap-presentation
     surrogate is a proven move; lift the same idea to the seed plane.)
   Start with a light head (logistic / small GBM) over frozen features; refit online.
3. **Spend full descents** on the top-ranked candidates.
4. **Accumulate**: a seed's fertility estimate is the **running average over its
   probes** (this is what defuses the single-draw winner's curse), stored in a
   **growing, deduped seed memory** — not a frozen list. Dedup in parameter space
   (near-duplicate `(c,p,z₋₁)` suppressed), analogous to an existing coordinate
   dup-cloud / novelty memory.
5. **Explore/exploit**: mostly draw near proven-fertile regions of parameter space,
   always spend a fixed fraction probing fresh skeleton (new `p`, unexplored `θ`),
   so the memory keeps expanding instead of collapsing onto an early basin.
6. **Reproducible** via a seeded RNG throughout.

This subsumes both naive options: it gives precompute's reproducibility and
curation **without** its ceiling or its single-draw contamination, and it keeps
tracking any later improvement to the seed scorer.

---

## 6. Suggested interface

```
propose_seed(rng, memory, surrogate) -> Seed
    # 1. explore/exploit branch on rng
    # 2. draw p (mostly |p|<1, incl. complex; classic-real sub-mode with low prob)
    # 3. draw branch ∈ {cardioid, period2, bulb(q)}, θ, offset, z₋₁
    # 4. c = closed_form(branch, p, θ) + offset * outward_normal(branch, p, θ)
    # 5. features = cheap_features(c, p, z₋₁)      # §5.2 step 2
    # 6. return Seed(c, p, z₋₁, features)          # caller ranks a batch by surrogate

Seed        = { c: complex, p: complex, z_m1: complex, features: vec }
MemoryEntry = { seed_key, n_probes, mean_quality, ... }   # running average, deduped
```

Batch-propose, surrogate-rank, descend the top few, update `memory` and refit
`surrogate` from realized descent quality. Never freeze the pool.

---

## 7. Assertions to bake in (cheap correctness net)

- `p = 0`: cardioid analog `c(θ)` equals `μ/2 − μ²/4` (`μ = e^{iθ}`); period-2 analog
  equals `e^{iθ}/4 − 1`. (Match M to ~1e-12.)
- `p = −0.5`: cardioid cusp `c = 9/16 = 0.5625` exactly; classic Ushiki seed
  `c ≈ 0.5667` lies just outside it (`offset ≈ +0.0042` along the real axis).
- `z₋₁ = 0` render is 180°-symmetric; `z₋₁ ≠ 0` is not — regression-guard the
  symmetry break so nobody silently re-pins `z₋₁`.
- Multiplier product check: numerically, `λ₁λ₂ = −p` at any fixed point;
  cycle-multiplier product `= p²` for the period-2 orbit.

---

## 8. One-line summary

The ∂M-for-Julia move has no literal analog (phoenix is an invertible Hénon map with
no critical point), but the **fixed-point / period-2 neutral-stability skeleton is an
exact closed-form boundary to sample near**, per `p`; open `p`, `c`, and especially
`z₋₁` for variety; and because a seed is graded by an expensive stochastic descent,
wrap the proposer in a **variance-decomposition check → surrogate-ranked, running-
average, deduped, explore/exploit memory** rather than freezing a static list.
