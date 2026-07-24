# The current "descent" algorithm — as-built (design input)

Read-only trace of what the descent does *today*, ahead of a redesign. Covers the three
flavors: (1) c-plane native (mandelbrot + multibrot3/4/5), (2) julia z-plane hook
(`--julia --c`, plus the `--julia-center` variant), (3) phoenix z-plane (`--run-phoenix`).

The descent has two layers:

- **The walk** — the Rust `guided-descend` subcommand (`src/guided_descend.rs`). This is
  the thing that "zooms and recenters." It is **blind to the aesthetic classifier**;
  it steers purely on field statistics and geometry.
- **The reward** — Python (`tools/atlas/production_seeder.py` →
  `tools/atlas_probe/step0_reanalysis.py` + `tools/reframe/reframe.py`). This runs *after*
  a walk finishes: it scores the emitted frames with the CORN classifier (active ckpt =
  **v7**, `tools/scoring/active_ckpt.py:56`) and reframes the best ones. **This is the only
  place a neural score enters the pipeline.**

> **Headline for the redesign.** One walk is a **single greedy chain** (no beam, no
> branching). Per rung it draws **4** policy-proposed candidate centers, filters them
> through black-cap → band → occupancy gates, and — at the shipped default — picks a
> **uniform-random survivor**. Zoom is **log-uniform per rung**. The classifier never
> touches the trajectory; it only judges finished frames. Families differ only in
> recurrence kernel, root-draw source, and per-degree band/box constants.

---

## 0. How the flavors are launched (orchestration)

All three shell out to the same Rust `guided-descend` binary (`prescreen.BIN`); the Python
only sets flags and does the reward pass.

| Flavor | Launcher (`production_seeder.py`) | guided-descend invocation |
|---|---|---|
| c-plane native | `_run` / `_gather` | `--seed-list` (survivors) `[--family multibrot{d}]` |
| julia hook | `run_julia_descent` (fired per qualifying parent in `_run`/`_gather` when `--julia-hook`) | `--julia --c <re> <im> [--family multibrot{d}] [--julia-center]` **native, never `--seed-list`** |
| phoenix | `run_phoenix_descent` (`_run_phoenix`, the `--run-phoenix` mode) | `--phoenix` native, repeated rounds |

**Production overrides vs. engine defaults (important).** The seeder does **not** run the
engine's built-in defaults — it passes an explicit walk config on every call
(`production_seeder.py:109-116`, applied in `generate_native_seeds`/`run_full_walks`/
`run_julia_descent`/`run_phoenix_descent`):

| Knob | Engine default | **Production value** |
|---|---|---|
| `--node-width` | 768 | **384** (`NODE_WIDTH`, foci low-pass; outcome-value-preserving per efficiency study) |
| `--sigma-band` | `16,20,24,28,32` | **`8,10,12,14,16`** (=engine ×0.5, matches the ×0.5 node width) |
| `--depth-min` / `--depth-max` | 4 / 17 | **4 / 14** (`DEPTH_MIN`/`DEPTH_MAX`; "≈depth-17 at lower cost") |
| `--descent-occ-floor` | 0.321 | 0.321 (`OCC_FLOOR`, unchanged) |
| `--descent-black-cap` | 0.30 | 0.30 (`BLACK_CAP`, unchanged) |
| `--per-walk-rng` | off | **on** (all walk + probe runs) |
| `--preview-width` / `--cols` | 640 / — | 48 / 40 (cosmetic previews only) |

The seed-generation call (`generate_native_seeds`) additionally forces
`--depth-min 1 --depth-max 1` to emit depth-1 roots only; the probe call
(`prescreen.prescreen`) forces `--depth-min 2 --depth-max 2`.

The remaining zoom / root-mixture / band / box knobs are left at their engine defaults
(the seeder never overrides them), so the numbers in §3/§6/§7 below are what production
actually runs for those.

---

## 1. Per-rung mechanics (the walk — depth ≥ 2)

Driver: `run_guided_descend` (`guided_descend.rs:261`); per-walk loop `for w in 0..n_walks`
(`:620`); per-rung loop `for d in 1..=target` (`:636`). Depth-1 is a separate root path
(§7). A single mutable `parent: Frame` is carried rung-to-rung (`:627`, reassigned `:763`).

**Candidate generation — 4 policy-proposed centers per rung.** New width is chosen first
(§3), then `best_of_n_step` (`:1085`) draws up to `n_cand` candidate centers,
`n_cand = --descent-candidates` **= 4** (`:2605`, loop `:1103`). Each center comes from
`pick_target` (`:1389`), a **3-branch stochastic policy** (one `rng.unit()` against the
branch weights `--w-foci 0.70 / --w-density 0.10 / --w-random 0.20`, `:2523-2532`):

- **Foci (p=0.70)** — `find_foci` (`:1712`): a genuine **σ-scale-space low-pass
  foci-finder** over the node-width smooth-iter (`mu`) field. Clips exterior at p90
  (`:1740`); per σ computes a normalized 3-box Gaussian `gauss(mu·valid)/gauss(valid)`
  (`:1760-1766`); dilates the interior mask by ~σ to suppress minibrot halos (`:1769`);
  takes local maxima above the per-scale 0.85 quantile (`MAX_FLOOR_PCT`, `:73`,
  `:1791-1839`); merges detections across σ (`:1846-1886`). Each focus scored
  **`peak_norm × isolation`** (`:1890`). Foci are then spread-suppressed
  (`spread_foci`, radius `--foci-diversity-radius` 0.12·node_w, `:1411`) and
  score-weighted sampled (`sample_focus`, `:1469`). **Not tiling / jitter / fixed-zoom —
  an actual scale-space blob detector.**
- **Density (p=0.10)** — `density_focus` (`:1491`): energy-weighted centroid of an edge-
  energy tile grid over the reused parent samples (void-guard snaps to the peak tile).
- **Random (p=0.20)** — near-boundary band point (`frame_boundary_point`, bottom 0.12 of
  exterior DE, `:1453`) when `--random-boundary` on (default true, and auto-off for
  dynamical kernels which carry no DE); else uniform interior point.

The chosen focus is placed into the child frame by `child_center` (`:1671`) per the
placement mixture `--placement 0.25,0.40,0.35` (center/off-center/edge); the Random branch
is forced `Placement::Center` (`:701-705`).

**σ-band / node-width role.** `--sigma-band` are the box-blur radii (px) of the foci low-
pass, applied on the `--node-width`-px field. Production: σ = `8,10,12,14,16` on a 384-px
field.

---

## 2. Per-rung ranking signal — gates, then (default) a coin flip

Inside `best_of_n_step` each candidate passes a **two-stage screen** (all field-statistic;
**no classifier**):

- **Stage 1** — cheap 128-px escape render; reject if `black_fraction ≥ black_cap`
  (`--descent-black-cap` 0.30, `:1113-1122`).
- **Stage 2** — node-width render → `screen_stats` → `descent_band.test(int_frac, spread,
  median)` rejects flat / instant-escape (`:1127`); then if the occ gate is on,
  reject if `energy::occupancy < occ_floor` (`--descent-occ-floor` 0.321, `:1131-1140`).

**Winner selection** — `SelectionMode`, **default `RandomSurvivor`** (`:2771`): reservoir-
samples **uniformly among survivors** (`:1147`). The opt-in `--selection least-interior`
picks the min-interior survivor (`:1146`). ⚠️ **So at the shipped default, the field
statistics only *gate*; they do not *rank* the winner.** The focus score (`peak×isolation`)
steers only *which pixel* a Foci candidate aims at, not which survivor wins.

**Classifier involvement inside the walk: none.** `render_node` (`:508`) invokes only
`PhoenixBackend`/`JuliaBackend`/`render_mandel_panel` — pure fractal kernels. Module doc:
"Geometric policies only — no CNN" (`:8`).

---

## 3. Zoom factor per rung

`new_fw = parent.frame_width * sample_log_uniform(zoom_lo, zoom_hi, rng)` (normal step
`:682`; julia-center step `:673`; `sample_log_uniform` `:1369`). **Not constant** —
log-uniform per rung in `[--zoom-lo, --zoom-hi]` = **[0.35, 0.50]** (`:2449`, `:2453`),
left at engine default by the seeder. The legacy fixed `--zoom-per-step` 0.4 (`:2443`) is
reported only and no longer drives stepping.

---

## 4. Trajectory shape

- **Single greedy path per walk**, one advancing `parent`. No beam, no branching.
- **Frames per walk = `reached`** (one `Candidate` per accepted rung, depths 1..reached),
  plus the depth-1 root. With production depth [4,14], a completed walk emits ~4–14 frames;
  a walk that dies early emits fewer.
- Every accepted rung → `cands.push(Candidate{…})` (`:747-761`).
- **`pool.jsonl`** — one row per frame (`:811-823`): `idx, walk, depth, target_depth,
  root_src, branch, placement, focus_score, cx, cy, fw, occ, png`.
- **`walks.jsonl`** — one row per walk (`:844-870`): `walk, root_src, target_depth,
  reached_depth, cause, death_depth, root_cx/cy/fw, child_occ` (child_occ = the depth-2
  admission occupancy, the value the probe reads).
- **Which frames are scored vs. only steering:** *all* pool frames are emitted, but the
  walk itself scores none of them. Downstream (§reward) the classifier raw-scores **every**
  emitted frame, then reframes only the **top `KRAW`=3**. Every intermediate frame is thus
  used only for steering + as a raw-score candidate; only the k3 winner's reframed crop
  becomes an outcome.

---

## 5. Termination conditions

The rung loop ends a walk on the first of (`EndCause`, `:157`):

| Cause | Trigger | Constant / value | file:line |
|---|---|---|---|
| `ReachedTerminalDepth` | `d == target` completes | `target ∈ [depth_min, depth_max]`; **prod [4,14]** (engine dflt [4,17]) | draw `:626`; `:2417`,`:2424` |
| `MinFwFloor` | `new_fw < min_fw` | `--min-fw` **1e-9** | `:674`,`:683`; `:2436` |
| `BlackCapExhausted` | all 4 candidates ≥ black cap | `--descent-black-cap` **0.30** | `:1117`,`:1172`; `:2614` |
| `OccFloorExhausted` | a cand cleared the band but all `occupancy < occ_floor` | `--descent-occ-floor` **0.321** | `:1135`,`:1172`; `:2623` |
| `DegenerateExhausted` | no cand cleared the band (flat / instant-escape) | see band below | `:1127`,`:1174` |

`descent_band` (`:425`) = `AcceptBand` with `interior_max` **disabled to 1.0** (black is a
presentation filter, not a nav cull). Its two live gates:
- **flat gate** `spread_min` (middle-90% smooth-iter spread p95−p5): per-family
  `flat_spread_min_default` (d2=20, `:2303`); override `--spread-min`.
- **instant-escape gate** `esc_median_min`: c-plane default 3.0; Julia loosened (§6);
  override `--esc-median-min`.

Extra: the occ floor is **skipped at the d1→d2 step** unless `--descent-occ-at-d1d2`
(default off) — d≥3 uses the full screen (`:590-593`, `:693`). Root-start widths are
guarded by `FW_FLOOR` 1e-9 (`:69`).

`target` (this walk's terminal depth) is drawn **once at walk start**, uniformly in
`[depth_min, depth_max]` (`:626`) — so a walk's length is fixed up front, not chosen by an
online "good enough, stop" test. There is **no score-based early exit** anywhere in the
walk.

---

## 6. The three flavors (what differs)

`WalkFamily` enum (`:2236`): `Mandelbrot | Multibrot3 | Multibrot4 | Multibrot5`;
`degree()` `:2249`. The per-rung loop (§1–5) is **identical** across flavors; only the
recurrence kernel, the root draw, and the per-degree band/box constants change.

**(a) c-plane native (mandelbrot + multibrot3/4/5).** `--family` (`:2658`). Kernel:
`render_mandel_panel` with `degree` (`:518`). Root: 8k-field/flat mixture (§7). Per-degree
constants:
- `root8k_band_defaults()` (`:2281`) `(mean_lo, mean_hi, var_floor)`: d2 `(8,120,6)`,
  d3 `(6,120,20)`, d4 `(7.2,120,209)`, d5 `(8.4,120,502)`.
- `flat_spread_min_default()` (`:2298`): d2=20, d3=15, d4=16, d5=17.
- `flat_box_default()` (`:2344`, the degree_bbox flat-box): d2 = cardioid
  `(-2.0,0.7,-1.2,1.2)`; d≥3 = origin square, half-width `2^(1/(d−1))·1.2`.

**(b) Julia z-plane (`--julia --c <re> <im>`).** `dynamical` set when `julia_c.is_some()`
(`:347`). Kernel: `JuliaBackend::new_degree(c, …, degree)` (`:513`). **Root differs
fundamentally**: `root_step_julia` (`:1298`) — the deterministic base-scale z-plane view at
center 0, width `--julia-root-fw` **3.0** (`:2698`). **Every walk shares this exact root**
(no sampling); all decorrelation comes from the downstream stochastic policy + per-walk
RNG. The DE-boundary Random branch is auto-disabled (dynamical kernels carry no DE:
`random_boundary && !dynamical`, `:357`). Bands from `julia_band_defaults()` (`:2327`)
`(esc_median_min, spread_min)`: d2 `(3.0,14)`, d3 `(2.0,10)`, d4 `(2.0,13)`, d5 `(1.8,13)`
— selected in `band()` only under `--julia` (`:2821`); these were promoted into the engine
so `--julia` resolves to them with no CLI override (the seeder's
`JULIA_GATHER_BANDS` table is kept bit-identical, `production_seeder.py:173`).
- **`--julia-center` variant** (`:2707`): pure centered zoom — every rung pinned at (0,0),
  **no finder / no best-of-N / no placement** (`center_step_julia`, `:1324`, dispatched
  `:667`). Still honors `min_fw` and the per-step zoom ratio. This is the "straight
  centered z-plane zoom" descent shape.
- **Why `--seed-list` is c-plane-only**: seeds are `(cx,cy,fw)` in the *parameter* plane;
  the dynamical root is a fixed *z*-plane view, so a seed list is meaningless and the engine
  errors (`:367`). Hence the julia hook always runs **native** (`run_julia_descent`,
  `JULIA_WALKS_PER_DESCENT`=3 walks/parent), never through the probe/`--seed-list` path —
  and standalone `--julia` under `--run` is blocked upstream by `resolve_family`.

**(c) Phoenix (`--phoenix`).** Also `dynamical` (`:347`), degree forced 2 (`:317`). Fixed
Ushiki `c` (`--c` dflt `0.5667,0`) + `p` (`--p` dflt `-0.5,0`). Kernel:
`PhoenixBackend::new(pc, pp, …)` (`:509`). Root: **identical to Julia** —
`root_step_julia` at center 0, width `--julia-root-fw` 3.0 (`:648`, `root_src="phoenix"`).
⚠️ Phoenix stays on the **c-plane Mandelbrot band defaults** (esc 3.0 + d2 spread), *not*
the loosened Julia table — `band()` gates the Julia table on `self.julia` only (`:2821`),
and Phoenix is not `julia`. Mutually exclusive with `--julia` and `multibrot*`.

---

## 7. Root draw / seeding (depth-1)

Dispatched `:637-666`, four mutually-exclusive sources:

1. **Injected (`--seed-list`)** — `root_step_injected` (`:1349`): pins depth-1 to row `w`'s
   `(cx,cy,fw)`, consumes no RNG (keeps the depth≥2 stream aligned across configs).
   c-plane-only. **This is the production c-plane path**: the seeder feeds depth-2-probe
   survivors here.
2. **Dynamical (julia/phoenix)** — `root_step_julia` (deterministic, shared root).
3. **8k-field root** — with prob `--root-mix` 0.5 (`:2466`, `rng.unit() < root_mix`):
   `root_step_8k` (`:1201`) draws a passing window uniformly from the pre-scanned 8k smooth-
   field windows (`RootField::passing_windows`), accepts first whose node black-fraction ≤
   `--root8k-black-max` 0.80. Windows scanned once at `--root-zoom-8k` 0.10 against the
   per-degree `root8k_score_cfg`.
4. **Flat sampler** — otherwise: `root_step_flat` (`:1247`): uniform-in-box center
   (`flat_box`) × log-uniform fw `[--flat-fw-lo 0.003, --flat-fw-hi 0.05]`, cheap-screened
   at `--flat-screen-width` 320 against the real `AcceptBand`.

Both native roots are **permissive** (≤80% black, occ floor *not* applied) — the depth≥2
gates do the tightening. **`--per-walk-rng`** (on in production, `:2732`) reseeds the RNG
per walk from `(seed, walk_index)`, so each walk's root is reproducible independent of how
many draws the finder consumed.

**Where the depth-1 seeds themselves come from (production c-plane):** the engine's *own*
depth-1 root draw (a separate `generate_native_seeds` run at `--depth-min/max 1`) proposes
seeds; the seeder applies q3-density **rejection sampling** in (cx,cy) space, then a
**depth-2 descendability probe** (`prescreen`, reached≥2, pure field-stat, **no
classifier**), and only survivors are fed to the full walk via `--seed-list`.

---

## 8. The reward path (post-walk — where the classifier lives)

Runs in Python after each walk, on the emitted `pool.jsonl` frames. Scorer = guarded CORN
v7; `score_paths` returns per frame `(score = p_notbad + p_good = E[ord] ∈ [0,2],
p_notbad = σ(logit₀), p_good = σ(logit₁))`, **one forward pass per frame**
(`score_lib.py:108-124`).

1. **`raw_screen_walk`** (`step0_reanalysis.py:110`): render every walk frame once at the
   reframe **x1.0 center rung**, **640×360 ss2**, twilight_shifted (byte-identical to
   reframe's original-framing tile), score all in one batch. Take the top **`KRAW`=3**
   raw frames (`:73`, `:157`).
2. **`reframe_location`** on each of the top-3 (`reframe.py:198`): a bounded, single-pass
   local search — a **4-rung fw ladder** `(0.5, 0.707, 1.0, 1.414)` at recenter (0,0), then
   a **3×3 recenter grid** `(-0.25, 0, 0.25)·fw` at the best fw ⇒ **12 distinct renders**
   (the x1.0/recenter-0 tile is shared), all **640×360 ss2**, each scored by E[ord]; argmax
   wins. The x1.0 no-op rung is always in the space, so `score ≥ original_score` by
   construction (monotone; asserted `step0_reanalysis.py:169`).
3. **k3 winner** = the best reframed crop across the 3 (`harvest_walk_reward`,
   `production_seeder.py:731`). Its `(p_notbad, p_good)` are CORN-decoded at the per-
   partition `t_good` (`corn_decode`); a **guard-passing, decoded-class-3** outcome that is
   not a near-dup of the q3 cloud is admitted. `reward_k3 = E[ord]` of that crop is the
   ranking value; the degenerate-field guard collapses fully-degenerate top-3s to a
   sentinel.

The reframe search geometry (fw ladder + recenter grid) is a *reward-time* re-selection —
it is **not** part of the walk trajectory and does not feed back into steering.

---

## 9. Cost profile (per descent)

Let a descent be `N` walks with mean `F` frames/walk (F ≈ 4–14 at prod depth).

| Stage | Renders | Resolution / path | Classifier fwd passes |
|---|---|---|---|
| (a) depth-2 pre-screen probe | engine-internal per seed to depth 2 | field-stat (node 384, preview 48) | **0** |
| (b) full walk (guided-descend) | engine-internal per node over the walk | field-stat | **0** |
| (c) `raw_screen_walk` | `F` per walk | 640×360 ss2 | `F` per walk |
| (d) `reframe_location` × top-3 | `3 × 12 = 36` per walk | 640×360 ss2 | `36` per walk |
| (e) `outcome_feature` (1280-D) | 1 per committed outcome | 640×360 ss2 | 1 (penultimate) |

Per walk: **`F + 36`** classifier forward passes and coarse renders on the Python side (all
640×360 ss2), on top of the engine-internal field-stat renders in (a)+(b) that the
classifier never sees. Wall time is dominated by the walk (engine, seconds–minutes) plus
the `N·(F+36)` coarse GPU forwards; production caps concurrency at
**`WORKERS`=4** (`production_seeder.py:131`). The seeder's `--time-only` mode projects
per-batch wall time from one smoke batch; `_run` is wall-clock-boxed at
`WALLCLOCK_BUDGET_MIN`=30 by default.

---

## 10. Dead / vestigial steering (flag for the redesign)

- **`--zoom-per-step`** (`:2443`) — legacy fixed zoom; no longer drives stepping (log-
  uniform band replaced it). Reported nominal only.
- **`--root-zoom`** (`:2459`) — retired rev1–3 boundary-sampler root; "kept only so old
  invocations still parse." Never read in the walk.
- **`FinderMode::Percentile` + all `--pct-*` flags** (`:2358`, `:1544`, `:2779-2797`) —
  explicitly **PARKED** ("escape-value banding is not a diversity axis; verified null").
  Dormant seam, not production.
- **`Focus.persistence`** (`:209`, computed `:1862`) — measured across σ but **never used**
  in the sampling weight (`score = peak_norm × isolation`, `:1890`). Reported attribute
  only.
- **`RandomSurvivor` (the default) wastes the field statistics for ranking** — interior-
  fraction and occupancy are computed and logged per candidate (`chosen_interiors` drift
  metric, `:614`,`:746`) but do **not** rank the winner; only `--selection least-interior`
  makes any statistic load-bearing for *selection*. At the shipped default the sole ranking
  pressure is the pass/fail gates, and the winner among survivors is a **coin flip**. This
  is the biggest "signal computed but not used to choose" surface in the walk.
