# Pipeline pre-flight — findings for the 6h unattended orchestrator

Date: 2026-07-12. Binary: `target/release/fractal-generator.exe` (built today, current vs source).
All timings measured on this box (RTX 2060 SUPER 8 GB, the release exe). Timed slices kept short and extrapolated.

---

## TL;DR

- **The prompt's 4-stage chain conflates two different pipelines.** `enrich --mode score` is the **labeling** screen (guided-descend `pool.jsonl` → label corpus). The **emission** chain that actually produces wallpapers is: `production_seeder` (scores v6 **inline** with the per-degree `t_good` gate) → `build_fresh_discovery` (ledger → wallpaper pool batch) → `emit_v1 --pool` (v3 gate + MAP-Elites + render) → `deploy_tail`. Tonight's run is the **emission** chain; `enrich` is characterized but off the critical path.
- **Bottleneck = Stage 1 discovery** (~27 q3/hr mandelbrot, less for other families). Everything downstream is fast. **Second-largest time sink = Stage 4 deploy-tail** (~2 min/emitted-location), which is time-boxable because it is resumable.
- **Flow should be DECOUPLED**, not lockstep: discovery fills a backlog for the bulk of 6h, then one fast emit drain + one time-boxed tail pass at the end. All stages share the single GPU, so run them in **GPU-exclusive phases**.
- **Gates confirmed resolution-invariant** — dropping the final emit render to 1024 ss2 does **not** move any gate or selection (verified live: gate scored 364 crops identically at head-fixed 384×224 while final render was 1024 ss2).
- **Blockers to fix before tonight** (details below): (B1) emit has **no eval-res flag** — source edit required; (B2) emit **manifest is written once at the end + fail-fast** — one bad render loses every recipe row; (B3) **recipe is incomplete** — manifest omits res/ss/filter/render-mode/canon, so eval-res output is indistinguishable from wallpaper output and can't be reproduced at full-res later (this defeats the deferral); (B4) **all-families emission gap** — `build_fresh_discovery` is mandelbrot + julia:mandelbrot **only**; (B5) known **palette/color collapse** in the pref ranking upstream of emit (the "all-fire" risk).

---

## Per-stage table

| Axis | S1 discovery (`production_seeder`) | S2 screen (`enrich --mode score`) † | S3 emit (`emit_v1`) | S4 deploy-tail (`deploy_tail`) |
|---|---|---|---|---|
| **Input** | Self-generates seeds (engine root-draw + rejection sampling). No hand-fed list. `--julia-hook` covers Julia twin. | Guided-descend `pool.jsonl` (cx/cy/fw + idx). **Not** the S1 ledger. | A **pool batch dir** (`images.jsonl` + crops), built from the S1 ledger by `build_fresh_discovery`/`build_humanq3`. | Auto-consumes the whole S3 `emit_v1/manifest.jsonl` (corpus-level). |
| **Output** | `data/discovery/outcome_ledger.jsonl` (append-only, one row/walk) + `outcome_feats.npz` + per-run summary. | `data/enrich/<run>/scored.jsonl` (once, at end) + incremental `score_meta.jsonl` sidecar. | `out/wallpaper/emit_v1/`: `wallpapers/*.png` (per-iter) + `manifest.jsonl` (**once, at end**) + contact sheet. | `<emit-home>/alternates.jsonl` + `alternates/*_2560x1440.png` (durable); scratch in `out/mining/deploy_tail/`. |
| **Auto-consume?** | Pull-based; nothing watches it. Downstream reads the ledger. | Popen-launches the Rust side itself (single python cmd). | Reads pool dir; runs v3 head **in-process** (no pre-scored ledger). | Reads emit manifest up front. |
| **Throughput (measured)** | Full pipeline (descent+probe+GPU reward): **~40 probed/hr, ~27 q3/hr** (mandelbrot; lower for high-deg). GPU reward dominates. | **~1.1–1.5 loc/s ≈ 4–5.5k loc/hr** screened+scored. CPU-iterate bound. | Gate over pool: **11 s one-time**. Eval-res 1024 ss2 render: **~12.5 s/emission** (dump-field ~9.7s + recolor ~3.4s). Yield selector-bound (~21 winners/pool). | Candidate render **~17.6 s/candidate** (7 modes/loc ≈ **2 min/loc**); gate cold-start ~12 s; keeper full-res ~90 s (≤25% of N). |
| **Yield / fan-out** | ~71% q3 (mandelbrot) → ~17% (multibrot3). Family-dependent. | Validity screen ~78% pass; each survivor → K recolors (default 4, up to 76). Doesn't itself apply `t_good`. | Pool (~360 cand) → **~21 emitted** (MAP-Elites cell-saturating; gate doesn't bind — passers already p_ge3>0.94). | ≤1 alt/loc; budget `B=round(0.25·N)`; floors bite only at **N≳36**. |
| **State / resume** | **Append-only, durable, resumable** (cloud rebuilt from ledger on restart). Atomic npz/summary. | `scored.jsonl` **not** incremental/resumable (written at end); meta sidecar flushes/200. | **Not resumable** — manifest all-or-nothing at end. PNGs survive but orphaned. | **Resumable, idempotent** (skip-if-exists, existing alts fixed, self-heal). No-op on unchanged corpus. |
| **Failure isolation** | `--run` **aborts** on engine failure (SystemExit). `--gather` is **hardened** skip-and-continue. | **Skip-and-continue** per location; hard-fail only at stream/process boundary. | **Fail-fast, crashes stage** (no per-item try/except); + manifest-at-end ⇒ one bad render discards all prior recipe rows. | **Skip-and-continue** per candidate; retries once. |
| **Gate resolution** | Guard/prescreen at **640×360 ss2** (interior_frac≥0.25 fail, field_std<6 fail). | v6 gate at **384×224 bicubic stretch** (fixed); validity screen at 1280×720 (black<0.30, occ≥0.321). | v3 gate + selector on **pre-rendered ss2 crops** at head-fixed geometry. **Independent of final render res.** ✅ | mining_v1 gate at **384×224** (fixed); candidates rendered to 1280×720 ss2 first. Independent of keeper res. ✅ |
| **Diversity tags** | **`family`** (partition) + decoded_class + coords. Sufficient for family diversity. | palette (`argmax_palette` + per-K); **not** family/mode. | **family + color_cell (CIELAB) + palette + location**. | render **mode** only (7 promoted modes); ≤1/loc. |

† S2 `enrich --mode score` feeds the **labeling** corpus, not emission. Its v6 `t_good` gate (0.24 deg-2, 0.30 jm3/4/5, 0.50 baseline) lives in the **discovery seeder** (`production_seeder.t_good_for`), applied **inline in Stage 1** — not inside `enrich`.

**Decision-resolution verdict (the critical check): PASS.** Every gate scores at its own fixed resolution (640×360, 384×224, or head-fixed crop geometry) and is **wholly independent of the final render resolution**. Dropping emit to 1024 ss2 leaves the passing set and MAP-Elites picks **byte-identical** to a full-res run. Verified live in the smoke: with `EMIT_W/H/SS=1024,576,2`, the gate still scored the 364 pool crops at 384×224 (`p_ge3` quantiles unchanged) and selected the same 21 winners.

---

## 6h budget model + bottleneck

Per-unit cost spans ~3 orders of magnitude:

| Stage | Per-unit | Units in 6h (if run flat-out) |
|---|---|---|
| S1 discovery | ~130 s per **q3 location** (full pipeline) | ~150 q3 across families (GPU-bound) |
| S3 emit | ~12.5 s per **emission**, ~21/pool | thousands possible — never the limit |
| S4 tail | ~120 s per **emitted location** (7 modes) | ~180 locations of curation |

**Bottleneck = S1 discovery.** It is the only stage that gates *volume + diversity* of q3 locations, and it is 10–100× slower per unit than emit. Emit is selector-bound (a pool emits ~21 winners in ~5 min regardless of how many q3 feed it), so it never limits. **Second cost = S4 tail** (~2 min/emitted-loc); at N≈100 emissions the tail alone is ~3 h — but it is resumable, so it can be **time-boxed** and safely cut.

**Budget allocation (recommended):**
- **~4.5–5.0 h: discovery only.** Hardened family-rotation loop filling the ledger. GPU dedicated to reward scoring.
- **~20–40 min: emit drain.** `build_fresh_discovery` → `emit_v1 --pool` (eval-res) over accumulated q3. Fast.
- **~30–60 min: deploy-tail**, time-boxed to remaining budget. Resumable ⇒ a hard cap that interrupts it keeps whatever it curated.

---

## Proposed inter-stage flow: DECOUPLED backlog (not lockstep)

Justified by the throughput asymmetry: discovery is ~100× slower per unit than emit, and all four stages share the single 8 GB GPU. Lockstep would idle the fast downstream while discovery crawls, and would force GPU contention.

```
Phase A (bulk):   discovery standing loop  ──appends──▶  outcome_ledger.jsonl   [GPU: reward scoring]
Phase B (drain):  build_fresh_discovery (ledger→pool) ─▶ emit_v1 --pool (eval-res) [GPU: v3 gate]
Phase C (curate): deploy_tail (corpus-level, time-boxed)                          [GPU: mining gate]
```

- **GPU-exclusive phases** — do not overlap A/B/C; the 8 GB card cannot host two of these torch stages comfortably alongside discovery's reward model.
- **Backlog is naturally capped** — emit is selector-saturating (MAP-Elites behavior space = family × 18 Lab cells), so unbounded q3 backlog does not translate into unbounded emit volume; no explicit ledger cap needed for correctness, though a soft cap keeps the tail bounded.
- **Deploy-tail must run LAST** (or time-boxed near the end): its budget `B=round(0.25·N)` and per-mode floors are functions of total corpus N and are degenerate on a partial set (below N≈36 floors round to 0). It *is* safe to run mid-way (idempotent) but only meaningfully curates once N is large.

---

## Orchestrator must-haves

1. **6h cap on a safe boundary.** Cap Phase A at a **family-rotation boundary** (gather-style `--minutes-per-class`), and **reserve a fixed tail window** (~45–60 min) for B+C so they are never cut off. Do not let discovery consume the whole budget.
2. **Per-unit hard-kill backstop.** Discovery `--run` and emit are **not** self-hardening. Wrap each stage subprocess with a watchdog timeout + `0xC0000142` DLL-wedge retry (gather_overnight already does the wedge retry for discovery). Emit especially needs a per-invocation timeout.
3. **Durable resume.** Discovery ✅ and tail ✅ resume cleanly. **Emit does not** — see B2. Mitigate by running emit in **small per-pool batches** so each writes its own atomic manifest, or make the manifest incremental.
4. **Failure isolation.** Use discovery's **hardened `--gather`-style** skip-and-continue path (not bare `--run`). Emit crashes the whole stage on one bad render — isolate by small batches. Tail is already skip-and-continue.
5. **Backlog caps.** Not required for correctness (emit is selector-bound), but cap the emitted-corpus N feeding the tail so Phase C fits its window (tail ≈ 2 min × N).

---

## Blockers (fix before tonight)

- **B1 — emit has no eval-res flag [HARD, quick fix].** `EMIT_W/H/SS` and `EMIT_FILTER` are hardcoded module constants (`emit_v1.py:76-77`); no `--width/--ss/--filter`. To run at 1024 ss2 you must **edit the constants** (verified working in the smoke). Recommend adding real CLI flags so the run is reproducible without a source edit.
- **B2 — emit manifest written once at the end, fail-fast [HARD].** `manifest.jsonl` is written only after every winner renders (`emit_v1.py:302`), and there is no per-item try/except. A single bad render **discards all recipe rows** for that invocation and leaves orphan PNGs. This is the biggest durability risk for an unattended run. Fix: incremental append, or small per-pool batches.
- **B3 — recipe incompleteness [HARD — defeats the deferral].** The manifest records location coords, palette id, color params — but **omits `eval_width/height/ss/filter`, `render_mode`, and the wallpaper canon target**. Consequences: (a) at eval-res the manifest is **byte-identical to a full-res manifest** — nothing records that these PNGs are 1024 ss2, not wallpapers; (b) the canon needed to reproduce full-res later (2560×1440 ss4 lanczos3) lives only in the source constants of the day; (c) `transfer`/`transfer_gamma` fall to defaults and the stored `eval_filter:"box"` is misleading (true filter is the unrecorded `EMIT_FILTER`). **The whole point of eval-res emission is a complete deferred recipe — this must be fixed** by writing per-row: `eval_width/height/ss/filter`, `render_mode`, and the target wallpaper canon.
- **B4 — all-families emission gap [MEDIUM, threatens the diversity goal].** `build_fresh_discovery` filters to **mandelbrot + julia:mandelbrot (deg-2) only**. The ledger has all 8 guarded families (mandelbrot, multibrot3/4/5, + julia twins; phoenix is gather-only), but no pool builder feeds multibrot/high-deg/phoenix into emit. **"All 9 families" emission does not exist end-to-end** — either generalize the pool builder or accept tonight covers 2 families. This directly limits the family-diversity axis the run is meant to test.
- **B5 — palette/color collapse upstream of emit [MEDIUM, the "all-fire" risk].** Palettes are **baked into the pool by the pref ranking beam**, and the pref-v3-gvo head is known to concentrate ~96% of per-location #1 picks on amber-gold/violet-purple (see memory `palette-family-collapse-localized`). So "all palettes in config" does **not** guarantee color diversity — the ranking collapses it before emit sees it. MAP-Elites (family × Lab-cell) partially counteracts, but the input is pre-collapsed. **Watch the CIELAB-cell histogram of emissions**; if it clusters, the collapse is upstream in the pref beam, not in emit or K.

---

## Cross-cutting

**Recipe completeness — INCOMPLETE (see B3).** Captured: `cx/cy/fw` (decimal strings), maxiter, family + `c_re/c_im`, palette id, colormap params (reverse, log_premap, gamma, phase, n_cycles, interior_color). **Missing and required for safe deferral:** render resolution/ss/filter, render_mode, wallpaper canon target, and `transfer`/`transfer_gamma`. **This is the single most important fix** — without it the eval-res deferral cannot be reproduced at full res and eval output can't be told apart from wallpaper output.

**Diversity metadata — captured, no readout tool.** Emit tags **family, color_cell (CIELAB), palette, location** per row; tail tags **render mode**; discovery tags **family**. Enough to measure diversity on every axis the run cares about — **except an explicit warm-fraction** (only the discretized Lab cell) and per-row render mode (emit is smooth-only). **No tool reads `emit_v1/manifest.jsonl` for a diversity readout** — the selector prints `per_family_spread`/`palette_hist`/`cells_filled` to stdout only, not persisted. A manifest-level readout (family × Lab-cell × palette histogram + warm-fraction) **must be built** (not built here; the tags exist to build it).

**GPU contention — NON-ISSUE; tonight can proceed.** The "blind-spot probe" is **not running**: its output (`scratchpad/blindspot/probe/rows.jsonl`) is **3 h stale**, no torch/python compute process is resident, and the GPU shows **868 MiB used / 17% util** (desktop/Chrome only — a loaded torch model would use GBs). The probe is a one-shot staging pass, already finished. The real contention is **internal**: all four pipeline stages use the single 8 GB GPU, which is why Phases A/B/C must be **GPU-exclusive** (do not overlap them). No need to wait for anything.

**Disk — ample headroom.** Free: **148 GB** on C:. Projected 6h footprint at eval-res: discovery ledger+feats <50 MB (tiny rows; guard-field renders transient); emit field dumps are **unlinked immediately** (peak ~9 MB one-at-a-time at eval-res), eval-res PNGs ~0.8–1.0 MB each (×~100 ≈ <150 MB); tail scoring crops disposable, keeper full-res PNGs ~5 MB × ≤0.25·N. **Total well under 1 GB.** Disk is not a constraint even if some renders run full-res.

---

## End-to-end smoke (eval-res) — PASS

Ran a tiny all-stages pass proving the handoffs connect and timing each, output redirected to `scratchpad/preflight/smoke/` (production emit home untouched — verified 10 manifest rows + 2 alternates intact after; `emit_v1.py` patch reverted via `git checkout`).

- **S3 emit @ eval-res 1024 ss2** (`--pool 2026-07-07_wallpaper_fresh_discovery_v1 --limit 2`): gate scored 364 crops in **11 s** (v3 head, cuda), 182/364 passed p_ge3>0.9, MAP-Elites selected **21** winners (cells 21/23, 13 palettes), rendered 2 at 1024 ss2 in **~12.5 s/emission**. Wall **52 s**. Manifest + contact sheet written to scratch. → **pool→emit handoff + eval-res render confirmed; gate res-invariance confirmed.**
- **S4 deploy-tail** (`--manifest scratch/manifest.jsonl --score-only`): read the scratch manifest, enumerated 2 locs × 7 promoted modes = 14 candidates, rendered them in **247 s** (~17.6 s/candidate), scored with the locked mining_v1 gate (cuda), allocated **B=0 (round(0.25·2))** with floors dormant — the **documented degenerate-small-N** behavior (floors bite at N≳36, far above this smoke's N=2, exactly as the prompt anticipated). `smooth-untouched / field-clean / idempotent` checks **ALL PASS**. Wall **256 s**. → **emit→tail handoff confirmed.**

Handoffs connect end-to-end; the only manual bridge is `build_fresh_discovery` (ledger→pool), demonstrated by the existing 2026-07-07 pool batch consumed above.
