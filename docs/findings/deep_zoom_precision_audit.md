# Audit: deep-zoom / beyond-fp64 precision state

Read-only inventory (no builds, no runs). Verdicts: **present / stubbed / absent**,
with `file:line` pointers. Absence claims checked against deleted git history
(`git log --all -S`, `--diff-filter=D`).

---

## Q1 — Kernel precision: all f64, four backends + one HP orbit

**Present (f64-only per-pixel).** Every iteration kernel does its per-pixel work in
plain `f64`. Backends behind `FractalBackend::sample` (`src/backend.rs:156`):

| Backend | `file:line` | Per-pixel numeric type | Deep-zoom capable |
|---|---|---|---|
| `F64Backend` (deg-2 Mandelbrot + deg≥3 multibrot) | `backend.rs:195`, kernels `sample_flags` `:245`, `sample_multibrot` `:365` | f64 | No (shallow escape-time) |
| `JuliaBackend` (deg-2 + Julia-multibrot deg≥3) | `backend.rs:467`, kernels `sample_deg2` `:502`, `sample_multibrot` `:563` | f64 | No — base-scale only, `de=0` |
| `PhoenixBackend` (Ushiki two-state `z²+c+p·z₋₁`) | `backend.rs:647`, kernel `:683` | f64 | No — base-scale only |
| `PerturbationBackend` (single-ref + rebasing) | `backend.rs:750`, kernel `:823` | f64 δ/dz per pixel; **BigFloat only for the reference orbit** (`new` `:767`) | **Yes** (deg-2 Mandelbrot only) |

`F64Backend` is **not** the sole backend, and the degree-parametric multibrot path
is also f64. The only non-f64 arithmetic in the entire engine is the perturbation
**reference orbit** at the frame center (`backend.rs:790–807`), computed in
`prec_bits`-bit `astro_float::BigFloat` and immediately projected to f64
(`hp::to_f64`, `backend.rs:801`). Orbit *values* stay O(1), so f64 storage of the
projected orbit is exact enough (`backend.rs:744–749`).

---

## Q2 — Perturbation apparatus: present (production), narrow

**Present.** Full single-reference perturbation with **Zhuoran rebasing** is the
production deep-zoom tier:
- Reference orbit in high precision, stored as f64 projections — `PerturbationBackend::new` `backend.rs:767`.
- Per-pixel delta iteration `δ_{n+1}=(2Z[m]+δ)δ+dc` — `backend.rs:866–876`.
- Rebase (re-anchor `δ:=z, m:=0` when `|z|²<|δ|²` or ref exhausted) — `backend.rs:915–922`.
- Auto-selection by pixel spacing `PERTURB_SPACING = 1e-13` — `probe.rs:25`, dispatch `main.rs:112` / `probe.rs:111`; `--backend f64|perturb|auto` override `cli.rs:16,79`.
- Validated against f64 ground truth at shallow depth — `tests/perturbation.rs:63,88`.

**Absent / partial (the sophistication a deep-zoom engine usually adds):**
- **Series approximation (SA) / bivariate SA / BLA** — *absent*. No iteration-skipping
  anywhere; every pixel runs the full delta loop. (`git log --all -S "series"`
  returns only palette/doc hits, no numeric code.)
- **Pauldelbrot glitch detection** — *absent by design*. Glitch handling is a single
  **underflow flag**: `δ` collapsing to exactly 0 on an offset pixel sets
  `glitched=true` (`backend.rs:908–913`), noted explicitly at `backend.rs`
  module doc / `PixelSample.glitched` `:50–53`. No relative-error test, no
  multi-reference recovery, no re-render of glitched tiles.
- **Scope** — perturbation is **degree-2 Mandelbrot only**. Multibrot (deg≥3), Julia,
  and Phoenix have no perturbation path; callers at those degrees force
  `BackendChoice::F64` (`probe.rs:116–122`).
- **"floatexp" tier** — *absent*, named only as a deferred aspiration in comments
  (`backend.rs:135`, `main.rs:65`, `render_one.rs:205`). `git log --all -S floatexp`
  shows only comment churn — no implementation ever existed.

---

## Q3 — High-precision numerics

**Rust — present, single crate, wired.** `astro-float 0.9.5` (pure Rust, no C dep),
`Cargo.toml:13`. Wired only through `src/hp.rs` (decimal parse `hp.rs:49`, precision
sizing `prec_bits` `:39`, fast f64 projection `to_f64` `:66`, decimal serialization
`to_decimal_string` `:22`) and consumed solely by `PerturbationBackend::new`. **No**
`rug`/MPFR, `dashu`, `malachite`, `num-bigint`, `twofloat`, double-double/`qd`, or
`f128`/quad — neither declared nor in deleted `Cargo.toml` history
(`git log --all -p -- Cargo.toml` grep: empty).

**Python — absent.** No `mpmath`/`gmpy2`; no high-precision arithmetic on the Python
side at all. Coordinates are carried as **decimal strings** and handed to Rust
untouched (see Q5). Python never *computes* in high precision — it only transports
the strings.

---

## Q4 — Depth-limit mechanics: two floors, one is precision, one is policy

There are **two independent floors**, and the binding one for the search is a
*configured policy floor*, not a mantissa wall:

1. **Engine hard-refuse (precision wall)** — `MIN_FRAME_WIDTH = 1e-300`
   (`main.rs:35`), enforced in `iterate_location` (`main.rs:62–68`): frames below it
   are rejected with a "past the v1 magnification cap (~1e300): f64 deltas would
   underflow to denormals" error. This **is** a precision bound — the point where the
   per-pixel f64 δ underflows — and it is the true ceiling of the perturbation tier
   (~1e300 magnification).

2. **Search-side conservative f64 floor** — `1e-9`, the one said to bind ~depth 24.
   This is a **configured policy floor, not a precision limit**:
   - `FW_FLOOR = 1e-9` guards the depth-1 root-start width — `guided_descend.rs:69`,
     explicitly "comfortably clear of the f64 precision wall (~5e-12)… No real
     deep-zoom handling (perturbation is the search's job)."
   - `--min-fw` (default `1e-9`) truncates descent steps — `guided_descend.rs:2771`;
     its doc (`:2762–2770`) states it sits "well above the ~1e-13 f64 cliff and far
     below anything current walks reach," chosen to keep walks inside the *tested*
     f64 regime, and is "Tunable/liftable later."
   - The Python steered-descent mirror uses a **dive stop-margin `2e-9`**
     (`--dive-min-fw`, `tools/atlas/steered_frontier.py:453,1557`): stop before a
     zoom would cross the floor.

   **Behavior at this floor is graceful, not a hard stop:** the walk is truncated,
   its already-collected candidates are **harvested**, and the event is counted as
   `min_fw_floor` in `summary.json` (`guided_descend.rs:181,890,1042,1257`). No
   degraded pixels, no error.

**So:** the search deliberately stops (~1e-9) *three-plus orders of magnitude short*
of the f64 auto-switch cliff (~1e-13, `PERTURB_SPACING`) and ~291 orders short of the
actual perturbation ceiling (~1e-300). The limit that bites today is policy, and the
machinery to go far deeper (perturbation) already exists but the search never engages
it.

---

## Q5 — Coordinate representation

**Center: arbitrary-precision-capable (decimal strings).** The canonical
`Location` (`tools/corpus/location.py:92`) carries `cx`, `cy`, `fw` as **`str`**
(dataclass fields `:99–101`), explicitly "Coords are decimal strings (an f64 is
meaningless at deep zoom)" (`:94`). Strings round-trip through the whole pipeline
(pool → key `location_key` `:135` → `render_one_flags` `:189` → sidecar parse
`from_sidecar` `:235`) with **no f64 narrowing on the Python side**. Rust re-parses
them to `BigFloat` at `prec_bits` (`main.rs:70–72`, `hp::parse_decimal`) for the
reference orbit; the absolute center is projected to f64 only for the shallow f64
backend (`main.rs:73`). `dc` (pixel offset) is formed from pixel geometry in f64,
never `c−center` (the critical-coordinate rule).

**Frame width: f64.** `fw` survives as a decimal string through Python, but inside
Rust it is a plain `f64` (`loc.frame_width`, `main.rs:62,70,77`). This is the field
that must widen for a below-1e-300 descent; today it is f64-typed end to end in the
kernel path.

---

## Present-vs-missing summary (the gap, not a build plan)

**What exists.** A working beyond-f64 descent tier is *already present* for degree-2
Mandelbrot: single-reference perturbation with Zhuoran rebasing, high-precision
(astro-float) reference orbit, decimal-string center coordinates that carry arbitrary
precision losslessly through the pipeline, and spacing-based auto-selection at 1e-13.
The engine can, in principle, render to ~1e300 magnification (fw ~1e-300).

**What a deeper / broader descent still needs, based only on what was found:**

1. **Engage the tier that already exists.** The search self-limits at `fw≈1e-9`
   (`FW_FLOOR` / `--min-fw`), ~4 orders above the f64→perturbation switch and ~291
   orders above the perturbation ceiling. Descending past ~1e-9 is a
   **plumbing/config change** (lift the floor, let auto-selection pick perturbation,
   carry `fw` deeper) — no new numerics required to reach ~1e-300.

2. **A `floatexp`/scaled-double delta type** to pass ~1e-300. The current wall
   (`MIN_FRAME_WIDTH=1e-300`, `main.rs:35`) is real: per-pixel f64 δ/dz underflow to
   denormals. Named as deferred in three comments; **no code exists**. This is the
   one genuinely-missing numeric component for arbitrarily deep zoom.

3. **Series approximation / BLA.** Absent. Without it, every deep pixel pays the full
   iteration count — deep renders are correctness-OK but slow (no iteration skipping).

4. **Real glitch handling.** Only an underflow flag (`glitched`) exists — no
   Pauldelbrot detection, no multi-reference recovery. Deeper/more-chaotic frames
   that current shallow validation never exercises may show unrecovered glitch pixels.

5. **Perturbation for non-Mandelbrot families.** Multibrot (deg≥3), Julia, and
   Phoenix are f64-only and shallow-only. Deep zoom in those families needs their own
   reference-orbit + delta recurrences (none stubbed).

6. **`fw` widening.** `frame_width` is f64 in the kernel path; a below-1e-300 descent
   needs it (and the delta arithmetic) in the floatexp representation of item 2.
