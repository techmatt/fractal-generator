# Wiring the parity gates + the T3-a value question

Companion to `out/inventory_followup.md` §1.1 and `out/code_inventory.md` §7.
Part 1 changes code (wires off-by-default parity gates into `pytest`); Part 2 is an
argument (no §7 moves applied).

---

## Part 1 — the gates that now run

### Code changes

| file | change |
|---|---|
| `palette_extractor/check_bytematch.py` | refactor: extract `cases()` + `run()` (importable, returns worst max\|Δ\|); ASCII-safe the per-case print so it can't `UnicodeEncodeError` on a cp1252 console under `pytest -s` and mask a real diff. |
| `palette_extractor/test_bytematch.py` | **new** — pytest gate: `run() < 1e-12`, `skipif(not BIN)`. |
| `tools/v5/build_plan.py` | extract `verify_recipe_parity()` (pure check, writes nothing) + `_load_recipe_inputs()`; `main()` now delegates to it. Emitted manifests proven byte-identical before/after. |
| `tools/v5/test_recipe_parity_v5.py` | **new** — pytest gate: `verify_recipe_parity()`, `skipif` manifests absent. |
| `tools/v6/build_plan.py` | same extraction as v5. |
| `tools/v6/test_recipe_parity_v6.py` | **new** — pytest gate, `skipif` manifests absent. |

The two new `build_plan` modules are loaded in their tests via `importlib` under
unique names (`v5_build_plan`/`v6_build_plan`), not `import build_plan`, so the three
same-basename modules under `tools/v{4,5,6}` can't collide in `sys.modules`; the test
files themselves carry unique basenames so pytest's prepend-import doesn't reject them.

### Enforcement tiers — after

| # | gate | tier now | requires | proven red |
|---|---|---|---|---|
| 1.1.2 | Rust LUT bake ↔ Python `bake_lut` (`check_bytematch`) | **pytest, with-binary** | release binary | ✅ |
| 1.1.4 | v5 recipe-parity (Mandelbrot cache rows == `data/v4/cache_manifest.jsonl`) | **pytest, with-corpus** | `data/v{4,5}` manifests on disk | ✅ |
| 1.1.4 | v6 recipe-parity (frozen v5 rows == `data/v5/cache_manifest.jsonl`) | **pytest, with-corpus** | `data/v{5,6}` manifests on disk | ✅ |
| 1.1.8 | Julia band-table hand-sync (`test_julia_bands_parity`) | **pytest, with-binary** (already a real test; confirmed running) | release binary (`dump-julia-bands`) | ✅ |
| 1.1.3 | `colormap_acceptance` ≤1-LSB (already wired via `test_acceptance.py`) | pytest, with-binary | release binary | (pre-existing) |

"with-binary" is backstopped by the `test_release_binary` canary — a no-build checkout
now goes one loud red instead of N invisible skips. "with-corpus" has no such canary
because the corpus is gitignored by design; on a machine with the corpus (this one) the
gate runs, and the whole-suite run below confirms it does.

### Proven-red log (each perturbed, observed, reverted)

- **check_bytematch** — flipped one ULP of the OKLab `_M2` L-row constant
  (`1.9779984951 → …950`) in `palette_lib/coloring.py`. Gate red: worst max\|Δ\| =
  **5.564e-10** ≫ 1e-12, named the mismatch. Reverted → green.
- **v5 recipe-parity** — `SHIFT_FRAC 0.4 → 0.5` in `tools/v5/build_plan.py`. Gate red:
  `recipe drift: shift offset at loc 0`. Reverted → green.
- **v6 recipe-parity** — `SHIFT_FRAC 0.4 → 0.5` in `tools/v6/build_plan.py` (separate
  gate, proven independently). Gate red: `recipe drift: shift offset at loc 0`.
  Reverted → green.
- **julia bands** — `multibrot3` spread `10.0 → 11.0` in
  `production_seeder.JULIA_GATHER_BANDS`. Gate red:
  `multibrot3: rust=(2, 10) != python=(2.0, 11.0)`. Reverted → green.

Whole suite after all reverts: **`pytest -m "not slow"` → 174 passed, 3 deselected,
zero collection errors**; the binary-dependent gates all ran (none skipped, binary
present). `git status data/` clean — the two builder re-runs rewrote the manifests
byte-for-byte.

### Still manual / not wired — and the honest reason for each

- **1.1.1 Deploy-transform parity (`present.rs` JPG ↔ `Transform(train=False)`) —
  still enforced by NOTHING, and it is not a wire-in.** There is no oracle in the repo
  to collect: no test renders through `present.rs` and compares the 1280×720→384×224
  bicubic-stretch+normalize pixel-for-pixel. Wiring can only surface a gate that
  exists; this one has to be *built*. It is GPU-free (binary + PIL) and is the single
  highest-value guarantee with zero coverage — it should be the next task, but it is
  out of scope for "make the off-by-default gates run." Flagged, not faked.
- **1.1.5 `tests/occupancy_parity.rs` — Rust `#[ignore]`, depends on the gitignored
  `data_large/label_crops/loose0` corpus.** Cannot be default (a fresh checkout has no
  corpus, so removing `#[ignore]` would red-fail everywhere). Correctly left ignored;
  it runs and passes on this machine (corpus present). Same tier as the with-corpus
  recipe gates, except it's a Rust test so it stays `cargo test -- --ignored` rather
  than gaining a pytest `skipif` wrapper.
- **1.1.2 (Rust side) `tests/palette_bytematch.rs` — stays `#[ignore]`, correctly.**
  It is a *dump fixture*, not an assertion (it writes a LUT for Python to compare); it
  has no oracle by itself. Wiring `check_bytematch` as a pytest gate now exercises it
  transitively on every run of the new gate, which is the right shape — the assertion
  lives on the Python side where the two bakes meet.
- **1.1.6 `viz_transfer.py` parity assert — left as-is (redundant + needs cached
  fields).** It guards `transfer='pct' == transfer='grad'@γ=0` in `colormap.render_candidate`;
  per the inventory it is redundant with the bake gate + `colormap_acceptance`, needs
  cached fields on disk, and is a validation *sheet* (writes to `out/`). Guards nothing
  unique → not worth a pytest wrapper.
- **1.1.7 Mining-gate deploy parity (`lock_mining_gate.py`) — GPU.** `build_parity_block()`
  runs `MiningScorer` (torch). Can't be default; stays a manual/`--no-parity`-toggled
  QA step.
- **1.1.9 Guard tripwire (`test_guard_tripwire.py`) — `slow` ∧ binary, by design.**
  Renders 81 f64 tiles; opt-in via `-m slow`. Left as-is.

**Net:** three gates moved from "manual / never-run" to `pytest`-collected (one
with-binary, two with-corpus), a fourth (julia bands) confirmed already-wired-and-live,
and each newly-wired gate was made to go red on purpose before being trusted. The one
crown-jewel with no gate at all — deploy-transform — is named as a build task, not
wired, because there is nothing to wire yet.

---

## Part 2 — is T3-a worth doing at all?

**T3-a** (collapse the four hand-synced OKLab/coloring copies) — **not worth doing.
Leave the copies; mark them as deliberately frozen ports with a header note; rely on
the Part-1 gates to catch drift.** The case for collapsing is aesthetic; the case
against is byte-risk on every image the pipeline emits, and it's the higher-value side.

### What the copies actually are

The OKLab round-trip matrices live in (canonical first):

- `src/palette.rs` — **canonical**, the bake every rendered image goes through.
- `palette_lib/coloring.py` — extractor-side bake, held **byte-exact** to Rust
  (`check_bytematch`, <1e-12).
- `tools/colormap.py` — the deploy coloring tail (~40-importer hub), held to **≤1 LSB
  after a full render** (`colormap_acceptance`, `TOL_MAX=2`) — a *looser* contract than
  byte-exact, and deliberately so.
- `tools/palettes/color.py` — forward-only (sRGB→OKLab), feeds palette categorization.

(Two more forward-only touches in `palette_extractor/palette_extract.py` and
`tools/mining/palette_families.py` — the "four copies" is the round-trip set above.)

### What collapsing buys — name it in a sentence

Less duplication of ~10 matrix literals. That's the whole ledger. It does not make any
image more correct, faster, or newly-correct-under-refactor: the numbers are already
identical where it matters (byte-exact) or intentionally within-tolerance where it
doesn't (the deploy tail's 1-LSB slack).

### What leaving them costs — also a sentence

A header note per file and the Part-1 gates staying green. Nothing breaks or slows.
The copies are *ports of a frozen upstream* (Ottosson's constants); a "frozen port"
duplicated behind an enforced byte-identity gate is not tech debt — it is the
normal shape of a cross-language numeric contract.

### Why collapse is actively the wrong trade here

1. **It's the single highest byte-risk item in the plan and buys the least.** LUT
   byte-exactness feeds every emitted image and is asserted in four places. Trading a
   guaranteed-safe state for a cosmetic dedup, on the one axis where a silent 1-ULP
   drift is a whole-corpus regression, is negative expected value.
2. **The two Python copies are held to *different* contracts on purpose.**
   `palette_lib/coloring.py` is byte-exact to Rust; `tools/colormap.py` is ≤1-LSB
   post-render. Merging them onto one shared bake (T3-a's "should `coloring.py` share
   the Rust-parity bake?") risks tightening `colormap.py` into a byte contract it was
   never required to meet, or loosening `coloring.py` out of the one it *is* required
   to meet. The duplication is encoding a real distinction, not an accident.
3. **The reason the copies were safe to have is now *enforced*, not assumed.** Before
   Part 1, "these four stay in sync" was guaranteed by a script nobody ran and a code
   comment. After Part 1 the byte-identity is a `pytest` gate that we watched go red on
   a 1-ULP perturbation. That removes the *only* real argument for collapsing (drift
   risk) — the gate catches drift without touching the bytes.

**Recommendation:** cut T3-a. Replace it with T0-style header notes on
`palette_lib/coloring.py`, `tools/colormap.py`, `tools/palettes/color.py`:
"Frozen OKLab port of `src/palette.rs`; byte-identity enforced by
`palette_extractor/test_bytematch.py` / `tools/test_acceptance.py`. Do not 'refactor'
to dedup — the copies are a deliberate cross-language contract." Cost: one comment
each. Guarantee: identical. Byte-risk: zero.

### Other §7 items whose value doesn't survive the same test

- **T2-e (consolidate root `tools/*.py` into `tools/color/`) — defer / probably cut.**
  Value: package tidiness. Cost: `colormap` is the ~40-importer hub — the largest
  import blast radius in the repo — and the rename has no functional payoff. Same shape
  as T3-a: aesthetic reward, real risk. If done at all, script it with a grep sweep and
  gate on `test_acceptance`; but "tidiness" alone doesn't justify touching a 40-site
  hub.
- **T1-a (rename the `test_`-prefixed non-tests) — keep, it's genuinely worth it.**
  Value is nameable: those files are collected by pytest and assert nothing, so they
  read as green coverage that doesn't exist. Renaming off `test_` removes a standing
  false-safe signal — that's the exact failure mode this whole exercise is about. Low
  cost, real value.
- **T3-b (archive closed studies) — keep, but it's bookkeeping, not machinery.** Value:
  shrinks the ambiguous bucket. It builds nothing and risks nothing (archive, don't
  `rm`); it's fine precisely because it *doesn't* add machinery to make low-value items
  safe — it removes them from the active surface.

The through-line for both parts: the restructure's byte-sensitive items (T3-a, T2-e)
are worth *less* now than when the plan was written, because Part 1 converted their
implicit safety into an enforced gate. Don't build machinery (or spend byte-risk) to
make a low-value dedup safe — enforce the invariant cheaply and leave the bytes alone.
