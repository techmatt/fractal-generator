# Palette library prototype — report

Python prototype (Rust port deferred). Downloader → importer → sampler → contact
sheet. Goal: a large diverse palette library sampled to put a *distribution* of
palettes on output. Built nothing past the contact sheet.

## Diagnosis

- **`.ugr` parser port status.** The original prototype's `coloring.py` does **not
  survive** — no `.py` anywhere in the repo or git history; `nu_deep.npy` is gone
  too. The *validated* parser + OKLab/LUT path now lives in **Rust**
  (`src/palette_io.rs`, `src/palette.rs`, `src/coloring.rs`). `palette_lib/coloring.py`
  is a faithful port of that reference: `parse_ugr`/`parse_map`, the Ottosson
  OKLab matrices (constants copied verbatim), cyclic OKLab interpolation, the
  4096-entry linear-RGB LUT bake, reverse, and `t = value·density + offset`.
  Verified against the Rust unit-test values: OKLab round-trip max sRGB error
  1.2e-6 (Rust bound 1e-4); `.ugr` COLORREF 7880 → (200,30,0); bake passes
  through stops to <5e-3 linear error.
- **Value-field.** No saved field survives, so `palette_lib/field.py` generates
  Mandelbrot smooth-iter crops (seahorse valley + a spiral). Palettes are the
  point; any structured field exposes a palette's character.
- **Reachable sources** (probed 2026-06-21; not hard-coded blindly — initial
  guessed `.ugr` URLs 404'd):
  - `matplotlib` 3.11 / `colorcet` 3.2.1 / `cmasher` 1.9.2 — pip, installed. The
    clean, zero-restriction backbone.
  - `fract4d/gnofract4d` `maps/` via `raw.githubusercontent.com` — 206 Fractint
    `.map` + 2 UltraFractal `.ugr` (~100 blocks). Reachable. **Third-party
    harvest** → cached under gitignored `palette_cache/`.
  - UltraFractal's formula DB (`formulas.ultrafractal.com`) did not resolve (SSL).
    Artist DeviantArt packs are per-artist copyright — deliberately not scraped;
    hand-drop into `palette_cache/harvest/` if wanted.

## Counts (per source, raw → survivors)

| source     | raw | ok  | sparse | busy | dup |
|------------|-----|-----|--------|------|-----|
| cmasher    |  57 |  57 |   0    |  0   |  0  |
| colorcet   | 213 | 101 |   0    |  0   | 112 |
| matplotlib |  88 |  83 |   0    |  0   |  5  |
| map        | 206 | 197 |   0    |  9   |  0  |
| ugr        | 200 | 200 |   0    |  0   |  0  |
| **total**  |**764**| | | | |

**638 survivors** after dedup + degenerate filter. Dedup collapsed 117 (almost
all colorcet long-name ↔ `CET_*` alias pairs). 9 Fractint maps flagged
palette-space **busy** (OKLab ring total-variation above the smooth population —
near-random color sequences). No palettes hit the **sparse** tail (≤2 distinct
stops / near-one-color) in this corpus.

## Where the library lives

- `palette_cache/harvest/gnofract4d/` — downloaded third-party `.ugr`/`.map`
  (**gitignored, never redistributed** — local working input only).
- `data/palettes/clean_colormaps.json` — the **committable** clean library: 241
  colormap-derived palettes (matplotlib/colorcet/cmasher only, no harvested
  colors). Safe for publishable output.
- The full deduped 638-palette library is assembled in memory by
  `importer.build_library`; harvested colors are never written to a tracked file.

## Sampler

`sampler.Sampler` — a distribution over the library. Default uniform; `weights`
(or `Sampler.by_source`) is the exposed knob for shifting the output
distribution. Not a per-candidate optimizer. Draws are seeded (the engine
forbids nondeterminism).

## Deliverable

`out/palette_contact_sheet.png` (seahorse) and `out/palette_contact_sheet_spiral.png`
— one location × 30 uniformly-sampled palettes (seed 0), captioned with
source:name. Same palette set across both locations. **Matt judges the sheet;
no quality claims.** The spiral location shows palette character best.

## Run

```bash
python -m palette_lib.build_sheet
```

Harvest is cached (re-run is download-free). `FORCE=1 python -m palette_lib.download`
to refresh the harvest.

## Licensing

Harvested third-party `.ugr`/`.map` collections → gitignored, never
redistributed. Importer code, colormap-derived palettes, and any
corpus-extracted palettes are clean and committable. For publishable/sellable
output, prefer the clean colormap-derived sources.
