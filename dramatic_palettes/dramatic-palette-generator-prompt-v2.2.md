# Dramatic Fractal Palette Generator — v2.2

> Paste this whole file into claude.ai, attach **three** example fractal renders, fill in the
> **Run conditioning** block, and send. Claude returns a JSON array of palettes.
>
> **v1 → v2:** sharp transitions are a `cliff` (soft-ramp realization, default width 0.08); the glow must
> feather (no cliff at the value peak); each palette is tagged with a **skeleton** so the batch spreads
> structurally instead of collapsing onto one shape.
>
> **v2 → v2.1:** chroma **spans muted→vivid across the batch** ("controlled" was misread as
> "desaturated"); high-key / inverted-arc palettes require a **genuinely deep** focal dark.
>
> **v2.1 → v2.2:** stops are emitted in **OKLCH** (`[L, C, H]`), not sRGB — the downstream densifier does
> the perceptual conversion, so you reason and emit in one space and never hand-convert. The **jewel-earth
> register is now a default, not a universal law** — moods pull away from it. Rules are split into
> **mechanical** (firm) and **aesthetic** (flex for variety).

---

## Run conditioning (edit these four lines each run)

```
BATCH SIZE:      20              # ignored for 6-ultra — that band emits ~6 by design (see Ultra note)
MOOD FAMILY:     fire-ice        # one name from the roster below, or "span" to vary across the batch
COMPLEXITY BAND: 3-4             # one of: 1-2 / 3-4 / 5 / 6-ultra
VALUE KEY:       span            # one of: low / mid / high / span
```

Run this prompt across a grid of (mood family × complexity band × value key) to accumulate a large,
systematically diverse set. Leave a knob at `span` to let Claude vary it within the batch. **Skeletons are
distributed automatically within each batch — they are not a run knob.**

---

## Your task

You are a color designer generating **dramatic palettes** for a fractal wallpaper renderer. Output the
requested number of palettes as a single JSON array, following the schema at the end.

A palette here is a short, ordered list of **color stops** (sparse — you mark structural events, not a
dense gradient). Downstream code densifies your stops into a smooth lookup table and applies it to fractal
renders. Your palettes must be **standalone and image-independent**: concrete colors at concrete
positions, renderable on their own.

## The three attached images are calibration only

Three finished fractal renders are attached. They exist **solely to calibrate your sense of the target
quality** — the kind of color drama that reads well on fractal geometry, and how palette structure looks
once applied to a complex escape field. **Do not extract, sample, reproduce, or describe their specific
palettes.** They are examples of the *problem*, not templates for the *answer*. Generate genuinely new
palettes from the principles below.

## What the renderer does with your palette (this is why the rules exist)

The renderer treats your palette as a **1-D color ramp indexed by a smooth per-pixel "escape" value.**
Densification turns your sparse stops into a fine gradient; the fractal reads into that gradient. Three
consequences you must design around:

1. **Texture is not your job — the fractal supplies it.** The shimmering filigree, grain, and fine detail
   in a good fractal render come from thousands of adjacent pixels reading *slightly different* positions
   of a *smooth* palette region at high spatial frequency. Never bake noise, dithering, stippling, or fake
   texture into a palette — palette-noise × field-noise = mud. Provide clean gradient structure; let the
   geometry supply the frequency.

2. **A cliff becomes crisp on busy detail and soft on smooth field.** A `cliff` is realized as a narrow
   soft ramp. Where the fractal is busy (a fast-changing escape field), that ramp is spatially thin and
   reads as a crisp filigree edge. Where the field is smooth, the same ramp spreads into a soft feathered
   band. So a cliff is safe anywhere — but its drama lands as *detail accent*, not as a value wall.

3. **Never put your main value contrast on a cliff.** A bright-to-dark hard transition placed at the
   palette's peak reads as an artificial wall on smooth field (this was v1's dominant failure). Your main
   value drama goes on the **smooth glow**; cliffs are hue-boundary / mid-tone accents only.

## How to read the rules below — mechanical vs aesthetic

There are two kinds of rule, and they get different obedience:

- **Mechanical constraints** (firm) come from how the renderer consumes the palette: cyclic endpoints
  match; the glow feathers (no cliff at the peak); cliffs stay within the value cap; no baked-in texture;
  put each distinct hue on its own stop so big jumps don't interpolate through grey. Keep these.
- **Aesthetic defaults** (flexible) are strong starting points, not laws: the jewel-earth register,
  "structure over brightness," neutral-only-at-the-anchor, the chroma spread. Depart from them
  deliberately for variety — especially at high complexity or when a mood calls for it. Variety across the
  batch matters more than any single aesthetic default.

When in doubt: keep the mechanical ones, flex the aesthetic ones.

## What "dramatic" means — the palette knowledge

Drama is **structure** first — though it can also be carried by bright, saturated color. Treat the
following as strong defaults, gated by complexity where noted.

**Value structure**
- **Wide value range within one palette.** Travel from a deep dark to a near-white light. The single
  biggest source of punch.
- **Non-monotone lightness.** Do **not** march monotonically dark→light. For complexity ≥3, include **≥2
  interior lightness extrema.** Brightness should rise and fall along the ramp (the skeleton, below, sets
  where the peak sits).
- **Hued darks and hued lights.** Shadows carry a hue (oxblood, plum, navy, petrol, umber); highlights
  carry warmth or coolness (cream, gold, ivory, bone, blush). Reserve neutral grey / near-black for a
  limited number of stops, typically only the deepest anchor.

**Temperature**
- **Temperature-tension backbone.** Build around a dominant warm mass against a cool field, or the
  reverse. The core relationship is **complementary or split-complementary**, rarely a single analogous
  family.
- **Deliberate temperature reversals along the hue path.** At least **1** for complexity 3, **≥2** for
  complexity 4–5. At high complexity with many stations you'll naturally carry more — that's expected, not
  a violation.

**Chroma register**
- **Default to a jewel-and-earth register — but let the mood pull it.** The common center of gravity is
  rich mineral/jewel tones (ochre over lemon, petrol over cyan, oxblood over scarlet, deep teal, cobalt).
  This is the *default*, not a universal law: moods legitimately depart — fire-ice runs brighter and more
  saturated, pastel-iridescent lighter and cooler, orchid-twilight into saturated violet/magenta. Cover
  different hue ranges across the batch so palettes don't all feel similar.
- **"Controlled" means not *muddy* and not *cheaply neon* — it does not mean desaturated.** Full-richness
  jewel tones are welcome; so is real vividness where a mood calls for it.
- **Span the chroma range across the batch.** Don't let the batch trend uniformly muted — the common
  failure. Muted / antique palettes are welcome as *part* of the range; a good share should be **vividly
  saturated**, reaching the chroma punch of the most colorful reference. Aim for roughly a third vivid, a
  third mid, a third muted.
- **Use sparingly and only on purpose:** neon primaries and pure saturated green/cyan (they read cheap and
  digital as a default). **Avoid** muddy grey-brown midpoints — that is the one genuine failure to design
  out.

**Roles (not uniform saturation)**
- Assign each stop a role. **One or two hues carry; the rest recede.** Roles: `ground/field` (recessive
  mass), `anchor` (the defining hue), `accent` (a *narrow* high-chroma pop), `glow` (a light / metallic
  band reading as illumination).

**Glow / light band** *(mechanical: feathering)*
- Include a glow band in nearly every palette. **It must feather: the segments both *into* and *out of*
  the glow stop are `smooth`. Never place a `cliff` at the glow.** The glow is your main value-drama —
  keep it soft so it reads as light, not a blown-out wall.
- **Vary the glow's temperature across the batch:** warm (cream, gold, amber) in some, cool or neutral
  (bone, ivory, silver-grey, pale-blue, pale-peach) in others. **Do not default to warm cream every
  time.** Aim for a roughly even warm/cool split.

**Cliffs (sharp transitions)** *(mechanical: value cap)*
- **Place cliffs at hue boundaries or between mid-tones — never at the value peak or trough.** Cap the
  value jump: the two colors across a cliff must **not** be the palette's brightest and darkest; keep
  their lightness gap **|ΔL| ≤ ~⅓ of the palette's full L range**.
- Most skeletons carry exactly one cliff; `cliff-in-mids` foregrounds it; `no-cliff` has none;
  `double-peak` may have one or none.

**Hue path** *(mechanical: one hue per stop)*
- **Big hue jumps happen as cliffs or through the glow band — never as smooth rainbow arcs.** A smooth
  interpolation from one vivid hue to a distant vivid hue passes through desaturated grey at the midpoint.
  Put every distinct hue on its **own stop**, and make large hue changes either a cliff or a pass through
  a pale/glow tone.

**Cycling** *(mechanical: endpoints match)*
- The palette is **cyclic**: first and last stop share the **same `oklch`** so it loops seamlessly.

## The flat anti-pattern — do NOT produce this

The failure mode is the **perceptually-uniform, single-hue-family, monotone-lightness sequential ramp**
(the scientific-colormap species: a smooth march from one dark end, through one hue, to one bright end).
On fractal geometry these render **flat** — the whole image collapses to one temperature with a dark
set-body and a monotone falloff. If a palette you're about to emit could be described as "a smooth
gradient from X to Y," start over.

## Structural templates (skeletons) — distribute the batch across all six

To force structural variety (not just hue variety), every palette is tagged with one of six **skeletons**,
and the batch is spread **roughly evenly** across them (≈3–4 each). You choose which palette takes which —
just spread them; do not let the batch collapse onto one shape. A skeleton describes the **shape of the
lightness arc and where the sharp event sits** — it is orthogonal to mood, value_key, and complexity.

1. **peak-early** — the lightness peak (glow) sits in the first third (~0.20–0.40); the palette descends
   from it through the anchor into hued dark across the long remainder.
2. **peak-late** — the glow sits in the last third (~0.70–0.88); a long rise from the dark ground through
   the anchor to a late glow, then a short return to loop.
3. **double-peak** — two separated lightness maxima (e.g. ~0.30 and ~0.72) with a darker trough between:
   two bright events, not one.
4. **cliff-in-mids** — defined by a single sharp `cliff` at a **mid-tone hue boundary** (roughly
   0.25–0.55) between two non-extreme colors; the glow is elsewhere and feathers smoothly.
5. **no-cliff** — no sharp edge anywhere; all segments `smooth`. Drama comes entirely from ≥2 temperature
   reversals and ≥2 lightness extrema. (Proves a palette can be dramatic without a cliff.)
6. **inverted-arc** — a pale / bright field carries the palette and the **dark is the event**: a hued-dark
   trough or dark accent is the focal moment against light surroundings. Pairs naturally with high
   value_key. **The focal dark must be genuinely deep** (deep navy, oxblood, deep teal) — a mid-value
   petrol/teal washes out against the pale field and the event vanishes.

Tag each palette's `skeleton`.

## The axes you're spanning

- **complexity** (`1`–`6`): how many structural events, and thus how much fractal busyness the palette
  suits. Station-count targets:

  | complexity | stations | structure |
  |---|---|---|
  | 1 | 4–5   | 1 anchor, 1 glow, hued dark, smooth washes |
  | 2 | 5–6   | + 1 accent or 1 cliff |
  | 3 | 6–7   | anchor + support + accent + glow, ≥1 temperature reversal, ≥2 lightness extrema |
  | 4 | 7–9   | ≥2 reversals, 1 cliff (skeleton-dependent) |
  | 5 | 8–10  | full hue path, ≥2 reversals, ≥2 lightness extrema |
  | 6 (**ultra, probe**) | 11–15 | push past the useful ceiling deliberately — every stop a *real event* |

  > **Ultra note:** complexity 6 is a probe to find where hand-authoring stops adding genuine structure
  > and starts merely sampling a curve. **Emit ~6 palettes regardless of BATCH SIZE** (one per skeleton is
  > a good spread). If a stop is just a midpoint between its neighbors, delete it.

- **value_key** (`low` / `mid` / `high`):
  - `low` — bottoms out in a deep hued dark (or near-black at the set body); full value sweep to bright.
  - `mid` — mid-toned field, darks and lights both present.
  - `high` — **no blacks.** A pale cream / blush / bone field carries the image, and a **single deep,
    genuinely-dark hued jewel tone (deep navy, oxblood, deep teal — dark enough to *be* the event, not a
    mid-value petrol/teal) does the anchoring in place of black.** Drama from temperature tension + that
    dark anchor, not from a value sweep. (Natural home for the `inverted-arc` skeleton.)

- **mood family** (categorical, drives batch diversity): pick per the run knob.

## Mood roster

Pick the one named in the run block (or span several if `span`). Each is a hue inventory + where the
drama lives — realize it freshly.

- **fire-ice** — molten warm core (orange, amber, rust) besieged by a cool field (cobalt, ice-blue,
  white); complementary tension is the point.
- **jewel-earth** — oxblood, petrol, ochre, olive, deep teal; rich mineral tones, hued darks, nothing
  brighter than cream.
- **atmospheric-deep** — deep-space navy and teal across a vast dark field with a single small luminous
  warm core.
- **antique-faded** — wine, slate-blue, bone, dusty rose; candlelit and desaturated, anchored by a hued
  dark and lifted by a pale glowing core.
- **high-key-luminous** — pale cream/blush field, no blacks; one saturated jewel tone does all the
  anchoring.
- **pastel-iridescent** — opalescent lilac, sky, peach, mint with silver-grey filigree and small
  saturated pops; soft but held by the metallic mid.
- **autumn-ember** — rust, burnt gold, umber, smoke, with a cream glow; warm-dominant but broken by one
  cool ash accent.
- **oceanic** — petrol, teal, foam-white, deep indigo, cut by a warm coral or amber accent.
- **orchid-twilight** — magenta, violet, plum, indigo, with a gold or chartreuse glint and a pale core.
- **verdigris-copper** — aged-metal oxide teal-green against warm copper and cream; complementary metal
  tension.
- **ember-in-ash** — near-monochrome smoke and charcoal with one fierce warm cliff and hued highlights;
  tonal restraint plus one violent event.
- **tonal-restrained** — one dominant hue family across a wide value range, rescued from flatness by one
  temperature-shifted accent and a real glow band. (Hardest to keep dramatic — lean on value extrema.)

## How to reason before you emit

Reason **and emit in OKLCH** — no sRGB conversion, the densifier handles it. Per palette:

1. **Pick the skeleton** (spread across the batch) — it sets where the glow peak sits and whether/where a
   cliff lives.
2. **Lightness ramp (L):** place the darks, the glow peak(s) per the skeleton, and any secondary extrema.
   Confirm non-monotone for complexity ≥3.
3. **Chroma envelope (C):** where chroma peaks (accents, anchor) and drops (grounds, glow). Hold the
   mood's register; keep C within gamut (see below).
4. **Hue path (H):** hue stations in order, temperature reversals placed; big hue jumps on cliffs or
   through the glow.
5. **Roles, glow, cliff:** assign role + segment. **Glow feathers (smooth both sides).** Any cliff sits at
   a hue boundary / mid-tone with |ΔL| within the cap.
6. **Loop check:** first and last stop share the same `oklch`.

Vary **skeleton, glow temperature, cliff placement, anchor hue, value key, and chroma (muted→vivid)**
across the batch so the palettes are not variations of one idea — even within a single mood family.

**OKLCH hue reference (degrees):** red/oxblood ~25–30 · rust/amber ~50–70 · ochre/gold ~80–95 · olive
~110–120 · green ~140 · teal/petrol ~180–200 · cobalt/blue ~250–265 · violet ~290–310 · magenta/plum
~330–350.

**Gamut:** achievable sRGB chroma is roughly **C ≤ ~0.13** for most hues at mid lightness (a bit more for
some reds/blues), and falls toward **0 as L→0 or L→1** (very dark and very light stops must be low-chroma).
Keep C in gamut so you get the color you intend — the densifier clamps out-of-gamut stops to the nearest
displayable color, which desaturates them.

## Output schema

Output a short plan (2–4 sentences on how you'll span the batch, including the skeleton distribution),
then a **single fenced ```json code block** containing an array of palette objects. Nothing after it.

```json
[
  {
    "name": "Olive Reliquary",
    "mood": "deep teal ground rising through a mid olive→oxblood snap and an ochre climb to a cool bone flare, petrol-blue on the descent",
    "skeleton": "cliff-in-mids",
    "axes": {"value_key": "mid", "complexity": 4, "temperature": "cool-ground / warm-core, two reversals"},
    "cycle": "cyclic",
    "stops": [
      {"pos": 0.00, "oklch": [0.34, 0.05, 195], "role": "ground", "segment": "smooth", "keypoint": null},
      {"pos": 0.16, "oklch": [0.50, 0.07, 215], "role": "field",  "segment": "smooth", "keypoint": null},
      {"pos": 0.34, "oklch": [0.58, 0.09, 115], "role": "mid",    "segment": "cliff",
        "keypoint": {"type": "hue_flip", "salience": 0.7}},
      {"pos": 0.36, "oklch": [0.45, 0.12,  28], "role": "anchor", "segment": "smooth", "keypoint": null},
      {"pos": 0.52, "oklch": [0.70, 0.12,  78], "role": "accent", "segment": "smooth",
        "keypoint": {"type": "accent_pop", "salience": 0.5}},
      {"pos": 0.70, "oklch": [0.93, 0.03,  90], "role": "glow",   "segment": "smooth",
        "keypoint": {"type": "glow_band", "salience": 1.0}},
      {"pos": 0.86, "oklch": [0.54, 0.06, 240], "role": "mid",    "segment": "smooth",
        "keypoint": {"type": "hue_flip", "salience": 0.4}},
      {"pos": 1.00, "oklch": [0.34, 0.05, 195], "role": "ground", "segment": "smooth", "keypoint": null}
    ]
  }
]
```

**Field definitions**
- `name` — short evocative label, unique within the batch.
- `mood` — one-line moodboard sentence: hue inventory + where the drama lives.
- `skeleton` — one of: `peak-early` | `peak-late` | `double-peak` | `cliff-in-mids` | `no-cliff` |
  `inverted-arc`.
- `axes.value_key` — `low` | `mid` | `high`.
- `axes.complexity` — integer `1`–`6`.
- `axes.temperature` — short description, e.g. `"cool-ground / warm-core, two reversals"`.
- `cycle` — always `"cyclic"` (endpoints match).
- `stops` — ordered, first `pos`=0.0, last `pos`=1.0, strictly increasing; count within the complexity
  band's station range.
  - `pos` — float 0.0–1.0.
  - `oklch` — `[L, C, H]`. **L** lightness 0–1; **C** chroma ≥ 0 (in gamut per above); **H** hue 0–360°.
    **First and last stop `oklch` identical.**
  - `role` — `ground` | `field` | `mid` | `anchor` | `accent` | `glow`.
  - `segment` — how this stop transitions **to the next**:
    - `smooth` — perceptual (OKLab) lerp. **The glow stop uses `smooth` on both sides.**
    - `cliff` — a fast transition realized as a soft ramp (default width **0.08** of cycle-position):
      crisp on busy detail, feathered on smooth field. Optionally add `"width": <float>` to this stop to
      sharpen (~0.04) or soften (~0.10); keep ≤ 0.12 (above that it over-blooms).
    - `ease` — smoothstep-eased lerp (slow-in/out).
    - The final stop has no successor; set `smooth`.
  - `keypoint` — `null`, or `{"type": ..., "salience": 0.0–1.0}` marking a dramatic event. Only mark real
    events; grounds and washes are `null`.
    - `type` — `value_cliff` | `hue_flip` | `glow_band` | `accent_pop` | `shadow_drop`.
    - `salience` — importance **relative to others in the same palette**. Reserve `1.0` for the single
      most defining moment (usually the glow band).

## Before you output — self-check

Per palette (mechanical checks are pass/fail; aesthetic ones are "did you honor the intent"):
- **[mechanical]** First and last `oklch` identical? `pos` strictly increasing 0→1? Station count fits the
  complexity band?
- **[mechanical]** Glow feathered — `smooth` on both sides, never a cliff at the peak?
- **[mechanical]** Every cliff at a hue boundary / mid-tone, |ΔL| ≤ ⅓ of the L range, never
  brightest↔darkest?
- **[mechanical]** Big hue jumps on their own stop (cliff or through the glow), not smooth arcs through
  grey?
- Non-monotone lightness (complexity ≥3, ≥2 extrema)? Hued darks *and* hued lights?
- Temperature reversal (≥1 at c3, ≥2 at c4–5)?
- High-key / inverted-arc: is the focal dark genuinely deep (not a mid-value petrol/teal)?
- Could it be described as "a smooth gradient from X to Y"? → **the flat anti-pattern; redo.**
- Any muddy grey-brown midpoint, or neon used as a default rather than on purpose? → fix.

Across the batch:
- **Skeletons spread across all six templates (≈3–4 each), not collapsed onto one shape?**
- **Glow temperature split warm/cool — not warm cream every time?**
- **Chroma spanned muted→vivid — NOT uniformly muted; a real share vividly saturated?**
- **Hue ranges and moods varied enough that palettes don't all feel similar?**
- **Cliff placements, anchors, and value keys varied — not variations of one idea?**
