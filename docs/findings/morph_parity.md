# morph_clip producer parity — recovering the canonical robust-z transfer

**Context.** The palette-BLIND grayscale morphology descriptor (`morph_clip`, the primary
axis for within-family visual dedup) was originally produced by `scratchpad/visual_dup/embed.py`.
That file was never committed and was lost. The base store
(`data/library_embeddings/embeddings.npz`, `morph_clip` = 62 curated rows) survived, but the
*producer* that generated it did not — so any new location could not be embedded into the same
space, and the whole dedup axis was stranded. The `library_gap_report.md` §C5 flagged this as a
dangling-reference risk; `visual_dup.md` (this dir) is the analysis that made recovery possible.

**Question.** Is there a scalar→grayscale transfer that, fed the same cached smooth fields and
the same CLIP model/transform, reproduces the stored `morph_clip` rows to numerical identity
(self-cos → 1.0)? If yes → fix the transfer in the production producer (`library_annotate.morph_gray_image`).
If no → the original inputs are unrecoverable and the axis must be re-embedded from scratch
(re-labeling all dedup verdicts).

**Method** (`scratchpad/morph_parity/`, one-time analysis; fixture = the 47 store `morph_uids`
that still have both record identity and a cached smooth field):

- `robustz_sweep.py` — sweep median/MAD robust-z tanh formulations `tanh(gain · (x−median)/(c·MAD))`
  over `{c ∈ 1.0, 1.4826}`, `{gain}`, `{linear, sRGB}` encodings against the stored rows (cached
  fields, no re-dump).
- `parity.py` — cross-check three whole configs end-to-end (fresh 640×360 ss2 re-dump vs cached
  field; `pct` transfer vs robust-z) and confirm the reconstruction reproduces the original
  producer's **0.974 dedup verdicts** (same ≥0.974 pairs, same single-linkage clusters), not just
  cosine.
- `confirm_and_tag.py` — re-embed the fixture through the *fixed production* `morph_gray_image`
  and assert self-cos > 0.9999; then non-destructively add a `morph_producer` provenance array to
  the base store (`morph_clip`/`morph_uids` untouched).

**Result — PARITY REACHED.** The canonical transfer is:

> **robust-z tanh, MAD scale `c = 1.4826`, tanh `gain = 0.5`, computed in LINEAR light.**

Sweep row: `median self-cos = 1.000000`, `min = 0.99999988`, `max = 1.00000024` over the fixture.
The next-best variant (`c = 1.0`, linear) fell to median 0.9924 / min 0.9706 — well below the
0.974 dedup threshold, i.e. it would flip verdicts. sRGB encodings were worse still. So the
recovery is a *sharp* optimum, not a broad basin — the exact `(c=1.4826, gain=0.5, linear)` triple
is load-bearing.

`confirm_and_tag.py` then verified the production `library_annotate.morph_gray_image` (now carrying
this transfer) reproduces the store at min self-cos > 0.9999, and tagged all 62 base rows with
`morph_producer = library_annotate.MORPH_PRODUCER`.

**Disposition.** The fix lives in production (`tools/wallpaper/library_annotate.py`, committed as
`fix(library): recover canonical robust-z morph_clip transfer`). The store tagger was promoted to
`tools/wallpaper/morph_producer_tag.py`. The sweep/parity analysis scripts were one-time and are
not retained; this document is their record. The 166 MB `fields640/` re-dump cache is regenerable
and was left in scratch.
