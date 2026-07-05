# v6 gather batch — detect too-dark-palette crops (diagnose, no re-render)

## Goal
Some crops in the `2026-07-05_gather_v6` label batch are rendered under palettes so dark the location can't be judged — the render is unlabelable regardless of the brightness augmentation applied at train time. Detect those crops and report them for a follow-up re-render pass. **Diagnose only — do not re-render anything here.** The "too dark to label" cut is a by-eye call; this pass produces the material to make it.

## 1. Palette-pool check (settle the expansion hypothesis)
From the batch `images.jsonl` provenance, collect the palette actually used per crop. Locate the palette pool this batch drew from and the palette set used for the **v1→v5** corpus renders (both should be on disk). Report:
- Whether the batch's pool expanded beyond the v1→v5 set — i.e. palettes used here that weren't available at v1→v5.
- A brightness measure per palette (palette-space, e.g. mean/median luma of the palette's colors), and whether the dark **renders** (§2) correlate with the newly-added palettes vs. being spread across the whole pool.

This answers: did an expanded pool introduce the dark palettes, or is it unlucky draws from the original pool?

## 2. Image-space darkness detection (interior-robust)
Crops render `interior_mode=black`, so raw mean luminance is confounded by interior fraction — a high-interior location is full of black pixels unrelated to the palette. Use an **interior-robust** measure:
- Preferred: mask the interior via the crop's field (non-escaping pixels) and measure luminance over **escaped pixels only**.
- And/or an **upper-percentile luminance** (P90 and P95 of pixel luma) — robust to interior mass, since a dark *palette* pulls even the brightest regions down while a bright-but-interior-heavy crop keeps a high upper percentile.

Report both if cheap. Compute over all 640 crops and rank darkest-first.

## 3. Contact sheet + histogram (for the by-eye cut)
- A **montage of the darkest ~80 crops**, each annotated with its darkness measure(s), palette id, and class — so the cut line can be placed by eye.
- A histogram of the darkness measure across the batch.
- A **suggested provisional threshold** (where the distribution/eyeball suggests the line sits), clearly marked as provisional — the final cut is Matt's.

## 4. Label cross-reference
At the provisional threshold, report how many flagged-dark crops are **already labeled** vs unlabeled, and the class breakdown. (These are the existing labels made under an unfair palette that will want revisiting after re-render.)

## 5. Output
Write a ranked list to a file: `image_id, darkness measure(s), palette_id, class, labeled?(+score)`, darkest-first. Plus the contact sheet and histogram. No re-rendering, no changes to the batch — this pass only reports.
