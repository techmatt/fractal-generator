# Campaign-1 stratified blind read (298 labels)

Human quality labels over the campaign-1 admissions (breadth 314 + dive 254 = 568;
`data/discovery/campaign1/{breadth,dive}`). Blind read of 298 tiles = **all 98 julia
(census)** + **200 non-julia stratified** by family × leg × pref_loc_v0 tercile (top
slice + uncertainty band + stratified-random). twilight_shifted 640×360 canonical tiles,
shuffled, no metadata shown. Scores 1/2/3 = bad/okay/good (169 / 92 / 37). Rates below are
**post-stratified** to family population (raw ≈ post-strat, within 1–2 pp); julia is census.

## 1. Per-family human good / bad rate

| family | admissions (share) | mean p_good | human good | human bad |
|---|---|---|---|---|
| multibrot3 | 163 (29%) | 0.678 | 7.6% | 67.3% |
| multibrot5 | 149 (26%) | 0.661 | 7.6% | 64.3% |
| mandelbrot | 113 (20%) | 0.711 | 13.9% | 46.5% |
| multibrot4 | 45 (8%) | **0.839** | **4.8%** | 64.8% |
| julia:multibrot3 (jm3) | 50 (9%) | 0.616 | 10.0% | 66.0% |
| julia:multibrot4 (jm4) | 26 (5%) | 0.783 | 23.1% | 26.9% |
| julia:multibrot5 (jm5) | 18 (3%) | 0.607 | **33.3%** | 38.9% |
| julia:mandelbrot | 4 (1%) | 0.598 | 25.0% | 50.0% |

Leg split (pop-weighted): **breadth good 7.8% / bad 63.8%; dive good 13.7% / bad 53.3%.**
Overall good 12.4% / bad 56.7%.

## 2. Headline — the multibrot question: **v7-flattered, NOT a rich vein.**

The campaign was multibrot-heavy by construction (mb3/4/5 = 63% of admissions vs mandelbrot
20%). That concentration was driven by v7 p_good — and v7 is exactly **wrong** about it:

- The family v7 was **most** confident in, multibrot4 (mean p_good **0.839**, highest), is
  the **worst** by human taste (good **4.8%**, bad 64.8%).
- Across the 8 families, **Spearman(mean p_good, human good-rate) = −0.57** — machine
  family-confidence *anti-correlates* with human quality. Textbook winner's-curse.
- Multibrot as a whole: good 7.3% / bad 65.7% — **half** mandelbrot's good rate (13.9%) at a
  much higher bad rate.

**The genuinely rich vein is Julia** — jm5 33%, julia:mandelbrot 25%, jm4 23% good — which the
campaign hooked as a scarce commodity and under-produced (18% of admissions). Verdict: reweight
generation away from the multibrot bulk toward Julia; do not trust p_good's *cross-family* mass.

## 3. Dive-at-scale

Prior small read (steered_v1_2_dive, n=21): ~38% good / ~5% bad, depth-doesn't-dilute. At
campaign scale (dive leg, pop 254):

- **The 38/5 does NOT replicate** — dive good 13.7% / bad 53.3%. That prior was a different
  generator + far more favorable family mix (this dive leg is multibrot-dominated, the bad vein).
- **But dive still beats breadth** apples-to-apples (good +5.9 pp, bad −10.5 pp): diving improves
  both rates within the campaign.
- **Depth-doesn't-dilute HOLDS.** dive good by depth: shallow(≤3) 25% (n=8) · mid(4–8) 12.8% ·
  deep(>8) 15.1%; bad rises only 52.6%→58.5%. Good-rate is flat-to-rising with depth — depth is
  not the quality lever, family is.

## 4. Ranker — pref_loc_v0 fresh eval, and pref_loc_v1

**pref_loc_v0 on campaign1 (fully out-of-sample, n=298):** Spearman(pref, human) **0.222**
(p=1e-4), AUC good-vs-rest 0.682. Pooled ranking is a touch *under* canon_pgood (0.291) but the
**within-family** story — v0's stated growth target — is the win: **every family now positive**,
incl. **jm3 +0.277** (v0 memo had jm3 ranking its bad above its good). Cross-family compression is
what drags the pooled number; within-eligible ordering (the actual consumer) improved.

**pref_loc_v1 — refit on all 379 accumulated labels (run2 60 + dive 21 + campaign1 298),
3-batch LOBO.** Winner **v7+colored:logi** (unchanged recipe; morph-free — campaign1 has no morph,
which v0 rejected anyway; no corpus prior, also rejected):

- mean-LOBO Spearman **+0.436** (AUC 0.765), vs canon_pgood **+0.115** / random +0.087.
- Pooled pct-normalized Spearman **+0.279**, 95% CI **[+0.185, +0.371]** (excludes 0), perm p **0.0002**.
- Per-family LOBO all positive: jm3 **+0.346**, jm5 +0.502, mb5 +0.267, mb3 +0.250, mb4 +0.244,
  jm4 +0.274, mandelbrot +0.144.
- **CERTIFIED → shipped provisional** to `data/ranker/pref_loc_v1/` (model + features + metrics).
  Scope unchanged: ranks the not-bad, never steers; keeper / emission-intake / dive-sort only.

## Artifacts
- Read: `out/campaign1_blind/` (tiles, blind_index, hidden manifest_key, blind_label.html);
  labels `labels/campaign1_blind_blind_scores.json` (tile-keyed).
- Features/scores: `data/ranker/campaign1/features.npz` (v7+colored, all 568); registry record
  `data/ranker/campaign1/blind_read.json`.
- v1: `data/ranker/pref_loc_v1/{model,features,metrics}.npz/json`.
- Builders: `tools/atlas/campaign1_manifest.py`, `tools/ranker/train_eval_v1.py`.
