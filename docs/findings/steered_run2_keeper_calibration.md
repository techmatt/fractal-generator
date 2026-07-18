# Steered run2 — keeper-bar calibration vs the blind human read

60 blind-scored tiles from the stratified manifest. Human labels: **26 bad / 24 okay / 10 good**. All tiles are discovery-admitted q3, so label==1 (bad) is the discovery false-positive on steered output. The set is stratified across p_good x depth x morph-cluster (good for cut calibration, biased for population rates).

## A. Does canonical p_good track the human judgement?

Spearman(p_good, human label) = **+0.398** over n=60. Spearman(p_good, human-good indicator) = **+0.279**.

| p_good tercile | n | mean human | %good(3) | %bad(1) |
|---|---:|---:|---:|---:|
| low | 20 | 1.25 | 0% | 75% |
| mid | 20 | 2.00 | 25% | 25% |
| high | 20 | 1.95 | 25% | 30% |

## B. Provisional keeper cut vs human-good (label==3)

Prediction = `corn_decode(p_notbad, p_good, keeper_cut) == 3`; positive = human good.

| family | keeper cut | n | pred-keepers | precision | recall | F0.5 |
|---|---:|---:|---:|---:|---:|---:|
| julia:mandelbrot | 0.55 | 1 | 0 | 0.00 | 0.00 | 0.00 |
| julia:multibrot3 | 0.27 | 5 | 5 | 0.40 | 1.00 | 0.45 |
| julia:multibrot5 | 0.15 | 1 | 1 | 0.00 | 0.00 | 0.00 |
| mandelbrot | 0.51 | 16 | 0 | 0.00 | 0.00 | 0.00 |
| multibrot3 | 0.5* | 11 | 11 | 0.18 | 1.00 | 0.22 |
| multibrot4 | 0.5* | 8 | 8 | 0.25 | 1.00 | 0.29 |
| multibrot5 | 0.5* | 18 | 18 | 0.22 | 1.00 | 0.26 |
| **pooled** | (per-fam) | 60 | 43 | 0.23 | 1.00 | 0.27 |

Pooled keeper confusion: TP=10 FP=33 FN=0. Of 43 predicted keepers, **10 were human-good** (precision 23%); of 10 human-good, **10 were kept** (recall 100%).

## C. Where should the cut move? (F0.5-optimal on THIS labeled set)

Small n — treat as directional. Pooled first, then per-family where n>=10.

- **pooled** F0.5-optimal p_good cut ~ **0.59** (F0.5=0.36, P=0.31 R=1.00).
- julia:mandelbrot: n=1 good=0 — too few to re-derive; keep provisional 0.55.
- julia:multibrot3: n=5 good=2 — too few to re-derive; keep provisional 0.27.
- julia:multibrot5: n=1 good=0 — too few to re-derive; keep provisional 0.15.
- mandelbrot: n=16 good=0 — too few to re-derive; keep provisional 0.51.
- multibrot3: n=11 good=2 -> F0.5-optimal ~ **0.80** (provisional 0.5).
- multibrot4: n=8 good=2 — too few to re-derive; keep provisional 0.5.
- multibrot5: n=18 good=4 -> F0.5-optimal ~ **0.59** (provisional 0.5).

## D. Do the deep steered admissions hold up?

| depth bucket | n | mean human | %good | %bad |
|---|---:|---:|---:|---:|
| shallow(<=3) | 12 | 1.75 | 17% | 42% |
| mid(4-8) | 33 | 1.70 | 15% | 45% |
| deep(>8) | 15 | 1.80 | 20% | 40% |

## E. Mandelbrot (21 discovery / 0 provisional-keepers)

16 mandelbrot tiles in the set. human label vs p_good (sorted by p_good):

| p_good | depth | human | provisional keeper |
|---:|---:|---:|---:|
| 0.370 | 3 | 1 | no |
| 0.361 | 15 | 1 | no |
| 0.338 | 5 | 1 | no |
| 0.331 | 5 | 1 | no |
| 0.315 | 2 | 1 | no |
| 0.262 | 3 | 2 | no |
| 0.262 | 7 | 1 | no |
| 0.231 | 7 | 1 | no |
| 0.186 | 3 | 1 | no |
| 0.179 | 13 | 1 | no |
| 0.165 | 5 | 1 | no |
| 0.164 | 11 | 1 | no |
| 0.152 | 10 | 1 | no |
| 0.148 | 4 | 2 | no |
| 0.146 | 5 | 2 | no |
| 0.145 | 4 | 1 | no |

Human called **0/16** mandelbrot tiles good; provisional keepers among them: **0**. 0 keepers looks JUSTIFIED.

## Verdict

1. **p_good is a BADNESS filter, not a GOODNESS ranker on steered output.** The low p_good tercile is 75% bad / 0% good — reliably weak. But the HIGH tercile is only 25% good, no better than mid: above the low band, higher p_good does NOT mean better. Spearman +0.40 is carried by the bad end.
2. **The keeper tier as derived is too permissive on steered output**: it keeps 43/60 tiles at 23% precision (every human-good is caught, but so are 33 non-good). The F0.5-optimal cut on this set moves UP (pooled ~0.59, multibrot ~0.6–0.8), but even optimal tops out near ~30% precision — NO p_good threshold cleanly isolates human-good here. The lever is a better ranking signal, not a higher cut.
3. **The depth expansion holds up.** Deep(>8) admissions are 20% good vs shallow 17% — the eviction-fix depth gain did NOT dilute quality; deep steered locations are as good as shallow.
4. **Mandelbrot 0-keepers confirmed** (0/16 human-good; all p_good below its 0.51 cut). Its discovery bar (t_good 0.14) admits locations the human uniformly rejects — a discovery-side over-admission, correctly zeroed by the keeper cut.
5. **Recommendation:** keep the keeper tier report-only; do NOT promote these cuts to a gate. Move multibrot keeper cuts up (~0.6) to trim the worst, but treat 'keeper' as 'not-clearly-bad', not 'good'. Delivering confident-good needs a preference/ranking head beyond CORN p_good (cf. pref-v3), or a human pass.
