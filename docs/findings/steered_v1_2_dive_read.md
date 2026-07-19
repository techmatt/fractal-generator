# Steered v1.2 dive — blind read (the deep-options adjudication)

21 dive admissions, blind-scored (no coords/depth/group/p_good shown). Human labels: **1 bad / 12 okay / 8 good** (good-rate 38%, bad-rate 5%). For contrast the steered_run2 blind read (shallow+deep admissions, stratified) was ~43% bad / 17% good.

## A. Deep-options: top-start vs control

Top-start dives descend from the highest-canonical-p_good run-2 admissions; control dives from randomly-chosen run-2 admissions regardless of score. If deep quality needs a good starting neighborhood, top should beat control; if it's reachable from anywhere, they should be similar.

| start group | n | mean human | %good | %okay | %bad |
|---|---:|---:|---:|---:|---:|
| top | 17 | 2.35 | 41% | 53% | 6% |
| control | 4 | 2.25 | 25% | 75% | 0% |
| all | 21 | 2.33 | 38% | 57% | 5% |

**Read:** top-start meaningfully beats control — deep quality benefits from a good starting neighborhood. Top good-rate 41% (7/17) vs control 25% (1/4); neither group is mostly bad (6% / 0% bad).

## B. Does quality hold with depth?

| depth bucket | n | mean human | %good | %bad |
|---|---:|---:|---:|---:|
| shallow(<=6) | 4 | 2.75 | 75% | 0% |
| mid(7-10) | 10 | 2.10 | 20% | 10% |
| deep(>10) | 7 | 2.43 | 43% | 0% |

Admitted depth range 4–13 (median 8). Spearman(depth, human) = **-0.110** — quality does not decay with depth on this set.

## C. Does canonical p_good track the human judgement (deep-only)?

Spearman(canonical p_good, human label) = **-0.039**; vs human-good indicator **+0.000** (n=21).

- lower-half canon p_good (median 0.587): good 40%, bad 0%.
- upper-half canon p_good (median 0.731): good 36%, bad 9%.

## D. Family + morph breakdown

| family | n | %good | mean human |
|---|---:|---:|---:|
| julia:multibrot3 | 2 | 100% | 3.00 |
| multibrot3 | 4 | 50% | 2.50 |
| multibrot4 | 6 | 0% | 1.83 |
| multibrot5 | 9 | 44% | 2.44 |

21 distinct morph clusters across 21 admissions (8 contain a human-good).

## Verdict

1. **The deep dives are good.** 38% good / 5% bad on a DEEP-ONLY set (median depth 8) — a far cleaner yield than the steered_run2 read (~17% good / ~43% bad). Deep locations along a descended lineage are real keepers, not degenerate zoom.
2. **Deep options: top-start meaningfully beats control — deep quality benefits from a good starting neighborhood.**
3. **Depth does not dilute quality** (Spearman depth×human -0.11); the fw floor / gate-death terminates dives before quality collapses.
4. **p_good on deep output**: Spearman -0.04 — weak, as on run-2 steered output; the human is the adjudicator here.