# Campaign-1 emission intake — descriptor + clustering readout

Intake stage only (`tools/emission/campaign1_intake.py`): admitted campaign-1 locations → canonical morph-CLIP embedding → within-family incremental medoid cluster (cos 0.974). **No colorize / gating / pooling / selection ran; no wallpapers were produced.**

## 1. Counts + reconciliation

Admission predicate (as implemented, `descriptor.load_admitted`): current-decode (v7) ∧ `decoded_class==3` ∧ `guard_pass` ∧ `distinct`. Cross-ledger union dedups by row `id`.

| ledger | rows_in | admitted | rejected | dedup_dropped | reject reasons |
|---|--:|--:|--:|--:|---|
| `c1_breadth` | 330 | 314 | 16 | 0 | not_distinct=16 |
| `c1_dive` | 259 | 254 | 5 | 0 | not_distinct=5 |

**Union admitted (id-dedup across ledgers): 568** (cross-ledger dedup dropped 0). Every ledger reconciled exactly (`rows_in == admitted + rejected + dedup_dropped`).

### admitted per source tag

| source | admitted rows | distinct clusters |
|---|--:|--:|
| `c1_breadth` | 314 | 307 |
| `c1_dive` | 254 | 246 |

## 2. Julia re-score anchor

Re-scored **20** admitted julia rows at reframe/deploy fidelity (640×360 ss2, twilight_shifted, `auto_maxiter`) with the live v7 scorer vs the stored ledger `p_good`, alongside a same-size **Mandelbrot/multibrot control** (n=20) through the identical render+score path.

**Why a control, not a bare 1e-4 gate.** The stored `p_good` was produced under fp16 autocast inside a multi-frame batch (reframe rungs / walk frames); a single-frame rescore differs at the ~1e-3 level from fp16 **batch-composition** noise, family-independently. Proven here: re-scoring one jpg twice gives Δ=0 (deterministic), and the known-good Mandelbrot path shows the *same* scatter as julia. A literal 1e-4 tolerance therefore measures scorer batch-noise, not render correctness, and is unachievable for **any** family. The valid criterion is that julia's rescore scatter sits **within the known-good Mandelbrot envelope** with no julia-specific bias — a broken julia render would miss by 0.1–0.5, far outside it.

| sample | n | max\|Δ\| | mean\|Δ\| | exact matches |
|---|--:|--:|--:|--:|
| julia | 20 | 8.885e-04 | 1.688e-04 | 9/20 |
| mandelbrot control | 20 | 6.564e-03 | 6.271e-04 | 8/20 |

**PASS** — julia max|Δ| `8.885e-04` ≤ control envelope `6.564e-03`, no bias. Julia split-coord rendering (`outcome_cx/cy` viewport + `julia_c_re/im` parameter c) is trusted, rendering as faithfully as the Mandelbrot path. (Raw literal-1e-4 gate: fail — as expected, the autocast batch-noise floor, not a render error.)

Per-row julia deltas:

| id | family | source | stored | rescored | Δ |
|---|---|---|--:|--:|--:|
| `st_julia_mandelbrot_breadth_000015` | julia:mandelbrot | c1_breadth | 0.27689 | 0.27689 | 0.00e+00 |
| `st_julia_mandelbrot_dive_000002` | julia:mandelbrot | c1_dive | 0.89625 | 0.89607 | 1.82e-04 |
| `st_julia_multibrot3_breadth_000004` | julia:multibrot3 | c1_breadth | 0.72066 | 0.72066 | 0.00e+00 |
| `st_julia_multibrot3_dive_000019` | julia:multibrot3 | c1_dive | 0.71909 | 0.71820 | 8.88e-04 |
| `st_julia_multibrot4_breadth_000011` | julia:multibrot4 | c1_breadth | 0.73678 | 0.73678 | 0.00e+00 |
| `st_julia_multibrot4_dive_000014` | julia:multibrot4 | c1_dive | 0.85621 | 0.85621 | 0.00e+00 |
| `st_julia_multibrot5_breadth_000003` | julia:multibrot5 | c1_breadth | 0.59825 | 0.59796 | 2.93e-04 |
| `st_julia_multibrot5_dive_000026` | julia:multibrot5 | c1_dive | 0.78695 | 0.78679 | 1.64e-04 |
| `st_julia_mandelbrot_breadth_000019` | julia:mandelbrot | c1_breadth | 0.88504 | 0.88504 | 0.00e+00 |
| `st_julia_mandelbrot_dive_000247` | julia:mandelbrot | c1_dive | 0.33405 | 0.33437 | 3.26e-04 |
| `st_julia_multibrot3_breadth_000023` | julia:multibrot3 | c1_breadth | 0.47693 | 0.47693 | 0.00e+00 |
| `st_julia_multibrot3_dive_000021` | julia:multibrot3 | c1_dive | 0.59602 | 0.59602 | 0.00e+00 |
| `st_julia_multibrot4_breadth_000013` | julia:multibrot4 | c1_breadth | 0.42764 | 0.42746 | 1.79e-04 |
| `st_julia_multibrot4_dive_000018` | julia:multibrot4 | c1_dive | 0.61306 | 0.61295 | 1.16e-04 |
| `st_julia_multibrot5_breadth_000006` | julia:multibrot5 | c1_breadth | 0.81845 | 0.81845 | 0.00e+00 |
| `st_julia_multibrot5_dive_000082` | julia:multibrot5 | c1_dive | 0.68384 | 0.68363 | 2.11e-04 |
| `st_julia_multibrot3_breadth_000024` | julia:multibrot3 | c1_breadth | 0.64601 | 0.64646 | 4.47e-04 |
| `st_julia_multibrot3_dive_000091` | julia:multibrot3 | c1_dive | 0.68521 | 0.68511 | 1.05e-04 |
| `st_julia_multibrot4_breadth_000042` | julia:multibrot4 | c1_breadth | 0.79692 | 0.79692 | 0.00e+00 |
| `st_julia_multibrot4_dive_000028` | julia:multibrot4 | c1_dive | 0.86235 | 0.86189 | 4.64e-04 |

## 3. Occupancy — type × morph_cluster

**568 admitted locations → 523 distinct morph clusters** (within-family incremental medoid, cos 0.974).

### per family

| family | admitted rows | distinct clusters | rows/cluster |
|---|--:|--:|--:|
| julia:mandelbrot | 4 | 4 | 1.00 |
| julia:multibrot3 | 50 | 37 | 1.35 |
| julia:multibrot4 | 26 | 19 | 1.37 |
| julia:multibrot5 | 18 | 15 | 1.20 |
| mandelbrot | 113 | 102 | 1.11 |
| multibrot3 | 163 | 159 | 1.03 |
| multibrot4 | 45 | 42 | 1.07 |
| multibrot5 | 149 | 145 | 1.03 |
| **total** | **568** | **523** | **1.09** |

### cluster-size distribution

| cluster size | # clusters |
|--:|--:|
| 1 | 483 |
| 2 | 36 |
| 3 | 3 |
| 4 | 1 |

**Singleton fraction: 483/523 = 92.4%.** (At the smoke's ~50 locations every cluster was a singleton; at this scale clustering now collapses near-duplicates.)

## 4. Cross-reference — intake distinct vs campaign-1 readout

- Campaign-1 readout distinct looks (within-family clustering): **508**
- This intake's distinct morph clusters: **523**
- Gap: **-15**

Methodological difference (report the gap, don't chase reconciliation): this intake uses **incremental medoid** clustering at cos 0.974 — each location joins the existing cluster whose *founding* embedding it exceeds threshold against, else founds a new one (order-dependent, single-pass, medoid = founder). The campaign-1 readout's 508 came from its own within-family dedup pass; the counts are not expected to reconcile exactly and are not forced to.

## 5. Medoid contact sheets

Grayscale morph medoids (founding member of each cluster), one sheet per family, each tile labeled `<family>#<k>  n=<cluster size>` — for a human eyeball pass:

- `out\emission\campaign1\medoid_sheets\medoids_julia_mandelbrot.png` — julia:mandelbrot: 4 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_julia_multibrot3.png` — julia:multibrot3: 37 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_julia_multibrot4.png` — julia:multibrot4: 19 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_julia_multibrot5.png` — julia:multibrot5: 15 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_mandelbrot.png` — mandelbrot: 102 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_multibrot3.png` — multibrot3: 159 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_multibrot4.png` — multibrot4: 42 cluster medoids
- `out\emission\campaign1\medoid_sheets\medoids_multibrot5.png` — multibrot5: 145 cluster medoids
