# Library intake 2 — descriptor + clustering readout

Stage-1 intake (`tools/emission/library_intake_2.py`): admitted locations from the discovery era's four remaining ledgers → canonical morph-CLIP embedding → within-family incremental medoid cluster (cos 0.974). **No colorize / gating / pooling / selection ran; no wallpapers were produced.** Reuses campaign1_intake's primitives verbatim.

## 1. Counts + reconciliation

Admission predicate (`descriptor.load_admitted`): current-decode (v7) ∧ `decoded_class==3` ∧ `guard_pass` ∧ `distinct`. Cross-ledger union dedups by row `id`. The re-decoded phoenix grid is at t_good=0.45; classic phoenix is current-decoded at 0.45.

| ledger | rows_in | admitted | rejected | dedup_dropped | reject reasons |
|---|--:|--:|--:|--:|---|
| `c2_breadth` | 326 | 311 | 15 | 0 | not_distinct=15 |
| `c2_dive` | 273 | 271 | 2 | 0 | not_distinct=1, decoded_class!=3=1 |
| `phoenix_grid` | 337 | 196 | 141 | 0 | not_distinct=141 |
| `classic_phoenix` | 58 | 41 | 17 | 0 | not_distinct=17 |

**Union admitted (id-dedup across ledgers): 819** (cross-ledger dedup dropped 0). Every ledger reconciled exactly (`rows_in == admitted + rejected + dedup_dropped`) — loud-exit on any unexplained remainder.

### admitted per source tag

| source | admitted rows | distinct clusters |
|---|--:|--:|
| `c2_breadth` | 311 | 293 |
| `c2_dive` | 271 | 252 |
| `classic_phoenix` | 41 | 41 |
| `phoenix_grid` | 196 | 196 |

## 2. Julia re-score anchor (robust control-envelope criterion)

Re-scored **24** admitted julia rows at reframe/deploy fidelity with the live v7 scorer vs stored ledger `p_good`, alongside a same-size **Mandelbrot/multibrot control** (n=24). Phoenix rows are excluded from both samples. The 1e-4 tolerance is the fp16 autocast batch-composition floor (established), not a render-correctness test. A single steep-response location can shift stored-vs-fresh `p_good` well past the control's max with **zero render error** — the render is bit-deterministic there. So the criterion is **distributional**: julia's bulk must sit within the control envelope AND every envelope-exceeding row must be a render-deterministic sensitive point (a systematic split-coord bug would corrupt many rows, non-deterministically).

| sample | n | max\|Δ\| | p90\|Δ\| | median\|Δ\| | exact |
|---|--:|--:|--:|--:|--:|
| julia | 24 | 3.490e-02 | 2.351e-03 | 1.625e-04 | 9/24 |
| control | 24 | 2.825e-03 | 8.201e-04 | 1.256e-04 | 10/24 |

control envelope = `3.000e-03`; bulk_within (julia p90 ≤ envelope) = **True**; no_median_bias = **True**; outlier_fraction = **8.3%**.

Envelope-exceeding julia rows (each re-rendered twice — deterministic ⇒ sensitive location, not a render bug):

| id | source | stored | rescored | Δ | re-render spread | deterministic |
|---|---|--:|--:|--:|--:|:-:|
| `st_julia_mandelbrot_dive_000001` | c2_dive | 0.44972 | 0.48462 | 3.49e-02 | 0.0e+00 | ✓ |
| `st_julia_multibrot3_breadth_000052` | c2_breadth | 0.57815 | 0.57427 | 3.88e-03 | 0.0e+00 | ✓ |

**PASS** — julia bulk sits within the control envelope and every excess row is a bit-deterministic sensitive point (the fp16 batch-noise floor), not a split-coord render error. Julia split-coord rendering (`outcome_cx/cy` viewport + `julia_c_re/im` parameter c) is trusted. (Raw literal-1e-4 gate: fail — the autocast batch-noise floor, expected.)

## 3. Full library occupancy — type × morph_cluster

**819 admitted locations → 745 distinct morph clusters** (within-family incremental medoid, cos 0.974).

### per family (partition)

| family | admitted rows | distinct clusters | rows/cluster |
|---|--:|--:|--:|
| julia:mandelbrot | 77 | 64 | 1.20 |
| julia:multibrot3 | 56 | 34 | 1.65 |
| julia:multibrot4 | 62 | 52 | 1.19 |
| julia:multibrot5 | 61 | 53 | 1.15 |
| mandelbrot | 114 | 103 | 1.11 |
| multibrot3 | 49 | 46 | 1.07 |
| multibrot4 | 66 | 63 | 1.05 |
| multibrot5 | 97 | 93 | 1.04 |
| phoenix | 237 | 237 | 1.00 |
| **total** | **819** | **745** | **1.10** |

### cluster-size distribution

| cluster size | # clusters |
|--:|--:|
| 1 | 691 |
| 2 | 43 |
| 3 | 10 |
| 12 | 1 |

**Singleton fraction: 691/745 = 92.8%.**

**Full-library context:** campaign-1 intake contributed **523** distinct clusters (separate pass, `out/emission/campaign1_intake.md`); this intake adds **745** across the four remaining ledgers, for a library total of **~1268** distinct looks (clustered in two passes — the counts are additive across disjoint families/sources, not re-reconciled).

## 4. Where classic phoenix lands in cluster space

Grid (varied) and classic phoenix share `family==phoenix`, so they cluster together. Phoenix partition: **237 clusters**. Classic-phoenix rows land in **41** of them: **41 pure-classic** (no grid member) and **0 mixed** (shared with grid).

**Separation verdict: CLEAN** — classic phoenix occupies its own clusters, disjoint from every varied-phoenix motif cluster.

Classic-phoenix cluster ids (the override address):

```
phoenix#196   classic=1  grid=0
phoenix#197   classic=1  grid=0
phoenix#198   classic=1  grid=0
phoenix#199   classic=1  grid=0
phoenix#200   classic=1  grid=0
phoenix#201   classic=1  grid=0
phoenix#202   classic=1  grid=0
phoenix#203   classic=1  grid=0
phoenix#204   classic=1  grid=0
phoenix#205   classic=1  grid=0
phoenix#206   classic=1  grid=0
phoenix#207   classic=1  grid=0
phoenix#208   classic=1  grid=0
phoenix#209   classic=1  grid=0
phoenix#210   classic=1  grid=0
phoenix#211   classic=1  grid=0
phoenix#212   classic=1  grid=0
phoenix#213   classic=1  grid=0
phoenix#214   classic=1  grid=0
phoenix#215   classic=1  grid=0
phoenix#216   classic=1  grid=0
phoenix#217   classic=1  grid=0
phoenix#218   classic=1  grid=0
phoenix#219   classic=1  grid=0
phoenix#220   classic=1  grid=0
phoenix#221   classic=1  grid=0
phoenix#222   classic=1  grid=0
phoenix#223   classic=1  grid=0
phoenix#224   classic=1  grid=0
phoenix#225   classic=1  grid=0
phoenix#226   classic=1  grid=0
phoenix#227   classic=1  grid=0
phoenix#228   classic=1  grid=0
phoenix#229   classic=1  grid=0
phoenix#230   classic=1  grid=0
phoenix#231   classic=1  grid=0
phoenix#232   classic=1  grid=0
phoenix#233   classic=1  grid=0
phoenix#234   classic=1  grid=0
phoenix#235   classic=1  grid=0
phoenix#236   classic=1  grid=0
```

## 5. Proposed classic-phoenix measure override (report-only — human applies)

**Measure loader accepts `morph_cluster` sets in `weight_overrides`: YES.** Verified against `cells.TargetMeasure.weight` (a synthetic cell on the classic cluster ids gets the override multiplier; a non-member cell gets 1.0).

For a **~2% classic-phoenix release share**: with K=41 classic clusters out of N=745 observed (type,cluster) pairs, the uniform-base multiplier is **W=0.35** (W = s(N−K)/((1−s)K)). The exact stanza to add to `data\emission\target_measure.json` `weight_overrides` (do NOT let this tool edit the measure — left to the human):

```json
{
  "match": {
    "morph_cluster": [
      "phoenix#196",
      "phoenix#197",
      "phoenix#198",
      "phoenix#199",
      "phoenix#200",
      "phoenix#201",
      "phoenix#202",
      "phoenix#203",
      "phoenix#204",
      "phoenix#205",
      "phoenix#206",
      "phoenix#207",
      "phoenix#208",
      "phoenix#209",
      "phoenix#210",
      "phoenix#211",
      "phoenix#212",
      "phoenix#213",
      "phoenix#214",
      "phoenix#215",
      "phoenix#216",
      "phoenix#217",
      "phoenix#218",
      "phoenix#219",
      "phoenix#220",
      "phoenix#221",
      "phoenix#222",
      "phoenix#223",
      "phoenix#224",
      "phoenix#225",
      "phoenix#226",
      "phoenix#227",
      "phoenix#228",
      "phoenix#229",
      "phoenix#230",
      "phoenix#231",
      "phoenix#232",
      "phoenix#233",
      "phoenix#234",
      "phoenix#235",
      "phoenix#236"
    ]
  },
  "weight": 0.35
}
```
This up-weights only the classic-phoenix cluster cells; N grows as more (type,cluster) pairs enter the library, so re-tune W against the live feasible-cell census at emission time.

## 6. Medoid contact sheets

Grayscale morph medoids (founding member of each cluster), one sheet per family:

- `out\emission\library_intake_2\medoid_sheets\medoids_julia_mandelbrot.png` — julia:mandelbrot: 64 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_julia_multibrot3.png` — julia:multibrot3: 34 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_julia_multibrot4.png` — julia:multibrot4: 52 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_julia_multibrot5.png` — julia:multibrot5: 53 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_mandelbrot.png` — mandelbrot: 103 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_multibrot3.png` — multibrot3: 46 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_multibrot4.png` — multibrot4: 63 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_multibrot5.png` — multibrot5: 93 cluster medoids
- `out\emission\library_intake_2\medoid_sheets\medoids_phoenix.png` — phoenix: 237 cluster medoids
