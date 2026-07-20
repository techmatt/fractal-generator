# Julia dup-metric audit — diagnosis (campaign-1)

Diagnosis only, no pipeline changes. Verdict per question below; forensics scripts in
`scratchpad/{julia_dup_forensics,render_integrity,lookcount_reconcile}.py`.

**Bottom line: the suspected bug is REAL.** Julia distinctness keys on the z-viewport only,
inside a per-*base-family* cloud that is shared across every distinct seed c. Distinct-c
Julias annihilate each other. The render/labeling path is clean (no dropped-c bug). Damage
is large in the breadth leg (191 distinct-c hooks wholly suppressed) and mostly not cheaply
recoverable from logs.

---

## 1. Keying — **z-coords only, per base-family; seed c never enters the dup check.**

Every julia distinctness test routes through `production_seeder.is_distinct` → `near_dup`,
which keys purely on the z-viewport `(cx, cy, fw)`:

```python
# production_seeder.py
def near_dup(a_cx,a_cy,a_fw, b_cx,b_cy,b_fw, k=DEDUP_K):
    d = float(np.hypot(a_cx-b_cx, a_cy-b_cy))
    return d < k * max(float(a_fw), float(b_fw))          # seed c absent
def is_distinct(cx,cy,fw, cloud, k=DEDUP_K):
    for h in cloud:
        if near_dup(cx,cy,fw, h["outcome_cx"],h["outcome_cy"],h["outcome_fw"], k):
            return False, h["id"]
```

The cloud is per-partition and the partition for a julia is its *base family* only —
`build_cloud(rows, family)` filters `r["family"] == family` where `family` is e.g.
`"julia:multibrot3"`, i.e. **all seed-c of that base family share one cloud**. All three
dup checks use this same z-only test against that shared cloud:

- **Within-run q3-cloud (pre-reframe skip):** `steered_frontier.py:778`
  `is_distinct(c["cx"], c["cy"], c["fw"], self.clouds.get(c["partition"], []))`.
- **Admission near-dup (post-reframe):** `steered_frontier.py:799`
  `is_distinct(ocx, ocy, ofw, self.clouds[c["partition"]])`.
- **Cross-run prior-corpus overlap:** same `near_dup`; `build_cloud` reconstructs the cloud
  from *all* cross-run rows of that family — still z-only.

Seed c participates in exactly one place — julia **root spawning** (`add_julia_root`), gated
against a separate `hooked_c` set by `REJECT_RADIUS`/`Q3_DENSITY_CAP`. The module comment
states the split explicitly (`steered_frontier.py:626-630`): "a julia partition's OUTCOME
cloud is keyed on the z-viewport … production keys its julia cloud on c directly; here the two
roles are split." So distinctness of a produced julia *image* is decided without ever
consulting which Julia set it came from.

**Amplifier:** `JULIA_ROOT_FW = 3.0` and `near_dup` radius `= 1.5·max(fw)`. A shallow julia
view sits near base scale, so its dedup radius is ~4.5 in the z-plane — larger than the whole
interesting region. After the first admission in a base-family cloud, essentially any later
shallow julia lands inside that radius and is killed regardless of its c.

## 2. Reject forensics — **over-kill, not genuine c-continuity churn.**

**Ledger dup rows (post-reframe, carry `dup_of` + both seed c's), 16 julia total.** Seed-c
distance from each reject to the admission it collided against:

| seed-c dist bucket | count |
|---|---|
| exact (<1e-9) — true same-c revisit | 4 |
| <0.20 (`REJECT_RADIUS`) | 2 |
| ≥1.0 — **different Julia plane** | 8 |

min 0 · median **1.032** · max **1.513**. **12 of 16** collided against a *different root*
(broad c-spread); only **4** are genuine same-c revisits. A c-distance >1 means the reject
and its "duplicate" are entirely different Julia sets that merely happened to descend to a
similar z-viewport. This is the over-kill signature.

**Depth.** Julia admissions: n=98, depth min **2**, median 5 (shallow admissions *do* exist —
5 at depth 2). Julia ledger-dup rejects: depth 3–7, median 5. So depth is *not* the tell here
(no systematic absence of shallow admissions); the c-distance histogram is.

**Churn magnitude (harvest log, all scored candidates).**

| leg | julia canon-q3 | admitted | rejected | churn | pre-reframe skip | post-reframe rej |
|---|---|---|---|---|---|---|
| breadth | 20,558 | 46 | 20,512 | **100%** | 20,499 | 13 |
| dive | 1,195 | 52 | 1,143 | **96%** | 1,140 | 3 |

The churn is dominated by pre-reframe skips (z-only test against the shared cloud). Because
the ledger-visible rejects are 12/16 broad-c, this churn is metric over-kill of distinct-c
Julias, not chain-neighbor continuity.

## 3. Render integrity — **clean; no dropped-seed-c bug.**

`steered_pilot_morph.loc_of_row` carries seed c explicitly for steered julia (cx/cy/fw =
z-viewport, c_re/c_im = `julia_c_*`). Re-rendered 8 julia ledger rows from stored coords via
this path and diffed against the stored blind-batch tiles:

| | mean-abs pixel diff vs stored tile |
|---|---|
| re-render **with** seed c (loc_of_row) | **0.000** (byte-identical), all 8 |
| adversarial re-render **dropping** seed c (parent plane, same z-viewport) | 37–88 |

The stored julia tiles are genuine seed-c Julia views; the render and labeling path is not the
source of what campaign-1 saw. The other candidate bug (a dropped c producing parent-plane
views) is ruled out.

## 4. Damage estimate (over-kill confirmed).

**Root-level footprint (harvest log, breadth):** 223 distinct julia roots produced
canonical-q3 frames; only **32** admitted ≥1 frame → **191 distinct julia roots had every
canonical-q3 frame rejected.** Each root is a distinct seed-c hook, so ~191 distinct Julia
sets were wholly suppressed in breadth alone. Dive: **46/46** roots admitted — dive is
single-track descent (one c per dive, no pileup into a shared cloud), so it escapes the
over-kill; this is why the pathology is a breadth phenomenon.

**Recoverability.**
- **Directly re-admittable:** the 16 ledger julia dup rows (12 different-c) — full
  `(z-viewport, seed c)` in the ledger.
- **The 191 suppressed roots — mostly NOT cheaply recoverable from logs.** Their candidate
  z-viewport coords survive in the durable scratch tree
  (`data/discovery/campaign1/breadth/scratch/expand_b*/julia_*/expand.jsonl`), **but the seed c
  is never durably logged for a zero-admit root** — it is absent from `harvest_log.jsonl`,
  from scratch `nodes.jsonl`/`expand.jsonl` (which record only the z-plane node), and there is
  no ledger row (zero admits). The seed c equals the parent c-plane admission's outcome
  coordinate (`add_julia_root(partition, c=parent_outcome, parent_oid)`), but `root→parent`
  isn't in the harvest log either. So exact reconstruction of the suppressed frames needs
  **re-discovery** — re-run the julia harvest with a c-aware dedup. That re-run is
  deterministic given the same c-plane parents, so it is cheap-ish, not free.

## 5. Look-count reconciliation — **508 canonical (within-family); 488 is a layout byproduct.**

Both counts run on the *same* population (`spm.admitted_q3` over both ledgers), the *same*
`STRICT_CUT = 0.974`, and the *same* CLIP morph_gray embedding recipe. The only difference is
the clustering scope:

- **508** — `campaign1_readout.distinct_look_count`: single-linkage **within family**
  (partitions indices by family, sums per-family component counts). This is the headline
  "distinct morph looks" metric.
- **488** — `campaign1_contact_sheet.cluster_order`: single-linkage **global / family-blind**
  (one `U @ Uᵀ` over all admissions). It is computed only to order tiles by adjacency;
  `n_clusters` there is a side effect of the layout sort.

Global clustering has every within-family edge *plus* cross-family edges, so it can only merge
more (never fewer) → 488 ≤ 508, with the gap = cross-family transitive merges. **Canonical:
508** (within-family is the designed metric, matching the morphology-dedup-pass convention).
Fix (next prompt): make `cluster_order` cluster within family (or have the contact sheet
report the within-family look total), so it reports 508.

> **Reproduced exactly** (`scratchpad/lookcount_reconcile.py`, CLIP pass over the shared
> population): population **= 568** (identical `admitted_q3` set for both tools), global
> single-linkage **= 488**, within-family single-linkage **= 508**. Confirms the divergence is
> purely clustering scope, nothing else.

---

### Recommendation for the fix prompt
The dup key for julia must include the seed c (or key the cloud per exact c, matching
production's "keys its julia cloud on c directly"). Separately, the base-scale dedup radius
(`1.5·max(fw)` at `JULIA_ROOT_FW=3.0` ≈ 4.5) is too coarse for shallow julia views even
within one c — hook spacing / a tighter shallow radius should be designed alongside the
c-aware key. And durably log julia seed c (or `root→parent_oid`) for *rejected* rows so future
over-kill is recoverable without re-discovery.
