# Julia dup-metric audit — diagnosis only, no pipeline changes

Context: in the campaign-1 blind read I saw zero julia center-descent views, and julia harvest checks were 78–99% dup-churn. Suspicion: the coord-dup machinery may be colliding *distinct-c* julias — every julia hook starts at the same fixed shared z-plane view, so if dup keying within a julia partition uses z-coordinates without the seed c, every near-root view after the first one collides regardless of c. That bug would mimic the "continuity in c" churn story in the aggregate numbers. This prompt decides which story is true before we design hook spacing and the pre-canonical filter. Diagnosis only — fixes are the next prompt.

Answer these, one-line verdict each, full detail in a findings doc under `docs/findings/`:

1. **Keying.** What exactly does each dup check key on for julia rows — the run's coord-dup/q3-cloud check, the admission near-dup check, and the cross-run prior-corpus overlap? Does the seed c participate anywhere, or is it z-coords-only within a partition? Quote the relevant code.

2. **Reject forensics.** Over campaign-1's julia dup rejects, pair each reject with the admitted row it collided against and histogram the seed-c distances. Chain-neighbor parents ⇒ genuine churn; broad c-spread ⇒ metric over-kill. Also: depth distribution of julia rejects vs julia admissions, and the minimum depth among the 98 julia admissions — systematic absence of shallow admissions is the over-kill signature.

3. **Render integrity.** Re-render a handful of julia ledger rows and a few of the labeled-batch julia tiles from their stored coords with the seed c explicit, and confirm they match the stored artifacts. This rules out the other bug consistent with what I saw: a dropped seed c producing parent-plane views somewhere in the render or labeling path.

4. **Damage estimate (only if over-kill confirms).** How many campaign-1 julia canonical-q3 frames were rejected against a different-c admission, and are they recoverable from the harvest logs (coords are durable → cheap re-admission) or does it take re-discovery?

5. **Look-count reconciliation.** The readout says 508 distinct looks, the contact sheet says 488. Find the divergence, name one number canonical, and make the other tool agree.

Most of this is log analysis — should be quick; if any step is long, estimate and background it.
