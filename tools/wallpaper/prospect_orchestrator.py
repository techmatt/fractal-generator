#!/usr/bin/env python
r"""Phase-1 prospecting orchestrator — unattended fresh-discovery -> location-LIBRARY loop.

Same loop as `overnight_orchestrator.py`, different tail. Each CYCLE is: fresh discovery
(production_seeder, all 9 families, run-scoped ledger) -> pool build (build_fresh_discovery over
ONLY this cycle's fresh q3s) -> fresh-isolation assert -> ANNOTATE + PERSIST LIBRARY RECORD. There
is NO emit phase: `emit_v1` is never called, nothing renders at wallpaper res. Instead every fresh
q3 location accrues a dense-cheap library record (identity + v6 location-potential + grayscale
morphology CLIP + thumbnail) into a durable store that survives `rm -r out/*`.

REUSED VERBATIM from the overnight orchestrator (imported, same code objects — the strongest
"behaves identically" guarantee): the run-scoped discovery dir + fresh-isolation assert, the
per-phase fully-exiting child processes (GPU freed between phases) + per-phase failure isolation,
the idempotent `state.json` + `--run-id` resume, the wall-cap discipline (never start a unit that
can't finish; 3x hard-kill backstop), and `purge_cycle_intermediates` at cycle boundaries. The
discovery + pool phase invocations and the fresh-isolation assertion are the SAME functions.

Fresh-generation (load-bearing, unchanged): every record originates from THIS run's empty
run-scoped ledger; the run-start emptiness precondition + per-cycle assert physically forbid a
banked/historical/earlier-cycle location reaching the store. Accumulating fresh records ACROSS
runs into the one library store is intended; each record carries run_id + source_ledger so the two
are distinguishable on inspection.

    # validate first (short throwaway mini-run; backgrounded):
    uv run python -u tools/wallpaper/prospect_orchestrator.py --mini
    # a real multi-day run (safe to run for days; 24h default cap):
    uv run python -u tools/wallpaper/prospect_orchestrator.py --run --cap-hours 24
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import overnight_orchestrator as oo  # sibling module — reuse its helpers/constants verbatim
import library_store as store

# --- production defaults (tunable) ---
CAP_HOURS = 24.0                       # default cap; SAFE to run for days
PER_FAMILY_MIN = oo.PER_FAMILY_MIN
PHOENIX_PER_CYCLE_MIN = oo.PHOENIX_PER_CYCLE_MIN
POOL_COUNT = oo.POOL_COUNT
EST_ANNOTATE_FIXED_S = 20.0            # CLIP model load, once per annotate phase
EST_ANNOTATE_LOC_S = 6.0              # per-location field dump (640x360 ss2) + embed + thumb
FIELD_CACHE_GB = 20.0

# --- reconciliation failure discrimination (see reconcile_cycle) ---
# A field render failure is OPERATIONAL, not selection-shaped: a lone location whose field won't
# render is location-specific noise (a pathological glitch/underflow at one coordinate), whereas a
# large fraction of a cycle failing to render is a systemic defect (broken backend, GPU OOM, disk).
# We therefore halt on field-fail by RATE, never per-event, with a small-sample floor so a 1-of-2
# tiny cycle can't false-halt.
FIELD_FAIL_RATE_HALT = 0.5             # halt if >=50% of a cycle's q3 fail to render (systemic)
FIELD_FAIL_MIN_COUNT = 2               # ...but a single render failure is always noise, never halts

ANNOTATOR = oo.ROOT / "tools" / "wallpaper" / "library_annotate.py"


MB_CPLANE_FAMILIES = ("multibrot3", "multibrot4", "multibrot5")


def _family_cplane_min(fam: str, args) -> float:
    """Per-family c-plane discovery budget (minutes). multibrot3/4/5 take --mb-cplane-min when set
    (the rebalance knob); everything else takes --per-family-min. Default (mb_cplane_min None) is
    NO cut — a conservative default; the instrumented long run measures whether a cut starves the
    hooks before one is applied."""
    if fam in MB_CPLANE_FAMILIES and args.mb_cplane_min is not None:
        return args.mb_cplane_min
    return args.per_family_min


def _record_family_instr(fam_instr: list, cycle: int, fam: str, budget_min: float,
                         summ_out: Path, log) -> None:
    """Append this family's per-cycle base/twin/parent rebalance metrics (read from the seeder's
    --summary-out mirror) so the run summary + logs answer 'did the cut starve the hooks?'."""
    s = _read_json(summ_out)
    reb = s.get("rebalance") or {}
    row = {"cycle": cycle, "family": fam, "budget_min": budget_min,
           "cplane_descents": reb.get("cplane_descents"),
           "qualifying_parents": reb.get("qualifying_parents"),
           "hook_descents": reb.get("hook_descents"),
           "fresh_q3_base": reb.get("fresh_q3_base"),
           "fresh_q3_twin": reb.get("fresh_q3_twin")}
    fam_instr.append(row)
    log(f"  rebalance[{fam}] budget={budget_min}m: cplane_desc={row['cplane_descents']} "
        f"qual_parents={row['qualifying_parents']} hook_desc={row['hook_descents']} "
        f"q3 base={row['fresh_q3_base']} twin={row['fresh_q3_twin']}")


def _read_json(path: Path) -> dict:
    """Best-effort JSON read; {} if absent/unparseable (a missing report surfaces as an
    unexplained leak in reconcile_cycle, which is the correct failure mode)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def reconcile_cycle(q3_found, records_written, selection_report, annotate_report, log=None,
                    deferred=0):
    """The per-cycle harvest invariant is **"no location is dropped without a reason recorded,"**
    NOT "no location is ever dropped." Every fresh q3 must be accountable as exactly one of:

      recorded    = became a library record.
      coord_dup   = a true coordinate-duplicate of one already kept — within-set key dups (build)
                    + store/within-batch coord dups (annotate). The ONE always-legitimate drop.
      field_fail  = its field could not render (annotate RuntimeError) OR it was unrenderable at
                    pool-build. OPERATIONAL, reason recorded. Tolerated as noise, halted by RATE
                    (see FIELD_FAIL_RATE_HALT) — never per-event.
      deferred    = the whole cycle was deferred to a retry (annotate crashed); the durable run
                    ledger makes it re-runnable. Passed in by the caller (0 on the normal path).

    Two residual classes HALT loudly, because they mean the harvest's premise is broken:
      * excl_head (a head-corpus exclusion / selection cap reappearing) — SELECTION-SHAPED loss:
        ranking sneaking back into a phase that must keep every fresh q3. Phase 1 runs
        --no-head-exclude, so this MUST be 0.
      * unexplained (any q3 left over after the above) — a location vanished with NO reason
        recorded. This is the exact failure the invariant exists to catch.

    Returns the breakdown dict. Raises SystemExit only on selection-shaped or unexplained loss, or
    on a field-fail RATE above threshold — an operationally-flaky cycle does not halt the run."""
    within_set = selection_report.get("within_set_dups_dropped", 0)
    excl_head = (selection_report.get("excluded_head_corpus_by_key", 0)
                 + selection_report.get("excluded_head_corpus_by_proximity", 0))
    unrenderable = selection_report.get("unrenderable_dropped", 0)
    coord_dup = annotate_report.get("dropped_coord_dup", 0)
    field_fail_ann = annotate_report.get("dropped_field_fail", 0)
    dropped_coord_dup = within_set + coord_dup
    field_fail = field_fail_ann + unrenderable       # all render failures (reason recorded)
    # unexplained = q3 not accounted by ANY reason (recorded/coord_dup/field_fail/deferred) and not
    # attributable to head exclusion. excl_head is a KNOWN-but-FORBIDDEN reason, kept separate so it
    # halts distinctly from a truly reasonless vanish.
    unexplained = (q3_found - records_written - dropped_coord_dup - field_fail
                   - deferred - excl_head)
    ff_rate = field_fail / q3_found if q3_found else 0.0
    ff_rate_halt = field_fail >= FIELD_FAIL_MIN_COUNT and ff_rate >= FIELD_FAIL_RATE_HALT
    bd = {
        "q3_found": q3_found, "records_written": records_written,
        "dropped_coord_dup": dropped_coord_dup, "field_fail": field_fail,
        "deferred": deferred, "excluded_head": excl_head, "unexplained": unexplained,
        # sub-components
        "within_set_dups": within_set, "store_batch_coord_dup": coord_dup,
        "field_fail_annotate": field_fail_ann, "unrenderable": unrenderable,
        "field_fail_rate": round(ff_rate, 4),
        # back-compat: "dropped_other" now means ONLY the truly-unexplained residual (0 = clean)
        "dropped_other": unexplained,
    }
    if log is not None:
        log(f"  reconcile: q3_found={q3_found} = recorded={records_written} + "
            f"coord_dup={dropped_coord_dup} (within_set={within_set}+store/batch={coord_dup}) + "
            f"field_fail={field_fail} (rate={ff_rate:.2f}) + deferred={deferred} "
            f"[excl_head={excl_head} unexplained={unexplained}]")
    if excl_head != 0:
        raise SystemExit(
            f"HARVEST LEAK (selection-shaped): excl_head={excl_head} != 0. A head-corpus exclusion "
            f"reappeared in a phase that runs --no-head-exclude — ranking/caps sneaking back into "
            f"Phase 1. Breakdown: {bd}. Halting: the harvest's keep-every-q3 premise is broken.")
    if unexplained != 0:
        raise SystemExit(
            f"HARVEST LEAK (unexplained): unexplained={unexplained} != 0. {unexplained} q3(s) "
            f"vanished with NO reason recorded (not a record, coord-dup, field-fail, or deferral). "
            f"Breakdown: {bd}. Halting rather than silently leaking product.")
    if ff_rate_halt:
        raise SystemExit(
            f"HARVEST DEFECT (field-fail rate): {field_fail}/{q3_found} = {ff_rate:.0%} of this "
            f"cycle's q3 failed to render (>= {FIELD_FAIL_RATE_HALT:.0%} floor, {FIELD_FAIL_MIN_COUNT}+ "
            f"failures) — a systemic render failure (broken backend / OOM / disk), not location "
            f"noise. Breakdown: {bd}. Halting for a human.")
    return bd


def annotate_with_retry(attempt, cycle, q3_count, watermark, salvaged, log=None):
    """Run the annotate phase, retrying ONCE on operational failure (crash / OOM / transient).

    `attempt(i)` performs the i-th invocation and returns (ok, reason): ok is True iff run_phase
    succeeded AND the annotate report was written (a clean annotate writes it last; a missing report
    means it crashed mid-cycle). `salvaged()` returns the records persisted so far (store delta).

    Returns None on success — the caller proceeds to reconcile. On a SECOND failure, returns a
    failed-cycle dict and the caller records it and CONTINUES the loop: the run-scoped ledger is
    durable, so the deferred q3 are re-runnable, not lost. Nothing here halts — operational
    flakiness must never take down a 24h run (that distinction is the whole point of this helper)."""
    ok, _ = attempt(0)
    if ok:
        return None
    if log is not None:
        log(f"  annotate cycle {cycle} INCOMPLETE — intermediates kept, retrying once.", "WARN")
    ok, reason = attempt(1)
    if ok:
        return None
    return {"cycle": cycle, "q3_deferred": q3_count, "records_salvaged": salvaged(),
            "reason": f"annotate failed twice ({reason})", "ledger_watermark": watermark}


def orchestrate(args):
    out_root = Path(args.out_root).resolve()
    disc_root = Path(args.discovery_root).resolve()
    run_dir = out_root / args.run_id
    disc_dir = disc_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    disc_dir.mkdir(parents=True, exist_ok=True)
    ledger = disc_dir / "outcome_ledger.jsonl"
    state_path = run_dir / "state.json"
    log = oo.Log(run_dir / "orchestrator.log")
    timing = oo.Timing(run_dir / "timing.jsonl", args.run_id)

    families = args.families
    per_family_s = args.per_family_min * 60.0
    cap_s = args.cap_hours * 3600.0
    # store sinks (overridable so --mini / tests don't pollute the production library)
    sinks = _store_sinks(args)

    # --- resume / idempotency (reuse the ORIGINAL deadline so total wall-clock <= cap) ---
    state = oo.load_state(state_path)
    if state is not None:
        deadline = state["deadline_epoch"]
        cycle = state["cycles_done"]
        log(f"RESUME run '{args.run_id}': {cycle} cycles done, "
            f"{(deadline - time.time())/3600:.2f}h remaining to original deadline")
        if not ledger.exists():
            log("  note: run ledger absent on resume (no discovery persisted yet) — continuing", "WARN")
    else:
        n0 = oo.ledger_line_count(ledger)
        if n0 != 0:
            raise SystemExit(
                f"run-start freshness precondition FAILED: {ledger} already has {n0} rows. "
                f"A fresh run requires an EMPTY run-scoped ledger. Use a new --run-id.")
        deadline = time.time() + cap_s
        cycle = 0
        log(f"START prospect run '{args.run_id}'  cap={args.cap_hours}h  families={families}  "
            f"julia_hook=on  per_family={args.per_family_min}m  pool=UNCAPPED(all-fresh-q3)  "
            f"retain_fields={args.retain_fields} field_cache={args.field_cache_gb}GB")
        log(f"  run ledger (empty, run-scoped): {ledger}")
        log(f"  library store: {sinks.records}  embeddings: {sinks.shards}")
        oo.save_state(state_path, {"run_id": args.run_id, "deadline_epoch": deadline,
                                   "cycles_done": 0, "started": time.strftime('%Y-%m-%dT%H:%M:%S')})

    oo.sweep_orphan_seeder_scratch(log)
    baseline_gpu = oo.gpu_used_mib()
    log(f"GPU baseline: {baseline_gpu} MiB" + ("" if baseline_gpu is not None
                                               else " (nvidia-smi unavailable; GPU checks skipped)"))

    est_loc_s = args.est_loc_s
    empty_streak = 0
    totals = {"cycles": 0, "discovery_families": 0, "fresh_q3": 0, "records_added": 0,
              "phase_failures": 0}
    cycle_recon = []        # per-cycle reconciliation breakdowns (records / coord-dup / leak)
    failed_cycles = []      # cycles whose annotate failed twice: {cycle, q3_deferred, reason, ...}
    fam_instr = []          # per-cycle per-family base/twin/parent instrumentation (Part 2)
    last_heartbeat = time.time()

    def remaining():
        return deadline - time.time()

    # ----------------------------- cycle loop ----------------------------- #
    while True:
        # a minimally-productive cycle needs one family of discovery + one location annotated.
        min_cycle_s = per_family_s + EST_ANNOTATE_FIXED_S + est_loc_s
        if remaining() < min_cycle_s:
            log(f"CAP: {remaining()/60:.1f}m left < min cycle {min_cycle_s/60:.1f}m — "
                f"stopping cleanly at a phase boundary.")
            break
        if empty_streak >= oo.MAX_EMPTY_CYCLES:
            log(f"SATURATION: {empty_streak} consecutive zero-fresh-q3 cycles — stopping.")
            break

        cycle += 1
        watermark = oo.ledger_line_count(ledger)
        log(f"===== CYCLE {cycle} =====  remaining {remaining()/3600:.2f}h  "
            f"ledger watermark {watermark}")

        # ---- PHASE 1: discovery, per family (GPU-exclusive) — VERBATIM invocation ----
        # Discovery-budget rebalance (Part 2): multibrot3/4/5 c-plane descent is barren at the
        # EMITTED level but supplies the julia-hook parents (the productive part), so it's a
        # per-family BUDGET knob, not a drop. --mb-cplane-min overrides the c-plane budget for
        # multibrot3/4/5 (default: == per_family_min, i.e. NO cut — a conservative default, tune
        # from the instrumented long run); --mb5-every runs multibrot5 only every Nth cycle.
        seeder_summ_dir = run_dir / "seeder_summaries"
        seeder_summ_dir.mkdir(parents=True, exist_ok=True)
        fams_run = 0
        for fi, fam in enumerate(families):
            if fam == "multibrot5" and (cycle - 1) % max(1, args.mb5_every) != 0:
                log(f"  mb5-every {args.mb5_every}: skipping multibrot5 c-plane discovery "
                    f"this cycle {cycle} (its twin isn't zero, so it's skipped, not dropped).")
                continue
            fam_budget_min = _family_cplane_min(fam, args)
            fam_budget_s = fam_budget_min * 60.0
            if remaining() < fam_budget_s:
                log(f"  CAP: {remaining()/60:.1f}m < family budget {fam_budget_min}m — "
                    f"skipping remaining discovery families this cycle.")
                break
            seed = args.seed + cycle * 1000 + fi * 137
            summ_out = seeder_summ_dir / f"cycle_{cycle:03d}_{fam}.json"
            cmd = [oo.PY, "-u", str(oo.SEEDER), "--run", "--discovery-dir", str(disc_dir),
                   "--family", fam, "--julia-hook",
                   "--budget", str(fam_budget_min), "--seed", str(seed),
                   "--summary-out", str(summ_out)]
            if args.disc_batch:
                cmd += ["--batch", str(args.disc_batch)]
            oo.gpu_boundary(log, baseline_gpu, f"pre-discovery[{fam}]")
            fam_pre = oo.ledger_line_count(ledger)
            try:
                res = oo.run_phase(log, f"discovery:{fam}", cmd, fam_budget_s)
                timing.record(cycle, f"discovery:{fam}", res,
                              ledger_rows_added=oo.ledger_line_count(ledger) - fam_pre,
                              fresh_q3=len(oo.new_fresh_q3(ledger, fam_pre)))
                if not res["ok"]:
                    totals["phase_failures"] += 1
                    log(f"  discovery:{fam} FAILED (isolated) — continuing", "WARN")
                else:
                    fams_run += 1
                _record_family_instr(fam_instr, cycle, fam, fam_budget_min, summ_out, log)
            except Exception as e:
                totals["phase_failures"] += 1
                log(f"  discovery:{fam} EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
            oo.gpu_boundary(log, baseline_gpu, f"post-discovery[{fam}]")
        totals["discovery_families"] += fams_run

        # ---- PHASE 1b: phoenix discovery (native z-descent -> same ledger) — VERBATIM ----
        phoenix_s = args.phoenix_min * 60.0
        if args.phoenix_min > 0 and remaining() >= phoenix_s:
            pseed = args.seed + cycle * 1000 + 900
            cmd = [oo.PY, "-u", str(oo.SEEDER), "--run-phoenix", "--discovery-dir", str(disc_dir),
                   "--budget", str(args.phoenix_min), "--seed", str(pseed)]
            if args.phoenix_walks:
                cmd += ["--phoenix-walks", str(args.phoenix_walks)]
            if args.disc_batch:
                cmd += ["--batch", str(args.disc_batch)]
            oo.gpu_boundary(log, baseline_gpu, "pre-discovery[phoenix]")
            ph_pre = oo.ledger_line_count(ledger)
            try:
                res = oo.run_phase(log, "discovery:phoenix", cmd, phoenix_s)
                timing.record(cycle, "discovery:phoenix", res,
                              ledger_rows_added=oo.ledger_line_count(ledger) - ph_pre,
                              fresh_q3=len(oo.new_fresh_q3(ledger, ph_pre)))
                if not res["ok"]:
                    totals["phase_failures"] += 1
                    log("  discovery:phoenix FAILED (isolated) — continuing", "WARN")
            except Exception as e:
                totals["phase_failures"] += 1
                log(f"  discovery:phoenix EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
            oo.gpu_boundary(log, baseline_gpu, "post-discovery[phoenix]")
        elif args.phoenix_min > 0:
            log(f"  CAP: {remaining()/60:.1f}m < phoenix budget {args.phoenix_min}m — "
                f"skipping phoenix discovery this cycle.")

        # ---- fresh-q3 over ONLY this cycle's appended rows ----
        fresh = oo.new_fresh_q3(ledger, watermark)
        log(f"  cycle {cycle} discovery: {fams_run}/{len(families)} families ran, "
            f"+{oo.ledger_line_count(ledger) - watermark} ledger rows, {len(fresh)} fresh q3")
        totals["fresh_q3"] += len(fresh)
        if not fresh:
            empty_streak += 1
            log(f"  no fresh q3 this cycle (empty streak {empty_streak}); skipping pool+annotate.")
            totals["cycles"] += 1
            oo.save_state(state_path, {**oo.load_state(state_path), "cycles_done": cycle})
            oo.purge_cycle_intermediates(log, cycle)
            continue
        empty_streak = 0

        # ---- PHASE 2: pool build (ONLY this cycle's fresh q3s) — VERBATIM invocation ----
        if remaining() < oo.EST_POOL_LOC_S:
            log(f"  CAP: {remaining()/60:.1f}m left — insufficient for pool build; stopping.")
            break
        batch_dir = run_dir / "pools" / f"cycle_{cycle:03d}"
        # UNCAPPED (Phase-1 thesis: mostly stop discarding). Every fresh q3 is pooled — the pool
        # phase performs NO selection; only a true coordinate-duplicate of a record already in the
        # store drops a location, and that decision lives downstream in library_annotate. --count /
        # --pool-limit are inherited from the emit orchestrator (volume-bound, so a cap is correct
        # there); Phase 1 has no such bound, so they must not decide which locations survive.
        est_pool_s = len(fresh) * oo.EST_POOL_LOC_S
        cmd = [oo.PY, "-u", str(oo.POOL_BUILDER), "--ledger", str(ledger),
               "--ledger-start-line", str(watermark), "--batch-dir", str(batch_dir),
               "--pool-all", "--no-head-exclude", "--seed", str(args.seed)]
        oo.gpu_boundary(log, baseline_gpu, "pre-pool")
        try:
            res = oo.run_phase(log, f"pool:cycle{cycle}", cmd, est_pool_s)
        except Exception as e:
            res = {"ok": False}
            log(f"  pool build EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
        oo.gpu_boundary(log, baseline_gpu, "post-pool")
        pool_locs = oo.ledger_line_count(batch_dir / "images.jsonl")
        timing.record(cycle, f"pool:cycle{cycle}", res,
                      fresh_q3_in=len(fresh), pool_locations=pool_locs)
        if not res["ok"] or not (batch_dir / "images.jsonl").exists():
            totals["phase_failures"] += 1
            log(f"  pool build failed / no batch (isolated) — skipping annotate this cycle.", "WARN")
            totals["cycles"] += 1
            oo.save_state(state_path, {**oo.load_state(state_path), "cycles_done": cycle})
            oo.purge_cycle_intermediates(log, cycle, batch_dir)
            continue

        # ---- the non-negotiable freshness assertion (halts on violation) — VERBATIM ----
        oo.assert_fresh_isolation(log, ledger, watermark, batch_dir)

        # ---- PHASE 3: ANNOTATE + PERSIST LIBRARY RECORD (replaces emit) ----
        n_unique = _unique_pool_locations(batch_dir / "images.jsonl")
        est_annotate_s = EST_ANNOTATE_FIXED_S + n_unique * est_loc_s
        if remaining() < EST_ANNOTATE_FIXED_S + est_loc_s:
            log(f"  CAP: {remaining()/60:.1f}m left — no budget for annotate; stopping.")
            break
        recs_before = store_summary_count(sinks)
        cmd = _annotate_cmd(batch_dir, ledger, watermark, args.run_id, cycle, sinks,
                            args.field_cache_gb, args.retain_fields)
        ann_report_path = batch_dir / "annotate_report.json"
        res_box = {}

        def _attempt_annotate(i):
            """One annotate invocation (i=0 initial, i=1 retry). Operational success == run_phase ok
            AND the report was written (a clean annotate writes it last; a missing report == crashed
            mid-cycle). Annotate dedups against the store, so a retry over the same pool is
            idempotent: records a partial first attempt persisted reappear as coord-dups, not
            double-writes. Increments phase_failures per failed attempt."""
            tag = "" if i == 0 else "(retry)"
            oo.gpu_boundary(log, baseline_gpu, f"pre-annotate{tag}")
            try:
                r = oo.run_phase(log, f"annotate:cycle{cycle}{tag}", cmd, est_annotate_s)
            except Exception as e:
                r = {"ok": False}
                log(f"  annotate{tag} EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
            oo.gpu_boundary(log, baseline_gpu, f"post-annotate{tag}")
            res_box["res"] = r
            ok = bool(r.get("ok")) and ann_report_path.exists()
            if not ok:
                totals["phase_failures"] += 1
                reason = ("run_phase not ok" if not r.get("ok")
                          else "annotate_report.json missing (crash)")
            else:
                reason = ""
            return ok, reason

        t_ann = time.time()
        # OPERATIONAL failure (crash / OOM / transient) is NOT selection-shaped loss: the durable
        # run ledger makes the cycle re-runnable, so we retry once and, if it still fails, record the
        # cycle as failed-with-reason and CONTINUE — never halt the run for flakiness.
        fc = annotate_with_retry(_attempt_annotate, cycle, len(fresh), watermark,
                                 lambda: store_summary_count(sinks) - recs_before, log=log)
        res = res_box["res"]
        if fc is not None:
            totals["records_added"] += fc["records_salvaged"]
            failed_cycles.append(fc)
            log(f"  annotate cycle {cycle} FAILED TWICE — {fc['q3_deferred']} q3 deferred to a "
                f"re-run (ledger durable @ watermark {watermark}); {fc['records_salvaged']} salvaged. "
                f"Continuing.", "WARN")
            timing.record(cycle, f"annotate:cycle{cycle}", res, pool_locations=pool_locs,
                          unique_locations=n_unique, records_added=fc["records_salvaged"], failed=True)
            totals["cycles"] += 1
            oo.save_state(state_path, {**oo.load_state(state_path), "cycles_done": cycle,
                                       "totals": totals, "failed_cycles": failed_cycles})
            # NOTE: intentionally NO purge_cycle_intermediates here — the cycle is re-runnable.
            continue

        recs_after = store_summary_count(sinks)
        n_added = recs_after - recs_before
        totals["records_added"] += n_added
        timing.record(cycle, f"annotate:cycle{cycle}", res,
                      pool_locations=pool_locs, unique_locations=n_unique, records_added=n_added)
        if n_added > 0:
            obs = (time.time() - t_ann - EST_ANNOTATE_FIXED_S) / n_added
            if obs > 0:
                est_loc_s = 0.5 * est_loc_s + 0.5 * obs
        log(f"  cycle {cycle} annotate: +{n_added} library records "
            f"(store now {recs_after}; est_loc now {est_loc_s:.1f}s)")

        # ---- reconciliation (per cycle): every fresh q3 accounted as recorded | coord_dup |
        # field_fail (rate-gated) | deferred; only selection-shaped or unexplained loss halts. ----
        sel_report = _read_json(batch_dir / "selection_report.json")
        ann_report = _read_json(ann_report_path)
        bd = reconcile_cycle(len(fresh), n_added, sel_report, ann_report, log=log)
        cycle_recon.append({"cycle": cycle, **bd})

        totals["cycles"] += 1
        oo.save_state(state_path, {**oo.load_state(state_path), "cycles_done": cycle,
                                   "totals": totals, "failed_cycles": failed_cycles})
        # cycle-boundary self-clean: records+embeddings are durable in the store, fresh-isolation
        # asserted before annotate — pool crops + killed-seeder scratch are safe to reclaim.
        oo.purge_cycle_intermediates(log, cycle, batch_dir)

        if time.time() - last_heartbeat > oo.HEARTBEAT_EVERY_S:
            last_heartbeat = time.time()
        log(f"HEARTBEAT cycle={cycle} remaining={remaining()/3600:.2f}h records_total="
            f"{totals['records_added']} fresh_q3_total={totals['fresh_q3']} "
            f"phase_failures={totals['phase_failures']}")

    # ----------------------------- final report ----------------------------- #
    fr, ff = oo._reclaim_seeder_scratch(log, "final")
    if fr:
        log(f"final seeder-scratch sweep: reclaimed {fr} dir(s), ~{ff/2**30:.2f} GiB")
    oo.gpu_boundary(log, baseline_gpu, "final")
    summ = store.store_summary(sinks.records, sinks.thumbs, sinks.shards)
    summary = {
        "run_id": args.run_id, "finished": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "cap_hours": args.cap_hours, "families": families, "julia_hook": True,
        "cycles": totals["cycles"], "discovery_families_run": totals["discovery_families"],
        "fresh_q3_total": totals["fresh_q3"], "records_added_this_run": totals["records_added"],
        "phase_failures": totals["phase_failures"],
        "library_records_total": summ["records"], "by_family": summ["by_family"],
        "thumbnails": summ["thumbs"], "embedding_shards": summ["shards"],
        "embeddings_total": summ["embeddings_total"],
        "records_path": str(sinks.records),
        "embeddings_shards": str(sinks.shards),
        "timing_jsonl": str((run_dir / "timing.jsonl")),
        "run_ledger": str(ledger), "run_ledger_rows": oo.ledger_line_count(ledger),
        "output_tree": str(run_dir),
        # failed cycles surfaced at the TOP level (the over-coffee glance), not buried in recon
        "cycles_failed": len(failed_cycles),
        "q3_deferred_to_rerun": sum(c["q3_deferred"] for c in failed_cycles),
        "failed_cycles": failed_cycles,
        # per-cycle harvest reconciliation (Part 1) + per-family base/twin/parent (Part 2)
        "reconciliation": {
            "per_cycle": cycle_recon,
            "totals": {
                "q3_found": sum(c["q3_found"] for c in cycle_recon),
                "records_written": sum(c["records_written"] for c in cycle_recon),
                "dropped_coord_dup": sum(c["dropped_coord_dup"] for c in cycle_recon),
                "field_fail": sum(c.get("field_fail", 0) for c in cycle_recon),
                "unexplained": sum(c.get("unexplained", 0) for c in cycle_recon),
                "dropped_other": sum(c["dropped_other"] for c in cycle_recon),
            },
        },
        "family_instrumentation": fam_instr,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rt = summary["reconciliation"]["totals"]
    log("=" * 70)
    # over-coffee banner: a failed cycle must be impossible to miss at the top of the summary.
    if failed_cycles:
        log("!" * 70)
        log(f"!!  {len(failed_cycles)} CYCLE(S) FAILED (annotate failed twice) — "
            f"{summary['q3_deferred_to_rerun']} q3 DEFERRED to a re-run (durable ledger).")
        for fc in failed_cycles:
            log(f"!!    cycle {fc['cycle']}: {fc['q3_deferred']} q3 deferred, "
                f"{fc['records_salvaged']} salvaged — {fc['reason']} (re-run from ledger).")
        log("!" * 70)
    else:
        log("  all cycles reconciled cleanly (0 failed).")
    log(f"RUN COMPLETE '{args.run_id}': {totals['cycles']} cycles ({len(failed_cycles)} failed), "
        f"+{totals['records_added']} library records this run "
        f"({summ['records']} total in store), {totals['fresh_q3']} fresh q3, "
        f"{totals['phase_failures']} phase failures")
    log(f"  reconciliation: q3_found={rt['q3_found']} = records={rt['records_written']} + "
        f"coord_dup={rt['dropped_coord_dup']} + field_fail={rt['field_fail']} "
        f"+ unexplained={rt['unexplained']} (unexplained MUST be 0)")
    for fam in sorted({r["family"] for r in fam_instr}):
        rows = [r for r in fam_instr if r["family"] == fam]
        def _s(k):
            return sum(r[k] or 0 for r in rows)
        log(f"  rebalance[{fam}]: cplane_desc={_s('cplane_descents')} "
            f"qual_parents={_s('qualifying_parents')} hook_desc={_s('hook_descents')} "
            f"q3 base={_s('fresh_q3_base')} twin={_s('fresh_q3_twin')} (over {len(rows)} cycle-runs)")
    log(f"  library records -> {sinks.records}")
    log(f"  embeddings -> {sinks.shards} ({summ['shards']} shards, {summ['embeddings_total']} vecs)")
    log(f"  thumbnails -> {sinks.thumbs} ({summ['thumbs']})")
    log(f"  run summary -> {run_dir / 'run_summary.json'}")
    log.close()
    timing.close()
    return summary


class _StoreSinks:
    """The three durable-store sink paths, overridable so --mini / tests don't touch the
    production library. Defaults are library_store's fixed data/ paths."""
    def __init__(self, records: Path, thumbs: Path, shards: Path, field_cache: Path):
        self.records, self.thumbs, self.shards, self.field_cache = \
            records, thumbs, shards, field_cache

    def annotate_flags(self) -> list:
        return ["--records", str(self.records), "--thumbs", str(self.thumbs),
                "--emb-shards", str(self.shards), "--field-cache-dir", str(self.field_cache)]


def _store_sinks(args) -> _StoreSinks:
    return _StoreSinks(
        Path(args.store_records) if args.store_records else store.RECORDS_PATH,
        Path(args.store_thumbs) if args.store_thumbs else store.THUMBS_DIR,
        Path(args.store_emb_shards) if args.store_emb_shards else store.EMB_SHARDS,
        Path(args.store_field_cache) if args.store_field_cache else store.FIELD_CACHE_DIR)


def store_summary_count(sinks: _StoreSinks) -> int:
    return store.store_summary(sinks.records, sinks.thumbs, sinks.shards)["records"]


def _annotate_cmd(batch_dir: Path, ledger: Path, watermark: int, run_id: str, cycle: int,
                  sinks: _StoreSinks, field_cache_gb: float, retain_fields: bool) -> list:
    """The library_annotate invocation for ONE cycle's pool. Shared by the main loop and
    --rerun-failed so the two can never build a divergent annotate command."""
    cmd = [oo.PY, "-u", str(ANNOTATOR), "--pool", str(batch_dir), "--ledger", str(ledger),
           "--ledger-start-line", str(watermark), "--run-id", run_id,
           "--cycle", str(cycle), "--field-cache-gb", str(field_cache_gb)]
    cmd += sinks.annotate_flags()
    if not retain_fields:
        cmd += ["--no-retain-fields"]
    return cmd


def _annotate_pool(batch_dir: Path, ledger: Path, watermark: int, run_id: str, cycle: int,
                   sinks: _StoreSinks, field_cache_gb: float, retain_fields: bool,
                   est_annotate_s: float, log, baseline_gpu, tag: str):
    """Run ONE annotate subprocess over a (retained) pool to completion. Returns (ok, res):
    ok == run_phase ok AND the annotate report was written (a clean annotate writes it last; a
    missing report == crashed mid-cycle). Isolated seam so the GPU-free drain test substitutes the
    store-append path for the real subprocess."""
    cmd = _annotate_cmd(batch_dir, ledger, watermark, run_id, cycle, sinks, field_cache_gb,
                        retain_fields)
    ann_report_path = Path(batch_dir) / "annotate_report.json"
    oo.gpu_boundary(log, baseline_gpu, f"pre-{tag}")
    try:
        res = oo.run_phase(log, tag, cmd, est_annotate_s)
    except Exception as e:
        res = {"ok": False}
        log(f"  {tag} EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
    oo.gpu_boundary(log, baseline_gpu, f"post-{tag}")
    return bool(res.get("ok")) and ann_report_path.exists(), res


def rerun_failed(args):
    """Drain a run's DEFERRED failed cycles: re-annotate each from its RETAINED pool, landing the
    q3 that `annotate failed twice` left logged-but-lost (`q3_deferred_to_rerun`). The failure path
    deliberately skips `purge_cycle_intermediates`, so each failed cycle's pool is still on disk and
    re-annotatable at the same geometry; the deferred entry carries its `ledger_watermark`.

    Idempotent BY CONSTRUCTION via store-dedup — no extra bookkeeping. A re-annotate over the same
    pool re-appends the same location_ids: whatever a partial first attempt already persisted comes
    back as a store coord-dup (0 net records), the rest lands. So a cycle drained twice adds records
    once then zero, and reconciliation balances every pass (q3_found == records + coord_dup +
    field_fail). Cannot rebuild a purged pool from the watermark: the pool builder reads
    watermark..EOF, which in a finished run spans LATER cycles too — the retained pool is the only
    correct source, so a missing one is reported and left deferred, never reconstructed."""
    out_root = Path(args.out_root).resolve()
    disc_root = Path(args.discovery_root).resolve()
    run_dir = out_root / args.run_id
    disc_dir = disc_root / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"--rerun-failed: run dir not found: {run_dir} "
                         f"(wrong --run-id / --out-root / --discovery-root?)")
    ledger = disc_dir / "outcome_ledger.jsonl"
    state_path = run_dir / "state.json"
    log = oo.Log(run_dir / "orchestrator.log")
    sinks = _store_sinks(args)

    state = oo.load_state(state_path)
    failed = list((state or {}).get("failed_cycles", []))
    if not failed:
        log(f"RERUN-FAILED '{args.run_id}': no deferred failed cycles recorded — nothing to drain.")
        log.close()
        return {"run_id": args.run_id, "records_added": 0, "drained": [], "still_failed": []}

    baseline_gpu = oo.gpu_used_mib()
    log(f"RERUN-FAILED '{args.run_id}': draining {len(failed)} deferred cycle(s) "
        f"(q3_deferred_total={sum(fc.get('q3_deferred', 0) for fc in failed)}); store={sinks.records}")
    drained, still_failed, total_added = [], [], 0
    for fc in failed:
        cycle = fc["cycle"]
        watermark = fc.get("ledger_watermark", 0)
        batch_dir = run_dir / "pools" / f"cycle_{cycle:03d}"
        images = batch_dir / "images.jsonl"
        if not images.exists():
            log(f"  cycle {cycle}: retained pool absent ({images}) — cannot re-derive; "
                f"leaving deferred.", "WARN")
            still_failed.append(fc)
            continue
        recs_before = store_summary_count(sinks)
        n_unique = _unique_pool_locations(images)
        est_annotate_s = EST_ANNOTATE_FIXED_S + n_unique * args.est_loc_s
        ok, _res = _annotate_pool(batch_dir, ledger, watermark, args.run_id, cycle, sinks,
                                  args.field_cache_gb, args.retain_fields, est_annotate_s,
                                  log, baseline_gpu, f"rerun-annotate:cycle{cycle}")
        if not ok:
            log(f"  cycle {cycle}: rerun annotate did not complete — leaving deferred.", "WARN")
            still_failed.append(fc)
            continue
        n_added = store_summary_count(sinks) - recs_before
        total_added += n_added
        # Reconcile exactly like a live cycle (deferred=0 now — the deferral is being resolved).
        sel_report = _read_json(batch_dir / "selection_report.json")
        ann_report = _read_json(batch_dir / "annotate_report.json")
        bd = reconcile_cycle(fc.get("q3_deferred", 0), n_added, sel_report, ann_report, log=log)
        drained.append({"cycle": cycle, "records_added": n_added, **bd})
        log(f"  cycle {cycle} DRAINED: +{n_added} records "
            f"(q3_deferred={fc.get('q3_deferred', 0)}; reconciled clean).")

    # A fully-drained cycle leaves failed_cycles (its q3 are now recorded-or-coord-dup); a cycle we
    # couldn't re-derive stays. Re-running is safe: an already-drained cycle re-injected here adds 0.
    if state is not None:
        state["failed_cycles"] = still_failed
        oo.save_state(state_path, state)
    report = {"run_id": args.run_id, "records_added": total_added,
              "drained": drained, "still_failed": still_failed}
    (run_dir / "rerun_failed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("=" * 70)
    log(f"RERUN-FAILED COMPLETE '{args.run_id}': drained {len(drained)} cycle(s), "
        f"+{total_added} records; {len(still_failed)} still deferred.")
    log(f"  report -> {run_dir / 'rerun_failed_report.json'}")
    log.close()
    return report


def _apply_mini_defaults(args) -> None:
    """Remap args to the throwaway mini profile (tight cap, 1 family, SCRATCH store roots). Shared
    by a fresh `--mini` run and `--mini --rerun-failed` (so a mini run's deferred cycle drains
    against the same scratch store, never the production library)."""
    scratch = oo.ROOT / "out" / "wallpaper" / "prospect_mini_scratch"
    args.out_root = str(scratch / "out")
    args.discovery_root = str(scratch / "disc")
    if args.cap_hours == CAP_HOURS:
        args.cap_hours = 0.75
    if args.per_family_min == PER_FAMILY_MIN:
        args.per_family_min = 2.0
    if args.pool_count == POOL_COUNT:
        args.pool_count = 12
    if args.disc_batch == 0:
        args.disc_batch = 6
    if args.pool_limit == 0:
        args.pool_limit = 4
    if args.families == oo.FAMILIES:
        args.families = ["mandelbrot"]
    if args.phoenix_min == PHOENIX_PER_CYCLE_MIN:
        args.phoenix_min = 15.0
    if args.phoenix_walks == 0:
        args.phoenix_walks = 8
    # keep the smoke's library out of the production store
    if args.store_records is None:
        args.store_records = str(scratch / "library" / "records.jsonl")
    if args.store_thumbs is None:
        args.store_thumbs = str(scratch / "library" / "thumbs")
    if args.store_emb_shards is None:
        args.store_emb_shards = str(scratch / "library_embeddings" / "shards")
    if args.store_field_cache is None:
        args.store_field_cache = str(scratch / "library" / "field_cache")
    print(f"[mini] throwaway roots under {scratch} (store -> scratch, not data/library)")


def _unique_pool_locations(images_jsonl: Path) -> int:
    if not images_jsonl.exists():
        return 0
    seen = set()
    with open(images_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            oid = (json.loads(line).get("provenance") or {}).get("source_oid")
            if oid:
                seen.add(oid)
    return len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="production run (24h cap by default)")
    ap.add_argument("--mini", action="store_true",
                    help="short throwaway validation run: tight cap, 1 family, scratch roots")
    ap.add_argument("--rerun-failed", dest="rerun_failed", action="store_true",
                    help="drain an EXISTING run's deferred failed cycles (needs --run-id, same "
                         "--out-root/--discovery-root/store the run used; add --mini for a mini "
                         "run's scratch store). Re-annotates each retained pool; idempotent via "
                         "store-dedup (a re-drain adds 0). Launches no discovery/pool phase.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    ap.add_argument("--per-family-min", type=float, default=PER_FAMILY_MIN)
    ap.add_argument("--phoenix-min", type=float, default=PHOENIX_PER_CYCLE_MIN)
    ap.add_argument("--phoenix-walks", type=int, default=0)
    # --- discovery-budget rebalance (Part 2): multibrot3/4/5 c-plane is barren at the emitted
    # level but supplies the julia-hook parents, so it's a per-family BUDGET knob, not a drop. ---
    ap.add_argument("--mb-cplane-min", type=float, default=None,
                    help="c-plane discovery budget (minutes) for multibrot3/4/5 (default: "
                         "== --per-family-min, i.e. NO cut). Dial DOWN to spend less on the barren "
                         "high-degree c-plane base while keeping enough parent supply for the "
                         "productive julia twins; watch fresh_q3_twin / qualifying_parents in the "
                         "run summary so a cut doesn't starve the hooks.")
    ap.add_argument("--mb5-every", type=int, default=1,
                    help="run multibrot5 c-plane discovery only every Nth cycle (default 1 = every "
                         "cycle). Its twin isn't zero, so mb5 is throttled, never deleted.")
    ap.add_argument("--pool-count", type=int, default=POOL_COUNT,
                    help="INERT for selection (the pool is uncapped: --pool-all pools every fresh "
                         "q3). Kept for CLI/log compatibility only.")
    ap.add_argument("--est-loc-s", type=float, default=EST_ANNOTATE_LOC_S,
                    help="initial per-location annotate estimate (refined at runtime)")
    ap.add_argument("--field-cache-gb", type=float, default=FIELD_CACHE_GB)
    ap.add_argument("--retain-fields", dest="retain_fields", action="store_true", default=True)
    ap.add_argument("--no-retain-fields", dest="retain_fields", action="store_false")
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--out-root", default=str(oo.ROOT / "out" / "wallpaper" / "prospect"))
    ap.add_argument("--discovery-root", default=str(oo.ROOT / "data" / "discovery" / "fresh_runs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disc-batch", type=int, default=0)
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="INERT for selection (the pool is uncapped; a cap would silently discard "
                         "the very q3s Phase 1 exists to keep). Kept for CLI compatibility only.")
    # durable-store sink overrides (default: library_store's data/ paths; --mini -> scratch)
    ap.add_argument("--store-records", default=None)
    ap.add_argument("--store-thumbs", default=None)
    ap.add_argument("--store-emb-shards", default=None)
    ap.add_argument("--store-field-cache", default=None)
    args = ap.parse_args()

    # --rerun-failed: drain an existing run's deferred cycles; no fresh discovery/pool loop.
    if args.rerun_failed:
        if args.run_id is None:
            raise SystemExit("--rerun-failed requires --run-id <run> (the run to drain)")
        if args.mini:
            _apply_mini_defaults(args)   # point at the same scratch store the mini run used
        rerun_failed(args)
        return

    if not (args.run or args.mini):
        ap.print_help()
        return
    if args.run_id is None:
        args.run_id = ("prospect_mini_" if args.mini else "prospect_") + time.strftime("%Y%m%d_%H%M%S")
    if args.families is None:
        args.families = oo.FAMILIES

    if args.mini:
        _apply_mini_defaults(args)
    elif args.disc_batch == 0:
        args.disc_batch = 12

    orchestrate(args)


if __name__ == "__main__":
    main()
