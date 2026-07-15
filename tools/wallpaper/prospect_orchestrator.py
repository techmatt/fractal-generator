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

ANNOTATOR = oo.ROOT / "tools" / "wallpaper" / "library_annotate.py"


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
            f"julia_hook=on  per_family={args.per_family_min}m  pool_count={args.pool_count}  "
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
        fams_run = 0
        for fi, fam in enumerate(families):
            if remaining() < per_family_s:
                log(f"  CAP: {remaining()/60:.1f}m < family budget {args.per_family_min}m — "
                    f"skipping remaining discovery families this cycle.")
                break
            seed = args.seed + cycle * 1000 + fi * 137
            cmd = [oo.PY, "-u", str(oo.SEEDER), "--run", "--discovery-dir", str(disc_dir),
                   "--family", fam, "--julia-hook",
                   "--budget", str(args.per_family_min), "--seed", str(seed)]
            if args.disc_batch:
                cmd += ["--batch", str(args.disc_batch)]
            oo.gpu_boundary(log, baseline_gpu, f"pre-discovery[{fam}]")
            fam_pre = oo.ledger_line_count(ledger)
            try:
                res = oo.run_phase(log, f"discovery:{fam}", cmd, per_family_s)
                timing.record(cycle, f"discovery:{fam}", res,
                              ledger_rows_added=oo.ledger_line_count(ledger) - fam_pre,
                              fresh_q3=len(oo.new_fresh_q3(ledger, fam_pre)))
                if not res["ok"]:
                    totals["phase_failures"] += 1
                    log(f"  discovery:{fam} FAILED (isolated) — continuing", "WARN")
                else:
                    fams_run += 1
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
        est_pool_s = min(len(fresh), args.pool_count) * oo.EST_POOL_LOC_S
        cmd = [oo.PY, "-u", str(oo.POOL_BUILDER), "--ledger", str(ledger),
               "--ledger-start-line", str(watermark), "--batch-dir", str(batch_dir),
               "--count", str(args.pool_count), "--seed", str(args.seed)]
        if args.pool_limit:
            cmd += ["--limit", str(args.pool_limit)]
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
        cmd = [oo.PY, "-u", str(ANNOTATOR), "--pool", str(batch_dir), "--ledger", str(ledger),
               "--ledger-start-line", str(watermark), "--run-id", args.run_id,
               "--cycle", str(cycle), "--field-cache-gb", str(args.field_cache_gb)]
        cmd += sinks.annotate_flags()
        if not args.retain_fields:
            cmd += ["--no-retain-fields"]
        oo.gpu_boundary(log, baseline_gpu, "pre-annotate")
        t_ann = time.time()
        try:
            res = oo.run_phase(log, f"annotate:cycle{cycle}", cmd, est_annotate_s)
        except Exception as e:
            res = {"ok": False}
            log(f"  annotate EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
        oo.gpu_boundary(log, baseline_gpu, "post-annotate")

        recs_after = store_summary_count(sinks)
        n_added = recs_after - recs_before
        totals["records_added"] += n_added
        timing.record(cycle, f"annotate:cycle{cycle}", res,
                      pool_locations=pool_locs, unique_locations=n_unique, records_added=n_added)
        if not res["ok"]:
            totals["phase_failures"] += 1
            log(f"  annotate reported failure (isolated); {n_added} records still persisted.", "WARN")
        if n_added > 0:
            obs = (time.time() - t_ann - EST_ANNOTATE_FIXED_S) / n_added
            if obs > 0:
                est_loc_s = 0.5 * est_loc_s + 0.5 * obs
        log(f"  cycle {cycle} annotate: +{n_added} library records "
            f"(store now {recs_after}; est_loc now {est_loc_s:.1f}s)")

        totals["cycles"] += 1
        oo.save_state(state_path, {**oo.load_state(state_path), "cycles_done": cycle,
                                   "totals": totals})
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
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("=" * 70)
    log(f"RUN COMPLETE '{args.run_id}': {totals['cycles']} cycles, "
        f"+{totals['records_added']} library records this run "
        f"({summ['records']} total in store), {totals['fresh_q3']} fresh q3, "
        f"{totals['phase_failures']} phase failures")
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
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS)
    ap.add_argument("--per-family-min", type=float, default=PER_FAMILY_MIN)
    ap.add_argument("--phoenix-min", type=float, default=PHOENIX_PER_CYCLE_MIN)
    ap.add_argument("--phoenix-walks", type=int, default=0)
    ap.add_argument("--pool-count", type=int, default=POOL_COUNT)
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
    ap.add_argument("--pool-limit", type=int, default=0)
    # durable-store sink overrides (default: library_store's data/ paths; --mini -> scratch)
    ap.add_argument("--store-records", default=None)
    ap.add_argument("--store-thumbs", default=None)
    ap.add_argument("--store-emb-shards", default=None)
    ap.add_argument("--store-field-cache", default=None)
    args = ap.parse_args()

    if not (args.run or args.mini):
        ap.print_help()
        return
    if args.run_id is None:
        args.run_id = ("prospect_mini_" if args.mini else "prospect_") + time.strftime("%Y%m%d_%H%M%S")
    if args.families is None:
        args.families = oo.FAMILIES

    if args.mini:
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
    elif args.disc_batch == 0:
        args.disc_batch = 12

    orchestrate(args)


if __name__ == "__main__":
    main()
