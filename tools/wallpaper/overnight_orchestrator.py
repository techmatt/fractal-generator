#!/usr/bin/env python
r"""Overnight emission orchestrator — unattended fresh-discovery -> eval-res emit loop.

Runs the emission chain continuously under a wall-clock cap for morning review. Each
CYCLE is: fresh discovery (production_seeder, all families, run-scoped ledger) -> pool
build (build_fresh_discovery over ONLY this cycle's fresh q3s) -> emit (emit_v1
--eval-res). Wallpapers + a durable aggregate manifest accrue every cycle, so a crash at
hour N keeps hours 1..N.

The three load-bearing invariants (see prompts/pipeline_orchestrator_prompt.md):

  1. GPU-EXCLUSIVE, DECOUPLED PHASES. Each phase is a child process that fully EXITS
     (releasing all GPU memory) before the next starts. Only ONE GPU consumer is ever
     resident. GPU residency is sampled (nvidia-smi) at every phase boundary and logged;
     a phase that leaves memory resident after exit is flagged as a suspected leak.

  2. FRESH-GENERATION ENFORCEMENT (non-negotiable). At run start a FRESH, EMPTY
     run-scoped discovery dir is created; discovery writes ONLY there (--discovery-dir);
     each cycle's pool reads ONLY that ledger, past a per-cycle line watermark
     (--ledger-start-line), so it pools ONLY the q3s that cycle's discovery just
     appended. An empty run ledger physically cannot contain a banked/historical q3. A
     per-cycle ASSERTION re-derives, from each pooled row's provenance.source_oid, that
     every pooled location originates from a ledger row this run wrote — the run HALTS on
     any violation.

  3. WALL-CLOCK CAP at the finest safe boundary: between phases, per discovery family,
     and (via emit_v1 --limit) per emit render. A unit that cannot finish in the
     remaining budget is never started. A hung unit (> KILL_MULT x its expected time) is
     hard-killed (process tree) and the loop continues.

Families: the c-plane families (mandelbrot, multibrot3/4/5) each with --julia-hook (adds
the julia:{fam} twins) run through the c-plane radius-rejection loop (`--run`). Phoenix has
NO parameter plane to prospect (resolve_family rejects --phoenix under --run), so it can't
ride that loop — but it IS freshly discoverable, just via a different recipe: a dedicated
phoenix phase runs the native z-plane descent (`--run-phoenix`) into the SAME run-scoped
ledger at its provisional t_good=0.18. That completes the 9-family set. Phoenix is a
low-yield, variety-poor garnish (~7 descents/keeper), so its phase gets an elevated-but-
bounded per-cycle descent budget (PHOENIX_PER_CYCLE_MIN) and its share of each emit pool is
capped by build_fresh_discovery's family-balanced round-robin (it can't dominate the loop).

    # validate first (short throwaway mini-run; backgrounded):
    uv run python -u tools/wallpaper/overnight_orchestrator.py --mini
    # the real 6h run (Matt starts the clock):
    uv run python -u tools/wallpaper/overnight_orchestrator.py --run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import corpus_common as cc  # noqa: E402  (is_v6_decoded — the v6-stamp discriminator)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PY = sys.executable   # the uv venv python this orchestrator runs under (torch cu124);
                      # children inherit the exact same interpreter/env, no `uv run` cost.
SEEDER = ROOT / "tools" / "atlas" / "production_seeder.py"
POOL_BUILDER = ROOT / "tools" / "wallpaper" / "build_fresh_discovery.py"
EMITTER = ROOT / "tools" / "wallpaper" / "emit_v1.py"

# All c-plane families; --julia-hook on each yields the julia:{fam} found-points too.
FAMILIES = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5"]

# --- production defaults (tunable) ---
CAP_HOURS = 6.0
PER_FAMILY_MIN = 7.0        # discovery budget per family per cycle
PHOENIX_PER_CYCLE_MIN = 10.0  # ELEVATED per-cycle phoenix z-descent budget (> PER_FAMILY_MIN):
                              # phoenix is a low-yield garnish (~7 descents/keeper at t_good=0.18),
                              # so it earns more time per keeper than a c-plane family. Bounded so
                              # it stays a modest stream (0 disables the phoenix phase entirely).
POOL_COUNT = 40             # build_fresh_discovery --count (family-balanced pool cap/cycle)
GATE = 0.90                 # emit_v1 --gate (v3 quality floor; smooth stem)
EST_EMIT_RENDER_S = 45.0    # per-winner eval-res (1024x576 ss2) render estimate; refined at runtime
EST_POOL_LOC_S = 120.0      # per-location pool-build estimate (beam + label crops)
KILL_MULT = 3.0             # hard-kill a unit exceeding this multiple of its expected time
KILL_SLACK_S = 300.0        # additive slack on top of KILL_MULT x expected (batch granularity)
LEAK_THRESH_MIB = 500       # residual GPU MiB above baseline after a phase exits -> suspected leak
MAX_EMPTY_CYCLES = 3        # consecutive zero-fresh-q3 cycles -> declare saturation, stop
HEARTBEAT_EVERY_S = 300     # idle heartbeat cadence (also logged at every phase boundary)


# =========================================================================== #
# Logging (stdout + durable run log)
# =========================================================================== #
class Log:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str, level: str = "INFO"):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{level}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# =========================================================================== #
# Timing (durable, structured per-phase wall-clock: timing.jsonl next to the
# aggregate manifest). One row per phase completion, appended+flushed on the
# spot — same crash/resume-safe pattern as the manifest, never held in memory.
# Pure instrumentation: wall-clock only, no extra render/scoring/GPU work, and
# no effect on any selection / gate / emit path.
# =========================================================================== #
class Timing:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.fh = open(path, "a", encoding="utf-8")

    def record(self, cycle: int, phase: str, res: dict, **yields):
        """Append one phase-completion row. `res` is a run_phase() result (carries
        start/end epoch + duration + rc/killed/ok); `yields` are whatever cost/yield
        counts are already on hand at the call site (ledger rows, fresh q3, emits)."""
        se, ee = res.get("start_epoch"), res.get("end_epoch")
        row = {
            "run_id": self.run_id, "cycle": cycle, "phase": phase,
            "start": None if se is None else time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(se)),
            "end": None if ee is None else time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ee)),
            "start_epoch": None if se is None else round(se, 3),
            "end_epoch": None if ee is None else round(ee, 3),
            "duration_s": None if res.get("elapsed_s") is None else round(res["elapsed_s"], 3),
            "rc": res.get("rc"), "killed": res.get("killed"), "ok": res.get("ok"),
        }
        row.update(yields)
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# =========================================================================== #
# GPU residency (best-effort; skipped if nvidia-smi is absent)
# =========================================================================== #
def gpu_used_mib():
    """Total GPU memory.used across visible GPUs (MiB), or None if nvidia-smi is absent."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        return sum(int(x) for x in r.stdout.split() if x.strip().isdigit())
    except Exception:
        return None


# =========================================================================== #
# Subprocess phase runner: streams child output to the run log, enforces a
# hard-kill backstop (process tree), and reports (ok, returncode, elapsed_s).
# =========================================================================== #
def _kill_tree(pid: int):
    """Windows-safe whole-tree kill (production_seeder/build/emit spawn render-one +
    guided-descend children; killing only the parent would orphan them)."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, timeout=30)
    except Exception:
        pass


def run_phase(log: Log, name: str, cmd: list, expected_s: float) -> dict:
    """Run one GPU phase to completion. The child's stdout/stderr are appended to the run
    log so the morning review can reconstruct the night. A child that runs past
    KILL_MULT x expected (+ slack) is a hung GPU op — killed (tree) and reported failed;
    the loop continues (per-phase failure isolation is the caller's job)."""
    kill_after = max(KILL_MULT * expected_s, expected_s + KILL_SLACK_S)
    log(f"PHASE start '{name}'  expected~{expected_s/60:.1f}m  kill@{kill_after/60:.1f}m")
    log(f"  cmd: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    log.fh.flush()
    # Child output -> the run log file (same fd) so it interleaves durably.
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log.fh,
                            stderr=subprocess.STDOUT)
    killed = False
    try:
        proc.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        killed = True
        log(f"PHASE '{name}' HUNG past {kill_after/60:.1f}m -> hard-killing tree (pid {proc.pid})",
            "WARN")
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    dt = time.time() - t0
    rc = proc.returncode
    ok = (not killed) and (rc == 0)
    log(f"PHASE end   '{name}'  rc={rc} killed={killed}  [{dt/60:.1f}m]",
        "INFO" if ok else "WARN")
    return {"ok": ok, "rc": rc, "elapsed_s": dt, "killed": killed,
            "start_epoch": t0, "end_epoch": t0 + dt}


def gpu_boundary(log: Log, baseline: int | None, when: str):
    """Sample + log GPU residency at a phase boundary; flag a suspected leak if a phase
    left memory resident after its process exited."""
    used = gpu_used_mib()
    if used is None or baseline is None:
        return
    delta = used - baseline
    flag = "  <-- SUSPECTED LEAK (phase left GPU resident)" if delta > LEAK_THRESH_MIB else ""
    log(f"GPU {when}: {used} MiB used (baseline {baseline}, Δ{delta:+d} MiB){flag}",
        "WARN" if flag else "INFO")


# =========================================================================== #
# Ledger helpers (the fresh-generation invariant lives here)
# =========================================================================== #
def ledger_line_count(ledger: Path) -> int:
    if not ledger.exists():
        return 0
    with open(ledger, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def new_fresh_q3(ledger: Path, start_line: int) -> list[dict]:
    """The fresh q3 rows appended after `start_line`: v6-stamped, guard_pass,
    decoded_class == 3 — exactly build_fresh_discovery's admission filter."""
    if not ledger.exists():
        return []
    rows = []
    with open(ledger, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line or not line.strip():
                continue
            d = json.loads(line)
            if cc.is_v6_decoded(d) and d.get("guard_pass") and d.get("decoded_class") == 3:
                rows.append(d)
    return rows


def assert_fresh_isolation(log: Log, ledger: Path, start_line: int, batch_dir: Path):
    """The non-negotiable invariant (prompt §3): EVERY pooled row must originate from a
    ledger row THIS RUN wrote at or after the cycle watermark. Re-derived independently of
    the pool builder from each row's provenance.source_oid vs the ids the ledger holds in
    [start_line:]. HALTS the run on any violation."""
    fresh_ids = set()
    with open(ledger, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line or not line.strip():
                continue
            fresh_ids.add(json.loads(line).get("id"))
    images = batch_dir / "images.jsonl"
    n = 0
    for line in images.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        oid = (json.loads(line).get("provenance") or {}).get("source_oid")
        if oid not in fresh_ids:
            raise SystemExit(
                f"FRESH-ISOLATION VIOLATION: pooled row source_oid={oid!r} is NOT among "
                f"this cycle's fresh ledger rows [{start_line}:] of {ledger}. A non-fresh "
                f"(banked/historical/earlier-cycle) location reached the emit pool. Halting.")
    log(f"  fresh-isolation OK: all {n} pooled rows originate from this cycle's ledger "
        f"rows [{start_line}:{ledger_line_count(ledger)}] ({len(fresh_ids)} fresh ids)")


# =========================================================================== #
# Aggregate manifest (durable; appended per cycle after emit lands)
# =========================================================================== #
def append_manifest(run_manifest: Path, cycle: int, emit_dir: Path) -> int:
    """Append a cycle's emit rows (each a COMPLETE replayable recipe) to the durable
    run-level manifest, tagged with the cycle. Returns the row count appended. emit_v1's
    own manifest is already durable per-PNG; this aggregates cycles into one file."""
    cyc_manifest = emit_dir / "manifest.jsonl"
    if not cyc_manifest.exists():
        return 0
    rows = [json.loads(l) for l in cyc_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    with open(run_manifest, "a", encoding="utf-8") as f:
        for r in rows:
            r["cycle"] = cycle
            f.write(json.dumps(r) + "\n")
    return len(rows)


# =========================================================================== #
# Orchestrator
# =========================================================================== #
def load_state(state_path: Path) -> dict | None:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return None


def save_state(state_path: Path, state: dict):
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def orchestrate(args):
    out_root = Path(args.out_root).resolve()
    disc_root = Path(args.discovery_root).resolve()
    run_dir = out_root / args.run_id
    disc_dir = disc_root / args.run_id             # run-scoped discovery store (--discovery-dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    disc_dir.mkdir(parents=True, exist_ok=True)
    ledger = disc_dir / "outcome_ledger.jsonl"
    run_manifest = run_dir / "manifest.jsonl"
    state_path = run_dir / "state.json"
    log = Log(run_dir / "orchestrator.log")
    timing = Timing(run_dir / "timing.jsonl", args.run_id)   # durable per-phase wall-clock

    families = args.families
    per_family_s = args.per_family_min * 60.0
    cap_s = args.cap_hours * 3600.0

    # --- resume / idempotency: reuse the ORIGINAL deadline so total wall-clock <= cap
    # across restarts; never clobber prior manifest/ledger. ---
    state = load_state(state_path)
    if state is not None:
        deadline = state["deadline_epoch"]
        cycle = state["cycles_done"]
        log(f"RESUME run '{args.run_id}': {cycle} cycles done, "
            f"{(deadline - time.time())/3600:.2f}h remaining to original deadline")
        if not ledger.exists():
            log("  note: run ledger absent on resume (no discovery persisted yet) — continuing",
                "WARN")
    else:
        # Run-start freshness precondition: the run ledger MUST be empty/absent.
        n0 = ledger_line_count(ledger)
        if n0 != 0:
            raise SystemExit(
                f"run-start freshness precondition FAILED: {ledger} already has {n0} rows. "
                f"A fresh run requires an EMPTY run-scoped ledger. Use a new --run-id.")
        deadline = time.time() + cap_s
        cycle = 0
        log(f"START run '{args.run_id}'  cap={args.cap_hours}h  families={families}  "
            f"julia_hook=on  per_family={args.per_family_min}m  pool_count={args.pool_count}  "
            f"gate={args.gate}  eval-res")
        log(f"  run ledger (empty, run-scoped): {ledger}")
        log(f"  output tree: {run_dir}")
        save_state(state_path, {"run_id": args.run_id, "deadline_epoch": deadline,
                                "cycles_done": 0, "started": time.strftime('%Y-%m-%dT%H:%M:%S')})

    baseline_gpu = gpu_used_mib()
    log(f"GPU baseline: {baseline_gpu} MiB" + ("" if baseline_gpu is not None
                                               else " (nvidia-smi unavailable; GPU checks skipped)"))

    est_render_s = args.est_render_s
    empty_streak = 0
    totals = {"cycles": 0, "discovery_families": 0, "fresh_q3": 0, "emitted": 0,
              "phase_failures": 0}
    last_heartbeat = time.time()

    def remaining():
        return deadline - time.time()

    # ----------------------------- cycle loop ----------------------------- #
    while True:
        # A minimally-productive cycle needs one family of discovery + one emit render.
        min_cycle_s = per_family_s + est_render_s
        if remaining() < min_cycle_s:
            log(f"CAP: {remaining()/60:.1f}m left < min cycle {min_cycle_s/60:.1f}m — "
                f"stopping cleanly at a phase boundary.")
            break
        if empty_streak >= MAX_EMPTY_CYCLES:
            log(f"SATURATION: {empty_streak} consecutive zero-fresh-q3 cycles — stopping.")
            break

        cycle += 1
        watermark = ledger_line_count(ledger)
        log(f"===== CYCLE {cycle} =====  remaining {remaining()/3600:.2f}h  "
            f"ledger watermark {watermark}")

        # ---- PHASE 1: discovery, one family at a time (GPU-exclusive, cap-gated) ----
        fams_run = 0
        for fi, fam in enumerate(families):
            if remaining() < per_family_s:
                log(f"  CAP: {remaining()/60:.1f}m < family budget {args.per_family_min}m — "
                    f"skipping remaining discovery families this cycle.")
                break
            seed = args.seed + cycle * 1000 + fi * 137     # distinct exploration per cycle/family
            cmd = [PY, "-u", str(SEEDER), "--run", "--discovery-dir", str(disc_dir),
                   "--family", fam, "--julia-hook",
                   "--budget", str(args.per_family_min), "--seed", str(seed)]
            if args.disc_batch:
                cmd += ["--batch", str(args.disc_batch)]
            gpu_boundary(log, baseline_gpu, f"pre-discovery[{fam}]")
            fam_pre = ledger_line_count(ledger)   # cheap: attributes this family's yield
            try:
                res = run_phase(log, f"discovery:{fam}", cmd, per_family_s)
                timing.record(cycle, f"discovery:{fam}", res,
                              ledger_rows_added=ledger_line_count(ledger) - fam_pre,
                              fresh_q3=len(new_fresh_q3(ledger, fam_pre)))
                if not res["ok"]:
                    totals["phase_failures"] += 1
                    log(f"  discovery:{fam} FAILED (isolated) — continuing", "WARN")
                else:
                    fams_run += 1
            except Exception as e:
                totals["phase_failures"] += 1
                log(f"  discovery:{fam} EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
            gpu_boundary(log, baseline_gpu, f"post-discovery[{fam}]")
        totals["discovery_families"] += fams_run

        # ---- PHASE 1b: phoenix discovery (native z-descent -> the SAME run-scoped ledger) ----
        # Phoenix has no parameter plane, so it can't ride the c-plane radius-rejection loop;
        # its own --run-phoenix phase descends the fixed Ushiki z-plane and appends v6-scored
        # rows (t_good=0.18) to the run-scoped ledger. new_fresh_q3 + assert_fresh_isolation
        # then admit phoenix rows exactly like the other families (both scan the whole ledger
        # past the watermark, family-agnostic). Elevated budget; skipped when the cap is tight.
        phoenix_s = args.phoenix_min * 60.0
        if args.phoenix_min > 0 and remaining() >= phoenix_s:
            pseed = args.seed + cycle * 1000 + 900     # distinct from every c-plane family seed
            cmd = [PY, "-u", str(SEEDER), "--run-phoenix", "--discovery-dir", str(disc_dir),
                   "--budget", str(args.phoenix_min), "--seed", str(pseed)]
            if args.phoenix_walks:
                cmd += ["--phoenix-walks", str(args.phoenix_walks)]  # deterministic keeper floor (mini)
            if args.disc_batch:
                cmd += ["--batch", str(args.disc_batch)]             # walks/round (phoenix reuse)
            gpu_boundary(log, baseline_gpu, "pre-discovery[phoenix]")
            ph_pre = ledger_line_count(ledger)
            try:
                res = run_phase(log, "discovery:phoenix", cmd, phoenix_s)
                timing.record(cycle, "discovery:phoenix", res,
                              ledger_rows_added=ledger_line_count(ledger) - ph_pre,
                              fresh_q3=len(new_fresh_q3(ledger, ph_pre)))
                if not res["ok"]:
                    totals["phase_failures"] += 1
                    log("  discovery:phoenix FAILED (isolated) — continuing", "WARN")
            except Exception as e:
                totals["phase_failures"] += 1
                log(f"  discovery:phoenix EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
            gpu_boundary(log, baseline_gpu, "post-discovery[phoenix]")
        elif args.phoenix_min > 0:
            log(f"  CAP: {remaining()/60:.1f}m < phoenix budget {args.phoenix_min}m — "
                f"skipping phoenix discovery this cycle.")

        # ---- fresh-q3 count over ONLY this cycle's appended rows ----
        fresh = new_fresh_q3(ledger, watermark)
        log(f"  cycle {cycle} discovery: {fams_run}/{len(families)} families ran, "
            f"+{ledger_line_count(ledger) - watermark} ledger rows, {len(fresh)} fresh q3")
        totals["fresh_q3"] += len(fresh)
        if not fresh:
            empty_streak += 1
            log(f"  no fresh q3 this cycle (empty streak {empty_streak}); skipping pool+emit.")
            totals["cycles"] += 1
            save_state(state_path, {**load_state(state_path), "cycles_done": cycle})
            continue
        empty_streak = 0

        # ---- PHASE 2: pool build (ONLY this cycle's fresh q3s; run-scoped ledger) ----
        if remaining() < EST_POOL_LOC_S:
            log(f"  CAP: {remaining()/60:.1f}m left — insufficient for pool build; stopping.")
            break
        batch_dir = run_dir / "pools" / f"cycle_{cycle:03d}"
        est_pool_s = min(len(fresh), args.pool_count) * EST_POOL_LOC_S
        cmd = [PY, "-u", str(POOL_BUILDER), "--ledger", str(ledger),
               "--ledger-start-line", str(watermark), "--batch-dir", str(batch_dir),
               "--count", str(args.pool_count), "--seed", str(args.seed)]
        if args.pool_limit:
            cmd += ["--limit", str(args.pool_limit)]   # cap locations actually rendered (mini)
        gpu_boundary(log, baseline_gpu, "pre-pool")
        try:
            res = run_phase(log, f"pool:cycle{cycle}", cmd, est_pool_s)
        except Exception as e:
            res = {"ok": False}
            log(f"  pool build EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
        gpu_boundary(log, baseline_gpu, "post-pool")
        pool_locs = ledger_line_count(batch_dir / "images.jsonl")   # locations pooled this cycle
        timing.record(cycle, f"pool:cycle{cycle}", res,
                      fresh_q3_in=len(fresh), pool_locations=pool_locs)
        if not res["ok"] or not (batch_dir / "images.jsonl").exists():
            totals["phase_failures"] += 1
            log(f"  pool build failed / no batch (isolated) — skipping emit this cycle.", "WARN")
            totals["cycles"] += 1
            save_state(state_path, {**load_state(state_path), "cycles_done": cycle})
            continue

        # ---- the non-negotiable freshness assertion (halts on violation) ----
        assert_fresh_isolation(log, ledger, watermark, batch_dir)

        # ---- PHASE 3: emit (eval-res; smooth stem). --limit bounds renders to fit budget. ----
        emit_dir = run_dir / "emit" / f"cycle_{cycle:03d}"
        emit_limit = int(remaining() // est_render_s)          # per-render cap granularity
        if emit_limit <= 0:
            log(f"  CAP: {remaining()/60:.1f}m left — no budget for even one render; stopping.")
            break
        est_emit_s = min(emit_limit, args.pool_count) * est_render_s
        cmd = [PY, "-u", str(EMITTER), "--pool", str(batch_dir), "--eval-res",
               "--gate", str(args.gate), "--out-dir", str(emit_dir), "--limit", str(emit_limit)]
        gpu_boundary(log, baseline_gpu, "pre-emit")
        t_emit = time.time()
        try:
            res = run_phase(log, f"emit:cycle{cycle}", cmd, est_emit_s)
        except Exception as e:
            res = {"ok": False}
            log(f"  emit EXCEPTION (isolated): {type(e).__name__}: {e}", "WARN")
        gpu_boundary(log, baseline_gpu, "post-emit")

        n_emitted = append_manifest(run_manifest, cycle, emit_dir)
        totals["emitted"] += n_emitted
        timing.record(cycle, f"emit:cycle{cycle}", res,
                      pool_locations=pool_locs, emitted=n_emitted, emit_limit=emit_limit)
        if not res["ok"]:
            totals["phase_failures"] += 1
            log(f"  emit reported failure (isolated); {n_emitted} rows still persisted.", "WARN")
        # refine the per-render estimate from observed emit throughput (EMA).
        if n_emitted > 0:
            obs = (time.time() - t_emit) / n_emitted
            est_render_s = 0.5 * est_render_s + 0.5 * obs
        log(f"  cycle {cycle} emit: {n_emitted} wallpapers -> {emit_dir}  "
            f"(aggregate manifest {ledger_line_count(run_manifest)} rows; "
            f"est_render now {est_render_s:.0f}s)")

        totals["cycles"] += 1
        save_state(state_path, {**load_state(state_path), "cycles_done": cycle,
                                "totals": totals})

        if time.time() - last_heartbeat > HEARTBEAT_EVERY_S:
            last_heartbeat = time.time()
        log(f"HEARTBEAT cycle={cycle} remaining={remaining()/3600:.2f}h emitted_total="
            f"{totals['emitted']} fresh_q3_total={totals['fresh_q3']} "
            f"phase_failures={totals['phase_failures']}")

    # ----------------------------- final report ----------------------------- #
    gpu_boundary(log, baseline_gpu, "final")
    summary = {
        "run_id": args.run_id, "finished": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "cap_hours": args.cap_hours, "families": families, "julia_hook": True,
        "cycles": totals["cycles"], "discovery_families_run": totals["discovery_families"],
        "fresh_q3_total": totals["fresh_q3"], "wallpapers_emitted": totals["emitted"],
        "phase_failures": totals["phase_failures"],
        "aggregate_manifest": str(run_manifest.relative_to(ROOT)),
        "aggregate_manifest_rows": ledger_line_count(run_manifest),
        "timing_jsonl": str((run_dir / "timing.jsonl").relative_to(ROOT)),
        "timing_rows": ledger_line_count(run_dir / "timing.jsonl"),
        "run_ledger": str(ledger),
        "run_ledger_rows": ledger_line_count(ledger),
        "output_tree": str(run_dir),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("=" * 70)
    log(f"RUN COMPLETE '{args.run_id}': {totals['cycles']} cycles, "
        f"{totals['emitted']} wallpapers, {totals['fresh_q3']} fresh q3, "
        f"{totals['phase_failures']} phase failures")
    log(f"  aggregate manifest ({summary['aggregate_manifest_rows']} recipes) -> {run_manifest}")
    log(f"  wallpapers -> {run_dir / 'emit'}/cycle_*/wallpapers/")
    log(f"  per-phase timing -> {run_dir / 'timing.jsonl'} ({ledger_line_count(run_dir / 'timing.jsonl')} rows)")
    log(f"  run summary -> {run_dir / 'run_summary.json'}")
    log.close()
    timing.close()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="production run (6h cap by default)")
    ap.add_argument("--mini", action="store_true",
                    help="short throwaway validation run: tight cap, 1 family, scratch roots")
    ap.add_argument("--run-id", default=None, help="run id (default: timestamp)")
    ap.add_argument("--cap-hours", type=float, default=CAP_HOURS, help="wall-clock cap (hours)")
    ap.add_argument("--per-family-min", type=float, default=PER_FAMILY_MIN,
                    help="discovery budget per family per cycle (minutes)")
    ap.add_argument("--phoenix-min", type=float, default=PHOENIX_PER_CYCLE_MIN,
                    help="per-cycle phoenix z-descent budget (minutes); 0 disables the phoenix phase")
    ap.add_argument("--phoenix-walks", type=int, default=0,
                    help="cap total phoenix walks per cycle (production_seeder --phoenix-walks; "
                         "0 = budget-only). The mini sets this to guarantee >=1 phoenix keeper.")
    ap.add_argument("--pool-count", type=int, default=POOL_COUNT,
                    help="build_fresh_discovery --count (family-balanced pool cap per cycle)")
    ap.add_argument("--gate", type=float, default=GATE, help="emit_v1 v3 quality-floor gate")
    ap.add_argument("--est-render-s", type=float, default=EST_EMIT_RENDER_S,
                    help="initial per-winner eval-res render estimate (refined at runtime)")
    ap.add_argument("--families", nargs="+", default=None,
                    help="override the family set (default: all c-plane families)")
    ap.add_argument("--out-root", default=str(ROOT / "out" / "wallpaper" / "overnight"),
                    help="output tree root (pools/emit/manifest/log; disposable, survives crash)")
    ap.add_argument("--discovery-root", default=str(ROOT / "data" / "discovery" / "fresh_runs"),
                    help="durable run-scoped discovery store root")
    ap.add_argument("--seed", type=int, default=0, help="base seed (varied per cycle/family)")
    ap.add_argument("--disc-batch", type=int, default=0,
                    help="production_seeder --batch (seeds/batch; 0 = seeder default 24). Smaller "
                         "= finer budget granularity (used to keep the mini-run short).")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="cap locations actually rendered per pool build (build_fresh_discovery "
                         "--limit; 0 = pool up to --count). Bounds mini-run wall-clock.")
    args = ap.parse_args()

    if not (args.run or args.mini):
        ap.print_help()
        return

    if args.run_id is None:
        args.run_id = ("mini_" if args.mini else "overnight_") + time.strftime("%Y%m%d_%H%M%S")
    if args.families is None:
        args.families = FAMILIES

    if args.mini:
        # Tight, self-contained validation: 1 family (mandelbrot + its julia twin via the
        # hook), short discovery, small pool, throwaway scratch roots that clean up.
        scratch = ROOT / "out" / "wallpaper" / "overnight_mini_scratch"
        args.out_root = str(scratch / "out")
        args.discovery_root = str(scratch / "disc")
        if args.cap_hours == CAP_HOURS:
            args.cap_hours = 0.75                      # ~45 min (room for >=2 full cycles now that
                                                       # the phoenix phase adds ~4 min/cycle)
        if args.per_family_min == PER_FAMILY_MIN:
            args.per_family_min = 2.0
        if args.pool_count == POOL_COUNT:
            args.pool_count = 12
        if args.disc_batch == 0:
            args.disc_batch = 6                        # small batch -> budget honored fast
        if args.pool_limit == 0:
            args.pool_limit = 4                        # render <=4 locations/cycle (room for a
                                                       # phoenix keeper alongside mandelbrot ones)
        if args.families == FAMILIES:
            args.families = ["mandelbrot"]
        # Phoenix mechanism test: a walk cap (not budget) is the deterministic keeper floor.
        # 8/cycle over >=2 cycles -> ~16+ descents; at the study's ~0.14 yield P(>=1 keeper)
        # ~0.9 (the guarded smoke ran hotter). Generous budget so the WALK CAP binds first.
        if args.phoenix_min == PHOENIX_PER_CYCLE_MIN:
            args.phoenix_min = 15.0
        if args.phoenix_walks == 0:
            args.phoenix_walks = 8
        # Low emit gate so a phoenix keeper's colored crop actually EMITS: the wallpaper head's
        # 0.90 crop-gate would otherwise reject samey phoenix crops and no phoenix recipe would
        # reach the manifest — this is a plumbing test, not a quality gate. Production keeps 0.90.
        if args.gate == GATE:
            args.gate = 0.05
        print(f"[mini] throwaway roots under {scratch}")
    elif args.disc_batch == 0:
        # Production: bound the per-family discovery batch to ~half the seeder default (24)
        # so a family's one-batch budget floor stays ~6-8 min and a full 4-family discovery
        # phase is ~30-45 min — a healthy cycle length (a crash then loses <= one cycle).
        args.disc_batch = 12

    orchestrate(args)


if __name__ == "__main__":
    main()
