#!/usr/bin/env python
"""kill_run.py — kill a background run by commandline substring, without self-matching.

The footgun this ends: a naive `Get-Process | where CommandLine -like '*build_emission*'`
(or `pkill -f build_emission`) also matches the *querying* shell — on Windows the `pwsh`
that runs the filter has the pattern in its own command line, so the cleanup kills itself
(and the harness's watcher) instead of the run. This helper enumerates processes, keeps only
interpreter processes (`python`/`uv`, and `pwsh`/`powershell` only when asked) whose command
line contains the pattern, and EXCLUDES this process and its whole ancestor chain before
touching anything. Dry-run by default; `--apply` actually kills.

  uv run python tools/kill_run.py build_emission_diversity_v1          # list matches
  uv run python tools/kill_run.py build_emission_diversity_v1 --apply  # kill them

`select_targets` is a pure function (process dicts in, ids out) so the exclusion/whitelist
logic is unit-tested (test_kill_run.py) without spawning or killing anything real.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Interpreter process names we are ever willing to kill. Bare `pwsh`/`powershell` are
# EXCLUDED by default (a shell that merely printed the pattern is not the run) — opt in
# with --include-shells only when the run itself is a PowerShell script.
INTERP_NAMES = {"python", "python.exe", "python3", "uv", "uv.exe"}
SHELL_NAMES = {"pwsh", "pwsh.exe", "powershell", "powershell.exe"}


def _base_name(name: str) -> str:
    return os.path.basename(str(name or "")).lower()


def select_targets(procs, pattern, exclude_pids, include_shells=False):
    """Pure selection core.

    procs: iterable of {"pid": int, "name": str, "cmdline": str}.
    pattern: case-sensitive substring that must appear in cmdline.
    exclude_pids: pids never returned (this process + its ancestors).
    include_shells: also match pwsh/powershell (default: interpreters only).

    Returns the sorted list of matching pids.
    """
    allowed = set(INTERP_NAMES) | (set(SHELL_NAMES) if include_shells else set())
    exclude = {int(p) for p in exclude_pids}
    hits = []
    for p in procs:
        pid = int(p["pid"])
        if pid in exclude:
            continue
        if _base_name(p.get("name")) not in allowed:
            continue
        if pattern not in (p.get("cmdline") or ""):
            continue
        hits.append(pid)
    return sorted(hits)


# --------------------------------------------------------------------------- #
# Platform process enumeration + ancestor walk.
# --------------------------------------------------------------------------- #
def _enumerate_windows():
    """[(pid, ppid, name, cmdline)] via CIM. One JSON blob, no per-process spawn."""
    ps = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress -Depth 2"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         capture_output=True, text=True)
    import json
    data = json.loads(out.stdout or "[]")
    if isinstance(data, dict):
        data = [data]
    procs = []
    for d in data:
        procs.append({"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"),
                      "name": d.get("Name"), "cmdline": d.get("CommandLine") or ""})
    return procs


def _enumerate_posix():
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="],
                         capture_output=True, text=True)
    procs = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        pid, ppid, comm = parts[0], parts[1], parts[2]
        args = parts[3] if len(parts) == 4 else ""
        procs.append({"pid": int(pid), "ppid": int(ppid), "name": comm, "cmdline": args})
    return procs


def enumerate_procs():
    return _enumerate_windows() if os.name == "nt" else _enumerate_posix()


def ancestor_pids(procs, start_pid):
    """`start_pid` plus every parent up the tree (so we never kill our own shell chain)."""
    by_pid = {int(p["pid"]): p for p in procs if p.get("pid") is not None}
    out, cur, guard = set(), int(start_pid), 0
    while cur in by_pid and guard < 128:
        out.add(cur)
        nxt = by_pid[cur].get("ppid")
        if nxt is None:
            break
        cur = int(nxt)
        guard += 1
    out.add(int(start_pid))
    return out


def _kill(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                           capture_output=True, text=True)
        else:
            os.kill(pid, 9)
        return True
    except Exception as e:                                   # noqa: BLE001
        print(f"  [kill] pid {pid} failed: {e!r}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="commandline substring identifying the run")
    ap.add_argument("--apply", action="store_true", help="actually kill (default: dry-run list)")
    ap.add_argument("--include-shells", action="store_true",
                    help="also match pwsh/powershell (default: python/uv only)")
    args = ap.parse_args()

    procs = enumerate_procs()
    exclude = ancestor_pids(procs, os.getpid())
    targets = select_targets(procs, args.pattern, exclude, include_shells=args.include_shells)
    by_pid = {int(p["pid"]): p for p in procs if p.get("pid") is not None}

    if not targets:
        print(f"[kill_run] no python/uv{'/shell' if args.include_shells else ''} process "
              f"matches {args.pattern!r} (excluded self+ancestors {sorted(exclude)})", flush=True)
        return
    print(f"[kill_run] {len(targets)} match(es) for {args.pattern!r} "
          f"(self+ancestors excluded):", flush=True)
    for pid in targets:
        cl = (by_pid.get(pid, {}).get("cmdline") or "")[:120]
        print(f"  pid {pid}  {_base_name(by_pid.get(pid, {}).get('name'))}  {cl}", flush=True)
    if not args.apply:
        print("[kill_run] dry-run — pass --apply to kill.", flush=True)
        return
    killed = sum(_kill(pid) for pid in targets)
    print(f"[kill_run] killed {killed}/{len(targets)}.", flush=True)


if __name__ == "__main__":
    main()
