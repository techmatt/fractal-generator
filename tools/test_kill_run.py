"""Tests for kill_run.select_targets / ancestor_pids — the self-match guard, purely.

No process is spawned or killed: the selection + ancestor-walk logic is exercised against
synthetic process dicts. Run: uv run pytest tools/test_kill_run.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import kill_run as K   # noqa: E402


PROCS = [
    {"pid": 10, "ppid": 1, "name": "python.exe", "cmdline": "python tools/emission/build_x.py --ledger a"},
    {"pid": 11, "ppid": 1, "name": "uv.exe", "cmdline": "uv run python tools/emission/build_x.py"},
    {"pid": 12, "ppid": 99, "name": "pwsh.exe", "cmdline": "pwsh -c Get-CimInstance ... build_x.py ..."},
    {"pid": 13, "ppid": 1, "name": "chrome.exe", "cmdline": "chrome build_x.py"},   # not an interpreter
    {"pid": 14, "ppid": 1, "name": "python.exe", "cmdline": "python other_run.py"},  # wrong pattern
]


def test_matches_interpreters_only_and_excludes_shell_by_default():
    hits = K.select_targets(PROCS, "build_x.py", exclude_pids=set())
    assert hits == [10, 11]                       # pwsh (12) and chrome (13) excluded; 14 wrong pattern


def test_include_shells_opt_in():
    hits = K.select_targets(PROCS, "build_x.py", exclude_pids=set(), include_shells=True)
    assert hits == [10, 11, 12]                   # pwsh now eligible


def test_self_and_ancestors_never_returned():
    # the querying pwsh (12) and its whole chain must be excludable even with --include-shells.
    procs = PROCS + [{"pid": 99, "ppid": 1, "name": "pwsh.exe", "cmdline": "pwsh login shell"}]
    excl = K.ancestor_pids(procs, start_pid=12)
    assert 12 in excl and 99 in excl              # self + parent
    hits = K.select_targets(procs, "build_x.py", exclude_pids=excl, include_shells=True)
    assert hits == [10, 11]                        # the self shell is gone


def test_ancestor_walk_terminates_on_cycle():
    cyc = [{"pid": 1, "ppid": 2, "name": "a", "cmdline": ""},
           {"pid": 2, "ppid": 1, "name": "b", "cmdline": ""}]
    got = K.ancestor_pids(cyc, start_pid=1)
    assert got == {1, 2}                            # no infinite loop
