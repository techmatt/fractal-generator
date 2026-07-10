#!/usr/bin/env python
"""Read-only scan: v5-decoded vs v6-stamped rows across the discovery ledgers.

Every discovery outcome ledger persists a `decoded_class` (the q3 hard-class
verdict of the k3-winning frame). production_seeder began stamping
`scorer_version="v6"` only partway through the gather runs, so a large body of
older rows carry a v5-vintage `decoded_class` and NO stamp. Those v5 verdicts must
never be consumed where a v6 readout is required (see `corpus_common.is_v6_decoded`
+ the fresh-discovery / dramatic wallpaper-head selectors).

This tallies, PER PARTITION (main ledger + each `gather/<class>`), how many rows
are v6-stamped vs v5/unstamped, broken out by decoded_class and by the guard-pass
deg-2 q3 slice (the honest "fresh machine-q3" pool the wallpaper head can draw on).
Purely diagnostic — reads the ledgers, mutates nothing.

    uv run python tools/atlas/check_ledger_decode_version.py
    uv run python tools/atlas/check_ledger_decode_version.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
import corpus_common as cc  # noqa: E402  (is_v6_decoded — canonical v6-stamp discriminator)

DISCOVERY = ROOT / "data" / "discovery"
DEG2_FAMILIES = ("mandelbrot", "julia:mandelbrot")


def _ledger_paths():
    """The main standing ledger + every gather/<class> partition, in a stable order."""
    paths = []
    main = DISCOVERY / "outcome_ledger.jsonl"
    if main.exists():
        paths.append(("main", main))
    gather = DISCOVERY / "gather"
    if gather.is_dir():
        for d in sorted(gather.iterdir()):
            led = d / "outcome_ledger.jsonl"
            if led.exists():
                paths.append((f"gather/{d.name}", led))
    return paths


def scan_ledger(path: Path) -> dict:
    """Tally one ledger. Counts are split v6-stamped vs v5/unstamped, and within
    each, the guard-pass deg-2 decoded_class==3 slice (the fresh-machine-q3 pool)."""
    stat = {
        "rows": 0,
        "v6": 0, "v5": 0,
        "v6_dc": Counter(), "v5_dc": Counter(),          # decoded_class distribution
        "v6_q3_guard_deg2": 0, "v5_q3_guard_deg2": 0,     # fresh-machine-q3-eligible slice
        "no_decoded_class": 0,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        stat["rows"] += 1
        v6 = cc.is_v6_decoded(r)
        dc = r.get("decoded_class")
        if "decoded_class" not in r:
            stat["no_decoded_class"] += 1
        q3_guard_deg2 = (dc == 3 and bool(r.get("guard_pass"))
                         and r.get("family") in DEG2_FAMILIES)
        if v6:
            stat["v6"] += 1
            stat["v6_dc"][dc] += 1
            stat["v6_q3_guard_deg2"] += int(q3_guard_deg2)
        else:
            stat["v5"] += 1
            stat["v5_dc"][dc] += 1
            stat["v5_q3_guard_deg2"] += int(q3_guard_deg2)
    return stat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    paths = _ledger_paths()
    report = {}
    for name, path in paths:
        report[name] = scan_ledger(path)

    if args.json:
        # Counter -> plain dict (keys stringified for JSON)
        out = {}
        for name, s in report.items():
            out[name] = {
                **{k: v for k, v in s.items() if not isinstance(v, Counter)},
                "v6_dc": {str(k): v for k, v in s["v6_dc"].items()},
                "v5_dc": {str(k): v for k, v in s["v5_dc"].items()},
            }
        print(json.dumps(out, indent=2))
        return

    tot = Counter()
    print(f"{'partition':<22} {'rows':>6} {'v6':>6} {'v5':>6}   "
          f"{'v6 q3&guard&deg2':>16} {'v5 q3&guard&deg2':>16}")
    print("-" * 88)
    for name, s in report.items():
        tot["rows"] += s["rows"]; tot["v6"] += s["v6"]; tot["v5"] += s["v5"]
        tot["v6_q3"] += s["v6_q3_guard_deg2"]; tot["v5_q3"] += s["v5_q3_guard_deg2"]
        flag = "  <-- v5-only partition" if s["v6"] == 0 and s["v5"] else ""
        print(f"{name:<22} {s['rows']:>6} {s['v6']:>6} {s['v5']:>6}   "
              f"{s['v6_q3_guard_deg2']:>16} {s['v5_q3_guard_deg2']:>16}{flag}")
    print("-" * 88)
    print(f"{'TOTAL':<22} {tot['rows']:>6} {tot['v6']:>6} {tot['v5']:>6}   "
          f"{tot['v6_q3']:>16} {tot['v5_q3']:>16}")
    print()
    print(f"Honest deg-2 fresh-machine-q3 pool (guard-pass, decoded_class==3, v6-stamped): "
          f"{tot['v6_q3']}")
    print(f"v5-decoded rows that would leak into that pool if unguarded:                    "
          f"{tot['v5_q3']}")


if __name__ == "__main__":
    main()
