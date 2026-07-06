#!/usr/bin/env python
"""Per-family instrumentation report for a monitored v6 harvest pass.

Reads the durable outcome ledger, restricts to the run timestamps in a harvest manifest
(written by monitored_harvest.py), and reports per PARTITION (family+degree):

  * p_good and p_notbad histograms of every scored outcome, with the 0.24 (deg-2) and
    0.50 (baseline) t_good lines marked. This is the first-class output: it resolves
    whether high-degree frames PILE INTO [0.24, 0.50] (→ holding them at 0.50 was
    load-bearing) or sit well below it (→ barren, as expected).
  * q3 yield: decoded_class == 3 count + rate (validates the deg-2 0.24 lift in prod).
  * guard stats: pass vs sentinel-dropped (production --run applies the guard model-free
    inside scoring; a walk whose whole top-3 is degenerate collapses to guard_pass=False).

Usage:
  uv run python tools/v6/harvest_report.py [manifest.json]
      default manifest: out/atlas/v6_monitored_harvest/manifest.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "discovery" / "outcome_ledger.jsonl"
DEFAULT_MANIFEST = ROOT / "out" / "atlas" / "v6_monitored_harvest" / "manifest.json"

T_DEG2, T_BASE = 0.24, 0.50
BINS = [i / 20 for i in range(21)]   # 0.00, 0.05, ... 1.00  (width 0.05)


def hist_lines(vals, marks):
    """20-bin [0,1] text histogram (bin width 0.05); `marks` = {value: label} annotate the
    bin each threshold falls in with an inline `<-- label` arrow."""
    counts = [0] * 20
    for v in vals:
        b = min(19, max(0, int(v * 20)))
        counts[b] += 1
    mx = max(counts) or 1
    mark_bin = {min(19, int(val * 20)): lab for val, lab in marks.items()}
    out = []
    for i in range(20):
        lo = i / 20
        bar = "#" * int(round(40 * counts[i] / mx))
        tag = f"   <-- {mark_bin[i]} line" if i in mark_bin else ""
        out.append(f"    {lo:4.2f} |{bar:<40}| {counts[i]}{tag}")
    return out


def report(rows_by_part: dict[str, list[dict]], guard_tel: dict[str, dict]):
    order = ["mandelbrot", "multibrot3", "multibrot4", "multibrot5",
             "julia:mandelbrot", "julia:multibrot3", "julia:multibrot4", "julia:multibrot5"]
    seen = [p for p in order if p in rows_by_part] + \
           [p for p in rows_by_part if p not in order]
    for part in seen:
        rows = rows_by_part[part]
        n = len(rows)
        t_good = T_DEG2 if part in ("mandelbrot", "julia:mandelbrot") else T_BASE
        dec = Counter(r.get("decoded_class") for r in rows)
        q3 = dec.get(3, 0)
        gp = sum(1 for r in rows if r.get("guard_pass", True))
        gd = n - gp
        # in-band pile-up: guard-passing frames with p_good in [0.24, 0.50)
        pg = [r["p_good"] for r in rows if r.get("p_good") is not None]
        pnb = [r["p_notbad"] for r in rows if r.get("p_notbad") is not None]
        inband = sum(1 for v in pg if T_DEG2 <= v < T_BASE)
        below = sum(1 for v in pg if v < T_DEG2)
        above = sum(1 for v in pg if v >= T_BASE)
        print("=" * 78)
        print(f"PARTITION: {part}   (t_good={t_good})   scored={n}")
        print("=" * 78)
        print(f"  decoded_class: 1={dec.get(1,0)} 2={dec.get(2,0)} 3={dec.get(3,0)} "
              f"guarded(None)={dec.get(None,0)}")
        print(f"  q3 YIELD: {q3}/{n} decoded-3  ({q3/n*100:.1f}%)" if n else "  q3 YIELD: n/a")
        print(f"  guard: pass={gp} dropped(sentinel)={gd}"
              + (f"   [run-level clean={guard_tel[part].get('clean_harvest')} "
                 f"salvaged={guard_tel[part].get('salvaged_harvest')} "
                 f"dropped={guard_tel[part].get('dropped')} "
                 f"frames_gated={guard_tel[part].get('frames_gated_total')}]"
                 if part in guard_tel else ""))
        print(f"  p_good split vs t_good lines: below 0.24={below}  "
              f"IN-BAND [0.24,0.50)={inband}  >=0.50={above}")
        if pg:
            print("  p_good histogram:")
            for l in hist_lines(pg, {T_DEG2: "0.24", T_BASE: "0.50"}):
                print(l)
        if pnb:
            print("  p_notbad histogram:")
            for l in hist_lines(pnb, {0.50: "0.50"}):
                print(l)
        print()


def main() -> int:
    mpath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    ts_set = {e["run_ts"] for e in manifest["runs"] if e.get("run_ts")}
    print(f"harvest manifest: {mpath}")
    print(f"run timestamps ({len(ts_set)}): {sorted(ts_set)}\n")

    # gather run-level guard telemetry per c-plane family from each run's summary.
    guard_tel: dict[str, dict] = {}
    for e in manifest["runs"]:
        sp = e.get("summary")
        if sp and Path(sp).exists():
            s = json.loads(Path(sp).read_text(encoding="utf-8"))
            guard_tel[s["config"]["family"]] = s.get("guard_telemetry", {})

    rows_by_part: dict[str, list[dict]] = defaultdict(list)
    with open(LEDGER, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ts") in ts_set:
                rows_by_part[r.get("family", "mandelbrot")].append(r)

    total = sum(len(v) for v in rows_by_part.values())
    print(f"scored outcomes in window: {total} across {len(rows_by_part)} partitions\n")
    report(rows_by_part, guard_tel)

    # decisions this pass gates
    print("=" * 78)
    print("DECISIONS")
    print("=" * 78)
    deg2 = [p for p in rows_by_part if p in ("mandelbrot", "julia:mandelbrot")]
    hi = [p for p in rows_by_part if p not in ("mandelbrot", "julia:mandelbrot")]
    for part in deg2:
        rows = rows_by_part[part]
        q3 = sum(1 for r in rows if r.get("decoded_class") == 3)
        would_q3_at_50 = sum(1 for r in rows
                             if r.get("p_notbad", 0) >= 0.5 and r.get("p_good", 0) >= 0.5)
        print(f"  (1) deg-2 {part}: q3@0.24={q3}  vs would-be q3@0.50={would_q3_at_50}  "
              f"(+{q3 - would_q3_at_50} from lowering)")
    for part in hi:
        rows = rows_by_part[part]
        pg = [r["p_good"] for r in rows if r.get("p_good") is not None]
        inband = sum(1 for v in pg if T_DEG2 <= v < T_BASE)
        below = sum(1 for v in pg if v < T_DEG2)
        n = len(pg) or 1
        verdict = ("PILES IN-BAND -> holding at 0.50 is load-bearing"
                   if inband / n > 0.15 else "sits below -> barren, as expected")
        print(f"  (2) high-deg {part}: in-band[0.24,0.50)={inband}/{len(pg)} "
              f"below={below}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
