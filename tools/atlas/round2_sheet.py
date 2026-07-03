#!/usr/bin/env python
"""Atlas round-2 — good-outcome contact sheets (the eyeball check behind the diversity
metric). Per arm: a grid of the GOOD outcomes' best reframed frames (the embed tiles),
sorted by k3. This is what the novel-region / coverage numbers measure — visual
repetition would show a yield win that is one location mined over and over.

Round-2 twin of round1_sheet (same layout), pointed at data/atlas/round2/. Run AFTER
round2_embed. Reuses round1_sheet.build_arm_sheet by overriding its module paths.

  uv run python tools/atlas/round2_sheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import round1_sheet as r1s  # noqa: E402

# Repoint round1_sheet's module-level paths at round 2.
r1s.D = ROOT / "data" / "atlas" / "round2"
r1s.TILES = ROOT / "out" / "atlas" / "round2" / "embed_tiles"
r1s.OUT = ROOT / "out" / "atlas" / "round2"
r1s.ARMS = [("arm1", "current seeder"), ("arm2", "atlas exploit"), ("arm3", "atlas explore")]


def main():
    for a, name in r1s.ARMS:
        if not (r1s.D / f"{a}_embed.npz").exists():
            print(f"skip {a}: no embed npz")
            continue
        out, n = r1s.build_arm_sheet(a, name)
        print(f"{a}: {n} good -> {out}")


if __name__ == "__main__":
    main()
