"""Confirm the Python `coloring.bake_lut` port is byte-matched to the Rust bake.

Drives the ignored Rust dump test `tests/palette_bytematch.rs` (which writes a
LUT_SIZE*3 little-endian f64 linear-RGB LUT), bakes the same stops with the same
mirror flag in Python, and reports max abs difference. The verbatim-port invariant
the handoff asserts; rerun after touching either bake. Tests both a CYCLIC map
(mirror off, must stay matched) and a SEQUENTIAL map (mirror on, the new path).

Usage (from repo root):  python palette_extractor/check_bytematch.py
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from palette_lib.coloring import bake_lut, LUT_SIZE

CMAPS = json.loads((ROOT / "data" / "palettes" / "clean_colormaps.json").read_text())
BY_NAME = {e["name"]: e for e in CMAPS}


def rust_lut(stops, mirror: bool) -> np.ndarray:
    spec = ";".join(f"{p},{c[0]},{c[1]},{c[2]}" for p, c in stops)
    out = Path(tempfile.gettempdir()) / "bytematch_lut.bin"
    if out.exists():
        out.unlink()
    env_args = [f"DUMP_STOPS={spec}", f"DUMP_MIRROR={'1' if mirror else '0'}",
                f"DUMP_OUT={out}"]
    # cargo test, ignored test, single-threaded; env passed inline.
    cmd = ["cargo", "test", "--release", "--test", "palette_bytematch",
           "dump_lut", "--", "--ignored", "--exact"]
    env = {**__import__("os").environ}
    for kv in env_args:
        k, v = kv.split("=", 1)
        env[k] = v
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(r.stdout, r.stderr)
        raise RuntimeError("rust dump failed")
    return np.fromfile(out, dtype="<f8").reshape(LUT_SIZE, 3)


def check(name: str, mirror: bool):
    stops = [(p, tuple(c)) for p, c in BY_NAME[name]["stops"]]
    py = bake_lut(stops, mirror=mirror)
    ru = rust_lut(stops, mirror)
    d = np.abs(py - ru).max()
    cls = BY_NAME[name]["cycle"]
    print(f"  {name:18s} cycle={cls:11s} mirror={mirror!s:5s}  max|d| = {d:.3e}  "
          f"{'BYTE-MATCH' if d < 1e-12 else 'MISMATCH' if d > 1e-9 else 'near'}")
    return d


def cases() -> list[tuple[str, bool]]:
    """The (map-name, mirror) cases the byte-match covers. Shared by the standalone
    CLI and the pytest gate (`test_bytematch.py`) so both check the identical set."""
    # cyclic map: mirror OFF must remain matched (and unchanged by this prompt)
    cyc = next(e["name"] for e in CMAPS if e["cycle"] == "cyclic")
    seq = next(e["name"] for e in CMAPS if e["mirror_needed"])
    cs = [
        (cyc, False),        # cyclic, no mirror — the settled path
        (seq, False),        # sequential, no mirror — baseline match
        (seq, True),         # sequential, mirrored — the selective pre-mirror path
        ("magma", True),
    ]
    if "viridis" in BY_NAME:
        cs.append(("viridis", True))
    return cs


def run() -> float:
    """Run every case; return the worst max|Δ| (Rust bake ↔ Python bake). Prints a
    per-case line as a side effect. `< 1e-12` is the byte-match acceptance bar."""
    return max(check(name, mirror) for name, mirror in cases())


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("Rust<->Python bake byte-match (linear-RGB LUT, LUT_SIZE=%d):" % LUT_SIZE)
    worst = run()
    print(f"\nworst max|d| across cases: {worst:.3e}  "
          f"({'PASS' if worst < 1e-12 else 'CHECK'})")


if __name__ == "__main__":
    main()
