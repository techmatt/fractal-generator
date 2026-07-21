"""Phoenix root plumbing — descend a proposed seed batch as phoenix roots (Phase A item 4).

The frontier/dive kernel is `guided-descend`; it now accepts the full phoenix parameter
point via `--phoenix --c <re> <im> --p <re> <im> --phoenix-z1 <re> <im>`. This module takes
a batch of proposals from `phoenix_sampler` (each a distinct `(c, p, z_{-1})` point),
constructs the STANDARD phoenix start view (base-scale z-plane at center 0, like
`run_phoenix_descent`), descends each seed, and stamps every outcome row with the seed's
parameter-point identity through the shared ledger machinery — so distinct seeds never
dup-collide (`production_seeder.build_cloud` keys phoenix on `(c, p, z_{-1})`).

Phase A scope: the plumbing + a tiny end-to-end smoke (test_phoenix_roots). No surrogate,
no fertility memory, no measure changes — the seed grid is the next prompt. This is NOT the
production overnight loop; it is the minimal driver that proves seed -> descend -> identity-
stamped ledger row end to end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "atlas"))
sys.path.insert(0, str(HERE))

import production_seeder as ps  # noqa: E402  (phoenix_ident_fields / build_cloud / row_ident)
import phoenix_sampler as psamp  # noqa: E402


def default_binary() -> Path:
    return ROOT / "target" / "release" / ("fractal-generator.exe" if os.name == "nt"
                                          else "fractal-generator")


def descend_seed(seed: psamp.Seed, *, binary: Path, out_dir: Path, n_walks: int = 3,
                 depth_min: int = 1, depth_max: int = 3, rng_seed: int = 0,
                 node_width: int = 96, root_fw: float = 3.0, maxiter: int = 400) -> Path:
    """Run one phoenix z-plane descent at the seed's `(c, p, z_{-1})`, base-scale root at
    center 0. Writes `pool.jsonl` under `out_dir/pool` and returns that pool dir. Mirrors
    `production_seeder.run_phoenix_descent`, generalized to the full parameter point."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = out_dir / "pool"
    cmd = [
        str(binary), "guided-descend",
        "--phoenix",
        "--c", repr(seed.c.real), repr(seed.c.imag),
        "--p", repr(seed.p.real), repr(seed.p.imag),
        "--phoenix-z1", repr(seed.z_m1.real), repr(seed.z_m1.imag),
        "--n-walks", str(n_walks), "--seed", str(rng_seed), "--per-walk-rng",
        "--depth-min", str(depth_min), "--depth-max", str(depth_max),
        "--node-width", str(node_width), "--julia-root-fw", str(root_fw),
        "--maxiter", str(maxiter),
        "--preview-width", "48", "--cols", "40",
        "--out-dir", str(pool),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"phoenix descent failed (c={seed.c}, p={seed.p}, "
                           f"z_m1={seed.z_m1}):\n{r.stderr[-1500:]}")
    return pool


def walk_outcomes(pool: Path) -> list[dict]:
    """The deepest frame of each walk in `pool/pool.jsonl` — the walk's outcome location
    `{cx, cy, fw, depth, walk}`. Empty if the pool produced no frames."""
    p = pool / "pool.jsonl"
    if not p.exists():
        return []
    by_walk: dict[int, dict] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            w = int(r["walk"])
            if w not in by_walk or int(r["depth"]) > int(by_walk[w]["depth"]):
                by_walk[w] = r
    return [by_walk[w] for w in sorted(by_walk)]


def seed_outcome_rows(seed: psamp.Seed, pool: Path, *, run_ts: str, seq0: int) -> list[dict]:
    """Ledger rows for one seed's descent outcomes, each stamped with the seed's parameter-
    point identity `(c, p, z_{-1})` so distinct seeds are distinct places. `decoded_class`
    is left None here (this driver does not run the scorer — the smoke proves the identity
    plumbing, not q3 classification)."""
    rows = []
    for i, o in enumerate(walk_outcomes(pool)):
        oid = f"phr_{run_ts}_{seq0 + i:06d}"
        rows.append({
            "id": oid, "ts": run_ts, "family": "phoenix", "descend_mode": "phoenix_seed",
            "outcome_cx": float(o["cx"]), "outcome_cy": float(o["cy"]),
            "outcome_fw": float(o["fw"]), "reached_depth": int(o["depth"]),
            "branch": seed.branch, "theta": seed.theta, "offset": seed.offset,
            **ps.phoenix_ident_fields((seed.c.real, seed.c.imag),
                                      (seed.p.real, seed.p.imag),
                                      (seed.z_m1.real, seed.z_m1.imag)),
        })
    return rows


def run_batch(seeds, *, binary: Path, scratch: Path, ledger_path: Path,
              run_ts: str = "smoke", **descend_kw) -> list[dict]:
    """Descend every seed in the batch and append identity-stamped rows to `ledger_path`.
    Returns all rows. Each seed gets its own scratch subdir (per-seed pool)."""
    scratch.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    seq = 0
    with open(ledger_path, "a", encoding="utf-8") as fh:
        for si, seed in enumerate(seeds):
            pool = descend_seed(seed, binary=binary, out_dir=scratch / f"seed_{si:03d}",
                                **descend_kw)
            rows = seed_outcome_rows(seed, pool, run_ts=run_ts, seq0=seq)
            seq += len(rows)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            all_rows.extend(rows)
    return all_rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Descend a phoenix seed batch as roots.")
    ap.add_argument("--seed", type=int, default=0, help="sampler RNG seed")
    ap.add_argument("--n", type=int, default=3, help="proposals to descend")
    ap.add_argument("--n-walks", type=int, default=3)
    ap.add_argument("--depth-max", type=int, default=4)
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "out" / "phoenix" / "roots"))
    args = ap.parse_args(argv)
    binary = default_binary()
    if not binary.exists():
        raise SystemExit(f"release binary not found: {binary} (cargo build --release)")
    seeds = psamp.propose_batch(args.seed, args.n)
    out = Path(args.out)
    rows = run_batch(seeds, binary=binary, scratch=out / "scratch",
                     ledger_path=out / "outcome_ledger.jsonl",
                     n_walks=args.n_walks, depth_max=args.depth_max)
    cloud = ps.build_cloud([dict(r, decoded_class=3, guard_pass=True) for r in rows], "phoenix")
    print(f"descended {len(seeds)} seeds -> {len(rows)} outcome rows; "
          f"{len(cloud)} distinct phoenix places (identity-keyed) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
