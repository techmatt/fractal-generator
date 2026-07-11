"""Render-mode pilot (500 stratified labels) — Step 1+2: enumerate the candidate
manifest over the 401 v3-gate-passers x every registered spec (except `smooth`), then
deterministically stratified-sample 500 (location, palette, mode, params) tuples.

Manifest rule (prompts/render-mode-pilot-500.md):
  * non-direct-trap modes (11): one tuple per gate-passing (location, palette) row.
  * direct-trap family (4 modes): PALETTE-DEDUPED to one-per-location (the family is
    palette-indifferent by construction), with the live opacity x threshold grid
    expanded (3x3 = 9 cells).  No DE spec is registered -> no de_scale axis.

Stratified sample (seed=RMP_SEED):
  * stratify by MODE, ~equal per mode (500 / 15 -> 33 or 34 each).
  * within a mode: distinct LOCATIONS, family-apportioned (round-robin equal capped by
    per-family location availability, leftover to the big families) so no one location
    or family dominates; one gate-passing palette per drawn location (RNG among its rows).
  * direct modes additionally spread the draw across the 9-cell param grid.

Writes scratchpad/rmp_sample_plan.jsonl (500 rows, the render plan — no rendering here)
and prints the manifest total + per-mode counts.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "corpus"))
import location as loc_mod  # noqa: E402

import numpy as np

RMP_SEED = 20260710
GATE_PASSERS = REPO / "scratchpad/gate_passers_v3.json"
OUT_PLAN = REPO / "scratchpad/rmp_sample_plan.jsonl"
N_SAMPLE = 500

# ---- mode registry (15; `smooth` excluded per the prompt) ------------------- #
# kind: pure=dump-field+python recolor (transfer=grad faithful);
#       composite / direct = Rust --coloring (transfer=grad not expressible).
DIRECT_OPACITY = [0.15, 0.30, 0.45]
DIRECT_THRESHOLD = [0.05, 0.08, 0.12]           # spans measured p75..p95 across shapes
DIRECT_GRID = [(op, th) for op in DIRECT_OPACITY for th in DIRECT_THRESHOLD]  # 9 cells

MODES = [
    # pure-field
    {"mode": "tia", "kind": "pure"},
    {"mode": "stripe", "kind": "pure"},
    {"mode": "exp_smoothing", "kind": "pure"},
    {"mode": "gaussian_int", "kind": "pure"},
    {"mode": "trap_circle", "kind": "pure"},
    {"mode": "curv_linear", "kind": "pure"},
    # composite (Rust)
    {"mode": "smooth_mean_angle", "kind": "composite"},
    {"mode": "smooth_angle_min", "kind": "composite"},
    {"mode": "composite_c7_smooth_trap_circle", "kind": "composite"},
    {"mode": "composite_c13_smooth_stripe", "kind": "composite"},
    {"mode": "composite_c17_smooth_curvature", "kind": "composite"},
    # direct-trap family (Rust; opacity x threshold sweep, palette-deduped/location)
    {"mode": "direct_trap_ring", "kind": "direct"},
    {"mode": "direct_trap_screen", "kind": "direct"},
    {"mode": "direct_trap_multiply", "kind": "direct"},
    {"mode": "direct_trap_lines", "kind": "direct"},
]
DIRECT_MODES = [m["mode"] for m in MODES if m["kind"] == "direct"]


def lockey(row):
    return loc_mod.from_render_block(row["render"]).key()


def apportion(N, fam_avail, fam_order):
    """Round-robin-equal allocation of N draws across families, capped by availability.
    Small families fill first; the remainder flows to the big families. -> {family: n}."""
    alloc = {f: 0 for f in fam_order}
    total = sum(min(N, fam_avail[f]) for f in fam_order)
    N = min(N, total)
    while sum(alloc.values()) < N:
        progressed = False
        for f in fam_order:
            if sum(alloc.values()) >= N:
                break
            if alloc[f] < fam_avail[f]:
                alloc[f] += 1
                progressed = True
        if not progressed:
            break
    return alloc


def main():
    rows = json.load(open(GATE_PASSERS))
    n_rows = len(rows)

    # location -> row indices; family -> set(locations)
    loc_rows = collections.defaultdict(list)
    loc_fam, fam_locs = {}, collections.defaultdict(set)
    for i, r in enumerate(rows):
        k = lockey(r)
        loc_rows[k].append(i)
        loc_fam[k] = r["family"]
        fam_locs[r["family"]].add(k)
    n_loc = len(loc_rows)
    fam_avail = {f: len(s) for f, s in fam_locs.items()}
    # family order: big families last so leftover naturally accrues to them
    fam_order = sorted(fam_avail, key=lambda f: fam_avail[f])

    # ---- manifest total ---- #
    n_nondirect_modes = sum(1 for m in MODES if m["kind"] != "direct")
    n_direct_modes = len(DIRECT_MODES)
    manifest_nondirect = n_nondirect_modes * n_rows
    manifest_direct = n_direct_modes * n_loc * len(DIRECT_GRID)
    manifest_total = manifest_nondirect + manifest_direct
    print("=" * 70)
    print(f"MANIFEST (pre-sample)")
    print(f"  gate-passers (401 rows): {n_rows}   distinct locations: {n_loc}")
    print(f"  registered modes (ex-smooth): {len(MODES)}  "
          f"(non-direct {n_nondirect_modes}, direct-trap {n_direct_modes})")
    print(f"  direct-trap grid: {len(DIRECT_OPACITY)} opacity x {len(DIRECT_THRESHOLD)} "
          f"threshold = {len(DIRECT_GRID)} cells")
    print(f"  non-direct tuples: {n_nondirect_modes} x {n_rows} = {manifest_nondirect}")
    print(f"  direct tuples:     {n_direct_modes} x {n_loc} loc x {len(DIRECT_GRID)} = {manifest_direct}")
    print(f"  TOTAL MANIFEST:    {manifest_total}")
    print(f"  per-family location availability: "
          f"{dict(sorted(fam_avail.items(), key=lambda x:-x[1]))}")

    # ---- per-mode target counts (largest-remainder, +1 to first `rem` modes) ---- #
    base, rem = divmod(N_SAMPLE, len(MODES))
    targets = {m["mode"]: base + (1 if i < rem else 0) for i, m in enumerate(MODES)}

    rng = np.random.default_rng(RMP_SEED)
    plan = []
    per_mode_counts = {}
    for m in MODES:
        mode, kind = m["mode"], m["kind"]
        N = targets[mode]
        alloc = apportion(N, fam_avail, fam_order)
        # draw distinct locations per family
        drawn_locs = []
        for f in fam_order:
            locs_f = sorted(fam_locs[f])
            idx = rng.permutation(len(locs_f))[:alloc[f]]
            drawn_locs += [locs_f[j] for j in idx]
        rng.shuffle(drawn_locs)
        # param-cell assignment (direct only): permuted cycle over the 9 cells
        if kind == "direct":
            cell_perm = rng.permutation(len(DIRECT_GRID))
            cells = [DIRECT_GRID[cell_perm[i % len(DIRECT_GRID)]] for i in range(len(drawn_locs))]
        for i, k in enumerate(drawn_locs):
            src_i = int(rng.choice(loc_rows[k]))      # one gate-passing palette/row for this loc
            src = rows[src_i]
            entry = {"mode": mode, "kind": kind, "location_key": k,
                     "family": loc_fam[k], "src_image_id": src["image_id"],
                     "palette": src["palette"], "p_ge3": src["p_ge3"],
                     "color_params": src["params"], "render": src["render"]}
            if kind == "direct":
                op, th = cells[i]
                entry["mode_params"] = {"direct_opacity": op, "direct_threshold": th}
            else:
                entry["mode_params"] = {}
            plan.append(entry)
        per_mode_counts[mode] = len(drawn_locs)

    # deterministic order + ids
    for i, e in enumerate(plan):
        e["image_id"] = f"rmp_{i:03d}"

    OUT_PLAN.write_text("\n".join(json.dumps(e) for e in plan) + "\n")
    print("=" * 70)
    print(f"SAMPLE PLAN ({len(plan)} rows) -> {OUT_PLAN.relative_to(REPO)}")
    print("  per-mode counts:")
    for m in MODES:
        print(f"    {m['mode']:34s} {per_mode_counts[m['mode']]:3d}  [{m['kind']}]")
    # family spread over the whole sample
    fam_spread = collections.Counter(e["family"] for e in plan)
    print(f"  family spread (whole sample): {dict(fam_spread.most_common())}")
    print(f"  distinct locations used: {len(set(e['location_key'] for e in plan))}")
    print(f"  distinct palettes used: {len(set(e['palette'] for e in plan))}")


if __name__ == "__main__":
    main()
