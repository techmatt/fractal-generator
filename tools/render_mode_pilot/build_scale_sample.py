"""Scaled render-mode batch (1000 labels) — Step 1+2: fix the location-disjoint
split over the 112 v3-gate-passer locations, then deterministically stratified-sample
1000 NEW (location, palette, mode, params) tuples over the LOCKED 13-mode roster.

Prompt: prompts/scaled-batch-1000.md.

Roster (13): tia, stripe, curv_linear, gaussian_int, composite_c7/c13/c17,
smooth_mean_angle, smooth_angle_min, direct_trap_ring/screen/multiply/lines.
Dropped from the pilot's 15: trap_circle (dead solo) + exp_smoothing (near-smooth).

Split (stamped, reused forever): EVAL_FRAC=0.40, location-disjoint, seed=0.
  * Julia children inherit their PARENT split — a Julia's seed c=(c_re,c_im) is the
    parent point in its base-family plane; all Julia locations sharing a seed AND any
    base-family location sitting on that seed land in ONE split unit (union-find).
  * Split units (not raw locations) are family-stratified then 40% -> eval.
  Pilot rasters inherit this same location->side map at train time -> disjoint corpus.

Allocation (1000, mode-stratified, tilt-to-yield): floor 50/mode (650) + the remaining
350 apportioned proportional to each mode's PILOT q3-rate (largest-remainder to 1000).

Within-mode: distinct locations, family-apportioned (round-robin-equal), one gate-passing
palette per drawn location; direct-trap palette-deduped w/ a permuted opacity x threshold
cell per location. EXCLUDE any (location, mode, palette[, cell]) tuple already in the pilot.

Writes scratchpad/rms_sample_plan.jsonl (+ the split map) and prints the manifest,
split summary, and per-mode allocation BEFORE any rendering.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "corpus"))
import location as loc_mod  # noqa: E402
import numpy as np

RMS_SEED = 20260711
SPLIT_SEED = 0
EVAL_FRAC = 0.40
N_SAMPLE = 1000
FLOOR = 50

GATE_PASSERS = REPO / "scratchpad/gate_passers_v3.json"
PILOT_IMAGES = REPO / "data/render_mode_corpus/batches/2026-07-10_render_mode_pilot_v1/images.jsonl"
PILOT_LABELS = REPO / "labels/render_mode_pilot_v1.json"
OUT_PLAN = REPO / "scratchpad/rms_sample_plan.jsonl"
OUT_SPLIT = REPO / "scratchpad/rms_split_map.json"

# ---- locked 13-mode roster ------------------------------------------------- #
DIRECT_OPACITY = [0.15, 0.30, 0.45]
DIRECT_THRESHOLD = [0.05, 0.08, 0.12]
DIRECT_GRID = [(op, th) for op in DIRECT_OPACITY for th in DIRECT_THRESHOLD]  # 9 cells

MODES = [
    {"mode": "tia", "kind": "pure"},
    {"mode": "stripe", "kind": "pure"},
    {"mode": "gaussian_int", "kind": "pure"},
    {"mode": "curv_linear", "kind": "pure"},
    {"mode": "smooth_mean_angle", "kind": "composite"},
    {"mode": "smooth_angle_min", "kind": "composite"},
    {"mode": "composite_c7_smooth_trap_circle", "kind": "composite"},
    {"mode": "composite_c13_smooth_stripe", "kind": "composite"},
    {"mode": "composite_c17_smooth_curvature", "kind": "composite"},
    {"mode": "direct_trap_ring", "kind": "direct"},
    {"mode": "direct_trap_screen", "kind": "direct"},
    {"mode": "direct_trap_multiply", "kind": "direct"},
    {"mode": "direct_trap_lines", "kind": "direct"},
]
MODE_KIND = {m["mode"]: m["kind"] for m in MODES}
DROPPED = {"trap_circle", "exp_smoothing"}

# Julia family -> base (parent) family whose c-plane the seed lives in.
JULIA_PARENT = {"julia": "mandelbrot", "julia_multibrot4": "multibrot4",
                "julia_multibrot5": "multibrot5", "julia_multibrot3": "multibrot3"}


def lockey(row):
    return loc_mod.from_render_block(row["render"]).key()


# --------------------------------------------------------------------------- #
# Split units: union-find over locations linked by Julia-seed == parent-point.
# --------------------------------------------------------------------------- #
class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _fkey(v, ndig=12):
    """Float-tolerant string key for a decimal-string coordinate."""
    return f"{float(v):.{ndig}g}"


def build_split(locs):
    """locs: {location_key: representative gate-passer row}. Returns {location_key: side}."""
    uf = UF()
    for k in locs:
        uf.find(k)  # every location is its own node initially

    # seed node per (parent_family, c_re, c_im); link Julia locations to it.
    seed_nodes = {}   # (pfam, ckr, cki) -> node id
    def seed_node(pfam, cr, ci):
        key = (pfam, _fkey(cr), _fkey(ci))
        node = seed_nodes.setdefault(key, f"seed::{pfam}::{key[1]}::{key[2]}")
        return node, key

    base_pts = collections.defaultdict(dict)   # base_family -> {(fkx,fky): location_key}
    for k, r in locs.items():
        fam = r["family"]; rd = r["render"]
        if fam not in JULIA_PARENT:
            base_pts[fam][(_fkey(rd["cx"]), _fkey(rd["cy"]))] = k

    for k, r in locs.items():
        fam = r["family"]; rd = r["render"]
        if fam in JULIA_PARENT:
            pfam = JULIA_PARENT[fam]
            node, _ = seed_node(pfam, rd["c_re"], rd["c_im"])
            uf.union(k, node)

    # link a base-family location sitting exactly on a Julia seed to that seed unit.
    linked_parents = 0
    for (pfam, ckr, cki), node in seed_nodes.items():
        pk = base_pts.get(pfam, {}).get((ckr, cki))
        if pk is not None:
            uf.union(pk, node)
            linked_parents += 1

    # components over the real locations only
    comp = collections.defaultdict(list)
    for k in locs:
        comp[uf.find(k)].append(k)
    units = list(comp.values())

    # unit family = family of a base-family member if present else the modal family
    def unit_family(members):
        fams = [locs[m]["family"] for m in members]
        base = [f for f in fams if f not in JULIA_PARENT]
        return collections.Counter(base or fams).most_common(1)[0][0]

    # family-stratified seeded split over UNITS -> 40% eval
    rng = np.random.default_rng(SPLIT_SEED)
    strata = collections.defaultdict(list)
    for members in units:
        strata[unit_family(members)].append(tuple(sorted(members)))
    side = {}
    n_eval_units = 0
    for fam in sorted(strata):
        us = sorted(strata[fam])
        order = rng.permutation(len(us))
        n_ev = int(round(EVAL_FRAC * len(us)))
        ev = set(order[:n_ev].tolist())
        for i, members in enumerate(us):
            s = "eval" if i in ev else "train"
            n_eval_units += (1 if i in ev else 0)
            for m in members:
                side[m] = s
    meta = {"n_locations": len(locs), "n_units": len(units),
            "n_multi_loc_units": sum(1 for u in units if len(u) > 1),
            "linked_base_parents": linked_parents,
            "n_eval_units": n_eval_units,
            "n_eval_loc": sum(1 for s in side.values() if s == "eval"),
            "n_train_loc": sum(1 for s in side.values() if s == "train")}
    return side, meta


# --------------------------------------------------------------------------- #
def apportion(N, fam_avail, fam_order):
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


def pilot_q3_rates():
    """{mode: q3-rate} from the pilot label store joined to provenance (roster modes)."""
    labels = json.load(open(PILOT_LABELS))
    rows = [json.loads(l) for l in PILOT_IMAGES.read_text().splitlines() if l.strip()]
    agg = collections.defaultdict(lambda: [0, 0, 0])  # mode -> [q1,q2,q3]
    for r in rows:
        m = r["render"]["render_mode"]
        s = labels.get(r["image_id"])
        if s in (1, 2, 3):
            agg[m][s - 1] += 1
    return {m: (v[2] / sum(v) if sum(v) else 0.0) for m, v in agg.items()}


def largest_remainder(weights, total):
    """Apportion `total` integer units across keys ∝ weights (Hamilton method)."""
    keys = list(weights)
    w = np.array([weights[k] for k in keys], float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    frac = raw - base
    for i in np.argsort(-frac)[:rem]:
        base[i] += 1
    return {k: int(base[i]) for i, k in enumerate(keys)}


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8")
        except Exception: pass
    rows = json.load(open(GATE_PASSERS))

    # location -> rows / family; distinct-location representative
    loc_rows = collections.defaultdict(list)
    loc_fam, fam_locs, loc_rep = {}, collections.defaultdict(set), {}
    for i, r in enumerate(rows):
        k = lockey(r)
        loc_rows[k].append(i)
        loc_fam[k] = r["family"]
        fam_locs[r["family"]].add(k)
        loc_rep.setdefault(k, r)
    n_loc = len(loc_rows)
    fam_avail = {f: len(s) for f, s in fam_locs.items()}
    fam_order = sorted(fam_avail, key=lambda f: fam_avail[f])  # small families first

    # ---- split ---- #
    side, split_meta = build_split(loc_rep)

    # ---- pilot exclusion tuples ---- #
    pilot = [json.loads(l) for l in PILOT_IMAGES.read_text().splitlines() if l.strip()]
    excl = set()
    for r in pilot:
        m = r["render"]["render_mode"]; k = r["provenance"]["location_key"]
        if MODE_KIND.get(m) == "direct":
            mp = r["provenance"].get("mode_params", {})
            excl.add((k, m, "cell", round(mp.get("direct_opacity", 0), 4),
                      round(mp.get("direct_threshold", 0), 4)))
        else:
            excl.add((k, m, r["render"]["palette"]))

    # ---- allocation ---- #
    q3 = pilot_q3_rates()
    roster = [m["mode"] for m in MODES]
    floors = {m: FLOOR for m in roster}
    extra = largest_remainder({m: q3.get(m, 0.0) for m in roster}, N_SAMPLE - FLOOR * len(roster))
    targets = {m: floors[m] + extra[m] for m in roster}

    # ---- manifest header (printed FIRST) ---- #
    print("=" * 74)
    print("SCALED RENDER-MODE BATCH — plan (no rendering)")
    print("=" * 74)
    print(f"gate-passers: {len(rows)} rows · {n_loc} distinct locations")
    print(f"roster: {len(roster)} modes (dropped: {sorted(DROPPED)})")
    print(f"per-family location availability: "
          f"{dict(sorted(fam_avail.items(), key=lambda x:-x[1]))}")
    print("-" * 74)
    print(f"SPLIT  seed={SPLIT_SEED} eval_frac={EVAL_FRAC}  (Julia children inherit parent)")
    print(f"  units: {split_meta['n_units']} (multi-loc {split_meta['n_multi_loc_units']}, "
          f"base-parents linked {split_meta['linked_base_parents']})")
    print(f"  eval units {split_meta['n_eval_units']}  ·  "
          f"locations eval {split_meta['n_eval_loc']} / train {split_meta['n_train_loc']}")
    ff = collections.Counter((loc_fam[k], side[k]) for k in loc_rows)
    print("  family x side (locations): " +
          ", ".join(f"{f}:{ff.get((f,'train'),0)}T/{ff.get((f,'eval'),0)}E"
                    for f in sorted(fam_avail)))
    print("-" * 74)
    print("ALLOCATION  floor 50/mode (650) + 350 ∝ pilot q3-rate  (largest-remainder)")
    print(f"{'mode':<34}{'kind':<10}{'pilot_q3':>9}{'target':>8}")
    for m in sorted(roster, key=lambda x: -targets[x]):
        print(f"{m:<34}{MODE_KIND[m]:<10}{q3.get(m,0.0):>8.1%}{targets[m]:>8d}")
    print(f"{'TOTAL':<34}{'':<10}{'':>9}{sum(targets.values()):>8d}")
    print("=" * 74)

    # ---- sample ---- #
    rng = np.random.default_rng(RMS_SEED)
    plan = []
    per_mode = {}
    capped = {}
    for m in MODES:
        mode, kind = m["mode"], m["kind"]
        T = targets[mode]
        # eligible locations for this mode: >=1 non-excluded palette (direct: >=1 free cell)
        if kind == "direct":
            elig = {}   # loc -> list of free cells
            for k in loc_rows:
                free = [c for c in DIRECT_GRID
                        if (k, mode, "cell", round(c[0], 4), round(c[1], 4)) not in excl]
                if free:
                    elig[k] = free
        else:
            elig = {}   # loc -> list of non-excluded palettes
            for k in loc_rows:
                pals = sorted({rows[i]["palette"] for i in loc_rows[k]
                               if (k, mode, rows[i]["palette"]) not in excl})
                if pals:
                    elig[k] = pals
        elig_fam_avail = collections.Counter(loc_fam[k] for k in elig)
        avail_total = len(elig)
        if T > avail_total:
            capped[mode] = (T, avail_total)
            T = avail_total
        alloc = apportion(T, {f: elig_fam_avail.get(f, 0) for f in fam_order}, fam_order)
        drawn = []
        for f in fam_order:
            locs_f = sorted([k for k in elig if loc_fam[k] == f])
            idx = rng.permutation(len(locs_f))[:alloc[f]]
            drawn += [locs_f[j] for j in idx]
        rng.shuffle(drawn)
        # direct: permuted-cycle cell assignment, honoring per-loc free cells
        if kind == "direct":
            base_perm = [DIRECT_GRID[j] for j in rng.permutation(len(DIRECT_GRID))]
        for i, k in enumerate(drawn):
            rep = loc_rep[k]
            if kind == "direct":
                free = elig[k]
                pick = next((c for c in base_perm[i % len(base_perm):] + base_perm if c in free), free[0])
                op, th = pick
                # palette-deduped: one gate palette for this location (RNG among its rows)
                pal_row = rows[int(rng.choice(loc_rows[k]))]
                mode_params = {"direct_opacity": op, "direct_threshold": th}
                src = pal_row
            else:
                pal = elig[k][int(rng.integers(len(elig[k])))]
                src = next(rows[i2] for i2 in loc_rows[k] if rows[i2]["palette"] == pal)
                mode_params = {}
            plan.append({
                "mode": mode, "kind": kind, "location_key": k, "family": loc_fam[k],
                "split_side": side[k], "src_image_id": src["image_id"],
                "palette": src["palette"], "p_ge3": src["p_ge3"],
                "color_params": src["params"], "render": src["render"],
                "mode_params": mode_params,
            })
        per_mode[mode] = len(drawn)

    rng.shuffle(plan)  # blind-label order independence; ids assigned post-shuffle
    for i, e in enumerate(plan):
        e["image_id"] = f"rms_{i:04d}"

    OUT_PLAN.write_text("\n".join(json.dumps(e) for e in plan) + "\n")
    OUT_SPLIT.write_text(json.dumps(
        {"seed": SPLIT_SEED, "eval_frac": EVAL_FRAC, "meta": split_meta,
         "location_side": side}, indent=1))

    print(f"SAMPLE PLAN  {len(plan)} rows -> {OUT_PLAN.relative_to(REPO)}")
    print(f"split map -> {OUT_SPLIT.relative_to(REPO)}")
    print("  final per-mode counts:")
    for m in MODES:
        tag = ""
        if m["mode"] in capped:
            t0, av = capped[m["mode"]]
            tag = f"  (CAPPED: target {t0} > {av} eligible locs)"
        print(f"    {m['mode']:<34}{per_mode[m['mode']]:>4d}  [{m['kind']}]{tag}")
    fam_spread = collections.Counter(e["family"] for e in plan)
    side_spread = collections.Counter(e["split_side"] for e in plan)
    print(f"  family spread: {dict(fam_spread.most_common())}")
    print(f"  split spread:  {dict(side_spread)}")
    print(f"  distinct locations used: {len(set(e['location_key'] for e in plan))}")
    print(f"  distinct palettes used:  {len(set(e['palette'] for e in plan))}")
    if capped:
        print(f"  NOTE: {len(capped)} mode(s) capped below target -> total "
              f"{len(plan)} < {N_SAMPLE}")


if __name__ == "__main__":
    main()
