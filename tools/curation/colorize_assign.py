"""colorize_assign - collision-aware palette placement (first colorizer increment).

Consumes the soft-spread machinery (`colored_clip_spread.marginal_share_penalty`,
tau) to pick ONE palette per emitted location trading palette **fit** against
**colored-space spread**: two geometrically-distinct locations whose argmax
palettes would land on the same colored_clip region get pushed apart onto their
next-best distinct candidate.

Standalone + provable on the existing 47-location library record set (same pattern
as `colored_clip_spread` / `soft_spread_calibrate`). NOT wired into emit_v1 yet -
this is the placement pass emit_v1's coloring step would call.

Model
-----
Emitted set = the 47 library locations, each carrying its K=12 palette candidates.
Per candidate we have:
  * `colored_clip` - palette-ON CLIP vector (the store, keyed location_id/variant_id).
  * fit = `pref_score` - the v3-gvo preference score already recorded per candidate
    (`palette_candidates[].pref_score`); NO re-scoring.

Objective (greedy):  maximize  Σ_loc fit(pick_loc)  −  lam · Σ_pairs share_penalty
with share_penalty the soft ramp over colored_clip cosine at **tau=0.95**. Greedy
order = most fit-constrained first (largest top1−top2 fit gap): locations that
strongly prefer one palette are placed first and keep it; flexible locations are
placed last and absorb the spread cost by shifting to a distinct near-tie palette.

Baseline = current behavior: per-location argmax fit, spread ignored (lam=0).

    uv run python -m tools.curation.colorize_assign            # sweep + named checks
    uv run python -m tools.curation.colorize_assign --sheet    # + before/after recolor
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.curation import colored_clip_spread as ccs   # noqa: E402
from tools.curation import colored_clip as cc           # noqa: E402

TAU = 0.95
OUT = ROOT / "scratchpad/colorize_assign"


# --------------------------------------------------------------------------- #
# Emitted set: locations -> K candidate keys + per-candidate fit.
# --------------------------------------------------------------------------- #
@dataclass
class EmittedSet:
    """The placement problem instance.

    locs      : ordered location_ids.
    cands     : loc -> [candidate keys "loc/variant"], in record order.
    fit       : key -> pref_score (v3-gvo fit signal).
    store     : ColoredStore (colored_clip vectors + palette meta).
    """

    locs: list[str]
    cands: dict[str, list[str]]
    fit: dict[str, float]
    store: ccs.ColoredStore

    def palette(self, key: str) -> str:
        return self.store.meta[key]["palette"]


def load_emitted(records_path: Path = cc.RECORDS) -> EmittedSet:
    store = ccs.load_store(records_path=records_path)
    locs, cands, fit = [], {}, {}
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        loc = rec["location_id"]
        locs.append(loc)
        keys = []
        for c in rec["palette_candidates"]:
            key = f"{loc}/{c['variant_id']}"
            keys.append(key)
            fit[key] = float(c["pref_score"])
        cands[loc] = keys
    return EmittedSet(locs=locs, cands=cands, fit=fit, store=store)


# --------------------------------------------------------------------------- #
# Assignment.
# --------------------------------------------------------------------------- #
@dataclass
class Assignment:
    pick: dict[str, str]         # loc -> chosen candidate key
    lam: float
    tau: float

    def keys(self, locs: list[str]) -> list[str]:
        return [self.pick[l] for l in locs]


def baseline_argmax(es: EmittedSet) -> Assignment:
    """Current behavior: per-location argmax fit, spread ignored."""
    pick = {loc: max(keys, key=lambda k: es.fit[k]) for loc, keys in es.cands.items()}
    return Assignment(pick=pick, lam=0.0, tau=TAU)


def constrained_order(es: EmittedSet, locs: list[str] | None = None) -> list[str]:
    """Most fit-constrained first: descending top1-top2 fit gap.

    A location whose best palette dominates its runners-up has the most to lose by
    being moved, so it is placed first (keeps its strong palette); flat-fit
    locations are placed last, where a near-tie alternative absorbs spread cheaply.

    `locs` restricts the ordering to a subset (default all `es.locs`) — the
    incremental pass orders only the NOT-yet-fixed locations.
    """
    def gap(loc: str) -> float:
        fs = sorted((es.fit[k] for k in es.cands[loc]), reverse=True)
        return fs[0] - (fs[1] if len(fs) > 1 else fs[0])
    return sorted(locs if locs is not None else es.locs, key=gap, reverse=True)


def assign(es: EmittedSet, lam: float, tau: float = TAU,
           order: list[str] | None = None,
           existing: dict[str, str] | None = None) -> Assignment:
    """Greedy collision-aware placement.

    For each location (in `order`), pick the candidate maximizing
    fit - lam * marginal_share_penalty(candidate, already-assigned) at `tau`.

    `existing` (loc -> already-chosen candidate key) implements the ONLINE /
    incremental `existing=` pattern: those locations are FIXED — their picks are
    pre-seeded into the placed set and never reconsidered — and only the remaining
    locations are placed greedily against them (and each other). This is what keeps
    a growing corpus from reshuffling: prior assignments locked, new locations
    spread against the fixed set. With `existing=None` (the default) the walk covers
    every location and is byte-identical to the non-incremental behavior.
    """
    existing = existing or {}
    to_place = [l for l in es.locs if l not in existing]
    order = order if order is not None else constrained_order(es, to_place)
    pick: dict[str, str] = dict(existing)
    placed: list[str] = list(existing.values())
    for loc in order:
        best_k, best_v = None, -np.inf
        for k in es.cands[loc]:
            pen = ccs.marginal_share_penalty(k, placed, es.store, tau=tau) if lam else 0.0
            v = es.fit[k] - lam * pen
            if v > best_v:
                best_v, best_k = v, k
        pick[loc] = best_k
        placed.append(best_k)
    return Assignment(pick=pick, lam=lam, tau=tau)


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #
def min_pairwise_dist(es: EmittedSet, keys: list[str]) -> float:
    """Min pairwise colored_clip cosine distance across the assigned set."""
    if len(keys) < 2:
        return float("inf")
    d = ccs.cosine_dist_matrix(es.store.rows(keys))
    iu = np.triu_indices(len(keys), k=1)
    return float(d[iu].min())


def n_pairs_over_tau(es: EmittedSet, keys: list[str], tau: float = TAU) -> int:
    """# assigned pairs whose colored_clip cosine >= tau (the collision count)."""
    if len(keys) < 2:
        return 0
    s = ccs.cosine_sim_matrix(es.store.rows(keys))
    iu = np.triu_indices(len(keys), k=1)
    return int((s[iu] >= tau).sum())


def collisions(es: EmittedSet, keys: list[str], tau: float = TAU) -> list[tuple]:
    """Assigned pairs with cos >= tau, most-similar first: (cos, loc_i, loc_j)."""
    s = ccs.cosine_sim_matrix(es.store.rows(keys))
    locs = [k.split("/", 1)[0] for k in keys]
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if s[i, j] >= tau:
                out.append((float(s[i, j]), locs[i], keys[i], locs[j], keys[j]))
    return sorted(out, reverse=True)


def total_fit(es: EmittedSet, a: Assignment) -> float:
    return float(sum(es.fit[k] for k in a.pick.values()))


def total_share_penalty(es: EmittedSet, a: Assignment, tau: float = TAU) -> float:
    return ccs.share_penalty(list(a.pick.values()), es.store, tau=tau)


def compare(es: EmittedSet, base: Assignment, a: Assignment) -> dict:
    bkeys, akeys = base.keys(es.locs), a.keys(es.locs)
    reassigned = [l for l in es.locs if base.pick[l] != a.pick[l]]
    return dict(
        lam=a.lam,
        n_reassigned=len(reassigned),
        reassigned=reassigned,
        total_fit=total_fit(es, a),
        fit_sacrificed=total_fit(es, base) - total_fit(es, a),
        min_dist_before=min_pairwise_dist(es, bkeys),
        min_dist_after=min_pairwise_dist(es, akeys),
        n_over_tau_before=n_pairs_over_tau(es, bkeys, a.tau),
        n_over_tau_after=n_pairs_over_tau(es, akeys, a.tau),
        share_penalty_before=total_share_penalty(es, base),
        share_penalty_after=total_share_penalty(es, a),
    )


# --------------------------------------------------------------------------- #
# Sweep + named checks + report.
# --------------------------------------------------------------------------- #
# afmhot twin: the same-appearance afmhot candidates on two different locations
# (colored_clip cos ~0.965). afmhot is the rank-1 near-tie behind each location's
# argmax, so a spread-BLIND tie-break could pull both onto afmhot and recreate the
# collision. The check: collision-aware placement must keep them on distinct palettes
# without collapsing fit.
NAMED_TWINS = [
    ("cycle_001_wfd_000_02", "wfd_000_01", "cycle_005_wfd_001_09", "wfd_001_01"),
]


def run(es: EmittedSet, lambdas: list[float], focus_lam: float | None = None) -> dict:
    base = baseline_argmax(es)
    order = constrained_order(es)
    assigns = {lam: assign(es, lam, order=order) for lam in lambdas}
    rows = [compare(es, base, assigns[lam]) for lam in lambdas]
    bt = total_fit(es, base)
    bmin = rows[0]["min_dist_before"]

    W = 96
    print("=" * W)
    print(f"colorize_assign - {len(es.locs)} locations x K={len(es.cands[es.locs[0]])} "
          f"candidates, tau={TAU}")
    print("=" * W)
    print(f"baseline (argmax fit): total_fit={bt:.2f}  min_pairwise_dist={bmin:.4f}  "
          f"pairs>=tau={rows[0]['n_over_tau_before']}  "
          f"share_penalty={rows[0]['share_penalty_before']:.2f}")
    from collections import Counter
    coll = Counter(es.palette(k) for k in base.pick.values())
    print("  argmax palette-NAME collisions (>1 loc): "
          + ", ".join(f"{p}x{n}" for p, n in coll.most_common() if n > 1))
    print("  note: name-collisions != colored_clip collisions (same palette on "
          "different geometry looks different); the objective targets the latter.")
    print()
    print(f"{'lambda':>7} {'reassign':>9} {'fit_sacr':>9} {'fit_sacr%':>9} "
          f"{'min_dist':>9} {'d_dist':>8} {'pairs>=t':>9} {'sharePen':>9}")
    print("-" * W)
    for r in rows:
        print(f"{r['lam']:>7.2f} {r['n_reassigned']:>9d} {r['fit_sacrificed']:>9.3f} "
              f"{r['fit_sacrificed']/bt*100:>8.2f}% {r['min_dist_after']:>9.4f} "
              f"{r['min_dist_after']-bmin:>+8.4f} {r['n_over_tau_after']:>9d} "
              f"{r['share_penalty_after']:>9.3f}")
    print()

    # --- collision resolution at a focus lambda ---
    focus_lam = focus_lam if focus_lam in assigns else lambdas[-1]
    base_coll = collisions(es, base.keys(es.locs), TAU)
    foc_coll = collisions(es, assigns[focus_lam].keys(es.locs), TAU)
    foc_set = {(a, b) for _, a, _, b, _ in foc_coll}
    print(f"collision resolution @ lambda={focus_lam:g}  "
          f"({len(base_coll)} baseline pairs>=tau -> {len(foc_coll)} remaining):")
    for cos, la, _, lb, _ in base_coll:
        status = "STILL" if (la, lb) in foc_set else "fixed"
        print(f"  [{status}] cos={cos:.4f}  {la:<22}[{es.palette(base.pick[la])}]  |  "
              f"{lb:<22}[{es.palette(base.pick[lb])}]")
    print()

    # --- named twin check across the sweep ---
    named = {}
    print("named check - afmhot twin (rank-1 near-tie for both; must NOT co-select):")
    for la, va, lb, vb in NAMED_TWINS:
        ka0, kb0 = f"{la}/{va}", f"{lb}/{vb}"
        twin_cos = float(es.store.vec(ka0) @ es.store.vec(kb0))
        print(f"  afmhot cand [{ka0}] vs [{kb0}]  colored_clip cos={twin_cos:.4f} "
              f"(fit {es.fit[ka0]:.2f}/{es.fit[kb0]:.2f}; each is its loc's rank-1)")
        pair_rows = []
        for lam in lambdas:
            a = assigns[lam]
            ka, kb = a.pick[la], a.pick[lb]
            both_afmhot = es.palette(ka) == "afmhot" and es.palette(kb) == "afmhot"
            split = es.palette(ka) != es.palette(kb)
            pair_rows.append(dict(lam=lam, pal_a=es.palette(ka), pal_b=es.palette(kb),
                                  split=split, both_afmhot=both_afmhot))
            flag = "BOTH-afmhot!" if both_afmhot else ("split" if split else "same-pal")
            print(f"    lam={lam:>5.2f}: {es.palette(ka):<26} | {es.palette(kb):<26} {flag}")
        named[f"{la}|{lb}"] = dict(twin_cos=twin_cos, rows=pair_rows)
    print("=" * W)

    return dict(
        baseline=dict(total_fit=bt, min_dist=bmin,
                      n_over_tau=rows[0]["n_over_tau_before"],
                      share_penalty=rows[0]["share_penalty_before"],
                      picks={l: es.palette(base.pick[l]) for l in es.locs},
                      collisions=[(c, a, b) for c, a, _, b, _ in base_coll]),
        sweep=rows, named=named, order=order, tau=TAU, focus_lam=focus_lam)


# --------------------------------------------------------------------------- #
# Optional before/after contact sheet for the reassigned locations.
# --------------------------------------------------------------------------- #
def build_sheet(es: EmittedSet, base: Assignment, a: Assignment, out_png: Path):
    from PIL import Image, ImageDraw, ImageFont
    from tools import colormap as cm

    reassigned = [l for l in es.locs if base.pick[l] != a.pick[l]]
    if not reassigned:
        print("no reassignments -> no sheet")
        return
    recs = {}
    for line in cc.RECORDS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["location_id"]] = r
    lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))

    # need base+new variant per reassigned loc; one field dump per loc.
    need: dict[str, set[str]] = {}
    for l in reassigned:
        need.setdefault(l, set()).update(
            {base.pick[l].split("/", 1)[1], a.pick[l].split("/", 1)[1]})

    imgs: dict[str, Image.Image] = {}
    for i, (loc, variants) in enumerate(need.items(), 1):
        rec = recs[loc]
        binp, jsonp = cc.ensure_field(rec)
        field = cm.load_field(str(binp), str(jsonp))
        prep = cm.stretch_field(field)
        profile = None
        by_var = {c["variant_id"]: c for c in rec["palette_candidates"]}
        for var in sorted(variants):
            cfg = cc.candidate_config(field, by_var[var])
            if cfg.transfer == "grad" and profile is None:
                profile = cm.gradient_transfer_profile(field, prep)
            rgb = cm.render_candidate(field, cfg, lib, prep=prep, profile=profile)
            imgs[f"{loc}/{var}"] = Image.fromarray(rgb)
        print(f"[{i}/{len(need)}] {loc}  +{len(variants)} variants", flush=True)

    TW, TH, PAD, LH = 256, 144, 8, 22
    font = _font(13)
    row_w = max(2 * TW + 3 * PAD, 900)   # widen so the reassignment label fits
    row_h = LH + TH + PAD
    sheet = Image.new("RGB", (row_w, PAD + row_h * len(reassigned)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    for i, loc in enumerate(reassigned):
        y = PAD + i * row_h
        bk, ak = base.pick[loc], a.pick[loc]
        cos = float(es.store.vec(bk) @ es.store.vec(ak))
        label = (f"{loc}   baseline [{es.palette(bk)}]  ->  new [{es.palette(ak)}]   "
                 f"fitd={es.fit[ak]-es.fit[bk]:+.2f}  recolor-cos={cos:.3f}")
        draw.text((PAD, y + 4), label, fill=(255, 210, 120), font=font)
        for j, key in enumerate((bk, ak)):
            th = imgs[key].resize((TW, TH), Image.LANCZOS)
            sheet.paste(th, (PAD + j * (TW + PAD), y + LH))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"wrote {out_png}  ({sheet.size[0]}x{sheet.size[1]})", flush=True)


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="fit<->spread knob sweep")
    ap.add_argument("--sheet", action="store_true",
                    help="also render before/after contact sheet at --sheet-lambda")
    ap.add_argument("--sheet-lambda", type=float, default=4.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    es = load_emitted()
    result = run(es, args.lambdas, focus_lam=args.sheet_lambda)
    (OUT / "sweep.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT/'sweep.json'}")

    if args.sheet:
        base = baseline_argmax(es)
        a = assign(es, args.sheet_lambda, order=result["order"])
        build_sheet(es, base, a, OUT / f"before_after_lam{args.sheet_lambda:g}.png")


if __name__ == "__main__":
    main()
