"""Category-conditioned colorizer — v1 (v3-gvo as-is).

Conditioning is a CONSTRAINT, not a model change: given a target k16 color category
(a "cell"), pick the best palette that LIVES IN that cell for a location. The
deployed pref scorer (pref-v3-gvo, `data/queries/scorer/data.ACTIVE_SCORER_DIR`) is
used verbatim — no retrain, no new scorer, nothing touching emit's v3 gate.

Why this bypasses the beam's K=12
---------------------------------
The emit beam (`sample_location.py`) is a GLOBAL argmax: it draws ~60 palettes,
keeps the 18 the scorer likes best, and never surfaces a palette the global argmax
disliked. Conditioning asks the opposite question — "of the palettes that are IN
cell X, which does this location like best?" — so it must score the CELL's palettes
directly rather than filter the beam's K candidates (filtering K would only ever
return cells the global argmax already liked, defeating the point). We therefore
recolor+score every pool palette in the target cell.

The pool the cells are defined over
-----------------------------------
k16 cells come from `data/palettes/palette_categories.json`, whose 987 `palettes`
entries are EXACTLY the **production** pool (`pool_colormaps.json`: curated_q3 76 +
curated_q2 115 + extracted 470 + dramatic 326 = 987) — NOT the 76-curated score-3
pool. So conditioning is over the full production pool. 19 cells: k16 leaves 1..16
(chromatic) + the three fixed specials (spectral 57, outlier 46, neutral 15).

Fit signal + the within-location fence
--------------------------------------
`fit` = the v3-gvo score of the recolored frame (`query_batch_gen.score_frames` on
the deploy transform). It is a single-tower RANKING-margin head: scores are
comparable ONLY WITHIN a location (identical geometry), never across locations. So
every conditioning quantity here is within-location relative:
  * global-argmax fit  = max fit over ALL 987 palettes (canonical recipe) for a loc.
  * fitΔ(cell)         = best_in_cell fit − global-argmax fit  (≤ 0). This is the
    **cost of conditioning** and, being a within-location delta, IS comparable
    across locations (the one quantity the realizability matrix aggregates).

Canonical-recipe simplification (documented, on purpose)
--------------------------------------------------------
Every palette is applied under ONE canonical recipe (pct stretch, gamma 1, no
reverse/phase/cycles) — `colormap`'s plain linear spec, the same anchor the query
generator uses. Conditioning chooses the best PALETTE in a cell; recipe optimization
(gamma / phase / transfer='grad') is the beam's orthogonal job and is NOT swept here
("apply candidate colormaps, score" — don't re-render per candidate). So a cell's fit
is that cell's best palette under a fixed recipe, uniformly across all cells — a fair
relative comparison, not the palette's tuned ceiling.

Composition with the spread work (optional)
-------------------------------------------
`best_in_cell(..., assigned=...)` optionally subtracts λ·marginal_share_penalty
(reuses `colored_clip_spread`; τ=0.95, λ=3) so conditioning composes with the
collision-aware placement. The penalty needs a colored_clip vector for the candidate
recolor; we resolve it from the store when the (loc,palette) pair is a beam candidate
already embedded there, else the penalty is skipped for that candidate (embedding a
non-beam recolor would need a CLIP forward — out of scope for the matrix, which runs
assigned=() anyway).

    uv run python -m tools.curation.conditioned_colorize --smoke        # 2 locs, quick
    uv run python -u tools/curation/conditioned_colorize.py             # full 47-loc run
    uv run python -m tools.curation.conditioned_colorize --report-only  # rebuild matrix/sheet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "queries"))
sys.path.insert(0, str(ROOT / "tools" / "queries" / "scorer"))

from tools import colormap as cm                       # noqa: E402
from tools.curation import colored_clip as cc          # noqa: E402  cached-field recolor + POOL/FEATURES
from tools.curation import colored_clip_spread as ccs   # noqa: E402  cell_label + marginal_share_penalty
import query_batch_gen as P                             # noqa: E402  score_frames (deploy transform)
import train as ST                                      # noqa: E402  build_model
import data as SD                                       # noqa: E402  ACTIVE_SCORER_DIR (pref-v3-gvo)

RECORDS = cc.RECORDS
CATEGORIES = ROOT / "data/palettes/palette_categories.json"
OUT = ROOT / "scratchpad/conditioned_colorize"
CELL_LEVEL = ccs.DEFAULT_CELL_LEVEL   # "k16"

# Composition knobs (mirror colorize_assign / recolor_pass).
TAU = 0.95
LAM = 3.0
MAX_WORKERS = 4   # hard cap (CLAUDE.md) — recolor thread pool


# --------------------------------------------------------------------------- #
# Cell map: production pool palette -> k16 cell tag (reusing ccs.cell_label).
# --------------------------------------------------------------------------- #
def _cat_to_color_category(entry: dict) -> dict:
    """palette_categories.json entry -> the record-shaped color_category dict that
    `ccs.cell_label` consumes. categories store cluster under `cluster{"8/12/16"}`;
    the record store keys them `k8/k12/k16`. Specials carry the string tag through."""
    cl = entry.get("cluster", {})
    return {
        "special": entry.get("special"),
        "k8": cl.get("8"),
        "k12": cl.get("12"),
        "k16": cl.get("16"),
        "leaf_pos": entry.get("leaf_pos"),
    }


def load_cell_map(level: str = CELL_LEVEL):
    """(palette_name -> cell_tag, cell_tag -> [palette_names]) over the production pool.

    cell_tag is exactly `ccs.cell_label(...)` so conditioning cells are byte-identical
    to the spread work's cells (special:* short-circuits the numbered leaves)."""
    cats = json.loads(CATEGORIES.read_text())["palettes"]
    name_to_cell, cell_to_names = {}, {}
    for name, entry in cats.items():
        tag = ccs.cell_label(_cat_to_color_category(entry), level)
        name_to_cell[name] = tag
        cell_to_names.setdefault(tag, []).append(name)
    return name_to_cell, cell_to_names


# --------------------------------------------------------------------------- #
# Scorer + canonical recolor.
# --------------------------------------------------------------------------- #
class Scorer:
    """The deployed pref-v3-gvo head, resolved from the single-source pointer."""

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ST.build_model().to(self.device)
        ck = torch.load(Path(SD.ACTIVE_SCORER_DIR) / "model_best.pt",
                        map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.epoch = ck.get("epoch")
        self.name = Path(SD.ACTIVE_SCORER_DIR).name

    def score(self, imgs):
        return P.score_frames(self.model, imgs, self.device)


def canonical_config(field, palette: str) -> cm.CandidateConfig:
    """The one canonical linear spec (gamma=1, pct, no reverse/phase/cycles), box
    filter — matches the query generator's anchor + colored_clip's morphology canon."""
    ow, oh = field.out_size
    return cm.CandidateConfig(
        palette=palette, location=field.location,
        eval_width=ow, eval_height=oh, filter="box",
    )


@dataclass
class LocationFits:
    """Per-location canonical-recipe fit of every scored pool palette."""
    loc_id: str
    fits: dict          # palette -> v3-gvo fit (canonical recipe)
    scorer: str
    coarse: bool

    def to_json(self) -> dict:
        return {"loc_id": self.loc_id, "scorer": self.scorer, "coarse": self.coarse,
                "fits": {k: round(float(v), 6) for k, v in self.fits.items()}}

    @staticmethod
    def from_json(d: dict) -> "LocationFits":
        return LocationFits(loc_id=d["loc_id"], fits=d["fits"],
                            scorer=d.get("scorer", "?"), coarse=bool(d.get("coarse", True)))


def score_location(rec: dict, palettes: list[str], lib, scorer: Scorer,
                   coarse: bool = True, workers: int = MAX_WORKERS) -> LocationFits:
    """Recolor `palettes` off the location's cached field (canonical recipe) and score
    every frame with v3-gvo. Recolor is threaded (numpy releases the GIL) up to
    `workers` (<=4); scoring is one batched GPU pass. Coarse path is SCORING-ONLY."""
    binp, jsonp = cc.ensure_field(rec)
    field = cm.load_field(str(binp), str(jsonp))
    prep = cm.stretch_field(field)
    cfield = cm.coarse_field(prep) if coarse else None

    def recolor(pn: str):
        cfg = canonical_config(field, pn)
        if coarse:
            return cm.render_candidate_coarse(cfield, cfg, lib)
        return cm.render_candidate(field, cfg, lib, prep=prep)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
            imgs = list(ex.map(recolor, palettes))
    else:
        imgs = [recolor(pn) for pn in palettes]
    scores = scorer.score(imgs)
    return LocationFits(rec["location_id"], dict(zip(palettes, scores)),
                        scorer=scorer.name, coarse=coarse)


# --------------------------------------------------------------------------- #
# The conditioning primitive.
# --------------------------------------------------------------------------- #
@dataclass
class CellFit:
    """best_in_cell result for one (location, cell)."""
    cell: str
    palette: str | None
    fit: float                 # best fit in the cell (canonical recipe; may include −λ·penalty)
    fit_raw: float             # best fit WITHOUT the spread penalty
    n_palettes: int            # how many pool palettes live in this cell (== how many scored)
    spread: float              # max−min raw fit across the cell (within-cell discrimination)
    std: float                 # std of raw fit across the cell


def best_in_cell(loc_fits: LocationFits, cell_to_names: dict, target_cell: str,
                 assigned: tuple = (), store=None, loc_id: str | None = None,
                 lam: float = LAM, tau: float = TAU) -> CellFit:
    """Best palette IN `target_cell` for a location, by v3-gvo fit.

    `loc_fits.fits` holds precomputed canonical-recipe fit for every scored palette
    (score_location did the expensive recolor+score once). This is a pure argmax over
    the cell's palettes — cheap, so a whole realizability row is a dict scan.

    Optional spread composition: subtract `lam * marginal_share_penalty(candidate,
    assigned)` (reuses colored_clip_spread, τ=`tau`). The candidate's colored_clip
    vector is resolved from `store` when (loc_id, palette) is a beam candidate already
    embedded there; otherwise that candidate incurs zero penalty (see module doc)."""
    fits = loc_fits.fits
    members = [p for p in cell_to_names.get(target_cell, []) if p in fits]
    if not members:
        return CellFit(target_cell, None, float("-inf"), float("-inf"), 0, 0.0, 0.0)

    raw = np.array([fits[p] for p in members], dtype=float)
    spread = float(raw.max() - raw.min())
    std = float(raw.std())

    pen_lookup = _store_vec_lookup(store, loc_id) if (store is not None and assigned) else None
    best_p, best_v, best_raw = None, float("-inf"), float("-inf")
    for p, r in zip(members, raw):
        v = r
        if pen_lookup is not None:
            key = pen_lookup(p)
            if key is not None:
                v = r - lam * ccs.marginal_share_penalty(key, list(assigned), store, tau=tau)
        if v > best_v:
            best_v, best_p, best_raw = v, p, r
    return CellFit(target_cell, best_p, float(best_v), float(best_raw),
                   len(members), spread, std)


def _store_vec_lookup(store, loc_id):
    """(palette -> store key) for the given location, over the beam candidates that
    ARE embedded in the colored_clip store. Returns None-tolerant callable."""
    if store is None or loc_id is None:
        return None
    by_pal = {}
    for k, m in store.meta.items():
        if m["location_id"] == loc_id:
            by_pal[m["palette"]] = k     # last variant wins; fine for a spread penalty
    return lambda p: by_pal.get(p)


def global_argmax(loc_fits: LocationFits) -> tuple[str, float]:
    """Best palette over ALL scored palettes (the within-location unconstrained argmax)."""
    p = max(loc_fits.fits, key=lambda k: loc_fits.fits[k])
    return p, float(loc_fits.fits[p])


# --------------------------------------------------------------------------- #
# Run: score every location, persist per-location fits (resumable).
# --------------------------------------------------------------------------- #
def _load_records(path: Path = RECORDS) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_scoring(records: list[dict], name_to_cell: dict, out: Path,
                coarse: bool = True, limit_palettes: int = 0, resume: bool = True):
    """Score every location against the production pool (canonical recipe). Writes
    `fits/<loc>.json` per location so a crash/interrupt resumes. Returns the fits dir."""
    fits_dir = out / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)

    lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))
    # Only score palettes that both live in a cell AND load in the library.
    palettes = [p for p in name_to_cell if p in lib.colormaps]
    if limit_palettes:
        palettes = palettes[:limit_palettes]
    print(f"[cond] pool={len(palettes)} palettes · {len(records)} locations · "
          f"coarse={coarse} · scorer={Path(SD.ACTIVE_SCORER_DIR).name}", flush=True)

    scorer = Scorer()
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        fp = fits_dir / f"{rec['location_id']}.json"
        if resume and fp.exists():
            try:
                lf = LocationFits.from_json(json.loads(fp.read_text()))
                if len(lf.fits) >= len(palettes) and lf.coarse == coarse:
                    print(f"[{i}/{len(records)}] {rec['location_id']}  cached", flush=True)
                    continue
            except Exception:
                pass
        tl = time.time()
        lf = score_location(rec, palettes, lib, scorer, coarse=coarse)
        fp.write_text(json.dumps(lf.to_json()))
        gp, gv = global_argmax(lf)
        print(f"[{i}/{len(records)}] {rec['location_id']}  {len(lf.fits)} scored  "
              f"[{time.time()-tl:.0f}s]  global-argmax={gp}({gv:.2f})", flush=True)
    print(f"[cond] scoring done · {time.time()-t0:.0f}s total -> {fits_dir}", flush=True)
    return fits_dir


def load_all_fits(fits_dir: Path) -> dict[str, LocationFits]:
    out = {}
    for fp in sorted(fits_dir.glob("*.json")):
        lf = LocationFits.from_json(json.loads(fp.read_text()))
        out[lf.loc_id] = lf
    return out


# --------------------------------------------------------------------------- #
# Realizability matrix (locations x cells) + report.
# --------------------------------------------------------------------------- #
def build_matrix(all_fits: dict[str, LocationFits], cell_to_names: dict,
                 name_to_cell: dict) -> dict:
    """47 locations x 19 cells. Per (loc, cell): best palette, best/raw fit, fitΔ vs the
    location's global argmax (cost of conditioning), within-cell spread + std."""
    cells = sorted(cell_to_names, key=_cell_sort_key)
    matrix = {}
    global_pick = {}
    for loc_id, lf in all_fits.items():
        gp, gv = global_argmax(lf)
        global_pick[loc_id] = {"palette": gp, "fit": gv, "cell": name_to_cell.get(gp)}
        row = {}
        for cell in cells:
            cf = best_in_cell(lf, cell_to_names, cell)
            if cf.palette is None:
                row[cell] = None
                continue
            row[cell] = {
                "palette": cf.palette,
                "fit": round(cf.fit_raw, 5),
                "fit_delta": round(cf.fit_raw - gv, 5),   # cost of conditioning (<= 0)
                "n_palettes": cf.n_palettes,
                "spread": round(cf.spread, 5),
                "std": round(cf.std, 5),
            }
        matrix[loc_id] = row
    return {"cells": cells, "global_pick": global_pick, "matrix": matrix}


def _cell_sort_key(cell: str):
    """k16 leaves numeric-ascending, then specials alpha — stable column order."""
    if cell.startswith("special:"):
        return (2, cell)
    if cell.startswith("k16:"):
        try:
            return (0, int(cell.split(":")[1]))
        except ValueError:
            return (1, cell)
    return (3, cell)


def analyze(mat: dict, name_to_cell: dict) -> dict:
    """The four questions the prompt asks, computed from the matrix (all within-location
    deltas, so cross-location aggregation is legitimate)."""
    cells = mat["cells"]
    matrix = mat["matrix"]
    locs = list(matrix)

    # -- per-cell cost of conditioning: distribution of fitΔ across locations --
    per_cell = {}
    for cell in cells:
        deltas = [matrix[l][cell]["fit_delta"] for l in locs if matrix[l][cell]]
        realized = [l for l in locs if matrix[l][cell]]
        n_pal = next((matrix[l][cell]["n_palettes"] for l in realized), 0)
        if deltas:
            arr = np.array(deltas)
            per_cell[cell] = {
                "n_palettes": n_pal,
                "n_locs_realized": len(realized),
                "median_fit_delta": round(float(np.median(arr)), 4),
                "best_fit_delta": round(float(arr.max()), 4),   # closest to 0 = most reachable
                "worst_fit_delta": round(float(arr.min()), 4),
                "mean_fit_delta": round(float(arr.mean()), 4),
            }
        else:
            per_cell[cell] = {"n_palettes": n_pal, "n_locs_realized": 0}

    # cells no location realizes well: rank by median fitΔ (most negative = hardest)
    hard_cells = sorted((c for c in cells if per_cell[c].get("n_locs_realized")),
                        key=lambda c: per_cell[c]["median_fit_delta"])

    # -- cost-of-conditioning distribution over ALL (loc, non-argmax-cell) pairs --
    all_deltas = []
    for l in locs:
        gcell = mat["global_pick"][l]["cell"]
        for cell in cells:
            if matrix[l][cell] and cell != _tag_of(gcell):
                all_deltas.append(matrix[l][cell]["fit_delta"])
    ad = np.array(all_deltas) if all_deltas else np.array([0.0])
    cost_dist = {
        "n_pairs": len(all_deltas),
        "median": round(float(np.median(ad)), 4),
        "mean": round(float(ad.mean()), 4),
        "p10": round(float(np.percentile(ad, 10)), 4),
        "p90": round(float(np.percentile(ad, 90)), 4),
        "frac_within_1pt": round(float((ad > -1.0).mean()), 4),
        "frac_within_2pt": round(float((ad > -2.0).mean()), 4),
    }

    # -- within-cell discrimination: does v3-gvo separate palettes INSIDE a cell? --
    # spread (max−min raw fit) per (loc, cell), aggregated. If ~flat, within-cell
    # choice is arbitrary => a conditioned scorer is worth building.
    spreads, stds = [], []
    per_cell_spread = {}
    for cell in cells:
        cs = [matrix[l][cell]["spread"] for l in locs if matrix[l][cell] and matrix[l][cell]["n_palettes"] > 1]
        ss = [matrix[l][cell]["std"] for l in locs if matrix[l][cell] and matrix[l][cell]["n_palettes"] > 1]
        spreads += cs
        stds += ss
        if cs:
            per_cell_spread[cell] = {"median_spread": round(float(np.median(cs)), 4),
                                     "median_std": round(float(np.median(ss)), 4)}
    sp = np.array(spreads) if spreads else np.array([0.0])
    return_within = {
        "median_within_cell_spread": round(float(np.median(sp)), 4),
        "mean_within_cell_spread": round(float(sp.mean()), 4),
        "median_within_cell_std": round(float(np.median(stds)) if stds else 0.0, 4),
        "per_cell": per_cell_spread,
    }

    return {
        "per_cell_cost": per_cell,
        "hardest_cells": [{"cell": c, **per_cell[c]} for c in hard_cells[:6]],
        "easiest_cells": [{"cell": c, **per_cell[c]} for c in hard_cells[::-1][:6]],
        "cost_distribution": cost_dist,
        "within_cell_discrimination": return_within,
    }


def _tag_of(cell_or_int):
    """global_pick stores the cell tag already; passthrough (guards a None)."""
    return cell_or_int


def bmw_crosscheck(all_fits: dict, mat: dict, name_to_cell: dict,
                   cell_to_names: dict, bmw="cet_linear_bmw_5_95_c86") -> dict:
    """The over-dense cet_linear_bmw cluster: which locations' GLOBAL argmax lands in
    bmw's cell, and can they be conditioned into OTHER cells cheaply (small |fitΔ|)?
    Bears on whether residual collisions are fixable by conditioning vs pool rebalancing."""
    bmw_cell = name_to_cell.get(bmw)
    matrix = mat["matrix"]
    cells = mat["cells"]
    locs_in_bmw = [l for l, gp in mat["global_pick"].items() if gp["cell"] == bmw_cell]
    rows = []
    for l in locs_in_bmw:
        # best alternative cell (largest fitΔ, i.e. closest to 0) other than bmw's cell
        alts = [(matrix[l][c]["fit_delta"], c, matrix[l][c]["palette"])
                for c in cells if matrix[l][c] and c != bmw_cell]
        alts.sort(reverse=True)
        rows.append({
            "loc": l,
            "global_palette": mat["global_pick"][l]["palette"],
            "global_fit": round(mat["global_pick"][l]["fit"], 4),
            "best_alt_cell": alts[0][1] if alts else None,
            "best_alt_palette": alts[0][2] if alts else None,
            "best_alt_fit_delta": round(alts[0][0], 4) if alts else None,
        })
    return {"bmw_palette": bmw, "bmw_cell": bmw_cell,
            "n_locs_global_in_bmw_cell": len(locs_in_bmw), "locations": rows}


# --------------------------------------------------------------------------- #
# Contact sheet: a few locations x several cells (one thumbnail per cell).
# --------------------------------------------------------------------------- #
def contact_sheet(records: list[dict], all_fits: dict, mat: dict, cell_to_names: dict,
                  out_png: Path, n_locs: int = 4, n_cells: int = 8):
    """For a few locations, render best-in-cell for the n_cells most-populated cells (full
    render_candidate path — a diagnostic thumbnail, NOT a scoring-fence coarse frame)."""
    from PIL import Image, ImageDraw, ImageFont

    recs_by_loc = {r["location_id"]: r for r in records}
    lib = cm.PaletteLibrary(str(cc.POOL_COLORMAPS), str(cc.FEATURES))

    # pick n_locs spread across families; pick the n_cells biggest chromatic cells
    locs = _spread_locations(records, all_fits, n_locs)
    cells = [c for c in sorted(cell_to_names, key=lambda c: -len(cell_to_names[c]))
             if c.startswith("k16:")][:n_cells]

    TW, TH, PAD, LH, HEAD = 200, 112, 6, 16, 40
    W = PAD + len(cells) * (TW + PAD)
    H = HEAD + len(locs) * (LH + TH + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = _font(11)
    draw.text((PAD, 4), "category-conditioned colorize — best palette per cell "
              "(fitΔ = cost vs global argmax)", fill=(235, 235, 235), font=_font(13))
    # column headers
    for j, cell in enumerate(cells):
        draw.text((PAD + j * (TW + PAD), HEAD - 14), cell, fill=(150, 200, 235), font=font)

    for i, loc in enumerate(locs):
        rec = recs_by_loc[loc]
        lf = all_fits[loc]
        binp, jsonp = cc.ensure_field(rec)
        field = cm.load_field(str(binp), str(jsonp))
        prep = cm.stretch_field(field)
        y = HEAD + i * (LH + TH + PAD)
        gcell = mat["global_pick"][loc]["cell"]
        draw.text((PAD, y), f"{loc}  (argmax cell {gcell})", fill=(220, 220, 160), font=font)
        for j, cell in enumerate(cells):
            cf = best_in_cell(lf, cell_to_names, cell)
            x = PAD + j * (TW + PAD)
            yy = y + LH
            if cf.palette is None:
                draw.rectangle([x, yy, x + TW, yy + TH], fill=(40, 40, 44))
                continue
            cfg = canonical_config(field, cf.palette)
            rgb = cm.render_candidate(field, cfg, lib, prep=prep)
            th = Image.fromarray(rgb).resize((TW, TH), Image.LANCZOS)
            sheet.paste(th, (x, yy))
            gv = mat["global_pick"][loc]["fit"]
            lbl = f"{cf.palette[:22]}"
            sub = f"d{cf.fit_raw - gv:+.2f}" + ("  ARGMAX" if cell == gcell else "")
            draw.text((x + 2, yy + TH - 22), lbl, fill=(240, 240, 240), font=font)
            draw.text((x + 2, yy + TH - 11), sub, fill=(160, 230, 170) if cell == gcell else (200, 200, 210), font=font)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"[sheet] {out_png}  ({W}x{H}) · {len(locs)} locs x {len(cells)} cells", flush=True)
    return locs, cells


def _spread_locations(records, all_fits, k):
    """k locations present in all_fits, spread across families then fw."""
    have = [r for r in records if r["location_id"] in all_fits]
    by_fam = {}
    for r in have:
        by_fam.setdefault(r["identity"].get("family"), []).append(r)
    out, fams = [], sorted(by_fam)
    fi = 0
    while len(out) < min(k, len(have)):
        fam = fams[fi % len(fams)]
        bucket = sorted(by_fam[fam], key=lambda r: float(r["identity"]["fw"]))
        for r in bucket:
            if r["location_id"] not in out:
                out.append(r["location_id"])
                break
        fi += 1
        if fi > 4 * k:
            break
    return out[:k]


def _font(sz):
    from PIL import ImageFont
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def write_report(out: Path, mat: dict, ana: dict, bmw: dict, name_to_cell: dict,
                 sheet_info=None):
    (out / "matrix.json").write_text(json.dumps(mat, indent=1))
    (out / "analysis.json").write_text(json.dumps(
        {"analysis": ana, "bmw_crosscheck": bmw}, indent=2))

    cd = ana["cost_distribution"]
    wc = ana["within_cell_discrimination"]
    L = []
    L.append("# Category-conditioned colorizer — realizability matrix (v3-gvo as-is)\n")
    L.append(f"Pool = **production** `pool_colormaps.json` (987 palettes = the set "
             f"`palette_categories.json` k16 is cut over). Scorer = **pref-v3-gvo**, "
             f"canonical recipe (pct/γ1/no reverse·phase·cycles). Fit is within-location "
             f"only; every number below is a within-location Δ.\n")
    L.append(f"**{len(mat['matrix'])} locations × {len(mat['cells'])} cells** "
             f"(16 k16 chromatic leaves + spectral/outlier/neutral specials).\n")

    L.append("## Cost of conditioning\n")
    L.append(f"Over all **{cd['n_pairs']}** (location, non-argmax cell) pairs, fitΔ = "
             f"best-in-cell − global-argmax:\n")
    L.append(f"- median **{cd['median']}** · mean **{cd['mean']}** · p10 **{cd['p10']}** · "
             f"p90 **{cd['p90']}**")
    L.append(f"- within 1 pt of the global argmax: **{cd['frac_within_1pt']*100:.0f}%** · "
             f"within 2 pt: **{cd['frac_within_2pt']*100:.0f}%**\n")

    L.append("## Cells hardest to realize (most-negative median fitΔ)\n")
    L.append("| cell | #pal | locs | median Δ | worst Δ | best Δ |")
    L.append("|------|-----:|-----:|---------:|--------:|-------:|")
    for r in ana["hardest_cells"]:
        L.append(f"| {r['cell']} | {r['n_palettes']} | {r['n_locs_realized']} | "
                 f"{r['median_fit_delta']} | {r['worst_fit_delta']} | {r['best_fit_delta']} |")
    L.append("")
    L.append("Easiest (least cost to condition into):\n")
    L.append("| cell | #pal | locs | median Δ |")
    L.append("|------|-----:|-----:|---------:|")
    for r in ana["easiest_cells"]:
        L.append(f"| {r['cell']} | {r['n_palettes']} | {r['n_locs_realized']} | {r['median_fit_delta']} |")
    L.append("")

    L.append("## Does v3-gvo discriminate WITHIN a cell?\n")
    L.append(f"Within-cell fit spread (max−min raw fit across the palettes in one cell), "
             f"per (loc, cell), median over all cells with ≥2 palettes:\n")
    L.append(f"- **median within-cell spread = {wc['median_within_cell_spread']}** "
             f"(mean {wc['mean_within_cell_spread']}) · median within-cell std "
             f"{wc['median_within_cell_std']}")
    L.append(f"- Compare to the cost-of-conditioning scale above (p10 {cd['p10']}). If the "
             f"within-cell spread is on the ORDER of the between-cell cost, v3-gvo IS "
             f"resolving palettes inside a cell and within-cell argmax is meaningful; if it "
             f"is ~0, within-cell choice is arbitrary and a conditioned scorer is worth "
             f"building.\n")

    L.append("## cet_linear_bmw over-dense cluster cross-check\n")
    L.append(f"`{bmw['bmw_palette']}` sits in cell **{bmw['bmw_cell']}**. "
             f"**{bmw['n_locs_global_in_bmw_cell']}** location(s) have their GLOBAL argmax "
             f"in that cell. Cheapest alternative cell per such location:\n")
    if bmw["locations"]:
        L.append("| loc | global palette | best alt cell | alt palette | alt fitΔ |")
        L.append("|-----|----------------|---------------|-------------|---------:|")
        for r in bmw["locations"]:
            L.append(f"| {r['loc']} | {r['global_palette'][:22]} | {r['best_alt_cell']} | "
                     f"{(r['best_alt_palette'] or '')[:22]} | {r['best_alt_fit_delta']} |")
    else:
        L.append("_No location's global argmax lands in the bmw cell._")
    L.append("")

    if sheet_info:
        locs, cells = sheet_info
        L.append("## Contact sheet\n")
        L.append(f"`contact_sheet.png` — {len(locs)} locations × {len(cells)} cells "
                 f"(one best-in-cell thumbnail per cell, ARGMAX cell tagged).\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {out/'report.md'}  +  matrix.json  +  analysis.json", flush=True)


# --------------------------------------------------------------------------- #
def _report_pipeline(out: Path, records, name_to_cell, cell_to_names, make_sheet=True):
    fits_dir = out / "fits"
    all_fits = load_all_fits(fits_dir)
    if not all_fits:
        raise SystemExit(f"no fits under {fits_dir} — run scoring first")
    print(f"[cond] loaded fits for {len(all_fits)} locations", flush=True)
    mat = build_matrix(all_fits, cell_to_names, name_to_cell)
    ana = analyze(mat, name_to_cell)
    bmw = bmw_crosscheck(all_fits, mat, name_to_cell, cell_to_names)
    sheet_info = None
    if make_sheet:
        recs_have = [r for r in records if r["location_id"] in all_fits]
        sheet_info = contact_sheet(recs_have, all_fits, mat, cell_to_names,
                                   out / "contact_sheet.png")
    write_report(out, mat, ana, bmw, name_to_cell, sheet_info)
    return mat, ana, bmw


def main():
    ap = argparse.ArgumentParser(description="Category-conditioned colorizer (v3-gvo as-is).")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--full-res", action="store_true",
                    help="score on the full render path (default: coarse SCORING path)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 locations x 120 palettes — quick end-to-end")
    ap.add_argument("--limit-locs", type=int, default=0)
    ap.add_argument("--limit-palettes", type=int, default=0)
    ap.add_argument("--report-only", action="store_true",
                    help="skip scoring; rebuild matrix/report/sheet from fits/")
    ap.add_argument("--no-sheet", action="store_true")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    name_to_cell, cell_to_names = load_cell_map()
    records = _load_records()
    if args.smoke:
        args.limit_locs, args.limit_palettes = args.limit_locs or 2, args.limit_palettes or 120
    if args.limit_locs:
        records = records[:args.limit_locs]

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_scoring(records, name_to_cell, args.out,
                    coarse=not args.full_res, limit_palettes=args.limit_palettes)
    _report_pipeline(args.out, records, name_to_cell, cell_to_names,
                     make_sheet=not args.no_sheet)


if __name__ == "__main__":
    main()
