"""Generate the `prefv2_dramatic_v1` labeling batch — dramatic-inclusive within-location
palette-preference queries that UNION with the existing pref-v2 corpus (coldstart_v2 +
warmstart_v1).

The point: the existing corpus carries within-palette param-ranking over POOL
(curated/extracted) palettes only — the 326 dramatic palettes were added to the pool
AFTER warmstart_v1 was generated (see Step-0 findings). This batch puts dramatic +
pool colorings on the SAME q3 locations so a retrained pref-v2 can rank the families
against each other, and adds a dramatic within-palette param-ranking arm that never
existed before.

Everything commensurable with the existing corpus is REUSED VERBATIM:
  * render spec — CANDIDATE_SS/EVAL_WIDTH/EVAL_HEIGHT/CANDIDATE_FILTER from query_sampler
    (1024x576, ss2, box, interior black), colormap.render_candidate (Recipe-2),
  * once-per-location ss2 field dump + cache (assemble_queries.ensure_field, out/fields/),
  * the batch artifact layout (records/ + images/ + per-query contact sheet + batch_meta)
    that launch_query_label_server.py + query_label.html consume — same 3-tier UI,
  * palette_source provenance (already wired in assemble_queries.candidate_record).

What is NEW here (and only here):
  * three query TYPES on the *palette-source* axis — within_dramatic / cross_source /
    param_variation — persisted as a `query_type` field on every candidate record so eval
    can stratify by type (pref-v2 is NOT consulted anywhere in selection),
  * the widened per-candidate param draw (n_cycles in {1..5}, gradient transfer, phase-0
    anchoring on dramatic) — the sampler D in query_sampler caps n_cycles at {1,2} and has
    no transfer, so the draw lives here (mirrors sample_location's transfer/n_cycles sweep).

Locations: q3 mandelbrot+julia (the 629 set), PREFERRING fresh (disjoint from the
existing corpus's locations). <=2 queries per location, and a doubled location's two
queries are a DIFFERENT type, so one field dump feeds multiple queries.

    uv run python tools/queries/prefv2_dramatic_v1.py --estimate   # plan + runtime est, exit
    uv run python tools/queries/prefv2_dramatic_v1.py              # full run (background it)
    uv run python tools/queries/prefv2_dramatic_v1.py --report-only
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import query_sampler as qs                 # noqa: E402  (pool, sampler, ranges, EVAL_*/SS/FILTER)
import assemble_queries as aq              # noqa: E402  (ensure_field, _field_key, query_record, OUT_FIELDS, EXE)
import location as loc_mod                 # noqa: E402  (canonical Location: key, from_render_block, render_one_flags)
sys.path.insert(0, str(qs.ROOT / "tools" / "palettes"))
import palette_features as pf              # noqa: E402  (distance_matrix, farthest_point_order over trajectory features)

cm = qs.cm
ROOT = qs.ROOT
BATCH_ID = "prefv2_dramatic_v1"
BATCH_DIR = ROOT / "data" / "queries" / BATCH_ID

# ===========================================================================
# Fixed batch spec (the approved grid).
# ===========================================================================

SEED = 3                                   # qids q003_XXXX (coldstart used 0, warmstart 2)
N_QUERIES = 600
QUERY_TYPE_COUNTS = {"within_dramatic": 240, "cross_source": 210, "param_variation": 150}
assert sum(QUERY_TYPE_COUNTS.values()) == N_QUERIES
CANDIDATES_PER_QUERY = 6

# Locations: prefer fresh q3 mandelbrot+julia; <=2 queries/loc (doubled => two distinct
# types). 350 distinct locations, 250 doubled + 100 single => 600 queries. Family mix
# mirrors the existing corpus (mandelbrot/julia 64/36 at the QUERY level), realized by
# doubling/singling proportionally so the query mix is exactly 64/36:
#   mandelbrot: 160 double + 64 single = 224 locs -> 384 queries
#   julia     :  90 double + 36 single = 126 locs -> 216 queries
N_LOCATIONS = 350
FAMILY_PLAN = {                            # family -> (n_double, n_single)
    "mandelbrot": (160, 64),
    "julia":      (90, 36),
}
assert sum(d + s for d, s in FAMILY_PLAN.values()) == N_LOCATIONS
assert sum(2 * d + s for d, s in FAMILY_PLAN.values()) == N_QUERIES

# Existing pref-v2 corpus (its locations are what "fresh" is disjoint from).
EXISTING_BATCHES = ("coldstart_v2", "warmstart_v1")

# --- per-candidate param draw (the sampler D, widened for this batch) --------
GAMMA_LO, GAMMA_HI = 1.0 / 3.0, 3.0        # gamma log-uniform [1/3, 3]
N_CYCLES = [1, 2, 3, 4, 5]                 # cyclic band-repeat multiplier
TRANSFER_GAMMAS = [0.25, 0.5, 1.0, 2.0]    # gradient-transfer exponent; drawn from {pct} U these
PREMAPS = ["none", "log"]                  # the real render_candidate premap set (Step 0)
PREMAP_LOG_P = 0.5                         # uniform over {none, log}
DRAMATIC_PHASE0_P = 1.0 / 3.0             # dramatic: phase=0 w.p. ~1/3, else U[0,1)
REVERSE_P = 0.5                            # non-cyclic (pool sequential) reverse Bernoulli

# --- cross_source composition -----------------------------------------------
CROSS_DRAMATIC_P = 0.6                      # each slot's source ~ 60/40 dramatic/pool
POOL_SOURCES = ("curated_q3", "curated_q2", "extracted")   # "pool" = curated + extracted

# --- per-query palette FP subsample (per-query variety over a fixed bucket) --
# farthest_point_order is deterministic; to give each query a DIFFERENT diverse set we
# FP over a per-query random subsample of the source bucket rather than the whole bucket.
FP_SUBSAMPLE = 48
PARAM_POOL = 24                            # param_variation: draw this many, spread-select 5 (+anchor)

# --- wall discipline ---------------------------------------------------------
CAP_MIN_DEFAULT = 88.0                     # soft: don't START a unit that can't finish by here
HARD_MIN_DEFAULT = 90.0                    # hard backstop: break between queries past here
DUMP_TIMEOUT = 120                         # per-dump subprocess timeout (s); q3 f64, none expected
RECOLOR_WORKERS = 4                        # max-4 cap (project rule)


# ===========================================================================
# Location selection: fresh q3 mandelbrot+julia, family-mixed.
# ===========================================================================

def existing_location_keys():
    """Canonical location keys already in the pref-v2 corpus (coldstart_v2 + warmstart_v1),
    read from their durable query records and keyed through the SAME canonical path the
    pool dedups on (location.from_render_block -> .key())."""
    keys = set()
    for b in EXISTING_BATCHES:
        for rp in glob.glob(str(ROOT / "data" / "queries" / b / "records" / "*.json")):
            L = json.loads(Path(rp).read_text())["location"]
            block = dict(fractal_type=L["family"], cx=L["cx"], cy=L["cy"], fw=L["fw"],
                         maxiter=L["maxiter"], c_re=L.get("c_re"), c_im=L.get("c_im"))
            keys.add(loc_mod.from_render_block(block).key())
    return keys


def select_locations(pool, seed):
    """Pick N_LOCATIONS fresh q3 mandelbrot+julia locations at the FAMILY_PLAN counts.

    Fresh = q3 location NOT in the existing corpus. Within a family, drawn by an fw
    (zoom/complexity) quantile spread so the set isn't clustered at one depth. Returns
    (units, stats) where units is a list of dicts {ref, family, n_queries} — n_queries in
    {1,2} per the double/single split — in a deterministic order."""
    rng = np.random.default_rng(seed)
    existing = existing_location_keys()
    q3 = {pl.ref.key(): pl for pl in pool.locations if pl.ref.kind in FAMILY_PLAN}
    fresh_by_fam = {}
    for k, pl in q3.items():
        if k not in existing:
            fresh_by_fam.setdefault(pl.ref.kind, []).append(pl)

    def spread_pick(cands, k):
        """k locations spread over fw: sort, cut into k quantile bins, one random per bin."""
        cands = sorted(cands, key=lambda pl: float(pl.ref.fw))
        n = len(cands)
        out = []
        for b in range(k):
            lo, hi = b * n // k, (b + 1) * n // k
            j = int(rng.integers(lo, max(lo + 1, hi)))
            out.append(cands[min(j, n - 1)])
        return out

    units = []
    stats = {"fresh_available": {f: len(v) for f, v in fresh_by_fam.items()},
             "existing_in_corpus": len(existing), "used_existing": 0}
    for fam, (n_dbl, n_sgl) in FAMILY_PLAN.items():
        avail = fresh_by_fam.get(fam, [])
        need = n_dbl + n_sgl
        if len(avail) < need:
            raise RuntimeError(f"only {len(avail)} fresh {fam} locations, need {need}")
        picked = spread_pick(avail, need)
        rng.shuffle(picked)
        for i, pl in enumerate(picked):
            n_q = 2 if i < n_dbl else 1        # first n_dbl are doubled
            units.append({"ref": pl.ref, "loc": pl, "family": fam, "n_queries": n_q})
    rng.shuffle(units)
    return units, stats


# ===========================================================================
# Query-type assignment: exact global counts, doubled units get two DISTINCT types.
# ===========================================================================

def assign_types(units, seed):
    """Attach a `types` list (len == n_queries) to each unit so global type counts hit
    QUERY_TYPE_COUNTS exactly and every doubled unit's two types differ.

    Builds the 600-token type multiset, shuffles, then walks the units: a single takes one
    token; a double takes two — if they collide, swap the second with the next differing
    token later in the stream (always possible: max type count 240 < 300 = #doubles+... )."""
    rng = np.random.default_rng(seed + 101)
    tokens = []
    for t, c in QUERY_TYPE_COUNTS.items():
        tokens += [t] * c
    rng.shuffle(tokens)
    tokens = list(tokens)

    # Order units doubles-first so collisions are resolved while the stream is longest.
    order = sorted(range(len(units)), key=lambda i: -units[i]["n_queries"])
    pos = 0
    for i in order:
        n = units[i]["n_queries"]
        if n == 1:
            units[i]["types"] = [tokens[pos]]
            pos += 1
        else:
            a = tokens[pos]
            b = tokens[pos + 1]
            if a == b:
                # find a later token != a and swap into pos+1
                for j in range(pos + 2, len(tokens)):
                    if tokens[j] != a:
                        tokens[pos + 1], tokens[j] = tokens[j], tokens[pos + 1]
                        b = tokens[pos + 1]
                        break
            assert a != b, "could not build a distinct-type pair"
            units[i]["types"] = [a, b]
            pos += 2
    assert pos == N_QUERIES
    return units


# ===========================================================================
# Palette buckets + per-query farthest-point selection.
# ===========================================================================

class Palettes:
    """Dramatic / pool name buckets + per-query FP selection over trajectory features."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.feats = sampler.feats
        names = [n for n in sampler.feats if n in sampler.library.colormaps]
        self.dramatic = [n for n in names if sampler.source_of(n) == "dramatic"]
        self.pool = [n for n in names if sampler.source_of(n) in POOL_SOURCES]

    def fp_select(self, rng, bucket, k):
        """k diverse palettes from `bucket` by farthest-point over a per-query random
        subsample (so different queries get different diverse sets)."""
        if k <= 0:
            return []
        if len(bucket) <= k:
            sub = list(bucket)
        else:
            m = min(FP_SUBSAMPLE, len(bucket))
            m = max(m, k)
            idx = rng.choice(len(bucket), size=m, replace=False)
            sub = [bucket[int(i)] for i in idx]
        return pf.farthest_point_order(sub, features_by_name=self.feats, k=k)


# ===========================================================================
# Per-candidate param draw (the widened sampler D).
# ===========================================================================

def _draw_transfer(rng):
    """Uniform over {pct} U TRANSFER_GAMMAS -> (transfer, transfer_gamma)."""
    k = int(rng.integers(len(TRANSFER_GAMMAS) + 1))
    return ("pct", 0.0) if k == 0 else ("grad", float(TRANSFER_GAMMAS[k - 1]))


def draw_params(rng, source, ptype):
    """One param draw. gamma log-uniform; premap uniform {none,log}; transfer uniform
    {pct}U{0.25,0.5,1,2}. Cyclic: n_cycles~U{1..5}, phase (dramatic: 0 w.p.1/3 else U;
    pool cyclic: U[0,1)), reverse=False. Non-cyclic (pool sequential): reverse~Bern, no
    phase/n_cycles."""
    gamma = float(math.exp(rng.uniform(math.log(GAMMA_LO), math.log(GAMMA_HI))))
    log_premap = "log" if rng.random() < PREMAP_LOG_P else "none"
    transfer, tg = _draw_transfer(rng)
    if ptype == "cyclic":
        n_cycles = int(N_CYCLES[int(rng.integers(len(N_CYCLES)))])
        if source == "dramatic":
            phase = 0.0 if rng.random() < DRAMATIC_PHASE0_P else float(rng.random())
        else:
            phase = float(rng.random())
        return dict(reverse=False, log_premap=log_premap, gamma=gamma, phase=phase,
                    n_cycles=n_cycles, transfer=transfer, transfer_gamma=tg)
    return dict(reverse=bool(rng.random() < REVERSE_P), log_premap=log_premap, gamma=gamma,
                phase=0.0, n_cycles=1, transfer=transfer, transfer_gamma=tg)


def _cfg(ref, palette, params):
    return cm.CandidateConfig(
        palette=palette, location=loc_mod.to_location_ref(ref),
        eval_width=qs.EVAL_WIDTH, eval_height=qs.EVAL_HEIGHT, filter=qs.CANDIDATE_FILTER,
        **params)


def _ptype(lib, name):
    return lib.palette_type(name)


# ===========================================================================
# The three query builders. Each returns [CandidateConfig] of length 6.
# ===========================================================================

def build_within_dramatic(ref, rng, pals, lib):
    """6 distinct dramatic palettes (FP over the dramatic subspace), one param draw each."""
    names = pals.fp_select(rng, pals.dramatic, CANDIDATES_PER_QUERY)
    return [_cfg(ref, n, draw_params(rng, "dramatic", _ptype(lib, n))) for n in names]


def build_cross_source(ref, rng, pals, lib):
    """6 slots, each source ~60/40 dramatic/pool with >=1 of each guaranteed; palettes FP
    within their source bucket; one param draw each."""
    n_dram = int(rng.binomial(CANDIDATES_PER_QUERY, CROSS_DRAMATIC_P))
    n_dram = min(max(n_dram, 1), CANDIDATES_PER_QUERY - 1)   # guarantee >=1 dramatic AND >=1 pool
    dram = pals.fp_select(rng, pals.dramatic, n_dram)
    poolp = pals.fp_select(rng, pals.pool, CANDIDATES_PER_QUERY - n_dram)
    entries = [(n, "dramatic") for n in dram] + [(n, "pool") for n in poolp]
    rng.shuffle(entries)
    return [_cfg(ref, n, draw_params(rng, src, _ptype(lib, n))) for n, src in entries]


def build_param_variation(ref, rng, pals, lib):
    """One dramatic palette, 6 param draws spanning the dials: a neutral phase=0 anchor
    plus 5 spread by farthest-point over param space (from a PARAM_POOL draw)."""
    palette = pals.dramatic[int(rng.integers(len(pals.dramatic)))]
    ptype = _ptype(lib, palette)   # dramatic -> cyclic
    anchor = dict(reverse=False, log_premap="none", gamma=1.0, phase=0.0, n_cycles=1,
                  transfer="pct", transfer_gamma=0.0)
    poolp = [draw_params(rng, "dramatic", ptype) for _ in range(PARAM_POOL)]

    def feat(p):
        tg = p["transfer_gamma"] / 2.0 if p["transfer"] == "grad" else 0.0
        return np.array([
            math.log(p["gamma"]) / math.log(GAMMA_HI),      # ~[-1,1]
            p["phase"],                                       # [0,1)
            (p["n_cycles"] - 1) / (len(N_CYCLES) - 1),        # [0,1]
            tg,                                               # [0,1]
            1.0 if p["log_premap"] == "log" else 0.0,
            1.0 if p["reverse"] else 0.0,
        ])

    afeat = feat(anchor)
    feats = [feat(p) for p in poolp]
    chosen = [anchor]                                          # anchor forced in
    min_d = np.array([np.linalg.norm(f - afeat) for f in feats])
    picked_idx = set()
    while len(chosen) < CANDIDATES_PER_QUERY and len(picked_idx) < len(poolp):
        j = int(np.argmax([d if i not in picked_idx else -np.inf for i, d in enumerate(min_d)]))
        picked_idx.add(j)
        chosen.append(poolp[j])
        min_d = np.minimum(min_d, [np.linalg.norm(f - feats[j]) for f in feats])
    return [_cfg(ref, palette, p) for p in chosen]


BUILDERS = {
    "within_dramatic": build_within_dramatic,
    "cross_source": build_cross_source,
    "param_variation": build_param_variation,
}


# ===========================================================================
# Field dump with a hard timeout (backstop against a pathological unit).
# ===========================================================================

def ensure_field_timeout(ref, timeout=DUMP_TIMEOUT):
    """aq.ensure_field, but with a subprocess timeout so one stuck dump can't hang the
    run. Returns (FieldData, dump_seconds); raises subprocess.TimeoutExpired on stall."""
    aq.OUT_FIELDS.mkdir(parents=True, exist_ok=True)
    stem = aq._field_key(ref)
    bin_path = aq.OUT_FIELDS / f"{stem}.bin"
    json_path = aq.OUT_FIELDS / f"{stem}.json"
    dump_secs = 0.0
    if not (bin_path.exists() and json_path.exists()):
        cmd = [str(aq.EXE), "render-one", "--cx", ref.cx, "--cy", ref.cy, "--fw", ref.fw,
               "--width", str(qs.EVAL_WIDTH), "--height", str(qs.EVAL_HEIGHT),
               "--supersample", str(qs.CANDIDATE_SS), "--maxiter", str(ref.maxiter),
               "--dump-field", str(bin_path)]
        cmd += loc_mod.render_one_flags(ref)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        dump_secs = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError(f"dump-field failed for {stem}:\n{r.stderr[-400:]}")
    return cm.load_field(str(bin_path), str(json_path)), dump_secs


# ===========================================================================
# Persistence + resume.
# ===========================================================================

def query_is_done(qid):
    rec_ok = (BATCH_DIR / "records" / f"{qid}.json").exists()
    imgs_ok = all((BATCH_DIR / "images" / f"{qid}_{k}.png").exists()
                  for k in range(CANDIDATES_PER_QUERY))
    return rec_ok and imgs_ok


def contact_sheet(imgs, cands, qid, query_type, sampler, out_path,
                  thumb_w=512, pad=8, bar=34):
    """3x2 eye-check sheet. Caption shows the NEW axes (source + transfer) beyond the
    stock gamma/phase/n_cycles/reverse/log."""
    cols, rows = 3, 2
    tw = thumb_w
    th = round(tw * qs.EVAL_HEIGHT / qs.EVAL_WIDTH)
    W = cols * tw + (cols + 1) * pad
    H = rows * (th + bar) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 2), f"{qid}  [{query_type}]", fill=(230, 230, 230))
    for i, (im, cfg) in enumerate(zip(imgs, cands)):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + bar + r * (th + bar + pad)
        sheet.paste(Image.fromarray(im).resize((tw, th), Image.BILINEAR), (x, y))
        src = sampler.source_of(cfg.palette)
        draw.text((x + 2, y + th + 2), f"{i}: {cfg.palette[:24]} <{src[:4]}>", fill=(220, 220, 160))
        sub = (f"g{cfg.gamma:.2f}"
               f"{' ph%.2f' % cfg.phase if cfg.phase else ''}"
               f"{' n%d' % cfg.n_cycles if cfg.n_cycles > 1 else ''}"
               f"{' T%.2f' % cfg.transfer_gamma if cfg.transfer == 'grad' else ''}"
               f"{' rev' if cfg.reverse else ''}{' log' if cfg.log_premap == 'log' else ''}")
        draw.text((x + 2, y + th + 15), sub, fill=(160, 190, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def persist_query(qid, loc, query_type, cands, imgs, sampler):
    """Write 6 images, the durable record (query_type on every candidate), the contact
    sheet, then append the ledger LAST (completion marker)."""
    image_rels = []
    for ci, im in enumerate(imgs):
        rel = f"images/{qid}_{ci}.png"
        Image.fromarray(im).save(BATCH_DIR / rel)
        image_rels.append(rel)
    rec = aq.query_record(qid, loc, query_type, cands, sampler, image_rels)
    for cand in rec["candidates"]:                 # NEW provenance: query_type per candidate
        cand["query_type"] = query_type
    (BATCH_DIR / "records" / f"{qid}.json").write_text(json.dumps(rec, indent=1))
    contact_sheet(imgs, cands, qid, query_type, sampler, BATCH_DIR / f"{qid}.png")
    with (BATCH_DIR / "ledger.jsonl").open("a") as f:
        f.write(json.dumps({"qid": qid, "query_type": query_type, "family": loc.ref.kind,
                            "palettes": [c.palette for c in cands],
                            "sources": [sampler.source_of(c.palette) for c in cands]}) + "\n")


# ===========================================================================
# Per-query rng (independent stream keyed by qid).
# ===========================================================================

def per_query_rng(qid):
    h = hashlib.sha1(f"{qid}|{BATCH_ID}|{SEED}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


# ===========================================================================
# Recolor pool — a persistent 4-process pool (max-4 cap). render_candidate is GIL-bound
# (threads give no speedup), so recolor runs in worker PROCESSES. Each worker lazily
# loads the current location's field from disk (cheap: ~160ms load+stretch+profile) and
# caches it, so a unit's 6-12 recolors ship only (stem, config) — never the field array.
# ===========================================================================

_W = {}   # per-worker cache (process-local): {'lib': PaletteLibrary, 'fld': (stem, fld, prep, prof)}


def _winit():
    _W["lib"] = qs.load_pool_library()
    _W["fld"] = None


def _wtask(stem, cfg_json):
    lib = _W["lib"]
    cached = _W["fld"]
    if cached is None or cached[0] != stem:        # only the current unit's field is held
        fld = cm.load_field(str(aq.OUT_FIELDS / f"{stem}.bin"))
        prep = cm.stretch_field(fld)
        prof = cm.gradient_transfer_profile(fld, prep)
        _W["fld"] = (stem, fld, prep, prof)
        cached = _W["fld"]
    _, fld, prep, prof = cached
    cfg = cm.CandidateConfig.from_json(cfg_json)
    return cm.render_candidate(fld, cfg, lib, prep=prep, profile=prof)


def recolor_unit(pool, stem, cands):
    """Recolor a unit's candidates on the 4-process pool; images returned in `cands` order."""
    futs = [pool.submit(_wtask, stem, c.to_json()) for c in cands]
    return [f.result() for f in futs]


# ===========================================================================
# Plan build (shared by --estimate, --report-only, and the run).
# ===========================================================================

def build_full_plan(pool, sampler, lib):
    """Deterministic list of query jobs. Returns (units, jobs) where each job is
    {qid, query_type, unit_idx} and each unit has ref/loc/family/types."""
    units, locstats = select_locations(pool, SEED)
    units = assign_types(units, SEED)
    jobs = []
    qi = 0
    for ui, u in enumerate(units):
        for t in u["types"]:
            jobs.append({"qid": f"q{SEED:03d}_{qi:04d}", "query_type": t, "unit_idx": ui})
            qi += 1
    assert qi == N_QUERIES
    return units, jobs, locstats


# ===========================================================================
# Driver.
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate the prefv2_dramatic_v1 labeling batch.")
    ap.add_argument("--estimate", action="store_true", help="plan + runtime estimate, exit")
    ap.add_argument("--report-only", action="store_true", help="rebuild composition report from records/")
    ap.add_argument("--no-resume", action="store_true", help="regenerate completed queries")
    ap.add_argument("--cap-min", type=float, default=CAP_MIN_DEFAULT, help="soft wall cap (min)")
    ap.add_argument("--hard-min", type=float, default=HARD_MIN_DEFAULT, help="hard backstop (min)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    for sub in ("images", "records"):
        (BATCH_DIR / sub).mkdir(parents=True, exist_ok=True)

    pool = qs.LocationPool.from_corpus(scores=(3,), verbose=False)
    lib = qs.load_pool_library()
    sampler = qs.PaletteSampler(lib)
    pals = Palettes(sampler)

    if args.report_only:
        report_composition()
        return

    units, jobs, locstats = build_full_plan(pool, sampler, lib)
    # composition preview
    tc = Counter(j["query_type"] for j in jobs)
    fc = Counter(units[j["unit_idx"]]["family"] for j in jobs)
    n_double = sum(1 for u in units if u["n_queries"] == 2)
    print(f"[{BATCH_ID}] {pool.report()}")
    print(f"[plan] fresh available: {locstats['fresh_available']}  existing-in-corpus: {locstats['existing_in_corpus']}")
    print(f"[plan] {len(units)} distinct locations ({n_double} doubled, {len(units)-n_double} single); "
          f"{len(jobs)} queries")
    print(f"[plan] query types: {dict(tc)}")
    print(f"[plan] query family mix: {dict(fc)}")
    print(f"[plan] palettes: dramatic={len(pals.dramatic)} pool={len(pals.pool)}")

    # --- estimate: time one dump + one recolor, size the run ---
    ref0 = units[0]["ref"]
    fld0, ds0 = ensure_field_timeout(ref0)
    prep0 = cm.stretch_field(fld0)
    c0 = build_within_dramatic(ref0, per_query_rng("est"), pals, lib)[0]
    t0 = time.time(); cm.render_candidate(fld0, c0, lib, prep=prep0); recolor_s = time.time() - t0
    need_dump = sum(1 for u in units
                    if not (aq.OUT_FIELDS / f"{aq._field_key(u['ref'])}.bin").exists())
    dump_est = 8.5   # measured mean; ds0 is likely a cache hit
    total_cands = len(jobs) * CANDIDATES_PER_QUERY
    est_s = need_dump * dump_est + total_cands * recolor_s / RECOLOR_WORKERS * 1.25
    print(f"[est] {need_dump} dumps @~{dump_est:.1f}s + {total_cands} recolors @~{recolor_s*1000:.0f}ms"
          f"/{RECOLOR_WORKERS}w => ~{est_s/60:.1f} min")
    if args.estimate:
        return

    # --- run with wall discipline ---
    done = set()
    if not args.no_resume:
        done = {j["qid"] for j in jobs if query_is_done(j["qid"])}
        if done:
            print(f"[resume] {len(done)}/{len(jobs)} queries already done")

    jobs_by_unit = {}
    for j in jobs:
        jobs_by_unit.setdefault(j["unit_idx"], []).append(j)

    dump_obs = []
    t_wall = time.time()
    n_gen = 0
    stopped = None
    pool = ProcessPoolExecutor(max_workers=RECOLOR_WORKERS, initializer=_winit)
    try:
        for ui in sorted(jobs_by_unit):
            unit_jobs = [j for j in jobs_by_unit[ui] if j["qid"] not in done]
            if not unit_jobs:
                continue
            u = units[ui]
            elapsed = (time.time() - t_wall) / 60.0
            stem = aq._field_key(u["ref"])
            cached = (aq.OUT_FIELDS / f"{stem}.bin").exists()
            med_dump = float(np.median(dump_obs)) if dump_obs else dump_est
            est_unit = (0.0 if cached else med_dump) + len(unit_jobs) * CANDIDATES_PER_QUERY * recolor_s / RECOLOR_WORKERS
            if elapsed + est_unit / 60.0 > args.cap_min:
                stopped = f"soft cap {args.cap_min}min (elapsed {elapsed:.1f}, est_unit {est_unit:.0f}s)"
                break
            if elapsed > args.hard_min:
                stopped = f"hard backstop {args.hard_min}min"
                break

            try:
                _, dsec = ensure_field_timeout(u["ref"])   # dump (rayon-multicore) if uncached
            except subprocess.TimeoutExpired:
                print(f"[warn] unit {ui} dump TIMEOUT ({DUMP_TIMEOUT}s) — skipping {stem}", flush=True)
                continue
            if dsec > 0:
                dump_obs.append(dsec)

            for j in unit_jobs:
                qid, qtype = j["qid"], j["query_type"]
                rng = per_query_rng(qid)
                cands = BUILDERS[qtype](u["ref"], rng, pals, lib)
                imgs = recolor_unit(pool, stem, cands)
                persist_query(qid, u["loc"], qtype, cands, imgs, sampler)
                n_gen += 1

            el = (time.time() - t_wall) / 60.0
            rate = el / max(1, n_gen)
            remaining = len([j for j in jobs if j["qid"] not in done]) - n_gen
            print(f"[gen] unit {ui} ({u['family']}, {'cached' if cached else f'{dsec:.1f}s dump'}) "
                  f"{len(unit_jobs)}q -> {n_gen} total  {el:.1f}min  eta ~{rate*remaining:.0f}min", flush=True)
    finally:
        pool.shutdown(wait=True)

    wall = (time.time() - t_wall) / 60.0
    total_done = len(done) + n_gen
    print(f"\n[done] generated {n_gen} this run; {total_done}/{N_QUERIES} total  ({wall:.1f} min)")
    if stopped:
        print(f"[STOPPED EARLY] {stopped} — re-run to resume ({N_QUERIES - total_done} queries left)")

    write_batch_meta(units, jobs, locstats, recolor_s, dump_obs)
    report_composition()
    print(f"\n[label] launch:  uv run python tools/queries/launch_query_label_server.py "
          f"--batch {BATCH_ID}")


# ===========================================================================
# Batch meta + composition report.
# ===========================================================================

def write_batch_meta(units, jobs, locstats, recolor_s, dump_obs):
    tc = Counter(j["query_type"] for j in jobs)
    fc = Counter(units[j["unit_idx"]]["family"] for j in jobs)
    meta = {
        "batch_id": BATCH_ID,
        "purpose": ("Dramatic-inclusive within-location palette-preference labeling batch. "
                    "UNIONS with the existing pref-v2 corpus (coldstart_v2 + warmstart_v1); "
                    "commensurable render spec (1024x576 ss2 box, Recipe-2 render_candidate). "
                    "Puts dramatic + pool colorings on the SAME q3 locations so a retrained "
                    "pref-v2 can rank the families against each other, and adds a dramatic "
                    "within-palette param-ranking arm the corpus never had. pref-v2 is NOT "
                    "consulted in selection."),
        "invocation": "uv run python tools/queries/prefv2_dramatic_v1.py",
        "seed": SEED,
        "n_queries": N_QUERIES,
        "candidates_per_query": CANDIDATES_PER_QUERY,
        "query_type_counts_planned": QUERY_TYPE_COUNTS,
        "query_type_counts_realized": dict(tc),
        "family_mix_realized": dict(fc),
        "n_locations": len(units),
        "family_plan": {f: {"double": d, "single": s} for f, (d, s) in FAMILY_PLAN.items()},
        "candidate_ss": qs.CANDIDATE_SS,
        "eval": [qs.EVAL_WIDTH, qs.EVAL_HEIGHT],
        "filter": qs.CANDIDATE_FILTER,
        "param_draw": {
            "gamma": [GAMMA_LO, GAMMA_HI], "n_cycles": N_CYCLES,
            "transfer_gammas": TRANSFER_GAMMAS, "premaps": PREMAPS,
            "dramatic_phase0_p": DRAMATIC_PHASE0_P, "cross_dramatic_p": CROSS_DRAMATIC_P,
        },
        "location_stats": locstats,
        "excluded_batches": list(EXISTING_BATCHES),
        "timing": {"recolor_s": recolor_s,
                   "dump_median_s": float(np.median(dump_obs)) if dump_obs else None,
                   "n_dumps_this_run": len(dump_obs)},
    }
    (BATCH_DIR / "batch_meta.json").write_text(json.dumps(meta, indent=2))


def report_composition():
    """Actual composition from the written records/ (the honest realized numbers)."""
    recs = [json.loads(p.read_text()) for p in sorted((BATCH_DIR / "records").glob("q*.json"))]
    if not recs:
        print("[report] no records yet")
        return
    qtypes = Counter(r["query_type"] for r in recs)
    fams = Counter(r["location"]["family"] for r in recs)
    locs = set()
    src_by_type = {t: Counter() for t in QUERY_TYPE_COUNTS}
    phase0_dram = [0, 0]   # [phase0, total] over dramatic candidates
    for r in recs:
        L = r["location"]
        locs.add((L["family"], L["cx"], L["cy"], L["fw"], L.get("c_re"), L.get("c_im")))
        for c in r["candidates"]:
            s = c["palette_source"]
            bucket = "dramatic" if s == "dramatic" else "pool"
            src_by_type[r["query_type"]][bucket] += 1
            if s == "dramatic":
                phase0_dram[1] += 1
                if float(c["config"]["phase"]) == 0.0:
                    phase0_dram[0] += 1
    n_field = len({(l[0], l[1], l[2], l[3], l[4], l[5]) for l in locs})
    print("\n" + "=" * 72)
    print(f"{BATCH_ID} — REALIZED COMPOSITION ({len(recs)} queries)")
    print("=" * 72)
    print(f"  query types:        {dict(qtypes)}  (target {QUERY_TYPE_COUNTS})")
    print(f"  query family mix:   {dict(fams)}")
    print(f"  distinct locations: {len(locs)}   (== field dumps expected)")
    print(f"  candidate source by query type:")
    for t in QUERY_TYPE_COUNTS:
        print(f"      {t:16} {dict(src_by_type[t])}")
    if phase0_dram[1]:
        print(f"  dramatic phase=0 fraction: {phase0_dram[0]}/{phase0_dram[1]} "
              f"= {phase0_dram[0]/phase0_dram[1]:.3f}  (target ~0.33 + anchors)")
    print(f"\n  batch dir: {BATCH_DIR}")


if __name__ == "__main__":
    main()
