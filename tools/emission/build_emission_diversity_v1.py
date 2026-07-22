#!/usr/bin/env python
"""build_emission_diversity_v1.py — diversity-aware emission v1.

Deficit-driven colorize + resume-safe persistent pool + greedy release selection, built
against a steered-frontier run's run-scoped ledger (the first ledger whose rows decode as
current). The flow:

  1. INTAKE   — admitted locations (current-decode ∧ q3 ∧ guard ∧ distinct), each given a
                canonical morph-CLIP embedding and a within-type morph-cluster id.
  2. DEFICIT  — joint counts over (type × morph_cluster × palette_flavor × render_style)
                for the gated pool; a hand-editable target measure yields a per-cell
                deficit (cells.py).
  3. COLORIZE — for each location (type + cluster fixed), pick the (palette flavor, render
                style) that maximizes the joint deficit (softmax tie-break), pick the best
                palette in that flavor (pref ranker), render the wallpaper, and score it
                with the wallpaper head.
  4. POOL     — a global absolute floor (default 0.75, below the 0.90 production gate) admits
                the wallpaper to an append-only, resume-safe pool with full descriptor, head
                scores, realized palette statistics, and provenance (pool.py).
  5. SELECT   — greedy max-marginal-gain selection of N from the gated pool (select.py):
                niche-relative quality × coverage gain under a per-axis similarity kernel.

Admissibility is routed through `corpus_common.is_current_decoded` — a v6/v5/unstamped row
is never consumed. See prompts/build_emission_diversity_v1.md.

  # smoke: build to ≥3×N gated, select N=12, write report + sheets:
  uv run python tools/emission/build_emission_diversity_v1.py \
      --ledger data/discovery/steered_run2/outcome_ledger.jsonl --release-n 12
  uv run python tools/emission/build_emission_diversity_v1.py --resume ...   # after a kill
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus", ROOT / "tools" / "mining",
          ROOT / "tools" / "wallpaper"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import descriptor as D       # noqa: E402
from tools.emission import cells as C            # noqa: E402
from tools.emission import selection as SEL       # noqa: E402
from tools.emission.pool import Pool             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- geometry -------------------------------------------------------------- #
POOL_W, POOL_H, POOL_SS, POOL_FILT = 1280, 720, 2, "lanczos3"     # head-scoring / pool render
REL_W, REL_H, REL_SS, REL_FILT = 2560, 1440, 4, "lanczos3"        # release full-res (wallpaper canon)
JPG_Q = 95

DEFAULT_FLOOR = 0.75          # wallpaper-head POOL floor (permissive; below the 0.90 gate)
DEFAULT_MINING_FLOOR = 0.25   # mining-head POOL floor (permissive; below the 0.50 gate)
# Per-head RELEASE floors — distinct from the pool floors above. Pool admission stays
# permissive (weak wallpapers remain inventory); SELECTION only draws release candidates
# above the head's release floor. Defaults = each head's production gate. See the emission
# v1 finding (a 0.26 mining tile shipped because the permissive pool floor was the only bar).
DEFAULT_RELEASE_FLOOR = 0.90          # smooth → wallpaper head production gate
DEFAULT_MINING_RELEASE_FLOOR = 0.50   # strange → mining head production gate
DEFAULT_TARGET_MEASURE = ROOT / "data" / "emission" / "target_measure.json"
DEFAULT_STRANGE_FRAC = 0.5    # target strange share of the release (render-mode split)
# Within the strange (mining-head) pass, floor the coverage kernel for same-render-style
# pairs at this value so greedy spreads across the promoted modes before doubling up on one.
STRANGE_STYLE_WEIGHT = 0.5


# --------------------------------------------------------------------------- #
# Render-style axis + wallpaper render (reuse deploy_tail's roster + render dispatch).
# --------------------------------------------------------------------------- #
def _deploy_tail():
    from tools.mining import deploy_tail as dt
    return dt


def render_styles(dt) -> list:
    """smooth (base carrier) + the registry-promoted strange modes (deploy_tail ROSTER)."""
    return ["smooth"] + [m for (m, _kind) in dt.ROSTER]


def _roster_kind(dt, style: str):
    for (m, kind) in dt.ROSTER:
        if m == style:
            return kind
    raise KeyError(style)


def render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt):
    """Smooth base carrier: dump the plain smooth field, apply the palette via the colormap
    tail (no --coloring spec). Mirrors deploy_tail.render_pure minus the field spec."""
    dt.FIELD_TMP.mkdir(parents=True, exist_ok=True)
    binp = dt.FIELD_TMP / f"{dt._field_stem(loc, 'smooth', w, h, ss)}.bin"
    try:
        dt._run([str(dt.EXE), "render-one"] + dt._locflags(loc) + [
            "--width", str(w), "--height", str(h), "--supersample", str(ss),
            "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        ptype = dt.lib().palette_type(palette)
        phase = cp["phase"] if ptype == "cyclic" else 0.0
        ncyc = cp["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=palette, location=fld.location, eval_width=ow,
                                 eval_height=oh, reverse=cp["reverse"], log_premap=cp["log_premap"],
                                 gamma=cp["gamma"], phase=phase, n_cycles=ncyc,
                                 transfer=cp["transfer"], transfer_gamma=cp["transfer_gamma"],
                                 filter=filt)
        prep = cm.stretch_field(fld)
        img = cm.render_candidate(fld, cfg, dt.lib(), prep=prep)
        dt._save(img, out_path)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)


def render_wallpaper(dt, cm, loc, style, palette, out_path, w, h, ss, filt):
    cp = dt._color_params({})       # canonical inherited coloring (transfer=pct, γ1, no reverse)
    if style == "smooth":
        render_smooth(dt, cm, loc, palette, cp, out_path, w, h, ss, filt)
    else:
        dt.render_candidate(loc, style, _roster_kind(dt, style), palette, cp,
                            out_path, w, h, ss, filt)


# --------------------------------------------------------------------------- #
# Realized palette statistics (hue/chroma histogram of the ACTUAL render).
# --------------------------------------------------------------------------- #
HUE_BINS, CHROMA_BINS = 12, 8
BLACK_V = 0.06


def realized_palette_stats(jpg_path: Path) -> dict:
    im = np.asarray(Image.open(jpg_path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = im.max(axis=2)
    mn = im.min(axis=2)
    chroma = mx - mn
    v = mx
    # hue in [0,1)
    hue = np.zeros_like(mx)
    nz = chroma > 1e-6
    with np.errstate(invalid="ignore"):
        rc = np.where(mx == r, (g - b) / np.where(chroma == 0, 1, chroma), 0)
        gc = np.where(mx == g, 2.0 + (b - r) / np.where(chroma == 0, 1, chroma), 0)
        bc = np.where(mx == b, 4.0 + (r - g) / np.where(chroma == 0, 1, chroma), 0)
    h6 = np.where(mx == r, rc, np.where(mx == g, gc, bc))
    hue = (h6 / 6.0) % 1.0
    hue = np.where(nz, hue, 0.0)
    nonblack = v >= BLACK_V
    black_fraction = float(1.0 - nonblack.mean())
    mask = nonblack & nz
    if mask.sum() > 0:
        hue_hist, _ = np.histogram(hue[mask], bins=HUE_BINS, range=(0, 1),
                                   weights=chroma[mask])
        hh = hue_hist / (hue_hist.sum() + 1e-9)
        chroma_hist, _ = np.histogram(chroma[nonblack], bins=CHROMA_BINS, range=(0, 1))
        ch = chroma_hist / (chroma_hist.sum() + 1e-9)
        mean_chroma = float(chroma[nonblack].mean())
    else:
        hh = np.zeros(HUE_BINS)
        ch = np.zeros(CHROMA_BINS)
        mean_chroma = 0.0
    return {
        "hue_hist": [round(float(x), 5) for x in hh],
        "chroma_hist": [round(float(x), 5) for x in ch],
        "mean_chroma": round(mean_chroma, 5),
        "black_fraction": round(black_fraction, 5),
    }


# --------------------------------------------------------------------------- #
# Palette ranker (pref-v3-gvo best-in-flavor; deterministic fallback).
# --------------------------------------------------------------------------- #
class PaletteRanker:
    """Best concrete palette IN a flavor for a location, by the deployed pref-v3-gvo head
    (conditioned_colorize.Scorer) scored on the location's cached 640×360 smooth field. If
    the pref stack cannot load, falls back to a deterministic representative (first pool
    palette in the flavor) so the pipeline still runs — the head floor still gates quality."""

    def __init__(self, dt, cell_to_names: dict, lib):
        self.dt = dt
        self.lib = lib
        self.cell_to_names = cell_to_names
        self.cache: dict = {}
        self.pref = None
        self.canonical_config = None
        try:
            from tools.studies import conditioned_colorize as cond
            self.pref = cond.Scorer()
            self.canonical_config = cond.canonical_config
            self.mode = f"pref:{self.pref.name}"
        except Exception as e:                       # noqa: BLE001
            print(f"[ranker] pref scorer unavailable ({e!r}); deterministic fallback", flush=True)
            self.mode = "deterministic"

    # cap the per-flavor candidate set so one colorize's pref pass stays cheap (a flavor
    # holds up to ~90 pool palettes; 32 is ample to pick a good one and bounds cost).
    MAX_PALETTES = 32

    def _members(self, flavor: str) -> list:
        members = [p for p in self.cell_to_names.get(flavor, []) if p in self.lib.colormaps]
        return members[:self.MAX_PALETTES]

    def best(self, loc_id: str, flavor: str, field_bin: str, field_json: str):
        key = (loc_id, flavor)
        if key in self.cache:
            return self.cache[key]
        members = self._members(flavor)
        if not members:
            self.cache[key] = (None, None)
            return None, None
        if self.pref is None:
            res = (members[0], None)
            self.cache[key] = res
            return res
        from tools import colormap as cm
        field = cm.load_field(field_bin, field_json)
        prep = cm.stretch_field(field)
        cfield = cm.coarse_field(prep)
        imgs = [cm.render_candidate_coarse(cfield, self.canonical_config(field, pn), self.lib)
                for pn in members]
        scores = self.pref.score(imgs)
        i = int(np.argmax(scores))
        res = (members[i], float(scores[i]))
        self.cache[key] = res
        return res


# --------------------------------------------------------------------------- #
# Head scoring — per render style.
#
# The prompt specifies "the wallpaper head" (v3, 0.90 production gate). That head was
# trained on SMOOTH wallpapers only and scores strange fields (tia/stripe/composite)
# ~0, which would collapse the render-style descriptor axis to smooth in the gated pool.
# The repo already gates the two render classes with two heads: the wallpaper head for
# smooth, the MINING head (render_mode_head/v1, 0.50 gate) for the promoted strange
# modes (deploy_tail). We therefore route each render style to its own head and apply a
# permissive floor below THAT head's production gate. Quality is only ever compared
# within a niche (which pins the style, hence the head), so the two heads never mix in a
# single comparison. This is the one place §4 is extended beyond the literal "wallpaper
# head"; it is flagged in the report.
# --------------------------------------------------------------------------- #
WALLPAPER_STYLES = {"smooth"}


def head_for_style(style: str) -> str:
    return "wallpaper" if style in WALLPAPER_STYLES else "mining"


class Heads:
    def __init__(self):
        import torch
        from tools.wallpaper import emit_v1
        from tools.mining.mining_gate import MiningScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wp_score, _cfg = emit_v1.load_v2_scorer(device)
        self.wp_gate = emit_v1.GATE_THRESHOLD                 # 0.90
        self.mining = MiningScorer()
        self.mining_gate = self.mining.threshold             # 0.50

    def score(self, style: str, jpg_path: Path) -> dict:
        head = head_for_style(style)
        if head == "wallpaper":
            _cond, marg, ssum = self.wp_score([str(jpg_path)])
            return {"head": "wallpaper", "gate": self.wp_gate, "p_ge2": float(marg[0, 0]),
                    "p_ge3": float(marg[0, 1]), "ssum": float(ssum[0])}
        ms = self.mining.score_paths([str(jpg_path)])[0]
        return {"head": "mining", "gate": self.mining_gate, "p_ge2": float(ms.p_ge2),
                "p_ge3": float(ms.p_ge3), "ssum": float(ms.score)}


# --------------------------------------------------------------------------- #
# The driver.
# --------------------------------------------------------------------------- #
class EmissionDiversity:
    def __init__(self, args):
        self.args = args
        # Multi-ledger intake: a release can draw on more than one run-scoped ledger
        # (e.g. steered_run2 + the dive) — every row is still admitted independently and
        # carries its own source ledger in provenance. `self.ledger` stays the first for
        # legacy single-ledger call sites / short report strings.
        ledger_args = args.ledger if isinstance(args.ledger, (list, tuple)) else [args.ledger]
        self.ledgers = [Path(x).resolve() for x in ledger_args]
        self.ledger = self.ledgers[0]
        self.out = Path(args.out).resolve()
        self.report_path = Path(args.report).resolve() if args.report else \
            (ROOT / "out" / "emission_v1_report.md")
        self.renders = self.out / "renders"
        self.release_dir = self.out / "release"
        self.field_cache = self.out / "fields"
        self.embs_path = self.out / "morph_embs.npz"
        self.ranker_feats_path = self.out / "ranker_feats.npz"
        self.ranker_tiles = self.out / "ranker_tiles"
        self.intake_path = self.out / "intake.json"
        self.colorize_log = self.out / "colorize_log.jsonl"
        self.floor = float(args.floor)                 # wallpaper-head POOL floor (smooth)
        self.mining_floor = float(args.mining_floor)   # mining-head POOL floor (strange styles)
        self.release_floor = float(args.release_floor)              # wallpaper-head RELEASE floor
        self.mining_release_floor = float(args.mining_release_floor)  # mining-head RELEASE floor
        self.intake_floor = None if args.intake_floor is None else float(args.intake_floor)
        self.release_n = int(args.release_n)
        self.strange_frac = float(args.strange_frac)   # target strange share of the release
        # release render geometry — default = wallpaper canon (REL_W/H/SS/FILT); overridable
        # for a fast judge pass (e.g. 1024×576 ss2). Defaults keep batch reproducibility.
        self.rel_w = int(getattr(args, "release_w", None) or REL_W)
        self.rel_h = int(getattr(args, "release_h", None) or REL_H)
        self.rel_ss = int(getattr(args, "release_ss", None) or REL_SS)
        self.rel_filt = getattr(args, "release_filt", None) or REL_FILT
        # target counts RELEASE-ELIGIBLE pool rows (above the per-head release floor), so the
        # colorize builds a 3×N surplus of genuinely release-grade candidates, not merely
        # pool-admitted ones (§4 "3× post-floor surplus").
        self.target_gated = args.target_gated or (3 * self.release_n)
        self.max_attempts = int(args.max_attempts)
        self.time_budget_s = float(args.time_budget_min) * 60.0
        # cover-all: colorize every admitted location exactly once, then stop. Explicit
        # one-pass semantics — bypasses the surplus-building target/attempt/time cutoffs
        # (which invite the round-robin double-dip when hand-encoded as huge --target-gated).
        self.cover_all = bool(getattr(args, "cover_all", False))
        self.seed = int(args.seed)
        for d in (self.out, self.renders):
            d.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(self.seed)
        self.pool = Pool(self.out)
        self.ranker_score: dict = {}     # location_id -> pref_loc_v0 rank score
        self.ranker_pct: dict = {}       # location_id -> percentile within the intake set
        self.ranker_mode = "unavailable"

    # ---- intake ---------------------------------------------------------- #
    @staticmethod
    def _loc_key(r) -> tuple:
        """The location identity of a ledger row (what the id is supposed to name)."""
        return (str(r.get("outcome_cx")), str(r.get("outcome_cy")), str(r.get("outcome_fw")),
                str(r.get("julia_c_re")), str(r.get("julia_c_im")))

    def _load_all_admitted(self) -> list:
        """Admitted rows across every ledger, each tagged with its source; dedup by id.

        A duplicate id whose row resolves to the SAME location is a legitimate cross-ledger
        overlap and is dropped (first-ledger wins). A duplicate id whose row resolves to a
        DIFFERENT location is a run-scoped-id COLLISION — `st_<fam>_<arm>_<seq>` ids are
        reused across independent campaigns for distinct wallpapers, so a silent dedup would
        drop a genuinely distinct location. That case RAISES: disambiguate the colliding
        ledgers with an id prefix (see `stage_first_release.py`'s `c1__` scheme) before
        unioning them."""
        seen: dict = {}
        rows = []
        for lg in self.ledgers:
            try:
                label = str(lg.relative_to(ROOT))
            except ValueError:
                label = str(lg)                  # ledger outside the repo (e.g. a test tmp dir)
            for r in D.load_admitted(lg):
                rid = r["id"]
                loc = self._loc_key(r)
                if rid in seen:
                    if seen[rid][0] != loc:
                        raise SystemExit(
                            f"[intake] run-scoped id COLLISION: id {rid!r} names different "
                            f"locations across ledgers ({seen[rid][1]} @ {seen[rid][0]} vs "
                            f"{label} @ {loc}). A union-by-id would silently drop a distinct "
                            f"wallpaper — prefix the colliding ledger's ids (cf. "
                            f"stage_first_release.py c1__) before unioning.")
                    continue
                seen[rid] = (loc, label)
                r["_source_ledger"] = label
                rows.append(r)
        return rows

    def intake(self):
        """Admit locations, then either REUSE a pre-staged snapshot or embed+cluster fresh.

        Snapshot-reuse contract (the `if self.intake_path.exists() and self.embs_path.exists()`
        branch): a run resumes/starts against a snapshot pinned at first intake, laid down as
        two sibling files under `self.out`:

          * `intake.json` — `{cluster_tags: {id: "<family>#<k>"}, fields: {id: [bin, json]},
                              n_admitted: int}`. `cluster_tags` is the AUTHORITATIVE membership
                              set: only its ids are colorized (a still-growing frontier ledger's
                              newer admits are deferred, counted, and logged — rerun fresh to
                              fold them in). `fields` maps each id to its cached 640×360 ss2
                              smooth field (bin + json) so no re-render is needed.
          * `morph_embs.npz` — `descriptor._save_embs` format ({ids, emb}); the morph-CLIP
                               embeddings keyed by the same ids, loaded via `D.load_embs`.

        `stage_first_release.py` writes exactly these two files to pre-union committed intake
        passes without re-embedding; the fresh branch below writes the identical shapes."""
        rows = self._load_all_admitted()
        if not rows:
            srcs = ", ".join(str(l) for l in self.ledgers)
            raise SystemExit(f"no admitted (current-decode ∧ q3 ∧ guard ∧ distinct) rows in {srcs}")
        if self.intake_path.exists() and self.embs_path.exists():
            meta = json.loads(self.intake_path.read_text(encoding="utf-8"))
            embs = D.load_embs(self.embs_path)
            fields = {k: tuple(v) for k, v in meta["fields"].items()}
            tags = meta["cluster_tags"]
            # Snapshot semantics: a resume works against the locations embedded at first
            # intake. The run-scoped ledger may keep growing (a live frontier appends), but
            # those newly-admitted locations are NOT in the cached embeddings/tags/fields —
            # restrict to the snapshot and log how many fresh admits are being deferred.
            snap = set(tags)
            n_new = sum(1 for r in rows if r["id"] not in snap)
            rows = [r for r in rows if r["id"] in snap]
            print(f"[intake] reused {len(rows)} admitted locations (snapshot), "
                  f"{len(set(tags.values()))} morph clusters"
                  + (f"; {n_new} newer admits deferred (rerun fresh to include)" if n_new else ""),
                  flush=True)
        else:
            print(f"[intake] {len(rows)} admitted locations — embedding morph + clustering ...", flush=True)
            embs, fields = D.embed_locations(rows, self.field_cache, self.embs_path)
            tags = D.assign_morph_clusters(rows, embs)
            self.intake_path.write_text(json.dumps(
                {"cluster_tags": tags, "fields": {k: list(v) for k, v in fields.items()},
                 "n_admitted": len(rows)}), encoding="utf-8")
            print(f"[intake] {len(set(tags.values()))} morph clusters "
                  f"across {len(set(r['family'] for r in rows))} types", flush=True)
        self.rows = rows
        self.by_id = {r["id"]: r for r in rows}
        self.embs = embs
        self.fields = fields
        self.cluster_tags = tags
        self._score_intake_with_ranker()

    def _score_intake_with_ranker(self):
        """Score every admitted location with the pref_loc_v0 ranker (v7 + colored on the
        canonical render — cache hits for run2/dive via features.npz, else rendered once).
        The ranker ORDERS the colorize queue (order, don't filter — see pick_location); it
        never gates admission or steers discovery (scorer.py HARD SCOPE). With --intake-floor
        set, locations below that ranker percentile are dropped from the queue (opt-in)."""
        from tools.ranker.score_locations import LocationRanker, rank_percentiles, DEFAULT_FEATURES
        try:
            # Also feed the run's OWN persisted feature cache: on a resume it covers every
            # location embedded last pass, so scoring is a pure cache hit instead of
            # re-embedding all ~1.4k tiles. (Output-identical — features are deterministic;
            # missing/first-run cache is silently skipped by _load_cache.)
            lr = LocationRanker(feature_caches=(DEFAULT_FEATURES, self.ranker_feats_path))
            self.ranker_score = lr.score_rows(self.rows, self.ranker_tiles,
                                              persist_npz=self.ranker_feats_path)
            self.ranker_pct = rank_percentiles(self.ranker_score)
            self.ranker_mode = f"{lr.scorer.head}:{'+'.join(lr.sets)}"
            print(f"[intake] ranker scored {len(self.ranker_score)} locations "
                  f"({self.ranker_mode})", flush=True)
        except Exception as e:                               # noqa: BLE001
            print(f"[intake] ranker unavailable ({e!r}); colorize queue stays "
                  f"coverage-order (unranked)", flush=True)
            self.ranker_score, self.ranker_pct, self.ranker_mode = {}, {}, "unavailable"
        if self.intake_floor is not None and self.ranker_pct:
            keep = [r for r in self.rows if self.ranker_pct.get(r["id"], 1.0) >= self.intake_floor]
            dropped = len(self.rows) - len(keep)
            print(f"[intake] --intake-floor {self.intake_floor}: dropped {dropped} locations "
                  f"below ranker pct {self.intake_floor}; {len(keep)} remain", flush=True)
            self.rows = keep
            self.by_id = {r["id"]: r for r in keep}

    # ---- axes + deficit model ------------------------------------------- #
    def build_axes(self, dt, cell_to_names: dict, lib):
        # palette flavors: only cells with at least one pool-loadable palette are feasible.
        self.flavors = sorted(f for f, names in cell_to_names.items()
                              if any(p in lib.colormaps for p in names))
        self.styles = render_styles(dt)
        observed = sorted({(self.by_id[i]["family"], self.cluster_tags[i]) for i in self.by_id})
        cfg = {}
        if Path(self.args.target_measure).exists():
            cfg = json.loads(Path(self.args.target_measure).read_text(encoding="utf-8"))
        self.target = C.TargetMeasure.from_config(cfg)
        feasible = C.build_feasible_cells(observed, self.flavors, self.styles)
        self.model = C.DeficitModel(feasible, self.target)
        # rebuild deficit counts from the DURABLE pool log (resume safety).
        for r in self.pool.rows:
            cell = tuple(r["cell"])
            if cell in self.model.support or cell in self.model.capped:
                self.model.record_attempt(cell)
                if r.get("passed"):
                    self.model.record_fill(cell)
        print(f"[axes] {len(observed)} (type,cluster) × {len(self.flavors)} flavors × "
              f"{len(self.styles)} styles = {len(feasible)} feasible cells "
              f"| resumed attempts={self.pool.n_attempts()} gated={len(self.pool.gated())}", flush=True)

    # ---- location pick (coverage round-robin, ranker-ordered within a round) --------- #
    def pick_location(self, exhausted: set):
        """Fewest-attempts-first preserves coverage (every location gets a colorize before
        any gets a second — diversity SUPPLY is untouched); within an equal-attempts round,
        the pref_loc_v0 ranker feeds higher-quality locations FIRST, so when the colorize
        budget runs out mid-round the good locations already spent it (order, don't filter)."""
        counts = self.pool.attempts_per_location()
        cand = [r for r in self.rows if r["id"] not in exhausted]
        if not cand:
            return None
        cand.sort(key=lambda r: (counts.get(r["id"], 0),
                                 -self.ranker_score.get(r["id"], float("-inf")), r["id"]))
        return cand[0]

    # ---- one colorize ---------------------------------------------------- #
    def floor_for(self, style: str) -> float:
        """POOL admission floor (permissive; weak wallpapers persist as inventory)."""
        return self.floor if head_for_style(style) == "wallpaper" else self.mining_floor

    def release_floor_for(self, style: str) -> float:
        """RELEASE eligibility floor (per head; distinct from and above the pool floor)."""
        return self.release_floor if head_for_style(style) == "wallpaper" \
            else self.mining_release_floor

    def release_eligible(self) -> list:
        """Gated pool rows whose head score clears their head's RELEASE floor."""
        return [r for r in self.pool.gated()
                if (r["p_ge3"] or 0.0) >= self.release_floor_for(r["render_style"])]

    def colorize(self, dt, cm, ranker, heads, row) -> dict | None:
        loc_id = row["id"]
        ftype = row["family"]
        cluster = self.cluster_tags[loc_id]
        choice = C.choose_option(self.model, ftype, cluster, self.flavors, self.styles, self.rng)
        if choice is None:
            return None                              # all cells for this (type,cluster) capped
        flavor, style, deficit, n_opts, _probs = choice
        fbin, fjson = self.fields[loc_id]
        palette, pref_fit = ranker.best(loc_id, flavor, fbin, fjson)
        if palette is None:
            return None
        emid = self.pool.next_id()
        jpg = self.renders / f"{emid}.jpg"
        loc = D.location_of(row)
        cell = (ftype, cluster, flavor, style)
        floor = self.floor_for(style)
        err = None
        head = None
        stats = None
        try:
            render_wallpaper(dt, cm, loc, style, palette, jpg, POOL_W, POOL_H, POOL_SS, POOL_FILT)
            head = heads.score(style, jpg)
            stats = realized_palette_stats(jpg)
        except Exception as e:                       # noqa: BLE001
            err = repr(e)[:300]
        passed = bool(head and head["p_ge3"] >= floor)
        capped = self.model.record_attempt(cell)
        if passed:
            self.model.record_fill(cell)
        rec = {
            "id": emid, "location_id": loc_id,
            "type": ftype, "morph_cluster": cluster,
            "palette_flavor": flavor, "render_style": style, "palette": palette,
            "cell": list(cell),
            "head": (head or {}).get("head"), "head_gate": (head or {}).get("gate"),
            "p_ge2": (head or {}).get("p_ge2"), "p_ge3": (head or {}).get("p_ge3"),
            "score": (head or {}).get("ssum"),
            "floor": floor, "passed": passed, "error": err,
            "realized_palette": stats,
            "render": {"w": POOL_W, "h": POOL_H, "ss": POOL_SS},
            "jpg": str(jpg.relative_to(ROOT)) if jpg.exists() else None,
            "pref_fit": pref_fit, "ranker": ranker.mode,
            "provenance": {
                "source_ledger": row.get("_source_ledger", str(self.ledger.relative_to(ROOT))),
                "ranker_score": self.ranker_score.get(loc_id),
                "ranker_pct": self.ranker_pct.get(loc_id),
                "source_run": row.get("ts"), "node_id": row.get("node_id"),
                "root_id": row.get("root_id"), "branch": row.get("branch"),
                "reached_depth": row.get("reached_depth"), "p_good": row.get("p_good"),
                "scorer_version": row.get("scorer_version"),
            },
        }
        self.pool.append(rec)
        with open(self.colorize_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": emid, "location_id": loc_id, "type": ftype, "cluster": cluster,
                "chosen_flavor": flavor, "chosen_style": style, "palette": palette,
                "deficit": round(deficit, 6), "n_options": n_opts,
                "p_ge3": (head or {}).get("p_ge3"), "passed": passed,
                "capped_cell": bool(capped), "error": err,
            }) + "\n")
        return rec

    # ---- main colorize loop --------------------------------------------- #
    def run_colorize(self):
        dt = _deploy_tail()
        from tools import colormap as cm
        from tools.studies import conditioned_colorize as cond
        _, cell_to_names = cond.load_cell_map()
        lib = dt.lib()
        self.build_axes(dt, cell_to_names, lib)
        ranker = PaletteRanker(dt, cell_to_names, lib)
        heads = Heads()
        print(f"[colorize] pool-floors: wallpaper={self.floor} mining={self.mining_floor} · "
              f"release-floors: wallpaper={self.release_floor} (gate {heads.wp_gate}) "
              f"mining={self.mining_release_floor} (gate {heads.mining_gate}) · "
              f"target={self.target_gated} release-eligible · palette-ranker={ranker.mode} · "
              f"loc-ranker={self.ranker_mode}", flush=True)
        if self.cover_all:
            print(f"[colorize] --cover-all: one colorize per location over "
                  f"{len(self.rows)} admitted locations (target/attempt/time cutoffs bypassed)",
                  flush=True)
        t0 = time.time()
        exhausted: set = set()
        while True:
            n_rel = len(self.release_eligible())
            if not self.cover_all:
                if n_rel >= self.target_gated:
                    print(f"[colorize] reached target: {n_rel} release-eligible ≥ "
                          f"{self.target_gated}", flush=True)
                    break
                if self.pool.n_attempts() >= self.max_attempts:
                    print(f"[colorize] hit max attempts {self.max_attempts} "
                          f"(release-eligible={n_rel})", flush=True)
                    break
                if time.time() - t0 > self.time_budget_s:
                    print(f"[colorize] hit time budget (release-eligible={n_rel})", flush=True)
                    break
            row = self.pick_location(exhausted)
            if row is None:
                print(f"[colorize] all locations exhausted (release-eligible={n_rel})", flush=True)
                break
            # cover-all stops the instant pick_location wraps to a 2nd pass: it returns
            # fewest-attempts-first, so an already-attempted row means every location has one.
            if self.cover_all and self.pool.attempts_per_location().get(row["id"], 0) >= 1:
                print(f"[colorize] --cover-all: every location colorized once — stopping "
                      f"before a 2nd pass (release-eligible={n_rel})", flush=True)
                break
            rec = self.colorize(dt, cm, ranker, heads, row)
            if rec is None:
                exhausted.add(row["id"])
                continue
            self.pool.save_state({"seed": self.seed, "rng": self.rng.bit_generator.state,
                                  "n_attempts": self.pool.n_attempts()})
            n_gated = len(self.pool.gated())
            n_rel = len(self.release_eligible())
            print(f"  [{self.pool.n_attempts()}] {rec['id']} {rec['type']}/{rec['morph_cluster']} "
                  f"{rec['palette_flavor']}/{rec['render_style']} p_ge3="
                  f"{rec['p_ge3'] if rec['p_ge3'] is not None else 'ERR'} "
                  f"{'PASS' if rec['passed'] else 'floor-rej'} | gated={n_gated} "
                  f"release-eligible={n_rel}", flush=True)
        return len(self.release_eligible())

    def ranker_reach(self) -> dict:
        """How far down the ranker ordering the colorize had to reach. Ordering = admitted
        locations by ranker score desc; 'rank' is the 0-based position in that order. Reports
        the deepest rank among locations that (a) got any colorize attempt and (b) contributed
        a RELEASE-ELIGIBLE pool row — the practical measure of whether ranked intake
        concentrated budget on good locations."""
        if not self.ranker_score:
            return {}
        order = [r["id"] for r in sorted(self.rows,
                                         key=lambda r: -self.ranker_score.get(r["id"], float("-inf")))]
        rank_of = {i: k for k, i in enumerate(order)}
        attempted = {r["location_id"] for r in self.pool.rows}
        rel_locs = {r["location_id"] for r in self.release_eligible()}
        att_ranks = [rank_of[i] for i in attempted if i in rank_of]
        rel_ranks = [rank_of[i] for i in rel_locs if i in rank_of]
        n = len(order)
        return {
            "n_locations": n,
            "n_attempted": len(att_ranks),
            "deepest_attempted_rank": max(att_ranks) + 1 if att_ranks else 0,
            "deepest_attempted_pct": (max(att_ranks) + 1) / n if att_ranks else 0.0,
            "n_release_locs": len(rel_ranks),
            "deepest_release_rank": max(rel_ranks) + 1 if rel_ranks else 0,
            "deepest_release_pct": (max(rel_ranks) + 1) / n if rel_ranks else 0.0,
        }

    # ---- release selection ---------------------------------------------- #
    def _release_entries(self, rows: list) -> list:
        """Pool rows → greedy_select entries, with the location's morph-CLIP embedding
        attached as a plain list (the coverage kernel's continuous-cos input)."""
        entries = [{
            "id": r["id"], "type": r["type"], "cluster": r["morph_cluster"],
            "flavor": r["palette_flavor"], "style": r["render_style"],
            "score": r["p_ge3"], "emb": self.embs.get(r["location_id"], None),
            "_rec": r,
        } for r in rows]
        for e in entries:
            emb = e["emb"]
            e["emb"] = emb.tolist() if emb is not None else None
        return entries

    def select_release(self):
        """Head-split release selection — the two heads are NEVER compared in one step.

        Smooth slots are filled from the wallpaper head (rel ≥ release_floor), strange slots
        from the mining head (rel ≥ mining_release_floor), by two DISJOINT within-head greedy
        passes. Slot budget: strange_slots = round(N·strange_frac), smooth = N − strange.
        Each pass draws ONLY above ITS head's release floor and tie-breaks on its own
        (within-head, commensurable) `p_ge3`; the previous single cross-head greedy compared
        the wallpaper and mining heads' absolute scores on incommensurable scales and shut
        strange out entirely (82 eligible → 0 slots).

        The strange pass runs with a `style_weight` coverage floor so it spreads across the
        promoted modes rather than filling with one. Coverage in both passes is continuous
        morph-CLIP cos (no categorical gate), so a second near-identical look is discounted
        across cells.

        Honest short-fill: if a head can't fill its quota above its floor, ship fewer of that
        head — never dip below a floor, never pad, never backfill from the other head. The
        realized split is reported."""
        eligible = self.release_eligible()
        smooth = [r for r in eligible if head_for_style(r["render_style"]) == "wallpaper"]
        strange = [r for r in eligible if head_for_style(r["render_style"]) == "mining"]
        strange_slots = int(round(self.release_n * self.strange_frac))
        smooth_slots = self.release_n - strange_slots

        # disjoint within-head passes — heads never enter the same greedy comparison.
        sm_sel, sm_log = SEL.greedy_select(self._release_entries(smooth), smooth_slots)
        st_sel, st_log = SEL.greedy_select(self._release_entries(strange), strange_slots,
                                           style_weight=STRANGE_STYLE_WEIGHT)
        selected = sm_sel + st_sel
        log = sm_log + st_log

        self.release_split = {
            "strange_frac_target": self.strange_frac,
            "smooth_slots": smooth_slots, "strange_slots": strange_slots,
            "smooth_eligible": len(smooth), "strange_eligible": len(strange),
            "smooth_selected": len(sm_sel), "strange_selected": len(st_sel),
            "strange_frac_realized": (len(st_sel) / len(selected)) if selected else 0.0,
            "strange_modes": dict(Counter(e["_rec"]["render_style"] for e in st_sel)),
            "style_weight": STRANGE_STYLE_WEIGHT,
        }
        self.release_short_fill = {
            "requested": self.release_n, "eligible": len(eligible), "selected": len(selected),
            "short_by": max(0, self.release_n - len(selected)),
            "smooth_short_by": max(0, smooth_slots - len(sm_sel)),
            "strange_short_by": max(0, strange_slots - len(st_sel)),
        }
        if len(selected) < self.release_n:
            print(f"[select] SHORT-FILL {len(selected)}/{self.release_n}: smooth "
                  f"{len(sm_sel)}/{smooth_slots} (elig {len(smooth)}, ≥{self.release_floor}) + "
                  f"strange {len(st_sel)}/{strange_slots} (elig {len(strange)}, "
                  f"≥{self.mining_release_floor}). Shipping fewer rather than dipping below a "
                  f"floor (no cross-head backfill).", flush=True)
        print(f"[select] head-split: smooth {len(sm_sel)} (wallpaper head) + strange "
              f"{len(st_sel)} (mining head) = {len(selected)}; realized strange frac "
              f"{self.release_split['strange_frac_realized']:.2f} (target {self.strange_frac}); "
              f"strange modes {self.release_split['strange_modes']}", flush=True)
        return selected, log

    def render_release(self, selected, skip_render=False):
        # skip_render: reuse the full-res PNGs already on disk (report/sheet regen without
        # re-paying the ~30-min wallpaper-canon render pass).
        if skip_render:
            return [(e["_rec"]["id"], self.release_dir / f"{e['_rec']['id']}.png")
                    for e in selected if (self.release_dir / f"{e['_rec']['id']}.png").exists()]
        dt = _deploy_tail()
        from tools import colormap as cm
        self.release_dir.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for e in selected:
            r = e["_rec"]
            loc = D.location_of(self.by_id[r["location_id"]])
            png = self.release_dir / f"{r['id']}.png"
            # Per-file resume: selection is deterministic from the durable pool, so a relaunch
            # picks the same N — reuse any complete PNG already on disk (a reaper kill mid-pass
            # then only re-renders the missing tiles, never restarts all N). Validate the file
            # is a whole PNG so a truncated mid-write victim is re-rendered, not reused.
            if png.exists():
                try:
                    with Image.open(png) as _im:
                        _im.verify()
                    out_paths.append((r["id"], png))
                    continue
                except Exception:              # noqa: BLE001  truncated/corrupt → re-render
                    png.unlink(missing_ok=True)
            try:
                render_wallpaper(dt, cm, loc, r["render_style"], r["palette"], png,
                                 self.rel_w, self.rel_h, self.rel_ss, self.rel_filt)
                out_paths.append((r["id"], png))
            except Exception as ex:                  # noqa: BLE001
                print(f"[release] {r['id']} full-res render failed: {ex!r}", flush=True)
        return out_paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", nargs="+",
                    default=["data/discovery/steered_run2/outcome_ledger.jsonl"],
                    help="one or more run-scoped ledgers; admitted rows are unioned (dedup by id)")
    ap.add_argument("--out", default="out/emission_v1")
    ap.add_argument("--report", default=None,
                    help="report .md path (default out/emission_v1_report.md)")
    ap.add_argument("--release-n", type=int, default=12)
    ap.add_argument("--strange-frac", type=float, default=DEFAULT_STRANGE_FRAC,
                    help="target strange share of the release; strange_slots = round(N·frac), "
                         "smooth = N − strange. Heads are selected by DISJOINT within-head "
                         "greedy passes (never compared in one step).")
    ap.add_argument("--target-gated", type=int, default=0,
                    help="0 → 3×release-n RELEASE-ELIGIBLE rows (post-floor surplus)")
    ap.add_argument("--cover-all", action="store_true",
                    help="colorize every admitted location exactly once, then stop (explicit "
                         "one-pass; bypasses --target-gated/--max-attempts/--time-budget-min)")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help="wallpaper-head POOL floor for smooth (permissive; below the 0.90 gate)")
    ap.add_argument("--mining-floor", type=float, default=DEFAULT_MINING_FLOOR,
                    help="mining-head POOL floor for strange styles (permissive; below the 0.50 gate)")
    ap.add_argument("--release-floor", type=float, default=DEFAULT_RELEASE_FLOOR,
                    help="wallpaper-head RELEASE floor (default = 0.90 production gate)")
    ap.add_argument("--mining-release-floor", type=float, default=DEFAULT_MINING_RELEASE_FLOOR,
                    help="mining-head RELEASE floor (default = 0.50 production gate)")
    ap.add_argument("--intake-floor", type=float, default=None,
                    help="OPTIONAL ranker percentile [0,1]; drop admitted locations below it "
                         "from the colorize queue. Default OFF (deliberate — the ranker ORDERS "
                         "the queue, it does not filter diversity supply; this is a later knob).")
    ap.add_argument("--release-w", type=int, default=None,
                    help=f"release render width (default wallpaper canon {REL_W})")
    ap.add_argument("--release-h", type=int, default=None,
                    help=f"release render height (default wallpaper canon {REL_H})")
    ap.add_argument("--release-ss", type=int, default=None,
                    help=f"release supersample (default wallpaper canon {REL_SS})")
    ap.add_argument("--release-filt", default=None,
                    help=f"release downsample filter (default {REL_FILT})")
    ap.add_argument("--target-measure", default=str(DEFAULT_TARGET_MEASURE))
    ap.add_argument("--max-attempts", type=int, default=240, help="hard-kill backstop")
    ap.add_argument("--time-budget-min", type=float, default=45.0, help="hard-kill backstop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue (pool log is durable)")
    ap.add_argument("--select-only", action="store_true", help="skip colorize; select from pool")
    ap.add_argument("--no-release-render", action="store_true",
                    help="with --select-only: reuse existing release PNGs (regen report/sheets only)")
    args = ap.parse_args()

    from tools.emission import report as R
    eng = EmissionDiversity(args)
    eng.intake()
    if not args.select_only:
        eng.run_colorize()
    else:
        # build axes so the report has the deficit model populated from the durable log.
        dt = _deploy_tail()
        from tools.studies import conditioned_colorize as cond
        _, cell_to_names = cond.load_cell_map()
        eng.build_axes(dt, cell_to_names, dt.lib())
    selected, sel_log = eng.select_release()
    rel_paths = eng.render_release(selected, skip_render=args.no_release_render)
    R.write_report(eng, selected, sel_log, rel_paths)


if __name__ == "__main__":
    main()
