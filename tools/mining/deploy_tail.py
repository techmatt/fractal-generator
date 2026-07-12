"""Render-mode deploy tail — production curation pass over the emission corpus.

Runs ON DEMAND over the ACCUMULATED corpus of emitted wallpapers (the smooth winners
`emit_v1` has already produced). NOT wired inline into `emit_v1`'s loop: at ~15-24
emissions/run the 25% budget's per-mode floors round to 0 and never engage; they only
bite across the accumulated corpus (floor >= 1 needs N >~ 4*(n+2) = 32 at n=6). `emit_v1`
and the smooth emission path stay untouched — this pass only ADDS strange-mode alternates
alongside the smooth wallpapers, never overwriting them.

For each already-emitted location (smooth winner + its approved palette/params chosen
upstream by emit_v1), render a lean set of strange-mode candidate variants over the
INHERITED approved palette, score each with the LOCKED `mining_v1` gate
(`tools/mining/mining_gate.MiningScorer`, threshold 0.50 on marginal p_ge3), and keep
gate-passers as alternate wallpapers.

Incremental / idempotent state (the load-bearing production delta over the pilot):
  * The durable state is `alternates.jsonl` (the kept strange alternates). On each run
    the pass reads it; existing alternates are FIXED — counted toward the 25% budget B
    and toward their modes' floors (via tail_alloc's `existing=`), their locations
    locked out. Only locations WITHOUT an alternate are re-rendered/scored, and only the
    remaining shortfall (B - #existing) is allocated over them. Never churns or re-emits
    an existing alternate; deterministic.
  * Re-running over an UNCHANGED corpus is a no-op: budget is a ceiling that a correct
    prior run already filled (or exhausted supply), so the shortfall is 0 -> no new
    allocation; scoring crops are cached (skip-if-exists) and the full-res keeper render
    is skip-if-exists too -> no new renders, no re-emits. Verified each run (§checks).

Load-bearing (see prompts/deploy_tail_emit_wirein_prompt.md + the memories):
  * GATE IS LOCKED — we IMPORT `MiningScorer`; never retrain / re-threshold. Keepers
    carry `gate_stamp(p_ge3)`.
  * SCORE CHEAP, FULL-RES ONLY FOR KEEPERS — candidates render + score at the head's
    label geometry (1280x720 ss2 lanczos3 jpg, == render_mode dataset_v1). Only variants
    that PASS the gate AND survive the quota render at 2560x1440 ss4 (wallpaper canon).
  * MODE RENDER PATHS DIFFER (== tools/render_mode_pilot/render_scale_batch.py, the
    dataset_v1 recipe the head learned):
      - pure  (tia, stripe): render-one --dump-field <mode field> -> colormap.render_candidate
        with the FULL inherited param set. Bit-faithful. Field dump keyed with the
        render-mode token (loc_mod.field_mode_token) so it NEVER collides with the cached
        smooth field.
      - composite (C13, C17): render-one --coloring @spec --palette (Rust, grad-less).
      - direct  (direct_trap_multiply, direct_trap_screen): render-one --coloring @spec,
        palette-indifferent (ONE candidate per direct-trap mode, no palette axis).
        direct_trap_screen at its sweet spot (opacity 0.15 / threshold 0.08, at/under the
        source cap) + the dataset_v1 soft_knee@0.35 highlight rolloff.
  * normal_map OFF for all modes (none of the specs enable it; `shade:none` composites).

Keep / diversity allocation (tail_alloc.allocate_strange):
  * keep iff p_ge3 >= 0.50; AT MOST ONE strange alternate per location.
  * Strange budget B = round(0.25 * n_emitted) is a CEILING across the batch.
  * Keepers are SPREAD across modes for diversity (not abundance-biased): each mode
    gets a floor ~B/(n+2), the surplus (~2/(n+2)*B) lands on the abundant modes by
    quality. Starved modes degrade gracefully; total passers < B -> keep all, never
    pad. Because a location may pass in several modes, this is a BATCH-LEVEL
    assignment (each location fills at most one mode's slot). See tail_alloc.py.
    Existing (fixed) alternates seed the allocation via `existing=` (above).

    uv run python -u tools/mining/deploy_tail.py            # curate the current corpus
    uv run python -u tools/mining/deploy_tail.py --score-only   # gate/select only, no full-res
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))
sys.path.insert(0, str(REPO / "tools" / "queries"))

import colormap as cm                     # noqa: E402
import location as loc_mod                 # noqa: E402
import query_sampler as qs                 # noqa: E402
from tools.mining.mining_gate import MiningScorer, gate_stamp, MINING_GATE_THRESHOLD  # noqa: E402
from tools.mining.tail_alloc import allocate_strange, BUDGET_FRAC  # noqa: E402

EXE = str(REPO / "target/release/fractal-generator.exe")
POOL_CMAPS = str(REPO / "data/palettes/pool_colormaps.json")

EMIT_MANIFEST = REPO / "out/wallpaper/emit_v1/manifest.jsonl"
OUT_DIR = REPO / "out/mining/deploy_tail"
SCORE_CROPS = OUT_DIR / "scoring_crops"     # 1280x720 candidate jpgs (kept for eyeball+parity)
FIELD_TMP = OUT_DIR / "_fields"             # disposable field dumps (deploy-tail-owned, token'd)
KEEP_DIR = OUT_DIR / "keepers"              # full-res keeper pngs (alternates, alongside smooth)
SBS_DIR = OUT_DIR / "sidebyside"            # smooth-vs-kept comparisons
ALTERNATES = OUT_DIR / "alternates.jsonl"   # DURABLE state: the kept strange alternates

# scoring geometry == render_mode dataset_v1 (the head's label crops).
SC_W, SC_H, SC_SS, SC_FILT = 1280, 720, 2, "lanczos3"
JPG_Q = 95
# full-res keeper geometry == wallpaper canon (render-one locked default / emit_v1).
EMIT_W, EMIT_H, EMIT_SS, EMIT_FILT = 2560, 1440, 4, "lanczos3"

# strange budget as a fraction of emitted locations (diversity allocation, tail_alloc).
STRANGE_BUDGET_FRAC = BUDGET_FRAC            # 0.25

# ---- candidate roster (lean, weighted to high-yield) ---------------------- #
# kind: pure -> dump field + colormap tail; composite/direct -> Rust --coloring.
PURE_FIELD_SPEC = {
    "tia": {"field": "tia", "skip": 1},
    "stripe": {"field": "stripe", "stripe_density": 6},
}
SPEC_FILE = {
    "composite_c13_smooth_stripe": "composite_c13_smooth_stripe",
    "composite_c17_smooth_curvature": "composite_c17_smooth_curvature",
    "direct_trap_multiply": "direct_trap_multiply",
    "direct_trap_screen": "direct_trap_screen",
}
# highlight rolloff, gated to the screen-family blowout — matches dataset_v1 exactly.
ROLLOFF = {"direct_trap_screen": ("soft_knee", 0.35)}
# direct_trap sweet spot: opacity 0.15 / threshold 0.08, at/under the source cap.
MODE_PARAMS = {"direct_trap_screen": {"direct_threshold": 0.08, "direct_opacity": 0.15}}

# (mode, kind). Order = display order in the report.
ROSTER = [
    ("tia", "pure"),
    ("stripe", "pure"),
    ("composite_c13_smooth_stripe", "composite"),
    ("composite_c17_smooth_curvature", "composite"),
    ("direct_trap_multiply", "direct"),
    ("direct_trap_screen", "direct"),
]

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = qs.load_pool_library()
    return _LIB


def _locflags(loc):
    return loc_mod.render_one_flags(loc) + ["--cx", loc.cx, "--cy", loc.cy,
                                            "--fw", loc.fw, "--maxiter", str(loc.maxiter)]


def _run(cmd, retries=1):
    """render-one shell-out with one retry (renders occasionally fail transiently
    under resource contention; the recipe is deterministic so a retry recovers)."""
    for attempt in range(retries + 1):
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        if r.returncode == 0:
            return
        if attempt == retries:
            raise RuntimeError(" ".join(map(str, cmd[:3])) + " ... :\n" + r.stderr[-800:])


def _color_params(emit_params: dict) -> dict:
    """Inherited approved coloring, defaulted to what emit_v1 emitted (transfer=pct)."""
    return {
        "reverse": bool(emit_params.get("reverse", False)),
        "log_premap": emit_params.get("log_premap", "none"),
        "gamma": float(emit_params.get("gamma", 1.0)),
        "phase": float(emit_params.get("phase", 0.0)),
        "n_cycles": int(emit_params.get("n_cycles", 1)),
        "transfer": emit_params.get("transfer", "pct"),
        "transfer_gamma": float(emit_params.get("transfer_gamma", 0.0)),
    }


def _field_stem(loc, mode, w, h, ss):
    """Field-dump stem carrying the render-mode token (loc_mod.field_mode_token) so a
    strange pure-field dump can never collide with the cached SMOOTH field."""
    tok = loc_mod.field_mode_token(mode)
    suffix = f"|{tok}" if tok else ""
    h16 = hashlib.sha1(f"{loc.key()}|{w}x{h}ss{ss}|{loc.maxiter}{suffix}".encode()).hexdigest()[:16]
    return f"{loc.family}_{h16}_{w}x{h}ss{ss}__{mode}"


# --------------------------------------------------------------------------- #
# Render one candidate at (w,h,ss,filt) -> out_path (jpg for scoring, png for keeper).
# --------------------------------------------------------------------------- #
def render_pure(loc, mode, palette, cp, out_path, w, h, ss, filt):
    spec = dict(PURE_FIELD_SPEC[mode])
    FIELD_TMP.mkdir(parents=True, exist_ok=True)
    binp = FIELD_TMP / f"{_field_stem(loc, mode, w, h, ss)}.bin"
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(w), "--height", str(h),
             "--supersample", str(ss), "--coloring", json.dumps(spec), "--dump-field", str(binp)])
        fld = cm.load_field(str(binp))
        ow, oh = fld.out_size
        ptype = lib().palette_type(palette)
        phase = cp["phase"] if ptype == "cyclic" else 0.0
        ncyc = cp["n_cycles"] if ptype == "cyclic" else 1
        cfg = cm.CandidateConfig(palette=palette, location=fld.location, eval_width=ow,
            eval_height=oh, reverse=cp["reverse"], log_premap=cp["log_premap"],
            gamma=cp["gamma"], phase=phase, n_cycles=ncyc, transfer=cp["transfer"],
            transfer_gamma=cp["transfer_gamma"], filter=filt)
        prep = cm.stretch_field(fld)
        prof = cm.gradient_transfer_profile(fld, prep) if cp["transfer"] == "grad" else None
        img = cm.render_candidate(fld, cfg, lib(), prep=prep, profile=prof)
        _save(img, out_path)
    finally:
        binp.unlink(missing_ok=True)
        binp.with_suffix(".json").unlink(missing_ok=True)
    return {"transfer_dropped": False}


def render_rust(loc, mode, palette, cp, out_path, w, h, ss, filt):
    spec = json.loads((REPO / "specs" / f"{SPEC_FILE[mode]}.json").read_text())
    spec.pop("tier", None)
    spec.pop("note", None)
    spec.update(MODE_PARAMS.get(mode, {}))
    ptype = lib().palette_type(palette)
    spec["transform"] = "log" if cp["log_premap"] == "log" else "linear"
    spec["gamma"] = cp["gamma"]
    spec["reverse"] = cp["reverse"]
    if ptype == "cyclic":
        spec["palette_cycles"] = float(cp["n_cycles"])
        spec["palette_offset"] = float(cp["phase"])
    rolloff = ROLLOFF.get(mode, ("none", 1.0))
    if rolloff[0] != "none":
        spec["rolloff"] = rolloff[0]
        spec["rolloff_strength"] = rolloff[1]
    transfer_dropped = cp["transfer"] == "grad"
    FIELD_TMP.mkdir(parents=True, exist_ok=True)
    tmp_png = FIELD_TMP / f"{loc.family}_{mode}_{w}x{h}.png"
    try:
        _run([EXE, "render-one"] + _locflags(loc) + ["--width", str(w), "--height", str(h),
             "--supersample", str(ss), "--filter", filt, "--palette", palette,
             "--colormaps", POOL_CMAPS, "--coloring", json.dumps(spec), "--out", str(tmp_png)])
        with Image.open(tmp_png) as im:
            _save(np.asarray(im.convert("RGB")), out_path)
    finally:
        tmp_png.unlink(missing_ok=True)
    return {"transfer_dropped": transfer_dropped, "rolloff": rolloff, "spec": spec}


def _save(img_arr, out_path):
    im = Image.fromarray(np.asarray(img_arr))
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        im.save(out_path, quality=JPG_Q)
    else:
        im.save(out_path)


def render_candidate(loc, mode, kind, palette, cp, out_path, w, h, ss, filt):
    if kind == "pure":
        return render_pure(loc, mode, palette, cp, out_path, w, h, ss, filt)
    return render_rust(loc, mode, palette, cp, out_path, w, h, ss, filt)


# --------------------------------------------------------------------------- #
# Side-by-side smooth (emit_v1) vs kept strange alternate.
# --------------------------------------------------------------------------- #
def _font(sz):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def side_by_side(smooth_png: Path, keep_png: Path, out_png: Path, caption: str):
    TW = 1100
    def _load(p):
        with Image.open(p) as im:
            im = im.convert("RGB")
            return im.resize((TW, round(TW * im.height / im.width)), Image.LANCZOS)
    a, b = _load(smooth_png), _load(keep_png)
    th = max(a.height, b.height)
    pad, cap = 16, 46
    W = TW * 2 + pad * 3
    H = th + pad * 2 + cap
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    sheet.paste(a, (pad, pad + cap))
    sheet.paste(b, (pad * 2 + TW, pad + cap))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 12), caption, font=_font(22), fill=(235, 235, 235))
    d.text((pad, pad + cap - 2), "smooth (shipped)", font=_font(18), fill=(150, 200, 235))
    d.text((pad * 2 + TW, pad + cap - 2), "strange keeper", font=_font(18), fill=(235, 200, 150))
    sheet.save(out_png)


# --------------------------------------------------------------------------- #
# Durable incremental state — the kept strange alternates (alternates.jsonl).
# --------------------------------------------------------------------------- #
def load_alternates() -> dict:
    """loc_id -> alternate record for every already-emitted strange alternate.
    This IS the incremental state: these are FIXED (never re-rendered / re-allocated)."""
    if not ALTERNATES.exists():
        return {}
    out = {}
    for line in ALTERNATES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["loc_id"]] = r
    return out


def save_alternates(recs: dict):
    """Rewrite alternates.jsonl from loc_id -> record (sorted for a stable diff)."""
    ALTERNATES.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(recs[k]) for k in sorted(recs)]
    ALTERNATES.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sha1_file(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _snapshot(paths) -> dict:
    """path -> sha1 for the paths that exist (correctness before/after diff)."""
    return {str(p): _sha1_file(p) for p in paths if Path(p).exists()}


def _dir_snapshot(d: Path) -> dict:
    """name -> (size, sha1) over a directory's files (emit field-dir tamper check)."""
    if not d.exists():
        return {}
    return {f.name: (f.stat().st_size, _sha1_file(f)) for f in sorted(d.iterdir()) if f.is_file()}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Render-mode deploy tail — production curation pass over the emission corpus.")
    ap.add_argument("--manifest", type=Path, default=EMIT_MANIFEST)
    ap.add_argument("--score-only", action="store_true", help="gate/select only; skip full-res")
    ap.add_argument("--limit", type=int, default=0, help="only the first N emitted locations")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # 1. Enumerate the accumulated emission corpus (the smooth winners emit_v1 produced).
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    n_emit = len(rows)                                    # N = corpus size
    corpus_ids = {r["image_id"] for r in rows}
    modes = [m for m, _ in ROSTER]

    # 2. Read the strange alternates that already exist (the DURABLE incremental state).
    #    These are FIXED: never re-rendered / re-allocated / re-emitted. Drop any stale
    #    row whose location is no longer in the corpus (corpus can only grow in practice,
    #    but stay robust to a shrunk/re-pointed manifest).
    all_alts = load_alternates()
    existing_alts = {k: v for k, v in all_alts.items() if k in corpus_ids}
    existing_list = [{"loc_id": k, "mode": v["mode"]} for k, v in existing_alts.items()]

    # Correctness snapshots taken BEFORE any work: the smooth wallpapers and emit_v1's
    # own field dir must be byte-unchanged after the pass (§checks).
    smooth_paths = [REPO / r["png"].replace("\\", "/") for r in rows]
    smooth_before = _snapshot(smooth_paths)
    emit_field_dir = args.manifest.parent / "fields"
    emit_fields_before = _dir_snapshot(emit_field_dir)
    # Field-bin self-containment (static guarantees): the pass dumps its OWN strange
    # fields under a deploy-tail-owned dir, disjoint from emit_v1's, and every pure-mode
    # dump carries a non-empty render-mode token so its key can NEVER collide with the
    # cached smooth field. Assert both up front rather than trusting them.
    assert FIELD_TMP.resolve() != emit_field_dir.resolve(), FIELD_TMP
    assert emit_field_dir.resolve() not in FIELD_TMP.resolve().parents, FIELD_TMP
    for mode, kind in ROSTER:
        if kind == "pure":
            assert loc_mod.field_mode_token(mode) not in ("", None), \
                f"pure mode {mode!r} has empty field token — would collide with smooth"

    # 3. Candidate locations = corpus locations WITHOUT an existing alternate. Only these
    #    are (re-)rendered and scored; already-curated locations are left untouched.
    new_rows = [r for r in rows if r["image_id"] not in existing_alts]
    SCORE_CROPS.mkdir(parents=True, exist_ok=True)
    print(f"[tail] corpus N={n_emit} · {len(existing_alts)} existing alternate(s) fixed · "
          f"{len(new_rows)} not-yet-curated location(s) x {len(ROSTER)} modes "
          f"= {len(new_rows)*len(ROSTER)} candidates @ {SC_W}x{SC_H} ss{SC_SS}")

    cands = []   # dicts: emit_index, loc_id, mode, kind, crop path, palette, info
    t0 = time.time()
    for row in new_rows:
        loc = loc_mod.from_render_block(row["location"])
        palette = row["params"]["palette"]
        cp = _color_params(row["params"])
        for mode, kind in ROSTER:
            cid = f"{row['image_id']}__{mode}"
            crop = SCORE_CROPS / f"{cid}.jpg"
            tc = time.time()
            # transfer_dropped is deterministic (rust modes drop grad; inherited is pct
            # here so it is False) — reconstructable without re-rendering, so a crop that
            # already exists on disk is reused (resume / idempotent re-run / cheap backfill).
            info = {"transfer_dropped": (kind != "pure" and cp["transfer"] == "grad")}
            if not crop.exists():
                try:
                    info = render_candidate(loc, mode, kind, palette, cp, crop, SC_W, SC_H, SC_SS, SC_FILT)
                except Exception as exc:
                    print(f"  [ERR] {cid}: {str(exc)[:180]}")
                    continue
            cands.append({"emit_index": row["emit_index"], "loc_id": row["image_id"],
                          "cid": cid, "mode": mode, "kind": kind, "crop": crop,
                          "palette": palette, "row": row, "cp": cp, "info": info,
                          "secs": time.time() - tc})
    print(f"[tail] rendered {len(cands)} candidates in {time.time()-t0:.0f}s")

    # 4. Score the new candidates through the LOCKED gate.
    scorer = MiningScorer()
    print(f"[gate] {scorer.__class__.__name__} thr={scorer.threshold} on {scorer.device}")
    scores = scorer.score_paths([str(c["crop"]) for c in cands]) if cands else []
    for c, s in zip(cands, scores):
        c["p_ge3"], c["p_ge2"], c["ord"], c["passed"] = s.p_ge3, s.p_ge2, s.score, s.passed

    by_loc = {}
    for c in cands:
        by_loc.setdefault(c["loc_id"], []).append(c)

    # 5. Shortfall allocation: existing alternates are FIXED (count toward budget B and
    #    per-mode floors); allocate only B-#existing over the new passers, spread across
    #    modes for diversity. A location may pass in several modes -> batch-level (each
    #    location fills at most one mode's slot).
    passers = [c for c in cands if c.get("passed")]
    keepers, alloc = allocate_strange(passers, n_emit, modes, STRANGE_BUDGET_FRAC,
                                      existing=existing_list)
    keepers.sort(key=lambda c: -c["p_ge3"])   # render/report in quality order
    n_loc_passer = len({c["loc_id"] for c in passers})
    achieved = alloc["achieved"]               # corpus-wide (existing + new)
    thr_bite = 4 * (alloc["n_modes"] + 2)      # N at which floor >= 1 (round(0.25 N) >= n+2)
    floors_engaged = alloc["floor"] >= 1
    print(f"[keep] budget B={alloc['B']} (round({STRANGE_BUDGET_FRAC:.0%} x {n_emit})) "
          f"floor={alloc['floor']}/mode · floors {'ENGAGED' if floors_engaged else 'dormant'} "
          f"(bite at N>~{thr_bite}) · {len(existing_alts)} fixed + {len(keepers)} new "
          f"= {len(existing_alts)+len(keepers)} alternate(s) over {n_loc_passer} new passer-locs")
    for m, _ in ROSTER:
        supply = len({c['loc_id'] for c in passers if c['mode'] == m})
        n_ex = sum(1 for v in existing_alts.values() if v["mode"] == m)
        print(f"       {m:32} floor={alloc['floor']} achieved={achieved[m]} "
              f"(existing={n_ex}, new-supply={supply} distinct-loc passers)")

    # 6. Full-res render the NEW keepers alongside the smooth wallpaper (skip-if-exists so
    #    a re-run / adopted pilot keeper never re-renders), + side-by-side for eyeball.
    if not args.score_only:
        KEEP_DIR.mkdir(parents=True, exist_ok=True)
        SBS_DIR.mkdir(parents=True, exist_ok=True)
        for c in keepers:
            loc = loc_mod.from_render_block(c["row"]["location"])
            keep_png = KEEP_DIR / f"{c['cid']}_{EMIT_W}x{EMIT_H}.png"
            tk = time.time()
            skipped = keep_png.exists()
            if not skipped:
                render_candidate(loc, c["mode"], c["kind"], c["palette"], c["cp"], keep_png,
                                 EMIT_W, EMIT_H, EMIT_SS, EMIT_FILT)
            c["keep_png"] = str(keep_png.relative_to(REPO))
            smooth_png = REPO / c["row"]["png"].replace("\\", "/")
            sbs = SBS_DIR / f"{c['cid']}_sbs.png"
            cap_txt = (f"{c['loc_id']} · {c['mode']} · p_ge3={c['p_ge3']:.3f} "
                       f"(gate>={scorer.threshold}) · {c['palette'][:28]}")
            if smooth_png.exists():
                side_by_side(smooth_png, keep_png, sbs, cap_txt)
                c["sbs"] = str(sbs.relative_to(REPO))
            c["gate_stamp"] = gate_stamp(c["p_ge3"], scorer.threshold)
            tag = "skip (exists)" if skipped else f"{time.time()-tk:.0f}s"
            print(f"[emit] {c['cid']:34} p_ge3={c['p_ge3']:.3f}  [{tag}] -> {keep_png.name}")

        # 7. Persist the new keepers into the durable alternates state (existing FIXED
        #    rows preserved verbatim). Idempotent: a re-run reads these back as `existing`.
        for c in keepers:
            all_alts[c["loc_id"]] = {
                "loc_id": c["loc_id"], "mode": c["mode"], "kind": c["kind"],
                "family": c["row"]["location"].get("fractal_type"),
                "palette": c["palette"], "p_ge3": round(c["p_ge3"], 6),
                "gate_stamp": c["gate_stamp"], "curated_at_N": n_emit,
                "smooth_png": c["row"]["png"], "keeper_png": c.get("keep_png"),
                "sidebyside": c.get("sbs"),
            }
        save_alternates(all_alts)
        print(f"[state] alternates.jsonl now holds {len(all_alts)} alternate(s)")

    # 8. Correctness checks (ship with checks, not just reasoning).
    checks = run_checks(smooth_before, smooth_paths, emit_fields_before, emit_field_dir,
                        passers, keepers, existing_list, n_emit, modes, args)

    # 9. Parity: re-score 1-2 kept scoring crops through the standalone gate CLI
    #    (deploy path == measurement path across two independent entry points).
    parity = run_parity(keepers)

    # 10. Report.
    write_report(rows, cands, by_loc, keepers, existing_alts, alloc, n_loc_passer,
                 scorer, parity, checks, thr_bite, floors_engaged, args)


def run_checks(smooth_before, smooth_paths, emit_fields_before, emit_field_dir,
               passers, keepers, existing_list, n_emit, modes, args):
    """Ship the correctness guarantees as verified checks, not just reasoning.

      * additive/smooth untouched — every smooth wallpaper is byte-unchanged.
      * field-bin self-containment — emit_v1's field dir is byte-unchanged (the pass
        reached for no pre-existing smooth field bin) and the deploy-tail field dir is
        empty afterward (every strange dump was token'd and cleaned up).
      * idempotent — simulate the NEXT run's allocation given the just-persisted state:
        existing = current alternates + these new keepers, passers = the same passers
        minus the kept locations. A correct run leaves 0 shortfall -> the next run
        allocates nothing new (and its scoring crops are cached, full-res is
        skip-if-exists) -> a genuine no-op.
    """
    # -- smooth untouched.
    smooth_after = _snapshot(smooth_paths)
    smooth_changed = [p for p in smooth_before if smooth_before[p] != smooth_after.get(p)]
    smooth_ok = not smooth_changed

    # -- emit_v1 field dir untouched (no reach for a pre-existing smooth field bin).
    emit_fields_after = _dir_snapshot(emit_field_dir)
    emit_fields_ok = emit_fields_before == emit_fields_after

    # -- deploy-tail field dir empty afterward (all token'd strange dumps cleaned up).
    leftover = [f.name for f in FIELD_TMP.iterdir() if f.is_file()] if FIELD_TMP.exists() else []
    field_tmp_clean = not leftover

    # -- idempotency: next-run allocation over the persisted state is empty.
    kept_ids = {c["loc_id"] for c in keepers}
    existing_after = existing_list + [{"loc_id": c["loc_id"], "mode": c["mode"]} for c in keepers]
    passers_after = [{"loc_id": c["loc_id"], "mode": c["mode"], "p_ge3": c["p_ge3"]}
                     for c in passers if c["loc_id"] not in kept_ids]
    sel2, _ = allocate_strange(passers_after, n_emit, modes, STRANGE_BUDGET_FRAC,
                               existing=existing_after)
    idempotent = (len(sel2) == 0)

    checks = {
        "smooth_untouched": {"ok": smooth_ok, "n_checked": len(smooth_before),
                             "changed": smooth_changed},
        "emit_field_dir_untouched": {"ok": emit_fields_ok, "n_files": len(emit_fields_before)},
        "field_tmp_clean": {"ok": field_tmp_clean, "leftover": leftover},
        "idempotent": {"ok": idempotent, "next_run_new_allocations": len(sel2)},
    }
    all_ok = all(v["ok"] for v in checks.values())
    print(f"[check] smooth-untouched={smooth_ok} ({len(smooth_before)} pngs) · "
          f"emit-field-dir-untouched={emit_fields_ok} · field-tmp-clean={field_tmp_clean} · "
          f"idempotent={idempotent}  ->  {'ALL PASS' if all_ok else 'FAILED'}")
    if not all_ok:
        print(f"[check][WARN] failing checks: "
              f"{[k for k, v in checks.items() if not v['ok']]}")
    checks["all_ok"] = all_ok
    return checks


def run_parity(keepers):
    out = []
    for c in keepers[:2]:
        r = subprocess.run([sys.executable, str(REPO / "tools/mining/mining_gate.py"), str(c["crop"])],
                           cwd=str(REPO), capture_output=True, text=True)
        cli_p = None
        for line in r.stdout.splitlines():
            if "p_ge3=" in line:
                try:
                    cli_p = float(line.split("p_ge3=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        d = abs(cli_p - c["p_ge3"]) if cli_p is not None else None
        out.append({"cid": c["cid"], "inproc_p_ge3": c["p_ge3"], "cli_p_ge3": cli_p,
                    "delta": d, "verdict_agree": (cli_p is not None and (cli_p >= MINING_GATE_THRESHOLD) == c["passed"])})
        print(f"[parity] {c['cid']}: in-proc {c['p_ge3']:.6f} vs CLI {cli_p} "
              f"Δ={d if d is None else f'{d:.2e}'}")
    return out


def write_report(rows, cands, by_loc, keepers, existing_alts, alloc, n_loc_passer,
                 scorer, parity, checks, thr_bite, floors_engaged, args):
    keep_ids = {c["cid"] for c in keepers}
    n_total_alt = len(existing_alts) + len(keepers)
    degenerate = not floors_engaged
    # corpus-wide existing count per mode (existing alternates aren't in `cands`).
    ex_by_mode = {m: sum(1 for v in existing_alts.values() if v["mode"] == m) for m, _ in ROSTER}
    rep = {
        "gate": {"version": "mining_v1", "threshold": scorer.threshold},
        "curation_pass": {
            "score_only": bool(args.score_only),
            "n_existing_alternates": len(existing_alts),
            "n_new_keepers": len(keepers),
            "n_total_alternates": n_total_alt,
        },
        "allocation": {
            "budget_frac": alloc["budget_frac"], "n_emitted": len(rows),
            "budget_B": alloc["B"], "floor_per_mode": alloc["floor"],
            "n_modes": alloc["n_modes"],
            "floor_bite_threshold_N": thr_bite,       # floor >= 1 needs N >~ 4*(n+2)
            "floors_engaged": floors_engaged,
            "degenerate_small_N": degenerate,
            "n_new_locations_with_passer": n_loc_passer,
            "n_strange_kept_total": n_total_alt,
            "pct_kept": (n_total_alt / len(rows)) if rows else 0.0,
            "per_mode": [
                {"mode": m, "floor": alloc["floor"], "achieved": alloc["achieved"][m],
                 "existing": ex_by_mode[m],
                 "new_supply": len({c["loc_id"] for c in cands
                                    if c["mode"] == m and c.get("passed")}),
                 "degraded": alloc["achieved"][m] < alloc["floor"]}
                for m, _ in ROSTER],
        },
        "checks": checks,
        "parity": parity,
        "per_location": [],
    }
    for row in rows:
        lid = row["image_id"]
        entry = {"loc_id": lid, "family": row["location"].get("fractal_type"),
                 "palette": row["params"]["palette"], "smooth_png": row["png"],
                 "state": "new", "candidates": [], "kept": None}
        if lid in existing_alts:                       # already-curated (fixed) — not re-scored.
            ea = existing_alts[lid]
            entry["state"] = "existing_alternate"
            entry["kept"] = {"mode": ea["mode"], "kind": ea.get("kind"),
                             "p_ge3": ea.get("p_ge3"), "gate_stamp": ea.get("gate_stamp"),
                             "keeper_png": ea.get("keeper_png"),
                             "sidebyside": ea.get("sidebyside"), "fixed": True}
            rep["per_location"].append(entry)
            continue
        cs = sorted(by_loc.get(lid, []), key=lambda c: -c["p_ge3"])
        for c in cs:
            cd = {"mode": c["mode"], "kind": c["kind"], "p_ge3": round(c["p_ge3"], 4),
                  "p_ge2": round(c["p_ge2"], 4), "E_ord": round(c["ord"], 4),
                  "passed": c["passed"], "kept": c["cid"] in keep_ids,
                  "transfer_dropped": c["info"].get("transfer_dropped", False)}
            if c["cid"] in keep_ids:
                cd["gate_stamp"] = c.get("gate_stamp", gate_stamp(c["p_ge3"], scorer.threshold))
                cd["keeper_png"] = c.get("keep_png")
                cd["sidebyside"] = c.get("sbs")
                cd["fixed"] = False
                entry["kept"] = cd
            entry["candidates"].append(cd)
        rep["per_location"].append(entry)

    (OUT_DIR / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # markdown
    L = []
    L.append("# Render-mode deploy tail — production curation pass\n")
    q = rep["allocation"]
    ck = checks
    L.append(f"**Gate** `mining_v1` (LOCKED) · threshold `{scorer.threshold}` on marginal p_ge3\n")
    L.append(f"**Corpus** N=**{q['n_emitted']}** emitted · budget "
             f"B=round({q['budget_frac']:.0%}×{q['n_emitted']})=**{q['budget_B']}** · "
             f"floor=**{q['floor_per_mode']}**/mode (n={q['n_modes']}) · floor-bite at "
             f"**N≥~{q['floor_bite_threshold_N']}** → floors "
             f"**{'ENGAGED' if floors_engaged else 'DORMANT'}** at this N\n")
    L.append(f"**Alternates** {rep['curation_pass']['n_existing_alternates']} fixed + "
             f"{rep['curation_pass']['n_new_keepers']} new = **{n_total_alt}** total "
             f"(**{q['pct_kept']:.0%}** of corpus) · new passer-locations "
             f"**{q['n_new_locations_with_passer']}**\n")
    if degenerate:
        L.append(f"> ⚠️ **Degenerate small-N regime.** N={q['n_emitted']} < "
                 f"{q['floor_bite_threshold_N']}, so the per-mode floor rounds to 0 and the "
                 f"diversity spread degrades to a global top-{q['budget_B']}-by-quality fill. "
                 f"The pass is **correct but degenerate** — the live floor/surplus/degradation "
                 f"mechanics only show once the corpus grows past ~{q['floor_bite_threshold_N']} "
                 f"emissions (via more `emit_v1` runs). The floor logic is exercised by "
                 f"`test_tail_alloc.py` at synthetic N.\n")
    L.append("**Correctness checks** — "
             f"smooth-untouched **{_mk(ck['smooth_untouched']['ok'])}** "
             f"({ck['smooth_untouched']['n_checked']} pngs) · "
             f"emit-field-dir-untouched **{_mk(ck['emit_field_dir_untouched']['ok'])}** · "
             f"field-tmp-clean **{_mk(ck['field_tmp_clean']['ok'])}** · "
             f"idempotent **{_mk(ck['idempotent']['ok'])}** "
             f"(next-run new allocs = {ck['idempotent']['next_run_new_allocations']})\n")
    L.append("| mode | floor | achieved | existing | new-supply | degraded |")
    L.append("|------|:-----:|:--------:|:--------:|:----------:|:--------:|")
    for pm in q["per_mode"]:
        L.append(f"| {pm['mode']} | {pm['floor']} | {pm['achieved']} | {pm['existing']} | "
                 f"{pm['new_supply']} | {'⚠️' if pm['degraded'] else ''} |")
    L.append("")
    if parity:
        L.append("**Parity** (in-proc MiningScorer vs `mining_gate.py` CLI):")
        for p in parity:
            L.append(f"- `{p['cid']}` — in-proc {p['inproc_p_ge3']:.6f} vs CLI {p['cli_p_ge3']} "
                     f"· Δ={p['delta']:.2e} · verdict agree: {p['verdict_agree']}")
        L.append("")
    L.append("## Per-location\n")
    for e in rep["per_location"]:
        if e["state"] == "existing_alternate":
            k = e["kept"]
            L.append(f"### {e['loc_id']} · {e['family']} · `{e['palette'][:32]}`  🔒 FIXED ALTERNATE")
            L.append(f"\n→ existing **{k['mode']}** (p_ge3={k.get('p_ge3')}) · side-by-side "
                     f"`{k.get('sidebyside')}` — carried over, not re-rendered.\n")
            continue
        star = "  ✅ NEW KEEPER" if e["kept"] else ""
        L.append(f"### {e['loc_id']} · {e['family']} · `{e['palette'][:32]}`{star}")
        L.append("")
        L.append("| mode | kind | p_ge3 | p_ge2 | E[ord] | pass | kept |")
        L.append("|------|------|------:|------:|-------:|:----:|:----:|")
        for c in e["candidates"]:
            L.append(f"| {c['mode']} | {c['kind']} | {c['p_ge3']:.3f} | {c['p_ge2']:.3f} | "
                     f"{c['E_ord']:.3f} | {'✓' if c['passed'] else ''} | {'★' if c['kept'] else ''} |")
        if e["kept"]:
            L.append(f"\n→ kept **{e['kept']['mode']}** · side-by-side `{e['kept'].get('sidebyside')}`")
        L.append("")
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[report] {OUT_DIR/'report.md'}  +  report.json")
    print(f"[done] {n_total_alt}/{len(rows)} strange alternates "
          f"({rep['curation_pass']['n_new_keepers']} new this pass, {q['pct_kept']:.0%} of corpus)")


def _mk(ok: bool) -> str:
    return "✅" if ok else "❌"


if __name__ == "__main__":
    main()
