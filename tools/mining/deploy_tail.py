"""Post-emission strange-mode deploy tail (pilot, STANDALONE — not wired into emit_v1).

For each ALREADY-EMITTED location (smooth winner + its approved palette/params chosen
upstream by emit_v1), render a lean set of strange-mode candidate variants over the
INHERITED approved palette, score each with the LOCKED `mining_v1` gate
(`tools/mining/mining_gate.MiningScorer`, threshold 0.50 on marginal p_ge3), and keep
gate-passers as alternate wallpapers. Smooth keeps shipping via emit_v1's normal path;
this tail only ADDS strange-mode alternates.

Load-bearing (see prompts/render_mode_deploy_tail_prompt.md + the memories):
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

    uv run python -u tools/mining/deploy_tail.py            # full pilot
    uv run python -u tools/mining/deploy_tail.py --score-only   # skip full-res render
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
FIELD_TMP = OUT_DIR / "_fields"             # disposable field dumps
KEEP_DIR = OUT_DIR / "keepers"              # full-res keeper pngs
SBS_DIR = OUT_DIR / "sidebyside"            # smooth-vs-kept comparisons

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
def main():
    ap = argparse.ArgumentParser(description="Post-emission strange-mode deploy tail (pilot).")
    ap.add_argument("--manifest", type=Path, default=EMIT_MANIFEST)
    ap.add_argument("--score-only", action="store_true", help="gate/select only; skip full-res")
    ap.add_argument("--limit", type=int, default=0, help="only the first N emitted locations")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    n_emit = len(rows)
    SCORE_CROPS.mkdir(parents=True, exist_ok=True)
    print(f"[tail] {n_emit} emitted locations x {len(ROSTER)} modes "
          f"= {n_emit*len(ROSTER)} candidates @ {SC_W}x{SC_H} ss{SC_SS}")

    # 1. render every candidate at scoring res.
    cands = []   # dicts: emit_index, image_id(loc), mode, kind, crop path, palette, info
    t0 = time.time()
    for row in rows:
        loc = loc_mod.from_render_block(row["location"])
        palette = row["params"]["palette"]
        cp = _color_params(row["params"])
        for mode, kind in ROSTER:
            cid = f"{row['image_id']}__{mode}"
            crop = SCORE_CROPS / f"{cid}.jpg"
            tc = time.time()
            # transfer_dropped is deterministic (rust modes drop grad; inherited is pct
            # here so it is False) — reconstructable without re-rendering, so a crop that
            # already exists on disk is reused (resume / cheap backfill of a failed crop).
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

    # 2. score all candidates through the LOCKED gate.
    scorer = MiningScorer()
    print(f"[gate] {scorer.__class__.__name__} thr={scorer.threshold} on {scorer.device}")
    scores = scorer.score_paths([str(c["crop"]) for c in cands])
    for c, s in zip(cands, scores):
        c["p_ge3"], c["p_ge2"], c["ord"], c["passed"] = s.p_ge3, s.p_ge2, s.score, s.passed

    by_loc = {}
    for c in cands:
        by_loc.setdefault(c["loc_id"], []).append(c)

    # 3-4. batch-level diversity allocation: spread strange keepers across modes.
    #      Each candidate carries loc_id/mode/p_ge3 -> allocate_strange assigns each
    #      location to at most one mode's slot, floors first then surplus by quality.
    passers = [c for c in cands if c["passed"]]
    keepers, alloc = allocate_strange(passers, n_emit, [m for m, _ in ROSTER],
                                      STRANGE_BUDGET_FRAC)
    keepers.sort(key=lambda c: -c["p_ge3"])   # render/report in quality order
    n_loc_passer = len({c["loc_id"] for c in passers})
    achieved = alloc["achieved"]
    print(f"[keep] budget B={alloc['B']} (round({STRANGE_BUDGET_FRAC:.0%} x {n_emit})) "
          f"floor={alloc['floor']}/mode ; {n_loc_passer} locations have a passer "
          f"-> keeping {len(keepers)}")
    for m, _ in ROSTER:
        supply = len({c['loc_id'] for c in passers if c['mode'] == m})
        print(f"       {m:32} floor={alloc['floor']} achieved={achieved[m]} "
              f"(supply={supply} distinct-loc passers)")

    # 5. full-res render the keepers + side-by-side.
    if not args.score_only:
        KEEP_DIR.mkdir(parents=True, exist_ok=True)
        SBS_DIR.mkdir(parents=True, exist_ok=True)
        for c in keepers:
            loc = loc_mod.from_render_block(c["row"]["location"])
            keep_png = KEEP_DIR / f"{c['cid']}_{EMIT_W}x{EMIT_H}.png"
            tk = time.time()
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
            print(f"[emit] {c['cid']:34} p_ge3={c['p_ge3']:.3f}  [{time.time()-tk:.0f}s] "
                  f"-> {keep_png.name}")

    # 6. parity: re-score 1-2 kept scoring crops through the standalone gate CLI
    #    (deploy path == measurement path across two independent entry points).
    parity = run_parity(keepers)

    # 7. report.
    write_report(rows, cands, by_loc, keepers, alloc, n_loc_passer, scorer, parity, args)


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


def write_report(rows, cands, by_loc, keepers, alloc, n_loc_passer, scorer, parity, args):
    keep_ids = {c["cid"] for c in keepers}
    rep = {
        "gate": {"version": "mining_v1", "threshold": scorer.threshold},
        "allocation": {
            "budget_frac": alloc["budget_frac"], "n_emitted": len(rows),
            "budget_B": alloc["B"], "floor_per_mode": alloc["floor"],
            "n_modes": alloc["n_modes"],
            "n_locations_with_passer": n_loc_passer,
            "n_strange_kept": len(keepers),
            "pct_kept": (len(keepers) / len(rows)) if rows else 0.0,
            "per_mode": [
                {"mode": m, "floor": alloc["floor"], "achieved": alloc["achieved"][m],
                 "supply": len({c["loc_id"] for c in cands
                                if c["mode"] == m and c["passed"]}),
                 "degraded": alloc["achieved"][m] < alloc["floor"]}
                for m, _ in ROSTER],
        },
        "parity": parity,
        "per_location": [],
    }
    for row in rows:
        lid = row["image_id"]
        cs = sorted(by_loc.get(lid, []), key=lambda c: -c["p_ge3"])
        entry = {"loc_id": lid, "family": row["location"].get("fractal_type"),
                 "palette": row["params"]["palette"], "smooth_png": row["png"],
                 "candidates": [], "kept": None}
        for c in cs:
            cd = {"mode": c["mode"], "kind": c["kind"], "p_ge3": round(c["p_ge3"], 4),
                  "p_ge2": round(c["p_ge2"], 4), "E_ord": round(c["ord"], 4),
                  "passed": c["passed"], "kept": c["cid"] in keep_ids,
                  "transfer_dropped": c["info"].get("transfer_dropped", False)}
            if c["cid"] in keep_ids:
                cd["gate_stamp"] = gate_stamp(c["p_ge3"], scorer.threshold)
                cd["keeper_png"] = c.get("keep_png")
                cd["sidebyside"] = c.get("sbs")
                entry["kept"] = cd
            entry["candidates"].append(cd)
        rep["per_location"].append(entry)

    (OUT_DIR / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    # markdown
    L = []
    L.append("# Render-mode deploy tail — pilot report\n")
    q = rep["allocation"]
    L.append(f"**Gate** `mining_v1` (LOCKED) · threshold `{scorer.threshold}` on marginal p_ge3\n")
    L.append(f"**Diversity allocation** — emitted **{q['n_emitted']}** · budget "
             f"B=round({q['budget_frac']:.0%}×{q['n_emitted']})=**{q['budget_B']}** · "
             f"floor=**{q['floor_per_mode']}**/mode (n={q['n_modes']}) · locations with a "
             f"gate-passer **{q['n_locations_with_passer']}** · strange kept "
             f"**{q['n_strange_kept']}** (**{q['pct_kept']:.0%}**)\n")
    L.append("| mode | floor | achieved | supply | degraded |")
    L.append("|------|:-----:|:--------:|:------:|:--------:|")
    for pm in q["per_mode"]:
        L.append(f"| {pm['mode']} | {pm['floor']} | {pm['achieved']} | {pm['supply']} | "
                 f"{'⚠️' if pm['degraded'] else ''} |")
    L.append("")
    if parity:
        L.append("**Parity** (in-proc MiningScorer vs `mining_gate.py` CLI):")
        for p in parity:
            L.append(f"- `{p['cid']}` — in-proc {p['inproc_p_ge3']:.6f} vs CLI {p['cli_p_ge3']} "
                     f"· Δ={p['delta']:.2e} · verdict agree: {p['verdict_agree']}")
        L.append("")
    L.append("## Per-location\n")
    for e in rep["per_location"]:
        star = "  ✅ KEEPER" if e["kept"] else ""
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
    print(f"[done] {len(keepers)}/{len(rows)} strange keepers ({q['pct_kept']:.0%})")


if __name__ == "__main__":
    main()
