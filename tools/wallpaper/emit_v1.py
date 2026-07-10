"""Ship the v1 emission — real 2560x1440 wallpapers from the humanq3 pool.

The emission back-end, end-to-end. Deploys **v2 as the gate head** (no retrain) and
wires gate -> select -> full-res emit so it outputs actual wallpaper PNGs.

Pipeline (see prompts/ship_v1_emission_prompt.md):
  1. GATE-SCORE  run v2 inference on the existing humanq3 crops (already the head's
     ss2 label-crop input spec) -> marginal p_ge3 + continuous readout `score`. No
     re-render for scoring.
  2. GATE        keep marginal p_ge3 > threshold (v3 default 0.90). Good-only.
  3. SELECT      the Stage-2d emission selector: family x Lab-cell MAP-Elites,
     <=1/location, palette cap; fitness = the continuous readout.
  4. EMIT        full-res render ONLY the selected winners at 2560x1440 ss4 Lanczos-3
     (wallpaper canon) via the render_candidate (Recipe 2) path: ONE field dump per
     emitted location + the colormap tail. NOT render-one --palette. normal_map OFF,
     smooth-only (v1). Same (location, palette, params) the head approved, higher res.

Output: emitted wallpaper PNGs + manifest.jsonl + a contact sheet to eyeball.

CAVEAT (report, don't fix): these locations were in v2's training set, so p_ge3 here
is OPTIMISTIC — true quality is closer to the held-out dry-run (~0.60 precision). Fine
for a first real output; the eyeball is the judge. Honest gating arrives with the
fresh-discovery front-end (unseen locations).

    uv run python -u tools/wallpaper/emit_v1.py --limit 2   # smoke: emit 2 winners
    uv run python -u tools/wallpaper/emit_v1.py             # full emission
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "queries"))
sys.path.insert(0, str(REPO / "tools" / "corpus"))

import colormap as cm                # noqa: E402  load_field / stretch_field / render_candidate
import location as loc_mod           # noqa: E402  from_render_block / render_one_flags
import query_sampler as qs           # noqa: E402  load_pool_library

# emission_selector by file path (module isn't a package).
_spec = importlib.util.spec_from_file_location(
    "emission_selector", REPO / "tools/wallpaper/emission_selector.py")
es = importlib.util.module_from_spec(_spec)
sys.modules["emission_selector"] = es
_spec.loader.exec_module(es)

# ---- config surface (tunable in place, no code changes) -------------------- #
POOL_DEFAULT = REPO / "data/wallpaper_corpus/batches/2026-07-05_wallpaper_humanq3_v1"
HEAD_CKPT = REPO / "data/wallpaper_head/v3/model_best.pt"   # rollback: revert to v2/model_best.pt
# Marginal p_ge3 > threshold (good-only). QUALITY-FLOOR / volume-policy dial, overridable
# via --gate. Retuned 0.5 -> 0.90 for head v3 (see prompts/prompt_gate_retune_v3.md):
# v3 gained a real precision GRADIENT the flat-on-v2 head lacked (eval precision of passers
# 0.58@0.5 -> 0.68@0.90 -> 0.78@0.99). On the current dramatic beam the gate is NOT a volume
# dial — the emission SELECTOR saturates first (winners flat ~21/52-loc-batch across
# thr in [0.5,0.90], all winners already p_ge3>0.94), so 0.90 buys a higher-quality floor
# feeding the selector at ZERO volume cost and holds the line on weaker/future pools. Raise
# toward 0.95+ only to trade ~1 winner for a bit more precision; lower for more volume.
GATE_THRESHOLD = 0.90
PALETTE_CAP_FRAC = 0.05          # selector palette cap = max(2, ceil(frac * N_reachable_cells))
GRID = es.ColorGrid()            # 3x3 a/b x 2 L = 18 color cells; family x cell = behavior space

# wallpaper canon (render-one locked default): 2560x1440 grid ss4 Lanczos-3.
EMIT_W, EMIT_H, EMIT_SS = 2560, 1440, 4
EMIT_FILTER = "lanczos3"

EXE = REPO / "target" / "release" / "fractal-generator.exe"
OUT_DIR = REPO / "out" / "wallpaper" / "emit_v1"
FIELD_DIR = OUT_DIR / "fields"         # disposable full-res field dumps (one per location)
WALL_DIR = OUT_DIR / "wallpapers"      # emitted PNGs
CELL_CACHE = OUT_DIR / "colorcells.json"


# ===========================================================================
# 1. Gate-score: v2 inference on the existing crops (no re-render).
# ===========================================================================
def load_v2_scorer(device):
    """Load the v2 wallpaper head + its deterministic deploy transform. Returns a
    callable crops-paths -> (cond, marg, ssum) matching train_wallpaper_v2.predict_all:
    cond = sigmoid(logits) CONDITIONAL, marg = cumprod (the MARGINAL gate probs)."""
    import timm
    from classifier.data import Transform
    from classifier.model import BACKBONE, score_from_logits

    ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    K = int(cfg["num_classes"])                     # 4
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=K - 1,
                              drop_rate=cfg.get("drop_rate", 0.2),
                              drop_path_rate=cfg.get("drop_path_rate", 0.1))
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)
    tf = Transform(geometry=cfg["geometry"], interp=cfg["interpolation"],
                   mean=tuple(cfg["mean"]), std=tuple(cfg["std"]), train=False)

    @torch.no_grad()
    def score(paths, batch_size=32):
        cond = np.zeros((len(paths), K - 1), dtype=np.float64)
        ssum = np.zeros(len(paths), dtype=np.float64)
        i = 0
        while i < len(paths):
            chunk = paths[i:i + batch_size]
            batch = []
            for p in chunk:
                with Image.open(p) as im:
                    im.load()
                    batch.append(tf(im.convert("RGB")))
            x = torch.stack(batch).to(device)
            logits = model(x).float()
            cond[i:i + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
            ssum[i:i + len(chunk)] = score_from_logits(logits, "ordinal").cpu().numpy()
            i += len(chunk)
        marg = np.cumprod(cond, axis=1)             # marg[:,1] = marginal P(>=3)
        return cond, marg, ssum

    return score, cfg


# ===========================================================================
# 2/3. Build candidates (color cells cached) + run the selector.
# ===========================================================================
def _thumb_rgb(jpg: Path, w: int = 96) -> np.ndarray:
    with Image.open(jpg) as im:
        im = im.convert("RGB")
        iw, ih = im.size
        im = im.resize((w, max(1, round(w * ih / iw))), Image.BILINEAR)
        return np.asarray(im)


def load_color_cells(rows, crops_dir: Path) -> dict[str, int]:
    cache = json.loads(CELL_CACHE.read_text()) if CELL_CACHE.exists() else {}
    missing = [r["image_id"] for r in rows if r["image_id"] not in cache]
    if missing:
        print(f"[color] dominant Lab for {len(missing)} crops ({len(cache)} cached)")
        for iid in missing:
            cache[iid] = GRID.cell(es.dominant_lab(_thumb_rgb(crops_dir / f"{iid}.jpg"),
                                                   method="median"))
        CELL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CELL_CACHE.write_text(json.dumps(cache))
    return {r["image_id"]: cache[r["image_id"]] for r in rows}


def build_and_select(pool_dir: Path, gate_thr: float):
    rows = [json.loads(l) for l in (pool_dir / "images.jsonl").read_text().splitlines() if l.strip()]
    crops_dir = pool_dir / "crops"
    paths = [str(crops_dir / f"{r['image_id']}.jpg") for r in rows]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_fn, cfg = load_v2_scorer(device)
    print(f"[gate] scoring {len(rows)} crops with head {HEAD_CKPT.parent.name} "
          f"(best_epoch {cfg.get('best_epoch')}) on {device.type}")
    t0 = time.time()
    cond, marg, ssum = score_fn(paths)
    p_ge3 = marg[:, 1]
    print(f"[gate] scored in {time.time()-t0:.0f}s  ·  p_ge3 quantiles "
          f".5/.75/.9/.95/max = {np.round(np.quantile(p_ge3,[.5,.75,.9,.95,1]),4).tolist()}")

    # v2-parity cross-check on any eval crops we happen to score (honest sanity check).
    _parity_check(rows, p_ge3, ssum)

    cells = load_color_cells(rows, crops_dir)
    cands = []
    for i, r in enumerate(rows):
        prov = r.get("provenance", {}) or {}
        cands.append(es.Candidate(
            location_id=loc_mod.from_render_block(r["render"]).key(),
            palette_id=r["render"]["palette"],
            family=prov.get("family") or r["render"].get("fractal_type") or "mandelbrot",
            fitness=float(ssum[i]),
            color_cell=cells[r["image_id"]],
            image_id=r["image_id"],
            meta={"p_ge3": float(p_ge3[i]), "row": r},
        ))

    n_pass = sum(1 for c in cands if c.meta["p_ge3"] > gate_thr)
    res = es.select(cands, gate=lambda c: c.meta["p_ge3"] > gate_thr,
                    grid=GRID, palette_cap_frac=PALETTE_CAP_FRAC)
    picks = sorted(res.picks, key=lambda c: -c.fitness)
    print(f"[gate] p_ge3 > {gate_thr}: {n_pass}/{len(cands)} pass  ->  "
          f"[select] {len(picks)} emitted  "
          f"(cells {res.report['cells_filled']}/{res.report['cells_reachable']}, "
          f"{res.report['n_distinct_palettes_picked']} palettes, cap {res.palette_cap})")
    print(f"[select] per-family: {res.report['per_family_spread']}")
    return picks, res, n_pass


def _parity_check(rows, p_ge3, ssum):
    ev_path = HEAD_CKPT.parent / "eval_scores.jsonl"   # track the deployed head (v3), not v2
    if not ev_path.exists():
        return
    ev = {json.loads(l)["image_id"]: json.loads(l)
          for l in ev_path.read_text().splitlines() if l.strip()}
    idx = {r["image_id"]: i for i, r in enumerate(rows)}
    dp, ds = [], []
    for iid, e in ev.items():
        if iid in idx:
            dp.append(abs(p_ge3[idx[iid]] - e["p_ge3"]))
            ds.append(abs(ssum[idx[iid]] - e["score"]))
    if dp:
        print(f"[parity] vs v2 eval_scores on {len(dp)} shared crops: "
              f"max|Δp_ge3|={max(dp):.2e}  max|Δscore|={max(ds):.2e}")


# ===========================================================================
# 4. Emit — full-res render_candidate (Recipe 2) for each selected winner.
# ===========================================================================
def ensure_emit_field(loc):
    """Dump (or reuse) the wallpaper-canon smooth field for `loc` — ONE dump per
    emitted location (field invariance). 2560x1440 ss4 -> 10240x5760 raw field."""
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    import hashlib
    h = hashlib.sha1(f"{loc.key()}|{EMIT_W}x{EMIT_H}ss{EMIT_SS}|{loc.maxiter}".encode()).hexdigest()[:16]
    stem = f"{loc.family}_{h}_{EMIT_W}x{EMIT_H}ss{EMIT_SS}"
    bin_path, json_path = FIELD_DIR / f"{stem}.bin", FIELD_DIR / f"{stem}.json"
    if not (bin_path.exists() and json_path.exists()):
        cmd = [str(EXE), "render-one",
               "--cx", loc.cx, "--cy", loc.cy, "--fw", loc.fw,
               "--width", str(EMIT_W), "--height", str(EMIT_H),
               "--supersample", str(EMIT_SS), "--maxiter", str(loc.maxiter),
               "--dump-field", str(bin_path)]
        cmd += loc_mod.render_one_flags(loc)
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"emit dump-field failed for {stem}:\n{r.stderr[-600:]}")
    return cm.load_field(str(bin_path), str(json_path)), bin_path


def config_from_params(params: dict, loc) -> cm.CandidateConfig:
    """Reconstruct the exact colouring recipe the head approved. filter -> lanczos3
    (wallpaper canon); the stored eval_filter (box) was the sampler's scoring filter."""
    return cm.CandidateConfig(
        palette=params["palette"],
        location=cm.LocationRef(kind=loc.family, cx=loc.cx, cy=loc.cy, fw=loc.fw,
                                maxiter=loc.maxiter, c_re=loc.c_re, c_im=loc.c_im),
        eval_width=EMIT_W, eval_height=EMIT_H,
        reverse=bool(params.get("reverse", False)),
        log_premap=params.get("log_premap", "none"),
        gamma=float(params.get("gamma", 1.0)),
        phase=float(params.get("phase", 0.0)),
        n_cycles=int(params.get("n_cycles", 1)),
        transfer=params.get("transfer", "pct"),
        transfer_gamma=float(params.get("transfer_gamma", 0.0)),
        interior_color=tuple(params.get("interior_color", (0.0, 0.0, 0.0))),
        filter=EMIT_FILTER,
    )


def emit(picks, lib, limit: int):
    WALL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    todo = picks[:limit] if limit else picks
    print(f"\n[emit] rendering {len(todo)} winners at {EMIT_W}x{EMIT_H} ss{EMIT_SS} {EMIT_FILTER}")
    for i, c in enumerate(todo):
        row = c.meta["row"]
        loc = loc_mod.from_render_block(row["render"])
        params = (row.get("provenance", {}) or {}).get("params", {})
        t0 = time.time()
        field, bin_path = ensure_emit_field(loc)
        cfg = config_from_params(params, loc)
        img = cm.render_candidate(field, cfg, lib)     # (1440,2560,3) uint8 sRGB
        assert img.shape[:2] == (EMIT_H, EMIT_W), (c.image_id, img.shape)
        del field
        bin_path.unlink(missing_ok=True)               # 236MB/loc, distinct per winner — drop after use
        out_png = WALL_DIR / f"emit_{i:03d}_{c.image_id}.png"
        Image.fromarray(img).save(out_png)
        rec = {
            "emit_index": i, "image_id": c.image_id, "png": str(out_png.relative_to(REPO)),
            "family": c.family, "palette": c.palette_id, "color_cell": c.color_cell,
            "p_ge3": c.meta["p_ge3"], "fitness": c.fitness,
            "location": {"cx": loc.cx, "cy": loc.cy, "fw": loc.fw, "maxiter": loc.maxiter,
                         "fractal_type": loc.family, "c_re": loc.c_re, "c_im": loc.c_im},
            "params": params,
        }
        manifest.append(rec)
        print(f"[emit] {i:03d}/{len(todo)} {c.family:16} {c.palette_id:22} "
              f"p_ge3={c.meta['p_ge3']:.3f} fit={c.fitness:.3f}  [{time.time()-t0:.0f}s]  -> {out_png.name}")
    (OUT_DIR / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in manifest) + "\n")
    print(f"[emit] wrote {OUT_DIR/'manifest.jsonl'} ({len(manifest)} rows)")
    return manifest


# ===========================================================================
# Contact sheet.
# ===========================================================================
def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(manifest, gate_thr, n_pass, n_pool):
    if not manifest:
        print("[sheet] nothing emitted, skipping")
        return
    COLS = min(6, len(manifest))
    TW, TH = 360, 203
    BORDER, CAP, PAD, HEADER = 4, 56, 12, 90
    cell_w, cell_h = TW + 2 * BORDER, TH + 2 * BORDER + CAP
    rows_n = (len(manifest) + COLS - 1) // COLS
    W = COLS * cell_w + (COLS + 1) * PAD
    H = HEADER + rows_n * cell_h + (rows_n + 1) * PAD
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    f_title, f_sub, f_cap, f_capb = _font(30), _font(17), _font(14), _font(15)
    d.text((PAD, 12), f"v1 emission · v2 gate p_ge3 > {gate_thr:g} · {EMIT_W}x{EMIT_H} ss{EMIT_SS}",
           font=f_title, fill=(235, 235, 235))
    d.text((PAD, 50), f"{len(manifest)} wallpapers emitted  ·  {n_pass}/{n_pool} passed gate  ·  "
                      f"CAVEAT: locations in v2 train set → p_ge3 optimistic (dry-run precision ~0.60)",
           font=f_sub, fill=(170, 170, 175))
    for i, rec in enumerate(manifest):
        rr, col = divmod(i, COLS)
        x = PAD + col * (cell_w + PAD)
        y = HEADER + PAD + rr * (cell_h + PAD)
        with Image.open(REPO / rec["png"]) as im:
            im = im.convert("RGB").resize((TW, TH), Image.LANCZOS)
        d.rectangle([x, y, x + cell_w - 1, y + TH + 2 * BORDER - 1], fill=(60, 60, 66))
        sheet.paste(im, (x + BORDER, y + BORDER))
        cy = y + TH + 2 * BORDER
        d.rectangle([x, cy, x + cell_w - 1, cy + CAP - 1], fill=(30, 30, 34))
        d.text((x + 6, cy + 5), f"p{rec['p_ge3']:.2f}  fit{rec['fitness']:.2f}",
               font=f_capb, fill=(230, 230, 230))
        d.text((x + 6, cy + 22), rec["family"], font=f_cap, fill=(150, 200, 235))
        pal = rec["palette"][:30]
        d.text((x + 6, cy + 38), pal, font=f_cap, fill=(150, 150, 155))
    out = OUT_DIR / "contact_sheet.png"
    sheet.save(out)
    print(f"[sheet] wrote {out} ({W}x{H}, {len(manifest)} tiles)")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Ship the v1 emission — real wallpapers.")
    ap.add_argument("--pool", type=Path, default=POOL_DEFAULT, help="candidate pool batch dir")
    ap.add_argument("--gate", type=float, default=GATE_THRESHOLD,
                    help="marginal p_ge3 quality-floor gate (v3 default 0.90; lower=more volume)")
    ap.add_argument("--limit", type=int, default=0, help="emit only the first N winners (smoke)")
    ap.add_argument("--no-emit", action="store_true", help="gate+select only, skip full-res render")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks, res, n_pass = build_and_select(args.pool, args.gate)
    n_pool = res.report["n_candidates"]
    if args.no_emit:
        print("[emit] --no-emit: stopping after selection")
        return
    lib = qs.load_pool_library()
    manifest = emit(picks, lib, args.limit)
    contact_sheet(manifest, args.gate, n_pass, n_pool)
    print(f"\n[done] {len(manifest)} wallpapers -> {WALL_DIR}")


if __name__ == "__main__":
    main()
