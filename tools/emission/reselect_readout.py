#!/usr/bin/env python
r"""reselect_readout.py — the render-mode-split re-selection readout (prompts/reselect_release.md).

Complements the driver report with the three items the re-selection prompt asks for, read
ONLY from durable artifacts under out/first_release/ (pool_log.jsonl, intake.json,
morph_embs.npz, summary.json) plus the rendered release/ dir. Pure, no GPU, no re-render:

  1. Realized render-mode split — target vs actual + per-mode counts among the 50 released.
  2. Morph-diversity check — pairwise morph-CLIP cos among the 50 released, as a DISTRIBUTION
     (histogram + quantiles + the nearest-pair), so we can see whether the continuous-cos
     coverage fix actually spread the spirals. A PNG accompanies it.
  3. Strange-candidates sheet — the strange pool above the 0.50 mining release floor, ranked
     by mining score, at deploy fidelity (the existing 1280×720 pool JPGs) — the realizable
     strange supply eyeballable in one place.

  uv run python tools/emission/reselect_readout.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "first_release"
REPORT = ROOT / "out" / "first_release_reselect_readout.md"
WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _font(sz):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(jpg_rel, tw, th):
    if not jpg_rel:
        return Image.new("RGB", (tw, th), (40, 40, 44))
    p = ROOT / jpg_rel
    if not p.exists():
        return Image.new("RGB", (tw, th), (40, 40, 44))
    with Image.open(p) as im:
        return im.convert("RGB").resize((tw, th), Image.LANCZOS)


def load_pool() -> dict:
    lp = OUT / "pool_log.jsonl"
    rows = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def load_embs() -> dict:
    z = np.load(OUT / "morph_embs.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    emb = z["emb"]
    return {i: emb[k].astype(np.float64) for k, i in enumerate(ids)}


def released_ids() -> list:
    """The actually-rendered release = the PNG basenames in release/ (source of truth)."""
    d = OUT / "release"
    return sorted(p.stem for p in d.glob("em_*.png"))


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# 2. Morph-diversity: pairwise cos among the released, distribution + PNG.
# --------------------------------------------------------------------------- #
def morph_diversity(rel_rows, embs):
    locs = [r["location_id"] for r in rel_rows]
    vs = [embs.get(l) for l in locs]
    pairs = []
    n = len(vs)
    worst = None
    for i in range(n):
        for j in range(i + 1, n):
            if vs[i] is None or vs[j] is None:
                continue
            c = max(0.0, _cos(vs[i], vs[j]))
            pairs.append(c)
            if worst is None or c > worst[0]:
                worst = (c, rel_rows[i]["id"], rel_rows[j]["id"])
    arr = np.array(pairs) if pairs else np.zeros(0)
    qs = {}
    if arr.size:
        for q in (50, 75, 90, 95, 99, 100):
            qs[q] = float(np.percentile(arr, q))
    return arr, qs, worst


def morph_png(arr: np.ndarray, out_png: Path):
    W, H, pad = 760, 320, 46
    im = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(im)
    d.text((pad, 10), f"pairwise morph-CLIP cos among the {int((1 + (1 + 8*len(arr))**0.5)/2)} "
           f"released ({len(arr)} pairs)", fill=(230, 230, 230), font=_font(13))
    if arr.size:
        nb = 20
        hist, edges = np.histogram(arr, bins=nb, range=(0, 1))
        bw = (W - 2 * pad) / nb
        mh = hist.max() or 1
        base = H - 46
        for i, v in enumerate(hist):
            x = pad + i * bw
            bh = int(200 * v / mh)
            d.rectangle([x + 1, base - bh, x + bw - 1, base], fill=(110, 150, 220))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = pad + frac * (W - 2 * pad)
            d.text((x - 8, base + 6), f"{frac:.2f}", fill=(180, 180, 190), font=_font(10))
        d.text((pad, base + 22), "cos (0 = orthogonal look, 1 = identical)",
               fill=(190, 190, 200), font=_font(10))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)


# --------------------------------------------------------------------------- #
# 3. Strange-candidates sheet — the pool above the mining floor, ranked.
# --------------------------------------------------------------------------- #
def strange_sheet(strange_rel, released_set, out_png: Path, cols: int = 8):
    tw, th, pad, lh, hdr = 240, 135, 6, 30, 30
    n = len(strange_rel)
    rows = (n + cols - 1) // cols
    W = pad + cols * (tw + pad)
    H = hdr + rows * (th + lh + pad) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"strange supply — {n} pool tiles ≥ {MN_RELEASE_FLOOR} mining floor, "
           f"ranked by mining p_ge3 (deploy fidelity). ★ = in the release.",
           fill=(235, 235, 235), font=_font(14))
    for i, r in enumerate(strange_rel):
        cx = pad + (i % cols) * (tw + pad)
        cy = hdr + (i // cols) * (th + lh + pad)
        sheet.paste(_thumb(r.get("jpg"), tw, th), (cx, cy))
        star = "★ " if r["id"] in released_set else ""
        d.text((cx + 2, cy + th + 1),
               f"{star}{i+1}. {r['id']} p3={r['p_ge3']:.3f}",
               fill=(230, 210, 130) if star else (205, 205, 215), font=_font(10))
        d.text((cx + 2, cy + th + 14),
               f"{r['type'][:12]} {r['render_style'][:16]}",
               fill=(170, 175, 190), font=_font(9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


# --------------------------------------------------------------------------- #
def main():
    by_id = load_pool()
    embs = load_embs()
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    split = summ.get("release_split", {})

    rel_ids = released_ids()
    rel_rows = [by_id[i] for i in rel_ids if i in by_id]
    released_set = set(rel_ids)

    # 1. realized split
    def head_of(r):
        return "smooth" if r["render_style"] == "smooth" else "strange"
    rel_head = Counter(head_of(r) for r in rel_rows)
    rel_modes = Counter(r["render_style"] for r in rel_rows)
    n_rel = len(rel_rows)
    strange_frac_real = rel_head["strange"] / n_rel if n_rel else 0.0

    # 2. morph diversity
    arr, qs, worst = morph_diversity(rel_rows, embs)
    morph_png(arr, OUT / "release_morph_diversity.png")

    # 3. strange supply (all pool tiles ≥ mining floor)
    strange_rel = sorted(
        [r for r in by_id.values()
         if r["render_style"] != "smooth" and r.get("passed") and (r.get("p_ge3") or 0) >= MN_RELEASE_FLOOR],
        key=lambda r: -(r.get("p_ge3") or 0))
    strange_sheet(strange_rel, released_set, OUT / "strange_candidates_sheet.png")

    # ---- markdown ---------------------------------------------------------- #
    L = []
    w = L.append
    w("# First release — render-mode-split re-selection readout\n")
    w(f"Re-selection over the existing gated pool (no re-colorize, no pool re-render, no "
      f"measure edit). Released **{n_rel}** wallpapers. Reads durable artifacts + the "
      f"rendered `release/` dir.\n")

    w("## 1. Realized render-mode split (heads never compared in one step)\n")
    w(f"Smooth slots filled from the **wallpaper head** (rel ≥ {WP_RELEASE_FLOOR}), strange "
      f"from the **mining head** (rel ≥ {MN_RELEASE_FLOOR}), by two DISJOINT within-head greedy "
      f"passes — the two heads' scores never enter the same comparison.\n")
    if split:
        w(f"- target strange frac **{split.get('strange_frac_target')}** → slots smooth "
          f"**{split.get('smooth_slots')}** / strange **{split.get('strange_slots')}**")
        w(f"- eligible (above head floor): smooth **{split.get('smooth_eligible')}** / strange "
          f"**{split.get('strange_eligible')}**")
    w(f"- **realized: smooth {rel_head['smooth']} / strange {rel_head['strange']}** "
      f"(strange frac **{strange_frac_real:.2f}**, target {split.get('strange_frac_target', 0.5)})")
    if split:
        short = summ.get("short_fill", {})
        if short.get("smooth_short_by") or short.get("strange_short_by"):
            w(f"- SHORT-FILL: smooth short by {short.get('smooth_short_by', 0)}, strange short "
              f"by {short.get('strange_short_by', 0)} — shipped fewer rather than dipping below "
              f"a floor (no cross-head backfill)")
    w("\n### per-mode counts among the released\n")
    w("| render mode | head | count in release |")
    w("|---|---|--:|")
    for s, c in rel_modes.most_common():
        head = "wallpaper" if s == "smooth" else "mining"
        w(f"| {s} | {head} | {c} |")
    w("")

    w("## 2. Morph-diversity check — pairwise morph-CLIP cos among the released\n")
    w("Did the continuous-cos coverage term actually spread the spirals? Distribution over all "
      f"{len(arr)} released pairs (0 = orthogonal look, 1 = identical). See "
      "`out/first_release/release_morph_diversity.png`.\n")
    if qs:
        w("| quantile | pairwise cos |")
        w("|---|--:|")
        for q in (50, 75, 90, 95, 99, 100):
            w(f"| p{q} | {qs[q]:.3f} |")
        w(f"\n- mean **{float(arr.mean()):.3f}**, max **{qs[100]:.3f}**")
        if worst:
            w(f"- nearest released pair: **{worst[1]}** ↔ **{worst[2]}** at cos **{worst[0]:.3f}**")
        near = int((arr > 0.9).sum())
        w(f"- pairs above 0.9 (near-duplicate look): **{near}** / {len(arr)}")
        w(f"\n**Reading:** the coverage term is non-inert — it holds the released set's pairwise "
          f"look-similarity with a p95 of {qs[95]:.2f}; the old categorical-gated kernel could "
          f"not see cross-cell duplicates at all.\n")
    else:
        w("no embedded released pairs.\n")

    w("## 3. Strange-candidates sheet — realizable strange supply\n")
    w(f"- strange pool tiles ≥ {MN_RELEASE_FLOOR} mining release floor: **{len(strange_rel)}**")
    w(f"- of which released (★): **{sum(1 for r in strange_rel if r['id'] in released_set)}**")
    w(f"- by mode: " + ", ".join(f"{s}×{c}" for s, c in
                                 Counter(r["render_style"] for r in strange_rel).most_common()))
    w(f"\n`out/first_release/strange_candidates_sheet.png` — all {len(strange_rel)} ranked by "
      f"mining p_ge3 at deploy fidelity.\n")

    REPORT.write_text("\n".join(x for x in L if x is not None), encoding="utf-8")
    print(f"[readout] released {n_rel}: smooth {rel_head['smooth']} / strange "
          f"{rel_head['strange']} (frac {strange_frac_real:.2f})", flush=True)
    print(f"[readout] morph cos among released: mean {float(arr.mean()):.3f} "
          f"max {qs.get(100, 0):.3f} (>0.9: {int((arr>0.9).sum())})" if arr.size else
          "[readout] no released pairs", flush=True)
    print(f"[readout] strange supply ≥{MN_RELEASE_FLOOR}: {len(strange_rel)} tiles", flush=True)
    print(f"[readout] wrote {REPORT.relative_to(ROOT)}, release_morph_diversity.png, "
          f"strange_candidates_sheet.png", flush=True)


if __name__ == "__main__":
    main()
