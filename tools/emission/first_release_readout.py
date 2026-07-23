#!/usr/bin/env python
r"""first_release_readout.py — the prompt-specific readout for the first real release,
complementing the driver's own report.py.

Reads ONLY the durable artifacts (out/first_release/{pool_log.jsonl, intake.json} +
data/emission/target_measure.json) — pure, no GPU, safe to run alongside the colorize or
after it. Produces the §Readout items report.py does not:

  1. Realized shares vs the target measure — per-partition (type) + palette-flavor + style
     marginals for the GATED pool and the RELEASE set, next to the measure's cell-normalized
     target (what the DeficitModel actually drives) — "where the release lands vs the order book".
  2. Cell reachability over the real library — best_in_cell from actual pool occupancy: which
     joint (type,cluster,flavor,style) cells filled, which (type,cluster) pairs never produced
     a gated wallpaper.
  3. Per-niche percentile health at scale — the within-cell percentile DISTRIBUTION (is the
     singleton-percentile degeneracy from the smoke gone at library scale? — reported, not assumed).
  4. Strange inventory above the 0.50 mining release floor.
  5. Realized hue/chroma histograms accumulated over the pool (+ a PNG).
  6. Reject autopsy — a fate-stratified contact sheet (gated / floor-rejected / errored) at
     deploy fidelity from the pool renders.
  7. Per-stage reconciliation: attempts == passed + floor-dropped + errored (loud on mismatch).

  uv run python tools/emission/first_release_readout.py
  uv run python tools/emission/first_release_readout.py --release-n 50   # mark the release picks
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools", ROOT / "tools" / "corpus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.emission import cells as C            # noqa: E402
from tools.emission import selection as SEL      # noqa: E402
from tools.emission import descriptor as D       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = ROOT / "out" / "first_release"
REPORT = ROOT / "out" / "first_release_readout.md"
MEASURE = ROOT / "data" / "emission" / "target_measure.json"
WP_RELEASE_FLOOR, MN_RELEASE_FLOOR = 0.90, 0.50
STYLES = ["smooth", "tia", "stripe", "smooth_mean_angle", "smooth_angle_min",
          "composite_c7_smooth_trap_circle", "composite_c13_smooth_stripe",
          "composite_c17_smooth_curvature"]
HUE_NAMES = ["red", "orange", "yellow", "chartreuse", "green", "spring",
             "cyan", "azure", "blue", "violet", "magenta", "rose"]


def _font(sz):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def load_pool() -> list:
    lp = OUT / "pool_log.jsonl"
    if not lp.exists():
        raise SystemExit(f"no pool log at {lp}")
    return [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]


def release_floor(style: str) -> float:
    return WP_RELEASE_FLOOR if style == "smooth" else MN_RELEASE_FLOOR


# --------------------------------------------------------------------------- #
# Target measure — cell-normalized marginals (what the DeficitModel drives).
# --------------------------------------------------------------------------- #
def target_marginals(cluster_tags: dict, by_id_family: dict, flavors, styles,
                     source_tags: dict | None = None):
    """The measure's target fraction per cell, aggregated to type / flavor / style marginals.
    Uses the full feasible support (no attempt-cap eviction) — the nominal order book.

    `source_tags` (location_id -> durable ledger tag, from the snapshot) resolves any
    source-tag measure override to concrete clusters so the readout reflects the SAME weighting
    the emission drove. Absent it, a source-tag override shows unresolved (no-op) and we say so
    loudly rather than silently under-report the up-weighted region."""
    cfg = json.loads(MEASURE.read_text(encoding="utf-8")) if MEASURE.exists() else {}
    tm = C.TargetMeasure.from_config(cfg)
    if source_tags:
        tm.resolve_source_tags(source_tags, cluster_tags)
    elif any("source_tag" in ov.get("match", {}) for ov in tm.weight_overrides):
        print("[readout] NOTE: snapshot carries no source_tags — source-tag overrides shown "
              "UNRESOLVED (no-op). Re-run stage_first_release to persist tags for faithful "
              "target marginals.", flush=True)
    observed = sorted({(by_id_family[i], cluster_tags[i]) for i in cluster_tags})
    feasible = C.build_feasible_cells(observed, flavors, styles)
    # Solve any target_share override the SAME way the emission drove it (post source-tag resolve),
    # so the target column reflects the absolute-share measure, not the unsolved no-op weighting.
    tm.solve_target_shares(feasible)
    w = np.array([tm.weight(c) for c in feasible], dtype=np.float64)
    tot = w.sum()
    tfrac = w / tot if tot > 0 else w
    typ, fla, sty = Counter(), Counter(), Counter()
    for c, f in zip(feasible, tfrac):
        typ[c[0]] += f
        fla[c[2]] += f
        sty[c[3]] += f
    return {"type": dict(typ), "flavor": dict(fla), "style": dict(sty),
            "n_feasible": len(feasible)}


def realized_marginals(rows: list):
    n = len(rows)
    typ, fla, sty = Counter(), Counter(), Counter()
    for r in rows:
        typ[r["type"]] += 1
        fla[r["palette_flavor"]] += 1
        sty[r["render_style"]] += 1
    def frac(c):
        return {k: v / n for k, v in c.items()} if n else {}
    return {"type": frac(typ), "flavor": frac(fla), "style": frac(sty), "n": n}


# --------------------------------------------------------------------------- #
# Fate-stratified reject autopsy sheet.
# --------------------------------------------------------------------------- #
def _thumb(jpg_rel, tw, th):
    if not jpg_rel:
        return Image.new("RGB", (tw, th), (40, 40, 44))
    p = ROOT / jpg_rel
    if not p.exists():
        return Image.new("RGB", (tw, th), (40, 40, 44))
    with Image.open(p) as im:
        return im.convert("RGB").resize((tw, th), Image.LANCZOS)


def fate_sheet(rows: list, out_png: Path, per_band: int = 8):
    """Rows stratified by fate: release-eligible / pool-only (below release floor) /
    floor-rejected / errored — a visual sample of admissions AND rejects."""
    def rf(r):
        return release_floor(r["render_style"])
    bands = [
        ("release-eligible (≥ head release floor)",
         [r for r in rows if r.get("passed") and (r.get("p_ge3") or 0) >= rf(r)]),
        ("pool inventory (passed pool floor, below release floor)",
         [r for r in rows if r.get("passed") and (r.get("p_ge3") or 0) < rf(r)]),
        ("floor-rejected (below pool floor)",
         [r for r in rows if not r.get("passed") and not r.get("error")]),
        ("render error",
         [r for r in rows if r.get("error")]),
    ]
    tw, th, pad, lh, hdr = 240, 135, 8, 30, 26
    rng = np.random.default_rng(0)
    picks = []
    for name, band in bands:
        band = [r for r in band if r.get("jpg")]
        band_sorted = sorted(band, key=lambda r: -(r.get("p_ge3") or 0))
        if len(band_sorted) > per_band:
            # top-2 + a spread sample across the rest, so the sheet shows the band's range
            idx = np.linspace(0, len(band_sorted) - 1, per_band).round().astype(int)
            sample = [band_sorted[i] for i in sorted(set(idx.tolist()))]
        else:
            sample = band_sorted
        picks.append((name, len(band), sample))
    ncol = per_band
    nrow = len(bands)
    W = 300 + ncol * (tw + pad) + pad
    H = hdr + nrow * (th + lh + hdr) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), "first release — fate-stratified autopsy (admissions AND rejects, deploy fidelity)",
           fill=(235, 235, 235), font=_font(14))
    y = hdr + 6
    for name, total, sample in picks:
        d.text((pad, y), f"{name}  (n={total})", fill=(200, 215, 235), font=_font(13))
        yy = y + 18
        for j, r in enumerate(sample):
            x = 12 + j * (tw + pad)
            sheet.paste(_thumb(r.get("jpg"), tw, th), (x, yy))
            p3 = r.get("p_ge3")
            p3s = f"{p3:.3f}" if p3 is not None else "ERR"
            d.text((x + 2, yy + th + 1), f"{r['id']} {r.get('head', '?')[:4]} p3={p3s}",
                   fill=(205, 205, 215), font=_font(9))
            d.text((x + 2, yy + th + 13), f"{r['type'][:14]} {r['render_style'][:14]}",
                   fill=(170, 175, 190), font=_font(9))
        y += th + lh + hdr
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return [(n, t) for n, t, _ in picks]


def hue_chroma_png(gated: list, out_png: Path):
    hue = np.zeros(12)
    chroma = np.zeros(8)
    nh = nc = 0
    for r in gated:
        rp = r.get("realized_palette") or {}
        if rp.get("hue_hist"):
            hue += np.array(rp["hue_hist"]); nh += 1
        if rp.get("chroma_hist"):
            chroma += np.array(rp["chroma_hist"]); nc += 1
    hue = hue / nh if nh else hue
    chroma = chroma / nc if nc else chroma
    W, H, pad = 900, 420, 40
    im = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(im)
    d.text((pad, 8), f"realized hue (chroma-weighted) + chroma histograms, pooled over "
           f"{nh} gated wallpapers", fill=(230, 230, 230), font=_font(13))
    # hue bars (colored by hue)
    import colorsys
    bw = (W - 2 * pad) / 12
    base = 60
    mh = hue.max() or 1
    for i, v in enumerate(hue):
        x = pad + i * bw
        bh = int(120 * v / mh)
        rr, gg, bb = colorsys.hsv_to_rgb((i + 0.5) / 12, 0.85, 0.95)
        d.rectangle([x + 2, base + 120 - bh, x + bw - 2, base + 120],
                    fill=(int(rr * 255), int(gg * 255), int(bb * 255)))
        d.text((x + 2, base + 122), HUE_NAMES[i][:5], fill=(190, 190, 195), font=_font(9))
    # chroma bars
    cb = (W - 2 * pad) / 8
    base2 = 260
    mc = chroma.max() or 1
    for i, v in enumerate(chroma):
        x = pad + i * cb
        bh = int(110 * v / mc)
        g = int(80 + 175 * i / 7)
        d.rectangle([x + 2, base2 + 110 - bh, x + cb - 2, base2 + 110], fill=(g, g, 210))
        d.text((x + 2, base2 + 112), f"{i/8:.2f}", fill=(190, 190, 195), font=_font(9))
    d.text((pad, base - 18), "hue (12 bins)", fill=(200, 200, 205), font=_font(11))
    d.text((pad, base2 - 18), "chroma (8 bins, 0→1)", fill=(200, 200, 205), font=_font(11))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return hue.tolist(), chroma.tolist(), nh


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-n", type=int, default=50)
    args = ap.parse_args()

    rows = load_pool()
    gated = [r for r in rows if r.get("passed")]
    intake = json.loads((OUT / "intake.json").read_text(encoding="utf-8"))
    tags = intake["cluster_tags"]
    # id -> family from the tag prefix (tag = "<family>#<k>")
    by_id_family = {i: t.rsplit("#", 1)[0] for i, t in tags.items()}
    flavors = sorted({r["palette_flavor"] for r in rows}) or ["k16:1"]

    n_att = len(rows)
    n_pass = len(gated)
    n_err = sum(1 for r in rows if r.get("error"))
    n_floor = sum(1 for r in rows if not r.get("passed") and not r.get("error"))
    recon_ok = (n_att == n_pass + n_floor + n_err)

    # release-eligible subset + a reconstructed release (matches the driver's greedy select)
    rel_elig = [r for r in gated if (r.get("p_ge3") or 0) >= release_floor(r["render_style"])]
    entries = [{"id": r["id"], "type": r["type"], "cluster": r["morph_cluster"],
                "flavor": r["palette_flavor"], "style": r["render_style"],
                "score": r["p_ge3"], "emb": None, "_rec": r} for r in rel_elig]
    selected, _log = SEL.greedy_select(entries, args.release_n)
    rel_rows = [e["_rec"] for e in selected]

    tgt = target_marginals(tags, by_id_family, flavors, STYLES, intake.get("source_tags"))
    real_g = realized_marginals(gated)
    real_r = realized_marginals(rel_rows)

    # niche (full cell) size distribution over gated — the percentile-degeneracy check
    niche = Counter(tuple(r["cell"]) for r in gated)
    niche_sizes = Counter(niche.values())
    n_singleton = sum(1 for _, s in niche.items() if s == 1)

    # cell reachability
    tc_pairs = {(by_id_family[i], t) for i, t in tags.items()}          # feasible (type,cluster)
    tc_filled = {(r["type"], r["morph_cluster"]) for r in gated}
    joint_filled = len(niche)

    # strange inventory above the mining release floor
    strange = [r for r in gated if r["render_style"] != "smooth"]
    strange_rel = [r for r in strange if (r.get("p_ge3") or 0) >= MN_RELEASE_FLOOR]

    hue, chroma, nh = hue_chroma_png(gated, OUT / "hue_chroma.png")
    fate = fate_sheet(rows, OUT / "reject_autopsy_sheet.png")

    # ---- markdown ---------------------------------------------------------- #
    L = []
    w = L.append
    done = (OUT / "summary.json").exists()
    w("# First release — supplementary readout\n")
    w(f"Status: **{'COMPLETE' if done else 'IN PROGRESS'}** — pool has **{n_att}** colorize "
      f"attempts so far ({n_pass} gated). Reads only the durable pool log + snapshot; "
      f"complements the driver's `out/first_release_report.md`.\n")

    w("## 0. Per-stage reconciliation\n")
    w(f"- attempts (found) **{n_att}** == passed (written) **{n_pass}** + floor-dropped "
      f"**{n_floor}** + errored **{n_err}** → **{'OK' if recon_ok else 'MISMATCH!'}**")
    w(f"- pool pass rate {n_pass/n_att:.1%}" if n_att else "")
    w("")

    w("## 1. Realized shares vs the target measure (order book)\n")
    w("Target = the measure's **cell-normalized** target fraction over the full feasible "
      f"support ({tgt['n_feasible']} cells) — what the DeficitModel actually drives (cluster-"
      "count-weighted, so a type with more morph clusters draws proportionally more unless its "
      "type-weight offsets it). Realized = the gated pool / the reconstructed release.\n")
    w("### per fractal_type\n")
    w("| type | target | gated | release |")
    w("|---|--:|--:|--:|")
    for t in sorted(set(tgt["type"]) | set(real_g["type"]) | set(real_r["type"])):
        w(f"| {t} | {tgt['type'].get(t,0):.1%} | {real_g['type'].get(t,0):.1%} | "
          f"{real_r['type'].get(t,0):.1%} |")
    w("\n### render_style marginal (gated)\n")
    w("| style | target | gated | release |")
    w("|---|--:|--:|--:|")
    for s in STYLES:
        w(f"| {s} | {tgt['style'].get(s,0):.1%} | {real_g['style'].get(s,0):.1%} | "
          f"{real_r['style'].get(s,0):.1%} |")
    w("")

    w("## 2. Cell reachability at library scale\n")
    w(f"- feasible (type,cluster) pairs: **{len(tc_pairs)}**; produced ≥1 gated wallpaper: "
      f"**{len(tc_filled)}** ({len(tc_filled)/len(tc_pairs):.1%})")
    w(f"- distinct joint cells (type,cluster,flavor,style) filled in the gated pool: "
      f"**{joint_filled}**")
    w(f"- (with one colorize per location, each location fills exactly one joint cell, so "
      f"joint-cell coverage tracks the per-location deficit pick, not exhaustive cell sweep)")
    w("")

    w("## 3. Per-niche percentile health at scale\n")
    w(f"Niche = full descriptor cell. Gated pool occupies **{len(niche)}** niches; "
      f"**{n_singleton}** are singletons (**{n_singleton/len(niche):.1%}**).\n"
      if niche else "no gated rows yet.\n")
    w("| niche size | # niches |")
    w("|--:|--:|")
    for size, cnt in sorted(niche_sizes.items()):
        w(f"| {size} | {cnt} |")
    degenerate = n_singleton / max(1, len(niche)) > 0.7
    reading = ("still largely degenerate (singletons → percentile 1.0; selection tie-breaks "
               "on absolute p_ge3)" if degenerate
               else "meaningfully populated (percentile discriminates within niches)")
    w(f"\n**Reading:** at ~one colorize per location the within-cell percentile is {reading}.\n")

    w("## 4. Strange inventory (mining head)\n")
    w(f"- strange (non-smooth) gated wallpapers: **{len(strange)}**")
    w(f"- above the {MN_RELEASE_FLOOR} mining release floor: **{len(strange_rel)}** "
      f"(toward mining-head calibration)")
    w("")

    w("## 5. Realized hue / chroma (pooled over the gated pool)\n")
    w(f"Accumulated over **{nh}** gated wallpapers (chroma-weighted hue histogram + chroma "
      f"histogram). See `out/first_release/hue_chroma.png`.\n")
    w("| hue bin | share |   | chroma bin | share |")
    w("|---|--:|---|---|--:|")
    for i in range(12):
        cb = f"{i/8:.2f}–{(i+1)/8:.2f}" if i < 8 else ""
        cv = f"{chroma[i]:.1%}" if i < 8 else ""
        hh = f"{hue[i]:.1%}" if sum(hue) else "—"
        w(f"| {HUE_NAMES[i]} | {hh} |   | {cb} | {cv} |")
    w("")

    w("## 6. Reject autopsy — fate-stratified sheet\n")
    w("`out/first_release/reject_autopsy_sheet.png` — a visual sample across every fate band:\n")
    for name, total in fate:
        w(f"- {name}: {total}")
    w("")

    REPORT.write_text("\n".join(x for x in L if x is not None), encoding="utf-8")
    print(f"[readout] reconciliation {'OK' if recon_ok else 'MISMATCH'} "
          f"({n_att}={n_pass}+{n_floor}+{n_err})", flush=True)
    print(f"[readout] wrote {REPORT.relative_to(ROOT)}, hue_chroma.png, "
          f"reject_autopsy_sheet.png", flush=True)


if __name__ == "__main__":
    main()
