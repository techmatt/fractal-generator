#!/usr/bin/env python
"""Finalize a descent-ablation campaign: per-arm + overview contact sheets, probes.json,
report.md. Consumes only the durable per-arm pool.jsonl/walks.jsonl + the tile PNG
previews the Rust binary already wrote at a fixed neutral palette (color held constant
across arms — geometry-only comparison). No re-render, no full-res.

Probes are navigation aids; eye judgment is the verdict. Persisted per-candidate fields
are idx/walk/depth/target_depth/root_src/branch/placement/focus_score/cx/cy/fw/occ/png —
so the terminal stat we report is **occupancy** (real, from pool.jsonl) plus tile-derived
luma proxies. interior_frac/esc-median/spread are NOT persisted (leaving pool.jsonl
schema untouched preserves the A0 byte-identical invariant), so they are not reported.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GRID_COLS = 10
THUMB_W, THUMB_H = 192, 108     # per-tile size in the composed grid
FEAT_W, FEAT_H = 32, 18         # diversity-proxy feature resolution
MAX_TILES = 100                 # per-arm terminal tile cap (stratified by depth)


def load_jsonl(p):
    rows = []
    if not Path(p).exists():
        return rows
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(xs):
    a = np.asarray([x for x in xs if x is not None and math.isfinite(x)], dtype=float)
    if a.size == 0:
        return dict(n=0, mean=None, sd=None, min=None, max=None, med=None)
    return dict(n=int(a.size), mean=float(a.mean()), sd=float(a.std()),
                min=float(a.min()), max=float(a.max()), med=float(np.median(a)))


def hist(xs):
    h = {}
    for x in xs:
        h[str(x)] = h.get(str(x), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (len(kv[0]), kv[0])))


def terminals(pool_rows):
    """Deepest accepted candidate per walk (tie -> largest idx)."""
    best = {}
    for r in pool_rows:
        w = r["walk"]
        cur = best.get(w)
        if cur is None or (r["depth"], r["idx"]) > (cur["depth"], cur["idx"]):
            best[w] = r
    return [best[w] for w in sorted(best)]


def tile_feature(arm_dir, png_rel):
    """Load a tile PNG -> (32x18 gray feature vector in [0,1], mean_luma, luma_sd)."""
    p = arm_dir / png_rel
    if not p.exists():
        return None, None, None
    im = Image.open(p).convert("L")
    small = im.resize((FEAT_W, FEAT_H), Image.BILINEAR)
    v = np.asarray(small, dtype=np.float32).flatten() / 255.0
    full = np.asarray(im, dtype=np.float32) / 255.0
    return v, float(full.mean()), float(full.std())


def mean_pairwise_l2(feats):
    if len(feats) < 2:
        return None
    X = np.stack(feats)  # (n, d)
    # ||xi - xj|| via gram trick; mean over off-diagonal pairs.
    sq = (X * X).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = np.clip(d2, 0.0, None)
    d = np.sqrt(d2)
    n = len(feats)
    return float(d.sum() / (n * (n - 1)))


def stratified_sample(terms, cap):
    if len(terms) <= cap:
        return terms
    # bucket by reached depth, sample proportionally.
    by_d = {}
    for t in terms:
        by_d.setdefault(t["depth"], []).append(t)
    out = []
    total = len(terms)
    for d, group in sorted(by_d.items()):
        k = max(1, round(cap * len(group) / total))
        step = max(1, len(group) // k)
        out.extend(group[::step][:k])
    return out[:cap]


def compose_grid(arm_dir, terms, title):
    """Tile terminal previews into a labeled grid PNG. Returns PIL image."""
    terms = stratified_sample(terms, MAX_TILES)
    n = len(terms)
    if n == 0:
        img = Image.new("RGB", (600, 60), (16, 17, 22))
        ImageDraw.Draw(img).text((8, 20), f"{title}: no terminals", fill=(220, 120, 120))
        return img
    cols = GRID_COLS
    rows = math.ceil(n / cols)
    pad, header = 3, 22
    cw, ch = THUMB_W + pad, THUMB_H + pad + 12
    W = cols * cw + pad
    H = header + rows * ch + pad
    grid = Image.new("RGB", (W, H), (14, 15, 19))
    dr = ImageDraw.Draw(grid)
    dr.text((6, 5), title, fill=(224, 178, 74))
    for i, t in enumerate(terms):
        r, c = divmod(i, cols)
        x = pad + c * cw
        y = header + r * ch
        p = arm_dir / t["png"]
        if p.exists():
            im = Image.open(p).convert("RGB").resize((THUMB_W, THUMB_H), Image.BILINEAR)
        else:
            im = Image.new("RGB", (THUMB_W, THUMB_H), (0, 0, 0))
        grid.paste(im, (x, y))
        dr.text((x + 2, y + THUMB_H + 1), f"w{t['walk']} d{t['depth']}/{t['target_depth']}",
                fill=(150, 160, 170))
    return grid


def probe_arm(arm_dir):
    pool = load_jsonl(arm_dir / "pool.jsonl")
    walks = load_jsonl(arm_dir / "walks.jsonl")
    if not pool:
        return None
    terms = terminals(pool)
    feats, lumas, luma_sds, occs = [], [], [], []
    for t in terms:
        v, ml, ls = tile_feature(arm_dir, t["png"])
        if v is not None:
            feats.append(v)
            lumas.append(ml)
            luma_sds.append(ls)
        occs.append(t.get("occ"))
    reached = [w["reached_depth"] for w in walks]
    target = [w["target_depth"] for w in walks]
    causes = [w["cause"] for w in walks]
    return dict(
        n_walks=len(walks),
        n_walks_emitting=len(terms),
        n_candidates=len(pool),
        diversity_proxy=mean_pairwise_l2(feats),
        reached_depth=summarize(reached),
        reached_depth_hist=hist(reached),
        target_depth_hist=hist(target),
        endcause_hist=hist(causes),
        terminal_occupancy=summarize(occs),
        terminal_luma_mean=summarize(lumas),
        terminal_luma_sd=summarize(luma_sds),
        terminal_fw=summarize([t["fw"] for t in terms]),
        n_terminals=len(terms),
    )


def fmt(s, key):
    v = s.get(key)
    return "-" if v is None else (f"{v:.3f}" if abs(v) < 1e4 else f"{v:.2e}")


def finalize(campaign):
    campaign = Path(campaign)
    arms_dir = campaign / "arms"
    ledger = load_jsonl(campaign / "campaign.jsonl")
    manifest = json.loads((campaign / "manifest.json").read_text()) if (campaign / "manifest.json").exists() else {}
    led_by_arm = {r["arm"]: r for r in ledger}

    probes = {}
    arm_ids = []
    for adir in sorted(arms_dir.glob("*")):
        if not adir.is_dir():
            continue
        pr = probe_arm(adir)
        if pr is None:
            continue
        arm_id = adir.name
        arm_ids.append(arm_id)
        meta = led_by_arm.get(arm_id, {})
        pr["meta"] = {k: meta.get(k) for k in ("finder", "weights", "selection", "pct", "core", "desc", "elapsed", "timed_out")}
        probes[arm_id] = pr
        # per-arm terminal grid
        terms = terminals(load_jsonl(adir / "pool.jsonl"))
        title = f"{arm_id}  {meta.get('finder','?')}/{meta.get('selection','?')}  " \
                f"w={meta.get('weights')}  pct={meta.get('pct')}  ({meta.get('desc','')})  " \
                f"— {len(terms)} terminals"
        compose_grid(adir, terms, title).save(campaign / f"terminals_{arm_id}.png")

    (campaign / "probes.json").write_text(json.dumps(probes, indent=2))

    # ---- overview.html --------------------------------------------------------
    html = ["<!doctype html><meta charset=utf-8><title>descent ablation overview</title>",
            "<style>body{font:13px ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;margin:0 14px}",
            "h1{font-size:16px;color:#eee}h2{color:#e0b24a;font-size:14px;margin-top:26px}",
            "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #23252e;padding:3px 8px;text-align:right}",
            "th{color:#e0b24a}td.l,th.l{text-align:left}img{width:100%;border:1px solid #23252e;margin:4px 0}",
            ".core{color:#5ec07a}.probe{color:#9aa}</style>",
            f"<h1>descent ablation — {campaign.name}</h1>",
            f"<div>R={manifest.get('R')} walks/arm · seed={manifest.get('seed')} · "
            f"palette={manifest.get('preview_palette')} (held constant) · node 768px · "
            f"diversity proxy = mean pairwise L2 over {FEAT_W}x{FEAT_H} gray terminal thumbnails (higher=more diverse)</div>"]

    # summary table
    html.append("<h2>arm summary</h2><table><tr>"
                "<th class=l>arm</th><th class=l>config</th><th>walks</th><th>emit</th><th>cands</th>"
                "<th>diversity</th><th>reached med</th><th>occ mean</th><th>luma sd mean</th><th>elapsed</th></tr>")
    for a in arm_ids:
        p = probes[a]
        m = p["meta"]
        cfg = f"{m['finder']}/{m['selection']} w={m['weights']} pct={m['pct']}"
        cls = "core" if m.get("core") else "probe"
        html.append(
            f"<tr><td class='l {cls}'>{a}</td><td class=l>{cfg}<br><span class=probe>{m.get('desc','')}</span></td>"
            f"<td>{p['n_walks']}</td><td>{p['n_walks_emitting']}</td><td>{p['n_candidates']}</td>"
            f"<td><b>{fmt(p,'diversity_proxy') if p['diversity_proxy'] is not None else '-'}</b></td>"
            f"<td>{fmt(p['reached_depth'],'med')}</td>"
            f"<td>{fmt(p['terminal_occupancy'],'mean')}</td>"
            f"<td>{fmt(p['terminal_luma_sd'],'mean')}</td>"
            f"<td>{m.get('elapsed','-')}s</td></tr>")
    html.append("</table>")

    for a in arm_ids:
        p = probes[a]
        html.append(f"<h2>{a} — reached {p['reached_depth_hist']} · endcause {p['endcause_hist']}</h2>")
        html.append(f"<img src='terminals_{a}.png'>")
    (campaign / "overview.html").write_text("".join(html), encoding="utf-8")

    # ---- report.md ------------------------------------------------------------
    write_report(campaign, manifest, probes, arm_ids, led_by_arm)
    print(f"finalize: wrote probes.json, overview.html, report.md, terminals_*.png for arms {arm_ids}")


def _div(probes, a):
    return probes[a]["diversity_proxy"] if a in probes else None


def write_report(campaign, manifest, probes, arm_ids, led_by_arm):
    L = []
    L.append(f"# Descent ablation + percentile strategy — {campaign.name}\n")
    R = manifest.get("R")
    L.append(f"R = **{R}** walks/arm · seed {manifest.get('seed')} · Mandelbrot c-plane · "
             f"node 768px · preview palette `{manifest.get('preview_palette')}` (held constant across arms).\n")
    L.append("Paired design: every arm shares one frozen root seed-list + the same `--seed` + "
             "`--per-walk-rng` (matched roots + matched per-walk sub-seeds). Diversity proxy = mean "
             "pairwise L2 over 32x18 gray terminal thumbnails (higher = more diverse). "
             "**Eye judgment is the verdict; the probes are navigation aids.**\n")

    # arm table
    L.append("## Arms\n")
    L.append("| arm | finder | weights | selection | pct | walks | emit | cands | diversity | reached med | occ mean | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arm_ids:
        p = probes[a]; m = p["meta"]
        led = led_by_arm.get(a, {})
        status = "ok" if led.get("ok") else ("TIMED-OUT" if led.get("timed_out") else "partial")
        div = f"{p['diversity_proxy']:.4f}" if p["diversity_proxy"] is not None else "-"
        L.append(f"| {a} | {m['finder']} | {m['weights'] or '-'} | {m['selection']} | {m['pct'] or '-'} | "
                 f"{p['n_walks']} | {p['n_walks_emitting']} | {p['n_candidates']} | {div} | "
                 f"{fmt(p['reached_depth'],'med')} | {fmt(p['terminal_occupancy'],'mean')} | {status} |")
    L.append("")

    def dv(a):
        v = _div(probes, a)
        return v

    # driver-localization read
    L.append("## Driver-localization read (does knob A [weights] or knob B [selection] move diversity more?)\n")
    have = all(x in probes for x in ("A0", "A1", "A2", "A3"))
    if have:
        d0, d1, d2, d3 = dv("A0"), dv("A1"), dv("A2"), dv("A3")
        dA = d1 - d0   # knob A alone
        dB = d2 - d0   # knob B alone
        dAB = d3 - d0  # both
        L.append(f"- A0 (current) diversity **{d0:.4f}**")
        L.append(f"- A1 (weights -> random-heavy 0.10/0.20/0.70): **{d1:.4f}**  (delta A = {dA:+.4f})")
        L.append(f"- A2 (selection -> random-survivor): **{d2:.4f}**  (delta B = {dB:+.4f})")
        L.append(f"- A3 (both): **{d3:.4f}**  (delta A x B = {dAB:+.4f})")
        driver = "knob A (branch weights)" if abs(dA) >= abs(dB) else "knob B (selection)"
        interact = dAB - (dA + dB)
        L.append(f"\n**Read:** {driver} moves diversity more (|deltaA|={abs(dA):.4f} vs |deltaB|={abs(dB):.4f}). "
                 f"Interaction (deltaAxB - deltaA - deltaB) = {interact:+.4f} "
                 f"({'super-additive' if interact>0 else 'sub-additive/antagonistic' if interact<0 else 'additive'}).\n")
    else:
        L.append("_(insufficient core arms completed for the A0/A1/A2/A3 comparison)_\n")

    # P vs A0
    L.append("## Strategy P vs current (A4 percentile+random-survivor vs A0)\n")
    if "A4" in probes and "A0" in probes:
        d4, d0 = dv("A4"), dv("A0")
        L.append(f"- A0 diversity **{d0:.4f}**, A4 diversity **{d4:.4f}** (delta = {d4-d0:+.4f}).")
        L.append(f"- A4 reached-depth median {fmt(probes['A4']['reached_depth'],'med')} vs "
                 f"A0 {fmt(probes['A0']['reached_depth'],'med')}; "
                 f"A4 occ mean {fmt(probes['A4']['terminal_occupancy'],'mean')} vs "
                 f"A0 {fmt(probes['A0']['terminal_occupancy'],'mean')}.")
        L.append(f"\n**Read:** the percentile finder {'DIVERSIFIES' if d4>d0 else 'does NOT diversify'} vs current "
                 f"on this proxy — confirm by eye in `terminals_A4.png` vs `terminals_A0.png`.\n")
    else:
        L.append("_(A4 or A0 missing)_\n")

    # A4 vs A5
    L.append("## Does the selector mask the finder? (A4 random-survivor vs A5 least-interior, same percentile band)\n")
    if "A4" in probes and "A5" in probes:
        d4, d5 = dv("A4"), dv("A5")
        L.append(f"- A4 (random-survivor) diversity **{d4:.4f}**, A5 (least-interior) diversity **{d5:.4f}** "
                 f"(delta = {d4-d5:+.4f}).")
        L.append(f"\n**Read:** {'least-interior selection SUPPRESSES the finder diversity (selector masks it)' if d4>d5 else 'selection does not mask the percentile finder here'}.\n")
    else:
        L.append("_(A4 or A5 missing — probe A5 may have been budget-truncated)_\n")

    # look here first
    L.append("## Look here first\n")
    if probes:
        ranked = sorted([a for a in arm_ids if probes[a]["diversity_proxy"] is not None],
                        key=lambda a: probes[a]["diversity_proxy"], reverse=True)
        if ranked:
            top = ranked[0]
            L.append(f"- Most-diverse arm by proxy: **{top}** (`terminals_{top}.png`).")
        if "A4" in probes:
            L.append("- Strategy-P verdict frame: compare `terminals_A4.png` (percentile) against "
                     "`terminals_A0.png` (current) side by side.")
        L.append("- Full interactive grid + per-arm reached/endcause: `overview.html`.")
    L.append("\n_No verdict-by-classifier; geometry/diversity + eye only._\n")

    (campaign / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    finalize(sys.argv[1])
