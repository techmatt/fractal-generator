"""results_sheet.html — the visual deliverable I judge by eye.

Held-out crops sorted by predicted score (best->worst): JPG thumbnail, predicted
score, true-label badge (1/2/3), and seed|comp|palette. Header carries the metric
table (CV mean+-std + holdout) and a precision@k curve. NO quality claims — just
the numbers and the pictures.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .eval import precision_at_k

BADGE = {1: ("#b03030", "1 bad"), 2: ("#b08a30", "2 ok"), 3: ("#2e8b3d", "3 good")}


def _fmt(stat: dict) -> str:
    m, s = stat["mean"], stat["std"]
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.3f} &plusmn; {s:.3f}"


def _pk_curve_svg(labels: np.ndarray, scores: np.ndarray, width=520, height=180) -> str:
    """precision@k vs k (not-bad relevance) as an inline SVG polyline."""
    not_bad = (np.asarray(labels) >= 2).astype(int)
    n = len(scores)
    ks = list(range(1, n + 1))
    pk = [precision_at_k(not_bad, scores, k) for k in ks]
    base = not_bad.mean() if n else 0.0
    pad = 36
    iw, ih = width - 2 * pad, height - 2 * pad

    def x(k): return pad + (k - 1) / max(1, n - 1) * iw
    def y(p): return pad + (1 - p) * ih

    pts = " ".join(f"{x(k):.1f},{y(p):.1f}" for k, p in zip(ks, pk))
    base_y = y(base)
    grid = ""
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y(gv)
        grid += (f'<line x1="{pad}" y1="{gy:.1f}" x2="{width-pad}" y2="{gy:.1f}" '
                 f'stroke="#eee"/><text x="4" y="{gy+4:.1f}" font-size="10" fill="#999">{gv:.2f}</text>')
    return f'''<svg width="{width}" height="{height}">
      {grid}
      <line x1="{pad}" y1="{base_y:.1f}" x2="{width-pad}" y2="{base_y:.1f}"
            stroke="#bbb" stroke-dasharray="4 3"/>
      <text x="{width-pad-2}" y="{base_y-4:.1f}" font-size="10" fill="#999"
            text-anchor="end">base rate {base:.2f}</text>
      <polyline points="{pts}" fill="none" stroke="#2b6cb0" stroke-width="2"/>
      <text x="{pad}" y="{height-6}" font-size="11" fill="#666">k (1..{n})</text>
      <text x="{pad}" y="{pad-10}" font-size="11" fill="#666">precision@k (not-bad)</text>
    </svg>'''


def _metric_table(metrics: dict, holdout: dict) -> str:
    cv = metrics.get("cv")
    rows = []
    rows.append("<tr><th>metric</th><th>CV (mean &plusmn; std, group=seed)</th><th>holdout (85/15)</th></tr>")

    def hv(key, fmt="{:.3f}"):
        v = holdout.get(key)
        return "n/a" if v is None or not np.isfinite(v) else fmt.format(v)

    pairs = [
        ("AP not-bad (label&ge;2)", "ap_not_bad", "ap_not_bad", "crop"),
        ("AP good (label=3)", "ap_good", "ap_good", "crop"),
        ("P@10 not-bad", "p_at_10_not_bad", "p_at_10_not_bad", "crop"),
        ("P@25 not-bad", "p_at_25_not_bad", "p_at_25_not_bad", "crop"),
        ("P@10 good", "p_at_10_good", "p_at_10_good", "crop"),
        ("P@25 good", "p_at_25_good", "p_at_25_good", "crop"),
        ("loc AP not-bad (best-over-palettes)", "loc_ap_not_bad", "loc_ap_not_bad", "loc"),
        ("loc AP good", "loc_ap_good", "loc_ap_good", "loc"),
        ("loc P@10 not-bad", "loc_p_at_10_not_bad", "loc_p_at_10_not_bad", "loc"),
        ("loc P@25 not-bad", "loc_p_at_25_not_bad", "loc_p_at_25_not_bad", "loc"),
    ]
    for label, cvkey, hkey, grp in pairs:
        cvcell = "n/a"
        if cv is not None:
            block = cv["crop"] if grp == "crop" else cv["loc"]
            if cvkey in block:
                cvcell = _fmt(block[cvkey])
        rows.append(f"<tr><td>{label}</td><td>{cvcell}</td><td>{hv(hkey)}</td></tr>")
    return "<table class='metrics'>" + "".join(rows) + "</table>"


def write_results_sheet(path: Path, rows_va, scores, holdout_metrics: dict,
                        metrics: dict, cfg: dict, root: Path):
    path = Path(path)
    out_dir = path.parent
    order = np.argsort(-np.asarray(scores))
    labels = np.array([r.label for r in rows_va])

    cards = []
    for rank, i in enumerate(order):
        r = rows_va[i]
        rel = os.path.relpath(r.jpg, out_dir).replace("\\", "/")
        color, badge = BADGE[r.label]
        cards.append(f'''<div class="card">
          <div class="rank">#{rank+1}</div>
          <img src="{rel}" loading="lazy"/>
          <div class="meta">
            <span class="score">{scores[i]:.3f}</span>
            <span class="badge" style="background:{color}">{badge}</span>
          </div>
          <div class="loc">seed {r.seed} | {r.composition} | {r.palette}</div>
        </div>''')

    n = len(rows_va)
    nb = int((labels >= 2).sum())
    good = int((labels == 3).sum())
    curve = _pk_curve_svg(labels, np.asarray(scores))
    table = _metric_table(metrics, holdout_metrics)

    cfg_summary = (f"backbone={cfg['backbone']} &middot; target={cfg['target']} &middot; "
                   f"geometry={cfg['geometry']} &middot; interp={cfg['interpolation']} &middot; "
                   f"loss={cfg['loss']} &middot; sampler={cfg['sampler']} &middot; "
                   f"best_epoch={cfg.get('best_epoch','?')} &middot; "
                   f"mean={tuple(round(x,3) for x in cfg['mean'])} std={tuple(round(x,3) for x in cfg['std'])}")

    html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>v1 aesthetic classifier — held-out results</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color:#222; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 .sub {{ color:#666; font-size:12px; margin-bottom:16px; }}
 .top {{ display:flex; gap:32px; align-items:flex-start; flex-wrap:wrap; margin-bottom:24px; }}
 table.metrics {{ border-collapse: collapse; font-size:13px; }}
 table.metrics th, table.metrics td {{ border:1px solid #ddd; padding:4px 10px; text-align:left; }}
 table.metrics th {{ background:#f4f4f4; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(300px,1fr)); gap:14px; }}
 .card {{ border:1px solid #e2e2e2; border-radius:6px; overflow:hidden; background:#fafafa; position:relative; }}
 .card img {{ width:100%; display:block; aspect-ratio: 16/9; object-fit:cover; }}
 .rank {{ position:absolute; top:6px; left:6px; background:rgba(0,0,0,.6); color:#fff;
          font-size:11px; padding:1px 6px; border-radius:3px; }}
 .meta {{ display:flex; justify-content:space-between; align-items:center; padding:6px 8px; }}
 .score {{ font-weight:600; font-variant-numeric: tabular-nums; }}
 .badge {{ color:#fff; font-size:11px; padding:2px 7px; border-radius:10px; }}
 .loc {{ font-size:11px; color:#666; padding:0 8px 8px; word-break:break-all; }}
</style></head><body>
<h1>v1 aesthetic classifier — held-out crops, sorted best &rarr; worst by predicted score</h1>
<div class="sub">{cfg_summary}</div>
<div class="sub">Held-out (unseen seeds): {n} crops &middot; {nb} not-bad (label&ge;2) &middot; {good} good (label=3).
 Badge = TRUE human label. No quality claims — judge the ordering by eye.</div>
<div class="top">
  <div>{table}</div>
  <div>{curve}</div>
</div>
<div class="grid">
{''.join(cards)}
</div>
</body></html>'''
    path.write_text(html, encoding="utf-8")
    return path
