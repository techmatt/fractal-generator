"""report.py — emission-v1 report + contact sheets.

Writes `out/emission_v1_report.md` (path anchored at repo root per the prompt) plus a
release contact sheet and a pool contact sheet grouped by niche, and a machine-readable
`summary.json`. Kept separate from the driver so the report can be rebuilt from the
durable pool log without re-colorizing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.emission import descriptor as D

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "out" / "emission_v1_report.md"


def _v1_release_reconstruction(v1_pool: Path, release_n: int):
    """Reconstruct what the v1 run (no release floors) would ship from its durable pool:
    greedy_select over ALL gated rows (the v1 behavior). Returns (rows_by_id, selected_ids,
    p_ge3_by_id) or None if the v1 pool is absent. Used only for the v2 side-by-side."""
    if not v1_pool.exists():
        return None
    from tools.emission import selection as SEL
    rows = [json.loads(l) for l in v1_pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    gated = [r for r in rows if r.get("passed")]
    if not gated:
        return None
    entries = [{
        "id": r["id"], "type": r["type"], "cluster": r["morph_cluster"],
        "flavor": r["palette_flavor"], "style": r["render_style"],
        "score": r["p_ge3"], "emb": None, "_rec": r,
    } for r in gated]
    selected, _log = SEL.greedy_select(entries, release_n)
    return {
        "by_id": {r["id"]: r for r in gated},
        "selected": [e["id"] for e in selected],
        "n_gated": len(gated),
    }


def _font(sz):
    for name in ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(jpg_rel: str, tw: int, th: int):
    if not jpg_rel:
        return Image.new("RGB", (tw, th), (40, 40, 44))
    p = ROOT / jpg_rel
    if not p.exists():
        return Image.new("RGB", (tw, th), (40, 40, 44))
    with Image.open(p) as im:
        return im.convert("RGB").resize((tw, th), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Contact sheets.
# --------------------------------------------------------------------------- #
def release_sheet(selected: list, sel_log: list, out_png: Path, cols: int = 4):
    tw, th, pad, lh, head = 300, 169, 8, 30, 34
    n = len(selected)
    rows = (n + cols - 1) // cols
    W = pad + cols * (tw + pad)
    H = head + rows * (th + lh + pad) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"emission v1 — release ({n} wallpapers), greedy max-marginal-gain",
           fill=(235, 235, 235), font=_font(15))
    logi = {l["id"]: l for l in sel_log}
    for i, e in enumerate(selected):
        r = e["_rec"]
        cx = pad + (i % cols) * (tw + pad)
        cy = head + (i // cols) * (th + lh + pad)
        sheet.paste(_thumb(r["jpg"], tw, th), (cx, cy))
        L = logi.get(r["id"], {})
        d.text((cx + 2, cy + th + 2),
               f"{i+1}. {r['type']} {r['morph_cluster']}", fill=(220, 220, 160), font=_font(11))
        d.text((cx + 2, cy + th + 15),
               f"{r['palette_flavor']}/{r['render_style']} p3={r['p_ge3']:.2f} "
               f"niche%={L.get('niche_pct', 0):.2f}", fill=(200, 210, 220), font=_font(10))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def pool_sheet(gated: list, out_png: Path, max_per_niche: int = 6):
    tw, th, pad, lh, head = 176, 99, 6, 16, 34
    by_niche = defaultdict(list)
    for r in gated:
        by_niche[tuple(r["cell"])].append(r)
    niches = sorted(by_niche)
    ncol = max((min(len(v), max_per_niche) for v in by_niche.values()), default=1)
    W = 260 + ncol * (tw + pad) + pad
    H = head + len(niches) * (th + lh) + pad
    sheet = Image.new("RGB", (W, H), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), f"emission v1 — gated pool by niche ({len(gated)} wallpapers, "
           f"{len(niches)} occupied cells)", fill=(235, 235, 235), font=_font(14))
    for i, niche in enumerate(niches):
        y = head + i * (th + lh)
        t, cl, f, s = niche
        d.text((pad, y + th // 2), f"{t}/{cl}\n{f}/{s}", fill=(180, 200, 230), font=_font(10))
        for j, r in enumerate(sorted(by_niche[niche], key=lambda z: -z["p_ge3"])[:max_per_niche]):
            x = 256 + j * (tw + pad)
            sheet.paste(_thumb(r["jpg"], tw, th), (x, y))
            d.text((x + 2, y + th + 1), f"{r['id']} {r['p_ge3']:.2f}",
                   fill=(200, 200, 210), font=_font(9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


# --------------------------------------------------------------------------- #
# Report body.
# --------------------------------------------------------------------------- #
def write_report(eng, selected: list, sel_log: list, rel_paths: list):
    out = eng.out
    gated = eng.pool.gated()
    allrows = eng.pool.rows
    n_att = len(allrows)
    occ = eng.model.occupancy()

    # sheets
    rel_png = out / "release_sheet.png"
    pool_png = out / "pool_sheet.png"
    release_sheet(selected, sel_log, rel_png)
    pool_sheet(gated, pool_png)

    # morph-cluster count among admitted
    clusters = Counter()
    for i, r in eng.by_id.items():
        clusters[r["family"]] += 0
    n_clusters_by_type = defaultdict(set)
    for lid, tag in eng.cluster_tags.items():
        n_clusters_by_type[eng.by_id[lid]["family"]].add(tag)
    total_clusters = len({t for t in eng.cluster_tags.values()})

    # realized vs nominal surplus (per head — smooth via wallpaper head, strange via mining)
    n_pass = len(gated)
    pass_rate = n_pass / n_att if n_att else 0.0
    n_err = sum(1 for r in allrows if r.get("error"))
    wp_gated = [r for r in gated if r.get("head") == "wallpaper"]
    mn_gated = [r for r in gated if r.get("head") == "mining"]
    wp_also = sum(1 for r in wp_gated if (r["p_ge3"] or 0) >= 0.90)
    mn_also = sum(1 for r in mn_gated if (r["p_ge3"] or 0) >= 0.50)
    also_090 = wp_also

    # colorizer choice — flavor + style distribution vs uniform-random baseline
    chosen_flavor = Counter(r["palette_flavor"] for r in allrows)
    chosen_style = Counter(r["render_style"] for r in allrows)
    n_flavors = len(eng.flavors)
    n_styles = len(eng.styles)
    uniform_flavor = n_att / n_flavors if n_flavors else 0
    uniform_style = n_att / n_styles if n_styles else 0

    ledger_labels = [str(l.relative_to(ROOT)) for l in getattr(eng, "ledgers", [eng.ledger])]
    reach = eng.ranker_reach() if hasattr(eng, "ranker_reach") else {}
    short = getattr(eng, "release_short_fill", {})

    L = []
    L.append("# Emission — diversity-aware emission (deficit colorize + ranker-ordered intake "
             "+ per-head release floors)\n")
    L.append("Source ledger(s): " + ", ".join(f"`{x}`" for x in ledger_labels) + ".\n")
    L.append(f"Location ranker (pref_loc_v0, **{eng.ranker_mode}**) ORDERS the colorize queue "
             f"(order, not filter — diversity supply untouched). Pool floors (permissive): "
             f"wallpaper **{eng.floor}** / mining **{eng.mining_floor}**. **Release floors** "
             f"(per head, distinct): wallpaper **{eng.release_floor}** / mining "
             f"**{eng.mining_release_floor}**. Release N=**{eng.release_n}** · target "
             f"**{eng.target_gated}** release-eligible (post-floor surplus).\n")

    L.append("## Intake — morph clusters among admitted locations\n")
    L.append(f"- **{len(eng.rows)}** admitted locations "
             f"(current-decode ∧ decoded_class==3 ∧ guard_pass ∧ distinct)")
    L.append(f"- **{total_clusters}** morph clusters (within-type, cos>{D.NEAR_DUP_THRESHOLD}) "
             f"across **{len(n_clusters_by_type)}** fractal types:")
    for t in sorted(n_clusters_by_type):
        n_loc = sum(1 for r in eng.rows if r["family"] == t)
        L.append(f"  - `{t}`: {n_loc} locations → {len(n_clusters_by_type[t])} clusters")
    L.append("")

    L.append("## Niche occupancy + deficit (before → after)\n")
    L.append(f"- feasible cells: **{occ['feasible_cells']}** "
             f"((type,cluster) × {n_flavors} flavors × {n_styles} styles)")
    L.append(f"- BEFORE (empty pool): 0 populated, deficit = uniform target over all "
             f"{occ['feasible_cells']} feasible cells")
    L.append(f"- AFTER: **{occ['populated_cells']}** distinct cells populated by the "
             f"{n_pass}-wallpaper gated pool; **{occ['capped']}** cells hit the attempt cap "
             f"and left support")
    L.append(f"- **{occ['populated_cells']}** distinct cells did the {n_pass}-surplus "
             f"populate (out of {occ['feasible_cells']} feasible).")
    # per-axis marginal occupancy of the gated pool (which axis values actually filled)
    ax_pop = {ax: Counter() for ax in ("type", "morph_cluster", "palette_flavor", "render_style")}
    for r in gated:
        for ax in ax_pop:
            ax_pop[ax][r[ax]] += 1
    L.append(f"- axis coverage in the gated pool: "
             f"**{len(ax_pop['type'])}** types · **{len(ax_pop['morph_cluster'])}** morph clusters · "
             f"**{len(ax_pop['palette_flavor'])}**/{n_flavors} palette flavors · "
             f"**{len(ax_pop['render_style'])}**/{n_styles} render styles")
    L.append(f"  - render styles present: "
             + ", ".join(f"{s}×{c}" for s, c in ax_pop['render_style'].most_common()))
    L.append("")

    # per-head release-eligibility (the new floors), and inventory banked below them.
    wp_rel = sum(1 for r in wp_gated if (r["p_ge3"] or 0) >= eng.release_floor)
    mn_rel = sum(1 for r in mn_gated if (r["p_ge3"] or 0) >= eng.mining_release_floor)
    n_rel = wp_rel + mn_rel

    L.append("## Pool inventory + per-head release floors\n")
    L.append(f"Render styles route to two heads: **smooth → wallpaper head** "
             f"(pool floor {eng.floor}, **release floor {eng.release_floor}**); "
             f"**strange → mining head** (pool floor {eng.mining_floor}, **release floor "
             f"{eng.mining_release_floor}**). Quality is only compared within a niche, which "
             f"pins the style/head, so the heads never mix. Pool admission is permissive "
             f"(weak wallpapers persist as inventory); SELECTION only draws above the release "
             f"floor.\n")
    L.append(f"- attempts: **{n_att}** · pool-admitted (gated): **{n_pass}** → pool pass rate "
             f"**{pass_rate:.1%}** · render errors: {n_err}")
    L.append("")
    L.append("| head | pool-admitted | release-eligible | inventory (below release floor) |")
    L.append("|---|--:|--:|--:|")
    L.append(f"| wallpaper (smooth, rel≥{eng.release_floor}) | {len(wp_gated)} | {wp_rel} | "
             f"{len(wp_gated) - wp_rel} |")
    L.append(f"| mining (strange, rel≥{eng.mining_release_floor}) | {len(mn_gated)} | {mn_rel} | "
             f"{len(mn_gated) - mn_rel} |")
    L.append(f"| **total** | **{n_pass}** | **{n_rel}** | **{n_pass - n_rel}** |")
    L.append("")
    L.append(f"**{n_pass - n_rel}/{n_pass}** pool wallpapers are banked as inventory below their "
             f"head's release floor — exactly the weak tiles the v1 permissive-only bar would "
             f"have let compete for a release slot. The colorize targeted **{eng.target_gated}** "
             f"release-eligible (post-floor) and reached **{n_rel}**.\n")

    if reach:
        L.append("## Ranker reach — did ranked intake concentrate budget on good locations?\n")
        L.append("Admitted locations ordered by pref_loc_v0 score (desc); 'reach' = the deepest "
                 "rank the colorize actually touched. If ranked intake works, colorize fills its "
                 "surplus from the TOP of the ordering and never has to reach deep.\n")
        L.append(f"- {reach['n_locations']} admitted locations; **{reach['n_attempted']}** got a "
                 f"colorize attempt, reaching rank **{reach['deepest_attempted_rank']}** "
                 f"(top {reach['deepest_attempted_pct']:.0%} of the ordering).")
        L.append(f"- **{reach['n_release_locs']}** locations contributed a release-eligible "
                 f"wallpaper, the deepest at rank **{reach['deepest_release_rank']}** "
                 f"(top {reach['deepest_release_pct']:.0%}).")
        L.append(f"- reading: the surplus was filled within the top "
                 f"**{reach['deepest_release_pct']:.0%}** of ranker-ordered locations "
                 f"{'— ranked intake concentrated budget on the good end.' if reach['deepest_release_pct'] < 0.9 else '(reached deep — pool is quality-thin, not a ranking failure).'}\n")

    L.append("## Colorizer choice — deficit-driven palette/style spread\n")
    L.append(f"Chosen palette-flavor distribution over {n_att} colorize attempts vs the "
             f"uniform-random expectation ({uniform_flavor:.1f}/flavor):\n")
    L.append("| palette flavor | chosen | uniform-random |")
    L.append("|---|---:|---:|")
    for f, c in chosen_flavor.most_common():
        L.append(f"| {f} | {c} | {uniform_flavor:.1f} |")
    L.append("")
    L.append(f"Render-style distribution (uniform-random {uniform_style:.1f}/style):\n")
    L.append("| render style | chosen |")
    L.append("|---|---:|")
    for s, c in chosen_style.most_common():
        L.append(f"| {s} | {c} |")
    L.append("")

    fill_note = ""
    if short.get("short_by"):
        fill_note = (f" — **SHORT-FILL {len(selected)}/{eng.release_n}**: only "
                     f"{short['eligible']} pool rows clear the release floors; shipping fewer "
                     f"rather than dipping below the floor")
    L.append(f"## Release selection — {len(selected)} picks (greedy max-marginal-gain){fill_note}\n")
    L.append("Selection draws ONLY from the release-eligible subset (per-head floor). Marginal "
             "gain = niche-relative quality (within-niche p_ge3 percentile) × coverage gain "
             "(1 − max similarity to already-selected under the per-axis kernel). `rk%` = the "
             "location's pref_loc_v0 percentile among admitted; `nearest` = the closest "
             "already-selected wallpaper (displacement).\n")
    L.append("| # | id | type/cluster | flavor/style | p_ge3 | niche% | rk% | cov.gain | nearest (sim) |")
    L.append("|--:|---|---|---|--:|--:|--:|--:|---|")
    for i, (e, l) in enumerate(zip(selected, sel_log), 1):
        r = e["_rec"]
        near = f"{l['nearest_selected']} ({l['nearest_sim']:.2f})" if l["nearest_selected"] else "—"
        rkp = eng.ranker_pct.get(r["location_id"])
        rkp_s = f"{rkp:.2f}" if rkp is not None else "—"
        L.append(f"| {i} | {r['id']} | {r['type']}/{r['morph_cluster']} | "
                 f"{r['palette_flavor']}/{r['render_style']} | {r['p_ge3']:.3f} | "
                 f"{l['niche_pct']:.2f} | {rkp_s} | {l['coverage_gain']:.2f} | {near} |")
    L.append("")

    # v1 side-by-side: reconstruct the v1 release (no release floors) from its durable pool.
    v1 = _v1_release_reconstruction(ROOT / "out" / "emission_v1" / "pool_log.jsonl", eng.release_n)
    if v1 and eng.out.name != "emission_v1":
        L.append("### vs the v1 release (no release floors) — side-by-side\n")
        L.append("v1 selected from ALL gated pool rows (permissive floor only). Reconstructed "
                 "here by the same greedy select over the durable v1 pool, annotated with which "
                 "picks would now fall BELOW their head's release floor (→ inventory, not a "
                 "release).\n")
        wp_rf, mn_rf = eng.release_floor, eng.mining_release_floor
        n_below = 0
        L.append("| v1 pick | type/style | p_ge3 | ≥ release floor? |")
        L.append("|---|---|--:|---|")
        for iid in v1["selected"]:
            r = v1["by_id"][iid]
            style = r["render_style"]
            rf = wp_rf if style == "smooth" else mn_rf
            p = r["p_ge3"] or 0.0
            ok = p >= rf
            n_below += 0 if ok else 1
            verdict = f"✓ ({p:.2f} ≥ {rf})" if ok else f"✗ {p:.2f} BELOW {rf} → inventory"
            L.append(f"| {iid} | {r['type']}/{style} | {p:.3f} | {verdict} |")
        L.append("")
        L.append(f"**{n_below}/{len(v1['selected'])}** v1 picks now drop to inventory under the "
                 f"release floors — the sub-floor tiles the v1 permissive-only bar shipped.\n")
    L.append("## Contact sheets\n")
    L.append(f"- `{rel_png.relative_to(ROOT)}` — the {len(selected)}-wallpaper release")
    L.append(f"- `{pool_png.relative_to(ROOT)}` — the gated pool grouped by niche\n")

    report_path = getattr(eng, "report_path", REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")

    summary = {
        "ledgers": ledger_labels,
        "n_admitted": len(eng.rows), "n_morph_clusters": total_clusters,
        "feasible_cells": occ["feasible_cells"], "populated_cells": occ["populated_cells"],
        "capped_cells": occ["capped"],
        "attempts": n_att, "gated": n_pass, "pass_rate": round(pass_rate, 4),
        "gated_also_090": also_090, "render_errors": n_err,
        "release_eligible": n_rel, "release_n": len(selected), "release_rendered": len(rel_paths),
        "pool_floor": eng.floor, "mining_pool_floor": eng.mining_floor,
        "release_floor": eng.release_floor, "mining_release_floor": eng.mining_release_floor,
        "loc_ranker": eng.ranker_mode, "ranker_reach": reach, "short_fill": short,
        "palette_ranker": selected[0]["_rec"]["ranker"] if selected else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[report] {report_path}\n[report] {rel_png}\n[report] {pool_png}\n"
          f"[report] {out/'summary.json'}", flush=True)
    return summary
