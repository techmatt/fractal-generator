"""Merge quality-passing extracted palettes + curated q2/q3 into the durable
preference-network palette pool, re-type under the binary cyclic/non_cyclic rule,
and report the (type x source) contingency table.

  uv run python tools/palettes/build_pool.py

Read-only against all existing sources (survey, harvest, curated libraries). Writes:
  * data/palettes/pool_colormaps.json   -- the merged pool (list; loads through the
    existing feature-module / colormap loaders with no loader changes).
  * data/palettes/palette_features.json  -- REGENERATED over the full merged pool
    (overwrites the old 76-entry q3-only file) so colormap.py's type lookup resolves
    for every pool entry.
  * out/palettes/pool_types.png          -- two-group (cyclic/non_cyclic) swatch grid.
  * out/palettes/pool_report.md          -- the contingency table + dedup report.

Sources / decisions (see report for the surfaced calls):
  * curated: every name scored 2 or 3 in labels/palette_scores.json, curve resolved
    from data/palettes/clean_colormaps.json.  source in {curated_q2, curated_q3}.
  * extracted: survey rows with composite >= 0.422 AND not near_dup AND not gate-`rejected`
    (== 583; the survey's net-new 603 minus the 20 gate/composite disagreements, which we
    now reject rather than keep), stored best-cycle form from
    data/wallpaper_harvest/palettes/<name>.json.  source == extracted.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette_features as pf  # noqa: E402
import build_features as bf     # noqa: E402

ROOT = pf.ROOT
SURVEY = os.path.join(ROOT, "out", "extracted_palette_survey.json")
SCORES = os.path.join(ROOT, "labels", "palette_scores.json")
CLEAN = os.path.join(ROOT, "data", "palettes", "clean_colormaps.json")
HARVEST_DIR = os.path.join(ROOT, "data", "wallpaper_harvest", "palettes")

POOL_JSON = os.path.join(ROOT, "data", "palettes", "pool_colormaps.json")
FEATURES_JSON = os.path.join(ROOT, "data", "palettes", "palette_features.json")
REPORT_MD = os.path.join(ROOT, "out", "palettes", "pool_report.md")
SWATCH_PNG = os.path.join(ROOT, "out", "palettes", "pool_types.png")

COMPOSITE_CUT = 0.422
DUP_THRESH = 0.06  # survey's provisional trajectory-distance near-dup threshold


def build_curated():
    """[{name, source: curated_q2|curated_q3, cycle, score, stops}], + diagnostics."""
    scores = json.load(open(SCORES))
    clean = {c["name"]: c for c in json.load(open(CLEAN))}
    entries, missing = [], []
    for name, sc in scores.items():
        if sc not in (2, 3):
            continue
        cm = clean.get(name)
        if cm is None:
            missing.append((name, sc))
            continue
        entries.append({
            "name": name,
            "source": "curated_q%d" % sc,
            "cycle": cm.get("cycle"),
            "score": sc,
            "stops": cm["stops"],
        })
    return entries, missing


def build_extracted():
    """[{name, source: extracted, cycle: cyclic, composite, stops}], + diagnostics.

    Cut = survey composite >= 0.422 AND not near_dup. Stops = the stored best-cycle
    form on disk (closed loop), NOT re-extracted / re-opened."""
    survey = json.load(open(SURVEY))
    rows = survey["rows"]
    raw_cut = [r for r in rows if r["composite"] >= COMPOSITE_CUT]
    # Drop near_dups AND gate-`rejected` rows (the 20 gate/composite disagreements) --
    # err toward rejecting; the pool has ample palettes. 603 -> 583.
    kept = [r for r in raw_cut if not r["near_dup"] and not r["rejected"]]

    entries = []
    for r in kept:
        p = json.load(open(os.path.join(HARVEST_DIR, r["name"] + ".json")))
        # stored form is a closed loop -> declared cyclic (closure field == "cyclic")
        entries.append({
            "name": p["name"],
            "source": "extracted",
            "cycle": "cyclic" if p.get("closed") else p.get("cycle_label"),
            "composite": r["composite"],
            "stops": p["stops"],
        })
    diag = {
        "n_raw_cut": len(raw_cut),
        "n_near_dup_in_cut": sum(1 for r in raw_cut if r["near_dup"]),
        "n_rejected_in_cut": sum(1 for r in raw_cut if r["rejected"]),
        # rows dropped specifically for the gate flag (rejected but not already near_dup)
        "n_rejected_kept_out": sum(1 for r in raw_cut if r["rejected"] and not r["near_dup"]),
        "n_kept": len(kept),
        "n_nocov_cut": sum(1 for r in rows if r["composite_nocov"] >= COMPOSITE_CUT),
    }
    return entries, diag


def contingency(pool, feats):
    """(type x source) counts with margins. Returns (table dict, sources, types)."""
    sources = ["curated_q2", "curated_q3", "extracted"]
    types = ["cyclic", "non_cyclic"]
    tab = {t: {s: 0 for s in sources} for t in types}
    for p in pool:
        t = pf.derive_type(feats[p["name"]])
        tab[t][p["source"]] += 1
    return tab, sources, types


def fmt_contingency(tab, sources, types):
    lines = []
    hdr = "%-12s " % "type" + "".join("%14s" % s for s in sources) + "%10s" % "row_tot"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    col_tot = {s: 0 for s in sources}
    for t in types:
        row = "%-12s " % t
        rt = 0
        for s in sources:
            v = tab[t][s]
            row += "%14d" % v
            col_tot[s] += v
            rt += v
        row += "%10d" % rt
        lines.append(row)
    lines.append("-" * len(hdr))
    grand = sum(col_tot.values())
    foot = "%-12s " % "col_tot" + "".join("%14d" % col_tot[s] for s in sources) + "%10d" % grand
    lines.append(foot)
    return "\n".join(lines), col_tot, grand


def dedup_report(pool, feats):
    """Trajectory-distance near-dup report (report only). Splits by curated/extracted."""
    names = [p["name"] for p in pool]
    src = {p["name"]: p["source"] for p in pool}
    D = pf.distance_matrix(feats, names)
    n = len(names)
    idx_ext = [i for i, nm in enumerate(names) if src[nm] == "extracted"]
    idx_cur = [i for i, nm in enumerate(names) if src[nm].startswith("curated")]

    # (a) within extracted: nearest OTHER extracted for each extracted
    ext_pairs = 0
    ext_min = []
    for i in idx_ext:
        d = D[i, idx_ext].copy()
        d[idx_ext.index(i)] = np.inf
        m = d.min()
        ext_min.append(m)
        if m < DUP_THRESH:
            ext_pairs += 1
    # (b) extracted<->curated: nearest curated for each extracted
    xc_close = 0
    xc_min = []
    for i in idx_ext:
        m = D[i][idx_cur].min()
        xc_min.append(m)
        if m < DUP_THRESH:
            xc_close += 1

    def pct(a):
        a = np.array(a)
        return {q: round(float(np.percentile(a, q)), 4)
                for q in (0, 10, 25, 50, 75, 90, 100)}

    return {
        "n_pool": n,
        "n_extracted": len(idx_ext),
        "n_curated": len(idx_cur),
        "within_extracted_lt_thresh": ext_pairs,
        "extracted_vs_curated_lt_thresh": xc_close,
        "within_extracted_mindist_dist": pct(ext_min),
        "extracted_vs_curated_mindist_dist": pct(xc_min),
    }


def main():
    import time
    curated, missing = build_curated()
    extracted, ediag = build_extracted()

    print("curated q2+q3: %d  (q2=%d q3=%d)"
          % (len(curated),
             sum(e["source"] == "curated_q2" for e in curated),
             sum(e["source"] == "curated_q3" for e in curated)))
    if missing:
        print("  !! q2/q3 that did NOT resolve:", missing)
    print("extracted: raw_cut(comp>=%.3f)=%d  near_dup_in_cut=%d  rejected_in_cut=%d  "
          "-> kept(& ~near_dup & ~rejected)=%d   [nocov_cut=%d]"
          % (COMPOSITE_CUT, ediag["n_raw_cut"], ediag["n_near_dup_in_cut"],
             ediag["n_rejected_in_cut"], ediag["n_kept"], ediag["n_nocov_cut"]))

    pool = curated + extracted
    names = [p["name"] for p in pool]
    assert len(names) == len(set(names)), "duplicate names in pool!"

    # --- write pool artifact (name, source, source-native quality, cycle, stops) ---
    pool_out = []
    for p in pool:
        e = {"name": p["name"], "source": p["source"]}
        if "score" in p:
            e["score"] = p["score"]
        else:
            e["composite"] = p["composite"]
        e["cycle"] = p["cycle"]
        e["stops"] = p["stops"]
        pool_out.append(e)
    os.makedirs(os.path.dirname(POOL_JSON), exist_ok=True)
    json.dump(pool_out, open(POOL_JSON, "w"), indent=1)
    print("\nwrote %s  (%d entries)" % (os.path.relpath(POOL_JSON, ROOT), len(pool_out)))

    # --- re-type + regenerate features over the FULL merged pool ---
    t0 = time.time()
    feats = pf.compute_all_features(pool)
    bf.write_features_json(pool, feats, path=FEATURES_JSON)
    print("wrote %s  (%d entries, features)" % (os.path.relpath(FEATURES_JSON, ROOT), len(feats)))

    # sanity: extracted are stored closed -> should derive cyclic. derive_type now reads the
    # TRUE authored seam gap (stop[0] vs stop[-1] in OKLab); for any residual non_cyclic,
    # contrast the old cell-centered anchor-endpoint distance with that seam gap.
    import color  # noqa
    ext_cyc = sum(1 for p in extracted if pf.derive_type(feats[p["name"]]) == "cyclic")
    ext_non_names = []
    for p in extracted:
        if pf.derive_type(feats[p["name"]]) == "non_cyclic":
            st = p["stops"]
            c0 = color.srgb_to_oklab(np.array(st[0][1]) / 255.0)
            cN = color.srgb_to_oklab(np.array(st[-1][1]) / 255.0)
            ext_non_names.append((p["name"],
                                  feats[p["name"]]["signals"]["endpoint_dist"],
                                  float(np.linalg.norm(c0 - cN))))
    ext_non = len(ext_non_names)
    print("sanity: extracted derive cyclic=%d  non_cyclic=%d %s"
          % (ext_cyc, ext_non, "(<-- surprise: closed form derived non_cyclic)" if ext_non else ""))

    # --- contingency table ---
    tab, sources, types = contingency(pool, feats)
    ctab, col_tot, grand = fmt_contingency(tab, sources, types)
    print("\n=== (type x source) contingency ===")
    print(ctab)

    # --- dedup report ---
    dd = dedup_report(pool, feats)
    dt = time.time() - t0
    print("\n=== dedup (report only, thresh=%.3f) ===" % DUP_THRESH)
    print("within-extracted pairs < thresh:", dd["within_extracted_lt_thresh"])
    print("extracted<->curated < thresh:   ", dd["extracted_vs_curated_lt_thresh"])
    print("within-extracted min-dist dist: ", dd["within_extracted_mindist_dist"])
    print("ext<->curated  min-dist dist:   ", dd["extracted_vs_curated_mindist_dist"])
    print("feature+distance time: %.2fs" % dt)

    # --- swatch grid over merged pool ---
    bf.render_swatch_grid(pool, feats, path=SWATCH_PNG)
    print("wrote", os.path.relpath(SWATCH_PNG, ROOT))

    # --- markdown report ---
    write_report(curated, extracted, missing, ediag, tab, sources, types,
                 col_tot, grand, ext_cyc, ext_non_names, dd, dt)
    print("wrote", os.path.relpath(REPORT_MD, ROOT))


def write_report(curated, extracted, missing, ediag, tab, sources, types,
                 col_tot, grand, ext_cyc, ext_non_names, dd, dt):
    ext_non = len(ext_non_names)
    L = []
    L.append("# Merged palette pool — contingency + dedup report\n")
    L.append("Pool = curated q2+q3 (`palette_scores.json` -> `clean_colormaps.json`) "
             "+ quality-passing extracted (`out/extracted_palette_survey.json` -> "
             "`data/wallpaper_harvest/palettes/`).\n")
    L.append("## Sources\n")
    L.append("- **curated q2+q3**: %d (q2=%d, q3=%d), all resolved in clean_colormaps.json."
             % (len(curated),
                sum(e["source"] == "curated_q2" for e in curated),
                sum(e["source"] == "curated_q3" for e in curated)))
    L.append("  - Unresolved q2/q3: %s" % (missing if missing else "none"))
    L.append("- **extracted**: composite >= %.3f = %d raw; minus %d survey `near_dup` "
             "and %d gate-`rejected` -> **%d** kept. Stored best-cycle form, "
             "no re-extraction." % (COMPOSITE_CUT, ediag["n_raw_cut"],
                                    ediag["n_near_dup_in_cut"], ediag["n_rejected_kept_out"],
                                    ediag["n_kept"]))
    L.append("  - Decision: %d of the raw-cut carry the gate `rejected` flag "
             "(extent<0.05 v arclen<0.3) yet clear the composite axis (the 20 gate/composite "
             "disagreements). We now **reject** them -- err toward rejecting, the pool has "
             "ample palettes. no-coverage composite at the same cut would yield %d."
             % (ediag["n_rejected_in_cut"], ediag["n_nocov_cut"]))
    L.append("")
    L.append("## Sanity: extracted derived type\n")
    L.append("Extracted are stored closed, so they should derive **cyclic**. "
             "Derived: cyclic=%d, non_cyclic=%d.\n" % (ext_cyc, ext_non))
    if ext_non_names:
        L.append("**Residual flagged (%d).** `derive_type` now dispatches on the literal "
                 "terminal-stop seam gap `stop[0]<->stop[-1]`, so any extracted still deriving "
                 "non_cyclic has a TRUE seam gap >= EPS_CYC=%.2f -- its stored 'closed' form "
                 "is not perceptually closed at the authored endpoints (not a false negative). "
                 "The `anchor_endpt` column is the old cell-centered measure, kept for "
                 "contrast.\n" % (ext_non, pf.EPS_CYC))
        L.append("| name | anchor_endpt (old test) | true seam_gap (derive_type) |")
        L.append("|---|---|---|")
        for nm, ed, seam in sorted(ext_non_names, key=lambda x: x[0]):
            L.append("| %s | %.3f | %.3f |" % (nm, ed, seam))
        L.append("")
    L.append("## Contingency table (type x source)\n")
    L.append("```")
    ctab, _, _ = fmt_contingency(tab, sources, types)
    L.append(ctab)
    L.append("```")
    L.append("Pool total: **%d**.  cyclic=%d, non_cyclic=%d.\n"
             % (grand, sum(tab["cyclic"].values()), sum(tab["non_cyclic"].values())))
    L.append("## Dedup (report only — no pruning; FPS handles diversity downstream)\n")
    L.append("Trajectory distance (`palette_distance`), threshold %.3f.\n" % DUP_THRESH)
    L.append("- within-extracted pairs < thresh: **%d** / %d extracted"
             % (dd["within_extracted_lt_thresh"], dd["n_extracted"]))
    L.append("- extracted<->curated < thresh: **%d**" % dd["extracted_vs_curated_lt_thresh"])
    L.append("- within-extracted min-dist distribution (p0..p100): %s"
             % dd["within_extracted_mindist_dist"])
    L.append("- extracted<->curated min-dist distribution (p0..p100): %s"
             % dd["extracted_vs_curated_mindist_dist"])
    L.append("\nfeature + %dx%d distance matrix: %.2fs\n" % (dd["n_pool"], dd["n_pool"], dt))
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    open(REPORT_MD, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
