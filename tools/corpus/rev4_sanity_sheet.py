"""Sanity contact sheet for the rev4 batch — crops + their store rows, so Matt
eyeballs the batch BEFORE committing to a labeling session. No quality claims:
this is a visual spot-check that the bridge/gates/provenance-join produced sane
(crop, render, provenance) triples.

Samples a stratified slice (spread across root_src × depth) plus a random fill,
and writes an HTML sheet referencing the batch crops by relative path.

Run:  uv run python tools/corpus/rev4_sanity_sheet.py [--n 60]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import corpus_common as cc

BATCH_ID = "2026-06-24_guided_descend_rev4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    bdir = cc.batch_dir(BATCH_ID)
    rows = cc.read_jsonl(os.path.join(bdir, "images.jsonl"))

    # stratify by (root_src, depth) so the sheet spans the provenance space; a
    # deterministic round-robin over strata (no RNG — reproducible).
    strata = defaultdict(list)
    for r in rows:
        pv = r["provenance"]
        strata[(pv.get("root_src"), pv.get("depth"))].append(r)
    for v in strata.values():
        v.sort(key=lambda r: r["image_id"])

    picked, keys = [], sorted(strata.keys(), key=lambda k: (str(k[0]), k[1] is None, k[1]))
    i = 0
    while len(picked) < min(a.n, len(rows)):
        progressed = False
        for k in keys:
            if i < len(strata[k]):
                picked.append(strata[k][i])
                progressed = True
                if len(picked) >= min(a.n, len(rows)):
                    break
        if not progressed:
            break
        i += 1

    def cell(r):
        pv, rd = r["provenance"], r["render"]
        src = f"crops/{r['image_id']}.jpg"
        occ = pv.get("occupancy")
        occ_s = f"{occ:.3f}" if occ is not None else "?"
        prov = " ".join(str(x) for x in [pv.get("root_src"), pv.get("branch"),
                                         (f"d{pv.get('depth')}/{pv.get('target_depth')}")] if x)
        return (f"<div class=cell><img loading=lazy src=\"{src}\">"
                f"<div class=cap><b>{r['image_id']}</b><br>"
                f"{prov} · occ {occ_s}<br>"
                f"{rd['composition']} · {rd['palette']}<br>"
                f"fw {rd['fw']}</div></div>")

    cells = "".join(cell(r) for r in picked)
    html = (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>{BATCH_ID} sanity</title><style>"
        ":root{color-scheme:dark}*{box-sizing:border-box}"
        "body{font:12px/1.45 ui-monospace,Consolas,monospace;background:#0e0f13;color:#ccc;margin:0}"
        "header{position:sticky;top:0;background:#12141a;border-bottom:1px solid #23252e;padding:10px 18px}"
        "h1{font-size:15px;margin:0 0 4px;color:#eee}.note{color:#888;font-size:12px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;padding:14px}"
        ".cell{border:1px solid #23252e;border-radius:5px;overflow:hidden;background:#000}"
        ".cell img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block}"
        ".cell .cap{padding:4px 7px;font-size:10.5px;color:#9aa;background:#12141a;border-top:1px solid #1c1f29;word-break:break-all}"
        ".cell .cap b{color:#cdd}"
        "</style></head><body><header>"
        f"<h1>{BATCH_ID} — sanity contact sheet</h1>"
        f"<div class=note>{len(picked)} of {len(rows)} units, stratified by (root_src × depth). "
        "Visual spot-check of the bridge/gate/provenance join — no quality claims. "
        "Label via tools/viz/corpus_label.html.</div></header>"
        f"<div class=grid>{cells}</div></body></html>"
    )

    out_path = os.path.join(bdir, "sanity_contact_sheet.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(out_path, cc.ROOT)} ({len(picked)} cells)")


if __name__ == "__main__":
    main()
