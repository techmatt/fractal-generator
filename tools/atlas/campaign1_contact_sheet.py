#!/usr/bin/env python
r"""campaign1_contact_sheet.py — eyeball the campaign's admissions.

Renders every distinct-q3 admission in a run's ledger to a colored 640x360 preview and packs
them into a labeled grid, ordered so morph near-dups (library recipe, single-linkage 0.974)
sit adjacent. Union breadth + dive if both are given. Reuses steered_pilot_morph wholesale
(render path, morph embed, clustering) — nothing reimplemented.

  uv run python tools/atlas/campaign1_contact_sheet.py \
      --run data/discovery/campaign1/breadth \
      [--run data/discovery/campaign1/dive] --out out/campaign1/contact_sheet.png
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "atlas", ROOT / "tools" / "studies",
          ROOT / "tools" / "corpus", ROOT / "tools" / "scoring", ROOT / "tools" / "wallpaper"):
    sys.path.insert(0, str(p))

import steered_pilot_morph as spm            # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COLS = 8
TW, TH, PAD, LAB = 320, 180, 4, 18
WORKERS = 4                                   # project rule: <=4


def load_admissions(run_dirs: list[Path]) -> list:
    rows = []
    for d in run_dirs:
        rows += spm.admitted_q3(spm.load_jsonl(d / "outcome_ledger.jsonl"))
    return rows


def cluster_order(rows, uids, fams, E):
    """Order indices so near-dups are adjacent: sort by (cluster size desc, cluster id, depth)."""
    if len(uids) == 0:
        return [], {}
    U = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    comps = spm.connected_components(len(uids), spm.STRICT_CUT, U @ U.T)
    comps = sorted(comps, key=lambda c: -len(c))
    cid_of = {}
    for cid, c in enumerate(comps):
        for i in c:
            cid_of[i] = cid
    order = sorted(range(len(uids)), key=lambda i: (cid_of[i], i))
    return order, cid_of


def build(run_dirs: list[Path], out_png: Path):
    rows = load_admissions(run_dirs)
    if not rows:
        print("no admissions yet — nothing to render.")
        return
    print(f"[contact] {len(rows)} admissions from {', '.join(d.name for d in run_dirs)}", flush=True)

    # embed (library morph_gray recipe) + cluster for adjacency ordering
    model, tf = spm.load_clip()
    tmp = out_png.parent / "morph_fields"
    uids, fams, depths, E = spm.embed_admissions(rows, tmp, model, tf)
    order, cid_of = cluster_order(rows, uids, fams, E)
    n_clusters = len(set(cid_of.values()))
    print(f"[contact] {n_clusters} distinct morph looks (CLIP>={spm.STRICT_CUT})", flush=True)

    by_id = {r["id"]: r for r in rows}
    tiledir = out_png.parent / "_tiles"
    tiles = {u: tiledir / f"{u}.jpg" for u in uids}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda u: spm.render_colored(spm.loc_of_row(by_id[u]), tiles[u]), uids))

    n = len(order)
    rows_n = (n + COLS - 1) // COLS
    W = COLS * (TW + PAD) + PAD
    H = rows_n * (TH + LAB + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (18, 18, 20))
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(order):
        r, c = divmod(k, COLS)
        x = PAD + c * (TW + PAD)
        y = PAD + r * (TH + LAB + PAD)
        tp = tiles[uids[i]]
        im = (Image.open(tp).convert("RGB").resize((TW, TH)) if tp.exists()
              else Image.new("RGB", (TW, TH), (60, 20, 20)))
        sheet.paste(im, (x, y))
        src = "dive" if by_id[uids[i]].get("mix_source") == "dive" else "brdth"
        tag = f"C{cid_of[i]} {fams[i]} d{depths[i]} {src}"
        dr.rectangle([x, y + TH, x + TW, y + TH + LAB], fill=(30, 30, 34))
        dr.text((x + 4, y + TH + 4), tag, fill=(230, 230, 120))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    print(f"[contact] wrote {out_png}  ({n} tiles, {n_clusters} looks, {W}x{H})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="run dir(s) with outcome_ledger.jsonl")
    ap.add_argument("--out", default=str(ROOT / "out" / "campaign1" / "contact_sheet.png"))
    args = ap.parse_args()
    build([Path(r).resolve() for r in args.run], Path(args.out))


if __name__ == "__main__":
    main()
