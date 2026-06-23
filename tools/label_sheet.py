#!/usr/bin/env python3
"""Light visual-first sanity viewer for a present label-crop run.

Reads a present `manifest.json` and writes a self-contained `sanity_sheet.html`
next to it: a sampled grid of crop thumbnails, each captioned with the exact
manifest fields the labeling harness will key on (draw_index, seed_index,
composition, palette, fw, interior_frac, black_fraction). Images are referenced
by basename so the HTML lives in the run dir alongside the PNGs.

Usage:
    python tools/label_sheet.py <manifest.json> [--sample N] [--seed S]
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--sample", type=int, default=120,
                    help="max crops to show (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    crops = m["crops"]
    run_dir = os.path.dirname(os.path.abspath(args.manifest))

    rng = random.Random(args.seed)
    shown = crops if args.sample <= 0 or len(crops) <= args.sample \
        else rng.sample(crops, args.sample)
    # stable display order: by draw_index, composition, palette
    shown = sorted(shown, key=lambda c: (c["draw_index"], c["composition"], c["palette"]))

    cells = []
    for c in shown:
        img = os.path.basename(c["output"])
        cap = (f"draw {c['draw_index']} · seed {c['seed_index']} · {c['composition']}<br>"
               f"{c['palette']}<br>"
               f"fw {c['fw']:.3e} · int {c['interior_frac']*100:.0f}% · "
               f"blk {c['black_fraction']*100:.0f}%")
        cells.append(
            f'<figure><img src="{img}" loading="lazy"><figcaption>{cap}</figcaption></figure>')

    head = (f"<b>{os.path.basename(run_dir)}</b> &mdash; "
            f"{m.get('total_seeds','?')} seeds · {m.get('accepted','?')} crops accepted · "
            f"{m.get('rejected_black','?')} comp-crops rejected (black) · "
            f"{len(crops)} label units · showing {len(shown)}")

    html = f"""<!doctype html><meta charset=utf-8>
<title>label crops &mdash; {os.path.basename(run_dir)}</title>
<style>
 body{{background:#111;color:#ccc;font:13px/1.4 system-ui,sans-serif;margin:16px}}
 header{{margin-bottom:12px;font-size:14px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
 figure{{margin:0;background:#1b1b1b;border:1px solid #2a2a2a;border-radius:6px;padding:6px}}
 img{{width:100%;display:block;border-radius:3px;background:#000}}
 figcaption{{margin-top:5px;font-size:11px;color:#9a9a9a}}
</style>
<header>{head}</header>
<div class=grid>
{chr(10).join(cells)}
</div>
"""
    out = os.path.join(run_dir, "sanity_sheet.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}  ({len(shown)} of {len(crops)} crops)")


if __name__ == "__main__":
    main()
