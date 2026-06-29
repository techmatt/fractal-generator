#!/usr/bin/env python
# [ACTIVE WORKFLOW] scale_2x2 corpus-building toolchain — in use, not scratch.
# Do not archive or remove without checking first.
"""scale-2x2 label analysis: faithfulness check (D vs run4 flat) then the cell read.

Measurement only. Reads:
  labels/scale_2x2_labelset.json                       (image_id -> 1/2/3)
  data/label_corpus/batches/2026-06-25_scale_2x2_labelset/images.jsonl
  data/label_corpus/batches/2026-06-24_guided_descend_rev4/images.jsonl (run4 baseline, root_src=flat)
Writes a montage of best crops per cell under data/eda/scale_2x2/.
"""
import json, os, collections, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(ROOT, "labels", "scale_2x2_labelset.json")
LS_DIR = os.path.join(ROOT, "data", "label_corpus", "batches", "2026-06-25_scale_2x2_labelset")
REV4_DIR = os.path.join(ROOT, "data", "label_corpus", "batches", "2026-06-24_guided_descend_rev4")
OUT = os.path.join(ROOT, "data", "eda", "scale_2x2")
os.makedirs(OUT, exist_ok=True)

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

labels = json.load(open(LABELS))
rows = load_jsonl(os.path.join(LS_DIR, "images.jsonl"))
by_id = {r["image_id"]: r for r in rows}

# ---- join coverage ----
n_rows = len(rows)
n_lab = len(labels)
missing_label = [r["image_id"] for r in rows if r["image_id"] not in labels]
orphan_label = [k for k in labels if k not in by_id]

NAME = {1: "bad", 2: "okay", 3: "good"}

def dist(score_iter):
    c = collections.Counter(score_iter)
    return c

# ---- per-cell tally (labelset) ----
cells = ["A", "B", "C", "D"]
cell_scores = {c: [] for c in cells}
for img_id, sc in labels.items():
    r = by_id.get(img_id)
    if r is None:
        continue
    cell = r["provenance"]["cell"]
    cell_scores[cell].append(sc)

# sanity: image_id prefix vs provenance.cell
prefix_mismatch = [img for img in labels if by_id.get(img) and img.split("_")[0] != by_id[img]["provenance"]["cell"]]

# ---- run4 flat baseline ----
rev4 = load_jsonl(os.path.join(REV4_DIR, "images.jsonl"))
flat_scores, eightk_scores = [], []
for r in rev4:
    sc = r["label"]["score"]
    if sc is None:
        continue
    if r["provenance"].get("root_src") == "flat":
        flat_scores.append(sc)
    elif r["provenance"].get("root_src") == "8k":
        eightk_scores.append(sc)

def pct(n, d):
    return f"{100*n/d:4.1f}%" if d else "  n/a"

def fmt_row(label, scores):
    c = dist(scores)
    n = len(scores)
    b, o, g = c.get(1,0), c.get(2,0), c.get(3,0)
    nb = o + g
    return (f"{label:18s} N={n:4d}  bad {b:4d} ({pct(b,n)})  okay {o:3d} ({pct(o,n)})  "
            f"good {g:3d} ({pct(g,n)})  | not-bad {nb:3d} ({pct(nb,n)})")

print("="*100)
print("PHASE 0 — discovery & join")
print("="*100)
print(f"labelset rows: {n_rows}   labels in file: {n_lab}")
print(f"rows missing a label: {len(missing_label)}   orphan labels (no row): {len(orphan_label)}")
print(f"image_id-prefix vs provenance.cell mismatches: {len(prefix_mismatch)}")
print(f"per-cell labeled counts: " + ", ".join(f"{c}={len(cell_scores[c])}" for c in cells))
print()
print("run4 baseline (rev4 batch): flat N={}  8k N={}".format(len(flat_scores), len(eightk_scores)))
print()
print("="*100)
print("PHASE 1 — faithfulness: cell D (flat, narrow)  vs  run4 flat")
print("="*100)
print(fmt_row("D (flat narrow)", cell_scores["D"]))
print(fmt_row("run4 flat", flat_scores))
print(fmt_row("run4 8k", eightk_scores))
print(fmt_row("run4 all", flat_scores + eightk_scores))
print()
print("="*100)
print("PHASE 2 — per-cell table (A=8k/wide  B=8k/narrow  C=flat/wide  D=flat/narrow)")
print("="*100)
for c in cells:
    print(fmt_row(f"cell {c}", cell_scores[c]))
print()
print("Comparisons:")
print("  B vs D (field vs flat, narrow): ")
print("   ", fmt_row("B", cell_scores["B"]))
print("   ", fmt_row("D", cell_scores["D"]))
print("  A vs B (8k wide vs narrow): ")
print("   ", fmt_row("A", cell_scores["A"]))
print("   ", fmt_row("B", cell_scores["B"]))
print("  C vs D (flat wide vs narrow): ")
print("   ", fmt_row("C", cell_scores["C"]))
print("   ", fmt_row("D", cell_scores["D"]))

# ---- region coverage: where do non-bad roots land ----
print()
print("="*100)
print("Region coverage of non-bad (okay/good) roots — (cx, cy) of label>=2")
print("="*100)
for c in cells:
    pts = []
    for img_id, sc in labels.items():
        r = by_id.get(img_id)
        if r and r["provenance"]["cell"] == c and sc >= 2:
            pts.append((float(r["render"]["cx"]), float(r["render"]["cy"]), sc))
    if pts:
        cxs = [p[0] for p in pts]; cys = [p[1] for p in pts]
        print(f"cell {c}: {len(pts)} non-bad   cx[{min(cxs):+.4f},{max(cxs):+.4f}] cy[{min(cys):+.4f},{max(cys):+.4f}]")
    else:
        print(f"cell {c}: 0 non-bad")

# ---- montage: best crops per cell (goods then top okays) ----
print()
print("="*100)
print("Montage assembly")
print("="*100)
montage_rows = {}
for c in cells:
    items = []
    for img_id, sc in labels.items():
        r = by_id.get(img_id)
        if r and r["provenance"]["cell"] == c:
            items.append((sc, img_id))
    # goods first (3), then okays (2)
    items.sort(key=lambda x: -x[0])
    best = [it for it in items if it[0] >= 2]
    montage_rows[c] = best
    print(f"cell {c}: {sum(1 for s,_ in best if s==3)} good + {sum(1 for s,_ in best if s==2)} okay = {len(best)} crops")

# copy crops + write an HTML montage
crops_src = os.path.join(LS_DIR, "crops")
html = ["<html><head><meta charset=utf-8><style>",
        "body{background:#111;color:#ddd;font-family:monospace}",
        ".cell{margin:18px 0}.cell h2{color:#fff}",
        "img{height:150px;margin:2px;border:1px solid #333;vertical-align:top}",
        ".g{border-color:#3c3}.o{border-color:#aa3}",
        ".tag{display:inline-block;width:152px;font-size:10px;text-align:center;vertical-align:top}",
        "</style></head><body>"]
html.append("<h1>scale-2x2 best crops per cell (goods then okays)</h1>")
for c in cells:
    html.append(f'<div class=cell><h2>cell {c} (n={len(montage_rows[c])})</h2>')
    for sc, img_id in montage_rows[c]:
        cls = "g" if sc == 3 else "o"
        src = os.path.relpath(os.path.join(crops_src, img_id + ".jpg"), OUT).replace("\\","/")
        html.append(f'<span class=tag><img class={cls} src="{src}"><br>{img_id[:34]}<br>[{NAME[sc]}]</span>')
    html.append("</div>")
html.append("</body></html>")
with open(os.path.join(OUT, "montage.html"), "w", encoding="utf-8") as f:
    f.write("\n".join(html))
print(f"\nwrote {os.path.join(OUT, 'montage.html')}")
