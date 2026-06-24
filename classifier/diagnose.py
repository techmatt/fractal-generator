"""Step-0 diagnosis for the v1 aesthetic classifier.

Read-only inventory of the label/manifest data and the join between them.
Prints everything the CC prompt's Step 0 asks for; makes NO changes.
"""
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "labels" / "location_labels.json"
MANIFEST = ROOT / "data" / "label_crops" / "loose0_v3" / "manifest.json"
BLACK_THRESH = 0.30  # present.rs: const BLACK_THRESH: f32 = 0.30; accept iff bf < THRESH

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def main():
    # --- 1. labels ---
    labels = json.loads(LABELS.read_text())
    print("=" * 70)
    print(f"1. LABELS  {LABELS.relative_to(ROOT)}")
    print(f"   total rows: {len(labels)}")
    vals = Counter(labels.values())
    print(f"   class histogram: " + ", ".join(f"{k}:{vals[k]}" for k in sorted(vals)))
    out_of_range = {k: v for k, v in labels.items() if v not in (1, 2, 3)}
    print(f"   values outside {{1,2,3}}: {len(out_of_range)}"
          + ("" if not out_of_range else f"  -> {list(out_of_range.items())[:10]}"))
    # key shape check
    bad_keys = [k for k in labels if len(k.split("|")) != 3]
    print(f"   keys not draw_index|composition|palette: {len(bad_keys)}"
          + ("" if not bad_keys else f"  -> {bad_keys[:5]}"))

    # --- 2. manifest ---
    man = json.loads(MANIFEST.read_text())
    crops = man["crops"]
    print("=" * 70)
    print(f"2. MANIFEST  {MANIFEST.relative_to(ROOT)}")
    print(f"   top-level keys: {list(man.keys())}")
    print(f"   crops: {len(crops)}")
    print(f"   per-crop field names: {sorted(crops[0].keys())}")
    needed = ["black_fraction", "occupancy", "focus", "void_guard",
              "draw_index", "composition", "palette", "output"]
    for f in needed:
        present_in_all = all(f in c for c in crops)
        print(f"     {'OK ' if present_in_all else 'MISSING'} '{f}'"
              + ("" if present_in_all else "  <-- not in every crop"))

    # build manifest key -> crop
    man_by_key = {}
    dup_keys = []
    for c in crops:
        key = f"{c['draw_index']}|{c['composition']}|{c['palette']}"
        if key in man_by_key:
            dup_keys.append(key)
        man_by_key[key] = c
    print(f"   duplicate (draw|comp|palette) keys in manifest: {len(dup_keys)}"
          + ("" if not dup_keys else f"  -> {dup_keys[:5]}"))

    # JPG dimensions (sample a handful)
    if HAVE_PIL:
        dims = Counter()
        missing_files = 0
        for c in crops[:50]:
            p = ROOT / c["output"]
            if p.exists():
                with Image.open(p) as im:
                    dims[im.size] += 1
            else:
                missing_files += 1
        print(f"   JPG dims (first 50 crops): {dict(dims)}  missing_files={missing_files}")
    else:
        print("   (PIL unavailable — skipped dim check)")

    # --- 3. join ---
    print("=" * 70)
    print("3. JOIN  labels -> manifest")
    resolved, missing_key, missing_jpg = 0, [], []
    jpg_to_labels = {}
    rows = []  # (key, label, crop) for resolvable rows
    for key, lab in labels.items():
        c = man_by_key.get(key)
        if c is None:
            missing_key.append(key)
            continue
        jpg = ROOT / c["output"]
        if not jpg.exists():
            missing_jpg.append((key, c["output"]))
            continue
        resolved += 1
        jpg_to_labels.setdefault(c["output"], []).append(key)
        rows.append((key, lab, c))
    print(f"   label rows: {len(labels)}")
    print(f"   resolve to existing JPG: {resolved}")
    print(f"   FAIL — key not in manifest: {len(missing_key)}"
          + ("" if not missing_key else f"  -> {missing_key[:8]}"))
    print(f"   FAIL — manifest output JPG missing on disk: {len(missing_jpg)}"
          + ("" if not missing_jpg else f"  -> {missing_jpg[:8]}"))
    multi = {j: ks for j, ks in jpg_to_labels.items() if len(ks) > 1}
    print(f"   JPGs referenced by >1 label row: {len(multi)}"
          + ("" if not multi else f"  -> {list(multi.items())[:5]}"))

    # --- 4. black-filter effect (mirror present.rs: accept iff bf < 0.30) ---
    print("=" * 70)
    print(f"4. BLACK FILTER  (accept iff black_fraction < {BLACK_THRESH})")
    pre = Counter(lab for _, lab, _ in rows)
    kept = [(k, lab, c) for k, lab, c in rows if c["black_fraction"] < BLACK_THRESH]
    post = Counter(lab for _, lab, _ in kept)
    dropped = Counter(lab for k, lab, c in rows if not (c["black_fraction"] < BLACK_THRESH))
    print(f"   resolvable rows (pre-filter):  "
          + ", ".join(f"{k}:{pre[k]}" for k in sorted(pre)) + f"  total {sum(pre.values())}")
    print(f"   kept (post-filter):            "
          + ", ".join(f"{k}:{post[k]}" for k in sorted(post)) + f"  total {sum(post.values())}")
    print(f"   dropped by black gate:         "
          + ", ".join(f"{k}:{dropped[k]}" for k in sorted(dropped)) + f"  total {sum(dropped.values())}")

    # how many distinct seeds/locations (group key)
    seeds = set(c["seed_index"] for _, _, c in kept) if kept and "seed_index" in kept[0][2] else None
    draws = set(c["draw_index"] for _, _, c in kept)
    print(f"   distinct draw_index in kept set: {len(draws)}"
          + (f"   distinct seed_index: {len(seeds)}" if seeds is not None else ""))


if __name__ == "__main__":
    main()
