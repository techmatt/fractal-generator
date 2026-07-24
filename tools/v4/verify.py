"""Phase C: verify the v4 augmentation cache + build a coherence montage.

  integrity : expected vs actual render count, missing/empty files, on-disk size.
  balance   : every location carries the IDENTICAL 42-slot
              (palette,scale,shift,aa) multiset, so every label class does too.
  coherence : 3 locations (one per class), each as a 6x7 contact sheet
              (6 palettes x [s0.7c, s0.7s, s1.0c, s1.0s, s1.3c, s1.3s, ss4]).

  uv run python tools/v4/verify.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
# aug_cache JPGs live under ARTIFACTS_ROOT now; manifest paths stay repo-relative.
sys.path.insert(0, str(ROOT / "tools" / "corpus"))
from artifacts import resolve as resolve_artifact  # noqa: E402

CACHE_MANIFEST = ROOT / "data" / "v4" / "cache_manifest.jsonl"
ROSTER = ROOT / "data" / "v4" / "aug_roster.json"
OUT = ROOT / "out" / "v4"
EXPECTED = 3622 * 42

SCALES = [0.7, 1.0, 1.3]
# coherence-grid columns: 6 ss1 slots + the ss4 slot
COLS = [(0.7, "center", "aliased"), (0.7, "shifted", "aliased"),
        (1.0, "center", "aliased"), (1.0, "shifted", "aliased"),
        (1.3, "center", "aliased"), (1.3, "shifted", "aliased"),
        (1.0, "center", "antialiased")]


def load():
    return [json.loads(l) for l in CACHE_MANIFEST.read_text().splitlines() if l.strip()]


def integrity(rows):
    print("== integrity ==")
    print(f"  manifest rows : {len(rows)}  (expected {EXPECTED})")
    missing, empty, size = [], [], 0
    for r in rows:
        p = resolve_artifact(r["path"])
        if not p.exists():
            missing.append(r["path"])
        else:
            sz = p.stat().st_size
            size += sz
            if sz == 0:
                empty.append(r["path"])
    print(f"  on-disk files : {len(rows) - len(missing)}")
    print(f"  missing       : {len(missing)}")
    print(f"  empty (0-byte): {len(empty)}")
    print(f"  total size    : {size/1e9:.2f} GB")
    for m in missing[:10]:
        print(f"    MISSING {m}")
    for e in empty[:10]:
        print(f"    EMPTY   {e}")
    return len(missing) == 0 and len(empty) == 0


def balance(rows):
    print("== balance ==")
    per_loc = defaultdict(list)
    loc_label = {}
    for r in rows:
        per_loc[r["location_id"]].append((r["palette"], r["scale"], r["shift_id"], r["aa_level"]))
        loc_label[r["location_id"]] = r["label"]
    # canonical per-location multiset (sorted tuple of the 42 axis-combos)
    canonical = tuple(sorted(next(iter(per_loc.values()))))
    bad = sum(1 for slots in per_loc.values() if tuple(sorted(slots)) != canonical)
    print(f"  locations            : {len(per_loc)}")
    print(f"  slots/location       : {len(canonical)}  (expected 42)")
    print(f"  locations off-pattern: {bad}  (MUST be 0)")
    # per-class confirmation: every label's per-location multiset == canonical.
    by_label_loc = defaultdict(set)
    for loc, slots in per_loc.items():
        by_label_loc[loc_label[loc]].add(tuple(sorted(slots)))
    labels = sorted(by_label_loc)
    identical = all(by_label_loc[l] == {canonical} for l in labels)
    for l in labels:
        nloc = sum(1 for loc in per_loc if loc_label[loc] == l)
        print(f"  label {l}: {nloc} locs, distinct multisets={len(by_label_loc[l])} "
              f"(==canonical: {by_label_loc[l] == {canonical}})")
    print(f"  per-class multiset identical across labels {labels}: {identical}")
    return bad == 0 and identical and len(canonical) == 42


def coherence(rows):
    print("== coherence montage ==")
    palettes = [r["name"] for r in json.loads(ROSTER.read_text())]
    by_loc = defaultdict(dict)
    loc_label = {}
    for r in rows:
        by_loc[r["location_id"]][(r["palette"], r["scale"], r["shift_id"], r["aa_level"])] = r["path"]
        loc_label[r["location_id"]] = r["label"]
    rng = random.Random(0)
    chosen = {}
    for lab in (1, 2, 3):
        cands = [l for l, la in loc_label.items() if la == lab]
        if cands:
            chosen[lab] = rng.choice(cands)
    paths = []
    cw, ch, pad = 192, 108, 4
    for lab, loc in chosen.items():
        slots = by_loc[loc]
        W = len(COLS) * cw + (len(COLS) + 1) * pad
        H = len(palettes) * ch + (len(palettes) + 1) * pad
        canvas = Image.new("RGB", (W, H), (20, 20, 20))
        for ri, pal in enumerate(palettes):
            for ci, (sc, sh, aa) in enumerate(COLS):
                key = (pal, sc, sh, aa)
                if key not in slots:
                    continue
                im = Image.open(resolve_artifact(slots[key])).convert("RGB").resize((cw, ch), Image.BICUBIC)
                canvas.paste(im, (pad + ci * (cw + pad), pad + ri * (ch + pad)))
        mp = OUT / f"coherence_L{lab}_loc{loc}.png"
        canvas.save(mp)
        paths.append(mp)
        print(f"  L{lab} loc {loc} -> {mp}")
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    ok_i = integrity(rows)
    ok_b = balance(rows)
    coherence(rows)
    print(f"\nintegrity OK: {ok_i}   balance OK: {ok_b}")


if __name__ == "__main__":
    main()
